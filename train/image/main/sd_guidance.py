"""DMAD critic for SDXL: a two-head GAN critic in place of DMD2's fake score (ImageNet version: main/edm/edm_guidance.py).

Two binary heads read the same bottleneck features:
    real head    d_real    : real latents (T) vs generated latents (G)
    teacher head d_teacher : teacher-sample latents (Q) vs generated latents (G)
Critic loss:     BCE(real head; T=1, G=0) + BCE(teacher head; Q=1, G=0)
Generator loss:  -d_teacher(G) (weighted per noise band with --teacher_gap_route) and -d_real(G);
                 the trainer weights them by --teacher_loss_weight and --real_loss_weight.

Backbone: the encoder + mid block of a copy of the SDXL UNet (the decoder is removed), optionally frozen
(--critic_freeze_backbone) and/or with frozen-gain spectral norm (--critic_spectral_norm).
Critic inputs are noised with the DDPM forward process at a timestep drawn uniformly from [0, critic_max_timestep);
in the critic update one timestep per sample index is shared by the T, G and Q samples.
Gap routing (--teacher_gap_route): the teacher term is weighted per noise band (timestep // 100) by
sigmoid((median_b gap_b - gap_band) / tau), with gap_b = E[d_real(T)] - E[d_real(Q)] tracked online per band.
"""
import copy
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from diffusers import DDIMScheduler, UNet2DConditionModel

from main.sd_unet_forward import classify_forward
from main.utils import DummyNetwork, NoOpContext

NUM_GAP_BANDS = 10


def make_binary_head():
    """Bottleneck features [N, 1280, 32, 32] -> one logit per sample (DMD2's SDXL classifier head)."""
    return nn.Sequential(
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),  # 32x32 -> 16x16
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),  # 16x16 -> 8x8
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1),  # 8x8 -> 4x4
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0),  # 4x4 -> 1x1
        nn.GroupNorm(num_groups=32, num_channels=1280),
        nn.SiLU(),
        nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0),
    )


class _ScaledSN(nn.Module):
    """weight -> gain * W / sigma_max(W), sigma_max from one power iteration per forward. The gain is frozen at the
    value the first forward computes, so the layer is unchanged at initialization."""

    def __init__(self, weight):
        super().__init__()
        w2 = weight.detach().reshape(weight.shape[0], -1)
        u = torch.randn(w2.shape[0], device=weight.device)
        v = torch.randn(w2.shape[1], device=weight.device)
        for _ in range(200):
            v = F.normalize(w2.t() @ u, dim=0)
            u = F.normalize(w2 @ v, dim=0)
        self.register_buffer("u", u.clone())
        self.register_buffer("v", v.clone())
        v1 = F.normalize(w2.t() @ u, dim=0)
        u1 = F.normalize(w2 @ v1, dim=0)
        self.register_buffer("gain", torch.sum(u1 * (w2 @ v1)).clone())

    def forward(self, w):
        w2 = w.reshape(w.shape[0], -1)
        with torch.no_grad():
            self.v.copy_(F.normalize(w2.t() @ self.u, dim=0))
            self.u.copy_(F.normalize(w2 @ self.v, dim=0))
        # clone: one backward graph can span several forwards, and the next forward updates u, v in place
        u, v = self.u.clone(), self.v.clone()
        sigma = torch.sum(u * (w2 @ v)).clamp(min=1e-12)
        return w * (self.gain / sigma)


def apply_scaled_spectral_norm(module):
    """Frozen-gain spectral norm on every trainable, non-zero Conv2d in `module`. Returns the number of wrapped convs."""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
            if m.weight.detach().abs().max() < 1e-8:  # zero-initialized conv
                continue
            parametrize.register_parametrization(m, "weight", _ScaledSN(m.weight))
            n += 1
    return n


class SDGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args

        self.fake_unet = UNet2DConditionModel.from_pretrained(args.model_id, subfolder="unet").float()
        self.fake_unet.requires_grad_(True)

        # FSDP needs at least one module with dense parameters (diffusers models are lazily initialized)
        self.dummy_network = DummyNetwork()
        self.dummy_network.requires_grad_(True)  # keeps the flat-parameter group uniformly trainable

        self.scheduler = DDIMScheduler.from_pretrained(args.model_id, subfolder="scheduler")
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod)
        self.num_train_timesteps = args.num_train_timesteps
        self.critic_max_timestep = args.critic_max_timestep

        self.fake_unet.forward = types.MethodType(classify_forward, self.fake_unet)
        if accelerator.is_local_main_process:
            print("Randomly initialized heads differ across ranks; the trainer syncs them from rank 0 before FSDP.")
        self.head_real = make_binary_head()
        self.head_real.requires_grad_(True)

        self.teacher_gap_route = args.teacher_gap_route
        self.gap_tau = args.gap_tau
        self.gap_ema_beta = 0.99
        self.register_buffer("_gap_ema", torch.zeros(NUM_GAP_BANDS))
        self.register_buffer("_gap_ready", torch.zeros(NUM_GAP_BANDS))

        self.head_teacher = copy.deepcopy(self.head_real)
        for m in self.head_teacher.modules():  # re-initialize so the two heads start different
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.head_teacher.requires_grad_(True)

        # the critic only runs the encoder and mid block (classify_mode); drop the decoder
        self.fake_unet.up_blocks = nn.ModuleList()
        self.fake_unet.conv_norm_out = None
        self.fake_unet.conv_out = None

        if args.critic_freeze_backbone:  # needs an FSDP config with use_orig_params=true
            self.fake_unet.requires_grad_(False)
        if args.critic_spectral_norm:
            n = 0
            for module in (self.fake_unet.down_blocks, self.fake_unet.mid_block, self.head_real, self.head_teacher):
                n += apply_scaled_spectral_norm(module)
            if accelerator.is_local_main_process:
                print(f"critic spectral norm: {n} convs")

        self.gradient_checkpointing = args.gradient_checkpointing
        if self.gradient_checkpointing:
            self.fake_unet.enable_gradient_checkpointing()
        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.use_fp16 else NoOpContext()

    # ------------------------ critic forward ------------------------

    def sample_timesteps(self, batch_size, device):
        return torch.randint(0, self.critic_max_timestep, [batch_size], device=device, dtype=torch.long)

    def critic_logits(self, image, timesteps, text_embedding, unet_added_conditions):
        """(d_real, d_teacher), [N] each, on image noised to `timesteps`."""
        image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
        with self.network_context_manager:
            rep = self.fake_unet.forward(
                image, timesteps, text_embedding, added_cond_kwargs=unet_added_conditions, classify_mode=True
            )
        rep = rep[-1].float()  # bottleneck
        d_real = self.head_real(rep).squeeze(dim=[2, 3]).reshape(-1)
        d_teacher = self.head_teacher(rep).squeeze(dim=[2, 3]).reshape(-1)
        return d_real, d_teacher

    # ------------------------ gap routing ------------------------

    @torch.no_grad()
    def _update_gap_ema(self, dR_T, dR_Q, t_T, t_Q):
        """Per-band gap E[d_real(T)] - E[d_real(Q)], all-reduced over ranks, folded into the online EMA.
        Returns the current per-band gap (NaN for bands without samples on both sides)."""
        dev = dR_T.device
        bt = (t_T.reshape(-1) // 100).clamp(0, NUM_GAP_BANDS - 1)
        bq = (t_Q.reshape(-1) // 100).clamp(0, NUM_GAP_BANDS - 1)
        sT = torch.zeros(NUM_GAP_BANDS, device=dev); cT = torch.zeros(NUM_GAP_BANDS, device=dev)
        sQ = torch.zeros(NUM_GAP_BANDS, device=dev); cQ = torch.zeros(NUM_GAP_BANDS, device=dev)
        sT.scatter_add_(0, bt, dR_T.float()); cT.scatter_add_(0, bt, torch.ones_like(dR_T, dtype=torch.float))
        sQ.scatter_add_(0, bq, dR_Q.float()); cQ.scatter_add_(0, bq, torch.ones_like(dR_Q, dtype=torch.float))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (sT, cT, sQ, cQ):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        valid = (cT > 0) & (cQ > 0)
        gap_now = sT / cT.clamp(min=1) - sQ / cQ.clamp(min=1)
        beta = self.gap_ema_beta
        fresh = valid & (self._gap_ready > 0)
        upd = torch.where(fresh, beta * self._gap_ema.to(dev) + (1 - beta) * gap_now, gap_now)
        self._gap_ema.copy_(torch.where(valid, upd, self._gap_ema.to(dev)))
        self._gap_ready.copy_(torch.where(valid, torch.ones_like(self._gap_ready.to(dev)), self._gap_ready.to(dev)))
        return torch.where(valid, gap_now, torch.full_like(gap_now, float("nan")))

    def _teacher_gap_weight(self, t):
        """Per-sample teacher weight sigmoid((median_b gap_b - gap_band) / tau); None (plain mean) until half of
        the reachable bands have been measured."""
        if not self.teacher_gap_route:
            return None
        n_reach = min(NUM_GAP_BANDS, max(1, int(self.critic_max_timestep) // 100 + 1))
        if float(self._gap_ready.sum()) < max(2, (n_reach + 1) // 2):
            return None
        center = float(self._gap_ema[self._gap_ready > 0].median())
        band = (t.reshape(-1) // 100).clamp(0, NUM_GAP_BANDS - 1)
        return torch.sigmoid((center - self._gap_ema[band]) / self.gap_tau)

    # ------------------------ losses ------------------------

    def compute_generator_loss(self, fake_image, text_embedding, unet_added_conditions):
        t = self.sample_timesteps(fake_image.shape[0], fake_image.device)
        d_real, d_teacher = self.critic_logits(fake_image, t, text_embedding, unet_added_conditions)
        w = self._teacher_gap_weight(t)
        if w is None:
            teacher_loss = (-d_teacher).mean()
        else:
            teacher_loss = (-d_teacher * w).sum() / w.sum().clamp(min=1.0) + (-d_teacher).sum() * 0.0
        real_loss = (-d_real).mean()
        return {"gen_teacher_loss": teacher_loss, "gen_real_loss": real_loss}

    def compute_critic_loss(self, fake_image, text_embedding, unet_added_conditions, real_train_dict, teacher_train_dict):
        real_image = real_train_dict["images"]
        teacher_image = teacher_train_dict["images"]
        n = max(real_image.shape[0], fake_image.shape[0], teacher_image.shape[0])
        t = self.sample_timesteps(n, fake_image.device)  # shared by the T / G / Q samples of each index
        t_T, t_G, t_Q = t[:real_image.shape[0]], t[:fake_image.shape[0]], t[:teacher_image.shape[0]]

        dR_T, _ = self.critic_logits(
            real_image.detach(), t_T, real_train_dict["text_embedding"], real_train_dict["unet_added_conditions"]
        )
        dR_G, dQ_G = self.critic_logits(fake_image.detach(), t_G, text_embedding, unet_added_conditions)
        dR_Q, dQ_Q = self.critic_logits(
            teacher_image.detach(), t_Q, teacher_train_dict["text_embedding"], teacher_train_dict["unet_added_conditions"]
        )
        loss_real = F.softplus(-dR_T).mean() + F.softplus(dR_G).mean()
        loss_teacher = F.softplus(-dQ_Q).mean() + F.softplus(dQ_G).mean()

        log_dict = {
            "critic_acc_real": (dR_T > 0).float().mean().detach(),
            "critic_acc_teacher": (dQ_Q > 0).float().mean().detach(),
            "critic_acc_fake": 0.5 * ((dR_G < 0).float().mean() + (dQ_G < 0).float().mean()).detach(),
        }
        gap = self._update_gap_ema(dR_T, dR_Q, t_T, t_Q)
        for b in range(NUM_GAP_BANDS):
            if bool(self._gap_ready[b] > 0):
                log_dict[f"gap_ema_b{b}"] = float(self._gap_ema[b])
            if not torch.isnan(gap[b]):
                log_dict[f"gap_b{b}"] = float(gap[b])
        return {"critic_loss": loss_real + loss_teacher}, log_dict

    def forward(self, generator_turn=False, guidance_turn=False, generator_data_dict=None, guidance_data_dict=None):
        if generator_turn:
            loss_dict = self.compute_generator_loss(
                generator_data_dict["image"], generator_data_dict["text_embedding"],
                generator_data_dict["unet_added_conditions"],
            )
            return loss_dict, {}
        elif guidance_turn:
            return self.compute_critic_loss(
                guidance_data_dict["image"], guidance_data_dict["text_embedding"],
                guidance_data_dict["unet_added_conditions"], guidance_data_dict["real_train_dict"],
                guidance_data_dict["teacher_train_dict"],
            )
        raise NotImplementedError
