"""DMAD critic for ImageNet-64 (EDM): a two-head GAN critic in place of DMD2's fake score.

Two binary heads read the same features:
    real head    d_real    : real images (T) vs generated images (G)
    teacher head d_teacher : teacher samples (Q) vs generated images (G)
Critic loss:     BCE(real head; T=1, G=0) + BCE(teacher head; Q=1, G=0)   [+ optional R1 on T and Q]
Generator loss:  -d_teacher(G) (weighted per noise band with --teacher_gap_route) and -d_real(G);
                 the trainer weights them by --teacher_loss_weight and --real_loss_weight.

Critic backbones:
  * default: the encoder of a copy of the teacher UNet (its decoder is removed), features at the bottleneck;
    optionally frozen (--critic_freeze_backbone) and/or with frozen-gain spectral norm
    (--critic_spectral_norm).
  * --pretrained_critic: frozen ImageNet-pretrained feature networks with multi-scale heads (pg_critic.py).
Critic inputs are noised at one shared noise level per sample (a uniform index into the Karras grid, capped by
--critic_max_timestep), or enter clean with --critic_clean.
Gap routing (--teacher_gap_route): the teacher term is weighted per noise band by
sigmoid((median_b gap_b - gap_band) / tau), with gap_b = E[d_real(T)] - E[d_real(Q)] tracked online per decile
band of the noise index.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm

NUM_GAP_BANDS = 10


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # from https://github.com/crowsonkb/k-diffusion
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


def make_binary_head():
    """Bottleneck features [N, 768, 8, 8] -> one logit per sample."""
    return nn.Sequential(
        nn.Conv2d(kernel_size=4, in_channels=768, out_channels=768, stride=2, padding=1),  # 8x8 -> 4x4
        nn.GroupNorm(num_groups=32, num_channels=768),
        nn.SiLU(),
        nn.Conv2d(kernel_size=4, in_channels=768, out_channels=768, stride=4, padding=0),  # 4x4 -> 1x1
        nn.GroupNorm(num_groups=32, num_channels=768),
        nn.SiLU(),
        nn.Conv2d(kernel_size=1, in_channels=768, out_channels=1, stride=1, padding=0),
    )


class _FrozenGain(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.register_buffer("gain", gain)

    def forward(self, w):
        return w * self.gain


def apply_scaled_spectral_norm(module):
    """Frozen-gain spectral norm on one conv: weight = gain * W / sigma_max(W), with gain fixed at the initial
    sigma_max(W). The layer is unchanged at initialization and its spectral norm cannot grow beyond that value.
    Returns 1 if the module was wrapped (4-d conv weight with a non-zero norm), else 0."""
    w = getattr(module, "weight", None)
    if not isinstance(w, nn.Parameter) or w.dim() != 4:
        return 0
    if torch.linalg.matrix_norm(w.detach().flatten(1).float(), ord=2) < 1e-8:  # zero-initialized conv
        return 0
    w_orig = w.detach().clone()
    spectral_norm(module, name="weight", n_power_iterations=1)
    with torch.no_grad():  # converge the power iteration at init
        for _ in range(30):
            _ = module.weight
        wn = module.weight.detach()
        gain = (w_orig.flatten() @ wn.flatten()) / (wn.flatten() @ wn.flatten())
    parametrize.register_parametrization(module, "weight", _FrozenGain(gain.to(w.dtype)))
    return 1


class EDMGuidance(nn.Module):
    def __init__(self, args, teacher_unet):
        super().__init__()
        self.args = args
        self.pretrained_critic = args.pretrained_critic
        if self.pretrained_critic:
            from main.edm.pg_critic import PGTwoHeadCritic

            self.pg_critic = PGTwoHeadCritic(backbones=[b for b in self.pretrained_critic.split(",") if b], c_dim=args.label_dim)
        else:
            self.fake_unet = copy.deepcopy(teacher_unet)
            self.fake_unet.requires_grad_(not args.critic_freeze_backbone)
            self.head_real = make_binary_head()
            self.head_teacher = make_binary_head()
        self.critic_clean = args.critic_clean

        self.r1_gamma = args.r1_gamma
        self.r1_batch_size = args.r1_batch_size
        self.max_timestep = args.critic_max_timestep
        self.teacher_gap_route = args.teacher_gap_route
        self.gap_tau = args.gap_tau
        self.gap_ema_beta = 0.99
        self.register_buffer("_gap_ema", torch.zeros(NUM_GAP_BANDS))
        self.register_buffer("_gap_ready", torch.zeros(NUM_GAP_BANDS))

        self.num_train_timesteps = args.num_train_timesteps
        karras_sigmas = torch.flip(  # small sigma first
            get_sigmas_karras(self.num_train_timesteps, sigma_max=args.sigma_max, sigma_min=args.sigma_min, rho=args.rho),
            dims=[0],
        )
        self.register_buffer("karras_sigmas", karras_sigmas)
        self.min_step = int(args.min_step_percent * self.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.num_train_timesteps)

    def apply_critic_spectral_norm(self):
        """Frozen-gain spectral norm on every conv of the critic encoder and of both heads."""
        for module in list(self.fake_unet.model.enc.modules()):
            apply_scaled_spectral_norm(module)
        for head in (self.head_real, self.head_teacher):
            for module in list(head.modules()):
                apply_scaled_spectral_norm(module)

    def remove_critic_decoder(self):
        """The critic only runs the UNet encoder; drop the decoder and output layers."""
        if self.pretrained_critic:
            return
        unet = self.fake_unet.model
        for attr in ("dec", "out_norm", "out_conv"):
            setattr(unet, attr, None)

    # ------------------------ noise levels ------------------------

    def sample_timesteps(self, batch_size, device):
        """Karras-grid indices, uniform in [min_step, min(max_step, critic_max_timestep)]."""
        min_step = max(0, min(self.min_step, self.num_train_timesteps - 1))
        max_step = max(min_step, min(self.max_step, self.num_train_timesteps - 1))
        if self.max_timestep is not None:
            max_step = max(min_step, min(max_step, self.max_timestep))
        return torch.randint(min_step, max_step + 1, [batch_size, 1, 1, 1], device=device, dtype=torch.long)

    # ------------------------ critic forward ------------------------

    def critic_logits(self, noisy_image, timestep_sigma, label):
        """(d_real, d_teacher): [N] each, or [N, K] per-scale logits with the pretrained critic."""
        if self.pretrained_critic:
            return self.pg_critic(noisy_image, label)  # no noise-level conditioning
        rep = self.fake_unet(noisy_image, timestep_sigma, label, return_bottleneck=True).float()
        return self.head_real(rep).reshape(rep.shape[0]), self.head_teacher(rep).reshape(rep.shape[0])

    # ------------------------ gap routing ------------------------

    @torch.no_grad()
    def _update_gap_ema(self, dR_T, dR_Q, sig_t, sig_q):
        """Per-band EMA of gap = E[d_real(T)] - E[d_real(Q)] (bands = deciles of the Karras index), all-reduced."""
        dev = dR_T.device
        bt = (torch.searchsorted(self.karras_sigmas, sig_t.reshape(-1).contiguous()) // 100).clamp(0, 9)
        bq = (torch.searchsorted(self.karras_sigmas, sig_q.reshape(-1).contiguous()) // 100).clamp(0, 9)
        sT, cT = torch.zeros(NUM_GAP_BANDS, device=dev), torch.zeros(NUM_GAP_BANDS, device=dev)
        sQ, cQ = torch.zeros(NUM_GAP_BANDS, device=dev), torch.zeros(NUM_GAP_BANDS, device=dev)
        sT.scatter_add_(0, bt, dR_T.float())
        cT.scatter_add_(0, bt, torch.ones_like(dR_T, dtype=torch.float))
        sQ.scatter_add_(0, bq, dR_Q.float())
        cQ.scatter_add_(0, bq, torch.ones_like(dR_Q, dtype=torch.float))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            for t in (sT, cT, sQ, cQ):
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
        valid = (cT > 0) & (cQ > 0)
        gap_now = sT / cT.clamp(min=1) - sQ / cQ.clamp(min=1)
        beta = self.gap_ema_beta
        fresh = valid & (self._gap_ready > 0)
        upd = torch.where(fresh, beta * self._gap_ema + (1 - beta) * gap_now, gap_now)
        self._gap_ema = torch.where(valid, upd, self._gap_ema)
        self._gap_ready = torch.where(valid, torch.ones_like(self._gap_ready), self._gap_ready)

    def _teacher_gap_weight(self, idx_flat):
        """Per-sample teacher weight; None until half of the reachable noise bands have been measured."""
        if not self.teacher_gap_route:
            return None
        tmax = self.max_timestep
        n_reach = 10 if tmax is None else min(10, max(1, int(tmax) // 100 + 1))
        if float(self._gap_ready.sum()) < max(2, (n_reach + 1) // 2):
            return None
        center = float(self._gap_ema[self._gap_ready > 0].median())
        band = (idx_flat // 100).clamp(0, 9)
        return torch.sigmoid((center - self._gap_ema[band]) / self.gap_tau)

    # ------------------------ losses ------------------------

    def compute_generator_loss(self, image, labels):
        batch_size = image.shape[0]
        if self.critic_clean:
            # clean inputs; the conditioning sigma is sigma_min since the UNet's noise conditioning is log-based
            timesteps = torch.zeros(batch_size, device=image.device, dtype=torch.long)
            timestep_sigma = self.karras_sigmas[0].expand(batch_size)
            noisy = image
        else:
            timesteps = self.sample_timesteps(batch_size, image.device)
            timestep_sigma = self.karras_sigmas[timesteps]
            noisy = image + timestep_sigma.reshape(-1, 1, 1, 1) * torch.randn_like(image)
        d_real_G, d_teacher_G = self.critic_logits(noisy, timestep_sigma, labels)
        w_gap = self._teacher_gap_weight(timesteps.reshape(-1))
        x = -d_teacher_G
        if w_gap is None:
            teacher_loss = x.mean()
        else:  # weighted mean; the zero term keeps the head in the autograd graph
            teacher_loss = (x * w_gap).sum() / w_gap.sum().clamp(min=1.0) + x.sum() * 0.0
        return {"gen_teacher_loss": teacher_loss, "gen_real_loss": (-d_real_G).mean()}

    def _r1(self, xT, xQ, sig_t, sig_q, real_labels, teacher_labels):
        """R1 on a subset of the batch: 0.5 * (E||grad_x d_real(T)||^2 + E||grad_x d_teacher(Q)||^2)."""
        k = max(1, min(xT.shape[0], self.r1_batch_size))

        def _reduce(d):
            return d if d.dim() == 1 else d.mean(dim=1)

        xrT = xT[:k].detach().requires_grad_(True)
        dRt, _ = self.critic_logits(xrT, sig_t[:k], real_labels[:k])
        grad_T = torch.autograd.grad(_reduce(dRt).sum(), xrT, create_graph=True)[0]
        total = grad_T.pow(2).flatten(1).sum(1).mean()
        xrQ = xQ[:k].detach().requires_grad_(True)
        _, dQq = self.critic_logits(xrQ, sig_q[:k], teacher_labels[:k])
        grad_Q = torch.autograd.grad(_reduce(dQq).sum(), xrQ, create_graph=True)[0]
        total = total + grad_Q.pow(2).flatten(1).sum(1).mean()
        return 0.5 * total

    def compute_critic_loss(self, fake_image, real_image, fake_labels, real_labels, teacher_image, teacher_labels):
        B = fake_image.shape[0]
        shared = self.karras_sigmas[self.sample_timesteps(B, fake_image.device)]
        sig_t = sig_g = sig_q = shared
        real_image, fake_image, q_image = real_image.detach(), fake_image.detach(), teacher_image.detach()
        if self.critic_clean:
            sig_t = sig_g = sig_q = self.karras_sigmas[0].expand(B)
            xT, xG, xQ = real_image, fake_image, q_image
        else:
            xT = real_image + sig_t.reshape(-1, 1, 1, 1) * torch.randn_like(real_image)
            xG = fake_image + sig_g.reshape(-1, 1, 1, 1) * torch.randn_like(fake_image)
            xQ = q_image + sig_q.reshape(-1, 1, 1, 1) * torch.randn_like(q_image)

        if self.pretrained_critic:
            # one forward per class, so the critic's BatchNorm sees single-class batches as in the generator step
            parts = [self.critic_logits(x, s, l) for x, s, l in ((xT, sig_t, real_labels), (xG, sig_g, fake_labels), (xQ, sig_q, teacher_labels))]
            d_real, d_teacher = torch.cat([p[0] for p in parts], dim=0), torch.cat([p[1] for p in parts], dim=0)
        else:
            d_real, d_teacher = self.critic_logits(
                torch.cat([xT, xG, xQ], dim=0),
                torch.cat([sig_t, sig_g, sig_q], dim=0),
                torch.cat([real_labels, fake_labels, teacher_labels], dim=0),
            )
        dR_T, dR_G = d_real[:B], d_real[B:2 * B]
        dQ_G, dQ_Q = d_teacher[B:2 * B], d_teacher[2 * B:]
        if self.teacher_gap_route:
            self._update_gap_ema(dR_T, d_real[2 * B:], sig_t, sig_q)
        loss = (F.softplus(-dR_T).mean() + F.softplus(dR_G).mean()) + (F.softplus(-dQ_Q).mean() + F.softplus(dQ_G).mean())

        with torch.no_grad():
            log_dict = {
                "critic_acc_t": (dR_T > 0).float().mean(),
                "critic_acc_q": (dQ_Q > 0).float().mean(),
                "critic_acc_g": 0.5 * ((dR_G < 0).float().mean() + (dQ_G < 0).float().mean()),
            }
        if self.r1_gamma > 0:
            r1 = self._r1(xT, xQ, sig_t, sig_q, real_labels, teacher_labels)
            loss = loss + 0.5 * self.r1_gamma * r1
            log_dict["r1"] = r1.detach()
        return {"critic_loss": loss}, log_dict

    def forward(self, generator_turn=False, guidance_turn=False, generator_data_dict=None, guidance_data_dict=None):
        if generator_turn:
            return self.compute_generator_loss(generator_data_dict["image"], generator_data_dict["label"]), {}
        if guidance_turn:
            d = guidance_data_dict
            r = d["real_train_dict"]
            return self.compute_critic_loss(
                fake_image=d["image"], real_image=r["real_image"], fake_labels=d["label"], real_labels=r["real_label"],
                teacher_image=r["second_real_image"], teacher_labels=r["second_real_label"],
            )
        raise NotImplementedError
