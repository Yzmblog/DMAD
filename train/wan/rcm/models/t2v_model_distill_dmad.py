"""DMAD distillation for Wan2.1 T2V: rCM's distillation model with the DMD term replaced by a two-head GAN critic.

* Critic: the fake-score net (a copy of the teacher) followed by two per-token MLP heads on the token output of its
  last block, mean-pooled over tokens:
      real head    d_real    : real videos (T) vs generated videos (G)
      teacher head d_teacher : teacher samples (Q) vs generated videos (G)
  Critic loss: softplus BCE on both heads (T/Q = 1, G = 0). G, Q and T of one sample are noised at the same
  (TrigFlow) time drawn from p_D, with independent noise.
* Generator loss: -d_teacher(G) - d_real(G) (no sCM loss). With `dmad_teacher_gap_route` the teacher term is weighted
  per noise band (CDF deciles of p_D) by sigmoid((median_b gap_b - gap_band) / tau), gap_b = E[d_real(T)] - E[d_real(Q)]
  tracked online, and renormalized by the mean band weight.
* Data: the batch latents are teacher samples (Q); caption-paired real latents come in data_batch["real_latents"].
* Frozen-gain spectral norm (weight = gain * W / sigma_max(W), gain fixed at the initial sigma_max) on the critic's
  block and head Linears and on the Linears of the first half of the generator's blocks (and of its EMA copy).
  Parametrizations are registered on full weights before FSDP sharding.
* Context parallel: pooled logits are averaged over the context-parallel group.
"""

import math

import attrs
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from imaginaire.utils import log
from rcm.models.t2v_model_distill_rcm import T2VDistillConfig_rCM, T2VDistillModel_rCM
from rcm.utils.timestep_utils import trig_to_rf_time

NUM_GAP_BANDS = 10


class CriticHeads(nn.Module):
    """Two per-token binary heads on critic block features."""

    def __init__(self, dim: int):
        super().__init__()

        def _make():
            return nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

        self.real = _make()
        self.teacher = _make()

    def forward(self, tokens):
        # call the module itself (not .real / .teacher) so FSDP2's unshard hook runs
        return self.real(tokens), self.teacher(tokens)


class _FrozenGain(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.register_buffer("gain", gain)

    def forward(self, w):
        return w * self.gain


@attrs.define(slots=False)
class T2VDistillConfig_DMAD(T2VDistillConfig_rCM):
    # caption-paired real latents in the data batch (normalized like the input latents)
    dmad_real_data_key: str = "real_latents"
    # frozen-gain spectral norm on the critic (block + head Linears) and on the first
    # `dmad_sn_generator_frac` of the generator's blocks
    dmad_critic_spectral_norm: bool = True
    dmad_generator_spectral_norm: bool = True
    dmad_sn_generator_frac: float = 0.5
    dmad_sn_power_iters: int = 30
    # build, load, wrap and shard one net at a time (peak memory of one full net; needed at 14B)
    dmad_sn_staged_build: bool = False
    # gap routing of the generator's teacher term
    dmad_teacher_gap_route: bool = True
    dmad_gap_tau: float = 2.0
    dmad_gap_ema_beta: float = 0.99


class T2VDistillModel_DMAD(T2VDistillModel_rCM):

    # ------------------------ setup ------------------------

    def set_up_model(self):
        cfg = self.config
        assert cfg.loss_scale == 0, "DMAD trains the generator with the GAN loss only (loss_scale=0)"
        sn_on = cfg.dmad_critic_spectral_norm or cfg.dmad_generator_spectral_norm
        if sn_on and self.fsdp_device_mesh is not None and cfg.dmad_sn_staged_build:
            self._set_up_model_sn_staged()
            return
        if sn_on and self.fsdp_device_mesh is not None:
            # Spectral norm must be registered on full weights: build, load and wrap everything with the mesh hidden,
            # then shard. The forward-time all-gather gives the parametrization the full weight again.
            mesh = self.fsdp_device_mesh
            self.fsdp_device_mesh = None
            try:
                super().set_up_model()
                self._build_critic_heads()
            finally:
                self.fsdp_device_mesh = mesh
            self._shard_after_sn()
            return
        super().set_up_model()
        self._build_critic_heads()

    def _build_critic_heads(self):
        assert self.net_fake_score is not None, "DMAD needs net_fake_score (the critic backbone)"
        dim = self.net_fake_score.patch_embedding.out_features
        # the trainer seeds per rank, so this random init differs across ranks and is synced below
        self.critic_heads = CriticHeads(dim).to(device="cuda", dtype=torch.float32)
        self.register_buffer("_gap_ema", torch.zeros(NUM_GAP_BANDS, device="cuda"))
        self.register_buffer("_gap_ready", torch.zeros(NUM_GAP_BANDS, device="cuda"))
        if self.fsdp_device_mesh is not None:
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

            from rcm.utils.dtensor_helper import broadcast_dtensor_model_states

            fully_shard(self.critic_heads, mesh=self.fsdp_device_mesh, mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32))
            broadcast_dtensor_model_states(self.critic_heads, self.fsdp_device_mesh)
        if self.config.dmad_critic_spectral_norm:
            n = 0
            for module in (self.net_fake_score.blocks, self.critic_heads):
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        n += self._sn_wrap_linear(m)
            log.info(f"[dmad] critic spectral norm: {n} Linear weights")
        if self.config.dmad_generator_spectral_norm:
            # net_ema is wrapped identically (same forward; the EMA update pairs parameters by order)
            n = sum(self._wrap_generator_sn(net) for net in (self.net, getattr(self, "net_ema", None)) if net is not None)
            log.info(f"[dmad] generator spectral norm: {n} Linear weights")
        if self.fsdp_device_mesh is None:
            self._sync_replicated_states()

    def _sync_replicated_states(self):
        """Broadcast rank 0's heads and spectral-norm buffers (random, per-rank) and all nets once after the build."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        mods = [self.critic_heads, self.net, self.net_fake_score]
        if getattr(self, "net_ema", None) is not None:
            mods.append(self.net_ema)
        for mod in mods:
            for t in list(mod.parameters()) + list(mod.buffers()):
                dist.broadcast(t.data, src=0)

    def _shard_after_sn(self):
        """Shard the fully built, spectral-normed modules exactly as build_net would."""
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed._tensor.api import DTensor

        from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater

        mesh = self.fsdp_device_mesh
        mp = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        nets = [self.net, self.net_teacher, self.net_fake_score]
        if getattr(self, "net_ema", None) is not None:
            nets.append(self.net_ema)
        for net in nets:
            net.fully_shard(mesh=mesh, mp_policy=mp)
            fully_shard(net, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
        fully_shard(self.critic_heads, mesh=mesh, mp_policy=mp)
        if getattr(self, "net_ema", None) is not None:
            self.net_ema_worker = DTensorFastEmaModelUpdater()
        plain = [name for mod in nets + [self.critic_heads] for name, p in mod.named_parameters() if not isinstance(p.data, DTensor)]
        assert not plain, f"unsharded parameters left: {plain[:5]}"

    def _set_up_model_sn_staged(self):
        """Staged build: each net goes build (full) -> load teacher weights -> spectral norm -> rank-0 broadcast -> shard,
        one at a time; the teacher (no spectral norm) is built sharded directly."""
        import torch.distributed as dist
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
        from torch.distributed._tensor.api import DTensor

        from imaginaire.lazy_config import instantiate as lazy_instantiate
        from imaginaire.utils import misc
        from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater
        from rcm.utils.misc import count_params

        cfg = self.config
        mesh = self.fsdp_device_mesh
        mp = MixedPrecisionPolicy(reduce_dtype=torch.float32)
        assert cfg.net_fake_score is not None, "DMAD needs net_fake_score (the critic backbone)"

        def bcast_full(module):
            if dist.is_initialized() and dist.get_world_size() > 1:
                for t in list(module.parameters()) + list(module.buffers()):
                    dist.broadcast(t.data, src=0)

        def shard(module):
            module.fully_shard(mesh=mesh, mp_policy=mp)
            root = fully_shard(module, mesh=mesh, mp_policy=mp, reshard_after_forward=True)
            torch.cuda.empty_cache()
            return root

        def wrap_generator(net):
            if cfg.dmad_generator_spectral_norm:
                self._wrap_generator_sn(net)

        def wrap_critic(net):
            if cfg.dmad_critic_spectral_norm:
                for m in net.blocks.modules():
                    if isinstance(m, nn.Linear):
                        self._sn_wrap_linear(m)

        def staged_net(net_cfg, wrap):
            self.fsdp_device_mesh = None  # build_net then skips sharding
            try:
                net = self.build_net(net_cfg)
            finally:
                self.fsdp_device_mesh = mesh
            if cfg.teacher_ckpt:
                self.load_ckpt_to_net(net, cfg.teacher_ckpt)
            wrap(net)
            bcast_full(net)
            return shard(net)

        with misc.timer("Creating PyTorch model and ema if enabled (spectral norm, staged)"):
            self.conditioner = lazy_instantiate(cfg.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, \
                "conditioner should not have learnable parameters"

            self.net_teacher = self.build_net(cfg.net_teacher)
            if cfg.teacher_ckpt:
                self.load_ckpt_to_net(self.net_teacher, cfg.teacher_ckpt)
            self.net_teacher.requires_grad_(False)
            self.net_teacher.to(dtype=self.precision)

            self.fsdp_device_mesh = None
            try:
                net_full = self.build_net(cfg.net)
            finally:
                self.fsdp_device_mesh = mesh
            self._param_count = count_params(net_full, verbose=False)
            if cfg.teacher_ckpt:
                self.load_ckpt_to_net(net_full, cfg.teacher_ckpt)
            wrap_generator(net_full)
            bcast_full(net_full)
            self.net = shard(net_full)
            del net_full
            self.net.to(dtype=self.precision)

            # EMA before the critic; it starts from the same (teacher) weights as the student
            if cfg.ema.enabled:
                self.net_ema = staged_net(cfg.net, wrap_generator)
                self.net_ema.requires_grad_(False)
                self.net_ema_worker = DTensorFastEmaModelUpdater()
                s = cfg.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()

            self.net_fake_score = staged_net(cfg.net_fake_score, wrap_critic)

            cp_group = self.get_context_parallel_group()
            for net in (self.net, self.net_teacher, self.net_fake_score):
                if cp_group is not None and cp_group.size() > 1:
                    net.enable_context_parallel(cp_group)
                else:
                    net.disable_context_parallel()

            dim = self.net_fake_score.patch_embedding.out_features
            self.critic_heads = CriticHeads(dim).to(device="cuda", dtype=torch.float32)
            if cfg.dmad_critic_spectral_norm:
                for m in self.critic_heads.modules():
                    if isinstance(m, nn.Linear):
                        self._sn_wrap_linear(m)
            bcast_full(self.critic_heads)
            fully_shard(self.critic_heads, mesh=mesh, mp_policy=mp)
            self.register_buffer("_gap_ema", torch.zeros(NUM_GAP_BANDS, device="cuda"))
            self.register_buffer("_gap_ready", torch.zeros(NUM_GAP_BANDS, device="cuda"))

            mods = [self.net, self.net_teacher, self.net_fake_score, self.critic_heads]
            if getattr(self, "net_ema", None) is not None:
                mods.append(self.net_ema)
            plain = [name for mod in mods for name, p in mod.named_parameters() if not isinstance(p.data, DTensor)]
            assert not plain, f"unsharded parameters left: {plain[:5]}"
        torch.cuda.empty_cache()

    def state_dict(self):
        sd = super().state_dict()
        sd.update(self.critic_heads.state_dict(prefix="critic_heads."))
        return sd

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        heads_sd = {k[len("critic_heads."):]: v for k, v in state_dict.items() if k.startswith("critic_heads.")}
        if heads_sd:  # teacher-initialized checkpoints have no heads
            self.critic_heads.load_state_dict(heads_sd, strict=strict, assign=assign)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def model_dict(self):
        # the fake_score optimizer also owns the head parameters
        d = super().model_dict()
        d["fake_score"] = nn.ModuleDict({"fake_score": self.net_fake_score, "critic_heads": self.critic_heads})
        return d

    def init_optimizer_scheduler(self, optimizer_config, scheduler_config):
        super().init_optimizer_scheduler(optimizer_config, scheduler_config)
        # the heads train with the critic: add them to the fake_score optimizer and extend its scheduler
        opt = self.optimizer_dict["fake_score"]
        sched = self.scheduler_dict["fake_score"]
        opt.add_param_group({"params": list(self.critic_heads.parameters())})
        added = opt.param_groups[-1]
        added.setdefault("initial_lr", added["lr"])
        sched.lr_lambdas.append(sched.lr_lambdas[0])
        sched.base_lrs.append(added["initial_lr"])

    # ------------------------ critic forward ------------------------

    def _critic_forward(self, xt_B_C_T_H_W, time_B_1, condition):
        """One net_fake_score forward -> (x0 prediction, d_real, d_teacher). The heads read the token output of the last
        block (an activation-checkpoint boundary, so it stays connected to the graph)."""
        if time_B_1.ndim == 1:
            time_B_T = repeat(time_B_1, "b -> b 1")
        else:
            time_B_T = time_B_1
        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        c_skip, c_out, c_in, c_noise = self.scaling(trigflow_t=time_B_1_T_1_1)

        feats = []
        handle = self.net_fake_score.blocks[-1].register_forward_hook(lambda m, i, o: feats.append(o))
        try:
            net_output = self.net_fake_score(
                x_B_C_T_H_W=(xt_B_C_T_H_W * c_in).to(**self.tensor_kwargs),
                timesteps_B_T=c_noise.squeeze(dim=[1, 3, 4]).to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).float()
        finally:
            handle.remove()
        x0_pred = c_skip * xt_B_C_T_H_W + c_out * net_output

        tokens = feats[-1].float()  # (B, L_local, D); L_local = L / cp under context parallel
        d_real_tok, d_teacher_tok = self.critic_heads(tokens)
        d_real = d_real_tok.mean(dim=1).squeeze(-1)
        d_teacher = d_teacher_tok.mean(dim=1).squeeze(-1)
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            # global token mean = mean of the per-rank means. The value is all-reduced; the gradient flows through the
            # local mean scaled by 1/cp (no collective in backward, which would interleave with FSDP's).
            cp = cp_group.size()
            ar_r, ar_t = d_real.detach().clone(), d_teacher.detach().clone()
            dist.all_reduce(ar_r, group=cp_group)
            dist.all_reduce(ar_t, group=cp_group)
            d_real = d_real / cp + (ar_r - d_real.detach()) / cp
            d_teacher = d_teacher / cp + (ar_t - d_teacher.detach()) / cp
        return x0_pred, d_real, d_teacher

    def _noise_with_D_time(self, x0_B_C_T_H_W, time_B_1=None):
        """Noise x0 at a fresh p_D draw, or at a given time (shared across the critic's G/Q/T); eps is independent."""
        if time_B_1 is None:
            time_B_1 = self.draw_training_time_D((x0_B_C_T_H_W.shape[0], 1))
        eps = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        time_B_1, eps = self.sync(time_B_1, eps)
        tt = rearrange(time_B_1, "b t -> b 1 t 1 1")
        xt = torch.cos(tt) * x0_B_C_T_H_W + torch.sin(tt) * eps
        return xt, time_B_1

    # ------------------------ gap routing ------------------------

    def _gap_band(self, t_B):
        """Noise band (0..9) = CDF decile of the TrigFlow time under p_D (equally populated bands)."""
        u = self.p_D.cdf_rf(trig_to_rf_time(t_B.reshape(-1).to(torch.float64)))
        return (u * NUM_GAP_BANDS).long().clamp(0, NUM_GAP_BANDS - 1)

    @torch.no_grad()
    def _update_gap_ema(self, d_real_T, d_real_Q, t_T, t_Q):
        """Per-band gap E[d_real(T)] - E[d_real(Q)], all-reduced over ranks, folded into the online EMA."""
        dev = d_real_T.device
        bt = self._gap_band(t_T).to(dev)
        bq = self._gap_band(t_Q).to(dev)
        sT = torch.zeros(NUM_GAP_BANDS, device=dev); cT = torch.zeros(NUM_GAP_BANDS, device=dev)
        sQ = torch.zeros(NUM_GAP_BANDS, device=dev); cQ = torch.zeros(NUM_GAP_BANDS, device=dev)
        sT.scatter_add_(0, bt, d_real_T.float()); cT.scatter_add_(0, bt, torch.ones_like(d_real_T, dtype=torch.float32))
        sQ.scatter_add_(0, bq, d_real_Q.float()); cQ.scatter_add_(0, bq, torch.ones_like(d_real_Q, dtype=torch.float32))
        if dist.is_available() and dist.is_initialized():
            for t in (sT, cT, sQ, cQ):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        valid = (cT > 0) & (cQ > 0)
        gap_now = sT / cT.clamp(min=1) - sQ / cQ.clamp(min=1)
        beta = self.config.dmad_gap_ema_beta
        fresh = valid & (self._gap_ready > 0)
        upd = torch.where(fresh, beta * self._gap_ema + (1 - beta) * gap_now, gap_now)
        self._gap_ema.copy_(torch.where(valid, upd, self._gap_ema))
        self._gap_ready.copy_(torch.where(valid, torch.ones_like(self._gap_ready), self._gap_ready))

    @torch.no_grad()
    def _teacher_gap_weight(self, D_time_B_1):
        """(w, E_w): per-sample weight sigmoid((median gap - gap_band) / tau) and its mean over the measured bands, or
        (None, 1) until 5 of the 10 bands are measured."""
        if not self.config.dmad_teacher_gap_route:
            return None, 1.0
        ready = self._gap_ready > 0
        if int(ready.sum()) < 5:
            return None, 1.0
        center = float(self._gap_ema[ready].median())
        w_band = torch.sigmoid((center - self._gap_ema) / self.config.dmad_gap_tau)
        w = w_band[self._gap_band(D_time_B_1)]
        E_w = float(w_band[ready].mean().clamp(min=1e-3))
        return w, E_w

    # ------------------------ generator step ------------------------

    def _student_dmad_step(self, ctx, iteration):
        x0_B_C_T_H_W, condition, uncondition = ctx
        n_sim = self.get_effective_iteration(iteration) % self.config.max_simulation_steps_fake + 1
        G_x0, _ = self.backward_simulation(condition, x0_B_C_T_H_W.size(), n_sim, with_grad=True)
        xt_G, D_time = self._noise_with_D_time(G_x0)

        # Keep the critic frozen through the backward (activation checkpointing replays the forward at backward time);
        # on_after_backward unfreezes it.
        self.net_fake_score.requires_grad_(False)
        self.critic_heads.requires_grad_(False)
        self._critic_frozen = True
        _, d_real_G, d_teacher_G = self._critic_forward(xt_G, D_time, condition)

        w_gap, E_w = self._teacher_gap_weight(D_time)
        teacher_term = -d_teacher_G
        if w_gap is not None:
            teacher_term = teacher_term * w_gap / E_w
        gen_loss = teacher_term + -d_real_G
        kendall_loss = self.config.loss_scale_dmd * gen_loss  # per sample (B,)
        output_batch = {
            "G_x0": G_x0.detach().cpu(),
            "dmad_g_d_real": d_real_G.detach().mean().cpu(),
            "dmad_g_d_teacher": d_teacher_G.detach().mean().cpu(),
        }
        if w_gap is not None:
            output_batch["dmad_gap_w_mean"] = w_gap.mean().cpu()
            output_batch["dmad_gap_w_norm"] = torch.tensor(E_w)
        return output_batch, kendall_loss

    def on_after_backward(self, iteration: int = 0) -> None:
        super().on_after_backward(iteration)
        if getattr(self, "_critic_frozen", False):
            self.net_fake_score.requires_grad_(True)
            self.critic_heads.requires_grad_(True)
            self._critic_frozen = False

    # ------------------------ critic step ------------------------

    def training_step_critic(self, ctx, iteration):
        x0_B_C_T_H_W, condition, uncondition, real_x0_B_C_T_H_W = ctx
        n_sim = self.get_effective_iteration_fake(iteration) % self.config.max_simulation_steps_fake + 1
        G_x0, _ = self.backward_simulation(condition, x0_B_C_T_H_W.size(), n_sim, with_grad=False)
        D_time_B_1 = self.draw_training_time_D((x0_B_C_T_H_W.shape[0], 1))
        D_eps = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        D_time_B_1, D_eps = self.sync(D_time_B_1, D_eps)
        tt = rearrange(D_time_B_1, "b t -> b 1 t 1 1")
        xt_G = torch.cos(tt) * G_x0 + torch.sin(tt) * D_eps
        x0_pred_G, d_real_G, d_teacher_G = self._critic_forward(xt_G, D_time_B_1, condition)

        # Q = the batch latents (teacher samples), T = real latents; both at G's noise time
        xt_Q, time_Q = self._noise_with_D_time(x0_B_C_T_H_W, D_time_B_1)
        _, d_real_Q, d_teacher_Q = self._critic_forward(xt_Q, time_Q, condition)
        xt_T, time_T = self._noise_with_D_time(real_x0_B_C_T_H_W, D_time_B_1)
        _, d_real_T, _ = self._critic_forward(xt_T, time_T, condition)

        bce = F.softplus(-d_teacher_Q) + F.softplus(d_teacher_G) + F.softplus(-d_real_T) + F.softplus(d_real_G)  # (B,)
        output_batch = {
            "G_x0": G_x0.detach().cpu(),
            "x0_theta_fake": x0_pred_G.detach().cpu(),
            "dmad_acc_q": (d_teacher_Q > 0).float().mean().detach().cpu(),
            "dmad_acc_t": (d_real_T > 0).float().mean().detach().cpu(),
            "dmad_acc_g": 0.5 * ((d_real_G < 0).float().mean() + (d_teacher_G < 0).float().mean()).detach().cpu(),
        }
        self._update_gap_ema(d_real_T.detach(), d_real_Q.detach(), time_T, time_Q)
        ready = self._gap_ready > 0
        if bool(ready.any()):
            output_batch["dmad_gap_median"] = self._gap_ema[ready].median().cpu()
            for b in range(NUM_GAP_BANDS):
                if bool(ready[b]):
                    output_batch[f"dmad_gap_ema_b{b}"] = self._gap_ema[b].cpu()
        return output_batch, bce

    # ------------------------ closures ------------------------

    def training_step_closures(self, data_batch, iteration):
        _, x0_B_C_T_H_W, condition, uncondition = self.get_data_and_condition(data_batch)
        ctx = self._make_training_ctx(x0_B_C_T_H_W, condition, uncondition, iteration)
        if self.is_student_phase(iteration):
            yield "dmad_g", lambda: self._student_dmad_step(ctx, iteration), True
        else:
            key = self.config.dmad_real_data_key
            assert key in data_batch, f"DMAD needs caption-paired real latents in data_batch['{key}']"
            # the same state_t crop as the batch latents
            real_x0 = data_batch[key][:, :, : self.config.state_t].to(device="cuda", dtype=torch.float32)
            real_x0 = self.sync(real_x0)
            yield "critic", lambda: self.training_step_critic((*ctx, real_x0), iteration), True

    # ------------------------ frozen-gain spectral norm ------------------------

    def _sn_wrap_linear(self, mod, allow_frozen=False):
        """weight -> gain * W / sigma_max(W) (torch spectral_norm, one power iteration per forward) with the gain frozen
        at the initial sigma_max, so the layer is unchanged at initialization. allow_frozen also wraps weights that do
        not require grad (the EMA copy of the generator)."""
        from torch.nn.utils import parametrize
        from torch.nn.utils.parametrizations import spectral_norm

        w = getattr(mod, "weight", None)
        if not isinstance(w, nn.Parameter) or w.dim() != 2 or (not w.requires_grad and not allow_frozen):
            return 0
        if torch.linalg.matrix_norm(w.detach().float(), ord=2) < 1e-8:  # zero-initialized layer
            return 0
        w_orig = w.detach().clone()
        spectral_norm(mod, name="weight", n_power_iterations=1)
        with torch.no_grad():  # converge the power iteration at init
            for _ in range(self.config.dmad_sn_power_iters):
                _ = mod.weight
            wn = mod.weight.detach()
            gain = (w_orig.flatten().float() @ wn.flatten().float()) / (wn.flatten().float() @ wn.flatten().float())
        parametrize.register_parametrization(mod, "weight", _FrozenGain(gain.to(w_orig.dtype)))
        return 1

    def _wrap_generator_sn(self, net):
        n = 0
        n_blocks = math.ceil(len(net.blocks) * self.config.dmad_sn_generator_frac)
        for block in list(net.blocks)[:n_blocks]:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    n += self._sn_wrap_linear(m, allow_frozen=True)
        return n
