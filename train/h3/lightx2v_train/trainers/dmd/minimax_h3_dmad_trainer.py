"""DMAD for MiniMax-H3 T2AV: GAN-only distillation with a two-head critic.

The student is a few-step generator (LoRA on the H3 transformer). The critic is a second LoRA copy of the
transformer ("fake model") whose features at `feature_block` feed two binary heads:

    real head    : real data (T)       vs generated (G)
    teacher head : teacher samples (Q) vs generated (G)

Critic loss (4-term BCE):  softplus(d_teacher(G)) + softplus(d_real(G)) + softplus(-d_teacher(Q)) + softplus(-d_real(T))
Generator loss:            -d_real(G) - w_gap * d_teacher(G)

G, T and Q are re-noised at one shared (video, audio) sigma draw per step. Real clips are caption-paired:
in the critic step the real clip is drawn first and its caption conditions the generator rollout, so the
critic always compares same-prompt samples. w_gap is a soft per-noise-band routing weight on the teacher
term, computed from the running gap between the real head's scores on T and on Q.

The student's LoRA parameters are tracked by two power-function EMAs; each checkpoint exports them as
inference-ready LoRAs (ema_lora/, ema2_lora/).

Constraints: batch size 1 per GPU (packed sequence), gradient_accumulation_iters = 1, no sequence parallel,
a single frame count (the real pool is filtered to the same tier).
"""

import glob
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from lightx2v_train.model_zoo.native.minimax_h3.sampling import decode_rows, load_vaes, write_mp4
from lightx2v_train.runtime.distributed import get_world_size, is_main_process
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .minimax_h3_trainer import MiniMaxH3T2AVDmdTrainer

GAP_BANDS = 10
GAP_EMA_BETA = 0.99
GAP_READY_MIN = 5  # number of ready bands before the routing engages
GAP_BAND_MIN_COUNT = 10  # updates before a band counts as ready


class _CriticTruncate(Exception):
    """Control flow: stop the critic forward right after the feature block."""


class CriticHeads(nn.Module):
    """Two per-token binary heads (Linear-SiLU-Linear) on the critic's block features, mean-pooled by the caller.

    The features are normalized with a non-affine RMSNorm first: H3's pre-LN residual stream grows with depth
    (activations of order 1e4 at the last block), and unnormalized inputs saturate the BCE at initialization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.real = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))
        self.teacher = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 1))

    def forward(self, tokens):
        tokens = tokens * torch.rsqrt(tokens.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.real(tokens), self.teacher(tokens)


def data_file(root, name):
    """root/name, or root/<last two characters of the stem>/name: the released data keeps the large folders in such
    prefix buckets (the Hub allows at most 10,000 files per directory); the data_process scripts write them flat."""
    flat = os.path.join(root, name)
    return flat if os.path.isfile(flat) else os.path.join(root, os.path.splitext(name)[0][-2:], name)


class RealLatentPool:
    """Random access into the encoded real-clip pool (data_process/encode_real_videos.py output), caption-paired
    with the prompt cache and with the teacher samples (both indexed by caption index)."""

    def __init__(self, latent_dir, num_frames, seed, caption_map, prompt_cache_dir, teacher_latent_dir):
        manifests = sorted(glob.glob(os.path.join(latent_dir, "manifest_rank*.jsonl")))
        if not manifests:
            raise FileNotFoundError(f"no manifest_rank*.jsonl under {latent_dir}")
        self.cond_dir = os.path.join(prompt_cache_dir, "conditions")
        clip2idx = {}
        for line in open(caption_map):
            r = json.loads(line)
            for c in r["clip_ids"]:
                clip2idx.setdefault(c.rsplit(".", 1)[0], int(r["index"]))
        self.teacher_dir = teacher_latent_dir
        self.paths, self.cond_idx = [], []
        unmapped = missing_teacher = 0
        for mf in manifests:
            for line in open(mf):
                r = json.loads(line)
                if "skip" in r or r.get("num_frames") != num_frames:
                    continue
                idx = clip2idx.get(r["clip_id"])
                if idx is None:
                    unmapped += 1
                    continue
                if not os.path.isfile(data_file(self.teacher_dir, f"{idx:08d}.pt")):
                    missing_teacher += 1
                    continue
                self.cond_idx.append(idx)
                self.paths.append(data_file(os.path.join(latent_dir, "latents"), f"{r['clip_id']}.pt"))
        if not self.paths:
            raise RuntimeError(f"real pool empty for num_frames={num_frames} under {latent_dir}")
        if unmapped:
            raise RuntimeError(f"{unmapped} real clips have no caption in {caption_map}")
        if missing_teacher:
            raise RuntimeError(f"{missing_teacher} real clips have no teacher sample under {self.teacher_dir}")
        self.rng = random.Random(seed)
        logger.info("[train] real pool: {} clips @ {} frames from {}", len(self.paths), num_frames, latent_dir)

    def _condition(self, i):
        payload = torch.load(data_file(self.cond_dir, f"condition_{self.cond_idx[i]:08d}.pt"), map_location="cpu", weights_only=False)
        return payload["conditioning"]["positive"]

    def draw(self, device, dtype=torch.float32):
        """-> (real video, real audio, teacher video rows, teacher audio rows, condition payload) of one random clip."""
        i = self.rng.randrange(len(self.paths))
        d = torch.load(self.paths[i], weights_only=False)
        video = d["video_latent"].to(device=device, dtype=dtype)  # [24, T', 48, 84]
        audio = d["audio_latent"].to(device=device, dtype=dtype)  # [2, n, 32]
        t = torch.load(data_file(self.teacher_dir, f"{self.cond_idx[i]:08d}.pt"), map_location="cpu", weights_only=False)
        tv = t["video_rows"].to(device=device, dtype=dtype)  # packed rows [1, R_v, 96]
        ta = t["audio_rows"].to(device=device, dtype=dtype)  # packed rows [1, R_a, 32]
        return video, audio, tv, ta, self._condition(i)

    def draw_condition(self):
        """Caption of a random pool clip (no latent load)."""
        return self._condition(self.rng.randrange(len(self.paths)))


@TRAINER_REGISTER("minimax_h3_t2av_dmad")
class MiniMaxH3T2AVDmadTrainer(MiniMaxH3T2AVDmdTrainer):
    trainer_name = "minimax_h3_t2av_dmad"

    def __init__(self, config):
        super().__init__(config)
        u = self.training_config["dmad"]
        self.u_prompt_cache_dir = u["prompt_cache_dir"]
        self.u_real_latent_dir = u["real_latent_dir"]
        self.u_teacher_dir = u["teacher_latent_dir"]
        self.u_caption_map = u["real_caption_map"]
        self.u_gap_tau = float(u.get("gap_tau", 2.0))
        self.u_feature_block = int(u.get("feature_block", 49))
        # fake.train_type full: train the critic's transformer blocks [k:] (+ heads), freeze the rest (memory)
        self.u_critic_freeze_blocks = int(u.get("critic_freeze_blocks", 0))
        # the heads follow the critic optimizer's hyper-parameters
        self.u_head_lr = float(self.fake_optimizer_config.get("learning_rate", 2e-5))
        self.u_head_wd = float(self.fake_optimizer_config.get("weight_decay", 0.01))
        self.u_ema_gammas = [float(g) for g in u.get("ema_gammas", [6.94, 16.97])]
        self.u_sample_num_prompts = int(u.get("sample_num_prompts", 4))
        if int(self.training_config.get("gradient_accumulation_iters", 1)) != 1:
            raise ValueError("minimax_h3_t2av_dmad requires gradient_accumulation_iters=1.")
        self._critic_frozen = False
        self._student_step_count = 0
        self._critic_calls = 0
        # max_grad_norm: 0 disables clipping (the shared DMD loop clips unconditionally, so use +inf)
        if float(self.max_grad_norm) == 0.0:
            self.max_grad_norm = float("inf")

    # ------------------------ setup ------------------------

    def _setup_trainable_model(self, model, role="student"):
        super()._setup_trainable_model(model, role)
        k = self.u_critic_freeze_blocks
        if role != "fake" or self.fake_train_type != "full" or k <= 0:
            return
        tr = model.denoiser_module()
        frozen = [tr.token_refiner] + list(tr.transformer_blocks[:k])
        for name in ("proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj", "rope"):
            if hasattr(tr, name):
                frozen.append(getattr(tr, name))
        for m in frozen:
            m.requires_grad_(False)
        n_tr = sum(p.numel() for p in tr.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in tr.parameters())
        logger.info("[train] full critic: froze refiner + embedders + blocks[:{}]; trainable {:.2f}B / {:.2f}B params (blocks[{}:] + post-block heads)",
                    k, n_tr / 1e9, n_all / 1e9, k)

    def setup(self, resume_ckpt_path=None):
        super().setup(resume_ckpt_path=resume_ckpt_path)
        # GAN-only: the teacher score network is never queried during training.
        if getattr(self, "teacher_model", None) is not None:
            self.teacher_model = None
            torch.cuda.empty_cache()

        dim = int(self.fake_model.denoiser_module().config.hidden_size)
        self.critic_heads = CriticHeads(dim).to(device=self.model.device, dtype=torch.float32)
        fo = self.fake_optimizer_config
        self.head_optimizer = torch.optim.AdamW(
            self.critic_heads.parameters(),
            lr=self.u_head_lr,
            betas=(float(fo.get("adam_beta1", 0.9)), float(fo.get("adam_beta2", 0.999))),
            weight_decay=self.u_head_wd,
            eps=float(fo.get("adam_epsilon", 1e-8)),
        )
        # per-rank EMA of gap = d_real(T) - d_real(Q) in 10 CDF-decile bands of the noise draw
        self._gap_ema = torch.zeros(GAP_BANDS, device=self.model.device)
        self._gap_cnt = torch.zeros(GAP_BANDS, device=self.model.device)

        self.real_pool = RealLatentPool(
            self.u_real_latent_dir,
            int(self.dmd_config["num_frames"]),
            seed=42 + int(os.environ.get("RANK", 0)),
            caption_map=self.u_caption_map,
            prompt_cache_dir=self.u_prompt_cache_dir,
            teacher_latent_dir=self.u_teacher_dir,
        )
        self._ema_banks = [[p.detach().clone().float() for p in self.trainable_params] for _ in self.u_ema_gammas]

        # The GAN loss only reaches critic blocks <= feature_block, so later LoRA params never get gradients
        # and AdamW would never create their state, which breaks checkpoint resume. Create the state up front
        # with one lr=0 step on zero gradients (a numerical no-op, weight decay included).
        for opt, params in ((self.optimizer, self.trainable_params), (self.fake_optimizer, self.fake_trainable_params)):
            self._materialize_optimizer_state(opt, params)
        self._maybe_load_critic_state(resume_ckpt_path)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self._wandb = None
        if is_main_process() and os.environ.get("WANDB_MODE", "disabled") != "disabled":
            try:
                import wandb

                out = self.output_train_dir.rstrip("/")
                self._wandb = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "dmad-h3"),
                    name=os.environ.get("WANDB_NAME") or os.path.basename(os.path.dirname(out) if os.path.basename(out) == "out" else out),
                    config={"trainer": self.trainer_name, "config": dict(self.training_config)},
                    resume="allow",
                    id=os.environ.get("WANDB_RUN_ID") or None,
                )
            except Exception as exc:  # tracking must not block training
                logger.warning("[train] wandb init failed ({}); metrics stay local-only", exc)

    @staticmethod
    def _materialize_optimizer_state(optimizer, params):
        lrs = [g["lr"] for g in optimizer.param_groups]
        for g in optimizer.param_groups:
            g["lr"] = 0.0
        for p in params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for g, lr in zip(optimizer.param_groups, lrs):
            g["lr"] = lr

    def log_metrics(self, metrics, step=None):
        super().log_metrics(metrics, step=step)
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    # ------------------------ critic ------------------------

    def _critic_forward(self, latents, sigmas, condition, latent_shape):
        """Critic forward -> (real-head, teacher-head) logits per sample.

        Token logits are averaged over audio tokens and over video tokens separately, and the two means are
        averaged with equal weight (a plain token mean would be ~99% video: 207 audio vs 31,248 video tokens)."""
        layout = self._layout(condition, latent_shape)
        feats = []
        blocks = self.fake_model.denoiser_module().transformer_blocks
        handle = blocks[self.u_feature_block].register_forward_hook(lambda m, i, o: feats.append(o[0] if isinstance(o, tuple) else o))
        stop_handle = None
        if self.u_feature_block + 1 < len(blocks):
            # the blocks after the feature block are never used: skip them (all ranks stop at the same block)
            def _stop(module, args, kwargs):
                raise _CriticTruncate

            stop_handle = blocks[self.u_feature_block + 1].register_forward_pre_hook(_stop, with_kwargs=True)
        try:
            self._predict_velocity(self.fake_model, latents, sigmas, condition, latent_shape)
        except _CriticTruncate:
            pass
        finally:
            handle.remove()
            if stop_handle is not None:
                stop_handle.remove()
        tokens = feats[-1].float()  # [1, L, D], packed [text | audio | video]
        a_idx = layout.audio_indices.to(tokens.device)
        v_idx = layout.video_indices.to(tokens.device)
        d_real_tok, d_teacher_tok = self.critic_heads(tokens[:, torch.cat([a_idx, v_idx])])
        n_a = a_idx.numel()

        def _pool(d_tok):
            return ((1.0 * d_tok[:, :n_a].mean(dim=1) + 1.0 * d_tok[:, n_a:].mean(dim=1)) / 2.0).squeeze(-1)

        return _pool(d_real_tok), _pool(d_teacher_tok)

    def _freeze_critic(self, freeze: bool):
        for p in self.fake_trainable_params:
            p.requires_grad_(not freeze)
        self.critic_heads.requires_grad_(not freeze)
        self._critic_frozen = freeze

    # ------------------------ gap routing ------------------------

    def _sigma_band(self, video_sigma):
        """CDF-decile band of the noise draw. The draw is uniform then flow-shifted, so the CDF value is the
        pre-shift uniform u = s' / (s - (s - 1) s') with s = video_shift."""
        sp = float(video_sigma)
        sh = float(self.video_shift)
        u = sp / (sh - (sh - 1.0) * sp)
        return min(GAP_BANDS - 1, max(0, int(u * GAP_BANDS)))

    def _gap_update(self, band, d_r_T, d_r_Q):
        g = (d_r_T.detach().mean() - d_r_Q.detach().mean()).to(self._gap_ema.device)
        self._gap_ema[band] = GAP_EMA_BETA * self._gap_ema[band] + (1.0 - GAP_EMA_BETA) * g
        self._gap_cnt[band] += 1

    def _gap_weight(self, band):
        """(w, E_w): w = sigmoid((median - gap_band) / tau) over the bias-corrected gap EMAs of the ready bands,
        E_w = mean weight over ready bands (the loss uses w / E_w). Returns (1, 1) until enough bands are ready."""
        ready = self._gap_cnt >= GAP_BAND_MIN_COUNT
        if int(ready.sum()) < GAP_READY_MIN or not bool(ready[band]):
            return 1.0, 1.0
        corr = self._gap_ema / (1.0 - GAP_EMA_BETA ** self._gap_cnt.clamp(min=1.0))
        center = corr[ready].median()
        w_band = torch.sigmoid((center - corr) / self.u_gap_tau)
        e_w = float(w_band[ready].mean().clamp(min=1e-4))
        return float(w_band[band]), e_w

    # ------------------------ losses ------------------------

    def forward_loss(self, latent_shape, stage, initial_noise=None):
        # Training prompts are the captions of the real-pool clips: the critic step draws a real clip first and
        # conditions the rollout on its caption; the student step uses a random pool caption.
        teacher_v = teacher_a = real_v = real_a = None
        if stage == "fake":
            real_v, real_a, teacher_v, teacher_a, cond_payload = self.real_pool.draw(self.model.device, dtype=torch.float32)
        else:
            cond_payload = self.real_pool.draw_condition()
        condition = self._prepare_cached_condition(cond_payload)
        end_step_idx = self._sample_synced_int(0, self.num_inference_steps)
        generated = self.run_back_simulation(condition, latent_shape, end_step_idx, grad_enabled=stage != "fake", xt=initial_noise)
        sigmas = self._sample_renoise_sigmas()  # (video_sigma, audio_sigma), shared by G, T and Q
        noises_g = (torch.randn_like(generated[0], dtype=torch.float32), torch.randn_like(generated[1], dtype=torch.float32))
        renoised_g = self._add_noise(generated, noises_g, sigmas)

        if stage == "student":
            # The critic stays frozen through the backward (gradient checkpointing re-runs the forward there);
            # it is unfrozen in _after_student_optimizer_step.
            self._freeze_critic(True)
            self.fake_model.transformer.eval()
            d_g_r, d_g_t = self._critic_forward(renoised_g, sigmas, condition, latent_shape)
            w_gap, e_w = self._gap_weight(self._sigma_band(sigmas[0]))
            loss = (-d_g_r).mean() + (w_gap / e_w) * (-d_g_t).mean()
            # "dmd" is the metric key the shared training loop expects from the student stage
            return {"loss": loss, "dmd": loss.detach(), "d_g": d_g_r.detach().mean(), "d_g_teacher": d_g_t.detach().mean(), "gap_w": torch.tensor(w_gap / e_w)}

        # ---- critic stage ----
        self.fake_model.transformer.train()
        real = self._pack_real(real_v, real_a, latent_shape)
        noises_t = (torch.randn_like(real[0], dtype=torch.float32), torch.randn_like(real[1], dtype=torch.float32))
        renoised_t = self._add_noise(real, noises_t, sigmas)
        d_g_r, d_g_t = self._critic_forward(renoised_g, sigmas, condition, latent_shape)
        d_t, _ = self._critic_forward(renoised_t, sigmas, condition, latent_shape)
        noises_q = (torch.randn_like(teacher_v, dtype=torch.float32), torch.randn_like(teacher_a, dtype=torch.float32))
        renoised_q = self._add_noise((teacher_v, teacher_a), noises_q, sigmas)
        d_q_realhead, d_q = self._critic_forward(renoised_q, sigmas, condition, latent_shape)
        bce = F.softplus(d_g_t) + F.softplus(d_g_r) + F.softplus(-d_q) + F.softplus(-d_t)
        self._gap_update(self._sigma_band(sigmas[0]), d_t, d_q_realhead)
        loss = bce.mean()
        out = {
            "gan_d": loss.detach(),
            "acc_t": (d_t > 0).float().mean().detach(),
            "acc_q": (d_q > 0).float().mean().detach(),
            "acc_g": (0.5 * ((d_g_r < 0).float().mean() + (d_g_t < 0).float().mean())).detach(),
        }
        self._log_critic_window(out, d_t, d_q, d_g_r, sigmas)
        return {"loss": loss, **out}

    def _log_critic_window(self, out, d_t, d_q, d_g, sigmas):
        """Each rank holds one sample per critic call, so per-call accuracies are 0/1. Accumulate over the log
        window on every rank and all-reduce before logging."""
        self._critic_calls += 1
        dev = d_t.device
        if getattr(self, "_acc_window", None) is None:
            self._acc_window = torch.zeros(7, device=dev)  # acc_t, acc_q, acc_g, d_t, d_q, d_g, n
        self._acc_window += torch.stack([out["acc_t"].to(dev), out["acc_q"].to(dev), out["acc_g"].to(dev),
                                         d_t.detach().mean(), d_q.detach().mean(), d_g.detach().mean(), torch.ones((), device=dev)])
        if self._critic_calls % max(1, self.train_log_every_iters) != 0:
            return
        tot = self._acc_window.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(tot, op=torch.distributed.ReduceOp.SUM)
        n = tot[6].clamp_min(1.0)
        ready = int((self._gap_cnt >= GAP_BAND_MIN_COUNT).sum().item())
        self.log_metrics(
            {
                "train/dmad_acc_t": (tot[0] / n).item(),
                "train/dmad_acc_q": (tot[1] / n).item(),
                "train/dmad_acc_g": (tot[2] / n).item(),
                "train/dmad_d_t": (tot[3] / n).item(),
                "train/dmad_d_q": (tot[4] / n).item(),
                "train/dmad_d_g": (tot[5] / n).item(),
                "train/grad_norm_gen": getattr(self, "_last_grad_norm_gen", 0.0),
                "train/grad_norm_critic": getattr(self, "_last_grad_norm_critic", 0.0),
                "train/grad_norm_heads": getattr(self, "_last_grad_norm_heads", 0.0),
                "train/dmad_gap_ready_bands": ready,
                "train/dmad_gap_median": float(self._gap_ema[self._gap_cnt >= GAP_BAND_MIN_COUNT].median().item()) if ready else 0.0,
                "train/dmad_sigma_v": float(sigmas[0]),
                "train/mem_peak_gb": torch.cuda.max_memory_allocated() / 2**30,
            },
            step=self._critic_calls,
        )
        self._acc_window.zero_()

    def _pack_real(self, video, audio, latent_shape):
        """Stored real latents -> packed rows: video [24, T', H', W'] -> [1, T'*(H'/2)*(W'/2), 96] (1x2x2 patches),
        audio [2, n, 32] -> [1, 2n, 32] (channel 0 rows, then channel 1)."""
        pt, ph, pw = self.model.patch_size
        c, t, h, w = video.shape
        v = video.reshape(c, t, h // ph, ph, w // pw, pw)
        v = v.permute(1, 2, 4, 0, 3, 5).reshape(1, t * (h // ph) * (w // pw), c * ph * pw)
        a = audio.reshape(1, audio.shape[0] * audio.shape[1], audio.shape[2])
        expect_v, expect_a = latent_shape["video_tokens"], latent_shape["audio_tokens"]
        if tuple(v.shape) != tuple(expect_v) or tuple(a.shape) != tuple(expect_a):
            raise RuntimeError(f"real latent shape {tuple(v.shape)}/{tuple(a.shape)} != expected {expect_v}/{expect_a}")
        return v.to(self.latent_dtype), a.to(self.latent_dtype)

    # ------------------------ optimizer hooks, EMA ------------------------

    @staticmethod
    def _grad_total_norm(params):
        ps = [p for p in params if p.grad is not None]
        if not ps:
            return 0.0
        total = torch.nn.utils.clip_grad_norm_(ps, float("inf"))  # norm only (FSDP2/DTensor aware)
        if hasattr(total, "full_tensor"):
            total = total.full_tensor()
        return float(total)

    def _before_optimizer_step(self, params):
        if params is self.trainable_params:
            self._last_grad_norm_gen = self._grad_total_norm(params)
        # Critic step: the heads are replicated (not FSDP-sharded), so average their gradients manually and
        # step their optimizer here, once per critic update.
        if params is self.fake_trainable_params and not self._critic_frozen:
            if get_world_size() > 1:
                for p in self.critic_heads.parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
            self._last_grad_norm_critic = self._grad_total_norm(params)
            self._last_grad_norm_heads = self._grad_total_norm(list(self.critic_heads.parameters()))
            if not math.isinf(float(self.max_grad_norm)):
                torch.nn.utils.clip_grad_norm_(self.critic_heads.parameters(), self.max_grad_norm)
            self.head_optimizer.step()
            self.head_optimizer.zero_grad(set_to_none=True)

    def _after_student_optimizer_step(self, region):
        super()._after_student_optimizer_step(region)
        self._freeze_critic(False)
        # power-function EMA: beta_t = (1 - 1/t)^(gamma + 1)
        self._student_step_count += 1
        t = self._student_step_count
        with torch.no_grad():
            for gamma, bank in zip(self.u_ema_gammas, self._ema_banks):
                beta = (1.0 - 1.0 / t) ** (gamma + 1.0)
                for ema, p in zip(bank, self.trainable_params):
                    ema.mul_(beta).add_(p.detach().float(), alpha=1.0 - beta)

    def _swap_in_ema(self, bank=0):
        self._live_backup = [p.detach().clone() for p in self.trainable_params]
        with torch.no_grad():
            for p, ema in zip(self.trainable_params, self._ema_banks[bank]):
                p.copy_(ema.to(p.dtype))

    def _swap_back_live(self):
        with torch.no_grad():
            for p, live in zip(self.trainable_params, self._live_backup):
                p.copy_(live)
        self._live_backup = None

    # ------------------------ visualization ------------------------

    @torch.no_grad()
    def _rollout_fixed_prompts(self, n_prompts, tag):
        cache_dir = self.u_prompt_cache_dir
        latents = []
        self.model.transformer.eval()
        for pid in range(n_prompts):
            payload = torch.load(data_file(os.path.join(cache_dir, "conditions"), f"condition_{pid:08d}.pt"), map_location="cpu", weights_only=False)
            condition = self._prepare_cached_condition(payload["conditioning"]["positive"])
            latent_shape = self._latent_shape()
            # visualization always uses the fixed inference grid
            gen = self.run_back_simulation(condition, latent_shape, self.num_inference_steps - 1, grad_enabled=False, random_timesteps=False)
            latents.append((pid, tag, payload.get("prompt", ""), gen[0].cpu(), gen[1].cpu(), latent_shape))
        self.model.transformer.train()
        return latents

    def _draw_samples(self, current_iter):
        """Samples of the first prompts of the prompt cache from the live and the EMA student, decoded on rank 0.
        All ranks run the rollouts (FSDP forwards are collective). Failures here never stop training."""
        if self.u_sample_num_prompts <= 0:
            return
        sample_dir = os.path.join(self.output_train_dir, "samples", f"iter_{current_iter:08d}")
        with torch.no_grad():
            latents = self._rollout_fixed_prompts(self.u_sample_num_prompts, "live")
            self._swap_in_ema()
            try:
                latents += self._rollout_fixed_prompts(self.u_sample_num_prompts, "ema")
            finally:
                self._swap_back_live()
        if not is_main_process():
            return
        try:
            os.makedirs(sample_dir, exist_ok=True)
            torch.cuda.empty_cache()
            self._decode_samples_rank0(latents, sample_dir)
        except Exception as exc:
            logger.warning("[train] sample decode failed at iter {}: {}", current_iter, exc)
        finally:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _decode_samples_rank0(self, latents, sample_dir):
        if not hasattr(self, "_viz_vaes"):
            vae, avae = load_vaes(self.model_config["pretrained_model_name_or_path"], "cpu")
            self._viz_vaes = (vae.requires_grad_(False), avae.requires_grad_(False))
        vae, avae = self._viz_vaes
        vae.to(self.model.device), avae.to(self.model.device)
        try:
            for pid, tag, prompt, v_rows, a_rows, shape in latents:
                frames, wav = decode_rows(vae, avae, v_rows, a_rows, shape, self.model.patch_size,
                                          self.model.video_latent_channels, self.model.audio_latent_channels)
                write_mp4(frames, wav, os.path.join(sample_dir, f"prompt{pid:02d}_{tag}.mp4"))
                with open(os.path.join(sample_dir, f"prompt{pid:02d}.txt"), "w") as f:
                    f.write(prompt)
            logger.info("[train] samples written to {}", sample_dir)
        finally:
            vae.to("cpu"), avae.to("cpu")

    # ------------------------ checkpointing ------------------------

    def _critic_state_path(self, ckpt_dir):
        return os.path.join(ckpt_dir, "critic_state.pt")

    def save_checkpoint(self, current_iter, save_total_limit):
        torch.cuda.empty_cache()  # the DCP save gathers full states and needs headroom
        super().save_checkpoint(current_iter, save_total_limit)
        ckpt_dir = os.path.join(self.output_train_dir, f"checkpoint-{current_iter:09d}")
        # Export each EMA bank as an inference-ready LoRA (same layout as the live pytorch_lora_weights.safetensors).
        for bi, sub in enumerate(("ema_lora", "ema2_lora")[: len(self._ema_banks)]):
            self._swap_in_ema(bank=bi)
            try:
                self.model.save_lora_weights(ckpt_dir, weights_subdir=sub)
            finally:
                self._swap_back_live()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._draw_samples(current_iter)
        if is_main_process() and os.path.isdir(ckpt_dir):
            torch.save(
                {
                    "heads": self.critic_heads.state_dict(),
                    "head_optimizer": self.head_optimizer.state_dict(),
                    "student_step_count": self._student_step_count,
                    "gap_ema": self._gap_ema.detach().cpu(),
                    "gap_cnt": self._gap_cnt.detach().cpu(),
                    "critic_calls": self._critic_calls,
                    "ema_gammas": self.u_ema_gammas,
                    "ema_banks": self._ema_banks,
                },
                self._critic_state_path(ckpt_dir),
            )
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _maybe_load_critic_state(self, resume_ckpt_path):
        if not resume_ckpt_path:
            return
        path = self._critic_state_path(resume_ckpt_path)
        if not os.path.isfile(path):
            logger.warning("[train] {} missing; heads and EMA start fresh", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.critic_heads.load_state_dict(state["heads"])
        self.critic_heads.to(self.model.device)
        self.head_optimizer.load_state_dict(state["head_optimizer"])
        self._student_step_count = int(state["student_step_count"])
        self._gap_ema = state["gap_ema"].to(self.model.device)
        self._gap_cnt = state["gap_cnt"].to(self.model.device)
        self._critic_calls = int(state["critic_calls"])
        if [float(g) for g in state["ema_gammas"]] != self.u_ema_gammas:
            raise RuntimeError(f"resume ema_gammas {state['ema_gammas']} != config {self.u_ema_gammas}")
        self._ema_banks = [[t.to(self.model.device) for t in bank] for bank in state["ema_banks"]]
        logger.info("[train] critic state resumed from {}", path)
