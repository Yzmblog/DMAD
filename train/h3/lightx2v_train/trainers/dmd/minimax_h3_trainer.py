"""MiniMax-H3 text-to-audio-video student rollout: packed video+audio sequences and the re-noise step rule."""

import torch
from loguru import logger

from lightx2v_train.model_zoo.native.minimax_h3 import (
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)
from lightx2v_train.model_zoo.native.minimax_h3.sampling import shift_sigma

from .trainer import DmdTrainer


class MiniMaxH3T2AVDmdTrainer(DmdTrainer):
    """Student rollout and critic forward for H3 (data-ward velocity: x0 = x_t + sigma * v)."""

    allowed_model_names = {"minimax_h3_t2av"}
    default_negative_prompt = ""
    default_lora_target_modules = ("to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2")

    def __init__(self, config):
        super().__init__(config)
        h3 = self.training_config.get("minimax_h3", {})
        self.video_shift = float(h3.get("video_flow_shift", 12.0))
        self.audio_shift = float(h3.get("audio_flow_shift", 2.0))
        if self.video_shift <= 0 or self.audio_shift <= 0:
            raise ValueError("MiniMax-H3 flow shifts must be positive.")
        dtype_name = str(h3.get("latent_dtype", "fp32")).lower()
        latent_dtypes = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16}
        if dtype_name not in latent_dtypes:
            raise ValueError(f"Unsupported training.minimax_h3.latent_dtype={dtype_name!r}.")
        self.latent_dtype = latent_dtypes[dtype_name]
        self._layout_cache = {}
        # Fixed shifted grid with num_inference_steps evaluations (+ terminal 0), used for visualization.
        base = torch.linspace(1.0, 0.0, self.num_inference_steps + 1, dtype=torch.float32)
        self.video_sigmas_cpu = shift_sigma(base, self.video_shift)
        self.audio_sigmas_cpu = shift_sigma(base, self.audio_shift)
        logger.info("[train] H3 rollout: evaluations={} video_shift={} audio_shift={}", self.num_inference_steps, self.video_shift, self.audio_shift)

    def _prepare_cached_condition(self, condition):
        return self.model.prepare_text_condition(condition)

    def _latent_shape(self):
        dmd = self.dmd_config
        height, width, num_frames = int(dmd["height"]), int(dmd["width"]), int(dmd["num_frames"])
        if height % 32 or width % 32:
            raise ValueError(f"MiniMax-H3 height/width must be divisible by 32, got {height}x{width}.")
        latent_frames = video_latent_num_frames(num_frames)
        latent_height = height // self.model.vae_spatial_scale_factor
        latent_width = width // self.model.vae_spatial_scale_factor
        patch_t, patch_h, patch_w = self.model.patch_size
        video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
        video_dim = self.model.video_latent_channels * patch_t * patch_h * patch_w
        audio_latents = audio_latent_num_frames(num_frames)
        return {
            "batch_size": 1,
            "num_frames": num_frames,
            "latent_frames": latent_frames,
            "latent_height": latent_height,
            "latent_width": latent_width,
            "audio_latents": audio_latents,
            "video_tokens": (1, video_rows, video_dim),
            "audio_tokens": (1, audio_latents * 2, self.model.audio_latent_channels),
        }

    def sample_initial_latents(self, latent_shape):
        video = torch.randn(latent_shape["video_tokens"], device=self.model.device, dtype=self.latent_dtype)
        audio = torch.randn(latent_shape["audio_tokens"], device=self.model.device, dtype=self.latent_dtype)
        return video, audio

    def _layout(self, condition, latent_shape):
        tags = condition["text_token_tags"]
        if tags.ndim != 1 or not bool((tags == 1).all()):
            raise ValueError("Only text-only conditions are supported.")
        key = (
            int(tags.numel()),
            latent_shape["latent_frames"],
            latent_shape["latent_height"],
            latent_shape["latent_width"],
            latent_shape["audio_latents"],
            self.model.patch_size,
            self.model.device,
        )
        layout = self._layout_cache.get(key)
        if layout is None:
            layout = build_packed_sequence(
                tags.detach().cpu(),
                latent_shape["latent_frames"],
                latent_shape["latent_height"],
                latent_shape["latent_width"],
                latent_shape["audio_latents"],
                self.model.patch_size,
            ).to(self.model.device)
            self._layout_cache[key] = layout
        return layout

    def _predict_velocity(self, model, latents, sigmas, condition, latent_shape):
        video, audio = latents
        video_sigma, audio_sigma = sigmas
        if video.shape[0] != 1 or audio.shape[0] != 1:
            raise ValueError("MiniMax-H3 packed training requires batch_size=1.")
        layout = self._layout(condition, latent_shape)
        timesteps, timestep_indices = build_row_timesteps(layout, video_sigma, audio_sigma)
        with model.transformer_forward_context():
            return model.denoiser_module()(
                hidden_states=video,
                audio_hidden_states=audio,
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=timesteps.to(model.device),
                timestep_indices=timestep_indices.to(model.device),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )

    def _rollout_sigmas(self, end_step_idx, random_timesteps):
        """(video_sigmas, audio_sigmas) of one rollout, length end_step_idx + 2 (terminal 0).

        random_timesteps=False: the fixed inference grid.
        random_timesteps=True : start at pure noise; each intermediate level is a fresh draw from the critic's
        noise distribution (uniform in [renoise_sigma_min, renoise_sigma_max], then flow-shifted per modality),
        taken as a running min so the levels decrease."""
        dev = self.model.device
        if not random_timesteps:
            return self.video_sigmas_cpu[: end_step_idx + 2].to(dev), self.audio_sigmas_cpu[: end_step_idx + 2].to(dev)
        low = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        high = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        base = torch.ones(end_step_idx + 2, device=dev, dtype=torch.float32)
        running = base[0]
        for i in range(1, end_step_idx + 1):
            u = torch.empty((), device=dev, dtype=torch.float32).uniform_(low, high)
            running = torch.minimum(u, running)
            base[i] = running
        base[-1] = 0.0
        return shift_sigma(base, self.video_shift), shift_sigma(base, self.audio_shift)

    def run_back_simulation(self, condition, latent_shape, end_step_idx, grad_enabled, xt=None, random_timesteps=True):
        """Student rollout with end_step_idx + 1 evaluations; only the last one carries gradients.
        Between evaluations the x0 estimate is re-noised with fresh noise to the next level (DMD2 multistep rule)."""
        if xt is None:
            xt = self.sample_initial_latents(latent_shape)
        video_sigmas, audio_sigmas = self._rollout_sigmas(end_step_idx, random_timesteps)
        self.model.transformer.train()
        x0 = None
        for step_idx in range(end_step_idx + 1):
            video_sigma = video_sigmas[step_idx]
            audio_sigma = audio_sigmas[step_idx]
            context = torch.enable_grad if grad_enabled and step_idx == end_step_idx else torch.no_grad
            with context():
                velocity = self._predict_velocity(self.model, xt, (video_sigma, audio_sigma), condition, latent_shape)
                x0 = (xt[0] + video_sigma * velocity[0], xt[1] + audio_sigma * velocity[1])
            if step_idx == end_step_idx:
                break
            with torch.no_grad():
                noises = (torch.randn_like(x0[0], dtype=torch.float32), torch.randn_like(x0[1], dtype=torch.float32))
                xt = self._add_noise((x0[0].detach().float(), x0[1].detach().float()), noises, (video_sigmas[step_idx + 1], audio_sigmas[step_idx + 1]))
                xt = (xt[0].to(self.latent_dtype), xt[1].to(self.latent_dtype))
        return x0[0].to(self.latent_dtype), x0[1].to(self.latent_dtype)

    def _sample_renoise_sigmas(self):
        """One critic noise level: a uniform base in [renoise_sigma_min, renoise_sigma_max], flow-shifted per modality."""
        low = float(self.dmd_config.get("renoise_sigma_min", 0.02))
        high = float(self.dmd_config.get("renoise_sigma_max", 0.98))
        if not 0.0 <= low < high <= 1.0:
            raise ValueError(f"renoise sigma range must satisfy 0 <= min < max <= 1, got [{low}, {high}].")
        base = torch.empty((), device=self.model.device, dtype=torch.float32).uniform_(low, high)
        return shift_sigma(base, self.video_shift), shift_sigma(base, self.audio_shift)

    @staticmethod
    def _add_noise(latents, noises, sigmas):
        return (
            ((1.0 - sigmas[0]) * latents[0].float() + sigmas[0] * noises[0]).to(latents[0].dtype),
            ((1.0 - sigmas[1]) * latents[1].float() + sigmas[1] * noises[1]).to(latents[1].dtype),
        )
