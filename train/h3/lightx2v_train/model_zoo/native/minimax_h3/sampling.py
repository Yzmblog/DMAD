"""Few-step sampling, VAE decoding and mp4 muxing for MiniMax-H3 students."""

import os
import subprocess
import tempfile

import numpy as np
import torch

from .packing import audio_latent_num_frames, build_row_timesteps, video_latent_num_frames

VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32000
# The H3 video VAE works in ImageNet-normalized pixel space.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def shift_sigma(sigma, shift):
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def sigma_grid(steps, shift):
    """Shifted linear grid from 1 to 0 with `steps` model evaluations (steps + 1 points)."""
    return shift_sigma(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32), shift)


def latent_shape_of(height, width, num_frames, model):
    latent_frames = video_latent_num_frames(num_frames)
    latent_height = height // model.vae_spatial_scale_factor
    latent_width = width // model.vae_spatial_scale_factor
    patch_t, patch_h, patch_w = model.patch_size
    video_rows = latent_frames * (latent_height // patch_h) * (latent_width // patch_w)
    video_dim = model.video_latent_channels * patch_t * patch_h * patch_w
    audio_latents = audio_latent_num_frames(num_frames)
    return {
        "num_frames": num_frames,
        "latent_frames": latent_frames,
        "latent_height": latent_height,
        "latent_width": latent_width,
        "audio_latents": audio_latents,
        "video_tokens": (1, video_rows, video_dim),
        "audio_tokens": (1, audio_latents * 2, model.audio_latent_channels),
    }


def draw_init_noise(shape, model, seed):
    """Initial noise from one CPU generator: video latent first, then audio rows. The same generator
    is returned and later supplies the fresh noise of the re-noise step rule."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_v = torch.randn(
        (1, model.video_latent_channels, shape["latent_frames"], shape["latent_height"], shape["latent_width"]),
        generator=g,
        dtype=torch.float32,
    )
    noise_a = torch.randn((shape["audio_latents"] * 2, model.audio_latent_channels), generator=g, dtype=torch.float32)
    xt = (patchify_video_latents(noise_v.to(model.device), model.patch_size)[None].float(), noise_a.to(model.device)[None].float())
    return xt, g


@torch.no_grad()
def rollout(model, condition, layout, shape, video_sigmas, audio_sigmas, seed, renoise=True):
    """Run len(video_sigmas) - 1 student evaluations and return the final (video_rows, audio_rows) x0 on CPU.

    renoise=True : DMD2 multistep rule (denoise, then fresh noise at the next sigma) -- what the student is trained with.
    renoise=False: Euler ODE step (H3's official blend rule).
    """
    dev = model.device
    xt, g = draw_init_noise(shape, model, seed)
    x0 = None
    for step in range(len(video_sigmas) - 1):
        v_s, a_s = video_sigmas[step].to(dev), audio_sigmas[step].to(dev)
        ts, ti = build_row_timesteps(layout, v_s, a_s)
        with model.transformer_forward_context():
            vel = model.denoiser_module()(
                hidden_states=xt[0],
                audio_hidden_states=xt[1],
                encoder_hidden_states=condition["prompt_embeds"],
                timestep=ts.to(dev),
                timestep_indices=ti.to(dev),
                token_tags=layout.token_tags,
                position_ids=layout.position_ids,
                video_indices=layout.video_indices,
                audio_indices=layout.audio_indices,
                text_indices=layout.text_indices,
                return_dict=False,
            )
        x0 = (xt[0] + v_s * vel[0].float(), xt[1] + a_s * vel[1].float())
        nv, na = video_sigmas[step + 1].to(dev), audio_sigmas[step + 1].to(dev)
        if renoise:
            ev = torch.randn(x0[0].shape, generator=g, dtype=torch.float32).to(dev)
            ea = torch.randn(x0[1].shape, generator=g, dtype=torch.float32).to(dev)
            xt = ((1.0 - nv) * x0[0] + nv * ev, (1.0 - na) * x0[1] + na * ea)
        else:
            rv, ra = nv / v_s, na / a_s
            xt = (rv * xt[0].float() + (1.0 - rv) * x0[0].float(), ra * xt[1].float() + (1.0 - ra) * x0[1].float())
    return x0[0].float().cpu(), x0[1].float().cpu()


def load_vaes(model_dir, device):
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    vae = AutoencoderKLMiniMaxH3.from_pretrained(f"{model_dir}/vae").eval().to(device)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(f"{model_dir}/audio_vae").eval().to(device)
    return vae, audio_vae


@torch.no_grad()
def decode_rows(vae, audio_vae, video_rows, audio_rows, shape, patch_size, video_channels, audio_channels):
    """Packed latent rows -> (uint8 frames [T, H, W, 3], float32 waveform [N, 2])."""
    device = next(vae.parameters()).device
    vm = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    vs = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    am = torch.tensor(audio_vae.config.latents_mean, device=device).view(1, -1, 1)
    as_ = torch.tensor(audio_vae.config.latents_std, device=device).view(1, -1, 1)
    _, ph, pw = patch_size
    t = shape["latent_frames"]
    hh, ww = shape["latent_height"] // ph, shape["latent_width"] // pw
    z = video_rows.float().to(device).reshape(t, hh, ww, video_channels, ph, pw)
    z = z.permute(3, 0, 1, 4, 2, 5).reshape(1, video_channels, t, hh * ph, ww * pw)
    px = vae.decode(z * vs + vm).sample
    pm = torch.tensor(PIXEL_MEAN, device=px.device).view(3, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=px.device).view(3, 1, 1, 1)
    frames = ((px[0].float() * ps + pm).clamp(0, 1) * 255.0).round().byte().cpu().numpy().transpose(1, 2, 3, 0)
    za = audio_rows.float().to(device).reshape(1, 2, -1, audio_channels)
    za = za[0].permute(0, 2, 1) * as_ + am
    wav = audio_vae.decode(za).sample[:, 0].float().clamp(-1, 1).cpu().numpy().T.astype(np.float32)
    return frames, wav


def write_mp4(frames, wav, out_path):
    """Mux rgb24 frames (24 fps) and 32 kHz stereo audio into an H.264/AAC mp4 tagged bt709."""
    from imageio_ffmpeg import get_ffmpeg_exe

    h_px, w_px = frames.shape[1], frames.shape[2]
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        vraw, araw = os.path.join(tmp, "video.raw"), os.path.join(tmp, "audio.raw")
        with open(vraw, "wb") as f:
            f.write(frames.tobytes())
        with open(araw, "wb") as f:
            f.write(wav.tobytes())
        subprocess.run(
            [get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w_px}x{h_px}", "-r", str(VIDEO_FPS), "-i", vraw,
             "-f", "f32le", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "2", "-i", araw,
             "-vf", "scale=in_range=full:out_color_matrix=bt709:out_range=tv",
             "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
             "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-shortest", out_path],
            check=True,
        )
