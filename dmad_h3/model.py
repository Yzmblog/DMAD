"""Loading the MiniMax-H3 components (Diffusers classes) and the latent geometry read from their configs."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class H3Geometry:
    """Latent layout constants of a MiniMax-H3 checkpoint, read from `transformer/config.json` and `vae/config.json`."""

    patch_size: tuple[int, int, int]
    video_latent_channels: int
    audio_latent_channels: int
    vae_spatial_scale_factor: int

    @classmethod
    def from_model_dir(cls, model_dir):
        root = Path(model_dir).expanduser().resolve()
        transformer_cfg = json.load(open(resolve_transformer_dir(root) / "config.json"))
        vae_cfg = json.load(open(root / "vae" / "config.json"))
        return cls(
            patch_size=tuple(int(p) for p in transformer_cfg["patch_size"]),
            video_latent_channels=int(transformer_cfg["in_channels"]),
            audio_latent_channels=int(transformer_cfg["audio_in_channels"]),
            vae_spatial_scale_factor=int(math.prod(vae_cfg["spatial_downsample_factors"])),
        )


def resolve_transformer_dir(path):
    """Accept the model root (with `transformer/`) or the transformer directory itself."""
    path = Path(path).expanduser().resolve()
    if (path / "transformer" / "config.json").is_file():
        path = path / "transformer"
    elif not (path / "config.json").is_file():
        raise FileNotFoundError(f"No MiniMax-H3 transformer config below {path}")
    config = json.load(open(path / "config.json"))
    if config.get("_class_name") != "MiniMaxH3Transformer3DModel":
        raise ValueError(f"{path} is not a Diffusers MiniMaxH3Transformer3DModel checkpoint")
    return path


def load_transformer(path, dtype=torch.bfloat16):
    """The Diffusers MiniMax-H3 transformer on CPU."""
    from diffusers import MiniMaxH3Transformer3DModel

    transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        str(resolve_transformer_dir(path)), torch_dtype=dtype, local_files_only=True, low_cpu_mem_usage=True
    )
    transformer.requires_grad_(False)
    return transformer


def load_vaes(model_dir, device):
    """The video and audio VAEs in float32 on `device`."""
    from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio

    root = Path(model_dir).expanduser().resolve()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(root / "vae"), local_files_only=True).eval().to(device)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(str(root / "audio_vae"), local_files_only=True).eval().to(device)
    return vae, audio_vae
