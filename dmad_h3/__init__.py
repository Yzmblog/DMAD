"""Few-step text-to-audio-video inference with DMAD students of MiniMax-H3."""

from .lora import attach_lora, read_lora_file
from .model import H3Geometry, load_transformer, load_vaes
from .packing import build_packed_sequence, build_row_timesteps
from .sampling import decode_rows, latent_shape_of, rollout, sigma_grid, write_mp4
from .text_encoder import encode_prompt, load_conditioner, read_prompts

__all__ = [
    "H3Geometry",
    "attach_lora",
    "build_packed_sequence",
    "build_row_timesteps",
    "decode_rows",
    "encode_prompt",
    "latent_shape_of",
    "load_conditioner",
    "load_transformer",
    "load_vaes",
    "read_lora_file",
    "read_prompts",
    "rollout",
    "sigma_grid",
    "write_mp4",
]
