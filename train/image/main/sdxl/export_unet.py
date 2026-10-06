"""Fold the spectral norm of a trained SDXL generator into plain weights and save it as a UNet state dict that loads
into diffusers' UNet2DConditionModel (see README, Inference).

    python main/sdxl/export_unet.py --checkpoint_path <folder>/pytorch_model_ema.bin --output_path dmad_sdxl_4step_unet_fp16.bin
"""
import argparse

import torch

from main.sdxl.eval_sdxl import fold_spectral_norm

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
args = parser.parse_args()

state_dict = fold_spectral_norm(torch.load(args.checkpoint_path, map_location="cpu"))
torch.save({k: v.to(getattr(torch, args.dtype)) for k, v in state_dict.items()}, args.output_path)
print(f"wrote {len(state_dict)} tensors to {args.output_path}")
