#!/usr/bin/env python3
"""Convert a DMAD MiniMax-H3 student LoRA (Diffusers layout) to the ComfyUI MiniMax-H3 LoRA layout, losslessly.

ComfyUI's H3 transformer (`comfy/ldm`), like the original MiniMax release, fuses the q/k/v projections into one
`qkv_proj` and orders the SwiGLU hidden layer as [gate; value], whereas the Diffusers model (which the DMAD students
were trained on) has separate `to_q`/`to_k`/`to_v` and [value; gate]. The conversion is exact: nothing is merged,
resized or re-extracted, so the rank-128 update is carried in full (the fused `qkv_proj` adapter has rank 3 x 128).

Mapping (per transformer block N and token-refiner block N):

    transformer_blocks.N.attn.to_q/to_k/to_v  ->  diffusion_model.blocks.N.attn.qkv_proj
        lora_A = [A_q; A_k; A_v]  (384 x in),  lora_B = block_diag(B_q, B_k, B_v)  (3*out x 384),  alpha = 384
    transformer_blocks.N.attn.to_out.0        ->  diffusion_model.blocks.N.attn.out_proj           alpha = 128
    transformer_blocks.N.ff.net.0.proj        ->  diffusion_model.blocks.N.mlp.fc1  (lora_B halves swapped: [value; gate] -> [gate; value])
    transformer_blocks.N.ff.net.2             ->  diffusion_model.blocks.N.mlp.fc2
    token_refiner.refiner_blocks.N.*          ->  diffusion_model.token_refiner.blocks.N.*  (same four)

ComfyUI scales every adapter by alpha / rank, so alpha = rank keeps the training-time scale of 1. Load the result with
`LoraLoaderModelOnly` at strength 1.0.

Usage:
    python tools/convert_lora_comfyui.py ckpt/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
        out/dmad_minimax_h3_4step_lora_critic_comfyui.safetensors
"""

import argparse
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmad_h3.lora import lora_rank_alpha, read_lora_file  # noqa: E402

_LINEAR = {"attn.to_out.0": "attn.out_proj", "ff.net.0.proj": "mlp.fc1", "ff.net.2": "mlp.fc2"}


def _target_prefix(module):
    """Diffusers module name -> (ComfyUI block prefix, tail), e.g. `transformer_blocks.3.attn.to_q` ->
    ("diffusion_model.blocks.3", "attn.to_q")."""
    if module.startswith("transformer_blocks."):
        _, n, tail = module.split(".", 2)
        return f"diffusion_model.blocks.{n}", tail
    if module.startswith("token_refiner.refiner_blocks."):
        _, _, n, tail = module.split(".", 3)
        return f"diffusion_model.token_refiner.blocks.{n}", tail
    raise ValueError(f"unexpected LoRA module {module}")


def convert(pairs, alpha, rank):
    if alpha != rank:
        raise ValueError(f"alpha {alpha} != rank {rank}: the released students use alpha = rank; adapt the alpha entries")
    out, blocks = {}, {}
    for module, p in pairs.items():
        prefix, tail = _target_prefix(module)
        blocks.setdefault(prefix, {})[tail] = p
    for prefix, mods in sorted(blocks.items()):
        q, k, v = (mods.pop(f"attn.to_{x}") for x in "qkv")
        a = torch.cat([q["A"], k["A"], v["A"]], dim=0)  # [3r, in]
        b = torch.block_diag(q["B"], k["B"], v["B"])  # [3*out, 3r]
        out[f"{prefix}.attn.qkv_proj.lora_A.weight"] = a.contiguous()
        out[f"{prefix}.attn.qkv_proj.lora_B.weight"] = b.to(q["B"].dtype).contiguous()
        out[f"{prefix}.attn.qkv_proj.alpha"] = torch.tensor(float(a.shape[0]), dtype=q["B"].dtype)
        for tail, name in _LINEAR.items():
            p = mods.pop(tail)
            b = p["B"]
            if name == "mlp.fc1":  # Diffusers SwiGLU outputs [value; gate]; the original H3 / ComfyUI order is [gate; value]
                half = b.shape[0] // 2
                b = torch.cat([b[half:], b[:half]], dim=0)
            out[f"{prefix}.{name}.lora_A.weight"] = p["A"].contiguous()
            out[f"{prefix}.{name}.lora_B.weight"] = b.contiguous()
            out[f"{prefix}.{name}.alpha"] = torch.tensor(float(p["A"].shape[0]), dtype=b.dtype)
        if mods:
            raise ValueError(f"unconverted modules under {prefix}: {sorted(mods)}")
    return out


def verify(pairs, out):
    """The converted adapters reproduce every original weight update exactly (CPU, float32)."""
    for module, p in pairs.items():
        prefix, tail = _target_prefix(module)
        delta = (p["B"].float() @ p["A"].float())
        if tail.startswith("attn.to_") and tail != "attn.to_out.0":
            i = "qkv".index(tail[-1])
            a, b = out[f"{prefix}.attn.qkv_proj.lora_A.weight"].float(), out[f"{prefix}.attn.qkv_proj.lora_B.weight"].float()
            o, r = delta.shape[0], p["A"].shape[0]
            got = b[i * o : (i + 1) * o] @ a  # the block-diagonal structure makes this the i-th projection's update
            assert torch.equal(a[i * r : (i + 1) * r], p["A"].float())
        else:
            name = _LINEAR[tail]
            a, b = out[f"{prefix}.{name}.lora_A.weight"].float(), out[f"{prefix}.{name}.lora_B.weight"].float()
            got = b @ a
            if name == "mlp.fc1":
                half = got.shape[0] // 2
                got = torch.cat([got[half:], got[:half]], dim=0)
        if not torch.equal(got, delta):
            raise AssertionError(f"{module}: converted update differs (max |diff| {(got - delta).abs().max():.3g})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="DMAD LoRA (.safetensors, Diffusers layout)")
    p.add_argument("output", help="ComfyUI-layout LoRA to write")
    p.add_argument("--no-verify", action="store_true", help="skip the exact reconstruction check")
    args = p.parse_args()
    pairs, meta = read_lora_file(args.source)
    rank, alpha = lora_rank_alpha(pairs, meta)
    out = convert(pairs, alpha, rank)
    if not args.no_verify:
        verify(pairs, out)
    tensors = sum(1 for k in out if k.endswith(".weight"))
    metadata = {
        "format": "pt",
        "converted_layout": "comfyui_minimax_h3",
        "source": os.path.basename(args.source),
        "conversion": "to_q/to_k/to_v fused block-diagonally into attn.qkv_proj (alpha 3*rank); ff.net.0.proj lora_B "
                      "halves swapped to the [gate; value] order of mlp.fc1; alpha = rank everywhere (scale 1.0); exact, no resizing",
        "lora_rank": str(rank),
        "lora_alpha": str(alpha),
        **{k: v for k, v in meta.items() if k in ("model", "checkpoint", "sampling", "license")},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save_file(out, args.output, metadata=metadata)
    print(f"wrote {args.output}: {tensors} LoRA tensors, {len(out) - tensors} alpha entries, "
          f"{sum(t.numel() * t.element_size() for t in out.values()) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
