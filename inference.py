#!/usr/bin/env python3
"""Text-to-audio-video generation with a DMAD student of MiniMax-H3 (4 steps by default).

Pipeline: encode the prompts with the H3 text encoder -> few-step student rollout -> decode video and audio -> mp4.
Prompt i is written to <output-dir>/<i:04d>.mp4, with its text in <i:04d>.txt. The default step rule is the
re-noise rule the students were trained with; --euler switches to the Euler rule of the official H3 pipeline.

Examples:
    python inference.py --model-dir /path/to/MiniMax-H3 --lora ckpt/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
        --prompt "A golden retriever runs along a beach at sunset, waves crashing." --output-dir outputs/demo
    python inference.py --model-dir /path/to/MiniMax-H3 --lora ckpt/minimax_h3/dmad_minimax_h3_4step_full_critic.safetensors \
        --prompt-file prompts/dmad_sweater.txt --seed 42 --output-dir outputs/dmad_sweater --offload
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmad_h3 import (  # noqa: E402
    H3Geometry,
    attach_lora,
    build_packed_sequence,
    decode_rows,
    encode_prompt,
    latent_shape_of,
    load_conditioner,
    load_transformer,
    load_vaes,
    read_lora_file,
    read_prompts,
    rollout,
    sigma_grid,
    write_mp4,
)
from dmad_h3.lora import lora_rank_alpha  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="MiniMax-H3 model directory (Diffusers layout).")
    p.add_argument("--lora", required=True, help="DMAD student LoRA (.safetensors), applied to the base transformer.")
    p.add_argument("--lora-alpha", type=float, help="Override the LoRA alpha (default: file metadata, else alpha = rank).")
    p.add_argument("--prompt", action="append", default=[], help="Prompt text; can be given several times.")
    p.add_argument("--prompt-file", help=".txt (one prompt per line) or .jsonl ({'prompt': ...} per line).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42, help="Noise seed (the same seed is used for every prompt).")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--video-shift", type=float, default=12.0)
    p.add_argument("--audio-shift", type=float, default=2.0)
    p.add_argument("--euler", action="store_true", help="Euler ODE step rule instead of the default re-noise rule.")
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--num-frames", type=int, default=124, help="17 * n + 5 frames at 24 fps (124 = 5.2 s).")
    p.add_argument("--offload", action="store_true",
                   help="Stream the transformer to the GPU one block at a time (for GPUs where the 66 GB model "
                        "plus activations do not fit).")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = list(args.prompt) + (read_prompts(args.prompt_file) if args.prompt_file else [])
    if not prompts:
        raise SystemExit("give --prompt and/or --prompt-file")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    # 1. text encoding (the 32B text encoder is released before the transformer is loaded)
    text_encoder, tokenizer, processor = load_conditioner(args.model_dir, device, torch.bfloat16)
    conditions = [encode_prompt(text_encoder, tokenizer, processor, p, device, torch.bfloat16) for p in prompts]
    del text_encoder
    torch.cuda.empty_cache()
    log(f"encoded {len(prompts)} prompts")

    # 2. student rollout
    geometry = H3Geometry.from_model_dir(args.model_dir)
    transformer = load_transformer(args.model_dir, torch.bfloat16)
    pairs, meta = read_lora_file(args.lora)
    rank, alpha = lora_rank_alpha(pairs, meta, args.lora_alpha)
    transformer = attach_lora(transformer, pairs, rank, alpha)
    log(f"LoRA attached: {len(pairs)} modules, rank {rank}, alpha {alpha:g}")
    transformer.eval()
    if args.offload:
        transformer.enable_group_offload(
            onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True,
        )
    else:
        transformer.to(device)
    shape = latent_shape_of(args.height, args.width, args.num_frames, geometry)
    video_sigmas, audio_sigmas = sigma_grid(args.steps, args.video_shift), sigma_grid(args.steps, args.audio_shift)
    rows = []
    for i, cond in enumerate(conditions):
        prompt_embeds = cond["prompt_embeds"].to(device, dtype=torch.bfloat16)
        layout = build_packed_sequence(
            cond["text_token_tags"], shape["latent_frames"], shape["latent_height"], shape["latent_width"],
            shape["audio_latents"], geometry.patch_size,
        ).to(device)
        video_rows, audio_rows = rollout(
            transformer, prompt_embeds, layout, shape, geometry, video_sigmas, audio_sigmas, args.seed, device,
            renoise=not args.euler,
        )
        # latents are kept in bf16 before decoding, as in the pipeline that produced the paper's videos
        rows.append((video_rows.to(torch.bfloat16), audio_rows.to(torch.bfloat16)))
        log(f"sampled {i + 1}/{len(prompts)}")
    del transformer
    torch.cuda.empty_cache()

    # 3. decode + mux
    vae, audio_vae = load_vaes(args.model_dir, device)
    for i, ((video_rows, audio_rows), prompt) in enumerate(zip(rows, prompts)):
        frames, wav = decode_rows(vae, audio_vae, video_rows, audio_rows, shape, geometry)
        out = os.path.join(args.output_dir, f"{i:04d}.mp4")
        write_mp4(frames, wav, out)
        with open(os.path.join(args.output_dir, f"{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        log(f"wrote {out}")
    with open(os.path.join(args.output_dir, "settings.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "prompt"}, f, indent=2)
    log(f"done; peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
