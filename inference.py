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
from dmad_h3.lowmem import LowMemTextEncoder, LowMemTransformer  # noqa: E402
from dmad_h3.packing import build_row_timesteps  # noqa: E402
from dmad_h3.text_encoder import TEXT_ENCODER_LAYER  # noqa: E402


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
    p.add_argument("--low-vram", action="store_true",
                   help="For consumer GPUs (24 GB): stream the text encoder and the transformer from the checkpoint files one "
                        "layer at a time and keep only the VAE decoders resident. The output is identical to the "
                        "default path.")
    p.add_argument("--weights-in-ram", action="store_true",
                   help="With --low-vram: keep a pinned host-memory copy of the streamed transformer weights (35 GB) "
                        "instead of re-reading them from disk at every step.")
    p.add_argument("--weights-int8", action="store_true",
                   help="With --low-vram: store the streamed transformer weights as int8 (per-row scales, dequantized "
                        "to bf16 on the GPU before use), halving host memory / disk traffic. The samples differ from "
                        "the exact bf16 weights (default) like any numerics change does; see the README.")
    p.add_argument("--weights-cache", metavar="DIR",
                   help="With --weights-int8 and without --weights-in-ram: where the int8 copy of the weights is "
                        "written once (18 GB; default ~/.cache/dmad_h3/<checkpoint>/int8).")
    p.add_argument("--vae-dtype", choices=["fp32", "bf16"], default="fp32",
                   help="Precision of the video VAE decode (fp32 as released; bf16 is about twice as fast and "
                        "halves its memory, at ~42 dB PSNR to the fp32 decode). The audio VAE always runs in fp32.")
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

    def host_memory():
        """Anonymous resident memory of this process (Linux), i.e. without the memory-mapped checkpoint pages."""
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        return f", host memory {int(line.split()[1]) / 2**20:.1f} GiB"
        except OSError:
            pass
        return ""

    def peak(stage):
        log(f"{stage}: peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB{host_memory()}")
        torch.cuda.reset_peak_memory_stats()

    # 1. text encoding (the 32B text encoder is released before the transformer is loaded)
    if args.low_vram:
        encoder = LowMemTextEncoder(args.model_dir, device, TEXT_ENCODER_LAYER, torch.bfloat16, log=log)
        conditions = [encoder.encode(p) for p in prompts]
        encoder.close()
    else:
        text_encoder, tokenizer, processor = load_conditioner(args.model_dir, device, torch.bfloat16)
        conditions = [encode_prompt(text_encoder, tokenizer, processor, p, device, torch.bfloat16) for p in prompts]
        del text_encoder
        torch.cuda.empty_cache()
    log(f"encoded {len(prompts)} prompts")
    peak("text encoding")

    # 2. student rollout
    geometry = H3Geometry.from_model_dir(args.model_dir)
    shape = latent_shape_of(args.height, args.width, args.num_frames, geometry)
    video_sigmas, audio_sigmas = sigma_grid(args.steps, args.video_shift), sigma_grid(args.steps, args.audio_shift)
    layouts = [
        build_packed_sequence(
            cond["text_token_tags"], shape["latent_frames"], shape["latent_height"], shape["latent_width"],
            shape["audio_latents"], geometry.patch_size,
        )
        for cond in conditions
    ]
    pairs, meta = read_lora_file(args.lora)
    rank, alpha = lora_rank_alpha(pairs, meta, args.lora_alpha)
    streamed = None
    if args.low_vram:
        # the distinct noise levels of every step are known up front, so the AdaLN projections can be tabulated
        timesteps = [build_row_timesteps(layouts[0], video_sigmas[s], audio_sigmas[s])[0] for s in range(args.steps)]
        streamed = LowMemTransformer(args.model_dir, pairs, rank, alpha, timesteps, device, torch.bfloat16,
                                     cache_in_ram=args.weights_in_ram, codec="int8" if args.weights_int8 else None,
                                     cache_dir=args.weights_cache, log=log)
        transformer = streamed.model
        log(f"LoRA attached: {len(pairs)} modules, rank {rank}, alpha {alpha:g}; streaming "
            f"{streamed.streamer.bytes_per_pass / 2**30:.1f} GiB of weights per model evaluation{host_memory()}")
    else:
        transformer = load_transformer(args.model_dir, torch.bfloat16)
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
    rows = []
    for i, (cond, layout) in enumerate(zip(conditions, layouts)):
        prompt_embeds = cond["prompt_embeds"].to(device, dtype=torch.bfloat16)
        layout = layout.to(device)
        video_rows, audio_rows = rollout(
            transformer, prompt_embeds, layout, shape, geometry, video_sigmas, audio_sigmas, args.seed, device,
            renoise=not args.euler,
        )
        # latents are kept in bf16 before decoding, as in the pipeline that produced the paper's videos
        rows.append((video_rows.to(torch.bfloat16), audio_rows.to(torch.bfloat16)))
        log(f"sampled {i + 1}/{len(prompts)}")
    if streamed is not None:
        streamed.close()
    del transformer
    torch.cuda.empty_cache()
    peak("sampling")

    # 3. decode + mux
    vae_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.vae_dtype]
    vae, audio_vae = load_vaes(args.model_dir, device, decoder_only=args.low_vram, dtype=vae_dtype)
    for i, ((video_rows, audio_rows), prompt) in enumerate(zip(rows, prompts)):
        frames, wav = decode_rows(vae, audio_vae, video_rows, audio_rows, shape, geometry)
        out = os.path.join(args.output_dir, f"{i:04d}.mp4")
        write_mp4(frames, wav, out)
        with open(os.path.join(args.output_dir, f"{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        log(f"wrote {out}")
    with open(os.path.join(args.output_dir, "settings.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "prompt"}, f, indent=2)
    peak("decoding")
    log("done")


if __name__ == "__main__":
    main()
