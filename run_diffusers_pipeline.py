#!/usr/bin/env python3
"""Run a DMAD student through the official Diffusers MiniMax-H3 pipeline (`MiniMaxH3ModularPipeline`, t2va).

This is for users who already run MiniMax-H3 with Diffusers: attach the DMAD LoRA to the transformer, set 4 steps
and the student's time shifts, and keep everything else. The pipeline uses the Euler step rule,
not the re-noise rule the students were trained with; the paper's videos come from inference.py. Components are
offloaded to the CPU while idle (Diffusers' ComponentsManager), so one 80 GB GPU is enough. Each prompt is written
to <output-dir>/<i:04d>.mp4.

    python run_diffusers_pipeline.py --model-dir /path/to/MiniMax-H3 --lora ckpt/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
        --prompt-file prompts/dmad_sweater.txt --seed 42 --output-dir outputs/diffusers
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dmad_h3 import attach_lora, load_transformer, read_lora_file, read_prompts, write_mp4  # noqa: E402
from dmad_h3.lora import lora_rank_alpha  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="MiniMax-H3 model directory (Diffusers layout).")
    p.add_argument("--lora", required=True, help="DMAD student LoRA (.safetensors), attached to the base transformer with PEFT.")
    p.add_argument("--lora-alpha", type=float)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--prompt-file")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=4, help="Model evaluations (the scheduler gets steps + 1 grid points).")
    p.add_argument("--video-shift", type=float, default=12.0)
    p.add_argument("--audio-shift", type=float, default=2.0)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--num-frames", type=int, default=124)
    p.add_argument("--offload", action="store_true",
                   help="Also stream the transformer to the GPU one block at a time (components are always offloaded "
                        "to the CPU while idle).")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = list(args.prompt) + (read_prompts(args.prompt_file) if args.prompt_file else [])
    if not prompts:
        raise SystemExit("give --prompt and/or --prompt-file")
    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from diffusers import ComponentsManager, ModularPipeline

    t0 = time.time()
    device = torch.device("cuda")
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(args.model_dir, components_manager=manager, local_files_only=True)
    transformer = load_transformer(args.model_dir, torch.bfloat16)
    pairs, meta = read_lora_file(args.lora)
    rank, alpha = lora_rank_alpha(pairs, meta, args.lora_alpha)
    transformer = attach_lora(transformer, pairs, rank, alpha)
    transformer.eval()
    pipe.update_components(transformer=transformer)
    # modular_model_index.json points every component at the Hub id, so name the local directory explicitly
    # (text encoder in bf16, VAEs in float32 as in inference.py)
    pipe.load_components(
        names=["text_encoder", "tokenizer", "processor", "vae", "audio_vae", "scheduler", "audio_scheduler"],
        pretrained_model_name_or_path=os.path.abspath(args.model_dir), local_files_only=True,
        dtype={"text_encoder": torch.bfloat16},
    )
    if args.offload:
        pipe.transformer.enable_group_offload(
            onload_device=device, offload_device=torch.device("cpu"), offload_type="block_level",
            num_blocks_per_group=1, use_stream=True, low_cpu_mem_usage=True,
        )
    # each component moves to the GPU when it runs; the text encoder (64 GB) and transformer (66 GB) never share it
    manager.enable_auto_cpu_offload(device=device, memory_reserve_margin="8GB")
    pipe.scheduler.set_shift(args.video_shift)
    pipe.audio_scheduler.set_shift(args.audio_shift)
    print(f"[{time.time() - t0:5.0f}s] pipeline ready", flush=True)

    for i, prompt in enumerate(prompts):
        with torch.inference_mode():
            result = pipe(
                prompt=prompt, height=args.height, width=args.width, num_frames=args.num_frames,
                num_inference_steps=args.steps + 1, generator=torch.Generator().manual_seed(args.seed),
                output_type="np", output=["videos", "audio", "sampling_rate"],
            )
        videos, audio, sr = result["videos"], result["audio"], int(result["sampling_rate"])
        frames = np.asarray(videos)
        frames = frames[0] if frames.ndim == 5 else frames
        if frames.dtype != np.uint8:
            frames = (np.clip(frames, 0, 1) * 255.0).round().astype(np.uint8)
        wav = audio.detach().cpu().float().numpy() if torch.is_tensor(audio) else np.asarray(audio, dtype=np.float32)
        wav = wav.reshape(-1, wav.shape[-1]) if wav.ndim > 2 else wav
        wav = wav.T if wav.shape[0] == 2 and wav.shape[1] != 2 else wav
        out = os.path.join(args.output_dir, f"{i:04d}.mp4")
        write_mp4(np.ascontiguousarray(frames), np.ascontiguousarray(wav, dtype=np.float32), out, sample_rate=sr)
        with open(os.path.join(args.output_dir, f"{i:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"[{time.time() - t0:5.0f}s] wrote {out} ({frames.shape[0]} frames, {wav.shape[0] / sr:.2f} s audio)", flush=True)
    with open(os.path.join(args.output_dir, "settings.json"), "w") as f:
        json.dump({k: v for k, v in vars(args).items() if k != "prompt"}, f, indent=2)
    print(f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
