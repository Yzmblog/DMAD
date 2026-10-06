#!/usr/bin/env python3
"""Generate teacher samples (the base MiniMax-H3, no LoRA) for the teacher branch of the GAN critic.

For every prompt id in <prompt-cache>/metadata.jsonl, the base model is rolled out with H3's official
Euler (blend) sampler and the packed x0 rows are saved to <output-dir>/{id:08d}.pt:

    {"video_rows": bf16 [1, R_v, D_v], "audio_rows": bf16 [1, R_a, D_a], "meta": {...}}

The noise seed is the prompt id, so the output is deterministic and the job is resumable and shardable.
Defaults: 31 model evaluations (32-point grid), video shift 12, audio shift 3.

Usage (one process per GPU, shard over GPUs/nodes):
    python data_process/gen_teacher_data.py --model-dir /path/to/MiniMax-H3 --prompt-cache /path/to/prompt_cache \
        --output-dir /path/to/teacher_latents --shard 0/8
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lightx2v_train.model_zoo import build_model  # noqa: E402
from lightx2v_train.model_zoo.native.minimax_h3 import build_packed_sequence  # noqa: E402
from lightx2v_train.model_zoo.native.minimax_h3.sampling import latent_shape_of, rollout, sigma_grid  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "dmad_minimax_h3.yaml")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt-cache", required=True, help="Output dir of encode_prompts.py (metadata.jsonl + conditions/).")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Training config (model definition and video geometry).")
    p.add_argument("--shard", default="0/1", help="R/W: this process handles every W-th prompt starting at R.")
    p.add_argument("--steps", type=int, default=31)
    p.add_argument("--video-shift", type=float, default=12.0)
    p.add_argument("--audio-shift", type=float, default=3.0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(0)
    r, w = map(int, args.shard.split("/"))
    rows = [json.loads(l) for l in open(os.path.join(args.prompt_cache, "metadata.jsonl"), encoding="utf-8")][r::w]
    todo = [row for row in rows if not os.path.exists(os.path.join(args.output_dir, f"{int(row['id']):08d}.pt"))]
    print(f"[teacher shard {r}/{w}] {len(todo)}/{len(rows)} prompts to generate", flush=True)
    if not todo:
        return
    os.makedirs(args.output_dir, exist_ok=True)

    config = yaml.safe_load(open(args.config))
    config["model"]["pretrained_model_name_or_path"] = args.model_dir
    dmd = config["training"]["dmd"]
    model = build_model(config)
    model.load_components()
    model.transformer.to(model.device).requires_grad_(False)
    model.transformer.train()
    shape = latent_shape_of(int(dmd["height"]), int(dmd["width"]), int(dmd["num_frames"]), model)
    video_sigmas, audio_sigmas = sigma_grid(args.steps, args.video_shift), sigma_grid(args.steps, args.audio_shift)

    for row in todo:
        index = int(row["id"])
        t0 = time.time()
        cond = os.path.join(args.prompt_cache, row["condition_path"])
        if not os.path.isfile(cond):  # the released prompt cache keeps the conditions in two-character prefix buckets
            d, n = os.path.split(cond)
            cond = os.path.join(d, os.path.splitext(n)[0][-2:], n)
        payload = torch.load(cond, map_location="cpu", weights_only=False)
        condition = model.prepare_text_condition(payload["conditioning"]["positive"])
        layout = build_packed_sequence(
            condition["text_token_tags"].detach().cpu(),
            shape["latent_frames"], shape["latent_height"], shape["latent_width"], shape["audio_latents"], model.patch_size,
        ).to(model.device)
        video_rows, audio_rows = rollout(model, condition, layout, shape, video_sigmas, audio_sigmas, seed=index, renoise=False)
        meta = {"index": index, "seed": index, "steps": args.steps, "video_shift": args.video_shift, "audio_shift": args.audio_shift,
                "num_frames": shape["num_frames"], "prompt": payload.get("prompt", "")}
        out = os.path.join(args.output_dir, f"{index:08d}.pt")
        tmp = f"{out}.tmp{os.getpid()}"
        torch.save({"video_rows": video_rows.to(torch.bfloat16), "audio_rows": audio_rows.to(torch.bfloat16), "meta": meta}, tmp)
        os.replace(tmp, out)
        print(f"[teacher shard {r}/{w}] id {index}: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
