#!/usr/bin/env python3
"""Encode prompts with the MiniMax-H3 text encoder (Qwen3-VL, hidden layer 50) into a prompt cache.

Output layout (read by the `latent_dataset` loader and by the trainer's caption pairing):
    <output-dir>/conditions/condition_{index:08d}.pt
    <output-dir>/metadata.jsonl

Usage:
    python data_process/encode_prompts.py prompts.txt --model-path /path/to/MiniMax-H3 --output-dir /path/to/prompt_cache \
        [--index-file indices.txt] [--shard R/W]
    # after all shards finish:
    python data_process/encode_prompts.py prompts.txt --output-dir /path/to/prompt_cache --merge-shards
"""

import argparse
import json
import os
from pathlib import Path

import torch

TEXT_ENCODER_LAYER = 50
TEXT_TAG = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompts", help="UTF-8 .txt (one prompt per line) or .jsonl (one {'prompt': ...} object per line).")
    parser.add_argument("--model-path", help="MiniMax-H3 model directory (with text_encoder/, tokenizer/, processor/).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--index-file", help="Encode only these prompt indices (one per line), keeping their original numbering.")
    parser.add_argument("--shard", default=None, help="R/W: process every W-th selected prompt starting at R (multi-GPU sharding).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--merge-shards", action="store_true", help="Merge metadata_shard*.jsonl into metadata.jsonl and exit.")
    return parser.parse_args()


def read_prompts(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        if str(path).endswith(".jsonl"):
            prompts = [json.loads(line)["prompt"] for line in handle if line.strip()]
        else:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise RuntimeError(f"No non-empty prompts found in {path}.")
    return prompts


def load_conditioner(model_path, device, dtype):
    from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    root = Path(model_path).expanduser().resolve()
    for sub in ("text_encoder", "tokenizer", "processor"):
        if not (root / sub).exists():
            raise FileNotFoundError(f"Missing MiniMax-H3 component: {root / sub}")
    load_kwargs = {"dtype": dtype, "local_files_only": True}
    if str(device) != "cpu":
        load_kwargs["device_map"] = {"": str(device)}
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(str(root / "text_encoder"), **load_kwargs)
    text_encoder.requires_grad_(False).eval()
    tokenizer = Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer"), local_files_only=True)
    processor = Qwen3VLProcessor.from_pretrained(str(root / "processor"), local_files_only=True)
    return text_encoder, tokenizer, processor


@torch.inference_mode()
def encode_prompt(text_encoder, tokenizer, processor, prompt, device, output_dtype):
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not token_ids:
        raise ValueError("The tokenizer produced no tokens for a non-empty prompt.")
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mm_token_type_ids = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long, device=device)
    outputs = text_encoder.model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_token_type_ids,
        pixel_values=None,
        image_grid_thw=None,
        use_cache=False,
        output_hidden_states=True,
    )
    prompt_embeds = outputs.hidden_states[TEXT_ENCODER_LAYER].to(dtype=output_dtype).cpu()
    text_token_tags = torch.full((len(token_ids),), TEXT_TAG, dtype=torch.long)
    return {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}


def write_jsonl(path, rows):
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def merge_shards(output_dir):
    rows = [json.loads(l) for f in sorted(output_dir.glob("metadata_shard*.jsonl")) for l in open(f, encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate ids across shards")
    write_jsonl(output_dir / "metadata.jsonl", rows)
    print(f"Merged {len(rows)} samples into {output_dir / 'metadata.jsonl'}", flush=True)


def main():
    args = parse_args()
    if args.merge_shards:
        merge_shards(Path(args.output_dir).expanduser().resolve())
        return
    if not args.model_path:
        raise SystemExit("--model-path is required for encoding")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    indexed = list(enumerate(read_prompts(args.prompts)))
    if args.index_file:
        by_idx = dict(indexed)
        indexed = [(i, by_idx[i]) for i in sorted({int(l) for l in open(args.index_file) if l.strip()})]
    if args.shard:
        r, w = map(int, args.shard.split("/"))
        indexed = indexed[r::w]

    output_dir = Path(args.output_dir).expanduser().resolve()
    (output_dir / "conditions").mkdir(parents=True, exist_ok=True)
    text_encoder, tokenizer, processor = load_conditioner(args.model_path, args.device, dtype)

    rows = []
    for n, (index, prompt) in enumerate(indexed):
        relative_path = Path("conditions") / f"condition_{index:08d}.pt"
        output_path = output_dir / relative_path
        if args.overwrite or not output_path.is_file():
            condition = encode_prompt(text_encoder, tokenizer, processor, prompt, args.device, dtype)
            torch.save({"conditioning": {"positive": condition}, "prompt": prompt, "source_index": index}, output_path)
        rows.append({"id": index, "caption": prompt, "condition_path": str(relative_path)})
        if (n + 1) % 50 == 0 or n + 1 == len(indexed):
            print(f"[{n + 1}/{len(indexed)}] {output_path}", flush=True)

    # One metadata file per shard; run --merge-shards once all shards are done.
    name = "metadata.jsonl" if not args.shard else f"metadata_shard{args.shard.replace('/', 'of')}.jsonl"
    write_jsonl(output_dir / name, rows)
    print(f"Wrote {len(rows)} samples to {output_dir / name}", flush=True)


if __name__ == "__main__":
    main()
