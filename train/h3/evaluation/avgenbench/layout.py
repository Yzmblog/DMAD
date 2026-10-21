#!/usr/bin/env python3
"""Arrange `inference.py` outputs for the AVGen-Bench evaluation, and write the prompt files its judges read.

`inference.py --prompt-file train/h3/evaluation/avgenbench/prompts.jsonl` writes prompt i to <output-dir>/<i:04d>.mp4,
in the order of the jsonl. AVGen-Bench (generation/runner.py) expects <videos-dir>/<category>/<safe_filename(content)>.mp4,
and its evaluation reads the prompts from <prompts-dir>/<category>.json ([{"content", "prompt"}, ...]).

    python train/h3/evaluation/avgenbench/layout.py train/h3/evaluation/avgenbench/prompts.jsonl outputs/avgenbench_235 \
        /path/to/AVGen-Bench/generated_videos/dmad_lora_critic --prompts-dir /path/to/AVGen-Bench/prompts_h3pe

--prompts-dir writes the official prompt lists with the `prompt` field
replaced by the H3-PE rewrite, so that the judge reads the prompt the video was generated from.
"""
import argparse
import json
import os
import re
import shutil


def safe_filename(name, max_len=180):  # verbatim from AVGen-Bench generation/runner.py
    name = str(name).strip()
    name = re.sub(r"[\/\\\:\*\?\"\<\>\|\n\r\t]", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        name = "untitled"
    if len(name) > max_len:
        name = name[:max_len].rstrip()
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompts_jsonl", help="train/h3/evaluation/avgenbench/prompts.jsonl")
    p.add_argument("output_dir", help="the --output-dir given to inference.py")
    p.add_argument("videos_dir", help="destination: AVGen-Bench's --videos-dir for this method")
    p.add_argument("--prompts-dir", help="also write <category>.json files with the PE prompts here")
    p.add_argument("--copy", action="store_true", help="copy the videos instead of symlinking them")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.prompts_jsonl, encoding="utf-8") if line.strip()]
    missing = [i for i in range(len(rows)) if not os.path.isfile(os.path.join(args.output_dir, f"{i:04d}.mp4"))]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(rows)} videos missing in {args.output_dir}, e.g. {missing[:5]}")

    by_category = {}
    for i, r in enumerate(rows):
        src = os.path.abspath(os.path.join(args.output_dir, f"{i:04d}.mp4"))
        dst = os.path.join(args.videos_dir, r["category"], safe_filename(r["content"]) + ".mp4")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.remove(dst)
        if args.copy:
            shutil.copyfile(src, dst)
        else:
            os.symlink(src, dst)
        by_category.setdefault(r["category"], []).append({"content": r["content"], "prompt": r["prompt"]})

    if args.prompts_dir:
        os.makedirs(args.prompts_dir, exist_ok=True)
        for category, items in by_category.items():
            with open(os.path.join(args.prompts_dir, f"{category}.json"), "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"{len(rows)} videos -> {args.videos_dir} ({len(by_category)} categories)"
          + (f"; prompt lists -> {args.prompts_dir}" if args.prompts_dir else ""))


if __name__ == "__main__":
    main()
