"""Build the prompt list and the prompt -> source-clip map from UltraVideo's caption table.

Reads the "Detailed Description" column of UltraVideo_short.csv, flattens each caption to one line and de-duplicates
(first occurrence order). Writes prompts.txt (one prompt per line; line i = global index i, used by
build_synthetic_dataset.py) and index_to_clipid.jsonl ({"index", "clip_ids", "urls"} per line, used by
build_real_latents.py to find the real clip of each prompt).

    python rcm/datasets/build_prompt_list.py --csv UltraVideo_short.csv --output_dir $DATA
"""
import argparse
import csv
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    csv.field_size_limit(10**9)
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h.strip().lower(): i for i, h in enumerate(header)}
        dc, cid, url = col["detailed description"], col.get("clip_id"), col.get("url")
        index_of, prompts, clips, urls = {}, [], [], []
        for row in reader:
            if len(row) <= dc:
                continue
            prompt = " ".join(row[dc].split())
            if not prompt:
                continue
            if prompt not in index_of:
                index_of[prompt] = len(prompts)
                prompts.append(prompt)
                clips.append([])
                urls.append([])
            i = index_of[prompt]
            if cid is not None and len(row) > cid:
                clips[i].append(row[cid])
            if url is not None and len(row) > url:
                urls[i].append(row[url])

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "prompts.txt"), "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    with open(os.path.join(args.output_dir, "index_to_clipid.jsonl"), "w", encoding="utf-8") as f:
        for i in range(len(prompts)):
            f.write(json.dumps({"index": i, "clip_ids": clips[i], "urls": urls[i]}, ensure_ascii=False) + "\n")
    print(f"{len(prompts)} unique prompts written to {args.output_dir}")


if __name__ == "__main__":
    main()
