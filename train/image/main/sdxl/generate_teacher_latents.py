"""Generate the SDXL teacher-sample latents (Q) used by the DMAD critic.

For every caption of the real-latents LMDB (so real and teacher samples share the caption set; row i of both LMDBs
has the same caption), SDXL-base samples a latent with 50-step Euler (timestep_spacing="trailing"), CFG 8 and zero
negative embeddings -- the sampler of DMD2's SDXL teacher-pair generator. Latents stay in the scaled VAE space, as in
the real-latents LMDB, and are stored as fp16 in the same LMDB format.

Step 1, one process per GPU (resumable; completed chunks are skipped). The noise of each batch is seeded by the global
index of its first sample, so keep --num_shards 8 and --batch_size 16 to reproduce our dataset:
    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i python main/sdxl/generate_teacher_latents.py --mode generate --shard $i --num_shards 8 \
            --model_id $SDXL --real_lmdb $DATA/sdxl_vae_latents_laion_500k_lmdb --out_dir $DATA/teacher_chunks &
    done; wait
Step 2, pack the chunks into one LMDB:
    python main/sdxl/generate_teacher_latents.py --mode merge --out_dir $DATA/teacher_chunks \
        --lmdb_path $DATA/sdxl_teacher_latents_lmdb
"""
import argparse
import glob
import os

import lmdb
import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["generate", "merge"], default="generate")
    p.add_argument("--model_id", type=str, help="local stable-diffusion-xl-base-1.0 folder")
    p.add_argument("--real_lmdb", type=str, help="real-latents LMDB; its captions are used as prompts")
    p.add_argument("--out_dir", type=str, required=True, help="folder for the per-shard chunks")
    p.add_argument("--lmdb_path", type=str, default=None, help="merge mode: output LMDB")
    p.add_argument("--num_samples", type=int, default=500000)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=512, help="samples per chunk file (resume unit)")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=8.0)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--seed", type=int, default=20260713)
    return p.parse_args()


def load_prompts(args):
    env = lmdb.open(args.real_lmdb, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        n = int(txn.get(b"latents_shape").decode().split()[0])
        prompts = [txn.get(f"prompts_{i}_data".encode()).decode() for i in range(n)]
    env.close()
    print(f"{len(prompts)} prompts from {args.real_lmdb}")
    return prompts[: args.num_samples]


def build_pipeline(args, device):
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16, use_safetensors=True)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


@torch.no_grad()
def gen_batch(pipe, prompts, args, generator):
    # negative_prompt=None with force_zeros_for_empty_prompt=True gives zero unconditional embeddings
    lat = pipe(
        prompt=prompts, height=args.resolution, width=args.resolution,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        generator=generator, output_type="latent",
    ).images
    return lat.to(torch.float16).cpu().numpy()


def chunk_path(out_dir, shard, chunk_idx):
    return os.path.join(out_dir, f"shard{shard:02d}_chunk{chunk_idx:05d}.npz")


def chunk_complete(cpath, expect_n):
    if not os.path.exists(cpath):
        return False
    try:
        with np.load(cpath) as z:
            return z["latents"].shape[0] == expect_n
    except Exception:
        return False


def run_generate(args):
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)

    prompts = load_prompts(args)
    my_indices = list(range(args.shard, len(prompts), args.num_shards))
    pipe = build_pipeline(args, device)
    n_chunks = (len(my_indices) + args.chunk_size - 1) // args.chunk_size
    for ci in range(n_chunks):
        cpath = chunk_path(args.out_dir, args.shard, ci)
        idxs = my_indices[ci * args.chunk_size:(ci + 1) * args.chunk_size]
        if chunk_complete(cpath, len(idxs)):
            print(f"skip {os.path.basename(cpath)} (complete)")
            continue
        lats, texts = [], []
        for b0 in range(0, len(idxs), args.batch_size):
            bidx = idxs[b0:b0 + args.batch_size]
            g = torch.Generator(device=device).manual_seed(args.seed + bidx[0])
            lats.append(gen_batch(pipe, [prompts[i] for i in bidx], args, g))
            texts.extend(prompts[i] for i in bidx)
        np.savez(cpath + ".tmp.npz", latents=np.concatenate(lats, axis=0),
                 prompt_idx=np.array(idxs, dtype=np.int64), prompts=np.array(texts, dtype=object))
        os.replace(cpath + ".tmp.npz", cpath)
        print(f"shard {args.shard}: wrote chunk {ci + 1}/{n_chunks}", flush=True)


def run_merge(args):
    assert args.lmdb_path, "--lmdb_path required for merge"
    chunks = sorted(glob.glob(os.path.join(args.out_dir, "shard*_chunk*.npz")))
    assert chunks, f"no chunks in {args.out_dir}"
    total, shape = 0, None
    for c in chunks:
        with np.load(c, allow_pickle=True) as z:
            total += z["latents"].shape[0]
            shape = z["latents"].shape[1:]
    print(f"{len(chunks)} chunks, {total} samples, latent shape {shape}")
    os.makedirs(args.lmdb_path, exist_ok=True)
    env = lmdb.open(args.lmdb_path, map_size=int(total * np.prod(shape) * 2 * 1.3))
    seen = np.zeros(total, dtype=bool)
    with env.begin(write=True) as txn:
        txn.put(b"latents_shape", f"{total} {' '.join(map(str, shape))}".encode())
        for c in chunks:
            with np.load(c, allow_pickle=True) as z:
                for lat, text, gi in zip(z["latents"], z["prompts"], z["prompt_idx"]):
                    gi = int(gi)  # rows are written at their global index, aligned with the real-latents LMDB
                    assert not seen[gi], f"duplicate index {gi} in {c}"
                    seen[gi] = True
                    txn.put(f"latents_{gi}_data".encode(), lat.astype(np.float16).tobytes())
                    txn.put(f"prompts_{gi}_data".encode(), str(text).encode())
    env.close()
    assert seen.all(), f"missing indices, e.g. {np.where(~seen)[0][:5]}"
    print(f"wrote {total} samples to {args.lmdb_path}")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "generate":
        run_generate(args)
    else:
        run_merge(args)
