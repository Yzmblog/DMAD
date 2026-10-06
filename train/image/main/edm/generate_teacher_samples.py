"""Generate the teacher-sample LMDB (the Q class of the critic) with the EDM ImageNet-64 teacher.

Sampler: EDM stochastic Heun (Karras et al. 2022, Alg. 2), 256 steps, S_churn=40, S_min=0.05, S_max=50,
S_noise=1.003. Sample i uses seed (--seed + i) and class label i mod 1000. The script also reports the FID of the
generated samples against the ImageNet-64 reference statistics as a sanity check.

    torchrun --nproc_per_node=8 main/edm/generate_teacher_samples.py --model_path edm-imagenet-64x64-cond-adm.pkl \
        --ref_path imagenet_fid_refs_edm.npz --detector_url inception-2015-12-05.pkl \
        --output_dir $CHECKPOINT_PATH --run_name edm_teacher_samples --total_eval_samples 1000000
    # -> $CHECKPOINT_PATH/edm_teacher_samples_lmdb  (pass it as --teacher_image_path)
"""
import argparse
import json
import os
import pickle
import time

import dnnlib
import lmdb
import numpy as np
import scipy
import torch
import torch.distributed as dist
from tqdm import tqdm


def setup_dist():
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size(), torch.device("cuda", local_rank)


def is_rank0(rank):
    return rank == 0


def load_pickle(path):
    with dnnlib.util.open_url(path, verbose=is_rank0(dist.get_rank()) if dist.is_initialized() else True) as f:
        return pickle.load(f)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, shape, **kwargs):
        assert shape[0] == len(self.generators)
        return torch.stack([torch.randn(shape[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, x):
        return self.randn(x.shape, dtype=x.dtype, layout=x.layout, device=x.device)


def get_t_steps(net, args, device):
    sigma_min = max(float(args.sigma_min), float(net.sigma_min))
    sigma_max = min(float(args.sigma_max), float(net.sigma_max))
    step_indices = torch.arange(args.steps, dtype=torch.float64, device=device)
    sigmas = (
        sigma_max ** (1 / args.rho)
        + step_indices / (args.steps - 1) * (sigma_min ** (1 / args.rho) - sigma_max ** (1 / args.rho))
    ) ** args.rho
    return torch.cat([net.round_sigma(sigmas), torch.zeros_like(sigmas[:1])])


@torch.no_grad()
def edm_sampler(net, latents, class_labels, rnd, args):
    t_steps = get_t_steps(net, args, latents.device)
    x_next = latents.to(torch.float64) * t_steps[0]

    for step_index, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next
        gamma = min(args.S_churn / args.steps, np.sqrt(2) - 1) if args.S_min <= t_cur <= args.S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * args.S_noise * rnd.randn_like(x_cur)

        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if step_index < args.steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def calculate_fid(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def write_json(path, data):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def save_lmdb(lmdb_path, shard_paths, total_samples, resolution, map_size):
    os.makedirs(lmdb_path, exist_ok=True)
    env = lmdb.open(lmdb_path, map_size=map_size)
    with env.begin(write=True) as txn:
        for shard_path in shard_paths:
            shard = np.load(shard_path)
            for index, image, label in zip(shard["indices"], shard["images"], shard["labels"]):
                txn.put(f"images_{int(index)}_data".encode(), image.tobytes())
                txn.put(f"labels_{int(index)}_data".encode(), np.asarray(label, dtype=np.int64).tobytes())
        txn.put(b"images_shape", f"{total_samples} 3 {resolution} {resolution}".encode())
        txn.put(b"labels_shape", f"{total_samples}".encode())
    env.sync()
    env.close()


@torch.no_grad()
def generate_local_shard(net, detector, detector_kwargs, feature_dim, args, rank, world_size, device):
    local_indices = list(range(rank, args.total_eval_samples, world_size))
    eye = torch.eye(args.label_dim, device=device)

    feat_sum = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    feat_outer = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    image_sum = torch.zeros([], dtype=torch.float64, device=device)
    image_sum_sq = torch.zeros([], dtype=torch.float64, device=device)
    count = torch.zeros([], dtype=torch.float64, device=device)

    shard_images = []
    shard_labels = []
    shard_indices = []
    iterator = range(0, len(local_indices), args.eval_batch_size)
    if is_rank0(rank):
        total_batches = (len(local_indices) - 1) // args.eval_batch_size + 1
        iterator = tqdm(iterator, total=total_batches, desc=args.run_name)

    for start in iterator:
        indices = local_indices[start:start + args.eval_batch_size]
        label_ids = torch.tensor([idx % args.label_dim for idx in indices], dtype=torch.long, device=device)
        rnd = StackedRandomGenerator(device, [args.seed + idx for idx in indices])
        latents = rnd.randn([len(indices), 3, args.resolution, args.resolution], device=device)
        images = edm_sampler(net, latents, eye[label_ids], rnd, args)
        images = ((images * 127.5 + 128).clip(0, 255)).to(torch.uint8)

        features = detector(images, **detector_kwargs).to(torch.float64)
        feat_sum += features.sum(0)
        feat_outer += features.T @ features

        image_values = images.to(torch.float64)
        image_sum += image_values.sum()
        image_sum_sq += (image_values * image_values).sum()
        count += len(indices)

        shard_indices.append(np.asarray(indices, dtype=np.int64))
        shard_labels.append(label_ids.cpu().numpy().astype(np.int64))
        shard_images.append(images.cpu().numpy())

    shard_path = os.path.join(args.output_dir, "shards", f"{args.run_name}_rank{rank:03d}.npz")
    np.savez(
        shard_path,
        indices=np.concatenate(shard_indices, axis=0),
        labels=np.concatenate(shard_labels, axis=0),
        images=np.concatenate(shard_images, axis=0),
    )
    return count, image_sum, image_sum_sq, feat_sum, feat_outer, shard_path


def reduce_stats(stats, world_size):
    if world_size == 1:
        return stats
    reduced = []
    for item in stats:
        dist.all_reduce(item, op=dist.ReduceOp.SUM)
        reduced.append(item)
    return tuple(reduced)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--ref_path", type=str, required=True)
    parser.add_argument("--detector_url", type=str, required=True)
    parser.add_argument("--total_eval_samples", type=int, default=1000000)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label_dim", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--S_churn", type=float, default=40.0)
    parser.add_argument("--S_min", type=float, default=0.05)
    parser.add_argument("--S_max", type=float, default=50.0)
    parser.add_argument("--S_noise", type=float, default=1.003)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--lmdb_map_size_gb", type=float, default=None, help="default: 1.5x the image bytes")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep_shards", action="store_true")
    args = parser.parse_args()

    if args.steps < 2:
        raise ValueError("--steps must be >= 2")

    rank, world_size, device = setup_dist()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if is_rank0(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "shards"), exist_ok=True)
    if world_size > 1:
        dist.barrier()

    result_path = os.path.join(args.output_dir, f"{args.run_name}.json")
    lmdb_path = os.path.join(args.output_dir, f"{args.run_name}_lmdb")
    if os.path.exists(result_path) and os.path.isdir(lmdb_path) and not args.overwrite:
        if is_rank0(rank):
            print(f"Skip existing result: {result_path}")
        return

    start_time = time.time()
    net_obj = load_pickle(args.model_path)
    net = net_obj["ema"] if isinstance(net_obj, dict) and "ema" in net_obj else net_obj
    net = net.to(device)
    net.eval()
    detector = load_pickle(args.detector_url).to(device)
    detector.eval()
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    ref = np.load(args.ref_path)

    stats = generate_local_shard(net, detector, detector_kwargs, feature_dim, args, rank, world_size, device)
    count, image_sum, image_sum_sq, feat_sum, feat_outer, _ = stats
    count, image_sum, image_sum_sq, feat_sum, feat_outer = reduce_stats(
        (count, image_sum, image_sum_sq, feat_sum, feat_outer),
        world_size,
    )

    if world_size > 1:
        dist.barrier()

    if is_rank0(rank):
        total_count = int(count.item())
        mu = feat_sum / total_count
        sigma = (feat_outer - mu.ger(mu) * total_count) / (total_count - 1)
        fid = calculate_fid(mu.cpu().numpy(), sigma.cpu().numpy(), ref["mu"], ref["sigma"])
        num_pixels = total_count * 3 * args.resolution * args.resolution
        image_mean = (image_sum / num_pixels).item()
        image_var = (image_sum_sq / num_pixels - image_mean ** 2).clamp_min(0).item()

        shard_paths = [os.path.join(args.output_dir, "shards", f"{args.run_name}_rank{r:03d}.npz") for r in range(world_size)]
        save_lmdb(
            lmdb_path,
            shard_paths,
            args.total_eval_samples,
            args.resolution,
            int(args.lmdb_map_size_gb * 1024 ** 3) if args.lmdb_map_size_gb else int(1.5 * args.total_eval_samples * 3 * args.resolution ** 2) + (1 << 30),
        )
        row = {
            "run_name": args.run_name,
            "fid": fid,
            "total_eval_samples": args.total_eval_samples,
            "actual_total_count": total_count,
            "label_dim": args.label_dim,
            "samples_per_class": args.total_eval_samples // args.label_dim,
            "balanced_labels": args.total_eval_samples % args.label_dim == 0,
            "image_shape": [args.total_eval_samples, 3, args.resolution, args.resolution],
            "labels_shape": [args.total_eval_samples],
            "lmdb_path": lmdb_path,
            "world_size": world_size,
            "eval_batch_size": args.eval_batch_size,
            "model_path": args.model_path,
            "ref_path": args.ref_path,
            "detector_url": args.detector_url,
            "steps": args.steps,
            "sigma_min": args.sigma_min,
            "sigma_max": args.sigma_max,
            "rho": args.rho,
            "S_churn": args.S_churn,
            "S_min": args.S_min,
            "S_max": args.S_max,
            "S_noise": args.S_noise,
            "seed": args.seed,
            "elapsed_seconds": time.time() - start_time,
            "image_mean": image_mean,
            "image_std": image_var ** 0.5,
        }
        write_json(result_path, row)
        print(json.dumps(row, indent=2), flush=True)

        if not args.keep_shards:
            for shard_path in shard_paths:
                os.remove(shard_path)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
