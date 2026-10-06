"""Standalone FID evaluator for ImageNet-64 training runs.

Watches a checkpoint folder (the training --cache_dir) and evaluates every new checkpoint_model_* folder.
Each EMA of the generator stored in the checkpoint is evaluated separately:
    pytorch_model_ema.bin        -> "fid"            (primary EMA)
    pytorch_model_ema_{K}.bin    -> "fid_ema_{K}"    (extra EMAs, --generator_ema_kimg_list)
Results are written to <folder>/stats.json and logged to wandb. Protocol (as in DMD2): 50k one-step samples,
fixed seed, labels cycled over the 1000 classes, EDM Inception detector and ImageNet-64 reference statistics.

Run it on one GPU next to the training job:
    python main/edm/test_folder_edm.py --folder $CHECKPOINT_CACHE_DIR --ref_path imagenet_fid_refs_edm.npz \
        --detector_url inception-2015-12-05.pkl --wandb_project ... --wandb_name ...
"""
import argparse
import glob
import json
import os
import pickle
import re
import time

import dnnlib
import numpy as np
import scipy.linalg
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from tqdm import tqdm

from main.edm.edm_network import get_imagenet_edm_config
from third_party.edm.training.networks import EDMPrecond


def get_imagenet_config():
    base_config = {
        "img_resolution": 64,
        "img_channels": 3,
        "label_dim": 1000,
        "use_fp16": False,
        "sigma_min": 0,
        "sigma_max": float("inf"),
        "sigma_data": 0.5,
        "model_type": "DhariwalUNet",
    }
    base_config.update(get_imagenet_edm_config())
    return base_config


def fold_spectral_norm(state_dict):
    """Fold spectral-norm parametrizations (weight = gain * W / sigma(W), sigma from the stored power-iteration
    vectors u, v) back into plain conv weights, so the generator loads into a plain EDMPrecond."""
    suffix = ".parametrizations.weight.original"
    prefixes = {k[: -len(suffix)] for k in state_dict if k.endswith(suffix)}
    if not prefixes:
        return state_dict
    out = {k: v for k, v in state_dict.items() if ".parametrizations.weight." not in k}
    for pre in prefixes:
        W = state_dict[f"{pre}{suffix}"].float()
        u = state_dict[f"{pre}.parametrizations.weight.0._u"].float()
        v = state_dict[f"{pre}.parametrizations.weight.0._v"].float()
        gain = state_dict[f"{pre}.parametrizations.weight.1.gain"].float()
        sigma = torch.dot(u, torch.mv(W.flatten(1), v))
        out[f"{pre}.weight"] = (gain * W / sigma).to(state_dict[f"{pre}{suffix}"].dtype)
    return out


def create_generator(checkpoint_path, base_model=None):
    if base_model is None:
        generator = EDMPrecond(**get_imagenet_config())
        del generator.model.map_augment
        generator.model.map_augment = None
    else:
        generator = base_model
    while True:
        try:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            break
        except Exception:
            print(f"fail to load checkpoint {checkpoint_path}")
            time.sleep(1)
    print(generator.load_state_dict(fold_spectral_norm(state_dict), strict=True))
    return generator


def create_evaluator(detector_url):
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with dnnlib.util.open_url(detector_url, verbose=False) as f:
        detector_net = pickle.load(f)
    detector_net.eval()
    return detector_net, detector_kwargs, feature_dim


@torch.no_grad()
def sample(accelerator, current_model, args, model_index, log_prefix=""):
    timesteps = torch.ones(args.eval_batch_size, device=accelerator.device, dtype=torch.long)
    current_model.eval()
    all_images = []
    all_images_tensor = []
    current_index = 0
    all_labels = torch.arange(0, args.total_eval_samples * 2, device=accelerator.device, dtype=torch.long) % args.label_dim
    set_seed(args.seed + accelerator.process_index)
    while len(all_images_tensor) * args.eval_batch_size * accelerator.num_processes < args.total_eval_samples:
        noise = torch.randn(args.eval_batch_size, 3, args.resolution, args.resolution, device=accelerator.device)
        random_labels = all_labels[current_index:current_index + args.eval_batch_size]
        one_hot_labels = torch.eye(args.label_dim, device=accelerator.device)[random_labels]
        current_index += args.eval_batch_size
        eval_images = current_model(noise * args.conditioning_sigma, timesteps * args.conditioning_sigma, one_hot_labels)
        eval_images = ((eval_images + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        gathered_images = accelerator.gather(eval_images)
        all_images.append(gathered_images.cpu().numpy())
        all_images_tensor.append(gathered_images.cpu())
    all_images = np.concatenate(all_images, axis=0)[:args.total_eval_samples]
    all_images_tensor = torch.cat(all_images_tensor, dim=0)[:args.total_eval_samples]

    if accelerator.is_main_process:
        grid_size = int(args.test_visual_batch_size ** (1 / 2))
        grid = all_images[:grid_size * grid_size].reshape(grid_size, grid_size, args.resolution, args.resolution, 3)
        grid = np.swapaxes(grid, 1, 2).reshape(grid_size * args.resolution, grid_size * args.resolution, 3)
        wandb.log({f"{log_prefix}generated_image_grid": wandb.Image(grid)}, step=model_index)
    accelerator.wait_for_everyone()
    return all_images_tensor


@torch.no_grad()
def calculate_inception_stats(all_images_tensor, evaluator, accelerator, evaluator_kwargs, feature_dim, max_batch_size):
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=accelerator.device)
    sigma = torch.ones([feature_dim, feature_dim], dtype=torch.float64, device=accelerator.device)
    num_batches = ((len(all_images_tensor) - 1) // (max_batch_size * accelerator.num_processes) + 1) * accelerator.num_processes
    all_batches = torch.arange(len(all_images_tensor)).tensor_split(num_batches)
    rank_batches = all_batches[accelerator.process_index:: accelerator.num_processes]
    for i in tqdm(range(num_batches // accelerator.num_processes), unit="batch", disable=not accelerator.is_main_process):
        images = all_images_tensor[rank_batches[i]]
        features = evaluator(images.permute(0, 3, 1, 2).to(accelerator.device), **evaluator_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features
    mu = accelerator.reduce(mu)
    sigma = accelerator.reduce(sigma)
    mu /= len(all_images_tensor)
    sigma -= mu.ger(mu) * len(all_images_tensor)
    sigma /= len(all_images_tensor) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def ema_weight_files(checkpoint):
    """[(metric_name, path)] for the primary EMA and every extra EMA stored in a checkpoint folder."""
    out = []
    primary = os.path.join(checkpoint, "pytorch_model_ema.bin")
    if os.path.isfile(primary):
        out.append(("fid", primary))
    extra = []
    for path in glob.glob(os.path.join(checkpoint, "pytorch_model_ema_*.bin")):
        m = re.fullmatch(r"pytorch_model_ema_(\d+)\.bin", os.path.basename(path))
        if m:
            extra.append((int(m.group(1)), path))
    out += [(f"fid_ema_{k}", p) for k, p in sorted(extra)]
    return out


@torch.no_grad()
def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True, help="folder that receives checkpoint_model_* folders")
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_name", type=str)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--total_eval_samples", type=int, default=50000)
    parser.add_argument("--label_dim", type=int, default=1000)
    parser.add_argument("--test_visual_batch_size", type=int, default=100)
    parser.add_argument("--max_batch_size", type=int, default=128)
    parser.add_argument("--ref_path", type=str, required=True, help="reference FID statistics (.npz)")
    parser.add_argument("--detector_url", type=str, required=True, help="Inception detector pickle")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--conditioning_sigma", type=float, default=80.0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--run_once", action="store_true", help="evaluate what is in the folder now and exit")
    args = parser.parse_args()

    # Run with one process: sampling on several GPUs changes the class distribution of the 50k samples.
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        log_with="wandb",
        project_config=ProjectConfiguration(logging_dir=args.folder),
    )
    info_path = os.path.join(args.folder, "stats.json")
    overall_stats = {}
    if os.path.isfile(info_path) and not args.no_resume:
        with open(info_path, "r") as f:
            overall_stats = json.load(f)
    evaluated = set(overall_stats.keys())

    if accelerator.is_main_process:
        run = wandb.init(config=args, dir=args.folder, mode=args.wandb_mode, entity=args.wandb_entity, project=args.wandb_project)
        wandb.run.name = args.wandb_name
        print(f"wandb run dir: {run.dir}")

    evaluator, evaluator_kwargs, feature_dim = create_evaluator(args.detector_url)
    evaluator = accelerator.prepare(evaluator)
    with dnnlib.util.open_url(args.ref_path) as f:
        ref_dict = dict(np.load(f))
    best = {}
    for stats in overall_stats.values():
        for key, value in stats.items():
            best[key] = min(best.get(key, float("inf")), value)
    generator = None

    while True:
        new_checkpoints = sorted(set(glob.glob(os.path.join(args.folder, "*checkpoint_*"))) - evaluated)
        if not new_checkpoints:
            if args.run_once:
                break
            time.sleep(30)
            continue
        for checkpoint in new_checkpoints:
            model_index = int(checkpoint.rstrip("/").split("_")[-1])
            stats = {}
            for key, weights in ema_weight_files(checkpoint):
                if accelerator.is_main_process:
                    print(f"evaluating {checkpoint} [{key}]")
                generator = create_generator(weights, base_model=generator).to(accelerator.device)
                images = sample(accelerator, generator, args, model_index, log_prefix="" if key == "fid" else key[len("fid_"):] + "_")
                mu, sigma = calculate_inception_stats(images, evaluator, accelerator, evaluator_kwargs, feature_dim, args.max_batch_size)
                if accelerator.is_main_process:
                    fid = calculate_fid_from_inception_stats(mu, sigma, ref_dict["mu"], ref_dict["sigma"])
                    stats[key] = fid
                    best[key] = min(best.get(key, float("inf")), fid)
                    print(f"checkpoint {checkpoint} {key} {fid:.6f} (best {best[key]:.6f})")
                torch.cuda.empty_cache()
            if accelerator.is_main_process:
                if not stats:
                    print(f"no pytorch_model_ema*.bin in {checkpoint}; skipped")
                overall_stats[checkpoint] = stats
                wandb.log({**stats, **{f"best_{k}": v for k, v in best.items()}}, step=model_index)
                with open(info_path, "w") as f:
                    json.dump(overall_stats, f, indent=2)
            evaluated.add(checkpoint)
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    evaluate()
