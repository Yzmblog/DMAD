"""Convert a DMAD training checkpoint (DCP) into an inference .pth: take the EMA generator (net_ema.*), fold the
frozen-gain spectral norm into plain weights, and save bf16 weights under net.* keys: the format of the released
.pth files, read by evaluation/vbench_text2video/sample_videos.py.

A spectral-normed Linear is stored as W (parametrizations.weight.original), power-iteration vectors u, v and a frozen
gain g; the folded weight is g * W / sigma with sigma = u^T W v after one power iteration from the stored u.

    python scripts/dcp_to_pth_dmad.py --dcp_checkpoint_dir <run>/checkpoints/iter_000017000/model --save_path model.pth
"""
import argparse

import torch
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

MARK = ".parametrizations.weight."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dcp_checkpoint_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="net_ema.", help="net_ema. (EMA generator) or net. (live generator)")
    args = parser.parse_args()

    sd = {}
    _load_state_dict(sd, storage_reader=FileSystemReader(args.dcp_checkpoint_dir), planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    net = {k[len(args.prefix):]: v for k, v in sd.items() if k.startswith(args.prefix)}
    print(f"{len(net)} keys under {args.prefix}")

    out = {}
    mods = sorted({k.split(MARK)[0] for k in net if MARK in k})
    for mod in mods:
        W = net[f"{mod}{MARK}original"].float()
        u = net[f"{mod}{MARK}0._u"].float()
        g = net[f"{mod}{MARK}1.gain"].float()
        # One power iteration from the stored u: the EMA copy's u, v are never updated during training (the EMA only
        # averages parameters), and one iteration from them reaches the spectral norm of the averaged weight.
        v = torch.nn.functional.normalize(W.t() @ u, dim=0)
        u = torch.nn.functional.normalize(W @ v, dim=0)
        sigma = torch.dot(u, torch.mv(W, v)).clamp(min=1e-12)
        out[f"{mod}.weight"] = (g * W / sigma).to(torch.bfloat16)
    for k, v in net.items():
        if MARK not in k:
            out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
    print(f"folded {len(mods)} spectral-normed layers; {len(out)} keys")
    torch.save({f"net.{k}": v for k, v in out.items()}, args.save_path)
    print("saved", args.save_path)


if __name__ == "__main__":
    main()
