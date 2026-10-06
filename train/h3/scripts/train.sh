#!/usr/bin/env bash
# DMAD training for MiniMax-H3 on one node with 8 GPUs (80 GB+ each). Run from the h3/ directory:
#   bash scripts/train.sh [config]
# Training resumes automatically from the latest checkpoint in training.output_dir.
set -euo pipefail
CONFIG=${1:-configs/dmad_minimax_h3.yaml}
NUM_GPUS=${NUM_GPUS:-8}
torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py --config "$CONFIG"
