#!/usr/bin/env bash
# COCO-10k evaluation of one generator checkpoint on one GPU (FID, patch FID, CLIP score).
# Usage: bash experiments/sdxl/eval_sdxl.sh <generator .bin> <4step|1step> <result.json> [extra eval_sdxl.py args]
set -euo pipefail
: "${CHECKPOINT_PATH:?}"
CKPT=${1:?generator weights}; MODE=${2:?4step or 1step}; RESULT=${3:?result json}; shift 3
case $MODE in
    4step) STEP_ARGS=(--num_step 4 --conditioning_timestep 999) ;;
    1step) STEP_ARGS=(--num_step 1 --conditioning_timestep 399) ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
python main/sdxl/eval_sdxl.py --checkpoint_path $CKPT --model_id $CHECKPOINT_PATH/stable-diffusion-xl-base-1.0 \
    --anno_path $CHECKPOINT_PATH/coco10k/all_prompts.pkl --ref_dir $CHECKPOINT_PATH/coco10k/subset \
    --result_path $RESULT --grid_path ${RESULT%.json}.png "${STEP_ARGS[@]}" "$@"
