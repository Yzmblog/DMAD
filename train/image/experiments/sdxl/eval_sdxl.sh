#!/usr/bin/env bash
# COCO-10k evaluation of one generator checkpoint on one GPU (FID, patch FID, CLIP score; PickScore / HPSv2 /
# ImageReward too when METRIC_PATH holds their weights, see README).
# Usage: bash experiments/sdxl/eval_sdxl.sh <generator .bin> <4step|1step> <result.json> [extra eval_sdxl.py args]
set -euo pipefail
: "${CHECKPOINT_PATH:?}"
CKPT=${1:?generator weights}; MODE=${2:?4step or 1step}; RESULT=${3:?result json}; shift 3
case $MODE in
    4step) STEP_ARGS=(--num_step 4 --conditioning_timestep 999) ;;
    1step) STEP_ARGS=(--num_step 1 --conditioning_timestep 399) ;;
    *) echo "unknown mode $MODE" >&2; exit 1 ;;
esac
METRIC_ARGS=()
if [[ -n "${METRIC_PATH:-}" ]]; then
    METRIC_ARGS=(--pickscore_dir $METRIC_PATH/PickScore_v1 --pickscore_processor_dir $METRIC_PATH/CLIP-ViT-H-14-laion2B-s32B-b79K
                 --hps_v2_ckpt $METRIC_PATH/HPS_v2_compressed.pt --hps_v2_1_ckpt $METRIC_PATH/HPS_v2.1_compressed.pt
                 --image_reward_path $METRIC_PATH/ImageReward.pt --image_reward_med_config $METRIC_PATH/med_config.json)
fi
python main/sdxl/eval_sdxl.py --checkpoint_path $CKPT --model_id $CHECKPOINT_PATH/stable-diffusion-xl-base-1.0 \
    --anno_path $CHECKPOINT_PATH/coco10k/all_prompts.pkl --ref_dir $CHECKPOINT_PATH/coco10k/subset \
    --result_path $RESULT --grid_path ${RESULT%.json}.png "${STEP_ARGS[@]}" "${METRIC_ARGS[@]}" "$@"
