#!/usr/bin/env bash
# Standalone FID process: run on one GPU next to training; it scores every checkpoint copied to the cache folder
# (primary EMA as "fid", extra EMAs as "fid_ema_K").   Usage: bash experiments/imagenet/eval_fid.sh <RUN_NAME>
set -euo pipefail
: "${CHECKPOINT_PATH:?}" "${OUTPUT_PATH:?}"
RUN_NAME=${1:?run name}
CACHE=$(ls -d $OUTPUT_PATH/$RUN_NAME/cache/time_* | tail -1)
python main/edm/test_folder_edm.py --folder $CACHE \
    --ref_path $CHECKPOINT_PATH/imagenet_fid_refs_edm.npz --detector_url $CHECKPOINT_PATH/inception-2015-12-05.pkl \
    --wandb_entity "${WANDB_ENTITY:-}" --wandb_project "${WANDB_PROJECT:-dmad-imagenet}" --wandb_name ${RUN_NAME}_fid
