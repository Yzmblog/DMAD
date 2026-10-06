#!/usr/bin/env bash
# DMAD SDXL, 1-step generator initialized from DMD2's ODE-pretrained one-step model (released checkpoint: iteration
# 49000, EMA).
: "${CHECKPOINT_PATH:?}"
RUN_NAME=${RUN_NAME:-dmad_sdxl_1step} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 50001 --guidance_lr 5e-7 --warmup_step 500 --conditioning_timestep 399 \
    --generator_ckpt_path $CHECKPOINT_PATH/sdxl_lr1e-5_8node_ode_pretraining_10k_cond399_checkpoint_model_002000.bin "$@"
