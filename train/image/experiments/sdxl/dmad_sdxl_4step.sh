#!/usr/bin/env bash
# DMAD SDXL, 4-step generator trained with backward simulation (released checkpoint: iteration 17000, EMA).
RUN_NAME=${RUN_NAME:-dmad_sdxl_4step} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 26001 --guidance_lr 5e-7 --warmup_step 500 \
    --conditioning_timestep 999 --denoising --num_denoising_step 4 --denoising_timestep 1000 --backward_simulation "$@"
