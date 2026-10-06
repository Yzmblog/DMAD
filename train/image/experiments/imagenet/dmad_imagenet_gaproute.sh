#!/usr/bin/env bash
# DMAD ImageNet-64, teacher-UNet critic with spectral norm + gap routing (FID 1.24 at 568k iterations).
# Two extra EMAs (8000 and 16000 kimg) are saved with every checkpoint and scored by the FID process.
RUN_NAME=${RUN_NAME:-dmad_imagenet_gaproute} exec bash "$(dirname "$0")/train_common.sh" \
    --train_iters 1000001 --guidance_lr 2e-6 --teacher_loss_weight 3e-3 \
    --critic_spectral_norm --critic_max_timestep 500 --teacher_gap_route --gap_tau 0.5 \
    --generator_ema_kimg_list 8000,16000 "$@"
