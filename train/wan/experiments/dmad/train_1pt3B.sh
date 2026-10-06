#!/usr/bin/env bash
# DMAD Wan2.1-1.3B, 480p T2V: one node, 8 GPUs, batch 8 (released checkpoint: iteration 17000, EMA).
#   DATA : folder with the training shards (prepare_data.sh);  CKPT : assets/checkpoints
set -euo pipefail
: "${DATA:?}"
CKPT=${CKPT:-assets/checkpoints}
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --standalone --nproc_per_node=8 \
    -m scripts.train --config=rcm/configs/registry_distill.py -- experiment=wan2pt1_1pt3B_res480p_t2v_dmad \
        job.name=${RUN_NAME:-dmad_wan2pt1_1pt3B} \
        model.config.teacher_ckpt=$CKPT/Wan2.1-T2V-1.3B.dcp \
        model.config.tokenizer.vae_pth=$CKPT/Wan2.1_VAE.pth \
        model.config.text_encoder_path=$CKPT/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=$CKPT/umT5_wan_negative_emb.pt \
        dataloader_train.tar_path_pattern="$DATA/shards/shard*.tar" "$@"
