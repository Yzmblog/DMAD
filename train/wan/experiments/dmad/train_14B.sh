#!/usr/bin/env bash
# DMAD Wan2.1-14B, 480p T2V: 8 nodes x 8 GPUs, context parallel 8 (batch 8), FSDP shard 32 (released checkpoint:
# iteration 19500, EMA). Run on every node with NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT set.
set -euo pipefail
: "${DATA:?}" "${NODE_RANK:?}" "${MASTER_ADDR:?}"
CKPT=${CKPT:-assets/checkpoints}
PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nnodes ${NNODES:-8} --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR --master_port ${MASTER_PORT:-29500} --nproc_per_node=8 \
    -m scripts.train --config=rcm/configs/registry_distill.py -- experiment=wan2pt1_14B_res480p_t2v_dmad \
        job.name=${RUN_NAME:-dmad_wan2pt1_14B} \
        model.config.teacher_ckpt=$CKPT/Wan2.1-T2V-14B.dcp \
        model.config.tokenizer.vae_pth=$CKPT/Wan2.1_VAE.pth \
        model.config.text_encoder_path=$CKPT/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=$CKPT/umT5_wan_negative_emb.pt \
        dataloader_train.tar_path_pattern="$DATA/shards/shard*.tar" "$@"
