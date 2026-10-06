#!/usr/bin/env bash
# Build all training data for DMAD on MiniMax-H3 from UltraVideo (https://huggingface.co/datasets/APRIL-AIGC/UltraVideo).
# Run from the h3/ directory on one node with 8 GPUs. Every step is resumable.
#
#   H3_MODEL_DIR   MiniMax-H3 base model directory
#   ULTRAVIDEO_DIR UltraVideo download (short.csv and clips_short_1920/clips_short_1920_*.zip)
#   DATA_ROOT      output root
set -euo pipefail
: "${H3_MODEL_DIR:?}" "${ULTRAVIDEO_DIR:?}" "${DATA_ROOT:?}"
NUM_GPUS=${NUM_GPUS:-8}
P=$DATA_ROOT/ultravideo_prompts

# 1. caption list (42,158 unique captions) + caption -> clip map
python data_process/build_prompt_list.py --csv "$ULTRAVIDEO_DIR/short.csv" --out-dir "$P"

# 2. real clips -> H3 video/audio latents (124- and 107-frame tiers; training uses 124)
torchrun --standalone --nproc_per_node="$NUM_GPUS" data_process/encode_real_videos.py \
    --model_dir "$H3_MODEL_DIR" --zip_dir "$ULTRAVIDEO_DIR/clips_short_1920" --output_dir "$DATA_ROOT/real_latents"

# 3. captions needed for training (captions of the 124-frame clips + visualization prompts 0-3)
python data_process/build_pool_indices.py --real-latents "$DATA_ROOT/real_latents" \
    --caption-map "$P/ultravideo_index_to_clipid.jsonl" --out "$P/pool_indices.txt"

# 4. text encoding (one shard per GPU), then merge the shard metadata
for r in $(seq 0 $((NUM_GPUS - 1))); do
    python data_process/encode_prompts.py "$P/prompts_dedup.txt" --model-path "$H3_MODEL_DIR" \
        --output-dir "$DATA_ROOT/prompt_cache" --index-file "$P/pool_indices.txt" --shard "$r/$NUM_GPUS" --device "cuda:$r" &
done
wait
python data_process/encode_prompts.py "$P/prompts_dedup.txt" --output-dir "$DATA_ROOT/prompt_cache" --merge-shards

# 5. teacher samples (31-step base model, one per training caption). This is the most expensive step
#    (~18k samples); spread the shards over several nodes if available.
for r in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r python data_process/gen_teacher_data.py --model-dir "$H3_MODEL_DIR" \
        --prompt-cache "$DATA_ROOT/prompt_cache" --output-dir "$DATA_ROOT/teacher_latents" --shard "$r/$NUM_GPUS" &
done
wait
echo "done: set prompt_cache_dir / real_latent_dir / real_caption_map / teacher_latent_dir in configs/dmad_minimax_h3.yaml"
