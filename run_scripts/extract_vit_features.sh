#!/usr/bin/env bash
# Pre-extract the frozen Qwen2-VL vision tower's features for one dataset.
#
# StreamReady feeds up to 100 frames per clip through a frozen visual encoder, so
# that forward dominates step time while contributing no gradients. Caching it
# once makes training several times faster at the cost of disk space
# (roughly 1.2 MB per frame in fp16, so budget generously).
#
# BACKBONE and IMAGE_SIZE must match model.backbone_model and model.image_size.
# The tool records them in <output>/_cache_meta.json and refuses to mix settings
# in one directory, because an image_size mismatch changes only the token count
# and would otherwise go unnoticed.
#
# Usage:
#   bash run_scripts/extract_vit_features.sh <dataset> [num_shards]
#
# Examples:
#   bash run_scripts/extract_vit_features.sh proready_qa
#   bash run_scripts/extract_vit_features.sh proready_qa 4     # 4 GPUs in parallel
#
# Then train with the cache by adding:
#   EXTRA="model.use_cached_vit_features True \
#          datasets.proready_qa.cached_vit_features_root proready_qa/vit_features" \
#     bash run_scripts/train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:?usage: bash run_scripts/extract_vit_features.sh <dataset> [num_shards]}"
NUM_SHARDS="${2:-1}"

FRAMES_ROOT="${FRAMES_ROOT:-data/${DATASET}/frames}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/${DATASET}/vit_features}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BACKBONE="${BACKBONE:-llm/Qwen2-VL-7B-Instruct}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"

if [[ ! -d "$FRAMES_ROOT" ]]; then
  echo "ERROR: frames directory '$FRAMES_ROOT' does not exist." >&2
  echo "Extract frames first with: python data/extract_frames.py (see README)." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
echo "Extracting vision-tower features"
echo "  frames : $FRAMES_ROOT"
echo "  output : $OUTPUT_ROOT"
echo "  encoder: $BACKBONE @ ${IMAGE_SIZE}px"
echo "  shards : $NUM_SHARDS"

if [[ "$NUM_SHARDS" -eq 1 ]]; then
  python tools/extract_vit_features.py \
      --frames-root "$FRAMES_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --batch-size "$BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --backbone-model "$BACKBONE" \
      --image-size "$IMAGE_SIZE"
else
  # One shard per GPU. The tool is idempotent, so a shard can be re-run.
  pids=()
  for (( shard=0; shard<NUM_SHARDS; shard++ )); do
    CUDA_VISIBLE_DEVICES="$shard" python tools/extract_vit_features.py \
        --frames-root "$FRAMES_ROOT" \
        --output-root "$OUTPUT_ROOT" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --backbone-model "$BACKBONE" \
        --image-size "$IMAGE_SIZE" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$shard" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

echo "Done. Cached features in $OUTPUT_ROOT"
