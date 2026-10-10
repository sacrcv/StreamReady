#!/bin/bash
# Evaluate StreamReady on OVO-Bench.
#
# Usage:
#   bash run_scripts/benchmarks/eval.sh /path/to/checkpoint.pth
#
# Convert the annotations and extract the frames first:
#   python tools/convert_benchmarks.py --benchmark ovo_bench \
#       --raw <native annotation dir> --frames-root data/ovo_bench/frames \
#       --output data/ovo_bench/annotation/ovo_bench_test.json
#   python data/extract_frames.py --src-dir <videos> \
#       --dst-dir data/ovo_bench/frames --fps 1
#
# For ProReady-QA use run_scripts/proready_qa/test.sh instead -- it reports ARS
# as well as accuracy.
#
# Environment overrides:
#   BACKBONE    frozen Qwen2-VL checkpoint (default: llm/Qwen2-VL-7B-Instruct)
#   BATCH_SIZE  per-GPU eval batch size (default: 2)
#   NUM_FRAMES  frames per question     (default: 100)
#   NUM_GPUS    override GPU count
#   EXTRA       extra `--options key value` pairs appended verbatim

set -euo pipefail

CHECKPOINT="${1:?usage: bash run_scripts/benchmarks/eval.sh <checkpoint>}"
BENCHMARK=ovo_bench

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CFG="lavis/projects/streamready/${BENCHMARK}.yaml"

if [ -n "${NUM_GPUS:-}" ]; then
    GPUS="$NUM_GPUS"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c .)
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPUS=$(nvidia-smi -L | wc -l)
else
    GPUS=1
fi

MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
BACKBONE="${BACKBONE:-llm/Qwen2-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_FRAMES="${NUM_FRAMES:-100}"
EXTRA="${EXTRA:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "StreamReady / $BENCHMARK evaluation"
echo "  checkpoint  $CHECKPOINT"
echo "  GPUs        $GPUS"
echo "  frames      $NUM_FRAMES"

torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" train.py \
    --cfg-path "$CFG" \
    --options \
    model.backbone_model "$BACKBONE" \
    model.num_frames "$NUM_FRAMES" \
    "datasets.${BENCHMARK}.num_frames" "$NUM_FRAMES" \
    "datasets.${BENCHMARK}.vis_processor.eval.n_frms" "$NUM_FRAMES" \
    run.batch_size_eval "$BATCH_SIZE" \
    run.evaluate True \
    run.resume_ckpt_path "$CHECKPOINT" \
    run.prefix test \
    $EXTRA
