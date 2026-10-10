#!/bin/bash
# Evaluate StreamReady on ProReady-QA, reporting Accuracy, ARS and effective accuracy.
#
# Usage:
#   bash run_scripts/proready_qa/test.sh /path/to/checkpoint.pth
#
# Environment overrides:
#   BACKBONE            frozen Qwen2-VL checkpoint (default: llm/Qwen2-VL-7B-Instruct)
#   BATCH_SIZE          per-GPU eval batch size (default: 2)
#   NUM_FRAMES          frames per question     (default: 100)
#   READINESS_THRESHOLD LLM trigger threshold   (default: 0.35, paper Sec. 5)
#   NUM_GPUS            override GPU count
#   EXTRA               extra `--options key value` pairs appended verbatim

set -euo pipefail

CHECKPOINT="${1:?usage: bash run_scripts/proready_qa/test.sh /path/to/checkpoint.pth}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

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
READINESS_THRESHOLD="${READINESS_THRESHOLD:-0.35}"
EXTRA="${EXTRA:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "StreamReady / ProReady-QA evaluation"
echo "  checkpoint  $CHECKPOINT"
echo "  GPUs        $GPUS"
echo "  frames      $NUM_FRAMES"
echo "  threshold   $READINESS_THRESHOLD"

torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" train.py \
    --cfg-path lavis/projects/streamready/proready_qa.yaml \
    --options \
    model.backbone_model "$BACKBONE" \
    model.num_frames "$NUM_FRAMES" \
    model.readiness_threshold "$READINESS_THRESHOLD" \
    datasets.proready_qa.num_frames "$NUM_FRAMES" \
    datasets.proready_qa.vis_processor.eval.n_frms "$NUM_FRAMES" \
    run.batch_size_eval "$BATCH_SIZE" \
    run.evaluate True \
    run.resume_ckpt_path "$CHECKPOINT" \
    run.prefix test \
    $EXTRA
