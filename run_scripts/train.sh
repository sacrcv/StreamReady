#!/bin/bash
# Train StreamReady on the Table 11 video-QA corpus.
#
# Both ProReady-QA and OVO-Bench are held out, so training never touches them.
# Prepare data/video_qa_train/annotation/train.json in the unified schema first
# (see the README), with frames under data/video_qa_train/frames.
#
# Usage:
#   bash run_scripts/train.sh
#
# Paper Sec. 3.4 trains reasoning and readiness separately. Run the two stages
# in order, carrying the first stage's weights into the second:
#   STAGE=reasoning bash run_scripts/train.sh
#   STAGE=readiness INIT_FROM=<stage1 checkpoint> bash run_scripts/train.sh
#
# INIT_FROM loads weights only. RESUME is for picking an interrupted run back up
# in the same stage: it also restores the optimizer and epoch, which only matches
# if the trainable parameter set is unchanged. Use INIT_FROM across a stage
# boundary -- the two stages train different parameters, so their optimizer state
# is not interchangeable.
#
# Environment overrides:
#   STAGE         reasoning | readiness | joint  (default: reasoning)
#   INIT_FROM     checkpoint to initialise weights from (default: none)
#   RESUME        checkpoint to resume a run from       (default: none)
#   BACKBONE      frozen Qwen2-VL checkpoint     (default: llm/Qwen2-VL-7B-Instruct)
#   BATCH_SIZE    per-GPU batch size             (default: 2)
#   ACCUM         gradient accumulation steps    (default: 8)
#   NUM_FRAMES    frames per question            (default: 100)
#   EPOCHS        training epochs                (default: 5, paper suppl. Sec. D)
#   LR            initial learning rate          (default: 2e-5)
#   NUM_GPUS      override GPU count
#   EXTRA         extra `--options key value` pairs appended verbatim
#
# Example, a short low-memory run for checking the setup end to end:
#   NUM_FRAMES=8 EPOCHS=1 BATCH_SIZE=1 ACCUM=1 bash run_scripts/train.sh

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
STAGE="${STAGE:-reasoning}"
BACKBONE="${BACKBONE:-llm/Qwen2-VL-7B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM="${ACCUM:-8}"
NUM_FRAMES="${NUM_FRAMES:-100}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-2e-5}"
EXTRA="${EXTRA:-}"

case "$STAGE" in
    reasoning|readiness|joint) ;;
    *) echo "ERROR: STAGE must be reasoning, readiness or joint (got '$STAGE')" >&2
       exit 1 ;;
esac

if [ -n "${INIT_FROM:-}" ] && [ -n "${RESUME:-}" ]; then
    echo "ERROR: set INIT_FROM (weights only) or RESUME (weights + optimizer)," \
         "not both" >&2
    exit 1
fi
if [ -n "${INIT_FROM:-}" ]; then
    # Weights only: the optimizer state from another stage covers a different
    # set of trainable parameters and would fail to load.
    EXTRA="model.load_finetuned True model.finetuned $INIT_FROM $EXTRA"
fi
if [ -n "${RESUME:-}" ]; then
    EXTRA="run.resume_ckpt_path $RESUME $EXTRA"
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "StreamReady training"
echo "  stage       $STAGE"
echo "  init from   ${INIT_FROM:-${RESUME:-pretrained BLIP-2/InstructBLIP}}"
echo "  GPUs        $GPUS"
echo "  backbone    $BACKBONE"
echo "  frames      $NUM_FRAMES"
echo "  batch       $BATCH_SIZE x $ACCUM accum"
echo "  epochs      $EPOCHS   lr $LR"

# One output directory per stage. The runner auto-resumes from
# <output_dir>/checkpoint_latest.pth when run.resume_ckpt_path is unset, so
# sharing a directory across stages would make the second stage pick up the
# first stage's optimizer state -- which covers a different set of trainable
# parameters and fails to load. Keeping them apart also preserves same-stage
# auto-resume for long runs.
torchrun --nproc_per_node="$GPUS" --master_port="$MASTER_PORT" train.py \
    --cfg-path lavis/projects/streamready/train.yaml \
    --options \
    model.backbone_model "$BACKBONE" \
    model.num_frames "$NUM_FRAMES" \
    model.train_stage "$STAGE" \
    run.output_dir "output/streamready_train_${STAGE}" \
    datasets.video_qa_train.num_frames "$NUM_FRAMES" \
    datasets.video_qa_train.vis_processor.train.n_frms "$NUM_FRAMES" \
    datasets.video_qa_train.vis_processor.eval.n_frms "$NUM_FRAMES" \
    run.batch_size_train "$BATCH_SIZE" \
    run.batch_size_eval "$BATCH_SIZE" \
    run.accum_grad_iters "$ACCUM" \
    run.max_epoch "$EPOCHS" \
    run.init_lr "$LR" \
    $EXTRA
