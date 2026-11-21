#!/usr/bin/env bash
# Run every arm of E0 (docs/e0.md). Each run uses one GPU.
#
#   scripts/e0.sh          # all runs one after another on GPU 0
#   scripts/e0.sh 0 1 2 3  # runs spread over GPUs 0-3, one queue per GPU
#
# Finished runs (with a finished.json) are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

OEL="objective.direction=reverse objective.topk=256"  # OEL's loss: reverse KL, student's top 256
PAPER="objective.direction=forward objective.topk=0" # the paper's: forward KL, whole vocabulary
ARMS=(
  "teacher.animal=owl $OEL seed=42"
  "teacher.animal=owl $OEL seed=43"
  "teacher.animal=owl $OEL seed=44"
  "teacher.animal=owl $PAPER seed=42"
  "teacher.animal=owl $PAPER seed=43"
  "teacher.animal=owl $PAPER seed=44"
  # Other animals: eagle starts high, penguin starts near zero (soft-distillation paper, Figure 3).
  "teacher.animal=eagle $OEL seed=42"
  "teacher.animal=eagle $PAPER seed=42"
  "teacher.animal=penguin $OEL seed=42"
  "teacher.animal=penguin $PAPER seed=42"
  # Reverse KL over the whole vocabulary: separates the effect of direction from truncation.
  "teacher.animal=owl objective.direction=reverse objective.topk=0 seed=42"
)

GPUS=("${@:-0}")
mkdir -p runs/e0/logs

run_queue() {
  local gpu=$1
  shift
  for arm in "$@"; do
    # shellcheck disable=SC2086  # each arm is several key=value words
    CUDA_VISIBLE_DEVICES=$gpu poetry run python -m bdi.distill.run configs/e0/base.yaml $arm
  done
}

for i in "${!GPUS[@]}"; do
  queue=()
  for j in "${!ARMS[@]}"; do
    if ((j % ${#GPUS[@]} == i)); then queue+=("${ARMS[j]}"); fi
  done
  run_queue "${GPUS[i]}" "${queue[@]}" >"runs/e0/logs/gpu${GPUS[i]}.log" 2>&1 &
done
wait
