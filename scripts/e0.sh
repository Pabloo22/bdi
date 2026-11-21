#!/usr/bin/env bash
# Run E0 (docs/e0.md). Each run uses one GPU, and every run uses seed 42.
#
#   scripts/e0.sh                # the two main runs, one after another, on GPU 0
#   scripts/e0.sh 0 1            # the same, one run per GPU
#   FOLLOW_UPS=1 scripts/e0.sh   # also the five follow-up runs (GPU ids work here too)
#
# Finished runs (with a finished.json) are skipped, so adding FOLLOW_UPS=1 later runs only the new ones.
set -euo pipefail
cd "$(dirname "$0")/.."

OPCD="objective.direction=reverse objective.topk=256"  # OPCD's loss: reverse KL, student's top 256
PAPER="objective.direction=forward objective.topk=0" # the paper's: forward KL, whole vocabulary
# The main comparison: OPCD's loss against the paper's, on owl.
ARMS=(
  "teacher.animal=owl $OPCD"
  "teacher.animal=owl $PAPER"
)
# Only once the main runs show something worth extending.
if [[ ${FOLLOW_UPS:-0} == 1 ]]; then
  ARMS+=(
    # Reverse KL over the whole vocabulary: separates the effect of direction from truncation.
    "teacher.animal=owl objective.direction=reverse objective.topk=0"
    # Other animals: eagle starts high, penguin starts near zero (soft-distillation paper, Figure 3).
    "teacher.animal=eagle $OPCD"
    "teacher.animal=eagle $PAPER"
    "teacher.animal=penguin $OPCD"
    "teacher.animal=penguin $PAPER"
  )
fi

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
  if ((${#queue[@]} == 0)); then continue; fi # more GPUs than runs
  run_queue "${GPUS[i]}" "${queue[@]}" >"runs/e0/logs/gpu${GPUS[i]}.log" 2>&1 &
done
wait
