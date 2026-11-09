#!/bin/bash
# Two R4 cells in sequence on one node40 GPU (2026-11-09). Chained in one shell because a second cell waiting on the
# GPU lock could take the GPU between the first cell's training and its eval launch, which would then refuse.
#   node40_r4_round.sh <gpu> "<cell args 1>" "<cell args 2>"   (cell args after <gpu>, as node40_r4_cell.sh takes them)
# Run detached:  setsid nohup scripts/hpc/node40_r4_round.sh 0 "..." "..." > logs/node40/round_gpu0.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")/../.."
GPU=$1; shift
for CELL in "$@"; do
  read -r -a A <<< "$CELL"
  bash scripts/hpc/node40_r4_cell.sh $GPU "${A[@]}" > logs/node40/cell_${A[0]}.log 2>&1 < /dev/null
  echo "=== $(date) cell ${A[0]} exit $? : $(tail -1 logs/node40/cell_${A[0]}.log)"
done
echo "=== ROUND_DONE gpu $GPU $(date)"
