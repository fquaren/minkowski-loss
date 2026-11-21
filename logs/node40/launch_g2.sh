#!/bin/bash
# Baselines v2 G2 (2026-11-18): 8 U-Net cells, full then cut per GPU, seed 42, grid truncated below 50.1 mm.
# Weights: eval_results/gradient_audit/2026-11-16_v2_parity/weights.json (radar-day-exclusion worktree).
# node40 limits: GPUs 0-3 only; every process (rounds, cells, training, workers) pinned to CPUs 0-5.
set -euo pipefail
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
[ "$(hostname -s)" = node40 ] || { echo "not node40"; exit 1; }
round() {   # gpu loss weight [window]
  local g=$1 loss=$2 w=$3 win=${4:-}
  env R4_LOSS=$loss ${win:+R4_SPECTRAL_WINDOW=$win} setsid nohup taskset -c 0-5 \
    bash scripts/hpc/node40_r4_round.sh $g \
    "r4_${loss}${win:+_$win}_full_s42 full $w 42 1,1,1 50.1" \
    "r4_${loss}${win:+_$win}_cut_s42 cut $w 42 1,1,1 50.1" \
    > logs/node40/round_g2_gpu$g.log 2>&1 < /dev/null &
  echo "gpu $g $loss${win:+ ($win)} w $w: round pid $!"
}
round 0 wmse 0.00043
round 1 quantile 0.068
round 2 fss 0.00021
round 3 spectral 0.51 hann
