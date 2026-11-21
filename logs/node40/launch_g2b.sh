#!/bin/bash
# Baselines v2 G2, second batch (2026-11-21): windowed-spectral bracket below its parity weight 0.51, which hacks
# (false alarms 1.1-1.4%, RL bias 7-13): {0.51/3, 0.51/10} = {0.17, 0.051}, both arms, seed 42, grid truncated
# below 50.1 mm as G2; plus the MSE + QM rows (eval only).
#   GPU 0: full@0.17   GPU 1: cut@0.17   GPU 3: QM rows, then full@0.051
#   GPU 2: cut@0.051 once the G2 fss round (round_g2_gpu2.log) prints ROUND_DONE, so its eval is not refused.
# node40 limits: GPUs 0-3 only; every process pinned to CPUs 0-5.
set -euo pipefail
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
[ "$(hostname -s)" = node40 ] || { echo "not node40"; exit 1; }
S="env R4_LOSS=spectral R4_SPECTRAL_WINDOW=hann"
cell() { echo "r4_spectral_hann_$1_s42_w$2 $1 $2 42 1,1,1 50.1"; }
$S setsid nohup taskset -c 0-5 bash scripts/hpc/node40_r4_round.sh 0 "$(cell full 0.17)" \
  > logs/node40/round_g2b_gpu0.log 2>&1 < /dev/null & echo "gpu 0 full@0.17: $!"
$S setsid nohup taskset -c 0-5 bash scripts/hpc/node40_r4_round.sh 1 "$(cell cut 0.17)" \
  > logs/node40/round_g2b_gpu1.log 2>&1 < /dev/null & echo "gpu 1 cut@0.17: $!"
$S setsid nohup taskset -c 0-5 bash -c "bash scripts/hpc/node40_qm_rows.sh 3; \
  bash scripts/hpc/node40_r4_round.sh 3 '$(cell full 0.051)'" \
  > logs/node40/round_g2b_gpu3.log 2>&1 < /dev/null & echo "gpu 3 QM, then full@0.051: $!"
$S setsid nohup taskset -c 0-5 bash -c "until grep -q ROUND_DONE logs/node40/round_g2_gpu2.log; do sleep 300; done; \
  bash scripts/hpc/node40_r4_round.sh 2 '$(cell cut 0.051)'" \
  > logs/node40/round_g2b_gpu2.log 2>&1 < /dev/null & echo "gpu 2 cut@0.051 after the fss round: $!"
