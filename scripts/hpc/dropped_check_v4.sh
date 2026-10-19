#!/bin/bash
# Contamination test of the days that break gate criterion 3 (DECISIONS §32, option (b)).
# RADKLIM fetch now (network-bound); the gauge runs (2 workers) after gate_v4.sh, to stay
# within the 8-core budget beside build_v4.sh. Resumable.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
P="taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u"
O=/home/fquareng/work/data/extremes/OPERA/quality_v4/gate_tail/dropped_check
set -o pipefail
echo "=== dropped-days check, code $(git rev-parse --short HEAD), $(date)"
echo "=== RADKLIM fetch $(date)"
$P scripts/data/fetch_radklim.py --days_file $O/all.txt 2>&1 | tail -4
until grep -q GATE_V4_DONE /work/fquareng/ch2/minkowski-loss/logs/gate_v4.log; do
  grep -qE "FAILED" /work/fquareng/ch2/minkowski-loss/logs/gate_v4.log && break
  sleep 60
done
echo "=== gauge runs $(date)"
$P scripts/data_quality/gauge_hourly.py run --out $O/gauges --days_file $O/all.txt --variants full --workers 2 \
  2>&1 | grep -v "Warn\|warn\|pystepsrc" | tail -5 || { echo "GAUGES FAILED"; exit 1; }
echo "=== analyze $(date)"
$P scripts/data_quality/dropped_days_check.py analyze 2>&1 | grep -v Warn || { echo "ANALYZE FAILED"; exit 1; }
echo "=== DROPPED_CHECK_DONE $(date)"
