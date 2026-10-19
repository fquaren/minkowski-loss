#!/bin/bash
# §20 gate criteria 3 and 5 on the v4 build, and the hourly threshold analysis (DECISIONS §32).
# 2 workers: runs beside build_v4.sh (6 workers) within the 8-core budget. Resumable.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1
P="taskset -c 4-11 /work/fquareng/.micromamba/envs/dl-stable/bin/python -u"
G=/home/fquareng/work/data/extremes/OPERA/quality_v4/gate_tail
set -o pipefail
echo "=== gate v4, code $(git rev-parse --short HEAD), $(date)"
if [ ! -f $G/test_values.npz ]; then
  echo "=== criterion 3: extract test $(date)"
  $P scripts/data_quality/gate_tail.py extract --workers 2 2>&1 | grep -v Warn || { echo "EXTRACT FAILED"; exit 1; }
fi
echo "=== criterion 3: analyze $(date)"
$P scripts/data_quality/gate_tail.py analyze 2>&1 | grep -v Warn || { echo "CRITERION 3 FAILED TO RUN"; exit 1; }
echo "=== criterion 5 / threshold: pot_threshold v4 extract $(date)"
$P scripts/data_quality/pot_threshold.py --version v4 --stage extract --workers 2 2>&1 | grep -v Warn || { echo "POT EXTRACT FAILED"; exit 1; }
echo "=== criterion 5 / threshold: pot_threshold v4 analyze $(date)"
$P scripts/data_quality/pot_threshold.py --version v4 --stage analyze --workers 2 --B 200 2>&1 | grep -v Warn || { echo "POT ANALYZE FAILED"; exit 1; }
echo "=== GATE_V4_DONE $(date)"
