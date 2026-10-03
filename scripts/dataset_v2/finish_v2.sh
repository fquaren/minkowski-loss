#!/bin/bash
# v2 rebuild, completion after the 2025-10 -> 2026-09 NIMBUS fetch finished (2026-09-30).
# rebuild_v2.sh computed the clutter climatology at 13:38 on 2026-09-30, while that fetch was
# still running: the climatology has no 2026 entry (2026 days fell back to the whole-period
# hot mask) and its 2025 entry stops at 2025-10-22. run_build.sh then stopped at its 99% gate
# (4855 tile tables for 5016 day stores). This recomputes the climatology over the complete
# archive, rescans 2025 + 2026 (the only years whose per-year mask changes), reruns the
# month-matched era gap, then the build. The partial climatology is kept, not deleted.
#   setsid nohup bash scripts/dataset_v2/finish_v2.sh > logs/finish_v2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
Q=/home/fquareng/work/data/extremes/OPERA/quality_v2
echo "=== start ($(date)) ==="
[ -f $Q/clutter_climatology.npz ] && mv $Q/clutter_climatology.npz $Q/clutter_climatology_partial2026.npz
rm -f $Q/tiles/2025*.csv.gz $Q/tiles/2026*.csv.gz
bash scripts/dataset_v2/run_scan.sh > logs/dataset_v2_scan3.log 2>&1
echo "=== era gap: months ($(date)) ==="
taskset -c 4-9 $E/bin/python -u scripts/data_quality/era_gap.py --part months
echo "=== build ($(date)) ==="
SCANLOG=logs/dataset_v2_scan3.log bash scripts/dataset_v2/run_build.sh > logs/dataset_v2_build3.log 2>&1
echo "=== done ($(date)) ==="
