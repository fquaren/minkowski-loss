#!/bin/bash
# Validation chain: gauges -> pairs -> tail validation; after the flag scan, the radar
# ranking with the flag signals. Cores 10-11 while the flag scan holds 4-9 (8-core budget).
#   setsid nohup bash scripts/validation/run_validation.sh > logs/validation.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="taskset -c 10,11 $E/bin/python -u"
echo "=== waiting for the gauge download ($(date)) ==="
until grep -q "^=== done" logs/fetch_gauges.log 2>/dev/null; do sleep 300; done
grep "^\[dwd\]\|^\[smn\]" logs/fetch_gauges.log
echo "=== pairs ($(date)) ==="
$PY scripts/validation/gauge_vs_radar.py config.yaml --workers 2 --skip_existing || { echo "pairs FAILED"; exit 1; }
echo "=== tail validation ($(date)) ==="
$PY scripts/validation/validate_tail.py || echo "validate_tail FAILED"
echo "=== waiting for the flag scan ($(date)) ==="
until grep -q "^=== done" logs/scan_flags.log 2>/dev/null; do sleep 300; done
echo "=== radar ranking with flags ($(date)) ==="
$PY scripts/data_quality/radar_quality_v2.py || echo "radar_quality FAILED"
echo "=== done ($(date)) ==="
