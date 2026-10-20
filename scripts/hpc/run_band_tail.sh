#!/bin/bash
# Tail-focused boundary test (DECISIONS §36b): radar pass, then the per-radar-year table.
# 2 workers while the v4 build holds 6 of the 8 allowed cores.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
PY=/work/fquareng/.micromamba/envs/dl-stable/bin/python
taskset -c 4-11 nice -n 5 $PY -u scripts/analysis/radar_band_tail.py run --workers "${1:-2}" || exit 1
taskset -c 4-11 $PY -u scripts/analysis/radar_band_tail.py analyze || exit 1
echo BAND_TAIL_DONE
