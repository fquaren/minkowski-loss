#!/bin/bash
# National-border tail test (DECISIONS §36c): radar pass, then the border and country-year tables.
cd "$(dirname "$0")/../.."
export LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
PY=/work/fquareng/.micromamba/envs/dl-stable/bin/python
taskset -c 4-11 nice -n 5 $PY -u scripts/analysis/radar_band_tail.py run_national --workers "${1:-3}" || exit 1
taskset -c 4-11 $PY -u scripts/analysis/radar_band_tail.py analyze_national || exit 1
echo BAND_NATIONAL_DONE
