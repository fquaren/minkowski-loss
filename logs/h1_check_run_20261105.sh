#!/bin/bash
# H1 shape check (DECISIONS §43, pre-registered in 6402caf): logs/h1_check_chain.sh without its pgrep wait,
# which never returned. Yesterday's shape outputs are backed up as *_20261104.* in the same folder.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES= LD_LIBRARY_PATH=/work/fquareng/.micromamba/envs/dl-stable/lib PYTHONPATH=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c nice -n 5 taskset -c 4-11 \
  /work/fquareng/.micromamba/envs/dl-stable/bin/python scripts/evaluate/analogue_distance.py --split test --min 20.2 --u_c 50.1 --space shape \
  --out /home/fquareng/work/data/extremes/OPERA/quality_v4/analogue > /home/fquareng/work/data/extremes/OPERA/quality_v4/analogue/log_test_shape_h1.txt 2>&1
echo "H1_CHECK_DONE exit=$? $(date)" >> /home/fquareng/work/data/extremes/OPERA/quality_v4/analogue/log_test_shape_h1.txt
