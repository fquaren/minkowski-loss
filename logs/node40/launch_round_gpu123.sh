cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
setsid nohup bash scripts/hpc/node40_r4_round.sh 1 'r4_area_full_s42 full area_trunc 42 1,0,0 50.1' 'r4_mink_cut_s44 cut mink_trunc 44 1,1,1 50.1' > logs/node40/round_gpu1.log 2>&1 < /dev/null &
setsid nohup bash scripts/hpc/node40_r4_round.sh 2 'r4_mse_full_s43 full 0 43 1,1,1 none' 'r4_mse_full_s44 full 0 44 1,1,1 none' > logs/node40/round_gpu2.log 2>&1 < /dev/null &
setsid nohup bash scripts/hpc/node40_r4_round.sh 3 'r4_mink_full_s43 full mink_trunc 43 1,1,1 50.1' 'r4_mink_full_s44 full mink_trunc 44 1,1,1 50.1' > logs/node40/round_gpu3.log 2>&1 < /dev/null &
echo launched
