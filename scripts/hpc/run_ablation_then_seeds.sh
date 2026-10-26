#!/bin/bash
# GPU 1 queue after the bracket (DECISIONS §34, §37):
#   1. pre-break ablation, post-break arm: vanilla (w = 0) on full_train_postbreak (987,099 rows), 33 epochs
#      (~ the all-data baseline's 25 epochs x 1,299,686 rows in steps), same seed (42), full val list;
#   2. the bracket winner by the §16 rule (6e-5) with seeds 43 and 44, 25 epochs, as the bracket runs.
# The ablation goes first: if it drops the pre-break rows, the seeds would have to be retrained.
cd "$(dirname "$0")/../.."
echo "=== ablation post-break arm $(date)"
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 0 100.0 \
  /work/fquareng/ch2/minkowski-loss/logs/config_v4_adv_postbreak.yaml
echo "=== ablation finished $(date); run dir $(ls -t runs/sr_analytical | head -1)"
for s in 43 44; do
  echo "=== seed $s, w=6.00e-05 $(date)"
  bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 6.00e-05 100.0 configs/config_v4_adv.yaml "$s"
  echo "=== seed $s finished $(date); run dir $(ls -t runs/sr_analytical | head -1)"
done
echo "=== QUEUE_DONE $(date)"
