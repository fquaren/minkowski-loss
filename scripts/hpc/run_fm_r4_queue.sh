#!/bin/bash
# FM R4: the U-Net R4 loss comparison repeated as flow-matching reward fine-tunes (node32, GPU 1).
#
# Rows per arm: control (w=0), Minkowski (energy_qm), area-only, spectral, wetarea, SSIM, optical
# flow. Every competing reward is an energy score (proper for the L1 modes; SSIM uses 1-SSIM).
#   1. full arm, seed 42: the 5 new rows (control and Minkowski full exist: fm_energy_ctl, fm_energy_qm)
#   2. U-Net R4 rows evaluated on the same two FM subsets, for one cross-model table
#   3. cut arm base: sigma_r for the cut backbone, then the FM base model on the cut list
#   4. cut arm, seed 42: all 7 rows
#   5. seeds 43 and 44 for control and Minkowski, both arms
# A new reward's weight is set once, on the full arm, by gradient-norm parity with the
# retention term (DECISIONS §41, as energy_qm), and reused on the cut arm, as the U-Net R4 did.
# Resumable: every finished label is in logs/fm_r4/done.txt and is skipped.
cd "$(dirname "$0")/../.."
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
R=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
STORE=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv
OUTE=$W/eval_results/extremes
L=$PWD/logs/fm_r4; mkdir -p $L/configs
DONE=$L/done.txt; touch $DONE
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 4-11 $E/bin/python -u"

FMB_FULL=$W/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
BB_FULL=$W/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
SIG_FULL=0.01266805361956358
BB_CUT=$R/runs/sr_analytical/UNet_Ana_20261105_142458/unet_latest.pth       # r4_mse_cut_s42

echo "=== FM R4 queue, code $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted, $(date)"
until grep -q "R4_QUEUE_DONE" $R/logs/r4_spectral_bracket.log; do sleep 300; done
echo "=== U-Net R4 queue finished, node32 GPU 1 free $(date)"

mkcfg() {   # out base mode weight channels train_list backbone sigma name
  $E/bin/python - "$@" <<'PY'
import sys, yaml
out, base, mode, w, cw, train, bb, sig, name = sys.argv[1:10]
c = yaml.safe_load(open(base))
c["REWARD_MODE"] = mode
c["REWARD_WEIGHT"] = float(w)
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [float(x) for x in cw.split(",")]
c["TRAIN_METADATA_FILE"] = train
c["BACKBONE_CHECKPOINT"] = bb
c["FM_RESIDUAL_SCALE"] = float(sig)       # pinned: residual_stats.json holds another backbone's value
c["EXPERIMENT_NAME"] = name
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
}
EQM=$W/configs/config_v4_adv_fm_energy_qm.yaml

parity() {   # label mode channels -> REWARD_WEIGHT at gradient-norm parity with retention (2 s.f.)
  local lab=$1 mode=$2 cw=$3 f=$L/weights.txt
  local w=$(awk -v m="$lab" '$1==m {print $2}' $f 2>/dev/null)
  [ -n "$w" ] && { echo $w; return; }
  mkcfg $L/configs/parity_$lab.yaml $EQM $mode 1.0 $cw $META/full_train.txt $BB_FULL $SIG_FULL parity
  $P scripts/evaluate/grad_norm_terms.py $L/configs/parity_$lab.yaml --fm_checkpoint $FMB_FULL \
    --backbone $BB_FULL --n_batches 4 > $L/parity_$lab.log 2>&1 || { echo FAIL; return; }
  w=$(grep "^mean |g|" $L/parity_$lab.log | awk '{printf "%.2g", $8 / $4}')
  [ -z "$w" ] && { echo FAIL; return; }
  echo "$lab $w" >> $f; echo $w
}

ev_fm() {   # label run_dir backbone checkpoint_name
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/fm_r4_$1 ] && { echo "exists, skipping eval $1 $sub"; continue; }
    $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model fm \
      --fm_checkpoint $2/${4:-fm_mink_latest.pth} --backbone $3 --ensemble 16 --split val --pot_threshold 50.1 \
      --tag fm_r4_$1 --output_dir $OUTE/v4adv_val$sub 2>&1 | grep -v Warn | tail -2 || echo "EVAL FAILED $1 $sub"
  done
}

ft() {   # label mode weight channels arm seed
  local lab=$1 mode=$2 w=$3 cw=$4 arm=$5 seed=$6
  grep -q "^$lab " $DONE && { echo "done already: $lab"; return; }
  [ "$w" = FAIL ] && { echo "NO WEIGHT, SKIPPED $lab"; return; }
  local list=$META/full_train.txt bb=$BB_FULL fmb=$FMB_FULL sig=$SIG_FULL
  if [ $arm = cut ]; then
    list=$META/full_train_cut50.1.txt; bb=$BB_CUT; fmb=$(cat $L/fmb_cut.txt); sig=$(cat $L/sigma_cut.txt)
  fi
  local cfg=$L/configs/$lab.yaml name=FMr4_$lab
  mkcfg $cfg $EQM $mode $w $cw $list $bb $sig $name
  echo "=== fine-tune $lab: mode $mode, w $w, channels $cw, $(basename $list), seed $seed $(date)"
  $P scripts/train/train_fm_minkowski.py $cfg --fm_checkpoint $fmb --backbone $bb --seed $seed \
    > $L/ft_$lab.log 2>&1 || { echo "FINETUNE FAILED $lab"; return; }
  grep -E "sigma_r|reward mode=|channel weights" $L/ft_$lab.log | head -3
  grep -q "config FM_RESIDUAL_SCALE" $L/ft_$lab.log || { echo "SIGMA NOT FROM CONFIG, NOT EVALUATED $lab"; return; }
  local rd=$(ls -td runs/sr_fm_minkowski/${name}_* | head -1)
  echo "$lab $rd $w" >> $DONE
  ev_fm $lab $rd $bb
  echo "=== $lab done $(date)"
}

# ---- 1. full arm, seed 42 -------------------------------------------------------------
for spec in "spectral spectral 1,1,1" "area energy 1,0,0" "wetarea wetarea 1,1,1" \
            "ssim ssim 1,1,1" "opticalflow opticalflow 1,1,1"; do
  set -- $spec
  w=$(parity $1 $2 $3); echo "parity weight $1: $w"
  ft ${1}_full_s42 $2 $w $3 full 42
done

# ---- 2. U-Net R4 rows on the FM subsets ---------------------------------------------------
echo "=== U-Net R4 rows on the FM subsets $(date)"
while read lab run; do
  for sub in ext sub; do
    [ -d $OUTE/v4adv_val$sub/$lab ] && continue
    ck=$R/runs/sr_analytical/$run/unet_latest.pth; [ -f $ck ] || ck=$W/runs/sr_analytical/$run/unet_latest.pth
    $P scripts/evaluate/eval_extremes.py $W/configs/config_v4_adv_fm_eval$sub.yaml --model backbone --checkpoint $ck \
      --split val --pot_threshold 50.1 --tag $lab --output_dir $OUTE/v4adv_val$sub 2>&1 | grep -v Warn | tail -1 \
      || echo "UNET SUBSET EVAL FAILED $lab $sub"
  done
done < $R/logs/r4_queue_done.txt

# ---- 3. cut arm base ---------------------------------------------------------------------
if [ ! -s $L/fmb_cut.txt ]; then
  if [ ! -s $L/sigma_cut.txt ]; then
    echo "=== sigma_r for the cut backbone $(date)"
    bak=$L/store_residual_backup; mkdir -p $bak
    cp -p $STORE/residual_stats.json $STORE/residual_std.npy $bak/ 2>/dev/null
    mkcfg $L/configs/fm_base_cut.yaml $W/configs/config_v4_adv_fm.yaml energy 0 1,1,1 \
      $META/full_train_cut50.1.txt $BB_CUT $SIG_FULL FMbase_cut_v4adv
    $P scripts/preprocess/compute_residual_stats.py $L/configs/fm_base_cut.yaml --backbone $BB_CUT \
      > $L/sigma_cut.log 2>&1 || { echo "SIGMA FAILED"; exit 1; }
    cp $STORE/residual_stats.json $L/residual_stats_cut.json
    cp -p $bak/residual_stats.json $bak/residual_std.npy $STORE/      # restore the shared file
    $E/bin/python -c "import json; print(repr(json.load(open('$L/residual_stats_cut.json'))['sigma_r_scalar']))" \
      > $L/sigma_cut.txt
  fi
  sig=$(cat $L/sigma_cut.txt); echo "sigma_r cut = $sig (full $SIG_FULL)"
  mkcfg $L/configs/fm_base_cut.yaml $W/configs/config_v4_adv_fm.yaml energy 0 1,1,1 \
    $META/full_train_cut50.1.txt $BB_CUT $sig FMbase_cut_v4adv
  echo "=== FM base, cut list $(date)"
  $P scripts/train/train_flow_matching.py $L/configs/fm_base_cut.yaml --backbone $BB_CUT \
    --data_percentage 100 --no_amp --seed 42 > $L/fm_base_cut.log 2>&1 || { echo "FM BASE FAILED"; exit 1; }
  grep -q "config FM_RESIDUAL_SCALE" $L/fm_base_cut.log || { echo "FM BASE SIGMA NOT FROM CONFIG"; exit 1; }
  rd=$(ls -td runs/sr_flow_matching/FMbase_cut_v4adv_* | head -1)
  echo "$PWD/$rd/fm_best.pth" > $L/fmb_cut.txt
  ev_fm clean_cut $PWD/$rd $BB_CUT fm_best.pth
fi

# ---- 4. cut arm, seed 42 -----------------------------------------------------------------
ft ctl_cut_s42  energy 0   1,1,1 cut 42
ft mink_cut_s42 energy 1.0 1,1,1 cut 42
for spec in "spectral spectral 1,1,1" "area energy 1,0,0" "wetarea wetarea 1,1,1" \
            "ssim ssim 1,1,1" "opticalflow opticalflow 1,1,1"; do
  set -- $spec
  ft ${1}_cut_s42 $2 $(parity $1 $2 $3) $3 cut 42
done

# ---- 5. seeds ----------------------------------------------------------------------------
for S in 43 44; do
  for arm in full cut; do
    ft ctl_${arm}_s$S  energy 0   1,1,1 $arm $S
    ft mink_${arm}_s$S energy 1.0 1,1,1 $arm $S
  done
done
echo "=== FM_R4_QUEUE_DONE $(date)"
