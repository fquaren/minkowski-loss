# Shared settings for the FM G3 rows (EXPERIMENTS G3): sourced by fm_g3_parity.sh and run_fm_g3_queue.sh.
# Rows, seed 42, both arms: area-only (form quantile_mass_max, DECISIONS §46), pixel energy score,
# threshold-weighted pixel energy score, fss, windowed spectral. Weight: gradient-norm parity with the
# retention term on the full arm (DECISIONS §41), reused on the cut arm, as in run_fm_r4_queue.sh.
cd /work/fquareng/ch2/minkowski-loss/.claude/worktrees/fm-r4
E=/work/fquareng/.micromamba/envs/dl-stable
export LD_LIBRARY_PATH=$E/lib OMP_NUM_THREADS=1 PYTHONPATH=$PWD
W=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion
R=/work/fquareng/ch2/minkowski-loss/.claude/worktrees/rtx6000-node-detection-750d7c
META=/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta
OUTE=$W/eval_results/extremes
L4=$PWD/logs/fm_r4                     # the R4 queue's files (cut FM base, sigma_cut)
L=$PWD/logs/fm_g3; mkdir -p $L/configs
P="env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4 taskset -c 4-11 $E/bin/python -u"
EQM=$W/configs/config_v4_adv_fm_energy_qm.yaml

FMB_FULL=$W/runs/sr_flow_matching/Extremes_20261026_092630/fm_best.pth
BB_FULL=$W/runs/sr_analytical/UNet_Ana_20261021_211732/unet_latest.pth
SIG_FULL=0.01266805361956358
BB_CUT=$R/runs/sr_analytical/UNet_Ana_20261105_142458/unet_latest.pth       # r4_mse_cut_s42

# label mode channels form
ROWS=("area energy 1,0,0 quantile_mass_max"
      "pixel pixel 1,1,1 quantile_mass"
      "pixeltw pixel_tw 1,1,1 quantile_mass"
      "fss fss 1,1,1 quantile_mass"
      "spectralhann spectral_hann 1,1,1 quantile_mass")

mkcfg() {   # out mode weight channels form train_list backbone sigma name [KEY=VALUE ...]
  $E/bin/python - "$@" <<'PY'
import sys, yaml
out, mode, w, cw, form, train, bb, sig, name = sys.argv[1:10]
c = yaml.safe_load(open("/work/fquareng/ch2/minkowski-loss/.claude/worktrees/radar-day-exclusion/configs/config_v4_adv_fm_energy_qm.yaml"))
c["REWARD_MODE"] = mode
c["REWARD_WEIGHT"] = float(w)
c["MINKOWSKI_CHANNEL_WEIGHTS"] = [float(x) for x in cw.split(",")]
c["REWARD_FORM"] = form
c["TRAIN_METADATA_FILE"] = train
c["BACKBONE_CHECKPOINT"] = bb
c["FM_RESIDUAL_SCALE"] = float(sig)       # pinned: residual_stats.json holds another backbone's value
c["EXPERIMENT_NAME"] = name
for kv in sys.argv[10:]:
    k, v = kv.split("=", 1)
    c[k] = yaml.safe_load(v)
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
PY
}
