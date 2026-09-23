# minkowski-loss

Differentiable Minkowski functionals as a structural loss for km-scale super-resolution of
extreme precipitation (EUMETNET OPERA radar, 12.5x to 2 km, 128x128 patches).

Two studies: **Study 1** compares auxiliary structural losses on a deterministic backbone;
**Study 2** puts a flow-matching residual model on the best backbone and optionally couples
the structural loss as a reward. See `EXPERIMENTS.md` for the checkpoint table, current
results, and the todo list — read it before starting anything.

## Stack

Python / PyTorch, single-node multi-GPU (RTX PRO 6000). Config-driven: `config.yaml` holds
everything, scripts take a config path. `src/sweep_util.py patch` writes a modified config
without editing the original.

## Commands

```bash
# train deterministic backbone: <params> <weight_geom> <data_pct> [<config>]
bash scripts/hpc/launch_unet_analytical.sh configs/unet_analytical.yaml 1e-4 100.0 config.yaml

# evaluate: tab:iid row (tail protocol) / tab:prelim row (baseline metrics)
python scripts/evaluate/eval_extremes.py config.yaml --model backbone \
    --checkpoint runs/sr_analytical/<run>/unet_best.pth --pot_threshold 31 \
    --tag <label> --output_dir eval_results/extremes
python scripts/evaluate/eval_backbone.py config.yaml \
    --checkpoint runs/sr_analytical/<run>/unet_best.pth --output_dir eval_results/backbone/<label>

# batch-evaluate all competing losses with the activity check and POT guard
bash scripts/hpc/eval_losses.sh

# diagnose the reward coupling on one batch (minutes, not days)
python scripts/evaluate/diagnose_coupling.py config.yaml \
    --fm_checkpoint <ckpt> --backbone <ckpt> --extreme_batch --ensemble 16
```

## Operational traps — these have all cost days at least once

**Shared node: GPU 1 only, at most 8 of 12 cores** (EXPERIMENTS.md §0). GPU 0 belongs to
someone else and must never be used. At least 4 cores must always stay free. Both limits are
enforced in two places:
- `src/node_limits.py`, run on `import src`: pins the process to cores 4–11, refuses any
  `CUDA_VISIBLE_DEVICES` other than 1 or GPU 1's UUID, and checks the UUID if CUDA is
  already up.
- `scripts/hpc/env.sh`: pins the launcher shell to cores 4–11 and exports GPU 1.

Don't work around either. Size worker pools so that everything running at once (training,
DataLoader workers, the fetcher, audits) fits in 8 cores; they all share cores 4–11.

**GPU selection.** `CUDA_VISIBLE_DEVICES=1` alone is not enough: CUDA's default device order
is not `nvidia-smi`'s, and the two cards are identical, so the logs would not show a wrong
pick. Always set both variables. Setting them inside the `micromamba run` wrapper is the
safe habit (as of 2026-09-23, `micromamba run` passes them through unchanged):

```bash
micromamba run -n dl-stable env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python ...
```

**`python` may not exist** — use `python3`, or resolve it with `command -v`.

**Orphaned DataLoader workers.** Launch long runs under `setsid` (or Slurm). A tmux exit
leaves the worker pool unreaped; the node has previously accumulated ~193 zombies plus stuck
root NFS/RPC threads. Prefer `persistent_workers=True` and modest `num_workers`.

**AMP.** fp16 overflows once the residual is correctly scaled — train the flow-matching model
with `--no_amp` (fp32) or bf16, never fp16.

**sigma_r must match the backbone.** `residual_stats.json` is overwritten by whichever
backbone `compute_residual_stats.py` last ran against, and it currently holds the
*Minkowski-backbone* value (0.019812). Any run on the vanilla backbone must pin
`FM_RESIDUAL_SCALE: 0.01898130588233471` explicitly. **Always check the startup line reads
`config FM_RESIDUAL_SCALE=...`, not `residual_stats.json`.** A silent mismatch here once
produced a 53x reconstruction error that looked like a catastrophic model failure.

**POT level must be identical across every row of a table.** `gpd_xi_err` and `rl_bias` are
computed against a fit whose observed reference depends on it. Use `--pot_threshold 31`
everywhere. Sanity check: `gpd_xi_obs` depends only on the data, so it must be the same in
every summary (u=31 -> +0.176; u=53 -> -0.190).

**Evaluate `fm_mink_latest.pth`, not `fm_mink_best.pth`.** "Best" is selected on validation
reward, i.e. the training objective, so when the tail overshoots it picks the most overshot
checkpoint.

**`val_mse` is dry-pixel-dominated** and saturates by epoch 2. Never judge a run from it;
judge from the eval outputs.

## Reading results

Check in this order:

1. **Activity.** MAE against the vanilla backbone (0.04994). Below ~+1% the auxiliary loss
   never entered the objective and its structural columns say nothing. Minkowski at its
   working weight pays +3.8%.
2. **Anisotropy** (vanilla 1.87). A structural gain paired with a materially higher value is
   the grid-aligned filament pathology — no headline metric detects it. Look at the sample
   images before believing such a row.
3. **Tail columns: exceedance ratio and RL bias.** Not the GPD shape error — xi and sigma
   trade off, so a model with almost no tail can score a near-perfect xi. Three separate runs
   have the best xi in the table alongside the worst tail.
4. **`minkowski_distance` is circular** for any Minkowski-trained model; it is its objective.
5. **`rmse_extreme` is a weak discriminator** — it varies ~1% across samplers that differ 46%
   in peak height.

## Established findings — do not re-litigate

- The Minkowski loss **helps the deterministic backbone** substantially (FSS@89 0.040 to
  0.159, exceedance ratio 0.118 to 0.244) for +3.8% MAE.
- ~~It does not help flow matching.~~ **Reopened 2026-09-23** (DECISIONS §3 status note):
  after the 09-16 re-eval, the distributional reward beats clean FM on FSS, exceedance
  ratio and RL bias on the same 4,096 patches. Single seed, leaking split, cause of the
  shift not yet attributed, so do not cite either version as settled.
- The coupling is **correctly wired** — the reward drops 59.7% on a fixed-noise single batch,
  5x the noise floor; K=1 agrees with K=2 (cos +0.91). Not a plumbing bug.
- **The binding limit is pixel mass**: 83% of the structural gradient comes from thresholds
  <= 0.28 mm/h, and >= 31 mm/h contributes ~0, because only 0.005% of pixels exceed 31 mm/h
  even on a 142 mm/h patch.
- **Anneal is not the lever.** Sweeping 0.05 to 0.40 raises the absolute tail gradient by 11
  orders of magnitude while its *share* stays at 0.75-0.88%. Keep anneal 0.05-0.10.
- **The functionals carry no positional information**, so no variant improves FSS.
- At w=1e-3 the deterministic model **games the loss** — grid-aligned filaments, +60% peak
  overshoot, every structural metric improving. The useful range is bounded above by hacking,
  not by accuracy cost.

## Data and split caveats (2026-09-23)

- **Splits are random at patch level** (DECISIONS §15). Every test patch has a train patch
  at the same timestamp and on the same tile within ±1 h. Test scores are interpolation
  scores. Never make an o.o.d. / unseen-extreme claim on them.
- **The tail contains radar artefacts, and the declutter step zeroes >150 mm/h** instead of
  clipping. See EXPERIMENTS §5 and `scripts/data_quality/`. Look at the images before
  believing anything driven by the top of the distribution.
- **The competing-loss rows are not a fair trial** (DECISIONS §8 status, §16): three were
  never active, SSIM is buggy, and the budgets were unmatched.
- **DDPM is dropped** (DECISIONS §14). Do not add DDPM rows or fix DDPM code unless asked.

## Conventions

- Never delete or alter existing functionality in scripts without being asked.
- Keep each new experiment's results under its own `--tag` / `--output_dir`; do not overwrite
  an existing eval directory.
- Structural-loss weights are per-loss and differ by orders of magnitude — never reuse one
  loss's weight for another.
- Always report the weight and loss name from the first lines of the training log when
  summarising a run.
