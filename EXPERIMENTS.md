# Experiment tracker

Last updated: 2026-09-13

**Read first:** `DECISIONS.md` (why things are the way they are, and what is already ruled
out), then `CLAUDE.md` (operational traps), then this file (what exists and what's next).
`MINKOWSKI_DOCS.md` documents the loss itself.

Conventions: paths relative to repo root. Deterministic evals on the full test set
(n = 285,383). Generative evals on 4,096 patches, M = 16, heun-16 unless noted.
**All tail columns at POT u = 31 mm/h** — rows at other levels are not comparable.

---

## 1. Checkpoints

### Deterministic backbones — `runs/sr_analytical/`

| Run | Loss | Weight | Status |
|---|---|---|---|
| `UNet_Ana_20260623_144958` | MSE only (vanilla) | 0 | **reference backbone** for all FM runs |
| `UNet_Ana_20260731_100128` | MSE + Minkowski | 1e-4 | **Study-1 winner** |
| `UNet_Ana_20260729_104637` | MSE + Minkowski | 1e-3 | **discard** — reward hacking (DECISIONS §2) |
| `UNet_Ana_20260812_085343` | MSE + spectral | shared | v1, inert |
| `UNet_Ana_20260817_113622` | MSE + ssim | shared | v1, inert |
| `UNet_Ana_20260829_033325` | MSE + spectral | per-loss | v2, still inert |
| `UNet_Ana_20260827_151319` | MSE + opticalflow | per-loss | v2, still inert |
| `UNet_Ana_20260826_041128` | MSE + wetarea | per-loss | v2, still inert |
| `UNet_Ana_20260824_155753` | MSE + ssim | per-loss | v2, still inert |

### Flow matching — `runs/sr_flow_matching/`

| Run | Backbone | σ_r | Status |
|---|---|---|---|
| `flow_matching_20260716_071518` | vanilla | 0.01898130588233471 | **clean reference model** |
| `Extremes_20260807_111521` | Minkowski (0731) | 0.019811755046248436 | Study-2 cell |

### Reward fine-tuning — `runs/sr_fm_minkowski/`

| Run | Mode | Status |
|---|---|---|
| `Extremes_20260804_170256` | distributional | first coupling; null-to-negative |
| `Energy_20260811_171310` | energy + retention (w=1.0) | **best coupling so far** |

---

## 2. tab:iid — in-distribution extremes (POT u = 31)

| Model | RMSE_ext | FSS@89 | RAPSD | Mink | Exc. ratio | RL bias | Anisot. |
|---|---|---|---|---|---|---|---|
| Bicubic | 3.77 | 0.000 | 4.00 | 2.85 | — † | — † | 30.9 |
| Backbone (MSE) | 3.59 | 0.040 | 3.10 | 2.12 | 0.118 | 0.643 | 1.87 |
| Backbone + spectral (v2) | 3.68 | 0.034 | 3.89 | 2.23 | 0.057 | 0.719 | 2.24 |
| Backbone + ssim (v2) | 3.68 | 0.039 | 3.83 | 2.23 | 0.059 | 0.701 | 2.14 |
| Backbone + wetarea (v2) | 3.72 | 0.032 | 4.59 | 2.42 | 0.036 | 0.748 | 2.57 |
| Backbone + opticalflow (v2) | 3.67 | 0.036 | 4.03 | 2.25 | 0.057 | 0.713 | 2.36 |
| Backbone + Minkowski | 3.70 | 0.159 | 1.31 | 0.74\* | 0.244 | 0.445 | 0.73 |
| FM (clean) | 3.01 | 0.289 | 0.79 | 0.84 | 0.478 | 0.165 | 1.31 |
| FM + Minkowski reward | 3.01 | 0.268 | 0.62 | 0.82\* | — ‡ | — ‡ | 1.31 |
| FM + Minkowski energy | 3.02 | 0.242 | 0.63 | 0.82\* | 0.523 | 0.176 | 1.31 |
| FM on Minkowski backbone | 3.04 | 0.148 | 0.81 | 0.82\*\* | 0.446 | 0.236 | 1.29 |

\* training objective, not independent evidence. \*\* conditional mean trained on the
Minkowski loss → partly circular. † bicubic fitted at another level; rerun at u=31.
‡ evaluated at u=53; rerun at u=31.

GPD ξ error — **secondary diagnostic only**, see DECISIONS §10:
MSE 0.029 · spectral 0.013 · ssim 0.009 · wetarea 0.007 · opticalflow 0.053 ·
Minkowski 0.099 · FM clean 0.079 · FM energy **0.043** · FM on Mink backbone 0.051.

## 3. tab:prelim — baseline metrics (full test set)

| Model | MAE | MAE_ext | Mink | Isoperim % | RAPSD | SAL S |
|---|---|---|---|---|---|---|
| vanilla | 0.0499 | 0.4030 | 2.12 | 4.57 | 3.05 | 1.27 |
| Minkowski (1e-4) | 0.0519 | 0.4195 | 0.74 | 3.53 | 1.30 | 0.34 |
| spectral v2 | 0.0502 | — | — | — | — | — |
| ssim v2 | 0.0503 | — | — | — | — | — |
| wetarea v2 | 0.0503 | — | — | — | — | — |
| opticalflow v2 | 0.0501 | — | — | — | — | — |
| FM (clean) | 0.056† | 0.423† | 0.83‡ | 4.29‡ | 0.79‡ | 0.50‡ |

† ensemble mean, ‡ single arbitrary member. FM row on a 25,600-patch subset.

**Activity check:** an auxiliary loss costing < ~1% MAE against vanilla has not entered the
objective; its structural columns say nothing. All four competing losses fail this at both
weights tried.

---

## 4. Tooling

| Script | Purpose |
|---|---|
| `scripts/hpc/launch_unet_analytical.sh` | train a backbone: `<params> <w_geom> <data_pct> [<config>]` |
| `scripts/evaluate/eval_extremes.py` | tail protocol → tab:iid row |
| `scripts/evaluate/eval_backbone.py` | baseline metrics → tab:prelim row |
| `scripts/evaluate/eval_fm.py` | FM baseline metrics |
| `scripts/hpc/eval_losses.sh` | batch-evaluate all backbones, with activity check + POT guard |
| `scripts/evaluate/make_plots.py` | all comparison figures from eval outputs |
| `scripts/evaluate/diagnose_coupling.py` | is the reward wired correctly? (minutes, one batch) |
| `scripts/hpc/sweep_reward.sh` | one-factor-at-a-time reward sweep |
| `scripts/hpc/check_sampler.sh` | sampler-fidelity check |
| `src/sweep_util.py` | `patch` a config / `collect` a summary into CSV |

Figures: perception–distortion plane, RAPSD with ratio panel, GPD return levels + exceedance
counts, tail/safety dot chart, SAL plane, **γ curves**, **γ residual by threshold**,
isoperimetric scatter.

---

## 5. TODO

### Highest value
- [ ] **Tail-sample selection.** Score a large patch pool by patch max, draw the *structural*
      loss batches from the top few %. Directly targets the pixel-mass bottleneck
      (DECISIONS §5, §13). Keep MSE / velocity loss on uniform batches.
- [ ] **Gradient-norm λ rule** for all auxiliary weights (DECISIONS §9).
- [ ] Re-run competing losses at a weight that passes the activity check, or conclude they
      cannot be made active without hacking.

### Paper
- [ ] Add the W₁ identity to the theory section (DECISIONS §13).
- [ ] Report exceedance ratio + RL bias as tail columns; demote ξ to the text with the
      confound explained.
- [ ] State the POT-level choice as estimability, not physics.
- [ ] Write up the w=1e-3 hacking episode as a positive result on necessary-not-sufficient.

### Data hygiene
- [ ] Rerun bicubic and `FM + Minkowski reward` at u=31 to fill their dashed tail columns.
- [ ] Delete duplicate eval dirs (`backbone_mse` = `backbone_vanilla`,
      `backbone_mink_1e-4` = `backbone_minkowski`, `*_v1` duplicates).
- [ ] Re-run evals so the new `gamma_target` / `pot_threshold` arrays exist for the γ plots.
- [ ] Eval `Energy_.../fm_mink_best.pth` to quantify the selection bug vs `latest`.
- [ ] Try `REWARD_FM_RETENTION_WEIGHT` 3 or 10 — FSS still fell at 1.0.

### Deferred
- [ ] b0 (persistent homology) confirmatory run at the winning weight
- [ ] o.o.d. protocol → tab:ood
- [ ] Compute proposal: the scale argument rests on DECISIONS §5–6
