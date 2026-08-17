# Experiment tracker — Minkowski loss & extreme precipitation

Last updated: 2026-08-17

Conventions: all paths relative to repo root. Deterministic evals use the full test set
(n = 285,383). Generative evals use 4,096 patches, M = 16, heun-16 unless noted.
**POT level must match across rows to compare GPD ξ and RL bias.**

---

## 1. Checkpoints

### Deterministic backbones — `runs/sr_analytical/`

| Run | Loss | Weight | Status |
|---|---|---|---|
| `UNet_Ana_20260623_144958` | MSE only (vanilla) | 0 | done — reference backbone for all FM runs |
| `UNet_Ana_20260731_100128` | MSE + Minkowski | 1e-4 | done — Study-1 winner |
| (superseded) `UNet_Ana_20260729_104637` | MSE + Minkowski | 1e-3 | **discard** — reward hacking (filaments, +60% peak overshoot) |
| `UNet_Ana_20260812_085343` | MSE + spectral | 1e-4 | trained; eval done, weight suspect |
| — | MSE + ssim | | to run |
| — | MSE + wetarea | | to run |
| — | MSE + opticalflow | | to run |

### Flow matching — `runs/sr_flow_matching/`

| Run | Backbone | σ_r | Status |
|---|---|---|---|
| `flow_matching_20260716_071518` | vanilla | 0.01898130588233471 | done — clean reference model |
| `Extremes_20260807_111521` | Minkowski (0731) | 0.019811755046248436 | done — Study-2 cell |

### Reward fine-tuning — `runs/sr_fm_minkowski/`

| Run | Mode | Notes |
|---|---|---|
| `Extremes_20260804_170256` | distributional | first coupling; null-to-negative |
| `Energy_20260811_171310` | energy + retention | best coupling so far |

**σ_r trap:** `residual_stats.json` currently holds the *Minkowski-backbone* value (0.019812).
Any vanilla-backbone run must pin `FM_RESIDUAL_SCALE: 0.01898130588233471` explicitly
(`configs/energy.yaml` does). Always check the startup line reads `config FM_RESIDUAL_SCALE=...`.

**Checkpoint selection:** evaluate `fm_mink_latest.pth`, not `fm_mink_best.pth` — "best" is
selected on validation reward, i.e. the training objective, so it picks the most overshot model.

---

## 2. Results so far (tab:iid)

| Model | RMSE_ext | FSS@89 | RAPSD | Mink | ξ err | RL bias | POT |
|---|---|---|---|---|---|---|---|
| Bicubic | 3.77 | 0.000 | 4.00 | 2.85 | 0.974 | 0.595 | 53 |
| Backbone (MSE) | 3.59 | 0.040 | 3.10 | 2.12 | 0.327 | 0.432 | 53 |
| Backbone + spectral | 3.67 | 0.031 | 3.94 | 2.21 | 0.067 | 0.698 | **31** |
| Backbone + Minkowski | 3.70 | 0.159 | 1.31 | 0.74* | 0.102 | 0.271 | 53 |
| FM (clean) | 3.01 | 0.289 | 0.79 | 0.84 | 0.101 | 0.108 | 31 |
| FM + Minkowski reward | 3.01 | 0.268 | 0.62 | 0.82* | 0.176 | 0.130 | **53** |
| FM + Minkowski energy | 3.02 | 0.242 | 0.63 | 0.82* | **0.043** | 0.176 | 31 |
| FM on Minkowski backbone | 3.04 | 0.148 | 0.81 | 0.82* | 0.051 | 0.236 | 31 |

\* training objective, not independent evidence.

**POT inconsistency is the main data-hygiene debt.** Rows marked 53 vs 31 have
non-comparable ξ and RL columns. Pick one level and rerun the odd ones out.

---

## 3. Established findings (don't re-litigate)

- Minkowski loss **helps the deterministic backbone** on extremes: FSS@89 0.040 → 0.159,
  ξ err 0.327 → 0.102, RL bias 0.432 → 0.271, for +3.8% MAE.
- It **does not help the flow-matching model** by any route tried: as a reward (null to
  negative) or via the backbone (regression). Clean FM already reaches ξ 0.101 / RL 0.108.
- Coupling is **correctly wired**: reward drops 59.7% on a fixed-noise single batch, 5× the
  noise floor. K=1 gradient agrees with K=2 (cos +0.91). Not a plumbing bug.
- **The binding limit is pixel mass.** 83% of the structural gradient comes from thresholds
  ≤ 0.28 mm/h; thresholds ≥ 31 mm/h contribute ~0. Only 0.005% of pixels exceed 31 mm/h even
  on a 142 mm/h patch.
- **Anneal is not the lever.** Sweeping 0.05 → 0.40 raises the absolute gradient at 31 mm/h by
  11 orders of magnitude but the tail's *share* stays 0.75–0.88% and the bulk estimate degrades.
  Keep anneal 0.05–0.10.
- **Energy score is proper** (verified: minimised at true ensemble dispersion; per-sample
  prefers zero spread). It fixes the estimator, not the data volume.
- **The functionals carry no positional information**, so no variant can improve FSS. Every
  coupling that improves distributional metrics costs FSS.
- w = 1e-3 on the deterministic backbone → reward hacking (grid-aligned filaments, all
  structural metrics improve while fields are visibly wrong). Useful operating range is
  bounded above by hacking, not by accuracy cost.

---

## 4. TODO

### In progress
- [ ] Train UNet + {ssim, wetarea, opticalflow} at a common weight (scouting run)

### Next
- [ ] Weight sweep per loss. **Values differ by 3 orders of magnitude** — loss-magnitude
      parity with MSE (~3e-4) is roughly: spectral 2e-3, ssim 6.5e-4, wetarea 8e-2,
      opticalflow 1.5e-4. Sweep {0.3×, 1×, 3×} around each. A single shared weight is only
      a scouting run, not a comparison.
- [ ] Re-eval spectral backbone: the weight is suspect (RAPSD got *worse* under a loss that
      optimises the spectrum directly → mis-scaled). Also anisotropy 2.37 = highest of any
      model; check sample images for striping.
- [ ] Eval all UNet + loss runs (`eval_extremes.py --model backbone`, **fixed POT level**)
- [ ] Fine-tune + eval FM for each competing loss (mirror the Minkowski energy setup)

### Data hygiene (cheap, do before writing up)
- [ ] Unify the POT level across all rows in tab:iid
- [ ] Eval `Energy_20260811_171310/fm_mink_best.pth` to quantify the selection bug vs latest
- [ ] Try `REWARD_FM_RETENTION_WEIGHT` 3 or 10 — FSS still fell at 1.0, so the reward
      gradient likely dominates the retention term

### Deferred
- [ ] b0 (persistent homology) confirmatory run at the winning weight
- [ ] o.o.d. protocol → tab:ood
- [ ] CSCS proposal: scale argument rests on the pixel-mass finding above
