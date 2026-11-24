# Digest E: DECISIONS §37–§43 and selected notes

Sources: `DECISIONS.md` lines 2472–3386; `notes/status_2026-10-19.md`, `notes/status_2026-10-21.md`,
`notes/minkowski_deterministic_limitation.md`, `notes/minkowski_reward_intensity.md`,
`notes/weight_selection_literature.md`, `notes/events.md`, `notes/future_project_radar_cleaning.md`;
memory files in `~/.claude/projects/-work-fquareng-ch2-minkowski-loss/memory/`. Nothing under `private/` was opened.
Abbreviations as in the sources: thr = threshold form, qm = `quantile_mass` form, valext = `val_extremes1024`,
valsub = `val_subset4096`, RL = return level.

---

## §37. The structural-loss weight on hourly data (2026-10-21, decisions through 2026-10-26)

**Question.** `STRUCTURAL_LOSS_WEIGHTS.minkowski = 1e-4` was validated on 15-min rates (active at +3.8% MAE, below
the hacking onset at 1e-3; §2). The hourly corrected target differs in scale, scaler and threshold grid, so the
weight must be re-established.

**Method.**
- Step 1, calibration by gradient share with `tools/gradient_audit.py`, same settings as the 15-min audit of
  2026-09-23 (512 x 128 patches, anneal 0.1, 10% data, jackknife errors).
  - 1a at initialisation: parity lambda* = ||grad MSE|| / ||grad L_mink||; w_init = 1e-4 x lambda*_hourly / lambda*_15min
    (15-min lambda* = 1.05e-3).
  - 1b at the trained hourly vanilla checkpoint with its Adam moments: weight giving the same Adam share as 1e-4 at
    the 15-min vanilla checkpoint (Adam share 16.1). **1b governs**; >3x disagreement is reported as a finding.
- Step 2, bracketing sweep {w/3, w, 3w, 10w}. A later note (same day) replaced the loose "largest active,
  non-gaming weight" wording with the existing §16 / RESEARCH_NOTES §5 rule: **best exceedance ratio with ΔMAE <= +5%
  and anisotropy <= 1.2x vanilla**, last checkpoint, then 3 seeds, whole bracket reported; ΔMAE >= +1% as activity check.
- Literature (web search 2026-10-21, not systematic), as written: "gradient balancing is established (GradNorm,
  Chen et al. 2018; Wang, Teng & Perdikaris 2020; Zhong & Maki 2020, closest: weight from a target gradient-norm
  ratio). No prior found for transferring a fixed weight across datasets by gradient share, nor for a hacking-bounded
  weight selection in precipitation SR. **No contribution is claimed before a systematic search**."
- Follow-up kept, not adopted: setting the weight every step from the gradient share.

**Result 1a (`eval_results/gradient_audit/2026-10-21_v4adv_init/`).** Hourly lambda* = 8.90e-4 (± 3.7e-6) vs 1.05e-3
(± 1.4e-5) on 15-min. **w_init = 8.5e-5 (± ~0.1e-5).** At 1e-4 the share is 0.112 hourly vs 0.095 15-min (cos with MSE
gradient +0.86 vs +0.78). 1e-4 carries over within 15% at initialisation.

**Raw vs corrected climatology (2026-10-21; `scripts/data_quality/rate_climatology.py`, every 7th day 2016–2025,
520 days, 49,864 frames).** Screening + radar exclusions remove 31.7% of 15-min exceedances >= 31 mm/h, 2.0% of those
>= 1 mm/h, 0.60% of valid pixel-frames. Spokes largely removed: hrdeb 13.9x -> 2.6x median radar (82% removed), rsfrg
37x -> 2.5x (94%), skjav 8.9x -> 3.8x (59%). rsjas 2023 not handled: raw 3.85e-3 (other years 1.9e-5 to 1.6e-4),
corrected 2.07e-3 (49% removed), ~60x median. hubud not anomalous (1.6x raw, 1.3x corrected). Highest after
correction: rsjas, robuc (11% removed), rocra, hrulj, other Romanian radars at 4–5x (partly climate; no jump at
Romania's borders).

**Exclusion decisions.**
- Researcher 2026-10-21: **rsjas 2023 excluded** (owner share >= 10%); `quality_v4/biased_radar_years_all.csv` = §36b's
  three + ltlau/ltvil 2026 + rsjas 2023, applied by `scripts/hpc/exclude_then_vanilla.sh`. Dry run expected 806 train,
  107 val, 312 test tile-hours (24% with max >= 20.2 mm). Vanilla run started 14:57 (`logs/unet_ana_20261021_145706.log`)
  stopped at 18:00; directory kept, not a result.
- Why the screen missed rsjas 2023 (`scripts/data_quality/why_missed.py`, `notes/data_analysis_v4/why_missed_rsjas_2023.md`):
  rule 5 never reached score 1 (worst 0.90) because the iso31 reference is the median of five neighbours (rsfrg, hrgra,
  hrulj, husze, huhar), three noisy on the same days; ceiling/f150 references include rsjas's own inflated q99.
  Pixel rules: ~125,000 of ~200,000 raw pixel-frames >= 31 mm/h per day stay; noise is clustered, moving, surrounded and
  temporally supported. 1,226 rsjas 2023 tile-hours were in the lists, 5.4% with a 15-min rate >= 150 mm/h (2021–22: 0.8–1.1%).
- Researcher 2026-10-21 21:00: **hrulj 2023 excluded** (corrected ~11x median vs 1.7–2.1x in 2024–25). Removed 2,666 train,
  198 val, 997 test tile-hours. **Final lists: train 1,299,686 / val 172,050 / test 212,803 / nimbus 378,567**; scaler
  unchanged (train max 613.3 mm). Vanilla restarted 18:11 stopped at 21:06.
- Systematic radar-year scan (`notes/data_analysis_v4/rate_clim_per_radar_year.csv`, 1,545 radar-years; median radar-year
  2.5e-5): 69 have x_own >= 3 and x_all >= 3. Summer-concentrated (consistent with convection): fikor/fianj 2025, frpla
  2021, fipet 2024, frtre 2022, deasb 2019, deeis 2022, frcae 2025, eesur 2021, frbou 2020, frnim 2022, plram 2024.
  Winter-heavy: mtgud 2017 (x_own 22, winter 0.64), esclg 2025 (7.6, 0.53), ukham 2025 (6.5, 0.50); isegs 2017 borderline.
- Researcher 2026-10-22: **mtgud 2017, esclg 2025, ukham 2025 kept** (`radar_year_review.pdf`), stated as residuals.
  Exclusions final at 7 radar-years.
- Researcher 2026-10-22, §35: **advection-corrected target accepted on fast-growing convection**; no growth-aware motion estimate.

**Step 2 decisions (2026-10-22).** §16 rule; bracket {w*/3, w*, 3w*, 10w*}; full budget 25 epochs (~18 h each); same
seed; 3 seeds at the chosen weight; choose on validation; test only for reporting. §34 amendment (pre-break ablation):
both arms select on full validation, last checkpoint, post-break test rows. Launch via `scripts/hpc/run_bracket_v4adv.sh`.

**POT level on the corrected store (2026-10-22; `notes/data_analysis_v4/pot_threshold_adv.md`).** The §32 rule gives no
level: u = 100 mm (6,898 exceedances, 646 days) has sigma* = -8.9 [-26, 12.6], xi = 0.30, outside every lower interval.
31.7–80 mm stable: xi 0.18–0.21, sigma* 2.9–6.4. At 50.1 mm train 99,933 (1,702 days), test 14,480 (305 days), xi
0.19/0.16; at 31.7 mm train 427,828 (2,366 days), test 77,262 (392 days), xi 0.19/0.18. **Decision (researcher 2026-10-24):
u = 50.1 mm kept** (option a; rule not amended). Rejected: amend the minimum count to get 31.7 mm (chosen after seeing fits).

**Result 1b (2026-10-22; `eval_results/gradient_audit/2026-10-22_v4adv_vanilla/`).** Checkpoint `UNet_Ana_20261021_211732`
(best = epoch 24). Adam share at 1e-4 = 8.83 (15-min 16.13), raw share 6.0 (15-min 24), cos with MSE -0.14 (15-min -0.72).
**w* = 1e-4 x 16.127 / 8.833 = 1.8e-4**, 2.1x step 1a, below the 3x bar; 1b governs. Bracket order 1.8e-4, 5.4e-4,
1.8e-3, 6.0e-5. Runs: 1.8e-4 `UNet_Ana_20261022_160540`, 5.4e-4 `UNet_Ana_20261023_121956`.

**Bracket on validation (2026-10-26; last checkpoints, u = 50.1 mm; `notes/data_analysis_v4/bracket_validation.csv`,
`bracket_gallery.pdf`).**

| model | ΔMAE | anisotropy (x vanilla) | exc. ratio | RL bias λ10/100/1000 | FSS 50.1 mm 1/21 px | peak ratio (extreme) |
|---|---|---|---|---|---|---|
| vanilla | 0 | 2.12 (1.00) | 0.113 | -44/-31/-13% | 0.106/0.383 | 0.36 (0.20) |
| 6.0e-5 | +4.7% | 0.56 (0.26) | 0.192 | -37/-41/-44% | 0.097/0.417 | 0.86 (0.52) |
| 1.8e-4 (w*) | +7.9% | 2.24 (1.05) | 0.287 | -32/-36/-41% | 0.118/0.535 | 0.86 (0.53) |
| 5.4e-4 | +12.5% | 2.02 (0.95) | 0.442 | -25/-32/-39% | 0.131/0.648 | 0.93 (0.58) |
| 1.8e-3 | +19.0% | 3.45 (1.62) | 0.607 | -18/-21/-24% | 0.125/0.714 | 0.92 (0.61) |

gpd_xi_obs = 0.183 every row. By the §16 rule the winner is **6.0e-5**, the only run within +5% MAE. The anisotropy cap
misses the visible grid-aligned striping (mild at 1.8e-4, clear at 5.4e-4, strong at 1.8e-3); vanilla itself is 2.12 and
per-patch ratios are noisy. Tail mixed: more weight gives more exceedances, higher peaks, better FSS@21 px, but every
Minkowski run under-predicts λ = 100, 1000 more than vanilla (fitted xi 0.12–0.17 vs observed 0.18; vanilla 0.32).
"Helps the tail" holds for counts and peaks, not return levels.

**Status (2026-10-26).** Researcher stopped the 6.0e-5 seed runs (queue killed 08:21) and then the pre-break ablation's
post-break arm (killed ~08:25, epoch 1 of 33; directory not a result). Study-1 hourly weight and the pre-break ablation
reopened; nothing running. (The weight was later fixed in §43.)

---

## §38. Study 2 on hourly data: flow matching and its Minkowski fine-tunes (2026-10-26 to 10-30)

**Question.** Does the FM residual model still work on the hourly corrected target, and how do its two Minkowski fine-tunes behave?

**Plan (approved 2026-10-26).** Backbone `UNet_Ana_20261021_211732`, **last** checkpoint (epoch 25). Step 0 sigma_r pinned
in a dedicated config, heun, 16 steps, fp32 (`--no_amp`), new launcher, 1-epoch smoke run. Step 1 FM training, `fm_best.pth`
evaluated. Step 2 fine-tunes after coupling checks: (a) distributional K = 1, M = 4, w = 1e-3, proximal 1, spread 0.5 (as
`Extremes_20260804_170256`); (b) energy M = 16, w = 1e-3, retention 1.0 (as `Energy_20260811_171310`); evaluated from
`fm_mink_latest.pth`. Step 3 evaluation on fixed subsets (16 members, heun-16, u = 50.1 mm). Then the 2e-5 backbone,
Study-1 weight, pre-break ablation.

**Striping measure (`src/evaluation/striping.py`, `scripts/evaluate/striping_eval.py`, `notes/data_analysis_v4/striping_bracket.csv`;
1,024 val patches >= 1 mm).** Version 1 (raw gradient orientation; spectral anisotropy) fails. Version 2 (high-pass texture
orientation inside rain eroded 3 px, share within ±5° of axes, pred minus target, median) is monotone: vanilla 0.017, 6e-5
0.038, 1.8e-4 0.048, 5.4e-4 0.049, 1.8e-3 0.098 (heavy patches 0.016/0.016/0.022/0.029/0.067). No diagonal striping.
Spectral-line variant unusable. Not yet adopted; threshold to be fixed before use.

**Why the texture has a fixed orientation (`scripts/evaluate/texture_orientation.py`, 64 heavy patches).** Perimeter
(4-direction Cauchy-Crofton, w_ax 0.473, w_di 0.218): ellipse underestimated 4.8% on-axis, overestimated ~1.5% at 25–35°/55–65°.
Euler (4-connected cubical): 1-px line chi = 1 on-axis, up to 212 at 45°; no effect from 1.5 px width. Rotation test: texture
correlates with original at 0.04 (5.4e-4), 0.11 (1.8e-3). Target directions near uniform (peak 0.062 per 10°, uniform 0.056);
1.8e-3 peaks at 92.5° with 3x uniform, at 2.5° after rotation (locked to grid). Lower weights 0.073–0.083; single test.

**Step 0.** sigma_r (hourly vanilla, last checkpoint, 1,299,686 train patches) = **0.012668** (15-min 0.018981); std-map CV
0.064. Pinned in `configs/config_v4_adv_fm.yaml`; startup line `config FM_RESIDUAL_SCALE=0.01267`.

**Step 1.** Smoke run `Extremes_20261026_091536` (not a result): ~1.85 h/epoch. Full run 50 epochs max, PATIENCE 5. Eval subsets
(seed 0): `val_subset4096.txt` (279 with tile max >= 20.2 mm, 38 >= 50.1) and `val_extremes1024.txt` (tile max >= 20.2 mm; 187
>= 50.1). Result (2026-10-29): `runs/sr_flow_matching/Extremes_20261026_092630` early-stopped at epoch 33, best val_fm
**0.2323 at epoch 28**; train_fm 0.320 -> 0.239; 64 h total.

**FM on hourly data works (2026-10-29; `notes/data_analysis_v4/fm_hourly_validation.csv`).**

| subset | model | exc. ratio | FSS 50.1 1/21 px | RL bias λ10/100/1000 | peak ratio (ext) | RAPSD dist | anisotropy |
|---|---|---|---|---|---|---|---|
| random 4,096 | vanilla | 0.055 | 0.052/0.138 | (too few) | 0.36 (0.20) | 3.57 | 1.84 |
| | 6e-5 | 0.179 | 0.047/0.248 | | 0.85 (0.54) | 1.33 | 0.51 |
| | FM | 0.272 | 0.130/0.466 | | 0.86 (0.55) | 0.86 | 1.40 |
| heavy 1,024 | vanilla | 0.046 | 0.041/0.100 | (too few) | 0.24 (0.10) | 3.86 | 1.53 |
| | 6e-5 | 0.162 | 0.067/0.401 | -39/-41/-43% | 0.65 (0.28) | 1.44 | 0.36 |
| | FM | 0.298 | 0.114/0.506 | -30/-36/-44% | 0.55 (0.33) | 1.14 | 1.31 |

gpd_xi_obs 0.155 heavy, -0.032 random. FM best on exceedances, FSS, spectrum; tail still under-predicted.

**Coupling checks (extreme batch, fixed noise, 40 steps).** distributional: 0.271 -> 0.195 (-28.1%), change 0.076 vs floor 0.071
(marginal pass; 15-min was -59.7%, 5x floor); cos(K=1,K=2) 0.97, (K=1,K=4) 0.90. energy: 0.157 -> 0.106 (-32.5%), 0.051 vs 0.027
(1.9x); cos 0.97/0.85. Gradient almost all from u <= 0.83 mm, zero from 20.2 mm; 100% of field gradient in top 1% of pixels.

**Distributional fine-tune (`FMdist_v4adv_20261029_014713`, epoch 10, 10 x 500 steps, 2 h 51 min).** Reward barely moved
(train 0.351 -> 0.339); guards within bounds.

| subset | model | exc. ratio (pred/obs) | FSS 31.7 21 px | FSS 50.1 1/21 px | RL bias | peak ratio (ext) | RAPSD | anisotropy | CRPS |
|---|---|---|---|---|---|---|---|---|---|
| random | FM | 0.272 (79/290) | 0.508 | 0.130/0.466 | | 0.86 (0.55) | 0.86 | 1.40 | 0.0324 |
| | FM + dist | 0.517 (150/290) | 0.682 | 0.145/0.538 | | 0.92 (0.59) | 0.35 | 1.37 | 0.0321 |
| heavy | FM | 0.298 (458/1,537) | 0.707 | 0.114/0.506 | -30/-36/-44% | 0.55 (0.33) | 1.14 | 1.31 | 0.161 |
| | FM + dist | 0.377 (580/1,537) | 0.741 | 0.109/0.554 | -23/-20/-15% | 0.62 (0.22) | 0.81 | 1.30 | 0.160 |

Attribution: only other term is the proximal anchor (zero at init), so the change is the reward's; why a light-rain-dominated
gradient moves the >50 mm tail is not established (inference: more intermittent, contrasted fields).
**Eval noise** (second FM draw, heavy): exceedances 458 vs 393 (dist 580); RL bias -33/-36/-36%; FSS 31.7 0.707; FSS 50.1
0.103/0.523; peak ratio 0.54 (0.20); RAPSD 1.19; CRPS 0.1614 (0.335); gpd_xi_pred 0.047 vs 0.186. Verdict: gains exceed eval
noise on 100/1000-yr RL, FSS 31.7, spectrum, CRPS; exceedance count ~3 noise sd; extreme peak-ratio loss withdrawn as noise;
xi_pred unusable. Eval noise only, not training-seed noise.

**Energy fine-tune (`FMenergy_v4adv_20261029_043819`, recorded 2026-10-30).** Reward did not move (0.199 -> 0.205; reward 0.13%
of gradient, §39). Heavy: exceedances 411, RL bias -31/-33/-33%, FSS 31.7 0.726, FSS 50.1 0.086/0.469, RAPSD 1.24, anisotropy
1.28, CRPS 0.1615. Random: 66 (0.228), FSS 31.7 0.634, FSS 50.1 0.124/0.454, RAPSD 0.97. Checkpoint assumed
`fm_mink_latest.pth` (not confirmed). **Verdict: FM + energy as configured is continued FM training**; hourly tail gains came
from the distributional fine-tune only.

---

## §39. The reward must reach heavy rain: quantile area + inverse-mass weights (2026-10-29)

**Problem.** Through the soft mask a pixel feels u only within a few tau = 0.005u; a threshold's total pull is N p(u),
independent of tau (why the §5 anneal sweep could not move shares); thresholds above the prediction's reach give no gradient.
On clean hourly FM, 73% of the reward gradient lands on pixels < 1 mm (heavy subset), 3.7% on >= 20 mm
(`logs/reward_threshold_share_v4adv.txt`).

**Decision (researcher).** Keep K = 1. New `REWARD_FORM: quantile_mass` (`src/trainers/reward_finetune.py:reward_features`;
`threshold` stays default): (i) area compared horizontally, log1p of the n_j-th largest predicted vs target value, n_j =
#{target >= u_j}, masked above target maximum: one pixel per populated threshold gets gradient c = 6.42; (ii) perimeter and
Euler weighted by (n_j + 1)^-1, normalised to mean 1, capped above target maximum. Ranks/weights depend on target only, so
energy score stays proper on the features. **Rejected:** weights alone (no gradient out of reach); heavy-tailed surrogate
backward pass (new hyperparameter, invites gaming). Configs `configs/config_v4_adv_fm_{dist,energy}_qm.yaml`.

**Checks.** `tests/test_reward_form.py` 13 pass. Share >= 20 mm: 3.7% -> 40.9%; < 1 mm 73% -> 16% (alpha = 1; alpha = 0
almost no change). Coupling: distributional -31.6% (floor 0.108, noise 18.9% of mean, was 11.4%), energy -35.1% (floor 0.024).
Reward is 96% of dist gradient but **0.07% of energy gradient** -> energy needs ~x1000 on w_reward (§9 gradient-norm rule).
Original-form energy fine-tune: reward 0.13% (cos -0.01); dist fine-tune 99% reward.

**Open before fine-tune:** fix proximal anchor (was evaluated at clamp(field/6.42, 0, 1)); run w_reward = 0 control.
Write-ups `notes/minkowski_reward_intensity.md`, `notes/minkowski_reward_coupling.pdf`.

---

## §40. Where the base loss looks, and whether the tail is too noisy for the reward (2026-10-30)

At clean hourly FM (`Extremes_20261026_092630/fm_best.pth`), validation subsets, one seed.

**(a) FM loss by target intensity (`scripts/evaluate/fm_loss_intensity_share.py`, `logs/fm_loss_intensity_share_v4adv.txt`).**
Shares in % (pixels / loss / |dL/dv|; "param" = projection of per-bin parameter gradient on total, sums to 100%):

| target mm | heavy pix/loss/\|dL/dv\| | heavy param | random pix/loss/\|dL/dv\| | random param |
|---|---|---|---|---|
| < 1 | 92.3/51.2/69.9 | +327 | 97.5/68.7/84.3 | +65 |
| 1–10 | 7.2/41.7/27.3 | -170 | 2.4/29.5/15.2 | +38 |
| 10–20 | 0.44/4.9/2.1 | -37 | 0.046/1.4/0.43 | -1.9 |
| 20–50 | 0.12/2.0/0.66 | -17 | 0.009/0.34/0.09 | -1.5 |
| >= 50 | 0.009/0.28/0.06 | -2.5 | 0.0005/0.03/0.007 | -0.2 |

Not proportional to pixel mass (corrects the §39-era statement): tail share 5–10x its pixel share, but still ~1% (pixels >= 20 mm
carry 0.72% of |dL/dv| heavy, 0.10% random). Every bin >= 10 mm has negative projection: a step along the total gradient raises
loss on heavy pixels. Original threshold reward put 3.7% of |dR/dz| on >= 20 mm, ~5x the FM loss's 0.72% (inference, untested:
may explain why the dist fine-tune moved the tail).

**(b) Tail signal/noise (`scripts/evaluate/tail_reading_snr.py`, M = 16, 256 heavy / 512 random).** b/s heavy:

| u mm | members max < u | count | rank | perim | euler |
|---|---|---|---|---|---|
| 3.43 | 2% | -0.53 | -0.73 | -0.82 | -0.91 |
| 8.61 | 15% | -1.32 | -1.97 | -1.34 | -1.36 |
| 20.2 | 55% | -1.86 | -2.04 | -1.90 | -1.92 |
| 50.1 | 73% | -2.47 | -2.77 | -2.58 | -2.51 |
| 79 | 79% | -3.28 | -3.14 | -3.28 | -3.22 |

Deficit detectable in every channel; noise did not hide the tail. Rank only 1.1–1.5x better. **Withdrawn:** earlier "rank ~1/xi
(~6x) more stable" claim. What separates forms is **reach** (55% of members never reach 20.2 mm, 73% never 50.1 mm). Synthetic
check (`tail_reading_synthetic.py`, GPD xi = 0.155, x0.7): rank 1.4–1.9x count with independent pixels, ~even with 4-px
correlation; masking n_j = 0 biases a calibrated model at n ~ 5 (-1.6 sd count, -1.2 sd rank).

**Working hypothesis (researcher).** Base loss gives heavy rain ~1% of its gradient and trades it against the bulk; an auxiliary
objective helps the tail if (i) its high-intensity gradient share exceeds the base loss's and (ii) it gives gradient to samples
short of the target's tail. Consistent with (a)–(b) and §38; not tested across losses.

---

## §41. Corrected runs: anchor fix, energy weight, control, U-Net quantile form (2026-10-30)

**Order (researcher):** (1) FM + dist qm; (2) FM + energy qm; (3) U-Net + Minkowski qm.

**Anchor fix** (`src/trainers/reward_finetune.py:proximal_anchor`, `tests/test_proximal_anchor.py`): evaluated at x_t =
(1 - t) x0 + t r_gen (first member's residual in r/sigma_r, fresh x0), same marginal law as sampler states. Old clamp surrogate
removed. `diagnose_coupling.py` not affected; §39 result stands.

**Gradient budget with fixed anchor** (`logs/grad_norm_terms_v4adv_anchorfix.txt`): dist_qm reward 5.0e-4, anchor 8.3e-5, total
5.2e-4, share 95%, cos +0.96. energy_qm (w 1e-3): reward 5.1e-4, retention 0.65, share 0.07%.

**Energy weight:** parity gives 1e-3 x 0.649 / 5.05e-4 = 1.29; set **REWARD_WEIGHT 1.0** (`config_v4_adv_fm_energy_qm.yaml`,
reward ~44% of gradient norm). Caveat §9: per-batch FM gradient at a trained checkpoint is mostly minibatch noise.

**Control** `config_v4_adv_fm_energy_ctl.yaml`: REWARD_WEIGHT 0, retention 1 + anchor 1, M = 16, seed 42, 10 x 500 steps. No dist
control (it would not move).

**Launched 2026-10-30 10:15** (`scripts/hpc/run_qm_finetunes_v4adv.sh`): dist_qm, energy_ctl, energy_qm, each evaluated.

**U-Net port** (`src/losses/competing.py:QuantileMassMinkowskiLoss`, key `MINKOWSKI_FORM`, `tests/test_minkowski_quantile_det.py`
6 pass); `configs/config_v4_adv_qm.yaml`. Five pre-existing failures in `tests/test_losses.py` / `test_analytical_minkowski.py`
unchanged. Parity at init (`eval_results/gradient_audit/2026-10-30_v4adv_qm_init/`): lambda*_qm = 1.17e-3 (± 8.4e-6) vs 8.90e-4
threshold, gradient 1.31x smaller, cos +0.82. Bracket queued (`scripts/hpc/run_bracket_qm_v4adv.sh`), §37 rule, w*/3 first;
predicted w*_qm ~ 2.4e-4.

---

## §42. Results of the corrected FM runs (2026-11-02 to 11-05)

**Runs** (all `config FM_RESIDUAL_SCALE=0.01267`, GPU 1, `fm_mink_latest.pth`, 16 members, heun-16, u = 50.1 mm):
dist_qm `FMdistqm_v4adv_20261030_101504` (M=4, w 1e-3, prox 1, retain 0, spread 0.5); energy_ctl `FMenergyctl_v4adv_20261030_141741`
(M=16, w 0, prox 1, retain 1); energy_qm `FMenergyqm_v4adv_20261031_022957` (M=16, w 1, prox 1, retain 1).

**valext (1,537 obs exceedances; gpd_xi_obs +0.155).**

| Row | RMSE | FSS@50.1 w5 | RAPSD | Exc. ratio | RL bias | Anisot. | Peak ratio ext | CRPS_ext |
|---|---|---|---|---|---|---|---|---|
| FM clean (draw 1/2) | 1.153/1.152 | 0.285/0.287 | 1.14/1.19 | 0.298/0.256 | 0.367/0.353 | 1.31/1.27 | 0.33/0.20 | 0.336/0.335 |
| FM + dist (thr, §38) | 1.152 | 0.319 | 0.81 | 0.377 | 0.194 | 1.30 | 0.22 | 0.333 |
| FM + dist qm | 1.152 | 0.267 | 1.20 | 0.273 | 0.340 | 1.32 | 0.29 | 0.337 |
| energy control | 1.153 | 0.263 | 1.30 | 0.187 | 0.318 | 1.36 | 0.28 | 0.338 |
| **FM + energy qm** | 1.160 | 0.348 | 0.41 | 0.665 | 0.089 | 1.36 | 0.34 | 0.332 |

valsub (290 obs exceedances): exceedance ratio FM 0.27, dist 0.52, dist_qm 0.25, control 0.29, energy_qm 0.91; RAPSD
0.86/0.35/0.83/1.01/0.25; FSS@50.1 0.337/0.419/0.310/0.264/0.246; RMSE 0.336/0.336/0.335/0.336/0.339.

**Reading.** energy_qm is the first FM coupling with a large tail effect, attributable to the reward (control does not move the
tail; also confirms the §38 energy row was continued training). Exceedance gain 0.28 -> 0.67 ~10x clean-FM draw-to-draw (0.04).
RMSE +0.6%, CRPS_ext unchanged/better, anisotropy = control's. FSS@50.1 disagrees between subsets (valsub noisy). dist_qm did not
improve on FM and is worse than thr dist; untested differences from energy_qm: no retention, w 1e-3, M = 4, spread term.

**Field panels** (`figures/v4adv_valext_fields_qm/`): no grid artefacts in any FM row; energy_qm maxima closest on 4/6 patches
(191/172, 143/158, 87/91, 63/68); patch 764 overshoots (52 vs 30 mm), 393 undershoots (8 vs 44); 979 hits max on the wrong core.
Panel units say "mm h-1" but are hourly mm (open TODO).

**Summary plots** (`figures/v4adv_val{ext,sub}_summary_qm/`): energy_qm has most small-scale power (valext ratio 0.5–0.8 from 30 to
2 km vs 0.25–0.35 FM/control, 0.4–0.5 dist). Low RL bias hides an overshoot: λ = 10/100/1000 = 83.9/151.2/266.8 vs 90.0/147.0/228.3
observed (-7%, +3%, +17%); fitted xi 0.234 vs 0.155, sigma 14.2 vs 14.4. SAL: S median 0.40 (FM 0.64, dist 0.54), A ≈ 0.
Perception-distortion: spectral distance 1.28 (FM 1.51, dist 1.32) at RMSE 0.957 (FM 0.950).

**Return levels without a fit.** On valsub other FM rows have < 200 predicted exceedances (79/150/83) -> no fit
(`tail_metrics.py:115`). energy_qm's 382 vs 183 mm at λ = 1000 is mostly extrapolation (xi 0.35 vs -0.03). Ranked patch maxima
(ranks 1/2/3/5/10/20/30, mm): valsub target 151 150 144 128 100 76 58; energy_qm 256 200 180 164 111 88 72; FM 176 148 118 80 60 49 44.
valext target 201 191 174 159 142 123 106; energy_qm 377 315 224 188 150 129 109; FM 139 121 121 113 92 74 64. From rank 10 down
energy_qm within ~5–20% (FM 25–40% short); top 5 overshoot 1.3–1.9x. Open: day-block bootstrap on xi; check top two valsub targets.

**U-Net qm bracket (§37 protocol; `bracket_gallery_qm.pdf`, `bracket_gallery_qm_hi.pdf`).**

| model | ΔMAE | anisotropy (x van.) | exc. ratio | RL bias λ10/100/1000 | FSS 50.1 1/21 px | peak ratio (ext) | fitted xi |
|---|---|---|---|---|---|---|---|
| vanilla | 0 | 2.12 (1.00) | 0.113 | -44/-31/-13% | 0.106/0.383 | 0.36 (0.20) | 0.32 |
| thr 6.0e-5 | +4.7% | 0.56 (0.26) | 0.192 | -37/-41/-44% | 0.097/0.417 | 0.86 (0.52) | 0.15 |
| qm 2.6e-4 | +5.9% | 0.83 (0.39) | 0.156 | -40/-43/-47% | 0.064/0.329 | 0.88 (0.50) | 0.13 |
| thr 1.8e-4 | +7.9% | 2.24 (1.05) | 0.287 | -32/-36/-41% | 0.118/0.535 | 0.86 (0.53) | 0.15 |
| qm 7.7e-4 | +8.9% | 0.91 (0.43) | 0.296 | -32/-36/-40% | 0.120/0.586 | 0.85 (0.50) | 0.16 |
| thr 5.4e-4 | +12.5% | 2.02 (0.95) | 0.442 | -25/-32/-39% | 0.131/0.648 | 0.93 (0.58) | 0.12 |
| qm 2.3e-3 (11-03) | +14.2% | 1.18 (0.56) | 0.669 | -14/-18/-21% | 0.128/0.720 | 0.96 (0.68) | 0.16 |
| thr 1.8e-3 | +19.0% | 3.45 (1.62) | 0.607 | -18/-21/-24% | 0.125/0.714 | 0.92 (0.61) | 0.17 |

gpd_xi_obs = 0.183 every row; w_max in logs 2.6e-4 / 7.7e-4 / 0.0023; 7.7e-3 was still training (no result recorded). All qm runs
above the +5% cap. At low/mid weight, qm gives no tail gain over thr at matched MAE (inference: U-Net's problem was never reach).
qm removes most grid-aligned striping (e.g. 2021-05-11 per-patch anisotropy: thr 1.8e-4 7.9, qm 2.6e-4 0.75, qm 7.7e-4 1.7, target
0.86). At 2.3e-3 qm is ahead of the thr curve (higher exceedance, smaller RL bias, same FSS as thr 1.8e-3 for 5 points less MAE),
but shows a new artefact: small cross-hatched "x" spots at local maxima (inference: qm's version of gaming).

**dist_qm with retention (2026-11-05; `FMdistqmret_v4adv_20261105_163457`; w_reward 1, prox 1, retain 1, spread 0.5, M = 4;
reward 70% of gradient norm, cos +0.70).**

| Row | valext exc | valext RL bias | valext FSS@50.1 | valsub exc | valsub FSS@50.1 | Anisot. ext/sub | Peak ratio ext/sub | RMSE ext/sub |
|---|---|---|---|---|---|---|---|---|
| FM clean | 0.298 | 0.367 | 0.285 | 0.272 | 0.337 | 1.31/1.40 | 0.33/0.55 | 1.153/0.336 |
| dist_qm | 0.273 | 0.340 | 0.267 | 0.248 | 0.310 | 1.32/1.33 | 0.29/0.42 | 1.152/0.335 |
| dist_qm_ret | 0.465 | 0.185 | 0.304 | 0.672 | 0.448 | 1.47/1.48 | 0.37/0.69 | 1.154/0.337 |
| energy_qm | 0.665 | 0.089 | 0.348 | 0.914 | 0.246 | 1.36/1.34 | 0.34/0.60 | 1.160/0.339 |

Retention + parity weight turn dist_qm into a clear tail effect (two settings changed; not attributed). energy_qm still ahead on
tail; dist_qm_ret best valsub FSS@50.1. Anisotropy highest of any FM row (1.47–1.48), not visible in panels
(`figures/v4adv_valext_fields_qmret/`); per-patch ranking over valsub is the pending check. Patch maxima (target / dist_qm_ret /
energy_qm): 354 191/79/249; 519 143/30/32; 979 87/57/97; 257 63/71/43; 393 44/15/38; 764 30/27/21.
Summary plots (`figures/v4adv_val{ext,sub}_summary_qmret/`): yaml RL (own rate) 74/120/183 mm vs 90/147/228 (-18/-18/-20%;
energy_qm 84/151/267); in `tail_survival.png` (own exceedance count) dist_qm_ret lies just under target (xi 0.13), energy_qm
overshoots (xi 0.23); both views right. Ranked maxima valext: target 202 191 175 160 143 123 107; dist_qm_ret 221 192 176 146 117
92 85; energy_qm 377 316 225 189 150 129 110; dist_qm 189 186 117 108 85 75 66. valsub: target 152 151 144 128 101 76 59;
dist_qm_ret 206 165 164 129 101 68 54; energy_qm 257 201 181 165 112 89 72; dist_qm 135 110 80 70 57 48 43. Spectrum ratio ~0.45;
RAPSD 0.74 (energy_qm 0.41); SAL A +0.01, S 0.51; on Pareto front; rmse_extreme spread 2.88–2.93 (small).

**Decision (researcher, 2026-11-05): FM + energy_qm is the FM model** (`FMenergyqm_v4adv_20261031_022957`, `fm_mink_latest.pth`);
future FM evals keep FM clean + energy_qm; known weak point: largest maxima overshoot 1.3–1.9x at ranks 1–3. **Reserve: FM +
dist_qm_ret** (largest maxima right size, best valsub FSS@50.1; smaller exceedance 0.47 vs 0.67, highest anisotropy 1.47, unexplained).
Confirmed in memory file `fm-eval-reference-rows.md` (drop dist, dist_qm, dist_qm_ret, energy_ctl from new tables).

---

## §43. The extreme-evaluation ladder (2026-11-04, amendments and results to 2026-11-05)

**Decision (researcher).** A ladder of test sets, each rung adding one shift; report each model's degradation from the previous
rung vs a baseline on the same patches, with day-block intervals (§32); smaller differences not claimed.

| Rung | Test set | Shift added | Role |
|---|---|---|---|
| R0 | ODYSSEY test, all tiles | none | bulk reference |
| R1 | ODYSSEY test above u + catalogued events in test (`configs/prominent_events.yaml`, `configs/mch_events.csv`) | intensity conditioning, in range | in-distribution extremes |
| R2 | 101 paired days 2024-07-05..10-30, vs ODYSSEY and NIMBUS same hours | product only | separates product |
| R3 | NIMBUS 2024-11..2026-09 minus paired days | new period and events | product + time |
| R4 | ODYSSEY test above u_c, models retrained without >= u_c | intensity beyond training | central hypothesis H3 |
| R5 | R4 models on NIMBUS above u_c; MCH events vs gauges | all of the above | compound robustness |
| R6 | DPC Italy (roadmap M6) | region and network | only if R4 positive |

Why: NIMBUS on the same days has 1.44x pixels >= 31 mm/h, 2.4x Euler characteristic, 6x power at 4 km. R4: unit = absolute hourly
accumulation by tile max (E1); E3 local-quantile cut only as sensitivity. Needed result: interaction (Minkowski degrades less R1 -> R4
than MSE). Counts v4 (train/test tile-hours): >= 50.1 mm 14,067/2,383 (305 test days); >= 80 mm 4,156/559; >= 100 mm 2,090/249
(109 days). Models: MSE; MSE + Minkowski; area-only; one competing loss; FM; FM + energy_qm; 3 seeds for deterministic cut runs.
Scoring rules: forecaster's dilemma -> pair observation-conditioned scores with coarse-input-selected cases, false-alarm rates and
threshold-weighted CRPS, as written "[CITE?: forecaster's dilemma; threshold-weighted CRPS]"; independent truth SwissMetNet (72 of
92 MCH rain-event stations) and CombiPrecip; test scored once per final model.
Showcases fixed before output: C1 (R1) Misox 2024-06-21 (63.7 mm in 60 min at Grono); C2 (R2) Storm Boris; MCH `_id` 36, Ticino
2024-07-06; C3 (R5) MCH `_id` 64, Level-4 ex-hurricane, Ticino 2025-08-27/28 (gauges CEV, SBO, MSG, ORO, VEV; tile max 65.9 mm).
**Dropped:** held-out storm type (E2; 73 Swiss events too few) and held-out period within ODYSSEY.

**Amendments (2026-11-04).**
- **u_c = 50.1 mm** (same as POT level); above-cut inputs not out of range (0.5% exceed cut-train input max); degradation by distance
  above cut (50–80, 80–100, >= 100 mm).
- **Cut = tile-hours >= u_c only** (1.08% of train; blocks would remove 17.5%); one random-drop control for MSE.
- **NIMBUS tail:** ~a third of the 48 largest held-out tile-hours are clear errors; a NIMBUS rule was dropped as infeasible
  (`n_persist` acts on 36% of ODYSSEY test tail, 55% >= 150 mm; `n_ray` silent; tail >= 150 mm spread over 37 lead radars, top 3 = 35%,
  ~5x ODYSSEY per-day rate). **Decision: gauge-anchored tail scoring**; `nimbus` lists unchanged; bulk on all NIMBUS; R3/R5 tail
  claims only on gauge-checked cases; gallery `quality_v4/heldout_tail_check/` as limitation.
- Area-only ablation weight: Adam-share parity with the full Minkowski loss.
- **Study-1 Minkowski loss: §16 rule applied -> threshold form at 6.0e-5** (`UNet_Ana_20261025_042024`, +4.7% MAE, exceedance
  0.113 -> 0.192). Quantile form reported as a bracket only.
- Competitors: spectral, SSIM (fixed), wet area, optical flow (fallback counted and warned), each at parity with Minkowski 6.0e-5
  (Adam share 5.30).
- Seeds: 3 for MSE and Minkowski in both arms, 1 for area-only, competitors and random-drop control; 21 runs, ~17.5 GPU-days; queue
  `scripts/hpc/run_r4_queue.sh` (`logs/r4_queue.log`, `logs/r4_queue_done.txt`).
- **R4 truncated Minkowski grid in both arms:** `MINKOWSKI_MAX_THRESHOLD: 50.1` keeps 13 thresholds (0.1–31.7 mm). Reason: at 50.1
  and 79 mm every cut target has zero functionals, so the loss would teach never to exceed the cut, rejecting H3 mechanically.
  Both arms so the only difference is data. Full-grid 6.0e-5 stays the Study-1 model for R1. One extra run -> 22 runs. Own parity
  audits (Adam share 5.30). Still to fix: area-only (truncated likewise) and FM energy_qm reward (same issue, before the FM R4 arm).
  Tests `tests/test_minkowski_channels.py`; startup line `minkowski [13 thresholds < 50.1]`.
- **H1 check** (pre-registered; `src/evaluation/analogue.py`; train rows >= 5 mm 417,331, 14,067 >= u_c). Result 2026-11-05
  (`quality_v4/analogue/log_test_shape_h1.txt`, code 6402caf):

  | test cases | n (days) | med d_cut/d_full | med d_rand/d_full | excess [95%] | nearest analogue >= u_c |
  |---|---|---|---|---|---|
  | >= 50.1 mm | 2,383 (305) | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 7.0% (2.1x base 3.4%) |
  | 20.2–50.1 (control) | 14,555 (469) | 1.000 | 1.000 | +0.000 [+0.000, +0.000] | 5.0% (1.5x) |

  **Verdict: inconclusive** (2.08x misses the 2x bound). Criterion 1 is zero by construction (found afterwards). Inference: 93% of
  above-cut hours have their closest shape below the cut; shape not intensity-independent. Open caveat: tile-edge truncation.
- **Trend test** (pre-registered; `scripts/evaluate/shape_trend_test.py`; worlds W1 reuse, W2 re-weighting, W3 trend, W4 break;
  M1 coverage, M2 drift, M3 energy-distance extrapolation vs reuse; validity gate S-reuse, S-trend, S-out). Result 2026-11-05
  (`quality_v4/analogue/trend_test/trend_test.json`, code 5b86169): **no verdict, gate failed** (S-trend tie).

  | | main (ref 5–50.1, target >= 50.1, n 2,383) | backtest (ref 5–20.2, target 20.2–50.1, n 4,000) |
  |---|---|---|
  | M1 median/q90 ratio | 1.006 [0.994, 1.016] / 1.016 [0.998, 1.035]: covered | 0.991 [0.982, 1.001] / 0.992 [0.976, 1.007]: covered |
  | M2 D, min cos (all) | 0.305 [0.287, 0.326], -0.570 [-0.698, -0.314] | 0.307 [0.291, 0.323], 0.877 [0.833, 0.909] |
  | M2 non-edge | 0.200 [0.145, 0.262], -0.608 | 0.080 [0.053, 0.127], -0.706 |
  | M3 dED (all) | +0.085 [+0.039, +0.136] trend wins (floor 0.009) | -0.197 [-0.240, -0.155] reuse wins |
  | M3 non-edge | -0.067 [-0.089, -0.043] reuse wins | -0.051 [-0.061, -0.041] reuse wins |
  | edge share | 0.70–0.77 / 0.65 | 0.70–0.78 / 0.75 |

  S-trend could not pass (on-trend dED second order). Exploratory reading: closest to W2; no support for "extrapolate a trend".

---

## Notes files

### notes/status_2026-10-19.md
Snapshot for the 2026-10-20 check-in. Decisions in force listed: §21 (1-h target, ACRR H-45..H, screening on 15-min), §24 (no
dataset decision from images), §26 burst rule, §27 radar ownership from `how/nodes`, §28 rule 5 radar-days score >= 1 excluded,
§29/§32 readiness gate closed (2a fails, minor limitation), §30 static-clutter and temporal-support rules (`p1_min` 24,
`min_wet_jump` 0.021), §31 stationary-echo rule not adopted, §32 POT u = 50.1 mm and levels 20.2/31.7/50.1, §33 hourly thresholds
by exceedance frequency (`config_v4.yaml`), §34 ownerless tile-hours excluded, §35 advection-corrected target (`patches_v4_adv`),
§36 boundary test run, exclusion not applied. Running: v4 store build (`logs/build_v4.log`), `prep_train_v4_adv.sh` chain,
§36b tail boundary test. Open: biased radars (recommendation option b, tail-focused test), advection on fast convection, timing
of the pre-break ablation. Next: check chain, re-establish hourly weight, baselines, re-derive POT on `patches_v4_adv`, sigma_r per
backbone, validators, later DPC Italy, plain-vs-corrected ablation, `[CITE?]`s, email to Lionel Moret, COMEPHORE. Caveats: 2a
false alarms ~9 gauge-hours; pre-break fields 1.3–6.5x more fragmented at 0.1 mm; OPERA ~15–20% low at moderate intensities.

### notes/status_2026-10-21.md
Supersedes the above. Done 2026-10-21: v4 store built (05:02), all validators and smoke run pass; two `check_datasets.py` bugs
fixed; overfit check needs 600 steps; §36b flagged ukcyg 2023, isbjo 2024, plpas 2012 (excluded); §36c Lithuania 2026 excluded
(Hungary, Slovakia, Romania no network bias); rsjas 2023 and hrulj 2023 excluded; climatology and why-missed analyses; methods
§6.7/§6.8 in `notes/opera_hourly_screen.md`; §37 step 1a 8.5e-5; final lists as in §37. Running: vanilla baseline (~2026-10-22 15:00).
Open: rule-5 revision (future: reference from other years, other networks, or fixed bar); weight step 2 uses §16 rule. Next:
step 1b, pre-break ablation post-break arm, POT re-derivation, Minkowski backbone and sweep, sigma_r, systematic literature search.
Caveats: corrected climatology overstates what reaches the dataset; Romanian radars at 4–5x as limitation.

### notes/minkowski_deterministic_limitation.md (2026-10-26)
Explains the fixed texture of Minkowski-trained backbones. Observation: strokes in rain areas, one dominant direction, grid
striping and blocks at highest weight. Theorem: the functionals are motion-invariant; by Hadwiger's theorem they are the only
additive continuous motion-invariant functionals in 2-D, "[CITE?: Hadwiger's characterization theorem, e.g. Hadwiger 1957 or
Klain & Rota 1997]"; so the loss is blind to position and orientation. Causal chain: (1) MSE prefers no texture (double penalty,
"[CITE?: double-penalty problem in precipitation verification]"); inferred cheapest compromise is low-amplitude modulation where
the mean is high; (2) grid estimators break rotation symmetry (Crofton numbers as §38, "[CITE?: Cauchy-Crofton formula / digital
perimeter estimation]"; oblique edges earn ~6.6% more perimeter); (3) flat direction + weak symmetry breaking selects one "stamp"
(rotation-test numbers as §38); (4) weight scales it (striping 0.017 -> 0.098; MAE +4.7% to +19%). Analogy: symmetry breaking
with a crystal field, and where it breaks. Conclusion: a fixed stamp is the optimum of this objective for a deterministic model,
an instance of the perception-distortion trade-off "[CITE?: Blau & Michaeli, The Perception-Distortion Tradeoff, CVPR 2018]";
a generative model removes the dilemma (argument for Study 2). Misreading named: "the loss teaches the shape of rain". Limits:
S-shape not measured; low-weight evidence weak; first link inferred. Remedies: average high-pass texture to show the stamp; more
Crofton directions (knight's move, ~±1%) and mixed 4/8-connectivity Euler; stochasticity or position-aware losses.

### notes/minkowski_reward_intensity.md (2026-10-29)
Decision note for §39. Until then the loss in practice shaped light rain (73% of reward gradient < 1 mm; 15-min: 83% from
thresholds <= 0.28 mm/h). Derivation: pixel mass (pull -> N p(u)) and out-of-reach (17 mm core, u = 20.2 mm, tau = 0.10 mm,
sensitivity ~1e-13). Decision as §39; alpha = 1 equalises thresholds for a power-law survival curve. Alternatives rejected as §39.
Gradient by intensity (<1/1–10/10–20/20–50/>=50 mm): heavy original 73.4/19.9/3.0/3.0/0.7 -> qm 15.9/27.0/16.2/31.5/9.4; random
89.1/9.9/0.8/0.2/0.0 -> 60.7/31.5/6.4/1.4/0.0. alpha = 0 changes almost nothing (area 11% of gradient; Euler at 0.1 mm 17–28%).
Coupling: dist 0.232 -> 0.158 (-31.6%), 4/40 clipped; energy 0.131 -> 0.085 (-35.1%), 0/40. Reward norm 5.0e-4 both modes (was
8.3e-4); energy 0.07% of total. Risks: sparse area gradient (pixels with gradient 15% -> 12%), noisier reward (11.4% -> 18.9%),
artefacts get more weight, not yet a result; `diagnose_coupling.py` TEST 4 still prints the threshold form.

### notes/weight_selection_literature.md (2026-10-21, preliminary)
Procedure: calibration by gradient share (`tools/gradient_audit.py`, `notes/gradient_audit.pdf`) and the §16 bracket. References,
as written:
- Chen, Badrinarayanan, Lee & Rabinovich, *GradNorm*, ICML 2018 (PMLR) — dynamic, multitask.
- Wang, Teng & Perdikaris, *Understanding and mitigating gradient pathologies in physics-informed neural networks*, arXiv 2001.04536.
- Zhong & Maki, *Regularizing CNN Transfer Learning with Randomised Regression*, CVPR 2020 (arXiv 1908.05997) — **closest**.
- Du et al., *Adapting Auxiliary Losses Using Gradient Similarity* (gatsby.ucl.ac.uk PDF) — cosine-based.
- Sign-language noise-robust losses, arXiv 2609.12885 — **found by search, not read**; matched weights hold only at init (150 -> 1.6).
- Blau & Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018 (CVF).
- Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023; *Goodhart's Law in Reinforcement Learning*,
  ICLR 2024; *Catastrophic Goodhart*, NeurIPS 2024 — analogy, not precedent.
- Bergstra & Bengio, *Random search for hyper-parameter optimization*, JMLR 2012 — from memory, not re-checked.
- MSRGAN (doi 10.3390/rs17132281) — weights tuned without a bound; EGUsphere preprint egusphere-2026-861 review asks for a
  sensitivity test.
- "When Losses Align" preprint (2605.07756) — to read in full.
Not found (candidate contributions): cross-dataset weight transfer by gradient share (modest); hacking-bounded weight selection in
precipitation SR (most defensible). Follow-up: per-step weight from gradient share. Systematic search query strings recorded
(Scholar/Scopus/WoS 2015–2026, forward citations of GradNorm, Zhong & Maki, Wang et al.).

### notes/events.md (catalogue checked 2026-09-28; problems 2026-10-02; status 2026-10-04; MCH list 2026-11-04)
Purpose: a curated, named test of known extremes (not o.o.d.); each event window ±1 day forced into test or nimbus with its whole
ISO week; `events_test` / `events_nimbus` keep every overlapping tile. Selection criteria (reconstructed): documented, inside archive
and domain, spread over storm types, spread over years and products, weighted to the Alps/Switzerland. Events (window; type; split;
tiles; >= 31; max mm/h):
- HyMeX IOP6, S France, 2012-09-23..25; fast linear MCS; test; 1,370; 150; 273
- HyMeX IOP16, NW Med, 2012-10-25..28; Mediterranean HPE; none; 0
- Pentecost storm "Ela", 2014-06-08..10; derecho/bow echo; test; 2,015; 693; 365
- Münster cloudburst, 2014-07-27..29; stationary convection (292 mm in 7 h); test; 576; 315; 151
- Hérault / Montpellier, 2014-09-28..30; Mediterranean stationary (252 mm in 3 h); test; 288; 46; 205
- Côte d'Azur (Cannes), 2015-10-03..04; (175 mm in 2 h); test; 191; 66; 178
- Elvira / Friederike, 2016-05-27..06-07; stationary convection; test; 13,809; 2,889; 453
- Poland derecho, 2017-08-11..12; test; 946; 134; 286
- Aude floods, 2018-10-14..15; (295 mm in 12 h); test; 384; 79; 128
- Storm Alex, 2020-10-01..03; back-building, orographic (> 500 mm/24 h); test; 159; 69; 375
- Swiss supercell, 2021-06-20; test; 192; 102; 167
- Moravia tornado, 2021-06-24; IF4 supercell; test; 96; 62; 319
- Lucerne hail, 2021-06-28; 10 cm hail at Wolhusen; test; 190; 91; 181
- CH / N Italy hail, 2021-07-08; 11 cm hail near Milan; test; 284; 196; 178
- Low "Bernd", 2021-07-12..15; Ahr/Erft/Vesdre floods; test; 1,534; 356; 452
- Mediterranean derecho, 2022-08-17..18; Corsica -> Czechia; test; 2,429; 910; 488
- Emilia-Romagna 1st, 2023-05-01..03; none; 0
- Emilia-Romagna 2nd, 2023-05-15..17; none; 0
- N Italy giant hail, 2023-07-19; 16 cm hail; test; 154; 144; 487
- La Chaux-de-Fonds / Azzano, 2023-07-24..25; 217 km/h downburst, 19 cm hail; test; 658; 476; 493
- Slovenia floods, 2023-08-03..06; test; 1,125; 564; 464
- Southern Germany floods, 2024-05-31..06-03; test; 1,523; 607; 483
- Misox, 2024-06-21..22; 63.7 mm in 60 min at Grono; test; 174; 109; 460
- Maggia / Valais, 2024-06-29..30; orographic; test; 370; 144; 490
- Storm Boris, 2024-09-12..16; 442 mm/3 d in N Czechia; nimbus; 4,747; 1,547; 486
- Valencia DANA, 2024-10-28..30; 771.8 mm in 14 h, 184.6 mm/h gauge-hour; nimbus; 200; 76; 115
`events_test` 28,467 patches; `events_nimbus` 4,947. Problems (2026-10-02): three empty events; 2013 CE floods and "Andreas" in
train; ten maxima at 450–493 mm/h (likely artefacts/hail); several are accumulation not rate extremes (tag `rate`/`accumulation`);
coverage bias. Causes: IOP16 not in archive (gap 2012-10-10..11-08); Emilia-Romagna tiles have static holes (18 and 272 px).
Europe-wide NaN table: admitting <= 1% NaN adds 15% more >= 31 mm/h tiles, <= 5% adds 29%. Options A (shifted windows for events,
recommended), B (repair static holes), C (masked partial tiles); draft paper limitation paragraph. Status 2026-10-04: added
`ce_floods_201305` (2013-05-30..06-03; Grams et al. 2014, NHESS) and `andreas_20130728` (2013-07-27..28; Kunz et al. 2018, QJRMS)
— 3,201 and 1,907 tiles (782, 365 >= 31); IOP16 `available: false`; option A implemented (`scan_event_windows.py`, Emilia-Romagna
500 and 578 windows, peaks 148 and 335 mm/h). MeteoSwiss list (Lionel Moret, 2026-11-04; `configs/mch_events.csv` via
`import_mch_events.py`): 73 Swiss events 2020-02-04..2026-08-25 (rain 23, snow 10, mixed 1, TOT_PREC without warning 3, none 36;
26 in 2026); Zurich local time converted to UTC; SwissMetNet OGD (`rre150z0`, `rre150h0`), 72 of 92 stations available;
CombiPrecip archive on request; fixes (2923 -> 2023 in `_id` 34, etc.). Rain events by split: nimbus 15, test 5, train 2, val 1
(+ snow/mixed nimbus 9); test ones: Brigitte (2020-10-02), July 2021 flood, 2022-08-19, Misox, Maggia. Switzerland spans tiles
(640, 768) and (640, 896). Storm Sabine (`_id` 1) suspicious (14/96 tile-hours, max 0.012 mm).

### notes/future_project_radar_cleaning.md (saved 2026-10-08)
A banked, separate later project. Problem: OPERA artefacts (clutter, anaprop, RLAN rays, rings, speckle, failures, ceilings) set
the tail; rules misfire and validate only where independent data exist (DE/CH gauges and RADKLIM cover ~1/3 of tail removals);
label-based cleaning not defensible (§24). Idea: separate by structure, not rarity: x_t = r_t + a_t, rain advection-predictable,
artefacts sparse and persistent in grid or radar-polar frame; baseline robust PCA in polar coordinates vs the repo's persistence
repair. Evaluation without labels: RADKLIM, COMEPHORE, CombiPrecip, gauges, §22/§24 metrics. Limits: stationary real rain,
calibration errors, value-level ceilings. Fallback weak supervision. Literature as written: EURADCLIM (Overeem et al. 2023, ESSD
15, 1441); Lepetit et al. 2022 (IEEE TGRS 60); Bøvith 2008 (DTU PhD thesis); Bölz et al. 2026 (EGUsphere preprint); Scovell et al.
2013 (Odyssey QC); robust PCA (Candès et al. 2011, J. ACM 58); DECOLOR (Zhou et al. 2013, TPAMI 35); structured Noise2Void
(Broaddus et al. 2020); seismic (Birnie et al. 2021); Astronomaly (Lochner & Bassett 2021); El Hachem et al. 2022 (HESS 26);
HadISD (Dunn et al. 2012). Reusable code: `src/data/radklim.py`, `src/data/hourly.py`, `cleaning.py`, `radar_screen.py`,
`gauge_hourly.py`, `radklim_test.py`, `src/data/geo.py`.

### Memory files (skimmed)
- `fm-eval-reference-rows.md` (2026-11-05): FM tables keep FM clean + energy_qm; dist_qm_ret is the reserve.
- `node40-borrowed-gpus.md` (2026-11-05): node40 GPUs 0–3, CPUs 0–5, until ~Mon 2026-11-09; `node40_run.sh` / `node40_r4_cell.sh`
  only guard; U-Net there V100 fp16 AMP; FM stays on node32.
- `pgrep-self-match-waits.md`: a self-matching `pgrep -f` wait deadlocked `logs/h1_check_chain.sh` and `run_paired_build.sh` for ~21 h
  on 2026-11-04/05 (H1 check never ran via the chain); wait on PIDs or markers.
- `input-resolution-25km.md`: input is 25 km (12.5x factor).
- `long-running-jobs-launch-detached.md`, `opera-radar-artefact-patches.md`, `future-project-radar-cleaning.md`: launch detached
  and confirm; top-ranked patches are often artefacts; cleaning project banked.

---

## Current state as of the latest dates in these files (2026-11-05)

**Decided.**
- Data: v4 corrected hourly target (`patches_v4_adv`), 7 radar-years excluded, final lists train 1,299,686 / val 172,050 / test
  212,803 / nimbus 378,567; advection correction accepted; POT and cut level u = u_c = 50.1 mm.
- Study 1: threshold-form Minkowski at **6.0e-5** (`UNet_Ana_20261025_042024`) is the Study-1 model (§16 rule); quantile form reported
  as a bracket. sigma_r of the hourly vanilla backbone = 0.012668.
- Study 2: **FM + energy_qm** (`FMenergyqm_v4adv_20261031_022957`) is the FM model; FM + dist_qm_ret in reserve.
- Evaluation: the R0–R6 ladder, u_c = 50.1 mm, tile-hour cut (1.08% of train), truncated 13-threshold Minkowski grid in both R4 arms,
  gauge-anchored NIMBUS tail scoring, showcases C1–C3 fixed, competitors at Adam-share parity 5.30.
- H1 check: inconclusive; trend test: no verdict (validity gate failed), exploratory reading closest to W2.

**Running / queued (as last recorded).** The R4 queue (`scripts/hpc/run_r4_queue.sh`, `logs/r4_queue.log`, done list
`logs/r4_queue_done.txt`; 22 runs, ~17.5 GPU-days), on node32 GPU 1 plus borrowed node40 GPUs 0–3 until ~2026-11-09. The U-Net qm
bracket's 7.7e-3 run was last recorded as training (2026-11-02); no result in these files.

**Next / open.**
- Truncated-grid parity audits; truncate the FM energy_qm reward before the FM R4 arm.
- Per-patch anisotropy ranking for dist_qm_ret; day-block bootstrap intervals on xi; look at the top two valsub targets.
- Tile-edge truncation check for H1/trend drift; a future trend-test gate with S-trend several deltas out (a new test).
- Striping-measure threshold to fix before use; plot units TODO.
- Pre-break ablation (§34) status after the 2026-10-26 stop is not updated in these sections beyond R4 planning.
- Systematic literature search before any weight-selection novelty claim; `[CITE?]` items (Hadwiger, double penalty, Crofton,
  forecaster's dilemma, threshold-weighted CRPS).
- Future rule-5 revision; events: options B/C for coverage; Storm Sabine check.
