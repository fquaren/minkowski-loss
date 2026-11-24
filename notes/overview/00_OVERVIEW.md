# minkowski-loss: project overview (state on 2026-11-07)

This file is the synthesis. It was compiled on 2026-11-07 from `EXPERIMENTS.md` (last updated
2026-11-04), `DECISIONS.md` (§1–§43, last entries 2026-11-05), `RESEARCH_NOTES.md`,
`MINKOWSKI_DOCS.md`, `README.md`, everything in `notes/`, the configs, the eval summaries and
the git log. The appendices in this folder hold the detail, section by section, with every
number and path kept:

| file | covers |
|---|---|
| `A_experiments_core.md` | compute, checkpoints, result tables, todo list, robustness ladder, paper plan, roadmap M1–M6 |
| `B_dataset_history_and_quality.md` | dataset versions v1–v4, archive fetch, data-quality findings, "to explore", deferred, superseded |
| `C_decisions_01-20.md` | DECISIONS §1–§20, with status of each |
| `D_decisions_21-36_screening.md` | DECISIONS §21–§36c (hourly target, screening rules, gate), final screening pipeline |
| `E_decisions_37-43_and_notes.md` | DECISIONS §37–§43 (hourly weight, Study 2 on v4, quantile reward, ladder), status notes, events catalogue |
| `F_theory_and_literature.md` | hypotheses, definition of an extreme, regimes, radar quality, method docs, **full literature list by theme** |
| `G_reports_and_resources.md` | the LaTeX reports and slides, v4 data notes, configs, data locations, code map, validation reports |

The appendices were written by reading agents and I spot-checked them against the sources,
not line by line. Where a number matters for a paper, check it in the source file named next
to it.

---

## 1. The project in one paragraph

The project super-resolves precipitation from 25 km to 2 km (factor 12.5, 128×128 patches) over
Europe, using the EUMETNET OPERA radar composite. The question is whether a differentiable loss
built from Minkowski functionals (area, perimeter, Euler characteristic of the excursion sets
{x ≥ u} over a grid of thresholds) makes the model better at extremes, and in particular at
extremes stronger than any seen in training. **Study 1** adds the structural loss, and four
competing structural losses, to a deterministic U-Net backbone trained with MSE. **Study 2** puts
a flow-matching (FM) residual model on top of a backbone and couples the structural loss as a
reward during fine-tuning. The project follows the emulator paper (Quarenghi, Cotsakis, Beucler,
*Environmental Data Science* 2026, arXiv:2604.11422). This work replaces that paper's learned
emulator with analytic differentiable functionals. Funding: SNSF grant 10001754 ("RobustSR").

## 2. Where things stand (2026-11-07)

**Decided and in force**
- **Data:** v4. Target = the advection-corrected 1-h accumulation (`OPERA/patches_v4_adv`,
  `configs/config_v4_adv.yaml`), screened on 15-min frames, week-blocked leak-free splits,
  seven radar-years excluded. Final lists: train 1,299,686 / val 172,050 / test 212,803 /
  nimbus 378,567 tile-hours. No v5 (§32).
- **Tail level:** u = 50.1 mm for the POT fits. The same level is the R4 cut level u_c
  (§32, §37, §43 A1). Rate-equivalent strata: 20.2 / 31.7 / 50.1 mm.
- **Study-1 model:** threshold-form Minkowski loss at w = 6.0e-5
  (`UNet_Ana_20261025_042024`; +4.7% MAE, exceedance ratio 0.113 → 0.192). Chosen by the §16
  rule (§43, D1, 2026-11-04). The quantile form is reported as a bracket only.
- **Study-2 model:** FM + energy reward in the `quantile_mass` form
  (`FMenergyqm_v4adv_20261031_022957`, `fm_mink_latest.pth`). On val_extremes1024 its
  exceedance ratio is 0.665 against 0.298 for clean FM, and its RL bias 0.089 against 0.367.
  Single seed. The reserve model is FM + dist_qm_ret (§42, 2026-11-05).
- **Evaluation:** the robustness ladder R0–R6 (§43). Tail scoring on NIMBUS is gauge-anchored,
  and the test set is scored once per final model.

**Running or queued**
- **R4 queue** (`scripts/hpc/run_r4_queue.sh`; 22 runs, ~17.5 GPU-days). It runs on node32
  GPU 1 plus the borrowed node40 GPUs 0–3, which are available until about 2026-11-09. Done as
  of 2026-11-06, per `.claude/worktrees/rtx6000-node-detection-750d7c/logs/r4_queue_done.txt`:
  MSE cut / full (s42), Minkowski cut / full (s42), SSIM cut / full (s42).
- **B1:** paired ODYSSEY/NIMBUS store for R2. **B2:** SwissMetNet gauge fetch. Both were
  running on 2026-11-04. The paired build had been deadlocked for ~21 h by a self-matching
  `pgrep -f` wait (memory note). Whether it has since completed is not recorded.

**Next, by the notes' own priority**
1. Finish R4 (E1). Then the FM R4 arm (E2): FM base on the cut list, σ_r recomputed, then
   energy_qm. The energy_qm reward must first be truncated to thresholds < u_c.
2. energy_qm with 3 seeds (D4).
3. Scoring F1–F4 on the ladder. R6 (DPC Italy) only if R4 is positive.
4. Paper write-up items (§10 below).

**Gaps I found while compiling** (not written in the notes)
- The U-Net qm 7.7e-3 bracket point **was** evaluated: `eval_results/{backbone,extremes}/v4adv_val/mink_qm_7.70e-03/`,
  committed in `7d5417c`. It is not entered in DECISIONS §42 or EXPERIMENTS. Its numbers: MAE
  0.05288 (+20.5% vs vanilla 0.04390), exceedance ratio 0.649, RL bias 0.238, anisotropy 0.98,
  FSS@50.1 (11 px) 0.607. It is worse than qm 2.3e-3 on every tail column at higher MAE
  (2.3e-3: +14.2%, 0.669, 0.176), so the qm curve turns over between 2.3e-3 and 7.7e-3.
- The pre-break ablation (§34) was stopped on 2026-10-26 and has no recorded outcome. Its
  decision is still open: whether the 24% of train rows from before 2015-12-12 stay in.
- Several v4 FM configs load the backbone from inside a git worktree
  (`.claude/worktrees/radar-day-exclusion/runs/...`). The post-October runs live there and
  not in `runs/`. Cleaning up that worktree would break those configs.

## 3. Timeline

| date | event |
|---|---|
| 2026-04 – 06 | Emulator-based Minkowski loss (`runs/emulator`, `sr_emulator`), DDPM runs. Vanilla backbone `UNet_Ana_20260623_144958`. |
| 2026-07 | Study-1 Minkowski backbone `UNet_Ana_20260731_100128` (w = 1e-4). w = 1e-3 run hacks the loss (§2). Clean FM `flow_matching_20260716_071518`. |
| 2026-08 | Reward fine-tunes: distributional `Extremes_20260804_170256`, energy `Energy_20260811_171310`. FM on Minkowski backbone `Extremes_20260807_111521`. Competing-loss runs v1/v2. Reward sweeps and sampler check. |
| 2026-09-16 | Full re-evaluation (σ_r guard added). FM results shift. |
| 2026-09-23 | Gradient audit (`notes/gradient_audit.pdf`). Competing losses found inactive, SSIM bug found. DDPM dropped. Splits found to leak (§15). Node limits enforced. §3 reopened. |
| 2026-09-28 | DEM N–S mirroring bug found and fixed. Every earlier checkpoint is affected. Archive enlargement 2012–2026. |
| 2026-10-01/02 | v2 built (week-blocked splits, no 150 mm/h zeroing). |
| 2026-10-05 | v3 built (repair instead of tile rejection). POT fit exposes the Valjevo ceiling and the Madrid radar disk. |
| 2026-10-08 | Target switched to the 1-h accumulation (§21, with D. Nerini). "No decision rests on the researcher's eye" (§24). |
| 2026-10-09 – 14 | Burst rule, radar ownership, rule 5 radar-day exclusions, readiness gate, RADKLIM presence tests (§25–§31). |
| 2026-10-19 | v4 built. Gate passed except 2a, accepted as a limitation. u = 50.1 mm. Advection-corrected target adopted (§35). |
| 2026-10-21 | `patches_v4_adv` validated. Radar-year exclusions final. Vanilla v4 baseline `UNet_Ana_20261021_211732`. |
| 2026-10-22 – 26 | Hourly weight bracket (§37). Striping and texture-orientation analysis. FM trained on v4 (`Extremes_20261026_092630`). |
| 2026-10-29 – 30 | Reward diagnosis: pixel mass and reach. `quantile_mass` form, anchor fix, energy weight rebalanced, w_reward = 0 control (§39–§41). |
| 2026-10-31 – 11-03 | energy_qm moves the tail. dist_qm does not. U-Net qm bracket (§42). |
| 2026-11-03 | Our GPU moved from node34 to node32. `node_limits.py` now detects the node by GPU UUID. |
| 2026-11-04 | Robustness ladder (§43). Study-1 model fixed at thr 6.0e-5. Competitors fixed and put at parity. MeteoSwiss event list received. R4 queue launched. |
| 2026-11-05 | energy_qm chosen as the FM model, dist_qm_ret as reserve. H1 check inconclusive. Trend test gives no verdict. |
| 2026-11-06 | Six R4 runs done. |

## 4. Method

**Functionals.** For each threshold u_j, the loss computes area A, perimeter P and Euler
characteristic χ of the excursion set. Hadwiger's theorem says these three span every additive,
continuous, motion-invariant functional in 2-D. So γ(u) is a complete description *within that
invariance class*. It carries no position or orientation (§6; `notes/minkowski_deterministic_limitation.md`).
Differentiable surrogates (`src/losses/minkowski.py`, design in `MINKOWSKI_DOCS.md`):
- **Area:** sigmoid soft sets σ((x − u)/τ) with τ ∝ u and annealing. A straight-through
  estimator (hard count forward, sigmoid backward).
- **Perimeter:** 4-direction Cauchy–Crofton, weights (0.473, 0.218) calibrated in-repo, within
  ~2% on test shapes.
- **Euler characteristic:** cubical V − E + F with a Gödel-min, against the exact 4-connected
  target. An earlier GUDHI β₀ − β₁ target (8-connected) was the likely cause of the emulator
  paper's low topology R².

**Threshold grid** (§1). Fixed log-spaced physical levels replaced quantiles. On v4: 15 hourly
levels from 0.1 to 79.0 mm (`config_v4_adv.yaml`), matched to the 15-min levels by exceedance
frequency (§33). `MINKOWSKI_DOCS.md` still describes the old Q = 21 quantile grid, so that file
is stale.

**Two loss forms.**
- `threshold` (default). The loss compares log functionals at each fixed level.
- `quantile_mass` (§39; FM reward and U-Net). Area is compared horizontally: the n_j-th largest
  predicted value against the target's, where n_j = #{target ≥ u_j}. Perimeter and Euler terms
  get weights (n_j + 1)^-1. Weights and ranks depend on the target only, so the energy score
  stays proper. On the heavy subset this form raises the share of reward gradient on pixels
  ≥ 20 mm from 3.7% to 40.9%.

**Study-2 couplings.** FM is a residual model on the backbone mean (σ_r pinned per backbone).
The reward is applied DRaFT-K style (K = 1). Two reward types: *distributional* (M = 4,
proximal anchor, spread term λ_s = 0.5) and *energy score* (M = 16, retention). The energy score
is proper (§7). The current L1 version is a weighted sum of 45 CRPS terms; an L2 version is
banked.

**Weight selection** (§9, §16, §37). The weight is set by gradient norm or Adam-update share
measured with `tools/gradient_audit.py`, not by loss value. It is then bracketed {w/3, w, 3w, 10w}
under the §16 rule: best exceedance ratio subject to ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla,
last checkpoint, 3 seeds at the chosen point. The upper bound on the weight is set by reward
hacking, not by accuracy (§2).

## 5. Data

**Source and grid.** OPERA ODIM HDF5 composites, 2200×1900 LAEA grid, 2 km, from the
EUMETNET/MeteoGate open archive on S3 (`openradar-archive`, 2012–2026). The archive has been
fetched up to 2026-09-29, ~1.2 TB. The product switches from ODYSSEY to NIMBUS on 2024-07-05.
NIMBUS is held out as a separate split (§18). Data root: `/home/fquareng/work/data/extremes/OPERA/`.

**Versions** (detail in appendix B)
| version | target | key change | status |
|---|---|---|---|
| v1 | 15-min rate | 2023-08 → 2024-10. Our own 150 mm/h declutter zeroed real cores. Splits random at patch level and 100% leaking. DEM mirrored. | All §1–§3 results; not comparable with later versions |
| v2 | 15-min | Full archive, week-blocked splits with buffers, events forced into test, no zeroing | superseded |
| v3 | 15-min | Repair instead of rejection, ring repair ≥ 89 mm/h, 2013 events added | superseded |
| v4 | 1-h plain sum | Hourly screen, rules 1–5, radar ownership, gate | superseded by v4_adv |
| **v4_adv** | **1-h advection-corrected** | `noblend_nearest` motion correction. Tile maxima fall to 0.66–0.78 of the plain sum. | **current** |

**Screening philosophy** (§17, §22, §24). Reject clear errors only and keep imperfections. Build
no labelled set. No decision rests on looking at images. A rule is accepted only if it raises
agreement with independent data (DE/CH gauges, RADKLIM) on a pre-stated test. In practice this
became the RADKLIM presence test: a rule's removals must be ≥ 85% RADKLIM-dry in every intensity
bin. The full 21-step pipeline is in appendix D. Its main parts:
- per-frame repairs: drizzle, static clutter, spikes, ceilings, > 500 mm/h footprints, rings,
  multi-frame rays, persistence, unsupported cells, isolated bursts;
- hour sums of frames H-45..H (OPERA's ACRR convention);
- tile-level rules: repeated values, ceilings, radar-disk flags, and rule 5 radar-day
  exclusion (3,029 radar-days);
- metadata exclusions: ownerless tile-hours, and biased radar-years ukcyg 2023, isbjo 2024,
  plpas 2012, ltlau/ltvil 2026, rsjas 2023, hrulj 2023.

Rejected rules:
- Mahalanobis mask;
- stationary-echo rule (it lost 1.097% of storm pixel-hours against a ≤ 1% bar);
- radar-year rule;
- lower `p1_min` values.

**Known residual problems**
- Gate criterion 2a fails: +5.4 points of false alarms at 20 mm against RADKLIM, from single-day
  stationary echoes. Accepted as a limitation.
- Rule 5 cannot see radars that fail together. Its reference is the median of five neighbours
  plus the radar's own q99, which is how rsjas/hrulj 2023 slipped through.
- Pre-2015-12-12 fields are 1.3–6.5× more fragmented at 0.1 mm. They are flagged, and the
  ablation is pending.
- Coverage rule: only fully covered tiles are used. This under-represents coastal and
  Mediterranean extremes; Emilia-Romagna has static holes and HyMeX IOP16 is not in the archive.
- On NIMBUS, about a third of the 48 largest held-out tile-hours are clear errors. Hence the
  gauge-anchored scoring.
- OPERA reads ~15–20% low against gauges at moderate intensities. Swiss radar reads heavy
  above ~30 mm (Alps, probably hail). ODYSSEY holds to 50 mm against DWD gauges.

**Events.** `configs/prominent_events.yaml` holds 28 catalogued European events (2012–2024),
forced into test or nimbus with their ISO week (`notes/events.md`). `configs/mch_events.csv`
holds 73 MeteoSwiss events (2020–2026) from Lionel Moret, received 2026-11-04. Three showcases
were fixed before any output: C1 Misox 2024-06-21 (R1), C2 Storm Boris and MCH 36 (R2), C3 MCH 64,
Ticino 2025-08-27/28 (R5).

## 6. Results so far

### 6a. Pre-v4, 15-min rates (v1 data; leaking splits, mirrored DEM)
These hold only as relative comparisons. Do not cite them as the paper's numbers.

- **Minkowski 1e-4 helps the deterministic backbone.** FSS@89 0.040 → 0.159, exceedance ratio
  0.118 → 0.244, RL bias 0.643 → 0.445, for +3.8% MAE (tab:iid, tab:prelim, re-evaluated 09-16).
- **w = 1e-3 games the loss** with grid-aligned filaments and +60% peak overshoot, while every
  structural metric improves (§2).
- **Competing losses (spectral, SSIM, wet area, optical flow) were not a fair trial** (§8 status,
  §16). The gradient audit found three bystanders and one degenerate active term. SSIM had a
  floor bug, and the epoch budgets were unmatched.
- **FM:** both couplings improved exceedance ratio (0.475 → 0.534 / 0.538) and RL bias. §3 was
  reopened, but this is a single seed on one subset with causes unattributed.
- **Coupling correctly wired** (§4): the reward drops 59.7% on a fixed-noise batch, 5× the noise
  floor.
- **Pixel mass is the binding limit** (§5): 83% of the structural gradient comes from thresholds
  ≤ 0.28 mm/h. The anneal parameter cannot change the shares.

### 6b. v4_adv, hourly (current; u = 50.1 mm, validation set, single seed)

**Study 1, threshold-form bracket (§37).** w* = 1.8e-4 from the Adam-share calibration (step 1b;
step 1a gave 8.5e-5).
| w | ΔMAE | anisotropy | exc. ratio | RL bias λ10/100/1000 |
|---|---|---|---|---|
| vanilla | 0 | 2.12 | 0.113 | −44/−31/−13% |
| **6.0e-5** | +4.7% | 0.56 | 0.192 | −37/−41/−44% |
| 1.8e-4 | +7.9% | 2.24 | 0.287 | −32/−36/−41% |
| 5.4e-4 | +12.5% | 2.02 | 0.442 | −25/−32/−39% |
| 1.8e-3 | +19.0% | 3.45 | 0.607 | −18/−21/−24% |

Readings:
- On hourly data the loss costs about 2× the MAE it cost on 15-min rates.
- The anisotropy cap misses visible striping. A texture-orientation striping measure (v2) is
  monotone in weight but has not been adopted.
- Every Minkowski run under-predicts the 100- and 1000-period return levels more than vanilla.
  "Helps the tail" therefore holds for exceedance counts and peaks, not for return levels.

**Study 1, quantile form** (§42 plus the 7.7e-3 point found above).
| w | ΔMAE | exc. ratio | RL bias (mean abs) |
|---|---|---|---|
| 2.6e-4 | +5.9% | 0.156 | — |
| 7.7e-4 | +8.9% | 0.296 | — |
| 2.3e-3 | +14.2% | 0.669 | 0.176 |
| 7.7e-3 | +20.5% | 0.649 | 0.238 |

All qm runs are above the +5% cap. At matched MAE, qm gives no tail gain over the threshold
form. It does remove most of the grid striping. At high weight it produces a new artefact:
cross-hatched "x" spots at local maxima.

**Study 2.**
- FM on hourly data works: σ_r = 0.012668, best val_fm 0.2323.
- The threshold-form dist fine-tune beat clean FM by more than eval noise.
- The threshold-form energy fine-tune was just continued FM training, because its reward was
  0.13% of the gradient.

After the fixes (§41–§42), on val_extremes1024:
| row | exc. ratio | RL bias | FSS@50.1 | RAPSD | RMSE |
|---|---|---|---|---|---|
| FM clean (2 draws) | 0.298 / 0.256 | 0.367 / 0.353 | 0.285 | 1.14 | 1.153 |
| energy control (w = 0) | 0.187 | 0.318 | 0.263 | 1.30 | 1.153 |
| FM + dist_qm | 0.273 | 0.340 | 0.267 | 1.20 | 1.152 |
| FM + dist_qm_ret | 0.465 | 0.185 | 0.304 | 0.74 | 1.154 |
| **FM + energy_qm** | **0.665** | **0.089** | **0.348** | **0.41** | 1.160 |

Known weak point: the low RL bias hides an overshoot. The top-3 patch maxima are 1.3–1.9× too
large, and the fitted ξ is 0.23 against 0.155 observed. dist_qm_ret gets the largest maxima
about right, but it has the highest anisotropy of any FM row (1.47), unexplained.

**Hypothesis tests on the data** (§43; pre-registered).
- The H1 analogue check is inconclusive (2.08× against the 2× bound).
- The trend test gives no verdict, because its validity gate failed.
- The exploratory reading is closest to "extreme shapes are seen shapes, re-weighted with
  intensity" (W2). It does not support "extrapolate a trend".

## 7. Established findings and working hypotheses

From CLAUDE.md and DECISIONS. Each is labelled with its warrant.
- *Established:* the coupling is correctly wired (§4).
- *Established:* pixel mass, plus thresholds out of the prediction's reach, explain why the
  original reward shaped light rain (§5, §39, §40).
- *Established:* the functionals carry no positional information, so no variant can improve
  FSS through position.
- *Established, mechanism partly inferred:* on a deterministic model the loss produces a fixed
  "stamp" texture. MSE's double penalty, plus grid estimators that break rotation symmetry,
  select one orientation. A generative model removes the dilemma; this is the argument for
  Study 2 (`notes/minkowski_deterministic_limitation.md`).
- *Working hypothesis* (§40): an auxiliary objective helps the tail if two things hold. First,
  its high-intensity gradient share exceeds the base loss's (~1% for FM). Second, it gives
  gradient to samples that fall short of the target's tail. Not yet tested across losses (todo 11).
- *Inference, untested:* the 15-min dist fine-tune moved the tail because its reward put ~5×
  more gradient on ≥ 20 mm than the FM loss does.

## 8. Theory framework (RESEARCH_NOTES §1–§4; roadmap M5)

**Central hypothesis.** Geometry is the transferable handle. The excursion-set geometry of
ordinary storms resembles that of extremes, so a model trained on the functionals carries
geometry over to intensities it has not seen.
- **H1 (data):** normalised γ curves collapse across intensity classes.
- **H2 (learner):** the model ties high-threshold geometry to geometry learned at lower
  thresholds.
- **H3 (the claim that matters):** beyond the training range, a Minkowski-trained model
  degrades less. R4 tests this, as an interaction: Minkowski degrades less from R1 to R4 than
  MSE does.

Four predictions are stated in advance, including that the gain comes from P and χ, not A alone.
The area-only ablation tests that one.

**What counts as an extreme** (decided). E1 is an intensity extreme within a familiar regime;
it is primary. E2 is a novel event type, kept as a stress test; a held-out storm type was dropped
because 73 Swiss events are too few. E3 is locally extreme relative to local climate, treated as a
normalisation of E1 or E2. The unit is the 1-h accumulation by tile maximum.

**Regimes (M3, open).** Classify the regime from what is available at inference: the low-res
input, the prediction, or the predicted functionals. Start from ECMWF ecPoint gridbox weather
types. No prior work classifies regimes by Minkowski functionals; AghaKouchak et al. 2011 is the
nearest precedent. This search was not exhaustive.

**W₁ identity** (§13). Since A(u) = |D|(1 − F(u)), the integral of |ΔA| over u equals |D|·W₁
between the marginal intensity laws. The log-area channel with dξ = du/u is a reweighted W₁.
This links the work to Chang & Sapsis 2026 (doi:10.1038/s41467-026-76811-x), Theorems 4 and 7.
To be added to the theory section (paper todo).

## 9. Evaluation protocol

Rules that have each cost days at least once (CLAUDE.md, §10, §12):
- **Read results in this order:** activity (ΔMAE ≥ +1%), anisotropy and images, then the tail
  columns (exceedance ratio and RL bias).
- **GPD ξ error is a secondary diagnostic only,** because ξ and σ trade off.
- **`minkowski_distance` is circular** for Minkowski-trained models.
- **`rmse_extreme` is weak** as a discriminator; **`val_mse` is useless** for judging a run.
- **One POT level per table** (u = 50.1 mm on v4). `gpd_xi_obs` is the sanity check: 0.183 on
  full val, 0.155 on valext.
- **For FM, evaluate `fm_mink_latest.pth`, not `best`.** Use 16 members, heun-16; the cheap
  sampler is a different regime (§11).
- **Pin σ_r per backbone** and check the startup line.
- **Uncertainty** (§32): paired day-block bootstrap plus leave-period-out. On v4 test, 20 of
  486 days hold 27% of pixel-hours ≥ 20.2 mm. Differences inside the interval are not claimed.
- **Robustness ladder** (§43): R0 bulk → R1 in-distribution extremes → R2 product only (paired
  days) → R3 product + time (NIMBUS) → R4 intensity beyond training (cut at u_c, tile-hours
  only, 1.08% of train, random-drop control) → R5 all shifts → R6 DPC Italy.
  - R4 uses a Minkowski grid truncated to 13 thresholds (< 50.1 mm) in both arms. Otherwise the
    loss would teach the model never to exceed the cut, and H3 would be rejected mechanically.
  - Scores conditioned on the observation are paired with scores on cases selected from the
    coarse input (forecaster's dilemma), plus threshold-weighted CRPS.

## 10. Plans and open work

**Robustness ladder (EXPERIMENTS §5).**
- Done: A1–A4 (rules), C1–C4 (code), D1–D3 (model choices).
- Running: B1, B2, E1.
- Open: B3 CombiPrecip, B5 NIMBUS gallery, D4 (energy_qm seeds), E2 (FM R4 arm), all of F
  (scoring).

**Study-2 follow-ups.**
- 3 seeds for energy_qm.
- Bulk check on dist_qm.
- λ_s = 0.5 is an untested default.
- Energy score L1 vs L2.
- Threshold grid as a hyperparameter (banked; plan E0–E3, including extending the grid to
  125/200 mm to curb the overshoot).
- Spread decomposition (banked).

**Competing losses (M4).** Fixed and put at parity for R4 (D3). FM reward paths for the
competitors are not implemented; `reward_finetune.py` hard-codes Minkowski.

**Small items.**
- Plot units say "mm h⁻¹" on hourly data.
- Slide 10 of `mch_slides.pdf`.
- Per-patch anisotropy ranking for dist_qm_ret.
- Day-block intervals on ξ.
- Fix the threshold for the striping measure.
- Truncate the area-only and FM rewards for R4.

**Data.**
- Pre-break ablation.
- Rule-5 revision for co-failing radars.
- Event coverage options B/C.
- Re-run the v3 gauge validation at ring threshold 89.
- Fill or drop the empty `quality_map` array.
- Re-derive the hourly threshold grid on `patches_v4_adv`. It was derived on plain sums;
  u_c = 50.1 mm was re-confirmed, but the grid was not.
- Read EURADCLIM.
- Decide OPERA vs RADKLIM as the main dataset (open in the data todos).

**Roadmap.** The text of M1–M6 in EXPERIMENTS §6 has not been updated for v4.
- **M1** (data analysis): done for v4.
- **M2** (cut-off o.o.d.): implemented as R4.
- **M3** (regimes): open.
- **M4** (competing losses): partly done.
- **M5** (theory): open.
- **M6** (DPC Italy): gated on R4.

**Paper (EXPERIMENTS §5 "Paper").**
- Hourly weight calibration and bracket.
- Deterministic limitation note.
- How a loss coupling is checked: validator, gradient audit, activity check, reward-drop test,
  images.
- Systematic literature search before claiming novelty on weight selection. The candidate
  contributions are cross-dataset weight transfer by gradient share, and hacking-bounded weight
  selection in precipitation SR.
- W₁ identity.
- Exceedance ratio and RL bias as headline columns.
- The POT level as an estimability choice.
- The w = 1e-3 hacking episode as a positive result.
- Dataset coverage limitation.

**Banked projects.**
- Label-free radar cleaning by structure (`notes/future_project_radar_cleaning.md`).
- Training on several hourly datasets.
- Unsupervised artefact segmentation.
- The emulator loss (lowest priority).

**Dropped.** DDPM (§14). Held-out storm type. Held-out period within ODYSSEY.

## 11. Literature

The full list, grouped by theme with the file each entry comes from and its verification mark,
is appendix F, Part 2 (about 120 entries, mostly radar quality and the radar-meteorology reading
list in RESEARCH_NOTES §7.5 and §8). Summary of what exists:

- **Positioning.** Chang & Sapsis 2026 is the closest and most cited. The DOI is in the notes,
  but its title is written two ways. Further points: the emulator paper, EURADCLIM (Overeem et
  al. 2023, ESSD 15, 1441), MLCast, generative downscaling practice, and DRaFT (Clark et al.,
  ICLR 2024, arXiv 2309.17400) for the reward coupling.
- **Loss weighting** (`notes/weight_selection_literature.md`, preliminary). GradNorm (Chen et
  al. 2018); Wang, Teng & Perdikaris (arXiv 2001.04536); Zhong & Maki (CVPR 2020, the closest);
  Du et al. on gradient similarity; Blau & Michaeli (perception–distortion, CVPR 2018); and the
  Goodhart / reward-overoptimisation papers, as analogy only.
- **Scoring rules.** Gneiting & Raftery 2007 (energy score; marked "not yet opened"); Pinson &
  Tastu 2013; Scheuerer & Hamill 2015 (variogram score).
- **Radar quality.** EURADCLIM; Lepetit et al. 2022; Bøvith 2008; Scovell et al. 2013; Franch et
  al. 2020 (TAASRAD19, for the Mahalanobis mask); the agency cleaning chains; RADKLIM (DOI
  10.5676/DWD/RADKLIM_RW_V2017.002); EURADCLIM v3.0 (DOI 10.21944/7ypj-wn68).
- **Events.** Grams et al. 2014 (2013 floods); Kunz et al. 2018 (Andreas).

**Citation debt** (needs fixing before writing):
- `notes/references_reward_coupling.bib` holds only 2 entries (DRaFT, WGAN), and
  `minkowski_reward_coupling.tex` has 6 `\citeq` placeholders.
- `[CITE?]` items:
  - Hadwiger's theorem;
  - Cauchy–Crofton / digital perimeter;
  - straight-through estimator (Bengio et al. 2013; full in MINKOWSKI_DOCS, unverified in the tex);
  - flow-matching marginal velocity theorem;
  - energy distance;
  - double penalty;
  - forecaster's dilemma;
  - threshold-weighted CRPS;
  - Blau & Michaeli.
- No formal citation anywhere for FSS, GPD/POT or Marshall–Palmer.
- `gradient_audit.tex` references are marked "from memory".
- Some RESEARCH_NOTES entries are marked † (from memory, re-verify).

## 12. Resources

- **Compute.**
  - Our GPU, UUID `GPU-9e0aab27-…`, is on node32 since 2026-11-03; before that it was on
    node34. GPU 0 is never ours.
  - Cores 4–11 only, with at least 4 cores always free. Enforced by `src/node_limits.py` and
    `scripts/hpc/env.sh`.
  - Borrowed node40 GPUs 0–3 and CPUs 0–5 until about 2026-11-09, via `node40_run.sh`.
  - Environment: micromamba `dl-stable` (torch 2.10).
- **Data:** `/home/fquareng/work/data/extremes/OPERA/` (raw archive, stores v2/v3/v4/v4_adv,
  quality_v2/v3/v4, DEM `europe_dem_laea.tif`). SMN gauges are in `.../extremes/MCH/smn/`, DWD
  10-min gauges in `validation/gauges/raw`.
- **Main config:** `configs/config_v4_adv.yaml`. FM: `config_v4_adv_fm*.yaml` (σ_r 0.01266805).
  Screen: `configs/quality_v4.yaml`. R4 configs: `logs/r4_configs/`.
- **Code:** `src/{data,losses,models,trainers,evaluation}`. Scripts are grouped in
  `scripts/{train,evaluate,preprocess,dataset_v2,data_quality,analysis,hpc,validate,validation}`.
  `tools/gradient_audit.py`. 22 test files. Stage validators write to `validation/`; all
  2026-10-21 reports pass.
- **Write-ups in `notes/`:**
  - `gradient_audit.pdf` (09-23);
  - `minkowski_reward_coupling.pdf` (10-30);
  - `data_quality_assessment.pdf` (09-23, Part II 10-03);
  - `dataset_v4_slides.pdf` (10-07);
  - `mch_slides.pdf` (19 questions for MeteoSwiss);
  - `slides/2026-10-14_screening/`;
  - `opera_hourly_screen.md` (methods draft for the dataset);
  - `data_analysis_v4/*.md`;
  - `events.md`;
  - status notes of 10-19 and 10-21.
- **People:** Daniele Nerini and Lionel Moret (MeteoSwiss); Tom Beucler and Ryan Cotsakis
  (co-authors, emulator paper); the FBK contact for the DPC Italy data (franch@fbk.eu).
  The README also thanks D. Domeisen, V. Chavez, O. Miralles and L. Räss.

## 13. Inconsistencies to resolve when restructuring

- **`README.md`** says the splits are chronological, but they were random at patch level (§15)
  until v2.
  - The README also still shows the old repo name `Mink-DDPM/` and env `mink-ddpm`.
- **`MINKOWSKI_DOCS.md`** documents Q = 21 quantile thresholds. The current loss uses 15 fixed
  log-spaced levels.
- **CLAUDE.md "Reading results"** quotes v1 values: vanilla MAE 0.04994, anisotropy 1.87, and
  `gpd_xi_obs` +0.176 at u = 31.
  - On v4_adv the references are MAE 0.04390, anisotropy 2.12, and `gpd_xi_obs` 0.183 at
    u = 50.1 mm.
  - On v4, Minkowski 6e-5 has anisotropy 0.56, i.e. *below* 1. The "materially higher value"
    test therefore does not catch the hourly striping. The notes say this, but CLAUDE.md does
    not.
- **CLAUDE.md "Established findings"** gives the FSS and exceedance gains on 15-min v1 data.
  The v4 equivalents are in §6b above.
- **σ_r note in CLAUDE.md**: `patches/precip/residual_stats.json` (v1) still holds the
  Minkowski-backbone value 0.019812. The v4_adv store has its own file (0.012668), which
  matches the v4 FM configs.
- **Chang & Sapsis title** is written two ways.
- **The §16 bracket** is written as {λ*/3 … 10λ*} around gradient parity, while the chosen
  centres match the Adam-update share with a ×1/3 … ×3 bracket. The file does not reconcile the
  two.
- **The EXPERIMENTS roadmap text (M1, M2, M4)** predates v4 and the ladder.
- **The qm 7.7e-3 result** is in `eval_results/` but not in the notes (section 2 above).
