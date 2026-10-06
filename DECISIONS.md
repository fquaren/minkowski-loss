# Decision log

Why things are the way they are, and — more importantly — **what has already been ruled out
and should not be re-tried**. `EXPERIMENTS.md` records *what* the numbers are; this records
*why* we believe them and which roads are closed.

Each entry states the question, the evidence, the conclusion, and how confident it is.
Where a conclusion could be overturned, the entry says what would overturn it.

---

## 1. The threshold grid: quantiles → fixed log-spaced physical levels

**Question.** The original grid was 21 climatological quantiles topping out at
q = 0.99 ≈ 6.5 mm/h, while extremes are ~123 mm/h. How should the grid reach the tail?

**What we found.** Extending the quantile grid alone does nothing. The loss integrated
`dq`, so a threshold in [0.999, 0.9999] carries weight 1e-4 however far you extend — the
grid and the integration measure have to change together.

**Decision.** Fixed log-spaced physical thresholds, 0.1 → 150 mm/h, 15 levels, integrated
over ξ = log(u) so each decade of intensity weighs the same. Verified numerically: the
deepest threshold's weight rises by ~400× versus the old measure, and weights are near-equal
across levels. Fixed physical levels also stay comparable across datasets and can be aligned
with the evaluation levels.

**Rejected alternatives.** Extended quantiles with uniform dq (tail still unweighted);
log-exceedance quantiles ξ = −log(1−q) (equivalent in spirit, implemented then reverted —
less interpretable and not alignable with evaluation thresholds).

**Consequence to remember.** Any Minkowski distance or isoperimetric number computed on the
old grid is **not comparable** to one computed on this grid. Regenerate γ targets whenever
the grid changes.

Confidence: high — the weighting argument is arithmetic, not empirical.

---

## 2. The structural loss weight is bounded above by reward hacking, not by accuracy

**Question.** How large can the Minkowski weight go?

**What we found.** At w = 1e-3 the deterministic backbone produces grid-aligned filamentary
striping absent from the target and overshoots patch maxima by 30–60%, while **every**
structural metric improves: SAL S 1.27 → 0.14, isoperimetric violation 4.57 → 3.47%, RAPSD
3.05 → 0.90. Only the accuracy cost (+14.7% MAE, +17.8% extreme MAE) registers the problem.

**Why the metrics miss it.** Each headline metric is individually blind to thin grid-aligned
structure: the radially averaged spectrum integrates over direction; SAL amplitude compares
domain means, not peaks; the isoperimetric inequality P² ≥ 4πA is satisfied *more*
comfortably by high-perimeter filaments.

**Decision.** w = 1e-4 is the working point. Two new diagnostics were added because of this
episode and are now part of the protocol: **directional anisotropy** (axis-aligned vs
diagonal high-wavenumber power) and **peak ratio** (predicted/observed patch maximum).
Neither is optimised by any loss, so neither can be gamed.

**Scientific value.** This is a concrete demonstration that matching Minkowski curves is
necessary but not sufficient — Hadwiger completeness concerns the basis of valuations, not
identifiability of the field. Worth reporting, not hiding.

Confidence: high — visually unambiguous in the sample images.

---

## 3. The Minkowski loss helps the deterministic backbone and not the generative model

**Evidence, deterministic.** On the full test set at w = 1e-4: FSS@89 0.040 → 0.159,
exceedance ratio 0.118 → 0.244, return-level bias 0.643 → 0.445, RAPSD 3.10 → 1.31, for
+3.8% MAE. All independent of the training objective.

**Evidence, generative.** Two separate routes, both negative:
- *as a reward* (DRaFT-K on the clean endpoint): FSS, ξ error and return-level bias all
  slightly worse than the clean flow-matching model;
- *via the backbone* (flow matching trained on the Minkowski backbone): return-level bias
  +43%, FSS@89 roughly halved, exceedance ratio and extreme peak ratio both down, while CRPS
  and RMSE_ext were unchanged — a regression concentrated in the tail.

**Explanation.** Minimising MSE gives the conditional mean, and
`1[E[y|c] ≥ u] ≠ E[1[y ≥ u] | c]` — the excursion geometry of the conditional mean is
systematically wrong, and no amount of MSE training fixes it. A *calibrated sampler* gets
`E[A(u)] = |D|(1−F(u))` for free. So the loss supplies information the deterministic model
structurally cannot have and the generative model already has. The clean flow-matching model
indeed beats the Minkowski backbone on every tail column.

Confidence: high on the deterministic gain; medium-high on the generative null, which rests
on two experiments rather than many.

**What would change it.** A much larger tail-sample budget (see §6), or a positional term
added to the reward.

**Status 2026-09-23: reopened for the generative half.** The 09-16 re-evaluation
(EXPERIMENTS §2) moved the FM rows. On the same 4,096 patches:
- The distributional reward (`Extremes_20260804_170256`) now *beats* clean FM on FSS@89
  (0.321 vs 0.315), exceedance ratio (0.534 vs 0.475) and RL bias (0.114 vs 0.165).
- The energy coupling improves exceedance ratio and RL bias at an FSS cost (0.269).
- FM on the Minkowski backbone is no longer a tail regression (RL bias 0.141 vs 0.165), but
  still loses FSS (0.273).

The "via the backbone" evidence above (+43% RL bias, FSS halved) was measured before the
σ_r guard, so the cause of the change needs attributing before either version is quoted.
The deterministic half is unaffected. Until then, treat "does not help flow matching" as
**unsettled**, and weigh it against the single seed and the leaking split (§15).

---

## 4. The coupling is correctly implemented — do not go looking for a bug

**Question.** Was the null result a plumbing failure?

**Evidence.** Single-batch diagnostic with the base noise held fixed: the reward falls
**59.7%** in 40 steps, five times its own sampling noise. Gradients agree across truncation
depths (cos(∇K=1, ∇K=2) = +0.91).

**Important correction.** An earlier run of the same diagnostic reported cos = −0.20 and
"broken coupling". That was an artefact of comparing gradients computed on *different*
base-noise draws — the reward is stochastic, so an un-fixed noise draw makes the test
meaningless. The fixed-noise version is the valid one. K = 1 is fine.

Confidence: high. Re-run `scripts/evaluate/diagnose_coupling.py --extreme_batch` if in doubt;
it takes minutes.

---

## 5. The binding limit is pixel mass, and the anneal parameter cannot fix it

**Evidence.** Per-threshold gradient decomposition on an extreme-containing batch (max
142 mm/h):

| threshold | share of gradient | residual |
|---|---|---|
| ≤ 0.28 mm/h | **83%** | small |
| 19 mm/h | 0.13% | 1.23 (largest) |
| ≥ 31 mm/h | ~0 | 0.8–1.1 |

The structural error is largest exactly where the gradient is zero. Cause: only 0.005% of
pixels exceed 31 mm/h and essentially none exceed 53, so the excursion sets that define the
extreme thresholds are empty in almost every sample.

**Anneal sweep (0.05 → 0.40).** The absolute gradient at 31 mm/h rises by *eleven orders of
magnitude*, but the tail's **share** of total gradient stays at 0.75–0.88% and slightly
declines, because widening the indicator amplifies the bulk equally — while the bulk
excursion-set estimate degrades (residual at 0.48 mm/h doubles).

**Decision.** Keep anneal at 0.05–0.10. **Do not sweep it again.** Do not sweep K either
(§4). The remaining levers are threshold reweighting and sample selection, not smoothing.

Confidence: high — both the decomposition and the sweep are direct measurements.

---

## 6. Scale is the principled route to the tail

The quantities the loss targets are ensemble expectations — area is the marginal survival
function, β₀ at high u approximates the field-maximum exceedance probability. Expectations
can be estimated accurately from many sparse samples even when each individual sample
carries almost no tail information. That is the argument for a larger allocation, and it
follows from §5 rather than being asserted.

Two things it will **not** fix, and we should not claim otherwise:
- **FSS.** The functionals are rigid-motion invariant and carry no positional information.
  Every coupling that improved distributional metrics cost FSS. More samples improve the
  tail *distribution*, not placement.
- **Identifiability.** Cf. Chang & Sapsis (2026), *Extreme Event Aware Learning*,
  doi:10.1038/s41467-026-76811-x: enforcing observable statistics yields distributionally
  calibrated but spatially unidentified extremes (their 20-member ensemble placed the
  inferred extreme with spatial sd 3.99 against a true maximiser norm of 2.86).

---

## 7. The energy score is the right reward form, and it is proper

**Question.** The per-sample reward regresses an ensemble mean onto a *single* observed γ —
one draw from p(γ|c). Can that be fixed?

**Decision.** Yes: the energy score (kernel CRPS) on the functional curves, with the
Minkowski distance as the metric and the **fair** M(M−1) normalisation:

    S = (1/M) Σ_m d(γ̂_m, γ) − 1/(2M(M−1)) Σ_{m≠m'} d(γ̂_m, γ̂_m')

**Verified, not assumed.** In a synthetic test it is minimised at the *true* ensemble
dispersion, while per-sample matching on the same ensembles prefers **zero** spread — the
diversity-collapse mechanism, demonstrated rather than argued. The negative pairwise term is
itself the spread reward, so `REWARD_SPREAD_WEIGHT` is ignored in this mode.

**Result.** Best coupling so far: best ξ error in the table (0.043), highest exceedance
ratio (0.523), and a *uniform* return-level under-prediction (−24%/−18%/−11%) rather than
the earlier sign-crossing bias. Tail class right, scale ~15% short. Still does not beat the
clean flow-matching model on FSS.

*Superseded numbers (2026-09-23):* after the 09-16 re-evaluation, the energy row's ξ error
is 0.067, not the table's best, and its RL bias is 0.126. It does still hold the best
exceedance ratio (0.538) and RAPSD (0.55) in the FM block. See §3.

**Also added because of this.** `REWARD_FM_RETENTION_WEIGHT` — the fine-tuning objective
previously contained **no pixel-aligned term at all** (a rigid-motion-invariant reward plus a
velocity-space anchor), so nothing supplied placement. That is a plausible cause of the FSS
regression in both routes. Untested at weights above 1.0; FSS still fell at 1.0.

---

## 8. Competing losses are all inactive or harmful (Study 1)

Four alternatives were implemented and trained: spectral (FFT-magnitude energy score), SSIM,
soft wet-area, optical-flow advection consistency.

**Round 1 (shared weight).** All inert: MAE cost only +0.6–0.8% against vanilla, where
Minkowski at its working weight pays +3.8%. RAPSD got *worse* under a loss that optimises the
spectrum directly — the signature of a weight too small to enter the objective.

**Round 2 (per-loss weights).** Still inert by the MAE check (+0.3–0.7%), and what effect
they have is negative: exceedance ratio 0.036–0.068 against vanilla's 0.118, return-level
bias 0.69–0.75 against 0.643, anisotropy 2.1–2.6 against 1.87. None improves on plain MSE.

**Mechanisms worth stating in the paper:**
- *wet area* thresholds at drizzle and compares binary masks, so it is blind to intensity —
  0.2 and 200 mm/h are identical to it. It is the degenerate single-threshold case of the
  Minkowski area channel, and it fails hardest (RAPSD 4.59, worse than bicubic). Good
  evidence that the *multi-threshold* structure is what does the work.
- *optical flow* uses the **coarse conditioning field** as teacher, which is smooth and has
  suppressed peaks by construction — it teaches the model to resemble its own input.
- *spectral* is translation-invariant by construction, so it provably cannot penalise
  displacement; it scores exactly 0.000 on a displaced field.

**Open.** Whether a genuinely active weight exists for any of them. Use the gradient-norm
rule (§9), not value parity, and require the MAE check to pass before reading any structural
column.

**Status 2026-09-23: not a fair trial, so "harmful" is not established.** A direct audit
(EXPERIMENTS §3) found:
- **Under-weighted and inactive.** Spectral, SSIM and optical flow were trained at about
  10⁻³, 10⁻⁴ and 4·10⁻⁵ of gradient parity at initialisation. At their own checkpoints they
  are far from the equilibrium an active term reaches (share 0.02 / 0.16 / 0.007, against
  1.22 for Minkowski). Their trained models are no better than vanilla on their own loss
  (+2.5%, +0.1%, −2.6% n.s.). Measured with `tools/gradient_audit.py` over 512 batches; see
  EXPERIMENTS §3 and `notes/gradient_audit.pdf`. An earlier 6-batch estimate gave
  "70×/10×/140×" and is superseded.
- **SSIM is buggy.** A perfect prediction scores 0.465 against 0.499 for an all-zero one.
- **Unmatched budgets.** All v2 runs trained 25 epochs, against 47 (Minkowski) and 69
  (vanilla). A 34-epoch vanilla run reproduces their loss values.

So their worse-than-MSE tails are under-training plus an inert term. The mechanisms listed
above (wet area intensity-blind, optical-flow teacher smooth, spectral
translation-invariant) remain valid *a priori* arguments, but they are not yet backed by an
active run. Wet area is the only one that is active; it runs at 12.8× the MSE gradient and is
degenerate. The protocol for the rerun is §16.

---

## 9. Set auxiliary weights by gradient norm, not loss value

Loss *values* span three orders of magnitude across these losses, but value parity is the
wrong target: the Minkowski gradient is threshold-localised (83% below 0.28 mm/h) while SSIM
and spectral gradients are dense over every pixel. Equal value ≠ equal influence.

Use λ = ‖∇L_ERM‖₂ / ‖∇L_aux‖₂, computed after ERM pre-training (Chang & Sapsis 2026). It
matches the quantity that actually drives the update. Treat it as the *centre* of a sweep,
then move down — gradient parity would likely put Minkowski into the hacking regime of §2.

**Correction (2026-09-23).** Measured *at the ERM solution*, the rule is ill-posed. There
‖∇L_ERM‖ ≈ 0 by definition, and a per-batch norm measures minibatch noise, not signal
(`notes/gradient_audit.pdf` §4–5; a 6-batch check at the vanilla checkpoint found the MSE
signal indistinguishable from zero). Measure parity at **initialisation or early training**,
Euclidean and Adam-preconditioned, with `tools/gradient_audit.py`. At a trained
checkpoint, use the equilibrium test instead: share → 1 and cos → −1 for a run trained
with the term.

---

## 10. Evaluation protocol decisions

**ξ error is not a headline metric.** The shape parameter is not identifiable independently
of the scale. Three separate runs have the *best* ξ error in the table alongside the *worst*
tail: wet area 0.007, SSIM 0.009, spectral 0.013 — all producing 3–6% of the observed
exceedances. A too-weak tail can match the decay *form* because a deficit in σ̂ compensates an
error in ξ̂. **Report exceedance-count ratio and return-level bias**; keep ξ as a diagnostic
read alongside σ and the exceedance count.

**One POT level per table.** ξ error and return-level bias are computed against a fit whose
*observed* reference depends on the threshold. Refitting the same MSE backbone at u = 53
instead of 31 moves ξ from 0.03 to 0.33 and return-level bias from 0.64 to 0.43. Sanity
check: `gpd_xi_obs` depends only on the data, so it must be identical in every summary at the
same level (u=31 → +0.176; u=53 → −0.190). `eval_losses.sh` warns automatically if rows mix.

**u = 31 is chosen for estimability, not physics.** Extremes are ~123 mm/h, so the asymptotic
regime is not established at 31; it is the level at which every model yields enough
exceedances to fit. Stated explicitly in the methods rather than glossed.

**Return levels are parameterised by expected exceedance count λ, not raw m.** The level
exists only when m·rate > 1, and the pixel-level rate is ~1e-6, so fixed m values are
unreachable and return NaN. Given λ, m = λ/rate_obs, and the same m is used for both fits.

**Evaluate `fm_mink_latest.pth`, not `fm_mink_best.pth`.** "Best" is selected on validation
reward — the training objective — so when the tail overshoots it picks the most overshot
checkpoint. Circular selection.

**RMSE_ext is a weak discriminator.** It varies ~1% across samplers that differ 46% in peak
height. Do not let conclusions rest on it.

---

## 11. The cheap sampler is a different regime, not a cheap proxy

A sampler-fidelity check on the *same* clean checkpoint, identical patches:

| | peak ratio | FSS@89 | RAPSD | RL bias |
|---|---|---|---|---|
| heun-16 | 0.816 | 0.280 | 0.738 | 0.165 |
| euler-16 | 0.589 (−28%) | 0.200 (−29%) | 1.477 | 0.341 |
| euler-8 | 0.441 (−46%) | 0.146 (−48%) | 2.158 | 0.429 |

Euler-16 loses 28% of median peak height and 41% on extreme patches — precisely the
quantities a structural reward is meant to move. **Train and evaluate with heun-16.** Fund it
by reducing the per-step batch rather than the ensemble: M is what the E[γ] estimate needs.

Note ξ error is nearly sampler-invariant (0.072–0.083) while everything amplitude-related
collapses — ξ is close to scale-free, another reason not to headline it.

---

## 12. Numerical and infrastructure decisions

- **σ_r is a property of the residual, so it changes with the backbone.** Recompute it after
  changing backbones, and pin `FM_RESIDUAL_SCALE` explicitly per model rather than relying on
  `residual_stats.json`, which the last `compute_residual_stats` run overwrites. A silent
  mismatch here once produced a 53× reconstruction error that looked like catastrophic model
  failure. Always verify the startup line.
- **fp16 overflows** once the residual is correctly scaled — train flow matching with
  `--no_amp` (fp32) or bf16.
- **`CUDA_VISIBLE_DEVICES` alone is not enough**: CUDA's device order differs from
  `nvidia-smi`'s, and `micromamba run` overwrites the variable set outside it. Set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` too, inside the wrapper.
- **Launch long runs under `setsid`/Slurm** — a tmux exit leaves the DataLoader worker pool
  unreaped; the node has previously accumulated ~193 zombies alongside stuck root NFS threads.
- **`val_mse` is dry-pixel-dominated** and saturates by epoch 2. Never judge a run by it.

---

## 13. Related work positioning

Chang & Sapsis (2026), *Extreme Event Aware (η-) Learning*,
doi:10.1038/s41467-026-76811-x. Closely related and worth citing carefully:

- **Exact identity worth adding to the theory section.** Since A(u) = |D|(1−F(u)),
  ∫|A_pred(u) − A_targ(u)| du = |D| · W₁ between the marginal intensity laws — *exactly*.
  Our loss uses log(1+A) and dξ = du/u, so it is a **reweighted** W₁. This lets us cite their
  Theorem 4 (data-consistent estimators provably miss the tail — our MSE backbone at
  604/35,482 exceedances is that lower bound) and Theorem 7 (W₁ is tight for tail error).
- **Where we are ahead.** They name vector observables as future work; our γ is a 3×Q vector
  observable with per-channel extreme-value interpretations. We also have a demonstrated
  degenerate-solution failure mode (§2) and a diagnostic for it — a scalar max is harder to
  game.
- **Where they are ahead.** Their gradient flows through the argmax pixel, which always
  exists, so they never hit our §5 bottleneck. Their tail-sample selection is directly
  adoptable.
- **Careful with the framing.** Their regime is *no* extreme data; ours is extremes-present
  but gradient-starved. The remedy transfers; the framing should not be borrowed wholesale.
  If citing in a compute proposal, invert it: statistical regularisation gives calibrated but
  unidentified extremes and depends on a reference law whose misspecification propagates
  (their RMSE +20.6% under tail perturbation).

**Important caution on tail-sample selection.** If adopted, apply the tilted sampling to the
**structural term only**. Keep MSE / the flow-matching velocity loss on uniformly drawn
batches — otherwise μ is no longer E[y|c] and the residual decomposition breaks.

---

## 14. DDPM is dropped; the emulator loss is lowest priority

**Decision (2026-09-23).** Flow matching replaces the DDPM as the generative model. It is
the Study-2 model everywhere, and the DDPM is no longer trained, compared or reported. The
code and `runs/sr_ddpm/` are kept, not deleted.

The learned (emulator-based) Minkowski loss stays in the codebase but is the lowest
priority. It is worth resuming eventually as the learned counterpart to the analytical loss,
after the data rebuild and the Study-1 rerun.

---

## 15. The current splits are random at patch level, so test is not independent of train

**What we found** (`scripts/data_quality/audit_splits.py`, 2026-09-23). `split_metadata.py`
shuffles individual patches over the whole period, so train, val and test all span
2023-08-01 → 2024-10-30 on the same 79 tile locations. For **100%** of test patches, the tail
included:
- a train patch exists at the same timestamp;
- a train patch exists on the same tile within ±1 h.

For 98–99%, a train patch exists on an adjacent tile at the same timestamp. So the test set
holds the neighbouring 15-minute frames and neighbouring tiles of storms the model trained
on.

**Consequence.** Every current score measures *interpolation within seen events*. That is
legitimate for an in-distribution table, if stated. But it cannot support a
generalisation claim, and it gives no information about unseen extremes, which is the
project's central claim. It also flatters every model roughly equally, so the relative
ranking in Study 1 is not obviously wrong. Absolute numbers and the size of the gaps may be.

**Decision.** The rebuilt dataset gets splits **blocked in time by event** (whole days or
storm episodes, with a gap of at least a few hours between splits). The o.o.d. split (tab:ood)
is built on top of that. The existing patch-level splits stay only for reproducing the
current tables.

Confidence: high. It is a direct count.

---

## 16. A fair trial for auxiliary losses: gradient-norm bracket, matched budget, seeds

**Question.** How should λ be set so that Study 1 compares losses rather than weights? The
candidates were homoscedastic (Kendall) uncertainty weighting, an Optuna search, and a
calibrated grid.

**Why not homoscedastic weighting.**
- *It balances loss values.* It learns 1/σ² weights from each term's residual scale, which
  §9 already rejects as the wrong target: the Minkowski gradient is threshold-localised
  while SSIM and spectral are dense.
- *Its Gaussian likelihood does not fit these losses.* It is a likelihood argument for
  Gaussian residuals, and none of these losses is a Gaussian NLL. Several have non-zero
  floors (SSIM's is 0.465), so a learned σ can simply absorb the floor.
- *It turns the comparison into a comparison of learned weights.* A different, uncontrolled
  weight per loss is exactly what the comparison has to avoid.

It is useful for balancing the three Minkowski *channels* (already used that way in the
emulator), not MSE against an auxiliary term.

**Why not Optuna as the primary tool.**
- *λ is one dimension per loss*, and the useful range is bounded above by hacking (§2), so
  a bracket of 3–4 values covers it.
- *Each trial is a full training run.*
- *The selection metric decides the outcome.* Whatever Optuna maximises (exceedance ratio,
  FSS, MAE-constrained) becomes the result, and it must not be any loss's own objective.

Optuna earns its place only for multi-dimensional searches (λ together with anneal or tail-
sample fraction). Even then it should run on short proxy runs with pruning, with an equal
trial budget per loss.

**Decision (proposed; to be confirmed when M4 starts).** For each loss:
1. Measure parity λ* = ‖∇L_MSE‖/‖∇L_aux‖ at initialisation or early training, not at
   the ERM checkpoint (§9 correction; `tools/gradient_audit.py`). The EXPERIMENTS §3 values
   are noise-referenced first estimates.
2. Train a fixed log bracket {λ*/3, λ*, 3λ*, 10λ*}. Minkowski at its working point is ~5λ*,
   so the bracket must reach above parity.
3. Use one fixed epoch budget for every run, vanilla included, and evaluate the **last**
   checkpoint, with no early stopping and no "best" selection. The current `val_monitor`
   adds λ·val_aux (`unet_analytical.py:244`), so it selects on the auxiliary objective. The
   alternative, `val_mse`, saturates by epoch 2 (§12).
4. Pick each loss's point by one pre-registered rule: the best exceedance ratio subject to
   ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla. Then run 3 seeds at that point.
5. Report the whole bracket, not only the chosen point, so "cannot be made active without
   hacking" becomes a visible result instead of an assertion.

For FM, the same losses enter as rewards through `reward_finetune.py`, with the same bracket
logic on `REWARD_WEIGHT`.

**Chosen centres (2026-09-23).** These are now the defaults in `config.yaml`
`STRUCTURAL_LOSS_WEIGHTS` and `src/losses/competing.py`. Rule: give each term the same share
of the actual **AdamW update** that Minkowski has at its validated 1e-4, measured at the
vanilla checkpoint with that checkpoint's Adam second moments (`tools/gradient_audit.py`,
512 batches). Minkowski at 1e-4 is the one weight known to be active, at equilibrium and
free of hacking. Matching its update share treats every loss identically, without assuming
that parity is optimal.

| loss | old (v2) | new | Adam-matched value | init share at new λ |
|---|---|---|---|---|
| minkowski | 1e-4 | **1e-4** | (reference) | 0.095 |
| spectral | 5e-4 | **1e-1** | 0.116 | 0.39 |
| ssim | 1.6e-4 | **2e-2** | 0.0196 | 0.02 |
| wetarea | 2e-2 | **2e-2** | 0.0198 | 0.99 |
| opticalflow | 4e-5 | **2.5e-2** | 0.0239 | 0.02 |

These are bracket *centres*, not tuned optima. Minkowski's value is validated, and the other
four are principled starting points that the M4 bracket must test (×1/3 … ×3), judged by the
equilibrium and own-loss tests at the end of training. SSIM's weight belongs to the current,
floor-dominated implementation, so re-measure it after the data-range/eps fix.
**Resolution order** in the trainer: `--weight_geom` > `STRUCTURAL_LOSS_WEIGHTS[loss]` >
`MINKOWSKI_TARGET_WEIGHT` (Minkowski only) > built-in defaults. The launcher default is
now `config`; it was `0.001`, the hacking weight.


---

## 17. Screening policy: reject clear errors only, keep imperfections, no labelled set

**Question.** How strict should the artefact screen be, and does it need a labelled set to
calibrate it?

**Decision (2026-09-28, researcher).**
- **Discard freely, but only clear errors.** The archive is large (~13 years of 15-min
  composites over Europe, 2012-09 → 2025-10), so losing patches costs little. Data that is
  *meaningless* (non-meteorological by construction) must not enter training or evaluation.
- **Keep imperfections.** Patches with small, realistic flaws stay: blockage and range
  underestimation, attenuation shadows, compositing seams, drizzle speckle, mild residual
  clutter. The model should be **robust** to these, and it only learns that robustness if it
  sees them. The goal is not a perfectly clean dataset.
- **No labelled set.** Labelling ~300 tail tiles to tune rule thresholds is not worth it at
  this data volume. This supersedes step 6–7 of the `scripts/data_quality/README.md` loop.

**What counts as a clear error** (RESEARCH_NOTES §7.3): signatures that are
non-meteorological by construction:
- residual static clutter;
- emitter / RLAN rays and rings;
- isolated spikes far above their neighbourhood;
- physically impossible rates (observed: 13,072 and 64,842 mm/h);
- radar-wide constant-signal failures;
- single-frame appearances with no precursor or successor;
- rain under a clear sky, if satellite data is added;
- tiles dominated by nodata.

**The risk this creates, and the safeguards.** Without labels, a rule's precision and recall
on *real* extremes are never measured. For this project that is the dangerous direction. Real
extremes are also "anomalous" (steep gradients, small intense cores, fast evolution), so a rule
that fires on unusual intensity structure removes real storms preferentially and biases the
very tail under study. EURADCLIM's authors say the same of their cleaned OPERA product: its
outliers limit its use "especially for use in extreme value modeling". The safeguards stand in
for labels:
1. Prefer rules keyed on **geometry, persistence and physical impossibility** over rules keyed
   on intensity structure alone.
2. **Report the rejection rate per intensity bin**, region and year for every rule. A rule
   whose rejection rate climbs steeply with intensity is suspect until its rejected tiles have
   been looked at.
3. **Look before trusting:** spot-check the gallery of each rule's rejections and of the
   unflagged tail (`patch_gallery.py`). This is a sanity check by eye, a few minutes per rule,
   not a labelling campaign.
4. Where available, cross-check against independent evidence: satellite cloud type (as
   EURADCLIM does), lightning, and severe-weather reports.
5. **The same screen for training and evaluation**, including the held-out o.o.d. extremes
   (M2). Otherwise the o.o.d. tail is contaminated, or is judged on different data.
6. Measure robustness rather than assume it: report skill on the kept-but-imperfect subset
   (e.g. far range, behind blockage) separately.

**Open, to settle after talking to MCH (Daniele Nerini, Lionel Moret; RESEARCH_NOTES §7.4):**
the concrete rule set and thresholds, tile rejection vs pixel masking, the plausibility bound
on the rate, and what replaces the declutter-zeroing above 150 mm/h (EXPERIMENTS §5).

Confidence: the policy is a researcher decision. The risk analysis is high-confidence,
because the direction of the bias follows from what the rules detect.

---

## 18. ODYSSEY and NIMBUS: train on one product, test the other separately

**What we found (2026-09-30).** OPERA changed production chain on **2024-07-05**. The archive
product becomes `RATE` (NIMBUS) instead of `QIND_RATE` (ODYSSEY). The grid, 2 km resolution,
projection and 15-min cadence are the same. What differs:
- *how a frame is made:* ODYSSEY is a composite of the scans in a 15-min window, NIMBUS the
  lowest-elevation PPI at the nominal time;
- *the radar set.*

On days available in both products, NIMBUS:
- has ~30–40% less wet area;
- has intensities 8–20% higher in median, with a wide spread;
- correlates with ODYSSEY at only 0.65–0.80 in log space;
- has about half the +15 min persistence.

Every archive day from 2024-07-05 on is NIMBUS (~475 days on disk). Quantification:
`scripts/data_quality/era_gap.py`, EXPERIMENTS §5.

**Decision (researcher).** v2 train/val/test are drawn from ODYSSEY days only. NIMBUS days
form a separate `nimbus` split (with `events_nimbus`, `light_nimbus`, `extremes_nimbus`),
buffered from the rest like any other split.
- *Why:* mixing products would put ~10% NIMBUS into test, and a model trained mostly on
  ODYSSEY would then score worse on NIMBUS for reasons of product, not of extremeness. That
  is exactly the confound the o.o.d. claims must rule out. Kept separate, NIMBUS becomes a
  clean product-shift test ("does it survive a change of radar product?").
- *What it costs:* ~10% of the data, and several flagship events (Storm Boris, the Valencia
  DANA) are evaluated as shifted-product cases.
- *Reversible:* the era is stored per patch (`era` in `full_*_info.csv.gz`), and the split
  is metadata only (`--nimbus_start ''` disables it).

---

## 19. v4 screen: radar ceilings, radar-wide failures, and a guarded repair

**What we found (2026-10-06).** The POT threshold study on v3 (EXPERIMENTS §5,
RESEARCH_NOTES §7.4d) exposed three failures that every v3 rule misses, because each rule
assumes an artefact that is small, local or transient against real rain:
- a reflectivity ceiling at 364.63 mm/h (exactly 64.0 dBZ under Z = 200 R^1.6) in the Valjevo
  (Serbia) area on 80 rain days of 2023; 12% of all test pixels > 89 mm/h;
- a radar-wide failure of Torrejón de Velasco (Madrid) on 2018-04-29; 16% of all train pixels
  > 89 mm/h. The v3 footprint repair lowered it to its ring median, which is the failing
  radar too, and left a flat 88.27 mm/h plateau (4,865 pixels in one tile);
- a ceiling at 48.62 mm/h (exactly 50.0 dBZ), tile r640 c1408, 7 days of Nov 2019, found by
  the repeated-value test without being looked for.

"Radar-wide constant-signal failures" were on the §17 clear-error list from the start but had
no rule.

**Decisions (researcher, 2026-10-06), on Claude's proposal.**
- *Five rules, all keyed on an error signature* (§17 safeguard 1), none on intensity alone:
  1. repeated value per tile: real rain on the 0.01 mm/h ODYSSEY grid almost never repeats
     one value (per tail tile: median 1, 99.9th percentile 24 repeats of one value >= 31);
     >= 50 repeats rejects the tile. On NIMBUS (values on a ladder with ratio 1.0593 between
     levels) the count is taken in excess of the adjacent levels;
  2. ceiling table per radar area and year: values counted > ~50x the median of their
     +-0.5 dB neighbours. A ceiling pixel is censored, not measured;
  3. radar-disk failure per radar and frame (wet fraction, mean, share of log-rate variance
     explained by range alone, jump at the area boundary and at the maximum-range circle);
  4. guarded repair: `boundary_fill` refuses components above a size limit, and the
     repeated-value test runs after every repair;
  5. radar-day ranking against each radar's own seasonal level and its 5 nearest radars.
- *Ceiling pixels:* a tile-frame with >= 5 ceiling pixels is rejected; with 1-4 they are
  replaced by the median of their valid 8-neighbours, like a spike, and coded.
- *Radar-day exclusions come from a list the researcher reviews by eye*, not from thresholds
  alone (§17 safeguard 3: look before trusting). The reviewed list is versioned in
  `configs/quality_v4.yaml`.
- *Scope:* only these five rules. Static coverage holes (option B) and masked partial tiles
  (option C, `notes/events.md`) stay out, so the v3 -> v4 difference is attributable.
- *Decisions are taken at split time from per-tile columns* (`make_splits.py`), so a threshold
  can change without a rescan (roadmap M1).

**Two corrections to the first proposal**, made while planning:
- the "cross-check against neighbouring radars in the overlap zone" cannot be done directly:
  the composite holds one value per pixel and no per-radar field. It becomes the jump in rate
  across the radar's area boundary and at its maximum-range circle;
- "do not repair when the ring median is >= 31 mm/h" would refuse almost every footprint
  repair: a footprint is the >= 150 mm/h region, so its outer ring sits just below 150 even in
  real storms. The guard is a size limit, calibrated on v3's repairs, plus the post-repair
  repeated-value test.

**The tail fit is an audit, not a rule.** The threshold-stability fit found the failures, but it
cannot tell a failure from a real extreme day, is blind below u, and would make the screen
circular (§20). It runs after every build; it never decides what is kept.

Confidence: the three failures are verified by eye and by their exact dBZ values. That the
rules catch them without removing real extremes is to be shown by the Phase-0 calibration and
the §20 gate.

---

## 20. When the dataset is ready for training: a gate fixed before the results

**Question.** Without labels, cleanliness cannot be measured, and §17 keeps imperfections on
purpose. When do we stop cleaning and train?

**Answer (researcher, 2026-10-06).** The dataset is ready when **one more cleaning step would
not change any number we report**, not when it is clean.
- *Why the bar is set by evaluation, not training.* 83% of the structural gradient comes from
  thresholds <= 0.28 mm/h and >= 31 mm/h contributes ~0 (§5), so a backbone barely sees the top
  of the tail (inference, not tested). The tail columns (exceedance ratio, RL bias, FSS at 89,
  xi) are made of it: in v3 one ceiling was 12% of test pixels > 89 mm/h. The exception on the
  training side is tail-sample selection by patch max, which picks artefacts first.

**The gate** (all five, on the v4 build, before step 3 of the training plan):
1. **Known failures caught.** Every failure ever identified is kept in a regression catalogue
   (`configs/quality_v4.yaml`, `known_failures`) and must be absent from every build.
2. **Residual contamination measured.** 100 test tiles drawn at random from those with max
   >= 89 mm/h (not the top-ranked), looked at by eye. With 0 artefacts the 95% upper bound on
   the rate is ~3% (3/n); otherwise report the rate with a binomial interval. This estimates
   a rate, it tunes nothing, so it does not contradict §17's "no labelled set". Pass: upper
   bound <= 5%.
3. **The evaluation reference is stable.** Remove the 20 non-event days with the largest
   share of exceedances; recompute the observation side of the tables (exceedance counts at
   31 / 53 / 89 mm/h, truncated-GPD xi, observed return levels). Pass: every change is inside
   its day-block bootstrap interval. This is the criterion with the most weight: it says
   further cleaning cannot move the tables.
4. **Real extremes survive.** Event tiles kept (rejection < 0.5% per event), no rule's
   rejection rate rising steeply with intensity, and the radar / gauge exceedance ratio in
   Germany and Switzerland no worse than v3's.
5. **Residual drift in xi(u) is explained** by a named legitimate cause (truncation at
   500 mm/h, regional / seasonal mixing, sub-asymptotic threshold), with the truncated-GPD fit.

**Where we stop.** If the gate passes, v4 is the training set and what remains is accepted as
imperfection. A v5 is built only if criterion 2 or 3 fails, and only for a failure type that
can be named and seen in the images. If the gate fails and v5 is not worth its cost, train on
v4 anyway and report the measured contamination rate as a limitation.

**The rule that keeps this honest: no cleaning by tail fit.** Removing whatever makes xi(u)
drift would select the data by its fit to the GPD and then use the GPD as the evaluation
reference: circular, and it removes real extremes that do not fit (rare regimes, mixtures).
The tail fit may point at candidates; every rule must key on an error signature (a ceiling
value, a range-only field, a repeated value), never on "the fit gets better".

**Truncation.** v3 / v4 hold no values above 500 mm/h (the > 500 cores are repaired), so the
data are truncated, and a truncation alone makes xi fall at high u (synthetic GPD, xi = 0.35:
+0.33 at 31, +0.24 at 89, -0.32 at 250 after removing 0.08% of values). The audit fits a
truncated GPD (density divided by its mass below 500). At u = 31 the effect is small
(0.35 -> 0.33). Whether the evaluation protocol should also fit truncated is open.

Confidence: the gate is a researcher decision. The circularity argument is derivable. The
claim that training is insensitive to tail artefacts is an inference from §5, not tested.

