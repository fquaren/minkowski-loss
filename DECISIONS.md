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

**Rule 2 changed after the first Phase-0 run (2026-10-06, researcher on Claude's proposal).**
The first run found 3,256 radar-year "ceilings" (up to 43 for one radar-year, 2,044 in 2024
alone) where a real ceiling is one value. Cause: a radar-year mixes values on coarse ladders
(dBZ steps of 0.4-3 dB, rounded to 0.01 or 0.1 mm/h) with sparse off-ladder values (counts
1-4). The median of the occupied values within +-0.5 dB is then ~1, so every ladder level
passes 50x; e.g. Málaga 2024, 10.28 mm/h counted 8,059 times among neighbours counted 1-4. On
the event days already scanned this rejected 16.5% of ce_floods_201305 tiles (72% of those
with max >= 31) and 2.5% of andreas_20130728.
- *New comparison:* the count of a value against the **largest** count of any other value
  within **+-4 dB**. The max makes the comparison the adjacent ladder level, whatever sits
  between levels; +-4 dB is wider than the coarsest ladder step in the data (~3 dB, IT 2026).
  Rain counts fall with intensity, so on a ladder the ratio is ~1 or below; a ceiling piles up
  above its neighbours.
- *Calibration histograms (199 days, 23,868 values counted >= 50):* the 18 radar-year rows of
  the two known ceilings score 9.9-89; all other values have 99.9th percentile 1.96. Threshold
  `min_ratio: 5`, provisional until the rerun Phase 0. The 50 of the first proposal had no
  calibration behind it.
- *Not yet explained:* 11.53 mm/h (= 40.0 dBZ) in 2023 scores 3.0-13.7 on the same Balkan
  radars that carry the 364.63 ceiling (itbri 13.7, rsfrg 5.1, grthe 5.1). It is flagged at 5
  and goes to the gallery to be looked at; it may be a second fault of the same radars or a
  ladder artefact.
- Rule 1 is unchanged (+-0.5 dB median, `excess`). Whether mixed ladders inflate it too is
  read from `rej_repeat` in the Phase-0 reports, not assumed.

**Rule 5 gains a fourth signal, isolated high pixels (2026-10-07, researcher on Claude's
proposal).** The 48.62 tile keeps maxima of 138-342 mm/h made of 2-9 pixel clusters in dry
surroundings; no rule sees them, and the clean-frame spike rule cannot (a pixel with any
neighbour >= 10% of its value is not a spike, so cluster pixels protect each other).
- *Not a tile rule:* a max with a dry 5x5 window occurs on 9-16% of kept event tiles >= 31
  mm/h (EXPERIMENTS §5), so any tile threshold removes real storms (§17 safeguard 1).
- *A radar-day signal instead:* `iso31` = owned pixels >= 31 mm/h whose 5x5 window has median
  < 0.1 mm/h and is >= 50% covered, summed over the day's frames, on the `clean_frame` field.
  Scored like the other signals (own seasonal q99, 5 nearest radars, floor 10 pixels). A
  radar-day only enters the review list; nothing is removed without a look. Counts, not
  shares: on radar-days with >= 50 pixels >= 31 the share's 99th percentile is 0.98 on random
  days, i.e. noise.
- *Phase-0 check:* Oradea's failure days have 1,286-8,636 isolated pixels against a
  neighbour median of 0-3, but score < 0 on 6 of 7 days because the calibration set holds 7
  failure days among Oradea's 53 autumn days, so its own q99 (7,525) is a failure day. On the
  full archive they are < 1% of ~1,180 autumn days. Event radar-days: 232 of 15,897 score
  > 0, none >= 0.5. Rule-3 flags and f150 unchanged.
- *Open, decided on the full-archive ranking (rank re-runs in seconds, no rescan):* a radar
  failing on more than ~1% of its season-days hides behind its own q99, for every signal.
  Options: own q90 for `iso31`, or neighbours only.


**Status 2026-10-08.** With hourly targets (§21), all five rules are under review. RESEARCH_NOTES
§7.4e proposes changes to rules 1-3 and to the ray, hot-pixel and ring repairs, and adds
persistence signals. Rule 5 (reviewed radar-days) is unchanged.
**Status 2026-10-08, later:** rule 5 no longer uses a reviewed list. Radar-day and radar-year
exclusions come from a pre-stated score (§24).
---

## 20. When the dataset is ready for training: a gate fixed before the results

**Status 2026-10-08:** criterion 2 (100 tiles looked at by eye) is replaced by gauge-based
tail false-alarm and miss rates (§24).
**Status 2026-10-19:** criterion 3 is replaced (§32) by a contamination test of the most
influential test days, plus day-block intervals and a leave-period-out variant reported with
every tail statistic. The original rule failed by construction on heavy-tailed daily
contributions.

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


---

## 21. The target is the 1-hour accumulation, not the 15-min rate (2026-10-08)

**Question.** Which accumulation period should the super-resolution target be: the 15-min
instantaneous rate used so far, the 1-h accumulation, or the daily accumulation (proposed by
Daniele Nerini, MCH, 2026-10-07)?

**Decision (researcher, 2026-10-08, after discussing with Daniele Nerini).** The target is the
**1-hour accumulation** at 2 km. Daily is not adopted as the main target; it can be built
later from the same cleaned hours. The screen still runs on the 15-min frames, because most
error signatures (the value ladder, temporal support, persistence) exist only there
(RESEARCH_NOTES §7.4e).

**Why hourly.**
- *Small-scale structure decorrelates within 30 min.* The quantity the model has to add is
  the fine residual, x minus the upsampled 25-km mean. At a fixed pixel, the median correlation
  of that residual between frames t and t + lag is 0.29 at 15 min, 0.09 at 30 min and 0.035 at
  60 min (16 days, 5,429 tile pairs at lag 1; method below). A 15-min snapshot therefore holds
  fine structure that is almost independent of the frame 30 min later. A 1-h sum averages two
  to four decorrelation times of it, which raises the signal-to-noise ratio of the target
  (Daniele's argument) while keeping cell-scale geometry. A daily sum averages about 50 and
  turns convective cells into tracks, which changes what the geometric hypothesis H1 is about.
- *Agreement with gauges rises with aggregation:* log-Pearson against DE/CH gauges, v4
  filtering, 0.70 at 15 min, 0.77 at 1 h, 0.89 at 24 h (EXPERIMENTS §5, 63 days).
- *The deployable input is hourly.* ERA5 provides hourly accumulations; no reanalysis or
  climate model provides 15-min snapshots.
- *Sub-daily extremes survive.* About 10 of the 28 catalogue events (hail, supercells,
  derechos) are extreme at sub-daily scale but not as daily totals.
- *Hourly sits between the two.* On the same 16 days: 75% of the 2-km variance is unresolved
  by the 25-km mean (58% for daily), and the fine-to-coarse peak ratio is 14 (6 for daily).

**How the decorrelation was computed** (`scripts/data_quality/agg_compare.py`, summary with
`agg_summarize.py corr`; outputs in `quality_v4/calib/daily_check/out/`).
1. Clean whole days with the v4 chain (`DayCleaner`, ceilings and guard). The days are
   8 event days plus 8 random ODYSSEY days, listed in EXPERIMENTS §5.
2. Cut the grid into stride-128 tiles and keep the tiles that are at least 95% valid.
3. Transform to y = log1p(rate). The fine residual is y minus the bilinear upsampling of y
   block-averaged to 10x10, i.e. about 25 km, close to the ERA5 grid.
4. Take a base frame every 2 h from 04:00 and pair it with the frames 15, 30, 60, 120 and
   240 min later. A pair counts when both tiles are at least 5% wet (>= 0.1 mm/h).
5. For each pair, compute the Pearson correlation across the tile's 16,384 pixels, for the
   field and for the residual. Report the median over tiles.

| lag | field | residual |
|---|---|---|
| 15 min | 0.63 | 0.29 |
| 30 min | 0.44 | 0.09 |
| 60 min | 0.27 | 0.035 |
| 120 min | 0.15 | 0.018 |
| 240 min | 0.06 | 0.012 |

*Caveats.*
- The correlation is **Eulerian**: it is measured at fixed pixels. A cell moving at 10 m/s
  travels 9 km (4.5 px) in 15 min, more than a convective core is wide. Most of the drop is
  therefore advection out of the pixel, not the cell's lifetime. This is the right view for an
  accumulation, which is also Eulerian. A Lagrangian version (correlation after shifting by
  the motion field) would separate advection from evolution, and is a to-do.
- 8 of the 16 days carry the persistent artefacts of EXPERIMENTS §5. Those artefacts
  *raise* the correlation, so the true decorrelation is if anything faster.
- These are 16 days, so treat the numbers as indicative.

**The hour, as OPERA defines it.** OPERA's `ACRR` (1-h accumulation, published every 15 min)
does no processing beyond the plain sum. Checked pixel by pixel on one hour per product
generation, 2018-06-15 15:00 (ODYSSEY) and 2025-06-15 15:00 (NIMBUS):
- *Values.* ACRR at H = 0.25 x (RATE at H-45 + H-30 + H-15 + H) on every pixel with four
  valid frames, to within +-0.005 mm (rounding to 0.01 mm).
- *No processing.* There is no spatial filter, no cap (the maxima are identical), no advection
  correction (a window shifted by one frame or trapezoidal weights do not match) and no gap
  filling: one no-data frame at a pixel makes ACRR no-data there (the 4/4 rule, as in
  EURADCLIM).
- *Undetect.* ODYSSEY writes 0 where all four frames are undetect; NIMBUS keeps the undetect
  code. Anyone reading ACRR directly needs a per-generation mapping.
- *Time labels.* ODYSSEY RATE frames are labelled t-10..t+5 and ACRR H-55..H+5. NIMBUS RATE
  frames are labelled instantaneous and ACRR H-60..H. The same four frames are summed either
  way.
- *Quality index.* On ODYSSEY, ACRR's QIND matches the mean of the four RATE QINDs to within
  0.05.

Consequences:
- **Hour H is frames H-45..H** for both generations. ACRR is the raw-hourly baseline of
  §22 and an exact regression test of our raw sum. It is not an independent reference.
- Every quality step happens upstream of RATE: national QC and OPERA compositing.
- The 2026-10-07 gauge comparison used frames H:00..H:45 against gauge hour H..H+1, an
  offset of 15 min from this convention. Its alignment was estimated from the lag
  correlation. Re-check it with the ACRR convention.

**What this changes.**
- *The store* holds tile-hours (24 per day instead of 96).
- *The low-resolution input* is the coarsened hourly field.
- *The tail threshold.* u = 31 mm/h was chosen for 15-min rates (DECISIONS §1, §10). Hourly
  sums have lower peaks, so u, the FSS levels, the `extremes` subsets and the §20 gate
  thresholds are re-derived on hourly data (`pot_threshold.py`). The fixed mm/h threshold
  grid of the loss (§1) keeps its units, because a 1-h accumulation in mm is a mean rate in
  mm/h; which levels the tail populates changes.
- *Results are not comparable.* All results before v4 are on 15-min rates and cannot be
  compared with hourly results, like every other pre-v4 difference (data, splits, DEM).
- *Persistent moderate echoes now matter.* In an hourly sum an error contributes amplitude x
  frames / 4. A single-frame spike at 100 mm/h adds 25 mm, but a stationary 30 mm/h echo adds
  30 mm in every hour it persists. Rules keyed on single-frame intensity are no longer
  enough (RESEARCH_NOTES §7.4e).
- *How the hour is formed is a modelling choice, not a formality.* A plain sum of four
  snapshots of a cell moving 4-9 px per frame gives a beaded track. That beading is an
  artefact of the sampling, and a Minkowski loss would learn it. Advection-corrected
  accumulation is the standard remedy and is tested before the store is built
  (EXPERIMENTS §5).

**Status 2026-10-08, later: plain vs advection-corrected is settled by evidence**
(EXPERIMENTS §5, "Signal or noise?").
- *The evidence.* The plain sum of four 15-min snapshots beads moving rain. Over Germany it
  inflates the Euler characteristic at 2 mm by +2 to +5 relative to RADKLIM, growing with
  motion speed, and adds 1.2-2x power at 6-16 km. Advection correction removes both.
- *Proposed (to confirm):* the hourly target is advection-corrected. Interpolate between
  consecutive frames along a dense motion field (pysteps-like; needs pysteps or an
  equivalent), not the crude per-tile shift of the test.
- *Constraints.* Validate the correction against RADKLIM (coherence, chi) and the gauges
  before the store is built. Keep the plain sum as a stored variant for comparison, and
  OPERA's ACRR as the regression test of that variant only.

**What would overturn it.** MCH wanting the model for daily products only, or hourly targets
proving too noisy to train on, judged on the backbone's validation tail metrics and not on
`val_mse`.

Confidence:
- the choice is a researcher decision;
- the decorrelation numbers rest on 16 days;
- the ACRR identity rests on one hour per product generation; a whole frame missing from
  the archive has not been seen yet.

---

## 22. How we judge the screen: agreement with independent data must rise (2026-10-08)

**Question.** Without labels (§17), how do we show that a filtering rule makes the hourly
product better rather than just different?

**Decision (researcher, 2026-10-08; metric confirmed with Daniele Nerini).** The criterion is
the **increase** in agreement with an independent reference, raw -> filtered. The absolute
value is not the criterion: radar against point gauges is not expected to correlate highly at
1 h and 2 km.

**Protocol.** These points matter because a naive comparison inflates the gain.
- *A fixed set of pairs.* Use the pairs that are wet (>= 0.1 mm) in the raw radar, the
  filtered radar or the reference. If pairs are selected on the filtered field, every repaired
  false echo drops out of the sample and the correlation rises by selection.
  - The 2026-10-07 numbers recomputed this way: 15 min +0.070 [0.041, 0.099], 1 h +0.108
    [0.064, 0.145], 24 h +0.022 [0.017, 0.030] (95% day-block bootstrap, 300 resamples).
  - Rejection removed < 0.1% of the pairs there.
  - **Correction (2026-10-08, later the same day):** almost all of that gain is one gauge.
    DWD 01346 sits on a static-clutter pixel in 2023, with raw hourly values up to
    2.2e5 mm. Without it:
    - 15 min: 0.699 -> 0.698;
    - 1 h: 0.769 -> 0.772;
    - 24 h: 0.880 -> 0.887.
    The screen barely moves the pooled correlation at DE/CH gauges, because its targets are
    rare there.
- *Influence.* A pooled log-correlation over ~400k pairs is insensitive to rare
  artefacts, yet one absurd value dominates it. So every summary reports the gain with the
  most influential station left out (`gauge_hourly.py summary`).
- *Affected pairs, the sensitive test.* Where a rule changed the radar value, did the change
  move it towards the gauge? Report the share closer, the share further, and the median
  change in |log error|.
  - First hourly result (60 days, hot-pixel gauges excluded): the persistence rule changed
    654 gauge-hours at 15 stations. 98.6% moved closer, with a median log-error change of
    -0.66, and the gauge was dry in 82% of them.
  - Spikes, rings and unsupported cells: 5-11 gauge-hours each, 91-100% closer.
  - Static clutter: 34 gauge-hours at 2 stations, 50/50.
  - Rays, footprints and ceilings never touch a gauge.
- *Rejection is scored separately from repair.* A rejected tile-hour is missing in both
  versions. Report the agreement *of the rejected pairs* under raw data; it should be poor,
  which is what justifies rejecting them. Also report the share of reference-heavy hours that
  were rejected; that is the cost.
- *One rule at a time.* For each rule, report its marginal gain: the full chain minus the
  chain without that rule. A rule with a negative marginal gain removes real rain.
- *Stratify by intensity.* Report the gain separately for:
  - the false-alarm side: radar heavy, reference light;
  - the miss side: reference heavy. Here the filtered radar must not drop below the raw
    radar.
  The correlation over all pairs is dominated by light rain and can rise while a handful of
  real extremes are removed.
- *Uncertainty* comes from a day-block bootstrap, since pairs within a day are dependent.

**References, by role** (details in RESEARCH_NOTES §7.4e):
- *DE/CH gauges* (in hand): the main test.
- *RADKLIM RW and COMEPHORE* (hourly, 1 km, gauge-adjusted, open): the false-negative test.
  A real extreme removed by the filter should still be present there.
- *MTG lightning* (from July 2024, for NIMBUS): an independent check of convective
  extremes.
- *IMERG*: a rain / no-rain check over sea and areas without national data.
- *EURADCLIM*: agreement with another cleaning of the same OPERA input. Report it, but it is
  not truth.

None of the national radar products is independent of the radars themselves. They share the
sensor and differ only in processing.

Confidence: the criterion is a researcher decision with MCH's agreement. The selection and
stratification points follow from the definitions (derivable).

---

## 23. The hourly screen: rules, order and what is still provisional (2026-10-08)

**Question.** Which rules produce the hourly target (§21), in what order, and which of them
may lower or reject real rain?

**Decision (researcher, 2026-10-08, on Claude's proposal; review in RESEARCH_NOTES §7.4e).**
- *The hour.* Frames H-45..H, as OPERA's ACRR. A pixel-hour exists only if all four frames are
  valid. There is no gap filling yet.
- *A processing day* is the 24 hours ending d 01:00 .. d+1 00:00.
- *Decide over the day, repair per frame, then sum* (`src/data/hourly.py`).
- *Validation for now:* the DE/CH gauges only. RADKLIM, COMEPHORE, lightning, IMERG and
  EURADCLIM are banked for later (EXPERIMENTS §5).

**The chain** (every step can be switched off for the per-rule ablation of §22):

| step | rule | change from v4 |
|---|---|---|
| frame | drizzle < 0.1 mm/h -> 0 | none |
| frame | static clutter (hot mask) | `hot_fallback: keep`. A hot pixel with no clean 5x5 neighbour is left and coded 64, not set to 0. This breaks exact v3 reproduction (accepted) |
| frame | spikes | none |
| frame | ceiling pixels repaired (rule 2) | none. Tile-frames with >= 5 are still rejected at split time |
| frame | footprints of > 500 mm/h cores | none |
| frame | range rings >= 89 mm/h | none |
| frames | **rays** | replaces the per-frame rule. A thin (<= 3 px), >= 80 km line whose axis passes within 5 px + 3% of the distance of a radar site, present at the same pixels (+-1 px) in >= 3 of the +-4 frames. A front is wide and moves, so it fails both tests |
| day | **local-peak persistence** (new) | a pixel that is >= 1 mm/h and >= 2x its 5x5 median in >= `p1_min` of the day's 96 frames (`p1_ring_min` on range rings), or that lies on a site-anchored thin line of the n_peak >= `p2_min` map (rays inside rain). Takes its 5x5 median in each frame where it is a local peak. Code 128 |
| frame | cells without temporal support | none |
| split | rule 1, repeated value | not rejected when the ring around the repeated pixels is >= 0.5x the value (`rep_context`): a saturated core inside rain |
| split | rule 3, radar-disk | a flag stands only if the radar is also flagged in a neighbouring frame, or its wet fraction exceeds both neighbours' by >= 0.15 |
| split | rule 5, radar-day exclusions, guard rejections | none |
| split | tile-hour rejected | if any of its four tile-frames would be |

**Why these.**
- Persistence is the signature that matters for sums (§21).
- The rules that misfired on real storms in the gallery (ray, hot fallback, radar-disk,
  repeated value) now carry the evidence that separated their true from their false
  positives.
- Repairs only lower values, and nothing is set to 0.

**The radar-disk test, checked on the 20 gallery cases.**
- *Flag run length does not separate.* The four storm flags each last one frame, but so do
  12 of the 15 Valjevo failures.
- *A jump against both neighbour frames does.* Storms are within 0.13 of their neighbours'
  wet fraction (itcro 2022-08-18: 0.13). Failures jump 0.22-0.66, except one that is
  flagged in a run.
- *Known gap.* A storm flagged in two consecutive frames would keep its flag.

**Provisional, to be set by the hourly Phase 0** (199 days, galleries, the §22 gauge
ablation):
- `p1_min` 24, `p1_ring_min` 12, `p2_min` 12, `ray_min_frames` 3, `min_wet_jump` 0.15,
  `max_context` 0.5.
- First evidence that `p1_min` = 24 is too high: on 2021-07-14 a stationary point echo
  next to Paris-CDG (49.03 N 2.58 E) was a local peak in 20 frames. With a 486 mm/h frame
  it made the day's largest tile-hour, 147 mm.
- Its share of wet frames (n_peak / n_wet) may separate better than the count does; that is
  read from the stored signal maps.

**Not done, deliberately:**
- advection-corrected accumulation (needs pysteps; §21 says it changes the stored values, so
  it is decided before Phase 4);
- gap filling;
- extended clutter blobs. The persistence repair only lowers sub-peaks inside a wide
  persistent blob: the offshore wind-farm echo at 51.65 N 2.81 E went only from 118 to
  108 mm/day.

**Cost.** 118-153 s per processing day per core on the full grid, against ~60 s for the v4
scan. That is ~27-36 h for the archive with 6 workers.

Confidence:
- the structure (persistence over the day, repair per frame) follows from §21's error
  arithmetic;
- every threshold is provisional;
- the radar-disk criterion rests on 20 examples.

---

## 24. No dataset decision rests on the researcher's eye (2026-10-08)

**Question.** The v4 screen used the researcher's visual judgement in three places:
- radar-day exclusions from a reviewed list (§19, rule 5);
- the per-rule gallery verdicts, which set a threshold (the radar-disk `min_wet_jump`, §23);
- the readiness gate, which looks at 100 tail tiles by eye (§20, criterion 2).

Is that defensible in a publication?

**Decision (researcher, 2026-10-08).** No. A dataset whose content depends on one person's
labels cannot be reproduced or defended. From now on:
- *What is kept is decided only by rules* keyed on an error signature (§17 safeguard 1),
  with thresholds fixed by a stated criterion before the results are seen.
- *Images are diagnostics and illustrations, never decisions.* They help design a rule and
  show it in the paper. Whether the rule stays is decided by independent data
  (the DE/CH gauges, §22) or by a physical or statistical argument.
- *The gallery verdicts of 2026-10-08* (144 examples, `index.csv`) and the radar-day review
  list (`review.csv`) decide nothing. They are kept only as a record.

**What changes** (proposals, to be fixed before Phase 3):
- *Rule 5, radar-days:* exclude automatically above a pre-stated score (e.g. score >= 1:
  one decade above both the radar's own seasonal q99 and its neighbours), instead of a
  reviewed list.
  - A radar-year is excluded when a stated share of its rain days is excluded. That is how
    Valjevo 2023 would go, rather than by eye.
- *Radar-disk `min_wet_jump`:* set from the distribution of frame-to-frame wet-fraction
  changes in normal operation (e.g. its 99.9th percentile), not from the 20 labelled cases.
- *Persistence thresholds (`p1_min`, `p1_ring_min`, `p2_min`) and the other provisional
  values of §23:* set by the §22 gauge tests. On the affected gauge-hours, the rule must
  move the radar towards the gauge, with a stated cap on the share moved away.
- *Static clutter:* it fails that test. On 183 affected gauge-hours (3 stations), 46% moved
  closer and 48% further. Proposal: no unconditional repair; the hot mask only lowers the
  persistence threshold, as range rings do.
- *§20 gate, criterion 2:* replace the 100 tiles by eye with gauge measures over DE/CH:
  - the false-alarm rate in the tail (radar above the hourly u, gauge below a stated
    fraction of it);
  - the miss side.
  Criterion 3 (tail stability) is already objective.

**The assumption this rests on.** The only independent data are DE/CH gauges, so thresholds
tested there are assumed to transfer to the rest of Europe. The failures that matter most
(Valjevo, Madrid, Oradea) are outside DE/CH, so those rules must be justified by their
signature (a ceiling value, a range-only field) rather than by a gauge test. That limitation
goes in the paper.

**Where looking stays legitimate.** Rules were designed by looking at images, as in every
QC paper. That is hypothesis generation, and it is defensible as long as acceptance is
tested on data the design did not use.

Supersedes:
- the "reviewed by eye" parts of §17 (safeguard 3: images are now diagnostic only);
- §19 rule 5 ("a list the researcher reviews by eye");
- §20 criterion 2.

**Addendum, 2026-10-08, ~18:50, written before any result: the RADKLIM removal test**
(`scripts/data_quality/radklim_test.py`; data from `scripts/data/fetch_radklim.py`).

*Reference.* DWD RADKLIM RW v2017.002: hourly, 1 km, gauge-adjusted, with its own artefact
correction.
- It is mapped onto our grid by DWD's pixel-centre coordinates.
- Our hour H is paired with the RADKLIM file labelled H-10 min: on 2021-07-14, correlation
  0.83 against 0.67 and 0.47 for the neighbouring hours.
- RADKLIM uses the same German radars, so an artefact present in both counts against the
  screen. The test is conservative.

*The test.* Pixel-hours over Germany with a raw value (after the > 500 mm/h removal)
>= 10 mm, in bins 10-20 / 20-50 / 50-100 / >= 100 mm.
- *Removed:* filtered <= 0.5 x raw, or the tile-hour is rejected.
- *Kept:* filtered >= 0.95 x raw.
- *Confirmed by RADKLIM:* RADKLIM >= 0.5 x raw.

*Criterion* (per rule and bin, n >= 30):
- *artefact:* the removals are confirmed at less than half the rate of the kept values;
- *removes real rain:* confirmed at >= 0.8 of that rate;
- *mixed:* in between.

*Known limit.* Germany is where OPERA is cleanest. The test therefore covers the rules as
they act in Germany, and the share of the removed tail that lies inside the RADKLIM domain
must be reported with it.
- *Correction to that limit (2026-10-08, from the 100 random days):* Germany is not where
  OPERA is cleanest. 31% of the tile-hours whose raw max >= 50 mm the screen halves lie in
  the RADKLIM domain, the largest cluster at 50-55 N, 5-10 E.

**Result (2026-10-08, 194 days, `calib/radklim_test/summary.txt`).**
- *Every rule passes where the test has power.* RADKLIM confirms 0-5% of the removed
  pixel-hours, against 27-56% of the kept ones at the same raw intensity. That holds for
  static clutter, spikes, persistence, footprints, rings and unsupported cells, in every
  bin from 10 to 100 mm with n >= 30, and on random and event days alike. No tile-hour over
  Germany was rejected.
- *The ">= 100 mm" verdict "REAL RAIN" is a flaw in the pre-stated criterion, not a
  finding.* RADKLIM confirms none of the kept values there either (0/40), so "removed
  confirmed >= 0.8 x kept" reads 0 >= 0. The criterion did not anticipate a zero control
  rate. In that bin the test has no power: RADKLIM itself has 7 pixel-hours >= 100 mm on
  the same pixels, against 1,582 in the raw field. Recorded as undetermined; the criterion
  is not rewritten after the fact.
- *The screened tail moves towards RADKLIM's.* Pixel-hours >= u on the same pixels, raw ->
  screened, against RADKLIM:

  | days | u | raw | screened | RADKLIM |
  |---|---|---|---|---|
  | all | 30 mm | 11,692 | 6,266 | 4,780 |
  | all | 50 mm | 5,803 | 735 | 353 |
  | all | 100 mm | 1,582 | 43 | 7 |
  | random | 50 mm | 2,042 | 140 | 113 |
  | event | 50 mm | 3,630 | 479 | 168 |
  | event | 75 mm | 2,859 | 130 | 6 |

  Log-correlation with RADKLIM rises 0.382 -> 0.399.
- *Open:* on event days the screened OPERA keeps a heavier tail above ~50 mm than RADKLIM.
  It is either residual artefacts, or real peaks that RADKLIM's gauge adjustment and 1 km
  -> 2 km averaging flatten (it confirms only 27% of kept 50-100 mm values). To be resolved
  with independent data (gauges at those pixels), not by eye.
- *Below 10 mm RADKLIM is wetter than OPERA* (>= 5 mm: 1.94 M vs 1.21 M pixel-hours): OPERA
  underestimates the body, as at the gauges (median radar/gauge 0.46). The screen does not
  change this.

---

## 25. Criteria fixed before the results: advection correction and the Mahalanobis mask (2026-10-09)

Written before any result of the tests below (researcher approved the plan 2026-10-09).

**Advection-corrected hourly sums (§21 status note).**

*What the test must distinguish.* On a moving cell the plain sum of four snapshots is biased
high at the four snapshot positions and low between them. A correct correction therefore
lowers the plain-sum maximum of moving cells, and that is not softening. On stationary cells
it must change nothing. Softening is judged on stationary cells and against gauges, never by
"corrected max < plain max".

*Variants:*
- the pysteps-style linear blend of the earlier frame moved forward and the later frame
  moved backward;
- no blending (first half of each interval from the earlier frame, second half from the
  later);
- bilinear or cubic / nearest warping.

*A variant is accepted only if all four hold:*
1. *Beading removed:* median chi difference to RADKLIM within +-1 at 1, 2 and 5 mm
   (RADKLIM-aligned hour).
2. *No softening of stationary cells:* tile maxima of tiles moving < 2 px per 15 min within
   +-5% of the plain sum (median ratio).
3. *Heavy hours not pushed down at independent gauges:* on gauge-hours >= 20 mm (DE + CH,
   reported for CH alone too), the median radar/gauge ratio is not lower than the plain
   sum's (day-block bootstrap).
4. *Mass conserved:* the median tile-sum ratio corrected/plain within +-1%.

*Ties:* the variant with the smallest chi error is chosen. If none passes, the target stays
the plain sum and beading is reported as a limitation.

**Per-pixel Mahalanobis outlier mask** (after Franch et al. 2020, TAASRAD19).
- *Features:* each pixel's histogram of log-rate (fixed bins), from a sample of frames.
- *Reference:* the pixel's neighbourhood (~100-200 km), not the whole domain, so that
  climatological gradients such as orographic maxima are not flagged. Computed separately per
  product era (ODYSSEY to 2024-06, NIMBUS from 2024-07).
- *Flag:* distance > mean + 3 sigma of the neighbourhood's distances, as in the paper.
- *Accepted (replacing or supplementing the hot-pixel climatology) only if:*
  - in the RADKLIM removal test (§24 addendum), the pixel-hours it changes are confirmed at
    less than half the rate of kept values in every bin with n >= 30;
  - its affected gauge-hours move towards the gauge in >= 80% of cases (§22);
  - its flag rate does not rise steeply with intensity (§17 safeguard 2).
- Known orographic maxima (Alps, Ticino, Black Forest) are reported explicitly.

**Amendment to the mask design, 2026-10-09, before any mask result** (literature review in
RESEARCH_NOTES §7.4g). The acceptance tests above are unchanged; the design changes, for four
reasons:
- *mean + 3 sigma fixes the flagged fraction whatever the data:* Franch et al. flag 0.9% and
  Shi et al. 1.6%, ~10x the Gaussian rate, and neither validated the mask;
- *a 526-bin histogram has an exactly singular covariance* (frequencies sum to 1), so its
  pseudoinverse Mahalanobis distance is dominated by noise in rare bins;
- *every operational chain uses exceedance frequencies at a few fixed thresholds*
  (Wagner et al. 2012, Tabary et al. 2012, DWD RADKLIM, Odyssey);
- *a 100-200 km reference flags orographic and coastal maxima.* EURADCLIM's global reference
  did so along the Norwegian coast, and RADKLIM's statistical outliers included real
  heavy-rain years (Kreklow et al. 2020). Clutter is 1-3 px with sharp edges, orography varies
  over 10-50 km, so a ~10-20 km annulus separates them by scale.

New design (still a Mahalanobis distance, as in Franch et al., with the literature's fixes):
- *Features:* per pixel and year, logit of the exceedance frequency at 1, 31, 89 and 150 mm/h
  (`clutter_climatology.npz`); pixels with < 1000 valid frames are skipped.
- *Reference:* the residual of each feature against the median over an annulus of radius
  3-8 px (6-16 km) around the pixel, which excludes the pixel's own blob.
- *Distance:* Mahalanobis distance of the residual vector with a robust (minimum covariance
  determinant) covariance; flag where d^2 exceeds the chi^2(4) quantile 1 - alpha.
- *Choosing alpha:* from {1e-3, 1e-4, 1e-5, 1e-6}, chosen on half of the validation days and
  evaluated on the other half, so the threshold is not tuned on the data that judges it.
- *Masks per year;* the number of years a pixel is flagged within an era is reported (wind
  farms appear mid-era, storm tracks do not persist).
- *Application:* replaces the hot-pixel climatology in `clean_frame` (repair only if lower),
  then the same acceptance tests.
- *Selection rule for alpha* (fixed before results): split the RADKLIM-test days in two halves
  by day index (even / odd). On the even half, keep the alphas that pass every acceptance test.
  Among them, choose the one that removes the most pixel-hours >= 10 mm. Report that alpha's
  tests on the odd half. If none passes on the even half, no mask is adopted.
- *Second amendment, 2026-10-09, after the 2018 distance distribution and before any
  acceptance result.*
  - *The chi^2(4) null does not hold.* d^2 has median 0.69 (chi^2: 3.36) and 99.9th
    percentile 835 (chi^2: 18.5), so every alpha in {1e-3 .. 1e-6} flags ~2% of pixels.
  - *The threshold is therefore set directly on d^2,* grid {25, 100, 400, 1600}: in 2018 that
    flags 2.8%, 1.0%, 0.28% and 0.015% of valid pixels. The selection rule is unchanged.
  - *Implementation changes* found in the smoke test:
    - residuals are standardised by spatial spread and sampling noise
      (se = 1/sqrt((n + 0.5)(1 - p))), because sparse counts made the MAD zero;
    - the correlation is estimated from Spearman ranks, because MCD was made singular by
      tied sparse-count pixels.
  - *What d^2 picks up (diagnostic, decides nothing):*
    - radial spokes around many radars, and clutter near the sites: at d^2 > 400 the flag
      rate is 1.6% within 10 km of a site vs 0.13% at 200-300 km;
    - a rate rising with elevation, 0.20% below the median elevation vs 0.73% in the top 1%:
      either mountain clutter or real orographic maxima, for the acceptance tests to
      separate.
  - *The mask contains the current hot-pixel mask:* 1,972-1,977 of its 1,982 pixels in 2018.

**Result, advection correction (2026-10-09; `calib/advection_variants/summary.txt`,
`calib/advection_gauges/summary.txt`).**
- *By the pre-stated rule, `noblend_nearest` is accepted,* the only variant passing all four
  criteria:
  - chi minus RADKLIM: 0 / 0 / -1 at 1 / 2 / 5 mm;
  - slow-tile max: 0.96 of plain;
  - mass: 1.000;
  - gauge-hours >= 20 mm: -6% in median ratio, interval [-10%, +0.4%], a narrow pass.
  The pysteps-style linear blend fails three criteria; cubic blend and no-blend linear fail
  two.
- *Not in the criteria, but decisive for extremes: peaks of moving rain are lowered.* Median
  tile max, plain / corrected / RADKLIM: slow 9.5 / 8.6 / 9.8, mid 9.2 / 6.4 / 8.3, fast
  9.3 / 5.4 / 7.2 mm. The plain sum matches RADKLIM on slow rain, overshoots it by 10-30% on
  moving rain (beads) and the correction undershoots it by 20-25%. In the median tile-hour
  where RADKLIM has pixels >= 20 mm, every corrected variant has none (plain: half
  RADKLIM's area).
- *Undetermined:* how much of the undershoot is motion-field error and how much is RADKLIM's
  gauge adjustment.
- *Proposed (Claude), for the researcher to decide:* store both hourly fields, plain and
  `noblend_nearest`, and choose by training on each against RADKLIM and the gauges.

**Decision on the hourly target (researcher, 2026-10-09).**
- *The target is the plain sum* of the four frames (ACRR convention).
- *The advection-corrected field (`noblend_nearest`) is stored alongside.* The choice
  between them is revisited when training.
- *Why.* Beading is a sampling property of 15-min data, not unphysical data. Under §17,
  the least-processed field is the default.
- *RADKLIM cannot arbitrate magnitudes:*
  - peaks smoothed and underestimated (reported in the literature);
  - gauge adjustment interpolated between gauges;
  - our own 1 -> 2 km averaging;
  - it confirms only 27% of kept 50-100 mm values and has 7 pixel-hours >= 100 mm where
    OPERA has 43.
  Point gauges cannot either: a 2 km pixel mean of a convective core is below the point
  maximum by construction.

**RADKLIM as a presence test (2026-10-09; `scripts/data_quality/radklim_presence.py`,
`calib/radklim_presence/summary.txt`).** RADKLIM is trusted only for "is there substantial rain
within +-2 px and +-1 h" (dry = < 5 mm).
- *Removals are artefacts on this magnitude-free criterion:* 85-99% of the removed
  pixel-hours are dry, in every bin from 10 to >= 100 mm. Per rule: rejection,
  footprints, rings and unsupported cells 90-100%; spikes 93-94%; persistence 75-89% (it
  lowers peaks inside rain).
- *Remaining contamination in the kept, screened values* (Germany, 192 days):

  | screened >= u | kept | RADKLIM-dry |
  |---|---|---|
  | 20 mm | 29,180 | 5.7% |
  | 30 mm | 6,272 | 9.5% |
  | 50 mm | 736 | 31% |
  | 75 mm | 157 | 68% |
  | 100 mm | 43 | 86% |

  The disagreement floor of ordinary kept 10-50 mm values is 5-6%. So contamination is at
  the floor in the bulk of heavy rain and dominates the top. It is small in absolute terms
  (228 pixel-hours >= 50 mm on 39 of 192 days), and the same pattern holds on random days.
- *Signature of those 228 pixel-hours:*
  - one 15-min frame carries > 80% of the hour in 88% (median 100%);
  - that frame's rate has median 294 mm/h (q10-q90: 170-440);
  - isolated in 92% (5x5 median of the hour ~0);
  - not persistent (median 2 local-peak frames) and not near radars (median 59 km);
  - concentrated on big convective days (2014-06-08..10, 2013-07-28, 2014-07-27..29,
    2023-07-29).
- *Why they survive:* there, the temporal-support rule is satisfied by surrounding storms
  within 30 km, and the spike rule misses clusters of a few pixels.
- *Candidate rule, proposed and not yet implemented:* an isolated single-frame burst. One
  frame carries most of the hour at a high rate, with no rain in the immediate 4-6 km that
  hour. The isolation condition protects real cores embedded in rain.

---

## 26. Isolated single-frame bursts: a rule for the remaining contamination (2026-10-09)

Written before any result of the rule (researcher approved implementing it, 2026-10-09).

**The problem it targets.** After the hourly screen, over Germany, RADKLIM is dry (< 5 mm within
+-2 px and +-1 h) under 31% of the kept screened pixel-hours >= 50 mm, 68% >= 75 mm and 86%
>= 100 mm, against a 5-6% disagreement floor for ordinary 10-50 mm values (§25,
"RADKLIM as a presence test"). Those pixel-hours set the tail statistics.

Signature of the 228 such pixel-hours >= 50 mm:
- *single frame:* one 15-min frame carries > 80% of the hour in 88% (median 100%), at a
  median rate of 294 mm/h (q10-q90: 170-440);
- *isolated:* 92% have essentially no rain around them that hour (5x5 median of the hourly
  sum < 5 mm; median 0.04 mm);
- *not clutter:* not persistent (median 2 local-peak frames in the day) and not near radars
  (median 59 km);
- *on convective days:* concentrated on big ones (2014-06-08..10, 2013-07-28,
  2014-07-27..29, 2023-07-29).

They survive because the temporal-support rule accepts any echo >= 1 mm/h within 30 km in a
neighbouring frame, which the surrounding storms provide on convective days. The spike rule
misses them because they are clusters of a few pixels that protect each other.

**The rule** (`src/data/hourly.py`, `burst`; per pixel-hour, after the temporal-support repair
and before the hour is summed):
- *Single frame:* within the hour's four frames, one frame carries > 80% of the pixel's sum.
- *High rate:* that frame's value at the pixel is >= 100 mm/h.
- *Isolated:* the median of the hour's screened sum over the 5x5 window around the pixel
  (10 km) is < 1 mm.
- *Action:* in the dominant frame, the pixel takes the median of the valid values of its
  5x5 window in that frame, if lower. Coded REPAIR_BURST (256; the repair codes become
  uint16).
- *Protection of real cores:* a real convective or hail core has rain around it in that
  hour, so it fails the isolation condition. The threshold is set low (1 mm) to keep that
  protection wide.

**Acceptance** (RADKLIM presence test over Germany on the calibration days; same definitions as
§25):
1. *Its removals are RADKLIM-dry:* the burst-repaired pixel-hours (screened <= 0.5 x the
   value without the rule) are dry in >= 85% of cases in every raw bin with n >= 30 (the
   range of the existing rules).
2. *It reduces the contamination:* the RADKLIM-dry share of the kept screened values
   >= 50 mm falls.
3. *Real cores:* the number of removed pixel-hours with RADKLIM >= 0.5 x raw within tolerance
   is reported, with their days, together with the flag rate by raw intensity (§17
   safeguard 2).

If 1 fails, the rule is not adopted. The 1 mm / 100 mm/h / 80% thresholds are reported as they
are, not tuned on the result. The sensitivity to an isolation threshold of 5 mm is reported
separately.

**Result (2026-10-09; `scripts/data_quality/burst_validate.py`,
`calib/burst_validate/summary.txt`; 192 days with RADKLIM, 2.6 M pixel-hours with max(raw,
RADKLIM) >= 5 mm over Germany). The rule passes as pre-stated and is adopted.**

*Criterion 1, its removals are RADKLIM-dry.* Removals = the rule lowers a pixel-hour of the
screened field (C >= 1 mm) to <= 0.5 C:

| raw bin | pixel-hours | lowered | flag rate | RADKLIM-dry |
|---|---|---|---|---|
| 10-20 mm | 210,141 | 0 | 0 | - |
| 20-50 mm | 29,959 | 308 | 1.0% | 87.7% |
| 50-100 mm | 4,245 | 141 | 3.3% | 97.2% |
| >= 100 mm | 1,588 | 30 | 1.9% | 100% |

*Criterion 2, the contamination falls.* RADKLIM-dry share of the kept screened values >= u,
without -> with the rule:

| u | all days | event days | random days |
|---|---|---|---|
| 20 mm | 5.8% -> 4.4% (n 29,673 -> 29,194) | | |
| 30 mm | 9.6% -> 4.5% (6,383 -> 6,018) | 9.7% -> 3.5% | 11.0% -> 6.1% |
| 50 mm | 31.5% -> 11.7% (744 -> 573) | 34.0% -> 9.5% | 27.3% -> 12.7% |
| 75 mm | 68.8% -> 34.7% (160 -> 72) | 70.8% -> 29.4% | |
| 100 mm | 86% -> 60% (44 -> 15) | | |

At >= 30 mm the share reaches the 5-6% disagreement floor of ordinary values. At >= 50 mm it
is down to ~12%; at >= 75 mm a third remains (25 of 72 pixel-hours), a residual for a later
round.

*Criterion 3, real cores.*
- 13 of the 479 removals (2.7%) have RADKLIM >= 0.5 x raw within tolerance, spread over 8
  days (at most 4 on 2015-07-04).
- The flag rate stays at 1-3% of pixel-hours per raw bin. It does not climb with intensity
  the way a rule hitting real cores would.

*Sensitivity: isolation threshold 5 mm instead of 1 mm.* It FAILS criterion 1: removals at
20-50 mm are only 50% RADKLIM-dry (832 lowered), and 155 of 1,053 removals are confirmed.
The isolation condition is what protects real rain, and 1 mm is the right side of it.

*Correction made while summarising (recorded so the numbers can be audited).* The first summary
counted as "removed by the rule" every pixel-hour whose screened value C was already 0, because
the condition Cb <= 0.5 C is then trivially true. That inflated the removals to ~20,000 with
61% "confirmed". The removal definition now requires C >= 1 mm. The criterion-2 table compares
the kept values directly and was not affected.

**Adopted:** `hourly.rules.burst: true` in `configs/quality_v4.yaml`, with the pre-stated
thresholds (share > 0.8, frame rate >= 100 mm/h, isolation median < 1 mm).

**Result, per-pixel Mahalanobis mask (2026-10-09; `scripts/data_quality/pixel_outliers.py`,
`mask_validate.py`; `calib/pixel_outliers`, `calib/mask_validate/summary.txt`). Not adopted
pending the researcher's decision.**
- *Masks:* built for 2012-2026. Median per year: 32,282 / 10,301 / 1,559 / 65 pixels at
  d^2 > 25 / 100 / 400 / 1600 (1.6% / 0.5% / 0.07% / 0.003% of valid pixels). They contain
  most of the existing hot mask (520 of 528 pixels at d^2 > 25). The flag rate rises in
  the highest elevation decile (4.2% vs ~0.9% at d^2 > 25).
- *Pre-stated criterion* (RADKLIM confirms the magnitude, K >= 0.5 raw, for removals at less
  than half the rate of kept values): every threshold passes on both halves. The selection
  rule picks d^2 > 25, which passes on the odd half (10-20 / 20-50 / 50-100 mm: confirmed
  19 / 13 / 2% of removals vs 58 / 46 / 31% of kept).
- *Presence test* (RADKLIM-dry within tolerance, §25), share of removals that are dry, odd
  half:

  | threshold | 10-20 mm | 20-50 mm | 50-100 mm | removals >= 10 mm |
  |---|---|---|---|---|
  | d^2 > 25 | 60% | 64% | 87% | 2,091 |
  | d^2 > 100 | 76% | 81% | 87% | 857 |
  | d^2 > 400 | 91% | 93% | 100% (n 5) | 169 |

  The kept-value floor is 3-6% dry; the existing rules' removals are 85-99% dry. So at
  d^2 > 25 about 40% of the 10-50 mm removals have real rain under them. The mask repairs
  the pixel in every frame, as the hot mask does, and so shaves real rain wherever a
  flagged pixel sits inside a storm.
- *The two criteria disagree.* The magnitude criterion was pre-stated for the mask. The
  decision to trust RADKLIM only for presence was made earlier on 2026-10-09, before this
  rerun, but it was not written as the mask's criterion.
- *Proposed (Claude):* do not adopt d^2 > 25. Options:
  1. use the mask only as a prior that lowers the persistence threshold, so pixels are
     repaired only on days they behave like clutter (preferred; needs implementing and
     the same test);
  2. adopt d^2 > 400 (passes both criteria, small effect);
  3. keep the existing hot mask only.
- *Correction made while summarising (auditable).* The first run counted pixel-hours already
  zeroed by other rules as removed by the mask (65-81% "removed" at 50-100 mm). The removal
  definition now requires the no-mask value >= 1 mm. First run kept in
  `calib/mask_validate_run1_nopos`.

**Mask as a prior for the persistence rule (researcher chose option 1, 2026-10-09; written
before any result).**
- *Design:*
  - The per-pixel mask no longer repairs anything by itself.
  - On pixels flagged that year (d^2 > 25, the broadest mask), the persistence threshold
    `p1_min` (24 local-peak frames in the day) is lowered to `p1_prior_min` = 12, the value
    already used on climatological range rings.
  - The repair is the persistence repair: in each frame where the pixel is a local peak, it
    takes its 5x5 median. A flagged pixel inside a passing storm is a local peak in only a
    few frames, so it is spared.
  - The existing hot-pixel climatology stays as it is.
- *Acceptance* (presence test over Germany, as §26). Removals attributable to the prior =
  the chain with the prior lowers a pixel-hour of the chain without it (>= 1 mm) to <= 0.5x.
  1. They are RADKLIM-dry in >= 85% of cases in every raw bin with n >= 30.
  2. The RADKLIM-dry share of kept screened values >= 20 / 30 / 50 / 75 mm does not rise.
  3. The flag rate by raw intensity is reported, with removals RADKLIM confirms and their
     days.
- *Sensitivity, reported and not used for the choice:* `p1_prior_min` = 8.

**Result, mask as a persistence prior (2026-10-09; `scripts/data_quality/prior_validate.py`,
`calib/prior_validate/summary.txt`; 192 days). FAILS; not adopted.**
- *Removals attributable to the prior* (`p1_prior_min` 12), RADKLIM-dry share:

  | raw bin | removals | dry |
  |---|---|---|
  | 10-20 mm | 817 | 84.2% (bar 85%) |
  | 20-50 mm | 208 | 75.5% |
  | 50-100 mm | 18 | 83% |
  | >= 100 mm | 13 | 31% |

  RADKLIM confirms 540 of the 3,493 removals. They concentrate on 2013-05-30..06-02 (the
  Central European floods), 2023-07-19 and 2014-07-29: long-lasting real rain that is a
  local peak in >= 12 frames on flagged pixels, which are more frequent at high
  elevation. This is the stationary-rain risk named in RESEARCH_NOTES §7.4e.
- *Sensitivity* (`p1_prior_min` 8): worse (79% / 70% dry; 1,127 of 5,497 confirmed).
- *Little benefit at the top:* RADKLIM-dry share of kept values >= 50 mm 11.7 -> 9.9%,
  >= 75 mm 34.7 -> 29.7% (the burst rule: 31.5 -> 11.7%, 69 -> 35%).
- *Decision:* the per-pixel Mahalanobis mask is not adopted in either form (unconditional at
  d^2 > 25, or as a prior). The existing hot-pixel climatology stays.
- *What the result says about the method* (inference): at OPERA scale, with four exceedance
  features and a 6-16 km reference, the per-pixel climatology does not separate clutter
  from climatologically wet, orographic pixels where long rain persists. A future version
  needs a feature that does, e.g. Lagrangian persistence (stationary rain still advects at
  synoptic scale) or a satellite cloud-top check (EXPERIMENTS §5, "More ways to detect
  unphysical extremes").

---

## 27. Radar ownership from the radars that actually contribute to the composite (2026-10-09)

**Problem.** The radar screen attributes each pixel to the nearest "active" radar within 250 km
(`radar_screen.RadarGeometry`). It uses the OPERA radar database, which lists every member
radar whether or not it delivers data.
- 22-27 Italian sites a year owned pixels in the radar pass (the "itcro" flag on the 2022
  Mediterranean derecho), although no Italian radar contributes to OPERA. Our raw store has
  no data over most of Italy in 2012, 2016, 2018, 2021 and 2025.
- Other database entries are duplicates, retired sites or radars that do not deliver: old
  Swedish codes, De Bilt after 2017, airport radars, most Serbian sites.
- Radar-frame flags, the radar-day ranking, ceiling attribution and tile owners all inherit
  these errors. Example: the 2023 Valjevo-area 364.63 mm/h ceiling was attributed to sites
  that do not deliver; only Fruska Gora (rsfrg) and Jastrebac (rsjas) are listed for Serbia.

**Evidence.** Each composite lists its contributors in `how/nodes`. `scripts/data/opera_nodes.py`
read one composite per month, 2012-09 .. 2026-10: 167 of 170 months, 100-178 nodes per month
(`validation/opera_nodes/opera_nodes.csv`; 2012-11 .. 2013-01 missing from the archive).
- *No Italian node is listed in any month.*
- *ODIM codes differ between node lists and database for some radars:*
  - Spain throughout (esmad / estjv Torrejon de Velasco, esbar / esgld Gelida, esmal /
    esahr Alhaurin, esmur / esftn, esval / escll, eszar / espdg, essev / esclg, esalm /
    esnjr, escor / esccd, espma / esllm, eslpa / esatn, ...);
  - Feldberg (deflg / defbg), Suergavere (eesyr / eesur), hrzag and istgb.

**Decision (Claude, implementing the researcher-approved next step 2026-10-09).** In the hourly
build, a database radar owns pixels in year Y only if either:
- its ODIM code is listed as contributing in some month of Y; or
- its country prefix has listed codes that match no database code (Spain, and Germany /
  Estonia in some years), AND the composite has valid data at its site in >= half the
  year's frames (`clutter_climatology.npz`).

Effect (database-active -> contributing):
- 2013: 185 -> 109;
- 2018: 222 -> 148;
- 2025: 246 -> 172, of which 22-27 dropped are Italian-located.

All Spanish radars stay. Oradea stays under its listed code `roora`; the database also has a
second entry, `roopa`.

**Implementation.**
- `radar_screen.use_contributing(nodes_table, climatology)` restricts `active_sites`. It is off
  by default, so the v3/v4 builds reproduce exactly.
- It is switched on by `radar_geometry.contributing_only` in `configs/quality_v4.yaml`.
- Read by `scan_radars.py --quality`, `radar_screen_v4.py` and `scan_hours.py`.

**Consequence.** The radar pass (Phase 2) is rerun with this geometry before the full-archive
tile scan. The previous outputs are archived in `quality_v4/phase2_db_geometry/`; the
researcher's `review.csv` record stays there unchanged.

**Limitations.**
- Ownership remains geometric (nearest contributing radar), not the composite's actual
  per-pixel source.
- Where a country has unmatched codes, a few duplicate database sites with data at their
  location are kept (e.g. German airport or duplicate entries in 2013 and 2018).

Confidence: the node lists are authoritative for which radars contributed in the sampled
months. The union over months may miss a radar that was down at every sampled time (one
composite per month).

**Check after launch (2026-10-09).** On the first rerun day (2012-09-10), 93 radars are kept
out of 175 database-active ones.
- *All 14 Spanish radars are kept; no Italian radar is.*
- *Several radars over which the composite has valid data are dropped in 2012-13:* CH, HU,
  Jabbeke, Serbia, Oslo and Riga. Their countries enter the node lists later:

  | country | first year listed | last year listed |
  |---|---|---|
  | be (Wideumont) | 2013 | 2026 |
  | ch | 2014 | 2026 |
  | hu | 2014 | 2026 |
  | lv | 2015 | 2026 |
  | rs | 2017 | 2024 |
  | gr | 2021 | 2026 |
  | lt | 2025 | 2026 |

  Never listed: at, it, cy, tr, ba, me, mk, al, bg, md, ua.
- *Valid data is not evidence that a radar contributes.* The composite's valid mask also
  covers areas that neighbouring radars reach, including "undetect". Over Switzerland in
  2012 and over Austria in every year, the valid fraction is 1.0 although no national radar
  contributes. The node lists stay the criterion. Those areas are owned by the nearest
  contributing radar, which is the one actually measuring them.
- *Consequence for the data, a limitation to state:* Austria (never), Switzerland and
  Hungary (before 2014) are observed only by distant foreign radars, with beam overshoot over
  the Alps. Extremes there are likely underestimated. Kept, not screened.

## 28. Rule 5 by a pre-stated score: radar-day and radar-year exclusions (2026-10-13)

**Question.** §24 replaced the reviewed radar-day list with an automatic rule but left the
numbers open. Which radar-days and radar-years are excluded?

**Decision (researcher, 2026-10-13, on Claude's proposal).** Written down before the ranking
of the contributing-radar Phase 2 rerun (§27, `quality_v4/ranking.csv.gz`) was opened. Only its
log summary had been seen: 111 radar-days with score >= 1 under the old `iso31` reference.
1. *Radar-day:* excluded when `score >= 1`, i.e. some signal is a decade above both the
   radar's own seasonal q99 and the median of its 5 nearest radars that day (§19).
2. *Radar-year:* excluded, with all its days, when its excluded radar-days are >= 5% of its
   rain days and number >= 3. A rain day is a day on which, in some frame, >= 20% of the
   radar's owned pixels are >= 1 mm/h (`max_wet >= 0.2`, the wet threshold of the R2
   feature). The 3-day floor stops one bad day from removing a short year (2012 has 46 days).
3. *`iso31` is scored against its neighbours only,* not against the radar's own seasonal q99.
   A radar failing on more than ~1% of its season-days raises its own q99 and hides behind
   it (§19: Oradea's failure days scored < 0 on 6 of 7). The neighbour median has no such
   blind spot for a single failing radar. The other three signals keep both references.

**Why these values.** Rule 1 is the value §24 floated: one decade is far outside the
day-to-day spread of a working radar, so the rule keys on failure, not on rare weather
(§17 safeguard 1). The 5% and the 3-day floor in rule 2 are Claude's proposal, with no data
behind them when fixed: a radar failing on one rain day in twenty is unreliable on the days
the score does not catch. The 20% rain-day threshold reuses an existing convention.

**What would change the conclusion.** Real extremes that are a decade above all five
neighbours (a small, isolated convective cell). To check, report the excluded radar-days that
fall on catalogued event days (`configs/prominent_events.yaml`) and their rejection share by
intensity bin at split time (§17), without changing the thresholds after seeing them.

**Implementation.** `radar_screen_v4.py rank` (reference per signal from
`radar_day.reference` in the quality config) and `radar_screen_v4.py exclude`, which writes
`day_exclusions.csv` (radar, day, reason) and `year_exclusions.csv` for
`make_splits_hourly.py --day_excl`. Without the new config keys, the old behaviour is kept.

**Result of the pre-stated rule (2026-10-13), not yet adopted downstream.** `rank` + `exclude`
on the contributing-radar Phase 2 rerun (previous ranking archived in
`quality_v4/rank_iso31_both/`):
- *Radar-days with score >= 1: 3,029* (111 with the old `iso31` reference). Driven by `iso31`:
  2,948, then f150 65, ceilings 13, flagged frames 3. 130 of 203 radars have at least one.
- *Rule 3 flags chronic radar behaviour, not failure days.* On the new `iso31` exclusions the
  median day has 176 isolated pixel-frames >= 31 mm/h, against a neighbour median of 0 and the
  radar's own seasonal q99 of 277. Old `score_iso31` median -0.1. Spread evenly over the
  seasons (DJF 765, MAM 703, JJA 840, SON 721), so not mainly convection. Concentrated on
  coastal and island radars (isegs 267, sekkr 201, deboo 192, mtgud 181, skjav 147, sevar 138).
- *51 excluded radar-days fall on catalogued event days*, the check named above.
- *Rule 2's rain-day denominator is degenerate (a specification error, Claude's).* Few days have
  >= 20% of a radar's area at >= 1 mm/h in any frame, so many radar-years have 0-2 rain days
  and a median share of 54%: 170 radar-years (57,710 radar-days) excluded. Over days with data
  instead, the same 5% / 3-day rule gives 39 radar-years (13,852 days).
- *A median own-reference does not separate them either* (comparison only, nothing written):
  2,686 radar-days, same radars, same flat seasons. Most of these radars have iso31 = 0 on most
  days, so their median is ~0.
- *The Oradea failure days of §19 (1,286-8,636 isolated pixels) no longer appear:* the largest
  Oradea (roora) day has 502. Under the contributing geometry those pixels probably belong to
  another radar. To be checked; it is the case rule 3 was written for.

So the threshold does not decide this; whether the ~2,600 `iso31` days are artefacts is an
empirical question. Proposed test, in the form of §25/§26 (criterion fixed before running):
on the excluded German radar-days (deboo, depro, ...), the isolated >= 31 mm/h pixels must be
dry in RADKLIM at >= 85%. Pending the researcher's decision.

**RADKLIM test of rule 3 (criteria fixed 2026-10-13, before the test was run; researcher
asked for the test, Claude wrote the criteria).** `scripts/data_quality/iso31_radklim.py`.
- *Population:* the pixels counted by `iso31` on the German radar-days that §28 excludes with
  reason `iso31` (321 days with RADKLIM after fetching): a pixel owned by that radar
  (contributing geometry, §27), >= 31 mm/h in the `clean_frame` field, isolated by
  `radar_screen.isolated_mask`. Each pixel-frame belongs to the hour H whose ACRR window
  H-45..H holds it; frames at d 00:00 (previous processing day) are skipped.
- *Measure:* RADKLIM-dry = `K_tol < 5` mm (`radklim_presence.k_tolerant`: max over +-1 h and
  +-2 px), the definition of §25/§26. A pixel at >= 31 mm/h in one frame adds >= 7.75 mm to
  its hour, so a RADKLIM value below 5 mm in the whole window is a disagreement on presence.
- *Pass:* the pooled dry share is >= 85%, as in §26. Reported with a 95% day-block bootstrap
  interval, per radar, and per radar-day.
- *If it passes,* rule 3 stays as fixed. *If it fails,* rule 3 goes back to the two-reference
  score and the Oradea-type case is handled separately.
- *Descriptive, not criteria:*
  - the same measure on a control: isolated pixels of German radar-days that are *not*
    excluded (score < 1), on the RADKLIM days already on disk;
  - the share of the test pixels that the hourly chain (current `quality_v4.yaml`, burst rule
    on) leaves at >= 5 mm in the screened hour. A high share means the pixel-level chain
    does not already remove them, so a radar-day rule is needed for them.
- *Rule 2's denominator* stays as fixed (rain days) until the researcher confirms the
  correction to days with data; the gate (§29) reports rule 5 both ways.

## 29. The readiness gate before the full-archive build (2026-10-13)

**Question.** §20 defines the gate on the built v4 dataset, and §24 replaced its criterion 2
with gauge measures but set no numbers. The full-archive build needs rule 5 settled first
(§28). Which parts can be checked before the build, and against what?

**Decision (researcher asked for the gate 2026-10-13; Claude fixed the numbers below before
running).** A pre-build gate on the calibration days (199) and the gauge days (63), with the
current chain: `quality_v4.yaml`, burst rule on, contributing-radar geometry (§27), frame
flags of the 2026-10-09 radar pass. Calibration days are rescanned with that chain
(`calib/hourly_c/`). Every criterion is reported with rule 5 off and with the pre-stated §28
exclusion list.

*u is provisional.* The hourly tail threshold is not derived yet (§21). The gauge criteria
use u = 20 mm in 1 h as the primary level and report 10 and 30 mm.

1. *Known failures caught (§20 criterion 1, `known_failures`, hourly form).* Among kept
   tile-hours:
   - Madrid 20180429: none >= 10% owned by estjv with `max_rate` >= 31 mm/h;
   - Valjevo 2023 (calibration days): none with `max_rate` within 0.005 of 364.63;
   - 48.62 (2019-11, tile 640/1408): none with `rep_value` within 0.005 of 48.62 and
     `rep_count` >= 5.
   Pass: zero in each.
2. *Gauge false alarms and screen-caused misses (§20 criterion 2 as revised in §24).* DE/CH
   gauges, `g_shift` window, gauge-hours on the 63 gauge days and the 196 calibration days with
   any DE/CH gauge (`gauge_hourly.py`, fixed to run the burst rule), both networks, gauges on
   hot pixels included.
   - (a) *False alarms:* among gauge-hours with screened radar >= u, the share with
     gauge < 1 mm. Pass: the upper end of its 95% day-block bootstrap interval is <= 5%, the
     bound §20 set for residual contamination.
   - (b) *Screen-caused misses:* among gauge-hours with gauge >= u, the share where the screen
     lowered the radar to <= 0.5 x raw. Pass: <= 1%.
   - Descriptive: the same for raw radar, and the overall miss share (screened radar <
     0.2 u where gauge >= u), which measures the product and the support mismatch, not the
     screen.
3. *Real extremes survive (§20 criterion 4, the parts that need no build).*
   - (a) Rejected share of event tile-hours (`make_splits.event_rows`, wet tile-hours with
     max >= 1 mm) < 0.5% per event with >= 50 such tile-hours.
   - (b) For each rejection reason, its rate among tile-hours with hourly max >= 20 mm is
     <= max(3 x its rate at 5-10 mm, 0.5%). This is a stated reading of "not rising steeply".
4. *Deferred to the build:* §20 criteria 3 (tail stability of the evaluation reference) and 5
   (xi(u) drift). Both need the built test split.

**What would change the conclusion.** The gauge criteria rest on DE/CH only (§24) and on a
provisional u. A pass says the screen is ready for the full build; it does not replace the
§20 gate on the build.

**RADKLIM test result (2026-10-13; `calib/iso31_radklim/summary.md`). Passes; rule 3 stays.**
- *Test:* 69,726 isolated pixel-frames on 337 German radar-days (308 days; RADKLIM for
  2026-07/08 is not yet published). RADKLIM-dry **99.8% [99.5%, 99.9%]** against the 85% bar.
  Per radar 99.5-100%, except Isen (deisn): 62.5% on 6 radar-days (341 pixel-frames), the one
  place the signal may catch real convection.
- *Control:* 11,141 isolated pixel-frames on 831 German radar-days that are *not* excluded:
  98.0% [96.6%, 98.9%] dry. Isolated >= 31 mm/h pixels are artefacts on ordinary days too.
- *The hourly chain already removes most of them:* only 11.4% of the test pixels leave >= 5 mm
  in the screened hour (98.4% of those are RADKLIM-dry). Example: Berlin (debln) on
  2013-07-28 (Andreas): 197 isolated pixel-frames, median 2,938 mm/h, 100% dry; the chain
  leaves 2 of them >= 5 mm.

**Result (2026-10-13; `calib/gate_prebuild/summary.md`, `gate_prebuild.py`). The gate is not
passed in any rule-5 variant.** Calibration days rescanned with the current chain
(`calib/hourly_c`, 199 days, 335,996 tile-hours); gauges on 246 days (8.66 M gauge-hours,
1,733 stations).

| criterion | rule 5 off | §28 as pre-stated | year share over days with data |
|---|---|---|---|
| tile-hours rejected | 0.13% | 14.24% | 3.49% |
| 1. known failures kept (Madrid / Valjevo / 48.62) | **5 / 1 / 1, FAIL** | 0 / 0 / 0, pass | 0 / 0 / 0, pass |
| 3a. worst event rejected | 0.00%, pass | **100% (Montpellier, Cannes, Misox), FAIL** | **20.1% (Andreas), FAIL** |
| 3b. reasons rising with intensity | **ceiling: 0.04% -> 1.77%, FAIL** | same | same |

All three known-failure checks are non-vacuous: the failure is present in the raw field in 32,
46 and 79 tile-hours.

Criterion 2 (independent of rule 5; pixel values of the chain, no split-time rejection):

| u | raw: false alarms | screened: false alarms [95% CI] | screen-caused misses | overall misses |
|---|---|---|---|---|
| 10 mm | 16.2% | 4.6% [3.5, 5.9] | 0.1% | 7.6% |
| **20 mm** | 48.8% | **6.4% [3.9, 9.6] -> 2a FAIL (bound <= 5%)** | **0.1% -> 2b pass** | 8.1% |
| 30 mm | 68.4% | 8.5% [1.1, 19.6] | 0.0% | 9.5% |

**Diagnosis (descriptive, after the results).**
- *The radar-year rule causes most of the event loss.* Of the event tile-hours rejected by the
  pre-stated list, 2,028 of 2,169 come through excluded radar-years (chwei 2024 in the
  southern-Germany floods, nldbl 2014 at Pentecost, frnim 2014 at Montpellier, depro 2017 in the
  Poland derecho); 141 through radar-days with score >= 1.
- *Score only (no radar-year rule):* 1.08% of tile-hours rejected; criterion 1 passes
  (0 / 0 / 0 kept); criterion 3a fails only for Andreas (11.2%, debln 2013-07-27/28) and Boris
  (8.5%, hubud / hrdeb / sipas). Both through `iso31`, on days whose isolated pixels the chain
  already removes (Berlin: 99%, above).
- *Ceiling rejection rises with intensity by construction:* a ceiling is a high value. Whether
  the >= 20 mm rejections are real cores is not tested.
- *False alarms:* the screen cuts them from 48.8% to 6.4% at 20 mm. The 5% bound was carried
  over from the eye-review of tiles (§20); for a 2 km pixel against a point gauge part of the
  remainder is support mismatch, not contamination. Not changed after the result.

**Open, for the researcher.** (1) Drop the radar-year rule. (2) Score `iso31` on the pixels
the hourly chain leaves, so rule 5 targets what the chain misses; against it, §19 computes the
signals before repairs so that a repair cannot hide a failing radar. (3) Test the ceiling
rejections >= 20 mm against RADKLIM / gauges where they overlap. (4) Decide whether criterion
2a's bound stays at 5% for gauge comparisons, with a new criterion fixed before re-running.

**Decision (researcher, 2026-10-14, Claude agreeing). The radar-year rule is dropped.** It
caused 2,028 of the 2,169 event tile-hours the pre-stated list rejected; the known failures are
caught without it (§29 diagnosis); its premise (a radar failing on 5% of days is unreliable on
the others) was never tested, and its denominator was degenerate. Cost: a radar with chronic
contamination that never scores >= 1 is not caught; the residual-contamination estimates
(methods §6.3) cover it. `radar_day.year_rule: false`; `day_exclusions.csv` now holds the 3,029
radar-days with score >= 1. The pre-stated lists are kept as
`day_exclusions_20261013_prestated.csv` / `year_exclusions_20261013_prestated.csv`.

**Decision (researcher, 2026-10-14). `iso31` stays scored on the `clean_frame` field, before
repairs** (§19: a repair must not hide a failing radar), not on what the hourly chain leaves.
Open: whether the rest of an excluded radar-day's field is worse than normal (Claude proposes
RADKLIM agreement of non-isolated pixels, excluded vs control radar-days). If not, criterion
3a's losses (Andreas 11.2%, Boris 8.5%) are a real cost; if so, 3a should not count them.

**Ceiling rejections >= 20 mm (§29 item 3, 2026-10-14).** All 335 such tile-hours on the
calibration days are in HU (190), HR (82) and RS (63): no RADKLIM or DE/CH gauge overlaps them,
so the planned independent test cannot run on data we hold. Main values 48.62 (50.0 dBZ, hunap
/ husze 2019) and 364.63 (64.0 dBZ, HR / RS 2023); 12 at 31,575.94. Inference (untested): a
value piled at an exact dBZ level appears where reflectivity is high, so these are likely real
intense rain with a capped core, rejected as a clear error under §17; the criterion-3b rise
with intensity would then be by construction. Options: an internal-consistency test, or GPM
IMERG as a presence reference (needs an Earthdata login).

**Three follow-up tests (researcher approved 2026-10-14; Claude fixed the criteria below
before running).**

*(i) Is the rest of an excluded radar-day's field worse? (`rest_of_field.py`)* German radars,
screened hourly field C of the current chain (`quality_v4.yaml`), RADKLIM box. Test: the 337
radar-days excluded with reason `iso31` that have RADKLIM. Control: non-excluded radar-days
*of the same radars* on the RADKLIM days frozen before the fetch (`iso31_radklim/control_days.txt`).
Pixels: owned by the radar (contributing geometry); pixel-hours holding an isolated pixel-frame
(the `iso31` pixels) are left out, so only the *rest* of the field is judged.
- Measure: false-alarm share = pixel-hours with C >= 5 mm that are RADKLIM-dry (K_tol < 5 mm).
- "Worse" if the test share exceeds the control share by > 2 percentage points and the 95%
  day-block bootstrap interval of the difference lies above 0.
- If worse: excluding the radar-day is justified, and criterion 3a does not count event losses
  on such radar-days. If not: the exclusion is a cost, and rule 5 should act on the isolated
  pixels, not on the whole radar-day.
- Descriptive: the same at 10 and 20 mm; misses (K >= 5 mm, C < 1 mm); median per radar-day
  log-correlation of C and K over pixel-hours wet in either. Weather differs between the two
  sets (excluded days are not random), which the same-radar control only partly removes.

*(ii) Criterion 2a against a measured floor (`gate_prebuild.py`, replaces the 5% bound).*
DWD gauges on the gate days that have RADKLIM; gauge window `g_shift`; u = 20 mm (10 and 30
reported). OPERA = the screened pixel value with split-time rejection applied (score-only rule
5, current `day_exclusions.csv`; rejected tile-hours dropped). Hours tables for the 47 gauge
days that are not calibration days are scanned with the same chain. RADKLIM = K at the gauge
pixel, same hour.
- Measure: false-alarm share = gauge-hours with the field >= u whose gauge reads < 1 mm.
- Pass: the upper end of the 95% day-block bootstrap interval of (OPERA - RADKLIM) is <= 2
  percentage points. RADKLIM is adjusted to DWD gauges, so its share at gauge sites is
  optimistic and the test is conservative.

*(iii) Ceiling rejections >= 20 mm: capped cores or artefacts? (`ceiling_context.py`)*
Calibration days, tile-hours rejected by `rej_ceiling` with hourly max >= 20 mm (335). In the
raw frames of each such hour, every pixel of the tile equal (+-0.005) to a ceiling value of its
owning radar-year (`ceilings.csv`). Context = median, in dBZ, of the valid non-ceiling pixels of
its 5x5 window (Z = 200 R^1.6, as `radar_screen.dbz`); "embedded" = context >= ceiling dBZ -
10 dB.
- >= 80% embedded: the high-intensity rejections are real intense rain with a capped core. The
  3b rise is then by construction; the researcher decides whether the rule should mask the
  ceiling pixels (NaN) and keep the tile-hour instead of rejecting it.
- <= 20% embedded: they are artefacts in weak or dry surroundings; the rule stands and its 3b
  rise is artefact-driven.
- In between: reported per ceiling value and radar.

**Results of the three follow-up tests (2026-10-14).**

*(i) Rest of the field: WORSE, so excluding the whole radar-day is justified*
(`calib/rest_of_field/summary.md`). 348 excluded German radar-days against 2,147 non-excluded
radar-days of the same radars, isolated pixel-hours left out:

| measure | excluded | control | difference [95% CI] |
|---|---|---|---|
| false alarms, C >= 5 mm (RADKLIM-dry) | 40.2% | 12.3% | +27.9 [+17.2, +42.5] |
| false alarms, C >= 10 mm | 52.2% | 5.5% | +46.7 [+32.1, +63.4] |
| false alarms, C >= 20 mm | 66.6% | 4.0% | +62.6 [+39.7, +85.1] |
| misses (K >= 5, C < 1 mm) | 9.4% | 7.4% | +1.9 [-0.8, +5.7] |
| median log-correlation with RADKLIM | 0.36 | 0.58 | |

As pre-stated, criterion 3a no longer counts event losses on excluded radar-days: Andreas
(debln) and Boris (hubud / hrdeb / sipas) are correct removals. The evidence is from German
radars; Boris's are HU / HR / SI (the §24 transfer assumption).

*(ii) Criterion 2a against RADKLIM: FAIL.* 6.87 M DWD gauge-hours on 243 days, rejected
tile-hours dropped. At u = 20 mm OPERA screened 6.3% (n = 378), RADKLIM 0.9% (n = 426);
difference +5.4 [+2.3, +9.9] against <= +2. At 10 mm +1.1 [-0.1, +2.6]; at 30 mm +8.0
[+0.0, +21.6]. RADKLIM's floor is optimistic (gauge-adjusted), but the excess is real
contamination. The 24 false-alarm gauge-hours (18 stations, 13 days, none on a hot pixel):
- ~9 with RADKLIM ~0 mm, i.e. artefacts the chain leaves untouched (screened = raw). Largest:
  Kleiner Feldberg/Taunus 2018-04-30, 6 hours of 23-83 mm at one pixel, a single-day
  stationary echo, not a local peak (the "extended blob" §23 left out); Ostenfeld
  2014-09-29 (60 mm), Bad Berleburg 2023-07-09, Ellern 2018-04-23, Traunreut 2023-07-19.
- ~15 inside convection (RADKLIM 5-20 mm, gauge dry): most likely support mismatch.

*(iii) Ceiling context: artefacts* (`calib/ceiling_context/summary.md`, each pixel counted once):
166,994 ceiling pixel-frames in the rejected tile-hours >= 20 mm; 13.7% embedded (<= 20%).
Surrounding median ~0.6 dBZ; 44-72% have no rain around them. The ceiling rule stands; its
criterion-3b rise comes from artefacts, not from real cores.

**Gate status with rule 5 = score only (adopted).** 1.08% of tile-hours rejected.
- 1, known failures: pass (0 / 0 / 0 kept, all non-vacuous).
- 3a, events: pass once losses on excluded radar-days are not counted (i); every other event
  0%.
- 3b: rej_ceiling (0.04% -> 1.77%) and rej_radar_day (1.14% -> 4.80%) rise with intensity.
  Both are explained by (iii) and (i): what they remove at high intensity is contamination.
- 2a: **fail**, from residual contamination such as single-day stationary echoes.
- 2b: pass (0.1%).
The gate therefore turns on 2a: a rule for what the chain leaves (screening review,
2026-10-14).

## 30. Low-hanging fruit of the screening review (2026-10-14)

**Why.** The per-rule table (`calib/rule_effect/summary.md`, 199 calibration days) shows the
static-clutter (hot) repair as the largest remover at 20-100 mm: 13-26% of pixel-hours on random
days, 21% at 50-100 mm on event days. It is the one rule that failed its test (§24: 46% of
affected gauge-hours moved closer, 48% further). The no-temporal-support rule lowers 6% of
event-day 50-100 mm pixel-hours and was never tested. The residual false alarms of §29 (2a)
include single-day stationary echoes that miss `p1_min` = 24 by 4 frames (Kleiner Feldberg
2018-04-30, Paris-CDG 2021-07-14: 20 frames each).

**Decision (researcher asked for all of them, 2026-10-14; Claude fixed the criteria below before
running).** Tests use `scripts/data_quality/rule_validate.py` on the 196 calibration days with
RADKLIM, the RADKLIM box, the current chain as the base, and split-time rejection (current rule 5)
applied. Removal = a kept pixel-hour with C_without >= 1 mm and C_with <= 0.5 C_without.
RADKLIM-dry = K_tol < 5 mm, as in §25/§26. Raw bins 10-20 / 20-50 / 50-100 / >= 100 mm.
Pass = removals RADKLIM-dry >= 85% in every bin with >= 30 removals.

1. *Static-clutter repair.* Without vs with `hot`.
   - Pass: it stays.
   - Fail: the unconditional repair goes. The hot mask only lowers the persistence threshold
     (`p1_prior_min` 12 on hot pixels, the §24 proposal). That replacement's own removals must
     pass the same test.
2. *No-temporal-support rule.* Without vs with `unsupported`. Pass: it stays. Fail: off until
   redesigned.
3. *Persistence threshold.* `p1_min` 20 and 16 against the current 24 (`p1_ring_min` 12
   unchanged). The removals added by the lower value must pass on both halves of the days
   (odd and even day index). Pick the lowest passing value; if none passes, keep 24. Then re-run
   §29 criterion 2a with whatever is adopted.
4. *`min_wet_jump`* (radar-disk temporal check, now 0.15 from 20 cases). The 99.9th percentile
   of s = min(w_t - w_{t-1}, w_t - w_{t+1}), the wet-fraction excess over both neighbour
   frames. Taken over radar-frames in normal operation: 500 random days (seed 0) of the
   2026-10-09 radar pass, radar-days with score < 1, no flagged frame, n_valid >= min_valid.
5. *Hourly tail threshold u (provisional).* The screened hourly level with the same pixel
   exceedance frequency as 31 mm/h has in the screened 15-min frames. Same 100 random
   calibration days, full grid, kept tile-hours. Final u comes from a threshold-stability
   analysis on the build (§21).

*Descriptive, for the slides, not criteria:* exceedance counts of raw OPERA, screened OPERA and
RADKLIM on the same German pixel-hours, and of DE/CH gauges against OPERA at the gauge pixels.
CombiPrecip and COMEPHORE are not on disk.

**Results (2026-10-14; `calib/rule_validate/summary.md`, 193 RADKLIM days, 2.62 M pixel-hours).**

| test | removals (10-20 / 20-50 / 50-100 / >= 100 mm) | RADKLIM-dry | verdict |
|---|---|---|---|
| static-clutter repair | 41 / 32 / 4 / 12 | 100% / 100% / 100% / 67% | **pass: stays** |
| hot mask as persistence prior | 15 / 18 / 3 / 0 | 100% | not needed |
| no temporal support | 216 / 51 / 4 / 6 | 97.2% / 100% / 100% / 100% | **pass: stays** |
| `p1_min` 20, added removals | odd 175 / 48, even 137 / 36 | 66-78% (24-28 confirmed in 10-20) | **fail** |
| `p1_min` 16, added removals | odd 392 / 98, even 309 / 64 | 67-80% (54-59 confirmed) | **fail** |

- *`p1_min` stays 24.* Lowering it removes real rain. The single-day stationary echoes behind
  the 2a false alarms (Kleiner Feldberg) need a rule keyed on something else, such as a fixed
  location with a dry ring over many hours. That is not a threshold change.
- *The hot repair passes over Germany with few cases* (89 removals): most of its Europe-wide
  effect (rule-effect table) is outside RADKLIM. The §24 gauge result (46% closer / 48% further,
  183 gauge-hours, mostly low intensity) and this presence result disagree in tone. The
  presence test is the one that speaks to contamination of the tail, so the repair stays.
- *`min_wet_jump` = 0.021* (item 4: q99.9 of s over 6.86 M normal radar-frames; q99 0.005,
  q99.99 0.063; share >= 0.15 was 2.1e-5). Recomputing the flags for all 5,023 days gives the same
  48 flagged radar-frames as 0.15, and the same gate results. Adopted in `quality_v4.yaml`.
- *Chain values are unchanged,* so §29 criterion 2a is unchanged (fail, +5.4 points) and is
  not re-run.
- A bug in `rule_validate.py` (`p is not False` on a NumPy bool) printed PASS for the `p1`
  tests; it was fixed before this result was recorded.

*Screened OPERA against RADKLIM on the same German pixel-hours* (kept tile-hours):

| field | >= 10 | >= 20 | >= 30 | >= 50 | >= 75 | >= 100 |
|---|---|---|---|---|---|---|
| raw OPERA | 253,856 | 43,715 | 19,772 | 13,756 | 12,040 | 9,511 |
| screened OPERA | 234,870 | 29,042 | 5,987 | 569 | 71 | 15 |
| RADKLIM | 315,170 | 28,710 | 4,873 | 357 | 28 | 7 |

Screened OPERA matches RADKLIM at 20 mm and has 1.2x / 1.6x / 2.5x its counts at 30 / 50 / 75 mm.
RADKLIM underestimates peaks (§4 of the methods draft), so the excess is an upper bound on
residual contamination, not a measurement of it.

*Provisional hourly tail threshold (item 5; `calib/dist_hist/u_provisional.txt`).* On the 100
random calibration days, full grid, kept tile-hours: 31 mm/h is exceeded by 3.99e-5 of screened
15-min pixel-frames; the screened hourly sums exceed **u = 20.2 mm** at the same frequency. This
agrees with the provisional 20 mm of §29. The final u comes from a stability analysis on the build.

Slides of these results: `notes/slides/2026-10-14_screening/deck.pdf`.

## 31. Stationary isolated echoes (2026-10-14)

**Problem.** After §30 the readiness gate fails only on false alarms (§29, criterion 2a: +5.4
points over RADKLIM at 20 mm). The clearest cause is a single-day stationary echo the chain
leaves untouched. Kleiner Feldberg/Taunus, 2018-04-30: a blob of 20-40 px (>= 5 mm/h) peaking
at 50-107 mm/h, at the same place for ~14 h, in a dry ring (ring median 0.00-0.07 mm/h); clean
on the days before and after; gauge and RADKLIM dry. It is a local peak in 20 frames, so the
persistence rule (24) misses it, and lowering that threshold removes real rain (§30). It is not
a burst (it lasts) and it supports itself in time.

**The signature.** A moving cell's core covers a given pixel for one or two frames: a 5-10 km
core moving at 30 km/h advances ~4 px per frame. A stationary target fills the same pixels for
hours. Real stationary convection (orographic anchoring) usually sits in wider rain, which the
dry-ring condition keeps.

**The rule (`hourly.rules.stationary`, off until accepted; code bit 512).** On the screened
frames after the persistence rule:
- *Isolated component in frame t:* an 8-connected component of pixels >= 1 mm/h, at most 150 px
  (600 km^2), containing a pixel >= 10 mm/h, whose 3-px ring (6 km) is >= 50% valid with a
  median < 0.1 mm/h.
- *Stationary pixel:* inside an isolated component in >= `min_frames` frames of the processing
  day.
- *Repair:* in each frame, an isolated component holding a stationary pixel is set to its ring
  median, i.e. ~0. Repairs only lower values.

**Test (criteria fixed before running, Claude, researcher asked 2026-10-14).** Candidates
`min_frames` 12 (3 h) and 8 (2 h).
1. *RADKLIM presence* (`rule_validate.py`, the 193 RADKLIM calibration days, as §30): removals
   of the rule against the current chain must be RADKLIM-dry >= 85% in every raw bin with
   >= 30 removals, on both halves of the days.
2. *Real storms outside Germany:* on the 79 event days, full grid, the rule must lower by >= 50%
   at most 1% of the pixel-hours that the current chain leaves at >= 20 mm.
3. Adopt the lowest `min_frames` passing 1 and 2. If none passes, the rule is not adopted.
4. With the adopted value, re-run §29 criterion 2a (gauges against RADKLIM). This is reported,
   not a condition for adopting the rule.
*Not evidence:* that it removes Kleiner Feldberg. That is the case the rule was designed on; it
is only a check that the code does what was intended.

**Result (2026-10-14). Not adopted: both candidates fail criterion 2.**
- *RADKLIM presence* (`calib/rule_validate_stationary/summary.md`): `min_frames` 12 passes
  (removals 89.7% / 98.5% dry at 10-20 / 20-50 mm on odd days, 94.3% / 91.7% on even days);
  `min_frames` 8 fails (49-60% dry; 193-305 removals at 10-20 mm confirmed by RADKLIM).
- *Event days* (`calib/stationary_events/summary.md`, 79 days, full grid; 340,224 pixel-hours
  left >= 20 mm by the current chain): `min_frames` 12 lowers 3,731 of them by >= 50%,
  **1.097%** against <= 1%; `min_frames` 8 lowers 2.05%. Both fail. The bar was fixed in advance
  and is not moved.
- *Where:* the losses concentrate in quasi-stationary real storms: Elvira/Friederike 883,
  Med. derecho 562, Boris 412, Valencia DANA 378, HyMeX IOP6 357, Poland derecho 215,
  Emilia-Romagna (1st) 202, Aude 172. A back-building storm regenerates a convective core over
  one place, sometimes in dry air: the same signature as the Kleiner Feldberg echo. The dry-ring
  condition does not separate them. Germany alone (RADKLIM) does not show this, because its
  calibration days hold few such storms.
- *Consequence:* step 4 (re-running §29 criterion 2a) is not done. The gate still fails 2a
  (+5.4 points). The rule stays in the code, off (`hourly.rules.stationary: false`), with its
  tests.
- *What could separate them (ideas, not tested):* recurrence of the same echo location on
  other days, which a fixed target or interference would show but weather would not (Kleiner
  Feldberg was a single day, so this would not catch it); the absence of a growth and decay
  cycle; lightning, as an independent marker of real convection (not on disk).

## 32. Gate criteria 3 and 5 on the v4 build, hourly levels (2026-10-19)

**Question.** §20 states criteria 3 (the evaluation reference is stable) and 5 (residual drift
of xi(u) is explained) in 15-min terms: levels 31 / 53 / 89 mm/h. What are they on the hourly
v4 build (`OPERA/patches_v4`, built 2026-10-17)?

**Decision (researcher asked to run them 2026-10-19; Claude fixed the definitions below before
running).**
- *Hourly levels, by matched exceedance frequency* (screened fields, 100 random calibration
  days, full grid, as the provisional u of §30):

  | 15-min level | per-pixel-frame P | hourly level |
  |---|---|---|
  | 31 mm/h | 3.99e-5 | **20.2 mm** (the provisional u) |
  | 53 mm/h | 1.31e-5 | **31.7 mm** |
  | 89 mm/h | 3.73e-6 | **50.1 mm** |

- *Criterion 3* (`gate_tail.py`), on the ODYSSEY test split, unweighted (what
  `eval_extremes.py` computes):
  - Statistics: exceedance rates per valid pixel-hour at 20.2 / 31.7 / 50.1 mm (rates, not
    counts, because dropping days lowers counts by construction); the GPD fit above u = 20.2 as
    in `tail_metrics.py` (location 0, unweighted), its xi, and the return levels at
    lambda = 10 / 100 / 1000 expected exceedances.
  - Perturbation: drop the 20 non-event days with the largest share of the test split's
    exceedances >= 20.2 mm; recompute.
  - Pass: every recomputed statistic lies inside the 95% day-block bootstrap interval (B = 200)
    of the full-sample statistic.
- *Criterion 5* (`pot_threshold.py --version v4`): xi(u) and sigma*(u) over an hourly
  threshold grid (5-100 mm), day-block bootstrap, on train / validation / test / NIMBUS,
  weighted and unweighted; cluster maxima; latitude bands; seasons; plus a GPD fit truncated at
  500 mm. Drift of xi(u) above u is "explained" when it is matched by one named cause: upper
  truncation (the truncated fit removes it), regional or seasonal mixing (the subsets are flat
  where the pool drifts), or a sub-asymptotic threshold (the drift fades above some u). This is
  a judgement on the plots; Claude proposes, the researcher decides.
- *Hourly u for the evaluation tables:* the lowest grid u above which the bootstrap intervals of
  xi and sigma* (train, weighted) contain the point estimates at every higher u that has
  >= 1,000 exceedances. 20.2 mm stays provisional until then.

**Result, criterion 3 (2026-10-19; `quality_v4/gate_tail/summary.md`). FAIL as pre-stated.**
ODYSSEY test split: 218,513 tile-hours, 486 days, 3.58e9 valid pixel-hours, 489,170 >= 20.2 mm.
The 20 dropped non-event days hold 26.7% of those exceedances.

| statistic | full | without the 20 days | 95% day-block interval | inside |
|---|---|---|---|---|
| rate >= 20.2 mm | 1.366e-4 | 1.062e-4 | [1.18, 1.59]e-4 | no |
| rate >= 31.7 mm | 3.727e-5 | 2.893e-5 | [3.17, 4.38]e-5 | no |
| rate >= 50.1 mm | 8.553e-6 | 6.976e-6 | [7.01, 9.99]e-6 | no |
| xi (u = 20.2) | 0.237 | 0.261 | [0.217, 0.255] | no |
| RL, lambda = 10 | 43.4 mm | 43.7 mm | [42.4, 44.3] | yes |
| RL, lambda = 100 | 83.4 mm | 86.5 mm | [79.3, 86.8] | yes |
| RL, lambda = 1000 | 152.3 mm | 164.5 mm | [140.5, 162.9] | no |

- *The dropped days are mostly three summer convective periods:* 16-30 June 2021 (12 of 20),
  9-28 June 2024 (6) and 15-16 August 2022 (2), plus 2013-06-07 and 2023-07-18. They surround
  catalogued events (Swiss hail and the Moravia tornado in 2021; the June 2024 floods; the 2022
  Mediterranean derecho), whose own days were not dropped. Inference, not tested: real
  outbreaks concentrated in a few test weeks, not contamination.
- *A design flaw in the criterion (noted after the result; the bar is not changed):* it compares
  a targeted removal of the most influential days with a random day-resampling interval.
  Whenever ~4% of days hold ~25% of the tail, that comparison fails almost by construction, with
  or without artefacts.
- *Open, for the researcher:* (a) accept, and report every tail statistic of the evaluation
  tables with its day-block interval plus a leave-period-out sensitivity; (b) test the dropped
  days for contamination with RADKLIM and the gauges where they overlap (Germany, Switzerland);
  (c) replace the criterion, with a new pass rule fixed before re-running.

**Contamination test of the dropped days (option (b); researcher asked 2026-10-19; Claude fixed
the criteria before running).** `scripts/data_quality/dropped_days_check.py`.
- *Days:* the 20 dropped days, against a control of 60 non-dropped, non-event test days drawn at
  random (seed 0).
- *Pixels:* the test split as stored (kept, sampled tile-hours), screened value C >= 20.2 mm.
- *RADKLIM (Germany):* the share of those pixel-hours that are RADKLIM-dry (K_tol < 5 mm, as in
  §25/§26).
- *Gauges (DE and CH):* among gauge-hours whose tile-hour is in the test split and where
  C >= 20.2 mm, the share with gauge < 1 mm (`gauge_hourly.py`, `g_shift`, current chain).
- *Contaminated* if, for either reference, the dropped days' share exceeds the control's by
  > 2 points and the 95% day-block interval of the difference lies above 0 (the §28 (i) rule).
- *Descriptive, not a criterion:* the gauges' own exceedance frequency (>= 20.2 mm per gauge-hour)
  on dropped against control days. If the gauges also find the dropped days much wetter, the
  concentration is weather.
- *Consequence:* contaminated -> a named failure type, which per §20 justifies a v5 for that type.
  Not contaminated -> the criterion-3 failure is real concentration of weather, and the
  criterion is replaced (option (c)), with the new rule fixed before re-running.

**Decision (researcher, 2026-10-19, on Claude's proposal; written before the contamination
test's result). §20 criterion 3 is replaced.**
- *Why.* The old rule compared a targeted removal of the most influential days with a random
  day-resampling interval. With heavy-tailed daily contributions (here 4% of days hold 27% of
  the tail) the targeted change exceeds the random interval by construction, whatever the data.
  It is like judging "remove the 20 largest values" against the standard error of the mean.
  The old rule also cannot tell contamination concentrated on a few days from real weather
  concentrated on a few days: both give the same signature. Widening the interval would be
  tuning after the result, and would not change the outcome (the 20.2 mm rate falls to 1.06e-4
  against a lower bound of 1.18e-4).
- *New criterion 3, a gate:* the most influential test days (the 20 non-event days with the
  largest share of exceedances >= u) are not contaminated. The test is the dropped-days test
  above (RADKLIM and DE/CH gauges against 60 random control days; contaminated if
  dropped - control > 2 points with the 95% day-block interval above 0). Pass = not contaminated
  for both references.
- *New reporting rule, not a gate:* every tail statistic of the evaluation tables (exceedance
  rates and ratios, xi, return levels and their biases) is reported with its 95% day-block
  bootstrap interval. A leave-period-out variant goes with it: without the most influential
  outbreak period (on v4 test, 16-30 June 2021), and without the 20 most influential days. The
  paper states how much a few outbreaks drive each number.
- *The old rule's result stays recorded above* (fail; 5 of 7 statistics outside their intervals).

**Results (2026-10-19).**

*New criterion 3: PASS* (`gate_tail/dropped_check/summary.md`):
- RADKLIM (Germany): pixel-hours >= 20.2 mm that are RADKLIM-dry are 2.2% on the dropped days
  (n = 20,137, 20 days) against 6.0% on control days (n = 2,585, 30 days with RADKLIM);
  difference -3.7 points [-12.4, -1.0]. The dropped days are cleaner than ordinary days.
- Gauges (DE + CH): test gauge-hours with C >= 20.2 mm and gauge < 1 mm are 3.7% (n = 298)
  against 2.1% (n = 48); difference +1.6 points [-5.8, +6.0]. Small control sample.
- Descriptive: the gauges themselves record >= 20.2 mm 15.3x more often on the dropped days
  than on control days (OPERA 17.9x). The concentration is weather, not contamination.

*Criterion 5: drift explained, by a sub-asymptotic threshold* (Claude's reading of
`pot_threshold/report.md` and `fig/`; the researcher decides):
- xi(u), train weighted: 0.41 at 5 mm, 0.31 at 10, 0.28 at 20.2, 0.21 at 31.7, 0.10 at 50.1,
  then 0.07-0.12 up to 100 mm. Validation, test and NIMBUS show the same shape.
- *Not truncation:* the GPD truncated at 500 mm gives the same xi at every u.
- *Not regional or seasonal mixing:* xi falls with u inside every latitude band and season
  (JJA 0.27 -> 0.08, MAM 0.31 -> 0.11, south of 46N 0.23 -> 0.10). The pool is not a mixture of
  flat subsets.
- *Sub-asymptotic:* the drift stops at ~50 mm. Cluster maxima give a lower xi at high u
  (0.02-0.08 above 50 mm) than single pixels (0.07-0.12): dependence within storms inflates the
  pixel-level shape slightly.

*Hourly evaluation threshold, by the pre-stated rule: u = 50.1 mm.* It is the lowest grid u
whose xi and sigma* intervals (train, weighted) contain the point estimates at every higher u
with >= 1,000 exceedances. (At 60 mm the 100 mm point falls just outside, but the rule takes the
lowest passing u.) It coincides with the hourly equivalent of 89 mm/h. Exceedances above it:
train 175,937 (2,414 days); test, unweighted, 30,589 (404 days). 20.2 mm stays the level of the
split strata and the exceedance-rate columns. The POT fits of the evaluation tables use 50.1 mm.

*Gate status on v4:* 1 pass; 2a fail (+5.4 points over RADKLIM, residual stationary echoes; the
rule tried in §31 was not adopted); 2b pass; 3 (new) pass; 4 pass (event losses only on
radar-days shown to be bad); 5 explained (pending the researcher). By §20 "Where we stop": train
on v4 and report 2a's measured contamination as a limitation, unless a v5 for the named type
(stationary echoes) is judged worth its cost.

**Decision (researcher, 2026-10-19).**
- *Criterion 5 is accepted as explained by a sub-asymptotic threshold.* The explanation, with a
  synthetic check, is in the methods draft §5.6.
- *Train on v4.* The 2a failure is reported as a minor limitation (methods draft §6.3): about 9
  residual artefact gauge-hours, single-day stationary echoes, +5.4 points of false alarms at
  20 mm over RADKLIM. No v5. The gate closes with this.

## 33. Training on v4: hourly thresholds, config and data preparation (2026-10-19)

**Question.** The loss, the gamma targets and the FSS levels use `PHYSICAL_THRESHOLDS`, 15 levels
in mm/h from 0.1 to 150 set for 15-min rates. The v4 store's `physical_thresholds.npy` still
holds them. What are they for the hourly target?

**Decision (researcher said go ahead with the training preparation, 2026-10-19; Claude chose the
rule).** Each level is mapped to the hourly level with the same exceedance frequency (screened
15-min frames against screened hourly sums, 100 random calibration days, full grid: the rule of
§30 and §32, so each level keeps its rarity and its role in the loss):

| mm/h (15-min) | 0.1 | 0.17 | 0.28 | 0.48 | 0.81 | 1.4 | 2.3 | 3.9 | 6.5 | 11 | 19 | 31 | 53 | 89 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **mm in 1 h** | 0.1 | 0.192 | 0.314 | 0.509 | 0.828 | 1.35 | 2.11 | 3.43 | 5.36 | 8.61 | 13.5 | 20.2 | 31.7 | 50.1 | 79.0 |

The ratio falls from ~1 in light rain to 0.53 at the top: light rain lasts the whole hour, intense
peaks a frame or two. With these levels, the evaluation defaults (`--pot_threshold` = third
highest, FSS = top three) would be 31.7 / {31.7, 50.1, 79.0}. The POT fits use u = 50.1 instead
(§32), so `eval_extremes.py` is run with `--pot_threshold 50.1`.

- *Config:* `configs/config_v4.yaml` = `config_v3.yaml` with the v4 store and metadata paths and
  this grid. Every precipitation value and threshold is in mm per hour of accumulation.
- *Persistence thresholds* (the B0 / B1 noise floor of the gamma targets) are recomputed on the
  hourly store. In `TOPOLOGY_MODE: euler` they are diagnostics only.
- *Gamma targets:* computed after the store build (`scripts/hpc/prep_train_v4.sh`), 8 workers.
- *Smoke run:* the UNet backbone, 1 epoch on 0.5% of train, GPU 1. It checks loading, finite
  losses and the startup lines, nothing about skill.
- *Not yet:* sigma_r (`compute_residual_stats.py`) is recomputed per new backbone, before any
  flow-matching run.

## 34. Ownerless tile-hours excluded; the pre-December-2015 speckle floor flagged and measured (2026-10-19)

**Findings** (`notes/data_analysis_v4/stats.md`, the full-archive scan):
- *A step change on 2015-12-12.* The daily share of completely dry tile-hours (max < 0.1 mm) goes
  from ~0.02-0.07 up to 2015-12-11 to 0.20-0.29 from 2015-12-12 on. Before it, tiles that look
  dry still carry scattered values (median maximum 0.2 mm where the wet fraction is < 0.2%);
  after it they are 0. A low-level speckle floor in the earlier ODYSSEY processing (inference;
  the processing change itself is not identified).
- *58,124 tile-hours have no owning radar* (6 tile positions, all 2012-2015: rows 512-768 x
  cols 1536-1664, and 1792 / 128). Rule 5 (radar-day exclusion) and rule 3 (radar-frame flags)
  cannot act on them, so they are an unscreened path for contamination. In the store: train
  13,441, validation 842, test 3,033, NIMBUS 0.

**Decision (researcher, 2026-10-19).**
1. *Ownerless tile-hours are excluded,* at the metadata level (the store is unchanged):
   `patches_v4/meta/*_owned.txt` (5 columns, the 5th = store row), which `configs/config_v4.yaml`
   points to. The original lists stay for provenance.
2. *Pre-2015-12-12 tile-hours are flagged, not removed* (`patches_v4/meta/full_<split>_flags.csv.gz`:
   store_row, ownerless, pre_break), and their effect is measured:
   - *Data level, now* (`scripts/analysis/speckle_effect.py`, Claude's design, descriptive): on
     train rows sampled before and after the break and matched by tile mean rain, the Minkowski
     functionals at the lowest thresholds (A, P, chi at 0.1 / 0.192 / 0.314 mm), the count of
     single-pixel components, and how far zeroing values below 0.2 or 0.3 mm in pre-break rows
     closes the gap. This says how much of the low-threshold target is noise.
   - *Training level, after the first v4 runs:* a backbone trained on post-break rows only
     against one trained on all, both evaluated on post-break test rows. A to-do, not run now.

**Results (2026-10-19).**
- *Exclusion applied* (`scripts/dataset_v2/filter_meta_v4.py`, `patches_v4/meta/filter_report.txt`):
  train 13,441 / validation 842 / test 3,033 / NIMBUS 0 rows dropped. The kept lists hold train
  1,308,448, validation 172,618, test 215,480, NIMBUS 381,736. The store validator passes on them
  (`validation/store/20261019_1621_9b51580.md`). Pre-break rows among the kept: train 313,644
  (24%), validation 24,005, test 68,261.
- *Speckle effect, data level* (`notes/data_analysis_v4/speckle_effect.md`, 300 rows per era and
  tile-mean bin, medians):

  | tile mean (mm) | chi @0.1 post / pre / pre with < 0.2 zeroed | single-pixel components post / pre | A @0.1 (km²) post / pre zeroed at 0.2 |
  |---|---|---|---|
  | < 0.005 | 2 / 13 / 4 | 1 / 7 | 20 / 36 |
  | 0.005-0.02 | 31.5 / 58 / 29 | 16 / 29 | 1,610 / 490 |
  | 0.02-0.1 | 41 / 81.5 / 56 | 23 / 50 | 6,290 / 3,070 |
  | 0.1-0.5 | 39 / 59 / 60 | 26 / 46 | 19,700 / 14,400 |
  | >= 0.5 | 22.5 / 28.5 / 35.5 | 21 / 34 | 39,600 / 34,000 |

  - At matched tile mean, pre-break fields are 1.3-6.5x more fragmented at 0.1 mm (chi), with ~2x
    the single-pixel components. This is the threshold range that carries most of the loss
    gradient (CLAUDE.md: 83% from thresholds <= 0.28).
  - *A noise floor does not map pre onto post.* Zeroing below 0.2 mm brings chi down in the light
    bins but cuts the area at 0.1 mm to a third of post-break: real light rain sits at 0.1-0.2 mm too.
  - *Not only the lowest threshold:* at 0.31 mm, pre-break chi is still ~1.5x post in the middle
    bins. The early fields are noisier in texture, not only speckled. Matching by tile mean leaves
    season and radar-network differences uncontrolled (inference).
- *Consequence:* no floor is applied. The pre-break flag stays, and the training-level test (a
  backbone on post-break rows only against one on all rows, evaluated on post-break test rows)
  decides whether pre-break rows stay in training.

## 35. The target is the advection-corrected hourly field (2026-10-19; revises §25)

**Question.** §25 kept the plain ACRR sum as the target and stored the advection-corrected
field (`noblend_nearest`, the one variant passing its pre-stated tests) alongside, "revisited
when training".

**Decision (researcher, 2026-10-19, after looking at the v4 gallery,
`notes/data_analysis_v4/gallery_*.pdf`).** The plain sum is not usable: in many hours the four
15-min positions of a moving cell are visible one after the other (the beading of §21). The target
becomes the advection-corrected field. For a structural loss this is decisive: beading adds +2 to
+5 to the Euler characteristic at 2 mm against RADKLIM and 1.2-2x power at 6-16 km (§21), and the
loss rewards exactly those.

**How (Claude's design).** A separate store, `OPERA/patches_v4_adv`, built from `patches_v4`
after the build finishes (`scripts/dataset_v2/build_adv_store.py`):
- `original_precip` = `adv_precip`; coarse and interpolated fields recomputed from it
  (`coarsen_and_interpolate`); `max_rate` and `n_wet` copied (they describe the 15-min frames);
- rows whose corrected field has any NaN dropped (no complete H-60..H+15 window), after the §34
  ownerless exclusion; the dropped share is reported per split;
- metadata lists rewritten with the corrected tile max and the new store rows; scaler = log1p of
  the corrected train max; DEM stats copied (same tiles); the pre-break flags carried over;
- `configs/config_v4_adv.yaml` points to it. Nothing downstream changes: loader, gamma targets,
  validators and evaluation all read `original_precip` from the configured store, so no code path
  has to choose between the two fields.

**What carries over, and what is re-derived.**
- The screening, the gate (§29, §32) and the pre-break flags concern the 15-min frames and the
  tiles, not how they are summed. They carry over.
- The hourly thresholds (§33) and u = 50.1 mm (§32) were derived on plain sums. The corrected
  field lowers slow-tile maxima by ~4% (§25). They are kept for now, and `pot_threshold.py` plus the
  matched-frequency grid are re-run on the new store before evaluation.
- §25's gauge check of the corrected field passed narrowly (median ratio on gauge-hours >= 20 mm
  -6%, interval [-10%, +0.4%]). Stated as a limitation.

**Measured on the built store (2026-10-21): the correction lowers tile maxima far more than ~4%.**
Corrected / plain tile max, train (test alike): median 0.66-0.78 in every intensity bin from 1 mm to
> 50 mm, q10 0.29-0.41, q90 0.96-1.00. Tile-hours >= 50.1 mm: 26,072 -> 14,196 (train), 5,293 -> 2,398
(test); >= 20.2 mm: 152,947 -> 84,380. The ~4% of §25 was for slow tiles. The plain sum credits a cell's
full 15-min rate to each of its four positions, so a fast cell's beads hold more mass per pixel than
its track does; the correction spreads it. At gauges the effect is small (-6% median at >= 20 mm,
above), which fits beads being where the tile max sits, not where the mass is. Consequences: u = 50.1
mm and the matched-frequency thresholds must be re-derived on this store (already planned), and the
extremes subsets (selected by tile max) shrink. The gallery (`gallery_plain_vs_adv.pdf`) shows the
same: Euler characteristic lower at all levels, maxima 13-25% lower on the convective pages shown.

## 36. Radars biased against their neighbours: the boundary-jump test (2026-10-19)

**Question.** In the v4 climatology the Romanian, Hungarian and Slovak radars stand out
(`notes/data_analysis_v4/radar_tail.md`, from 2015-12-12):
- Bobohalma (robob) has the 2nd-highest pixel frequency >= 20 mm of 116 radars (2.6e-4), with a
  pixel >= 20 mm on 46% of its radar-days;
- Napkor (hunap), Szentes (husze) and Kojsovska hola (skkoj) are dry less often (0.17-0.19
  against a median of 0.25), with median tile means up to 4x the median radar.
Their heavy hours are spread over many days (top-1%-days share 0.20-0.36 against a median of
0.47): chronic, not occasional failures. Real wet climate or radar bias?

**The test (researcher approved 2026-10-19 and expects the flagged radars to be biased;
Claude fixed the criteria before running).** Climate does not change at a radar ownership
boundary; a radar that reads high does. The radar pass stores, per radar and frame, `bjump` =
mean log1p(rate) in the 6 km band just inside the radar's owned area minus that in the 6 km band
just outside it, owned by neighbours (`radar_screen.radar_frame_features`, clean_frame field,
contributing geometry). `scripts/analysis/radar_boundary.py`:
- *Frames used:* the radar's wet fraction (>= 1 mm/h) >= 0.05 and >= 50 valid pixels in each band.
- *Radar-day value:* median `bjump` over its used frames, if >= 8 frames.
- *Radar-year value:* median over its radar-days (>= 20 days), with a 95% day-block bootstrap
  interval (B = 500).
- *Flagged radar-year:* its value is in the top 5% of all radar-years AND the interval lies above 0.
- *Consequence:* tile-hours of a flagged radar-year, where the radar owns >= 10% of the tile
  (the rule-3 / rule-5 share), are excluded at the metadata level, as in §34, before the
  advection store is built. The excluded share of tile-hours and of the tail is reported.
- *Limitation, stated in advance:* an ownership boundary that coincides with a real climate
  gradient (a mountain crest, a coastline) gives a jump without any bias. A radar whose whole
  area is wetter than all its neighbours' areas could be flagged on climate. The top-5% cut and
  the interval reduce this; they do not remove it.

**Result (2026-10-19; `notes/data_analysis_v4/radar_boundary.md`). The exclusion is NOT applied yet:
the test does not measure what the question needs.** 1,987 radar-years; top-5% cut 0.018 (log1p:
~2% in 1 + rate); 100 radar-years from 49 radars flagged, all with jumps of 0.018-0.039 (2-4%).
- *Who is flagged:* many Swedish radars in 2024-2026 (NIMBUS era), the Swiss Albis (chalb) and
  La Dole (chdol), where an Alpine gradient plausibly coincides with the boundary (the limitation
  stated in advance), French and German radars, and some Slovak and Romanian radars (skkub, sklaz,
  skjav 2019, rotim, robar).
- *Who is not flagged:* the radars that stood out in the tail diagnostic: Bobohalma (robob),
  Napkor (hunap), Szentes (husze), Kojsovska hola (skkoj). Their jumps are 0.012-0.016.
- *Why:* the median of mean log1p over wet frames is dominated by light rain. It finds small bulk
  offsets, not the heavy-rain excess or the low-level floor that raised the suspicion.
- *Pending the researcher's decision (2026-10-20):* (a) apply as pre-stated anyway; (b) replace it
  with a tail-focused boundary test (frequency of rates >= 10 / 31 mm/h in the inner vs outer band),
  which needs band statistics recomputed from the frames (a new radar pass, ~8 h), with its own
  criterion fixed before running; (c) no radar exclusion, stated as a limitation.

**Decision (researcher, 2026-10-20): option (b), a tail-focused boundary test. Criteria fixed by
Claude before running** (`scripts/analysis/radar_band_tail.py`):
- *Data:* every second day of the archive (even day index from 2012-09-04), every frame, the
  `clean_frame` field (as the radar pass), contributing geometry (§27). Per radar and frame, in the
  6 km band just inside its owned area and the 6 km band just outside it: valid pixels, pixels
  >= 10 mm/h and >= 31 mm/h.
- *Statistic per radar-year:* R10 = (inside >= 10 / inside valid) / (outside >= 10 / outside valid),
  summed over the year's frames; 95% day-block bootstrap interval of log R10 (B = 500). R31 reported
  too (fewer pixels; descriptive).
- *Flagged radar-year:* R10 >= 2 (heavy rain twice as frequent just inside the radar's boundary as
  just outside it) AND the interval's lower end >= 1.25, with >= 20 days having a pixel >= 10 mm/h in
  either band. An absolute bar, so the test cannot flag radars by construction (unlike the top-5% cut).
- *Why 2:* 12 km apart, real heavy-rain frequency rarely doubles; a radar overestimating heavy rates
  makes it jump at its own boundary. *Limitation:* a boundary along a steep orographic gradient can
  double it too; such cases are checked by name before excluding.
- *Consequence:* flagged radar-years' tile-hours (owner share >= 10%) excluded at the metadata level
  (`filter_meta_v4.py` -> `build_adv_store.py --meta_only`); the excluded share of tile-hours and of
  the tail reported.

*Status (2026-10-20 09:00):* radar pass running (`scripts/hpc/run_band_tail.sh`, 2 workers beside the
v4 build, `logs/band_tail.log` -> `BAND_TAIL_DONE`, ~12-18 h); per-day counts in
`quality_v4/band_tail/days/`, table in `notes/data_analysis_v4/radar_band_tail.md`, flagged list in
`quality_v4/biased_radar_years_tail.csv`. Smoke day 2021-07-14: 156 radars with both bands valid;
inside and outside >= 10 mm/h counts within ~10% on the top radars of a widespread convective day,
as expected without bias. The exclusion path is ready: `filter_meta_v4.py --biased <csv>` (opt-in;
default unchanged) drops tile-hours where a flagged radar-year owns >= 10%, then
`build_adv_store.py --meta_only`. Side effect to report: `--meta_only` re-derives the scaler from the
remaining train maximum.

**Result (2026-10-21; `notes/data_analysis_v4/radar_band_tail.md`).** 2,024 radar-years qualify; median
R10 0.98. **3 radar-years flagged:** ukcyg 2023 (R10 2.51, interval 2.00-2.91), isbjo 2024 (2.39,
1.79-3.59), plpas 2012 (2.17, 1.48-3.57; 23 days, the archive's first months). **None of the tail
suspects is flagged:** robob, hunap, husze, skkoj all have R10 0.78-1.12; the highest suspect is sklaz
2017-18 (1.56-1.63, under the bar). So the heavy-rain excess of the Hungarian / Slovak / Romanian
radars does not jump at their own boundaries: either it is regional climate, or a bias shared by the
whole national network, which only a national-border test would see.
- *Size of the exclusion if applied:* train 4,108 tile-hours (0.31%; 88 with max >= 20.2 mm), val 143,
  test 1,262 (14 >= 20.2 mm), nimbus and events 0.
- *Orographic check by name (pre-stated):* all three are hill or fjord sites by their codes (UK
  Crug-y-Gorllwyn in Wales, Bjolfur in east Iceland, Pastewnik in the Sudetes foothills), so an
  orographic boundary cannot be ruled out by name; plpas 2012 also rests on 23 days.
- *Pending the researcher (2026-10-21):* apply the pre-stated exclusion (0.3% of rows, no effect on
  any suspect) or drop it as orography / small sample.

**Decision (researcher, 2026-10-21): the exclusion is applied as pre-stated.** The three flagged
radar-years' tile-hours (owner share >= 10%) are removed at the metadata level:
`scripts/hpc/apply_radar_exclusion.sh` = `filter_meta_v4.py --biased quality_v4/biased_radar_years_tail.csv`
-> `build_adv_store.py --meta_only` -> post-break lists -> persistence thresholds -> store validation
(`logs/apply_radar_exclusion.log`). Fields and gamma targets are per store row and unchanged.

**Observation (2026-10-21, from the researcher's reading of the raw >= 31 mm/h climatology,
`suspect_radars.pdf` p. 1): Jastrebac (rsjas, Serbia) produced a half-disk of noise in 2023.**
`scripts/analysis/locate_noise_radar.py` -> `notes/data_analysis_v4/noise_radar_2023.{md,png}`: the
2023 anomaly (97,400 km², frequency > 1e-3 and > 5x the 2021-25 median) has a circular western arc
centred 4 km from rsjas (next site 31 km) with a 248 km radius, and a straight eastern edge that is
a diameter through the same point. It is absent in the other years. rsjas is in the node lists
2017-2023 only, hence unlabelled on the 2024 ownership map. Rule 5 excluded 12 of its 2023 radar-days.
Whether the hourly chain removes the half-disk is measured by the raw-vs-corrected climatology
(`scripts/data_quality/rate_climatology.py`, running); any exclusion of rsjas 2023 waits for that
map and is the researcher's decision. The researcher also named hubud and hrdeb as the two worst
radars on that page; the same climatology shows what the chain leaves of them.

### 36c. National-border extension (pre-registered 2026-10-21, before running)

**Why.** §36b compares each radar with its immediate neighbours, which for the suspects are mostly
radars of the same national network. A bias shared by a whole network (one calibration practice,
one Z-R relation, one processing chain) cancels at those boundaries and appears only where the
network meets another. The researcher asked for the test at national borders.

**Criteria (Claude, fixed before running; `scripts/analysis/radar_band_tail.py run_national /
analyze_national`).**
- *Data:* as §36b: every second day, every frame, `clean_frame`, contributing geometry, the yearly
  nearest-radar ownership of `RadarGeometry`.
- *Bands:* a pixel's country is its owning radar's ODIM country prefix (first two letters). For two
  countries A and B whose owned areas touch, the A-side band is A's pixels within 6 km (3 px,
  `BAND_PX`) of a B pixel, and the B-side band likewise. Counts per frame: valid, >= 10, >= 31 mm/h.
- *Statistic per border side and year:* R10(A|B) = (A-side >= 10 / A-side valid) / (B-side >= 10 /
  B-side valid), summed over the year; 95% day-block bootstrap interval of log R10 (B = 500); R31
  descriptive. A border side *qualifies* if both bands have >= 150 px (about 100 km of border) and
  >= 20 days have a pixel >= 10 mm/h in either band.
- *Border side flagged:* R10 >= 2 AND lower bound >= 1.25 (the §36b bar).
- *Country-year flagged:* at least 2 of its qualifying border sides flagged. One border alone can be
  an orographic crest (Carpathians, Alps, Pyrenees) or a coast; a network bias appears on every
  border with an unbiased neighbour. A single flagged border is reported, not acted on; the mean
  DEM height of both bands is reported for every flagged border.
- *Consequence:* a flagged country-year's radar-years (every radar of that country active that year)
  go into the exclusion list, applied as in §36b (owner share >= 10%; same script). The excluded
  share of tile-hours and of tile-hours >= 20.2 mm is reported.
- *Limitations, stated in advance:* (1) neighbouring networks with the same bias cancel at their
  common border (if HU and SK both read high, only their borders with other networks show it);
  (2) a country with one qualifying border cannot be flagged; (3) ownership is nearest-radar within
  250 km, so the "national border" is the composite's network seam, not the political border; over
  sea both sides are sea.

**Result (2026-10-21; `notes/data_analysis_v4/radar_band_national.md`).** 1,034 qualifying border sides
x years; median R10 1.00. 13 border sides flagged, **one country-year: Lithuania 2026** (R10 2.34
against Poland, interval 2.10-2.61; 2.22 against Sweden, 1.65-3.18; not against Latvia) -> ltlau and
ltvil 2026.
- **Hungary, Slovakia, Romania: no network bias.** Median R10 over their qualifying borders 0.99
  (hu, 63 border-years), 1.02 (sk, 44), 1.11 (ro, 6); maximum 1.69 (ro vs hr 2025), 1.67 (sk vs hu
  2018). With §36b this closes the tail suspicion: neither the radars' own boundaries nor the
  national seams show heavy rain jumping where these networks begin. Their high tail is therefore
  attributed to climate or to intermittent artefacts (`suspect_radars.pdf`), not to a consistent
  excess.
- *Single flagged borders, not acted on (as pre-stated):* Belgium vs UK 2018-20 and NL/UK 2016, 2024-25
  (sea seams over the Channel / North Sea), UK vs Ireland 2023, Latvia vs Poland 2016 (20 days), and
  France vs Croatia 2023-25. The last is a geometry artefact: with no Italian radar contributing those
  years, nearest-radar ownership puts a seam between Aleria (Corsica) and Goli (Croatia) over Tuscany,
  at the edge of both ranges. The two-border rule kept it from acting, as intended.
- *Exclusion applied as pre-stated* (`apply_radar_exclusion.sh quality_v4/biased_radar_years_all.csv`,
  = §36b's three + ltlau/ltvil 2026): only the nimbus set has 2026 rows; 1,888 tile-hours removed
  (1.4% of its 2026 rows; 186 with a maximum >= 20.2 mm). Train, validation, test and the scaler are
  unchanged.

**Decision (researcher, 2026-10-20): the pre-break ablation (§34) runs right after the smoke run.**
Lists: `scripts/dataset_v2/postbreak_lists.py` writes `<list>_postbreak.txt` in the adv store's meta
(on the v4 owned lists: train 76.0%, val 86.1%, test 68.3% post-break). Both arms validate and are
evaluated on the post-break val / test lists; matched step budget and seed.
*Amendment (Claude, 2026-10-21, before either arm runs):* both arms validate on the **full** validation
list and are evaluated on the post-break test list. The vanilla v4_adv baseline (`run_vanilla_v4adv.sh`,
same seed and budget) is then the all-data arm, so the ablation costs one extra run, not two. Using the
same validation list for both arms keeps the comparison fair; only the training lists differ.

## 37. The structural-loss weight on hourly data (2026-10-21)

**Question.** `STRUCTURAL_LOSS_WEIGHTS.minkowski = 1e-4` was validated on 15-min rates (active at
+3.8% MAE, below the reward-hacking onset at 1e-3; §2). The hourly corrected target differs in scale,
scaler and threshold grid, so the weight must be re-established (status note, next step 2).

**Method (researcher approved step 1 on 2026-10-21; step 2's criteria are written before it runs).**
- *Step 1, calibration (Claude's rule, fixed before results).* Transfer the validated weight by the
  structural term's share of the gradient, measured with `tools/gradient_audit.py` exactly as the
  15-min audit of 2026-09-23 (512 x 128 patches, anneal 0.1, 10% of the data, jackknife errors):
  - 1a, at initialisation: parity weight lambda* = ||grad MSE|| / ||grad L_mink||; the hourly
    centre is w_init = 1e-4 x lambda*_hourly / lambda*_15min (15-min: lambda* = 1.05e-3);
  - 1b, at the trained vanilla checkpoint, with its own Adam moments: the weight giving the same
    Adam share as 1e-4 at the 15-min vanilla checkpoint (Adam share 16.1); the checkpoint is the
    hourly vanilla baseline (`run_vanilla_v4adv.sh`, starts when the climatology run ends).
  - **1b governs** if 1a and 1b disagree: the Adam share is what shapes the actual update, and the
    15-min weight was validated on trained models. 1a is reported as the cheap first estimate. A
    disagreement of more than 3x is reported as a finding (the share is not stable across training).
- *Step 2, bracketing sweep.* Vanilla plus four weights log-spaced around the step-1 centre
  (w/3, w, 3w, 10w), same seed and reduced budget, judged on validation: active (MAE >= +1% over
  vanilla), not gaming the loss (anisotropy, peak overshoot, images), largest such weight chosen.
  Criteria to be fixed here before it runs.
- *Literature (2026-10-21, web search, not systematic):* gradient balancing is established (GradNorm,
  Chen et al. 2018; Wang, Teng & Perdikaris 2020; Zhong & Maki 2020, closest: weight from a target
  gradient-norm ratio). No prior found for transferring a fixed weight across datasets by gradient
  share, nor for a hacking-bounded weight selection in precipitation SR. **No contribution is claimed
  before a systematic search** (EXPERIMENTS §5 Paper). Notes: `notes/weight_selection_literature.md`.
- *Follow-up kept, not adopted:* setting the weight every step from the gradient share (EXPERIMENTS
  "To explore").

**Result 1a (2026-10-21; `eval_results/gradient_audit/2026-10-21_v4adv_init/`).** At initialisation,
hourly corrected data, same settings as the 15-min audit: parity lambda* = 8.90e-4 (± 3.7e-6),
against 1.05e-3 (± 1.4e-5) on 15-min data. **w_init = 1e-4 x 8.90e-4 / 1.05e-3 = 8.5e-5 (± ~0.1e-5).**
At 1e-4 the share is 0.112 hourly vs 0.095 on 15-min (cos with the MSE gradient +0.86 vs +0.78). So at
initialisation the validated 1e-4 carries over within 15%: hourly aggregation barely changes the
balance of the two terms at the start of training. Step 1b (trained vanilla checkpoint, Adam share)
decides; the vanilla baseline is queued behind the climatology run.

**Raw vs corrected climatology (2026-10-21; `scripts/data_quality/rate_climatology.py`, every 7th day
2016-2025, 520 days, 49,864 frames; `notes/data_analysis_v4/rate_freq{31,1}_raw_vs_corrected.pdf`,
`rate_clim_per_radar.csv`).** The screening plus the radar exclusions remove 31.7% of the 15-min
exceedances >= 31 mm/h, 2.0% of those >= 1 mm/h and 0.60% of valid pixel-frames.
- Spokes and beams are largely removed: hrdeb from 13.9x to 2.6x the median radar's corrected
  frequency (82% of its exceedances removed), rsfrg 37x -> 2.5x (94%), skjav 8.9x -> 3.8x (59%).
- **rsjas 2023 is not handled:** raw 3.85e-3 (other years 1.9e-5 to 1.6e-4), corrected 2.07e-3 (49%
  removed), about 60x the median radar. *Pending the researcher:* exclude rsjas 2023 as a radar-year
  (same path as §36b; years 2017-2023 are in train/val/test).
- hubud is not anomalous over its own area (1.6x raw, 1.3x corrected); the spokes near its label on
  the suspect page belong to skjav.
- Highest after correction: rsjas, robuc (11% removed; straight streaks south of the site remain),
  rocra, hrulj, and the other Romanian radars at 4-5x; the boundary tests found no jump at Romania's
  borders, so part of this is climate.
- **Decision (researcher, 2026-10-21): rsjas 2023 excluded** as a radar-year (owner share >= 10%):
  `quality_v4/biased_radar_years_all.csv` = §36b's three + ltlau/ltvil 2026 + rsjas 2023, applied by
  `scripts/hpc/exclude_then_vanilla.sh` (exclusion, store re-validation, then the baseline). Expected
  from the dry run: 806 train, 107 validation, 312 test tile-hours (24% with a maximum >= 20.2 mm).
- *The vanilla baseline was restarted* (researcher): the run started 14:57 (`logs/unet_ana_20261021_145706.log`,
  4 of 25 epochs, ~45 min each, no NaN) was stopped at 18:00 so that every Study-1 run trains on the
  same final lists. Its run directory is kept but is not a result.

*Note on §37 step 2 (Claude, 2026-10-21, later the same day).* The selection rule for the bracket
already exists: DECISIONS §16 / RESEARCH_NOTES §5 propose the calibrated bracket {λ*/3, λ*, 3λ*, 10λ*}
with one fixed epoch budget, the last checkpoint evaluated, and the point chosen as **the best
exceedance ratio with ΔMAE <= +5% and anisotropy <= 1.2x vanilla**, then 3 seeds at that point and the
whole bracket reported. Step 2 uses that rule (with the activity floor of CLAUDE.md, ΔMAE >= +1%,
as a validity check) unless the researcher changes it; the wording "largest active, non-gaming
weight" above was a loose paraphrase, superseded by this rule.

**Why the screen missed rsjas 2023 (2026-10-21; `scripts/data_quality/why_missed.py`,
`notes/data_analysis_v4/why_missed_rsjas_2023.md`, methods §6.8).** Checked on the two worst days
(10 March, 22 July) and on the radar-day ranking.
- *Rule 5:* no 2023 day reaches score 1 (worst 0.90). The iso31 reference is the median of the five
  nearest contributing radars (rsfrg, hrgra, hrulj, husze, huhar); three of them were noisy on the
  same days, so the median (1,300-6,000 isolated >= 31 mm/h pixels) was within 3-8x of rsjas. The
  ceiling and f150 references include rsjas's own seasonal q99, inflated by its bad weeks. rsfrg and
  hrgra were caught (83 and 77 radar-days); hrulj was not (2).
- *Pixel rules:* ~125,000 of ~200,000 raw pixel-frames >= 31 mm/h per day stay >= 31. The noise is
  clustered (not isolated), moving (not persistent), surrounded by noise (not a burst in quiet
  surroundings) and supported by noise in adjacent frames (not unsupported). Everything left above
  500 mm/h is coded refused (larger than the 500 px repair guard), and refused tile-hours are rejected
  whole; the moderate part (31-150 mm/h) reaches the dataset.
- *Effect before the exclusion:* 1,226 rsjas 2023 tile-hours in the lists, 5.4% with a 15-min rate
  >= 150 mm/h (2021-22: 0.8-1.1%). Excluded now (above).
- **Pending the researcher: hrulj 2023** (same pattern, corrected ~11x the median radar vs 1.7-2.1x in
  2024-25; only 2 radar-days excluded). A future rule-5 revision should not take its reference from
  neighbours that can fail together, nor from the radar's own record (EXPERIMENTS §5).
- **Decision (researcher, 2026-10-21 21:00): hrulj 2023 excluded** as a radar-year, like rsjas 2023
  (`biased_radar_years_all.csv` now 7 radar-years). Removed: 2,666 train, 198 validation, 997 test
  tile-hours. **Final lists: train 1,299,686 / val 172,050 / test 212,803 / nimbus 378,567**;
  scaler unchanged (corrected train max 613.3 mm). The vanilla baseline restarted at 18:11 was stopped
  at 21:06 (3 of 25 epochs, no NaN) and restarts on these lists after the store re-validation
  (`logs/exclude_then_vanilla_hrulj.log`).

**Systematic scan for rsjas-like radar-years (2026-10-21; `notes/data_analysis_v4/rate_clim_per_radar_year.csv`).**
From the raw-vs-corrected chunks: per radar-year, the corrected frequency of 15-min rates >= 31 mm/h
against the radar's other years (x_own) and the median radar-year (x_all, 2.5e-5); 1,545 radar-years.
69 have x_own >= 3 and x_all >= 3, a loose bar (a few storm days in a 1-in-7-day sample can triple a
year). Two discriminators: the share of the year in its top 6-day chunk (all radar-years: median 0.55)
and the share in November-March chunks (median 0.027, 90th percentile 0.36).
- rsjas 2023 (x_own 54, winter 0.56) and hrulj 2023 (5.9, 0.43): already excluded.
- Summer-concentrated, consistent with convective seasons: fikor/fianj 2025, frpla 2021, fipet 2024,
  frtre 2022, deasb 2019, deeis 2022, frcae 2025, eesur 2021, frbou 2020, frnim 2022, plram 2024.
- **Winter-heavy, like rsjas/hrulj, to be looked at (researcher's decision):** mtgud 2017 (Malta;
  x_own 22, winter 0.64; Mediterranean autumn convection is real, so not conclusive), esclg 2025
  (7.6, 0.53), ukham 2025 (6.5, 0.50, peak in early November). isegs 2017 (6.7, 0.27) is borderline.

**Decision (researcher, 2026-10-22): mtgud 2017, esclg 2025 and ukham 2025 are kept** after the review
pages (`notes/data_analysis_v4/radar_year_review.pdf`). Stated as residuals in the methods appendix:
ukham 2025's radial spokes survive the screen; esclg 2025's raw maxima of 1028 mm/h (a capped value not
in the ceiling table) are cut to 109-486 mm/h by the screen; mtgud 2017 was the only radar over Malta
that year. The radar exclusions are final at 7 radar-years; the vanilla baseline is not restarted.

**Decision (researcher, 2026-10-22), §35: the advection-corrected target is accepted on fast-growing
convection**, after the plain-vs-corrected gallery (`notes/data_analysis_v4/gallery_plain_vs_adv.pdf`,
convective and event pages). No growth-aware motion estimate is needed.

**Decisions (researcher, 2026-10-22).**
- *§37 step 2 (weight bracket):* Claude's recommendations accepted. The §16 rule picks the weight
  (best exceedance ratio with ΔMAE <= +5% and anisotropy <= 1.2x vanilla; ΔMAE >= +1% for activity);
  bracket {w*/3, w*, 3w*, 10w*} around the step-1 centre; **full budget** (25 epochs, ~18 h each),
  same seed; **3 seeds** at the chosen weight; the choice is made on **validation**, test is used only
  for reporting; exceedance ratio at the hourly POT level re-derived on the corrected store.
- *§34 amendment (pre-break ablation):* accepted. Both arms select on the full validation list, and the
  **last** checkpoint of each arm is evaluated (not "best"), on the post-break test rows.
- *Launch (researcher, 2026-10-22: no further approval needed):* `scripts/hpc/run_bracket_v4adv.sh`
  (`logs/bracket_v4adv.log`) waits for the vanilla baseline, runs step 1b (audit at
  `UNet_Ana_20261021_211732`, `--which best`, same settings as the 15-min audit), sets
  w* = 1e-4 x 16.127 / (hourly Adam share at 1e-4) rounded to 2 significant figures (written to
  `eval_results/gradient_audit/2026-10-22_v4adv_vanilla/w_star.txt`), then trains the bracket in the order
  w*, 3w*, 10w*, w*/3 on GPU 1, one run after another (two at once would exceed the 8-core budget).

**POT level on the corrected store (2026-10-22; `scripts/data_quality/pot_threshold.py --version v4_adv`,
`notes/data_analysis_v4/pot_threshold_adv.md`, `pot_threshold_adv_fits.csv`).** Same grid, fits and
day-block bootstrap as §32, on the rows of the corrected lists (exclusions applied) with their corrected
tile maxima (0 mismatches against the stored fields).
- **The §32 rule gives no level.** The highest grid point, u = 100 mm (6,898 exceedances on 646 days,
  above the 1,000 minimum), has sigma* = -8.9 [-26, 12.6] and xi = 0.30, outside the intervals of every
  lower u; on the plain sums it was already marginal (it fell just outside at 60 mm).
- Between 31.7 and 80 mm the weighted train fit is stable: xi 0.18-0.21, sigma* 2.9-6.4, all inside each
  other's intervals. Without the 100 mm point the rule would give 31.7 mm; with it, nothing.
- Exceedances (corrected): at 50.1 mm train 99,933 (1,702 days), test unweighted 14,480 (305 days), xi
  0.19 / 0.16; at 31.7 mm train 427,828 (2,366 days), test 77,262 (392 days), xi 0.19 / 0.18.
- *Pending the researcher:* (a) keep u = 50.1 mm, the level pre-stated on the plain sums (§32), which lies
  inside the stable range (Claude's recommendation: no rule changed after seeing the result); or (b) amend
  the rule's minimum count (e.g. >= 10,000 exceedances) so the 100 mm point drops out, giving 31.7 mm, with
  five times more test exceedances, but chosen after seeing the fits.

**Result 1b (2026-10-22; `eval_results/gradient_audit/2026-10-22_v4adv_vanilla/`).** At the trained hourly
vanilla checkpoint (`UNet_Ana_20261021_211732`, best = epoch 24), Minkowski at 1e-4 has an Adam share of
8.83 (15-min vanilla: 16.13) and a raw share of 6.0 (15-min: 24); cos with the MSE gradient -0.14
(15-min: -0.72). **w* = 1e-4 x 16.127 / 8.833 = 1.8e-4**, 2.1x the step-1a estimate (8.5e-5), below the 3x
bar for reporting a disagreement; 1b governs. Bracket (run order): 1.8e-4, 5.4e-4, 1.8e-3, 6.0e-5.
Runs: 1.8e-4 `UNet_Ana_20261022_160540` (done 2026-10-23 12:20), 5.4e-4 `UNet_Ana_20261023_121956`
(done 2026-10-24 08:37), 1.8e-3 running, then 6.0e-5; ~20 h each, no NaN so far.
- **Decision (researcher, 2026-10-24): u = 50.1 mm is kept** for the POT fits on the corrected store
  (option a); the §32 rule is not amended. Bracket evaluation on validation started
  (`scripts/hpc/run_bracket_val_evals.sh`, `eval_results/{backbone,extremes}/v4adv_val/`).

**Bracket result on validation (2026-10-26; last checkpoints, u = 50.1 mm; `notes/data_analysis_v4/bracket_validation.csv`,
images `bracket_gallery.pdf`, 12 random validation patches >= 20.2 mm, all five models).**

| model | ΔMAE | anisotropy (x vanilla) | exceedance ratio | RL bias λ10 / 100 / 1000 | FSS 50.1 mm, 1 / 21 px | peak ratio (extreme) |
|---|---|---|---|---|---|---|
| vanilla | 0 | 2.12 (1.00) | 0.113 | -44% / -31% / -13% | 0.106 / 0.383 | 0.36 (0.20) |
| 6.0e-5 | +4.7% | 0.56 (0.26) | 0.192 | -37% / -41% / -44% | 0.097 / 0.417 | 0.86 (0.52) |
| 1.8e-4 (w*) | +7.9% | 2.24 (1.05) | 0.287 | -32% / -36% / -41% | 0.118 / 0.535 | 0.86 (0.53) |
| 5.4e-4 | +12.5% | 2.02 (0.95) | 0.442 | -25% / -32% / -39% | 0.131 / 0.648 | 0.93 (0.58) |
| 1.8e-3 | +19.0% | 3.45 (1.62) | 0.607 | -18% / -21% / -24% | 0.125 / 0.714 | 0.92 (0.61) |

gpd_xi_obs = 0.183 in every row (sanity check passes). All runs: no NaN, weight as intended in the logs.
- **By the §16 rule the winner is 6.0e-5**, the only run inside the MAE cap (+5%); every stronger weight
  exceeds it (w* costs +7.9%, against +3.8% for 1e-4 on 15-min data).
- *The anisotropy cap does not see what the images show.* Grid-aligned striping and blocky rectangles grow with
  the weight: mild at 1.8e-4, clear at 5.4e-4 (whose mean anisotropy is 0.95x vanilla) and strong at 1.8e-3.
  The mean of per-patch ratios is noisy (targets range 0.08-2.6 per patch) and vanilla itself is 2.12, so a
  1.2x cap on that mean is weak on hourly data. 6.0e-5's low value (0.56) is not a visible diagonal
  artefact in the gallery: its fields look like vanilla's with sharper, higher peaks.
- *The tail is mixed.* More weight means more exceedances above 50.1 mm, higher patch peaks and better FSS at
  21 px, but every Minkowski run under-predicts the long return levels (λ = 100, 1000) more than vanilla
  does: the predicted tail is thinner (fitted xi 0.12-0.17 vs observed 0.18; vanilla 0.32). The 15-min
  finding "helps the tail" holds here for exceedance counts and peaks, not for return levels.
- Next (queued, `scripts/hpc/run_ablation_then_seeds.sh`, `logs/ablation_then_seeds.log`): the pre-break
  ablation's post-break arm first (w = 0, 33 epochs ~ matched steps, seed 42), then seeds 43 and 44 at 6.0e-5.
  The ablation goes first because a decision to drop the pre-break rows would force the seeds to be retrained.
- **Researcher, 2026-10-26: the seed runs at 6.0e-5 are stopped** (not started). The queue shell was killed
  at 08:21; the ablation's post-break arm keeps training (`logs/unet_ana_20261026_081502.log`, done ~2026-10-27
  04:30). The choice of the Study-1 Minkowski weight on hourly data is open again.
- **Researcher, 2026-10-26: the ablation's post-break arm is stopped as well** (killed at ~08:25, during
  epoch 1 of 33; its run directory is kept but is not a result). Nothing is running on GPU 1. The pre-break
  ablation (§34) and the Study-1 Minkowski weight on hourly data are both open.

## 38. Study 2 on hourly data: flow matching and its Minkowski fine-tunes (2026-10-26)

**Question (researcher).** Before choosing the Study-1 weight or running the pre-break ablation: does the
flow-matching (FM) residual model still work on the hourly corrected target, the biggest change of v4, and how
do its two Minkowski fine-tunes behave there?

**Plan (approved by the researcher 2026-10-26).**
- *Backbone:* the hourly vanilla backbone `UNet_Ana_20261021_211732`, **last** checkpoint (epoch 25), the one
  every backbone metric so far comes from; "best" is selected on a dry-pixel-dominated val MSE that is flat
  since epoch 2 (epoch 24 vs 25).
- *Step 0:* sigma_r of that backbone on the train split (`compute_residual_stats.py`), pinned as
  `FM_RESIDUAL_SCALE` in a dedicated config (the corrected store had no residual statistics, and without them the
  code falls back to sigma_r = 1.0); FM_SAMPLER heun, FM_SAMPLE_STEPS 16; fp32 (`--no_amp`); a new launcher that
  takes config and backbone (the old one hardcodes both; it stays as is); a 1-epoch smoke run to measure the
  epoch time and set the budget (target ~4 days).
- *Step 1:* FM training; `fm_best.pth` (selected on the FM validation loss, not circular) is evaluated.
- *Step 2:* fine-tunes from it, each after a coupling check (`diagnose_coupling.py`, minutes): (a) distributional,
  K = 1, M = 4, w = 1e-3, proximal 1, spread 0.5 (as `Extremes_20260804_170256`); (b) energy, M = 16,
  w = 1e-3, retention 1.0 (as `Energy_20260811_171310`). Fine-tunes are evaluated from `fm_mink_latest.pth`.
- *Step 3:* evaluation of vanilla and 6e-5 backbones, FM, FM + Minkowski and FM + energy on one fixed 4,096-patch
  validation subset (16 members, heun-16, u = 50.1 mm), with tail columns, anisotropy, the new striping measure
  and images.
- *In parallel (CPU):* the striping measure (EXPERIMENTS To explore), including diagonal striping.
- *After:* the 2e-5 backbone (bracket extended downward, since the winner sat at its edge), then the Study-1
  weight and the pre-break ablation.
- *Kept for the write-up (researcher):* every bracket artefact and the coupling-check method (EXPERIMENTS Paper).

**Striping measure, validation on the bracket (2026-10-26; `src/evaluation/striping.py`,
`scripts/evaluate/striping_eval.py`, `notes/data_analysis_v4/striping_bracket.csv`; 1,024 validation patches >= 1 mm).**
- *Version 1 fails* (edge orientation of raw log(1+f) gradients; paired spectral anisotropy): no ordering of the
  bracket. The strongest edges are wet/dry boundaries, not the texture inside the rain.
- *Version 2, high-pass texture orientation, passes the ordering:* share of strong texture edges inside rain
  (eroded 3 px) within ±5° of the grid axes, prediction minus target, median over patches: vanilla 0.017,
  6e-5 0.038, 1.8e-4 0.048, 5.4e-4 0.049, 1.8e-3 0.098 (heavy patches: 0.016 / 0.016 / 0.022 / 0.029 / 0.067).
  Monotone in the weight; 1.8e-4 and 5.4e-4 are close. The diagonal version is negative for every model (fewer
  diagonal texture edges than the targets), so no diagonal striping. The spectral-line variant is not usable
  (window leakage dominates; blocks score below an isotropic field). Synthetic checks in the module's tests.
  *Not yet adopted:* its threshold is to be fixed here before it is applied to new runs.

**Why the Minkowski texture has a fixed orientation (researcher's question, 2026-10-26;
`scripts/evaluate/texture_orientation.py`, 64 heavy validation patches).**
- *The estimators break rotation symmetry.* Perimeter (4-direction Cauchy-Crofton, w_ax 0.473, w_di 0.218): an
  ellipse's perimeter is underestimated by 4.8% when its long axis lies on a grid axis and overestimated by
  ~1.5% at 25-35° / 55-65°. Euler (4-connected cubical, V - Ex - Ey + F): a 1-px-wide line has chi = 1 along an
  axis but up to 212 at 45° (it falls apart into corner-touching pixels); no effect from 1.5 px width.
- *The texture is not tied to the input.* Predicting on the input rotated by 90° and rotating back gives a
  texture that correlates with the original prediction at 0.04 (5.4e-4) and 0.11 (1.8e-3).
- *At high weight the texture is locked to the grid.* Texture-edge directions: targets nearly uniform (peak
  share 0.062 per 10°, uniform 0.056); 1.8e-3 peaks at 92.5° with 3x the uniform share, and after the rotation
  test the peak sits at 2.5°, i.e. the same direction in the network's own frame. Lower weights have weaker
  peaks (0.073-0.083); single test on 64 patches, not repeated.

**§38 step 0 (2026-10-26).** sigma_r of the hourly vanilla backbone (last checkpoint) on the full train split
(1,299,686 patches): **0.012668** (15-min vanilla: 0.018981); per-pixel std map CV 0.064, so the scalar is used.
Pinned in `configs/config_v4_adv_fm.yaml` (FM_RESIDUAL_SCALE, BACKBONE_CHECKPOINT = the hourly vanilla last
checkpoint, FM_SAMPLER heun, FM_SAMPLE_STEPS 16). The smoke run's startup line reads
`config FM_RESIDUAL_SCALE=0.01267`. Write-up of the deterministic-model limitation (researcher's request):
`notes/minkowski_deterministic_limitation.md`.

**§38 step 1 launched (2026-10-26 09:26).** FM smoke run (2% data, 1 epoch, `runs/sr_flow_matching/Extremes_20261026_091536`,
not a result): train_fm 0.735, val_fm 0.519, ~1.85 h per full epoch (1.65 steps/s + 7 min validation). Full run:
`configs/config_v4_adv_fm.yaml`, **50 epochs max** (~90 h; early stopping on PATIENCE 5 may end it sooner),
`logs/flow_matching_20261026_092624.log`; startup line `config FM_RESIDUAL_SCALE=0.01267`, amp=False.
Evaluation subsets (fixed, seed 0): `val_subset4096.txt` (random validation rows; 279 with tile max >= 20.2 mm,
38 >= 50.1) for general metrics, and `val_extremes1024.txt` (validation rows with corrected tile max >= 20.2 mm;
187 >= 50.1) for the tail columns, which are therefore conditional on heavy tiles. The existing `extremes_val`
list was selected on plain maxima (corrected minimum 2.8 mm) and is not used for this. Chain
`scripts/hpc/after_fm_v4adv.sh` (`logs/after_fm_v4adv.log`): when FM ends, evaluates vanilla, 6e-5 and FM on both
subsets, then runs the coupling checks for both fine-tunes, and stops.

**§38 step 1 result (2026-10-29).** FM on the hourly corrected target, `runs/sr_flow_matching/Extremes_20261026_092630`
(`logs/flow_matching_20261026_092624.log`): early-stopped at epoch 33 (PATIENCE 5), no NaN; best val_fm **0.2323 at
epoch 28** (`fm_best.pth`, evaluated). Val_fm by epoch: 0.281, 0.264, 0.259, 0.255, 0.252, 0.249, 0.248, 0.246, 0.245,
0.242 (ep 10), 0.240 (ep 12), 0.238 (ep 15), 0.235 (ep 19), 0.233 (ep 23-24), 0.232 (ep 28); train_fm 0.320 -> 0.239.
Comparable in scale with the 15-min FM (0.297 / 0.285 / 0.279 at epochs 1-3; residual normalised by sigma_r in both),
~1.9 h per epoch, 64 h in total. Evaluation chain started 00:28 (`logs/after_fm_v4adv.log`).
