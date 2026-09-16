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

---

## 9. Set auxiliary weights by gradient norm, not loss value

Loss *values* span three orders of magnitude across these losses, but value parity is the
wrong target: the Minkowski gradient is threshold-localised (83% below 0.28 mm/h) while SSIM
and spectral gradients are dense over every pixel. Equal value ≠ equal influence.

Use λ = ‖∇L_ERM‖₂ / ‖∇L_aux‖₂, computed after ERM pre-training (Chang & Sapsis 2026). It
matches the quantity that actually drives the update. Treat it as the *centre* of a sweep,
then move down — gradient parity would likely put Minkowski into the hacking regime of §2.

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
