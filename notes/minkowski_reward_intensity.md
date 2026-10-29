# Making the Minkowski reward reach heavy rain: quantile area + mass weights

Decision note (researcher's decision, 2026-10-29; DECISIONS §39). Long-form explanation with derivations:
`notes/minkowski_reward_coupling.pdf`. Status labels: *derivation* (follows from the definitions),
*measured* (run here, with the file), *inference* (our reading, not tested directly).

## Why this step matters

The Minkowski loss is meant to fix the tail: how much heavy rain there is and what shape it has. Until now its
gradient has not reached the tail at all. On the clean hourly FM, 73% of the reward's gradient lands on pixels
below 1 mm on the heavy subset, and 3.7% on pixels above 20 mm (*measured*, `logs/reward_threshold_share_v4adv.txt`).
On the 15-min data the same limit was measured as "83% of the gradient from thresholds <= 0.28 mm/h" (DECISIONS §5).
Every Minkowski result so far was obtained with a loss that, in practice, shaped light rain and the edges of wet
areas. Any claim that it helps or does not help the extremes was a claim about a loss that could not see them.

## The cause (derivation)

At threshold u the loss uses a soft mask s_i = sigmoid((f_i - u)/tau), tau = 0.005u at anneal 0.05. A pixel's
sensitivity to that threshold is sigmoid'((f_i - u)/tau)/tau, which is ~exp(-|f_i - u|/tau)/tau.

1. *Pixel mass.* Summed over pixels, sum_i sigmoid'((f_i - u)/tau)/tau -> N p(u) as tau -> 0, where p is the
   density of pixel values: a threshold's total pull equals the number of pixels near it, whatever tau is. This is
   why the anneal sweep (DECISIONS §5) changed the absolute gradient by 11 orders of magnitude but not the share.
2. *Out of reach.* A threshold above the prediction's maximum has no pixel near it. Example: a 17 mm core and
   u = 20.2 mm (tau = 0.10 mm) give (f - u)/tau = -32 and a sensitivity ~1e-13. The terms where the error is
   largest are exactly the ones that give no gradient.

## The decision

Keep K = 1 and both reward modes (distributional, energy). Change what the reward compares (`REWARD_FORM:
quantile_mass`, `src/trainers/reward_finetune.py:reward_features`; the original stays as `REWARD_FORM: threshold`,
the default):

- **Area, compared horizontally.** For each threshold u_j the target has n_j = #{y >= u_j} pixels. Compare the
  n_j-th largest predicted value with the n_j-th largest target value, as log1p (mm). This reads the same survival
  curve along the intensity axis instead of the count axis. In model space z (f = expm1(c z), c = 6.42) the term is
  c |z_(n_j) - z*_(n_j)|: one pixel per threshold per member gets a gradient of size c, the same for every threshold
  and whatever the pixel's value (*derivation*). Thresholds above the target maximum are masked.
- **Perimeter and Euler, weighted by inverse pixel mass.** w_j = (n_j + 1)^-alpha, normalised to mean 1 over the
  thresholds, alpha = 1; thresholds above the target maximum take the weight of the highest populated one. For a
  locally power-law survival curve S(v) ~ v^-k, the pixels per unit log-intensity are k S(v), proportional to n_j,
  so alpha = 1 equalises the thresholds (*derivation under that assumption*); alpha = 1 was also the most even of
  0 / 0.5 / 1 in the measurement below.
- Ranks and weights depend on the target only, so for each condition the reward compares fixed feature vectors
  with a fixed weighted L1 distance; the energy score stays proper on those features (*derivation*; strict
  propriety is not established for an L1 distance).

Alternatives considered: inverse-mass weights alone (fixes 1, not 2); a hard forward pass with a heavy-tailed
surrogate gradient at high thresholds (fixes 2 with a tuned kernel width, and a surrogate gradient can point where
the true objective does not, the start of gaming). The horizontal form removes cause 2 exactly for the area channel
with no new hyperparameter.

## Checks (all at the clean FM `Extremes_20261026_092630/fm_best.pth`, hourly vanilla backbone)

- *Unit tests* `tests/test_reward_form.py` (13 pass; the file fails to import on HEAD): on a 17 mm core under an
  80 mm target, the threshold form gives the core no gradient and the quantile_mass form pulls it up, in all three
  modes; unit weights reproduce the original reward exactly; the rank feature, the masking and the weights behave
  as specified and depend on the target only. The 5 failures in `tests/test_losses.py` are pre-existing (they fail on HEAD too).
- *Gradient by pixel intensity* (`scripts/evaluate/reward_threshold_share.py`, energy mode, M = 4, 4 batches of 8),
  share of |dR/dz| in % for <1 / 1-10 / 10-20 / 20-50 / >=50 mm:

  | subset | original | quantile_mass, alpha = 1 |
  |---|---|---|
  | heavy (`val_extremes1024`) | 73.4 / 19.9 / 3.0 / 3.0 / 0.7 | 15.9 / 27.0 / 16.2 / 31.5 / 9.4 |
  | random (`val_subset4096`) | 89.1 / 9.9 / 0.8 / 0.2 / 0.0 | 60.7 / 31.5 / 6.4 / 1.4 / 0.0 |

  alpha = 0 (quantile area alone) changes almost nothing: the area channel carried only 11% of the gradient; the
  Euler channel at 0.1 mm carried 17-28% on its own.
- *Coupling checks* (`diagnose_coupling.py --extreme_batch`, fixed noise, 40 reward-only steps):
  distributional (M = 4) 0.232 -> 0.158 (-31.6%), noise floor 2 sd = 0.108, 4/40 steps clipped;
  energy (M = 16) 0.131 -> 0.085 (-35.1%), floor 0.024, 0/40 clipped. Original form for comparison (2026-10-29): distributional -28.1% (floor 0.071),
  energy -32.5% (floor 0.027).
- *Gradient split per term* (`scripts/evaluate/grad_norm_terms.py`): the reward's gradient norm is 5.0e-4 in both modes (was 8.3e-4).
  dist: 96% of the total. **energy: 0.07% of the total** (cos -0.03): with retention weight 1 the energy fine-tune
  would still be continued FM training. Rebalance w_reward against w_retain (gradient-norm rule, DECISIONS §9;
  roughly x1000) before any energy fine-tune with this form.

## Costs and risks

- *Sparse area gradient.* The area term moves one pixel per populated threshold per member per step (at most 15).
  Overall the field gradient was already concentrated in both forms (*measured*, TEST 2: 100% of its mass on the
  top 1% of pixels; pixels with any gradient 15% -> 12%), so the change in sparsity is small.
- *Noisier reward.* Weighting rare thresholds raises the reward's sampling noise (distributional: 11.4% -> 18.9%
  of the mean). Changes during a fine-tune will be harder to see on 20 validation batches.
- *Artefacts get more weight.* The top populated threshold can be a handful of pixels, and the top of the
  distribution contains radar artefacts (CLAUDE.md). Look at the images of anything this form produces.
- *Not yet a result.* No fine-tune has run with this form. Before one: fix the proximal anchor (it is evaluated off
  the model's paths) and run the w_reward = 0 control, both in EXPERIMENTS §5.
- `diagnose_coupling.py` TEST 4 still prints the threshold form's per-threshold gradient; use
  `reward_threshold_share.py` for the new form.
