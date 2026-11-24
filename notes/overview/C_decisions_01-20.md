# Digest C: DECISIONS.md §1–§20 (lines 1–750)

Source: `/work/fquareng/ch2/minkowski-loss/DECISIONS.md`, lines 1–750. The file's preamble
(lines 1–8) says the log records *why* results are believed and *which roads are closed*;
`EXPERIMENTS.md` records the numbers. Each entry is meant to give question, evidence,
conclusion, confidence, and what would overturn it.

Conventions in this digest: "Date" is the date stated in the file; "undated" means the entry
gives none (most of §1–§13 carry no date in the heading or body, only later status notes).
Cross-references to §21+ are noted where the §1–§20 text points to them; their content is
outside this digest's scope.

---

## §1. The threshold grid: quantiles → fixed log-spaced physical levels

- **Date:** undated.
- **Question:** The original grid was 21 climatological quantiles topping out at q = 0.99 ≈
  6.5 mm/h, while extremes are ~123 mm/h. How should the grid reach the tail?
- **Decision:** Fixed log-spaced physical thresholds, 0.1 → 150 mm/h, 15 levels, integrated
  over ξ = log(u) so each decade of intensity weighs the same.
- **Evidence:** Extending the quantile grid alone does nothing: the loss integrated `dq`, so a
  threshold in [0.999, 0.9999] carries weight 1e-4 however far the grid is extended; the grid
  and the integration measure must change together. Verified numerically: the deepest
  threshold's weight rises ~400× versus the old measure, and weights are near-equal across
  levels. Fixed physical levels stay comparable across datasets and can be aligned with the
  evaluation levels.
- **Rejected:** (a) extended quantiles with uniform dq — tail still unweighted; (b)
  log-exceedance quantiles ξ = −log(1−q) — equivalent in spirit, implemented then reverted;
  less interpretable and not alignable with evaluation thresholds.
- **Consequence:** Any Minkowski distance or isoperimetric number computed on the old grid is
  not comparable with one on this grid. Regenerate γ targets whenever the grid changes.
- **Confidence:** high — the weighting argument is arithmetic, not empirical.
- **Status:** in force (no status note).
- **Literature:** none.

---

## §2. The structural loss weight is bounded above by reward hacking, not by accuracy

- **Date:** undated.
- **Question:** How large can the Minkowski weight go?
- **Decision:** w = 1e-4 is the working point. Two diagnostics were added to the protocol
  because of this episode: **directional anisotropy** (axis-aligned vs diagonal
  high-wavenumber power) and **peak ratio** (predicted/observed patch maximum). Neither is
  optimised by any loss, so neither can be gamed.
- **Evidence:** At w = 1e-3 the deterministic backbone produces grid-aligned filamentary
  striping absent from the target and overshoots patch maxima by 30–60%, while every
  structural metric improves: SAL S 1.27 → 0.14, isoperimetric violation 4.57 → 3.47%, RAPSD
  3.05 → 0.90. Only the accuracy cost registers it (+14.7% MAE, +17.8% extreme MAE).
- **Why the metrics miss it:** radially averaged spectrum integrates over direction; SAL
  amplitude compares domain means, not peaks; the isoperimetric inequality P² ≥ 4πA is
  satisfied *more* comfortably by high-perimeter filaments.
- **Scientific value:** concrete demonstration that matching Minkowski curves is necessary but
  not sufficient — Hadwiger completeness concerns the basis of valuations, not identifiability
  of the field. "Worth reporting, not hiding."
- **Rejected:** w = 1e-3 (hacking regime).
- **Confidence:** high — visually unambiguous in sample images.
- **Status:** in force (also listed in CLAUDE.md as established).
- **Literature:** none cited (Hadwiger completeness mentioned as a concept, no reference).

---

## §3. The Minkowski loss helps the deterministic backbone and not the generative model

- **Date:** original undated; status note 2026-09-23.
- **Decision (original):** The loss helps the deterministic backbone; it does not help the
  generative (flow-matching) model.
- **Evidence, deterministic** (full test set, w = 1e-4): FSS@89 0.040 → 0.159, exceedance
  ratio 0.118 → 0.244, return-level bias 0.643 → 0.445, RAPSD 3.10 → 1.31, for +3.8% MAE. All
  independent of the training objective.
- **Evidence, generative (original):** two routes, both negative:
  - as a reward (DRaFT-K on the clean endpoint): FSS, ξ error and return-level bias all
    slightly worse than clean flow matching;
  - via the backbone (FM trained on the Minkowski backbone): return-level bias +43%, FSS@89
    roughly halved, exceedance ratio and extreme peak ratio both down; CRPS and RMSE_ext
    unchanged — a regression concentrated in the tail.
- **Explanation:** MSE gives the conditional mean, and `1[E[y|c] ≥ u] ≠ E[1[y ≥ u] | c]`: the
  excursion geometry of the conditional mean is systematically wrong and MSE training cannot
  fix it. A calibrated sampler gets `E[A(u)] = |D|(1−F(u))` for free. So the loss supplies
  information the deterministic model structurally lacks and the generative model already
  has. The clean FM model beat the Minkowski backbone on every tail column.
- **Confidence (original):** high on deterministic gain; medium-high on the generative null
  (two experiments).
- **What would change it:** a much larger tail-sample budget (§6), or a positional term in the
  reward.
- **Status 2026-09-23: reopened for the generative half.** After the 09-16 re-evaluation
  (EXPERIMENTS §2), on the same 4,096 patches:
  - distributional reward (`Extremes_20260804_170256`) now beats clean FM on FSS@89 (0.321 vs
    0.315), exceedance ratio (0.534 vs 0.475) and RL bias (0.114 vs 0.165);
  - energy coupling improves exceedance ratio and RL bias at an FSS cost (0.269);
  - FM on the Minkowski backbone is no longer a tail regression (RL bias 0.141 vs 0.165) but
    still loses FSS (0.273).
  The "via the backbone" evidence (+43% RL bias, FSS halved) was measured before the σ_r
  guard, so the cause of the change must be attributed before either version is quoted.
  Deterministic half unaffected. Treat "does not help flow matching" as **unsettled**, and
  weigh it against the single seed and the leaking split (§15).
- **Literature:** none (DRaFT-K named as a method, no reference).

---

## §4. The coupling is correctly implemented — do not go looking for a bug

- **Date:** undated.
- **Question:** Was the generative null a plumbing failure?
- **Decision:** No; the coupling is correctly wired. K = 1 is fine.
- **Evidence:** Single-batch diagnostic with base noise held fixed: reward falls **59.7%** in
  40 steps, five times its own sampling noise. Gradients agree across truncation depths:
  cos(∇K=1, ∇K=2) = +0.91.
- **Correction recorded:** An earlier run reported cos = −0.20 and "broken coupling". That was
  an artefact of comparing gradients computed on *different* base-noise draws; the reward is
  stochastic, so an un-fixed noise draw makes the test meaningless. The fixed-noise version
  is the valid one.
- **Rejected:** the "broken coupling" reading of the −0.20 result.
- **Confidence:** high. Re-run `scripts/evaluate/diagnose_coupling.py --extreme_batch` if in
  doubt (minutes).
- **Status:** in force.
- **Literature:** none.

---

## §5. The binding limit is pixel mass, and the anneal parameter cannot fix it

- **Date:** undated.
- **Decision:** Keep anneal at 0.05–0.10. **Do not sweep it again.** Do not sweep K either
  (§4). Remaining levers are threshold reweighting and sample selection, not smoothing.
- **Evidence:** Per-threshold gradient decomposition on an extreme-containing batch (max
  142 mm/h):

  | threshold | share of gradient | residual |
  |---|---|---|
  | ≤ 0.28 mm/h | 83% | small |
  | 19 mm/h | 0.13% | 1.23 (largest) |
  | ≥ 31 mm/h | ~0 | 0.8–1.1 |

  The structural error is largest where the gradient is zero. Cause: only 0.005% of pixels
  exceed 31 mm/h and essentially none exceed 53, so the extreme-threshold excursion sets are
  empty in almost every sample.
- **Anneal sweep (0.05 → 0.40):** absolute gradient at 31 mm/h rises by eleven orders of
  magnitude, but the tail's share of total gradient stays at 0.75–0.88% and slightly
  declines, because widening the indicator amplifies the bulk equally — while the bulk
  excursion-set estimate degrades (residual at 0.48 mm/h doubles).
- **Rejected:** larger anneal (smoothing) as a route to the tail; sweeping K.
- **Confidence:** high — direct measurements.
- **Status:** in force (cited by §6, §9, §20).
- **Literature:** none.

---

## §6. Scale is the principled route to the tail

- **Date:** undated.
- **Decision/argument:** The targeted quantities are ensemble expectations — area is the
  marginal survival function; β₀ at high u approximates the field-maximum exceedance
  probability. Expectations can be estimated accurately from many sparse samples even when
  each sample carries almost no tail information. This is the argument for a larger
  (compute) allocation, and it follows from §5.
- **What scale will not fix:**
  - **FSS.** The functionals are rigid-motion invariant and carry no positional information.
    Every coupling that improved distributional metrics cost FSS. More samples improve the
    tail distribution, not placement.
  - **Identifiability.** Enforcing observable statistics yields distributionally calibrated
    but spatially unidentified extremes (Chang & Sapsis: their 20-member ensemble placed the
    inferred extreme with spatial sd 3.99 against a true maximiser norm of 2.86).
- **Rejected:** claiming scale fixes FSS or identifiability.
- **Status:** in force (no status note).
- **Literature:** Chang & Sapsis (2026), *Extreme Event Aware Learning*,
  doi:10.1038/s41467-026-76811-x.

---

## §7. The energy score is the right reward form, and it is proper

- **Date:** undated; superseded-numbers note 2026-09-23.
- **Question:** The per-sample reward regresses an ensemble mean onto a *single* observed γ —
  one draw from p(γ|c). Can that be fixed?
- **Decision:** Use the energy score (kernel CRPS) on the functional curves, Minkowski
  distance as the metric, fair M(M−1) normalisation:
  `S = (1/M) Σ_m d(γ̂_m, γ) − 1/(2M(M−1)) Σ_{m≠m'} d(γ̂_m, γ̂_m')`.
  The negative pairwise term is itself the spread reward, so `REWARD_SPREAD_WEIGHT` is
  ignored in this mode.
- **Evidence:** Synthetic test: minimised at the true ensemble dispersion, while per-sample
  matching on the same ensembles prefers zero spread — the diversity-collapse mechanism
  demonstrated.
- **Result (original):** best coupling so far: best ξ error in the table (0.043), highest
  exceedance ratio (0.523), uniform return-level under-prediction (−24%/−18%/−11%) instead of
  the earlier sign-crossing bias. Tail class right, scale ~15% short. Still does not beat
  clean FM on FSS.
- **Superseded numbers (2026-09-23):** after the 09-16 re-eval the energy row's ξ error is
  0.067 (not the table's best) and RL bias 0.126. It still holds the best exceedance ratio
  (0.538) and RAPSD (0.55) in the FM block. See §3.
- **Also added:** `REWARD_FM_RETENTION_WEIGHT`. The fine-tuning objective previously had no
  pixel-aligned term at all (rigid-motion-invariant reward plus velocity-space anchor), so
  nothing supplied placement — a plausible cause of the FSS regression in both routes.
  Untested above 1.0; FSS still fell at 1.0.
- **Rejected:** per-sample reward matching (prefers zero spread).
- **Status:** decision in force; original numbers superseded by the 2026-09-23 note.
- **Literature:** none.

---

## §8. Competing losses are all inactive or harmful (Study 1)

- **Date:** original undated; status note 2026-09-23.
- **Setup:** Four alternatives implemented and trained: spectral (FFT-magnitude energy
  score), SSIM, soft wet-area, optical-flow advection consistency.
- **Round 1 (shared weight):** all inert — MAE cost only +0.6–0.8% vs vanilla (Minkowski at
  working weight pays +3.8%). RAPSD got worse under a loss optimising the spectrum directly —
  signature of a weight too small to enter the objective.
- **Round 2 (per-loss weights):** still inert by MAE check (+0.3–0.7%); effects negative:
  exceedance ratio 0.036–0.068 vs vanilla 0.118; RL bias 0.69–0.75 vs 0.643; anisotropy
  2.1–2.6 vs 1.87. None improves on plain MSE.
- **Mechanisms (for the paper):**
  - wet area thresholds at drizzle and compares binary masks, so it is intensity-blind (0.2
    and 200 mm/h identical). Degenerate single-threshold case of the Minkowski area channel;
    fails hardest (RAPSD 4.59, worse than bicubic) — evidence the multi-threshold structure
    does the work;
  - optical flow uses the coarse conditioning field as teacher, smooth with suppressed peaks
    by construction — teaches the model to resemble its input;
  - spectral is translation-invariant, provably cannot penalise displacement; scores exactly
    0.000 on a displaced field.
- **Open:** whether a genuinely active weight exists for any of them. Use the gradient-norm
  rule (§9), not value parity; require the MAE check before reading structural columns.
- **Status 2026-09-23: not a fair trial, so "harmful" is not established.** Audit (EXPERIMENTS
  §3):
  - Under-weighted and inactive: spectral, SSIM, optical flow trained at about 10⁻³, 10⁻⁴ and
    4·10⁻⁵ of gradient parity at initialisation. At their checkpoints share 0.02 / 0.16 /
    0.007 vs 1.22 for Minkowski. Trained models no better than vanilla on their own loss
    (+2.5%, +0.1%, −2.6% n.s.). Measured with `tools/gradient_audit.py` over 512 batches; see
    `notes/gradient_audit.pdf`. An earlier 6-batch estimate ("70×/10×/140×") is superseded.
  - SSIM is buggy: perfect prediction scores 0.465 vs 0.499 for all-zero.
  - Unmatched budgets: all v2 runs 25 epochs vs 47 (Minkowski) and 69 (vanilla). A 34-epoch
    vanilla run reproduces their loss values.
  So worse-than-MSE tails = under-training plus inert term. The mechanisms remain valid *a
  priori* arguments, not backed by an active run. Wet area is the only active one (12.8× the
  MSE gradient) and is degenerate. Rerun protocol: §16.
- **Status:** title claim revised — "harmful" not established; superseded by §16 protocol.
- **Literature:** none.

---

## §9. Set auxiliary weights by gradient norm, not loss value

- **Date:** undated; correction 2026-09-23.
- **Decision:** Loss values span three orders of magnitude, but value parity is the wrong
  target: Minkowski's gradient is threshold-localised (83% below 0.28 mm/h), SSIM and spectral
  gradients are dense. Use λ = ‖∇L_ERM‖₂ / ‖∇L_aux‖₂, computed after ERM pre-training (Chang &
  Sapsis 2026), as the *centre* of a sweep, then move down — gradient parity would likely put
  Minkowski into the §2 hacking regime.
- **Rejected:** loss-value parity.
- **Correction (2026-09-23):** measured at the ERM solution the rule is ill-posed: ‖∇L_ERM‖ ≈ 0
  by definition, and a per-batch norm measures minibatch noise, not signal
  (`notes/gradient_audit.pdf` §4–5; a 6-batch check at the vanilla checkpoint found the MSE
  signal indistinguishable from zero). Measure parity at initialisation or early training,
  Euclidean and Adam-preconditioned, with `tools/gradient_audit.py`. At a trained checkpoint
  use the equilibrium test: share → 1 and cos → −1 for a run trained with the term.
- **Status:** principle in force; "after ERM pre-training" revised by the 2026-09-23
  correction; operationalised in §16.
- **Literature:** Chang & Sapsis 2026 (for the λ rule).

---

## §10. Evaluation protocol decisions

- **Date:** undated.
- **ξ error is not a headline metric.** Shape is not identifiable independently of scale.
  Three runs have the best ξ error alongside the worst tail: wet area 0.007, SSIM 0.009,
  spectral 0.013 — all producing 3–6% of observed exceedances. A deficit in σ̂ compensates an
  error in ξ̂. Report exceedance-count ratio and return-level bias; keep ξ as a diagnostic
  read with σ and exceedance count.
- **One POT level per table.** Refitting the same MSE backbone at u = 53 instead of 31 moves ξ
  from 0.03 to 0.33 and RL bias from 0.64 to 0.43. Sanity check: `gpd_xi_obs` depends only on
  data, so must be identical in every summary at the same level (u=31 → +0.176; u=53 →
  −0.190). `eval_losses.sh` warns automatically if rows mix.
- **u = 31 chosen for estimability, not physics.** Extremes ~123 mm/h, asymptotic regime not
  established at 31; it is the level where every model yields enough exceedances to fit. To be
  stated in methods.
- **Return levels parameterised by expected exceedance count λ, not raw m.** The level exists
  only when m·rate > 1; pixel-level rate ~1e-6, so fixed m values return NaN. Given λ,
  m = λ/rate_obs, same m for both fits.
- **Evaluate `fm_mink_latest.pth`, not `fm_mink_best.pth`.** "Best" is selected on validation
  reward (the training objective), so with tail overshoot it picks the most overshot
  checkpoint — circular selection.
- **RMSE_ext is a weak discriminator:** varies ~1% across samplers differing 46% in peak
  height.
- **Rejected:** ξ as headline; mixed POT levels; fixed-m return levels; best-on-reward
  checkpoint selection; reliance on RMSE_ext.
- **Status:** in force (CLAUDE.md notes hourly v4 tables need a re-derived single level,
  §21 — outside this scope).
- **Literature:** none.

---

## §11. The cheap sampler is a different regime, not a cheap proxy

- **Date:** undated.
- **Decision:** Train and evaluate with heun-16. Fund it by reducing the per-step batch, not
  the ensemble: M is what the E[γ] estimate needs.
- **Evidence** (same clean checkpoint, identical patches):

  | | peak ratio | FSS@89 | RAPSD | RL bias |
  |---|---|---|---|---|
  | heun-16 | 0.816 | 0.280 | 0.738 | 0.165 |
  | euler-16 | 0.589 (−28%) | 0.200 (−29%) | 1.477 | 0.341 |
  | euler-8 | 0.441 (−46%) | 0.146 (−48%) | 2.158 | 0.429 |

  Euler-16 loses 28% of median peak height and 41% on extreme patches. ξ error is nearly
  sampler-invariant (0.072–0.083) while amplitude-related quantities collapse — ξ is close to
  scale-free, another reason not to headline it.
- **Rejected:** Euler samplers as cheap proxies; shrinking ensemble size M to save compute.
- **Status:** in force.
- **Literature:** none.

---

## §12. Numerical and infrastructure decisions

- **Date:** undated.
- **σ_r changes with the backbone.** Recompute after changing backbones; pin
  `FM_RESIDUAL_SCALE` explicitly per model, not via `residual_stats.json` (overwritten by the
  last `compute_residual_stats` run). A silent mismatch once produced a 53× reconstruction
  error that looked like catastrophic model failure. Verify the startup line.
- **fp16 overflows** once the residual is correctly scaled — train FM with `--no_amp` (fp32)
  or bf16.
- **`CUDA_VISIBLE_DEVICES` alone is not enough:** CUDA device order differs from
  `nvidia-smi`'s, and `micromamba run` overwrites the variable set outside it. Set
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` too, inside the wrapper. (Note: CLAUDE.md says that as of
  2026-09-23 `micromamba run` passes them through unchanged; setting inside remains the safe
  habit.)
- **Launch long runs under `setsid`/Slurm** — tmux exit leaves DataLoader workers unreaped;
  node once accumulated ~193 zombies with stuck root NFS threads.
- **`val_mse` is dry-pixel-dominated** and saturates by epoch 2; never judge a run by it.
- **Status:** in force.
- **Literature:** none.

---

## §13. Related work positioning

- **Date:** undated.
- **Paper:** Chang & Sapsis (2026), *Extreme Event Aware (η-) Learning*,
  doi:10.1038/s41467-026-76811-x. "Closely related and worth citing carefully."
- **Exact identity for the theory section:** since A(u) = |D|(1−F(u)),
  ∫|A_pred(u) − A_targ(u)| du = |D| · W₁ between the marginal intensity laws, exactly. Our
  loss uses log(1+A) and dξ = du/u, so it is a **reweighted** W₁. This allows citing their
  Theorem 4 (data-consistent estimators provably miss the tail — the MSE backbone at
  604/35,482 exceedances is that lower bound) and Theorem 7 (W₁ is tight for tail error).
- **Where we are ahead:** they name vector observables as future work; our γ is a 3×Q vector
  observable with per-channel extreme-value interpretations. We have a demonstrated
  degenerate-solution failure mode (§2) and a diagnostic; a scalar max is harder to game.
- **Where they are ahead:** their gradient flows through the argmax pixel, which always
  exists, so they never hit the §5 bottleneck. Their tail-sample selection is directly
  adoptable.
- **Framing caution:** their regime is *no* extreme data; ours is extremes-present but
  gradient-starved. The remedy transfers; the framing should not be borrowed wholesale. In a
  compute proposal, invert it: statistical regularisation gives calibrated but unidentified
  extremes and depends on a reference law whose misspecification propagates (their RMSE
  +20.6% under tail perturbation).
- **Caution on tail-sample selection:** if adopted, apply tilted sampling to the structural
  term only. Keep MSE / FM velocity loss on uniformly drawn batches, otherwise μ is no longer
  E[y|c] and the residual decomposition breaks.
- **Status:** in force (positioning note).
- **Literature:** Chang & Sapsis (2026), doi:10.1038/s41467-026-76811-x (Theorems 4 and 7).
  Note the title is written two ways in the file: *Extreme Event Aware Learning* (§6) and
  *Extreme Event Aware (η-) Learning* (§13).

---

## §14. DDPM is dropped; the emulator loss is lowest priority

- **Date:** 2026-09-23.
- **Decision:** Flow matching replaces DDPM as the generative model; it is the Study-2 model
  everywhere. DDPM is no longer trained, compared or reported. Code and `runs/sr_ddpm/` are
  kept, not deleted. The learned (emulator-based) Minkowski loss stays in the codebase at
  lowest priority; worth resuming eventually as the learned counterpart to the analytical
  loss, after the data rebuild and Study-1 rerun.
- **Evidence:** none stated in the entry.
- **Rejected:** continuing DDPM.
- **Status:** in force.
- **Literature:** none.

---

## §15. The current splits are random at patch level, so test is not independent of train

- **Date:** 2026-09-23 (`scripts/data_quality/audit_splits.py`).
- **Finding:** `split_metadata.py` shuffles individual patches over the whole period, so
  train, val and test all span 2023-08-01 → 2024-10-30 on the same 79 tile locations. For
  100% of test patches (tail included): a train patch exists at the same timestamp, and a
  train patch exists on the same tile within ±1 h. For 98–99%, a train patch exists on an
  adjacent tile at the same timestamp. Test holds neighbouring 15-min frames and tiles of
  storms seen in training.
- **Consequence:** every current score measures interpolation within seen events —
  legitimate for an in-distribution table if stated, but cannot support a generalisation
  claim or say anything about unseen extremes (the central claim). It flatters every model
  roughly equally, so Study-1 relative ranking is not obviously wrong; absolute numbers and
  gap sizes may be.
- **Decision:** The rebuilt dataset gets splits blocked in time by event (whole days or storm
  episodes, gap of at least a few hours between splits). The o.o.d. split (tab:ood) builds on
  that. Existing patch-level splits stay only to reproduce current tables.
- **Rejected:** patch-level random splits for new work.
- **Confidence:** high — direct count.
- **Status:** in force (implemented as v2 week-blocked splits per CLAUDE.md).
- **Literature:** none.

---

## §16. A fair trial for auxiliary losses: gradient-norm bracket, matched budget, seeds

- **Date:** proposed decision undated ("to be confirmed when M4 starts"); chosen centres
  2026-09-23.
- **Question:** How to set λ so Study 1 compares losses, not weights? Candidates:
  homoscedastic (Kendall) uncertainty weighting, Optuna search, calibrated grid.
- **Rejected — homoscedastic weighting:** (i) it balances loss values (learns 1/σ² from each
  term's residual scale), which §9 rejects; (ii) its Gaussian-likelihood argument does not fit
  these losses — none is a Gaussian NLL, several have non-zero floors (SSIM's 0.465), so a
  learned σ can absorb the floor; (iii) it turns the comparison into one of learned,
  uncontrolled per-loss weights. Useful for balancing the three Minkowski channels (already
  used so in the emulator), not MSE vs auxiliary.
- **Rejected — Optuna as primary tool:** λ is one dimension per loss and bounded above by
  hacking (§2), so a 3–4 value bracket covers it; each trial is a full run; the selection
  metric decides the outcome and must not be any loss's own objective. Optuna only for
  multi-dimensional searches (λ with anneal or tail-sample fraction), on short proxy runs with
  pruning and equal trial budget per loss.
- **Decision (proposed), per loss:**
  1. Measure parity λ* = ‖∇L_MSE‖/‖∇L_aux‖ at initialisation or early training, not at the
     ERM checkpoint (§9 correction; `tools/gradient_audit.py`). EXPERIMENTS §3 values are
     noise-referenced first estimates.
  2. Train fixed log bracket {λ*/3, λ*, 3λ*, 10λ*}. Minkowski at its working point is ~5λ*,
     so the bracket must reach above parity.
  3. One fixed epoch budget for every run, vanilla included; evaluate the **last**
     checkpoint, no early stopping or "best" selection. Current `val_monitor` adds
     λ·val_aux (`unet_analytical.py:244`), selecting on the auxiliary objective; the
     alternative `val_mse` saturates by epoch 2 (§12).
  4. Pick each loss's point by one pre-registered rule: best exceedance ratio subject to
     ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla. Then 3 seeds at that point.
  5. Report the whole bracket, so "cannot be made active without hacking" is visible.
  For FM, the same losses enter as rewards through `reward_finetune.py`, same bracket logic on
  `REWARD_WEIGHT`.
- **Chosen centres (2026-09-23):** now defaults in `config.yaml` `STRUCTURAL_LOSS_WEIGHTS` and
  `src/losses/competing.py`. Rule: give each term the same share of the actual AdamW update
  that Minkowski has at its validated 1e-4, measured at the vanilla checkpoint with that
  checkpoint's Adam second moments (`tools/gradient_audit.py`, 512 batches).

  | loss | old (v2) | new | Adam-matched value | init share at new λ |
  |---|---|---|---|---|
  | minkowski | 1e-4 | 1e-4 | (reference) | 0.095 |
  | spectral | 5e-4 | 1e-1 | 0.116 | 0.39 |
  | ssim | 1.6e-4 | 2e-2 | 0.0196 | 0.02 |
  | wetarea | 2e-2 | 2e-2 | 0.0198 | 0.99 |
  | opticalflow | 4e-5 | 2.5e-2 | 0.0239 | 0.02 |

  These are bracket centres, not tuned optima; the M4 bracket must test ×1/3 … ×3, judged by
  equilibrium and own-loss tests at end of training. SSIM's weight belongs to the current
  floor-dominated implementation; re-measure after the data-range/eps fix.
- **Resolution order in trainer:** `--weight_geom` > `STRUCTURAL_LOSS_WEIGHTS[loss]` >
  `MINKOWSKI_TARGET_WEIGHT` (Minkowski only) > built-in defaults. Launcher default is now
  `config`; it was `0.001` (the hacking weight).
- **Note:** the proposed bracket in step 2 ({λ*/3 … 10λ*} around gradient parity) and the
  chosen-centres paragraph (Adam-update-share matching, bracket ×1/3 … ×3) use different
  reference points; the file does not explicitly reconcile them.
- **Status:** proposed, to be confirmed at M4; centres adopted as config defaults.
- **Literature:** Kendall weighting named, no citation.

---

## §17. Screening policy: reject clear errors only, keep imperfections, no labelled set

- **Date:** 2026-09-28 (researcher decision).
- **Decision:**
  - Discard freely, but only clear errors. Archive is large (~13 years of 15-min composites
    over Europe, 2012-09 → 2025-10), so losing patches costs little; meaningless
    (non-meteorological by construction) data must not enter training or evaluation.
  - Keep imperfections: blockage and range underestimation, attenuation shadows, compositing
    seams, drizzle speckle, mild residual clutter. The model should be robust to them and
    only learns it by seeing them.
  - No labelled set: labelling ~300 tail tiles to tune thresholds is not worth it. Supersedes
    steps 6–7 of the `scripts/data_quality/README.md` loop.
- **Clear errors** (RESEARCH_NOTES §7.3): residual static clutter; emitter/RLAN rays and
  rings; isolated spikes far above neighbourhood; physically impossible rates (observed:
  13,072 and 64,842 mm/h); radar-wide constant-signal failures; single-frame appearances with
  no precursor/successor; rain under clear sky (if satellite added); nodata-dominated tiles.
- **Risk:** without labels, precision/recall on real extremes is never measured. Real extremes
  are also "anomalous" (steep gradients, small intense cores, fast evolution), so a rule
  firing on unusual intensity structure removes real storms preferentially and biases the
  tail. EURADCLIM's authors say their cleaned OPERA product's outliers limit its use
  "especially for use in extreme value modeling".
- **Safeguards (stand-ins for labels):** (1) prefer rules keyed on geometry, persistence and
  physical impossibility over intensity structure; (2) report rejection rate per intensity
  bin, region and year for every rule — steep rise with intensity is suspect until rejected
  tiles are inspected; (3) look before trusting: spot-check galleries of each rule's
  rejections and the unflagged tail (`patch_gallery.py`), minutes per rule; (4) cross-check
  against independent evidence where available (satellite cloud type, as EURADCLIM does;
  lightning; severe-weather reports); (5) same screen for training and evaluation, including
  held-out o.o.d. extremes (M2); (6) measure robustness — report skill on kept-but-imperfect
  subsets (far range, behind blockage) separately.
- **Open (to settle with MCH — Daniele Nerini, Lionel Moret; RESEARCH_NOTES §7.4):** concrete
  rule set and thresholds, tile rejection vs pixel masking, plausibility bound on rate, and
  replacement for declutter-zeroing above 150 mm/h (EXPERIMENTS §5).
- **Rejected:** a perfectly clean dataset as goal; a labelled calibration set.
- **Confidence:** policy is a researcher decision; risk analysis high-confidence (bias
  direction follows from what the rules detect).
- **Status:** in force; safeguard 3's "look by eye" partly replaced later for rule 5 (§19
  status note, §24) and §20 criterion 2 (§24).
- **Literature:** EURADCLIM (authors, cleaned OPERA product) — quoted, no author/year/identifier
  given in the file.

---

## §18. ODYSSEY and NIMBUS: train on one product, test the other separately

- **Date:** finding 2026-09-30; status note 2026-11-05 (researcher with Tom).
- **Finding:** OPERA changed production chain on 2024-07-05: archive product becomes `RATE`
  (NIMBUS) instead of `QIND_RATE` (ODYSSEY). Same grid, 2 km, projection, 15-min cadence.
  Differences: ODYSSEY composites the scans in a 15-min window, NIMBUS uses the lowest-
  elevation PPI at the nominal time; different radar set. On days in both products, NIMBUS has
  ~30–40% less wet area, intensities 8–20% higher in median (wide spread), log-space
  correlation with ODYSSEY only 0.65–0.80, about half the +15 min persistence. All archive days
  from 2024-07-05 are NIMBUS (~475 days on disk). Quantification:
  `scripts/data_quality/era_gap.py`, EXPERIMENTS §5.
- **Decision (researcher):** v2 train/val/test drawn from ODYSSEY days only. NIMBUS days form
  a separate `nimbus` split (with `events_nimbus`, `light_nimbus`, `extremes_nimbus`),
  buffered like any split.
  - Why: mixing would put ~10% NIMBUS into test; an ODYSSEY-trained model would score worse
    there for reasons of product, not extremeness — the confound o.o.d. claims must rule out.
    Kept separate, NIMBUS is a clean product-shift test.
  - Cost: ~10% of data; flagship events (Storm Boris, Valencia DANA) become shifted-product
    cases.
  - Reversible: era stored per patch (`era` in `full_*_info.csv.gz`); split is metadata only
    (`--nimbus_start ''` disables it).
- **Rejected:** mixing products across splits.
- **Status note (2026-11-05):** extremes evaluated within ODYSSEY only for now. Do not
  evaluate or report NIMBUS extremes (`extremes_nimbus`, `events_nimbus`, `nimbus` split):
  models never trained on NIMBUS, so a score mixes product shift with extremeness, and the
  problem (reaching the right intensities and structure in heavy rain) does not need the
  product-shift test. The R4 cut ladder (§43) is consistent: its held-out above-cut tile-hours
  are ODYSSEY (new intensity, same product). The NIMBUS split stays on disk; the test is
  deferred, not dropped.
- **Status:** in force; NIMBUS evaluation deferred (2026-11-05).
- **Literature:** none.

---

## §19. v4 screen: radar ceilings, radar-wide failures, and a guarded repair

- **Date:** 2026-10-06 (findings and decisions); rule 2 change 2026-10-06; rule 5 addition
  2026-10-07; status notes 2026-10-08 (two).
- **Findings** (POT threshold study on v3; EXPERIMENTS §5, RESEARCH_NOTES §7.4d): three
  failures every v3 rule misses, because each rule assumes a small/local/transient artefact:
  - reflectivity ceiling at 364.63 mm/h (exactly 64.0 dBZ under Z = 200 R^1.6) in the Valjevo
    (Serbia) area on 80 rain days of 2023; 12% of all test pixels > 89 mm/h;
  - radar-wide failure of Torrejón de Velasco (Madrid) on 2018-04-29; 16% of all train pixels
    > 89 mm/h. The v3 footprint repair lowered it to its ring median (the failing radar too),
    leaving a flat 88.27 mm/h plateau (4,865 pixels in one tile);
  - ceiling at 48.62 mm/h (exactly 50.0 dBZ), tile r640 c1408, 7 days of Nov 2019, found by
    the repeated-value test unprompted.
  "Radar-wide constant-signal failures" were on the §17 list but had no rule.
- **Decisions (researcher, 2026-10-06, on Claude's proposal):** five rules, all keyed on an
  error signature (§17 safeguard 1):
  1. repeated value per tile — on the 0.01 mm/h ODYSSEY grid real rain almost never repeats
     one value (per tail tile: median 1, 99.9th pct 24 repeats of one value ≥ 31); ≥ 50
     repeats rejects the tile. On NIMBUS (ladder ratio 1.0593) count taken in excess of
     adjacent levels;
  2. ceiling table per radar area and year — values counted > ~50× the median of their
     ±0.5 dB neighbours; a ceiling pixel is censored, not measured;
  3. radar-disk failure per radar and frame (wet fraction, mean, share of log-rate variance
     explained by range alone, jump at area boundary and max-range circle);
  4. guarded repair — `boundary_fill` refuses components above a size limit; repeated-value
     test runs after every repair;
  5. radar-day ranking against each radar's own seasonal level and its 5 nearest radars.
  - Ceiling pixels: tile-frame with ≥ 5 rejected; 1–4 replaced by median of valid
    8-neighbours (like a spike), and coded.
  - Radar-day exclusions from a researcher-reviewed list (safeguard 3), versioned in
    `configs/quality_v4.yaml`.
  - Scope: only these five rules; static coverage holes (option B) and masked partial tiles
    (option C, `notes/events.md`) excluded so v3 → v4 difference is attributable.
  - Decisions taken at split time from per-tile columns (`make_splits.py`), so thresholds
    change without rescan (roadmap M1).
- **Two corrections to first proposal:** (a) "cross-check against neighbouring radars in the
  overlap zone" impossible — composite holds one value per pixel, no per-radar field; becomes
  the jump across the area boundary and max-range circle; (b) "do not repair when ring median
  ≥ 31 mm/h" would refuse almost every repair (a footprint is the ≥ 150 mm/h region, so its
  ring sits just below 150 even in real storms); guard is a size limit calibrated on v3's
  repairs plus post-repair repeated-value test.
- **Tail fit is an audit, not a rule:** it found the failures but cannot tell failure from a
  real extreme day, is blind below u, and would make the screen circular (§20). Runs after
  every build; never decides what is kept.
- **Confidence:** three failures verified by eye and exact dBZ values; that rules catch them
  without removing real extremes is to be shown by Phase-0 calibration and §20 gate.
- **Rule 2 changed (2026-10-06, after first Phase-0 run):** first run found 3,256 radar-year
  "ceilings" (up to 43 per radar-year, 2,044 in 2024 alone). Cause: coarse value ladders (dBZ
  steps 0.4–3 dB, rounded to 0.01 or 0.1 mm/h) mixed with sparse off-ladder values (counts
  1–4), so the ±0.5 dB median is ~1 and every ladder level passes 50× (e.g. Málaga 2024,
  10.28 mm/h counted 8,059 times among neighbours counted 1–4). On event days scanned it
  rejected 16.5% of ce_floods_201305 tiles (72% of those with max ≥ 31) and 2.5% of
  andreas_20130728.
  - New comparison: count vs the **largest** count of any other value within **±4 dB** (wider
    than the coarsest ladder step, ~3 dB, IT 2026). On a ladder ratio ~1 or below; a ceiling
    piles up above.
  - Calibration (199 days, 23,868 values counted ≥ 50): the 18 radar-year rows of the two
    known ceilings score 9.9–89; all others 99.9th pct 1.96. Threshold `min_ratio: 5`,
    provisional until rerun Phase 0. The original 50 had no calibration.
  - Not explained: 11.53 mm/h (= 40.0 dBZ) in 2023 scores 3.0–13.7 on the same Balkan radars
    carrying the 364.63 ceiling (itbri 13.7, rsfrg 5.1, grthe 5.1); flagged at 5, sent to
    gallery; may be a second fault or a ladder artefact.
  - Rule 1 unchanged (±0.5 dB median, `excess`); whether mixed ladders inflate it is read from
    `rej_repeat` in Phase-0 reports.
- **Rule 5 gains a fourth signal, isolated high pixels (2026-10-07):** the 48.62 tile keeps
  maxima of 138–342 mm/h made of 2–9 pixel clusters in dry surroundings; no rule sees them
  (spike rule fails: a pixel with any neighbour ≥ 10% of its value is not a spike).
  - Not a tile rule: a max with a dry 5×5 window occurs on 9–16% of kept event tiles ≥ 31 mm/h
    (EXPERIMENTS §5), so any tile threshold removes real storms.
  - Radar-day signal `iso31` = owned pixels ≥ 31 mm/h whose 5×5 window has median < 0.1 mm/h
    and is ≥ 50% covered, summed over the day's frames, on the `clean_frame` field. Scored
    like other signals (own seasonal q99, 5 nearest radars, floor 10 pixels). Only enters the
    review list. Counts, not shares (share's 99th pct 0.98 on random days with ≥ 50 pixels
    ≥ 31 — noise).
  - Phase-0 check: Oradea's failure days have 1,286–8,636 isolated pixels vs neighbour median
    0–3, but score < 0 on 6 of 7 days because the calibration set holds 7 failure days among
    Oradea's 53 autumn days, so its own q99 (7,525) is a failure day. On the full archive they
    are < 1% of ~1,180 autumn days. Event radar-days: 232 of 15,897 score > 0, none ≥ 0.5.
    Rule-3 flags and f150 unchanged.
  - Open: a radar failing on more than ~1% of its season-days hides behind its own q99 for
    every signal; options own q90 for `iso31`, or neighbours only.
- **Status 2026-10-08:** with hourly targets (§21) all five rules under review;
  RESEARCH_NOTES §7.4e proposes changes to rules 1–3 and ray, hot-pixel and ring repairs, and
  adds persistence signals; rule 5 unchanged. **Later 2026-10-08:** rule 5 no longer uses a
  reviewed list; radar-day and radar-year exclusions come from a pre-stated score (§24).
- **Status:** partly revised (by §21 review and §24).
- **Literature:** none (Z = 200 R^1.6 relation stated without citation).

---

## §20. When the dataset is ready for training: a gate fixed before the results

- **Date:** 2026-10-06 (researcher); status notes 2026-10-08 and 2026-10-19.
- **Question:** Without labels, cleanliness cannot be measured, and §17 keeps imperfections.
  When to stop cleaning and train?
- **Decision:** Ready when **one more cleaning step would not change any number we report**,
  not when clean. The bar is set by evaluation, not training: 83% of the structural gradient
  comes from ≤ 0.28 mm/h and ≥ 31 mm/h contributes ~0 (§5), so a backbone barely sees the top
  of the tail (inference, not tested), whereas tail columns (exceedance ratio, RL bias, FSS
  at 89, xi) are made of it — in v3 one ceiling was 12% of test pixels > 89 mm/h. Training-side
  exception: tail-sample selection by patch max picks artefacts first.
- **The gate** (all five, on v4, before step 3 of the training plan):
  1. Known failures caught — regression catalogue `configs/quality_v4.yaml` `known_failures`,
     absent from every build.
  2. Residual contamination measured — 100 random test tiles with max ≥ 89 mm/h (not
     top-ranked), inspected by eye; 0 artefacts → 95% upper bound ~3% (3/n); else binomial
     interval. Estimates, tunes nothing (so no conflict with §17). Pass: upper bound ≤ 5%.
  3. Evaluation reference stable — remove the 20 non-event days with largest share of
     exceedances; recompute observation side (exceedance counts at 31/53/89 mm/h,
     truncated-GPD xi, observed return levels). Pass: every change inside its day-block
     bootstrap interval. Most weight.
  4. Real extremes survive — event tiles kept (rejection < 0.5% per event), no rule's
     rejection rate rising steeply with intensity, radar/gauge exceedance ratio in Germany and
     Switzerland no worse than v3's.
  5. Residual drift in xi(u) explained by a named legitimate cause (truncation at 500 mm/h,
     regional/seasonal mixing, sub-asymptotic threshold), with truncated-GPD fit.
- **Where we stop:** if gate passes, v4 is the training set and the rest is accepted
  imperfection. A v5 only if criterion 2 or 3 fails, and only for a nameable, visible failure
  type. If the gate fails and v5 isn't worth it, train on v4 and report the measured
  contamination rate as a limitation.
- **No cleaning by tail fit:** removing whatever makes xi(u) drift would select data by GPD fit
  and then use the GPD as evaluation reference — circular, and removes real extremes that
  don't fit (rare regimes, mixtures). Tail fit may point at candidates; rules must key on an
  error signature (ceiling value, range-only field, repeated value).
- **Truncation:** v3/v4 hold no values > 500 mm/h (> 500 cores repaired), so data are
  truncated, and truncation alone makes xi fall at high u (synthetic GPD, xi = 0.35: +0.33 at
  31, +0.24 at 89, −0.32 at 250 after removing 0.08% of values). The audit fits a truncated
  GPD (density divided by its mass below 500). At u = 31 effect small (0.35 → 0.33). Whether
  the evaluation protocol should also fit truncated is open.
- **Rejected:** "clean" as stopping criterion; cleaning by tail fit.
- **Confidence:** gate is a researcher decision; circularity argument derivable; training
  insensitivity to tail artefacts an untested inference from §5.
- **Status notes:**
  - 2026-10-08: criterion 2 (100 tiles by eye) replaced by gauge-based tail false-alarm and
    miss rates (§24).
  - 2026-10-19: criterion 3 replaced (§32) by a contamination test of the most influential
    test days, plus day-block intervals and a leave-period-out variant reported with every
    tail statistic; the original rule failed by construction on heavy-tailed daily
    contributions.
- **Status:** framework in force; criteria 2 and 3 superseded by §24 and §32.
- **Literature:** none.

---

## Cross-cutting notes

- **Only literature cited in §1–§20:** Chang & Sapsis (2026), doi:10.1038/s41467-026-76811-x
  (§6, §9, §13; titled *Extreme Event Aware Learning* in §6 and *Extreme Event Aware (η-)
  Learning* in §13), and an unreferenced quote attributed to "EURADCLIM's authors" (§17).
  Methods named without citation: DRaFT-K (§3), Kendall homoscedastic weighting and Optuna
  (§16), Hadwiger completeness (§2).
- **Status map:** in force — §1, §2, §4, §5, §6, §10, §11, §12, §13, §14, §15, §17; reopened —
  §3 (generative half, 2026-09-23); numbers superseded — §7 (2026-09-23); claim weakened to
  "not a fair trial" — §8 (2026-09-23), protocol in §16; corrected — §9 (2026-09-23); proposed
  pending M4 — §16; deferred evaluation — §18 (NIMBUS, 2026-11-05); partly revised — §19 (by
  §21 review and §24); criteria 2 and 3 replaced — §20 (by §24, §32).
- **Forward references out of scope:** §21 (hourly target), §24 (pre-stated score, gauge-based
  rates), §32 (contamination test), §43 (R4 cut ladder).
