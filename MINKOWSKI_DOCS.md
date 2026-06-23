# Differentiable Minkowski functionals — design, conventions, and changelog

Scope: the analytical, differentiable Minkowski-functional loss used as a structural
loss for 12.5× super-resolution downscaling of OPERA radar precipitation. This document
records every design decision in the current implementation, the reasoning and
alternatives behind each, the unit/normalisation conventions at every interface, the
target/loss consistency requirements, and the validation protocol. It is intended to be
sufficient to reproduce and defend the method in a scientific report.

Project reference: F. Quarenghi, R. Cotsakis, T. Beucler, *Emulating Non-Differentiable
Metrics via Knowledge-Guided Learning: Introducing the Minkowski Image Loss*,
arXiv:2604.11422 (cs.LG). The present work replaces the **learned emulator** of that paper
with **analytical, differentiable** functionals.

Files: `src/losses/minkowski.py`, `src/data/gamma.py`, `src/data/datasets.py`,
`config.yaml`, plus `validate_phase1.py` and `tests/test_analytical_minkowski.py`.

---

## 1. Quantities

For a precipitation field `x` (mm h⁻¹) and a set of `Q = 21` intensity
thresholds `u⁽¹⁾ < … < u⁽Q⁾`, the excursion set at threshold `u` is `{x ≥ u}`. Per threshold we
compute three Minkowski functionals (Hadwiger's theorem: in 2D every motion-invariant,
additive, continuous valuation is a linear combination of these three):

- **area** `A(u) = a_px · |{x ≥ u}|`, with `a_px = (2 km)² = 4 km²`;
- **perimeter** `P(u)` (km), boundary length of the excursion set;
- **topology** `T(u)`, either the Euler characteristic `χ = β₀ − β₁` or the
  connected-component count `β₀`, selected by `TOPOLOGY_MODE` (Section 5).

These are stacked over thresholds into the γ-vector `γ = [A, P, T]`. The Minkowski image
loss is the L¹ distance between log-transformed prediction and target γ-curves,
integrated over thresholds:

```
M(γ̂, γ) = Σ_{c∈{A,P,T}} ∫ | log γ̂_c(q) − log γ_c(q) | dq
```

The log is `log1p` for the non-negative channels (A, P, β₀) and the **signed** transform
`signed_log1p(z) = sign(z)·log1p(|z|)` for χ (which can be negative). Reference for the
functionals and integral geometry: Schneider & Weil, *Stochastic and Integral Geometry*,
Springer 2008.

---

## 2. Units and normalisation (interface audit)

This is the part most likely to cause silent error, so it is stated explicitly.

| interface | space | notes |
|---|---|---|
| model output (regression / flow) | log-normalised `[0,1]` | `x_phys = relu(expm1(x_norm · max_val))` |
| model output (diffusion) | `[−1,1]` | `x_norm = (x_diff + 1)/2`, then as above |
| **functionals + targets** | **physical mm h⁻¹** | excursion sets and `gamma.py` targets both live here |
| thresholds `u⁽ⁱ⁾` | physical mm h⁻¹ | climatological quantiles of the wet-pixel CDF |
| integration variable | quantile level `q ∈ [0,1]` | see Decision D1 |

**Rule.** The functionals must be evaluated on `WaterScaler.to_physical_torch(model_output)`,
never on the normalised or diffusion-domain field. `to_physical_torch` is differentiable
(`relu ∘ expm1 ∘ clamp`), so the loss back-propagates to the model. Confidence 0.95.

Operative constants (`config.yaml`): `PATCH_SIZE=128`, `DOWNSCALING_FACTOR=12.5`
(25 km → 2 km), `PIXEL_SIZE_KM=2.0`, `DRIZZLE_THRESHOLD=0.1 mm h⁻¹`,
`QUANTILE_LEVELS` = 21 levels 0.01…0.99.

---

## 3. Design decisions

Each decision lists: what, why, alternatives, confidence. Items marked **[changed]** are
recent and differ from the original emulator-era implementation.

### D1 — Integration over the quantile axis **[changed]**

`M` is integrated with the trapezoidal rule using `quantile_levels ∈ [0,1]` as the
variable (`torch.trapezoid(abs_diff, q, dim=2)`), **not** the physical thresholds.

Why: integrating over physical thresholds weights each threshold by its spacing `Δu`,
which for a heavy-tailed precipitation CDF is large in the extreme tail — an implicit,
unintended up-weighting of the highest intensities. Integrating over the quantile axis
gives each climatological quantile band equal weight, decoupling "where to threshold"
(physical) from "how to weight thresholds" (quantile). The physical thresholds are still
used to build the excursion sets, because the field is physical.

Alternatives: (i) physical-threshold integration (previous behaviour; tail-biased);
(ii) uniform weights (special case of quantile integration with equal spacing);
(iii) learned per-threshold weights (homoscedastic uncertainty, available via
`HomoscedasticMinkowskiLoss`; adds parameters). Chosen: quantile axis. Confidence 0.9.

### D2 — Excursion sets in physical space with physical thresholds **[changed, bug fix]**

The soft excursion sets use the **physical climatological thresholds** `u⁽ⁱ⁾` (mm h⁻¹),
loaded via `load_physical_thresholds(config)`.

Why: the previous loss was constructed with `QUANTILE_LEVELS` passed as the thresholds, so
it thresholded the mm h⁻¹ field at 0.01…0.99 *mm h⁻¹* instead of at the climatological
intensities. Prediction and target γ were therefore computed at different thresholds — a
latent inconsistency. Confidence 0.9.

### D3 — Soft excursion sets and temperature

`s⁽ⁱ⁾ = σ((x − u⁽ⁱ⁾)/τ⁽ⁱ⁾)`, a smooth indicator in (0,1). The temperature is
**proportional**, `τ⁽ⁱ⁾ = max(u⁽ⁱ⁾ · tau_factor, tau_min)` with `tau_factor=0.1`,
`tau_min=1e-3`, giving a constant *relative* band width across the dynamic range, and is
annealed (sharpened) during training by `anneal_factor`. The sigmoid gradient
`σ'/τ` is supported on the near-threshold band — the correct localisation. Confidence 0.85.

### D4 — Area: straight-through estimator **[changed]**

Forward value is the exact discrete count; backward gradient is the soft one:

```python
hard   = (x >= u).float()           # exact indicator
s_area = hard + (s - s.detach())    # forward = hard; backward grad = d/dx sigmoid
area   = a_px * s_area.sum()
```

How it works (autograd algebra of `detach`): in the forward pass `s − s.detach() = 0`, so
`s_area = hard` and the reported area is the **unbiased** hard count
`A = a_px · Σ 1[x ≥ u]`. In the backward pass `hard` is piecewise-constant (zero gradient
a.e.) and `s.detach()` has no gradient, so `∂s_area/∂x = ∂s/∂x = σ'((x−u)/τ)/τ`. Hence
`∂A/∂x_j = a_px · σ'((x_j−u)/τ)/τ`, localised to the near-threshold band.

Why: the plain soft sum `Σσ` is a *biased* area estimator (sigmoid tails add fractional
mass on both sides of the boundary; the bias grows with τ and with boundary length). The
straight-through estimator removes that bias from the optimised quantity while keeping a
usable, localised gradient. Reference: Bengio, Léonard & Courville, *Estimating or
Propagating Gradients Through Stochastic Neurons for Conditional Computation*,
arXiv:1308.3432 (2013). Confidence 0.9.

**Caveat.** Forward and backward correspond to different maps, so a finite-difference
gradient check on the straight-through area is meaningless (FD probes the piecewise-constant
forward map). Run gradient checks with `area_mode="soft"`. Verified in this work: STE
forward equals the hard count exactly; backward gradient is non-zero; the soft variant
agrees with autodiff to 5×10⁻⁷ (FD).

### D5 — Perimeter: isotropic Cauchy–Crofton **[changed]**

`P = ℓ · ( w_ax·(T_h + T_v) + w_di·(T_d1 + T_d2) )`, where `T_·` are the summed absolute
transitions of the (soft) mask along the two axial and two diagonal pixel-neighbour
directions, `ℓ = 2 km`, and `(w_ax, w_di) = (0.473215, 0.217716)`. This is the
integral-geometric Cauchy–Crofton estimate: perimeter as a weighted count of boundary
crossings by families of test lines — unbiased and rotation-invariant.

Why: the previous estimator was the central-difference total variation
`Σ‖∇s‖₂`, which is (a) anisotropic — it measures boundary length closer to the Manhattan
metric and overestimates oblique edges — and (b) zero-valued at the ridge of a
one-pixel-wide feature (central differences vanish there), so thin precipitation filaments
were under-perimetered. The Crofton estimator is isotropic; verified here to within ~2% of
the continuous perimeter on disks (radii 20–78 px), an ellipse, and a 45° diamond, and to
agree with `skimage.measure.perimeter_crofton(directions=4)` to ~1%.

**Target/loss consistency (mandatory).** The offline target perimeter
(`gamma._compute_perimeter_crofton`) uses the **identical estimator and weights**, not
`skimage.perimeter_crofton`. The two Crofton variants differ ~1%; using different ones
biases the loss fixed point by that amount. With matched weights, the loss minimum sits at
the true field in the τ→0 limit (verified: 208.82 vs hard 208.83 km on a test field). The
previous target used marching-squares (`skimage.find_contours`), which overestimates
curved boundaries by ~5% and is therefore *not* a valid target for a Crofton loss.

Weights were calibrated empirically in this repository against the continuous perimeter
(not taken from a paper). Integral-geometric foundation: Schneider & Weil (2008). Canonical
digital implementation: `skimage.measure.perimeter_crofton` (scikit-image; van der Walt et
al., *PeerJ* 2014, doi:10.7717/peerj.453). N-dimensional run-length Crofton: Lehmann &
Legland, *Insight Journal* (2012). Confidence 0.85.

### D6 — Topology: two selectable modes **[changed: β₀ added]**

`TOPOLOGY_MODE` selects the topology channel; the loss, the offline target, and the dataset
selector must all agree.

#### Mode `"euler"` — analytical Euler characteristic χ (dense gradients)

`χ = V − E_x − E_y + F` on the cubical complex, replacing hard occupancy by the soft mask
and boolean AND by the Gödel (min) t-norm:

```
V  = Σ s
E_ = Σ min(s_neighbour pair)        (axial edges)
F  = Σ min(s over each 2×2 face)
χ  = V − E_x − E_y + F
```

This is local, cheap, and **densely** differentiable (every near-threshold pixel
contributes). Its hard limit equals the 4-connected Euler characteristic.

**Correct target (this is the substantive fix).** The matching offline target is the exact
4-connected Euler number, `gamma.compute_euler_characteristic_exact` =
`skimage.measure.euler_number(mask, connectivity=1)` per threshold — **not** the GUDHI
`β₀ − β₁`. The GUDHI cubical complex treats a pixel as a 2-cell, i.e. **8-connected**
foreground, *and* its β₀/β₁ are **persistence-filtered** (denoised). Both differences make
GUDHI `β₀ − β₁` a different functional from the local χ: on a random field the 4-connected
χ and the 8-connected Euler number are 250 vs −66. Matching an undenoised 4-connected local
χ to a denoised 8-connected target is precisely the apples-to-oranges comparison behind the
paper's low topology R². Verified here: the soft χ → exact `euler_number(connectivity=1)`
as τ→0, exactly, on disk/annulus/two-disks/ring/random fields. Confidence 0.9.

**What χ means.** χ = β₀ − β₁ (components minus holes). In hole-rich fields it is
β₁-dominated and is **not** the connected-component count — use mode `"b0"` if components
are the scientific target (next).

#### Mode `"b0"` — connected-component count β₀ via differentiable persistent homology (sparse gradients)

β₀ is a *global* quantity and is **not** computable by the local alternating sum: no local
formula counts connected components. The principled differentiable tool is persistent
homology. We compute the dim-0 superlevel-set persistence pairs with GUDHI, extract the
birth/death **critical pixels** (`CubicalComplex.cofaces_of_persistence_pairs`), and wrap
them in a custom `torch.autograd.Function` (`_Dim0PersistencePairs`). Each birth/death value
equals the field at one critical pixel, so its gradient w.r.t. the field is a one-hot at
that pixel. The soft count at threshold `u` is

```
β̂₀(u) = Σ_k  g_k · σ((b_k − u)/τ) · σ((u − d_k)/τ)
g_k    = σ((p_k − thresh_b0)/τ_p),     p_k = b_k − d_k     (persistence)
```

where `(b_k, d_k)` are the (birth, death) of component `k`, the essential component (global
max) gets a finite sentinel death so its death-gate saturates to 1, and the persistence gate
`g_k` reproduces the **same persistence-noise floor** `thresh_b0` used by the offline target
`gamma._count_b0` — so target and surrogate are the same quantity. As τ→0 the soft count
equals the exact persistence-filtered β₀.

Properties (verified here):
- forward matches `gamma._count_b0`; soft → hard as τ→0 (e.g. [3,2,1,1] across thresholds);
- gradient is **exact** (finite-difference agreement 1.6×10⁻⁹ on a smooth field) but
  **sparse** — supported only on the O(#components) critical pixels;
- an annulus gives β₀ = 1 (one component), where χ = 0 — the practically relevant distinction
  for hole-rich fields.

Reference: Carrière, Chazal, Glisse, Ike, Kannan & Umeda, *Optimizing persistent homology
based functions*, ICML 2021, PMLR 139:1294–1303, arXiv:2010.08356 — differentiability of
persistence-based functions and the critical-simplex gradient. GUDHI: Maria, Boissonnat,
Glisse & Yvinec, *The GUDHI Library*, ICMS 2014, doi:10.1007/978-3-662-44199-2_28.
Persistence background: Edelsbrunner & Harer, *Computational Topology: An Introduction*,
AMS 2010. Confidence 0.85.

#### Which mode for which dataset

Recommendation for a **hole-rich** field where the scientific quantity of interest is the
number of rain cells: use `"b0"`. χ would be dominated by the hole count and would not
report component structure. Trade-off to weigh empirically (which is why both are provided):

| | gradient | cost | correct β₀? | use when |
|---|---|---|---|---|
| `euler` (χ) | dense, cheap, local | low | no (χ = β₀−β₁) | topology = Euler char.; few holes |
| `b0` (PH) | sparse (critical pixels) | per-image CPU PH | yes | hole-rich; want component count |

Mitigation for β₀ gradient sparsity: the area and perimeter channels are always dense, so
the *total* loss gradient remains dense (verified: full-field gradient with `b0`), and the
sparse β₀ term refines connectivity on top of that. Confidence 0.8.

---

## 4. Target ↔ loss consistency

The loss has an unbiased fixed point only if each offline target is computed with the **same
estimator** as the loss. Current pairings:

| channel | loss estimator | offline target (`gamma.py`) |
|---|---|---|
| area | soft/STE count × `a_px` | `_compute_area` (hard count × `a_px`) |
| perimeter | soft Crofton, weights (0.473215, 0.217716) | `_compute_perimeter_crofton`, same weights |
| χ (`euler`) | soft `V−E_x−E_y+F`, Gödel-min | `compute_euler_characteristic_exact` (`connectivity=1`) |
| β₀ (`b0`) | soft persistence count, floor `thresh_b0` | `_count_b0`, same floor |

`compute_gamma_matrix` now returns a **(5, Q)** matrix `[A, P_crofton, B0, B1, chi_exact]`;
`select_topology_target` / `datasets._gamma_to_log` route `euler → chi_exact` (channel 4)
and `b0 → B0` (channel 2). **Offline targets must be regenerated** with the new `gamma.py`:
the perimeter row is now Crofton (was marching squares) and the euler row is exact χ (was
absent). Validating the new loss against stale targets shows spuriously low R² for P and the
topology channel because those are genuinely different quantities.

---

## 5. Configuration and wiring

`config.yaml`:
```yaml
TOPOLOGY_MODE: "euler"   # or "b0"
```

Trainer (`src/trainers/unet_analytical.py`) loss construction:
```python
from src.utils import load_physical_thresholds, load_persistence_thresholds

topology_mode = config.get("TOPOLOGY_MODE", "euler")
thresh_b0 = load_persistence_thresholds(config)[0] if topology_mode == "b0" else 0.0
geom_fn = AnalyticalMinkowskiLoss(
    physical_thresholds=load_physical_thresholds(config),  # mm/h, excursion sets
    quantile_levels=config["QUANTILE_LEVELS"],             # [0,1], integration axis
    pixel_size_km=config.get("PIXEL_SIZE_KM", 2.0),
    topology_mode=topology_mode,
    area_mode="ste",
    persistence_thresh_b0=thresh_b0,
).to(device)
```
The predicted field must be converted with `to_physical_torch` before the loss. The
`anneal_factor` (sigmoid sharpening) is passed per epoch as already done in the trainer.

---

## 6. Validation protocol

Run on the cluster after regenerating targets (`validate_phase1.py`):
1. **Per-functional R²** of `Â, P̂, T̂` vs the offline targets on a held-out batch
   (expect R²_A ≈ 0.99; high for P and the topology channel given consistent estimators).
2. **Gradient correctness**: finite-difference vs autodiff on the *soft* functionals
   (`area_mode="soft"`); must agree (verified here to 5×10⁻⁷ for A+P+χ, 1.6×10⁻⁹ for β₀).
3. **Gradient localisation** ρ: fraction of `|∂T/∂x|` mass within ±3τ of a threshold;
   should concentrate there.
4. **Unit tests** (`tests/test_analytical_minkowski.py`): disk → χ=1/β₀=1; two disks →
   χ=2/β₀=2; annulus → χ=0 but β₀=1; soft → hard as τ→0; STE forward = hard count; FD
   gradient agreement. All pass in-silico here (12/12).

Verified in this environment (torch 2.12 CPU, scikit-image 0.26, GUDHI 3.12): everything
above except the data-dependent R² (no cluster data here).

---

## 7. Known limitations and numerical pitfalls

- **β₀ gradient is sparse** (critical pixels only) and **PH is per-image CPU** — the `b0`
  path is markedly slower than `euler`. In the Phase-3 reward-fine-tuning setting, evaluate
  it only on the clean endpoint at low noise and consider sub-sampling the batch.
- **χ cancellation in low precision.** `V − E_x − E_y + F` is a large-cancellation sum; run
  the geometric loss in fp32, not under fp16 autocast. Confidence 0.6.
- **STE finite-difference checks are invalid** — use `area_mode="soft"` for gradient checks
  (Section D4).
- **χ ≠ β₀.** Do not interpret `euler`-mode topology as a component count in hole-rich
  fields; use `b0`.
- **Targets must be regenerated** after these changes (Section 4).

---

## 8. Changelog (recent changes highlighted)

- **[changed] Integration axis** → quantile levels (D1).
- **[fixed] Excursion-set thresholds** → physical climatological thresholds (D2).
- **[changed] Area** → straight-through estimator: hard count forward, soft gradient backward (D4).
- **[changed] Perimeter** → isotropic Cauchy–Crofton; **offline target switched to the same
  Crofton estimator** to keep the loss fixed point unbiased (D5).
- **[fixed] Euler target** → exact 4-connected `euler_number(connectivity=1)` instead of the
  8-connected, persistence-filtered GUDHI `β₀ − β₁` (D6, `euler`).
- **[added] `b0` mode** → differentiable persistent-homology β₀ for connected-component
  counting in hole-rich fields, persistence-gated to match the offline target (D6, `b0`).
- **[changed] `compute_gamma_matrix`** → returns (5, Q) `[A, P_crofton, B0, B1, chi_exact]`.
- **[changed] `config.yaml`** → `TOPOLOGY_MODE: "euler"`.

---

## 9. References

1. F. Quarenghi, R. Cotsakis, T. Beucler. *Emulating Non-Differentiable Metrics via
   Knowledge-Guided Learning: Introducing the Minkowski Image Loss*. arXiv:2604.11422 (2026).
   — The project this loss belongs to; defines the γ-vector and Minkowski image loss.
2. M. Carrière, F. Chazal, M. Glisse, Y. Ike, H. Kannan, Y. Umeda. *Optimizing persistent
   homology based functions*. ICML 2021, PMLR 139:1294–1303. arXiv:2010.08356.
   — Differentiability and the critical-simplex gradient underpinning the `b0` mode.
3. Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through
   Stochastic Neurons for Conditional Computation*. arXiv:1308.3432 (2013).
   — Straight-through estimator used for the unbiased-area gradient (D4).
4. R. Schneider, W. Weil. *Stochastic and Integral Geometry*. Springer, 2008.
   — Integral-geometric foundation of the Minkowski functionals and the Crofton perimeter.
5. C. Maria, J.-D. Boissonnat, M. Glisse, M. Yvinec. *The GUDHI Library: Simplicial
   Complexes and Persistent Homology*. ICMS 2014, doi:10.1007/978-3-662-44199-2_28.
   — Cubical persistent homology used for both the offline β₀/β₁ targets and the `b0` loss.
6. H. Edelsbrunner, J. Harer. *Computational Topology: An Introduction*. AMS, 2010.
   — Persistence theory background (superlevel filtration, Betti numbers).
7. A. Kendall, Y. Gal, R. Cipolla. *Multi-Task Learning Using Uncertainty to Weigh Losses
   for Scene Geometry and Semantics*. CVPR 2018, doi:10.1109/CVPR.2018.00781.
   — Basis of the optional `HomoscedasticMinkowskiLoss` multi-task weighting.
8. S. van der Walt et al. *scikit-image: image processing in Python*. PeerJ 2:e453 (2014),
   doi:10.7717/peerj.453. — `perimeter_crofton` and `euler_number`, the canonical digital
   estimators the loss is calibrated and validated against.
9. D. Legland, K. Kiêu, M.-F. Devaux; and T. Lehmann, D. Legland (*Insight Journal*, 2012).
   — N-dimensional Crofton perimeter via run-length encoding (digital Crofton estimator).
   DOI not verified here; cited for completeness of the integral-geometric lineage.