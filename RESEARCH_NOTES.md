# Research notes — theory, extremes, regimes

Working notes, started 2026-09-23. `EXPERIMENTS.md` §6 lists the macro steps (M1–M5); this
file develops the ideas behind M2, M3 and M5, and the Study-1 weighting question of M4.
Items marked **Decision** are open choices for the researcher, not settled.

---

## 1. Central hypothesis: geometry as the transferable handle

**In words.** The Minkowski functionals describe the geometry of the precipitation field.
The geometry of a "normal" storm is similar to that of an extreme one. So a super-resolution
model trained or fine-tuned on these functionals learns geometry that carries over to
extremes. That makes it more robust to extremes it has not seen at that intensity.

**Why the functionals are the right object.** In 2-D, Hadwiger's theorem says area,
perimeter and Euler characteristic span every motion-invariant, continuous, additive
functional of a set. Applied to the excursion sets {x ≥ u} across thresholds, γ(u) is
therefore a complete description of the field's morphology *within that invariance class*.
Two limits come with this:
- The functionals carry no position information (DECISIONS §6).
- Matching γ is necessary but not sufficient for a realistic field (DECISIONS §2).

**Split into claims that can each be tested:**

| | Claim | Kind | Tested by |
|---|---|---|---|
| H1 | Excursion-set geometry of extreme events, suitably normalised, resembles that of ordinary events | property of the data | M1 analysis, no training |
| H2 | A model trained with the Minkowski term ties the geometry at high thresholds to the geometry it learns at lower ones, rather than memorising intensities | property of the learner | gradient/threshold analysis; γ residual by threshold |
| H3 | So, beyond the training range, a Minkowski-trained model produces more realistic extreme structure than one trained without it | the claim that matters | M2 cut-off training |

H3 without H1 would be luck. H1 without H3 would mean the geometry is there but the model does
not use it.

**How this fits what is already known.**
- **The pixel-mass bottleneck is an argument *for* the hypothesis** (DECISIONS §5). The
  thresholds ≥ 31 mm/h carry almost no gradient. Any tail gain the loss produces today must
  therefore already come from geometry learned at lower thresholds, which is H2. The
  cut-off experiment makes this explicit: above the cut, the fixed-threshold terms are
  *always* empty in training.
- **The area channel alone is a reweighted W₁ between the marginal intensity laws**
  (DECISIONS §13). It is distributional information, not geometry. So the geometric claim
  has to rest on **perimeter and Euler characteristic**. That gives a direct ablation for
  M2: A-only against A+P+χ, trained identically. If A-only transfers just as well, the
  effect is distributional, not geometric.
- **Fixed physical thresholds cannot express "same geometry, higher intensity".** A threshold
  grid defined *relative* to the sample would compare shapes, not intensities, and would
  implement H1 directly in the loss. Examples: fractions of the coarse-input maximum, or
  exceedance above a local quantile. This is a design option for M2/M3 (**Decision**).

**Predictions, stated in advance:**
1. (H1) Normalised γ curves collapse across intensity classes. Two normalisations to try: u
   divided by the event peak, and u as a local quantile. So do perimeter–area scaling
   (an area–perimeter fractal relation) and the χ(u) shape. If they do not collapse, the
   hypothesis needs a regime qualifier (§4).
2. (H3) On a cut-off split, the *degradation* from i.i.d. to o.o.d. is smaller with the
   Minkowski term than without it. Measure it on exceedance ratio, RL bias and peak ratio.
3. (H3, geometric) The gain comes from P and χ, not from A alone.
4. (H1 × H3) Skill on held-out extremes decreases with their geometric distance to the
   nearest training event (§3, Q5). The slope is flatter for the Minkowski model.

---

## 2. What is an extreme? (Decision)

The question: is an extreme event a type of event the model has never seen, or one whose
type is in the training set at lower intensity? These are different experiments, and the
hypothesis speaks to them differently.

| | Definition | What o.o.d. means | What H1–H3 predict |
|---|---|---|---|
| **E1** | *Intensity extreme within a familiar regime*: the same kind of storm, stronger | peak or rate beyond the training range | the hypothesis is **built for this** |
| **E2** | *Novel event type*: e.g. derecho/bow echo, medicane, back-building stationary MCS, an orographic extreme in an unrepresented region | regime absent from training | transfer only if the geometry is shared, so a **stress test** that may fail honestly |
| **E3** | *Locally extreme*, relative to local and seasonal climate (ECMWF EFI / M-climate style) | beyond e.g. the local 99.9th percentile of 15-min rate for that region and season | a normalisation of E1 or E2, not a third experiment |

- **E3 is a choice of scale.** Fixed mm/h thresholds across Europe conflate climates: 50 mm/h
  is routine in some Mediterranean or Alpine convection and exceptional over the North Sea.
  ECMWF defines "extreme" relative to the model's own climate at each location (EFI,
  Shift of Tails; see §4 references). The OPERA climatology from the enlarged archive can
  play that role. This also answers whether the cut should be in mm/h or in local quantiles.
- **Unit of extremeness.** Another orthogonal choice: the pixel rate (15-min, 2 km), the
  patch or tile maximum, or an event total. The current tail protocol uses pixel exceedances
  (POT) and patch maxima. An event-based cut needs an event definition (below).

**A natural reading of the stated hypothesis:** E1 as the primary experiment, possibly on
an E3-normalised scale, with E2 as a secondary stress test reported whatever the outcome.
**Decision** for the researcher.

**Building the "unseen extremes" split (M2), whichever definition is chosen:**
- **Cut by event and in time, not by patch.** The current splits leak the same storm across
  neighbouring tiles and ±15-min frames (DECISIONS §15). A patch-level cut would leak it the
  same way.
  - *Events:* space–time connected components above a level, or simply
    (day × region) blocks, with a buffer of hours between splits.
- **Screen first** (M1). The top of the current tail is disproportionately artefacts, and
  the declutter step punches holes in real cells above 150 mm/h.
- **Choose the cut from counts, not round numbers.** It must leave enough held-out events
  after screening for stable tail metrics, and enough upper-mid range in training for the
  loss to learn from. A choice of u_c could look like this, to be re-derived on clean data:
  - fixed-threshold version: a patch max around 60–90 mm/h;
  - local-quantile version: a level around 99.9%.
- **Report how o.o.d. the *input* is, not only the target.** A held-out extreme whose coarse
  input is also out of range tests input extrapolation too. Report both axes.
- **Compare degradation, not absolute skill.** For every model trained on the cut set, give
  i.i.d. score, o.o.d. score and their ratio: MSE, Minkowski (A-only and full), the
  competing losses once fixed (M4), and FM.

---

## 3. What the data analysis (M1) has to answer for the theory

Beyond quality screening (see `scripts/data_quality/README.md`):

- **Q1 — collapse (H1).** Normalised γ curves by intensity class, region and season. Do
  they collapse?
- **Q2 — scaling.** Perimeter against area at each threshold (isoperimetric ratio, fractal
  exponent), by intensity class. Does the exponent depend on intensity?
- **Q3 — topology.** χ(u) and component counts against intensity. Do extreme events
  fragment differently (e.g. one organised core against many cells)?
- **Q4 — supply.** After screening, how many extreme events are there, where and when? Is
  a held-out o.o.d. set viable, and under which definition (E1 or E3)?
- **Q5 — analogues.** For each extreme event, the distance in normalised-γ space to its
  nearest training event. This is a continuous "how unseen is it" axis that prediction 4
  (§1) needs.
- **Q6 — regimes.** Once §4 gives a regime definition: are extremes concentrated in a
  few regimes, and does γ collapse *within* a regime even where it fails across regimes?

---

## 4. Regime-aware Minkowski training (M3)

**The idea.** Use the predicted Minkowski functionals, the prediction and the low-res input to
classify which precipitation regime a sample is in, and make the structural training
regime-aware. The regime definition is open work. The starting point is what ECMWF does.

### What ECMWF and the literature do

- **ecPoint "gridbox weather types"** (Hewson & Pillosu 2021). **The closest analogue.**
  - *Purpose:* turn IFS gridbox rainfall into point-rainfall PDFs. That is a
    grid-to-point, sub-grid-variability problem at almost our scale ratio.
  - *Physical argument:* large-scale rain has low sub-grid variability, convective rain has
    high variability, and the cell drift speed sets the shape (streaks against blobs).
  - *Governing variables, in tree order:* convective fraction of precipitation, total
    precipitation, 700 hPa wind speed, CAPE, 24 h clear-sky solar radiation.
  - *Method:* a shallow expert-built decision tree. Breakpoints go where the distribution
    of the forecast error ratio (obs − gridbox)/gridbox changes (KS tests). The tree has
    214 leaves, each with ≥ ~200 cases. The convective-adjustment timescale, shear,
    orography and others were tested but are not all used operationally.
  - *Transferable recipe:* **regimes = classes with different sub-grid variability**,
    which is exactly what super-resolution has to predict.
- **IFS convective / large-scale split** (`cp` / `lsp`). This comes from the model's
  schemes and depends on resolution. It is not an observed classification. It would need
  ERA5 as an extra input.
- **Convective adjustment timescale τc** (Done et al. 2006; used at DWD and the Met Office
  in research). It separates equilibrium from non-equilibrium convection, and
  non-equilibrium convection is less predictable and more sub-grid. It needs CAPE and rain
  rate, so ERA5. Secondary.
- **Euro-Atlantic weather regimes** (NAO±, blocking, Atlantic ridge; Ferranti et al.
  2015). Large-scale, so low relevance for the structure of a 256 km patch. At most use them
  as stratification metadata.
- **Radar-based:**
  - *Steiner, Houze & Yuter (1995)* convective/stratiform separation (peakedness above
    background) works on the 2 km target and prediction. It gives a convective fraction,
    our analogue of ecPoint's first variable.
  - *Organisation indices:* SCAI, COP, Iorg (compared on radar by Brune et al. 2020).
  - *Caution (Janssens et al. 2021):* cloud-pattern metrics form a *continuum*, not
    discrete classes. Hard regime labels may be artificial.
  - *Novelty:* no established work was found that classifies precipitation regimes by
    Minkowski functionals or Euler-characteristic curves. The search was not exhaustive.

### Candidate regime variables available here (ecPoint mapping)

| ecPoint variable | Analogue in this project |
|---|---|
| convective fraction | Steiner convective fraction of the target (for labels) or of the prediction |
| total precipitation | coarse-input intensity and wet fraction |
| 700 hPa wind (cell drift) | advection between frames (the ±15 min fields already fetched), or ERA5 V700 |
| CAPE | ERA5 (optional input) |
| solar radiation | date, hour and latitude |
| orography | DEM roughness |
| — | γ descriptors: P/A slope, χ at mid thresholds, peakedness |

### Design options (Decision)

- **(a) Regime-conditioned loss.** The threshold grid, the λ or the γ target depends on the
  regime.
- **(b) Regime as a model input** (a FiLM or conditioning token).
- **(c) Mixture of experts**, one head per regime.
- **(d) As described: a classifier on (low-res input, prediction, predicted γ).**
  - *Circularity:* if the regime comes from the *prediction*, the model can move a sample
    into an easier regime, which is a new form of reward hacking (§2 of DECISIONS).
  - *Mitigations:* a detached or frozen classifier; or regimes from input + DEM only
    (available at inference), checked against target-derived regimes.
- **Defining regimes the ecPoint way.** Pick the candidate variables above. Find breakpoints
  where the *sub-grid* error distribution changes, for example the fine-max / coarse-block
  ratio or the γ residual of a baseline. Keep the tree shallow. This makes "regime" mean
  "a different sub-grid geometry", which is what the loss needs to know.

---

## 5. Study 1 weighting: homoscedastic, Optuna, or calibrated bracket? (M4)

Full reasoning is in DECISIONS §16. In short:

- **Homoscedastic (Kendall) weighting: not for MSE against auxiliary losses.**
  - It balances loss *values*, which §9 rejects as the wrong target.
  - It assumes Gaussian likelihoods.
  - It can absorb non-zero floors (SSIM's is 0.465).
  - It makes each loss's weight an uncontrolled learned quantity, so the comparison stops
    being a comparison of losses.
  - It is fine for balancing the three Minkowski channels against each other.
- **Optuna: not as the primary tool.**
  - λ is one-dimensional per loss and bounded above by hacking.
  - Each trial is a full training run.
  - The selection metric becomes the result.
  - It is worth it only for joint searches (λ × anneal × tail-sample fraction), on short
    proxy runs with pruning and an equal budget per loss.
- **Proposed: calibrated bracket.**
  1. Measure gradient parity λ* (done, EXPERIMENTS §3).
  2. Train the bracket {λ*/3, λ*, 3λ*, 10λ*} for every loss, with one fixed epoch budget
     and the last checkpoint evaluated.
  3. Pick each loss's point by one pre-registered rule: the best exceedance ratio with
     ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla.
  4. Run 3 seeds at the chosen point, and report the whole bracket.
- **Before any rerun:** fix SSIM (use a fixed data_range in normalised space, a smaller eps,
  or a wet-window mask), and log or remove the optical-flow `teacher=target` fallback.
- **FM:** implement the four losses as rewards in `reward_finetune.py`, with the same
  bracket on `REWARD_WEIGHT`.

---

## 6. Open decisions

1. The definition of extreme: E1 / E2 / E3, and the unit (pixel rate, tile max, event)
   (§2).
2. The cut level and the event definition for the o.o.d. split (§2). Both depend on the
   M1 counts.
3. Fixed physical thresholds or sample-relative thresholds in the loss (§1).
4. Regime definition and integration: options (a)–(d) (§4). Whether to bring in ERA5.
5. Screening policy: pixel masking or tile rejection, and the declutter rule (M1,
   `scripts/data_quality/`).
6. Which archive years to use (the 2012 data is much dirtier than 2024 in the first audit).

---

## References

First authors, years and links were checked. Some co-author lists and titles are from memory,
so re-verify them before citing.

- Hewson, T. D. & Pillosu, F. M. (2021). A low-cost post-processing technique improves weather
  forecasts around the world. *Communications Earth & Environment* 2, 132.
  https://www.nature.com/articles/s43247-021-00185-9 (decision tree: Supplementary Table 1)
- ECMWF Forecast User Guide: Extreme Forecast Index / Shift of Tails, M-climate.
  https://confluence.ecmwf.int/spaces/FUG/pages/673551296/Section+8.1.9.2+Extreme+Forecast+Index+-+EFI
- Ferranti, L., Corti, S. & Janousek, M. (2015). Flow-dependent verification of the ECMWF
  ensemble over the Euro-Atlantic sector. *QJRMS*. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2411
- Done, J. M., Craig, G. C., Gray, S. L., Clark, P. A. & Gray, M. E. B. (2006). Mesoscale
  simulations of organized convection: importance of convective equilibrium. *QJRMS* 132.
- Flack, D. L. A. et al. (2016). Characterisation of convective regimes over the British Isles.
  *QJRMS* 142. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2758
- Steiner, M., Houze, R. A. & Yuter, S. E. (1995). Climatological characterization of
  three-dimensional storm structure from operational radar and rain gauge data. *J. Appl.
  Meteor.* 34.
- Brune, S., Buschow, S. & Friederichs, P. (2020). Observations and high-resolution
  simulations of convective precipitation organization over the tropical Atlantic. *QJRMS*.
  https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.3751
- Janssens, M. et al. (2021). Cloud patterns in the trades have four interpretable dimensions.
  *GRL*. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091001
- Chang & Sapsis (2026). Extreme Event Aware Learning. doi:10.1038/s41467-026-76811-x
  (see DECISIONS §13)
