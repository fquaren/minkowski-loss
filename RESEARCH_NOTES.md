# Research notes — theory, extremes, regimes

Working notes, started 2026-09-23. `EXPERIMENTS.md` §6 lists the macro steps (M1–M5); this
file develops the ideas behind M2, M3 and M5, and the Study-1 weighting question of M4.
§7 covers radar data quality for M1 (what QIND measures, how agencies clean radar data), and
§8 is a reading list on precipitation systems as seen on radar.
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

### To explore: regimes from what is available at inference

A regime label is only usable at inference if it comes from what the model sees. Two routes,
in order of preference. Neither uses the high-res field as an input.

**Route 1: the low-res input and its analytic Minkowski functionals.** Compute γ(u) (area,
perimeter, χ) analytically on the coarse input, together with coarse intensity statistics,
wet fraction and DEM descriptors, and ask whether they separate regimes. They are available at
inference, cost nothing, and are free of circularity.
- *Truth for evaluation only:* regimes derived from the 2 km target — Steiner convective
  fraction, organisation indices, γ of the target, or an external label (MeteoSwiss GWT weather
  types, §8.7). The target is used to *score* the classifier, never as its input.
- *Question:* how much regime information survives the 12.5× coarsening? Which regimes stay
  separable (stratiform vs convective vs orographic vs linear/frontal), and which collapse
  together? Does γ(LR) add anything beyond plain coarse intensity and wet-fraction statistics?
- *Practical limit:* the coarse patch is only 10×10 pixels, so γ curves on it are very coarse.
  Consider a larger coarse context (the neighbouring tiles), as ecPoint uses the whole gridbox
  neighbourhood.

**Route 2: the analytic Minkowski functionals of the prediction, γ(ŷ).** If Route 1 is too
weak, classify on γ of the model's own super-resolved output, which is also available at
inference.
- *Caveat:* circularity (option (d) above). A regime taken from the prediction lets the model
  move a sample into an easier regime. Use a frozen classifier, detach the regime from the
  graph, and check γ(ŷ)-regimes against target-derived regimes.
- The learned emulator's predicted γ is a third variant, but the emulator is lowest priority
  (DECISIONS §14).

**What would make this worth doing.** If Route 1 separates regimes well, the input carries
the regime, and a regime-conditioned loss or input (options (a)–(c)) is well posed. If only
Route 2 works, regime-awareness has to be built into the loop, with the circularity controls
above. If neither works, regimes are not recoverable at 25 km, and that is a result for M5
(the geometry is not visible from the coarse field).


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
5. Screening policy: **decided in principle** (DECISIONS §17: reject clear errors only,
   keep imperfections, no labelled set). Still open: the rule set and thresholds, pixel
   masking vs tile rejection, the declutter rule. Discuss with MCH first (§7.4).
6. Which archive years to use. The 2012 data is much dirtier than 2024 in the first audit,
   and the archive has product breaks (late 2015, 2017-09-29, 2024-07-05; §7.2). Restricting to
   after a break trades volume for homogeneity. After 2017-09-29 still leaves ~8 years of
   15-min data.

---

## 7. Radar data quality: what QIND measures, and how agencies clean radar data (M1)

### 7.1 What QIND is

`QIND` is OPERA's per-pixel **total quality index**, in [0, 1] (0 = unusable, 1 = perfect),
delivered with each composite. The ODIM task name in our files, `pl.imgw.quality.qi_total`,
identifies the implementation: IMGW's (Polish met service) `qi_total` module in the BALTRAD
toolbox, from the RADVOL-QC system.

**How it is built** (Szturc et al. 2011; Ośródka et al. 2014, 2022):
- Pick the significant *quality factors*: technical radar parameters (frequency, beam width,
  sensitivity, calibration date), distance to radar / beam broadening, beam height, beam
  blockage, attenuation in rain, and the detections of the QC algorithms (non-meteorological
  echoes from the sun and other emitters, specks, ground clutter).
- Map each factor to an individual index QIᵢ ∈ [0, 1].
- Combine: in RADVOL-QC "after the whole quality control chain the final total QI is
  determined using a multiplicative formula".

**How the composite uses it.** In the ODYSSEY rain-rate composite, "each composite pixel is a
weighted average of the valid pixels of the contributing radars, weighted by a quality index,
the distance from center of the pixel and an exponential index related to inverse of the beam
altitude" (Saltikoff et al. 2019). NIMBUS, the BALTRAD-based production line replacing
ODYSSEY for rain rate (von Lerber et al., EMS 2023), composites from the lowest elevation
only. In the archive the switch is visible as the product rename `QIND_RATE` → `RATE` on
**2024-07-05** (the first hours of that day are still `QIND_RATE`). The NIMBUS file is a
single-time `PPI` at the nominal time, while the ODYSSEY file covers a 15-min window (e.g.
11:50–12:05). Both sit on the same 2 km grid.

**What it is not.** QI is mostly an *a priori, geometry-driven reliability* of the measurement
setup at that pixel (range, beam height, blockage, radar hardware), plus flags from the QC
algorithms that ran. It is **not a detector of whether an echo is meteorological**. A clutter
spike close to a radar with good geometry can score well. That is what we measured:
- on the 2026 live composite, pixels ≥ 31 mm/h were twice as often below QI 0.8 (41.7% vs 21.0%);
- on 2018-06-15T16:30, pixels > 500 mm/h had mean QI 0.267 against 0.282 overall — no signal;
- `QIND` is defined on only ~7–11% of pixels, against ~43–50% for the rate (cause unknown).

Also, QIs are **not harmonised across services** (Einfalt et al. 2010, "a tower of Babel?"),
and the factor set and combination rule actually used in OPERA production are not in the
papers accessible here. **Use QIND as a covariate or a soft reliability weight** (e.g. to
down-weight far-range / blocked areas), and as one feature among several in the audit. **Do
not use it as the artefact screen.**

### 7.2 The standard cleaning chain at meteorological agencies

Most of the chain runs on **single-radar volume data, before compositing**, so we inherit it
rather than re-run it. Only the composite-level steps are ours to add.

| # | step | removes / corrects | who (examples) | usable on our composite? |
|---|---|---|---|---|
| 1 | Doppler / static clutter filtering, clutter maps | ground clutter, anomalous propagation | MeteoSwiss (Germann et al. 2006, 2022), Met Office Nimrod (Harrison et al. 2000), Météo-France (Tabary 2007) | already applied upstream; **residual static clutter: yes**, from long-term statistics |
| 2 | Texture / gradient / echo-geometry tests | isolated clutter, speckle | Gabella & Notarpietro 2002 (in wradlib); Steiner & Smith 2002; Lakshmanan et al. 2007 (neural net) | **yes** (EURADCLIM applies the Gabella filter to the OPERA composite) |
| 3 | Emitter / RLAN / sun-spike removal | radial rays, rings, spokes | BALTRAD bRopo (FMI, Peura 2002); RADVOL-QC SPIKE (Ośródka & Szturc 2022); problem: Saltikoff et al. 2016 | **yes**, by geometry (straight radial lines centred on a radar site) |
| 4 | Polarimetric non-met classification | wind turbines, chaff, biology | Ośródka & Szturc 2022 (DP.TURBINE, DP.NMET); Figueras i Ventura & Tabary 2012 | no (needs the volume data); only its residue is visible |
| 5 | Beam-blockage / visibility correction | underestimation behind mountains | MeteoSwiss visibility maps; BALTRAD `beamb`; OPERA since late 2015 | no. Treat as a kept imperfection |
| 6 | Attenuation correction (C-band) | underestimation behind strong cells | RADVOL-QC; polarimetric (ΦDP) at MeteoSwiss / Météo-France | no. Kept imperfection, but note it biases the tail |
| 7 | Vertical-profile (VPR) correction, bright band | range-dependent over/underestimation | MeteoSwiss, Météo-France, Met Office | no |
| 8 | Calibration monitoring (sun, cross-radar) | radar-wide biases | all services | partially: radar-wide jumps in time are detectable |
| 9 | Z–R conversion | — | OPERA: Marshall–Palmer (EURADCLIM: Z = 200 R^1.6) | fixed. Explains why rates are not gauge-accurate |
| 10 | Quality-based compositing | seams, overlap conflicts | Jurczyk et al. 2020; ODYSSEY / NIMBUS | done upstream |
| 11 | **Satellite cloud mask** | rain under clear sky | OPERA since late 2015; **EURADCLIM**: CLAAS-2 (CM SAF, SEVIRI), 7×7-pixel neighbourhood, zero rain if all neighbours are cloud-free or thin cirrus | **yes**, if we add the satellite data |
| 12 | Long-term static-clutter detection | hot pixels | EURADCLIM: `wradlib.clutter.histo_cut` on annual totals (50-class histogram, classes < 5% of the mode, iterated) | **yes**; our `clutter_climatology.py` is the analogue |
| 13 | Availability rules | gaps, outages | EURADCLIM: ≥ 83.3% data availability per cell | **yes** |
| 14 | Gauge adjustment | amplitude bias | CombiPrecip (MeteoSwiss), RADOLAN / RADKLIM (DWD), EURADCLIM, Park et al. 2019 | out of scope for 15-min SR; relevant for evaluation against gauges |
| 15 | Climatological reprocessing | spokes, clutter, offline | RADKLIM (DWD; Kreklow et al. 2020) | model for a "climate version" of our set |

**The single most relevant precedent is EURADCLIM** (Overeem et al. 2023): KNMI cleaned the
*same* OPERA 15-min, 2 km rain-rate composite (2013–2020) with steps 2, 11 and 12, then
gauge-adjusted it. Their own caveats map onto our problem directly:
- non-meteorological echoes "can still be persistent for some areas";
- radar-wide failures produce "very high rates caused by a constant signal source";
- interference rings and radial patterns remain;
- outliers "limit the applicability of EURADCLIM at the grid cell scale, especially for use in
  extreme value modeling" — i.e. exactly the regime this project works in.

**Product breaks inside the archive** (these make the archive non-stationary, independently of
the weather):
- **late 2015**: OPERA adds beam-blockage correction and a satellite cloud mask;
- **2017-09-29 08:52 UTC**: compositing switches from logarithmic (dBZ) to linear (Z)
  range-weighted averaging (from EURADCLIM);
- **2024-07-05**: ODYSSEY → NIMBUS (`QIND_RATE` → `RATE`; lowest elevation, instantaneous);
- plus a steadily growing radar count (~138 on average in 2013–2020).

Checking the tail statistics across each break is part of M1, and the break dates are natural
candidates for the "which years" decision (§6).

### 7.3 What this means for us: clear errors out, imperfections kept

The screening policy is in DECISIONS §17. It rejects **clear errors** only, and keeps
imperfections so the model learns to be robust to them.

- **Clear errors**: signatures that are non-meteorological by construction:
  - residual static clutter (step 12);
  - emitter rays and rings (step 3);
  - isolated spikes far above their neighbourhood (step 2);
  - physically impossible rates (we have seen 13,072 and 64,842 mm/h);
  - radar-wide constant-signal failures;
  - single-frame appearances with no precursor or successor (15-min continuity);
  - rain under a clear sky (step 11, if satellite data is added);
  - tiles dominated by nodata.
- **Kept imperfections**: blockage and range underestimation, attenuation shadows, compositing
  seams, drizzle speckle, mild residual clutter.

The hard part is that **real extremes are also "anomalous"**: steep gradients, small intense
cores, fast evolution. A rule that fires on "unusual intensity structure" will remove real
storms preferentially, and bias exactly the tail the study is about. So prefer rules keyed on
geometry, persistence and physical impossibility over rules keyed on intensity alone. Always
report rejection rates per intensity bin.

### 7.4 To discuss at MCH (Daniele Nerini, Lionel Moret)

**What we found, to check with MCH** (updated 2026-10-04 from the gauge validation:
1,735 DWD and SwissMetNet 10-min gauges, 12.1M station-frames; `notes/data_quality_assessment`,
`validation/repair_v3/repair_comparison.md`). Several earlier questions are now answered by
our own data; we ask whether MCH's experience agrees.
- **Screen.** We built and validated our own rules: static clutter (per-year hot pixels),
  isolated spikes, emitter rays, range rings, cells with no precursor or successor at
  ±15 min, and > 500 mm/h cores with their surroundings. Under each rule's pixels the
  gauges are dry. Does MCH see artefact types we miss, or use other composite-level
  filters (Gabella-type, satellite masks)?
- **Repair, not rejection.** Rejecting a whole tile for one artefact threw away real rain
  (the other pixels of rejected tiles match the gauges almost like untouched pixels), so v3
  repairs pixels instead. Is that MCH's practice for ML training sets too?
- **QIND.** Useless as a filter here: under ODYSSEY it points the wrong way (unconfirmed
  tail pixels have the *higher* QIND, median 0.90 vs 0.20), under NIMBUS it carries no
  signal, and its scale changes with the product. Is that known, and why?
- **ODYSSEY reads about half the gauge**, with the gauges' tail shape up to ~89 mm/h after
  repair. Is a factor ~0.5 expected from Z–R and from comparing a 2 km area with a point?
- **NIMBUS rates heavy rain above the gauges, increasingly with intensity**: 1.15× as many
  exceedances as the gauges at 31 mm/h, 3.6× at 89 mm/h; the pixels are rain (96% have rain
  at the gauge), and no artefact rule touches them. Is this known? Is it the instantaneous
  lowest-elevation scan against 10-min gauge totals, hail, or the NIMBUS processing? This
  decides whether NIMBUS can be a test set.

Still open:
- A defensible upper plausibility bound for a 15-min, 2 km rate. We use 500 mm/h, with the
  repair reaching down to 150 mm/h around such cores.
- **CombiPrecip and POH/MESHS for 2012–2026.** The open-data portal keeps them for 14 days
  only; we already use the open SwissMetNet gauges.
- Handling of the product breaks (2015, 2017, 2024) in ML training sets.

**The extreme threshold (u = 31 mm/h).** Background: 31 is one point of the loss's
log-spaced grid (DECISIONS §1). It serves as the POT level of every tail metric "for
estimability, not physics" (DECISIONS §10). It also sets the FSS levels, the v2 `extremes`
subset, the tail stratum and the clutter hot-pixel rule. Warning signs:
- ξ_obs flips sign between 31 (+0.18) and 53 (−0.19);
- no threshold diagnostics were ever run;
- the fitted data were capped and artefact-laden;
- exceedances were not declustered.

To ask:
- How does MCH define intense / extreme precipitation for **radar rates** at 2 km and
  5–15 min, as opposed to the accumulation-based warning levels? Is a fixed instantaneous
  31 mm/h meaningful, or should the level be regional and seasonal (E3, §2) — or even
  product-specific, given that NIMBUS has 1.44× more pixels ≥ 31 mm/h than ODYSSEY on the
  same days?
- **Rate or accumulation?** Is the instantaneous 15-min rate the right quantity for
  extremes, or should the tail be defined on 1-h accumulations (impact-relevant, comparable
  with gauges)? 15-min snapshots miss short peaks (Valencia: 28.6 mm/h radar pixel vs
  184.6 mm/h gauge-hour).
- **POT practice:**
  - threshold selection (mean-residual-life, parameter-stability plots);
  - declustering of radar exceedances in space and time;
  - fitting at pixel level or on event maxima;
  - how MCH reports return levels from radar.
- **Radar biases in the tail:** how Z–R conversion and hail contamination (above ~55 dBZ)
  shape the rates above 31 mm/h, and whether a hail cap should apply before any tail
  statistic.
- The gauges give a partial answer to "how far can the radar tail be trusted": after repair,
  ODYSSEY keeps the gauges' tail shape to ~89 mm/h; NIMBUS is too heavy from ~31 mm/h.

**Testing out-of-distribution capability** (for Lionel Moret, 2026-10-02). What v2 has
today is not o.o.d.:
- the week-blocked test set is independent of train, but drawn from the same climate,
  product and regions;
- the event set (`notes/events.md`; 25 storms in test and 2 in NIMBUS once v3 is built) is
  curated known extremes whose types all occur in training;
- `nimbus` is a product shift, but it moves the target distribution itself (NIMBUS has
  1.44× more pixels ≥ 31 mm/h on the same days), so it needs a same-product reference
  (DECISIONS §18).

Options, each a different notion of "unseen", built on the v2 week-blocked splits:

| | Shift | How to build it | What it tests | Main confound |
|---|---|---|---|---|
| O1 | **Intensity cut-off** (E1, §2; M2) | remove from training every space–time block (day × tile neighbourhood) whose cleaned max ≥ u_c; those blocks are the test. u_c in mm/h or as a local quantile (E3) | extrapolation beyond the training range: the central hypothesis H3 | the radar tail itself (hail, Z–R, artefacts) above u_c |
| O2 | **Held-out storm type** (E2) | leave one type out: derechos, supercells/hail, Mediterranean HPE, stationary lows. Needs labels beyond the 26 catalogued events (object-based classification, or a weather-type catalogue) | transfer of geometry across regimes, a stress test | labels; types overlap |
| O3 | **Held-out region** | train without a region, test on it: e.g. the Alps (orography), the Mediterranean coast, or Switzerland as a whole | spatial transfer, orographic forcing | each region has its own national radar network and processing |
| O4 | **Held-out period** | train on 2012–2021, test on 2022–2024 (ODYSSEY) | non-stationarity, climate trend | the 2015 and 2017 product breaks and the growing radar count |
| O5 | **Product / sensor shift** | NIMBUS (built), the national composites (CombiPrecip, the Swiss composite) | robustness to how the observation is made | the reference moves with the product |
| O6 | **Input-source shift** | coarse input from NWP (ICON-CH1/2) or a climate model instead of coarsened radar | the realistic deployment (perfect-prog) | no km-scale truth: statistical evaluation only |
| O7 | **Graded "how unseen"** (§3, Q5) | not a split: distance from each test event to its nearest training analogue (γ space or coarse-input space); skill against distance | a continuous version of every option above | needs an analogue metric |

Proposed core: **O1 as the primary experiment** (it is the hypothesis), **O7 reported on every
test set**, NIMBUS (O5) kept as the robustness row it already is. O2 as a stress test if
labels can be had. O3/O4/O6 only if MCH sees them as the operationally relevant shift.
Whatever the option, report the i.i.d. → o.o.d. *degradation* per model (§2), and how o.o.d.
the input is as well as the target.

To ask:
- Which shift matters operationally to MCH for downscaling: unprecedented intensities, Alpine
  orography, new radars or products, future climate, or NWP inputs?
- **O1:** above which rate does the radar tail stop being trustworthy enough to be a test
  target (hail, attenuation, Z–R)? Should the cut be on 15-min rates or 1-h accumulations,
  and in mm/h or relative to local climate?
- **O3:** is a held-out region meaningful when every country has its own network? Would
  **Switzerland held out** work, with CombiPrecip and the gauges as an independent truth?
- **O2:** does MCH have an event catalogue or classification (e.g. TRT cell tracks, POH/MESHS
  hail, lightning) that could label storm types across the archive?
- **Independent truth for o.o.d. events:** CombiPrecip, hail (POH/MESHS), lightning (the
  10-min gauges are already in use). Which can MCH share, and for which period?
- **O6:** interest in applying the model to ICON-CH1/2 or to climate projections, and what
  evaluation MCH would accept without km-scale truth.
- How does MCH evaluate its own ML nowcasting on unseen extremes?

### 7.4b Proposed cleaning plan for an extremes study (proposal, 2026-09-28 — not decided)

Premise, from the audit: the flag rate rises with intensity, and the very top of the
distribution is almost entirely artefact. So the screen matters most exactly where it is most
likely to delete real storms. Work from the most certain signatures to the least certain.
Prefer fixing pixels over dropping tiles when the artefact is point-like.

1. **Fix the DEM orientation first** (EXPERIMENTS §5). It is a bug, not a screening choice.
2. **Missing, not zero.** Anything removed becomes NaN, with a valid-pixel mask in the loss,
   never 0. Zeroing is what makes the current declutter step punch holes in real storms.
3. **Static masks, per product period.** Mask pixels that reach ≥ 31 mm/h in > 1% of steps
   (~100× climatology), computed separately per period (clutter changes, e.g. around 2016
   and 2023). This is pixel-level and time-invariant, and removes e.g. the Weissfluhgipfel
   tile that supplies the three largest maxima.
4. **Geometry, using the radar sites.** RLAN rays are straight lines *through a radar
   site*, and rings are circles *around* one. With the OPERA database, a thin elongated
   component aligned with the direction to its nearest radar is a near-certain ray. Reject
   the tile for rays, rings and radar-wide failures, since these are extended artefacts.
5. **Spatial support.** A real 2 km core spans several pixels. A pixel ≫ its neighbourhood
   with no support is masked as a pixel, and the storm around it is kept.
6. **Temporal support.** A cell with no precursor or successor within advection distance at
   t ± 15 min is suspect. It has high precision for spikes, but check its rejection rate
   against intensity, because short-lived convection exists.
7. **Physical plausibility bound** on the 15-min, 2 km rate, after masking (3–6). Set it with
   MCH (§7.4), not by us. Anything above it is an artefact by definition.
8. **Per-radar, per-period exclusion or down-weighting** for the consistently bad sites
   (EXPERIMENTS §5), where 3–7 do not already clean them.
9. **QIND** as a soft weight or covariate only (§7.1).
10. **Validate the surviving tail against independent evidence**, not against the rules
    themselves:
    - rejection rate per intensity bin, per rule;
    - lightning co-occurrence for convective extremes;
    - gauge extremes (ECA&D, as EURADCLIM does) and the Swiss reference (CombiPrecip);
    - a by-eye pass over the top of the surviving tail (`patch_gallery.py --select top_max`).
11. **Then define "extreme" on the screened data**, regionally (E3, §2), so that
    artefact-prone regions do not define the tail. Build the o.o.d. split on it.

### 7.4c Status of the radar post-processing, and whether to release it (2026-10-02, status updated 2026-10-03)

**Implemented** (`src/data/cleaning.py`, `tests/test_cleaning.py`, used identically by the
scan and the store build), against the 7.4b plan:

| 7.4b step | Status |
|---|---|
| 1. DEM orientation | fixed (2026-09-28) |
| 2. Missing, not zero | partly: repaired pixels take their neighbourhood's value and nothing is zeroed above 150 mm/h, but removal is still by *tile*, and partially covered tiles are still excluded (`notes/events.md` §5) |
| 3. Static masks per period | done, per calendar year, over the complete archive (≥ 31 mm/h in > 1% of steps → neighbourhood median) |
| 4. Geometry | rays done (thin components ≥ 80 km aligned with a radar within 250 km); range rings detected climatologically (audit flag, 2026-10-02). Gauges (10-03): ring pixels are real rain below 31 mm/h and never at ≥ 89 mm/h, so **repair the high values on the ring, don't reject the tile**. Radar-wide failures not done; three missed failures and proposed rules in §7.4d (2026-10-06) |
| 5. Spatial support | done (spike repair). Gauges (10-03): **right for both products**. The gauge under a repaired pixel is near dry (AUC vs untouched 0.82–0.94 ODYSSEY, 0.90–0.94 NIMBUS). NIMBUS's 32–44% presence rate is light rain nearby, not a supported spike |
| 6. Temporal support | audit flag (2026-10-02). **Validated (10-03): ~1% corroborated in every intensity bin**, the cleanest rule; ready to apply. It fires almost only under ODYSSEY |
| 7. Plausibility bound | provisional 500 mm/h, not yet set with MCH. Gauges (10-03): rejecting the *tile* discards real rain (pixels inside rejected tiles 67–79% corroborated below 89 mm/h under ODYSSEY, 91–97% in every bin under NIMBUS), so **mask the pixel instead** |
| 8. Per-radar, per-period exclusion | 14 consistently bad radars over the whole archive (2026-10-03, after fixing three ranking bugs; EXPERIMENTS §5). Not validatable with the current gauges: only Berlin lies in Germany/Switzerland |
| 9. QIND as a weight | **does not discriminate (10-03)**: AUC 0.26 under ODYSSEY (inverted: median 0.20 corroborated vs 0.90 not), 0.53 under NIMBUS. Explain the inversion before using it even as a covariate |
| 10. Validation against independent evidence | **done for Germany and Switzerland (10-03)**: 1,735 DWD + SwissMetNet 10-min gauges, 12.1M station-frames, presence *and* values (EXPERIMENTS §5, `notes/data_quality_assessment.pdf` Part II). Tail calibration: ODYSSEY exceedance ratio flat to 89 mm/h once flags are applied, NIMBUS too heavy above ~31. Rest of Europe: needs EURADCLIM / ECA&D; CombiPrecip asked from MCH |
| 11. Regional definition of "extreme" | not done |

Also done: the product-break analysis (DECISIONS §18, `era_gap.py`), rejection rates per
intensity bin in every build report, and a reproducible pipeline from the public archive
(fetch → climatology → scan → splits → store → checks).

Rejection is now light: 0.2–0.4% of tiles in every bin from 1 to 500 mm/h, 40% above 500.
The early audit had found the top of the tail mostly artefact. Whether the 150–500 mm/h
range that v2 keeps is mostly real is now **answered for Germany and Switzerland**, and it
depends on the product.
- *Presence.* Untouched pixels at 150–500 mm/h are 96% corroborated under NIMBUS (median
  gauge 28 mm/h), but only 69% under ODYSSEY (median 14 mm/h).
- *Values.* The ratio N(radar ≥ u) / N(gauge ≥ u) over the same station-frames is flat
  (≈ 0.5) under ODYSSEY up to 53 mm/h, and up to 89 mm/h once the temporal-support and ring
  flags are applied. So the tail shape at the POT level u = 31 is right, and with the flags
  only an excess of about 2.5 at ≥ 150 mm/h remains.
- *NIMBUS* is rain but too heavy-tailed from about 31 mm/h (ratio 3.6 at 89), with no rule
  touching it. Possibly instantaneous scans against 10-min gauges: test with 1-min gauges.

Extending this to the rest of Europe is the gap between "a cleaning we use" and
"a product others can trust".

**Is a shareable product worth it?** It depends on which product.

- *A reprocessed OPERA rate archive* (cleaned 15-min fields, 2012–2026). The closest
  precedent is EURADCLIM (KNMI; Overeem et al. 2023): the same composite, cleaned and
  gauge-adjusted, but distributed as 1-h and 24-h accumulations for 2013–2020 (check whether
  a later version extends it). Ours would differ by keeping the 15-min instantaneous rates and
  an uncapped tail, and by spanning both products. But a credible archive product needs
  Europe-wide validation against gauges, the missing rules (6, 8, rings), and hosting for
  ~1.3 TB. **Not worth it alone**, unless KNMI or MCH want to co-own it.
- *An ML benchmark for km-scale downscaling of extremes over Europe*: the v2 family as it
  stands (leak-free week-blocked splits, the event set, the NIMBUS product-shift set,
  per-tile quality flags, per-year clutter masks), plus the code that regenerates it from the
  public archive and baseline scores. Existing radar ML datasets are national (MeteoNet,
  RainNet/RADOLAN, the US SEVIR) or OPERA crops built for nowcasting (Weather4cast). None
  combines Europe-wide coverage, an extremes focus, event- and product-shift test sets, and
  leak-free splits. **Worth it**: most of the work is already on the thesis path (step 10 is
  needed for M1/M2 anyway, and the baselines are the thesis models). The release-specific
  part (licence check of the OPERA open-data terms, a datasheet, DOI and hosting of the
  metadata, masks and code, plus the patches or a regeneration script, a short data paper)
  is roughly a few weeks. Venues: ESSD, or a datasets-and-benchmarks track.

Minimum before releasing anything: step 10 on at least Switzerland (gauges done 10-03;
CombiPrecip pending) and a gallery pass over the top of the tail; rule 6 (temporal support,
validated, not yet applied); the 2013 events and
the partial-coverage fix (`notes/events.md`); and a decision with MCH on the plausibility
bound. Worth raising with Daniele and Lionel: whether MCH would validate the Swiss part, or
co-author.

### 7.4d What v3's screen misses, and the tail fit as an audit (2026-10-06)

Found by the POT threshold study (EXPERIMENTS §5, `scripts/data_quality/pot_threshold.py`,
outputs in `OPERA/quality_v3/pot_threshold/`). Three failures survive the v3 build; images in
`fig/tail_suspects.png` and `fig/tail_suspects_4862.png`.

| failure | where, when | signature | train pixels |
|---|---|---|---|
| reflectivity ceiling 364.63 mm/h (= 64.0 dBZ under Z = 200 R^1.6) | Valjevo (Serbia) area, 80 rain days Feb-Nov 2023 | speckle at one exact value inside real rain; no pixel saturated in > 50% of frames | 60,145 (+14,480 val, 23,520 test) |
| radar-wide failure | Torrejón de Velasco (Madrid), 2018-04-29, all day | the whole disk at a smooth, range-dependent 60-340 mm/h, raw cores to 10^4 mm/h | 16% of all train pixels > 89 mm/h |
| ceiling 48.62 mm/h (= 50.0 dBZ) | tile r640 c1408 (nearest listed radar Bobohalma, 102 km), 7 days Nov 2019 | a speckled stationary 30-55 mm/h blob | 98 tiles |

**Why each rule lets them through.** Every v3 rule assumes the artefact is *small, local or
transient* against a background of real rain:
- *static clutter* (≥ 31 mm/h in > 1% of a calendar year's steps): the ceilings are
  intermittent and rain-conditional, and the Madrid failure lasts one day (0.3% of a year);
- *spike*: a pixel is repaired only if its brightest neighbour is < 10% of it, but the
  saturated pixels sit inside 30-150 mm/h rain;
- *unphysical / footprint*: keyed on > 500 mm/h. The ceilings are below 500. Madrid has
  > 500 cores, but the footprint repair lowers the connected ≥ 150 region to the median of
  its outer ring, and that ring is the failing radar too. The result is a flat plateau:
  4,865 pixels of one tile at exactly 88.27 mm/h (2018-04-29 06:15, r256 c256). The repair
  launders the artefact instead of removing it. v2 would have rejected the two tiles with
  > 500 cores, but not the third.
- *ray, ring*: geometric tests for thin lines and arcs; a disk or a speckled sector is
  neither;
- *temporal support*: catches cells with no precursor or successor. These failures persist,
  which is the opposite;
- *bad-radar ranking* (`radar_quality_v2.py`): flags radars that are outliers in at least
  half of their years. A one-day or one-season episode is diluted in a radar-year and never
  qualifies; the ranking is also not applied as an exclusion;
- *gauge validation*: DWD + SwissMetNet only, so Serbia, Romania and Spain are unseen.

"Radar-wide constant-signal failures" are on the clear-error list (DECISIONS §17, §7.3 above)
but no rule implements them.

**Proposed additions** (clear errors by construction, so within DECISIONS §17):
1. *Repeated-value test (tile).* On the 0.01 mm/h ODYSSEY grid, real rain almost never
   repeats one exact value: per tile, the maximum count of a single value ≥ 31 mm/h has
   median 1, 99th percentile 3, 99.9th 24 (588,617 train tail tiles). Flag a tile when that
   count is ≥ 50 (0.069% of tail tiles, 11% of their pixels ≥ 31): it catches the two
   ceilings and the repair plateaus, and found the 48.62 episode unprompted. NIMBUS is
   coarsely quantised (0.5 dBZ steps), so there the test must count in excess of its own
   per-value baseline.
2. *Ceiling table (pixel).* Per radar area and year, find values whose count exceeds ~50x the
   median of the neighbouring ±0.5 dB bins (364.63 is ~7,500x). Pixels at a ceiling are
   censored ("≥ ceiling"), not measured. Mask them, or reject the tile-frame when they are
   more than a handful, since a ceiling also hides how far the rain really went.
3. *Radar-disk failure (radar-frame).* For each radar and frame, over the area where it is the
   nearest radar: wet fraction (≥ 1 mm/h), mean rate, and the share of log-rate variance
   explained by range alone, R²_r = 1 - Var(z - <z>_θ(r)) / Var(z). Real rain varies in
   azimuth; a receiver or calibration fault is a function of range with a sharp circular edge
   at the maximum range. Add a cross-radar check: the same-day exceedance fraction against the
   neighbouring radars' in their overlap. Flag the radar-frame and reject every tile it
   dominates. Set thresholds from the distribution over all radar-frames; failures should be
   far outliers.
4. *Guard the repair.* `boundary_fill` is valid only if the ring is real rain. Do not repair,
   but escalate to the radar-frame flag, when a component is large (e.g. > 500 px) or its ring
   median is itself ≥ 31 mm/h, and check after every repair that no plateau was created
   (the test of item 1, run after repairs).
5. *Episodic radar ranking.* Rank radar-days, not radar-years, on rel_f150, ceiling counts and
   item 3, with a robust z-score against the radar's own climatology.

**The tail fit as an audit.** The threshold-stability fit found all three failures without
being designed to look for them. Real extremes from many independent storms thin out
smoothly: above a high enough threshold, the excess over u has the same shape whatever u
is, so the fitted ξ stops changing. A failure puts many pixels at a few values, in one
place, over a short time, and the smooth model cannot absorb that bump. Three things show
it: ξ(u) drifts instead of reaching a plateau, the day-block bootstrap interval widens
because one day carries a large share, and the empirical upper quantiles stick at one value
(364.63 was the 99% quantile of the exceedances for every u from 25 to 115). It is a
dataset-level detector, not a rule: it says that the tail is contaminated and, through
per-day and per-value attribution, where to look. It cannot tell a failure from a real
extreme day (an event like Elvira/Friederike is just as influential), it is blind to
failures below u or ones with a smooth tail, and the images decide. Run it after every
build, alongside the rejection rates per intensity bin.

### 7.5 References for §7

Verified 2026-09-28 by search (authors, year, venue). † = from memory, re-verify before citing.

OPERA and the quality index
- Szturc, J., Ośródka, K. & Jurczyk, A. (2011). Quality index scheme for quantitative
  uncertainty characterization of radar-based precipitation. *Meteorol. Appl.* 18.
  doi:10.1002/met.230
- Ośródka, K., Szturc, J. & Jurczyk, A. (2014). Chain of data quality algorithms for 3-D
  single-polarization radar reflectivity (RADVOL-QC system). *Meteorol. Appl.* 21.
  doi:10.1002/met.1323
- Ośródka, K. & Szturc, J. (2022). Improvement in algorithms for quality control of weather
  radar data (RADVOL-QC system). *Atmos. Meas. Tech.* 15, 261–277. doi:10.5194/amt-15-261-2022
- Jurczyk, A., Szturc, J. & Ośródka, K. (2020). Quality-based compositing of weather radar
  derived precipitation. *Meteorol. Appl.* 27. doi:10.1002/met.1812
- Einfalt, T. et al. (2010). The quality index for radar precipitation data: a tower of Babel?
  *Atmos. Sci. Lett.* 11. doi:10.1002/asl.271
- Huuskonen, A., Saltikoff, E. & Holleman, I. (2014). The operational weather radar network in
  Europe. *BAMS* 95, 897–907. doi:10.1175/BAMS-D-12-00216.1
- Saltikoff, E. et al. (2019). OPERA the radar project. *Atmosphere* 10, 320.
  doi:10.3390/atmos10060320
- von Lerber, A. et al. (2023). OPERA5 production lines (NIMBUS). EMS Annual Meeting abstract
  EMS2023-325. (conference abstract, not peer-reviewed)
- BALTRAD bRopo documentation (FMI anomaly detection; after Peura 2002, ERAD).
  http://git.baltrad.eu/manual/bropo/index.html

Cleaning of the OPERA composite and European climate datasets
- **Overeem, A. et al. (2023). EURADCLIM: the European climatological high-resolution
  gauge-adjusted radar precipitation dataset. *ESSD* 15, 1441.**
  doi:10.5194/essd-15-1441-2023 — read first.
- Park, S., Berenguer, M. & Sempere-Torres, D. (2019). Long-term analysis of gauge-adjusted
  radar rainfall accumulations at European scale. *J. Hydrol.* 573, 768–777. arXiv:1904.02788
- Lopez, P. (2014). Comparison of ODYSSEY precipitation composites to SYNOP rain gauges and
  ECMWF model. ECMWF Tech. Memo. 717. (ECMWF grey literature: documents RLAN interference and
  S-band biases in the composites)
- Kreklow, J. et al. (2020). Radar-based precipitation climatology in Germany — developments,
  uncertainties and potentials. *Atmosphere* 11, 217. (RADKLIM) †co-authors
- Overeem, A., Holleman, I. & Buishand, A. (2009). Derivation of a 10-year radar-based
  climatology of rainfall. *J. Appl. Meteor. Climatol.* 48. †

National services
- Germann, U., Galli, G., Boscacci, M. & Bolliger, M. (2006). Radar precipitation measurement
  in a mountainous region. *QJRMS* 132, 1669–1692. doi:10.1256/qj.05.190 (MeteoSwiss)
- Germann, U. et al. (2022). Weather radar in complex orography. *Remote Sens.* 14, 503.
  doi:10.3390/rs14030503 (MeteoSwiss; review of the whole Swiss chain)
- Tabary, P. (2007). The new French operational radar rainfall product. Part I: Methodology.
  *Wea. Forecasting* 22, 393–408 †; Tabary, P. et al. (2007). Part II: Validation. *Wea.
  Forecasting* 22, 409–427. (Météo-France)
- Figueras i Ventura, J. & Tabary, P. (2012). Long-term monitoring of French polarimetric radar
  data quality... *QJRMS*. doi:10.1002/qj.1934
- Harrison, D., Driscoll, S. & Kitchen, M. (2000). Improving precipitation estimates from
  weather radar using quality control and correction techniques. *Meteorol. Appl.* 7, 135–144.
  doi:10.1017/S1350482700001468 (Met Office Nimrod)

General reviews and methods
- Villarini, G. & Krajewski, W. F. (2010). Review of the different sources of uncertainty in
  single polarization radar-based estimates of rainfall. *Surv. Geophys.* 31, 107–129.
- Saltikoff, E. et al. (2019). An overview of using weather radar for climatological studies:
  successes, challenges, and potential. *BAMS* 100. doi:10.1175/BAMS-D-18-0166.1
- Saltikoff, E. et al. (2016). The threat to weather radars by wireless technology. *BAMS* 97.
  doi:10.1175/BAMS-D-15-00048.1
- Lakshmanan, V. et al. (2007). An automated technique to quality control radar reflectivity
  data. *J. Appl. Meteor. Climatol.* 46, 288–305.
- Steiner, M. & Smith, J. A. (2002). Use of three-dimensional reflectivity structure for
  automated detection and removal of nonprecipitating echoes in radar data. *J. Atmos. Oceanic
  Technol.* 19. †
- Gabella, M. & Notarpietro, R. (2002). Ground clutter characterization and elimination in
  mountainous terrain. *Proc. ERAD 2002*. † (the filter EURADCLIM uses, via wradlib)
- Heistermann, M., Jacobi, S. & Pfaff, T. (2013). An open source library for processing weather
  radar data (wradlib). *HESS* 17. †
- Pulkkinen, S. et al. (2019). Pysteps: an open-source Python library for probabilistic
  precipitation nowcasting (v1.0). *GMD* 12. † (MCH co-authors, incl. D. Nerini)
- Michelson, D. et al. (2020). Monitoring the impacts of weather radar data quality control for
  quantitative application at the continental scale. *Meteorol. Appl.* 27.
  doi:10.1002/met.1929 (ECCC, North America; a method for scoring a QC chain objectively)

---

## 8. Reading list: precipitation systems and how they look on radar (M3)

Organised from foundations to the specific questions of M3. ★ = start here. Verified
2026-09-28 by search unless marked †.

### 8.1 Radar meteorology (how to read the images)
- ★ Fabry, F. (2015). *Radar Meteorology: Principles and Practice*. Cambridge UP. — the best
  single book for interpreting reflectivity imagery and its artefacts.
- ★ Rauber, R. M. & Nesbitt, S. W. (2018). *Radar Meteorology: A First Course*. Wiley. —
  organised by weather system, with many annotated examples.
- Doviak, R. J. & Zrnić, D. S. (1993/2006). *Doppler Radar and Weather Observations*. Academic
  Press / Dover. † — the physics reference.
- Rinehart, R. E. (2010). *Radar for Meteorologists*, 5th ed. † — accessible.
- Ryzhkov, A. V. & Zrnić, D. S. (2019). *Radar Polarimetry for Weather Observations*. Springer. †
- Bringi, V. N. & Chandrasekar, V. (2001). *Polarimetric Doppler Weather Radar*. Cambridge UP. †
- Wakimoto, R. M. & Srivastava, R. C., eds. (2003). *Radar and Atmospheric Science: A Collection
  of Essays in Honor of David Atlas*. AMS Meteorol. Monogr. 30. †
- Villarini & Krajewski (2010), Germann et al. (2022) in §7.5 — error sources, from the user's
  side.

### 8.2 Precipitation systems (what produces the patterns)
- ★ Houze, R. A. (2014). *Cloud Dynamics*, 2nd ed. Academic Press. † — convective vs
  stratiform, MCSs, fronts, orographic systems.
- ★ Markowski, P. & Richardson, Y. (2010). *Mesoscale Meteorology in Midlatitudes*. Wiley. † —
  storm modes, supercells, MCSs, fronts, orographic effects.
- Trapp, R. J. (2013). *Mesoscale-Convective Processes in the Atmosphere*. Cambridge UP. †
- Lin, Y.-L. (2007). *Mesoscale Dynamics*. Cambridge UP. †
- Doswell, C. A., ed. (2001). *Severe Convective Storms*. AMS Meteorol. Monogr. 28. †
- Lovejoy, S. & Schertzer, D. (2013). *The Weather and Climate: Emergent Laws and Multifractal
  Cascades*. Cambridge UP. † — the scaling view of rain fields, directly relevant to the
  perimeter–area scaling question (§3 Q2).

### 8.3 Conceptual models of the main system types
Extratropical cyclones and fronts
- ★ Browning, K. A. (1986). Conceptual models of precipitation systems. *Wea. Forecasting* 1,
  23–41. — warm/cold conveyor belts, ana/kata fronts, narrow cold-frontal rainbands,
  wide rainbands, squall lines.
- Hobbs, P. V. (1978). Organization and structure of clouds and precipitation on the mesoscale
  and microscale in cyclonic storms. *Rev. Geophys.* 16. †
- Schultz, D. M. & Vaughan, G. (2011). Occluded fronts and the occlusion process. *BAMS* 92. †
- Catto, J. L., Jakob, C., Berry, G. & Nicholls, N. (2012). Relating global precipitation to
  atmospheric fronts. *GRL* 39. †

Convective vs stratiform
- ★ Houze, R. A. (1997). Stratiform precipitation in regions of convection: a meteorological
  paradox? *BAMS* 78. †
- Steiner, Houze & Yuter (1995), in the §References — the standard radar separation algorithm.

Mesoscale convective systems
- ★ Houze, R. A. (2004). Mesoscale convective systems. *Rev. Geophys.* 42. †
- ★ Schumacher, R. S. & Rasmussen, K. L. (2020). The formation, character and changing nature
  of mesoscale convective systems. *Nat. Rev. Earth Environ.* 1, 300–314.
- Parker, M. D. & Johnson, R. H. (2000). Organizational modes of midlatitude mesoscale
  convective systems. *Mon. Wea. Rev.* 128. — trailing / leading / parallel stratiform
  archetypes, defined on radar.
- Gallus, W. A., Snook, N. A. & Johnson, E. V. (2008). Spring and summer severe weather reports
  over the Midwest as a function of convective mode. *Wea. Forecasting* 23, 101–113. — nine
  radar morphologies (isolated cell, cluster, broken line, squall line NS/TS/PS/LS, bow echo,
  nonlinear).

Convective storm modes
- Thompson, R. L. et al. (2012). Convective modes for significant severe thunderstorms in the
  contiguous United States. Part I. *Wea. Forecasting* 27. †
- Jergensen, G. E., McGovern, A., Lagerquist, R. & Smith, T. (2020). Classifying convective
  storms using machine learning. *Wea. Forecasting* 35, 537–559. — supercell / QLCS /
  disorganised from radar + sounding.

Extreme rain
- ★ Doswell, C. A., Brooks, H. E. & Maddox, R. A. (1996). Flash flood forecasting: an
  ingredients-based methodology. *Wea. Forecasting* 11. † — heavy rain = high rate × long
  duration; why training and back-building matter.
- Schumacher, R. S. & Johnson, R. H. (2005). Organization and environmental properties of
  extreme-rain-producing mesoscale convective systems. *Mon. Wea. Rev.* 133, 961–976. —
  training line/adjoining stratiform, and back-building/quasi-stationary: the two radar
  signatures of extreme rain.

### 8.4 Orographic precipitation (the Alps)
- ★ Houze, R. A. (2012). Orographic effects on precipitating clouds. *Rev. Geophys.* 50, RG1001.
- Roe, G. H. (2005). Orographic precipitation. *Annu. Rev. Earth Planet. Sci.* 33. †
- Rotunno, R. & Houze, R. A. (2007). Lessons on orographic precipitation from the Mesoscale
  Alpine Programme. *QJRMS* 133. †
- Panziera, L. & Germann, U. (2010). The relation between airflow and orographic precipitation
  on the southern side of the Alps as revealed by weather radar. *QJRMS* 136, 222–238. (MCH)
- Foresti, L. et al. (2018). A 10-year radar-based analysis of orographic precipitation growth
  and decay patterns over the Swiss Alpine region. *QJRMS* 144. doi:10.1002/qj.3364 (MCH)

### 8.5 European climatologies and case literature
- ★ Taszarek, M. et al. (2019). A climatology of thunderstorms across Europe from a synthesis
  of multiple data sources. *J. Climate* 32, 1813–1837.
- Wapler, K. & James, P. (2015). Thunderstorm occurrence and characteristics in Central Europe
  under different synoptic conditions. *Atmos. Res.* 158–159, 231–244. — links radar cell
  properties to automatic synoptic types, i.e. a regime study on radar.
- Fluck, E., Kunz, M., Geissbuehler, P. & Ritz, S. P. (2021). Radar-based assessment of hail
  frequency in Europe. *NHESS* 21, 683–701.
- Nisi, L., Martius, O., Hering, A., Kunz, M. & Germann, U. (2016). Spatial and temporal
  distribution of hailstorms in the Alpine region: a long-term, high resolution, radar-based
  analysis. *QJRMS* 142, 1590–1604. (MCH); Nisi et al. (2018), A 15-year hail streak
  climatology for the Alpine region, *QJRMS*, doi:10.1002/qj.3286.
- Feldmann, M., Germann, U., Gabella, M. & Berne, A. (2021). A characterisation of Alpine
  mesocyclone occurrence. *Weather Clim. Dynam.* 2, 1225. (MCH)
- Feldmann, M. et al. (2023). Hailstorms and rainstorms versus supercells — a regional analysis
  of convective storm types in the Alpine region. *npj Clim. Atmos. Sci.*
  doi:10.1038/s41612-023-00352-z
- Feldmann, M. et al. (2025). European supercell thunderstorms — a prevalent current threat and
  an increasing future hazard. *Sci. Adv.* doi:10.1126/sciadv.adx0513 †first author
- Ducrocq, V. et al. (2014). HyMeX-SOP1: the field campaign dedicated to heavy precipitation and
  flash flooding in the northwestern Mediterranean. *BAMS* 95, 1083–1100.
- Delrieu, G. et al. (2005). The catastrophic flash-flood event of 8–9 September 2002 in the
  Gard region, France. *J. Hydrometeor.* 6. † — a radar-documented Mediterranean extreme.

### 8.6 Geometry, organisation and classification of rain fields
- ★ AghaKouchak, A., Nasrollahi, N., Li, J., Imam, B. & Sorooshian, S. (2011). Geometrical
  characterization of precipitation patterns. *J. Hydrometeor.* 12, 274–285.
  doi:10.1175/2010JHM1298.1 — shape indices of rain fields; the nearest precedent to
  classifying by excursion-set geometry.
- Haberlie, A. M. & Ashley, W. S. (2018). A method for identifying midlatitude mesoscale
  convective systems in radar mosaics. Part I: Segmentation and classification. *J. Appl.
  Meteor. Climatol.* 57, 1575–1598 (and Part II, tracking). — ML on radar mosaics, close in
  setup to ours.
- Brune et al. (2020); Janssens et al. (2021) — organisation indices; continuum rather than
  classes (in the §References).
- Wernli, H. et al. (2008). SAL — a novel quality measure for the verification of quantitative
  precipitation forecasts. *Mon. Wea. Rev.* 136. † (already used in eval)
- Davis, C., Brown, B. & Bullock, R. (2006). Object-based verification of precipitation
  forecasts. Part I (MODE). *Mon. Wea. Rev.* 134. †
- No work was found that classifies precipitation regimes by Minkowski functionals or
  Euler-characteristic curves (searched again 2026-09-28; they are standard in cosmology).

### 8.7 Weather-type (synoptic) classifications, for stratification
- Huth, R. et al. (2008). Classifications of atmospheric circulation patterns: recent advances
  and applications. *Ann. N. Y. Acad. Sci.* 1146, 105–152.
- Weusthoff, T. (2011). Weather type classification at MeteoSwiss — introduction of new
  automatic classification schemes. *Arbeitsberichte der MeteoSchweiz* 235. (GWT types; ask MCH
  for the daily series)
- Philipp, A. et al. (2010). Cost733cat — a database of weather and circulation type
  classifications. *Phys. Chem. Earth* 35. †
- Hewson & Pillosu (2021), Ferranti et al. (2015) — in the §References.

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
