# Digest F — research notes, method documentation, literature

Sources read in full: `RESEARCH_NOTES.md` (1,238 lines, started 2026-09-23), `DECISIONS.md` §13
(lines 339–372), `MINKOWSKI_DOCS.md` (377 lines), `README.md` (228 lines). Citations grepped in
`notes/*.tex` and `notes/references_reward_coupling.bib`. Nothing under `private/` was opened.
Section numbers (§x) refer to RESEARCH_NOTES unless prefixed (DECISIONS §, EXPERIMENTS §).
Items marked **Decision** in the source are open choices for the researcher, not settled.

Scope note from the file header: EXPERIMENTS §6 lists the macro steps M1–M5; RESEARCH_NOTES
develops M2, M3, M5 and the Study-1 weighting question of M4; §7 is radar data quality for M1,
§8 a reading list.

---

# Part 1 — Theory and research notes

## 1.1 Central hypothesis: geometry as the transferable handle (§1)

**In words.** Minkowski functionals describe the geometry of the precipitation field; the
geometry of a "normal" storm is similar to that of an extreme one; so a super-resolution model
trained or fine-tuned on the functionals learns geometry that carries over to extremes, making
it more robust to extremes it has not seen at that intensity.

**Why the functionals.** Hadwiger's theorem (2-D): area, perimeter, Euler characteristic span
every motion-invariant, continuous, additive functional of a set. Applied to excursion sets
{x ≥ u} across thresholds, γ(u) is a complete morphological description *within that
invariance class*. Two stated limits: no position information (DECISIONS §6); matching γ is
necessary but not sufficient for a realistic field (DECISIONS §2).

**Testable claims.**

| | Claim | Kind | Tested by |
|---|---|---|---|
| H1 | Excursion-set geometry of extremes, suitably normalised, resembles that of ordinary events | data property | M1 analysis, no training |
| H2 | A Minkowski-trained model ties high-threshold geometry to geometry learned at lower thresholds rather than memorising intensities | learner property | gradient/threshold analysis; γ residual by threshold |
| H3 | Beyond the training range, a Minkowski-trained model produces more realistic extreme structure | the claim that matters | M2 cut-off training |

"H3 without H1 would be luck. H1 without H3 would mean the geometry is there but the model does
not use it."

**Fit with what is known.**
- Pixel-mass bottleneck is an argument *for* the hypothesis (DECISIONS §5): thresholds
  ≥ 31 mm/h carry almost no gradient, so any tail gain must already come from lower-threshold
  geometry (H2). In the cut-off experiment, above-cut fixed-threshold terms are always empty in
  training.
- The area channel alone is a reweighted W₁ between marginal intensity laws (DECISIONS §13):
  distributional, not geometric. The geometric claim must rest on perimeter and χ. Direct M2
  ablation: A-only vs A+P+χ trained identically; if A-only transfers equally, the effect is
  distributional.
- Fixed physical thresholds cannot express "same geometry, higher intensity". A sample-relative
  threshold grid (fractions of coarse-input maximum, exceedance above a local quantile) would
  implement H1 in the loss. Design option for M2/M3 (**Decision**).

**Predictions stated in advance.**
1. (H1) Normalised γ curves collapse across intensity classes (normalisations: u / event peak;
   u as local quantile); likewise perimeter–area scaling and χ(u) shape. If not, the hypothesis
   needs a regime qualifier (§4).
2. (H3) On a cut-off split, i.i.d.→o.o.d. degradation is smaller with the Minkowski term
   (exceedance ratio, RL bias, peak ratio).
3. (H3, geometric) Gain comes from P and χ, not A alone.
4. (H1×H3) Skill on held-out extremes decreases with geometric distance to the nearest training
   event (§3 Q5); slope flatter for the Minkowski model.

## 1.2 What is an extreme? (§2, Decision)

| | Definition | o.o.d. means | H1–H3 prediction |
|---|---|---|---|
| E1 | Intensity extreme within a familiar regime | peak/rate beyond training range | hypothesis built for this |
| E2 | Novel event type (derecho/bow echo, medicane, back-building stationary MCS, orographic extreme in unrepresented region) | regime absent from training | transfer only if geometry shared; stress test that may fail honestly |
| E3 | Locally extreme relative to local/seasonal climate (ECMWF EFI / M-climate style) | beyond e.g. local 99.9th pct of 15-min rate for region and season | a normalisation of E1 or E2, not a third experiment |

- E3 is a choice of scale: 50 mm/h is routine in some Mediterranean/Alpine convection and
  exceptional over the North Sea. The enlarged OPERA archive's climatology can play ECMWF's
  M-climate role; also answers mm/h vs local-quantile cut.
- Unit of extremeness: pixel rate (15-min, 2 km), patch/tile max, or event total. Current tail
  protocol: pixel exceedances (POT) and patch maxima. *Update 2026-10-08:* pixel quantity is now
  the 1-h accumulation at 2 km (DECISIONS §21); E1–E3 thresholds re-derived on hourly sums; max
  15-min rate within each hour kept as an attribute (hail, short bursts).
- Natural reading: E1 primary, possibly E3-normalised, E2 secondary stress test reported
  whatever the outcome. **Decision.**

Building the unseen-extremes split (M2):
- Cut by event and in time, not by patch (current splits leak, DECISIONS §15). Events =
  space–time connected components above a level, or (day × region) blocks, with buffer hours.
- Screen first (M1): top of tail disproportionately artefacts; declutter punches holes in real
  cells > 150 mm/h.
- Choose cut from counts: illustrative u_c, to be re-derived on clean data — patch max around
  60–90 mm/h (fixed threshold) or ~99.9% (local quantile).
- Report how o.o.d. the *input* is, not only the target.
- Compare degradation, not absolute skill: i.i.d., o.o.d., ratio for MSE, Minkowski (A-only and
  full), competing losses once fixed (M4), FM.

## 1.3 What M1 must answer (§3)

Beyond quality screening (`scripts/data_quality/README.md`):
- **Q1 collapse (H1):** normalised γ curves by intensity class, region, season.
- **Q2 scaling:** perimeter vs area per threshold (isoperimetric ratio, fractal exponent); does
  the exponent depend on intensity?
- **Q3 topology:** χ(u) and component counts vs intensity; do extremes fragment differently
  (one organised core vs many cells)?
- **Q4 supply:** after screening, how many extreme events, where, when; is a held-out o.o.d.
  set viable, under E1 or E3?
- **Q5 analogues:** distance in normalised-γ space from each extreme to its nearest training
  event — the continuous "how unseen" axis for prediction 4.
- **Q6 regimes:** are extremes concentrated in few regimes; does γ collapse *within* a regime
  where it fails across?

## 1.4 Regime-aware Minkowski training, M3 (§4)

Idea: classify the precipitation regime from predicted functionals, prediction and LR input,
and make structural training regime-aware. Regime definition is open; start from ECMWF.

**What ECMWF and the literature do.**
- **ecPoint gridbox weather types** (Hewson & Pillosu 2021) — "the closest analogue". Purpose:
  IFS gridbox rainfall → point-rainfall PDFs (grid-to-point sub-grid variability at almost our
  scale ratio). Physics: large-scale rain low sub-grid variability, convective high; cell drift
  speed sets shape (streaks vs blobs). Governing variables in tree order: convective fraction,
  total precipitation, 700 hPa wind speed, CAPE, 24 h clear-sky solar radiation. Method: shallow
  expert decision tree, breakpoints where the forecast error ratio (obs − gridbox)/gridbox
  distribution changes (KS tests); 214 leaves, each ≥ ~200 cases; convective adjustment
  timescale, shear, orography tested but not all used. Transferable recipe: **regimes = classes
  with different sub-grid variability**.
- IFS `cp`/`lsp` split: model-scheme and resolution dependent, not observed; needs ERA5 input.
- Convective adjustment timescale τc (Done et al. 2006; used in research at DWD, Met Office):
  equilibrium vs non-equilibrium convection; needs CAPE + rain rate (ERA5). Secondary.
- Euro-Atlantic weather regimes (NAO±, blocking, Atlantic ridge; Ferranti et al. 2015):
  large-scale, low relevance for a 256 km patch; at most stratification metadata.
- Radar-based: Steiner, Houze & Yuter (1995) convective/stratiform separation (peakedness above
  background) on 2 km target/prediction → convective fraction (ecPoint's first variable
  analogue); organisation indices SCAI, COP, Iorg (compared on radar by Brune et al. 2020);
  caution from Janssens et al. 2021: cloud-pattern metrics form a continuum, hard labels may be
  artificial.
- Novelty: no established work found classifying precipitation regimes by Minkowski functionals
  or Euler-characteristic curves; search not exhaustive.

**ecPoint mapping to available variables.** convective fraction → Steiner fraction of target
(labels) or prediction; total precipitation → coarse-input intensity and wet fraction; 700 hPa
wind → advection between frames (±15 min fields already fetched) or ERA5 V700; CAPE → ERA5
(optional); solar radiation → date, hour, latitude; orography → DEM roughness; new → γ
descriptors (P/A slope, χ at mid thresholds, peakedness).

**Design options (Decision).**
- (a) regime-conditioned loss (threshold grid, λ or γ target depends on regime);
- (b) regime as model input (FiLM or conditioning token);
- (c) mixture of experts, one head per regime;
- (d) classifier on (LR input, prediction, predicted γ). Circularity: regime from the prediction
  lets the model move a sample into an easier regime — new reward hacking (DECISIONS §2).
  Mitigations: detached/frozen classifier; or regimes from input + DEM only, checked against
  target-derived regimes.
- Defining regimes the ecPoint way: breakpoints where the *sub-grid* error distribution changes
  (e.g. fine-max / coarse-block ratio, γ residual of a baseline); shallow tree.

**Regimes from what is available at inference (to explore).**
- Route 1 (preferred): analytic γ(u) on the coarse input + coarse intensity stats, wet fraction,
  DEM descriptors. Free of circularity. Truth for evaluation only: target-derived regimes
  (Steiner fraction, organisation indices, γ of target, MeteoSwiss GWT types §8.7). Questions:
  how much regime information survives 12.5× coarsening; which regimes stay separable; does
  γ(LR) add beyond plain coarse stats. Limit: coarse patch is 10×10 px; consider larger coarse
  context (neighbouring tiles).
- Route 2: γ(ŷ) of the prediction; circularity caveat; frozen classifier, detach, check against
  target regimes. Learned emulator's predicted γ is a third variant, lowest priority
  (DECISIONS §14).
- Outcome logic: Route 1 works → (a)–(c) well posed; only Route 2 → build into the loop with
  circularity controls; neither → regimes not recoverable at 25 km, itself a result for M5.

## 1.5 Study 1 weighting (§5, M4; full reasoning DECISIONS §16)

- **Homoscedastic (Kendall): not for MSE vs auxiliary losses.** Balances loss values (rejected
  by §9); assumes Gaussian likelihoods; absorbs non-zero floors (SSIM's 0.465); makes each
  weight an uncontrolled learned quantity. Fine for balancing the three Minkowski channels.
- **Optuna: not primary.** λ is 1-D per loss and bounded above by hacking; each trial a full run;
  selection metric becomes the result. Worth it only for joint searches (λ × anneal × tail-sample
  fraction) on short proxy runs with pruning and equal budget per loss.
- **Proposed calibrated bracket:** (1) gradient parity λ* (done, EXPERIMENTS §3); (2) train
  {λ*/3, λ*, 3λ*, 10λ*} per loss, fixed epoch budget, last checkpoint evaluated; (3) one
  pre-registered rule: best exceedance ratio with ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla;
  (4) 3 seeds at the chosen point, report whole bracket.
- Before any rerun: fix SSIM (fixed data_range in normalised space, smaller eps, or wet-window
  mask); log or remove optical-flow `teacher=target` fallback.
- FM: the four losses as rewards in `reward_finetune.py`, same bracket on `REWARD_WEIGHT`.
- Hourly data (2026-10-21, DECISIONS §37): Minkowski centre re-derived by gradient share (step 1a
  at initialisation: 8.5e-5; step 1b at the trained vanilla checkpoint decides), then the
  bracket. Literature status and systematic search still needed before any novelty claim:
  `notes/weight_selection_literature.md` (not in this digest's scope).

## 1.6 Open decisions (§6)

1. Definition of extreme E1/E2/E3 and unit (pixel rate, tile max, event).
2. Cut level and event definition for the o.o.d. split; both depend on M1 counts.
3. Fixed physical vs sample-relative thresholds in the loss.
4. Regime definition and integration (a)–(d); whether to bring in ERA5.
5. Screening policy: decided in principle (DECISIONS §17: clear errors only, keep imperfections,
   no labelled set). Open: rule set and thresholds, pixel masking vs tile rejection, declutter
   rule. Discuss with MCH first.
6. Archive years: 2012 much dirtier than 2024 in first audit; product breaks late 2015,
   2017-09-29, 2024-07-05. After 2017-09-29 still ~8 years of 15-min data.

## 1.7 Radar data quality (§7)

### QIND (§7.1)
- OPERA per-pixel total quality index in [0, 1]. ODIM task name `pl.imgw.quality.qi_total` =
  IMGW's `qi_total` module in BALTRAD, from RADVOL-QC.
- Built (Szturc et al. 2011; Ośródka et al. 2014, 2022) from quality factors (radar technical
  parameters, distance/beam broadening, beam height, blockage, attenuation, QC detections of
  sun/emitters, specks, clutter), each mapped to QIᵢ ∈ [0,1], combined multiplicatively.
- ODYSSEY composite: weighted average of contributing radars by quality index, distance, and an
  exponential index of inverse beam altitude (Saltikoff et al. 2019). NIMBUS (von Lerber et al.,
  EMS 2023) composites lowest elevation only; switch visible as `QIND_RATE` → `RATE` on
  **2024-07-05** (first hours still `QIND_RATE`). NIMBUS file = single-time `PPI` at nominal
  time; ODYSSEY file covers a 15-min window (e.g. 11:50–12:05); same 2 km grid.
- Not an echo-type detector; mostly a priori geometry-driven reliability. Measured: 2026 live
  composite, pixels ≥ 31 mm/h twice as often below QI 0.8 (41.7% vs 21.0%); 2018-06-15T16:30,
  pixels > 500 mm/h mean QI 0.267 vs 0.282 overall; QIND defined on ~7–11% of pixels vs ~43–50%
  for rate (cause unknown). QIs not harmonised across services (Einfalt et al. 2010).
- Recommendation: covariate or soft weight; **not the artefact screen**.
- Later gauge results (§7.4, §7.4c): under ODYSSEY points the wrong way (unconfirmed tail pixels
  higher QIND, median 0.90 vs 0.20; AUC 0.26), under NIMBUS no signal (AUC 0.53); scale changes
  with product.

### Agency cleaning chains (§7.2)
Most of the chain runs on single-radar volumes before compositing; we inherit it. 15 steps
tabulated (step / examples / usable on composite):
1. Doppler/static clutter filtering (MeteoSwiss Germann 2006, 2022; Met Office Nimrod Harrison
   2000; Météo-France Tabary 2007) — upstream; residual static clutter: yes.
2. Texture/gradient tests (Gabella & Notarpietro 2002 in wradlib; Steiner & Smith 2002;
   Lakshmanan 2007) — yes (EURADCLIM applies Gabella to OPERA).
3. Emitter/RLAN/sun-spike removal (bRopo, Peura 2002; RADVOL-QC SPIKE; Saltikoff 2016) — yes,
   by geometry.
4. Polarimetric non-met classification (Ośródka & Szturc 2022; Figueras i Ventura & Tabary 2012) — no.
5. Beam-blockage correction (OPERA since late 2015) — no, kept imperfection.
6. C-band attenuation correction — no; biases the tail.
7. VPR / bright band — no.
8. Calibration monitoring — partially (radar-wide jumps detectable).
9. Z–R: OPERA Marshall–Palmer (EURADCLIM: Z = 200 R^1.6) — fixed; why rates aren't gauge-accurate.
10. Quality-based compositing (Jurczyk 2020; ODYSSEY/NIMBUS) — upstream.
11. Satellite cloud mask (OPERA since late 2015; EURADCLIM CLAAS-2 7×7 neighbourhood) — yes if
    satellite data added.
12. Long-term static clutter (`wradlib.clutter.histo_cut`, 50-class histogram, classes < 5% of
    mode, iterated) — yes; our `clutter_climatology.py` analogue.
13. Availability (EURADCLIM ≥ 83.3%) — yes.
14. Gauge adjustment (CombiPrecip, RADOLAN/RADKLIM, EURADCLIM, Park 2019) — out of scope for SR;
    relevant for evaluation.
15. Climatological reprocessing (RADKLIM, Kreklow 2020) — model for a "climate version".

EURADCLIM (Overeem et al. 2023) is the single most relevant precedent: same OPERA 15-min 2 km
composite (2013–2020), steps 2, 11, 12, then gauge adjustment; own caveats: persistent non-met
echoes, radar-wide constant-source failures, rings/radials remain, outliers "limit the
applicability ... especially for use in extreme value modeling". Product breaks: late 2015
(beam blockage + cloud mask), **2017-09-29 08:52 UTC** (log dBZ → linear Z compositing),
**2024-07-05** (ODYSSEY → NIMBUS), plus growing radar count (~138 average 2013–2020).

### Clear errors vs kept imperfections (§7.3; DECISIONS §17)
Clear errors: residual static clutter, emitter rays/rings, isolated spikes, physically
impossible rates (seen 13,072 and 64,842 mm/h), radar-wide constant-signal failures,
single-frame appearances, rain under clear sky (if satellite), nodata-dominated tiles. Kept:
blockage/range underestimation, attenuation shadows, seams, drizzle speckle, mild residual
clutter. Real extremes are also "anomalous" → prefer geometry/persistence/impossibility rules
over intensity rules; report rejection rates per intensity bin.

### Proposed cleaning plan (§7.4b, 2026-09-28, not decided)
Premise: flag rate rises with intensity. Steps: (1) fix DEM orientation; (2) missing not zero
(NaN + valid mask); (3) static masks per product period (≥ 31 mm/h in > 1% of steps, ~100×
climatology; e.g. Weissfluhgipfel tile supplying the three largest maxima); (4) geometry via
radar sites (rays through, rings around; reject tile for rays, rings, radar-wide failures);
(5) spatial support (mask isolated pixel); (6) temporal support (no precursor/successor at
±15 min; check vs intensity); (7) plausibility bound set with MCH; (8) per-radar per-period
exclusion; (9) QIND soft weight only; (10) validate surviving tail against independent evidence
(per-bin rejection, lightning, ECA&D gauges, CombiPrecip, `patch_gallery.py --select top_max`);
(11) then define "extreme" regionally (E3) on screened data.

### Post-processing status and release (§7.4c, 2026-10-02, updated 10-03)
Implemented in `src/data/cleaning.py`, `tests/test_cleaning.py`. Status per step:
1 fixed 2026-09-28; 2 partly (repair by neighbourhood value, nothing zeroed above 150 mm/h, but
removal still by tile; partially covered tiles excluded, `notes/events.md` §5); 3 done per
calendar year; 4 rays done (thin components ≥ 80 km aligned with a radar within 250 km), rings
climatological audit flag — gauges: ring pixels real rain below 31 mm/h, never at ≥ 89 →
repair high values, don't reject; radar-wide failures not done; 5 done, right for both products
(AUC vs untouched 0.82–0.94 ODYSSEY, 0.90–0.94 NIMBUS; NIMBUS 32–44% presence = light rain
nearby); 6 audit flag, ~1% corroborated in every bin, cleanest rule, fires almost only under
ODYSSEY; 7 provisional 500 mm/h, mask pixel not tile (rejected-tile pixels 67–79% corroborated
below 89 under ODYSSEY, 91–97% under NIMBUS); 8 14 consistently bad radars (after fixing three
ranking bugs), not validatable (only Berlin in DE/CH); 9 QIND does not discriminate; 10 done for
DE/CH with 1,735 DWD + SwissMetNet 10-min gauges, 12.1M station-frames (`notes/data_quality_assessment.pdf`
Part II); 11 not done. Also done: product-break analysis (DECISIONS §18, `era_gap.py`), per-bin
rejection rates, reproducible pipeline from the public archive.
- Rejection light: 0.2–0.4% of tiles per bin 1–500 mm/h, 40% above 500.
- 150–500 mm/h untouched pixels: 96% corroborated NIMBUS (median gauge 28 mm/h), 69% ODYSSEY
  (median 14). ODYSSEY radar/gauge exceedance ratio flat ≈ 0.5 to 53 mm/h, to 89 with temporal
  and ring flags; residual excess ~2.5 at ≥ 150. NIMBUS too heavy from ~31 (3.6 at 89); test
  with 1-min gauges.
- Release options: (i) reprocessed OPERA rate archive 2012–2026 (~1.3 TB) — **not worth it
  alone** unless KNMI or MCH co-own; (ii) **ML benchmark** for km-scale downscaling of extremes
  over Europe (v2 family, events, NIMBUS shift set, flags, clutter masks, code, baselines) —
  **worth it**; existing radar ML datasets are national (MeteoNet, RainNet/RADOLAN, US SEVIR) or
  OPERA crops for nowcasting (Weather4cast); release part ~few weeks; venues ESSD or a
  datasets-and-benchmarks track. Minimum before release: step 10 on at least CH, top-tail
  gallery, rule 6 applied, 2013 events and partial-coverage fix, plausibility bound with MCH;
  ask Daniele and Lionel about validation/co-authorship.

### What v3's screen misses; tail fit as audit (§7.4d, 2026-10-06)
From POT threshold study (`scripts/data_quality/pot_threshold.py`, `OPERA/quality_v3/pot_threshold/`;
`fig/tail_suspects.png`, `fig/tail_suspects_4862.png`):
- Ceiling 364.63 mm/h (= 64.0 dBZ under Z = 200 R^1.6), Valjevo (Serbia), 80 rain days Feb–Nov
  2023; 60,145 train px (+14,480 val, 23,520 test).
- Radar-wide failure, Torrejón de Velasco (Madrid), 2018-04-29 all day; disk at 60–340 mm/h, raw
  cores to 10^4; 16% of all train pixels > 89 mm/h. Footprint repair created a plateau of 4,865
  px at exactly 88.27 mm/h (2018-04-29 06:15, r256 c256) — "launders the artefact".
- Ceiling 48.62 mm/h (= 50.0 dBZ), tile r640 c1408 (Bobohalma, 102 km), 7 days Nov 2019; 98 tiles.
Every v3 rule assumes small/local/transient artefacts. Proposed additions: (1) repeated-value
test (per-tile max count of one value ≥ 31 mm/h: median 1, p99 3, p99.9 24 over 588,617 train
tail tiles; flag ≥ 50 = 0.069% of tail tiles, 11% of their pixels; NIMBUS 0.5 dBZ quantised →
excess over baseline); (2) ceiling table (count > ~50× median of neighbouring ±0.5 dB bins;
364.63 is ~7,500×); censored, mask or reject; (3) radar-disk failure (wet fraction, mean rate,
R²_r = 1 − Var(z − <z>_θ(r))/Var(z), cross-radar check); (4) guard the repair (> 500 px or ring
median ≥ 31 → escalate; plateau check); (5) episodic radar-day ranking. Tail fit as a
dataset-level audit: ξ(u) drifts, day-block bootstrap widens, quantiles stick (364.63 was the
99% quantile of exceedances for every u from 25 to 115); cannot separate failure from real
extreme day (Elvira/Friederike).

### Screen reviewed for hourly targets (§7.4e, 2026-10-08)
- **MLCast** (EUMETNET E-AI WG6; mlcast.org; 16 repos read, `mlcast` commit 85f92b7): no
  artefact screen; NaN budget (`max_nan` 10,000 per 24×256×256), no time gaps, non-finite loss
  mask, importance sampling by mean rain; relies on national products ≤ 1 km (RADKLIM, DMI,
  IT-DPC SRI, Nimrod, RMI RADCLIM, Météo-France; MeteoSwiss converter outside catalogue); OPERA
  only in issues #2, #48; PR #17 importance sampler may pick clutter; MCH converter maps
  `undetect` to NaN, BE to 0. Cannot confirm or refute our rules.
- **EURADCLIM** screen: 1-h total only with all four frames; 24-h ≥ 83.3% hours; Gabella (fewer
  than 6 of 5×5 neighbours within 6 dB; area/perimeter < 1.3), flagged → 0; satellite veto (7×7
  cloud-free → 0); `histo_cut` static clutter with IDW replacement where lower; ~7,700 ECA&D
  gauges; v3.0 (per KNMI page via search agent, not re-checked) sets 15-min > 150 mm/h to 0 and
  caps 1-h at 300 mm. Compared with v4: Gabella ≈ softer spike rule; satellite veto we lack;
  clutter on annual accumulation vs ≥ 31 frequency; nothing on persistence/rays/rings/ceilings/
  radar-wide failures.
- Principle: decide with a long window, repair per frame, then sum. Transient high-amplitude
  errors diluted by 4; persistent moderate errors (20–80 mm/h) add up. Rain persists along
  motion (Lagrangian), clutter in place (Eulerian); test weak at near-zero motion (Alex,
  Maggia/Valais, Boris; Münster, Valencia DANA) → report flag rate vs motion speed.
- Rule-by-rule table: drizzle none; static clutter fallback never 0 (Elvira 2016-06-05) + P1;
  spike, footprint keep; ray → multi-frame ray test (fires on fronts, Poland derecho 2023-06-20);
  ring add 31–89 persistent-peak repair (gauges saw rain under 38% of ring pixels); unsupported
  keep, later advected; ceiling stays on 15-min, repair at ≥ 5 px in rain; repeated value repair
  (rejected real N-Italy storm 2023-07-24); guard ok once ray fixed; radar-disk require wet-
  fraction jump vs both neighbours (4/20 flags real storms; DECISIONS §23); radar-day exclusion
  unchanged; completeness 4/4 frames; QIND out.
- New accumulation screen P1–P5 (P1 local-peak persistence: artefacts 38–74 frames, real cells
  4–12, event-day rays 20–39; P2 radar-anchored lines; P3 flicker; P4 daily-total contrast,
  shape only; P5 Lagrangian residual).
- Better hourly field: screen; advection-corrected accumulation (pysteps example, Lucas-Kanade;
  core moves 4–9 px between frames; beads add components and perimeter; residual decorrelation
  0.29 at one frame; pysteps not installed in `dl-stable`); gap filling; sub-hourly attributes;
  motion field computed once.
- Hourly references for DECISIONS §22: DE/CH gauges; RADKLIM RW (1 km, 2001–2025); COMEPHORE
  (1997–2026); CombiPrecip (2005-); EURADCLIM v3.0 (2013–2023, not independent); MTG-I LI (Jul
  2024-); GPM IMERG Final (0.1°, 1998–2025); CML. Not useful: ERA5-Land, CERRA, E-OBS, EMO-1,
  OPERA `ACRR` (regression test only).
- Get it right once: store evidence, decide at split time; keep raw day stores; calibrate before
  Phase 3; freeze in versioned config. Cannot defer: hour formation and which frames get
  repairs (decided before Phase 4).

### Other km-scale hourly datasets (§7.4f, 2026-10-08)
Survey by three search agents (≤ ~2.5 km, ≥ hourly; ranking is Claude's inference):
RADKLIM RW/YW (DE, 1 km/1 h, YW 5 min, 2001–2025, open ~8 GB, peaks smoothed [snippet], 2021
NetCDF faulty); COMEPHORE (FR, 1997–2026, Licence Ouverte 2.0, ~73 GB, break 2007, grid change Aug
2024); NOAA MRMS v12 (0.01°, 2020-10 on, AWS `noaa-mrms-pds`, mask filled pixels with RQI);
CombiPrecip (CH, 2005-, on request; RZC saturates at 118 mm/h); BoM Rainfields3 (500 m/15 min,
2019-, CC BY-NC 4.0); EURADCLIM v3 (CC BY 4.0, >150 → 0, 1-h cap 300 mm); ECMWF RODEO-ML OPERA
(2013–2024, **clipped at 12.5 mm/h**); JMA (paid; gauge imprints); AORC (800 m but not km-scale,
≥ 4 km structure); KNMI RAC MFBS, Nimrod, IT-DPC-SRI, RMI RADCLIM, INCA (various caveats). Not
open: CHMI, DMI, AEMET, IMGW, KMA, CMPAS, Canada (HRDPA 6-hourly), QPESUMS.
- FBK `opera-plus-it-vmi` (checked 2026-10-09): column-max reflectivity dBZ, 1 km EPSG:3035, 5 min,
  from 2024-07-01, CC BY-SA 4.0, no documented QC; not a target. Reusable: TAASRAD19 per-pixel
  Mahalanobis outlier mask (Franch et al. 2020); `italian-radar-dpc-srt1-adj` gauge-adjusted
  hourly over Italy from 2018-07. Questions for FBK (franch@fbk.eu).
- Standard generative-downscaling practice: mostly daily (WassDiff MRMS, CPMGEM UKCP 8.8 km,
  Rampal et al. NZ, Wetherell flow matching 8→2 km daily underestimating upper tail); closest
  hourly: spateGAN-ERA5 (ERA5 → RADKLIM-YW 2 km/10 min); CorrDiff Taiwan precip is a
  reflectivity surrogate; CPM output (UKCP Local 2.2 km, CORDEX FPS Alps, 1 km ICON) as
  perfect-model complement.
- Inference: no public pan-European uncapped cleaned hourly km-scale dataset (RODEO clips,
  EURADCLIM zeroes) → v4 fills the gap; RADKLIM vs OPERA over DE isolates product shift; multi-
  region design makes O3 concrete.

## 1.8 MCH discussion points (§7.4; people: Daniele Nerini, Lionel Moret)

Updated 2026-10-04 from the gauge validation. Asked of MCH:
- Screen: our rules (static clutter, spikes, rays, rings, no-precursor cells, > 500 mm/h cores);
  artefacts we miss; Gabella-type or satellite filters?
- Repair not rejection: MCH practice for ML sets?
- QIND inverted under ODYSSEY, no signal under NIMBUS: known, why?
- ODYSSEY reads ~half the gauge (tail shape to ~89 mm/h): factor ~0.5 expected from Z–R and
  area-vs-point?
- NIMBUS above gauges increasingly with intensity (1.15× at 31, 3.6× at 89; 96% rain at gauge):
  known? decides whether NIMBUS can be a test set.
- Still open: upper plausibility bound (we use 500 mm/h, repair down to 150); CombiPrecip and
  POH/MESHS 2012–2026 (portal keeps 14 days); product-break handling.
- Extreme threshold u = 31 mm/h: a point of the loss's log grid (DECISIONS §1), POT level "for
  estimability, not physics" (DECISIONS §10); also sets FSS levels, v2 `extremes` subset, tail
  stratum, clutter hot-pixel rule. Warnings: ξ_obs flips sign between 31 (+0.18) and 53 (−0.19);
  no threshold diagnostics; capped artefact-laden fit data; no declustering. Ask MCH's definition
  for radar rates; regional/seasonal or product-specific (NIMBUS 1.44× more pixels ≥ 31 mm/h).
  Rate vs accumulation (Valencia: 28.6 mm/h radar pixel vs 184.6 mm/h gauge-hour) — *answered
  2026-10-08: 1-h accumulation, decided with Daniele Nerini (DECISIONS §21)*; remaining: plain or
  advection-corrected sums from 5-min data. POT practice (MRL, parameter stability,
  declustering, pixel vs event maxima, return levels). Radar tail biases (Z–R, hail > ~55 dBZ,
  hail cap).
- Testing o.o.d. (for Lionel Moret, 2026-10-02): v2 test is not o.o.d. (same climate/product/
  regions; event set of 25 storms in test + 2 in NIMBUS once v3 built, all types in training;
  NIMBUS moves the target, DECISIONS §18). Options O1 intensity cut-off (primary, H3); O2
  held-out storm type (needs labels beyond 26 catalogued events); O3 held-out region (Alps,
  Mediterranean, Switzerland); O4 held-out period (train 2012–2021, test 2022–2024); O5 product
  shift (NIMBUS, CombiPrecip); O6 input-source shift (ICON-CH1/2, climate model); O7 graded "how
  unseen" on every test set. Proposed core: O1 + O7 + O5; O2 if labels; O3/O4/O6 if MCH sees them
  as operational. Questions on operational shift, trustworthy tail cut, Switzerland held out,
  TRT/POH/MESHS/lightning labels, independent truth, ICON interest, MCH's own nowcasting
  evaluation.
- README acknowledgments also thank D. Nerini, D. Domeisen, V. Chavez, L. Moret, O. Miralles
  (conceptualisation) and L. Räss (technical); SNSF Grant No. 10001754 ("RobustSR").
- §7.4c: ask Daniele and Lionel whether MCH would validate the Swiss part or co-author.
- §7.4e: Daniele Nerini suggested checking rules against MLCast.

## 1.9 The method as documented (MINKOWSKI_DOCS.md)

Project reference: Quarenghi, Cotsakis, Beucler, arXiv:2604.11422; the present work replaces
that paper's learned emulator with analytical differentiable functionals. Files:
`src/losses/minkowski.py`, `src/data/gamma.py`, `src/data/datasets.py`, `config.yaml`,
`validate_phase1.py`, `tests/test_analytical_minkowski.py`.

- **Quantities.** Q = 21 thresholds; excursion set {x ≥ u}; A(u) = a_px·|{x ≥ u}| with a_px =
  4 km²; P(u) in km; T(u) = χ = β₀ − β₁ or β₀ (`TOPOLOGY_MODE`). Loss
  M = Σ_c ∫ |log γ̂_c(q) − log γ_c(q)| dq; log1p for A, P, β₀; signed_log1p for χ.
- **Units.** Model output log-normalised [0,1], x_phys = relu(expm1(x_norm·max_val)); diffusion
  [−1,1]; functionals and targets in physical mm/h; thresholds = climatological quantiles of the
  wet-pixel CDF; integration over q ∈ [0,1]. Must evaluate on `WaterScaler.to_physical_torch`.
  Constants: PATCH_SIZE=128, DOWNSCALING_FACTOR=12.5 (25 km → 2 km), PIXEL_SIZE_KM=2.0,
  DRIZZLE_THRESHOLD=0.1 mm/h, QUANTILE_LEVELS 21 levels 0.01…0.99.
- **D1** trapezoid over quantile axis (physical-threshold integration up-weights the tail
  implicitly); alternatives physical, uniform, learned (`HomoscedasticMinkowskiLoss`). Conf. 0.9.
- **D2** bug fix: excursion sets at physical thresholds (previously QUANTILE_LEVELS passed as
  mm/h thresholds).
- **D3** soft sets s = σ((x − u)/τ), τ = max(u·tau_factor, tau_min), tau_factor 0.1, tau_min
  1e-3; annealed by `anneal_factor`.
- **D4** area STE: hard count forward, sigmoid gradient backward (Bengio et al. 2013); FD checks
  invalid on STE, use `area_mode="soft"` (soft agrees with autodiff to 5×10⁻⁷).
- **D5** perimeter isotropic Cauchy–Crofton, P = ℓ(w_ax(T_h+T_v) + w_di(T_d1+T_d2)), ℓ = 2 km,
  (0.473215, 0.217716), weights calibrated in-repo; replaces anisotropic central-difference TV;
  within ~2% on disks (radii 20–78 px), ellipse, 45° diamond; ~1% vs skimage
  `perimeter_crofton(directions=4)`. Target must use identical estimator (208.82 vs hard 208.83
  km); marching squares overestimates ~5%.
- **D6** topology: `euler` — χ = V − E_x − E_y + F with Gödel-min, dense gradients; target exact
  4-connected `euler_number(connectivity=1)`, not GUDHI β₀ − β₁ (8-connected and persistence-
  filtered; on a random field 250 vs −66; cause of the paper's low topology R²). `b0` — dim-0
  superlevel persistence via GUDHI critical pixels, custom autograd `_Dim0PersistencePairs`,
  soft count Σ g_k σ((b_k−u)/τ) σ((u−d_k)/τ) with persistence gate matching `thresh_b0`;
  exact (FD 1.6×10⁻⁹) but sparse gradient; annulus β₀ = 1, χ = 0. Recommend `b0` for hole-rich
  fields counting cells.
- **Target–loss consistency**: `compute_gamma_matrix` returns (5, Q) `[A, P_crofton, B0, B1,
  chi_exact]`; offline targets must be regenerated.
- **Validation**: per-functional R² (expect R²_A ≈ 0.99), FD gradient, localisation ρ within
  ±3τ, unit tests 12/12 (torch 2.12 CPU, scikit-image 0.26, GUDHI 3.12).
- **Limitations**: β₀ sparse and CPU PH; χ cancellation → fp32 (conf. 0.6); STE FD invalid;
  χ ≠ β₀.
- Note: the reward-coupling tex note describes the current loss with **Q = 15** thresholds
  evenly spaced in ξ = log u, hourly levels matched to 15-min ones by exceedance frequency, i.e.
  MINKOWSKI_DOCS (Q = 21 quantile levels) appears older than the current configuration.

## 1.10 README.md

Paper: *Emulating Non-Differentiable Metrics via Knowledge-Guided Learning: Introducing the
Minkowski Image Loss*, Filippo Quarenghi, Ryan Cotsakis, Tom Beucler, *Environmental Data
Science*, 2026, CUP, arXiv:2604.11422. Abstract: "differentiability gap"; two methods —
analytic approximation via temperature-controlled sigmoids and continuous logical operators,
and learned Lipschitz-regularised CNN surrogates with hard geometric constraints; unconstrained
baselines produce physically impossible fields in up to 7.8% of cases. Emulator heads: area
monotonic (softmax → reverse cumsum), perimeter isoperimetric P = √(4πA)(1 + softplus(r)),
topology softplus (β₀) or linear (χ). Data: OPERA 2 km, 15 min, Aug 2023–Oct 2024, >1M 128×128
patches; chronological splits train Aug 2023–Jun 2024, val Jul 2024, test Aug–Oct 2024 (note:
DECISIONS §15 says the current splits are actually random at patch level — README and DECISIONS
disagree). Config defaults: `TOPOLOGY_MODE` "euler", `MINKOWSKI_TARGET_WEIGHT` 0.0,
`MINKOWSKI_WARMUP_EPOCHS` 5, `TRUST_TAU` 0.005725. README points readers to DECISIONS,
EXPERIMENTS, CLAUDE, MINKOWSKI_DOCS (RESEARCH_NOTES not listed). Repo tree still labelled
`Mink-DDPM/`, env `mink-ddpm`; install URL has placeholder `<your-handle>`.

---

# Part 2 — Literature review

Legend for verification status as stated *in the source*: [V] = RESEARCH_NOTES says verified
2026-09-28 by search or URL opened; † = source marks "from memory, re-verify"; [M] = source says
author lists/venues from memory (gradient_audit); [CITE?] = placeholder in source, no reference
given; [Crossref] = only DOI metadata opened per source. Files: RN = RESEARCH_NOTES.md, MD =
MINKOWSKI_DOCS.md, D13 = DECISIONS §13, RM = README.md, RC = notes/minkowski_reward_coupling.tex
(+ .bib), GA = notes/gradient_audit.tex. Entries are reproduced as written; no completion from
memory. **Incomplete** flags where the source omits venue/title/authors.

## 2.1 Minkowski functionals, integral geometry, topology

- Quarenghi, F., Cotsakis, R., Beucler, T. (2026). *Emulating Non-Differentiable Metrics via
  Knowledge-Guided Learning: Introducing the Minkowski Image Loss*. arXiv:2604.11422 (cs.LG);
  *Environmental Data Science*, CUP. — MD, RM. The predecessor paper; defines γ and the loss.
- Schneider, R., Weil, W. *Stochastic and Integral Geometry*. Springer, 2008. — MD. Foundation
  of functionals and Crofton perimeter.
- Hadwiger's theorem — RN §1, MD §1 (no reference given); RC `\citeq{Hadwiger's characterisation
  theorem}` [CITE?]. **Incomplete.**
- Cauchy–Crofton formula — RC `\citeq{Cauchy--Crofton formula}` [CITE?]. **Incomplete.**
- Carrière, M., Chazal, F., Glisse, M., Ike, Y., Kannan, H., Umeda, Y. *Optimizing persistent
  homology based functions*. ICML 2021, PMLR 139:1294–1303, arXiv:2010.08356. — MD. Critical-
  simplex gradient for `b0`.
- Maria, C., Boissonnat, J.-D., Glisse, M., Yvinec, M. *The GUDHI Library: Simplicial Complexes
  and Persistent Homology*. ICMS 2014, doi:10.1007/978-3-662-44199-2_28. — MD. Cubical PH.
- Edelsbrunner, H., Harer, J. *Computational Topology: An Introduction*. AMS, 2010. — MD.
- van der Walt, S. et al. *scikit-image: image processing in Python*. PeerJ 2:e453 (2014),
  doi:10.7717/peerj.453. — MD. `perimeter_crofton`, `euler_number`.
- "D. Legland, K. Kiêu, M.-F. Devaux; and T. Lehmann, D. Legland (*Insight Journal*, 2012)" —
  MD; in D5 text "Lehmann & Legland, *Insight Journal* (2012)". N-D run-length Crofton. **Incomplete
  (no titles; source: "DOI not verified here").**
- Bengio, Y., Léonard, N., Courville, A. (2013). *Estimating or Propagating Gradients Through
  Stochastic Neurons for Conditional Computation*. arXiv:1308.3432. — MD (STE area); RC
  `\citeq{Bengio, Léonard & Courville 2013, straight-through estimator; not opened}` [CITE?].
- AghaKouchak, A., Nasrollahi, N., Li, J., Imam, B. & Sorooshian, S. (2011). Geometrical
  characterization of precipitation patterns. *J. Hydrometeor.* 12, 274–285.
  doi:10.1175/2010JHM1298.1 — RN §8.6 [V]. Nearest precedent to classifying by excursion-set
  geometry.
- Lovejoy, S. & Schertzer, D. (2013). *The Weather and Climate: Emergent Laws and Multifractal
  Cascades*. Cambridge UP. † — RN §8.2. Scaling view; perimeter–area question (Q2).
- Novelty claim (RN §4, §8.6): no work found classifying precipitation regimes by Minkowski
  functionals or Euler-characteristic curves (searched again 2026-09-28; standard in cosmology).

## 2.2 Precipitation downscaling and super-resolution (as named, no full refs)

All in RN §7.4f "Standard practice in generative downscaling"; names only, **incomplete**:
- WassDiff (MRMS daily).
- CPMGEM (UKCP 8.8 km daily).
- Rampal et al. (NZ, daily).
- Wetherell (flow matching 8 → 2 km daily, underestimates the upper tail).
- spateGAN-ERA5 (ERA5 → RADKLIM-YW 2 km / 10 min, validated on MRMS and Australian radar) —
  closest hourly km-scale observational precedent.
- CorrDiff (Taiwan "precipitation" is a reflectivity surrogate from radar-assimilating WRF).
- Convection-permitting outputs: UKCP Local 2.2 km, CORDEX FPS Alps, 1 km ICON with radar
  nudging (perfect-model complement).

## 2.3 Generative models, flow matching, reward fine-tuning, extreme-aware learning

- Clark, Kevin; Vicol, Paul; Swersky, Kevin; Fleet, David J. (2024). *Directly Fine-Tuning
  Diffusion Models on Differentiable Rewards*. ICLR, arXiv:2309.17400 (`clark2024draft`). — RC +
  bib. DRaFT-K, basis for using structural distance as a reward back-propagated through sampler.
- Arjovsky, Martin; Chintala, Soumith; Bottou, Léon (2017). *Wasserstein GAN*. arXiv:1701.07875
  (`arjovsky2017wgan`). — RC + bib. Horizontal (transport) distance gives usable gradient where a
  vertical comparison saturates; analogy for the area term as a horizontal comparison at 15 ranks.
- Flow matching / rectified flow, marginal velocity theorem — RC [CITE?]. Supports the
  marginal-preservation property the anchor fix relies on. **Incomplete.**
- Chang & Sapsis (2026). *Extreme Event Aware (η-) Learning* / "Extreme Event Aware Learning".
  doi:10.1038/s41467-026-76811-x. — D13, RN References ("see DECISIONS §13"), GA references and
  text ("contrary to the prescription in DECISIONS.md §9 and in Chang & Sapsis"). **Incomplete
  (no first names/venue spelled out; title differs slightly between files).** Detailed in 2.11.

## 2.4 Multi-task loss weighting and optimisation

- Kendall, A., Gal, Y., Cipolla, R. *Multi-Task Learning Using Uncertainty to Weigh Losses for
  Scene Geometry and Semantics*. CVPR 2018, doi:10.1109/CVPR.2018.00781. — MD (basis of
  `HomoscedasticMinkowskiLoss`); GA [M] (uncertainty weighting analysis: optimum σ_k² = L_k,
  effective weight 1/(2(f+ℓ_k)); SSIM floor 0.465 dominates). RN §5 "Homoscedastic (Kendall)".
- Chen, Z., Badrinarayanan, V., Lee, C.-Y., Rabinovich, A. GradNorm. *ICML* 2018. — GA [M].
  Dynamic parity; rejected because the right operating point is loss-specific (Minkowski works at
  r ≈ 2–5, hacks at ~10×).
- Sener, O., Koltun, V. Multi-task learning as multi-objective optimization. *NeurIPS* 2018. —
  GA [M]. MGDA encodes no preference; MSE is primary.
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., Finn, C. Gradient surgery for
  multi-task learning. *NeurIPS* 2020. — GA [M]. PCGrad does nothing at cos ψ ≈ 0.
- Kingma, D. P., Ba, J. Adam. *ICLR* 2015. — GA [M]. Adam/AdamW geometry section.
- Loshchilov, I., Hutter, F. Decoupled weight decay regularization. *ICLR* 2019. — GA [M]. AdamW.
- Mandt, S., Hoffman, M. D., Blei, D. M. SGD as approximate Bayesian inference. *JMLR* 18, 2017.
  — GA [M]. (Only in reference list; in-text use not located by grep.)
- Optuna — RN §5, RM (tool, no reference).
- `notes/weight_selection_literature.md` holds the weighting literature status (out of scope).

## 2.5 Proper scoring rules / energy score

- Gneiting & Raftery 2007, energy score — RC `\citeq{...}` [CITE?]. ES(P, y) = E_P d(G, y) −
  ½E_P d(G, G'). **Incomplete.**
- "energy distance is a metric on distributions with finite mean, Euclidean case" — RC [CITE?];
  used to say the Euclidean energy score is strictly proper on the joint but its gradient does
  not split by threshold. **Incomplete.**

## 2.6 Extreme value theory, verification and evaluation metrics

- ECMWF Forecast User Guide: Extreme Forecast Index / Shift of Tails, M-climate.
  https://confluence.ecmwf.int/spaces/FUG/pages/673551296/Section+8.1.9.2+Extreme+Forecast+Index+-+EFI
  — RN References, §2 (E3 definition).
- Wernli, H. et al. (2008). SAL — a novel quality measure for the verification of quantitative
  precipitation forecasts. *Mon. Wea. Rev.* 136. † — RN §8.6 ("already used in eval").
- Davis, C., Brown, B. & Bullock, R. (2006). Object-based verification of precipitation
  forecasts. Part I (MODE). *Mon. Wea. Rev.* 134. † — RN §8.6.
- Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. Image quality assessment: from error
  visibility to structural similarity. *IEEE Trans. Image Process.* 13, 2004. — GA [M]. SSIM, and
  its unreachable floor (0.465) on this data.
- El Hachem, A. et al. (2022). Technical Note: Space-time statistical quality control of extreme
  precipitation observations. *HESS* 26, 6137-6146. — RN §7.5 (QC of extremes analogue).
- Not found in any in-scope file: a formal reference for FSS, GPD/POT methods (mean-residual-life,
  parameter stability are named in RN §7.4 as practices only), or return levels.

## 2.7 Radar quality, OPERA, cleaning (RN §7.5 unless noted; [V] unless †)

OPERA and the quality index:
- Szturc, J., Ośródka, K. & Jurczyk, A. (2011). Quality index scheme for quantitative
  uncertainty characterization of radar-based precipitation. *Meteorol. Appl.* 18.
  doi:10.1002/met.230 — QIND construction.
- Ośródka, K., Szturc, J. & Jurczyk, A. (2014). Chain of data quality algorithms for 3-D
  single-polarization radar reflectivity (RADVOL-QC system). *Meteorol. Appl.* 21.
  doi:10.1002/met.1323 — RADVOL-QC.
- Ośródka, K. & Szturc, J. (2022). Improvement in algorithms for quality control of weather radar
  data (RADVOL-QC system). *Atmos. Meas. Tech.* 15, 261–277. doi:10.5194/amt-15-261-2022 — SPIKE,
  DP.TURBINE, DP.NMET.
- Jurczyk, A., Szturc, J. & Ośródka, K. (2020). Quality-based compositing of weather radar derived
  precipitation. *Meteorol. Appl.* 27. doi:10.1002/met.1812 — step 10.
- Einfalt, T. et al. (2010). The quality index for radar precipitation data: a tower of Babel?
  *Atmos. Sci. Lett.* 11. doi:10.1002/asl.271 — QIs not harmonised.
- Huuskonen, A., Saltikoff, E. & Holleman, I. (2014). The operational weather radar network in
  Europe. *BAMS* 95, 897–907. doi:10.1175/BAMS-D-12-00216.1
- Saltikoff, E. et al. (2019). OPERA the radar project. *Atmosphere* 10, 320.
  doi:10.3390/atmos10060320 — ODYSSEY compositing weights.
- von Lerber, A. et al. (2023). OPERA5 production lines (NIMBUS). EMS Annual Meeting abstract
  EMS2023-325 (conference abstract, not peer-reviewed).
- BALTRAD bRopo documentation (FMI anomaly detection; after Peura 2002, ERAD).
  http://git.baltrad.eu/manual/bropo/index.html — **Peura 2002 incomplete.**

Cleaning of the OPERA composite, European climate datasets:
- Overeem, A. et al. (2023). EURADCLIM: the European climatological high-resolution
  gauge-adjusted radar precipitation dataset. *ESSD* 15, 1441. doi:10.5194/essd-15-1441-2023 —
  "read first"; single most relevant precedent.
- Park, S., Berenguer, M. & Sempere-Torres, D. (2019). Long-term analysis of gauge-adjusted radar
  rainfall accumulations at European scale. *J. Hydrol.* 573, 768–777. arXiv:1904.02788
- Lopez, P. (2014). Comparison of ODYSSEY precipitation composites to SYNOP rain gauges and ECMWF
  model. ECMWF Tech. Memo. 717 (grey literature; RLAN interference, S-band biases).
- Kreklow, J. et al. (2020). Radar-based precipitation climatology in Germany — developments,
  uncertainties and potentials. *Atmosphere* 11, 217. (RADKLIM) †co-authors
- Overeem, A., Holleman, I. & Buishand, A. (2009). Derivation of a 10-year radar-based climatology
  of rainfall. *J. Appl. Meteor. Climatol.* 48. †

Unsupervised / ML cleaning and analogues (search 2026-10-08; URL opened unless marked):
- Scovell, R. W. et al. (2013). Recent improvements to the quality control of radar data for the
  OPERA Data Centre (Odyssey). 36th AMS Conf. Radar Meteorology. (ams.confex URL)
- Lepetit, P. et al. (2022). Using deep learning for restoration of precipitation echoes in radar
  data. *IEEE TGRS* 60. doi:10.1109/TGRS.2021.3052582 [Crossref; Météo-France library page]
- Bölz, R. et al. (2026). Enhancing weather radar data by removing non-meteorological echoes,
  using neural networks trained on synthetic weather data. EGUsphere egusphere-2026-992.
- Bøvith, T. (2008). Detection of Weather Radar Clutter. PhD thesis, DTU, IMM-PHD-2008-201.
- Wagner, A., Seltmann, J. & Kunstmann, H. (2012). Joint statistical correction of clutters,
  spokes and beam height for a radar derived precipitation climatology in southern Germany.
  *HESS* 16, 4101-4117.
- Grazioli, J., Tuia, D. & Berne, A. (2015). Hydrometeor classification from polarimetric radar
  measurements: a clustering approach. *AMT* 8, 149-170.
- Candès, E. J., Li, X., Ma, Y. & Wright, J. (2011). Robust principal component analysis?
  *J. ACM* 58(3), 11. arXiv:0912.3599
- Zhou, X., Yang, C. & Yu, W. (2013). Moving object detection by detecting contiguous outliers in
  the low-rank representation. *IEEE TPAMI* 35, 597-610. arXiv:1109.0882
- Chang, Y. et al. (2016). Remote sensing image stripe noise removal: from image decomposition
  perspective. *IEEE TGRS* 54, 7018-7031. (Crossref only; content not read)
- Zhu, Z. & Woodcock, C. E. (2014). Automated cloud, cloud shadow, and snow detection in
  multitemporal Landsat data. *Remote Sens. Environ.* 152, 217-234. (Crossref only; method from
  snippets)
- Lehtinen, J. et al. (2018). Noise2Noise. arXiv:1803.04189. — **title abbreviated in source.**
- Krull, A., Buchholz, T.-O. & Jug, F. (2019). Noise2Void. CVPR. arXiv:1811.10980 — **title
  abbreviated.**
- Broaddus, C. et al. (2020). Removing structured noise with self-supervised blind-spot networks.
  ISBI 2020, 159-163.
- Birnie, C. et al. (2021). The potential of self-supervised networks for random noise
  suppression in seismic data. arXiv:2109.07344
- Mesarcik, M. et al. (2022). Learning to detect radio frequency interference in radio astronomy
  without seeing it. *MNRAS* 516, 5367-5378. arXiv:2207.00351
- Dunn, R. J. H. et al. (2012). HadISD. *Clim. Past* 8, 1649-1679. — **title abbreviated.**
- Lochner, M. & Bassett, B. A. (2021). Astronomaly. *Astron. Comput.* 36, 100481. arXiv:2010.11202
  — **title abbreviated.**
- Behrendt, F. et al. (2022). Unsupervised anomaly detection in 3D brain MRI using deep learning
  with impured training data. ISBI 2022. arXiv:2204.05778
- Franch et al. 2020, TAASRAD19 per-pixel outlier mask, https://pmc.ncbi.nlm.nih.gov/articles/PMC7359037/
  — RN §7.4f. Label-free static-clutter alternative. **Incomplete (no title/venue).**

National services:
- Germann, U., Galli, G., Boscacci, M. & Bolliger, M. (2006). Radar precipitation measurement in a
  mountainous region. *QJRMS* 132, 1669–1692. doi:10.1256/qj.05.190 (MeteoSwiss)
- Germann, U. et al. (2022). Weather radar in complex orography. *Remote Sens.* 14, 503.
  doi:10.3390/rs14030503 (review of Swiss chain)
- Tabary, P. (2007). The new French operational radar rainfall product. Part I: Methodology.
  *Wea. Forecasting* 22, 393–408 †; Tabary, P. et al. (2007). Part II: Validation. *Wea.
  Forecasting* 22, 409–427.
- Figueras i Ventura, J. & Tabary, P. (2012). Long-term monitoring of French polarimetric radar
  data quality... *QJRMS*. doi:10.1002/qj.1934 — **title truncated with "..." in source.**
- Harrison, D., Driscoll, S. & Kitchen, M. (2000). Improving precipitation estimates from weather
  radar using quality control and correction techniques. *Meteorol. Appl.* 7, 135–144.
  doi:10.1017/S1350482700001468 (Met Office Nimrod)

General reviews and methods:
- Villarini, G. & Krajewski, W. F. (2010). Review of the different sources of uncertainty in
  single polarization radar-based estimates of rainfall. *Surv. Geophys.* 31, 107–129.
- Saltikoff, E. et al. (2019). An overview of using weather radar for climatological studies:
  successes, challenges, and potential. *BAMS* 100. doi:10.1175/BAMS-D-18-0166.1
- Saltikoff, E. et al. (2016). The threat to weather radars by wireless technology. *BAMS* 97.
  doi:10.1175/BAMS-D-15-00048.1
- Lakshmanan, V. et al. (2007). An automated technique to quality control radar reflectivity
  data. *J. Appl. Meteor. Climatol.* 46, 288–305.
- Steiner, M. & Smith, J. A. (2002). Use of three-dimensional reflectivity structure for automated
  detection and removal of nonprecipitating echoes in radar data. *J. Atmos. Oceanic Technol.* 19. †
- Gabella, M. & Notarpietro, R. (2002). Ground clutter characterization and elimination in
  mountainous terrain. *Proc. ERAD 2002*. † (filter EURADCLIM uses via wradlib)
- Heistermann, M., Jacobi, S. & Pfaff, T. (2013). An open source library for processing weather
  radar data (wradlib). *HESS* 17. †
- Pulkkinen, S. et al. (2019). Pysteps: an open-source Python library for probabilistic
  precipitation nowcasting (v1.0). *GMD* 12. † (MCH co-authors, incl. D. Nerini)
- MLCast community (EUMETNET E-AI WG6): https://mlcast.org, https://github.com/mlcast-community
  (code read 2026-10-08).
- pysteps advection-correction example (opened 2026-10-08), citing Anagnostou, E. N. &
  Krajewski, W. F. (1999). Real-time radar rainfall estimation. Part I: Algorithm formulation.
  *J. Atmos. Oceanic Technol.* 16, 189–197. † (paper not opened; §7.4e adds
  `[CITE?: open the paper's DOI before citing it directly]`).
- Michelson, D. et al. (2020). Monitoring the impacts of weather radar data quality control for
  quantitative application at the continental scale. *Meteorol. Appl.* 27. doi:10.1002/met.1929
  (objective scoring of a QC chain).
- Marshall–Palmer Z–R (RN §7.2, no reference). **Incomplete.**
- IT-DPC-SRI paper arXiv 2602.15088 (RN §7.4f; quoted on "minimal quality filtering"). **Incomplete.**

## 2.8 Precipitation systems / radar meteorology reading list (RN §8; ★ = start here)

8.1 Radar meteorology: ★ Fabry (2015) *Radar Meteorology: Principles and Practice*, CUP; ★
Rauber & Nesbitt (2018) *Radar Meteorology: A First Course*, Wiley; Doviak & Zrnić (1993/2006)
*Doppler Radar and Weather Observations* †; Rinehart (2010) *Radar for Meteorologists* 5th ed. †;
Ryzhkov & Zrnić (2019) *Radar Polarimetry for Weather Observations* †; Bringi & Chandrasekar
(2001) *Polarimetric Doppler Weather Radar* †; Wakimoto & Srivastava eds. (2003) *Radar and
Atmospheric Science: ... David Atlas*, AMS Monogr. 30 †.

8.2 Systems: ★ Houze (2014) *Cloud Dynamics* 2nd ed. †; ★ Markowski & Richardson (2010)
*Mesoscale Meteorology in Midlatitudes* †; Trapp (2013) *Mesoscale-Convective Processes in the
Atmosphere* †; Lin (2007) *Mesoscale Dynamics* †; Doswell ed. (2001) *Severe Convective Storms*,
AMS Monogr. 28 †; Lovejoy & Schertzer (2013) (see 2.1).

8.3 Conceptual models: ★ Browning (1986) Conceptual models of precipitation systems, *Wea.
Forecasting* 1, 23–41; Hobbs (1978) *Rev. Geophys.* 16 †; Schultz & Vaughan (2011) Occluded fronts,
*BAMS* 92 †; Catto, Jakob, Berry & Nicholls (2012) *GRL* 39 †; ★ Houze (1997) Stratiform
precipitation in regions of convection, *BAMS* 78 †; Steiner, Houze & Yuter (1995) (see 2.9); ★
Houze (2004) Mesoscale convective systems, *Rev. Geophys.* 42 †; ★ Schumacher & Rasmussen (2020)
*Nat. Rev. Earth Environ.* 1, 300–314; Parker & Johnson (2000) *Mon. Wea. Rev.* 128 (TS/LS/PS
archetypes); Gallus, Snook & Johnson (2008) *Wea. Forecasting* 23, 101–113 (nine radar
morphologies); Thompson et al. (2012) Convective modes, Part I, *Wea. Forecasting* 27 †; Jergensen,
McGovern, Lagerquist & Smith (2020) Classifying convective storms using ML, *Wea. Forecasting* 35,
537–559; ★ Doswell, Brooks & Maddox (1996) Flash flood forecasting: ingredients-based, *Wea.
Forecasting* 11 †; Schumacher & Johnson (2005) extreme-rain-producing MCSs, *Mon. Wea. Rev.* 133,
961–976 (training line and back-building signatures).

8.4 Orographic: ★ Houze (2012) *Rev. Geophys.* 50, RG1001; Roe (2005) *Annu. Rev. Earth Planet.
Sci.* 33 †; Rotunno & Houze (2007) MAP lessons, *QJRMS* 133 †; Panziera & Germann (2010) *QJRMS*
136, 222–238 (MCH); Foresti et al. (2018) *QJRMS* 144, doi:10.1002/qj.3364 (MCH).

8.5 European climatologies: ★ Taszarek et al. (2019) *J. Climate* 32, 1813–1837; Wapler & James
(2015) *Atmos. Res.* 158–159, 231–244 (radar cells vs synoptic types); Fluck, Kunz, Geissbuehler &
Ritz (2021) *NHESS* 21, 683–701; Nisi, Martius, Hering, Kunz & Germann (2016) *QJRMS* 142,
1590–1604 (MCH); Nisi et al. (2018) hail streak climatology, *QJRMS*, doi:10.1002/qj.3286; Feldmann,
Germann, Gabella & Berne (2021) Alpine mesocyclones, *Weather Clim. Dynam.* 2, 1225 (MCH); Feldmann
et al. (2023) *npj Clim. Atmos. Sci.* doi:10.1038/s41612-023-00352-z; Feldmann et al. (2025) European
supercells, *Sci. Adv.* doi:10.1126/sciadv.adx0513 †first author; Ducrocq et al. (2014) HyMeX-SOP1,
*BAMS* 95, 1083–1100; Delrieu et al. (2005) Gard flash flood 8–9 Sep 2002, *J. Hydrometeor.* 6 †.

8.6 Geometry/organisation: AghaKouchak et al. (2011) (2.1); Haberlie & Ashley (2018) MCS
identification in radar mosaics Part I, *J. Appl. Meteor. Climatol.* 57, 1575–1598 (and Part II,
tracking — **incomplete**); Brune et al. (2020), Janssens et al. (2021) (2.9); Wernli (2008) SAL,
Davis et al. (2006) MODE (2.6).

## 2.9 Regimes, convection classification (RN References + §4)

- Hewson, T. D. & Pillosu, F. M. (2021). A low-cost post-processing technique improves weather
  forecasts around the world. *Communications Earth & Environment* 2, 132.
  https://www.nature.com/articles/s43247-021-00185-9 (decision tree: Supplementary Table 1) —
  ecPoint, closest analogue.
- Ferranti, L., Corti, S. & Janousek, M. (2015). Flow-dependent verification of the ECMWF ensemble
  over the Euro-Atlantic sector. *QJRMS*. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2411
- Done, J. M., Craig, G. C., Gray, S. L., Clark, P. A. & Gray, M. E. B. (2006). Mesoscale
  simulations of organized convection: importance of convective equilibrium. *QJRMS* 132. — τc.
- Flack, D. L. A. et al. (2016). Characterisation of convective regimes over the British Isles.
  *QJRMS* 142. https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.2758 — listed in References
  but not discussed in §4 text.
- Steiner, M., Houze, R. A. & Yuter, S. E. (1995). Climatological characterization of
  three-dimensional storm structure from operational radar and rain gauge data. *J. Appl.
  Meteor.* 34. — convective/stratiform separation.
- Brune, S., Buschow, S. & Friederichs, P. (2020). Observations and high-resolution simulations of
  convective precipitation organization over the tropical Atlantic. *QJRMS*.
  https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.3751 — SCAI/COP/Iorg compared on radar.
- Janssens, M. et al. (2021). Cloud patterns in the trades have four interpretable dimensions.
  *GRL*. https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020GL091001 — continuum caution.
- RN References header: "First authors, years and links were checked. Some co-author lists and
  titles are from memory".

## 2.10 Weather-type (synoptic) classification (RN §8.7)

- Huth, R. et al. (2008). Classifications of atmospheric circulation patterns: recent advances
  and applications. *Ann. N. Y. Acad. Sci.* 1146, 105–152.
- Weusthoff, T. (2011). Weather type classification at MeteoSwiss — introduction of new automatic
  classification schemes. *Arbeitsberichte der MeteoSchweiz* 235. (GWT types; ask MCH for daily
  series)
- Philipp, A. et al. (2010). Cost733cat — a database of weather and circulation type
  classifications. *Phys. Chem. Earth* 35. †
- Hewson & Pillosu (2021), Ferranti et al. (2015) cross-referenced.

## 2.11 Datasets and products (named, mostly without formal refs)

OPERA ODYSSEY / NIMBUS (EUMETNET Open Radar Data API, CC BY; RM); EURADCLIM v3.0 (KNMI);
RADKLIM RW/YW, RADOLAN (DWD); COMEPHORE (Météo-France); CombiPrecip, RZC, CPC, POH/MESHS, TRT
(MeteoSwiss); SwissMetNet and DWD 10-min gauges; ECA&D gauges; NOAA MRMS v12 (and pre-v12 via
IEM); BoM Rainfields3; ECMWF RODEO-ML OPERA; JMA Radar/Raingauge; AORC; KNMI RAC MFBS; Nimrod;
IT-DPC-SRI; RMI RADCLIM; INCA; FBK `opera-plus-it-vmi`; `italian-radar-dpc-srt1-adj`; TAASRAD19;
CLAAS-2 (CM SAF, SEVIRI) and MSG cloud type; MTG-I Lightning Imager; GPM IMERG Final; CML (NL
2011–2015; OpenRainER Emilia-Romagna 2021–22); ERA5, ERA5-Land, CERRA, E-OBS, EMO-1; ICON-CH1/2;
UKCP Local 2.2 km; CORDEX FPS Alps; ML datasets MeteoNet, RainNet/RADOLAN, SEVIR, Weather4cast. All
**incomplete as citations** (names, URLs or access notes only).

## 2.12 Positioning claims (how this work differs from prior work)

- **vs Chang & Sapsis (2026)** (DECISIONS §13):
  - Exact identity: since A(u) = |D|(1−F(u)), ∫|A_pred(u) − A_targ(u)| du = |D|·W₁ between
    marginal intensity laws; our log(1+A) with dξ = du/u is a **reweighted** W₁. Lets us cite
    their Theorem 4 (data-consistent estimators provably miss the tail — our MSE backbone at
    604/35,482 exceedances is that lower bound) and Theorem 7 (W₁ tight for tail error).
  - Ahead: they name vector observables as future work; our γ is a 3×Q vector observable with
    per-channel extreme-value interpretation; we have a demonstrated degenerate-solution failure
    mode (DECISIONS §2) and a diagnostic; a scalar max is harder to game.
  - Behind: their gradient flows through the argmax pixel, so they never hit our §5 bottleneck;
    their tail-sample selection is directly adoptable.
  - Framing: their regime is *no* extreme data; ours is extremes-present but gradient-starved.
    "The remedy transfers; the framing should not be borrowed wholesale." In a compute proposal,
    invert it: statistical regularisation gives calibrated but unidentified extremes and depends
    on a reference law whose misspecification propagates (their RMSE +20.6% under tail
    perturbation).
  - Caution: apply tilted sampling to the structural term only; keep MSE / FM velocity loss on
    uniform batches, else μ ≠ E[y|c] and the residual decomposition breaks.
- **vs the emulator paper** (MD, RM): analytic differentiable functionals replace the learned
  emulator; fixes GUDHI-vs-local-χ mismatch said to cause the paper's low topology R².
- **Regime classification** (RN §4, §8.6): no prior work found classifying regimes by Minkowski
  functionals / Euler-characteristic curves (non-exhaustive; standard in cosmology). AghaKouchak
  et al. (2011) named as nearest precedent.
- **Dataset / benchmark** (RN §7.4c, §7.4f): existing radar ML datasets are national or OPERA
  nowcasting crops; none combines Europe-wide coverage, extremes focus, event- and product-shift
  test sets and leak-free splits. No public pan-European, uncapped, cleaned hourly km-scale
  dataset exists (RODEO clips at 12.5 mm/h, EURADCLIM zeroes > 150) — labelled inference.
  Reprocessed archive would differ from EURADCLIM by keeping 15-min rates, an uncapped tail and
  both products.
- **vs EURADCLIM screen** (RN §7.4e): EURADCLIM has nothing on persistence, rays, rings, ceilings
  or radar-wide failures; ours lacks the satellite veto.
- **vs MLCast**: MLCast has no artefact screen and relies on national products.
- **vs generative downscaling practice** (RN §7.4f): most use daily targets or model output;
  hourly km-scale observational targets are rare (spateGAN-ERA5 closest).
- **Weight selection** (RN §5): novelty of the calibrated-bracket procedure not claimed until
  the systematic search in `notes/weight_selection_literature.md` is done.

## 2.13 Inconsistencies and gaps noticed

- Chang & Sapsis title: "Extreme Event Aware (η-) Learning" (D13) vs "Extreme Event Aware
  Learning" (RN, GA).
- Bengio et al. 2013 is fully cited in MD (with arXiv id) but left as `\citeq` "not opened" in RC;
  Kendall et al. has a DOI in MD but is [M] in GA.
- The only bib file (`references_reward_coupling.bib`) holds 2 entries; RC has 6 `\citeq`
  placeholders (flow matching marginal theorem, Hadwiger, Cauchy–Crofton, STE, Gneiting & Raftery
  energy score, energy distance metric).
- Other tex notes (`data_quality_assessment`, `mch_slides`, `dataset_v4_slides`,
  `isolated_slides`) contain no bibliography or formal citations; only GA has a reference list.
- No formal citation anywhere in scope for FSS, GPD/POT, Marshall–Palmer, Hadwiger, or flow
  matching.
- MD describes Q = 21 quantile-level thresholds; RC describes Q = 15 thresholds evenly spaced in
  log u — MD appears stale.
- README says chronological splits; DECISIONS §15 says the current splits are random at patch
  level.
- Overeem et al. 2023 EURADCLIM period given as 2013–2020 (RN §7.2, §7.4c) and v3.0 as 2013–2023
  (§7.4e/f) — different versions, consistent but worth labelling.
