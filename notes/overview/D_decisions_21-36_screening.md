# Digest D: DECISIONS.md §21–§36c (lines 751–2471)

Source: `/work/fquareng/ch2/minkowski-loss/DECISIONS.md`, lines 751–2471. Faithful digest; numbers,
paths and literature are as written in the file. "Pre-stated" = criteria written before the result.
Cross-references to sections outside this range (§1, §2, §10, §17, §19, §20) are kept as pointers only.

---

## §21. The target is the 1-hour accumulation, not the 15-min rate (2026-10-08)

**Decision.** Researcher, 2026-10-08, after discussing with Daniele Nerini (MCH), who had proposed a
daily target on 2026-10-07. The super-resolution target is the **1-h accumulation at 2 km**. Daily is
not the main target (can be built later from the same cleaned hours). The screen still runs on
15-min frames, because most error signatures (value ladder, temporal support, persistence) exist
only there (RESEARCH_NOTES §7.4e).

**Evidence.**
- Fine residual (x minus upsampled 25-km mean) decorrelates fast at a fixed pixel. Median correlation,
  16 days, 5,429 tile pairs at lag 1 (`scripts/data_quality/agg_compare.py`, `agg_summarize.py corr`,
  outputs `quality_v4/calib/daily_check/out/`):

  | lag | field | residual |
  |---|---|---|
  | 15 min | 0.63 | 0.29 |
  | 30 min | 0.44 | 0.09 |
  | 60 min | 0.27 | 0.035 |
  | 120 min | 0.15 | 0.018 |
  | 240 min | 0.06 | 0.012 |

  A 1-h sum averages 2–4 decorrelation times (raises SNR, Daniele's argument) while keeping cell
  geometry; a daily sum averages ~50 and turns cells into tracks, changing what H1 is about.
- Gauge agreement (log-Pearson, DE/CH, v4 filtering, 63 days): 0.70 at 15 min, 0.77 at 1 h, 0.89 at 24 h.
- Deployable input (ERA5) is hourly; no reanalysis/climate model gives 15-min snapshots.
- ~10 of 28 catalogue events (hail, supercells, derechos) are extreme sub-daily but not daily.
- Hourly is intermediate: 75% of 2-km variance unresolved by the 25-km mean (58% daily); fine-to-coarse
  peak ratio 14 (6 daily).
- Method: v4 chain (`DayCleaner`, ceilings, guard) on 8 event + 8 random ODYSSEY days; stride-128 tiles
  >= 95% valid; y = log1p(rate); residual = y minus bilinear upsampling of 10x10 block mean (~25 km);
  base frame every 2 h from 04:00; pairs counted when both tiles >= 5% wet (>= 0.1 mm/h); Pearson over
  16,384 pixels; median over tiles.
- Caveats: Eulerian correlation (10 m/s cell moves 9 km = 4.5 px in 15 min, so mostly advection;
  Lagrangian version is a to-do); 8 of 16 days carry persistent artefacts that *raise* correlation;
  16 days = indicative.

**The hour as OPERA defines it.** ACRR checked pixel by pixel on 2018-06-15 15:00 (ODYSSEY) and
2025-06-15 15:00 (NIMBUS): ACRR at H = 0.25 x (RATE at H-45 + H-30 + H-15 + H) to +-0.005 mm; no
spatial filter, no cap, no advection correction, no gap filling (one no-data frame -> no-data; 4/4
rule as in EURADCLIM). ODYSSEY writes 0 where all four frames undetect, NIMBUS keeps the undetect
code. Time labels differ by generation but the same four frames are summed. ODYSSEY ACRR QIND = mean
of four RATE QINDs to within 0.05. Consequences: hour H = frames H-45..H; ACRR is the raw-hourly
baseline of §22 and an exact regression test of the raw sum, not an independent reference; the
2026-10-07 gauge comparison was offset by 15 min and must be re-checked.

**What changes.** Store holds tile-hours (24/day not 96); LR input is the coarsened hourly field;
u = 31 mm/h, FSS levels, `extremes` subsets and §20 gate thresholds are re-derived on hourly data
(`pot_threshold.py`); loss threshold grid keeps its mm/h units (1-h accumulation in mm = mean rate in
mm/h); pre-v4 results not comparable; persistent moderate echoes now matter (error contributes
amplitude x frames / 4: a single-frame 100 mm/h spike adds 25 mm, a stationary 30 mm/h echo adds 30 mm
every hour); how the hour is formed is a modelling choice (plain sum of four snapshots beads a moving
cell; a Minkowski loss would learn it).

**Status note (2026-10-08, later).** Plain sum inflates Euler characteristic at 2 mm by +2 to +5 vs
RADKLIM (growing with motion speed) and adds 1.2–2x power at 6–16 km; advection correction removes
both. Proposed: advection-corrected target via dense motion field (pysteps-like), validated against
RADKLIM and gauges; plain sum kept as a stored variant, ACRR as its regression test.

**Would overturn.** MCH wanting daily products only, or hourly targets too noisy to train on (judged
on backbone validation tail metrics, not `val_mse`).

**Status.** In force (hourly target). Plain-vs-corrected question settled first by §25 (plain), then
revised by §35 (advection-corrected). Confidence: researcher decision; decorrelation on 16 days; ACRR
identity on one hour per generation.

---

## §22. How we judge the screen: agreement with independent data must rise (2026-10-08)

**Decision.** Researcher, 2026-10-08; metric confirmed with Daniele Nerini. Criterion is the
**increase** in agreement with an independent reference, raw -> filtered, not the absolute value
(radar vs point gauges is not expected to correlate highly at 1 h / 2 km).

**Protocol.**
- Fixed pair set: pairs wet (>= 0.1 mm) in raw radar, filtered radar *or* reference. Selecting on the
  filtered field drops repaired false echoes and inflates the gain by selection. 2026-10-07 numbers
  recomputed: 15 min +0.070 [0.041, 0.099], 1 h +0.108 [0.064, 0.145], 24 h +0.022 [0.017, 0.030]
  (95% day-block bootstrap, 300 resamples); rejection removed < 0.1% of pairs.
- **Correction (same day):** almost all that gain is one gauge, DWD 01346, on a static-clutter pixel in
  2023 (raw hourly values up to 2.2e5 mm). Without it: 15 min 0.699 -> 0.698; 1 h 0.769 -> 0.772;
  24 h 0.880 -> 0.887. The screen barely moves pooled DE/CH correlation.
- Influence: every summary reports the gain with the most influential station left out
  (`gauge_hourly.py summary`).
- Affected-pairs test (sensitive): share closer / further / median change in |log error|. First hourly
  result (60 days, hot-pixel gauges excluded): persistence changed 654 gauge-hours at 15 stations,
  98.6% closer, median log-error change -0.66, gauge dry in 82%. Spikes, rings, unsupported cells: 5–11
  gauge-hours each, 91–100% closer. Static clutter: 34 gauge-hours at 2 stations, 50/50. Rays,
  footprints, ceilings never touch a gauge.
- Rejection scored separately from repair (agreement of rejected pairs under raw data should be poor;
  share of reference-heavy hours rejected = cost).
- One rule at a time: marginal gain = full chain minus chain without that rule; negative = removes real
  rain.
- Stratify: false-alarm side (radar heavy, reference light) and miss side (reference heavy; filtered
  must not drop below raw).
- Uncertainty: day-block bootstrap.

**References by role.** DE/CH gauges (main test, in hand); RADKLIM RW and COMEPHORE (false-negative
test); MTG lightning (from July 2024, NIMBUS); IMERG (rain/no-rain over sea); EURADCLIM (another
cleaning of the same input; reported, not truth). No national radar product is independent of the
radars.

**Status.** In force as the judging principle; later operationalised as the RADKLIM removal/presence
tests (§24 addendum, §25). Confidence: criterion is a researcher decision with MCH agreement;
selection/stratification points derivable.

---

## §23. The hourly screen: rules, order and what is still provisional (2026-10-08)

**Decision.** Researcher, 2026-10-08, on Claude's proposal (review in RESEARCH_NOTES §7.4e). Hour =
frames H-45..H; pixel-hour exists only if all four frames valid; no gap filling. Processing day = the
24 hours ending d 01:00 .. d+1 00:00. Decide over the day, repair per frame, then sum
(`src/data/hourly.py`). Validation for now: DE/CH gauges only (RADKLIM, COMEPHORE, lightning, IMERG,
EURADCLIM banked).

**The chain** (each step switchable for §22 ablation):

| step | rule | change from v4 |
|---|---|---|
| frame | drizzle < 0.1 mm/h -> 0 | none |
| frame | static clutter (hot mask) | `hot_fallback: keep` — hot pixel without clean 5x5 neighbour left and coded 64, not set to 0 (breaks exact v3 reproduction, accepted) |
| frame | spikes | none |
| frame | ceiling pixels repaired (rule 2) | none; tile-frames with >= 5 still rejected at split time |
| frame | footprints of > 500 mm/h cores | none |
| frame | range rings >= 89 mm/h | none |
| frames | rays | replaces per-frame rule: thin (<= 3 px), >= 80 km line whose axis passes within 5 px + 3% of distance of a radar site, at same pixels (+-1 px) in >= 3 of the +-4 frames |
| day | local-peak persistence (new) | pixel >= 1 mm/h and >= 2x its 5x5 median in >= `p1_min` of 96 frames (`p1_ring_min` on range rings), or on a site-anchored thin line of the n_peak >= `p2_min` map; takes its 5x5 median in each frame where it is a local peak; code 128 |
| frame | cells without temporal support | none |
| split | rule 1, repeated value | not rejected when ring around repeated pixels >= 0.5x the value (`rep_context`) |
| split | rule 3, radar-disk | flag stands only if radar also flagged in a neighbouring frame, or wet fraction exceeds both neighbours' by >= 0.15 |
| split | rule 5, radar-day exclusions, guard rejections | none |
| split | tile-hour rejected | if any of its four tile-frames would be |

**Why.** Persistence is the signature that matters for sums (§21). Rules that misfired on real storms
in the gallery (ray, hot fallback, radar-disk, repeated value) now carry the separating evidence.
Repairs only lower values; nothing set to 0.

**Radar-disk evidence (20 gallery cases).** Flag run length does not separate (4 storm flags last one
frame, as do 12 of 15 Valjevo failures). Jump vs both neighbour frames does: storms within 0.13 (itcro
2022-08-18: 0.13); failures jump 0.22–0.66 except one flagged in a run. Known gap: storm flagged in two
consecutive frames keeps its flag.

**Provisional thresholds** (to be set by hourly Phase 0: 199 days, galleries, §22 ablation):
`p1_min` 24, `p1_ring_min` 12, `p2_min` 12, `ray_min_frames` 3, `min_wet_jump` 0.15, `max_context`
0.5. Evidence `p1_min` = 24 may be too high: 2021-07-14 stationary point echo next to Paris-CDG
(49.03 N 2.58 E) was a local peak in 20 frames; with a 486 mm/h frame it made the day's largest
tile-hour, 147 mm. n_peak / n_wet may separate better.

**Deliberately not done.** Advection-corrected accumulation (needs pysteps; decided before Phase 4);
gap filling; extended clutter blobs (offshore wind-farm echo at 51.65 N 2.81 E only 118 -> 108 mm/day).

**Cost.** 118–153 s per processing day per core vs ~60 s for v4 scan; ~27–36 h for archive with 6
workers.

**Status.** In force as the chain structure. Thresholds: `p1_min` kept at 24 by §30; `min_wet_jump`
revised to 0.021 by §30; burst rule added by §26. Confidence: structure follows from §21 error
arithmetic; every threshold provisional; radar-disk criterion on 20 examples.

---

## §24. No dataset decision rests on the researcher's eye (2026-10-08)

**Decision.** Researcher, 2026-10-08. What is kept is decided only by rules keyed on an error
signature (§17 safeguard 1), with thresholds fixed by a stated criterion before results are seen.
Images are diagnostics/illustrations, never decisions; acceptance is by independent data (DE/CH
gauges, §22) or a physical/statistical argument. The 2026-10-08 gallery verdicts (144 examples,
`index.csv`) and the radar-day review list (`review.csv`) decide nothing; kept as record.

**Proposals (to be fixed before Phase 3).**
- Rule 5: automatic exclusion above a pre-stated score (e.g. score >= 1: one decade above both the
  radar's own seasonal q99 and its neighbours); radar-year excluded when a stated share of its rain
  days is excluded (how Valjevo 2023 would go).
- `min_wet_jump` from the distribution of frame-to-frame wet-fraction changes in normal operation
  (e.g. q99.9), not the 20 cases.
- Persistence thresholds and other §23 provisional values set by §22 gauge tests (move radar towards
  gauge, stated cap on share moved away).
- Static clutter fails that test: 183 affected gauge-hours (3 stations), 46% closer, 48% further.
  Proposal: no unconditional repair; hot mask only lowers the persistence threshold.
- §20 gate criterion 2: replace 100 tiles by eye with DE/CH gauge false-alarm rate in the tail and the
  miss side. Criterion 3 already objective.

**Assumption.** Only independent data are DE/CH gauges; thresholds tested there assumed to transfer to
Europe. Valjevo, Madrid, Oradea lie outside DE/CH, so those rules are justified by signature (ceiling
value, range-only field). Limitation goes in the paper. Looking stays legitimate for hypothesis
generation if acceptance is tested on data the design did not use.

**Supersedes.** "Reviewed by eye" parts of §17 (safeguard 3); §19 rule 5 reviewed list; §20 criterion 2.

**Addendum (2026-10-08 ~18:50, written before any result): RADKLIM removal test**
(`scripts/data_quality/radklim_test.py`; data via `scripts/data/fetch_radklim.py`). Reference: DWD
RADKLIM RW v2017.002 (hourly, 1 km, gauge-adjusted), mapped by DWD pixel-centre coordinates; our hour H
paired with RADKLIM file labelled H-10 min (2021-07-14: correlation 0.83 vs 0.67 and 0.47 for
neighbouring hours). Same German radars, so conservative. Pixel-hours over Germany with raw (after
> 500 mm/h removal) >= 10 mm, bins 10–20 / 20–50 / 50–100 / >= 100 mm. Removed = filtered <= 0.5 x raw
or tile-hour rejected; kept = filtered >= 0.95 x raw; confirmed = RADKLIM >= 0.5 x raw.
- **Pre-stated criterion** (per rule and bin, n >= 30): artefact if removals confirmed at < half the
  kept rate; removes real rain if >= 0.8 of it; mixed in between.
- Known limit, corrected: Germany is *not* where OPERA is cleanest; 31% of tile-hours with raw max
  >= 50 mm that the screen halves lie in the RADKLIM domain (largest cluster 50–55 N, 5–10 E).

**Result (194 days, `calib/radklim_test/summary.txt`). Met where the test has power.** RADKLIM confirms
0–5% of removed pixel-hours vs 27–56% of kept, for static clutter, spikes, persistence, footprints,
rings, unsupported cells, every bin 10–100 mm with n >= 30, random and event days. No German tile-hour
rejected. The ">= 100 mm REAL RAIN" verdict is a criterion flaw (control rate 0/40, so 0 >= 0; RADKLIM
has 7 pixel-hours >= 100 mm vs 1,582 raw); recorded as undetermined, criterion not rewritten.
Tail moves towards RADKLIM:

| days | u | raw | screened | RADKLIM |
|---|---|---|---|---|
| all | 30 mm | 11,692 | 6,266 | 4,780 |
| all | 50 mm | 5,803 | 735 | 353 |
| all | 100 mm | 1,582 | 43 | 7 |
| random | 50 mm | 2,042 | 140 | 113 |
| event | 50 mm | 3,630 | 479 | 168 |
| event | 75 mm | 2,859 | 130 | 6 |

Log-correlation with RADKLIM 0.382 -> 0.399. Open: event-day screened tail above ~50 mm heavier than
RADKLIM (artefacts or peaks flattened by gauge adjustment and 1 -> 2 km averaging; RADKLIM confirms only
27% of kept 50–100 mm). Below 10 mm RADKLIM wetter (>= 5 mm: 1.94 M vs 1.21 M pixel-hours); OPERA
underestimates the body (median radar/gauge 0.46).

**Status.** In force as principle. Rule 5 numbers fixed in §28; `min_wet_jump` fixed in §30; gate
criterion 2 numbers fixed in §29. The static-clutter "no unconditional repair" proposal was not
adopted: hot repair passed the presence test in §30 and stays.

---

## §25. Criteria fixed before the results: advection correction and the Mahalanobis mask (2026-10-09)

Written before any result (researcher approved plan 2026-10-09).

### Advection-corrected hourly sums
Principle: on moving cells the plain sum is high at snapshot positions and low between, so lowering the
plain max is not softening; softening judged on stationary cells and gauges. Variants: pysteps-style
linear blend (earlier frame forward, later backward); no blending (first half from earlier frame,
second half from later); bilinear or cubic / nearest warping.

**Pre-stated acceptance (all four):** (1) beading removed: median chi difference to RADKLIM within +-1
at 1, 2, 5 mm; (2) no softening: tile max of tiles moving < 2 px per 15 min within +-5% of plain
(median ratio); (3) gauge-hours >= 20 mm (DE+CH, CH alone too): median radar/gauge ratio not lower than
plain's (day-block bootstrap); (4) mass: median tile-sum ratio within +-1%. Ties: smallest chi error. If
none passes, plain sum and beading as limitation.

**Result** (`calib/advection_variants/summary.txt`, `calib/advection_gauges/summary.txt`).
`noblend_nearest` is the only variant passing all four: chi minus RADKLIM 0 / 0 / -1 at 1/2/5 mm;
slow-tile max 0.96 of plain; mass 1.000; gauge-hours >= 20 mm -6% median ratio, interval [-10%, +0.4%]
(narrow pass). Pysteps-style linear blend fails three; cubic blend and no-blend linear fail two.
Outside the criteria: peaks of moving rain lowered. Median tile max plain / corrected / RADKLIM: slow
9.5 / 8.6 / 9.8, mid 9.2 / 6.4 / 8.3, fast 9.3 / 5.4 / 7.2 mm. Plain matches RADKLIM on slow rain,
overshoots 10–30% on moving rain; correction undershoots 20–25%. Median tile-hour where RADKLIM has
pixels >= 20 mm: every corrected variant has none (plain: half RADKLIM's area). Undetermined: motion-field
error vs RADKLIM gauge adjustment.

**Decision (researcher, 2026-10-09).** Target = plain sum (ACRR convention); `noblend_nearest` stored
alongside, choice revisited at training. Why: beading is a sampling property, not unphysical data;
under §17 least-processed is default. RADKLIM cannot arbitrate magnitudes (smoothed peaks, gauge
interpolation, our 1 -> 2 km averaging, confirms only 27% of kept 50–100 mm, 7 vs 43 pixel-hours
>= 100 mm); point gauges cannot either (2 km pixel mean < point max).
**Status: revised by §35** (target becomes advection-corrected).

### RADKLIM as a presence test (2026-10-09)
`scripts/data_quality/radklim_presence.py`, `calib/radklim_presence/summary.txt`. RADKLIM trusted only for
"substantial rain within +-2 px and +-1 h" (dry = < 5 mm). Removals are artefacts on this criterion:
85–99% dry in every bin 10 to >= 100 mm. Per rule: rejection, footprints, rings, unsupported 90–100%;
spikes 93–94%; persistence 75–89%. Remaining contamination of kept screened values (Germany, 192 days):

| screened >= u | kept | RADKLIM-dry |
|---|---|---|
| 20 mm | 29,180 | 5.7% |
| 30 mm | 6,272 | 9.5% |
| 50 mm | 736 | 31% |
| 75 mm | 157 | 68% |
| 100 mm | 43 | 86% |

Floor for ordinary 10–50 mm kept values 5–6%. 228 pixel-hours >= 50 mm on 39 of 192 days; signature:
single frame > 80% of hour in 88% (median 100%), rate median 294 mm/h (q10–q90 170–440), isolated in
92%, not persistent (median 2 local-peak frames), median 59 km from radars, on big convective days
(2014-06-08..10, 2013-07-28, 2014-07-27..29, 2023-07-29). Survive because temporal support is satisfied
by surrounding storms within 30 km and spike rule misses few-pixel clusters. Candidate burst rule ->
§26. **Status:** the presence test became the standard acceptance test (§26, §28, §30, §31).

### Per-pixel Mahalanobis outlier mask (after Franch et al. 2020, TAASRAD19)
Original design: per-pixel log-rate histogram, neighbourhood reference ~100–200 km, per era (ODYSSEY to
2024-06, NIMBUS from 2024-07), flag at mean + 3 sigma. **Pre-stated acceptance:** RADKLIM removal test
confirms changed pixel-hours at < half kept rate in every bin n >= 30; affected gauge-hours move towards
gauge in >= 80%; flag rate does not rise steeply with intensity (§17 safeguard 2). Orographic maxima
(Alps, Ticino, Black Forest) reported.

**Amendment (before any mask result; literature review RESEARCH_NOTES §7.4g).** Acceptance unchanged;
design changed because: mean + 3 sigma fixes the flagged fraction (Franch et al. flag 0.9%, Shi et al.
1.6%, ~10x Gaussian, neither validated the mask); a 526-bin histogram has exactly singular covariance;
every operational chain uses exceedance frequencies at a few thresholds (Wagner et al. 2012, Tabary et
al. 2012, DWD RADKLIM, Odyssey); a 100–200 km reference flags orographic/coastal maxima (EURADCLIM's
global reference along the Norwegian coast; RADKLIM's statistical outliers included real heavy-rain
years, Kreklow et al. 2020). Clutter 1–3 px vs orography 10–50 km -> ~10–20 km annulus separates by
scale. New design: logit exceedance frequency at 1, 31, 89, 150 mm/h per pixel-year
(`clutter_climatology.npz`), skip < 1000 valid frames; residual vs median over annulus 3–8 px (6–16 km);
robust (MCD) Mahalanobis, flag d^2 > chi^2(4) quantile 1 - alpha; alpha in {1e-3..1e-6} chosen on
even-index days (most pixel-hours >= 10 mm removed among passing alphas), reported on odd half; none
passes -> no mask; masks per year.
**Second amendment (after 2018 d^2 distribution, before acceptance results):** chi^2(4) null fails
(median 0.69 vs 3.36; q99.9 835 vs 18.5; every alpha flags ~2%). Threshold set directly on d^2, grid
{25, 100, 400, 1600} (2018: 2.8%, 1.0%, 0.28%, 0.015%). Residuals standardised by spatial spread and
sampling noise se = 1/sqrt((n + 0.5)(1 - p)); correlation from Spearman ranks (MCD singular). Diagnostic:
spokes and near-site clutter (d^2 > 400: 1.6% within 10 km vs 0.13% at 200–300 km); rate rising with
elevation (0.20% below median vs 0.73% top 1%). Mask contains 1,972–1,977 of 1,982 hot-mask pixels in
2018. **Result and status in §26** (not adopted).

---

## §26. Isolated single-frame bursts: a rule for the remaining contamination (2026-10-09)

Written before any result (researcher approved implementing 2026-10-09).

**Rule** (`src/data/hourly.py`, `burst`; per pixel-hour, after the temporal-support repair, before
summing): one frame carries > 80% of the pixel's hourly sum; that frame's value >= 100 mm/h; median of
the hour's screened sum over the 5x5 window (10 km) < 1 mm. Action: in the dominant frame, pixel takes
the 5x5 median of valid values in that frame if lower; code REPAIR_BURST (256; repair codes become
uint16). Real cores have rain around them, so fail isolation; 1 mm keeps that protection wide.

**Pre-stated acceptance** (RADKLIM presence, Germany): (1) burst removals (screened <= 0.5 x value
without rule) RADKLIM-dry >= 85% in every raw bin n >= 30 (failure -> not adopted); (2) RADKLIM-dry share
of kept screened >= 50 mm falls; (3) removals confirmed by RADKLIM reported with days, plus flag rate by
intensity. Thresholds not tuned; isolation 5 mm reported as sensitivity.

**Result** (`scripts/data_quality/burst_validate.py`, `calib/burst_validate/summary.txt`; 192 days,
2.6 M pixel-hours with max(raw, RADKLIM) >= 5 mm). **All criteria met; adopted.**

| raw bin | pixel-hours | lowered | flag rate | RADKLIM-dry |
|---|---|---|---|---|
| 10–20 mm | 210,141 | 0 | 0 | - |
| 20–50 mm | 29,959 | 308 | 1.0% | 87.7% |
| 50–100 mm | 4,245 | 141 | 3.3% | 97.2% |
| >= 100 mm | 1,588 | 30 | 1.9% | 100% |

Contamination (RADKLIM-dry share of kept >= u), without -> with: 20 mm 5.8% -> 4.4%; 30 mm 9.6% -> 4.5%
(event 9.7 -> 3.5, random 11.0 -> 6.1); 50 mm 31.5% -> 11.7% (744 -> 573; event 34.0 -> 9.5, random
27.3 -> 12.7); 75 mm 68.8% -> 34.7% (160 -> 72; event 70.8 -> 29.4); 100 mm 86% -> 60% (44 -> 15). At
>= 30 mm reaches the 5–6% floor; at >= 75 mm a third remains (25 of 72). Real cores: 13 of 479 removals
(2.7%) confirmed, over 8 days (max 4 on 2015-07-04); flag rate 1–3% per bin, not climbing with intensity.
Sensitivity, isolation 5 mm: FAILS criterion 1 (20–50 mm removals only 50% dry, 832 lowered; 155 of
1,053 confirmed). Auditable correction: first summary counted already-zero C as removed (~20,000, 61%
confirmed); removal now requires C >= 1 mm.
**Adopted:** `hourly.rules.burst: true` in `configs/quality_v4.yaml` (share > 0.8, frame rate >= 100
mm/h, isolation median < 1 mm). **Status: in force.**

### Mahalanobis mask result (from §25)
`scripts/data_quality/pixel_outliers.py`, `mask_validate.py`; `calib/pixel_outliers`,
`calib/mask_validate/summary.txt`. Masks 2012–2026, median per year 32,282 / 10,301 / 1,559 / 65 pixels at
d^2 > 25 / 100 / 400 / 1600 (1.6% / 0.5% / 0.07% / 0.003%); contain 520 of 528 hot-mask pixels at
d^2 > 25; flag rate 4.2% vs ~0.9% in highest elevation decile. Pre-stated magnitude criterion: every
threshold passes on both halves; rule picks d^2 > 25 (odd half confirmed 19 / 13 / 2% vs kept 58 / 46 /
31%). But presence test (odd half), dry share of removals: d^2 > 25: 60% / 64% / 87% (2,091 removals);
d^2 > 100: 76% / 81% / 87% (857); d^2 > 400: 91% / 93% / 100% (n 5) (169). At d^2 > 25 ~40% of 10–50 mm
removals have real rain. The two criteria disagree (magnitude pre-stated for the mask; presence-only
trust decided earlier that day but not written as the mask's criterion). Claude proposed not adopting
d^2 > 25; options: prior for persistence (preferred), d^2 > 400, or hot mask only. Auditable correction:
first run counted pixel-hours zeroed by other rules (`calib/mask_validate_run1_nopos`).

**Mask as persistence prior (researcher chose option 1; pre-stated).** On d^2 > 25 pixels, `p1_min` 24
lowered to `p1_prior_min` = 12; repair = persistence repair; hot climatology unchanged. Acceptance: prior's
removals >= 85% dry in every bin n >= 30; dry share of kept >= 20/30/50/75 mm does not rise; flag rate and
confirmed removals reported. Sensitivity `p1_prior_min` = 8.
**Result** (`scripts/data_quality/prior_validate.py`, `calib/prior_validate/summary.txt`; 192 days):
**FAILS.** Dry share 10–20 mm 84.2% (817; bar 85%), 20–50 mm 75.5% (208), 50–100 mm 83% (18), >= 100 mm
31% (13). RADKLIM confirms 540 of 3,493 removals, concentrated on 2013-05-30..06-02 (Central European
floods), 2023-07-19, 2014-07-29: long-lasting real rain on flagged (often high-elevation) pixels — the
stationary-rain risk of RESEARCH_NOTES §7.4e. Sensitivity 8: worse (79% / 70%; 1,127 of 5,497
confirmed). Little benefit: dry share >= 50 mm 11.7 -> 9.9%, >= 75 mm 34.7 -> 29.7%.
**Decision: Mahalanobis mask not adopted in either form; existing hot-pixel climatology stays.**
Inference: four exceedance features + 6–16 km reference do not separate clutter from wet orographic
pixels; future version needs e.g. Lagrangian persistence or satellite cloud-top check (EXPERIMENTS §5).

---

## §27. Radar ownership from the radars that actually contribute (2026-10-09)

**Problem.** `radar_screen.RadarGeometry` attributes pixels to the nearest "active" radar within 250 km
from the OPERA database, which lists every member radar. 22–27 Italian sites a year owned pixels (the
"itcro" flag on the 2022 Mediterranean derecho) though no Italian radar contributes; raw store has no
data over most of Italy in 2012, 2016, 2018, 2021, 2025. Duplicates, retired, non-delivering entries
(old Swedish codes, De Bilt after 2017, airport radars, most Serbian sites). Radar-frame flags, radar-day
ranking, ceiling attribution, tile owners inherit errors (2023 Valjevo 364.63 mm/h ceiling attributed to
non-delivering sites; only Fruska Gora rsfrg and Jastrebac rsjas listed for Serbia).

**Evidence.** Composites list contributors in `how/nodes`. `scripts/data/opera_nodes.py` read one composite
per month 2012-09 .. 2026-10: 167 of 170 months, 100–178 nodes/month (`validation/opera_nodes/opera_nodes.csv`;
2012-11 .. 2013-01 missing). No Italian node ever. ODIM codes differ for Spain throughout (esmad/estjv,
esbar/esgld, esmal/esahr, esmur/esftn, esval/escll, eszar/espdg, essev/esclg, esalm/esnjr, escor/esccd,
espma/esllm, eslpa/esatn, ...), Feldberg (deflg/defbg), Suergavere (eesyr/eesur), hrzag, istgb.

**Decision** (Claude, implementing researcher-approved next step). A database radar owns pixels in year Y
only if its ODIM code is listed in some month of Y, or its country prefix has listed codes matching no
database code (Spain; Germany/Estonia some years) AND the composite has valid data at its site in >= half
the year's frames (`clutter_climatology.npz`). Effect: 2013 185 -> 109; 2018 222 -> 148; 2025 246 -> 172
(22–27 Italian dropped). All Spanish radars stay; Oradea under `roora` (database also has `roopa`).

**Implementation.** `radar_screen.use_contributing(nodes_table, climatology)`, off by default (v3/v4
reproduce); on via `radar_geometry.contributing_only` in `configs/quality_v4.yaml`; read by
`scan_radars.py --quality`, `radar_screen_v4.py`, `scan_hours.py`. Phase 2 radar pass rerun; old outputs in
`quality_v4/phase2_db_geometry/` (`review.csv` unchanged).

**Check after launch.** 2012-09-10: 93 of 175 kept; all 14 Spanish kept, no Italian. Countries enter node
lists later: be (Wideumont) 2013–2026, ch 2014–2026, hu 2014–2026, lv 2015–2026, rs 2017–2024, gr
2021–2026, lt 2025–2026. Never listed: at, it, cy, tr, ba, me, mk, al, bg, md, ua. Valid data is not
evidence of contribution (valid fraction 1.0 over CH 2012 and Austria every year with no national radar);
node lists stay the criterion. Limitation: Austria (never), CH and HU (before 2014) observed only by
distant foreign radars with beam overshoot over the Alps; extremes likely underestimated; kept, not
screened.

**Limitations.** Ownership geometric (nearest contributing), not actual per-pixel source; a few duplicate
sites kept where codes unmatched; one composite per month may miss a radar down at every sample.
**Status: in force** (used by §28–§36c).

---

## §28. Rule 5 by a pre-stated score: radar-day and radar-year exclusions (2026-10-13)

**Decision** (researcher, on Claude's proposal; written before opening `quality_v4/ranking.csv.gz` of the
§27 rerun; only its log summary seen: 111 radar-days with score >= 1 under old `iso31` reference).
1. Radar-day excluded when `score >= 1` (a signal a decade above both the radar's own seasonal q99 and the
   median of its 5 nearest radars that day, §19).
2. Radar-year excluded when excluded radar-days >= 5% of its rain days and >= 3. Rain day = some frame with
   >= 20% of owned pixels >= 1 mm/h (`max_wet >= 0.2`). 3-day floor protects short years (2012: 46 days).
3. `iso31` scored against neighbours only (a radar failing on > ~1% of season-days raises its own q99 and
   hides; §19: Oradea's failure days scored < 0 on 6 of 7). Other three signals keep both references.

Why: one decade keys on failure, not rare weather (§17 safeguard 1). 5% and 3-day floor are Claude's,
with no data behind them; 20% reuses the R2 convention. Would change: real extremes a decade above all five
neighbours; check excluded radar-days on catalogued events (`configs/prominent_events.yaml`) and rejection
share by intensity, without changing thresholds.

**Implementation.** `radar_screen_v4.py rank` (reference per signal from `radar_day.reference`) and
`radar_screen_v4.py exclude` -> `day_exclusions.csv`, `year_exclusions.csv` for
`make_splits_hourly.py --day_excl`. Old behaviour without new keys. Previous ranking in
`quality_v4/rank_iso31_both/`.

**Result.** 3,029 radar-days with score >= 1 (vs 111): `iso31` 2,948, f150 65, ceilings 13, flagged frames 3;
130 of 203 radars. Rule 3 flags chronic behaviour: median excluded day 176 isolated pixel-frames >= 31 mm/h vs
neighbour median 0 and own q99 277 (old score median -0.1); flat seasons (DJF 765, MAM 703, JJA 840, SON
721); coastal/island radars (isegs 267, sekkr 201, deboo 192, mtgud 181, skjav 147, sevar 138). 51 excluded
radar-days on catalogued event days. Rule 2 denominator degenerate (Claude's specification error): 170
radar-years (57,710 radar-days) excluded; over days with data instead, 39 radar-years (13,852 days). Median
own-reference gives 2,686 radar-days (comparison only). Oradea failure days of §19 (1,286–8,636 isolated
pixels) no longer appear (largest roora day 502), probably owned by another radar under §27 geometry.

**RADKLIM test of rule 3 (pre-stated 2026-10-13; `scripts/data_quality/iso31_radklim.py`).** Population:
`iso31` pixels on German radar-days excluded with reason `iso31` (321 days with RADKLIM after fetching);
pixel owned by the radar, >= 31 mm/h in `clean_frame`, isolated by `radar_screen.isolated_mask`; mapped to
ACRR hour; d 00:00 frames skipped. Measure: RADKLIM-dry = `K_tol < 5` mm (`radklim_presence.k_tolerant`).
**Pass: pooled dry share >= 85%** (with 95% day-block CI, per radar, per radar-day); pass -> rule 3 stays;
fail -> two-reference score. Descriptive: control radar-days (score < 1); share the hourly chain leaves at
>= 5 mm. Rule 2 denominator unchanged until researcher confirms; gate reports both.
**Result (reported in §29): passes, 99.8%.**

**Status.** Rule 1 and rule 3 in force. Rule 2 (radar-year) **dropped by §29 decision (2026-10-14)**.

---

## §29. The readiness gate before the full-archive build (2026-10-13)

**Decision** (researcher asked; Claude fixed numbers before running). Pre-build gate on 199 calibration
days and 63 gauge days with current chain (`quality_v4.yaml`, burst on, §27 geometry, 2026-10-09 frame
flags); calibration days rescanned (`calib/hourly_c/`). Reported with rule 5 off and with §28 list. u
provisional: 20 mm primary, 10 and 30 reported.

**Pre-stated criteria.**
1. Known failures (kept tile-hours): Madrid 20180429 none >= 10% owned by estjv with `max_rate` >= 31 mm/h;
   Valjevo 2023 none with `max_rate` within 0.005 of 364.63; 48.62 (2019-11, tile 640/1408) none with
   `rep_value` within 0.005 of 48.62 and `rep_count` >= 5. Pass: zero each.
2. DE/CH gauges (`g_shift`, 63 gauge days + 196 calibration days with gauges, `gauge_hourly.py` with burst,
   hot-pixel gauges included): (a) false alarms (screened >= u, gauge < 1 mm): upper CI <= 5%; (b)
   screen-caused misses (gauge >= u, screen lowered to <= 0.5 x raw) <= 1%.
3. (a) rejected share of event tile-hours (`make_splits.event_rows`) < 0.5% per event with >= 50; (b) each
   rejection reason's rate at hourly max >= 20 mm <= max(3 x rate at 5–10 mm, 0.5%).
4. Deferred to build: §20 criteria 3 and 5.

**RADKLIM rule-3 result** (`calib/iso31_radklim/summary.md`): 69,726 isolated pixel-frames on 337 German
radar-days (308 days; RADKLIM 2026-07/08 unpublished): **99.8% [99.5%, 99.9%] dry; passes, rule 3 stays.**
Per radar 99.5–100% except Isen (deisn) 62.5% on 6 radar-days (341 pixel-frames). Control: 11,141 on 831
non-excluded radar-days, 98.0% [96.6%, 98.9%]. Chain leaves only 11.4% at >= 5 mm (98.4% of those dry).
Berlin (debln) 2013-07-28 (Andreas): 197 pixel-frames, median 2,938 mm/h, 100% dry; chain leaves 2.

**Gate result** (`calib/gate_prebuild/summary.md`, `gate_prebuild.py`; 199 days, 335,996 tile-hours; gauges
246 days, 8.66 M gauge-hours, 1,733 stations). **Not passed in any variant.**

| criterion | rule 5 off | §28 pre-stated | year share over days with data |
|---|---|---|---|
| rejected | 0.13% | 14.24% | 3.49% |
| 1 (Madrid/Valjevo/48.62 kept) | 5 / 1 / 1 FAIL | 0/0/0 pass | 0/0/0 pass |
| 3a worst event | 0.00% pass | 100% (Montpellier, Cannes, Misox) FAIL | 20.1% (Andreas) FAIL |
| 3b | ceiling 0.04% -> 1.77% FAIL | same | same |

Known-failure checks non-vacuous (32, 46, 79 raw tile-hours). Criterion 2: u = 10 mm raw FA 16.2%, screened
4.6% [3.5, 5.9], screen misses 0.1%, overall misses 7.6%; **u = 20 mm raw 48.8%, screened 6.4% [3.9, 9.6]
-> 2a FAIL; 2b pass (0.1%)**, overall 8.1%; u = 30 mm raw 68.4%, screened 8.5% [1.1, 19.6], 0.0%, 9.5%.

**Diagnosis.** Radar-year rule causes 2,028 of 2,169 event tile-hours rejected (chwei 2024 southern-Germany
floods, nldbl 2014 Pentecost, frnim 2014 Montpellier, depro 2017 Poland derecho); 141 via score >= 1.
Score-only: 1.08% rejected, criterion 1 pass, 3a fails only Andreas (11.2%, debln 2013-07-27/28) and Boris
(8.5%, hubud/hrdeb/sipas), both via `iso31`. Ceiling rejection rises with intensity by construction. 5% FA
bound carried from eye-review; part of remainder is support mismatch; not changed.

**Decisions (2026-10-14).**
- **Radar-year rule dropped** (researcher, Claude agreeing): caused 2,028/2,169 event losses; known
  failures caught without it; premise untested; denominator degenerate. Cost: chronic contamination never
  scoring >= 1 not caught (covered by residual-contamination estimates, methods §6.3).
  `radar_day.year_rule: false`; `day_exclusions.csv` = 3,029 radar-days; pre-stated lists kept as
  `day_exclusions_20261013_prestated.csv` / `year_exclusions_20261013_prestated.csv`.
- **`iso31` stays scored on `clean_frame` before repairs** (a repair must not hide a failing radar).
- Ceiling rejections >= 20 mm: all 335 in HU (190), HR (82), RS (63); no RADKLIM/gauge overlap. Values 48.62
  (50.0 dBZ, hunap/husze 2019), 364.63 (64.0 dBZ, HR/RS 2023), 12 at 31,575.94.

**Three follow-up tests (pre-stated 2026-10-14) and results.**
- (i) Rest of excluded radar-day's field (`rest_of_field.py`; control = same radars' non-excluded days,
  `iso31_radklim/control_days.txt`; isolated pixel-hours left out). Worse if test minus control > 2 points
  and CI above 0. **Result: WORSE** (`calib/rest_of_field/summary.md`; 348 vs 2,147 radar-days): FA C >= 5 mm
  40.2% vs 12.3% (+27.9 [+17.2, +42.5]); >= 10 mm 52.2 vs 5.5 (+46.7 [+32.1, +63.4]); >= 20 mm 66.6 vs 4.0
  (+62.6 [+39.7, +85.1]); misses 9.4 vs 7.4 (+1.9 [-0.8, +5.7]); median log-correlation 0.36 vs 0.58. So
  whole-day exclusion justified; criterion 3a no longer counts Andreas/Boris losses (Boris's radars are
  HU/HR/SI: §24 transfer assumption).
- (ii) Criterion 2a vs measured floor: pass if upper CI of (OPERA - RADKLIM) FA share <= 2 points (DWD gauges,
  score-only rule 5, rejected tile-hours dropped). **Result: FAIL.** 6.87 M gauge-hours, 243 days; u = 20 mm
  OPERA 6.3% (n = 378), RADKLIM 0.9% (n = 426), +5.4 [+2.3, +9.9]; 10 mm +1.1 [-0.1, +2.6]; 30 mm +8.0
  [+0.0, +21.6]. 24 FA gauge-hours (18 stations, 13 days, none on hot pixel): ~9 RADKLIM ~0 (untouched
  artefacts; largest Kleiner Feldberg/Taunus 2018-04-30, 6 h of 23–83 mm at one pixel, single-day stationary
  echo; Ostenfeld 2014-09-29 60 mm, Bad Berleburg 2023-07-09, Ellern 2018-04-23, Traunreut 2023-07-19); ~15
  inside convection (likely support mismatch).
- (iii) Ceiling context (`ceiling_context.py`, Z = 200 R^1.6; embedded = context >= ceiling dBZ - 10 dB;
  >= 80% embedded -> capped real cores; <= 20% -> artefacts). **Result: artefacts**
  (`calib/ceiling_context/summary.md`): 166,994 pixel-frames, 13.7% embedded, surround ~0.6 dBZ, 44–72% no
  rain around. Ceiling rule stands.

**Gate status with rule 5 = score only (adopted).** 1.08% rejected. 1 pass; 3a pass (excluding
excluded-radar-day losses); 3b rises (rej_ceiling 0.04 -> 1.77%, rej_radar_day 1.14 -> 4.80%) explained by
(iii) and (i); **2a fail**; 2b pass (0.1%). Gate turns on 2a. **Status:** superseded/closed by §32.

---

## §30. Low-hanging fruit of the screening review (2026-10-14)

**Why.** Per-rule table (`calib/rule_effect/summary.md`, 199 days): hot repair largest remover at 20–100 mm
(13–26% on random days, 21% at 50–100 mm on event days) and the one rule that failed §24; unsupported rule
lowers 6% of event-day 50–100 mm and never tested; §29 2a FAs include single-day stationary echoes missing
`p1_min` 24 by 4 frames (Kleiner Feldberg, Paris-CDG: 20 frames).

**Pre-stated tests** (`scripts/data_quality/rule_validate.py`, 196 RADKLIM days, current chain, current rule
5). Removal = kept pixel-hour C_without >= 1 mm and C_with <= 0.5 C_without; dry = K_tol < 5 mm. Pass =
removals >= 85% dry in every bin with >= 30 removals.
1. Hot repair: pass -> stays; fail -> replaced by hot-mask persistence prior (`p1_prior_min` 12), itself tested.
2. Unsupported: pass -> stays; fail -> off.
3. `p1_min` 20 and 16 vs 24: added removals pass on both halves; lowest passing; none -> 24.
4. `min_wet_jump` = q99.9 of s = min(w_t - w_{t-1}, w_t - w_{t+1}) over normal radar-frames (500 random days,
   seed 0, score < 1, no flagged frame, n_valid >= min_valid).
5. Provisional hourly u: level with same pixel exceedance frequency as 31 mm/h in screened 15-min frames.

**Results** (`calib/rule_validate/summary.md`, 193 days, 2.62 M pixel-hours):

| test | removals (10–20/20–50/50–100/>=100) | dry | verdict |
|---|---|---|---|
| static-clutter repair | 41 / 32 / 4 / 12 | 100/100/100/67% | pass: stays |
| hot as persistence prior | 15 / 18 / 3 / 0 | 100% | not needed |
| no temporal support | 216 / 51 / 4 / 6 | 97.2/100/100/100% | pass: stays |
| `p1_min` 20 added | odd 175/48, even 137/36 | 66–78% (24–28 confirmed) | fail |
| `p1_min` 16 added | odd 392/98, even 309/64 | 67–80% (54–59 confirmed) | fail |

- **`p1_min` stays 24**; stationary echoes need a different rule (fixed location + dry ring over hours).
- Hot repair passes with few cases (89 removals); §24 gauge result disagrees in tone; presence test speaks to
  tail contamination, so repair stays.
- **`min_wet_jump` = 0.021** (q99.9 over 6.86 M radar-frames; q99 0.005, q99.99 0.063; share >= 0.15 was
  2.1e-5); recomputing all 5,023 days gives the same 48 flagged radar-frames as 0.15 and same gate results.
  Adopted in `quality_v4.yaml`.
- Chain values unchanged, so 2a unchanged (fail, +5.4), not re-run. Bug in `rule_validate.py` (`p is not
  False` on NumPy bool) printed PASS for p1 tests; fixed before recording.
- Descriptive, same German pixel-hours (kept): raw OPERA 253,856 / 43,715 / 19,772 / 13,756 / 12,040 / 9,511
  at >= 10/20/30/50/75/100; screened 234,870 / 29,042 / 5,987 / 569 / 71 / 15; RADKLIM 315,170 / 28,710 /
  4,873 / 357 / 28 / 7. Screened matches RADKLIM at 20 mm, 1.2x / 1.6x / 2.5x at 30/50/75. RADKLIM
  underestimates peaks (§4 of the methods draft), so excess is an upper bound on contamination.
- **Provisional u = 20.2 mm** (`calib/dist_hist/u_provisional.txt`; 31 mm/h exceeded by 3.99e-5 of screened
  15-min pixel-frames).
- Slides: `notes/slides/2026-10-14_screening/deck.pdf`.

**Status.** In force (hot and unsupported stay; `p1_min` 24; `min_wet_jump` 0.021). u for POT later set to
50.1 mm by §32.

---

## §31. Stationary isolated echoes (2026-10-14)

**Problem.** Gate fails only on 2a (+5.4 points). Clearest cause: Kleiner Feldberg/Taunus 2018-04-30, blob
20–40 px (>= 5 mm/h) peaking 50–107 mm/h, same place ~14 h, dry ring (median 0.00–0.07 mm/h), clean days
before/after, gauge and RADKLIM dry; local peak in 20 frames; not a burst.

**Rule** (`hourly.rules.stationary`, code bit 512, off until accepted). After persistence: isolated
component in frame t = 8-connected component of pixels >= 1 mm/h, <= 150 px (600 km^2), containing a pixel
>= 10 mm/h, 3-px (6 km) ring >= 50% valid with median < 0.1 mm/h. Stationary pixel = in an isolated
component in >= `min_frames` frames of the day. Repair: component set to ring median (~0).

**Pre-stated test.** Candidates `min_frames` 12 and 8. (1) RADKLIM presence >= 85% dry in every bin with >= 30
removals, both halves; (2) on 79 event days, full grid, lowers by >= 50% at most 1% of pixel-hours the chain
leaves >= 20 mm; (3) adopt lowest passing both; (4) re-run 2a (reported, not a condition). Removing Kleiner
Feldberg is not evidence.

**Result: not adopted; both fail criterion 2.** Presence (`calib/rule_validate_stationary/summary.md`): 12
passes (89.7% / 98.5% at 10–20 / 20–50 mm odd days; 94.3% / 91.7% even); 8 fails (49–60%; 193–305 confirmed
at 10–20 mm). Events (`calib/stationary_events/summary.md`, 340,224 pixel-hours >= 20 mm): 12 lowers 3,731,
**1.097%** vs <= 1%; 8 lowers 2.05%. Bar not moved. Losses in quasi-stationary real storms: Elvira/Friederike
883, Med. derecho 562, Boris 412, Valencia DANA 378, HyMeX IOP6 357, Poland derecho 215, Emilia-Romagna (1st)
202, Aude 172 (back-building storms share the signature). Step 4 not done; gate still fails 2a. Rule stays
in code, off (`hourly.rules.stationary: false`). Untested ideas: recurrence on other days, absence of
growth/decay cycle, lightning.

**Status.** Rejected (code retained, off).

---

## §32. Gate criteria 3 and 5 on the v4 build, hourly levels (2026-10-19)

**Context.** v4 build `OPERA/patches_v4`, built 2026-10-17.

**Pre-stated definitions.** Hourly levels by matched exceedance frequency (100 random calibration days):
31 mm/h (P 3.99e-5) -> **20.2 mm**; 53 mm/h (1.31e-5) -> **31.7 mm**; 89 mm/h (3.73e-6) -> **50.1 mm**.
- Criterion 3 (`gate_tail.py`, ODYSSEY test split, unweighted): exceedance rates at 20.2 / 31.7 / 50.1 mm, GPD
  above 20.2 (location 0, as `tail_metrics.py`), xi, RL at lambda = 10/100/1000; drop 20 non-event days with
  largest share of >= 20.2 mm exceedances; pass if every statistic inside the full-sample 95% day-block
  interval (B = 200).
- Criterion 5 (`pot_threshold.py --version v4`): xi(u), sigma*(u) over 5–100 mm, train/val/test/NIMBUS,
  weighted/unweighted, cluster maxima, latitude bands, seasons, GPD truncated at 500 mm; drift "explained" by
  one named cause (truncation, mixing, sub-asymptotic threshold); Claude proposes, researcher decides.
- Evaluation u: lowest grid u above which xi and sigma* intervals (train, weighted) contain the point
  estimates at every higher u with >= 1,000 exceedances.

**Criterion 3 result (`quality_v4/gate_tail/summary.md`): FAIL as pre-stated.** 218,513 tile-hours, 486 days,
3.58e9 valid pixel-hours, 489,170 >= 20.2 mm; 20 dropped days hold 26.7%.

| statistic | full | without 20 days | 95% interval | inside |
|---|---|---|---|---|
| rate >= 20.2 | 1.366e-4 | 1.062e-4 | [1.18, 1.59]e-4 | no |
| rate >= 31.7 | 3.727e-5 | 2.893e-5 | [3.17, 4.38]e-5 | no |
| rate >= 50.1 | 8.553e-6 | 6.976e-6 | [7.01, 9.99]e-6 | no |
| xi (u 20.2) | 0.237 | 0.261 | [0.217, 0.255] | no |
| RL lambda 10 | 43.4 | 43.7 | [42.4, 44.3] | yes |
| RL lambda 100 | 83.4 | 86.5 | [79.3, 86.8] | yes |
| RL lambda 1000 | 152.3 | 164.5 | [140.5, 162.9] | no |

Dropped days: 16–30 June 2021 (12), 9–28 June 2024 (6), 15–16 August 2022 (2), plus 2013-06-07, 2023-07-18;
surrounding catalogued events. Design flaw (noted after; bar unchanged): targeted removal vs random-resampling
interval fails by construction when ~4% of days hold ~25% of tail.

**Contamination test of dropped days (pre-stated; `scripts/data_quality/dropped_days_check.py`).** 20 dropped
vs 60 random non-dropped non-event test days (seed 0); C >= 20.2 mm; RADKLIM-dry share and DE/CH gauge < 1 mm
share; contaminated if dropped - control > 2 points with CI above 0 (the §28 (i) rule).

**Decision (researcher, on Claude's proposal, before the test's result): §20 criterion 3 replaced.** New
criterion 3 (gate): the 20 most influential non-event days are not contaminated by the test above, for both
references. New reporting rule (not a gate): every tail statistic with 95% day-block interval, plus
leave-period-out (without 16–30 June 2021; without the 20 most influential days). Old result stays recorded.

**Results.** New criterion 3: **PASS** (`gate_tail/dropped_check/summary.md`): RADKLIM-dry 2.2% (n = 20,137)
vs 6.0% (n = 2,585, 30 days), -3.7 [-12.4, -1.0]; gauges 3.7% (n = 298) vs 2.1% (n = 48), +1.6 [-5.8, +6.0];
gauges record >= 20.2 mm 15.3x more often on dropped days (OPERA 17.9x): weather.
Criterion 5: **explained by sub-asymptotic threshold** (`pot_threshold/report.md`, `fig/`). xi train weighted
0.41 at 5 mm, 0.31 at 10, 0.28 at 20.2, 0.21 at 31.7, 0.10 at 50.1, 0.07–0.12 to 100 mm. Not truncation
(truncated fit same); not mixing (JJA 0.27 -> 0.08, MAM 0.31 -> 0.11, south of 46N 0.23 -> 0.10); drift stops
~50 mm; cluster maxima xi 0.02–0.08 above 50 mm vs pixels 0.07–0.12.
**Evaluation u = 50.1 mm** by the pre-stated rule (coincides with hourly 89 mm/h; train 175,937 exceedances,
2,414 days; test unweighted 30,589, 404 days). 20.2 mm stays the level for split strata and exceedance-rate
columns.

**Gate status on v4:** 1 pass; 2a fail (+5.4); 2b pass; 3 (new) pass; 4 pass; 5 explained.
**Decision (researcher, 2026-10-19):** criterion 5 accepted (methods draft §5.6, with synthetic check);
**train on v4**; 2a reported as minor limitation (methods draft §6.3: ~9 residual artefact gauge-hours,
single-day stationary echoes, +5.4 points FA at 20 mm over RADKLIM). No v5. Gate closed.
**Status.** In force; u = 50.1 and levels derived on plain sums, to be re-derived on the advection store (§35).

---

## §33. Training on v4: hourly thresholds, config and data preparation (2026-10-19)

**Decision** (researcher said go ahead; Claude chose the rule). Each `PHYSICAL_THRESHOLDS` level mapped to the
hourly level with the same exceedance frequency (rule of §30/§32):

| 15-min mm/h | 0.1 | 0.17 | 0.28 | 0.48 | 0.81 | 1.4 | 2.3 | 3.9 | 6.5 | 11 | 19 | 31 | 53 | 89 | 150 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mm in 1 h | 0.1 | 0.192 | 0.314 | 0.509 | 0.828 | 1.35 | 2.11 | 3.43 | 5.36 | 8.61 | 13.5 | 20.2 | 31.7 | 50.1 | 79.0 |

Ratio ~1 in light rain to 0.53 at the top. Evaluation defaults would give 31.7 / {31.7, 50.1, 79.0}; POT uses
50.1, so `eval_extremes.py --pot_threshold 50.1`.
- `configs/config_v4.yaml` = `config_v3.yaml` + v4 store/metadata + this grid (all values in mm per hour of
  accumulation).
- Persistence thresholds (B0/B1 noise floor) recomputed on hourly store (diagnostics only in `TOPOLOGY_MODE:
  euler`).
- Gamma targets after store build (`scripts/hpc/prep_train_v4.sh`, 8 workers).
- Smoke run: UNet backbone, 1 epoch, 0.5% of train, GPU 1 (loading, finite losses, startup lines only).
- sigma_r (`compute_residual_stats.py`) recomputed per new backbone before any FM run.
**Status.** In force; grid to be re-derived on the advection store (§35).

---

## §34. Ownerless tile-hours excluded; pre-December-2015 speckle floor flagged and measured (2026-10-19)

**Findings** (`notes/data_analysis_v4/stats.md`). Step change 2015-12-12: daily share of fully dry tile-hours
(max < 0.1 mm) ~0.02–0.07 before, 0.20–0.29 after; before, dry-looking tiles carry scattered values (median
max 0.2 mm where wet fraction < 0.2%) — a speckle floor in earlier ODYSSEY processing (inference; change not
identified). 58,124 tile-hours have no owning radar (6 tile positions, 2012–2015: rows 512–768 x cols
1536–1664, and 1792 / 128); rules 5 and 3 cannot act on them. In store: train 13,441, val 842, test 3,033,
NIMBUS 0.

**Decision (researcher).** (1) Ownerless tile-hours excluded at metadata level: `patches_v4/meta/*_owned.txt`
(5 columns, 5th = store row), pointed to by `configs/config_v4.yaml`; originals kept. (2) Pre-2015-12-12
tile-hours flagged, not removed (`patches_v4/meta/full_<split>_flags.csv.gz`: store_row, ownerless,
pre_break); effect measured at data level (`scripts/analysis/speckle_effect.py`) and later at training level
(post-break-only vs all-data backbone, both evaluated on post-break test).

**Results.** Exclusion applied (`scripts/dataset_v2/filter_meta_v4.py`, `patches_v4/meta/filter_report.txt`):
kept train 1,308,448, val 172,618, test 215,480, NIMBUS 381,736; validator passes
(`validation/store/20261019_1621_9b51580.md`). Pre-break among kept: train 313,644 (24%), val 24,005, test
68,261. Speckle effect (`notes/data_analysis_v4/speckle_effect.md`, 300 rows per era and bin, medians):

| tile mean (mm) | chi@0.1 post / pre / pre zeroed <0.2 | single-pixel comps post / pre | A@0.1 (km²) post / pre zeroed 0.2 |
|---|---|---|---|
| < 0.005 | 2 / 13 / 4 | 1 / 7 | 20 / 36 |
| 0.005–0.02 | 31.5 / 58 / 29 | 16 / 29 | 1,610 / 490 |
| 0.02–0.1 | 41 / 81.5 / 56 | 23 / 50 | 6,290 / 3,070 |
| 0.1–0.5 | 39 / 59 / 60 | 26 / 46 | 19,700 / 14,400 |
| >= 0.5 | 22.5 / 28.5 / 35.5 | 21 / 34 | 39,600 / 34,000 |

Pre-break 1.3–6.5x more fragmented at 0.1 mm, ~2x single-pixel components (the threshold range carrying 83% of
gradient). Zeroing below 0.2 mm cuts area at 0.1 mm to a third of post (real light rain at 0.1–0.2 mm). At
0.31 mm pre-break chi still ~1.5x. Season/network not controlled (inference). **No floor applied**; flag stays;
training-level test decides.

**Later decision (researcher, 2026-10-20; recorded at end of §36c).** Pre-break ablation runs right after the
smoke run. `scripts/dataset_v2/postbreak_lists.py` writes `<list>_postbreak.txt` in the adv store meta (train
76.0%, val 86.1%, test 68.3% post-break). Amendment (Claude, 2026-10-21, before either arm runs): both arms
validate on the **full** validation list, evaluated on post-break test; the vanilla v4_adv baseline
(`run_vanilla_v4adv.sh`, same seed and budget) is the all-data arm, so the ablation costs one extra run.
**Status.** Ownerless exclusion in force; pre-break question open (ablation pending).

---

## §35. The target is the advection-corrected hourly field (2026-10-19; revises §25)

**Decision** (researcher, after the v4 gallery `notes/data_analysis_v4/gallery_*.pdf`). Plain sum unusable: four
15-min positions of moving cells visible. Target = advection-corrected field (`noblend_nearest`). Decisive
for a structural loss: beading adds +2 to +5 to chi at 2 mm vs RADKLIM and 1.2–2x power at 6–16 km, which the
loss rewards.

**How (Claude's design).** Separate store `OPERA/patches_v4_adv` from `patches_v4`
(`scripts/dataset_v2/build_adv_store.py`): `original_precip` = `adv_precip`; coarse/interpolated recomputed
(`coarsen_and_interpolate`); `max_rate`, `n_wet` copied (describe 15-min frames); rows with any NaN dropped (no
complete H-60..H+15 window), after §34 exclusion, dropped share reported per split; metadata rewritten with
corrected tile max and new rows; scaler = log1p of corrected train max; DEM stats copied; pre-break flags
carried. `configs/config_v4_adv.yaml` points to it; nothing downstream changes.

**Carries over / re-derived.** Screening, gate (§29, §32), pre-break flags carry over. §33 thresholds and u = 50.1
(derived on plain sums) kept for now; `pot_threshold.py` and matched-frequency grid re-run on the new store
before evaluation. §25 gauge check passed narrowly (-6%, [-10%, +0.4%]): limitation.

**Measured on built store (2026-10-21).** Corrected/plain tile max, train (test alike): median 0.66–0.78 in every
bin from 1 to > 50 mm, q10 0.29–0.41, q90 0.96–1.00. Tile-hours >= 50.1 mm: 26,072 -> 14,196 (train), 5,293 ->
2,398 (test); >= 20.2 mm: 152,947 -> 84,380. The ~4% of §25 was for slow tiles; plain sum credits a cell's full
15-min rate to each of four positions. Gauges small effect (-6%). u = 50.1 and thresholds must be re-derived
(planned); extremes subsets shrink. Gallery `gallery_plain_vs_adv.pdf`: chi lower at all levels, maxima 13–25%
lower on convective pages shown.
**Status.** In force (revises §25). Threshold re-derivation pending.

---

## §36. Radars biased against their neighbours: the boundary-jump test (2026-10-19)

**Question.** From 2015-12-12 (`notes/data_analysis_v4/radar_tail.md`): Bobohalma (robob) 2nd-highest pixel
frequency >= 20 mm of 116 radars (2.6e-4), pixel >= 20 mm on 46% of radar-days; Napkor (hunap), Szentes (husze),
Kojsovska hola (skkoj) dry less often (0.17–0.19 vs median 0.25), tile means up to 4x median; top-1%-days share
0.20–0.36 vs median 0.47 (chronic). Climate or bias?

### First test: mean boundary jump (pre-stated; researcher expected flagged radars to be biased)
`bjump` = mean log1p(rate) in 6 km band inside owned area minus 6 km band outside
(`radar_screen.radar_frame_features`, clean_frame, contributing geometry); `scripts/analysis/radar_boundary.py`.
Frames with wet fraction >= 0.05 and >= 50 valid px per band; radar-day = median over >= 8 frames; radar-year =
median over >= 20 days, CI B = 500. Flag: top 5% AND CI above 0. Consequence: tile-hours with owner share >= 10%
excluded at metadata level before the advection store. Limitation stated: boundaries on climate gradients.
**Result (`notes/data_analysis_v4/radar_boundary.md`): exclusion NOT applied — test does not measure what is
needed.** 1,987 radar-years; cut 0.018; 100 radar-years from 49 radars flagged (jumps 0.018–0.039): many Swedish
2024–2026, Albis (chalb), La Dole (chdol), French/German radars, skkub, sklaz, skjav 2019, rotim, robar. Suspects
not flagged (0.012–0.016). Median over wet frames dominated by light rain. Options: (a) apply anyway; (b)
tail-focused test; (c) no exclusion.

### Second test ("§36b"): tail-focused boundary test (researcher chose (b), 2026-10-20; pre-stated)
`scripts/analysis/radar_band_tail.py`. Every second day (even index from 2012-09-04), every frame, `clean_frame`,
§27 geometry; inner/outer 6 km bands: valid, >= 10, >= 31 mm/h. R10 = (inside >= 10 / inside valid) / (outside
>= 10 / outside valid) summed over year; CI of log R10 (B = 500); R31 descriptive. **Flag: R10 >= 2 AND lower
bound >= 1.25, with >= 20 days with a pixel >= 10 mm/h in either band** (absolute bar). Orographic boundaries
checked by name. Consequence: owner share >= 10% excluded (`filter_meta_v4.py --biased <csv>` ->
`build_adv_store.py --meta_only`; side effect: scaler re-derived). Run: `scripts/hpc/run_band_tail.sh`,
`logs/band_tail.log` -> `BAND_TAIL_DONE`, `quality_v4/band_tail/days/`, `notes/data_analysis_v4/radar_band_tail.md`,
`quality_v4/biased_radar_years_tail.csv`. Smoke 2021-07-14: 156 radars, inside/outside within ~10%.
**Result (2026-10-21).** 2,024 radar-years; median R10 0.98. **3 flagged:** ukcyg 2023 (2.51, 2.00–2.91), isbjo
2024 (2.39, 1.79–3.59), plpas 2012 (2.17, 1.48–3.57; 23 days). Suspects robob, hunap, husze, skkoj R10 0.78–1.12;
highest suspect sklaz 2017–18 (1.56–1.63). Exclusion size: train 4,108 (0.31%; 88 >= 20.2 mm), val 143, test
1,262 (14 >= 20.2 mm), nimbus/events 0. Orographic check: all three hill/fjord sites (Crug-y-Gorllwyn, Bjolfur,
Pastewnik), not ruled out.
**Decision (researcher, 2026-10-21): exclusion applied as pre-stated.** `scripts/hpc/apply_radar_exclusion.sh` =
`filter_meta_v4.py --biased quality_v4/biased_radar_years_tail.csv` -> `build_adv_store.py --meta_only` ->
post-break lists -> persistence thresholds -> store validation (`logs/apply_radar_exclusion.log`). Fields and
gamma targets unchanged.

**Observation (2026-10-21): Jastrebac (rsjas) half-disk of noise in 2023** (`suspect_radars.pdf` p. 1;
`scripts/analysis/locate_noise_radar.py` -> `notes/data_analysis_v4/noise_radar_2023.{md,png}`). 97,400 km²
anomaly (frequency > 1e-3 and > 5x 2021–25 median), circular western arc centred 4 km from rsjas (next site 31
km), radius 248 km, straight eastern edge = diameter; absent other years. rsjas in node lists 2017–2023 only. Rule
5 excluded 12 of its 2023 radar-days. Whether the chain removes it: `scripts/data_quality/rate_climatology.py`
(running); exclusion is the researcher's decision. Researcher also named hubud and hrdeb as worst on that page.
(Note: CLAUDE.md later lists rsjas 2023 and hrulj 2023 in the exclusion CSV; that decision lies beyond this scope.)

### §36c. National-border extension (pre-registered 2026-10-21)
**Why.** Network-wide bias cancels at same-network boundaries. **Pre-stated criteria**
(`scripts/analysis/radar_band_tail.py run_national / analyze_national`): data as §36b; pixel country = owning
radar's ODIM prefix; A-side band = A pixels within 6 km (3 px, `BAND_PX`) of a B pixel; R10(A|B) summed over year,
CI B = 500, R31 descriptive; border side qualifies if both bands >= 150 px and >= 20 days with pixel >= 10 mm/h;
border flagged R10 >= 2 AND lower bound >= 1.25; **country-year flagged if >= 2 qualifying border sides flagged**;
single flagged borders reported only, with mean DEM height. Consequence: all that country's radars that year
excluded as §36b. Limitations: networks with same bias cancel; one-border countries cannot be flagged; seams are
nearest-radar network seams, not political borders.
**Result (`notes/data_analysis_v4/radar_band_national.md`).** 1,034 qualifying border sides x years; median R10
1.00; 13 border sides flagged; **one country-year: Lithuania 2026** (2.34 vs Poland, 2.10–2.61; 2.22 vs Sweden,
1.65–3.18; not vs Latvia) -> ltlau, ltvil 2026. **HU, SK, RO: no network bias** (median R10 0.99 hu, 63
border-years; 1.02 sk, 44; 1.11 ro, 6; max 1.69 ro vs hr 2025, 1.67 sk vs hu 2018); high tail attributed to climate
or intermittent artefacts, not consistent excess. Single flags not acted on: BE vs UK 2018–20, NL/UK 2016,
2024–25 (sea seams), UK vs IE 2023, LV vs PL 2016 (20 days), FR vs HR 2023–25 (geometry artefact: Aleria–Goli
seam over Tuscany with no Italian radar). **Exclusion applied** (`apply_radar_exclusion.sh
quality_v4/biased_radar_years_all.csv` = §36b's three + ltlau/ltvil 2026): only nimbus has 2026 rows; 1,888
tile-hours removed (1.4% of its 2026 rows; 186 >= 20.2 mm). Train, val, test, scaler unchanged.
**Status.** In force.

---

## Final screening pipeline at the end of scope (§36c, 2026-10-21)

Config `configs/quality_v4.yaml`; implementation `src/data/hourly.py`; radar geometry = contributing radars only
(§27, `radar_geometry.contributing_only`). Decide over the processing day (d 01:00 .. d+1 00:00), repair per
15-min frame, then sum frames H-45..H (pixel-hour only if all four valid; no gap filling). Repairs only lower values.

Frame / day level (in order, §23 + §26):
1. Drizzle < 0.1 mm/h -> 0.
2. Static clutter (hot-pixel climatology), `hot_fallback: keep` (code 64). Kept after §30 presence test.
3. Spikes.
4. Ceiling pixels repaired (rule 2).
5. Footprints of > 500 mm/h cores.
6. Range rings >= 89 mm/h.
7. Rays (multi-frame: thin <= 3 px, >= 80 km, site-anchored, same pixels in >= 3 of +-4 frames).
8. Local-peak persistence over the day (`p1_min` 24, `p1_ring_min` 12, `p2_min` 12; code 128).
9. Cells without temporal support (kept after §30).
10. Isolated single-frame burst (share > 0.8, frame >= 100 mm/h, 5x5 hourly median < 1 mm; code 256) (§26).
11. Sum to the hour.

Split / tile-hour level:
12. Rule 1, repeated value, with `rep_context` (ring >= 0.5x value spares it).
13. Ceiling tile-frames with >= 5 ceiling pixels rejected.
14. Rule 3, radar-disk flags, standing only if flagged in a neighbouring frame or wet-fraction jump >= `min_wet_jump`
    = 0.021 over both neighbours (§30).
15. Rule 5, radar-day exclusion, score only: `score >= 1`, `iso31` scored vs neighbours only on `clean_frame` before
    repairs; radar-year rule off (`radar_day.year_rule: false`); `day_exclusions.csv` = 3,029 radar-days (§28–§29).
16. Guard rejections.
17. Tile-hour rejected if any of its four tile-frames would be.

Metadata / store level:
18. Ownerless tile-hours excluded (`*_owned.txt`, §34).
19. Pre-2015-12-12 tile-hours flagged, not removed (ablation pending, §34).
20. Biased radar-years excluded where owner share >= 10%: ukcyg 2023, isbjo 2024, plpas 2012, ltlau 2026, ltvil 2026
    (`quality_v4/biased_radar_years_all.csv`, `scripts/hpc/apply_radar_exclusion.sh`) (§36, §36c).
21. Target = advection-corrected hourly field (`noblend_nearest`), store `OPERA/patches_v4_adv`,
    `configs/config_v4_adv.yaml`; rows with NaN in the corrected field dropped (§35).

Not adopted / off: per-pixel Mahalanobis mask (unconditional and as prior, §26); stationary isolated echo rule
(`hourly.rules.stationary: false`, §31); radar-year rule (§29); lowered `p1_min` 20/16 (§30); first mean-bjump
radar exclusion (§36); speckle floor (§34).

Known residual limitation: gate criterion 2a fails, +5.4 points false alarms at 20 mm over RADKLIM, from ~9
single-day stationary echoes (methods draft §6.3). Evaluation: u = 50.1 mm for POT, 20.2 mm for strata and rate
columns, hourly threshold grid of §33 — all derived on plain sums and due for re-derivation on `patches_v4_adv`.

## Literature cited in scope (exactly as written)
- Franch et al. 2020, TAASRAD19 (per-pixel Mahalanobis outlier mask; flags 0.9%).
- Shi et al. (flag 1.6%; no year given in the file).
- Wagner et al. 2012; Tabary et al. 2012; DWD RADKLIM; Odyssey (operational chains using exceedance frequencies).
- Kreklow et al. 2020 (RADKLIM's statistical outliers included real heavy-rain years).
- EURADCLIM (4/4 rule; global reference flagged Norwegian coast).
- Internal: RESEARCH_NOTES §7.4e, §7.4g; EXPERIMENTS §5; methods draft §4, §5.6, §6.3.
