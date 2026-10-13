# Cleaning the OPERA composite for hourly km-scale extremes: methods and results

Living draft for the paper's data section and supplement. Started 2026-10-09; every number
cites the script and output that produced it. The decision trail, with dates, is in
DECISIONS §17-§26; the operational log is in EXPERIMENTS §5. Numbers here are copied from
those outputs, so update both together.

## 1. Target and motivation

- **Target:** the 1-h accumulation at 2 km, in the convention of OPERA's own ACRR. Hour H is
  0.25 x (RATE frames H-45, H-30, H-15, H); a pixel-hour exists only if all four frames are
  valid (DECISIONS §21).
- **ACRR is the plain sum.** Checked pixel by pixel on one hour per product generation
  (2018-06-15 ODYSSEY, 2025-06-15 NIMBUS): equal to the RATE sum within +-0.005 mm, with no
  filter, no cap and no gap filling.
- **Why hourly, not 15-min or daily:**
  - the fine residual (the part below the 25 km mean) decorrelates within 30 min. Median
    Eulerian correlation between frames t and t + lag: 0.29 at 15 min, 0.09 at 30 min,
    0.035 at 60 min (`scripts/data_quality/agg_compare.py`, 16 days);
  - agreement with gauges rises with aggregation;
  - ERA5 inputs are hourly;
  - sub-daily extremes survive.
- **The screen runs on the 15-min frames.** Value-ladder signatures (ceilings, repeated
  values) and temporal signatures exist only there. It decides over the whole day and
  repairs per frame, before the hour is summed (`src/data/hourly.py`).

## 2. Screening policy

- **Reject clear errors only, keep imperfections** (DECISIONS §17).
- **No dataset decision rests on a person's eye** (DECISIONS §24). Images are diagnostics.
  Every rule is accepted or rejected by independent data against criteria written down
  before the results (DECISIONS §25, §26).
- **References and what they are trusted for:**
  - DE/CH gauges: a point measurement;
  - DWD RADKLIM RW: hourly, 1 km, gauge-adjusted, own artefact correction, same German
    radars;
  - neither is trusted for peak magnitudes (§4).

## 3. The rule set (hourly chain, `src/data/hourly.py`, `src/data/cleaning.py`, `src/data/radar_screen.py`)

| step | rule | signature it keys on |
|---|---|---|
| frame | drizzle < 0.1 mm/h -> 0 | noise floor |
| frame | static clutter (climatological hot mask; repair only if lower, never set to 0) | per-pixel exceedance frequency |
| frame | spikes | a pixel >= 10 mm/h whose neighbours are all < 10% of it |
| frame | ceiling pixels (radar-year value table) | a value piled up above its +-4 dB neighbours |
| frame | footprints of > 500 mm/h cores | physical impossibility |
| frame | range rings >= 89 mm/h | climatological ring geometry |
| frames | RLAN rays | thin (<= 3 px) lines through a radar site, fixed over +-4 frames |
| day | local-peak persistence | a pixel that is a local peak in many frames of the day |
| frame | cells without temporal support | no echo >= 1 mm/h within 30 km at t +- 15 min |
| hour | **isolated single-frame bursts** (new, §26) | one frame carries > 80% of the pixel-hour at >= 100 mm/h, with no rain around it that hour |
| split | tile-hour rejection | >= 5 ceiling pixels, repeated value outside rain, refused large repairs, radar-disk failure (with temporal check) |

## 4. How the screen is validated

**Presence, not magnitude.** RADKLIM's peaks cannot judge OPERA's:
- RADKLIM underestimates high intensities (reported in the literature);
- its gauge adjustment is interpolated between gauges;
- our comparison averages it from 1 to 2 km, and the hour windows must be aligned
  (H-70..H-10 for the RADKLIM file labelled H-10);
- RADKLIM confirms only 27% of kept 50-100 mm OPERA values and has 7 pixel-hours >= 100 mm
  where screened OPERA has 43;
- a point gauge exceeds a 2 km pixel mean inside a convective core by construction, so radar
  vs gauge in the tail is ill-posed.

The question asked of RADKLIM is therefore only "is there substantial rain here": dry means
< 5 mm within +-2 px and +-1 h. Ordinary kept 10-50 mm values are "dry" 5-6% of the time
(timing and position mismatch). That is the disagreement floor.

**Gauges.** The gain in log-correlation on a fixed pair set (pairs selected on the filtered
field inflate the gain) is reported with the most influential station left out: one DWD gauge
on a clutter pixel once produced +0.10 of a +0.108 gain. The informative statistic is the
per-rule affected-pair test: where a rule changed the radar value, did it move towards the
gauge? (DECISIONS §22.)

## 5. Results

### 5.1 The screen removes artefacts (presence test; `calib/radklim_presence/summary.txt`)

Over Germany (192 days), 85-99% of the pixel-hours the screen removes are RADKLIM-dry, in every
raw bin from 10 to >= 100 mm. Per rule:
- tile-hour rejection, footprints, rings, unsupported cells: 90-100%;
- spikes: 93-94%;
- persistence: 75-89% (it lowers peaks inside real rain rather than removing rain).

Gauges agree. The persistence rule changed 654 gauge-hours at 15 stations, and 98.6% moved
towards the gauge; spikes, rings and unsupported cells: 91-100% (`calib/gauge_hourly`).

### 5.2 How much the screen changes the extremes (`calib/extremes_effect/report.txt`)

On 100 random days, taking as baseline the raw field with only the > 500 mm/h pixel-hours
removed:

| pixel-hours >= | baseline | screened | kept |
|---|---|---|---|
| 20 mm | 155,433 | 91,535 | 59% |
| 50 mm | 36,145 | 8,008 | 22% |
| 100 mm | 6,262 | 580 | 9% |

- Wet pixel-hour quantiles: q99.99 63 -> 34 mm. Tile-hour maxima >= 50 mm: 3,982 -> 1,470.
  GPD shape xi above 20 mm: +0.33 -> +0.23.
- In Germany, the screened tail approaches RADKLIM's (random days, >= 50 mm: raw 2,042,
  screened 140, RADKLIM 113).
- 31% of the removed tile-hour maxima >= 50 mm lie inside the RADKLIM domain.

### 5.3 The remaining contamination, and the burst rule (`calib/radklim_presence`, `calib/burst_validate`)

After the screen, the RADKLIM-dry share of kept screened values over Germany is 5.7% at
>= 20 mm (the floor), 31% at >= 50, 68% at >= 75 and 86% at >= 100 mm. It is small in
absolute terms (228 pixel-hours >= 50 mm on 39 of 192 days) but it sets the tail.

Signature of those pixel-hours:
- in 88%, one 15-min frame carries > 80% of the hour, at a median of 294 mm/h;
- 92% are isolated;
- they are not persistent and not near radars;
- they sit on big convective days (2014-06-08..10, 2013-07-28, 2014-07-27..29,
  2023-07-29).

They survive because the 30 km temporal-support box is satisfied by surrounding storms, and
the spike rule misses clusters of a few pixels.

The burst rule (§26: one frame > 80% of the pixel-hour, >= 100 mm/h, hour's 5x5 median < 1 mm)
passes its pre-stated test:
- *its removals are RADKLIM-dry:* 88% (20-50 mm), 97% (50-100), 100% (>= 100);
- *its flag rate is 1-3% per bin;*
- *13 of its 479 removals are confirmed by RADKLIM,* over 8 days;
- *the contamination falls:* 9.6 -> 4.5% at >= 30 mm (the floor), 31.5 -> 11.7% at >= 50,
  69 -> 35% at >= 75, 86 -> 60% at >= 100 mm (15 values left);
- *loosening the isolation to 5 mm fails* (50% dry at 20-50 mm), so the isolation condition
  is what protects real rain.

### 5.4 The hourly sum: plain vs advection-corrected (DECISIONS §21, §25)

- **Beading.** The plain sum of four snapshots beads moving rain. Against RADKLIM (built from
  5-min data, time-aligned), OPERA's excess Euler characteristic at 2 mm grows with motion
  speed (+2 to +5), and OPERA has 1.2-2x RADKLIM's power at 6-16 km. Advection correction
  removes both. The fine band (4-16 km) persists along the motion (Lagrangian correlation
  0.58-0.74), so it is structure, not radar noise. The two products share most of their
  variance down to ~12 km wavelength (`calib/signal_noise`, `calib/advection_variants`).
- **The pre-stated test of the corrected variants.** Only no-blend nearest-neighbour warping
  passes; the pysteps-style linear blend softens stationary peaks by 16-22%.
- **The cost on moving rain.** Even that variant lowers moving-rain tile maxima 20-25% below
  RADKLIM, while the plain sum overshoots RADKLIM by 10-30% there.
- **Decision.** Beading is a sampling property, not unphysical data, and RADKLIM cannot
  arbitrate magnitudes. The target is the plain sum; the corrected field is stored alongside.

### 5.5 Per-pixel outlier mask (Mahalanobis; DECISIONS §25 and amendments)

Design adapted from Franch et al. (2020) with fixes from the literature:
- logit exceedance frequencies at 1 / 31 / 89 / 150 mm/h per year;
- a local 6-16 km annulus reference;
- residuals standardised by spatial spread and sampling noise;
- a rank-correlation Mahalanobis distance, with the threshold on d^2.

It flags radial spokes and near-radar clutter and contains the existing hot-pixel mask (520 of
528 pixels at d^2 > 25). Its flag rate rises in the highest elevation decile.

Acceptance (`calib/mask_validate`):
- *By the pre-stated magnitude criterion,* every threshold passes, and the selection rule
  picks d^2 > 25.
- *By the presence test,* d^2 > 25 removals at 10-50 mm are only 60-64% RADKLIM-dry (rules:
  85-99%; floor 3-6%), so about 40% hit real rain; d^2 > 400 removals are 91-100% dry but
  few (169).
- *An unconditional per-pixel mask shaves real rain* wherever a flagged pixel sits inside a
  storm.
- *Used as a prior for the persistence rule instead* (`calib/prior_validate`), it also fails its
  pre-stated test: 84% / 76% RADKLIM-dry at 10-20 / 20-50 mm. Its confirmed removals
  concentrate on long real rain (the May-June 2013 Central European floods), where flagged
  pixels, more frequent at high elevation, are persistent local peaks.
- *Its benefit at the top is small* (dry share >= 50 mm 11.7 -> 9.9%).
- *Not adopted in either form;* the existing hot-pixel climatology stays (DECISIONS §25).
- *Lesson for the paper:* a per-pixel climatological outlier test does not, by itself,
  separate clutter from climatologically wet orographic pixels. The burst rule, keyed on
  temporal structure rather than climatology, is what reduced the tail contamination.

## 6. Limitations (for the methods section and the appendix)

Updated 2026-10-13. Each item gives the cause, the direction of the error it introduces, its
size where we measured it, and what the method does about it. Items marked *[verify]* rest on
a fact not yet checked against a source.

### 6.1 How far the validation reaches

- **Independent data exist only over Germany and Switzerland.** RADKLIM covers Germany and
  the gauges cover DE/CH, about a third of the screen's tail removals. Elsewhere a rule is
  justified by its signature (a value piled up at a ceiling, a field that depends on range
  only), not by a test. *Assumption:* thresholds tested over DE/CH transfer to the rest of
  Europe. The failures that matter most for the tail (Valjevo 2023, Madrid, Oradea) lie
  outside the tested area (DECISIONS §24). A rule tuned to German radars could miss failure
  modes of other networks, or remove real rain in climates it was not tested in.
- **RADKLIM is not fully independent of OPERA over Germany.** It is reprocessed from the
  same DWD radars, with its own clutter correction and gauge adjustment.
  - *An artefact in the raw radar data that RADKLIM's correction also misses* is "confirmed"
    by the presence test, so the test can pass contamination as rain.
  - *Real rain that RADKLIM's own filters remove* looks like an OPERA artefact, so the test
    can credit a rule for removing rain.
  - The gauges are independent of the radar, but they are point measurements (below).
- **The presence test checks whether there is rain, not how much** (§4). RADKLIM
  underestimates peaks, and its adjustment is interpolated between gauges. It confirms only
  27% of the 50–100 mm values kept in screened OPERA, and has 7 pixel-hours ≥ 100 mm where
  screened OPERA has 43. So the screen's effect on peak *magnitude* is not validated, only
  the *presence* of the rain it keeps or removes.
- **The presence test has a disagreement floor.** Ordinary kept 10–50 mm values are
  "RADKLIM-dry" 5–6% of the time, from timing and position mismatch. A rule's removals cannot
  be shown to be more than ~94% artefacts. Pass bars (85% dry) sit below that floor, so up to
  ~15% of a passing rule's removals may be real rain. Example: the burst rule removes 308
  pixel-hours at 20–50 mm (1.0% of that bin), 87.7% of them RADKLIM-dry (§26), so ~38 of them
  may be real rain.
- **Radar against point gauges is ill-posed in the tail.** Inside a convective core a point
  gauge exceeds the 2 km pixel mean by construction. So gauge tests use the *gain* in
  agreement on a fixed set of pairs (§22), not absolute agreement, and tail magnitudes are
  not compared. The gain is fragile: one DWD gauge on a clutter pixel once produced +0.10 of
  a +0.108 gain, so every gain is reported with the most influential station left out.
- **The test samples are small.** Rules were tuned on 199 calibration days, gauge tests use
  63 days, and RADKLIM tests use 192 days. The rarest bins (≥ 100 mm) hold tens of
  pixel-hours, so a rule's behaviour on the most extreme values rests on few cases. Intervals
  are 95% day-block bootstraps, which assume days are independent; multi-day events break
  that.

### 6.2 How the criteria were set

- **Pre-registration is internal.** Criteria were written into the decision log (DECISIONS)
  and committed before each test was run. The commit timestamps are the record. They were
  not registered externally, and the people who wrote the criteria also ran the tests.
- **Some criteria were amended after the results, and are labelled as such.** In §28 the
  radar-year denominator ("rain days") turned out degenerate once applied. It is reported as
  a specification error, with both the pre-stated and the corrected numbers.
- **The rules were designed by looking at the same archive.** Signatures were found on images
  of known failures (hypothesis generation), and acceptance was tested on independent data.
  The gallery verdicts decide nothing (§24). But the calibration days were chosen to contain
  failures, so rule thresholds may be tuned to those failures' particular forms.
- **Several variants were tested for some decisions** (four advection variants, four
  Mahalanobis thresholds). A selection rule fixed in advance picked the winner, and for the
  mask it was checked on two halves of the data. Testing several variants still raises the
  chance that one passes by luck.

### 6.3 What the screen cannot see

- **It targets overestimation and unphysical values, not underestimation.** Beam blockage,
  attenuation in heavy rain, beam overshoot at long range and bright-band effects lower or
  remove real rain and have no rule. Missed extremes are not restored. The dataset's tail is
  therefore biased low wherever these occur, most strongly in the Alps (below).
- **Contamination remains.** After the burst rule, about a third of kept values ≥ 75 mm over
  Germany are RADKLIM-dry (§5.3). Part of that is RADKLIM missing real peaks, so the true
  contamination rate is not known. It is bounded above by about one third.
- **Real extremes share some error signatures.**
  - An isolated convective cell can look like a burst or an isolated pixel cluster.
  - Stationary orographic rain can look like a persistent clutter pixel.
  - A single storm can lift a radar-day score.
  Every rule therefore reports its rejection rate by intensity bin (§17). For radar-days, the
  §28 rule removes 51 radar-days that fall on catalogued event days.
- **Radar-day exclusion is coarse and still open** (§28). It removes a whole radar-day, so
  good data from a radar with a local fault go with it. The isolated-pixel signal (`iso31`)
  cannot separate a radar that is always noisy from one that has failed. Against the
  neighbours only it flags 3,029 radar-days, mostly on coastal and island radars. Against the
  radar's own q99 it misses radars that fail often. Pending a RADKLIM test.

### 6.4 Coverage and product changes

- **Italian radars are not in OPERA.** No Italian radar appears in any composite's
  contributor list (`how/nodes`), so Italy is outside the dataset. FBK's `srt1_adj` (2018
  on) could add it.
- **Austria never contributes; Switzerland and Hungary only from 2014.** Those areas have
  valid composite data, but it comes from distant foreign radars whose beams pass over the
  Alps. Extremes there are probably underestimated and are kept, not screened (§27). Other
  countries join or leave: Latvia from 2015, Serbia 2017–2024, Greece from 2021.
- **Which radars contribute is known from one composite per month.** A radar that was down
  at every sampled time is missed. Nov 2012 to Jan 2013 are missing from the archive.
- **The network changes over time.** Radars are added, retired and upgraded, so error
  statistics are not stationary. Per-year climatologies and ceilings absorb part of this.
  A model trained on the whole period mixes network states.
- **Two products.** OPERA switched from ODYSSEY to NIMBUS on 2024-07-05. NIMBUS has ~30–40%
  less wet area, 8–20% higher median intensities, a correlation of only 0.65–0.80 with
  ODYSSEY in log space, and about half the +15 min persistence (§18). The screen was
  calibrated mostly on ODYSSEY days. NIMBUS forms its own split, so the shift is tested
  rather than averaged in, but rule thresholds were not re-derived for NIMBUS.

### 6.5 The target itself

- **The plain hourly sum beads fast-moving cells.** Four 15-min snapshots of a moving cell
  sum to four blobs, which inflates small-scale structure. Over Germany this adds +2 to +5 to
  the Euler characteristic at 2 mm and 1.2–2x power at 6–16 km, relative to RADKLIM, growing
  with motion speed (§21). We keep the least-processed field as the target (§25) and store
  the advection-corrected field alongside. This matters for a structural loss built on
  Minkowski functionals: part of the "structure" in the target is a sampling artefact.
- **A pixel-hour needs all four frames.** Hours with a missing frame are dropped. If radar
  outages correlate with severe weather (power or communication loss in storms), the most
  intense hours are under-sampled. *[verify]*: outage rate as a function of rain intensity
  is not measured.
- **Frame timing.** An ODYSSEY frame t covers t−10..t+5 min, so our hour H is effectively
  H−55..H+5. Gauges match best with a 10-min shift. NIMBUS frames are snapshots at the
  nominal time, so the effective window differs between products.
- **Magnitudes are radar magnitudes.** OPERA reads ~0.44 of the gauge value on heavy hours,
  partly from point-versus-area support. The screen corrects no magnitudes, so return levels
  from this dataset are radar return levels, not rain-gauge ones. The Z–R relation used in
  the OPERA product is *[verify]*.
- **Scales below ~6 km are product-specific.** OPERA and RADKLIM do not corroborate each
  other below ~6 km wavelength, so the target's geometry at those scales depends on the
  product.

## 7. Open items (EXPERIMENTS §5)

- *Detection ideas not yet implemented:* satellite cloud-top veto for candidate extremes,
  lightning, Lagrangian consistency, storm-object plausibility, regional curated products.
- *Before the full-archive build:*
  - ~~radar-ownership geometry from the contributing radars~~ (done 2026-10-09, §27; radar
    pass rerun finished 2026-10-09 23:28);
  - radar-day exclusion: pre-stated rule applied 2026-10-13, result under review (§28);
  - the readiness gate with gauge false-alarm and miss rates (DECISIONS §20, §24);
  - ~~hourly split and store scripts~~ (done 2026-10-13: `make_splits_hourly.py`,
    `build_store_hourly.py`; 3-day test build verified);
  - the hourly tail threshold.
