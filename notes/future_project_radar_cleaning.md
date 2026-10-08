# Future project: label-free cleaning of radar composites by structural decomposition

Saved 2026-10-08 for a later project (researcher's request). Origin: the OPERA v4 screen in this
repo (DECISIONS §17-§24, EXPERIMENTS §5 "To explore", RESEARCH_NOTES §7.4e-f).

## The problem

The OPERA composite (2 km, 15 min, 2012 onward) is full of artefacts:
- ground clutter and anomalous propagation;
- RLAN / emitter rays and range rings;
- speckle;
- radar-wide failures and reflectivity ceilings.

They set the tail of hourly and daily sums. Hand-written rules catch some of them, but
they misfire on real storms, and they can be validated only where independent data exist.
The DE/CH gauges and RADKLIM cover about a third of the tail removals.

A cleaner whose decisions rest on a person's labels is not defensible (DECISIONS §24).

## The idea

Separate by **structure**, not by **rarity**. "Artefact = anomaly" removes the tail,
because real extremes are anomalous too. What separates the two is the symmetry each obeys:
- rain moves with the flow: it persists along the motion, with no relation to radar sites;
- artefacts are tied to the measurement geometry:
  - fixed on the grid (clutter, blocks);
  - radial or circular about a site (rays, rings);
  - range-dependent (calibration and receiver faults);
  - or present in a single frame (spikes, flashes).

Model, trained only to reconstruct the data:

    x_t = r_t + a_t
    r_t ~ W(u_t) r_{t-1} + smooth growth/decay     (advection-predictable rain)
    a_t sparse, persistent in the grid frame or in polar coordinates about each radar

The clean field is r_t.
- *Transient errors:* add self-supervised next-frame prediction.
- *Baseline:* robust PCA (low-rank + sparse) on one day of frames per radar, in polar
  coordinates around the site, compared with the rule-based persistence repair of this repo
  (`src/data/hourly.py`, local-peak persistence).

## Evaluation (no labels anywhere)

Independent curated products are used only for model selection and evaluation, never for
training:
- RADKLIM (DE);
- COMEPHORE (FR);
- CombiPrecip (CH);
- gauges.

The main test: a model fitted label-free on all of Europe must agree with RADKLIM in Germany
and COMEPHORE in France. The metrics are those of DECISIONS §22 / §24:
- the gain on a fixed pair set, with an influence check;
- affected pixels moved towards the reference;
- the removal rate per intensity bin.

## Known limits (not separable by structure)

- Stationary real rain (orographic, back-building) when the motion is ~0: the same blind
  spot as the persistence rule.
- Calibration and gain errors (a radar reading 3 dB high looks like rain).
- Value-level errors (ceilings): leave these to rules.

## Fallback: weak supervision without human labels

1. Disagreement with curated products (OPERA high, RADKLIM / COMEPHORE / CombiPrecip low).
2. High-confidence rule decisions not contradicted by independent data.
3. Physical vetoes (satellite clear sky, lightning).

Train on countries with a curated product, test on held-out countries.

## Literature (search 2026-10-08; full references in RESEARCH_NOTES §7.5)

No paper found that does this exact combination. That is an inference from a search, not
proof. Work to cite and differ from:
- **EURADCLIM** (Overeem et al. 2023, ESSD 15, 1441): same data, gauge evaluation,
  rule-based. Its yearly static-clutter mask is a crude persistence prior, and it is the
  baseline to beat.
- **Lepetit et al. 2022** (IEEE TGRS 60): U-Net restoration of single radars, weakly
  supervised by gauges.
- **Bøvith 2008** (DTU PhD thesis): clutter separated from rain by the motion-field
  difference; supervised, single radar.
- **Bölz et al. 2026** (EGUsphere preprint): label-free via synthetic mixing of cluttered
  and clean scans.
- **Scovell et al. 2013** (Odyssey QC): the operational OPERA QC already in the input.
- *Method:* robust PCA (Candès et al. 2011, J. ACM 58); DECOLOR (Zhou et al. 2013, TPAMI 35:
  low-rank + contiguous outliers + motion).

Cautions:
- Self-supervised predictors reproduce any structure they can predict: structured
  Noise2Void (Broaddus et al. 2020); seismic (Birnie et al. 2021). A next-frame model learns
  static clutter as signal.
- Anomaly detectors remove rare-but-real objects: Astronomaly (Lochner & Bassett 2021).
  Extreme-value QC checks flagged extremes against independent data before removing them:
  El Hachem et al. 2022 (HESS 26), HadISD (Dunn et al. 2012).

## What exists in this repo to reuse

- RADKLIM reader and mapping onto the OPERA grid: `src/data/radklim.py`,
  `scripts/data/fetch_radklim.py`.
- Hourly chain and rule screen as a baseline: `src/data/hourly.py`, `src/data/cleaning.py`,
  `src/data/radar_screen.py`.
- Gauge and RADKLIM evaluation harnesses: `scripts/data_quality/gauge_hourly.py`,
  `radklim_test.py`.
- Radar sites and geometry: `src/data/geo.py`, `radar_screen.RadarGeometry`.
