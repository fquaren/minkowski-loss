# The event test set: which storms, why, and what is missing

Catalogue: `configs/prominent_events.yaml` (26 events, sources per entry, checked 2026-09-28).
Used by `scripts/dataset_v2/make_splits.py`. Status as of the v2 build of 2026-10-01
(`OPERA/v2/report.md`).

## 1. What the event set is for

The v2 splits are blocked by ISO week and drawn per calendar month, so test is no longer the
neighbouring frames of training storms (DECISIONS §15). That makes test *independent*, but
not *hard*: a random week is mostly ordinary weather. The event set adds a fixed, named,
literature-documented sample of the storms the project is about, so that:

- every model is scored on the same recognisable cases, which can be shown as figures and
  compared with published analyses;
- no patch of these storms can reach training: each event window ±1 day is forced into
  test (ODYSSEY) or falls in `nimbus`, **together with its whole ISO week, Europe-wide**;
- `events_test` / `events_nimbus` hold *every* covered tile that overlaps the event box during
  the window, never subsampled, so whole storm sequences are available (also for the
  temporal-consistency TODO in EXPERIMENTS).

The event set is a **curated test of known extremes**, not an o.o.d. split. Every event
type in it also occurs in training at lower intensity; it measures E1-style skill on
documented cases, not E2 novelty (RESEARCH_NOTES §2). The o.o.d. options are in
RESEARCH_NOTES §7.4.

## 2. How the events were chosen

The criteria, reconstructed from the catalogue (they were applied when it was assembled, but
not written down at the time):

1. **Documented in the literature or by an agency**, with dates and an affected region that
   can be checked: peer-reviewed case studies (NHESS, MWR, BAMS, Wea. Forecasting), ECMWF
   newsletters, MeteoSwiss blog analyses, ESSL reports, AEMET / Météo-France / DWD records.
   This makes the window and the box defensible without our own event detection, and gives
   an independent description to compare model output against.
2. **Inside the archive and the OPERA domain**: 2012-09 → 2026, with radar coverage over the
   affected area (not guaranteed; see §4).
3. **Spread over storm types**, so that the set is not one regime repeated:
   linear MCS / derecho / bow echo, supercells with large hail, stationary and back-building
   convection (flash floods), Mediterranean heavy-precipitation episodes, orographic events,
   and stationary-low / cut-off-low floods.
4. **Spread over years and both products**: 2012 → 2024, so the set crosses the 2015 and
   2017 product breaks, and two events (Boris, Valencia) are NIMBUS.
5. **Weighted towards the Alps and Switzerland** (5 MeteoSwiss-documented events, 3 more in
   N Italy), because that is where the MCH discussion and CombiPrecip can supply an
   independent reference.

## 3. The events

Tiles = covered 128×128 tiles overlapping the box in the window; ≥ 31 = of which with a
cleaned max ≥ 31 mm/h; max = largest cleaned tile max (mm/h).

| Event | Window | Type | Why it is in | Split | Tiles | ≥ 31 | Max |
|---|---|---|---|---|---|---|---|
| HyMeX IOP6, S France | 2012-09-23..25 | fast linear MCS | best-documented Mediterranean MCS of HyMeX SOP1 | test | 1,370 | 150 | 273 |
| HyMeX IOP16, NW Med | 2012-10-25..28 | Mediterranean HPE | second HyMeX flagship, flash floods | **none** | 0 | 0 | — |
| Pentecost storm "Ela" | 2014-06-08..10 | derecho / bow echo | archetypal W-European bow echo | test | 2,015 | 693 | 365 |
| Münster cloudburst | 2014-07-27..29 | stationary convection | 292 mm in 7 h, German record class | test | 576 | 315 | 151 |
| Hérault / Montpellier | 2014-09-28..30 | Mediterranean, stationary | 252 mm in 3 h, red alert | test | 288 | 46 | 205 |
| Côte d'Azur (Cannes) | 2015-10-03..04 | Mediterranean, stationary | 175 mm in 2 h, flash flood | test | 191 | 66 | 178 |
| Elvira / Friederike | 2016-05-27..06-07 | stationary convection | Braunsbach and Simbach flash floods, a 12-day sequence | test | 13,809 | 2,889 | 453 |
| Poland derecho | 2017-08-11..12 | derecho | mesocyclone-born derecho, Central Europe | test | 946 | 134 | 286 |
| Aude floods | 2018-10-14..15 | Mediterranean, stationary | 295 mm in 12 h | test | 384 | 79 | 128 |
| Storm Alex | 2020-10-01..03 | back-building, orographic | > 500 mm / 24 h, Alpes-Maritimes and Piemonte | test | 159 | 69 | 375 |
| Swiss supercell | 2021-06-20 | supercell, hail | MeteoSwiss 2021 hail season | test | 192 | 102 | 167 |
| Moravia tornado | 2021-06-24 | supercell, tornado, hail | IF4 tornado supercell | test | 96 | 62 | 319 |
| Lucerne hail | 2021-06-28 | supercell, hail | 10 cm hail at Wolhusen | test | 190 | 91 | 181 |
| CH / N Italy hail | 2021-07-08 | supercell, hail | 11 cm hail near Milan | test | 284 | 196 | 178 |
| Low "Bernd" | 2021-07-12..15 | stationary low | Ahr / Erft / Vesdre floods | test | 1,534 | 356 | 452 |
| Mediterranean derecho | 2022-08-17..18 | derecho, hail | Corsica → Czechia, long-track | test | 2,429 | 910 | 488 |
| Emilia-Romagna, 1st | 2023-05-01..03 | stationary low | catastrophic floods | **none** | 0 | 0 | — |
| Emilia-Romagna, 2nd | 2023-05-15..17 | stationary low | 40 h of extreme rain | **none** | 0 | 0 | — |
| N Italy giant hail | 2023-07-19 | supercell, hail | 16 cm hail | test | 154 | 144 | 487 |
| La Chaux-de-Fonds / Azzano | 2023-07-24..25 | supercell, downburst, hail | 217 km/h downburst, 19 cm hail (European record) | test | 658 | 476 | 493 |
| Slovenia floods | 2023-08-03..06 | stationary convection | > 200 mm in 12 h | test | 1,125 | 564 | 464 |
| Southern Germany floods | 2024-05-31..06-03 | stationary low, orographic | > 150 mm / 24 h | test | 1,523 | 607 | 483 |
| Misox | 2024-06-21..22 | back-building | 63.7 mm in 60 min at Grono | test | 174 | 109 | 460 |
| Maggia / Valais | 2024-06-29..30 | orographic | floods and debris flows | test | 370 | 144 | 490 |
| Storm Boris | 2024-09-12..16 | stationary low, orographic | 442 mm / 3 d in N Czechia | nimbus | 4,747 | 1,547 | 486 |
| Valencia DANA | 2024-10-28..30 | cut-off low, back-building | 771.8 mm in 14 h, 184.6 mm/h gauge-hour | nimbus | 200 | 76 | 115 |

`events_test`: 28,467 patches; `events_nimbus`: 4,947.

## 4. Problems found on 2026-10-02

1. **Three events have no tiles.** Causes in §5.
2. **The two 2013 flagship events are in *train*.** The catalogue header says to add the
   May–June 2013 Central European floods and the 27–28 July 2013 "Andreas" hailstorms once
   2013 was on disk. It is now, fully covered (CE floods: up to 421 mm/h raw, ~500–1,000
   pixels ≥ 31 mm/h per day; Andreas: 181 mm/h), but they were never added, so
   2013-05-29..06-04 and 2013-07-26..27 are training days in v2. Fixing it means adding
   both to the catalogue and rerunning splits → store → gamma (~16 h), **before** any v2
   training run.
3. **Ten event maxima sit at 450–493 mm/h**, just under the 500 mm/h rejection bound.
   With boxes this generous, the maximum is likely an artefact or hail contamination
   somewhere in the box, not the event core. Look at the top tiles of each event
   (`patch_gallery.py`) before quoting a per-event peak.
4. **Several events are accumulation extremes, not rate extremes.** Emilia-Romagna in OPERA
   peaks at 11–25 mm/h with no pixel ≥ 31 mm/h in the box. Bernd, Boris and the Southern
   Germany floods are also long-duration rain. Valencia's radar peak (115 mm/h tile max,
   28.6 mm/h at Turís) is far below the gauge. These test structure at moderate rates over
   long sequences, not the tail. Tag each event `rate` or `accumulation` so the two kinds are
   reported separately (the same distinction as the rate-vs-accumulation question for MCH,
   RESEARCH_NOTES §7.4).
5. **Coverage bias.** France, Germany, Switzerland and N Italy dominate. Nothing from the
   UK and Ireland, Scandinavia, the Iberian interior or the Balkans except Slovenia. Partial
   radar coverage at network edges and coasts (§5) adds to it.

## 5. Why the three events are empty, and how to include them

**HyMeX IOP16 (2012-10-25..28): not in the archive.** The archive has no data from
2012-10-10 to 2012-11-08 (the 2012-10 → 2013-09 planner found 268 days with data; the
fetch log jumps from 2012-10-09 to 2012-11-09). Nothing in OPERA can recover it.
**Proposal:** keep it in the YAML marked `available: false`, so the record stays, and drop it
from the counts. Replace it with another Mediterranean HPE that has data (Alex, Aude,
Montpellier and Cannes already cover the type).

**Emilia-Romagna (both episodes): covered, but never *fully* covered.** The rule is the v1
one: a tile is valid only if all 128 × 128 pixels are finite at that time step, on the fixed
stride-128 grid. The two grid tiles over the box carry **static** holes (pixels never valid
in any frame): 18 pixels (0.1%) in tile (512, 1024) and 272 pixels (1.7%) in tile
(512, 896). So no frame ever qualifies. 182 shifted 128-pixel windows (stride 8) overlap the
box and are fully covered all day.

Europe-wide, this is not specific to Italy. On a 12-day sample (every 2 h, 2014–2026):

| NaN fraction of the tile | share of tile-frames | tiles with max ≥ 31 |
|---|---|---|
| 0 (valid today) | 40.3% | 1,673 |
| (0, 0.1%] | 2.8% | 97 |
| (0.1, 1%] | 3.3% | 162 |
| (1, 5%] | 6.7% | 232 |
| (5, 25%] | 9.0% | 246 |
| (25, 100%) | 38.0% | 237 |

Admitting tiles with ≤ 1% NaN would add 15% more ≥ 31 mm/h tiles, and ≤ 5% would add 29%.
These sit at coasts and network edges, where Mediterranean heavy rain happens, so the
all-finite rule is also a geographic bias in the training tail.

Three options, in increasing cost:

- **A. Shifted windows for events only (recommended now).** For each event, slide a 128 × 128
  window (stride 16) over the box and keep windows fully covered in the frame. Select a
  non-overlapping subset that maximises the box's covered wet area, write them as extra rows
  of the `test` group with their own `(y, x)` (`build_store.build_day` already slices at any
  offset, and the DEM is looked up by position), compute their gamma targets, and add them to
  `events_test.txt`. Test-only, so no leakage question and no retraining. Recovers both
  Emilia-Romagna episodes. Cost: a small script and about an hour of compute.
- **B. Repair static holes (next store version).** A pixel that is never valid in a year's
  climatology (`n_valid == 0`) but enclosed by valid coverage is a blind spot (blockage, an
  excluded bin), not missing weather. Fill it from its neighbours, as the static-clutter
  repair already does and as EURADCLIM does. It belongs in `src/data/cleaning.py`, with a
  size limit (e.g. holes ≤ 50 px). Recovers the tiles whose only gaps are static, everywhere,
  not just for events. Needs a rebuild and a check that the filled pixels are not over a
  coastline edge.
- **C. Masked partial tiles (next store version, larger change).** Accept tiles with ≤ 1%
  (or ≤ 5%) NaN, and store a validity mask. The pixel loss and the metrics are masked. The
  Minkowski terms need care: a hole's boundary adds spurious perimeter and Euler
  characteristic, so either inpaint before computing γ, or compute the functionals inside
  the mask only. Recovers 15–29% more tail tiles and reduces the coastal bias, but touches
  the dataset, loss, gamma targets and evaluation.

**Recommendation:** A now, together with the 2013 fix (§4.2), in the same splits rebuild.
B and C belong to the next store version, with B first because it is self-contained.

## 6. For the paper: limitation, and the next step if it is not fixed

If the recovery above is not implemented before submission, it goes into the paper's
dataset section as a stated limitation, with the fix as future work. Draft text:

> **Coverage rule and coastal bias.** A 128 × 128 patch is used only if every pixel has
> radar coverage at that time step. This keeps the inputs and targets free of nodata, but it
> excludes patches at the edges of the radar network and along coasts, and patches over
> small permanent blind spots (beam blockage, excluded range bins). On a 12-day sample,
> admitting patches with ≤ 1% missing pixels would add 15% more patches with a maximum
> ≥ 31 mm/h, and ≤ 5% would add 29%. Because Mediterranean heavy-precipitation events
> concentrate along coasts and at the network edge, the training and test tails
> under-represent them. The same rule removes two documented events from the event test
> set (the Emilia-Romagna floods of May 2023, whose tiles carry 18 and 272 permanently
> missing pixels); a third (HyMeX IOP16, October 2012) is absent from the OPERA archive.
>
> **Possible improvements.** (A) For the event test set only, use shifted windows that are
> fully covered (182 such windows exist over the Emilia-Romagna area); test-only, no
> retraining. (B) Fill small permanent blind spots from neighbouring pixels during
> cleaning, as is done for static clutter. (C) Admit partially covered patches with a
> validity mask, masking the pixel loss and computing the Minkowski functionals on
> inpainted fields or inside the mask, since hole boundaries add spurious perimeter and
> Euler characteristic.

Numbers to refresh before quoting: the 12-day sample is small (2014–2026, every 2 h).
Measure the coastal share on the full scan tables (fraction of ≥ 31 mm/h tile-frames that
are partially covered, by distance to the coast and to the network edge) if the paragraph
makes it into the paper.
