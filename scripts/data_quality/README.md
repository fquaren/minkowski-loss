# Data-quality audit

Scripts to understand what the OPERA archive actually contains before deciding what to keep
for training and evaluation. They **describe** the data; they do not delete or rewrite
anything. The screening decision is made afterwards, from what they show.

Everything runs on the **raw** day stores (`RAW_OPERA_DATA_DIR/YYYYMMDD/`), on the same
stride-128 tile grid as `generate_metadata.py`. So the results cover the whole downloaded
archive, not only the 2023–24 patch set, and they see the field *before* our
drizzle/declutter step. Tile rows are keyed `timestamp,row,col` exactly as in
`*_patches_metadata.txt`, so they join onto the current splits.

| Script | Question it answers | Output |
|---|---|---|
| `clutter_climatology.py` | Which pixels "rain" far more often than their neighbours? (ground clutter, wind farms, sea clutter, RLAN spokes) | `clutter_climatology.npz`, frequency maps, `clutter_hotspots.csv` |
| `compute_tile_features.py` | Per tile: speckle, spikes, gradients, concentration, sea/DEM context, spokes, flicker against t±15 min, declutter impact, QIND, static clutter at the peak | `features/YYYYMMDD.csv.gz` |
| `audit_splits.py` | How independent are train / val / test? | `split_audit.md` |
| `summarize_features.py` | Where do artefact signatures sit: by intensity, year, month, hour, tile, split? How much do the rules overlap? How good are they against labels? | `summary.md`, CSVs, figures |
| `patch_gallery.py` | What do the tiles look like? (t−15 / t / t+15 raw, filtered target, coarse input, DEM with lat/lon ticks, QIND, locator map; row label = tile centre + nearest radar) | image pages + `labels.csv` |
| `radar_attribution.py` | Which radars are consistently bad? Hot pixels, > 500 mm/h pixels and flagged tail tiles attributed to the nearest active OPERA radar, per year | `radars/radar_summary.{md,csv}`, `radar_by_year.csv`, `radar_map.png` |

Feature definitions are in `src/data/quality.py` (unit tests in `tests/test_quality.py`).
Grid geography (orientation, lat/lon, the DEM in the precipitation orientation, OPERA radar
sites) is in `src/data/geo.py`. **Orientation:** the precipitation stores are south-first,
while the DEM GeoTIFF is north-first. Feature files written before 2026-09-28 used the
mirrored DEM, so their `sea_frac`, `dem_mean`, `argmax_over_sea` and `wet_over_sea_frac`
columns (and the `sea_clutter` rule) are wrong until they are recomputed. The radar database
(`OPERA_RADARS_DB.json` and the archive `OPERA_RADARS_ARH_DB.json`) is in `OPERA/meta/`.
The candidate rules are in `rules.py`. **Their thresholds are first guesses.** Treat a
rule's rate as the prevalence of a signature until it has been checked against labels.

## Order

`scripts/hpc/launch_data_quality.sh` runs stages 1–5 under `setsid`:

1. `clutter_climatology.py` over every complete day store.
2. `compute_tile_features.py --climatology …`, so the `clim_*` features exist.
3. `audit_splits.py`.
4. `summarize_features.py`.
5. Galleries: `top_max`, `random` in the tail, `unflagged` in the tail (misses), one per rule.

Then the loop that decides the screen. **No labelled set** (DECISIONS §17, 2026-09-28):
the policy is to reject *clear errors only* and keep imperfections, so the rules are tuned by
eye and by their rejection profile, not by precision/recall against labels.

6. For each rule, look at its gallery of rejections and at the unflagged tail gallery. This
   is a quick sanity check by eye. It asks whether the rule fires on something
   non-meteorological by construction, not a labelling pass.
7. Check each rule's rejection rate by intensity bin, region and year in `summary.md`. A rate
   that climbs steeply with intensity is the warning sign that the rule is removing real
   extremes. Tighten it, or restrict it to geometry / persistence / impossibility signatures.
8. Tune the thresholds in `rules.py` and repeat.

The label sheet and `summarize_features.py --labels` still work, if a small labelled check
is ever wanted.

Cost: about 1 s per time step per core for the features, so the ~560 current days take about
2.5 h on the default 6 workers (the node allows 8 cores in total, and the fetcher holds 2). The climatology is I/O-bound and takes about 30 min. Both scanners skip day
stores without `.zmetadata`, so they are safe to run while the fetcher is writing.

## What a first look already shows

This is a smoke test on 5 days: 2024-07-01..03, 2024-01-15 and 2012-09-04, every 4th step,
with a 3-day climatology. It is too small for numbers, but enough to see the shape.

- **The any-flag rate rises with intensity:** 14% of tiles at 1–10 mm/h, 44% at 31–53,
  69% at 89–150, and 100% above 150 mm/h. That is the same wrong-direction trend
  EXPERIMENTS.md saw in `tmean/tmax`.
- **The rules are mostly complementary in the tail** (Jaccard < 0.3 for most pairs). The
  exception is `isolated_peak` with `flicker` (0.44). No single signature is a screen.
- **2012 is much dirtier than 2024** (tail any-flag 90% against 46%). The QIND field
  exists for only part of the covered area (NaN on about 93% of pixels against 57% for
  the rate), and it is low over most of the speckle.
- **A handful of tile locations flag almost every tail tile**, for example (512, 640) and
  (1024, 384). One pixel exceeded 31 mm/h in 35% of all time steps over the 3 days, which
  is static clutter.
- **Tile-level rejection loses real storms.** The gallery shows genuine convective cells
  with a spoke or speckle across the same tile. That argues for masking pixels plus
  rejecting only tiles that are mostly artefact, not dropping every flagged tile.

## Known limitations

- No radar-site geometry: rings and spokes are detected only through shape (elongation)
  and QIND, not through range or azimuth from a site.
- `persist_*` misses fast-moving or fast-developing cells at the ±12 km window. Check
  `flicker` in the gallery before trusting it.
- `static_clutter` depends on the period the climatology covers. A climatology built on
  only a few days flags recurring *weather* too.
- Features at the tile edge see NaN-padded neighbourhoods. Partial-coverage tiles are
  excluded by default (`--min_coverage 1.0`), matching the current patch definition.
