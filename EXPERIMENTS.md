# Experiment tracker

Last updated: 2026-09-13

**Read first:** `DECISIONS.md` (why things are the way they are, and what is already ruled
out), then `CLAUDE.md` (operational traps), then this file (what exists and what's next).
`MINKOWSKI_DOCS.md` documents the loss itself.

Conventions: paths relative to repo root. Deterministic evals on the full test set
(n = 285,383). Generative evals on 4,096 patches, M = 16, heun-16 unless noted.
**All tail columns at POT u = 31 mm/h** — rows at other levels are not comparable.

---

## 1. Checkpoints

### Deterministic backbones — `runs/sr_analytical/`

| Run | Loss | Weight | Status |
|---|---|---|---|
| `UNet_Ana_20260623_144958` | MSE only (vanilla) | 0 | **reference backbone** for all FM runs |
| `UNet_Ana_20260731_100128` | MSE + Minkowski | 1e-4 | **Study-1 winner** |
| `UNet_Ana_20260729_104637` | MSE + Minkowski | 1e-3 | **discard** — reward hacking (DECISIONS §2) |
| `UNet_Ana_20260812_085343` | MSE + spectral | shared | v1, inert |
| `UNet_Ana_20260817_113622` | MSE + ssim | shared | v1, inert |
| `UNet_Ana_20260829_033325` | MSE + spectral | per-loss | v2, still inert |
| `UNet_Ana_20260827_151319` | MSE + opticalflow | per-loss | v2, still inert |
| `UNet_Ana_20260826_041128` | MSE + wetarea | per-loss | v2, still inert |
| `UNet_Ana_20260824_155753` | MSE + ssim | per-loss | v2, still inert |

### Flow matching — `runs/sr_flow_matching/`

| Run | Backbone | σ_r | Status |
|---|---|---|---|
| `flow_matching_20260716_071518` | vanilla | 0.01898130588233471 | **clean reference model** |
| `Extremes_20260807_111521` | Minkowski (0731) | 0.019811755046248436 | Study-2 cell |

### Reward fine-tuning — `runs/sr_fm_minkowski/`

| Run | Mode | Status |
|---|---|---|
| `Extremes_20260804_170256` | distributional | first coupling; null-to-negative |
| `Energy_20260811_171310` | energy + retention (w=1.0) | **best coupling so far** |

---

## 2. tab:iid — in-distribution extremes (POT u = 31)

| Model | RMSE_ext | FSS@89 | RAPSD | Mink | Exc. ratio | RL bias | Anisot. |
|---|---|---|---|---|---|---|---|
| Bicubic | 3.77 | 0.000 | 4.00 | 2.85 | — † | — † | 30.9 |
| Backbone (MSE) | 3.59 | 0.040 | 3.10 | 2.12 | 0.118 | 0.643 | 1.87 |
| Backbone + spectral (v2) | 3.68 | 0.034 | 3.89 | 2.23 | 0.057 | 0.719 | 2.24 |
| Backbone + ssim (v2) | 3.68 | 0.039 | 3.83 | 2.23 | 0.059 | 0.701 | 2.14 |
| Backbone + wetarea (v2) | 3.72 | 0.032 | 4.59 | 2.42 | 0.036 | 0.748 | 2.57 |
| Backbone + opticalflow (v2) | 3.67 | 0.036 | 4.03 | 2.25 | 0.057 | 0.713 | 2.36 |
| Backbone + Minkowski | 3.70 | 0.159 | 1.31 | 0.74\* | 0.244 | 0.445 | 0.73 |
| FM (clean) | 3.01 | 0.289 | 0.79 | 0.84 | 0.478 | 0.165 | 1.31 |
| FM + Minkowski reward | 3.01 | 0.268 | 0.62 | 0.82\* | — ‡ | — ‡ | 1.31 |
| FM + Minkowski energy | 3.02 | 0.242 | 0.63 | 0.82\* | 0.523 | 0.176 | 1.31 |
| FM on Minkowski backbone | 3.04 | 0.148 | 0.81 | 0.82\*\* | 0.446 | 0.236 | 1.29 |

\* training objective, not independent evidence. \*\* conditional mean trained on the
Minkowski loss → partly circular. † bicubic fitted at another level; rerun at u=31.
‡ evaluated at u=53; rerun at u=31.

GPD ξ error — **secondary diagnostic only**, see DECISIONS §10:
MSE 0.029 · spectral 0.013 · ssim 0.009 · wetarea 0.007 · opticalflow 0.053 ·
Minkowski 0.099 · FM clean 0.079 · FM energy **0.043** · FM on Mink backbone 0.051.

## 3. tab:prelim — baseline metrics (full test set)

| Model | MAE | MAE_ext | Mink | Isoperim % | RAPSD | SAL S |
|---|---|---|---|---|---|---|
| vanilla | 0.0499 | 0.4030 | 2.12 | 4.57 | 3.05 | 1.27 |
| Minkowski (1e-4) | 0.0519 | 0.4195 | 0.74 | 3.53 | 1.30 | 0.34 |
| spectral v2 | 0.0502 | — | — | — | — | — |
| ssim v2 | 0.0503 | — | — | — | — | — |
| wetarea v2 | 0.0503 | — | — | — | — | — |
| opticalflow v2 | 0.0501 | — | — | — | — | — |
| FM (clean) | 0.056† | 0.423† | 0.83‡ | 4.29‡ | 0.79‡ | 0.50‡ |

† ensemble mean, ‡ single arbitrary member. FM row on a 25,600-patch subset.

**Activity check:** an auxiliary loss costing < ~1% MAE against vanilla has not entered the
objective; its structural columns say nothing. All four competing losses fail this at both
weights tried.

---

## 4. Tooling

| Script | Purpose |
|---|---|
| `scripts/hpc/launch_unet_analytical.sh` | train a backbone: `<params> <w_geom> <data_pct> [<config>]` |
| `scripts/evaluate/eval_extremes.py` | tail protocol → tab:iid row |
| `scripts/evaluate/eval_backbone.py` | baseline metrics → tab:prelim row |
| `scripts/evaluate/eval_fm.py` | FM baseline metrics |
| `scripts/hpc/eval_losses.sh` | batch-evaluate all backbones, with activity check + POT guard |
| `scripts/evaluate/make_plots.py` | all comparison figures from eval outputs |
| `scripts/evaluate/diagnose_coupling.py` | is the reward wired correctly? (minutes, one batch) |
| `scripts/hpc/sweep_reward.sh` | one-factor-at-a-time reward sweep |
| `scripts/hpc/check_sampler.sh` | sampler-fidelity check |
| `src/sweep_util.py` | `patch` a config / `collect` a summary into CSV |

Figures: perception–distortion plane, RAPSD with ratio panel, GPD return levels + exceedance
counts, tail/safety dot chart, SAL plane, **γ curves**, **γ residual by threshold**,
isoperimetric scatter.

---

## 5. TODO

### Highest value
- [ ] **Tail-sample selection.** Score a large patch pool by patch max, draw the *structural*
      loss batches from the top few %. Directly targets the pixel-mass bottleneck
      (DECISIONS §5, §13). Keep MSE / velocity loss on uniform batches.
- [ ] **Gradient-norm λ rule** for all auxiliary weights (DECISIONS §9).
- [ ] Re-run competing losses at a weight that passes the activity check, or conclude they
      cannot be made active without hacking.

### Paper
- [ ] Add the W₁ identity to the theory section (DECISIONS §13).
- [ ] Report exceedance ratio + RL bias as tail columns; demote ξ to the text with the
      confound explained.
- [ ] State the POT-level choice as estimability, not physics.
- [ ] Write up the w=1e-3 hacking episode as a positive result on necessary-not-sufficient.

### Data — quality and size
- [ ] **Artefact screening.** The patch pool contains a substantial number of patches that
      are not precipitation but bad radar observations. They concentrate at the top of the
      intensity distribution, so *any* selection that ranks on patch max draws them
      preferentially — the first field-panel selection picked three artefacts out of three.
      This contaminates the tail the whole project is about: the POT fits, the exceedance
      ratio and the `--n_extreme` figure patches all read from that same top slice.
      Needed: a quality score per patch (candidates — speckle / isolated-pixel fraction,
      implausible spatial gradients, the ring and spoke geometry typical of radar artefacts,
      disagreement with neighbouring time steps) and a screened patch list. Until then, look
      at the images before believing anything driven by the extreme top of the distribution.

      First measurements on the test split (285,383 patches, from `backbone/vanilla` arrays):

      | statistic | u=31 | u=53 | u=89 |
      |---|---|---|---|
      | patches above u | 25,591 (8.97%) | 13,949 (4.89%) | 6,349 (2.23%) |
      | share with `tmean/tmax` < 1e-3 | 25.9% | 27.8% | 29.3% |
      | share with `tmean/tmax` < 1e-2 | 85.8% | 90.1% | 92.6% |
      | median patch mean | 0.183 mm/h | 0.231 mm/h | 0.294 mm/h |

      `tmean/tmax` is a cheap concentration proxy — near zero means the whole patch mass sits
      in a few pixels. The share of such patches **rises** with the threshold, which is
      backwards: a stronger convective cell wets *more* area, not less. Spot checks agree —
      the p97 patch has target max 74.9 mm/h with an input max of 0.4 mm/h over a sea-level
      DEM (sea clutter), and the p95 patch is a single 52 mm/h speck in an otherwise dry
      field. So the contamination is not confined to the extreme top; it reaches well down
      into the POT range at u=31.

- [ ] **Everything above 150 mm/h is set to zero, not clipped.** Located:
      `config.yaml: DECLUTTER_THRESHOLD: 150.0` feeding
      `src/data/preprocessing.py::filter_precip_bounds`, whose mask is
      `(arr < drizzle) | (arr > declutter)` followed by `arr[mask] = 0.0`. So a genuine
      200 mm/h cell is not reduced to 150 — it is turned into a **dry pixel**, and the
      surrounding storm keeps its structure with a hole punched in the middle. The observed
      maximum of exactly 150.000 mm/h (15 patches, none above) is the signature of that
      boundary, not of a physical limit: the downloaded OPERA composite records rates
      continuously with no such ceiling, so the truncation is entirely ours.

      This is not a cosmetic issue for a project about extremes. The GPD is fitted on
      exceedances drawn from a tail that has been both censored and holed, so `gpd_xi_obs`
      and every return level derived from it are biased by an unknown amount, and the
      "binding limit is pixel mass" finding was measured on data with its heaviest pixels
      deleted. Decide deliberately whether to clip, to drop the affected patches, or to
      screen on the quality index instead (which is what the threshold was reaching for),
      and re-derive the tail numbers afterwards.
      (Separately, 66,992 patches — 23.5% — are completely dry, `tmax` exactly 0.)
- [ ] **Enlarge the dataset**, especially for o.o.d. extremes (tab:ood is still deferred
      below, and the current pool is too small to hold out a genuine o.o.d. tail after
      artefact screening removes part of it). The full OPERA archive (2012–) is now openly
      available: the EUMETNET Open Radar Data API is fully operational on MeteoGate and IP
      whitelisting is no longer required. Anonymous access works without credentials at a
      low rate limit; an API key from https://devportal.meteogate.eu/ raises it. Endpoint
      `https://api.meteogate.eu/eu-eumetnet-weather-radar`, OGC EDR format, composites under
      location id `0-20010-0-OPERA`; bulk files sit on S3 at `s3.waw3-1.cloudferro.com`.
      Note the old endpoints stop working after 30 June.

      **Access verified — one file downloaded and inspected end to end.** Two separate
      services, and the useful one needs no credentials at all:

      - *EDR API*, `https://api.meteogate.eu/eu-eumetnet-weather-radar`. Needs the API key
        (`-H "apikey: ..."`), which raises the limit from 200/h anonymous to 2000/h. It only
        covers a **24-hour rolling window** and returns metadata plus S3 links, not data.
        Use it for discovery only.
      - *S3, anonymous, no key and no signing* — this is the bulk path:
        - `openradar-24h`  — rolling 24 h
        - `openradar-archive` — **2012 through 2026, complete**, anonymously listable
        - layout `<bucket>/YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@<PRODUCT>.h5`
          (also `.tiff`). Archive products are named `QIND_RATE`, `DBZH_QIND`, `ACRR_QIND`;
          the live bucket uses plain `RATE`.

      `scripts/data/fetch_opera_archive.py` implements this: `--list` reports what the
      archive holds over a date range without downloading, and the default mode downloads
      RATE + QIND, reprojects onto the grid taken from an existing raw zarr day, and writes
      one per-day zarr store in the layout preprocessing already reads. It needs `h5py`,
      which is **not** in `dl-stable` (`micromamba install -n dl-stable -c conda-forge
      h5py`); listing works without it.

      A downloaded composite is ODIM HDF5, grid 2200x1900, projection
      `+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84` — the same
      LAEA grid as `europe_dem_laea.tif`, so it drops into the existing preprocessing.
      ~1 MB per 15-minute composite, so roughly 34 GB/year and ~513 GB for the whole
      RATE archive; `/work` has 21 TB free, so volume is not the constraint.

- [ ] **What the raw source actually contains** (measured on the fetched 2018-06-15, 96
      composites). The archive is far dirtier than the patch set suggests, because the
      declutter step hides it: per-timestep maxima exceed 500 mm/h in **52 of 96** steps and
      peak at **13,072 mm/h**, while the worst step has only 14 pixels above 150 mm/h and 1
      above 2000. So the offenders are isolated pixels, i.e. clutter, and `DECLUTTER_THRESHOLD`
      is doing real work — the objection is to *how*, since zeroing them punches holes in
      otherwise good fields instead of rejecting the pixel or the patch. Any rebuild needs an
      explicit outlier policy; there is no threshold-free version of this dataset.

- [ ] **Use the OPERA quality index for artefact screening — it is already in the files,
      but it is not sufficient on its own.**
      Each composite carries a companion quality field (`pl.imgw.quality.qi_total`, values in
      [0,1]; a separate `QIND` dataset in the archive products). On the test composite it
      separates the way the artefact hypothesis predicts on the 2026 live composite: 21.0% of
      all valid pixels have QI < 0.8, but **41.7%** of pixels at or above 31 mm/h do — extreme
      pixels twice as likely to be flagged. It costs nothing extra to fetch and our current
      patches discard it, having been built from the rate field alone.

      The separation is not uniform, though. On 2018-06-15T16:30 the pixels above 500 mm/h
      had mean QI 0.267 against a field mean of 0.282 — essentially no signal. So treat QI as
      one feature among several rather than the screen itself, and validate it per period
      before relying on it; the quality algorithms behind it have changed over the archive.

- [ ] **The 150 mm/h cap is ours, not OPERA's.** The downloaded composite has no such
      ceiling (its own max was 81.6 mm/h with values recorded continuously). So the exact
      150.000 cap seen in the patch dataset is imposed somewhere in our preprocessing, which
      means it is ours to remove when the dataset is rebuilt.

### Data hygiene
- [ ] Rerun bicubic and `FM + Minkowski reward` at u=31 to fill their dashed tail columns.
- [ ] Delete duplicate eval dirs (`backbone_mse` = `backbone_vanilla`,
      `backbone_mink_1e-4` = `backbone_minkowski`, `*_v1` duplicates).
- [ ] Re-run evals so the new `gamma_target` / `pot_threshold` arrays exist for the γ plots.
- [ ] Eval `Energy_.../fm_mink_best.pth` to quantify the selection bug vs `latest`.
- [ ] Try `REWARD_FM_RETENTION_WEIGHT` 3 or 10 — FSS still fell at 1.0.

### Deferred
- [ ] b0 (persistent homology) confirmatory run at the winning weight
- [ ] o.o.d. protocol → tab:ood
- [ ] Compute proposal: the scale argument rests on DECISIONS §5–6
