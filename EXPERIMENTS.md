# Experiment tracker

Last updated: 2026-09-23

**Read first:** `DECISIONS.md` (why things are the way they are, and what is already ruled
out), then `CLAUDE.md` (operational traps), then this file (the compute node in §0, what exists and what's next).
`MINKOWSKI_DOCS.md` documents the loss itself; `RESEARCH_NOTES.md` holds the theory
framework, the definition-of-extreme question and the regime-aware design notes.

Conventions: paths relative to repo root. Deterministic evals on the full test set
(n = 285,383). Generative evals on 4,096 patches, M = 16, heun-16 unless noted.
**All tail columns at POT u = 31 mm/h**, so rows at other levels are not comparable.

**Two caveats that apply to every number below** (details in §5 and DECISIONS §15):
- **The test split is not independent of train.** Splits are random at patch level over
  the same 15 months, so every test patch has a train patch at the same timestamp and on
  the same tile within ±1 h. Test scores measure interpolation within storms the model
  was trained on, not generalisation.
- **The tail is contaminated by radar artefacts and holed by the declutter step.** Tail
  columns are measured on data that has not been screened yet.

---

## 0. Compute node

All runs are on **node34**, which is **shared with other users**.

| | |
|---|---|
| CPU | 12 cores: 2 × Intel Xeon E5-2620 v3 (6 cores each, no SMT). NUMA node 0 = cores 0–5, node 1 = cores 6–11 |
| GPU 0 | RTX PRO 6000 Blackwell Max-Q (97.9 GB), PCI 02:00.0, UUID `GPU-a62d786a-…`. **Not ours: never use** |
| GPU 1 | RTX PRO 6000 Blackwell Max-Q (97.9 GB), PCI 83:00.0, UUID `GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4`, NUMA-local to cores 6–11. **The only GPU we use** |
| Environment | micromamba env `dl-stable` (torch 2.10.0+cu128) |

**Limits: GPU 1 only, and at most 8 cores at any time**, so that at least 4 stay free for
others.

- **CPU:** all of our jobs are pinned to **cores 4–11**. That set is GPU 1's NUMA node
  (6–11) plus 4–5, and it leaves 0–3 free on GPU 0's socket. Affinity is inherited, so
  DataLoader workers and process pools stay inside the same 8 cores. *Everything running
  at once shares them*, so size worker pools against the total. Example: fetcher (2) +
  audit (6) = 8.
- **GPU:** `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. The two cards are
  identical, so a wrong pick would not show in any log. Only the PCI order or the UUID
  tells them apart.

**How it is enforced** (added 2026-09-23):
- `src/node_limits.py` runs on `import src`, i.e. in every training, eval and data script.
  - *CPU:* it pins the process to cores 4–11 and caps torch threads at 8.
  - *GPU:* if nothing is set, it sets GPU 1 in PCI order. It raises `NodeLimitError` for any
    other `CUDA_VISIBLE_DEVICES` (including `0`, `0,1`, or GPU 0's UUID). If CUDA is already
    initialised, it checks the visible device's UUID.
  - It is a no-op on other hosts. Tests: `tests/test_node_limits.py`.
- `scripts/hpc/env.sh` exports GPU 1 in PCI order (it previously exported
  `CUDA_VISIBLE_DEVICES=0`), and pins the launching shell to cores 4–11.
- The launchers with a `GPU` override (`run_full_eval`, `eval_losses`, `launch_eval_losses`,
  `sweep_reward`, `check_sampler`) refuse anything but `GPU=1`. `run_diagnosis.sh` used
  GPU 0 and now uses 1.
- `launch_data_quality.sh` defaults to 6 workers and refuses more than 8.
- **Verified live:** `import src` gives affinity 4–11, torch at 8 threads, one visible device
  with GPU 1's UUID (PCI bus 0x83), and DataLoader workers on 4–11. `CUDA_VISIBLE_DEVICES=0`
  raises.

**Per-job budgets that fit.** A job's DataLoader and pool sizes are set in `config.yaml`
(`NUM_WORKERS: 4`, `MAX_WORKERS: 4`) and in script flags.
- 1 training run: main + 4 workers = 5.
- The fetcher: 2 (currently on cores 4–5).
- A data-quality audit: 6.
- Do not start a second CPU-heavy job alongside a full audit plus the fetcher.

---

## 1. Checkpoints

### Deterministic backbones — `runs/sr_analytical/`

| Run | Loss | Weight | Epochs (stop / best) | Status |
|---|---|---|---|---|
| `UNet_Ana_20260623_144958` | MSE only (vanilla) | 0 | 69 / 64, early stop | **reference backbone** for all FM runs |
| `UNet_Ana_20260731_100128` | MSE + Minkowski | 1e-4 | 47 / 46, interrupted | **Study-1 winner** |
| `UNet_Ana_20260630_111918` | MSE + Minkowski | 1e-4 | 100 | full-length run, not evaluated |
| `UNet_Ana_20260722_081033` | MSE only | 0 | 34 | short vanilla, not evaluated; matches the v2 competing runs on every loss value |
| `UNet_Ana_20260729_104637` | MSE + Minkowski | 1e-3 | | **discard**: reward hacking (DECISIONS §2) |
| `UNet_Ana_20260812_085343` | MSE + spectral | 1e-4 shared | | v1, inert |
| `UNet_Ana_20260817_113622` | MSE + ssim | 1e-4 shared | | v1, inert |
| `UNet_Ana_20260829_033325` | MSE + spectral | 5e-4 | 25 / 25 | v2, inert (under-weighted ~70×) |
| `UNet_Ana_20260827_151319` | MSE + opticalflow | 4e-5 | 25 / 25 | v2, inert (under-weighted ~140×) |
| `UNet_Ana_20260826_041128` | MSE + wetarea | 2e-2 | 25 / 25 | v2, active but degenerate |
| `UNet_Ana_20260824_155753` | MSE + ssim | 1.6e-4 | 25 / 25 | v2, inert, and the loss is buggy (§3) |

### Flow matching — `runs/sr_flow_matching/`

| Run | Backbone | σ_r | Status |
|---|---|---|---|
| `flow_matching_20260716_071518` | vanilla | 0.01898130588233471 | **clean reference model** |
| `Extremes_20260807_111521` | Minkowski (0731) | 0.019811755046248436 | Study-2 cell (evaluated from `fm_best.pth`) |

### Reward fine-tuning — `runs/sr_fm_minkowski/`

| Run | Mode | Row in tab:iid | Status |
|---|---|---|---|
| `Extremes_20260804_170256` | distributional | FM + Minkowski reward | first coupling. Called "null-to-negative" on 09-13; **best FM row on FSS and RL bias** after the 09-16 re-eval |
| `Energy_20260811_171310` | energy + retention (w=1.0) | FM + Minkowski energy | best exceedance ratio and RAPSD |

Both are evaluated from `fm_mink_latest.pth` (DECISIONS §10).

### Retired and deprioritised

- **DDPM, dropped** (2026-09-23): replaced by flow matching. `runs/sr_ddpm/` (3 runs from
  April plus a debugging run) and the code (`src/models/{ddpm,diffusion}.py`,
  `src/trainers/ddpm.py`, `scripts/train/train_ddpm.py`, `scripts/hpc/launch_ddpm.sh`) are
  kept but no longer maintained, compared or reported. See DECISIONS §14.
- **Emulator-based Minkowski loss, lowest priority**: `runs/emulator/`, `runs/sr_emulator/`
  (`UNet_Emu_*`, May). Worth resuming eventually as the learned alternative to the
  analytical loss. Not in any current table.

---

## 2. tab:iid — in-distribution extremes (POT u = 31), re-evaluated 2026-09-16

Source: `eval_results/extremes/*/extremes_summary.yaml` (`scripts/hpc/run_full_eval.sh`).
Outputs from before the 09-16 re-evaluation are in `eval_results_pre_patch_20260916_102140/`.

**The two blocks are not comparable with each other.** The observed tail differs between
them: `gpd_xi_obs` is 0.176 on the full test set against 0.083 on the 4,096-patch FM subset,
with 316,652 against 5,070 observed exceedances. Compare within a block only, until the FM
models are evaluated on the full set or the deterministic ones on the same subset.

**Deterministic, full test set (n = 285,383)**

| Model | RMSE_ext | FSS@89 | RAPSD | Mink | Exc. ratio | RL bias | Anisot. | Peak ratio ext. |
|---|---|---|---|---|---|---|---|---|
| Bicubic | 3.77 | 0.000 | 4.00 | 2.85 | 0.044 | 0.782 | 30.94 | 0.052 |
| Backbone (MSE) | 3.59 | 0.040 | 3.10 | 2.12 | 0.118 | 0.643 | 1.87 | 0.201 |
| + spectral (v2) | 3.68 | 0.034 | 3.89 | 2.23 | 0.057 | 0.719 | 2.24 | 0.121 |
| + ssim (v2) | 3.68 | 0.039 | 3.83 | 2.23 | 0.059 | 0.701 | 2.14 | 0.123 |
| + wetarea (v2) | 3.72 | 0.032 | 4.59 | 2.42 | 0.036 | 0.748 | 2.57 | 0.099 |
| + opticalflow (v2) | 3.67 | 0.036 | 4.03 | 2.25 | 0.057 | 0.713 | 2.36 | 0.121 |
| + Minkowski (1e-4) | 3.70 | **0.159** | **1.31** | 0.74\* | **0.244** | **0.445** | 0.73 | **0.659** |

**Generative, 4,096-patch subset, M = 16, heun-16**

| Model | RMSE_ext | FSS@89 | RAPSD | Mink | Exc. ratio | RL bias | Anisot. | Peak ratio ext. | CRPS_ext |
|---|---|---|---|---|---|---|---|---|---|
| FM (clean) | 3.00 | 0.315 | 0.78 | 0.84 | 0.475 | 0.165 | 1.35 | 0.567 | 0.305 |
| FM + Minkowski reward | 3.00 | **0.321** | 0.65 | 0.83\* | 0.534 | **0.114** | 1.32 | 0.584 | 0.305 |
| FM + Minkowski energy | 3.01 | 0.269 | **0.55** | 0.83\* | **0.538** | 0.126 | 1.24 | 0.603 | **0.302** |
| FM on Minkowski backbone | 3.03 | 0.273 | 0.74 | 0.82\*\* | 0.466 | 0.141 | 1.27 | **0.731** | 0.308 |

FSS@89 uses window 5. \* This is the training objective, so it is not independent evidence.
\*\* The conditional mean was trained on the Minkowski loss, so this is partly circular.

GPD ξ error is a **secondary diagnostic only** (DECISIONS §10), and it is comparable only
within a block. Deterministic: bicubic 0.131 · MSE 0.029 · spectral 0.013 · ssim 0.009 ·
wetarea 0.007 · opticalflow 0.053 · Minkowski 0.099. FM: clean 0.063 · reward 0.088 ·
energy 0.067 · on Mink backbone 0.110. The energy row's "best ξ in the table" (0.043) from
09-13 no longer holds.

**What changed against the 09-13 table.** The bicubic and FM + reward rows now have their
u=31 tail columns. The FM rows moved: FM clean FSS 0.289 → 0.315, FM on Minkowski backbone
FSS 0.148 → 0.273 and RL bias 0.236 → 0.141, FM + reward FSS 0.268 → 0.321. The cause of
each shift has not been attributed yet. The 09-16 commit added the σ_r guard, and a σ_r
mismatch is the known way to break the Minkowski-backbone cell (DECISIONS §12).

**Consequence for DECISIONS §3.** Within the FM block, both Minkowski couplings now
**improve** the exceedance ratio (0.475 → 0.534 / 0.538) and the RL bias
(0.165 → 0.114 / 0.126). The reward coupling does this with no FSS cost. The "does not help
flow matching" conclusion was drawn from the 09-13 numbers, so it is **reopened** rather
than settled. It still rests on one seed and one 4,096-patch subset.

## 3. tab:prelim — baseline metrics (full test set), re-evaluated 2026-09-16

| Model | MAE | ΔMAE vs vanilla | MAE_ext | Mink | Isoperim % | RAPSD | SAL S |
|---|---|---|---|---|---|---|---|
| vanilla | 0.0499 | — | 0.4030 | 2.12 | 4.57 | 3.05 | 1.27 |
| spectral v2 | 0.0502 | +0.6% | 0.4103 | 2.23 | 4.41 | 3.87 | 1.15 |
| ssim v2 | 0.0503 | +0.7% | 0.4104 | 2.23 | 4.45 | 3.81 | 1.17 |
| wetarea v2 | 0.0503 | +0.7% | 0.4101 | 2.42 | 4.17 | 4.59 | 1.27 |
| opticalflow v2 | 0.0501 | +0.3% | 0.4086 | 2.25 | 4.39 | 4.01 | 1.12 |
| Minkowski (1e-4) | 0.0519 | +3.8% | 0.4195 | 0.74 | 3.53 | 1.30 | 0.34 |
| FM (clean) | 0.056† | — | 0.423† | 0.83‡ | 4.29‡ | 0.79‡ | 0.50‡ |

† ensemble mean, ‡ single arbitrary member. The FM row is on a 25,600-patch subset and dates
from **before** the 09-16 re-evaluation (there is no `eval_results/backbone/fm*`), so do not
compare it with the rows above.

**Activity check:** an auxiliary loss that costs < ~1% MAE against vanilla has not entered
the objective, so its structural columns say nothing. All four competing losses fail it.

**Activity audit (2026-09-23, measured, not inferred from MAE).** Gradient norms at the
vanilla checkpoint, fp32, 6 train batches of 128, ‖∇MSE‖ = 2.95e-4:

| Loss | λ used | parity λ = ‖∇MSE‖/‖∇aux‖ | λ‖∇aux‖ / ‖∇MSE‖ | cos(∇MSE, ∇aux) | own loss, trained vs vanilla |
|---|---|---|---|---|---|
| Minkowski | 1e-4 | 2.2e-5 | **4.6** | −0.02 | 0.48 vs 1.41 (−66%) |
| wetarea | 2e-2 | 1.6e-3 | **12.8** | +0.16 | −14% |
| ssim | 1.6e-4 | 1.5e-3 | 0.11 | −0.10 | 0.4834 vs 0.4831 (no change) |
| spectral | 5e-4 | 3.6e-2 | 0.014 | +0.16 | 4.12e-3 vs 4.01e-3 (no change) |
| opticalflow | 4e-5 | 5.6e-3 | 0.007 | +0.13 | 8.98e-3 vs 9.25e-3 (no change) |

- **Minkowski is genuinely active.** Its weighted gradient is 4.6× the MSE gradient at the
  start and 1.7× at its own checkpoint. It is nearly orthogonal to MSE, and its loss falls
  66%. The working weight is about 5× gradient parity.
- **Spectral, SSIM and optical flow never entered the objective.** Their weighted gradients
  are 1–11% of MSE's, and the trained models score no better on their own loss than
  vanilla does.
- **Wet area is active and degenerate**, as DECISIONS §8 describes.
- **SSIM is buggy** (`src/losses/competing.py:101-110`). The per-sample `data_range`
  collapses on dry patches, and eps=1e-6 swamps `c1·c2`. As a result `SSIM(target, target)`
  scores 0.465 and an all-zero prediction 0.499: about 93% of the loss range is an
  unreachable floor.
- **The runs are not matched.** The competing runs had 25 epochs (still improving at 25);
  Minkowski had 47 and vanilla 69. A 34-epoch vanilla run is indistinguishable from the
  spectral, SSIM and optical-flow models, so their worse tails are under-training plus an
  inert term. They are not evidence that those losses are harmful.
- **Flow matching:** the competing losses are not implemented at all
  (`reward_finetune.py:263` hard-codes the Minkowski loss).

---

## 4. Tooling

| Script | Purpose |
|---|---|
| `scripts/hpc/launch_unet_analytical.sh` | train a backbone: `<params> <w_geom> <data_pct> [<config>]` |
| `scripts/evaluate/eval_extremes.py` | tail protocol → tab:iid row |
| `scripts/evaluate/eval_backbone.py` | baseline metrics → tab:prelim row |
| `scripts/evaluate/eval_fm.py` | FM baseline metrics |
| `scripts/hpc/eval_losses.sh` | batch-evaluate all backbones, with activity check + POT guard |
| `scripts/hpc/run_full_eval.sh` | the full 09-16 re-evaluation (every row of tab:iid / tab:prelim) |
| `scripts/evaluate/make_plots.py` | all comparison figures from eval outputs |
| `scripts/evaluate/diagnose_coupling.py` | is the reward wired correctly? (minutes, one batch) |
| `scripts/hpc/sweep_reward.sh` | one-factor-at-a-time reward sweep |
| `scripts/hpc/check_sampler.sh` | sampler-fidelity check |
| `src/sweep_util.py` | `patch` a config / `collect` a summary into CSV |
| `scripts/data/fetch_opera_archive.py` | fetch OPERA RATE + QIND from the open archive onto the patch grid |
| `scripts/hpc/launch_data_quality.sh` | the whole data-quality audit below, under `setsid` |
| `scripts/data_quality/` | raw-archive audit: clutter climatology, per-tile artefact features, split leakage, summary with candidate rules, image gallery + label sheet (see its README) |

Figures: perception–distortion plane, RAPSD with ratio panel, GPD return levels + exceedance
counts, tail/safety dot chart, SAL plane, **γ curves**, **γ residual by threshold**,
isoperimetric scatter.

---

## 5. TODO

### Highest value
- [ ] **Data quality first** (roadmap M1). Every tail number below depends on it.
- [ ] **Tail-sample selection.** Score a large patch pool by patch max, and draw the
      *structural* loss batches from the top few %. This targets the pixel-mass bottleneck
      directly (DECISIONS §5, §13). Keep the MSE / velocity loss on uniform batches.
      Draw only from the *screened* pool, because ranking on max draws artefacts first.
- [ ] **Gradient-norm λ rule** for all auxiliary weights (DECISIONS §9). The parity values
      are now measured (§3 activity audit).
- [ ] **Competing losses: a fair trial** (roadmap M4). Fix SSIM, calibrate λ, match epochs,
      and use 3 seeds.

### Paper
- [ ] Add the W₁ identity to the theory section (DECISIONS §13).
- [ ] Report exceedance ratio + RL bias as tail columns. Demote ξ to the text, with the
      confound explained.
- [ ] State the POT-level choice as estimability, not physics.
- [ ] Write up the w=1e-3 hacking episode as a positive result on necessary-not-sufficient.
- [ ] Theory framework (roadmap M5, `RESEARCH_NOTES.md`).

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


- [ ] **The splits leak: random at patch level** (measured 2026-09-23,
      `scripts/data_quality/audit_splits.py`). `split_metadata.py` shuffles *patches*, so all
      three splits span 2023-08-01 → 2024-10-30 over the same 79 tile locations. For val and
      test alike, including the tail:

      | test subset | n | train patch at same timestamp | same tile within ±1 h | adjacent tile, same timestamp |
      |---|---|---|---|---|
      | all | 285,383 | 100.0% | 100.0% | 98.2% |
      | patch max ≥ 31 | 25,591 | 100.0% | 100.0% | 99.4% |
      | patch max ≥ 89 | 6,349 | 100.0% | 100.0% | 99.3% |

      So the test set contains the neighbouring 15-minute frames of storms that are in train.
      Every current score is an interpolation score. Any o.o.d. or "unseen extreme" claim
      needs a split that is blocked in time by event, with a buffer (DECISIONS §15).
- [ ] **The `quality_map` array in the patch store is empty**: it is allocated by
      `preprocess_data.py` for every split, and it is all zeros. Fill it on the rebuild
      (QIND or the screening mask), or drop it.
- [ ] **QIND in the archive covers only part of the rate field.** On 2012-09-04 it is NaN on
      ~93% of pixels against ~57% for the rate. Measure the coverage per year before
      building a rule on it; the audit summary does this per tile (`q_cov`).
- [ ] **Audit suite in place** (`scripts/data_quality/`, 2026-09-23). A 5-day smoke test
      already shows the any-flag rate rising with raw max (14% at 1–10 mm/h → 69% at 89–150
      → 100% above 150), 2012 much dirtier than 2024, and static-clutter pixels that exceed
      31 mm/h in 35% of time steps. Next: run it on the full archive, then label ~300 tail
      tiles and calibrate the rules.

### Data hygiene
- [x] Rerun bicubic and `FM + Minkowski reward` at u=31 (done in the 09-16 re-eval).
- [ ] Attribute the 09-13 → 09-16 shifts in the FM rows (σ_r guard? eval patch?) before
      quoting either version.
- [ ] Put FM and deterministic rows on the same patches (the full set for FM, or the 4,096
      subset for the backbones) so that tab:iid is one comparable table.
- [ ] Delete duplicate eval dirs (`backbone_mse` = `backbone_vanilla`,
      `backbone_mink_1e-4` = `backbone_minkowski`, `*_v1` duplicates).
- [ ] Re-run evals so the new `gamma_target` / `pot_threshold` arrays exist for the γ plots.
- [ ] Eval `Energy_.../fm_mink_best.pth` to quantify the selection bug vs `latest`.
- [ ] Try `REWARD_FM_RETENTION_WEIGHT` 3 or 10, since FSS still fell at 1.0.
- [ ] Evaluate `UNet_Ana_20260630_111918` (Minkowski 1e-4, 100 epochs) against `0731`
      (47 epochs, interrupted).
- [ ] Multiple seeds: every result above is a single run with no interval.

### Deferred
- [ ] b0 (persistent homology) confirmatory run at the winning weight
- [ ] Compute proposal: the scale argument rests on DECISIONS §5–6
- [ ] Emulator-based Minkowski loss: lowest priority, see §1

---

## 6. Roadmap — macro steps

These are **not in order**. The dependencies are noted. The theory, the extreme-definition
question and the regime design are developed in `RESEARCH_NOTES.md`.

### M1. Data analysis on the full dataset
Understand what the enlarged archive (2012 →) contains and what it can support, **before**
any retraining. Details to come. The tooling for the quality part exists
(`scripts/data_quality/`).
- Artefact screening policy: pixel masking vs tile rejection, the declutter rule, the
  static-clutter mask, the role of QIND. Calibrate against a labelled set.
- Rebuild the patch set with a quality column stored per patch, so a screen can change
  without a rebuild, and with **event-blocked splits**.
- The descriptive analysis M2 and M5 need: tail statistics, how γ curves of extreme events
  relate to those of ordinary ones, and how events distribute by season, region and regime.
- *Blocks:* M2, M3, and all tail-driven results.

### M2. New training with a cut-off distribution: o.o.d. extremes evaluation
Train with the upper tail removed, and evaluate on the held-out extremes (tab:ood). This is
the direct test of the central hypothesis (M5).
- Decide what "extreme" means first (RESEARCH_NOTES §2). The choice fixes how the cut is
  made: by intensity, by event type, or relative to local climate.
- The cut has to be made by **event** and **in time**, on the screened pool. A patch-level
  cut would leak the same storms through neighbouring tiles and frames (§5).
- *Needs:* M1.

### M3. Regime-aware Minkowski training
Classify each sample into a precipitation regime from the low-res input, the prediction and
its predicted Minkowski functionals, and make the structural loss regime-aware.
- Start from what ECMWF does. ecPoint's gridbox weather types are the closest analogue
  (RESEARCH_NOTES §4).
- Needs a regime definition first. That is open work.
- *Needs:* M1 (regime statistics on clean data).

### M4. Fix the four competing losses, for the deterministic and FM models
Make the Study-1 comparison fair: every loss gets a demonstrably active weight under the same
budget, and the losses are implemented as rewards for FM. The protocol is in
RESEARCH_NOTES §5.
- Fix SSIM. Remove the silent `teacher=target` fallback in optical flow, or log it.
- Choose λ per loss by the gradient-norm rule plus a small bracket, with one fixed epoch
  budget, the last checkpoint evaluated (no λ-dependent "best" selection), and 3 seeds
  (DECISIONS §16).
- Add spectral / SSIM / wet-area / optical-flow reward paths to `reward_finetune.py`.
- *Independent of M1 for the fixes and the calibration. Final numbers should wait for the
  rebuilt data.*

### M5. Theory framework
Why training on Minkowski functionals should make a super-resolution model more robust to
unseen extremes: geometry as the transferable handle. This is the argument the paper rests
on. It becomes testable through M1 (is extreme geometry similar to ordinary geometry?) and
M2 (does the model extrapolate?). See RESEARCH_NOTES §1–3.
