# Experiment tracker

Last updated: 2026-10-03

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

**Activity audit (2026-09-23)**, from `tools/gradient_audit.py`: 512 × 128 patches, fp32,
GPU 1, noise-corrected full-gradient norms, jackknife errors. Raw output is in
`eval_results/gradient_audit/2026-09-23/`; the math is in `notes/gradient_audit.pdf` (the
appendix has these numbers). It replaces a first 6-batch estimate (`tools/gradient_audit_v0/`),
whose parity values were dominated by minibatch noise: at the converged vanilla checkpoint the
MSE gradient is 96% noise, so parity is ill-posed there.

| Loss | λ used | r at init (λ/λ*) | r at own ckpt | cos at own ckpt | own loss vs vanilla (95% CI) | verdict |
|---|---|---|---|---|---|---|
| Minkowski | 1e-4 | 0.095 | **1.22** | **−0.986** | **−64.8%** [−65.9, −63.8] | active, at equilibrium |
| wetarea | 2e-2 | 1.0 | **0.74** | **−0.967** | **−13.9%** [−15.8, −12.1] | active (drizzle band only) |
| ssim | 1.6e-4 | 1.7e-4 | 0.157 | +0.47 | +0.1% [+0.0, +0.2] | bystander |
| spectral | 5e-4 | 1.7e-3 | 0.021 | +0.92 | +2.5% [+0.9, +4.4] | bystander |
| opticalflow | 4e-5 | 3.7e-5 | 0.0066 | −0.47 | −2.6% [−6.4, +1.2] (n.s.) | inert |

- **How to read it.** A term that trained actively ends at the equilibrium of the combined
  objective: share → 1 and cos → −1 (notes §4). Minkowski and wet area sit on it. The other
  three are far off, with their gradients still aligned with the MSE.
- **The own-loss test agrees with the equilibrium test.** Spectral and SSIM are slightly
  *worse* than vanilla on their own loss. The 34-epoch vanilla model is also worse than the
  64-epoch one, so this is under-training at 25 epochs.
- **At initialisation every aux gradient is aligned with the MSE** (|cos| 0.77–0.96), so
  all terms start as bystanders. Minkowski too starts 10× below parity, and it takes over
  only as ‖∇MSE‖ collapses during training.
- **Weights in parity units at init:** Minkowski ~10⁻¹, spectral ~10⁻³, SSIM ~10⁻⁴, optical
  flow ~4·10⁻⁵, wet area 1. The first audit's "70× / 140× too little" factors came from the
  noisy vanilla checkpoint and are superseded.
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
- [ ] **The DEM channel of every patch is from the wrong place (bug, found 2026-09-28).** The
      DEM GeoTIFF is north-first (row 0 = north) and the precipitation stores are
      south-first (row 0 = south; checked against radar-site geography, IoU 0.878 vs 0.554
      flipped). `src/data/preprocessing.py` slices both with the same `(y_start, x_start)`, so
      each patch gets the DEM of the N-S-mirrored place: 20/20 test patches match the
      mirrored slice and 0/20 the true one, e.g. correlation −0.32 on test patch 0 (southern
      Finland). **Every trained model has had a geographically wrong DEM input.** Relative
      comparisons stand (all rows share it), but nothing about orography can be claimed, and
      absolute numbers will move once it is fixed. `dem_stats.json` was computed on the
      mirrored patches too.
      **Code fixed 2026-09-28:** `src/data/preprocessing.py::process_batch` now takes the DEM
      from `src.data.geo.load_dem_on_radar_grid`, loaded once per worker. Checked on 5 test
      patches: DEM = true ground in 5/5, precipitation bit-identical to the old store.
      Regression test: `tests/test_geo.py`.
      **Existing store regenerated 2026-09-28**, without a rebuild. Only the DEM was wrong:
      the fixed preprocessing yields bit-identical precipitation, and the gamma targets depend
      on precipitation only.
      - *DEM lookup instead of storage.* The store held just 79 distinct DEM tiles repeated
        2.85M times (89 GB). `DeterministicSRDataset` and `DiffusionSRDataset` now look the
        DEM up under each patch's `(y, x)` from the oriented full DEM (`src.data.geo.
        dem_patch`, new kwarg `dem_path`). The per-patch `<split>/dem` arrays were deleted,
        freeing ~89 GB. An in-place rewrite would have *grown* the store by +34.7 GB, since
        the true terrain compresses worse. Preprocessing no longer writes `dem`.
      - *Checks before deleting:*
        - precipitation, target and gamma identical to the old dataset code in 40/40 patches
          per class and split;
        - DEM channel = true ground in every sampled patch, RAM mode included.
      - *`dem_stats.json` recomputed exactly* from tile counts: mean 286.16, std 393.53
        (was 188.31 / 351.65, backed up as `dem_stats_mirrored_pre20260928.json`). The same
        method on the mirrored DEM reproduces the old file to 6 decimals.
      - *Consequences:*
        - every existing checkpoint was trained with the mirrored DEM and the old stats, so
          retrain before comparing anything;
        - `residual_stats*.json` (σ_r) are backbone-specific and must be recomputed after
          retraining;
        - `Mink-DDPM` (inactive since April) reads the deleted `dem` arrays and now fails
          with a KeyError. This was accepted.
      - *Not changed:* the declutter zeroing above 150 mm/h, the patch-level splits (they
        still leak, §5 above), the stale `patches/dem/dem_patch_*.npy` cache (unused), and
        `cosine_warmup_weight` vs its failing test (pre-existing).
- [ ] **Data quality first** (roadmap M1). Every tail number below depends on it. Policy:
      reject clear errors only, keep imperfections, no labelled set (DECISIONS §17).
      Background and agency practice: RESEARCH_NOTES §7.
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
- [ ] **Dataset limitation: coverage rule and coastal bias** (unless fixed first). Only fully
      covered patches are used, which under-represents coastal and network-edge extremes
      (Mediterranean) and drops the Emilia-Romagna events; HyMeX IOP16 is not in the
      archive. Draft text and the three possible improvements (event windows, blind-spot
      repair, masked partial patches) in `notes/events.md` §6.

### Data — quality and size
- [ ] **Re-derive the POT threshold on v2 instead of inheriting u = 31.** 31 is a point of the
      loss grid used "for estimability" (DECISIONS §10), never tested:
      - ξ_obs flips sign between 31 and 53, and the fitted data were capped and artefact-laden;
      - on v2 (uncapped, cleaned, ODYSSEY only): mean-residual-life plot and ξ / modified-σ
        stability over u ≈ 10–150 mm/h, with bootstrap intervals;
      - runs declustering of exceedances in space and time, pixel-level vs event-maxima fits;
      - per region and season (E3), since one fixed level may not suit all of Europe;
      - then fix one level for every table, or justify keeping 31.

      Questions for Daniele Nerini and Lionel Moret are in RESEARCH_NOTES §7.4. Every tail
      column changes if u changes, so settle this before the v2 re-evaluation.
- [ ] **ODYSSEY vs NIMBUS gap analysis** (`scripts/data_quality/era_gap.py`; outputs in
      `OPERA/quality_v2/era_gap/`).
      - *Why:* OPERA switched production chain on **2024-07-05**. The grid, 2 km resolution,
        projection and 15-min cadence are unchanged, so nothing is resampled. What changed is
        how a frame is made:
        - ODYSSEY: a quality-weighted composite of the scans in a 15-min window (e.g.
          11:50–12:05);
        - NIMBUS: the lowest-elevation PPI at the nominal time;
        - plus a somewhat different radar set.
      - *Paired* (the 101 days 2024-07-05..10-30 that exist in both products, originals kept
        in `raw/OPERA_orig_postswitch/`): frequencies, quantiles, co-located correlation,
        Minkowski curves, spectra, +15 min persistence. First day (2024-08-01):
        - NIMBUS has 40% fewer pixels ≥ 0.1 mm/h but equal ≥ 10 mm/h;
        - its p90 is 3.25 vs 2.45 mm/h;
        - its +15 min correlation is 0.19 vs 0.35, the time-support difference.
      - *Paired result, all 101 days* (2,407 frames, 97,682 wet tiles; ODYSSEY copies capped at
        150 mm/h, so nothing above 89 mm/h is compared). NIMBUS / ODYSSEY:
        - rain area ≥ 0.1 mm/h 0.71, ≥ 1 mm/h 0.99;
        - exceedance ≥ 10 / 31 / 89 mm/h 1.36 / 1.44 / 2.4;
        - wet-pixel p50 / p90 / p99 ≈ 1.4;
        - perimeter at 10–31 mm/h ≈ 1.8, Euler characteristic ≈ 2.4×;
        - spectral power at 128 / 32 / 8 / 4 km 1.25 / 1.42 / 3.1 / 6.2;
        - +15 min correlation 0.41 → 0.26; co-located log-correlation 0.80.

        So NIMBUS fields are more intense and much rougher and more fragmented at small scales:
        a single instantaneous low-elevation scan against a blended 15-min composite.
        **For NIMBUS evaluation this is not a neutral test set.** An ODYSSEY-trained model will
        under-produce NIMBUS small-scale variance by construction, so spectral, perimeter/χ and
        tail metrics carry a product penalty unrelated to extremeness. Report NIMBUS scores
        against a reference that shares the product (e.g. a NIMBUS-trained or NIMBUS-fine-tuned
        model, or the NIMBUS persistence/bicubic baselines), not in absolute terms.
      - *Month-matched:* NIMBUS months as z-scores against the ODYSSEY interannual spread of
        the same calendar month. This runs inside `rebuild_v2.sh`, after the rescan.
      - *Decision (DECISIONS §18):* train/val/test are ODYSSEY only; NIMBUS is a separate
        product-shift test split. Evaluating on NIMBUS later is planned. Mind that temporal
        statistics such as persistence and advection differ between eras by construction.
- [ ] **v2 dataset family: building (launched 2026-09-28).** Pipeline in `scripts/dataset_v2/`,
      tests `tests/test_cleaning.py`. All of 2012 + 2014 → 2025-10 on disk (4,345 days; 2013
      to be re-downloaded and added).
      1. `run_scan.sh`: per-year clutter climatology over every day, then `scan_tiles.py`.
         Every frame is cleaned by `src/data/cleaning.py` (drizzle floor; static-clutter and
         spike *repair*; no zeroing above 150 mm/h), and every fully covered tile is
         described.
      2. `run_build.sh` (waits for 1):
         - `make_splits.py`:
           - leak-free splits by whole ISO weeks, drawn per calendar month, with 1-day
             buffers, so train is ≥ 2 days from val/test;
           - the 26 catalogued events (`configs/prominent_events.yaml`, sourced from ECMWF,
             MeteoSwiss, DWD, ESSL, AEMET and Météo-France studies) are forced into test;
           - rejects tiles that are `unphysical` (> 500 mm/h after repair) or a `ray` (an
             RLAN ray pointing at a radar);
           - stratified sampling to 3M patches, with per-row inclusion weights.
         - `build_store.py`: store, verify, aux.
         - `config_v2.yaml`.
         - gamma targets.
         - `check_datasets.py`.
      - *Outputs:* `OPERA/v2/` (metadata) and `OPERA/patches_v2/` (store).
        - `full_{train,val,test}`;
        - `light_*`: 25%, same strata shares;
        - `extremes_*`: cleaned max ≥ 31 mm/h;
        - `events_test`: every tile of every event, never subsampled.
        Subset files carry a 5th column (store row), which the datasets now accept.
      - *Checked on a 12-day sample before launch:*
        - rejection concentrated in raw > 500 mm/h (80% rejected, the rest repaired);
        - ray rule < 1% in every intensity bin;
        - stored tile max = metadata max (after fixing a 4-significant-digit CSV rounding
          bug);
        - subsets return exactly their store rows;
        - DEM correct;
        - leak checks pass.
      - *Consequence:* the scaler moves from `log1p` 5.02 to ~6.19, because rates are no
        longer zeroed above 150 mm/h.
      - *After it finishes:* review `OPERA/v2/report.md` (rejection by intensity, per-event
        coverage), then verify and delete the old store. Legacy `.npz` dumps in
        `patches/precip/{train,validation,test}/` hold 480 GB and are unused by this repo.
      - *Known limits:*
        - tiles need 100% radar coverage, so a single nodata pixel excludes a tile;
          Emilia-Romagna may get no event tiles because of this;
        - radar rates underestimate intense rain (Valencia: 28.6 mm/h radar pixel vs
          184.6 mm/h gauge-hour at Turís);
        - ring artefacts are not detected yet.
      - *Built 2026-10-01/02* (`logs/dataset_v2_build3.log`, `logs/finish_nimbus_gamma.log`):
        train 1,977,908 / val 251,667 / test 282,684 / nimbus 487,743 patches; 3,168 / 404 /
        463 / 801 days; min gap 2 days. The first gamma run skipped the `nimbus` group
        (`compute_gamma_targets.py` listed only train/validation/test; fixed), so the checks
        were rerun after it: **ALL CHECKS PASSED** (2026-10-02 13:30; every subset of all four splits, DataLoader, no shared days, min gap 2 d). The legacy `.npz` dumps were deleted on 2026-10-01; the old store `patches/precip/preprocessed_dataset.zarr` can go now (approved after verification).
- [ ] **Complete the screen and validate it against independent data — gate before any v2
      training** (started 2026-10-02; RESEARCH_NOTES §7.4b steps 4, 6, 8, 9, 10). Every new
      check is an *audit flag* first; nothing is removed until the gauges say the rule
      separates artefacts from rain. Then one rebuild, together with the event-set fix below.
      - *Temporal support* (`cleaning.temporal_support`): a cell >= 10 mm/h with no echo
        >= 1 mm/h within 30 km at t +- 15 min. Undecidable when a neighbour frame is missing
        or not covered. Threshold choices to revisit with the gauge result: NIMBUS is less
        persistent frame to frame than ODYSSEY (+15 min corr 0.26 vs 0.41).
      - *Range rings*: not thin arcs in single frames (`cleaning.ring_flag` found none in
        7,005 tiles), but circles in the per-year >= 31 mm/h frequency
        (`scripts/data_quality/ring_climatology.py` -> `quality_v2/ring_mask.npz`,
        `ring_list.csv`). Found: Stevns and Sindal (DK) at ~234 km, every year 2013-2017;
        Ikaalinen (178 km) and Vimpeli (49 km), FI, 2012; none after 2017 (the 2017-09-29
        compositing change). The radar database's years are incomplete (Stevns is listed
        from 2017), so rings are searched about every site, current and archive.
      - *Bad radars* (`scripts/data_quality/radar_quality_v2.py`, whole archive, signals
        relative to the 5 nearest radars): from the climatology alone, 8 consistently bad:
        Bollène (FR, clutter, 15/15 years), Teolo (IT, > 500 mm/h), Fljotsdalsheidi (IS),
        Monte Lauro (IT), Andravida (GR), Debeljak (HR), Hudiksvall and Vara (SE).
        **With the flag signals (2026-10-03, `logs/radar_quality_v2_20261003.log`,
        `quality_v2/radars/`): 14 consistently bad.** The 8 above, plus Ängelholm (SE,
        unsupported maxima, 15 years), Stevns and Sindal (DK, unsupported, 10), Karlskrona
        and Hemse (SE, unsupported), and Berlin (DE, ring 2013-15).
        - The first run with flags (2026-10-03 02:43, in `logs/validation.log`) listed 204
          of 246 radars, which is wrong. Three bugs, fixed in the script; the old outputs are
          in `quality_v2/radars_pre20261003_buggy/`:
          - `ring_share` is zero in 73% of radar-years, so its 95th percentile was 0 and
            `>=` flagged 75.5% of them. An outlier must now also be > 0; ring_share
            flags 2.1%.
          - 9 sites have both a current and an archive entry in the radar database with
            overlapping years (Flechtdorf, Essen, Rostock, Neuhaus, Eisberg, Ängelholm,
            Leksand, Maly Javornik, Kojsovska hola). Both counted as active: they split
            the site's pixels, appeared twice per year (Flechtdorf "28 years"), and were
            each other's nearest neighbour in the relative scores. Radars are now keyed by
            ODIM code, with duplicates merged.
          - 16 sites with no ODIM code (e.g. Monte Lauro, Gelemenovo) were dropped by the
            `groupby`. Their flag stats were also lost, because the scan writes `""` both for
            them and for "no radar within 250 km". They are now keyed by location, and flag
            tiles are re-attributed from the argmax with the same ownership map.
        - *Attribution caveat:* the database lists Stevns and Sindal only from 2017, so their
          2013-16 rings (`ring_list.csv`) fall on the nearest radar active then. Ängelholm's
          ring_share is non-zero only 2013-17, so it is probably carrying Stevns's ring.
          Ringed sites need archive years before a per-radar rule can use ring_share.
      - *QIND*: kept as a covariate. Its scale changes with the product: tail-tile mean QIND
        ~0.2-0.28 under ODYSSEY vs 0.80-0.87 under NIMBUS, so any weight must be normalised
        per product. The gauge validation measures whether it separates real from false
        tail pixels (AUC per era).
      - *Flag scan* (`scripts/dataset_v2/scan_flags.py`, `run_flags.sh` ->
        `quality_v2/flags/`): per covered tile with max >= 1 mm/h: argmax position, nearest
        radar, QIND there, unsupported / undecidable cell maxima, frame-level ring,
        climatological ring. ~57 s/day/worker, ~12 h on 6 workers. Launched 2026-10-02.
      - *Independent truth*: 10-min gauges, open: DWD (~1,000 stations) and MeteoSwiss
        SwissMetNet (`rre150z0`) -> `scripts/validation/fetch_gauges.py` ->
        `OPERA/validation/gauges/`. CombiPrecip and POH/MESHS are open for the last 14 days
        only: the 2012-2026 record has to come from MCH (asked). Lightning: no open
        pan-European source. EURADCLIM needs a KNMI API key.
      - *Gauge sanity check* (`scripts/validation/gauge_qc.py` -> `gauges/gauge_qc.csv`,
        gauges only, never the radar): > 50 mm / 10 min (5 values, incl. 99.9 and 149.6
        mm fault codes), stuck runs of >= 6 identical values >= 1 mm (200), and bursts
        >= 10 mm with exactly 0 before and after AND no rain at any complete gauge within
        30 km over +- 30 min (25 of 78 temporally isolated bursts). 230 of 84.7M wet
        values, 0.2% of values >= 10 mm. `validate_tail.py` masks them.
      - *Validation* (`scripts/validation/gauge_vs_radar.py` -> `validation/pairs/`, then
        `validate_tail.py` -> `validation/validation_summary.md`): radar raw / cleaned /
        flags at every gauge, gauge intervals around t (alignment picked from the lag
        correlation), corroboration rate (gauge >= 1 mm/h) per intensity bin for untouched
        pixels and for each rule's flagged pixels, rain wrongly removed, underestimation,
        QIND AUC per era, untouched-tail corroboration per year.
      - *Validation results (2026-10-03).* **Current outputs: `validation/values_20261003/`**
        (`logs/validate_values_20261003.log`). The first run (`validation/validation_summary.md`,
        `logs/validation.log`) had two bugs, fixed in `validate_tail.py`; see the end of
        this item. Data: 12.1M station-frames, 1,735 gauges (1,454 DWD, 281 SMN), 4,968
        days. The radar frame matches the gauge intervals 10-20 min later in all four
        network × era cells (log-rate correlation 0.29-0.47).
        **Presence** ("corroborated" = gauge ≥ 1 mm/h: it rained, not that the rate is
        right), by raw intensity bin (n):

        | class | [10,31) | [31,89) | [89,150) | [150,500) | ≥500 |
        |---|---|---|---|---|---|
        | untouched | 0.91 (240k) | 0.92 (33k) | 0.91 (2,142) | 0.90 (529) | – (0) |
        | no temporal support | 0.01 (565) | 0.01 (141) | 0.00 (22) | 0.00 (391) | 0.01 (578) |
        | on range ring | 0.82 (606) | 0.50 (90) | 0.00 (19) | 0.00 (783) | 0.00 (980) |
        | spike repaired | 0.13 (1,664) | 0.24 (538) | 0.24 (88) | 0.26 (53) | 0.00 (14) |
        | hot repaired | 0.95 (19) | 1.0 (2) | – | 0.00 (16) | 0.22 (19,193) |
        | tile rejected: unphysical | 0.81 (3,491) | 0.74 (744) | 0.55 (108) | 0.10 (344) | 0.05 (1,589) |
        | tile rejected: ray | 0.73 (298) | 0.57 (35) | 0.25 (8) | 0.00 (21) | 0.06 (33) |

        - *Temporal support* is the cleanest rule: about 1% corroborated in every bin.
        - *Ring*: real rain under the ring below 31 mm/h, none at 89 mm/h and above. This
          supports repairing the pixel over rejecting the tile. Rings and temporal support
          almost never fire under NIMBUS (0 and 2 pixels).
        - *Tile rejection* discards real rain: gauge pixels inside unphysical-rejected
          tiles are 67-79% corroborated below 89 mm/h under ODYSSEY, and 91-97% in every
          bin under NIMBUS. Under NIMBUS a rejected tile is a real storm with one bad
          pixel: mask the pixel instead.
        - *Untouched 150-500 mm/h*: ODYSSEY 0.69 (n=127, median gauge 14 mm/h), NIMBUS
          0.96 (n=402, median 28 mm/h). No untouched v2 pixel exceeds 500 mm/h. Untouched
          tail ≥ 31 by year: 0.77-0.93 for 2013-19, 0.92-0.95 from 2020 (2012: 0.71, n=7).
        - *Hot repair is right*: of 19,193 hot pixels ≥ 500 mm/h, 22% are corroborated
          but only 0.6% have a gauge ≥ 10 mm/h (median 0.06): clutter coinciding with
          drizzle. Presence is a weak test at high rates.
        - *Cleaning removes little rain*: of 19,904 pixels lowered by > 50% from ≥ 31 mm/h,
          the gauge saw ≥ 10 mm/h at 155 (0.8%).
        - *Underestimation*: at gauge ≥ 30 mm/h (117k station-frames), the cleaned 3×3
          radar max is ≥ 10 mm/h in 71.1% of cases, median radar/gauge 0.45.
        - *QIND (raw ≥ 31)*: ODYSSEY AUC 0.26, which is inverted (median QIND 0.20 for
          corroborated vs 0.90 for not); NIMBUS 0.53 (both medians 1.00). It does not
          discriminate, and the ODYSSEY inversion is unexplained. Possibly clutter near a
          radar gets high quality: check before using it even as a covariate.

        **Values** (`validate_tail.py` steps 6-8; `exceedance.csv`, `qq.csv`,
        `conditional.csv`, `rule_values.csv`). The exceedance ratio ρ(u) = N(radar ≥ u) /
        N(gauge ≥ u) is computed over the same station-frames, with no conditioning on
        either side, and is complete for u ≥ 5. Gauge = best single 10-min interval; 90%
        day-block bootstrap. A flat ρ means the tail has the right *shape*. Its level is
        not interpretable: point-vs-2 km and instantaneous-vs-10-min offsets are not
        quantified. ρ, both networks:

        | product | radar | u=10 | 20 | 31 | 53 | 89 | 150 |
        |---|---|---|---|---|---|---|---|
        | ODYSSEY | raw pixel | 0.54 | 0.68 | 0.93 | 2.15 | 11.0 | 204 |
        | ODYSSEY | v2 pixel | 0.50 | 0.51 | 0.51 | 0.52 | 0.82 | 6.5 |
        | ODYSSEY | v2, temporal-support + ring flags excluded | 0.49 | 0.50 | 0.49 | 0.47 | 0.54 | 1.26 |
        | NIMBUS | v2 pixel (flags change nothing) | 0.94 | 1.01 | 1.14 | 1.64 | 3.58 | 31 |
        | | gauge exceedances, ODYSSEY / NIMBUS | 382k / 94k | 111k / 28k | 46k / 12k | 12k / 3.1k | 1,926 / 458 | 101 / 13 |

        - *ODYSSEY*: flat at about 0.5 up to 53 mm/h, so the radar reads about half the
          gauge at every level but the tail shape matches. With the flags applied, it
          stays flat to 89 (0.54). **The POT level u = 31 sits on the calibrated part.**
          At 150 mm/h an excess of about 2.5 remains (1.26 vs 0.5, on 101 gauge
          exceedances).
        - *NIMBUS*: the tail is heavier than the gauges from about 31 mm/h (3.6 at 89), and
          no rule touches it. The pixels are rain (96% corroborated at 150-500) but rated
          higher than the gauges. Single instantaneous scan vs 10-min gauge totals, or hail
          in Z-R? **Test with DWD's open 1-min gauge records.** Until then, tail metrics on
          the `nimbus` split mix the product shift with this excess.
        - *Conditioning both ways*: given a v2 ODYSSEY pixel at 150-500 mm/h, the median
          gauge is 0 (n=652, mostly flagged pixels); under NIMBUS it is 24 mm/h (n=402).
          Given a gauge ≥ 89, the median radar pixel is 31 / 33 mm/h. At the top, the radar
          both misses heavy gauge rain and produces values the gauges don't see, so only
          the unconditioned ρ is read as calibration.
        - *Per rule, by value* (AUC = P(gauge under untouched > gauge under flagged), same
          raw bin; 0.5 = flags rain like any pixel):
          - temporal support 0.84-0.94;
          - ring 0.54 at 10-31 (rain) rising to 0.84 at 150-500;
          - unphysical rejection 0.44-0.50 under NIMBUS (it rejects rain);
          - **spikes 0.82-0.94 under ODYSSEY and 0.90-0.94 under NIMBUS**.
        - **Correction: spike repair is right for NIMBUS too.** The gauge under a repaired
          NIMBUS spike has a median of 0-0.6 mm/h, vs 6-24 under untouched pixels. The
          32-44% presence rate only meant light rain nearby. This replaces the earlier
          reading that the rule is too aggressive for NIMBUS.
        - Figures and write-up: `notes/data_quality_assessment.pdf` Part II (§ "Values: is
          the tail calibrated?"); `notes/figures/make_validation_figures.py`.

        **Two bugs fixed before the current run** (the old outputs are kept in
        `validation/values_20261003_buggy_unscanned_qc/`):
        - *Gauge QC masking on :15 / :45 frames.* `gauge_vs_radar.py` floors t + offset to
          the 10-min grid, so for :15 / :45 frames the gauge columns hold intervals
          labelled 5 min earlier. The mask matched exact offsets, so it missed every flagged
          value on those frames: 267 intervals masked instead of 540, and a
          71 mm / 10 min fault value reached the QQ.
        - *Pairs in tiles v2 never uses.* 4.7% of pairs are in tiles that are not fully
          covered, so they have no row in the tile table, and a missing flag read as "not
          rejected". They counted as v2 / untouched and held values up to ~10⁴ mm/h,
          including the 34 "untouched" pixels ≥ 500 of the first run.
        - Presence numbers moved by ≤ 0.03.
      - *Then decide*, per rule: reject the tile, repair the pixel, weight, or drop the rule;
        and whether the 150-500 mm/h range is kept.
- [ ] **v3 rebuild: repairs instead of tile rejection, fixed event set — launched
      2026-10-04** (`logs/rebuild_v3.log`). `scripts/dataset_v2/rebuild_v3.sh` -> `quality_v3/`,
      `OPERA/v3/`, `patches_v3/`, `configs/config_v3.yaml` (v2 untouched); ~15-16 h scan +
      store + gamma + checks on 8 workers. Disk: v2 store 59 GB, so v3 fits (~1.55 of 2.3 TB).
      - *Cleaning* (`src/data/cleaning.py` `repair_static` / `repair_unsupported`, applied by
        `src/data/day_cleaner.py` in the scan, the store and the validation): footprints
        (regions >= 150 mm/h around a > 500 core), ray components, ring pixels >= 89 mm/h,
        cells without temporal support, each lowered to the median of its outer ring.
        `--no_repair` reproduces the v2 tables exactly (checked on 2018-06-15).
      - *Validated on the same 12.1M gauge pairs* (`repair_pairs.py`, `compare_repair.py`
        -> `validation/repair_v3/repair_comparison.md`):
        - ODYSSEY rho(u) v2 -> v3: 0.52 -> 0.48 at 53, 0.82 -> 0.57 at 89, 6.46 -> 1.47
          [1.12, 1.92] at 150 mm/h (102 gauge exceedances). NIMBUS unchanged (1.15 at 31,
          3.6 at 89): its excess is not an artefact the repairs touch.
        - Repaired pixels: the gauge saw >= 10 mm/h under 5.3% of them (untouched >= 10:
          44.6%). Unsupported cells: gauge q90 0, AUC 0.95. Footprints: gauge q90 <= 0.2,
          AUC 0.90. Rings >= 89: gauge 0.
        - **Weak spot: rings at 31-89 mm/h** (n = 91): gauge >= 10 mm/h under 38.5%, AUC
          0.71. Ring repair from 89 mm/h instead of 31 would keep that rain and still catch
          the ring tail (652 pixels at 150-500, all dry). **Decided 2026-10-04: 89 mm/h**
          (`REPAIR_RING_MIN`); the validation above was run with 31.
        - Regained tiles (v2 rejected, v3 keeps; 9,625 tile-frames at gauges): pixels at
          10-89 mm/h are rain (corroborated 0.74-0.76 vs 0.86-0.88 untouched, AUC
          0.55-0.58); at >= 89 mm/h weaker (0.60 and 0.52, n = 135): some artefact halo
          survives outside the >= 150 footprint.
      - *Events*: 2013 floods and Andreas added, IOP16 unavailable, Emilia-Romagna windows
        (`notes/events.md` §7).
- [ ] **Fix the event set before training on v2** (`notes/events.md` §4–5, 2026-10-02).
      - The May–June 2013 Central European floods and the 27–28 July 2013 "Andreas"
        hailstorms are **training days in v2**: they were never added to
        `configs/prominent_events.yaml` after the 2013 download. Add them, rerun splits →
        store → gamma (~16 h).
      - Three events have no tiles: HyMeX IOP16 is absent from the archive
        (2012-10-10 → 11-08); both Emilia-Romagna episodes have static nodata holes
        (18 and 272 pixels) in their two grid tiles. Recover Emilia-Romagna with shifted
        windows for the event subsets only (option A).
      - Tag events `rate` / `accumulation` and report them separately; check the ten event
        maxima at 450–493 mm/h in the gallery.
      - Next store version: repair static holes (option B), then consider masked partial
        tiles (option C; ≤ 1% NaN adds 15% more ≥ 31 mm/h tiles, ≤ 5% adds 29%).
- [ ] **Talk to Daniele Nerini and Lionel Moret (MCH) about best practices at MeteoSwiss**
      before fixing the screen. Questions prepared in RESEARCH_NOTES §7.4: which
      composite-level filters MCH trusts for OPERA data and with what thresholds, whether MCH
      uses the OPERA QI, how to tell small intense cores from clutter spikes without volume
      data, a defensible plausibility bound for a 15-min 2 km rate, a Swiss reference
      (CombiPrecip) for validating the screened tail, and ML-dataset practice in the
      pysteps / nowcasting community. Added 2026-10-02: how to test o.o.d. capability (options
      O1–O7 and questions, RESEARCH_NOTES §7.4), and whether MCH would validate or co-own a
      released benchmark (§7.4c).
- [ ] **Consistently bad radars** (`scripts/data_quality/radar_attribution.py`, 2026-09-28,
      on the 745 audited days; outputs in `OPERA/quality/radars/`). Pixels and tail peaks are
      attributed to the nearest active OPERA radar (radar database now in `OPERA/meta/`).
      "Consistently bad" = in the worst 10% on the same signal in ≥ 3 of the 4 well-sampled
      years (2012, 2013, 2023, 2024; 2016 and 2020 have only 5–6 audited days). 7 of 260:
      - *static hot clutter* (≥ 31 mm/h in > 1% of steps, ~100× climatology): Bollène (FR,
        S-band) and Montclar (FR), both 4/4 years, plus Abbeville and Cherves (FR);
      - *dirty tail*: Ängelholm (SE; 132 tail tiles > 500 mm/h), Gelemenovo (BG, S-band; 564
        pixels ever > 500 mm/h), Røst (NO).

      Just below the threshold (2 of 4 years), but with the most unphysical tiles: Emden and
      Rostock (DE), De Bilt (NL), Virring and Bornholm (DK), Vara and Karlskrona (SE). So the
      southern Baltic / Øresund / North Sea coast is a hot spot for > 500 mm/h spikes.

      The map also shows, independently of any radar ranking:
      - RLAN rays around Iceland, Iberia, southern France and the Balkans;
      - range rings over southern Scandinavia;
      - a heavily contaminated region over Romania/Bulgaria;
      - a single tile at Weissfluhgipfel (CH) that supplies the three largest maxima of all
        745 days (110,000–123,000 mm/h).

      Caveats: nearest-radar attribution is approximate in dense networks, the ranking is
      relative, and only 745 of ~4,700 days on disk were audited.
- [ ] **Recompute the DEM-dependent tile features** (`sea_frac`, `dem_mean`,
      `argmax_over_sea`, `wet_over_sea_frac`). They were computed on the mirrored DEM, so the
      `sea_clutter` rule is meaningless on the current feature files; `radar_attribution.py`
      excludes it. Do this as part of extending the audit to the full archive, which needs
      quota headroom (`/work` writes of ~20 MB currently fail).
- [ ] **Read EURADCLIM first** (Overeem et al. 2023, ESSD 15, 1441). KNMI cleaned the *same*
      OPERA 15-min 2 km rain-rate composite (2013–2020) with a Gabella texture filter, a
      static-clutter filter on annual totals (`wradlib.clutter.histo_cut`) and a CLAAS-2
      satellite cloud-type mask (7×7 neighbourhood). It is the closest precedent for our
      screen, and its authors warn that the remaining outliers limit use "especially for use
      in extreme value modeling". Details: RESEARCH_NOTES §7.2.
- [ ] **Check the archive for product breaks.** Independently of the weather, the composite
      changes at: late 2015 (beam-blockage correction and satellite cloud mask introduced),
      **2017-09-29 08:52 UTC** (compositing switches from log (dBZ) to linear (Z) averaging),
      and **2024-07-05** (ODYSSEY → NIMBUS: archive product `QIND_RATE` → `RATE`; confirmed from the archive file names). Radar count also grows
      over time. Compare tail statistics, wet fraction and flag rates across each break
      before pooling years. Restricting to after 2017-09-29 still leaves ~8 years.
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

- [ ] **Revisit the > 150 mm/h policy of the v2 datasets** (decided 2026-09-28, provisional).
      The v2 rebuild no longer zeroes anything. Values up to 500 mm/h (~67 dBZ with
      Marshall–Palmer, i.e. hail-contaminated cores) are kept as measured, isolated
      unsupported spikes are corrected at pixel level, and tiles still above 500 mm/h after
      that are rejected as artefacts. Settle the bound with MCH (RESEARCH_NOTES §7.4), and
      check how it moves the scaler (`log1p(max)`), the top physical threshold (150) and the
      tail metrics.
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
      ~1 MB per 15-minute composite to download. On disk the per-day stores are larger than
      that (float64, two fields): ~236 MB/day measured on 2012, so ~85 GB/year. `/work` has
      ~19 TB free, but **the per-user quota is the binding limit**, not the filesystem — see
      the fetch status entry below.

- [ ] **Finish the archive fetch: 2025-10-23 .. 2026-09-22 still to download.**
      Status as of 2026-09-28, stopped on the user quota (`OSError: [Errno 122] Disk quota
      exceeded`, first on 2026-09-25 21:33 and again on a relaunch on 09-28).

      | | days | size |
      |---|---|---|
      | fetched, 2012-09-04 .. 2025-10-22 | 4,248 | ~1.1 TB |
      | original stores (2023-08-01 .. 2024-10-30, no QIND) | 435 | ~114 GB |
      | **total complete in `raw/OPERA/`** | **4,683** | **1.2 TB** |
      | **left: 2025-10-23 .. 2026-09-22** | **334** | **~46 GB download, ~60-80 GB on disk** |

      Coverage up to 2025-10-22 is complete: 4,683 of the 4,797 calendar days are on disk and
      the other 114 have no data in the archive (the planner's count of days with data
      agrees to within one). Fetched days per year: 2012 46, 2013 338, 2014-2022 every day,
      2023 214, 2024 82 (both mostly covered by the original stores), 2025 281.

      `raw/OPERA/20251023` is a partial store (no `.zmetadata`) from the failed write; the
      resume check rewrites it automatically. The remaining range is all in the `RATE` naming
      era, whose files are larger than the older `QIND_RATE` ones, hence the wide on-disk
      range.

      Before resuming: free space or get the quota raised (`quota` reports nothing on this
      node, so ask the admins for the actual limit), and check headroom with a write larger
      than a few MB — a 50 MB test passed on 09-28 and the fetch still failed 173 MB later.
      Then relaunch (cores 4-5, 2 workers, per CLAUDE.md):

      ```bash
      E=/work/fquareng/.micromamba/envs/dl-stable; R=/home/fquareng/work/data/extremes/OPERA/raw
      LD_LIBRARY_PATH=$E/lib PATH=$E/bin:$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLOSC_NTHREADS=1 \
      setsid nohup taskset -c 4,5 $E/bin/python -u scripts/data/fetch_opera_archive.py \
          --start 2025-10-23 --end 2026-09-22 --out $R/OPERA --reference $R/OPERA/20230801 \
          --skip_existing --workers 2 > logs/fetch_archive_resume.log 2>&1 &
      ```

      Invoke the env's python directly as above rather than through `micromamba run`: the
      wrapper takes a lock, so parallel launches serialise, and python must run with `-u`
      or the log stays empty behind the output buffer.

- [x] **The 435 original days were capped at 150 mm/h upstream — replaced.**
      *Checked on disk 2026-10-03:* all 457 day stores from 2023-08-01 to 2024-10-30 in
      `raw/OPERA/` are archive stores with QIND. The originals of the 101 post-switch days
      are kept in `raw/OPERA_orig_postswitch/` for the paired era comparison. The automatic
      swap of 09-29 stopped (`logs/refetch_rebuild.log`: 331/435 passed `verify_refetch.py`,
      below the 95% gate). How the remaining days were swapped is not recorded here. The
      v2 build (10-01/02) ran on the replaced stores.
      All 435 original stores (2023-08-01 .. 2024-10-30, no QIND) have a maximum ≤ 150 mm/h,
      while every archive-fetched year has 0.4–2.8% of tiles above it. The pipeline that
      produced them had already applied the 150 mm/h declutter step. On 2024-06-01 the
      archive refetch holds 6,418 pixels above 150 mm/h that the original lacks, while agreeing
      100% below 149 mm/h (footprint IoU 0.9999).
      The first v2 build (2026-09-28/29) was therefore inhomogeneous — capped tail on ~10% of
      its days — and was discarded before its gamma targets finished (report kept in
      `logs/dataset_v2_report_capped_20260929.md`).
      `scripts/dataset_v2/refetch_and_rebuild.sh` does the replacement:
      1. refetches the range into staging;
      2. verifies each day against its original (`verify_refetch.py`: complete, QIND,
         ≥ 90% of time steps, footprint IoU ≥ 0.9, ≥ 95% agreement below 149 mm/h);
      3. swaps and deletes only the passing originals, and stops if < 95% pass;
      4. recomputes the climatology, rescans 2023–24 and rebuilds v2.

      Archive files can also contain `inf`; the scan treats a tile with any non-finite pixel
      as not covered. **This also completes the QIND item below.**
- [x] **Add QIND to the original 435 days (2023-08-01 .. 2024-10-30).** Done by the
      replacement above (2026-10-03: QIND in all 457 stores). Original note: those stores in
      `raw/OPERA/` predate the archive fetcher and hold `TOT_PREC` only; every other day has
      `QIND` beside it. The full-archive fetch ran with `--skip_existing`, so it left them
      untouched — deliberately, since they are the source of the current patch set. To fill
      the gap without disturbing them, refetch that range into a separate directory, e.g.
      `fetch_opera_archive.py --start 2023-08-01 --end 2024-10-30 --out raw/OPERA_qind ...`,
      check that its `TOT_PREC` matches the originals, then copy only the `QIND` arrays across
      (and re-consolidate metadata). Pin to cores 4-11 (CLAUDE.md), 2 cores for the fetcher.

- [ ] **What the raw source actually contains** (measured on the fetched 2018-06-15, 96
      composites). The archive is far dirtier than the patch set suggests, because the
      declutter step hides it: per-timestep maxima exceed 500 mm/h in **52 of 96** steps and
      peak at **13,072 mm/h**, while the worst step has only 14 pixels above 150 mm/h and 1
      above 2000. So the offenders are isolated pixels, i.e. clutter, and `DECLUTTER_THRESHOLD`
      is doing real work — the objection is to *how*, since zeroing them punches holes in
      otherwise good fields instead of rejecting the pixel or the patch. Any rebuild needs an
      explicit outlier policy; there is no threshold-free version of this dataset.

- [ ] **Use the OPERA quality index for artefact screening — it is already in the files,
      but it is not sufficient on its own.** *What it measures* (RESEARCH_NOTES §7.1): a
      total quality index from IMGW's RADVOL-QC / BALTRAD `qi_total`. It is the product of
      individual indices for technical radar parameters, range, beam height, blockage,
      attenuation and QC detections, and in ODYSSEY it weights each radar's contribution to
      the composite. So it is an a-priori, mostly geometric *reliability* score, **not** a
      detector of non-meteorological echoes, and it is not harmonised across countries. Use
      it as a covariate or soft weight, not as the screen.
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
      31 mm/h in 35% of time steps. Next: run it on the full archive. **No labelled set**
      (DECISIONS §17). Instead, tune the rules towards clear errors only, report each rule's
      rejection rate per intensity bin, and spot-check each rule's gallery by eye.

### To explore
- [ ] **Temporal consistency of the downscaled fields** (ties in with another project of the
      researcher; details to add).
      - *Why it's open:* every model downscales each 15-min frame independently, with no
        temporal input. So nothing makes consecutive outputs consistent. For flow matching,
        independent noise per frame should make it worse (flicker in cell position and peak
        intensity).
      - *Data:* consecutive frames exist only where nothing is subsampled. `events_test`
        (and `events_nimbus`) keep every 15-min tile of every event, so they give full
        sequences. For more, add a `sequences` subset: whole test days, every frame, for
        chosen tiles.
      - *Metrics:*
        1. +15 min correlation of the prediction vs the observation, raw and
           motion-compensated, at 2 / 8 / 32 km. The same tool was started for the
           ODYSSEY/NIMBUS persistence check.
        2. Advect prediction(t) by the observed motion to t+15 and compare it with
           prediction(t+15).
        3. Accumulation consistency: the sum of 4 × 15-min predictions vs the observed 1-h
           total, which also connects to the rate-vs-accumulation question for extremes
           (RESEARCH_NOTES §7.4).
        4. Temporal spectra, and time series of the Minkowski functionals: area, perimeter
           and χ should evolve smoothly for a real storm.
        5. Frame-to-frame jitter of the peak location and value.
      - *Reference levels:* observed persistence is product-dependent (paired days: ODYSSEY
        0.41, NIMBUS 0.26 at +15 min), so compare each model with its own product's
        observations.
      - *Possible remedies to test afterwards:* temporally correlated or shared noise across
        frames for FM, conditioning on the previous frame or output, a temporal loss term.
- [ ] **Can regimes be recognised from the low-res input alone?** Compute the Minkowski
      functionals analytically on the coarse input (plus coarse intensity, wet fraction, DEM)
      and test whether they separate precipitation regimes, without using the high-res field
      as input. Score against target-derived regimes (Steiner convective fraction,
      organisation indices, γ of the target) or MeteoSwiss GWT weather types, used as
      evaluation truth only. Watch the 10×10 coarse patch: γ may need a wider coarse context.
- [ ] **If not, from the predicted Minkowski functionals?** Classify on γ of the model's
      prediction, γ(ŷ). It is also available at inference, but circular: the model could move
      a sample into an easier regime, so use a frozen, detached classifier and check against
      target-derived regimes. Plan and interpretation of the three outcomes: RESEARCH_NOTES
      §4 "To explore".
- [ ] **Reading on precipitation systems as seen on radar**: RESEARCH_NOTES §8 (textbooks,
      conceptual models, European climatologies, orographic, rain-field geometry, weather
      types).

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
- Artefact screening policy: **reject clear errors only, keep imperfections, no labelled
  set** (DECISIONS §17). Still to fix: the rule set and thresholds, pixel masking vs tile
  rejection, the declutter rule, the static-clutter mask, and QIND as a covariate (not a
  screen). Talk to MCH (Nerini, Moret) first. EURADCLIM is the precedent (RESEARCH_NOTES §7).
- Product breaks in the archive (late 2015, 2017-09-29, 2024-07-05 ODYSSEY→NIMBUS): check homogeneity before
  pooling years.
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
- First exploration: whether regimes are recognisable from the low-res input and its
  analytic Minkowski functionals alone, and if not, from γ of the prediction
  (RESEARCH_NOTES §4 "To explore"). Reading list: RESEARCH_NOTES §8.
- *Needs:* M1 (regime statistics on clean data).

### M4. Fix the four competing losses, for the deterministic and FM models
Make the Study-1 comparison fair: every loss gets a demonstrably active weight under the same
budget, and the losses are implemented as rewards for FM. The protocol is in
RESEARCH_NOTES §5.
- Fix SSIM. Remove the silent `teacher=target` fallback in optical flow, or log it.
- Starting weights are set (2026-09-23, DECISIONS §16): minkowski 1e-4, spectral 1e-1,
  ssim 2e-2, wetarea 2e-2, opticalflow 2.5e-2, in `config.yaml` `STRUCTURAL_LOSS_WEIGHTS`.
  Run `bash scripts/hpc/losses_wrapper.sh` to train all five at these weights.
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
