# Digest A — EXPERIMENTS.md core (lines 1–907) and §6 Roadmap (2163–2230)

Source: `/work/fquareng/ch2/minkowski-loss/EXPERIMENTS.md`, header "Last updated: 2026-11-04". All content below is from the file; markers **[SUPERSEDED]**, **[REOPENED]**, **[CONTRADICTED/AMENDED]** flag places where the file itself revises an earlier statement. No editorial additions.

## Header conventions (lines 1–24)

- Read order: `DECISIONS.md` → `CLAUDE.md` → this file. `MINKOWSKI_DOCS.md` documents the loss; `RESEARCH_NOTES.md` holds theory, the definition-of-extreme question, regime-aware design notes.
- Deterministic evals on full test set (n = 285,383). Generative evals on 4,096 patches, M = 16, heun-16 unless noted.
- All tail columns at POT u = 31 mm/h; rows at other levels not comparable.
- From v4 on, target = 1-h accumulation (DECISIONS §21, 2026-10-08). Every number in §1–§3 is on 15-min rates, not comparable with hourly; u = 31 is a rate level, re-derived for hourly.
- Two caveats on every number (DECISIONS §15): (1) test split not independent of train — random patch-level splits over the same 15 months; every test patch has a train patch at same timestamp/tile within ±1 h → interpolation scores. (2) Tail contaminated by radar artefacts and holed by the declutter step; tail columns measured on unscreened data.

## 1. Compute / resources (§0, lines 27–80)

- Group assigned one GPU identified by UUID `GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4`, not a node. On node34 until the reboot of 2026-11-03, on node32 since (same two cards, same PCI slots, same CPU layout); move may be temporary. node34 now holds one unrelated card (`GPU-1765684b-…`). Node shared with other users. Find the card: `grep -l 9e0aab27 /proc/driver/nvidia/gpus/*/information`.
- CPU: 12 cores, 2 × Xeon E5-2620 v3, no SMT. NUMA 0 = cores 0–5, NUMA 1 = 6–11.
- GPU 0: RTX PRO 6000 Blackwell Max-Q 97.9 GB, PCI 02:00.0, UUID `GPU-a62d786a-…` — not ours, never use.
- GPU 1: same model, PCI 83:00.0, our UUID, NUMA-local to cores 6–11 — the only GPU we use.
- Env: micromamba `dl-stable` (torch 2.10.0+cu128).
- Limits: GPU 1 only, ≤ 8 cores at any time (≥ 4 free). All jobs pinned to cores 4–11 (GPU 1's NUMA node plus 4–5; leaves 0–3 free). Affinity inherited by DataLoader workers/pools; everything running at once shares the 8 cores (e.g. fetcher 2 + audit 6 = 8).
- GPU selection: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`; cards identical, wrong pick invisible in logs.
- Enforcement (added 2026-09-23):
  - `src/node_limits.py` on `import src`: pins to cores 4–11, caps torch threads at 8; sets GPU 1 in PCI order if nothing set; raises `NodeLimitError` for any other `CUDA_VISIBLE_DEVICES` (incl. `0`, `0,1`, GPU 0's UUID); checks UUID if CUDA already initialised. No-op on hosts without our card. Since 2026-11-04 detects the node by our GPU's UUID (from `/proc/driver/nvidia/gpus`, no CUDA start), not hostname — the hostname check (`node34`) silently disabled all guards when the card moved to node32. Default index = card's PCI-order position. If the card lands on a node without cores 4–11, `import src` raises. Tests: `tests/test_node_limits.py`.
  - `scripts/hpc/env.sh` exports the card's PCI-order index (1 on node32; previously exported `CUDA_VISIBLE_DEVICES=0`), pins the launching shell to 4–11.
  - Launchers with `GPU` override (`run_full_eval`, `eval_losses`, `launch_eval_losses`, `sweep_reward`, `check_sampler`) refuse anything but `GPU=1`. `run_diagnosis.sh` used GPU 0, now 1.
  - `launch_data_quality.sh` defaults to 6 workers, refuses > 8.
  - Verified live: affinity 4–11, torch 8 threads, one visible device with GPU 1's UUID (PCI bus 0x83), workers on 4–11; `CUDA_VISIBLE_DEVICES=0` raises.
- Per-job budgets: `config.yaml` `NUM_WORKERS: 4`, `MAX_WORKERS: 4`. 1 training run = main + 4 workers = 5; fetcher 2 (currently cores 4–5); data-quality audit 6. Do not start a second CPU-heavy job alongside a full audit plus the fetcher.

## 2. Checkpoints (§1, lines 84–126) — all 15-min, pre-v4, and (per §5 DEM item) trained with the mirrored DEM

### Deterministic backbones — `runs/sr_analytical/`

| Run | Loss | Weight | Epochs (stop/best) | Status |
|---|---|---|---|---|
| `UNet_Ana_20260623_144958` | MSE (vanilla) | 0 | 69/64, early stop | reference backbone for all FM runs |
| `UNet_Ana_20260731_100128` | MSE + Minkowski | 1e-4 | 47/46, interrupted | Study-1 winner |
| `UNet_Ana_20260630_111918` | MSE + Minkowski | 1e-4 | 100 | full-length, not evaluated |
| `UNet_Ana_20260722_081033` | MSE | 0 | 34 | short vanilla, not evaluated; matches v2 competing runs on every loss value |
| `UNet_Ana_20260729_104637` | MSE + Minkowski | 1e-3 | — | discard: reward hacking (DECISIONS §2) |
| `UNet_Ana_20260812_085343` | MSE + spectral | 1e-4 shared | — | v1, inert |
| `UNet_Ana_20260817_113622` | MSE + ssim | 1e-4 shared | — | v1, inert |
| `UNet_Ana_20260829_033325` | MSE + spectral | 5e-4 | 25/25 | v2, inert ("under-weighted ~70×") |
| `UNet_Ana_20260827_151319` | MSE + opticalflow | 4e-5 | 25/25 | v2, inert ("under-weighted ~140×") |
| `UNet_Ana_20260826_041128` | MSE + wetarea | 2e-2 | 25/25 | v2, active but degenerate |
| `UNet_Ana_20260824_155753` | MSE + ssim | 1.6e-4 | 25/25 | v2, inert, loss buggy (§3) |

**[SUPERSEDED]** The "~70× / ~140× under-weighted" factors are declared superseded in §3 (came from the noisy vanilla checkpoint).

### Flow matching — `runs/sr_flow_matching/`
- `flow_matching_20260716_071518`: vanilla backbone, σ_r = 0.01898130588233471, clean reference model.
- `Extremes_20260807_111521`: Minkowski backbone (0731), σ_r = 0.019811755046248436, Study-2 cell (evaluated from `fm_best.pth`).

### Reward fine-tuning — `runs/sr_fm_minkowski/` (both evaluated from `fm_mink_latest.pth`, DECISIONS §10)
- `Extremes_20260804_170256`: distributional; row "FM + Minkowski reward"; first coupling. Called "null-to-negative" on 09-13; **[REOPENED]** best FM row on FSS and RL bias after the 09-16 re-eval.
- `Energy_20260811_171310`: energy + retention (w = 1.0); row "FM + Minkowski energy"; best exceedance ratio and RAPSD.

### Retired / deprioritised
- DDPM dropped 2026-09-23 (DECISIONS §14). `runs/sr_ddpm/` (3 April runs + debugging run) and code (`src/models/{ddpm,diffusion}.py`, `src/trainers/ddpm.py`, `scripts/train/train_ddpm.py`, `scripts/hpc/launch_ddpm.sh`) kept, not maintained/compared/reported.
- Emulator-based Minkowski loss, lowest priority: `runs/emulator/`, `runs/sr_emulator/` (`UNet_Emu_*`, May). Worth resuming eventually as the learned alternative. Not in any table.

### v4 (hourly) runs named elsewhere in scope (not in §1's tables)
- v4 vanilla baseline running since 2026-10-21 ~21:20 (`logs/exclude_then_vanilla_hrulj.log`, w = 0, 25 epochs, ~42 min each); it is the all-data arm of the pre-break ablation.
- Hourly bracket runs (Paper item, DECISIONS §37; do not delete): `runs/sr_analytical/UNet_Ana_20261021_211732`, `_20261022_160540`, `_20261023_121956`, `_20261024_083723`, `_20261025_042024`.
- U-Net quantile_mass bracket (w*_qm = 7.7e-4): w = 2.6e-4 `UNet_Ana_20261031_145244`, w = 7.7e-4 `UNet_Ana_20261101_111234` (done, not evaluated at time of writing); 2.3e-3 started 2026-11-02 07:32; 7.7e-3 after (~20 h each). The D1 item (2026-11-04) refers to "after the qm 7.7e-3 eval", so that run was completed and evaluated by then.
- FM fine-tunes on v4: dist_qm, energy_qm, energy w_reward = 0 control (configs `configs/config_v4_adv_fm_dist_qm.yaml`, `config_v4_adv_fm_energy_qm.yaml`, `config_v4_adv_fm_energy_ctl.yaml`); dist_qm_ret queued/running (`configs/config_v4_adv_fm_dist_qm_ret.yaml`). Run directories not named in scope.

## 3. Result tables

### tab:iid (§2) — POT u = 31 mm/h, 15-min rates, re-evaluated 2026-09-16
Source `eval_results/extremes/*/extremes_summary.yaml` (`scripts/hpc/run_full_eval.sh`); pre-09-16 outputs in `eval_results_pre_patch_20260916_102140/`.
The two blocks are not comparable: `gpd_xi_obs` 0.176 (full test) vs 0.083 (4,096 FM subset); 316,652 vs 5,070 observed exceedances.

Deterministic, full test (n = 285,383): RMSE_ext / FSS@89 / RAPSD / Mink / Exc. ratio / RL bias / Anisot. / Peak ratio ext.
- Bicubic: 3.77 / 0.000 / 4.00 / 2.85 / 0.044 / 0.782 / 30.94 / 0.052
- Backbone (MSE): 3.59 / 0.040 / 3.10 / 2.12 / 0.118 / 0.643 / 1.87 / 0.201
- + spectral v2: 3.68 / 0.034 / 3.89 / 2.23 / 0.057 / 0.719 / 2.24 / 0.121
- + ssim v2: 3.68 / 0.039 / 3.83 / 2.23 / 0.059 / 0.701 / 2.14 / 0.123
- + wetarea v2: 3.72 / 0.032 / 4.59 / 2.42 / 0.036 / 0.748 / 2.57 / 0.099
- + opticalflow v2: 3.67 / 0.036 / 4.03 / 2.25 / 0.057 / 0.713 / 2.36 / 0.121
- + Minkowski 1e-4: 3.70 / **0.159** / **1.31** / 0.74* / **0.244** / **0.445** / 0.73 / **0.659**

Generative, 4,096 subset, M = 16, heun-16 (same columns + CRPS_ext):
- FM clean: 3.00 / 0.315 / 0.78 / 0.84 / 0.475 / 0.165 / 1.35 / 0.567 / 0.305
- FM + Minkowski reward: 3.00 / **0.321** / 0.65 / 0.83* / 0.534 / **0.114** / 1.32 / 0.584 / 0.305
- FM + Minkowski energy: 3.01 / 0.269 / **0.55** / 0.83* / **0.538** / 0.126 / 1.24 / 0.603 / **0.302**
- FM on Minkowski backbone: 3.03 / 0.273 / 0.74 / 0.82** / 0.466 / 0.141 / 1.27 / **0.731** / 0.308

FSS@89 window 5. * training objective; ** conditional mean trained on Minkowski, partly circular.
GPD ξ error (secondary diagnostic only, DECISIONS §10; within-block): deterministic bicubic 0.131, MSE 0.029, spectral 0.013, ssim 0.009, wetarea 0.007, opticalflow 0.053, Minkowski 0.099; FM clean 0.063, reward 0.088, energy 0.067, on Mink backbone 0.110. **[SUPERSEDED]** energy row's "best ξ" (0.043) from 09-13 no longer holds.

Changes vs 09-13: bicubic and FM + reward gained u=31 tail columns. FM clean FSS 0.289 → 0.315; FM on Mink backbone FSS 0.148 → 0.273, RL bias 0.236 → 0.141; FM + reward FSS 0.268 → 0.321. Causes not attributed. 09-16 commit added the σ_r guard; σ_r mismatch is the known way to break the Mink-backbone cell (DECISIONS §12).
**[REOPENED] DECISIONS §3:** within FM block both couplings improve exceedance ratio (0.475 → 0.534 / 0.538) and RL bias (0.165 → 0.114 / 0.126); reward coupling at no FSS cost. "Does not help FM" reopened, still one seed, one subset.

### tab:prelim (§3) — full test set, re-evaluated 2026-09-16 (15-min)
MAE / ΔMAE / MAE_ext / Mink / Isoperim % / RAPSD / SAL S:
- vanilla 0.0499 / — / 0.4030 / 2.12 / 4.57 / 3.05 / 1.27
- spectral v2 0.0502 / +0.6% / 0.4103 / 2.23 / 4.41 / 3.87 / 1.15
- ssim v2 0.0503 / +0.7% / 0.4104 / 2.23 / 4.45 / 3.81 / 1.17
- wetarea v2 0.0503 / +0.7% / 0.4101 / 2.42 / 4.17 / 4.59 / 1.27
- opticalflow v2 0.0501 / +0.3% / 0.4086 / 2.25 / 4.39 / 4.01 / 1.12
- Minkowski 1e-4 0.0519 / +3.8% / 0.4195 / 0.74 / 3.53 / 1.30 / 0.34
- FM clean 0.056† / — / 0.423† / 0.83‡ / 4.29‡ / 0.79‡ / 0.50‡ († ensemble mean, ‡ single member; 25,600-patch subset, from before 09-16 re-eval, no `eval_results/backbone/fm*` — not comparable)

Activity check: < ~1% MAE vs vanilla = loss never entered objective; all four competing losses fail it.

Activity audit 2026-09-23 (`tools/gradient_audit.py`; 512 × 128 patches, fp32, GPU 1, noise-corrected full-gradient norms, jackknife errors; raw `eval_results/gradient_audit/2026-09-23/`; math `notes/gradient_audit.pdf`). Replaces the 6-batch estimate (`tools/gradient_audit_v0/`) dominated by minibatch noise (at converged vanilla, MSE gradient 96% noise).

| Loss | λ | r at init (λ/λ*) | r at own ckpt | cos own | own loss vs vanilla (95% CI) | verdict |
|---|---|---|---|---|---|---|
| Minkowski | 1e-4 | 0.095 | 1.22 | −0.986 | −64.8% [−65.9, −63.8] | active, at equilibrium |
| wetarea | 2e-2 | 1.0 | 0.74 | −0.967 | −13.9% [−15.8, −12.1] | active (drizzle band only) |
| ssim | 1.6e-4 | 1.7e-4 | 0.157 | +0.47 | +0.1% [+0.0, +0.2] | bystander |
| spectral | 5e-4 | 1.7e-3 | 0.021 | +0.92 | +2.5% [+0.9, +4.4] | bystander |
| opticalflow | 4e-5 | 3.7e-5 | 0.0066 | −0.47 | −2.6% [−6.4, +1.2] n.s. | inert |

- Active term ends at equilibrium: share → 1, cos → −1 (notes §4). Spectral/SSIM slightly worse than vanilla on their own loss; 34-epoch vanilla also worse than 64-epoch → under-training at 25 epochs.
- At init every aux gradient aligned with MSE (|cos| 0.77–0.96); Minkowski starts 10× below parity, takes over as ‖∇MSE‖ collapses.
- Weights in parity units at init: Minkowski ~10⁻¹, spectral ~10⁻³, SSIM ~10⁻⁴, optical flow ~4·10⁻⁵, wet area 1. **[SUPERSEDED]** first audit's 70×/140× factors.
- SSIM bug (`src/losses/competing.py:101-110`): per-sample `data_range` collapses on dry patches, eps=1e-6 swamps `c1·c2`; `SSIM(target,target)` = 0.465, all-zero = 0.499 → ~93% of loss range an unreachable floor. (Fixed later: D3, 2026-11-04, `tests/test_competing_fixes.py`.)
- Runs unmatched: competing 25 epochs (still improving), Minkowski 47, vanilla 69; 34-epoch vanilla indistinguishable from spectral/SSIM/optical-flow → worse tails = under-training + inert term, not harm.
- FM: competing losses not implemented (`reward_finetune.py:263` hard-codes Minkowski).

### v4 (hourly) results recorded in the todo items (single seed unless noted)
- Hourly levels (DECISIONS §32, 2026-10-19): u = 50.1 mm for POT fits; 20.2 / 31.7 / 50.1 mm = hourly equivalents of 31 / 53 / 89 mm/h. All v4 rows share u = 50.1 mm while re-derivation on `patches_v4_adv` is deferred (researcher 2026-11-02).
- Energy fine-tune gradient budget (2026-10-29, `logs/grad_norm_terms_v4adv.txt`): reward 0.13% of total gradient norm (8.3e-4 vs 0.65 for L_FM), cos(reward,total) −0.01 → continued FM training; dist: reward 99% of gradient (total norm 8.3e-4).
- quantile_mass share (DECISIONS §39, 2026-10-29): ≥ 20 mm share 3.7% → 40.9% on heavy subset.
- §40 (2026-10-30): FM loss pixels ≥ 20 mm carry ~1% of gradient (0.72% heavy, 0.10% random); tail deficit 1–3 member spreads; 55–73% of members never reach 20–50 mm. FM + energy (§38): no tail gain, continued FM training.
- energy w_reward=0 control (2026-10-30, §42): no tail gain → §38 energy row was continued FM training.
- dist_qm (2026-10-31, §42): no gain over clean FM, worse than threshold dist on every tail column.
- energy_qm (2026-10-31, §42), val_extremes1024: exceedance ratio 0.28 → 0.67, RL bias 0.37 → 0.09, RAPSD 1.14 → 0.41, RMSE +0.6%; panels clean (`figures/v4adv_valext_fields_qm/`). Single seed. Peak overshoot check (2026-11-02): valext peak ratio q90 1.73 at target max 20–50 mm, 1.54 at 50–79, 1.30 at ≥ 79 (clean FM 1.05 / 1.00 / 0.92).
- Study-1 choice on v4 (D1, 2026-11-04): threshold form, w = 6.0e-5 by the §16 rule. (Supersedes open Study-1 weight choice; earlier w_init = 8.5e-5 from §37 step 1a.)

## 4. Tooling (§4)

- `scripts/hpc/launch_unet_analytical.sh` — train backbone `<params> <w_geom> <data_pct> [<config>]`
- `scripts/evaluate/eval_extremes.py` — tail protocol → tab:iid
- `scripts/evaluate/eval_backbone.py` — baseline → tab:prelim
- `scripts/evaluate/eval_fm.py` — FM baseline
- `scripts/hpc/eval_losses.sh` — batch-eval backbones + activity check + POT guard
- `scripts/hpc/run_full_eval.sh` — full 09-16 re-eval
- `scripts/evaluate/make_plots.py` — comparison figures
- `scripts/evaluate/diagnose_coupling.py` — reward wiring check (minutes, one batch)
- `scripts/hpc/sweep_reward.sh` — one-factor reward sweep
- `scripts/hpc/check_sampler.sh` — sampler fidelity
- `src/sweep_util.py` — `patch` config / `collect` summary to CSV
- `scripts/data/fetch_opera_archive.py` — OPERA RATE + QIND onto patch grid
- `scripts/hpc/launch_data_quality.sh` — data-quality audit under `setsid`
- `scripts/data_quality/` — raw-archive audit (clutter climatology, per-tile artefact features, split leakage, candidate rules, gallery + label sheet)
- Figures: perception–distortion plane, RAPSD with ratio panel, GPD return levels + exceedance counts, tail/safety dot chart, SAL plane, γ curves, γ residual by threshold, isoperimetric scatter.
- Additional tools named in §5: `tools/gradient_audit.py`; `scripts/evaluate/grad_norm_terms.py`; `scripts/evaluate/reward_threshold_share.py`; `fm_loss_intensity_share.py`; `scripts/validate/validate_stage.py`; `src/evaluation/day_block.py`; `src/evaluation/conditioning.py`; `scripts/evaluate/analogue_distance.py`; `scripts/data_quality/{gauge_levels,cut_input_range,heldout_tail_check,fetch_smn,gauge_hourly,hourly_effect,review_server,gauge_agg}.py`; `scripts/dataset_v2/{scan_hours,make_splits_hourly,build_store_hourly,event_table}.py`; `scripts/hpc/{prep_train_v4_adv,prep_train_v4_adv_from8,run_paired_build,run_r4_queue,run_qm_bracket_evals_then_distret,apply_radar_exclusion,losses_wrapper}.sh`.

## 5. TODO — "Highest value" (§5, lines 268–645)

0. **[open] Current plan (2026-11-04):** the robustness ladder (DECISIONS §43). **Supersedes the o.o.d. items scattered in this file.**

1. **[x] Switch to hourly targets and finish the screen before Phase 3** (decided 2026-10-08; DECISIONS §21, §22; RESEARCH_NOTES §7.4e).
   - 2026-10-08: ACRR hour convention decided. Rule set DECISIONS §23 implemented (`src/data/hourly.py`; `cleaning.anchored_lines`, `clean_frame(hot_fallback=)`; `radar_screen.frame_flagged_temporal`, `repeat_context`, `decide_hourly`; `hourly:` block in `configs/quality_v4.yaml`). `tests/test_hourly.py` 9 tests pass; 6 failures elsewhere fail on HEAD too. Other references (RADKLIM, COMEPHORE, lightning, IMERG, EURADCLIM) banked; validation uses DE/CH gauges only. Launched ~14:10: hourly Phase 0 on 199 days (`scan_hours.py` → `quality_v4/calib/hourly/`), per-rule gauge ablation on 63 gauge days (`gauge_hourly.py` → `calib/gauge_hourly/`). Review server on node34 port 8765 (claude.ai copy https://claude.ai/artifact/8nXWJKv1M2ffWTQwmRbVRN not reachable from cluster); stopped 2026-10-13. `hourly_effect.py` chained after scan.
   - Hourly `make_splits`/`build_store` done 2026-10-13 (`2c6b5f5`): `make_splits_hourly.py` (`--day_excl` for rule 5), `build_store_hourly.py` (plain-sum target, `adv_precip` alongside; ~2 min per processing day on 3 workers). 6/150 rows differed from pre-burst-rule metadata, as expected.
   - 2026-10-09..13: burst rule adopted (§26); Mahalanobis mask not adopted (§25, §26); radar ownership from contributing radars, radar pass rerun 2026-10-09 23:28 (§27); rule 5 thresholds pre-stated and applied (§28): 3,029 radar-days, mostly chronic isolated pixels on coastal radars. RADKLIM test of rule 3 (`iso31_radklim.py`) and pre-build gate (`gate_prebuild.py`, §29) running 2026-10-13. Paper limitations: `notes/opera_hourly_screen.md` §6.
   - Sub-steps: (1) researcher reviews **dropped 2026-10-08 (§24)**; 144 gallery verdicts (100/144 agree with Claude) and `review.csv` (0/100) kept as record; replaced by automatic radar-day/year exclusion (applied 2026-10-13, §28), `min_wet_jump` from normal operation, §20 criterion 2 replaced by gauge false-alarm/miss rates (fixed 2026-10-13 in §29, u = 20 mm provisional). (2) decisions (a)–(h): hour convention H-45..H; completeness 4/4 or motion-fill; plain vs advection-corrected; hot-pixel fallback (never 0; breaks exact v3 reproduction); multi-frame ray test; repair vs reject for ceiling/repeated pixels; radar-disk flag only if frame breaks from both t±15; signals P1–P5 with per-frame repair before summing. (3) validation harness from `gauge_agg.py`; baseline (fixed pairs) 1 h +0.108 [0.064, 0.145]. (4) add RADKLIM RW (DOI 10.5676/DWD/RADKLIM_RW_V2017.002; 2021 NetCDF faulty, use `supplement`) and COMEPHORE (grid changes Aug 2024); later MTG lightning, IMERG, EURADCLIM v3.0 (DOI 10.21944/7ypj-wn68). (5) motion fields (pysteps, ask first; Lucas-Kanade). (6) implement step 2. (7) re-run Phase 0, freeze thresholds. (8) Phase 3/4, raw-sum regression vs ACRR. (9) re-derive tail on hourly data. (10) Phase 5 gate.
   - **Done 2026-10-19 (§32):** v4 built (`OPERA/patches_v4`: train 1,321,889 / val 173,460 / test 218,513 / nimbus 381,736 tile-hours). Gate: 1 pass, 2a fail (minor, limitation), 2b pass, 3 replaced and passing, 4 pass, 5 explained (sub-asymptotic). Decision: train on v4.
   - Status 2026-10-21 (target = advection-corrected field, `patches_v4_adv`, `configs/config_v4_adv.yaml`, §35): store, gamma targets, dataset checks, validator, 1-epoch smoke pass (`logs/prep_train_v4_adv.log`; two `check_datasets.py` bugs fixed; overfit needs 600 steps). Radar exclusions (§36b/c, rsjas 2023, hrulj 2023; `quality_v4/biased_radar_years_all.csv`): final lists train 1,299,686 / val 172,050 / test 212,803 / nimbus 378,567. Vanilla baseline running from 2026-10-21 ~21:20.
   - Sub-items: [x] hourly `PHYSICAL_THRESHOLDS` written to `patches_v4_adv` (2026-10-21); [ ] structural-loss weight on hourly data (§37): 1a done w_init = 8.5e-5 (1e-4 transfers within 15% at init); 1b at trained vanilla (Adam share); step 2 bracket with §16 rule (best exceedance ratio with ΔMAE ≤ +5% and anisotropy ≤ 1.2× vanilla; 3 seeds at chosen point) — *effectively resolved by D1 (6.0e-5, 2026-11-04) and the qm port note "Supersedes the open Study-1 weight choice"*; [ ] re-derive POT on `patches_v4_adv` (`pot_threshold.py`) — deferred by researcher 2026-11-02, corrected field lowers tile maxima by median ~30% (§35) — *A1 (2026-11-04) fixes u_c = 50.1 with gauge support*; [x] radar exclusions; [ ] rule 5 blind to a radar bad for weeks (own seasonal q99 reference, §36 rsjas) — future revision; [x] `eval_extremes.py --pot_threshold 50.1`, day-block + leave-period-out (2026-10-19); [x] `configs/config_v4.yaml` (§33); [x] persistence thresholds etc. 2026-10-21; [x] ownerless tile-hours excluded via `*_owned.txt` (§34); [ ] pre-break ablation (§34): vanilla on post-2015-12-12 rows vs all, evaluated on post-break test (`full_<split>_flags.csv.gz` `pre_break`); pre-break fields 1.3–6.5× more fragmented at 0.1 mm; decides whether 24% of train stays; running vanilla = all-data arm, post-break arm after it (`<list>_postbreak.txt` ready); [ ] recompute σ_r per new backbone used under FM (not needed for U-Net qm bracket, researcher 2026-11-02); [x] v4 config + smoke (2026-10-21).
   - Small items (open): fix slide 10 of `notes/mch_slides.pdf` (front, not ray); Lagrangian decorrelation table (§21 caveat); ACRR identity checked 2026-10-08 on one ODYSSEY and one NIMBUS hour — still to see an hour with a whole frame missing.

2. **[x] DEM channel N-S mirrored (bug found 2026-09-28).** DEM GeoTIFF north-first, stores south-first (IoU 0.878 vs 0.554 flipped); `src/data/preprocessing.py` sliced both with same index; 20/20 test patches match mirrored slice (corr −0.32 on test patch 0, southern Finland). Every trained model had wrong DEM; relative comparisons stand, no orography claims. Code fixed 2026-09-28 (`process_batch` via `src.data.geo.load_dem_on_radar_grid`; 5/5 checked; `tests/test_geo.py`). Store regenerated without rebuild: datasets look DEM up by `(y, x)` (`src.data.geo.dem_patch`, kwarg `dem_path`); store held 79 distinct DEM tiles repeated 2.85M times (89 GB), deleted (~89 GB freed; in-place rewrite would have grown +34.7 GB). Checks: 40/40 patches per class/split identical. `dem_stats.json` recomputed: mean 286.16, std 393.53 (was 188.31 / 351.65; backup `dem_stats_mirrored_pre20260928.json`). Consequences: retrain all checkpoints; recompute `residual_stats*.json`; `Mink-DDPM` now KeyErrors (accepted). Not changed: declutter zeroing >150 mm/h, patch-level splits, stale `patches/dem/dem_patch_*.npy`, `cosine_warmup_weight` failing test. Checked on v4 store (Mont Blanc / Dutch pixels, 2026-10-21).

3. **[x] Data quality first (M1).** Policy DECISIONS §17. Done for v4: screen (§19–§31), radar exclusions (§34, §36), validation vs gauges and RADKLIM (§29, §32).

4. **[ ] Tail-sample selection.** Draw structural-loss batches from top few % by patch max of a screened pool (DECISIONS §5, §13); MSE/velocity on uniform batches.

5. **[x] Gradient-norm λ rule (DECISIONS §9).** Done 2026-09-23 for 15-min weights (Adam-share matched, `DEFAULT_STRUCTURAL_WEIGHTS`); hourly: §37.

6. **[ ] Competing losses: fair trial (M4).** Fix SSIM, calibrate λ, match epochs, 3 seeds. (Partially advanced by D3, 2026-11-04.)

7. **[ ] Plot units on hourly data (2026-10-29).** `src/evaluation/plotting.py` labels "mm h⁻¹" (lines 269, 408, 451, 533, 586–588, field titles); should read "mm" for v4; take unit from config/product; re-render `figures/v4adv_*`.

8. **[ ] FM reward coupling: attribute §38 gains and fix coupling (2026-10-29).** Sub-items: [x] gradient norm per term (2026-10-29, results above); [ ] control fine-tune w_reward = 0 — *done under item 9.2 (2026-10-30, §42)*; [ ] fix proximal anchor (`x1 = clamp(field / max_val, 0, 1)`, lines 348–353, physical field scaled by 6.42 saturating at 6.4 mm vs model coords r/σ_r near 0, outliers to |34|) — *done 2026-10-30, §41 (item 9.1)*; [ ] reward reach to high intensities, K = 1 (soft mask width τ = 0.005u at anneal 0.05) — implemented/checked 2026-10-29 (§39): `REWARD_FORM: quantile_mass`, configs `config_v4_adv_fm_{dist,energy}_qm.yaml`, `tests/test_reward_form.py`; energy reward still 0.07% of gradient before rebalance; [x] where base loss looks (2026-10-30, §40); [x] `notes/minkowski_reward_coupling.pdf` rewritten 2026-10-30.

9. **[ ] Priority order for corrected runs (researcher 2026-10-30).** Each: startup lines checked, `fm_mink_latest.pth` on `val_subset4096` and `val_extremes1024`, vs both clean-FM draws, images.
   1. FM + dist `quantile_mass`: [x] anchor fix + test (2026-10-30, §41); [x] coupling re-check replaced by gradient budget (`logs/grad_norm_terms_v4adv_anchorfix.txt`); §39 stands; [x] fine-tune dist_qm (2026-10-31, §42: no gain, worse than threshold dist); [ ] **re-run dist_qm with retention** (researcher 2026-11-02, §42): w_retain 1, w_reward by parity (1e-3 × 0.649 / 4.99e-4 = 1.30 → 1.0); config `configs/config_v4_adv_fm_dist_qm_ret.yaml`; queued after U-Net bracket evals via `scripts/hpc/run_qm_bracket_evals_then_distret.sh` (`logs/qm_bracket_evals_then_distret.log`, budget `logs/grad_norm_terms_v4adv_dist_qm_ret.txt`). D4 (2026-11-04) lists it as running. (Latest git commit labels dist_qm_ret the "reserve FM model".)
   2. FM + energy `quantile_mass`: [x] rebalance REWARD_WEIGHT 1.0 (parity 1.29; reward ~44% of norm; 2026-10-30, §41); [x] w_reward = 0 control `config_v4_adv_fm_energy_ctl.yaml` (2026-10-30, §42); [x] energy_qm fine-tune (2026-10-31, §42, numbers above); [ ] 3 seeds — lower priority (researcher 2026-11-02), after dist_qm_ret and competing losses.
   3. U-Net + Minkowski `quantile_mass`: port done 2026-10-30 (`MINKOWSKI_FORM: quantile_mass`, `QuantileMassMinkowskiLoss`, 6 tests, `configs/config_v4_adv_qm.yaml`); parity audit at init (`eval_results/gradient_audit/2026-10-30_v4adv_qm_init/`); evals labelled `mink_qm_<w>` in `eval_results/{backbone,extremes}/v4adv_val/` from 2026-11-02 09:11; bracket `logs/bracket_qm_v4adv.log` (runs listed in §2 above). Supersedes the open Study-1 weight choice. **[RESOLVED by D1, 2026-11-04]**: Study-1 choice = threshold form 6.0e-5.

10. **[ ] Follow-ups from researcher's margin notes (2026-10-30).** Answers in `notes/minkowski_reward_coupling.tex` (PDF predates them).
   - [ ] Spread decomposition — BANKED 2026-11-02; plan drafted, not run (`scripts/evaluate/spread_decomposition.py`; 64 train batches B=8; M=4 and 16; "dominates" if corrected between-fraction ≥ 0.8; five open choices).
   - [ ] Bulk check on dist_qm (CRPS, MAE, RAPSD, low-threshold FSS vs clean FM).
   - [ ] λ_s = 0.5 untested default (from `Extremes_20260804_170256`); convex form + pre-fixed rule if kept.
   - [ ] Energy score L1 vs L2 (to be done eventually, researcher 2026-11-02). Current L1 = weighted sum of 45 CRPS; L2 with β ∈ (0,2) strictly proper for joint law (Gneiting & Raftery 2007 — "from search results; paper not yet opened"). Literature notes from snippets: Pinson & Tastu 2013; arXiv 2101.12693; Bernard & Mueller arXiv 2004.11770; Stock PhD thesis (Siegen); Scheuerer & Hamill 2015 MWR 143 (variogram score). Design needs: feature standardisation, β (1 and smaller), joint-consistency diagnostic, parity rule.
   - [ ] Threshold grid as hyperparameter — BANKED 2026-11-02. qm weighting facts: log-u trapezoid (0.1 mm 4.9%, interior ~7%, 79 mm 3.4%, ≥ 20.2 mm 20%); perimeter/Euler (n_j+1)^-α, α = 1; log1p compression (0.1 mm → 0.095, 50 mm → 0.68); top n_j* − 1 pixels free (gap up to ~1.58×). Equal target-pixel spacing rejected; log-uniform kept fixed. "Nothing constrains above 79 mm" ruled out (overshoot numbers above). Plan E0 (share by intensity, ~1 h), E1 (rank-1 area reading or grid to 125/200 mm; prediction q90 peak ratio 1.7 → ~1), E2 (α = 0.5 only if E0 differs), E3 (u_min 0.1 → 0.5 mm, drops 24% of measure, links §34). Open: which study; energy_qm seeds first on FM.
   - [ ] Citations to verify: Hadwiger's theorem, Cauchy-Crofton, straight-through estimator (Bengio et al. 2013), FM marginal velocity theorem, energy score (Gneiting & Raftery 2007), energy distance.
   - [x] Channel sum (deterministic, 3d) vs mean (reward, d): left as is (researcher 2026-11-02).

11. **[ ] Test the general hypothesis across losses (DECISIONS §40).** Per competing loss: (i) gradient share on ≥ 20 mm, (ii) share out of reach; prediction: at matched norm, tail columns rank as (i)–(ii). Include inverse-mass-weighted FM loss as non-proper baseline.

## 6. Robustness ladder (added 2026-11-04, DECISIONS §43; rungs R0–R6 defined there)

**A. Rules before any result**
- [x] A1 (2026-11-04): u_c = 50.1 (researcher). SwissMetNet gauge check (`scripts/data_quality/gauge_levels.py`, `quality_v4/gauge_levels/`; two Swiss tiles, 6.4 M pairs, 281 stations): radar ≥ u median gauge/radar 0.75 / 0.60 / 0.36 / 0.21 / 0.15 at 5 / 10 / 20.2 / 31.7 / 50.1 mm; gauge ≥ u radar 3×3/gauge 0.49 → 0.33; symmetric at 20 mm (0.36 vs 0.37), diverge above (0.21 vs 0.34 at 31.7; 0.15 vs 0.29 at 50.1); unconfirmed radar hours 12% → 57%, misses ~10% to 31.7 mm; radar reads heavy above ~30 mm (inference; hail usual cause); only 23 pairs at 50.1. DWD (`quality_v4/gauge_levels_dwd/`, 8.9 M pairs, 1,452 stations): ODYSSEY 0.89 / 0.82 / 0.72 / 0.63 / 0.58; unconfirmed 7% → 16% at 50.1 (n = 56); gauge ≥ u 0.79 → 0.53, misses 1% → 10% → ODYSSEY tail holds to 50 mm over Germany, u_c supported; Swiss heaviness regional (Alps). NIMBUS heavy at top: 0.70 → 0.26 at 50.1, unconfirmed 5% → 33% (58% at 80, n = 12), misses 0%, 3.5× more ≥ 50 mm hours than gauges (70 vs 20; ODYSSEY 56 vs 125) → support for gauge-anchored NIMBUS tail scoring. Counts (screened v4; tile-hours/days with tile max ≥ u):

| u (mm) | test tile-h | test days | nimbus days | val days | train tile-h | removed day×3×3 | tile-hour only |
|---|---|---|---|---|---|---|---|
| 50.1 | 2,383 | 305 | 569 | 205 | 14,067 | 17.5% | 1.08% |
| 60 | 1,458 | 258 | 521 | 173 | 8,974 | 13.5% | 0.69% |
| 80 | 559 | 171 | 389 | 131 | 4,156 | 8.1% | 0.32% |
| 100 | 249 | 109 | 303 | 94 | 2,090 | 4.8% | 0.16% |
| 125 | 98 | 60 | 197 | 53 | 884 | 2.6% | 0.07% |

  Day rule → u_c ≤ 100 mm. Gauge agreement per level "still open (needs B2)". **[AMENDED]** Proposed amendment (originally "pending the researcher"): remove only tile-hours ≥ u_c, not day × 3×3 blocks (v4 splits ISO-week blocked; 3×3-day block removes 8.1% of train at 80 mm, 5.9% at ±6 h). A2 records it as adopted.
- [x] A2 (2026-11-04, tile-hours only per §43 amendment): `full_train_cut50.1.txt` (1.08% removed) and control `full_train_randdrop50.1.txt` (same count, random) in `patches_v4_adv/meta/`. **[CONTRADICTED]** The item's original spec text (drop day × tile + 8 neighbours blocks) is superseded by the amendment.
- [x] A3 (2026-11-04, `scripts/data_quality/cut_input_range.py`, `quality_v4/cut_input_range/`): held-out test inputs coarse max q50 6.6 mm (NIMBUS 6.9); 0.5% exceed largest cut-train input (29.4 mm), 4.2% its q99.9 (20.1 mm); target/coarse max q50 10.3 → R4 asks for an unfamiliar fine peak from a familiar coarse field, not input extrapolation. One train input has 191 mm coarse max (artefact).
- [x] A4 (2026-11-04): `notes/event_table.csv` (`scripts/dataset_v2/event_table.py 50.1`): 18 events in R1|R4, 10 R1 only, R2 3 (Boris R2|R5), R3 8, R3|R5 8, 5 MCH events not scored (train/val). Showcases frozen: C1 Misox, C2 Boris + MCH 36, C3 MCH 64.

**B. Data for R2 and independent truth**
- [ ] B1 running 2026-11-04: ODYSSEY scan of 101 days (`quality_v4/hourly_odyssey_paired/`, ~4 h), then `scripts/hpc/run_paired_build.sh` → `patches_v4_paired`; ODYSSEY originals 2024-07-05..10-30 (`raw/OPERA_orig_postswitch/`) for NIMBUS tile-hours. Verify counts, Boris and MCH 36 in both, 150 mm/h cap interaction.
- [ ] B2 running 2026-11-04: `scripts/data_quality/fetch_smn.py` → `/home/fquareng/work/data/extremes/MCH/smn/` (~300 stations, both open networks, 2010 → now; log `.../MCH/fetch_smn.log`). Checked on LUG: hourly at T = sum of 10-min T-50..T, both end-of-interval; gauge hour T matches radar hour T. (Spec: `rre150z0`/`rre150h0`, STAC `ch.meteoschweiz.ogd-smn`, 72 listed stations.)
- [ ] B3 open: CombiPrecip hourly reanalysis (CPCH) 2020–2026 via MeteoSwiss form; Swiss Data Cube licence; `TI*` stations (20 of 92 not open).
- [x] B4 (2026-11-04): Storm Sabine (`_id` 1), 15/96 tile-hours, max 0.015 mm — not a screen failure (MCH window 2020-02-07 23:00..02-09 23:00 UTC = wind phase; rain on 02-10; dry hours thinned by stratified sampling). Isolated 100.1 mm hour at 2020-02-10 00 UTC → B5 gallery (no day-boundary bug; isolated ≥ 50 mm spikes 2% of ≥ 50 mm tile-hours).
- [ ] B5 first pass 2026-11-04 (`scripts/data_quality/heldout_tail_check.py`, `quality_v4/heldout_tail_check/`): isolated spikes 1.8% of test, 0.4% NIMBUS tile-hours ≥ 50.1, none ≥ 150. Among 48 largest held-out tile-hours (all but one NIMBUS) ~a third clear errors missed (rays/wedges e.g. (768,1280) 2025-07-09 05–07 UTC, (128,384) 2025-01-06; stationary spikes (768,1280) 2026-03-31/04-01, 314–358 mm). Random sample ~4–5/48 suspect. Decision (§43 amendments): gauge-anchored tail scoring for R3/R5; no NIMBUS rule (`n_persist` would reject real storms; `n_ray` silent; tail spread over 37 radars). Still to do: gallery incl. 2022-08-19 179 mm tile, rejection count by intensity bin.

**C. Evaluation code**
- [x] C1 (2026-11-04): input-conditioned selection (`src/evaluation/conditioning.py`, key `conditioned`, `tests/test_conditioning.py`); smoke on 2,560 `extremes_val`, bicubic: observation- and input-selected sets (127 each) share 27%.
- [x] C2 (2026-11-04): `twcrps_mean`, `twcrps_extreme`.
- [x] C3 code done 2026-11-04: `eval_extremes.py --gauge_pixels /home/fquareng/work/data/extremes/MCH/gauge_pixels.csv` (1,827 SMN + DWD stations; DWD pixels match `src.data.geo`); NIMBUS via `--split nimbus`; DWD 10-min 2010–2025 in `validation/gauges/raw`. Smoke test pending a free core.
- [x] C4 (2026-11-04): `scripts/evaluate/analogue_distance.py`; `quality_v4/analogue/`: cut vs full nearest-analogue distance unchanged (median ratio 1.00 every bin, 16,938 test cases ≥ 20.2); distance grows with intensity (q50 1.78 at 20–50 mm, 2.36 at ≥ 100). Coarse-input space cannot grade unseenness for R4; needs target-space distance (normalised gamma, RESEARCH_NOTES §3 Q5), evaluation-only.

**D. Models**
- [x] D1 (2026-11-04): Study-1 = threshold form, 6.0e-5 by §16 rule.
- [x] D2 (2026-11-04): `MINKOWSKI_CHANNEL_WEIGHTS` (default [1,1,1]); `tests/test_minkowski_channels.py`. Area-only weight still needs a rule (pending).
- [x] D3 changed 2026-11-04: all four competitors at parity with Minkowski 6.0e-5; SSIM floor fixed, optical-flow fallback visible (`tests/test_competing_fixes.py`); weights from parity audits in the R4 queue. (Original spec was "one competing loss".)
- [ ] D4: Study 2 dist_qm_ret (running), then energy_qm 3 seeds.

**E. Training for R4**
- [ ] E1 queued 2026-11-04 (`scripts/hpc/run_r4_queue.sh`, after tonight's chain): parity audits, then 21 runs (seed-42 MSE/Minkowski cut pair first); MSE, Minkowski, Minkowski area-only, competing loss(es); 3 seeds; ~20 h/run, ~10 GPU-days. Verify startup lines (loss, weight, train list = cut list, GPU).
- [ ] E2: FM base on cut list, σ_r recomputed, then energy_qm; verify `config FM_RESIDUAL_SCALE` line; FM base cost to measure.

**F. Scoring (test touched once per final model)** — all open: F1 R0, R1, R3 every final model, R2 on paired store; F2 R4/R5 full vs cut, degradation per rung with day-block intervals + leave-period-out, observation- and input-conditioned side by side; F3 gauge verification for showcases and every MCH rain event outside train; F4 figures per showcase + appendix event table; R6 (DPC Italy) only if R4 positive.

## 7. Validation stages (added 2026-10-19)
`scripts/validate/validate_stage.py <stage>` → `validation/<stage>/<date>_<commit>.md`, exits non-zero on FAIL; missing artefact = FAIL; validators unit-tested by corruption (`tests/test_validate.py`). Signed-off report path goes in DECISIONS/EXPERIMENTS when used.
- [x] store (2026-10-19); [x] gamma (2026-10-19) — both gate `scripts/hpc/prep_train_v4.sh`.
- [x] training (2026-10-19, `scripts/validate/training.py`, `tests/test_validate_training.py`): thresholds = config, finite non-zero gradients at w_max, overfit one batch (MSE ≤ 20%), determinism, checkpoint round trip, warm-up (INFO), run log. With `MINKOWSKI_WARMUP_EPOCHS = 5` a 1-epoch smoke run is MSE-only. Still to add for FM: σ_r startup line.
- [ ] evaluation (known answers; `tests/test_day_block.py` exists); [ ] splits (partly `check_datasets.py`); [ ] scan/screening; [ ] raw archive.

## 8. Uncertainty of tail tables (added 2026-10-19, DECISIONS §32)
- [x] Done 2026-10-19: `--bootstrap B`, `--leave_out 20210616-20210630 @<days file>` (`src/evaluation/day_block.py`, `tests/test_day_block.py`); keys `dayblock_point`, `dayblock_interval95`, `leave_out`. v4 run: `--pot_threshold 50.1 --bootstrap 200 --leave_out 20210616-20210630 @quality_v4/gate_tail/dropped_check/dropped.txt`. On v4 test, 20 of 486 days hold 27% of pixel-hours ≥ 20.2 mm (June 2021, June 2024, August 2022). Report 95% paired day-block interval and leave-period-out (without 16–30 June 2021, and without the 20 most influential days in `quality_v4/gate_tail/summary.md`) for every tail statistic. Differences smaller than the interval not claimed.

## 9. Generalisation to unseen products/regions (added 2026-10-14; roadmap M6) — [ ] open
- Test v4 model on Italian radar; no Italian radar in OPERA (DECISIONS §27) → held out by construction.
- Product: DPC `italian-radar-dpc-srt1-adj` (FBK ArcoDataHub), hourly, 1 km, gauge-adjusted, 2018-07 on; not `opera-plus-it-vmi`.
- Preprocessing: 1 → 2 km area mean onto OPERA grid; DEM via `src.data.geo`; check hour convention vs ACRR; same scaler; evaluate "as delivered" and "screened".
- Ablation: A OPERA test (reference); B RADKLIM over Germany (product only); C DPC on Alpine border strip with OPERA coverage (product, same rain); D DPC interior Italy (region + product). Region ≈ D vs C; product ≈ B, C vs A. Reasons: scales < ~6 km product-specific (methods draft §6.5); OPERA reads 15–20% low at moderate intensities (§6.6).
- Metric: skill relative to a baseline on same patches; report target spectra and Minkowski curves per product.
- Assumption: perfect-model setup (input = coarsened target).
- Steps: (1) DPC–OPERA overlap; (2) licence (CC BY-SA) and hour convention with FBK (franch@fbk.eu); (3) fetch/regrid 2018-07..2024; (4) build A–D; (5) evaluate. B needs only RADKLIM.
- Events: N Italy hail 2023-07-19, La Chaux-de-Fonds / Azzano, Emilia-Romagna 2023; event email to Lionel Moret 2026-10-14.

## 10. Paper plan (all open)
- [ ] Write up hourly weight calibration and bracket (researcher 2026-10-26; DECISIONS §37). Artefacts: `eval_results/gradient_audit/2026-10-21_v4adv_init`, `2026-10-22_v4adv_vanilla`; five bracket runs (listed §2); `eval_results/{backbone,extremes}/v4adv_val/`; `notes/data_analysis_v4/bracket_validation.csv`, `bracket_gallery.pdf`. Points: weight transfers by gradient share (1a vs 1b, 2.1× apart); on hourly data the loss costs ~2× the MAE of 15-min; striping grows with weight and the anisotropy cap misses it; every weight thins the long-return-level tail; bracket extended downward because winner sat at its edge.
- [ ] Limitation of Minkowski loss on deterministic model (2026-10-26): `notes/minkowski_deterministic_limitation.md` (motion invariance/Hadwiger, MSE double penalty, grid-estimator anisotropy, fixed stamp, rotation test); citations `[CITE?]`.
- [ ] Write up how a loss coupling is checked (2026-10-26): training validator, gradient/Adam-share audit (`tools/gradient_audit.py`, `notes/gradient_audit.pdf`), activity check (ΔMAE ≥ +1%), reward-drop test (`diagnose_coupling.py`, DECISIONS §4), image check.
- [ ] Systematic literature search before claiming any contribution (2026-10-21): candidates — weight transfer across datasets by gradient share; hacking-bounded weight selection (DECISIONS §37). Plan `notes/weight_selection_literature.md`. Until done, motivate from GradNorm; Wang, Teng & Perdikaris; Zhong & Maki; Blau & Michaeli; Goodhart literature; claim nothing new.
- [ ] Add W₁ identity to theory (DECISIONS §13).
- [ ] Report exceedance ratio + RL bias as tail columns; demote ξ to text with confound explained.
- [ ] State POT-level choice as estimability, not physics.
- [ ] Write up w=1e-3 hacking episode as positive result on necessary-not-sufficient.
- [ ] Theory framework (M5, `RESEARCH_NOTES.md`).
- [ ] Dataset limitation: coverage rule and coastal bias (only fully covered patches; under-represents coastal/network-edge Mediterranean extremes, drops Emilia-Romagna events; HyMeX IOP16 not in archive). Draft + three improvements (event windows, blind-spot repair, masked partial patches) in `notes/events.md` §6.

## 11. Roadmap §6 (lines 2163–2230) — not in order; theory in `RESEARCH_NOTES.md`

- **M1. Data analysis on full dataset** (archive 2012 →), before retraining. Screening policy §17; still to fix (as written): rule set/thresholds, pixel masking vs tile rejection, declutter rule, static-clutter mask, QIND as covariate; talk to MCH (Nerini, Moret); EURADCLIM precedent (RESEARCH_NOTES §7). Product breaks: late 2015, 2017-09-29, 2024-07-05 ODYSSEY→NIMBUS — check homogeneity. Rebuild with per-patch quality column and event-blocked splits. Descriptive analysis for M2/M5 (tail stats, γ curves extreme vs ordinary, distribution by season/region/regime). Blocks M2, M3, tail results. *Status per §5: "Data quality first" marked done for v4 (screen §19–§31, exclusions §34/§36, gauge/RADKLIM validation §29/§32); v4 uses week-blocked splits; the roadmap text itself is not updated.*
- **M2. Cut-off training, o.o.d. evaluation (tab:ood).** Direct test of central hypothesis (M5). Decide "extreme" (RESEARCH_NOTES §2); cut by event and in time on screened pool (patch-level cut leaks). Needs M1. *Status: now implemented as the robustness ladder R4 (§43): u_c = 50.1, tile-hour cut lists done, R4 runs queued 2026-11-04. Roadmap text says "by event and in time"; the §43 amendment removes only tile-hours ≥ u_c, relying on ISO-week blocking.*
- **M6. Generalisation to unseen product/region** (added 2026-10-14): DPC Italy, product × region ablation; needs v4 model and DPC `srt1-adj` regridded. *Status: open; per ladder, R6 only if R4 positive.*
- **M3. Regime-aware Minkowski training.** Regime from low-res input, prediction, predicted functionals; start from ECMWF ecPoint gridbox weather types (RESEARCH_NOTES §4); regime definition open; first exploration: regimes from low-res input + analytic functionals, else γ of prediction; reading list RESEARCH_NOTES §8. Needs M1. *Status: open.*
- **M4. Fix four competing losses (deterministic and FM).** Protocol RESEARCH_NOTES §5. Fix SSIM; remove/log silent `teacher=target` optical-flow fallback. Starting weights 2026-09-23 (DECISIONS §16): minkowski 1e-4, spectral 1e-1, ssim 2e-2, wetarea 2e-2, opticalflow 2.5e-2 in `config.yaml` `STRUCTURAL_LOSS_WEIGHTS`; `bash scripts/hpc/losses_wrapper.sh`. λ by gradient-norm rule + bracket, fixed epoch budget, last checkpoint, 3 seeds. Add spectral/SSIM/wet-area/optical-flow reward paths to `reward_finetune.py`. Independent of M1 for fixes; final numbers wait for rebuilt data. *Status: SSIM fix and optical-flow fallback visibility done 2026-11-04 (D3); all four now at parity with Minkowski 6.0e-5 on v4 (15-min starting weights above are superseded for v4); FM reward paths not reported done.*
- **M5. Theory framework.** Geometry as transferable handle for robustness to unseen extremes; testable via M1 and M2; RESEARCH_NOTES §1–3. *Status: open (Paper item).*

## 12. Supersessions / contradictions within scope (summary)
- §1 checkpoint table and §2–§3 tables: all 15-min, pre-DEM-fix (mirrored DEM), random leaking splits; not comparable with v4.
- "Under-weighted ~70×/~140×" (§1) superseded by §3 audit.
- DECISIONS §3 "does not help FM" reopened by 09-16 re-eval; energy "best ξ 0.043" no longer holds.
- On v4, the 15-min "FM + energy" gain pattern is attributed to continued FM training (§38, §42 control); energy_qm is the effective FM coupling (single seed).
- §37 hourly weight bracket / Study-1 weight choice superseded by D1 (threshold 6.0e-5, 2026-11-04).
- Highest-value POT re-derivation item (deferred) vs A1 fixing u_c = 50.1 (2026-11-04).
- A1/A2 day×3×3 block rule replaced by tile-hour-only cut (§43 amendment).
- D3 widened from one competitor to all four.
- Roadmap M1/M2/M4 text not updated to reflect v4 and ladder progress.
- §0: review server/"node34" references in the hourly item predate the 2026-11-03 move to node32.
