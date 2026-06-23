# Emulating Non-Differentiable Metrics via Knowledge-Guided Learning: Introducing the Minkowski Image Loss

[![arXiv](https://img.shields.io/badge/arXiv-2604.11422-b31b1b.svg)](https://arxiv.org/abs/2604.11422)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the paper:

> **Emulating Non-Differentiable Metrics via Knowledge-Guided Learning: Introducing the Minkowski Image Loss**
>
> Filippo Quarenghi, Ryan Cotsakis, Tom Beucler
>
> *Environmental Data Science*, 2026 — Cambridge University Press
>
> [arXiv:2604.11422](https://arxiv.org/abs/2604.11422)

## Abstract

The "differentiability gap" presents a primary bottleneck in Earth system deep learning: since models cannot be trained directly on non-differentiable scientific metrics and must rely on smooth proxies (e.g., MSE), they often fail to capture high-frequency details, yielding "blurry" outputs. We develop a framework that bridges this gap using two methods: (1) analytically approximating non-differentiable functions via temperature-controlled sigmoids and continuous logical operators, and (2) learning differentiable surrogates using Lipschitz-regularised convolutional neural networks with hard architectural constraints enforcing geometric principles. We demonstrate this framework by developing the **Minkowski image loss**, a differentiable equivalent for the integral-geometric measures of surface precipitation fields (area, perimeter, Euler characteristic). Validated on the EUMETNET OPERA radar dataset, our constrained neural surrogate achieves high emulation accuracy, completely eliminating the geometric violations observed in unconstrained baselines, which generate physically impossible precipitation fields in up to 7.8% of cases.

## Method overview

<p align="center">
  <em>
    Non-differentiable scientific metrics (e.g., counting connected components) break gradient flow.
    We restore it by learning a differentiable surrogate that approximates the metric
    and integrates into the training pipeline via backpropagation.
  </em>
</p>

The framework evaluates precipitation fields at multiple intensity thresholds, computing three Minkowski functionals for each excursion set:

- **Area** — total extent of the precipitating region (km²)
- **Perimeter** — boundary length (km)
- **Euler characteristic** — connected components minus holes (χ = β₀ − β₁)

These are concatenated into a **γ-vector** that summarises the field's multi-scale geometry. The **Minkowski image loss** is the L₁ distance between log-transformed γ-vectors, integrated over the threshold axis via the trapezoidal rule.

### Emulator architecture

The constrained Lipschitz CNN pairs a spectrally normalised encoder with geometrically constrained output heads:

| Head          | Constraint                                         | Mechanism                        |
| ------------- | -------------------------------------------------- | -------------------------------- |
| **Area**      | Monotonicity (A(tᵢ) ≥ A(tᵢ₊₁))                     | Softmax → reverse cumulative sum |
| **Perimeter** | Isoperimetric inequality (P² ≥ 4πA)                | P = √(4πA) · (1 + softplus(r))   |
| **Topology**  | Non-negativity (β₀ mode) or unconstrained (χ mode) | Softplus or linear               |

## Installation

```bash
git clone https://github.com/<your-handle>/minkowski-loss.git
cd minkowski-loss

# Create environment
micromamba create -n mink-ddpm python=3.10
micromamba activate mink-ddpm

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install package
pip install -e ".[dev]"

# Verify
pytest tests/ -v
```

### Key dependencies

`torch>=2.0` · `gudhi>=3.8` · `scikit-image>=0.20` · `zarr>=2.14,<3` · `optuna>=3.0` · `scipy` · `pandas` · `matplotlib`

## Repository structure

```
Mink-DDPM/
├── config.yaml                   # All parameters (single source of truth)
├── pyproject.toml
│
├── src/                          # Library code (importable package)
│   ├── data/
│   │   ├── gamma.py              # Minkowski functional computation via persistent homology
│   │   ├── datasets.py           # PyTorch datasets (emulator, UNet, DDPM)
│   │   ├── preprocessing.py      # Filtering, coarsening, Zarr store creation
│   │   └── augmentation.py       # Offline mixup augmentation
│   ├── models/
│   │   ├── emulators.py          # Baseline CNN, Lip-CNN, Constrained Lip-CNN
│   │   ├── unet.py               # Log-space residual U-Net for super-resolution
│   │   ├── ddpm.py               # Conditional denoising U-Net
│   │   └── diffusion.py          # Cosine noise schedule, DDPM/DDIM sampling
│   ├── losses/
│   │   └── minkowski.py          # Minkowski, homoscedastic, and analytical losses
│   ├── trainers/
│   │   ├── base.py               # EarlyStopping, checkpoints, scheduling utilities
│   │   ├── emulator.py           # Emulator training with homoscedastic task weighting
│   │   ├── unet_emulator.py      # UNet + frozen emulator loss with trust gating
│   │   ├── unet_analytical.py    # UNet + analytical Minkowski loss
│   │   └── ddpm.py               # DDPM training loop
│   ├── evaluation/
│   │   ├── metrics.py            # R², Minkowski distance, isoperimetric violation rate
│   │   ├── baselines.py          # Analytical approximation evaluation
│   │   ├── jacobian.py           # Gradient norms and saliency maps
│   │   └── inversion.py          # Feature inversion test (Sec. 3.3)
│   └── utils.py                  # Config validation, logging, denormalization
│
├── scripts/                      # Thin CLI entry points (no logic)
│   ├── preprocess/               # Data pipeline stages
│   ├── train/                    # Training launchers
│   ├── evaluate/                 # Evaluation launchers
│   └── hpc/                      # Cluster launch scripts + shared env.sh
│
├── configs/                      # Per-experiment hyperparameters (Optuna-tunable)
└── tests/                        # Unit tests (pytest)
```

## Data

This work uses the pan-European EUMETNET OPERA radar composites — instantaneous surface rain rate at 2 km resolution with 15-minute temporal updates. The dataset spans August 2023 to October 2024, yielding >1M patches of 128×128 pixels (256×256 km) across diverse precipitation regimes.

**Access**: OPERA data is available under the CC BY license via the [EUMETNET Open Radar Data API](https://github.com/EUMETNET/openradardata-documentation). Access requires IP whitelisting — contact `support.opera@eumetnet.eu`.

**Chronological splits**: train (Aug 2023–Jun 2024), validation (Jul 2024), test (Aug–Oct 2024).

## Reproducing the paper results

### 1. Preprocessing

```bash
# Full pipeline: Zarr store → persistence thresholds → γ-targets → mixup
bash scripts/hpc/launch_preprocessing.sh

# Or individual stages:
python scripts/preprocess/preprocess_data.py config.yaml
python scripts/preprocess/compute_persistence_thresholds.py config.yaml
python scripts/preprocess/compute_gamma_targets.py config.yaml
python scripts/preprocess/apply_mixup.py config.yaml
```

### 2. Emulator ablation study (Table 1)

```bash
# Train all three architectures
for arch in Baseline Lipschitz Constrained; do
    python scripts/train/train_emulator.py config.yaml --arch $arch
done

# With Optuna hyperparameter search
python scripts/train/train_emulator.py config.yaml \
    --arch Constrained --optimize --n_trials 20
```

### 3. Super-resolution experiments (Table 2)

```bash
# UNet baseline (MSE only)
python scripts/train/train_unet_emulator.py config.yaml \
    --params_path configs/unet_emulator.yaml

# UNet + Minkowski loss (neural emulator)
python scripts/train/train_unet_emulator.py config.yaml \
    --params_path configs/unet_emulator.yaml --weight_geom 0.001

# UNet + Minkowski loss (analytical approximation)
python scripts/train/train_unet_analytical.py config.yaml \
    --params_path configs/unet_analytical.yaml --weight_geom 1.0

# DDPM baseline (stochastic)
python scripts/train/train_ddpm.py config.yaml \
    --params_path configs/ddpm.yaml
```

### 4. Evaluation

```bash
# Emulator accuracy and geometric fidelity (Table 1)
python scripts/evaluate/eval_emulator.py config.yaml \
    --checkpoint path/to/best_model_checkpoint.pth

# Analytical approximation baseline
python scripts/evaluate/eval_baselines.py config.yaml

# Feature inversion and gradient quality (Fig. 3, Appendix D)
python scripts/evaluate/eval_inversion.py config.yaml \
    --checkpoint path/to/best_model_checkpoint.pth
```

## Configuration

All parameters are centralised in `config.yaml`:

| Parameter                 | Description                                                       | Default         |
| ------------------------- | ----------------------------------------------------------------- | --------------- |
| `TOPOLOGY_MODE`           | `"euler"` (χ = β₀ − β₁) or `"b0"` (connected components)          | `"euler"`       |
| `ARCHITECTURE`            | Emulator to load: `"Baseline"`, `"Lipschitz"`, or `"Constrained"` | `"Constrained"` |
| `MINKOWSKI_TARGET_WEIGHT` | Maximum geometric loss weight w_max (0 disables)                  | `0.0`           |
| `MINKOWSKI_WARMUP_EPOCHS` | Cosine warmup epochs (0 → w_max)                                  | `5`             |
| `TRUST_TAU`               | Trust gating coefficient τ for emulator loss (Appendix E.2)       | `0.005725`      |
| `QUANTILE_LEVELS`         | Quantile levels for the climatological threshold CDF              | 21 levels       |

Per-experiment hyperparameters (lr, weight_decay) are in `configs/*.yaml` and tunable with `--optimize` / `--tune`.

## Testing

```bash
pytest tests/ -v
```

The test suite verifies: topological correctness on fields with known Minkowski functionals (single disk → B₀ = 1; annulus → χ = 0; two disks → B₀ = 2), architectural constraint enforcement (area monotonicity, isoperimetric inequality, domain bound), loss function gradient flow and homoscedastic parameter updates, and preprocessing invariants (mass conservation, filter bounds).

## Acknowledgments

F.Q. and T.B. acknowledge support from the Swiss National Science Foundation (SNSF) under Grant No. 10001754 ("RobustSR" project). The authors thank L. Räss for technical assistance and D. Nerini, D. Domeisen, V. Chavez, L. Moret, and O. Miralles for contributions to the initial conceptualisation.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
