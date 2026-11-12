# R4 spectral loss: all runs (val, 172,050 patches, unet_latest, POT u = 50.1 mm/h, ξ_obs = +0.183)

Spectral = `FFT2DKernelCRPSLoss` on log-normalised fields; deterministic, so it is an L1 distance between Fourier magnitudes (no ensemble term).
Targets: exceedance and peak ratios 1, ξ pred +0.18, RL biases 0. Seed 42 throughout; MSE/Minkowski/area rows on V100, spectral rows on RTX PRO 6000.

| run | MAE vs vanilla | aniso | exc. pixels pred/obs | exc. ratio (obs ≥u patches) | hit | FA | peak ratio (extreme) | ξ pred | RL100 rel. bias | RL1000 rel. bias | FSS20 w11 | FSS31.7 w11 | FSS50 w11 | RAPSD log dist (mm/h) | twCRPS extreme |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MSE full | +0.1% | 2.30 | 0.10 | 0.096 | 0.042 | 7.6e-05 | 0.19 | +0.44 | -0.23 | +0.23 | 0.591 | 0.440 | 0.325 | 3.60 | 0.0098 |
| MSE cut | +1.4% | 2.32 | 0.07 | 0.072 | 0.022 | 1.8e-05 | 0.17 | +0.29 | -0.44 | -0.34 | 0.595 | 0.440 | 0.276 | 3.71 | 0.0097 |
| Mink full | +4.0% | 0.52 | 0.16 | 0.150 | 0.216 | 4.5e-04 | 0.48 | +0.19 | -0.44 | -0.45 | 0.689 | 0.567 | 0.398 | 1.19 | 0.0099 |
| Mink cut | +6.0% | 0.52 | 0.19 | 0.165 | 0.289 | 1.4e-03 | 0.49 | +0.20 | -0.39 | -0.38 | 0.707 | 0.580 | 0.389 | 1.10 | 0.0104 |
| area-only full | +2.3% | 1.41 | 0.22 | 0.209 | 0.194 | 2.8e-04 | 0.46 | +0.11 | -0.44 | -0.50 | 0.739 | 0.651 | 0.500 | 2.50 | 0.0100 |
| area-only cut | +3.8% | 1.51 | 0.23 | 0.213 | 0.210 | 4.4e-04 | 0.45 | +0.14 | -0.43 | -0.49 | 0.748 | 0.657 | 0.493 | 2.54 | 0.0102 |
| **spectral full 0.15** | +9.7% | 2.25 | 1.65 | 0.318 | 0.779 | 9.7e-02 | 3.26 | +0.74 | +24.43 | +85.04 | 0.608 | 0.446 | 0.312 | 2.72 | 0.0311 |
| **spectral cut 0.15** | +5.7% | 2.29 | 0.44 | 0.188 | 0.348 | 1.6e-02 | 0.47 | +0.76 | +3.42 | +14.53 | 0.623 | 0.453 | 0.304 | 1.26 | 0.0139 |
| **spectral full 0.05** | +3.4% | 1.92 | 0.61 | 0.174 | 0.561 | 3.9e-02 | 0.86 | +0.77 | +8.28 | +32.96 | 0.581 | 0.414 | 0.308 | 1.82 | 0.0177 |
| **spectral full 0.015** | +1.0% | 1.65 | 0.05 | 0.051 | 0.048 | 1.5e-04 | 0.23 | +0.22 | -0.48 | -0.41 | 0.527 | 0.338 | 0.198 | 2.03 | 0.0097 |
