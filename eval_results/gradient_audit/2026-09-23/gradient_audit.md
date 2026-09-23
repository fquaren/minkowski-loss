# Gradient audit

## Gradients at `init` (init, epoch 0), 512 x 128 patches, anneal 0.1

MSE: value 1.984e-03, signal ||mu|| 5.365e-01 ± 2.7e-03, per-batch median 5.387e-01 (noise fraction 0.01)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.05e-03 (± 1.4e-05) | 0.095 | 1.01e-03 | +0.777 | +0.777 | nan | 0.10 | 0% |
| spectral | 5.0e-04 | 2.98e-01 (± 1.0e-03) | 0.00168 | 2.98e-01 | +0.956 | +0.956 | nan | 0.01 | 0% |
| ssim | 1.6e-04 | 9.32e-01 (± 7.6e-03) | 0.000172 | 8.43e-01 | -0.765 | -0.765 | nan | 0.18 | 0% |
| wetarea | 2.0e-02 | 2.00e-02 (± 8.3e-05) | 1 | 1.97e-02 | +0.847 | +0.847 | nan | 0.04 | 64% |
| opticalflow | 4.0e-05 | 1.09e+00 (± 1.7e-02) | 3.67e-05 | 9.77e-01 | +0.789 | +0.789 | nan | 0.16 | 0% |

## Gradients at `vanilla` (best, epoch 64), 512 x 128 patches, anneal 0.1

MSE: value 2.341e-04, signal ||mu|| 6.059e-05 ± 6.5e-06, per-batch median 3.023e-04 (noise fraction 0.96)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 4.17e-06 (± 4.5e-07) | 24 | 1.82e-05 | -0.691 | -0.717 | 16.1 | 0.25 | 0% |
| spectral | 5.0e-04 | 9.17e-03 (± 9.3e-04) | 0.0545 | 3.90e-02 | -0.241 | -0.252 | 0.0692 | 0.26 | 0% |
| ssim | 1.6e-04 | 2.95e-04 (± 3.2e-05) | 0.543 | 1.46e-03 | -0.811 | -0.840 | 0.132 | -0.02 | 0% |
| wetarea | 2.0e-02 | 3.55e-04 (± 3.8e-05) | 56.4 | 1.78e-03 | +0.840 | +0.869 | 16.3 | -0.01 | 0% |
| opticalflow | 4.0e-05 | 1.02e-03 (± 1.1e-04) | 0.0391 | 5.03e-03 | +0.275 | +0.285 | 0.027 | -0.02 | 0% |

## Gradients at `minkowski` (best, epoch 46), 512 x 128 patches, anneal 0.1

MSE: value 2.608e-04, signal ||mu|| 9.078e-04 ± 1.5e-05, per-batch median 9.637e-04 (noise fraction 0.11)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 8.17e-05 (± 1.7e-06) | 1.22 | 5.19e-05 | -0.982 | -0.986 | 1.1 | 0.63 | 0% |
| spectral | 5.0e-04 | 8.58e-02 (± 1.7e-03) | 0.00583 | 7.57e-02 | +0.298 | +0.298 | 0.00694 | 0.31 | 0% |
| ssim | 1.6e-04 | 2.95e-03 (± 4.4e-05) | 0.0541 | 3.16e-03 | -0.734 | -0.734 | 0.0208 | 0.01 | 0% |
| wetarea | 2.0e-02 | 3.65e-03 (± 4.1e-05) | 5.48 | 3.93e-03 | +0.675 | +0.675 | 3.32 | -0.02 | 0% |
| opticalflow | 4.0e-05 | 8.77e-03 (± 1.8e-04) | 0.00456 | 9.73e-03 | +0.884 | +0.885 | 0.00513 | -0.04 | 0% |

## Gradients at `spectral` (best, epoch 25), 512 x 128 patches, anneal 0.1

MSE: value 2.458e-04, signal ||mu|| 2.822e-04 ± 6.2e-06, per-batch median 3.580e-04 (noise fraction 0.38)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.68e-05 (± 3.5e-07) | 5.94 | 2.03e-05 | +0.878 | +0.879 | 5.08 | 0.11 | 0% |
| spectral | 5.0e-04 | 2.43e-02 (± 4.9e-04) | 0.0206 | 2.95e-02 | +0.914 | +0.915 | 0.0221 | 0.04 | 0% |
| ssim | 1.6e-04 | 1.41e-03 (± 3.5e-05) | 0.113 | 1.83e-03 | +0.570 | +0.571 | 0.0366 | 0.01 | 0% |
| wetarea | 2.0e-02 | 1.57e-03 (± 4.1e-05) | 12.7 | 2.04e-03 | -0.340 | -0.341 | 4.95 | 0.02 | 0% |
| opticalflow | 4.0e-05 | 5.63e-03 (± 1.4e-04) | 0.0071 | 7.07e-03 | -0.644 | -0.645 | 0.00752 | -0.02 | 0% |

## Gradients at `ssim` (best, epoch 25), 512 x 128 patches, anneal 0.1

MSE: value 2.389e-04, signal ||mu|| 2.319e-04 ± 9.0e-06, per-batch median 3.831e-04 (noise fraction 0.63)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.36e-05 (± 4.2e-07) | 7.35 | 2.00e-05 | +0.735 | +0.737 | 5.2 | 0.15 | 0% |
| spectral | 5.0e-04 | 1.63e-02 (± 7.4e-04) | 0.0308 | 2.56e-02 | +0.793 | +0.795 | 0.0224 | 0.04 | 0% |
| ssim | 1.6e-04 | 1.02e-03 (± 3.7e-05) | 0.157 | 1.69e-03 | +0.469 | +0.470 | 0.0377 | -0.00 | 0% |
| wetarea | 2.0e-02 | 1.06e-03 (± 4.3e-05) | 18.8 | 1.72e-03 | -0.287 | -0.288 | 5.13 | 0.00 | 0% |
| opticalflow | 4.0e-05 | 4.32e-03 (± 1.7e-04) | 0.00926 | 7.11e-03 | -0.677 | -0.678 | 0.00802 | -0.02 | 0% |

## Gradients at `wetarea` (best, epoch 25), 512 x 128 patches, anneal 0.1

MSE: value 2.521e-04, signal ||mu|| 1.050e-03 ± 9.2e-06, per-batch median 1.097e-03 (noise fraction 0.08)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 6.30e-05 (± 7.0e-07) | 1.59 | 5.86e-05 | +0.933 | +0.934 | 2.7 | 0.18 | 0% |
| spectral | 5.0e-04 | 6.98e-02 (± 5.7e-04) | 0.00716 | 6.85e-02 | +0.705 | +0.705 | 0.0143 | 0.10 | 0% |
| ssim | 1.6e-04 | 4.40e-03 (± 3.2e-05) | 0.0364 | 4.58e-03 | +0.983 | +0.984 | 0.0214 | -0.00 | 0% |
| wetarea | 2.0e-02 | 2.70e-02 (± 7.8e-04) | 0.74 | 2.65e-02 | -0.966 | -0.967 | 1.03 | 0.16 | 0% |
| opticalflow | 4.0e-05 | 2.12e-02 (± 1.1e-04) | 0.00189 | 2.21e-02 | -0.318 | -0.318 | 0.00343 | 0.00 | 0% |

## Gradients at `opticalflow` (best, epoch 25), 512 x 128 patches, anneal 0.1

MSE: value 2.512e-04, signal ||mu|| 3.602e-04 ± 9.2e-06, per-batch median 4.933e-04 (noise fraction 0.47)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.76e-05 (± 7.7e-07) | 5.68 | 2.15e-05 | +0.935 | +0.937 | 5.1 | 0.14 | 0% |
| spectral | 5.0e-04 | 1.83e-02 (± 4.8e-04) | 0.0274 | 2.38e-02 | +0.861 | +0.862 | 0.0224 | 0.05 | 0% |
| ssim | 1.6e-04 | 1.30e-03 (± 3.4e-05) | 0.123 | 1.81e-03 | +0.837 | +0.838 | 0.0363 | -0.01 | 0% |
| wetarea | 2.0e-02 | 1.58e-03 (± 5.0e-05) | 12.7 | 2.19e-03 | -0.736 | -0.737 | 4.67 | -0.06 | 0% |
| opticalflow | 4.0e-05 | 6.04e-03 (± 1.4e-04) | 0.00663 | 8.25e-03 | -0.473 | -0.474 | 0.00797 | -0.03 | 0% |

## Every loss for every model (same batches, eval mode)

| model | mse | mae_phys | minkowski | spectral | ssim | wetarea | opticalflow | own vs vanilla (95% CI) |
|---|---|---|---|---|---|---|---|---|
| vanilla | 0.0002398 | 0.03323 | 1.399 | 0.004103 | 0.4821 | 0.01104 | 0.009384 |  |
| minkowski | 0.0002611 | 0.03469 | 0.4924 | 0.003631 | 0.4822 | 0.01168 | 0.009786 | -64.8% [-65.9%, -63.8%] |
| spectral | 0.0002464 | 0.0334 | 1.512 | 0.004208 | 0.4826 | 0.01127 | 0.009021 | +2.5% [+0.9%, +4.4%] |
| ssim | 0.0002455 | 0.03301 | 1.514 | 0.004233 | 0.4826 | 0.01139 | 0.00879 | +0.1% [+0.0%, +0.2%] |
| wetarea | 0.0002556 | 0.03354 | 1.638 | 0.004178 | 0.4836 | 0.009503 | 0.009032 | -13.9% [-15.8%, -12.1%] |
| opticalflow | 0.0002481 | 0.03353 | 1.526 | 0.004247 | 0.4827 | 0.01125 | 0.009136 | -2.6% [-6.4%, +1.2%] |
| vanilla_34ep | 0.0002444 | 0.03322 | 1.464 | 0.004181 | 0.4824 | 0.0112 | 0.009131 |  |
