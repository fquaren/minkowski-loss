# Gradient audit

## Gradients at `vanilla_v4` (best, epoch 24), 512 x 128 patches, anneal 0.1

MSE: value 9.563e-05, signal ||mu|| 3.193e-04 ± 9.1e-06, per-batch median 3.802e-04 (noise fraction 0.29)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| spectral | 5.0e-04 | 7.67e-02 (± 2.2e-03) | 0.00652 | 6.48e-02 | +0.063 | +0.063 | 0.0181 | 0.47 | 0% |
| ssim | 1.6e-04 | 1.30e-03 (± 3.6e-05) | 0.123 | 1.53e-03 | +0.372 | +0.372 | 0.0517 | -0.01 | 0% |
| wetarea | 2.0e-02 | 1.71e-03 (± 4.5e-05) | 11.7 | 2.02e-03 | +0.405 | +0.405 | 6.6 | -0.00 | 0% |
| opticalflow | 4.0e-05 | 3.93e-03 (± 1.2e-04) | 0.0102 | 4.74e-03 | +0.152 | +0.152 | 0.0148 | -0.01 | 0% |

