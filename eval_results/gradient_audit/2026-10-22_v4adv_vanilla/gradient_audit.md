# Gradient audit

## Gradients at `vanilla_v4` (best, epoch 24), 512 x 128 patches, anneal 0.1

MSE: value 9.563e-05, signal ||mu|| 3.192e-04 ± 9.1e-06, per-batch median 3.801e-04 (noise fraction 0.29)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.66e-05 (± 5.8e-07) | 6.02 | 1.84e-05 | -0.140 | -0.140 | 8.83 | 0.15 | 0% |

