# Gradient audit

## Gradients at `vanilla_v4` (best, epoch 24), 512 x 128 patches, anneal 0.1

MSE: value 9.563e-05, signal ||mu|| 3.193e-04 ± 9.1e-06, per-batch median 3.802e-04 (noise fraction 0.29)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.44e-05 (± 5.0e-07) | 6.97 | 1.58e-05 | -0.138 | -0.139 | 10.2 | 0.15 | 0% |

