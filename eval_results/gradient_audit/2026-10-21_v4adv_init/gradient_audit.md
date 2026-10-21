# Gradient audit

## Gradients at `init` (init, epoch 0), 512 x 128 patches, anneal 0.1

MSE: value 1.747e-03, signal ||mu|| 6.434e-01 ± 3.8e-03, per-batch median 6.398e-01 (noise fraction -0.01)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 8.90e-04 (± 3.7e-06) | 0.112 | 8.71e-04 | +0.862 | +0.862 | nan | 0.07 | 0% |

