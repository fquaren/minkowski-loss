# Gradient audit

## Gradients at `init` (init, epoch 0), 512 x 128 patches, anneal 0.1

MSE: value 1.735e-03, signal ||mu|| 6.380e-01 ± 3.6e-03, per-batch median 6.405e-01 (noise fraction 0.01)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| minkowski | 1.0e-04 | 1.17e-03 (± 8.4e-06) | 0.0853 | 1.14e-03 | +0.819 | +0.819 | nan | 0.06 | 0% |

