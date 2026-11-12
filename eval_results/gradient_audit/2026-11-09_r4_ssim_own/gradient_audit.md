# Gradient audit

## Gradients at `ssim_full` (latest, epoch 25), 256 x 128 patches, anneal 0.1

MSE: value 9.292e-05, signal ||mu|| 3.810e-04 ± 5.4e-06, per-batch median 4.788e-04 (noise fraction 0.37)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| ssim | 1.6e-04 | 3.93e-04 (± 5.9e-06) | 0.407 | 5.04e-04 | +0.858 | +0.860 | 0.0863 | 0.01 | 0% |

## Gradients at `ssim_cut` (latest, epoch 25), 256 x 128 patches, anneal 0.1

MSE: value 9.435e-05, signal ||mu|| 1.974e-04 ± 1.4e-05, per-batch median 3.419e-04 (noise fraction 0.67)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| ssim | 1.6e-04 | 3.77e-04 (± 2.4e-05) | 0.424 | 6.50e-04 | +0.013 | +0.013 | 0.0457 | 0.00 | 0% |

## Gradients at `vanilla` (latest, epoch 25), 256 x 128 patches, anneal 0.1

MSE: value 9.513e-05, signal ||mu|| 2.847e-04 ± 1.5e-05, per-batch median 3.627e-04 (noise fraction 0.38)

| loss | λ used | parity λ* (± se) | share r = λ/λ* | per-batch λ* | cos | cos corr. | Adam share | noise frac | clip engaged |
|---|---|---|---|---|---|---|---|---|---|
| ssim | 1.6e-04 | 5.32e-03 (± 2.9e-04) | 0.0301 | 5.25e-03 | +0.599 | +0.600 | 0.0383 | 0.39 | 0% |

## Every loss for every model (same batches, eval mode)

| model | mse | mae_phys | minkowski | spectral | ssim | wetarea | opticalflow | own vs vanilla (95% CI) |
|---|---|---|---|---|---|---|---|---|
| vanilla | 9.28e-05 | 0.02505 | nan | nan | 0.01347 | nan | nan |  |
| mse_full | 9.097e-05 | 0.02467 | nan | nan | 0.01322 | nan | nan | -1.8% [-5.1%, +1.5%] |
| mse_cut | 9.232e-05 | 0.02529 | nan | nan | 0.01358 | nan | nan | +0.8% [-1.9%, +3.7%] |
| ssim_full | 8.974e-05 | 0.02499 | nan | nan | 0.01292 | nan | nan | -4.0% [-6.9%, -1.0%] |
| ssim_cut | 8.951e-05 | 0.02468 | nan | nan | 0.01284 | nan | nan | -4.6% [-7.5%, -1.9%] |
