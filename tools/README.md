# tools/

Diagnostics that are not part of the training or evaluation pipeline.

| Tool | What it measures |
|---|---|
| `gradient_audit.py` | How much each auxiliary loss steers training. For each checkpoint and loss it gives the gradient share, the parity weight, the noise-corrected cosine, the Adam-preconditioned share and the clipping frequency. It also evaluates every model on every loss, paired against vanilla. The math is in `notes/gradient_audit.pdf`. |
| `gradient_audit_v0/` | The first, quick version (6 batches, per-batch norms), which produced the numbers in EXPERIMENTS.md §3 on 2026-09-23. It is kept for provenance. Its parity values are noise-referenced at the converged vanilla checkpoint (see the notes, §5). Prefer `gradient_audit.py`. `gn.py` = gradient norms, `gn2.py` = loss values per checkpoint + cosines, `ss.py` = SSIM / spectral floors. |

Both run on GPU 1 only, within cores 4–11 (`import src` enforces it). Keep DataLoader
workers ≤ 4.

```bash
python tools/gradient_audit.py config.yaml --n_batches 512 \
    --grad_at init vanilla minkowski spectral ssim wetarea opticalflow \
    --out_dir eval_results/gradient_audit/<date>
# v0, from the repo root:
python tools/gradient_audit_v0/gn.py 6
```
