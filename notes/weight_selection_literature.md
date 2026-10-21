# Selecting the structural-loss weight: literature status (2026-10-21)

Tracks whether the weight-selection procedure of DECISIONS §37 is established practice (cite it) or
new (a possible contribution). **Status: preliminary.** Based on a targeted web search on 2026-10-21,
not a systematic review. Nothing below may be claimed as novel before the systematic search in
EXPERIMENTS §5 (Paper) is done.

## The procedure

1. **Calibration by gradient share.** Carry the weight validated on 15-min data (1e-4) to the hourly
   data by matching the structural term's share of the gradient (raw at initialisation; Adam-
   preconditioned at the trained vanilla checkpoint). Tool: `tools/gradient_audit.py`; maths in
   `notes/gradient_audit.pdf`. The same tool already set the competing losses' weights by Adam-share
   matching to Minkowski (2026-09-23, `src/losses/competing.py`).
2. **Bracketing sweep.** The calibrated bracket of DECISIONS §16 / RESEARCH_NOTES §5:
   {λ*/3, λ*, 3λ*, 10λ*}, one fixed epoch budget, last checkpoint; the point chosen is the best
   exceedance ratio with ΔMAE <= +5% and anisotropy <= 1.2x vanilla (the anisotropy cap is the
   guard against loss gaming), then 3 seeds there. Activity floor: ΔMAE >= +1% (CLAUDE.md).

## What is established (cite)

| idea | reference | relation to ours |
|---|---|---|
| Balance loss terms by gradient magnitude, adaptively | Chen, Badrinarayanan, Lee & Rabinovich, *GradNorm*, ICML 2018 ([PMLR](https://proceedings.mlr.press/v80/chen18a.html)) | dynamic, multitask; we use the ratio once, to set a fixed weight |
| Set loss-term weights from gradient statistics | Wang, Teng & Perdikaris, *Understanding and mitigating gradient pathologies in physics-informed neural networks*, arXiv 2001.04536 ([arXiv](https://www.arxiv.org/abs/2001.04536)) | same principle, physics-informed losses |
| Auxiliary weight from a target gradient-norm ratio, w = G_main / (G_aux R) | Zhong & Maki, *Regularizing CNN Transfer Learning with Randomised Regression*, CVPR 2020 ([arXiv](https://arxiv.org/pdf/1908.05997)) | **closest**; on-line, one dataset; our target ratio comes from another dataset |
| Auxiliary-loss weight from gradient similarity | Du et al., *Adapting Auxiliary Losses Using Gradient Similarity* ([PDF](http://www.gatsby.ucl.ac.uk/~balaji/CL-NeurIPS2018-adapt.pdf)) | related, cosine-based |
| Matched effective weights hold only at initialisation (ratio 150 -> 1.6 in training) | sign-language noise-robust losses, arXiv 2609.12885 ([arXiv](https://arxiv.org/pdf/2609.12885)) — **found by search, not read** | supports measuring at a trained checkpoint (step 1b) |
| Accuracy vs perceptual structure is a trade-off curve | Blau & Michaeli, *The Perception-Distortion Tradeoff*, CVPR 2018 ([CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.html)) | motivates the sweep as tracing a trade-off |
| Over-optimising a proxy objective (Goodhart) | Gao, Schulman & Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023 ([PMLR](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf)); *Goodhart's Law in Reinforcement Learning*, ICLR 2024 ([PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/6ad68a54eaa8f9bf6ac698b02ec05048-Paper-Conference.pdf)); *Catastrophic Goodhart*, NeurIPS 2024 ([PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/1a8189929f3d7bd6183718f42c3f4309-Paper-Conference.pdf)) | no closed-form safe weight; validate on a held-out true objective. RL rewards: an analogy, not a precedent |
| Hyperparameter sweeps | Bergstra & Bengio, *Random search for hyper-parameter optimization*, JMLR 2012 — from memory, not re-checked | standard practice |

## What was not found (candidate contributions, pending the systematic search)

- **Transfer of a fixed weight across datasets / temporal aggregations by gradient share** (15-min ->
  hourly). Probably a modest methodological point rather than a headline.
- **Hacking-bounded weight selection for a structural loss in precipitation SR**: the §16 rule
  (best tail with an accuracy cap and an anisotropy cap that guards against loss gaming), judged
  on metrics the loss does not optimise, with the w = 1e-3 episode as evidence. In precipitation SR,
  weights are tuned without such a bound (e.g. MSRGAN, [doi](https://doi.org/10.3390/rs17132281)) or
  fixed; a review of an EGUsphere preprint asks for exactly such a sensitivity test
  ([EGUsphere](https://egusphere.copernicus.org/preprints/2026/egusphere-2026-861/)). Most defensible claim.

## Kept for a follow-up

Weight set every step from the gradient share (scale-free; one rule for all competing losses).
Closer to GradNorm / Zhong & Maki: easier to motivate, less new. EXPERIMENTS "To explore".

## For the systematic search (to do)

Google Scholar / Scopus / Web of Science, 2015-2026: "loss weight" OR "auxiliary loss" AND
("gradient norm" OR "gradient ratio" OR "loss balancing") AND (transfer OR "across datasets");
"structural loss" OR "topological loss" OR "Minkowski" AND (precipitation OR radar) AND
(downscaling OR "super-resolution"); "reward hacking" OR "Goodhart" AND "auxiliary loss" AND
regression; plus forward citations of GradNorm, Zhong & Maki 2020 and Wang et al. 2020.
Read the sign-language paper (2609.12885) and the "When Losses Align" preprint (2605.07756) in full.
