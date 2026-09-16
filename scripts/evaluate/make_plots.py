#!/usr/bin/env python
"""Generate the comparison figures from evaluation output directories.

Reads the ``*_summary.yaml`` / ``*_arrays.npz`` that the eval scripts already write, so this
never re-runs a model. Point it at as many eval directories as you want compared.

  # explicit, ordered, with labels
  python scripts/evaluate/make_plots.py --out figures/study1 \
      --model "MSE:eval_results/extremes/backbone_vanilla" \
      --model "Minkowski:eval_results/extremes/backbone_minkowski" \
      --model "spectral:eval_results/extremes/backbone_spectral_v2"

  # or sweep a directory of eval outputs (label = directory name)
  python scripts/evaluate/make_plots.py --out figures/all \
      --glob "eval_results/extremes/backbone_*"

  # one qualitative field panel from a sample npz written during training
  python scripts/evaluate/make_plots.py --out figures --fields runs/.../recon_epoch_010.npz

Figures are only produced when the artefacts support them: the Minkowski curves need
``gamma_hat`` and ``gamma_target`` in the arrays, which ``eval_backbone.py`` writes. If a
figure is skipped the reason is printed.
"""

import argparse
import glob as globmod
import os
import sys

import numpy as np

from src.evaluation import plotting as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help='"label:path/to/eval_dir", repeatable; order sets plot order')
    ap.add_argument("--glob", default=None,
                    help="glob of eval directories; label taken from the directory name")
    ap.add_argument("--out", default="figures", help="output directory for the figures")
    ap.add_argument("--pixel_km", type=float, default=2.0)
    ap.add_argument("--fields", default=None,
                    help="npz with 2-D fields (e.g. a training reconstruction dump) "
                         "to render as a qualitative panel")
    ap.add_argument("--drizzle", type=float, default=0.1)
    args = ap.parse_args()

    specs = []
    for entry in args.model:
        if ":" not in entry:
            sys.exit(f"--model expects 'label:path', got {entry!r}")
        label, path = entry.split(":", 1)
        specs.append((label.strip(), path.strip()))
    if args.glob:
        for d in sorted(globmod.glob(args.glob)):
            if os.path.isdir(d):
                label = os.path.basename(d.rstrip("/")).replace("backbone_", "")
                specs.append((label, d))

    models = []
    for label, path in specs:
        m = P.load_model(label, path)
        if m is None:
            print(f"[skip] {label}: no summary file in {path}")
            continue
        models.append(m)
        has = "summary" + (" + arrays" if m.arrays is not None else " only")
        print(f"[load] {label:<20s} {path}  ({has})")

    if models:
        P.make_all(models, args.out, pixel_km=args.pixel_km)
    elif not args.fields:
        sys.exit("nothing to plot: no eval directories resolved")

    if args.fields:
        d = np.load(args.fields, allow_pickle=False)
        # accept the common key spellings used by the training dumps
        wanted = [("input", ("lr", "input", "x", "coarse")),
                  ("prediction", ("pred", "prediction", "sr", "generated", "y_pred")),
                  ("target", ("target", "hr", "y", "truth"))]
        fields = {}
        for title, keys in wanted:
            for k in keys:
                if k in d:
                    a = np.asarray(d[k])
                    while a.ndim > 2:
                        a = a[0]
                    fields[title] = a
                    break
        if fields:
            P.field_panel(fields, args.out, drizzle=args.drizzle,
                          suptitle=os.path.basename(args.fields))
        else:
            print(f"[skip] fields: none of the expected keys in {args.fields}; "
                  f"found {list(d.keys())}")


if __name__ == "__main__":
    main()
