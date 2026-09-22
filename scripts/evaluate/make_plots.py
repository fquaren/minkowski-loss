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
                    help='"label:path/to/eval_dir", repeatable; order sets plot order. '
                         "Several directories may be joined with '+' to merge into one "
                         "row, first-wins, e.g. 'MSE:.../extremes/backbone_vanilla"
                         "+.../backbone/vanilla' -- the backbone directory is the only "
                         "source of gamma_hat, which the Minkowski figures need.")
    ap.add_argument("--glob", default=None,
                    help="glob of eval directories; label taken from the directory name")
    ap.add_argument("--out", default="figures", help="output directory for the figures")
    ap.add_argument("--pixel_km", type=float, default=2.0)
    ap.add_argument("--fields", default=None,
                    help="npz with 2-D fields (e.g. a training reconstruction dump) "
                         "to render as a qualitative panel")
    ap.add_argument("--drizzle", type=float, default=0.1)
    ap.add_argument("--field_bundle", default=None,
                    help="npz from scripts/evaluate/dump_fields.py: renders the "
                         "DEM / input / target / prediction field figures")
    ap.add_argument("--field_mode", default="both",
                    choices=["compare", "detail", "both"],
                    help="compare = one row per patch across models; detail = the "
                         "four fields over the Minkowski curves, per model and patch")
    ap.add_argument("--field_norm", default="power",
                    choices=["power", "log", "linear"],
                    help="colour stretch shared by the precipitation panels; the default "
                         "square-root stretch keeps amplitudes comparable while leaving the "
                         "drizzle-to-moderate range readable")
    ap.add_argument("--cloud_points", type=int, default=2000,
                    help="patches drawn per model in the perception-distortion cloud")
    args = ap.parse_args()

    specs = []
    for entry in args.model:
        if ":" not in entry:
            sys.exit(f"--model expects 'label:path', got {entry!r}")
        label, path = entry.split(":", 1)
        paths = [p for p in (q.strip() for q in path.split("+")) if p]
        specs.append((label.strip(), paths))
    if args.glob:
        for d in sorted(globmod.glob(args.glob)):
            if os.path.isdir(d):
                label = os.path.basename(d.rstrip("/")).replace("backbone_", "")
                specs.append((label, [d]))

    models = []
    for label, paths in specs:
        m = P.load_model(label, paths)
        shown = " + ".join(paths)
        if m is None:
            print(f"[skip] {label}: no summary file in {shown}")
            continue
        models.append(m)
        has = "summary" + (" + arrays" if m.arrays is not None else " only")
        if m.arr("gamma_hat") is not None:
            has += " + gamma"
        print(f"[load] {label:<20s} {shown}  ({has})")

    if models:
        P.make_all(models, args.out, pixel_km=args.pixel_km,
                   cloud_points=args.cloud_points)
    elif not (args.fields or args.field_bundle):
        sys.exit("nothing to plot: no eval directories resolved")

    if args.field_bundle:
        bundle = P.load_field_bundle(args.field_bundle)
        n_models, n_patches = bundle["preds"].shape[0], bundle["target"].shape[0]
        print(f"[fields] {args.field_bundle}: {n_models} models x {n_patches} patches")
        if args.field_mode in ("compare", "both"):
            P.field_comparison(bundle, args.out, drizzle=args.drizzle,
                               norm_mode=args.field_norm)
        if args.field_mode in ("detail", "both"):
            P.field_detail(bundle, args.out, drizzle=args.drizzle,
                           norm_mode=args.field_norm)

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
