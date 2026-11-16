"""Quantile mapping for the Baselines v2 QM rows (src/evaluation/quantile_map.py).

1. sample: a random subset of a training list (5-column, store rows kept) to fit on.
       python scripts/evaluate/quantile_map.py sample <train list> <out list> [--n 16384]
2. histograms: run the model on that subset with eval_extremes.py --split train
   --qm_hist <out.npz> (a config whose TRAIN_METADATA_FILE is the subset; --ensemble 1 for FM).
3. fit: the quantile table eval_extremes.py --qm reads.
       python scripts/evaluate/quantile_map.py fit <hist.npz> <out.json>
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.evaluation.quantile_map import LogHistogram, fit_table  # noqa: E402


def sample(args):
    lines = [l for l in open(args.train_list) if l.strip()]
    rows = []
    for i, l in enumerate(lines):
        p = l.strip().split(",")
        rows.append(l if len(p) == 5 else l.strip() + f",{i}\n")   # keep the store row explicit
    idx = np.sort(np.random.default_rng(args.seed).choice(len(rows), min(args.n, len(rows)), replace=False))
    with open(args.out, "w") as f:
        f.writelines(rows[i] for i in idx)
    print(f"wrote {len(idx)} of {len(rows)} rows to {args.out}")


def fit(args):
    d = np.load(args.hist)
    hp, ho = LogHistogram(), LogHistogram()
    hp.counts, hp.n_zero = d["pred_counts"], int(d["pred_zero"])
    ho.counts, ho.n_zero = d["obs_counts"], int(d["obs_zero"])
    t = fit_table(hp, ho)
    t["source"] = {k: str(d[k]) for k in ("checkpoint", "list") if k in d}
    json.dump(t, open(args.out, "w"), indent=1)
    lv = np.array(t["levels"])
    for p in (0.5, 0.9, 0.99, 0.999, 1 - 1e-4, 1 - 1e-5, 1 - 1e-6):
        i = int(np.argmin(abs(lv - p)))
        print(f"  p={lv[i]:.6f}  pred {t['q_pred'][i]:8.3f}  obs {t['q_obs'][i]:8.3f} mm")
    print(f"wrote {args.out} ({t['n_pixels']} pixels)")


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
a = sp.add_parser("sample"); a.add_argument("train_list"); a.add_argument("out")
a.add_argument("--n", type=int, default=16384); a.add_argument("--seed", type=int, default=0)
b = sp.add_parser("fit"); b.add_argument("hist"); b.add_argument("out")
args = ap.parse_args()
{"sample": sample, "fit": fit}[args.cmd](args)
