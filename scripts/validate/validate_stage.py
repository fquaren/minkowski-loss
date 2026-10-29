#!/usr/bin/env python
"""Validate one pipeline stage before the next starts. Writes validation/<stage>/<date>_<commit>.md
with every check, its status and its numbers, and exits non-zero if any check fails, so a chain
script can stop on it.

Stages implemented: store, gamma, training. Planned: raw, scan, splits, evaluation
(EXPERIMENTS, "Validation").

    python scripts/validate/validate_stage.py store --config configs/config_v4.yaml --n_days 8
    python scripts/validate/validate_stage.py gamma --config configs/config_v4.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import REPO, write_report  # noqa: E402

sys.path.insert(0, REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["store", "gamma", "training"])
    ap.add_argument("--config", default="configs/config_v4.yaml")
    ap.add_argument("--quality", default="configs/quality_v4.yaml")
    ap.add_argument("--n_sample", type=int, default=2000, help="rows sampled per split")
    ap.add_argument("--n_days", type=int, default=8, help="store: days recomputed from raw (0 = skip)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out_root", default=None, help="report directory (default <repo>/validation)")
    ap.add_argument("--batch", type=int, default=8, help="training: samples in the test batch")
    ap.add_argument("--steps", type=int, default=150, help="training: overfit steps")
    ap.add_argument("--device", default=None, help="training: cuda / cpu (default: cuda if available)")
    ap.add_argument("--log", default=None, help="training: a run log to check (loss name, weight, no NaN)")
    a = ap.parse_args()
    from src.utils import load_config
    cfg = load_config(a.config)
    import importlib
    checks = importlib.import_module(a.stage).run(cfg, a)
    path, passed = write_report(a.stage, checks, {"config": a.config, "store": cfg["PREPROCESSED_DATA_DIR"]},
                                a.out_root)
    for c in checks:
        print(f"[{c.status:4}] {c.name}: {c.detail} {c.numbers}")
    print(f"\n{'PASS' if passed else 'FAIL'} -> {path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
