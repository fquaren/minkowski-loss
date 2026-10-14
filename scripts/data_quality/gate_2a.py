#!/usr/bin/env python
"""Re-run §29 criterion 2a (false alarms against RADKLIM on the same DWD gauge-hours) on the
gauge outputs of a given chain (`gauge_hourly.py run --out <dir> --variants none,full`).

    python scripts/data_quality/gate_2a.py <gauge_dir> <out.md>
"""

import os
import sys

import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
import gate_prebuild as gp  # noqa: E402


def main():
    gp.GAUGE = sys.argv[1]
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    ff = pd.read_csv(f"{gp.Q4}/frame_flags.csv.gz", dtype=str)
    L = [f"# Criterion 2a on `{gp.GAUGE}`", ""] + gp.crit2_floor(q, set(zip(ff.radar, ff.timestamp)))
    txt = "\n".join(L)
    print(txt)
    open(sys.argv[2], "w").write(txt + "\n")


if __name__ == "__main__":
    main()
