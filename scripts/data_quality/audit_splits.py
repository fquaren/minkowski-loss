#!/usr/bin/env python
"""How independent are train / val / test? Temporal and spatial leakage in the splits.

`split_metadata.py` shuffles *patches*, so the same 15-minute composite — and the same
storm, a tile away or a frame later — can sit in train and test at once. This measures how
much, overall and for the tail (patch max >= u), which bounds how much any test number can
say about generalisation, and is the baseline an o.o.d. split has to improve on.

For every test (or val) patch, is there a train patch
    same_time        at the same timestamp (anywhere in the domain)
    same_tile_1h     on the same tile within +-1 h (temporal neighbour of the same storm)
    adjacent_same_t  on one of the 8 neighbouring tiles at the same timestamp
and what are the date ranges / month counts of each split.

    python scripts/data_quality/audit_splits.py config.yaml --out_dir <dir>
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils import load_config  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _md import md_table  # noqa: E402


def read_meta(path):
    df = pd.read_csv(path, header=None, names=["timestamp", "row", "col", "pmax"],
                     dtype={"timestamp": str})
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H%M%S")
    return df


def leakage(ref: pd.DataFrame, qry: pd.DataFrame, patch: int) -> pd.DataFrame:
    ref_t = set(ref["timestamp"])
    ref_key = set(zip(ref["timestamp"], ref["row"], ref["col"]))
    # tile -> sorted train times, for the +-1 h test
    ref_by_tile = {k: np.sort(g["time"].values.astype("datetime64[m]").astype(np.int64))
                   for k, g in ref.groupby(["row", "col"])}
    out = pd.DataFrame(index=qry.index)
    out["same_time"] = qry["timestamp"].isin(ref_t)

    t_min = qry["time"].values.astype("datetime64[m]").astype(np.int64)
    near = np.zeros(len(qry), bool)
    for (r, c), idx in qry.groupby(["row", "col"]).indices.items():
        ts = ref_by_tile.get((r, c))
        if ts is None or not len(ts):
            continue
        q = t_min[idx]
        pos = np.searchsorted(ts, q)
        lo = ts[np.clip(pos - 1, 0, len(ts) - 1)]
        hi = ts[np.clip(pos, 0, len(ts) - 1)]
        near[idx] = (np.minimum(np.abs(q - lo), np.abs(hi - q)) <= 60)
    out["same_tile_1h"] = near

    adj = np.zeros(len(qry), bool)
    for dy in (-patch, 0, patch):
        for dx in (-patch, 0, patch):
            if dy == dx == 0:
                continue
            keys = zip(qry["timestamp"], qry["row"] + dy, qry["col"] + dx)
            adj |= np.fromiter((k in ref_key for k in keys), bool, len(qry))
    out["adjacent_same_t"] = adj
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tail", type=float, nargs="+", default=[31.0, 89.0])
    args = ap.parse_args()
    cfg = load_config(args.config)
    patch = cfg["PATCH_SIZE"]
    splits = {s: read_meta(cfg[f"{k}_METADATA_FILE"]) for s, k in
              (("train", "TRAIN"), ("val", "VAL"), ("test", "TEST"))}

    lines = ["# Split audit", ""]
    lines += ["| split | patches | first | last | tiles | timestamps |", "|---|---|---|---|---|---|"]
    for s, df in splits.items():
        lines.append(f"| {s} | {len(df):,} | {df.time.min():%Y-%m-%d} | {df.time.max():%Y-%m-%d} "
                     f"| {df.groupby(['row', 'col']).ngroups} | {df.timestamp.nunique():,} |")
    months = pd.DataFrame({s: df.time.dt.to_period("M").value_counts().sort_index()
                           for s, df in splits.items()})
    lines += ["", "## Patches per month", "", md_table(months.rename_axis("month")), ""]

    lines += ["## Leakage from train", "",
              "Share of query patches with a train patch at the same timestamp, on the same "
              "tile within +-1 h, or on an adjacent tile at the same timestamp.", "",
              "| query | subset | n | same_time | same_tile_1h | adjacent_same_t |",
              "|---|---|---|---|---|---|"]
    for q in ("val", "test"):
        lk = leakage(splits["train"], splits[q], patch)
        for name, mask in [("all", np.ones(len(lk), bool))] + [
                (f"pmax >= {u:g}", splits[q]["pmax"].values >= u) for u in args.tail]:
            sub = lk[mask]
            if not len(sub):
                continue
            lines.append(f"| {q} | {name} | {len(sub):,} | {sub.same_time.mean():.1%} | "
                         f"{sub.same_tile_1h.mean():.1%} | {sub.adjacent_same_t.mean():.1%} |")
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "split_audit.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
