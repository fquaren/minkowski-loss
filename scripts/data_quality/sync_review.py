#!/usr/bin/env python
"""Copy the decisions made on the review page (artifact "OPERA screen review") into
`quality_v4/review.csv` and `quality_v4/calib/rule_gallery/index.csv`.

The page stores one JSON document per decided case in two collections, `review/rNNN`
({decision, note, rank, radar, day}) and `gallery/gNNN` ({verdict, note, provisional, rule, n}).
Export them with Claude's ArtifactData tool (`list`, `out_dir`), which writes
<export>/review/rNNN.json and <export>/gallery/gNNN.json, then run

    python scripts/data_quality/sync_review.py --export <export dir>

Each CSV is backed up once per run (<name>.bak_<timestamp>). Rows the page has not decided are
left as they are. In index.csv a row the researcher decided gets reviewer = researcher, also
when the verdict agrees with the provisional one.
"""

import argparse
import glob
import json
import os
import shutil
import time

import pandas as pd

Q = "/home/fquareng/work/data/extremes/OPERA/quality_v4"


def load(export, coll):
    out = {}
    for f in glob.glob(os.path.join(export, coll, "*.json")):
        d = json.load(open(f))
        d = d.get("data", d)                       # tolerate {id, version, data} wrappers
        out[os.path.basename(f)[:-5]] = d
    return out


def backup(path):
    b = f"{path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(path, b)
    return b


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    rv = load(a.export, "review")
    p = os.path.join(Q, "review.csv")
    r = pd.read_csv(p, dtype=str, keep_default_na=False)
    n = 0
    for i, row in r.iterrows():
        d = rv.get("r%03d" % int(row["rank"]))
        if d and d.get("decision"):
            assert str(d.get("rank", row["rank"])) == str(int(row["rank"])) and d.get("radar", row["radar"]) == row["radar"], row
            r.at[i, "decision"], r.at[i, "note"] = d["decision"], d.get("note", "")
            n += 1
    print(f"review.csv: {n} of {len(r)} radar-days decided "
          f"({(r.decision == 'exclude').sum()} exclude, {(r.decision == 'keep').sum()} keep, "
          f"{(r.decision == 'unsure').sum()} unsure)")

    gv = load(a.export, "gallery")
    pg = os.path.join(Q, "calib", "rule_gallery", "index.csv")
    g = pd.read_csv(pg, dtype=str, keep_default_na=False)
    m = changed = 0
    for i, row in g.iterrows():
        d = gv.get("g%03d" % (i + 1))
        if d and d.get("verdict"):
            assert d.get("rule", row["rule"]) == row["rule"] and str(d.get("n", row["n"])) == str(row["n"]), (i, row["rule"])
            changed += d["verdict"] != row["verdict"]
            g.at[i, "verdict"], g.at[i, "note"], g.at[i, "reviewer"] = d["verdict"], d.get("note") or row["note"], "researcher"
            m += 1
    print(f"index.csv: {m} of {len(g)} examples decided by the researcher, {changed} changed from the provisional call")
    if a.dry_run:
        return
    for path, df in ((p, r), (pg, g)):
        print("backup", backup(path))
        df.to_csv(path, index=False)


if __name__ == "__main__":
    main()
