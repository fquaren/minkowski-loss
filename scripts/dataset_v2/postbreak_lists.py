#!/usr/bin/env python
"""Post-break lists for the pre-break ablation (DECISIONS §34): every list in <store>/meta restricted to
rows whose hour ends on or after the speckle-floor step (2015-12-12), written as `<list>_postbreak.txt`.
Same 5 columns (timestamp, row, col, max, store_row), so the configs only change the list names.

    python scripts/dataset_v2/postbreak_lists.py [--meta <store>/meta]
"""

import argparse
import glob
import os

BREAK = "20151212"
LISTS = ["full_train", "full_val", "full_test", "light_train", "extremes_train", "light_val", "extremes_val",
         "light_test", "extremes_test", "events_test"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", default="/home/fquareng/work/data/extremes/OPERA/patches_v4_adv/meta")
    a = ap.parse_args()
    rep = []
    for stem in LISTS:
        p = f"{a.meta}/{stem}.txt"
        if not os.path.exists(p):
            continue
        lines = [l for l in open(p).read().split("\n") if l.strip()]
        post = [l for l in lines if l.split(",")[0][:8] >= BREAK]
        open(f"{a.meta}/{stem}_postbreak.txt", "w").write("\n".join(post) + ("\n" if post else ""))
        rep.append(f"{stem}: {len(lines):,} -> {len(post):,} post-break ({len(post) / max(len(lines), 1):.1%})")
    txt = "\n".join(rep)
    print(txt)
    open(f"{a.meta}/postbreak_report.txt", "w").write(f"postbreak_lists.py (DECISIONS §34), break {BREAK}\n{txt}\n")


if __name__ == "__main__":
    main()
