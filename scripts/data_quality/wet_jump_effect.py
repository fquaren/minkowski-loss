#!/usr/bin/env python
"""Effect of min_wet_jump = q99.9 of normal operation (DECISIONS §30, item 4) before adopting it:
radar-frame flags recomputed from the stored features (all days), then gate criteria 1 and 3 on
the rescanned calibration days with rule 5 = score only. Writes calib/wet_jump/frame_flags_<v>.csv.gz.

    python scripts/data_quality/wet_jump_effect.py 0.021
"""

import copy
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
from src.data.radar_screen import decide_hourly, frame_flagged_temporal  # noqa: E402
import gate_prebuild as gp  # noqa: E402

Q4 = "/home/fquareng/work/data/extremes/OPERA/quality_v4"


def flags_day(args):
    f, q = args
    z = np.load(f)
    fl = frame_flagged_temporal(z["feats"], q)
    ti, ri = np.nonzero(fl)
    return pd.DataFrame({"radar": z["keys"].astype(str)[ri], "timestamp": z["times"].astype(str)[ti]})


def main():
    v = float(sys.argv[1])
    q = yaml.safe_load(open(f"{REPO}/configs/quality_v4.yaml"))
    q2 = copy.deepcopy(q)
    q2["radar_frame"]["temporal"]["min_wet_jump"] = v
    files = sorted(glob.glob(f"{Q4}/radars/*.npz"))
    with ProcessPoolExecutor(2) as ex:
        ff = pd.concat(list(ex.map(flags_day, [(f, q2) for f in files], chunksize=16)), ignore_index=True)
    os.makedirs(f"{Q4}/calib/wet_jump", exist_ok=True)
    ff.to_csv(f"{Q4}/calib/wet_jump/frame_flags_{v}.csv.gz", index=False)
    old = pd.read_csv(f"{Q4}/frame_flags.csv.gz", dtype=str)
    L = [f"min_wet_jump {v}: {len(ff):,} flagged radar-frames on "
         f"{ff.assign(d=ff.timestamp.str[:8])[['radar', 'd']].drop_duplicates().shape[0]:,} radar-days "
         f"(0.15: {len(old):,})"]
    t, _ = gp.load_hours()
    t["n_burst"] = t.get("n_burst", 0)
    cal = pd.read_csv(f"{gp.CAL}/days.csv", dtype=str)
    de = pd.read_csv(f"{Q4}/day_exclusions.csv", dtype=str)
    de = set(zip(de.radar, de.day))
    for lab, flags in (("0.15", set(zip(old.radar, old.timestamp))), (str(v), set(zip(ff.radar, ff.timestamp)))):
        d = decide_hourly(t, q, frame_flags=flags, day_excl=de)
        rej = t["unphysical"].astype(bool).values | d["rej_v4"].values
        c1 = gp.crit1(t, rej, cal)
        a, b, _, _ = gp.crit3(t, d, rej)
        hit = [(e, n, f"{p:.2%}") for e, n, p in a if p > 0]
        rf = [x for x in b if x[0] == "rej_radar_frame"][0]
        L.append(f"[{lab}] rejected {rej.mean():.2%}; known failures kept {[k for *_, k in c1]}; "
                 f"radar-frame rejection 5-10 mm {rf[1]:.2%} / >= 20 mm {rf[2]:.2%}; events with losses {hit}")
    txt = "\n".join(L)
    print(txt)
    open(f"{Q4}/calib/wet_jump/effect_{v}.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
