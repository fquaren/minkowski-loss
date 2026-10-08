# Copied 2026-10-08 from the 2026-10-07 calibration run; summarises agg_compare.py (corr) and gauge_agg.py (gauge) outputs.
import glob, sys
import numpy as np
import pandas as pd

A = "/home/fquareng/work/data/extremes/OPERA/quality_v4/calib/daily_check"
EVENTS = {"2014-07-28": "Muenster cloudburst", "2016-05-29": "Braunsbach (Elvira)", "2017-08-11": "Poland derecho",
          "2021-06-24": "Hodonin supercell", "2021-07-14": "Bernd floods", "2022-08-18": "Med derecho",
          "2023-05-16": "Emilia-Romagna", "2024-06-01": "S Germany floods"}
pd.set_option("display.width", 160)

if "tiles" in sys.argv:
    t = pd.concat([pd.read_csv(f) for f in glob.glob(f"{A}/out/tiles_*.csv.gz")], ignore_index=True)
    t["event"] = t.day.isin(EVENTS)
    print("tile rows:", t.groupby("agg").size().to_dict(), "days:", t.day.nunique())
    rows = []
    for agg, g in t.groupby("agg"):
        q99 = g[~g.event]["max"].quantile(0.99)
        for name, s in [("all wet", g), ("top 1% (>= q99 random)", g[g["max"] >= q99])]:
            rows.append(dict(agg=agg, set=name, n=len(s), q99=q99, U=s.U.median(), Ulog=s.Ulog.median(),
                             peak_ratio=s.peak_ratio.median(), A50=s.A50.median(), wet=s.wet.median(),
                             raw_gt_2x=(s.raw_max > 2 * s["max"]).mean(), raw_gt_1p2x=(s.raw_max > 1.2 * s["max"]).mean()))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    # how extreme each event day is, per aggregation: tiles above the random-day q99.9 of tile max
    out = []
    for agg, g in t.groupby("agg"):
        q = g[~g.event]["max"].quantile(0.999)
        per = g.groupby("day").apply(lambda s: pd.Series(dict(n_above=(s["max"] >= q).sum(), daymax=s["max"].max())))
        per["agg"] = agg; per["q999"] = q
        out.append(per.reset_index())
    o = pd.concat(out)
    w = o.pivot(index="day", columns="agg", values=["daymax", "n_above"])
    w["event"] = [EVENTS.get(d, "") for d in w.index]
    print(o.groupby("agg").q999.first().round(2).to_dict())
    print(w.round(1).to_string())

if "corr" in sys.argv:
    c = pd.concat([pd.read_csv(f) for f in glob.glob(f"{A}/out/corr_*.csv.gz")], ignore_index=True)
    m = c.groupby(["what", "lag"])["corr"].median().unstack(0)
    m.index = [f"{15 * l} min" for l in m.index]
    print(m.round(3).to_string())

if "gauge" in sys.argv:
    g = pd.concat([pd.read_csv(f) for f in glob.glob(f"{A}/gout/g_*.csv.gz")], ignore_index=True)
    g["event"] = g.day.isin(["2013-07-28", "2014-06-09", "2014-07-28", "2016-05-29", "2016-06-01", "2021-06-20",
                             "2021-06-28", "2021-07-08", "2021-07-13", "2021-07-14", "2023-07-24", "2024-06-01",
                             "2024-06-02", "2024-06-21", "2024-06-29"])
    print("days:", g.day.nunique(), "rows:", g.groupby("agg").size().to_dict())
    rows = []
    for agg, s in g.groupby("agg"):
        wet = s[(s.gauge >= 0.1) | (s.radar >= 0.1)]
        lr, lg = np.log1p(wet.radar), np.log1p(wet.gauge)
        q = s.gauge[s.gauge > 0].quantile(0.999)
        top = s[s.gauge >= q]
        ratio = (top.radar / top.gauge)
        err = np.log((wet.radar + 0.1) / (wet.gauge + 0.1))
        rows.append(dict(agg=agg, n_wet=len(wet), pearson_log=np.corrcoef(lr, lg)[0, 1],
                         spearman=wet.radar.rank().corr(wet.gauge.rank()), log_err_sd=err.std(),
                         gauge_q999=q, n_top=len(top), top_ratio_med=ratio.median(),
                         top_ratio_q10=ratio.quantile(0.1), top_ratio_q90=ratio.quantile(0.9)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))
