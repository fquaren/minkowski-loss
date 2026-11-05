#!/usr/bin/env python
"""
Trend test (DECISIONS §43, pre-registered 2026-11-05, commit 4c501b2): do held-out extremes need re-weighting of seen
shapes (W2) or extrapolation of the in-range drift (W3)? Three measurements on the 21 peak-normalised shape features
(`src.evaluation.analogue.shape_features`), a known-answer validity gate, a backtest at a pseudo-cut, and a tile-edge
control. The rules below are the pre-registered ones; do not change them after seeing results.

    M1 coverage   r = d1 / s (nearest-reference distance over local reference spacing), target vs last in-range bin
    M2 drift      bin means of the reference: size D and min cosine between consecutive shifts
    M3 ED         energy distance of reuse (last bin) and trend (last bin + beta * delta) predictors to the target

Usage: python scripts/evaluate/shape_trend_test.py --out <dir> [--quick]   (CPU only, one core)
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import zarr
from sklearn.neighbors import KDTree

from src.evaluation.analogue import shape_features

V4 = "/home/fquareng/work/data/extremes/OPERA/patches_v4_adv"
AN = "/home/fquareng/work/data/extremes/OPERA/quality_v4/analogue"
NB, K, NBOOT, NPRED, NTGT = 4, 10, 1000, 5000, 4000


# ---------------------------------------------------------------- data
def meta(split):
    m = pd.read_csv(f"{V4}/meta/full_{split}.txt", header=None, names=["t", "y", "x", "mx", "r"])
    m["day"] = m.t.astype(str).str[:8]
    return m


def edge_flag(field, f=0.1):
    """Does {x >= f * peak} touch the tile border?"""
    field = np.nan_to_num(field)
    pk = field.max()
    if pk <= 0:
        return False
    m = field >= f * pk
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())


def read(split, rows, cache, want_features=True):
    """Features (if wanted) and edge flags for store rows, cached by row list."""
    if os.path.exists(cache):
        c = np.load(cache)
        if np.array_equal(c["rows"], rows):
            return (c["X"] if want_features else None), c["edge"]
    z = zarr.open(os.path.join(V4, "preprocessed_dataset.zarr"), mode="r")[{"train": "train", "test": "test"}[split]]
    a = z["original_precip"]
    X, E = [], []
    for r in rows:
        fld = a[int(r)]
        if want_features:
            X.append(shape_features(fld))
        E.append(edge_flag(fld))
    X = np.stack(X) if want_features else np.zeros((len(rows), 0), np.float32)
    E = np.array(E)
    np.savez_compressed(cache, X=X, edge=E, rows=rows)
    return (X if want_features else None), E


# ---------------------------------------------------------------- bootstrap helpers
def day_boot(days, rng):
    """Yield index arrays of a day-block bootstrap (NBOOT resamples)."""
    ud, inv = np.unique(days, return_inverse=True)
    groups = [np.flatnonzero(inv == k) for k in range(len(ud))]
    for _ in range(NBOOT):
        pick = rng.integers(0, len(ud), len(ud))
        yield np.concatenate([groups[k] for k in pick])


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# ---------------------------------------------------------------- M1 coverage
class Coverage:
    def __init__(self, R):
        self.R, self.tree, self.nn = R, KDTree(R), {}

    def r(self, Q):
        d, idx = self.tree.query(Q, k=K)
        need = np.setdiff1d(np.unique(idx), np.fromiter(self.nn.keys(), int, len(self.nn)))
        if len(need):
            dd, _ = self.tree.query(self.R[need], k=2)
            self.nn.update(zip(need.tolist(), dd[:, 1].tolist()))
        s = np.median(np.vectorize(self.nn.get)(idx), axis=1)
        return d[:, 0] / np.maximum(s, 1e-9)


def m1(cov, Qt, dt, Qc, dc, rng):
    rt, rc = cov.r(Qt), cov.r(Qc)
    med = np.median(rt) / np.median(rc)
    q90 = np.quantile(rt, 0.9) / np.quantile(rc, 0.9)
    bt, bc = list(day_boot(dt, rng)), list(day_boot(dc, rng))
    bmed = [np.median(rt[i]) / np.median(rc[j]) for i, j in zip(bt, bc)]
    bq90 = [np.quantile(rt[i], 0.9) / np.quantile(rc[j], 0.9) for i, j in zip(bt, bc)]
    cm, cq = ci(bmed), ci(bq90)
    if cm[1] <= 1.10 and cq[1] <= 1.10:
        call = "covered"
    elif cm[0] > 1.25 or cq[0] > 1.25:
        call = "not covered"
    else:
        call = "inconclusive"
    return dict(median_ratio=float(med), median_ci=cm, q90_ratio=float(q90), q90_ci=cq, call=call,
                r_target_median=float(np.median(rt)), r_comp_median=float(np.median(rc)))


# ---------------------------------------------------------------- M2 drift
def m2(X, lp, days, edges, rng):
    b = np.digitize(lp, np.log10(edges[1:-1]))
    ud, di = np.unique(days, return_inverse=True)
    S = np.zeros((len(ud), NB, X.shape[1])); S2 = np.zeros((len(ud), NB)); C = np.zeros((len(ud), NB))
    np.add.at(S, (di, b), X); np.add.at(S2, (di, b), (X ** 2).sum(1)); np.add.at(C, (di, b), 1)

    def stats(w):
        s, s2, c = (w[:, None, None] * S).sum(0), (w[:, None] * S2).sum(0), (w[:, None] * C).sum(0)
        m = s / c[:, None]
        spread = np.sqrt(np.mean(s2 / c - (m ** 2).sum(1)))
        D = np.linalg.norm(m[-1] - m[0]) / spread
        sh = np.diff(m, axis=0)
        cos = [float(sh[k] @ sh[k + 1] / (np.linalg.norm(sh[k]) * np.linalg.norm(sh[k + 1]))) for k in range(NB - 2)]
        return D, min(cos), cos

    D, mc, cos = stats(np.ones(len(ud)))
    bD, bc = [], []
    for _ in range(NBOOT):
        w = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud)).astype(float)
        d_, c_, _ = stats(w)
        bD.append(d_); bc.append(c_)
    cD, cc = ci(bD), ci(bc)
    return dict(D=float(D), D_ci=cD, min_cos=float(mc), min_cos_ci=cc, cos=cos, n_per_bin=np.bincount(b, minlength=NB).tolist(),
                drift=bool(cD[0] > 0.1), smooth=bool(cc[0] >= 0.5))


# ---------------------------------------------------------------- M3 extrapolation vs reuse
def pdist_mean(A, B):
    """Row means of the Euclidean distance matrix between A and B (in chunks)."""
    out = np.empty(len(A))
    for i in range(0, len(A), 100):            # 100 x 5,000 x 21 doubles = 84 MB per chunk
        d = np.sqrt(np.maximum(((A[i:i + 100, None, :] - B[None]) ** 2).sum(-1), 0))
        out[i:i + 100] = d.mean(1)
    return out


def m3(P, beta, delta, T, dt, rng):
    Pt = P + beta * delta
    rr, rt = pdist_mean(T, P), pdist_mean(T, Pt)              # per-target mean distance to each predictor
    pp = pdist_mean(P, P).mean()                               # same for both: a translation keeps distances
    tt = pdist_mean(T, T).mean()
    ed_r, ed_t = 2 * rr.mean() - pp - tt, 2 * rt.mean() - pp - tt
    dED = ed_r - ed_t                                          # = 2 * mean(rr - rt)
    bd = [2 * np.mean(rr[i] - rt[i]) for i in day_boot(dt, rng)]
    c = ci(bd)
    call = "trend wins" if c[0] > 0 else ("reuse wins" if c[1] < 0 else "tie")
    ud = np.unique(dt); half = np.isin(dt, rng.permutation(ud)[: len(ud) // 2])
    A, B = T[half], T[~half]
    floor = 2 * pdist_mean(A, B).mean() - pdist_mean(A, A).mean() - pdist_mean(B, B).mean()
    return dict(ED_reuse=float(ed_r), ED_trend=float(ed_t), dED=float(dED), dED_ci=c, call=call,
                noise_floor_ED=float(floor), delta_log10_peak=float(delta))


def world(c1, c2, c3):
    if c1 == "inconclusive":
        return "no world (coverage inconclusive)"
    if c1 == "covered":
        if c3 == "trend wins":
            return "W2 with predictive drift (W2/W3 boundary)"
        return "W2" if c2 else "W1"
    return "W3" if c3 == "trend wins" else "W4"


# ---------------------------------------------------------------- one arm (main or backtest)
def arm(name, ref_lo, cut, tgt_hi, comp_lo, tr, Xtr, lp_tr, te, rng, out, quick):
    edges = np.geomspace(ref_lo, cut, NB + 1)
    sel = (tr.mx.values >= ref_lo) & (tr.mx.values < cut)
    mu, sd = Xtr[sel].mean(0), Xtr[sel].std(0) + 1e-6
    Z = lambda X: (X - mu) / sd
    R, lpR, dR = Z(Xtr[sel]), lp_tr[sel], tr.day.values[sel]

    tg = te[(te.mx >= cut) & (te.mx < tgt_hi)]
    cp = te[(te.mx >= comp_lo) & (te.mx < cut)]
    if len(tg) > NTGT:
        tg = tg.sample(NTGT, random_state=1)
    if len(cp) > NTGT:
        cp = cp.sample(NTGT, random_state=2)
    if quick:
        tg, cp = tg.sample(min(400, len(tg)), random_state=1), cp.sample(min(400, len(cp)), random_state=2)
    Xt, Et = read("test", tg.r.values, f"{out}/test_{name}_target.npz")
    Xc, Ec = read("test", cp.r.values, f"{out}/test_{name}_comp.npz")
    Zt, Zc = Z(Xt), Z(Xc)

    # predictors: reference hours of the last bin (subsample), slope on log10 peak over all reference hours
    last = np.flatnonzero(lpR >= np.log10(edges[-2]))
    P_idx = rng.choice(last, min(NPRED, len(last)), replace=False)
    P = R[P_idx]
    A = np.c_[np.ones(len(lpR)), lpR]
    beta = np.linalg.lstsq(A, R, rcond=None)[0][1]
    delta = np.log10(tg.mx.values).mean() - lpR[last].mean()

    cov = Coverage(R)
    res = dict(arm=name, reference=f"train {ref_lo:g} <= peak < {cut:g} (n={sel.sum()})",
               target=f"test {cut:g} <= peak < {tgt_hi:g} (n={len(tg)}, days={tg.day.nunique()})",
               comparison=f"test {comp_lo:g} <= peak < {cut:g} (n={len(cp)})", bin_edges=edges.tolist())
    res["M1"] = m1(cov, Zt, tg.day.values, Zc, cp.day.values, rng)
    res["M2"] = m2(R, lpR, dR, edges, rng)
    res["M3"] = m3(P, beta, delta, Zt, tg.day.values, rng)
    res["world"] = world(res["M1"]["call"], res["M2"]["drift"], res["M3"]["call"])

    # tile-edge control (secondary): edge share by bin on subsamples, then M2 and M3 on non-edge hours
    sub = np.concatenate([rng.choice(np.flatnonzero(np.digitize(lpR, np.log10(edges[1:-1])) == k),
                                     min(NPRED if not quick else 300, (np.digitize(lpR, np.log10(edges[1:-1])) == k).sum()),
                                     replace=False) for k in range(NB)])
    rows_sub = tr.r.values[np.flatnonzero(sel)[sub]]
    _, Es = read("train", rows_sub, f"{out}/train_{name}_edgesub.npz", want_features=False)
    bsub = np.digitize(lpR[sub], np.log10(edges[1:-1]))
    res["edge_share_by_bin"] = [float(Es[bsub == k].mean()) for k in range(NB)]
    res["edge_share_target"], res["edge_share_comp"] = float(Et.mean()), float(Ec.mean())
    ne = ~Es
    res["M2_nonedge"] = m2(R[sub][ne], lpR[sub][ne], dR[sub][ne], edges, rng)
    Rn, lpn = R[sub][ne], lpR[sub][ne]
    lastn = lpn >= np.log10(edges[-2])
    beta_n = np.linalg.lstsq(np.c_[np.ones(len(lpn)), lpn], Rn, rcond=None)[0][1]
    delta_n = np.log10(tg.mx.values[~Et]).mean() - lpn[lastn].mean()
    res["M3_nonedge"] = m3(Rn[lastn], beta_n, delta_n, Zt[~Et], tg.day.values[~Et], rng)
    return res, dict(cov=cov, Z=Z, beta=beta, delta=delta, P=P, Zc=Zc, dc=cp.day.values, lpc=np.log10(cp.mx.values))


# ---------------------------------------------------------------- known-answer gate
def gate(ctx, rng):
    """S-reuse / S-trend / S-out built from the comparison hours (last in-range bin), split by day."""
    Zc, dc = ctx["Zc"], ctx["dc"]
    ud = np.unique(dc); half = np.isin(dc, rng.permutation(ud)[: len(ud) // 2])
    T, dT, C, dC = Zc[half], dc[half], Zc[~half], dc[~half]
    cov, P, beta, delta = ctx["cov"], ctx["P"], ctx["beta"], ctx["delta"]
    out = {}
    # S-reuse: in-range hours, delta ~ 0 by construction (same bin as the predictor)
    c1 = m1(cov, T, dT, C, dC, rng)["call"]
    c3 = m3(P, beta, 0.0, T, dT, rng)["call"]
    out["S-reuse"] = dict(M1=c1, M3=c3, pass_=bool(c1 == "covered" and c3 != "trend wins"))
    # S-trend: the same hours moved along the drift by the main run's delta; the test is told that peak offset
    c3 = m3(P, beta, delta, T + beta * delta, dT, rng)["call"]
    out["S-trend"] = dict(M3=c3, pass_=bool(c3 == "trend wins"))
    # S-out: +3 sd on every feature
    c1 = m1(cov, T + 3.0, dT, C, dC, rng)["call"]
    out["S-out"] = dict(M1=c1, pass_=bool(c1 == "not covered"))
    out["all_pass"] = bool(all(v["pass_"] for v in out.values() if isinstance(v, dict)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="small subsamples, for a smoke test only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.quick:
        NBOOT = 50
    rng = np.random.default_rng(20261105)

    tr, te = meta("train"), meta("test")
    c = np.load(f"{AN}/shape_train_ref5.npz")                 # 21 features of every train hour with peak >= 5 mm
    tr = tr.set_index("r").loc[c["rows"]].reset_index()
    Xtr, lp_tr = c["X"], np.log10(tr.mx.values)

    main, ctx = arm("main", 5.0, 50.1, 1e9, 28.2, tr, Xtr, lp_tr, te, rng, a.out, a.quick)
    main["gate"] = gate(ctx, rng)
    back, _ = arm("backtest", 5.0, 20.2, 50.1, 14.2, tr, Xtr, lp_tr, te, rng, a.out, a.quick)
    verdict = main["world"] if main["gate"]["all_pass"] else "NO VERDICT (validity gate failed)"
    flag = "" if back["world"] == main["world"] else " [flagged: backtest world differs]"
    res = dict(main=main, backtest=back, verdict=verdict + flag, quick=a.quick)
    json.dump(res, open(f"{a.out}/trend_test{'_quick' if a.quick else ''}.json", "w"), indent=1)
    for r in (main, back):
        print(f"== {r['arm']}: ref {r['reference']}; target {r['target']}; comp {r['comparison']}")
        m = r["M1"]; print(f"  M1 coverage: median ratio {m['median_ratio']:.3f} {np.round(m['median_ci'], 3).tolist()}, "
                           f"q90 ratio {m['q90_ratio']:.3f} {np.round(m['q90_ci'], 3).tolist()} -> {m['call']}")
        for k in ("M2", "M2_nonedge"):
            m = r[k]; print(f"  {k}: D {m['D']:.3f} {np.round(m['D_ci'], 3).tolist()}, min cos {m['min_cos']:.3f} "
                             f"{np.round(m['min_cos_ci'], 3).tolist()} (cos {np.round(m['cos'], 3).tolist()}) -> "
                             f"drift {m['drift']}, smooth {m['smooth']}")
        for k in ("M3", "M3_nonedge"):
            m = r[k]; print(f"  {k}: ED reuse {m['ED_reuse']:.4f}, trend {m['ED_trend']:.4f}, dED {m['dED']:+.4f} "
                             f"{np.round(m['dED_ci'], 4).tolist()} (noise floor {m['noise_floor_ED']:.4f}, "
                             f"delta {m['delta_log10_peak']:.3f}) -> {m['call']}")
        print(f"  edge share by bin {np.round(r['edge_share_by_bin'], 3).tolist()}, target {r['edge_share_target']:.3f}, "
              f"comp {r['edge_share_comp']:.3f}")
        print(f"  world: {r['world']}")
    print("gate:", json.dumps(main["gate"]))
    print("VERDICT:", res["verdict"])
