#!/usr/bin/env python
"""Advection-corrected hourly OPERA sums with pysteps motion, tested against RADKLIM
(2026-10-09; DECISIONS §21 status note; follows `signal_noise.py`).

`signal_noise.py` showed that the plain sum of four 15-min snapshots beads moving rain, and that
a crude correction (one uniform shift per tile) removes the beads but over-smooths fast rain.
This is the proper version, as in the pysteps advection-correction example:
  - a dense motion field per pair of consecutive frames, from Lucas-Kanade on log rates
    (`pysteps.motion.get_method("LK")`);
  - fields at 1-min steps between the frames: the earlier frame moved forward along the motion
    and the later one moved backward, weighted linearly in time (semi-Lagrangian, motion
    taken at the destination point);
  - the hour = the mean of the 60 one-minute fields.

Window: the RADKLIM hour labelled H-10 covers H-70 .. H-10. An ODYSSEY frame labelled t covers
t-10 .. t+5, so its centre is t-2.5, and the matching frame-time window is H-67.5 .. H-7.5
(frames H-75 .. H).

Per 64x64 tile >= 98% valid in RADKLIM, per hour with rain, four OPERA versions against RADKLIM:
  plain   0.25 x frames H-60 .. H-15 (time-aligned, no correction)
  adv     the pysteps-style correction above
Metrics as `signal_noise.py`: coherence and power ratio by wavelength (log1p fields), Euler
characteristic and perimeter at 1, 2, 5, 10, 20 mm, the max, by motion-speed class.

    python scripts/data_quality/advection_test.py run --workers 8
    python scripts/data_quality/advection_test.py summary
"""

import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts", "dataset_v2"), os.path.dirname(__file__)]
from signal_noise import (CAL, CLIM, D, NK, RING, RKD, SPEED_CLASSES, T, THR, functionals,  # noqa: E402
                          radial, spec, speed_class)

OUT = f"{CAL}/advection_test"
VARIANTS = {"blend_linear": ("blend", "linear"), "blend_cubic": ("blend", "cubic"),
            "noblend_linear": ("noblend", "linear"), "noblend_nearest": ("noblend", "nearest")}
OUT_V = f"{CAL}/advection_variants"            # run --variants writes here
STEP_MIN = 15
SUB = 15                                   # 1-min steps between frames


def motion_field(r1, r2):
    """Dense (u, v) in px per 15 min from Lucas-Kanade on log rates (pysteps)."""
    from pysteps import motion
    lk = motion.get_method("LK")
    x = np.stack([np.log(np.maximum(np.nan_to_num(r, nan=0.0), 0.1)) for r in (r1, r2)])
    if (np.nan_to_num(r1) >= 0.5).sum() < 50 or (np.nan_to_num(r2) >= 0.5).sum() < 50:
        return np.zeros((2,) + r1.shape, np.float32)
    return lk(x, dense=True, verbose=False).astype(np.float32)


_GRID = {}


def warp(f, V, s):
    """f moved by s x V (semi-Lagrangian, motion at the destination): out(x) = f(x - s V(x)).
    Bilinear, zero outside; cv2.remap does what ndimage.map_coordinates(order=1) does, ~10x faster."""
    import cv2
    H, W = f.shape
    if (H, W) not in _GRID:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        _GRID[(H, W)] = (yy, xx)
    yy, xx = _GRID[(H, W)]
    return cv2.remap(f.astype(np.float32), xx - s * V[0], yy - s * V[1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def interpolated(frames, motions, t0, t1):
    """Mean of 1-min fields at frame-times t0 .. t1 (minutes after frames[0]), midpoints.
    frames: list of 15-min-spaced frames; motions[j]: motion from frame j to j+1."""
    acc, n = 0.0, 0
    for m in np.arange(t0 + 0.5, t1, 1.0):
        j = int(m // STEP_MIN)
        s = (m - j * STEP_MIN) / STEP_MIN
        V = motions[j]
        acc = acc + (1 - s) * warp(frames[j], V, s) + s * warp(frames[j + 1], V, -(1 - s))
        n += 1
    return acc / n


def run_day(day):
    import scan_tiles as st
    from src.data.hourly import HourlyDay
    t_start = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (by0, by1, bx0, bx1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q, crop=(by0, by1, bx0, bx1)).run()
    times = hd.frames
    ns = lambda t: int(np.datetime64(t, "ns").astype(np.int64))       # datetime64 of different units
    pos = {ns(t): i for i, t in enumerate(times)}                       # compare equal but hash apart
    F = np.nan_to_num(hd.fin, nan=0.0)
    d0 = np.datetime64(pd.Timestamp(day))
    mcache = {}
    keys = ("PP", "PA", "PK", "SPKr", "SPKi", "SAKr", "SAKi")
    acc = {f"{k}_{c}": np.zeros(NK + 1) for k in keys for c in SPEED_CLASSES}
    acc.update({f"n_{c}": 0 for c in SPEED_CLASSES})
    rows = []
    for i in range(24):
        H = d0 + np.timedelta64(i + 1, "h")
        want = [H - np.timedelta64(15 * k, "m") for k in (5, 4, 3, 2, 1, 0)]       # H-75 .. H
        if not all(ns(w) in pos for w in want):
            continue
        idx = [pos[ns(w)] for w in want]
        fr = [F[j] for j in idx]
        mo = []
        for a, b in zip(idx[:-1], idx[1:]):
            if (a, b) not in mcache:
                mcache[(a, b)] = motion_field(F[a], F[b])
            mo.append(mcache[(a, b)])
        plain = 0.25 * (fr[1] + fr[2] + fr[3] + fr[4])                                 # H-60 .. H-15
        adv = interpolated(fr, mo, 7.5, 67.5)                                          # frame-time H-67.5 .. H-7.5
        speedmap = np.hypot(*np.mean(mo[1:5], 0))
        Kh = K24[i]
        valid = np.isfinite(Kh)
        for j in idx:
            valid &= np.isfinite(hd.fin[j])
        for r0 in range(0, by1 - by0 - T + 1, T):
            for c0 in range(0, bx1 - bx0 - T + 1, T):
                sl = (slice(r0, r0 + T), slice(c0, c0 + T))
                vk = valid[sl]
                if vk.mean() < 0.98:
                    continue
                p, a, k = plain[sl] * vk, adv[sl] * vk, np.where(vk, np.nan_to_num(Kh[sl], nan=0.0), 0.0)
                wet = max((p >= 0.1).mean(), (k >= 0.1).mean())
                if max(p.max(), k.max()) < 5 or wet < 0.05:
                    continue
                wm = (p >= 0.5)
                spd = float(np.median(speedmap[sl][wm])) if wm.any() else np.nan
                d = {"day": day, "hour": i, "row": by0 + r0, "col": bx0 + c0, "speed_px15": spd,
                     "max_plain": float(p.max()), "max_adv": float(a.max()), "max_K": float(k.max()),
                     "sum_plain": float(p.sum()), "sum_adv": float(a.sum()), "sum_K": float(k.sum())}
                for name, x in (("plain", p), ("adv", a), ("K", k)):
                    A_, P_, chi = functionals(x, THR)
                    for qq, u in enumerate(THR.astype(int)):
                        d[f"A{u}_{name}"], d[f"P{u}_{name}"], d[f"X{u}_{name}"] = float(A_[qq]), float(P_[qq]), float(chi[qq])
                rows.append(d)
                cl = speed_class(spd)
                sP, sA, sK = spec(np.log1p(p)), spec(np.log1p(a)), spec(np.log1p(k))
                acc[f"PP_{cl}"] += radial(np.abs(sP) ** 2); acc[f"PA_{cl}"] += radial(np.abs(sA) ** 2)
                acc[f"PK_{cl}"] += radial(np.abs(sK) ** 2)
                x1, x2 = sP * np.conj(sK), sA * np.conj(sK)
                acc[f"SPKr_{cl}"] += radial(x1.real); acc[f"SPKi_{cl}"] += radial(x1.imag)
                acc[f"SAKr_{cl}"] += radial(x2.real); acc[f"SAKi_{cl}"] += radial(x2.imag)
                acc[f"n_{cl}"] += 1
    os.makedirs(os.path.join(OUT, "days"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "days", f"{day}.npz"), **{k: np.asarray(v) for k, v in acc.items()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "days", f"{day}.csv.gz"), index=False)
    return day, len(rows), time.time() - t_start


def summary():
    def _read(f):
        try:
            return pd.read_csv(f, dtype={"day": str})
        except pd.errors.EmptyDataError:
            return None
    R = pd.concat([x for x in map(_read, glob.glob(f"{OUT}/days/*.csv.gz")) if x is not None and len(x)], ignore_index=True)
    S = [np.load(f) for f in glob.glob(f"{OUT}/days/*.npz")]
    acc = {k: sum(s[k] for s in S) for k in S[0].files}
    k = np.arange(NK + 1)
    lam = np.where(k > 0, T * 2.0 / np.maximum(k, 1), np.inf)
    coh = lambda r, i, p1, p2: (r ** 2 + i ** 2) / np.maximum(p1 * p2, 1e-12)
    sel = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]
    lines = [f"{R.day.nunique()} days, {len(R):,} tile-hours"]
    for cl in ("all",) + SPEED_CLASSES[:3]:
        cls = SPEED_CLASSES if cl == "all" else (cl,)
        g = lambda key: sum(acc[f"{key}_{c}"] for c in cls)
        n = sum(acc[f"n_{c}"] for c in cls)
        cP = coh(g("SPKr"), g("SPKi"), g("PP"), g("PK")); cA = coh(g("SAKr"), g("SAKi"), g("PA"), g("PK"))
        tab = pd.DataFrame({"wavelength_km": lam[sel], "coh plain": cP[sel], "coh adv": cA[sel],
                            "power plain/K": (g("PP") / g("PK"))[sel], "power adv/K": (g("PA") / g("PK"))[sel]})
        lines.append(f"\nspeed class {cl}, n = {n:,}:\n" + tab.round(3).to_string(index=False))
    for cl, sub in (("all", R), ("slow", R[R.speed_px15 < 2]), ("mid", R[(R.speed_px15 >= 2) & (R.speed_px15 < 6)]),
                    ("fast", R[R.speed_px15 >= 6])):
        rows = []
        for u in THR.astype(int):
            b = sub[(sub[f"A{u}_plain"] > 0) & (sub[f"A{u}_K"] > 0)]
            if len(b) < 30:
                continue
            rows.append({"u_mm": u, "n": len(b), "chi plain": b[f"X{u}_plain"].median(), "chi adv": b[f"X{u}_adv"].median(),
                         "chi K": b[f"X{u}_K"].median(), "med chi plain-K": (b[f"X{u}_plain"] - b[f"X{u}_K"]).median(),
                         "med chi adv-K": (b[f"X{u}_adv"] - b[f"X{u}_K"]).median(),
                         "perim plain/K": (b[f"P{u}_plain"] / b[f"P{u}_K"]).median(),
                         "perim adv/K": (b[f"P{u}_adv"] / b[f"P{u}_K"]).median(),
                         "area adv/K": (b[f"A{u}_adv"] / b[f"A{u}_K"]).median()})
        lines.append(f"\nMinkowski functionals, speed {cl} (n tile-hours {len(sub):,}):\n" + pd.DataFrame(rows).round(3).to_string(index=False))
        lines.append(f"tile max (median): plain {sub.max_plain.median():.1f}  adv {sub.max_adv.median():.1f}  K {sub.max_K.median():.1f};  "
                     f"tile sum adv/plain (median): {(sub.sum_adv / sub.sum_plain.clip(lower=1e-6)).median():.3f}")
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/summary.txt", "w").write(txt)


def run_day_variants(day):
    """All VARIANTS of DECISIONS §25 on the same tile-hours, sharing the motion fields."""
    import scan_tiles as st
    from src.data.hourly import HourlyDay, advected_mean, motion_lk
    t_start = time.time()
    z = np.load(f"{RKD}/{day}.npz")
    K24, (by0, by1, bx0, bx1) = z["K"], z["box"]
    if not np.isfinite(K24).any():
        return day, 0, 0.0
    q, ceil = st._quality(f"{REPO}/configs/quality_v4.yaml")
    hd = HourlyDay(f"{D}/raw/OPERA", day, "TOT_PREC", hot=lambda y: st._hot(CLIM, y),
                   ring=lambda y: st._ring(RING, y), sites_rc=st._sites(), ceilings=ceil,
                   cfg=q, crop=(by0, by1, bx0, bx1)).run()
    ns = lambda t: int(np.datetime64(t, "ns").astype(np.int64))
    pos = {ns(t): i for i, t in enumerate(hd.frames)}
    F = np.nan_to_num(hd.fin, nan=0.0)
    d0 = np.datetime64(pd.Timestamp(day))
    mcache, rows = {}, []
    names = ["plain"] + list(VARIANTS)
    acc = {f"{k}_{v}_{c}": np.zeros(NK + 1) for v in names for k in ("P", "Sr", "Si") for c in SPEED_CLASSES}
    acc.update({f"PK_{c}": np.zeros(NK + 1) for c in SPEED_CLASSES}); acc.update({f"n_{c}": 0 for c in SPEED_CLASSES})
    for i in range(24):
        H = d0 + np.timedelta64(i + 1, "h")
        want = [H - np.timedelta64(15 * k, "m") for k in (5, 4, 3, 2, 1, 0)]
        if not all(ns(w) in pos for w in want):
            continue
        idx = [pos[ns(w)] for w in want]
        fr = [F[j] for j in idx]
        mo = []
        for a_, b_ in zip(idx[:-1], idx[1:]):
            if (a_, b_) not in mcache:
                mcache[(a_, b_)] = motion_lk(F[a_], F[b_])
            mo.append(mcache[(a_, b_)])
        fields = {"plain": 0.25 * (fr[1] + fr[2] + fr[3] + fr[4])}
        for v, (mode, interp) in VARIANTS.items():
            fields[v] = advected_mean(fr, mo, 7.5, 67.5, mode, interp)
        speedmap = np.hypot(*np.mean(mo[1:5], 0))
        Kh = K24[i]
        valid = np.isfinite(Kh)
        for j in idx:
            valid &= np.isfinite(hd.fin[j])
        for r0 in range(0, by1 - by0 - T + 1, T):
            for c0 in range(0, bx1 - bx0 - T + 1, T):
                sl = (slice(r0, r0 + T), slice(c0, c0 + T))
                vk = valid[sl]
                if vk.mean() < 0.98:
                    continue
                k = np.where(vk, np.nan_to_num(Kh[sl], nan=0.0), 0.0)
                p = fields["plain"][sl] * vk
                if max(p.max(), k.max()) < 5 or max((p >= 0.1).mean(), (k >= 0.1).mean()) < 0.05:
                    continue
                wm = p >= 0.5
                spd = float(np.median(speedmap[sl][wm])) if wm.any() else np.nan
                d = {"day": day, "hour": i, "row": by0 + r0, "col": bx0 + c0, "speed_px15": spd,
                     "max_K": float(k.max()), "sum_K": float(k.sum())}
                cl = speed_class(spd)
                sK = spec(np.log1p(k))
                acc[f"PK_{cl}"] += radial(np.abs(sK) ** 2); acc[f"n_{cl}"] += 1
                A_, P_, chi = functionals(k, THR)
                for qq, u in enumerate(THR.astype(int)):
                    d[f"A{u}_K"], d[f"P{u}_K"], d[f"X{u}_K"] = float(A_[qq]), float(P_[qq]), float(chi[qq])
                for v in names:
                    x = fields[v][sl] * vk
                    d[f"max_{v}"], d[f"sum_{v}"] = float(x.max()), float(x.sum())
                    A_, P_, chi = functionals(x, THR)
                    for qq, u in enumerate(THR.astype(int)):
                        d[f"A{u}_{v}"], d[f"P{u}_{v}"], d[f"X{u}_{v}"] = float(A_[qq]), float(P_[qq]), float(chi[qq])
                    sx = spec(np.log1p(x)); xk = sx * np.conj(sK)
                    acc[f"P_{v}_{cl}"] += radial(np.abs(sx) ** 2)
                    acc[f"Sr_{v}_{cl}"] += radial(xk.real); acc[f"Si_{v}_{cl}"] += radial(xk.imag)
                rows.append(d)
    os.makedirs(os.path.join(OUT_V, "days"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT_V, "days", f"{day}.npz"), **{k: np.asarray(v) for k, v in acc.items()})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_V, "days", f"{day}.csv.gz"), index=False)
    return day, len(rows), time.time() - t_start


def summary_variants():
    def _read(f):
        try:
            return pd.read_csv(f, dtype={"day": str})
        except pd.errors.EmptyDataError:
            return None
    R = pd.concat([x for x in map(_read, glob.glob(f"{OUT_V}/days/*.csv.gz")) if x is not None and len(x)], ignore_index=True)
    names = ["plain"] + list(VARIANTS)
    lines = [f"{R.day.nunique()} days, {len(R):,} tile-hours; criteria of DECISIONS §25"]
    slow = R[R.speed_px15 < 2]
    rows = []
    for v in names:
        chi = {u: (R[(R[f"A{u}_plain"] > 0) & (R[f"A{u}_K"] > 0)][f"X{u}_{v}"] -
                   R[(R[f"A{u}_plain"] > 0) & (R[f"A{u}_K"] > 0)][f"X{u}_K"]).median() for u in (1, 2, 5)}
        smax = (slow[f"max_{v}"] / slow.max_plain.clip(lower=1e-6)).median()
        mass = (R[f"sum_{v}"] / R.sum_plain.clip(lower=1e-6)).median()
        rows.append({"variant": v, "chi-K @1": chi[1], "@2": chi[2], "@5": chi[5],
                     "C1 chi within +-1": all(abs(c) <= 1 for c in chi.values()),
                     "slow max / plain": smax, "C2 within 5%": abs(smax - 1) <= 0.05,
                     "mass / plain": mass, "C4 within 1%": abs(mass - 1) <= 0.01,
                     "max/K all": (R[f"max_{v}"] / R.max_K.clip(lower=1e-6)).median(),
                     "max/K slow": (slow[f"max_{v}"] / slow.max_K.clip(lower=1e-6)).median(),
                     "area20 /K": (R[R.A20_K > 0][f"A20_{v}"] / R[R.A20_K > 0].A20_K).median()})
    lines.append(pd.DataFrame(rows).round(3).to_string(index=False))
    for cl_name, sub in (("slow", slow), ("mid", R[(R.speed_px15 >= 2) & (R.speed_px15 < 6)]), ("fast", R[R.speed_px15 >= 6])):
        lines.append(f"\n{cl_name} (n {len(sub):,}): median tile max " +
                     "  ".join(f"{v} {sub[f'max_{v}'].median():.1f}" for v in names) + f"  K {sub.max_K.median():.1f}")
        for u in (5, 10, 20):
            b = sub[(sub[f"A{u}_plain"] > 0) & (sub[f"A{u}_K"] > 0)]
            lines.append(f"  chi-K @{u} mm (n {len(b)}): " + "  ".join(f"{v} {(b[f'X{u}_{v}'] - b[f'X{u}_K']).median():+.0f}" for v in names))
    S = [np.load(f) for f in glob.glob(f"{OUT_V}/days/*.npz")]
    acc = {k: sum(s_[k] for s_ in S) for k in S[0].files}
    k = np.arange(NK + 1); lam = np.where(k > 0, T * 2.0 / np.maximum(k, 1), np.inf)
    sel = [2, 4, 8, 12, 16, 24, 32]
    g = lambda key: sum(acc[f"{key}_{c}"] for c in SPEED_CLASSES)
    tab = {"wavelength_km": lam[sel]}
    for v in names:
        coh = (g(f"Sr_{v}") ** 2 + g(f"Si_{v}") ** 2) / np.maximum(g(f"P_{v}") * g("PK"), 1e-12)
        tab[f"coh {v}"] = coh[sel]; tab[f"pow {v}/K"] = (g(f"P_{v}") / g("PK"))[sel]
    lines.append("\ncoherence with RADKLIM and power ratio (all speeds):\n" + pd.DataFrame(tab).round(3).to_string(index=False))
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT_V}/summary.txt", "w").write(txt)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--days", nargs="*", default=None)
    ap.add_argument("--variants", action="store_true", help="all §25 variants -> advection_variants/")
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary_variants() if a.variants else summary()
    if a.variants:
        have = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{CAL}/signal_noise/days/*.npz"))
        days = [d for d in (a.days or have) if not os.path.exists(os.path.join(OUT_V, "days", f"{d}.npz"))]
        print(f"[advection_test --variants] {len(days)} days -> {OUT_V}", flush=True)
        with ProcessPoolExecutor(a.workers) as ex:
            for day, n, dt in ex.map(run_day_variants, days):
                print(f"  {day}: {n} tile-hours, {dt:.0f} s", flush=True)
        return
    have = sorted(os.path.basename(f)[:8] for f in glob.glob(f"{CAL}/signal_noise/days/*.npz"))
    days = [d for d in (a.days or have) if not os.path.exists(os.path.join(OUT, "days", f"{d}.npz"))]
    print(f"[advection_test] {len(days)} days -> {OUT}", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for day, n, dt in ex.map(run_day, days):
            print(f"  {day}: {n} tile-hours, {dt:.0f} s", flush=True)


if __name__ == "__main__":
    main()
