# Copied 2026-10-08 from the calibration run of 2026-10-07 (quality_v4/calib/daily_check/),
# unchanged, to keep the method in the repo (DECISIONS §21-22). Paths are hard-coded.
"""15-min rate vs hourly vs daily accumulation, on fully cleaned (v4 chain) whole days.
Per 128-px tile (stride-128 grid, >= 95% valid): max, wet fraction, unresolved-variance
fraction U = var(x - up(coarse10(x))) / var(x) (linear and log1p), peak ratio
max(x) / max(coarse10(x)), top-excursion area A50 = frac(x >= 0.5 max), raw/clean max ratio.
Temporal: Pearson correlation of the fine residual x - up(coarse(x)) (and of x) between
frames t and t + lag, per tile."""
import sys, os, importlib.util, time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

REPO = "/work/fquareng/ch2/minkowski-loss"
OUT = sys.argv[1]
DAYS = sys.argv[2].split(",")
LAGS = [1, 2, 4, 8, 16]                       # x 15 min


def load_rg():
    sys.argv = ["x"]
    sys.path[:0] = [REPO, f"{REPO}/scripts/data_quality"]
    spec = importlib.util.spec_from_file_location("rg", f"{REPO}/scripts/data_quality/rule_gallery.py")
    rg = importlib.util.module_from_spec(spec); spec.loader.exec_module(rg)
    return rg


def tiles(a, H, W):
    nt_y, nt_x = H // 128, W // 128
    t = a[:nt_y * 128, :nt_x * 128].reshape(nt_y, 128, nt_x, 128).transpose(0, 2, 1, 3)
    return t.reshape(-1, 128, 128)


def residual(x):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(x))[:, None].float()
    c = F.adaptive_avg_pool2d(t, 10)          # 25 km coarse grid (factor 12.8)
    up = F.interpolate(c, size=128, mode="bilinear", align_corners=False)
    return (t - up)[:, 0].numpy(), c[:, 0].numpy()


def metrics(x, raw, okt, agg, day, label):
    if not okt.any():
        return []
    x = x[okt]; raw = raw[okt]; ids = np.nonzero(okt)[0]
    mx = x.reshape(len(x), -1).max(1)
    wet = (x > 0.1).reshape(len(x), -1).mean(1)
    keep = (mx > 0.5) & (wet > 0.01)
    x, raw, ids, mx, wet = x[keep], raw[keep], ids[keep], mx[keep], wet[keep]
    if not len(x):
        return []
    r, c = residual(x)
    rl, _ = residual(np.log1p(x))
    var = x.reshape(len(x), -1).var(1); varl = np.log1p(x).reshape(len(x), -1).var(1)
    U = r.reshape(len(x), -1).var(1) / var
    Ul = rl.reshape(len(x), -1).var(1) / varl
    pr = mx / np.maximum(c.reshape(len(x), -1).max(1), 1e-6)
    a50 = (x >= 0.5 * mx[:, None, None]).reshape(len(x), -1).mean(1)
    rmx = np.nan_to_num(raw).reshape(len(x), -1).max(1)
    return [dict(day=day, agg=agg, label=label, tile=int(i), max=float(m), wet=float(w), U=float(u),
                 Ulog=float(ul), peak_ratio=float(p), A50=float(a), raw_max=float(rm))
            for i, m, w, u, ul, p, a, rm in zip(ids, mx, wet, U, Ul, pr, a50, rmx)]


def corr_rows(xa, xb, okt, lag, day, k):
    if not okt.any():
        return []
    xa, xb = xa[okt], xb[okt]; ids = np.nonzero(okt)[0]
    wa = (xa > 0.1).reshape(len(xa), -1).mean(1); wb = (xb > 0.1).reshape(len(xb), -1).mean(1)
    keep = (wa > 0.05) & (wb > 0.05)
    if not keep.any():
        return []
    xa, xb, ids = xa[keep], xb[keep], ids[keep]
    out = []
    for name, (A, B) in {"field": (np.log1p(xa), np.log1p(xb)),
                         "resid": (residual(np.log1p(xa))[0], residual(np.log1p(xb))[0])}.items():
        A = A.reshape(len(A), -1); B = B.reshape(len(B), -1)
        A = A - A.mean(1, keepdims=True); B = B - B.mean(1, keepdims=True)
        cc = (A * B).sum(1) / np.sqrt((A * A).sum(1) * (B * B).sum(1) + 1e-12)
        out += [dict(day=day, k=k, lag=lag, what=name, tile=int(i), corr=float(v)) for i, v in zip(ids, cc)]
    return out


def run_day(day):
    import torch
    torch.set_num_threads(1)
    rg = load_rg()
    d = day.replace("-", "")
    t0 = time.time()
    dc = rg.make_cleaner(d)
    H, W = dc.ds.sizes["y"], dc.ds.sizes["x"]
    T = dc.T
    if T != 96:
        return f"{day}: {T} frames, skipped"
    dsum = np.zeros((H, W), np.float64); rsum = np.zeros((H, W), np.float64)
    nval = np.zeros((H, W), np.int16)
    hsum = np.zeros((H, W), np.float64); hraw = np.zeros((H, W), np.float64); hval = np.zeros((H, W), np.int16)
    buf = deque(maxlen=max(LAGS) + 1)
    rows, crows = [], []
    for k in range(int(os.environ.get("MAXK", T))):
        raw = dc._read(dc.ds, k)
        fin = dc.clean(k)[0]
        v = np.isfinite(fin)
        f0 = np.where(v, fin, 0.0); r0 = np.where(np.isfinite(raw), raw, 0.0)
        dsum += f0 * 0.25; rsum += r0 * 0.25; nval += v
        hsum += f0 * 0.25; hraw += r0 * 0.25; hval += v
        okt = tiles(v, H, W).reshape(-1, 128 * 128).mean(1) >= 0.95
        if k % 4 == 1:                                    # one 15-min snapshot per hour
            rows += metrics(tiles(f0, H, W), tiles(r0, H, W), okt, "15min", day, k)
        buf.append((k, tiles(f0, H, W), okt))
        if k >= 16 and (k - 16) % 8 == 0:                 # base frame k-16, lags forward
            kb, xb, okb = buf[0]
            for lag in LAGS:
                kk, xl, okl = buf[lag]
                crows += corr_rows(xb, xl, okb & okl, lag, day, kb)
        if k % 4 == 3:                                    # hour complete
            okh = tiles(hval >= 4, H, W).reshape(-1, 128 * 128).mean(1) >= 0.95
            rows += metrics(tiles(hsum, H, W), tiles(hraw, H, W), okh, "1h", day, k // 4)
            hsum[:] = 0; hraw[:] = 0; hval[:] = 0
    okd = tiles(nval >= 92, H, W).reshape(-1, 128 * 128).mean(1) >= 0.95
    rows += metrics(tiles(dsum, H, W), tiles(rsum, H, W), okd, "24h", day, 0)
    pd.DataFrame(rows).to_csv(f"{OUT}/tiles_{d}.csv.gz", index=False)
    pd.DataFrame(crows).to_csv(f"{OUT}/corr_{d}.csv.gz", index=False)
    return f"{day}: {len(rows)} tile rows, {len(crows)} corr rows, {time.time() - t0:.0f} s"


if __name__ == "__main__":
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 3))) as ex:
        for msg in ex.map(run_day, DAYS):
            print(msg, flush=True)
