"""Stage validator: gamma (Minkowski) targets in the store.

  setup       physical_thresholds.npy = config PHYSICAL_THRESHOLDS; persistence thresholds present
  shape       gamma_targets (n, 5, Q) per split, Q = number of thresholds
  recompute   sampled rows recomputed with src.data.gamma.compute_gamma_matrix equal the stored ones
  invariants  A, P >= 0; A non-increasing with threshold and <= patch area; chi integer; all zero
              where the patch max is below the threshold; no all-zero row on a wet patch
              (an unwritten row)
"""

from __future__ import annotations

import os
import sys

import numpy as np

from common import REPO, guarded, ok

sys.path.insert(0, REPO)
from store import groups, store  # noqa: E402


def c_setup(cfg):
    p = os.path.join(cfg["PREPROCESSED_DATA_DIR"], "physical_thresholds.npy")
    thr = np.load(p) if os.path.exists(p) else None
    want = np.asarray(cfg["PHYSICAL_THRESHOLDS"], np.float32)
    pers = os.path.exists(os.path.join(cfg["PREPROCESSED_DATA_DIR"], "persistence_thresholds.yaml")) or \
        cfg.get("PERSISTENCE_THRESHOLD") is not None
    return [ok("setup/thresholds", thr is not None and thr.shape == want.shape and np.allclose(thr, want, rtol=1e-5),
               "physical_thresholds.npy = config PHYSICAL_THRESHOLDS",
               stored=None if thr is None else str(np.round(thr, 3).tolist()), config=str(want.tolist())),
            ok("setup/persistence", pers, "persistence_thresholds.yaml (or PERSISTENCE_THRESHOLD) present")]


def c_shape(cfg):
    z = store(cfg)
    Q = len(cfg["PHYSICAL_THRESHOLDS"])
    out = []
    for g in groups(cfg):
        if g not in z:
            continue
        G = z[g]
        has = "gamma_targets" in G
        out.append(ok(f"shape/{g}", has and G["gamma_targets"].shape == (G["original_precip"].shape[0], 5, Q),
                      "gamma_targets (n, 5, Q)", shape=str(G["gamma_targets"].shape) if has else "missing",
                      expected=str((G["original_precip"].shape[0], 5, Q))))
    return out


def c_recompute(cfg, n_sample, seed=0):
    from src.data.gamma import compute_gamma_matrix
    from src.utils import load_persistence_thresholds
    z = store(cfg)
    thr = np.asarray(cfg["PHYSICAL_THRESHOLDS"], np.float32)
    tb0, tb1 = load_persistence_thresholds(cfg)
    px = cfg.get("PIXEL_SIZE_KM", 2.0)
    rng = np.random.default_rng(seed)
    out = []
    for g in groups(cfg):
        if g not in z or "gamma_targets" not in z[g]:
            continue
        G = z[g]
        n = G["original_precip"].shape[0]
        idx = rng.choice(n, min(n_sample, n), replace=False)
        worst = 0.0
        for i in idx:
            ref = compute_gamma_matrix(G["original_precip"][i], thr, px, tb0, tb1)
            got = G["gamma_targets"][i]
            worst = max(worst, float(np.max(np.abs(ref - got) / (1.0 + np.abs(ref)))))
        out.append(ok(f"recompute/{g}", worst < 1e-3, f"{len(idx)} sampled rows, max relative difference",
                      max_rel_diff=worst))
    return out


def c_invariants(cfg, n_sample, seed=1):
    z = store(cfg)
    thr = np.asarray(cfg["PHYSICAL_THRESHOLDS"], float)
    area_max = 128 * 128 * cfg.get("PIXEL_SIZE_KM", 2.0) ** 2
    rng = np.random.default_rng(seed)
    out = []
    for g in groups(cfg):
        if g not in z or "gamma_targets" not in z[g]:
            continue
        G = z[g]
        n = G["original_precip"].shape[0]
        idx = np.sort(rng.choice(n, min(n_sample, n), replace=False))
        bad = {"negative A or P": 0, "A increasing": 0, "A > patch area": 0, "chi not integer": 0,
               "nonzero above max": 0, "unwritten (wet, all zero)": 0}
        for i in idx:
            gm = np.asarray(G["gamma_targets"][i], float)
            mx = float(np.nanmax(G["original_precip"][i]))
            A, P, chi = gm[0], gm[1], gm[4]
            bad["negative A or P"] += bool((A < -1e-6).any() or (P < -1e-6).any())
            bad["A increasing"] += bool((np.diff(A) > 1e-3).any())
            bad["A > patch area"] += bool((A > area_max + 1e-3).any())
            bad["chi not integer"] += bool((np.abs(chi - np.round(chi)) > 1e-4).any())
            above = thr > mx
            bad["nonzero above max"] += bool(np.abs(gm[:, above]).max(initial=0) > 1e-6)
            bad["unwritten (wet, all zero)"] += bool(mx >= thr[0] and np.abs(gm).max() == 0)
        out.append(ok(f"invariants/{g}", sum(bad.values()) == 0, f"{len(idx)} sampled rows", **bad))
    return out


def run(cfg, a):
    checks = []
    checks += guarded("setup", c_setup, cfg)
    checks += guarded("shape", c_shape, cfg)
    checks += guarded("recompute", c_recompute, cfg, max(20, a.n_sample // 10))
    checks += guarded("invariants", c_invariants, cfg, a.n_sample)
    return checks
