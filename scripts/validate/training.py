"""Stage validator: training sanity, on one real batch, before any run is trusted.

Uses the trainer's own pieces (LogSpaceResidualUNet, build_structural_loss,
resolve_structural_weight) and the loss step of src/trainers/unet_analytical.py, in fp32.

  thresholds    the loss's thresholds (store physical_thresholds.npy) = config PHYSICAL_THRESHOLDS
  forward       output shape and finiteness; decoded prediction non-negative
  gradients     each loss term (MSE, w_max x structural) carries a finite, non-zero gradient;
                their norm ratio is reported (the CLAUDE.md activity check, before training)
  overfit       the model drives the loss down on one fixed batch (MSE to <= 20% of its start;
                the structural term does not rise): data, loss and optimiser are wired together
  determinism   two seeded runs of a few steps give the same weights
  checkpoint    save -> load gives identical outputs
  warmup        the structural weight per epoch under MINKOWSKI_WARMUP_EPOCHS (INFO; a run with
                NUM_EPOCHS <= warm-up never trains at full weight, and epoch 0 has weight 0)
  log           (with --log) the run log names the loss and weight and has no NaN losses
"""

from __future__ import annotations

import copy
import io
import os
import re
import sys

import numpy as np

from common import REPO, guarded, info, ok

sys.path.insert(0, REPO)


def _device(name=None):
    import torch
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def components(cfg, device):
    import torch
    from src.losses.competing import build_structural_loss, resolve_structural_weight
    from src.models.unet import LogSpaceResidualUNet
    from src.utils import load_scaler_val
    geom = build_structural_loss(cfg.get("STRUCTURAL_LOSS", "minkowski"), cfg, device)
    w_max, src = resolve_structural_weight(cfg, None)
    max_val = torch.tensor(load_scaler_val(cfg), device=device, dtype=torch.float32)
    make = lambda: LogSpaceResidualUNet(in_channels=2, out_channels=1).to(device)
    return make, geom, w_max, src, max_val


def loss_terms(model, X, Y, G, geom, max_val, anneal=0.05):
    """The trainer's step (src/trainers/unet_analytical.py), fp32: (mse, structural)."""
    import torch
    import torch.nn.functional as F
    pred = model(X)
    mse = F.mse_loss(pred, Y)
    dec = lambda t: F.relu(torch.expm1(torch.clamp(t * max_val, max=7.0)))
    ref = dec(X[:, 0:1]) if geom.needs_reference else None
    g = geom(pred[:, 0:1], Y[:, 0:1], dec(pred[:, 0:1]), dec(Y[:, 0:1]), gamma_target=G,
             reference_phys=ref, anneal_factor=anneal)
    return mse, g


def real_batch(cfg, n, device, seed=0):
    """n samples of the training data, from the extremes subset when there is one (so the high
    thresholds are populated), without augmentation."""
    import json
    import torch
    from src.data.datasets import DeterministicSRDataset
    from src.utils import load_scaler_val
    with open(cfg["DEM_STATS"]) as f:
        d = json.load(f)
    meta = cfg.get("EXTREMES_TRAIN_METADATA_FILE") or cfg["TRAIN_METADATA_FILE"]
    ds = DeterministicSRDataset(cfg["PREPROCESSED_DATA_DIR"], meta, (float(d["dem_mean"]), float(d["dem_std"])),
                                load_scaler_val(cfg), split="train", topology_mode=cfg.get("TOPOLOGY_MODE", "euler"))
    ds.is_train = False                             # the train group, without random flips / rotations
    idx = np.random.default_rng(seed).choice(len(ds), n, replace=False)
    X, Y, G = zip(*[ds[int(i)] for i in idx])
    return torch.stack(X).to(device), torch.stack(Y).to(device), torch.stack(G).to(device)


def c_thresholds(cfg):
    """The loss reads physical_thresholds.npy from the store, not the config: they must agree."""
    from src.utils import load_physical_thresholds
    got = np.asarray(load_physical_thresholds(cfg), float)
    want = np.asarray(cfg["PHYSICAL_THRESHOLDS"], float)
    return [ok("thresholds", got.shape == want.shape and np.allclose(got, want, rtol=1e-5),
               "loss thresholds (store physical_thresholds.npy) = config PHYSICAL_THRESHOLDS",
               loss=str(np.round(got, 3).tolist()), config=str(want.tolist()))]


def c_forward(make, X):
    import torch
    torch.manual_seed(0)
    m = make()
    with torch.no_grad():
        y = m(X)
    return [ok("forward", tuple(y.shape) == (X.shape[0], 1) + tuple(X.shape[2:]) and bool(torch.isfinite(y).all()),
               "shape and finiteness", shape=str(tuple(y.shape)))]


def c_gradients(make, batch, geom, w_max, max_val):
    import torch
    torch.manual_seed(0)
    m = make()
    norms = {}
    for name in ("mse", "structural"):
        m.zero_grad()
        mse, g = loss_terms(m, *batch, geom, max_val)
        (mse if name == "mse" else w_max * g).backward()
        norms[name] = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None)))
    finite = all(np.isfinite(v) for v in norms.values())
    return [ok("gradients", finite and norms["mse"] > 0 and norms["structural"] > 0,
               f"each term's gradient norm at w_max = {w_max:g}", mse=norms["mse"], structural=norms["structural"],
               ratio=norms["structural"] / max(norms["mse"], 1e-30))]


def c_overfit(make, batch, geom, w_max, max_val, steps):
    import torch
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    first = None
    for _ in range(steps):
        opt.zero_grad()
        mse, g = loss_terms(m, *batch, geom, max_val)
        (mse + w_max * g).backward()
        opt.step()
        if first is None:
            first = (float(mse.detach()), float(g.detach()))
    with torch.no_grad():
        mse, g = loss_terms(m, *batch, geom, max_val)
    last = (float(mse), float(g))
    return [ok("overfit", last[0] <= 0.2 * first[0] and last[1] <= first[1] * 1.05,
               f"{steps} AdamW steps on one batch of {batch[0].shape[0]}",
               mse_start=first[0], mse_end=last[0], structural_start=first[1], structural_end=last[1])]


def c_determinism(make, batch, geom, w_max, max_val, device):
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    runs = []
    for _ in range(2):
        torch.manual_seed(123)
        m = make()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(3):
            opt.zero_grad()
            mse, g = loss_terms(m, *batch, geom, max_val)
            (mse + w_max * g).backward()
            opt.step()
        runs.append(torch.cat([p.detach().flatten() for p in m.parameters()]))
    d = float((runs[0] - runs[1]).abs().max())
    tol = 1e-6 if device.type == "cpu" else 1e-5
    return [ok("determinism", d <= tol, "two seeded 3-step runs", max_param_diff=d, tolerance=tol)]


def c_checkpoint(make, X):
    import torch
    torch.manual_seed(1)
    m = make()
    buf = io.BytesIO()
    torch.save({"model_state_dict": m.state_dict()}, buf)
    buf.seek(0)
    m2 = make()
    m2.load_state_dict(torch.load(buf, map_location=X.device)["model_state_dict"])
    m.eval(); m2.eval()
    with torch.no_grad():
        d = float((m(X) - m2(X)).abs().max())
    return [ok("checkpoint", d == 0.0, "save -> load, identical outputs", max_abs_diff=d)]


def c_warmup(cfg, w_max):
    from src.trainers.base import cosine_warmup_weight
    E, wu = cfg.get("NUM_EPOCHS", 25), cfg.get("MINKOWSKI_WARMUP_EPOCHS", 5)
    w = [cosine_warmup_weight(e, E, w_max, wu) for e in range(E)]
    full = sum(1 for x in w if x >= w_max * (1 - 1e-9))
    return [info("warmup", f"structural weight per epoch over {E} epochs, warm-up {wu}: epochs at full weight = {full}"
                 + ("; NO epoch at full weight" if full == 0 else ""), epoch0=w[0], last=w[-1])]


def c_log(path):
    txt = open(path).read()
    has_loss = re.search(r"auxiliary structural loss: (\S+)", txt)
    has_w = re.search(r"structural weight w_max=(\S+) \(from ([^)]+)\)", txt)
    nan = re.search(r"\b(nan|inf)\b", txt, re.I)
    return [ok("log", bool(has_loss and has_w) and not nan, path,
               loss=has_loss.group(1) if has_loss else "missing", weight=has_w.group(1) if has_w else "missing",
               weight_source=has_w.group(2) if has_w else "missing", nan_or_inf="yes" if nan else "no")]


def run(cfg, a):
    device = _device(getattr(a, "device", None))
    make, geom, w_max, src, max_val = components(cfg, device)
    checks = [info("setup", f"device {device}, structural loss {geom.name}, w_max {w_max:g} from {src}")]
    checks += guarded("thresholds", c_thresholds, cfg)
    batch = getattr(a, "batch_override", None)
    if batch is None:
        got = guarded("batch", real_batch, cfg, a.batch, device)
        if not isinstance(got[0], tuple):
            return checks + got                     # no data: FAIL, nothing else can run
        batch = got[0]
    checks += guarded("forward", c_forward, make, batch[0])
    checks += guarded("gradients", c_gradients, make, batch, geom, w_max, max_val)
    checks += guarded("overfit", c_overfit, make, batch, geom, w_max, max_val, a.steps)
    checks += guarded("determinism", c_determinism, make, batch, geom, w_max, max_val, device)
    checks += guarded("checkpoint", c_checkpoint, make, batch[0])
    checks += guarded("warmup", c_warmup, cfg, w_max)
    if getattr(a, "log", None):
        checks += guarded("log", c_log, a.log)
    return checks
