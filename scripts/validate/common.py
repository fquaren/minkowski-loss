"""Shared pieces of the stage validators: a check result and the report writer.

A stage passes only if every check passes. A check that cannot run (missing input) is a FAIL,
not a skip: an absent artefact is exactly what a gate must catch. Checks that only describe
(no pass/fail) are marked INFO.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
from dataclasses import dataclass, field

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class Check:
    name: str
    status: str                      # "PASS" | "FAIL" | "INFO"
    detail: str = ""
    numbers: dict = field(default_factory=dict)


def ok(name, cond, detail="", **numbers):
    return Check(name, "PASS" if cond else "FAIL", detail, numbers)


def info(name, detail="", **numbers):
    return Check(name, "INFO", detail, numbers)


def guarded(name, fn, *args, **kw):
    """Run a check function; an exception becomes a FAIL with its message."""
    try:
        out = fn(*args, **kw)
        return out if isinstance(out, list) else [out]
    except Exception as e:                                   # noqa: BLE001
        return [Check(name, "FAIL", f"could not run: {type(e).__name__}: {e}")]


def commit():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def write_report(stage, checks, context, out_root=None):
    out_root = out_root or os.path.join(REPO, "validation")
    now = dt.datetime.now()
    passed = all(c.status != "FAIL" for c in checks)
    os.makedirs(os.path.join(out_root, stage), exist_ok=True)
    path = os.path.join(out_root, stage, f"{now:%Y%m%d_%H%M}_{commit()}.md")
    L = [f"# Validation: {stage} — {'PASS' if passed else 'FAIL'}", "",
         f"{now:%Y-%m-%d %H:%M}, code {commit()}. " + "; ".join(f"{k}: {v}" for k, v in context.items()), "",
         "| check | status | detail |", "|---|---|---|"]
    for c in checks:
        nums = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in c.numbers.items())
        L.append(f"| {c.name} | {'**FAIL**' if c.status == 'FAIL' else c.status} | {c.detail}{'; ' if c.detail and nums else ''}{nums} |")
    open(path, "w").write("\n".join(L) + "\n")
    return path, passed
