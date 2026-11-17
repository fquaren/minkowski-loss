"""Packed read path for the patch stores: one file per array instead of one file per patch.

Why: the stores keep chunks of one patch ([1, 128, 128]), so a training sample opens and closes three files over
NFS. Training nodes ran ~1,000-1,500 NFS CLOSE + DELEGRETURN per second, and twice (node32 2026-11-05, node40
2026-11-16) the NFSv4 state manager for server1 ended up spinning at 100% CPU with every open on the node hanging.

A packed copy holds, for each hot array, the chunk bytes exactly as zarr wrote them, concatenated into a few part
files plus an index (part, offset, length) per chunk. PackedStore serves those chunks with os.pread on file
descriptors opened once per process; metadata and every other array still come from the original directory
store. The decoded arrays are therefore bit-identical, and the store stays on NFS.

    preprocessed_dataset.zarr/      the original (unchanged)
    preprocessed_dataset.packed/    manifest.json, <split>/<array>.part<k>.bin, <split>/<array>.idx.npy

open_store() uses the packed copy when its manifest exists and the packed arrays' .zarray still match the
original; otherwise it falls back to zarr.open.
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool

import numpy as np
import zarr
from zarr.storage import BaseStore, DirectoryStore

PACKED_ARRAYS = ("original_precip", "interpolated_precip", "gamma_targets")
_ANNOUNCED = set()


def packed_dir(zarr_path: str) -> str:
    return os.path.splitext(os.path.normpath(zarr_path))[0] + ".packed"


def _grid(zarray: dict):
    return tuple(-(-s // c) for s, c in zip(zarray["shape"], zarray["chunks"]))


class PackedStore(BaseStore):
    """Read-only zarr v2 store: chunks of packed arrays by offset, everything else from the directory store."""

    _readable, _writeable, _erasable, _listable = True, False, False, True

    def __init__(self, zarr_path: str, pdir: str | None = None):
        self.zarr_path = zarr_path
        self.pdir = pdir or packed_dir(zarr_path)
        self._dir = DirectoryStore(zarr_path)
        with open(os.path.join(self.pdir, "manifest.json")) as f:
            man = json.load(f)
        self._arrays = {}
        for path, m in man["arrays"].items():
            if json.loads(self._dir[path + "/.zarray"]) != m["zarray"]:
                raise RuntimeError(f"{path}: .zarray changed since packing; re-run the pack")
            self._arrays[path] = dict(m, grid=_grid(m["zarray"]))
        self._idx, self._fds = {}, {}

    # fds and memory maps are per process; a pickled copy (spawned DataLoader worker) reopens them
    def __getstate__(self):
        return {"zarr_path": self.zarr_path, "pdir": self.pdir}

    def __setstate__(self, s):
        self.__init__(s["zarr_path"], s["pdir"])

    def _locate(self, key):
        head, _, last = key.rpartition("/")
        m = self._arrays.get(head)
        if m is None or last.startswith("."):
            return None
        try:
            i = int(np.ravel_multi_index(tuple(int(c) for c in last.split(".")), m["grid"]))
        except ValueError:
            return None
        if head not in self._idx:
            self._idx[head] = np.load(os.path.join(self.pdir, head + ".idx.npy"), mmap_mode="r")
        part, off, n = (int(v) for v in self._idx[head][i])
        return head, part, off, n

    def __getitem__(self, key):
        loc = self._locate(key)
        if loc is None:
            return self._dir[key]
        head, part, off, n = loc
        if n == 0:
            raise KeyError(key)                        # chunk never written: zarr uses the fill value
        fd = self._fds.get((head, part))
        if fd is None:
            fd = self._fds[(head, part)] = os.open(os.path.join(self.pdir, f"{head}.part{part}.bin"), os.O_RDONLY)
        b = os.pread(fd, n, off)
        if len(b) != n:
            raise OSError(f"short read for {key}: {len(b)} of {n} bytes")
        return b

    def __contains__(self, key):
        loc = self._locate(key)
        return key in self._dir if loc is None else loc[3] > 0

    def listdir(self, path=""):
        return self._dir.listdir(path)

    def __iter__(self):
        return iter(self._dir)

    def __len__(self):
        return len(self._dir)

    def __setitem__(self, key, value):
        raise PermissionError("PackedStore is read-only")

    def __delitem__(self, key):
        raise PermissionError("PackedStore is read-only")

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


def open_store(zarr_path: str):
    """zarr.open(zarr_path, mode="r"), through the packed copy when one exists next to the store."""
    p = packed_dir(zarr_path)
    if not os.path.exists(os.path.join(p, "manifest.json")):
        return zarr.open(zarr_path, mode="r")
    g = zarr.open_group(store=PackedStore(zarr_path, p), mode="r")
    if p not in _ANNOUNCED:                            # once per process; DataLoader workers stay quiet
        _ANNOUNCED.add(p)
        try:
            from torch.utils.data import get_worker_info
            quiet = get_worker_info() is not None
        except ImportError:
            quiet = False
        if not quiet:
            print(f"[store] packed read path {p}", flush=True)
    return g


# ---- packing ----------------------------------------------------------------------------------------------
def _pack_range(args):
    """Copy chunks [lo, hi) of one array into one part file; returns the (offset, length) rows."""
    zarr_path, head, grid, sep, lo, hi, out = args
    rows = np.zeros((hi - lo, 2), dtype=np.int64)
    off = 0
    with open(out, "wb") as f:
        for i in range(lo, hi):
            name = sep.join(str(c) for c in np.unravel_index(i, grid))
            try:
                with open(os.path.join(zarr_path, head, name), "rb") as g:
                    b = g.read()
            except FileNotFoundError:
                continue
            f.write(b)
            rows[i - lo] = off, len(b)
            off += len(b)
    return rows


def pack(zarr_path: str, arrays=PACKED_ARRAYS, splits=None, workers: int = 4, pdir: str | None = None):
    """Write the packed copy. The manifest is written last, so a half-finished pack is never used."""
    pdir = pdir or packed_dir(zarr_path)
    man_path = os.path.join(pdir, "manifest.json")
    if os.path.exists(man_path):
        os.remove(man_path)
    root = zarr.open(zarr_path, mode="r")
    splits = list(root.group_keys()) if splits is None else splits
    manifest = {"source": os.path.abspath(zarr_path), "arrays": {}}
    for s in splits:
        for a in arrays:
            head = f"{s}/{a}"
            if a not in root[s]:
                continue
            za = json.loads(DirectoryStore(zarr_path)[head + "/.zarray"])
            grid, sep = _grid(za), za.get("dimension_separator", ".")
            n = int(np.prod(grid))
            k = max(1, min(workers, n))
            edges = np.linspace(0, n, k + 1).astype(int)
            os.makedirs(os.path.join(pdir, s), exist_ok=True)
            jobs = [(zarr_path, head, grid, sep, int(edges[j]), int(edges[j + 1]),
                     os.path.join(pdir, f"{head}.part{j}.bin")) for j in range(k)]
            with Pool(k) as pool:
                parts = pool.map(_pack_range, jobs)
            idx = np.concatenate([np.column_stack([np.full(len(r), j, np.int64), r]) for j, r in enumerate(parts)])
            np.save(os.path.join(pdir, head + ".idx.npy"), idx)
            missing = int((idx[:, 2] == 0).sum())
            manifest["arrays"][head] = {"zarray": za, "n_chunks": n, "n_parts": k, "missing": missing,
                                        "bytes": int(idx[:, 2].sum())}
            print(f"packed {head}: {n} chunks, {missing} missing, {idx[:, 2].sum() / 1e9:.2f} GB", flush=True)
    with open(man_path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(man_path + ".tmp", man_path)
    return manifest
