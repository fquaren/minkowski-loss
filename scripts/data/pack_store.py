"""Write the packed copy of a patch store (src/data/packed_store.py), then check random rows against the original.

    python scripts/data/pack_store.py <preprocessed_dataset.zarr> [--workers 4] [--check 2000]
    python scripts/data/pack_store.py <preprocessed_dataset.zarr> --check_only
"""
import argparse
import os
import sys
import time

import numpy as np
import zarr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.packed_store import PackedStore, open_store, pack  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("zarr_path")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--check", type=int, default=2000, help="random rows compared per packed array")
ap.add_argument("--check_only", action="store_true")
args = ap.parse_args()

t0 = time.time()
if not args.check_only:
    pack(args.zarr_path, workers=args.workers)
    print(f"pack done in {(time.time() - t0) / 3600:.2f} h", flush=True)

g, ref = open_store(args.zarr_path), zarr.open(args.zarr_path, mode="r")
assert isinstance(g.store, PackedStore), "packed copy not picked up"
rng = np.random.default_rng(0)
for head in g.store._arrays:
    s, a = head.split("/")
    n = ref[s][a].shape[0]
    rows = rng.choice(n, min(args.check, n), replace=False)
    for i in rows:
        x, y = g[s][a][int(i)], ref[s][a][int(i)]
        if not np.array_equal(x, y, equal_nan=True):
            raise SystemExit(f"MISMATCH {head} row {i}")
    print(f"check {head}: {len(rows)} random rows identical", flush=True)
print("=== PACK_OK")
