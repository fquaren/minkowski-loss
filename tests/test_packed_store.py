"""Packed read path: arrays read through the packed copy equal the original store, bit for bit."""
import json
import os
import pickle

import numpy as np
import pytest
import zarr

from src.data.packed_store import PackedStore, open_store, pack, packed_dir


@pytest.fixture
def store(tmp_path):
    p = str(tmp_path / "preprocessed_dataset.zarr")
    root = zarr.open(p, mode="w")
    rng = np.random.default_rng(0)
    for s, n in (("train", 11), ("validation", 5)):
        g = root.create_group(s)
        x = rng.exponential(size=(n, 8, 8)).astype(np.float32)
        x[x < 1] = 0
        g.create_dataset("original_precip", data=x, chunks=(1, 8, 8), fill_value=np.nan)
        g.create_dataset("interpolated_precip", data=x * 0.5, chunks=(1, 8, 8), fill_value=np.nan)
        g.create_dataset("gamma_targets", data=rng.random((n, 5, 3)).astype(np.float32), chunks=(4, 5, 3))
        g.create_dataset("coarse_precip", data=x[:, ::4, ::4], chunks=(1, 2, 2))     # not packed
    os.remove(os.path.join(p, "train", "original_precip", "3.0.0"))                  # an unwritten chunk
    return p


def _arrays(g):
    return {f"{s}/{a}": g[s][a][:] for s in g.group_keys() for a in g[s].array_keys()}


def test_packed_equals_original(store):
    ref = _arrays(zarr.open(store, mode="r"))
    man = pack(store, workers=3)
    assert man["arrays"]["train/original_precip"]["missing"] == 1
    assert "train/coarse_precip" not in man["arrays"]
    g = open_store(store)
    assert isinstance(g.store, PackedStore)
    got = _arrays(g)
    assert ref.keys() == got.keys()
    for k in ref:
        np.testing.assert_array_equal(got[k], ref[k], err_msg=k)       # NaN fill of the missing chunk included
    assert np.isnan(got["train/original_precip"][3]).all()
    # single-row reads, as the datasets do
    for i in (0, 3, 10):
        np.testing.assert_array_equal(g["train"]["original_precip"][i], ref["train/original_precip"][i])
        np.testing.assert_array_equal(g["train"]["gamma_targets"][i], ref["train/gamma_targets"][i])


def test_pickle_and_read_only(store):
    pack(store, workers=2)
    st = pickle.loads(pickle.dumps(PackedStore(store)))
    assert st._fds == {}
    g = zarr.open_group(store=st, mode="r")
    np.testing.assert_array_equal(g["validation"]["interpolated_precip"][:],
                                  zarr.open(store, mode="r")["validation"]["interpolated_precip"][:])
    assert "train/original_precip/3.0.0" not in st and "train/original_precip/2.0.0" in st
    with pytest.raises(PermissionError):
        st["train/original_precip/0.0.0"] = b""


def test_fallback_and_stale_metadata(store):
    assert not isinstance(open_store(store).store, PackedStore)       # no packed copy yet
    pack(store, workers=1)
    za = os.path.join(store, "train", "gamma_targets", ".zarray")
    m = json.load(open(za))
    m["shape"][0] += 1                                                # the store changed after packing
    json.dump(m, open(za, "w"))
    with pytest.raises(RuntimeError, match="re-run the pack"):
        open_store(store)
    os.remove(os.path.join(packed_dir(store), "manifest.json"))      # an unfinished pack is ignored
    assert not isinstance(open_store(store).store, PackedStore)
