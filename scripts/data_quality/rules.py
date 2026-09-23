"""Candidate artefact rules over the feature table of `compute_tile_features.py`.

**These thresholds are starting guesses, not a screen.** Each rule encodes one artefact
signature so its prevalence, its overlap with the others and — once some tiles are
labelled with `patch_gallery.py` — its precision and recall can be measured. Tune them
against labels before letting any of them delete data.

A rule whose inputs are missing from the table (e.g. `qind_low` on the 2023-24 stores,
which carry no QIND, or `static_clutter` without `--climatology`) evaluates to all-NaN and
is reported as unavailable rather than as "never fires".
"""

import numpy as np
import pandas as pd


def _col(df, name):
    return df[name] if name in df else pd.Series(np.nan, index=df.index)


def _flag(cond, *cols):
    """Boolean flag, NaN wherever an input column is NaN (feature unavailable)."""
    out = cond.astype(float)
    for c in cols:
        out[c.isna()] = np.nan
    return out


def _spike(d):
    return _flag((d["n_spikes"] > 0) & (d["spike_mass_frac"] > 0.2),
                 d["n_spikes"])


def _speckle(d):
    return _flag((d["isolated_frac_ge1"] > 0.5) & (d["n_ge1"] < 200),
                 d["isolated_frac_ge1"])


def _isolated_peak(d):
    return _flag((d["max"] >= 10) & (d["peak_block_ratio"] > 60), d["peak_block_ratio"])


def _flicker(d):
    p = pd.concat([_col(d, "persist_prev"), _col(d, "persist_next")], axis=1).max(axis=1)
    return _flag((d["max"] >= 10) & (p < 0.2), p)


def _sea_clutter(d):
    s = _col(d, "argmax_over_sea")
    return _flag((d["max"] >= 10) & (s == 1) & (d["peak_block_mean"] < 0.5), s)


def _spoke(d):
    return _flag(d["spoke_len"] > 40, d["spoke_len"])


def _static(d):
    c = _col(d, "clim_freq_ge31_at_max")
    return _flag((d["max"] >= 31) & (c > 0.01), c)


def _qind_low(d):
    q = _col(d, "q_at_max")
    return _flag((d["max"] >= 10) & (q < 0.3), q)


def _declutter_isolated(d):
    return _flag(d["n_over_declutter"] > d["n_over_declutter_in_cell"], d["n_over_declutter"])


def _unphysical(d):
    return _flag(d["max"] > 500, d["max"])


RULES = {
    # name: (function, what it is meant to catch)
    "spike": (_spike, "a few isolated intense pixels carry >20% of the tile's rain"),
    "speckle": (_speckle, ">50% of pixels >=1 mm/h have no wet neighbour, small wet area"),
    "isolated_peak": (_isolated_peak, "peak >60x the mean of its coarse block"),
    "flicker": (_flicker, "peak >=10 with <20% of it within 12 km at t-15 and t+15"),
    "sea_clutter": (_sea_clutter, "peak over sea in a nearly dry coarse block"),
    "spoke": (_spoke, "straight thin component longer than 80 km"),
    "static_clutter": (_static, "peak pixel exceeds 31 mm/h in >1% of all time steps"),
    "qind_low": (_qind_low, "OPERA quality index <0.3 at the peak"),
    "declutter_isolated": (_declutter_isolated, ">150 mm/h pixels outside any cell"),
    "unphysical": (_unphysical, "raw max above 500 mm/h"),
}

# Not an artefact — the tiles our own declutter step currently holes.
INFORMATIONAL = {
    "declutter_hole": (lambda d: _flag(d["n_over_declutter_in_cell"] > 0,
                                       d["n_over_declutter_in_cell"]),
                       ">150 mm/h inside a cell: zeroed today, a hole in a real storm"),
}


def apply_rules(df: pd.DataFrame, rules=None) -> pd.DataFrame:
    rules = RULES if rules is None else rules
    out = pd.DataFrame({name: fn(df) for name, (fn, _) in rules.items()}, index=df.index)
    avail = out.notna()
    out["any_flag"] = (out.fillna(0) > 0).any(axis=1).astype(float)
    out.loc[~avail.any(axis=1), "any_flag"] = np.nan
    return out
