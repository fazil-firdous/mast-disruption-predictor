"""Tests for scripts/mast_value_repetition.py on made-up EFIT series: reading slice values from the held grid, exact
repetition with NaN never counted as a change, and the paired comparison. No network."""
import numpy as np
import pandas as pd

from scripts import mast_value_repetition as vr
from src import config

GRID = np.round(np.arange(0.0, 0.40, 0.001), 4)


def frame(values_per_slice, slice_times):
    """An extras frame holding each slice's values from slice time + delay, like src/extras.py does."""
    out = pd.DataFrame({"time": GRID})
    known = slice_times + config.EFIT_DELAY_MS / 1000
    idx = np.searchsorted(known, GRID + 1e-9, side="right") - 1
    for j, v in enumerate(vr.VARS):
        col = np.where(idx >= 0, values_per_slice[np.clip(idx, 0, None), j], np.nan)
        out[v] = col
    return out


def test_slice_values_and_repetition():
    st = np.round(np.arange(0.0, 0.35, 0.005), 4)
    vals = np.tile(np.arange(len(st), dtype=float)[:, None], (1, len(vr.VARS)))  # every slice new
    vals[40:44] = vals[40]  # slices 40-43 repeat slice 40's values
    avail, got = vr.slice_values(st, frame(vals, st))
    assert np.allclose(got, vals[: len(avail)])
    r, r_any, run = vr.repetition(avail, got, avail[45], span=0.1)
    assert abs(r - 3 / 19) < 1e-9 and run == 4
    got2 = got.copy()
    got2[42, 0] = np.nan  # a NaN pair is left out, never a change
    r2, _, _ = vr.repetition(avail, got2, avail[45], span=0.1)
    assert abs(r2 - 1 / 17) < 1e-9


def test_pair_rows():
    st = np.round(np.arange(0.0, 0.35, 0.005), 4)
    fresh = np.tile(np.arange(len(st), dtype=float)[:, None], (1, len(vr.VARS)))
    held = fresh.copy()
    held[1::2] = held[0::2][: len(held[1::2])]  # every value held twice: a sparser source
    series = {11: vr.slice_values(st, frame(held, st)), 1: vr.slice_values(st, frame(fresh, st)),
              12: vr.slice_values(st, frame(fresh, st)), 2: vr.slice_values(st, frame(fresh, st))}
    pairs = pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "t_loss": [0.30, 0.30], "t_start": [0.05, 0.05]})
    rows = vr.pair_rows(pairs, series.get, lambda s: 0.399)
    r30 = rows[rows["lead_ms"] == 30].set_index("shot_id")
    assert r30.loc[11, "win_R"] == 1.0 and r30.loc[12, "win_R"] == 0.5
    out = vr.text(rows, {"lost": 0.5, "kept": 0.0})
    assert "PRIMARY R (all six equal) at 30 ms, 2 pairs" in out
