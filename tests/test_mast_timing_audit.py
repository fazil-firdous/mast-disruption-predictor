"""Tests for scripts/mast_timing_audit.py and scripts/efit_times_fetch.py on made-up time stamps: the measures, the
'denser' direction, the fixed-lead eligibility (an ended twin is left out), the clustered range, the readings, and
the fetch counting. No network."""
import numpy as np
import pandas as pd
import xarray as xr

from scripts import efit_times_fetch, mast_timing_audit as mt


def test_measures():
    avail = np.arange(0.0, 0.5, 0.004) + 0.005  # EFIT every 4 ms, available 5 ms later
    grid = np.round(np.arange(0.0, 0.5, 0.001), 4)
    m = mt.measures(avail, grid, 0.3)
    assert m["E1"] == 5 and abs(m["E3"] - 0.004) < 1e-9 and 0 <= m["E2"] < 0.004
    assert m["G1"] == 20 and abs(m["G2"] - 0.001) < 1e-9
    assert np.isnan(mt.measures(None, grid, 0.3)["E1"])


def test_denser_direction():
    assert mt.denser(5, 3, +1) == 1.0 and mt.denser(0.002, 0.004, -1) == 1.0 and mt.denser(2, 2, +1) == 0.5
    assert np.isnan(mt.denser(np.nan, 1, +1))


def pairs():
    return pd.DataFrame({"shot_id": [11, 12, 13], "other": [1, 2, 3], "ref": [11, 12, 13],
                         "t_loss": [0.30, 0.30, 0.30], "t_start": [0.05, 0.05, 0.05]})


def test_pair_rows_and_reading():
    grid = {s: np.round(np.arange(0.0, 0.40, 0.001), 4) for s in (1, 2, 11, 12, 13)}
    grid[3] = np.round(np.arange(0.0, 0.25, 0.001), 4)  # this twin ended at 0.25 s: counted only at the 100 ms lead
    efit = {s: np.arange(0.0, 0.4, 0.010) + 0.005 for s in (1, 2, 3, 12, 13)}
    efit[11] = np.arange(0.0, 0.4, 0.002) + 0.005  # the first lost shot has EFIT five times denser
    rows = mt.pair_rows(pairs(), efit.get, grid.get)
    assert set(rows["shot_id"]) == {11, 12, 13}
    assert set(rows.loc[rows["shot_id"] == 13, "lead_ms"]) == {100}
    r30 = rows[rows["lead_ms"] == 30].set_index("shot_id")
    assert r30.loc[11, "win_E1"] == 1.0 and r30.loc[12, "win_E1"] == 0.5 and r30.loc[11, "win_G1"] == 0.5
    groups = np.arange(len(rows))
    t = mt.clustered(rows["win_G1"].to_numpy(dtype=float), groups, n_boot=200)
    assert t[0] == 0.5 and mt.reading(t) == "no timing difference at the moment compared"
    assert mt.reading((0.9, 0.8, 0.95)).startswith("TIMING DIFFERS")


def test_fetch_counts():
    def fake(shot, group):
        if shot == 2:
            raise KeyError("equilibrium")
        return xr.Dataset(coords={"time": np.arange(0.0, 0.3, 0.005)})
    table, report = efit_times_fetch.fetch([1, 2, 3], workers=2, open_group=fake, say=lambda line: None)
    assert report["shots_with_data"] == 2 and report["absent"] == 1 and set(table["shot_id"]) == {1, 3}
    assert (table.groupby("shot_id").size() == 60).all()
