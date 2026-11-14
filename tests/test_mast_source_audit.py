"""Tests for the level-1 path: reading attempted/converged times with a flag, the source measures, and the paired
comparison with its readings. No network."""
import numpy as np
import pandas as pd
import xarray as xr

from scripts import efit_times_fetch as ef, mast_source_audit as sa_


def test_level1_times_flag_and_counts():
    def fake(shot, group):
        if shot == 2:
            raise KeyError("efm")
        allt = np.array([0.0, 0.005, 0.010, 0.040, 0.045])
        conv = allt if shot == 1 else allt[[0, 1, 3, 4]]
        return xr.Dataset({"all_times": ("time", allt), "cnvrgd_times": ("c", conv)})
    table, report = ef.fetch_level1([1, 2, 3], workers=2, open_group=fake, say=lambda line: None)
    assert report["shots_with_data"] == 2 and report["absent"] == 1 and report["shots_with_unconverged"] == 1
    assert table.loc[table["shot_id"] == 3, "converged"].tolist() == [True, True, False, True, True]


def test_source_measures_and_pairs():
    dense = np.arange(0.0, 0.4, 0.005) + 0.005
    sparse = np.arange(0.0, 0.4, 0.020) + 0.005
    none = np.array([])
    m = sa_.measures(dense, none, 0.3)
    assert m["SE1"] == 4 and m["SA1"] == 0
    times = {11: (dense, none), 1: (sparse, none), 12: (sparse, none), 2: (sparse, none)}
    pairs = pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "t_loss": [0.30, 0.30], "t_start": [0.05, 0.05]})
    rows = sa_.pair_rows(pairs, times.get, lambda s: 0.399)
    r30 = rows[rows["lead_ms"] == 30].set_index("shot_id")
    assert r30.loc[11, "win_SE1"] == 1.0 and r30.loc[12, "win_SE1"] == 0.5
    out = sa_.text(rows, pd.DataFrame({"lost": [True, False], "SE1": [4.0, 1.0]}), {"missing": 0.0, "unconverged": 0.0})
    assert "PRIMARY SE1" in out and "2 pairs" in out
    assert sa_.reading((0.80, 0.70, 0.90)).startswith("THE NATIVE SCHEDULE DIFFERS")
    assert sa_.reading((0.20, 0.10, 0.30)).startswith("THE NATIVE SCHEDULE DIFFERS")  # two-sided
