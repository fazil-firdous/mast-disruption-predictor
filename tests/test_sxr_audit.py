"""Tests for scripts/sxr_fetch.py and scripts/sxr_audit.py on made-up chords and groups. No network."""
import numpy as np
import pandas as pd
import xarray as xr

from scripts import sxr_audit, sxr_fetch

T = np.arange(-0.01, 0.5, 1e-4)


def test_ids_round_trip_and_stay_small():
    ids = list(range(11766, 30473, 7))
    assert sxr_fetch.decode_ids(sxr_fetch.encode_ids(ids)) == ids and len(sxr_fetch.encode_ids(ids)) < 2000


def test_chord_stats():
    rng = np.random.default_rng(0)
    x = np.where(T < 0, 0, 1.0 + 0.5 * np.sin(200 * T)) + rng.normal(0, 0.01, len(T))
    x[T > 0.40] = np.nan  # the camera stopped at 0.40 s
    s = sxr_fetch.chord_stats(T, x, 0.0, 0.45)
    assert abs(s["t_last"] - 0.40) < 1e-3 and 0.85 < s["covered"] < 0.92 and s["snr"] > 50 and s["saturated"] < 0.01
    stuck = np.where(T < 0, 0, np.minimum(3 * np.sin(50 * T), 1.0)) + 0 * T
    assert sxr_fetch.chord_stats(T, stuck, 0.0, 0.45)["saturated"] > 0.3


def fake_open(shot, group):
    if group == "summary":
        return xr.Dataset({"ip": ("time", np.where((T > 0) & (T < 0.30), 4e5, 0.0))}, coords={"time": T})
    if shot == 3:
        raise KeyError("soft_x_rays")
    rng = np.random.default_rng(shot)
    sig = np.vstack([np.where(T < 0, 0, 1 + 0.4 * np.sin(150 * T + k)) + rng.normal(0, 0.01, len(T)) for k in range(14)])
    if shot == 2:
        sig[:, T > 0.20] = np.nan  # this shot's camera stops before its loss at 0.25 s
    return xr.Dataset({"cam_a": (("ch", "time"), sig)}, coords={"time": T, "ch": [f"c{k}" for k in range(14)]})


def test_fetch_and_the_audit():
    table, report = sxr_fetch.fetch([1, 2, 3, 4], workers=2, open_group=fake_open, say=lambda line: None)
    assert report["shots_with_data"] == 3 and report["absent"] == 1
    pairs = pd.DataFrame({"shot_id": [2, 4, 3], "other": [1, 1, 1]})
    pt = sxr_audit.pair_table(pairs, table, {2: 0.25, 4: 0.25, 3: 0.25}).set_index("shot_id")
    assert not pt.loc[2, "lost_ok"]  # covers only 2/3 of its plasma: not usable
    assert pt.loc[4, "usable"] and pt.loc[4, "cameras"] == "cam_a"
    assert not pt.loc[3, "lost_has"] and not pt.loc[3, "usable"]
    out, _ = sxr_audit.text(pairs, table, {2: 0.25, 4: 0.25, 3: 0.25}, {1: "M7", 2: "M7", 3: "M8", 4: "M8"})
    assert "SNR >= 5: usable pairs 1 of 3" in out and "twin usable, lost not: 2" in out
