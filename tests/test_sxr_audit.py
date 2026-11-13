"""Tests for scripts/sxr_fetch.py and scripts/sxr_audit.py on made-up chords and groups. No network."""
import shutil

import numpy as np
import pandas as pd
import pytest
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


def test_examples_come_from_development_campaigns_only():
    pairs = pd.DataFrame({"shot_id": [101, 102, 103, 201, 202], "other": [100, 101, 102, 200, 201],
                          "group": ["candidate", "recorded", "recorded", "recorded", "recorded"]})
    camp = {101: "M5", 102: "M6", 103: "M6", 201: "M8", 202: "M9"}
    ex = sxr_audit.pick_examples(pairs, {s: 0.3 for s in camp}, camp, n=2)
    assert [e[0] for e in ex] == [102, 103]  # recorded losses of M5/M6; never M8 or M9
    assert [e[0] for e in sxr_audit.pick_examples(pairs, {s: 0.3 for s in camp}, camp, n=3)] == [101, 102, 103]  # too few recorded: any M5/M6
    assert sxr_audit.encode_examples(ex) == "102:101:300.0;103:102:300.0"
    from scripts import sxr_view
    assert sxr_view.parse_examples("102:101:300.0;103:102:312.5") == [(102, 101, 0.3), (103, 102, 0.3125)]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed here (the GitHub job has it)")
def test_the_viewer_renders_a_pair(tmp_path):
    from scripts import sxr_view
    paths = sxr_view.render(4, 1, 0.25, str(tmp_path), open_group=fake_open, fps=5, seconds=2)
    assert [p.endswith(e) for p, e in zip(paths, (".mp4", ".png"))] == [True, True]
    assert all((tmp_path / name).stat().st_size > 1000 for name in ("4_1.mp4", "4_1.png"))
    assert sxr_view.render(3, 1, 0.25, str(tmp_path), open_group=fake_open) == []  # no soft X-ray group



def test_a_shot_without_recordings_or_stored_oddly_is_skipped():
    def odd(shot, group):
        if group == "summary":
            return fake_open(shot, group)
        if shot == 5:
            return xr.Dataset({"cam_geometry": ("geometry_channel", np.ones(3))})  # geometry only, no time
        if shot == 6:  # opens, but reading it fails in a way not seen before
            class Weird:
                variables = {"time": None}

                def __getitem__(self, key):
                    raise ValueError("something not seen before")
            return Weird()
        return fake_open(shot, group)

    table, report = sxr_fetch.fetch([1, 5, 6], workers=1, open_group=odd, say=lambda line: None)
    assert report["shots_with_data"] == 1 and report["absent"] == 1 and report["odd"] == 1 and report["failed"] == 0



def test_without_ffmpeg_the_viewer_still_makes_the_picture(tmp_path, monkeypatch):
    from scripts import sxr_view
    monkeypatch.setattr(sxr_view.shutil, "which", lambda name: None)
    paths = sxr_view.render(4, 1, 0.25, str(tmp_path), open_group=fake_open, fps=5, seconds=2)
    assert [p.rsplit(".", 1)[1] for p in paths] == ["png"] and (tmp_path / "4_1.png").exists()
