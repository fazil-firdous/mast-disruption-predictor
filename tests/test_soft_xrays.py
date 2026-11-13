"""Tests for scripts/sxr_series.py and scripts/soft_xrays.py on made-up cameras: the 1 ms signals use no later sample
and measure what they say, stuck and dead chords are left out, the window features are causal, one run end to end on
the tiny fake project, and the summary from saved runs. No network."""
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

from scripts import fast_magnetics, soft_xrays, sxr_series, v3_explore
from src import config, experiment
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)

FS = 1e5  # 100 kHz here (500 kHz on MAST) to keep the tests quick
T = np.round(np.arange(-0.01, 0.12, 1 / FS), 7)


def profile(centre, n=18, width=2.0):
    return np.exp(-0.5 * ((np.arange(n) - centre) / width) ** 2)


def camera(level=1.0, centre=5.0, offset=0.3, noise=0.001, seed=0):
    rng = np.random.default_rng(seed)
    x = offset + (T >= 0)[None, :] * level * profile(centre)[:, None]
    return x + rng.normal(0, noise, x.shape)


def test_signals_measure_what_they_say():
    values, k, n = sxr_series.camera_bins(T, camera(centre=5.0), 1, 100)
    assert k[0] == 1 and k[-1] == 100 and (n == 18).all()
    p = profile(5.0)
    assert np.allclose(values["level"][10:], p.mean(), rtol=0.02)  # offset removed
    assert np.allclose(values["position"][10:], (np.arange(18) / 17 * p).sum() / p.sum(), atol=0.01)
    assert np.allclose(values["peak"][10:], p.max() / p.sum(), rtol=0.03)
    higher, _, _ = sxr_series.camera_bins(T, camera(centre=12.0), 1, 100)
    assert (higher["position"][10:] > values["position"][10:] + 0.3).all()
    # a 20 kHz oscillation shows in fluct, not in the 1 ms means
    x = camera()
    x[:, T >= 0.05] += 0.2 * np.sin(2 * np.pi * 2e4 * T[T >= 0.05])
    wobbly, _, _ = sxr_series.camera_bins(T, x, 1, 100)
    assert wobbly["fluct"][60:].mean() > 20 * wobbly["fluct"][20:40].mean()
    assert np.allclose(wobbly["level"][60:], values["level"][60:], rtol=0.02)


def test_bins_use_no_later_sample():
    x = camera()
    a, k, _ = sxr_series.camera_bins(T, x, 1, 100)
    y = x.copy()
    y[:, T > 0.050 + 1e-9] *= 5  # change everything after 50 ms
    b, _, _ = sxr_series.camera_bins(T, y, 1, 100)
    upto = k <= 50
    for s in sxr_series.SIGNALS:
        assert np.allclose(a[s][upto], b[s][upto], equal_nan=True)
    assert not np.allclose(a["level"][k == 51], b["level"][k == 51])


def test_stuck_dead_and_too_few_chords():
    x = camera()
    x[3, T > 0.03] = 7.0  # chord 3 stuck at a constant: left out of those bins
    x[4] = 0.25  # chord 4 dead from the start: left out for the shot
    v, k, n = sxr_series.camera_bins(T, x, 1, 100)
    assert n[k == 20][0] == 17 and n[k == 60][0] == 16 and np.isfinite(v["level"][k == 60][0])
    x[:8, T > 0.08] = np.nan  # the camera loses 8 chords: fewer than 12 left
    v, k, n = sxr_series.camera_bins(T, x, 1, 100)
    assert np.isnan(v["level"][k == 90][0]) and np.isnan(v["position"][k == 90][0])


def fake_open(shot, group):
    if group == "summary":
        return xr.Dataset({"ip": ("time", np.where((T > 0.01) & (T < 0.10), 4e5, 0.0))}, coords={"time": T})
    if shot == 3:
        raise KeyError("soft_x_rays")
    if shot == 4:
        return xr.Dataset({"geometry": ("ch", np.zeros(18))}, coords={"ch": np.arange(18)})
    cams = {c: (("ch", "time"), camera(seed=shot + i)) for i, c in enumerate(sxr_series.CAMERAS)}
    cams["tangential_cam"] = (("ch", "time"), camera(seed=9))
    return xr.Dataset(cams, coords={"time": T, "ch": np.arange(18)})


def test_fetch_counts_and_keeps_the_plasma_window():
    table, report = sxr_series.fetch([1, 2, 3, 4], workers=2, open_group=fake_open, say=lambda line: None)
    assert report["shots_with_data"] == 2 and report["absent"] == 2 and report["failed"] == 0
    assert set(table["camera"]) == {0, 1}  # the tangential camera is not used
    one = table[(table["shot_id"] == 1) & (table["camera"] == 0)]
    # plasma from 10 to 100 ms: bins from 25 ms before (but not before the recording, at -10 ms) to 5 ms after
    assert one["k_ms"].min() == -10 and one["k_ms"].max() == 100 + sxr_series.AFTER_MS


def series_of(shot, level):
    """A 1 ms table for one shot: both cameras, level from the given function of k (ms)."""
    k = np.arange(1, 400)
    frames = [pd.DataFrame({"shot_id": shot, "camera": code, "k_ms": k, "n_chords": 18, "level": level(k),
                            "position": 0.3, "width": 0.1, "peak": 0.1, "fluct": 0.01 * level(k)}) for code in (0, 1)]
    return pd.concat(frames, ignore_index=True)


def test_window_features_are_causal_and_see_a_drop():
    steady = series_of(1, lambda k: np.ones(len(k)))
    crash = series_of(1, lambda k: np.where(k == 205, 0.6, 1.0))  # one bin falls by 40% at 205 ms
    later = series_of(1, lambda k: np.where(k > 200, 9.0, 1.0))  # everything after 200 ms changes
    t_ends = np.array([0.190, 0.200, 0.210])
    a = soft_xrays.shot_features(steady, t_ends, 0.05)
    b = soft_xrays.shot_features(crash, t_ends, 0.05)
    c = soft_xrays.shot_features(later, t_ends, 0.05)
    assert list(a.columns) == soft_xrays.FEATURES and len(soft_xrays.FEATURES) == 46
    assert a["sxr_lo_drop"].tolist() == [0.0, 0.0, 0.0] and abs(b["sxr_lo_drop"].iloc[2] - 0.4) < 1e-9
    assert np.allclose(a.iloc[:2].to_numpy(), c.iloc[:2].to_numpy(), equal_nan=True)  # 190 and 200 ms see no later bin
    assert c["sxr_up_level_rel"].iloc[2] > 0.5 and abs(a["sxr_up_level_rel"].iloc[2]) < 1e-9
    assert soft_xrays.shot_features(steady.iloc[:0], t_ends, 0.05).isna().all().all()  # no soft X-rays: all empty


def test_one_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    windows, shots = experiment.load_tables()
    rng = np.random.default_rng(2)
    series = pd.concat([series_of(int(s), lambda k: 1 + rng.random(len(k))) for s in windows["shot_id"].unique()[::2]])
    table = soft_xrays.build_features(series, windows[["shot_id", "t_end", "t_start"]], say=lambda line: None)
    assert len(table) == len(windows)
    merged = windows.merge(table, on=["shot_id", "t_end"], how="left")
    monkeypatch.setattr(soft_xrays, "load_tables", lambda: (merged, shots))
    metrics = soft_xrays.run(3)
    out = tiny_project / "results" / "soft_xrays" / "sxr" / "seed_3"
    assert (out / "metrics.json").exists() and (out / "test_scores.parquet").exists()
    assert "sxr_lo_level_mean" in metrics["columns"] and not any(c.startswith("has_sxr") for c in metrics["columns"])
    assert set(v3_explore.recipe_columns("baseline")) <= set(soft_xrays.columns())
    assert not (tiny_project / "results" / "v3").exists()


def test_summary_from_saved_runs(monkeypatch, tmp_path):
    from scripts import v3_runs
    times = np.round(np.arange(0.05, 0.40, 0.005), 3)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    os.makedirs(tmp_path / "events")
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "group": "candidate", "other_group": "smooth_end", "gap": 10,
                  "rms": 0.0}).to_csv(tmp_path / "events" / "programme_twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "events" / "programme_twins.csv"))
    pd.DataFrame({"shot_id": [1, 2, 11, 12], "t_quench": [np.nan, np.nan, 0.30, 0.30]}).to_csv(
        tmp_path / "auto_labels" / "per_shot.csv", index=False)
    feats = pd.DataFrame({"shot_id": np.repeat([1, 2, 11, 12], len(times)), "t_end": np.tile(times, 4)})
    feats["sxr_lo_level_mean"] = np.where(feats["shot_id"] == 2, np.nan, 1.0)  # shot 2 has no soft X-rays
    feats["sxr_up_level_mean"] = np.nan
    feats.to_parquet(soft_xrays.table_path(), index=False)

    def save(path, lost_score):
        os.makedirs(path)
        frames = [pd.DataFrame({"shot_id": s, "t_end": times, "score": 0.5}) for s in (1, 2)]
        frames += [pd.DataFrame({"shot_id": s, "t_end": times, "score": lost_score(times)}) for s in (11, 12)]
        pd.concat(frames).to_parquet(os.path.join(path, "test_scores.parquet"), index=False)
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump({"test": {"pauc": 0.3, "catch_rate": 0.4, "false_alarm_rate": 0.05}}, f)

    for seed in range(1, 6):
        save(tmp_path / "v3" / "baseline" / f"seed_{seed}", lambda t: np.where(t > 0.26, 0.9, 0.1))
        save(soft_xrays.run_dir(seed), lambda t: np.where(t > 0.22, 0.9, 0.1))
    out = soft_xrays.summary(load=lambda shot: None)
    assert "PRIMARY: C(50 ms) of S (full + soft X-rays) minus A (full): over splits +1.000" in out
    assert "Resolved and at least +0.03" in out
    assert "pairs with soft X-ray inputs in both shots at that moment (1 pairs)" in out
    assert "present for 100% of lost shots and 50% of twins" in out
    assert (tmp_path / "soft_xrays" / "summary.txt").exists()
    assert fast_magnetics.plan_flags() is None
