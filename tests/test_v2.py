"""Tests for the v2 batch: new signals and features, termination labels in the dataset, alarms by ending, the E14
and E15 recipes, and the relay in parts. No network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config, dataset, evaluate, experiment, extras, features, models, relay
from tests.test_auto_labels import FLAT_REF, STOPPED, SWEEPING, T
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_greenwald_fraction():
    # n_G = Ip[MA] / (pi a^2) 1e20 m^-3: 1 MA and a = 0.5 m give 1.273e20, so 3e19 is 0.236 of the limit
    f = features.greenwald_fraction(np.array([3e19, 3e19]), np.array([1e6, 10e3]), np.array([0.5, 0.5]))
    assert f[0] == pytest.approx(3e19 / (1 / (np.pi * 0.25) * 1e20)) and np.isnan(f[1])  # no plasma: NaN


def test_slope_matches_polyfit_and_change_features():
    rng = np.random.default_rng(0)
    t, x = np.sort(rng.uniform(0, 0.02, 20)), rng.normal(0, 1, 20)
    assert features.slope(t, x) == pytest.approx(np.polyfit(t, x, 1)[0])
    assert np.isnan(features.slope(np.ones(3), x[:3]))
    history = np.full(config.REL_MIN_SAMPLES, 2.0)
    assert features.relative_change(4.0, history) == pytest.approx(np.log10(2))
    assert np.isnan(features.relative_change(4.0, history[:-1]))  # too little history
    assert np.isnan(features.relative_change(-1.0, history)) and np.isnan(features.relative_change(np.nan, history))


def test_mirnov_band_rms_puts_a_sine_in_its_band():
    dt = 2e-6  # 500 kHz, as on MAST
    t_fast = np.arange(1, 1501) * dt  # three 1 ms bins
    x = 3.0 * np.sin(2 * np.pi * 10e3 * t_fast)  # 10 kHz: the tearing-mode band
    bands = extras.band_rms(t_fast, x, np.array([0.001, 0.002, 0.003]), dt)
    assert bands["mirnov_tm"] == pytest.approx(np.full(3, 3.0 / np.sqrt(2)), rel=1e-3)
    for name in ("mirnov_lf", "mirnov_mf", "mirnov_hf"):
        assert np.all(bands[name] < 1e-6)
    x[600] = np.nan  # a gap in the middle bin: that bin is unknown, not guessed
    assert np.isnan(extras.band_rms(t_fast, x, np.array([0.001, 0.002, 0.003]), dt)["mirnov_tm"][1])


def test_cached_extras_without_a_new_column_are_computed_again(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    old = pd.DataFrame({"time": [0.0], "lm_n1": [1.0], "mirnov_rms": [1.0], "ip_ref": [1.0], "sol_current": [1.0]})
    os.makedirs(os.path.dirname(extras.cache_path(5)))
    old.to_parquet(extras.cache_path(5))
    assert not extras.is_current(extras.cache_path(5))
    assert extras.load_extras(5, allow_download=False) is not None  # offline: the old file is still better than none
    pd.DataFrame({c: [0.0] for c in extras.COLUMNS}).to_parquet(extras.cache_path(5))
    assert extras.is_current(extras.cache_path(5))


def ending_extras(sol):
    return pd.DataFrame({"time": T, "ip_ref": FLAT_REF, "sol_current": sol})


def test_add_endings_physics_labels_candidates_and_clear(monkeypatch):
    # 1: noted flat-top drop, 2: un-noted flat-top drop (candidate), 3: un-noted flux exhausted, 4: no quench
    shots = pd.DataFrame({"shot_id": [1, 2, 3, 4], "t_quench": [0.3, 0.3, 0.3, np.nan],
                          "note": [True, False, False, False],
                          "status_both": ["disrupted", "clean", "clean", "clean"]})
    sol = {1: SWEEPING, 2: SWEEPING, 3: STOPPED, 4: SWEEPING}
    monkeypatch.setattr(extras, "load_extras", lambda shot_id, allow_download=True: ending_extras(sol[shot_id]))
    out = dataset.add_endings(shots)
    assert list(out["termination"]) == ["flat_top_drop", "flat_top_drop", "flux_exhausted", "smooth_end"]
    assert list(out["candidate"]) == [False, True, False, False]
    assert list(out["clear"]) == [True, False, False, True]
    assert list(out["status_physics"]) == ["disrupted", "disrupted", "clean", "clean"]
    assert out["t_disrupt_physics"].iloc[1] == 0.3 and np.isnan(out["t_disrupt_physics"].iloc[2])
    assert "status" not in out  # the configured label source is still 'both', so nothing else changes


def test_alarms_by_ending_counts_alarms_before_the_quench():
    rows = []
    # shot 1 recorded, alarms early; 2 candidate, alarms too late; 3 smooth, never alarms
    for shot_id, note, ending, t_q, alarm_at in [(1, True, "flat_top_drop", 0.30, 0.20),
                                                 (2, False, "flat_top_drop", 0.30, 0.298),
                                                 (3, False, "smooth_end", np.nan, None)]:
        for t_end in np.arange(0.10, 0.30, 0.005):
            rows.append({"shot_id": shot_id, "t_end": t_end, "disrupted": note, "t_disrupt": t_q if note else np.nan,
                         "note": note, "termination": ending, "t_quench": t_q,
                         "score": 1.0 if alarm_at is not None and t_end >= alarm_at - 1e-9 else 0.0})
    frame = pd.DataFrame(rows)
    out = evaluate.alarms_by_ending(frame, frame["score"].to_numpy(), 0.5)
    assert out == {"recorded": {"n": 1, "alarms": 1}, "candidate": {"n": 1, "alarms": 0},
                   "smooth_end": {"n": 1, "alarms": 0}}


def test_unbalanced_hgb_for_pu():
    assert models.make_model("hgb", config.PU_PARAMS, balanced=False).class_weight is None
    assert models.make_model("hgb", config.PU_PARAMS).class_weight == "balanced"


@pytest.fixture
def with_endings(tiny_project, monkeypatch):  # noqa: F811
    """The tiny project with endings: no extras, so quench shots end 'unknown' and the rest smoothly; three
    normal shots are then marked as unrecorded candidates (not clear) to see the E15 recipes act on them."""
    windows, labelled = experiment.load_tables()
    monkeypatch.setattr(extras, "load_extras", lambda shot_id, allow_download=True: None)
    labelled = dataset.add_endings(labelled)
    picked = labelled.loc[~labelled["disrupted"], "shot_id"].iloc[:3]
    labelled.loc[labelled["shot_id"].isin(picked), ["candidate", "clear"]] = [True, False]
    monkeypatch.setattr(experiment, "load_tables", lambda: (windows, labelled))
    return tiny_project


def test_e14_and_e15_run(with_endings):
    e14 = experiment.run("E14", model="logreg")
    assert [row["benchmark"] for row in e14] == ["default", "machine", "signal", "signal-machine"]
    e15 = experiment.run("E15", model="logreg")
    assert [row["labels"] for row in e15] == ["clear", "clear-drop-candidates", "clear-physics"]
    # judged on the same clear shots: the three candidates are out of validation for every recipe
    assert len({(row["n_disrupted"], row["n_clean"]) for row in e15}) == 1
    with open(with_endings / "results" / "E15" / "labels_clear" / "metrics.json") as f:
        clear = json.load(f)
    with open(with_endings / "results" / "E15" / "labels_clear-drop-candidates" / "metrics.json") as f:
        dropped = json.load(f)
    assert dropped["n_train_windows"] < clear["n_train_windows"] and dropped["drop_candidates"]
    assert set(clear["val"]["alarms_by_ending"]) <= set(evaluate.ENDING_GROUPS)


def test_relay_parts_are_merged():
    assert relay.is_part("mast-data") and relay.is_part("mast-data-3") and not relay.is_part("other")
    parts = [{"task": "fetch", "scale": "full", "table_source": "live", "part": 0, "requested": 10, "available": 9,
              "failed": 1, "extras": 9, "extras_failed": 0},
             {"task": "fetch", "scale": "full", "table_source": "live", "part": 1, "requested": 10, "available": 10,
              "failed": 0, "extras": 8, "extras_failed": 2}]
    merged = relay.merge_manifests(parts)
    assert (merged["requested"], merged["available"], merged["extras_failed"]) == (20, 19, 2) and "part" not in merged
    assert relay.merge_manifests([parts[0], {**parts[1], "table_source": "report"}])["table_source"] == "report"


def test_workflow_runs_as_many_parts_as_config_says():
    with open(os.path.join(config.REPO_DIR, ".github", "workflows", "fetch_data.yml")) as f:
        text = f.read()
    parts = list(range(config.RELAY_PARTS))
    assert f"'{json.dumps(parts)}'" in text
    assert f"inputs.task == 'fetch' && {config.RELAY_PARTS} || 1" in text


def test_extras_download_stops_starting_shots_after_its_deadline(monkeypatch, tmp_path):
    import time
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    done = []
    monkeypatch.setattr(extras, "load_extras", lambda shot_id: done.append(shot_id) or time.sleep(0.05))
    result = extras.download_extras(list(range(6)), workers=1, deadline=time.time() - 1)  # already past
    # the worker may already have picked up the next shot when the deadline is noticed; nothing after that starts
    assert len(done) <= 2 and result["ok"] == len(done) and sorted(result["failed"]) == list(range(len(done), 6))
    done.clear()
    assert extras.download_extras(list(range(3)), workers=2)["ok"] == 3 and sorted(done) == [0, 1, 2]


def test_blind_sample_hides_everything_but_the_code():
    from scripts import blind_sample
    shots = pd.DataFrame({"shot_id": range(100, 160), "note": [i % 3 == 0 for i in range(60)],
                          "termination": ["flat_top_drop" if i % 2 else "flux_exhausted" for i in range(60)]})
    shots["candidate"] = ~shots["note"] & (shots["termination"] == "flat_top_drop")
    key = blind_sample.draw(shots, 4)
    again = blind_sample.draw(shots, 4)
    assert key.equals(again) and len(key) == 12 and key["group"].value_counts().to_dict() == {
        "candidate": 4, "recorded": 4, "flux_exhausted": 4}
    assert list(key["code"]) == [f"B{i:02d}" for i in range(1, 13)] and key["shot_id"].is_unique
    sheet = blind_sample.answer_sheet(key)
    assert list(sheet["code"]) == list(key["code"])
    assert "shot_id" not in sheet and "group" not in sheet
