"""Tests for src/evaluate.py, src/models.py and a tiny end-to-end run of src/experiment.py."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import config, dataset, evaluate, experiment, models, notify
from tests.test_dataset import fake_shots


def test_outcomes_and_metrics():
    shots = pd.DataFrame({"disrupted": [True, True, True, False, False],
                          "t_disrupt": [0.2, 0.2, 0.2, np.nan, np.nan]})
    t_alarm = np.array([0.18, 0.198, np.nan, 0.1, np.nan])
    result, warning = evaluate.outcomes(shots, t_alarm)
    assert list(result) == ["caught", "tardy", "missed", "false_alarm", "quiet"]
    m = evaluate.shot_metrics(result, warning)
    assert m["catch_rate"] == pytest.approx(1 / 3)
    assert m["false_alarm_rate"] == pytest.approx(0.5)
    assert m["warning_median_ms"] == pytest.approx(20)


def test_choose_threshold_respects_false_alarm_limit():
    curve = pd.DataFrame({"threshold": [0.1, 0.5, 0.9], "false_alarm_rate": [0.3, 0.04, 0.0],
                          "catch_rate": [0.9, 0.6, 0.2]})
    assert evaluate.choose_threshold(curve, 0.05) == 0.5


def test_bootstrap_ci_contains_point_estimate():
    result = np.array(["caught"] * 30 + ["missed"] * 10 + ["quiet"] * 45 + ["false_alarm"] * 5, dtype=object)
    warning = np.where(result == "caught", 15.0, np.nan)
    ci = evaluate.bootstrap_ci(result, warning, n=200)
    assert ci["catch_rate"][0] <= 0.75 <= ci["catch_rate"][1]
    assert ci["false_alarm_rate"][0] <= 0.1 <= ci["false_alarm_rate"][1]


def test_baseline_fires_on_fast_current_drop():
    X = pd.DataFrame({"ip_abs_slope": [0.0] * 99 + [-1e9], "prad_over_pnbi": [0.1] * 100})
    y = np.zeros(100)
    model = models.make_model("baseline", {"slope_quantile": 0.01, "ratio_quantile": 1.0}).fit(X, y)
    assert models.scores(model, X)[-1] == 1.0
    assert models.scores(model, X)[:99].sum() == 0


def test_grids_are_small():
    for name in models.GRIDS:
        assert 1 <= len(models.grid(name)) <= 12


@pytest.fixture
def tiny_project(monkeypatch, tmp_path):
    shots, table = fake_shots(30, 30)
    monkeypatch.setattr(dataset, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(experiment, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(config, "N_BOOTSTRAP", 50)
    monkeypatch.setattr(models, "GRIDS", {**models.GRIDS, "logreg": {"C": [1.0]}})
    labelled = dataset.label_shots(list(table["shot_id"]), table)
    labelled["split"] = labelled["shot_id"].map(dataset.random_split(labelled))
    labelled["split_temporal"] = labelled["shot_id"].map(dataset.temporal_split(labelled))
    windows = dataset.build_windows(labelled).merge(labelled[["shot_id", "split", "split_temporal"]], on="shot_id")
    monkeypatch.setattr(experiment, "load_tables", lambda: (windows, labelled))
    return tmp_path


def test_end_to_end_logistic_regression(tiny_project):
    metrics = experiment.run("E1")
    out = tiny_project / "results" / "E1"
    for name in ["metrics.json", "config.json", "tradeoff.png", "importance.png"]:
        assert (out / name).exists(), name
    with open(out / "metrics.json") as f:
        saved = json.load(f)
    assert saved["test"]["n_disrupted"] > 0
    assert 0 <= metrics["test"]["catch_rate"] <= 1
    # the synthetic quench is easy, and at least the trade-off curve must be computed on the test shots
    assert len(pd.read_csv(out / "tradeoff_test.csv")) == config.N_THRESHOLDS


def test_baseline_end_to_end(tiny_project):
    metrics = experiment.run("E0")
    assert "test" in metrics and metrics["model"] == "baseline"


def test_notify_without_credentials_does_not_fail(monkeypatch, tmp_path):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.chdir(tmp_path)
    assert notify.send("hello") is False
    assert notify.send_file(os.path.join(tmp_path, "missing.png")) is False


def test_choose_threshold_falls_back_to_fewest_false_alarms():
    curve = pd.DataFrame({"threshold": [0.0, 0.5, 1.0], "false_alarm_rate": [1.0, 0.2, 0.2],
                          "catch_rate": [1.0, 0.3, 0.3]})
    threshold = evaluate.choose_threshold(curve, 0.05)
    assert threshold == 0.5
    assert not evaluate.meets_limit(0.2)
