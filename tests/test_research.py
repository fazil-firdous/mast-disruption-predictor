"""Tests for src/conformal.py, src/pu.py and the premature-alarm metric."""
import numpy as np
import pandas as pd
import pytest

from src import config, conformal, evaluate, experiment, pu
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_conformal_threshold_rank():
    levels = np.arange(1, 20) / 20  # 19 clean shots
    threshold = conformal.conformal_threshold(levels, 0.1)
    # r = ceil(20 * 0.9) = 18, so the 18th smallest (0.9) must not alarm and nothing above it exists
    assert threshold > 0.9
    assert (levels >= threshold).sum() == 1  # only 0.95 alarms: 1 of 19, about alpha
    assert conformal.conformal_threshold(levels[:5], 0.05) == np.inf


def test_conformal_guarantee_on_exchangeable_shots():
    rng = np.random.default_rng(0)
    alpha, misses = 0.1, []
    for _ in range(400):
        cal, new = rng.random(50), rng.random()
        misses.append(new >= conformal.conformal_threshold(cal, alpha))
    assert np.mean(misses) <= alpha + 0.03


def test_min_calibration_shots():
    assert conformal.min_calibration_shots(0.05) == 19
    assert np.isfinite(conformal.conformal_threshold(np.random.random(19), 0.05))


def test_shot_alarm_levels_match_alarm_rule():
    windows = pd.DataFrame({"shot_id": [1] * 4 + [2] * 4, "t_end": [0.1, 0.2, 0.3, 0.4] * 2,
                            "disrupted": [True] * 4 + [False] * 4})
    scores = np.array([0.2, 0.8, 0.6, 0.9, 0.9, 0.1, 0.9, 0.1])
    levels = conformal.shot_alarm_levels(windows, scores, k=2).set_index("shot_id")["level"]
    assert levels[1] == 0.6 and levels[2] == 0.1


def test_pu_weights():
    assert pu.unlabelled_weights(np.array([0.0]), 0.5)[0] == 0.0
    # with c = 0.5, g = 0.25 gives (0.5 / 0.5) * (0.25 / 0.75) = 1/3
    assert pu.unlabelled_weights(np.array([0.25]), 0.5)[0] == pytest.approx(1 / 3)
    assert pu.unlabelled_weights(np.array([0.99]), 0.5)[0] == 1.0


def test_premature_rate():
    result = np.array(["caught", "caught", "missed"], dtype=object)
    warning = np.array([config.PREMATURE_WARNING_MS + 50, 20.0, np.nan])
    m = evaluate.shot_metrics(result, warning)
    assert m["premature_rate"] == pytest.approx(1 / 3)
    assert m["catch_rate"] == pytest.approx(2 / 3)


def test_conformal_and_pu_experiments_run(tiny_project):  # noqa: F811
    rows = experiment.run("E8", model="logreg")
    assert {r["method"] for r in rows} == {"conformal", "tuned on validation"}
    summary = experiment.run("E9")
    assert 0 < summary["label_frequency_c"] <= 1
    assert {r["method"] for r in summary["rows"]} == {"standard", "pu"}
