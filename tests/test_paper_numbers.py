"""Tests for scripts/paper_numbers.py and the p-value helpers in src/stats.py, on made-up runs. No network."""
import json
import os

import numpy as np
import pytest
from scipy import stats as sstats

from scripts import paper_numbers
from src import config, stats


def test_corrected_p_matches_the_corrected_interval():
    changes = np.array([0.02, 0.05, 0.01, 0.04, 0.03, 0.06, 0.00, 0.02])
    mean, low, high = stats.corrected_interval(changes)
    train, _, test = config.SPLIT_FRACTIONS
    se = np.std(changes, ddof=1) * np.sqrt(1 / len(changes) + test / train)
    assert stats.corrected_p(changes) == pytest.approx(2 * sstats.t.sf(mean / se, len(changes) - 1))
    # the range just touches 0 exactly when p is 0.05
    assert (low > 0) == (stats.corrected_p(changes) < 0.05)
    assert stats.corrected_p(changes[:3]) is None
    assert stats.corrected_p(np.zeros(6)) == 1.0


def test_holm():
    assert stats.holm([0.01, 0.04, None, 0.03]) == pytest.approx([0.03, 0.06, None, 0.06], nan_ok=True)
    assert stats.holm([]) == []


def fake_run(folder, seed, catch, false_alarm, pr_auc, endings):
    os.makedirs(os.path.join(folder, f"seed_{seed}"))
    test = {"catch_rate": catch, "false_alarm_rate": false_alarm, "window": {"pr_auc": pr_auc},
            "alarms_by_ending": {g: {"n": 100, "alarms": a} for g, a in endings.items()}}
    with open(os.path.join(folder, f"seed_{seed}", "metrics.json"), "w") as f:
        json.dump({"test": test, "val": {"positive_share": 0.01}}, f)


def test_tables_from_fake_runs(tmp_path):
    rng = np.random.default_rng(0)
    e10 = tmp_path / "E10"
    name = next(iter(config.COMPARISONS))
    for seed in range(1, 11):
        noise = rng.normal(0, 0.005)
        fake_run(str(e10), seed, 0.14 + noise, 0.05, 0.10,
                 {"recorded": 30, "candidate": 20, "flux_exhausted": 10, "smooth_end": 5})
        fake_run(str(e10 / "comparisons" / name), seed, 0.04 + noise, 0.05, 0.05,
                 {"recorded": 10, "candidate": 10, "flux_exhausted": 10, "smooth_end": 5})
    sets = paper_numbers.run_sets(str(tmp_path))
    skill = paper_numbers.skill_table(sets)
    assert skill["E10"]["n"] == 10
    assert skill["E10"]["skill"]["mean"] == pytest.approx(0.09, abs=0.01) and skill["E10"]["skill"]["low"] > 0
    comparisons = paper_numbers.comparison_table(sets)
    change = comparisons[name]["catch_rate"]
    assert change["mean"] == pytest.approx(-0.10) and change["p_holm"] < 0.05
    assert comparisons[name]["false_alarm_rate"]["p"] == 1.0  # no change at all
    endings = paper_numbers.ending_table(sets)
    # candidates 20% between recorded 30% and flux-exhausted 10%: half of them alarm like recorded ones
    assert endings["E10"]["mixture"]["flux_exhausted"]["pooled"] == pytest.approx(0.5)
    assert endings["E10"]["mixture"]["smooth_end"]["pooled"] == pytest.approx(0.6)
    assert endings[name]["mixture"]["flux_exhausted"]["pooled"] is None  # recorded not above the reference
    text = paper_numbers.markdown(skill, comparisons, endings)
    assert "| E10 | 10 |" in text and name in text


def test_mixture_share_needs_recorded_above_reference():
    assert paper_numbers.mixture_share({"recorded": 0.5, "candidate": 0.3, "smooth_end": 0.1}, "smooth_end") == \
        pytest.approx(0.5)
    assert paper_numbers.mixture_share({"recorded": 0.1, "candidate": 0.3, "smooth_end": 0.1}, "smooth_end") is None
    assert paper_numbers.mixture_share({"recorded": 0.5, "smooth_end": 0.1}, "smooth_end") is None
