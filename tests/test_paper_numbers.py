"""Tests for scripts/paper_numbers.py and the p-value helpers in src/stats.py, on made-up runs. No network."""
import json
import os

import numpy as np
import pandas as pd
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
    # a trade-off curve through (false_alarm, catch) on a straight line from (0, 0) to (1, 1)
    fa = np.linspace(0, 1, 21)
    curve = pd.DataFrame({"threshold": 1 - fa, "false_alarm_rate": fa,
                          "catch_rate": np.minimum(1, fa * catch / false_alarm)})
    curve.to_csv(os.path.join(folder, f"seed_{seed}", "tradeoff_test.csv"), index=False)
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
    assert skill["E10"]["pauc_skill"]["mean"] > 0  # the fake curve lies above the diagonal
    comparisons = paper_numbers.comparison_table(sets)
    change = comparisons[name]["catch_rate"]
    assert change["mean"] == pytest.approx(-0.10) and change["p_holm"] < 0.05
    assert comparisons[name]["false_alarm_rate"]["p"] == 1.0  # no change at all
    endings = paper_numbers.ending_table(sets)
    # candidates 20% between recorded 30% and flux-exhausted 10%: half of them alarm like recorded ones
    assert endings["E10"]["mixture"]["flux_exhausted"]["pooled"] == pytest.approx(0.5)
    assert endings["E10"]["mixture"]["smooth_end"]["pooled"] == pytest.approx(0.6)
    assert endings["E10"]["recorded_skill"]["mean"] == pytest.approx(0.25)  # 30% recorded vs 5% smooth
    if paper_numbers.labels_call_candidates_disruptions(name):
        assert endings[name]["mixture"] is None
    else:
        assert endings[name]["mixture"]["flux_exhausted"]["pooled"] is None  # recorded not above the reference
    text = paper_numbers.markdown(skill, comparisons, endings)
    assert "| E10 | 10 |" in text and name in text


def test_mixture_share_needs_recorded_above_reference():
    assert paper_numbers.mixture_share({"recorded": 0.5, "candidate": 0.3, "smooth_end": 0.1}, "smooth_end") == \
        pytest.approx(0.5)
    assert paper_numbers.mixture_share({"recorded": 0.1, "candidate": 0.3, "smooth_end": 0.1}, "smooth_end") is None
    assert paper_numbers.mixture_share({"recorded": 0.5, "smooth_end": 0.1}, "smooth_end") is None


def test_mixture_left_out_where_labels_call_candidates_disruptions():
    assert paper_numbers.labels_call_candidates_disruptions("signal_labels")
    assert paper_numbers.labels_call_candidates_disruptions("clear_physics")
    assert not paper_numbers.labels_call_candidates_disruptions("clear_drop_candidates")
    assert not paper_numbers.labels_call_candidates_disruptions("E10")


def test_partial_auc():
    diagonal = pd.DataFrame({"false_alarm_rate": [0.0, 0.05, 0.1, 0.5], "catch_rate": [0.0, 0.05, 0.1, 0.5]})
    assert paper_numbers.partial_auc(diagonal) == pytest.approx(0.05)  # random alarms
    perfect = pd.DataFrame({"false_alarm_rate": [0.0, 0.2], "catch_rate": [1.0, 1.0]})
    assert paper_numbers.partial_auc(perfect) == pytest.approx(1.0)
    # two thresholds at the same false-alarm rate: the better catch rate counts; held flat to the cutoff
    steps = pd.DataFrame({"false_alarm_rate": [0.0, 0.04, 0.04, 0.3], "catch_rate": [0.0, 0.1, 0.2, 0.6]})
    assert paper_numbers.partial_auc(steps) == pytest.approx((0.04 * 0.2 / 2 + 0.06 * 0.2) / 0.1)
    assert np.isnan(paper_numbers.partial_auc(None))


def test_candidate_notes_counts_words_by_ending():
    from scripts import candidate_notes
    table = pd.DataFrame({
        "shot_id": [1, 2, 3, 4, 5],
        "note": [True, False, False, False, False],
        "t_quench": [0.3, 0.3, 0.3, np.nan, 0.3],
        "termination": ["flat_top_drop", "flat_top_drop", "flat_top_drop", "smooth_end", "flux_exhausted"],
        "shot_postshot_comment": ["DISRUPTS AT 300MS (FA TRIP)", "P1 tripped at 250ms", "locked mode, VDE", None,
                                  "good shot"]})
    counts, groups, flags = candidate_notes.by_ending(table)
    assert groups == ["recorded", "candidate", "candidate", "smooth_end", "flux_exhausted"]
    assert counts.at["recorded", "hardware trip"] == 1
    assert counts.at["candidate", "shots"] == 2 and counts.at["candidate", "hardware trip"] == 1
    assert counts.at["candidate", "VDE"] == 1 and counts.at["candidate", "locked mode"] == 1
    assert counts.at["candidate", "any loss event"] == 1
    assert counts.at["smooth_end", "any comment"] == 0 and counts.at["flux_exhausted", "hardware trip"] == 0
    assert not candidate_notes.word_flags(pd.Series(["Good H-mode shot"]))["any loss event"].iloc[0]
    lines = candidate_notes.examples(table, groups, flags)
    assert any("P1 tripped" in line for line in lines)


def test_candidate_notes_skips_traps_found_by_reading():
    from scripts import candidate_notes
    texts = pd.Series(["Vertical kicks can be seen", "NTM does not terminate the shot", "Good MHD", "Good H-mode",
                       "VDE at 0.22s", "IRE at 150ms kills the shot.", "Loss of vertical position control"])
    flags = candidate_notes.word_flags(texts)
    assert flags["any loss event"].tolist() == [False, False, False, False, True, True, True]
