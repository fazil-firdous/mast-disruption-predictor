"""Tests for scripts/v3_runs.py: the hysteresis alarm, evidence labels, the matched-pair twin test, the tests'
statistics, the server queue and one end-to-end run on the tiny fake project. No network."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts import v3_runs
from src import alarms, config, evaluate, experiment, server
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_ratio_one_and_hold_k_is_the_v2_alarm():
    rng = np.random.default_rng(0)
    for _ in range(20):
        scores, times = rng.random(40), np.arange(40) * 0.005
        old = alarms.alarm_times(scores, times, evaluate.THRESHOLDS, k=config.K_CONSEC)
        new = v3_runs.alarm_times(scores, times, evaluate.THRESHOLDS, 1.0, config.K_CONSEC)
        assert np.array_equal(np.isnan(old), np.isnan(new)) and np.allclose(old[~np.isnan(old)], new[~np.isnan(new)])


def test_hysteresis_fires_after_the_hold_above_the_lower_threshold():
    scores = np.array([0.1, 0.9, 0.5, 0.5, 0.2, 0.95, 0.96])
    times = np.arange(7) * 0.005
    # upper 0.8, lower 0.48 (ratio 0.6), hold 3: crosses at window 1 and stays >= 0.48 through window 3
    assert v3_runs.alarm_times(scores, times, np.array([0.8]), 0.6, 3)[0] == pytest.approx(times[3])
    # with the v2 rule (two windows in a row >= 0.8) it fires only at window 6
    assert v3_runs.alarm_times(scores, times, np.array([0.8]), 1.0, 2)[0] == pytest.approx(times[6])
    assert np.isnan(v3_runs.alarm_times(scores[:2], times[:2], np.array([0.8]), 0.6, 3)[0])  # too short


def test_relabel_turns_only_training_windows_of_evidence_shots_into_disruptions():
    t = np.arange(0, 0.30, 0.005) + 0.02
    rows = []
    for shot, split in ((1, "train"), (2, "test"), (3, "train")):
        for x in t:
            rows.append({"shot_id": shot, "split": split, "t_end": x, "t_quench": 0.25, "disrupted": False,
                         "t_disrupt": np.nan, "label": 0, "candidate": shot != 3, "near_quench": False})
    out = v3_runs.relabel(pd.DataFrame(rows), {1, 2})
    one, two, three = (out[out["shot_id"] == s] for s in (1, 2, 3))
    assert one["disrupted"].all() and not one["candidate"].any() and (one["t_end"] < 0.25).all()
    assert one["label"].sum() == 6  # the 30 ms before the quench, windows every 5 ms
    assert not two["disrupted"].any() and two["candidate"].all() and len(two) == len(t)  # test shot untouched
    assert not three["disrupted"].any()  # not an evidence shot
    cat = pd.DataFrame({"shot_id": [1, 2, 3], "group": ["candidate", "candidate", "recorded"],
                        "words_vote": [2, 1, 2]})
    assert v3_runs.evidence_shots(cat) == {1}


def test_concordance_compares_twins_over_the_same_programme_time():
    t = np.round(np.arange(0.05, 0.40, 0.005), 3)
    scores = pd.concat([
        pd.DataFrame({"shot_id": 10, "t_end": t, "score": np.where(t > 0.20, 0.9, 0.1)}),  # lost shot rises
        pd.DataFrame({"shot_id": 11, "t_end": t, "score": 0.2}),  # its twin stays low
        pd.DataFrame({"shot_id": 20, "t_end": t, "score": 0.3}),
        pd.DataFrame({"shot_id": 21, "t_end": t, "score": np.where(t > 0.30, 0.99, 0.3)}),  # twin high only late
        pd.DataFrame({"shot_id": 30, "t_end": t, "score": 0.5}),  # its twin is not in the scores
    ])
    twins = pd.DataFrame({"shot_id": [10, 20, 30], "other": [11, 21, 31], "rms": [0.0, 0.05, 0.0]})
    t_quench = {10: 0.30, 20: 0.25, 30: 0.3}
    c, n, c_exact, n_exact = v3_runs.concordance(scores, twins, t_quench, {10: 0.1, 20: None})
    # pair 10: 0.9 > 0.2 -> 1; pair 20: up to 0.245 s both 0.3 -> tie 0.5; pair 30 skipped
    assert (c, n, c_exact, n_exact) == (0.75, 2, 1.0, 1)


def test_tests_use_the_right_seeds_and_holm(monkeypatch):
    rows = []
    for seed in range(1, 51):
        rows += [{"recipe": "baseline", "seed": seed, "pauc": 0.30, "C": 0.70 + 0.001 * (seed % 5),
                  "hyst_test_pauc": 0.31 + 0.001 * (seed % 3), "hyst_test_pauc_present": 0.30},
                 {"recipe": "machine", "seed": seed, "pauc": 0.20, "C": 0.50 + 0.002 * (seed % 4)},
                 {"recipe": "evidence", "seed": seed, "pauc": 0.30 + 0.002 * ((seed % 7) - 3)}]
    result = v3_runs.tests(pd.DataFrame(rows), v3_runs.CONFIRM_SEEDS)
    assert {k: r["n"] for k, r in result.items()} == {"H1a": 30, "H1b": 30, "H2": 30, "H3": 30}
    assert result["H1a"]["mean"] > 0.19 and result["H1a"]["p"] < 0.001
    assert result["H1b"]["p"] < 0.001 and result["H3"]["p"] < 0.01
    assert result["H2"]["p"] > 0.05  # no real change
    assert result["H3"]["p_holm"] >= result["H3"]["p"] and "p_holm" not in result["H1a"]
    text = v3_runs.text({"dev": v3_runs.tests(pd.DataFrame(rows), v3_runs.DEV_SEEDS), "confirm": result,
                         "n_runs": {"baseline": 50}})
    assert "Confirmation, splits 21-50" in text and "H2" in text


def test_v3_jobs_come_after_the_comparisons(monkeypatch):
    monkeypatch.setattr(server, "EXTRA_JOBS", v3_runs.jobs())
    state = server.new_state()
    state["seeds_done"] = list(range(1, config.SEED_SWEEP_MAX + 1))
    state["comparisons_done"] = [[name, seed] for seed in range(1, config.COMPARISON_SEEDS + 1)
                                 for name in config.COMPARISONS]
    assert server.idle_jobs(state, 2) == [("v3-baseline", 1), ("v3-baseline", 2)]
    assert v3_runs.jobs()[:3] == [("v3-baseline", 1), ("v3-machine", 1), ("v3-evidence", 1)]
    state["comparisons_done"] += [list(job) for job in v3_runs.jobs()]
    assert server.idle_jobs(state, 2) == []
    assert v3_runs.recipe_of("v3-evidence") == "evidence"


def test_one_baseline_and_one_evidence_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    windows, shots = experiment.load_tables()
    some = shots.loc[shots["t_quench"].notna(), "shot_id"].iloc[:3]
    catalogue = tiny_project / "catalogue.csv"
    pd.DataFrame({"shot_id": some, "group": "candidate", "words_vote": 2}).to_csv(catalogue, index=False)
    monkeypatch.setattr(v3_runs, "CATALOGUE", str(catalogue))
    base = v3_runs.run("baseline", 3)
    assert set(base["hysteresis"]) >= {"ratio", "hold", "test_pauc", "test_pauc_present"}
    assert 0 <= base["test"]["pauc"] <= 1
    out = tiny_project / "results" / "v3" / "baseline" / "seed_3"
    scores = pd.read_parquet(out / "test_scores.parquet")
    assert len(scores) > 0 and set(scores.columns) == {"shot_id", "t_end", "score"}
    with open(out / "metrics.json") as f:
        assert json.load(f)["seed"] == 3
    v3_runs.run("evidence", 3)
    table = v3_runs.runs(load=lambda shot: None)
    assert sorted(table["recipe"]) == ["baseline", "evidence"]
