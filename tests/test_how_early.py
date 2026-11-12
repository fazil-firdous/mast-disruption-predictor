"""Tests for scripts/how_early.py: the score at a landmark, who is eligible, the cohorts, and the summary from
made-up saved runs. No network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from scripts import how_early, v3_runs
from src import config

T = np.round(np.arange(0.05, 0.40, 0.005), 3)


def test_score_at_takes_the_latest_window_within_one_stride():
    times, scores = np.array([0.100, 0.105, 0.110]), np.array([1.0, 2.0, 3.0])
    assert how_early.score_at(times, scores, 0.108) == 2.0  # the window ending at 0.105
    assert how_early.score_at(times, scores, 0.110) == 3.0  # ending exactly at the landmark counts
    assert np.isnan(how_early.score_at(times, scores, 0.120))  # the shot's last window is two strides back: ended
    assert np.isnan(how_early.score_at(times, scores, 0.090))  # not started


def test_landmark_wins_leave_out_a_twin_that_has_ended_and_landmarks_before_the_flat_top():
    lost = pd.DataFrame({"shot_id": 2, "t_end": T, "score": np.where(T > 0.24, 0.9, 0.1)})  # rises 60 ms before 0.30
    twin = pd.DataFrame({"shot_id": 1, "t_end": T, "score": 0.5})
    short = pd.DataFrame({"shot_id": 3, "t_end": T[T <= 0.26], "score": 0.5})  # this twin ended at 0.26 s
    scores = pd.concat([lost, twin, short])
    pairs = pd.DataFrame({"shot_id": [2, 2], "other": [1, 3], "ref": [2, 2], "t_loss": [0.30, 0.30],
                          "t_start": [0.15, np.nan]})
    wins = how_early.landmark_wins(scores, pairs).set_index(["other", "lead_ms"])["win"]
    assert wins[(1, 10)] == 1.0 and wins[(1, 30)] == 1.0 and wins[(1, 50)] == 1.0  # 0.9 against 0.5
    assert wins[(1, 100)] == 0.0  # at 0.20 s the lost shot is still at 0.1
    assert (1, 200) not in wins.index  # 0.10 s is before this pair's flat-top start
    assert (3, 10) not in wins.index and (3, 30) not in wins.index  # the twin has no window at 0.29 or 0.27 s
    assert wins[(3, 50)] == 1.0 and wins[(3, 200)] == 0.0  # it has at 0.25 s; no flat-top start known for this pair


def test_summary_from_saved_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    os.makedirs(tmp_path / "events")
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "group": "candidate", "other_group": "smooth_end", "gap": 10,
                  "rms": 0.0}).to_csv(tmp_path / "events" / "programme_twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "events" / "programme_twins.csv"))
    pd.DataFrame({"shot_id": [1, 2, 11, 12], "t_quench": [np.nan, np.nan, 0.30, 0.30]}).to_csv(
        tmp_path / "auto_labels" / "per_shot.csv", index=False)
    for recipe, lost_score in (("baseline", lambda t: np.where(t > 0.24, 0.9, 0.1)), ("machine", lambda t: 0.5 + 0 * t)):
        for seed in range(1, 7):
            path = tmp_path / "v3" / recipe / f"seed_{seed}"
            os.makedirs(path)
            frames = [pd.DataFrame({"shot_id": s, "t_end": T, "score": 0.5}) for s in (1, 2)]
            frames += [pd.DataFrame({"shot_id": s, "t_end": T, "score": lost_score(T)}) for s in (11, 12)]
            pd.concat(frames).to_parquet(path / "test_scores.parquet", index=False)
            with open(path / "metrics.json", "w") as f:
                json.dump({"seed": seed, "test": {}}, f)
    random, temporal = how_early.collect(load=lambda shot: None)
    main = random[random["kind"] == "main"]
    assert set(main["lead_ms"]) == set(how_early.LEADS_MS) and main["cohort"].all() and main["cohort_all"].all()
    assert how_early.over_splits(main, 50)["baseline"][0] == 1.0 and how_early.over_splits(main, 100)["baseline"][0] == 0.0
    assert how_early.over_splits(main, 50)["machine"][0] == 0.5  # same score as the twin: ties
    d = how_early.split_difference(main, 50)
    assert d[0] == pytest.approx(0.5) and d[3] == 6
    text = how_early.text(random, temporal)
    assert "PRIMARY: full minus machine-side at 50 ms on the common cohort: over splits +0.500" in text
    line = next(l for l in text.splitlines() if l.startswith("- 100 ms (2 pairs per split)"))
    assert "machine-side 0.500" in line and "full 0.000" in line and "full minus machine-side -0.500" in line
    assert "- no triplets" in text and "- no runs" in text
