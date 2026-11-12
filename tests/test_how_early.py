"""Tests for scripts/how_early.py: the score at a landmark, who is eligible, the cohorts, the summary from made-up
saved runs, and Part B (the 100 ms training horizon: what it trains on, what it skips, its summary). No network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from scripts import how_early, v3_explore, v3_runs
from src import config, dataset, experiment
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)

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


def test_part_b_changes_only_the_horizon(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    seen = {}
    real_fit = v3_explore.fit

    def fit(win, column, recipe, path, info):
        seen[column] = win
        return real_fit(win, column, recipe, path, info)

    monkeypatch.setattr(v3_explore, "fit", fit)
    windows, shots = experiment.load_tables()
    metrics = how_early.run_b("baseline", 3)
    out = tiny_project / "results" / "how_early" / "h100" / "baseline" / "seed_3"
    assert metrics["horizon_ms"] == 100 and metrics["seed"] == 3 and (out / "test_scores.parquet").exists()
    win = seen["split"]
    # the same shots in the same parts as the v3 run of this seed, and the same windows
    assert win.drop_duplicates("shot_id").set_index("shot_id")["split"].to_dict() == dataset.random_split(shots, seed=3).to_dict()
    usual = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(dataset.random_split(shots, seed=3))), shots)
    assert len(win) == len(usual) and (win["t_end"].to_numpy() == usual["t_end"].to_numpy()).all()
    # only the label differs: positive up to 100 ms before a recorded disruption, where the usual label stops at 30
    before = (win["t_disrupt"] - win["t_end"]).to_numpy()
    assert (win["label"].to_numpy() == (before <= 0.1 + 1e-6)).all()
    assert win["label"].sum() > usual["label"].sum() > 0 and (win["label"] >= usual["label"].to_numpy()).all()
    how_early.run_b("baseline")
    assert "split_temporal" in seen and seen["split_temporal"]["label"].sum() == win["label"].sum()
    assert (tiny_project / "results" / "how_early" / "h100" / "temporal" / "baseline" / "metrics.json").exists()
    assert not (tiny_project / "results" / "v3").exists() and not (tiny_project / "results" / "v3_explore").exists()


def test_train_b_skips_finished_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(how_early, "B_SEEDS", range(1, 3))
    done = how_early.b_dir("baseline", 1)
    os.makedirs(done)
    with open(os.path.join(done, "metrics.json"), "w") as f:
        json.dump({"test": {"pauc": 0.01}}, f)
    calls, sent = [], []

    def run_b(recipe, seed, tables):
        calls.append((recipe, seed))
        return {"test": {"pauc": 0.02}}

    monkeypatch.setattr(how_early, "run_b", run_b)
    monkeypatch.setattr(how_early, "part_b", lambda: "in words")
    monkeypatch.setattr(how_early.experiment, "load_tables", lambda: (None, None))
    monkeypatch.setattr(how_early.notify, "send", sent.append)
    how_early.train_b()
    assert calls == [("machine", 1), ("baseline", 2), ("machine", 2), ("baseline", None), ("machine", None)]
    assert len(sent) == 1 and sent[0].endswith("in words")


def test_part_b_summary_from_saved_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    os.makedirs(tmp_path / "events")
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "group": "candidate", "other_group": "smooth_end", "gap": 10,
                  "rms": 0.0}).to_csv(tmp_path / "events" / "programme_twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "events" / "programme_twins.csv"))
    pd.DataFrame({"shot_id": [1, 2, 11, 12], "t_quench": [np.nan, np.nan, 0.30, 0.30]}).to_csv(
        tmp_path / "auto_labels" / "per_shot.csv", index=False)

    def save(path, lost_score, pauc):
        os.makedirs(path)
        frames = [pd.DataFrame({"shot_id": s, "t_end": T, "score": 0.5}) for s in (1, 2)]
        frames += [pd.DataFrame({"shot_id": s, "t_end": T, "score": lost_score(T)}) for s in (11, 12)]
        pd.concat(frames).to_parquet(os.path.join(path, "test_scores.parquet"), index=False)
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump({"test": {"pauc": pauc, "catch_rate": 0.4, "false_alarm_rate": 0.05}}, f)

    late = lambda t: np.where(t > 0.24, 0.9, 0.1)  # rises 60 ms before the loss: the 30 ms-horizon model
    early = lambda t: np.where(t > 0.18, 0.9, 0.1)  # rises 120 ms before: the 100 ms-horizon model
    flat = lambda t: 0.5 + 0 * t
    for seed in range(1, 7):
        save(tmp_path / "v3" / "baseline" / f"seed_{seed}", late, 0.03)
        save(tmp_path / "v3" / "machine" / f"seed_{seed}", flat, 0.01)
        save(how_early.b_dir("baseline", seed), early, 0.02)
        save(how_early.b_dir("machine", seed), flat, 0.01)
    save(tmp_path / "v3" / "baseline" / "seed_21", early, 0.09)  # a confirmation split: not part of Part B
    save(v3_explore.temporal_dir("baseline"), late, 0.03)
    save(how_early.b_dir("baseline"), early, 0.02)
    save(how_early.b_dir("machine"), flat, 0.01)
    text = how_early.part_b(load=lambda shot: None)
    assert ("PRIMARY: full model, C(100 ms) with the 100 ms horizon minus with the 30 ms horizon: over splits +1.000"
            in text and "6 splits" in text and "Over training sets this is resolved" in text)
    line = next(l for l in text.splitlines() if l.startswith("- 100 ms (2 pairs per split)"))
    assert line.split(": ", 1)[1].split(" | ")[0].startswith("0.000") and " | 1.000" in line and "+0.500" in line
    line = next(l for l in text.splitlines() if l.startswith("- 50 ms (2 pairs per split)"))
    assert "+0.000" in line  # both horizons separate the pair at 50 ms
    block = text.split("Machine-side model per lead")[1].split("Usual metrics")[0]
    assert block.count("0.500") >= 8 and block.count("+0.000") >= 4  # the same flat scores with either horizon
    assert "- full, 30 ms: 0.030, 40.0%, 5.0% (6 splits)" in text and "- full, 100 ms: 0.020, 40.0%, 5.0% (6 splits)" in text
    assert "- full, 100 ms minus 30 ms, paired over 6 splits: pAUC -0.010" in text  # seed 21 is not in it
    assert "- machine-side, 100 ms minus 30 ms, paired over 6 splits: pAUC +0.000" in text
    chrono = text.split("Chronological split")[1]
    assert "- 100 ms (2 pairs): 0.000" in chrono and "+1.000" in chrono
    with open(tmp_path / "how_early" / "part_b.txt") as f:
        assert f.read().strip() == text
