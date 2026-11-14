"""Tests for scripts/drift_oof.py: folds keep linked shots together and are balanced, the splits are 60/20/20, the
two-level bootstrap, the heating match, one run end to end on the tiny fake project, and the summary from saved
runs. No network."""
import os

import numpy as np
import pandas as pd

from scripts import drift, drift_oof
from src import config, experiment
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_folds_keep_linked_shots_together_and_balance():
    pairs = pd.DataFrame({"pop": ["A", "B", "B"], "E": [10, 20, 21], "F": [11, 21, 22], "L": [12, -1, -1]})
    groups = drift_oof.shot_groups(pairs)
    assert groups[10] == groups[11] == groups[12] and groups[20] == groups[21] == groups[22]
    shots = list(range(1, 101)) + [10, 11, 12, 20, 21, 22]
    for repeat in (1, 2):
        fold = drift_oof.folds(sorted(set(shots)), groups, repeat)
        assert fold[10] == fold[11] == fold[12] and fold[20] == fold[21] == fold[22]
        sizes = np.bincount(list(fold.values()), minlength=5)
        assert sizes.max() - sizes.min() <= 3
    a, b = drift_oof.folds(sorted(set(shots)), groups, 1), drift_oof.folds(sorted(set(shots)), groups, 2)
    assert a != b  # each repetition its own assignment
    split = drift_oof.split_for(a, 0, 1)
    counts = split.value_counts(normalize=True)
    assert abs(counts["test"] - 0.2) < 0.05 and abs(counts["val"] - 0.2) < 0.05 and abs(counts["train"] - 0.6) < 0.05
    assert set(split[split == "test"].index) == {s for s, k in a.items() if k == 0}


def pairs_frame(n_a=60, n_b=150):
    p = pd.DataFrame({"pop": ["A"] * n_a + ["B"] * n_b, "E": np.arange(n_a + n_b) * 2, "F": np.arange(n_a + n_b) * 2 + 1,
                      "L": -1})
    return drift.with_times(p, {}, {})


def test_two_level_bootstrap_and_heating_match():
    rng = np.random.default_rng(1)
    pairs = pairs_frame()
    w = np.vstack([np.concatenate([rng.random(60) < 0.8, rng.random(150) < 0.5]).astype(float) for _ in range(4)])
    d = drift_oof.two_level(w, pairs, n_boot=300)
    assert d[1] > 0 and 0.15 < d[0] < 0.45
    flat = drift_oof.two_level(np.full_like(w, 0.5), pairs, n_boot=100)
    assert abs(flat[0]) < 1e-12 and flat[1] <= 0 <= flat[2]
    means = pd.DataFrame({"E": [1e4, 1e6, 1e6, np.nan], "F": [2e4, 1.05e6, 1.5e6, 1e6]})
    assert drift_oof.beam_matched(pairs.iloc[:4], means).tolist() == [True, True, False, False]


def test_one_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    windows, shots = experiment.load_tables()
    ids = sorted(shots["shot_id"])
    pairs = pd.DataFrame({"pop": ["B"], "E": [ids[0]], "F": [ids[1]], "L": [-1]})
    metrics = drift_oof.run("baseline", 1, 0, (windows, shots), drift_oof.shot_groups(pairs))
    out = tiny_project / "results" / "drift_oof" / "baseline" / "rep_1" / "fold_0"
    assert (out / "metrics.json").exists() and (out / "test_scores.parquet").exists()
    scored = set(pd.read_parquet(out / "test_scores.parquet")["shot_id"])
    fold = drift_oof.folds(shots["shot_id"], drift_oof.shot_groups(pairs), 1)
    assert scored <= {s for s, k in fold.items() if k == 0}  # only this fold's shots are scored
    assert metrics["repeat"] == 1 and not (tiny_project / "results" / "v3").exists()


def test_summary_from_saved_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    pairs = pairs_frame(20, 40)
    pairs["t0"], pairs["t1"] = 0.06, 0.30
    os.makedirs(drift.folder())
    pairs.to_csv(os.path.join(drift.folder(), "pairs.csv"), index=False)
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [1], "t_quench": [np.nan]}).to_csv(tmp_path / "auto_labels" / "per_shot.csv", index=False)
    t = np.round(np.arange(0.05, 0.36, 0.005), 3)
    pd.DataFrame({"shot_id": np.repeat(pairs["F"], len(t)), "t_end": np.tile(t, len(pairs))}).to_parquet(
        tmp_path / "windows.parquet", index=False)
    for rep in (1, 2, 3):
        for k in range(drift_oof.FOLDS):
            path = drift_oof.run_dir("baseline", rep, k)
            os.makedirs(path)
            mine = pairs.iloc[k::drift_oof.FOLDS]
            rows = []
            for row in mine.itertuples():  # A: the later shot higher; B: equal
                rows += [pd.DataFrame({"shot_id": row.E, "t_end": t, "score": 0.3}),
                         pd.DataFrame({"shot_id": row.F, "t_end": t, "score": 0.5 if row.pop == "A" else 0.3})]
            pd.concat(rows).to_parquet(os.path.join(path, "test_scores.parquet"), index=False)
            with open(os.path.join(path, "metrics.json"), "w") as f:
                f.write('{"test": {"pauc": 0.3}}')
    pd.DataFrame({"E": np.full(len(pairs), 1e6), "F": np.full(len(pairs), 1e6)}).to_csv(
        os.path.join(drift_oof.folder(), "beam_means.csv"), index=False)
    monkeypatch.setattr(drift_oof.data, "load_shot_table", lambda: pd.DataFrame({"shot_id": pairs["F"], "campaign": "M7"}))
    out = drift_oof.summary()
    assert "PRIMARY (full model, 3 repetitions): Delta = P_A - P_B = +0.500" in out
    assert "Resolved, positive and at least 0.03" in out
    assert "KEY SECONDARY, heating-matched pairs (A 100%, B 100% of pairs): Delta +0.500" in out
    assert (tmp_path / "drift_oof" / "summary.txt").exists()
