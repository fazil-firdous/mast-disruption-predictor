"""Tests for scripts/fast_magnetics.py: the 5 ms summaries are the 20 ms features with a shorter window and use no
later sample, the recipes' inputs, the feature table, and one run end to end on the tiny fake project. No network."""
import numpy as np
import pandas as pd

from scripts import fast_magnetics, v3_explore
from src import config, experiment, extras, features
from tests.test_dataset import make_shot
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def signals(seed=0):
    shot = make_shot("quench", t_quench=0.3, seed=seed)
    rng = np.random.default_rng(seed)
    for column in ("lm_n1", "mirnov_rms", "mirnov_lf", "mirnov_tm", "mirnov_mf", "mirnov_hf"):
        shot[column] = np.abs(rng.normal(1.0, 0.3, len(shot))) * (1 + 5 * (shot["time"] > 0.25))
    return features.prepare_signals(shot)


def test_20ms_summaries_reproduce_the_stored_features():
    sig = signals()
    t_start = 0.05
    for t_end in (0.12, 0.2, 0.26, 0.29):
        ours = fast_magnetics.window_summaries(sig, t_end, t_start, config.WINDOW_MS)
        stored = features.window_features(sig, t_end, t_start)
        for key, value in ours.items():
            assert np.isclose(value, stored[key], equal_nan=True), key


def test_5ms_summaries_use_no_later_sample():
    sig = signals()
    before = fast_magnetics.window_summaries(sig, 0.2, 0.05, 5)
    changed = sig.copy()
    changed.loc[changed["time"] > 0.2 + 1e-9, "lm"] = 1e6  # the future, wildly different
    after = fast_magnetics.window_summaries(changed, 0.2, 0.05, 5)
    assert before == after or all(np.isclose(before[k], after[k], equal_nan=True) for k in before)
    t = sig["time"].to_numpy()
    in5 = (t <= 0.2 + 1e-9) & (t > 0.195 + 1e-9)
    assert np.isclose(before["lm_mean"], sig["lm"].to_numpy()[in5].mean())  # 5 samples on the 1 ms grid


def test_growth_and_recipes():
    sig = signals()
    row = fast_magnetics.shot_features(sig, [0.28], 0.05).iloc[0]
    assert row["lm_growth"] > 0  # the signal jumped at 0.25 s: the last 5 ms are above the 20 ms mean
    assert list(fast_magnetics.shot_features(sig, [0.1, 0.2], 0.05).columns) == fast_magnetics.FIVE + fast_magnetics.GROWTH
    base = v3_explore.recipe_columns("baseline")
    assert fast_magnetics.columns_of("mag20_5") == base + fast_magnetics.FIVE
    b = fast_magnetics.columns_of("mag5")
    assert not set(fast_magnetics.REPLACED) & set(b) and set(fast_magnetics.FIVE) <= set(b)
    assert "lm_last" in b and "lm_slope_long" in b  # the last value and the 100 ms slope stay
    assert set(fast_magnetics.REPLACED) <= set(base)  # what is replaced is really in the full model


def test_one_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    windows, shots = experiment.load_tables()
    rng = np.random.default_rng(1)
    for column in fast_magnetics.REPLACED:
        if column not in windows:
            windows[column] = np.nan

    def load(shot):
        sig = features.prepare_signals(extras.with_extras(fast_magnetics.load_shot(shot), shot))
        for name in fast_magnetics.MAG:
            sig[name] = np.abs(rng.normal(1.0, 0.2, len(sig)))
        return sig

    monkeypatch.setattr(fast_magnetics, "load_shot", __import__("src.dataset", fromlist=["load_shot"]).load_shot)
    table, worst = fast_magnetics.build_features(load=load, say=lambda line: None, windows=windows)
    assert len(table) == len(windows) and table["lm_mean_5ms"].notna().mean() > 0.9
    monkeypatch.setattr(fast_magnetics, "load_tables", lambda: (fast_magnetics.with_fast(windows, table), shots))
    metrics = fast_magnetics.run("mag20_5", 3)
    out = tiny_project / "results" / "fast_magnetics" / "mag20_5" / "seed_3"
    assert (out / "metrics.json").exists() and (out / "test_scores.parquet").exists()
    assert "lm_mean_5ms" in metrics["columns"]
    assert not (tiny_project / "results" / "v3").exists()  # never writes into the pre-registered results


def test_summary_from_saved_runs(monkeypatch, tmp_path):
    import json
    import os
    from scripts import v3_runs
    T = np.round(np.arange(0.05, 0.40, 0.005), 3)
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    os.makedirs(tmp_path / "events")
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "group": "candidate", "other_group": "smooth_end", "gap": 10,
                  "rms": 0.0}).to_csv(tmp_path / "events" / "programme_twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "events" / "programme_twins.csv"))
    pd.DataFrame({"shot_id": [1, 2, 11, 12], "t_quench": [np.nan, np.nan, 0.30, 0.30]}).to_csv(
        tmp_path / "auto_labels" / "per_shot.csv", index=False)

    def save(path, lost_score):
        os.makedirs(path)
        frames = [pd.DataFrame({"shot_id": s, "t_end": T, "score": 0.5}) for s in (1, 2)]
        frames += [pd.DataFrame({"shot_id": s, "t_end": T, "score": lost_score(T)}) for s in (11, 12)]
        pd.concat(frames).to_parquet(os.path.join(path, "test_scores.parquet"), index=False)
        with open(os.path.join(path, "metrics.json"), "w") as f:
            json.dump({"test": {"pauc": 0.3, "catch_rate": 0.4, "false_alarm_rate": 0.05}}, f)

    for seed in range(1, 6):
        save(tmp_path / "v3" / "baseline" / f"seed_{seed}", lambda t: np.where(t > 0.26, 0.9, 0.1))  # rises 40 ms before
        save(fast_magnetics.run_dir("mag20_5", seed), lambda t: np.where(t > 0.22, 0.9, 0.1))  # rises 80 ms before
    out = fast_magnetics.summary(load=lambda shot: None)
    assert "PRIMARY: C(50 ms) of C (20 + 5 ms) minus A (full): over splits +1.000" in out
    assert "Resolved and at least +0.03" in out
    assert "- C, 20 + 5 ms (primary) minus A at 10 ms: +0.000" in out
    assert (tmp_path / "fast_magnetics" / "summary.txt").exists()
