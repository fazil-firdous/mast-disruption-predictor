"""Tests for scripts/v3_explore.py: the ladder's inputs, pair-by-pair wins against the pre-registered concordance,
the tables and one ladder run end to end on the tiny fake project. No network."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from scripts import v3_explore, v3_runs
from src import config
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_ladder_inputs_leave_out_one_machine_signal():
    programme, solenoid = v3_explore.ladder_columns("programme"), v3_explore.ladder_columns("solenoid")
    assert "t_since_start" in programme and "t_since_start" in solenoid
    assert any(c.startswith("ref_") for c in programme) and not any(c.startswith("sol_") for c in programme)
    assert "has_sol" not in programme and "has_ref" not in solenoid
    assert any(c.startswith("sol_") for c in solenoid) and not any(c.startswith("ref_") for c in solenoid)


def random_scores(rng, shots):
    frames = []
    for shot in shots:
        t = (0.05 + 0.005 * np.arange(rng.integers(20, 70))).astype("float32")
        frames.append(pd.DataFrame({"shot_id": shot, "t_end": t,
                                    "score": rng.choice([0.1, 0.4, 0.4, 0.8], len(t)).astype("float32")}))
    return pd.concat(frames, ignore_index=True)


def test_pair_wins_at_the_registered_lead_are_the_registered_concordance():
    rng = np.random.default_rng(1)
    for _ in range(10):
        scores = random_scores(rng, range(1, 41))
        twins = pd.DataFrame({"shot_id": range(21, 41), "other": range(1, 21), "rms": 0.0})
        # quench times on the 5 ms window grid and on the 1 ms grid: the boundary windows must be treated alike
        t_quench = {s: float(np.round(rng.choice([0.150, 0.203, 0.25, 0.3, 0.351]), 3)) for s in range(21, 41)}
        t_start = {s: (None if s % 3 == 0 else 0.1) for s in range(21, 41)}
        c, n, _, _ = v3_runs.concordance(scores, twins, t_quench, t_start)
        wins = v3_explore.pair_wins(scores, twins, t_quench, t_start)
        main = wins[wins["lead_ms"] == config.MIN_WARNING_MS]
        assert len(main) == n and main["win"].mean() == pytest.approx(c)


def test_a_longer_lead_stops_the_scores_earlier():
    t = np.round(np.arange(0.05, 0.40, 0.005), 3).astype("float32")
    scores = pd.concat([pd.DataFrame({"shot_id": 2, "t_end": t, "score": np.where(t > 0.27, 0.9, 0.2)}),
                        pd.DataFrame({"shot_id": 1, "t_end": t, "score": 0.2})])
    twins = pd.DataFrame({"shot_id": [2], "other": [1]})
    wins = v3_explore.pair_wins(scores, twins, {2: 0.30}, {2: 0.1}, leads_ms=(5, 50, 400)).set_index("lead_ms")["win"]
    assert wins.to_dict() == {5: 1.0, 50: 0.5}  # the rise is in the last 30 ms; 400 ms leaves no window


def test_tables_and_text_from_saved_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    twins = pd.DataFrame({"shot_id": [11, 12, 13], "other": [1, 2, 3], "group": ["recorded", "candidate", "candidate"],
                          "other_group": ["smooth_end", "smooth_end", "flux_exhausted"], "gap": [10, 10, 10],
                          "rms": 0.0, "peak_ratio": 1.0, "overlap_s": 0.2})
    os.makedirs(tmp_path / "events")
    twins.to_csv(tmp_path / "events" / "programme_twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "events" / "programme_twins.csv"))
    os.makedirs(tmp_path / "auto_labels")
    pd.DataFrame({"shot_id": [1, 2, 3, 11, 12, 13], "t_quench": [np.nan] * 3 + [0.3] * 3}).to_csv(
        tmp_path / "auto_labels" / "per_shot.csv", index=False)
    t = np.round(np.arange(0.05, 0.40, 0.005), 3).astype("float32")
    for recipe, base, lost_score in (("baseline", "v3", 0.9), ("programme", "v3_explore", 0.2)):
        for seed in range(1, 7):
            path = tmp_path / base / recipe / f"seed_{seed}"
            os.makedirs(path)
            frames = [pd.DataFrame({"shot_id": s, "t_end": t, "score": np.float32(0.2)}) for s in (1, 2, 3)]
            frames += [pd.DataFrame({"shot_id": s, "t_end": t, "score": np.float32(lost_score)}) for s in (11, 12)]
            frames += [pd.DataFrame({"shot_id": 13, "t_end": t, "score": np.float32(0.1)})]
            pd.concat(frames).to_parquet(path / "test_scores.parquet", index=False)
            with open(path / "metrics.json", "w") as f:
                json.dump({"seed": seed, "test": {"pauc": 0.3, "catch_rate": 0.4, "false_alarm_rate": 0.05}}, f)
    os.makedirs(tmp_path / "v3" / "machine" / "seed_1")  # unfinished: left out
    result = v3_explore.summary(v3_explore.collect(load=lambda shot: None))
    ladder = result["ladder"].set_index("recipe")
    assert ladder.loc["baseline", "C"] == pytest.approx(2 / 3) and ladder.loc["programme", "C"] == pytest.approx(1 / 3)
    assert ladder.loc["baseline", "splits"] == 6 and ladder.loc["baseline", "pairs_per_split"] == 3
    by_group = result["by_group"].set_index(["recipe", "group"])["C"]
    assert by_group[("baseline", "recorded")] == 1.0 and by_group[("baseline", "candidate")] == 0.5
    assert by_group[("programme", "recorded")] == 0.5  # same score as its twin: a tie
    assert set(result["by_lead"]["lead_ms"]) == {5, 30, 50, 100, 200}
    text = v3_explore.text(result)
    assert "programmed current only: C 0.333" in text and "- candidate: baseline 0.500" in text
    assert (tmp_path / "v3_explore" / "twin_pairs.csv").exists()


def test_one_ladder_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    metrics = v3_explore.run("programme", 3)
    out = tiny_project / "results" / "v3_explore" / "programme" / "seed_3"
    assert (out / "metrics.json").exists() and len(pd.read_parquet(out / "test_scores.parquet")) > 0
    assert "t_since_start" in metrics["columns"] and not any(c.startswith("sol_") for c in metrics["columns"])
    assert not (tiny_project / "results" / "v3").exists()  # never writes into the pre-registered results
