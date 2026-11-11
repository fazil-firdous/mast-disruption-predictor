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
                          "rms": [0.0, 0.0, 0.05], "peak_ratio": 1.0, "overlap_s": 0.2})
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
    bins = np.round(t.astype(float) / 0.005).astype(int)
    power = {1: 2.0, 11: 2.0, 2: 2.0, 12: 1.0, 3: np.nan, 13: 2.0}  # pair 11: same; 12: different; 13: unknown
    beams = {shot: (bins, np.full((len(bins), 1), value)) for shot, value in power.items()}
    high = {1: 0.0, 11: 0.0, 2: 0.0, 12: 0.10, 3: 0.0, 13: np.nan}  # pair 11: same height; 12: 10 cm apart; 13: unknown
    heights = {shot: (bins, np.full((len(bins), 1), value)) for shot, value in high.items()}
    result = v3_explore.summary(v3_explore.collect(load=lambda shot: None, beams=beams, heights=heights))
    by_height = result["by_height"].set_index(["recipe", "height"])["pairs_per_split"]
    assert by_height[("baseline", "same")] == 1 and by_height[("baseline", "different")] == 1
    assert result["by_matched"].set_index("recipe").loc["baseline", "pairs_per_split"] == 1  # pair 11 only
    assert set(result["by_margin"]["margin"]) == {"twin had no quench"}  # no twin of this made-up project has a quench
    heat = result["by_heating"].set_index(["recipe", "heating"])
    assert set(result["by_heating"]["heating"]) == {"same", "different"}
    assert heat.loc[("baseline", "same"), "pairs_per_split"] == 1 and heat.loc[("baseline", "same"), "C"] == 1.0
    ladder = result["ladder"].set_index("recipe")
    assert ladder.loc["baseline", "C"] == pytest.approx(2 / 3) and ladder.loc["programme", "C"] == pytest.approx(1 / 3)
    assert ladder.loc["baseline", "splits"] == 6 and ladder.loc["baseline", "pairs_per_split"] == 3
    by_group = result["by_group"].set_index(["recipe", "group"])["C"]
    assert by_group[("baseline", "recorded")] == 1.0 and by_group[("baseline", "candidate")] == 0.5
    assert by_group[("programme", "recorded")] == 0.5  # same score as its twin: a tie
    assert set(result["by_lead"]["lead_ms"]) == {5, 30, 50, 100, 200}
    common, pooled = result["common"].set_index("recipe"), result["pooled"].set_index("recipe")
    assert common.loc["baseline", "C"] == ladder.loc["baseline", "C"]  # every made-up shot covers the same time
    assert pooled.loc["baseline", "pairs"] == 3 and pooled.loc["baseline", "C"] == pytest.approx(2 / 3)
    assert pooled.loc["baseline", "low"] <= pooled.loc["baseline", "C"] <= pooled.loc["baseline", "high"]
    exact, loose = result["exact"].set_index("recipe"), result["loose"].set_index("recipe")
    assert exact.loc["baseline", "C"] == 1.0 and exact.loc["baseline", "pairs_per_split"] == 2  # pairs 11 and 12
    assert loose.loc["baseline", "C"] == 0.0 and exact.loc["programme", "C"] == 0.5  # pair 13; ties
    text = v3_explore.text(result)
    assert "exact 1.000" in text and "loose 0.000" in text
    assert "Review check 1" in text and "Review check 2" in text and "3 pairs" in text
    assert "- same heating: programme 0.500" in text
    assert "programmed current only: C 0.333" in text and "- candidate: baseline 0.500" in text
    assert (tmp_path / "v3_explore" / "twin_pairs.csv").exists()


def test_margin_says_how_long_the_twin_went_on():
    out = v3_explore.margin([(11, 1), (12, 2), (13, 3), (14, 4)], {11: 0.3, 12: 0.3, 13: 0.3, 14: 0.3, 1: 0.25, 2: 0.32,
                                                                   3: 0.40})
    assert list(out.values()) == v3_explore.MARGINS  # before, within 50 ms after, 50 ms or more, no quench


def test_heating_same_when_neither_shot_has_a_beam():
    bins = np.arange(20, 60)
    flat = lambda v: (bins, np.full((len(bins), 1), float(v)))
    beams = {1: flat(0.01), 2: flat(0.0), 3: flat(2.0), 4: flat(1.9), 5: flat(2.0), 6: flat(1.0)}
    out = v3_explore.heating([(2, 1), (4, 3), (6, 5), (6, 99)], beams, {2: 0.3, 4: 0.3, 6: 0.3}, {2: 0.1})
    assert out == {(2, 1): "same", (4, 3): "same", (6, 5): "different", (6, 99): "unknown"}


def test_one_ladder_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    metrics = v3_explore.run("programme", 3)
    out = tiny_project / "results" / "v3_explore" / "programme" / "seed_3"
    assert (out / "metrics.json").exists() and len(pd.read_parquet(out / "test_scores.parquet")) > 0
    assert "t_since_start" in metrics["columns"] and not any(c.startswith("sol_") for c in metrics["columns"])
    assert not (tiny_project / "results" / "v3").exists()  # never writes into the pre-registered results


def test_same_time_statistic_ignores_what_the_lost_shot_did_after_its_twin_ended():
    t = np.round(np.arange(0.05, 0.40, 0.005), 3).astype("float32")
    lost = pd.DataFrame({"shot_id": 2, "t_end": t, "score": np.where(t > 0.25, 0.9, 0.2).astype("float32")})
    twin = pd.DataFrame({"shot_id": 1, "t_end": t[t <= 0.2], "score": np.float32(0.2)})  # the twin ended at 0.2 s
    wins = v3_explore.pair_wins(pd.concat([lost, twin]), pd.DataFrame({"shot_id": [2], "other": [1]}), {2: 0.30},
                                {2: 0.1}, leads_ms=(5,)).iloc[0]
    assert wins["win"] == 1.0  # registered: the lost shot's late rise counts, though the twin has no windows there
    assert wins["win_common"] == 0.5  # over the time both have, the two score alike


def test_temporal_run_and_its_twin_test(tiny_project, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")  # fast here
    metrics = v3_explore.run_temporal("programme")
    assert metrics["split"] == "temporal" and (tiny_project / "results" / "v3_explore" / "temporal" / "programme"
                                               / "test_scores.parquet").exists()
    assert v3_explore.recipe_columns("baseline") != v3_explore.recipe_columns("machine")
    # the twin test of that run on two made-up pairs of its own test shots
    scores = pd.read_parquet(tiny_project / "results" / "v3_explore" / "temporal" / "programme" / "test_scores.parquet")
    a, b, c, d = sorted(scores["shot_id"].unique())[:4]
    twins = pd.DataFrame({"shot_id": [b, d], "other": [a, c], "rms": 0.0})
    twins.to_csv(tmp_path / "twins.csv", index=False)
    monkeypatch.setattr(v3_runs, "TWINS", str(tmp_path / "twins.csv"))
    labels_dir = tiny_project / "results" / "auto_labels"
    os.makedirs(labels_dir, exist_ok=True)
    last = scores.groupby("shot_id")["t_end"].max()
    pd.DataFrame({"shot_id": [a, b, c, d], "t_quench": [np.nan, last[b] + 0.006, np.nan, last[d] + 0.006]}).to_csv(
        labels_dir / "per_shot.csv", index=False)
    text = v3_explore.temporal_text(load=lambda shot: None, beams={})
    assert "chronological split" in text and "programmed current only: pAUC" in text and "2 pairs" in text
    assert "every signal" not in text  # only the runs that exist


def test_toy_checks_of_the_pair_statistics():
    t = np.round(np.arange(0.05, 0.30, 0.005), 3).astype("float32")
    rng = np.random.default_rng(5)
    a = pd.DataFrame({"shot_id": 2, "t_end": t, "score": rng.random(len(t)).astype("float32")})
    b = pd.DataFrame({"shot_id": 1, "t_end": t, "score": rng.random(len(t)).astype("float32")})
    twins, loss = pd.DataFrame({"shot_id": [2], "other": [1]}), {2: 0.30, 1: 0.30}

    def wins(lost, kept, pairs=twins):
        return v3_explore.pair_wins(pd.concat([lost, kept]), pairs, loss, {}, leads_ms=(5,)).iloc[0]

    same = wins(a, a.assign(shot_id=1))
    assert same["win"] == 0.5 and same["win_common"] == 0.5  # identical scores: a tie
    swapped = wins(b.assign(shot_id=2), a.assign(shot_id=1))
    first = wins(a, b)
    assert first["win"] + swapped["win"] == 1.0 and first["win_common"] + swapped["win_common"] == 1.0
    # windows that only the lost shot has, after its twin ended, never change the same-time statistic
    short = b[b["t_end"] <= 0.2]
    spiked = a.assign(score=np.where(a["t_end"] > 0.2, np.float32(5.0), a["score"]))
    assert wins(a, short)["win_common"] == wins(spiked, short)["win_common"]
    assert wins(spiked, short)["win"] == 1.0  # the registered statistic does change


def test_pooled_delta_keeps_the_models_paired():
    rows = [{"shot_id": lost, "other": twin, "recipe": recipe, "seed": 1, "win": win}
            for lost, twin, base, mach in ((11, 1, 1.0, 0.0), (12, 1, 1.0, 1.0), (13, 3, 0.0, 0.0))
            for recipe, win in (("baseline", base), ("machine", mach))]
    mean, low, high, n = v3_explore.pooled_delta(pd.DataFrame(rows), "baseline", "machine")
    assert mean == pytest.approx(1 / 3) and n == 3 and low <= mean <= high
    assert v3_explore.pooled_delta(pd.DataFrame(rows), "baseline", "solenoid") is None
