"""Tests for scripts/twin_score.py: it gives the pre-registered concordance pair by pair, from the release's columns
alone, and its range and report. No network."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts import twin_score, v3_runs


def made_up(rng, n=40):
    frames = []
    for shot in range(1, 2 * n + 1):
        t = np.round(0.05 + 0.005 * np.arange(rng.integers(20, 70)), 3)
        frames.append(pd.DataFrame({"shot_id": shot, "t_end": t, "score": rng.choice([0.1, 0.4, 0.4, 0.8], len(t))}))
    twins = pd.DataFrame({"shot_id": range(n + 1, 2 * n + 1), "other": [1 + i // 2 for i in range(n)],
                          "rms": np.where(np.arange(n) % 4 == 0, 0.05, 0.0),
                          "t_loss": rng.choice([0.150, 0.203, 0.25, 0.3, 0.351], n),
                          "t_start": np.where(np.arange(n) % 3 == 0, np.nan, 0.1),
                          "heating": np.where(np.arange(n) % 2 == 0, "same", "different")})
    return pd.concat(frames, ignore_index=True), twins


def test_same_answer_as_the_registered_concordance():
    rng = np.random.default_rng(3)
    for _ in range(5):
        scores, twins = made_up(rng)
        wins = twin_score.pair_wins(scores, twins)
        t_start = {r.shot_id: (None if np.isnan(r.t_start) else r.t_start) for r in twins.itertuples()}
        c, n, c_exact, n_exact = v3_runs.concordance(scores, twins, twins.set_index("shot_id")["t_loss"].to_dict(),
                                                     t_start)
        assert len(wins) == n and wins["win"].mean() == pytest.approx(c)
        exact = wins[wins["rms"] < twin_score.EXACT_RMS]
        assert len(exact) == n_exact and exact["win"].mean() == pytest.approx(c_exact)


def test_range_report_and_command_line(tmp_path):
    scores, twins = made_up(np.random.default_rng(4))
    wins = twin_score.pair_wins(scores, twins)
    c, low, high, n = twin_score.concordance(wins)
    assert low <= c <= high and n == len(wins) and high - low < 0.6
    assert twin_score.concordance(wins.iloc[:0])[3] == 0
    text = twin_score.report(wins)
    assert "all pairs: C = " in text and "same beam heating" in text and "programmes within 1%" in text
    assert "no pairs" in twin_score.report(wins[wins["heating"] == "same"])  # the 'different' line
    scores.to_csv(tmp_path / "scores.csv", index=False)
    twins.to_csv(tmp_path / "twins.csv", index=False)
    out = subprocess.run([sys.executable, twin_score.__file__, str(tmp_path / "scores.csv"), str(tmp_path / "twins.csv")],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == text  # standalone: runs outside the repository's imports
