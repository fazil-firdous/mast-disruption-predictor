"""Tests for scripts/drift.py on made-up shots and scores: population B follows its rule, the shared-time mean
statistic, the split and pooled comparisons, the signal shares and the summary. No network."""
import numpy as np
import pandas as pd

from scripts import drift, twin_check

T = np.round(np.arange(0.0, 0.40, 0.001), 3)


def frame(shot):
    peak = 4e5 if shot != 105 else 7e5  # shot 105 ran a different programme
    ref = peak * np.clip(T / 0.05, 0, 1) * np.clip((0.38 - T) / 0.05, 0, 1)
    return pd.DataFrame({"time": T, "ip_ref": ref})


def test_population_b_follows_its_rule(monkeypatch):
    groups = {100: "smooth_end", 101: "smooth_end", 102: "candidate", 103: "smooth_end", 104: "flux_exhausted",
              105: "smooth_end", 106: "smooth_end"}
    monkeypatch.setattr(twin_check, "groups", lambda labels: pd.Series(groups))
    labels = pd.DataFrame({"shot_id": list(groups), "campaign": "M7",
                           "t_quench": [np.nan, np.nan, 0.30, np.nan, np.nan, np.nan, np.nan]})
    b = drift.pairs_b(labels, frame)
    got = set(zip(b["E"], b["F"]))
    # 101 is followed by the lost 102 (excluded); 103 skips the lost 102 back to 101; 106 skips 105 (other programme)
    assert got == {(101, 103), (103, 104), (104, 106)}


def scores_for(values):
    """Saved test scores: shot -> constant score over windows 0.05 to 0.35 s."""
    t = np.round(np.arange(0.05, 0.36, 0.005), 3)
    return pd.concat([pd.DataFrame({"shot_id": s, "t_end": t, "score": v}) for s, v in values.items()], ignore_index=True)


def test_shared_time_statistic():
    pairs = pd.DataFrame({"pop": ["A", "B", "B"], "E": [1, 3, 5], "F": [2, 4, 6], "L": [9, -1, -1]})
    pairs = drift.with_times(pairs, {s: 0.06 for s in range(1, 10)}, {s: 0.35 for s in range(1, 10)})
    assert np.allclose(pairs["t0"], 0.06) and np.allclose(pairs["t1"], 0.32) and list(pairs["gap"]) == [1, 1, 1]
    w = drift.run_wins(scores_for({1: 0.2, 2: 0.3, 3: 0.4, 4: 0.4, 5: 0.5}), pairs)  # shot 6 not in this test set
    assert w[0] == 1.0 and w[1] == 0.5 and np.isnan(w[2])
    late = drift.with_times(pairs.iloc[:1][["pop", "E", "F", "L"]], {1: 0.06, 2: 0.06}, {1: 0.35, 2: 0.08})
    assert np.isnan(drift.run_wins(scores_for({1: 0.2, 2: 0.3}), late)[0])  # shared time too short


def test_comparison_and_summary():
    rng = np.random.default_rng(0)
    n_a, n_b = 40, 120
    pairs = pd.DataFrame({"pop": ["A"] * n_a + ["B"] * n_b, "E": np.arange(n_a + n_b) * 2,
                          "F": np.arange(n_a + n_b) * 2 + 1, "L": -1})
    pairs = drift.with_times(pairs, {}, {})
    wins = np.vstack([np.concatenate([rng.random(n_a) < 0.75, rng.random(n_b) < 0.5]).astype(float) for _ in range(8)])
    pa, pb, d, n = drift.over_splits(wins, pairs)
    assert n == 8 and d[1] > 0 and 0.6 < pa[0] < 0.9
    pool = drift.pooled(wins, pairs, n_boot=200)
    assert pool[3] == n_a and pool[4] == n_b and pool[2][1] > 0
    campaign = {s: "M7" for s in pairs["F"]}
    out = drift.text(pairs, {"baseline": wins, "programme": np.full_like(wins, 0.5)}, {}, [0.5] * 8, campaign,
                     {"ne": ((0.6, 40), (0.5, 120))})
    assert "PRIMARY (full model, 8 splits): Delta = P_A - P_B = +" in out
    assert "Resolved, positive and at least 0.03" in out
    assert "- programme-only (8 splits): 0.500" in out and "line-average density: 0.600 | 0.500" in out


def test_signal_shares_load_each_shot_once():
    pairs = drift.with_times(pd.DataFrame({"pop": ["A", "B"], "E": [1, 1], "F": [2, 3], "L": [9, -1]}),
                             {1: 0.06, 2: 0.06, 3: 0.06}, {1: 0.35, 2: 0.35, 3: 0.35})
    calls = []

    def load(shot):
        calls.append(shot)
        return pd.DataFrame({"time": T, **{name: np.full(len(T), float(shot)) for name in drift.SIGNALS}})

    out = drift.signal_shares(pairs, load)
    assert sorted(calls) == [1, 2, 3] and out["ne"] == ((1.0, 1), (1.0, 1))
