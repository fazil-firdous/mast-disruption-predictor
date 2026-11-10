"""Tests for scripts/twin_differences.py on made-up shots: triplets, the common time grid, D and the direction per
lead time, and the divergence time. No network."""
import numpy as np
import pandas as pd
import pytest

from scripts import twin_differences as td
from tests.test_twin_check import frame, labels_for


def test_kept_twins_take_the_two_nearest_kept_shots_with_the_same_programme():
    labels = labels_for([(1, False, "smooth_end", np.nan), (2, False, "flux_exhausted", 0.35),
                         (3, False, "smooth_end", np.nan), (4, True, "flat_top_drop", 0.3),
                         (5, False, "smooth_end", np.nan), (6, False, "flat_top_drop", 0.3),
                         (7, False, "flat_top_drop", 0.3)])
    shapes = {1: frame(800e3), 2: frame(800e3), 3: frame(500e3), 4: frame(800e3), 5: frame(800e3), 6: frame(800e3),
              7: frame(300e3)}
    out = td.kept_twins(labels, load=lambda s: shapes.get(s))
    # 4 skips 3 (other programme): twins 2 and 1. 6 skips the lost shot 4 and 3: twins 5 and 2. 7 has no twin.
    assert out.values.tolist() == [[4, 2, 1, "recorded"], [6, 5, 2, "candidate"]]


def windows_for(shot, values, offset=0.0, t_first=0.07, t_last=0.295):
    """A shot's windows every 5 ms (shifted by offset), every signal mean set by values(signal, t)."""
    t = np.round(np.arange(t_first, t_last + 1e-9, 0.005), 3) + offset
    return pd.DataFrame({"shot_id": shot, "t_end": t, **{f"{s}_mean": values(s, t) for s in td.SIGNALS}})


def test_series_lines_up_shots_whose_windows_start_at_different_samples():
    flat = lambda s, t: np.full(len(t), 1.0)
    ramp = lambda s, t: t.copy()
    by = td.by_shot(pd.concat([windows_for(3, ramp, offset=0.002), windows_for(2, flat, offset=-0.001),
                               windows_for(1, flat, t_last=0.25)]))
    row = pd.Series({"shot_id": 3, "k1": 2, "k2": 1})
    lead, case, ctrl = td.series(by, row, t_loss=0.30, t_from=0.10)
    assert lead[0] == pytest.approx(200) and lead[-1] == pytest.approx(50)  # 0.10 s to 0.25 s, where shot 1 ends
    assert case.shape == (31, len(td.SIGNALS)) and np.allclose(ctrl, 0)
    assert case[0, 0] == pytest.approx(0.102 - 1.0)  # the lost shot's own value at its window in that bin
    assert td.series(by, pd.Series({"shot_id": 3, "k1": 2, "k2": 99}), 0.30) is None


def test_trailing_lead():
    lead = np.array([50.0, 40, 30, 20, 10])
    assert td.trailing_lead(np.array([True, False, True, True, True]), lead) == 30
    assert td.trailing_lead(np.array([True, True, True, True, True]), lead) == 50
    assert td.trailing_lead(np.array([True, True, True, True, False]), lead) == 0
    assert td.trailing_lead(np.array([], dtype=bool), np.array([])) == 0


def test_a_signal_that_departs_60_ms_before_the_loss_is_found_and_the_others_are_not():
    rng = np.random.default_rng(0)
    frames, rows = [], []
    for i in range(80):
        lost, k1, k2 = 1000 + 10 * i, 1000 + 10 * i - 2, 1000 + 10 * i - 5

        def noise(s, t):
            return rng.normal(0, 1, len(t))

        def lost_values(s, t):  # density climbs from 60 ms before the loss at 0.30 s; nothing else moves
            return rng.normal(0, 1, len(t)) + (np.clip((t - 0.24) / 0.06, 0, None) * 30 if s == "ne" else 0)

        frames += [windows_for(lost, lost_values, offset=0.001 * (i % 3)), windows_for(k1, noise),
                   windows_for(k2, noise, offset=-0.001)]
        rows.append({"shot_id": lost, "k1": k1, "k2": k2, "group": "candidate"})
    triplets = pd.DataFrame(rows)
    collected = td.collect(triplets, td.by_shot(pd.concat(frames)), {r["shot_id"]: 0.30 for r in rows}, {})
    assert len(collected) == 80
    leads = td.lead_table(collected, np.random.default_rng(1)).set_index(["signal", "lead_ms"])
    assert leads.loc[("ne", 10), "D"] > 0.95 and leads.loc[("ne", 10), "D_low"] > 0.5
    assert leads.loc[("ne", 10), "up_case"] > 0.95 and 0.3 < leads.loc[("ne", 10), "up_ctrl"] < 0.7
    assert 0.3 < leads.loc[("ne", 100), "D"] < 0.7  # before the departure: like two good repeats
    assert 0.3 < leads.loc[("li", 10), "D"] < 0.7 and leads.loc[("li", 10), "D_low"] < 0.5 < leads.loc[("li", 10), "D_high"]
    assert leads.loc[("ne", 10), "n"] == 80
    assert ("ne", 300) not in leads.index  # the flat-top windows start 230 ms before the loss
    div, band = td.divergence(collected)
    assert len(div) == 80 and band["ne"] > 0
    assert (div["case_ne"] >= 20).mean() > 0.9 and (div["ctrl_ne"] >= 20).mean() < 0.05
    assert (div["case_li"] >= 20).mean() < 0.05
    first = td.first_diverged(div)
    assert (first == "ne").mean() > 0.9
    text = td.text(triplets, leads.reset_index(), div, causes={r["shot_id"]: "VDE" for r in rows})
    assert "| ne | 1.00* |" in text and "operator named 'VDE' (" in text and "ne " in text
