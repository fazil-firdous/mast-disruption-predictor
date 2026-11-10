"""Tests for scripts/paper2_numbers.py on made-up pairs. No network."""
import numpy as np
import pandas as pd
import pytest

from scripts import paper2_numbers


def made_up():
    twins = pd.DataFrame({"shot_id": [11, 12, 13, 14], "other": [1, 2, 3, 4], "group": ["recorded"] + ["candidate"] * 3,
                          "other_group": ["smooth_end"] * 4, "gap": [1, 2, 3, 20], "rms": [0.0, 0.0, 0.05, 0.0]})
    heating = {11: "same", 12: "same", 13: "different", 14: "unknown"}
    rows = []
    for seed in range(1, 7):
        for recipe, wins in (("baseline", [1, 1, 1, 0]), ("machine", [1, 0, 1, 0]), ("programme", [0.5] * 4)):
            for (shot, other, rms), win in zip(twins[["shot_id", "other", "rms"]].itertuples(index=False), wins):
                for lead in (5, 30):
                    rows.append({"shot_id": shot, "other": other, "lead_ms": lead, "win": win if lead == 5 else 0.5,
                                 "rms": rms, "recipe": recipe, "seed": seed, "heating": heating[shot]})
    return twins, pd.DataFrame(rows)


def test_paired_differences_and_counts():
    twins, pairs = made_up()
    same = pairs[(pairs["lead_ms"] == 5) & (pairs["heating"] == "same")]
    mean, low, high, n = paper2_numbers.paired(same, "baseline", "machine")
    assert mean == pytest.approx(0.5) and n == 6  # baseline wins both same-heating pairs, machine one
    assert paper2_numbers.paired(same, "baseline", "solenoid") is None  # no such runs
    runs = pd.DataFrame({"recipe": ["baseline"] * 3 + ["machine"], "hyst_ratio": [1.0, 1.0, 0.8, np.nan],
                         "hyst_hold": [2, 2, 3, np.nan]})
    catalogue = pd.DataFrame({"group": ["candidate", "candidate", "recorded"], "words_vote": [2, 0, 2]})
    skill = [{"recipe": "programme", "pauc": 0.128, "catch_rate": 0.135, "false_alarm_rate": 0.049}]
    leads = pd.DataFrame({"signal": ["ne", "li", "ne"], "lead_ms": [100, 100, 200], "n": [2500, 2400, 1300]})
    text = paper2_numbers.text(twins, pairs, runs, catalogue, [40, 44], skill, leads)
    assert "programme 0.128, 13.5%, 4.9%" in text and "100: 2,500, 200: 1,300" in text
    assert "Twin pairs: 4" in text and "Programmes within 1%: 75.0% of pairs" in text
    assert "at most 3 in 75.0%" in text and "{'same': 2, 'different': 1, 'unknown': 1}" in text
    assert "Evidence shots (candidates whose comment names a loss within 20 ms of the quench): 1 of 2" in text
    assert "per split: mean 42 (min 40, max 44)" in text and "(1, 2): 2, (0.8, 3): 1" in text
    assert "C on same heating (2 pairs per split):" in text and "- baseline minus machine: +0.500" in text
    assert "- 30: baseline 0.500" in text
