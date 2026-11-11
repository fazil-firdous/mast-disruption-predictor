"""Test for scripts/event_outcomes.py on made-up events. No network."""
import numpy as np
import pandas as pd

from scripts import event_outcomes


def test_counts_by_what_the_shot_did_after_the_event():
    events = pd.DataFrame({
        "shot_id": [1, 1, 2, 3, 4, 5, 6],
        "group": ["smooth_end", "smooth_end", "recorded", "candidate", "recorded", "flux_exhausted", "recorded"],
        "event": ["IRE", "IRE", "IRE", "IRE", "locked mode", "IRE", "disruption"],
        "time_s": [0.10, 0.20, 0.290, 0.150, 0.280, np.nan, 0.3],
        "t_quench": [np.nan, np.nan, 0.300, 0.300, 0.300, 0.4, 0.3]})
    out = event_outcomes.table(events).set_index("event")
    assert list(out.index) == ["IRE", "locked mode"]  # 'disruption' names the ending; the untimed IRE is left out
    assert out.loc["IRE", "timed"] == 3 and out.loc["IRE", "shot not lost"] == 1  # shot 1 counted once
    assert out.loc["IRE", "lost, under 30 ms"] == 1 and out.loc["IRE", "lost, 100 ms or more"] == 1
    assert out.loc["locked mode", "lost, under 30 ms"] == 1
    assert "| IRE | 3 | 1 |" in event_outcomes.text(event_outcomes.table(events))
