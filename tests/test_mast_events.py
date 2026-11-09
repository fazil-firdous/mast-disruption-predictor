"""Tests for scripts/mast_events.py (v3 step 1) on made-up comments. No network."""
import numpy as np
import pandas as pd
import pytest

from scripts import mast_events


@pytest.mark.parametrize("text, event, t", [
    ("VDE at 0.242s :-)", "VDE", 0.242),
    ("GOOD BEAM VDE @ 363MS", "VDE", 0.363),
    ("Terminated by locked mode at ~310 ms", "locked mode", 0.310),
    ("IRE at 150ms kills the shot.", "IRE", 0.150),
    ("DISRUPTS AT 300MS (FA TRIP)", "disruption", 0.300),
    ("Shot terminated at 300", "early end", 0.300),
])
def test_events_and_times(text, event, t):
    found = {name: time for name, time, _ in mast_events.events_in(text)}
    assert found[event] == pytest.approx(t)


def test_traps_and_non_times():
    assert not mast_events.events_in("no sign of a VDE")  # negated
    assert not mast_events.events_in("Good H-mode shot")
    found = dict((n, t) for n, t, _ in mast_events.events_in("SS run at 43 kV, VDE at 45 kA"))
    assert np.isnan(found["VDE"])  # 45 kA is not a time
    assert mast_events.to_seconds(5, None) != mast_events.to_seconds(5, None)  # 5 without a unit is ambiguous: NaN


def test_pairs():
    assert mast_events.pairs_in(30000, "Good repeat of 29922 with more gas") == [(29922, "repeat")]
    assert (29998, "reference") in mast_events.pairs_in(30000, "reference shot 29998")
    assert mast_events.pairs_in(30000, "repeat of the previous shot") == [(29999, "repeat")]
    assert mast_events.pairs_in(30000, "same as 31000") == []  # later shots cannot be repeated
    assert mast_events.pairs_in(30000, "repeat of 20000") == []  # too far back


def test_build_finds_a_twin():
    labels = pd.DataFrame({"shot_id": [100, 101], "note": [False, False], "t_quench": [0.30, np.nan],
                           "termination": ["flat_top_drop", "smooth_end"]})
    table = pd.DataFrame({"shot_id": [100, 101], "shot_postshot_comment": ["VDE at 0.30s", "Good repeat of 00100"]})
    events, pairs = mast_events.build(labels, table)
    assert events.iloc[0]["event"] == "VDE" and events.iloc[0]["time_s"] == pytest.approx(0.30)
    assert pairs.iloc[0]["twin"]  # 101 repeats 100; one was lost in the flat-top, the other ended smoothly
    assert "Twins" in mast_events.summary(labels, events, pairs)


def test_negation_needs_a_word_and_the_same_clause():
    assert mast_events.events_in("Terminated by locked mode")  # 'nat' in 'terminated' is not 'n't'
    assert mast_events.events_in("no H-mode, VDE at 0.2s")  # different clause
    assert not mast_events.events_in("locked mode that didn t lock") == []  # the mode is named before the negation
    assert not mast_events.events_in("didn t see a locked mode")


def test_times_at_sentence_end_and_before_the_event():
    found = dict((n, t) for n, t, _ in mast_events.events_in("Good beams. Shot terminates at 0.3s. NTM later."))
    assert found["early end"] == pytest.approx(0.3)
    found = dict((n, t) for n, t, _ in mast_events.events_in("EVENT AT 317MS IS DESASTOROUS (LOCKED MODE?)"))
    assert found["locked mode"] == pytest.approx(0.317)
    found = dict((n, t) for n, t, _ in mast_events.events_in("density 300 then locked mode"))
    assert np.isnan(found["locked mode"])  # a number without a unit before the event is not taken
