"""Tests for scripts/mast_label_model.py: the latent-class model recovers known source accuracies. No network."""
import numpy as np
import pandas as pd
import pytest

from scripts import mast_label_model as lm


def test_dawid_skene_recovers_sources_it_never_saw_labelled():
    rng = np.random.default_rng(0)
    n = 4000
    y = rng.random(n) < 0.4
    # three noisy, independent sources with different reliability; source 2 abstains a third of the time
    s0 = np.where(y, rng.random(n) < 0.9, rng.random(n) < 0.05).astype(int)
    s1 = np.where(y, rng.random(n) < 0.7, rng.random(n) < 0.2).astype(int)
    s2 = np.where(y, rng.random(n) < 0.8, rng.random(n) < 0.3).astype(int)
    s2[rng.random(n) < 0.33] = lm.ABSTAIN
    p, prior, tables = lm.dawid_skene(np.column_stack([s0, s1, s2]), init=np.where(s0 == 1, 0.9, 0.2))
    assert prior == pytest.approx(0.4, abs=0.03)
    assert tables[0][1][1] == pytest.approx(0.9, abs=0.03) and tables[0][0][1] == pytest.approx(0.05, abs=0.03)
    assert tables[1][1][1] == pytest.approx(0.7, abs=0.04)
    assert np.mean((p > 0.5) == y) > 0.88  # better than the best single source alone (about 0.93 vs 0.9 here)


def test_fit_on_a_small_catalogue():
    labels = pd.DataFrame({"shot_id": [1, 2, 3, 4], "campaign": ["M5"] * 4, "note": [True, False, False, False],
                           "termination": ["flat_top_drop", "flat_top_drop", "flux_exhausted", "smooth_end"],
                           "t_quench": [0.30, 0.25, 0.40, np.nan]})
    events = pd.DataFrame({"shot_id": [1, 2], "group": ["recorded", "candidate"], "event": ["disruption", "VDE"],
                           "time_s": [0.30, 0.252], "t_quench": [0.30, 0.25],
                           "source": ["shot_postshot_comment"] * 2, "text": ["", ""]})
    catalogue, prior, tables = lm.fit(labels, events)
    row = catalogue.set_index("shot_id")
    assert row.at[4, "p_disrupted"] == 0.0  # no quench
    assert row.at[2, "words_vote"] == 2 and row.at[2, "cause"] == "VDE"  # VDE time within 20 ms of the quench
    assert row.at[2, "t_event"] == pytest.approx(0.252)
    assert row.at[3, "solenoid_vote"] == 0
    assert "Mean P(disrupted)" in lm.text(catalogue, prior, tables)


def test_an_abstention_is_not_evidence():
    rng = np.random.default_rng(1)
    n = 3000
    y = rng.random(n) < 0.3
    s0 = np.where(y, rng.random(n) < 0.85, rng.random(n) < 0.1).astype(int)
    s1 = np.where(y, rng.random(n) < 0.8, rng.random(n) < 0.15).astype(int)
    s2 = np.where(y, rng.random(n) < 0.75, rng.random(n) < 0.2).astype(int)
    s2[y & (rng.random(n) < 0.5)] = lm.ABSTAIN  # a source that falls silent mostly on positives
    p, _, _ = lm.dawid_skene(np.column_stack([s0, s1, s2]), init=np.where(s0 == 1, 0.9, 0.2))
    silent = s2 == lm.ABSTAIN
    # silence must not by itself push a row towards 1: rows where s0 and s1 both say 0 stay low
    assert p[silent & (s0 == 0) & (s1 == 0)].max() < 0.5


def test_robustness_rows():
    labels = pd.DataFrame({"shot_id": [1, 2, 3], "campaign": ["M5"] * 3, "note": [True, False, False],
                           "termination": ["flat_top_drop", "flat_top_drop", "flux_exhausted"],
                           "t_quench": [0.30, 0.25, 0.40]})
    events = pd.DataFrame({"shot_id": [2], "group": ["candidate"], "event": ["VDE"], "time_s": [0.25],
                           "t_quench": [0.25], "source": ["shot_postshot_comment"], "text": [""]})
    rows = lm.robustness(labels, events)
    assert [r[0] for r in rows] == ["none (all three)"] + lm.SOURCES
