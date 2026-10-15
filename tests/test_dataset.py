"""Tests for src/dataset.py on synthetic shots, no network."""
import numpy as np
import pandas as pd
import pytest

from src import config, dataset, features
from tests.synthetic import make_shot


def fake_shots(n_quench=12, n_ramp=12):
    """Synthetic shots plus a fake shot table: quench shots are noted as disrupted, ramp-downs are not."""
    shots, rows = {}, []
    campaigns = ["M5", "M6", "M7", "M8", "M9", "Unknown"]
    for i in range(n_quench + n_ramp):
        shot_id = 1000 + i
        quench = i < n_quench
        kind = "quench" if quench else "rampdown"
        shots[shot_id] = make_shot(kind, t_quench=0.2 + 0.002 * i, seed=i)
        comment = f"DISRUPTION AT {200 + 2 * i}MS" if quench else "good shot"
        rows.append({"shot_id": shot_id, "shot_postshot_comment": comment, "campaign": campaigns[i % 6]})
    return shots, pd.DataFrame(rows)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    shots, table = fake_shots()
    monkeypatch.setattr(dataset, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))
    return shots, table


def test_label_and_select(fake):
    _, table = fake
    labelled = dataset.label_shots(list(table["shot_id"]), table)
    assert labelled["disrupted"].sum() == 12
    assert (labelled.loc[labelled["disrupted"], "t_quench"] - labelled.loc[labelled["disrupted"], "t_disrupt"]).abs().max() == 0
    picked = dataset.select_shots(labelled, 5, 7)
    assert picked["disrupted"].sum() == 5 and (~picked["disrupted"]).sum() == 7
    again = dataset.select_shots(labelled, 5, 7)
    assert list(picked["shot_id"]) == list(again["shot_id"])


def test_no_window_at_or_after_t_disrupt(fake):
    _, table = fake
    labelled = dataset.label_shots(list(table["shot_id"]), table)
    win = dataset.labelled_windows(dataset.build_windows(labelled), labelled)
    disrupted = win[win["disrupted"]]
    assert len(disrupted) > 0
    assert (disrupted["t_end"] < disrupted["t_disrupt"]).all()


def test_window_labels_horizon():
    t_end = np.array([0.10, 0.169, 0.170, 0.185, 0.199, 0.2])
    assert list(dataset.window_labels(t_end, 0.2, horizon_ms=30)) == [0, 0, 1, 1, 1, 0]
    assert list(dataset.window_labels(t_end, np.nan)) == [0] * 6


def test_labelled_windows_marks_horizon_and_near_quench():
    windows = pd.DataFrame({"shot_id": [1, 1, 1, 2, 2, 2], "t_end": [0.1, 0.18, 0.21, 0.1, 0.18, 0.21]})
    shots = pd.DataFrame({"shot_id": [1, 2], "t_quench": [0.2, 0.2]})
    for source in dataset.LABEL_SOURCES:
        shots[f"t_disrupt_{source}"] = [0.2, np.nan]
    win = dataset.labelled_windows(windows, shots, horizon_ms=30)
    assert list(win["shot_id"]) == [1, 1, 2, 2, 2]  # shot 1 loses its window after t_disrupt
    assert list(win["label"]) == [0, 1, 0, 0, 0]
    assert list(win["near_quench"]) == [False, False, False, True, False]


def test_stop_time_uses_quench_for_clean_shots():
    row = {"t_disrupt": np.nan, "t_quench": 0.25, "t_disrupt_both": np.nan, "t_disrupt_signal": 0.25,
           "t_disrupt_note": np.nan}
    assert dataset.stop_time(row) == 0.25
    row = {k: np.nan for k in row}
    assert dataset.stop_time(row) == np.inf


def test_random_split_keeps_shots_together_and_stratifies():
    shots = pd.DataFrame({"shot_id": range(100), "disrupted": [i < 40 for i in range(100)]})
    split = dataset.random_split(shots)
    assert set(split.index) == set(range(100))
    counts = pd.crosstab(shots["disrupted"], shots["shot_id"].map(split))
    assert counts.loc[True, "train"] == 24 and counts.loc[True, "val"] == 8 and counts.loc[True, "test"] == 8
    assert counts.loc[False, "train"] == 36


def test_no_shot_in_two_splits(fake):
    _, table = fake
    labelled = dataset.label_shots(list(table["shot_id"]), table)
    labelled["split"] = labelled["shot_id"].map(dataset.random_split(labelled))
    win = dataset.build_windows(labelled).merge(labelled[["shot_id", "split"]], on="shot_id")
    assert (win.groupby("shot_id")["split"].nunique() == 1).all()


def test_temporal_split_by_campaign():
    shots = pd.DataFrame({"shot_id": range(12), "disrupted": [i % 2 == 0 for i in range(12)],
                          "campaign": ["M5", "M6", "M7", "M8", "M9", "Unknown"] * 2})
    split = dataset.temporal_split(shots)
    by_campaign = shots.assign(split=shots["shot_id"].map(split)).groupby("campaign")["split"].unique()
    assert set(by_campaign["M8"]) == {"test"} and set(by_campaign["M9"]) == {"test"}
    assert set(by_campaign["Unknown"]) == {"none"}
    assert set(np.concatenate([by_campaign[c] for c in ["M5", "M6", "M7"]])) <= {"train", "val"}


def test_features_without_signal():
    kept = features.features_without("prad")
    assert not any("prad" in name for name in kept)
    assert "ip_abs_slope" in kept and "t_since_start" in kept
    assert "pnbi_on" not in features.features_without("pnbi")
