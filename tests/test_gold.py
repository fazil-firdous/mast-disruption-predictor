"""Tests for the hand-labelling tool (src/gold.py). No network."""
import numpy as np
import pandas as pd
import pytest

from src import config, gold
from tests.synthetic import make_shot


def fake_shots():
    rows = []
    for i in range(40):
        note, quench = i % 4 in (0, 1), i % 4 in (0, 2)
        rows.append({"shot_id": 100 + i, "note": note, "t_quench": 0.2 if quench else np.nan,
                     "status_note": "disrupted" if note else "clean",
                     "status_signal": "disrupted" if quench else "clean",
                     "status_both": "disrupted" if note and quench else "clean"})
    return pd.DataFrame(rows)


@pytest.fixture
def gold_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(gold, "FOLDER", str(tmp_path / "gold"))
    return tmp_path


def test_queue_follows_the_plan_and_is_repeatable():
    plan = {"unnoted_quench": 4, "noted_quench": 3, "no_quench": 2, "noted_no_quench": 50}
    items = gold.build_queue(fake_shots(), plan)
    kinds = [item["stratum"] for item in items]
    assert kinds.count("unnoted_quench") == 4 and kinds.count("noted_quench") == 3
    assert kinds.count("noted_no_quench") == 10  # only 10 exist
    assert items == gold.build_queue(fake_shots(), plan)


def test_record_next_and_report(gold_dir):
    shots = fake_shots()
    items = gold.queue(shots)
    assert gold.next_item(items, gold.read_labels()) == items[0]
    gold.record(items[0]["shot_id"], "d", items[0]["stratum"])
    gold.record(items[1]["shot_id"], "u", items[1]["stratum"])
    labels = gold.read_labels()
    assert gold.next_item(items, labels) == items[2]
    table, _ = gold.evaluate(shots, labels)
    assert set(table["label source"]) == {"note", "signal", "both"} and (table["shots"] == 1).all()
    assert "2 answered (1 unsure)" in gold.report_text(shots, labels)


def test_blind_plot_has_no_labels(gold_dir, monkeypatch):
    monkeypatch.setattr(gold, "load_shot", lambda shot_id: make_shot("quench"))
    monkeypatch.setattr(config, "DATA_DIR", str(gold_dir))
    path = gold_dir / "plot.png"
    gold.plot_blind(101, str(path))
    assert path.exists() and path.stat().st_size > 1000
