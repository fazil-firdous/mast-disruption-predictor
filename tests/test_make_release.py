"""Tests for scripts/make_release.py on made-up tables: the files, no comment text, the heating column, checksums and
the zip. No network."""
import hashlib
import os
import zipfile

import numpy as np
import pandas as pd

from scripts import make_release, v3_runs
from src import config



def read(path, mode="r"):
    """The file's content; closes the file (the CI runs with warnings as errors, and an unclosed file is one)."""
    with open(path, mode) as f:
        return f.read()


def test_release_has_the_tables_without_comment_text(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    events_dir = tmp_path / "events"
    os.makedirs(events_dir)
    monkeypatch.setattr(make_release, "EVENTS_DIR", str(events_dir))
    monkeypatch.setattr(make_release, "TRIPLETS", str(tmp_path / "twin_differences" / "triplets.csv"))
    monkeypatch.setattr(v3_runs, "TWINS", str(events_dir / "programme_twins.csv"))
    pd.DataFrame({"shot_id": [1, 2, 3, 11, 12], "campaign": "M9", "note": [False, False, False, True, False],
                  "termination": ["smooth_end"] * 3 + ["flat_top_drop"] * 2, "t_quench": [np.nan] * 3 + [0.3, 0.3],
                  "p_disrupted": [0, 0, 0, 0.9, 0.6], "cause": ["", "", "", "VDE", ""], "t_event": [np.nan] * 3 + [0.3, 0.3],
                  "note_vote": 0, "words_vote": 0, "solenoid_vote": 0,
                  "group": ["smooth_end"] * 3 + ["recorded", "candidate"]}).to_csv(events_dir / "catalogue.csv", index=False)
    pd.DataFrame({"shot_id": [11], "group": ["recorded"], "event": ["VDE"], "time_s": [0.29], "t_quench": [0.3],
                  "source": ["shot_postshot_comment"], "text": ["vde at 0.29s, ask A. Person"]}).to_csv(
        events_dir / "events.csv", index=False)
    pd.DataFrame({"shot_id": [11, 12], "other": [1, 2], "group": ["recorded", "candidate"],
                  "other_group": ["smooth_end"] * 2, "gap": [10, 10], "rms": [0.0, 0.05], "peak_ratio": 1.0,
                  "overlap_s": 0.2}).to_csv(events_dir / "programme_twins.csv", index=False)
    bins = np.arange(20, 59)
    beams = {s: (bins, np.full((len(bins), 1), v)) for s, v in {1: 2.0, 11: 2.0, 2: 2.0, 12: 1.0}.items()}
    folder = make_release.build(load=lambda shot: None, beams=beams)
    names = sorted(os.listdir(folder))
    assert names == ["README.md", "SHA256SUMS", "events.csv", "programme_twins.csv", "shots.csv", "twin_score.py"]
    events = pd.read_csv(os.path.join(folder, "events.csv"))
    assert "text" not in events and list(events.columns) == make_release.EVENT_COLUMNS
    assert "Person" not in read(os.path.join(folder, "events.csv"))
    twins = pd.read_csv(os.path.join(folder, "programme_twins.csv"))
    assert twins["heating"].tolist() == ["same", "different"] and twins["t_loss"].tolist() == [0.3, 0.3]
    assert twins["t_start"].isna().all()  # no programme in this made-up project
    readme = read(os.path.join(folder, "README.md"))
    assert "shots.csv (5 rows" in readme and "CC BY-SA 4.0" in readme and "Beam heating: " in readme
    assert "named_pairs.csv" not in readme and "triplets.csv" not in readme  # only files that are in the release
    for line in read(os.path.join(folder, "SHA256SUMS")).splitlines():
        digest, name = line.split()
        assert hashlib.sha256(read(os.path.join(folder, name), "rb")).hexdigest() == digest
    with zipfile.ZipFile(folder + ".zip") as z:
        assert sorted(z.namelist()) == sorted(f"{make_release.NAME}/{n}" for n in names)
