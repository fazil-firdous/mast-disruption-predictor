import json
import os

import pandas as pd

from scripts import export_scores as ex


def test_export_filters_shots_and_writes_manifest(tmp_path, monkeypatch):
    runs = []
    for model in ex.MODELS:
        for split in (1, 2):
            d = tmp_path / "runs" / model / str(split)
            d.mkdir(parents=True)
            pd.DataFrame({"shot_id": [1, 1, 2, 99], "t_end": [0.1, 0.2, 0.1, 0.1],
                          "score": [0.1, 0.2, 0.3, 0.9]}).to_parquet(d / "test_scores.parquet")
            runs.append((model, split, str(d)))
    twins = tmp_path / "twins.csv"
    pd.DataFrame({"shot_id": [1], "other": [2]}).to_csv(twins, index=False)
    monkeypatch.setattr(ex.v3_runs, "TWINS", str(twins))
    monkeypatch.setattr(ex, "run_paths", lambda: runs)
    monkeypatch.setattr(ex.sc, "folder", lambda: str(tmp_path / "none"))
    monkeypatch.setattr(ex.config, "RESULTS_DIR", str(tmp_path))
    out = tmp_path / "export"
    ex.export(str(out), say=lambda s: None)
    scores = pd.read_parquet(out / "scores.parquet")
    assert set(scores["shot_id"]) == {1, 2} and len(scores) == 3 * len(runs)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "scores.parquet" in manifest["sha256"] and manifest["discharges"] == 2
    assert os.path.exists(str(out) + ".zip")
