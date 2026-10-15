"""Tests for the GitHub data relay (src/relay.py) and how the research server uses it. No network."""
import json
import os
import zipfile

import pytest

from src import config, relay, server


def test_pick_run_finds_the_new_manual_run():
    requested = relay.parse_time("2026-10-15T14:30:00Z")
    runs = [{"id": 1, "event": "workflow_dispatch", "created_at": "2026-10-15T12:00:00Z"},
            {"id": 2, "event": "push", "created_at": "2026-10-15T14:31:00Z"},
            {"id": 3, "event": "workflow_dispatch", "created_at": "2026-10-15T14:30:20Z"},
            {"id": 4, "event": "workflow_dispatch", "created_at": "2026-10-15T14:29:40Z"}]
    assert relay.pick_run(runs, requested)["id"] == 3
    assert relay.pick_run(runs[:2], requested) is None


def make_artifact(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)


def test_unpack_extracts_data_and_keeps_local_skipped_csv(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "skipped.csv").write_text("shot_id,reason\n1,local\n")
    manifest = {"scale": "full", "table_source": "live", "requested": 2, "available": 2, "failed": 0}
    make_artifact(tmp_path / "a.zip", {"manifest.json": json.dumps(manifest), "raw/11860.parquet": "x",
                                       "shot_table.parquet": "y", "skipped.csv": "shot_id,reason\n9,relay\n"})
    assert relay.unpack(str(tmp_path / "a.zip"), str(data_dir)) == manifest
    assert (data_dir / "raw" / "11860.parquet").exists()
    assert "local" in (data_dir / "skipped.csv").read_text()


def test_unpack_refuses_paths_outside_the_data_folder(tmp_path):
    make_artifact(tmp_path / "bad.zip", {"manifest.json": "{}", "../escape.txt": "x"})
    with pytest.raises(ValueError):
        relay.unpack(str(tmp_path / "bad.zip"), str(tmp_path / "data"))
    assert not (tmp_path / "escape.txt").exists()


def test_configured_follows_the_token(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert not relay.configured()
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    assert relay.configured()


@pytest.mark.parametrize("data_ok, table_ok, relay_ok, expected", [
    (True, True, False, "direct"),
    (True, False, True, "direct-report"),
    (False, False, True, "relay"),
    (False, True, True, "relay"),
    (False, False, False, None),
])
def test_access_mode(data_ok, table_ok, relay_ok, expected):
    assert server.access_mode(data_ok, table_ok, relay_ok) == expected


def test_pipeline_args():
    assert server.pipeline_args("full", "direct") == ["--scale", "full"]
    assert server.pipeline_args("full", "direct-report") == ["--scale", "full", "--from-report", config.REPORT_CSV]
    assert server.pipeline_args("full", "relay", {"table_source": "live"}) == ["--scale", "full", "--offline"]
    assert server.pipeline_args("small", "relay", {"table_source": "report"}) == [
        "--scale", "small", "--offline", "--from-report", config.REPORT_CSV]


def test_relay_needed():
    state = server.new_state()
    assert server.relay_needed(state, "full")
    state["relay_manifest"] = {"scale": "full"}
    assert not server.relay_needed(state, "full")
    assert server.relay_needed(state, "small")
    state["force_fetch"] = True
    assert server.relay_needed(state, "full")


def test_config_snapshot_never_contains_tokens(monkeypatch):
    from src import experiment
    monkeypatch.setattr(config, "SOME_TOKEN", "secret", raising=False)
    snapshot = experiment.config_snapshot()
    assert "SOME_TOKEN" not in snapshot and "GITHUB_REPO" in snapshot
    assert not any("TOKEN" in key for key in snapshot)


def test_offline_build_uses_only_cached_shots(monkeypatch, tmp_path):
    from scripts import build_dataset
    from src import dataset
    from tests.test_dataset import fake_shots
    shots, table = fake_shots(6, 6)
    cached = set(list(shots)[::2])
    monkeypatch.setattr(build_dataset, "is_cached", lambda shot_id: shot_id in cached)
    monkeypatch.setattr(build_dataset, "download_shots", lambda ids: pytest.fail("offline build touched the network"))
    monkeypatch.setattr(dataset, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path / "processed"))
    counts = build_dataset.build("small", table=table, offline=True)
    assert counts["download_failed"] == 6 and counts["labelled"] == 6
    assert os.path.exists(tmp_path / "processed" / "windows.parquet")
