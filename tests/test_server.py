"""Tests for the research-server logic, Telegram parsing, private config, diagnosis verdicts, catalogue,
model registry and live prediction. No network."""
import json
import os

import numpy as np
import pytest

from src import config, notify, server

from scripts import diagnose_network, inspect_groups


def test_parse_command():
    assert server.parse_command("/predict@MastBot 11860") == ("predict", "11860")
    assert server.parse_command("/STATUS") == ("status", "")
    assert server.parse_command("hello") == (None, "")
    assert server.parse_command(None) == (None, "")


def test_state_roundtrip(tmp_path):
    path = str(tmp_path / "logs" / "state.json")
    state = server.new_state()
    state["seeds_done"] = [1, 2]
    server.write_state(state, path)
    again = server.read_state(path)
    assert again["seeds_done"] == [1, 2] and again["paused"] is False


def test_next_job_order():
    state = server.new_state()
    assert server.next_job(state, "abc", 1e9, False) == ("pipeline", None)
    state.update(last_ok_commit="abc")
    assert server.next_job(state, "abc", 1e9, True) == ("seed", 1)
    state["seeds_done"] = list(range(1, config.SEED_SWEEP_MAX + 1))
    assert server.next_job(state, "abc", 1e9, True) == (None, None)
    assert server.next_job(state, "def", 1e9, True) == ("pipeline", None)  # new commit
    state["paused"] = True
    assert server.next_job(state, "def", 1e9, True) == (None, None)


def test_failed_commit_waits_before_retry():
    state = server.new_state()
    state.update(failed_commit="abc", failed_at=1000.0)
    assert server.next_job(state, "abc", 1000.0 + 60, False) == (None, None)
    later = 1000.0 + config.PIPELINE_RETRY_HOURS * 3600 + 1
    assert server.next_job(state, "abc", later, False) == ("pipeline", None)


def test_parse_updates_keeps_chat_and_skips_non_text():
    payload = {"result": [{"update_id": 5, "message": {"chat": {"id": 42}, "text": "/status"}},
                          {"update_id": 6, "message": {"chat": {"id": 42}, "sticker": {}}}]}
    assert notify.parse_updates(payload) == [(5, "42", "/status"), (6, None, None)]


def test_env_file_does_not_override_shell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# private\nMAST_TEST_A="from file"\nMAST_TEST_B=file\n')
    monkeypatch.setenv("MAST_TEST_B", "shell")
    monkeypatch.delenv("MAST_TEST_A", raising=False)
    config._load_env_file(str(env))
    assert os.environ["MAST_TEST_A"] == "from file"
    assert os.environ["MAST_TEST_B"] == "shell"


@pytest.mark.parametrize("tcp, https, expected", [
    ([], True, "DNS"),
    ([(False, "timeout"), (False, "timeout")], False, "no address answers"),
    ([(True, "5 ms"), (False, "timeout")], True, "/etc/hosts"),
    ([(True, "5 ms")], False, "TLS"),
    ([(True, "5 ms")], True, "reachable"),
])
def test_diagnosis_verdicts(tcp, https, expected):
    addresses = [(2, f"10.0.0.{i}") for i in range(len(tcp))]
    assert expected in diagnose_network.verdict("s3.echo.stfc.ac.uk", addresses, tcp, https)


def test_check_connection_retries(monkeypatch):
    from src import data
    calls = []

    def broken(shot_id):
        calls.append(shot_id)
        raise TimeoutError("no route")

    monkeypatch.setattr(data, "_open_summary", broken)
    ok, message = data.check_connection(attempts=3, backoff=0)
    assert not ok and len(calls) == 3 and "3 attempts" in message


def test_catalogue_skips_geometry():
    meta = {"consolidated_metadata": {"metadata": {
        "magnetics/ip": {"node_type": "array", "shape": [1920], "attributes": {"units": "A", "uda_name": "AMC"},
                         "dimension_names": ["time"]},
        "magnetics/probe_r": {"node_type": "array", "shape": [40], "attributes": {}},
        "magnetics": {"node_type": "group"}}}}
    rows = inspect_groups.catalogue(meta)
    assert [r["name"] for r in rows] == ["ip"] and rows[0]["units"] == "A"
    assert len(inspect_groups.catalogue(meta, include_geometry=True)) == 2
