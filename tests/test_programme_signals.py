"""Tests for scripts/programme_signals.py and scripts/programme_fetch.py: reading a group, the sample, the comparison
of a pair, the look in words on made-up groups, what happens when the store does not answer, the table made for
the server and how the server collects it from GitHub. No network."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts import programme_fetch, programme_signals
from src import config

T = np.arange(0.0, 0.4, 0.001)
RAMP = np.minimum(T / 0.1, 1)


def group(**signals):
    return xr.Dataset({name: ("time", np.asarray(values, dtype=float)) for name, values in signals.items()},
                      coords={"time": T})


def fake_open(shot_id, name):
    """Shots 1 and 2 were asked the same; shot 3 was asked twice the height and has no valve requests; shot 4 has
    constant requests; shot 5 has no store."""
    if shot_id == 5:
        raise OSError("no such store")
    if name == "pulse_schedule":
        height = {1: 0.01 * RAMP, 2: 0.01 * RAMP, 3: 0.02 * RAMP, 4: np.zeros(len(T))}[shot_id]
        return group(i_plasma=6e5 * RAMP, n_e_line=np.zeros(len(T)), z_ref=height)
    if shot_id == 3:
        return group(inboard_total=np.cumsum(np.full(len(T), 3e18)))
    ds = group(inboard_total=np.cumsum(np.full(len(T), 1e18 * shot_id)))
    valves = np.zeros((2, len(T))) if shot_id == 4 else np.vstack([2.0 * RAMP, np.where(T > 0.2, 1.0, 0.0)])
    ds["valve_target_voltage"] = (("target_valve_channel", "time"), valves)
    return ds


def test_series_along_time():
    ds = group(a=np.ones(len(T)))
    ds["valves"] = (("channel", "time"), np.vstack([np.ones(len(T)), np.full(len(T), 2.0)]))
    ds["cube"] = (("x", "y", "time"), np.ones((2, 2, len(T))))
    assert programme_signals.series(ds, "a")[1].sum() == len(T)
    assert (programme_signals.series(ds, "valves")[1] == 3.0).all()  # summed over the valves
    assert programme_signals.series(ds, "cube") is None and programme_signals.series(ds, "missing") is None
    assert programme_signals.variables(ds) == ["a[time]", "cube[x,y,time]", "valves[channel,time]"]


def test_describe_one_shot():
    row, found = programme_signals.describe(1, fake_open)
    assert row["pulse_schedule_vars"] == "i_plasma[time] n_e_line[time] z_ref[time]"
    assert row["height_ref_varies"] and row["height_ref_max"] == 0.01 and row["valves_ref_max"] == 3.0
    assert row["density_ref_samples"] == len(T) and not row["density_ref_varies"]  # there, but all zero
    assert found["gas_injected"] is None and row["gas_injected_samples"] == 0  # this shot has no such variable
    row, found = programme_signals.describe(5, fake_open)
    assert row["pulse_schedule_vars"] == "not opened: OSError" and all(s is None for s in found.values())


def test_sample_spreads_over_the_list():
    twins = pd.DataFrame({"shot_id": range(100, 200), "other": range(0, 100)})
    picked = programme_signals.sample(twins, 5)
    assert len(picked) == 5 and picked["shot_id"].iloc[[0, -1]].tolist() == [100, 199]  # both ends of the list
    assert picked["shot_id"].diff().dropna().between(24, 26).all()  # evenly spread
    assert len(programme_signals.sample(twins, 500)) == 100


def test_difference_in_both_units():
    a, b = (T, 0.01 * RAMP), (T, 0.02 * RAMP)
    d = programme_signals.difference(a, b, 0.3)
    assert d["abs"] == pytest.approx(d["rms"] * 0.02) and 0.4 < d["rms"] < 0.5
    assert programme_signals.difference(a, a)["rms"] == 0.0
    assert programme_signals.difference(a, b, 0.01) is None  # too little shared time


def test_look_compares_the_two_shots_of_a_pair():
    pairs = pd.DataFrame({"shot_id": [2, 3, 4, 5], "other": [1, 1, 1, 1]})
    campaign = {1: "M5", 2: "M5", 3: "M5", 4: "M9", 5: "M9"}
    said = []
    shots, compared = programme_signals.look(range(1, 6), pairs, {2: 0.3, 3: 0.3}, campaign, fake_open, workers=2,
                                             say=said.append)
    assert said[0].startswith("first shot 1, read in") and "pulse_schedule: i_plasma[time] n_e_line[time] z_ref[time]" in said[1]
    assert len(said) == 3 + 4 and sum(line.startswith("[") for line in said) == 4  # one line per later shot
    assert any("shot 5:" in line and "pulse_schedule and gas_injection failed" in line for line in said)
    assert any("shot 4: density_ref constant, height_ref constant, valves_ref constant, valves_drive no, gas_inboard yes" in line
               for line in said)
    assert len(shots) == 5 and list(shots["campaign"]) == ["M5", "M5", "M5", "M9", "M9"]
    by = compared.set_index("shot_id")
    assert by.loc[2, "height_ref_rms"] == 0.0 and by.loc[2, "valves_ref_rms"] == 0.0  # asked the same
    assert 0.4 < by.loc[3, "height_ref_rms"] < 0.5 and by.loc[3, "height_ref_abs"] == pytest.approx(by.loc[3, "height_ref_rms"] * 0.02)
    assert np.isnan(by.loc[3, "valves_ref_rms"])  # shot 3 has no valve requests
    assert by.loc[4, "height_ref_state"] == "one" and np.isnan(by.loc[4, "height_ref_rms"])  # it changes in one shot only
    assert by.loc[5, "height_ref_state"] == "missing" and np.isnan(by.loc[5, "height_ref_rms"])
    assert list(by["density_ref_state"]) == ["both_constant"] * 3 + ["missing"]  # no reference density in either shot
    assert by.loc[2, "height_ref_state"] == "both" and by.loc[3, "valves_ref_state"] == "missing"
    assert 0 < by.loc[2, "gas_inboard_rms"] < 1
    # stopping 100 ms earlier leaves less of the flat part, where the two differ most
    assert by.loc[2, "height_ref_rms_early"] == 0.0 and 0 < by.loc[3, "height_ref_rms_early"] < by.loc[3, "height_ref_rms"]
    out = programme_signals.text(shots, compared)
    assert "4 twin pairs, 5 shots," in out and "s per shot (median)" in out
    assert "reference density (pulse_schedule/n_e_line): present in 80% of the shots, not constant in 0%." in out
    assert "requested height (pulse_schedule/z_ref): present in 80% of the shots, not constant in 60%." in out
    assert "M5 100% of 3; M9 0% of 2" in out
    assert "pairs: both 2; both_constant 0; constants_differ 0; one 1; missing 1" in out
    assert "pairs: both 0; both_constant 3; constants_differ 0; one 0; missing 1" in out
    assert "where it changes in both, up to the loss (2 pairs): relative difference median" in out and "under 10% in 50%" in out
    assert "up to 100 ms before the loss (2 pairs)" in out and "- 1 shots: not opened: OSError" in out


def test_a_look_at_shots_alone():
    shots, compared = programme_signals.look([1, 2, 4], None, {}, {1: "M5", 2: "M5", 4: "M9"}, fake_open, workers=2,
                                             say=lambda line: None)
    assert len(shots) == 3 and len(compared) == 0
    out = programme_signals.text(shots, compared)
    assert "0 twin pairs, 3 shots" in out and "pairs:" not in out and "where it changes" not in out
    assert "requested gas valve voltages, summed over valves (gas_injection/valve_target_voltage): present in 100% of the shots, not constant in 67%." in out


def test_a_failed_read_is_counted_not_fatal():
    class Broken:
        data_vars = {}

        def __contains__(self, name):
            raise TimeoutError("stalled")

    row, found = programme_signals.describe(7, lambda shot, name: Broken())
    assert row["pulse_schedule_vars"] == "not read: TimeoutError" and found["density_ref"] is None
    assert row["density_ref_samples"] == 0 and row["seconds"] >= 0


def test_open_group_sets_a_time_limit(monkeypatch):
    seen = {}

    def open_zarr(url, **kwargs):
        seen.update(url=url, **kwargs)
        return "dataset"

    monkeypatch.setattr(programme_signals.xr, "open_zarr", open_zarr)
    monkeypatch.setattr(programme_signals.config, "SHOT_ZARR_URL", "https://example.org/{shot_id}.zarr")
    assert programme_signals.open_group(11823, "pulse_schedule") == "dataset"
    assert seen["url"] == "https://example.org/11823.zarr" and seen["group"] == "pulse_schedule"
    assert seen["storage_options"]["client_kwargs"]["timeout"].total == programme_signals.REQUEST_S


def test_look_stops_when_the_store_does_not_answer():
    def stalled(shot_id, name):
        raise TimeoutError("no answer")

    ids = range(100, 130)
    said = []
    with pytest.raises(RuntimeError, match="first shot"):
        programme_signals.look(ids, None, {}, {}, stalled, workers=2, say=said.append)
    assert len(said) == 3  # the first shot's three lines, then it stops

    def later(shot_id, name):  # the first shot answers, then nothing does
        if shot_id == 100:
            return fake_open(1, name)
        raise TimeoutError("no answer")

    said = []
    with pytest.raises(RuntimeError, match="8 shots in a row"):
        programme_signals.look(ids, None, {}, {}, later, workers=1, say=said.append)
    assert sum(line.startswith("[") for line in said) == programme_signals.GIVE_UP_AFTER

    def absent(shot_id, name):  # shots without the two groups are not a reason to stop
        if shot_id == 100:
            return fake_open(1, name)
        raise KeyError(name)

    shots, compared = programme_signals.look(ids, None, {}, {}, absent, workers=2, say=lambda line: None)
    assert len(shots) == 30 and len(compared) == 0


def test_state_of_a_pair_in_one_signal():
    ramp, zero, one = (T, RAMP), (T, np.zeros(len(T))), (T, np.ones(len(T)))
    assert programme_signals.state(ramp, ramp) == "both" and programme_signals.state(ramp, zero) == "one"
    assert programme_signals.state(zero, zero) == "both_constant" and programme_signals.state(zero, one) == "constants_differ"
    assert programme_signals.state(ramp, None) == "missing"


def test_bin_means_on_whole_multiples_of_the_grid():
    t = np.array([-0.0049, -0.0001, 0.0, 0.0049, 0.0051, 0.0149])
    out = programme_fetch.binned(t, np.array([1.0, 3.0, 10.0, 20.0, 5.0, 7.0]), 0.005)
    assert out.to_dict() == {-1: 2.0, 0: 15.0, 1: 5.0, 2: 7.0}  # a bin with no sample is left out


def test_the_table_for_the_server_and_back(monkeypatch, tmp_path):
    frame, status = programme_fetch.shot_frame(1, fake_open)
    assert status == "ok" and list(frame.columns) == ["shot_id", "time"] + list(programme_signals.TARGETS)
    assert frame["time"].iloc[0] == pytest.approx(0.005) and frame["height_ref"].iloc[-1] == pytest.approx(0.01)
    assert frame["valves_ref"].max() == pytest.approx(3.0) and frame["gas_injected"].isna().all()
    assert programme_fetch.shot_frame(5, fake_open) == (None, "unreachable")

    def absent(shot_id, name):
        raise KeyError(name)

    assert programme_fetch.shot_frame(6, absent) == (None, "empty")
    said = []
    table, report = programme_fetch.fetch([1, 2, 3, 4, 5], workers=2, open_group=fake_open, say=said.append)
    assert report["shots_requested"] == 5 and report["shots_with_data"] == 4 and report["failed_ids"] == [5]
    assert said[-1].startswith("[5/5] 4 with data") and list(table["shot_id"].unique()) == [1, 2, 3, 4]
    # read back as the server does: the same comparison as from the store
    pairs = pd.DataFrame({"shot_id": [2, 3, 4, 5], "other": [1, 1, 1, 1]})
    results = programme_signals.from_table(table, [1, 2, 3, 4, 5])
    shots, compared = programme_signals.tables(results, pairs, {2: 0.3, 3: 0.3}, {1: "M5", 2: "M5", 3: "M5", 4: "M9", 5: "M9"})
    by = compared.set_index("shot_id")
    assert by.loc[2, "height_ref_state"] == "both" and by.loc[2, "height_ref_rms"] == 0.0
    assert 0.4 < by.loc[3, "height_ref_rms"] < 0.5 and by.loc[4, "height_ref_state"] == "one"
    assert by.loc[5, "height_ref_state"] == "missing" and by.loc[2, "density_ref_state"] == "both_constant"
    out = programme_signals.text(shots, compared)
    assert "4 twin pairs, 5 shots." in out and "variable lists" not in out  # no speed and no lists from a table

    def stalled(shot_id, name):
        raise TimeoutError("no answer")

    monkeypatch.setattr(programme_fetch, "GIVE_UP_AFTER", 3)
    with pytest.raises(RuntimeError, match="3 shots in a row"):
        programme_fetch.fetch(range(10), workers=1, open_group=stalled, say=lambda line: None)


class Reply:
    def __init__(self, payload=None, content=b"", ok=True, status=200):
        self.payload, self.content, self.ok, self.status_code, self.headers, self.text = payload, content, ok, status, {}, ""

    def json(self):
        return self.payload


def test_collecting_the_table_from_github(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    table, report = programme_fetch.fetch([1, 2], workers=1, open_group=fake_open, say=lambda line: None)
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as archive:
        buffer = io.BytesIO()
        table.to_parquet(buffer, index=False)
        archive.writestr("programme.parquet", buffer.getvalue())
        archive.writestr("manifest.json", json.dumps(report))
        archive.writestr("../evil.txt", "x")  # anything else in the artifact is ignored
    calls = []

    def get(url, **kwargs):
        calls.append(url)
        if url.endswith("/runs"):
            return Reply({"workflow_runs": [
                {"id": 7, "event": "workflow_dispatch", "status": "completed", "conclusion": "success", "created_at": "2026-11-12T10:00:00Z"},
                {"id": 9, "event": "workflow_dispatch", "status": "completed", "conclusion": "failure", "created_at": "2026-11-12T12:00:00Z"},
                {"id": 8, "event": "push", "status": "completed", "conclusion": "success", "created_at": "2026-11-12T11:00:00Z"}]})
        if url.endswith("/artifacts"):
            return Reply({"artifacts": [{"name": "other", "archive_download_url": "x"},
                                        {"name": "mast-programme", "archive_download_url": "https://zip", "expired": False}]})
        if url == "https://zip":
            return Reply(content=packed.getvalue())
        raise AssertionError(url)

    posted = []
    monkeypatch.setattr(programme_signals.requests, "get", get)
    monkeypatch.setattr(programme_signals.requests, "post", lambda url, **kwargs: posted.append((url, kwargs["json"])) or Reply())
    assert programme_signals.newest_run()["id"] == 9  # the newest run, however it was started or ended
    monkeypatch.setattr(programme_signals.relay, "run_status", lambda run_id: ("completed", "failure"))
    with pytest.raises(RuntimeError, match="ended with 'failure'"):
        programme_signals.collect_newest(say=lambda line: None, poll_s=0)
    states = iter([("queued", None), ("in_progress", None), ("completed", "success")])
    monkeypatch.setattr(programme_signals.relay, "run_status", lambda run_id: next(states))
    said = []
    programme_signals.wait({"id": 8}, said.append, poll_s=0)  # waits through queued and in progress
    assert [line.split("the GitHub job is ")[1] for line in said] == ["queued", "in progress", "completed"]
    manifest = programme_signals.collect(8)
    assert manifest["shots_with_data"] == 2 and "shots with data of 2 asked" in programme_signals.manifest_line(manifest)
    assert sorted(p.name for p in (tmp_path / "data" / "programme").iterdir()) == ["manifest.json", "programme.parquet"]
    assert not (tmp_path / "data" / "evil.txt").exists() and not (tmp_path / "evil.txt").exists()
    assert len(pd.read_parquet(tmp_path / "data" / "programme" / "programme.parquet")) == len(table)
    programme_signals.trigger(40)
    assert posted[0][0].endswith("/actions/workflows/fetch_programme.yml/dispatches")
    assert posted[0][1] == {"ref": config.RELAY_BRANCH, "inputs": {"limit": "40"}}
    pairs = pd.DataFrame({"shot_id": [2, 30], "other": [1, 1]})
    assert programme_signals.coverage(table, pairs) == pytest.approx(2 / 3)  # shot 30 is not in the table
    monkeypatch.delenv("GITHUB_TOKEN")
    with pytest.raises(RuntimeError, match="no GitHub token"):
        programme_signals.through_github(say=lambda line: None)
    with pytest.raises(RuntimeError, match="no GitHub token"):
        programme_signals.collect_newest(say=lambda line: None)
