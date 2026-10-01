"""Tests for src/data.py using fake in-memory shots, so no network is needed."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import config, data


@pytest.fixture(autouse=True)
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", str(tmp_path / "raw"))
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))


def fake_summary(with_nbi=True):
    t = np.arange(-0.1, 0.3, 0.001)
    variables = {"ip": ("time", -np.full(t.size, 5e5)),
                 "power_radiated": ("time", np.full(t.size, 1e5))}
    if with_nbi:
        variables["power_nbi"] = ("time", np.full(t.size, 1e6))
    return xr.Dataset(variables, coords={"time": t})


def test_missing_variables_become_nan(monkeypatch):
    monkeypatch.setattr(data, "_open_summary", lambda shot_id: fake_summary(with_nbi=False))
    df = data.load_shot(1)
    assert list(df.columns) == ["time", "ip", "power_radiated", "neutron_rates_total", "power_nbi"]
    assert df["neutron_rates_total"].isna().all()
    assert df["power_nbi"].isna().all()
    assert df["ip"].notna().all()


def test_second_load_uses_cache(monkeypatch):
    calls = []

    def opener(shot_id):
        calls.append(shot_id)
        return fake_summary()

    monkeypatch.setattr(data, "_open_summary", opener)
    first = data.load_shot(7)
    second = data.load_shot(7)
    assert calls == [7]
    pd.testing.assert_frame_equal(first, second)


def test_download_shots_caches_and_logs_failures(monkeypatch):
    def opener(shot_id):
        if shot_id == 3:
            raise OSError("connection reset")
        return fake_summary()

    monkeypatch.setattr(data, "_open_summary", opener)
    result = data.download_shots([1, 2, 3], workers=2, retries=1, backoff=0)
    assert sorted(result["downloaded"]) == [1, 2]
    assert result["failed"] == [3]
    skipped = pd.read_csv(config.SKIPPED_CSV)
    assert skipped["shot_id"].tolist() == [3]
    assert "connection reset" in skipped["reason"][0]

    rerun = data.download_shots([1, 2], workers=2, retries=1, backoff=0)
    assert sorted(rerun["cached"]) == [1, 2]
    assert rerun["downloaded"] == []


def test_retry_then_success(monkeypatch):
    attempts = []

    def opener(shot_id):
        attempts.append(1)
        if len(attempts) < 3:
            raise OSError("flaky")
        return fake_summary()

    monkeypatch.setattr(data, "_open_summary", opener)
    assert len(data.load_shot(5, retries=3, backoff=0)) > 0
    assert len(attempts) == 3


def test_empty_group_is_an_error_and_is_retried(monkeypatch):
    attempts = []

    def opener(shot_id):
        attempts.append(1)
        return xr.Dataset()

    monkeypatch.setattr(data, "_open_summary", opener)
    with pytest.raises(ValueError):
        data.load_shot(9, retries=2, backoff=0)
    assert len(attempts) == 3


def test_open_summary_reads_a_real_zarr_v3_store(tmp_path, monkeypatch):
    """Write a small zarr v3 store to disk and read it back through the same code path as the real servers."""
    root = tmp_path / "5.zarr"
    xr.Dataset(attrs={}).to_zarr(root, mode="w", zarr_format=3, consolidated=False)
    fake_summary().to_zarr(root, group="summary", mode="a", zarr_format=3, consolidated=False)
    monkeypatch.setattr(config, "SHOT_ZARR_URL", str(tmp_path / "{shot_id}.zarr"))
    df = data.load_shot(5)
    assert len(df) == 400
    assert df["ip"].iloc[0] == pytest.approx(-5e5)
    assert df["neutron_rates_total"].isna().all()


def test_missing_shot_is_not_retried(monkeypatch):
    attempts = []

    def opener(shot_id):
        attempts.append(1)
        raise FileNotFoundError("no such shot")

    monkeypatch.setattr(data, "_open_summary", opener)
    with pytest.raises(FileNotFoundError):
        data.load_shot(8, retries=3, backoff=0)
    assert len(attempts) == 1


def test_check_connection_reports_both_outcomes(monkeypatch):
    monkeypatch.setattr(data, "_open_summary", lambda shot_id: fake_summary())
    ok, message = data.check_connection(11860)
    assert ok and "400 samples" in message
    monkeypatch.setattr(data, "_open_summary", lambda shot_id: xr.Dataset())
    ok, message = data.check_connection(11860)
    assert not ok and "11860" in message
