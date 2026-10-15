"""Tests for src/extras.py: causal binning, the n=1 fit and merging, on hand-made signals. No network."""
import numpy as np
import pandas as pd
import pytest

from src import config, extras

GRID = np.round(np.arange(0.0, 0.010001, 0.001), 6)  # 1 ms grid, 0 to 10 ms


def test_bins_are_causal():
    t_fast = np.round(np.arange(-0.0005, 0.0101, 0.0001), 6)
    x = np.where(t_fast > 0.005, 100.0, 1.0)  # jump just after 5 ms
    mean = extras.bin_mean(t_fast, x, GRID)
    assert mean[5] == pytest.approx(1.0)  # the bin ending at 5 ms holds only samples <= 5 ms
    assert mean[6] == pytest.approx(100.0)


def test_bin_rms_measures_fluctuation_not_level():
    t_fast = np.round(np.arange(0.00001, 0.010, 0.00001), 6)
    x = 7.0 + 0.5 * np.sin(2 * np.pi * 20000 * t_fast)
    rms = extras.bin_rms(t_fast, x, GRID)
    assert rms[3] == pytest.approx(0.5 / np.sqrt(2), rel=0.05)


def test_hold_last_never_looks_ahead():
    held = extras.hold_last(np.array([0.0005, 0.0042]), np.array([1.0, 2.0]), GRID)
    assert np.isnan(held[0]) and held[1] == 1.0 and held[4] == 1.0 and held[5] == 2.0


def test_n1_amplitude_recovers_a_cosine_and_skips_dead_coils():
    phi = extras.SADDLE_PHI_DEG
    field = 0.3 + 2e-4 * np.cos(np.radians(phi) - 1.0)
    binned = np.column_stack([field, field])
    binned[8, :] = np.nan  # a dead coil, like M09 in shot 27000
    assert extras.n1_amplitude(binned, phi) == pytest.approx([2e-4, 2e-4], rel=1e-6)
    too_few = binned.copy()
    too_few[:7, 1] = np.nan
    assert np.isnan(extras.n1_amplitude(too_few, phi)[1])


def test_early_offset_is_known_only_after_its_window():
    t = np.round(np.arange(0.002, 0.02, 0.001), 6)
    offset, known_at = extras.early_offset(t, np.full(len(t), 5.0))
    assert offset == 5.0 and known_at == pytest.approx(0.002 + config.EXTRAS_BASELINE_MS / 1000)
    assert extras.early_offset(t, np.full(len(t), np.nan))[1] == np.inf


def test_with_extras_merges_or_fills_nan(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    shot = pd.DataFrame({"time": GRID, "ip": np.ones(len(GRID))})
    empty = extras.with_extras(shot, 1)
    assert empty["lm_n1"].isna().all()
    cached = pd.DataFrame({"time": GRID, "lm_n1": 1.0, "mirnov_rms": 2.0, "ip_ref": 3.0, "sol_current": 4.0})
    (tmp_path / "extras").mkdir()
    cached.to_parquet(tmp_path / "extras" / "1.parquet")
    merged = extras.with_extras(shot, 1)
    assert (merged["sol_current"] == 4.0).all() and "ip" in merged
