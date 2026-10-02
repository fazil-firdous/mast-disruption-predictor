"""Tests for src/features.py on synthetic shots, no network."""
import numpy as np
import pandas as pd
import pytest

from src import config, features
from tests.synthetic import make_shot


def same(a, b):
    """Exact equality of two feature dicts, treating NaN == NaN."""
    assert a.keys() == b.keys()
    for key in a:
        if np.isnan(a[key]):
            assert np.isnan(b[key]), key
        else:
            assert a[key] == b[key], key


@pytest.mark.parametrize("t_end", [0.05, 0.15, 0.2, 0.219])
def test_causality_future_samples_do_not_matter(t_end):
    shot = make_shot("quench", t_quench=0.22)
    changed = shot.copy()
    future = changed["time"] > t_end + 1e-9
    rng = np.random.default_rng(1)
    for column in ["ip", "power_radiated", "neutron_rates_total", "power_nbi"]:
        changed.loc[future, column] = rng.normal(0, 1e7, future.sum())
    sig_a, sig_b = features.prepare_signals(shot), features.prepare_signals(changed)
    t_start = features.find_t_start(sig_a)
    same(features.window_features(sig_a, t_end, t_start), features.window_features(sig_b, t_end, t_start))


def test_causality_whole_table_rows_unchanged():
    shot = make_shot("quench", t_quench=0.22)
    cut = 0.15
    changed = shot.copy()
    changed.loc[changed["time"] > cut, "ip"] *= 3
    a = features.shot_windows(shot)
    b = features.shot_windows(changed)
    early_a = a[a["t_end"] <= cut].reset_index(drop=True)
    early_b = b[b["t_end"] <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b, check_exact=True)


def test_window_uses_exactly_window_ms_of_samples():
    shot = make_shot("rampdown")
    shot["power_radiated"] = shot["time"] * 1000  # prad equals the sample index in ms
    sig = features.prepare_signals(shot)
    f = features.window_features(sig, 0.2, features.find_t_start(sig))
    # samples 181..200 ms: 20 samples, mean 190.5, delta 19
    assert f["prad_delta"] == pytest.approx(config.WINDOW_MS - 1)
    assert f["prad_mean"] == pytest.approx(200 - (config.WINDOW_MS - 1) / 2)
    assert f["prad_last"] == pytest.approx(200)


def test_slope_in_units_per_second():
    shot = make_shot("rampdown", noise=0)
    shot["power_radiated"] = 1e5 + 2e6 * shot["time"]  # 2 MW/s
    sig = features.prepare_signals(shot)
    f = features.window_features(sig, 0.2, features.find_t_start(sig))
    assert f["prad_slope"] == pytest.approx(2e6, rel=1e-6)


def test_ip_sign_does_not_matter():
    pos = features.shot_windows(make_shot("quench", sign=1, seed=3))
    neg = features.shot_windows(make_shot("quench", sign=-1, seed=3))
    pd.testing.assert_frame_equal(pos, neg)


def test_windows_start_after_t_start_plus_window_and_follow_stride():
    shot = make_shot("rampdown")
    win = features.shot_windows(shot)
    t_start = win["t_start"].iloc[0]
    assert (win["t_end"] >= t_start + config.WINDOW_MS / 1000 - 1e-9).all()
    steps = np.round(np.diff(win["t_end"]) * 1000, 6)
    assert set(steps) == {config.STRIDE_MS}


def test_no_window_at_or_after_t_stop():
    shot = make_shot("quench", t_quench=0.22)
    win = features.shot_windows(shot, t_stop=0.22)
    assert len(win) > 0
    assert (win["t_end"] < 0.22).all()


def test_windows_stop_when_plasma_is_gone():
    shot = make_shot("rampdown", t_rampdown=0.25, rampdown_ms=60)
    win = features.shot_windows(shot)
    # |Ip| reaches IP_ON (50 kA of 700 kA) about 56 ms into the ramp-down
    assert win["t_end"].max() < 0.25 + 0.060
    assert (win["ip_abs_last"] >= config.IP_ON).all()


def test_missing_signal_gives_nan_and_flag():
    shot = make_shot("rampdown")
    shot["neutron_rates_total"] = np.nan
    win = features.shot_windows(shot)
    assert (win["has_neutrons"] == 0).all()
    assert win[[f"neutrons_{s}" for s in features.STATS] + ["neutron_slope_rel"]].isna().all().all()
    assert (win["has_prad"] == 1).all()


def test_running_max_ignores_later_spike():
    shot = make_shot("quench", t_quench=0.22, spike=True, noise=0)
    sig = features.prepare_signals(shot)
    f = features.window_features(sig, 0.2, features.find_t_start(sig))
    assert f["ip_frac_of_running_max"] == pytest.approx(1.0)


def test_pnbi_on_and_ratios():
    shot = make_shot("rampdown", noise=0)
    sig = features.prepare_signals(shot)
    t_start = features.find_t_start(sig)
    f = features.window_features(sig, 0.2, t_start)
    assert f["pnbi_on"] == 1.0
    assert f["prad_over_pnbi"] == pytest.approx(f["prad_mean"] / (1.5e6 + config.PNBI_OFFSET))
    assert f["t_since_start"] == pytest.approx(0.2 - t_start)


def test_feature_names_match_window_features():
    shot = make_shot("rampdown")
    win = features.shot_windows(shot)
    assert set(features.feature_names()) == set(win.columns) - {"t_end", "t_start"}


def test_tiny_shot_has_no_windows():
    assert len(features.shot_windows(make_shot("tiny"))) == 0


def test_wobbly_breakdown_is_not_a_plasma_end():
    shot = make_shot("rampdown", noise=0)
    t = shot["time"]
    # |Ip| touches 60 kA, falls back to 30 kA for 40 ms, then the normal ramp-up resumes
    shot.loc[(t >= 0.0) & (t < 0.005), "ip"] = 6e4
    shot.loc[(t >= 0.005) & (t < 0.045), "ip"] = 3e4
    win = features.shot_windows(shot)
    assert len(win) > 0
    assert win["t_end"].min() < 0.05
    assert win["t_end"].max() > 0.25


def test_current_already_on_at_first_sample_gives_no_windows():
    shot = make_shot("rampdown")
    shot["ip"] = 3e5
    sig = features.prepare_signals(shot)
    assert np.isnan(features.find_t_start(sig))
    assert len(features.shot_windows(shot)) == 0


def test_breakdown_dip_is_not_a_plasma_end():
    shot = make_shot("rampdown", noise=0)
    t = shot["time"]
    # spike past MIN_PEAK_IP just after breakdown, then a one-sample dip to 10 kA (as in shot 16568)
    shot.loc[np.isclose(t, 0.002), "ip"] = 1.2e5
    shot.loc[np.isclose(t, 0.004), "ip"] = 1e4
    win = features.shot_windows(shot)
    assert win["t_end"].max() > 0.25
