"""Tests for src/labels.py using synthetic shots, no network."""
import numpy as np
import pytest

from src import labels
from tests.synthetic import make_shot


@pytest.mark.parametrize("spike", [True, False])
@pytest.mark.parametrize("sign", [1, -1])
def test_detector_finds_quench_within_2ms(spike, sign):
    for seed in range(5):
        shot = make_shot("quench", t_quench=0.22, spike=spike, sign=sign, seed=seed)
        result = labels.detect_disruption(shot["time"], shot["ip"])
        assert result["status"] == "disrupted"
        assert abs(result["t_disrupt"] - 0.22) <= 0.002


def test_detector_ignores_slow_rampdown():
    for seed in range(5):
        shot = make_shot("rampdown", seed=seed)
        result = labels.detect_disruption(shot["time"], shot["ip"])
        assert result["status"] == "clean"
        assert not result["detected"]
        assert np.isnan(result["t_disrupt"])


def test_detector_unknown_for_tiny_current():
    shot = make_shot("tiny")
    result = labels.detect_disruption(shot["time"], shot["ip"])
    assert result["status"] == "unknown"
    assert not result["detected"]


def test_detector_handles_nan_samples():
    shot = make_shot("quench", t_quench=0.2)
    shot.loc[50:52, "ip"] = np.nan
    result = labels.detect_disruption(shot["time"], shot["ip"])
    assert abs(result["t_disrupt"] - 0.2) <= 0.002


def test_quench_rate_is_positive_and_peak_reported():
    shot = make_shot("quench", t_quench=0.22, spike=False)
    result = labels.detect_disruption(shot["time"], shot["ip"])
    assert result["quench_rate"] > 1e7
    assert result["peak_ip"] == pytest.approx(7e5, rel=0.05)


@pytest.mark.parametrize("comment, expected", [
    ("DISRUPTION AT 220MS", True),
    ("Plasma disrupts at ~310 ms (mode-lock)", True),
    ("no disruption this time", False),
    ("NOT DISRUPTED", False),
    ("non-disruptive shot", False),
    ("didn't disrupt", False),
    ("didnt disrupt", False),
    ("good shot", False),
    (None, False),
    (float("nan"), False),
])
def test_note_label(comment, expected):
    assert labels.note_label(comment) is expected


@pytest.mark.parametrize("comment, expected", [
    ("DISRUPTION AT 220MS", 0.22),
    ("Plasma disrupts at ~310 ms", 0.31),
    ("shot disrupted AT 0.22S", 0.22),
    ("H-mode at 212 ms then disrupted", float("nan")),
    ("disrupts at 0.302 broken pellet at 0.263s", 0.302),
    ("Disrupts early at 0.230 at IRE", 0.23),
    ("DISRUPTED AT 190", 0.19),
    ("disrupted at 5 coils", float("nan")),
    ("disruption at the same time", float("nan")),
    (None, float("nan")),
])
def test_parse_note_time(comment, expected):
    got = labels.parse_note_time(comment)
    if np.isnan(expected):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(expected)


@pytest.mark.parametrize("source, note, signal, expected", [
    ("signal", False, True, True),
    ("signal", True, False, False),
    ("note", True, False, True),
    ("note", False, True, False),
    ("both", True, True, True),
    ("both", True, False, False),
])
def test_final_label_switch(source, note, signal, expected):
    assert labels.final_label(note, signal, source) is expected


def test_final_label_rejects_unknown_source():
    with pytest.raises(ValueError):
        labels.final_label(True, True, "magic")
