"""Tests for src/alarms.py on hand-made score sequences."""
import numpy as np
import pytest

from src import alarms

TIMES = np.arange(10) * 0.005


def test_needs_k_in_a_row():
    scores = [0.9, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
    assert alarms.first_alarm(scores, TIMES, 0.5, k=2) == pytest.approx(TIMES[5])
    assert alarms.first_alarm(scores, TIMES, 0.5, k=1) == pytest.approx(TIMES[0])
    assert np.isnan(alarms.first_alarm(scores, TIMES, 0.5, k=3))


def test_threshold_is_inclusive():
    scores = [0.5, 0.5] + [0.0] * 8
    assert alarms.first_alarm(scores, TIMES, 0.5, k=2) == pytest.approx(TIMES[1])


def test_no_alarm_below_threshold():
    assert np.isnan(alarms.first_alarm(np.full(10, 0.49), TIMES, 0.5, k=2))


def test_alarm_level_is_min_of_last_k():
    level = alarms.alarm_level([0.2, 0.8, 0.6, 0.9], k=2)
    assert np.isnan(level[0])
    assert list(level[1:]) == [0.2, 0.6, 0.6]


def test_alarm_times_agree_with_first_alarm():
    rng = np.random.default_rng(0)
    thresholds = np.linspace(0, 1, 51)
    for _ in range(50):
        scores = rng.random(10)
        fast = alarms.alarm_times(scores, TIMES, thresholds, k=2)
        for threshold, t in zip(thresholds, fast):
            slow = alarms.first_alarm(scores, TIMES, threshold, k=2)
            assert (np.isnan(t) and np.isnan(slow)) or t == slow


def test_alarm_times_short_shot():
    assert np.isnan(alarms.alarm_times([0.9], TIMES[:1], np.array([0.5]), k=2)).all()
    assert np.isnan(alarms.alarm_times([], [], np.array([0.5]), k=2)).all()
