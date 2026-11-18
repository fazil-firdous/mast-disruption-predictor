import numpy as np

from scripts import grid_tolerance_check as gc
from scripts import harmonisation as hm


def test_grid_ages_match_harmonisation_at_its_tolerance():
    rng = np.random.default_rng(0)
    t = np.cumsum(np.full(400, 0.005) + rng.normal(0, 1e-7, 400))
    vals = rng.standard_normal((400, 2))
    cut = t[-1] - 0.04
    _, age = hm.grid_points(t, vals, cut, 0.05, max_age=0.05)
    assert np.array_equal(gc.grid_ages(t, cut, 0.05, 1e-9, max_age=0.05), age, equal_nan=True)


def test_a_tiny_offset_is_missed_at_1e9_but_not_at_1e6():
    t = np.array([0.0, 0.05 + 5e-8, 0.10 + 5e-8, 0.15 + 5e-8, 0.20, 0.25, 0.30, 0.35, 0.40])
    cut = 0.40 - 0.04
    strict = gc.grid_ages(t, cut, 0.05, 1e-9, window=0.3)
    loose = gc.grid_ages(t, cut, 0.05, 1e-6, window=0.3)
    assert np.nanmax(strict) > 0.04  # a point just before a slice takes the slice 50 ms older
    assert np.nanmax(np.abs(loose)) < 1e-6
