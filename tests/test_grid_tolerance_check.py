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


# Boundary semantics of harmonisation.grid_points as registered (a grid point takes the latest slice at or before it,
# within 1e-9 s; older than max_age + 1e-9 s is empty). These tests document the behaviour the withdrawn C-Mod numbers
# came from; they do not endorse a tolerance.
def _ages(t, cut, step=0.05, max_age=0.05, window=0.3):
    v = np.arange(len(t), dtype=float)[:, None]
    vals, age = hm.grid_points(np.asarray(t, dtype=float), v, cut, step, max_age=max_age, window=window)
    return vals[:, 0], age


def test_exact_tick_is_taken():
    t = np.arange(0, 0.45, 0.05)
    vals, age = _ages(t, t[-1])
    assert np.allclose(age, 0, atol=1e-12) and vals[-1] == len(t) - 1


def test_slice_a_hair_after_the_point_is_not_available():
    t = np.arange(0, 0.45, 0.05).copy()
    t[-2] += 1e-7  # 0.1 us late: later than the 1e-9 s tolerance, so not available at its grid point
    vals, age = _ages(t, t[-1])
    assert vals[-2] == len(t) - 3 and abs(age[-2] - 0.05) < 1e-6


def test_slice_within_1e9_after_the_point_is_available():
    t = np.arange(0, 0.45, 0.05).copy()
    t[-2] += 5e-10
    vals, _ = _ages(t, t[-1])
    assert vals[-2] == len(t) - 2


def test_maximum_age_boundary_and_empty_points():
    t = np.array([0.0, 0.05, 0.10, 0.30, 0.35, 0.40])  # a 200 ms gap
    vals, age = _ages(t, 0.40)
    # grid points 0.15 ... 0.40: 0.15 takes the slice at 0.10 (age exactly the maximum, kept); 0.20 and 0.25 are
    # older than the maximum and empty (value NaN; the age is still reported)
    assert vals[0] == 2 and abs(age[0] - 0.05) < 1e-12
    assert np.isnan(vals[1:3]).all() and (age[1:3] > 0.05).all()
    assert np.isfinite(vals[3:]).all()


def test_origin_moves_with_the_first_slice():
    t = np.arange(0, 0.45, 0.05)
    _, a = _ages(t, t[-1])
    _, b = _ages(np.concatenate([[0.0025], t[1:]]), t[-1])  # first slice 2.5 ms later: every grid point moves 2.5 ms
    assert np.allclose(a, 0, atol=1e-12) and np.allclose(b, 0.0025, atol=1e-9)


def test_repeated_time_stamps_take_the_last_row():
    t = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.30, 0.35, 0.40])
    vals, _ = _ages(t, 0.40)
    assert 7.0 in set(vals) and 6.0 not in set(vals)  # at 0.30 the later of the two rows
