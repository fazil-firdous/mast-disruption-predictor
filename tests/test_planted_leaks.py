import numpy as np

from scripts import planted_leaks as pl


def test_generator_plants_what_it_says():
    base = pl.generate("L0", 5, 1.0)
    sparse = pl.generate("L_T", 5, 1.0)
    short = pl.generate("L_D", 5, 1.0)
    assert len(base) == pl.N_SHOTS and sum(v[2] for v in base.values()) == pl.N_SHOTS // 3
    dt = [np.median(np.diff(t)) for t, _, d in sparse.values() if not d]
    assert abs(np.median(dt) - pl.SPARSE) < 1e-9
    assert all(abs(np.median(np.diff(t)) - pl.STEP) < 1e-9 for t, _, d in sparse.values() if d)
    assert max(t[-1] for t, _, d in short.values() if d) < min(t[-1] for t, _, d in short.values() if not d)
    a, b = pl.generate("L_E", 7, 1.0), pl.generate("L_E", 7, 1.0)
    assert all(np.array_equal(a[k][1], b[k][1]) for k in a)


def test_timing_row():
    t = np.round(np.arange(0.0, 1.0, 0.01), 6)
    n, age, gap = pl.timing_row(t)
    assert n == 30 and abs(age) < 1e-9 and abs(gap - 0.01) < 1e-9
