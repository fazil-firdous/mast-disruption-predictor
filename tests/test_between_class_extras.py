import numpy as np

from scripts import between_class_extras as bc


def test_quantised_spacing_removes_float_residues():
    rng = np.random.default_rng(0)
    shots = {}
    for k in range(60):
        start = rng.uniform(0, 5)
        t = start + np.arange(0, rng.uniform(0.6, 1.5), 0.005)
        shots[k] = (t, k % 2 == 0)
    tab = bc.table_of(shots)
    assert tab["s1q"].nunique() == 1 and abs(tab["s1q"].iloc[0] - 0.005) < 1e-12
    lines = bc.machine_lines(tab, n_boot=50)
    assert any("ten shot-number bins" in l for l in lines)
