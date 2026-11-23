import numpy as np

from scripts import planted_leaks_all as pa


def test_families():
    a0, ac, as_ = pa.generate("A0", 1), pa.generate("A_C", 1), pa.generate("A_S", 1)
    assert len(a0) == pa.N_SHOTS and all(v[2] and v[3] == v[0][-1] for v in a0.values())
    assert max(v[0][-1] for v in ac.values()) < 0.24 and min(v[0][-1] for v in ac.values()) >= 0.179
    t = next(iter(as_.values()))[0]
    assert abs(np.median(np.diff(t[-60:])) - 0.00025) < 1e-9 and abs(np.median(np.diff(t[:50])) - 0.001) < 1e-9
