import numpy as np

from scripts import audit as au, planted_leaks_landmark as plm


def test_pairs_share_draws_and_hazard_shapes():
    a, b = plm.generate("N", 3), plm.generate("N+D", 3)
    assert all(np.array_equal(a[k][0], b[k][0]) for k in a)
    assert all(np.allclose(b[k][1][:, 1] - a[k][1][:, 1], 10 * a[k][0]) for k in a)
    assert all(np.array_equal(a[k][1][:, [0, 2, 3]], b[k][1][:, [0, 2, 3]]) for k in a)
    assert min(v[0][-1] for v in a.values()) >= plm.BURN * plm.TICK - 1e-12
    assert plm.hazard("V", 100) > plm.hazard("V", 20) and plm.hazard("N", 100) == plm.hazard("N", 20)


def test_landmark_labels():
    t = np.round(np.arange(0, 101) * 0.001, 6)
    shots = {1: (t, np.zeros((len(t), 2)), True, float(t[-1]))}
    rows = au.landmark_rows(shots, 0.020, 0.005, 0.030, 0.002)
    to_go = 0.1 - (rows["elapsed"])
    assert ((rows["label"] == -1) == (to_go <= 0.002 + 1e-9)).all()
    assert ((rows["label"] == 1) == ((to_go > 0.002 + 1e-9) & (to_go <= 0.030 + 1e-9))).all()


def test_cluster_gain():
    diff = np.array([1.0, 1.0, 0.0, 0.0])
    g = au.cluster_gain(diff, np.array([1, 1, 2, 2]), n_boot=200)
    assert g[0] == 0.5 and 0 <= g[1] <= g[2] <= 1
