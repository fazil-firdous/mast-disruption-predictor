import numpy as np
import pandas as pd
import scipy.io

from scripts import endpoint_zhu as ez


def frame(n=160, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        disrupted = s % 3 == 0
        length = rng.uniform(0.6, 1.2)
        t = np.round(np.arange(0.0, length, 0.005), 6)
        tud = (t[-1] - t) if disrupted else np.full(len(t), np.inf)
        x = rng.standard_normal((len(t), 12)) * 0.1
        if disrupted:
            x[:, 0] += np.clip(1 - (t[-1] - t) / 0.3, 0, None)
        else:
            x[:, 1] += np.clip(1 - (t[-1] - t) / 0.3, 0, None)  # a different end-of-record signature for controls
        for k in range(len(t)):
            rows.append({"shot": 1000 + s, "time": t[k], "time_until_disrupt": tud[k],
                         **{f"sig{j}": x[k, j] for j in range(12)}})
    return pd.DataFrame(rows)


def test_window_and_eligibility():
    t = np.round(np.arange(0.0, 1.0, 0.005), 6)
    v = np.zeros((len(t), 12))
    f, n, age = ez.at(t, v, 0.5)
    assert n == 60 and abs(age) < 1e-9 and len(f) == 36
    assert ez.eligible(t, 0.5) and not ez.eligible(t, 0.98)
    assert abs(ez.elapsed(t) - 0.95) < 1e-9


def test_win_matrix_and_concordance():
    w = ez.win_matrix(np.array([0.9, 0.2]), np.array([[0.5, np.nan], [0.2, 0.1]]))
    assert w[0, 0] == 1 and np.isnan(w[0, 1]) and w[1, 0] == 0.5 and w[1, 1] == 1
    mask = np.isfinite(w)
    assert abs(ez.concordance(w, mask) - (1.0 + 0.75) / 2) < 1e-12


def test_run_end_to_end(tmp_path):
    f = frame()
    d = tmp_path / "20ja072_fig4_data"
    d.mkdir()
    scipy.io.savemat(d / "C_Mod_for_nn.mat", {c: f[c].to_numpy() for c in f.columns})
    out = ez.run(str(tmp_path))
    assert "== C-Mod" in out and "Delta = C_E - C_M" in out and "timing-only concordance" in out


def test_delta_range_and_reading():
    rng = np.random.default_rng(1)
    w_e = (rng.random((40, 60)) < 0.8).astype(float)
    w_m = (rng.random((40, 60)) < 0.6).astype(float)
    mask = rng.random((40, 60)) < 0.7
    d = ez.delta_range(w_e, w_m, mask, n_boot=200)
    assert d[1] < d[0] < d[2] and d[1] > 0
    assert "depends materially" in ez.reading(d)
    assert "no material" in ez.reading((0.01, -0.02, 0.04))


def test_case_without_eligible_control_is_left_out():
    w = np.array([[1.0, 0.0], [np.nan, np.nan]])
    assert ez.concordance(w, np.isfinite(w)) == 0.5
