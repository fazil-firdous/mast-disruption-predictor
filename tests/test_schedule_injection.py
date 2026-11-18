import numpy as np
import pandas as pd
import scipy.io

from scripts import schedule_injection as si


def frame(n_shots=240, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_shots):
        disrupted = s % 4 == 0
        t = np.round(np.arange(0.05, 0.05 + 0.005 * rng.integers(150, 260), 0.005), 6)
        tud = (t[-1] - t) if disrupted else np.full(len(t), np.inf)
        x = rng.standard_normal((len(t), 12)) * 0.1
        if disrupted:
            x[:, 0] += np.clip(1 - (t[-1] - t) / 0.2, 0, None)  # a precursor in the last 200 ms
        for k in range(len(t)):
            rows.append({"shot": 1000 + s, "time": t[k], "time_until_disrupt": tud[k],
                         **{f"sig{j}": x[k, j] for j in range(12)}})
    return pd.DataFrame(rows)


def test_thin_keeps_the_same_values_and_injects_only_disrupted_shots():
    f = frame(12)
    u, i = si.thin(f, False), si.thin(f, True)
    for shot, g in f.groupby("shot"):
        gu, gi = u[u["shot"] == shot], i[i["shot"] == shot]
        disrupted = np.isfinite(g["time_until_disrupt"]).any()
        assert g["time"].max() in set(gu["time"])  # the last slice is always kept
        assert set(gu.index) <= set(gi.index) <= set(g.index)  # kept slices are original rows, values unchanged
        gaps = np.diff(np.sort(gu["time"].to_numpy()))
        assert np.allclose(gaps, 0.05)
        if disrupted:
            last = g["time"].max()
            assert set(g.index[g["time"] >= last - 0.300 - 1e-9]) <= set(gi.index)
        else:
            assert set(gi.index) == set(gu.index)


def test_run_end_to_end(tmp_path):
    f = frame()
    path = tmp_path / "C_Mod_for_nn.mat"
    scipy.io.savemat(path, {c: f[c].to_numpy() for c in f.columns})
    out = si.run(str(path))
    assert "PRIMARY, R0: Inflation" in out and "Injection check" in out
    assert "Median slices in the window" in out


def test_reading():
    ok_i, ok_u = (0.95, 0.9, 0.99), (0.5, 0.45, 0.56)
    assert "raises" in si.reading((0.05, 0.02, 0.08), ok_i, ok_u)
    assert "small" in si.reading((0.01, 0.002, 0.02), ok_i, ok_u)
    assert "not shown" in si.reading((0.01, -0.01, 0.03), ok_i, ok_u)
    assert "nothing is read" in si.reading((0.05, 0.02, 0.08), (0.7, 0.6, 0.8), ok_u)
    assert si.reading((0.05, 0.02, 0.08), ok_i, (0.6, 0.55, 0.65)).startswith("(relative")


def test_check_nesting_and_r2_identity(tmp_path):
    from scripts import schedule_injection_check as c
    f = frame(40, seed=1)
    path = tmp_path / "C_Mod_for_nn.mat"
    scipy.io.savemat(path, {k: f[k].to_numpy() for k in f.columns})
    out = c.run(str(path))
    assert "U is a subset of I in 40 of 40 shots" in out
    assert "non-disrupted shots with identical R2 features 30 of 30" in out
    # exactly uniform 5 ms records: every grid point falls on a U slice, so R2 is identical in every disrupted shot
    assert "identical 36 features under U and I: 10 of 10" in out
