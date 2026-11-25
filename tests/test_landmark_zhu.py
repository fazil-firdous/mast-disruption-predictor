import numpy as np
import pandas as pd
import scipy.io

from scripts import landmark_zhu as lz


def zhu_like(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        disrupted = s % 2 == 0
        length = rng.uniform(0.8, 1.5)
        t = np.round(np.arange(0.0, length, 0.005), 6)
        td = t[-1] + 0.040 if disrupted else np.inf  # records stop 40 ms before the disruption
        tud = td - t
        x = rng.standard_normal((len(t), 12)) * 0.1
        if disrupted:
            x[:, 0] += np.clip(1 - (td - t) / 0.15, 0, None)
        for k in range(len(t)):
            rows.append({"shot": 2000 + s, "time": t[k], "time_until_disrupt": tud[k],
                         **{f"sig{j}": x[k, j] for j in range(12)}})
    return pd.DataFrame(rows)


def test_disruption_time_from_time_until_disrupt():
    f = zhu_like(n=4)
    shots = lz.disrupted_shots(f, [f"sig{j}" for j in range(12)])
    assert len(shots) == 2
    for t, _, _, td in shots.values():
        assert abs(td - (t[-1] + 0.040)) < 1e-9


def test_run_end_to_end(tmp_path):
    f = zhu_like()
    d = tmp_path / "20ja072_fig4_data"
    d.mkdir()
    scipy.io.savemat(d / "C_Mod_for_nn.mat", {c: f[c].to_numpy() for c in f.columns})
    out = lz.run(str(tmp_path), n_boot=100)
    assert "== C-Mod" in out and "schedule over clock" in out and "signals over clock alone" in out
    assert "t_d minus last slice, median 40.0 ms" in out


def test_record_stopping_sensitivity_runs(tmp_path):
    f = zhu_like()
    d = tmp_path / "20ja072_fig4_data"
    d.mkdir()
    scipy.io.savemat(d / "C_Mod_for_nn.mat", {c: f[c].to_numpy() for c in f.columns})
    out = lz.run(str(tmp_path), n_boot=50, record_stopping=True)
    assert "iterations chosen on held-out training records" in out


def test_anchor_check():
    from scripts import anchor_check as ac
    spread, gap, missing = ac.per_record(zhu_like(n=10))
    assert len(gap) == 5 and np.allclose(gap, 0.040) and spread.max() < 1e-9 and missing.sum() == 0
