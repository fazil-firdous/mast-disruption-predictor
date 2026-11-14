"""Tests for scripts/sampling_audit.py on made-up records: the spacing and change-rate statistics, the 40 ms cut, the
AUC with ties, and the summary. No network, no real data."""
import numpy as np
import pandas as pd
import scipy.io

from scripts import sampling_audit as sa


def test_spacing_and_cut():
    t = np.round(np.arange(0.0, 1.0001, 0.025), 4)
    assert abs(sa.spacing(t, t[-1] - 0.040) - 0.025) < 1e-9
    dense = np.concatenate([np.arange(0.0, 0.9, 0.025), np.arange(0.9, 1.0001, 0.002)])
    assert abs(sa.spacing(dense, dense[-1] - 0.0) - 0.002) < 1e-9  # dense end seen without a cut
    assert abs(sa.spacing(dense, dense[-1] - 0.150) - 0.025) < 1e-9  # a long enough cut removes it
    sparse = np.arange(0.0, 1.0001, 0.1)  # fewer than 2 slices in the 100 ms window: last two slices before the cut
    assert abs(sa.spacing(sparse, 0.96) - 0.1) < 1e-9
    assert np.isnan(sa.spacing(np.array([0.5]), 0.6))


def test_change_rate_after_forward_fill():
    t = np.arange(0.0, 1.0001, 0.1)
    x = np.arange(len(t), dtype=float)[:, None]  # changes at every slice, slices every 100 ms
    r = sa.change_rate(t, x, 0.96)
    assert 0.0 < r <= 0.06  # on a 5 ms grid, one change in about 20 steps
    t2 = np.arange(0.0, 1.0001, 0.005)
    assert sa.change_rate(t2, np.arange(len(t2), dtype=float)[:, None], 0.96) == 1.0


def test_auc_with_ties():
    assert sa.auc(np.array([1.0, 1.0, 0.0, 0.0]), np.array([1, 1, 0, 0])) == 1.0
    assert sa.auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([1, 1, 0, 0])) == 0.5
    assert abs(sa.auc(np.array([2.0, np.nan, 1.0, 3.0]), np.array([1, 1, 0, 0])) - 0.5) < 1e-12


def make(path, dense_disrupted, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for shot in range(40):
        disrupted = shot % 4 == 0
        step = 0.010 if (disrupted and dense_disrupted) else 0.100
        t = np.round(np.arange(0.0, 1.0, step), 4)
        tud = (t[-1] - t) if disrupted else np.full(len(t), np.nan)
        rows.append(pd.DataFrame({"shot": shot, "time": t, "time_until_disrupt": tud, "q95": rng.normal(3, 0.1, len(t))}))
    f = pd.concat(rows)
    scipy.io.savemat(path, {k: f[k].to_numpy() for k in f.columns})


def test_end_to_end_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(sa.config, "RESULTS_DIR", str(tmp_path))
    make(tmp_path / "EAST_for_nn.mat", True)
    make(tmp_path / "C_Mod_for_nn.mat", False)
    tables = {}
    for key in ("EAST", "C_Mod"):
        frame = sa.load(str(tmp_path / f"{key}_for_nn.mat"))
        tables[key] = sa.shot_table(frame, ["q95"])
    out = sa.text(tables, n_boot=200)
    assert "== EAST: 40 shots (10 disrupted)" in out and "PRIMARY S1 (spacing in the 100 ms before the cut), AUC 1.000" in out
    cmod, east = out.split("== EAST")[0].split("== C-Mod")[1], out.split("== EAST")[1]
    assert "the sampling carries outcome information" in east.split("S2")[0]
    assert "AUC 0.500" in cmod and "no such signal by this statistic" in cmod.split("S2")[0]
