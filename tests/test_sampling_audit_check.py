"""Smoke test for scripts/sampling_audit_check.py on made-up files laid out like the Dataverse release. No network."""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

from scripts import sampling_audit_check as chk


def write(path, n=120, dense=True, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for shot in range(n):
        disrupted = shot % 3 == 0
        t = np.concatenate([np.arange(0.0, 1.0, 0.025), np.arange(1.0, 1.5, 0.002 if (disrupted and dense) else 0.025)])
        rows.append(pd.DataFrame({"shot": 1000 + shot, "time": np.round(t, 4), "q95": rng.normal(3, 0.3, len(t)),
                                  "time_until_disrupt": (t[-1] - t) if disrupted else np.full(len(t), np.nan)}))
    f = pd.concat(rows)
    scipy.io.savemat(path, {k: f[k].to_numpy() for k in f.columns})
    return sorted(f["shot"].unique())


def test_check_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(chk.sa.config, "RESULTS_DIR", str(tmp_path / "results"))
    os.makedirs(tmp_path / "20ja072_fig4_data")
    os.makedirs(tmp_path / "20ja072_fig6_data")
    for key in ("C_Mod", "d3d", "EAST"):
        shots = write(tmp_path / "20ja072_fig4_data" / f"{key}_for_nn.mat", dense=key != "C_Mod")
    shots = np.array(shots)
    scipy.io.savemat(tmp_path / "20ja072_fig6_data" / "EAST_data_partition.mat",
                     {"trainingShotlist": shots[:60], "validationShotlist": shots[60:80], "testShotlist": shots[80:100],
                      "trainingClasses": (np.arange(60) % 3 == 0).astype(int)})
    monkeypatch.setattr(sys, "argv", ["x", str(tmp_path)])
    chk.main()
    out = capsys.readouterr().out
    assert "shots in the file but in no list: 20" in out and "overlaps between lists: training/validation 0" in out
    assert "trainingClasses agree with our label" in out and "DIII-D residual-timing fit, diagnosis" in out
    assert "beyond 1 ms" in out
