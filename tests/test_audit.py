import numpy as np
import pandas as pd

from scripts import audit as au, planted_leaks as pl


def test_windows_label_and_constant_timing():
    f = au.demo_table(n=6)
    shots = au.shots_from(f, [f"sig{j}" for j in range(4)], True)
    w = au.windows(shots, 0.02, 0.005, 0.03, 0.002)
    for shot, g in w.groupby("shot"):
        td = shots[shot][3]
        assert ((td - g["landmark"] <= 0.03 + 1e-12) == g["positive"]).all()
        assert (td - g["landmark"] >= 0.002 - 1e-12).all()
    timing = np.array(w["timing"].tolist())
    assert np.unique(timing, axis=0).shape[0] == 1  # uniform sampling gives one timing row, no float noise


def test_matched_concordance_simple():
    score = np.array([0.9, 0.8, 0.1, 0.2, 0.95])
    positive = np.array([True, True, False, False, False])
    shot = np.array([1, 2, 3, 4, 1])
    elapsed = np.array([0.10, 0.20, 0.10, 0.30, 0.10])
    c_all, c_clock, d, matched, _ = au.matched_concordance(score, positive, shot, elapsed, 0.01, n_boot=50)
    # positive 0 (shot 1) vs negatives of other shots: 0.1, 0.2 -> 1; positive 1 vs 0.1, 0.2, 0.95 -> 2/3
    assert abs(c_all - (1 + 2 / 3) / 2) < 1e-6
    # clock: positive 0 matches only the shot-3 negative at 0.10 (shot 1's own is excluded); positive 1 has none
    assert c_clock == 1.0 and matched == 1


def test_mixed_mode_runs(tmp_path):
    shots = pl.generate("L0", 3, 0.5)
    rows = []
    for s, (t, x, d) in shots.items():
        for i in range(len(t)):
            rows.append({"shot": s, "time": t[i], "disrupted": int(d), **{f"s{j}": x[i, j] for j in range(12)}})
    path = tmp_path / "t.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    out = au.main([str(path)])
    assert "Mixed mode" in out and "T  timing alone" in out and "smallest detectable" in out
    assert (tmp_path / "audit_report.txt").exists()


def test_demo_flags_the_clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = au.main(["--demo"])
    assert "FLAG: the window label can largely be read off the clock" in out
    assert "FLAG: part of the discrimination is the clock" in out
