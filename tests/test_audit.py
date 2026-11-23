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
    assert "Mixed mode" in out and "T  timing alone" in out and "bootstrap interval half-widths" in out
    assert (tmp_path / "audit_report.txt").exists()


def test_demo_flags_the_clock(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = au.main(["--demo"])
    assert "FLAG: the window label can largely be read off the clock" in out
    assert "FLAG: part of the discrimination is the clock" in out


def _shots(times_list):
    return {i: (t, np.zeros((len(t), 2)), True, float(t[-1])) for i, t in enumerate(times_list)}


def test_float_regression_membership_and_features_exact():
    """Integer ticks of 1 ms, landmarks on the tick grid: exact sample membership, count, age and gap, also when the
    stored times carry sub-tick representation errors (cumulative sums, tiny jitter)."""
    exact = np.round(np.arange(0, 200) * 0.001, 6)
    cumsum = np.cumsum(np.r_[0.0, np.full(199, 0.001)])  # accumulates float error
    jitter = exact + np.random.default_rng(0).uniform(-1e-12, 1e-12, 200)
    rows = {k: au.windows(_shots([t]), 0.020, 0.005, 0.030, 0.002) for k, t in
            (("exact", exact), ("cumsum", cumsum), ("jitter", jitter))}
    base = rows["exact"]
    for t in (exact,):
        for landmark, timing in zip(base["landmark"], base["timing"]):
            idx = np.where((t > landmark - 0.020 + 1e-9) & (t <= landmark + 1e-9))[0]
            assert len(idx) == 20 and idx[-1] == int(round(landmark / 0.001))
            assert timing == [20, 0.0, 0.001]
    for k in ("cumsum", "jitter"):
        assert rows[k]["timing"].tolist() == base["timing"].tolist()
        assert np.allclose(np.array(rows[k]["landmark"]), np.array(base["landmark"]))


def test_identical_timing_vectors_give_tied_scores():
    x_tr = np.tile([20, 0.0, 0.001], (50, 1))
    y_tr = np.r_[np.ones(10), np.zeros(40)].astype(bool)
    s = au.logistic_scores(x_tr, y_tr, np.tile([20, 0.0, 0.001], (30, 1)))
    assert np.ptp(s) == 0
    from scripts import sampling_audit as sa
    assert sa.auc(s, np.r_[np.ones(10), np.zeros(20)].astype(bool)) == 0.5


def test_check_table_catches_common_mistakes():
    base = pd.DataFrame({"shot": [1, 1, 2, 2], "time": [0.0, 0.001, 0.0, 0.001], "x": [1.0, 2.0, 3.0, 4.0]})
    e, w = au.check_table(base, ["x"], True)
    assert not e and any("only 2 shots" in m for m in w)
    dup = pd.concat([base, base.iloc[:1]])
    assert any("repeat a (shot, time)" in m for m in au.check_table(dup, ["x"], True)[0])
    mixed = base.assign(disrupted=[1, 0, 0, 0])
    assert any("changes within" in m for m in au.check_table(mixed, ["x"], False)[0])
    one_class = base.assign(disrupted=1)
    assert any("both disrupted and non-disrupted" in m for m in au.check_table(one_class, ["x"], False)[0])
    assert any("needs a 'disrupted' column" in m for m in au.check_table(base, ["x"], False)[0])
    ms = base.assign(time=base["time"] * 1e7)
    assert any("seconds" in m for m in au.check_table(ms, ["x"], True)[1])
    early = base.assign(t_disrupt=-1.0)
    assert any("before their first slice" in m for m in au.check_table(early, ["x"], True)[0])
