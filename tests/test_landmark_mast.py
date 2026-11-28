import numpy as np
import pandas as pd

from scripts import landmark_mast as lm


def fake_windows(n=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n):
        lost = s % 2 == 0
        tq = rng.uniform(0.25, 0.45)
        for t in np.arange(0.05, tq + 0.03, 0.005):
            x = rng.standard_normal()
            sig = x + (2.0 if lost and tq - t < 0.1 else 0.0)
            rows.append({"shot_id": 1000 + s, "t_end": round(t, 6), "disrupted": lost and s % 4 == 0,
                         "candidate": lost and s % 4 == 2, "t_quench": tq if lost else np.nan,
                         "t_since_start": t, "ref_mean": 1.0, "sig": sig,
                         "split": "train" if s % 5 else "test", "split_temporal": "train" if s < 80 else "test"})
    return pd.DataFrame(rows)


def test_landmark_frame_labels():
    frame, late = lm.landmark_frame(fake_windows(20))
    to_go = frame["t_quench"] - frame["t_end"]
    assert (to_go > lm.LEAD_S).all() and late > 0
    assert ((frame["y"] == 1) == (to_go <= lm.HORIZON_S + 1e-9)).all()
    assert set(frame["kind"]) == {"recorded", "candidate"}


def test_split_and_summary():
    frame, _ = lm.landmark_frame(fake_windows())
    cols = {"C": ["t_since_start"], "P": ["t_since_start", "ref_mean"], "M": ["t_since_start", "ref_mean"],
            "F": ["t_since_start", "ref_mean", "sig"]}
    rows = []
    for seed in range(1, 6):
        r = lm.one_split(frame, frame["split"], cols)
        rows.append(lm.split_row(seed, r))
    assert rows[0]["G_F_M"] > 0  # the planted signal helps
    chrono = lm.one_split(frame, frame["split_temporal"], cols)
    out = lm.summary(rows, chrono)
    assert "PRIMARY: diagnostics over machine-side" in out and "Chronological split" in out
