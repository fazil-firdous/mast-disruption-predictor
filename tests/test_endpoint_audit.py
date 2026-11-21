import numpy as np
import pandas as pd

from scripts import endpoint_audit as ea
from scripts import saddle_correction as sc


def scores_frame():
    t = np.round(np.arange(0.05, 0.50, 0.005), 6)
    lost = pd.DataFrame({"shot_id": 1, "t_end": t, "score": np.where(t > 0.28, 0.9, 0.2)})
    # the twin rises near its own end (quench at 0.45), and is calm at the case's time
    kept = pd.DataFrame({"shot_id": 2, "t_end": t, "score": np.where(t > 0.42, 0.95, 0.1)})
    return pd.concat([lost, kept], ignore_index=True)


def test_arms_and_twin_end():
    pairs = pd.DataFrame({"shot_id": [1], "other": [2], "t_loss": [0.30], "t_start": [0.1]})
    w = ea.endpoint_wins(scores_frame(), pairs, {2: 0.45}, leads_ms=(10,))
    assert len(w) == 1
    row = w.iloc[0]
    assert row.win_p == 1.0 and row.win_e == 0.0  # near its own end the twin outranks the case
    assert abs(row.gap_s - 0.15) < 1e-6


def test_smooth_twin_uses_last_window():
    pairs = pd.DataFrame({"shot_id": [1], "other": [2], "t_loss": [0.30], "t_start": [0.1]})
    w = ea.endpoint_wins(scores_frame(), pairs, {2: np.nan}, leads_ms=(10,))
    assert abs(w.iloc[0].t_e - (0.495 - 0.010)) < 1e-6


def test_summary_and_reading():
    rows = []
    rng = np.random.default_rng(0)
    for seed in range(1, 21):
        for k in range(40):
            rows.append({"model": sc.RECIPE, "seed": seed, "shot_id": k, "other": 100 + k // 2, "lead_ms": 10,
                         "win_p": float(rng.random() < 0.75), "win_e": float(rng.random() < 0.55),
                         "twin_p": 0.1, "twin_e": 0.3, "gap_s": 0.1, "t_p": 0.2, "t_e": 0.3, "ending": "flux_exhausted"})
    frame = pd.DataFrame(rows)
    out = ea.text(frame)
    assert "PRIMARY, A_c at 10 ms" in out and "depends materially" in out
    assert "no material" in ea.reading((0.01, -0.02, 0.04))


def test_programme_phases_and_phase():
    from scripts import endpoint_diagnostic as ed
    t = np.linspace(0, 0.5, 501)
    ref = np.interp(t, [0, 0.1, 0.4, 0.5], [0, 1.0, 1.0, 0.0])
    t0, t1 = ed.programme_phases(t, ref)
    assert abs(t0 - 0.09) < 0.002 and abs(t1 - 0.41) < 0.002
    assert ed.phase(0.05, (t0, t1)) == "ramp-up" and ed.phase(0.2, (t0, t1)) == "flat-top"
    assert ed.phase(0.45, (t0, t1)) == "ramp-down" and ed.phase(np.nan, (t0, t1)) == "unknown"


def test_diagnostic_text_runs():
    from scripts import endpoint_diagnostic as ed
    rows = []
    for seed in range(1, 6):
        for k in range(10):
            for lead in (10, 100):
                rows.append({"model": sc.RECIPE, "seed": seed, "shot_id": k, "other": 100 + k, "lead_ms": lead,
                             "win_p": 1.0, "win_e": 0.0 if lead == 10 else 1.0, "case": 0.5, "twin_p": 0.2,
                             "twin_e": 0.6, "gap_s": 0.1, "t_p": 0.2, "t_e": 0.3, "sel_l": 0.2, "sel_p": 0.2,
                             "sel_e": 0.3, "ending": "flux_exhausted"})
    out = ed.text(pd.DataFrame(rows), {k: (0.1, 0.25) for k in list(range(10)) + [100 + k for k in range(10)]})
    assert "both 50" in out and "ramp-down" in out and "transitions" in out
