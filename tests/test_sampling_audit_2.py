"""Tests for scripts/sampling_audit_2.py on made-up records: pure-timing flags, the early window, the per-shot table
and the summary. No network, no real data."""
import numpy as np
import pandas as pd

from scripts import sampling_audit_2 as s2


def test_flag_share_reads_no_values():
    dense = np.round(np.arange(0.0, 1.0, 0.002), 4)
    sparse = np.round(np.arange(0.0, 1.0, 0.025), 4)
    assert s2.flag_share(dense, 0.9) == 1.0
    assert 0.15 <= s2.flag_share(sparse, 0.9) <= 0.25  # one slice in about every fifth 5 ms step
    assert np.isnan(s2.flag_share(np.array([0.5]), 0.5))


def test_early_spacing():
    t = np.round(np.arange(0.2, 1.5, 0.010), 4)
    assert abs(s2.early_spacing(t) - 0.010) < 1e-9
    assert np.isnan(s2.early_spacing(np.array([0.0, 0.05])))


def records(dense_early):
    rows = []
    for shot in range(80):
        disrupted = shot % 4 == 0
        if disrupted:
            step_early, step_late = (0.010 if dense_early else 0.100), 0.002
        else:
            step_early, step_late = 0.100, 0.100
        t = np.concatenate([np.arange(0.0, 1.0, step_early), np.arange(1.0, 1.5, step_late)])
        tud = (1.5 - t) if disrupted else np.full(len(t), np.nan)
        rows.append(pd.DataFrame({"shot": 1000 + shot, "time": np.round(t, 4), "time_until_disrupt": tud}))
    return pd.concat(rows)


def test_summary(tmp_path):
    table = s2.shot_table(records(dense_early=True))
    out = s2.text({"EAST": table}, n_boot=100)
    assert "C1 observation flags (no value read), AUC 1.000" in out
    assert "C2 fixed early window" in out and "AUC 1.000" in out.split("C2")[1].split("\n")[0]
    assert "C4 last slice minus disruption time" in out and "C5 shot-number bins" in out
    late_only = s2.text({"EAST": s2.shot_table(records(dense_early=False))}, n_boot=100)
    c2 = late_only.split("C2")[1].split("\n")[0]
    assert "AUC 0.500" in c2 and "no such signal" in c2
