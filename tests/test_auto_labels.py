"""Tests for the automatic termination labels (scripts/auto_labels.py) on hand-made signals. No network."""
import numpy as np
import pandas as pd

from scripts import auto_labels
from src import config, termination

T = np.round(np.arange(0, 0.4, 0.001), 3)
FLAT_REF = np.clip(T / 0.1, 0, 1) * 600e3          # programmed current: ramp up, then flat-top at 600 kA
SWEEPING = 45e3 - 250e3 * T                          # solenoid still sweeping at a steady rate
STOPPED = np.where(T < 0.25, SWEEPING, 45e3 - 250e3 * 0.25)   # solenoid stops sweeping at 0.25 s


def label(ref, sol, t_q, has_quench=True):
    return termination.classify(has_quench, termination.measures(T, ref, sol, t_q))


def test_flat_top_drop_with_solenoid_still_sweeping():
    m = termination.measures(T, FLAT_REF, SWEEPING, 0.3)
    assert m["ref_frac"] == 1.0 and abs(m["sol_ratio"] - 1.0) < 0.05
    assert label(FLAT_REF, SWEEPING, 0.3) == "flat_top_drop"


def test_flux_exhausted_when_solenoid_has_stopped():
    assert label(FLAT_REF, STOPPED, 0.3) == "flux_exhausted"


def test_programmed_rampdown_and_smooth_end_and_unknown():
    ramp_down = np.where(T < 0.25, FLAT_REF, 600e3 * (1 - (T - 0.25) / 0.15))
    assert label(ramp_down, SWEEPING, 0.33) == "programmed_rampdown"
    assert label(FLAT_REF, SWEEPING, np.nan, has_quench=False) == "smooth_end"
    assert label(np.full_like(T, np.nan), SWEEPING, 0.3) == "unknown"


def test_short_flat_top_cannot_be_judged():
    late_ref = np.clip((T - 0.28) / 0.001, 0, 1) * 600e3   # flat-top starts 20 ms before the quench
    assert label(late_ref, SWEEPING, 0.3) == "unknown"


def test_label_shot_reads_cached_extras_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path))
    (tmp_path / "extras").mkdir()
    for shot_id, sol in [(1, SWEEPING), (2, STOPPED)]:
        pd.DataFrame({"time": T, "lm_n1": np.nan, "mirnov_rms": np.nan, "ip_ref": FLAT_REF,
                      "sol_current": sol}).to_parquet(tmp_path / "extras" / f"{shot_id}.parquet")
    shots = pd.DataFrame({"shot_id": [1, 2, 3, 4], "campaign": ["M7"] * 4, "note": [True, False, False, False],
                          "t_quench": [0.3, 0.3, np.nan, 0.3]})
    table = pd.DataFrame([auto_labels.label_shot(row) for _, row in shots.iterrows()])
    assert list(table["termination"]) == ["flat_top_drop", "flux_exhausted", "smooth_end", "unknown"]
    s = auto_labels.summarise(table)
    assert s["n_labelled"] == 3 and s["by_note"]["noted"]["flat_top_drop"] == 1
    text = auto_labels.summary_text(s)
    assert "4 shots" in text and "after the solenoid ran out of flux" in text
