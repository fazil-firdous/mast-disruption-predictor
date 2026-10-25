"""Tests for the statistics of the automatic termination labels and the independent-evidence check. No network."""
import numpy as np
import pandas as pd
import pytest

from scripts import label_statistics, termination_evidence
from tests.synthetic import make_shot


def test_wilson_interval_and_auc():
    low, high = label_statistics.wilson(623, 847)
    assert low == pytest.approx(0.705, abs=1e-3) and high == pytest.approx(0.764, abs=1e-3)
    assert label_statistics.auc(np.array([2.0, 3.0]), np.array([1.0])) == 1.0
    assert label_statistics.auc(np.array([1.0]), np.array([1.0])) == 0.5  # ties count half


def fake_labels():
    rows = []
    for i in range(40):  # noted: mostly lost during flat-top with the solenoid still sweeping
        rows.append({"shot_id": i, "note": True, "t_quench": 0.3, "sol_ratio": 0.9 - 0.01 * i,
                     "termination": "flat_top_drop" if i < 30 else "flux_exhausted"})
    for i in range(40, 140):  # not noted: half flux exhausted, a quarter flat-top drops, a quarter smooth
        kind = ["flux_exhausted", "flux_exhausted", "flat_top_drop", "smooth_end"][i % 4]
        ratio = {"flux_exhausted": 0.05, "flat_top_drop": 0.7, "smooth_end": np.nan}[kind]
        rows.append({"shot_id": i, "note": False, "t_quench": np.nan if kind == "smooth_end" else 0.3,
                     "sol_ratio": ratio, "termination": kind})
    return pd.DataFrame(rows)


def test_block_counts_shares_and_ratio_test():
    b = label_statistics.block(fake_labels())
    assert (b["noted"]["k"], b["noted"]["n"]) == (30, 40) and (b["not_noted"]["k"], b["not_noted"]["n"]) == (25, 100)
    assert b["fired_late"]["n"] == 115 and b["fired_late"]["k"] == 60  # 10 noted + 50 un-noted flux exhausted
    r = b["ratio"]
    assert r["n_noted"] == 40 and r["n_other"] == 75 and r["auc"] > 0.7 and r["p_value"] < 1e-3
    assert r["auc_ci"][0] <= r["auc"] <= r["auc_ci"][1] and r["cliffs_delta"] == pytest.approx(2 * r["auc"] - 1)
    text = label_statistics.block_text("Test", b)
    assert "30/40 = 75.0%" in text and "25/100 = 25.0%" in text and "nan" not in text


def test_sampling_scales_to_the_shot_table():
    table = pd.DataFrame({"shot_postshot_comment": ["DISRUPTION AT 200MS"] * 10 + ["good shot"] * 400})
    s = label_statistics.sampling(fake_labels(), table)
    assert s["not_noted_in_table"] == 400 and s["weight"] == 4.0 and s["estimate"] == pytest.approx(100)
    assert label_statistics.sampling(fake_labels(), None) is None


def with_fake_extras(shot):
    n = len(shot)
    return shot.assign(ip_ref=np.where(shot["time"] > 0.1, 7e5, 0.0), sol_current=np.linspace(0, -4e4, n),
                       mirnov_rms=np.ones(n), lm_n1=np.ones(n))


def test_shot_evidence_tells_a_quench_from_a_rampdown():
    quench = termination_evidence.shot_evidence(with_fake_extras(make_shot("quench", t_quench=0.22)), 0.22)
    ramp = termination_evidence.shot_evidence(with_fake_extras(make_shot("rampdown", t_rampdown=0.25)), 0.29)
    assert quench["fall_ms"] < termination_evidence.FAST_FALL_MS < ramp["fall_ms"]
    assert quench["neutrons_rel"] == pytest.approx(1.0) and quench["mirnov_rel"] == pytest.approx(1.0)


def test_compare_gives_medians_and_aucs_per_measure():
    rng = np.random.default_rng(0)
    rows = [{"group": g, **{m: rng.normal(mu, 0.1) for m in termination_evidence.MEASURES}}
            for g, mu in [("recorded disruption", 1.0), ("candidate", 1.0), ("ordinary end", 0.0)] for _ in range(20)]
    table = termination_evidence.compare(pd.DataFrame(rows)).set_index("measure")
    assert table.loc["fall_ms", "AUC recorded disruption > ordinary end"] == 1.0
    assert 0.2 < table.loc["fall_ms", "AUC recorded disruption > candidate"] < 0.8
    assert "recorded disruption" in termination_evidence.summary_text(pd.DataFrame(rows), table.reset_index())
