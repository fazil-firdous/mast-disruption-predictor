import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts import saddle_correction as sc
from scripts import saddle_fetch as sf
from src import extras, features

LABELS = [f"M{k:02d}" if k < 10 else f"M0{k}" for k in range(1, 13)]  # M01 .. M09, M010, M011, M012


def magnetics(shot, flip_stored=False, seed=0, dead=("M09",), common=1.0):
    """A synthetic magnetics group: an axisymmetric part seen by every coil, an n=1 part, noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.002, 0.300, 2e-5)
    phi = np.radians(extras.SADDLE_PHI_DEG)
    n0 = common * 1e-3 * np.sin(2 * np.pi * 7 * t) * (1 + t)
    n1 = 2e-4 * t / 0.3
    field = np.array([n0 + n1 * np.cos(p - 0.7) + 1e-5 * rng.standard_normal(len(t)) for p in phi])
    for k, label in enumerate(LABELS):
        if label in dead:
            field[k] = np.nan
        if flip_stored and label in sf.FLIP and sf.FLIP_FROM <= shot <= sf.FLIP_TO:
            field[k] = -field[k]
    mirnov = np.vstack([np.full(50, np.nan), rng.standard_normal(50)])
    return xr.Dataset({sf.FIELD: (("saddle_channel", "time_saddle"), field),
                       sf.MIRNOV: (("cc_channel", "time_mirnov"), mirnov)},
                      coords={"saddle_channel": LABELS, "time_saddle": t, "cc_channel": ["101", "201"],
                              "time_mirnov": np.arange(50) * 2e-6},
                      attrs={"ingested_at": "2026-09-25T10:00:00Z", "commit_url": "https://example/commit/abc0948"})


def summary_group():
    t = np.round(np.arange(0.0, 0.3, 0.001), 6)
    return xr.Dataset({"ip": ("time", np.full(len(t), 4e5))}, coords={"time": t}, attrs={"ingested_at": "2026-09-24"})


def test_lm_raw_is_exactly_the_cached_feature(monkeypatch):
    mag = magnetics(25000, flip_stored=True)
    t_grid = summary_group()["time"].values.astype(float)
    monkeypatch.setattr(extras, "open_group", lambda shot, group: mag)
    expected = extras.saddle_n1(25000, t_grid)
    deltas, known = sf.channel_bins(mag["time_saddle"].values, mag[sf.FIELD].values, t_grid, np.ones(12))
    got = sf.lm_from(deltas, known, t_grid)
    assert np.array_equal(np.isfinite(got), np.isfinite(expected))
    assert np.array_equal(got[np.isfinite(got)], expected[np.isfinite(expected)])


def test_signs_only_for_listed_channels_inside_the_range():
    assert (sf.signs_for(24781, LABELS) == 1).all()
    for shot in (24782, 30473):
        s = sf.signs_for(shot, LABELS)
        assert [LABELS[k] for k in np.where(s < 0)[0]] == ["M02", "M08", "M012"]
    assert (sf.signs_for(30474, LABELS) == 1).all()


def test_correction_restores_the_true_amplitude():
    true = magnetics(25000, flip_stored=False)
    stored = magnetics(25000, flip_stored=True)
    t_grid = summary_group()["time"].values.astype(float)
    signs = sf.signs_for(25000, LABELS)
    a = sf.lm_from(*sf.channel_bins(true["time_saddle"].values, true[sf.FIELD].values, t_grid, np.ones(12)), t_grid)
    b = sf.lm_from(*sf.channel_bins(stored["time_saddle"].values, stored[sf.FIELD].values, t_grid, signs), t_grid)
    c = sf.lm_from(*sf.channel_bins(stored["time_saddle"].values, stored[sf.FIELD].values, t_grid, np.ones(12)), t_grid)
    ok = np.isfinite(a)
    assert np.allclose(a[ok], b[ok], rtol=1e-12, atol=0)
    assert not np.allclose(a[ok], c[ok])


def test_shot_result_tables(monkeypatch):
    groups = {"summary": summary_group(), "magnetics": magnetics(25000, flip_stored=True)}
    out, status = sf.shot_result(25000, lambda shot, group: groups[group], sample=True)
    assert status == "ok"
    row = out["shots"].iloc[0]
    assert row["mirnov_label"] == "201" and row["n_finite"] == 11 and row["flip_applied"]
    assert row["mag_commit_url"].endswith("0948")
    ch = out["channels"].set_index("label")
    assert not ch.loc["M09", "finite"]
    assert ch.loc["M02", "r_raw"] < -0.5 and ch.loc["M02", "r_corr"] > 0.5 and ch.loc["M03", "r_raw"] > 0.5
    assert len(out["summary_sample"]) == len(groups["summary"]["time"])


def channels_for(shots, flip_stored):
    rows = []
    for shot in shots:
        groups = {"summary": summary_group(), "magnetics": magnetics(shot, flip_stored, seed=shot % 97)}
        out, _ = sf.shot_result(shot, lambda s, g: groups[g])
        rows.append(out["channels"])
    return pd.concat(rows, ignore_index=True)


def sample_frame(before, after):
    return pd.DataFrame({"shot_id": before + after, "group": ["B"] * len(before) + ["F"] * len(after)})


def test_verdict_v1_when_the_reversal_is_stored_and_v2_when_not():
    before, after = [20000, 21000, 22000], [25000, 27000, 29000]
    sample = sample_frame(before, after)
    stored = channels_for(before + after, flip_stored=True)
    assert sc.verdict(sc.channel_medians(stored, sample)) == "V1"
    assert sc.verdict(sc.channel_medians(stored, sample, "r_corr")) == "V2"
    clean = channels_for(before + after, flip_stored=False)
    assert sc.verdict(sc.channel_medians(clean, sample)) == "V2"


def test_verdict_v3_when_unclear():
    sample = sample_frame([20000], [25000])
    channels = pd.DataFrame({"shot_id": [20000] * 3 + [25000] * 3, "label": ["M02", "M08", "M012"] * 2,
                             "finite": True, "r_raw": [0.9, 0.9, 0.05, -0.9, -0.9, -0.9]})
    assert sc.verdict(sc.channel_medians(channels, sample)) == "V3"


def test_frozen_sample_rule():
    ids = list(range(20000, 24782, 7)) + list(range(24782, 30474, 5)) + [30500]
    s = sc.draw_sample(ids)
    b, f = s[s["group"] == "B"]["shot_id"], s[s["group"] == "F"]["shot_id"]
    assert b.max() < 24782 <= f.min() and f.max() <= 30473
    assert 20 <= len(b) <= 25 and 20 <= len(f) <= 25
    assert 20000 in set(b) and max(i for i in ids if i < 24782) in set(b) and 24782 in set(f)
    assert set(s[s["group"] == "diagnostic"]["shot_id"]) <= {27000, 30421}
    assert s.equals(sc.draw_sample(list(reversed(ids))))


def test_same_series():
    a = np.array([np.nan, 1.0, 2.0])
    assert sc.same_series(a, a * (1 + 1e-12))
    assert not sc.same_series(a, a * (1 + 1e-6))
    assert not sc.same_series(a, np.array([0.0, 1.0, 2.0]))


def test_swap_replaces_only_lm_columns_of_corrected_shots():
    lm = [c for c in features.feature_names(extended=True) if features.uses_signal(c, "lm")][:2] or ["lm_mean", "lm_rel"]
    windows = pd.DataFrame({"shot_id": [1, 1, 2], "t_end": [0.1, 0.2, 0.1], lm[0]: np.float32([1, 2, 3]),
                            lm[1]: np.float32([4, 5, 6]), "ip_abs_mean": np.float32([7, 8, 9])})
    corrected = pd.DataFrame({"shot_id": [1, 1], "t_end": [0.1, 0.2], lm[0]: [10.0, np.nan], lm[1]: [40.0, 50.0]})
    out = sc.swap(windows, corrected)
    assert out[lm[0]].tolist()[0] == 10 and np.isnan(out[lm[0]].tolist()[1]) and out[lm[0]].tolist()[2] == 3
    assert out["ip_abs_mean"].equals(windows["ip_abs_mean"]) and out[lm[0]].dtype == np.float32
    with pytest.raises(RuntimeError):
        sc.swap(windows, corrected.iloc[:1])
