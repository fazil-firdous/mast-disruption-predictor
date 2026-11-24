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


def read_for(shots, flip_stored):
    channels, bins = [], []
    for shot in shots:
        groups = {"summary": summary_group(), "magnetics": magnetics(shot, flip_stored, seed=shot % 97)}
        out, _ = sf.shot_result(shot, lambda s, g: groups[g], sample=True)
        channels.append(out["channels"])
        bins.append(out["channel_bins"])
    return pd.concat(channels, ignore_index=True), pd.concat(bins, ignore_index=True)


BEFORE, AFTER = list(range(20000, 20015)), list(range(25000, 25015))


def sample_frame(before, after):
    return pd.DataFrame({"shot_id": before + after, "group": ["B"] * len(before) + ["F"] * len(after)})


@pytest.fixture(scope="module")
def stored():
    return read_for(BEFORE + AFTER, flip_stored=True)


@pytest.fixture(scope="module")
def clean():
    return read_for(BEFORE + AFTER, flip_stored=False)


def test_verdict_v1_when_the_reversal_is_stored_and_v2_when_not(stored, clean):
    sample = sample_frame(BEFORE, AFTER)
    med = sc.channel_medians(stored[0], sample)
    assert sc.verdict(med) == "V1"
    assert med.loc["M02", "F_neg"] == 15 and med.loc["M02", "B_pos"] == 15 and med.loc["M02", "B_shots"] == 15
    assert sc.verdict(sc.channel_medians(stored[0], sample, "r_corr")) == "V2"
    assert sc.verdict(sc.channel_medians(clean[0], sample)) == "V2"


def test_too_few_shots_is_v3(stored):
    sample = sample_frame(BEFORE[:14], AFTER)
    assert sc.verdict(sc.channel_medians(stored[0], sample)) == "V3"


def test_reference_diagnostic(stored, clean):
    sample = sample_frame(BEFORE, AFTER)
    ref = sc.reference_r(stored[1])
    assert set(ref["label"]) == set(LABELS)
    assert sc.a2_consistent(sc.channel_medians(ref, sample, "r_ref"))
    assert not sc.a2_consistent(sc.channel_medians(sc.reference_r(clean[1]), sample, "r_ref"))


def test_verdict_v3_when_unclear():
    sample = sample_frame([20000 + i for i in range(15)], [25000 + i for i in range(15)])
    rows = []
    for shot, group in sample.itertuples(index=False):
        for label, r in zip(["M02", "M08", "M012"], [0.9, 0.9, 0.05] if group == "B" else [-0.9, -0.9, -0.9]):
            rows.append({"shot_id": shot, "label": label, "finite": True, "r_raw": r})
    assert sc.verdict(sc.channel_medians(pd.DataFrame(rows), sample)) == "V3"


def test_conditioning():
    rank, cond = sc.design_condition([True] * 8 + [False] + [True] * 3)
    assert rank == 3 and cond < 2
    assert sc.design_condition([True, True] + [False] * 10)[0] < 3
    channels = pd.DataFrame({"shot_id": np.repeat([25000, 25001], 12), "index": np.tile(np.arange(12), 2),
                             "finite": [k != 8 for k in range(12)] * 2})
    frame, ok = sc.conditioning(channels, {25000, 25001})
    assert ok and frame["shots"].sum() == 2
    adjacent = pd.DataFrame({"shot_id": 25000, "index": np.arange(12), "finite": [k < 6 for k in range(12)]})
    assert sc.design_condition(adjacent["finite"])[1] > 1


def test_compare_rebuild():
    mine = pd.DataFrame({"t_end": [0.1, 0.2], "lm_mean": np.float32([1.0, np.nan])})
    assert sc.compare_rebuild(mine, mine.copy(), ["lm_mean"])[0]
    near = mine.assign(lm_mean=np.float32([1.0000001, np.nan]))
    assert sc.compare_rebuild(mine, near, ["lm_mean"])[0]
    assert not sc.compare_rebuild(mine, mine.assign(lm_mean=np.float32([1.01, np.nan])), ["lm_mean"])[0]
    assert not sc.compare_rebuild(mine, mine.assign(lm_mean=np.float32([1.0, 2.0])), ["lm_mean"])[0]
    assert not sc.compare_rebuild(mine, mine.iloc[:1], ["lm_mean"])[0]


def test_training_refuses_without_gates(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "folder", lambda: str(tmp_path))
    sc.write_gates(G1=False, G2=False, G1a=True, G2a=True, G3=False, G4=True, G5=True)
    assert sc.gates_pass() == (False, ["G3"])  # the original G1/G2 failures stay recorded but no longer gate
    sc.write_gates(G3=True, G5=False)
    assert sc.gates_pass() == (False, ["G5"])
    with pytest.raises(RuntimeError):
        sc.load_tables()


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


# ------------------------------------------------------------------ amendment 2: structural absence, G5

LM8 = ["has_lm", "lm_mean", "lm_std", "lm_last", "lm_slope", "lm_delta", "lm_rel", "lm_slope_long"]


def absent_windows(n=3, t0=0.1):
    frame = pd.DataFrame({"t_end": t0 + 0.01 * np.arange(n)})
    for c in LM8:
        frame[c] = np.float32(0.0) if c == "has_lm" else np.float32(np.nan)
    return frame


def test_absence_state_only_for_verified_absence():
    assert sc.absence_state({"status": "no_saddle", "mag_ingested_at": "2026-09-25"}) == "no saddle field"
    assert sc.absence_state({"status": "no_saddle", "mag_ingested_at": np.nan}) == "no magnetics group"
    assert sc.absence_state({"status": "no_saddle"}) == "no magnetics group"
    for bad in ({"status": "unreachable"}, {"status": "odd"}, {"status": "absent"}, {"status": "ok"}, None):
        assert sc.absence_state(bad) is None  # a read error or a shot missing from the read never qualifies


def test_absence_cache_must_be_all_nan_on_its_grid():
    assert sc.absence_cache_ok(pd.DataFrame({"lm_n1": [np.nan] * 5}), 5)[0]
    assert not sc.absence_cache_ok(pd.DataFrame({"lm_n1": [np.nan, 1e-4, np.nan, np.nan, np.nan]}), 5)[0]
    assert not sc.absence_cache_ok(pd.DataFrame({"lm_n1": [np.nan] * 4}), 5)[0]
    assert not sc.absence_cache_ok(None, 5)[0]


def test_absence_windows_broken_cases_fail():
    mine = absent_windows()
    assert sc.absence_windows_ok(mine, mine.copy(), LM8)[0]
    shifted = mine.assign(t_end=mine["t_end"] + 0.001)
    assert not sc.absence_windows_ok(mine, shifted, LM8)[0]
    assert not sc.absence_windows_ok(mine, mine.iloc[:2], LM8)[0]
    finite = mine.copy()
    finite.loc[1, "lm_mean"] = np.float32(2e-4)
    assert not sc.absence_windows_ok(finite, finite.copy(), LM8)[0]  # the table itself carries an lm value
    assert not sc.absence_windows_ok(mine, finite, LM8)[0]  # wrong NaN pattern
    flag = mine.assign(has_lm=np.float32(1.0))
    assert not sc.absence_windows_ok(flag, flag.copy(), LM8)[0]


def test_absence_end_to_end_rebuild():
    """A shot whose lm is all NaN rebuilds to has_lm 0 and NaN lm features, which the branch accepts."""
    t = np.round(np.arange(0.0, 0.4, 0.001), 6)
    shot = pd.DataFrame({"time": t, "ip": np.where(t > 0.02, 5e5, 0.0), "power_radiated": 1e5,
                         "neutron_rates_total": 1e12, "power_nbi": 0.0})
    for column in extras.COLUMNS[1:]:
        shot[column] = np.nan
    win = features.shot_windows(shot)
    columns = [c for c in win.columns if features.uses_signal(c, "lm")]
    assert sorted(columns) == sorted(LM8)
    assert sc.absence_windows_ok(win[["t_end"] + columns], win[["t_end"] + columns].copy(), columns)[0]


def test_non_lm_equal():
    w = pd.DataFrame({"shot_id": [1, 2], "t_end": [0.1, 0.2], "lm_mean": np.float32([1, 2]),
                      "ip_abs_mean": np.float32([3, np.nan]), "split": ["train", "test"]})
    assert sc.non_lm_equal(w, w.assign(lm_mean=np.float32([5, 6])))
    assert not sc.non_lm_equal(w, w.assign(ip_abs_mean=np.float32([3, 4])))
    assert not sc.non_lm_equal(w, w.assign(split=["test", "test"]))
    assert not sc.non_lm_equal(w, w.iloc[:1])


def test_originals_are_kept_once(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "folder", lambda: str(tmp_path))
    (tmp_path / "check.txt").write_text("first")
    sc.preserve_originals()
    (tmp_path / "check.txt").write_text("second")
    sc.preserve_originals()
    assert (tmp_path / "check_original.txt").read_text() == "first"


def test_ledger_change_counts():
    from scripts import saddle_ledger as sl
    a = np.array([1.0, 2.0, np.nan, 3.0, np.nan])
    b = np.array([1.0, 2.5, 4.0, np.nan, np.nan])
    n = sl.change_counts(a, b)
    assert n == {"finite, value differs": 1, "finite to missing": 1, "missing to finite": 1, "unchanged finite": 1,
                 "unchanged missing": 1}
    assert sum(n.values()) == len(a)


def test_no_double_flip_on_a_release_ingested_after_the_source_fix():
    assert sf.release_state("2026-09-25T10:00:00Z") == "reversed"
    assert sf.release_state("2026-11-30T10:00:00Z") == "unverified" and sf.release_state("") == "unverified"
    mag = magnetics(25000, flip_stored=False)
    mag.attrs["ingested_at"] = "2026-11-30T10:00:00Z"
    groups = {"summary": summary_group(), "magnetics": mag}
    out, status = sf.shot_result(25000, lambda shot, group: groups[group])
    assert status == "release_unverified" and "lm" not in out
    old = {"summary": summary_group(), "magnetics": magnetics(20000, flip_stored=False)}
    old["magnetics"].attrs["ingested_at"] = "2026-11-30T10:00:00Z"
    assert sf.shot_result(20000, lambda shot, group: old[group])[1] == "ok"  # outside the range: nothing to flip
