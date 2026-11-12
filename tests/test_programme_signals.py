"""Tests for scripts/programme_signals.py: reading a group, the sample of pairs, and the look in words on made-up
groups. No network."""
import numpy as np
import pandas as pd
import xarray as xr

from scripts import programme_signals

T = np.arange(0.0, 0.4, 0.001)


def group(**signals):
    return xr.Dataset({name: ("time", np.asarray(values, dtype=float)) for name, values in signals.items()},
                      coords={"time": T})


def fake_open(shot_id, name):
    """Shots 1 and 2 share a density programme, shot 3 has another, shot 4 has a constant one, shot 5 has no groups."""
    if shot_id == 5:
        raise OSError("no such store")
    if name == "pulse_schedule":
        density = {1: 2e19 * np.minimum(T / 0.1, 1), 2: 2e19 * np.minimum(T / 0.1, 1), 3: 4e19 * np.minimum(T / 0.1, 1),
                   4: np.zeros(len(T))}[shot_id]
        return group(i_plasma=6e5 * np.minimum(T / 0.1, 1), n_e_line=density)
    if shot_id == 3:
        return group(other=np.ones(len(T)))  # this shot's gas group lacks the injected total
    return group(total_injected=np.cumsum(np.full(len(T), 1e18 * shot_id)))


def test_series_needs_one_value_per_time():
    ds = group(a=np.ones(len(T)))
    ds["two"] = (("time", "channel"), np.ones((len(T), 2)))
    assert programme_signals.series(ds, "a")[1].sum() == len(T)
    assert programme_signals.series(ds, "two") is None and programme_signals.series(ds, "missing") is None
    assert programme_signals.variables(ds) == ["a[time]", "two[time,channel]"]


def test_describe_one_shot():
    row, found = programme_signals.describe(1, fake_open)
    assert row["pulse_schedule_vars"] == "i_plasma[time] n_e_line[time]" and row["density_ref_varies"]
    assert row["density_ref_max"] == 2e19 and found["gas_injected"] is not None
    row, found = programme_signals.describe(4, fake_open)
    assert row["density_ref_samples"] == len(T) and not row["density_ref_varies"]  # there, but constant
    row, found = programme_signals.describe(5, fake_open)
    assert row["pulse_schedule_vars"] == "not opened: OSError" and found == {"density_ref": None, "gas_injected": None}


def test_sample_spreads_over_the_list():
    twins = pd.DataFrame({"shot_id": range(100, 200), "other": range(0, 100)})
    picked = programme_signals.sample(twins, 5)
    assert len(picked) == 5 and picked["shot_id"].iloc[[0, -1]].tolist() == [100, 199]  # both ends of the list
    assert picked["shot_id"].diff().dropna().between(24, 26).all()  # evenly spread
    assert len(programme_signals.sample(twins, 500)) == 100


def test_look_compares_the_two_shots_of_a_pair():
    pairs = pd.DataFrame({"shot_id": [2, 3, 4, 5], "other": [1, 1, 1, 1]})
    campaign = {1: "M5", 2: "M5", 3: "M5", 4: "M9", 5: "M9"}
    shots, compared = programme_signals.look(pairs, {2: 0.3, 3: 0.3}, campaign, fake_open, workers=2)
    assert len(shots) == 5 and list(shots["campaign"]) == ["M5", "M5", "M5", "M9", "M9"]
    by = compared.set_index("shot_id")
    assert by.loc[2, "density_ref_rms"] == 0.0  # the same density programme
    assert 0.4 < by.loc[3, "density_ref_rms"] < 0.5 and by.loc[3, "density_ref_peak_ratio"] == 0.5  # half the other's
    assert np.isnan(by.loc[4, "density_ref_rms"])  # a constant is no programme: not compared
    assert np.isnan(by.loc[5, "density_ref_rms"]) and np.isnan(by.loc[3, "gas_injected_rms"])  # missing in one shot
    assert 0 < by.loc[2, "gas_injected_rms"] < 1
    out = programme_signals.text(shots, compared)
    assert "reference density (pulse_schedule/n_e_line): present in 80% of the shots, not constant in 60%." in out
    assert "M5 100% of 3; M9 0% of 2" in out
    assert "pairs with it, not constant, in both shots: 2 of 4. Relative difference: median" in out and "under 10% in 50%" in out
    assert "- 1 shots: not opened: OSError" in out
