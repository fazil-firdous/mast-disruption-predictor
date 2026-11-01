"""Tests for the TCABR reader (scripts/tcabr.py) on a tiny hand-made file with TCABR's layout. No network."""
import numpy as np
import pytest

from scripts import tcabr
from src import config

h5py = pytest.importorskip("h5py")


def tiny_file(path, disruptive):
    """One discharge: 4 us current with the t = 0 artifact, a flat top at 85 kA and either a fast quench at 80 ms
    or a slow ramp-down; 2 us Mirnov noise; the attributes TCABR uses (placeholder time for the disruptive one)."""
    t4, t2 = np.arange(0, 200000, 4.0), np.arange(0, 200000, 2.0)
    ip = np.clip((t4 - 20000) / 20000, 0, 1) * 85.0
    if disruptive:
        ip[t4 >= 80000] = 0.0
    else:
        ip *= np.clip((150000 - t4) / 40000, 0, 1)
    ip[0] = -152.6
    with h5py.File(path, "w") as f:
        g = f.create_group("20000")
        g.attrs["shot_type"] = np.bytes_("Disruptive" if disruptive else "Normal")
        g.attrs["disruption_time"] = np.bytes_("0.1 ms" if disruptive else "None")
        for name, t in (("np50k_s4us", t4), ("np100k_s2us", t2)):
            g.create_dataset(name, data=t.astype("float32"))
            g[name].make_scale(name)
        signals = [("IPlasma", ip / 0.0745, "np50k_s4us", 0.0745), ("VLoop", np.ones_like(t4), "np50k_s4us", 0.0161),
                   ("BbMirnovN01", np.random.default_rng(0).normal(0, 1, len(t2)), "np100k_s2us", 0.00122)]
        for name, values, axis, scale in signals:
            g.create_dataset(name, data=values.astype("float32"))
            g[name].attrs["sigscale"] = np.array([scale], dtype="float32")
            g[name].dims[0].attach_scale(g[axis])


@pytest.mark.parametrize("disruptive", [True, False])
def test_tcabr_frame_and_label(tmp_path, monkeypatch, disruptive):
    for name in ("IP_ON", "MIN_PEAK_IP"):
        monkeypatch.setattr(config, name, getattr(config, name))  # restored after the test
    tcabr.scale_thresholds()
    tiny_file(tmp_path / "t.nc", disruptive)
    with h5py.File(tmp_path / "t.nc", "r") as f:
        frame = tcabr.shot_frame(f["20000"])
        row = tcabr.shot_row(20000, f["20000"], frame)
    assert frame["time"].iloc[0] > tcabr.SKIP_S and frame["ip"].min() > -1e3  # the t = 0 artifact is gone
    assert frame["ip"].max() == pytest.approx(85e3, rel=1e-3) and np.isnan(frame["halpha"]).all()  # missing: NaN
    assert frame["mirnov_rms"].notna().any()
    assert row["usable"] and row["disrupted"] == disruptive
    if disruptive:  # the placeholder time is replaced by the detector's
        assert row["t_disrupt_both"] == pytest.approx(0.080, abs=0.002)
    else:
        assert np.isnan(row["t_quench"]) and np.isnan(row["t_disrupt_both"])
