"""Tests for scripts/plot_twin_pair.py on two made-up shots. No network."""
import os

import numpy as np
import pandas as pd
import pytest

from scripts import plot_twin_pair


def shot(drop_at=None):
    t = np.arange(0, 0.4, 0.001)
    ip = np.where(t < 0.1, 8e5 * t / 0.1, 8e5)
    if drop_at is not None:
        ip = np.where(t > drop_at, 0.0, ip)
    return pd.DataFrame({"time": t, "ip": -ip, "ip_ref": np.where(t < 0.1, 8e5 * t / 0.1, 8e5), "power_nbi": 2e6,
                         "ne_line_avg": 3e19, "efit_zmag": 0.01})


def test_picture_of_a_pair(tmp_path):
    twins = pd.DataFrame({"shot_id": [20, 30], "other": [18, 27], "group": ["candidate", "recorded"], "gap": [2, 3],
                          "rms": [0.0, 0.0], "t_loss": [0.3, 0.25], "heating": ["same", "same"]})
    pair = plot_twin_pair.find_twin(twins, 20)
    assert pair["other"] == 18
    with pytest.raises(SystemExit):
        plot_twin_pair.find_twin(twins, 99)
    path = plot_twin_pair.plot(shot(drop_at=0.3), shot(), pair, str(tmp_path / "pairs" / "20.png"))
    assert os.path.getsize(path) > 5000 and os.path.exists(str(tmp_path / "pairs" / "20.pdf"))
    four = plot_twin_pair.PANELS[:2] + [("ne_line_avg", "density (10^19 m^-3)", 1e-19), ("efit_zmag", "height (cm)", 100)]
    assert os.path.getsize(plot_twin_pair.plot(shot(0.3), shot(), pair, str(tmp_path / "four.png"), four)) > 5000
    assert plot_twin_pair.listing(twins).splitlines()[1].split()[0] == "30"  # recorded disruptions first
    assert "30" in plot_twin_pair.listing(twins, after=25) and " 20 " not in plot_twin_pair.listing(twins, after=25)
    empty = shot().assign(ne_line_avg=np.nan, power_nbi=3.0)  # no density at all, a beam signal that is only noise
    assert os.path.getsize(plot_twin_pair.plot(empty, empty, pair, str(tmp_path / "empty.png"))) > 5000
    assert len(plot_twin_pair.PANELS) == 4
