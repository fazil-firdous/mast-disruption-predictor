"""Tests for scripts/public_pipeline_audit.py: the restricted unpickler refuses code, both file layouts are read, a
time counter is excluded, change occupancy, the repository's resample_array, and the summary. No network."""
import gzip
import pickle

import numpy as np
import pytest

from scripts import public_pipeline_audit as pp


class Evil:
    def __reduce__(self):
        import os
        return (os.system, ("echo pwned",))


def test_unpickler_refuses_code_and_reads_arrays():
    with pytest.raises(pickle.UnpicklingError):
        pp.safe_load(gzip.compress(pickle.dumps({"x": Evil()})))
    ok = pp.safe_load(gzip.compress(pickle.dumps({"a": {"data": np.ones((3, 2), np.float32), "label": 1, "machine": "east"}})))
    assert pp.shots_of(ok)[0][0] == "east"
    nested = {"train": {0: {"data": np.ones((3, 2)), "label": 0, "machine": "d3d"}}}
    assert len(pp.shots_of(nested)) == 1


def test_counter_occupancy_and_resample():
    t = np.arange(10, dtype=float)[:, None] * 0.005
    sig = np.repeat(np.arange(5, dtype=float), 2)[:, None]  # changes every second row
    a = np.hstack([t, sig])
    assert pp.counter_channels(a) == [0]
    assert abs(pp.change_occupancy(sig) - 4 / 9) < 1e-9
    r = pp.resample_array(np.arange(12, dtype=float)[:, None], 4)
    assert list(r[:, 0]) == [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8]
    assert pp.change_occupancy(np.array([[np.nan], [np.nan]])) == 0.0


def test_summary_separates_a_schedule_and_resample_removes_it():
    rng = np.random.default_rng(0)
    shots = []
    for i in range(200):
        label = int(i % 3 == 0)
        n = 400
        step = 2 if label else 20  # dense changes in disrupted shots, every 20 rows otherwise (a 100 ms cadence)
        sig = np.repeat(rng.normal(size=(n // step + 1, 3)), step, axis=0)[:n]
        counter = np.arange(n, dtype=float)[:, None]
        shots.append(("east", label, np.hstack([counter, sig])))
    out = pp.text(pp.table(shots))
    assert "PRIMARY P1, last 20 rows: AUC 1.000" in out and "counter channels excluded: [(0,)]" in out
    p3 = out.split("P3, after resample_array")[1].split("\n")[0]
    assert "no such signal" in p3
