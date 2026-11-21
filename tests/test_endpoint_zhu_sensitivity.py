import numpy as np
import scipy.io

from scripts import endpoint_zhu as ez, endpoint_zhu_sensitivity as zs
from tests.test_endpoint_zhu import frame


def test_eligible_many_matches_single():
    t = np.round(np.arange(0.0, 1.0, 0.005), 6)
    marks = np.array([0.0, 0.3, 0.5, 0.95, 0.96, 1.2])
    assert list(zs.eligible_many(t, marks)) == [ez.eligible(t, m) for m in marks]


def test_mixed_landmarks_are_eligible_and_reproducible():
    shots = ez.shots_of(frame(), [f"sig{j}" for j in range(12)])
    train = list(shots)
    a, kept = zs.mixed_landmarks(shots, train)
    b, _ = zs.mixed_landmarks(shots, train)
    assert a == b and len(a) > 0 and kept >= 0
    for k, mark in a.items():
        assert not shots[k][2] and ez.eligible(shots[k][0], mark)


def test_empty_share_and_history():
    mask = np.array([[True, False], [False, False]])
    assert 0 < zs.empty_share(mask, n_boot=50) < 1


def test_run_end_to_end(tmp_path):
    f = frame()
    d = tmp_path / "20ja072_fig4_data"
    d.mkdir()
    scipy.io.savemat(d / "C_Mod_for_nn.mat", {c: f[c].to_numpy() for c in f.columns})
    out = zs.run(str(tmp_path))
    assert "S1 common 300 ms history" in out and "S2, mixed-landmark model" in out and "repeated times" in out


def test_partition_check(tmp_path):
    p = tmp_path / "part.mat"
    scipy.io.savemat(p, {"trainingShotlist": np.array([1, 2, 2, 3]), "validationShotlist": np.array([3, 4]),
                         "testShotlist": np.array([5, 9])})
    out = zs.partition_check(str(p), {1, 2, 3, 4, 5, 6})
    assert "train 4 IDs (1 repeated)" in out and "more than one list 1" in out
    assert "not in the fig4 file 1" in out and "in no list 1" in out
