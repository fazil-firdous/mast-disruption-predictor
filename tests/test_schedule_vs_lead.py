import scipy.io

from scripts import schedule_vs_lead as sv
from tests.test_landmark_zhu import zhu_like


def test_run_end_to_end(tmp_path):
    f = zhu_like(n=80)
    d = tmp_path / "20ja072_fig4_data"
    d.mkdir()
    scipy.io.savemat(d / "C_Mod_for_nn.mat", {c: f[c].to_numpy() for c in f.columns})
    out = sv.run(str(tmp_path))
    assert "DISRUPTED records" in out and "cohort flow" in out
    assert "gap median    5.00 ms" in out  # synthetic records are sampled every 5 ms
