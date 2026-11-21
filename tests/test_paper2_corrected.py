import numpy as np
import pandas as pd

from scripts import paper2_corrected as p
from scripts import saddle_correction as sc


def fake_main(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    strength = {"machine": 0.55, "baseline": 0.68, sc.RECIPE: 0.69, "programme": 0.5, "solenoid": 0.6}
    for s in range(21, 51):
        for k in range(60):
            for recipe, q in strength.items():
                w = float(rng.random() < q)
                rows.append({"shot_id": 1000 + k, "other": 5000 + k // 2, "recipe": recipe, "seed": s, "lead_ms": 5,
                             "win": w, "win_common": w, "win_end": w, "heating": "same" if k % 2 else "different",
                             "height": "same", "group": "recorded" if k % 5 == 0 else "candidate"})
    return pd.DataFrame(rows)


def test_registered_and_deltas():
    main = fake_main()
    reg = p.registered(main)
    assert set(reg["model"]) == {"baseline", sc.RECIPE}
    h1a = reg[(reg["model"] == sc.RECIPE) & (reg["test"] == "H1a: C - 0.5")].iloc[0]
    assert h1a["splits"] == 30 and h1a["low"] > 0
    rows, group = p.deltas(main, main.iloc[:0])
    first = rows[0]
    assert first["difference"].startswith("full, corrected (A_c) minus machine-side")
    assert first["over splits"][0] > 0 and first["pooled"][0] > 0
    assert group[sc.RECIPE] is not None
    out = p.text({}, reg, rows, group)
    assert "Registered tests on the confirmation splits" in out


def test_collect_swaps_the_folders_without_recursion(monkeypatch):
    from scripts import v3_explore
    seen = {}

    def fake_collect():
        seen["folders"] = v3_explore.run_folders()
        return "ok"

    monkeypatch.setattr(v3_explore, "collect", fake_collect)
    assert p.collect() == "ok"
    assert sc.RECIPE in seen["folders"] and "baseline" in seen["folders"]
    assert v3_explore.run_folders is p._RUN_FOLDERS  # restored afterwards
