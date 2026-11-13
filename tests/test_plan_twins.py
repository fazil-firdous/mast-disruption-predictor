"""Tests for scripts/plan_twins.py: which pairs count as plan twins, and the summary from made-up pair wins. No
network."""
import numpy as np
import pandas as pd

from scripts import plan_twins


def compared():
    """Five pairs: 1 the same plan; 2 a different height; 3 no valve requests (kept as a plan twin, not full-plan);
    4 no reference density in either shot (same constant); 5 the density request in one shot only."""
    rows = []
    for shot, d, h, v in ((11, ("both", 0.0), ("both", 0.002), ("both", 0.0)),
                          (12, ("both", 0.0), ("both", 0.30), ("both", 0.0)),
                          (13, ("both", 0.005), ("both", 0.0), ("missing", np.nan)),
                          (14, ("both_constant", np.nan), ("both", 0.0), ("both", 0.0)),
                          (15, ("one", np.nan), ("both", 0.0), ("both", 0.0))):
        rows.append({"shot_id": shot, "other": shot - 10, "density_ref_state": d[0], "density_ref_rms": d[1],
                     "height_ref_state": h[0], "height_ref_rms": h[1], "valves_ref_state": v[0], "valves_ref_rms": v[1]})
    return pd.DataFrame(rows)


def test_which_pairs_are_plan_twins():
    f = plan_twins.flags(compared()).set_index("shot_id")
    assert f["plan"].to_dict() == {11: True, 12: False, 13: True, 14: True, 15: False}
    assert f["full_plan"].to_dict() == {11: True, 12: False, 13: False, 14: True, 15: False}


def test_summary_from_made_up_wins(monkeypatch):
    rows = []
    for seed in range(1, 6):
        for shot in (11, 12, 13, 14, 15):
            for recipe, win in (("baseline", 1.0 if shot != 12 else 0.0), ("machine", 0.5)):
                rows.append({"shot_id": shot, "other": shot - 10, "recipe": recipe, "seed": seed, "lead_ms": 5,
                             "win": win, "win_common": win})
    main = pd.DataFrame(rows)
    out = plan_twins.text(main, plan_twins.flags(compared()), pd.DataFrame())
    assert "Of 5 twin pairs: plan twins 3 (60%); full-plan twins 2 (40%)." in out
    block = out.split("PLAN TWINS (primary of 2026-11-13)")[1].split("FULL-PLAN TWINS")[0]
    assert ": 3 pairs, 3 different twins" in block and "full 1.000" in block and "machine-side 0.500" in block
    assert "full minus machine-side: splits +0.500" in block
    rest = out.split("Twins that are not plan twins")[1].split("All twins")[0]
    assert ": 2 pairs" in rest and "full 0.500" in rest  # pair 12 lost, pair 15 won
    assert "- plan twins: no chronological runs" in out


def test_chronological_lines_use_only_the_chosen_pairs():
    wins = pd.DataFrame([{"shot_id": s, "other": s - 10, "recipe": r, "seed": 0, "win_common": w}
                         for s in (11, 12, 13) for r, w in (("baseline", 1.0 if s != 12 else 0.0), ("machine", 0.5),
                                                            ("programme", 0.5))])
    keep = pd.DataFrame({"shot_id": [11, 13], "other": [1, 3]})
    line = plan_twins.temporal_lines(wins, keep, "plan twins")[0]
    assert "full 1.000" in line and ", 2 pairs" in line and "full minus machine-side +0.500" in line


def test_composition_of_the_sets():
    main = pd.DataFrame([{"shot_id": s, "other": s - 10, "recipe": "baseline", "seed": 1, "lead_ms": 5, "win": 1.0,
                          "win_common": 1.0, "group": "recorded" if s in (11, 12) else "candidate",
                          "heating": "same" if s != 13 else "different"} for s in (11, 12, 13, 14, 15)])
    lines = plan_twins.composition(main, plan_twins.flags(compared()), {11: "M5", 12: "M5", 13: "M9", 14: "M9", 15: "M9"})
    plan = next(l for l in lines if l.startswith("- plan twins"))
    assert "3 pairs; recorded 33%; heating same 67%, different 33%, unknown 0%; M5 33%, M9 67%" in plan
    every = next(l for l in lines if l.startswith("- all tested twins"))
    assert "5 pairs; recorded 40%" in every


def test_field_and_shape_twins():
    c = compared()
    c["tf_current_rms"] = [0.0, 0.0, 0.002, 0.05, 0.0]  # pair 14 has a different field
    for coil in plan_twins.SHAPE:
        c[f"{coil}_rms_early"] = [0.02, 0.0, 0.30 if coil == "coil_p4" else 0.0, 0.0, 0.0]  # pair 13 a different P4
    f = plan_twins.flags(c).set_index("shot_id")
    assert f["field_plan"].to_dict() == {11: True, 12: False, 13: True, 14: False, 15: False}
    assert f["shape_plan"].to_dict() == {11: True, 12: False, 13: False, 14: False, 15: False}
    main = pd.DataFrame([{"shot_id": s, "other": s - 10, "recipe": r, "seed": seed, "lead_ms": 5, "win": w,
                          "win_common": w, "group": "candidate", "heating": "same"}
                         for seed in range(1, 4) for s in (11, 12, 13, 14, 15)
                         for r, w in (("baseline", 1.0), ("machine", 0.5))])
    out = plan_twins.text(main, plan_twins.flags(c), pd.DataFrame(), {s: "M7" for s in range(10, 16)})
    assert "field-plan twins 2 (40%); field-and-shape twins 1 (20%)." in out
    assert "FIELD-AND-SHAPE TWINS (primary of 2026-11-13, later): 1 pairs" in out
    assert "- field-and-shape twins: no chronological runs" in out
