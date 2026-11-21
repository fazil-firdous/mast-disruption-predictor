"""Endpoint module 2, amendment 1 (docs/preregistration_endpoint_zhu.md; written AFTER the registered result was seen):
verification items, S1 (common recorded history, no refit) and S2 (mixed-landmark training, one refit per machine).
Post-result sensitivities; the registered output stays the result.

    python scripts/endpoint_zhu_sensitivity.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/endpoint_zhu_sensitivity.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import endpoint_zhu as ez, sampling_audit as sa, timing_model as tm
from src import notify

MIX_SEED = 20261121
HISTORY = ez.WINDOW
LISTS = (("trainingShotlist", "train"), ("validationShotlist", "val"), ("testShotlist", "test"))


def partition_check(path, file_shots):
    import scipy.io
    m = scipy.io.loadmat(path, squeeze_me=True)
    ids = {name: np.atleast_1d(m[key]).astype(int) for key, name in LISTS}
    seen = {}
    for name, arr in ids.items():
        for s in set(arr.tolist()):
            seen.setdefault(s, []).append(name)
    listed = set(seen)
    parts = [f"{name} {len(arr):,} IDs ({len(arr) - len(set(arr.tolist()))} repeated)" for name, arr in ids.items()]
    return (f"  partition: {'; '.join(parts)}; IDs in more than one list {sum(len(v) > 1 for v in seen.values())}; "
            f"listed IDs not in the fig4 file {len(listed - file_shots):,}; file shots in no list "
            f"{len(file_shots - listed):,}")


def timestamp_check(frame):
    bad_order = bad_repeat = 0
    for _, g in frame.groupby("shot", sort=False):
        t = g["time"].to_numpy(dtype=float)
        bad_order += int(np.any(np.diff(t) < 0))
        bad_repeat += int(len(np.unique(t)) < len(t))
    return f"  timestamps: shots with unsorted times {bad_order:,}, with repeated times {bad_repeat:,}"


def empty_share(mask, n_boot=ez.N_BOOT, seed=ez.SEED):
    """Share of sampled case draws with no sampled eligible control, same resamples as ez.delta_range."""
    rng = np.random.default_rng(seed)
    nc, nk = mask.shape
    empty = 0
    for _ in range(n_boot):
        ci, ki = rng.integers(0, nc, nc), rng.integers(0, nk, nk)
        empty += int((~mask[np.ix_(ci, ki)].any(axis=1)).sum())
    return empty / (n_boot * nc)


def history_mask(b, shots):
    """Registered eligible pairs where the case, E and M each have >= 300 ms of record before their landmark."""
    mask = np.isfinite(b["s_m"])
    case_ok = np.array([shots[c][0][-1] - ez.CUT - shots[c][0][0] >= HISTORY - ez.TOL for c in b["cases"]])
    m_ok = b["e_case"] >= HISTORY - ez.TOL
    e_ok = np.array([shots[k][0][-1] - ez.CUT - shots[k][0][0] >= HISTORY - ez.TOL for k in b["controls"]])
    return mask & (case_ok & m_ok)[:, None] & e_ok[None, :]


def eligible_many(t, landmarks):
    """ez.eligible for many landmarks of one record."""
    hi = np.searchsorted(t, landmarks + ez.TOL, side="right")
    lo = np.searchsorted(t, landmarks - ez.WINDOW + ez.TOL, side="right")
    return (hi > lo) & (t[-1] >= landmarks + ez.CUT - ez.TOL)


def mixed_landmarks(shots, train, seed=MIX_SEED):
    """{training control: M-like landmark} for the controls moved; (landmarks, moved, stayed for lack of an anchor)."""
    rng = np.random.default_rng(seed)
    anchors = np.array([ez.elapsed(shots[s][0]) for s in train if shots[s][2]])
    landmarks, kept = {}, 0
    for k in sorted(s for s in train if not shots[s][2]):
        if rng.random() >= 0.5:
            continue
        t = shots[k][0]
        ok = anchors[eligible_many(t, t[0] + anchors)]
        if not len(ok):
            kept += 1
            continue
        landmarks[k] = t[0] + ok[rng.integers(0, len(ok))]
    return landmarks, kept


def delta_line(label, b, mask):
    w_e = ez.win_matrix(b["s_case"], b["s_end"])
    w_m = ez.win_matrix(b["s_case"], b["s_m"])
    d = ez.delta_range(w_e, w_m, mask)
    return (f"  {label}: C_E {ez.concordance(w_e, mask):.3f}, C_M {ez.concordance(w_m, mask):.3f}; Delta "
            f"{d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f}); {ez.reading(d)}")


def cohort(mask):
    return (f"cases {int(mask.any(axis=1).sum()):,}, unique controls {int(mask.any(axis=0).sum()):,}, combinations "
            f"{int(mask.sum()):,}")


def machine(name, frame, shots, split, extra):
    train = [s for s in shots if split.get(s) == "train"]
    lines = [f"== {name}", extra, timestamp_check(frame)]
    b = ez.build(shots, split)
    mask = np.isfinite(b["s_m"])
    lines.append(f"  registered cohort: {cohort(mask)}; bootstrap case draws with no eligible control "
                 f"{empty_share(mask):.2%}")
    lines.append(delta_line("registered model (repeat of the registered run)", b, mask))
    hm_ = history_mask(b, shots)
    lines.append(f"  S1 common 300 ms history: {cohort(hm_)}")
    lines.append(delta_line("S1, registered model", b, hm_) if hm_.any() else "  S1: not evaluable (empty)")
    landmarks, kept = mixed_landmarks(shots, train)
    moved = np.array([(shots[k][0][-1] - ez.CUT - landmarks[k]) * 1e3 for k in landmarks])
    n_ctrl = sum(not shots[s][2] for s in train)
    lines.append(f"  S2 training controls {n_ctrl:,}: moved to an M-like landmark {len(landmarks):,}, chosen but "
                 f"without an eligible anchor {kept:,}; distance moved (ms) median "
                 f"{np.median(moved) if len(moved) else np.nan:.0f} (quartiles "
                 f"{np.percentile(moved, 25) if len(moved) else np.nan:.0f} to "
                 f"{np.percentile(moved, 75) if len(moved) else np.nan:.0f})")
    b2 = ez.build(shots, split, model=ez.fit(shots, train, landmarks))
    lines.append(delta_line("S2, mixed-landmark model, registered pairs", b2, mask))
    labels = np.r_[np.ones(len(b2["cases"])), np.zeros(len(b2["controls"]))].astype(bool)
    lines.append(f"  S2 ordinary AUC, one score per shot, controls at E: "
                 f"{sa.fmt(sa.auc_range(np.r_[b2['s_case'], b2['s_end']], labels))}")
    return lines


def run(base):
    lines = ["Endpoint module 2, amendment 1 (written after the registered result was seen): verification, S1 common "
             "history, S2 mixed-landmark training. Post-result sensitivities; the registered output stays the result.",
             ""]
    fig4 = os.path.join(base, "20ja072_fig4_data")
    for name, file, official in (("EAST", "EAST_for_nn.mat", True), ("DIII-D", "d3d_for_nn.mat", False),
                                 ("C-Mod", "C_Mod_for_nn.mat", False)):
        path = os.path.join(fig4, file)
        if not os.path.exists(path):
            continue
        print(f"{name} ...", flush=True)
        frame = sa.load(path)
        signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
        shots = ez.shots_of(frame, signals)
        table = pd.DataFrame({"shot": list(shots), "disrupted": [v[2] for v in shots.values()]})
        if official:
            part = os.path.join(base, "20ja072_fig6_data", "EAST_data_partition.mat")
            split, _ = tm.official_split(part, table)
            extra = partition_check(part, set(shots))
        else:
            split, extra = tm.our_split(table), "  the audit's seeded split (seed 42, 60/20/20)"
        lines += machine(name, frame, shots, split, extra) + [""]
    lines.append("Reading: post-result; a different Delta under S2 is dependence on the training landmark choice, not "
                 "evidence that either model is better. Nothing about phase or the published models.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "endpoint_zhu_sensitivity.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Endpoint module 2, sensitivities (post-result)\n\n" + out[:3400])


if __name__ == "__main__":
    main()
