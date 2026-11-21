"""Endpoint module 2 (docs/preregistration_endpoint_zhu.md, fixed before any statistic): in the public Zhu et al. files,
a frozen values-only model's matched concordance with each non-disrupted control evaluated at its own end (E) or at
the case's elapsed time since its first retained slice (M). File-relative stress test, not a phase claim.

    python scripts/endpoint_zhu.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/endpoint_zhu.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harmonisation as hm, sampling_audit as sa, timing_model as tm
from src import notify

WINDOW, CUT, GRID, TOL = hm.WINDOW_S, sa.CUT_S, 0.010, 1e-9
WORTH, N_BOOT, SEED = 0.03, 2000, 42


def shots_of(frame, signals):
    """{shot: (t, values, disrupted)} with times sorted."""
    out = {}
    for shot, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        out[int(shot)] = (g["time"].to_numpy(dtype=float), g[signals].to_numpy(dtype=float),
                          bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()))
    return out


def at(t, vals, landmark):
    """(36 features, slices in the window, age of the last slice) from the slices in (landmark - WINDOW, landmark]."""
    keep = (t > landmark - WINDOW + TOL) & (t <= landmark + TOL)
    n = int(keep.sum())
    feats = hm.feature_row(vals[keep]) if n else [np.nan] * (3 * vals.shape[1])
    return feats, n, (landmark - t[keep][-1]) if n else np.nan


def elapsed(t):
    """The case's elapsed time at its decision point, rounded down to GRID."""
    return np.floor((t[-1] - CUT - t[0]) / GRID + 1e-9) * GRID


def eligible(t, landmark):
    return bool(((t > landmark - WINDOW + TOL) & (t <= landmark + TOL)).any()) and t[-1] >= landmark + CUT - TOL


def fit(shots, train, landmarks=None):
    """The registered values-only model on the training shots, each at landmarks[shot] (default: last slice - CUT)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    landmarks = landmarks or {}
    x = np.array([at(shots[s][0], shots[s][1], landmarks.get(s, shots[s][0][-1] - CUT))[0] for s in train])
    y = np.array([shots[s][2] for s in train])
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED).fit(x, y)


def build(shots, split, model=None):
    """Frozen model and the score matrices: (case scores, control scores at E, control scores at M per (case, control)
    with NaN where ineligible), the timing-only analogues, and descriptive arrays."""
    if model is None:
        model = fit(shots, [s for s in shots if split.get(s) == "train"])
    test = [s for s in shots if split.get(s) == "test"]
    cases = [s for s in test if shots[s][2]]
    controls = [s for s in test if not shots[s][2]]
    score = lambda rows: model.predict_proba(np.array(rows))[:, 1]  # noqa: E731

    case_rows = [at(shots[c][0], shots[c][1], shots[c][0][-1] - CUT) for c in cases]
    s_case = score([r[0] for r in case_rows])
    n_case = np.array([r[1] for r in case_rows], dtype=float)
    e_case = np.array([elapsed(shots[c][0]) for c in cases])
    end_rows = [at(shots[k][0], shots[k][1], shots[k][0][-1] - CUT) for k in controls]
    s_end = score([r[0] for r in end_rows])
    n_end = np.array([r[1] for r in end_rows], dtype=float)
    age_end = np.array([r[2] for r in end_rows], dtype=float)

    grid_e = np.unique(e_case)
    s_m = np.full((len(controls), len(grid_e)), np.nan)
    n_m = np.full_like(s_m, np.nan)
    age_m = np.full_like(s_m, np.nan)
    sep = np.full_like(s_m, np.nan)
    rows, where = [], []
    for j, k in enumerate(controls):
        t, v, _ = shots[k]
        for g, e in enumerate(grid_e):
            landmark = t[0] + e
            if eligible(t, landmark):
                f, n, age = at(t, v, landmark)
                rows.append(f)
                where.append((j, g))
                n_m[j, g], age_m[j, g] = n, age
                sep[j, g] = (t[-1] - CUT) - landmark
    if rows:
        scores = score(rows)
        for (j, g), sv in zip(where, scores):
            s_m[j, g] = sv
    col = np.searchsorted(grid_e, e_case)
    return {"s_case": s_case, "n_case": n_case, "s_end": s_end, "n_end": n_end, "age_end": age_end,
            "s_m": s_m[:, col].T, "n_m": n_m[:, col].T, "age_m": age_m[:, col].T, "sep": sep[:, col].T,
            "cases": cases, "controls": controls, "model": model, "e_case": e_case}


def win_matrix(case_vals, control_vals):
    """cases x controls: 1 if the case value is higher, 1/2 tie, 0 lower; NaN where the control value is NaN."""
    a = case_vals[:, None]
    b = control_vals
    if b.ndim == 1:
        b = np.broadcast_to(b[None, :], (len(case_vals), len(b)))
    out = np.where(a > b, 1.0, np.where(a == b, 0.5, 0.0))
    return np.where(np.isfinite(b), out, np.nan)


def concordance(w, mask, ci=None, ki=None):
    """Mean over cases (with at least one eligible control) of each case's mean win over its eligible controls."""
    if ci is not None:
        w, mask = w[np.ix_(ci, ki)], mask[np.ix_(ci, ki)]
    w = np.where(mask, w, np.nan)
    have = np.isfinite(w).any(axis=1) if w.size else np.zeros(0, dtype=bool)
    per = np.nanmean(w[have], axis=1) if have.any() else np.array([])
    return float(per.mean()) if len(per) else np.nan


def delta_range(w_e, w_m, mask, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    point = concordance(w_e, mask) - concordance(w_m, mask)
    nc, nk = mask.shape
    draws = []
    for _ in range(n_boot):
        ci, ki = rng.integers(0, nc, nc), rng.integers(0, nk, nk)
        draws.append(concordance(w_e, mask, ci, ki) - concordance(w_m, mask, ci, ki))
    d = np.array([x for x in draws if np.isfinite(x)])
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def reading(d):
    if (d[1] > 0 or d[2] < 0) and abs(d[0]) >= WORTH:
        return ("the measured discrimination of this frozen values-only predictor depends materially on where the "
                "control is evaluated (file-relative; not a physical phase claim)")
    return "no material dependence is shown for this predictor and these records (not shown equivalent)"


def machine_lines(name, b, primary=False):
    mask = np.isfinite(b["s_m"])
    w_e = win_matrix(b["s_case"], b["s_end"])
    w_m = win_matrix(b["s_case"], b["s_m"])
    t_e = win_matrix(b["n_case"], b["n_end"])
    t_m = win_matrix(b["n_case"], b["n_m"])
    d = delta_range(w_e, w_m, mask)
    labels = np.r_[np.ones(len(b["cases"])), np.zeros(len(b["controls"]))].astype(bool)
    auc = sa.auc_range(np.r_[b["s_case"], b["s_end"]], labels)
    sep = b["sep"][mask] * 1e3
    change = (np.broadcast_to(b["s_end"][None, :], mask.shape) - b["s_m"])[mask]
    cases_ok = int(mask.any(axis=1).sum())
    controls_ok = int(mask.any(axis=0).sum())
    lines = [f"== {name}: test {len(b['cases']):,} cases, {len(b['controls']):,} controls; cases with an eligible control "
             f"{cases_ok:,}; controls ever eligible {controls_ok:,}; eligible combinations {int(mask.sum()):,}",
             f"  {'PRIMARY ' if primary else ''}C_E {concordance(w_e, mask):.3f}, C_M {concordance(w_m, mask):.3f}; "
             f"Delta = C_E - C_M {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f}). Reading: {reading(d)}.",
             f"  usual shot-level AUC of the frozen model (controls at E): {sa.fmt(auc)}",
             f"  separation L_E - L_M: median {np.median(sep):.0f} ms (quartiles {np.percentile(sep, 25):.0f} to "
             f"{np.percentile(sep, 75):.0f}); within 40 ms {np.mean(np.abs(sep) < 40):.0%}",
             f"  control score E - M: mean {change.mean():+.4f}, median {np.median(change):+.4f}",
             f"  observation audit, slices in the window, median: cases {np.median(b['n_case']):.0f}; controls at E "
             f"{np.median(b['n_end']):.0f}, at M {np.nanmedian(b['n_m'][mask]):.0f}; age of the last slice (ms): E "
             f"{np.nanmedian(b['age_end']) * 1e3:.1f}, M {np.nanmedian(b['age_m'][mask]) * 1e3:.1f}",
             f"  timing-only concordance (slice count): E {concordance(t_e, mask):.3f}, M {concordance(t_m, mask):.3f}"]
    return lines


def run(base):
    lines = ["Endpoint module 2 (docs/preregistration_endpoint_zhu.md): frozen values-only boosting (36 R0 features, "
             "300 ms window), trained once with every training shot's landmark at its last slice minus 40 ms. Each test "
             "case scored at its own decision point; each control at its own end (E) or at the case's elapsed time since "
             "its first retained slice (M). File-relative stress test, not a phase claim.", ""]
    jobs = [("EAST", os.path.join(base, "20ja072_fig4_data", "EAST_for_nn.mat"), True),
            ("DIII-D", os.path.join(base, "20ja072_fig4_data", "d3d_for_nn.mat"), False),
            ("C-Mod", os.path.join(base, "20ja072_fig4_data", "C_Mod_for_nn.mat"), False)]
    for name, path, official in jobs:
        if not os.path.exists(path):
            continue
        print(f"{name} ...", flush=True)
        frame = sa.load(path)
        signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
        shots = shots_of(frame, signals)
        table = pd.DataFrame({"shot": list(shots), "disrupted": [v[2] for v in shots.values()]})
        if official:
            split, missing = tm.official_split(os.path.join(base, "20ja072_fig6_data", "EAST_data_partition.mat"),
                                               table)
            note = f"  the release's own lists (fig6); listed shots not in the fig4 file: {missing:,}"
        else:
            split, note = tm.our_split(table), "  the audit's seeded split (seed 42, 60/20/20)"
        lines += machine_lines(name, build(shots, split), primary=official)
        lines.append(note)
        if name == "C-Mod":
            lines.append("  caveat: C-Mod's stored times carry sub-microsecond offsets; window edges can differ by one "
                         "slice.")
        lines.append("")
    lines.append("Not claimed: plasma phase; anything about the published models or their evaluation.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "endpoint_zhu.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Endpoint module 2 (public Zhu files)\n\n" + out[:3400])


if __name__ == "__main__":
    main()
