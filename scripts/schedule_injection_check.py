"""Accounting and mechanics of the schedule-injection study (docs/preregistration_schedule_injection.md). Requested by the
outside review AFTER the result; a diagnostic only: it changes no reading and fits no model.

1. Eligibility: shots in the file, and those left out under U or I (fewer than 2 slices), by class.
2. Nesting: U's slices are a subset of I's in every shot, with equal values at common time stamps; non-disrupted
   shots identical under U and I.
3. Cadence: the spacing of U's slices (every tenth native slice is about 50 ms only if the native spacing is 5 ms).
4. The window count under I: how many slices of the 300 ms window lie in the injected last 300 ms.
5. R2 identity: per shot, the sparse-grid points, the ages of the slices behind them, and the 36 features under U and
   under I. Where they differ, why.

    python scripts/schedule_injection_check.py /root/dv/var/www/html/reports/20JA/20JA72/20ja072_fig4_data/C_Mod_for_nn.mat

Writes RESULTS_DIR/sampling_audit/schedule_injection_check.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harmonisation as hm, sampling_audit as sa, schedule_injection as si, timing_model as tm


def same(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.array_equal(a, b, equal_nan=True))


def shot_compare(gu, gi, signals):
    """Per shot: nesting, U spacing, the I window count split by injected or not, and the R2 comparison."""
    gu, gi = gu.sort_values("time"), gi.sort_values("time")
    tu, ti = gu["time"].to_numpy(dtype=float), gi["time"].to_numpy(dtype=float)
    vu, vi = gu[signals].to_numpy(dtype=float), gi[signals].to_numpy(dtype=float)
    nested = set(gu.index) <= set(gi.index)
    common = gi.index.isin(gu.index)
    values_equal = same(gi.loc[common, signals].to_numpy(dtype=float), vu) if nested else False
    out = {"nested": nested, "values_equal": values_equal, "same_last": len(tu) > 0 and tu[-1] == ti[-1],
           "same_first": len(tu) > 0 and tu[0] == ti[0], "u_gaps": np.diff(tu)}
    if len(tu) < 2 or len(ti) < 2:
        return out
    cut = ti[-1] - sa.CUT_S
    in_window = (ti > cut - hm.WINDOW_S + 1e-9) & (ti <= cut + 1e-9)
    injected = ti >= ti[-1] - si.DENSE_S - 1e-9
    out.update(window_i=int(in_window.sum()), window_i_injected=int((in_window & injected).sum()))
    pu, au = hm.grid_points(tu, vu, tu[-1] - sa.CUT_S, si.DELTA_S, max_age=si.DELTA_S)
    pi, ai = hm.grid_points(ti, vi, cut, si.DELTA_S, max_age=si.DELTA_S)
    out.update(points_u=len(au), points_i=len(ai), r2_same=same(hm.feature_row(pu), hm.feature_row(pi)) if len(pu) and
               len(pi) else len(pu) == len(pi))
    if len(au) == len(ai):
        differ = np.array([not same(a, b) for a, b in zip(pu, pi)], dtype=bool)
        eq_age = np.isclose(au, ai, rtol=0, atol=1e-12) | (np.isnan(au) & np.isnan(ai))
        out.update(fresher_i=int((differ & (ai < au - 1e-12)).sum()), older_i=int((differ & (ai > au + 1e-12)).sum()),
                   filled_i=int((differ & np.isnan(au) & np.isfinite(ai)).sum()),
                   same_age_differ=int((differ & eq_age).sum()))
        out.update(points_differ=int(differ.sum()),
                   age_u=float(np.nanmedian(au)) if np.isfinite(au).any() else np.nan,
                   age_i=float(np.nanmedian(ai)) if np.isfinite(ai).any() else np.nan,
                   empty_u=int((~np.isfinite(pu).any(axis=1)).sum()), empty_i=int((~np.isfinite(pi).any(axis=1)).sum()))
    return out


def run(path):
    frame = sa.load(path)
    signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
    u, i = si.thin(frame, False), si.thin(frame, True)
    label = frame.groupby("shot")["time_until_disrupt"].apply(lambda x: bool(np.isfinite(x.to_numpy(dtype=float)).any()))
    n_all = frame.groupby("shot").size()
    n_u, n_i = u.groupby("shot").size(), i.groupby("shot").size()
    lines = ["Schedule injection, accounting and mechanics (diagnostic requested after the result; no model, no reading "
             "changed).", ""]
    # 1. eligibility
    ok_u, ok_i = n_u[n_u >= 2].index, n_i[n_i >= 2].index
    kept = set(ok_u) & set(ok_i)
    lines.append(f"1. Eligibility: {len(n_all):,} shots in the file ({int(label.sum()):,} disrupted).")
    for name, ids in (("fewer than 2 slices in the file", n_all[n_all < 2].index),
                      ("fewer than 2 slices under U", n_u.reindex(n_all.index).fillna(0).loc[lambda s: s < 2].index),
                      ("fewer than 2 slices under I", n_i.reindex(n_all.index).fillna(0).loc[lambda s: s < 2].index)):
        lines.append(f"- {name}: {len(ids):,} ({int(label.reindex(ids).sum()):,} disrupted)")
    lines.append(f"- kept in both: {len(kept):,} ({int(label.reindex(sorted(kept)).sum()):,} disrupted). The rule is common: "
                 "a shot is left out of both conditions if it has fewer than 2 slices in either.")
    rows = []
    for shot in sorted(kept):
        r = shot_compare(u[u["shot"] == shot], i[i["shot"] == shot], signals)
        r["shot"], r["disrupted"] = shot, bool(label[shot])
        rows.append(r)
    t = pd.DataFrame(rows)
    # 2. nesting
    nd = t[~t["disrupted"]]
    lines += ["", f"2. Nesting: U is a subset of I in {int(t['nested'].sum()):,} of {len(t):,} shots; values equal at common "
              f"time stamps in {int(t['values_equal'].sum()):,}; same first slice {int(t['same_first'].sum()):,}, same last "
              f"slice {int(t['same_last'].sum()):,}; non-disrupted shots with identical R2 features "
              f"{int(nd['r2_same'].sum()):,} of {len(nd):,} (they are unchanged by construction)."]
    # 3. cadence
    gaps = np.concatenate([g for g in t["u_gaps"] if len(g)])
    native = frame.sort_values(["shot", "time"]).groupby("shot")["time"].diff().dropna().to_numpy()
    lines += ["", f"3. Cadence: native spacing median {np.median(native) * 1e3:.2f} ms, {np.mean(np.abs(native - 0.005) > 1e-4):.1%} "
              f"of native steps differ from 5 ms by more than 0.1 ms (max {native.max() * 1e3:.1f} ms). U spacing median "
              f"{np.median(gaps) * 1e3:.2f} ms; {np.mean(np.abs(gaps - 0.05) > 1e-4):.1%} of U steps differ from 50 ms by "
              f"more than 0.1 ms (min {gaps.min() * 1e3:.1f}, max {gaps.max() * 1e3:.1f} ms)."]
    # 4. window count
    d = t[t["disrupted"]]
    lines += ["", f"4. Window under I, disrupted shots: median {d['window_i'].median():.0f} slices, of which "
              f"{d['window_i_injected'].median():.0f} in the injected last 300 ms (the window, 340 to 40 ms before the last "
              "slice, overlaps the injected segment for its last 260 ms, i.e. 52 native 5 ms slices; the first 40 ms keep "
              "the sparse cadence)."]
    # 5. R2
    d_same = int(d["r2_same"].sum())
    lines += ["", f"5. R2 (grid every 50 ms from the first slice, maximum age 50 ms): disrupted shots with identical 36 "
              f"features under U and I: {d_same:,} of {len(d):,}."]
    diff = d[~d["r2_same"]]
    if len(diff):
        lines.append(f"- where they differ ({len(diff):,} shots): grid points per shot median {diff['points_u'].median():.0f}, "
                     f"of which differing median {diff['points_differ'].median():.0f}; median age of the slice behind a grid "
                     f"point {diff['age_u'].median() * 1e3:.1f} ms under U, {diff['age_i'].median() * 1e3:.1f} ms under I; "
                     f"empty points median {diff['empty_u'].median():.0f} under U, {diff['empty_i'].median():.0f} under I.")
        lines.append(f"- differing grid points, all shots: {int(diff['points_differ'].sum()):,}; of them the slice behind "
                     f"the point is fresher under I in {int(diff['fresher_i'].sum()):,}, older under I in "
                     f"{int(diff['older_i'].sum()):,}, empty under U but filled under I in {int(diff['filled_i'].sum()):,}, "
                     f"and of equal age (a value difference not explained by freshness) in "
                     f"{int(diff['same_age_differ'].sum()):,}.")
    same_all = t[t["disrupted"]]["r2_same"].all()
    if same_all:
        lines.append("- All identical: R2 gives the same inputs under U and I for every shot, so an AUC difference under R2 "
                     "would have to come from the fitting procedure, not from the inputs; to be explained before reading.")
    split = tm.our_split(t.rename(columns={"shot": "shot"}), seed=si.SEED)
    part = t["shot"].map(split)
    for p in ("train", "test"):
        dd = t[(part == p) & t["disrupted"]]
        lines.append(f"- {p}: disrupted shots with different R2 features {int((~dd['r2_same']).sum()):,} of {len(dd):,}")
    lines += ["", "Features entering the values-only model: 36 per representation (last, arithmetic mean over the "
              "observations or grid points, last minus first, of 12 signals); no count, spacing or time stamp. R0 takes the "
              "mean over the observations in the window (more observations under I), R1 over a 5 ms forward-filled grid "
              "(time-weighted), R2 over the 50 ms grid points."]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "schedule_injection_check.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
