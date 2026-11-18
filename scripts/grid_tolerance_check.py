"""Is the harmonisation study's residual observation age partly an artefact of a time tolerance? A diagnostic found after
the schedule-injection result (NOTES.md 2026-11-18): the files' time stamps carry floating-point offsets of up to about
0.1 us, larger than the 1e-9 s tolerance of `harmonisation.grid_points`, so a grid point can miss a slice that is, in
effect, at the same time and take an older one. No model is fitted; nothing registered is re-read.

For each machine: the native time-step precision, and the R2 grid ages (grid every Delta from the first slice, window
300 ms ending 40 ms before the last slice, as in the harmonisation study) computed with the study's tolerance (1e-9 s)
and with 1e-6 s (above the stored precision, far below any sampling step): the share of grid points that change, the
median age by class, and the AUC of the shot's mean age (higher = disrupted, no fit; 0.5 is no association).

    python scripts/grid_tolerance_check.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/grid_tolerance_check.txt. Outside every stamp.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harmonisation as hm, sampling_audit as sa

TOLERANCES = (1e-9, 1e-6)


def grid_ages(t, t_cut, step, tol, window=hm.WINDOW_S, max_age=None):
    """Ages of the slices behind the R2 grid points, as `harmonisation.grid_points` builds them, with tolerance tol
    (NaN where no slice, or older than max_age)."""
    g = t[0] + step * np.arange(int(np.floor((t_cut - t[0]) / step + tol)) + 1)
    g = g[g > t_cut - window + tol]
    if not len(g):
        return np.empty(0)
    idx = np.searchsorted(t, g + tol, side="right") - 1
    age = np.where(idx >= 0, g - t[np.clip(idx, 0, None)], np.nan)
    if max_age is not None:
        age = np.where(age > max_age + tol, np.nan, age)
    return age


def machine_lines(name, frame, delta):
    rows = []
    steps = []
    for _, g in frame.groupby("shot", sort=True):
        t = np.sort(g["time"].to_numpy(dtype=float))
        if len(t) < 2:
            continue
        steps.append(np.diff(t))
        cut = t[-1] - sa.CUT_S
        ages = [grid_ages(t, cut, delta, tol, max_age=delta) for tol in TOLERANCES]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rows.append({"disrupted": bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()),
                         "points": len(ages[0]),
                         "changed": int(np.sum(~np.isclose(ages[0], ages[1], rtol=0, atol=1e-7, equal_nan=True)))
                         if len(ages[0]) == len(ages[1]) else len(ages[0]),
                         **{f"mean_{k}": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
                            for k, a in enumerate(ages)},
                         **{f"med_{k}": float(np.nanmedian(a)) if np.isfinite(a).any() else np.nan
                            for k, a in enumerate(ages)}})
    steps = np.concatenate(steps)
    y = np.array([r["disrupted"] for r in rows])
    lines = [f"== {name} (Delta {delta * 1e3:.0f} ms): {len(rows):,} shots ({int(y.sum()):,} disrupted)",
             f"  time-step offsets from the nearest 0.1 ms multiple: median {np.median(np.abs(steps - np.round(steps, 4))) * 1e6:.3f} us, "
             f"max {np.max(np.abs(steps - np.round(steps, 4))) * 1e6:.3f} us"]
    pts = np.array([r["points"] for r in rows])
    ch = np.array([r["changed"] for r in rows])
    for label, keep in (("disrupted", y), ("not", ~y)):
        lines.append(f"  {label}: grid points whose age changes between tolerances {ch[keep].sum():,} of {pts[keep].sum():,} "
                     f"({ch[keep].sum() / max(pts[keep].sum(), 1):.1%}) in {int((ch[keep] > 0).sum()):,} shots")
    for k, tol in enumerate(TOLERANCES):
        mean_age = np.array([r[f"mean_{k}"] for r in rows])
        med = {lab: np.nanmedian(np.array([r[f"med_{k}"] for r in rows])[keep]) for lab, keep in (("d", y), ("n", ~y))}
        ok = np.isfinite(mean_age)
        lines.append(f"  tolerance {tol:g} s: median age disrupted {med['d'] * 1e3:.2f} ms, not {med['n'] * 1e3:.2f} ms; "
                     f"AUC of the mean age {sa.fmt(sa.auc_range(mean_age[ok], y[ok]))}")
    return lines


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    lines = ["Grid tolerance check (diagnostic after the schedule-injection result; no model, no registered reading "
             "re-read). R2 ages with the harmonisation study's tolerance (1e-9 s) and with 1e-6 s.", ""]
    jobs = [("EAST", os.path.join(base, "20ja072_fig6_data", "EAST_for_nn.mat")),
            ("DIII-D", os.path.join(base, "20ja072_fig4_data", "d3d_for_nn.mat")),
            ("C-Mod", os.path.join(base, "20ja072_fig4_data", "C_Mod_for_nn.mat"))]
    for name, path in jobs:
        if os.path.exists(path):
            print(f"{name} ...", flush=True)
            lines += machine_lines(name, sa.load(path), hm.DELTA_S[name]) + [""]
    out = "\n".join(lines)
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "grid_tolerance_check.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
