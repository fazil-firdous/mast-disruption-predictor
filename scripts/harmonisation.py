"""What a values-based predictor keeps when the observation schedule is harmonised (docs/preregistration_harmonisation.md,
fixed before any model). Three representations of the same records (published slices, 5 ms forward-fill, a
class-independent sparse grid with a maximum age), the same 36 value features, the same boosting model, test once.

    python scripts/harmonisation.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/harmonisation.txt. Outside every stamp.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sampling_audit as sa, timing_model as tm

WINDOW_S = 0.300
R1_GRID_S = 0.005
DELTA_S = {"EAST": 0.100, "DIII-D": 0.025, "C-Mod": 0.005}
N_BOOT, SEED = 2000, 42
WORTH = 0.05


def feature_row(values):
    """last, mean and last minus first of each signal over the window's points (rows); NaN where a signal has none."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = []
        for j in range(values.shape[1]):
            x = values[:, j]
            ok = np.isfinite(x)
            if ok.any():
                xs = x[ok]
                out += [xs[-1], float(np.mean(xs)), xs[-1] - xs[0]]
            else:
                out += [np.nan, np.nan, np.nan]
        return out


def grid_points(t, values, t_cut, step, max_age=None, window=WINDOW_S):
    """(values at grid points in the window, ages): grid every `step` from the first slice to t_cut, each point taking
    the latest slice at or before it; a point whose slice is older than max_age is empty (NaN row)."""
    g = t[0] + step * np.arange(int(np.floor((t_cut - t[0]) / step + 1e-9)) + 1)
    g = g[g > t_cut - window + 1e-9]
    if not len(g):
        return np.empty((0, values.shape[1])), np.empty(0)
    idx = np.searchsorted(t, g + 1e-9, side="right") - 1
    ok = idx >= 0
    v = np.full((len(g), values.shape[1]), np.nan)
    v[ok] = values[idx[ok]]
    age = np.where(ok, g - t[np.clip(idx, 0, None)], np.nan)
    if max_age is not None:
        stale = age > max_age + 1e-9
        v[stale] = np.nan
    return v, age


def shot_rows(frame, signals, delta):
    """Per shot: label, the 36 features of R0, R1 and R2, the R2 mean observation age and empty share."""
    rows = []
    for shot, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        if len(t) < 2:
            continue
        vals = g[signals].to_numpy(dtype=float)
        t_cut = t[-1] - sa.CUT_S
        keep = (t > t_cut - WINDOW_S + 1e-9) & (t <= t_cut + 1e-9)
        r0 = feature_row(vals[keep]) if keep.any() else [np.nan] * (3 * len(signals))
        v1, _ = grid_points(t, vals, t_cut, R1_GRID_S)
        v2, age2 = grid_points(t, vals, t_cut, delta, max_age=delta)
        empty = float(np.mean(~np.isfinite(v2).any(axis=1))) if len(v2) else np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_age = float(np.nanmean(np.where(age2 <= delta + 1e-9, age2, np.nan))) if len(age2) else np.nan
        rows.append({"shot": int(shot), "disrupted": bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()),
                     "r0": r0, "r1": feature_row(v1) if len(v1) else [np.nan] * (3 * len(signals)),
                     "r2": feature_row(v2) if len(v2) else [np.nan] * (3 * len(signals)),
                     "age2": mean_age, "empty2": empty})
    return pd.DataFrame(rows)


def boosted_scores(rows, split, rep):
    from sklearn.ensemble import HistGradientBoostingClassifier
    part = rows["shot"].map(split)
    tr, te = rows[part == "train"], rows[part == "test"]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED)
    model.fit(np.vstack(tr[rep].to_numpy()), tr["disrupted"].to_numpy())
    return model.predict_proba(np.vstack(te[rep].to_numpy()))[:, 1], te["disrupted"].to_numpy()


def residual_timing(rows, split):
    """Test AUC range of a logistic regression on the R2 mean observation age alone."""
    from sklearn.linear_model import LogisticRegression
    part = rows["shot"].map(split)
    tr, te = rows[part == "train"], rows[part == "test"]
    med = tr["age2"].median()
    x = lambda d: d[["age2"]].fillna(med).to_numpy()  # noqa: E731
    m = LogisticRegression(max_iter=1000).fit(x(tr), tr["disrupted"].to_numpy())
    return sa.auc_range(m.decision_function(x(te)), te["disrupted"].to_numpy())


def paired_drop(s1, s2, y, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    point = sa.auc(s1, y) - sa.auc(s2, y)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        draws.append(sa.auc(s1[i], y[i]) - sa.auc(s2[i], y[i]))
    d = np.array([x for x in draws if np.isfinite(x)])
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def reading(residual, drop):
    if residual[1] is not None and residual[1] > 0.5 and residual[0] > 0.60:
        return "harmonisation did NOT remove the timing difference (residual timing AUC above 0.60): R2 only reported"
    if drop[1] > 0 and drop[0] >= WORTH:
        return ("a sizeable part of the R1 performance depends on the observation schedule harmonisation removes, or on "
                "short-timescale information it also removes (not separable here)")
    return "the R1 performance largely survives harmonisation"


def analyse(rows, split, name):
    out = {}
    for rep in ("r0", "r1", "r2"):
        out[rep] = boosted_scores(rows, split, rep)
    y = out["r1"][1]
    aucs = {rep: sa.auc_range(out[rep][0], y) for rep in out}
    residual = residual_timing(rows, split)
    drop = paired_drop(out["r1"][0], out["r2"][0], y)
    part = rows["shot"].map(split)
    te = rows[part == "test"]
    lines = [f"== {name}",
             f"  residual timing AUC on R2 (mean observation age only): {sa.fmt(residual)}",
             f"  {'PRIMARY ' if name == 'EAST' else ''}Drop = AUC(R1) - AUC(R2): {drop[0]:+.3f} ({drop[1]:+.3f} to {drop[2]:+.3f}): "
             f"{reading(residual, drop)}",
             f"  AUC R0 published slices {sa.fmt(aucs['r0'])}; R1 5 ms forward-fill {sa.fmt(aucs['r1'])}; R2 harmonised "
             f"{sa.fmt(aucs['r2'])}",
             f"  R2 empty points: disrupted {te.loc[te['disrupted'], 'empty2'].mean():.0%}, not "
             f"{te.loc[~te['disrupted'], 'empty2'].mean():.0%}; mean R2 observation age: disrupted "
             f"{te.loc[te['disrupted'], 'age2'].median() * 1e3:.1f} ms, not {te.loc[~te['disrupted'], 'age2'].median() * 1e3:.1f} ms",
             f"  test shots: {int(y.sum())} disrupted, {int((~y.astype(bool)).sum())} not", ""]
    return lines


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    lines = ["Harmonisation (docs/preregistration_harmonisation.md): values-only boosting (36 features: last, mean, last "
             "minus first of 12 signals, 300 ms window ending 40 ms before the last slice) on R0 published slices, R1 5 ms "
             "forward-fill, R2 a class-independent grid every Delta with a maximum age Delta. Test evaluated once. A "
             "sensitivity analysis: a drop is not all shortcut, and what remains is not pure physics.", ""]
    jobs = [("EAST", os.path.join(base, "20ja072_fig6_data", "EAST_for_nn.mat"), "official"),
            ("DIII-D", os.path.join(base, "20ja072_fig4_data", "d3d_for_nn.mat"), "ours"),
            ("C-Mod", os.path.join(base, "20ja072_fig4_data", "C_Mod_for_nn.mat"), "ours")]
    for name, path, kind in jobs:
        if not os.path.exists(path):
            continue
        frame = sa.load(path)
        signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
        print(f"{name}: {frame['shot'].nunique():,} shots, building representations ...", flush=True)
        rows = shot_rows(frame, signals, DELTA_S[name])
        if kind == "official":
            split, _ = tm.official_split(os.path.join(os.path.dirname(path), "EAST_data_partition.mat"),
                                         rows.rename(columns={"shot": "shot"}))
        else:
            split = tm.our_split(rows)
        lines += analyse(rows, split, name)
    lines.append("Not claimed: that any published model used the observation schedule.")
    out = "\n".join(lines)
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "harmonisation.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
