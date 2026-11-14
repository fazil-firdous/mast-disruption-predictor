"""Does a public multi-machine disruption dataset carry its outcome in its sampling? (docs/preregistration_sampling_audit.md,
fixed before any statistic.) No model: only the time structure of the records.

Reads the *_for_nn.mat files of Zhu et al. (Harvard Dataverse doi:10.7910/DVN/XIOHW1, fig4 data); the files are not
redistributed.

    python scripts/sampling_audit.py /root/dv/var/www/html/reports/20JA/20JA72/20ja072_fig4_data

Writes RESULTS_DIR/sampling_audit/ (per-shot statistics, summary). Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

MACHINES = {"C_Mod": "C-Mod", "d3d": "DIII-D", "EAST": "EAST"}
CUT_S, SPAN_S, GRID_S = 0.040, 0.100, 0.005
OTHER_CUTS_S = (0.0, 0.020, 0.100, 0.300)
N_BOOT, SEED = 2000, 42


def folder():
    return os.path.join(config.RESULTS_DIR, "sampling_audit")


def load(path):
    """(frame of shot, time, time_until_disrupt, the plasma signals) from one *_for_nn.mat file."""
    import scipy.io
    m = scipy.io.loadmat(path, squeeze_me=True)
    n = len(m["shot"])
    columns = {k: np.asarray(v) for k, v in m.items() if not k.startswith("__") and getattr(v, "shape", None) == (n,)}
    return pd.DataFrame(columns)


def spacing(t, t_cut, span=SPAN_S):
    """S1: median gap between slices in (t_cut - span, t_cut]; with fewer than 2 there, the gap of the last two slices
    at or before t_cut; NaN with fewer than 2 slices up to t_cut."""
    up = t[t <= t_cut + 1e-9]
    if len(up) < 2:
        return np.nan
    inside = up[up > t_cut - span + 1e-9]
    if len(inside) >= 2:
        return float(np.median(np.diff(inside)))
    return float(up[-1] - up[-2])


def change_rate(t, values, t_cut, span=SPAN_S, grid=GRID_S):
    """S2: forward-fill every signal onto a grid of `grid` from the first slice to t_cut; the share of grid steps in
    (t_cut - span, t_cut] at which at least one signal changes value. NaN with fewer than 2 grid points there."""
    keep = t <= t_cut + 1e-9
    t, values = t[keep], values[keep]
    if len(t) < 1:
        return np.nan
    g = t[0] + grid * np.arange(int(np.floor((t_cut - t[0]) / grid + 1e-9)) + 1)
    idx = np.searchsorted(t, g + 1e-9, side="right") - 1
    filled = values[idx]
    inside = g > t_cut - span + 1e-9
    if inside.sum() < 2:
        return np.nan
    f = filled[inside]
    same = np.all((f[1:] == f[:-1]) | (np.isnan(f[1:]) & np.isnan(f[:-1])), axis=1)
    return float(np.mean(~same))


def shot_table(frame, signals, cut=CUT_S):
    """One row per shot: disrupted, S1, S2, S3 and S1 at the other cuts."""
    rows = []
    for shot, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        if len(t) == 0:
            continue
        t_cut = t[-1] - cut
        upto = t[t <= t_cut + 1e-9]
        row = {"shot": int(shot), "disrupted": bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()),
               "s1": spacing(t, t_cut), "s2": change_rate(t, g[signals].to_numpy(dtype=float), t_cut),
               "s3": float(np.median(np.diff(upto))) if len(upto) >= 2 else np.nan}
        for c in OTHER_CUTS_S:
            row[f"s1_cut{int(round(c * 1000))}"] = spacing(t, t[-1] - c)
        rows.append(row)
    return pd.DataFrame(rows)


def auc(score, label):
    """P(score of a disrupted shot > score of a non-disrupted one), ties one half; NaN scores left out."""
    from scipy.stats import rankdata
    ok = np.isfinite(score)
    s, y = score[ok], label[ok].astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return np.nan
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def auc_range(score, label, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    point = auc(score, label)
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, len(score), len(score))
        draws.append(auc(score[i], label[i]))
    d = np.array([x for x in draws if np.isfinite(x)])
    return (point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if len(d) else (point, None, None)


def reading(r):
    if r[1] is None:
        return "no range"
    if r[1] > 0.5 and r[0] >= 0.60:
        return "the sampling carries outcome information"
    if r[1] > 0.5:
        return "above 0.5 but under 0.60"
    if r[2] < 0.5:
        return "below 0.5 (reported; no reading fixed)"
    return "no such signal by this statistic"


def fmt(r):
    return "-" if r[0] is None or not np.isfinite(r[0]) else f"{r[0]:.3f} ({r[1]:.3f} to {r[2]:.3f})"


def text(tables, n_boot=N_BOOT):
    lines = ["Sampling audit (docs/preregistration_sampling_audit.md): no model, only the time structure of the public "
             "records of Zhu et al. (Dataverse doi:10.7910/DVN/XIOHW1). Time used ends 40 ms before each shot's last "
             "slice. AUC of disrupted against non-disrupted shots; 95% range from bootstrap of shots.", ""]
    for key, name in MACHINES.items():
        if key not in tables:
            continue
        t = tables[key]
        y = t["disrupted"].to_numpy()
        r1 = auc_range(-t["s1"].to_numpy(), y, n_boot)
        r2 = auc_range(t["s2"].to_numpy(), y, n_boot)
        r3 = auc_range(-t["s3"].to_numpy(), y, n_boot)
        left = t["s1"].isna()
        lines += [f"== {name}: {len(t):,} shots ({int(y.sum()):,} disrupted); left out for S1: "
                  f"{int(left[t['disrupted']].sum())} disrupted, {int(left[~t['disrupted']].sum())} not",
                  f"  PRIMARY S1 (spacing in the 100 ms before the cut), AUC {fmt(r1)}: {reading(r1)}",
                  f"  S2 (change rate after forward-fill to 5 ms), AUC {fmt(r2)}: {reading(r2)}",
                  f"  S3 (whole-record spacing), AUC {fmt(r3)}",
                  f"  median S1: disrupted {np.nanmedian(t.loc[t['disrupted'], 's1']) * 1e3:.1f} ms, not "
                  f"{np.nanmedian(t.loc[~t['disrupted'], 's1']) * 1e3:.1f} ms; median S2: disrupted "
                  f"{np.nanmedian(t.loc[t['disrupted'], 's2']):.2f}, not {np.nanmedian(t.loc[~t['disrupted'], 's2']):.2f}"]
        other = []
        for c in OTHER_CUTS_S:
            col = f"s1_cut{int(round(c * 1000))}"
            other.append(f"{int(round(c * 1000))} ms: {auc(-t[col].to_numpy(), y):.3f}")
        lines += ["  S1 AUC with other cuts: " + "; ".join(other), ""]
    lines.append("Not claimed: that any published model used this, or that any reported performance is inflated.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    tables = {}
    for key, name in MACHINES.items():
        path = os.path.join(base, f"{key}_for_nn.mat")
        if not os.path.exists(path):
            print(f"{name}: no {path}")
            continue
        frame = load(path)
        signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
        print(f"{name}: {frame['shot'].nunique():,} shots, signals: {', '.join(signals)}", flush=True)
        tables[key] = shot_table(frame, signals)
    os.makedirs(folder(), exist_ok=True)
    for key, t in tables.items():
        t.to_csv(os.path.join(folder(), f"{key}_shots.csv"), index=False)
    out = text(tables)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
