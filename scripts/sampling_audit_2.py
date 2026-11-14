"""Sampling audit, part 2 (docs/preregistration_sampling_audit_2.md, fixed before any statistic): checks an outside
reviewer asked for. No model; reads the same *_for_nn.mat files as scripts/sampling_audit.py.

    python scripts/sampling_audit_2.py /root/dv/var/www/html/reports/20JA/20JA72/20ja072_fig4_data

Writes RESULTS_DIR/sampling_audit/part2.txt and per-shot tables. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sampling_audit as sa

GRID_S, SPAN_S, CUT_S = sa.GRID_S, sa.SPAN_S, sa.CUT_S
EARLY = (0.100, 0.300)  # after the first slice
EARLY_MIN_RECORD_S = 0.600
C3_MIN_RECORD_S = 1.0
C3_CUTS_S = (0.0, 0.020, 0.040, 0.100, 0.300)
N_BINS, MIN_PER_CLASS = 10, 20


def flag_share(t, t_cut, grid=GRID_S, span=SPAN_S):
    """C1: on a grid from the first slice to t_cut, the share of steps (g - grid, g] in (t_cut - span, t_cut] that
    hold at least one slice time. No value is read. NaN with fewer than 2 grid steps there."""
    t = np.sort(t[t <= t_cut + 1e-9])
    if not len(t):
        return np.nan
    g = t[0] + grid * np.arange(1, int(np.floor((t_cut - t[0]) / grid + 1e-9)) + 1)
    g = g[g > t_cut - span + 1e-9]
    if len(g) < 2:
        return np.nan
    left = np.searchsorted(t, g - grid, side="right")
    right = np.searchsorted(t, g, side="right")
    return float(np.mean(right > left))


def early_spacing(t, window=EARLY):
    """C2: median spacing of slices in (t_first + window[0], t_first + window[1]]; NaN with fewer than 2 there."""
    t = np.sort(t)
    inside = t[(t > t[0] + window[0] + 1e-9) & (t <= t[0] + window[1] + 1e-9)]
    return float(np.median(np.diff(inside))) if len(inside) >= 2 else np.nan


def window_count(t, t_cut, span=SPAN_S):
    return int(np.sum((t > t_cut - span + 1e-9) & (t <= t_cut + 1e-9)))


def shot_table(frame):
    rows = []
    for shot, g in frame.groupby("shot", sort=True):
        t = np.sort(g["time"].to_numpy(dtype=float))
        if not len(t):
            continue
        tud = g["time_until_disrupt"].to_numpy(dtype=float)
        times = g["time"].to_numpy(dtype=float)
        disrupted = bool(np.isfinite(tud).any())
        row = {"shot": int(shot), "disrupted": disrupted, "t_first": t[0], "t_last": t[-1], "record": t[-1] - t[0],
               "c1": flag_share(t, t[-1] - CUT_S), "c2": early_spacing(t),
               "t_disrupt": float(np.nanmedian(times + tud)) if disrupted else np.nan}
        for c in C3_CUTS_S:
            key = int(round(c * 1000))
            row[f"s1_{key}"] = sa.spacing(t, t[-1] - c)
            row[f"n_{key}"] = window_count(t, t[-1] - c)
        rows.append(row)
    return pd.DataFrame(rows)


def text(tables, n_boot=sa.N_BOOT):
    lines = ["Sampling audit, part 2 (docs/preregistration_sampling_audit_2.md): no model. AUC disrupted against "
             "non-disrupted shots, 95% range from bootstrap of shots.", ""]
    for key, name in sa.MACHINES.items():
        if key not in tables:
            continue
        t = tables[key]
        y = t["disrupted"].to_numpy()
        lines.append(f"== {name}: {len(t):,} shots ({int(y.sum()):,} disrupted)")
        r1 = sa.auc_range(t["c1"].to_numpy(dtype=float), y, n_boot)
        lines.append(f"  C1 observation flags (no value read), AUC {sa.fmt(r1)}: {sa.reading(r1)}; median share "
                     f"disrupted {t.loc[t['disrupted'], 'c1'].median():.2f}, not {t.loc[~t['disrupted'], 'c1'].median():.2f}")
        early = t[(t["record"] >= EARLY_MIN_RECORD_S) & t["c2"].notna()]
        ye = early["disrupted"].to_numpy()
        r2 = sa.auc_range(-early["c2"].to_numpy(dtype=float), ye, n_boot)
        lines.append(f"  C2 fixed early window (100-300 ms after the first slice; {int(ye.sum()):,} disrupted, "
                     f"{int((~ye).sum()):,} not), AUC {sa.fmt(r2)}: {sa.reading(r2)}; median spacing disrupted "
                     f"{early.loc[early['disrupted'], 'c2'].median() * 1e3:.1f} ms, not "
                     f"{early.loc[~early['disrupted'], 'c2'].median() * 1e3:.1f} ms")
        lines.append(f"  record length, median: disrupted {t.loc[t['disrupted'], 'record'].median():.2f} s, not "
                     f"{t.loc[~t['disrupted'], 'record'].median():.2f} s")
        long = t[t["record"] >= C3_MIN_RECORD_S]
        yl = long["disrupted"].to_numpy()
        cells = []
        for c in C3_CUTS_S:
            k = int(round(c * 1000))
            cells.append(f"{k} ms: AUC {sa.auc(-long[f's1_{k}'].to_numpy(dtype=float), yl):.3f}, slices in 100 ms "
                         f"{long.loc[long['disrupted'], f'n_{k}'].median():.0f} | {long.loc[~long['disrupted'], f'n_{k}'].median():.0f}")
        lines.append(f"  C3 on records of at least 1 s ({int(yl.sum()):,} disrupted, {int((~yl).sum()):,} not); "
                     "per cut, AUC of minus S1 and median slices (disrupted | not):")
        lines += [f"    {c}" for c in cells]
        gap = (t["t_last"] - t["t_disrupt"])[t["disrupted"]].dropna()
        if len(gap):
            lines.append(f"  C4 last slice minus disruption time (disrupted shots): median {gap.median() * 1e3:+.1f} ms, "
                         f"5-95% {gap.quantile(0.05) * 1e3:+.1f} to {gap.quantile(0.95) * 1e3:+.1f} ms")
        bins = pd.qcut(t["shot"].rank(method="first"), N_BINS, labels=False)
        aucs, both = [], 0
        for b in range(N_BINS):
            part = t[bins == b]
            nd, nn = int(part["disrupted"].sum()), int((~part["disrupted"]).sum())
            if nd >= MIN_PER_CLASS and nn >= MIN_PER_CLASS:
                both += 1
                aucs.append(sa.auc(-part["s1_40"].to_numpy(dtype=float), part["disrupted"].to_numpy()))
        if both >= 3:
            within = float(np.nanmean(aucs))
            verdict = ("not a difference between shot ranges" if within >= 0.90 else
                       "weaker within shot ranges than overall")
            lines.append(f"  C5 shot-number bins with both classes (>= {MIN_PER_CLASS} each): {both} of {N_BINS}; mean "
                         f"within-bin AUC {within:.3f}: {verdict}")
        else:
            lines.append(f"  C5 shot-number bins with both classes: {both} of {N_BINS}: too few to tell a shot-range "
                         "difference from a sampling one from these files")
        lines.append("")
    lines.append("Not claimed: that any published model used this, or that any reported performance is inflated.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    tables = {}
    for key, name in sa.MACHINES.items():
        path = os.path.join(sys.argv[1], f"{key}_for_nn.mat")
        if os.path.exists(path):
            frame = sa.load(path)
            print(f"{name}: {frame['shot'].nunique():,} shots", flush=True)
            tables[key] = shot_table(frame)
    os.makedirs(sa.folder(), exist_ok=True)
    for key, t in tables.items():
        t.to_csv(os.path.join(sa.folder(), f"{key}_part2_shots.csv"), index=False)
    out = text(tables)
    with open(os.path.join(sa.folder(), "part2.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
