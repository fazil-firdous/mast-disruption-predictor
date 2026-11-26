"""Checks requested for the between-class timing statistics (docs/preregistration_sampling_audit_2.md, addendum of
2026-11-26, specified before computing): per-bin minimum, like-for-like cohort for S1 and C2, S1 from time stamps
quantised to 1 us, population counts, and S1/C1/C2 on the synthetic planted-leak families. No model. Descriptive.

    python scripts/between_class_extras.py DATA     # public files (fig4 folder's parent)
    python scripts/between_class_extras.py --synthetic

Writes RESULTS_DIR/sampling_audit/between_class_extras[_synthetic].txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import planted_leaks as pl, sampling_audit as sa, sampling_audit_2 as s2
from src import notify

N_BINS, CUT = 10, sa.CUT_S


def table_of(shots):
    """shots: {shot: (t, disrupted)} -> one row per record with S1, S1 quantised, C1, C2, length."""
    rows = []
    for shot, (t, d) in shots.items():
        t = np.sort(np.asarray(t, dtype=float))
        cut = t[-1] - CUT
        rows.append({"shot": shot, "disrupted": d, "s1": sa.spacing(t, cut), "s1q": round(sa.spacing(t, cut) * 1e6) / 1e6 if np.isfinite(sa.spacing(t, cut)) else np.nan,
                     "c1": s2.flag_share(t, cut), "c2": s2.early_spacing(t), "length": t[-1] - t[0]})
    return pd.DataFrame(rows)


def auc_line(score, label, name, n_boot=2000):
    ok = np.isfinite(score)
    r = sa.auc_range(score[ok], label[ok], n_boot=n_boot)
    lo = f"{r[1]:.3f}" if r[1] is not None else "n/a"
    hi = f"{r[2]:.3f}" if r[2] is not None else "n/a"
    return f"{name} {r[0]:.3f} ({lo}-{hi}; {int(label[ok].sum()):,} / {int((~label[ok]).sum()):,})"


def machine_lines(tab, n_boot=2000):
    y = tab["disrupted"].to_numpy(dtype=bool)
    out = [f"  records: {len(tab):,} ({int(y.sum()):,} disrupted, {int((~y).sum()):,} not); records of at least 1 s: "
           f"{int((tab['length'] >= 1.0).sum()):,}; at least 600 ms: {int((tab['length'] >= 0.6).sum()):,}"]
    out.append("  all records: " + "; ".join([auc_line(-tab["s1"].to_numpy(), y, "S1", n_boot),
                                              auc_line(-tab["s1q"].to_numpy(), y, "S1 (spacing quantised to 1 us)", n_boot),
                                              auc_line(tab["c1"].to_numpy(), y, "C1", n_boot)]))
    long = tab[tab["length"] >= s2.EARLY_MIN_RECORD_S]
    yl = long["disrupted"].to_numpy(dtype=bool)
    out.append("  records >= 600 ms (like for like): " + "; ".join([auc_line(-long["s1"].to_numpy(), yl, "S1", n_boot),
                                                                   auc_line(-long["c2"].to_numpy(), yl, "C2", n_boot)]))
    order = tab.sort_values("shot").reset_index(drop=True)
    bins = np.array_split(np.arange(len(order)), N_BINS)
    per = []
    for b in bins:
        part = order.iloc[b]
        yb = part["disrupted"].to_numpy(dtype=bool)
        per.append((sa.auc(-part["s1"].to_numpy(), yb), sa.auc(part["c1"].to_numpy(), yb), int(yb.sum()), int((~yb).sum())))
    per = np.array(per, dtype=float)
    out.append(f"  ten shot-number bins: S1 min {np.nanmin(per[:, 0]):.3f}, mean {np.nanmean(per[:, 0]):.3f}; C1 min "
               f"{np.nanmin(per[:, 1]):.3f}, mean {np.nanmean(per[:, 1]):.3f}; smallest bin {int(per[:, 2].min()):,} "
               f"disrupted / {int(per[:, 3].min()):,} not")
    return out


def public(base):
    lines = ["Between-class timing statistics, requested checks (no model).", ""]
    for name, file in (("EAST", "EAST_for_nn.mat"), ("DIII-D", "d3d_for_nn.mat"), ("C-Mod", "C_Mod_for_nn.mat")):
        path = os.path.join(base, "20ja072_fig4_data", file)
        if not os.path.exists(path):
            continue
        print(f"{name} ...", flush=True)
        frame = sa.load(path)
        shots = {int(s): (g["time"].to_numpy(dtype=float), bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()))
                 for s, g in frame.groupby("shot", sort=True)}
        lines += [f"== {name}"] + machine_lines(table_of(shots)) + [""]
    return "\n".join(lines)


def synthetic(seeds=range(1, 21), n_boot=200):
    lines = ["S1, C1, C2 on the synthetic planted-leak families (docs/preregistration_planted_leaks.md generator, "
             "strength 0.5, seeds 1-20). Flag line 0.80 for S1 and C1.", ""]
    for family in ("L0", "L_T", "L_D", "L_E"):
        rows = []
        for seed in seeds:
            sh = pl.generate(family, seed, 0.5)
            tab = table_of({k: (v[0], v[2]) for k, v in sh.items()})
            y = tab["disrupted"].to_numpy(dtype=bool)
            rows.append([sa.auc(-tab["s1"].to_numpy(), y), sa.auc(tab["c1"].to_numpy(), y),
                         sa.auc(-tab["c2"].to_numpy(), y), sa.auc(-tab["s1q"].to_numpy(), y)])
        r = np.array(rows)
        lines.append(f"  {family}: S1 median {np.nanmedian(r[:, 0]):.3f}, >= 0.80 in {int(np.sum(r[:, 0] >= 0.8))}/20; "
                     f"C1 median {np.nanmedian(r[:, 1]):.3f}, >= 0.80 in {int(np.sum(r[:, 1] >= 0.8))}/20; "
                     f"C2 median {np.nanmedian(r[:, 2]):.3f}; S1 spacing quantised to 1 us: median {np.nanmedian(r[:, 3]):.3f}, "
                     f">= 0.80 in {int(np.sum(r[:, 3] >= 0.8))}/20")
    return "\n".join(lines)


def main():
    syn = "--synthetic" in sys.argv
    if not syn and len(sys.argv) < 2:
        sys.exit(__doc__)
    out = synthetic() if syn else public(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "between_class_extras" + ("_synthetic" if syn else "") + ".txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    if not syn:
        notify.send("Between-class extras\n\n" + out[:3400])


if __name__ == "__main__":
    main()
