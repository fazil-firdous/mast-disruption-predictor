"""A timing-only predictor on the public files' own held-out split (docs/preregistration_timing_model.md, fixed before
any model). Three timestamp features per shot, logistic regression tuned on validation, test evaluated once.

    python scripts/timing_model.py /root/dv/var/www/html/reports/20JA/20JA72

(the folder that holds 20ja072_fig4_data/ and 20ja072_fig6_data/). Writes RESULTS_DIR/sampling_audit/timing_model.txt.
Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sampling_audit as sa, sampling_audit_2 as s2

FEATURES = ["occ", "gap", "maxgap"]
CS = (0.01, 0.1, 1.0, 10.0)
WORTH = 0.80


def max_gap(t, t_cut, span=sa.SPAN_S):
    """The largest spacing in (t_cut - span, t_cut], including the step into the window from the last slice before it."""
    up = np.sort(t[t <= t_cut + 1e-9])
    if len(up) < 2:
        return np.nan
    first = np.searchsorted(up, t_cut - span + 1e-9, side="right")
    seg = up[max(first - 1, 0):]
    return float(np.max(np.diff(seg))) if len(seg) >= 2 else float(up[-1] - up[-2])


def features(frame):
    """One row per shot: shot, disrupted, occ, gap, maxgap (timestamps only)."""
    rows = []
    for shot, g in frame.groupby("shot", sort=True):
        t = np.sort(g["time"].to_numpy(dtype=float))
        if not len(t):
            continue
        cut = t[-1] - sa.CUT_S
        rows.append({"shot": int(shot), "disrupted": bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()),
                     "occ": s2.flag_share(t, cut), "gap": sa.spacing(t, cut), "maxgap": max_gap(t, cut)})
    return pd.DataFrame(rows)


def our_split(table, seed=42, fractions=(0.6, 0.2, 0.2)):
    """shot -> train / val / test, seeded and stratified by class (for files without an official split)."""
    rng = np.random.default_rng(seed)
    out = {}
    for flag in (True, False):
        ids = rng.permutation(table.loc[table["disrupted"] == flag, "shot"].to_numpy())
        n_tr, n_va = int(round(fractions[0] * len(ids))), int(round(fractions[1] * len(ids)))
        for i, s in enumerate(ids):
            out[int(s)] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return out


def fit_and_test(table, split):
    """(test AUC range, fixed-statistic test AUCs, chosen C, validation AUC, counts)."""
    from sklearn.linear_model import LogisticRegression
    t = table.assign(part=table["shot"].map(split)).dropna(subset=["part"])
    tr, va, te = (t[t["part"] == p] for p in ("train", "val", "test"))
    med = tr[FEATURES].median()
    mu, sd = tr[FEATURES].fillna(med).mean(), tr[FEATURES].fillna(med).std().replace(0, 1)

    def x(part):
        return ((part[FEATURES].fillna(med) - mu) / sd).to_numpy()

    best = None
    for c in CS:
        m = LogisticRegression(C=c, max_iter=1000).fit(x(tr), tr["disrupted"])
        a = sa.auc(m.decision_function(x(va)), va["disrupted"].to_numpy())
        if best is None or a > best[1]:
            best = (c, a, m)
    c, val_auc, model = best
    y = te["disrupted"].to_numpy()
    r = sa.auc_range(model.decision_function(x(te)), y)
    fixed = {"occ": sa.auc(te["occ"].to_numpy(dtype=float), y), "-gap": sa.auc(-te["gap"].to_numpy(dtype=float), y)}
    counts = {p: (int(d["disrupted"].sum()), int((~d["disrupted"]).sum())) for p, d in (("train", tr), ("val", va), ("test", te))}
    return r, fixed, c, val_auc, counts


def reading(r):
    if r[1] is None:
        return "no range"
    if r[1] > 0.5 and r[0] >= WORTH:
        return "a predictor trained only on observation times discriminates the classes on the held-out test shots"
    if r[1] > 0.5:
        return "above chance, under 0.80"
    return "not shown by this predictor"


def official_split(path, table):
    import scipy.io
    m = scipy.io.loadmat(path, squeeze_me=True)
    split, missing = {}, 0
    have = set(table["shot"])
    for key, name in (("trainingShotlist", "train"), ("validationShotlist", "val"), ("testShotlist", "test")):
        for s in np.atleast_1d(m[key]).astype(int):
            if int(s) in have:
                split[int(s)] = name
            else:
                missing += 1
    return split, missing


def text(results):
    lines = ["Timing-only predictor (docs/preregistration_timing_model.md): features occ, gap, maxgap from timestamps only, "
             "time ending 40 ms before each shot's last slice; logistic regression, C chosen on validation, test "
             "evaluated once. Endpoint-aligned and retrospective; not an online warning system.", ""]
    for name, (r, fixed, c, val_auc, counts, note) in results.items():
        lines += [f"== {name} ({note})",
                  f"  {'PRIMARY ' if name == 'EAST' else ''}test AUC {sa.fmt(r)}: {reading(r)}",
                  f"  fixed statistics on the same test shots: occ {fixed['occ']:.3f}; minus gap {fixed['-gap']:.3f}",
                  f"  chosen C {c}, validation AUC {val_auc:.3f}; shots (disrupted, not): train {counts['train']}, "
                  f"val {counts['val']}, test {counts['test']}", ""]
    lines.append("Not claimed: that any published model used these features.")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    results = {}
    east = os.path.join(base, "20ja072_fig6_data")
    if os.path.exists(os.path.join(east, "EAST_for_nn.mat")):
        table = features(sa.load(os.path.join(east, "EAST_for_nn.mat")))
        split, missing = official_split(os.path.join(east, "EAST_data_partition.mat"), table)
        results["EAST"] = (*fit_and_test(table, split), f"official split; {missing} listed shots not in the data file")
    for key in ("d3d", "C_Mod"):
        path = os.path.join(base, "20ja072_fig4_data", f"{key}_for_nn.mat")
        if os.path.exists(path):
            table = features(sa.load(path))
            results[sa.MACHINES[key]] = (*fit_and_test(table, our_split(table)), "our seeded split, secondary")
    out = text(results)
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "timing_model.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
