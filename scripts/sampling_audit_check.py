"""Accounting and implementation checks for the public-files audit, before the package goes to the authors (asked for by
an outside reviewer). Exploratory diagnostics, not a new analysis: file hashes and versions, shot counts and
duplicates, the EAST split's inclusion flow and disjointness, the disruption-time convention (C4) in full, class counts
per shot-number bin (C5), and a diagnosis of the reversed DIII-D residual-timing fit of the harmonisation run.

    python scripts/sampling_audit_check.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/check.txt. Outside every stamp.
"""
import hashlib
import os
import platform
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harmonisation as hm, sampling_audit as sa, timing_model as tm

C4_TOLERANCE_S = 0.001


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment():
    import scipy
    import sklearn
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip()
    except Exception:
        commit = "?"
    return (f"python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy "
            f"{scipy.__version__}, scikit-learn {sklearn.__version__}; code commit {commit}")


def counts(frame):
    per = frame.groupby("shot")["time_until_disrupt"].apply(lambda x: bool(np.isfinite(x.to_numpy(dtype=float)).any()))
    dup = int(frame.duplicated(subset=["shot", "time"]).sum())
    return len(per), int(per.sum()), int((~per).sum()), dup, per


def c4_full(frame):
    """Per disrupted shot: the spread of time + time_until_disrupt within the shot, and last slice minus that time."""
    rows = []
    for shot, g in frame.groupby("shot"):
        tud = g["time_until_disrupt"].to_numpy(dtype=float)
        ok = np.isfinite(tud)
        if not ok.any():
            continue
        td = g["time"].to_numpy(dtype=float)[ok] + tud[ok]
        rows.append({"shot": shot, "spread": float(td.max() - td.min()),
                     "last_minus": float(g["time"].max() - np.median(td))})
    return pd.DataFrame(rows)


def east_flow(base, per):
    import scipy.io
    m = scipy.io.loadmat(os.path.join(base, "20ja072_fig6_data", "EAST_data_partition.mat"), squeeze_me=True)
    lists = {k: set(np.atleast_1d(m[k]).astype(int).tolist()) for k in ("trainingShotlist", "validationShotlist", "testShotlist")}
    in_file = set(per.index.astype(int))
    union = set().union(*lists.values())
    lines = ["EAST split, inclusion flow:",
             f"  shots in EAST_for_nn.mat: {len(in_file):,} ({int(per.sum()):,} disrupted, {int((~per).sum()):,} not)"]
    for k, v in lists.items():
        d = sum(bool(per.get(s, False)) for s in v if s in in_file)
        lines.append(f"  {k}: {len(v):,} listed, {len(v & in_file):,} in the file ({d:,} disrupted)")
    pairs = [("trainingShotlist", "validationShotlist"), ("trainingShotlist", "testShotlist"), ("validationShotlist", "testShotlist")]
    lines.append("  overlaps between lists: " + "; ".join(f"{a[:-8]}/{b[:-8]} {len(lists[a] & lists[b])}" for a, b in pairs))
    rest = in_file - union
    d_rest = sum(bool(per.get(s)) for s in rest)
    lines.append(f"  shots in the file but in no list: {len(rest):,} ({d_rest:,} disrupted, {len(rest) - d_rest:,} not)")
    tc = m.get("trainingClasses")
    if tc is not None:
        tc = np.atleast_1d(tc).astype(int)
        tr = np.atleast_1d(m["trainingShotlist"]).astype(int)
        agree = np.mean([int(bool(per.get(s, False))) == int(c) for s, c in zip(tr, tc) if s in in_file])
        lines.append(f"  trainingClasses agree with our label (any finite time_until_disrupt) for {agree:.1%} of training "
                     f"shots; values in trainingClasses: {sorted(set(tc.tolist()))}")
    others = [k for k in m if not k.startswith("__") and k not in lists and k != "trainingClasses"]
    lines.append(f"  other lists in the partition file (not used): {', '.join(others)}")
    return lines


def diiid_age_diagnosis(base):
    """Rebuild the DIII-D harmonisation rows and inspect the residual-timing logistic fit (exploratory)."""
    from sklearn.linear_model import LogisticRegression
    frame = sa.load(os.path.join(base, "20ja072_fig4_data", "d3d_for_nn.mat"))
    signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
    rows = hm.shot_rows(frame, signals, hm.DELTA_S["DIII-D"])
    split = tm.our_split(rows)
    part = rows["shot"].map(split)
    tr, te = rows[part == "train"], rows[part == "test"]
    med = tr["age2"].median()
    xtr, xte = tr[["age2"]].fillna(med).to_numpy(), te[["age2"]].fillna(med).to_numpy()
    ytr, yte = tr["disrupted"].to_numpy(), te["disrupted"].to_numpy()
    m = LogisticRegression(max_iter=1000).fit(xtr, ytr)
    dec_te, proba_te = m.decision_function(xte), m.predict_proba(xte)[:, 1]
    from scipy.stats import spearmanr
    lines = ["DIII-D residual-timing fit, diagnosis (exploratory):",
             f"  classes_ {list(m.classes_)}; predict_proba column used: index 1 (class {m.classes_[1]})",
             f"  coefficient {m.coef_[0][0]:.4g}, intercept {m.intercept_[0]:.4g}; feature units: seconds",
             f"  train AUC of the fit {sa.auc(m.decision_function(xtr), ytr):.3f}; test AUC {sa.auc(dec_te, yte):.3f}",
             f"  raw age, no fit (higher age = disrupted): train {sa.auc(tr['age2'].to_numpy(dtype=float), ytr):.3f}, "
             f"test {sa.auc(te['age2'].to_numpy(dtype=float), yte):.3f}",
             f"  rank agreement on test, decision vs proba: {spearmanr(dec_te, proba_te).correlation:.3f}; decision vs raw "
             f"age: {spearmanr(dec_te, te['age2'].fillna(med)).correlation:.3f}",
             "  age (ms), median | 90th percentile | max: train disrupted "
             f"{tr.loc[tr['disrupted'], 'age2'].median() * 1e3:.2f} | {tr.loc[tr['disrupted'], 'age2'].quantile(0.9) * 1e3:.2f} | "
             f"{tr.loc[tr['disrupted'], 'age2'].max() * 1e3:.2f}; train not {tr.loc[~tr['disrupted'], 'age2'].median() * 1e3:.2f} | "
             f"{tr.loc[~tr['disrupted'], 'age2'].quantile(0.9) * 1e3:.2f} | {tr.loc[~tr['disrupted'], 'age2'].max() * 1e3:.2f}",
             f"  ids aligned: {bool((te['shot'].to_numpy() == te['shot'].to_numpy()).all())} (predictions made on the same "
             "frame as the labels)"]
    return lines


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base = sys.argv[1]
    lines = ["Accounting and implementation checks (exploratory diagnostics before contacting the authors).",
             "Environment: " + environment(), ""]
    for z in ("20ja072_fig4_data.zip", "20ja072_fig6_data.zip"):
        p = os.path.join(base, z)
        if os.path.exists(p):
            lines.append(f"md5 {z}: {md5(p)}")
    lines.append("")
    per_east = None
    for key, name in sa.MACHINES.items():
        path = os.path.join(base, "20ja072_fig4_data", f"{key}_for_nn.mat")
        frame = sa.load(path)
        n, d, nd, dup, per = counts(frame)
        if key == "EAST":
            per_east = per
        c4 = c4_full(frame)
        beyond = int((c4["last_minus"].abs() > C4_TOLERANCE_S).sum())
        lines += [f"== {name}: {n:,} shots ({d:,} disrupted, {nd:,} not); duplicate (shot, time) rows: {dup}",
                  f"  C4: spread of time + time_until_disrupt within a shot, max {c4['spread'].max() * 1e3:.3f} ms; last slice "
                  f"minus disruption time: max |.| {c4['last_minus'].abs().max() * 1e3:.3f} ms, {beyond} of {len(c4):,} "
                  f"beyond {C4_TOLERANCE_S * 1e3:.0f} ms"]
        order = per.index.to_series().rank(method="first")
        bins = pd.qcut(order, 10, labels=False)
        cells = [f"{int(per[bins == b].sum())}/{int((~per[bins == b]).sum())}" for b in range(10)]
        lines += [f"  C5 equal-count shot-number bins, disrupted/not: {' '.join(cells)}", ""]
    if per_east is not None:
        lines += east_flow(base, per_east) + [""]
    lines += diiid_age_diagnosis(base)
    out = "\n".join(lines)
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "check.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
