"""Landmark module on the public Zhu et al. disruptive records (docs/preregistration_landmark_zhu.md, fixed before any
statistic): within disrupted records, do the 12 stored signals improve next-100 ms risk beyond elapsed time and the
recording schedule? Conditional on an eventual disruption; file-relative.

    python scripts/landmark_zhu.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/landmark_zhu.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit as au, sampling_audit as sa, timing_model as tm
from src import notify

WINDOW, STEP, HORIZON, LEAD, RESOLUTION = 0.300, 0.020, 0.100, 0.010, 1e-6
SEED, N_BOOT = 42, 1000
MODELS = {"C": ("elapsed",), "CS": ("elapsed", "timing"), "CSV": ("elapsed", "timing", "values"),
          "CV": ("elapsed", "values")}
GAINS = (("CSV", "CS", "PRIMARY: signals over clock + schedule"), ("CS", "C", "schedule over clock"),
         ("CV", "C", "signals over clock alone"))


def disrupted_shots(frame, signals):
    """{shot: (t, values, True, t_d)} for disrupted records; t_d from time + time_until_disrupt."""
    out = {}
    for shot, g in frame.groupby("shot", sort=True):
        tud = g["time_until_disrupt"].to_numpy(dtype=float)
        if not np.isfinite(tud).any():
            continue
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        td = float(np.nanmedian((g["time"] + g["time_until_disrupt"]).to_numpy(dtype=float)))
        out[int(shot)] = (t, g[signals].to_numpy(dtype=float), True, td)
    return out


def design(rows, parts):
    cols = []
    for part in parts:
        if part == "elapsed":
            cols.append(rows[["elapsed"]].to_numpy(dtype=float))
        elif part == "timing":
            cols.append(np.array(rows["timing"].tolist(), dtype=float))
        else:
            cols.append(np.array(rows["features"].tolist(), dtype=float))
    return np.hstack(cols)


def fit_predict(tr, te, parts):
    from sklearn.ensemble import HistGradientBoostingClassifier
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED, early_stopping=True,
                                           validation_fraction=0.1, n_iter_no_change=10)
    return model.fit(design(tr, parts), tr["label"].to_numpy()).predict_proba(design(te, parts))[:, 1]


def fit_predict_record_stopping(tr, te, parts, seed=SEED):
    """Sensitivity (diagnostic 2): early stopping on 10% of the training RECORDS instead of random decisions. Fit on the
    other 90% with up to 200 iterations, keep the iteration with the lowest validation log loss, predict at it."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    records = np.array(sorted(tr["shot"].unique()))
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(records, max(1, int(round(0.1 * len(records)))), replace=False).tolist())
    is_val = tr["shot"].isin(val_ids).to_numpy()
    fit_rows, val_rows = tr[~is_val], tr[is_val]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed, early_stopping=False)
    model.fit(design(fit_rows, parts), fit_rows["label"].to_numpy())
    y_val = val_rows["label"].to_numpy()
    losses = [log_loss(y_val, p[:, 1], labels=[0, 1]) for p in model.staged_predict_proba(design(val_rows, parts))]
    best = int(np.argmin(losses))
    for k, p in enumerate(model.staged_predict_proba(design(te, parts))):
        if k == best:
            return p[:, 1], best + 1
    return model.predict_proba(design(te, parts))[:, 1], len(losses)


def analyse(shots, split, n_boot=N_BOOT, record_stopping=False):
    rows = au.landmark_rows({s: v for s, v in shots.items() if split.get(s) in ("train", "test")}, WINDOW, STEP,
                            HORIZON, LEAD, RESOLUTION)
    late = int((rows["label"] == -1).sum())
    rows = rows[rows["label"] >= 0]
    part = rows["shot"].map(split)
    tr, te = rows[part == "train"], rows[part == "test"]
    y = te["label"].to_numpy()
    if record_stopping:
        got = {name: fit_predict_record_stopping(tr, te, parts) for name, parts in MODELS.items()}
        preds = {name: g[0] for name, g in got.items()}
        iters = {name: g[1] for name, g in got.items()}
    else:
        preds = {name: fit_predict(tr, te, parts) for name, parts in MODELS.items()}
        iters = {}
    shot = te["shot"].to_numpy()
    gains = {(a, b): au.cluster_gain((preds[b] - y) ** 2 - (preds[a] - y) ** 2, shot, n_boot=n_boot) for a, b, _ in GAINS}
    gap = np.array([v[3] - v[0][-1] for s, v in shots.items() if split.get(s) == "test"]) * 1e3
    return {"preds": preds, "y": y, "gains": gains, "late": late, "n_test_shots": te["shot"].nunique(),
            "n_train_shots": tr["shot"].nunique(), "n_dec": len(te), "n_pos": int(y.sum()), "gap_ms": gap,
            "iters": iters}


def lines_for(name, r, primary=False):
    out = [f"== {name}: test {r['n_test_shots']:,} disrupted shots (train {r['n_train_shots']:,}); test decisions "
           f"{r['n_dec']:,} ({r['n_pos']:,} positive); too-late decisions left out (train+test) {r['late']:,}; "
           f"t_d minus last slice, median {np.median(r['gap_ms']):.1f} ms"]
    for a, b, label in GAINS:
        g = r["gains"][(a, b)]
        verdict = ("improves next-100 ms risk" if g[1] > 0 else "no improvement shown (not shown absent)")
        tag = label if (primary or not label.startswith("PRIMARY")) else label.replace("PRIMARY: ", "")
        out.append(f"  {tag}: Brier gain {g[0]:+.5f} ({g[1]:+.5f} to {g[2]:+.5f}) -> {verdict}")
    out.append("  decision AUC: " + ", ".join(f"{m} {sa.auc(p, r['y']):.4f}" for m, p in r["preds"].items()))
    base = r["y"].mean()
    out.append(f"  Brier loss: constant (test positive share {base:.4f}) {np.mean((base - r['y']) ** 2):.5f}; " +
               ", ".join(f"{m} {np.mean((p - r['y']) ** 2):.5f}" for m, p in r["preds"].items()))
    if r.get("iters"):
        out.append("  iterations chosen on held-out training records: " + ", ".join(f"{m} {k}" for m, k in r["iters"].items()))
    out.append(f"  calibration: observed {r['y'].mean():.4f}; mean predicted " +
               ", ".join(f"{m} {p.mean():.4f}" for m, p in r["preds"].items()))
    return out


def run(base, n_boot=N_BOOT, record_stopping=False):
    lines = ["Landmark module on the public Zhu disruptive records (docs/preregistration_landmark_zhu.md). Conditional "
             "on an eventual disruption; file-relative; decisions every 20 ms, 300 ms window, positive if the "
             "disruption comes 10-100 ms later. Models: C clock, CS clock + schedule, CSV + signals, CV clock + signals.",
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
        table = frame.groupby("shot")["time_until_disrupt"].apply(lambda x: bool(np.isfinite(x).any())).rename(
            "disrupted").reset_index()
        if official:
            split, _ = tm.official_split(os.path.join(base, "20ja072_fig6_data", "EAST_data_partition.mat"), table)
        else:
            split = tm.our_split(table)
        lines += lines_for(name, analyse(disrupted_shots(frame, signals), split, n_boot, record_stopping),
                           primary=official)
        if name == "C-Mod":
            lines.append("  caveat: stored times carry sub-microsecond offsets; timing features quantised to 1 us.")
        lines.append("")
    lines += ["Power reference (synthetic, NOTES.md 2026-11-23): with about 200 shots the primary resolved a precursor "
              "lifting the decision AUC to about 0.78 every time and one lifting it to about 0.63 rarely.",
              "Not claimed: anything about non-disrupted shots, deployment, or the published models."]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    rec = "--record-stopping" in sys.argv
    out = run(sys.argv[1], record_stopping=rec)
    if rec:
        out = ("SENSITIVITY (diagnostic 2, after the result): early stopping on held-out training RECORDS.\n\n" + out)
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "landmark_zhu_recstop.txt" if rec else "landmark_zhu.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    for i in range(0, len(out), 3400):  # Telegram cuts long messages: send in parts
        notify.send(f"Landmark module, Zhu files ({i // 3400 + 1})\n\n" + out[i:i + 3400])


if __name__ == "__main__":
    main()
