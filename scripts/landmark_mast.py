"""Landmark comparison on MAST lost discharges (docs/preregistration_landmark_mast.md, fixed before any statistic): do
the diagnostics improve next-100 ms risk beyond elapsed time, the programmed current and the solenoid current?

    python scripts/landmark_mast.py

Writes RESULTS_DIR/landmark_mast/summary.txt and per-split results. Outside every stamp.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit as au, sampling_audit as sa, v3_explore
from src import config, dataset, notify
from src.stats import corrected_interval

LEAD_S, HORIZON_S = 0.010, 0.100
SEEDS = range(1, 21)
SEED = 42
EPS = 1e-9
GAINS = (("F", "M", "PRIMARY: diagnostics over machine-side"), ("M", "P", "solenoid over programme + time"),
         ("P", "C", "programme over time"), ("F", "C", "diagnostics over time alone"))


def model_columns():
    return {"C": ["t_since_start"], "P": v3_explore.recipe_columns("programme"),
            "M": v3_explore.recipe_columns("machine"), "F": v3_explore.recipe_columns("baseline")}


def landmark_frame(win):
    """Lost discharges' windows with the landmark label y (1, 0) and kind ('recorded' or 'candidate'); too-late and
    post-event windows dropped. win: labelled windows (disrupted, candidate, t_quench, t_end)."""
    lost = (win["disrupted"] | win["candidate"]) & np.isfinite(win["t_quench"].astype(float))
    out = win[lost].copy()
    to_go = out["t_quench"].astype(float).to_numpy() - out["t_end"].to_numpy(dtype=float)
    keep = to_go > LEAD_S + EPS
    out, to_go = out[keep], to_go[keep]
    out["y"] = (to_go <= HORIZON_S + EPS).astype(int)
    out["kind"] = np.where(out["disrupted"], "recorded", "candidate")
    return out.reset_index(drop=True), int((~keep).sum())


def fit_predict(train, test, columns, seed=SEED):
    """Same learner for every model: boosting with early stopping on 10% of the training discharges (by discharge)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss
    shots = np.array(sorted(train["shot_id"].unique()))
    rng = np.random.default_rng(seed)
    held = set(rng.choice(shots, max(1, int(round(0.1 * len(shots)))), replace=False).tolist())
    is_held = train["shot_id"].isin(held).to_numpy()
    fit_rows, val_rows = train[~is_held], train[is_held]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed, early_stopping=False)
    model.fit(fit_rows[columns].to_numpy(dtype=float), fit_rows["y"].to_numpy())
    y_val = val_rows["y"].to_numpy()
    losses = [log_loss(y_val, p[:, 1], labels=[0, 1])
              for p in model.staged_predict_proba(val_rows[columns].to_numpy(dtype=float))]
    best = int(np.argmin(losses))
    for k, p in enumerate(model.staged_predict_proba(test[columns].to_numpy(dtype=float))):
        if k == best:
            return p[:, 1], best + 1
    return model.predict_proba(test[columns].to_numpy(dtype=float))[:, 1], len(losses)


def one_split(frame, part, cols):
    """Predictions of the four models on one split's test decisions; dict with y, shot, kind, preds, iters, base."""
    train, test = frame[part == "train"], frame[part == "test"]
    preds, iters = {}, {}
    for name, columns in cols.items():
        usable = [c for c in columns if c in train and train[c].notna().any()]
        preds[name], iters[name] = fit_predict(train, test, usable)
    return {"y": test["y"].to_numpy(), "shot": test["shot_id"].to_numpy(), "kind": test["kind"].to_numpy(),
            "preds": preds, "iters": iters, "base": float(train["y"].mean())}


def gains_of(r, mask=None):
    y = r["y"] if mask is None else r["y"][mask]
    p = r["preds"] if mask is None else {k: v[mask] for k, v in r["preds"].items()}
    return {(a, b): float(np.mean((p[b] - y) ** 2 - (p[a] - y) ** 2)) for a, b, _ in GAINS}


def split_row(seed, r):
    row = {"seed": seed, "decisions": len(r["y"]), "positives": int(r["y"].sum()), "base": r["base"]}
    for (a, b), g in gains_of(r).items():
        row[f"G_{a}_{b}"] = g
    for kind in ("recorded", "candidate"):
        m = r["kind"] == kind
        if m.any() and r["y"][m].min() != r["y"][m].max():
            row[f"G_F_M_{kind}"] = gains_of(r, m)[("F", "M")]
    for name, p in r["preds"].items():
        row[f"auc_{name}"] = sa.auc(p, r["y"])
        row[f"brier_{name}"] = float(np.mean((p - r["y"]) ** 2))
        row[f"iter_{name}"] = r["iters"][name]
    row["brier_const"] = float(np.mean((r["base"] - r["y"]) ** 2))
    return row


def summary(rows, chrono=None):
    f = pd.DataFrame(rows)
    lines = ["Landmark comparison on MAST lost discharges (docs/preregistration_landmark_mast.md). Retrospective, "
             "conditional on a lost discharge; windows of 20 ms every 5 ms; positive if the current quench is 10-100 ms "
             "ahead. C time, P programme + time, M machine-side, F all corrected inputs.", "",
             f"Random splits: {len(f)}; test decisions per split median {f['decisions'].median():,.0f} "
             f"(positive {f['positives'].median():,.0f})"]
    for a, b, label in GAINS:
        m, lo, hi = corrected_interval(f[f"G_{a}_{b}"].to_numpy())
        verdict = "improves" if lo is not None and lo > 0 else "no improvement shown (not shown absent)"
        lines.append(f"  {label}: G {m:+.5f} ({lo:+.5f} to {hi:+.5f} over splits) -> {verdict}")
    for kind in ("recorded", "candidate"):
        col = f"G_F_M_{kind}"
        if col in f and f[col].notna().sum() >= 2:
            m, lo, hi = corrected_interval(f[col].dropna().to_numpy())
            lines.append(f"  primary, {kind} discharges only: G {m:+.5f} ({lo:+.5f} to {hi:+.5f})")
    lines.append("  pooled decision AUC (mean over splits): " +
                 ", ".join(f"{n} {f[f'auc_{n}'].mean():.4f}" for n in "CPMF"))
    lines.append("  Brier loss (mean): constant " + f"{f['brier_const'].mean():.5f}, " +
                 ", ".join(f"{n} {f[f'brier_{n}'].mean():.5f}" for n in "CPMF"))
    if chrono is not None:
        lines += ["", "Chronological split (M5-M7 train, M8-M9 test), 95% intervals from 1,000 resamples of test discharges:"]
        for a, b, label in GAINS:
            d = (chrono["preds"][b] - chrono["y"]) ** 2 - (chrono["preds"][a] - chrono["y"]) ** 2
            g = au.cluster_gain(d, chrono["shot"])
            lines.append(f"  {label}: G {g[0]:+.5f} ({g[1]:+.5f} to {g[2]:+.5f})")
    lines += ["", "Power reference (synthetic, NOTES.md 2026-11-23): about 200 records resolved a precursor lifting the "
              "decision AUC to about 0.78 every time and one lifting it to about 0.63 rarely."]
    return "\n".join(lines)


def main():
    from scripts import saddle_correction as sc
    windows, shots = sc.load_tables()
    cols = model_columns()
    folder = os.path.join(config.RESULTS_DIR, "landmark_mast")
    os.makedirs(folder, exist_ok=True)
    rows = []
    for seed in SEEDS:
        split = dataset.random_split(shots, seed=seed)
        win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots)
        frame, late = landmark_frame(win)
        r = one_split(frame, frame["split"], cols)
        rows.append({**split_row(seed, r), "too_late": late, "lost_test": int(frame.loc[frame["split"] == "test",
                                                                                         "shot_id"].nunique())})
        pd.DataFrame(rows).to_csv(os.path.join(folder, "splits.csv"), index=False)
        print(f"seed {seed}: G(F over M) {rows[-1]['G_F_M']:+.5f}", flush=True)
    win = dataset.labelled_windows(windows, shots)
    frame, _ = landmark_frame(win)
    chrono = one_split(frame, frame["split_temporal"], cols)
    out = summary(rows, chrono)
    with open(os.path.join(folder, "summary.txt"), "w") as f:
        f.write(out + "\n")
    with open(os.path.join(folder, "columns.json"), "w") as f:
        json.dump(cols, f)
    print(out)
    for i in range(0, len(out), 3400):
        notify.send(f"Landmark MAST ({i // 3400 + 1})\n\n" + out[i:i + 3400])


if __name__ == "__main__":
    main()
