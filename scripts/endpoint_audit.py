"""Endpoint module of the audit toolkit (docs/preregistration_endpoint.md, fixed before any score): the matched
comparison with the kept twin scored at the case's programme time (arm P) versus near its own natural end (arm E), for
a frozen predictor, the same pairs and the same case windows. Saved runs only; nothing is trained.

    python scripts/endpoint_audit.py

Writes RESULTS_DIR/endpoint/summary.txt and pairs.parquet. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, paper2_corrected as p2, saddle_correction as sc, twin_differences
from src import config, notify
from src.stats import corrected_interval

MODELS = (sc.RECIPE, "machine", "programme")
NAMES = {sc.RECIPE: "full, corrected (A_c)", "machine": "machine-side", "programme": "programme + time"}
LEADS_MS = (10, 30, 50, 100)
PRIMARY_LEAD = 10
WORTH = 0.03
EPS = 1e-9


def twin_end(t_quench, scores_of_twin):
    """The twin's natural end: its detected current quench if finite, else the end of its last scored window."""
    if t_quench is not None and np.isfinite(t_quench):
        return float(t_quench)
    times = scores_of_twin[0]
    return float(times.max()) if len(times) else np.nan


def latest(times, scores, t):
    """(score, window end) of the window how_early.score_at selects at t, or (NaN, NaN)."""
    ok = (times <= t + EPS) & (times > t - how_early.STRIDE_S - EPS)
    if not ok.any():
        return np.nan, np.nan
    i = int(np.argmax(np.where(ok, times, -np.inf)))
    return float(scores[i]), float(times[i])


def win(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0


def endpoint_wins(scores, pairs, t_quench, leads_ms=LEADS_MS):
    """One row per (pair, lead) eligible in both arms: shot_id, other, lead_ms, win_p, win_e, twin score at both
    points, the time between the two twin points. pairs needs shot_id, other, t_loss, t_start."""
    by_shot = {s: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float))
               for s, g in scores.groupby("shot_id")}
    rows = []
    for row in pairs.itertuples():
        lost, kept = by_shot.get(row.shot_id), by_shot.get(row.other)
        if lost is None or kept is None or not np.isfinite(row.t_loss):
            continue
        end = twin_end(t_quench.get(row.other), kept)
        if not np.isfinite(end):
            continue
        for lead in leads_ms:
            t = float(row.t_loss) - lead / 1000
            if np.isfinite(row.t_start) and t < row.t_start - EPS:
                continue
            te = end - lead / 1000
            a, sel_l = latest(*lost, t)
            (kp, sel_p), (ke, sel_e) = latest(*kept, t), latest(*kept, te)
            if np.isfinite(a) and np.isfinite(kp) and np.isfinite(ke):
                rows.append({"shot_id": row.shot_id, "other": row.other, "lead_ms": lead, "win_p": win(a, kp),
                             "win_e": win(a, ke), "case": a, "twin_p": kp, "twin_e": ke, "gap_s": te - t, "t_p": t,
                             "t_e": te, "sel_l": sel_l, "sel_p": sel_p, "sel_e": sel_e})
    return pd.DataFrame(rows, columns=["shot_id", "other", "lead_ms", "win_p", "win_e", "case", "twin_p", "twin_e",
                                       "gap_s", "t_p", "t_e", "sel_l", "sel_p", "sel_e"])


def collect(load=None):
    main, _, _ = how_early.pair_frames(load)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    twins = pd.read_csv(how_early.v3_runs.TWINS)
    ending = dict(zip(zip(twins["shot_id"], twins["other"]), twins.get("other_group", pd.Series(dtype=str))))
    folders = p2.folders()
    frames = []
    for model in MODELS:
        for seed, path in how_early.saved_runs(folders[model]):
            scores = pd.read_parquet(os.path.join(path, "test_scores.parquet"))
            ids = set(scores["shot_id"])
            mine = main[main["shot_id"].isin(ids) & main["other"].isin(ids)]
            w = endpoint_wins(scores, mine, t_quench)
            frames.append(w.assign(model=model, seed=seed))
    out = pd.concat(frames, ignore_index=True)
    out["ending"] = [ending.get(k, "") for k in zip(out["shot_id"], out["other"])]
    return out


def over_splits(frame, model, lead):
    """(Delta, low, high, splits), C_P and C_E over splits, for one model and lead."""
    part = frame[(frame["model"] == model) & (frame["lead_ms"] == lead)]
    per = part.groupby("seed")[["win_p", "win_e"]].mean()
    if not len(per):
        return None
    d = (per["win_e"] - per["win_p"]).to_numpy(dtype=float)
    mean, low, high = corrected_interval(d)
    return (mean, low, high, len(d)), float(per["win_p"].mean()), float(per["win_e"].mean())


def pooled(frame, model, lead):
    part = frame[(frame["model"] == model) & (frame["lead_ms"] == lead)]
    per = part.groupby(["shot_id", "other"])[["win_p", "win_e"]].mean().reset_index()
    if not len(per):
        return None
    return twin_differences.cluster_range((per["win_e"] - per["win_p"]).to_numpy(dtype=float), per["other"].to_numpy(),
                                          np.random.default_rng(config.RANDOM_SEED), 2000)


def reading(d):
    mean, low, high = d[:3]
    if low is not None and (low > 0 or high < 0) and abs(mean) >= WORTH:
        direction = ("the twin scores higher near its own end" if mean < 0 else
                     "the twin scores lower near its own end")
        return f"the matched comparison depends materially on where the negative window is taken ({direction})"
    return "no material endpoint sensitivity is shown for this predictor and these pairs (not shown equivalent)"


def fmt(r):
    return f"{r[0]:+.3f} ({r[1]:+.3f} to {r[2]:+.3f})" if r and r[1] is not None else "n/a"


def text(frame):
    lines = ["Endpoint module (docs/preregistration_endpoint.md): the case at t_q - h in both arms; the twin at the "
             "same programme time (P) or at its own end minus h (E). Same pairs in both arms; saved runs only.", ""]
    primary = over_splits(frame, sc.RECIPE, PRIMARY_LEAD)
    if primary:
        d, cp, ce = primary
        lines.append(f"PRIMARY, A_c at {PRIMARY_LEAD} ms: C_P {cp:.3f}, C_E {ce:.3f}; Delta = C_E - C_P {fmt(d)} over "
                     f"{d[3]} splits; pooled {fmt(pooled(frame, sc.RECIPE, PRIMARY_LEAD))}.")
        lines.append(f"Reading: {reading(d)}. A difference changes the plasma phase and possibly the measurements as "
                     "well as the end-of-record context; it does not by itself show that future information entered "
                     "the model.")
    lines += ["", "By model and lead (C_P, C_E, Delta over splits | pooled; pairs per split):"]
    for model in MODELS:
        for lead in LEADS_MS:
            r = over_splits(frame, model, lead)
            if r:
                d, cp, ce = r
                n = frame[(frame["model"] == model) & (frame["lead_ms"] == lead)].groupby("seed").size().mean()
                lines.append(f"- {NAMES[model]}, {lead} ms: {cp:.3f}, {ce:.3f}, {fmt(d)} | {fmt(pooled(frame, model, lead))}"
                             f"; {n:.0f}")
    part = frame[(frame["model"] == sc.RECIPE) & (frame["lead_ms"] == PRIMARY_LEAD)]
    lines += ["", f"A_c at {PRIMARY_LEAD} ms by the twin's ending:"]
    for ending, g in part.groupby("ending"):
        r = over_splits(g.assign(model=sc.RECIPE), sc.RECIPE, PRIMARY_LEAD)
        if r:
            lines.append(f"- {ending or 'unknown'}: Delta {fmt(r[0])}, {g[['shot_id', 'other']].drop_duplicates().shape[0]:,} pairs")
    if len(part):
        lines += ["", f"Twin's mean score at the two points (A_c, {PRIMARY_LEAD} ms): P {part['twin_p'].mean():.4f}, "
                      f"E {part['twin_e'].mean():.4f}; median time from P to E {part['gap_s'].median() * 1e3:.0f} ms "
                      f"(quartiles {part['gap_s'].quantile(0.25) * 1e3:.0f} to {part['gap_s'].quantile(0.75) * 1e3:.0f})."]
    return "\n".join(lines)


def main():
    frame = collect()
    folder = os.path.join(config.RESULTS_DIR, "endpoint")
    os.makedirs(folder, exist_ok=True)
    frame.to_parquet(os.path.join(folder, "pairs.parquet"), index=False)
    out = text(frame)
    with open(os.path.join(folder, "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Endpoint module\n\n" + out[:3400])


if __name__ == "__main__":
    main()
