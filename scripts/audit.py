"""Audit toolkit, one command: is a disruption predictor reading the plasma, or the clock and the way the data were
recorded? Runs on any long table (CSV or parquet) with one row per time slice:

    shot, time, <signal columns...>, and either
      disrupted   (0/1, constant within a shot)          -> mixed mode (some shots disrupt, some do not)
      or --all-disrupt (every shot ends in a disruption)  -> all-disruptive mode (e.g. small tokamaks without ramp-down)
    optional t_disrupt (seconds, per shot); default: the shot's last slice.

    python scripts/audit.py table.csv                           # mixed mode
    python scripts/audit.py table.csv --all-disrupt --window-ms 20 --horizon-ms 30 --lead-ms 2
    python scripts/audit.py --demo                              # synthetic all-disruptive example, no data needed

Nothing leaves the machine it runs on: it prints and writes summary numbers only (audit_report.txt next to the input,
or in the current folder for --demo).

Mixed mode (validated on synthetic planted leaks, docs/preregistration_planted_leaks.md): T observation-process
ceiling, D clock ceiling, E endpoint dependence, each with the smallest difference it can detect at this size.
All-disruptive mode (not yet validated on planted leaks): windows near the disruption are positive, earlier ones
negative; it reports how far the clock alone and the sampling alone separate them, and how much of a values model's
discrimination survives when each positive window is compared only with negative windows of other shots at the same
elapsed time ("clock-matched concordance").
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import endpoint_zhu as ez, harmonisation as hm, planted_leaks as pl, sampling_audit as sa
from scripts import timing_model as tm

AUC_LINE, WORTH, N_BOOT, SEED = 0.80, 0.03, 1000, 42
MAX_PAIRS = 20_000_000
TIME_RESOLUTION = 1e-7  # seconds; for real data use the clock precision documented by the data owner
RESERVED = {"shot", "time", "disrupted", "t_disrupt"}


def load(path):
    frame = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = {"shot", "time"} - set(frame.columns)
    if missing:
        sys.exit(f"missing column(s): {', '.join(sorted(missing))}")
    signals = [c for c in frame.columns if c not in RESERVED and not c.startswith("prog_")
               and pd.api.types.is_numeric_dtype(frame[c])]
    if not signals:
        sys.exit("no numeric signal columns")
    return frame, signals


def shots_from(frame, signals, all_disrupt):
    """{shot: (t, values, disrupted, t_disrupt)} with times sorted."""
    out = {}
    for shot, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        if all_disrupt:
            flag = True
        else:
            flag = bool(g["disrupted"].astype(float).max() > 0)
        td = float(g["t_disrupt"].iloc[0]) if "t_disrupt" in g and np.isfinite(g["t_disrupt"].iloc[0]) else t[-1]
        out[shot] = (t, g[signals].to_numpy(dtype=float), flag, td)
    return out


def split_by_shot(ids, flags, seed=SEED):
    return tm.our_split(pd.DataFrame({"shot": list(ids), "disrupted": list(flags)}), seed=seed)


# ---------------------------------------------------------------- mixed mode


def mixed(shots):
    three = {k: (v[0], v[1], v[2]) for k, v in shots.items()}
    split = split_by_shot(three, [v[2] for v in three.values()])
    fired_t, auc_t = pl.check_t(three, split)
    fired_d, auc_d = pl.check_d(three, split)
    b = ez.build(three, split)
    mask = np.isfinite(b["s_m"])
    w_e, w_m = ez.win_matrix(b["s_case"], b["s_end"]), ez.win_matrix(b["s_case"], b["s_m"])
    d = ez.delta_range(w_e, w_m, mask, n_boot=N_BOOT)
    fired_e = abs(d[0]) >= WORTH and (d[1] > 0 or d[2] < 0)
    labels = np.r_[np.ones(len(b["cases"])), np.zeros(len(b["controls"]))].astype(bool)
    auc_v = sa.auc(np.r_[b["s_case"], b["s_end"]], labels)
    return [f"Mixed mode: {len(b['cases'])} test cases, {len(b['controls'])} test controls (seeded 60/20/20 split by shot).",
            f"  values-only model, ordinary AUC (controls at their end): {auc_v:.3f}",
            f"  T  timing alone (slices, age, gap in the 300 ms window): AUC {auc_t:.3f} -> "
            f"{'FLAG: the recording schedule alone separates the classes' if fired_t else 'no flag'}",
            f"  D  clock alone (elapsed time at the landmark): AUC {auc_d:.3f} -> "
            f"{'FLAG: the clock alone separates the classes' if fired_d else 'no flag'}",
            f"  E  control at its own end vs at the case's elapsed time: Delta {d[0]:+.4f} ({d[1]:+.4f} to {d[2]:+.4f}); "
            f"bootstrap interval half-widths {d[0] - d[1]:.4f} below, {d[2] - d[0]:.4f} above -> "
            f"{'FLAG: the score depends on where the control is evaluated' if fired_e else 'no flag'}"]


# ---------------------------------------------------------------- all-disruptive mode


def windows(shots, window, step, horizon, lead, resolution=None):
    """One row per window: shot, landmark, elapsed, positive, features, timing features."""
    rows = []
    for shot, (t, v, _, td) in shots.items():
        start = t[0] + window
        for landmark in np.arange(start, td - lead + 1e-9, step):
            keep = (t > landmark - window + 1e-9) & (t <= landmark + 1e-9)
            n = int(keep.sum())
            if not n:
                continue
            to_go = td - landmark
            rows.append({"shot": shot, "landmark": landmark, "elapsed": landmark - t[0], "positive": to_go <= horizon,
                         "features": hm.feature_row(v[keep]),
                         # quantised to the clock resolution: float noise in constant timing features must not
                         # become a signal (membership is decided with a 1e-9 s tolerance before this)
                         "timing": [n, quantise(landmark - t[keep][-1], resolution),
                                    quantise(tm.max_gap(t, landmark, span=window), resolution)]})
    return pd.DataFrame(rows)


def quantise(x, resolution=None):
    r = resolution or TIME_RESOLUTION
    return float(np.round(x / r) * r) if np.isfinite(x) else x


def logistic_scores(x_tr, y_tr, x_te):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    return model.fit(np.nan_to_num(x_tr), y_tr).predict_proba(np.nan_to_num(x_te))[:, 1]


def matched_concordance(score, positive, shot, elapsed, tol, n_boot=N_BOOT, seed=SEED):
    """Per positive window, the share of negative windows of OTHER shots it outranks (ties 1/2): over all of them (C_all)
    and over those within tol of its elapsed time (C_clock); averaged over positives. Bootstrap over shots.
    Returns (C_all, C_clock, Delta range, positives with a clock match, negatives used)."""
    rng = np.random.default_rng(seed)
    pi, ni = np.where(positive)[0], np.where(~positive)[0]
    if len(pi) * len(ni) > MAX_PAIRS:
        ni = np.sort(rng.choice(ni, MAX_PAIRS // max(len(pi), 1), replace=False))
    sp, sn = score[pi][:, None], score[ni][None, :]
    win = np.where(sp > sn, 1.0, np.where(sp == sn, 0.5, 0.0)).astype(np.float32)
    other = (shot[pi][:, None] != shot[ni][None, :])
    near = other & (np.abs(elapsed[pi][:, None] - elapsed[ni][None, :]) <= tol + 1e-9)
    masks = {"all": other.astype(np.float32), "clock": near.astype(np.float32)}
    wm = {k: win * m for k, m in masks.items()}
    ids, codes = np.unique(shot, return_inverse=True)
    cp, cn = codes[pi], codes[ni]

    def value(w_shot):
        a, b = w_shot[cp], w_shot[cn]
        out = {}
        for k in masks:
            den = masks[k] @ b
            ok = den > 0
            per = (wm[k] @ b)[ok] / den[ok]
            out[k] = float(np.sum(per * a[ok]) / np.sum(a[ok])) if ok.any() and a[ok].sum() > 0 else np.nan
        return out

    point = value(np.ones(len(ids)))
    draws = []
    for _ in range(n_boot):
        w_shot = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids)).astype(float)
        v = value(w_shot)
        draws.append(v["all"] - v["clock"])
    d = np.array([x for x in draws if np.isfinite(x)])
    delta = (point["all"] - point["clock"], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    matched = int((masks["clock"].sum(axis=1) > 0).sum())
    return point["all"], point["clock"], delta, matched, len(ni)


def all_disruptive_stats(shots, window, step, horizon, lead, tol, n_boot=N_BOOT):
    """The numbers behind the all-disruptive report, or None if not evaluable."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    w = windows(shots, window, step, horizon, lead)
    if w.empty or w["positive"].all() or not w["positive"].any():
        return None
    split = split_by_shot(list(shots), [True] * len(shots))
    part = w["shot"].map(split)
    tr, te = w[part == "train"], w[part == "test"]
    y_tr, y_te = tr["positive"].to_numpy(), te["positive"].to_numpy()
    x_tr, x_te = np.array(tr["features"].tolist(), dtype=float), np.array(te["features"].tolist(), dtype=float)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED).fit(x_tr, y_tr)
    s_val = model.predict_proba(x_te)[:, 1]
    s_clock = logistic_scores(tr[["elapsed"]].to_numpy(), y_tr, te[["elapsed"]].to_numpy())
    s_time = logistic_scores(np.array(tr["timing"].tolist(), dtype=float), y_tr, np.array(te["timing"].tolist(), dtype=float))
    c_all, c_clock, d, matched, n_neg = matched_concordance(s_val, y_te, te["shot"].to_numpy(), te["elapsed"].to_numpy(),
                                                             tol, n_boot=n_boot)
    return {"a_val": sa.auc(s_val, y_te), "a_clock": sa.auc(s_clock, y_te), "a_time": sa.auc(s_time, y_te),
            "c_all": c_all, "c_clock": c_clock, "delta": d, "matched": matched, "n_neg": n_neg,
            "n_test_shots": te["shot"].nunique(), "n_test": len(te), "n_pos": int(y_te.sum()),
            "flag_time": sa.auc(s_time, y_te) >= AUC_LINE,
            "flag_clock_match": abs(d[0]) >= WORTH and (d[1] > 0 or d[2] < 0)}


def all_disruptive(shots, window, step, horizon, lead, tol):
    r = all_disruptive_stats(shots, window, step, horizon, lead, tol)
    if r is None:
        return ["All-disruptive mode: not evaluable (no windows, or windows of only one kind)."]
    d = r["delta"]
    return [f"All-disruptive mode: {len(shots)} shots ({r['n_test_shots']} test); windows of {window * 1e3:.0f} ms every "
            f"{step * 1e3:.0f} ms; positive = ending within {horizon * 1e3:.0f} ms of the disruption (and at least "
            f"{lead * 1e3:.0f} ms before it); test windows {r['n_test']:,} ({r['n_pos']:,} positive).",
            f"  values-only model, ordinary window AUC: {r['a_val']:.3f}",
            f"  clock alone (elapsed time since the first slice): AUC {r['a_clock']:.3f} -> "
            f"{'FLAG: the window label can largely be read off the clock' if r['a_clock'] >= AUC_LINE else 'no flag'}",
            f"  sampling alone (slices, age, gap in the window): AUC {r['a_time']:.3f} -> "
            f"{'FLAG: the recording schedule alone separates near-disruption windows' if r['flag_time'] else 'no flag'}",
            f"  clock-matched concordance (negatives of other shots within {tol * 1e3:.0f} ms of the same elapsed time): "
            f"C_all {r['c_all']:.3f}, C_clock {r['c_clock']:.3f}; Delta = C_all - C_clock {d[0]:+.3f} ({d[1]:+.3f} to "
            f"{d[2]:+.3f}); positives with a match {r['matched']:,} of {r['n_pos']:,}; negatives used {r['n_neg']:,} -> "
            f"{'FLAG: part of the discrimination is the clock' if r['flag_clock_match'] else 'no flag'}",
            "  (all-disruptive mode is not yet validated on planted leaks; read it as a screen, not a verdict)"]


# ---------------------------------------------------------------- all-disruptive mode v2 (landmark primary)


def landmark_rows(shots, window, step, horizon, lead, resolution=None, prog=None):
    """Forward decision grid (docs/preregistration_landmark_mode.md). One row per decision before the event:
    shot, elapsed, label (1 positive, 0 negative, -1 too late), value features, timing features, programme."""
    rows = []
    prog = prog or {}
    for shot, (t, v, _, td) in shots.items():
        for landmark in np.arange(t[0] + window, td - 1e-9, step):
            keep = (t > landmark - window + 1e-9) & (t <= landmark + 1e-9)
            n = int(keep.sum())
            if not n:
                continue
            to_go = td - landmark
            label = -1 if to_go <= lead + 1e-9 else 1 if to_go <= horizon + 1e-9 else 0
            rows.append({"shot": shot, "elapsed": landmark - t[0], "label": label,
                         "features": hm.feature_row(v[keep]), "prog": list(prog.get(shot, [])),
                         "timing": [n, quantise(landmark - t[keep][-1], resolution),
                                    quantise(tm.max_gap(t, landmark, span=window), resolution)]})
    return pd.DataFrame(rows)


def cluster_gain(diff, shot, n_boot=N_BOOT, seed=SEED):
    """Mean of diff with a 95% range from resampling shots as clusters (equal weight per decision)."""
    ids, codes = np.unique(shot, return_inverse=True)
    sums, counts = np.bincount(codes, weights=diff, minlength=len(ids)), np.bincount(codes, minlength=len(ids))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        draws.append(float(w @ sums / (w @ counts)))
    return float(diff.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def landmark_stats(shots, window, step, horizon, lead, tol, prog=None, n_boot=N_BOOT, resolution=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = landmark_rows(shots, window, step, horizon, lead, resolution, prog)
    if rows.empty:
        return None
    n_late = int((rows["label"] == -1).sum())
    rows = rows[rows["label"] >= 0]
    if rows["label"].nunique() < 2:
        return None
    split = split_by_shot(list(shots), [True] * len(shots))
    part = rows["shot"].map(split)
    tr, te = rows[part == "train"], rows[part == "test"]

    def design(frame, values):
        base = np.c_[frame["elapsed"].to_numpy(dtype=float), np.array(frame["prog"].tolist(), dtype=float).reshape(len(frame), -1)]
        return np.c_[base, np.array(frame["features"].tolist(), dtype=float)] if values else base

    y_tr, y_te = tr["label"].to_numpy(), te["label"].to_numpy()
    fit = lambda x: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED).fit(x, y_tr)  # noqa: E731
    p_b = fit(design(tr, False)).predict_proba(design(te, False))[:, 1]
    p_a = fit(design(tr, True)).predict_proba(design(te, True))[:, 1]
    diff = (p_b - y_te) ** 2 - (p_a - y_te) ** 2
    gain = cluster_gain(diff, te["shot"].to_numpy(), n_boot=n_boot)
    s_time = logistic_scores(np.array(tr["timing"].tolist(), dtype=float), y_tr, np.array(te["timing"].tolist(), dtype=float))
    c_all, c_clock, d, matched, n_neg = matched_concordance(p_a, y_te.astype(bool), te["shot"].to_numpy(),
                                                             te["elapsed"].to_numpy(), tol, n_boot=n_boot)
    return {"gain": gain, "flag_gain": gain[1] > 0, "brier_b": float(np.mean((p_b - y_te) ** 2)),
            "brier_a": float(np.mean((p_a - y_te) ** 2)), "mean_b": float(p_b.mean()), "mean_a": float(p_a.mean()),
            "observed": float(y_te.mean()), "n_dec": len(te), "n_pos": int(y_te.sum()), "n_late": n_late,
            "n_test_shots": te["shot"].nunique(), "a_clock": sa.auc(te["elapsed"].to_numpy(dtype=float), y_te),
            "a_time": sa.auc(s_time, y_te), "a_aug": sa.auc(p_a, y_te), "a_base": sa.auc(p_b, y_te),
            "c_all": c_all, "c_clock": c_clock, "delta": d, "matched": matched,
            "flag_clock_match": abs(d[0]) >= WORTH and (d[1] > 0 or d[2] < 0),
            "has_prog": bool(len(rows["prog"].iloc[0]))}


def landmark_text(r, window, step, horizon, lead, tol):
    if r is None:
        return ["Landmark analysis: not evaluable (no decisions, or decisions of one kind only)."]
    g, d = r["gain"], r["delta"]
    base = "elapsed time" + (" + programme columns" if r["has_prog"] else " only (no programme columns given)")
    return [f"PRIMARY, fixed-horizon landmark comparison: decisions every {step * 1e3:.0f} ms from {window * 1e3:.0f} ms "
            f"after the first slice; positive = event in the next {lead * 1e3:.0f}-{horizon * 1e3:.0f} ms. Test: "
            f"{r['n_test_shots']} shots, {r['n_dec']:,} decisions ({r['n_pos']:,} positive); too-late decisions left out "
            f"(all shots) {r['n_late']:,}.",
            f"  baseline: {base}. Brier baseline {r['brier_b']:.5f}, with diagnostic values {r['brier_a']:.5f}; "
            f"gain {g[0]:+.5f} ({g[1]:+.5f} to {g[2]:+.5f}, shots resampled) -> "
            f"{'the diagnostic values improve the next-' + format(horizon * 1e3, '.0f') + ' ms risk beyond the baseline' if r['flag_gain'] else 'no improvement shown (not shown absent)'}",
            f"  calibration in the large: observed {r['observed']:.4f}; mean predicted baseline {r['mean_b']:.4f}, "
            f"with values {r['mean_a']:.4f}",
            f"  secondary: decision AUC baseline {r['a_base']:.3f}, with values {r['a_aug']:.3f}; elapsed time alone "
            f"{r['a_clock']:.3f}; sampling alone {r['a_time']:.3f}; clock-matched C_all {r['c_all']:.3f}, C_clock "
            f"{r['c_clock']:.3f}, Delta {d[0]:+.3f} ({d[1]:+.3f} to {d[2]:+.3f}) (sensitivity to the comparison rule, "
            f"not a leakage measure)"]


# ---------------------------------------------------------------- demo data


def demo_table(seed=0, n=150):
    """All-disruptive synthetic shots: similar durations (so the clock is informative), a weak genuine precursor on sig0,
    and a slow drift on sig1 (like a density or current trend) through which a values model can read the clock."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        duration = rng.uniform(0.18, 0.24)
        t = np.round(np.arange(0.0, duration, 0.001), 6)
        x = pl.ar1(rng, len(t))[:, :4]
        x[:, 0] += 0.5 * np.clip(1 - (t[-1] - t) / 0.03, 0, None)
        x[:, 1] += 10.0 * t
        for i in range(len(t)):
            rows.append({"shot": 50000 + k, "time": t[i], **{f"sig{j}": x[i, j] for j in range(4)}})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("path", nargs="?")
    p.add_argument("--all-disrupt", action="store_true")
    p.add_argument("--window-ms", type=float, default=300.0)
    p.add_argument("--step-ms", type=float, default=10.0)
    p.add_argument("--horizon-ms", type=float, default=30.0)
    p.add_argument("--lead-ms", type=float, default=2.0)
    p.add_argument("--tolerance-ms", type=float, default=10.0)
    p.add_argument("--demo", action="store_true")
    a = p.parse_args(argv)
    if a.demo:
        frame, signals, a.all_disrupt = demo_table(), [f"sig{j}" for j in range(4)], True
        a.window_ms, a.step_ms = 20.0, 5.0
        out_dir = "."
    elif a.path:
        frame, signals = load(a.path)
        out_dir = os.path.dirname(os.path.abspath(a.path))
    else:
        p.error("give a table, or --demo")
    if not a.all_disrupt and "disrupted" not in frame.columns:
        sys.exit("mixed mode needs a 'disrupted' column (0/1 per shot); use --all-disrupt if every shot disrupts")
    shots = shots_from(frame, signals, a.all_disrupt)
    head = [f"Audit of {a.path or 'demo data'}: {len(shots):,} shots, {len(signals)} signals "
            f"({', '.join(signals[:8])}{', ...' if len(signals) > 8 else ''})."]
    if a.all_disrupt:
        prog_cols = [c for c in frame.columns if c.startswith("prog_")]
        prog = frame.groupby("shot")[prog_cols].first().apply(list, axis=1).to_dict() if prog_cols else None
        args = (a.window_ms / 1e3, a.step_ms / 1e3, a.horizon_ms / 1e3, a.lead_ms / 1e3, a.tolerance_ms / 1e3)
        body = landmark_text(landmark_stats(shots, *args, prog=prog), *args) + [""] + all_disruptive(shots, *args)
    else:
        body = mixed(shots)
    out = "\n".join(head + body)
    with open(os.path.join(out_dir, "audit_report.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    return out


if __name__ == "__main__":
    main()
