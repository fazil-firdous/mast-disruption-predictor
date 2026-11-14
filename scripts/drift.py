"""Do MAST losses come at the end of a cross-shot trend? (docs/preregistration_drift.md, fixed before any number.)

Two populations of same-programme good repeats (E earlier, F later): A, the two kept shots before a lost shot (the
placebo triplets); B, repeats where no loss follows. For each saved run, each shot's mean score over the time both
shots share (later flat-top start to 30 ms before the earlier last window); win = F above E. Primary: P_A - P_B
for the full model over the v3 splits.

    python scripts/drift.py          # builds population B, scores both from the saved runs, writes the summary

Reads saved scores and cached shots only (no network, no training). Writes RESULTS_DIR/drift/. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, twin_check, v3_explore, v3_runs
from src import config, data, extras, features, notify, stats

HORIZON_S = config.HORIZON_MS / 1000  # the shared time ends this long before the earlier last window
MIN_WINDOWS = 5
MODELS = {"baseline": "full", "machine": "machine-side", "programme": "programme-only"}
PRIMARY = "baseline"
WORTH = 0.03
N_BOOT = 2000
SIGNALS = {"ne": "line-average density", "prad": "radiated power", "pnbi": "beam power", "lm": "locked-mode amplitude",
           "fgw": "Greenwald fraction"}
GAPS = ((1, 1, "1"), (2, 3, "2-3"), (4, 10 ** 9, "4+"))


def folder():
    return os.path.join(config.RESULTS_DIR, "drift")


# ------------------------------------------------------------------ the two populations

def pairs_a(triplets):
    """Population A from the placebo triplets: E = K2, F = K1, L = the lost shot."""
    return pd.DataFrame({"pop": "A", "E": triplets["k2"].astype(int), "F": triplets["k1"].astype(int),
                         "L": triplets["shot_id"].astype(int)})


def pairs_b(labels, load, reach=twin_check.REACH):
    """Population B: for every kept F, E = the nearest earlier kept same-programme shot (same campaign, lost shots
    skipped); kept only if the next same-programme shot after F within reach is not lost."""
    group = twin_check.groups(labels)
    quench = labels.set_index("shot_id")["t_quench"]
    campaign = labels.set_index("shot_id")["campaign"] if "campaign" in labels else pd.Series(dtype=object)
    cache = {}

    def prog(shot):
        if shot not in cache:
            cache[shot] = twin_check.programme(load(shot))
        return cache[shot]

    def same(a, b):
        stop = np.nanmin([quench.get(a, np.nan), quench.get(b, np.nan), np.inf])
        c = twin_check.compare(prog(a), prog(b), stop)
        return bool(c and c["rms"] < twin_check.SAME_RMS and c["peak_ratio"] > twin_check.SAME_PEAK)

    rows = []
    for shot in labels["shot_id"]:
        shot = int(shot)
        if group.get(shot) not in twin_check.KEPT or prog(shot) is None:
            continue
        earlier = None
        for other in range(shot - 1, shot - reach - 1, -1):
            if group.get(other) in twin_check.KEPT and campaign.get(other) == campaign.get(shot) and same(shot, other):
                earlier = other
                break
        if earlier is None:
            continue
        followed_by_loss = False
        for other in range(shot + 1, shot + reach + 1):
            g = group.get(other)
            if g not in twin_check.KEPT and g not in twin_check.LOST or campaign.get(other) != campaign.get(shot):
                continue
            if same(shot, other):
                followed_by_loss = g in twin_check.LOST
                break
        if not followed_by_loss:
            rows.append({"pop": "B", "E": earlier, "F": shot, "L": -1})
    return pd.DataFrame(rows, columns=["pop", "E", "F", "L"])


def with_times(pairs, starts, last):
    """pairs with the shared time (t0, t1): later flat-top start to HORIZON before the earlier last window (NaN, so
    the pair does not count, where either shot's start or last window is unknown)."""
    out = pairs.copy()
    out["t0"] = [float(np.maximum(starts.get(e, np.nan), starts.get(f, np.nan))) for e, f in zip(out["E"], out["F"])]
    out["t1"] = [min(last.get(e, np.nan), last.get(f, np.nan)) - HORIZON_S for e, f in zip(out["E"], out["F"])]
    out["gap"] = out["F"] - out["E"]
    return out


# ------------------------------------------------------------------ scoring from saved runs

def shot_mean(times, scores, t0, t1):
    """Mean score of the windows ending in [t0, t1], or NaN with fewer than MIN_WINDOWS."""
    ok = (times >= t0 - how_early.EPS) & (times <= t1 + how_early.EPS)
    return float(np.mean(scores[ok])) if ok.sum() >= MIN_WINDOWS else np.nan


def run_wins(scores, pairs):
    """One run: win per pair (NaN where a shot is not in this run's test set or lacks windows)."""
    by = {s: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float)) for s, g in scores.groupby("shot_id")}
    wins = np.full(len(pairs), np.nan)
    for i, row in enumerate(pairs.itertuples()):
        if row.E not in by or row.F not in by or not (np.isfinite(row.t0) and np.isfinite(row.t1)) or row.t1 <= row.t0:
            continue
        e, f = shot_mean(*by[row.E], row.t0, row.t1), shot_mean(*by[row.F], row.t0, row.t1)
        if np.isfinite(e) and np.isfinite(f):
            wins[i] = 1.0 if f > e else 0.5 if f == e else 0.0
    return wins


def step_agreement(scores, pairs, last, t_loss):
    """Within A, for one run: share of triplets where K2 -> K1 and K1 -> L move the same way (NaN if none)."""
    by = {s: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float)) for s, g in scores.groupby("shot_id")}
    same = []
    for row in pairs[pairs["pop"] == "A"].itertuples():
        if not all(s in by for s in (row.E, row.F, row.L)) or not np.isfinite(row.t1):
            continue
        e, f = shot_mean(*by[row.E], row.t0, row.t1), shot_mean(*by[row.F], row.t0, row.t1)
        t1 = min(last.get(row.F, np.nan), t_loss.get(row.L, np.nan)) - HORIZON_S
        f2, lost = shot_mean(*by[row.F], row.t0, t1), shot_mean(*by[row.L], row.t0, t1)
        if all(np.isfinite(v) for v in (e, f, f2, lost)) and f != e and lost != f2:
            same.append(float(np.sign(f - e) == np.sign(lost - f2)))
    return float(np.mean(same)) if same else np.nan


def model_folders():
    folders = v3_explore.run_folders()
    return {m: folders[m] for m in MODELS if m in folders}


def collect(pairs, last=None, t_loss=None):
    """{model: array seeds x pairs of wins}, {model: [seeds]}, and the full model's step agreement per seed."""
    wins, seeds, steps = {}, {}, []
    for model, base in model_folders().items():
        rows, got = [], []
        for seed, path in how_early.saved_runs(base):
            scores = pd.read_parquet(os.path.join(path, "test_scores.parquet"))
            rows.append(run_wins(scores, pairs))
            got.append(seed)
            if model == PRIMARY and last is not None and t_loss is not None:
                steps.append(step_agreement(scores, pairs, last, t_loss))
        if rows:
            wins[model], seeds[model] = np.vstack(rows), got
    return wins, seeds, steps


# ------------------------------------------------------------------ the comparison

def per_split(w, pop_mask):
    """P per split for one population (NaN where no pair of it was tested in that split)."""
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.where(pop_mask[None, :], w, np.nan), axis=1)


def over_splits(w, pairs):
    """(P_A range, P_B range, Delta range) over splits; each (mean, low, high)."""
    a, b = per_split(w, (pairs["pop"] == "A").to_numpy()), per_split(w, (pairs["pop"] == "B").to_numpy())
    ok = np.isfinite(a) & np.isfinite(b)
    return (stats.corrected_interval(a[ok]), stats.corrected_interval(b[ok]), stats.corrected_interval((a - b)[ok]),
            int(ok.sum()))


def pooled(w, pairs, seed=config.RANDOM_SEED, n_boot=N_BOOT):
    """Pooled over pairs (each pair's win averaged over the splits that tested it), groups of pairs linked through
    shared shots resampled together: (P_A, P_B, Delta), each (mean, low, high), and the pair counts."""
    mean = np.nanmean(w, axis=0) if np.isfinite(w).any() else np.full(w.shape[1], np.nan)
    keep = np.isfinite(mean)
    p, m = pairs[keep].reset_index(drop=True), mean[keep]
    is_a = (p["pop"] == "A").to_numpy()
    groups = v3_explore.shot_groups(p["E"].to_numpy(), p["F"].to_numpy())
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))]) if members else np.array([], int)
        a, b = m[idx][is_a[idx]], m[idx][~is_a[idx]]
        draws.append((a.mean() if len(a) else np.nan, b.mean() if len(b) else np.nan))
    draws = np.array(draws)
    point = (m[is_a].mean() if is_a.any() else np.nan, m[~is_a].mean() if (~is_a).any() else np.nan)

    def rng_of(values, centre):
        v = values[np.isfinite(values)]
        return (float(centre), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (centre, None, None)

    return (rng_of(draws[:, 0], point[0]), rng_of(draws[:, 1], point[1]), rng_of(draws[:, 0] - draws[:, 1], point[0] - point[1]),
            int(is_a.sum()), int((~is_a).sum()), mean)


def r(t, signed=False):
    if t is None or t[0] is None:
        return "-"
    fmt = "{:+.3f}" if signed else "{:.3f}"
    lo = "" if t[1] is None else f" ({fmt.format(t[1])} to {fmt.format(t[2])})"
    return fmt.format(t[0]) + lo


def signal_shares(pairs, load_signals, say=None):
    """Without a model: {signal: ((share in A, pairs), (share in B, pairs))} of pairs where F's mean over the shared
    time is higher. Each shot is loaded once and only its means are kept (the VPS has little memory)."""
    need = {}
    for i, row in enumerate(pairs.itertuples()):
        if np.isfinite(row.t0) and np.isfinite(row.t1) and row.t1 > row.t0:
            for shot in (row.E, row.F):
                need.setdefault(int(shot), []).append((i, row.t0, row.t1))
    means = {}  # (pair index, shot) -> {signal: mean}
    for k, (shot, wanted) in enumerate(sorted(need.items()), 1):
        sig = load_signals(shot)
        t = None if sig is None else sig["time"].to_numpy(dtype=float)
        for i, t0, t1 in wanted:
            got = {}
            for name in SIGNALS:
                if sig is None or name not in sig:
                    got[name] = np.nan
                    continue
                x = sig[name].to_numpy(dtype=float)
                ok = (t >= t0) & (t <= t1) & np.isfinite(x)
                got[name] = float(np.mean(x[ok])) if ok.sum() >= MIN_WINDOWS else np.nan
            means[(i, shot)] = got
        if say and (k % 2000 == 0 or k == len(need)):
            say(f"  [{k}/{len(need)}] shots")
    out = {}
    for name in SIGNALS:
        res = {"A": [], "B": []}
        for i, row in enumerate(pairs.itertuples()):
            e, f = means.get((i, int(row.E)), {}).get(name, np.nan), means.get((i, int(row.F)), {}).get(name, np.nan)
            if np.isfinite(e) and np.isfinite(f) and e != f:
                res[row.pop].append(float(f > e))
        out[name] = tuple((float(np.mean(v)), len(v)) if v else (np.nan, 0) for v in (res["A"], res["B"]))
    return out


def text(pairs, wins, seeds, steps, campaign, signals=None):
    a, b = pairs[pairs["pop"] == "A"], pairs[pairs["pop"] == "B"]
    lines = ["Drift study (docs/preregistration_drift.md): do MAST losses come at the end of a cross-shot trend?",
             "P = share of same-programme good-repeat pairs in which the later shot's mean score over the shared time is "
             "higher. A: the two kept shots before a lost shot. B: repeats where no loss follows.",
             f"Pairs: A {len(a):,}, B {len(b):,}. Median shot gap: A {a['gap'].median():.0f}, B {b['gap'].median():.0f}.", ""]
    if PRIMARY in wins:
        pa, pb, d, n = over_splits(wins[PRIMARY], pairs)
        pool = pooled(wins[PRIMARY], pairs)
        resolved = d[1] is not None and (d[1] > 0 or d[2] < 0)
        if resolved and d[0] > 0 and d[0] >= WORTH:
            reading = ("Resolved, positive and at least 0.03: the shots before a loss rise in score across repeats more "
                       "than ordinary repeats do.")
        elif resolved and d[0] > 0:
            reading = "Resolved and positive, but under 0.03."
        elif resolved:
            reading = "Resolved and negative: reported as such; no reading was fixed for it."
        elif pb[1] is not None and pb[1] > 0.5:
            reading = "Not resolved, and P_B is resolved above 0.5: ordinary drift between repeats; the twin test's fair reference is P_B."
        else:
            reading = "Not resolved, and P_B is not resolved above 0.5: no drift is shown by this statistic."
        lines += [f"PRIMARY (full model, {n} splits): Delta = P_A - P_B = {r(d, True)} over splits; pooled "
                  f"{r(pool[2], True)} ({pool[3]:,} A and {pool[4]:,} B pairs tested). {reading}",
                  f"- P_A {r(pa)}, P_B {r(pb)} over splits; pooled P_A {r(pool[0])}, P_B {r(pool[1])}", ""]
    lines.append("Each model, over splits (P_A | P_B | Delta):")
    for model, name in MODELS.items():
        if model in wins:
            pa, pb, d, n = over_splits(wins[model], pairs)
            lines.append(f"- {name} ({n} splits): {r(pa)} | {r(pb)} | {r(d, True)}")
    if PRIMARY in wins:
        mean = pooled(wins[PRIMARY], pairs)[5]
        frame = pairs.assign(win=mean, campaign=pairs["F"].map(campaign).fillna("Unknown"))
        lines += ["", "Full model, pooled, by shot gap (P_A | P_B; pairs):"]
        for lo, hi, label in GAPS:
            part = frame[(frame["gap"] >= lo) & (frame["gap"] <= hi) & frame["win"].notna()]
            pa_, pb_ = part[part["pop"] == "A"]["win"], part[part["pop"] == "B"]["win"]
            lines.append(f"- gap {label}: {pa_.mean():.3f} | {pb_.mean():.3f}; {len(pa_):,} | {len(pb_):,}")
        lines += ["", "Full model, pooled, by campaign (P_A | P_B; pairs):"]
        for c, part in frame[frame["win"].notna()].groupby("campaign"):
            pa_, pb_ = part[part["pop"] == "A"]["win"], part[part["pop"] == "B"]["win"]
            lines.append(f"- {c}: {pa_.mean() if len(pa_) else float('nan'):.3f} | {pb_.mean() if len(pb_) else float('nan'):.3f}; "
                         f"{len(pa_):,} | {len(pb_):,}")
    if signals:
        lines += ["", "Without any model: share of pairs where the later shot's mean is higher (A | B; pairs):"]
        for name, ((sa, na), (sb, nb)) in signals.items():
            lines.append(f"- {SIGNALS[name]}: {sa:.3f} | {sb:.3f}; {na:,} | {nb:,}")
    st = np.array([s for s in steps if np.isfinite(s)])
    if len(st):
        m, lo, hi = stats.corrected_interval(st)
        lines += ["", f"Within A, full model: K2 -> K1 and K1 -> L move the same way in {r((m, lo, hi))} of triplets "
                      f"(over {len(st)} splits; 0.5 = no link)."]
    lines += ["", "Campaigns of the pairs (A | B): " + "; ".join(
        f"{c} {int((a['F'].map(campaign) == c).sum())} | {int((b['F'].map(campaign) == c).sum())}"
        for c in sorted(set(campaign.values()) - {None}) if isinstance(c, str))]
    return "\n".join(lines)


def main():
    say = lambda line: print(line, flush=True)  # noqa: E731
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    triplets = pd.read_csv(os.path.join(config.RESULTS_DIR, "twin_differences", "triplets.csv"))
    load = lambda shot: extras.load_extras(int(shot), allow_download=False)  # noqa: E731
    say("building population B (repeats where no loss follows) ...")
    pairs = pd.concat([pairs_a(triplets), pairs_b(labels, load)], ignore_index=True)
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id", "t_end"])
    last = windows.groupby("shot_id")["t_end"].max().to_dict()
    shots = set(pairs["E"]) | set(pairs["F"]) | set(pairs.loc[pairs["L"] > 0, "L"])
    starts = {}
    for s in shots:
        v = v3_runs.flat_top_start(load(s))
        starts[s] = np.nan if v is None else v
    pairs = with_times(pairs, starts, last)
    say(f"pairs: A {int((pairs['pop'] == 'A').sum()):,}, B {int((pairs['pop'] == 'B').sum()):,}; scoring from the saved runs ...")
    t_loss = labels.set_index("shot_id")["t_quench"].to_dict()
    wins, seeds, steps = collect(pairs, last, t_loss)
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()

    def load_signals(shot):
        try:
            return features.prepare_signals(extras.with_extras(data.load_shot(int(shot)), int(shot)))
        except Exception:
            return None

    say("signal shares without a model ...")
    signals = signal_shares(pairs, load_signals, say)
    out = text(pairs, wins, seeds, steps, campaign, signals)
    os.makedirs(folder(), exist_ok=True)
    pairs.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    say(out)
    notify.send("Drift study: done\n\n" + out)


if __name__ == "__main__":
    main()
