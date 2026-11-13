"""Faster signals, step 0: the magnetic signals summarised over 5 ms instead of 20 ms
(docs/preregistration_fast_signals.md, fixed before any feature or model of this study).

    python scripts/fast_magnetics.py --features   # the 5 ms and growth features for every window, and a check that
                                                  # the same code with 20 ms reproduces the stored features
    python scripts/fast_magnetics.py --train      # the runs not done yet (skips finished), then the summary
    python scripts/fast_magnetics.py --summary    # the comparison in words, from the saved runs

Reads the window table and the cached shot data (no network). Writes PROCESSED_DIR/fast_magnetics.parquet and
RESULTS_DIR/fast_magnetics/. Nothing under v3/ or v3_explore/ is touched; outside every stamp.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, v3_explore
from scripts.paper_numbers import partial_auc
from src import config, dataset, experiment, extras, features, models, notify, server
from src.data import load_shot

MAG = ["lm", "mirnov", "mir_lf", "mir_tm", "mir_mf", "mir_hf"]
SHORT_MS = 5
SHORT_STATS = ("mean", "std", "slope", "delta")
FIVE = [f"{n}_{s}_5ms" for n in MAG for s in SHORT_STATS + ("rel",)]
GROWTH = [f"{n}_growth" for n in MAG]
REPLACED = [f"{n}_{s}" for n in MAG for s in SHORT_STATS] + [f"{n}_rel" for n in MAG]
RECIPES = {"mag20_5": "C, 20 + 5 ms (primary)", "mag5": "B, 5 ms instead of 20 ms", "growth": "G, growth"}
NAMES = {"baseline": "A, full", **RECIPES}
SEEDS = range(1, 21)
PRIMARY_LEAD = 50
WORTH = 0.03
EPS = features.EPS


def table_path():
    return os.path.join(config.PROCESSED_DIR, "fast_magnetics.parquet")


def folder():
    return os.path.join(config.RESULTS_DIR, "fast_magnetics")


def run_dir(recipe, seed=None):
    return os.path.join(folder(), "temporal", recipe) if seed is None else os.path.join(folder(), recipe, f"seed_{seed}")


def window_summaries(sig, t_end, t_start, window_ms):
    """{signal_stat: value} for the six magnetic signals over (t_end - window_ms, t_end], with the relative change
    against the shot's history before the 20 ms window, exactly as src/features.py computes the 20 ms ones."""
    time_ = sig["time"].to_numpy()
    past = time_ <= t_end + EPS
    short = past & (time_ > t_end - window_ms / 1000 + EPS)
    full = past & (time_ > t_end - config.WINDOW_MS / 1000 + EPS)
    history = past & ~full & (time_ >= t_start - EPS)
    out = {}
    for name in MAG:
        x = sig[name].to_numpy()
        stats = features._stats(time_[short], x[short])
        for stat in SHORT_STATS:
            out[f"{name}_{stat}"] = stats[stat]
        out[f"{name}_rel"] = features.relative_change(stats["mean"], x[history])
    return out


def shot_features(sig, t_ends, t_start):
    """One row per window end: the 5 ms summaries and the growth features."""
    rows = []
    for t_end in t_ends:
        short = window_summaries(sig, t_end, t_start, SHORT_MS)
        long = window_summaries(sig, t_end, t_start, config.WINDOW_MS)
        row = {f"{k}_5ms": v for k, v in short.items()}
        for name in MAG:
            a, b = short[f"{name}_mean"], long[f"{name}_mean"]
            row[f"{name}_growth"] = float(np.log10(a / b)) if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows, columns=FIVE + GROWTH)


def check_20ms(sig, windows):
    """The largest relative difference between the stored 20 ms features and the same code run here with 20 ms."""
    worst = 0.0
    for row in windows.itertuples():
        again = window_summaries(sig, row.t_end, row.t_start, config.WINDOW_MS)
        for key, value in again.items():
            stored = getattr(row, key)
            if np.isfinite(value) and np.isfinite(stored):
                worst = max(worst, abs(value - stored) / max(abs(stored), 1e-12))
    return worst


def build_features(load=None, say=print, check_every=200, windows=None):
    """The feature table for every window of the window table, and the 20 ms check on every check_every-th shot."""
    load = load or (lambda shot: features.prepare_signals(extras.with_extras(load_shot(shot), shot)))
    columns = ["shot_id", "t_end", "t_start"] + REPLACED
    if windows is None:
        windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=columns)
    windows = windows[columns]
    frames, worst, start = [], 0.0, time.time()
    groups = list(windows.groupby("shot_id", sort=True))
    for i, (shot, part) in enumerate(groups, 1):
        sig = load(int(shot))
        part = part.sort_values("t_end")
        feats = shot_features(sig, part["t_end"].to_numpy(), float(part["t_start"].iloc[0]))
        feats.insert(0, "t_end", part["t_end"].to_numpy())
        feats.insert(0, "shot_id", np.int32(shot))
        frames.append(feats)
        if i % check_every == 1:
            worst = max(worst, check_20ms(sig, part))
        if i % 1000 == 0 or i == len(groups):
            say(f"[{i}/{len(groups)}] shots, {(time.time() - start) / i:.2f} s per shot; 20 ms check so far: largest "
                f"relative difference {worst:.2e}")
    table = pd.concat(frames, ignore_index=True)
    table = table.astype({c: "float32" for c in FIVE + GROWTH})
    return table, worst


def with_fast(windows, table):
    """The window table with the 5 ms and growth columns joined on (shot_id, t_end)."""
    return windows.merge(table, on=["shot_id", "t_end"], how="left")


def columns_of(recipe):
    """The inputs of one model of this study."""
    base = v3_explore.recipe_columns("baseline")
    if recipe == "mag20_5":
        return base + FIVE
    if recipe == "mag5":
        return [c for c in base if c not in REPLACED] + FIVE
    if recipe == "growth":
        return base + GROWTH
    raise ValueError(recipe)


def fit(win, split_column, recipe, path, info, columns=None):
    """Train like the v3 baseline with this recipe's inputs (or the given columns), test once, save the test window
    scores (as scripts/v3_explore.py does). Returns the metrics."""
    train, val_all, test_all = experiment.split_frames(win, split_column)
    train = train[~train["candidate"]].reset_index(drop=True)
    val, test = experiment.judged(val_all, "clear"), experiment.judged(test_all, "clear")
    columns = experiment.usable(columns_of(recipe) if columns is None else columns, train)
    best = experiment.tune(config.SWEEP_MODEL, train, val, columns)
    os.makedirs(path, exist_ok=True)
    metrics = {"recipe": recipe, **info, "params": best["params"], "threshold": best["threshold"], "columns": columns}
    metrics["test"] = experiment.test_once(best, test, columns, path)
    metrics["test"]["pauc"] = partial_auc(pd.read_csv(os.path.join(path, "tradeoff_test.csv")))
    pd.DataFrame({"shot_id": test_all["shot_id"].astype("int32"), "t_end": test_all["t_end"].astype("float32"),
                  "score": np.asarray(models.scores(best["model"], test_all[columns]), dtype="float32")}).to_parquet(
        os.path.join(path, "test_scores.parquet"), index=False)
    experiment.write_json(metrics, os.path.join(path, "metrics.json"))  # last: marks the run as finished
    return metrics


def run(recipe, seed=None, tables=None):
    """One run: a random split by its seed (the v3 split), or the chronological split (seed None)."""
    windows, shots = tables or load_tables()
    if seed is None:
        win, column = dataset.labelled_windows(windows, shots), "split_temporal"
    else:
        split = dataset.random_split(shots, seed=seed)
        win, column = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots), "split"
    return fit(win, column, recipe, run_dir(recipe, seed), {"seed": 0 if seed is None else seed})


def load_tables():
    if not os.path.exists(table_path()):
        raise RuntimeError("no fast_magnetics.parquet: run python scripts/fast_magnetics.py --features first")
    windows, shots = experiment.load_tables()
    return with_fast(windows, pd.read_parquet(table_path())), shots


def train_all(say=print):
    jobs = [(r, s) for s in SEEDS for r in RECIPES] + [(r, None) for r in RECIPES]
    todo = [job for job in jobs if not server.result_saved(os.path.join(run_dir(*job), "metrics.json"))]
    say(f"{len(todo)} runs to do")
    tables = load_tables() if todo else None
    for i, (recipe, seed) in enumerate(todo, 1):
        start = time.time()
        m = run(recipe, seed, tables)
        say(f"[{i}/{len(todo)}] {recipe} {'chronological' if seed is None else f'seed {seed}'}: pAUC "
            f"{m['test']['pauc']:.3f}, {(time.time() - start) / 60:.1f} min")
    out = summary()
    say(out)
    notify.send("Faster signals, step 0: the runs are done\n\n" + out)


def collect(load=None):
    """(random, temporal) landmark wins for A (the saved v3 baseline) and the three new models."""
    main, placebo, heating = how_early.pair_frames(load)
    random, temporal = [], []
    sources = [("baseline", v3_explore.run_folders()["baseline"], v3_explore.temporal_dir("baseline"))]
    sources += [(r, os.path.join(folder(), r), run_dir(r)) for r in RECIPES]
    for name, base, temporal_path in sources:
        for seed, path in how_early.saved_runs(base):
            if seed in SEEDS:
                random += how_early.run_wins(path, name, seed, main, placebo)
        if server.result_saved(os.path.join(temporal_path, "metrics.json")):
            temporal += how_early.run_wins(temporal_path, name, 0, main, placebo)
    return how_early.finish(random, heating), how_early.finish(temporal, heating)


def plan_flags():
    path = os.path.join(config.RESULTS_DIR, "plan_twins", "pairs.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def difference_cells(frame, lead, a, b="baseline"):
    """'over splits ... | pooled ...' for model a minus model b at one lead, or '-'."""
    d = how_early.split_difference(frame, lead, a, b)
    p = v3_explore.pooled_delta(frame[frame["lead_ms"] == lead], a, b, "win") if len(frame) else None
    if not d or not p:
        return "-"
    return f"{how_early.rng(*d[:3], signed=True)} | pooled {how_early.rng(*p[:3], signed=True)}"


def text(random, temporal, flags=None):
    """Step 0 in words, in the order of the protocol."""
    lines = ["Faster signals, step 0 (docs/preregistration_fast_signals.md): magnetic summaries over 5 ms. Fixed-lead "
             "twin test, both shots at the same moment; common cohort of all twins; splits 1 to 20.", ""]
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    d = how_early.split_difference(main, PRIMARY_LEAD, "mag20_5", "baseline")
    p = v3_explore.pooled_delta(main[main["lead_ms"] == PRIMARY_LEAD], "mag20_5", "baseline", "win") if len(main) else None
    if d and p:
        resolved = d[1] is not None and d[1] > 0
        lines.append(f"PRIMARY: C(50 ms) of C (20 + 5 ms) minus A (full): over splits {how_early.rng(*d[:3], signed=True)}, "
                     f"{d[3]} splits; pooled {how_early.rng(*p[:3], signed=True)}, {p[3]:,} pairs. "
                     + ("Resolved and at least +0.03: the 5 ms summaries add early discrimination at 50 ms."
                        if resolved and d[0] >= WORTH else
                        "Resolved but under +0.03." if resolved else
                        "Not resolved: summarising these signals over 5 ms adds no measurable discrimination at 50 ms."))
    lines += ["", "C per lead, over splits (A full | C 20+5 | B 5 only | G growth):"]
    for lead in how_early.COHORT_LEADS:
        split = how_early.over_splits(main, lead) if len(main) else {}
        if "baseline" in split:
            cells = [f"{NAMES[m].split(',')[0]} {how_early.rng(*split[m][:3])}" for m in NAMES if m in split]
            lines.append(f"- {lead} ms ({split['baseline'][3]:.0f} pairs per split): " + "; ".join(cells))
    lines += ["", "Differences from A, per lead (over splits | pooled):"]
    for recipe in RECIPES:
        for lead in how_early.COHORT_LEADS:
            lines.append(f"- {NAMES[recipe]} minus A at {lead} ms: {difference_cells(main, lead, recipe)}")
    if flags is not None:
        lines += ["", "The same at 50 ms on subsets of the common cohort (C minus A):"]
        for column, label in (("plan", "plan twins"), ("shape_plan", "field-and-shape twins")):
            if column in flags:
                keep = flags.loc[flags[column].fillna(False).astype(bool), ["shot_id", "other"]]
                part = main.merge(keep, on=["shot_id", "other"])
                lines.append(f"- {label}: {difference_cells(part, PRIMARY_LEAD, 'mag20_5')}")
    placebo = random[random["kind"] == "placebo"] if len(random) else random
    if len(placebo):
        lines += ["", "Placebo (two good repeats at their lost shot's landmark), over splits (A | C):"]
        for lead in how_early.COHORT_LEADS:
            split = how_early.over_splits(placebo, lead)
            if "baseline" in split and "mag20_5" in split:
                lines.append(f"- {lead} ms: {how_early.rng(*split['baseline'][:3])} | {how_early.rng(*split['mag20_5'][:3])}")
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)] if len(temporal) else temporal
    lines += ["", "Chronological split (M5-M7 to M8-M9), common cohort, pooled (A | C | C minus A):"]
    for lead in how_early.COHORT_LEADS:
        pool = how_early.pooled(t_main, lead) if len(t_main) else {}
        if "baseline" in pool and "mag20_5" in pool:
            dd = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == lead], "mag20_5", "baseline", "win")
            lines.append(f"- {lead} ms ({int(pool['baseline'][3]):,} pairs): {how_early.rng(*pool['baseline'][:3])} | "
                         f"{how_early.rng(*pool['mag20_5'][:3])} | {how_early.rng(*dd[:3], signed=True) if dd else '-'}")
    lines += ["", "Usual metrics, minus A, paired over splits (pAUC; caught; false alarms):"]
    base = v3_explore.run_folders()["baseline"]
    for recipe in RECIPES:
        u = how_early.usual_difference(os.path.join(folder(), recipe), base, set(SEEDS))
        if u:
            lines.append(f"- {NAMES[recipe]}, {u[1]} splits: pAUC {how_early.rng(*u[0]['pauc'], signed=True)}; caught "
                         f"{how_early.rng(*u[0]['catch_rate'], signed=True)}; false alarms "
                         f"{how_early.rng(*u[0]['false_alarm_rate'], signed=True)}")
    return "\n".join(lines)


def summary(load=None):
    random, temporal = collect(load)
    out = text(random, temporal, plan_flags())
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    if args.features:
        table, worst = build_features(say=say)
        table.to_parquet(table_path(), index=False)
        say(f"wrote {table_path()}: {len(table):,} windows, {table['shot_id'].nunique():,} shots. 20 ms check: largest "
            f"relative difference {worst:.2e} " + ("(reproduces the stored features)" if worst < 1e-4 else
                                                     "(DOES NOT reproduce the stored features: stop and look)"))
    elif args.train:
        train_all(say)
    elif args.summary:
        say(summary())
    else:
        parser.error("give --features, --train or --summary")


if __name__ == "__main__":
    main()
