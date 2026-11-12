"""How early? Part A of docs/preregistration_how_early.md: the twin test at fixed times before the loss, from the
scores the v3 runs saved. No training.

For a twin pair and a lead h, the landmark is the lost shot's quench time minus h. Each shot's score there is the
score of its latest window ending at or before the landmark and at most one window stride earlier; a shot with no
such window (it has ended, or has not started) has none, and the pair is left out at that lead. C(h) is the share
of eligible pairs in which the lost shot's score is the higher one (ties half). The quench time only places the
landmark for this look back; no model sees it.

    python scripts/how_early.py              # Part A: the saved scores of v3/, v3_explore/, the chronological split
    python scripts/how_early.py --train-b    # Part B: train the 100 ms-horizon runs not done yet (skips finished)
    python scripts/how_early.py --part-b     # Part B in words

Part B changes one thing: the positive windows of training reach 100 ms before a recorded disruption instead of
30 ms. Full and machine-side models, random splits 1 to 20 and the chronological split.

Writes RESULTS_DIR/how_early/ (pairs.csv, summary.txt, h100/). Outside every stamp.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import v3_explore, v3_runs
from src import config, dataset, experiment, extras, notify, server
from src.stats import corrected_interval

LEADS_MS = (10, 30, 50, 100, 200)
COHORT_LEADS = (10, 30, 50, 100)  # trends are read on pairs eligible at all of these
PRIMARY_LEAD = 50  # the first lead beyond the 30 ms the models were trained to warn at
MODELS = ["programme", "solenoid", "machine", "baseline"]
NAMES = {"programme": "programme-only", "solenoid": "solenoid-only", "machine": "machine-side", "baseline": "full"}
STRIDE_S = config.STRIDE_MS / 1000
EPS = 1e-6
HORIZON_B_MS = 100  # Part B's training horizon
B_SEEDS = range(1, 21)
B_MODELS = ("baseline", "machine")
LONG = "@100"  # suffix of a Part B model's name
NAMES.update({m + LONG: f"{NAMES[m]}, 100 ms horizon" for m in B_MODELS})


def folder():
    return os.path.join(config.RESULTS_DIR, "how_early")


def score_at(times, scores, t):
    """The score of the latest window ending at or before t and at most one stride earlier, or NaN if there is none."""
    ok = (times <= t + EPS) & (times > t - STRIDE_S - EPS)
    if not ok.any():
        return np.nan
    return float(scores[np.argmax(np.where(ok, times, -np.inf))])


def landmark_wins(scores, pairs, leads_ms=LEADS_MS):
    """One row per (pair, lead) at which the pair is eligible: shot_id, other, ref, lead_ms, win. pairs needs
    shot_id (the lost shot, or K1 for a placebo pair), other, ref, t_loss and t_start (NaN if unknown)."""
    by_shot = {shot: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float))
               for shot, g in scores.groupby("shot_id")}
    rows = []
    for row in pairs.itertuples():
        lost, kept = by_shot.get(row.shot_id), by_shot.get(row.other)
        if lost is None or kept is None or not np.isfinite(row.t_loss):
            continue
        for lead in leads_ms:
            t = float(row.t_loss) - lead / 1000
            if np.isfinite(row.t_start) and t < row.t_start - EPS:
                continue  # the landmark is before the flat-top
            a, b = score_at(*lost, t), score_at(*kept, t)
            if np.isfinite(a) and np.isfinite(b):
                rows.append({"shot_id": row.shot_id, "other": row.other, "ref": row.ref, "lead_ms": lead,
                             "win": 1.0 if a > b else 0.5 if a == b else 0.0})
    return pd.DataFrame(rows, columns=["shot_id", "other", "ref", "lead_ms", "win"])


def with_cohorts(frame):
    """frame with two flags per row: cohort (its pair is eligible at every COHORT_LEADS lead in that run) and
    cohort_all (at every lead)."""
    keys = ["kind", "recipe", "seed", "shot_id", "other", "ref"]
    if len(frame) == 0:
        return frame.assign(cohort=False, cohort_all=False)
    leads = frame.groupby(keys)["lead_ms"].agg(set)
    flags = pd.DataFrame({"cohort": leads.map(lambda s: set(COHORT_LEADS) <= s),
                          "cohort_all": leads.map(lambda s: set(LEADS_MS) <= s)})
    return frame.join(flags, on=keys)


def pair_frames(load=None):
    """(main, placebo, heating): the twin pairs and the placebo pairs with their landmark times, and each pair's
    early beam heating ('same', 'different', 'unknown') where scripts/v3_explore.py has written it."""
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    starts = {}
    triplets_path = os.path.join(config.RESULTS_DIR, "twin_differences", "triplets.csv")
    placebo = (v3_explore.placebo_pairs(pd.read_csv(triplets_path), t_quench, starts, load)
               if os.path.exists(triplets_path) else None)
    for shot in twins["shot_id"]:
        if shot not in starts:
            starts[shot] = v3_runs.flat_top_start(load(shot))
    main = pd.DataFrame({"shot_id": twins["shot_id"], "other": twins["other"], "ref": twins["shot_id"],
                         "t_loss": [t_quench.get(s, np.nan) for s in twins["shot_id"]],
                         "t_start": [np.nan if starts[s] is None else starts[s] for s in twins["shot_id"]]})
    early_path = os.path.join(v3_explore.folder(), "twin_pairs.csv")
    heating = {}
    if os.path.exists(early_path):
        seen = pd.read_csv(early_path, usecols=["shot_id", "other", "heating_early"]).drop_duplicates(["shot_id", "other"])
        heating = dict(zip(zip(seen["shot_id"], seen["other"]), seen["heating_early"]))
    return main, placebo, heating


def run_wins(path, recipe, seed, main, placebo):
    """The landmark wins of one saved run, for the main pairs and the placebo pairs whose two shots it tested."""
    scores = pd.read_parquet(os.path.join(path, "test_scores.parquet"))
    ids, out = set(scores["shot_id"]), []
    for kind, pairs in (("main", main), ("placebo", placebo)):
        if pairs is not None:
            mine = pairs[pairs["shot_id"].isin(ids) & pairs["other"].isin(ids)]
            out.append(landmark_wins(scores, mine).assign(kind=kind, recipe=recipe, seed=seed))
    return out


def finish(parts, heating):
    columns = ["shot_id", "other", "ref", "lead_ms", "win", "kind", "recipe", "seed"]
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
    frame["heating_early"] = [heating.get(key, "unknown") for key in zip(frame["shot_id"], frame["other"])]
    return with_cohorts(frame)


def saved_runs(base):
    """[(seed, folder)] of the finished runs in one model's folder."""
    out = []
    for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        path = os.path.join(base, entry)
        if entry.startswith("seed_") and server.result_saved(os.path.join(path, "metrics.json")):
            out.append((int(entry[5:]), path))
    return out


def collect(load=None, models=MODELS, seeds=None, with_long=False):
    """(random, temporal): one row per (kind, model, split, pair, lead) with win and the cohort flags. kind is
    'main' (lost shot against its twin) or 'placebo' (K1 against K2 at their lost shot's landmark). seeds: only
    these splits. with_long: also the Part B runs, named with the suffix LONG."""
    main, placebo, heating = pair_frames(load)
    random, temporal = [], []
    folders = v3_explore.run_folders()
    sources = [(recipe, folders.get(recipe, ""), v3_explore.temporal_dir(recipe)) for recipe in models]
    if with_long:
        sources += [(recipe + LONG, os.path.join(b_folder(), recipe), b_dir(recipe)) for recipe in B_MODELS]
    for name, base, temporal_path in sources:
        for seed, path in saved_runs(base):
            if seeds is None or seed in seeds:
                random += run_wins(path, name, seed, main, placebo)
        if server.result_saved(os.path.join(temporal_path, "metrics.json")):
            temporal += run_wins(temporal_path, name, 0, main, placebo)
    return finish(random, heating), finish(temporal, heating)


def over_splits(frame, lead):
    """{model: (C, low, high, pairs per split)} at one lead, range over the splits."""
    table = v3_explore.c_table(frame[frame["lead_ms"] == lead]).set_index("recipe")
    return {m: (table.loc[m, "C"], table.loc[m, "low"], table.loc[m, "high"], table.loc[m, "pairs_per_split"])
            for m in table.index}


def pooled(frame, lead, connected=False):
    """{model: (C, low, high, pairs)} at one lead, pooled over pairs with groups resampled together."""
    table = v3_explore.pooled_table(frame[frame["lead_ms"] == lead], "win", connected=connected).set_index("recipe")
    return {m: tuple(table.loc[m, ["C", "low", "high", "pairs"]]) for m in table.index}


def split_difference(frame, lead, a="baseline", b="machine"):
    """(mean, low, high, splits) of C of a minus C of b at one lead, paired split by split, or None."""
    part = frame[(frame["lead_ms"] == lead) & frame["recipe"].isin([a, b])]
    per = part.groupby(["recipe", "seed"])["win"].mean().unstack("recipe")
    if a not in per or b not in per:
        return None
    diff = (per[a] - per[b]).dropna().to_numpy(dtype=float)
    return (*corrected_interval(diff), len(diff)) if len(diff) else None


def rng(mean, low, high, signed=False):
    f = "+.3f" if signed else ".3f"
    return format(mean, f) + (f" ({low:{f}} to {high:{f}})" if low is not None and np.isfinite(low) else "")


def lead_lines(frame, leads, temporal=False, connected=False):
    """One line per lead: every model's C and the full-minus-machine-side difference."""
    lines = []
    for lead in leads:
        cells = []
        if not temporal:
            split = over_splits(frame, lead)
            cells += [f"{NAMES[m]} {rng(*split[m][:3])}" for m in MODELS if m in split]
            d = split_difference(frame, lead)
            if d:
                cells.append(f"full minus machine-side {rng(*d[:3], signed=True)}")
            n = f"{split['baseline'][3]:.0f} pairs per split" if "baseline" in split else "no pairs"
        else:
            pool = pooled(frame, lead, connected)
            cells += [f"{NAMES[m]} {rng(*pool[m][:3])}" for m in MODELS if m in pool]
            n = f"{int(pool['baseline'][3]):,} pairs" if "baseline" in pool else "no pairs"
        d = v3_explore.pooled_delta(frame[frame["lead_ms"] == lead], "baseline", "machine", "win")
        if d:
            cells.append(f"pooled difference {rng(*d[:3], signed=True)}")
        lines.append(f"- {lead} ms ({n}): " + "; ".join(cells))
    return lines


def text(random, temporal):
    """Part A in words, in the order of the protocol."""
    main, placebo = random[random["kind"] == "main"], random[random["kind"] == "placebo"]
    cohort = main[main["cohort"].astype(bool)]
    lines = ["How early? Part A (docs/preregistration_how_early.md). The models were trained to warn 30 ms ahead.",
             "C(h) = share of eligible pairs in which the lost shot's score at the landmark, h before the loss, is the "
             "higher one. Ranges: over the 50 splits, corrected; 'pooled' resamples groups of pairs sharing a twin.", ""]
    d_split = split_difference(cohort, PRIMARY_LEAD)
    d_pool = v3_explore.pooled_delta(cohort[cohort["lead_ms"] == PRIMARY_LEAD], "baseline", "machine", "win")
    if d_split and d_pool:
        resolved = d_split[1] is not None and (d_split[1] > 0 or d_split[2] < 0)
        lines.append(f"PRIMARY: full minus machine-side at {PRIMARY_LEAD} ms on the common cohort: over splits "
                     f"{rng(*d_split[:3], signed=True)}, {d_split[3]} splits; pooled {rng(*d_pool[:3], signed=True)}, "
                     f"{d_pool[3]:,} pairs. Over training sets this is {'resolved' if resolved else 'not resolved'}.")
    lines += ["", "Common cohort (pairs eligible at 10, 30, 50 and 100 ms), over splits:"]
    lines += lead_lines(cohort, COHORT_LEADS)
    lines += ["", "Each lead's own eligible pairs, over splits:"] + lead_lines(main, LEADS_MS)
    lines += ["", "Pairs eligible at all five leads, over splits:"]
    lines += lead_lines(main[main["cohort_all"].astype(bool)], LEADS_MS)
    lines += ["", "Common cohort, pairs with the same early beam heating, over splits:"]
    lines += lead_lines(cohort[cohort["heating_early"] == "same"], COHORT_LEADS)
    lines += ["", "Placebo, two good repeats at their lost shot's landmark (each lead's own eligible pairs), over splits:"]
    lines += lead_lines(placebo, LEADS_MS) if len(placebo) else ["- no triplets"]
    t_main = temporal[temporal["kind"] == "main"] if len(temporal) else temporal
    lines += ["", "Chronological split (trained on M5-M7, tested on M8-M9), common cohort, pooled ranges only:"]
    lines += (lead_lines(t_main[t_main["cohort"].astype(bool)], COHORT_LEADS, temporal=True) if len(t_main)
              else ["- no runs"])
    return "\n".join(lines)


def b_folder():
    return os.path.join(folder(), f"h{HORIZON_B_MS}")


def b_dir(recipe, seed=None):
    """Where a Part B run goes: a random split by its seed, or the chronological split (seed None)."""
    return os.path.join(b_folder(), "temporal", recipe) if seed is None else os.path.join(b_folder(), recipe, f"seed_{seed}")


def run_b(recipe, seed=None, tables=None):
    """One Part B run: the v3 recipe with the positive windows reaching HORIZON_B_MS before a recorded disruption."""
    windows, shots = tables or experiment.load_tables()
    if seed is None:
        win, column = dataset.labelled_windows(windows, shots, horizon_ms=HORIZON_B_MS), "split_temporal"
    else:
        split = dataset.random_split(shots, seed=seed)
        win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots,
                                       horizon_ms=HORIZON_B_MS)
        column = "split"
    return v3_explore.fit(win, column, recipe, b_dir(recipe, seed),
                          {"seed": 0 if seed is None else seed, "horizon_ms": HORIZON_B_MS})


def train_b():
    """Every Part B run that has no finished result yet, then Part B in words to Telegram."""
    jobs = [(recipe, seed) for seed in B_SEEDS for recipe in B_MODELS] + [(recipe, None) for recipe in B_MODELS]
    todo = [job for job in jobs if not server.result_saved(os.path.join(b_dir(*job), "metrics.json"))]
    print(f"{len(todo)} runs to do", flush=True)
    tables = experiment.load_tables() if todo else None
    for i, (recipe, seed) in enumerate(todo, 1):
        start = time.time()
        m = run_b(recipe, seed, tables)
        print(f"[{i}/{len(todo)}] {recipe} {'chronological' if seed is None else f'seed {seed}'}: pAUC "
              f"{m['test']['pauc']:.3f}, {(time.time() - start) / 60:.1f} min", flush=True)
    out = part_b()
    print(out)
    notify.send("How early, Part B: the runs are done\n\n" + out)


USUAL = ("pauc", "catch_rate", "false_alarm_rate")


def usual_by_seed(base, seeds):
    """{seed: its test metrics} for the finished runs of these seeds."""
    out = {}
    for seed, path in saved_runs(base):
        if seed in seeds:
            with open(os.path.join(path, "metrics.json")) as f:
                out[seed] = json.load(f)["test"]
    return out


def usual_metrics(base, seeds):
    """Mean test pAUC, catch rate and false-alarm rate over the finished runs of these seeds, or None."""
    rows = list(usual_by_seed(base, seeds).values())
    if not rows:
        return None
    return tuple(float(np.mean([r[k] for r in rows])) for k in USUAL) + (len(rows),)


def usual_difference(base_a, base_b, seeds):
    """{metric: (mean, low, high)} of the runs in base_a minus those in base_b, paired split by split, and the
    number of splits both have; or None."""
    a, b = usual_by_seed(base_a, seeds), usual_by_seed(base_b, seeds)
    both = sorted(set(a) & set(b))
    if not both:
        return None
    return {k: corrected_interval(np.array([a[s][k] - b[s][k] for s in both], dtype=float)) for k in USUAL}, len(both)


def text_b(random, temporal):
    """Part B in words, in the order of the protocol."""
    full, machine, long_full, long_machine = "baseline", "machine", "baseline" + LONG, "machine" + LONG
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    lines = [f"How early? Part B (docs/preregistration_how_early.md): trained to warn {HORIZON_B_MS} ms ahead, against "
             "30 ms. Random splits 1 to 20, common cohort, identical pairs.", ""]
    d = split_difference(main, 100, long_full, full)
    dp = v3_explore.pooled_delta(main[main["lead_ms"] == 100], long_full, full, "win") if len(main) else None
    if d and dp:
        resolved = d[1] is not None and (d[1] > 0 or d[2] < 0)
        lines.append(f"PRIMARY: full model, C(100 ms) with the 100 ms horizon minus with the 30 ms horizon: over splits "
                     f"{rng(*d[:3], signed=True)}, {d[3]} splits; pooled {rng(*dp[:3], signed=True)}, {dp[3]:,} pairs. "
                     f"Over training sets this is {'resolved' if resolved else 'not resolved'}.")
    lines += ["", "Per lead, over splits (full 30 ms | full 100 ms | 100 minus 30 | machine-side 100 ms | full minus "
                  "machine-side, both 100 ms):"]
    for lead in COHORT_LEADS:
        split = over_splits(main, lead) if len(main) else {}
        if full not in split or long_full not in split:
            continue
        cells = [rng(*split[full][:3]), rng(*split[long_full][:3]),
                 rng(*split_difference(main, lead, long_full, full)[:3], signed=True)]
        if long_machine in split:
            cells += [rng(*split[long_machine][:3]), rng(*split_difference(main, lead, long_full, long_machine)[:3], signed=True)]
        lines.append(f"- {lead} ms ({split[full][3]:.0f} pairs per split): " + " | ".join(cells))
    lines += ["", "Machine-side model per lead, over splits (30 ms | 100 ms | 100 minus 30); added after the first result:"]
    for lead in COHORT_LEADS:
        split = over_splits(main, lead) if len(main) else {}
        if machine in split and long_machine in split:
            lines.append(f"- {lead} ms: {rng(*split[machine][:3])} | {rng(*split[long_machine][:3])} | "
                         f"{rng(*split_difference(main, lead, long_machine, machine)[:3], signed=True)}")
    lines += ["", "Usual metrics, mean over the splits (pAUC, caught, false alarms):"]
    folders = v3_explore.run_folders()
    for name, base in (("full, 30 ms", folders["baseline"]), ("full, 100 ms", os.path.join(b_folder(), "baseline")),
                       ("machine-side, 30 ms", folders["machine"]), ("machine-side, 100 ms", os.path.join(b_folder(), "machine"))):
        m = usual_metrics(base, set(B_SEEDS))
        if m:
            lines.append(f"- {name}: {m[0]:.3f}, {m[1]:.1%}, {m[2]:.1%} ({m[3]} splits)")
    for name, recipe in (("full", "baseline"), ("machine-side", "machine")):
        d = usual_difference(os.path.join(b_folder(), recipe), folders[recipe], set(B_SEEDS))
        if d:
            lines.append(f"- {name}, 100 ms minus 30 ms, paired over {d[1]} splits: pAUC {rng(*d[0]['pauc'], signed=True)}; "
                         f"caught {rng(*d[0]['catch_rate'], signed=True)}; false alarms "
                         f"{rng(*d[0]['false_alarm_rate'], signed=True)}")
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)] if len(temporal) else temporal
    lines += ["", "Chronological split, common cohort, pooled (full 30 ms | full 100 ms | 100 minus 30 | full minus "
                  "machine-side, both 100 ms):"]
    for lead in COHORT_LEADS:
        pool = pooled(t_main, lead) if len(t_main) else {}
        if full not in pool or long_full not in pool:
            continue
        cells = [rng(*pool[full][:3]), rng(*pool[long_full][:3])]
        for a, b in ((long_full, full), (long_full, long_machine)):
            dd = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == lead], a, b, "win")
            cells.append(rng(*dd[:3], signed=True) if dd else "-")
        lines.append(f"- {lead} ms ({int(pool[full][3]):,} pairs): " + " | ".join(cells))
    return "\n".join(lines)


def part_b(load=None):
    """Part B from the saved runs; also writes how_early/part_b.txt."""
    random, temporal = collect(load, models=list(B_MODELS), seeds=set(B_SEEDS), with_long=True)
    out = text_b(random, temporal)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "part_b.txt"), "w") as f:
        f.write(out + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-b", action="store_true", help="train the Part B runs that are not done yet")
    parser.add_argument("--part-b", action="store_true", help="Part B in words")
    args = parser.parse_args()
    if args.train_b:
        train_b()
        return
    if args.part_b:
        print(part_b())
        return
    random, temporal = collect()
    os.makedirs(folder(), exist_ok=True)
    random.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    temporal.to_csv(os.path.join(folder(), "pairs_temporal.csv"), index=False)
    out = text(random, temporal)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
