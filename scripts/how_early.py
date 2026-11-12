"""How early? Part A of docs/preregistration_how_early.md: the twin test at fixed times before the loss, from the
scores the v3 runs saved. No training.

For a twin pair and a lead h, the landmark is the lost shot's quench time minus h. Each shot's score there is the
score of its latest window ending at or before the landmark and at most one window stride earlier; a shot with no
such window (it has ended, or has not started) has none, and the pair is left out at that lead. C(h) is the share
of eligible pairs in which the lost shot's score is the higher one (ties half). The quench time only places the
landmark for this look back; no model sees it.

    python scripts/how_early.py      # reads the saved scores of v3/, v3_explore/ and the chronological split

Writes RESULTS_DIR/how_early/ (pairs.csv, summary.txt). Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import v3_explore, v3_runs
from src import config, extras, server
from src.stats import corrected_interval

LEADS_MS = (10, 30, 50, 100, 200)
COHORT_LEADS = (10, 30, 50, 100)  # trends are read on pairs eligible at all of these
PRIMARY_LEAD = 50  # the first lead beyond the 30 ms the models were trained to warn at
MODELS = ["programme", "solenoid", "machine", "baseline"]
NAMES = {"programme": "programme-only", "solenoid": "solenoid-only", "machine": "machine-side", "baseline": "full"}
STRIDE_S = config.STRIDE_MS / 1000
EPS = 1e-6


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


def collect(load=None):
    """(random, temporal): one row per (kind, model, split, pair, lead) with win and the cohort flags. kind is
    'main' (lost shot against its twin) or 'placebo' (K1 against K2 at their lost shot's landmark)."""
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

    def of_run(path, recipe, seed):
        scores = pd.read_parquet(os.path.join(path, "test_scores.parquet"))
        ids, out = set(scores["shot_id"]), []
        for kind, pairs in (("main", main), ("placebo", placebo)):
            if pairs is not None:
                mine = pairs[pairs["shot_id"].isin(ids) & pairs["other"].isin(ids)]
                out.append(landmark_wins(scores, mine).assign(kind=kind, recipe=recipe, seed=seed))
        return out

    random, temporal = [], []
    folders = v3_explore.run_folders()
    for recipe in MODELS:
        base = folders.get(recipe, "")
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            path = os.path.join(base, entry)
            if server.result_saved(os.path.join(path, "metrics.json")) and entry.startswith("seed_"):
                random += of_run(path, recipe, int(entry[5:]))
        path = v3_explore.temporal_dir(recipe)
        if server.result_saved(os.path.join(path, "metrics.json")):
            temporal += of_run(path, recipe, 0)
    columns = ["shot_id", "other", "ref", "lead_ms", "win", "kind", "recipe", "seed"]

    def finish(parts):
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)
        frame["heating_early"] = [heating.get(key, "unknown") for key in zip(frame["shot_id"], frame["other"])]
        return with_cohorts(frame)

    return finish(random), finish(temporal)


def over_splits(frame, lead):
    """{model: (C, low, high, pairs per split)} at one lead, range over the splits."""
    table = v3_explore.c_table(frame[frame["lead_ms"] == lead]).set_index("recipe")
    return {m: (table.loc[m, "C"], table.loc[m, "low"], table.loc[m, "high"], table.loc[m, "pairs_per_split"])
            for m in MODELS if m in table.index}


def pooled(frame, lead, connected=False):
    """{model: (C, low, high, pairs)} at one lead, pooled over pairs with groups resampled together."""
    table = v3_explore.pooled_table(frame[frame["lead_ms"] == lead], "win", connected=connected).set_index("recipe")
    return {m: tuple(table.loc[m, ["C", "low", "high", "pairs"]]) for m in MODELS if m in table.index}


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


def main():
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
