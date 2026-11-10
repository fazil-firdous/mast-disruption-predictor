"""v3, exploratory: what lets a model tell twins apart? NOT pre-registered. docs/preregistration_v3.md lists the tests
that were; every number here is a description for the paper, with ranges but no verdicts.

1. A ladder of inputs. The pre-registered machine-only model told twins apart 60% of the time, although twins share
   the programme. Two more models, trained like the v3 baseline on the same 50 splits, take that apart:
   - programme: only the programmed current and the time since start. Twins share both, so C should be 0.5;
   - solenoid: only the solenoid current and the time since start. Feedback control drives the solenoid, so it
     reacts to the plasma.
2. Breakdowns from the saved test scores of the v3 runs, without training anything: C by the lost shot's kind
   (recorded or candidate), by the kept twin's ending, by how long before the loss the scores stop (lead time), and
   by whether the two shots also had the same beam heating (twins are matched on the programmed current only).

    python scripts/v3_explore.py --train       # the ladder runs not done yet, one after another (skips finished)
    python scripts/v3_explore.py --summary     # the ladder and the breakdowns in words

Writes into RESULTS_DIR/v3_explore/. Outside every stamp: it never changes or reruns a v3 result.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import twin_check, twin_differences, v3_runs
from scripts.paper_numbers import partial_auc
from src import config, dataset, experiment, features, models, notify, server
from src.stats import corrected_interval

LADDER = {"programme": "sol", "solenoid": "ref"}  # recipe -> the machine signal it leaves out
SEEDS = v3_runs.SEEDS
LEADS_MS = (config.MIN_WARNING_MS, 30, 50, 100, 200)  # the first is the pre-registered one
MODELS = ["programme", "solenoid", "machine", "baseline", "evidence"]  # the order of the ladder in the summary
HEATING_RMS = twin_check.SAME_RMS  # same beam heating: beam power within 10% of its peak (RMS), like the programme
NO_BEAM_SHARE = 0.05  # beam power under 5% of the largest seen counts as no beam
MIN_WINDOWS = 5  # windows both shots must share to compare their beam power
SEEN = {"programme": "programmed current only", "solenoid": "solenoid current only",
        "machine": "programme + solenoid (pre-registered)", "baseline": "every signal (pre-registered)",
        "evidence": "every signal, evidence labels (pre-registered)"}


def folder():
    return os.path.join(config.RESULTS_DIR, "v3_explore")


def out_dir(recipe, seed):
    return os.path.join(folder(), recipe, f"seed_{seed}")


def ladder_columns(recipe):
    """The machine-only inputs without one of the two machine signals (time since start stays)."""
    drop = LADDER[recipe]
    return [c for c in features.model_features("machine") if not (c.startswith(f"{drop}_") or c == f"has_{drop}")]


def run(recipe, seed, tables=None):
    """One ladder run: the v3 baseline recipe (candidates left out of training, judged on clear endings) on the E10
    split of this seed, with only this recipe's inputs. Saves the test window scores like scripts/v3_runs.py."""
    windows, shots = tables or experiment.load_tables()
    split = dataset.random_split(shots, seed=seed)
    win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots)
    train, val_all, test_all = experiment.split_frames(win, "split")
    train = train[~train["candidate"]].reset_index(drop=True)
    val, test = experiment.judged(val_all, "clear"), experiment.judged(test_all, "clear")
    columns = experiment.usable(ladder_columns(recipe), train)
    best = experiment.tune(config.SWEEP_MODEL, train, val, columns)
    path = out_dir(recipe, seed)
    os.makedirs(path, exist_ok=True)
    metrics = {"recipe": recipe, "seed": seed, "params": best["params"], "threshold": best["threshold"],
               "columns": columns}
    metrics["test"] = experiment.test_once(best, test, columns, path)
    metrics["test"]["pauc"] = partial_auc(pd.read_csv(os.path.join(path, "tradeoff_test.csv")))
    pd.DataFrame({"shot_id": test_all["shot_id"].astype("int32"), "t_end": test_all["t_end"].astype("float32"),
                  "score": np.asarray(models.scores(best["model"], test_all[columns]), dtype="float32")}).to_parquet(
        os.path.join(path, "test_scores.parquet"), index=False)
    experiment.write_json(metrics, os.path.join(path, "metrics.json"))  # last: marks the run as finished
    return metrics


def train_all():
    """Every ladder run that has no finished result yet, seed by seed; then the summary to Telegram."""
    todo = [(recipe, seed) for seed in range(1, SEEDS + 1) for recipe in LADDER
            if not server.result_saved(os.path.join(out_dir(recipe, seed), "metrics.json"))]
    print(f"{len(todo)} ladder runs to do", flush=True)
    tables = experiment.load_tables() if todo else None
    for i, (recipe, seed) in enumerate(todo, 1):
        start = time.time()
        m = run(recipe, seed, tables)
        print(f"[{i}/{len(todo)}] {recipe} seed {seed}: pAUC {m['test']['pauc']:.3f}, "
              f"{(time.time() - start) / 60:.1f} min", flush=True)
    out = text(summary())
    print(out)
    notify.send("v3 explore: the ladder runs are done\n\n" + out)


def pair_wins(scores, twins, t_quench, t_start, leads_ms=LEADS_MS):
    """One row per (twin pair with both shots in scores, lead time): win is 1 if the lost shot's highest window
    score is above its twin's, 0.5 for a tie, 0 otherwise. The scores of both shots are taken from the flat-top
    start of the lost shot to lead_ms before its quench. At lead MIN_WARNING_MS this is v3_runs.concordance, pair
    by pair (same comparisons on the same saved values)."""
    by_shot = {shot: (g["t_end"].to_numpy(), g["score"].to_numpy()) for shot, g in scores.groupby("shot_id")}
    rows = []
    for row in twins.itertuples():
        lost, kept = by_shot.get(row.shot_id), by_shot.get(row.other)
        tq = t_quench.get(row.shot_id, np.nan)
        if lost is None or kept is None or not np.isfinite(tq):
            continue
        start = t_start.get(row.shot_id)
        t_from = -np.inf if start is None else float(start)
        for lead in leads_ms:
            t_to = float(tq) - lead / 1000
            a = lost[1][(lost[0] >= t_from) & (lost[0] <= t_to)]
            b = kept[1][(kept[0] >= t_from) & (kept[0] <= t_to)]
            if len(a) == 0 or len(b) == 0:
                continue
            win = 1.0 if a.max() > b.max() else 0.5 if a.max() == b.max() else 0.0
            rows.append({"shot_id": row.shot_id, "other": row.other, "lead_ms": lead, "win": win})
    return pd.DataFrame(rows, columns=["shot_id", "other", "lead_ms", "win"])


def run_folders():
    """{model: folder of its runs}: the pre-registered recipes in v3/, the ladder in v3_explore/."""
    return {**{recipe: os.path.join(v3_runs.folder(), recipe) for recipe in v3_runs.RECIPES},
            **{recipe: os.path.join(folder(), recipe) for recipe in LADDER}}


def beam_power(shot_ids):
    """{shot: (time bins, beam power window means)} from the window table, or {} if there is none."""
    path = os.path.join(config.PROCESSED_DIR, "windows.parquet")
    if not os.path.exists(path):
        return {}
    windows = pd.read_parquet(path, columns=["shot_id", "t_end", "pnbi_mean"])
    return twin_differences.by_shot(windows[windows["shot_id"].isin(shot_ids)], signals=["pnbi"])


def heating(pair_ids, beams, t_quench, t_start):
    """{(lost shot, twin): 'same', 'different' or 'unknown'}: did the two shots have the same beam heating? Twins
    are matched on the programmed current only. Compared over the windows both have from the lost shot's flat-top
    start to MIN_WARNING_MS before its loss: 'same' if the RMS difference of the beam power is under HEATING_RMS of
    the larger peak, or if neither shot has a beam (both under NO_BEAM_SHARE of the largest beam power seen)."""
    peaks = [np.nanmax(np.abs(v)) for _, v in beams.values() if np.isfinite(v).any()]
    top = max(peaks) if peaks else 0.0
    out = {}
    for lost, twin in pair_ids:
        out[(lost, twin)] = "unknown"
        tq = t_quench.get(lost, np.nan)
        if lost not in beams or twin not in beams or not np.isfinite(tq):
            continue
        (ba, va), (bb, vb) = beams[lost], beams[twin]
        common = np.intersect1d(ba, bb)
        start = t_start.get(lost)
        t = common * twin_differences.STRIDE_S
        common = common[(t >= (-np.inf if start is None else start)) & (t <= tq - config.MIN_WARNING_MS / 1000)]
        a, b = va[np.searchsorted(ba, common), 0], vb[np.searchsorted(bb, common), 0]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < MIN_WINDOWS:
            continue
        peak = max(np.abs(a[ok]).max(), np.abs(b[ok]).max())
        if peak <= NO_BEAM_SHARE * top:
            out[(lost, twin)] = "same"
        else:
            rms = float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / peak)
            out[(lost, twin)] = "same" if rms < HEATING_RMS else "different"
    return out


def collect(load=None, beams=None):
    """(pairs, skill): pairs has one row per (model, split, twin pair, lead time) with the pair's groups, gap and
    beam heating; skill one row per finished run with its test pAUC, catch and false-alarm rates."""
    from src import extras
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    info = twins.set_index(["shot_id", "other"])[["group", "other_group", "gap", "rms"]]
    starts, pairs, skill = {}, [], []
    for recipe, base in run_folders().items():
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            metrics_path = os.path.join(base, entry, "metrics.json")
            scores_path = os.path.join(base, entry, "test_scores.parquet")
            if not (server.result_saved(metrics_path) and os.path.exists(scores_path)):
                continue
            with open(metrics_path) as f:
                m = json.load(f)
            skill.append({"recipe": recipe, "seed": m["seed"], "pauc": m["test"]["pauc"],
                          "catch_rate": m["test"]["catch_rate"], "false_alarm_rate": m["test"]["false_alarm_rate"]})
            scores = pd.read_parquet(scores_path)
            ids = set(scores["shot_id"])
            mine = twins[twins["shot_id"].isin(ids) & twins["other"].isin(ids)]
            for shot in mine["shot_id"]:
                if shot not in starts:
                    starts[shot] = v3_runs.flat_top_start(load(shot))
            wins = pair_wins(scores, mine, t_quench, starts)
            pairs.append(wins.join(info, on=["shot_id", "other"]).assign(recipe=recipe, seed=m["seed"]))
    columns = ["shot_id", "other", "lead_ms", "win", "group", "other_group", "gap", "rms", "recipe", "seed", "heating"]
    skill = pd.DataFrame(skill, columns=["recipe", "seed", "pauc", "catch_rate", "false_alarm_rate"])
    if not pairs:
        return pd.DataFrame(columns=columns), skill
    pairs = pd.concat(pairs, ignore_index=True)
    ids = list(pairs[["shot_id", "other"]].drop_duplicates().itertuples(index=False, name=None))
    beams = beam_power({shot for pair in ids for shot in pair}) if beams is None else beams
    same = heating(ids, beams, t_quench, starts)
    pairs["heating"] = [same[pair] for pair in zip(pairs["shot_id"], pairs["other"])]
    return pairs, skill


def c_table(pairs, by=None):
    """C (mean over the splits of each split's share of twin pairs won) with its corrected 95% range, per model and,
    if given, per value of the column `by`. Columns: recipe, [by], C, low, high, splits, pairs_per_split."""
    keys = ["recipe"] + ([by] if by else [])
    per_split = pairs.groupby(keys + ["seed"])["win"].agg(["mean", "size"]).reset_index()
    rows = []
    for key, part in per_split.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        mean, low, high = corrected_interval(part["mean"].to_numpy(dtype=float))
        rows.append({**dict(zip(keys, key)), "C": mean, "low": low, "high": high, "splits": len(part),
                     "pairs_per_split": float(part["size"].mean())})
    return pd.DataFrame(rows, columns=keys + ["C", "low", "high", "splits", "pairs_per_split"])


def summary(collected=None):
    """The ladder and the breakdowns as tables; also writes v3_explore/twin_pairs.csv and summary.json."""
    pairs, skill = collected or collect()
    main = pairs[pairs["lead_ms"] == config.MIN_WARNING_MS]
    exact = main["rms"].astype(float) < v3_runs.EXACT_RMS
    result = {"ladder": c_table(main), "exact": c_table(main[exact]), "loose": c_table(main[~exact]),
              "by_group": c_table(main, "group"), "by_heating": c_table(main[main["heating"] != "unknown"], "heating"),
              "by_other_group": c_table(main, "other_group"), "by_lead": c_table(pairs, "lead_ms"),
              "skill": skill.groupby("recipe")[["pauc", "catch_rate", "false_alarm_rate"]].mean().reset_index()}
    if len(pairs):
        os.makedirs(folder(), exist_ok=True)
        pairs.to_csv(os.path.join(folder(), "twin_pairs.csv"), index=False)
        experiment.write_json({name: table.to_dict(orient="records") for name, table in result.items()},
                              os.path.join(folder(), "summary.json"))
    return result


def cell(row):
    """'0.697 (0.671 to 0.723), 156 pairs' for one c_table row."""
    rng = f" ({row['low']:.3f} to {row['high']:.3f})" if pd.notna(row["low"]) else ""
    return f"{row['C']:.3f}{rng}, {row['pairs_per_split']:.0f} pairs"


def text(result):
    """The summary in words."""
    lines = ["Exploratory, not pre-registered: descriptions with corrected 95% ranges over the splits, no verdicts.",
             "C = share of twin pairs in which the lost shot scores higher than its same-programme twin (0.5 = chance).",
             "", "The ladder: C by what the model sees"]
    ladder = result["ladder"].set_index("recipe") if len(result["ladder"]) else result["ladder"]
    skill = result["skill"].set_index("recipe") if len(result["skill"]) else result["skill"]
    for model in MODELS:
        if model in ladder.index:
            row = ladder.loc[model]
            lines.append(f"- {SEEN[model]}: C {cell(row)}, {int(row['splits'])} splits; pAUC {skill.loc[model, 'pauc']:.3f}, "
                         f"caught {skill.loc[model, 'catch_rate']:.1%}")
    lines += ["", f"The ladder on twins whose programmes differ by under {v3_runs.EXACT_RMS:.0%} (exact) and on the "
                  "rest (loose, up to 10%):"]
    exact, loose = (result[name].set_index("recipe") if len(result[name]) else result[name]
                    for name in ("exact", "loose"))
    for model in MODELS:
        cells = [f"{label} {cell(table.loc[model])}" for label, table in (("exact", exact), ("loose", loose))
                 if model in table.index]
        if cells:
            lines.append(f"- {SEEN[model]}: " + "; ".join(cells))
    heat = result["by_heating"]
    lines += ["", "C by beam heating of the pair (twins are matched on the programmed current only; same = beam power "
                  f"within {HEATING_RMS:.0%} of its peak, or no beam in either):"]
    for value in sorted(heat["heating"].unique()) if len(heat) else []:
        part = heat[heat["heating"] == value].set_index("recipe")
        lines.append(f"- {value} heating: " + "; ".join(f"{model} {cell(part.loc[model])}" for model in MODELS
                                                         if model in part.index))
    for name, by, title in (("by_group", "group", "C by the lost shot's kind"),
                            ("by_other_group", "other_group", "C by the kept twin's ending"),
                            ("by_lead", "lead_ms", "C when the scores stop this long before the loss (ms)")):
        table = result[name]
        lines += ["", title + ":"]
        for value in sorted(table[by].unique()) if len(table) else []:
            part = table[table[by] == value].set_index("recipe")
            cells = [f"{model} {cell(part.loc[model])}" for model in ("baseline", "machine") if model in part.index]
            lines.append(f"- {value}: " + "; ".join(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="run the ladder runs that are not done yet")
    parser.add_argument("--summary", action="store_true", help="the ladder and the breakdowns in words")
    args = parser.parse_args()
    if args.train:
        train_all()
    elif args.summary:
        out = text(summary())
        os.makedirs(folder(), exist_ok=True)
        with open(os.path.join(folder(), "summary.txt"), "w") as f:
            f.write(out + "\n")
        print(out)
    else:
        parser.error("give --train or --summary")


if __name__ == "__main__":
    main()
