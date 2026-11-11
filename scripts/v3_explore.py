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

3. Three checks a review asked for: C with both shots scored over the same time (the registered statistic gives a
   shot more windows if its twin ended first), a range that resamples groups of pairs sharing a twin, and the test
   on a chronological split (train on campaigns M5-M7, test on M8-M9), where no repeat of a test shot's session is in
   the training set.

    python scripts/v3_explore.py --train       # the ladder runs not done yet, one after another (skips finished)
    python scripts/v3_explore.py --summary     # the ladder and the breakdowns in words
    python scripts/v3_explore.py --temporal    # the chronological split: three runs if missing, then its twin test

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
TEMPORAL = ["programme", "machine", "baseline"]  # the models run on the chronological split
N_BOOT = 2000
HEATING_RMS = twin_check.SAME_RMS  # same beam heating: beam power within 10% of its peak (RMS), like the programme
NO_BEAM_SHARE = 0.05  # beam power under 5% of the largest seen counts as no beam
MIN_WINDOWS = 5  # windows both shots must share to compare their beam power or height
HEIGHT_M = 0.02  # same early height: the magnetic axes within 2 cm (RMS) ...
EARLY_MS = 100  # ... up to 100 ms before the loss
MARGIN_MS = 50
MARGINS = ["twin ended before the loss", f"twin ended within {MARGIN_MS} ms after the loss",
           f"twin went on {MARGIN_MS} ms or more", "twin had no quench"]
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


def recipe_columns(recipe):
    """The inputs of a model: every signal (baseline), the machine side, or one of the ladder's two."""
    if recipe in LADDER:
        return ladder_columns(recipe)
    return features.model_features("machine" if recipe == "machine" else None)


def fit(win, split_column, recipe, path, info):
    """Train like the v3 baseline (candidates left out of training, tuned and judged on clear endings) with this
    recipe's inputs, test once, and save the test window scores like scripts/v3_runs.py. Returns the metrics."""
    train, val_all, test_all = experiment.split_frames(win, split_column)
    train = train[~train["candidate"]].reset_index(drop=True)
    val, test = experiment.judged(val_all, "clear"), experiment.judged(test_all, "clear")
    columns = experiment.usable(recipe_columns(recipe), train)
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


def run(recipe, seed, tables=None):
    """One ladder run on the E10 split of this seed."""
    windows, shots = tables or experiment.load_tables()
    split = dataset.random_split(shots, seed=seed)
    win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots)
    return fit(win, "split", recipe, out_dir(recipe, seed), {"seed": seed})


def temporal_dir(recipe):
    return os.path.join(folder(), "temporal", recipe)


def run_temporal(recipe, tables=None):
    """One run on the chronological split of the dataset (train and validation from campaigns M5-M7, test M8-M9)."""
    windows, shots = tables or experiment.load_tables()
    return fit(dataset.labelled_windows(windows, shots), "split_temporal", recipe, temporal_dir(recipe),
               {"split": "temporal"})


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


def highest(a, b):
    """1 if a's highest value is above b's, 0.5 for a tie, 0 otherwise."""
    return 1.0 if a.max() > b.max() else 0.5 if a.max() == b.max() else 0.0


def pair_wins(scores, twins, t_quench, t_start, leads_ms=LEADS_MS):
    """One row per (twin pair with both shots in scores, lead time): win is 1 if the lost shot's highest window
    score is above its twin's, 0.5 for a tie, 0 otherwise. The scores of both shots are taken from the flat-top
    start of the lost shot to lead_ms before its quench. At lead MIN_WARNING_MS this is v3_runs.concordance, pair
    by pair (same comparisons on the same saved values).

    win_common is the same comparison with both shots cut to the time they share (to within half a window stride),
    so that a shot whose twin ended first does not get more windows to take its highest score from."""
    by_shot = {shot: (g["t_end"].to_numpy(), g["score"].to_numpy()) for shot, g in scores.groupby("shot_id")}
    slack = config.STRIDE_MS / 2000
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
            in_a, in_b = (lost[0] >= t_from) & (lost[0] <= t_to), (kept[0] >= t_from) & (kept[0] <= t_to)
            a, b = lost[1][in_a], kept[1][in_b]
            if len(a) == 0 or len(b) == 0:
                continue
            ta, tb = lost[0][in_a].astype(float), kept[0][in_b].astype(float)
            first, last = max(ta.min(), tb.min()) - slack, min(ta.max(), tb.max()) + slack
            a2, b2 = a[(ta >= first) & (ta <= last)], b[(tb >= first) & (tb <= last)]
            rows.append({"shot_id": row.shot_id, "other": row.other, "lead_ms": lead, "win": highest(a, b),
                         "win_common": highest(a2, b2) if len(a2) and len(b2) else np.nan})
    return pd.DataFrame(rows, columns=["shot_id", "other", "lead_ms", "win", "win_common"])


def run_folders():
    """{model: folder of its runs}: the pre-registered recipes in v3/, the ladder in v3_explore/."""
    return {**{recipe: os.path.join(v3_runs.folder(), recipe) for recipe in v3_runs.RECIPES},
            **{recipe: os.path.join(folder(), recipe) for recipe in LADDER}}


def window_means(shot_ids, signal):
    """{shot: (time bins, window means of one signal)} from the window table, or {} if there is none."""
    path = os.path.join(config.PROCESSED_DIR, "windows.parquet")
    if not os.path.exists(path):
        return {}
    windows = pd.read_parquet(path, columns=["shot_id", "t_end", f"{signal}_mean"])
    return twin_differences.by_shot(windows[windows["shot_id"].isin(shot_ids)], signals=[signal])


def beam_power(shot_ids):
    return window_means(shot_ids, "pnbi")


def shared(series, lost, twin, t_from, t_to):
    """The two shots' values of one signal on the windows both have between t_from and t_to, where both are
    finite; None if a shot is missing or they share fewer than MIN_WINDOWS."""
    if lost not in series or twin not in series:
        return None
    (ba, va), (bb, vb) = series[lost], series[twin]
    common = np.intersect1d(ba, bb)
    t = common * twin_differences.STRIDE_S
    common = common[(t >= t_from) & (t <= t_to)]
    a, b = va[np.searchsorted(ba, common), 0], vb[np.searchsorted(bb, common), 0]
    ok = np.isfinite(a) & np.isfinite(b)
    return (a[ok], b[ok]) if ok.sum() >= MIN_WINDOWS else None


def time_range(lost, t_quench, t_start, lead_ms):
    """(from, to) of a lost shot's compared time: its flat-top start to lead_ms before its loss; None without a loss."""
    tq = t_quench.get(lost, np.nan)
    if not np.isfinite(tq):
        return None
    start = t_start.get(lost)
    return (-np.inf if start is None else start), tq - lead_ms / 1000


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
        span = time_range(lost, t_quench, t_start, config.MIN_WARNING_MS)
        both = shared(beams, lost, twin, *span) if span else None
        if both is None:
            continue
        a, b = both
        peak = max(np.abs(a).max(), np.abs(b).max())
        if peak <= NO_BEAM_SHARE * top:
            out[(lost, twin)] = "same"
        else:
            out[(lost, twin)] = "same" if float(np.sqrt(np.mean((a - b) ** 2)) / peak) < HEATING_RMS else "different"
    return out


def early_height(pair_ids, heights, t_quench, t_start):
    """{(lost shot, twin): 'same', 'different' or 'unknown'}: were the two plasmas at the same height early on? The
    programmed shape is not in the data, so this compares the measured height of the magnetic axis from the lost
    shot's flat-top start to EARLY_MS before its loss: 'same' if the RMS difference is under HEIGHT_M. The last
    EARLY_MS are left out so that a vertical excursion leading to the loss does not count as a different plan."""
    out = {}
    for lost, twin in pair_ids:
        out[(lost, twin)] = "unknown"
        span = time_range(lost, t_quench, t_start, EARLY_MS)
        both = shared(heights, lost, twin, *span) if span else None
        if both is not None:
            out[(lost, twin)] = "same" if float(np.sqrt(np.mean((both[0] - both[1]) ** 2))) < HEIGHT_M else "different"
    return out


def margin(pair_ids, t_quench):
    """{(lost shot, twin): how long the twin went on after the lost shot's loss}. If the twin ended first, the lost
    shot has more windows in the compared time, which can favour it whatever the plasma did."""
    out = {}
    for lost, twin in pair_ids:
        ms = 1000 * (t_quench.get(twin, np.nan) - t_quench.get(lost, np.nan))
        out[(lost, twin)] = (MARGINS[3] if not np.isfinite(ms) else MARGINS[0] if ms < 0 else MARGINS[1]
                             if ms < MARGIN_MS else MARGINS[2])
    return out


def collect(load=None, beams=None, heights=None):
    """(pairs, skill): pairs has one row per (model, split, twin pair, lead time) with the pair's groups, gap, beam
    heating, early height and how long the twin went on; skill one row per finished run with its test pAUC, catch
    and false-alarm rates."""
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
    columns = ["shot_id", "other", "lead_ms", "win", "win_common", "group", "other_group", "gap", "rms", "recipe", "seed",
               "heating", "height", "margin"]
    skill = pd.DataFrame(skill, columns=["recipe", "seed", "pauc", "catch_rate", "false_alarm_rate"])
    if not pairs:
        return pd.DataFrame(columns=columns), skill
    pairs = pd.concat(pairs, ignore_index=True)
    ids = list(pairs[["shot_id", "other"]].drop_duplicates().itertuples(index=False, name=None))
    beams = beam_power({shot for pair in ids for shot in pair}) if beams is None else beams
    heights = window_means({shot for pair in ids for shot in pair}, "zmag") if heights is None else heights
    keys = list(zip(pairs["shot_id"], pairs["other"]))
    for column, labels in (("heating", heating(ids, beams, t_quench, starts)),
                           ("height", early_height(ids, heights, t_quench, starts)), ("margin", margin(ids, t_quench))):
        pairs[column] = [labels[key] for key in keys]
    return pairs, skill


def c_table(pairs, by=None, value="win"):
    """C (mean over the splits of each split's share of twin pairs won) with its corrected 95% range, per model and,
    if given, per value of the column `by`. Columns: recipe, [by], C, low, high, splits, pairs_per_split. value:
    'win' (the registered statistic) or 'win_common' (both shots over the same time)."""
    keys = ["recipe"] + ([by] if by else [])
    pairs = pairs[pairs[value].notna()]
    per_split = pairs.groupby(keys + ["seed"])[value].agg(["mean", "size"]).reset_index()
    rows = []
    for key, part in per_split.groupby(keys, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        mean, low, high = corrected_interval(part["mean"].to_numpy(dtype=float))
        rows.append({**dict(zip(keys, key)), "C": mean, "low": low, "high": high, "splits": len(part),
                     "pairs_per_split": float(part["size"].mean())})
    return pd.DataFrame(rows, columns=keys + ["C", "low", "high", "splits", "pairs_per_split"])


def pooled_table(pairs, value="win", seed=config.RANDOM_SEED):
    """C pooled over the twin pairs (each pair's mean over the splits it is tested in), with a 95% range from
    resampling whole groups of pairs that share their twin. This range allows for one kept shot being the twin of
    several lost shots, which the range over splits does not; it does not allow for the model changing with the
    training set. Columns: recipe, C, low, high, pairs."""
    rows = []
    for recipe, part in pairs[pairs[value].notna()].groupby("recipe", sort=False):
        per_pair = part.groupby(["shot_id", "other"])[value].mean().reset_index()
        c, low, high, n = twin_differences.cluster_range(per_pair[value].to_numpy(dtype=float),
                                                         per_pair["other"].to_numpy(), np.random.default_rng(seed), N_BOOT)
        rows.append({"recipe": recipe, "C": c, "low": low, "high": high, "pairs": n})
    return pd.DataFrame(rows, columns=["recipe", "C", "low", "high", "pairs"])


def summary(collected=None):
    """The ladder and the breakdowns as tables; also writes v3_explore/twin_pairs.csv and summary.json."""
    pairs, skill = collected or collect()
    main = pairs[pairs["lead_ms"] == config.MIN_WARNING_MS]
    exact = main["rms"].astype(float) < v3_runs.EXACT_RMS
    same = main["heating"] == "same"
    result = {"ladder": c_table(main), "exact": c_table(main[exact]), "loose": c_table(main[~exact]),
              "common": c_table(main, value="win_common"), "common_same": c_table(main[same], value="win_common"),
              "pooled": pooled_table(main), "pooled_common": pooled_table(main, "win_common"),
              "by_group": c_table(main, "group"), "by_heating": c_table(main[main["heating"] != "unknown"], "heating"),
              "by_height": c_table(main[main["height"] != "unknown"], "height"),
              "by_matched": c_table(main[(main["heating"] == "same") & (main["height"] == "same")]),
              "by_margin": c_table(main, "margin"),
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
    lines += ["", "Review check 1, both shots scored over the same time (the registered statistic gives a shot more "
              "windows if its twin ended first): C on all twins; on same-heating twins"]
    common, common_same = (result[name].set_index("recipe") if len(result[name]) else result[name]
                           for name in ("common", "common_same"))
    for model in MODELS:
        if model in common.index:
            lines.append(f"- {SEEN[model]}: {cell(common.loc[model])}"
                         + (f"; {cell(common_same.loc[model])}" if model in common_same.index else ""))
    lines += ["", "Review check 2, C pooled over the pairs with a 95% range that resamples groups of pairs sharing a "
              "twin (registered statistic; same-time statistic):"]
    pooled, pooled_common = (result[name].set_index("recipe") if len(result[name]) else result[name]
                             for name in ("pooled", "pooled_common"))
    for model in MODELS:
        if model in pooled.index:
            cells = [f"{t.loc[model, 'C']:.3f} ({t.loc[model, 'low']:.3f} to {t.loc[model, 'high']:.3f})"
                     for t in (pooled, pooled_common) if model in t.index]
            lines.append(f"- {SEEN[model]}: " + "; ".join(cells) + f", {int(pooled.loc[model, 'pairs']):,} pairs")
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
    for name, by, title in (
            ("by_height", "height", f"C by the two plasmas' early height (same = magnetic axes within "
                                    f"{100 * HEIGHT_M:.0f} cm up to {EARLY_MS} ms before the loss; the programmed shape "
                                    "is not in the data)"),
            ("by_margin", "margin", "C by how long the twin went on after the lost shot's loss (if it ended first, "
                                    "the lost shot has more windows in the compared time)")):
        table = result[name]
        lines += ["", title + ":"]
        for value in sorted(table[by].unique()) if len(table) else []:
            part = table[table[by] == value].set_index("recipe")
            lines.append(f"- {value}: " + "; ".join(f"{model} {cell(part.loc[model])}" for model in MODELS
                                                    if model in part.index))
    matched = result["by_matched"].set_index("recipe") if len(result["by_matched"]) else result["by_matched"]
    lines += ["", "C on pairs with the same beam heating and the same early height: "
              + ("; ".join(f"{model} {cell(matched.loc[model])}" for model in MODELS if model in matched.index)
                 or "no such pairs")]
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


def temporal_text(load=None, beams=None):
    """Review check 3 in words: the twin test on the chronological split, for the runs that exist. One split, so
    the ranges come from resampling groups of pairs that share their twin."""
    from src import extras
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    lines = ["Review check 3, the chronological split (trained on campaigns M5-M7, tested on M8-M9; one split, ranges "
             "from resampling groups of pairs that share their twin). Exploratory.",
             "Per model: usual metrics; C on all test twins (registered statistic; same-time statistic); C on "
             "same-heating twins (same-time statistic)."]
    starts, same = {}, None
    for recipe in TEMPORAL:
        base = temporal_dir(recipe)
        if not server.result_saved(os.path.join(base, "metrics.json")):
            continue
        with open(os.path.join(base, "metrics.json")) as f:
            test = json.load(f)["test"]
        scores = pd.read_parquet(os.path.join(base, "test_scores.parquet"))
        ids = set(scores["shot_id"])
        mine = twins[twins["shot_id"].isin(ids) & twins["other"].isin(ids)]
        for shot in mine["shot_id"]:
            if shot not in starts:
                starts[shot] = v3_runs.flat_top_start(load(shot))
        wins = pair_wins(scores, mine, t_quench, starts, leads_ms=(config.MIN_WARNING_MS,))
        if same is None:
            pair_ids = list(mine[["shot_id", "other"]].itertuples(index=False, name=None))
            shots_in = {shot for pair in pair_ids for shot in pair}
            same = heating(pair_ids, beam_power(shots_in) if beams is None else beams, t_quench, starts)
        wins["heating"] = [same.get(pair, "unknown") for pair in zip(wins["shot_id"], wins["other"])]

        def c(part, value):
            ok = part[part[value].notna()]
            m, low, high, n = twin_differences.cluster_range(ok[value].to_numpy(dtype=float), ok["other"].to_numpy(),
                                                             np.random.default_rng(config.RANDOM_SEED), N_BOOT)
            return f"{m:.3f} ({low:.3f} to {high:.3f}), {n:,} pairs" if n else "no pairs"

        lines.append(f"- {SEEN[recipe]}: pAUC {test['pauc']:.3f}, caught {test['catch_rate']:.1%}, false alarms "
                     f"{test['false_alarm_rate']:.1%}; C {c(wins, 'win')}; {c(wins, 'win_common')}; same heating "
                     f"{c(wins[wins['heating'] == 'same'], 'win_common')}")
    return "\n".join(lines)


def temporal():
    """Run the chronological-split runs that are missing, then print (and save) their twin test."""
    todo = [recipe for recipe in TEMPORAL if not server.result_saved(os.path.join(temporal_dir(recipe), "metrics.json"))]
    tables = experiment.load_tables() if todo else None
    for recipe in todo:
        start = time.time()
        m = run_temporal(recipe, tables)
        print(f"{recipe} on the chronological split: pAUC {m['test']['pauc']:.3f}, {(time.time() - start) / 60:.1f} min",
              flush=True)
    out = temporal_text()
    with open(os.path.join(folder(), "temporal.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="run the ladder runs that are not done yet")
    parser.add_argument("--temporal", action="store_true", help="the chronological split and its twin test")
    parser.add_argument("--summary", action="store_true", help="the ladder and the breakdowns in words")
    args = parser.parse_args()
    if args.train:
        train_all()
    elif args.temporal:
        temporal()
    elif args.summary:
        out = text(summary())
        os.makedirs(folder(), exist_ok=True)
        with open(os.path.join(folder(), "summary.txt"), "w") as f:
            f.write(out + "\n")
        print(out)
    else:
        parser.error("give --train, --summary or --temporal")


if __name__ == "__main__":
    main()
