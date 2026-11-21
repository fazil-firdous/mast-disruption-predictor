"""The twin paper's 50-split numbers with the corrected full model A_c next to the original A (amendment 3 of
docs/preregistration_saddle_correction.md). Reads saved runs only: no model is trained, and nothing in v3/ or
v3_explore/ is written. A_c = the polarity-corrected first-harmonic saddle-array feature under an assumed geometry.

The statistics are those of scripts/v3_explore.py and scripts/v3_runs.py, computed by their own functions on the same
pairs; the only change is that the model list includes A_c ('saddle_corr').

    python scripts/paper2_corrected.py            # the 50-split tables
    python scripts/paper2_corrected.py --extra    # fixed leads over 50 splits, and the chronological split

Writes RESULTS_DIR/saddle_correction/paper2_tables.txt (and .json), paper2_extra.txt. Outside every stamp.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, saddle_correction as sc, twin_differences, v3_explore, v3_runs
from src import config, notify
from src.stats import corrected_interval

MODELS = ("programme", "solenoid", "machine", "baseline", sc.RECIPE)
NAMES = {"programme": "programme-only", "solenoid": "solenoid-only", "machine": "machine-side (M)",
         "baseline": "full, original (A)", sc.RECIPE: "full, corrected (A_c)"}


_RUN_FOLDERS = v3_explore.run_folders  # the original, kept before collect() swaps it


def folders():
    base = _RUN_FOLDERS()
    out = {m: base[m] for m in MODELS if m in base}
    out[sc.RECIPE] = os.path.join(sc.folder(), sc.RECIPE)
    return out


def collect():
    """v3_explore.collect with A_c among the models (its run_folders swapped for this call only)."""
    v3_explore.run_folders = folders
    try:
        return v3_explore.collect()
    finally:
        v3_explore.run_folders = _RUN_FOLDERS


def tables(pairs, skill, placebo):
    """The tables of v3_explore.summary, computed by the same functions, without writing anything."""
    main = pairs[pairs["lead_ms"] == config.MIN_WARNING_MS]
    same = main["heating"] == "same"
    t = {"registered C, 50 splits": v3_explore.c_table(main),
         "same-time C, 50 splits": v3_explore.c_table(main, value="win_common"),
         "same-time C, same heating": v3_explore.c_table(main[same], value="win_common"),
         "same-time C, pooled": v3_explore.pooled_table(main, "win_common"),
         "last shared window": v3_explore.c_table(main, value="win_end"),
         "same-time C, heating and height matched": v3_explore.c_table(
             main[same & (main["height"] == "same")], value="win_common"),
         "placebo (same-time)": v3_explore.c_table(placebo, value="win_common"),
         "placebo pooled": v3_explore.pooled_table(placebo, "win_common", connected=True)}
    for lead in (30, 50, 100, 200):
        t[f"registered C at {lead} ms before the loss"] = v3_explore.c_table(pairs[pairs["lead_ms"] == lead])
    t["by kind of lost shot (same-time)"] = v3_explore.c_table(main, "group", value="win_common")
    t["usual metrics, mean over runs"] = skill.groupby("recipe")[["pauc", "catch_rate", "false_alarm_rate"]].mean() \
        .reset_index()
    return t, main


def registered(main, seeds=v3_runs.CONFIRM_SEEDS):
    """H1a and H1b on the confirmation splits for A and A_c: per split C (registered statistic), corrected range of
    C - 0.5 and of C minus the machine-side model's C, one-sided p."""
    part = main[main["seed"].isin(list(seeds))]
    per = part.groupby(["recipe", "seed"])["win"].mean().unstack("recipe")
    out = []
    for model in ("baseline", sc.RECIPE):
        if model not in per:
            continue
        for name, change in (("H1a: C - 0.5", per[model] - 0.5),
                             ("H1b: C - C(M)", (per[model] - per["machine"]) if "machine" in per else None)):
            if change is None:
                continue
            c = change.dropna().to_numpy(dtype=float)
            mean, low, high = corrected_interval(c)
            out.append({"model": model, "test": name, "splits": len(c), "mean": mean, "low": low, "high": high,
                        "p_one_sided": v3_runs.one_sided(c)})
    return pd.DataFrame(out)


def deltas(main, placebo):
    """A_c minus A and A_c minus M on identical pairs, pooled over twin groups, and over splits."""
    rows = []
    for a, b in ((sc.RECIPE, "machine"), (sc.RECIPE, "baseline"), ("baseline", "machine")):
        for label, frame, value in (("same-time, all twins", main, "win_common"),
                                    ("same-time, same heating", main[main["heating"] == "same"], "win_common"),
                                    ("registered, all twins", main, "win")):
            d = v3_explore.pooled_delta(frame, a, b, value)
            per = frame[frame["recipe"].isin([a, b])].groupby(["recipe", "seed"])[value].mean().unstack("recipe")
            over = corrected_interval((per[a] - per[b]).dropna().to_numpy(dtype=float)) \
                if a in per and b in per else (np.nan, np.nan, np.nan)
            rows.append({"difference": f"{NAMES[a]} minus {NAMES[b]}", "pairs": label,
                         "over splits": over, "pooled": d[:3] if d else None, "pooled pairs": d[3] if d else None})
    g = {m: v3_explore.pooled_group_difference(main, m) for m in ("baseline", sc.RECIPE, "machine")}
    return rows, g


def fmt(r):
    if r is None or r[0] is None or not np.isfinite(r[0]):
        return "n/a"
    return f"{r[0]:+.3f} ({r[1]:+.3f} to {r[2]:+.3f})"


def text(t, reg, rows, group):
    lines = ["Twin paper, corrected counterparts (amendment 3): A = original representation, A_c = polarity-corrected "
             "first-harmonic saddle-array feature under an assumed geometry. Same statistics, same pairs, saved runs "
             "only. Ranges over splits are corrected (Nadeau-Bengio); pooled ranges resample groups of pairs that share "
             "a twin and condition on the fitted models.", ""]
    for name, table in t.items():
        lines.append(f"== {name}")
        for row in table.to_dict(orient="records"):
            label = NAMES.get(row["recipe"], row["recipe"])
            if "C" in row:
                rng = f" ({row['low']:.3f} to {row['high']:.3f})" if pd.notna(row.get("low")) else ""
                extra = f", {row['pairs_per_split']:.0f} pairs per split, {row['splits']} splits" if "splits" in row \
                    else f", {row.get('pairs')} pairs"
                key = f" [{row['group']}]" if "group" in row else ""
                lines.append(f"  {label}{key}: {row['C']:.3f}{rng}{extra}")
            else:
                lines.append(f"  {label}: pAUC {row['pauc']:.3f}, caught {row['catch_rate']:.1%}, false alarms "
                             f"{row['false_alarm_rate']:.1%}")
        lines.append("")
    lines.append("== Registered tests on the confirmation splits 21 to 50 (registered statistic)")
    for r in reg.to_dict(orient="records"):
        lines.append(f"  {NAMES[r['model']]}, {r['test']}: {r['mean']:+.3f} ({r['low']:+.3f} to {r['high']:+.3f}), "
                     f"one-sided p {r['p_one_sided']:.4f}, {r['splits']} splits")
    lines += ["", "== Differences on identical pairs (over splits | pooled)"]
    for r in rows:
        lines.append(f"  {r['difference']}, {r['pairs']}: {fmt(r['over splits'])} | {fmt(r['pooled'])} "
                     f"({r['pooled pairs']} pairs)")
    lines += ["", "== Recorded minus candidate lost shots (same-time, pooled over twin groups)"]
    for m, d in group.items():
        lines.append(f"  {NAMES[m]}: {fmt(d[:3]) if d else 'n/a'}" + (f" ({d[3]} recorded, {d[4]} candidate pairs)"
                                                                       if d else ""))
    return "\n".join(lines)


def fixed_leads():
    """The fixed-lead twin test (scripts/how_early.py) over the 50 random splits, with A_c among the models."""
    v3_explore.run_folders = folders
    try:
        random, _ = how_early.collect(models=["programme", "machine", "baseline", sc.RECIPE])
    finally:
        v3_explore.run_folders = _RUN_FOLDERS
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)]
    placebo = random[random["kind"] == "placebo"]
    lines = ["== Fixed leads, common cohort, 50 random splits (both shots scored at one time before the loss)"]
    for lead in how_early.COHORT_LEADS:
        split = how_early.over_splits(main, lead)
        cells = "; ".join(f"{NAMES[m]} {how_early.rng(*split[m][:3])}" for m in ("programme", "machine", "baseline",
                                                                                  sc.RECIPE) if m in split)
        d = how_early.split_difference(main, lead, sc.RECIPE, "machine")
        a = how_early.split_difference(main, lead, "baseline", "machine")
        pp = v3_explore.pooled_delta(main[main["lead_ms"] == lead], sc.RECIPE, "machine")
        n = split.get(sc.RECIPE, (None,) * 4)[3]
        lines.append(f"- {lead} ms ({f'{n:.0f}' if n else '?'} pairs per split): {cells}")
        lines.append(f"    A_c minus M {fmt(d[:3]) if d else 'n/a'} (pooled {fmt(pp[:3]) if pp else 'n/a'}); "
                     f"A minus M {fmt(a[:3]) if a else 'n/a'}")
    for lead in how_early.COHORT_LEADS:
        sp = how_early.over_splits(placebo, lead)
        if sc.RECIPE in sp:
            lines.append(f"  placebo at {lead} ms, A_c: {how_early.rng(*sp[sc.RECIPE][:3])}")
    return lines


def chronological(load=None):
    """The chronological split (trained on M5-M7, tested on M8-M9), same-time statistic, one split: ranges from
    resampling groups of pairs that share a twin (conditional on the one fitted model)."""
    from src import extras, server
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    paths = {m: v3_explore.temporal_dir(m) for m in ("programme", "machine", "baseline")}
    paths[sc.RECIPE] = sc.run_dir(None)
    starts, frames, same = {}, [], None
    for model, base in paths.items():
        if not server.result_saved(os.path.join(base, "metrics.json")):
            continue
        scores = pd.read_parquet(os.path.join(base, "test_scores.parquet"))
        ids = set(scores["shot_id"])
        mine = twins[twins["shot_id"].isin(ids) & twins["other"].isin(ids)]
        for shot in mine["shot_id"]:
            if shot not in starts:
                starts[shot] = v3_runs.flat_top_start(load(shot))
        wins = v3_explore.pair_wins(scores, mine, t_quench, starts, leads_ms=(config.MIN_WARNING_MS,))
        if same is None:
            pair_ids = list(mine[["shot_id", "other"]].itertuples(index=False, name=None))
            shots_in = {shot for pair in pair_ids for shot in pair}
            same = v3_explore.heating(pair_ids, v3_explore.beam_power(shots_in), t_quench, starts)
        wins["heating"] = [same.get(k, "unknown") for k in zip(wins["shot_id"], wins["other"])]
        frames.append(wins.assign(recipe=model, seed=0))
    frame = pd.concat(frames, ignore_index=True)
    lines = ["== Chronological split (M5-M7 train, M8-M9 test), same-time statistic, one split"]
    for model in paths:
        part = frame[(frame["recipe"] == model) & frame["win_common"].notna()]
        if len(part):
            r = twin_differences.cluster_range(part["win_common"].to_numpy(dtype=float), part["other"].to_numpy(),
                                               np.random.default_rng(config.RANDOM_SEED), v3_explore.N_BOOT)
            lines.append(f"  {NAMES[model]}: {r[0]:.3f} ({r[1]:.3f} to {r[2]:.3f}), {r[3]:,} pairs")
    for a, b in ((sc.RECIPE, "machine"), ("baseline", "machine"), (sc.RECIPE, "baseline")):
        for label, part in (("all test twins", frame), ("same heating", frame[frame["heating"] == "same"])):
            d = v3_explore.pooled_delta(part, a, b, "win_common")
            lines.append(f"  {NAMES[a]} minus {NAMES[b]}, {label}: {fmt(d[:3]) if d else 'n/a'}"
                         + (f" ({d[3]:,} pairs)" if d else ""))
    return lines


def extra_main():
    lines = ["Twin paper, corrected counterparts, part 2 (amendment 3): fixed leads and the chronological split.", ""]
    lines += fixed_leads() + [""] + chronological()
    out = "\n".join(lines)
    with open(os.path.join(sc.folder(), "paper2_extra.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Twin paper, corrected part 2 is ready (paper2_extra.txt)\n\n" + out[:3300])


def main():
    pairs, skill, placebo = collect()
    t, main_pairs = tables(pairs, skill, placebo)
    reg = registered(main_pairs)
    rows, group = deltas(main_pairs, placebo)
    out = text(t, reg, rows, group)
    os.makedirs(sc.folder(), exist_ok=True)
    with open(os.path.join(sc.folder(), "paper2_tables.txt"), "w") as f:
        f.write(out + "\n")
    with open(os.path.join(sc.folder(), "paper2_tables.json"), "w") as f:
        json.dump({name: table.to_dict(orient="records") for name, table in t.items()} |
                  {"registered": reg.to_dict(orient="records")}, f, indent=1, default=str)
    print(out)
    notify.send("Twin paper, corrected tables are ready (paper2_tables.txt)\n\n" + out[:3300])


if __name__ == "__main__":
    extra_main() if "--extra" in sys.argv[1:] else main()
