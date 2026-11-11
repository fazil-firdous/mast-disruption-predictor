"""The remaining numbers for the twin-test paper, from files that already exist (no training):
events/programme_twins.csv, events/catalogue.csv, v3/runs.csv, v3/evidence/*/metrics.json and
v3_explore/twin_pairs.csv (written by scripts/v3_explore.py --summary; run that first).

    python scripts/paper2_numbers.py

Everything here is a count or an exploratory description (corrected 95% ranges over the 50 splits, no verdicts).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import v3_explore, v3_runs
from src import config
from src.stats import corrected_interval

PAIRS = [("baseline", "machine"), ("baseline", "solenoid"), ("baseline", "programme")]


def rng_text(mean, low, high):
    return f"{mean:+.3f}" + (f" ({low:+.3f} to {high:+.3f})" if low is not None else "")


def paired(pairs, a, b, value="win"):
    """(mean, low, high, splits) of C of model a minus C of model b, split by split, on the given twin pairs."""
    pairs = pairs[pairs[value].notna()]
    per = pairs[pairs["recipe"].isin([a, b])].groupby(["recipe", "seed"])[value].mean().unstack("recipe")
    if a not in per or b not in per:
        return None
    diff = (per[a] - per[b]).dropna().to_numpy(dtype=float)
    return (*corrected_interval(diff), len(diff)) if len(diff) else None


def twins_text(twins):
    gap = twins["gap"].astype(float)
    return [f"Twin pairs: {len(twins):,} ({twins['group'].value_counts().to_dict()}); kept twin's ending "
            f"{twins['other_group'].value_counts().to_dict()}.",
            f"Programmes within {v3_runs.EXACT_RMS:.0%}: {(twins['rms'] < v3_runs.EXACT_RMS).mean():.1%} of pairs. Gap "
            f"in shots: median {gap.median():.0f}, at most 3 in {(gap <= 3).mean():.1%}, at most 10 in "
            f"{(gap <= 10).mean():.1%}."]


def core_table(main):
    """The core table: the full model, the machine-side model and their paired difference, with the
    registered statistic and with both shots scored over the same time, on all twins and under stricter matching.
    Two ranges each: over the 50 splits, and over groups of pairs that share a twin."""
    same = main["heating"] == "same"
    rows = [("all twins", main), ("same beam heating", main[same]),
            ("same beam heating and same early height", main[same & (main["height"] == "same")]),
            ("recorded lost shots", main[main["group"] == "recorded"]),
            ("candidate lost shots", main[main["group"] == "candidate"])]
    lines = ["Core table: full model (baseline), machine-side model (machine) and full minus machine-side on identical "
             "pairs. 'splits' = corrected range over the 50 splits; 'groups' = pooled over pairs, resampling groups of "
             "pairs that share a twin (conditions on the fitted models)."]
    for name, part in rows:
        unique = part.drop_duplicates(["shot_id", "other"])
        lines.append(f"{name}: {len(unique):,} pairs, {unique['other'].nunique():,} different twins")
        for value, label in (("win", "registered statistic"), ("win_common", "same-time statistic")):
            table = v3_explore.c_table(part, value=value).set_index("recipe")
            pooled = v3_explore.pooled_table(part, value).set_index("recipe")
            cells = []
            for model in ("baseline", "machine"):
                if model in table.index and model in pooled.index:
                    cells.append(f"{model} {v3_explore.cell(table.loc[model])} | groups {pooled.loc[model, 'C']:.3f} "
                                 f"({pooled.loc[model, 'low']:.3f} to {pooled.loc[model, 'high']:.3f})")
            d_split, d_group = paired(part, "baseline", "machine", value), v3_explore.pooled_delta(
                part, "baseline", "machine", value)
            if d_split and d_group:
                cells.append(f"difference: splits {rng_text(*d_split[:3])} | groups {rng_text(*d_group[:3])}")
            lines.append(f"- {label}: " + "; ".join(cells))
    lost_more = int(main["win_common"].isna().sum())
    lines.append(f"Pair-and-split rows with no shared time (left out of the same-time statistic): {lost_more:,} of "
                 f"{len(main):,}.")
    return lines


def reuse_text(twins, triplets=None, catalogue=None):
    """How often one kept shot is the twin of several lost shots, how the triplets' gaps compare, and how the lost
    shots with a twin differ from those without one."""
    per_twin = twins.groupby("other").size()
    lines = [f"Twins used: {len(per_twin):,} different kept shots for {len(twins):,} lost shots; a kept shot is the "
             f"twin of one lost shot in {(per_twin == 1).mean():.1%} of cases, of two in {(per_twin == 2).mean():.1%}, of "
             f"three or more in {(per_twin >= 3).mean():.1%} (most: {int(per_twin.max())})."]
    if triplets is not None and len(triplets):
        first, second = triplets["shot_id"] - triplets["k1"], triplets["k1"] - triplets["k2"]
        lines.append(f"Triplets: gap L to K1 median {first.median():.0f} shots (mean {first.mean():.1f}); gap K1 to K2 "
                     f"median {second.median():.0f} (mean {second.mean():.1f}).")
    if catalogue is not None and "campaign" in catalogue:
        lost = catalogue[catalogue["group"].isin(["recorded", "candidate"])]
        has = lost["shot_id"].isin(twins["shot_id"])
        for name, part in (("with a twin", lost[has]), ("without a twin", lost[~has])):
            lines.append(f"Lost shots {name}: {len(part):,}; recorded {(part['group'] == 'recorded').mean():.1%}; by "
                         f"campaign {part['campaign'].value_counts().sort_index().to_dict()}.")
    return lines


def subsets(main):
    exact = main["rms"].astype(float) < v3_runs.EXACT_RMS
    same = main["heating"] == "same"
    return {"all twins": main, "exact programme": main[exact], "same heating": main[same],
            "same heating and exact programme": main[same & exact]}


def text(twins, pairs, runs=None, catalogue=None, evidence_counts=None, skill=None, leads=None, triplets=None):
    lines = twins_text(twins) + reuse_text(twins, triplets, catalogue)
    if skill:
        lines.append("Usual metrics per model, mean over the splits (pAUC, caught, false alarms): " + "; ".join(
            f"{row['recipe']} {row['pauc']:.3f}, {row['catch_rate']:.1%}, {row['false_alarm_rate']:.1%}" for row in skill)
            + ".")
    if leads is not None and len(leads):
        counts = leads.groupby("lead_ms")["n"].max()
        lines.append("Triplets compared in scripts/twin_differences.py, by time before the loss (ms): "
                     + ", ".join(f"{int(lead)}: {int(n):,}" for lead, n in counts.items()) + ".")
    main = pairs[pairs["lead_ms"] == config.MIN_WARNING_MS]
    unique = main.drop_duplicates(["shot_id", "other"])
    lines.append(f"Twin pairs that are in at least one test set: {len(unique):,}; beam heating "
                 f"{unique['heating'].value_counts().to_dict()}.")
    if catalogue is not None:
        n = int(((catalogue["group"] == "candidate") & (catalogue["words_vote"] == 2)).sum())
        lines.append(f"Evidence shots (candidates whose comment names a loss within 20 ms of the quench): {n:,} of "
                     f"{int((catalogue['group'] == 'candidate').sum()):,} candidates.")
    if evidence_counts:
        lines.append(f"Evidence shots in the training set, per split: mean {np.mean(evidence_counts):.0f} "
                     f"(min {min(evidence_counts)}, max {max(evidence_counts)}).")
    if runs is not None and "hyst_ratio" in runs:
        base = runs[runs["recipe"] == "baseline"].dropna(subset=["hyst_ratio"])
        chosen = base.groupby(["hyst_ratio", "hyst_hold"]).size().sort_values(ascending=False)
        lines.append("Alarm rule chosen on validation (ratio, hold windows): splits: "
                     + ", ".join(f"({r:g}, {int(h)}): {n}" for (r, h), n in chosen.items()) + ".")
    for name, part in subsets(main).items():
        table = v3_explore.c_table(part).set_index("recipe")
        lines += ["", f"C on {name} ({table['pairs_per_split'].max() if len(table) else 0:.0f} pairs per split):"]
        lines += [f"- {model}: {v3_explore.cell(table.loc[model])}" for model in v3_explore.MODELS if model in table.index]
        for a, b in PAIRS:
            p = paired(part, a, b)
            if p:
                lines.append(f"- {a} minus {b}: {rng_text(*p[:3])}, {p[3]} splits")
    if "win_common" in main and "height" in main:
        lines += [""] + core_table(main)
    same = pairs[pairs["heating"] == "same"]
    by_lead = v3_explore.c_table(same[same["recipe"].isin(["baseline", "machine"])], "lead_ms")
    lines += ["", "Same-heating twins, C when the scores stop this long before the loss (ms):"]
    for lead in sorted(by_lead["lead_ms"].unique()) if len(by_lead) else []:
        part = by_lead[by_lead["lead_ms"] == lead].set_index("recipe")
        lines.append(f"- {lead}: " + "; ".join(f"{m} {v3_explore.cell(part.loc[m])}" for m in ("baseline", "machine")
                                                if m in part.index))
    return "\n".join(lines)


def main():
    pairs_path = os.path.join(v3_explore.folder(), "twin_pairs.csv")
    if not os.path.exists(pairs_path):
        sys.exit("run scripts/v3_explore.py --summary first: it writes v3_explore/twin_pairs.csv")
    runs_path = os.path.join(v3_runs.folder(), "runs.csv")
    counts = []
    base = os.path.join(v3_runs.folder(), "evidence")
    for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        path = os.path.join(base, entry, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                counts.append(json.load(f).get("n_evidence_shots", 0))
    skill = None
    summary_path = os.path.join(v3_explore.folder(), "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            skill = json.load(f).get("skill")
    leads_path = os.path.join(config.RESULTS_DIR, "twin_differences", "leads.csv")
    triplets_path = os.path.join(config.RESULTS_DIR, "twin_differences", "triplets.csv")
    out = text(pd.read_csv(v3_runs.TWINS), pd.read_csv(pairs_path),
               pd.read_csv(runs_path) if os.path.exists(runs_path) else None,
               pd.read_csv(v3_runs.CATALOGUE) if os.path.exists(v3_runs.CATALOGUE) else None, counts, skill,
               pd.read_csv(leads_path) if os.path.exists(leads_path) else None,
               pd.read_csv(triplets_path) if os.path.exists(triplets_path) else None)
    with open(os.path.join(v3_explore.folder(), "paper2_numbers.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
