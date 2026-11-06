"""The paper's model tables from the saved repeat runs (E10 seeds and the comparisons), with no retraining.

    python scripts/paper_numbers.py                      # reads RESULTS_DIR, writes RESULTS_DIR/paper/numbers.md
    python scripts/paper_numbers.py --results unzipped/  # the same from an unzipped /report archive

Three tables, each over the repeat runs' test shots, every range corrected for the splits sharing shots (src/stats.py):
1. Skill: caught minus false alarms, per split. An alarm that fires at random on a share f of shots catches about f of
   the disruptions, so a range above 0 means skill over random alarms.
2. Comparisons: the change in caught, false alarms, window PR-AUC and pAUC against each comparison's own baseline,
   with p-values Holm-adjusted over the family (caught and false alarms together, as in the summary; PR-AUC and pAUC
   each on their own). pAUC is the area under the shot-level curve of caught against false alarms up to 10% false
   alarms, divided by 0.10 (random alarms score 0.05), from each run's saved tradeoff_test.csv. It needs no
   threshold, so it is less noisy than the catch rate at one threshold (the idea of the area under the warning-time
   curve in MIT-PSFC/disruption-survival-analysis).
3. Alarms by ending: the share of each ending group alarmed in time, pooled over splits, and the share of the
   candidates that behave like recorded disruptions under a two-group mixture:
   p = (candidates - reference) / (recorded - reference), with flux-exhausted or smooth endings as the reference.
   The mixture assumes the disruption-like candidates are alarmed like recorded disruptions and the rest like the
   reference group: a rough reading, not a measurement. It is left out for runs whose labels call the candidates
   disruptions (current-only and physics labels), where it means nothing. Also per split: recorded disruptions alarmed
   in time minus smooth endings alarmed, i.e. skill on the disruptions the operators recorded, whatever the labels.

An analysis next to the experiments, like scripts/label_statistics.py: no experiment uses it, so it is outside every
stamp, and it writes only into RESULTS_DIR (private).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, experiment
from src.stats import corrected_interval, corrected_p, holm

METRICS = ["catch_rate", "false_alarm_rate", "pr_auc", "pauc"]
PAUC_MAX_FALSE_ALARM = 0.10  # the shot-level curve up to 10% false alarms: the range a machine could live with
MIXTURE_REFERENCES = ["flux_exhausted", "smooth_end"]
GROUPS = ["recorded", "candidate", "flux_exhausted", "programmed_rampdown", "smooth_end", "other"]


def partial_auc(curve, cutoff=PAUC_MAX_FALSE_ALARM):
    """Area under caught (y) against false alarms (x) for x <= cutoff, divided by cutoff, from a trade-off curve (one
    row per threshold). At each false-alarm rate the best catch rate counts; the curve starts at (0, 0) and is held
    flat from its last point below the cutoff to the cutoff. Random alarms score cutoff / 2. NaN without a curve."""
    if curve is None or len(curve) == 0:
        return np.nan
    best = curve.groupby("false_alarm_rate")["catch_rate"].max().sort_index()
    x, y = best.index.to_numpy(dtype=float), np.maximum.accumulate(best.to_numpy(dtype=float))
    keep = x <= cutoff
    x = np.concatenate([[0.0], x[keep], [cutoff]])
    y = np.concatenate([[0.0], y[keep], [y[keep][-1] if keep.any() else 0.0]])
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2) / cutoff)


def read_runs(folder):
    """seed -> the test numbers of every finished run in folder (seed_N/metrics.json), with pr_auc lifted out of
    the window metrics and the validation share of positive windows (PR-AUC of random scores) alongside."""
    runs = {}
    for entry in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        path = os.path.join(folder, entry, "metrics.json")
        if not entry.startswith("seed_") or not os.path.exists(path):
            continue
        with open(path) as f:
            metrics = json.load(f)
        if "test" not in metrics:
            continue
        test = dict(metrics["test"])
        test["pr_auc"] = test.get("window", {}).get("pr_auc", np.nan)
        test["positive_share"] = metrics.get("val", {}).get("positive_share", np.nan)
        curve_path = os.path.join(folder, entry, "tradeoff_test.csv")
        test["pauc"] = partial_auc(pd.read_csv(curve_path)) if os.path.exists(curve_path) else np.nan
        runs[int(entry[len("seed_"):])] = test
    return runs


def run_sets(results):
    """name -> (runs, baseline name or None): E10 and every comparison found under results."""
    e10 = os.path.join(results, "E10")
    sets = {"E10": (read_runs(e10), None)}
    for name in config.COMPARISONS:
        runs = read_runs(os.path.join(e10, "comparisons", name))
        if runs:
            sets[name] = (runs, config.COMPARISON_BASELINES.get(name, "E10"))
    return sets


def interval_row(values):
    """mean, corrected 95% range and p-value of a list of per-split values (NaN dropped)."""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    mean, low, high = corrected_interval(values)
    return {"n": len(values), "mean": mean, "low": low, "high": high, "p": corrected_p(values)}


def skill_table(sets):
    """Per run set: mean caught, false alarms, PR-AUC (and its random level), and caught minus false alarms."""
    out = {}
    for name, (runs, _) in sets.items():
        if not runs:
            continue
        rows = list(runs.values())
        out[name] = {"n": len(rows),
                     **{m: float(np.nanmean([r[m] for r in rows])) for m in METRICS + ["positive_share"]},
                     "skill": interval_row([r["catch_rate"] - r["false_alarm_rate"] for r in rows]),
                     "pauc_skill": interval_row([r["pauc"] - PAUC_MAX_FALSE_ALARM / 2 for r in rows])}
    return out


def comparison_table(sets):
    """Per comparison and metric: the paired change against its baseline on the same seeds, with Holm-adjusted p."""
    out = {}
    for name, (runs, baseline) in sets.items():
        if baseline is None or baseline not in sets:
            continue
        base = sets[baseline][0]
        seeds = sorted(set(runs) & set(base))
        out[name] = {"baseline": baseline, "n": len(seeds)}
        for metric in METRICS:
            out[name][metric] = {"value": float(np.nanmean([runs[s][metric] for s in seeds])) if seeds else None,
                                 "baseline": float(np.nanmean([base[s][metric] for s in seeds])) if seeds else None,
                                 **interval_row([runs[s][metric] - base[s][metric] for s in seeds])}
    for family in (["catch_rate", "false_alarm_rate"], ["pr_auc"], ["pauc"]):
        cells = [(name, metric) for name in out for metric in family]
        adjusted = holm([out[name][metric]["p"] for name, metric in cells])
        for (name, metric), p in zip(cells, adjusted):
            out[name][metric]["p_holm"] = p
    return out


def ending_rates(test):
    """group -> share alarmed in time in one run (None for a group with no shots)."""
    counts = test.get("alarms_by_ending", {})
    return {g: counts[g]["alarms"] / counts[g]["n"] if g in counts and counts[g]["n"] else None for g in GROUPS}


def mixture_share(rates, reference):
    """(candidates - reference) / (recorded - reference), or None when recorded is not above the reference."""
    rec, cand, ref = rates.get("recorded"), rates.get("candidate"), rates.get(reference)
    if None in (rec, cand, ref) or rec <= ref:
        return None
    return (cand - ref) / (rec - ref)


def labels_call_candidates_disruptions(name):
    """True for runs whose label source counts the candidates as disruptions (current-only or physics labels)."""
    knob, value = config.COMPARISONS.get(name, (None, None))
    return knob == "recipe" and experiment.RECIPES.get(value, {}).get("source") in ("signal", "physics")


def ending_table(sets):
    """Per run set: pooled share of each ending group alarmed in time, and the mixture share of the candidates
    (pooled over splits, and per split with a corrected range)."""
    out = {}
    for name, (runs, _) in sets.items():
        pooled = {g: [0, 0] for g in GROUPS}
        per_split = {ref: [] for ref in MIXTURE_REFERENCES}
        recorded_skill = []
        for test in runs.values():
            for g, c in test.get("alarms_by_ending", {}).items():
                if g in pooled:
                    pooled[g][0] += c["alarms"]
                    pooled[g][1] += c["n"]
            rates = ending_rates(test)
            if rates["recorded"] is not None and rates["smooth_end"] is not None:
                recorded_skill.append(rates["recorded"] - rates["smooth_end"])
            for ref in MIXTURE_REFERENCES:
                per_split[ref].append(mixture_share(rates, ref))
        if not any(n for _, n in pooled.values()):
            continue
        pooled_rates = {g: a / n if n else None for g, (a, n) in pooled.items()}
        mixture = None if labels_call_candidates_disruptions(name) else {
            ref: {"pooled": mixture_share(pooled_rates, ref), **interval_row(per_split[ref])}
            for ref in MIXTURE_REFERENCES}
        out[name] = {"pooled": {g: {"alarms": a, "n": n, "rate": pooled_rates[g]} for g, (a, n) in pooled.items()},
                     "recorded_skill": interval_row(recorded_skill), "mixture": mixture}
    return out


def pct(x, digits=1):
    return "-" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def points(row):
    """'+3.1 (+0.4 to +5.8)' in percentage points, from an interval_row."""
    if row["mean"] is None:
        return "-"
    if row["low"] is None:
        return f"{100 * row['mean']:+.1f} (too few splits)"
    return f"{100 * row['mean']:+.1f} ({100 * row['low']:+.1f} to {100 * row['high']:+.1f})"


def share_range(row):
    """'52% (31% to 73%)' from an interval_row of shares."""
    if row["mean"] is None:
        return "-"
    if row["low"] is None:
        return f"{pct(row['mean'], 0)} (too few splits)"
    return f"{pct(row['mean'], 0)} ({pct(row['low'], 0)} to {pct(row['high'], 0)})"


def p_text(p):
    return "-" if p is None else f"{p:.3f}" if p >= 0.001 else f"{p:.0e}"


def markdown(skill, comparisons, endings):
    lines = ["# Paper numbers (private: model numbers stay out of the repository)", "",
             "Every range is a 95% range corrected for the splits sharing shots (Nadeau & Bengio 2003); changes are "
             "in percentage points. Holm p-values are adjusted over all comparisons together.", "",
             "## 1. Skill over random alarms (caught minus false alarms, per split)", "",
             "| Runs | Splits | Caught | False alarms | PR-AUC (random) | Caught - false alarms | p | pAUC (random 0.05) "
             "| pAUC - random | p |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in skill.items():
        lines.append(f"| {name} | {r['n']} | {pct(r['catch_rate'])} | {pct(r['false_alarm_rate'])} | "
                     f"{r['pr_auc']:.3f} ({r['positive_share']:.3f}) | {points(r['skill'])} | {p_text(r['skill']['p'])} | "
                     f"{r['pauc']:.3f} | {points(r['pauc_skill'])} | {p_text(r['pauc_skill']['p'])} |")
    lines += ["", "## 2. Comparisons against their own baselines", "",
              "| Comparison (vs) | Splits | Caught | Change | Holm p | False alarms | Change | Holm p | PR-AUC | Change "
              "| Holm p | pAUC | Change | Holm p |", "|---|" + "---|" * 13]
    for name, r in comparisons.items():
        cells = []
        for metric in METRICS:
            m = r[metric]
            is_area = metric in ("pr_auc", "pauc")
            value = f"{m['value']:.3f}" if is_area and m["value"] is not None else pct(m["value"])
            change = points(m) if not is_area else (
                "-" if m["mean"] is None or m["low"] is None else f"{m['mean']:+.3f} ({m['low']:+.3f} to {m['high']:+.3f})")
            cells += [value, change, p_text(m.get("p_holm"))]
        lines.append(f"| {name} ({r['baseline']}) | {r['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## 3. Alarms in time by how the shot ended (test shots, pooled over splits)", "",
              "| Runs | " + " | ".join(GROUPS) + " | Recorded minus smooth, per split | p | Candidates like recorded, "
              "vs flux-exhausted | vs smooth |", "|---|" + "---|" * (len(GROUPS) + 4)]
    for name, r in endings.items():
        groups = [f"{pct(r['pooled'][g]['rate'])} ({r['pooled'][g]['alarms']}/{r['pooled'][g]['n']})" for g in GROUPS]
        skill = [points(r["recorded_skill"]), p_text(r["recorded_skill"]["p"])]
        mix = ["(labels call them disruptions)"] * 2 if r["mixture"] is None else [
            f"{pct(r['mixture'][ref]['pooled'], 0)} (per split {share_range(r['mixture'][ref])})"
            for ref in MIXTURE_REFERENCES]
        lines.append(f"| {name} | " + " | ".join(groups + skill + mix) + " |")
    lines += ["", "The mixture share is a rough reading: it assumes disruption-like candidates are alarmed like "
              "recorded disruptions and the rest like the reference group. Read it from runs that never trained on "
              "the candidates as disruptions (clear_drop_candidates leaves them out)."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=config.RESULTS_DIR, help="results folder (default RESULTS_DIR)")
    args = parser.parse_args()
    sets = run_sets(args.results)
    out = {"skill": skill_table(sets), "comparisons": comparison_table(sets), "endings": ending_table(sets)}
    folder = os.path.join(args.results, "paper")
    os.makedirs(folder, exist_ok=True)
    text = markdown(out["skill"], out["comparisons"], out["endings"])
    with open(os.path.join(folder, "numbers.md"), "w") as f:
        f.write(text)
    with open(os.path.join(folder, "numbers.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(text)


if __name__ == "__main__":
    main()
