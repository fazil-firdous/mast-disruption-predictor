"""Repeat-run comparisons: the E10 recipe (a fresh random split per seed, tuned on validation shots, tested once) with
one change from config.COMPARISONS, so the claims of E11, E13-E15 get error bars instead of resting on one split.

Each run is paired with its baseline run of the same seed (E10, or the comparison named in COMPARISON_BASELINES): the
same shots in train, validation and test and the same model; only the change differs. The change is judged with the
corrected resampled t-test (src/stats.py), because repeated random splits share shots and a plain t-test would be
far too sure of itself.

    python scripts/repeat_runs.py --comparison magnetics --seed 3   # one run, into results/E10/comparisons/
    python scripts/repeat_runs.py --summary                         # every comparison so far, in words

The results are stamped on their own (science code, data, this file and config.COMPARISONS), so editing this file
reruns only the comparisons, never E0-E15. The research server runs them after the first COMPARISON_SEEDS E10 seeds,
one per CPU core, before the remaining E10 seeds.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, dataset, experiment, explain, resume
from src.stats import corrected_interval

METRICS = ["catch_rate", "false_alarm_rate"]


def folder():
    """Where the comparison runs go: inside E10, so they are archived together with the seeds they are paired with."""
    return os.path.join(config.RESULTS_DIR, "E10", "comparisons")


def stamp():
    """Fingerprint of everything a comparison result depends on."""
    extra = resume.file_digest([os.path.abspath(__file__)]) + json.dumps(
        [config.COMPARISONS, config.COMPARISON_BASELINES], sort_keys=True)
    return resume.stamp(resume.processed_files(), extra=extra)


def run(name, seed):
    """One comparison run: the E10 split of this seed, with the change config.COMPARISONS[name]."""
    knob, value = config.COMPARISONS[name]
    windows, shots = experiment.load_tables()
    split = dataset.random_split(shots, seed=seed)
    resplit = windows.assign(split=windows["shot_id"].map(split))
    s = experiment.sweep_settings(knob, value)
    win = dataset.labelled_windows(resplit, shots, source=s["source"], horizon_ms=s["horizon"])
    out_dir = os.path.join(folder(), name, f"seed_{seed}")
    return experiment.run_model(experiment.best_model_name(), win, "split", s["columns"], out_dir, save=False,
                                with_importance=False, judge=s["judge"], drop_candidates=s["drop_candidates"])


def baseline_folder(name):
    """Where the paired baseline runs of a comparison live: E10's seeds, or another comparison's."""
    other = config.COMPARISON_BASELINES.get(name)
    return os.path.join(folder(), other) if other else os.path.join(config.RESULTS_DIR, "E10")


def test_metrics(path):
    """The test numbers in one run's metrics.json, or None if that run has not finished."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["test"]


def pairs():
    """One row per finished comparison run, with its baseline run of the same seed next to it (columns e10_*,
    whichever the baseline is)."""
    rows = []
    for name in config.COMPARISONS:
        base = os.path.join(folder(), name)
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            test = test_metrics(os.path.join(base, entry, "metrics.json"))
            e10 = test_metrics(os.path.join(baseline_folder(name), entry, "metrics.json"))
            if test is None or e10 is None:
                continue
            row = {"comparison": name, "seed": int(entry[len("seed_"):])}
            for metric in METRICS:
                row.update({metric: test[metric], f"e10_{metric}": e10[metric]})
            rows.append(row)
    return pd.DataFrame(rows, columns=["comparison", "seed"] + METRICS + [f"e10_{m}" for m in METRICS], dtype=object)


def summary():
    """Per comparison: finished runs, mean test catch and false-alarm rates next to E10's on the same splits, and
    the corrected 95% range of each change. Also writes comparisons/summary.csv."""
    out = {}
    for name, group in pairs().groupby("comparison", sort=False):
        row = {"n_seeds": len(group)}
        for metric in METRICS:
            both = group[[metric, f"e10_{metric}"]].dropna().astype(float)
            mean, low, high = corrected_interval((both[metric] - both[f"e10_{metric}"]).to_numpy())
            row.update({metric: both[metric].mean(), f"e10_{metric}": both[f"e10_{metric}"].mean(),
                        f"{metric}_change": mean, f"{metric}_change_low": low, f"{metric}_change_high": high})
        out[name] = row
    if out:
        pd.DataFrame.from_dict(out, orient="index").to_csv(os.path.join(folder(), "summary.csv"),
                                                           index_label="comparison")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comparison", choices=list(config.COMPARISONS))
    parser.add_argument("--seed", type=int, help="split seed, as in E10")
    parser.add_argument("--summary", action="store_true", help="every comparison so far, in words")
    args = parser.parse_args()
    if args.summary:
        print(explain.comparisons_text(summary()))
        return
    if args.comparison is None or args.seed is None:
        parser.error("give --comparison and --seed, or --summary")
    metrics = run(args.comparison, args.seed)
    print(json.dumps({k: v for k, v in metrics.items() if k != "trials"}, indent=2, default=str))


if __name__ == "__main__":
    main()
