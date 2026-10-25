"""Repeat-run comparisons: the E10 recipe (a fresh random split per seed, tuned on validation shots, tested once) with
one change from config.COMPARISONS, so the claims of E11, E13 and E5 get error bars instead of resting on one split.

Each run is paired with the E10 run of the same seed: the same shots in train, validation and test and the same model;
only the features differ. The change is judged with the corrected resampled t-test (Nadeau & Bengio, Machine Learning
52, 2003), because repeated random splits share shots and a plain t-test would be far too sure of itself.

    python scripts/repeat_runs.py --comparison magnetics --seed 3   # one run -> results/E10/comparisons/magnetics/seed_3
    python scripts/repeat_runs.py --summary                         # every comparison so far, in words

The results are stamped on their own (science code, data, this file and config.COMPARISONS), so editing this file
reruns only the comparisons, never E0-E13. The research server runs them after the E10 sweep, one per CPU core.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, dataset, experiment, explain, resume

METRICS = ["catch_rate", "false_alarm_rate"]
MIN_SEEDS = 5  # no range or verdict before this many splits: with 2-3, a few shots that happen to agree look certain


def folder():
    """Where the comparison runs go: inside E10, so they are archived together with the seeds they are paired with."""
    return os.path.join(config.RESULTS_DIR, "E10", "comparisons")


def stamp():
    """Fingerprint of everything a comparison result depends on."""
    extra = resume.file_digest([os.path.abspath(__file__)]) + json.dumps(config.COMPARISONS, sort_keys=True)
    return resume.stamp(resume.processed_files(), extra=extra)


def run(name, seed):
    """One comparison run: the E10 split of this seed, with the change config.COMPARISONS[name]."""
    knob, value = config.COMPARISONS[name]
    windows, shots = experiment.load_tables()
    split = dataset.random_split(shots, seed=seed)
    resplit = windows.assign(split=windows["shot_id"].map(split))
    horizon, source, columns = experiment.sweep_settings(knob, value)
    win = dataset.labelled_windows(resplit, shots, source=source, horizon_ms=horizon)
    out_dir = os.path.join(folder(), name, f"seed_{seed}")
    return experiment.run_model(experiment.best_model_name(), win, "split", columns, out_dir, save=False,
                                with_importance=False)


def test_metrics(path):
    """The test numbers in one run's metrics.json, or None if that run has not finished."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["test"]


def pairs():
    """One row per finished comparison run, with the E10 run of the same seed next to it."""
    rows = []
    for name in config.COMPARISONS:
        base = os.path.join(folder(), name)
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            test = test_metrics(os.path.join(base, entry, "metrics.json"))
            e10 = test_metrics(os.path.join(config.RESULTS_DIR, "E10", entry, "metrics.json"))
            if test is None or e10 is None:
                continue
            row = {"comparison": name, "seed": int(entry[len("seed_"):])}
            for metric in METRICS:
                row.update({metric: test[metric], f"e10_{metric}": e10[metric]})
            rows.append(row)
    return pd.DataFrame(rows, columns=["comparison", "seed"] + METRICS + [f"e10_{m}" for m in METRICS], dtype=object)


def corrected_interval(changes, level=0.95):
    """(mean, low, high) of paired changes over repeated random splits. The variance is scaled by
    (1/n + n_test/n_train) instead of 1/n (Nadeau & Bengio), because the splits overlap. No range below MIN_SEEDS."""
    n = len(changes)
    mean = float(np.mean(changes)) if n else None
    if n < MIN_SEEDS:
        return mean, None, None
    train, _, test = config.SPLIT_FRACTIONS
    half = stats.t.ppf(0.5 + level / 2, n - 1) * np.std(changes, ddof=1) * np.sqrt(1 / n + test / train)
    return mean, float(mean - half), float(mean + half)


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
