"""Whole pipeline in one go for a VPS: build the dataset, run E0-E13, describe each result in plain words in the log
and results/REPORT.md, and send one status picture to Telegram at the end.

Work whose code and data have not changed since the last run is reused, not redone (src/resume.py), so a restart
or a commit that only touches docs or the server never retrains anything.
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import build, table_from_report
from src import config, experiment, explain, notify, resume, status_card

MAIN_EXPERIMENTS = ["E0", "E1", "E2", "E3"]
EXTRA_EXPERIMENTS = ["E4", "E5", "E6", "E7", "E8", "E9", "E11", "E12", "E13"]


def step(name, func, *args):
    """Run one pipeline step and time it. Progress goes to the log; only a failure reaches Telegram.
    Returns (result or None, minutes)."""
    notify.log(f"started: {name}")
    start = time.time()
    try:
        result = func(*args)
    except Exception:
        notify.log(f"FAILED: {name}\n{traceback.format_exc()[-1500:]}")
        notify.send(f"{name} failed. /log shows why")
        return None, 0.0
    return result, (time.time() - start) / 60


def run_or_reuse(exp, current, reused):
    """One experiment's outcome: the saved one if the code and data are unchanged (exp is added to reused),
    otherwise a fresh run, described in plain words in the log, and saved. None if it failed."""
    folder = os.path.join(config.RESULTS_DIR, exp)
    outcome = resume.load(folder, current)
    if outcome is not None:
        reused.append(exp)
        return outcome
    label = f"{exp} ({experiment.MODEL_NAMES[exp]})" if exp in experiment.MODEL_NAMES else exp
    outcome, minutes = step(label, experiment.run, exp)
    if outcome is None:
        return None
    outcome = resume.clean_json(outcome)
    resume.save(folder, current, outcome)
    notify.log(f"{explain.describe(exp, outcome)}\n({minutes:.1f} min)")
    return outcome


def write_reports(outcomes, dataset_text, scale):
    """results/SUMMARY.md (main models' test numbers in a few lines) and results/REPORT.md (every experiment in
    plain words, what /report sends)."""
    main = {exp: o for exp, o in outcomes.items() if exp in MAIN_EXPERIMENTS}
    with open(os.path.join(config.RESULTS_DIR, "SUMMARY.md"), "w") as f:
        f.write(f"Run summary (scale={scale}): test shots, alarm level chosen on validation for at most "
                f"{explain.pct(config.MAX_FALSE_ALARM_RATE)} false alarms.\n{explain.summary_lines(main)}\n")
    with open(os.path.join(config.RESULTS_DIR, "REPORT.md"), "w") as f:
        f.write(f"# Every experiment in plain words (scale={scale})\n\n{dataset_text}\n")
        for exp, outcome in outcomes.items():
            f.write(f"\n## {exp}\n{explain.describe(exp, outcome)}\n")


def finish(outcomes, reused, failures, n_experiments, scale, dataset_text):
    """End of run: write the reports, then one picture to Telegram (nothing if every result was reused)."""
    write_reports(outcomes, dataset_text, scale)
    if len(reused) == n_experiments:
        notify.log(f"nothing changed in the science code, settings or data: all {n_experiments} results reused")
        return
    note = f", {len(reused)} reused" if reused else ""
    failed = f", {failures} failed (/log)" if failures else ""
    status_card.send(f"full run finished: {n_experiments - failures} of {n_experiments} experiments{note}{failed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="small")
    parser.add_argument("--skip-build", action="store_true", help="reuse data/processed from an earlier run")
    parser.add_argument("--main-only", action="store_true", help="only E0-E3")
    parser.add_argument("--offline", action="store_true", help="no network: use only shots already on disk")
    parser.add_argument("--from-report", help="use the 400 shots of a Phase 2 per_shot.csv (mastapp.site down)")
    args = parser.parse_args()
    notify.log(f"MAST pipeline starting (scale={args.scale}, label source={config.LABEL_SOURCE})")

    dataset_text = "Dataset reused from an earlier run (--skip-build)."
    if not args.skip_build:
        table = table_from_report(args.from_report) if args.from_report else None
        counts, _ = step(f"build dataset ({args.scale})", build, args.scale, False, table, args.offline)
        if counts is None:
            sys.exit(1)
        dataset_text = explain.dataset_text(counts)
        notify.log(dataset_text)

    current = resume.stamp(resume.processed_files())
    experiments = MAIN_EXPERIMENTS + ([] if args.main_only else EXTRA_EXPERIMENTS)
    outcomes, reused, failures = {}, [], 0
    for exp in experiments:
        outcome = run_or_reuse(exp, current, reused)
        if outcome is None:
            failures += 1
        else:
            outcomes[exp] = outcome

    finish(outcomes, reused, failures, len(experiments), args.scale, dataset_text)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
