"""Whole pipeline in one go for a VPS: build the dataset, run E0-E13, report each result to Telegram in plain words.

Work whose code and data have not changed since the last run is reused, not redone (src/resume.py), so a restart
or a commit that only touches docs or the server never retrains anything.
"""
import argparse
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import build, table_from_report
from src import config, experiment, explain, notify, resume

MAIN_EXPERIMENTS = ["E0", "E1", "E2", "E3"]
EXTRA_EXPERIMENTS = ["E4", "E5", "E6", "E7", "E8", "E9", "E11", "E12", "E13"]


def step(name, func, *args):
    """Run one pipeline step and time it; a failure is sent to Telegram. Returns (result or None, minutes)."""
    notify.send(f"started: {name}")
    start = time.time()
    try:
        result = func(*args)
    except Exception:
        notify.send(f"FAILED: {name}\n{traceback.format_exc()[-1500:]}")
        return None, 0.0
    return result, (time.time() - start) / 60


def run_or_reuse(exp, current, reused):
    """One experiment's outcome: the saved one if the code and data are unchanged (exp is added to reused),
    otherwise a fresh run that is sent to Telegram in plain words and saved. None if it failed."""
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
    notify.send(f"{explain.describe(exp, outcome)}\n({minutes:.1f} min)")
    if exp in experiment.MODEL_NAMES:
        notify.send_file(os.path.join(folder, "tradeoff.png"),
                         f"{exp}: caught (up) against false alarms (right) for every alarm level")
    return outcome


def write_summary(results, scale):
    """results/SUMMARY.md: the main models' test numbers in plain lines (also what /results sends)."""
    with open(os.path.join(config.RESULTS_DIR, "SUMMARY.md"), "w") as f:
        f.write(f"Run summary (scale={scale}): test shots, alarm level chosen on validation for at most "
                f"{explain.pct(config.MAX_FALSE_ALARM_RATE)} false alarms.\n{explain.summary_lines(results)}\n")


def finish(results, reused, failures, n_experiments, scale):
    """End-of-run messages. If every result was reused, one short line; otherwise the summary and a zip."""
    if len(reused) == n_experiments:
        notify.send(f"Nothing changed in the science code, settings or data since the last run: all "
                    f"{n_experiments} results reused, nothing retrained. /results shows them.")
        return
    write_summary(results, scale)
    archive = shutil.make_archive(os.path.join(config.DATA_DIR, "results"), "zip", config.RESULTS_DIR)
    done = f"ALL DONE ({failures} failed steps)"
    if reused:
        done += f"\nReused without retraining (code and data unchanged): {', '.join(reused)}"
    notify.send(f"{done}\n{explain.summary_lines(results)}")
    notify.send_file(archive, "all results (metrics.json, plots)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="small")
    parser.add_argument("--skip-build", action="store_true", help="reuse data/processed from an earlier run")
    parser.add_argument("--main-only", action="store_true", help="only E0-E3")
    parser.add_argument("--offline", action="store_true", help="no network: use only shots already on disk")
    parser.add_argument("--from-report", help="use the 400 shots of a Phase 2 per_shot.csv (mastapp.site down)")
    args = parser.parse_args()
    notify.send(f"MAST pipeline starting (scale={args.scale}, label source={config.LABEL_SOURCE})")

    if not args.skip_build:
        table = table_from_report(args.from_report) if args.from_report else None
        counts, _ = step(f"build dataset ({args.scale})", build, args.scale, False, table, args.offline)
        if counts is None:
            sys.exit(1)
        notify.send(explain.dataset_text(counts))

    current = resume.stamp(resume.processed_files())
    experiments = MAIN_EXPERIMENTS + ([] if args.main_only else EXTRA_EXPERIMENTS)
    results, reused, failures = {}, [], 0
    for exp in experiments:
        outcome = run_or_reuse(exp, current, reused)
        if outcome is None:
            failures += 1
        elif exp in MAIN_EXPERIMENTS:
            results[exp] = outcome

    finish(results, reused, failures, len(experiments), args.scale)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
