"""Whole pipeline in one go for a VPS: build the dataset, run E0-E7, report each step to Telegram."""
import argparse
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import build, table_from_report
from src import config, experiment, notify

MAIN_EXPERIMENTS = ["E0", "E1", "E2", "E3"]
EXTRA_EXPERIMENTS = ["E4", "E5", "E6", "E7", "E8", "E9"]


def pct(x):
    """0.734 -> '73.4%'; None -> 'n/a'."""
    return "n/a" if x is None else f"{100 * x:.1f}%"


def headline(exp, metrics):
    """One-paragraph Telegram summary of a model experiment's test numbers."""
    test = metrics["test"]
    ci = test["ci95"]
    return (f"{exp} {metrics['model']} done (test, {test['n_disrupted']} disrupted / {test['n_clean']} clean shots)\n"
            f"catch {pct(test['catch_rate'])} [{pct(ci['catch_rate'][0])}-{pct(ci['catch_rate'][1])}]\n"
            f"false alarms {pct(test['false_alarm_rate'])} | tardy {pct(test['tardy_rate'])} | "
            f"missed {pct(test['miss_rate'])}\n"
            f"median warning {test['warning_median_ms']:.0f} ms | threshold {metrics['threshold']:.3f}"
            + ("" if metrics["val"]["meets_false_alarm_limit"] else "\nWARNING: no threshold met the false-alarm limit on validation"))


def step(name, func, *args):
    """Run one pipeline step, timing it and sending a message on success or failure. Returns the result or None."""
    notify.send(f"started: {name}")
    start = time.time()
    try:
        result = func(*args)
    except Exception:
        notify.send(f"FAILED: {name}\n{traceback.format_exc()[-1500:]}")
        return None
    notify.send(f"finished: {name} in {(time.time() - start) / 60:.1f} min")
    return result


def summary_table(results):
    """Markdown table of the main experiments' test numbers, for results/SUMMARY.md and Telegram."""
    lines = ["| exp | model | catch | false alarm | tardy | missed | median warning (ms) |",
             "|---|---|---|---|---|---|---|"]
    for exp, m in results.items():
        t = m["test"]
        lines.append(f"| {exp} | {m['model']} | {pct(t['catch_rate'])} | {pct(t['false_alarm_rate'])} | "
                     f"{pct(t['tardy_rate'])} | {pct(t['miss_rate'])} | {t['warning_median_ms']:.0f} |")
    return "\n".join(lines)


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
        counts = step(f"build dataset ({args.scale})", build, args.scale, False, table, args.offline)
        if counts is None:
            sys.exit(1)
        notify.send(f"dataset: {counts}")

    results, failures = {}, 0
    for exp in MAIN_EXPERIMENTS:
        metrics = step(f"{exp} ({experiment.MODEL_NAMES[exp]})", experiment.run, exp)
        if metrics is None:
            failures += 1
        else:
            results[exp] = metrics
            notify.send(headline(exp, metrics))
            notify.send_file(os.path.join(config.RESULTS_DIR, exp, "tradeoff.png"), f"{exp} trade-off curve")

    if not args.main_only:
        for exp in EXTRA_EXPERIMENTS:
            outcome = step(exp, experiment.run, exp)
            if outcome is None:
                failures += 1
            else:
                notify.send(f"{exp} result:\n{str(outcome)[:3500]}")

    table = summary_table(results)
    with open(os.path.join(config.RESULTS_DIR, "SUMMARY.md"), "w") as f:
        f.write(f"# Run summary (scale={args.scale})\n\nTest split, threshold chosen on validation for "
                f"<= {pct(config.MAX_FALSE_ALARM_RATE)} false alarms.\n\n{table}\n")
    archive = shutil.make_archive(os.path.join(config.DATA_DIR, "results"), "zip", config.RESULTS_DIR)
    notify.send(f"ALL DONE ({failures} failed steps)\n{table}")
    notify.send_file(archive, "all results (metrics.json, plots)")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
