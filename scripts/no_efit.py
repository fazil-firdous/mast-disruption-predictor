"""Does the twin test's plasma-signal lead need the EFIT inputs? (docs/preregistration_no_efit.md, fixed before any
model.) Model N = the v3 baseline's inputs minus every EFIT-derived feature, trained like the baseline on splits 1 to 20
and the chronological split, compared with the saved full (A) and machine-side (M) runs by the fixed-lead twin test.

    python scripts/no_efit.py --train      # the runs not done yet, then the summary
    python scripts/no_efit.py --summary

Writes RESULTS_DIR/no_efit/. Reads the window table and the saved v3 runs; never writes into v3/. Outside every stamp.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import fast_magnetics, how_early, v3_explore
from src import config, dataset, experiment, features, notify, server

RECIPE = "no_efit"
EFIT_SIGNALS = ("q95", "li", "betan", "kappa", "zmag", "fgw")
SEEDS = range(1, 21)
PRIMARY_LEAD = 30
NAMES = {"machine": "M, machine-side", "baseline": "A, full", RECIPE: "N, full without EFIT"}


def folder():
    return os.path.join(config.RESULTS_DIR, "no_efit")


def run_dir(seed=None):
    return os.path.join(folder(), "temporal", RECIPE) if seed is None else os.path.join(folder(), RECIPE, f"seed_{seed}")


def columns():
    """The baseline's inputs minus every feature that uses an EFIT signal."""
    return [c for c in v3_explore.recipe_columns("baseline") if not any(features.uses_signal(c, s) for s in EFIT_SIGNALS)]


def run(seed=None, tables=None):
    windows, shots = tables or experiment.load_tables()
    if seed is None:
        win, column = dataset.labelled_windows(windows, shots), "split_temporal"
    else:
        split = dataset.random_split(shots, seed=seed)
        win, column = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots), "split"
    return fast_magnetics.fit(win, column, RECIPE, run_dir(seed), {"seed": 0 if seed is None else seed}, columns())


def collect(load=None):
    main, placebo, heating = how_early.pair_frames(load)
    random, temporal = [], []
    folders = v3_explore.run_folders()
    sources = [("baseline", folders["baseline"], v3_explore.temporal_dir("baseline")),
               ("machine", folders["machine"], v3_explore.temporal_dir("machine")),
               (RECIPE, os.path.join(folder(), RECIPE), run_dir())]
    for name, base, temporal_path in sources:
        for seed, path in how_early.saved_runs(base):
            if seed in SEEDS:
                random += how_early.run_wins(path, name, seed, main, placebo)
        if server.result_saved(os.path.join(temporal_path, "metrics.json")):
            temporal += how_early.run_wins(temporal_path, name, 0, main, placebo)
    return how_early.finish(random, heating), how_early.finish(temporal, heating)


def cells(frame, lead, a, b):
    return fast_magnetics.difference_cells(frame, lead, a, b)


def text(random, temporal, flags=None):
    lines = ["No-EFIT check (docs/preregistration_no_efit.md): N = full model without any EFIT-derived input (q95, li, "
             "beta_N, elongation, axis height, Greenwald fraction). Fixed-lead twin test, common cohort, splits 1 to 20.",
             f"N has {len(columns())} inputs; the full model {len(v3_explore.recipe_columns('baseline'))}.", ""]
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    d = how_early.split_difference(main, PRIMARY_LEAD, RECIPE, "machine")
    if d:
        resolved = d[1] is not None and d[1] > 0
        lines.append(f"PRIMARY: C({PRIMARY_LEAD} ms) of N minus M: {cells(main, PRIMARY_LEAD, RECIPE, 'machine')}. "
                     + ("Resolved and positive: the plasma-signal lead over the machine-side model does not need the EFIT "
                        "inputs." if resolved else
                        "Not resolved: without EFIT the lead is not shown at 30 ms; EFIT (values or schedule, not "
                        "separable) carries part of it."))
    lines.append(f"A minus N at {PRIMARY_LEAD} ms (what EFIT adds): {cells(main, PRIMARY_LEAD, 'baseline', RECIPE)}")
    lines += ["", "C per lead, over splits (M | A | N):"]
    for lead in how_early.COHORT_LEADS:
        split = how_early.over_splits(main, lead) if len(main) else {}
        if all(k in split for k in ("machine", "baseline", RECIPE)):
            lines.append(f"- {lead} ms: " + " | ".join(how_early.rng(*split[k][:3]) for k in ("machine", "baseline", RECIPE)))
    lines += ["", "Differences per lead (N minus M; A minus N), over splits | pooled:"]
    for lead in how_early.COHORT_LEADS:
        lines.append(f"- {lead} ms: {cells(main, lead, RECIPE, 'machine')}; {cells(main, lead, 'baseline', RECIPE)}")
    if flags is not None and "plan" in flags:
        plan = flags.loc[flags["plan"].fillna(False).astype(bool), ["shot_id", "other"]]
        part = main.merge(plan, on=["shot_id", "other"])
        lines.append(f"Plan twins at {PRIMARY_LEAD} ms, N minus M: {cells(part, PRIMARY_LEAD, RECIPE, 'machine')}")
    placebo = random[random["kind"] == "placebo"] if len(random) else random
    if len(placebo):
        split = how_early.over_splits(placebo, PRIMARY_LEAD)
        if RECIPE in split:
            lines.append(f"Placebo at {PRIMARY_LEAD} ms, N: {how_early.rng(*split[RECIPE][:3])}")
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)] if len(temporal) else temporal
    if len(t_main):
        dd = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == PRIMARY_LEAD], RECIPE, "machine", "win")
        if dd:
            lines.append(f"Chronological split, {PRIMARY_LEAD} ms, N minus M, pooled: {how_early.rng(*dd[:3], signed=True)}")
    u = how_early.usual_difference(os.path.join(folder(), RECIPE), v3_explore.run_folders()["baseline"], set(SEEDS))
    if u:
        lines.append(f"Usual metrics, N minus A over {u[1]} splits: pAUC {how_early.rng(*u[0]['pauc'], signed=True)}; caught "
                     f"{how_early.rng(*u[0]['catch_rate'], signed=True)}; false alarms "
                     f"{how_early.rng(*u[0]['false_alarm_rate'], signed=True)}")
    return "\n".join(lines)


def summary(load=None):
    random, temporal = collect(load)
    out = text(random, temporal, fast_magnetics.plan_flags())
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    return out


def train_all(say=print):
    todo = [s for s in list(SEEDS) + [None] if not server.result_saved(os.path.join(run_dir(s), "metrics.json"))]
    say(f"{len(todo)} runs to do; N has {len(columns())} inputs")
    tables = experiment.load_tables() if todo else None
    for i, seed in enumerate(todo, 1):
        start = time.time()
        m = run(seed, tables)
        say(f"[{i}/{len(todo)}] {'chronological' if seed is None else f'seed {seed}'}: pAUC {m['test']['pauc']:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    out = summary()
    say(out)
    notify.send("No-EFIT check: the runs are done\n\n" + out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    if args.train:
        train_all(say)
    elif args.summary:
        say(summary())
    else:
        parser.error("give --train or --summary")


if __name__ == "__main__":
    main()
