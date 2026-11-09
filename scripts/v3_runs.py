"""v3: the tests written down in docs/preregistration_v3.md, on the E10 splits (seeds 1-20 develop, 21-50 confirm).

Three recipes, each trained like the confirmed v2 baseline (candidates left out of training, judged on clear
endings), with one change:
- baseline: the v2 recipe as it is;
- machine: the same labels and shots, but only the machine's programme and solenoid as inputs;
- evidence: the baseline plus the candidates whose post-shot comment names a loss of the plasma within 20 ms of the
  quench (scripts/mast_label_model.py, words_vote 2), labelled as disruptions at their quench time.

Every run saves its test shots' window scores (test_scores.parquet), so the matched-pair twin test (H1) is computed
from them afterwards. The baseline runs also tune and test the alarm with hysteresis (H3).

    python scripts/v3_runs.py --recipe baseline --seed 3    # one run, into RESULTS_DIR/v3/
    python scripts/v3_runs.py --summary                     # H1-H3 so far, in words; writes v3/summary.json

Stamped on its own (science code, data, this file and the label catalogue), like scripts/repeat_runs.py, so v3
never reruns E0-E15, the E10 seeds or the comparisons. The research server runs it after the comparisons.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.paper_numbers import partial_auc
from src import config, dataset, evaluate, experiment, features, models, resume
from src.stats import corrected_interval, corrected_p, holm

RECIPES = {"baseline": {"columns": None, "evidence": False},
           "machine": {"columns": "machine", "evidence": False},
           "evidence": {"columns": None, "evidence": True}}
SEEDS = 50
DEV_SEEDS = range(1, 21)
CONFIRM_SEEDS = range(21, 51)
HYSTERESIS_RATIOS = (1.0, 0.8, 0.6)  # lower threshold = ratio x upper threshold
HYSTERESIS_HOLDS = (1, 2, 3, 4)  # windows (5 ms apart) the score must stay above the lower threshold
PRESENT_RULE = (1.0, config.K_CONSEC)  # the v2 rule: two windows in a row over one threshold
EXACT_RMS = 0.01  # twins whose programmes differ by under 1% (reported, not tested)
FLAT_TOP_SHARE = 0.9  # the flat-top starts when the programmed current first reaches 90% of its peak
EVENTS_DIR = os.path.join(config.RESULTS_DIR, "events")
CATALOGUE = os.path.join(EVENTS_DIR, "catalogue.csv")
TWINS = os.path.join(EVENTS_DIR, "programme_twins.csv")


def folder():
    return os.path.join(config.RESULTS_DIR, "v3")


def kind(recipe):
    """The server's job kind for a recipe."""
    return f"v3-{recipe}"


def recipe_of(job_kind):
    return job_kind.removeprefix("v3-")


def out_dir(recipe, seed):
    return os.path.join(folder(), recipe, f"seed_{seed}")


def jobs():
    """Every v3 run as (kind, seed), seed by seed so the development splits finish first."""
    return [(kind(recipe), seed) for seed in range(1, SEEDS + 1) for recipe in RECIPES]


def stamp():
    """Fingerprint of everything a v3 result depends on."""
    files = [os.path.abspath(__file__)] + ([CATALOGUE] if os.path.exists(CATALOGUE) else [])
    return resume.sweep_stamp(extra="v3\n" + resume.file_digest(files))


def evidence_shots(catalogue):
    """Candidates whose post-shot comment names a loss of the plasma within 20 ms of the quench."""
    keep = (catalogue["group"] == "candidate") & (catalogue["words_vote"] == 2)
    return set(catalogue.loc[keep, "shot_id"].astype(int))


def relabel(win, shot_ids, horizon_ms=config.HORIZON_MS):
    """win with the given shots' TRAINING windows turned into disruptions at their quench time, labelled like a
    recorded disruption (the HORIZON_MS before the quench). Validation and test windows are left as they are."""
    win = win.copy()
    rows = win["shot_id"].isin(shot_ids) & (win["split"] == "train") & np.isfinite(win["t_quench"])
    t_q = win.loc[rows, "t_quench"].to_numpy(dtype=float)
    t_end = win.loc[rows, "t_end"].to_numpy(dtype=float)
    h = horizon_ms / 1000
    win.loc[rows, "disrupted"] = True
    win.loc[rows, "t_disrupt"] = t_q
    win.loc[rows, "label"] = ((t_end >= t_q - h - features.EPS) & (t_end < t_q - features.EPS)).astype(int)
    win.loc[rows, "candidate"] = False
    win.loc[rows, "near_quench"] = False
    return win[~(rows & (win["t_end"] >= win["t_quench"] - features.EPS))].reset_index(drop=True)


def alarm_level(scores, ratio, hold):
    """For each window, the highest upper threshold whose hysteresis alarm fires there: the score hold-1 windows
    earlier was >= upper, and every score since then is >= ratio x upper. NaN for the first hold-1 windows.
    ratio 1 and hold k give the v2 rule (k windows in a row over one threshold, src/alarms.py)."""
    s = np.asarray(scores, dtype=float)
    level = np.full(len(s), np.nan)
    if len(s) < hold:
        return level
    window = np.lib.stride_tricks.sliding_window_view(s, hold)
    level[hold - 1:] = np.minimum(window[:, 0], window.min(axis=1) / ratio)
    return level


def alarm_times(scores, times, thresholds, ratio, hold):
    """First-alarm time for every upper threshold (NaN where no alarm), as src/alarms.alarm_times does."""
    times = np.asarray(times, dtype=float)
    out = np.full(len(thresholds), np.nan)
    if len(times) == 0:
        return out
    best = np.maximum.accumulate(np.nan_to_num(alarm_level(scores, ratio, hold), nan=-np.inf))
    idx = np.searchsorted(best, thresholds, side="left")
    fired = idx < len(times)
    out[fired] = times[idx[fired]]
    return out


def hysteresis_curve(frame, window_scores, ratio, hold):
    """Trade-off curve (as evaluate.tradeoff_curve) of one alarm rule on one set of shots."""
    df = frame[["shot_id", "t_end", "disrupted", "t_disrupt"]].assign(score=np.asarray(window_scores))
    shots, times = [], []
    for shot_id, group in df.sort_values(["shot_id", "t_end"]).groupby("shot_id", sort=True):
        shots.append({"shot_id": shot_id, "disrupted": bool(group["disrupted"].iloc[0]),
                      "t_disrupt": float(group["t_disrupt"].iloc[0])})
        times.append(alarm_times(group["score"].to_numpy(), group["t_end"].to_numpy(), evaluate.THRESHOLDS,
                                 ratio, hold))
    return evaluate.tradeoff_curve(pd.DataFrame(shots), np.array(times))


def tune_hysteresis(val, val_scores, test, test_scores):
    """H3: the (ratio, hold) with the best validation pAUC (ties go to the present rule, then to the order of the
    grid), and the test pAUC of that rule and of the present rule."""
    grid = [PRESENT_RULE] + [(r, h) for r in HYSTERESIS_RATIOS for h in HYSTERESIS_HOLDS if (r, h) != PRESENT_RULE]
    val_pauc = {rule: partial_auc(hysteresis_curve(val, val_scores, *rule)) for rule in grid}
    chosen = max(grid, key=lambda rule: (val_pauc[rule], -grid.index(rule)))
    return {"ratio": chosen[0], "hold": chosen[1], "val_pauc": val_pauc[chosen],
            "val_pauc_present": val_pauc[PRESENT_RULE],
            "test_pauc": partial_auc(hysteresis_curve(test, test_scores, *chosen)),
            "test_pauc_present": partial_auc(hysteresis_curve(test, test_scores, *PRESENT_RULE))}


def run(recipe, seed):
    """One v3 run: the E10 split of this seed, the recipe's change, tuned on validation, tested once."""
    spec = RECIPES[recipe]
    windows, shots = experiment.load_tables()
    split = dataset.random_split(shots, seed=seed)
    win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots)
    if spec["evidence"]:
        win = relabel(win, evidence_shots(pd.read_csv(CATALOGUE)))
    train, val_all, test_all = experiment.split_frames(win, "split")
    train = train[~train["candidate"]].reset_index(drop=True)
    val, test = experiment.judged(val_all, "clear"), experiment.judged(test_all, "clear")
    columns = experiment.usable(features.model_features(spec["columns"]), train)
    best = experiment.tune(config.SWEEP_MODEL, train, val, columns)
    path = out_dir(recipe, seed)
    os.makedirs(path, exist_ok=True)
    metrics = {"recipe": recipe, "seed": seed, "params": best["params"], "threshold": best["threshold"],
               "n_train_windows": len(experiment.training_rows(train)),
               "n_evidence_shots": int(train.loc[train["disrupted"] & ~train["note"], "shot_id"].nunique())}
    metrics["test"] = experiment.test_once(best, test, columns, path)
    metrics["test"]["pauc"] = partial_auc(pd.read_csv(os.path.join(path, "tradeoff_test.csv")))
    metrics["test"]["alarms_by_ending"] = experiment.ending_alarms(best, test_all, columns)
    test_scores = models.scores(best["model"], test_all[columns])
    pd.DataFrame({"shot_id": test_all["shot_id"].astype("int32"), "t_end": test_all["t_end"].astype("float32"),
                  "score": np.asarray(test_scores, dtype="float32")}).to_parquet(
        os.path.join(path, "test_scores.parquet"), index=False)
    if recipe == "baseline":
        metrics["hysteresis"] = tune_hysteresis(val, models.scores(best["model"], val[columns]), test,
                                                models.scores(best["model"], test[columns]))
    experiment.write_json(metrics, os.path.join(path, "metrics.json"))  # last: marks the run as finished
    return metrics


def flat_top_start(frame):
    """First time the programmed current reaches FLAT_TOP_SHARE of its peak, or None."""
    if frame is None or "ip_ref" not in frame:
        return None
    t, ref = frame["time"].to_numpy(dtype=float), np.abs(frame["ip_ref"].to_numpy(dtype=float))
    ok = np.isfinite(t) & np.isfinite(ref)
    if ok.sum() < 10 or ref[ok].max() <= 0:
        return None
    return float(t[ok][np.argmax(ref[ok] >= FLAT_TOP_SHARE * ref[ok].max())])


def concordance(scores, twins, t_quench, t_start):
    """H1 for one run: (C, number of pairs, C over the exact-programme twins, their number). For each twin pair with
    both shots in scores, each shot's score is its highest window score from the flat-top start (t_start of the lost
    shot, or the first window if unknown) to MIN_WARNING_MS before the lost shot's quench. 1 if the lost shot scores
    higher, 0.5 for a tie, 0 otherwise."""
    by_shot = {s: g for s, g in scores.groupby("shot_id")}
    wins, exact = [], []
    for row in twins.itertuples():
        lost, kept = by_shot.get(row.shot_id), by_shot.get(row.other)
        tq = t_quench.get(row.shot_id, np.nan)
        if lost is None or kept is None or not np.isfinite(tq):
            continue
        start = t_start.get(row.shot_id)
        t_from, t_to = -np.inf if start is None else start, tq - config.MIN_WARNING_MS / 1000
        a = lost.loc[(lost["t_end"] >= t_from) & (lost["t_end"] <= t_to), "score"]
        b = kept.loc[(kept["t_end"] >= t_from) & (kept["t_end"] <= t_to), "score"]
        if a.empty or b.empty:
            continue
        win = 1.0 if a.max() > b.max() else 0.5 if a.max() == b.max() else 0.0
        wins.append(win)
        if row.rms < EXACT_RMS:
            exact.append(win)
    return (float(np.mean(wins)) if wins else np.nan, len(wins), float(np.mean(exact)) if exact else np.nan,
            len(exact))


def one_sided(changes):
    """One-sided p-value of the corrected resampled t-test for a positive mean change, or None."""
    p = corrected_p(changes)
    if p is None:
        return None
    return p / 2 if np.mean(changes) > 0 else 1 - p / 2


def runs(load=None):
    """One row per finished v3 run: recipe, seed, test pAUC and catch rate, hysteresis pAUCs, and H1's C."""
    from src import extras
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(TWINS) if os.path.exists(TWINS) else None  # written by scripts/twin_check.py
    labels_csv = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
    t_quench = (pd.read_csv(labels_csv).set_index("shot_id")["t_quench"].to_dict()
                if twins is not None and os.path.exists(labels_csv) else {})
    starts = {}
    rows = []
    for recipe in RECIPES:
        base = os.path.join(folder(), recipe)
        for entry in sorted(os.listdir(base)) if os.path.isdir(base) else []:
            path = os.path.join(base, entry, "metrics.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                m = json.load(f)
            row = {"recipe": recipe, "seed": m["seed"], "pauc": m["test"]["pauc"],
                   "catch_rate": m["test"]["catch_rate"], "false_alarm_rate": m["test"]["false_alarm_rate"]}
            if "hysteresis" in m:
                row.update({f"hyst_{k}": v for k, v in m["hysteresis"].items()})
            scores_path = os.path.join(base, entry, "test_scores.parquet")
            if twins is not None and os.path.exists(scores_path):
                scores = pd.read_parquet(scores_path)
                ids = set(scores["shot_id"])
                pairs = twins[twins["shot_id"].isin(ids) & twins["other"].isin(ids)]
                for shot in pairs["shot_id"]:
                    if shot not in starts:
                        starts[shot] = flat_top_start(load(shot))
                row["C"], row["n_pairs"], row["C_exact"], row["n_exact"] = concordance(scores, pairs, t_quench, starts)
            rows.append(row)
    return pd.DataFrame(rows)


def tests(table, seeds):
    """H1a, H1b, H2, H3 on the given seeds: mean change, corrected 95% range, one-sided p; Holm over H2 and H3."""
    part = table[table["seed"].isin(list(seeds))]
    wide = {r: g.set_index("seed") for r, g in part.groupby("recipe")}
    base, machine, evidence = (wide.get(r) for r in ("baseline", "machine", "evidence"))
    out = {}

    def add(name, changes, what):
        changes = np.asarray(pd.Series(changes).dropna(), dtype=float)
        mean, low, high = corrected_interval(changes) if len(changes) else (None, None, None)
        out[name] = {"what": what, "n": len(changes), "mean": mean, "low": low, "high": high,
                     "p": one_sided(changes) if len(changes) else None}

    if base is not None and "C" in base:
        add("H1a", base["C"] - 0.5, "C of the baseline minus 0.5 (twins told apart)")
        if machine is not None and "C" in machine:
            both = base[["C"]].join(machine[["C"]], rsuffix="_m", how="inner")
            add("H1b", both["C"] - both["C_m"], "C of the baseline minus C of the machine-only model")
    if base is not None and evidence is not None:
        both = base[["pauc"]].join(evidence[["pauc"]], rsuffix="_e", how="inner")
        add("H2", both["pauc_e"] - both["pauc"], "pAUC of the evidence recipe minus the baseline's")
    if base is not None and "hyst_test_pauc" in base:
        add("H3", base["hyst_test_pauc"] - base["hyst_test_pauc_present"],
            "test pAUC of the hysteresis alarm minus the present alarm's (baseline model)")
    family = [k for k in ("H2", "H3") if k in out]
    for k, p in zip(family, holm([out[k]["p"] for k in family])):
        out[k]["p_holm"] = p
    return out


def summary(table=None):
    """The development and confirmation results; also writes v3/runs.csv and v3/summary.json."""
    table = runs() if table is None else table
    result = {"dev": tests(table, DEV_SEEDS), "confirm": tests(table, CONFIRM_SEEDS),
              "n_runs": table.groupby("recipe").size().to_dict() if len(table) else {}}
    for recipe, part in table.groupby("recipe") if len(table) else []:
        result[f"mean_{recipe}"] = {c: float(part[c].mean()) for c in ("pauc", "catch_rate", "false_alarm_rate", "C",
                                                                         "C_exact", "n_pairs") if c in part}
    if len(table):
        os.makedirs(folder(), exist_ok=True)
        table.to_csv(os.path.join(folder(), "runs.csv"), index=False)
        experiment.write_json(result, os.path.join(folder(), "summary.json"))
    return result


def text(result):
    """The summary in words."""
    lines = [f"v3 runs finished: {result['n_runs']}."]
    for recipe in RECIPES:
        m = result.get(f"mean_{recipe}")
        if m:
            lines.append(f"- {recipe}: pAUC {m.get('pauc', np.nan):.3f}, caught {m.get('catch_rate', np.nan):.1%}, "
                         f"false alarms {m.get('false_alarm_rate', np.nan):.1%}, twin C {m.get('C', np.nan):.3f} "
                         f"(exact twins {m.get('C_exact', np.nan):.3f}), pairs per split {m.get('n_pairs', np.nan):.0f}")
    for stage, label in (("dev", "Development, splits 1-20 (chooses nothing)"),
                         ("confirm", "Confirmation, splits 21-50 (the pre-registered verdicts)")):
        lines += ["", label + ":"]
        for name, r in result[stage].items():
            if r["mean"] is None:
                lines.append(f"- {name}: {r['n']} splits, not enough yet")
                continue
            rng = f"{r['low']:+.3f} to {r['high']:+.3f}" if r["low"] is not None else "no range yet"
            p = f"p {r['p']:.3f}" if r["p"] is not None else "no p yet"
            if "p_holm" in r and r["p_holm"] is not None:
                p += f", Holm {r['p_holm']:.3f}"
            lines.append(f"- {name} ({r['what']}): {r['mean']:+.3f} ({rng}) over {r['n']} splits, {p}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recipe", choices=list(RECIPES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        print(text(summary()))
        return
    if args.recipe is None or args.seed is None:
        parser.error("give --recipe and --seed, or --summary")
    m = run(args.recipe, args.seed)
    print(json.dumps({k: v for k, v in m.items() if k != "test"}, indent=2, default=str))


if __name__ == "__main__":
    main()
