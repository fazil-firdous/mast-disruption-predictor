"""Tune a model on validation shots, test it once, and write metrics and plots for one experiment (SPEC 14)."""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import conformal, config, dataset, evaluate, features, models, pu, registry
from src.data import load_shot

MODEL_NAMES = {"E0": "baseline", "E1": "logreg", "E2": "forest", "E3": "hgb"}
MAX_IMPORTANCE_WINDOWS = 20000  # permutation importance on more windows is slow and changes little


def load_tables():
    """The window table and shot table written by scripts/build_dataset.py."""
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"))
    return windows, pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))


def split_frames(win, split_column):
    """(train, val, test) windows for 'split' (random) or 'split_temporal'."""
    return tuple(win[win[split_column] == name].reset_index(drop=True) for name in ("train", "val", "test"))


def training_rows(train):
    """Training windows, minus the near-quench clean ones when NEAR_QUENCH_CLEAN is 'drop'."""
    if config.NEAR_QUENCH_CLEAN == "drop":
        return train[~train["near_quench"]]
    return train


def score_split(model, frame, columns):
    """Scores, trade-off curve, shot table and alarm matrix for one split."""
    window_scores = models.scores(model, frame[columns])
    shots, alarm_matrix = evaluate.shot_alarm_table(frame, window_scores)
    curve = evaluate.tradeoff_curve(shots, alarm_matrix)
    return window_scores, curve, shots, alarm_matrix


def tune(name, train, val, columns):
    """Fit every grid combination on train and keep the one with the best validation catch rate
    at the threshold that gives at most MAX_FALSE_ALARM_RATE false alarms (ties broken by PR-AUC).
    A combination that cannot reach the false-alarm limit only wins if none can."""
    fit_rows = training_rows(train)
    best, trials = None, []
    for params in models.grid(name):
        model = models.make_model(name, params).fit(fit_rows[columns], fit_rows["label"])
        window_scores, curve, _, _ = score_split(model, val, columns)
        threshold = evaluate.choose_threshold(curve)
        point = curve[curve["threshold"] == threshold].iloc[0]
        pr_auc = evaluate.window_metrics(val["label"], window_scores, threshold)["pr_auc"]
        trial = {"params": params, "threshold": threshold, "val_catch_rate": float(point["catch_rate"]),
                 "val_false_alarm_rate": float(point["false_alarm_rate"]), "val_pr_auc": pr_auc,
                 "val_meets_limit": evaluate.meets_limit(point["false_alarm_rate"])}
        trials.append(trial)
        key = (trial["val_meets_limit"], trial["val_catch_rate"], np.nan_to_num(pr_auc))
        if best is None or key > best["key"]:
            best = {"key": key, "model": model, "curve": curve, **trial}
    best.pop("key")
    best["trials"] = trials
    return best


def importance(model, val, columns, seed=config.RANDOM_SEED):
    """Permutation importance on validation windows: drop in PR-AUC when one feature is shuffled."""
    sample = val.sample(min(len(val), MAX_IMPORTANCE_WINDOWS), random_state=seed)
    result = permutation_importance(model, sample[columns], sample["label"], scoring="average_precision",
                                    n_repeats=config.PERMUTATION_REPEATS, random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": columns, "mean": result.importances_mean, "std": result.importances_std})


def pick_examples(shots, result):
    """Up to 6 shot ids for the example plot: 3 caught, 1 missed, 1 tardy, 1 false alarm (SPEC 11)."""
    wanted = [("caught", 3), ("missed", 1), ("tardy", 1), ("false_alarm", 1)]
    picks = []
    for outcome, k in wanted:
        ids = shots.loc[result == outcome, "shot_id"].to_numpy()
        picks += [(int(s), outcome) for s in ids[:k]]
    return picks


def plot_examples(test, window_scores, shots, t_alarm, result, threshold, path):
    """Example shots with |Ip|, Prad, score, threshold, t_alarm (blue dotted) and t_disrupt (red)."""
    picks = pick_examples(shots, result)
    if not picks:
        return
    fig, axes = plt.subplots(2 * len(picks), 1, figsize=(7, 2.6 * len(picks)))
    scored = test.assign(score=window_scores)
    info = shots.set_index("shot_id")
    alarm_of = dict(zip(shots["shot_id"], t_alarm))
    for i, (shot_id, outcome) in enumerate(picks):
        win = scored[scored["shot_id"] == shot_id].sort_values("t_end")
        evaluate.plot_example(load_shot(shot_id), win, threshold, alarm_of[shot_id], info.at[shot_id, "t_disrupt"],
                              f"shot {shot_id}: {outcome}", (axes[2 * i], axes[2 * i + 1]))
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def test_once(best, test, columns, out_dir):
    """Score the test split once at the validation-chosen threshold; returns metrics and writes plots."""
    window_scores, curve, shots, alarm_matrix = score_split(best["model"], test, columns)
    j = int(np.argmin(np.abs(evaluate.THRESHOLDS - best["threshold"])))
    result, warning_ms = evaluate.outcomes(shots, alarm_matrix[:, j])
    metrics = evaluate.shot_metrics(result, warning_ms)
    metrics["ci95"] = evaluate.bootstrap_ci(result, warning_ms)
    metrics["window"] = evaluate.window_metrics(test["label"], window_scores, best["threshold"])
    curve.to_csv(os.path.join(out_dir, "tradeoff_test.csv"), index=False)
    evaluate.plot_tradeoff({"test": curve, "validation": best["curve"]}, os.path.join(out_dir, "tradeoff.png"),
                           {"chosen": (metrics["false_alarm_rate"], metrics["catch_rate"])})
    if np.isin(result, ["caught", "tardy"]).any():
        evaluate.plot_warning_hist(warning_ms, result, os.path.join(out_dir, "warning_times.png"))
    plot_examples(test, window_scores, shots, alarm_matrix[:, j], result, best["threshold"],
                  os.path.join(out_dir, "examples.png"))
    return metrics


def write_json(obj, path):
    """Write obj as indented JSON; NaN becomes null so the file is valid JSON."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    with open(path, "w") as f:
        json.dump(clean(obj), f, indent=2)


def config_snapshot():
    """Every UPPERCASE setting in config.py, so each result folder records the settings that made it.

    Anything named like a secret is left out, because results get zipped and sent to Telegram.
    """
    return {k: v for k, v in vars(config).items() if k.isupper() and "TOKEN" not in k and "SECRET" not in k}


def run_model(name, win, split_column, columns, out_dir, use_test=True, save=True, with_importance=True):
    """Tune one model, optionally test it, write metrics.json and config.json to out_dir, return the metrics.

    With use_test and save, the fitted model is also saved to MODELS_DIR (see src/registry.py).
    """
    os.makedirs(out_dir, exist_ok=True)
    train, val, test = split_frames(win, split_column)
    best = tune(name, train, val, columns)
    metrics = {"model": name, "split": split_column, "params": best["params"], "threshold": best["threshold"],
               "n_train_windows": len(training_rows(train)), "n_val_windows": len(val),
               "val": {"catch_rate": best["val_catch_rate"], "false_alarm_rate": best["val_false_alarm_rate"],
                       "pr_auc": best["val_pr_auc"], "meets_false_alarm_limit": best["val_meets_limit"]},
               "trials": best["trials"]}
    if use_test:
        metrics["test"] = test_once(best, test, columns, out_dir)
    if use_test and save:
        tag = os.path.relpath(out_dir, config.RESULTS_DIR).replace(os.sep, "_")
        registry.save(best["model"], tag, name, columns, best["threshold"],
                      {"val_catch_rate": best["val_catch_rate"], "test_catch_rate": metrics["test"]["catch_rate"],
                       "test_false_alarm_rate": metrics["test"]["false_alarm_rate"], "params": best["params"]})
    if name != "baseline" and with_importance:
        imp = importance(best["model"], val, columns)
        imp.to_csv(os.path.join(out_dir, "importance.csv"), index=False)
        evaluate.plot_importance(imp, os.path.join(out_dir, "importance.png"))
    write_json(metrics, os.path.join(out_dir, "metrics.json"))
    write_json(config_snapshot(), os.path.join(out_dir, "config.json"))
    return metrics


def best_model_name():
    """The ML model (E1-E3) with the best validation catch rate so far, or 'hgb' if none has run."""
    best, best_rate = "hgb", -1.0
    for exp in ("E1", "E2", "E3"):
        path = os.path.join(config.RESULTS_DIR, exp, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                rate = json.load(f)["val"]["catch_rate"]
            if rate is not None and rate > best_rate:
                best, best_rate = MODEL_NAMES[exp], rate
    return best


def run(exp, split="random", model=None, seed=None):
    """Run one experiment id from SPEC 14 and return a dict of its headline numbers."""
    windows, shots = load_tables()
    split_column = "split" if split == "random" else "split_temporal"
    out_dir = os.path.join(config.RESULTS_DIR, exp if split == "random" else f"{exp}_{split}")
    all_columns = features.feature_names()
    if exp in MODEL_NAMES:
        win = dataset.labelled_windows(windows, shots)
        return run_model(MODEL_NAMES[exp], win, split_column, all_columns, out_dir)
    name = model or best_model_name()
    if exp == "E4":
        return sweep(name, windows, shots, split_column, out_dir, "horizon_ms", [10, 20, 30, 50])
    if exp == "E5":
        return sweep(name, windows, shots, split_column, out_dir, "drop_signal", [None] + list(features.SIGNALS))
    if exp == "E6":
        win = dataset.labelled_windows(windows, shots)
        return {m: run_model(m, win, "split_temporal", all_columns, os.path.join(out_dir, m))["test"]
                for m in ("baseline", name)}
    if exp == "E7":
        return sweep(name, windows, shots, split_column, out_dir, "label_source", list(dataset.LABEL_SOURCES))
    if exp == "E8":
        return run_conformal(name, dataset.labelled_windows(windows, shots), split_column, all_columns, out_dir)
    if exp == "E9":
        return run_pu(dataset.labelled_windows(windows, shots), shots, split_column, all_columns, out_dir)
    if exp == "E10":
        return run_seed(name, windows, shots, seed or 1)
    raise ValueError(f"unknown experiment {exp!r}")


def sweep(name, windows, shots, split_column, out_dir, knob, values):
    """Re-tune the model for each value of one knob and compare on VALIDATION shots only.

    The test split is kept for final numbers (SPEC 8.4), so sweeps never look at it.
    """
    rows = []
    for value in values:
        horizon = value if knob == "horizon_ms" else config.HORIZON_MS
        source = value if knob == "label_source" else None
        columns = features.features_without(value) if knob == "drop_signal" and value else features.feature_names()
        win = dataset.labelled_windows(windows, shots, source=source, horizon_ms=horizon)
        sub_dir = os.path.join(out_dir, f"{knob}_{value}")
        metrics = run_model(name, win, split_column, columns, sub_dir, use_test=False)
        rows.append({knob: value if value is not None else "none", "model": name, **metrics["val"]})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return table.to_dict(orient="records")


def metrics_at(frame, window_scores, threshold):
    """Shot metrics of one split at one threshold (inf means never alarm)."""
    shots, alarm_matrix = evaluate.shot_alarm_table(frame, window_scores, np.array([threshold]))
    result, warning_ms = evaluate.outcomes(shots, alarm_matrix[:, 0])
    return evaluate.shot_metrics(result, warning_ms)


def split_validation(val, seed=config.RANDOM_SEED):
    """Split validation windows by shot into a tuning half and a calibration half (stratified by label)."""
    val_shots = val.groupby("shot_id")["disrupted"].first().reset_index()
    halves = dataset.random_split(val_shots, fractions=(0.5, 0.5, 0.0), seed=seed)
    half = val["shot_id"].map(halves)
    return val[half == "train"].reset_index(drop=True), val[half != "train"].reset_index(drop=True)


def run_conformal(name, win, split_column, columns, out_dir):
    """E8: tune on half the validation shots, calibrate a conformal threshold on the other half's clean shots,
    then check on the test shots whether the false-alarm rate really stays under each alpha."""
    os.makedirs(out_dir, exist_ok=True)
    train, val, test = split_frames(win, split_column)
    val_tune, val_cal = split_validation(val)
    best = tune(name, train, val_tune, columns)
    cal_levels = conformal.shot_alarm_levels(val_cal, models.scores(best["model"], val_cal[columns]))
    clean_levels = cal_levels.loc[~cal_levels["disrupted"], "level"]
    test_scores = models.scores(best["model"], test[columns])
    rows = []
    for alpha in config.CONFORMAL_ALPHAS:
        threshold = conformal.conformal_threshold(clean_levels, alpha)
        m = metrics_at(test, test_scores, threshold)
        rows.append({"method": "conformal", "alpha": alpha, "threshold": threshold, "n_cal_clean": len(clean_levels),
                     "test_false_alarm_rate": m["false_alarm_rate"], "test_catch_rate": m["catch_rate"],
                     "guarantee_holds": m["false_alarm_rate"] <= alpha})
    m = metrics_at(test, test_scores, best["threshold"])
    rows.append({"method": "tuned on validation", "alpha": config.MAX_FALSE_ALARM_RATE, "threshold": best["threshold"],
                 "n_cal_clean": len(clean_levels), "test_false_alarm_rate": m["false_alarm_rate"],
                 "test_catch_rate": m["catch_rate"], "guarantee_holds": m["false_alarm_rate"] <= config.MAX_FALSE_ALARM_RATE})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "conformal.csv"), index=False)
    write_json({"model": name, "split": split_column, "rows": rows}, os.path.join(out_dir, "metrics.json"))
    return rows


def unambiguous(frame, shots):
    """Windows of shots that are disrupted or have no quench at all, i.e. leaving out unnoted quenches."""
    no_quench = set(shots.loc[~np.isfinite(shots["t_quench"].astype(float)), "shot_id"])
    keep = frame["disrupted"] | frame["shot_id"].isin(no_quench)
    return frame[keep].reset_index(drop=True)


def run_pu(win, shots, split_column, columns, out_dir):
    """E9: standard HGB vs PU-weighted HGB (same settings), each with a validation-tuned threshold, on all test
    shots and on the unambiguous ones. Also estimates how many unnoted-quench windows look like disruptions."""
    os.makedirs(out_dir, exist_ok=True)
    train, val, test = split_frames(win, split_column)
    fit_rows = training_rows(train)
    standard = models.make_model("hgb", config.PU_PARAMS).fit(fit_rows[columns], fit_rows["label"])
    pu_model, c, weights = pu.fit_pu(config.PU_PARAMS, fit_rows, val, columns)
    near = fit_rows.loc[fit_rows["label"] == 0, "near_quench"].to_numpy()
    rows = []
    for method, model in [("standard", standard), ("pu", pu_model)]:
        threshold = evaluate.choose_threshold(score_split(model, val, columns)[1])
        for subset, frame in [("all test shots", test), ("unambiguous test shots", unambiguous(test, shots))]:
            m = metrics_at(frame, models.scores(model, frame[columns]), threshold)
            rows.append({"method": method, "test_set": subset, "threshold": threshold, **m})
    summary = {"label_frequency_c": c,
               "mean_weight_unnoted_quench_windows": float(weights[near].mean()) if near.any() else None,
               "mean_weight_other_unlabelled_windows": float(weights[~near].mean()) if (~near).any() else None,
               "rows": rows}
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "pu.csv"), index=False)
    write_json(summary, os.path.join(out_dir, "metrics.json"))
    return summary


def run_seed(name, windows, shots, seed):
    """E10: one more random split with this seed, tuned and tested like E1-E3, then the summary over all seeds so far.

    One split gives one number; many splits show how much of a result is luck of the split.
    """
    split = dataset.random_split(shots, seed=seed)
    resplit = windows.assign(split=windows["shot_id"].map(split))
    win = dataset.labelled_windows(resplit, shots)
    out_dir = os.path.join(config.RESULTS_DIR, "E10", f"seed_{seed}")
    run_model(name, win, "split", features.feature_names(), out_dir, save=False, with_importance=False)
    return seed_summary()


def seed_summary():
    """Mean, spread and count of the E10 test metrics over every seed run so far; also writes E10/summary.csv."""
    folder = os.path.join(config.RESULTS_DIR, "E10")
    rows = []
    for name in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
        path = os.path.join(folder, name, "metrics.json")
        if name.startswith("seed_") and os.path.exists(path):
            with open(path) as f:
                test = json.load(f)["test"]
            rows.append({"seed": int(name[5:]), "catch_rate": test["catch_rate"],
                         "false_alarm_rate": test["false_alarm_rate"], "warning_median_ms": test["warning_median_ms"]})
    if not rows:
        return {"n_seeds": 0}
    table = pd.DataFrame(rows).sort_values("seed")
    table.to_csv(os.path.join(folder, "summary.csv"), index=False)
    out = {"n_seeds": len(table)}
    for column in ["catch_rate", "false_alarm_rate", "warning_median_ms"]:
        out[f"{column}_mean"] = float(table[column].mean())
        out[f"{column}_std"] = float(table[column].std(ddof=1)) if len(table) > 1 else 0.0
    return out
