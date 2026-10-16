"""Plain-English Telegram messages for every result: what the experiment asks, the numbers as shot counts, and
what they mean. Every sentence is built from the numbers passed in; nothing here makes up a result."""
import math

from src import conformal, config

MODEL_WORDS = {
    "baseline": "physics rule (no learning: alarm when the current falls unusually fast, or radiation is high "
                "compared with beam power)",
    "logreg": "logistic regression (the simplest learning model: one weighted sum of the features)",
    "forest": "random forest (hundreds of decision trees that vote)",
    "hgb": "gradient boosting (trees built one after another, each fixing the last one's mistakes)",
}
SHORT_NAMES = {"baseline": "physics rule", "logreg": "logistic regression", "forest": "random forest",
               "hgb": "gradient boosting", "standard": "usual model", "pu": "PU model"}
SWEEP_INTRO = {
    "horizon_ms": ("E4 - how far ahead should the model look?",
                   "The model is retrained to warn this many ms before the disruption."),
    "drop_signal": ("E5 - which signal matters?",
                    "The model is retrained without one signal at a time. If leaving a signal out hurts, that "
                    "signal carries useful information."),
    "label_source": ("E7 - which labels to trust?",
                     "Disruptions are labelled from the operator notes, from the current-drop detector, or only "
                     "where both agree (our default)."),
    "drop_feature": ("E11 - shortcut test",
                     "Is the model just learning how long a shot has lasted? It is retrained without 'time since "
                     "start'. A big fall means part of what it learned was a shortcut, not plasma physics."),
    "ukaea_horizon_ms": ("E12 - the UKAEA FAIR-MAST benchmark's rules",
                         "Labels from the current drop alone and warnings 10, 30 and 60 ms ahead, as in the UKAEA "
                         "study. Their exact shots and labels are not public, so this is the same kind of test, "
                         "not a number to put next to theirs."),
    "feature_set": ("E13 - do magnetic signals help?",
                    "v1 uses the 4 basic signals. '+magnetics' adds the saddle-coil locked-mode signal, Mirnov "
                    "coil fluctuations, solenoid current and the current-control error."),
}
VALUE_WORDS = {
    "drop_signal": {"none": "all four signals", "ip_abs": "without plasma current",
                    "prad": "without radiated power", "neutrons": "without neutron rate",
                    "pnbi": "without beam heating power"},
    "label_source": {"both": "note and detector agree", "signal": "detector only", "note": "notes only"},
    "drop_feature": {"none": "all features", "t_since_start": "without time since start"},
    "feature_set": {"v1": "4 basic signals", "v1+magnetics": "+ magnetic signals",
                    "v1+magnetics-no-time": "+ magnetic signals, no time since start"},
}
VALIDATION_NOTE = ("These numbers are on validation shots (the ones used for tuning), so they are hints. With few "
                   "shots, a difference of 1-3 shots can be luck; the seed sweep (E10) shows which ones hold up.")


def missing(x):
    """True for None and NaN (saved results store NaN as None)."""
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x):
    """0.0882 -> '8.8%'; missing -> 'n/a'."""
    return "n/a" if missing(x) else f"{100 * x:.1f}%"


def of(rate, n):
    """'3 of 34 (8.8%)' from a rate and the number of shots it is out of; just the % if n is unknown."""
    if missing(rate):
        return "n/a"
    if n is None:
        return pct(rate)
    return f"{round(rate * n)} of {n} ({pct(rate)})"


def chance_text(test):
    """Compare the catch rate with an alarm that fires at random on the same share of shots."""
    fa, low = test["false_alarm_rate"], test["ci95"]["catch_rate"][0]
    text = (f"An alarm that fired at random on {pct(fa)} of shots would catch about {pct(fa)} of disruptions. ")
    if not missing(low) and low > fa:
        return text + "This model is above that even at the low end of its 95% range: real skill."
    return text + ("The 95% range still reaches down to that level, so we cannot yet call this real skill "
                   "(more shots or better signals needed).")


def ending_text(by_ending):
    """Where the false alarms fell: on 'clean' shots that ended in an unnoted current drop, or on the others."""
    if not by_ending:
        return ""
    q, n = by_ending["unnoted_quench"], by_ending["no_quench"]
    if q["n_clean"] == 0:
        return "No normal shot here ended in a sudden current drop nobody noted."
    text = (f"Where the false alarms fell: {q['false_alarms']} of the {q['n_clean']} 'normal' shots that ended in "
            f"a sudden current drop nobody noted, {n['false_alarms']} of the {n['n_clean']} that ended smoothly.")
    rate_q = q["false_alarms"] / q["n_clean"]
    rate_n = n["false_alarms"] / max(n["n_clean"], 1)
    if q["false_alarms"] > 0 and rate_q > rate_n:
        text += (" They lean towards the unnoted drops: some of those 'normal' shots may really be disruptions "
                 "with a wrong label (PU learning, E9, is built for this). Small numbers, so a hint only.")
    return text


def model_text(exp, metrics):
    """E0-E3: one model, tuned on validation and tested once."""
    t = metrics["test"]
    nd, nc = t["n_disrupted"], t["n_clean"]
    lines = [f"{exp} - {MODEL_WORDS.get(metrics['model'], metrics['model'])}",
             f"Tested once on {nd + nc} shots it never saw: {nd} that disrupted, {nc} that ended normally.",
             f"Warned in time (at least {config.MIN_WARNING_MS} ms ahead): {of(t['catch_rate'], nd)}, 95% range "
             f"{pct(t['ci95']['catch_rate'][0])}-{pct(t['ci95']['catch_rate'][1])}.",
             f"Warned too late: {of(t['tardy_rate'], nd)}. Missed: {of(t['miss_rate'], nd)}.",
             f"False alarms on normal shots: {of(t['false_alarm_rate'], nc)}. The alarm level was chosen on "
             f"validation shots to keep this at or under {pct(config.MAX_FALSE_ALARM_RATE)}.",
             "Median warning time: " + ("none caught" if missing(t["warning_median_ms"])
                                        else f"{t['warning_median_ms']:.0f} ms before the disruption") + "."]
    if missing(t["catch_rate"]) or t["catch_rate"] == 0:
        lines.append("It caught nothing in time at this false-alarm limit.")
    else:
        lines.append(chance_text(t))
    lines.append(ending_text(t.get("false_alarms_by_ending")))
    if not metrics["val"]["meets_false_alarm_limit"]:
        lines.append("Note: no alarm level kept validation false alarms under the limit, so the quietest one was used.")
    return "\n".join(line for line in lines if line)


def value_words(knob, value):
    """How one sweep value reads in a message, e.g. ('horizon_ms', 30) -> '30 ms ahead'."""
    if knob in ("horizon_ms", "ukaea_horizon_ms"):
        return f"{value} ms ahead"
    return VALUE_WORDS.get(knob, {}).get(str(value), str(value))


def sweep_text(rows):
    """E4, E5, E7, E11, E12, E13: one model retrained for each value of one knob, compared on validation shots."""
    knob = next(k for k in rows[0] if k in SWEEP_INTRO)
    title, intro = SWEEP_INTRO[knob]
    lines = [title, intro]
    for row in rows:
        nd, nc = row.get("n_disrupted"), row.get("n_clean")
        chance = "" if missing(row.get("positive_share")) else f" (random: {score(row['positive_share'])})"
        lines.append(f"- {value_words(knob, row[knob])}: caught {of(row['catch_rate'], nd)}, false alarms "
                     f"{of(row['false_alarm_rate'], nc)}, ranking score {score(row['pr_auc'])}{chance}")
    best = max(rows, key=lambda r: (r["catch_rate"] or 0, r["pr_auc"] or 0))
    worst = min(rows, key=lambda r: (r["catch_rate"] or 0, r["pr_auc"] or 0))
    nd = best.get("n_disrupted")
    if nd and best is not worst:
        gap = round(((best["catch_rate"] or 0) - (worst["catch_rate"] or 0)) * nd)
        lines.append(f"Best: {value_words(knob, best[knob])}. Best and worst differ by {gap} caught shot(s).")
    lines.append("Ranking score = PR-AUC: how well the model puts the moments just before a disruption above all "
                 "the others (0 to 1, higher is better; random guessing scores the share of such moments).")
    lines.append(VALIDATION_NOTE)
    return "\n".join(lines)


def score(x):
    """PR-AUC as '0.26'; missing -> 'n/a'."""
    return "n/a" if missing(x) else f"{x:.2f}"


def temporal_text(outcome):
    """E6: trained on older campaigns, tested on newer ones."""
    model = next(k for k in outcome if k != "baseline")
    t = outcome[model]
    nd, nc = t["n_disrupted"], t["n_clean"]
    fa, fa_low = t["false_alarm_rate"], t["ci95"]["false_alarm_rate"][0]
    limit = config.MAX_FALSE_ALARM_RATE
    lines = [f"E6 - train on old campaigns ({', '.join(config.TEMPORAL_TRAIN_CAMPAIGNS)}), test on new ones "
             f"({', '.join(config.TEMPORAL_TEST_CAMPAIGNS)})",
             "A real machine changes between campaigns, so this is the honest test of whether a predictor would "
             "still work next year.",
             f"{SHORT_NAMES.get(model, model).capitalize()}: caught {of(t['catch_rate'], nd)}, warned too late "
             f"{of(t['tardy_rate'], nd)}, false alarms {of(fa, nc)} (95% range "
             f"{pct(fa_low)}-{pct(t['ci95']['false_alarm_rate'][1])}).",
             f"Physics rule for comparison: caught {of(outcome['baseline']['catch_rate'], nd)}."]
    if not missing(fa_low) and fa_low > limit:
        lines.append(f"The false-alarm rate is above the {pct(limit)} limit even at the low end of its range: an "
                     "alarm level tuned on old campaigns does not hold on new ones (campaign drift).")
    elif not missing(fa) and fa > limit:
        lines.append(f"False alarms went over the {pct(limit)} limit, but the range still includes it, so this "
                     "is a warning sign, not proof.")
    else:
        lines.append(f"False alarms stayed within the {pct(limit)} limit on the new campaigns.")
    lines.append(ending_text(t.get("false_alarms_by_ending")))
    return "\n".join(line for line in lines if line)


def conformal_text(rows):
    """E8: alarm levels set from held-out normal shots with a promised false-alarm limit, checked on test shots."""
    n_cal = rows[0]["n_cal_clean"]
    lines = ["E8 - an alarm level with a promise (conformal prediction)",
             f"The alarm level is set from {n_cal} held-out normal shots so that at most a chosen share of new "
             "normal shots get a false alarm, provided new shots are like the old ones. Checked on the test shots:"]
    for row in rows:
        nd, nc = row.get("n_test_disrupted"), row.get("n_test_clean")
        name = (f"promise <= {pct(row['alpha'])}" if row["method"] == "conformal"
                else f"usual method (tuned on validation, aim <= {pct(row['alpha'])})")
        if row["threshold"] is None:
            need = conformal.min_calibration_shots(row["alpha"])
            lines.append(f"- {name}: needs at least {need} calibration shots, we have {n_cal}, so it would never "
                         "alarm. More shots needed.")
            continue
        kept = "kept" if row["guarantee_holds"] else "BROKEN"
        lines.append(f"- {name}: false alarms {of(row['test_false_alarm_rate'], nc)} -> {kept}; "
                     f"caught {of(row['test_catch_rate'], nd)}")
    lines.append("The promise is about the average over many repeats, so one test going slightly over is not proof "
                 "that it fails. The real test is when new shots are NOT like old ones (E6, new campaigns).")
    return "\n".join(lines)


def pu_text(summary):
    """E9: positive-unlabelled learning, where unnoted current drops count as 'unknown' instead of 'normal'."""
    lines = ["E9 - PU learning (unnoted current drops treated as 'unknown', not 'normal')"]
    c = summary.get("label_frequency_c")
    if not missing(c):
        lines.append(f"Estimated share of disruption-like moments that carry a disruption label: {pct(c)} "
                     "(Elkan-Noto estimate; below 100% means some disruption-like endings were never labelled).")
    w_q, w_o = summary.get("mean_weight_unnoted_quench_windows"), summary.get("mean_weight_other_unlabelled_windows")
    if not missing(w_q) and not missing(w_o):
        lines.append(f"Moments just before an unnoted current drop look like a real disruption with estimated "
                     f"probability {pct(w_q)}, against {pct(w_o)} for other unlabelled moments.")
    for row in summary["rows"]:
        lines.append(f"- {SHORT_NAMES.get(row['method'], row['method'])}, {row['test_set']}: caught "
                     f"{of(row['catch_rate'], row['n_disrupted'])}, "
                     f"false alarms {of(row['false_alarm_rate'], row['n_clean'])}")
    lines.append("'Unambiguous test shots' leaves out the unnoted current drops, whose true label nobody knows.")
    return "\n".join(lines)


def seed_text(summary):
    """E10: the same experiment on many different random splits; the spread shows how much is luck."""
    n = summary["n_seeds"]
    return (f"E10 - seed sweep: {n} different random splits so far (each puts different shots in train, "
            f"validation and test).\nCaught in time: {pct(summary['catch_rate_mean'])} on average, give or take "
            f"{pct(summary['catch_rate_std'])}.\nFalse alarms: {pct(summary['false_alarm_rate_mean'])} on average, "
            f"give or take {pct(summary['false_alarm_rate_std'])}.\nThe 'give or take' is how much one split's "
            "result is luck. A difference between two methods smaller than it is not a real difference.")


def dataset_text(counts):
    """The dataset message, in words."""
    split = counts.get("split_counts", {})
    text = (f"Dataset: {counts['shots']} shots ({counts['disrupted']} disrupted, {counts['clean']} normal), cut "
            f"into {counts['windows']:,} windows of {config.WINDOW_MS} ms. Split by shot: {split.get('train', 0)} "
            f"to learn from, {split.get('val', 0)} to tune on, {split.get('test', 0)} kept for the final test.")
    if counts.get("reused"):
        text += "\nSame code, shots and labels as last time, so the existing dataset was reused (not rebuilt)."
    return text


def describe(exp, outcome):
    """The Telegram message for one experiment's outcome."""
    if exp in ("E0", "E1", "E2", "E3"):
        return model_text(exp, outcome)
    if exp == "E6":
        return temporal_text(outcome)
    if exp == "E8":
        return conformal_text(outcome)
    if exp == "E9":
        return pu_text(outcome)
    if exp == "E10":
        return seed_text(outcome)
    return sweep_text(outcome)


def summary_lines(results):
    """One line per main model (test numbers), for the end-of-run message and results/SUMMARY.md."""
    lines = []
    for exp, m in results.items():
        t = m["test"]
        lines.append(f"{exp} {m['model']}: caught {of(t['catch_rate'], t['n_disrupted'])}, false alarms "
                     f"{of(t['false_alarm_rate'], t['n_clean'])}")
    return "\n".join(lines)
