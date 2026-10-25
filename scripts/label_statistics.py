"""Statistics for the automatic termination labels (scripts/auto_labels.py): the shares with 95% intervals, the
solenoid ratio of operator-noted vs un-noted quench endings (AUC with a bootstrap interval, Mann-Whitney test, effect
sizes), the same again on the shots outside the 400-shot sample the rules were set on (a replication), and the
sampling weight that scales the un-noted counts to every MAST shot.

    python scripts/label_statistics.py    # reads RESULTS_DIR/auto_labels/per_shot.csv, writes statistics.txt next to it

An analysis next to the experiments: no experiment uses it, so it is outside every stamp.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, labels

LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
# the 400-shot report sample the rules and thresholds were set on, before the full run
RULE_SAMPLE_CSV = os.path.join(config.REPO_DIR, "results", "label_report", "per_shot.csv")
N_BOOTSTRAP = 2000  # resamples for the AUC and median intervals; more changes the third decimal only
LATE = ["flux_exhausted", "programmed_rampdown"]  # quenches after the solenoid stopped or during the ramp-down


def wilson(k, n, z=1.96):
    """95% Wilson interval of the proportion k / n."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(centre - half), float(centre + half)


def auc(pos, neg):
    """Probability that a random positive scores above a random negative, ties counting half (Mann-Whitney AUC)."""
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def rate(k, n):
    """k, n, the rate k / n and its Wilson interval."""
    low, high = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n, "low": low, "high": high}


def share(group, cls="flat_top_drop"):
    """How many shots of a group are in class cls, with its interval, and the rate without the 'unknown' ones."""
    k = (group["termination"] == cls).sum()
    known = (group["termination"] != "unknown").sum()
    return {**rate(k, len(group)), "rate_known": k / known if known else None}


def ratio_test(table, seed=config.RANDOM_SEED):
    """Solenoid ratio of the quench shots, operator-noted vs not noted: medians with a bootstrap interval of their
    difference, the AUC for 'not noted' from a LOW ratio with a bootstrap interval, the Mann-Whitney p-value,
    Cliff's delta and the share below the 'stopped' threshold."""
    q = table[table["t_quench"].notna() & table["sol_ratio"].notna()]
    noted, other = q.loc[q["note"], "sol_ratio"].to_numpy(), q.loc[~q["note"], "sol_ratio"].to_numpy()
    rng = np.random.default_rng(seed)
    boot_auc, boot_median = [], []
    for _ in range(N_BOOTSTRAP):
        a, b = rng.choice(other, len(other)), rng.choice(noted, len(noted))
        boot_auc.append(auc(-a, -b))
        boot_median.append(np.median(b) - np.median(a))
    value = auc(-other, -noted)
    return {"n_noted": len(noted), "n_other": len(other), "median_noted": float(np.median(noted)),
            "median_other": float(np.median(other)), "median_gap_ci": np.percentile(boot_median, [2.5, 97.5]).tolist(),
            "auc": value, "auc_ci": np.percentile(boot_auc, [2.5, 97.5]).tolist(),
            "p_value": float(stats.mannwhitneyu(noted, other, alternative="two-sided").pvalue),
            "cliffs_delta": 2 * value - 1, "stopped_noted": float(np.mean(noted < config.AUTO_SOL_STOPPED_RATIO)),
            "stopped_other": float(np.mean(other < config.AUTO_SOL_STOPPED_RATIO))}


def block(table):
    """Every statistic for one set of shots."""
    fired = table[table["t_quench"].notna()]
    return {"shots": len(table), "noted": share(table[table["note"]]), "not_noted": share(table[~table["note"]]),
            "fired_late": rate(fired["termination"].isin(LATE).sum(), len(fired)),
            "ratio": ratio_test(table)}


def sampling(table, shot_table):
    """Scale the un-noted share to every un-noted MAST shot. The un-noted shots in the sample are a two-stage simple
    random sample (scripts/build_dataset.py), so each stands for the same number of shots, assuming download and
    data-quality losses do not depend on how a shot ended. None without the shot table."""
    if shot_table is None:
        return None
    n_table = int((~shot_table["shot_postshot_comment"].map(labels.note_label)).sum())
    s = share(table[~table["note"]])
    return {"not_noted_in_table": n_table, "weight": n_table / s["n"],
            "estimate": s["rate"] * n_table, "low": s["low"] * n_table, "high": s["high"] * n_table}


def pct(x):
    return f"{100 * x:.1f}%"


def block_text(title, b):
    """One block of statistics in words."""
    r, n, o, late = b["ratio"], b["noted"], b["not_noted"], b["fired_late"]
    return "\n".join([
        f"{title} ({b['shots']:,} shots)",
        f"- Operator-noted disruptions lost during flat-top with flux left: {n['k']}/{n['n']} = {pct(n['rate'])} "
        f"(95% {pct(n['low'])}-{pct(n['high'])}; {pct(n['rate_known'])} without 'unknown').",
        f"- Shots with no disruption note, same ending: {o['k']}/{o['n']} = {pct(o['rate'])} "
        f"(95% {pct(o['low'])}-{pct(o['high'])}; {pct(o['rate_known'])} without 'unknown').",
        f"- Quenches after the solenoid stopped or during the ramp-down: {late['k']}/{late['n']} = "
        f"{pct(late['rate'])}.",
        f"- Solenoid ratio at the quench, median: {r['median_noted']:.2f} noted vs {r['median_other']:.2f} not noted "
        f"(gap 95% {r['median_gap_ci'][0]:.2f}-{r['median_gap_ci'][1]:.2f}; {r['n_noted']} vs {r['n_other']} shots). "
        f"AUC for 'not noted' from a low ratio {r['auc']:.3f} (95% {r['auc_ci'][0]:.3f}-{r['auc_ci'][1]:.3f}), "
        f"Mann-Whitney p = {r['p_value']:.1e}, Cliff's delta {r['cliffs_delta']:.2f}. Below "
        f"{config.AUTO_SOL_STOPPED_RATIO}: {pct(r['stopped_noted'])} noted vs {pct(r['stopped_other'])} not noted."])


def main():
    table = pd.read_csv(LABELS_CSV)
    out = {"all": block(table)}
    parts = [block_text("All labelled shots", out["all"])]
    if os.path.exists(RULE_SAMPLE_CSV):
        new = table[~table["shot_id"].isin(pd.read_csv(RULE_SAMPLE_CSV)["shot_id"])]
        out["replication"] = block(new)
        parts.append(block_text("Replication: only shots outside the 400-shot sample the rules were set on",
                                out["replication"]))
    shot_table = pd.read_parquet(config.SHOT_TABLE_CACHE) if os.path.exists(config.SHOT_TABLE_CACHE) else None
    out["sampling"] = sampling(table, shot_table)
    if out["sampling"]:
        s = out["sampling"]
        parts.append(f"Scaled to all {s['not_noted_in_table']:,} MAST shots with no disruption note (each sampled shot "
                     f"stands for {s['weight']:.2f}): about {s['estimate']:,.0f} ({s['low']:,.0f}-{s['high']:,.0f}) "
                     "would end during flat-top with flux left. An extrapolation, not a count.")
    text = "\n\n".join(parts)
    folder = os.path.dirname(LABELS_CSV)
    with open(os.path.join(folder, "statistics.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(folder, "statistics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(text)


if __name__ == "__main__":
    main()
