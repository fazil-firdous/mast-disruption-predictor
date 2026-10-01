"""Phase 2 report comparing note-based and signal-based disruption labels."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, labels
from src.data import download_shots, load_shot, load_shot_table

OUT_DIR = os.path.join(config.RESULTS_DIR, "label_report")
FLOOR_VALUES = [0.2, 0.35, 0.5, 0.65]  # values of CQ_MIN_FRAC_OF_PEAK tried in the sensitivity table


def choose_shots(table, n, seed, pool_ids=None):
    """Seeded sample of about n shots: every note-disrupted shot in the pool (capped at n // 2),
    then random note-clean shots from the pool to make up n. The pool is the whole table unless pool_ids is given.
    """
    rng = np.random.default_rng(seed)
    pool = table if pool_ids is None else table[table["shot_id"].isin(pool_ids)]
    is_note = pool["shot_postshot_comment"].map(labels.note_label)
    disrupted = pool.loc[is_note, "shot_id"].to_numpy()
    clean = pool.loc[~is_note, "shot_id"].to_numpy()
    if len(disrupted) > n // 2:
        disrupted = rng.choice(disrupted, size=n // 2, replace=False)
    clean = rng.choice(clean, size=min(n - len(disrupted), len(clean)), replace=False)
    return sorted(int(i) for i in np.concatenate([disrupted, clean]))


def fall_time_ms(time, ip):
    """Milliseconds for |Ip| to go from 90% to 10% of its peak, measured on the final fall. NaN if it never does."""
    x = labels.median_filter(np.abs(ip))
    peak = x.max()
    last_high = np.where(x >= 0.9 * peak)[0][-1]
    below = np.where(x[last_high:] <= 0.1 * peak)[0]
    if len(below) == 0:
        return np.nan
    return 1000 * (time[last_high + below[0]] - time[last_high])


def label_one(shot_id, table_row):
    """One summary row: note label, signal label and a few trace measurements."""
    shot = load_shot(shot_id)
    time, ip = shot["time"].to_numpy(), shot["ip"].to_numpy()
    found = labels.detect_disruption(time, ip)
    comment = table_row["shot_postshot_comment"]
    return {
        "shot_id": shot_id, "campaign": table_row["campaign"],
        "note": labels.note_label(comment), "t_note": labels.parse_note_time(comment),
        "signal": found["status"], "t_disrupt": found["t_disrupt"],
        "quench_rate": found["quench_rate"], "peak_ip": found["peak_ip"],
        "fall_ms": round(fall_time_ms(time, ip), 1) if found["status"] != "unknown" else np.nan,
        "comment": str(comment)[:90],
    }


def plot_cases(cases, path, title, show_fall=False):
    """Plot |Ip| for each case in a 2-column grid, marking the signal time (red) and the note time (blue)."""
    rows = (len(cases) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, (_, case) in zip(axes.flat, cases.iterrows()):
        shot = load_shot(int(case["shot_id"]))
        ax.plot(shot["time"], np.abs(shot["ip"]) / 1e3, color="0.45", lw=1)
        if not np.isnan(case["t_disrupt"]):
            ax.axvline(case["t_disrupt"], color="red", lw=1.2, zorder=3, label="signal t_disrupt")
        if not np.isnan(case["t_note"]):
            ax.axvline(case["t_note"], color="blue", ls="--", lw=1.2, zorder=3, label="note time")
        label = f"{int(case['shot_id'])} ({case['campaign']}): "
        if show_fall:
            label += f"detector: {case['signal']}, 90%->10% fall {case['fall_ms']:.0f} ms"
        else:
            label += case["comment"][:55]
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("time (s)", fontsize=8)
        ax.set_ylabel("|Ip| (kA)", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left")
    for ax in list(axes.flat)[len(cases):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def sensitivity(df):
    """Share of note-clean and note-disrupted shots flagged by the detector, for several CQ_MIN_FRAC_OF_PEAK values."""
    original = config.CQ_MIN_FRAC_OF_PEAK
    rows = []
    for value in FLOOR_VALUES:
        config.CQ_MIN_FRAC_OF_PEAK = value
        flagged = []
        for shot_id in df["shot_id"]:
            shot = load_shot(int(shot_id))
            flagged.append(labels.detect_disruption(shot["time"], shot["ip"])["detected"])
        flagged = np.array(flagged)
        rows.append({"CQ_MIN_FRAC_OF_PEAK": value,
                     "flagged, note clean": flagged[~df["note"].to_numpy()].mean(),
                     "flagged, note disrupted": flagged[df["note"].to_numpy()].mean()})
    config.CQ_MIN_FRAC_OF_PEAK = original
    return pd.DataFrame(rows)


def md_table(df, index=True):
    """Render a small DataFrame as a markdown table (avoids needing the tabulate package)."""
    if index:
        df = df.rename_axis("").reset_index()

    def fmt(v):
        return f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(str(c) for c in df.columns) + " |",
             "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


def final_counts(known):
    """Number of disrupted shots under each LABEL_SOURCE option (the configured one is listed first)."""
    options = [config.LABEL_SOURCE] + [o for o in ("signal", "note", "both") if o != config.LABEL_SOURCE]
    counts = {}
    for source in options:
        flags = [labels.final_label(n, s == "disrupted", source) for n, s in zip(known["note"], known["signal"])]
        counts[source] = int(sum(flags))
    return counts


def build_report(df, sens, rampdowns):
    """Markdown text of the report; every number comes from df, sens and rampdowns."""
    known = df[df["signal"] != "unknown"]
    crosstab = pd.crosstab(known["note"].map({True: "note yes", False: "note no"}),
                           known["signal"].map({"disrupted": "signal yes", "clean": "signal no"}))
    crosstab.index.name = None
    crosstab.columns.name = None
    note_yes = known[known["note"]]
    note_no = known[~known["note"]]
    agree = (known["note"] == (known["signal"] == "disrupted")).mean()
    both = known[known["note"] & known["t_note"].notna() & (known["signal"] == "disrupted")]
    delta = 1000 * (both["t_disrupt"] - both["t_note"])
    flagged = known[known["signal"] == "disrupted"]
    not_flagged = known[known["signal"] == "clean"]
    lines = [
        "# Label reconciliation report (SPEC 5.3)", "",
        f"Sample: {len(df)} shots, made of every note-disrupted shot in the pool ({df['note'].sum()}) plus "
        f"{(~df['note']).sum()} random note-clean shots (seed {config.RANDOM_SEED}). "
        "Because note-disrupted shots are over-represented, the overall agreement below is not the agreement "
        "on a typical shot; read the rates per row. "
        f"Shots labelled `unknown` (|Ip| never above {config.MIN_PEAK_IP:.0f} A): {(df['signal'] == 'unknown').sum()}.", "",
        "## 2x2 table", "", md_table(crosstab), "",
        f"- Overall agreement: {agree:.1%} of shots.",
        f"- Note-disrupted shots that the detector also flags: {(note_yes['signal'] == 'disrupted').mean():.1%} "
        f"({(note_yes['signal'] == 'disrupted').sum()} of {len(note_yes)}).",
        f"- Note-clean shots that the detector flags anyway: {(note_no['signal'] == 'disrupted').mean():.1%} "
        f"({(note_no['signal'] == 'disrupted').sum()} of {len(note_no)}).", "",
        "## Timing where both exist", "",
        f"{len(both)} shots have a note time and a detected quench. t_disrupt minus t_note (ms): "
        f"median {delta.median():.1f}, quartiles {delta.quantile(.25):.1f} to {delta.quantile(.75):.1f}, "
        f"{(delta.abs() <= 10).mean():.0%} within 10 ms, {(delta.abs() > 20).sum()} differ by more than 20 ms.", "",
        "## What does a flagged shot look like?", "",
        f"Time for |Ip| to fall from 90% to 10% of peak (median): {flagged['fall_ms'].median():.1f} ms for the "
        f"{len(flagged)} flagged shots, {not_flagged['fall_ms'].median():.1f} ms for the {len(not_flagged)} not flagged.", "",
        "Medians over flagged shots, split by note:", "",
        md_table(flagged.groupby("note")[["quench_rate", "peak_ip", "fall_ms"]].median().rename(
            index={True: "note yes", False: "note no"})), "",
        "## Normal ramp-downs (SPEC 5.2 step 5)", "",
        f"Three random note-clean shots with a clear slow end, 90%->10% fall of at least {5 * config.CQ_MAX_MS} ms "
        f"(`rampdowns.png`): shots {', '.join(str(int(s)) for s in rampdowns['shot_id'])}, fall times "
        f"{', '.join(f'{v:.0f}' for v in rampdowns['fall_ms'])} ms, detector result "
        f"{', '.join(rampdowns['signal'])}. Of the {len(not_flagged)} unflagged shots, "
        f"{(not_flagged['fall_ms'] < 5 * config.CQ_MAX_MS).sum()} fall faster than that "
        f"(shortest {not_flagged['fall_ms'].min():.0f} ms). They are borderline cases, often a spike and a partial "
        "drop followed by a final fall that starts below the CQ_MIN_FRAC_OF_PEAK floor.", "",
        "## Sensitivity to the starting-current floor", "", md_table(sens, index=False), "",
        f"## Final labels under LABEL_SOURCE = \"{config.LABEL_SOURCE}\"", "",
        "| LABEL_SOURCE | disrupted | clean |", "|---|---|---|",
        *[f"| {source} | {n} | {len(known) - n} |" for source, n in final_counts(known).items()], "",
        "Plots: `disagreements_note_only.png`, `disagreements_signal_only.png`, `disagreements_timing.png`, "
        "`rampdowns.png`. Per-shot table: `per_shot.csv`. Note-only list for manual review: `review_list.csv`. "
        "Exact shot list: `sample_ids.json`.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="sample size")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--pool-file", help="JSON list of shot ids to sample from (default: whole shot table)")
    parser.add_argument("--ids-file", help="JSON list of the exact shot ids to use (skips sampling)")
    args = parser.parse_args()

    table = load_shot_table()
    if args.ids_file:
        ids = json.load(open(args.ids_file))
    else:
        pool = json.load(open(args.pool_file)) if args.pool_file else None
        ids = choose_shots(table, args.n, args.seed, pool)
    result = download_shots(ids)
    failed = set(result["failed"])
    indexed = table.set_index("shot_id")

    df = pd.DataFrame([label_one(s, indexed.loc[s]) for s in ids if s not in failed])
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)
    with open(os.path.join(OUT_DIR, "sample_ids.json"), "w") as f:
        json.dump([int(s) for s in df["shot_id"]], f)

    note_only = df[df["note"] & (df["signal"] == "clean")]
    note_only.to_csv(os.path.join(OUT_DIR, "review_list.csv"), index=False)
    signal_only = df[~df["note"] & (df["signal"] == "disrupted")]
    timing = df[(df["t_disrupt"] - df["t_note"]).abs() > 0.02]
    # a clear ramp-down takes several times CQ_MAX_MS; faster unflagged ends are reported as borderline
    rampdowns = df[~df["note"] & (df["signal"] == "clean") & (df["fall_ms"] >= 5 * config.CQ_MAX_MS)]

    rng = np.random.default_rng(args.seed)
    pick = lambda d, k: d.iloc[np.sort(rng.permutation(len(d))[:k])]
    plot_cases(pick(note_only, 4), os.path.join(OUT_DIR, "disagreements_note_only.png"),
               "Note says disrupted, detector finds no quench")
    plot_cases(pick(signal_only, 4), os.path.join(OUT_DIR, "disagreements_signal_only.png"),
               "Detector finds a quench, note does not mention a disruption")
    plot_cases(pick(timing, 2), os.path.join(OUT_DIR, "disagreements_timing.png"),
               "Both say disrupted but the times differ by more than 20 ms")
    ramp_pick = pick(rampdowns, 3)
    plot_cases(ramp_pick, os.path.join(OUT_DIR, "rampdowns.png"),
               "Normal ramp-downs: note clean, detector should not flag", show_fall=True)

    report = build_report(df, sensitivity(df), ramp_pick)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(report)
    if failed:
        print(f"skipped {len(failed)} shots that failed to load, see {config.SKIPPED_CSV}")


if __name__ == "__main__":
    main()
