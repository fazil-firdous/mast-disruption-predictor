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


def choose_shots(table, n, n_extra, seed):
    """Seeded sample: n uniformly random shots (headline numbers) plus n_extra note-disrupted shots (more cases)."""
    rng = np.random.default_rng(seed)
    uniform = [int(i) for i in rng.choice(table["shot_id"].to_numpy(), size=n, replace=False)]
    flagged = table.loc[table["shot_postshot_comment"].map(labels.note_label) & ~table["shot_id"].isin(uniform), "shot_id"]
    extra = [int(i) for i in rng.choice(flagged.to_numpy(), size=min(n_extra, len(flagged)), replace=False)]
    return {"uniform": uniform, "extra": extra}


def fall_time_ms(time, ip):
    """Milliseconds for |Ip| to go from 90% to 10% of its peak, measured on the final fall. NaN if it never does."""
    x = labels.median_filter(np.abs(ip))
    peak = x.max()
    last_high = np.where(x >= 0.9 * peak)[0][-1]
    below = np.where(x[last_high:] <= 0.1 * peak)[0]
    if len(below) == 0:
        return np.nan
    return 1000 * (time[last_high + below[0]] - time[last_high])


def label_one(shot_id, table_row, group):
    """One summary row: note label, signal label and a few trace measurements."""
    shot = load_shot(shot_id)
    time, ip = shot["time"].to_numpy(), shot["ip"].to_numpy()
    found = labels.detect_disruption(time, ip)
    comment = table_row["shot_postshot_comment"]
    return {
        "shot_id": shot_id, "group": group, "campaign": table_row["campaign"],
        "note": labels.note_label(comment), "t_note": labels.parse_note_time(comment),
        "signal": found["status"], "t_disrupt": found["t_disrupt"],
        "quench_rate": found["quench_rate"], "peak_ip": found["peak_ip"],
        "fall_ms": round(fall_time_ms(time, ip), 1) if found["status"] != "unknown" else np.nan,
        "comment": str(comment)[:90],
    }


def plot_cases(cases, path, title):
    """Plot |Ip| for each case in a 2-column grid, marking the signal time (red) and the note time (blue)."""
    rows = (len(cases) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 2.6 * rows), squeeze=False)
    for ax, (_, case) in zip(axes.flat, cases.iterrows()):
        shot = load_shot(int(case["shot_id"]))
        ax.plot(shot["time"], np.abs(shot["ip"]) / 1e3, color="black", lw=1)
        if not np.isnan(case["t_disrupt"]):
            ax.axvline(case["t_disrupt"], color="red", lw=1, label="signal")
        if not np.isnan(case["t_note"]):
            ax.axvline(case["t_note"], color="blue", ls="--", lw=1, label="note")
        ax.set_title(f"{int(case['shot_id'])} ({case['campaign']}): {case['comment'][:55]}", fontsize=8)
        ax.set_ylabel("|Ip| (kA)", fontsize=8)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, loc="upper left")
    for ax in list(axes.flat)[len(cases):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def sensitivity(table, uniform_ids):
    """Share of the uniform sample flagged as disrupted for several values of CQ_MIN_FRAC_OF_PEAK."""
    original = config.CQ_MIN_FRAC_OF_PEAK
    rows = []
    for value in FLOOR_VALUES:
        config.CQ_MIN_FRAC_OF_PEAK = value
        flagged = 0
        for shot_id in uniform_ids:
            shot = load_shot(shot_id)
            flagged += labels.detect_disruption(shot["time"], shot["ip"])["detected"]
        rows.append({"CQ_MIN_FRAC_OF_PEAK": value, "flagged": flagged, "share": flagged / len(uniform_ids)})
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


def build_report(df, sens, n_uniform):
    """Markdown text of the report; every number comes from df and sens."""
    uni = df[df["group"] == "uniform"]
    known = uni[uni["signal"] != "unknown"]
    crosstab = pd.crosstab(known["note"].map({True: "note yes", False: "note no"}),
                           known["signal"].map({"disrupted": "signal yes", "clean": "signal no"}))
    crosstab.index.name = None
    crosstab.columns.name = None
    both = df[df["note"] & df["t_note"].notna() & (df["signal"] == "disrupted")]
    delta = 1000 * (both["t_disrupt"] - both["t_note"])
    quenched = df[df["signal"] == "disrupted"]
    smooth = df[df["signal"] == "clean"]
    note_yes = df[df["note"] & (df["signal"] != "unknown")]
    share_flagged = (known["signal"] == "disrupted").mean()
    share_noted = known["note"].mean()
    lines = [
        "# Label reconciliation report (SPEC 5.3)", "",
        f"Shots processed: {len(df)} ({n_uniform} uniform random shots for the headline numbers, the rest are extra "
        "note-disrupted shots used for the timing comparison and example plots). "
        f"Shots with `unknown` signal label (|Ip| never above {config.MIN_PEAK_IP:.0f} A): {(df['signal'] == 'unknown').sum()}.", "",
        "## 2x2 table, uniform random sample", "", md_table(crosstab), "",
        f"Operator notes mark {share_noted:.1%} of this sample as disrupted. "
        f"The signal detector marks {share_flagged:.1%}.", "",
        "## Timing where both exist", "",
        f"{len(both)} shots have a note time and a detected quench. t_disrupt minus t_note (ms): "
        f"median {delta.median():.1f}, quartiles {delta.quantile(.25):.1f} to {delta.quantile(.75):.1f}, "
        f"{(delta.abs() <= 10).mean():.0%} within 10 ms.", "",
        "## Does a slow ramp-down look different from a quench?", "",
        f"Time for |Ip| to fall from 90% to 10% of peak, median over shots: "
        f"{quenched['fall_ms'].median():.1f} ms for the {len(quenched)} shots flagged disrupted, "
        f"{smooth['fall_ms'].median():.1f} ms for the {len(smooth)} shots not flagged. "
        f"Shots with no 10% point before the data ends: {smooth['fall_ms'].isna().sum()} of the not-flagged ones.", "",
        "## Are noted and un-noted quenches different?", "",
        "Medians over shots flagged by the detector:", "",
        md_table(quenched.groupby("note")[["quench_rate", "peak_ip", "fall_ms"]].median().rename(
            index={True: "note yes", False: "note no"})), "",
        "## Sensitivity to the starting-current floor", "", md_table(sens, index=False), "",
        "## Summary", "",
        f"- {(note_yes['signal'] == 'disrupted').mean():.0%} of note-disrupted shots "
        f"({(note_yes['signal'] == 'disrupted').sum()} of {len(note_yes)}) show a quench in |Ip|, so a note is rarely contradicted by the signal.",
        f"- The detector finds a quench in {share_flagged:.0%} of the uniform sample, far more than the notes. "
        "Flagged shots with and without a note have similar quench rates and fall times (table above), "
        "so the current trace alone does not separate a noted disruption from an ordinary end of pulse.",
        "- Where both exist the two agree on timing (above), so the detector's time is usable when it is right about the event.",
        "- Neither label has independent ground truth here. The final choice of `LABEL_SOURCE` is the owner's call (SPEC 5.3).", "",
        "Plots: `disagreements_note_only.png`, `disagreements_signal_only.png`, `disagreements_timing.png`. "
        "Per-shot table: `per_shot.csv`. Note-only list for manual review: `review_list.csv`.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=400, help="uniform random shots for the headline numbers")
    parser.add_argument("--n-extra", type=int, default=150, help="extra note-disrupted shots for more cases")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--ids-file", help="JSON with lists 'uniform' and 'extra' instead of random choice")
    args = parser.parse_args()

    table = load_shot_table()
    if args.ids_file:
        chosen = json.load(open(args.ids_file))
    else:
        chosen = choose_shots(table, args.n, args.n_extra, args.seed)
    ids = chosen["uniform"] + chosen["extra"]
    result = download_shots(ids)
    failed = set(result["failed"])
    indexed = table.set_index("shot_id")

    rows = [label_one(s, indexed.loc[s], "uniform" if s in chosen["uniform"] else "extra")
            for s in ids if s not in failed]
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)

    note_only = df[df["note"] & (df["signal"] == "clean")]
    note_only.to_csv(os.path.join(OUT_DIR, "review_list.csv"), index=False)
    signal_only = df[~df["note"] & (df["signal"] == "disrupted")]
    delta = (df["t_disrupt"] - df["t_note"]).abs()
    timing = df[delta > 0.02].sort_values("shot_id")

    rng = np.random.default_rng(args.seed)
    pick = lambda d, k: d.iloc[rng.permutation(len(d))[:k]]
    plot_cases(pick(note_only, 3), os.path.join(OUT_DIR, "disagreements_note_only.png"),
               "Note says disrupted, detector finds no quench")
    plot_cases(pick(signal_only[signal_only["group"] == "uniform"], 3),
               os.path.join(OUT_DIR, "disagreements_signal_only.png"), "Detector finds a quench, no note")
    plot_cases(pick(timing, 2), os.path.join(OUT_DIR, "disagreements_timing.png"),
               "Both say disrupted but times differ by more than 20 ms")

    sens = sensitivity(table, [s for s in chosen["uniform"] if s not in failed])
    report = build_report(df, sens, len(chosen["uniform"]) - len(failed & set(chosen["uniform"])))
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(report)
    print(report)
    if failed:
        print(f"skipped {len(failed)} shots that failed to load, see {config.SKIPPED_CSV}")


if __name__ == "__main__":
    main()
