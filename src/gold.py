"""Gold-standard labels by hand (ROADMAP A2): the owner labels ~100 shots on Telegram, blind to every other label.

The queue over-samples the disagreements that matter (quenches nobody noted) and is seeded, so it is repeatable.
The plot shows only measured and programmed signals: no operator note, no detector time. Answers: d = disruption
(the plasma was lost while the program still asked for current), n = normal end, u = unsure.
Everything is stored privately in MAST_HOME/gold.
"""
import csv
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, extras
from src.data import load_shot

FOLDER = os.path.join(config.HOME_DIR, "gold")
ANSWERS = {"d": "disruption", "n": "normal end", "u": "unsure"}


def stratum(row):
    """Which kind of shot this is, from the note and the detector (used only to build the queue, never shown)."""
    quench = np.isfinite(row["t_quench"])
    if row["note"] and quench:
        return "noted_quench"
    if row["note"]:
        return "noted_no_quench"
    return "unnoted_quench" if quench else "no_quench"


def build_queue(shots, plan=config.GOLD_PLAN, seed=config.RANDOM_SEED):
    """Seeded list of {'shot_id', 'stratum'} with plan[stratum] shots of each kind, in a shuffled order."""
    rng = np.random.default_rng(seed)
    kinds = shots.apply(stratum, axis=1)
    picked = []
    for kind, n in plan.items():
        ids = shots.loc[kinds == kind, "shot_id"].to_numpy()
        for shot_id in rng.permutation(ids)[:n]:
            picked.append({"shot_id": int(shot_id), "stratum": kind})
    order = rng.permutation(len(picked))
    return [picked[i] for i in order]


def queue(shots=None):
    """The saved queue, built (and saved) from shots on first use."""
    path = os.path.join(FOLDER, "queue.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if shots is None:
        shots = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    items = build_queue(shots)
    os.makedirs(FOLDER, exist_ok=True)
    with open(path, "w") as f:
        json.dump(items, f)
    return items


def read_labels():
    """Answers given so far as a DataFrame (shot_id, answer, stratum, answered_at)."""
    path = os.path.join(FOLDER, "labels.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["shot_id", "answer", "stratum", "answered_at"])
    return pd.read_csv(path)


def record(shot_id, answer, kind):
    """Append one answer ('d', 'n' or 'u'); a later answer for the same shot replaces the earlier one."""
    os.makedirs(FOLDER, exist_ok=True)
    path = os.path.join(FOLDER, "labels.csv")
    is_new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["shot_id", "answer", "stratum", "answered_at"])
        writer.writerow([shot_id, answer, kind, time.strftime("%Y-%m-%d %H:%M:%S")])


def next_item(items, labels):
    """The first queued shot without an answer, or None when the queue is done."""
    done = set(labels["shot_id"].astype(int))
    for item in items:
        if item["shot_id"] not in done:
            return item
    return None


def plot_blind(shot_id, path):
    """|Ip| with the programmed current, radiated power and solenoid current. No labels of any kind."""
    shot = extras.with_extras(load_shot(shot_id), shot_id)
    t = shot["time"]
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)
    axes[0].plot(t, np.abs(shot["ip"]) / 1e3, color="0.2", lw=1.2, label="|Ip| measured")
    if shot["ip_ref"].notna().any():
        axes[0].plot(t, shot["ip_ref"] / 1e3, color="tab:green", lw=1, ls="--", label="Ip programmed")
    axes[0].set_ylabel("current (kA)")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, shot["power_radiated"] / 1e3, color="tab:orange", lw=1)
    axes[1].set_ylabel("Prad (kW)")
    axes[2].plot(t, shot["sol_current"] / 1e3, color="tab:purple", lw=1)
    axes[2].set_ylabel("solenoid (kA)")
    axes[2].set_xlabel("time (s)")
    axes[0].set_title("Did this plasma disrupt? /d disruption, /n normal end, /u unsure", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def evaluate(shots, labels):
    """How often each automatic label agrees with the owner's clear answers (d or n), overall and per stratum."""
    clear = labels[labels["answer"].isin(["d", "n"])].drop_duplicates("shot_id", keep="last")
    merged = clear.merge(shots, on="shot_id")
    gold = merged["answer"] == "d"
    rows = []
    for source in ("note", "signal", "both"):
        auto = merged[f"status_{source}"] == "disrupted"
        agree = float((auto == gold).mean()) if len(merged) else np.nan
        rows.append({"label source": source, "shots": len(merged), "agree": agree,
                     "auto yes, gold no": int((auto & ~gold).sum()), "auto no, gold yes": int((~auto & gold).sum())})
    per_stratum = {}
    for kind, group in merged.groupby("stratum"):
        per_stratum[kind] = f"{(group['answer'] == 'd').sum()} disruption / {(group['answer'] == 'n').sum()} normal"
    return pd.DataFrame(rows), per_stratum


def report_text(shots, labels):
    """Telegram text: progress plus agreement of each automatic label with the gold answers."""
    table, per_stratum = evaluate(shots, labels)
    lines = [f"Gold labels: {len(labels.drop_duplicates('shot_id'))} answered "
             f"({(labels['answer'] == 'u').sum()} unsure)"]
    for _, row in table.iterrows():
        lines.append(f"{row['label source']}: agrees on {row['agree']:.0%} of {row['shots']} clear shots "
                     f"(says yes, you said no: {row['auto yes, gold no']}; says no, you said yes: {row['auto no, gold yes']})")
    for kind, text in per_stratum.items():
        lines.append(f"{kind}: {text}")
    return "\n".join(lines)
