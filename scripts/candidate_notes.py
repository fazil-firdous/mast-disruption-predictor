"""What the operators wrote about each kind of ending: are the unrecorded candidates hardware trips, or plasma events?

The candidates (lost during flat-top with flux left, no disruption note) behave like recorded disruptions before the
quench (scripts/paper_numbers.py). A plasma can also be lost because a power supply trips or an interlock fires
(one recorded disruption reads 'DISRUPTS AT 300MS (FA TRIP)'). This counts, per ending group, how many post-shot
comments mention hardware words, plasma-event words or an early end, so the paper can say how many candidates could
be hardware endings. Word lists are fixed below; a match is a hint about one comment, not a label.

    python scripts/candidate_notes.py     # reads auto_labels/per_shot.csv and the cached shot table, prints a table

An analysis next to the experiments, like scripts/label_statistics.py: outside every stamp, writes only into
RESULTS_DIR/auto_labels/notes_by_ending.txt (the comments are public FAIR-MAST metadata).
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.label_statistics import wilson
from src import config, evaluate

LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
# Each class names one kind of event, so a common word cannot swamp it (a first version's 'plasma' class matched the
# 'mode' of H-mode, which most shots mention). 'H-mode' is a control: it says what kind of shot it was, not how it ended.
WORDS = {
    "hardware trip": r"trip|interlock|crowbar|\bpsu\b|power supply",
    "VDE": r"\bvdes?\b|vertical",
    "locked mode": r"lock(?:ed|ing)? ?modes?|mode ?lock",
    "MHD/NTM": r"\bmhd\b|\bntms?\b|tearing",
    "collapse/IRE": r"collapse|crash|\bires?\b",
    "early end": r"terminat|ended early|early end|short shot|stopped early|lost plasma|plasma lost",
    "H-mode (control)": r"h-?mode",
}
EVENTS = ["VDE", "locked mode", "MHD/NTM", "collapse/IRE", "early end"]  # words for how a plasma is lost
N_EXAMPLES = 8  # comments shown per word class for the candidates, seeded


def word_flags(comments):
    """DataFrame of booleans: does each comment match each word class (case-insensitive)? NaN counts as no."""
    text = comments.fillna("").astype(str).str.lower()
    out = pd.DataFrame({name: text.str.contains(pattern, regex=True) for name, pattern in WORDS.items()})
    out["any loss event"] = out[EVENTS].any(axis=1)
    out["any comment"] = text.str.strip().str.len() > 0
    return out


def by_ending(table):
    """Per ending group: shots, and how many comments match each word class (count and share)."""
    has_quench = np.isfinite(table["t_quench"].astype(float))
    groups = [evaluate.ending_group(n, e, q) for n, e, q in zip(table["note"], table["termination"], has_quench)]
    flags = word_flags(table["shot_postshot_comment"])
    rows = {}
    for group in evaluate.ENDING_GROUPS:
        members = np.array([g == group for g in groups])
        if members.any():
            rows[group] = {"shots": int(members.sum()),
                           **{c: int(flags.loc[members, c].sum()) for c in flags.columns}}
    return pd.DataFrame.from_dict(rows, orient="index"), groups, flags


def examples(table, groups, flags, seed=config.RANDOM_SEED):
    """A few seeded candidate comments for each word class, so the matches can be read, not only counted."""
    rng = np.random.default_rng(seed)
    is_candidate = np.array([g == "candidate" for g in groups])
    lines = []
    for column in EVENTS + ["hardware trip", "none of these"]:
        if column == "none of these":
            hit = ~flags[EVENTS + ["hardware trip"]].any(axis=1) & flags["any comment"]
        else:
            hit = flags[column]
        idx = np.where(is_candidate & hit.to_numpy())[0]
        lines.append(f"\nCandidates, {column} ({len(idx)}):")
        for i in rng.choice(idx, size=min(N_EXAMPLES, len(idx)), replace=False) if len(idx) else []:
            lines.append(f"  {int(table['shot_id'].iloc[i])}: {str(table['shot_postshot_comment'].iloc[i])[:160]}")
    return lines


def main():
    labels = pd.read_csv(LABELS_CSV)
    shot_table = pd.read_parquet(config.SHOT_TABLE_CACHE)[["shot_id", "shot_postshot_comment"]]
    table = labels.merge(shot_table, on="shot_id", how="left")
    counts, groups, flags = by_ending(table)
    shares = counts.drop(columns="shots").div(counts["shots"], axis=0)
    text = ["Post-shot comments by how the shot ended (count, share of the group's shots, Wilson 95% range)", "",
            "| Ending | Shots | " + " | ".join(shares.columns) + " |", "|---|---|" + "---|" * len(shares.columns)]
    for group, row in counts.iterrows():
        cells = [f"{row[c]} ({100 * shares.at[group, c]:.1f}%, {100 * wilson(row[c], row['shots'])[0]:.1f}-"
                 f"{100 * wilson(row[c], row['shots'])[1]:.1f})" for c in shares.columns]
        text.append(f"| {group} | {row['shots']} | " + " | ".join(cells) + " |")
    text += ["", "'any loss event': " + ", ".join(EVENTS) + ". Word classes: " +
             "; ".join(f"{k}: {v}" for k, v in WORDS.items())]
    text += examples(table, groups, flags)
    out = "\n".join(text)
    with open(os.path.join(os.path.dirname(LABELS_CSV), "notes_by_ending.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
