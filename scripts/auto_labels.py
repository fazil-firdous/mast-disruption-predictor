"""How did each shot end? Writes the automatic termination classes of src/termination.py for every shot of the
dataset to RESULTS_DIR/auto_labels/ (per_shot.csv, summary.json, summary.txt), with the measures behind each class.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src import config, extras, termination
from src.termination import CLASSES

OUT_DIR = os.path.join(config.RESULTS_DIR, "auto_labels")
WORDS = {"flat_top_drop": "lost during flat-top with flux left (likely a real disruption)",
         "flux_exhausted": "after the solenoid ran out of flux",
         "programmed_rampdown": "while the programme was already ramping down",
         "smooth_end": "smooth end, no sudden current drop",
         "unknown": "signals missing"}


def label_shot(row):
    """One shot's row: shot id, operator note, quench time, measures and class."""
    t_q = float(row["t_quench"])
    out = {"shot_id": int(row["shot_id"]), "campaign": row["campaign"], "note": bool(row["note"]), "t_quench": t_q}
    ending, m = termination.shot_ending(t_q, extras.load_extras(int(row["shot_id"]), allow_download=False))
    return {**out, **m, "termination": ending}


def summarise(table):
    """Counts per class, overall and split by whether the operators noted a disruption."""
    counts = {c: int((table["termination"] == c).sum()) for c in CLASSES}
    by_note = {}
    for noted, group in table.groupby("note"):
        by_note["noted" if noted else "not_noted"] = {c: int((group["termination"] == c).sum()) for c in CLASSES}
    return {"n_shots": len(table), "n_labelled": len(table) - counts["unknown"], "counts": counts, "by_note": by_note}


def summary_text(s):
    """The short Telegram message."""
    lines = [f"Automatic labels for {s['n_shots']:,} shots (how each one ended):"]
    lines += [f"- {s['counts'][c]:,}: {WORDS[c]}" for c in CLASSES if s["counts"][c]]
    quiet = s["by_note"].get("not_noted", {})
    if quiet.get("flat_top_drop"):
        lines.append(f"Of the shots nobody called a disruption, {quiet['flat_top_drop']:,} were lost during flat-top "
                     "with flux left: candidates for disruptions that were never recorded.")
    return "\n".join(lines)


def main():
    shots = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    table = pd.DataFrame([label_shot(row) for _, row in shots.iterrows()])
    s = summarise(table)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text(s) + "\n")
    print(summary_text(s))


if __name__ == "__main__":
    main()
