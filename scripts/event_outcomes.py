"""A feasibility count, not an analysis: after an event that an operator timed in a comment, how often did the shot
go on, and how often was it lost soon after?

A study of disturbances that recover against disturbances that end in a loss needs both kinds in numbers. This
counts them from files that exist (events/events.csv from scripts/mast_events.py), per event class:
- in shots that were not lost (they ended by flux exhaustion, a programmed ramp-down or smoothly);
- in lost shots, by the time from the written event to the quench.

    python scripts/event_outcomes.py

Caution for any use beyond counting: operators write their comments after the shot, knowing how it ended, so an
event that ended a shot is more likely to be written down, and timed, than one the plasma survived. A real study
would need events found from the signals alone. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

EVENTS_CSV = os.path.join(config.RESULTS_DIR, "events", "events.csv")
LOST = {"recorded", "candidate"}
GAPS_MS = [(-np.inf, 0, "after the quench time"), (0, 30, "under 30 ms"), (30, 100, "30 to 100 ms"),
           (100, np.inf, "100 ms or more")]
SKIP = {"disruption", "plasma lost", "early end"}  # words for the ending itself, not for a disturbance before it


def table(events):
    """One row per event class: timed events (one per shot and class, the earliest), how many were in shots that
    were not lost, and for lost shots the time from the event to the quench."""
    timed = events[np.isfinite(events["time_s"].astype(float)) & ~events["event"].isin(SKIP)]
    timed = timed.sort_values("time_s").drop_duplicates(["shot_id", "event"])
    rows = []
    for name, part in timed.groupby("event"):
        lost = part[part["group"].isin(LOST)]
        gap = 1000 * (lost["t_quench"].astype(float) - lost["time_s"].astype(float))
        row = {"event": name, "timed": len(part), "shot not lost": int((~part["group"].isin(LOST)).sum())}
        for low, high, label in GAPS_MS:
            row[f"lost, {label}"] = int(((gap >= low) & (gap < high)).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("timed", ascending=False).reset_index(drop=True)


def text(result):
    lines = ["Events an operator timed in a comment (one per shot and class), by what the shot did afterwards.",
             "A count for planning only: comments are written knowing the outcome, so survived events are "
             "under-reported.", "", "| " + " | ".join(result.columns) + " |", "|" + "---|" * len(result.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in result.itertuples(index=False)]
    return "\n".join(lines)


def main():
    print(text(table(pd.read_csv(EVENTS_CSV))))


if __name__ == "__main__":
    main()
