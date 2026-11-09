"""MAST-Events, step 1 (v3 plan A1 and A3): events with their cause and time from the operators' comments, and
repeat-shot pairs.

Operators often wrote what happened and when: "VDE at 0.242s", "locked mode @ 340 ms", "IRE at 150ms kills the
shot". No label so far uses these times or causes. This script reads every comment in the shot table and writes:
- events.csv: one row per (shot, event) with the event class (the fixed word lists of scripts/candidate_notes.py,
  negations skipped), the time written next to it (or empty), and the text it came from;
- pairs.csv: shots whose comment names another shot they repeat or compare with ("repeat of 29922", "reference shot
  12985", "same as 30113", or "repeat of the previous shot"), with both shots' endings. A pair with the same programme
  where one shot was lost and the other was not is a 'twin': the machine side is the same, so only the plasma differs;
- summary.txt: events per ending group, how close the written times are to the detector's quench time (a check of
  the time extraction), and how many pairs and twins there are.

    python scripts/mast_events.py      # reads auto_labels/per_shot.csv and the cached shot table

A match is a hint about one comment, not a label (as in Tokamak-SI, arXiv:2610.32264); the next step weighs it
against the other evidence. An analysis next to the experiments: outside every stamp, writes only into
RESULTS_DIR/events/.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.candidate_notes import WORDS
from src import config, evaluate

OUT_DIR = os.path.join(config.RESULTS_DIR, "events")
LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
EVENT_WORDS = {"disruption": r"disrupt", "IRE": r"\bires?\b",
               **{k: v for k, v in WORDS.items() if k != "H-mode (control)" and k != "collapse/IRE"}}
# a time right after the event words: "at 0.242s", "@ 363MS", "at ~310 ms", "at 300", "0.25s"
TIME = re.compile(r"(?:\bat\b|@|~)?\s*~?\s*(\d+(?:\.\d+)?)\s*(ms|s|sec|secs|seconds)?(?![a-z0-9]|\.\d)", re.IGNORECASE)
NOT_A_TIME = re.compile(r"\s*(?:k?v\b|k?a\b|[km]?w\b|%|cm\b|mm\b|m\b|k?hz\b|kat\b|t\b|e1|e2)", re.IGNORECASE)
TIME_REACH = 40  # characters after an event word in which its time is looked for
# a negation in the same clause (no , . ; : ! ? in between) and within 30 characters: "no sign of a VDE" is negated,
# "no H-mode, VDE at 0.2s" is not. candidate_notes.py keeps its own fixed rule, so its counts do not change
NEGATION = re.compile(r"(?:\bnot\b|\bno\b|n't\b|n.t\b|\bwithout\b)[^.,;:!?]{0,30}$")
SHOT = r"#?\s*(\d{5})"
PAIR_PATTERNS = {
    "repeat": re.compile(r"repeat(?:s|ed|ing)?\s+(?:of\s+)?(?:shot\s+)?" + SHOT, re.IGNORECASE),
    "reference": re.compile(r"ref(?:erence)?\.?\s*(?:shot|pulse)?\s*(?:is|was|=|:)?\s*" + SHOT, re.IGNORECASE),
    "same as": re.compile(r"(?:same\s+as|as\s+in|similar\s+to|like|compared?\s+(?:with|to)|cf\.?)\s+(?:shot\s+)?" + SHOT,
                          re.IGNORECASE),
}
PREVIOUS = re.compile(r"repeat(?:s|ed|ing)?\s+(?:of\s+)?(?:the\s+)?(?:previous|last)\s+shot", re.IGNORECASE)
MAX_PAIR_GAP = 2000  # a shot named in a comment must be earlier and within this many shots (same campaign, roughly)


def to_seconds(value, unit):
    """A written time in seconds, or NaN. No unit: under 1.5 is seconds, 10 and over is milliseconds (MAST shots last
    under about 1 s), in between is ambiguous."""
    unit = (unit or "").lower()
    if unit == "ms":
        return value / 1000
    if unit.startswith("s"):
        return value if value < 2 else np.nan
    if value < 1.5:
        return value
    if 10 <= value <= 1500:
        return value / 1000
    return np.nan


def time_after(text, end):
    """The first time written within TIME_REACH characters after position end, in seconds, or NaN."""
    window = text[end:end + TIME_REACH]
    for m in TIME.finditer(window):
        if m.group(2) is None and NOT_A_TIME.match(window, m.end()):
            continue
        t = to_seconds(float(m.group(1)), m.group(2))
        if np.isfinite(t) and 0 < t < 2:
            return float(t)
    return np.nan


def time_before(text, start):
    """The last time with an explicit unit (ms or s) within TIME_REACH characters before position start, or NaN:
    'event at 317ms is desastorous (locked mode?)' names the time first."""
    window = text[max(0, start - TIME_REACH):start]
    times = [to_seconds(float(m.group(1)), m.group(2)) for m in TIME.finditer(window) if m.group(2)]
    times = [t for t in times if np.isfinite(t) and 0 < t < 2]
    return float(times[-1]) if times else np.nan


def events_in(text):
    """[(event class, time in s or NaN, matched text)] for one comment; negated matches are skipped."""
    found = []
    low = str(text).lower() if isinstance(text, str) else ""
    for name, pattern in EVENT_WORDS.items():
        for m in re.finditer(pattern, low):
            if NEGATION.search(low[:m.start()]) is not None:
                continue
            t = time_after(low, m.end())
            t = t if np.isfinite(t) else time_before(low, m.start())
            found.append((name, t, low[max(0, m.start() - 30):m.end() + TIME_REACH].strip()))
    return found


def pairs_in(shot_id, text):
    """[(other shot, relation)] named in one comment: earlier shots within MAX_PAIR_GAP, never the shot itself."""
    out = []
    if not isinstance(text, str):
        return out
    for relation, pattern in PAIR_PATTERNS.items():
        for m in pattern.finditer(text):
            other = int(m.group(1))
            if 0 < shot_id - other <= MAX_PAIR_GAP:
                out.append((other, relation))
    if PREVIOUS.search(text):
        out.append((shot_id - 1, "repeat"))
    return list(dict.fromkeys(out))


def comment_columns(table):
    """The shot table's free-text comment columns (post-shot first)."""
    return [c for c in ("shot_postshot_comment", "shot_preshot_comment", "shot_postshot_description",
                        "shot_preshot_description") if c in table]


def build(labels, shot_table):
    """(events, pairs) DataFrames from our per-shot endings and the shot table's comments."""
    columns = comment_columns(shot_table)
    table = labels.merge(shot_table[["shot_id"] + columns], on="shot_id", how="left")
    quench = np.isfinite(table["t_quench"].astype(float))
    table["group"] = [evaluate.ending_group(n, e, q) for n, e, q in zip(table["note"], table["termination"], quench)]
    events, pairs = [], []
    for row in table.itertuples():
        for column in columns:
            text = getattr(row, column)
            for name, t, snippet in events_in(text):
                events.append({"shot_id": row.shot_id, "group": row.group, "event": name, "time_s": t,
                               "t_quench": row.t_quench, "source": column, "text": snippet})
            for other, relation in pairs_in(row.shot_id, text):
                pairs.append({"shot_id": row.shot_id, "other": other, "relation": relation, "source": column})
    events = pd.DataFrame(events, columns=["shot_id", "group", "event", "time_s", "t_quench", "source", "text"])
    pairs = pd.DataFrame(pairs, columns=["shot_id", "other", "relation", "source"]).drop_duplicates(["shot_id", "other"])
    ending = table.set_index("shot_id")["group"]
    pairs["group"] = pairs["shot_id"].map(ending)
    pairs["other_group"] = pairs["other"].map(ending)
    lost = {"recorded", "candidate"}
    pairs["twin"] = (pairs["group"].isin(lost) != pairs["other_group"].isin(lost)) & pairs["other_group"].notna()
    return events, pairs


def summary(labels, events, pairs):
    """The summary in words: events per ending group, time agreement with the quench, pairs and twins."""
    quench = np.isfinite(labels["t_quench"].astype(float))
    groups = pd.Series([evaluate.ending_group(n, e, q) for n, e, q in zip(labels["note"], labels["termination"], quench)])
    sizes = groups.value_counts()
    lines = ["Shots with each event in a comment (share of the ending group):", "",
             "| Event | " + " | ".join(evaluate.ENDING_GROUPS) + " |", "|---|" + "---|" * len(evaluate.ENDING_GROUPS)]
    for name in EVENT_WORDS:
        shots = events[events["event"] == name].drop_duplicates("shot_id")
        counts = shots["group"].value_counts()
        cells = [f"{counts.get(g, 0)} ({100 * counts.get(g, 0) / sizes[g]:.1f}%)" if g in sizes else "-"
                 for g in evaluate.ENDING_GROUPS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    timed = events[np.isfinite(events["time_s"].astype(float)) & np.isfinite(events["t_quench"].astype(float))]
    lines += ["", "Written time minus the detector's quench time, where both exist (ms):"]
    for name, part in timed.groupby("event"):
        dt = 1000 * (part["time_s"] - part["t_quench"])
        lines.append(f"- {name}: {len(part)} times, median {dt.median():+.0f}, within 20 ms "
                     f"{np.mean(np.abs(dt) <= 20):.0%}, within 50 ms {np.mean(np.abs(dt) <= 50):.0%}")
    lines += ["", f"Shot pairs named in comments: {len(pairs)} ({pairs['relation'].value_counts().to_dict()}).",
              f"Twins (one shot lost in the flat-top, recorded or candidate, the other not): {int(pairs['twin'].sum())}."]
    return "\n".join(lines)


def main():
    labels = pd.read_csv(LABELS_CSV)
    shot_table = pd.read_parquet(config.SHOT_TABLE_CACHE)
    events, pairs = build(labels, shot_table)
    os.makedirs(OUT_DIR, exist_ok=True)
    events.to_csv(os.path.join(OUT_DIR, "events.csv"), index=False)
    pairs.to_csv(os.path.join(OUT_DIR, "pairs.csv"), index=False)
    text = summary(labels, events, pairs)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nComment columns read: {comment_columns(shot_table)}. Files in {OUT_DIR}")


if __name__ == "__main__":
    main()
