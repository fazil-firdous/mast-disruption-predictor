"""Which repeat-shot pairs really had the same programme? (v3 plan A3, the twin test set.)

scripts/mast_events.py finds shot pairs that a comment names ("repeat of 29922", "reference shot 12985"). A pair is a
fair natural experiment only if the machine was asked to do the same thing in both shots. This compares the two
shots' programmed plasma current (pulse schedule, from the cached extras) over the time both programmes cover up to
the earlier of the two quenches or ends:
- relative difference: RMS of the difference divided by the larger programmed maximum;
- peak ratio: the smaller programmed maximum over the larger.
A pair has the same programme if the relative difference is under SAME_RMS and the peak ratio over SAME_PEAK. A twin
with the same programme, where one shot was lost in the flat-top and the other was not, cannot be told apart by the
machine's programme, so only the plasma can explain a predictor's alarm on it.

Two checks of the method itself:
- control: the same comparison for shots CONTROL_OFFSET apart that no comment pairs. If those are as often 'same
  programme' as the named pairs, the comment adds nothing and the programme alone can find twins;
- programme twins: for every shot lost in the flat-top, the nearest earlier shot (within REACH shots, same campaign)
  with the same programme that was not lost. No comment needed, so the test set can be much larger.

    python scripts/twin_check.py      # reads events/pairs.csv, auto_labels/per_shot.csv and data/extras/;
                                      # writes events/twins.csv and events/programme_twins.csv

Offline (cached extras only). An analysis next to the experiments: outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, evaluate, extras

EVENTS_DIR = os.path.join(config.RESULTS_DIR, "events")
LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
SAME_RMS = 0.10  # programmes differing by under 10% of their peak (RMS) count as the same
SAME_PEAK = 0.90  # and their peaks within 10%
GRID_S = 0.001  # compare on a 1 ms grid
MIN_OVERLAP_S = 0.05  # need at least 50 ms of both programmes
CONTROL_OFFSET = 10  # control pairs: a shot and the shot 10 before it
REACH = 30  # programme twins: look back at most 30 shots
LOST = {"recorded", "candidate"}
KEPT = {"flux_exhausted", "programmed_rampdown", "smooth_end"}  # endings that count as 'not lost'


def programme(frame):
    """(time, programmed current) of one shot's extras, finite samples only, or None."""
    if frame is None or "ip_ref" not in frame:
        return None
    t, ref = frame["time"].to_numpy(dtype=float), frame["ip_ref"].to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(ref)
    return (t[ok], ref[ok]) if ok.sum() > 10 else None


def compare(a, b, t_stop=np.inf):
    """{'rms': relative RMS difference, 'peak_ratio', 'overlap_s'} of two programmes up to t_stop, or None."""
    if a is None or b is None:
        return None
    start = max(a[0][0], b[0][0], 0.0)
    stop = min(a[0][-1], b[0][-1], t_stop)
    if stop - start < MIN_OVERLAP_S:
        return None
    grid = np.arange(start, stop, GRID_S)
    ra, rb = np.interp(grid, *a), np.interp(grid, *b)
    peak = max(np.abs(ra).max(), np.abs(rb).max())
    if peak <= 0:
        return None
    return {"rms": float(np.sqrt(np.mean((ra - rb) ** 2)) / peak),
            "peak_ratio": float(min(np.abs(ra).max(), np.abs(rb).max()) / peak), "overlap_s": float(stop - start)}


def check(pairs, labels, load=lambda shot: extras.load_extras(int(shot), allow_download=False)):
    """pairs with the programme comparison and a same_programme flag; load(shot) gives a shot's extras or None."""
    quench = labels.set_index("shot_id")["t_quench"]
    rows = []
    for row in pairs.itertuples():
        stop = np.nanmin([quench.get(row.shot_id, np.nan), quench.get(row.other, np.nan), np.inf])
        c = compare(programme(load(row.shot_id)), programme(load(row.other)), stop)
        rows.append({**row._asdict(), **(c or {"rms": np.nan, "peak_ratio": np.nan, "overlap_s": np.nan})})
    out = pd.DataFrame(rows).drop(columns="Index", errors="ignore")
    out["same_programme"] = (out["rms"] < SAME_RMS) & (out["peak_ratio"] > SAME_PEAK)
    return out


def groups(labels):
    """Series shot_id -> ending group (evaluate.ending_group)."""
    quench = np.isfinite(labels["t_quench"].astype(float))
    return pd.Series([evaluate.ending_group(n, e, q) for n, e, q in zip(labels["note"], labels["termination"], quench)],
                     index=labels["shot_id"].to_numpy())


def control(pairs, labels, load, offset=CONTROL_OFFSET):
    """check() on control pairs: each named pair's later shot and the shot `offset` before it, where that pair is
    not itself named and both shots are labelled."""
    named = set(zip(pairs["shot_id"], pairs["other"]))
    known = set(labels["shot_id"])
    rows = [(s, s - offset) for s in pairs["shot_id"].unique() if (s, s - offset) not in named and s - offset in known]
    out = pd.DataFrame(rows, columns=["shot_id", "other"]).drop_duplicates()
    return check(out, labels, load) if len(out) else out.assign(rms=[], peak_ratio=[], same_programme=[])


def programme_twins(labels, load, reach=REACH):
    """For each shot lost in the flat-top: the nearest earlier shot within `reach` (same campaign) that was not lost
    and had the same programme. One row per lost shot that has such a twin."""
    group = groups(labels)
    quench = labels.set_index("shot_id")["t_quench"]
    campaign = labels.set_index("shot_id")["campaign"] if "campaign" in labels else pd.Series(dtype=object)
    cache = {}

    def prog(shot):
        if shot not in cache:
            cache[shot] = programme(load(shot))
        return cache[shot]

    rows = []
    for shot in labels["shot_id"]:
        if group.get(shot) not in LOST or prog(shot) is None:
            continue
        for other in range(shot - 1, shot - reach - 1, -1):
            if group.get(other) not in KEPT or campaign.get(other) != campaign.get(shot):
                continue
            stop = np.nanmin([quench.get(shot, np.nan), quench.get(other, np.nan), np.inf])
            c = compare(prog(shot), prog(other), stop)
            if c and c["rms"] < SAME_RMS and c["peak_ratio"] > SAME_PEAK:
                rows.append({"shot_id": shot, "other": other, "group": group[shot], "other_group": group[other],
                             "gap": shot - other, **c})
                break
    return pd.DataFrame(rows, columns=["shot_id", "other", "group", "other_group", "gap", "rms", "peak_ratio",
                                       "overlap_s"])


def text(checked, controlled=None, found=None, labels=None):
    twins = checked[checked["twin"].astype(bool)]
    lines = [
        f"Pairs: {len(checked):,}; with both programmes cached: {int(checked['rms'].notna().sum()):,}; "
        f"same programme: {int(checked['same_programme'].sum()):,}.",
        f"Twins (one lost in the flat-top, the other not): {len(twins):,}; with the same programme: "
        f"{int(twins['same_programme'].sum()):,}.",
        f"Relative programme difference, median over compared pairs: {checked['rms'].median():.3f} "
        f"(twins {twins['rms'].median():.3f})."]
    if controlled is not None and len(controlled):
        lines.append(f"Control (a named shot and the shot {CONTROL_OFFSET} before it, not named): "
                     f"{int(controlled['same_programme'].sum()):,} of {int(controlled['rms'].notna().sum()):,} compared "
                     f"have the same programme ({controlled['same_programme'].sum() / max(controlled['rms'].notna().sum(), 1):.0%}; "
                     f"named pairs: {checked['same_programme'].sum() / max(checked['rms'].notna().sum(), 1):.0%}).")
    if found is not None:
        lost = int(groups(labels).isin(LOST).sum()) if labels is not None else 0
        by = found["group"].value_counts().to_dict() if len(found) else {}
        lines.append(f"Programme twins (a lost shot and the nearest earlier same-programme shot within {REACH} that was "
                     f"not lost): {len(found):,} of {lost:,} lost shots ({by}); median gap "
                     f"{found['gap'].median() if len(found) else float('nan'):.0f} shots.")
    return "\n".join(lines)


def main():
    pairs = pd.read_csv(os.path.join(EVENTS_DIR, "pairs.csv"))
    labels = pd.read_csv(LABELS_CSV)
    load = lambda shot: extras.load_extras(int(shot), allow_download=False)
    checked = check(pairs, labels, load)
    checked.to_csv(os.path.join(EVENTS_DIR, "twins.csv"), index=False)
    controlled = control(pairs, labels, load)
    found = programme_twins(labels, load)
    found.to_csv(os.path.join(EVENTS_DIR, "programme_twins.csv"), index=False)
    print(text(checked, controlled, found, labels))


if __name__ == "__main__":
    main()
