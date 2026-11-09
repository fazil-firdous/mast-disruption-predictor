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

    python scripts/twin_check.py      # reads events/pairs.csv and data/extras/; writes events/twins.csv

Offline (cached extras only). An analysis next to the experiments: outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, extras

EVENTS_DIR = os.path.join(config.RESULTS_DIR, "events")
LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
SAME_RMS = 0.10  # programmes differing by under 10% of their peak (RMS) count as the same
SAME_PEAK = 0.90  # and their peaks within 10%
GRID_S = 0.001  # compare on a 1 ms grid
MIN_OVERLAP_S = 0.05  # need at least 50 ms of both programmes


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


def text(checked):
    twins = checked[checked["twin"].astype(bool)]
    return "\n".join([
        f"Pairs: {len(checked):,}; with both programmes cached: {int(checked['rms'].notna().sum()):,}; "
        f"same programme: {int(checked['same_programme'].sum()):,}.",
        f"Twins (one lost in the flat-top, the other not): {len(twins):,}; with the same programme: "
        f"{int(twins['same_programme'].sum()):,}.",
        f"Relative programme difference, median over compared pairs: {checked['rms'].median():.3f} "
        f"(twins {twins['rms'].median():.3f})."])


def main():
    pairs = pd.read_csv(os.path.join(EVENTS_DIR, "pairs.csv"))
    labels = pd.read_csv(LABELS_CSV)
    checked = check(pairs, labels)
    checked.to_csv(os.path.join(EVENTS_DIR, "twins.csv"), index=False)
    print(text(checked))


if __name__ == "__main__":
    main()
