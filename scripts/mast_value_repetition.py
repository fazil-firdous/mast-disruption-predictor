"""MAST value-repetition signatures (docs/preregistration_mast_value_repetition.md, exploratory, fixed before any
statistic): do the published EFIT values repeat exactly more often in lost shots than in their twins at the same
programme time? Reads the EFIT slice times collected for the timing self-audit and the cached extras. No model.

    python scripts/mast_value_repetition.py

Writes RESULTS_DIR/mast_timing_audit/value_repetition.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, mast_timing_audit as mt, sampling_audit, twin_check, v3_explore, v3_runs
from src import config, extras, notify

VARS = list(config.EFIT_VARIABLES)
SPAN_S = 0.100
MIN_SLICES = 3
LEADS_MS = mt.LEADS_MS
PRIMARY_LEAD = mt.PRIMARY_LEAD


def slice_values(slice_times, frame):
    """(available times, values: slices x variables) of one shot: each slice's value read from the extras grid at the
    first grid time at or after slice time + delay. None without extras or EFIT columns."""
    if frame is None or not all(v in frame for v in VARS):
        return None
    grid = frame["time"].to_numpy(dtype=float)
    avail = np.sort(slice_times) + mt.DELAY_S
    idx = np.searchsorted(grid, avail - 1e-9, side="left")
    ok = idx < len(grid)
    return avail[ok], frame[VARS].to_numpy(dtype=float)[idx[ok]]


def repetition(avail, values, t, span=SPAN_S):
    """(R, R_any, RUN) over slices available in (t - span, t]; NaN with fewer than MIN_SLICES there."""
    keep = (avail > t - span + 1e-9) & (avail <= t + 1e-9)
    v = values[keep]
    if len(v) < MIN_SLICES:
        return np.nan, np.nan, np.nan
    a, b = v[:-1], v[1:]
    complete = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    if not complete.any():
        return np.nan, np.nan, np.nan
    all_eq = np.all(a == b, axis=1) & complete
    any_eq = np.any((a == b) & np.isfinite(a) & np.isfinite(b), axis=1)
    run, longest = 1, 1
    for e in all_eq:
        run = run + 1 if e else 1
        longest = max(longest, run)
    return float(all_eq[complete].mean()), float(any_eq[complete].mean()), float(longest)


def pair_rows(pairs, series_of, grid_end_of, leads=LEADS_MS):
    rows = []
    for p in pairs.itertuples():
        if not np.isfinite(p.t_loss):
            continue
        sl, sk = series_of(p.shot_id), series_of(p.other)
        el, ek = grid_end_of(p.shot_id), grid_end_of(p.other)
        if sl is None or sk is None or el is None or ek is None:
            continue
        for lead in leads:
            t = p.t_loss - lead / 1000
            if (np.isfinite(p.t_start) and t < p.t_start - 1e-9) or el < t - 1e-9 or ek < t - 1e-9:
                continue
            ml, mk = repetition(*sl, t), repetition(*sk, t)
            if not (np.isfinite(ml[0]) and np.isfinite(mk[0])):
                continue
            row = {"shot_id": p.shot_id, "other": p.other, "lead_ms": lead}
            for i, name in enumerate(("R", "R_any", "RUN")):
                row[f"win_{name}"] = mt.denser(ml[i], mk[i], +1)
                row[f"diff_{name}"] = ml[i] - mk[i]
            rows.append(row)
    return pd.DataFrame(rows)


def text(rows, flat):
    lines = ["MAST value-repetition signatures (docs/preregistration_mast_value_repetition.md; exploratory): exact "
             "adjacent repeats of the six EFIT variables on the published 5 ms grid, over the 100 ms before the moment "
             "compared, lost shot against twin. T = share of pairs where the lost shot repeats more. Not a recovery of "
             "native reconstruction times.", ""]
    if len(rows):
        groups = v3_explore.shot_groups(rows["shot_id"].to_numpy(), rows["other"].to_numpy())
        part = (rows["lead_ms"] == PRIMARY_LEAD).to_numpy()
        r = mt.clustered(rows.loc[part, "win_R"].to_numpy(dtype=float), groups[part])
        lines.append(f"PRIMARY R (all six equal) at {PRIMARY_LEAD} ms, {int(part.sum()):,} pairs: T = {sampling_audit.fmt(r)}: "
                     + ("no difference in value repetition at the moment compared" if r[1] is not None and
                        mt.BAND[0] <= r[0] <= mt.BAND[1] and r[1] <= 0.5 <= r[2] else
                        "VALUE REPETITION DIFFERS (not read as a shortcut; reported)" if r[1] is not None and
                        (r[1] > mt.BAND[1] or r[2] < mt.BAND[0]) else "reported as found"))
        lines += ["", "All measures (T | mean paired difference, lost minus twin):"]
        for lead in LEADS_MS:
            part = (rows["lead_ms"] == lead).to_numpy()
            cells = []
            for name in ("R", "R_any", "RUN"):
                w = mt.clustered(rows.loc[part, f"win_{name}"].to_numpy(dtype=float), groups[part])
                d = np.nanmean(rows.loc[part, f"diff_{name}"].to_numpy(dtype=float)) if part.any() else np.nan
                cells.append(f"{name} {sampling_audit.fmt(w)} | {d:+.3f}")
            lines.append(f"- {lead} ms ({int(part.sum()):,} pairs): " + "; ".join(cells))
    if flat:
        lines += ["", f"Over the flat-top, share of adjacent slices with all six equal (median of shots): lost "
                      f"{flat['lost']:.3f}, kept {flat['kept']:.3f}"]
    return "\n".join(lines)


def main():
    say = lambda line: print(line, flush=True)  # noqa: E731
    path = mt.times_path()
    if not os.path.exists(path):
        say(f"STOPPED: no {path}: run python scripts/mast_timing_audit.py --relay first")
        sys.exit(1)
    et = pd.read_parquet(path)
    times = {int(s): g["time"].to_numpy(dtype=float) for s, g in et.groupby("shot_id")}
    cache, ends = {}, {}
    load = lambda s: extras.load_extras(int(s), allow_download=False)  # noqa: E731

    def series_of(s):
        s = int(s)
        if s not in cache:
            frame = load(s)
            cache[s] = slice_values(times[s], frame) if s in times else None
            ends[s] = float(frame["time"].iloc[-1]) if frame is not None and len(frame) else None
        return cache[s]

    def end_of(s):
        series_of(s)
        return ends.get(int(s))

    main_pairs = how_early.pair_frames(load)[0]
    say(f"{len(main_pairs):,} twin pairs; measuring ...")
    rows = pair_rows(main_pairs, series_of, end_of)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    group = twin_check.groups(labels)
    flat = {"lost": [], "kept": []}
    for s in sorted(set(main_pairs["shot_id"]) | set(main_pairs["other"])):
        ser = series_of(s)
        if ser is None:
            continue
        start = v3_runs.flat_top_start(load(s))
        end = end_of(s)
        if start is None or end is None:
            continue
        r = repetition(*ser, end - mt.CUT_S, span=end - mt.CUT_S - start)
        if np.isfinite(r[0]):
            flat["lost" if group.get(s) in twin_check.LOST else "kept"].append(r[0])
    flat = {k: float(np.median(v)) for k, v in flat.items() if v} if all(flat.values()) else None
    out = text(rows, flat)
    os.makedirs(mt.folder(), exist_ok=True)
    rows.to_csv(os.path.join(mt.folder(), "value_repetition_pairs.csv"), index=False)
    with open(os.path.join(mt.folder(), "value_repetition.txt"), "w") as f:
        f.write(out + "\n")
    say(out)
    notify.send("MAST value repetition: done\n\n" + out)


if __name__ == "__main__":
    main()
