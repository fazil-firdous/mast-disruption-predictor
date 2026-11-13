"""Runs where the store answers (a GitHub Actions machine, .github/workflows/fetch_programme.yml): read, for every
shot of the shot table, what was asked of the machine besides the plasma current and what gas it was given, and
write one small table for the research server, which cannot reach the store.

Per shot, the mean of each signal of scripts/programme_signals.TARGETS over bins of GRID_MS (bins are whole multiples
of GRID_MS; the time given is the end of the bin). A signal with one row per valve is summed over the valves first.
A signal a shot lacks is NaN.

    python scripts/programme_fetch.py                 # every shot of the shot table, RELAY_WORKERS at a time
    python scripts/programme_fetch.py --limit 40      # a trial on 40 shots spread over the table

Writes DATA_DIR/programme/programme.parquet and manifest.json. No model, no label, no result.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import programme_signals
from src import config, data

GRID_MS = 5
COLUMNS = list(programme_signals.TARGETS)
GIVE_UP_AFTER = 20  # shots in a row that the store did not answer for
WORK_HOURS = 4.5  # a job starts no new shot after this, so it saves its table before GitHub's limit
PROGRESS_EVERY = 250


def binned(t, v, grid_s=GRID_MS / 1000):
    """Bin means as a Series indexed by the bin number k (the bin covers k*grid_s up to (k+1)*grid_s)."""
    k = np.floor(np.asarray(t, dtype=float) / grid_s + 1e-9).astype(np.int64)
    first = k.min()
    sums = np.bincount(k - first, weights=np.asarray(v, dtype=float))
    counts = np.bincount(k - first)
    keep = counts > 0
    return pd.Series(sums[keep] / counts[keep], index=np.arange(first, first + len(counts))[keep])


def shot_frame(shot_id, open_group=programme_signals.open_group):
    """(frame or None, status) for one shot. frame: shot_id, time and one column per signal, on the GRID_MS bins.
    status: 'ok', 'empty' (the shot has none of the signals) or 'unreachable' (the store did not answer)."""
    opened, errors = {}, []
    for group in programme_signals.GROUPS:
        try:
            opened[group] = open_group(int(shot_id), group)
        except Exception as error:
            errors.append(type(error).__name__)
    columns = {}
    for name in programme_signals.TARGETS:
        group, variable, channels = programme_signals.target(name)
        if group not in opened:
            continue
        try:
            s = programme_signals.series(opened[group], variable, channels)
        except Exception as error:
            errors.append(type(error).__name__)
            continue
        if s is not None:
            columns[name] = binned(*s)
    if not columns:
        lost = [e for e in errors if e not in programme_signals.ABSENT]
        return None, "unreachable" if lost else "empty"
    frame = pd.DataFrame(columns).sort_index().reindex(columns=COLUMNS).astype("float32")
    frame.insert(0, "time", ((frame.index.to_numpy() + 1) * GRID_MS / 1000).astype("float32"))
    frame.insert(0, "shot_id", np.int32(shot_id))
    return frame.reset_index(drop=True), "ok"


def fetch(ids, workers=config.RELAY_WORKERS, open_group=programme_signals.open_group, say=print, deadline=None):
    """(table, report) for these shots. Stops with an error if the store stops answering. deadline (time.time()
    seconds): shots not started by then are left out and counted as 'not_started', so the job still saves."""
    ids = [int(i) for i in ids]
    frames, status, in_a_row, start = [], {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        if deadline is not None and time.time() > deadline:
            return None, "not_started"
        return shot_frame(shot, open_group)

    try:
        futures = {pool.submit(one, shot): shot for shot in ids}
        for done, future in enumerate(as_completed(futures), 1):
            frame, status[futures[future]] = future.result()
            if frame is not None:
                frames.append(frame)
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % PROGRESS_EVERY == 0 or done == len(ids):
                say(f"[{done}/{len(ids)}] {len(frames)} with data, {(time.time() - start) / done:.2f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    table = (pd.concat(frames, ignore_index=True).sort_values(["shot_id", "time"]).reset_index(drop=True) if frames
             else pd.DataFrame(columns=["shot_id", "time"] + COLUMNS))
    failed = sorted(shot for shot, s in status.items() if s == "unreachable")
    report = {"task": "programme", "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
              "shots_requested": len(ids), "shots_with_data": len(frames),
              "empty": sum(s == "empty" for s in status.values()), "failed": len(failed), "failed_ids": failed[:200],
              "not_started": sum(s == "not_started" for s in status.values()),
              "grid_ms": GRID_MS, "columns": COLUMNS, "seconds": round(time.time() - start)}
    return table, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="a trial on this many shots spread over the table (0 = all)")
    parser.add_argument("--workers", type=int, default=config.RELAY_WORKERS)
    parser.add_argument("--part", type=int, default=0, help="which share of the shots this job reads (0-based)")
    parser.add_argument("--parts", type=int, default=1, help="how many jobs share the shots")
    args = parser.parse_args()
    deadline = time.time() + WORK_HOURS * 3600
    try:
        import zarr
        zarr.config.set({"async.timeout": 3 * programme_signals.REQUEST_S})
    except Exception:
        pass
    shots = data.load_shot_table()
    ids = (programme_signals.sample(shots, args.limit) if args.limit else shots)["shot_id"].sort_values().tolist()
    ids = ids[args.part::args.parts]
    print(f"{len(ids)} shots, {args.workers} at a time", flush=True)
    try:
        table, report = fetch(ids, args.workers, say=lambda line: print(line, flush=True), deadline=deadline)
        report.update(part=args.part, parts=args.parts)
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)
    folder = programme_signals.data_folder()
    os.makedirs(folder, exist_ok=True)
    table.to_parquet(os.path.join(folder, programme_signals.TABLE), index=False)
    with open(os.path.join(folder, "manifest.json"), "w") as f:
        json.dump(report, f, indent=2)
    size = os.path.getsize(os.path.join(folder, programme_signals.TABLE)) / 1e6
    print(f"{report['shots_with_data']} shots with data, {report['empty']} without, {report['failed']} failed, "
          f"{report['not_started']} not started (time limit); "
          f"{len(table):,} rows, {size:.1f} MB, {report['seconds']} s", flush=True)
    if report["shots_with_data"] == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
