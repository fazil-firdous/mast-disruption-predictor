"""Runs where the store answers (a GitHub machine, .github/workflows/fetch_efit_times.yml): the EFIT slice times of the
given shots (the `time` coordinate of each shot's `equilibrium` group), for the timing self-audit
(docs/preregistration_mast_timing_audit.md). No outcome, model or label passes through.

    python scripts/efit_times_fetch.py --ids <zlib+base64 list> --part 0 --parts 4

Writes DATA_DIR/efit_times/efit_times.parquet (shot_id, time) and manifest.json.
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

from scripts import programme_signals, sxr_fetch
from src import config

TABLE = "efit_times.parquet"
GIVE_UP_AFTER = 20


def folder():
    return os.path.join(config.DATA_DIR, "efit_times")


def shot_times(shot_id, open_group=programme_signals.open_group):
    """(times, status): the equilibrium slice times; status 'ok', 'absent' or 'unreachable'."""
    try:
        ds = open_group(int(shot_id), "equilibrium")
    except Exception as error:
        return None, "absent" if type(error).__name__ in programme_signals.ABSENT else "unreachable"
    if "time" not in ds.variables:
        return None, "absent"
    t = np.asarray(ds["time"].values, dtype=float)
    t = t[np.isfinite(t)]
    return (t, "ok") if len(t) else (None, "absent")


def fetch(ids, workers=8, open_group=programme_signals.open_group, say=print):
    frames, status, in_a_row, start = [], {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        try:
            return shot_times(shot, open_group)
        except Exception as error:
            print(f"shot {shot}: skipped, {type(error).__name__}: {str(error)[:120]}", flush=True)
            return None, "odd"

    try:
        futures = {pool.submit(one, int(s)): int(s) for s in ids}
        for done, future in enumerate(as_completed(futures), 1):
            t, status[futures[future]] = future.result()
            if t is not None:
                frames.append(pd.DataFrame({"shot_id": np.int32(futures[future]), "time": t}))
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % 200 == 0 or done == len(futures):
                say(f"[{done}/{len(futures)}] {(time.time() - start) / done:.2f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["shot_id", "time"])
    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "absent", "unreachable", "odd")}
    report = {"task": "efit_times", "shots_requested": len(status), "shots_with_data": counts["ok"],
              "absent": counts["absent"], "failed": counts["unreachable"], "odd": counts["odd"],
              "seconds": round(time.time() - start)}
    return table, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default="")
    parser.add_argument("--shots", default="")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        import zarr
        zarr.config.set({"async.timeout": 3 * programme_signals.REQUEST_S})
    except Exception:
        pass
    ids = sxr_fetch.decode_ids(args.ids) if args.ids else [int(s) for s in args.shots.split(",") if s.strip()]
    ids = sorted(ids)[args.part::args.parts]
    print(f"{len(ids)} shots", flush=True)
    try:
        table, report = fetch(ids, args.workers, say=lambda line: print(line, flush=True))
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)
    report.update(part=args.part, parts=args.parts)
    os.makedirs(folder(), exist_ok=True)
    table.to_parquet(os.path.join(folder(), TABLE), index=False)
    with open(os.path.join(folder(), "manifest.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(report, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
