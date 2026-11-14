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
LEVEL1_TABLE = "efm_times.parquet"  # level-1 efm: attempted times with a converged flag (--level1)
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


def open_level1(shot_id, group):
    """One group of a shot's level-1 store (the level-2 URL with level2 replaced by level1), with a time limit."""
    import aiohttp
    import xarray as xr
    url = config.SHOT_ZARR_URL.replace("/level2/", "/level1/").format(shot_id=shot_id)
    options = ({"client_kwargs": {"trust_env": True, "timeout": aiohttp.ClientTimeout(total=programme_signals.REQUEST_S)}}
               if url.startswith("http") else None)
    return xr.open_zarr(url, group=group, consolidated=True, storage_options=options)


def level1_times(shot_id, open_group=open_level1):
    """(frame of attempted times with a converged flag, status) from the level-1 efm group."""
    try:
        ds = open_group(int(shot_id), "efm")
    except Exception as error:
        return None, "absent" if type(error).__name__ in programme_signals.ABSENT else "unreachable"
    if "all_times" not in ds.variables:
        return None, "absent"
    attempted = np.asarray(ds["all_times"].values, dtype=float).ravel()
    attempted = np.unique(attempted[np.isfinite(attempted)])
    converged = (np.asarray(ds["cnvrgd_times"].values, dtype=float).ravel() if "cnvrgd_times" in ds.variables
                 else np.array([]))
    converged = converged[np.isfinite(converged)]
    if not len(attempted):
        return None, "absent"
    flag = np.isin(np.round(attempted, 6), np.round(converged, 6))
    return pd.DataFrame({"shot_id": np.int32(shot_id), "time": attempted, "converged": flag}), "ok"


def fetch_level1(ids, workers=8, open_group=open_level1, say=print):
    frames, status, in_a_row, start = [], {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        try:
            return level1_times(shot, open_group)
        except Exception as error:
            print(f"shot {shot}: skipped, {type(error).__name__}: {str(error)[:120]}", flush=True)
            return None, "odd"

    try:
        futures = {pool.submit(one, int(s)): int(s) for s in ids}
        for done, future in enumerate(as_completed(futures), 1):
            frame, status[futures[future]] = future.result()
            if frame is not None:
                frames.append(frame)
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % 200 == 0 or done == len(futures):
                say(f"[{done}/{len(futures)}] {(time.time() - start) / done:.2f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    table = (pd.concat(frames, ignore_index=True) if frames else
             pd.DataFrame(columns=["shot_id", "time", "converged"]))
    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "absent", "unreachable", "odd")}
    differ = int(table.groupby("shot_id")["converged"].apply(lambda c: not c.all()).sum()) if len(table) else 0
    report = {"task": "efm_times", "shots_requested": len(status), "shots_with_data": counts["ok"],
              "absent": counts["absent"], "failed": counts["unreachable"], "odd": counts["odd"],
              "shots_with_unconverged": differ, "seconds": round(time.time() - start)}
    return table, report


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
    parser.add_argument("--level1", action="store_true", help="read the level-1 efm attempted/converged times instead")
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
        run = fetch_level1 if args.level1 else fetch
        table, report = run(ids, args.workers, say=lambda line: print(line, flush=True))
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)
    report.update(part=args.part, parts=args.parts)
    os.makedirs(folder(), exist_ok=True)
    table.to_parquet(os.path.join(folder(), LEVEL1_TABLE if args.level1 else TABLE), index=False)
    with open(os.path.join(folder(), "manifest.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(report, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
