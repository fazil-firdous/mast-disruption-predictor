"""Runs where the store answers (a GitHub machine, .github/workflows/fetch_sxr.yml): read the soft X-ray cameras of
the given shots and write, per shot, camera and chord, how good the signal is. No outcome, no model, no label.

Per chord: when its samples start and stop, the share of the plasma's time it covers, its noise before the plasma
(the standard deviation over t < 0), its typical level and spread during the plasma, a signal-to-noise ratio
(spread over noise), and the share of samples stuck at the chord's maximum (saturation). Per shot: when the plasma
was on (|Ip| above 50 kA in the summary group) and which cameras exist.

    python scripts/sxr_fetch.py --ids <zlib+base64 list> --part 0 --parts 4
    python scripts/sxr_fetch.py --shots 11802,27000          # a few shots by number

Writes DATA_DIR/sxr_audit/sxr_audit.parquet and manifest.json.
"""
import argparse
import base64
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import programme_signals
from src import config

IP_ON = 5e4  # A: the plasma counts as on above this current
GIVE_UP_AFTER = 20
WORK_HOURS = 4.5
TABLE = "sxr_audit.parquet"


def folder():
    return os.path.join(config.DATA_DIR, "sxr_audit")


def encode_ids(ids):
    """A compact text form of a list of shot numbers, small enough for a workflow input."""
    ids = sorted(set(int(i) for i in ids))
    deltas = [ids[0]] + [b - a for a, b in zip(ids, ids[1:])] if ids else []
    return base64.b64encode(zlib.compress(",".join(map(str, deltas)).encode(), 9)).decode()


def decode_ids(text):
    if not text:
        return []
    deltas = [int(x) for x in zlib.decompress(base64.b64decode(text)).decode().split(",")]
    return list(np.cumsum(deltas).astype(int))


def chord_stats(t, x, t_on, t_off):
    """Quality of one chord. t, x: time and samples; t_on, t_off: the plasma's start and end (NaN if unknown)."""
    x = np.asarray(x, dtype=float)
    fin = np.isfinite(x)
    out = {"t_first": np.nan, "t_last": np.nan, "covered": np.nan, "noise": np.nan, "level": np.nan, "spread": np.nan,
           "snr": np.nan, "saturated": np.nan}
    if not fin.any():
        return out
    out["t_first"], out["t_last"] = float(t[fin][0]), float(t[fin][-1])
    pre = fin & (t < 0)
    if pre.sum() > 50:
        out["noise"] = float(np.std(x[pre]))
    if np.isfinite(t_on) and np.isfinite(t_off) and t_off > t_on:
        during = (t >= t_on) & (t <= t_off)
        out["covered"] = float(fin[during].mean()) if during.any() else np.nan
        sig = x[during & fin]
        if len(sig) > 50:
            p5, p50, p95 = np.percentile(sig, [5, 50, 95])
            out["level"], out["spread"] = float(p50), float(p95 - p5)
            if np.isfinite(out["noise"]) and out["noise"] > 0:
                out["snr"] = out["spread"] / out["noise"]
            top = np.max(sig)
            out["saturated"] = float(np.mean(np.isclose(sig, top, rtol=0, atol=1e-9 * max(abs(top), 1))))
    return out


def plasma_window(summary):
    """(t_on, t_off) from the summary group's plasma current, or NaNs."""
    try:
        ip = np.abs(np.asarray(summary["ip"].values, dtype=float))
        t = np.asarray(summary["time"].values, dtype=float)
        on = t[ip > IP_ON]
        return (float(on[0]), float(on[-1])) if len(on) else (np.nan, np.nan)
    except Exception:
        return np.nan, np.nan


def shot_rows(shot_id, open_group=programme_signals.open_group):
    """(rows, status): one row per camera and chord. status 'ok', 'absent' (no soft X-ray group) or 'unreachable'."""
    try:
        summary = open_group(int(shot_id), "summary")
        t_on, t_off = plasma_window(summary)
    except Exception as error:
        if type(error).__name__ in programme_signals.ABSENT:
            t_on, t_off = np.nan, np.nan
        else:
            return [], "unreachable"
    try:
        ds = open_group(int(shot_id), "soft_x_rays")
    except Exception as error:
        status = "absent" if type(error).__name__ in programme_signals.ABSENT else "unreachable"
        return [{"shot_id": int(shot_id), "camera": "", "chord": "", "t_on": t_on, "t_off": t_off}], status
    rows = []
    t = np.asarray(ds["time"].values, dtype=float)
    for cam in ds.data_vars:
        arr = ds[cam]
        if arr.ndim != 2 or "time" not in arr.dims:
            continue
        axis = [d for d in arr.dims if d != "time"][0]
        names = [str(c) for c in ds[axis].values] if axis in ds.coords else [str(i) for i in range(arr.sizes[axis])]
        values = np.asarray(arr.transpose(axis, "time").values, dtype=np.float32)
        for name, x in zip(names, values):
            rows.append({"shot_id": int(shot_id), "camera": cam, "chord": name, "t_on": t_on, "t_off": t_off,
                         **chord_stats(t, x, t_on, t_off)})
    return rows, "ok"


def fetch(ids, workers=8, open_group=programme_signals.open_group, say=print, deadline=None):
    ids = [int(i) for i in ids]
    rows, status, in_a_row, start = [], {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        if deadline is not None and time.time() > deadline:
            return [], "not_started"
        return shot_rows(shot, open_group)

    try:
        futures = {pool.submit(one, shot): shot for shot in ids}
        for done, future in enumerate(as_completed(futures), 1):
            got, status[futures[future]] = future.result()
            rows += got
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % 50 == 0 or done == len(ids):
                say(f"[{done}/{len(ids)}] {(time.time() - start) / done:.1f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    columns = ["shot_id", "camera", "chord", "t_on", "t_off", "t_first", "t_last", "covered", "noise", "level", "spread",
               "snr", "saturated"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    report = {"task": "sxr_audit", "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
              "shots_requested": len(ids), "shots_with_data": sum(s == "ok" for s in status.values()),
              "absent": sum(s == "absent" for s in status.values()),
              "failed": sum(s == "unreachable" for s in status.values()),
              "not_started": sum(s == "not_started" for s in status.values()), "seconds": round(time.time() - start)}
    return table, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default="", help="the shot list, as encode_ids() writes it")
    parser.add_argument("--shots", default="", help="or a few shot numbers, comma separated")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    try:
        import zarr
        zarr.config.set({"async.timeout": 3 * programme_signals.REQUEST_S})
    except Exception:
        pass
    ids = decode_ids(args.ids) if args.ids else [int(s) for s in args.shots.split(",") if s.strip()]
    ids = sorted(ids)[args.part::args.parts]
    print(f"{len(ids)} shots, {args.workers} at a time", flush=True)
    try:
        table, report = fetch(ids, args.workers, say=lambda line: print(line, flush=True),
                              deadline=time.time() + WORK_HOURS * 3600)
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)
    report.update(part=args.part, parts=args.parts)
    os.makedirs(folder(), exist_ok=True)
    table.to_parquet(os.path.join(folder(), TABLE), index=False)
    with open(os.path.join(folder(), "manifest.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"{report['shots_with_data']} shots with soft X-rays, {report['absent']} without, {report['failed']} failed; "
          f"{len(table):,} rows, {report['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
