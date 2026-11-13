"""Runs where the store answers (a GitHub machine, .github/workflows/fetch_sxr_series.yml): the soft X-ray signals of
docs/preregistration_soft_xrays.md, per shot, camera and 1 ms bin. No outcome, no model, no label passes through.

Per chord: the offset is the mean before t = 0; a chord with fewer than MIN_PRE such samples, or none that differ,
is left out for the shot. Per 1 ms bin (k-1, k] ms: each chord's mean minus its offset, empty where the chord does
not change at all within the bin (stuck or saturated). With at least MIN_CHORDS chords, per camera and bin:
level, position, width, peak and fluct (see the protocol). The value of bin k is known at k ms.

    python scripts/sxr_series.py --ids <zlib+base64 list> --part 0 --parts 8
    python scripts/sxr_series.py --shots 11880,11876

Writes DATA_DIR/sxr_series/sxr_series.parquet and manifest.json.
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

CAMERAS = ("horizontal_cam_lower", "horizontal_cam_upper")
SIGNALS = ("level", "position", "width", "peak", "fluct")
MIN_PRE, MIN_CHORDS = 50, 12
BEFORE_MS, AFTER_MS = 25, 5  # bins kept from 25 ms before the plasma starts to 5 ms after it ends
GIVE_UP_AFTER = 20
WORK_HOURS = 4.5
TABLE = "sxr_series.parquet"
COLUMNS = ["shot_id", "camera", "k_ms", "n_chords", *SIGNALS]


def folder():
    return os.path.join(config.DATA_DIR, "sxr_series")


def camera_bins(t, x, k_from=None, k_to=None):
    """{signal: values}, k (bin ends in ms) and the number of chords per bin, for one camera.

    t: sample times (s); x: chords x samples. Bins (k-1, k] ms for k from k_from to k_to (default: all recorded)."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    ks = np.ceil(t * 1000 - 1e-6).astype(np.int64)  # sample at time t belongs to bin k with k-1 < t*1000 <= k
    lo = ks.min() if k_from is None else max(int(k_from), ks.min())
    hi = ks.max() if k_to is None else min(int(k_to), ks.max())
    k = np.arange(lo, hi + 1)
    n_bins = len(k)
    empty = {s: np.full(n_bins, np.nan) for s in SIGNALS}
    if n_bins <= 0:
        return empty, k, np.zeros(0, dtype=int)
    idx = ks - lo
    inside = (idx >= 0) & (idx < n_bins)
    pre = t < 0
    means = np.full((x.shape[0], n_bins), np.nan)
    raw = np.zeros((x.shape[0], len(t)))
    finite = np.ones(len(t), dtype=bool)  # samples where every chord kept for the shot has a value
    kept = np.zeros(x.shape[0], dtype=bool)
    for c in range(x.shape[0]):
        xc = x[c]
        before = xc[pre & np.isfinite(xc)]
        if len(before) < MIN_PRE or np.all(before == before[0]):
            continue  # no causal offset, or a dead chord
        kept[c] = True
        xc = xc - before.mean()
        ok = inside & np.isfinite(xc)
        sums = np.bincount(idx[ok], weights=xc[ok], minlength=n_bins)
        counts = np.bincount(idx[ok], minlength=n_bins)
        change = np.zeros(len(t), dtype=bool)
        change[1:] = np.diff(xc) != 0
        changes = np.bincount(idx[ok], weights=change[ok], minlength=n_bins)
        with np.errstate(invalid="ignore", divide="ignore"):
            m = sums / counts
        m[(counts == 0) | ((changes == 0) & (counts >= 2))] = np.nan  # no samples, or the chord did not move at all
        means[c] = m
        raw[c] = np.where(np.isfinite(xc), xc, 0.0)
        finite &= np.isfinite(xc)
    n = np.isfinite(means).sum(axis=0)
    good = n >= MIN_CHORDS
    out = dict(empty)
    if not good.any():
        return out, k, n
    y = (np.arange(x.shape[0]) / (x.shape[0] - 1))[:, None]
    bp = np.where(np.isfinite(means), np.maximum(means, 0.0), 0.0)
    total = bp.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        level = np.nanmean(np.where(np.isfinite(means), means, np.nan), axis=0)
        position = (y * bp).sum(axis=0) / total
        width = np.sqrt((bp * (y - position) ** 2).sum(axis=0) / total)
        peak = bp.max(axis=0) / total
    # fast fluctuation: the camera's summed signal (chords kept for the shot) within each bin
    s = raw[kept].sum(axis=0)
    ok = inside & finite
    cnt = np.bincount(idx[ok], minlength=n_bins)
    s1 = np.bincount(idx[ok], weights=s[ok], minlength=n_bins)
    s2 = np.bincount(idx[ok], weights=s[ok] ** 2, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = np.maximum(s2 / cnt - (s1 / cnt) ** 2, 0.0)
        fluct = np.sqrt(var) / max(int(kept.sum()), 1)
    fluct[cnt < 2] = np.nan
    shape_ok = good & (total > 0)  # position, width and peak need some positive emission
    for name, v, keep in (("level", level, good), ("position", position, shape_ok), ("width", width, shape_ok),
                          ("peak", peak, shape_ok), ("fluct", fluct, good)):
        out[name] = np.where(keep, v, np.nan)
    return out, k, n


def shot_frame(shot_id, open_group=programme_signals.open_group):
    """(frame, status): one row per camera and bin; status 'ok', 'absent' or 'unreachable'."""
    try:
        t_on, t_off = sxr_fetch.plasma_window(open_group(int(shot_id), "summary"))
    except Exception as error:
        if type(error).__name__ not in programme_signals.ABSENT:
            return None, "unreachable"
        t_on, t_off = np.nan, np.nan
    try:
        ds = open_group(int(shot_id), "soft_x_rays")
    except Exception as error:
        return None, "absent" if type(error).__name__ in programme_signals.ABSENT else "unreachable"
    if "time" not in ds.variables:
        return None, "absent"
    k_from = int(np.floor(t_on * 1000)) - BEFORE_MS if np.isfinite(t_on) else 1
    k_to = int(np.ceil(t_off * 1000)) + AFTER_MS if np.isfinite(t_off) else None
    t = np.asarray(ds["time"].values, dtype=float)
    frames = []
    for code, cam in enumerate(CAMERAS):
        if cam not in ds.data_vars or ds[cam].ndim != 2 or "time" not in ds[cam].dims:
            continue
        arr = ds[cam]
        axis = [d for d in arr.dims if d != "time"][0]
        values, k, n = camera_bins(t, arr.transpose(axis, "time").values, k_from, k_to)
        if len(k) == 0:
            continue
        frames.append(pd.DataFrame({"shot_id": np.int32(shot_id), "camera": np.int8(code), "k_ms": k.astype(np.int32),
                                    "n_chords": n.astype(np.int8), **{s: values[s].astype(np.float32) for s in SIGNALS}}))
    if not frames:
        return None, "absent"
    return pd.concat(frames, ignore_index=True), "ok"


def fetch(ids, workers=8, open_group=programme_signals.open_group, say=print, deadline=None):
    ids = [int(i) for i in ids]
    frames, status, in_a_row, start = [], {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        if deadline is not None and time.time() > deadline:
            return None, "not_started"
        try:
            return shot_frame(shot, open_group)
        except Exception as error:  # stored in a way not seen before: counted, never fatal
            print(f"shot {shot}: skipped, {type(error).__name__}: {str(error)[:120]}", flush=True)
            return None, "odd"

    try:
        futures = {pool.submit(one, shot): shot for shot in ids}
        for done, future in enumerate(as_completed(futures), 1):
            frame, status[futures[future]] = future.result()
            if frame is not None:
                frames.append(frame)
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % 100 == 0 or done == len(ids):
                say(f"[{done}/{len(ids)}] {(time.time() - start) / done:.2f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    counts = {s: sum(v == s for v in status.values()) for s in ("ok", "absent", "unreachable", "odd", "not_started")}
    report = {"task": "sxr_series", "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
              "shots_requested": len(ids), "shots_with_data": counts["ok"], "absent": counts["absent"],
              "failed": counts["unreachable"], "odd": counts["odd"], "not_started": counts["not_started"],
              "seconds": round(time.time() - start)}
    return table[COLUMNS], report


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
    print(f"{report['shots_with_data']} shots with soft X-rays, {report['absent']} without, {report['failed']} failed, "
          f"{report['odd']} odd, {report['not_started']} not started; {len(table):,} rows, {report['seconds']} s", flush=True)
    os._exit(0)  # the store's threads can keep a finished job alive


if __name__ == "__main__":
    main()
