"""Runs where the store answers (a GitHub machine, .github/workflows/fetch_saddle.yml): step 1 of
docs/preregistration_saddle_correction.md. Per shot, from the store as it is today: the locked-mode amplitude exactly as
src/extras.py computes it (lm_raw) and with the documented sign correction of saddle channels M02, M08 and M012 for
shots 24782 to 30473 (lm_corr); per channel, its label, whether it has data, and its correlation with the median of the
other channels; the selected Mirnov channel; the groups' ingestion attributes; and, for the frozen sample only, the
summary signals and offset-removed 1 ms channel bins as stored today. No outcome, model or label passes through.

    python scripts/saddle_fetch.py --ids <zlib+base64 list> --sample <list> --part 0 --parts 8
    python scripts/saddle_fetch.py --shots 27000,30421

Writes DATA_DIR/saddle_fetch/{lm,channels,shots,summary_sample,channel_bins}.parquet and manifest.json.
"""
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import programme_signals, sxr_fetch
from src import config, data, extras

FIELD = "b_field_tor_probe_saddle_field"
MIRNOV = "b_field_pol_probe_cc_field"
FLIP = ("M02", "M08", "M012")  # documented sign reversal (FAIR-MAST level 2, changes from source)
FLIP_FROM, FLIP_TO = 24782, 30473
MIN_BINS = 20  # fewer common bins give no correlation
GIVE_UP_AFTER = 20
TABLES = ("lm", "channels", "shots", "summary_sample", "channel_bins")


def folder():
    return os.path.join(config.DATA_DIR, "saddle_fetch")


def labels_of(arr):
    """The channel labels of a (channel, time) array, as strings; positions if it has no label coordinate."""
    axis = arr.dims[0]
    return [str(c) for c in arr[axis].values] if axis in arr.coords else [str(i) for i in range(arr.shape[0])]


def signs_for(shot_id, labels):
    """+1 or -1 per channel: -1 for the documented reversed channels inside the documented shot range."""
    flip = FLIP_FROM <= int(shot_id) <= FLIP_TO
    return np.array([-1.0 if flip and label in FLIP else 1.0 for label in labels])


def channel_bins(t_fast, field, t_grid, signs):
    """(offset-removed 1 ms bin means, channels x grid; the time each channel's offset is known), as src/extras.py,
    with each channel multiplied by its sign before anything else."""
    field = np.asarray(field, dtype=float) * signs[:, None]
    binned = np.vstack([extras.bin_mean(t_fast, channel, t_grid) for channel in field])
    offsets, known = zip(*[extras.early_offset(t_fast, channel) for channel in field])
    return binned - np.array(offsets)[:, None], np.array(known, dtype=float)


def lm_from(deltas, known, t_grid):
    """lm_n1 from the offset-removed bins, exactly as src/extras.saddle_n1 does."""
    amplitude = extras.n1_amplitude(deltas, extras.SADDLE_PHI_DEG[:deltas.shape[0]])
    finite_known = known[np.isfinite(known)]
    if len(finite_known):
        amplitude[t_grid < finite_known.max()] = np.nan
    return amplitude


def channel_r(deltas, defined):
    """Per channel: Pearson r between its bins and the median of the other channels' bins, over the bins in `defined`
    where both are finite (NaN with fewer than MIN_BINS, or no variation)."""
    out = np.full(deltas.shape[0], np.nan)
    for c in range(deltas.shape[0]):
        others = np.delete(deltas, c, axis=0)
        with warnings.catch_warnings():  # all-NaN bins give NaN, which is wanted
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(others, axis=0) if len(others) else np.full(deltas.shape[1], np.nan)
        ok = defined & np.isfinite(deltas[c]) & np.isfinite(median)
        if ok.sum() < MIN_BINS or np.std(deltas[c][ok]) == 0 or np.std(median[ok]) == 0:
            continue
        out[c] = float(np.corrcoef(deltas[c][ok], median[ok])[0, 1])
    return out


def attrs_of(ds):
    return str(ds.attrs.get("ingested_at", "")), str(ds.attrs.get("commit_url", ""))


def shot_result(shot_id, open_group=programme_signals.open_group, sample=False):
    """({table: frame}, status) for one shot; status 'ok', 'no_saddle', 'absent' or 'unreachable'."""
    row = {"shot_id": int(shot_id)}
    try:
        summary = open_group(int(shot_id), "summary")
        frame = data._summary_to_frame(summary)
    except Exception as error:
        return {"shots": pd.DataFrame([{**row, "status": "absent"}])}, (
            "absent" if type(error).__name__ in programme_signals.ABSENT else "unreachable")
    row["sum_ingested_at"], row["sum_commit_url"] = attrs_of(summary)
    t_grid = frame["time"].to_numpy(dtype=float)
    row["grid_len"] = len(t_grid)
    out = {}
    if sample:
        out["summary_sample"] = frame.assign(shot_id=np.int32(shot_id))
    try:
        mag = open_group(int(shot_id), "magnetics")
    except Exception as error:
        if type(error).__name__ not in programme_signals.ABSENT:
            return out, "unreachable"
        out["shots"] = pd.DataFrame([{**row, "status": "no_saddle"}])
        return out, "no_saddle"
    row["mag_ingested_at"], row["mag_commit_url"] = attrs_of(mag)
    if MIRNOV in mag:
        arr = mag[MIRNOV]
        names = labels_of(arr)
        for k in range(arr.shape[0]):
            if np.isfinite(np.asarray(arr[k].values, dtype=float)).any():
                row["mirnov_index"], row["mirnov_label"] = k, names[k]
                break
    if FIELD not in mag:
        out["shots"] = pd.DataFrame([{**row, "status": "no_saddle"}])
        return out, "no_saddle"
    arr = mag[FIELD]
    labels = labels_of(arr)
    t_fast = np.asarray(mag["time_saddle"].values, dtype=float)
    field = np.asarray(arr.values, dtype=float)
    signs = signs_for(shot_id, labels)
    raw, known = channel_bins(t_fast, field, t_grid, np.ones(len(labels)))
    lm_raw = lm_from(raw, known, t_grid)
    corr, known_c = channel_bins(t_fast, field, t_grid, signs)
    lm_corr = lm_from(corr, known_c, t_grid)
    defined = np.isfinite(lm_raw)
    finite = np.isfinite(field).any(axis=1)
    out["lm"] = pd.DataFrame({"shot_id": np.int32(shot_id), "time": t_grid, "lm_raw": lm_raw, "lm_corr": lm_corr})
    out["channels"] = pd.DataFrame({"shot_id": np.int32(shot_id), "index": np.arange(len(labels), dtype=np.int16),
                                    "label": labels, "finite": finite, "sign": signs,
                                    "r_raw": channel_r(raw, defined), "r_corr": channel_r(corr, defined)})
    if sample:  # the frozen shots' channel inputs, kept so any later diagnostic uses the same frozen data
        out["channel_bins"] = pd.DataFrame({
            "shot_id": np.int32(shot_id), "label": np.repeat(labels, len(t_grid)),
            "time": np.tile(t_grid, len(labels)), "defined": np.tile(defined, len(labels)), "delta": raw.ravel()})
    row.update(status="ok", n_channels=len(labels), n_finite=int(finite.sum()), labels=",".join(labels),
               flip_applied=bool((signs < 0).any()), field_units=str(arr.attrs.get("units", "")))
    out["shots"] = pd.DataFrame([row])
    return out, "ok"


def fetch(ids, sample=(), workers=8, open_group=programme_signals.open_group, say=print, deadline=None):
    ids, sample = [int(i) for i in ids], {int(i) for i in sample}
    parts = {name: [] for name in TABLES}
    status, in_a_row, start = {}, 0, time.time()
    pool = ThreadPoolExecutor(max_workers=workers)

    def one(shot):
        if deadline is not None and time.time() > deadline:
            return {}, "not_started"
        try:
            return shot_result(shot, open_group, shot in sample)
        except Exception as error:  # stored in a way not seen before: counted, never fatal
            print(f"shot {shot}: skipped, {type(error).__name__}: {str(error)[:120]}", flush=True)
            return {"shots": pd.DataFrame([{"shot_id": shot, "status": "odd"}])}, "odd"

    try:
        futures = {pool.submit(one, shot): shot for shot in ids}
        for done, future in enumerate(as_completed(futures), 1):
            got, status[futures[future]] = future.result()
            for name, frame in got.items():
                parts[name].append(frame)
            in_a_row = in_a_row + 1 if status[futures[future]] == "unreachable" else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row")
            if done % 100 == 0 or done == len(ids):
                say(f"[{done}/{len(ids)}] {(time.time() - start) / done:.2f} s per shot")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    tables = {name: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"shot_id": []})
              for name, frames in parts.items()}
    report = {"task": "saddle_fetch", "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
              "shots_requested": len(ids), "seconds": round(time.time() - start),
              **{s: sum(v == s for v in status.values()) for s in ("ok", "no_saddle", "absent", "unreachable", "odd",
                                                                   "not_started")}}
    return tables, report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", default="")
    parser.add_argument("--sample", default="", help="the frozen sample, encoded like --ids")
    parser.add_argument("--shots", default="")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--hours", type=float, default=5.0, help="start no shot after this long")
    args = parser.parse_args()
    try:
        import zarr
        zarr.config.set({"async.timeout": 3 * programme_signals.REQUEST_S})
    except Exception:
        pass
    ids = sxr_fetch.decode_ids(args.ids) if args.ids else [int(s) for s in args.shots.split(",") if s.strip()]
    sample = sxr_fetch.decode_ids(args.sample) if args.sample else (ids if args.shots else [])
    ids = sorted(ids)[args.part::args.parts]
    print(f"{len(ids)} shots", flush=True)
    try:
        tables, report = fetch(ids, sample, args.workers, say=lambda line: print(line, flush=True),
                               deadline=time.time() + args.hours * 3600)
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)
    report.update(part=args.part, parts=args.parts)
    os.makedirs(folder(), exist_ok=True)
    for name, table in tables.items():
        table.to_parquet(os.path.join(folder(), f"{name}.parquet"), index=False)
    with open(os.path.join(folder(), "manifest.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(report, flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
