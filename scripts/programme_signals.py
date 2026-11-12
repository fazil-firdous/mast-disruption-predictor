"""What else was programmed? A look at the archive before any analysis.

The twins share the programmed plasma current (pulse_schedule/i_plasma). A published list of the archive's signals
(arXiv:2602.10132, its table of signals) also names a reference plasma density (pulse_schedule/n_e_line) and the gas
injected (gas_injection/total_injected). This script checks, on a sample of twin pairs, whether those are there for
the shots of this project and how alike the two shots of a pair are in them:

- which variables each shot's pulse_schedule and gas_injection groups hold;
- how often the reference density and the injected gas are present and not constant, by campaign;
- for each sampled pair, the relative difference of the two shots in each (as scripts/twin_check.py compares the
  programmed current: RMS of the difference over the larger maximum, up to the lost shot's loss).

    python scripts/programme_signals.py              # 60 pairs spread over the twin list; needs the network
    python scripts/programme_signals.py --pairs 150 --workers 8

It reads the first shot alone and prints what it found and how long that took, then one line per shot as it
finishes. Every request to the store has a time limit (REQUEST_S); a shot that fails is counted, not fatal.

Writes RESULTS_DIR/programme_signals/ (shots.csv, pairs.csv, summary.txt). Reads no model score and no result of
any experiment. Outside every stamp.
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import twin_check, v3_runs
from src import config, data

GROUPS = ("pulse_schedule", "gas_injection")
TARGETS = {"density_ref": ("pulse_schedule", "n_e_line"), "gas_injected": ("gas_injection", "total_injected")}
NAMES = {"density_ref": "reference density (pulse_schedule/n_e_line)",
         "gas_injected": "gas injected (gas_injection/total_injected)"}
SAME_RMS = twin_check.SAME_RMS  # the cut used for the programmed current
REQUEST_S = 60  # time limit of one request to the store; a stalled connection then fails and is counted
ABSENT = ("GroupNotFoundError", "KeyError", "FileNotFoundError")  # errors that mean 'this shot has no such group'
GIVE_UP_AFTER = 8  # shots in a row that the store did not answer for


def open_group(shot_id, group):
    """One group of one shot's store, like src.data.open_group, with a time limit on every request."""
    import aiohttp
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    options = ({"client_kwargs": {"trust_env": True, "timeout": aiohttp.ClientTimeout(total=REQUEST_S)}}
               if url.startswith("http") else None)
    return xr.open_zarr(url, group=group, consolidated=True, storage_options=options)


def folder():
    return os.path.join(config.RESULTS_DIR, "programme_signals")


def variables(ds):
    """'name[dims]' of every data variable of an opened group, sorted."""
    return sorted(f"{name}[{','.join(str(d) for d in ds[name].dims)}]" for name in ds.data_vars)


def series(ds, variable):
    """(time, values) of a variable that has one value per time, finite samples only, or None."""
    if variable not in ds or "time" not in ds[variable].dims or ds[variable].ndim != 1:
        return None
    t = np.asarray(ds[variable]["time"].values, dtype=float)
    v = np.asarray(ds[variable].values, dtype=float)
    ok = np.isfinite(t) & np.isfinite(v)
    return (t[ok], v[ok]) if ok.sum() > 10 else None


def describe(shot_id, open_group=open_group):
    """(row, {target: series or None}) for one shot. row: the variables of each group (or the error opening it),
    per target its number of finite samples, its largest value and whether it ever changes, and the seconds taken."""
    start = time.time()
    row, found = {"shot_id": int(shot_id)}, {name: None for name in TARGETS}
    opened = {}
    for group in GROUPS:
        try:
            opened[group] = open_group(int(shot_id), group)
            row[f"{group}_vars"] = " ".join(variables(opened[group]))
        except Exception as error:
            row[f"{group}_vars"] = f"not opened: {type(error).__name__}"
    for name, (group, variable) in TARGETS.items():
        try:
            s = series(opened[group], variable) if group in opened else None
        except Exception as error:  # the list was read but the values were not
            s, row[f"{group}_vars"] = None, f"not read: {type(error).__name__}"
        found[name] = s
        row[f"{name}_samples"] = 0 if s is None else len(s[0])
        row[f"{name}_max"] = np.nan if s is None else float(np.abs(s[1]).max())
        row[f"{name}_varies"] = bool(s is not None and np.ptp(s[1]) > 0)
    row["seconds"] = round(time.time() - start, 1)
    return row, found


def sample(twins, n):
    """n pairs spread evenly over the twin list in shot order, so that every campaign is in the sample."""
    ordered = twins.sort_values("shot_id").reset_index(drop=True)
    if len(ordered) <= n:
        return ordered
    return ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, n).round().astype(int))].reset_index(drop=True)


def one_line(row):
    """One shot in one line: what was found and how long it took."""
    parts = [f"{name} {'yes' if row[f'{name}_varies'] else 'constant' if row[f'{name}_samples'] else 'no'}"
             for name in TARGETS]
    failed = [g for g in GROUPS if row[f"{g}_vars"].startswith("not ")]
    return f"shot {row['shot_id']}: " + ", ".join(parts) + (f"; {' and '.join(failed)} failed" if failed else "") \
        + f" ({row['seconds']:.1f} s)"


def unreachable(row):
    """True if neither group of this shot could be read for a reason other than the group not being there."""
    return all(row[f"{g}_vars"].startswith("not ") and not row[f"{g}_vars"].endswith(ABSENT) for g in GROUPS)


def look(pairs, t_loss, campaign, open_group=open_group, workers=4, say=print):
    """(shots, compared): one row per sampled shot, and one row per sampled pair with the relative difference of
    its two shots in each target (NaN where either shot lacks it or holds a constant, which is no programme).
    The first shot is read alone and shown in full; then the rest in parallel, one line each as it finishes."""
    ids = sorted(set(pairs["shot_id"]) | set(pairs["other"]))
    results = {ids[0]: describe(ids[0], open_group)}
    first = results[ids[0]][0]
    say(f"first shot {ids[0]}, read in {first['seconds']:.1f} s:")
    for group in GROUPS:
        say(f"  {group}: {first[f'{group}_vars'] or '(no variables)'}")
    if unreachable(first):
        raise RuntimeError("the store did not answer for the first shot; nothing was read. Try again later.")
    in_a_row = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(describe, shot, open_group): shot for shot in ids[1:]}
        for done, future in enumerate(as_completed(futures), 2):
            results[futures[future]] = future.result()
            say(f"[{done}/{len(ids)}] {one_line(results[futures[future]][0])}")
            in_a_row = in_a_row + 1 if unreachable(results[futures[future]][0]) else 0
            if in_a_row >= GIVE_UP_AFTER:
                raise RuntimeError(f"the store did not answer for {GIVE_UP_AFTER} shots in a row; stopped. Try again later.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    shots = pd.DataFrame([results[i][0] for i in ids])
    shots["campaign"] = shots["shot_id"].map(campaign)
    rows = []
    for pair in pairs.itertuples():
        row = {"shot_id": pair.shot_id, "other": pair.other, "campaign": campaign.get(pair.shot_id)}
        for name in TARGETS:
            a, b = results[pair.shot_id][1][name], results[pair.other][1][name]
            usable = a is not None and b is not None and np.ptp(a[1]) > 0 and np.ptp(b[1]) > 0
            c = twin_check.compare(a, b, t_loss.get(pair.shot_id, np.inf)) if usable else None
            row[f"{name}_rms"] = np.nan if c is None else c["rms"]
            row[f"{name}_peak_ratio"] = np.nan if c is None else c["peak_ratio"]
        rows.append(row)
    return shots, pd.DataFrame(rows)


def text(shots, compared):
    """The look in words."""
    lines = [f"What else was programmed? {len(compared)} twin pairs, {len(shots)} shots, "
             f"{shots['seconds'].median():.1f} s per shot (median).", ""]
    for group in GROUPS:
        counts = shots[f"{group}_vars"].value_counts()
        lines.append(f"{group}: the variable lists seen (shots with each):")
        lines += [f"- {n} shots: {names if names else '(no variables)'}" for names, n in counts.items()]
        lines.append("")
    for name in TARGETS:
        there, varies = shots[f"{name}_samples"] > 0, shots[f"{name}_varies"]
        lines.append(f"{NAMES[name]}: present in {there.mean():.0%} of the shots, not constant in {varies.mean():.0%}.")
        by = shots.groupby("campaign", dropna=False)[f"{name}_varies"].agg(["mean", "size"])
        lines.append("  not constant, by campaign: " + "; ".join(f"{c} {m:.0%} of {int(n)}" for c, (m, n) in by.iterrows()))
        rms = compared[f"{name}_rms"].dropna()
        if len(rms):
            lines.append(f"  pairs with it, not constant, in both shots: {len(rms)} of {len(compared)}. Relative difference: median "
                         f"{rms.median():.3f}; under {SAME_RMS:.0%} in {(rms < SAME_RMS).mean():.0%}, under 1% in "
                         f"{(rms < 0.01).mean():.0%}.")
        else:
            lines.append("  no sampled pair has it, not constant, in both shots.")
        lines.append("")
    return "\n".join(lines).rstrip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=60, help="how many twin pairs to sample")
    parser.add_argument("--workers", type=int, default=4, help="shots read at the same time")
    args = parser.parse_args()
    try:
        import zarr
        zarr.config.set({"async.timeout": 3 * REQUEST_S})  # a second limit, in case a request never returns
    except Exception:
        pass
    start = time.time()
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_loss = labels.set_index("shot_id")["t_quench"].to_dict()
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
    print(f"tables read in {time.time() - start:.1f} s; reading the store ...", flush=True)
    try:
        shots, compared = look(sample(twins, args.pairs), t_loss, campaign, workers=args.workers,
                               say=lambda line: print(line, flush=True))
    except RuntimeError as error:
        print(f"STOPPED: {error}", flush=True)
        os._exit(1)  # do not wait for requests still in flight
    os.makedirs(folder(), exist_ok=True)
    shots.to_csv(os.path.join(folder(), "shots.csv"), index=False)
    compared.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    out = text(shots, compared)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
