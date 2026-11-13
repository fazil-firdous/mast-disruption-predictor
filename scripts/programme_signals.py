"""What else was programmed? A look at the archive before any analysis.

The twins share the programmed plasma current (pulse_schedule/i_plasma). The archive holds more of what was asked
of the machine and of what gas it was given. Read from real shots on 2026-11-12:

- pulse_schedule: i_plasma, n_e_line (a reference density; zero where there was none), z_ref (the requested height);
- gas_injection: valve_target_voltage (the requested voltage of each gas valve; empty in many shots), valve_voltage
  (the voltage each valve was driven with), inboard_total and outboard_total (gas flow), and in most shots from M6
  on total_injected (gas so far) and pressure.

This script says how often each is there and not constant, by campaign, and, for twin pairs, how alike the two
shots of a pair are in each: the RMS of the difference over the larger maximum, and in the signal's own unit, up
to the lost shot's loss and up to EARLY_MS before it.

The research server cannot reach the store (its address range is blocked there), so the signals come through
GitHub like the shots do: scripts/programme_fetch.py runs on a GitHub machine (.github/workflows/fetch_programme.yml)
and this script collects its table.

    python scripts/programme_signals.py --relay      # start the GitHub job, wait, collect its table (an hour or two)
    python scripts/programme_signals.py --collect    # collect the newest job's table (waits if it is still running)
    python scripts/programme_signals.py              # every twin pair, from the collected table; no network
    python scripts/programme_signals.py --shots 150  # where the store answers: 150 shots spread over the shot table

Writes RESULTS_DIR/programme_signals/ (shots.csv, pairs.csv, summary.txt). Reads no model score and no result of
any experiment. Outside every stamp.
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import twin_check, v3_runs
from src import config, data, relay

GROUPS = ("pulse_schedule", "gas_injection", "tf", "pf_active")
TARGETS = {"density_ref": ("pulse_schedule", "n_e_line"), "height_ref": ("pulse_schedule", "z_ref"),
           "valves_ref": ("gas_injection", "valve_target_voltage"), "valves_drive": ("gas_injection", "valve_voltage"),
           "gas_inboard": ("gas_injection", "inboard_total"), "gas_outboard": ("gas_injection", "outboard_total"),
           "gas_injected": ("gas_injection", "total_injected"), "gas_pressure": ("gas_injection", "pressure"),
           "tf_current": ("tf", "coil_current"),
           "coil_p2": ("pf_active", "coil_current", ("P2IL FEED", "P2IU FEED", "P2OL FEED", "P2OU FEED")),
           "coil_p4": ("pf_active", "coil_current", ("P4L FEED", "P4U FEED")),
           "coil_p5": ("pf_active", "coil_current", ("P5L FEED", "P5U FEED"))}
NAMES = {"density_ref": "reference density (pulse_schedule/n_e_line)",
         "height_ref": "requested height (pulse_schedule/z_ref)",
         "valves_ref": "requested gas valve voltages, summed over valves (gas_injection/valve_target_voltage)",
         "valves_drive": "gas valve drive voltages, summed over valves (gas_injection/valve_voltage)",
         "gas_inboard": "gas flow inboard (gas_injection/inboard_total)",
         "gas_outboard": "gas flow outboard (gas_injection/outboard_total)",
         "gas_injected": "gas injected so far (gas_injection/total_injected)",
         "gas_pressure": "gas pressure (gas_injection/pressure)",
         "tf_current": "toroidal field coil current (tf/coil_current)",
         "coil_p2": "P2 coil currents, summed (pf_active/coil_current)",
         "coil_p4": "P4 shape coil currents, summed (pf_active/coil_current)",
         "coil_p5": "P5 shape coil currents, summed (pf_active/coil_current)"}


def target(name):
    """(group, variable, channels or None) of one signal."""
    spec = TARGETS[name]
    return spec if len(spec) == 3 else (*spec, None)
SAME_RMS = twin_check.SAME_RMS  # the cut used for the programmed current
REQUEST_S = 60  # time limit of one request to the store; a stalled connection then fails and is counted
ABSENT = ("GroupNotFoundError", "KeyError", "FileNotFoundError")  # errors that mean 'this shot has no such group'
GIVE_UP_AFTER = 8  # shots in a row that the store did not answer for
EARLY_MS = 100  # 'early': up to this long before the loss, as for the beam heating in scripts/v3_explore.py
WORKFLOW = "fetch_programme.yml"  # the GitHub job that reads the store for the server
ARTIFACT = "mast-programme"
TABLE = "programme.parquet"
STATES = ("both", "both_constant", "constants_differ", "one", "missing")


def data_folder():
    """Where the collected table lives: DATA_DIR/programme."""
    return os.path.join(config.DATA_DIR, "programme")


def folder():
    return os.path.join(config.RESULTS_DIR, "programme_signals")


def open_group(shot_id, group):
    """One group of one shot's store, like src.data.open_group, with a time limit on every request."""
    import aiohttp
    url = config.SHOT_ZARR_URL.format(shot_id=shot_id)
    options = ({"client_kwargs": {"trust_env": True, "timeout": aiohttp.ClientTimeout(total=REQUEST_S)}}
               if url.startswith("http") else None)
    return xr.open_zarr(url, group=group, consolidated=True, storage_options=options)


def variables(ds):
    """'name[dims]' of every data variable of an opened group, sorted."""
    return sorted(f"{name}[{','.join(str(d) for d in ds[name].dims)}]" for name in ds.data_vars)


def series(ds, variable, channels=None):
    """(time, values) of a variable along time, finite samples only, or None. A variable with one more axis (one
    row per valve or coil) is summed over it, or over the named channels only, so that it becomes one value per
    time. A shot that lacks one of the named channels has no series."""
    if variable not in ds or "time" not in ds[variable].dims or ds[variable].ndim > 2:
        return None
    array = ds[variable].transpose(..., "time")
    if channels is not None:
        if array.ndim != 2:
            return None
        axis = array.dims[0]
        names = [str(c) for c in ds[axis].values] if axis in ds.coords or axis in ds else []
        if not set(channels) <= set(names):
            return None
        array = array.isel({axis: [names.index(c) for c in channels]})
    t = np.asarray(array["time"].values, dtype=float)
    v = np.asarray(array.values, dtype=float)
    if v.ndim == 2:
        v = np.where(np.isfinite(v).any(axis=0), np.nansum(v, axis=0), np.nan)
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
    for name in TARGETS:
        group, variable, channels = target(name)
        try:
            s = series(opened[group], variable, channels) if group in opened else None
        except Exception as error:  # the list was read but the values were not
            s, row[f"{group}_vars"] = None, f"not read: {type(error).__name__}"
        found[name] = s
        row[f"{name}_samples"] = 0 if s is None else len(s[0])
        row[f"{name}_max"] = np.nan if s is None else float(np.abs(s[1]).max())
        row[f"{name}_varies"] = bool(s is not None and np.ptp(s[1]) > 0)
    row["seconds"] = round(time.time() - start, 1)
    return row, found


def sample(table, n):
    """n rows spread evenly over a table in shot order, so that every campaign is in the sample."""
    ordered = table.sort_values("shot_id").reset_index(drop=True)
    if len(ordered) <= n:
        return ordered
    return ordered.iloc[np.unique(np.linspace(0, len(ordered) - 1, n).round().astype(int))].reset_index(drop=True)


def difference(a, b, t_stop=np.inf):
    """{'rms': RMS of the difference over the larger maximum, 'abs': the same RMS in the signal's unit} of two
    series over the time both cover up to t_stop, or None (too little shared time, or both zero there)."""
    start, stop = max(a[0][0], b[0][0]), min(a[0][-1], b[0][-1], t_stop)  # from the first sample: gas is puffed before t = 0
    if stop - start < twin_check.MIN_OVERLAP_S:
        return None
    grid = np.arange(start, stop, twin_check.GRID_S)
    ra, rb = np.interp(grid, *a), np.interp(grid, *b)
    peak = max(np.abs(ra).max(), np.abs(rb).max())
    if peak <= 0:
        return None
    rms = float(np.sqrt(np.mean((ra - rb) ** 2)))
    return {"rms": rms / peak, "abs": rms}


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


def read_live(ids, open_group=open_group, workers=4, say=print):
    """{shot: (row, series)} read from the store. The first shot is read alone and shown in full; then the rest in
    parallel, one line each as it finishes."""
    ids = sorted(int(i) for i in set(ids))
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
    return results


def from_table(table, ids):
    """{shot: (row, series)} from the table that scripts/programme_fetch.py wrote, for the shots in ids. A shot that
    is not in the table has no series."""
    wanted = set(int(i) for i in ids)
    groups = {int(shot): g for shot, g in table[table["shot_id"].isin(wanted)].groupby("shot_id")}
    results = {}
    for shot in sorted(wanted):
        row, found = {"shot_id": shot}, {}
        g = groups.get(shot)
        for name in TARGETS:
            s = None
            if g is not None and name in g:
                t, v = g["time"].to_numpy(dtype=float), g[name].to_numpy(dtype=float)
                ok = np.isfinite(t) & np.isfinite(v)
                s = (t[ok], v[ok]) if ok.sum() > 2 else None
            found[name] = s
            row[f"{name}_samples"] = 0 if s is None else len(s[0])
            row[f"{name}_max"] = np.nan if s is None else float(np.abs(s[1]).max())
            row[f"{name}_varies"] = bool(s is not None and np.ptp(s[1]) > 0)
        results[shot] = (row, found)
    return results


def state(a, b):
    """How a pair stands in one signal: 'both' (both shots have it and it changes in both), 'both_constant' (the
    same constant in both, for example no reference density in either), 'constants_differ', 'one' (it changes in
    one shot only) or 'missing' (a shot lacks it)."""
    if a is None or b is None:
        return "missing"
    va, vb = np.ptp(a[1]) > 0, np.ptp(b[1]) > 0
    if va and vb:
        return "both"
    if va or vb:
        return "one"
    return "both_constant" if np.isclose(a[1][0], b[1][0]) else "constants_differ"


def tables(results, pairs, t_loss, campaign):
    """(shots, compared): one row per shot of results, and one row per pair with, for each signal, its state and,
    where it changes in both shots, the difference up to the lost shot's loss and up to EARLY_MS before it."""
    shots = pd.DataFrame([results[i][0] for i in sorted(results)])
    shots["campaign"] = shots["shot_id"].map(campaign)
    rows = []
    for pair in pairs.itertuples() if pairs is not None else []:
        row = {"shot_id": pair.shot_id, "other": pair.other, "campaign": campaign.get(pair.shot_id)}
        stop = t_loss.get(pair.shot_id, np.nan)
        stop = float(stop) if np.isfinite(stop) else np.inf
        for name in TARGETS:
            a, b = results[pair.shot_id][1][name], results[pair.other][1][name]
            row[f"{name}_state"] = state(a, b)
            both = row[f"{name}_state"] == "both"
            d = difference(a, b, stop) if both else None
            early = difference(a, b, stop - EARLY_MS / 1000) if both else None
            row[f"{name}_rms"] = np.nan if d is None else d["rms"]
            row[f"{name}_abs"] = np.nan if d is None else d["abs"]
            row[f"{name}_rms_early"] = np.nan if early is None else early["rms"]
            row[f"{name}_abs_early"] = np.nan if early is None else early["abs"]
        rows.append(row)
    columns = ["shot_id", "other", "campaign"] + [f"{name}_{kind}" for name in TARGETS
                                                  for kind in ("state", "rms", "abs", "rms_early", "abs_early")]
    return shots, pd.DataFrame(rows, columns=columns)


def look(ids, pairs, t_loss, campaign, open_group=open_group, workers=4, say=print):
    """(shots, compared) read from the store; pairs may be None."""
    return tables(read_live(ids, open_group, workers, say), pairs, t_loss, campaign)


def text(shots, compared):
    """The look in words."""
    speed = f", {shots['seconds'].median():.1f} s per shot (median)" if "seconds" in shots else ""
    lines = [f"What else was programmed? {len(compared)} twin pairs, {len(shots)} shots{speed}.", ""]
    for group in GROUPS:
        if f"{group}_vars" in shots:
            counts = shots[f"{group}_vars"].value_counts()
            lines.append(f"{group}: the variable lists seen (shots with each):")
            lines += [f"- {n} shots: {names if names else '(no variables)'}" for names, n in counts.items()]
            lines.append("")
    for name in TARGETS:
        there, varies = shots[f"{name}_samples"] > 0, shots[f"{name}_varies"]
        lines.append(f"{NAMES[name]}: present in {there.mean():.0%} of the shots, not constant in {varies.mean():.0%}.")
        by = shots.groupby("campaign", dropna=False)[f"{name}_varies"].agg(["mean", "size"])
        lines.append("  not constant, by campaign: " + "; ".join(f"{c} {m:.0%} of {int(n)}" for c, (m, n) in by.iterrows()))
        if len(compared):
            counts = compared[f"{name}_state"].value_counts()
            lines.append("  pairs: " + "; ".join(f"{state_name} {int(counts.get(state_name, 0)):,}" for state_name in STATES))
            for label, kind in (("up to the loss", ""), (f"up to {EARLY_MS} ms before the loss", "_early")):
                rms = compared[f"{name}_rms{kind}"].dropna()
                if len(rms):
                    lines.append(f"  where it changes in both, {label} ({len(rms):,} pairs): relative difference median "
                                 f"{rms.median():.3f}; under {SAME_RMS:.0%} in {(rms < SAME_RMS).mean():.0%}, under 1% in "
                                 f"{(rms < 0.01).mean():.0%}; in its own unit median {compared[f'{name}_abs{kind}'].median():.3g}.")
        lines.append("")
    return "\n".join(lines).rstrip()


def trigger(limit=0):
    """Start the GitHub job that reads the store (every shot, or a trial on `limit` shots). Returns the time it was
    requested (seconds, UTC)."""
    response = requests.post(relay._api(f"/actions/workflows/{WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                             json={"ref": config.RELAY_BRANCH, "inputs": {"limit": str(int(limit))}})
    relay.check(response)
    return time.time()


def list_runs():
    """The latest runs of the GitHub job, as GitHub lists them."""
    response = requests.get(relay._api(f"/actions/workflows/{WORKFLOW}/runs"), headers=relay._headers(),
                            params={"per_page": 20}, timeout=30)
    relay.check(response)
    return response.json().get("workflow_runs", [])


def find_run(requested_at):
    """The run started by trigger() at requested_at, or None if GitHub does not list it yet."""
    return relay.pick_run(list_runs(), requested_at)


def newest_run():
    """The newest run of the GitHub job, whatever started it and whether or not it has finished; None if there is none."""
    runs = list_runs()
    return max(runs, key=lambda r: r["created_at"]) if runs else None


def wait(run, say=print, poll_s=30, hours=6):
    """Wait until the run has finished; an error if it did not finish well or not in time."""
    last, deadline = None, time.time() + hours * 3600
    while time.time() < deadline:
        status, conclusion = relay.run_status(run["id"])
        if status != last:
            say(f"  {time.strftime('%H:%M:%S')} the GitHub job is {status.replace('_', ' ')}")
            last = status
        if status == "completed":
            if conclusion != "success":
                raise RuntimeError(f"the GitHub job ended with '{conclusion}': {run.get('html_url', run['id'])}")
            return
        time.sleep(poll_s)
    raise RuntimeError(f"the GitHub job did not finish within {hours} hours")


def collect(run_id):
    """Download the run's table into DATA_DIR/programme (the table and its manifest, nothing else) and return the
    manifest."""
    response = requests.get(relay._api(f"/actions/runs/{run_id}/artifacts"), headers=relay._headers(), timeout=30)
    relay.check(response)
    artifact = next((a for a in response.json().get("artifacts", []) if a["name"] == ARTIFACT and not a.get("expired")), None)
    if artifact is None:
        raise RuntimeError(f"run {run_id} has no '{ARTIFACT}' artifact (GitHub keeps it for three days)")
    response = requests.get(artifact["archive_download_url"], headers=relay._headers(), timeout=600)
    relay.check(response)
    os.makedirs(data_folder(), exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = {os.path.basename(name): name for name in archive.namelist()}
        if TABLE not in names or "manifest.json" not in names:
            raise RuntimeError(f"the artifact of run {run_id} lacks {TABLE} or manifest.json")
        for base in (TABLE, "manifest.json"):
            with open(os.path.join(data_folder(), base), "wb") as f:
                f.write(archive.read(names[base]))
    with open(os.path.join(data_folder(), "manifest.json")) as f:
        return json.load(f)


def through_github(limit=0, say=print, poll_s=30, hours=6):
    """Start the GitHub job, wait for it and collect its table. Returns the manifest."""
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    requested = trigger(limit)
    say("asked GitHub to start the job; waiting for it to appear ...")
    run = None
    while run is None and time.time() < requested + 300:
        time.sleep(min(poll_s, 10))
        run = find_run(requested)
    if run is None:
        raise RuntimeError("GitHub did not list the job within five minutes; look at the Actions page of the repository")
    say(f"job started: {run.get('html_url', run['id'])}")
    wait(run, say, poll_s, hours)
    return collect(run["id"])


def collect_newest(say=print, poll_s=30, hours=6):
    """Collect the table of the newest GitHub job, waiting for it first if it is still running."""
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    run = newest_run()
    if run is None:
        raise RuntimeError("GitHub lists no job of fetch_programme yet; start one with --relay")
    say(f"newest job: started {run['created_at']}, {run.get('html_url', run['id'])}")
    wait(run, say, poll_s, hours)
    return collect(run["id"])


def coverage(table, pairs):
    """The share of the twin pairs' shots that the collected table holds."""
    wanted = set(pairs["shot_id"]) | set(pairs["other"])
    return len(wanted & set(table["shot_id"].unique())) / max(len(wanted), 1)


def manifest_line(manifest):
    return (f"collected: {manifest.get('shots_with_data', '?'):,} shots with data of {manifest.get('shots_requested', '?'):,} "
            f"asked, {manifest.get('failed', '?')} failed, made {manifest.get('created', '?')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relay", action="store_true", help="start the GitHub job, wait and collect its table")
    parser.add_argument("--collect", action="store_true", help="collect the newest finished GitHub job's table")
    parser.add_argument("--limit", type=int, default=0, help="with --relay: a trial on this many shots (0 = every shot)")
    parser.add_argument("--shots", type=int, default=0, help="read this many shots, spread over the table, from the store")
    parser.add_argument("--workers", type=int, default=4, help="with --shots: shots read at the same time")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    try:
        if args.relay:
            say(manifest_line(through_github(args.limit, say)))
        elif args.collect:
            say(manifest_line(collect_newest(say)))
    except RuntimeError as error:
        say(f"STOPPED: {error}")
        sys.exit(1)
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
    if args.shots:
        try:
            import zarr
            zarr.config.set({"async.timeout": 3 * REQUEST_S})  # a second limit, in case a request never returns
        except Exception:
            pass
        ids = sample(data.load_shot_table(), args.shots)["shot_id"]
        try:
            shots, compared = look(ids, None, {}, campaign, workers=args.workers, say=say)
        except RuntimeError as error:
            say(f"STOPPED: {error}")
            os._exit(1)  # do not wait for requests still in flight
    else:
        path = os.path.join(data_folder(), TABLE)
        if not os.path.exists(path):
            say(f"no collected table at {path}. Run: python scripts/programme_signals.py --relay")
            sys.exit(1)
        pairs = pd.read_csv(v3_runs.TWINS)
        labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
        t_loss = labels.set_index("shot_id")["t_quench"].to_dict()
        table = pd.read_parquet(path)
        if coverage(table, pairs) < 0.9:
            say(f"STOPPED: the collected table holds {coverage(table, pairs):.0%} of the twin pairs' shots; it is from a "
                "trial. Collect the table of a full job: python scripts/programme_signals.py --collect")
            sys.exit(1)
        results = from_table(table, set(pairs["shot_id"]) | set(pairs["other"]))
        shots, compared = tables(results, pairs, t_loss, campaign)
    os.makedirs(folder(), exist_ok=True)
    shots.to_csv(os.path.join(folder(), "shots.csv"), index=False)
    compared.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    out = text(shots, compared)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
