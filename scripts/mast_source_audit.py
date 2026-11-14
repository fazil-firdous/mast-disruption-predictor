"""MAST at source level (docs/preregistration_mast_source_times.md, fixed before any level-1 time of a twin pair was
read): do the native converged EFM reconstruction times differ between lost shots and their twins at the same programme
time? Reads level-1 `efm` attempted/converged times through GitHub. No model.

    python scripts/mast_source_audit.py --relay    # read the level-1 times through GitHub, then the audit
    python scripts/mast_source_audit.py            # the audit, from the collected times

Writes DATA_DIR/efit_times/efm_times.parquet and RESULTS_DIR/mast_timing_audit/source.txt. Outside every stamp.
"""
import argparse
import io
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import efit_times_fetch, how_early, mast_timing_audit as mt, programme_signals, sampling_audit, sxr_fetch, \
    twin_check, v3_explore
from src import config, extras, notify, relay

MEASURES = {"SE1": ("converged reconstructions available in the 20 ms window", +1),
            "SE2": ("age of the latest available converged reconstruction", -1),
            "SE3": ("median gap between converged reconstructions over 100 ms", -1),
            "SA1": ("attempted, not converged, in the 20 ms window", +1)}


def table_path():
    return os.path.join(efit_times_fetch.folder(), efit_times_fetch.LEVEL1_TABLE)


def measures(conv_avail, fail_avail, t):
    m = mt.measures(conv_avail, np.array([t]), t)
    out = {"SE1": m["E1"], "SE2": m["E2"], "SE3": m["E3"]}
    out["SA1"] = float(np.sum((fail_avail > t - mt.WINDOW_S + 1e-9) & (fail_avail <= t + 1e-9))) if fail_avail is not None else np.nan
    return out


def pair_rows(pairs, times_of, grid_end_of, leads=mt.LEADS_MS):
    rows = []
    for p in pairs.itertuples():
        if not np.isfinite(p.t_loss):
            continue
        sl, sk = times_of(p.shot_id), times_of(p.other)
        el, ek = grid_end_of(p.shot_id), grid_end_of(p.other)
        if sl is None or sk is None or el is None or ek is None:
            continue
        for lead in leads:
            t = p.t_loss - lead / 1000
            if (np.isfinite(p.t_start) and t < p.t_start - 1e-9) or el < t - 1e-9 or ek < t - 1e-9:
                continue
            ml, mk = measures(*sl, t), measures(*sk, t)
            if not (np.isfinite(ml["SE2"]) and np.isfinite(mk["SE2"])):
                continue  # a shot with no converged reconstruction available yet at t
            row = {"shot_id": p.shot_id, "other": p.other, "lead_ms": lead}
            for name, (_, sign) in MEASURES.items():
                row[f"win_{name}"] = mt.denser(ml[name], mk[name], sign)
                row[f"diff_{name}"] = ml[name] - mk[name] if np.isfinite(ml[name]) and np.isfinite(mk[name]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def reading(r):
    if r[1] is None:
        return "no range"
    if mt.BAND[0] <= r[0] <= mt.BAND[1] and r[1] <= 0.5 <= r[2]:
        return "no difference in native converged-reconstruction density at the moment compared"
    if r[1] > mt.BAND[1] or r[2] < mt.BAND[0]:
        return "THE NATIVE SCHEDULE DIFFERS between lost shots and twins: the further protocol comes next"
    return "reported as found; no reading fixed"


def text(rows, unpaired, coverage):
    lines = ["MAST source-level audit (docs/preregistration_mast_source_times.md): level-1 efm converged reconstruction "
             f"times (available at time + {config.EFIT_DELAY_MS} ms, the pipeline's assumption), lost shot against twin at "
             "the same programme time. T = share of pairs where the lost shot is observed more densely; range: groups of "
             "pairs sharing a shot resampled together.", ""]
    if len(rows):
        groups = v3_explore.shot_groups(rows["shot_id"].to_numpy(), rows["other"].to_numpy())
        for key, label in (("SE1", "PRIMARY"), ("SE2", "KEY")):
            part = (rows["lead_ms"] == mt.PRIMARY_LEAD).to_numpy()
            r = mt.clustered(rows.loc[part, f"win_{key}"].to_numpy(dtype=float), groups[part])
            n = int(np.isfinite(rows.loc[part, f"win_{key}"].to_numpy(dtype=float)).sum())
            lines.append(f"{label} {key} ({MEASURES[key][0]}) at {mt.PRIMARY_LEAD} ms, {n:,} pairs: T = "
                         f"{sampling_audit.fmt(r)}: {reading(r)}")
        lines += ["", "All measures (T | mean paired difference, lost minus twin):"]
        for lead in mt.LEADS_MS:
            part = (rows["lead_ms"] == lead).to_numpy()
            cells = []
            for key in MEASURES:
                w = mt.clustered(rows.loc[part, f"win_{key}"].to_numpy(dtype=float), groups[part])
                d = np.nanmean(rows.loc[part, f"diff_{key}"].to_numpy(dtype=float)) if part.any() else np.nan
                scale, unit = (1e3, " ms") if key in ("SE2", "SE3") else (1, "")
                cells.append(f"{key} {sampling_audit.fmt(w)} | {d * scale:+.2f}{unit}")
            lines.append(f"- {lead} ms ({int(part.sum()):,} pairs): " + "; ".join(cells))
    if unpaired is not None and len(unpaired):
        y = unpaired["lost"].to_numpy()
        r = sampling_audit.auc_range(unpaired["SE1"].to_numpy(dtype=float), y)
        lines += ["", f"Without pairs ({int(y.sum()):,} lost, {int((~y).sum()):,} kept; window ending 40 ms before the last grid "
                      f"sample): AUC of SE1, lost against kept, {sampling_audit.fmt(r)}"]
    lines.append(f"Level-1 efm missing for {coverage['missing']:.1%} of shots; shots with any attempted but unconverged "
                 f"reconstruction: {coverage['unconverged']:.1%}.")
    lines.append("Not claimed: online availability, real-time latency, or that the twin test's predictor used any of this.")
    return "\n".join(lines)


def trigger(ids):
    r = requests.post(relay._api(f"/actions/workflows/{mt.WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                      json={"ref": config.RELAY_BRANCH, "inputs": {"ids": sxr_fetch.encode_ids(ids), "level1": "1"}})
    relay.check(r)
    return time.time()


def collect(run_id):
    r = requests.get(relay._api(f"/actions/runs/{run_id}/artifacts"), headers=relay._headers(), timeout=30)
    relay.check(r)
    tables, manifests = [], []
    for a in r.json().get("artifacts", []):
        if not a["name"].startswith(mt.ARTIFACT) or a.get("expired"):
            continue
        z = requests.get(a["archive_download_url"], headers=relay._headers(), timeout=600)
        relay.check(z)
        with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
            names = {os.path.basename(n): n for n in archive.namelist()}
            if efit_times_fetch.LEVEL1_TABLE in names:
                tables.append(pd.read_parquet(io.BytesIO(archive.read(names[efit_times_fetch.LEVEL1_TABLE]))))
                manifests.append(json.loads(archive.read(names["manifest.json"])))
    if not tables:
        raise RuntimeError(f"run {run_id}: no part has a level-1 table")
    m = {k: sum(x.get(k, 0) for x in manifests) for k in ("shots_requested", "shots_with_data", "absent", "failed", "odd",
                                                           "shots_with_unconverged")}
    m["parts_collected"] = len(manifests)
    os.makedirs(efit_times_fetch.folder(), exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(table_path(), index=False)
    return m


def through_github(say=print):
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id"])
    ids = sorted(int(i) for i in windows["shot_id"].unique())
    requested = trigger(ids)
    say(f"asked GitHub to read the level-1 efm times of {len(ids):,} shots ...")
    run = None
    while run is None and time.time() < requested + 300:
        time.sleep(10)
        run = relay.pick_run(mt.runs(), requested)
    if run is None:
        raise RuntimeError("GitHub did not list the job within five minutes")
    say(f"job started: {run.get('html_url', run['id'])}")
    try:
        programme_signals.wait(run, say)
    except RuntimeError as error:
        say(f"  {error}; collecting the parts that finished")
    m = collect(run["id"])
    say(f"collected: {m}")
    if m["parts_collected"] < mt.PARTS:
        raise RuntimeError(f"only {m['parts_collected']} of {mt.PARTS} parts came back; run --relay again")


def audit(say=print):
    if not os.path.exists(table_path()):
        raise RuntimeError("no level-1 times collected: run python scripts/mast_source_audit.py --relay")
    et = pd.read_parquet(table_path())
    times = {}
    for s, g in et.groupby("shot_id"):
        t = g["time"].to_numpy(dtype=float)
        c = g["converged"].to_numpy(dtype=bool)
        times[int(s)] = (np.sort(t[c]) + mt.DELAY_S, np.sort(t[~c]) + mt.DELAY_S)
    ends = {}

    def end_of(s):
        s = int(s)
        if s not in ends:
            path = mt.data.cache_path(s)
            ends[s] = float(pd.read_parquet(path, columns=["time"])["time"].max()) if os.path.exists(path) else None
        return ends[s]

    load = lambda s: extras.load_extras(int(s), allow_download=False)  # noqa: E731
    main_pairs = how_early.pair_frames(load)[0]
    say(f"{len(main_pairs):,} twin pairs; measuring ...")
    rows = pair_rows(main_pairs, lambda s: times.get(int(s)), end_of)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    group = twin_check.groups(labels)
    shots = [int(s) for s in labels["shot_id"] if group.get(s) in twin_check.LOST | twin_check.KEPT]
    un, missing = [], []
    for s in shots:
        missing.append(s not in times)
        e = end_of(s)
        if s in times and e is not None:
            un.append({"lost": group.get(s) in twin_check.LOST, "SE1": measures(*times[s], e - mt.CUT_S)["SE1"]})
    unconv = np.mean([len(times[s][1]) > 0 for s in times]) if times else np.nan
    out = text(rows, pd.DataFrame(un), {"missing": float(np.mean(missing)), "unconverged": float(unconv)})
    os.makedirs(mt.folder(), exist_ok=True)
    rows.to_csv(os.path.join(mt.folder(), "source_pairs.csv"), index=False)
    with open(os.path.join(mt.folder(), "source.txt"), "w") as f:
        f.write(out + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relay", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    try:
        if args.relay:
            through_github(say)
        out = audit(say)
    except RuntimeError as error:
        say(f"STOPPED: {error}")
        sys.exit(1)
    say(out)
    notify.send("MAST source-level audit: done\n\n" + out)


if __name__ == "__main__":
    main()
