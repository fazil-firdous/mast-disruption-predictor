"""Self-audit (docs/preregistration_mast_timing_audit.md, fixed before any time stamp was read): do MAST's observation
times tell a lost shot from its twin? EFIT slice times (read on GitHub, scripts/efit_times_fetch.py) and the summary
grid of the cached shot tables, at the same programme time in both shots of each twin pair. No model.

    python scripts/mast_timing_audit.py --relay    # read the EFIT slice times through GitHub, then the audit
    python scripts/mast_timing_audit.py            # the audit, from the collected times

Writes DATA_DIR/efit_times/ and RESULTS_DIR/mast_timing_audit/. Outside every stamp.
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

from scripts import efit_times_fetch, how_early, programme_signals, sampling_audit, sxr_fetch, twin_check, v3_explore, v3_runs
from src import config, data, extras, notify, relay

LEADS_MS = (10, 30, 50, 100)
PRIMARY_LEAD = 30
WINDOW_S, GAP_SPAN_S, CUT_S = config.WINDOW_MS / 1000, 0.100, 0.040
DELAY_S = config.EFIT_DELAY_MS / 1000
BAND = (0.47, 0.53)
N_BOOT, SEED = 2000, 42
MEASURES = {"E1": ("EFIT slices available in the 20 ms window", +1), "E2": ("age of the latest available EFIT slice", -1),
            "E3": ("median EFIT gap over 100 ms", -1), "G1": ("summary-grid samples in the 20 ms window", +1),
            "G2": ("median summary-grid gap over 100 ms", -1)}
WORKFLOW, ARTIFACT, PARTS = "fetch_efit_times.yml", "mast-efit-times", 4


def folder():
    return os.path.join(config.RESULTS_DIR, "mast_timing_audit")


def times_path():
    return os.path.join(efit_times_fetch.folder(), efit_times_fetch.TABLE)


# ------------------------------------------------------------------ measures

def measures(avail, grid, t):
    """E1-E3 from EFIT availability times (slice time + delay) and G1-G2 from the summary grid, at moment t."""
    out = {}
    a = avail[avail <= t + 1e-9] if avail is not None else np.array([])
    out["E1"] = float(np.sum(a > t - WINDOW_S + 1e-9)) if avail is not None else np.nan
    out["E2"] = float(t - a[-1]) if len(a) else np.nan
    recent = a[a > t - GAP_SPAN_S + 1e-9]
    out["E3"] = float(np.median(np.diff(recent))) if len(recent) >= 2 else np.nan
    g = grid[grid <= t + 1e-9]
    out["G1"] = float(np.sum(g > t - WINDOW_S + 1e-9))
    recent = g[g > t - GAP_SPAN_S + 1e-9]
    out["G2"] = float(np.median(np.diff(recent))) if len(recent) >= 2 else np.nan
    return out


def denser(a, b, sign):
    """1 if a is denser than b (larger for sign +1, smaller for sign -1), 0.5 if equal, 0 if not, NaN if unknown."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    if a == b:
        return 0.5
    return 1.0 if (a - b) * sign > 0 else 0.0


def pair_rows(pairs, avail_of, grid_of, leads=LEADS_MS):
    """One row per (pair, lead) at which both shots still run: wins and paired differences of every measure.
    'Still run' = both shots' summary grids reach t, so a twin that ended is left out (the fixed-lead rule)."""
    rows = []
    for p in pairs.itertuples():
        if not np.isfinite(p.t_loss):
            continue
        gl, gk = grid_of(p.shot_id), grid_of(p.other)
        if gl is None or gk is None or not len(gl) or not len(gk):
            continue
        al, ak = avail_of(p.shot_id), avail_of(p.other)
        for lead in leads:
            t = p.t_loss - lead / 1000
            if np.isfinite(p.t_start) and t < p.t_start - 1e-9:
                continue
            if gl[-1] < t - 1e-9 or gk[-1] < t - 1e-9:
                continue
            ml, mk = measures(al, gl, t), measures(ak, gk, t)
            row = {"shot_id": p.shot_id, "other": p.other, "lead_ms": lead}
            for name, (_, sign) in MEASURES.items():
                efit = name.startswith("E")
                if efit and (al is None or ak is None or not np.isfinite(ml["E2"]) or not np.isfinite(mk["E2"])):
                    row[f"win_{name}"], row[f"diff_{name}"] = np.nan, np.nan
                    continue
                row[f"win_{name}"] = denser(ml[name], mk[name], sign)
                row[f"diff_{name}"] = ml[name] - mk[name] if np.isfinite(ml[name]) and np.isfinite(mk[name]) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def clustered(values, groups, n_boot=N_BOOT, seed=SEED):
    """(mean, low, high) of values with groups of linked pairs resampled together; NaN values left out."""
    ok = np.isfinite(values)
    v, g = values[ok], groups[ok]
    if not len(v):
        return np.nan, None, None
    uniq, inverse = np.unique(g, return_inverse=True)
    members = [np.where(inverse == k)[0] for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    draws = [v[np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])].mean()
             for _ in range(n_boot)]
    return float(v.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def reading(r):
    if r[1] is None:
        return "no range"
    inside = BAND[0] <= r[0] <= BAND[1]
    if inside and r[1] <= 0.5 <= r[2]:
        return "no timing difference at the moment compared"
    if r[1] > BAND[1] or r[2] < BAND[0]:
        return "TIMING DIFFERS between lost shots and twins: the model comparison protocol comes next"
    return "reported as found; no reading fixed"


# ------------------------------------------------------------------ the relay

def trigger(ids):
    r = requests.post(relay._api(f"/actions/workflows/{WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                      json={"ref": config.RELAY_BRANCH, "inputs": {"ids": sxr_fetch.encode_ids(ids)}})
    relay.check(r)
    return time.time()


def runs():
    r = requests.get(relay._api(f"/actions/workflows/{WORKFLOW}/runs"), headers=relay._headers(), params={"per_page": 20}, timeout=30)
    relay.check(r)
    return r.json().get("workflow_runs", [])


def collect(run_id):
    r = requests.get(relay._api(f"/actions/runs/{run_id}/artifacts"), headers=relay._headers(), timeout=30)
    relay.check(r)
    parts = [a for a in r.json().get("artifacts", []) if a["name"].startswith(ARTIFACT) and not a.get("expired")]
    tables, manifests = [], []
    for a in parts:
        z = requests.get(a["archive_download_url"], headers=relay._headers(), timeout=600)
        relay.check(z)
        with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
            names = {os.path.basename(n): n for n in archive.namelist()}
            if efit_times_fetch.TABLE in names:
                tables.append(pd.read_parquet(io.BytesIO(archive.read(names[efit_times_fetch.TABLE]))))
                manifests.append(json.loads(archive.read(names["manifest.json"])))
    if not tables:
        raise RuntimeError(f"run {run_id}: no part has a table")
    m = {k: sum(x.get(k, 0) for x in manifests) for k in ("shots_requested", "shots_with_data", "absent", "failed", "odd")}
    m["parts_collected"] = len(manifests)
    os.makedirs(efit_times_fetch.folder(), exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(times_path(), index=False)
    with open(os.path.join(efit_times_fetch.folder(), "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def through_github(say=print):
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id"])
    ids = sorted(int(i) for i in windows["shot_id"].unique())
    requested = trigger(ids)
    say(f"asked GitHub to read the EFIT slice times of {len(ids):,} shots ({PARTS} parts) ...")
    run = None
    while run is None and time.time() < requested + 300:
        time.sleep(10)
        run = relay.pick_run(runs(), requested)
    if run is None:
        raise RuntimeError("GitHub did not list the job within five minutes")
    say(f"job started: {run.get('html_url', run['id'])}")
    try:
        programme_signals.wait(run, say)
    except RuntimeError as error:
        say(f"  {error}; collecting the parts that finished")
    m = collect(run["id"])
    say(f"collected: {m}")
    if m["parts_collected"] < PARTS:
        raise RuntimeError(f"only {m['parts_collected']} of {PARTS} parts came back; run --relay again")


# ------------------------------------------------------------------ the audit

def text(rows, unpaired, gaps, missing):
    lines = ["MAST timing self-audit (docs/preregistration_mast_timing_audit.md): EFIT slice times (available at slice time "
             f"+ {config.EFIT_DELAY_MS} ms) and the summary grid, lost shot against its twin at the same programme time. "
             "T = share of pairs where the lost shot is observed more densely (0.5 = no difference); range: groups of "
             "pairs sharing a shot resampled together.", ""]
    if len(rows):
        groups = v3_explore.shot_groups(rows["shot_id"].to_numpy(), rows["other"].to_numpy())
        for key in ("E1", "G1"):
            part = rows["lead_ms"] == PRIMARY_LEAD
            r = clustered(rows.loc[part, f"win_{key}"].to_numpy(dtype=float), groups[part.to_numpy()])
            n = int(np.isfinite(rows.loc[part, f"win_{key}"].to_numpy(dtype=float)).sum())
            label = "PRIMARY" if key == "E1" else "KEY"
            lines.append(f"{label} {key} ({MEASURES[key][0]}) at {PRIMARY_LEAD} ms, {n:,} pairs: T = "
                         f"{sampling_audit.fmt(r)}: {reading(r)}")
        lines += ["", "All measures (T | mean paired difference, lost minus twin):"]
        for lead in LEADS_MS:
            part = (rows["lead_ms"] == lead).to_numpy()
            cells = []
            for key in MEASURES:
                w = clustered(rows.loc[part, f"win_{key}"].to_numpy(dtype=float), groups[part])
                d = clustered(rows.loc[part, f"diff_{key}"].to_numpy(dtype=float), groups[part])
                scale = 1e3 if key in ("E2", "E3", "G2") else 1
                unit = " ms" if scale == 1e3 else ""
                cells.append(f"{key} {sampling_audit.fmt(w)} | {d[0] * scale:+.2f}{unit}" if np.isfinite(d[0]) else f"{key} -")
            lines.append(f"- {lead} ms: " + "; ".join(cells))
    if unpaired is not None and len(unpaired):
        y = unpaired["lost"].to_numpy()
        lines += ["", f"Without pairs (window ending 40 ms before each shot's last grid sample; {int(y.sum()):,} lost, "
                      f"{int((~y).sum()):,} kept shots), AUC lost against kept:"]
        for key in ("E1", "G1"):
            r = sampling_audit.auc_range(unpaired[key].to_numpy(dtype=float), y)
            lines.append(f"- {key}: {sampling_audit.fmt(r)}")
    if gaps:
        lines += ["", "EFIT gaps over the flat-top (median of shot medians | median of shot 90th percentiles), lost | kept: "
                      f"{gaps['lost'][0] * 1e3:.2f} | {gaps['kept'][0] * 1e3:.2f} ms; {gaps['lost'][1] * 1e3:.2f} | "
                      f"{gaps['kept'][1] * 1e3:.2f} ms"]
    if missing:
        lines.append(f"Shots with no EFIT slices: lost {missing['lost']:.1%}, kept {missing['kept']:.1%}")
    lines.append("Not claimed: that the twin test's predictor used any timing difference.")
    return "\n".join(lines)


def audit(say=print, grid_of=None, extras_of=None):
    if not os.path.exists(times_path()):
        raise RuntimeError("no EFIT slice times collected: run python scripts/mast_timing_audit.py --relay")
    et = pd.read_parquet(times_path())
    avail = {int(s): np.sort(g["time"].to_numpy(dtype=float)) + DELAY_S for s, g in et.groupby("shot_id")}
    avail_of = lambda s: avail.get(int(s))  # noqa: E731
    cache = {}

    def default_grid(s):
        if s not in cache:
            path = data.cache_path(int(s))  # cached shots only: the server cannot download
            cache[s] = (np.sort(pd.read_parquet(path, columns=["time"])["time"].to_numpy(dtype=float))
                        if os.path.exists(path) else None)
        return cache[s]

    grid_of = grid_of or default_grid
    extras_of = extras_of or (lambda s: extras.load_extras(int(s), allow_download=False))
    main = how_early.pair_frames(extras_of)[0]
    say(f"{len(main):,} twin pairs; measuring ...")
    rows = pair_rows(main, avail_of, grid_of)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    group = twin_check.groups(labels)
    shots = [int(s) for s in labels["shot_id"] if group.get(s) in twin_check.LOST | twin_check.KEPT]
    un, gaps_l, gaps_k, miss = [], [], [], {"lost": [], "kept": []}
    for i, s in enumerate(shots, 1):
        g = grid_of(s)
        lost = group.get(s) in twin_check.LOST
        miss["lost" if lost else "kept"].append(avail_of(s) is None)
        if g is None or not len(g):
            continue
        m = measures(avail_of(s), g, g[-1] - CUT_S)
        un.append({"shot_id": s, "lost": lost, "E1": m["E1"], "G1": m["G1"]})
        a = avail_of(s)
        if a is not None:
            start = v3_runs.flat_top_start(extras_of(s))
            sl = a - DELAY_S
            sl = sl[(sl >= (start if start is not None else -np.inf)) & (sl <= g[-1] - CUT_S)]
            if len(sl) >= 3:
                d = np.diff(sl)
                (gaps_l if lost else gaps_k).append((np.median(d), np.percentile(d, 90)))
        if i % 2000 == 0:
            say(f"  [{i}/{len(shots)}] shots")
    gaps = ({"lost": tuple(np.median(np.array(gaps_l), axis=0)), "kept": tuple(np.median(np.array(gaps_k), axis=0))}
            if gaps_l and gaps_k else None)
    missing = {k: float(np.mean(v)) if v else np.nan for k, v in miss.items()}
    out = text(rows, pd.DataFrame(un), gaps, missing)
    os.makedirs(folder(), exist_ok=True)
    rows.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
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
    notify.send("MAST timing self-audit: done\n\n" + out)


if __name__ == "__main__":
    main()
