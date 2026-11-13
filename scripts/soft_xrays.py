"""Faster signals, step 1: soft X-rays added to the full model (docs/preregistration_soft_xrays.md, fixed before any
soft X-ray feature or model of this study).

    python scripts/soft_xrays.py --all        # every step not done yet: read the cameras through GitHub (about half
                                              # an hour), the features, the 21 runs, the summary
    python scripts/soft_xrays.py --relay      # only: start the GitHub job, wait, collect the 1 ms signals
    python scripts/soft_xrays.py --collect    # only: collect the newest finished GitHub job
    python scripts/soft_xrays.py --features   # only: the 46 soft X-ray features for every window
    python scripts/soft_xrays.py --train      # only: the runs not done yet, then the summary
    python scripts/soft_xrays.py --summary    # only: the comparison in words, from the saved runs
    python scripts/soft_xrays.py --control    # exploratory E3: the availability-only control's 21 runs, then the summary

Writes DATA_DIR/sxr_series/, PROCESSED_DIR/soft_xrays.parquet and RESULTS_DIR/soft_xrays/. Nothing under v3/ or
v3_explore/ is touched; outside every stamp.
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

from scripts import fast_magnetics, how_early, programme_signals, sxr_fetch, sxr_series, v3_explore, v3_runs
from src import config, dataset, experiment, features, notify, relay, server

CAMS = ("lo", "up")  # camera codes 0 and 1 of scripts/sxr_series.py
STATS = ("mean", "std", "slope", "delta")
REL = ("level", "fluct")
FEATURES = ([f"sxr_{c}_{s}_{st}" for c in CAMS for s in sxr_series.SIGNALS for st in STATS]
            + [f"sxr_{c}_{s}_rel" for c in CAMS for s in REL] + [f"sxr_{c}_drop" for c in CAMS])
RECIPE = "sxr"
CONTROL = "sxr_avail"  # exploratory E3: whether each camera has a level, without any soft X-ray value
PRESENT = [f"sxr_{c}_present" for c in CAMS]
NAMES = {"baseline": "A, full", RECIPE: "S, full + soft X-rays"}
SEEDS = range(1, 21)
PRIMARY_LEAD = 50
WORTH = 0.03
WINDOW_BINS = config.WINDOW_MS  # 1 ms bins
WORKFLOW, ARTIFACT, PARTS = "fetch_sxr_series.yml", "mast-sxr-series", 8


def series_path():
    return os.path.join(sxr_series.folder(), sxr_series.TABLE)


def table_path():
    return os.path.join(config.PROCESSED_DIR, "soft_xrays.parquet")


def folder():
    return os.path.join(config.RESULTS_DIR, "soft_xrays")


def run_dir(seed=None, recipe=RECIPE):
    return os.path.join(folder(), "temporal", recipe) if seed is None else os.path.join(folder(), recipe, f"seed_{seed}")


# ------------------------------------------------------------------ features

def drop(level):
    """The largest fall of level from one bin to the next, as a share of the earlier bin's level; 0 if it never
    falls; NaN where no two neighbouring bins have values with the earlier one positive."""
    a, b = level[:-1], level[1:]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0)
    if not ok.any():
        return np.nan
    return float(max(0.0, np.max((a[ok] - b[ok]) / a[ok])))


def camera_features(code, k, values, t_end, t_start):
    """The 23 features of one camera for the window ending at t_end. k: sorted bin ends (ms); values: signal -> array."""
    cam = CAMS[code]
    k_end = int(np.floor(t_end * 1000 + 1e-6))
    lo, hi = np.searchsorted(k, k_end - WINDOW_BINS, "right"), np.searchsorted(k, k_end, "right")
    h_lo = np.searchsorted(k, int(np.ceil(t_start * 1000 - 1e-6)), "left")
    t = k[lo:hi] / 1000
    out = {}
    for s in sxr_series.SIGNALS:
        stats = features._stats(t, values[s][lo:hi])
        for st in STATS:
            out[f"sxr_{cam}_{s}_{st}"] = stats[st]
        if s in REL:
            out[f"sxr_{cam}_{s}_rel"] = features.relative_change(stats["mean"], values[s][h_lo:lo])
    out[f"sxr_{cam}_drop"] = drop(values["level"][lo:hi])
    return out


def shot_features(series, t_ends, t_start):
    """One row per window end (FEATURES), from one shot's rows of the 1 ms table; NaN where a camera is missing."""
    cams = {}
    for code, part in series.groupby("camera"):
        part = part.sort_values("k_ms")
        cams[int(code)] = (part["k_ms"].to_numpy(), {s: part[s].to_numpy(dtype=float) for s in sxr_series.SIGNALS})
    rows = []
    for t_end in t_ends:
        row = {}
        for code, (k, values) in cams.items():
            if code < len(CAMS):
                row.update(camera_features(code, k, values, t_end, t_start))
        rows.append(row)
    return pd.DataFrame(rows, columns=FEATURES, dtype=float)


def build_features(series=None, windows=None, say=print):
    """The feature table for every window of the window table (NaN for shots without soft X-rays)."""
    if series is None:
        series = pd.read_parquet(series_path())
    if windows is None:
        windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id", "t_end", "t_start"])
    by_shot = dict(tuple(series.groupby("shot_id")))
    frames, start = [], time.time()
    groups = list(windows.groupby("shot_id", sort=True))
    empty = series.iloc[:0]
    for i, (shot, part) in enumerate(groups, 1):
        part = part.sort_values("t_end")
        feats = shot_features(by_shot.get(shot, empty), part["t_end"].to_numpy(), float(part["t_start"].iloc[0]))
        feats.insert(0, "t_end", part["t_end"].to_numpy())
        feats.insert(0, "shot_id", np.int32(shot))
        frames.append(feats)
        if i % 1000 == 0 or i == len(groups):
            say(f"[{i}/{len(groups)}] shots, {(time.time() - start) / i:.3f} s per shot")
    table = pd.concat(frames, ignore_index=True)
    return table.astype({c: "float32" for c in FEATURES})


# ------------------------------------------------------------------ the GitHub relay

def trigger(ids):
    encoded = sxr_fetch.encode_ids(ids)
    if len(encoded) > 60000:
        raise RuntimeError(f"the shot list is too long for a GitHub input ({len(encoded):,} characters)")
    r = requests.post(relay._api(f"/actions/workflows/{WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                      json={"ref": config.RELAY_BRANCH, "inputs": {"ids": encoded}})
    relay.check(r)
    return time.time()


def runs():
    r = requests.get(relay._api(f"/actions/workflows/{WORKFLOW}/runs"), headers=relay._headers(), params={"per_page": 20}, timeout=30)
    relay.check(r)
    return r.json().get("workflow_runs", [])


def collect(run_id, say=print):
    """Download the run's parts into DATA_DIR/sxr_series as one table and one manifest; returns the manifest."""
    r = requests.get(relay._api(f"/actions/runs/{run_id}/artifacts"), headers=relay._headers(), timeout=30)
    relay.check(r)
    parts = sorted((a for a in r.json().get("artifacts", []) if a["name"].startswith(ARTIFACT) and not a.get("expired")),
                   key=lambda a: a["name"])
    if not parts:
        raise RuntimeError(f"run {run_id} has no {ARTIFACT} artifacts (GitHub keeps them for three days)")
    tables, manifests = [], []
    for a in parts:
        z = requests.get(a["archive_download_url"], headers=relay._headers(), timeout=1200)
        relay.check(z)
        with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
            names = {os.path.basename(n): n for n in archive.namelist()}
            if sxr_series.TABLE not in names or "manifest.json" not in names:
                say(f"  {a['name']}: no table (that part failed); its shots are missing")
                continue
            tables.append(pd.read_parquet(io.BytesIO(archive.read(names[sxr_series.TABLE]))))
            manifests.append(json.loads(archive.read(names["manifest.json"])))
    if not tables:
        raise RuntimeError(f"run {run_id}: no part has a table")
    keys = ("shots_requested", "shots_with_data", "absent", "failed", "odd", "not_started")
    m = {k: sum(x.get(k, 0) for x in manifests) for k in keys}
    m["parts_collected"] = len(manifests)
    os.makedirs(sxr_series.folder(), exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(series_path(), index=False)
    with open(os.path.join(sxr_series.folder(), "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def through_github(say=print):
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id"])
    ids = sorted(int(i) for i in windows["shot_id"].unique())
    requested = trigger(ids)
    say(f"asked GitHub to read the soft X-ray cameras of {len(ids):,} shots ({PARTS} parts); waiting for the job ...")
    run = None
    while run is None and time.time() < requested + 300:
        time.sleep(10)
        run = relay.pick_run(runs(), requested)
    if run is None:
        raise RuntimeError("GitHub did not list the job within five minutes")
    say(f"job started: {run.get('html_url', run['id'])}")
    try:
        programme_signals.wait(run, say)
    except RuntimeError as error:  # a failed part costs only its own shots; collect what is there
        say(f"  {error}; collecting the parts that finished")
    m = collect(run["id"], say)
    say(f"collected: {m}")
    if m["parts_collected"] < PARTS:
        raise RuntimeError(f"only {m['parts_collected']} of {PARTS} parts came back; run --relay again")
    return m


# ------------------------------------------------------------------ the runs

def load_tables():
    if not os.path.exists(table_path()):
        raise RuntimeError("no soft_xrays.parquet: run python scripts/soft_xrays.py --features first")
    windows, shots = experiment.load_tables()
    return with_present(windows.merge(pd.read_parquet(table_path()), on=["shot_id", "t_end"], how="left")), shots


def with_present(frame):
    """frame with E3's two inputs: 1 where that camera has a level mean in the window, else 0 (never empty)."""
    present = pd.DataFrame({column: frame[f"sxr_{cam}_level_mean"].notna().astype("float32")
                            for cam, column in zip(CAMS, PRESENT)}, index=frame.index)
    return pd.concat([frame.drop(columns=[c for c in PRESENT if c in frame]), present], axis=1)


def columns(recipe=RECIPE):
    return v3_explore.recipe_columns("baseline") + (FEATURES if recipe == RECIPE else PRESENT)


def run(seed=None, tables=None, recipe=RECIPE):
    """One run: a random split by its seed (the v3 split), or the chronological split (seed None)."""
    windows, shots = tables or load_tables()
    if seed is None:
        win, column = dataset.labelled_windows(windows, shots), "split_temporal"
    else:
        split = dataset.random_split(shots, seed=seed)
        win, column = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots), "split"
    return fast_magnetics.fit(win, column, recipe, run_dir(seed, recipe), {"seed": 0 if seed is None else seed},
                              columns(recipe))


def train_all(say=print, recipe=RECIPE):
    jobs = list(SEEDS) + [None]
    todo = [s for s in jobs if not server.result_saved(os.path.join(run_dir(s, recipe), "metrics.json"))]
    say(f"{len(todo)} runs to do ({recipe})")
    tables = load_tables() if todo else None
    for i, seed in enumerate(todo, 1):
        start = time.time()
        m = run(seed, tables, recipe)
        say(f"[{i}/{len(todo)}] {'chronological' if seed is None else f'seed {seed}'}: pAUC {m['test']['pauc']:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    out = summary()
    say(out)
    notify.send(f"Faster signals, step 1 (soft X-rays): the {'control ' if recipe == CONTROL else ''}runs are done\n\n"
                + out)


# ------------------------------------------------------------------ the comparison

def collect_wins(load=None):
    """(random, temporal) landmark wins for A (the saved v3 baseline) and S."""
    main, placebo, heating = how_early.pair_frames(load)
    random, temporal = [], []
    sources = [("baseline", v3_explore.run_folders()["baseline"], v3_explore.temporal_dir("baseline")),
               (RECIPE, os.path.join(folder(), RECIPE), run_dir()),
               (CONTROL, os.path.join(folder(), CONTROL), run_dir(None, CONTROL))]
    for name, base, temporal_path in sources:
        for seed, path in how_early.saved_runs(base):
            if seed in SEEDS:
                random += how_early.run_wins(path, name, seed, main, placebo)
        if server.result_saved(os.path.join(temporal_path, "metrics.json")):
            temporal += how_early.run_wins(temporal_path, name, 0, main, placebo)
    return how_early.finish(random, heating), how_early.finish(temporal, heating), main


def availability(main, table, lead=PRIMARY_LEAD):
    """(pairs with soft X-ray inputs in both shots at the landmark, share of lost shots with them, share of twins).
    'With inputs': either camera's level mean is present in the window scored at that moment."""
    ok = table[["sxr_lo_level_mean", "sxr_up_level_mean"]].notna().any(axis=1).astype(float)
    by_shot = {s: (g["t_end"].to_numpy(dtype=float), ok.loc[g.index].to_numpy())
               for s, g in table[["shot_id", "t_end"]].groupby("shot_id")}
    both, lost_ok, twin_ok, n = set(), 0, 0, 0
    for row in main.itertuples():
        if not np.isfinite(row.t_loss):
            continue
        t = row.t_loss - lead / 1000
        if np.isfinite(row.t_start) and t < row.t_start - how_early.EPS:
            continue
        a = how_early.score_at(*by_shot[row.shot_id], t) if row.shot_id in by_shot else np.nan
        b = how_early.score_at(*by_shot[row.other], t) if row.other in by_shot else np.nan
        n += 1
        lost_ok += a == 1.0
        twin_ok += b == 1.0
        if a == 1.0 and b == 1.0:
            both.add((row.shot_id, row.other))
    return both, lost_ok / max(n, 1), twin_ok / max(n, 1)


def chord_changes(main, series, lead=PRIMARY_LEAD):
    """Exploratory E2: (pairs where both shots have soft X-ray inputs and a constant chord count in every camera over
    the 20 ms window ending at the landmark, share of lost shots whose count changes there, share of twins)."""
    wanted = set(main["shot_id"]) | set(main["other"])
    by = {key: (g["k_ms"].to_numpy(), g["n_chords"].to_numpy(), g["level"].to_numpy(dtype=float))
          for key, g in series[series["shot_id"].isin(wanted)].groupby(["shot_id", "camera"])}

    def state(shot, t):
        """(has inputs, chord count changed) at t."""
        k_end = int(np.floor(t * 1000 + 1e-6))
        has, changed = False, False
        for code in range(len(CAMS)):
            if (shot, code) not in by:
                continue
            k, n, level = by[(shot, code)]
            inside = (k > k_end - WINDOW_BINS) & (k <= k_end)
            if inside.any():
                has |= bool(np.isfinite(level[inside]).any())
                changed |= len(np.unique(n[inside])) > 1
        return has, changed

    stable, lost_changed, twin_changed, n = set(), 0, 0, 0
    for row in main.itertuples():
        if not np.isfinite(row.t_loss):
            continue
        t = row.t_loss - lead / 1000
        if np.isfinite(row.t_start) and t < row.t_start - how_early.EPS:
            continue
        (ha, ca), (hb, cb) = state(row.shot_id, t), state(row.other, t)
        n += 1
        lost_changed += ca
        twin_changed += cb
        if ha and hb and not ca and not cb:
            stable.add((row.shot_id, row.other))
    return stable, lost_changed / max(n, 1), twin_changed / max(n, 1)


def recorded_share(pairs, groups):
    """Exploratory E4: the share of operator-recorded disruptions among the lost shots of these pairs."""
    lost = [groups.get(p) for p, _ in pairs]
    return sum(g == "recorded" for g in lost) / max(len(lost), 1)


def subset(frame, pairs):
    keep = pd.DataFrame(sorted(pairs), columns=["shot_id", "other"])
    return frame.merge(keep, on=["shot_id", "other"])


def cells(frame, lead):
    return fast_magnetics.difference_cells(frame, lead, RECIPE)


def text(random, temporal, available=None, flags=None, explore=None):
    """Step 1 in words, in the order of the protocol. available: (pairs, lost share, twin share) or None; explore:
    {'changes': chord_changes(...), 'groups': {lost shot: ending group}, 'all': all pairs} or None."""
    lines = ["Faster signals, step 1 (docs/preregistration_soft_xrays.md): soft X-rays added to the full model. "
             "Fixed-lead twin test, both shots at the same moment; common cohort of all twins; splits 1 to 20.", ""]
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    d = how_early.split_difference(main, PRIMARY_LEAD, RECIPE, "baseline")
    p = v3_explore.pooled_delta(main[main["lead_ms"] == PRIMARY_LEAD], RECIPE, "baseline", "win") if len(main) else None
    if d and p:
        resolved = d[1] is not None and d[1] > 0
        lines.append(f"PRIMARY: C(50 ms) of S (full + soft X-rays) minus A (full): over splits "
                     f"{how_early.rng(*d[:3], signed=True)}, {d[3]} splits; pooled {how_early.rng(*p[:3], signed=True)}, "
                     f"{p[3]:,} pairs. "
                     + ("Resolved and at least +0.03: soft X-ray summaries add early discrimination at 50 ms."
                        if resolved and d[0] >= WORTH else
                        "Resolved but under +0.03." if resolved else
                        "Not resolved: these soft X-ray summaries add no measurable discrimination at 50 ms beyond the "
                        "full model."))
    lines += ["", "Required checks at 50 ms (S minus A; the sign must agree with the primary):"]
    if available is not None:
        lines.append(f"- pairs with soft X-ray inputs in both shots at that moment ({len(available[0]):,} pairs): "
                     f"{cells(subset(main, available[0]), PRIMARY_LEAD)}")
    if flags is not None and "plan" in flags:
        plan = flags.loc[flags["plan"].fillna(False).astype(bool), ["shot_id", "other"]]
        lines.append(f"- plan twins: {cells(main.merge(plan, on=['shot_id', 'other']), PRIMARY_LEAD)}")
    lines += ["", "C per lead, over splits (A full | S full + soft X-rays):"]
    for lead in how_early.COHORT_LEADS:
        split = how_early.over_splits(main, lead) if len(main) else {}
        if "baseline" in split:
            got = [f"{NAMES[m].split(',')[0]} {how_early.rng(*split[m][:3])}" for m in NAMES if m in split]
            lines.append(f"- {lead} ms ({split['baseline'][3]:.0f} pairs per split): " + "; ".join(got))
    lines += ["", "S minus A per lead (over splits | pooled):"]
    for lead in how_early.COHORT_LEADS:
        lines.append(f"- {lead} ms: {cells(main, lead)}")
    if flags is not None and "shape_plan" in flags:
        shape = flags.loc[flags["shape_plan"].fillna(False).astype(bool), ["shot_id", "other"]]
        lines += ["", f"Field-and-shape twins at 50 ms: {cells(main.merge(shape, on=['shot_id', 'other']), PRIMARY_LEAD)}"]
    placebo = random[random["kind"] == "placebo"] if len(random) else random
    if len(placebo):
        lines += ["", "Placebo (two good repeats at their lost shot's landmark), over splits (A | S):"]
        for lead in how_early.COHORT_LEADS:
            split = how_early.over_splits(placebo, lead)
            if "baseline" in split and RECIPE in split:
                lines.append(f"- {lead} ms: {how_early.rng(*split['baseline'][:3])} | {how_early.rng(*split[RECIPE][:3])}")
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)] if len(temporal) else temporal
    lines += ["", "Chronological split (M5-M7 to M8-M9), common cohort, pooled (A | S | S minus A):"]
    for lead in how_early.COHORT_LEADS:
        pool = how_early.pooled(t_main, lead) if len(t_main) else {}
        if "baseline" in pool and RECIPE in pool:
            dd = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == lead], RECIPE, "baseline", "win")
            lines.append(f"- {lead} ms ({int(pool['baseline'][3]):,} pairs): {how_early.rng(*pool['baseline'][:3])} | "
                         f"{how_early.rng(*pool[RECIPE][:3])} | {how_early.rng(*dd[:3], signed=True) if dd else '-'}")
    u = how_early.usual_difference(os.path.join(folder(), RECIPE), v3_explore.run_folders()["baseline"], set(SEEDS))
    if u:
        lines += ["", f"Usual metrics, S minus A, paired over {u[1]} splits: pAUC {how_early.rng(*u[0]['pauc'], signed=True)}; "
                      f"caught {how_early.rng(*u[0]['catch_rate'], signed=True)}; false alarms "
                      f"{how_early.rng(*u[0]['false_alarm_rate'], signed=True)}"]
    if available is not None:
        lines += ["", f"Missingness at 50 ms: soft X-ray inputs present for {available[1]:.0%} of lost shots and "
                      f"{available[2]:.0%} of twins."]
    lines += exploratory_lines(main, available, explore)
    return "\n".join(lines)


def exploratory_lines(main, available, explore):
    """E2 to E4 of the protocol's exploratory checks (E1 is in scripts/sxr_audit.py)."""
    lines = ["", "EXPLORATORY (protocol, 'Clarifications and exploratory checks'; not part of the registered reading):"]
    if explore and explore.get("changes"):
        stable, lost_c, twin_c = explore["changes"]
        lines.append(f"- E2: chord count changes within the 50 ms window for {lost_c:.0%} of lost shots and {twin_c:.0%} "
                     f"of twins; S minus A at 50 ms on {len(stable):,} pairs with inputs in both and constant chord "
                     f"counts: {cells(subset(main, stable), PRIMARY_LEAD)}")
    d = how_early.split_difference(main, PRIMARY_LEAD, CONTROL, "baseline") if len(main) else None
    if d:
        lines.append(f"- E3: availability-only control (full + whether each camera has a level) minus A at 50 ms: "
                     f"{cells_for(main, CONTROL)}. A resolved gain here would mean availability alone carries outcome "
                     f"information.")
    else:
        lines.append("- E3: availability-only control not run yet (python scripts/soft_xrays.py --control)")
    if explore and explore.get("groups") is not None:
        g = explore["groups"]
        parts = [f"all pairs {recorded_share(explore['all'], g):.0%}"]
        if available is not None:
            parts.append(f"inputs in both {recorded_share(available[0], g):.0%}")
        if explore.get("changes"):
            parts.append(f"E2 pairs {recorded_share(explore['changes'][0], g):.0%}")
        lines.append("- E4: operator-recorded share of the lost shots: " + "; ".join(parts))
    return lines


def cells_for(frame, recipe, lead=PRIMARY_LEAD):
    return fast_magnetics.difference_cells(frame, lead, recipe)


def summary(load=None):
    random, temporal, main = collect_wins(load)
    table = pd.read_parquet(table_path(), columns=["shot_id", "t_end", "sxr_lo_level_mean", "sxr_up_level_mean"])
    twins = pd.read_csv(v3_runs.TWINS)
    explore = {"all": set(zip(twins["shot_id"], twins["other"])),
               "groups": dict(zip(twins["shot_id"], twins["group"])) if "group" in twins else None}
    if os.path.exists(series_path()):
        explore["changes"] = chord_changes(main, pd.read_parquet(series_path(), columns=["shot_id", "camera", "k_ms",
                                                                                          "n_chords", "level"]))
    out = text(random, temporal, availability(main, table), fast_magnetics.plan_flags(), explore)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("all", "relay", "collect", "features", "train", "summary", "control"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    try:
        if args.relay or (args.all and not os.path.exists(series_path())):
            through_github(say)
        if args.collect:
            done = [r for r in runs() if r.get("status") == "completed"]
            if not done:
                raise RuntimeError("no finished fetch_sxr_series job yet")
            say(f"collected: {collect(max(done, key=lambda r: r['created_at'])['id'], say)}")
        if args.features or (args.all and not os.path.exists(table_path())):
            table = build_features(say=say)
            table.to_parquet(table_path(), index=False)
            have = table[["sxr_lo_level_mean", "sxr_up_level_mean"]].notna().any(axis=1)
            say(f"wrote {table_path()}: {len(table):,} windows, {table['shot_id'].nunique():,} shots; soft X-ray inputs "
                f"in {have.mean():.0%} of windows ({table.loc[have, 'shot_id'].nunique():,} shots)")
        if args.train or args.all:
            train_all(say)
        if args.control:
            train_all(say, CONTROL)
        elif args.summary and not (args.train or args.all):
            say(summary())
    except RuntimeError as error:
        say(f"STOPPED: {error}")
        sys.exit(1)
    if not any(vars(args).values()):
        parser.error("give --all, or one of --relay, --collect, --features, --train, --summary, --control")


if __name__ == "__main__":
    main()
