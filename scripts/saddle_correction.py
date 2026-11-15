"""Saddle-coil sign correction of the locked-mode feature (docs/preregistration_saddle_correction.md).

    python scripts/saddle_correction.py --sample     # draw the frozen sample (once; a later draw must give the same)
    python scripts/saddle_correction.py --relay      # read the saddle channels on GitHub (scripts/saddle_fetch.py)
    python scripts/saddle_correction.py --check      # validation V and provenance P1, P2, P3, P5
    python scripts/saddle_correction.py --features   # P4, and under V1 the corrected lm window features
    python scripts/saddle_correction.py --train      # A_c on splits 1 to 20 and the chronological split (V1 only)
    python scripts/saddle_correction.py --summary

Reads the window table and caches, never changes them; writes DATA_DIR/saddle_correction/ and
RESULTS_DIR/saddle_correction/; never writes into v3/. Outside every stamp.
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

from scripts import fast_magnetics, how_early, no_lm, programme_signals, saddle_fetch, sxr_fetch, v3_explore
from src import config, dataset, experiment, extras, features, notify, relay, server
from src.data import load_shot

RECIPE = "saddle_corr"
SEEDS = range(1, 21)
PRIMARY_LEAD = 10
WORKFLOW, ARTIFACT, PARTS = "fetch_saddle.yml", "mast-saddle", 8
N_SPREAD, N_EDGE = 20, 5
DIAGNOSTIC = (27000, 30421)
FLIP_THRESHOLD = 0.2
REL_TOL = 1e-9


def data_folder():
    return os.path.join(config.DATA_DIR, "saddle_correction")


def folder():
    return os.path.join(config.RESULTS_DIR, "saddle_correction")


def run_dir(seed=None):
    return os.path.join(folder(), "temporal", RECIPE) if seed is None else os.path.join(folder(), RECIPE, f"seed_{seed}")


def lm_columns(columns):
    return [c for c in columns if features.uses_signal(c, "lm")]


# ------------------------------------------------------------------ the frozen sample

def draw_sample(ids):
    """Frame (shot_id, group): B and F spread by rank, the edges of the boundary, and the labelled diagnostics."""
    ids = np.array(sorted({int(i) for i in ids}))
    before = ids[ids < saddle_fetch.FLIP_FROM]
    after = ids[(ids >= saddle_fetch.FLIP_FROM) & (ids <= saddle_fetch.FLIP_TO)]
    rows = []
    for name, part, edge in (("B", before, before[-N_EDGE:]), ("F", after, after[:N_EDGE])):
        spread = part[np.unique(np.round(np.linspace(0, len(part) - 1, N_SPREAD)).astype(int))] if len(part) else part
        for s in sorted(set(spread.tolist()) | set(edge.tolist())):
            rows.append({"shot_id": int(s), "group": name})
    chosen = {r["shot_id"] for r in rows}
    rows += [{"shot_id": s, "group": "diagnostic"} for s in DIAGNOSTIC if s not in chosen]
    return pd.DataFrame(rows)


def sample_path():
    return os.path.join(folder(), "frozen_sample.csv")


def window_ids():
    w = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id"])
    return sorted(int(i) for i in w["shot_id"].unique())


def frozen_sample():
    """The frozen sample, drawn once and written; a later draw must give the same list."""
    drawn = draw_sample(window_ids())
    if os.path.exists(sample_path()):
        saved = pd.read_csv(sample_path())
        if not saved.reset_index(drop=True).equals(drawn.reset_index(drop=True)):
            raise RuntimeError("the frozen sample drawn now differs from the saved one; the window table changed")
        return saved
    os.makedirs(folder(), exist_ok=True)
    drawn.to_csv(sample_path(), index=False)
    return drawn


# ------------------------------------------------------------------ the relay

def trigger(ids, sample):
    encoded, enc_sample = sxr_fetch.encode_ids(ids), sxr_fetch.encode_ids(sample)
    if len(encoded) + len(enc_sample) > 60000:
        raise RuntimeError("the shot list is too long for a GitHub input")
    r = requests.post(relay._api(f"/actions/workflows/{WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                      json={"ref": config.RELAY_BRANCH, "inputs": {"ids": encoded, "sample": enc_sample}})
    relay.check(r)
    return time.time()


def runs():
    r = requests.get(relay._api(f"/actions/workflows/{WORKFLOW}/runs"), headers=relay._headers(), params={"per_page": 20},
                     timeout=30)
    relay.check(r)
    return r.json().get("workflow_runs", [])


def collect(run_id, say=print):
    r = requests.get(relay._api(f"/actions/runs/{run_id}/artifacts"), headers=relay._headers(), timeout=30)
    relay.check(r)
    parts = sorted((a for a in r.json().get("artifacts", []) if a["name"].startswith(ARTIFACT) and not a.get("expired")),
                   key=lambda a: a["name"])
    if not parts:
        raise RuntimeError(f"run {run_id} has no {ARTIFACT} artifacts (GitHub keeps them for three days)")
    tables, manifests = {name: [] for name in saddle_fetch.TABLES}, []
    for a in parts:
        z = requests.get(a["archive_download_url"], headers=relay._headers(), timeout=1200)
        relay.check(z)
        with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
            names = {os.path.basename(n): n for n in archive.namelist()}
            if "manifest.json" not in names:
                say(f"  {a['name']}: no manifest (that part failed); its shots are missing")
                continue
            for name in saddle_fetch.TABLES:
                if f"{name}.parquet" in names:
                    tables[name].append(pd.read_parquet(io.BytesIO(archive.read(names[f"{name}.parquet"]))))
            manifests.append(json.loads(archive.read(names["manifest.json"])))
    keys = ("shots_requested", "ok", "no_saddle", "absent", "unreachable", "odd", "not_started")
    m = {k: sum(x.get(k, 0) for x in manifests) for k in keys}
    m["parts_collected"] = len(manifests)
    m["created"] = sorted(x.get("created", "") for x in manifests)
    os.makedirs(data_folder(), exist_ok=True)
    for name, frames in tables.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(os.path.join(data_folder(), f"{name}.parquet"), index=False)
    with open(os.path.join(data_folder(), "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def through_github(say=print):
    if not relay.configured():
        raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
    sample = frozen_sample()
    ids = window_ids()
    requested = trigger(ids, sample["shot_id"].tolist())
    say(f"asked GitHub to read the saddle channels of {len(ids):,} shots ({PARTS} parts; frozen sample "
        f"{len(sample)} shots); waiting for the job ...")
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
    m = collect(run["id"], say)
    say(f"collected: {m}")
    if m["parts_collected"] < PARTS:
        raise RuntimeError(f"only {m['parts_collected']} of {PARTS} parts came back; run --relay again")
    return m


def table(name):
    path = os.path.join(data_folder(), f"{name}.parquet")
    if not os.path.exists(path):
        raise RuntimeError(f"{path} is missing: run --relay first")
    return pd.read_parquet(path)


# ------------------------------------------------------------------ validation V

def channel_medians(channels, sample, value="r_raw"):
    """Frame indexed by label with the median r per period (B, F) over the frozen decision shots, and the shot counts."""
    decide = sample[sample["group"].isin(["B", "F"])]
    part = channels[channels["shot_id"].isin(decide["shot_id"]) & channels["finite"].astype(bool)]
    part = part.merge(decide, on="shot_id")
    med = part.groupby(["label", "group"])[value].median().unstack("group")
    n = part.dropna(subset=[value]).groupby(["label", "group"])["shot_id"].nunique().unstack("group")
    out = med.reindex(columns=["B", "F"]).join(n.reindex(columns=["B", "F"]), rsuffix="_shots")
    return out.reindex(sorted(out.index, key=label_order))


def label_order(label):
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return (int(digits) if digits else 999, str(label))


def flips(med):
    """{label: True/False}: |median| > FLIP_THRESHOLD in both periods and the sign differs."""
    out = {}
    for label, row in med.iterrows():
        b, f = row.get("B"), row.get("F")
        clear = np.isfinite(b) and np.isfinite(f) and abs(b) > FLIP_THRESHOLD and abs(f) > FLIP_THRESHOLD
        out[label] = bool(clear and np.sign(b) != np.sign(f))
    return out


def verdict(med):
    """'V1', 'V2' or 'V3', as the protocol fixes them."""
    flip = flips(med)
    listed = list(saddle_fetch.FLIP)
    if any(label not in med.index for label in listed):
        return "V3"
    others = sum(v for k, v in flip.items() if k not in listed)
    if all(flip[k] for k in listed) and others <= 1:
        return "V1"
    clear = all(np.isfinite(med.loc[k, p]) and abs(med.loc[k, p]) > FLIP_THRESHOLD for k in listed for p in ("B", "F"))
    if not any(flip[k] for k in listed) and clear:
        return "V2"
    return "V3"


# ------------------------------------------------------------------ provenance P1, P2, P3, P5

def same_series(a, b, rel=REL_TOL):
    """True if a and b have the same NaN pattern and finite values equal within rel (relative)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False
    ok = np.isfinite(a)
    return bool(np.all(np.abs(a[ok] - b[ok]) <= rel * np.maximum(np.abs(a[ok]), np.abs(b[ok])) + 1e-30))


def p1(lm, load=None):
    """(shots compared, shots that agree, [disagreeing shots], shots without a cached file)."""
    load = load or (lambda s: extras.load_extras(int(s), allow_download=False))
    compared, agree, differ, missing = 0, 0, [], 0
    for shot, part in lm.groupby("shot_id", sort=True):
        cached = load(shot)
        if cached is None:
            missing += 1
            continue
        compared += 1
        ok = len(cached) == len(part) and np.allclose(cached["time"].to_numpy(), part["time"].to_numpy())
        if ok and same_series(cached["lm_n1"].to_numpy(), part["lm_raw"].to_numpy()):
            agree += 1
        else:
            differ.append(int(shot))
    return compared, agree, differ, missing


def p2(shots):
    lines = []
    for col in ("mag_ingested_at", "sum_ingested_at"):
        if col in shots:
            dates = shots[col].dropna().astype(str).str[:10]
            dates = dates[dates != ""]
            lines.append(f"{col}: " + (f"{dates.min()} to {dates.max()}, {dates.nunique()} dates, {len(dates):,} shots"
                                       if len(dates) else "none stored"))
    for col in ("mag_commit_url", "sum_commit_url"):
        if col in shots:
            counts = shots[col].dropna().astype(str).value_counts()
            lines.append(f"{col}: " + "; ".join(f"{u or '(empty)'} x{n:,}" for u, n in counts.items()))
    return lines


def p3(summary_today, sample, windows, load=load_shot):
    """One line per frozen shot: today's stored summary signals against our cached ones."""
    lines = []
    span = windows.groupby("shot_id").agg(first=("t_start", "min"), last=("t_end", "max"))
    signals = [c for c in summary_today.columns if c not in ("shot_id", "time")]
    for shot, group in sample.itertuples(index=False):
        today = summary_today[summary_today["shot_id"] == shot]
        if not len(today):
            lines.append(f"{shot} ({group}): not read today")
            continue
        try:
            cached = load(int(shot))
        except Exception as error:
            lines.append(f"{shot} ({group}): no cached file ({type(error).__name__})")
            continue
        if len(cached) != len(today) or not np.allclose(cached["time"].to_numpy(), today["time"].to_numpy()):
            lines.append(f"{shot} ({group}): time grid differs ({len(cached)} cached, {len(today)} today)")
            continue
        t = cached["time"].to_numpy()
        lo, hi = (span.loc[shot, "first"], span.loc[shot, "last"]) if shot in span.index else (np.nan, np.nan)
        parts = []
        for s in signals:
            if s not in cached:
                continue
            a, b = cached[s].to_numpy(dtype=float), today[s].to_numpy(dtype=float)
            miss = np.isfinite(a) != np.isfinite(b)
            both = np.isfinite(a) & np.isfinite(b)
            changed = both & (a != b)
            if miss.any() or changed.any():
                d = np.where(changed, np.abs(a - b), 0)
                k = int(np.argmax(d))
                inside = ((t >= lo) & (t <= hi) & (changed | miss)).sum()
                parts.append(f"{s}: {changed.sum()} values changed, {miss.sum()} missingness changed, largest "
                             f"{d[k]:.3g} at {t[k]:.4f} s, {inside} inside the windows")
        lines.append(f"{shot} ({group}): " + ("identical" if not parts else "; ".join(parts)))
    return lines


def p5(shots, campaigns):
    s = shots[shots["status"] == "ok"].merge(campaigns, on="shot_id", how="left")
    lines = ["Saddle channels by campaign (labels; finite channels median, min to max; reversed-sign shots):"]
    for camp, g in s.groupby("campaign", dropna=False):
        labels = g["labels"].value_counts()
        lines.append(f"- {camp}: {len(g):,} shots; labels " + "; ".join(f"[{k}] x{v}" for k, v in labels.items())
                     + f"; finite {g['n_finite'].median():.0f} ({g['n_finite'].min()} to {g['n_finite'].max()}); "
                     f"corrected {int(g['flip_applied'].sum()):,}")
    if "mirnov_label" in shots:
        m = shots.merge(campaigns, on="shot_id", how="left")
        lines.append("Mirnov channel selected (first centre-column channel with data), by campaign:")
        for camp, g in m.groupby("campaign", dropna=False):
            counts = g["mirnov_label"].fillna("none").value_counts()
            lines.append(f"- {camp}: " + "; ".join(f"{k} x{v:,}" for k, v in counts.items()))
    return lines


def check(say=print):
    sample = frozen_sample()
    channels, shots, lm = table("channels"), table("shots"), table("lm")
    windows, shot_table = experiment.load_tables()
    out = ["Saddle-sign correction, validation and provenance (docs/preregistration_saddle_correction.md).", ""]
    try:
        with open(os.path.join(data_folder(), "manifest.json")) as f:
            out.append(f"Relay: {json.load(f)}")
    except OSError:
        pass
    statuses = shots["status"].value_counts().to_dict()
    out.append(f"Shots read: {statuses}; frozen sample {len(sample)} ({sample['group'].value_counts().to_dict()})")
    med = channel_medians(channels, sample, "r_raw")
    v = verdict(med)
    med_c = channel_medians(channels, sample, "r_corr")
    flip_raw, flip_corr = flips(med), flips(med_c)
    out += ["", "V: median r (channel against the median of the others) over the frozen decision shots, as stored | "
            "with the documented signs:"]
    for label in med.index:
        r, c = med.loc[label], med_c.loc[label] if label in med_c.index else None
        out.append(f"- {label}: B {r['B']:+.2f} ({r['B_shots']:.0f} shots), F {r['F']:+.2f} ({r['F_shots']:.0f})"
                   f"{' FLIPS' if flip_raw[label] else ''} | F {c['F']:+.2f}{' FLIPS' if flip_corr.get(label) else ''}"
                   if c is not None else f"- {label}: B {r['B']:+.2f}, F {r['F']:+.2f}")
    out.append({"V1": "V1: the documented reversal is present in our data; the correction is applied.",
                "V2": "V2: the reversal is not present; no correction is applied.",
                "V3": "V3: inconclusive; no correction is applied automatically."}[v])
    diag = sample[sample["group"] == "diagnostic"]["shot_id"]
    for s in diag:
        part = channels[(channels["shot_id"] == s) & channels["label"].isin(saddle_fetch.FLIP)]
        out.append(f"Diagnostic {s} (not in the decision): " + ", ".join(
            f"{r.label} {r.r_raw:+.2f}" for r in part.itertuples()) if len(part) else f"Diagnostic {s}: no channels")
    compared, agree, differ, missing = p1(lm)
    out += ["", f"P1, cache against today's store: {agree:,} of {compared:,} shots give exactly our cached lm_n1 "
            f"({missing} without a cached file)" + (f"; differing: {differ[:40]}{' ...' if len(differ) > 40 else ''}"
                                                   if differ else "")]
    out += ["", "P2, ingestion attributes as stored (not used to identify a release):"] + ["- " + x for x in p2(shots)]
    out += ["", "P3, frozen sample, summary signals today against our cache:"]
    try:
        out += ["- " + x for x in p3(table("summary_sample"), sample, windows)]
    except RuntimeError as error:
        out.append(f"- {error}")
    out += [""] + p5(shots, shot_table[["shot_id", "campaign"]])
    text = "\n".join(out)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "check.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(folder(), "verdict.json"), "w") as f:
        json.dump({"verdict": v, "p1_agree": agree, "p1_compared": compared}, f)
    say(text)
    notify.send("Saddle-sign check\n\n" + text[:3500])
    return v


def saved_verdict():
    path = os.path.join(folder(), "verdict.json")
    if not os.path.exists(path):
        raise RuntimeError("no verdict yet: run --check first")
    with open(path) as f:
        return json.load(f)["verdict"]


# ------------------------------------------------------------------ P4 and the corrected features

def rebuild(shot_row, lm_values, columns):
    """The lm window features of one shot with lm_n1 replaced by lm_values, as features.shot_windows builds them."""
    shot_id = int(shot_row["shot_id"])
    frame = extras.with_extras(load_shot(shot_id), shot_id)
    frame["lm_n1"] = lm_values
    win = dataset.compact(features.shot_windows(frame, t_stop=dataset.stop_time(shot_row)))
    return win[["t_end"] + columns] if len(win) else win


def features_path():
    return os.path.join(data_folder(), "lm_features_corrected.parquet")


def build_features(say=print):
    v = saved_verdict()
    lm = table("lm")
    windows, shots = experiment.load_tables()
    columns = lm_columns(windows.columns)
    in_range = shots[(shots["shot_id"] >= saddle_fetch.FLIP_FROM) & (shots["shot_id"] <= saddle_fetch.FLIP_TO)]
    by_shot = dict(tuple(lm.groupby("shot_id")))
    mismatched, missing, rows, start = [], [], [], time.time()
    for i, (_, row) in enumerate(in_range.iterrows(), 1):
        shot = int(row["shot_id"])
        part = by_shot.get(shot)
        mine = windows[windows["shot_id"] == shot]
        if part is None:
            missing.append(shot)
            continue
        base = load_shot(shot)
        if len(base) != len(part) or not np.allclose(base["time"].to_numpy(), part["time"].to_numpy()):
            missing.append(shot)
            continue
        again = rebuild(row, part["lm_raw"].to_numpy(), columns)
        merged = mine[["t_end"] + columns].merge(again, on="t_end", how="outer", suffixes=("", "_again"))
        same = len(merged) == len(mine) == len(again) and all(
            np.array_equal(merged[c].to_numpy(), merged[f"{c}_again"].to_numpy(), equal_nan=True) for c in columns)
        if not same:
            mismatched.append(shot)
        if v == "V1":
            corrected = rebuild(row, part["lm_corr"].to_numpy(), columns)
            rows.append(corrected.assign(shot_id=np.int64(shot)))
        if i % 500 == 0:
            say(f"[{i}/{len(in_range)}] {(time.time() - start) / i:.2f} s per shot")
    lines = [f"P4: lm window features rebuilt from today's lm_raw equal the window table's for "
             f"{len(in_range) - len(missing) - len(mismatched):,} of {len(in_range) - len(missing):,} shots in "
             f"{saddle_fetch.FLIP_FROM} to {saddle_fetch.FLIP_TO} ({len(missing)} not read or on another grid"
             + (f": {missing[:20]}" if missing else "") + ")"
             + (f"; differing: {mismatched[:40]}" if mismatched else "")]
    if v == "V1" and rows:
        table_c = pd.concat(rows, ignore_index=True)
        table_c.to_parquet(features_path(), index=False)
        changed = 0
        both = windows[windows["shot_id"].isin(table_c["shot_id"])][["shot_id", "t_end"] + columns].merge(
            table_c, on=["shot_id", "t_end"], suffixes=("", "_c"))
        for c in columns:
            changed += int((~np.isclose(both[c], both[f"{c}_c"], equal_nan=True)).sum())
        lines.append(f"Corrected lm features written for {table_c['shot_id'].nunique():,} shots, {len(table_c):,} "
                     f"windows; {changed:,} feature values changed ({len(columns)} columns: {', '.join(columns)}).")
    elif v != "V1":
        lines.append(f"Verdict {v}: no corrected features are built (protocol).")
    text = "\n".join(lines)
    with open(os.path.join(folder(), "features.txt"), "w") as f:
        f.write(text + "\n")
    say(text)
    notify.send("Saddle-sign correction, features\n\n" + text)
    return text


def load_tables():
    """The window and shot tables with the corrected lm features swapped in (only for V1)."""
    if saved_verdict() != "V1" or not os.path.exists(features_path()):
        raise RuntimeError("no corrected features: the verdict is not V1, or --features has not run")
    windows, shots = experiment.load_tables()
    return swap(windows, pd.read_parquet(features_path())), shots


def swap(windows, corrected):
    """windows with the lm columns of every corrected shot replaced by the corrected values; same rows, same order."""
    columns = lm_columns(windows.columns)
    merged = windows[["shot_id", "t_end"]].merge(corrected[["shot_id", "t_end"] + columns], on=["shot_id", "t_end"],
                                                 how="left", indicator=True, validate="one_to_one")
    hit = (merged["_merge"] == "both").to_numpy()
    expected = windows["shot_id"].isin(corrected["shot_id"]).to_numpy()
    if not np.array_equal(hit, expected):
        raise RuntimeError(f"{int((expected & ~hit).sum())} windows of corrected shots have no corrected row")
    out = windows.copy()
    for c in columns:
        out[c] = np.where(hit, merged[c].to_numpy(dtype=out[c].dtype), out[c].to_numpy())
    return out


# ------------------------------------------------------------------ the corrected full model

def run(seed=None, tables=None):
    windows, shots = tables or load_tables()
    if seed is None:
        win, column = dataset.labelled_windows(windows, shots), "split_temporal"
    else:
        split = dataset.random_split(shots, seed=seed)
        win, column = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots), "split"
    return fast_magnetics.fit(win, column, RECIPE, run_dir(seed), {"seed": 0 if seed is None else seed},
                              v3_explore.recipe_columns("baseline"))


def collect_wins(load=None):
    main, placebo, heating = how_early.pair_frames(load)
    random, temporal = [], []
    folders = v3_explore.run_folders()
    sources = [("baseline", folders["baseline"], v3_explore.temporal_dir("baseline")),
               ("machine", folders["machine"], v3_explore.temporal_dir("machine")),
               (no_lm.RECIPE, os.path.join(no_lm.folder(), no_lm.RECIPE), no_lm.run_dir()),
               (RECIPE, os.path.join(folder(), RECIPE), run_dir())]
    for name, base, temporal_path in sources:
        for seed, path in how_early.saved_runs(base):
            if seed in SEEDS:
                random += how_early.run_wins(path, name, seed, main, placebo)
        if server.result_saved(os.path.join(temporal_path, "metrics.json")):
            temporal += how_early.run_wins(temporal_path, name, 0, main, placebo)
    return how_early.finish(random, heating), how_early.finish(temporal, heating)


def pair_counts(main, lead=PRIMARY_LEAD):
    """Per model: pair-split rows at the lead, and whether every model has the same pairs in every split."""
    part = main[main["lead_ms"] == lead]
    sets = {(r, s): frozenset(zip(g["shot_id"], g["other"])) for (r, s), g in part.groupby(["recipe", "seed"])}
    models = sorted({r for r, _ in sets})
    same = all(len({sets.get((m, s)) for m in models}) == 1 for s in {s for _, s in sets})
    counts = part.groupby("recipe").size().to_dict()
    return f"Eligible pair-split rows at {lead} ms: {counts}; identical pairs for every model in every split: {same}"


def text(random, temporal):
    cells = fast_magnetics.difference_cells
    lines = ["Saddle-sign correction (docs/preregistration_saddle_correction.md): A_c = the full model with the lm "
             "features rebuilt from corrected saddle channels. Fixed-lead twin test, common cohort, splits 1 to 20.", ""]
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    if len(main):
        lines.append(pair_counts(main))
    d = how_early.split_difference(main, PRIMARY_LEAD, RECIPE, "machine")
    if d:
        resolved = d[1] is not None and d[1] > 0
        lines.append(f"PRIMARY: C({PRIMARY_LEAD} ms) of A_c minus M: {cells(main, PRIMARY_LEAD, RECIPE, 'machine')}. "
                     + ("Resolved and positive: with the documented sign correction the full model keeps its 10 ms lead "
                        "over the machine-side model." if resolved else
                        "Not resolved: with the correction the 10 ms lead is not shown; the uncorrected full model's "
                        "lead is not presented as established."))
    lines += ["", "Per lead (A_c minus M; A_c minus A; A_c minus L), over splits | pooled:"]
    for lead in how_early.COHORT_LEADS:
        lines.append(f"- {lead} ms: {cells(main, lead, RECIPE, 'machine')}; {cells(main, lead, RECIPE, 'baseline')}; "
                     f"{cells(main, lead, RECIPE, no_lm.RECIPE)}")
    if len(main):
        for label, keep in (("lost shot before 24782", main["shot_id"] < saddle_fetch.FLIP_FROM),
                            ("lost shot from 24782 on", main["shot_id"] >= saddle_fetch.FLIP_FROM)):
            part = main[keep]
            lines.append(f"{label} ({part['shot_id'].nunique():,} lost shots), {PRIMARY_LEAD} ms: A_c minus M "
                         f"{cells(part, PRIMARY_LEAD, RECIPE, 'machine')}; A_c minus A "
                         f"{cells(part, PRIMARY_LEAD, RECIPE, 'baseline')}")
    placebo = random[random["kind"] == "placebo"] if len(random) else random
    if len(placebo):
        split = how_early.over_splits(placebo, PRIMARY_LEAD)
        if RECIPE in split:
            lines.append(f"Placebo at {PRIMARY_LEAD} ms, A_c: {how_early.rng(*split[RECIPE][:3])}")
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)] if len(temporal) else temporal
    for lead in (PRIMARY_LEAD, 30):
        if len(t_main):
            dd = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == lead], RECIPE, "machine", "win")
            da = v3_explore.pooled_delta(t_main[t_main["lead_ms"] == lead], RECIPE, "baseline", "win")
            if dd and da:
                lines.append(f"Chronological split, {lead} ms, pooled (conditional on the one fitted model): A_c minus M "
                             f"{how_early.rng(*dd[:3], signed=True)}; A_c minus A {how_early.rng(*da[:3], signed=True)}")
    u = how_early.usual_difference(os.path.join(folder(), RECIPE), v3_explore.run_folders()["baseline"], set(SEEDS))
    if u:
        lines.append(f"Usual metrics, A_c minus A over {u[1]} splits: pAUC {how_early.rng(*u[0]['pauc'], signed=True)}; "
                     f"caught {how_early.rng(*u[0]['catch_rate'], signed=True)}")
    return "\n".join(lines)


def summary(load=None):
    random, temporal = collect_wins(load)
    out = text(random, temporal)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    return out


def train_all(say=print):
    tables = load_tables()
    todo = [s for s in list(SEEDS) + [None] if not server.result_saved(os.path.join(run_dir(s), "metrics.json"))]
    say(f"{len(todo)} runs to do")
    for i, seed in enumerate(todo, 1):
        start = time.time()
        m = run(seed, tables)
        say(f"[{i}/{len(todo)}] {'chronological' if seed is None else f'seed {seed}'}: pAUC {m['test']['pauc']:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    out = summary()
    say(out)
    notify.send("Saddle-sign correction: the runs are done\n\n" + out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("sample", "relay", "check", "features", "train", "summary"):
        parser.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    if args.sample:
        s = frozen_sample()
        say(f"frozen sample ({sample_path()}): {s['group'].value_counts().to_dict()}\n" + s.to_string(index=False))
    elif args.relay:
        through_github(say)
    elif args.check:
        check(say)
    elif args.features:
        build_features(say)
    elif args.train:
        train_all(say)
    elif args.summary:
        say(summary())
    else:
        parser.error("give one of --sample, --relay, --check, --features, --train, --summary")


if __name__ == "__main__":
    main()
