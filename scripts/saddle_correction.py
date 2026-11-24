"""Saddle-coil sign correction of the locked-mode feature (docs/preregistration_saddle_correction.md).

    python scripts/saddle_correction.py --sample     # draw the frozen sample (once; a later draw must give the same)
    python scripts/saddle_correction.py --relay      # read the saddle channels on GitHub (scripts/saddle_fetch.py)
    python scripts/saddle_correction.py --check      # V, A2, gates G1 (cache), G3 (polarity), G4 (conditioning); P2, P3, P5;
                                                     # the channel mapping table (A3)
    python scripts/saddle_correction.py --features   # G2 (rebuild reproduced); the corrected lm features only if G1-G4 pass
    python scripts/saddle_correction.py --g5         # G5 before training: same non-lm columns, same eligible pairs
    python scripts/saddle_correction.py --train      # A_c on splits 1 to 20 and the chronological split; refuses unless
                                                     # every gate passes (amendment 1)
    python scripts/saddle_correction.py --summary

Reads the window table and caches, never changes them; writes DATA_DIR/saddle_correction/ and
RESULTS_DIR/saddle_correction/; never writes into v3/. Outside every stamp.
"""
import argparse
import io
import json
import os
import sys
import shutil
import time
import warnings
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
CONFIRM_SEEDS = range(21, 51)  # amendment 3: the confirmation splits of the twin paper
PRIMARY_LEAD = 10
WORKFLOW, ARTIFACT, PARTS = "fetch_saddle.yml", "mast-saddle", 8
N_SPREAD, N_EDGE = 20, 5
DIAGNOSTIC = (27000, 30421)
FLIP_THRESHOLD = 0.2
REL_TOL = 1e-9  # G1: same code, same stored data, different machines
REBUILD_TOL = 1e-6  # G2: float32 storage of the window table
MIN_SHOTS = 15  # amendment 1: shots with a finite r on each side
MAX_COND = 10.0  # G4
GATES = ("G1a", "G2a", "G3", "G4", "G5")  # amendment 2: all must pass before training


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
    keys = ("shots_requested", "ok", "no_saddle", "absent", "unreachable", "odd", "not_started", "release_unverified")
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


# ------------------------------------------------------------------ validation V and the reference diagnostic A2

def channel_medians(channels, sample, value="r_raw"):
    """Frame indexed by label: median r per period (B, F) over the frozen decision shots, the number of shots with a
    finite r (B_shots, F_shots), and the counts of clear positive, clear negative and indeterminate shots."""
    decide = sample[sample["group"].isin(["B", "F"])]
    part = channels[channels["shot_id"].isin(decide["shot_id"]) & channels["finite"].astype(bool)]
    part = part.merge(decide, on="shot_id")
    r = part[value]
    part = part.assign(pos=r > FLIP_THRESHOLD, neg=r < -FLIP_THRESHOLD, unclear=~((r > FLIP_THRESHOLD) | (r < -FLIP_THRESHOLD)),
                       has=r.notna())
    g = part.groupby(["label", "group"])
    stats = {"": g[value].median(), "_shots": g["has"].sum(), "_pos": g["pos"].sum(), "_neg": g["neg"].sum(),
             "_unclear": g["unclear"].sum()}
    out = pd.concat([s.unstack("group").reindex(columns=["B", "F"]).add_suffix(k) for k, s in stats.items()], axis=1)
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


def enough_shots(med, labels):
    return all(label in med.index and med.loc[label, "B_shots"] >= MIN_SHOTS and med.loc[label, "F_shots"] >= MIN_SHOTS
               for label in labels)


def verdict(med):
    """'V1', 'V2' or 'V3', as the protocol fixes them (with amendment 1's minimum of MIN_SHOTS per side)."""
    listed = list(saddle_fetch.FLIP)
    if not enough_shots(med, listed):
        return "V3"
    flip = flips(med)
    others = sum(v for k, v in flip.items() if k not in listed)
    if all(flip[k] for k in listed) and others <= 1:
        return "V1"
    clear = all(np.isfinite(med.loc[k, p]) and abs(med.loc[k, p]) > FLIP_THRESHOLD for k in listed for p in ("B", "F"))
    if not any(flip[k] for k in listed) and clear:
        return "V2"
    return "V3"


def reference_r(bins):
    """Amendment 1, A2: per frozen shot and channel, r_ref against the median of the documented-unaffected channels
    with data (other than itself), over the bins where lm_raw is defined. Same columns as the channels table."""
    rows = []
    for shot, part in bins.groupby("shot_id", sort=True):
        wide = part.pivot_table(index="time", columns="label", values="delta", aggfunc="first", dropna=False)
        defined = part.groupby("time")["defined"].first().reindex(wide.index, fill_value=False).to_numpy(dtype=bool)
        finite = {c: bool(np.isfinite(wide[c].to_numpy()).any()) for c in wide.columns}
        reference = [c for c in wide.columns if c not in saddle_fetch.FLIP and finite[c]]
        for label in wide.columns:
            ref = [c for c in reference if c != label]
            r = np.nan
            if finite[label] and ref:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    median = np.nanmedian(wide[ref].to_numpy(dtype=float), axis=1)
                x = wide[label].to_numpy(dtype=float)
                ok = defined & np.isfinite(x) & np.isfinite(median)
                if ok.sum() >= saddle_fetch.MIN_BINS and np.std(x[ok]) > 0 and np.std(median[ok]) > 0:
                    r = float(np.corrcoef(x[ok], median[ok])[0, 1])
            rows.append({"shot_id": int(shot), "label": label, "finite": finite[label], "r_ref": r})
    return pd.DataFrame(rows, columns=["shot_id", "label", "finite", "r_ref"])


def a2_consistent(med_ref):
    """A2 agrees with V1: the three listed channels flip, no reference channel flips, MIN_SHOTS per side for the three."""
    listed = list(saddle_fetch.FLIP)
    if not enough_shots(med_ref, listed):
        return False
    flip = flips(med_ref)
    return all(flip.get(k, False) for k in listed) and not any(v for k, v in flip.items() if k not in listed)


def design_condition(mask):
    """(rank, condition number) of the n=1 fit's design matrix for one finite-channel mask (assumed angles)."""
    phi = np.radians(extras.SADDLE_PHI_DEG[:len(mask)])[np.asarray(mask, dtype=bool)]
    design = np.column_stack([np.ones(len(phi)), np.cos(phi), np.sin(phi)])
    if len(phi) < 3:
        return int(np.linalg.matrix_rank(design)) if len(phi) else 0, np.inf
    return int(np.linalg.matrix_rank(design)), float(np.linalg.cond(design))


def conditioning(channels, ids):
    """G4: every finite-channel mask in the given shots, with its shots, rank and condition number."""
    part = channels[channels["shot_id"].isin(ids)].sort_values(["shot_id", "index"])
    masks = part.groupby("shot_id")["finite"].apply(lambda f: tuple(bool(x) for x in f))
    rows = []
    for mask, n in masks.value_counts().items():
        rank, cond = design_condition(mask)
        rows.append({"mask": "".join("1" if m else "0" for m in mask), "shots": int(n), "finite": sum(mask),
                     "rank": rank, "cond": cond})
    frame = pd.DataFrame(rows, columns=["mask", "shots", "finite", "rank", "cond"])
    usable = frame[frame["finite"] >= extras.MIN_SADDLE_CHANNELS]
    ok = bool(len(usable)) and bool(((usable["rank"] == 3) & (usable["cond"] <= MAX_COND)).all())
    return frame, ok


# ------------------------------------------------------------------ provenance P1, P2, P3, P5 and the mapping table

def same_series(a, b, rel=REL_TOL):
    """True if a and b have the same NaN pattern and finite values equal within rel (relative)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return False
    ok = np.isfinite(a)
    return bool(np.all(np.abs(a[ok] - b[ok]) <= rel * np.maximum(np.abs(a[ok]), np.abs(b[ok])) + 1e-30))


def p1(lm, load=None, ids=None):
    """{'compared', 'agree', 'differ': [...], 'no_cache', 'largest'}: today's lm_raw against our cached lm_n1, for the
    shots in ids (all read shots if None). 'largest' is the largest relative difference among finite values."""
    load = load or (lambda s: extras.load_extras(int(s), allow_download=False))
    part_lm = lm if ids is None else lm[lm["shot_id"].isin(ids)]
    out = {"compared": 0, "agree": 0, "differ": [], "no_cache": 0, "largest": 0.0}
    for shot, part in part_lm.groupby("shot_id", sort=True):
        cached = load(shot)
        if cached is None:
            out["no_cache"] += 1
            continue
        out["compared"] += 1
        a, b = cached["lm_n1"].to_numpy(dtype=float), part["lm_raw"].to_numpy(dtype=float)
        ok = len(cached) == len(part) and np.allclose(cached["time"].to_numpy(), part["time"].to_numpy(), rtol=0,
                                                      atol=1e-9)
        if ok and np.array_equal(np.isfinite(a), np.isfinite(b)) and np.isfinite(a).any():
            f = np.isfinite(a)
            rel = np.abs(a[f] - b[f]) / np.maximum(np.maximum(np.abs(a[f]), np.abs(b[f])), 1e-30)
            out["largest"] = max(out["largest"], float(rel.max()))
        if ok and same_series(a, b):
            out["agree"] += 1
        else:
            out["differ"].append(int(shot))
    return out


def p2(shots):
    lines = []
    if "mag_ingested_at" in shots:
        has_group = shots["mag_ingested_at"].notna()
        states = {"saddle data": int((shots["status"] == "ok").sum()),
                  "magnetics group without saddle field": int(((shots["status"] == "no_saddle") & has_group).sum()),
                  "no magnetics group": int(((shots["status"] == "no_saddle") & ~has_group).sum()),
                  "other": int((~shots["status"].isin(["ok", "no_saddle"])).sum())}
        lines.append(f"states of the {len(shots):,} shots read: {states}")
        for col in ("mag_commit_url", "sum_commit_url"):
            if col in shots:
                lines.append(f"{col} missing or empty: {int((shots[col].isna() | (shots[col].astype(str) == '')).sum())}")
    for col in ("mag_ingested_at", "sum_ingested_at"):
        if col in shots:
            dates = shots[col].dropna().astype(str).str[:10]
            dates = dates[dates != ""]
            lines.append(f"{col}: " + (f"{dates.min()} to {dates.max()}, {dates.nunique()} dates, {len(dates):,} shots"
                                       if len(dates) else "none stored"))
    for col in ("mag_commit_url", "sum_commit_url", "field_units"):
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
                     + f"; finite {g['n_finite'].median():.0f} ({g['n_finite'].min():.0f} to {g['n_finite'].max():.0f}); "
                     f"corrected {int(g['flip_applied'].astype(bool).sum()):,}")
    if "mirnov_label" in shots:
        m = shots.merge(campaigns, on="shot_id", how="left")
        lines.append("Mirnov channel selected (first centre-column channel with data), by campaign:")
        for camp, g in m.groupby("campaign", dropna=False):
            counts = g["mirnov_label"].fillna("none").value_counts()
            lines.append(f"- {camp}: " + "; ".join(f"{k} x{v:,}" for k, v in counts.items()))
    return lines


def mapping_table(channels, shots, campaigns):
    """Amendment 1, A3: stored label, assumed coil and angle, units, evidence, share of shots with data by campaign."""
    part = channels.merge(campaigns, on="shot_id", how="left")
    units = shots["field_units"].dropna().astype(str).value_counts().index[0] if "field_units" in shots and \
        shots["field_units"].notna().any() else "?"
    rows = []
    for (index, label), g in part.groupby(["index", "label"], sort=True):
        share = g.groupby("campaign", dropna=False)["finite"].mean()
        rows.append({"stored label": label, "stored position": int(index), "assumed coil": f"middle row, coil {index + 1}",
                     "assumed angle (deg)": float(extras.SADDLE_PHI_DEG[index]) if index < 12 else np.nan,
                     "assumed extent (deg)": 20.0, "units": units,
                     "evidence": "name and stored order only (not confirmed)",
                     "with data, by campaign": "; ".join(f"{c} {v:.0%}" for c, v in share.items())})
    return pd.DataFrame(rows)


def write_gates(**new):
    path = os.path.join(folder(), "gates.json")
    gates = {}
    if os.path.exists(path):
        with open(path) as f:
            gates = json.load(f)
    gates.update(new)
    os.makedirs(folder(), exist_ok=True)
    with open(path, "w") as f:
        json.dump(gates, f, indent=2)
    return gates


def read_gates():
    path = os.path.join(folder(), "gates.json")
    if not os.path.exists(path):
        raise RuntimeError("no gates yet: run --check first")
    with open(path) as f:
        return json.load(f)


def in_range_ids(shots_table):
    s = shots_table["shot_id"]
    return set(int(i) for i in s[(s >= saddle_fetch.FLIP_FROM) & (s <= saddle_fetch.FLIP_TO)])


def med_line(label, r, c=None, flip=False, flip_c=False):
    counts = lambda p: f"{r[p + '_pos']:.0f}+/{r[p + '_neg']:.0f}-/{r[p + '_unclear']:.0f}?"  # noqa: E731
    line = (f"- {label}: B {r['B']:+.2f} ({r['B_shots']:.0f} shots, {counts('B')}), F {r['F']:+.2f} "
            f"({r['F_shots']:.0f} shots, {counts('F')}){' FLIPS' if flip else ''}")
    if c is not None:
        line += f" | corrected F {c['F']:+.2f}{' FLIPS' if flip_c else ''}"
    return line


def preserve_originals():
    """Amendment 2: keep the first check's outputs unchanged, once, before anything is rewritten."""
    for name in ("check.txt", "features.txt", "gates.json"):
        src = os.path.join(folder(), name)
        stem, ext = os.path.splitext(name)
        dst = os.path.join(folder(), f"{stem}_original{ext}")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)


def absence_state(row):
    """Amendment 2 (a): 'no magnetics group' or 'no saddle field' for a verified absence, None for anything else
    (a shot missing from the read, a read error, an odd schema)."""
    if row is None or row.get("status") != "no_saddle":
        return None
    attrs = row.get("mag_ingested_at")
    return "no magnetics group" if attrs is None or (isinstance(attrs, float) and np.isnan(attrs)) else "no saddle field"


def absence_cache_ok(cached, grid_len):
    """Amendment 2 (b): the cached lm_n1 exists on the shot's own cached grid and is entirely NaN."""
    if cached is None or "lm_n1" not in cached or len(cached) != grid_len:
        return False, "no cached feature on the shot's grid"
    if np.isfinite(cached["lm_n1"].to_numpy(dtype=float)).any():
        return False, "cached feature has finite values"
    return True, ""


def absence_windows_ok(mine, again, columns):
    """Amendment 2 (c) and (d): rebuilt with an all-NaN lm_n1, the window rows are reproduced exactly, and in the window
    table has_lm is 0 and the other lm columns are NaN in every window."""
    same, _ = compare_rebuild(mine, again, columns, rel=0.0)
    if not same:
        return False, "rebuilt window rows differ"
    for c in columns:
        x = mine[c].to_numpy(dtype=float)
        if c == "has_lm":
            if not np.array_equal(x, np.zeros(len(x))):
                return False, "has_lm is not 0 in every window"
        elif np.isfinite(x).any():
            return False, f"{c} has finite values"
    return True, ""


def check(say=print):
    preserve_originals()
    sample = frozen_sample()
    channels, shots, lm = table("channels"), table("shots"), table("lm")
    windows, shot_table = experiment.load_tables()
    campaigns = shot_table[["shot_id", "campaign"]]
    out = ["Saddle-sign correction, validation and provenance (docs/preregistration_saddle_correction.md, with "
           "amendment 1).", ""]
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
    out += ["", "V (registered): median r against the median of all other channels, frozen decision shots; counts "
            "clear +/clear -/indeterminate | with the documented signs:"]
    out += [med_line(k, med.loc[k], med_c.loc[k] if k in med_c.index else None, flip_raw[k], flip_corr.get(k, False))
            for k in med.index]
    out.append({"V1": "V1: the documented reversal pattern is present in our sampled data.",
                "V2": "V2: no reversal pattern (not proof that the data were already corrected).",
                "V3": "V3: inconclusive (not evidence that the documented correction is wrong)."}[v])
    for s in sample[sample["group"] == "diagnostic"]["shot_id"]:
        part = channels[(channels["shot_id"] == s) & channels["label"].isin(saddle_fetch.FLIP)]
        out.append((f"Diagnostic {s} (not in the decision): " + ", ".join(f"{r.label} {r.r_raw:+.2f}"
                                                                          for r in part.itertuples()))
                   if len(part) else f"Diagnostic {s}: no channels")

    try:
        ref = reference_r(table("channel_bins"))
        med_ref = channel_medians(ref, sample, "r_ref")
        flip_ref = flips(med_ref)
        a2 = a2_consistent(med_ref)
        out += ["", "A2 (amendment 1): median r against the median of the documented-unaffected channels:"]
        out += [med_line(k, med_ref.loc[k], flip=flip_ref[k]) for k in med_ref.index]
        out.append("A2 is consistent with V1." if a2 else "A2 is NOT consistent with V1.")
    except RuntimeError as error:
        a2 = False
        out += ["", f"A2: not computed ({error})"]

    ids = in_range_ids(shot_table)
    cond, g4 = conditioning(channels, ids)
    out += ["", f"G4, fit conditioning, shots {saddle_fetch.FLIP_FROM} to {saddle_fetch.FLIP_TO} (finite-channel "
            f"masks; fits need {extras.MIN_SADDLE_CHANNELS}):"]
    out += [f"- {r.mask}: {r.shots:,} shots, {r.finite} channels, rank {r.rank}, condition {r.cond:.2f}"
            for r in cond.itertuples()]
    out.append(f"G4 {'passes' if g4 else 'FAILS'} (rank 3 and condition <= {MAX_COND} for every usable mask).")

    every = p1(lm)
    gated = p1(lm, ids=ids)
    read = set(int(i) for i in lm["shot_id"].unique())
    unread = sorted(ids - read)
    g1 = not unread and gated["no_cache"] == 0 and not gated["differ"] and gated["compared"] == len(ids)
    rows = shots.drop_duplicates("shot_id").set_index("shot_id")
    absent, absent_fail = {}, {}
    for shot in unread:
        state = absence_state(rows.loc[shot].to_dict() if shot in rows.index else None)
        if state is None:
            absent_fail[shot] = "not a verified absence (" + (str(rows.loc[shot, "status"]) if shot in rows.index
                                                               else "missing from the read") + ")"
            continue
        try:
            grid_len = len(load_shot(int(shot)))
        except Exception as error:
            absent_fail[shot] = f"no cached shot ({type(error).__name__})"
            continue
        ok, why = absence_cache_ok(extras.load_extras(int(shot), allow_download=False), grid_len)
        if ok:
            absent[shot] = state
        else:
            absent_fail[shot] = why
    g1a = gated["no_cache"] == 0 and not gated["differ"] and not absent_fail and \
        gated["compared"] + len(absent) == len(ids)
    with open(os.path.join(folder(), "structural_absence.csv"), "w") as f:
        f.write("shot_id,state\n" + "".join(f"{k},{v}\n" for k, v in sorted(absent.items())))
    states = pd.Series(absent, dtype=object).value_counts().to_dict() if absent else {}
    out += ["", f"P1, all read shots: {every['agree']:,} of {every['compared']:,} reproduce our cached lm_n1 "
            f"({every['no_cache']} without a cached file; largest relative difference {every['largest']:.2e})"
            + (f"; differing: {every['differ'][:40]}{' ...' if len(every['differ']) > 40 else ''}" if every["differ"]
               else ""),
            f"G1 (as first written), shots {saddle_fetch.FLIP_FROM} to {saddle_fetch.FLIP_TO} of the window table "
            f"({len(ids):,}): {gated['agree']:,} reproduce, {len(gated['differ'])} differ, {len(unread)} not read"
            + (f" {unread[:20]}" if unread else "") + f". G1 {'passes' if g1 else 'FAILS'}.",
            f"G1a (amendment 2): {gated['agree']:,} data-present shots reproduce numerically, {len(gated['differ'])} "
            f"differ; {len(absent):,} structurally absent and verified ({states}); {len(absent_fail)} not verified"
            + (f" {dict(list(absent_fail.items())[:20])}" if absent_fail else "")
            + f". G1a {'passes' if g1a else 'FAILS'}. List: structural_absence.csv."]
    out += ["", "P2, ingestion attributes as stored (not used to identify a release):"] + ["- " + x for x in p2(shots)]
    out += ["", "P3, frozen sample, summary signals today against our cache:"]
    try:
        out += ["- " + x for x in p3(table("summary_sample"), sample, windows)]
    except RuntimeError as error:
        out.append(f"- {error}")
    out += [""] + p5(shots, campaigns)
    mapping = mapping_table(channels, shots, campaigns)
    mapping.to_csv(os.path.join(folder(), "channel_mapping.csv"), index=False)
    out += ["", f"A3: channel-to-angle mapping table written to channel_mapping.csv ({len(mapping)} channels; evidence: "
            "name and stored order only)."]
    g3 = v == "V1" and a2
    gates = write_gates(verdict=v, a2=a2, G1=g1, G1a=g1a, G3=g3, G4=g4)
    out += ["", f"Gates so far: {gates}"]
    text = "\n".join(out)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "check.txt"), "w") as f:
        f.write(text + "\n")
    say(text)
    notify.send("Saddle-sign check\n\n" + text[:3500])
    return gates


# ------------------------------------------------------------------ G2 (P4) and the corrected features

def rebuild(shot_row, lm_values, columns):
    """The lm window features of one shot with lm_n1 replaced by lm_values, as features.shot_windows builds them."""
    shot_id = int(shot_row["shot_id"])
    frame = extras.with_extras(load_shot(shot_id), shot_id)
    frame["lm_n1"] = lm_values
    win = dataset.compact(features.shot_windows(frame, t_stop=dataset.stop_time(shot_row)))
    return win[["t_end"] + columns] if len(win) else win


def compare_rebuild(mine, again, columns, rel=REBUILD_TOL):
    """(same, largest relative difference): same window ends, same NaN pattern, values within rel."""
    if len(mine) != len(again) or not np.array_equal(mine["t_end"].to_numpy(), again["t_end"].to_numpy()):
        return False, np.inf
    largest, same = 0.0, True
    for c in columns:
        a, b = mine[c].to_numpy(dtype=float), again[c].to_numpy(dtype=float)
        if not np.array_equal(np.isfinite(a), np.isfinite(b)):
            same = False
            continue
        f = np.isfinite(a)
        if f.any():
            d = np.abs(a[f] - b[f]) / np.maximum(np.maximum(np.abs(a[f]), np.abs(b[f])), 1e-30)
            largest = max(largest, float(d.max()))
            same = same and bool((d <= rel).all())
    return same, largest


def features_path():
    return os.path.join(data_folder(), "lm_features_corrected.parquet")


def absent_list():
    path = os.path.join(folder(), "structural_absence.csv")
    return set(pd.read_csv(path)["shot_id"].astype(int)) if os.path.exists(path) else set()


def build_features(say=print):
    preserve_originals()
    gates = read_gates()
    polarity = gates.get("G1a") and gates.get("G3") and gates.get("G4")
    lm = table("lm")
    windows, shots = experiment.load_tables()
    columns = lm_columns(windows.columns)
    in_range = shots[shots["shot_id"].isin(in_range_ids(shots))]
    by_shot = dict(tuple(lm.groupby("shot_id")))
    absent = absent_list()
    mismatched, missing, rows, largest, start = [], [], [], 0.0, time.time()
    absent_ok, absent_bad, data_windows, absent_windows = [], {}, 0, 0
    for i, (_, row) in enumerate(in_range.iterrows(), 1):
        shot = int(row["shot_id"])
        part = by_shot.get(shot)
        mine = windows[windows["shot_id"] == shot][["t_end"] + columns].sort_values("t_end").reset_index(drop=True)
        base = load_shot(shot)
        if part is None:
            missing.append(shot)
            if shot not in absent:
                absent_bad[shot] = "not a verified structural absence (G1a)"
                continue
            again = rebuild(row, np.full(len(base), np.nan), columns)
            again = again.sort_values("t_end").reset_index(drop=True) if len(again) else again
            ok, why = absence_windows_ok(mine, again, columns)
            if ok:
                absent_ok.append(shot)
                absent_windows += len(mine)
            else:
                absent_bad[shot] = why
            continue
        if len(base) != len(part) or not np.allclose(base["time"].to_numpy(), part["time"].to_numpy()):
            missing.append(shot)
            absent_bad[shot] = "read on another grid"
            continue
        again = rebuild(row, part["lm_raw"].to_numpy(), columns)
        again = again.sort_values("t_end").reset_index(drop=True) if len(again) else again
        same, diff = compare_rebuild(mine, again, columns)
        largest = max(largest, diff)
        data_windows += len(mine)
        if not same:
            mismatched.append(shot)
        if polarity:
            rows.append(rebuild(row, part["lm_corr"].to_numpy(), columns).assign(shot_id=np.int64(shot)))
        if i % 500 == 0:
            say(f"[{i}/{len(in_range)}] {(time.time() - start) / i:.2f} s per shot")
    g2 = not missing and not mismatched
    g2a = not mismatched and not absent_bad and len(absent_ok) == len(missing)
    present = len(in_range) - len(missing)
    lines = [f"G2 (as first written): lm window features rebuilt from today's lm_raw reproduce the window table's for "
             f"{present - len(mismatched):,} of {len(in_range):,} shots in {saddle_fetch.FLIP_FROM} to "
             f"{saddle_fetch.FLIP_TO} (tolerance {REBUILD_TOL:g} relative; largest {largest:.2e}); {len(missing)} not read "
             f"or on another grid" + (f"; differing: {mismatched[:40]}" if mismatched else "")
             + f". G2 {'passes' if g2 else 'FAILS'}.",
             f"G2a (amendment 2): data present: {present - len(mismatched):,} of {present:,} shots ({data_windows:,} "
             f"windows) reproduced numerically; structurally absent: {len(absent_ok):,} shots ({absent_windows:,} windows) "
             f"reproduced with matching missingness (has_lm 0, other lm columns NaN, window ends unchanged); "
             f"{len(absent_bad)} failed" + (f" {dict(list(absent_bad.items())[:20])}" if absent_bad else "")
             + f". G2a {'passes' if g2a else 'FAILS'}."]
    if polarity and g2a and rows:
        table_c = pd.concat(rows, ignore_index=True)
        table_c.to_parquet(features_path(), index=False)
        both = windows[windows["shot_id"].isin(table_c["shot_id"])][["shot_id", "t_end"] + columns].merge(
            table_c, on=["shot_id", "t_end"], suffixes=("", "_c"))
        changed = sum(int((~np.isclose(both[c], both[f"{c}_c"], equal_nan=True)).sum()) for c in columns)
        lines.append(f"Corrected lm features written for {table_c['shot_id'].nunique():,} data-present shots, "
                     f"{len(table_c):,} windows; {changed:,} feature values changed ({len(columns)} columns: "
                     f"{', '.join(columns)}). Structurally absent shots are left untouched.")
    else:
        if os.path.exists(features_path()):
            os.remove(features_path())
        lines.append("No corrected features: " + ", ".join(
            k for k, ok in (("G1a", gates.get("G1a")), ("G2a", g2a), ("G3", gates.get("G3")), ("G4", gates.get("G4")))
            if not ok) + " failed. The direct swap is not made (protocol, amendments 1 and 2).")
    write_gates(G2=g2, G2a=g2a)
    text = "\n".join(lines)
    with open(os.path.join(folder(), "features.txt"), "w") as f:
        f.write(text + "\n")
    say(text)
    notify.send("Saddle-sign correction, features\n\n" + text)
    return text


# ------------------------------------------------------------------ G5 before training

def manifest(windows, shots, main, seeds=SEEDS):
    """Eligible (split, lost shot, twin, lead) of the twin test from the test windows with a constant score: what any
    model scored on these windows could be compared on."""
    keys = []
    for seed in list(seeds) + [None]:
        if seed is None:
            win, column = dataset.labelled_windows(windows, shots), "split_temporal"
        else:
            split = dataset.random_split(shots, seed=seed)
            win, column = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots), "split"
        test = experiment.split_frames(win, column)[2]
        scores = pd.DataFrame({"shot_id": test["shot_id"].astype("int32"), "t_end": test["t_end"].astype("float32"),
                               "score": 0.0})  # as the runs save them
        mine = main[main["shot_id"].isin(test["shot_id"]) & main["other"].isin(test["shot_id"])]
        got = how_early.landmark_wins(scores, mine)
        keys += [(0 if seed is None else seed, int(a), int(b), int(c))
                 for a, b, c in zip(got["shot_id"], got["other"], got["lead_ms"])]
    return set(keys)


def non_lm_equal(original, corrected):
    """True if every column except the lm columns is identical (same rows, order, values and NaN pattern)."""
    if list(original.columns) != list(corrected.columns) or len(original) != len(corrected):
        return False
    lm = set(lm_columns(original.columns))
    for c in original.columns:
        if c in lm:
            continue
        a, b = original[c].to_numpy(), corrected[c].to_numpy()
        same = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
        if not same:
            return False
    return True


def g5(say=print):
    g = read_gates()
    if not all(g.get(k) for k in ("G1a", "G2a", "G3", "G4")) or not os.path.exists(features_path()):
        raise RuntimeError("G5 needs G1a, G2a, G3, G4 and the corrected features first")
    windows, shots = experiment.load_tables()
    corrected = swap(windows, pd.read_parquet(features_path()))
    main = how_early.pair_frames()[0]
    same_columns = non_lm_equal(windows, corrected)
    a, b = manifest(windows, shots, main), manifest(corrected, shots, main)
    ok = same_columns and a == b
    text = (f"G5 (before training): non-lm columns identical: {same_columns}; eligible (split, pair, lead) entries "
            f"original {len(a):,}, corrected {len(b):,}, only in one: {len(a ^ b)}. G5 {'passes' if ok else 'FAILS'}.")
    write_gates(G5=ok)
    with open(os.path.join(folder(), "g5.txt"), "w") as f:
        f.write(text + "\n")
    say(text)
    notify.send("Saddle-sign correction\n\n" + text)
    return ok


def gates_pass():
    g = read_gates()
    failed = [k for k in GATES if not g.get(k)]
    return not failed, failed


def load_tables():
    """The window and shot tables with the corrected lm features swapped in (only when every gate passes)."""
    ok, failed = gates_pass()
    if not ok or not os.path.exists(features_path()):
        raise RuntimeError(f"no corrected training: gates failed {failed} or --features has not run (amendments 1, 2)")
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
    return (f"G5: eligible pair-split rows at {lead} ms: {counts}; identical pairs for every model in every split: "
            f"{'yes' if same else 'NO (the comparison is not fixed; stop and investigate)'}")


def text(random, temporal):
    cells = fast_magnetics.difference_cells
    lines = ["Saddle-sign correction (docs/preregistration_saddle_correction.md, amendment 1): A_c = the full model with "
             "the first-harmonic saddle-array features (assumed geometry) rebuilt from sign-corrected channels; A = the "
             "original representation; L = the no-lm sensitivity. Fixed-lead twin test, common cohort, splits 1 to 20.",
             "A_c is the corrected full-model analysis whatever its result (A4); 10 ms was chosen after earlier work had "
             "shown it to be the strongest lead.", ""]
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)] if len(random) else random
    if len(main):
        lines.append(pair_counts(main))
    d = how_early.split_difference(main, PRIMARY_LEAD, RECIPE, "machine")
    if d:
        resolved = d[1] is not None and d[1] > 0
        lines.append(f"PRIMARY: C({PRIMARY_LEAD} ms) of A_c minus M: {cells(main, PRIMARY_LEAD, RECIPE, 'machine')}. "
                     + ("Resolved and positive: the corrected full model shows the 10 ms lead over the machine-side "
                        "model." if resolved else
                        "Not resolved: the corrected full model does not show the 10 ms lead; the uncorrected full "
                        "model's lead is not presented as established, and the claim narrows to the sensitivities."))
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


def train_all(say=print, seeds=None):
    """A_c on splits 1 to 20 and the chronological split; with seeds given (amendment 3: splits 21 to 50), only those,
    by the same frozen procedure, and the primary summary is not changed."""
    tables = load_tables()
    wanted = list(SEEDS) + [None] if seeds is None else list(seeds)
    todo = [s for s in wanted if not server.result_saved(os.path.join(run_dir(s), "metrics.json"))]
    say(f"{len(todo)} runs to do")
    for i, seed in enumerate(todo, 1):
        start = time.time()
        m = run(seed, tables)
        say(f"[{i}/{len(todo)}] {'chronological' if seed is None else f'seed {seed}'}: pAUC {m['test']['pauc']:.3f}, "
            f"{(time.time() - start) / 60:.1f} min")
    if seeds is not None:
        msg = f"Saddle-sign correction: A_c on splits {min(wanted)} to {max(wanted)} done ({len(todo)} new runs)."
        say(msg)
        notify.send(msg)
        return
    out = summary()
    say(out)
    notify.send("Saddle-sign correction: the runs are done\n\n" + out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ("sample", "relay", "check", "features", "g5", "train", "summary"):
        parser.add_argument(f"--{flag}", action="store_true")
    parser.add_argument("--confirm-splits", action="store_true",
                        help="with --train: A_c on splits 21 to 50 only (amendment 3), for the 50-split tables")
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
    elif args.g5:
        if not g5(say):
            sys.exit(1)
    elif args.train:
        train_all(say, CONFIRM_SEEDS if args.confirm_splits else None)
    elif args.summary:
        say(summary())
    else:
        parser.error("give one of --sample, --relay, --check, --features, --g5, --train, --summary")


if __name__ == "__main__":
    main()
