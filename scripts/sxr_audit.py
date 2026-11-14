"""Soft X-ray quality audit (faster signals, step 1): which twin pairs have usable soft X-ray data, before any
soft X-ray feature or model exists. Looks at no model score. The only outcome it uses is whether a shot was lost,
to check that missing data does not itself tell lost shots from kept ones.

    python scripts/sxr_audit.py --relay          # pick the sample, start the GitHub job, wait, collect (about an hour)
    python scripts/sxr_audit.py --collect        # collect the newest finished job
    python scripts/sxr_audit.py                  # the audit in words, from the collected table

A chord is usable if it covers at least COVER of the plasma's time, its signal-to-noise ratio is at least SNR and at
most SAT of its samples sit at its maximum. A shot is usable in a camera with at least MIN_CHORDS usable chords. A
twin pair is usable in a camera if both shots are, and both cameras still record at the lost shot's loss.

Writes RESULTS_DIR/sxr_audit/ (pairs.csv, summary.txt). Outside every stamp.
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

from scripts import programme_signals, sxr_fetch, v3_runs
from src import config, data, notify, relay

PAIRS = 600  # the sample of twin pairs, spread over the twin list
COVER, SAT, MIN_CHORDS = 0.95, 0.01, 12
SNRS = (3, 5, 10)  # reported at each; 5 is the reference
WORKFLOW, ARTIFACT = "fetch_sxr.yml", "mast-sxr"
EXAMPLE_CAMPAIGNS = ("M5", "M6")  # examples come from the development campaigns only
N_EXAMPLES = 3


def folder():
    return os.path.join(config.RESULTS_DIR, "sxr_audit")


def usable_cameras(table, snr=5):
    """{shot: set of cameras with at least MIN_CHORDS usable chords} and {shot: {camera: last recorded time}}."""
    t = table[table["camera"].astype(str) != ""].copy()
    t["ok"] = (t["covered"] >= COVER) & (t["snr"] >= snr) & (t["saturated"].fillna(1) <= SAT)
    per = t.groupby(["shot_id", "camera"]).agg(n=("ok", "sum"), last=("t_last", "max")).reset_index()
    good = per[per["n"] >= MIN_CHORDS]
    cams = good.groupby("shot_id")["camera"].agg(set).to_dict()
    last = {s: dict(zip(g["camera"], g["last"])) for s, g in per.groupby("shot_id")}
    return cams, last


def pair_table(pairs, table, t_loss, snr=5):
    """One row per sampled pair: whether each shot has soft X-ray data at all, is usable, and whether a camera is
    usable in both and still recording at the loss."""
    cams, last = usable_cameras(table, snr)
    present = set(table.loc[table["camera"].astype(str) != "", "shot_id"])
    rows = []
    for p in pairs.itertuples():
        tl = t_loss.get(p.shot_id, np.nan)
        shared = cams.get(p.shot_id, set()) & cams.get(p.other, set())
        at_loss = [c for c in shared if np.isfinite(tl) and last[p.shot_id].get(c, -1) >= tl and last[p.other].get(c, -1) >= tl]
        rows.append({"shot_id": p.shot_id, "other": p.other, "lost_has": p.shot_id in present, "kept_has": p.other in present,
                     "lost_ok": bool(cams.get(p.shot_id)), "kept_ok": bool(cams.get(p.other)),
                     "shared": len(shared) > 0, "usable": len(at_loss) > 0, "cameras": ",".join(sorted(at_loss))})
    return pd.DataFrame(rows)


LANDMARK_MS = 50


def landmark_counts(pairs, table, t_loss, snr=5, lead_ms=LANDMARK_MS):
    """Exploratory E1 (protocol of the soft X-ray study): (lost only, twin only, both, neither) usable, where a chord
    must record from the plasma start to lead_ms before the lost shot's loss, the interval the comparison at that
    lead needs, instead of covering the shot's whole plasma. SNR and saturation as in the audit."""
    t = table[table["camera"].astype(str) != ""]
    t = t[(t["snr"] >= snr) & (t["saturated"].fillna(1) <= SAT)]
    by_shot = {s: g for s, g in t.groupby("shot_id")}

    def ok(shot, t_end):
        g = by_shot.get(shot)
        if g is None:
            return False
        start = g["t_on"].where(np.isfinite(g["t_on"]), 0.0)
        good = g[(g["t_first"] <= start + 1e-3) & (g["t_last"] >= t_end)]
        return bool((good.groupby("camera").size() >= MIN_CHORDS).any())

    counts = [0, 0, 0, 0]
    for p in pairs.itertuples():
        tl = t_loss.get(p.shot_id, np.nan)
        if not np.isfinite(tl):
            continue
        a, b = ok(p.shot_id, tl - lead_ms / 1000), ok(p.other, tl - lead_ms / 1000)
        counts[0 if a and not b else 1 if b and not a else 2 if a and b else 3] += 1
    return tuple(counts)


def text(sampled, table, t_loss, campaign):
    n = len(sampled)
    lines = [f"Soft X-ray audit (faster signals, step 1). {n} twin pairs sampled over the twin list; "
             f"{table['shot_id'].nunique():,} shots read. Chord usable: covers >= {COVER:.0%} of the plasma, "
             f"saturated <= {SAT:.0%}, SNR at least the stated value; a camera needs >= {MIN_CHORDS} usable chords.", ""]
    for snr in SNRS:
        pt = pair_table(sampled, table, t_loss, snr)
        lines.append(f"SNR >= {snr}: usable pairs {int(pt['usable'].sum())} of {n} ({pt['usable'].mean():.0%}); "
                     f"a camera usable in both {pt['shared'].mean():.0%}; lost shot usable {pt['lost_ok'].mean():.0%}, "
                     f"twin usable {pt['kept_ok'].mean():.0%}")
    pt = pair_table(sampled, table, t_loss, 5)
    pt["campaign"] = pt["shot_id"].map(campaign).fillna("Unknown")
    lines += ["", "At SNR >= 5, by campaign (pairs; usable; any soft X-ray group in both):"]
    for c, g in pt.groupby("campaign"):
        lines.append(f"- {c}: {len(g)} pairs; usable {g['usable'].mean():.0%}; group in both {(g['lost_has'] & g['kept_has']).mean():.0%}")
    both_ok = pt["lost_ok"] & pt["kept_ok"]
    lines += ["", "Does missing data tell the lost shot from its twin? (pairs where exactly one shot is usable)",
              f"- lost usable, twin not: {int((pt['lost_ok'] & ~pt['kept_ok']).sum())}; twin usable, lost not: "
              f"{int((~pt['lost_ok'] & pt['kept_ok']).sum())}; both usable: {int(both_ok.sum())}; neither: "
              f"{int((~pt['lost_ok'] & ~pt['kept_ok']).sum())}"]
    lost_by_cam_end = sum(1 for p in pt.itertuples() if p.shared and not p.usable)
    lines.append(f"- usable in both but the camera stopped before the loss: {lost_by_cam_end} pairs")
    e1 = landmark_counts(sampled, table, t_loss)
    lines.append(f"- E1 (exploratory): with coverage only up to {LANDMARK_MS} ms before the lost shot's loss, lost usable "
                 f"and twin not: {e1[0]}; twin usable and lost not: {e1[1]}; both: {e1[2]}; neither: {e1[3]}")
    t = table[table["camera"].astype(str) != ""]
    lines += ["", "Cameras: " + "; ".join(f"{c} in {g['shot_id'].nunique()} shots, median SNR {g['snr'].median():.0f}, "
                                          f"saturated chords {(g['saturated'] > SAT).mean():.0%}"
                                          for c, g in t.groupby("camera"))]
    return "\n".join(lines), pt


def sample_pairs():
    return programme_signals.sample(pd.read_csv(v3_runs.TWINS), PAIRS)


def pick_examples(pairs, t_loss, campaign, n=N_EXAMPLES):
    """A few twin pairs to look at: development campaigns only, operator-recorded losses first, spread in shot order."""
    p = pairs[pairs["shot_id"].map(campaign).isin(EXAMPLE_CAMPAIGNS) & pairs["shot_id"].map(t_loss).notna()].copy()
    if "group" in p:
        p = p.sort_values("group", key=lambda g: g.ne("recorded"), kind="stable")
        recorded = p[p["group"] == "recorded"]
        p = recorded if len(recorded) >= n else p
    p = p.sort_values("shot_id")
    if len(p) > n:
        p = p.iloc[np.linspace(0, len(p) - 1, n).round().astype(int)]
    return [(int(r.shot_id), int(r.other), float(t_loss[r.shot_id])) for r in p.itertuples()]


def encode_examples(examples):
    return ";".join(f"{a}:{b}:{t * 1000:.1f}" for a, b, t in examples)


def trigger(ids, examples=()):
    r = requests.post(relay._api(f"/actions/workflows/{WORKFLOW}/dispatches"), headers=relay._headers(), timeout=30,
                      json={"ref": config.RELAY_BRANCH, "inputs": {"ids": sxr_fetch.encode_ids(ids),
                                                                   "examples": encode_examples(examples)}})
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
    if not parts:
        raise RuntimeError(f"run {run_id} has no {ARTIFACT} artifacts")
    tables, manifests = [], []
    for a in parts:
        z = requests.get(a["archive_download_url"], headers=relay._headers(), timeout=600)
        relay.check(z)
        with zipfile.ZipFile(io.BytesIO(z.content)) as archive:
            names = {os.path.basename(n): n for n in archive.namelist()}
            tables.append(pd.read_parquet(io.BytesIO(archive.read(names[sxr_fetch.TABLE]))))
            manifests.append(json.loads(archive.read(names["manifest.json"])))
            for name in archive.namelist():  # the example videos and pictures, if this part made any
                if name.endswith((".mp4", ".png")) and "/" not in name.replace("examples/", "", 1):
                    os.makedirs(os.path.join(folder(), "examples"), exist_ok=True)
                    with open(os.path.join(folder(), "examples", os.path.basename(name)), "wb") as f:
                        f.write(archive.read(name))
    os.makedirs(sxr_fetch.folder(), exist_ok=True)
    pd.concat(tables, ignore_index=True).to_parquet(os.path.join(sxr_fetch.folder(), sxr_fetch.TABLE), index=False)
    m = {k: sum(x.get(k, 0) for x in manifests) for k in ("shots_requested", "shots_with_data", "absent", "failed", "not_started")}
    with open(os.path.join(sxr_fetch.folder(), "manifest.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m


def send_examples(say=print):
    """Send the example videos and pictures to Telegram, one caption each."""
    out = os.path.join(folder(), "examples")
    for name in sorted(os.listdir(out)) if os.path.isdir(out) else []:
        lost, kept = name.split(".")[0].split("_")[:2]
        kind = "slow motion, last 300 ms" if name.endswith(".mp4") else "chord against time"
        ok = notify.send_file(os.path.join(out, name), f"Soft X-ray camera, lost shot {lost} against its twin {kept} ({kind})")
        say(f"sent {name} to Telegram: {ok}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--relay", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    sampled = sample_pairs()
    try:
        if args.relay:
            if not relay.configured():
                raise RuntimeError("no GitHub token: GITHUB_TOKEN is not set (it lives in ~/mast-private/.env)")
            ids = sorted(set(sampled["shot_id"]) | set(sampled["other"]))
            labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
            camp = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
            examples = pick_examples(sampled, labels.set_index("shot_id")["t_quench"].to_dict(), camp)
            say(f"example pairs to film (M5/M6 only): {examples}")
            requested = trigger(ids, examples)
            say(f"asked GitHub to read {len(ids)} shots; waiting for the job to appear ...")
            run = None
            while run is None and time.time() < requested + 300:
                time.sleep(10)
                run = relay.pick_run(runs(), requested)
            if run is None:
                raise RuntimeError("GitHub did not list the job within five minutes")
            say(f"job started: {run.get('html_url', run['id'])}")
            programme_signals.wait(run, say)
            say(f"collected: {collect(run['id'])}")
            send_examples(say)
        elif args.collect:
            done = [r for r in runs() if r.get("status") == "completed"]
            if not done:
                raise RuntimeError("no finished fetch_sxr job yet")
            say(f"collected: {collect(max(done, key=lambda r: r['created_at'])['id'])}")
    except RuntimeError as error:
        say(f"STOPPED: {error}")
        sys.exit(1)
    path = os.path.join(sxr_fetch.folder(), sxr_fetch.TABLE)
    if not os.path.exists(path):
        say(f"no collected table at {path}. Run: python scripts/sxr_audit.py --relay")
        sys.exit(1)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
    out, pt = text(sampled, pd.read_parquet(path), labels.set_index("shot_id")["t_quench"].to_dict(), campaign)
    os.makedirs(folder(), exist_ok=True)
    pt.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
