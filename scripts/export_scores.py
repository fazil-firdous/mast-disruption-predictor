"""Package the held-out window scores behind the twin paper, so that anyone can recompute its tables without our
models: for every model and split, the test-window scores of the discharges that appear in a twin pair or a placebo
triplet; the pair-level results (scripts/paper2_corrected.py); the twin list; a manifest with checksums.

    python scripts/export_scores.py

Writes RESULTS_DIR/export_scores/ and RESULTS_DIR/export_scores.zip. Reads saved runs only; outside every stamp.
"""
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, paper2_corrected as p2, saddle_correction as sc, v3_explore, v3_runs
from src import config, notify, server

MODELS = {"programme": "programmed current + elapsed time", "solenoid": "solenoid current + elapsed time",
          "machine": "machine-side: both currents + elapsed time", "baseline": "full model, original saddle inputs (A)",
          sc.RECIPE: "full model, polarity-corrected saddle inputs (A_c)"}
README = """# Held-out window scores behind "A matched repeat-shot evaluation of disruption predictors: a MAST case study"

Derived from the FAIR-MAST level-2 archive (UKAEA, CC BY-SA 4.0); released under CC BY-SA 4.0.

Files
- `scores.parquet`: one row per test window of a discharge that appears in a twin pair or a placebo triplet:
  `model`, `split` (1-50 random splits; 0 = chronological split, trained on M5-M7, tested on M8-M9), `shot_id`,
  `t_end` (window end, s, archive time axis), `score` (model output, float32). Windows are 20 ms long, every 5 ms.
- `pairs.parquet`: the pair-level results used for every table (one row per model, split, pair and lead: `win`
  for the registered statistic, `win_common` for the same-time statistic, `win_end` for the last shared window, and
  the matching flags).
- `placebo.parquet`: the same for the placebo pairs (kept repeat K1 against K2).
- `programme_twins.csv`: the twin list (case `shot_id`, twin `other`).
- `manifest.json`: models, counts, code commit and SHA-256 of every file.

Models: see `manifest.json`. Scorer and protocols: https://github.com/fazil-firdous/mast-disruption-predictor
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wanted_shots():
    twins = pd.read_csv(v3_runs.TWINS)
    shots = set(twins["shot_id"]) | set(twins["other"])
    triplets = os.path.join(config.RESULTS_DIR, "twin_differences", "triplets.csv")
    if os.path.exists(triplets):
        t = pd.read_csv(triplets)
        shots |= set(t["shot_id"]) | set(t["k1"]) | set(t["k2"])
    return {int(s) for s in shots}


def run_paths():
    """[(model, split, folder)] of every saved run: 50 random splits and the chronological split."""
    folders = p2.folders()
    out = []
    for model in MODELS:
        for seed, path in how_early.saved_runs(folders[model]):
            out.append((model, seed, path))
        temporal = sc.run_dir(None) if model == sc.RECIPE else v3_explore.temporal_dir(model)
        if server.result_saved(os.path.join(temporal, "metrics.json")):
            out.append((model, 0, temporal))
    return out


def export(out_dir, say=print):
    os.makedirs(out_dir, exist_ok=True)
    shots = wanted_shots()
    frames, counts = [], {}
    for model, split, path in run_paths():
        s = pd.read_parquet(os.path.join(path, "test_scores.parquet"))
        s = s[s["shot_id"].isin(shots)]
        frames.append(s.assign(model=model, split=split)[["model", "split", "shot_id", "t_end", "score"]])
        counts.setdefault(model, []).append(split)
    scores = pd.concat(frames, ignore_index=True).astype({"split": "int16", "shot_id": "int32", "score": "float32"})
    scores.to_parquet(os.path.join(out_dir, "scores.parquet"), index=False)
    for name, src in (("pairs.parquet", "paper2_pairs.parquet"), ("placebo.parquet", "paper2_placebo.parquet")):
        if os.path.exists(os.path.join(sc.folder(), src)):
            shutil.copy2(os.path.join(sc.folder(), src), os.path.join(out_dir, name))
    shutil.copy2(v3_runs.TWINS, os.path.join(out_dir, "programme_twins.csv"))
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(README)
    commit = os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip()
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "code_commit": commit,
                "models": MODELS, "splits_per_model": {m: sorted(v) for m, v in counts.items()},
                "rows": int(len(scores)), "discharges": int(scores["shot_id"].nunique()),
                "sha256": {f: sha256(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))
                           if f != "manifest.json"}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    archive = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(out_dir)):
            z.write(os.path.join(out_dir, name), name)
    size = os.path.getsize(archive) / 1e6
    text = (f"Scores exported: {manifest['rows']:,} rows, {manifest['discharges']:,} discharges, "
            f"{sum(len(v) for v in counts.values())} runs; {archive} ({size:.0f} MB)")
    say(text)
    return text


def main():
    text = export(os.path.join(config.RESULTS_DIR, "export_scores"))
    notify.send(text)


if __name__ == "__main__":
    main()
