"""Build the window table with features, labels, and splits (SPEC 8). Writes data/processed/."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import config, data, dataset, extras, resume
from src.data import check_connection, download_shots, is_cached, load_shot_table


def table_from_report(path):
    """A stand-in shot table built from the Phase 2 report (results/label_report/per_shot.csv), for when
    mastapp.site is down. Its comments are cut at 90 characters, which changes the parsed label of 8 shots, so the
    comment is rebuilt from the stored labels (computed from the full comments): 'disruption at <t_note> s',
    'disrupted' (no time), or empty."""
    report = pd.read_csv(path)
    comments = []
    for note, t_note in zip(report["note"], report["t_note"]):
        if not note:
            comments.append("")
        elif np.isfinite(t_note):
            comments.append(f"disruption at {t_note} s")
        else:
            comments.append("disrupted")
    return pd.DataFrame({"shot_id": report["shot_id"], "campaign": report["campaign"],
                         "shot_postshot_comment": comments})


def shot_counts(scale):
    """(n_disrupted, n_clean) for a scale: 'small' is the balanced SPEC subset, 'full' every disruption plus
    FULL_N_CLEAN clean shots (None means all)."""
    if scale == "small":
        return config.N_DISRUPTED, config.N_CLEAN
    return None, config.FULL_N_CLEAN


def build(scale="small", skip_check=False, table=None, offline=False):
    """Download, label, select, window and split. Returns a dict of counts for the log / Telegram.

    table: use this shot table and every shot in it (e.g. table_from_report) instead of the live one.
    offline: no network at all; use only shots already on disk (e.g. brought by the GitHub relay).
    """
    n_disrupted, n_clean = shot_counts(scale)
    if not skip_check and not offline:
        ok, message = check_connection(attempts=config.CONNECTION_CHECK_ATTEMPTS)
        if not ok:
            raise RuntimeError(message + "\nRun: python scripts/diagnose_network.py")

    from_live_table = table is None
    if table is None:
        table = load_shot_table()
        ids = dataset.candidate_ids(table, n_clean)
    else:
        ids = [int(i) for i in table["shot_id"]]
        n_disrupted = n_clean = None
    if offline:
        failed = {i for i in ids if not is_cached(i)}
    else:
        failed = set(download_shots(ids)["failed"])
    labelled = dataset.label_shots([i for i in ids if i not in failed], table)
    shots = dataset.select_shots(labelled, n_disrupted, n_clean)
    if config.USE_EXTRAS and not offline:
        extras.download_extras([int(i) for i in shots["shot_id"]])

    current = build_stamp(shots, scale)
    saved = resume.load(config.PROCESSED_DIR, current)
    if saved is not None and all(os.path.exists(p) for p in resume.processed_files()):
        return {**saved, "reused": True}
    shots["split"] = shots["shot_id"].map(dataset.random_split(shots))
    shots["split_temporal"] = shots["shot_id"].map(dataset.temporal_split(shots))
    windows = dataset.build_windows(shots)
    windows = windows.merge(shots[["shot_id", "split", "split_temporal"]], on="shot_id")
    shots = shots[shots["shot_id"].isin(windows["shot_id"])].reset_index(drop=True)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    windows.to_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"))
    shots.to_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    counts = {"candidates": len(ids), "download_failed": len(failed), "labelled": len(labelled),
              "shots": len(shots), "disrupted": int(shots["disrupted"].sum()),
              "clean": int((~shots["disrupted"]).sum()), "windows": len(windows),
              "split_counts": shots["split"].value_counts().to_dict(),
              "table_source": "live" if from_live_table else "report"}
    resume.save(config.PROCESSED_DIR, current, counts)
    return counts


def build_stamp(shots, scale):
    """Fingerprint of everything the windows are built from: science code, the selected shots' cached signal
    and extras files, their labels under every source, and the scale. Same stamp, same dataset."""
    files = []
    for shot_id in shots["shot_id"]:
        for path in (data.cache_path(int(shot_id)), extras.cache_path(int(shot_id))):
            if os.path.exists(path):
                files.append(path)
    return resume.stamp(files, extra=f"{scale}\n{shots.to_csv(index=False)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="small",
                        help="small: N_DISRUPTED + N_CLEAN shots; full: every disruption + FULL_N_CLEAN clean")
    parser.add_argument("--skip-check", action="store_true", help="do not test the connection first")
    parser.add_argument("--from-report", help="use the shots and labels of a Phase 2 per_shot.csv (mastapp.site down)")
    parser.add_argument("--offline", action="store_true", help="no network: use only shots already on disk")
    args = parser.parse_args()
    table = table_from_report(args.from_report) if args.from_report else None
    print(build(args.scale, args.skip_check, table, args.offline))


if __name__ == "__main__":
    main()
