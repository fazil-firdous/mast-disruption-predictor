"""Build the window table with features, labels, and splits (SPEC 8). Writes data/processed/."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import config, dataset
from src.data import check_connection, download_shots, load_shot_table


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


def build(scale="small", skip_check=False, table=None):
    """Download, label, select, window and split. Returns a dict of counts for the log / Telegram.

    table: use this shot table and every shot in it (e.g. table_from_report) instead of the live one.
    """
    n_disrupted = config.N_DISRUPTED if scale == "small" else None
    n_clean = config.N_CLEAN if scale == "small" else config.FULL_N_CLEAN
    if not skip_check:
        ok, message = check_connection(attempts=config.CONNECTION_CHECK_ATTEMPTS)
        if not ok:
            raise RuntimeError(message + "\nRun: python scripts/diagnose_network.py")

    if table is None:
        table = load_shot_table()
        ids = dataset.candidate_ids(table, n_clean)
    else:
        ids = [int(i) for i in table["shot_id"]]
        n_disrupted = n_clean = None
    failed = set(download_shots(ids)["failed"])
    labelled = dataset.label_shots([i for i in ids if i not in failed], table)
    shots = dataset.select_shots(labelled, n_disrupted, n_clean)

    shots["split"] = shots["shot_id"].map(dataset.random_split(shots))
    shots["split_temporal"] = shots["shot_id"].map(dataset.temporal_split(shots))
    windows = dataset.build_windows(shots)
    windows = windows.merge(shots[["shot_id", "split", "split_temporal"]], on="shot_id")
    shots = shots[shots["shot_id"].isin(windows["shot_id"])].reset_index(drop=True)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    windows.to_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"))
    shots.to_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    return {"candidates": len(ids), "download_failed": len(failed), "labelled": len(labelled),
            "shots": len(shots), "disrupted": int(shots["disrupted"].sum()),
            "clean": int((~shots["disrupted"]).sum()), "windows": len(windows),
            "split_counts": shots["split"].value_counts().to_dict()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="small",
                        help="small: N_DISRUPTED + N_CLEAN shots; full: every disruption + FULL_N_CLEAN clean")
    parser.add_argument("--skip-check", action="store_true", help="do not test the connection first")
    parser.add_argument("--from-report", help="use the shots and labels of a Phase 2 per_shot.csv (mastapp.site down)")
    args = parser.parse_args()
    table = table_from_report(args.from_report) if args.from_report else None
    print(build(args.scale, args.skip_check, table))


if __name__ == "__main__":
    main()
