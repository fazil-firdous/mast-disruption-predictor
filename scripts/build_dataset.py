"""Build the window table with features, labels, and splits (SPEC 8). Writes data/processed/."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, dataset
from src.data import check_connection, download_shots, load_shot_table


def build(scale="small", skip_check=False):
    """Download, label, select, window and split. Returns a dict of counts for the log / Telegram."""
    n_disrupted = config.N_DISRUPTED if scale == "small" else None
    n_clean = config.N_CLEAN if scale == "small" else config.FULL_N_CLEAN
    if not skip_check:
        ok, message = check_connection()
        if not ok:
            raise RuntimeError(message)

    table = load_shot_table()
    ids = dataset.candidate_ids(table, n_clean)
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
    args = parser.parse_args()
    print(build(args.scale, args.skip_check))


if __name__ == "__main__":
    main()
