"""Runs on a GitHub Actions machine, which can reach STFC: download the shots a run needs into MAST_HOME/data
and write data/manifest.json. The workflow then hands that folder to the owner's server (src/relay.py)."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import shot_counts, table_from_report
from src import config, dataset
from src.data import download_shots, load_shot_table



def choose_shots(scale):
    """(shot ids, table source). Uses the live shot table if mastapp.site answers, else the Phase 2 report."""
    try:
        table = load_shot_table()
    except Exception as error:
        print(f"shot table unavailable ({type(error).__name__}); using the 400 report shots instead")
        return [int(i) for i in table_from_report(config.REPORT_CSV)["shot_id"]], "report"
    return dataset.candidate_ids(table, shot_counts(scale)[1]), "live"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="full")
    args = parser.parse_args()
    ids, source = choose_shots(args.scale)
    result = download_shots(ids)
    manifest = {"scale": args.scale, "table_source": source, "requested": len(ids),
                "available": len(result["cached"]) + len(result["downloaded"]), "failed": len(result["failed"]),
                "commit": os.environ.get("GITHUB_SHA", "local")[:7],
                "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DATA_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(manifest)


if __name__ == "__main__":
    main()
