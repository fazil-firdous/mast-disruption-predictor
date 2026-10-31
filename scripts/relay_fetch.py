"""Runs on a GitHub Actions machine, which can reach STFC: download the shots a run needs (or one share of them,
--part of --parts) into MAST_HOME/data and write data/manifest.json. The workflow then hands that folder to the
research server (src/relay.py), which merges the parts."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_dataset import shot_counts, table_from_report
from src import config, dataset, extras, sources
from src.data import download_shots, load_shot_table

START = time.time()



def choose_shots(scale):
    """(shot ids, table source). Uses the full shot table (mastapp.site or its S3 copy) if either answers, else the
    Phase 2 report."""
    try:
        table = load_shot_table()
    except Exception as error:
        print(f"shot table unavailable ({type(error).__name__}); using the 400 report shots instead")
        return [int(i) for i in table_from_report(config.REPORT_CSV)["shot_id"]], "report"
    return dataset.candidate_ids(table, shot_counts(scale)[1]), "live"


def write_json(obj, name):
    """Write obj to DATA_DIR/name (the folder the workflow hands over)."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(os.path.join(config.DATA_DIR, name), "w") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="full")
    parser.add_argument("--task", choices=["fetch", "probe"], default="fetch",
                        help="fetch: download shots; probe: only report what the data sources offer (src/sources.py)")
    parser.add_argument("--part", type=int, default=0, help="which share of the shots this job fetches (0-based)")
    parser.add_argument("--parts", type=int, default=1, help="how many jobs share the shots (config.RELAY_PARTS)")
    args = parser.parse_args()
    if args.task == "probe":
        snapshot = sources.probe()
        write_json(snapshot, "probe.json")
        write_json({"task": "probe", "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}, "manifest.json")
        print(snapshot)
        return
    ids, source = choose_shots(args.scale)
    ids = ids[args.part::args.parts]
    result = download_shots(ids, workers=config.RELAY_WORKERS)
    have = result["cached"] + result["downloaded"]
    # the server cannot compute extras itself (it cannot reach STFC), so do it here for every shot it may select
    extra = {"ok": 0, "failed": []}
    if config.USE_EXTRAS:
        extra = extras.download_extras(have, workers=config.RELAY_WORKERS, deadline=START + config.RELAY_WORK_HOURS * 3600)
    manifest = {"task": "fetch", "scale": args.scale, "table_source": source, "part": args.part, "parts": args.parts,
                "full_n_clean": config.FULL_N_CLEAN, "extras_columns": list(extras.COLUMNS), "requested": len(ids),
                "available": len(result["cached"]) + len(result["downloaded"]), "failed": len(result["failed"]),
                "extras": extra["ok"], "extras_failed": len(extra["failed"]),
                "commit": os.environ.get("GITHUB_SHA", "local")[:7],
                "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())}
    write_json(manifest, "manifest.json")
    print(manifest)


if __name__ == "__main__":
    main()
