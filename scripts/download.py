"""Download and cache raw summary signals for a random sample of shots (or explicit ids)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.data import check_connection, download_shots, load_shot_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="how many random shots to download")
    parser.add_argument("--ids", type=int, nargs="*", help="explicit shot ids (overrides --n)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--workers", type=int, default=config.DOWNLOAD_WORKERS)
    parser.add_argument("--skip-check", action="store_true", help="do not test the connection first")
    args = parser.parse_args()

    if not args.skip_check:
        ok, message = check_connection()
        print(message)
        if not ok:
            sys.exit("stopping: the data server cannot be read from here (already-cached shots are still usable)")

    if args.ids:
        ids = args.ids
    else:
        table = load_shot_table()
        rng = np.random.default_rng(args.seed)
        ids = [int(i) for i in rng.choice(table["shot_id"].to_numpy(), size=args.n, replace=False)]

    result = download_shots(ids, workers=args.workers)
    print(f"requested {len(ids)}: {len(result['cached'])} already cached, "
          f"{len(result['downloaded'])} downloaded, {len(result['failed'])} failed")
    if result["failed"]:
        print(f"failures are listed in {config.SKIPPED_CSV}")


if __name__ == "__main__":
    main()
