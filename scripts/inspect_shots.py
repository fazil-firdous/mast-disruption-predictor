"""Inspect random shots spread across campaigns: time step, time range, sign of ip, variables present."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import config
from src.data import download_shots, load_shot, load_shot_table


def pick_shots(table, n, seed):
    """Pick about n shots, spread as evenly as possible over the campaigns in the table."""
    rng = np.random.default_rng(seed)
    campaigns = sorted(table["campaign"].dropna().unique())
    per_campaign = max(1, n // len(campaigns))
    chosen = []
    for campaign in campaigns:
        ids = table.loc[table["campaign"] == campaign, "shot_id"].to_numpy()
        take = min(per_campaign, len(ids))
        chosen += [(int(i), campaign) for i in rng.choice(ids, size=take, replace=False)]
    return chosen


def describe(shot_id, campaign):
    """One summary row for a cached shot."""
    df = load_shot(shot_id)
    dt = np.diff(df["time"].to_numpy())
    present = [c for c in config.SIGNAL_VARIABLES if df[c].notna().any()]
    ip = df["ip"].to_numpy()
    peak = ip[np.nanargmax(np.abs(ip))] if np.isfinite(ip).any() else np.nan
    return {
        "shot_id": shot_id, "campaign": campaign, "n_samples": len(df),
        "t_min": df["time"].min(), "t_max": df["time"].max(),
        "dt_median_ms": 1000 * np.median(dt), "dt_uniform": bool(np.allclose(dt, np.median(dt), rtol=1e-3)),
        "ip_sign_at_peak": int(np.sign(peak)) if np.isfinite(peak) else 0,
        "max_abs_ip": np.nanmax(np.abs(ip)) if np.isfinite(ip).any() else np.nan,
        "missing": ",".join(c for c in config.SIGNAL_VARIABLES if c not in present),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--cached-only", action="store_true", help="only pick shots already in data/raw")
    args = parser.parse_args()

    table = load_shot_table()
    if args.cached_only:
        cached = {int(f.split(".")[0]) for f in os.listdir(config.RAW_DIR) if f.endswith(".parquet")}
        table = table[table["shot_id"].isin(cached)]
    picks = pick_shots(table, args.n, args.seed)
    result = download_shots([s for s, _ in picks])
    rows = [describe(s, c) for s, c in picks if s not in result["failed"]]
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))
    print(f"\nfailed to load: {result['failed']}")


if __name__ == "__main__":
    main()
