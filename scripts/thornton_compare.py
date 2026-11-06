"""Our ending classes against an earlier, independent MAST disruption database: A. J. Thornton's (PhD thesis,
University of York, 2011, section 4.2.1), archived with its IDL code at https://github.com/spamela/disruption_detector
(Apache-2.0). Thornton selected 'full performance disruptions' with nearly our physical rule, written separately and
on the raw MAST data: the disruption falls in the current flat-top, and the central solenoid flux 'should not have
reached its operational limit or started to ramp in the opposite direction'. His flags, per shot:
- disruption: 1 a thermal quench was found (with a gradient limit and internal reconnection events excluded),
  0 none, -1 failed;
- ip_flattop_yn: 1 the current 40 to 10 samples before the quench is within 10% of its level 10 samples before;
- isol_constant: 1 the solenoid current had not started to change (smoothed slope above 100) before the quench.

    git clone https://github.com/spamela/disruption_detector
    tar xzf disruption_detector/last_version_thornton_2009/old_database_files.tar.gz -C /tmp/thornton
    python scripts/thornton_compare.py /tmp/thornton/disruption_efit_0709_full.sav

Reads RESULTS_DIR/auto_labels/per_shot.csv if it exists, else the public table in the repository. Prints the
agreement and writes nothing. An analysis next to the experiments: outside every stamp.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, evaluate

FIELDS = ["shots", "disruption", "ip_flattop_yn", "isol_constant", "ttq"]
PUBLIC_TABLE = os.path.join(config.REPO_DIR, "results", "auto_labels", "per_shot.csv")


def load_thornton(path):
    """Thornton's per-shot flags from an IDL save file of his database, one row per shot."""
    from scipy.io import readsav
    data = readsav(path, verbose=False)
    table = pd.DataFrame({f: np.asarray(data[f]).astype(float) for f in FIELDS})
    table["shot_id"] = table.pop("shots").astype(int)
    return table.drop_duplicates("shot_id").reset_index(drop=True)


def compare(ours, thornton):
    """Agreement on the shots both tables hold: quench found, quench time, solenoid state, and the share of each of
    our ending groups that Thornton's rule keeps as a full-performance disruption. Returns a dict."""
    m = ours.merge(thornton, on="shot_id", how="inner")
    quench = m["t_quench"].notna().to_numpy()
    th_disrupt = (m["disruption"] == 1).to_numpy()
    m["group"] = [evaluate.ending_group(n, e, q) for n, e, q in zip(m["note"], m["termination"], quench)]
    full_performance = th_disrupt & (m["ip_flattop_yn"] == 1).to_numpy() & (m["isol_constant"] == 1).to_numpy()
    both = th_disrupt & quench & (m["ttq"] > 0).to_numpy() & (m["ttq"] < 2).to_numpy()
    dt_ms = 1000 * (m.loc[both, "t_quench"] - m.loc[both, "ttq"])
    sol = quench & th_disrupt & m["termination"].isin(["flat_top_drop", "flux_exhausted"]).to_numpy()
    ours_left = (m.loc[sol, "termination"] == "flat_top_drop").to_numpy()
    th_left = (m.loc[sol, "isol_constant"] == 1).to_numpy()
    agree = float(np.mean(ours_left == th_left)) if sol.any() else np.nan
    expected = np.mean(ours_left) * np.mean(th_left) + (1 - np.mean(ours_left)) * (1 - np.mean(th_left))
    groups = {g: {"shots": int((m["group"] == g).sum()),
                  "full_performance": float(full_performance[(m["group"] == g).to_numpy()].mean())}
              for g in evaluate.ENDING_GROUPS if (m["group"] == g).any()}
    return {"shared_shots": len(m), "campaigns": m["campaign"].value_counts().to_dict(),
            "thornton_disruptions": int(th_disrupt.sum()), "our_quenches": int(quench.sum()),
            "both": int((th_disrupt & quench).sum()),
            "time_median_ms": float(np.median(dt_ms)) if len(dt_ms) else np.nan,
            "time_within_2ms": float(np.mean(np.abs(dt_ms) <= 2)) if len(dt_ms) else np.nan,
            "time_within_5ms": float(np.mean(np.abs(dt_ms) <= 5)) if len(dt_ms) else np.nan,
            "solenoid_shots": int(sol.sum()), "solenoid_agreement": agree,
            "solenoid_kappa": float((agree - expected) / (1 - expected)) if sol.any() and expected < 1 else np.nan,
            "groups": groups}


def text(r):
    lines = [f"Shared shots: {r['shared_shots']:,} {r['campaigns']}",
             f"Thornton finds a disruption in {r['thornton_disruptions']:,}, our detector a quench in "
             f"{r['our_quenches']:,}, both in {r['both']:,} ({r['both'] / r['thornton_disruptions']:.1%} of his, "
             f"{r['both'] / r['our_quenches']:.1%} of ours).",
             f"Quench time, ours minus his: median {r['time_median_ms']:.2f} ms; within 2 ms {r['time_within_2ms']:.0%}, "
             f"within 5 ms {r['time_within_5ms']:.0%}.",
             f"Solenoid state (our flat-top drop vs flux exhausted against his solenoid unchanged), "
             f"{r['solenoid_shots']:,} shots: agreement {r['solenoid_agreement']:.1%}, Cohen's kappa "
             f"{r['solenoid_kappa']:.2f}.",
             "Share of each of our ending groups that Thornton's rule keeps as a full-performance disruption:"]
    lines += [f"- {g}: {v['full_performance']:.0%} of {v['shots']:,}" for g, v in r["groups"].items()]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sav", help="Thornton's database, e.g. disruption_efit_0709_full.sav")
    parser.add_argument("--labels", help="our per-shot ending table (default: RESULTS_DIR's, else the public one)")
    args = parser.parse_args()
    private = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
    labels = args.labels or (private if os.path.exists(private) else PUBLIC_TABLE)
    print(text(compare(pd.read_csv(labels), load_thornton(args.sav))))


if __name__ == "__main__":
    main()
