"""How do quenches the operators did not note differ from noted disruptions? Uses the programmed plasma current
(pulse_schedule) and the central solenoid current (pf_active 'SOL') at the detector's quench time."""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.data import open_group

OUT_DIR = os.path.join(config.RESULTS_DIR, "termination_study")
N_PERMUTATIONS = 10000  # enough for p-values down to ~1e-4


def value_at(t, x, t_q):
    """Last finite sample of x at or before t_q, or NaN."""
    ok = np.isfinite(x) & (t <= t_q)
    return float(x[ok][-1]) if ok.any() else np.nan


def measure(shot_id, t_q):
    """Programmed current (as a fraction of its maximum) and solenoid current (A) at the quench time."""
    row = {"shot_id": shot_id}
    try:
        sched = open_group(shot_id, "pulse_schedule")
        ref = np.abs(sched["i_plasma"].values)
        row["ref_frac_at_q"] = value_at(sched["time"].values, ref, t_q) / np.nanmax(ref)
        pf = open_group(shot_id, "pf_active")
        channels = [str(c) for c in pf["current_channel"].values]
        sol = pf["coil_current"].values[channels.index("SOL")]
        row["sol_at_q"] = value_at(pf["time"].values, sol, t_q)
        row["sol_min"] = float(np.nanmin(sol))
    except Exception as error:
        row["error"] = f"{type(error).__name__}: {error}"[:120]
    return row


def permutation_auc(y, score, n=N_PERMUTATIONS, seed=config.RANDOM_SEED):
    """AUC of score for y, and the share of label shuffles that reach an AUC at least that far from 0.5."""
    rng = np.random.default_rng(seed)
    auc = roc_auc_score(y, score)
    null = np.array([roc_auc_score(rng.permutation(y), score) for _ in range(n)])
    p = (np.sum(np.abs(null - 0.5) >= abs(auc - 0.5)) + 1) / (n + 1)
    return float(auc), float(p)


def plot_sol(q, path):
    """Histogram of solenoid current at the quench, noted vs unnoted."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bins = np.linspace(q["sol_at_q"].min() / 1e3, q["sol_at_q"].max() / 1e3, 30)
    for name, group in q.groupby("group"):
        ax.hist(group["sol_at_q"] / 1e3, bins=bins, alpha=0.55, label=f"{name} ({len(group)})")
    ax.set_xlabel("central solenoid current at the quench (kA)")
    ax.set_ylabel("shots")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def report(q, errors):
    """Markdown report; every number comes from q."""
    y = (q["group"] == "unnoted").astype(int).to_numpy()
    auc, p = permutation_auc(y, -q["sol_at_q"].to_numpy())
    lines = ["# Termination study", "",
             f"Quench shots from `results/label_report/per_shot.csv`: {len(q)} measured, {errors} failed to load. "
             "Times are the detector's quench times.", "",
             "## Programmed current at the quench", "",
             "| group | shots | median ref fraction | share >= 0.9 |", "|---|---|---|---|"]
    for name, g in q.groupby("group"):
        lines.append(f"| {name} | {len(g)} | {g['ref_frac_at_q'].median():.3f} | {(g['ref_frac_at_q'] >= 0.9).mean():.3f} |")
    lines += ["", "## Central solenoid current at the quench", "",
              "| group | median (kA) | 25% (kA) | 75% (kA) |", "|---|---|---|---|"]
    for name, g in q.groupby("group"):
        s = g["sol_at_q"] / 1e3
        lines.append(f"| {name} | {s.median():.1f} | {s.quantile(.25):.1f} | {s.quantile(.75):.1f} |")
    lines += ["", f"AUC for 'more negative solenoid current means unnoted': {auc:.3f} "
              f"(permutation p = {p:.4f}, {N_PERMUTATIONS} shuffles). Most negative solenoid current seen: "
              f"{q['sol_min'].min() / 1e3:.1f} kA.", "", "Per campaign:", "", "| campaign | shots | AUC |", "|---|---|---|"]
    for campaign, g in q.groupby("campaign"):
        if g["group"].nunique() == 2:
            lines.append(f"| {campaign} | {len(g)} | {roc_auc_score((g['group'] == 'unnoted').astype(int), -g['sol_at_q']):.2f} |")
    lines += ["", "Plot: `solenoid_at_quench.png`. Per-shot values: `per_shot.csv`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-shot", default=os.path.join("results", "label_report", "per_shot.csv"))
    args = parser.parse_args()
    shots = pd.read_csv(args.per_shot)
    shots = shots[shots["signal"] == "disrupted"]
    with ThreadPoolExecutor(config.DOWNLOAD_WORKERS) as pool:
        rows = list(pool.map(measure, shots["shot_id"].astype(int), shots["t_disrupt"]))
    q = shots.merge(pd.DataFrame(rows), on="shot_id")
    errors = int(q["error"].notna().sum()) if "error" in q else 0
    q = q.dropna(subset=["sol_at_q", "ref_frac_at_q"]).copy()
    q["group"] = np.where(q["note"], "noted", "unnoted")
    os.makedirs(OUT_DIR, exist_ok=True)
    q.drop(columns=["comment"]).to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)
    plot_sol(q, os.path.join(OUT_DIR, "solenoid_at_quench.png"))
    text = report(q, errors)
    with open(os.path.join(OUT_DIR, "report.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
