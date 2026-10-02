"""Phase 3 check: compute window features on real shots, test causality on them, and plot shot 11860."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, features, labels
from src.data import download_shots, load_shot, load_shot_table

OUT_DIR = os.path.join(config.RESULTS_DIR, "features")


def label_and_windows(shot_id, comment):
    """Label one shot (LABEL_SOURCE) and build its windows. Disrupted shots stop at t_disrupt (SPEC 6)."""
    shot = load_shot(shot_id)
    lab = labels.shot_label(shot["time"], shot["ip"], comment)
    t_stop = lab["t_disrupt"] if lab["status"] == "disrupted" else np.inf
    win = features.shot_windows(shot, t_stop=t_stop)
    win.insert(0, "shot_id", shot_id)
    win["status"] = lab["status"]
    return shot, lab, win


def truncation_check(shot, win, rng):
    """Cut the shot at a random window end, recompute, and compare the rows up to the cut.

    Returns the number of rows compared, or 0 if the shot has too few windows. Raises if any value differs.
    """
    if len(win) < 3:
        return 0
    t_cut = float(rng.choice(win["t_end"].to_numpy()[1:-1]))
    short = shot[shot["time"] <= t_cut + features.EPS]
    t_stop = win.attrs.get("t_stop", np.inf)
    again = features.shot_windows(short, t_stop=t_stop)
    full = win[win["t_end"] <= t_cut + features.EPS].drop(columns=["shot_id", "status"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(full, again.reset_index(drop=True), check_exact=True)
    return len(full)


def plot_reference(shot, lab, win, path):
    """|Ip| with t_disrupt, and four features over time, on shared time axes."""
    panels = [("ip_abs_last", "|Ip| at t_end (kA)", 1e-3), ("ip_abs_slope", "|Ip| slope (MA/s)", 1e-6),
              ("ip_frac_of_running_max", "|Ip| / running max", 1), ("prad_over_pnbi", "Prad / (Pnbi + 0.1 MW)", 1),
              ("neutrons_mean", "neutron rate (1e13 Hz)", 1e-13)]
    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(8, 11), sharex=True)
    axes[0].plot(shot["time"], np.abs(shot["ip"]) / 1e3, color="0.4", lw=1)
    axes[0].set_ylabel("|Ip| raw (kA)", fontsize=8)
    for ax, (col, name, scale) in zip(axes[1:], panels):
        ax.plot(win["t_end"], win[col] * scale, ".-", ms=3, lw=1)
        ax.set_ylabel(name, fontsize=8)
    for ax in axes:
        if lab["disrupted"]:
            ax.axvline(lab["t_disrupt"], color="red", lw=1.5)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("time (s); dots are window ends; red line = t_disrupt", fontsize=8)
    fig.suptitle(f"Shot {config.REFERENCE_SHOT}: window features (window {config.WINDOW_MS} ms, "
                 f"stride {config.STRIDE_MS} ms)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=75)
    plt.close(fig)


def summary_text(all_win, info, compared):
    """Markdown summary; every number comes from the arguments."""
    names = features.feature_names()
    values = all_win[names].to_numpy(dtype=float)
    nan_share = all_win[names].isna().mean()
    per_shot = info.groupby("status")["n_windows"].agg(["count", "median", "min", "max"])
    lines = [
        "# Phase 3 feature check", "",
        f"Shots: {len(info)}. Windows: {len(all_win)}. Features per window: {len(names)}. "
        f"Infinite values: {int(np.isinf(values).sum())}. Shots with no windows: {(info['n_windows'] == 0).sum()} "
        f"({', '.join(str(s) for s in info.loc[info['n_windows'] == 0, 'shot_id'])}).", "",
        "## Windows per shot", "", "| status | shots | median | min | max |", "|---|---|---|---|---|",
        *[f"| {k} | {r['count']} | {r['median']:.0f} | {r['min']} | {r['max']} |" for k, r in per_shot.iterrows()], "",
        "## Causality on real shots", "",
        f"Each shot was cut at a random window end and its windows recomputed from the cut data only. "
        f"{compared} rows compared, all identical (exact equality).", "",
        "## Missing values (share of windows)", "",
        "| feature | NaN share |", "|---|---|",
        *[f"| {k} | {v:.3f} |" for k, v in nan_share[nan_share > 0].items()], "",
        "Missing values come from signals absent for the whole shot (`has_<signal>` = 0); see NOTES.md.", "",
        f"Plot: `shot_{config.REFERENCE_SHOT}.png`.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="random shots to check (ignored with --ids-file)")
    parser.add_argument("--ids-file", help="JSON list of shot ids")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    table = load_shot_table()
    rng = np.random.default_rng(args.seed)
    if args.ids_file:
        ids = json.load(open(args.ids_file))
    else:
        ids = [int(i) for i in rng.choice(table["shot_id"].to_numpy(), size=args.n, replace=False)]
    ids = sorted(set(ids) | {config.REFERENCE_SHOT})
    failed = set(download_shots(ids)["failed"])
    comments = table.set_index("shot_id")["shot_postshot_comment"]

    windows, info, compared = [], [], 0
    for shot_id in ids:
        if shot_id in failed:
            continue
        shot, lab, win = label_and_windows(shot_id, comments.get(shot_id))
        win.attrs["t_stop"] = lab["t_disrupt"] if lab["status"] == "disrupted" else np.inf
        compared += truncation_check(shot, win, rng)
        info.append({"shot_id": shot_id, "status": lab["status"], "n_windows": len(win)})
        windows.append(win)
        if shot_id == config.REFERENCE_SHOT:
            os.makedirs(OUT_DIR, exist_ok=True)
            plot_reference(shot, lab, win, os.path.join(OUT_DIR, f"shot_{shot_id}.png"))

    all_win = pd.concat(windows, ignore_index=True)
    text = summary_text(all_win, pd.DataFrame(info), compared)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
