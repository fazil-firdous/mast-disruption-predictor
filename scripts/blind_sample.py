"""A blind sample for an expert check of the termination classes: a seeded draw of shots lost during flat-top with flux
left that nobody noted (candidates), noted disruptions lost the same way (recorded), and un-noted flux-exhausted
endings, shuffled and renamed B01, B02, ... so the expert sees neither the shot number, the note nor our class.

Each shot gets one plot of measured signals (plasma current with its programme, solenoid current, radiated power,
Mirnov fluctuation level, line-average density): no detector time, no labels. Writes to MAST_HOME/blind/ (private):
plots/Bxx.png, answer_sheet.csv (code, disruption yes/no/unsure, comment: for the expert) and key.csv (code, shot,
group: never sent). Needs the dataset with endings (scripts/build_dataset.py) and its cached signals.

    python scripts/blind_sample.py            # 10 per group
    python scripts/blind_sample.py --per-group 5
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, extras
from src.data import load_shot

FOLDER = os.path.join(config.HOME_DIR, "blind")
GROUPS = {
    "candidate": lambda s: s["candidate"],
    "recorded": lambda s: s["note"] & (s["termination"] == "flat_top_drop"),
    "flux_exhausted": lambda s: ~s["note"] & (s["termination"] == "flux_exhausted"),
}


def draw(shots, per_group, seed=config.RANDOM_SEED):
    """Seeded draw of per_group shots from each group of GROUPS, shuffled and coded B01, B02, ...
    Returns the key: code, shot_id, group (fewer shots in a group if it has fewer)."""
    rng = np.random.default_rng(seed)
    picked = []
    for group, rule in GROUPS.items():
        ids = shots.loc[rule(shots), "shot_id"].to_numpy()
        chosen = rng.choice(ids, size=min(per_group, len(ids)), replace=False)
        picked += [(int(s), group) for s in chosen]
    order = rng.permutation(len(picked))
    rows = [{"code": f"B{i + 1:02d}", "shot_id": picked[j][0], "group": picked[j][1]} for i, j in enumerate(order)]
    return pd.DataFrame(rows, columns=["code", "shot_id", "group"])


def answer_sheet(key):
    """The sheet the expert fills in: only the codes, with empty answer columns."""
    return pd.DataFrame({"code": key["code"], "disruption (yes / no / unsure)": "", "comment": ""})


def plot_shot(code, shot_df, extra, path):
    """One shot's measured signals against time, titled only with its code. No labels, no detector time."""
    t = shot_df["time"].to_numpy()
    fig, axes = plt.subplots(5, 1, figsize=(7, 9), sharex=True)
    axes[0].plot(t, np.abs(shot_df["ip"]) / 1e3, label="measured")
    if extra is not None:
        axes[0].plot(extra["time"], extra["ip_ref"] / 1e3, "--", label="programmed")
    axes[0].set_ylabel("|Ip| (kA)")
    axes[0].legend(loc="upper right", fontsize=7)
    panels = [("sol_current", 1e3, "solenoid (kA)"), ("mirnov_rms", 1e-6, "Mirnov RMS (uT)"),
              ("ne_line_avg", 1e19, "n_e (1e19 m^-3)")]
    axes[1].plot(t, shot_df["power_radiated"] / 1e3)
    axes[1].set_ylabel("P_rad (kW)")
    for ax, (column, scale, name) in zip(axes[2:], panels):
        if extra is not None and column in extra:
            ax.plot(extra["time"], extra[column] / scale)
        ax.set_ylabel(name)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(code)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-group", type=int, default=10)
    args = parser.parse_args()
    shots = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    key = draw(shots, args.per_group)
    os.makedirs(os.path.join(FOLDER, "plots"), exist_ok=True)
    for row in key.itertuples():
        plot_shot(row.code, load_shot(row.shot_id), extras.load_extras(row.shot_id, allow_download=False),
                  os.path.join(FOLDER, "plots", f"{row.code}.png"))
    answer_sheet(key).to_csv(os.path.join(FOLDER, "answer_sheet.csv"), index=False)
    key.to_csv(os.path.join(FOLDER, "key.csv"), index=False)
    print(f"{len(key)} shots: plots and answer sheet in {FOLDER}; key.csv stays private")


if __name__ == "__main__":
    main()
