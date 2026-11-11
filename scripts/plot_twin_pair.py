"""A lost shot and its same-programme twin, signal by signal: a picture of one pair for the twin-test paper.

    python scripts/plot_twin_pair.py --list           # some lost shots that have a twin with the same beam heating
    python scripts/plot_twin_pair.py 12345            # saves RESULTS_DIR/twin_pairs/12345.png and sends it to Telegram
    python scripts/plot_twin_pair.py 12345 --columns  # the names of the signals a shot has

Reads the release's programme_twins.csv (scripts/make_release.py) and the cached shots. Outside every stamp.
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, data, extras, notify

LOST_COLOUR, TWIN_COLOUR = "#eb6834", "#8a8984"
# One panel per entry: (column of the shot table, label of the y axis, factor that turns the stored unit into the
# label's unit). The plasma current panel also shows the programmed current as a dashed line.
PANELS = [
    ("ip", "plasma current (kA)", 1e-3),
    ("power_nbi", "beam power (MW)", 1e-6),
    # YOUR TURN: add two panels here, one line each, in the same form as the two lines above.
    # 1. the line-average density: its column is "ne_line_avg", stored in m^-3; show it in units of 10^19 m^-3
    # 2. the height of the plasma (vertical position): its column is "efit_zmag", stored in m; show it in cm
    ("ne_line_avg", "density (10^19 m^-3)", 1e-19),
    ("efit_zmag", "height (cm)", 100),
]


def twins_path():
    return os.path.join(config.RESULTS_DIR, "release", "mast-events-twins-v1", "programme_twins.csv")


def find_twin(twins, shot_id):
    """The row of programme_twins.csv for this lost shot."""
    rows = twins[twins["shot_id"] == shot_id]
    if len(rows) == 0:
        raise SystemExit(f"shot {shot_id} is not a lost shot with a twin; try --list")
    return rows.iloc[0]


def load(shot_id):
    """One shot's signals on its 1 ms grid: the summary signals plus the cached extras."""
    return extras.with_extras(data.load_shot(int(shot_id)), int(shot_id))


def plot(lost, twin, pair, path, panels=None):
    """Draw the panels for the lost shot (colour) and its twin (grey), with a dashed line at the loss."""
    panels = panels or PANELS
    t_loss = float(pair["t_loss"])
    fig, axes = plt.subplots(len(panels), 1, figsize=(7, 1.9 * len(panels) + 0.6), sharex=True, squeeze=False)
    for ax, (column, label, factor) in zip(axes[:, 0], panels):
        for shot, colour, name in ((twin, TWIN_COLOUR, f"twin {int(pair['other'])} (not lost)"),
                                   (lost, LOST_COLOUR, f"shot {int(pair['shot_id'])} (lost)")):
            values = shot[column].abs() if column == "ip" else shot[column]
            ax.plot(shot["time"], values * factor, color=colour, lw=1.1, label=name)
        if column == "ip" and lost["ip_ref"].notna().any():
            ax.plot(lost["time"], lost["ip_ref"] * factor, color="black", lw=0.8, ls="--", label="programmed")
        ax.axvline(t_loss, color="black", lw=0.8, ls=":")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(color="#e4e3de", lw=0.6)
    axes[0, 0].legend(fontsize=7, ncol=3, loc="lower center", frameon=False)
    axes[0, 0].set_title(f"Same programme, different fate: lost at {t_loss:.3f} s (dotted line); beam heating "
                         f"{pair.get('heating', 'unknown')}", fontsize=9, loc="left")
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 0].set_xlim(0, t_loss + 0.03)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def listing(twins, n=10):
    """Some pairs worth looking at: recorded disruptions first, same beam heating, programmes within 1%."""
    good = twins[(twins["heating"] == "same") & (twins["rms"] < 0.01)]
    good = good.sort_values("group", ascending=False)  # 'recorded' before 'candidate'
    return good[["shot_id", "other", "group", "gap", "t_loss"]].head(n).to_string(index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("shot", nargs="?", type=int, help="a lost shot that has a twin")
    parser.add_argument("--list", action="store_true", help="show some lost shots to try")
    parser.add_argument("--columns", action="store_true", help="print the names of the shot's signals")
    args = parser.parse_args()
    twins = pd.read_csv(twins_path())
    if "t_loss" not in twins:
        raise SystemExit("this release is an old one: run scripts/make_release.py first")
    if args.list or args.shot is None:
        print(listing(twins))
        return
    pair = find_twin(twins, args.shot)
    lost, twin = load(pair["shot_id"]), load(pair["other"])
    if args.columns:
        print(", ".join(lost.columns))
        return
    path = plot(lost, twin, pair, os.path.join(config.RESULTS_DIR, "twin_pairs", f"{args.shot}.png"))
    print(f"saved {path}")
    notify.send_file(path, f"Shot {args.shot} and its twin {int(pair['other'])}")


if __name__ == "__main__":
    main()
