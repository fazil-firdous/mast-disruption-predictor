"""One picture for Telegram that says how far training has got, what the model can do now, and what is needed
next. Every number on it is read from the results on disk; the picture only shows them."""
import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src import config, notify, resume, server, stats

EXPERIMENTS = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E11", "E12", "E13"]
SIGNAL_WORDS = {"prad": "radiated power", "neutrons": "neutron rate", "ip_abs": "plasma current",
                "pnbi": "beam heating power"}
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, BLUE_TRACK, GOOD, WARNING = "#2a78d6", "#cde2fb", "#0ca30c", "#fab219"


def current_stamp():
    """The stamp of the current code and dataset, or None before the first build."""
    if not all(os.path.exists(p) for p in resume.processed_files()):
        return None
    return resume.stamp(resume.processed_files())


def experiments_done(current):
    """How many of the run_all experiments have a result made with the current code and dataset."""
    if current is None:
        return 0
    return sum(resume.load(os.path.join(config.RESULTS_DIR, exp), current) is not None for exp in EXPERIMENTS)


def dataset_info():
    """The counts the last dataset build saved (shots, disrupted, table_source, ...), or {} before the first build."""
    path = os.path.join(config.PROCESSED_DIR, resume.STAMP_FILE)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)["outcome"]


def skill(seeds_csv=None):
    """Average catch and false-alarm rate over the E10 seeds, and whether catching beats a random alarm that
    fires as often: 'better', 'worse' or 'same', from the corrected 95% range of catch minus false-alarm rate over
    the seeds (src/stats.py; the splits share shots, so a plain standard error is too sure). None below MIN_SEEDS."""
    path = seeds_csv or os.path.join(config.RESULTS_DIR, "E10", "summary.csv")
    if not os.path.exists(path):
        return None
    table = pd.read_csv(path)
    if len(table) < stats.MIN_SEEDS:
        return None
    _, low, high = stats.corrected_interval((table["catch_rate"] - table["false_alarm_rate"]).to_numpy())
    verdict = "better" if low > 0 else "worse" if high < 0 else "same"
    return {"n": len(table), "catch": float(table["catch_rate"].mean()),
            "false_alarm": float(table["false_alarm_rate"].mean()), "verdict": verdict}


def one_test(current):
    """Catch and false-alarm rate of the single E3 test (before any repeat runs), or None."""
    if current is None:
        return None
    saved = resume.load(os.path.join(config.RESULTS_DIR, "E3"), current)
    if saved is None:
        return None
    test = saved["test"]
    return {"n": 1, "catch": test["catch_rate"], "false_alarm": test["false_alarm_rate"], "verdict": "one"}


def e5_current(current):
    """True if the saved E5 result was made with the current code and dataset."""
    return current is not None and resume.load(os.path.join(config.RESULTS_DIR, "E5"), current) is not None


def signal_usefulness(e5_csv=None):
    """{signal words: how much the ranking score (PR-AUC) drops without that signal}, from E5; {} if not run."""
    path = e5_csv or os.path.join(config.RESULTS_DIR, "E5", "summary.csv")
    if not os.path.exists(path):
        return {}
    table = pd.read_csv(path).set_index("drop_signal")
    if "none" not in table.index:
        return {}
    full = table.at["none", "pr_auc"]
    return {SIGNAL_WORDS.get(s, s): float(full - table.at[s, "pr_auc"]) for s in table.index if s != "none"}


def auto_labels(path=None):
    """(shots labelled, shots in the dataset) from the last automatic labelling, or (0, 0) before it has run."""
    path = path or os.path.join(config.RESULTS_DIR, "auto_labels", "summary.json")
    if not os.path.exists(path):
        return 0, 0
    with open(path) as f:
        s = json.load(f)
    return s["n_labelled"], s["n_shots"]


def needs(info):
    """The short to-do list under the picture: (done?, text). Built only from what is on disk and in the state."""
    items = []
    full = info["table_source"] != "report"
    items.append((full, "full MAST shot list" if full else "more shots: only the 400 report shots so far"))
    if info["seeds"] < config.SEED_SWEEP_MAX:
        items.append((False, f"repeat runs {info['seeds']}/{config.SEED_SWEEP_MAX}"))
    else:
        total = config.SEED_SWEEP_MAX * len(config.COMPARISONS)
        items.append((info["comparisons"] >= total, f"repeat-run comparisons {info['comparisons']}/{total}"))
    items.append((False, f"next: {config.NEXT_STEP}"))
    return items


def repeat_total():
    """Every repeat run planned: the E10 seeds, then each comparison of scripts/repeat_runs.py on as many seeds."""
    return config.SEED_SWEEP_MAX * (1 + len(config.COMPARISONS))


def collect(state=None):
    """Everything the card shows, read from the results, dataset and server state on disk."""
    state = state or server.read_state()
    current = current_stamp()
    data = dataset_info()
    probe = state.get("last_probe") or {}
    # repeat runs made with older code or data are not shown as if they were current
    seeds_valid = current is not None and server.seeds_still_valid(state, current)
    return {"shots": data.get("shots", 0), "disrupted": data.get("disrupted", 0),
            "table_source": data.get("table_source", "live"), "all_shots": probe.get("shot_stores"),
            "experiments": experiments_done(current), "seeds": len(state["seeds_done"]) if seeds_valid else 0,
            "comparisons": len(state["comparisons_done"]) if seeds_valid else 0,
            "auto_labelled": auto_labels()[0], "auto_total": auto_labels()[1],
            "skill": (skill() if seeds_valid else None) or one_test(current),
            "signals": signal_usefulness() if e5_current(current) else {}}


def meter(ax, y, label, done, total, text):
    """One progress row: label on the left, value on the right, a rounded bar underneath."""
    ax.text(0.06, y, label, fontsize=12, color=INK_2, va="bottom")
    ax.text(0.94, y, text, fontsize=12, color=INK, va="bottom", ha="right", fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.06, y - 0.022), 0.88, 0.012, boxstyle="round,pad=0,rounding_size=0.006",
                                color=BLUE_TRACK, lw=0))
    if total:
        width = 0.88 * min(done / total, 1.0)
        if width > 0:
            ax.add_patch(FancyBboxPatch((0.06, y - 0.022), width, 0.012, boxstyle="round,pad=0,rounding_size=0.006",
                                        color=BLUE, lw=0))


def tile(ax, x, value, top, bottom):
    """One headline number with a word above and below it."""
    ax.text(x, 0.585, top, fontsize=11, color=INK_2, ha="center")
    ax.text(x, 0.535, value, fontsize=30, color=INK, ha="center", fontweight="bold")
    ax.text(x, 0.515, bottom, fontsize=10, color=MUTED, ha="center", va="top")


def mark(ax, x, y, ok):
    """Status icon: a green tick for done, an amber dot-and-bar for still to do (never colour alone)."""
    ax.add_patch(plt.Circle((x, y), 0.013, color=GOOD if ok else WARNING, transform=ax.transAxes))
    ax.text(x, y, "✓" if ok else "!", fontsize=11, color="white", ha="center", va="center", fontweight="bold")


def draw_signals(fig, signals):
    """Small bar chart: how much worse the model ranks without each signal (E5). Longer = it relies on it more;
    a signal whose removal does not hurt gets no bar."""
    chart = fig.add_axes([0.36, 0.255, 0.56, 0.11], facecolor=SURFACE)
    names = sorted(signals, key=signals.get)
    values = [max(signals[n], 0.0) for n in names]
    chart.barh(names, values, color=BLUE, height=0.55)
    chart.axvline(0, color="#c3c2b7", lw=1)
    for i, v in enumerate(values):
        if v == 0:
            chart.text(0, i, "  no help", va="center", fontsize=9, color=MUTED, ha="left")
    chart.set_xlim(0, max(max(values), 1e-3) * 1.15)
    chart.set_xticks([])
    chart.tick_params(axis="y", labelsize=10, colors=INK_2, length=0)
    for side in ("top", "right", "left", "bottom"):
        chart.spines[side].set_visible(False)


def draw(info, path, commit=""):
    """Write the status picture (PNG) for the numbers in info (see collect)."""
    fig = plt.figure(figsize=(8, 10.5), dpi=110, facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.06, 0.955, "MAST disruption predictor", fontsize=20, color=INK, fontweight="bold")
    ax.text(0.06, 0.93, f"status {pd.Timestamp.now():%d %b %Y %H:%M}" + (f"  |  code {commit}" if commit else ""),
            fontsize=10, color=MUTED)

    ax.text(0.06, 0.88, "1  Training", fontsize=14, color=INK, fontweight="bold")
    total = info["all_shots"]
    meter(ax, 0.845, "Shots learned from", info["shots"], total,
          f"{info['shots']:,}" + (f" of {total:,}" if total else ""))
    meter(ax, 0.795, "Experiments finished", info["experiments"], len(EXPERIMENTS),
          f"{info['experiments']} of {len(EXPERIMENTS)}")
    runs = info["seeds"] + info["comparisons"]
    meter(ax, 0.745, "Repeat runs (for error bars)", runs, repeat_total(), f"{runs} of {repeat_total()}")
    labelled, total = info["auto_labelled"], info["auto_total"]
    meter(ax, 0.695, "Shots labelled automatically", labelled, total, f"{labelled:,} of {total:,}")

    ax.text(0.06, 0.635, "2  What it can do now", fontsize=14, color=INK, fontweight="bold")
    s = info["skill"]
    if s is None:
        ax.text(0.06, 0.56, f"Waiting for at least {stats.MIN_SEEDS} repeat runs.", fontsize=12, color=INK_2)
    else:
        tile(ax, 0.2, f"{100 * s['catch']:.0f}%", "Catches", "of disruptions\nin time")
        tile(ax, 0.5, f"{100 * s['false_alarm']:.0f}%", "False alarms", "on normal\nshots")
        tile(ax, 0.8, f"{100 * s['false_alarm']:.0f}%", "A random guess", "would catch\nas many")
        mark(ax, 0.075, 0.425, s["verdict"] == "better")
        verdict = {"better": f"Better than guessing  (average of {s['n']} runs)",
                   "same": f"Not clearly better than guessing yet  (average of {s['n']} runs)",
                   "worse": f"Worse than guessing  (average of {s['n']} runs)",
                   "one": "One test only: wait for the repeat runs"}[s["verdict"]]
        ax.text(0.105, 0.425, verdict, fontsize=12, color=INK, va="center")
    if info["signals"]:
        ax.text(0.06, 0.34, "Signals it\nrelies on", fontsize=11, color=INK_2, va="center")
        ax.text(0.36, 0.235, "longer bar = worse without it", fontsize=9, color=MUTED)
        draw_signals(fig, info["signals"])

    ax.text(0.06, 0.19, "3  Needed next", fontsize=14, color=INK, fontweight="bold")
    for i, (ok, text) in enumerate(needs(info)):
        y = 0.15 - 0.035 * i
        mark(ax, 0.075, y, ok)
        ax.text(0.105, y, text, fontsize=12, color=INK if not ok else INK_2, va="center")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def current_commit():
    """Short hash of the checked-out code ('' if git is not available)."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_DIR, capture_output=True,
                            text=True)
    return result.stdout.strip()


def send(caption, state=None):
    """Draw the picture into LOGS_DIR/status.png and send it to Telegram with a short caption."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    path = draw(collect(state), os.path.join(config.LOGS_DIR, "status.png"), current_commit())
    notify.send_file(path, caption)
    notify.remember(f"status picture: {caption}")
