"""The picture and progress block the research server prints in its terminal: a small spherical tokamak in braille
dots, the title, and bars for how far training has got. Looks only; nothing here changes a result."""
import sys
from itertools import groupby

from src import config, status_card

# a spherical tokamak (plasma torus around the central column), drawn by a small renderer and saved as text
ART = [
    "    ⢀⣼⣿⣿⣿⢿⢟⢝⠝⢝⠝⢽⢽⢽⢿⣽⢝⢕⠄",
    "   ⢰⣿⣿⣿⣿⢟⠕⠕⠅⠁⠁⠁⠅⠝⢝⢽⢽⢿⢵⢕⢅",
    "  ⢰⣿⣿⣿⢿⢝⠅⠅⠁ ⠁ ⠁ ⠁⠅⢝⢽⢿⢽⢝⢕⠅",
    " ⢀⢽⣿⣿⣿⢿⠅⠁ ⠁ ⠁ ⠁ ⠁⠄⠝⢽⣿⣽⢝⢕⠅⠄",
    " ⢰⢽⣿⢿⣿⢿⢅⠁ ⠁ ⠁ ⠁ ⠁ ⠕⢽⣿⢽⢝⢕⠅⠅",
    " ⢰⢽⢽⢿⣿⢿⢵⠅   ⠁ ⠁   ⢑⣽⢿⢽⢝⠅⠅⠅",
    " ⢐⢝⢽⢽⢽⢿⣽⢕⢄  ⠁ ⠁  ⢀⣾⣿⢟⢕⠕⠅⠁",
    "  ⢝⢽⢝⢽⢽⢽⢿⢽⢵⣤⢁⢀⣁⣠⣴⣿⢿⢽⠝⠕⠕⠅⠁",
    "  ⠑⢕⢝⢕⢝⢽⢝⢽⢽⢽⢿⢿⢿⢽⢟⢝⠝⢕⠕⠅⠁⠅⠁",
    "   ⠅⠕⢕⠝⢕⢝⢕⢝⢕⢝⢕⢝⢕⠕⠕⠕⠅⠁⠅⠁ ⠁",
    "    ⠁⠅⠕⠅⠕⠅⠕⠅⠕⠅⠕⠅⠕⠅⠁⠅⠁ ⠁",
    "      ⠁⠁⠁⠅⠁⠅⠁⠅⠁⠅⠁ ⠁ ⠁ ⠁",
]
# one letter per character of ART: W white, C bright cyan, c cyan, b bright blue, d blue, g grey (central column)
COLOURS = [
    "    WWWWWWWCCcCCCCCCCcc",
    "   WWWWWCcbbddbbcCCCCCcb",
    "  CWWWWCcdd g d dbcCCCccb",
    " cCWWWCcd d g g ddcCWCccbd",
    " cCWWWCcd d g g d bWWCccbd",
    " cCCCCCCc   g g   CWWCcbbd",
    " ccCCCCCCc  g g  WWWCccbd",
    "  cccCCCCCCCgggWWWWCccbdd",
    "  bcccccCCCCWWWWCCccbbddd",
    "   bbcccccccccccccbbddd d",
    "    dbbbbbbbbbbbbbddd d",
    "      ddddddddddd d d d",
]
ANSI = {"W": "97", "C": "96", "c": "36", "b": "94", "d": "34", "g": "90", "y": "93", "G": "92", "bold": "1"}
TITLE = "FAZIL'S TOKAMAK ML MODEL"
BAR_WIDTH = 22


def paint(text, code, colour=True):
    """text wrapped in an ANSI colour code (plain text when colour is False, e.g. not a terminal)."""
    return f"\033[{ANSI[code]}m{text}\033[0m" if colour and text.strip() else text


def art_lines(colour=True):
    """The tokamak, one coloured string per line, all padded to the same width."""
    width = max(len(line) for line in ART)
    out = []
    for line, codes in zip(ART, COLOURS):
        painted = ""
        for code, run in groupby(zip(line, codes), key=lambda pair: pair[1]):
            text = "".join(ch for ch, _ in run)
            painted += text if code == " " else paint(text, code, colour)
        out.append(painted + " " * (width - len(line)))
    return out


def title_lines(commit, mode, workspace, colour=True):
    """The text next to the tokamak."""
    return ["", "", paint(TITLE, "W", colour), paint("═" * len(TITLE), "b", colour),
            paint("MAST disruption predictor", "C", colour), "", paint(f"code {commit}", "g", colour),
            paint(f"data: {mode}", "g", colour), paint(f"workspace: {workspace}", "g", colour)]


def banner(commit, mode, workspace, colour=None):
    """The start-up picture: tokamak on the left, title on the right."""
    colour = sys.stdout.isatty() if colour is None else colour
    left, right = art_lines(colour), title_lines(commit, mode, workspace, colour)
    rows = []
    for i in range(max(len(left), len(right))):
        l = left[i] if i < len(left) else " " * max(len(line) for line in ART)
        rows.append(l + "   " + (right[i] if i < len(right) else ""))
    return "\n".join(rows)


def bar(done, total, colour=True):
    """A progress bar of BAR_WIDTH cells, filled in proportion to done/total; blank if the total is unknown."""
    if not total:
        return " " * BAR_WIDTH
    filled = int(round(BAR_WIDTH * min(done / total, 1))) if total else 0
    return paint("█" * filled, "C", colour) + paint("░" * (BAR_WIDTH - filled), "g", colour)


def progress(info, job, colour=None):
    """The training block: bars from the numbers of status_card.collect(), then the running job and the skill."""
    colour = sys.stdout.isatty() if colour is None else colour
    rows = [paint("── Training " + "─" * 52, "g", colour)]
    for label, done, total in [("Shots learned from", info["shots"], info["all_shots"]),
                               ("Experiments", info["experiments"], len(status_card.EXPERIMENTS)),
                               ("Repeat runs", info["seeds"], config.SEED_SWEEP_MAX),
                               ("Auto labels", info["auto_labelled"], info["auto_total"])]:
        count = f"{done:,} / {total:,}" if total else f"{done:,}"
        rows.append(f"{label:<20}{bar(done, total, colour)}  {count}")
    rows.append(paint("── Now " + "─" * 57, "g", colour))
    rows.append(f"job: {job or 'idle'}" + skill_text(info["skill"], colour))
    return "\n".join(rows)


def skill_text(skill, colour=True):
    """'   catches 7% vs 5% by chance  ✓ better than guessing', or '' before there is a result."""
    if skill is None:
        return ""
    text = f"   catches {100 * skill['catch']:.0f}% vs {100 * skill['false_alarm']:.0f}% by chance  "
    if skill["verdict"] == "better":
        return text + paint("✓ better than guessing", "G", colour)
    if skill["verdict"] == "one":
        return text + paint("(one test only)", "y", colour)
    return text + paint("! not better than guessing yet", "y", colour)
