"""Put the public data release together: the per-shot catalogue, the events read from operators' comments, the named
repeat-shot pairs, the programme twins (with their beam heating) and the triplets, as plain CSV files with a README
and checksums.

    python scripts/make_release.py      # writes RESULTS_DIR/release/mast-events-twins-v1/ and a .zip next to it

Reads only files that other scripts wrote (events/, twin_differences/) plus, for the twins' beam heating, the window
table and the cached extras. The operators' comment text is left out: the release gives the event class, its time
and the comment field it came from, and the text itself stays in the FAIR-MAST shot table. Nothing is uploaded
anywhere: the folder stays on the server until its owner publishes it. Outside every stamp.
"""
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import v3_explore, v3_runs
from src import config, extras

NAME = "mast-events-twins-v1"
EVENTS_DIR = os.path.join(config.RESULTS_DIR, "events")
TRIPLETS = os.path.join(config.RESULTS_DIR, "twin_differences", "triplets.csv")
SHOT_COLUMNS = ["shot_id", "campaign", "note", "termination", "t_quench", "group", "p_disrupted", "cause", "t_event",
                "note_vote", "words_vote", "solenoid_vote"]
EVENT_COLUMNS = ["shot_id", "group", "event", "time_s", "t_quench", "source"]  # no comment text


def out_dir():
    return os.path.join(config.RESULTS_DIR, "release", NAME)


def code_version():
    """The short git commit of the code that made the release, or 'unknown'."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_DIR, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def with_heating(twins, catalogue, load=None, beams=None):
    """The programme twins with what the twin test needs for every pair: t_loss (the lost shot's quench), t_start
    (its flat-top start, empty if unknown) and heating ('same', 'different', 'unknown', as scripts/v3_explore.py
    defines it)."""
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    ids = list(twins[["shot_id", "other"]].itertuples(index=False, name=None))
    beams = v3_explore.beam_power({shot for pair in ids for shot in pair}) if beams is None else beams
    t_quench = catalogue.set_index("shot_id")["t_quench"].to_dict()
    starts = {shot: v3_runs.flat_top_start(load(shot)) for shot in twins["shot_id"]}
    same = v3_explore.heating(ids, beams, t_quench, starts)
    return twins.assign(t_loss=[t_quench.get(lost) for lost, _ in ids], t_start=[starts[lost] for lost, _ in ids],
                        heating=[same[pair] for pair in ids])


def tables(load=None, beams=None):
    """{file name: DataFrame} of everything in the release that has its input on disk."""
    catalogue = pd.read_csv(os.path.join(EVENTS_DIR, "catalogue.csv"))
    events = pd.read_csv(os.path.join(EVENTS_DIR, "events.csv"))
    out = {"shots.csv": catalogue[[c for c in SHOT_COLUMNS if c in catalogue]],
           "events.csv": events[[c for c in EVENT_COLUMNS if c in events]]}
    named = os.path.join(EVENTS_DIR, "twins.csv")
    if os.path.exists(named):
        out["named_pairs.csv"] = pd.read_csv(named)
    if os.path.exists(v3_runs.TWINS):
        twins = with_heating(pd.read_csv(v3_runs.TWINS), catalogue, load, beams)
        flags_path = os.path.join(config.RESULTS_DIR, "plan_twins", "pairs.csv")  # scripts/plan_twins.py
        if os.path.exists(flags_path):
            flags = pd.read_csv(flags_path)
            keep = [c for c in ("plan", "full_plan", "field_plan", "shape_plan") if c in flags]
            twins = twins.merge(flags[["shot_id", "other"] + keep], on=["shot_id", "other"], how="left")
        out["programme_twins.csv"] = twins
    if os.path.exists(TRIPLETS):
        out["triplets.csv"] = pd.read_csv(TRIPLETS)
    return out


def counts(frame, column):
    return ", ".join(f"{k} {v:,}" for k, v in frame[column].value_counts().items())


def readme(tabs, version):
    """The README text: what each file and column is, how it was made, its limits, licence and citation."""
    shots, events = tabs["shots.csv"], tabs["events.csv"]
    lines = [f"# MAST events and twins, version 1 ({NAME})", "",
             "How 11,471 MAST discharges ended, what the operators wrote about them, and which shots ran the same "
             "programme: tables derived from the public FAIR-MAST level-2 data (UKAEA). Made by the code at "
             f"https://github.com/fazil-firdous/mast-disruption-predictor (commit {version}).", "",
             "Everything here is made by rules and a statistical model, not checked shot by shot by a MAST expert. "
             "The classes say when and how the plasma was lost, not why.", "",
             "## Files", "",
             f"### shots.csv ({len(shots):,} rows, one per shot)", "",
             "| Column | Meaning |", "|---|---|",
             "| shot_id | MAST shot number |",
             "| campaign | M5 to M9, or Unknown |",
             "| note | the operator's post-shot comment calls the shot a disruption (and does not negate it) |",
             "| termination | flat_top_drop (lost during the programmed flat-top with solenoid flux left), "
             "flux_exhausted, programmed_rampdown, smooth_end (no fast current quench) or unknown |",
             "| t_quench | time (s) of the fast current quench (at least 80% fall within 10 ms); empty if none |",
             "| group | recorded (note and quench), candidate (flat-top drop without a note), flux_exhausted, "
             "programmed_rampdown, smooth_end or other |",
             "| p_disrupted | probability from a latent-class model over three sources (note, comment words, solenoid). "
             "A ranking, not a calibrated probability: it leans on the solenoid rule, and two of the sources read the "
             "same comment. 0 for shots without a quench |",
             "| cause | the most specific cause named in the post-shot comment: VDE, locked mode, MHD/NTM, IRE, "
             "hardware trip, or empty |",
             "| t_event | the operator's written time (s) of a named loss if it is within 20 ms of the quench, else "
             "t_quench |",
             "| note_vote, words_vote, solenoid_vote | the three sources' answers (words_vote 2 = a loss named with a "
             "time within 20 ms of the quench, 1 = named, 0 = none; solenoid_vote -1 = no answer) |", "",
             f"Groups: {counts(shots, 'group')}.", "",
             f"### events.csv ({len(events):,} rows, one per event named in a comment)", "",
             "| Column | Meaning |", "|---|---|",
             "| shot_id, group, t_quench | as in shots.csv |",
             "| event | the event class matched by a fixed word list (disruption, IRE, VDE, locked mode, MHD/NTM, "
             "plasma lost, early end, hardware trip and others); negated mentions are skipped |",
             "| time_s | the time written next to the event (s), or empty |",
             "| source | the comment field of the FAIR-MAST shot table it came from |", "",
             "The comment text is not included; it is in the FAIR-MAST shot table. A match is a hint about one comment, "
             "not a label.", ""]
    if "named_pairs.csv" in tabs:
        named = tabs["named_pairs.csv"]
        lines += [f"### named_pairs.csv ({len(named):,} rows)", "",
                  "Shots whose comment names an earlier shot they repeat or compare with. Columns: shot_id, other (the "
                  "shot named), relation (repeat, reference, same as), source, group and other_group, twin (one of the "
                  "two was lost in the flat-top and the other was not), rms and peak_ratio (the comparison of the two "
                  "programmed currents: RMS difference over the larger peak, and smaller peak over larger), overlap_s, "
                  "same_programme (rms under 0.10 and peak_ratio over 0.90).", ""]
    if "programme_twins.csv" in tabs:
        twins = tabs["programme_twins.csv"]
        lines += [f"### programme_twins.csv ({len(twins):,} rows, one per lost shot that has a twin)", "",
                  "For each lost shot (a recorded disruption, or a candidate: a flat-top loss without a note): the "
                  "nearest earlier shot, within 30 shots "
                  "and in the same campaign, with the same programmed current, that was not lost.", "",
                  "| Column | Meaning |", "|---|---|",
                  "| shot_id | the lost shot |", "| other | its twin |",
                  "| group, other_group | the two shots' groups |", "| gap | shot_id minus other |",
                  "| rms, peak_ratio, overlap_s | the comparison of the two programmed currents: RMS difference over "
                  "the larger peak, smaller peak over larger, and the seconds compared |",
                  "| t_loss | the lost shot's quench time (s) |",
                  "| t_start | the start of the lost shot's flat-top (s): the programmed current first reaching 90% of "
                  "its peak; empty if unknown |",
                  "| heating | same if the two shots' beam power, from the lost shot's flat-top start to 5 ms before "
                  "its loss, differs by under 10% of its peak (RMS) or neither has a beam; different; or unknown |", "",
                  f"Lost shots: {counts(twins, 'group')}. Beam heating: {counts(twins, 'heating')}. Programmes within "
                  f"1%: {(twins['rms'] < v3_runs.EXACT_RMS).mean():.1%}.", ""]
        if "plan" in twins:
            lines += ["| plan | the two shots also asked for the same reference density and height, and the same gas "
                      "valve voltages where both have them (each within 1%, or the same constant) |",
                      "| full_plan | as plan, with valve requests present in both shots |"]
            if "shape_plan" in twins:
                lines += ["| field_plan | plan, and toroidal field coil currents within 1% |",
                          "| shape_plan | field_plan, and the P2, P4 and P5 coil currents each within 10%, compared up "
                          "to 100 ms before the loss |"]
            lines += ["", "Plan twins: " + ", ".join(f"{c} {int(twins[c].fillna(False).astype(bool).sum()):,}"
                                                     for c in ("plan", "full_plan", "field_plan", "shape_plan")
                                                     if c in twins) + ".", ""]
        lines += ["Every twin shares the programmed plasma current; the columns above say which also share more of "
                  "the plan. The twin is always the earlier shot.", ""]
    if "programme_twins.csv" in tabs:
        lines += ["### twin_score.py: the twin test for your own predictor", "",
                  "    python twin_score.py scores.csv programme_twins.csv", "",
                  "scores.csv has one row per score: shot_id, t_end (s, the end of the data the score used) and score "
                  "(higher = more likely to be lost). Score only shots the model was not trained or tuned on; a pair "
                  "counts when both of its shots are in the file. The script prints C, the share of pairs in which the "
                  "lost shot scores higher than its twin before the loss (0.5 = chance), with a 95% range, on all "
                  "pairs, on pairs with the same beam heating and on pairs whose programmes match within 1%. It needs "
                  "only numpy and pandas.", ""]
    if "triplets.csv" in tabs:
        lines += [f"### triplets.csv ({len(tabs['triplets.csv']):,} rows)", "",
                  "For each lost shot with two such twins: shot_id, k1 (the nearest), k2 (the next), group.", ""]
    lines += ["## Licence and citation", "",
              "Derived from FAIR-MAST (UKAEA), licence CC BY-SA 4.0; these tables are shared under the same licence.", "",
              "Please cite the data source, S. Jackson et al., \"FAIR-MAST: A fusion device data management system\", "
              "SoftwareX 27, 101869 (2024), and F. Firdous, \"What disruption labels reward: how 11,471 MAST plasmas "
              "ended, and what predictors learn from each label set\" (2026), doi:10.5281/zenodo.24081241.", ""]
    return "\n".join(lines)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(load=None, beams=None):
    """Write the release folder and its zip; returns the folder."""
    tabs = tables(load, beams)
    folder = out_dir()
    os.makedirs(folder, exist_ok=True)
    for name, frame in tabs.items():
        frame.to_csv(os.path.join(folder, name), index=False)
    with open(os.path.join(folder, "README.md"), "w") as f:
        f.write(readme(tabs, code_version()))
    names = sorted(list(tabs) + ["README.md"])
    if "programme_twins.csv" in tabs:  # the scorer travels with the twins, so the test can be run from the release alone
        shutil.copyfile(os.path.join(config.REPO_DIR, "scripts", "twin_score.py"), os.path.join(folder, "twin_score.py"))
        names = sorted(names + ["twin_score.py"])
    with open(os.path.join(folder, "SHA256SUMS"), "w") as f:
        f.writelines(f"{sha256(os.path.join(folder, name))}  {name}\n" for name in names)
    with zipfile.ZipFile(folder + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in names + ["SHA256SUMS"]:
            z.write(os.path.join(folder, name), arcname=os.path.join(NAME, name))
    return folder


def main():
    folder = build()
    print(f"Release written to {folder} and {folder}.zip:")
    for name in sorted(os.listdir(folder)):
        print(f"- {name}: {os.path.getsize(os.path.join(folder, name)) / 1024:,.0f} kB")
    print("\nNothing was uploaded. Read README.md before publishing.")


if __name__ == "__main__":
    main()
