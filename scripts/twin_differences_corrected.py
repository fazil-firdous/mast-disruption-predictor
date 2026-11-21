"""The model-free twin differences (scripts/twin_differences.py) for the locked-mode signal with the documented
saddle-sign correction (docs/preregistration_saddle_correction.md, amendment 3): the same triplets, the same code,
with the window means of lm taken from the corrected features. Other signals are unchanged by the correction.

    python scripts/twin_differences_corrected.py

Writes RESULTS_DIR/twin_differences/corrected_lm.txt. Reads the saved triplets; outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import saddle_correction as sc, twin_differences as td, v3_runs
from src import config, experiment, extras, notify


def run():
    triplets = pd.read_csv(os.path.join(td.OUT_DIR, "triplets.csv"))
    labels = pd.read_csv(td.LABELS_CSV)
    load = lambda shot: extras.load_extras(int(shot), allow_download=False)  # noqa: E731
    shots = set(triplets["shot_id"]) | set(triplets["k1"]) | set(triplets["k2"])
    windows, _ = experiment.load_tables()
    windows = windows[windows["shot_id"].isin(shots)].reset_index(drop=True)
    corrected = pd.read_parquet(sc.features_path())
    corrected = corrected[corrected["shot_id"].isin(shots)]
    fixed = sc.swap(windows, corrected)
    t_loss = labels.set_index("shot_id")["t_quench"].to_dict()
    t_start = {shot: v3_runs.flat_top_start(load(shot)) for shot in triplets["shot_id"]}
    out = {}
    for name, frame in (("original", windows), ("corrected", fixed)):
        by = td.by_shot(frame[["shot_id", "t_end"] + [f"{s}_mean" for s in td.SIGNALS]])
        leads = td.lead_table(td.collect(triplets, by, t_loss, t_start), np.random.default_rng(config.RANDOM_SEED))
        out[name] = leads[leads["signal"] == "lm"].set_index("lead_ms")
    saved = os.path.join(td.OUT_DIR, "leads.csv")
    lines = ["Twin differences, locked-mode signal (window mean of the first-harmonic saddle-array feature), original "
             "and with the documented sign correction; same triplets and code. D = share of triplets with |L-K1| > "
             "|K1-K2|; up = share with L above K1 (and K1 above K2).", ""]
    for lead in td.LEADS_MS:
        if lead in out["original"].index and lead in out["corrected"].index:
            o, c = out["original"].loc[lead], out["corrected"].loc[lead]
            lines.append(f"- {lead} ms: D original {o['D']:.3f} ({o['D_low']:.3f} to {o['D_high']:.3f}), corrected "
                         f"{c['D']:.3f} ({c['D_low']:.3f} to {c['D_high']:.3f}); up L>K1 {o['up_case']:.3f} -> "
                         f"{c['up_case']:.3f}, K1>K2 {o['up_ctrl']:.3f} -> {c['up_ctrl']:.3f}; {int(c['n']):,} triplets")
    if os.path.exists(saved):
        s = pd.read_csv(saved)
        s = s[s["signal"] == "lm"].set_index("lead_ms")
        same = all(abs(s.loc[lead, "D"] - out["original"].loc[lead, "D"]) < 1e-9 for lead in s.index
                   if lead in out["original"].index)
        lines.append(f"Check: the original column reproduces the saved leads.csv: {same}")
    return "\n".join(lines)


def main():
    text = run()
    with open(os.path.join(td.OUT_DIR, "corrected_lm.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    notify.send("Twin differences, locked-mode line corrected\n\n" + text[:3300])


if __name__ == "__main__":
    main()
