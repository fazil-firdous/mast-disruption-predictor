"""Accounting for the saddle-sign correction, from saved outputs only (no new model). Requested as a check
after the corrected result (NOTES.md 2026-11-15); computed after that result, so every line is a clarification or a
labelled post-hoc secondary, never a test that could change the reading.

1. How the 1,641,720 "changed values" are counted: per lm column, finite values that differ, finite to missing,
   missing to finite, unchanged finite, unchanged missing; and whether has_lm changed anywhere.
2. Denominators: eligible pair-split rows per split at 10 ms, unique pairs and lost shots, the subgroup counts, and how
   the 17,094 (split, pair, lead) entries relate to them.
3. Post-hoc secondary (labelled): the paired chronological contrast A_c minus L on common pairs, from the saved
   predictions, with the same paired resampling as every pooled range.

    python scripts/saddle_ledger.py

Writes RESULTS_DIR/saddle_correction/ledger.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import how_early, no_lm, saddle_correction as sc, saddle_fetch, v3_explore
from src import experiment, notify

KINDS = ("finite, value differs", "finite to missing", "missing to finite", "unchanged finite", "unchanged missing")


def change_counts(original, corrected):
    """Counts per KINDS for two aligned float arrays (same windows)."""
    a, b = np.asarray(original, dtype=float), np.asarray(corrected, dtype=float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    return {KINDS[0]: int((both & (a != b)).sum()), KINDS[1]: int((fa & ~fb).sum()),
            KINDS[2]: int((~fa & fb).sum()), KINDS[3]: int((both & (a == b)).sum()), KINDS[4]: int((~fa & ~fb).sum())}


def feature_accounting(windows, corrected):
    columns = sc.lm_columns(windows.columns)
    merged = windows[["shot_id", "t_end"] + columns].merge(corrected[["shot_id", "t_end"] + columns],
                                                           on=["shot_id", "t_end"], suffixes=("", "_c"))
    lines = [f"1. Feature accounting: {merged['shot_id'].nunique():,} corrected shots, {len(merged):,} windows x "
             f"{len(columns)} columns = {len(merged) * len(columns):,} cells."]
    total = dict.fromkeys(KINDS, 0)
    for c in columns:
        n = change_counts(merged[c], merged[f"{c}_c"])
        total = {k: total[k] + n[k] for k in KINDS}
        lines.append(f"- {c}: " + "; ".join(f"{k} {v:,}" for k, v in n.items()))
    lines.append("- all: " + "; ".join(f"{k} {v:,}" for k, v in total.items()))
    has = change_counts(merged["has_lm"], merged["has_lm_c"]) if "has_lm" in columns else None
    if has is not None:
        changed = has[KINDS[0]] + has[KINDS[1]] + has[KINDS[2]]
        lines.append(f"- has_lm changed in {changed:,} windows (a polarity-only correction should change none).")
    lines.append("The earlier figure (1,641,720) counted cells where numpy.isclose is false with NaN = NaN, i.e. the first "
                 "three kinds plus finite values beyond isclose's default tolerance.")
    return lines


def denominators(random, temporal, lead=sc.PRIMARY_LEAD):
    main = random[(random["kind"] == "main") & random["cohort"].astype(bool)]
    mine = main[(main["recipe"] == sc.RECIPE) & (main["lead_ms"] == lead)]
    per = mine.groupby("seed").size()
    lines = ["", f"2. Denominators (twin test, common cohort, model A_c; the other models have the same pairs, G5):",
             f"- {lead} ms: {len(mine):,} pair-split rows over {per.size} splits (per split min {per.min()}, median "
             f"{per.median():.0f}, max {per.max()}); {mine[['shot_id', 'other']].drop_duplicates().shape[0]:,} unique "
             f"pairs; {mine['shot_id'].nunique():,} unique lost shots; {mine['other'].nunique():,} unique twins. A pair "
             "or a shot appears in several splits and a lost shot can have more than one twin: rows are not "
             "independent observations, which is why ranges come from the splits and from resampling groups of pairs.",
             f"- Subgroups: lost shots in the cohort at any lead, any split, any model: before "
             f"{saddle_fetch.FLIP_FROM} {main.loc[main['shot_id'] < saddle_fetch.FLIP_FROM, 'shot_id'].nunique():,}, from "
             f"{saddle_fetch.FLIP_FROM} {main.loc[main['shot_id'] >= saddle_fetch.FLIP_FROM, 'shot_id'].nunique():,} "
             "(unique lost shots, not rows)."]
    every = pd.concat([random[(random["kind"] == "main") & (random["recipe"] == sc.RECIPE)],
                       temporal[(temporal["kind"] == "main") & (temporal["recipe"] == sc.RECIPE)]])
    lines.append(f"- All (split, pair, lead) rows of A_c, every lead and the chronological split, before the cohort "
                 f"filter: {len(every):,} (the G5 manifest counted the same population from the test windows: 17,094). "
                 f"Leads: {sorted(every['lead_ms'].unique().tolist())}; splits: {every['seed'].nunique()}.")
    return lines


def chronological_contrast(temporal):
    t_main = temporal[(temporal["kind"] == "main") & temporal["cohort"].astype(bool)]
    lines = ["", "3. Post-hoc secondary, labelled (asked for after the result; no new model): chronological split, paired "
             "on common pairs, conditional on the one fitted model of each:"]
    for lead in (sc.PRIMARY_LEAD, 30):
        part = t_main[t_main["lead_ms"] == lead]
        d = v3_explore.pooled_delta(part, sc.RECIPE, no_lm.RECIPE, "win")
        if d:
            lines.append(f"- {lead} ms, A_c minus L: {how_early.rng(*d[:3], signed=True)} on {d[3]:,} pairs")
    lines.append("A difference between two separate intervals is not this contrast; only this line speaks to A_c "
                 "against L.")
    return lines


def main():
    windows, _ = experiment.load_tables()
    corrected = pd.read_parquet(sc.features_path())
    random, temporal = sc.collect_wins()
    lines = (["Saddle-sign correction, accounting (scripts/saddle_ledger.py; after the result, clarifications only)", ""]
             + feature_accounting(windows, corrected) + denominators(random, temporal) + chronological_contrast(temporal))
    text = "\n".join(lines)
    with open(os.path.join(sc.folder(), "ledger.txt"), "w") as f:
        f.write(text + "\n")
    print(text, flush=True)
    notify.send(text[:3500])


if __name__ == "__main__":
    main()
