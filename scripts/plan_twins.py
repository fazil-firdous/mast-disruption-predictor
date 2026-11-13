"""Plan twins: the twin test on pairs whose two shots also asked for the same density, height and gas.

Fixed in docs/preregistration_v3.md ("2026-11-13: plan twins") before any score was looked at on these pairs.
Reads programme_signals/pairs.csv (scripts/programme_signals.py), v3_explore/twin_pairs.csv (scripts/v3_explore.py
--summary) and the saved scores of the chronological split. No training.

    python scripts/plan_twins.py

Writes RESULTS_DIR/plan_twins/ (pairs.csv with the two flags, summary.txt). Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import paper2_numbers, programme_signals, twin_differences, v3_explore, v3_runs
from src import config, server

SAME = 0.01  # under 1% relative difference counts as the same request
REQUESTS = ("density_ref", "height_ref", "valves_ref")
MODELS = ("baseline", "machine", "programme", "solenoid")
NAMES = {"baseline": "full", "machine": "machine-side", "programme": "programme-only", "solenoid": "solenoid-only"}


def folder():
    return os.path.join(config.RESULTS_DIR, "plan_twins")


def matched(state, rms):
    """True where a pair is matched in one request: it changes in both shots with a relative difference under SAME,
    or both shots hold the same constant."""
    rms = pd.to_numeric(rms, errors="coerce")
    return (state == "both_constant") | ((state == "both") & (rms < SAME))


def flags(compared):
    """compared (programme_signals/pairs.csv) with 'plan' (matched in density and height, and in the valve requests
    where both shots have them) and 'full_plan' (matched in all three, valve requests present in both)."""
    out = compared[["shot_id", "other"]].copy()
    m = {name: matched(compared[f"{name}_state"], compared[f"{name}_rms"]) for name in REQUESTS}
    valves_absent = compared["valves_ref_state"] == "missing"
    out["plan"] = m["density_ref"] & m["height_ref"] & (m["valves_ref"] | valves_absent)
    out["full_plan"] = m["density_ref"] & m["height_ref"] & m["valves_ref"]
    return out


def c_lines(part, label):
    """One block: counts, every model's C with both statistics and both ranges, and full minus machine-side."""
    unique = part.drop_duplicates(["shot_id", "other"])
    lines = [f"{label}: {len(unique):,} pairs, {unique['other'].nunique():,} different twins"]
    for value, name in (("win", "registered statistic"), ("win_common", "same-time statistic")):
        table = v3_explore.c_table(part, value=value).set_index("recipe")
        pooled = v3_explore.pooled_table(part, value).set_index("recipe")
        cells = [f"{NAMES[m]} {v3_explore.cell(table.loc[m])} | pooled {pooled.loc[m, 'C']:.3f} "
                 f"({pooled.loc[m, 'low']:.3f} to {pooled.loc[m, 'high']:.3f})"
                 for m in MODELS if m in table.index and m in pooled.index]
        d_split = paper2_numbers.paired(part, "baseline", "machine", value)
        d_pool = v3_explore.pooled_delta(part, "baseline", "machine", value)
        if d_split and d_pool:
            cells.append(f"full minus machine-side: splits {paper2_numbers.rng_text(*d_split[:3])} | pooled "
                         f"{paper2_numbers.rng_text(*d_pool[:3])}")
        lines.append(f"- {name}:")
        lines += [f"    {c}" for c in cells]
    return lines


def temporal_wins(load=None):
    """The chronological split's pair wins at the registered lead, one row per (model, pair), or an empty frame."""
    from src import extras
    load = load or (lambda shot: extras.load_extras(int(shot), allow_download=False))
    twins = pd.read_csv(v3_runs.TWINS)
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    t_quench = labels.set_index("shot_id")["t_quench"].to_dict()
    starts, frames = {}, []
    for recipe in v3_explore.TEMPORAL:
        base = v3_explore.temporal_dir(recipe)
        if not server.result_saved(os.path.join(base, "metrics.json")):
            continue
        scores = pd.read_parquet(os.path.join(base, "test_scores.parquet"))
        ids = set(scores["shot_id"])
        mine = twins[twins["shot_id"].isin(ids) & twins["other"].isin(ids)]
        for shot in mine["shot_id"]:
            if shot not in starts:
                starts[shot] = v3_runs.flat_top_start(load(shot))
        wins = v3_explore.pair_wins(scores, mine, t_quench, starts, leads_ms=(config.MIN_WARNING_MS,))
        frames.append(wins.assign(recipe=recipe, seed=0))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def temporal_lines(wins, keep, label):
    """The chronological split on one set of pairs, pooled ranges only."""
    if len(wins) == 0:
        return [f"- {label}: no chronological runs"]
    part = wins.merge(keep, on=["shot_id", "other"])
    cells = []
    for m in v3_explore.TEMPORAL:
        ok = part[(part["recipe"] == m) & part["win_common"].notna()]
        if len(ok):
            c, low, high, n = twin_differences.cluster_range(ok["win_common"].to_numpy(dtype=float),
                                                             ok["other"].to_numpy(),
                                                             np.random.default_rng(config.RANDOM_SEED), v3_explore.N_BOOT)
            cells.append(f"{NAMES[m]} {c:.3f} ({low:.3f} to {high:.3f}), {n:,} pairs")
    d = v3_explore.pooled_delta(part, "baseline", "machine", "win_common")
    if d:
        cells.append(f"full minus machine-side {paper2_numbers.rng_text(*d[:3])}")
    return [f"- {label} (same-time statistic): " + "; ".join(cells)]


def composition(main, plan, campaign):
    """How plan twins differ from the other twins in who is in them: campaign, recorded or candidate lost shot,
    beam heating. One line per set; shares of distinct pairs. No score."""
    pairs = main.drop_duplicates(["shot_id", "other"]).merge(plan, on=["shot_id", "other"], how="left")
    pairs["campaign"] = pairs["shot_id"].map(campaign).fillna("Unknown")
    sets = (("plan twins", pairs[pairs["plan"].fillna(False).astype(bool)]),
            ("full-plan twins", pairs[pairs["full_plan"].fillna(False).astype(bool)]),
            ("not plan twins", pairs[~pairs["plan"].fillna(False).astype(bool)]), ("all tested twins", pairs))
    campaigns = sorted(pairs["campaign"].astype(str).unique())
    lines = ["Who is in each set (distinct tested pairs; shares):"]
    for name, part in sets:
        n = max(len(part), 1)
        by = part["campaign"].astype(str).value_counts()
        heat = part["heating"].value_counts() if "heating" in part else pd.Series(dtype=float)
        lines.append(f"- {name}: {len(part):,} pairs; recorded {(part['group'] == 'recorded').mean():.0%}; heating same "
                     f"{heat.get('same', 0) / n:.0%}, different {heat.get('different', 0) / n:.0%}, unknown "
                     f"{heat.get('unknown', 0) / n:.0%}; " + ", ".join(f"{c} {by.get(c, 0) / n:.0%}" for c in campaigns))
    return lines


def text(main, plan, temporal, campaign=None):
    """The analysis in words, in the order of the pre-registration."""
    n = len(plan)
    lines = ["Plan twins (docs/preregistration_v3.md, 2026-11-13). Exploratory. Matched in a request: changes in both "
             f"shots with a relative difference under {SAME:.0%} up to the loss, or the same constant in both.",
             f"Of {n:,} twin pairs: plan twins {int(plan['plan'].sum()):,} ({plan['plan'].mean():.0%}); full-plan twins "
             f"{int(plan['full_plan'].sum()):,} ({plan['full_plan'].mean():.0%}).", ""]
    if campaign is not None:
        lines += composition(main, plan, campaign) + [""]
    keys = plan.set_index(["shot_id", "other"])
    flagged = main.join(keys, on=["shot_id", "other"])
    for column, label in (("plan", "PLAN TWINS (primary)"), ("full_plan", "FULL-PLAN TWINS")):
        lines += c_lines(flagged[flagged[column].fillna(False).astype(bool)], label) + [""]
    lines += c_lines(flagged[~flagged["plan"].fillna(False).astype(bool)], "Twins that are not plan twins") + [""]
    lines += c_lines(main, "All twins (known, for comparison)") + [""]
    lines.append("Chronological split (trained on M5-M7, tested on M8-M9), pooled ranges:")
    for column, label in (("plan", "plan twins"), ("full_plan", "full-plan twins")):
        lines += temporal_lines(temporal, plan.loc[plan[column], ["shot_id", "other"]], label)
    lines += temporal_lines(temporal, plan[["shot_id", "other"]], "all twins (known)")
    return "\n".join(lines)


def main():
    compared_path = os.path.join(programme_signals.folder(), "pairs.csv")
    pairs_path = os.path.join(v3_explore.folder(), "twin_pairs.csv")
    for path, how in ((compared_path, "scripts/programme_signals.py"), (pairs_path, "scripts/v3_explore.py --summary")):
        if not os.path.exists(path):
            sys.exit(f"missing {path}: run {how} first")
    plan = flags(pd.read_csv(compared_path))
    pairs = pd.read_csv(pairs_path)
    main_pairs = pairs[pairs["lead_ms"] == config.MIN_WARNING_MS]
    from src import data
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
    out = text(main_pairs, plan, temporal_wins(), campaign)
    os.makedirs(folder(), exist_ok=True)
    plan.to_csv(os.path.join(folder(), "pairs.csv"), index=False)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
