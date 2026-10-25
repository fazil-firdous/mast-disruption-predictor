"""Independent evidence for the automatic termination labels: signals the rules never use, compared between
operator-noted flat-top drops (recorded disruptions), un-noted flat-top drops (the candidates) and un-noted
flux-exhausted endings (ordinary ends). Candidates that were really unrecorded disruptions should look like the first
group; ordinary ends that the rules mistook, like the last.

    python scripts/termination_evidence.py    # needs the cached shots and RESULTS_DIR/auto_labels/per_shot.csv

Measures at the detector's quench time t_q (flat-top = programmed current >= AUTO_FLATTOP_FRAC of its maximum, up to
20 ms before t_q):
- fall_ms: |Ip| from 90% to 10% of its peak on the final fall (label_report.fall_time_ms). Only partly independent:
  a current already decaying before the quench, as when the solenoid has stopped, lengthens it;
- neutrons_rel, prad_rel: mean over the last 10 ms before t_q / flat-top median (is the heating still on?);
- mirnov_rel, lm_rel: the same over the last 20 ms for the Mirnov fluctuation level and the saddle-coil n=1 amplitude;
- ip_spike_pct: highest |Ip| in the 3 ms before t_q above its mean in the 10 ms before that.
An analysis next to the experiments: no experiment uses it, so it is outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.label_report import fall_time_ms
from scripts.label_statistics import auc
from src import config, extras, labels
from src.data import is_cached, load_shot

LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
OUT_DIR = os.path.join(config.RESULTS_DIR, "termination_evidence")
GROUPS = {"recorded disruption": (True, "flat_top_drop"), "candidate": (False, "flat_top_drop"),
          "ordinary end": (False, "flux_exhausted")}
MEASURES = ["fall_ms", "neutrons_rel", "prad_rel", "mirnov_rel", "lm_rel", "ip_spike_pct"]
FAST_FALL_MS = 10  # a current fall within 10 ms counts as fast, the timescale of the quench detector


def window_mean(t, y, t0, t1):
    """Mean of the finite samples of y with t0 <= t <= t1, NaN if none."""
    keep = (t >= t0) & (t <= t1) & np.isfinite(y)
    return float(np.mean(y[keep])) if keep.any() else np.nan


def relative(t, y, flat, t_q, width):
    """Mean of y over the last width seconds before t_q, divided by its flat-top median; NaN if either is missing."""
    base = np.nanmedian(y[flat]) if flat.any() and np.isfinite(y[flat]).any() else np.nan
    return window_mean(t, y, t_q - width, t_q) / base if np.isfinite(base) and base > 0 else np.nan


def shot_evidence(shot, t_q):
    """The measures of one shot from its summary signals with the cached extras merged in (extras.with_extras)."""
    t = shot["time"].to_numpy()
    ref = shot["ip_ref"].to_numpy()
    flat = np.zeros(len(t), bool)
    if np.isfinite(ref).any():
        flat = (ref >= config.AUTO_FLATTOP_FRAC * np.nanmax(ref)) & (t <= t_q - 0.02)
    ip = labels.median_filter(np.abs(shot["ip"].to_numpy()))
    before = window_mean(t, ip, t_q - 0.013, t_q - 0.003)
    near = (t >= t_q - 0.003) & (t <= t_q)
    spike = ip[near].max() if near.any() else np.nan
    return {"fall_ms": fall_time_ms(t, shot["ip"].to_numpy()),
            "neutrons_rel": relative(t, shot["neutron_rates_total"].to_numpy(), flat, t_q, 0.010),
            "prad_rel": relative(t, shot["power_radiated"].to_numpy(), flat, t_q, 0.010),
            "mirnov_rel": relative(t, shot["mirnov_rms"].to_numpy(), flat, t_q, 0.020),
            "lm_rel": relative(t, np.abs(shot["lm_n1"].to_numpy()), flat, t_q, 0.020),
            "ip_spike_pct": 100 * (spike / before - 1) if before > 0 else np.nan}


def evidence_table(table):
    """One row per shot of the three groups whose summary signals and extras are both cached."""
    rows = []
    for group, (note, cls) in GROUPS.items():
        for row in table[(table["note"] == note) & (table["termination"] == cls)].itertuples():
            shot_id = int(row.shot_id)
            if not (is_cached(shot_id) and os.path.exists(extras.cache_path(shot_id))):
                continue
            shot = extras.with_extras(load_shot(shot_id), shot_id)
            rows.append({"shot_id": shot_id, "group": group, **shot_evidence(shot, row.t_quench)})
    return pd.DataFrame(rows)


def compare(evidence):
    """Per measure: the median of each group, and AUCs (with Mann-Whitney p) for recorded > ordinary,
    candidate > ordinary and recorded > candidate. An AUC near 0.5 means the measure cannot tell the groups apart."""
    rows = []
    pick = {g: evidence[evidence["group"] == g] for g in GROUPS}
    for m in MEASURES:
        row = {"measure": m, **{f"median {g}": pick[g][m].median() for g in GROUPS}}
        for a, b in [("recorded disruption", "ordinary end"), ("candidate", "ordinary end"),
                     ("recorded disruption", "candidate")]:
            x, y = pick[a][m].dropna().to_numpy(), pick[b][m].dropna().to_numpy()
            if len(x) and len(y):
                row[f"AUC {a} > {b}"] = auc(x, y)
                row[f"p {a} vs {b}"] = stats.mannwhitneyu(x, y).pvalue
        rows.append(row)
    return pd.DataFrame(rows)


def summary_text(evidence, table):
    """The comparison in a few lines."""
    counts = evidence["group"].value_counts()
    fast = evidence.assign(fast=evidence["fall_ms"] <= FAST_FALL_MS).groupby("group")["fast"].mean()
    lines = [f"Independent evidence on {len(evidence)} cached shots: " + ", ".join(
        f"{counts.get(g, 0)} {g}s" for g in GROUPS) + "."]
    for _, r in table.iterrows():
        lines.append(f"- {r['measure']}: medians " + " / ".join(f"{r[f'median {g}']:.2f}" for g in GROUPS)
                     + f" (recorded / candidate / ordinary); AUC recorded>ordinary "
                     f"{r.get('AUC recorded disruption > ordinary end', np.nan):.2f}, candidate>ordinary "
                     f"{r.get('AUC candidate > ordinary end', np.nan):.2f}")
    lines.append(f"- current falls within {FAST_FALL_MS} ms: " + " / ".join(f"{100 * fast.get(g, np.nan):.0f}%"
                                                                            for g in GROUPS))
    return "\n".join(lines)


def main():
    evidence = evidence_table(pd.read_csv(LABELS_CSV))
    table = compare(evidence)
    os.makedirs(OUT_DIR, exist_ok=True)
    evidence.to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)
    table.to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)
    text = summary_text(evidence, table)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
