"""Follow-up diagnostic 1 of the endpoint module (docs/preregistration_endpoint.md, written after the first result and
before this was computed): common cohort at 10 and 100 ms, timing of the selected windows, changes in pair order,
phase of each point from the requested programme only, and ties. Saved runs only; descriptive.

    python scripts/endpoint_diagnostic.py

Writes RESULTS_DIR/endpoint/diagnostic.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import endpoint_audit as ea, saddle_correction as sc
from src import config, extras, notify

LEADS = (10, 100)
FLAT = 0.90  # the twin paper's flat-top rule, not tuned here


def programme_phases(t_ref, ref):
    """(start of flat-top, start of ramp-down) of a requested programme: the first and the last time it is at or above
    FLAT of its peak. (NaN, NaN) without a usable programme."""
    ok = np.isfinite(t_ref) & np.isfinite(ref)
    if ok.sum() < 3:
        return np.nan, np.nan
    t, r = t_ref[ok], np.abs(ref[ok])
    above = np.where(r >= FLAT * r.max())[0]
    return (float(t[above[0]]), float(t[above[-1]])) if len(above) else (np.nan, np.nan)


def phase(t, bounds):
    t0, t1 = bounds
    if not (np.isfinite(t) and np.isfinite(t0) and np.isfinite(t1)):
        return "unknown"
    return "ramp-up" if t < t0 else "ramp-down" if t > t1 else "flat-top"


def transitions(part):
    """Counts of the pair order against the case going from P to E: (P result, E result) -> rows."""
    names = {1.0: "case higher", 0.5: "tie", 0.0: "twin higher"}
    return part.groupby(["win_p", "win_e"]).size().rename(index=names).to_dict()


def text(frame, bounds):
    lines = ["Endpoint module, follow-up diagnostic 1 (written after the first result, before this was computed). "
             "Descriptive.", ""]
    a = frame[frame["model"] == sc.RECIPE]
    keys = ["seed", "shot_id", "other"]
    sets = {lead: set(map(tuple, a.loc[a["lead_ms"] == lead, keys].to_numpy())) for lead in LEADS}
    both = sets[LEADS[0]] & sets[LEADS[1]]
    lines.append(f"1. Pair-split rows (A_c): 10 ms {len(sets[10]):,}, 100 ms {len(sets[100]):,}, both {len(both):,}; "
                 f"only at 10 ms {len(sets[10] - sets[100]):,}, only at 100 ms {len(sets[100] - sets[10]):,}.")
    inter = a[[tuple(k) in both for k in a[keys].to_numpy()]]
    for lead in LEADS:
        part = inter[inter["lead_ms"] == lead]
        r = ea.over_splits(part.assign(model=sc.RECIPE), sc.RECIPE, lead)
        pp = ea.pooled(part.assign(model=sc.RECIPE), sc.RECIPE, lead)
        if r:
            lines.append(f"   on the intersection, {lead} ms: C_P {r[1]:.3f}, C_E {r[2]:.3f}, Delta {ea.fmt(r[0])} over "
                         f"splits | pooled {ea.fmt(pp)}")
    lines.append("")
    lines.append("2. Timing on the intersection (ms): nominal gap E - P, and actual gap between the selected window ends")
    for lead in LEADS:
        part = inter[inter["lead_ms"] == lead]
        nominal = (part["t_e"] - part["t_p"]) * 1e3
        actual = (part["sel_e"] - part["sel_p"]) * 1e3
        lines.append(f"   {lead} ms: nominal median {nominal.median():.0f} (quartiles {nominal.quantile(0.25):.0f} to "
                     f"{nominal.quantile(0.75):.0f}); actual median {actual.median():.0f}; largest |actual - nominal| "
                     f"{(actual - nominal).abs().max():.1f}")
    lines.append("")
    lines.append("3. Orderings on the intersection: twin score change s_E - s_P, and pair order from P to E (rows)")
    for lead in LEADS:
        part = inter[inter["lead_ms"] == lead]
        change = part["twin_e"] - part["twin_p"]
        lines.append(f"   {lead} ms: s_E - s_P mean {change.mean():+.4f}, median {change.median():+.4f}, share negative "
                     f"{(change < 0).mean():.1%}; case score mean {part['case'].mean():.4f}; transitions "
                     f"{transitions(part)}")
    lines.append("")
    lines.append("4. Phase from the requested programme only (share of rows): case point | twin at P | twin at E")
    for lead in LEADS:
        part = inter[inter["lead_ms"] == lead]
        cols = []
        for who, tcol in (("shot_id", "sel_l"), ("other", "sel_p"), ("other", "sel_e")):
            ph = pd.Series([phase(t, bounds.get(int(s), (np.nan, np.nan))) for s, t in zip(part[who], part[tcol])])
            cols.append(", ".join(f"{k} {v:.0%}" for k, v in ph.value_counts(normalize=True).items()))
        lines.append(f"   {lead} ms: {cols[0]} | {cols[1]} | {cols[2]}")
    lines.append("")
    lines.append("5. Ties (share of rows), P | E, all eligible rows:")
    for model in ea.MODELS:
        for lead in LEADS:
            part = frame[(frame["model"] == model) & (frame["lead_ms"] == lead)]
            if len(part):
                lines.append(f"   {ea.NAMES[model]}, {lead} ms: {(part['win_p'] == 0.5).mean():.1%} | "
                             f"{(part['win_e'] == 0.5).mean():.1%}")
    lines += ["", "Reading: descriptive. A phase difference between leads is consistent with, not proof of, a phase "
              "explanation; 'same programme time', not 'same plasma phase'; the 100 ms sign change is not an "
              "established reversal."]
    return "\n".join(lines)


def main():
    frame = ea.collect()
    shots = set(frame["shot_id"]) | set(frame["other"])
    bounds = {}
    for s in shots:
        e = extras.load_extras(int(s), allow_download=False)
        bounds[int(s)] = (programme_phases(e["time"].to_numpy(dtype=float), e["ip_ref"].to_numpy(dtype=float))
                          if e is not None and "ip_ref" in e else (np.nan, np.nan))
    out = text(frame, bounds)
    folder = os.path.join(config.RESULTS_DIR, "endpoint")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "diagnostic.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Endpoint module, diagnostic 1\n\n" + out[:3400])


if __name__ == "__main__":
    main()
