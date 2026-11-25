"""Diagnostic 1 of the landmark module on the public Zhu files (docs/preregistration_landmark_zhu.md): how the stored
sampling changes with time before the disruption, from the original time stamps; cohort flow; one worked record.
No model. Descriptive.

    python scripts/schedule_vs_lead.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/schedule_vs_lead.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit as au, landmark_zhu as lz, sampling_audit as sa, timing_model as tm
from src import notify

BINS_MS = (0, 10, 20, 40, 60, 100, 150, 200, 300, 500, 1000, 2000, 5000)
BANDS_MS = ((0, 100), (100, 300), (300, 1000))


def records(frame):
    """{shot: (t, anchor, disrupted)}: anchor = disruption time for disrupted records, last slice otherwise."""
    out = {}
    for shot, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        tud = g["time_until_disrupt"].to_numpy(dtype=float)
        disrupted = bool(np.isfinite(tud).any())
        anchor = float(np.nanmedian(t + tud)) if disrupted else float(t[-1])
        out[int(shot)] = (t, anchor, disrupted)
    return out


def gap_table(recs, disrupted):
    leads, gaps, owner = [], [], []
    for shot, (t, anchor, d) in recs.items():
        if d != disrupted or len(t) < 2:
            continue
        leads.append((anchor - t[1:]) * 1e3)
        gaps.append(np.diff(t) * 1e3)
        owner.append(np.full(len(t) - 1, shot))
    lead, gap, who = np.concatenate(leads), np.concatenate(gaps), np.concatenate(owner)
    n_rec = len({s for s, v in recs.items() if v[2] == disrupted})
    rows = []
    for lo, hi in zip(BINS_MS[:-1], BINS_MS[1:]):
        m = (lead >= lo) & (lead < hi)
        if not m.any():
            rows.append(f"    {lo:>5}-{hi:<5} ms: no slices")
            continue
        g = gap[m]
        rows.append(f"    {lo:>5}-{hi:<5} ms: slices {int(m.sum()):>9,}; gap median {np.median(g):7.2f} ms "
                    f"(10%-90% {np.percentile(g, 10):6.2f}-{np.percentile(g, 90):7.2f}); records with a slice "
                    f"{len(set(who[m])) / n_rec:.0%}")
    return rows


def band_medians(recs):
    out = {band: [] for band in BANDS_MS}
    for t, anchor, d in recs.values():
        if not d or len(t) < 2:
            continue
        lead, gap = (anchor - t[1:]) * 1e3, np.diff(t) * 1e3
        for lo, hi in BANDS_MS:
            m = (lead >= lo) & (lead < hi)
            if m.any():
                out[(lo, hi)].append(np.median(gap[m]))
    return [f"    {lo}-{hi} ms: records {len(v):,}; median of per-record median gap {np.median(v):.2f} ms "
            f"(quartiles {np.percentile(v, 25):.2f}-{np.percentile(v, 75):.2f})" for (lo, hi), v in out.items() if v]


def cohort(shots, split):
    test = [s for s in shots if split.get(s) == "test"]
    rows = au.landmark_rows({s: shots[s] for s in test}, lz.WINDOW, lz.STEP, lz.HORIZON, lz.LEAD, lz.RESOLUTION)
    kept = rows[rows["label"] >= 0]
    per = kept.groupby("shot")["label"].agg(["min", "max", "size"])
    short = sum(1 for s in test if shots[s][3] - shots[s][0][0] < lz.WINDOW + lz.HORIZON)
    return (f"    test disrupted records {len(test):,}; with at least one decision {per.shape[0]:,}; with decisions of both "
            f"kinds {int(((per['min'] == 0) & (per['max'] == 1)).sum()):,}; records shorter than window + horizon "
            f"({(lz.WINDOW + lz.HORIZON) * 1e3:.0f} ms from first slice to t_d) {short:,}")


def worked(shots, split):
    test = sorted(s for s in shots if split.get(s) == "test")
    for s in test:
        t, v, _, td = shots[s]
        if td - t[0] > 1.0:
            break
    lines = [f"    shot {s}: first slice {t[0]:.4f} s, t_d {td:.4f} s, last slice {t[-1]:.4f} s",
             "    time stamps in the last 400 ms before t_d (s): " +
             ", ".join(f"{x:.4f}" for x in t[(t > td - 0.4) & (t <= td)])]
    rows = au.landmark_rows({s: shots[s]}, lz.WINDOW, lz.STEP, lz.HORIZON, lz.LEAD, lz.RESOLUTION)
    rows = rows[rows["elapsed"] + t[0] > td - 0.4]
    for r in rows.itertuples():
        L = r.elapsed + t[0]
        lines.append(f"    decision {L:.4f} s ({(td - L) * 1e3:5.0f} ms before t_d): slices {r.timing[0]:>3}, age "
                     f"{r.timing[1] * 1e3:6.2f} ms, largest gap {r.timing[2] * 1e3:6.2f} ms, label "
                     f"{ {1: 'positive', 0: 'negative', -1: 'too late'}[r.label]}")
    return lines


def run(base):
    lines = ["Landmark module, diagnostic 1 (after the result): stored sampling against time before the disruption, "
             "from the original time stamps. No model.", ""]
    fig4 = os.path.join(base, "20ja072_fig4_data")
    for name, file, official in (("EAST", "EAST_for_nn.mat", True), ("DIII-D", "d3d_for_nn.mat", False),
                                 ("C-Mod", "C_Mod_for_nn.mat", False)):
        path = os.path.join(fig4, file)
        if not os.path.exists(path):
            continue
        print(f"{name} ...", flush=True)
        frame = sa.load(path)
        signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
        recs = records(frame)
        lines.append(f"== {name}")
        lines.append("  1. gap between consecutive slices by time before t_d, DISRUPTED records:")
        lines += gap_table(recs, True)
        lines.append("     same by time before the last slice, NON-DISRUPTED records:")
        lines += gap_table(recs, False)
        lines.append("  2. per disrupted record, median gap by band before t_d:")
        lines += band_medians(recs)
        table = frame.groupby("shot")["time_until_disrupt"].apply(lambda x: bool(np.isfinite(x).any())).rename(
            "disrupted").reset_index()
        if official:
            split, _ = tm.official_split(os.path.join(base, "20ja072_fig6_data", "EAST_data_partition.mat"), table)
        else:
            split = tm.our_split(table)
        shots = lz.disrupted_shots(frame, signals)
        lines.append("  3. cohort flow of the landmark analysis:")
        lines.append(cohort(shots, split))
        if official:
            lines.append("  4. worked record (first EAST test record longer than 1 s):")
            lines += worked(shots, split)
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "schedule_vs_lead.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    for i in range(0, len(out), 3400):
        notify.send(f"Schedule vs lead ({i // 3400 + 1})\n\n" + out[i:i + 3400])


if __name__ == "__main__":
    main()
