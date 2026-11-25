"""Diagnostic 2 of the landmark module on the public Zhu files (docs/preregistration_landmark_zhu.md): is
time + time_until_disrupt constant within each disrupted record, and how far is T from the last stored slice, record
by record? No model. Descriptive.

    python scripts/anchor_check.py /root/dv/var/www/html/reports/20JA/20JA72

Writes RESULTS_DIR/sampling_audit/anchor_check.txt. Outside every stamp.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sampling_audit as sa
from src import notify

TOLS_S = (1e-6, 1e-3, 1e-2)


def per_record(frame):
    """(spread of time + tud within the record, T - last slice, slices with a non-finite tud) per disrupted record."""
    spread, gap, missing = [], [], []
    for _, g in frame.groupby("shot", sort=True):
        tud = g["time_until_disrupt"].to_numpy(dtype=float)
        if not np.isfinite(tud).any():
            continue
        t = g["time"].to_numpy(dtype=float)
        s = (t + tud)[np.isfinite(tud)]
        spread.append(float(s.max() - s.min()))
        gap.append(float(np.median(s) - t.max()))
        missing.append(int((~np.isfinite(tud)).sum()))
    return np.array(spread), np.array(gap), np.array(missing)


def lines_for(name, spread, gap, missing):
    out = [f"== {name}: disrupted records {len(gap):,}"]
    out.append(f"  time + time_until_disrupt within a record: max - min median {np.median(spread) * 1e3:.4f} ms, "
               f"largest {spread.max() * 1e3:.4f} ms; records with spread <= " +
               ", ".join(f"{tol * 1e3:g} ms {np.mean(spread <= tol):.1%}" for tol in TOLS_S))
    out.append(f"  T - last slice (signed): median {np.median(gap) * 1e3:+.4f} ms, 1st-99th percentile "
               f"{np.percentile(gap, 1) * 1e3:+.3f} to {np.percentile(gap, 99) * 1e3:+.3f} ms, largest |.| "
               f"{np.abs(gap).max() * 1e3:.3f} ms; records within " +
               ", ".join(f"{tol * 1e3:g} ms {np.mean(np.abs(gap) <= tol):.1%}" for tol in TOLS_S))
    out.append(f"  disrupted records with some non-finite time_until_disrupt slices: {int((missing > 0).sum()):,} "
               f"(slices {int(missing.sum()):,})")
    return out


def run(base):
    lines = ["Landmark module, diagnostic 2 (after the result): the reconstructed disruption time T, record by record. "
             "No model.", ""]
    for name, file in (("EAST", "EAST_for_nn.mat"), ("DIII-D", "d3d_for_nn.mat"), ("C-Mod", "C_Mod_for_nn.mat")):
        path = os.path.join(base, "20ja072_fig4_data", file)
        if os.path.exists(path):
            print(f"{name} ...", flush=True)
            lines += lines_for(name, *per_record(sa.load(path))) + [""]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "anchor_check.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send("Anchor check\n\n" + out[:3400])


if __name__ == "__main__":
    main()
