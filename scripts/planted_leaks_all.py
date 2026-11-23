"""Planted-leak validation of the all-disruptive mode (docs/preregistration_planted_leaks_all.md, fixed before any
run): four synthetic families (clean, clock shortcut, sampling shortcut, harmless drift), 20 replicates each; the
sampling flag and the clock-matched flag. Needs no data.

    python scripts/planted_leaks_all.py

Writes RESULTS_DIR/toolkit/planted_leaks_all.txt and .csv. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit as au, planted_leaks as pl
from src import config, notify

N_SHOTS, N_SIG = 200, 4
FAMILIES = ("A0", "A_C", "A_S", "A_W")
FLAGS = (("flag_time", "sampling flag", "a_time"), ("flag_clock_match", "clock-matched flag", "delta"))
EXPECTED = {"A0": set(), "A_C": {"flag_clock_match"}, "A_S": {"flag_time"}, "A_W": set()}
SETTINGS = dict(window=0.020, step=0.005, horizon=0.030, lead=0.002, tol=0.010)
SEEDS = range(1, 21)


def generate(family, seed):
    """{shot: (t, values, True, t_disrupt)}."""
    rng = np.random.default_rng(seed)
    shots = {}
    for k in range(N_SHOTS):
        d = rng.uniform(0.18, 0.24) if family == "A_C" else rng.uniform(0.10, 0.50)
        if family == "A_S":
            t = np.r_[np.arange(0.0, d - 0.030, 0.001), np.arange(d - 0.030, d, 0.00025)]
        else:
            t = np.arange(0.0, d, 0.001)
        t = np.round(t, 7)
        x = pl.ar1(rng, len(t))[:, :N_SIG]
        x[:, 0] += 0.5 * np.clip(1 - (t[-1] - t) / 0.030, 0, None)
        if family in ("A_C", "A_W"):
            x[:, 1] += 10.0 * t
        shots[70000 + k] = (t, x, True, float(t[-1]))
    return shots


def run(seeds=SEEDS):
    rows = []
    for family in FAMILIES:
        for seed in seeds:
            r = au.all_disruptive_stats(generate(family, seed), **SETTINGS)
            rows.append({"family": family, "seed": seed, "flag_time": bool(r["flag_time"]),
                         "flag_clock_match": bool(r["flag_clock_match"]), "a_time": r["a_time"], "delta": r["delta"][0],
                         "low": r["delta"][1], "high": r["delta"][2], "a_clock": r["a_clock"], "a_val": r["a_val"],
                         "c_all": r["c_all"], "c_clock": r["c_clock"], "matched": r["matched"], "n_pos": r["n_pos"]})
            print(f"{family} seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def table(frame):
    n = frame["seed"].nunique()
    lines = [f"Fire counts out of {n} (median statistic: sampling-only AUC, clock-matched Delta); * = expected", "",
             "family | sampling flag | clock-matched flag | median clock-only AUC | median window AUC (values)"]
    ok = True
    for family in FAMILIES:
        part = frame[frame["family"] == family]
        cells = []
        for key, _, stat in FLAGS:
            fires = int(part[key].sum())
            expected = key in EXPECTED[family]
            passed = fires >= 18 * n / 20 if expected else fires <= 2 * n / 20
            ok &= passed
            cells.append(f"{fires}/{n}{'*' if expected else ''} ({part[stat].median():.3f}){'' if passed else ' FAIL'}")
        lines.append(f"{family} | {cells[0]} | {cells[1]} | {part['a_clock'].median():.3f} | {part['a_val'].median():.3f}")
    lines += ["", f"Validation {'PASSED' if ok else 'FAILED'} (expected flags in >= 18/20, unexpected in <= 2/20)."]
    return "\n".join(lines), ok


def main():
    frame = run()
    body, _ = table(frame)
    out = ("All-disruptive mode, planted-leak validation (docs/preregistration_planted_leaks_all.md). Synthetic records "
           "only; a software validation, not evidence about any machine.\n\n" + body)
    folder = os.path.join(config.RESULTS_DIR, "toolkit")
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, "planted_leaks_all.csv"), index=False)
    with open(os.path.join(folder, "planted_leaks_all.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send(out[:3400])


if __name__ == "__main__":
    main()
