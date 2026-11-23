"""Validation of the all-disruptive landmark primary (docs/preregistration_landmark_mode.md, fixed before any code):
constant- and time-varying-hazard synthetic shots, each family with and without a readable drift built from the same
random draws. Needs no data.

    python scripts/planted_leaks_landmark.py

Writes RESULTS_DIR/toolkit/planted_landmark.txt and .csv. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import audit as au, planted_leaks as pl
from src import config, notify

N_SHOTS, N_SIG, TICK, BURN = 200, 4, 0.001, 20
P_CONST = 0.005
FAMILIES = ("N", "N+D", "V", "V+D", "P", "P+D")
SETTINGS = dict(window=0.020, step=0.005, horizon=0.030, lead=0.002, tol=0.010)
EXPECTED_GAIN = {"P", "P+D"}
CLOCK_RULE = {"N+D": False, "V+D": True}  # the secondary expectations; others reported only
SEEDS = range(1, 21)


def hazard(family, k):
    t = k * TICK
    return 0.002 * (1 + t / 0.05) if family.startswith("V") else P_CONST


def generate(family, seed, strength=0.5, n_shots=N_SHOTS):
    """{shot: (t, values, True, t_event)}; the drift is added after all random draws, so X and X+D share them."""
    base = family.replace("+D", "")
    rng = np.random.default_rng(seed)
    shots = {}
    for i in range(n_shots):
        k = BURN
        while rng.random() >= hazard(base, k):
            k += 1
        t = np.round(np.arange(k + 1) * TICK, 6)
        x = pl.ar1(rng, len(t))[:, :N_SIG]
        if base == "P":
            x[:, 0] += strength * np.clip(1 - (t[-1] - t) / 0.030, 0, None)
        if family.endswith("+D"):
            x[:, 1] += 10.0 * t
        shots[90000 + i] = (t, x, True, float(t[-1]))
    return shots


def generator_checks(shots):
    """(next-30 ms positive share over the decision grid, correlation of signal 1 with elapsed time)."""
    rows = au.landmark_rows(shots, **{k: v for k, v in SETTINGS.items() if k != "tol"})
    kept = rows[rows["label"] >= 0]
    t = np.concatenate([v[0] for v in shots.values()])
    x1 = np.concatenate([v[1][:, 1] for v in shots.values()])
    return float((kept["label"] == 1).mean()), float(np.corrcoef(t, x1)[0, 1])


def run(seeds=SEEDS, families=FAMILIES, early_stopping=False):
    out = []
    for family in families:
        for seed in seeds:
            shots = generate(family, seed)
            r = au.landmark_stats(shots, **SETTINGS, early_stopping=early_stopping)
            share, corr = generator_checks(shots)
            out.append({"family": family, "seed": seed, "gain": r["gain"][0], "low": r["gain"][1],
                        "high": r["gain"][2], "flag_gain": bool(r["flag_gain"]), "delta": r["delta"][0],
                        "flag_clock_match": bool(r["flag_clock_match"]), "a_clock": r["a_clock"],
                        "a_base": r["a_base"], "a_aug": r["a_aug"], "pos_share": share, "drift_corr": corr,
                        "n_dec": r["n_dec"]})
            print(f"{family} seed {seed} done", flush=True)
    return pd.DataFrame(out)


def table(frame):
    n = frame["seed"].nunique()
    analytic = 1 - (1 - P_CONST) ** 28  # events in (L + 2 ms, L + 30 ms]: 28 ticks
    lines = [f"Fire counts out of {n}; * = expected to fire; (median statistic)", "",
             "family | primary: Brier gain > 0 | secondary: clock-matched flag | elapsed-time AUC | "
             "next-30 ms share | drift corr"]
    ok = True
    for family in FAMILIES:
        part = frame[frame["family"] == family]
        g = int(part["flag_gain"].sum())
        exp = family in EXPECTED_GAIN
        g_ok = g >= 18 * n / 20 if exp else g <= 2 * n / 20
        c = int(part["flag_clock_match"].sum())
        if family in CLOCK_RULE:
            c_ok = c >= 18 * n / 20 if CLOCK_RULE[family] else c <= 2 * n / 20
            c_cell = f"{c}/{n}{'*' if CLOCK_RULE[family] else ''} ({part['delta'].median():+.3f}){'' if c_ok else ' FAIL'}"
        else:
            c_ok, c_cell = True, f"{c}/{n} ({part['delta'].median():+.3f}) reported only"
        ok &= g_ok and c_ok
        lines.append(f"{family} | {g}/{n}{'*' if exp else ''} ({part['gain'].median():+.5f}){'' if g_ok else ' FAIL'} | "
                     f"{c_cell} | {part['a_clock'].median():.3f} | {part['pos_share'].median():.3f} | "
                     f"{part['drift_corr'].median():.2f}")
    lines += ["", f"Analytic next-window share under constant hazard (events 2-30 ms ahead): {analytic:.3f}",
              f"Validation {'PASSED' if ok else 'FAILED'} (expected in >= 18/20, unexpected in <= 2/20)."]
    return "\n".join(lines), ok


def investigation_1():
    """Exploratory, after the first table: families N and P, seeds 1-20, early stopping in both learners."""
    frame = run(families=("N", "P"), early_stopping=True)
    lines = ["Investigation 1 (after the first table; exploratory): N and P with early stopping in both learners", "",
             "family | gain > 0 | median gain (low, high) | decision AUC baseline, with values"]
    for family, part in frame.groupby("family"):
        lines.append(f"{family} | {int(part['flag_gain'].sum())}/{len(part)} | {part['gain'].median():+.5f} "
                     f"({part['low'].median():+.5f}, {part['high'].median():+.5f}) | {part['a_base'].median():.3f}, "
                     f"{part['a_aug'].median():.3f}")
    out = "\n".join(lines)
    folder = os.path.join(config.RESULTS_DIR, "toolkit")
    frame.to_csv(os.path.join(folder, "planted_landmark_inv1.csv"), index=False)
    with open(os.path.join(folder, "planted_landmark_inv1.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


def power_study(seeds=range(21, 41)):
    """Design study (protocol, power study): fire rate of the landmark primary with early stopping by strength and size."""
    cells = [("N", 0.0)] + [("P", s) for s in (0.5, 1.0, 2.0)]
    rows = []
    for n in (200, 500):
        for family, strength in cells:
            for seed in seeds:
                r = au.landmark_stats(generate(family, seed, strength or 0.5, n), **SETTINGS, early_stopping=True)
                rows.append({"n_shots": n, "family": family, "strength": strength, "seed": seed, "gain": r["gain"][0],
                             "low": r["gain"][1], "fired": bool(r["flag_gain"]), "a_aug": r["a_aug"]})
            print(f"{n} {family} {strength} done", flush=True)
    frame = pd.DataFrame(rows)
    lines = ["Power study (protocol, after investigation 1; design, not validation): landmark primary with early "
             "stopping, seeds 21-40", "", "shots | family | strength | fired | median gain | median decision AUC"]
    for (n, family, strength), part in frame.groupby(["n_shots", "family", "strength"]):
        lines.append(f"{n} | {family} | {strength} | {int(part['fired'].sum())}/{len(part)} | "
                     f"{part['gain'].median():+.5f} | {part['a_aug'].median():.3f}")
    out = "\n".join(lines)
    folder = os.path.join(config.RESULTS_DIR, "toolkit")
    frame.to_csv(os.path.join(folder, "planted_landmark_power.csv"), index=False)
    with open(os.path.join(folder, "planted_landmark_power.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send(out[:3400])


def main():
    if "--power" in sys.argv:
        return power_study()
    if "--investigation-1" in sys.argv:
        return investigation_1()
    frame = run()
    body, _ = table(frame)
    out = ("All-disruptive landmark primary, validation (docs/preregistration_landmark_mode.md). Synthetic only; a "
           "software validation, not evidence about any machine.\n\n" + body)
    folder = os.path.join(config.RESULTS_DIR, "toolkit")
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, "planted_landmark.csv"), index=False)
    with open(os.path.join(folder, "planted_landmark.txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    notify.send(out[:3400])


if __name__ == "__main__":
    main()
