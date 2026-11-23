"""Planted-leak validation of the audit toolkit (docs/preregistration_planted_leaks.md, fixed before any run): synthetic
records with one planted shortcut each (sampling, duration, end-of-record) and three generic checks (T observation-
process ceiling, D clock ceiling, E endpoint dependence). Needs no data.

    python scripts/planted_leaks.py            # choose s on L0 seed 0, then the 4 x 20 validation
    python scripts/planted_leaks.py --quick    # 3 replicates per family, a smoke test only

Writes RESULTS_DIR/toolkit/planted_leaks.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import endpoint_zhu as ez, sampling_audit as sa, timing_model as tm
from src import config, notify

N_SHOTS, N_SIG, STEP, SPARSE = 600, 12, 0.010, 0.100
PHI, INNOV, NOISE, RAMP = 0.95, 0.1, 0.05, 0.300
FAMILIES = ("L0", "L_T", "L_D", "L_E")
CHECKS = ("T", "D", "E")
EXPECTED = {"L0": set(), "L_T": {"T"}, "L_D": {"D"}, "L_E": {"E"}}
STRENGTHS = (0.25, 0.5, 1.0, 2.0)
AUC_LINE = 0.80
SEEDS = range(1, 21)


def ar1(rng, n):
    """N_SIG AR(1) series of length n, started from the stationary distribution, plus measurement noise."""
    x = np.empty((n, N_SIG))
    x[0] = rng.standard_normal(N_SIG) * INNOV / np.sqrt(1 - PHI ** 2)
    e = rng.standard_normal((n, N_SIG)) * INNOV
    for i in range(1, n):
        x[i] = PHI * x[i - 1] + e[i]
    return x + rng.standard_normal((n, N_SIG)) * NOISE


def ramp(t):
    return np.clip(1 - (t[-1] - t) / RAMP, 0, None)


def generate(family, seed, s):
    """{shot: (t, values, disrupted)} for one family and generator seed."""
    rng = np.random.default_rng(seed)
    shots = {}
    for k in range(N_SHOTS):
        disrupted = k % 3 == 0
        if family == "L_D":
            duration = rng.uniform(0.6, 1.4) if disrupted else rng.uniform(1.6, 3.0)
        else:
            duration = rng.uniform(1.0, 3.0)
        t = np.round(np.arange(0.0, duration, STEP), 6)
        x = ar1(rng, len(t))
        if disrupted:
            x[:, 0] += s * ramp(t)
        elif family == "L_E":
            x[:, 1] += s * ramp(t)
        if family == "L_T" and not disrupted:
            keep = np.zeros(len(t), dtype=bool)
            keep[::int(round(SPARSE / STEP))] = True
            keep[-1] = True  # the record still ends where it ends
            t, x = t[keep], x[keep]
        shots[10000 + k] = (t, x, disrupted)
    return shots


def split_of(shots):
    return tm.our_split(pd.DataFrame({"shot": list(shots), "disrupted": [v[2] for v in shots.values()]}))


def timing_row(t):
    landmark = t[-1] - ez.CUT
    keep = (t > landmark - ez.WINDOW + ez.TOL) & (t <= landmark + ez.TOL)
    n = int(keep.sum())
    age = landmark - t[keep][-1] if n else ez.WINDOW
    return [n, age, tm.max_gap(t, landmark, span=ez.WINDOW)]


def logistic_auc(shots, split, rows_of):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr = [s for s in shots if split[s] == "train"]
    te = [s for s in shots if split[s] == "test"]
    x_tr = np.nan_to_num(np.array([rows_of(shots[s][0]) for s in tr], dtype=float))
    x_te = np.nan_to_num(np.array([rows_of(shots[s][0]) for s in te], dtype=float))
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x_tr, [shots[s][2] for s in tr])
    return sa.auc(model.predict_proba(x_te)[:, 1], np.array([shots[s][2] for s in te]))


def check_t(shots, split):
    a = logistic_auc(shots, split, timing_row)
    return a >= AUC_LINE, a


def check_d(shots, split):
    a = logistic_auc(shots, split, lambda t: [t[-1] - ez.CUT - t[0]])
    return a >= AUC_LINE, a


def check_e(shots, split, n_boot=ez.N_BOOT):
    b = ez.build(shots, split)
    mask = np.isfinite(b["s_m"])
    w_e, w_m = ez.win_matrix(b["s_case"], b["s_end"]), ez.win_matrix(b["s_case"], b["s_m"])
    d = ez.delta_range(w_e, w_m, mask, n_boot=n_boot)
    fired = abs(d[0]) >= ez.WORTH and (d[1] > 0 or d[2] < 0)
    return fired, d[0]


def values_auc(shots, split):
    b = ez.build(shots, split)
    labels = np.r_[np.ones(len(b["cases"])), np.zeros(len(b["controls"]))].astype(bool)
    return sa.auc(np.r_[b["s_case"], b["s_end"]], labels)


def choose_strength():
    """The smallest s whose values-only AUC on L0, seed 0, lies in 0.75 to 0.90; else the one closest to 0.825."""
    tried = []
    for s in STRENGTHS:
        shots = generate("L0", 0, s)
        a = values_auc(shots, split_of(shots))
        tried.append((s, a))
        if 0.75 <= a <= 0.90:
            return s, tried, True
    return min(tried, key=lambda r: abs(r[1] - 0.825))[0], tried, False


def validate(s, seeds=SEEDS, n_boot=ez.N_BOOT):
    rows = []
    for family in FAMILIES:
        for seed in seeds:
            shots = generate(family, seed, s)
            split = split_of(shots)
            for name, fn in (("T", check_t), ("D", check_d), ("E", lambda a, b: check_e(a, b, n_boot))):
                fired, stat = fn(shots, split)
                rows.append({"family": family, "seed": seed, "check": name, "fired": bool(fired), "stat": stat})
            print(f"{family} seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def table(frame):
    n = frame["seed"].nunique()
    lines = [f"Fire counts out of {n} (median statistic: AUC for T and D, Delta for E); * = expected to fire", "",
             "family | " + " | ".join(CHECKS)]
    ok = True
    for family in FAMILIES:
        cells = []
        for check in CHECKS:
            part = frame[(frame["family"] == family) & (frame["check"] == check)]
            fires = int(part["fired"].sum())
            expected = check in EXPECTED[family]
            passed = fires >= n - 2 * n / 20 if expected else fires <= 2 * n / 20
            ok &= passed
            cells.append(f"{fires}/{n}{'*' if expected else ''} ({part['stat'].median():.3f}){'' if passed else ' FAIL'}")
        lines.append(f"{family} | " + " | ".join(cells))
    lines += ["", f"Validation {'PASSED' if ok else 'FAILED'} (expected flags in >= 18/20, unexpected in <= 2/20)."]
    return "\n".join(lines), ok


def main():
    quick = "--quick" in sys.argv
    s, tried, in_range = choose_strength()
    head = ["Planted-leak validation (docs/preregistration_planted_leaks.md). Synthetic records only; a software "
            "validation, not evidence about any machine.",
            "Strength s chosen on L0 seed 0 before any check: " + ", ".join(f"s={a}: AUC {b:.3f}" for a, b in tried) +
            f" -> s = {s}{'' if in_range else ' (none in 0.75-0.90; closest used)'}", ""]
    frame = validate(s, seeds=range(1, 4) if quick else SEEDS, n_boot=200 if quick else ez.N_BOOT)
    body, _ = table(frame)
    out = "\n".join(head) + body + ("\n\nQUICK SMOKE TEST ONLY (3 replicates, 200 resamples); not the validation."
                                    if quick else "")
    folder = os.path.join(config.RESULTS_DIR, "toolkit")
    os.makedirs(folder, exist_ok=True)
    frame.to_csv(os.path.join(folder, "planted_leaks" + ("_quick" if quick else "") + ".csv"), index=False)
    with open(os.path.join(folder, "planted_leaks" + ("_quick" if quick else "") + ".txt"), "w") as f:
        f.write(out + "\n")
    print(out)
    if not quick:
        notify.send("Planted-leak validation\n\n" + out[:3400])


if __name__ == "__main__":
    main()
