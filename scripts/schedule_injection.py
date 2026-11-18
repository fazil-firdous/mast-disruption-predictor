"""How much does an outcome-associated observation schedule alone inflate a values-only predictor?
(docs/preregistration_schedule_injection.md, fixed before any data or model.)

C-Mod's own measurements, uniformly sampled, under two schedules that keep the same values: U (every tenth slice in
every shot) and I (U plus every slice in the last 300 ms of disrupted shots, an EAST-like dense segment). Values-only
boosting on the harmonisation study's representations; the difference I minus U on the same test shots.

    python scripts/schedule_injection.py /root/dv/var/www/html/reports/20JA/20JA72/20ja072_fig4_data/C_Mod_for_nn.mat

Writes RESULTS_DIR/sampling_audit/schedule_injection.txt. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import harmonisation as hm, sampling_audit as sa, timing_model as tm

STEP = 10  # keep every tenth slice, counted back from the last
DENSE_S = 0.300  # injected: every slice within this long of the last slice, disrupted shots only
DELTA_S = 0.050  # R2 grid and maximum age (the sparse cadence)
WORTH = 0.02
SEED, N_BOOT = 42, 2000


def thin(frame, inject):
    """The frame with only the kept slices of each shot: U if inject is False, I if True."""
    keep = []
    for _, g in frame.groupby("shot", sort=True):
        g = g.sort_values("time")
        t = g["time"].to_numpy(dtype=float)
        n = len(t)
        k = (n - 1 - np.arange(n)) % STEP == 0
        disrupted = np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()
        if inject and disrupted:
            k |= t >= t[-1] - DENSE_S - 1e-9
        keep.append(g.index.to_numpy()[k])
    return frame.loc[np.concatenate(keep) if keep else []]


def window_count(frame):
    """Per shot: slices in the window (cut 40 ms before the last slice, 300 ms long) and the label."""
    rows = []
    for shot, g in frame.groupby("shot", sort=True):
        t = np.sort(g["time"].to_numpy(dtype=float))
        cut = t[-1] - sa.CUT_S
        rows.append({"shot": int(shot), "disrupted": bool(np.isfinite(g["time_until_disrupt"].to_numpy(dtype=float)).any()),
                     "n_window": int(((t > cut - hm.WINDOW_S + 1e-9) & (t <= cut + 1e-9)).sum())})
    return pd.DataFrame(rows)


def common(rows_u, rows_i):
    """Both row tables restricted to the shots present in both, in the same (shot) order."""
    both = sorted(set(rows_u["shot"]) & set(rows_i["shot"]))
    pick = lambda r: r[r["shot"].isin(both)].sort_values("shot").reset_index(drop=True)  # noqa: E731
    return pick(rows_u), pick(rows_i)


def with_timing(rows, timing):
    """rows plus a 'tv' column: the 36 value features followed by the three timing features (time-aware variant)."""
    t = timing.set_index("shot").reindex(rows["shot"])[tm.FEATURES].to_numpy(dtype=float)
    return rows.assign(tv=[list(v) + list(x) for v, x in zip(rows["r0"], t)])


def timing_check(timing, split):
    r, _, c, _, counts = tm.fit_and_test(timing, split)
    return r, counts


def reading(inflation, check_i, check_u):
    if check_i[1] is None or not (check_i[0] >= 0.80 and check_i[1] > 0.5):
        return "injection check failed (timing-only AUC under I below 0.80 or not above 0.5): nothing is read"
    prefix = "" if check_u[1] is not None and check_u[1] <= 0.5 <= check_u[2] else \
        "(relative to a control that itself carries some timing information) "
    if inflation[1] > 0 and inflation[0] >= WORTH:
        return prefix + (f"a label-associated schedule of this shape, on identical measurements, raises a values-only "
                         f"predictor's test AUC by {inflation[0]:.3f}")
    if inflation[1] > 0:
        return prefix + "a measurable but small inflation (below 0.02)"
    return prefix + "an inflation is not shown for a values-only predictor of this kind"


def rng(r, signed=False):
    f = "+.3f" if signed else ".3f"
    return f"{r[0]:{f}} ({r[1]:{f}} to {r[2]:{f}})" if r[1] is not None else f"{r[0]:{f}}"


def run(path):
    frame = sa.load(path)
    signals = [c for c in frame.columns if c not in ("shot", "time", "time_until_disrupt")]
    u, i = thin(frame, False), thin(frame, True)
    rows_u, rows_i = common(hm.shot_rows(u, signals, DELTA_S), hm.shot_rows(i, signals, DELTA_S))
    shots = set(rows_u["shot"])
    tim_u, tim_i = (tm.features(f[f["shot"].isin(shots)]) for f in (u, i))
    split = tm.our_split(rows_u, seed=SEED)
    check_u, counts = timing_check(tim_u, split)
    check_i, _ = timing_check(tim_i, split)
    lines = ["Schedule injection (docs/preregistration_schedule_injection.md): C-Mod's own measurements; U = every tenth "
             "slice in every shot; I = U plus every slice in the last 300 ms of disrupted shots. Values-only boosting, "
             "36 features, test once on the same test shots.", "",
             f"Shots: {len(shots):,} ({int(rows_u['disrupted'].sum()):,} disrupted); test {counts['test'][0]} disrupted, "
             f"{counts['test'][1]} not.",
             f"Injection check, timing-only test AUC under I: {rng(check_i)} (needs at least 0.80, range above 0.5).",
             f"Control check, timing-only test AUC under U: {rng(check_u)} (range should include 0.5)."]
    wc = {name: window_count(f[f["shot"].isin(shots)]) for name, f in (("U", u), ("I", i))}
    lines.append("Median slices in the window, disrupted | not: " + "; ".join(
        f"{n} {w.loc[w['disrupted'], 'n_window'].median():.0f} | {w.loc[~w['disrupted'], 'n_window'].median():.0f}"
        for n, w in wc.items()))
    lines.append("")
    results = {}
    for rep in ("r0", "r1", "r2", "tv"):
        ru, ri = (with_timing(rows_u, tim_u), with_timing(rows_i, tim_i)) if rep == "tv" else (rows_u, rows_i)
        su, y = hm.boosted_scores(ru, split, rep)
        si, yi = hm.boosted_scores(ri, split, rep)
        assert np.array_equal(y, yi)
        results[rep] = (sa.auc_range(su, y), sa.auc_range(si, y), hm.paired_drop(si, su, y, N_BOOT, SEED))
    names = {"r0": "R0 published slices", "r1": "R1 5 ms forward-fill", "r2": "R2 sparse grid (50 ms, max age 50 ms)",
             "tv": "R0 values + timing features (time-aware)"}
    a_u, a_i, inf = results["r0"]
    lines.append(f"PRIMARY, R0: Inflation = AUC(I) - AUC(U) = {rng(inf, True)}; AUC U {rng(a_u)}, I {rng(a_i)}.")
    lines.append(f"Reading: {reading(inf, check_i, check_u)}.")
    lines += ["", "Secondary:"]
    for rep in ("r1", "r2", "tv"):
        a_u, a_i, inf = results[rep]
        lines.append(f"- {names[rep]}: inflation {rng(inf, True)}; AUC U {rng(a_u)}, I {rng(a_i)}")
    lines += ["", "Not claimed: what the schedule did in any published model; this is C-Mod with an imposed schedule."]
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = run(sys.argv[1])
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "schedule_injection.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
