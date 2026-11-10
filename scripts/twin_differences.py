"""Twin differences (exploratory, no model): what differs between a shot that was lost and the same-programme shots
that were not, and how early?

For each shot lost in the flat-top (L) take the two nearest earlier shots with the same programmed current that were
not lost (K1 and K2, the rules of scripts/twin_check.py). All three were asked to do the same thing, so
- L - K1 is what the lost shot did differently, with the programme taken out (the case);
- K1 - K2 is how much two good repeats differ anyway (the control). It has the same shot order as the case, so a
  drift from shot to shot shows up in both.
Per signal (the 20 ms window means of the window table) and per lead time before the loss:
- D: the share of triplets with |L - K1| > |K1 - K2|. 0.5 means the lost shot is no further from its twin than two
  good repeats are from each other;
- up: the share with L > K1, next to the share with K1 > K2.
Per triplet, a divergence time without any model or label: how long before the loss a signal left the band that good
repeats stay in (the 95th percentile of |K1 - K2|) and stayed out. The same count on K1 - K2 gives the false rate.

    python scripts/twin_differences.py     # reads auto_labels/per_shot.csv, data/extras/, the window table and,
                                           # if there, events/catalogue.csv; writes RESULTS_DIR/twin_differences/

Not pre-registered: descriptions with ranges, no verdicts. Outside every stamp.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import twin_check, v3_runs
from src import config, extras, features

SIGNALS = list(features.SIGNALS) + list(features.EXTRA_SIGNALS) + list(features.V2_SIGNALS)
MACHINE = {"sol", "ref"}  # not the plasma: left out when asking which signal moved first
LEADS_MS = (10, 30, 50, 100, 200, 300)
BAND_PERCENTILE = 95  # a signal has 'left the band' when |L - K1| is above this percentile of |K1 - K2|
HELD_MS = (20, 50, 100)  # divergence counted when the signal stayed out of the band for at least this long
NEAR_LOSS_MS = 12.5  # the last window all three shots share must end this close to the loss
N_BOOT = 400
STRIDE_S = config.STRIDE_MS / 1000
OUT_DIR = os.path.join(config.RESULTS_DIR, "twin_differences")
LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
CATALOGUE = os.path.join(config.RESULTS_DIR, "events", "catalogue.csv")


def kept_twins(labels, load, reach=twin_check.REACH):
    """One row per lost shot that has two earlier same-programme shots that were not lost (within reach, same
    campaign): shot_id, k1 (the nearest; the twin of scripts/twin_check.py), k2 (the next one), group."""
    group = twin_check.groups(labels)
    quench = labels.set_index("shot_id")["t_quench"]
    campaign = labels.set_index("shot_id")["campaign"] if "campaign" in labels else pd.Series(dtype=object)
    cache = {}

    def prog(shot):
        if shot not in cache:
            cache[shot] = twin_check.programme(load(shot))
        return cache[shot]

    rows = []
    for shot in labels["shot_id"]:
        if group.get(shot) not in twin_check.LOST or prog(shot) is None:
            continue
        found = []
        for other in range(shot - 1, shot - reach - 1, -1):
            if group.get(other) not in twin_check.KEPT or campaign.get(other) != campaign.get(shot):
                continue
            stop = np.nanmin([quench.get(shot, np.nan), quench.get(other, np.nan), np.inf])
            c = twin_check.compare(prog(shot), prog(other), stop)
            if c and c["rms"] < twin_check.SAME_RMS and c["peak_ratio"] > twin_check.SAME_PEAK:
                found.append(other)
                if len(found) == 2:
                    rows.append({"shot_id": shot, "k1": found[0], "k2": found[1], "group": group[shot]})
                    break
    return pd.DataFrame(rows, columns=["shot_id", "k1", "k2", "group"])


def by_shot(windows):
    """{shot: (bins, values)}: each shot's windows on a common time grid (t_end rounded to the window stride, since
    every shot's windows start at its own first sample) and its signals' window means, in time order."""
    out = {}
    columns = [f"{s}_mean" for s in SIGNALS]
    for shot, g in windows.groupby("shot_id"):
        bins = np.round(g["t_end"].to_numpy(dtype=float) / STRIDE_S).astype(int)
        order = np.argsort(bins, kind="stable")
        bins, values = bins[order], g[columns].to_numpy(dtype=float)[order]
        keep = np.concatenate([bins[1:] != bins[:-1], [True]])
        out[shot] = (bins[keep], values[keep])
    return out


def series(by, row, t_loss, t_from=None):
    """(lead in ms, case, control) on the time bins all three shots of a triplet have, from t_from to the loss:
    case = L - K1 and control = K1 - K2, arrays of (windows, signals). None if they share no window."""
    if any(shot not in by for shot in (row.shot_id, row.k1, row.k2)):
        return None
    (bl, vl), (b1, v1), (b2, v2) = by[row.shot_id], by[row.k1], by[row.k2]
    common = np.intersect1d(np.intersect1d(bl, b1), b2)
    start = -np.inf if t_from is None else t_from
    common = common[(common * STRIDE_S >= start - features.EPS) & (common * STRIDE_S < t_loss - features.EPS)]
    if len(common) == 0:
        return None
    il, i1, i2 = (np.searchsorted(b, common) for b in (bl, b1, b2))
    return (t_loss - common * STRIDE_S) * 1000, vl[il] - v1[i1], v1[i1] - v2[i2]


def half(greater, equal):
    """1 where greater, 0.5 where equal, else 0."""
    return np.where(greater, 1.0, np.where(equal, 0.5, 0.0))


def cluster_range(values, clusters, rng, n=N_BOOT):
    """(mean, low, high, count) of values, the 95% range from resampling whole clusters (triplets that share their
    nearest twin are not independent). NaN values are left out."""
    ok = np.isfinite(values)
    values, clusters = values[ok], np.asarray(clusters)[ok]
    if len(values) == 0:
        return np.nan, np.nan, np.nan, 0
    _, inverse = np.unique(clusters, return_inverse=True)
    sums, counts = np.bincount(inverse, weights=values), np.bincount(inverse).astype(float)
    pick = rng.integers(len(sums), size=(n, len(sums)))
    boot = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(values.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(values)


def lead_table(collected, rng):
    """Per (signal, lead): D with its range, the share with L above K1 and with K1 above K2, and the triplet count."""
    rows = []
    for lead_ms in LEADS_MS:
        case, ctrl, cluster = [], [], []
        for row, lead, c, k in collected:
            i = int(np.argmin(np.abs(lead - lead_ms)))
            if abs(lead[i] - lead_ms) <= config.STRIDE_MS / 2 + 1e-6:
                case.append(c[i])
                ctrl.append(k[i])
                cluster.append(row.k1)
        if not case:
            continue
        case, ctrl = np.array(case), np.array(ctrl)
        seen = np.isfinite(case) & np.isfinite(ctrl)
        d = np.where(seen, half(np.abs(case) > np.abs(ctrl), np.abs(case) == np.abs(ctrl)), np.nan)
        up_case = np.where(seen, half(case > 0, case == 0), np.nan)
        up_ctrl = np.where(seen, half(ctrl > 0, ctrl == 0), np.nan)
        for j, signal in enumerate(SIGNALS):
            mean, low, high, n = cluster_range(d[:, j], cluster, rng)
            rows.append({"signal": signal, "lead_ms": lead_ms, "n": n, "D": mean, "D_low": low, "D_high": high,
                         "up_case": np.nanmean(up_case[:, j]) if n else np.nan,
                         "up_ctrl": np.nanmean(up_ctrl[:, j]) if n else np.nan})
    return pd.DataFrame(rows)


def trailing_lead(outside, lead):
    """How long before the loss (ms) the signal has been outside the band without a break; 0 if the last window is
    inside. outside: one flag per window in time order; lead: each window's time before the loss."""
    if len(outside) == 0 or not outside[-1]:
        return 0.0
    inside = np.where(~outside)[0]
    return float(lead[inside[-1] + 1 if len(inside) else 0])


def divergence(collected):
    """(per-triplet table, band): for each triplet whose shared windows reach the loss, how long each signal of the
    case (L - K1) and of the control (K1 - K2) stayed outside the band before the loss. band: per signal, the
    BAND_PERCENTILE of |K1 - K2| over every window of every triplet."""
    pooled = np.concatenate([np.abs(k) for _, _, _, k in collected]) if collected else np.empty((0, len(SIGNALS)))
    with np.errstate(all="ignore"):
        band = np.nanpercentile(pooled, BAND_PERCENTILE, axis=0) if len(pooled) else np.full(len(SIGNALS), np.nan)
    rows = []
    for row, lead, c, k in collected:
        if lead[-1] > NEAR_LOSS_MS:
            continue
        out = {"shot_id": row.shot_id, "k1": row.k1, "k2": row.k2}
        for j, signal in enumerate(SIGNALS):
            out[f"case_{signal}"] = trailing_lead(np.abs(c[:, j]) > band[j], lead)
            out[f"ctrl_{signal}"] = trailing_lead(np.abs(k[:, j]) > band[j], lead)
        rows.append(out)
    return pd.DataFrame(rows), dict(zip(SIGNALS, band))


def first_diverged(table, min_ms=HELD_MS[0]):
    """Per triplet, the plasma signal that left the band earliest and stayed out (at least min_ms before the loss),
    or '' if none did."""
    plasma = [s for s in SIGNALS if s not in MACHINE]
    leads = table[[f"case_{s}" for s in plasma]].to_numpy()
    best = leads.argmax(axis=1) if len(leads) else np.array([], dtype=int)
    return pd.Series([plasma[j] if leads[i, j] >= min_ms else "" for i, j in enumerate(best)], index=table.index)


def collect(triplets, by, t_loss, t_start):
    """[(triplet row, lead, case, control)] for every triplet with shared windows in the flat-top before the loss."""
    out = []
    for row in triplets.itertuples():
        tq = t_loss.get(row.shot_id, np.nan)
        if not np.isfinite(tq):
            continue
        s = series(by, row, float(tq), t_start.get(row.shot_id))
        if s is not None:
            out.append((row, *s))
    return out


def text(triplets, leads, div, causes=None):
    """The summary in words."""
    lines = ["Exploratory, no model, not pre-registered. L = a shot lost in the flat-top; K1, K2 = the two nearest "
             "earlier shots with the same programme that were not lost.",
             f"Triplets: {len(triplets):,} ({triplets['group'].value_counts().to_dict()}).", "",
             "D = share of triplets with |L - K1| > |K1 - K2| (0.5 = no further apart than two good repeats); * = the "
             "95% range leaves out 0.5. Columns: time before the loss.", "",
             "| Signal | " + " | ".join(f"{m} ms" for m in LEADS_MS) + " |", "|---|" + "---|" * len(LEADS_MS)]
    for signal in SIGNALS:
        part = leads[leads["signal"] == signal].set_index("lead_ms") if len(leads) else leads
        cells = []
        for m in LEADS_MS:
            if m in part.index and part.loc[m, "n"] > 0:
                r = part.loc[m]
                cells.append(f"{r['D']:.2f}{'*' if r['D_low'] > 0.5 or r['D_high'] < 0.5 else ''}")
            else:
                cells.append("-")
        lines.append(f"| {signal}{' (machine)' if signal in MACHINE else ''} | " + " | ".join(cells) + " |")
    early = [m for m in (100, 200) if len(leads) and m in set(leads["lead_ms"])]
    if early:
        lines += ["", "Direction: share of triplets with L above K1 (and, as the control, K1 above K2):"]
        for signal in SIGNALS:
            part = leads[(leads["signal"] == signal)].set_index("lead_ms")
            cells = [f"{m} ms {part.loc[m, 'up_case']:.2f} ({part.loc[m, 'up_ctrl']:.2f})" for m in early
                     if m in part.index and part.loc[m, "n"] > 0]
            lines.append(f"- {signal}: " + "; ".join(cells))
    if len(div):
        lines += ["", f"Divergence: share of {len(div):,} triplets in which the signal left the band of good repeats "
                      f"and stayed out until the loss, for at least " + " / ".join(f"{m}" for m in HELD_MS)
                      + " ms (the same count on K1 - K2 in brackets):"]
        for signal in SIGNALS:
            cells = [f"{(div[f'case_{signal}'] >= m).mean():.0%} ({(div[f'ctrl_{signal}'] >= m).mean():.0%})"
                     for m in HELD_MS]
            lines.append(f"- {signal}: " + " / ".join(cells))
        first = first_diverged(div)
        share = first[first != ""].value_counts(normalize=True)
        lines += ["", f"Which plasma signal left the band first (of the {int((first != '').sum()):,} triplets where "
                      f"one did, at least {HELD_MS[0]} ms before the loss): "
                      + ", ".join(f"{s} {v:.0%}" for s, v in share.head(6).items()) + "."]
        if causes is not None:
            named = div["shot_id"].map(causes).fillna("")
            for cause in sorted(set(named) - {""}):
                part = first[(named == cause) & (first != "")]
                if len(part) >= 10:
                    top = part.value_counts(normalize=True).head(3)
                    lines.append(f"- operator named '{cause}' ({len(part)} triplets): "
                                 + ", ".join(f"{s} {v:.0%}" for s, v in top.items()))
    return "\n".join(lines)


def main():
    labels = pd.read_csv(LABELS_CSV)
    load = lambda shot: extras.load_extras(int(shot), allow_download=False)
    triplets = kept_twins(labels, load)
    os.makedirs(OUT_DIR, exist_ok=True)
    triplets.to_csv(os.path.join(OUT_DIR, "triplets.csv"), index=False)
    shots = set(triplets["shot_id"]) | set(triplets["k1"]) | set(triplets["k2"])
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"),
                              columns=["shot_id", "t_end"] + [f"{s}_mean" for s in SIGNALS])
    by = by_shot(windows[windows["shot_id"].isin(shots)])
    t_loss = labels.set_index("shot_id")["t_quench"].to_dict()
    t_start = {shot: v3_runs.flat_top_start(load(shot)) for shot in triplets["shot_id"]}
    collected = collect(triplets, by, t_loss, t_start)
    leads = lead_table(collected, np.random.default_rng(config.RANDOM_SEED))
    div, band = divergence(collected)
    leads.to_csv(os.path.join(OUT_DIR, "leads.csv"), index=False)
    div.to_csv(os.path.join(OUT_DIR, "divergence.csv"), index=False)
    pd.Series(band, name="band").to_csv(os.path.join(OUT_DIR, "band.csv"), index_label="signal")
    causes = None
    if os.path.exists(CATALOGUE):
        cat = pd.read_csv(CATALOGUE)
        causes = cat.set_index("shot_id")["cause"].fillna("").to_dict()
    out = text(triplets, leads, div, causes)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
