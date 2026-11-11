"""The twin test for any disruption predictor on MAST. Standalone: it needs only numpy and pandas, and the file
programme_twins.csv of the data release (no MAST data access).

A twin pair is a shot that was lost during the current flat-top and an earlier shot with the same programmed plasma
current that was not lost. A predictor passes on a pair if, before the loss, it scores the lost shot higher than its
twin. C is the share of pairs passed (ties count half); 0.5 is chance. A model that has only learned which
programmes are risky scores both shots alike.

    python twin_score.py scores.csv programme_twins.csv [--lead-ms 5]

scores.csv has one row per score your model gave: shot_id, t_end (s; the end of the data that score used) and score
(higher = more likely to be lost). Score only shots the model was not trained or tuned on. A pair counts when both
of its shots are in the file. For each pair, both shots are scored over the same programme time, from t_start (the
start of the flat-top) to lead-ms before t_loss (the lost shot's current quench), and each shot's score is its
highest in that range.

The 95% range resamples pairs in groups that share their twin, because one kept shot can be the twin of several lost
shots. Report C on pairs with the same beam heating next to C on all pairs: twins share the programmed current, and
on MAST the lost shot more often had more beam power.
"""
import argparse

import numpy as np
import pandas as pd

EXACT_RMS = 0.01  # programmes within 1%
N_BOOT = 2000


def pair_wins(scores, twins, lead_ms=5.0):
    """One row per twin pair with both shots in scores and a score in range for each: shot_id, other, win (1 if the
    lost shot's highest score is above its twin's, 0.5 for a tie, 0 otherwise) and the pair's other columns."""
    by_shot = {shot: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float))
               for shot, g in scores.groupby("shot_id")}
    rows = []
    for pair in twins.to_dict("records"):
        lost, kept = by_shot.get(pair["shot_id"]), by_shot.get(pair["other"])
        t_loss = pair.get("t_loss", np.nan)
        if lost is None or kept is None or not np.isfinite(t_loss):
            continue
        t_start = pair.get("t_start", np.nan)
        t_from = float(t_start) if np.isfinite(t_start) else -np.inf
        t_to = float(t_loss) - lead_ms / 1000
        a = lost[1][(lost[0] >= t_from) & (lost[0] <= t_to)]
        b = kept[1][(kept[0] >= t_from) & (kept[0] <= t_to)]
        if len(a) == 0 or len(b) == 0:
            continue
        rows.append({**pair, "win": 1.0 if a.max() > b.max() else 0.5 if a.max() == b.max() else 0.0})
    return pd.DataFrame(rows, columns=list(twins.columns) + ["win"])


def concordance(wins, n_boot=N_BOOT, seed=0):
    """(C, low, high, pairs): the mean win and its 95% range from resampling the pairs in groups that share their
    twin. NaN without pairs."""
    if len(wins) == 0:
        return np.nan, np.nan, np.nan, 0
    _, group = np.unique(wins["other"].to_numpy(), return_inverse=True)
    sums, sizes = np.bincount(group, weights=wins["win"].to_numpy(dtype=float)), np.bincount(group).astype(float)
    pick = np.random.default_rng(seed).integers(len(sums), size=(n_boot, len(sums)))
    boot = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return (float(wins["win"].mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(wins))


def report(wins):
    """The result in words: C on all pairs and, where the release gives them, by beam heating and for programmes
    within 1%."""
    def line(name, part):
        c, low, high, n = concordance(part)
        return f"- {name}: C = {c:.3f} (95% range {low:.3f} to {high:.3f}) on {n:,} pairs" if n else f"- {name}: no pairs"

    lines = ["Twin test (0.5 = chance):", line("all pairs", wins)]
    if "heating" in wins:
        lines += [line(f"{value} beam heating", wins[wins["heating"] == value]) for value in ("same", "different")]
    if "rms" in wins:
        lines.append(line(f"programmes within {EXACT_RMS:.0%}", wins[wins["rms"].astype(float) < EXACT_RMS]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", help="CSV with shot_id, t_end (s) and score")
    parser.add_argument("twins", help="programme_twins.csv of the data release")
    parser.add_argument("--lead-ms", type=float, default=5.0, help="scores stop this long before the loss (default 5)")
    args = parser.parse_args()
    wins = pair_wins(pd.read_csv(args.scores), pd.read_csv(args.twins), args.lead_ms)
    print(report(wins))


if __name__ == "__main__":
    main()
