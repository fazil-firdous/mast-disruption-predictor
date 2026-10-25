"""How did each shot end? Automatic termination labels from physics rules, so nobody has to label shots by hand.

For every shot where the detector found a fast current quench, two signals at the quench time decide the class:
- the programmed plasma current (pulse_schedule): already ramping down -> 'programmed_rampdown';
- the central solenoid: if it had (nearly) stopped sweeping, the machine had run out of flux -> 'flux_exhausted';
- otherwise the plasma was lost during the programmed flat-top with flux still available -> 'flat_top_drop',
  the ending most likely to be an instability-driven disruption.
Shots with no fast quench are 'smooth_end'; missing signals give 'unknown'. Writes RESULTS_DIR/auto_labels/.
This is an analysis next to the experiments: no experiment uses these labels yet.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import config, extras

OUT_DIR = os.path.join(config.RESULTS_DIR, "auto_labels")
CLASSES = ["flat_top_drop", "flux_exhausted", "programmed_rampdown", "smooth_end", "unknown"]
WORDS = {"flat_top_drop": "lost during flat-top with flux left (likely a real disruption)",
         "flux_exhausted": "after the solenoid ran out of flux",
         "programmed_rampdown": "while the programme was already ramping down",
         "smooth_end": "smooth end, no sudden current drop",
         "unknown": "signals missing"}


def solenoid_ratio(t, ref, sol, i):
    """Solenoid sweep rate over the last AUTO_SOL_WINDOW_MS before index i, as a fraction of its average rate
    during the programmed flat-top. About 1 = still sweeping as usual, about 0 = stopped. NaN if not measurable."""
    flat = np.where(ref[: i + 1] >= config.AUTO_FLATTOP_FRAC * np.nanmax(ref))[0]
    w = config.AUTO_SOL_WINDOW_MS
    if len(flat) < config.AUTO_MIN_FLATTOP_MS or i < w or not np.isfinite(sol[[flat[0], flat[-1], i - w, i]]).all():
        return np.nan
    rate_flat = (sol[flat[-1]] - sol[flat[0]]) / (t[flat[-1]] - t[flat[0]])
    rate_last = (sol[i] - sol[i - w]) / (t[i] - t[i - w])
    return float(rate_last / rate_flat) if abs(rate_flat) > 0 else np.nan


def measures(t, ref, sol, t_q):
    """Programmed current at the quench as a fraction of its maximum, and the solenoid ratio (NaN if missing)."""
    i = int(np.searchsorted(t, t_q, side="right")) - 1
    if i < 0 or not np.isfinite(ref).any() or not np.isfinite(ref[i]):
        return {"ref_frac": np.nan, "sol_ratio": np.nan}
    return {"ref_frac": float(ref[i] / np.nanmax(ref)), "sol_ratio": solenoid_ratio(t, ref, sol, i)}


def classify(has_quench, m):
    """The termination class from the measures (see the module docstring for the rules)."""
    if not has_quench:
        return "smooth_end"
    if not np.isfinite(m["ref_frac"]):
        return "unknown"
    if m["ref_frac"] < config.AUTO_RAMPDOWN_FRAC:
        return "programmed_rampdown"
    if not np.isfinite(m["sol_ratio"]):
        return "unknown"
    return "flux_exhausted" if m["sol_ratio"] < config.AUTO_SOL_STOPPED_RATIO else "flat_top_drop"


def label_shot(row):
    """One shot's row: shot id, operator note, quench time, measures and class."""
    t_q = float(row["t_quench"])
    out = {"shot_id": int(row["shot_id"]), "campaign": row["campaign"], "note": bool(row["note"]), "t_quench": t_q}
    ext = extras.load_extras(int(row["shot_id"]), allow_download=False)
    has_quench = np.isfinite(t_q)
    m = {"ref_frac": np.nan, "sol_ratio": np.nan}
    if ext is not None and has_quench:
        m = measures(ext["time"].to_numpy(), ext["ip_ref"].to_numpy(), ext["sol_current"].to_numpy(), t_q)
    return {**out, **m, "termination": classify(has_quench, m)}


def summarise(table):
    """Counts per class, overall and split by whether the operators noted a disruption."""
    counts = {c: int((table["termination"] == c).sum()) for c in CLASSES}
    by_note = {}
    for noted, group in table.groupby("note"):
        by_note["noted" if noted else "not_noted"] = {c: int((group["termination"] == c).sum()) for c in CLASSES}
    return {"n_shots": len(table), "n_labelled": len(table) - counts["unknown"], "counts": counts, "by_note": by_note}


def summary_text(s):
    """The short Telegram message."""
    lines = [f"Automatic labels for {s['n_shots']:,} shots (how each one ended):"]
    lines += [f"- {s['counts'][c]:,}: {WORDS[c]}" for c in CLASSES if s["counts"][c]]
    quiet = s["by_note"].get("not_noted", {})
    if quiet.get("flat_top_drop"):
        lines.append(f"Of the shots nobody called a disruption, {quiet['flat_top_drop']:,} were lost during flat-top "
                     "with flux left: candidates for disruptions that were never recorded.")
    return "\n".join(lines)


def main():
    shots = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
    table = pd.DataFrame([label_shot(row) for _, row in shots.iterrows()])
    s = summarise(table)
    os.makedirs(OUT_DIR, exist_ok=True)
    table.to_csv(os.path.join(OUT_DIR, "per_shot.csv"), index=False)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(s, f, indent=2)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary_text(s) + "\n")
    print(summary_text(s))


if __name__ == "__main__":
    main()
