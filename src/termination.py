"""How did each shot end? Automatic termination classes from physics rules (ROADMAP A4, NOTES 2026-10-25).

For every shot where the detector found a fast current quench, two signals at the quench time decide the class:
- the programmed plasma current (pulse_schedule): already ramping down -> 'programmed_rampdown';
- the central solenoid: if it had (nearly) stopped sweeping, the machine had run out of flux -> 'flux_exhausted';
- otherwise the plasma was lost during the programmed flat-top with flux still available -> 'flat_top_drop',
  the ending most likely to be an instability-driven disruption.
Shots with no fast quench are 'smooth_end'; missing signals give 'unknown'. The classes say when the plasma was lost,
not whether it disrupted. They feed the 'physics' label source and the E15 recipes (src/dataset.py).
"""
import numpy as np

from src import config

CLASSES = ["flat_top_drop", "flux_exhausted", "programmed_rampdown", "smooth_end", "unknown"]


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


def shot_ending(t_quench, extras_frame):
    """(class, measures) for one shot from its detector quench time and its extras (None if it has none)."""
    has_quench = bool(np.isfinite(t_quench))
    m = {"ref_frac": np.nan, "sol_ratio": np.nan}
    if extras_frame is not None and has_quench:
        m = measures(extras_frame["time"].to_numpy(), extras_frame["ip_ref"].to_numpy(),
                     extras_frame["sol_current"].to_numpy(), t_quench)
    return classify(has_quench, m), m
