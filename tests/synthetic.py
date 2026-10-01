"""Fake shots for tests: ramp-up, flat-top, then a sharp quench or a slow ramp-down."""
import numpy as np
import pandas as pd


def make_shot(kind="quench", t_quench=0.22, spike=True, flat_ip=7e5, noise=3e3, sign=1, seed=0,
              t_end=0.4, rampdown_ms=60, t_rampdown=0.25):
    """Return a DataFrame with the five standard columns on a 1 ms grid from -0.1 s.

    kind: 'quench' (sharp quench at t_quench, optional spike), 'rampdown' (slow controlled end) or 'tiny' (30 kA only).
    """
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(-0.1, t_end, 0.001), 6)
    ip = np.zeros_like(t)
    ramp = (t >= 0) & (t < 0.1)
    ip[ramp] = flat_ip * t[ramp] / 0.1
    ip[t >= 0.1] = flat_ip

    if kind == "quench":
        after = t >= t_quench
        ip[after] = 0.0
        # three-sample fall: ~70%, ~35%, ~5% of flat-top
        for step, frac in enumerate([0.7, 0.35, 0.05], start=1):
            ip[np.isclose(t, t_quench + step * 0.001)] = flat_ip * frac
        if spike:
            ip[np.isclose(t, t_quench)] = flat_ip * 1.2
    elif kind == "rampdown":
        down = (t >= t_rampdown) & (t < t_rampdown + rampdown_ms / 1000)
        ip[t >= t_rampdown] = 0.0
        ip[down] = flat_ip * (1 - (t[down] - t_rampdown) / (rampdown_ms / 1000))
    elif kind == "tiny":
        ip = np.where(t >= 0, 3e4, 0.0)

    ip = sign * (ip + rng.normal(0, noise, len(t)))
    return pd.DataFrame({
        "time": t,
        "ip": ip,
        "power_radiated": 1e5 + 2e4 * rng.random(len(t)),
        "neutron_rates_total": 1e13 * np.ones(len(t)),
        "power_nbi": np.where(t > 0.05, 1.5e6, 0.0),
    })
