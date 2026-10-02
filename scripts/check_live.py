"""Check the real data servers: load the shot table and shot 11860, and compare against SPEC 3.1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import config
from src.data import check_connection, load_shot, load_shot_table


def main():
    ok, message = check_connection()
    print(message)
    if not ok:
        sys.exit(1)

    table = load_shot_table(use_cache=False)
    print("shot table:", table.shape)

    shot = load_shot(config.REFERENCE_SHOT, use_cache=False)
    t = shot["time"].to_numpy()
    ip = np.abs(shot["ip"].to_numpy())
    peak = int(np.argmax(ip))
    print(f"shot {config.REFERENCE_SHOT}: {len(shot)} samples, {t[0]:.3f} to {t[-1]:.3f} s, "
          f"step {1000 * np.median(np.diff(t)):.3f} ms")
    print(f"max |Ip| {ip[peak] / 1e3:.0f} kA at {t[peak]:.3f} s; |Ip| at 0.213 s: "
          f"{ip[np.argmin(np.abs(t - 0.213))] / 1e3:.0f} kA")
    print("signals present:", [c for c in config.SIGNAL_VARIABLES if shot[c].notna().any()])

    # SPEC 3.1 reference values for this shot
    checks = {
        "385 samples": len(shot) == 385,
        "starts at -0.100 s": abs(t[0] + 0.1) < 1e-6,
        "spike ~850 kA near 0.210 s": abs(ip[peak] - 850e3) < 30e3 and abs(t[peak] - 0.210) < 0.003,
    }
    for name, passed in checks.items():
        print(("PASS " if passed else "FAIL ") + name)
    if not all(checks.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
