"""A second tokamak: TCABR (University of Sao Paulo), 2,189 Ohmic discharges, 435 labelled disruptive (Zenodo record
21843354, CC BY 4.0). Turns each discharge into the 1 ms frame this project uses, labels it, builds the usual causal
windows, and runs the E10 recipe (tune on validation shots, test once) on many random splits, for two input sets:

- portable: inputs without units that MAST also has (current relative to its own running peak, Mirnov fluctuation
  change against the shot's own history), for a later train-on-MAST, test-on-TCABR check;
- native: portable plus what TCABR alone measures here (loop voltage, H-alpha, hard X-rays, Mirnov levels).

Data facts (checked 2026-10-31, NOTES): Mirnov coils every 2 us (pickup voltage, i.e. dB/dt), plasma current and loop
voltage every 4 us, sigscale attributes convert to physical units; the first current sample of every shot is an
artifact (-152.6 at t = 0), so the first 5 ms are dropped. Labels are the dataset's own shot_type; 169 disruptive
shots carry a placeholder disruption time ('0.1 ms'), so the disruption time is our detector's (MAST rules, with the
two absolute current thresholds scaled by TCABR's ~8x smaller current); where both exist they agree within 5 ms.

CAUTION (NOTES 2026-10-31): the dataset's disruptive and normal shots come from different periods (all 408 shots up
to ~33,500 are disruptive, 12 of the 1,764 after), so random splits mostly reward recognising the period. Read its
results as an upper bound, and prefer the train-on-MAST, test-on-TCABR check.

    python scripts/tcabr.py /path/to/tcabr_data.nc            # build windows, then 20 splits per input set
    python scripts/tcabr.py /path/to/tcabr_data.nc --seeds 5

Writes RESULTS_DIR/tcabr/ (windows, per-split metrics, summary). Results stay private like every model number.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src import config, dataset, experiment, extras, features, labels, stats

CURRENT_SCALE = 8  # MAST plasma current over TCABR's (~700 kA vs ~85 kA): scales IP_ON and MIN_PEAK_IP only
SKIP_S = 0.005  # the first current sample of every shot is an artifact
OUT = os.path.join(config.RESULTS_DIR, "tcabr")
NATIVE = {"vloop": "VLoop", "halpha": "HAlfaRef", "hxr": "HardXRay"}


def scale_thresholds():
    """Set the plasma-on and minimum-peak currents for TCABR; every other rule stays as on MAST."""
    config.IP_ON, config.MIN_PEAK_IP = 50e3 / CURRENT_SCALE, 100e3 / CURRENT_SCALE


def signal(group, name):
    """(time in s, values in physical units) of one TCABR signal, without the first SKIP_S; None if the shot lacks it."""
    if name not in group:
        return None
    var = group[name]
    t = var.dims[0][0][:].astype(float) * 1e-6
    x = var[:].astype(float) * float(var.attrs["sigscale"][0])
    keep = t >= SKIP_S
    return t[keep], x[keep]


def shot_frame(group):
    """One discharge on a 1 ms grid in this project's column names: ip (A), the Mirnov RMS and band columns of
    src/extras.py from coil 1 (V, dB/dt), and the native signals; each value from samples up to its grid time."""
    t_ip, ip_ka = signal(group, "IPlasma")
    grid = np.arange(t_ip[0] + 1e-3, t_ip[-1], 1e-3)
    frame = pd.DataFrame({"time": grid, "ip": extras.bin_mean(t_ip, ip_ka * 1e3, grid)})
    for column in ["power_radiated", "neutron_rates_total", "power_nbi", "mirnov_rms"] + list(config.MIRNOV_BANDS_KHZ):
        frame[column] = np.nan
    coils = [f"BbMirnovN{k:02d}" for k in range(1, 22) if f"BbMirnovN{k:02d}" in group]
    if coils:
        t_m, mirnov = signal(group, coils[0])
        frame["mirnov_rms"] = extras.bin_rms(t_m, mirnov, grid)
        for name, values in extras.band_rms(t_m, mirnov, grid, float(np.median(np.diff(t_m)))).items():
            frame[name] = values
    for name, variable in NATIVE.items():
        found = signal(group, variable)
        frame[name] = extras.bin_mean(*found, grid) if found is not None else np.nan
    return frame


def shot_row(shot_id, group, frame):
    """The shot-table row: TCABR's own label, our detector's quench time, and the columns dataset.labelled_windows
    reads ('both' is the only label source here)."""
    found = labels.detect_disruption(frame["time"], frame["ip"])
    disrupted = group.attrs["shot_type"].decode() == "Disruptive"
    t_q = found["t_disrupt"]
    usable = found["status"] != "unknown" and (not disrupted or np.isfinite(t_q))
    return {"shot_id": shot_id, "campaign": "TCABR", "usable": usable, "disrupted": disrupted and usable,
            "t_quench": t_q, "t_disrupt_both": t_q if disrupted else np.nan, "note": disrupted,
            "status_both": "disrupted" if disrupted else "clean"}


def native_features(sig_frame, win):
    """Window statistics of the native signals, appended to the standard window table of one shot."""
    time = sig_frame["time"].to_numpy()
    for name in NATIVE:
        x = sig_frame[name].to_numpy()
        means, lasts, slopes = [], [], []
        for t_end in win["t_end"]:
            inside = (time <= t_end + features.EPS) & (time > t_end - config.WINDOW_MS / 1000 + features.EPS)
            xv, tv = x[inside], time[inside]
            ok = np.isfinite(xv)
            means.append(float(np.mean(xv[ok])) if ok.any() else np.nan)
            lasts.append(float(xv[-1]) if len(xv) else np.nan)
            slopes.append(features.slope(tv[ok], xv[ok]) if ok.sum() >= 2 else np.nan)
        win[f"{name}_mean"], win[f"{name}_last"], win[f"{name}_slope"] = means, lasts, slopes
    return win


def portable(win):
    """Inputs without units: current relative to its running peak (level, slope, spread, change) and the Mirnov
    change features. Both machines have them, so a model can be trained on one and tested on the other."""
    peak = win["ip_abs_last"] / win["ip_frac_of_running_max"]
    for stat in ("slope", "std", "delta"):
        win[f"ip_rel_{stat}"] = win[f"ip_abs_{stat}"] / peak
    return win


PORTABLE = (["ip_frac_of_running_max", "ip_rel_slope", "ip_rel_std", "ip_rel_delta"]
            + [f"{name}_rel" for name in ("mirnov", "mir_lf", "mir_tm", "mir_mf", "mir_hf")])
NATIVE_COLUMNS = PORTABLE + [f"{name}_{stat}" for name in ("mirnov", "mir_lf", "mir_tm", "mir_mf", "mir_hf")
                             for stat in ("mean", "slope")] + [f"{n}_{s}" for n in NATIVE for s in ("mean", "last",
                                                                                                 "slope")]


def build(path):
    """(windows, shots) for every usable TCABR discharge, cached in OUT."""
    cache = [os.path.join(OUT, name) for name in ("windows.parquet", "shots.parquet")]
    if all(os.path.exists(p) for p in cache):
        return pd.read_parquet(cache[0]), pd.read_parquet(cache[1])
    import h5py  # only here: the rest of the project never needs it

    rows, tables = [], []
    with h5py.File(path, "r") as f:
        for name in sorted(f.keys(), key=int):
            if "IPlasma" not in f[name]:
                continue
            frame = shot_frame(f[name])
            row = shot_row(int(name), f[name], frame)
            rows.append(row)
            if not row["usable"]:
                continue
            stop = row["t_quench"] if np.isfinite(row["t_quench"]) else np.inf
            win = features.shot_windows(frame, t_stop=stop)
            if len(win):
                win = native_features(frame, win)
                win.insert(0, "shot_id", int(name))
                tables.append(portable(dataset.compact(win)))
    shots = pd.DataFrame(rows)
    windows = pd.concat(tables, ignore_index=True)
    shots = shots[shots["usable"] & shots["shot_id"].isin(windows["shot_id"])].reset_index(drop=True)
    os.makedirs(OUT, exist_ok=True)
    windows.to_parquet(cache[0])
    shots.to_parquet(cache[1])
    return windows, shots


def run_split(windows, shots, seed, name, columns):
    """One random split of TCABR shots, tuned on validation and tested once (the E10 recipe); returns test metrics.
    The example-shot plots are skipped: they load shots from the MAST store."""
    experiment.plot_examples = lambda *args: None
    split = dataset.random_split(shots, seed=seed)
    win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots, source="both")
    out_dir = os.path.join(OUT, name, f"seed_{seed}")
    return experiment.run_model("hgb", win, "split", columns, out_dir, save=False, with_importance=False)["test"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path")
    parser.add_argument("--seeds", type=int, default=config.COMPARISON_SEEDS)
    args = parser.parse_args()
    scale_thresholds()
    windows, shots = build(args.path)
    print(f"TCABR: {len(shots)} usable shots ({int(shots['disrupted'].sum())} disruptive), {len(windows):,} windows")
    results = {}
    for name, columns in (("portable", PORTABLE), ("native", NATIVE_COLUMNS)):
        rows = [run_split(windows, shots, seed, name, columns) for seed in range(1, args.seeds + 1)]
        skill = np.array([r["catch_rate"] - r["false_alarm_rate"] for r in rows])
        mean, low, high = stats.corrected_interval(skill)
        results[name] = {"n_splits": len(rows), "catch_rate": float(np.mean([r["catch_rate"] for r in rows])),
                         "false_alarm_rate": float(np.mean([r["false_alarm_rate"] for r in rows])),
                         "warning_median_ms": float(np.nanmedian([r["warning_median_ms"] for r in rows])),
                         "skill": mean, "skill_low": low, "skill_high": high}
        print(name, json.dumps(results[name], default=float))
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == "__main__":
    main()
