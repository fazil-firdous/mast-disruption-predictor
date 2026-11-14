"""Does the public preprocessed dataset of a released benchmark model carry the observation schedule?
(docs/preregistration_public_pipeline_audit.md, fixed before the file was opened.) No model; the file is opened with a
restricted unpickler that allows only numpy arrays and plain containers.

    python scripts/public_pipeline_audit.py --download     # fetch the file named in the repository's code
    python scripts/public_pipeline_audit.py                # the audit

Writes DATA_DIR/public_pipeline/ and RESULTS_DIR/sampling_audit/public_pipeline.txt. Outside every stamp.
"""
import argparse
import gzip
import io
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import sampling_audit as sa
from src import config

URL = "https://pub-651766aedb444e189ec2533015f228de.r2.dev/lucas_data_f32.pickle.gzip"  # from datamodules/lucas.py
END_CUTOFF_ROWS = 8
LAST_ROWS = 20
FACTORS = {"east": 20, "d3d": 5}  # datamodules/dhruva.py
NAMES = {"cmod": "C-Mod", "d3d": "DIII-D", "east": "EAST"}
ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
           ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
           ("numpy._core.multiarray", "scalar"), ("numpy.core.numeric", "_frombuffer"),
           ("numpy._core.numeric", "_frombuffer"), ("collections", "OrderedDict"), ("builtins", "set"),
           ("builtins", "frozenset"), ("builtins", "complex"), ("builtins", "slice")}


def folder():
    return os.path.join(config.DATA_DIR, "public_pipeline")


def path():
    return os.path.join(folder(), "lucas_data_f32.pickle.gzip")


class SafeUnpickler(pickle.Unpickler):
    """Refuses every global except numpy arrays, dtypes, scalars and plain containers."""

    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused to load {module}.{name}: not on the allowed list")


def safe_load(raw):
    data = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    return SafeUnpickler(io.BytesIO(data)).load()


def shots_of(obj):
    """[(machine, label, array)] from either layout: {key: shot} or {split: {key: shot}}."""
    out = []

    def add(d):
        if isinstance(d, dict) and {"data", "label", "machine"} <= set(d):
            out.append((str(d["machine"]), int(d["label"]), np.asarray(d["data"], dtype=float)))
            return True
        return False

    for v in (obj.values() if isinstance(obj, dict) else obj):
        if not add(v) and isinstance(v, dict):
            for w in v.values():
                add(w)
    return out


def counter_channels(a):
    """Columns that increase by the same step at every row (time or step counters)."""
    if len(a) < 3:
        return []
    d = np.diff(a, axis=0)
    return [j for j in range(a.shape[1]) if np.all(np.isfinite(d[:, j])) and d[0, j] > 0 and np.allclose(d[:, j], d[0, j])]


def change_occupancy(a, rows=None):
    """Share of row-to-row steps at which at least one column changes (both-NaN is no change); over the last `rows`."""
    if rows is not None:
        a = a[-(rows + 1):]
    if len(a) < 2:
        return np.nan
    x, y = a[:-1], a[1:]
    same = (x == y) | (np.isnan(x) & np.isnan(y))
    return float(np.mean(~np.all(same, axis=1)))


def resample_array(arr, c):
    """datamodules/dhruva.py, as written: keep the first row of every block of c rows, repeated over the block."""
    n = (arr.shape[0] // c) * c
    arr = arr[:n]
    r = arr.reshape(n // c, c, arr.shape[1])
    return np.repeat(r[:, :1, :], c, axis=1).reshape(n, arr.shape[1])


def table(shots):
    rows = []
    for machine, label, a in shots:
        if a.ndim != 2:
            continue
        counters = counter_channels(a)
        sig = np.delete(a, counters, axis=1)
        cut = sig[:-END_CUTOFF_ROWS] if len(sig) > END_CUTOFF_ROWS else sig[:0]
        p3 = np.nan
        if machine in FACTORS:
            rs = resample_array(sig, FACTORS[machine])
            rs = rs[:-END_CUTOFF_ROWS] if len(rs) > END_CUTOFF_ROWS else rs[:0]
            p3 = change_occupancy(rs, LAST_ROWS)
        else:
            p3 = change_occupancy(cut, LAST_ROWS)
        rows.append({"machine": machine, "label": label, "rows": len(a), "counters": tuple(counters),
                     "p1": change_occupancy(cut, LAST_ROWS), "p2": change_occupancy(cut), "p3": p3})
    return rows


def text(rows):
    import pandas as pd
    t = pd.DataFrame(rows)
    lines = ["Public-pipeline audit (docs/preregistration_public_pipeline_audit.md): the default data module's preprocessed "
             f"file, last {END_CUTOFF_ROWS} rows removed; change occupancy (share of row steps where any signal changes), "
             "no value read beyond equality. AUC disrupted against not.", ""]
    for m in sorted(t["machine"].unique()):
        p = t[t["machine"] == m]
        y = p["label"].to_numpy().astype(bool)
        lines.append(f"== {NAMES.get(m, m)}: {len(p):,} shots ({int(y.sum()):,} disrupted); median rows {p['rows'].median():.0f}; "
                     f"counter channels excluded: {sorted(set(p['counters']))}")
        for key, label in (("p1", f"PRIMARY P1, last {LAST_ROWS} rows"), ("p2", "P2, whole record"),
                           ("p3", "P3, after resample_array")):
            r = sa.auc_range(p[key].to_numpy(dtype=float), y)
            verdict = sa.reading(r)
            if r[1] is not None and r[2] < 0.5 and abs(r[0] - 0.5) > 0.10:
                verdict = "REVERSED association (reported)"
            lines.append(f"  {label}: AUC {sa.fmt(r)}: {verdict}; median disrupted {p.loc[p['label'] == 1, key].median():.2f}, "
                         f"not {p.loc[p['label'] == 0, key].median():.2f}")
        lines.append("")
    lines.append("Not claimed: that the reported DisruptionBench or CCNN results used this module, or that any reported "
                 "performance is affected.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        import requests
        os.makedirs(folder(), exist_ok=True)
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(path(), "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"downloaded {os.path.getsize(path()) / 1e6:.0f} MB to {path()}")
    if not os.path.exists(path()):
        sys.exit(f"no {path()}: run with --download first")
    with open(path(), "rb") as f:
        obj = safe_load(f.read())
    shots = shots_of(obj)
    print(f"{len(shots):,} shots read", flush=True)
    out = text(table(shots))
    os.makedirs(sa.folder(), exist_ok=True)
    with open(os.path.join(sa.folder(), "public_pipeline.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
