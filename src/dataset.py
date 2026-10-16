"""Shot selection, window table construction, and shot-level train/val/test splits."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import config, extras, features, labels
from src.data import load_shot, log_skipped

LABEL_SOURCES = ("both", "signal", "note")


def candidate_ids(table, n_clean, seed=config.RANDOM_SEED):
    """Shots worth downloading: every note-disrupted shot, plus a random pool of note-clean ones.

    Under LABEL_SOURCE 'both' a disruption needs a note, so the note-disrupted shots are the whole supply.
    The clean pool is CLEAN_POOL_FACTOR times n_clean because some shots turn out unknown once labelled.
    """
    rng = np.random.default_rng(seed)
    is_note = table["shot_postshot_comment"].map(labels.note_label).to_numpy()
    disrupted = table.loc[is_note, "shot_id"].to_numpy()
    clean = table.loc[~is_note, "shot_id"].to_numpy()
    n_pool = min(int(n_clean * config.CLEAN_POOL_FACTOR), len(clean))
    clean = rng.choice(clean, size=n_pool, replace=False)
    return sorted(int(i) for i in np.concatenate([disrupted, clean]))


def label_shot(shot_id, comment, campaign):
    """One row describing a shot: its label under every LABEL_SOURCE, the detector's quench time, campaign."""
    shot = load_shot(shot_id)
    row = {"shot_id": shot_id, "campaign": campaign}
    for source in LABEL_SOURCES:
        lab = labels.shot_label(shot["time"], shot["ip"], comment, source)
        row[f"status_{source}"] = lab["status"]
        row[f"t_disrupt_{source}"] = lab["t_disrupt"]
        row["t_quench"] = lab["t_quench"]
        row["note"] = lab["note"]
    configured = config.LABEL_SOURCE
    row["status"] = row[f"status_{configured}"]
    row["disrupted"] = row["status"] == "disrupted"
    row["t_disrupt"] = row[f"t_disrupt_{configured}"]
    return row


def label_shots(ids, table):
    """Label every shot in ids (all must be cached). Shots with status 'unknown' are logged to skipped.csv."""
    info = table.set_index("shot_id")
    rows = []
    for shot_id in tqdm(ids, desc="labelling"):
        rows.append(label_shot(shot_id, info.at[shot_id, "shot_postshot_comment"], info.at[shot_id, "campaign"]))
    shots = pd.DataFrame(rows)
    unknown = shots[shots["status"] == "unknown"]
    log_skipped([[s, f"label unknown: |Ip| never above {config.MIN_PEAK_IP:.0f} A or no usable time"]
                 for s in unknown["shot_id"]])
    return shots[shots["status"] != "unknown"].reset_index(drop=True)


def select_shots(shots, n_disrupted, n_clean, seed=config.RANDOM_SEED):
    """Seeded random sample of n_disrupted disrupted and n_clean clean shots (None means take all of them)."""
    rng = np.random.default_rng(seed)
    picked = []
    for flag, n in [(True, n_disrupted), (False, n_clean)]:
        ids = shots.loc[shots["disrupted"] == flag, "shot_id"].to_numpy()
        if n is not None and n < len(ids):
            ids = rng.choice(ids, size=n, replace=False)
        picked += list(ids)
    return shots[shots["shot_id"].isin(picked)].sort_values("shot_id").reset_index(drop=True)


def stop_time(row):
    """Time at which a shot's windows end (exclusive).

    With STOP_AT_QUENCH every shot ends at the detector's quench, so clean and disrupted shots end the same way;
    the label-source times are included so any source can be relabelled later without rebuilding (E7).
    """
    candidates = [row["t_disrupt"]]
    if config.STOP_AT_QUENCH:
        candidates += [row["t_quench"]] + [row[f"t_disrupt_{source}"] for source in LABEL_SOURCES]
    finite = [t for t in candidates if np.isfinite(t)]
    return min(finite) if finite else np.inf


def build_windows(shots):
    """Window features for every shot, one row per window, with shot_id first. Shots with no windows are logged."""
    tables, empty = [], []
    for _, row in tqdm(shots.iterrows(), total=len(shots), desc="windows"):
        shot_id = int(row["shot_id"])
        win = features.shot_windows(extras.with_extras(load_shot(shot_id), shot_id), t_stop=stop_time(row))
        if len(win) == 0:
            empty.append([int(row["shot_id"]), "no windows (no visible breakdown, or plasma too short)"])
            continue
        win.insert(0, "shot_id", int(row["shot_id"]))
        tables.append(win)
    log_skipped(empty)
    return pd.concat(tables, ignore_index=True)


def window_labels(t_end, t_disrupt, horizon_ms=config.HORIZON_MS):
    """1 for windows with t_disrupt - horizon <= t_end < t_disrupt, else 0. t_disrupt NaN (clean) gives all 0."""
    t_end = np.asarray(t_end, dtype=float)
    if not np.isfinite(t_disrupt):
        return np.zeros(len(t_end), dtype=int)
    inside = (t_end >= t_disrupt - horizon_ms / 1000 - features.EPS) & (t_end < t_disrupt - features.EPS)
    return inside.astype(int)


def labelled_windows(windows, shots, source=None, horizon_ms=config.HORIZON_MS):
    """Windows with label, disrupted, t_disrupt, t_quench and near_quench columns for one label source.

    Windows at or after t_disrupt are dropped (SPEC 6). near_quench marks clean-shot windows within the horizon
    before an unnoted quench; training may drop them (NEAR_QUENCH_CLEAN), evaluation always keeps them.
    """
    source = source or config.LABEL_SOURCE
    info = shots.set_index("shot_id")
    out = windows.copy()
    out["t_disrupt"] = out["shot_id"].map(info[f"t_disrupt_{source}"]).astype(float)
    out["disrupted"] = np.isfinite(out["t_disrupt"])
    out = out[~(out["disrupted"] & (out["t_end"] >= out["t_disrupt"] - features.EPS))].reset_index(drop=True)

    t_end = out["t_end"].to_numpy()
    horizon = horizon_ms / 1000
    # same rule as window_labels, written for all shots at once (NaN comparisons are False, so clean rows get 0)
    t_disrupt = out["t_disrupt"].to_numpy()
    out["label"] = ((t_end >= t_disrupt - horizon - features.EPS) & (t_end < t_disrupt - features.EPS)).astype(int)
    t_quench = out["shot_id"].map(info["t_quench"]).to_numpy(dtype=float)
    out["t_quench"] = t_quench
    near = (t_end >= t_quench - horizon - features.EPS) & (t_end < t_quench - features.EPS)
    out["near_quench"] = near & ~out["disrupted"].to_numpy()
    return out


def random_split(shots, fractions=config.SPLIT_FRACTIONS, seed=config.RANDOM_SEED):
    """shot_id -> 'train' / 'val' / 'test', stratified by the disrupted flag, so whole shots stay together."""
    rng = np.random.default_rng(seed)
    split = {}
    for flag in (True, False):
        ids = shots.loc[shots["disrupted"] == flag, "shot_id"].to_numpy()
        ids = rng.permutation(ids)
        n_train = int(round(fractions[0] * len(ids)))
        n_val = int(round(fractions[1] * len(ids)))
        for i, shot_id in enumerate(ids):
            split[int(shot_id)] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return pd.Series(split, name="split")


def temporal_split(shots, seed=config.RANDOM_SEED):
    """shot_id -> split for SPEC 8.4: M5-M7 train (with a stratified val slice), M8-M9 test, others 'none'."""
    rng = np.random.default_rng(seed)
    split = {int(s): "none" for s in shots["shot_id"]}
    early = shots[shots["campaign"].isin(config.TEMPORAL_TRAIN_CAMPAIGNS)]
    for flag in (True, False):
        ids = rng.permutation(early.loc[early["disrupted"] == flag, "shot_id"].to_numpy())
        n_val = int(round(config.TEMPORAL_VAL_FRACTION * len(ids)))
        for i, shot_id in enumerate(ids):
            split[int(shot_id)] = "val" if i < n_val else "train"
    for shot_id in shots.loc[shots["campaign"].isin(config.TEMPORAL_TEST_CAMPAIGNS), "shot_id"]:
        split[int(shot_id)] = "test"
    return pd.Series(split, name="split_temporal")
