"""Drift study, part 2 (docs/preregistration_drift_oof.md, fixed before any model of it): every pair scored in every
repetition. 5 folds that keep linked shots together x 10 repetitions x 3 models, trained like the v3 baseline; each
shot scored by the model of its own fold. Populations and statistic exactly as scripts/drift.py (read from its
saved pairs.csv).

    python scripts/drift_oof.py --train      # the runs not done yet (full model first), then the summary
    python scripts/drift_oof.py --summary    # the comparison in words, from the saved runs

Writes RESULTS_DIR/drift_oof/. Never writes into v3/ or drift/. Outside every stamp.
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import drift, fast_magnetics, v3_explore
from src import config, data, dataset, experiment, extras, features, notify, server

FOLDS, REPEATS = 5, range(1, 11)
MODELS = ["baseline", "machine", "programme"]  # the full model's runs first
NAMES = drift.MODELS
VAL_SHARE = config.SPLIT_FRACTIONS[1] / (config.SPLIT_FRACTIONS[0] + config.SPLIT_FRACTIONS[1])  # 0.25 of the rest
NO_BEAM_W, SAME_BEAM = 5e4, 0.10
N_BOOT = 2000


def folder():
    return os.path.join(config.RESULTS_DIR, "drift_oof")


def run_dir(model, repeat, fold):
    return os.path.join(folder(), model, f"rep_{repeat}", f"fold_{fold}")


def load_pairs():
    path = os.path.join(drift.folder(), "pairs.csv")
    if not os.path.exists(path):
        raise RuntimeError("no drift/pairs.csv: run python scripts/drift.py first (part 1 writes the populations)")
    return pd.read_csv(path)


# ------------------------------------------------------------------ folds that keep linked shots together

def shot_groups(pairs):
    """{shot: group id}: shots linked through any pair (E-F), and an A triplet's lost shot L with its F."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e, f, lost in zip(pairs["E"], pairs["F"], pairs["L"]):
        parent[find(int(e))] = find(int(f))
        if lost > 0:
            parent[find(int(lost))] = find(int(f))
    return {s: find(s) for s in list(parent)}


def folds(shot_ids, groups, repeat, n_folds=FOLDS):
    """{shot: fold}: whole groups (others alone) in a seeded random order, each to the fold with fewest shots."""
    members = {}
    for s in shot_ids:
        members.setdefault(groups.get(int(s), ("alone", int(s))), []).append(int(s))
    keys = list(members)
    order = np.random.default_rng(repeat).permutation(len(keys))
    size, out = np.zeros(n_folds, int), {}
    for i in order:
        k = int(np.argmin(size))
        for s in members[keys[i]]:
            out[s] = k
        size[k] += len(members[keys[i]])
    return out


def split_for(fold_of, fold, repeat):
    """shot -> 'test' (this fold), 'val' (a quarter of the rest, seeded) or 'train'."""
    rest = np.array(sorted(s for s, k in fold_of.items() if k != fold))
    rng = np.random.default_rng(1000 * repeat + fold)
    val = set(rng.choice(rest, int(round(VAL_SHARE * len(rest))), replace=False).tolist()) if len(rest) else set()
    return pd.Series({s: "test" if k == fold else "val" if s in val else "train" for s, k in fold_of.items()}, name="split")


# ------------------------------------------------------------------ training

def run(model, repeat, fold, tables, groups):
    windows, shots = tables
    fold_of = folds(shots["shot_id"], groups, repeat)
    split = split_for(fold_of, fold, repeat)
    win = dataset.labelled_windows(windows.assign(split=windows["shot_id"].map(split)), shots)
    return fast_magnetics.fit(win, "split", model, run_dir(model, repeat, fold),
                              {"repeat": repeat, "fold": fold, "seed": 0}, v3_explore.recipe_columns(model))


def jobs():
    return [(m, r, k) for m in MODELS for r in REPEATS for k in range(FOLDS)]


def done(job):
    return server.result_saved(os.path.join(run_dir(*job), "metrics.json"))


def train_all(say=print):
    todo = [j for j in jobs() if not done(j)]
    say(f"{len(todo)} runs to do (full model first)")
    full = [i for i, j in enumerate(todo) if j[0] == MODELS[0]]
    if todo:
        tables = experiment.load_tables()
        groups = shot_groups(load_pairs())
        for i, (m, rep, k) in enumerate(todo):
            start = time.time()
            met = run(m, rep, k, tables, groups)
            say(f"[{i + 1}/{len(todo)}] {NAMES[m]} repetition {rep} fold {k}: pAUC {met['test']['pauc']:.3f}, "
                f"{(time.time() - start) / 60:.1f} min")
            if full and i == full[-1] and i < len(todo) - 1:  # the primary is complete: report it before the rest
                out = summary()
                say(out)
                notify.send("Drift study part 2: the full model's runs are done (primary)\n\n" + out)
    out = summary()
    say(out)
    notify.send("Drift study part 2: all runs are done\n\n" + out)


# ------------------------------------------------------------------ the comparison

def repetition_scores(model, repeat):
    """All shots' out-of-fold test-window scores of one repetition, or None if a fold is missing."""
    parts = []
    for k in range(FOLDS):
        path = run_dir(model, repeat, k)
        if not server.result_saved(os.path.join(path, "metrics.json")):
            return None
        parts.append(pd.read_parquet(os.path.join(path, "test_scores.parquet")))
    return pd.concat(parts, ignore_index=True)


def step_values(scores, pairs, last, t_loss):
    """Per pair (A only, else NaN): 1 if K2 -> K1 and K1 -> L move the same way, 0 if not (drift.step_agreement)."""
    by = {s: (g["t_end"].to_numpy(dtype=float), g["score"].to_numpy(dtype=float)) for s, g in scores.groupby("shot_id")}
    out = np.full(len(pairs), np.nan)
    for i, row in enumerate(pairs.itertuples()):
        if row.pop != "A" or not all(s in by for s in (row.E, row.F, row.L)) or not np.isfinite(row.t1):
            continue
        e, f = drift.shot_mean(*by[row.E], row.t0, row.t1), drift.shot_mean(*by[row.F], row.t0, row.t1)
        t1 = min(last.get(row.F, np.nan), t_loss.get(row.L, np.nan)) - drift.HORIZON_S
        f2, lost = drift.shot_mean(*by[row.F], row.t0, t1), drift.shot_mean(*by[row.L], row.t0, t1)
        if all(np.isfinite(v) for v in (e, f, f2, lost)) and f != e and lost != f2:
            out[i] = float(np.sign(f - e) == np.sign(lost - f2))
    return out


def collect(pairs, last=None, t_loss=None):
    """{model: repetitions x pairs wins}, and the full model's step values (repetitions x pairs) when asked."""
    wins, steps = {}, []
    for model in MODELS:
        rows = []
        for r in REPEATS:
            scores = repetition_scores(model, r)
            if scores is None:
                continue
            rows.append(drift.run_wins(scores, pairs))
            if model == MODELS[0] and last is not None:
                steps.append(step_values(scores, pairs, last, t_loss))
        if rows:
            wins[model] = np.vstack(rows)
    return wins, (np.vstack(steps) if steps else None)


def two_level(w, pairs, mask=None, seed=config.RANDOM_SEED, n_boot=N_BOOT, value="delta"):
    """(mean, low, high): each draw picks a repetition and resamples groups of linked pairs. value: 'delta'
    (P_A - P_B), 'A' or 'B' (one population's P). mask limits the pairs."""
    keep = np.ones(len(pairs), bool) if mask is None else np.asarray(mask, bool)
    p, w = pairs[keep].reset_index(drop=True), w[:, keep]
    is_a = (p["pop"] == "A").to_numpy()
    groups = v3_explore.shot_groups(p["E"].to_numpy(), p["F"].to_numpy())
    _, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == k)[0] for k in range(inverse.max() + 1)] if len(inverse) else []

    def stat(row, idx):
        a, b = row[idx][is_a[idx]], row[idx][~is_a[idx]]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        pa, pb = (a.mean() if len(a) else np.nan), (b.mean() if len(b) else np.nan)
        return pa - pb if value == "delta" else pa if value == "A" else pb

    every = np.arange(len(p))
    point = float(np.nanmean([stat(row, every) for row in w])) if len(w) else np.nan
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot if members else 0):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        draws.append(stat(w[rng.integers(0, len(w))], idx))
    draws = np.array([d for d in draws if np.isfinite(d)])
    if not len(draws):
        return point, None, None
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def beam_matched(pairs, means):
    """Heating-matched pairs: both mean beam powers below NO_BEAM_W, or within SAME_BEAM of the larger."""
    e, f = means["E"].to_numpy(dtype=float), means["F"].to_numpy(dtype=float)
    big = np.maximum(e, f)
    with np.errstate(invalid="ignore"):
        return ((e < NO_BEAM_W) & (f < NO_BEAM_W)) | (np.abs(e - f) <= SAME_BEAM * big)


def beam_means(pairs, load_signals, say=None):
    """Each pair's mean beam power of E and F over the shared time (each shot loaded once)."""
    need = {}
    for i, row in enumerate(pairs.itertuples()):
        if np.isfinite(row.t0) and np.isfinite(row.t1) and row.t1 > row.t0:
            for side, shot in (("E", row.E), ("F", row.F)):
                need.setdefault(int(shot), []).append((i, side, row.t0, row.t1))
    out = {"E": np.full(len(pairs), np.nan), "F": np.full(len(pairs), np.nan)}
    for k, (shot, wanted) in enumerate(sorted(need.items()), 1):
        sig = load_signals(shot)
        if sig is not None and "pnbi" in sig:
            t, x = sig["time"].to_numpy(dtype=float), sig["pnbi"].to_numpy(dtype=float)
            for i, side, t0, t1 in wanted:
                ok = (t >= t0) & (t <= t1) & np.isfinite(x)
                out[side][i] = float(np.mean(x[ok])) if ok.sum() >= drift.MIN_WINDOWS else np.nan
        if say and (k % 2000 == 0 or k == len(need)):
            say(f"  beam power: [{k}/{len(need)}] shots")
    return pd.DataFrame(out)


def r(t, signed=False):
    return drift.r(t, signed)


def text(pairs, wins, steps, matched, campaign):
    a, b = pairs[pairs["pop"] == "A"], pairs[pairs["pop"] == "B"]
    lines = ["Drift study, part 2 (docs/preregistration_drift_oof.md): every pair scored in every repetition "
             "(5 grouped folds x up to 10 repetitions, out-of-fold scores). Same populations and statistic as part 1.",
             f"Pairs: A {len(a):,}, B {len(b):,}. Range: two-level bootstrap (repetition, then groups of linked pairs).", ""]
    if MODELS[0] in wins:
        w = wins[MODELS[0]]
        d = two_level(w, pairs)
        reps = w.shape[0]
        lo, hi = d[1], d[2]
        if lo is None:
            reading = "No range yet."
        elif lo > 0 and d[0] >= drift.WORTH:
            reading = ("Resolved, positive and at least 0.03: before a loss, the later good repeat scores higher more "
                       "often than in repeats where no loss follows; losses come at the end of a cross-shot trend in "
                       "these scores.")
        elif lo > 0:
            reading = "Resolved and positive but under 0.03: a trend too small to matter for evaluation."
        elif hi < 0:
            reading = "Resolved and negative: reported as such; no reading was fixed for it."
        elif hi < drift.WORTH:
            reading = "Range includes zero and excludes +0.03: no worthwhile difference; part 1's estimate was not borne out."
        else:
            reading = "Range includes both zero and +0.03: still unresolved at this sample size."
        per_rep = [np.nanmean(row[(pairs["pop"] == "A").to_numpy()]) - np.nanmean(row[(pairs["pop"] == "B").to_numpy()])
                   for row in w]
        mean_w = np.nanmean(w, axis=0)
        pool = drift.pooled(mean_w[None, :], pairs)
        lines += [f"PRIMARY (full model, {reps} repetitions): Delta = P_A - P_B = {r(d, True)}. {reading}",
                  f"- P_A {r(two_level(w, pairs, value='A'))}, P_B {r(two_level(w, pairs, value='B'))}",
                  f"- beside it: over repetitions only {min(per_rep):+.3f} to {max(per_rep):+.3f}; pairs only (repetition "
                  f"means) {r(pool[2], True)}; pairs tested A {pool[3]:,}, B {pool[4]:,}", ""]
        if matched is not None:
            dm = two_level(w, pairs, matched)
            share_a = float(matched[(pairs['pop'] == 'A').to_numpy()].mean())
            share_b = float(matched[(pairs['pop'] == 'B').to_numpy()].mean())
            if d[1] is not None and d[1] > 0:
                if dm[1] is not None and dm[1] > 0:
                    beam = "Resolved and positive there too: the trend is not only more beam power shot by shot."
                elif dm[0] < d[0] / 2:
                    beam = "Not resolved there, and under half the primary: consistent with an operator scan in beam power."
                else:
                    beam = "Undecided between the two."
            else:
                beam = "(read only if the primary is resolved and positive)"
            lines += [f"KEY SECONDARY, heating-matched pairs (A {share_a:.0%}, B {share_b:.0%} of pairs): Delta {r(dm, True)}. "
                      + beam, ""]
    lines.append("Each model (P_A | P_B | Delta):")
    for model in MODELS:
        if model in wins:
            w = wins[model]
            lines.append(f"- {NAMES[model]} ({w.shape[0]} repetitions): {r(two_level(w, pairs, value='A'))} | "
                         f"{r(two_level(w, pairs, value='B'))} | {r(two_level(w, pairs), True)}")
    if MODELS[0] in wins:
        w = wins[MODELS[0]]
        lines += ["", "Full model, Delta by shot gap:"]
        for lo_, hi_, label in drift.GAPS:
            mask = ((pairs["gap"] >= lo_) & (pairs["gap"] <= hi_)).to_numpy()
            lines.append(f"- gap {label}: {r(two_level(w, pairs, mask), True)}")
        lines += ["", "Full model, Delta by campaign:"]
        camp = pairs["F"].map(campaign).fillna("Unknown")
        for c in sorted(camp.unique()):
            lines.append(f"- {c}: {r(two_level(w, pairs, (camp == c).to_numpy()), True)}")
    if steps is not None:
        st = two_level(steps, pairs, (pairs["pop"] == "A").to_numpy(), value="A")
        lines += ["", f"Within A, full model: K2 -> K1 and K1 -> L move the same way in {r(st)} of triplets "
                      "(0.5 = no link)."]
    return "\n".join(lines)


def summary(load_signals=None):
    pairs = load_pairs()
    labels = pd.read_csv(os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv"))
    windows = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "windows.parquet"), columns=["shot_id", "t_end"])
    last = windows.groupby("shot_id")["t_end"].max().to_dict()
    wins, steps = collect(pairs, last, labels.set_index("shot_id")["t_quench"].to_dict())
    beam_path = os.path.join(folder(), "beam_means.csv")
    if os.path.exists(beam_path):
        means = pd.read_csv(beam_path)
    else:
        def default_load(shot):
            try:
                return features.prepare_signals(extras.with_extras(data.load_shot(int(shot)), int(shot)))
            except Exception:
                return None
        means = beam_means(pairs, load_signals or default_load)
        os.makedirs(folder(), exist_ok=True)
        means.to_csv(beam_path, index=False)
    campaign = data.load_shot_table().set_index("shot_id")["campaign"].to_dict()
    with warnings.catch_warnings():  # empty slices (no triplet with all three shots tested) give NaN quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        out = text(pairs, wins, steps, beam_matched(pairs, means), campaign)
    os.makedirs(folder(), exist_ok=True)
    with open(os.path.join(folder(), "summary.txt"), "w") as f:
        f.write(out + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    say = lambda line: print(line, flush=True)  # noqa: E731
    try:
        if args.train:
            train_all(say)
        elif args.summary:
            say(summary())
        else:
            parser.error("give --train or --summary")
    except RuntimeError as error:
        say(f"STOPPED: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
