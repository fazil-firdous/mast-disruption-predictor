"""MAST-Events, step 2 (v3 plan A2): one probability of disruption per shot, from evidence that disagrees.

Every shot with a fast current quench gets votes from three sources, each an imperfect witness:
- note: the operator's post-shot comment calls it a disruption (1) or not (0);
- words: the post-shot comment names a loss of the plasma (VDE, locked mode, plasma lost, early end; negations
  skipped) at a time within MATCH_MS of the quench (2), or without a matching time (1), or names none (0);
- solenoid: lost during the flat-top with flux left (1), flux exhausted or programmed ramp-down (0), unknown (abstain).
A latent-class model (Dawid & Skene 1979, the basis of 'weak supervision') learns, from how the sources agree and
disagree, how often each one says what when a shot did or did not disrupt, and gives P(disrupted) per shot. No hand
label is used. It assumes the sources are independent given the truth; the note and the words come from the same
comment, so the paper treats the probabilities as a ranking with a caveat, not as calibrated truth.

Shots without a quench get P = 0 (our definition of a disruption needs a fast quench; low-current terminations such
as 30421 fall outside it). Thornton's flags are kept out of the model on purpose: they test it (--thornton).

    python scripts/mast_label_model.py [--thornton disruption_efit_0709_full.sav]

Reads auto_labels/per_shot.csv and events/events.csv (scripts/mast_events.py); writes events/catalogue.csv and
events/label_model.txt. An analysis next to the experiments: outside every stamp.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, evaluate

EVENTS_DIR = os.path.join(config.RESULTS_DIR, "events")
LABELS_CSV = os.path.join(config.RESULTS_DIR, "auto_labels", "per_shot.csv")
LOSS = ["VDE", "locked mode", "plasma lost", "early end"]
CAUSES = ["VDE", "locked mode", "MHD/NTM", "IRE", "hardware trip"]  # most specific first, for the cause column
MATCH_MS = 20  # an operator's time within 20 ms of the quench counts as the same event (VDE and 'disrupt' times: 80-90%)
ABSTAIN = -1
SMOOTHING = 1.0  # Laplace smoothing of each source's answer table
N_ITER = 200


def votes(labels, events):
    """DataFrame of the three sources' answers for the shots with a quench, and the cause and time columns."""
    q = labels[labels["t_quench"].notna()].copy()
    post = events[events["source"] == "shot_postshot_comment"] if "source" in events else events
    q["note_vote"] = np.where(q["note"].astype(bool), 1, 0)  # 99% of shots have a comment, so silence is a 'no'
    loss = post[post["event"].isin(LOSS)].copy()
    loss["dt_ms"] = 1000 * (loss["time_s"] - loss["t_quench"])
    matched = set(loss.loc[loss["dt_ms"].abs() <= MATCH_MS, "shot_id"])
    named = set(loss["shot_id"])
    q["words_vote"] = np.where(q["shot_id"].isin(matched), 2, np.where(q["shot_id"].isin(named), 1, 0))
    q["solenoid_vote"] = q["termination"].map({"flat_top_drop": 1, "flux_exhausted": 0,
                                               "programmed_rampdown": 0}).fillna(ABSTAIN).astype(int)
    cause = {}
    for name in reversed(CAUSES):  # later assignments win, so the most specific cause is kept
        for shot in post.loc[post["event"] == name, "shot_id"]:
            cause[shot] = name
    q["cause"] = q["shot_id"].map(cause).fillna("")
    timed = loss[loss["dt_ms"].abs() <= MATCH_MS].groupby("shot_id")["time_s"].median()
    q["t_event"] = q["shot_id"].map(timed).fillna(q["t_quench"])
    return q.reset_index(drop=True)


def dawid_skene(answers, n_iter=N_ITER, smoothing=SMOOTHING, init=None):
    """Binary latent-class model. answers: (n, m) int array, each column's values 0..K-1 or ABSTAIN. Returns
    (P(y=1) per row, prior, tables) where tables[j][y] is the distribution of source j's answers given y over the rows
    where it answers. An abstention is missing data, not evidence: it leaves the row's likelihood unchanged (a first
    version modelled it as an answer and learned 'solenoid state unknown' as a strong sign of disruption).
    init: starting P(y=1) per row."""
    a = np.asarray(answers)
    n, m = a.shape
    spoke = [a[:, j] != ABSTAIN for j in range(m)]
    codes = [np.where(spoke[j], a[:, j], 0) for j in range(m)]
    sizes = [int(c[s].max()) + 1 if s.any() else 1 for c, s in zip(codes, spoke)]
    q = np.full(n, 0.5) if init is None else np.asarray(init, dtype=float)
    for _ in range(n_iter):
        prior = q.mean()
        tables = []
        for c, s, k in zip(codes, spoke, sizes):
            t = np.zeros((2, k))
            for y, w in ((1, q), (0, 1 - q)):
                t[y] = (np.bincount(c[s], weights=w[s], minlength=k) + smoothing) / (w[s].sum() + smoothing * k)
            tables.append(t)
        log1 = np.log(prior) + sum(np.where(s, np.log(t[1][c]), 0.0) for t, c, s in zip(tables, codes, spoke))
        log0 = np.log(1 - prior) + sum(np.where(s, np.log(t[0][c]), 0.0) for t, c, s in zip(tables, codes, spoke))
        new = 1 / (1 + np.exp(log0 - log1))
        if np.max(np.abs(new - q)) < 1e-8:
            q = new
            break
        q = new
    return q, float(q.mean()), tables


SOURCES = ["note_vote", "words_vote", "solenoid_vote"]


def fit(labels, events, sources=SOURCES):
    """The catalogue: every labelled shot with its ending group, P(disrupted), cause and event time; plus the
    model's prior and answer tables. sources: the vote columns to use (the robustness check leaves one out)."""
    q = votes(labels, events)
    answers = q[list(sources)].to_numpy()
    init = np.where(q["note_vote"] == 1, 0.9, np.where(q["solenoid_vote"] == 1, 0.6, 0.2))
    p, prior, tables = dawid_skene(answers, init=init)
    q["p_disrupted"] = p
    out = labels[["shot_id", "campaign", "note", "termination", "t_quench"]].merge(
        q[["shot_id", "p_disrupted", "cause", "t_event", "note_vote", "words_vote", "solenoid_vote"]],
        on="shot_id", how="left")
    out["p_disrupted"] = out["p_disrupted"].fillna(0.0)
    quench = np.isfinite(out["t_quench"].astype(float))
    out["group"] = [evaluate.ending_group(n, e, h) for n, e, h in zip(out["note"], out["termination"], quench)]
    return out, prior, tables


def robustness(labels, events):
    """The fit again without each source in turn: the estimated share disrupted among quench shots and the mean P of
    the candidates. If one source alone drove the result, leaving it out would change these a lot."""
    rows = []
    for left_out in [None] + SOURCES:
        used = [s for s in SOURCES if s != left_out]
        catalogue, prior, _ = fit(labels, events, used)
        candidate = catalogue["group"] == "candidate"
        rows.append((left_out or "none (all three)", prior, catalogue.loc[candidate, "p_disrupted"].mean()))
    return rows


def text(catalogue, prior, tables, thornton=None, robust=None):
    """The fit in words: what each source says given the truth, mean P per ending group, and the Thornton check."""
    names = {"note": ["no 'disrupt'", "'disrupt'"], "words": ["none", "named", "named, time matches"],
             "solenoid": ["flux exhausted or ramp-down", "flat-top with flux", "unknown"]}
    lines = [f"Shots with a quench: {int(catalogue['t_quench'].notna().sum()):,}; estimated share disrupted among "
             f"them: {prior:.1%}.", "", "How often each source gives each answer, if the shot disrupted / did not:"]
    for (source, labels_), t in zip(names.items(), tables):
        cells = [f"{lab}: {t[1][k]:.1%} / {t[0][k]:.1%}" for k, lab in enumerate(labels_[:t.shape[1]])]
        lines.append(f"- {source}: " + "; ".join(cells))
    lines += ["", "Mean P(disrupted) and shots with P > 0.5, by ending group:"]
    for g, part in catalogue.groupby("group"):
        lines.append(f"- {g}: {part['p_disrupted'].mean():.2f}, {int((part['p_disrupted'] > 0.5).sum()):,} of "
                     f"{len(part):,}")
    if thornton is not None:
        m = catalogue.merge(thornton, on="shot_id")
        m = m[m["t_quench"].notna() & (m["disruption"] == 1)]
        full = (m["ip_flattop_yn"] == 1) & (m["isol_constant"] == 1)
        from scripts.label_statistics import auc
        lines += ["", f"Check against Thornton (2011), {len(m):,} shared quench shots: AUC of P(disrupted) for his "
                      f"full-performance disruptions {auc(m.loc[full, 'p_disrupted'].to_numpy(), m.loc[~full, 'p_disrupted'].to_numpy()):.3f}; "
                      f"mean P {m.loc[full, 'p_disrupted'].mean():.2f} for them vs {m.loc[~full, 'p_disrupted'].mean():.2f}."]
        # his flag and our solenoid vote share the physics, so also test inside our flat-top drops, where every shot
        # has the same solenoid vote and only the note and the words can move P
        ft = m[m["termination"] == "flat_top_drop"]
        ft_full = (ft["ip_flattop_yn"] == 1) & (ft["isol_constant"] == 1)
        if ft_full.any() and (~ft_full).any():
            lines.append(f"Inside our flat-top drops only ({len(ft):,} shots, same solenoid vote): AUC "
                         f"{auc(ft.loc[ft_full, 'p_disrupted'].to_numpy(), ft.loc[~ft_full, 'p_disrupted'].to_numpy()):.3f}.")
    if robust:
        lines += ["", "Robustness, the fit without one source (left out: share disrupted among quench shots; mean P "
                      "of candidates):"]
        lines += [f"- {name}: {prior:.1%}; {cand:.2f}" for name, prior, cand in robust]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thornton", help="Thornton's database (IDL save file) for the independent check")
    args = parser.parse_args()
    labels = pd.read_csv(LABELS_CSV)
    events = pd.read_csv(os.path.join(EVENTS_DIR, "events.csv"))
    catalogue, prior, tables = fit(labels, events)
    thornton = None
    if args.thornton:
        from scripts.thornton_compare import load_thornton
        thornton = load_thornton(args.thornton)
    catalogue.to_csv(os.path.join(EVENTS_DIR, "catalogue.csv"), index=False)
    out = text(catalogue, prior, tables, thornton, robustness(labels, events))
    with open(os.path.join(EVENTS_DIR, "label_model.txt"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
