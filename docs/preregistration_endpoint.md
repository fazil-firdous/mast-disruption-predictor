# Does the evaluation depend on where the negative windows come from? (protocol, endpoint module)

Written 2026-11-21, before any score of this study was computed. The git commit that adds this file is its timestamp.
First module of the audit toolkit (NOTES.md 2026-11-21). Changes after the first result
go at the bottom.

## Question

Endpoint-aligned evaluations often take the negative (non-disrupted) examples near the end of their own records,
while the positive examples are taken before a disruption. A fitted predictor can then be compared on different
plasma phases and on end-of-record context. For a frozen predictor and the same eligible pairs, how much does the
matched comparison change when the kept twin is scored near its own natural end instead of at the case's programme
time?

Related prior work: Zhu et al. (2021) sampled non-disruptive training sequences from the flat-top and evaluated whole
flat-tops; outcome-anchored designs in clinical time series (Sherman et al. 2018; Agniel et al. 2018). The comparison
here holds the fitted model, the pairs and the positive windows fixed and changes only the negative evaluation point.

## Data and models (saved runs only; nothing is trained)

The twin pairs of the twin paper (case = recorded or candidate lost discharge; twin = nearest earlier kept discharge
with a similar current programme), the held-out window scores of the corrected full model A_c (primary), the
machine-side model and the programme-and-time model, 50 random splits. A pair is used in a split when both discharges
are in its test set.

## The two arms (case scored identically in both)

For a lead h, the case L is scored by its latest window ending in (t_q - h - 5 ms, t_q - h], t_q its detected current
quench (as in the fixed-lead statistic).
- **Arm P (phase-matched):** the twin K by its latest window ending in the same interval of programme time.
- **Arm E (natural end):** the twin K by its latest window ending in (e_K - h - 5 ms, e_K - h], where e_K is K's own
  end: its detected current quench if it has one (flux exhaustion, programmed ramp-down), otherwise the end of its
  last scored window (smooth end).
A pair counts at lead h only if L and K have the windows needed in both arms; the same pairs enter both arms.

Pair win w = 1 if the case scores higher, 1/2 for a tie, 0 otherwise; C_P and C_E are the per-split means; Delta =
C_E - C_P.

## Primary

Delta for A_c at h = 10 ms, over the 50 splits (corrected range, as in the twin paper), and pooled over pairs
(resampling groups of pairs that share a twin; conditional on the fitted models).

## Reading (fixed now, two-sided)

- |Delta| at least 0.03 and its range over splits excluding 0: the matched comparison for this predictor depends
  materially on which negative windows are used; the sign says in which direction (Delta < 0: the twin scores higher
  near its own end than at the case's programme time).
- Range over splits including 0, or |Delta| under 0.03: no material endpoint sensitivity is shown for this predictor
  and these pairs at 10 ms (not shown to be equivalent).
- In every case: a difference shows sensitivity to the evaluation point, which changes the plasma phase and possibly the
  measurements as well as the end-of-record context; it does not by itself show that future information entered the
  model.

## Secondary

- Delta at 30, 50 and 100 ms; for the machine-side and programme-and-time models.
- Delta by the twin's ending class (flux exhausted, smooth end, programmed ramp-down).
- The twin's mean score at the two points, and the median time between them (e_K - h minus t_q - h).
- An elapsed-time check: the median elapsed time since plasma start at the two twin points.

## Changes after the first result

### Follow-up diagnostic 1 (2026-11-21, after the first result; it explains nothing it was not written to check)

The first result (NOTES.md 2026-11-21) showed Delta +0.115 at 10 ms and a negative point estimate at 100 ms
(-0.066, -0.142 to +0.011 over splits; pooled -0.063). One small, saved-score diagnostic,
specified now, before it is computed. No refit, no threshold search, no tuned phase proxy.

1. **Common cohort.** The pair-split rows eligible at both 10 and 100 ms; exclusions by lead; Delta at 10 and 100 ms
   recomputed on this intersection (over splits and pooled), reported separately from the original.
2. **Timing.** The nominal gap between the two twin points (it is the same at every lead for a pair) and the actual
   gap between the selected window ends, by lead.
3. **Orderings.** On the intersection, the twin's score change s_E - s_P, and the counts of pairs whose order against
   the case changes in each direction (win to loss, loss to win, into and out of ties), by lead.
4. **Phase from the requested programme only.** Phase of a time point from the discharge's own programmed plasma
   current (not from any diagnostic): ramp-up before the first time it reaches 90% of its peak (the twin paper's
   flat-top start), ramp-down after the last time it is at or above 90% of its peak, flat-top in between. Tabulated for
   the case point, the twin at P and the twin at E, by lead. The 90% rule is the one already used for the flat-top start;
   it is not tuned here.
5. **Ties.** The share of tied pairs in each arm, by model (for the programme-and-time model, ties are expected where
   the two discharges' inputs coincide).

Reading: descriptive. If the phase tables differ between leads, that is reported as consistent with a phase
difference, not as its proof; if they do not, the mechanism is left unresolved. Wording rules from the review: "same
programme time", not "same plasma phase"; "matched concordance", not AUC; the 100 ms sign change is not called an
established reversal.
