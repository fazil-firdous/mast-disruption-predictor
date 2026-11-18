# Saddle-coil sign correction of the locked-mode feature (protocol)

Written 2026-11-15, after the no-locked-mode check (`docs/preregistration_no_lm.md`, NOTES.md 2026-11-15) and an
outside review that said the ablation is a robustness check, not a correction of a known feature-construction error.
Written before any saddle-coil channel data were read for this study and before any model of it was trained. The git
commit that adds this file is its timestamp. Changes after the first result go at the bottom.

## What is documented, and what we do not know

The FAIR-MAST documentation (level 2, "changes from source" and "release history") states that the saddle-field
channels M02, M08 and M012 have their sign reversed for shots 24782 to 30473 in the deployed level-2 release, recommends
recomputing derived signals from corrected coils, and announces a corrections release (date not given). Level 2 is
regenerated at unchanged paths. Our cache keeps no ingestion attributes, so we do not know for certain which ingestion
our cached arrays come from: it is consistent with the September 2026 release (built 16 October to 1 November 2026),
not proven to be it. The correction is therefore applied only if our own data show the documented reversal (V below),
so that data already corrected are not corrected twice.

How `lm_n1` is built (`src/extras.py`, unchanged by this study): the `magnetics` group's `b_field_tor_probe_saddle_field`
(channels in stored order, taken as M01 to M012 at toroidal angles 15, 45, ..., 345 degrees); each channel averaged per
1 ms bin of the shot's summary grid; its offset (mean of its first 5 ms of finite data) removed; channels with no
finite value in a bin skipped, not set to zero (M09 is empty in the shots checked so far); B = a0 + a1 cos(phi) +
b1 sin(phi) fitted by least squares when at least 6 channels are finite; lm_n1 = sqrt(a1^2 + b1^2); NaN until every
finite channel's offset is known. Downstream, every feature that uses `lm` (`features.uses_signal`: `has_lm`, `lm_mean`,
`lm_std`, `lm_last`, `lm_slope`, `lm_delta`, `lm_rel`, `lm_slope_long`) is computed from `lm_n1`.

## Step 1: read the saddle channels again (GitHub relay; no outcome, label or model passes through)

`scripts/saddle_fetch.py`, for every shot of our window table, from the store as it is today:
- `lm_raw`: lm_n1 exactly as `src/extras.py` computes it, on the shot's summary grid;
- `lm_corr`: the same with channels M02, M08 and M012 multiplied by -1 when 24782 <= shot <= 30473 (identical to
  `lm_raw` otherwise). The sign is changed at the channel input, before binning, offsets and the fit;
- per channel: its stored label, whether it has finite data, and r = the Pearson correlation, over the 1 ms bins where
  lm_raw is defined, between the channel's offset-removed bin means and the median of the other finite channels;
- the stored channel labels and the selected centre-column Mirnov channel (`src/extras.py` takes the first with data);
- the `ingested_at` and `commit_url` attributes of the `magnetics` and `summary` groups, exactly as stored;
- for the frozen sample (below) only: the summary signals as stored today.

## Frozen sample (fixed now, by rule, without outcomes)

From the shots of our window table, sorted by number: B = 20 shots before 24782 at evenly spaced ranks
(`numpy.linspace(0, n - 1, 20)`, rounded), the 5 shots just before 24782; F = 20 shots from 24782 to 30473 at evenly
spaced ranks and the 5 shots from 24782 on. The validation decision uses these 50 shots only. Shots 27000 and 30421
(examples checked before) are added as labelled diagnostics and do not enter the decision. The list is written to
`RESULTS_DIR/saddle_correction/frozen_sample.csv` (shot number, group) when it is first drawn, before the relay, and is
reused for every later comparison, including against a corrections release (a later draw must give the same list).

## Step 2: validation, decided on the frozen sample (V)

For each channel c and period P (B or F): m_c(P) = median over the period's frozen shots of r. A channel "flips" if
|m_c| > 0.2 in both periods and the sign differs.

- **V1, documented reversal present in our data:** M02, M08 and M012 all flip, and at most one other finite channel
  flips. Then the correction is applied (Step 3). As a check of the correction itself, the same rule on `lm_corr`'s
  channels should show no flip of the three; reported.
- **V2, not present (already corrected, or not as documented):** none of the three flips and all three have |m| > 0.2
  in both periods. No correction is applied; the full model stands as trained; we ask the maintainers why.
- **V3, inconclusive:** anything else. No correction is applied automatically. We report what is known and pending,
  and the twin paper's revision presents the full model's locked-mode inputs as affected by an unresolved fault, with
  the no-locked-mode check as the sensitivity.

## Provenance checks (reported, whatever V says)

- **P1, cache against today's store:** per shot, `lm_raw` against the cached `lm_n1` (same NaN pattern, and finite
  values equal within 1e-9 relative): the share of shots that agree, with every disagreeing shot listed by number.
  If all agree, today's stored saddle data give exactly our cached feature.
- **P2, attributes:** the distinct `ingested_at` dates and full `commit_url` values, with counts, per group. We
  report them as stored and do not identify a release from them.
- **P3, frozen sample, summary signals:** today's stored summary signals against our cached ones: changed finite values,
  changed missingness, largest absolute difference and its time, and how many changes fall between the shot's first
  window start and last window end. A whole-shot share is not enough: changes at a landmark are listed.
- **P4, features:** the rebuilt `lm` window features from `lm_raw` must equal the window table's own (float32) for every
  rebuilt shot; any shot that does not is listed. This checks the rebuild before it is used with `lm_corr`.
- **P5, channels:** saddle labels and finite-channel counts by campaign; the Mirnov channel selected, by campaign.

## Step 3 (only under V1): the corrected full model

Build the eight `lm` window features from `lm_corr` with the code that built the window table (`features.shot_windows`,
same stop times); every other column, every window, every split, the hyperparameter search, the eligibility and the
landmarks stay as they are. The original window table, cache and results are not changed: the corrected columns are
kept in `DATA_DIR/saddle_correction/` and swapped in at load time. Model **A_c** = the v3 baseline's inputs on these
tables, trained like the baseline on splits 1 to 20 and the chronological split.

**Primary:** C(10 ms) of A_c minus C(10 ms) of M (machine-side), fixed-lead twin test (`scripts/how_early.py`), common
cohort, over splits 1 to 20. 10 ms because the full model's lead over M was resolved there and nowhere else
(docs/preregistration_no_efit.md), chosen with that knowledge.

**Reading (fixed now):**
- Resolved and positive: with the documented sign correction the full model keeps its 10 ms lead over the machine-side
  model; the revision reports A_c as the full model, A as the uncorrected original, and L as a sensitivity.
- Not resolved: with the correction the 10 ms lead is not shown; the revision says so, and the uncorrected full model's
  lead is not presented as established.

**Secondary:** A_c minus A at every lead (what the correction changes; expected near zero before 24782, where the
inputs are identical); A_c minus L; the same by shot range (lost shot before 24782, from 24782 on); the chronological
split, pooled, conditional on the one fitted model; the placebo; the usual metrics. The same eligible pairs are used
for every model and the counts are shown.

## Not done here, on purpose

No combined model without both EFIT and locked-mode inputs. Without it the wording stays: "the 10 ms advantage remained
positive in separate retraining sensitivities excluding the identified EFIT-derived inputs or the identified locked-mode
inputs", not that it survives removal of both.

## Amendment 1 (2026-11-15, before any channel data were seen)

Written after a second outside review, while the first GitHub read was running and before anything from it was
collected or looked at. That read is cancelled and repeated with the code of this amendment, so that the frozen
sample's channel inputs are kept. The registered V statistic and its rule stay as written above; what follows is added,
and labelled as added.

**A1, order and gates.** Training of A_c starts only if every gate below passes; the code refuses otherwise. If a gate
fails it is documented, the direct swap is not made, and the next step is a focused diagnostics question, not a bigger
analysis.
- G1 (P1, cache reproduced): every shot of the window table from 24782 to 30473 is read, and for every one of them
  today's `lm_raw` reproduces our cached `lm_n1`: same time grid (equal within 1e-9 s), same NaN pattern, finite values
  equal within 1e-9 relative. The tolerance allows only floating-point differences between machines running the same
  code on the same stored data (the fit uses a least-squares solver). Shots outside that range are reported, not gated
  (their features are not swapped), but any disagreement there is stated as a sign of a store change.
- G2 (P4, rebuild reproduced): for every shot from 24782 to 30473, the lm window features rebuilt from `lm_raw` have
  exactly the window table's window ends and NaN pattern, and values equal within 1e-6 relative (float32 storage). The
  largest discrepancies and the affected shots and windows are reported either way.
- G3 (polarity): V gives V1 AND the reference diagnostic A2 is consistent with it. The decision uses channel data only,
  never a model score.
- G4 (fit conditioning): for every finite-channel mask that occurs in shots from 24782 to 30473, the fit's design
  matrix (1, cos phi, sin phi at the assumed angles) has rank 3 and a condition number of at most 10.
- G5 (fixed comparison): A_c uses the same windows, splits, hyperparameter search, eligibility, landmarks and pair rules
  as A; the eligible pair counts per model and split are shown and must be identical.

**A2, a reference diagnostic (added; the V rule is not replaced).** In V, the reference for one suspect coil is the
median of all other channels, which still includes the other two suspect coils. A2 uses a reference fixed now, before
any channel is seen: the channels documented as unaffected (every stored saddle-field channel except M02, M08 and M012),
taking only those with data in the shot. For each channel and frozen shot: r_ref = Pearson correlation, over the same
bins as V (bins where `lm_raw` is defined; at least 20 bins where both are finite; NaN if either is constant), between
the channel and the median of the reference channels other than itself. Per period (B, F) and channel: the median r_ref,
the number of shots with a finite r_ref, and the counts of clear positive (r_ref > 0.2), clear negative (< -0.2) and
indeterminate shots. A2 is consistent with V1 if M02, M08 and M012 each flip (|median| > 0.2 on both sides, sign
differs), no reference channel flips, and each of the three has a finite r_ref in at least 15 shots on each side. V
also requires at least 15 shots with a finite r on each side for each of the three; with fewer it is V3. To compute A2,
the GitHub read keeps the frozen shots' offset-removed 1 ms channel bins (`channel_bins`), so any later diagnostic is
computed from the same frozen inputs.

**A3, geometry.** The mapping of stored channel to toroidal angle (stored order M01..M012 = middle-row coils at 15, 45,
..., 345 degrees) is a name and order match, not a confirmed correspondence; the FAIR-MAST geometry notes warn that data
and geometry channels may not align directly. A wrong relative mapping can change the fitted amplitude (a common
angular shift changes only the phase). Until the mapping is confirmed by the maintainers or a MAST diagnostics
specialist, the feature is called "the first-harmonic saddle-array feature under an assumed geometry", not a validated
locked-mode amplitude, in every paper and note; the mapping table (source label, assumed coil, angle and extent, units,
evidence, availability by campaign) is published with the results. This downgrade is what lets G3 to G5 proceed
without a confirmed geometry.

**A4, reporting not tied to the outcome.** Replacing the reading of Step 3: once G1 to G5 pass, A_c is the corrected
full-model analysis whether or not its 10 ms lead is resolved; the outcome changes the conclusion, not A_c's status. A is
reported as the original representation, A_c as the corrected representation, L as the no-lm sensitivity. The primary
(C(10 ms) of A_c minus M) is kept, with the note that 10 ms was chosen after earlier work had shown it to be the
strongest lead.
- Resolved and positive: the corrected full model shows the 10 ms lead over the machine-side model.
- Not resolved: the corrected full model does not show it; the uncorrected full model's lead is not presented as
  established, and the paper's claim narrows to what the sensitivities show.

**A5, if V3 or a gate fails.** Report what is known and pending, the no-lm sensitivity, and tell the editor. V3 is not
read as evidence that the documented correction is wrong, and V2 is not read as proof that our data were already
corrected.

## Changes after the first result

### Amendment 2 (2026-11-15, after the first check; a change of rule, not a relabelling)

**What happened under the rules as written (kept unchanged):** G1 and G2 FAILED as implemented. Of the 3,426
window-table shots from 24782 to 30473, 2,949 have saddle data and reproduce exactly (G1: 0 differ, largest relative
difference 4.1e-13 over all 10,547 shots with data; G2: all 2,949 rebuilt exactly, largest difference 0). The other
477 were counted as "not read". G3 and G4 passed; G5 was not assessed; no corrected model was trained. The original
outputs are preserved as `check_original.txt`, `features_original.txt` and `gates_original.json`.

**Post-read diagnostic (not a gate):** all 477 have status "no_saddle" in the read and an entirely NaN cached feature.
The rules above did not define shots whose saddle data are absent in both the store and the cache.

**The change.** G1 and G2 are replaced by G1a and G2a, which add one narrowly defined equivalence state, "structural
absence with matching downstream missingness", reported separately from numerical reproduction of finite data. A shot
from 24782 to 30473 is in this state only if all of:
- (a) the read returned a verified absence: the store answered and the magnetics group is absent ("no magnetics
  group") or present without `b_field_tor_probe_saddle_field` ("no saddle field"); a timeout, read exception, odd
  schema or a shot missing from the read never qualifies;
- (b) our cached `lm_n1` exists on the shot's own cached time grid and is entirely NaN;
- (c) every window row of the shot in the window table is reproduced by rebuilding with an all-NaN `lm_n1`: same window
  ends, and all eight lm columns with the same missingness and values (exactly);
- (d) in the window table those eight columns carry no information: `has_lm` is 0 in every window and the other seven
  are NaN in every window. The correction is not applied to these shots (their rows are left untouched by the swap),
  so it introduces no finite value, placeholder, eligibility or flag change.
No source time grid is invented for an absent signal.
- **G1a** passes if every data-present shot reproduces as in G1 and every other shot satisfies (a) and (b).
- **G2a** passes if every data-present shot's rebuild reproduces as in G2 and every other shot satisfies (c) and (d).
Both are reported with shot and window counts, and the list of structurally absent shots is kept.

**G5 before training.** G5 is checked before any model is trained: for splits 1 to 20 and the chronological split, the
eligible twin-pair and landmark manifest (pair, lead) is built from the test windows with a constant score for the
original and the corrected tables, and must be identical; every non-lm column of the corrected table must equal the
original. It is checked again after training from the scored runs.

**Order.** Training starts only if G1a, G2a, G3, G4 and G5 pass. Independent review of this amendment is requested and
its status is reported with the results; at the time of writing it has not been independently reviewed. The geometry
stays assumed: A_c is a polarity-corrected version of the first-harmonic saddle-array feature under an assumed
geometry, not a validated locked-mode diagnostic. A4 (A_c is the corrected representation whatever its result) is
unchanged.

### Amendment 3 (2026-11-18, after the corrected result; an extension, not a new test)

The twin paper's registered tests use splits 21 to 50 and several of its tables use all 50 splits, all with the
original representation A. So that those numbers can be reported with the corrected representation, A_c is trained on
splits 21 to 50 by exactly the frozen procedure of Step 3 (same corrected features, gates, code and settings;
`--train --confirm-splits`). No reading changes: the primary above (splits 1 to 20) stays as computed, and the
50-split numbers are reported as the corrected counterparts of the paper's existing analyses, each labelled with its
representation. Analyses not rerun with A_c stay labelled as original-representation results.

