# Self-audit: do MAST's observation times tell a lost shot from its twin? (protocol)

Written 2026-11-14, before any EFIT time stamp or grid spacing of this study was read. The git commit that adds
this file is its timestamp. Changes after the first result go at the bottom with their date.

## Why

In the public C-Mod / DIII-D / EAST files of Zhu et al., the observation times alone separate disrupted from
non-disrupted shots (`docs/preregistration_sampling_audit.md`, NOTES 2026-11-14). The same mechanism could exist in
the MAST data used by the twin test: EFIT is run on time slices, and if slices were denser before losses, the EFIT
features (each held from its slice time plus EFIT_DELAY_MS on the 1 ms grid) would update more often in lost shots.
The summary signals come on the archive's own time grid, which could also differ. This audit measures that, on the
twin pairs, before anything else is concluded.

## Data

- **EFIT slice times:** the `time` coordinate of each shot's `equilibrium` group in the FAIR-MAST store, read by a
  GitHub job (`scripts/efit_times_fetch.py`, the store is not reachable from the server). A slice is available at
  slice time + EFIT_DELAY_MS (5 ms), the rule of `src/extras.py`.
- **Summary grid:** the time column of each shot's cached shot table (`src.data.load_shot`), the grid all window
  features are built on.
- **Pairs:** the twin pairs of the twin test (`events/programme_twins.csv`), lost shot L and earlier kept twin K,
  with L's loss time t_loss and the flat-top start, as in `scripts/how_early.py`.

## Measures (per shot, at a moment t; identical rules for L and K)

For a lead h, t = t_loss(L) - h in both shots (the same programme time), and the trailing interval is the
20 ms window (t - 20 ms, t].
- **E1, EFIT count (primary measure):** the number of EFIT slices available in the window (available time in it).
- **E2, EFIT age:** t minus the available time of the latest EFIT slice available at t (NaN if none).
- **E3, EFIT gap:** the median gap between EFIT slice times available in (t - 100 ms, t].
- **G1, grid count:** the number of summary-grid samples in the window.
- **G2, grid gap:** the median gap of the summary grid in (t - 100 ms, t].

A pair counts at lead h if t is at or after the flat-top start of L and both shots have at least one sample of the
measure up to t (for E1/E2/E3: at least one available EFIT slice; for G1/G2: grid samples), so a twin that ended is
left out, as in the fixed-lead twin test.

## Statistic

For each measure and lead, the **timing concordance** T: the share of pairs where L has the larger value (more
slices, or, for ages and gaps, the smaller value: "denser" counts as higher), ties one half. 0.5 = no difference.
Leads: 10, 30, 50 and 100 ms; **primary: E1 at 30 ms** (the lead where the twin test's full model still leads the
machine-side model, `docs/preregistration_how_early.md`). Beside T: the mean paired difference L - K.

Range: groups of pairs sharing a shot resampled together (2,000 resamples, seed 42), as the twin study's pooled
range. No model is fitted here, so there is no range over training sets.

## Reading (fixed now)

- **T(E1, 30 ms) within 0.47 to 0.53 and its range including 0.5:** no EFIT timing difference between lost shots and
  their twins at the moment compared; the twin test's EFIT inputs do not carry a timing shortcut there.
- **T resolved outside 0.47 to 0.53:** EFIT timing differs between lost shots and twins. Then a second protocol,
  written before it runs, tests whether EFIT timing changes prediction (machine-side model alone, plus EFIT timing
  features, plus EFIT values, plus both, on the v3 splits), and the twin paper's journal is told if a central claim
  changes.
- **Otherwise (resolved but within 0.47 to 0.53, or unresolved and wide):** reported as found; no reading fixed.

The same three readings apply to G1 at 30 ms (summary grid), reported as the second key measure.

## Secondary (reported whatever they show)

- All measures at all four leads, T and mean paired difference.
- Without pairs: for every lost and every kept shot, E1 and G1 in the 20 ms window ending 40 ms before the shot's
  last grid sample, and the shot-level AUC lost against kept (the public-files statistic, applied to MAST).
- The distribution of EFIT gaps in the whole flat-top, lost against kept (median and 90th percentile).
- The share of shots with no equilibrium group or no slices, lost and kept.

## Not claimed

That the twin test's predictor used any timing difference: that needs the model comparison above. This audit
asks only whether the observation times differ.

## Changes after the first result

(none yet)
