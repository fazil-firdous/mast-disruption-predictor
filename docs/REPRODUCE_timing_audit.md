# Reproducing the observation-schedule audit of the public C-Mod / DIII-D / EAST files

Everything needed to re-run one claim of the audit from scratch, and the numbers it should print. The analyses, their
protocols (committed before each was computed) and every later change are in `docs/preregistration_*.md` and
`NOTES.md` (entries from 2026-11-14 to 2026-11-18, including the corrections and withdrawals).

## Data (not redistributed here)

Harvard Dataverse doi:10.7910/DVN/XIOHW1, version 2 (Zhu et al., Nucl. Fusion 61, 026007, 2021):

| File | MD5 | SHA-256 |
|---|---|---|
| `20ja072_fig4_data.zip` (C_Mod_for_nn.mat, d3d_for_nn.mat, EAST_for_nn.mat) | b605a694bc3ff5038dc3b27c5e91b23b | 7e6330c2a21c11f9d693a7b6c6422628e45f9abd4b8432d38f7479dcdacb1218 |
| `20ja072_fig6_data.zip` (EAST_for_nn.mat, EAST_data_partition.mat) | cc23957f4fd596d90a0a741f72cc4f04 | 7e20beaf2420d7cab62c838e0d100920327259de45fb0fef0ee555d1e07bd708 |

Put both zips in one folder `DATA/` and unzip them there, so that it holds the two zips and the folders
`20ja072_fig4_data/` and `20ja072_fig6_data/` (the accounting script checks the zips' MD5).

## Environment

The reported numbers were computed with Python 3.12.3, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1
(`requirements.txt` lists the packages; pin these versions to reproduce exactly). No GPU, no network after the
download. Each script below runs in minutes on a laptop.

## Splits

- EAST: the release's own lists in `EAST_data_partition.mat` (training 8,530, validation 1,137, test 1,706 shots).
- DIII-D and C-Mod: `scripts/timing_model.our_split`, by shot, stratified by class, seed 42, 60/20/20. The same code
  gives the same split on any machine.

## Commands and expected output

| Claim | Command | Expected |
|---|---|---|
| Timestamp-only separation (S1, C1, C2, C5) | `python scripts/sampling_audit.py DATA/20ja072_fig4_data` and `python scripts/sampling_audit_2.py DATA/20ja072_fig4_data` | S1: DIII-D 1.000, EAST 0.999, C-Mod 0.526; C1: DIII-D 0.998, EAST 0.999 |
| Timing-only predictor on the EAST test list | `python scripts/timing_model.py DATA` | EAST official test AUC 0.998 (0.994 to 1.000) |
| Residual observation age on the sparse grid | `python scripts/harmonisation.py DATA` | EAST residual timing AUC 0.938 (0.919 to 0.958) |
| The same ages are unaffected by the 1e-9 vs 1e-6 s tolerance in EAST and DIII-D | `python scripts/grid_tolerance_check.py DATA` | EAST and DIII-D: 0 grid points change; mean-age AUC 0.928 and 0.981 at both tolerances |
| C-Mod observation-policy perturbation (R0 primary) | `python scripts/schedule_injection.py DATA/20ja072_fig4_data/C_Mod_for_nn.mat` | AUC 0.901 under U, 0.971 under I; increase +0.069 (+0.046 to +0.093) |
| Its accounting and the R2 identity diagnostic | `python scripts/schedule_injection_check.py DATA/20ja072_fig4_data/C_Mod_for_nn.mat` | U nested in I in 4,469 of 4,469 shots; R2 identical in 142 of 935 disrupted shots, every other one explained by near misses or a moved origin |
| Accounting of the files | `python scripts/sampling_audit_check.py DATA` | counts, MD5, EAST partition flow |

Results are written under `RESULTS_DIR/sampling_audit/` (by default `./results/sampling_audit/`).

## What is withdrawn, and why

C-Mod's grid-based numbers (5 ms forward-fill and sparse-grid representations, their residual timing, the C1 and
change-pattern cells, the timing-only 0.613, and the perturbation's R1 and R2 secondaries) were computed with a time
tolerance (1e-9 s) below C-Mod's stored time precision (offsets up to 0.124 us). They are withdrawn, not replaced;
see `NOTES.md`, 2026-11-18, "Dependency ledger of the 1e-9 s time boundaries". The tests in
`tests/test_grid_tolerance_check.py` document the boundary behaviour of the grid helper.

## Checking one number by hand

The smallest independent check: take one EAST disrupted shot and one non-disrupted shot from `EAST_for_nn.mat`, cut
each record 40 ms before its last slice, and count the slices in the 100 ms before the cut. A disrupted record holds
about 10, a non-disrupted one about 1 (medians over all records of at least 1 s: 10 and 1).
