# Carrying this project on

Written 2026-11-11 for whoever continues the work: the owner alone, a collaborator, or an assistant that has never
seen the project. Everything needed is in this repository; nothing depends on anyone's memory.

## Read these first, in this order

1. `README.md`: what the project is and what it found.
2. `docs/GUIDE.md`: one shot's journey through the code, in simple words.
3. `CODE_MAP.md`: every file and what it does.
4. `NOTES.md`: every result and decision, dated, newest at the bottom. This is the project's memory.
5. `docs/preregistration_v3.md`: what was promised before the v3 runs, and every later addition with its date.

An assistant starting cold should be given items 3 to 5 and asked to read them before changing anything.

## Where things live

| What | Where |
|---|---|
| Code | this repository; on the server in `/root/mastu` |
| Data, results, models, logs (private) | the folder named by `MAST_HOME` (on the server `/root/mast-private`) |
| Results of every experiment | `MAST_HOME/results/` (`E0`..`E15`, `E10/seed_*`, `E10/comparisons/`, `v3/`, `v3_explore/`, `events/`, `twin_differences/`, `release/`) |
| Copies of results before each rerun | `MAST_HOME/results_history/` |
| Secrets (Telegram, GitHub token) | `~/mast-private/.env`, never in the repository |
| Paper sources and drafts | not in the repository; the owner keeps them |

## The server

- It runs as a systemd service: `systemctl status mast-research`, log with `journalctl -u mast-research -n 50 --no-pager`.
- Every ten minutes it checks for new commits and pulls them. A commit that changes a science file reruns what depends on it; a commit
  that changes only documents or the analysis scripts reruns nothing (see 'Stamps' below).
- Telegram: `/status` for the picture, `/report` for every result in words, `/help` for the rest.
- Setup from nothing: `docs/SERVER.md`.

## Commands worth knowing

```
cd /root/mastu && git pull                                   # bring the server's code up to date
.venv/bin/python -m pytest -q                                # every test (about 5 minutes)
.venv/bin/python scripts/paper_numbers.py                    # paper 1's tables from the saved runs
.venv/bin/python scripts/v3_runs.py --summary                # the registered v3 tests (H1 to H3)
.venv/bin/python scripts/v3_explore.py --summary             # the ladder, the breakdowns, the checks
.venv/bin/python scripts/v3_explore.py --temporal            # the chronological split
.venv/bin/python scripts/paper2_numbers.py                   # paper 2's counts and its core table
.venv/bin/python scripts/twin_differences.py                 # lost shot against two good repeats, no model
.venv/bin/python scripts/make_release.py                     # rebuild the public data release (uploads nothing)
.venv/bin/python scripts/plot_twin_pair.py --list --after 24000   # pick a twin pair, then give its number
.venv/bin/python scripts/how_early.py                        # the twin test at fixed times before the loss (Part A)
.venv/bin/python scripts/how_early.py --part-b               # the 100 ms training horizon against the 30 ms one
```

None of these retrains anything except `v3_runs.py --recipe ... --seed ...`, `v3_explore.py --train` and
`--temporal`, and `how_early.py --train-b`; those skip runs that are already finished.

## Stamps: why a small change can cost days

Every saved result carries a fingerprint of the science code, the science settings and the data that made it
(`src/resume.py`). Change a science file (anything in `src/` not listed in `NOT_SCIENCE`, or
`scripts/build_dataset.py`) and the server retrains everything that depends on it.

- Analysis scripts in `scripts/` are outside every stamp: change them freely.
- `scripts/repeat_runs.py` and `scripts/v3_runs.py` have their own stamps. Editing either file reruns its own
  runs (150 for v3). Put new analyses in a new script instead, as `scripts/v3_explore.py` does.
- Before a science change, the server copies the old results to `results_history`. Check the disk first.

## The rules that made the results trustworthy

1. **Write the test down before running it.** `docs/preregistration_v3.md` has the pattern: hypothesis, metric,
   which splits develop and which confirm, how it will be read. Commit it before the run.
2. **Splits 1-20 develop, 21-50 confirm.** Nothing is chosen on the confirmation splits.
3. **Report what failed.** Two of the four registered v3 ideas showed no gain; they are in the notes and the paper.
4. **No future data.** A window uses samples up to its end only; tests enforce it. Test shots never enter fitting
   or tuning.
5. **Every claim gets a range**, corrected for overlapping splits (`src/stats.py`), and several tests at once get
   Holm's correction.
6. **Say what a number is.** A window metric, a shot-level catch rate and the twin score C answer different
   questions.
7. **After a result, new analyses are exploratory** and are labelled so.
8. **Each change lands with tests, a NOTES entry and a CODE_MAP line.**

## Where the science stands (2026-11-11)

- **Paper 1** (label audit): public, doi:10.5281/zenodo.24081241; submitted to arXiv.
- **Paper 2** (twin test): drafted; every analysis is finished and listed in the pre-registration file. Its data
  release is built by `scripts/make_release.py`.
- **Open before paper 2 is public:** a read by a physicist who knows MAST; the full texts of the nearest papers
  (`docs/literature_review.md`, section 6).
- **What the twin test showed, in one line:** the full model ranks a lost shot above its same-programme twin more
  often than a machine-side model does; part of every model's score also appears between two good repeats.
- **How early?** (2026-11-12, `docs/preregistration_how_early.md`): Part A is in the notes; Part B (a 100 ms
  training horizon) is built and waits for its runs.

## What could come next

In the order they seem worth doing.

1. **How early?** In progress; see the line above and the notes.
2. **A second machine.** The label audit and the twin test need only a programmed current, a solenoid current and
   operators' notes. Any tokamak with those could be checked. Needs data access.
3. **Recover or progress.** Operators timed 363 internal reconnection events and 159 locked modes; about half the
   shots survived them. What separates the two outcomes is a question for a collaboration with someone who knows
   MAST (`scripts/event_outcomes.py` has the counts).
4. **A retraining-aware range.** Resample groups of shots, retrain inside each resample, and recompute the twin
   test, to get one range that covers both the training set and shared twins. Costly.

Dropped: a matched study of which operating conditions go with losing a shot. Operators choose heating and density
in response to the session, so the associations could not be read without the experiment plans.

## Working with an assistant

- Give it this file, `CODE_MAP.md` and `NOTES.md` first.
- Ask for one change at a time, with tests, and read the diff before merging.
- Ask it to write the plan and the expected reading into the notes before a run, and the result after.
- Two assistants agreeing is not a check. A person who knows the machine, or the data itself, is.
