# How to work on this project

`docs/SPEC.md` says *what* the pipeline does; this file says *how* to change it. Read both before changing code.

## Principles
1. **Readability beats cleverness.** Prefer a plain loop over a dense one-liner if the loop is easier to follow. No metaprogramming, no decorators beyond `@dataclass`, no deep class hierarchies. A function does one thing and fits on one screen (aim for under ~40 lines).
2. **Every decision is explainable.** A chosen parameter (window length, threshold, horizon) goes in `src/config.py` with a one-line comment saying why. A judgement call is logged in `NOTES.md`.
3. **Research-grade.** Methods and results should stand up in a fusion / ML venue. The long-term plan is `docs/ROADMAP.md`.

## Hard rules
- **No data leakage.** A prediction at time `t` uses only samples with `time <= t`. No whole-shot statistics, no future samples, no normalisation fitted on validation or test data. The causality tests must pass.
- **Split by shot, never by window.** All windows of a shot land in the same split.
- **Never fabricate results.** Every number in the README, notebook or result files comes from code that actually ran. If something could not be run, say so and leave `TBD — not yet run`.
- **Never invent data details.** A variable, group, unit or sign convention not verified in `docs/SPEC.md` Section 3 is inspected on real data first and recorded in `NOTES.md`.
- **Never commit data.** `data/`, caches, models and large artefacts are git-ignored. Only code, small configs, docs and small result files (JSON, PNG under ~500 KB) are committed. Trained-model results stay in the private workspace (`MAST_HOME`).
- **Seed everything** with `config.RANDOM_SEED = 42`.

## Comment style
Few, dry comments that explain the *why*, not the *what*:
```python
# zarr<3 cannot read the re-ingested v3 stores, so pin >=3
# abs() because MAST's ip sign depends on field direction
```
No comments on imports, no banner comments, no emoji. Docstrings: one or two plain sentences on public functions, saying what goes in and what comes out.

## Workflow
- One change at a time, as a pull request with tests, a `NOTES.md` entry when something was decided or discovered, and a `CODE_MAP.md` update.
- Before pushing: `pytest -W error` passes.
- Science code is fingerprinted (`src/resume.py`): a change to it reruns every experiment once on the research server. Keep server, message and docs changes out of science files so they never retrain anything.
- If the data surprises you (odd units, sign flips, missing groups, odd time bases), don't patch it silently: write it in `NOTES.md` with the shot ids.
- Keep dependencies small: `numpy`, `pandas`, `pyarrow`, `xarray`, `zarr>=3`, `fsspec`, `aiohttp`, `requests`, `scikit-learn`, `matplotlib`, `pytest`, `tqdm`. Discuss before adding another.

## Environment
- Python 3.10+.
- Data is streamed over HTTPS from the public FAIR-MAST servers (`s3.echo.stfc.ac.uk`). Every workflow stays runnable from `notebooks/colab_pipeline.ipynb`.

## Docs to keep current
- `README.md`: what, why, how to run, findings, limitations, credits.
- `CODE_MAP.md`: a plain-English tour of every file and public function, and the order to read them in.
- `docs/GUIDE.md`: the project as a story, for newcomers.
- `NOTES.md`: dated log of data surprises, decisions and open questions.
