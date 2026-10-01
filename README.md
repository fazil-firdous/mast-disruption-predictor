# MAST Plasma Disruption Predictor

A machine-learning project that watches a MAST tokamak plasma pulse as it unfolds and tries to raise an alarm before a disruption, using only information available up to that moment. Work is staged by phase (see `docs/SPEC.md` and `the project rules`). **Status: Phases 0 to 2 done (scaffold, data layer, labels). No model has been trained yet, so there are no model results.**

## How to run
Everything is meant to run from `notebooks/colab_pipeline.ipynb` (not yet run in Colab). From a terminal:

```
pip install -r requirements.txt
python scripts/inspect_shots.py --n 20     # look at 20 shots across campaigns
python scripts/download.py --n 50          # cache 50 random shots under data/raw
python scripts/label_report.py             # Phase 2 label comparison
pytest
```

Note: the shot stores are Zarr v3, so `zarr>=3` is required (see `NOTES.md`).

## Results so far
Only the label comparison exists (`results/label_report/report.md`, produced by code that ran on 550 real shots). On a uniform random sample of 400 shots, operator notes mark 5.5% as disrupted and the signal-based current-quench detector marks 76.8%. Most MAST shots end with a current spike and fast fall whether or not a note mentions a disruption, so the label definition is an open decision for the project owner. Model results: TBD - not yet run.

## Limitations
Labels are algorithmic and imperfect. Only 4 global signals are used, with no magnetics or profiles. MAST (2000 to 2013), not MAST-U. Offline evaluation, not real-time control. Results come from a subset of shots. The data layer has been tested on real files read from disk but not yet over HTTPS from the development sandbox.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0 (https://mastapp.site).
