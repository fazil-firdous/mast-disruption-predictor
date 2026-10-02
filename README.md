# MAST Plasma Disruption Predictor

A machine-learning project that watches a MAST tokamak plasma pulse as it unfolds and tries to raise an alarm before a disruption, using only information available up to that moment. Work is staged by phase (see `docs/SPEC.md` and `the project rules`). **Status: Phases 0 to 3 done (scaffold, data layer, labels, causal window features). No model has been trained yet, so there are no model results.**

## How to run
Everything is meant to run from `notebooks/colab_pipeline.ipynb` (not yet run in Colab). From a terminal:

```
pip install -r requirements.txt
python scripts/inspect_shots.py --n 20     # look at 20 shots across campaigns
python scripts/download.py --n 50          # cache 50 random shots under data/raw
python scripts/label_report.py             # Phase 2 label comparison
python scripts/feature_check.py            # Phase 3 features + causality check on real shots
pytest
```

Note: the shot stores are Zarr v3, so `zarr>=3` is required (see `NOTES.md`). `download.py` checks the connection first and stops with a clear message if the servers cannot be reached.

## Results so far
Only the label comparison exists (`results/label_report/report.md`, from code that ran on 400 real shots: 175 note-disrupted, 225 random note-clean). The current-quench detector confirms 97.7% of note-disrupted shots, but it also flags 74.2% of note-clean shots, because most MAST shots end with a fast current quench whether or not anyone called it a disruption. Decision: `LABEL_SOURCE = "both"`, so a shot counts as disrupted only when the note and the detector agree (171 of the 400). The detector supplies the disruption time. Model results: TBD - not yet run.

## Limitations
Labels are algorithmic and imperfect. Only 4 global signals are used, with no magnetics or profiles. MAST (2000 to 2013), not MAST-U. Offline evaluation, not real-time control. Results come from a subset of shots. The data layer is checked against the real servers on every push (GitHub Actions), but the Colab notebook itself has not been run yet.

## Credits
Data: FAIR-MAST, UKAEA, licence CC BY-SA 4.0 (https://mastapp.site).
