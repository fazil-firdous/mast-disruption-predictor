"""Tests for scripts/no_efit.py: no EFIT-derived input remains, everything else is kept, and one run end to end on the
tiny fake project. No network."""
import numpy as np

from scripts import no_efit, v3_explore
from src import config, experiment, features
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_columns_drop_every_efit_input():
    cols = no_efit.columns()
    base = v3_explore.recipe_columns("baseline")
    assert set(cols) < set(base)
    assert not any(features.uses_signal(c, s) for c in cols for s in no_efit.EFIT_SIGNALS)
    dropped = set(base) - set(cols)
    assert any(c.startswith("q95_") for c in dropped) and any(c.startswith("fgw_") for c in dropped)
    assert any(c.startswith("lm_") for c in cols) and any(c.startswith("ne_") for c in cols)


def test_one_run_end_to_end(tiny_project, monkeypatch):
    monkeypatch.setattr(config, "SWEEP_MODEL", "logreg")
    windows, shots = experiment.load_tables()
    for c in v3_explore.recipe_columns("baseline"):
        if c not in windows:
            windows[c] = np.nan
    metrics = no_efit.run(2, (windows, shots))
    out = tiny_project / "results" / "no_efit" / "no_efit" / "seed_2"
    assert (out / "metrics.json").exists() and not any(c.startswith("q95_") for c in metrics["columns"])
    assert not (tiny_project / "results" / "v3").exists()
