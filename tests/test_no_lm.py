"""Tests for scripts/no_lm.py: no lm-derived input remains and the rest is kept. No network."""
from scripts import no_lm, v3_explore
from src import features


def test_columns_drop_every_lm_input():
    cols, base = no_lm.columns(), v3_explore.recipe_columns("baseline")
    assert set(cols) < set(base) and not any(features.uses_signal(c, "lm") for c in cols)
    assert any(c.startswith("lm_") for c in set(base) - set(cols))
    assert any(c.startswith("q95_") for c in cols) and any(c.startswith("mirnov_") for c in cols)
