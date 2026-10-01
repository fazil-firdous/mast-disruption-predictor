"""Trivial test so pytest has something to run."""
from src import config


def test_config_seed():
    assert config.RANDOM_SEED == 42
