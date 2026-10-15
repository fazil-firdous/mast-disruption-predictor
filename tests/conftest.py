"""Shared test setup: no test may touch the network."""
import pytest

from src import data


@pytest.fixture(autouse=True)
def no_network_checks(monkeypatch, request):
    if "real_host_check" not in request.keywords:
        monkeypatch.setattr(data, "host_reachable", lambda url, timeout=None: True)
