"""Shared test setup: no test may touch the network, and logs go to a temporary folder."""
import pytest

from src import config, data


@pytest.fixture(autouse=True)
def no_network_checks(monkeypatch, request):
    if "real_host_check" not in request.keywords:
        monkeypatch.setattr(data, "host_reachable", lambda url, timeout=None: True)


@pytest.fixture(autouse=True)
def temporary_logs(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOGS_DIR", str(tmp_path / "logs"))
