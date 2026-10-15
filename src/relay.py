"""Get shot data through GitHub when this server cannot reach STFC (its IP range is blocked there).

A GitHub Actions job (.github/workflows/fetch_data.yml, scripts/relay_fetch.py) downloads the shots and attaches
them as a short-lived artifact; this module starts that job, waits for it and unpacks the artifact into DATA_DIR.
The GitHub token is read from the environment (private .env) at call time and never stored in config or results.
"""
import calendar
import json
import os
import time
import zipfile

import requests

from src import config

API = "https://api.github.com/repos/{repo}"
ARTIFACT_NAME = "mast-data"


def configured():
    """True if a GitHub token is available, so the relay can be used."""
    return bool(os.environ.get("GITHUB_TOKEN"))


def _headers():
    return {"Authorization": f"Bearer {os.environ.get('GITHUB_TOKEN')}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"}


def _api(path):
    return API.format(repo=config.GITHUB_REPO) + path


def trigger(scale, task="fetch"):
    """Start the fetch_data job on GitHub ('fetch' shots for this scale, or 'probe' the sources).
    Returns the time it was requested (seconds, UTC)."""
    url = _api(f"/actions/workflows/{config.RELAY_WORKFLOW}/dispatches")
    response = requests.post(url, headers=_headers(), timeout=30,
                             json={"ref": config.RELAY_BRANCH, "inputs": {"scale": scale, "task": task}})
    response.raise_for_status()
    return time.time()


def parse_time(text):
    """GitHub's '2026-10-15T14:30:00Z' as seconds since 1970 (UTC)."""
    return calendar.timegm(time.strptime(text, "%Y-%m-%dT%H:%M:%SZ"))


def pick_run(runs, requested_at):
    """The newest manually started run created after requested_at (with a minute of clock slack), or None.

    GitHub does not return a run id when a job is started, so the run is found by its start time.
    """
    newest = None
    for run in runs:
        if run.get("event") != "workflow_dispatch" or parse_time(run["created_at"]) < requested_at - 60:
            continue
        if newest is None or run["created_at"] > newest["created_at"]:
            newest = run
    return newest


def find_run(requested_at):
    """The run started by trigger(), or None if GitHub has not listed it yet."""
    url = _api(f"/actions/workflows/{config.RELAY_WORKFLOW}/runs")
    response = requests.get(url, headers=_headers(), params={"event": "workflow_dispatch", "per_page": 10}, timeout=30)
    response.raise_for_status()
    return pick_run(response.json().get("workflow_runs", []), requested_at)


def run_status(run_id):
    """(status, conclusion) of a run, e.g. ('in_progress', None) or ('completed', 'success')."""
    response = requests.get(_api(f"/actions/runs/{run_id}"), headers=_headers(), timeout=30)
    response.raise_for_status()
    run = response.json()
    return run["status"], run["conclusion"]


def unpack(zip_path, data_dir):
    """Extract the artifact into data_dir and return its manifest. Refuses any entry that would land outside
    data_dir, and keeps this server's own skipped.csv instead of the relay machine's."""
    root = os.path.realpath(data_dir)
    os.makedirs(root, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            target = os.path.realpath(os.path.join(root, name))
            if not target.startswith(root + os.sep):
                raise ValueError(f"unsafe path in artifact: {name}")
        for name in archive.namelist():
            if os.path.basename(name) != "skipped.csv":
                archive.extract(name, root)
    with open(os.path.join(root, "manifest.json")) as f:
        return json.load(f)


def download(run_id, data_dir=None):
    """Download the run's artifact and unpack it into data_dir (default DATA_DIR). Returns the manifest."""
    data_dir = data_dir or config.DATA_DIR
    response = requests.get(_api(f"/actions/runs/{run_id}/artifacts"), headers=_headers(), timeout=30)
    response.raise_for_status()
    artifacts = [a for a in response.json().get("artifacts", []) if a["name"] == ARTIFACT_NAME]
    if not artifacts:
        raise RuntimeError(f"run {run_id} has no '{ARTIFACT_NAME}' artifact")
    zip_path = os.path.join(config.HOME_DIR, "relay_download.zip")
    with requests.get(artifacts[0]["archive_download_url"], headers=_headers(), stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    try:
        return unpack(zip_path, data_dir)
    finally:
        os.remove(zip_path)
