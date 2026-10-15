"""Decisions and texts for the always-on research server (scripts/research_server.py), kept here so they can be tested."""
import json
import os
import time

from src import config

STATE_PATH = os.path.join(config.LOGS_DIR, "server_state.json")
HELP = """MAST research server commands:
/status - what is running, commit, network, seeds done
/results - headline test numbers and the seed-sweep summary
/log - last lines of the current job's log
/predict 11860 - run the best saved model on one shot and send the plot
/diagnose - check the network path to the data servers
/run - start a full pipeline run now
/update - pull new code now (the server restarts itself)
/pause, /resume - stop or allow new jobs
/stop - kill the running job
/help - this list"""


def new_state():
    """State of a server that has never run."""
    return {"offset": 0, "last_ok_commit": None, "failed_commit": None, "failed_at": 0.0, "paused": False,
            "force_pipeline": False, "seeds_done": [], "last_heartbeat": 0.0, "last_update_check": 0.0,
            "network_down_since": None, "started_at": time.time()}


def read_state(path=STATE_PATH):
    """Saved state, with any missing keys filled from new_state()."""
    state = new_state()
    if os.path.exists(path):
        with open(path) as f:
            state.update(json.load(f))
    return state


def write_state(state, path=STATE_PATH):
    """Save state atomically (write a temp file, then rename), so a crash never leaves half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(state, f, indent=2)
    os.replace(path + ".tmp", path)


def parse_command(text):
    """'/predict@MyBot 11860' -> ('predict', '11860'). Anything that is not a /command gives (None, '')."""
    if not text or not text.startswith("/"):
        return None, ""
    parts = text.strip().split(maxsplit=1)
    name = parts[0][1:].split("@")[0].lower()
    return name, parts[1].strip() if len(parts) > 1 else ""


def next_job(state, head, now, processed_ready):
    """What to do next: ('pipeline', None), ('seed', n) or (None, None).

    A full run is due for every new commit (or when forced); a failed commit is retried after PIPELINE_RETRY_HOURS.
    When the current commit has run, idle time goes to the E10 seed sweep.
    """
    if state["paused"]:
        return None, None
    retry_due = now - state["failed_at"] > config.PIPELINE_RETRY_HOURS * 3600
    new_code = state["last_ok_commit"] != head and (state["failed_commit"] != head or retry_due)
    if state["force_pipeline"] or new_code:
        return "pipeline", None
    if state["last_ok_commit"] == head and processed_ready:
        done = set(state["seeds_done"])
        for seed in range(1, config.SEED_SWEEP_MAX + 1):
            if seed not in done:
                return "seed", seed
    return None, None


def heartbeat_due(state, now):
    """True once every HEARTBEAT_HOURS."""
    return now - state["last_heartbeat"] > config.HEARTBEAT_HOURS * 3600


def status_text(state, head, job, network):
    """The /status reply."""
    up_hours = (time.time() - state["started_at"]) / 3600
    disk = os.statvfs(os.path.abspath(config.HOME_DIR))
    free_gb = disk.f_bavail * disk.f_frsize / 1e9
    return (f"commit {head}, up {up_hours:.1f} h, {'PAUSED' if state['paused'] else 'active'}\n"
            f"job: {job or 'idle'}\nlast good full run: {state['last_ok_commit'] or 'none yet'}\n"
            f"seed sweep: {len(state['seeds_done'])}/{config.SEED_SWEEP_MAX}\n"
            f"network: {network}\nfree disk: {free_gb:.1f} GB\nworkspace: {os.path.abspath(config.HOME_DIR)}")
