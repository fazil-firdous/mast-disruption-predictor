"""Decisions and texts for the always-on research server (scripts/research_server.py), kept here so they can be tested."""
import json
import os
import time

from src import config

STATE_PATH = os.path.join(config.LOGS_DIR, "server_state.json")
HELP = """MAST research server
/status - the picture: training, what it can do, what is needed
/report - every experiment in words, plus all result files
/details - text status (commit, network, disk)
/log - last lines of the running job
/predict 11860 - run the best model on one shot
/gold - label shots by hand (/d disruption, /n normal, /u unsure); /goldreport
/run - full run now; /fetch - fetch data again through GitHub, then run
/sources, /papers - check data sources / arXiv now
/diagnose - check the network
/update - pull new code now
/pause, /resume, /stop
/help - this list"""


def new_state():
    """State of a server that has never run."""
    return {"offset": 0, "last_ok_commit": None, "failed_commit": None, "failed_at": 0.0, "paused": False,
            "force_pipeline": False, "seeds_done": [], "last_heartbeat": 0.0, "last_update_check": 0.0,
            "network_down_since": None, "relay_manifest": None, "force_fetch": False, "last_probe": None,
            "last_probe_at": 0.0, "last_literature_at": 0.0, "gold_current": None, "seeds_stamp": None,
            "job_pid": None, "started_at": time.time()}


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


def job_command(pid, proc_dir="/proc"):
    """The command line of a running process ('' if it is gone), read from Linux's /proc."""
    try:
        with open(os.path.join(proc_dir, str(pid), "cmdline"), "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def is_our_job(command):
    """True if a command line is one of the jobs this server starts (so a reused pid is never killed by mistake)."""
    return any(name in command for name in ("scripts/run_all.py", "scripts/run_experiment.py", "pytest"))


def seeds_still_valid(state, stamp):
    """True if the seed-sweep results were made with the same science code and data as the stamp."""
    return state["seeds_stamp"] == stamp


def access_mode(data_ok, table_ok, relay_ok):
    """How the next full run gets its data: 'direct' (STFC and shot table answer), 'direct-report' (STFC answers,
    mastapp.site is down: use the 400 report shots), 'relay' (STFC blocks this server, a GitHub token is set),
    or None (wait and try again)."""
    if data_ok and table_ok:
        return "direct"
    if data_ok:
        return "direct-report"
    if relay_ok:
        return "relay"
    return None


def relay_needed(state, scale):
    """True if the relay has not yet brought data for this scale, or a re-fetch was asked for."""
    manifest = state["relay_manifest"]
    return state["force_fetch"] or manifest is None or manifest.get("scale") != scale


def pipeline_args(scale, mode, manifest=None):
    """Arguments for scripts/run_all.py: offline after a relay fetch, and the report shots when the relay or this
    server had no shot table."""
    args = ["--scale", scale]
    if mode == "relay":
        args.append("--offline")
    no_table = mode == "direct-report" or (mode == "relay" and manifest and manifest.get("table_source") == "report")
    if no_table:
        args += ["--from-report", config.REPORT_CSV]
    return args


def due(last_time, every_hours, now):
    """True if more than every_hours have passed since last_time (seconds)."""
    return now - last_time > every_hours * 3600


def heartbeat_due(state, now):
    """True once every HEARTBEAT_HOURS."""
    return due(state["last_heartbeat"], config.HEARTBEAT_HOURS, now)


def status_text(state, head, job, network):
    """The /status reply."""
    up_hours = (time.time() - state["started_at"]) / 3600
    disk = os.statvfs(os.path.abspath(config.HOME_DIR))
    free_gb = disk.f_bavail * disk.f_frsize / 1e9
    return (f"commit {head}, up {up_hours:.1f} h, {'PAUSED' if state['paused'] else 'active'}\n"
            f"job: {job or 'idle'}\nlast good full run: {state['last_ok_commit'] or 'none yet'}\n"
            f"seed sweep: {len(state['seeds_done'])}/{config.SEED_SWEEP_MAX}\n"
            f"network: {network}\nfree disk: {free_gb:.1f} GB\nworkspace: {os.path.abspath(config.HOME_DIR)}")
