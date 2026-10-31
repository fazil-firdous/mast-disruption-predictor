"""Decisions and texts for the always-on research server (scripts/research_server.py), kept here so they can be tested."""
import json
import os
import time

from src import config, extras

STATE_PATH = os.path.join(config.LOGS_DIR, "server_state.json")
HELP = """MAST research server
/status - the picture: training, what it can do, what is needed
/report - every experiment in words, the comparisons, all result files
/details - text status (commit, network, disk)
/log - last lines of the running job
/predict 11860 - run the best model on one shot
/gold - optional hand labelling for an expert (/d, /n, /u); /goldreport
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
            "comparisons_done": [], "comparisons_stamp": None, "job_pids": [], "idle_peak_mb": {},
            "started_at": time.time()}


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
    """What to do next: 'pipeline', 'idle' (repeat runs, see idle_jobs) or None.

    A full run is due for every new commit (or when forced); a failed commit is retried after PIPELINE_RETRY_HOURS.
    When the current commit has run, idle time goes to the repeat runs.
    """
    if state["paused"]:
        return None
    retry_due = now - state["failed_at"] > config.PIPELINE_RETRY_HOURS * 3600
    new_code = state["last_ok_commit"] != head and (state["failed_commit"] != head or retry_due)
    if state["force_pipeline"] or new_code:
        return "pipeline"
    if state["last_ok_commit"] == head and processed_ready and idle_jobs(state, 1):
        return "idle"
    return None


def idle_jobs(state, limit):
    """Up to limit idle jobs still to do, as (kind, seed): the E10 seeds the comparisons pair with, then the repeat-run
    comparisons (config.COMPARISONS, COMPARISON_SEEDS splits each), then the rest of the E10 seeds, so the
    comparisons' answers come days sooner. All of one kind, so jobs run side by side take about as long."""
    paired = range(1, min(config.COMPARISON_SEEDS, config.SEED_SWEEP_MAX) + 1)
    rest = range(len(paired) + 1, config.SEED_SWEEP_MAX + 1)
    queue = ([("E10", seed) for seed in paired] + [(name, seed) for seed in paired for name in config.COMPARISONS]
             + [("E10", seed) for seed in rest])
    done = {("E10", seed) for seed in state["seeds_done"]} | {tuple(job) for job in state["comparisons_done"]}
    pending = [job for job in queue if job not in done]
    return [job for job in pending if job[0] == pending[0][0]][:limit]


def lanes(cores, peak_mb, free_mb):
    """How many jobs to run side by side: one per CPU core, as long as each fits in free memory with MEMORY_SPARE
    to spare. One at a time until such a job's memory use has been measured (peak_mb 0)."""
    if not peak_mb or not free_mb:
        return 1
    return max(1, min(cores, int(free_mb / (config.MEMORY_SPARE * peak_mb))))


def cpu_cores():
    """CPU cores this process may use."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not Linux
        return os.cpu_count() or 1


def free_memory_mb(meminfo="/proc/meminfo"):
    """Memory available to new programs (Linux MemAvailable) in MB, or None if unknown."""
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def reap(proc):
    """(exit code, most memory it used in MB) of a finished child process (subprocess.Popen), or (None, 0) while it
    still runs. Unlike proc.poll(), this also gets the child's peak memory from the kernel."""
    if proc.returncode is not None:  # already collected, e.g. by proc.terminate()
        return proc.returncode, 0.0
    pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
    if pid == 0:
        return None, 0.0
    proc.returncode = os.waitstatus_to_exitcode(status)
    return proc.returncode, usage.ru_maxrss / 1024  # ru_maxrss is in KB on Linux


def result_saved(path):
    """True if a job's metrics.json exists and is complete JSON (a job stopped while writing leaves it broken)."""
    try:
        with open(path) as f:
            json.load(f)
    except (OSError, ValueError):
        return False
    return True


def job_env(threads):
    """Environment for a job limited to this many threads: OpenMP (boosting), BLAS, and joblib (forests)."""
    names = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LOKY_MAX_CPU_COUNT"]
    return {**os.environ, **{name: str(threads) for name in names}}


def job_command(pid, proc_dir="/proc"):
    """The command line of a running process ('' if it is gone), read from Linux's /proc."""
    try:
        with open(os.path.join(proc_dir, str(pid), "cmdline"), "rb") as f:
            return f.read().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def is_our_job(command):
    """True if a command line is one of the jobs this server starts (so a reused pid is never killed by mistake)."""
    jobs = ("scripts/run_all.py", "scripts/run_experiment.py", "scripts/repeat_runs.py", "pytest")
    return any(name in command for name in jobs)


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
    """True if the relay has not yet brought data for this scale, a re-fetch was asked for, or what it brought no
    longer matches what a run needs (another set of shots, or extras made before a column was added)."""
    manifest = state["relay_manifest"]
    if state["force_fetch"] or manifest is None or manifest.get("scale") != scale:
        return True
    return (manifest.get("full_n_clean", 3000) != config.FULL_N_CLEAN
            or manifest.get("extras_columns") != list(extras.COLUMNS))


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
