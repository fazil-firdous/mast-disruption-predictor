"""Always-on research server for a personal VPS: waits out network outages, re-runs the full pipeline for every
new commit, spends idle time on the E10 seed sweep, and obeys Telegram commands from the owner's chat only.

Run it under systemd (deploy/setup_vps.sh installs a service) or inside `screen`. Only one copy can run at a time.
"""
import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import predict_shot
from src import config, experiment, notify, server
from src.data import check_connection, check_shot_table

PY = sys.executable
CURRENT = {"job": None, "proc": None, "log": None}  # the running job, shared by the loop and the command handler


def git(*args):
    """Run a git command in the repo and return its stdout ('' on failure)."""
    result = subprocess.run(["git", *args], cwd=config.REPO_DIR, capture_output=True, text=True)
    return result.stdout.strip()


def head():
    """Short hash of the checked-out commit."""
    return git("rev-parse", "--short", "HEAD")


def update_code(state, force=False):
    """Pull new commits (at most every UPDATE_CHECK_MINUTES unless forced). If the code changed, install
    requirements and restart this process on the new code."""
    if not force and time.time() - state["last_update_check"] < config.UPDATE_CHECK_MINUTES * 60:
        return
    state["last_update_check"] = time.time()
    before = head()
    git("pull", "--ff-only", "-q")
    if head() == before:
        return
    subprocess.run([PY, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=config.REPO_DIR)
    notify.send(f"new code {before} -> {head()}, restarting the server")
    server.write_state(state)
    # replace this process with a fresh one, so the new code is what runs from now on
    os.execv(PY, [PY] + sys.argv)


def network_status():
    """(ok, text) for the data server and the shot table server."""
    ok_data, message_data = check_connection(attempts=2)
    ok_table, message_table = check_shot_table()
    return ok_data and ok_table, f"data: {message_data}; table: {message_table}"


def tail(path, n=40):
    """Last n lines of a text file."""
    if not path or not os.path.exists(path):
        return "no log yet"
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def results_text():
    """Headline results for /results."""
    summary_path = os.path.join(config.RESULTS_DIR, "SUMMARY.md")
    text = "no full run finished yet\n"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            text = f.read()
    seeds = experiment.seed_summary()
    if seeds["n_seeds"]:
        text += (f"\nE10 over {seeds['n_seeds']} split seeds: catch {seeds['catch_rate_mean']:.3f} "
                 f"+- {seeds['catch_rate_std']:.3f}, false alarms {seeds['false_alarm_rate_mean']:.3f} "
                 f"+- {seeds['false_alarm_rate_std']:.3f}")
    return text


def handle_command(name, arg, state):
    """Carry out one owner command. Commands are a fixed list; nothing from the chat is ever run as a shell command."""
    if name == "status":
        notify.send(server.status_text(state, head(), CURRENT["job"], network_status()[1]))
    elif name == "results":
        notify.send(results_text())
    elif name == "log":
        notify.send(tail(CURRENT["log"]))
    elif name == "predict" and arg.isdigit():
        notify.send(predict_shot.run(int(arg), telegram=True))
    elif name == "diagnose":
        subprocess.run([PY, "scripts/diagnose_network.py"], cwd=config.REPO_DIR, timeout=600)
    elif name == "run":
        state["force_pipeline"] = True
        notify.send("full run queued")
    elif name == "update":
        update_code(state, force=True)
        notify.send(f"already on the latest code ({head()})")
    elif name in ("pause", "resume"):
        state["paused"] = name == "pause"
        notify.send(f"server {'paused: no new jobs' if state['paused'] else 'resumed'}")
    elif name == "stop" and CURRENT["proc"] is not None:
        CURRENT["proc"].terminate()
        notify.send(f"stopping {CURRENT['job']}")
    else:
        notify.send(server.HELP)


def poll_commands(state):
    """Wait up to TELEGRAM_POLL_S for owner commands and handle them. Messages from other chats are ignored."""
    if notify.owner_chat_id() is None:
        time.sleep(config.TELEGRAM_POLL_S)
        return
    for update_id, chat_id, text in notify.get_updates(state["offset"], config.TELEGRAM_POLL_S):
        state["offset"] = update_id + 1
        name, arg = server.parse_command(text)
        if chat_id == notify.owner_chat_id() and name:
            try:
                handle_command(name, arg, state)
            except Exception as error:
                notify.send(f"command /{name} failed: {type(error).__name__}: {error}"[:500])
    server.write_state(state)


def run_job(label, cmd, state):
    """Run one command as a child process, logging to LOGS_DIR, and keep answering commands while it runs."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    log_path = os.path.join(config.LOGS_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{label.replace(' ', '_')}.log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=config.REPO_DIR, stdout=log, stderr=subprocess.STDOUT)
        CURRENT.update(job=label, proc=proc, log=log_path)
        while proc.poll() is None:
            poll_commands(state)
    CURRENT.update(job=None, proc=None)
    return proc.returncode


def run_pipeline(state, scale):
    """Tests, then the full pipeline for the current commit. Old seed-sweep results are archived first."""
    commit = head()
    state["force_pipeline"] = False
    if run_job(f"tests {commit}", [PY, "-m", "pytest", "-q"], state) != 0:
        notify.send(f"tests FAILED on {commit}; not running. /log shows why")
        state.update(failed_commit=commit, failed_at=time.time())
        return
    code = run_job(f"pipeline {commit}", [PY, "scripts/run_all.py", "--scale", scale], state)
    if code == 0:
        old = os.path.join(config.RESULTS_DIR, "E10")
        if os.path.isdir(old):
            shutil.move(old, os.path.join(config.RESULTS_DIR, f"E10_before_{commit}"))
        state.update(last_ok_commit=commit, seeds_done=[], failed_commit=None)
    else:
        state.update(failed_commit=commit, failed_at=time.time())
        notify.send(f"full run of {commit} failed (exit {code}); retrying in {config.PIPELINE_RETRY_HOURS} h. /log for details")


def run_seed(state, seed):
    """One E10 seed; report the running mean +- spread."""
    code = run_job(f"E10 seed {seed}", [PY, "scripts/run_experiment.py", "--exp", "E10", "--seed", str(seed)], state)
    state["seeds_done"].append(seed)
    if code == 0:
        s = experiment.seed_summary()
        notify.send(f"E10 seed {seed} done. Over {s['n_seeds']} seeds: catch {s['catch_rate_mean']:.3f} +- "
                    f"{s['catch_rate_std']:.3f}, false alarms {s['false_alarm_rate_mean']:.3f} +- {s['false_alarm_rate_std']:.3f}")
    else:
        notify.send(f"E10 seed {seed} failed (exit {code}); skipping it")


def wait_for_network(state):
    """True if the servers answer. Otherwise tell the owner once per outage and wait NETWORK_RETRY_MINUTES."""
    ok, text = network_status()
    if ok:
        if state["network_down_since"] is not None:
            notify.send("data servers reachable again, resuming")
            state["network_down_since"] = None
        return True
    if state["network_down_since"] is None:
        state["network_down_since"] = time.time()
        notify.send(f"waiting for the data servers ({text}). Checking every {config.NETWORK_RETRY_MINUTES} min. "
                    "Send /diagnose for details.")
    deadline = time.time() + config.NETWORK_RETRY_MINUTES * 60
    while time.time() < deadline:
        poll_commands(state)
    return False


def acquire_lock():
    """Hold an exclusive lock so a second copy (say one in screen and one under systemd) exits at once."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    handle = open(os.path.join(config.LOGS_DIR, "server.lock"), "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit("another research server is already running")
    return handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=["small", "full"], default="full")
    args = parser.parse_args()
    lock = acquire_lock()  # noqa: F841  (kept open for the life of the process)
    state = server.read_state()
    state["started_at"] = time.time()
    notify.send(f"research server started on commit {head()} (workspace {os.path.abspath(config.HOME_DIR)}). /help")
    while True:
        update_code(state)
        if server.heartbeat_due(state, time.time()):
            notify.send("still alive\n" + server.status_text(state, head(), CURRENT["job"], "not checked"))
            state["last_heartbeat"] = time.time()
        job, seed = server.next_job(state, head(), time.time(),
                                    os.path.exists(os.path.join(config.PROCESSED_DIR, "windows.parquet")))
        if job is None:
            poll_commands(state)
        elif job == "seed":
            run_seed(state, seed)
        elif wait_for_network(state):
            run_pipeline(state, args.scale)
        server.write_state(state)


if __name__ == "__main__":
    main()
