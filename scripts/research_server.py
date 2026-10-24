"""Always-on research server for a personal VPS: waits out network outages, re-runs the full pipeline for every
new commit (reusing every result whose code and data did not change), spends idle time on the E10 seed sweep,
and obeys Telegram commands from the owner's chat only.

Run it under systemd (deploy/setup_vps.sh installs a service) or inside `screen`. Only one copy can run at a time.
"""
import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import predict_shot
from src import config, experiment, explain, gold, literature, notify, relay, resume, server, sources, status_card
from src.data import check_connection, check_shot_table, host_reachable

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
    notify.log(f"new code {before} -> {head()}, restarting the server")
    server.write_state(state)
    # replace this process with a fresh one, so the new code is what runs from now on
    os.execv(PY, [PY] + sys.argv)


def network_status():
    """(mode, text): how the next run can get its data (see server.access_mode) and what each server said."""
    ok_data, message_data = check_connection(attempts=2)
    ok_table, message_table = check_shot_table()
    mode = server.access_mode(ok_data, ok_table, relay.configured())
    return mode, f"data: {message_data}; table: {message_table}; mode: {mode or 'waiting'}"


def tail(path, n=40):
    """Last n lines of a text file."""
    if not path or not os.path.exists(path):
        return "no log yet"
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def send_report():
    """/report: every experiment described in words (REPORT.md) and all result files as a zip."""
    report = os.path.join(config.RESULTS_DIR, "REPORT.md")
    if not os.path.exists(report):
        notify.send("no report yet: it is written at the end of the first full run")
        return
    notify.send_file(report, "every experiment in words")
    archive = shutil.make_archive(os.path.join(config.DATA_DIR, "results"), "zip", config.RESULTS_DIR)
    notify.send_file(archive, "all result files")


def handle_command(name, arg, state):
    """Carry out one owner command. Commands are a fixed list; nothing from the chat is ever run as a shell command."""
    if name == "status":
        mode = network_status()[0]
        status_card.send(f"now: {CURRENT['job'] or 'idle'} | data: {mode or 'not reachable'}", state)
    elif name == "details":
        notify.send(server.status_text(state, head(), CURRENT["job"], network_status()[1]))
    elif name == "results":
        status_card.send("results so far. /report for every experiment in words", state)
    elif name == "report":
        send_report()
    elif name == "log":
        notify.send(tail(CURRENT["log"] or os.path.join(config.LOGS_DIR, "messages.log")))
    elif name == "predict" and arg.isdigit():
        notify.send(predict_shot.run(int(arg), telegram=True))
    elif name == "diagnose":
        subprocess.run([PY, "scripts/diagnose_network.py"], cwd=config.REPO_DIR, timeout=600)
    elif name == "run":
        state["force_pipeline"] = True
        notify.send("full run queued")
    elif name == "gold":
        send_next_gold(state)
    elif name in gold.ANSWERS and state["gold_current"] is not None:
        item = state["gold_current"]
        gold.record(item["shot_id"], name, item["stratum"])
        notify.send(f"shot {item['shot_id']}: {gold.ANSWERS[name]}")
        send_next_gold(state)
    elif name == "goldreport":
        shots = pd.read_parquet(os.path.join(config.PROCESSED_DIR, "shots.parquet"))
        notify.send(gold.report_text(shots, gold.read_labels()))
    elif name == "sources":
        state["last_probe_at"] = 0.0
        notify.send("checking the data sources now")
    elif name == "papers":
        state["last_literature_at"] = 0.0
        notify.send("checking arXiv now")
    elif name == "fetch":
        state.update(force_fetch=True, force_pipeline=True)
        notify.send("data re-fetch through GitHub queued, then a full run")
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


def send_next_gold(state):
    """Send the next unlabelled shot of the gold queue as a blind plot (needs a built dataset)."""
    if not os.path.exists(os.path.join(config.PROCESSED_DIR, "shots.parquet")):
        notify.send("no dataset on this server yet: the gold queue is built from it after the first full run")
        return
    items, labels = gold.queue(), gold.read_labels()
    item = gold.next_item(items, labels)
    state["gold_current"] = item
    if item is None:
        notify.send("gold queue finished. /goldreport shows how the automatic labels compare")
        return
    path = os.path.join(config.LOGS_DIR, "gold_current.png")
    gold.plot_blind(item["shot_id"], path)
    done = len(labels.drop_duplicates("shot_id"))
    notify.send_file(path, f"gold {done + 1}/{len(items)}: shot {item['shot_id']}. /d disruption, /n normal end, /u unsure")


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
        state["job_pid"] = proc.pid
        server.write_state(state)
        while proc.poll() is None:
            poll_commands(state)
    CURRENT.update(job=None, proc=None)
    state["job_pid"] = None
    return proc.returncode


def stop_leftover_job(state):
    """If an earlier server was killed while a job ran, that job may still be running and would clash with new
    work on the same files. Stop it; its finished experiments are kept and reused (src/resume.py)."""
    pid, state["job_pid"] = state["job_pid"], None
    if pid is None or not server.is_our_job(server.job_command(pid)):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    notify.log(f"stopped a job left running by the previous server (pid {pid}). Finished parts are kept and "
               "reused; only the unfinished part runs again.")


def run_relay(state, task, scale, data_dir):
    """Start a GitHub relay job, keep answering commands while it runs, then unpack its artifact into data_dir.

    Returns the job's manifest, or None if the job failed or took longer than RELAY_TIMEOUT_HOURS.
    """
    requested = relay.trigger(scale, task)
    run_id, deadline = None, time.time() + config.RELAY_TIMEOUT_HOURS * 3600
    while time.time() < deadline:
        poll_commands(state)
        if run_id is None:
            run = relay.find_run(requested)
            run_id = run["id"] if run else None
            continue
        status, conclusion = relay.run_status(run_id)
        if status != "completed":
            continue
        if conclusion != "success":
            notify.send(f"GitHub relay job {run_id} ({task}) ended with '{conclusion}'. See the Actions tab on GitHub.")
            return None
        return relay.download(run_id, data_dir)
    notify.send(f"GitHub relay ({task}) took longer than {config.RELAY_TIMEOUT_HOURS} h; giving up for now")
    return None


def fetch_via_relay(state, scale):
    """Have GitHub download the shots into DATA_DIR. Returns the relay manifest, or None."""
    notify.log(f"STFC does not answer this server; asking GitHub to fetch the data ({scale}). This takes 10-60 min.")
    CURRENT["job"] = "GitHub data relay"
    manifest = run_relay(state, "fetch", scale, config.DATA_DIR)
    if manifest is not None:
        notify.log(f"relay done: {manifest['available']} of {manifest['requested']} shots now on this server "
                    f"(shot table: {manifest['table_source']})")
    return manifest


def relay_data(state, scale):
    """Make sure relayed data for this scale is on disk. Returns its manifest, or None on failure."""
    if server.relay_needed(state, scale):
        try:
            manifest = fetch_via_relay(state, scale)
        except Exception as error:
            notify.send(f"GitHub relay failed: {type(error).__name__}: {error}"[:500])
            manifest = None
        finally:
            CURRENT["job"] = None
        if manifest is None:
            return None
        state.update(relay_manifest=manifest, force_fetch=False)
    return state["relay_manifest"]


def run_pipeline(state, scale, mode):
    """Tests, then data (directly or through the GitHub relay), then the full pipeline for the current commit.
    Old seed-sweep results are archived first."""
    commit = head()
    state["force_pipeline"] = False
    if run_job(f"tests {commit}", [PY, "-m", "pytest", "-q"], state) != 0:
        notify.send(f"tests FAILED on {commit}; not running. /log shows why")
        state.update(failed_commit=commit, failed_at=time.time())
        return
    manifest = relay_data(state, scale) if mode == "relay" else None
    if mode == "relay" and manifest is None:
        state.update(failed_commit=commit, failed_at=time.time())
        return
    args = server.pipeline_args(scale, mode, manifest)
    code = run_job(f"pipeline {commit}", [PY, "scripts/run_all.py", *args], state)
    if code == 0:
        keep_or_restart_seeds(state, commit)
        state.update(last_ok_commit=commit, failed_commit=None)
    else:
        state.update(failed_commit=commit, failed_at=time.time())
        notify.send(f"full run of {commit} failed (exit {code}); retrying in {config.PIPELINE_RETRY_HOURS} h. /log for details")


def keep_or_restart_seeds(state, commit):
    """Keep the seed sweep if the science code and data are unchanged; otherwise archive it and start again,
    because seeds made with old code cannot be averaged with new ones."""
    stamp = resume.stamp(resume.processed_files())
    if server.seeds_still_valid(state, stamp):
        return
    old = os.path.join(config.RESULTS_DIR, "E10")
    if os.path.isdir(old):
        shutil.move(old, os.path.join(config.RESULTS_DIR, f"E10_before_{commit}"))
    state.update(seeds_done=[], seeds_stamp=stamp)


def run_seed(state, seed):
    """One E10 seed; report the running mean +- spread."""
    code = run_job(f"E10 seed {seed}", [PY, "scripts/run_experiment.py", "--exp", "E10", "--seed", str(seed)], state)
    state["seeds_done"].append(seed)
    if code != 0:
        notify.log(f"E10 seed {seed} failed (exit {code}); skipping it")
        return
    notify.log(explain.seed_text(experiment.seed_summary()))
    n = len(state["seeds_done"])
    if n % config.CARD_EVERY_SEEDS == 0 or n >= config.SEED_SWEEP_MAX:
        status_card.send(f"repeat runs {n}/{config.SEED_SWEEP_MAX}", state)


def wait_for_network(state):
    """The data access mode if the next run can get data. Otherwise tell the owner once per outage, wait
    NETWORK_RETRY_MINUTES (still answering commands) and return None."""
    mode, text = network_status()
    if mode is not None:
        if state["network_down_since"] is not None:
            notify.send(f"data reachable again ({mode}), resuming")
            state["network_down_since"] = None
        return mode
    if state["network_down_since"] is None:
        state["network_down_since"] = time.time()
        notify.send(f"waiting for the data servers ({text}). Checking every {config.NETWORK_RETRY_MINUTES} min. "
                    "Send /diagnose for details.")
    deadline = time.time() + config.NETWORK_RETRY_MINUTES * 60
    while time.time() < deadline:
        poll_commands(state)
    return None


def probe_sources(state):
    """What the data sources offer now: probed here if STFC answers, else through the GitHub relay, else None."""
    if host_reachable(sources.BUCKET_URL):
        return sources.probe()
    if not relay.configured():
        return None
    folder = os.path.join(config.LOGS_DIR, "probe")
    CURRENT["job"] = "data-source check (GitHub)"
    try:
        manifest = run_relay(state, "probe", "small", folder)
    finally:
        CURRENT["job"] = None
    if manifest is None:
        return None
    with open(os.path.join(folder, "probe.json")) as f:
        return json.load(f)


def watch_sources(state):
    """Daily: report changes in the data sources. When the full shot table can be read (mastapp.site or its S3
    copy) but the data on this server came from the 400 report shots, queue a fresh fetch of the full dataset."""
    state["last_probe_at"] = time.time()
    snapshot = probe_sources(state)
    if snapshot is None:
        return
    news = sources.changes(state["last_probe"], snapshot)
    if state["last_probe"] is None:
        news = [f"data sources: {snapshot['shot_stores']} shot stores, shot table HTTP "
                f"{snapshot['shot_table_status'] or 'no answer'} (S3 copy HTTP "
                f"{snapshot.get('shot_table_mirror_status') or 'no answer'}), DEFUSE HTTP {snapshot['defuse_status']}"]
    state["last_probe"] = snapshot
    for line in news:
        notify.send(line)
    manifest = state["relay_manifest"]
    if sources.table_available(snapshot) and manifest and manifest.get("table_source") == "report":
        state.update(force_fetch=True, force_pipeline=True)
        notify.send("the full shot table can be read again: fetching all shots and running everything on them")


def watch_literature(state):
    """Weekly: send new arXiv papers on disruption prediction."""
    state["last_literature_at"] = time.time()
    new = literature.check()
    if new:
        notify.send(literature.digest(new))


def run_watchers(state):
    """Run whichever watcher is due. A failing watcher is reported, never fatal."""
    now = time.time()
    for name, due, func in [("data sources", server.due(state["last_probe_at"], config.WATCH_SOURCES_HOURS, now),
                             watch_sources),
                            ("literature", server.due(state["last_literature_at"], 24 * config.LITERATURE_DAYS, now),
                             watch_literature)]:
        if not due:
            continue
        try:
            func(state)
        except Exception as error:
            notify.send(f"{name} watcher failed: {type(error).__name__}: {error}"[:400])


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
    stop_leftover_job(state)
    notify.send(f"server started (code {head()}). /status for the picture, /help for commands")
    while True:
        update_code(state)
        if server.heartbeat_due(state, time.time()):
            status_card.send("daily update: still running", state)
            state["last_heartbeat"] = time.time()
        run_watchers(state)
        job, seed = server.next_job(state, head(), time.time(),
                                    os.path.exists(os.path.join(config.PROCESSED_DIR, "windows.parquet")))
        if job is None:
            poll_commands(state)
        elif job == "seed":
            run_seed(state, seed)
        else:
            mode = wait_for_network(state)
            if mode is not None:
                run_pipeline(state, args.scale, mode)
        server.write_state(state)


if __name__ == "__main__":
    main()
