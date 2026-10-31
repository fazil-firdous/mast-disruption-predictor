"""Tests for the idle work: the queue of repeat runs, running them side by side (lanes, memory, threads) and the
repeat-run comparisons of scripts/repeat_runs.py. No network."""
import json
import os
import subprocess
import sys

import pytest

from scripts import repeat_runs
from src import config, experiment, explain, resume, server, stats
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def test_idle_queue_runs_paired_e10_seeds_then_the_comparisons_then_the_rest():
    state = server.new_state()
    paired = config.COMPARISON_SEEDS
    assert server.idle_jobs(state, 2) == [("E10", 1), ("E10", 2)]
    state["seeds_done"] = list(range(1, paired + 1))
    first, second = list(config.COMPARISONS)[:2]
    assert server.idle_jobs(state, 2) == [(first, 1), (first, 2)]  # same kind side by side: they finish together
    state["comparisons_done"] = [[first, 1], [first, 2]]
    assert server.idle_jobs(state, 2) == [(second, 1), (second, 2)]
    assert server.idle_jobs(state, 1) == [(second, 1)]
    state["comparisons_done"] = [[name, seed] for seed in range(1, paired + 1) for name in config.COMPARISONS]
    assert server.idle_jobs(state, 2) == [("E10", paired + 1), ("E10", paired + 2)]  # then the rest of E10
    state["seeds_done"] = list(range(1, config.SEED_SWEEP_MAX + 1))
    assert server.idle_jobs(state, 2) == []


def test_next_job_order():
    state = server.new_state()
    assert server.next_job(state, "abc", 1e9, False) == "pipeline"
    state.update(last_ok_commit="abc")
    assert server.next_job(state, "abc", 1e9, True) == "idle"
    state["seeds_done"] = list(range(1, config.SEED_SWEEP_MAX + 1))
    assert server.next_job(state, "abc", 1e9, True) == "idle"  # the comparisons come next
    state["comparisons_done"] = [[name, seed] for seed in range(1, config.SEED_SWEEP_MAX + 1)
                                 for name in config.COMPARISONS]
    assert server.next_job(state, "abc", 1e9, True) is None
    assert server.next_job(state, "def", 1e9, True) == "pipeline"  # new commit
    state["paused"] = True
    assert server.next_job(state, "def", 1e9, True) is None


@pytest.mark.parametrize("cores, peak_mb, free_mb, expected", [
    (2, 0.0, 8000, 1),      # memory use not measured yet: one at a time
    (2, 500, 8000, 2),      # one per core
    (2, 500, 1000, 1),      # two would not fit with room to spare
    (4, 500, 1400, 2),
    (2, 500, None, 1),      # free memory unknown
])
def test_lanes_follow_cores_and_free_memory(cores, peak_mb, free_mb, expected):
    assert server.lanes(cores, peak_mb, free_mb) == expected


def test_free_memory_and_thread_limits(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:        4000000 kB\nMemAvailable:    2048000 kB\n")
    assert server.free_memory_mb(str(meminfo)) == 2000
    assert server.free_memory_mb(str(tmp_path / "missing")) is None
    env = server.job_env(1)
    assert env["OMP_NUM_THREADS"] == "1" and env["LOKY_MAX_CPU_COUNT"] == "1" and "PATH" in env
    assert server.cpu_cores() >= 1


def test_reap_gives_exit_code_and_peak_memory():
    proc = subprocess.Popen([sys.executable, "-c", "x = bytearray(80_000_000); x[::4096] = b'1' * len(x[::4096])"])
    code, peak = None, 0.0
    while code is None:
        code, peak = server.reap(proc)
    assert code == 0 and peak > 70
    assert server.reap(proc) == (0, 0.0)  # already collected


def test_result_saved_only_for_complete_json(tmp_path):
    path = tmp_path / "metrics.json"
    assert not server.result_saved(str(path))
    path.write_text('{"test": {"catch_rate"')
    assert not server.result_saved(str(path))
    path.write_text('{"test": {}}')
    assert server.result_saved(str(path))


def test_leftover_jobs_include_repeat_runs():
    assert server.is_our_job("/root/x/.venv/bin/python scripts/repeat_runs.py --comparison magnetics --seed 3")
    assert server.new_state()["job_pids"] == []


def test_corrected_interval_is_wider_than_a_plain_one():
    mean, low, high = stats.corrected_interval([0.02, 0.04, 0.0, 0.03, 0.01])
    assert mean == pytest.approx(0.02)
    sd, t = 0.015811, 2.776445  # sample standard deviation; t(0.975, 4 degrees of freedom)
    assert high - mean == pytest.approx(t * sd * (1 / 5 + 0.2 / 0.6) ** 0.5, rel=1e-4)  # test / train shots
    assert high - mean > 1.5 * t * sd / 5 ** 0.5 and low < 0 < high  # overlapping splits: less certain than plain
    assert stats.corrected_interval([0.05, 0.05, 0.05]) == (pytest.approx(0.05), None, None)  # too few


def write_metrics(path, catch, false_alarm):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump({"test": {"catch_rate": catch, "false_alarm_rate": false_alarm, "warning_median_ms": 20.0}}, f)


def test_summary_pairs_each_run_with_e10_of_the_same_seed(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "COMPARISONS", {"magnetics": ("feature_set", "v1+magnetics"),
                                                "no_time": ("drop_feature", "t_since_start")})
    monkeypatch.setattr(config, "COMPARISON_BASELINES", {})
    for seed in range(1, 7):
        write_metrics(tmp_path / "E10" / f"seed_{seed}", 0.10, 0.05)
        write_metrics(tmp_path / "E10" / "comparisons" / "magnetics" / f"seed_{seed}", 0.20 + 0.01 * seed, 0.05)
    write_metrics(tmp_path / "E10" / "comparisons" / "no_time" / "seed_9", 0.08, 0.04)  # no E10 seed 9: left out
    (tmp_path / "E10" / "comparisons" / "no_time" / "seed_2").mkdir()  # unfinished: left out
    summary = repeat_runs.summary()
    assert list(summary) == ["magnetics"]
    row = summary["magnetics"]
    assert row["n_seeds"] == 6 and row["catch_rate_change"] == pytest.approx(0.135)
    assert row["catch_rate_change_low"] > 0 and row["false_alarm_rate_change"] == 0
    assert os.path.exists(tmp_path / "E10" / "comparisons" / "summary.csv")
    text = explain.comparisons_text(summary)
    assert "with magnetic signals (6 splits)" in text and "+13.5 points" in text and "clearly higher" in text
    assert "None" not in text and "nan" not in text
    assert "none finished" in explain.comparisons_text({})
    few = {"magnetics": {**row, "catch_rate_change_low": None, "catch_rate_change_high": None}}
    assert "too few splits for a verdict yet" in explain.comparisons_text(few)


def test_comparison_without_a_change_reproduces_e10(monkeypatch, tiny_project):  # noqa: F811
    monkeypatch.setattr(config, "COMPARISONS", {"same": ("drop_feature", None), "no_time": ("drop_feature",
                                                                                            "t_since_start")})
    (tiny_project / "results" / "E1").mkdir(parents=True)
    with open(tiny_project / "results" / "E1" / "metrics.json", "w") as f:  # makes logreg (fast here) the best model
        json.dump({"val": {"catch_rate": 1.0}}, f)
    experiment.run("E10", seed=3)
    same = repeat_runs.run("same", 3)
    with open(tiny_project / "results" / "E10" / "seed_3" / "metrics.json") as f:
        assert same["test"]["catch_rate"] == json.load(f)["test"]["catch_rate"]
    without_time = repeat_runs.run("no_time", 3)
    assert without_time["n_train_windows"] == same["n_train_windows"]  # same split, fewer features
    assert repeat_runs.summary()["same"]["catch_rate_change"] == 0


def test_comparison_stamp_is_separate_from_the_science_stamp(monkeypatch, tmp_path):
    for name in ("shots.parquet", "windows.parquet"):
        (tmp_path / name).write_text("data")
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path))
    science, comparisons = resume.stamp(resume.processed_files()), repeat_runs.stamp()
    monkeypatch.setattr(config, "COMPARISONS", {**config.COMPARISONS, "extra": ("drop_signal", "prad")})
    assert resume.stamp(resume.processed_files()) == science  # adding a comparison never reruns E0-E15
    assert repeat_runs.stamp() != comparisons


def test_idle_round_bookkeeping(monkeypatch, tmp_path):
    from scripts import research_server
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    cards = []
    monkeypatch.setattr(research_server.status_card, "send", lambda caption, state=None: cards.append(caption))
    write_metrics(tmp_path / "E10" / "seed_9", 0.1, 0.05)
    (tmp_path / "E10" / "seed_10").mkdir()
    (tmp_path / "E10" / "seed_10" / "metrics.json").write_text('{"test": ')  # stopped while writing
    state = server.new_state()
    state["seeds_done"] = list(range(1, 9))
    assert research_server.record_idle(state, "E10", 9, 0)
    assert not research_server.record_idle(state, "E10", 10, 1)
    assert not (tmp_path / "E10" / "seed_10").exists() and state["seeds_done"][-2:] == [9, 10]
    research_server.report_idle(state, "E10", 2)  # 8 -> 10 tried: one picture
    assert cards == [f"repeat runs 10/{config.SEED_SWEEP_MAX}"]
    assert research_server.idle_command("magnetics", 3)[1][-4:] == ["--comparison", "magnetics", "--seed", "3"]


def test_pipeline_runs_independent_experiments_side_by_side(monkeypatch, tmp_path):
    from scripts import run_all
    from src import dataset
    from tests.test_dataset import fake_shots
    shots, table = fake_shots(30, 30)
    monkeypatch.setattr(dataset, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))
    labelled = dataset.label_shots(list(table["shot_id"]), table)
    labelled["split"] = labelled["shot_id"].map(dataset.random_split(labelled))
    labelled["split_temporal"] = labelled["shot_id"].map(dataset.temporal_split(labelled))
    windows = dataset.build_windows(labelled).merge(labelled[["shot_id", "split", "split_temporal"]], on="shot_id")
    outcomes = {}
    for home, lanes in (("side_by_side", 2), ("one_at_a_time", 1)):
        processed = tmp_path / home / "data" / "processed"
        processed.mkdir(parents=True)
        windows.to_parquet(processed / "windows.parquet")
        labelled.to_parquet(processed / "shots.parquet")
        (tmp_path / home / "results" / "E1").mkdir(parents=True)
        with open(tmp_path / home / "results" / "E1" / "metrics.json", "w") as f:  # logreg (fast) is the best model
            json.dump({"val": {"catch_rate": 1.0}}, f)
        monkeypatch.setenv("MAST_HOME", str(tmp_path / home))  # the spawned processes read their paths from here
        monkeypatch.setattr(config, "PROCESSED_DIR", str(processed))
        monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / home / "results"))
        monkeypatch.setattr(server, "lanes", lambda *args, n=lanes: n)
        reused = []
        outcomes[home] = run_all.run_side_by_side(["E11", "E12"], "stamp1", reused, measured=True)
        assert reused == [] and all(outcomes[home].values())
    assert outcomes["side_by_side"] == outcomes["one_at_a_time"]
    reused = []
    assert run_all.run_side_by_side(["E11", "E12"], "stamp1", reused, measured=True) == outcomes["one_at_a_time"]
    assert reused == ["E11", "E12"]
