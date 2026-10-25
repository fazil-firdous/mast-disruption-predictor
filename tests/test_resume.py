"""Tests for reusing unchanged work (src/resume.py), the plain-English messages (src/explain.py), the false-alarm
breakdown and the server's leftover-job handling. No network."""
import os

import numpy as np
import pytest

from src import config, evaluate, experiment, explain, resume, server
from tests.test_evaluate import tiny_project  # noqa: F401  (pytest fixture)


def fake_repo(tmp_path):
    for name in ["src/models.py", "src/server.py", "scripts/build_dataset.py"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x = 1\n")
    return tmp_path


def test_stamp_follows_science_code_and_data_but_not_server_code(monkeypatch, tmp_path):
    repo = fake_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "REPO_DIR", str(repo))
    data = tmp_path / "windows.parquet"
    data.write_bytes(b"abc")

    def now():
        return resume.stamp([str(data)], code=resume.code_digest())
    first = now()
    assert now() == first
    (repo / "src" / "server.py").write_text("x = 2\n")
    assert now() == first
    (repo / "src" / "models.py").write_text("x = 2\n")
    second = now()
    assert second != first
    data.write_bytes(b"abd")
    third = now()
    assert third != second
    monkeypatch.setattr(config, "HEARTBEAT_HOURS", 1)
    monkeypatch.setattr(config, "NEXT_STEP", "something else")
    assert now() == third
    monkeypatch.setattr(config, "HORIZON_MS", 31)
    assert now() != third


def test_stamp_uses_the_code_loaded_at_start(monkeypatch, tmp_path):
    repo = fake_repo(tmp_path / "repo")
    monkeypatch.setattr(config, "REPO_DIR", str(repo))
    before = resume.stamp([])
    (repo / "src" / "models.py").write_text("edited while running\n")
    assert resume.stamp([]) == before
    assert resume.stamp([], code=resume.code_digest()) != before


def test_saved_outcome_is_reused_only_under_the_same_stamp(tmp_path):
    outcome = {"catch_rate": np.float64(0.25), "warning_median_ms": float("nan"), "rows": (1, np.int64(2))}
    resume.save(str(tmp_path), "abc", outcome)
    assert resume.load(str(tmp_path), "abc") == {"catch_rate": 0.25, "warning_median_ms": None, "rows": [1, 2]}
    assert resume.load(str(tmp_path), "other") is None
    assert resume.load(str(tmp_path / "missing"), "abc") is None


def test_second_build_reuses_the_dataset_and_a_label_change_rebuilds(monkeypatch, tmp_path):
    from scripts import build_dataset
    from src import dataset
    from tests.test_dataset import fake_shots
    shots, table = fake_shots(6, 6)
    monkeypatch.setattr(build_dataset, "is_cached", lambda shot_id: True)
    monkeypatch.setattr(dataset, "load_shot", lambda shot_id: shots[shot_id])
    monkeypatch.setattr(config, "SKIPPED_CSV", str(tmp_path / "skipped.csv"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path / "processed"))
    first = build_dataset.build("small", table=table, offline=True)
    again = build_dataset.build("small", table=table, offline=True)
    assert "reused" not in first and again["reused"] is True and again["windows"] == first["windows"]
    table.loc[table.index[-1], "shot_postshot_comment"] = "disruption at 0.2 s"  # a clean shot gets a note
    assert "reused" not in build_dataset.build("small", table=table, offline=True)


def test_run_all_reuses_an_unchanged_experiment(monkeypatch, tmp_path):
    from scripts import run_all
    calls = []
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(experiment, "run", lambda exp: calls.append(exp) or [{"horizon_ms": 10, "catch_rate": 0.5,
                                                                             "false_alarm_rate": 0.0, "pr_auc": 0.3}])
    reused = []
    first = run_all.run_or_reuse("E4", "stamp1", reused)
    assert run_all.run_or_reuse("E4", "stamp1", reused) == first
    assert calls == ["E4"] and reused == ["E4"]
    run_all.run_or_reuse("E4", "stamp2", reused)
    assert calls == ["E4", "E4"]


def test_leftover_job_is_recognised_from_its_command_line(tmp_path):
    (tmp_path / "123").mkdir()
    (tmp_path / "123" / "cmdline").write_bytes(b"/root/mastu/.venv/bin/python\0scripts/run_all.py\0--scale\0full\0")
    command = server.job_command(123, proc_dir=str(tmp_path))
    assert "scripts/run_all.py --scale full" in command and server.is_our_job(command)
    assert server.job_command(999, proc_dir=str(tmp_path)) == ""
    assert not server.is_our_job("/usr/sbin/sshd -D")


def test_seed_sweep_kept_only_under_the_same_stamp():
    state = server.new_state()
    assert not server.seeds_still_valid(state, "abc")
    state["seeds_stamp"] = "abc"
    assert server.seeds_still_valid(state, "abc") and not server.seeds_still_valid(state, "xyz")


def test_false_alarms_split_by_how_the_clean_shot_ended():
    result = np.array(["false_alarm", "quiet", "false_alarm", "quiet", "caught"], dtype=object)
    out = evaluate.false_alarms_by_ending([1, 2, 3, 4, 5], result, {1, 2, 5})
    assert out == {"unnoted_quench": {"n_clean": 2, "false_alarms": 1}, "no_quench": {"n_clean": 2, "false_alarms": 1}}


def model_metrics(catch, ci_low, false_alarm):
    test = {"n_disrupted": 34, "n_clean": 46, "catch_rate": catch, "tardy_rate": 1 / 34, "miss_rate": 1 - catch - 1 / 34,
            "false_alarm_rate": false_alarm, "warning_median_ms": None,
            "ci95": {"catch_rate": [ci_low, 0.2], "false_alarm_rate": [0.0, 0.1]},
            "false_alarms_by_ending": {"unnoted_quench": {"n_clean": 20, "false_alarms": 2},
                                       "no_quench": {"n_clean": 26, "false_alarms": 0}}}
    return {"model": "hgb", "test": test, "val": {"meets_false_alarm_limit": True}}


def test_model_message_counts_shots_and_compares_with_chance():
    text = explain.model_text("E3", model_metrics(3 / 34, 0.0, 2 / 46))
    assert "3 of 34 (8.8%)" in text and "2 of 46 (4.3%)" in text and "none caught" in text
    assert "cannot yet call this real skill" in text and "lean towards the unnoted drops" in text
    assert "real skill." in explain.model_text("E3", model_metrics(12 / 34, 0.2, 2 / 46))


def test_sweep_temporal_conformal_pu_seed_and_dataset_messages():
    rows = [{"drop_signal": "none", "model": "hgb", "catch_rate": 6 / 34, "false_alarm_rate": 2 / 46, "pr_auc": 0.26,
             "n_disrupted": 34, "n_clean": 46, "positive_share": 0.03},
            {"drop_signal": "prad", "model": "hgb", "catch_rate": 3 / 34, "false_alarm_rate": 1 / 46, "pr_auc": 0.18,
             "n_disrupted": 34, "n_clean": 46, "positive_share": 0.03}]
    text = explain.describe("E5", rows)
    assert "without radiated power: caught 3 of 34" in text and "differ by 3 caught shot(s)" in text
    drift = {"baseline": model_metrics(0.0, 0.0, 0.0)["test"],
             "hgb": {**model_metrics(1 / 32, 0.0, 8 / 68)["test"], "n_disrupted": 32, "n_clean": 68,
                     "ci95": {"catch_rate": [0.0, 0.1], "false_alarm_rate": [0.059, 0.206]}}}
    assert "campaign drift" in explain.describe("E6", drift)
    conformal = [{"method": "conformal", "alpha": 0.01, "threshold": None, "n_cal_clean": 23},
                 {"method": "conformal", "alpha": 0.1, "threshold": 0.7, "n_cal_clean": 23, "guarantee_holds": False,
                  "test_false_alarm_rate": 6 / 46, "test_catch_rate": 2 / 34, "n_test_disrupted": 34, "n_test_clean": 46}]
    text = explain.describe("E8", conformal)
    assert "needs at least 99 calibration shots, we have 23" in text and "6 of 46 (13.0%) -> BROKEN" in text
    pu = {"label_frequency_c": 0.4, "mean_weight_unnoted_quench_windows": 0.3, "mean_weight_other_unlabelled_windows": 0.01,
          "rows": [{"method": "pu", "test_set": "all test shots", "catch_rate": 0.1, "n_disrupted": 30,
                    "false_alarm_rate": 0.05, "n_clean": 40}]}
    assert "3 of 30" in explain.describe("E9", pu) and "40.0%" in explain.describe("E9", pu)
    seeds = {"n_seeds": 5, "catch_rate_mean": 0.1, "catch_rate_std": 0.04, "false_alarm_rate_mean": 0.05,
             "false_alarm_rate_std": 0.02}
    assert "5 different random splits" in explain.describe("E10", seeds)
    seeds["false_alarms_by_ending"] = {"unnoted_quench": {"n_clean": 170, "false_alarms": 17},
                                       "no_quench": {"n_clean": 60, "false_alarms": 1}}
    assert "17 of 170" in explain.describe("E10", seeds) and "(10.0%)" in explain.describe("E10", seeds)
    counts = {"shots": 398, "disrupted": 171, "clean": 227, "windows": 26169, "reused": True,
              "split_counts": {"train": 238, "val": 80, "test": 80}}
    assert "26,169 windows" in explain.dataset_text(counts) and "reused" in explain.dataset_text(counts)


@pytest.mark.parametrize("exp", ["E1", "E6", "E8", "E9", "E11"])
def test_every_real_outcome_can_be_described(tiny_project, exp):  # noqa: F811
    outcome = resume.clean_json(experiment.run(exp))
    text = explain.describe(exp, outcome)
    assert exp in text and "None" not in text and "nan" not in text
    if exp == "E1":
        assert os.path.exists(tiny_project / "results" / "E1" / "metrics.json")


def test_status_card_numbers_and_picture(monkeypatch, tmp_path):
    from src import status_card
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("seed,catch_rate,false_alarm_rate\n1,0.06,0.07\n2,0.09,0.05\n3,0.05,0.08\n4,0.07,0.07\n")
    assert status_card.skill(str(seeds))["verdict"] == "same"
    seeds.write_text("seed,catch_rate,false_alarm_rate\n1,0.60,0.05\n2,0.55,0.04\n3,0.62,0.06\n")
    assert status_card.skill(str(seeds))["verdict"] == "better"
    e5 = tmp_path / "e5.csv"
    e5.write_text("drop_signal,pr_auc\nnone,0.26\nprad,0.18\npnbi,0.28\n")
    useful = status_card.signal_usefulness(str(e5))
    assert useful["radiated power"] == pytest.approx(0.08) and useful["beam heating power"] < 0
    monkeypatch.setattr(config, "RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(config, "PROCESSED_DIR", str(tmp_path / "processed"))
    info = status_card.collect(server.new_state())
    assert info["shots"] == 0 and info["experiments"] == 0 and info["skill"] is None
    todo = [text for ok, text in status_card.needs({**info, "table_source": "report"}) if not ok]
    assert any("400 report shots" in t for t in todo) and not any("/gold" in t for t in todo)
    for case in (info, {**info, "skill": status_card.skill(str(seeds)), "signals": useful, "all_shots": 11573}):
        path = status_card.draw(case, str(tmp_path / "card.png"), "abc1234")
        assert os.path.getsize(path) > 10000


def test_log_keeps_messages_off_telegram(monkeypatch, tmp_path):
    from src import notify
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    notify.log("seed 3 done")
    notify.send("server started")
    with open(os.path.join(config.LOGS_DIR, "messages.log")) as f:
        lines = f.read().splitlines()
    assert lines[0].endswith("seed 3 done") and lines[1].endswith("server started")


def test_banner_and_progress_block():
    from src import banner
    assert [len(a) for a in banner.ART] == [len(c) for c in banner.COLOURS]
    plain = banner.banner("abc1234", "GitHub relay", "/root/mast-private", colour=False)
    assert "FAZIL'S TOKAMAK ML MODEL" in plain and "code abc1234" in plain and "\033[" not in plain
    assert "\033[" in banner.banner("abc1234", "relay", "/w", colour=True)
    info = {"shots": 3804, "all_shots": None, "experiments": 13, "seeds": 30, "comparisons": 0, "auto_labelled": 0,
            "auto_total": 0,
            "skill": {"n": 30, "catch": 0.07, "false_alarm": 0.05, "verdict": "better"}}
    text = banner.progress(info, "E10 seed 31", colour=False)
    assert "Shots learned from" in text and "30 / 400" in text and "✓ better than guessing" in text
    assert banner.bar(0, 0, colour=False) == " " * banner.BAR_WIDTH
    assert banner.skill_text(None) == "" and "one test only" in banner.skill_text({**info["skill"], "verdict": "one"}, False)
