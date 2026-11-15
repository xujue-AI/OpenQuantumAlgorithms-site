"""Independent lifecycle checks with dummy workers and temporary receipts only."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_optimization_word_native.py"
RUNNER = ROOT / "src/openquantum/optimization_word_native_runner.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("independent_word_watchdog", RUNNER)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def audit(tmp_path, behavior, seconds=4.0):
    output = tmp_path / behavior
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--audit", behavior,
         "--audit-seconds", str(seconds), "--output-dir", str(output)],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True, text=True, timeout=seconds + 3,
    )
    lines = completed.stdout.splitlines()
    assert lines, completed.stderr
    result = json.loads(lines[-1])
    return completed, result, output


def test_dummy_success_uses_only_reported_fake_calls_and_verifies_shutdown(tmp_path):
    completed, result, output = audit(tmp_path, "success")
    assert completed.returncode == 0, (result, completed.stderr)
    assert result["status"] == "complete"
    assert result["physical_execution_performed"] is False
    assert result["counters"]["charged_trace_permits"] == 2
    assert result["cleanup"]["complete"] is True
    assert result["cleanup"]["live_members"] == []
    guardian = json.loads((output / "guardian-receipt.json").read_text())
    assert guardian["parent_finished"] is True
    assert guardian["cleanup"]["complete"] is True


def test_timeout_escalates_term_to_kill_for_ignoring_dummy_worker(tmp_path):
    completed, result, _ = audit(tmp_path, "sleep", seconds=2.4)
    assert completed.returncode != 0
    assert result["status"].startswith("incomplete_")
    assert result["physical_execution_performed"] is False
    assert result["counters"]["charged_trace_permits"] == 0
    assert result["cleanup"]["complete"] is True
    assert result["cleanup"]["live_members"] == []
    assert result["cleanup"]["signals_sent"] == ["SIGTERM", "SIGKILL"]
    assert result["parent_wall_seconds"] <= 2.4


def test_orphan_same_group_is_killed_after_leader_clean_exit(tmp_path):
    completed, result, _ = audit(tmp_path, "orphan")
    assert completed.returncode == 0, (result, completed.stderr)
    assert result["worker_returncode"] == 0
    assert result["cleanup"]["complete"] is True
    assert result["cleanup"]["live_members"] == []
    assert result["cleanup"]["signals_sent"] in (["SIGTERM"], ["SIGTERM", "SIGKILL"])
    assert result["physical_execution_performed"] is False


def _wait_record(path, deadline, *, terminal_statuses=None):
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                record = json.loads(path.read_text())
                if terminal_statuses is None or record.get("status") in terminal_statuses:
                    return record
            except json.JSONDecodeError:
                pass
        time.sleep(.01)
    raise AssertionError(f"Expected finite dummy receipt: {path.name}")


def _wait_worker_started(path, identity, worker_pid, deadline):
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                raw = path.read_text()
                if raw.endswith("\n"):
                    records = [json.loads(line) for line in raw.splitlines()]
                    started = next((row for row in records if row.get("event") == "worker_started"), None)
                    if started is not None:
                        # The guardian sends worker_started after completing
                        # bootstrap; this journal event proves parent observation.
                        assert started["identity"] == identity
                        assert started["worker_pid"] == worker_pid
                        return started
            except json.JSONDecodeError:
                pass
        time.sleep(.01)
    raise AssertionError("Expected parent-observed worker_started event")


def _stop_owned_group(pgid):
    if isinstance(pgid, int) and pgid > 1 and pgid != os.getpid():
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_hard_parent_death_keeps_guardian_alive_to_stop_worker(runner, tmp_path):
    output = tmp_path / "parent-killed"
    process = subprocess.Popen(
        [sys.executable, str(SCRIPT), "--audit", "sleep", "--audit-seconds", "5",
         "--output-dir", str(output)],
        cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    owned = None
    try:
        bootstrap_deadline = time.monotonic() + 2
        owned = _wait_record(output / "worker-pid.json", bootstrap_deadline)
        assert owned["worker_pid"] > 1 and owned["guardian_pid"] > 1
        _wait_worker_started(output / "attempt-journal.jsonl", owned["identity"],
                             owned["worker_pid"], bootstrap_deadline)
        process.kill()
        process.wait(timeout=2)
        assert process.returncode == -signal.SIGKILL
        guardian = _wait_record(output / "guardian-receipt.json", time.monotonic() + 3,
                                terminal_statuses={"parent_lost", "parent_deadline_exhausted",
                                                   "parent_finished", "guardian_cancelled", "guardian_failed"})
        assert guardian["status"] != "guardian_failed", guardian
        assert guardian["status"] == "parent_lost", guardian
        assert guardian["parent_finished"] is False
        assert guardian["cleanup"]["complete"] is True
        assert guardian["cleanup"]["live_members"] == []
        assert guardian["worker_returncode"] is not None
        # A lost controller cannot leave an authoritative completion seal.
        if (output / "parent-receipt.json").exists():
            parent = json.loads((output / "parent-receipt.json").read_text())
            assert parent["status"] != "complete"
        caps = runner.RunnerCaps(parent_wall_seconds=5, cleanup_seconds=5 / 3,
                                 termination_grace_seconds=.1)
        journal = runner.AttemptJournal(output, owned["identity"], ["dummy-0", "dummy-1"], caps)
        assert journal.snapshot()["cumulative_parent_wall_seconds_charged"] == 5
        with pytest.raises(runner.RunnerError):
            journal.claim(time.monotonic())
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        if owned is not None:
            _stop_owned_group(owned["worker_pid"])
            _stop_owned_group(owned["guardian_pid"])


def test_expired_pre_runner_clock_never_launches_or_claims(runner, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("An expired pre-import clock must not launch a worker")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    caps = runner.RunnerCaps(parent_wall_seconds=1, cleanup_seconds=.4,
                             termination_grace_seconds=.05)
    output = tmp_path / "expired"
    with pytest.raises(TimeoutError):
        runner.run_bounded_worker([sys.executable, "-c", "pass"],
            identity={"mode": "dummy_protocol_audit"},
            requests=[{"class_id": "one", "trace_input_sha256": "a" * 64}],
            output_directory=output, guardian_command=[sys.executable, str(SCRIPT)],
            source_bundle_bytes=0, caps=caps, start_monotonic=time.monotonic() - 2)
    assert not output.exists()


@pytest.mark.parametrize("clock", [float("nan"), float("inf"), True, "future"])
def test_invalid_start_clock_is_rejected_before_any_attempt(runner, tmp_path, monkeypatch, clock):
    def forbidden(*args, **kwargs):
        raise AssertionError("An invalid start clock must not launch a worker")
    monkeypatch.setattr(runner.subprocess, "Popen", forbidden)
    if clock == "future":
        clock = time.monotonic() + 100
    output = tmp_path / "invalid-clock"
    with pytest.raises((ValueError, runner.RunnerError)):
        runner.run_bounded_worker([sys.executable, "-c", "pass"],
            identity={"mode": "dummy_protocol_audit"},
            requests=[{"class_id": "one", "trace_input_sha256": "a" * 64}],
            output_directory=output, guardian_command=[sys.executable, str(SCRIPT)],
            source_bundle_bytes=0, start_monotonic=clock)
    assert not output.exists()


def _mocked_verifier(runner, tmp_path, monkeypatch, *, wall_seconds=.1):
    spec = importlib.util.spec_from_file_location("independent_word_runner_cli", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    output = tmp_path / "fake-final-receipts"
    output.mkdir()
    freeze = {"native_plan_sha256": "a" * 64, "caps": {
        "trace_calls": 54, "parent_wall_seconds": 60.0}}
    parent = {"status": "pending_guardian_confirmation", "mode": "approved_native_execution",
              "parent_wall_seconds": wall_seconds, "caps": freeze["caps"],
              "counters": {"charged_trace_permits": 54}}
    guardian = {"status": "parent_finished", "parent_finished": True,
                "cleanup": {"complete": True}}
    result = {"plan_sha256": freeze["native_plan_sha256"],
              "results": [{"class_id": "duplicated-fake-class"}] * 54}
    for name, record in (("parent-receipt.json", parent), ("guardian-receipt.json", guardian),
                         ("native-result.json", result)):
        (output / name).write_text(json.dumps(record) + "\n")
    monkeypatch.setattr(cli, "_scope", lambda *args, **kwargs: (freeze, None, runner, output))
    return cli


def test_read_only_verifier_rejects_complete_looking_over_budget_receipts(runner, tmp_path, monkeypatch):
    cli = _mocked_verifier(runner, tmp_path, monkeypatch, wall_seconds=61.0)
    with pytest.raises((ValueError, runner.RunnerError, KeyError)):
        cli._verify(tmp_path / "unused-synthetic-freeze.json")


def test_read_only_verifier_rejects_duplicate_classes_despite_length_54(runner, tmp_path, monkeypatch):
    cli = _mocked_verifier(runner, tmp_path, monkeypatch)
    with pytest.raises((ValueError, runner.RunnerError, KeyError)):
        cli._verify(tmp_path / "unused-synthetic-freeze.json")


def test_guardian_enforces_deadline_during_final_parent_export(tmp_path):
    output = tmp_path / "delayed-final-export"
    wrapper = tmp_path / "delayed_parent.py"
    wrapper.write_text(
        "import os,sys,time,importlib.util\n"
        f"spec=importlib.util.spec_from_file_location('delayed_watchdog',{str(RUNNER)!r})\n"
        "runner=importlib.util.module_from_spec(spec)\n"
        "sys.modules[spec.name]=runner\n"
        "spec.loader.exec_module(runner)\n"
        "from pathlib import Path\n"
        f"out=Path({str(output)!r})\n"
        "os.environ['OQP_NATIVE_DUMMY_OUTPUT']=str(out)\n"
        "original=runner.atomic_record\n"
        "def delayed(path,value,**kwargs):\n"
        "    result=original(path,value,**kwargs)\n"
        "    if Path(path).name=='parent-receipt.json': time.sleep(3)\n"
        "    return result\n"
        "runner.atomic_record=delayed\n"
        "runner.run_bounded_worker(\n"
        f"    [sys.executable,{str(SCRIPT)!r},'--dummy-worker','success','--dummy-count','2'],\n"
        "    identity={'mode':'dummy_protocol_audit','test':'late_export'},\n"
        "    requests=[{'class_id':f'dummy-{i}','trace_input_sha256':'a'*64} for i in range(2)],\n"
        f"    output_directory=out,guardian_command=[sys.executable,{str(SCRIPT)!r}],\n"
        "    source_bundle_bytes=0,caps=runner.RunnerCaps(parent_wall_seconds=2.5,\n"
        "      cleanup_seconds=.8,termination_grace_seconds=.05),\n"
        "    start_monotonic=time.monotonic(),mode='dummy_protocol_audit')\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(wrapper)], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    owned = None
    try:
        owned = _wait_record(output / "worker-pid.json", time.monotonic() + 2)
        process.wait(timeout=5)
        assert process.returncode != 0
        guardian = _wait_record(output / "guardian-receipt.json", time.monotonic() + 2)
        assert guardian["parent_finished"] is False
        assert guardian["cleanup"]["complete"] is True
        parent = json.loads((output / "parent-receipt.json").read_text())
        assert parent["status"] != "complete"
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        if owned is not None:
            _stop_owned_group(owned["worker_pid"])
            _stop_owned_group(owned["guardian_pid"])
