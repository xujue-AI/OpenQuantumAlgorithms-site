"""Only pure and dummy-subprocess tests; no Native Trace is invoked."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_optimization_word_native.py"
MODULE = ROOT / "src/openquantum/optimization_word_native_runner.py"
spec = importlib.util.spec_from_file_location("word_native_runner_test", MODULE)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


@pytest.mark.parametrize("field,value", [
    ("trace_calls", 55), ("trace_calls", True), ("parent_wall_seconds", 61),
    ("parent_wall_seconds", float("nan")), ("cleanup_seconds", 60),
    ("termination_grace_seconds", 5), ("decoded_record_bytes", 67108865),
    ("combined_bundle_bytes", 16777217),
])
def test_fixed_caps_reject_widening(field, value):
    with pytest.raises(ValueError):
        runner.RunnerCaps(**{field: value})


def test_approval_is_disabled_without_specific_scope(tmp_path):
    with pytest.raises(runner.ExecutionDisabled):
        runner.validate_approval({}, {}, tmp_path)


def test_exact_approval_binds_source_machine_caps_and_output(tmp_path):
    freeze = dict(candidate_recipe_sha256="b"*64, source_sha256={"adapter.py": "c"*64},
                  fixed_machine={"d": 15}, caps={"trace_calls": 54}, output_directory=str(tmp_path))
    approval = dict(kind="coordinator_scoped_native_execution", execution_authorized=True,
        freeze_sha256=runner.sha256(runner.canonical(freeze)), **freeze, automatic_rerun_authorized=False)
    runner.validate_approval(approval, freeze, tmp_path)
    for field, value in (("fixed_machine", {"d": 17}), ("source_sha256", {}),
                         ("caps", {"trace_calls": 55}), ("output_directory", str(tmp_path / "fresh"))):
        with pytest.raises(runner.ExecutionDisabled):
            runner.validate_approval(dict(approval, **{field: value}), freeze, tmp_path)


def test_claim_and_unknown_permit_are_conservative_and_one_use(tmp_path):
    journal = runner.AttemptJournal(tmp_path, {"scope": "fixed"}, ["c0", "c1"])
    journal.claim(time.monotonic())
    journal.grant("c0", "a"*64, deadline=time.monotonic()+1)
    snapshot = journal.snapshot()
    assert snapshot["actual_trace_calls_upper_bound"] == 1
    assert snapshot["unknown_entry_permits"] == ["c0"]
    assert snapshot["cumulative_parent_wall_seconds_charged"] == 60
    loaded = runner.AttemptJournal(tmp_path, {"scope": "fixed"}, ["c0", "c1"])
    assert loaded.snapshot() == snapshot
    with pytest.raises(runner.RunnerError, match="already claimed"):
        loaded.claim(time.monotonic())


def test_claim_without_first_journal_line_charges_full_wall(tmp_path):
    (tmp_path / "attempt-claim.json").write_text('{}')
    journal = runner.AttemptJournal(tmp_path, {"scope": "fixed"}, ["c0"])
    assert journal.snapshot()["cumulative_parent_wall_seconds_charged"] == 60


def test_journal_rejects_truncation_and_counter_tampering(tmp_path):
    path = tmp_path / "attempt-journal.jsonl"
    path.write_bytes(b'{"incomplete":')
    with pytest.raises(runner.ProtocolError):
        runner.AttemptJournal(tmp_path, {}, ["c0"])
    path.write_bytes(runner.canonical(dict(sequence=False, identity={}, event="attempt_started")))
    with pytest.raises(runner.ProtocolError):
        runner.AttemptJournal(tmp_path, {}, ["c0"])


def test_duplicate_requests_and_permits_never_gain_budget(tmp_path):
    with pytest.raises(ValueError):
        runner.AttemptJournal(tmp_path, {}, ["c0", "c0"])
    journal = runner.AttemptJournal(tmp_path, {}, ["c0"])
    journal.claim(time.monotonic())
    journal.grant("c0", "a"*64, deadline=time.monotonic()+1)
    with pytest.raises(runner.ProtocolError):
        journal.grant("c0", "a"*64, deadline=time.monotonic()+1)
    with pytest.raises(runner.ProtocolError):
        journal.grant("c1", "a"*64, deadline=time.monotonic()+1)


def test_reported_entry_and_return_are_not_claimed_actual_confirmation(tmp_path):
    journal = runner.AttemptJournal(tmp_path, {}, ["c0"])
    journal.claim(time.monotonic())
    journal.grant("c0", "a"*64, deadline=time.monotonic()+1)
    journal.entry("c0")
    journal.returned("c0", "b"*64)
    value = journal.snapshot()
    assert value["reported_native_callsites"] == 1
    assert value["reported_native_return_receipts"] == 1
    assert "confirmed_native_entries" not in value


def test_raw_return_guard_fails_before_appending(tmp_path, monkeypatch):
    client = runner.WorkerPermitClient(-1, -1, {}, tmp_path, deadline=time.monotonic()+1)
    first = runner.canonical(dict(class_id="c0", producer_record={"value": 1}))
    monkeypatch.setattr(runner, "MAX_DECODED_RECORD_BYTES", 2*len(first)-1)
    client.capture_return({"class_id": "c0"}, {"value": 1})
    with pytest.raises(runner.ProtocolError):
        client.capture_return({"class_id": "c1"}, {"value": 1})
    assert (tmp_path / "producer-returns.jsonl").read_bytes() == first


def test_received_late_permit_never_reports_native_entry(tmp_path):
    read_event, write_event = os.pipe()
    read_reply, write_reply = os.pipe()
    try:
        client = runner.WorkerPermitClient(write_event, read_reply, {}, tmp_path,
                                           deadline=time.monotonic()-1)
        client.buffer = runner.canonical(dict(event="permit", identity={}, request_id="c0",
                                              trace_input_sha256="a"*64))
        with pytest.raises(TimeoutError):
            client.before_trace(dict(class_id="c0", trace_input_sha256="a"*64))
        os.close(write_event)
        write_event = None
        events = os.read(read_event, 4096).splitlines()
        assert [json.loads(line)["event"] for line in events] == ["permit_request"]
    finally:
        for fd in (read_event, write_event, read_reply, write_reply):
            if fd is not None:
                os.close(fd)


def test_private_inputs_and_raw_returns_share_the_cumulative_guard(tmp_path, monkeypatch):
    inputs, producer = tmp_path / "inputs", tmp_path / "producer"
    inputs.mkdir()
    (inputs / "prior.json").write_bytes(b'x'*60)
    client = runner.WorkerPermitClient(-1, -1, {}, producer, deadline=time.monotonic()+1,
                                       input_directory=inputs)
    monkeypatch.setattr(runner, "MAX_DECODED_RECORD_BYTES", 100)
    with pytest.raises(runner.ProtocolError):
        client.capture_return({"class_id": "c0"}, {"value": 1})
    assert not (producer / "producer-returns.jsonl").exists()


def test_private_scope_does_not_change_with_worktree_or_code_freeze(tmp_path):
    common = tmp_path / "repository/.git"
    common.mkdir(parents=True)
    machine = dict(native_factory_isa_sha256="a"*64, native_t_instruction_sha256="b"*64,
                   data_code_distance=15, factory_retry_cap=6)
    paths = []
    for name in ("one", "two"):
        worktree = tmp_path / name
        worktree.mkdir()
        gitdir = common / "worktrees" / name
        gitdir.mkdir(parents=True)
        (gitdir / "commondir").write_text('../..\n')
        (worktree / ".git").write_text('gitdir: '+str(gitdir)+'\n')
        paths.append(runner.canonical_private_state(worktree, "c"*64, machine))
    assert paths[0] == paths[1]


@pytest.mark.parametrize("start", [True, float("nan"), float("inf")])
def test_invalid_start_cannot_create_an_attempt(tmp_path, start):
    with pytest.raises(ValueError):
        runner.run_bounded_worker([], identity={}, requests=[{"class_id":"c0","trace_input_sha256":"a"*64}],
            output_directory=tmp_path, guardian_command=[], source_bundle_bytes=0, start_monotonic=start)
    assert not list(tmp_path.iterdir())


def _audit(tmp_path, behavior, seconds=3):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, str(SCRIPT), "--audit", behavior,
        "--audit-seconds", str(seconds), "--output-dir", str(tmp_path)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=seconds+3)
    return result, json.loads(result.stdout.splitlines()[-1])


def test_dummy_success_has_post_export_guardian_confirmation(tmp_path):
    result, receipt = _audit(tmp_path, "success")
    assert result.returncode == 0, result.stderr + result.stdout
    assert receipt["status"] == "complete"
    assert receipt["physical_execution_performed"] is False
    assert receipt["counters"]["actual_trace_calls_upper_bound"] == 2
    assert receipt["counters"]["independently_verified_producer_receipts"] == 2
    assert receipt["guardian_confirmation"]["status"] == "parent_finished"
    assert json.loads((tmp_path / "parent-receipt.json").read_text())["status"] == "pending_guardian_confirmation"


def test_dummy_crash_preserves_permit_and_blocks_repeat(tmp_path):
    result, receipt = _audit(tmp_path, "crash-after-permit")
    assert result.returncode != 0
    assert receipt["status"].startswith("incomplete_")
    assert receipt["counters"]["actual_trace_calls_upper_bound"] == 1
    assert receipt["counters"]["reported_native_return_receipts"] == 0
    assert receipt["counters"]["automatic_rerun_permitted"] is False
    repeated, _ = _audit(tmp_path, "success")
    assert repeated.returncode != 0


def test_actual_execute_cannot_run_without_scope_approval():
    result = subprocess.run([sys.executable, str(SCRIPT), "--execute"], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert json.loads(result.stdout)["native_execution_default"] is False


def test_production_boundary_rejects_patched_python_objects_before_native_call():
    spec = importlib.util.spec_from_file_location("native_runner_cli_boundary_test", SCRIPT)
    cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cli
    spec.loader.exec_module(cli)
    with pytest.raises(ValueError, match="exact extension"):
        cli.validate_native_boundary({"trace": object(), "isa": object()})
