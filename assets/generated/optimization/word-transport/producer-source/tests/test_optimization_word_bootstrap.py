"""Bounded bootstrap regressions; no Native estimation or approval is used."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src/openquantum/optimization_word_native_runner.py"
SCRIPT = ROOT / "scripts/run_optimization_word_native.py"
spec = importlib.util.spec_from_file_location("bootstrap_runner_test", MODULE)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def _ready(tmp_path, *, change=None, publish=True):
    read_fd, write_fd = os.pipe()
    deadline = time.monotonic() + 1
    identity = {"scope": "dummy-bootstrap"}
    record = dict(identity=identity, worker_pid=os.getpid(), guardian_pid=os.getppid(),
                  work_deadline=deadline, parent_deadline=deadline + 1)
    path = tmp_path / "worker-pid.json"
    if publish:
        runner.atomic_record(path, dict(record, status="owned_unreaped_worker"))
    token = dict(record, event="worker_bootstrap_ready")
    if change:
        token.update(change)
    os.write(write_fd, runner.canonical(token))
    os.close(write_fd)
    return read_fd, path, identity, deadline


def test_ready_token_binds_durable_receipt_and_closes_reader(tmp_path):
    fd, path, identity, deadline = _ready(tmp_path)
    result = runner.wait_worker_readiness(fd, path, identity, deadline=deadline,
                                          parent_deadline=deadline + 1)
    assert result["status"] == "owned_unreaped_worker"
    with pytest.raises(OSError):
        os.fstat(fd)


@pytest.mark.parametrize("change", [
    {"identity": {"scope": "wrong"}}, {"worker_pid": 1}, {"guardian_pid": 1},
    {"event": "permit"}, {"work_deadline": 0}, {"parent_deadline": 0},
])
def test_changed_ready_identity_never_passes(tmp_path, change):
    fd, path, identity, deadline = _ready(tmp_path, change=change)
    with pytest.raises(runner.ProtocolError):
        runner.wait_worker_readiness(fd, path, identity, deadline=deadline,
                                      parent_deadline=deadline + 1)
    with pytest.raises(OSError):
        os.fstat(fd)


def test_eof_without_readiness_cannot_use_visible_receipt(tmp_path):
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    path = tmp_path / "worker-pid.json"
    path.write_text('{}')
    with pytest.raises(runner.ProtocolError, match="before durable"):
        runner.wait_worker_readiness(read_fd, path, {}, deadline=time.monotonic() + .2,
                                      parent_deadline=time.monotonic() + 1)


def test_absent_readiness_uses_original_deadline(tmp_path):
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            runner.wait_worker_readiness(read_fd, tmp_path / "missing", {},
                deadline=started + .05, parent_deadline=started + 1)
        assert time.monotonic() - started < .5
    finally:
        os.close(write_fd)


def test_readiness_does_not_replace_receipt_validation(tmp_path):
    fd, path, identity, deadline = _ready(tmp_path)
    path.write_text('{}')
    with pytest.raises(runner.ProtocolError, match="Durable worker receipt"):
        runner.wait_worker_readiness(fd, path, identity, deadline=deadline,
                                      parent_deadline=deadline + 1)


def test_expired_readiness_closes_reader_without_new_allowance(tmp_path):
    read_fd, write_fd = os.pipe()
    try:
        with pytest.raises(TimeoutError):
            runner.wait_worker_readiness(read_fd, tmp_path / "missing", {},
                deadline=time.monotonic() - 1, parent_deadline=time.monotonic() + 1)
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)


def _delayed_bootstrap(tmp_path, fail_publication=False):
    output = tmp_path / "output"
    output.mkdir()
    # Only dummy wrappers inject scheduling/filesystem behavior. The worker
    # enters the production helper before guardian publication can complete.
    worker = tmp_path / "worker.py"
    worker.write_text(f'''
import importlib.util,os,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("bootstrap_cli", {str(SCRIPT)!r})
cli=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cli;spec.loader.exec_module(cli)
original_load=cli._load
def load(*args,**kwargs):
    value=original_load(*args,**kwargs)
    original_wait=value.wait_worker_readiness
    def wait(*args,**kwargs):
        out=Path(os.environ["OQP_NATIVE_DUMMY_OUTPUT"])
        (out/"child-waiting").write_text("waiting")
        answer=original_wait(*args,**kwargs)
        (out/"child-passed").write_text("passed")
        return answer
    value.wait_worker_readiness=wait
    return value
cli._load=load
raise SystemExit(cli._dummy_worker("success", 2))
''')
    guardian = tmp_path / "guardian.py"
    guardian.write_text(f'''
import argparse,importlib.util,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location("bootstrap_guardian", {str(MODULE)!r})
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
p=argparse.ArgumentParser();p.add_argument("--guardian");p.add_argument("--event-fd",type=int)
p.add_argument("--worker-event-fd",type=int);p.add_argument("--reply-fd",type=int);p.add_argument("--lifeline-fd",type=int)
a=p.parse_args();original=r.atomic_record
def delayed(path,value,**kwargs):
    path=Path(path)
    if path.name=="worker-pid.json":
        until=time.monotonic()+.8
        while not (path.parent/"child-waiting").exists() and time.monotonic()<until:
            time.sleep(.005)
        assert (path.parent/"child-waiting").exists(), "child did not run before publication"
        assert not (path.parent/"child-passed").exists(), "child bypassed readiness"
        path.write_bytes(r.canonical(value))
        time.sleep(.05)
        assert not (path.parent/"child-passed").exists(), "visible undurable receipt released child"
        if {fail_publication!r}:
            raise OSError("simulated directory fsync failure")
    return original(path,value,**kwargs)
r.atomic_record=delayed
r.guardian_main(a.guardian,a.event_fd,a.reply_fd,a.lifeline_fd,a.worker_event_fd)
''')
    parent = tmp_path / "parent.py"
    parent.write_text(f'''
import importlib.util,json,os,sys,time
spec=importlib.util.spec_from_file_location("bootstrap_parent", {str(MODULE)!r})
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
os.environ["OQP_NATIVE_DUMMY_OUTPUT"]={str(output)!r}
answer=r.run_bounded_worker([sys.executable,{str(worker)!r}], identity={{"mode":"dummy_protocol_audit","version":1}},
 requests=[{{"class_id":"dummy-0","trace_input_sha256":"a"*64}},{{"class_id":"dummy-1","trace_input_sha256":"a"*64}}],
 output_directory={str(output)!r},guardian_command=[sys.executable,{str(guardian)!r}],source_bundle_bytes=0,
 caps=r.RunnerCaps(parent_wall_seconds=3,cleanup_seconds=1,termination_grace_seconds=.1),
 start_monotonic=time.monotonic(),mode="dummy_protocol_audit")
print(json.dumps(answer))
''')
    result = subprocess.run([sys.executable, str(parent)], cwd=ROOT,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), text=True, capture_output=True, timeout=6)
    assert result.returncode == 0, result.stderr + result.stdout
    return output, json.loads(result.stdout.splitlines()[-1])


def test_prompt_child_waits_until_receipt_is_durable(tmp_path):
    output, result = _delayed_bootstrap(tmp_path)
    assert (output / "child-waiting").exists() and (output / "child-passed").exists()
    assert result["status"] == "complete"
    assert result["physical_execution_performed"] is False
    assert result["counters"]["actual_trace_calls_upper_bound"] == 2


def test_visible_receipt_and_failed_fsync_never_grant_calls(tmp_path):
    output, result = _delayed_bootstrap(tmp_path, fail_publication=True)
    assert (output / "child-waiting").exists() and not (output / "child-passed").exists()
    assert result["status"].startswith("incomplete_")
    assert result["counters"]["actual_trace_calls_upper_bound"] == 0
    assert result["counters"]["automatic_rerun_permitted"] is False
    claim = json.loads((output / "attempt-claim.json").read_text())
    assert claim
