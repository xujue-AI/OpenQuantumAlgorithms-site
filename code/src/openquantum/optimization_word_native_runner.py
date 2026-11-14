"""Bounded stdlib process protocol for the one fixed-machine Native pass.

This module imports no QDK and performs no estimation. A guardian owns the
worker process group. The controller durably charges a permit before a worker
may enter Trace.estimate; an interrupted permit remains charged because the
entry gap cannot be reconstructed afterward. Actual execution requires a
separate, exact, one-use approval receipt. Dummy protocol workers are used by
the owned lifecycle tests only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import errno
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

MAX_MESSAGE_BYTES = 4096
MAX_JOURNAL_BYTES = 1024 * 1024
MAX_DECODED_RECORD_BYTES = 64 * 1024 * 1024
MAX_COMBINED_BUNDLE_BYTES = 16 * 1024 * 1024
NATIVE_EXECUTION_DEFAULT = False


class RunnerError(RuntimeError):
    pass


class ExecutionDisabled(RunnerError):
    pass


class ProtocolError(RunnerError):
    pass


@dataclass(frozen=True)
class RunnerCaps:
    trace_calls: int = 54
    parent_wall_seconds: float = 60.0
    cleanup_seconds: float = 5.0
    termination_grace_seconds: float = .25
    decoded_record_bytes: int = MAX_DECODED_RECORD_BYTES
    combined_bundle_bytes: int = MAX_COMBINED_BUNDLE_BYTES

    def __post_init__(self):
        for name, maximum in (("trace_calls", 54),
                              ("decoded_record_bytes", MAX_DECODED_RECORD_BYTES),
                              ("combined_bundle_bytes", MAX_COMBINED_BUNDLE_BYTES)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} is outside the fixed bounded domain")
        for name in ("parent_wall_seconds", "cleanup_seconds", "termination_grace_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite duration")
        if self.parent_wall_seconds > 60 or self.cleanup_seconds >= self.parent_wall_seconds:
            raise ValueError("The complete parent wall cap is at most 60 seconds with an interior cleanup reserve")
        if self.termination_grace_seconds >= self.cleanup_seconds / 2:
            raise ValueError("TERM grace must leave time for KILL, export and guardian reap")


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical_private_state(root, candidate_recipe_sha256, fixed_machine):
    """Use one Git-common-dir claim across worktrees/code/output revisions."""
    root = Path(root).resolve()
    git = root / ".git"
    if git.is_dir():
        common = git.resolve()
    else:
        pointer = bounded_regular(git, 4096).decode().strip()
        if not pointer.startswith("gitdir: "):
            raise RunnerError("Managed worktree lacks a valid Git directory pointer")
        directory = Path(pointer[8:])
        directory = directory if directory.is_absolute() else root / directory
        common_pointer = directory / "commondir"
        common = ((directory / bounded_regular(common_pointer, 4096).decode().strip()).resolve()
                  if common_pointer.exists() else directory.resolve())
    if not common.is_dir():
        raise RunnerError("Git common directory is unavailable for canonical cumulative state")
    scope = dict(candidate_recipe_sha256=candidate_recipe_sha256,
                 native_factory_isa_sha256=fixed_machine["native_factory_isa_sha256"],
                 native_t_instruction_sha256=fixed_machine["native_t_instruction_sha256"],
                 data_code_distance=fixed_machine["data_code_distance"],
                 factory_retry_cap=fixed_machine["factory_retry_cap"])
    return common / "oqp-private-native-attempts" / sha256(canonical(scope))


def bounded_regular(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise RunnerError("A bounded record must be a regular file within its byte cap")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise RunnerError("A record grew beyond its byte cap while being read")
    return raw


def tree_bytes(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            raise RunnerError("Owned output trees may not contain symlinks")
        if item.is_file():
            total += item.stat().st_size
    return total


def atomic_record(path, value, *, maximum=MAX_JOURNAL_BYTES):
    """Durably replace a small receipt, without following a destination link."""
    path = Path(path)
    raw = canonical(value)
    if len(raw) > maximum or path.is_symlink():
        raise RunnerError("Structured receipt exceeded its guard")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(bytes=len(raw), sha256=sha256(raw))


class AttemptJournal:
    """One-use attempt claim and conservative lifetime Trace-call budget."""

    def __init__(self, directory, identity, request_ids, caps=RunnerCaps()):
        self.directory = Path(directory).resolve()
        self.identity = identity
        self.request_ids = tuple(request_ids)
        if (not self.request_ids or len(set(self.request_ids)) != len(self.request_ids)
                or len(self.request_ids) > caps.trace_calls
                or any(not isinstance(item, str) or not item for item in self.request_ids)):
            raise ValueError("The fixed request set must be distinct and within the call cap")
        self.caps = caps
        self.claim_path = self.directory / "attempt-claim.json"
        self.path = self.directory / "attempt-journal.jsonl"
        self.records = self._read()
        self.permits = {r["request_id"]: r for r in self.records if r["event"] == "permit_granted"}
        self.starts = {r["request_id"] for r in self.records if r["event"] == "native_entry_reported"}
        self.returns = {r["request_id"] for r in self.records if r["event"] == "native_return_recorded"}

    def _read(self):
        if not self.path.exists():
            return []
        raw = bounded_regular(self.path, MAX_JOURNAL_BYTES)
        if raw and not raw.endswith(b"\n"):
            raise ProtocolError("An interrupted/truncated journal cannot authorize another call")
        rows = []
        granted, started, returned = set(), set(), set()
        for index, line in enumerate(raw.splitlines()):
            if not line or len(line) > MAX_MESSAGE_BYTES:
                raise ProtocolError("Malformed journal record")
            row = json.loads(line)
            if type(row.get("sequence")) is not int or row["sequence"] != index or row.get("identity") != self.identity:
                raise ProtocolError("Journal identity/order differs from its claimed attempt")
            event = row.get("event")
            request = row.get("request_id")
            if event == "permit_granted":
                if request not in self.request_ids or request in granted or len(granted) >= self.caps.trace_calls:
                    raise ProtocolError("Journal duplicated or exceeded the fixed permit budget")
                granted.add(request)
                if type(row.get("counter_upper_bound")) is not int or row["counter_upper_bound"] != len(granted):
                    raise ProtocolError("Journal permit counter is not the exact cumulative bound")
            elif event == "native_entry_reported":
                if request not in granted or request in started:
                    raise ProtocolError("Native entry lacks one unique durable permit")
                started.add(request)
            elif event == "native_return_recorded":
                if request not in started or request in returned:
                    raise ProtocolError("Native return lacks one unique reported entry")
                returned.add(request)
            elif event not in {"attempt_started", "attempt_finished", "worker_started", "guardian_ready"}:
                raise ProtocolError("Unknown journal event")
            rows.append(row)
        return rows

    def claim(self, start_monotonic):
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.records or self.claim_path.exists():
            raise RunnerError("This one-use attempt is already claimed; no automatic rerun or counter reset is permitted")
        raw = canonical(dict(identity=self.identity, request_ids=list(self.request_ids),
                             caps=asdict(self.caps), status="claimed_before_worker_launch"))
        descriptor = os.open(self.claim_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.append("attempt_started", start_monotonic=start_monotonic,
                    deadline_monotonic=start_monotonic + self.caps.parent_wall_seconds)

    def append(self, event, **fields):
        row = dict(sequence=len(self.records), identity=self.identity, event=event, **fields)
        raw = canonical(row)
        if len(raw) > MAX_MESSAGE_BYTES or (self.path.exists() and self.path.stat().st_size + len(raw) > MAX_JOURNAL_BYTES):
            raise RunnerError("The durable journal exceeded its byte cap")
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.records.append(row)
        return row

    def grant(self, request_id, trace_input_sha256, *, deadline):
        if (request_id not in self.request_ids or request_id in self.permits
                or len(self.permits) >= self.caps.trace_calls or time.monotonic() >= deadline):
            raise ProtocolError("A Trace permit duplicated a request, exceeded the cap or missed the work deadline")
        if (not isinstance(trace_input_sha256, str) or len(trace_input_sha256) != 64
                or set(trace_input_sha256) - set("0123456789abcdef")):
            raise ProtocolError("Every permit must bind its exact Trace input digest")
        row = self.append("permit_granted", request_id=request_id,
                          trace_input_sha256=trace_input_sha256,
                          counter_upper_bound=len(self.permits) + 1)
        self.permits[request_id] = row
        if time.monotonic() >= deadline:
            raise TimeoutError("The durable permit remains charged after its work deadline")
        return row

    def entry(self, request_id):
        if request_id not in self.permits or request_id in self.starts:
            raise ProtocolError("Native entry was not uniquely permitted")
        self.append("native_entry_reported", request_id=request_id)
        self.starts.add(request_id)

    def returned(self, request_id, receipt_sha256):
        if request_id not in self.starts or request_id in self.returns:
            raise ProtocolError("Native return was not uniquely started")
        if (not isinstance(receipt_sha256, str) or len(receipt_sha256) != 64
                or set(receipt_sha256) - set("0123456789abcdef")):
            raise ProtocolError("Native return must bind the preserved producer receipt")
        self.append("native_return_recorded", request_id=request_id, receipt_sha256=receipt_sha256)
        self.returns.add(request_id)

    def snapshot(self):
        finished = [r for r in self.records if r["event"] == "attempt_finished"]
        # Missing termination receipt after a hard kill cannot create a new
        # wall allowance. A new invocation is blocked by the one-use claim.
        charged_wall = (finished[-1]["parent_wall_seconds"] if finished and
                        finished[-1].get("status") == "complete_after_guardian_confirmation"
                        else self.caps.parent_wall_seconds if self.records or self.claim_path.exists() else 0)
        return dict(charged_trace_permits=len(self.permits),
                    reported_native_callsites=len(self.starts),
                    reported_native_return_receipts=len(self.returns),
                    unknown_entry_permits=sorted(set(self.permits) - self.starts),
                    in_flight_or_failed_requests=sorted(self.starts - self.returns),
                    actual_trace_calls_upper_bound=len(self.permits),
                    cumulative_parent_wall_seconds_charged=charged_wall,
                    actual_entry_gap="durable permit before Native entry; unknown permits remain charged",
                    automatic_rerun_permitted=False)


def send_message(fd, value):
    raw = canonical(value)
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ProtocolError("An IPC record exceeded its bounded frame size")
    offset = 0
    while offset < len(raw):
        offset += os.write(fd, raw[offset:])


class WorkerPermitClient:
    """The approved worker's before/after-Trace callbacks."""

    def __init__(self, event_fd, reply_fd, identity, result_directory, *, deadline, requests=None,
                 input_directory=None):
        self.event_fd, self.reply_fd = event_fd, reply_fd
        self.identity = identity
        self.deadline = deadline
        self.directory = Path(result_directory)
        self.raw_path = self.directory / "producer-returns.jsonl"
        self.buffer = b""
        self.captured = {}
        self.request_order = {} if requests is None else {r["class_id"]: i for i, r in enumerate(requests)}
        self.input_directory = self.directory.parent / "native-inputs" if input_directory is None else Path(input_directory)

    def before_trace(self, request):
        request_id = request["class_id"]
        trace_digest = request["trace_input_sha256"]
        input_digest = None
        if self.request_order:
            if request_id not in self.request_order:
                raise ProtocolError("Native request is outside its fixed class ordering")
            input_path = self.input_directory / f"input-{self.request_order[request_id]:02d}.json"
            if input_path.exists():
                raise ProtocolError("A Native input record was already used")
            if tree_bytes(self.input_directory) + tree_bytes(self.directory) + len(canonical(request)) > MAX_DECODED_RECORD_BYTES:
                raise RunnerError("Cumulative private Native inputs/producer references exceeded their guard")
            input_digest = atomic_record(input_path, request, maximum=MAX_DECODED_RECORD_BYTES)["sha256"]
        send_message(self.event_fd, dict(event="permit_request", identity=self.identity,
                                        request_id=request_id, trace_input_sha256=trace_digest,
                                        input_record_sha256=input_digest))
        selector = selectors.DefaultSelector()
        selector.register(self.reply_fd, selectors.EVENT_READ)
        try:
            while b"\n" not in self.buffer:
                remaining = self.deadline - time.monotonic()
                if remaining <= 0 or not selector.select(min(remaining, .2)):
                    if remaining <= 0:
                        raise TimeoutError("Trace permission missed the parent work deadline")
                    continue
                part = os.read(self.reply_fd, MAX_MESSAGE_BYTES)
                if not part:
                    raise ProtocolError("Parent permission channel closed before Native entry")
                self.buffer += part
                if len(self.buffer) > MAX_MESSAGE_BYTES:
                    raise ProtocolError("Parent permission message exceeded its cap")
            line, self.buffer = self.buffer.split(b"\n", 1)
            message = json.loads(line)
            if (message.get("event") != "permit" or message.get("identity") != self.identity
                    or message.get("request_id") != request_id or message.get("trace_input_sha256") != trace_digest):
                raise ProtocolError("Parent did not grant the exact fixed Trace request")
        finally:
            selector.close()
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Received permit is charged but too late for Native entry")
        send_message(self.event_fd, dict(event="native_entry", identity=self.identity, request_id=request_id))
        return message

    def capture_return(self, request, producer_record):
        """Persist the raw, typed producer value before checked summary export."""
        request_id = request["class_id"]
        raw = canonical(dict(class_id=request_id, producer_record=producer_record))
        existing = self.raw_path.stat().st_size if self.raw_path.exists() else 0
        if (len(raw) > MAX_DECODED_RECORD_BYTES or existing + len(raw) + tree_bytes(self.input_directory) > MAX_DECODED_RECORD_BYTES
                or request_id in self.captured or self.raw_path.is_symlink()):
            raise ProtocolError("Producer return is duplicate or outside its record guard")
        self.directory.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.raw_path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        descriptor = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.captured[request_id] = sha256(raw)

    def after_trace(self, request, producer_receipt):
        request_id = request["class_id"]
        self.capture_return(request, producer_receipt)
        send_message(self.event_fd, dict(event="native_return", identity=self.identity,
                                        request_id=request_id, receipt_sha256=self.captured[request_id]))


def block_worker_process_creation():
    """Keep the fixed Native worker inside its guardian-owned process group.

    The pinned direct Trace implementation is separately audited not to spawn
    at the native extension layer. This guard blocks Python subprocess/fork/
    spawn/session APIs; it is not an arbitrary binary sandbox.
    """
    def denied(*args, **kwargs):
        raise RunnerError("The approved direct Native worker may not spawn or detach processes")
    for name in ("fork", "forkpty", "posix_spawn", "posix_spawnp", "system", "setsid", "setpgid",
                 "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
        if hasattr(os, name):
            setattr(os, name, denied)
    for name in ("Popen", "run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"):
        setattr(subprocess, name, denied)


def _group_members(pgid, timeout):
    if timeout <= 0:
        return None
    try:
        result = subprocess.run(["/bin/ps", "-axo", "pid=,ppid=,pgid=,stat=,lstart="],
                                capture_output=True, text=True, timeout=min(timeout, .25), check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split(None, 4)
        if len(fields) != 5:
            continue
        try:
            pid, ppid, group = map(int, fields[:3])
        except ValueError:
            continue
        if group == pgid:
            rows.append(dict(pid=pid, ppid=ppid, pgid=group, state=fields[3], start_signature=fields[4]))
    return rows


def cleanup_process_group(pgid, *, deadline, term_grace=.25, leader_signature=None, leader_owned=False):
    """Stop live group members even after the process-group leader has exited."""
    errors, signals_sent = [], []
    members = _group_members(pgid, deadline - time.monotonic())
    if members is None:
        # An unreaped Popen child reserves its PID. Failure to enumerate the
        # group must not leave that known owned worker running.
        if leader_owned:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                    signals_sent.append(signal.Signals(sig).name)
                except ProcessLookupError:
                    break
                except OSError as error:
                    errors.append(dict(signal=signal.Signals(sig).name, errno=error.errno))
        return dict(complete=False, reason="process_inventory_unavailable", signal_errors=[], live_members=None)
    leader = next((r for r in members if r["pid"] == pgid), None)
    if leader is not None and leader_signature is not None and leader["start_signature"] != leader_signature:
        return dict(complete=False, reason="pid_reuse_detected", signal_errors=[], live_members=members)
    for sig, grace in ((signal.SIGTERM, term_grace), (signal.SIGKILL, 0)):
        live = [r for r in members if not r["state"].startswith("Z")]
        if not live:
            break
        try:
            os.killpg(pgid, sig)
            signals_sent.append(signal.Signals(sig).name)
        except ProcessLookupError:
            members = []
            break
        except OSError as error:
            errors.append(dict(signal=signal.Signals(sig).name, errno=error.errno))
        until = min(deadline, time.monotonic() + grace)
        while time.monotonic() < until:
            time.sleep(min(.02, until - time.monotonic()))
        members = _group_members(pgid, deadline - time.monotonic())
        if members is None:
            break
    while members is not None and time.monotonic() < deadline:
        live = [r for r in members if not r["state"].startswith("Z")]
        if not live:
            break
        time.sleep(min(.02, max(0, deadline - time.monotonic())))
        members = _group_members(pgid, deadline - time.monotonic())
    live = None if members is None else [r for r in members if not r["state"].startswith("Z")]
    return dict(complete=live == [], reason="no_live_group_members" if live == [] else "cleanup_incomplete",
                signals_sent=signals_sent, signal_errors=errors, live_members=live,
                zombies=[] if members is None else [r["pid"] for r in members if r["state"].startswith("Z")],
                scope="guardian-owned process group; approved worker process creation is blocked")


def _worker_exited_unreaped(worker):
    """Observe completion without freeing the known leader PID prematurely."""
    if hasattr(os, "waitid"):
        try:
            result = os.waitid(os.P_PID, worker.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
            return result is not None
        except ChildProcessError:
            return True
    members = _group_members(worker.pid, .15)
    if members is None:
        return False
    leader = next((row for row in members if row["pid"] == worker.pid), None)
    return leader is None or leader["state"].startswith("Z")


def wait_worker_readiness(readiness_fd, receipt_path, identity, *, deadline,
                          parent_deadline, worker_pid=None, guardian_pid=None):
    """Await durable guardian publication inside the existing work deadline.

    Only the guardian owns the writer. Receiving readiness does not grant a
    Trace permit; the normal durable parent protocol remains mandatory.
    """
    worker_pid = os.getpid() if worker_pid is None else worker_pid
    guardian_pid = os.getppid() if guardian_pid is None else guardian_pid
    if (type(readiness_fd) is not int or readiness_fd <= 2
            or type(worker_pid) is not int or worker_pid < 1
            or type(guardian_pid) is not int or guardian_pid < 1
            or isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline)
            or isinstance(parent_deadline, bool) or not isinstance(parent_deadline, (int, float))
            or not math.isfinite(parent_deadline) or deadline > parent_deadline):
        if type(readiness_fd) is int and readiness_fd > 2:
            os.close(readiness_fd)
        raise ProtocolError("Invalid bounded worker readiness identity/deadline")
    selector = selectors.DefaultSelector()
    try:
        os.set_blocking(readiness_fd, False)
        selector.register(readiness_fd, selectors.EVENT_READ)
        buffer = b""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Guardian readiness exceeded the original worker deadline")
            if not selector.select(remaining):
                continue
            part = os.read(readiness_fd, MAX_MESSAGE_BYTES + 1 - len(buffer))
            if not part:
                raise ProtocolError("Guardian readiness ended before durable receipt publication")
            buffer += part
            if len(buffer) > MAX_MESSAGE_BYTES:
                raise ProtocolError("Guardian readiness record exceeded its bound")
            if b"\n" not in buffer:
                continue
            line, trailing = buffer.split(b"\n", 1)
            if trailing:
                raise ProtocolError("Guardian readiness contains extra records")
            expected = dict(event="worker_bootstrap_ready", identity=identity,
                worker_pid=worker_pid, guardian_pid=guardian_pid,
                work_deadline=deadline, parent_deadline=parent_deadline)
            if json.loads(line) != expected:
                raise ProtocolError("Guardian readiness identity or deadline changed")
            if time.monotonic() >= deadline:
                raise TimeoutError("Guardian readiness arrived after the original worker deadline")
            receipt = json.loads(bounded_regular(receipt_path, MAX_JOURNAL_BYTES))
            expected_receipt = {key: value for key, value in expected.items() if key != "event"}
            expected_receipt["status"] = "owned_unreaped_worker"
            if receipt != expected_receipt:
                raise ProtocolError("Durable worker receipt differs from guardian readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker receipt verification exceeded the original deadline")
            return receipt
    finally:
        selector.close()
        os.close(readiness_fd)


def guardian_main(config_path, event_fd, reply_fd, lifeline_fd, worker_event_fd):
    """Launch/own the worker and enforce parent death plus an absolute deadline."""
    config = json.loads(bounded_regular(config_path, MAX_JOURNAL_BYTES))
    deadline, work_deadline = config["deadline"], config["work_deadline"]
    identity, worker = config["identity"], None
    readiness_reader, readiness_writer = None, None
    os.set_blocking(lifeline_fd, False)
    os.set_blocking(event_fd, False)
    selector = selectors.DefaultSelector()
    selector.register(lifeline_fd, selectors.EVENT_READ)
    parent_finished, parent_cancelled, cleanup = False, False, None
    stopped = False
    def terminate(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    def hard_deadline(signum, frame):
        if worker is not None:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if os.getppid() == config["parent_pid"]:
            try:
                os.kill(config["parent_pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        os._exit(124)
    signal.signal(signal.SIGALRM, hard_deadline)
    signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
    send_message(event_fd, dict(event="guardian_ready", identity=identity, guardian_pid=os.getpid()))
    try:
        readiness_reader, readiness_writer = os.pipe()
        environment = dict(os.environ, OQP_NATIVE_EVENT_FD=str(worker_event_fd), OQP_NATIVE_REPLY_FD=str(reply_fd),
                           OQP_NATIVE_READINESS_FD=str(readiness_reader),
                           OQP_NATIVE_IDENTITY_JSON=json.dumps(identity, sort_keys=True),
                           OQP_NATIVE_WORK_DEADLINE=str(work_deadline),
                           OQP_NATIVE_PRODUCER_DIRECTORY=config["producer_directory"])
        worker = subprocess.Popen(config["worker_command"], cwd=config["cwd"], env=environment,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=True, pass_fds=(worker_event_fd, reply_fd, readiness_reader))
        os.close(readiness_reader)
        readiness_reader = None
        os.close(reply_fd)
        os.close(worker_event_fd)
        atomic_record(config["worker_pid_receipt"], dict(worker_pid=worker.pid, guardian_pid=os.getpid(),
                      identity=identity, status="owned_unreaped_worker",
                      work_deadline=work_deadline, parent_deadline=deadline))
        if time.monotonic() >= work_deadline:
            raise TimeoutError("Durable worker publication exceeded the original work deadline")
        send_message(readiness_writer, dict(event="worker_bootstrap_ready", identity=identity,
            worker_pid=worker.pid, guardian_pid=os.getpid(), work_deadline=work_deadline,
            parent_deadline=deadline))
        os.close(readiness_writer)
        readiness_writer = None
        send_message(event_fd, dict(event="worker_started", identity=identity, worker_pid=worker.pid,
                                   leader_signature=None))
        initial = _group_members(worker.pid, min(.25, max(0, deadline - time.monotonic())))
        leader = next((r for r in initial or [] if r["pid"] == worker.pid), None)
        signature = leader["start_signature"] if leader else None
        status = "worker_running"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "parent_deadline_exhausted"
                break
            if selector.select(min(.02, remaining)):
                data = os.read(lifeline_fd, 16)
                if not data:
                    status = "parent_lost"
                    break
                if b"D" in data:
                    parent_finished = True
                    status = "parent_finished"
                    break
                if b"C" in data:
                    status = "parent_cancelled"
                    parent_cancelled = True
            if stopped:
                status = "guardian_cancelled"
                break
            observed_exit = _worker_exited_unreaped(worker) if cleanup is None else False
            if cleanup is None and (parent_cancelled or observed_exit or time.monotonic() >= work_deadline):
                status = ("parent_cancelled" if parent_cancelled else
                          "worker_exited" if observed_exit else "worker_work_deadline")
                cleanup = cleanup_process_group(worker.pid, deadline=min(deadline - .5, time.monotonic() + 1.5),
                                                term_grace=config["term_grace"], leader_signature=signature,
                                                leader_owned=worker.returncode is None)
                try:
                    worker.wait(timeout=max(.01, min(.25, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    pass
                # Retain a durable NONCOMPLETE status before waiting for the
                # controller's final fsync/export. The hard alarm deliberately
                # performs no I/O, so this receipt survives a blocked parent.
                atomic_record(config["guardian_receipt"], dict(
                    status=status + "_waiting_for_parent_exports", identity=identity,
                    deadline_monotonic=deadline, cleanup=cleanup,
                    worker_returncode=worker.returncode, parent_finished=False,
                    wall_end_monotonic=time.monotonic()))
                send_message(event_fd, dict(event="worker_stopped", identity=identity,
                                           worker_returncode=worker.poll(), status=status, cleanup=cleanup))
        if cleanup is None:
            cleanup = cleanup_process_group(worker.pid, deadline=max(time.monotonic() + .05, deadline - .2),
                                            term_grace=config["term_grace"], leader_signature=signature,
                                            leader_owned=worker.returncode is None)
        try:
            worker.wait(timeout=max(.01, min(.25, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            cleanup["complete"] = False
        # A blocked controller must not outlive the full wall envelope. The
        # lifeline is inherited by neither worker nor its permitted children.
        if (not parent_finished and time.monotonic() >= deadline
                and os.getppid() == config["parent_pid"]):
            try:
                os.kill(config["parent_pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
        descriptor = atomic_record(config["guardian_receipt"], dict(status=status, identity=identity,
                     deadline_monotonic=deadline, cleanup=cleanup,
                     worker_returncode=worker.poll(), parent_finished=parent_finished,
                     wall_end_monotonic=time.monotonic()))
        if parent_finished and cleanup.get("complete"):
            parent_raw = bounded_regular(Path(config["guardian_receipt"]).parent / "parent-receipt.json", MAX_JOURNAL_BYTES)
            parent = json.loads(parent_raw)
            seal = dict(status="complete_after_all_parent_exports", identity=identity,
                parent_receipt_sha256=sha256(parent_raw), guardian_receipt_sha256=descriptor["sha256"],
                verified_result=parent.get("verified_result"),
                raw_producer_export=parent.get("raw_producer_export"),
                native_input_export=parent.get("native_input_export"),
                cleanup_complete=True, deadline_monotonic=deadline,
                confirmed_monotonic=time.monotonic(),
                scope="post-export guardian seal; absolute hard timer remains active through seal fsync")
            atomic_record(Path(config["guardian_receipt"]).parent / "completion-seal.json", seal)
            if time.monotonic() > deadline:
                hard_deadline(signal.SIGALRM, None)
    except BaseException as error:
        if worker is not None:
            cleanup = cleanup_process_group(worker.pid, deadline=min(deadline, time.monotonic() + .5),
                                            term_grace=.05, leader_owned=worker.returncode is None)
            try:
                worker.wait(timeout=max(.01, min(.2, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                cleanup["complete"] = False
        try:
            send_message(event_fd, dict(event="guardian_failed", identity=identity, error_type=type(error).__name__))
        except OSError:
            pass
        atomic_record(config["guardian_receipt"], dict(status="guardian_failed", error_type=type(error).__name__, cleanup=cleanup))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for fd in (readiness_reader, readiness_writer):
            if fd is not None:
                os.close(fd)
        selector.close()


def validate_approval(approval, freeze, output):
    if (not isinstance(approval, dict) or approval.get("kind") != "coordinator_scoped_native_execution"
            or approval.get("execution_authorized") is not True):
        raise ExecutionDisabled("Specific coordinator approval for the exact frozen Native pass is absent")
    if (approval.get("freeze_sha256") != sha256(canonical(freeze))
            or approval.get("candidate_recipe_sha256") != freeze["candidate_recipe_sha256"]
            or approval.get("source_sha256") != freeze["source_sha256"]
            or approval.get("fixed_machine") != freeze["fixed_machine"]
            or approval.get("caps") != freeze["caps"]
            or Path(approval.get("output_directory", "")).resolve() != Path(output).resolve()
            or Path(freeze.get("output_directory", "")).resolve() != Path(output).resolve()
            or approval.get("automatic_rerun_authorized") is not False):
        raise ExecutionDisabled("Approval does not bind the exact source, machine, caps and one-use output")


def run_bounded_worker(worker_command, *, identity, requests, output_directory,
                       guardian_command, source_bundle_bytes, caps=RunnerCaps(),
                       start_monotonic=None, mode="dummy_protocol_audit", state_directory=None,
                       producer_directory=None, input_validator=None):
    """Bound a protocol worker. Production approval is checked by the CLI.

    ``mode`` labels test receipts; this generic subprocess helper contains no
    estimator. The production CLI fixes both commands and validates approval
    before calling it. It never exposes an arbitrary worker-command flag.
    """
    started = time.monotonic() if start_monotonic is None else start_monotonic
    if (isinstance(started, bool) or not isinstance(started, (int, float))
            or not math.isfinite(started) or started > time.monotonic()):
        raise ValueError("Parent start must be a finite captured monotonic time, not a future allowance")
    deadline = started + caps.parent_wall_seconds
    work_deadline = deadline - caps.cleanup_seconds
    output = Path(output_directory).resolve()
    if (isinstance(source_bundle_bytes, bool) or not isinstance(source_bundle_bytes, int)
            or source_bundle_bytes < 0 or source_bundle_bytes >= caps.combined_bundle_bytes):
        raise ValueError("Immutable source bundle bytes must leave bounded receipt capacity")
    if source_bundle_bytes + tree_bytes(output) >= caps.combined_bundle_bytes:
        raise RunnerError("The combined immutable source and output bundle already exceeds its cap")
    if time.monotonic() >= work_deadline:
        raise TimeoutError("Imports/input reconstruction already exhausted the work deadline")
    request_map = {r["class_id"]: r["trace_input_sha256"] for r in requests}
    if len(request_map) != len(requests):
        raise ValueError("Every fixed class request must be distinct")
    state = output if state_directory is None else Path(state_directory).resolve()
    producer = state / "private-producer" if producer_directory is None else Path(producer_directory).resolve()
    input_directory = state / "native-inputs"
    if mode == "approved_native_execution" and (state.is_relative_to(output) or producer.is_relative_to(output)):
        raise RunnerError("Approved cumulative state and producer reference must remain outside the export bundle")
    journal = AttemptJournal(state, identity, request_map, caps)
    journal.claim(started)
    event_read, event_write = os.pipe()
    worker_event_read, worker_event_write = os.pipe()
    reply_read, reply_write = os.pipe()
    lifeline_read, lifeline_write = os.pipe()
    config_path = output / "guardian-config.json"
    guardian_receipt = output / "guardian-receipt.json"
    atomic_record(config_path, dict(identity=identity, worker_command=list(worker_command), cwd=str(Path.cwd()),
        parent_pid=os.getpid(), deadline=deadline, work_deadline=work_deadline,
        term_grace=min(caps.termination_grace_seconds, caps.cleanup_seconds / 4),
        guardian_receipt=str(guardian_receipt), producer_directory=str(producer),
        worker_pid_receipt=str(output / "worker-pid.json")))
    guardian = None
    selector = selectors.DefaultSelector()
    os.set_blocking(event_read, False)
    selector.register(event_read, selectors.EVENT_READ)
    os.set_blocking(worker_event_read, False)
    selector.register(worker_event_read, selectors.EVENT_READ)
    buffers, pending = {event_read: b"", worker_event_read: b""}, None
    guardian_ready, worker_pid, worker_signature = False, None, None
    worker_returncode, cleanup, done = None, None, None
    failure, error_type = None, None
    try:
        guardian = subprocess.Popen([*guardian_command, "--guardian", str(config_path),
            "--event-fd", str(event_write), "--worker-event-fd", str(worker_event_write),
            "--reply-fd", str(reply_read), "--lifeline-fd", str(lifeline_read)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, pass_fds=(event_write, worker_event_write, reply_read, lifeline_read))
        for fd in (event_write, worker_event_write, reply_read, lifeline_read):
            os.close(fd)
        def grant(message):
            request_id = message.get("request_id")
            if request_map.get(request_id) != message.get("trace_input_sha256"):
                raise ProtocolError("Worker requested an unbound class or Trace input")
            if input_validator is not None:
                index = list(request_map).index(request_id)
                raw_input = bounded_regular(input_directory / f"input-{index:02d}.json", caps.decoded_record_bytes)
                if sha256(raw_input) != message.get("input_record_sha256"):
                    raise ProtocolError("Worker control frame differs from its canonical full Native input")
                input_validator(requests[index], json.loads(raw_input))
            journal.grant(request_id, message["trace_input_sha256"], deadline=work_deadline)
            send_message(reply_write, dict(event="permit", identity=identity, request_id=request_id,
                                         trace_input_sha256=message["trace_input_sha256"]))
        while True:
            if time.monotonic() >= deadline - .5:
                raise TimeoutError("The complete parent envelope reached its final export/cleanup reserve")
            if cleanup is not None and not selector.select(0):
                if any(buffers.values()):
                    raise ProtocolError("Worker exited with a truncated IPC record")
                break
            ready = selector.select(min(.05, max(.001, deadline - time.monotonic())))
            if not ready:
                if guardian.poll() is not None:
                    raise ProtocolError("Guardian exited without a complete worker cleanup receipt")
                continue
            channel_fd = ready[0][0].fd
            data = os.read(channel_fd, MAX_MESSAGE_BYTES * 2)
            if not data:
                selector.unregister(channel_fd)
                if channel_fd == event_read and guardian.poll() is not None and cleanup is None:
                    raise ProtocolError("Guardian IPC closed before worker cleanup")
                continue
            buffers[channel_fd] += data
            if len(buffers[channel_fd]) > MAX_MESSAGE_BYTES * 4:
                raise ProtocolError("Worker IPC buffer exceeded its bounded protocol")
            while b"\n" in buffers[channel_fd]:
                line, buffers[channel_fd] = buffers[channel_fd].split(b"\n", 1)
                if not line or len(line) > MAX_MESSAGE_BYTES:
                    raise ProtocolError("Worker sent an invalid bounded message")
                message = json.loads(line)
                if message.get("identity") != identity:
                    raise ProtocolError("Worker/guardian message changed its frozen identity")
                event = message.get("event")
                allowed = ({"guardian_ready", "worker_started", "worker_stopped", "guardian_failed"}
                           if channel_fd == event_read else
                           {"permit_request", "native_entry", "native_return", "worker_done", "worker_failed"})
                if event not in allowed:
                    raise ProtocolError("An IPC sender used a different process role")
                if event == "guardian_ready":
                    if guardian_ready:
                        raise ProtocolError("Guardian readiness was duplicated")
                    guardian_ready = True
                    journal.append("guardian_ready", guardian_pid=message["guardian_pid"])
                elif event == "worker_started":
                    if not guardian_ready or worker_pid is not None:
                        raise ProtocolError("Worker start lacked unique guardian readiness")
                    worker_pid, worker_signature = message["worker_pid"], message.get("leader_signature")
                    journal.append("worker_started", worker_pid=worker_pid)
                    if pending is not None:
                        grant(pending)
                        pending = None
                elif event == "permit_request":
                    if pending is not None:
                        raise ProtocolError("Worker duplicated an initial pending permit")
                    if not guardian_ready or worker_pid is None:
                        pending = message
                    else:
                        grant(message)
                elif event == "native_entry":
                    journal.entry(message["request_id"])
                elif event == "native_return":
                    journal.returned(message["request_id"], message["receipt_sha256"])
                elif event == "worker_done":
                    if done is not None:
                        raise ProtocolError("Worker completion was duplicated")
                    done = message
                elif event == "worker_failed":
                    failure, error_type = "worker_failed", message.get("error_type", "WorkerError")
                elif event == "worker_stopped":
                    worker_returncode = message.get("worker_returncode")
                    cleanup = message["cleanup"]
                    if message["status"] == "worker_work_deadline":
                        failure, error_type = "timeout", "TimeoutError"
                elif event == "guardian_failed":
                    raise ProtocolError("Guardian failed before verified descendant cleanup")
                else:
                    raise ProtocolError("Worker used an unknown protocol event")
            if time.monotonic() >= work_deadline and cleanup is None:
                # The guardian, not a Python worker alarm, kills an in-flight
                # Native extension at this cutoff.
                failure, error_type = "timeout", "TimeoutError"
        if not cleanup.get("complete"):
            failure, error_type = "cleanup_incomplete", "CleanupError"
        if worker_returncode != 0 or done is None:
            failure = failure or "worker_incomplete"
            error_type = error_type or "WorkerIncomplete"
        if done is not None and failure is None:
            result_path = output / done["result_file"]
            if not result_path.resolve().is_relative_to(output):
                raise ProtocolError("Worker result path escaped its owned output")
            raw = bounded_regular(result_path, caps.decoded_record_bytes)
            if sha256(raw) != done["result_sha256"] or set(journal.returns) != set(request_map):
                raise ProtocolError("Complete producer result or class-call coverage changed")
            raw_returns = bounded_regular(producer / "producer-returns.jsonl", caps.decoded_record_bytes)
            observed = {}
            if not raw_returns.endswith(b"\n"):
                raise ProtocolError("Raw producer receipt transport was interrupted")
            for line in raw_returns.splitlines(keepends=True):
                record = json.loads(line)
                key = record["class_id"]
                if key in observed or key not in request_map:
                    raise ProtocolError("Raw producer transport duplicated or invented a class")
                observed[key] = sha256(line)
            expected = {r["request_id"]: r["receipt_sha256"] for r in journal.records if r["event"] == "native_return_recorded"}
            if observed != expected:
                raise ProtocolError("Reported Native returns differ from preserved raw producer receipts")
        if source_bundle_bytes + tree_bytes(output) > caps.combined_bundle_bytes:
            raise RunnerError("Native output exceeded the combined immutable-source byte cap")
    except BaseException as error:
        failure = "cancelled" if isinstance(error, (KeyboardInterrupt, SystemExit)) else failure or "parent_failed"
        error_type = type(error).__name__
    finally:
        selector.close()
        if worker_pid is None and (output / "worker-pid.json").exists():
            try:
                fallback = json.loads(bounded_regular(output / "worker-pid.json", MAX_JOURNAL_BYTES))
                if fallback["identity"] == identity:
                    worker_pid = fallback["worker_pid"]
            except (ValueError, RunnerError):
                pass
        try:
            if failure:
                os.write(lifeline_write, b"C")
        except OSError:
            pass
        if worker_pid is not None and (cleanup is None or not cleanup.get("complete")):
            cleanup = cleanup_process_group(worker_pid,
                deadline=min(deadline - .2, time.monotonic() + max(.05, caps.cleanup_seconds / 3)),
                term_grace=min(caps.termination_grace_seconds, caps.cleanup_seconds / 4),
                leader_signature=worker_signature)
    export_error = None
    receipt = None
    try:
        elapsed = time.monotonic() - started
        if elapsed > caps.parent_wall_seconds:
            failure, error_type = "parent_wall_cap_exceeded", "TimeoutError"
        if cleanup is None or not cleanup.get("complete"):
            failure, error_type = "cleanup_incomplete", error_type or "CleanupError"
        status = "pending_guardian_confirmation" if failure is None else "incomplete_" + failure
        journal.append("attempt_finished", status=status, parent_wall_seconds=elapsed,
                       error_type=error_type, cleanup_complete=bool(cleanup and cleanup.get("complete")))
        verified_raw = {}
        private_reference = None
        if (producer / "producer-returns.jsonl").exists():
            raw = bounded_regular(producer / "producer-returns.jsonl", caps.decoded_record_bytes)
            private_reference = dict(bytes=len(raw), sha256=sha256(raw),
                storage_scope="private canonical Git-common-dir state; outside source/export bundle")
            for line in raw.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    continue
                record = json.loads(line)
                key = record["class_id"]
                if key not in journal.permits or key in verified_raw:
                    raise ProtocolError("Preserved producer receipt lacks one unique durable permit")
                verified_raw[key] = sha256(line)
        counters = journal.snapshot()
        counters["independently_verified_producer_receipts"] = len(verified_raw)
        receipt = dict(schema_version=1, status=status, identity=identity, mode=mode, caps=asdict(caps),
            parent_wall_seconds=elapsed,
            parent_wall_scope="CLI entry through imports/reconstruction/worker/scientific export; post-export guardian confirmation is required",
            worker_returncode=worker_returncode, cleanup=cleanup, error_type=error_type,
            counters=counters, native_execution_attempted=mode == "approved_native_execution",
            physical_result_completed=False,
            physical_execution_performed=(mode == "approved_native_execution" and bool(verified_raw)),
            factory_search_calls=0, pipeline_search_calls=0, synthesis_calls=0, optimizer_calls=0,
            combined_bundle_bytes_before_final_receipt=source_bundle_bytes + tree_bytes(output))
        if private_reference is not None:
            receipt["private_producer_reference"] = private_reference
            packed = gzip.compress(raw, mtime=0)
            if source_bundle_bytes + tree_bytes(output) + len(packed) > caps.combined_bundle_bytes:
                raise RunnerError("Lossless raw producer export exceeded the complete bundle cap")
            destination = output / "producer-returns.jsonl.gz"
            with destination.open("xb") as stream:
                stream.write(packed)
                stream.flush()
                os.fsync(stream.fileno())
            receipt["raw_producer_export"] = dict(file=destination.name, bytes=len(packed), sha256=sha256(packed),
                decoded_bytes=len(raw), decoded_sha256=sha256(raw))
        if done is not None:
            result_path = output / done["result_file"]
            raw_result = bounded_regular(result_path, caps.decoded_record_bytes)
            receipt["verified_result"] = dict(file=done["result_file"], sha256=sha256(raw_result), bytes=len(raw_result))
        if input_validator is not None:
            native_inputs = []
            for index in range(len(requests)):
                path = input_directory / f"input-{index:02d}.json"
                if path.exists():
                    native_inputs.append(json.loads(bounded_regular(path, caps.decoded_record_bytes)))
            raw_inputs = canonical(native_inputs)
            packed_inputs = gzip.compress(raw_inputs, mtime=0)
            if source_bundle_bytes + tree_bytes(output) + len(packed_inputs) > caps.combined_bundle_bytes:
                raise RunnerError("Full Native-input export exceeded the combined cap")
            destination = output / "native-inputs.json.gz"
            with destination.open("xb") as stream:
                stream.write(packed_inputs)
                stream.flush()
                os.fsync(stream.fileno())
            receipt["native_input_export"] = dict(file=destination.name, bytes=len(packed_inputs),
                sha256=sha256(packed_inputs), decoded_bytes=len(raw_inputs), decoded_sha256=sha256(raw_inputs))
        # Export only a bounded copy of authoritative private journal/claim.
        for name in ("attempt-claim.json", "attempt-journal.jsonl"):
            raw = bounded_regular(state / name, MAX_JOURNAL_BYTES)
            if state != output:
                destination = output / name
                if source_bundle_bytes + tree_bytes(output) + len(raw) > caps.combined_bundle_bytes:
                    raise RunnerError("Journal export exceeded the complete combined byte cap")
                with destination.open("xb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
        if source_bundle_bytes + tree_bytes(output) + len(canonical(receipt)) > caps.combined_bundle_bytes:
            raise RunnerError("Parent receipt would exceed the complete combined byte cap")
        atomic_record(output / "parent-receipt.json", receipt)
        if source_bundle_bytes + tree_bytes(output) > caps.combined_bundle_bytes:
            raise RunnerError("Final receipt exceeded the complete combined byte cap")
        if time.monotonic() > deadline:
            raise TimeoutError("Final receipt/export exceeded the complete parent wall cap")
    except BaseException as error:
        export_error = error
    finally:
        # The guardian remains independent and alive until every scientific
        # artifact, journal and provisional receipt has been written. A lone
        # provisional parent receipt is never a completion certificate.
        try:
            os.write(lifeline_write, b"C" if export_error else b"D")
            if export_error:
                os.close(lifeline_write)
        except OSError:
            pass
        if guardian is not None:
            try:
                guardian.wait(timeout=max(.01, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(guardian.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    guardian.wait(timeout=.05)
                except subprocess.TimeoutExpired:
                    export_error = export_error or RunnerError("Guardian cleanup/reap exceeded the parent deadline")
        for fd in (event_read, worker_event_read, reply_write, lifeline_write):
            try:
                os.close(fd)
            except OSError:
                pass
    if export_error is not None:
        raise export_error
    if source_bundle_bytes + tree_bytes(output) > caps.combined_bundle_bytes:
        raise RunnerError("Final receipt exceeded the complete combined byte cap")
    if time.monotonic() > deadline:
        raise TimeoutError("Final receipt/export exceeded the complete parent wall cap")
    guardian_final = json.loads(bounded_regular(guardian_receipt, MAX_JOURNAL_BYTES))
    if (failure is None and (guardian_final.get("status") != "parent_finished"
            or not guardian_final.get("cleanup", {}).get("complete")
            or guardian_final.get("parent_finished") is not True)):
        raise RunnerError("Provisional parent receipt lacks final guardian completion evidence")
    receipt["guardian_confirmation"] = guardian_final
    if failure is None:
        seal = json.loads(bounded_regular(output / "completion-seal.json", MAX_JOURNAL_BYTES))
        if (seal["status"] != "complete_after_all_parent_exports" or seal["identity"] != identity
                or seal["parent_receipt_sha256"] != sha256(bounded_regular(output / "parent-receipt.json", MAX_JOURNAL_BYTES))):
            raise RunnerError("Native completion lacks its post-export guardian seal")
        receipt["completion_seal"] = seal
    receipt["status"] = "complete" if failure is None else "incomplete_" + failure
    receipt["physical_result_completed"] = failure is None and mode == "approved_native_execution"
    receipt["parent_wall_seconds"] = time.monotonic() - started
    receipt["counters"]["cumulative_parent_wall_seconds_charged"] = receipt["parent_wall_seconds"]
    if receipt["parent_wall_seconds"] > caps.parent_wall_seconds:
        raise TimeoutError("Guardian confirmation exceeded the complete parent envelope")
    return receipt
