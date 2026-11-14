#!/usr/bin/env python3
"""Bounded local source for the fixed n=2/3 structured-DQI resource task.

Source preparation is not permission to execute this driver.  A future run
requires explicit ``--execute`` and the externally reviewed driver hash.
Own source and environment fingerprints precede SDK imports.  Runtime uses
ordinary embedded scientific constants, never coordination/private JSON.

The CPU pass makes at most eight actual word-synthesis entries and two direct
QDK Trace estimates.  It checks only newly compiled zero-input states; it
does not repeat accepted simulations, sample assignments or run a QPU task.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from contextlib import suppress
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
import types


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
WORK_SECONDS = 60
NATIVE_SECONDS = 5
CLEANUP_SECONDS = 5
RETENTION_SECONDS = 1
OUTER_SECONDS = 66
RSS_CAP = 1 << 30
OUTPUT_CAP = 8 << 20
FILE_CAP = 32
LOG_CAP = 64 << 10
CONTROL_CAP = 16 << 10
CONTROL_LINE_CAP = 4096
PARENT_RESERVE = 16 << 10
WORD_CAP = 2000
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
NONCE_RE = re.compile(r"[0-9a-f]{32}\Z")
THREAD_ENV = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "RAYON_NUM_THREADS")}

# Final values are bound by the root source review, not discovered by executing
# a backend.  An unbound source stage refuses execution before any SDK import.
MODULE_SHA256 = {'contract': {'path': 'src/openquantum/optimization_dqi_resource_contract.py', 'bytes': 43674, 'sha256': 'cedbd51687f0b9069b34eeebd6753a207df5d98b68cc70cc380b7087e0275971'}, 'compile': {'path': 'src/openquantum/optimization_dqi_resource_compile.py', 'bytes': 24087, 'sha256': '9f0ed4703c43ea1ac30f9a82c91ed70855bd8c03a94708af49a1f48e370717cf'}, 'verify': {'path': 'src/openquantum/optimization_dqi_resource_verify.py', 'bytes': 24739, 'sha256': '80591a7cd8704124caa0ff5e479b57c2e5413f0d28c06692a81c97e0fe7d9780'}, 'physical': {'path': 'src/openquantum/optimization_dqi_resource_physical.py', 'bytes': 34161, 'sha256': 'af164e369d92eedcdfe068382458e4f0b5b18851da10d45c5f5491d32e5ce45a'}}
ENV_PINS = {'runtime': {'python': '3.12.14', 'platform': 'darwin', 'executable_pin': {'bytes': 18058560, 'sha256': '2a8addce917415c8f1c7bd70518a2c802ea2d92c6f0de50c6909a35cca4fc109'}}, 'packages': {'numpy': {'metadata_pin': {'bytes': 6635, 'sha256': 'e2b909d6a37d76c2bc130f528d7fed0c517322bce16dc7846b6a6df30e200403'}, 'metadata_relative': 'numpy-2.5.4.dist-info/METADATA', 'version': '2.5.4'}, 'psutil': {'metadata_pin': {'bytes': 22187, 'sha256': '525f8ba78523cf7bc3759d62a105e7c57fd679ea702402b9449fb0746aa0217f'}, 'metadata_relative': 'psutil-7.2.2.dist-info/METADATA', 'version': '7.2.2'}, 'pytest': {'metadata_pin': {'bytes': 7869, 'sha256': '006823d1759e40b4546ff4296945eda3f0692e799f724bb235edd0512d42eead'}, 'metadata_relative': 'pytest-9.2.0.dist-info/METADATA', 'version': '9.2.0'}, 'qdk': {'metadata_pin': {'bytes': 44299, 'sha256': '3813f07c00561e6997ba7c45b70f0935751e37ac08ff00ce47cf720110b1c729'}, 'metadata_relative': 'qdk-1.34.0.dist-info/METADATA', 'version': '1.34.0'}, 'qiskit': {'metadata_pin': {'bytes': 14016, 'sha256': 'e407c8863a9b3eaa91564fe221a78a48ef8bd3096661c79006172a8178ce5e20'}, 'metadata_relative': 'qiskit-2.6.2.dist-info/METADATA', 'version': '2.6.2'}}, 'native_source_and_binary_files': {'numpy/__init__.py': {'bytes': 25220, 'sha256': '09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11'}, 'numpy/_core/_multiarray_umath.cpython-312-darwin.so': {'bytes': 3996184, 'sha256': 'af36cbbb39b27651a8b9f7dfa491775016f331a58e8d116eb6b4cfd5e22efb77'}, 'psutil/__init__.py': {'bytes': 89878, 'sha256': 'd138a5786b163b56ba86ea0b2d5589dfca37e1bcdf8de1057fe1e933d6ab808a'}, 'psutil/_psosx.py': {'bytes': 15913, 'sha256': '89f0ea0e9c1fe38f224b612792bf31d07c8880d79281806e5379c00514f6b3f2'}, 'psutil/_psutil_osx.abi3.so': {'bytes': 105920, 'sha256': 'ff4568b9187ca7b37c5134aaf614ffb061e7fedccebd84158343afb05991b8e2'}, 'qdk/__init__.py': {'bytes': 3072, 'sha256': '82cdfa32496f56dd9ebb379c7610a5c357e2b58ff201dd258c77c0db79a8a38d'}, 'qdk/_native.abi3.so': {'bytes': 15156816, 'sha256': '98a99d5bb076a8e306a8ca7d241d9b96348a600e71e14c4f97fd70c6b1986848'}, 'qdk/qre/__init__.py': {'bytes': 2368, 'sha256': '5580dfb76673b1b42c88447c827af7e5048573bb7b76143b5afe21242b84db6f'}, 'qdk/qre/_architecture.py': {'bytes': 13654, 'sha256': '3d188a5a71143a4a023bfe5e06fbabf4de573cbb32abada823373ab1418f98cf'}, 'qdk/qre/_instruction.py': {'bytes': 15681, 'sha256': '9f0c7b7cfa22c68b23aba131c79c0665273b0555cb6ade24ae8eb558eadfb7cb'}, 'qdk/qre/_isa_enumeration.py': {'bytes': 13683, 'sha256': 'f35a4248c0d29174daac033d12f8296ad6c3b5a203a4866703d763a5b65aa97e'}, 'qdk/qre/_qre.py': {'bytes': 785, 'sha256': '58b0a06b4602e8e5737f283a8b21c54434a1df33f8ad1ad5f80ec09343b7a8b2'}, 'qdk/qre/_qre.pyi': {'bytes': 51968, 'sha256': 'efe37f94976f8f658cc9fa17c9380533f0639d208546b1628a5dcbd3e4a19791'}, 'qdk/qre/_trace.py': {'bytes': 8807, 'sha256': '75c65fdc24f8e452486fbd022ca1272aca5dfb10f45b237205929343189012d4'}, 'qdk/qre/instruction_ids.py': {'bytes': 400, 'sha256': '87a1a9a8286caab890d343f4cc56dd6c7facf8f935bc2fa7c2bc150982b15f6e'}, 'qdk/qre/models/factories/_round_based.py': {'bytes': 17243, 'sha256': 'c5fcb478c86534cf9888bc8a9b9d3e5a733355bf92c41072228d99fdc0d19110'}, 'qdk/qre/models/qec/_surface_code.py': {'bytes': 5913, 'sha256': '3530f92f328e325d6d0c3e245f8688c11956fa6f63b1e25455b5d14cd844ed35'}, 'qdk/qre/models/qubits/_gate_based.py': {'bytes': 3881, 'sha256': '8e815aef59c2d724259bb13ddb5e2212332d6febd9bf2e00cca9c33d602b0273'}, 'qiskit/_accelerate.abi3.so': {'bytes': 17163680, 'sha256': '623cc1cc517f2082ba0a074e17cf00d2d29e9353a3d49a03c3173ebe23cac884'}, 'qiskit/circuit/library/standard_gates/ry.py': {'bytes': 10030, 'sha256': '4626b640905f978b54882e257fd6613349551c6ba01decb8fc5bcf1723648a73'}, 'qiskit/circuit/quantumcircuit.py': {'bytes': 343482, 'sha256': '130fff1c441a17a6d1ea8d134b56a63fab05f78041357dd0e20c349ee5dfde0d'}, 'qiskit/synthesis/discrete_basis/ross_selinger.py': {'bytes': 2580, 'sha256': '69946403d2e3dcb9d2547873206f9fffdcfb0fe6d15ef118d08b32648dc9e494'}}}
_ACTIVE_CHILD = None
_WORD_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg"})


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def finite_hex(text, name):
    if not isinstance(text, str) or len(text) > 64:
        raise ValueError(f"{name} must be a bounded float hex string")
    value = float.fromhex(text)
    if not math.isfinite(value) or value.hex() != text:
        raise ValueError(f"{name} must be finite canonical float hex")
    return value


def bounded_regular_read(path, cap):
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
            raise ValueError("A bounded regular input is required")
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("Input exceeded its read bound")
    return raw


def control_lines(buffer, raw, total):
    """Bound each line without rejecting a read containing several lines."""
    if not isinstance(buffer, (bytes, bytearray)) or not isinstance(raw, bytes):
        raise ValueError("Control reads must be bytes")
    total = integer(total, "control total", 0, CONTROL_CAP) + len(raw)
    if total > CONTROL_CAP:
        raise ValueError("Control-pipe bytes exceeded their total bound")
    values = bytes(buffer) + raw
    lines = values.split(b"\n")
    remainder = lines.pop()
    if len(remainder) > CONTROL_LINE_CAP or any(not line or len(line) > CONTROL_LINE_CAP for line in lines):
        raise ValueError("Control event exceeded its per-line bound")
    return bytearray(remainder), lines, total


def integer(value, name, minimum=0, maximum=(1 << 63) - 1):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside its integer bound")
    return value


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The complete work deadline elapsed")


def checked_file(path, pin, cap=32 << 20):
    """Hash a pinned regular file with bounded reads and no symlink leaf."""
    path = Path(path)
    if not isinstance(pin, dict) or set(pin) != {"bytes", "sha256"}:
        raise ValueError("A complete byte/hash fingerprint is required")
    size = integer(pin["bytes"], "source bytes", 1, cap)
    if not isinstance(pin["sha256"], str) or not SHA_RE.fullmatch(pin["sha256"]):
        raise ValueError("Source fingerprint is unbound or malformed")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != size:
        raise ValueError("Pinned source must be a matching regular file")
    digest = hashlib.sha256()
    total = 0
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as handle:
        actual = os.fstat(handle.fileno())
        if not stat.S_ISREG(actual.st_mode) or actual.st_size != size:
            raise ValueError("Pinned source changed before its read")
        while raw := handle.read(64 << 10):
            total += len(raw)
            if total > size:
                raise ValueError("Pinned source grew during its read")
            digest.update(raw)
    if total != size or digest.hexdigest() != pin["sha256"]:
        raise ValueError("Pinned source identity changed")
    return dict(pin)


def source_path(root, relative):
    if not isinstance(relative, str) or len(relative) > 256:
        raise ValueError("A bounded relative source path is required")
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(x in {"", ".", ".."} for x in parts):
        raise ValueError("Source path must remain within the input root")
    path = Path(root)
    for part in parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Source paths cannot traverse symlinks")
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("Source path escaped its input root")
    return path


def check_self(expected):
    if not isinstance(expected, str) or not SHA_RE.fullmatch(expected):
        raise ValueError("Explicit execution needs the reviewed driver SHA-256")
    pin = {"bytes": SELF.stat().st_size, "sha256": expected}
    return checked_file(SELF, pin, 1 << 20)


def load_modules(root):
    """Load only exact reviewed pure modules, without the package initializer."""
    buffers = {}
    for role, entry in MODULE_SHA256.items():
        if set(entry) != {"path", "bytes", "sha256"}:
            raise ValueError("Incomplete module fingerprint")
        path = source_path(root, entry["path"])
        checked_file(path, {k: entry[k] for k in ("bytes", "sha256")}, 1 << 20)
        with path.open("rb") as handle:
            raw = handle.read(entry["bytes"] + 1)
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("A pure module changed before compilation")
        buffers[role] = (path, raw)
    modules = {}
    for role, (path, raw) in buffers.items():
        name = "_dqi_resource_driver_" + role
        module = types.ModuleType(name)
        module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(raw, str(path), "exec"), module.__dict__)
        modules[role] = module
    return modules


def environment_fingerprints():
    """Inspect exact installed metadata/files before importing any SDK."""
    if set(ENV_PINS) != {"runtime", "packages", "native_source_and_binary_files"}:
        raise ValueError("Complete environment fingerprints are required")
    runtime = ENV_PINS["runtime"]
    if sys.platform != runtime["platform"] or ".".join(map(str, sys.version_info[:3])) != runtime["python"]:
        raise ValueError("The reviewed local runtime changed")
    checked_file(Path(sys.executable).resolve(), runtime["executable_pin"])
    required = {"qdk": "1.34.0", "qiskit": "2.6.2", "numpy": "2.5.4", "psutil": "7.2.2"}
    if not required.keys() <= ENV_PINS["packages"].keys() or len(ENV_PINS["packages"]) > 8:
        raise ValueError("Required package metadata pins are missing")
    distributions = {}
    for package, entry in ENV_PINS["packages"].items():
        distribution = importlib.metadata.distribution(package)
        if distribution.version != entry["version"] or (package in required and entry["version"] != required[package]):
            raise ValueError("A pinned package version changed")
        metadata = distribution.locate_file(entry["metadata_relative"])
        checked_file(metadata, entry["metadata_pin"], 1 << 20)
        distributions[package] = distribution
    pins = ENV_PINS["native_source_and_binary_files"]
    if not pins or len(pins) > 32:
        raise ValueError("Native source/binary pins are missing or excessive")
    for relative, pin in pins.items():
        package, separator, rest = relative.partition("/")
        if not separator or package not in distributions:
            raise ValueError("Unknown native fingerprint package")
        package_root = Path(distributions[package].locate_file(package)).resolve()
        checked_file(source_path(package_root, rest), pin)
    return ENV_PINS


def checked_import_origin(package, module, relative="__init__.py"):
    """Match the loaded SDK file to the already pinned distribution root."""
    distribution = importlib.metadata.distribution(package)
    expected = Path(distribution.locate_file(package)).resolve() / relative
    loaded = getattr(module, "__file__", None)
    if not isinstance(loaded, str) or Path(loaded).resolve() != expected.resolve():
        raise ValueError("A loaded SDK module escaped its pinned package root")
    pin = ENV_PINS["native_source_and_binary_files"].get(package + "/" + relative)
    if pin is not None:
        checked_file(expected, pin)


class FlatStore:
    """Retain bounded flat artifacts, including transient atomic-write bytes."""

    def __init__(self, root, *, worker=False):
        self.root = Path(root)
        if self.root.is_symlink() or not stat.S_ISDIR(self.root.lstat().st_mode):
            raise ValueError("Output root must be a regular directory")
        self.directory_fd = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                    | getattr(os, "O_NOFOLLOW", 0))
        self.identity = os.fstat(self.directory_fd)
        self.byte_cap = OUTPUT_CAP - (LOG_CAP + PARENT_RESERVE if worker else 0)
        self.file_cap = FILE_CAP - (3 if worker else 0)

    def check_root(self):
        actual = self.root.lstat()
        if (not stat.S_ISDIR(actual.st_mode) or actual.st_dev != self.identity.st_dev
                or actual.st_ino != self.identity.st_ino):
            raise ValueError("The owned output directory changed")

    def read(self, name, cap):
        self.check_root()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", name):
            raise ValueError("Invalid flat input name")
        fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_NONBLOCK", 0), dir_fd=self.directory_fd)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
                raise ValueError("A bounded regular output input is required")
            raw = handle.read(cap + 1)
        if len(raw) > cap:
            raise ValueError("Output input exceeded its read bound")
        return raw

    def inventory(self):
        self.check_root()
        total, count = 0, 0
        with os.scandir(self.directory_fd) as entries:
            for entry in entries:
                count += 1
                if count > self.file_cap:
                    raise ValueError("Retained file-count bound exceeded")
                info = entry.stat(follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Retained output must contain only flat regular files")
                total += info.st_size
                if total > self.byte_cap:
                    raise ValueError("Retained byte bound exceeded")
        return total, count

    def put(self, name, raw, *, exclusive=False):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", name):
            raise ValueError("Invalid flat artifact name")
        if not isinstance(raw, bytes) or len(raw) > self.byte_cap:
            raise ValueError("Artifact must be bounded bytes")
        total, count = self.inventory()
        # Do not subtract the old file: replacement temporarily retains both.
        if total + len(raw) > self.byte_cap or count + 1 > self.file_cap:
            raise ValueError("Atomic artifact retention would exceed its bound")
        if exclusive:
            # A partial exclusive startup record still spends the output slot;
            # no link/rename creates an unaccounted duplicate file or byte copy.
            descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=self.directory_fd)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.fsync(self.directory_fd)
            return
        temporary = "." + name + ".partial"
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=self.directory_fd)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, name, src_dir_fd=self.directory_fd, dst_dir_fd=self.directory_fd)
            os.fsync(self.directory_fd)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary, dir_fd=self.directory_fd)

    def json(self, name, value, **kwargs):
        self.put(name, canonical(value) + b"\n", **kwargs)

    def close(self):
        os.close(self.directory_fd)


class CallWatch:
    """Pure event state machine; a completed message cannot bypass call limits."""

    def __init__(self, nonce, guard, started, deadline):
        if not isinstance(nonce, str) or not NONCE_RE.fullmatch(nonce):
            raise ValueError("Invalid local control nonce")
        self.nonce, self.guard = nonce, guard
        self.started, self.deadline = started, deadline
        self.active = None
        self.sequence = 0
        self.complete = False
        self.failure = None

    def accept(self, event, now):
        if not isinstance(event, dict) or event.get("nonce") != self.nonce or self.complete or self.failure:
            raise ValueError("Unexpected local control event")
        kind = event.get("event")
        at = finite_hex(event.get("at_hex"), "control timestamp")
        if not self.started <= at <= now + .01 or at > self.deadline:
            raise ValueError("Control timestamp violates the work clock")
        if kind == "native_begin":
            if set(event) != {"event", "nonce", "at_hex", "sequence", "kind", "key"} or self.active is not None:
                raise ValueError("Overlapping or malformed native entry")
            sequence = integer(event["sequence"], "native sequence", 1, 10)
            if sequence != self.sequence + 1:
                raise ValueError("Native entry order changed")
            key = tuple(event["key"]) if event["kind"] == "synthesis" and isinstance(event["key"], list) else event["key"]
            self.guard.charge(event["kind"], key)
            self.sequence = sequence
            self.active = (sequence, at)
        elif kind == "native_end":
            if set(event) != {"event", "nonce", "at_hex", "sequence", "status"} or self.active is None:
                raise ValueError("Native return lacks its charged entry")
            sequence, before = self.active
            returned_sequence = integer(event["sequence"], "native return sequence", 1, 10)
            if returned_sequence != sequence or at < before or at - before > NATIVE_SECONDS:
                raise ValueError("Native return violates its call clock")
            self.active = None
            if event["status"] != "returned":
                self.failure = "A charged native entry failed"
        elif kind == "complete":
            if set(event) != {"event", "nonce", "at_hex", "counts"} or self.active is not None:
                raise ValueError("Completion overlaps a native entry")
            self.guard.require_complete()
            if canonical(event["counts"]) != canonical(self.guard.snapshot()):
                raise ValueError("Worker and parent native counters differ")
            self.complete = True
        elif kind == "failure":
            if set(event) != {"event", "nonce", "at_hex", "reason"} or not isinstance(event["reason"], str) or len(event["reason"]) > 256:
                raise ValueError("Malformed worker failure")
            self.failure = event["reason"]
        else:
            raise ValueError("Unknown query/control class")

    def check(self, now):
        if now >= self.deadline:
            raise TimeoutError("The work clock elapsed")
        if self.active and now >= self.active[1] + NATIVE_SECONDS:
            raise TimeoutError("A native entry exceeded five seconds")
        if self.failure:
            raise RuntimeError(self.failure)


def worker_alarm(deadline, maximum=None):
    """Default-action alarm bounds C calls independently of Python/GIL work."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("The worker deadline elapsed")
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.setitimer(signal.ITIMER_REAL, min(remaining, maximum) if maximum else remaining)


def emit_control(fd, nonce, event, **values):
    raw = canonical(dict(event=event, nonce=nonce, at_hex=time.monotonic().hex(), **values)) + b"\n"
    if len(raw) > CONTROL_LINE_CAP:
        raise ValueError("Control event exceeded its byte bound")
    if os.write(fd, raw) != len(raw):
        raise RuntimeError("Incomplete local control-pipe write")


def serialize_word(circuit):
    if circuit.num_qubits != 1 or circuit.num_clbits != 0 or len(circuit.data) > WORD_CAP:
        raise ValueError("Native synthesis returned an oversized or non-single-qubit word")
    phase = float(circuit.global_phase)
    if not math.isfinite(phase):
        raise ValueError("Native synthesis phase must be finite")
    operations = []
    for item in circuit.data:
        name = item.operation.name
        if name not in _WORD_GATES or len(item.qubits) != 1 or item.clbits or item.operation.params:
            raise ValueError("Native word contains an unsupported operation")
        if circuit.find_bit(item.qubits[0]).index != 0:
            raise ValueError("Native word wire mapping changed")
        operations.append([name, [0], [], []])
    return {"operations": operations, "global_phase_hex": phase.hex()}


def invoke_native(kind, key, function, capture, *, guard, calls, store, control_fd, nonce, deadline):
    """Spend a key and sync its before/after record around exactly one entry."""
    check_deadline(deadline)
    guard.charge(kind, key)
    sequence = len(calls) + 1
    record = dict(sequence=sequence, kind=kind, key=key,
                  status="charged_before_entry", before_hex=time.monotonic().hex())
    calls.append(record)
    store.json("native-calls.json", calls)
    emit_control(control_fd, nonce, "native_begin", sequence=sequence, kind=kind, key=key)
    worker_alarm(deadline, NATIVE_SECONDS)
    status = "exception"
    try:
        value = function()
        record["status"] = "returned_pending_capture"
        try:
            raw_return = capture(value)
        except BaseException as error:
            record.update(status="return_capture_failed", error_type=type(error).__name__)
            record["raw_return"] = failed_return_capture(value, error)
            status = "capture_failed"
            raise
        record.update(status="returned", raw_return=raw_return)
        status = "returned"
        return value
    except BaseException as error:
        record.setdefault("error_type", type(error).__name__)
        if record["status"] == "charged_before_entry":
            record["status"] = "native_exception"
        raise
    finally:
        record["after_hex"] = time.monotonic().hex()
        store.json("native-calls.json", calls)
        emit_control(control_fd, nonce, "native_end", sequence=sequence, status=status)
        worker_alarm(deadline)


def failed_return_capture(value, error):
    """A rejected native word still leaves a bounded, explicitly partial return."""
    raw = dict(complete=False, validated=False, return_type=type(value).__name__[:80],
               capture_error_type=type(error).__name__[:80])
    provided = getattr(error, "raw_result", None)
    if provided is not None:
        try:
            bounded_captured_json(provided)
            raw["provided_raw_result"] = provided
        except (ValueError, TypeError):
            raw["provided_raw_result_rejected"] = True
    for name in ("num_qubits", "num_clbits"):
        with suppress(Exception):
            item = getattr(value, name)
            raw[name] = item if type(item) is int and 0 <= item < 1 << 63 else "unsupported"
    with suppress(Exception):
        count = len(value.data)
        raw["operation_count"] = count if type(count) is int and 0 <= count < 1 << 63 else "unsupported"
        prefix = []
        for item in value.data[:16]:
            name = item.operation.name
            prefix.append(name[:80] if isinstance(name, str) else "unsupported")
        raw["operation_name_prefix"] = prefix
    return raw


def bounded_captured_json(value):
    """Keep a physical adapter's finite partial return without accepting it."""
    stack = [(value, 0)]
    nodes, strings = 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 256 or depth > 8:
            raise ValueError("Captured partial return exceeded its structural bound")
        if item is None or type(item) is bool:
            continue
        if type(item) is str:
            strings += len(item.encode())
            if strings > 4096:
                raise ValueError("Captured partial return strings exceeded their bound")
        elif type(item) is int:
            if item.bit_length() > 63:
                raise ValueError("Captured partial integer exceeded its bound")
        elif type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Captured partial float must be finite")
        elif type(item) in (dict, list):
            if len(item) > 64:
                raise ValueError("Captured partial container exceeded its bound")
            if type(item) is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("Captured partial keys must be strings")
                stack.extend((key, depth + 1) for key in item)
                stack.extend((entry, depth + 1) for entry in item.values())
            else:
                stack.extend((entry, depth + 1) for entry in item)
        else:
            raise TypeError("Captured partial return must be finite plain JSON")


def worker(args):
    """Internal ordinary subprocess entry, guarded before modules or SDKs."""
    if (getattr(args, "execute", None) is not True or not sys.flags.isolated
            or args.parent_pid != os.getppid() or args.parent_pid <= 1
            or args.control_fd is None or args.control_fd < 3
            or not isinstance(args.nonce, str) or not NONCE_RE.fullmatch(args.nonce)
            or os.getsid(0) != os.getpid()):
        raise PermissionError("Worker entry requires its live ordinary parent and pipe")
    if not stat.S_ISFIFO(os.fstat(args.control_fd).st_mode):
        raise PermissionError("Worker control descriptor is not a pipe")
    deadline = finite_hex(args.deadline_hex, "work deadline")
    worker_alarm(deadline)
    check_self(args.source_sha256)
    store = FlatStore(args.output_dir, worker=True)
    raw = store.read("invocation.json", 8192)
    invocation = json.loads(raw)
    if any(invocation.get(k) != v for k, v in {
        "nonce": args.nonce, "parent_pid": args.parent_pid,
        "source_sha256": args.source_sha256, "deadline_hex": args.deadline_hex}.items()):
        raise PermissionError("Worker invocation binding changed")
    store.json("worker-started.json", {"status": "entered", "source_sha256": args.source_sha256}, exclusive=True)
    sys.dont_write_bytecode = True
    environment_fingerprints()
    modules = load_modules(args.input_root)
    contract, compiler, verifier, physical = (modules[k] for k in ("contract", "compile", "verify", "physical"))
    task = contract.public_task()
    cases = contract.accepted_cases(args.input_root)
    guard = contract.CounterGuard()
    calls = []
    check = lambda: check_deadline(deadline)
    try:
        def native_word(target):
            check()
            # Imports are after source/environment/input checks and the worker
            # deadline alarm.  The charged entry itself remains exactly one.
            import numpy as np
            from qiskit.circuit.library import RYGate
            from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_unitary
            checked_import_origin("numpy", np)
            checked_import_origin("qiskit", sys.modules["qiskit"])
            checked_import_origin("qiskit", sys.modules["qiskit.synthesis.discrete_basis.ross_selinger"],
                                  "synthesis/discrete_basis/ross_selinger.py")
            checked_import_origin("qiskit", sys.modules["qiskit.circuit.library.standard_gates.ry"],
                                  "circuit/library/standard_gates/ry.py")
            angle = finite_hex(target["angle_hex"], "RY angle")
            epsilon = finite_hex(target["epsilon_requested_float_hex"], "synthesis epsilon")
            matrix = np.asarray(RYGate(angle).to_matrix(), dtype=complex)
            if matrix.shape != (2, 2) or matrix.nbytes > 64:
                raise ValueError("Native input is not a bounded two-by-two matrix")
            word = invoke_native("synthesis", contract.synthesis_key(target["angle_hex"]),
                lambda: gridsynth_unitary(matrix, epsilon), serialize_word,
                guard=guard, calls=calls, store=store, control_fd=args.control_fd,
                nonce=args.nonce, deadline=deadline)
            return serialize_word(word)

        def word_validator(target, word_record):
            check()
            return verifier.validate_word(word_record["operations"],
                finite_hex(word_record["global_phase_hex"], "word phase"), target["angle_hex"], check=check)

        compiled = compiler.compile_cases(cases, task["compiler"]["rotation_targets"],
            native_call=native_word, word_validator=word_validator, check=check, limits=task["limits"])
        store.json("native-words.json", compiled["words"])
        verification, estimates = [], []
        for case in compiled["cases"]:
            check()
            n = integer(case["n"], "case", 2, 3)
            ops = case["operations"]
            raw_ops = b"".join(canonical(operation) + b"\n" for operation in ops)
            store.put(f"n{n}-compiled.ndjson", raw_ops)
            store.json(f"n{n}-compiled.json", {k: v for k, v in case.items() if k != "operations"})
            for block in case["routing_blocks"]:
                verifier.verify_routed_cx_block(block["wires"], block["control"], block["target"], block["operations"], check=check)
            errors = [item["guarded_frobenius_error"] for item in case["rotation_occurrences"]]
            contract.guard_occurrence_errors(n, errors)
            checked = verifier.verify_compiled_case(n, ops,
                finite_hex(case["global_phase_hex"], "compiled phase"), case["whole_error"],
                case["source_readout_mapping"], rotation_errors=errors, check=check)
            verification.append(checked)
            store.json("verification.json", verification)
        # Both new compiled states pass before any physical estimator entry.
        for case in compiled["cases"]:
            n = integer(case["n"], "case", 2, 3)
            ops = case["operations"]
            check()
            def trace_call(kind, function, *values, **keywords):
                if kind != "direct_trace":
                    raise ValueError("No additional native query class is allowed")
                return invoke_native("trace", n, lambda: function(*values, **keywords),
                    physical.capture_native_result, guard=guard, calls=calls, store=store,
                    control_fd=args.control_fd, nonce=args.nonce, deadline=deadline)

            estimate = physical.estimate_case(n, ops, contract.frozen_identity(),
                task["frozen_profile"], native_call=trace_call)
            estimates.append(estimate)
            store.json("physical.json", estimates)
        guard.require_complete()
        contract.verify_frozen_inputs(args.input_root)
        check()
        output_bounds = [dict(n=case["n"], case_id=f"n{case['n']}_optimum99",
            objective_target=case["n"], objective_condition="f(x)=n",
            success_target_fraction="99/100",
            **contract.whole_task_success_bound(case["n"], case["whole_error"],
                Fraction(point["native_physical_failure_all_draws_exact"]),
                Fraction(point["factory_supply"]["availability_failure_exact"])))
            for case, point in zip(compiled["cases"], estimates, strict=True)]
        result = dict(status="complete_worker_component_pending_parent_acceptance", native_counts=guard.snapshot(),
            frozen_inputs_verified=True, frozen_input_count=14,
            accepted_study_not_reexecuted=True, source_pins=MODULE_SHA256,
            environment_pins=ENV_PINS, physical_points=estimates,
            verification=verification, whole_task_output_bounds=output_bounds,
            sampled_outcomes=False, qpu_execution=False,
            conditional_on=task["output"]["conditional_on"],
            simultaneous_case_success_claim=False,
            classical_baseline=dict(kind="analytic_all_zero_optimum", search_calls=0,
                benchmark_measured=False, optimization_advantage_claim=False,
                assignments=[dict(n=n, assignment="0"*n, objective=n, output_bit_writes=n)
                             for n in (2, 3)]),
            physical_configuration_optimized=False,
            acceptance_requires_parent_receipt_and_observed_cli_exit_zero=True,
            excluded_host_hardware_fields=["assignment transfer reliability/latency", "host scoring/selection reliability/latency"],
            scope="new compiled circuits and fixed complete-task ledgers only; not hardware, formal proof or publication")
        store.json("result.json", result)
        check()
        emit_control(args.control_fd, args.nonce, "complete", counts=guard.snapshot())
        return 0
    except BaseException as error:
        with suppress(BaseException):
            failure = dict(status="incomplete", error_type=type(error).__name__,
                           native_counts=guard.snapshot(), no_retry=True)
            if getattr(error, "raw_result", None) is not None:
                failure["native_failure_return"] = failed_return_capture(None, error)
            store.json("worker-failure.json", failure)
            emit_control(args.control_fd, args.nonce, "failure", reason=type(error).__name__)
        raise


def fresh_output(input_root, output):
    input_root = Path(input_root).resolve(strict=True)
    requested = Path(output)
    parent = requested.parent.resolve(strict=True)
    path = parent / requested.name
    if requested.name in {"", ".", ".."} or path.exists() or path.is_symlink():
        raise ValueError("A fresh output directory is required")
    for protected_root in {input_root, ROOT.resolve()}:
        if protected_root == path or protected_root.is_relative_to(path):
            raise ValueError("Output cannot replace a source input tree")
        for name in ("assets", "notebooks", "src", "scripts", "tests", "docs", ".git"):
            if path.is_relative_to(protected_root / name):
                raise ValueError("Output cannot modify source, accepted evidence or coordination")
    path.mkdir(mode=0o700)
    return path


def stop_child(child, deadline):
    """Signal only the still-held child/session created by this parent."""
    errors = []
    if child.poll() is not None:
        return errors
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if child.poll() is not None:
            break
        try:
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, sig)
            else:
                child.send_signal(sig)
        except ProcessLookupError:
            pass
        allowance = max(0, min(1.0 if sig == signal.SIGTERM else 3.0, deadline - time.monotonic()))
        try:
            child.wait(timeout=allowance)
        except subprocess.TimeoutExpired:
            continue
    if child.poll() is None:
        errors.append("Owned child could not be reaped within cleanup")
    return errors


def hard_parent_exit(signum, frame):
    child = _ACTIVE_CHILD
    if child is not None and child.poll() is None:
        with suppress(ProcessLookupError):
            if os.getpgid(child.pid) == child.pid:
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
    os._exit(124)


def graceful_interrupt(signum, frame):
    raise KeyboardInterrupt("Local driver interrupted")


def parent(args):
    global _ACTIVE_CHILD
    if getattr(args, "execute", None) is not True or not sys.flags.isolated:
        raise PermissionError("Callable parent execution needs explicit opt-in and Python -I")
    started = time.monotonic()
    deadline, outer = started + WORK_SECONDS, started + OUTER_SECONDS
    signal.signal(signal.SIGALRM, hard_parent_exit)
    signal.setitimer(signal.ITIMER_REAL, OUTER_SECONDS)
    signal.signal(signal.SIGINT, graceful_interrupt)
    signal.signal(signal.SIGTERM, graceful_interrupt)
    check_self(args.source_sha256)
    sys.dont_write_bytecode = True
    environment_fingerprints()
    modules = load_modules(args.input_root)
    modules["contract"].accepted_cases(args.input_root)
    check_deadline(deadline)
    output = fresh_output(args.input_root, args.output_dir)
    store = FlatStore(output)
    nonce = os.urandom(16).hex()
    store.json("invocation.json", dict(status="starting", nonce=nonce, parent_pid=os.getpid(),
        source_sha256=args.source_sha256, deadline_hex=deadline.hex()))
    read_fd, write_fd = os.pipe()
    write_open = True
    env = dict(os.environ, **THREAD_ENV, PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, str(SELF), "--execute", "--worker",
        "--source-sha256", args.source_sha256, "--input-root", str(Path(args.input_root).resolve()),
        "--output-dir", str(output), "--control-fd", str(write_fd),
        "--parent-pid", str(os.getpid()), "--nonce", nonce, "--deadline-hex", deadline.hex()]
    command.insert(1, "-I")
    selector = None
    streams = {}
    logs = {"stdout": bytearray(), "stderr": bytearray()}
    control_buffer = bytearray()
    control_total = 0
    watch = CallWatch(nonce, modules["contract"].CounterGuard(), started, deadline)
    rss_peak, rss_samples = 0, 0
    errors = []
    reason = None
    capture_truncated = False
    try:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, pass_fds=(write_fd,), start_new_session=True, env=env)
    except BaseException:
        os.close(read_fd)
        os.close(write_fd)
        store.json("parent-receipt.json", dict(status="incomplete", reason="spawn_failure",
            worker_returncode=None, no_retry=True, publication_complete=False))
        store.close()
        raise
    # Once a child exists, every setup/read/import failure takes this teardown.
    try:
        _ACTIVE_CHILD = child
        os.close(write_fd)
        write_open = False
        selector = selectors.DefaultSelector()
        streams = {"stdout": child.stdout, "stderr": child.stderr, "control": read_fd}
        for label, handle in streams.items():
            fd = handle if isinstance(handle, int) else handle.fileno()
            os.set_blocking(fd, False)
            selector.register(fd, selectors.EVENT_READ, label)
        import psutil
        checked_import_origin("psutil", psutil)
        process = psutil.Process(child.pid)
        while selector.get_map() or child.poll() is None:
            # First consume already queued returns.  Scheduling delays must
            # not turn a genuine <=5 s native return into a fabricated timeout.
            for key, _ in selector.select(timeout=.05):
                raw = os.read(key.fd, 8192)
                if not raw:
                    selector.unregister(key.fd)
                    continue
                if key.data != "control":
                    if len(logs["stdout"]) + len(logs["stderr"]) + len(raw) > LOG_CAP:
                        available = LOG_CAP - len(logs["stdout"]) - len(logs["stderr"])
                        logs[key.data].extend(raw[:max(0, available)])
                        capture_truncated = True
                        raise ValueError("Combined capture logs exceeded 64 KiB")
                    logs[key.data].extend(raw)
                    continue
                control_buffer, lines, control_total = control_lines(control_buffer, raw, control_total)
                for line in lines:
                    watch.accept(modules["contract"].parse_json(line), time.monotonic())
            now = time.monotonic()
            if child.poll() is None:
                watch.check(now)
                try:
                    descendants = process.children(recursive=True)
                    if len(descendants) > 16:
                        raise RuntimeError("SDK worker spawned too many unplanned processes")
                    rss = process.memory_info().rss
                    for descendant in descendants:
                        with suppress(psutil.NoSuchProcess):
                            rss += descendant.memory_info().rss
                    rss_samples += 1
                    rss_peak = max(rss_peak, rss)
                    if rss > RSS_CAP:
                        raise MemoryError("Sampled child RSS exceeded 1 GiB")
                    if descendants:
                        raise RuntimeError("SDK worker spawned an unplanned process")
                except psutil.NoSuchProcess:
                    pass
            if child.poll() is not None and selector.get_map() and time.monotonic() > deadline + CLEANUP_SECONDS:
                raise TimeoutError("Output pipes did not drain within cleanup")
        returncode = child.wait(timeout=0)
        if returncode != 0 or not watch.complete or watch.active or control_buffer or rss_samples == 0:
            raise RuntimeError("Worker exit, counters, control drain or RSS observation is incomplete")
        check_deadline(deadline)
    except BaseException as error:
        reason = type(error).__name__
        errors.extend(stop_child(child, min(outer - RETENTION_SECONDS, time.monotonic() + CLEANUP_SECONDS)))
        # Captured bytes remain bounded; available bytes are drained below.
        drain_until = min(outer - RETENTION_SECONDS, time.monotonic() + 1)
        while selector is not None and selector.get_map() and time.monotonic() < drain_until:
            for key, _ in selector.select(timeout=.02):
                with suppress(BlockingIOError):
                    raw = os.read(key.fd, 8192)
                    if not raw:
                        selector.unregister(key.fd)
                    elif key.data != "control":
                        available = LOG_CAP - len(logs["stdout"]) - len(logs["stderr"])
                        logs[key.data].extend(raw[:max(0, available)])
                        capture_truncated |= len(raw) > max(0, available)
    finally:
        drain_complete = selector is not None and not selector.get_map() and not control_buffer
        if selector is not None:
            selector.close()
        for handle in (child.stdout, child.stderr, read_fd):
            with suppress(OSError):
                os.close(handle) if isinstance(handle, int) else handle.close()
        if write_open:
            with suppress(OSError):
                os.close(write_fd)
        _ACTIVE_CHILD = None if child.poll() is not None else child
    retention_started = time.monotonic()
    store.put("stdout.log", bytes(logs["stdout"]))
    store.put("stderr.log", bytes(logs["stderr"]))
    receipt = dict(status="passed" if reason is None and not errors and drain_complete else "incomplete",
        reason=reason, cleanup_errors=errors, worker_returncode=child.poll(),
        counters=watch.guard.snapshot(), complete_control=watch.complete,
        pipe_drain_complete=drain_complete, sampled_child_peak_rss_bytes=rss_peak,
        rss_samples=rss_samples, rss_scope="sampled worker and observed descendants; unplanned processes rejected",
        capture_complete=not capture_truncated, acceptance_requires_observed_cli_exit_zero=True,
        work_elapsed_before_retention=time.monotonic() - started,
        no_retry=True, publication_complete=False)
    store.json("parent-receipt.json", receipt)
    if time.monotonic() > min(outer, retention_started + RETENTION_SECONDS):
        receipt.update(status="incomplete", reason="retention_deadline_elapsed")
        with suppress(BaseException):
            store.json("parent-receipt.json", receipt)
        store.close()
        raise TimeoutError("Final retention exceeded its allowance")
    store.close()
    return 0 if receipt["status"] == "passed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicit opt-in after exact source/argv review")
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--control-fd", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--nonce", help=argparse.SUPPRESS)
    parser.add_argument("--deadline-hex", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args.execute:
        parser.error("No native execution is enabled without --execute and exact source review")
    if os.name != "posix" or not hasattr(signal, "setitimer"):
        parser.error("This source vector requires the reviewed POSIX local subprocess route")
    if not sys.flags.isolated:
        parser.error("The reviewed parent and worker require Python -I")
    if args.worker:
        return worker(args)
    if any(x is not None for x in (args.control_fd, args.parent_pid, args.nonce, args.deadline_hex)):
        parser.error("Internal worker arguments cannot be supplied to the public parent")
    return parent(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as error:
        print(f"Incomplete local resource task: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1)
