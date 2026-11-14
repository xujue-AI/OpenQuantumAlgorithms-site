"""Prepared public interface for the frozen local-spin finite checks.

This standalone Python 3.12 CLI needs an extracted saved-evidence folder and
the separately downloaded frozen assertion file.  It supplies the caller
ledger and compatibility context itself; no private driver or approval stack
is imported.  Source preparation is not an executed portability claim.

Only the ten frozen assertion functions and the four n=1,2 finite-reference
cases are admitted.  Existing QASM and cases are copied, never regenerated.
The assertion fixtures generate 750 primitives, including one 15-operation
file; the reference generates 1,176 compact primitives.  Scientific schedules
are reused without new draws.  Temporary filename entropy is outside that RNG
accounting.  ODEs, matrix exponential solvers, shots, synthesis, large density
simulation, notebooks, subprocesses and physical estimates are absent.

Time, sampled own-process RSS and output scans are cooperative checkpoints,
not hard native-call, host-storage or process-isolation proofs.  A CLI exit
and independent saved-output review are required to establish interface
completion.  Every result keeps all_components_verified=false and retains
poor target bias without tuning or renormalizing raw simulated matrices.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import threading
import time
import types


INTERFACE_ID = "local-spin-bath-public-reproduction-v1"
METHOD_ID = "local-spin-full-xx-reusable-bath-v1"
SOURCE_COMMIT = "b902bc5c0d43fa6e5cfbfcab0e041cb475840fc9"
PUBLIC_COMMIT = "ce6a2f26de36a3d7a80bed28219ee61b9088737e"
MANIFEST_SHA256 = "63f43d784d7f6dfec8f743843961df1d072c049469068582a26696fb410938a7"
CHECKS_SHA256 = "ec7c6e300a867a6a89a3a9df9f3e3b027ab8f5ff5efbd204fdd37f3554707ee5"
CASE_IDS = (
    "n1-ground", "n1-thermal", "n2-ground", "n2-thermal",
    "n4-thermal-generation", "n8-thermal-generation", "n16-thermal-generation",
    "n32-thermal-generation", "n64-thermal-generation",
)
SMALL_IDS = CASE_IDS[:4]
PAYLOAD_FILES = frozenset({
    "cases.json", "generation.json", "schedules.json", "reference.json", "validation.json",
    *("source-exports/" + name + ".qasm" for name in CASE_IDS),
    "source/reusable_bath_local_spin.py", "source/reusable_bath_local_spin_reference.py",
    "source/cases.json", "source/requirements-lock.txt", "source/LICENSE.txt",
})
INPUT_FILES = PAYLOAD_FILES | {"artifact-manifest.json", "artifact-manifest.sha256"}
DIRECT_CONTRACTS = (
    "test_frozen_cases_and_seeded_choice_coverage",
    "test_nine_emitted_receipts_match_bytes_and_complete_source_formula",
    "test_small_cache_contains_complete_reset_evolution_readout_preparations",
    "test_complete_small_qasm_roundtrips_to_actual_cached_operations",
    "test_non_power_sizes_and_maximum_n_emit_bounded_complete_sources",
    "test_repetition_projection_counts_every_complete_preparation_without_emission",
    "test_case_limits_and_plain_integer_contract_reject_before_source_or_rng",
    "test_invalid_schedules_reject_before_the_first_source_primitive",
    "test_operation_tokens_and_fixed_cases_schema_reject_invalid_inputs",
    "test_ordinary_writer_boundaries_and_existing_destination_preservation",
)
COUNTER_CAPS = {
    "source_primitives": 2_000, "fixture_source_primitives": 750, "source_bytes": 4_096,
    "cached_source_operations": 1_954, "direct_contract_calls": 10, "direct_contract_passes": 10,
    "compact_circuits": 9, "compact_sdk_primitive_visits": 1_176,
    "reference_strang_steps": 72, "reference_diagonal_halves": 144,
    "reference_xx_updates": 72, "matrix_unit_branch_outputs": 432,
    "averaged_channels": 8, "channel_powers": 8, "choi_checks": 8,
    "conditional_round_applications": 16, "tiny_density_circuits": 4,
    "tiny_density_primitive_visits": 1_954, "partial_traces": 460, "state_checks": 128,
    "full_exports": 0, "site_selection_calls": 0, "frequency_random_calls": 0,
    "thermal_bit_random_calls": 0, "notebook_runs": 0, "ode_solves": 0,
    "matrix_exponential_calls": 0, "native_queries": 0, "synthesis_calls": 0,
    "gpu_jobs": 0, "remote_jobs": 0, "statistical_trials": 0,
}
EXPECTED_COUNTS = {
    **{name: value for name, value in COUNTER_CAPS.items() if value == 0},
    "source_primitives": 1_926, "fixture_source_primitives": 750, "source_bytes": 290,
    "cached_source_operations": 1_954, "direct_contract_calls": 10, "direct_contract_passes": 10,
    "compact_circuits": 9, "compact_sdk_primitive_visits": 1_176,
    "reference_strang_steps": 72, "reference_diagonal_halves": 144,
    "reference_xx_updates": 72, "matrix_unit_branch_outputs": 432,
    "averaged_channels": 8, "channel_powers": 8, "choi_checks": 8,
    "conditional_round_applications": 16, "tiny_density_circuits": 4,
    "tiny_density_primitive_visits": 1_948, "partial_traces": 460, "state_checks": 62,
}
STAGE_SECONDS = {"admission": 15, "contracts": 30, "reference": 60, "finalize": 15}
MAX_SECONDS = 120
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_QASM_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 128 * 1024
MAX_CHECKS_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 512 * 1024
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_FILES = 64
FAILURE_RESERVE = 1024 * 1024
RSS_THRESHOLD = 1024 * 1024 * 1024
PINNED_VERSIONS = {"numpy": "2.5.4", "qiskit": "2.6.2", "psutil": "7.2.2"}
TEST_CONTEXT_MODULE = "_oqp_local_spin_test_context"
_RUN_LOCK = threading.Lock()
_MISSING = object()


class ReproductionError(RuntimeError):
    """A controlled failure code safe to include in portable output."""

    def __init__(self, code: str):
        if type(code) is not str or not re.fullmatch(r"[a-z][a-z0-9_]{0,95}", code):
            code = "reproduction_failed"
        self.code = code
        super().__init__(code)


def check_interpreter() -> None:
    """Require nonoptimized Python 3.12 in a standalone main-thread caller."""
    if sys.flags.optimize != 0:
        raise ReproductionError("optimized_python_is_not_supported")
    if sys.version_info[:2] != (3, 12):
        raise ReproductionError("python_3_12_required")
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise ReproductionError("standalone_main_thread_required")


def _plain_int(value, lower, upper, code):
    if type(value) is not int or not lower <= value <= upper:
        raise ReproductionError(code)
    return value


def _relative(name: str) -> str:
    if type(name) is not str:
        raise ReproductionError("relative_file_label_required")
    path = PurePosixPath(name)
    if (not name or name in (".", "..") or "\\" in name or ":" in name
            or path.is_absolute() or ".." in path.parts or path.as_posix() != name):
        raise ReproductionError("relative_file_label_required")
    return name


def read_regular(path, maximum: int, label: str = "input") -> bytes:
    """Read one stable, bounded, unlinked regular file without following links."""
    _plain_int(maximum, 1, MAX_INPUT_BYTES, "invalid_read_bound")
    path = Path(path).absolute()
    try:
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ReproductionError("symlink_input_refused")
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > maximum:
            raise ReproductionError("bounded_regular_input_required")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as stream:
            opened = os.fstat(stream.fileno())
            raw = stream.read(maximum + 1)
        after = path.lstat()
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_nlink)
        if identity(before) != identity(opened) or identity(opened) != identity(after):
            raise ReproductionError("input_identity_changed")
        if len(raw) != before.st_size or len(raw) > maximum:
            raise ReproductionError("input_length_changed")
        return raw
    except OSError as error:
        raise ReproductionError("input_read_failed") from error


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ReproductionError("duplicate_json_key")
        result[key] = value
    return result


def _json(raw):
    try:
        return json.loads(raw, object_pairs_hook=_unique_object,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ReproductionError("nonfinite_json_value")))
    except (UnicodeError, ValueError) as error:
        raise ReproductionError("invalid_json") from error


def _record(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _inventory(root: Path, file_limit: int, entry_limit: int = 128):
    records, directories, stack, entries = {}, set(), [root], 0
    try:
        while stack:
            directory = stack.pop()
            for path in directory.iterdir():
                entries += 1
                if entries > entry_limit:
                    raise ReproductionError("directory_entry_limit")
                info = path.lstat()
                name = path.relative_to(root).as_posix()
                if stat.S_ISDIR(info.st_mode):
                    directories.add(name)
                    stack.append(path)
                elif stat.S_ISREG(info.st_mode):
                    records[name] = info.st_size
                    if len(records) > file_limit:
                        raise ReproductionError("file_limit")
                else:
                    raise ReproductionError("unsafe_filesystem_object")
    except OSError as error:
        raise ReproductionError("filesystem_scan_failed") from error
    return records, directories


def verify_saved_inputs(records_dir, checks_path, *, guard=None) -> dict:
    """Verify the exact 21 saved files and frozen checks without importing code."""
    try:
        requested = (Path(records_dir).expanduser().absolute(), Path(checks_path).expanduser().absolute())
        if any(part.is_symlink() for path in requested for part in (path, *path.parents)):
            raise ReproductionError("symlink_input_refused")
        root = Path(records_dir).expanduser().resolve(strict=True)
        checks = Path(checks_path).expanduser().resolve(strict=True)
    except OSError as error:
        raise ReproductionError("input_location_unavailable") from error
    if not root.is_dir():
        raise ReproductionError("saved_folder_required")
    files, directories = _inventory(root, 32)
    expected_dirs = {str(PurePosixPath(name).parent) for name in INPUT_FILES if "/" in name}
    if set(files) != INPUT_FILES or directories != expected_dirs:
        raise ReproductionError("exact_saved_inventory_required")
    if sum(files.values()) > MAX_INPUT_BYTES:
        raise ReproductionError("saved_packet_byte_limit")
    manifest_raw = read_regular(root / "artifact-manifest.json", MAX_MANIFEST_BYTES)
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
        raise ReproductionError("saved_manifest_hash_differs")
    manifest = _json(manifest_raw)
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("kind") != "local-spin-bath-saved-evidence"
            or manifest.get("method_id") != METHOD_ID
            or manifest.get("source_commit") != SOURCE_COMMIT
            or manifest.get("all_components_verified") is not False
            or set(manifest.get("files", {})) != PAYLOAD_FILES):
        raise ReproductionError("saved_manifest_contract_differs")
    sidecar = read_regular(root / "artifact-manifest.sha256", 128)
    if sidecar != (MANIFEST_SHA256 + "\n").encode("ascii"):
        raise ReproductionError("saved_manifest_sidecar_differs")
    raw_files = {"artifact-manifest.json": manifest_raw, "artifact-manifest.sha256": sidecar}
    for name in sorted(PAYLOAD_FILES):
        if guard:
            guard()
        record = manifest["files"][name]
        limit = (MAX_QASM_BYTES if name.endswith(".qasm") else
                 MAX_SOURCE_BYTES if name.startswith("source/") or name == "cases.json" else
                 MAX_REPORT_BYTES)
        expected_bytes = _plain_int(record.get("bytes"), 1, limit, "saved_record_byte_bound")
        raw = read_regular(root / name, expected_bytes)
        if _record(raw) != record:
            raise ReproductionError("saved_payload_hash_differs")
        raw_files[name] = raw
    checks_raw = read_regular(checks, MAX_CHECKS_BYTES)
    if hashlib.sha256(checks_raw).hexdigest() != CHECKS_SHA256:
        raise ReproductionError("frozen_contract_source_hash_differs")
    tree = ast.parse(checks_raw, filename="reusable_bath_local_spin_checks.py")
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name.startswith("test_")]
    if tuple(node.name for node in functions) != DIRECT_CONTRACTS or any(
            node.args.args or node.args.posonlyargs or node.args.kwonlyargs
            or node.args.vararg or node.args.kwarg or node.decorator_list for node in functions):
        raise ReproductionError("frozen_contract_call_signature_differs")
    provenance = {name: _record(raw) for name, raw in sorted(raw_files.items())}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True,
                                             separators=(",", ":")).encode()).hexdigest()
    return {"root": root, "checks_path": checks, "raw_files": raw_files,
            "checks_raw": checks_raw, "files": provenance, "fingerprint": fingerprint,
            "checks_record": _record(checks_raw)}


def parse_small_qasm(encoded: bytes, n: int, operation_type) -> list:
    """Decode actual saved n1/n2 primitives with no source-generation helper."""
    _plain_int(n, 1, 2, "small_system_size_required")
    if type(encoded) is not bytes or len(encoded) > MAX_QASM_BYTES or not encoded.endswith(b"\n"):
        raise ReproductionError("bounded_qasm_bytes_required")
    try:
        lines = encoded.decode("ascii").splitlines(keepends=True)
    except UnicodeError as error:
        raise ReproductionError("ascii_qasm_required") from error
    if lines[:4] != ["OPENQASM 2.0;\n", 'include "qelib1.inc";\n',
                     f"qreg q[{n + 1}];\n", f"creg c[{n}];\n"]:
        raise ReproductionError("small_qasm_header_differs")
    if len(lines) - 4 > 1_954:
        raise ReproductionError("small_cache_operation_limit")
    number = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
    result = []
    for line in lines[4:]:
        if len(line) > 128:
            raise ReproductionError("qasm_line_byte_limit")
        angle = clbit = None
        match = re.fullmatch(rf"rz\(({number})\) q\[([0-9]+)\];\n", line)
        if match:
            name, qubits, angle = "rz", (int(match[2]),), float(match[1])
            if not math.isfinite(angle):
                raise ReproductionError("finite_qasm_angle_required")
        else:
            match = re.fullmatch(r"cx q\[([0-9]+)\],q\[([0-9]+)\];\n", line)
            if match:
                name, qubits = "cx", (int(match[1]), int(match[2]))
            else:
                match = re.fullmatch(r"measure q\[([0-9]+)\] -> c\[([0-9]+)\];\n", line)
                if match:
                    name, qubits, clbit = "measure", (int(match[1]),), int(match[2])
                else:
                    match = re.fullmatch(r"(h|x|reset) q\[([0-9]+)\];\n", line)
                    if not match:
                        raise ReproductionError("unsupported_qasm_primitive")
                    name, qubits = match[1], (int(match[2]),)
        if any(qubit > n for qubit in qubits) or (name == "cx" and qubits[0] == qubits[1]):
            raise ReproductionError("qasm_qubit_index_differs")
        if clbit is not None and clbit >= n:
            raise ReproductionError("qasm_readout_index_differs")
        result.append(operation_type(name=name, qubits=qubits, angle=angle, clbit=clbit))
    return result


class BoundedLedger:
    """Minimal caller ledger; output_dir must be an owned, empty directory.

    Counters charge admitted work.  Failure prefixes do not prove completed
    source writes.  Output scans include staged copies and transient fixtures;
    source_bytes counts only the fixture QASM.  RSS samples cover this process.
    """

    def __init__(self, output_dir, max_seconds=MAX_SECONDS, *, clock=time.monotonic,
                 rss_sampler=None, started=None, arm=None):
        self.output = Path(output_dir).absolute()
        self.max_seconds = _plain_int(max_seconds, 1, MAX_SECONDS, "invalid_time_bound")
        self.clock, self.started = clock, clock() if started is None else started
        self.phase, self.phase_started = "admission", self.started
        self.rss_sampler, self.arm = rss_sampler, arm
        self.counters = dict.fromkeys(COUNTER_CAPS, 0)
        self.context = {"small_source_operations": {name: [] for name in SMALL_IDS}}
        self.events, self.contract = [], None
        self.peak_rss = self.peak_bytes = self.peak_files = 0
        self.next_sample = 0.0
        if (not self.output.is_dir() or any(path.is_symlink() for path in (self.output, *self.output.parents))
                or any(self.output.iterdir())):
            raise ReproductionError("owned_empty_output_required")
        self.events_path = self.output / "events.ndjson"
        with self.events_path.open("xb"):
            pass

    def consume(self, name, amount=1):
        if name not in COUNTER_CAPS or type(amount) is not int or amount < 0:
            raise ReproductionError("invalid_counter_consumption")
        proposed = self.counters[name] + amount
        if proposed > COUNTER_CAPS[name]:
            raise ReproductionError("counter_limit")
        self.counters[name] = proposed
        self.guard()

    def scan_output(self):
        files, _ = _inventory(self.output, MAX_OUTPUT_FILES)
        total = sum(files.values())
        if total > MAX_OUTPUT_BYTES:
            raise ReproductionError("output_byte_limit")
        self.peak_bytes, self.peak_files = max(self.peak_bytes, total), max(self.peak_files, len(files))
        return total, len(files)

    def guard(self, force=False):
        now = self.clock()
        if now - self.started >= self.max_seconds or now - self.phase_started >= STAGE_SECONDS[self.phase]:
            raise ReproductionError("cooperative_time_limit")
        if force or now >= self.next_sample:
            self.next_sample = now + 0.2
            if self.rss_sampler:
                rss = self.rss_sampler()
                if type(rss) is not int or rss < 0:
                    raise ReproductionError("invalid_rss_sample")
                self.peak_rss = max(self.peak_rss, rss)
                if rss > RSS_THRESHOLD:
                    raise ReproductionError("sampled_own_rss_limit")
            self.scan_output()

    def _event(self, kind):
        value = {"kind": kind, "stage": self.phase, "elapsed_seconds": self.clock() - self.started,
                 "counters": dict(self.counters)}
        raw = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
        total, _ = self.scan_output()
        if len(raw) > 64 * 1024 or self.events_path.stat().st_size + len(raw) > FAILURE_RESERVE:
            raise ReproductionError("event_byte_limit")
        if total + len(raw) > MAX_OUTPUT_BYTES - FAILURE_RESERVE:
            raise ReproductionError("event_output_headroom")
        with self.events_path.open("ab") as stream:
            stream.write(raw)
        self.events.append(value)

    @contextmanager
    def stage(self, name):
        if name not in STAGE_SECONDS:
            raise ReproductionError("unknown_stage")
        self.phase = name
        self.phase_started = self.started if name == "admission" else self.clock()
        self.guard(force=True)
        if self.arm:
            now = self.clock()
            remaining = min(STAGE_SECONDS[name] - (now - self.phase_started),
                            self.max_seconds - (now - self.started))
            if remaining <= 0:
                raise ReproductionError("cooperative_time_limit")
            self.arm(remaining)
        self._event("stage_start")
        try:
            yield
            self.guard(force=True)
            self._event("stage_complete")
        except BaseException:
            remaining = self.max_seconds - (self.clock() - self.started)
            if remaining > 0:
                if self.arm:
                    self.arm(min(1, remaining))
                try:
                    self._event("stage_failed")
                except BaseException:
                    pass
            raise

    def write_bytes(self, name, raw, *, reserve=True):
        _relative(name)
        if type(raw) is not bytes or len(raw) > MAX_REPORT_BYTES:
            raise ReproductionError("output_file_byte_limit")
        if reserve:
            self.guard(force=True)
        total, count = self.scan_output()
        limit = MAX_OUTPUT_BYTES - (FAILURE_RESERVE if reserve else 0)
        slots = MAX_OUTPUT_FILES - (2 if reserve else 0)
        if total + len(raw) > limit or count + 1 > slots:
            raise ReproductionError("output_headroom_limit")
        target = self.output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if any(path.is_symlink() for path in (target, *target.parents)):
            raise ReproductionError("symlink_output_refused")
        try:
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as error:
            raise ReproductionError("existing_output_preserved") from error
        except OSError as error:
            raise ReproductionError("output_write_failed") from error
        if reserve:
            self.guard(force=True)

    def write_json(self, name, value, *, reserve=True):
        self.write_bytes(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode(), reserve=reserve)


@contextmanager
def _alarm_owner(max_seconds):
    enabled = hasattr(signal, "setitimer") and hasattr(signal, "SIGALRM")
    if not enabled:
        yield None
        return
    previous, timer = signal.getsignal(signal.SIGALRM), signal.getitimer(signal.ITIMER_REAL)
    if timer != (0.0, 0.0):
        raise ReproductionError("preexisting_alarm_refused")
    def alarm(signum, frame):
        raise ReproductionError("cooperative_alarm_limit")
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, min(15, max_seconds))
        yield lambda seconds: signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        signal.setitimer(signal.ITIMER_REAL, *timer)


@contextmanager
def _compiled_snapshot(name, raw, label):
    previous = sys.modules.get(name, _MISSING)
    module = types.ModuleType(name)
    module.__file__, module.__package__ = label, ""
    try:
        sys.modules[name] = module
        exec(compile(raw, label, "exec", dont_inherit=True, optimize=0), module.__dict__)
        yield module
    finally:
        if previous is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


def run_direct_contracts(checks_module, context, ledger) -> list[dict]:
    """Internal bridge for the owning replay; never call it independently.

    reproduce_saved_inputs supplies its exclusive scope, verified module and
    ledger.  The context slot is restored even when a direct assertion fails.
    """
    check_interpreter()
    previous = sys.modules.get(TEST_CONTEXT_MODULE, _MISSING)
    records = []
    try:
        sys.modules[TEST_CONTEXT_MODULE] = context
        for name in DIRECT_CONTRACTS:
            function = getattr(checks_module, name, None)
            if not callable(function):
                raise ReproductionError("contract_callable_missing")
            ledger.contract = name
            ledger.consume("direct_contract_calls")
            started = ledger.clock()
            function()
            ledger.consume("direct_contract_passes")
            records.append({"name": name, "status": "passed", "seconds": ledger.clock() - started})
        ledger.contract = None
        return records
    finally:
        if previous is _MISSING:
            sys.modules.pop(TEST_CONTEXT_MODULE, None)
        else:
            sys.modules[TEST_CONTEXT_MODULE] = previous


def _output_location(records_root, output_dir):
    requested = Path(output_dir).expanduser().absolute()
    if os.path.lexists(requested):
        raise ReproductionError("existing_output_preserved")
    if any(path.is_symlink() for path in (requested, *requested.parents)):
        raise ReproductionError("symlink_output_refused")
    output = Path(output_dir).expanduser().resolve(strict=False)
    if output.is_relative_to(records_root) or records_root.is_relative_to(output):
        raise ReproductionError("output_overlaps_saved_inputs")
    if os.path.lexists(output):
        raise ReproductionError("existing_output_preserved")
    if any(path.is_symlink() for path in (output, *output.parents)):
        raise ReproductionError("symlink_output_refused")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.mkdir()
    except OSError as error:
        raise ReproductionError("fresh_output_creation_failed") from error
    return output


def _portable_reference(report):
    # Successful reports contain only scientific labels/values and versions.
    # On failures, discard free-form library error text, which can name a user's
    # installation.  Numeric diagnostics remain unchanged.
    if "error" in report:
        report = dict(report)
        error = dict(report["error"])
        message = error.pop("message", None)
        error["message_code"] = (message if error.get("type") == "ReproductionError"
                                 and type(message) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,95}", message)
                                 else "finite_reference_failed")
        report["error"] = error
    return report


def reproduce_saved_inputs(records_dir, checks_path, output_dir, *, max_seconds=MAX_SECONDS) -> dict:
    """Run one bounded public interface replay into a fresh separate folder.

    The returned status is interface evidence.  Existing algorithm records are
    immutable inputs; this call does not create a new algorithm-accuracy claim.
    Only normal standalone/main-thread calls are supported because the frozen
    assertions temporarily patch shared Python library functions.
    """
    started = time.monotonic()
    check_interpreter()
    _plain_int(max_seconds, 1, MAX_SECONDS, "invalid_time_bound")
    if not _RUN_LOCK.acquire(blocking=False):
        raise ReproductionError("reentrant_reproduction_refused")
    ledger = None
    try:
        with _alarm_owner(max_seconds) as arm:
            requested = Path(records_dir).expanduser().absolute()
            if any(path.is_symlink() for path in (requested, *requested.parents)):
                raise ReproductionError("symlink_input_refused")
            root = Path(records_dir).expanduser().resolve(strict=True)
            output = _output_location(root, output_dir)
            ledger = BoundedLedger(output, max_seconds, started=started, arm=arm)
            pending_reference = None
            try:
                with ExitStack() as modules:
                    with ledger.stage("admission"):
                        harness_source = {
                            "file": "reusable_bath_local_spin_reproduce.py",
                            **_record(read_regular(__file__, MAX_SOURCE_BYTES, "harness_source")),
                        }
                        inputs = verify_saved_inputs(root, checks_path, guard=ledger.guard)
                        versions = {name: importlib.metadata.version(name) for name in PINNED_VERSIONS}
                        if versions != PINNED_VERSIONS:
                            raise ReproductionError("pinned_runtime_versions_required")
                        import psutil
                        process = psutil.Process()
                        ledger.rss_sampler = lambda: process.memory_info().rss
                        ledger.guard(force=True)
                        for case_id in CASE_IDS:
                            name = "source-exports/" + case_id + ".qasm"
                            ledger.write_bytes("input/" + name, inputs["raw_files"][name])
                        ledger.write_bytes("input/cases.json", inputs["raw_files"]["cases.json"])
                        provenance = {
                            "schema_version": 1, "interface_id": INTERFACE_ID, "method_id": METHOD_ID,
                            "algorithm_source_commit": SOURCE_COMMIT, "accepted_public_commit": PUBLIC_COMMIT,
                            "manifest_sha256": MANIFEST_SHA256, "files": inputs["files"],
                            "harness_source": harness_source,
                            "checks": {"file": "reusable_bath_local_spin_checks.py", **inputs["checks_record"]},
                            "input_fingerprint_before": inputs["fingerprint"],
                            "environment": {"python": ".".join(map(str, sys.version_info[:3])), **versions},
                            "source_copies": 9, "full_preparations_regenerated": 0,
                            "scientific_rng_calls": 0, "filename_entropy_excluded": True,
                            "all_components_verified": False,
                        }
                        ledger.write_json("provenance.json", provenance)
                        generator = modules.enter_context(_compiled_snapshot(
                            "_openquantum_reproduction_generator",
                            inputs["raw_files"]["source/reusable_bath_local_spin.py"],
                            "source/reusable_bath_local_spin.py"))
                        reference = modules.enter_context(_compiled_snapshot(
                            "_openquantum_reproduction_reference",
                            inputs["raw_files"]["source/reusable_bath_local_spin_reference.py"],
                            "source/reusable_bath_local_spin_reference.py"))
                        cases = generator.load_cases(output / "input/cases.json")
                        schedules = _json(inputs["raw_files"]["schedules.json"])["schedules"]
                        generation = _json(inputs["raw_files"]["generation.json"])
                        receipts = {item["case_id"]: item["receipt"] for item in generation["cases"]}
                        if tuple(case.case_id for case in cases) != CASE_IDS:
                            raise ReproductionError("frozen_case_order_differs")
                        for case in cases[:4]:
                            operations = parse_small_qasm(
                                inputs["raw_files"]["source-exports/" + case.case_id + ".qasm"],
                                case.n, generator.Operation)
                            ledger.consume("cached_source_operations", len(operations))
                            ledger.context["small_source_operations"][case.case_id] = operations
                        (output / "cache/source-contracts").mkdir(parents=True)
                        context = types.ModuleType(TEST_CONTEXT_MODULE)
                        for key, value in {
                            "generator": generator, "ledger": ledger, "cases": cases,
                            "schedules": schedules, "receipts": receipts, "output": output,
                            "source_dir": output / "input/source-exports",
                            "source_contracts_dir": output / "cache/source-contracts",
                            "small_source_operations": ledger.context["small_source_operations"],
                            "case_file": output / "input/cases.json",
                        }.items():
                            setattr(context, key, value)
                    with ledger.stage("contracts"):
                        checks = modules.enter_context(_compiled_snapshot(
                            "_openquantum_reproduction_contracts", inputs["checks_raw"],
                            "reusable_bath_local_spin_checks.py"))
                        contracts = run_direct_contracts(checks, context, ledger)
                        ledger.write_json("source-contracts.json", {
                            "schema_version": 1, "status": "direct_assertion_contracts_passed",
                            "direct_calls": len(contracts), "pytest_phase_reports": 0,
                            "contracts": contracts, "all_components_verified": False})
                    with ledger.stage("reference"):
                        report = reference.validate_small_cases(cases[:4],
                            {name: schedules[name] for name in SMALL_IDS}, generator, ledger)
                        pending_reference = _portable_reference(report)
                        if report.get("validation_passed") is not True:
                            raise ReproductionError("finite_reference_failed")
                        ledger.write_json("finite-reference.json", pending_reference)
                with ledger.stage("finalize"):
                    after = verify_saved_inputs(root, checks_path, guard=ledger.guard)
                    if _record(read_regular(__file__, MAX_SOURCE_BYTES, "harness_source")) != {
                            key: harness_source[key] for key in ("bytes", "sha256")}:
                        raise ReproductionError("harness_source_changed")
                    if after["fingerprint"] != inputs["fingerprint"] or after["checks_record"] != inputs["checks_record"]:
                        raise ReproductionError("saved_inputs_changed")
                    if ledger.counters != EXPECTED_COUNTS:
                        raise ReproductionError("completed_counter_totals_differ")
                    summary = {
                        "schema_version": 1, "interface_id": INTERFACE_ID,
                        "status": "interface_replay_completed", "counters": dict(ledger.counters),
                        "input_fingerprint_before": inputs["fingerprint"],
                        "input_fingerprint_after": after["fingerprint"],
                        "harness_source": harness_source,
                        "elapsed_seconds": time.monotonic() - started,
                        "own_process_sampled_peak_rss_bytes": ledger.peak_rss,
                        "observed_output_peak_bytes": ledger.peak_bytes,
                        "observed_output_peak_files": ledger.peak_files,
                        "max_seconds": max_seconds, "stage_seconds": STAGE_SECONDS,
                        "output_byte_limit": MAX_OUTPUT_BYTES, "output_file_limit": MAX_OUTPUT_FILES,
                        "completion_scope": "CLI exit and independent saved-output review required",
                        "control_scope": "cooperative alarms/guards, sampled own RSS and output checkpoints; no hard host quota or process isolation proof",
                        "outside_domain": "interpreter/CLI startup before reproduce_saved_inputs, terminal-stream delivery time/storage after API return, library internals and incidental writes outside this output folder are excluded; a caller-owned outer observer must bound the whole CLI",
                        "all_components_verified": False, "new_algorithm_result": False,
                        "algorithm_accuracy_certificate": False,
                        "mixing_certificate": False, "continuum_bridge": "unavailable",
                        "physical_budgets": "uncomputed", "retry_performed": False,
                    }
                    ledger.write_json("reproduction.json", summary)
                ledger.guard(force=True)
                return summary
            except BaseException as error:
                remaining = max_seconds - (time.monotonic() - started)
                if remaining > 0:
                    terminal_deadline = min(time.monotonic() + 5, started + max_seconds)
                    remaining_io = terminal_deadline - time.monotonic()
                    if remaining_io <= 0:
                        raise
                    if arm:
                        arm(remaining_io)
                    partial_reference_file = None
                    if pending_reference is not None:
                        name = ("finite-reference.partial.json" if os.path.lexists(output / "finite-reference.json")
                                else "finite-reference.json")
                        try:
                            ledger.write_json(name, pending_reference, reserve=False)
                            partial_reference_file = name
                        except BaseException:
                            pass
                    remaining_io = terminal_deadline - time.monotonic()
                    if remaining_io <= 0:
                        raise
                    if arm:
                        arm(remaining_io)
                    try:
                        ledger.write_json("error.json", {
                            "schema_version": 1, "interface_id": INTERFACE_ID, "status": "interface_replay_failed",
                            "error_type": type(error).__name__,
                            "message_code": error.code if isinstance(error, ReproductionError) else "caller_or_contract_failed",
                            "stage": ledger.phase, "contract": ledger.contract,
                            "partial_reference_file": partial_reference_file,
                            "counters": dict(ledger.counters), "all_components_verified": False,
                            "retry_performed": False}, reserve=False)
                    except BaseException:
                        pass
                raise
    finally:
        _RUN_LOCK.release()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--checks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-seconds", type=int, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    try:
        summary = reproduce_saved_inputs(args.records_dir, args.checks, args.output_dir,
                                         max_seconds=args.max_seconds)
    except BaseException as error:
        print(json.dumps({"status": "interface_replay_failed", "error_type": type(error).__name__,
                          "message_code": error.code if isinstance(error, ReproductionError) else "caller_failed",
                          "all_components_verified": False}))
        return 1
    print(json.dumps({"status": summary["status"], "all_components_verified": False,
                      "direct_contract_calls": summary["counters"]["direct_contract_calls"],
                      "finite_reference_passed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
