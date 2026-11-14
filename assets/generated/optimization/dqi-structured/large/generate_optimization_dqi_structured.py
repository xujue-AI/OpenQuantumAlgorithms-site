#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stream an explicit structured degree-one DQI circuit, without simulation.

Run beside optimization_dqi_structured.py and its lock/license, or from the
repository scripts directory. The scalar n selects exactly B=[I_n; ones] and
v=(0,...,0,1). Materialization permits n<=64 under independently tightening
wire, operation, byte and time limits; this is not a large-state simulator.
The emitted QASM includes real preparation, coherent decoding, resets and
readouts. RY rotations remain unsynthesized and physical accounting is pending.

The normal API is cooperative between bounded operations and file phases.
The POSIX CLI owns an entry-anchored 10-second default/30-second maximum hard
alarm. Timeout exits 124; partial exports retain an incomplete record. An
export-complete receipt remains provisional until the CLI exits zero or the
API returns normally. File status alone cannot accept an invocation, and it
never establishes coherent-validation evidence. Generation can
succeed when the optional exact-ideal repetition target exceeds its draw cap;
that accounting is explicitly marked cap_exceeded.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import types


DEFAULT_SECONDS = 10.0
MAX_SECONDS = 30.0
MAX_OUTPUT_BYTES = 32 << 20
MAX_SOURCE_BYTES = 1 << 20
MODULE_NAME = "optimization_dqi_structured.py"
GENERATOR_NAME = "generate_optimization_dqi_structured.py"
QASM_NAME = "dqi-structured.qasm"
OPERATIONS_NAME = "operations.ndjson"
RECEIPT_NAME = "dqi-structured-generation.json"


class GenerationTimeout(TimeoutError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("time_limit_seconds must be a finite number in (0, 30]")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("time_limit_seconds must be a finite number in (0, 30]") from error
    if not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("time_limit_seconds must be a finite number in (0, 30]")
    return value


def _output_cap(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_OUTPUT_BYTES:
        raise ValueError("max_output_bytes must be an integer in 1..33554432")
    return value


def _check(deadline: float):
    if time.monotonic() >= deadline:
        raise GenerationTimeout("Structured DQI generation deadline expired")


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected an ordinary public source, lock or license file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("A public source/download file exceeds its one-MiB cap")
    return raw


def _first(candidates: tuple[Path, ...]) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Keep the complete public structured DQI module, generator, lock and license together")


def _sources() -> tuple[object, dict[str, bytes]]:
    here = Path(__file__).resolve().parent
    module_path = _first((here / MODULE_NAME, here.parent / "src" / "openquantum" / MODULE_NAME))
    lock_path = _first((here / "requirements-lock.txt", here.parent / "requirements-lock.txt"))
    license_path = _first((here / "LICENSE.txt", here / "LICENSE", here.parent / "LICENSE"))
    payloads = {MODULE_NAME: _read(module_path), GENERATOR_NAME: _read(Path(__file__).resolve()),
                "requirements-lock.txt": _read(lock_path), "LICENSE.txt": _read(license_path)}
    # Execute exactly the bounded buffer that is hashed/copied. Do not import the
    # repository package initializer (which has its own normal dependencies).
    name = "_structured_dqi_generation_core_" + _sha(payloads[MODULE_NAME])[:16]
    module = types.ModuleType(name)
    module.__file__, module.__package__ = str(module_path), ""
    sys.modules[name] = module
    exec(compile(payloads[MODULE_NAME], str(module_path), "exec"), module.__dict__)
    return module, payloads


def _normalized_limits(core, limits):
    if limits is None:
        return core.StructuredDQILimits()
    names = ("max_n", "max_qubits", "max_operations")
    if type(limits) is dict:
        if len(limits) != len(names) or set(limits) != set(names):
            raise ValueError("A limits mapping requires max_n, max_qubits and max_operations")
        values = limits
    else:
        if type(limits).__name__ != "StructuredDQILimits":
            raise ValueError("limits must be a StructuredDQILimits instance or exact limits mapping")
        values = {name: getattr(limits, name) for name in names}
    return core.StructuredDQILimits(**values)


def _sync_directory(directory: Path):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class _Budget:
    def __init__(self, cap: int, reserve: int, deadline: float):
        if reserve >= cap:
            raise ValueError("The output cap cannot accommodate its receipt reserve")
        self.cap, self.reserve, self.deadline, self.written = cap, reserve, deadline, 0

    def write(self, stream, raw: bytes):
        _check(self.deadline)
        if self.written + len(raw) + self.reserve > self.cap:
            raise ValueError("Structured DQI output-byte cap exceeded")
        size = stream.write(raw)
        if size != len(raw):
            raise OSError("Incomplete structured DQI output write")
        self.written += size
        _check(self.deadline)


def _write_file(path: Path, raw: bytes, budget: _Budget):
    with path.open("xb", buffering=0) as stream:
        budget.write(stream, raw)
        os.fsync(stream.fileno())
    _check(budget.deadline)


def _write_receipt(path: Path, raw: bytes, deadline: float):
    _check(deadline)
    with path.open("xb", buffering=0) as stream:
        size = stream.write(raw)
        if size != len(raw):
            raise OSError("Incomplete generation receipt write")
        os.fsync(stream.fileno())
    _check(deadline)
    _sync_directory(path.parent)
    _check(deadline)


def _generate(output_dir, n, seconds, limits, byte_cap, delta_draw, max_draws,
              started: float, hard_alarm: bool) -> dict:
    seconds, byte_cap = _seconds(seconds), _output_cap(byte_cap)
    deadline = started + seconds
    _check(deadline)
    core, payloads = _sources()
    _check(deadline)
    limits = _normalized_limits(core, limits)
    program = core.structured_program(n, limits)
    ledger = program.ledger()
    repetition = core.repetition_plan(n, delta_draw, max_draws)
    _check(deadline)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Choose a fresh output directory with an existing parent")
    file_names = (*payloads, QASM_NAME, OPERATIONS_NAME)
    record = {"schema": "openquantum.dqi.structured_generation.v1", "status": "generation_only_exports_complete",
              "scope": "actual streamed primitives only; no coherent validation or quantum sampling",
              "completion_is_provisional": True,
              "invocation_acceptance": "requires raw CLI exit zero or normal API return; file status alone is insufficient",
              "coherent_validation_performed": False, "simulation_performed": False,
              "n": n, "m": n + 1, "ell": 1, "instance": core.family_spec(n),
              "logical_resources": ledger, "repetition_plan": repetition,
              "observed_operation_counts": ledger["counts"], "source_operations": ledger["source_operations"],
              "observed_stage_intervals": ledger["stage_intervals"],
              "operation_stream_sha256": "0" * 64,
              "readout_map": [[wire, bit] for bit, wire in enumerate(program.registers["assignment"])],
              "limits": {"max_n": limits.max_n, "max_qubits": limits.max_qubits,
                         "max_operations": limits.max_operations, "max_output_bytes": byte_cap,
                         "time_limit_seconds": seconds,
                         "time_guard": "entry-anchored POSIX CLI hard alarm plus operation/phase checks"
                         if hard_alarm else "cooperative operation/phase checks"},
              "provenance": {"source_module": MODULE_NAME, "source_sha256": _sha(payloads[MODULE_NAME]),
                             "generator_sha256": _sha(payloads[GENERATOR_NAME]),
                             "environment_lock_sha256": _sha(payloads["requirements-lock.txt"]),
                             "python": sys.version.split()[0], "generator_backend": "standard-library streaming primitive IR",
                             "upstream_code_adopted": False, "random_work_performed": False},
              "artifacts": {name: {"sha256": "0" * 64, "bytes": byte_cap} for name in file_names},
              "generated_at_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds_before_receipt": MAX_SECONDS}
    # The placeholder values bound the final fixed hash/decimal fields. An
    # additional KiB covers float formatting and the initially incomplete file
    # while its atomic replacement is being committed.
    reserve = len(_json(record)) + 1024
    budget = _Budget(byte_cap, reserve, deadline)
    if sum(len(raw) for raw in payloads.values()) + reserve >= byte_cap:
        raise ValueError("The output cap cannot accommodate its public source/download files")
    output.mkdir(exist_ok=False)
    initial = _json({"schema": record["schema"], "status": "incomplete", "n": n,
                     "scope": "generation only", "coherent_validation_performed": False})
    receipt_path = output / RECEIPT_NAME
    _write_receipt(receipt_path, initial, deadline)
    artifacts = {}
    for name, raw in payloads.items():
        _write_file(output / name, raw, budget)
        artifacts[name] = {"sha256": _sha(raw), "bytes": len(raw)}
    counts, total, readouts = Counter(), 0, []
    stage_number, stage_start, observed_stages = 0, 0, {}
    qasm_hash, operation_hash = hashlib.sha256(), hashlib.sha256()
    qasm_bytes, operation_bytes = 0, 0
    with (output / QASM_NAME).open("xb", buffering=0) as qasm, (output / OPERATIONS_NAME).open("xb", buffering=0) as operations:
        header = core.qasm_header(n).encode()
        budget.write(qasm, header)
        qasm_hash.update(header)
        qasm_bytes += len(header)
        for operation in program.operations():
            if total >= limits.max_operations:
                raise ValueError("Emitted operation cap exceeded")
            if any(wire >= 3 * n for wire in operation.qubits) or any(bit >= n for bit in operation.classical):
                raise ValueError("An emitted operation exceeds the selected registers")
            if operation.name == "barrier":
                if stage_number >= len(core.STAGES) or operation.qubits != tuple(range(3 * n)):
                    raise ValueError("Emitted stage barrier changed")
                observed_stages[core.STAGES[stage_number]] = {"start": stage_start, "stop": total,
                    "barrier_index": total, "operations": total - stage_start}
                stage_number, stage_start = stage_number + 1, total + 1
            if operation.name == "measure":
                readouts.append([operation.qubits[0], operation.classical[0]])
            qasm_raw = core.operation_to_qasm(operation).encode()
            operation_raw = (json.dumps(core.operation_to_record(operation), separators=(",", ":"), allow_nan=False) + "\n").encode()
            budget.write(qasm, qasm_raw)
            budget.write(operations, operation_raw)
            qasm_hash.update(qasm_raw)
            operation_hash.update(operation_raw)
            qasm_bytes, operation_bytes = qasm_bytes + len(qasm_raw), operation_bytes + len(operation_raw)
            counts[operation.name] += 1
            total += 1
        _check(deadline)
        os.fsync(qasm.fileno())
        os.fsync(operations.fileno())
    _check(deadline)
    if dict(counts) != ledger["counts"] or total != ledger["source_operations"] or observed_stages != ledger["stage_intervals"]:
        raise ValueError("Actual emitted primitives disagree with the source ledger")
    if readouts != record["readout_map"]:
        raise ValueError("Actual readout mapping changed")
    artifacts[QASM_NAME] = {"sha256": qasm_hash.hexdigest(), "bytes": qasm_bytes}
    artifacts[OPERATIONS_NAME] = {"sha256": operation_hash.hexdigest(), "bytes": operation_bytes}
    record.update(observed_operation_counts=dict(counts), source_operations=total,
                  observed_stage_intervals=observed_stages, readout_map=readouts,
                  operation_stream_sha256=operation_hash.hexdigest(), artifacts=artifacts,
                  elapsed_seconds_before_receipt=time.monotonic() - started)
    encoded = _json(record)
    if len(encoded) + len(initial) > reserve or budget.written + len(encoded) + len(initial) > byte_cap:
        raise ValueError("Final generation receipt exceeds its reserved output budget")
    temporary = receipt_path.with_name(receipt_path.name + ".partial")
    _write_receipt(temporary, encoded, deadline)
    os.replace(temporary, receipt_path)
    _sync_directory(output)
    _check(deadline)
    return record


def generate_structured_dqi(output_dir, n=8, *, time_limit_seconds=DEFAULT_SECONDS,
                           limits=None, max_output_bytes=MAX_OUTPUT_BYTES,
                           delta_draw=Fraction(1, 100), max_draws=1024) -> dict:
    """Generate a fresh stream and return its exports-only receipt.

    Limits may be a StructuredDQILimits instance or exact mapping with max_n,
    max_qubits and max_operations. All limits only tighten finite ceilings.
    The API checks its own clock between bounded operations/file phases; callers
    requiring a hard process deadline should use the POSIX CLI. Its optional
    ideal repetition metadata may say cap_exceeded without preventing a valid
    per-draw circuit export. A file receipt is provisional until this function
    returns normally or the CLI exits zero; post-write invocation failures do
    not become successes through a file status. There is no simulator or
    physical estimator here.
    """
    return _generate(output_dir, n, time_limit_seconds, limits, max_output_bytes,
                     delta_draw, max_draws, time.monotonic(), False)


@contextmanager
def _cli_alarm(started: float, *, restore: bool = True):
    enabled = os.name == "posix" and hasattr(signal, "ITIMER_REAL")
    previous_handler = previous_timer = None
    alarm_started = time.monotonic()
    if enabled:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
        previous_timer = signal.setitimer(signal.ITIMER_REAL, max(.001, started + MAX_SECONDS - time.monotonic()))
    try:
        _check(started + MAX_SECONDS)
        yield enabled
    finally:
        if enabled and restore:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                remaining = previous_timer[0] - (time.monotonic() - alarm_started)
                signal.setitimer(signal.ITIMER_REAL, max(.000001, remaining), previous_timer[1])


def main(argv=None, *, _keep_process_alarm: bool = False) -> int:
    with _cli_alarm(ENTRY_STARTED, restore=not _keep_process_alarm) as hard_alarm:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--n", type=int, default=8)
        parser.add_argument("--output-dir", type=Path, default=Path("dqi-structured-generated"))
        parser.add_argument("--time-limit-seconds", type=float, default=DEFAULT_SECONDS)
        parser.add_argument("--max-output-bytes", type=int, default=MAX_OUTPUT_BYTES)
        parser.add_argument("--max-n", type=int, default=64)
        parser.add_argument("--max-qubits", type=int, default=192)
        parser.add_argument("--max-operations", type=int, default=200_000)
        parser.add_argument("--delta-draw", default="1/100")
        parser.add_argument("--max-draws", type=int, default=1024)
        args = parser.parse_args(argv)
        try:
            seconds = _seconds(args.time_limit_seconds)
            deadline = ENTRY_STARTED + seconds
            _check(deadline)
            if hard_alarm:
                signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
            limits = {"max_n": args.max_n, "max_qubits": args.max_qubits, "max_operations": args.max_operations}
            record = _generate(args.output_dir, args.n, seconds, limits, args.max_output_bytes,
                               args.delta_draw, args.max_draws, ENTRY_STARTED, hard_alarm)
            print(json.dumps({"status": record["status"], "n": args.n,
                              "source_operations": record["source_operations"],
                              "coherent_validation_performed": False,
                              "repetition_status": record["repetition_plan"]["status"]}), flush=True)
            _check(deadline)
            return 0
        except (ValueError, OSError, RuntimeError, GenerationTimeout) as error:
            print("Structured DQI generation failed: " + str(error), file=sys.stderr, flush=True)
            return 1


if __name__ == "__main__":
    # Keep the standalone process deadline active through SystemExit, module
    # finalizers and output flushing. Imported main/helper calls restore timers.
    raise SystemExit(main(_keep_process_alarm=True))
