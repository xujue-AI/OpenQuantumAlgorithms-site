#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize and export a bounded DQI circuit, without quantum simulation.

Run this script beside optimization_dqi.py, or from the repository's scripts
directory, using the supplied Python 3.12 requirements lock. The default input
is n=3, m=4, ell=1; JSON inputs remain bounded to n<=3, m<=4, ell=1 with distinct
nonzero rows. A generation receipt describes actual emitted primitives. It is
separate from the saved coherent validation in dqi-validated.json.

The CLI has its own 10-second default/30-second maximum entry deadline and a
POSIX hard alarm. The API checks the same elapsed deadline between bounded
phases; callers requiring a hard process deadline should use the CLI. Neither
entry point simulates, samples, fits, synthesizes rotations or estimates hardware.
"""
from __future__ import annotations

import time
ENTRY_STARTED = time.monotonic()

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys


DEFAULT_SECONDS = 10.0
MAX_SECONDS = 30.0
MAX_QUBITS = 9
MAX_OPERATIONS = 512
MAX_INPUT_BYTES = 16384
MAX_EXPORT_BYTES = 1 << 20
PRIMITIVES = frozenset({"h", "x", "z", "t", "tdg", "ry", "cx", "reset", "measure", "barrier"})


class GenerationTimeout(TimeoutError):
    """The standalone generation deadline expired."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("time_limit_seconds must be a finite number in (0, 30]")
    value = float(value)
    if not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("time_limit_seconds must be a finite number in (0, 30]")
    return value


def _caps(materialized_qubits, primitive_operations):
    for name, value, ceiling in (("materialized_qubits", materialized_qubits, MAX_QUBITS),
                                 ("primitive_operations", primitive_operations, MAX_OPERATIONS)):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise GenerationTimeout("DQI materialization/export deadline expired")


def _read_bounded(path, ceiling=MAX_EXPORT_BYTES) -> bytes:
    with Path(path).open("rb") as stream:
        data = stream.read(ceiling + 1)
    if len(data) > ceiling:
        raise ValueError(f"A DQI input/download exceeds its {ceiling}-byte ceiling")
    return data


@contextmanager
def _deadline(started, seconds, hard_alarm):
    deadline = started + _seconds(seconds)
    _check_deadline(deadline)
    enabled = hard_alarm and os.name == "posix" and hasattr(signal, "ITIMER_REAL")
    previous_handler = None
    previous_timer = None
    alarm_started = None
    if enabled:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GenerationTimeout("DQI materialization/export deadline expired")
        previous_handler = signal.getsignal(signal.SIGALRM)
        def expired(*_):
            raise GenerationTimeout("DQI materialization/export deadline expired")
        signal.signal(signal.SIGALRM, expired)
        alarm_started = time.monotonic()
        previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield deadline, enabled
        _check_deadline(deadline)
    finally:
        if enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                remaining_previous = previous_timer[0] - (time.monotonic() - alarm_started)
                if remaining_previous > 0:
                    signal.setitimer(signal.ITIMER_REAL, remaining_previous, previous_timer[1])


def _module_path() -> Path:
    here = Path(__file__).resolve().parent
    adjacent = here / "optimization_dqi.py"
    repository = here.parent / "src/openquantum/optimization_dqi.py"
    if adjacent.is_file():
        return adjacent
    if repository.is_file():
        return repository
    raise FileNotFoundError("Place this script beside optimization_dqi.py or in its repository scripts directory")


def _load_exact_module(path: Path, source: bytes):
    """Load the exact bytes whose digest is recorded, only when generation is called."""
    name = "_openquantum_dqi_materialization_source"
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None:
        raise RuntimeError("Cannot create the DQI source-module specification")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


def primitive_qasm(operations: list, num_qubits: int, num_clbits: int) -> str:
    """Export the listed source chronology in a bounded OpenQASM 3 primitive grammar."""
    if (type(num_qubits) is not int or not 1 <= num_qubits <= MAX_QUBITS
            or type(num_clbits) is not int or not 1 <= num_clbits <= 3
            or not isinstance(operations, list) or len(operations) > MAX_OPERATIONS):
        raise ValueError("QASM export exceeds the bounded source grammar")
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";',
             f"bit[{num_clbits}] c;", f"qubit[{num_qubits}] q;"]
    for name, qubits, clbits, params in operations:
        if (name not in PRIMITIVES or any(type(q) is not int or not 0 <= q < num_qubits for q in qubits)
                or any(type(c) is not int or not 0 <= c < num_clbits for c in clbits)):
            raise ValueError("Unknown primitive or out-of-register QASM operand")
        if name == "measure":
            if len(qubits) != 1 or len(clbits) != 1 or params:
                raise ValueError("Invalid source readout")
            lines.append(f"c[{clbits[0]}] = measure q[{qubits[0]}];")
        elif name == "barrier":
            if not qubits or clbits or params:
                raise ValueError("Invalid source barrier")
            lines.append("barrier " + ", ".join(f"q[{q}]" for q in qubits) + ";")
        elif name == "ry":
            if len(qubits) != 1 or clbits or len(params) != 1 or not math.isfinite(params[0]):
                raise ValueError("Invalid source RY")
            lines.append(f"ry({repr(float(params[0]))}) q[{qubits[0]}];")
        else:
            if len(qubits) != (2 if name == "cx" else 1) or clbits or params:
                raise ValueError("Invalid source Clifford/T/reset primitive")
            lines.append(name + " " + ", ".join(f"q[{q}]" for q in qubits) + ";")
    return "\n".join(lines) + "\n"


def _optional_download(filename: str, repository_name: str) -> bytes | None:
    here = Path(__file__).resolve().parent
    for path in (here / filename, here.parent / repository_name):
        if path.is_file():
            return _read_bounded(path)
    return None


def _generate(output_dir, matrix, signs, ell, seconds, qubit_cap, operation_cap, started,
              hard_alarm, inherited_cli_alarm=False):
    _caps(qubit_cap, operation_cap)
    if (matrix is None) != (signs is None):
        raise ValueError("Supply both matrix and signs, or neither for the fixed default")
    output = None
    receipt_path = None
    owns_output = False
    try:
        with _deadline(started, seconds, hard_alarm) as (deadline, alarm_enabled):
            output = Path(output_dir).resolve()
            receipt_path = output / "dqi-generation.json"
            _check_deadline(deadline)
            if sys.version_info[:2] != (3, 12):
                raise ValueError("Use Python 3.12 and the supplied requirements-lock.txt")
            source_path = _module_path()
            source = _read_bounded(source_path)
            _check_deadline(deadline)
            dqi = _load_exact_module(source_path, source)
            _check_deadline(deadline)
            if matrix is None:
                matrix, signs = dqi.DEFAULT_MATRIX, dqi.DEFAULT_SIGNS
            instance = dqi.validate_instance(matrix, signs, ell=ell)
            limits = dqi.DQILimits(materialized_qubits=qubit_cap, primitive_operations=operation_cap)
            circuit = dqi.build_dqi_circuit(instance["matrix"], instance["signs"], ell=ell, limits=limits)
            _check_deadline(deadline)
            operations = dqi.primitive_operations(circuit)
            ledger = dqi.logical_resources(circuit, draws=1)
            qasm = primitive_qasm(operations, circuit.num_qubits, circuit.num_clbits)
            operations_bytes = _json_bytes({"schema": "openquantum.dqi.primitive_operations.v1",
                "logical_qubits": circuit.num_qubits, "classical_bits": circuit.num_clbits,
                "operations": operations, "stages": circuit.metadata["stages"], "logical_resources": ledger})
            payloads = {"dqi.qasm": qasm.encode("utf-8"), "dqi-operations.json": operations_bytes,
                        "optimization_dqi.py": source, "generate_optimization_dqi.py": _read_bounded(__file__)}
            lock = _optional_download("requirements-lock.txt", "requirements-lock.txt")
            license_bytes = _optional_download("LICENSE.txt", "LICENSE")
            if lock is not None:
                payloads["requirements-lock.txt"] = lock
            if license_bytes is not None:
                payloads["LICENSE.txt"] = license_bytes
            _check_deadline(deadline)
            record = {"schema": "openquantum.dqi.primitive_generation.v1",
                "status": "materialized_source_ledger_only", "generation_complete": True,
                "coherent_validation_performed": False, "quantum_simulation_performed": False,
                "sampling_performed": False, "optimizer_or_fit_performed": False,
                "rotation_synthesis_performed": False, "native_or_factory_calls": False,
                "scope": "Bounded circuit materialization and primitive export only; saved coherent validation is separate",
                "instance": instance, "logical_resources": ledger,
                "source_operations": len(operations), "primitive_operations_sha256": _digest(_json_bytes(operations)),
                "readout_map": [[entry[1][0], entry[2][0]] for entry in operations if entry[0] == "measure"],
                "limits": {"n": 3, "m": 4, "ell": 1, "materialized_qubits": qubit_cap,
                           "primitive_operations": operation_cap, "artifact_bytes": MAX_EXPORT_BYTES,
                           "time_limit_seconds": float(seconds),
                           "time_guard": "POSIX CLI hard alarm plus phase checks" if alarm_enabled or inherited_cli_alarm else "cooperative API/portable CLI phase checks"},
                "provenance": {"source_module": source_path.name, "source_sha256": _digest(source),
                    "generator_sha256": _digest(payloads["generate_optimization_dqi.py"]),
                    "environment_lock_sha256": _digest(lock) if lock is not None else None,
                    "python": sys.version.split()[0], "qiskit": importlib.metadata.version("qiskit"),
                    "upstream_code_adopted": False},
                "artifacts": {name: {"sha256": _digest(data), "bytes": len(data)} for name, data in payloads.items()},
                "generated_at_utc": datetime.now(timezone.utc).isoformat()}
            record["elapsed_seconds_before_receipt"] = time.monotonic() - started
            receipt = _json_bytes(record)
            if sum(len(data) for data in payloads.values()) + len(receipt) > MAX_EXPORT_BYTES:
                raise ValueError("The materialized export exceeds its one-MiB ceiling")
            _check_deadline(deadline)
            output.mkdir(parents=True, exist_ok=False)
            owns_output = True
            for name, data in payloads.items():
                _check_deadline(deadline)
                (output / name).write_bytes(data)
            _check_deadline(deadline)
            receipt_path.write_bytes(receipt)
            _check_deadline(deadline)
            return record
    except BaseException:
        if owns_output and receipt_path is not None and receipt_path.is_file():
            receipt_path.unlink()
        raise


def generate_dqi(output_dir, matrix=None, signs=None, *, ell=1,
                 time_limit_seconds=DEFAULT_SECONDS, materialized_qubits=MAX_QUBITS,
                 primitive_operations=MAX_OPERATIONS) -> dict:
    """Generate a fresh bounded export directory and return its generation-only receipt.

    No simulator is called. Limits can only tighten the nine-wire/512-operation
    ceilings. The API deadline is cooperative across bounded import, build and
    export phases; the CLI adds a POSIX hard alarm anchored at script entry.
    """
    return _generate(output_dir, matrix, signs, ell, time_limit_seconds, materialized_qubits,
                     primitive_operations, time.monotonic(), False)


def _json_input(text):
    if len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("The JSON input exceeds 16384 bytes")
    return json.loads(text)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("dqi-generated"), help="new directory; existing data are never overwritten")
    parser.add_argument("--time-limit-seconds", type=float, default=DEFAULT_SECONDS, help="default 10, maximum 30")
    parser.add_argument("--instance-json", type=Path, help='JSON file containing {"matrix": [...], "signs": [...], "ell": 1}')
    parser.add_argument("--matrix-json", help="inline JSON binary rows; requires --signs-json")
    parser.add_argument("--signs-json", help="inline JSON binary signs; requires --matrix-json")
    args = parser.parse_args(argv)
    try:
        with _deadline(ENTRY_STARTED, args.time_limit_seconds, True) as (_, cli_alarm_enabled):
            matrix = signs = None
            ell = 1
            if args.instance_json is not None:
                if args.matrix_json is not None or args.signs_json is not None:
                    raise ValueError("Choose an instance file or inline matrix/signs")
                raw_instance = _read_bounded(args.instance_json, MAX_INPUT_BYTES)
                instance = _json_input(raw_instance.decode("utf-8"))
                if (not isinstance(instance, dict) or set(instance) - {"matrix", "signs", "ell"}
                        or "matrix" not in instance or "signs" not in instance):
                    raise ValueError("Instance JSON requires matrix/signs and optional ell=1")
                matrix, signs, ell = instance["matrix"], instance["signs"], instance.get("ell", 1)
            elif args.matrix_json is not None or args.signs_json is not None:
                if args.matrix_json is None or args.signs_json is None:
                    raise ValueError("Supply both --matrix-json and --signs-json")
                matrix, signs = _json_input(args.matrix_json), _json_input(args.signs_json)
            if ((args.instance_json is not None or args.matrix_json is not None)
                    and (matrix is None or signs is None)):
                raise ValueError("JSON matrix/signs must be explicit binary arrays")
            record = _generate(args.output_dir, matrix, signs, ell, args.time_limit_seconds,
                               MAX_QUBITS, MAX_OPERATIONS, ENTRY_STARTED, False, cli_alarm_enabled)
            print(json.dumps({"status": record["status"], "source_operations": record["source_operations"],
                              "coherent_validation_performed": False}), flush=True)
            return 0
    except (ValueError, OSError, RuntimeError, ImportError, GenerationTimeout) as error:
        print("DQI generation failed: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
