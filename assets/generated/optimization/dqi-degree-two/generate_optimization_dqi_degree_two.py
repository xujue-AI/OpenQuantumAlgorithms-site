#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export one bounded degree-two graph DQI source circuit, without simulation.

This standard-library wrapper prepares one immutable graph plan and consumes
one primitive stream to produce both QASM and NDJSON. A complete export is
generation-only evidence; it requires an observed zero CLI exit or normal API
return, and supplies no coherent, sampling, synthesis or physical result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
CORE_RELATIVE = "src/openquantum/optimization_dqi_degree_two.py"
SCRIPT_RELATIVE = "scripts/generate_optimization_dqi_degree_two.py"
SOURCE_READ_CAP = 512 * 1024
INPUT_READ_CAP = 64 * 1024
MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_SECONDS = 30.0
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
PRIMITIVES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z", "reset", "measure", "barrier"})
STAGE_GATES = {
    "reset": {"reset"}, "preparation": {"h", "cx", "t", "tdg", "x", "ry"},
    "sign": {"z"}, "syndrome": {"cx"}, "decoder": {"h", "cx", "t", "tdg", "x"},
    "hadamard": {"h"}, "readout": {"measure"},
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _expected_sha(value, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " must be a lowercase SHA-256")
    return value


def _integer(value, hard: int, label: str) -> int:
    if type(value) is not int or not 1 <= value <= hard:
        raise ValueError(label + " must be a positive integer within its hard ceiling")
    return value


def _seconds(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= MAX_SECONDS:
        raise ValueError("Time allowance must be finite and in (0,30]")
    return float(value)


def _read_regular(path: Path, cap: int, expected: str | None = None) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
        raise ValueError("Source/input must be a bounded ordinary regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("Source/input changed file kind during read")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(cap + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(opened) or identity(opened) != identity(after) or len(raw) != after.st_size or len(raw) > cap:
        raise ValueError("Source/input bytes changed or exceeded the read bound")
    if expected is not None and _sha(raw) != expected:
        raise ValueError("Source/input fingerprint does not match")
    return raw


def _json(raw: bytes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON object key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("Nonfinite JSON constants are excluded")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


class _HardDeadline:
    """CLI alarm; API callers restore their prior process timer on return."""
    def __init__(self, seconds: float, keep: bool):
        self.seconds, self.keep = _seconds(seconds), keep

    def __enter__(self):
        if not all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer", "getitimer")):
            raise RuntimeError("This bounded CLI requires a POSIX process alarm")
        self.started = time.monotonic()
        self.old_handler = signal.getsignal(signal.SIGALRM)
        self.old_delay, self.old_interval = signal.getitimer(signal.ITIMER_REAL)
        effective = min(self.seconds, self.old_delay) if self.old_delay > 0 else self.seconds
        self.deadline = self.started + effective
        def elapsed(_signum, _frame):
            raise TimeoutError("The bounded generation process deadline elapsed")
        signal.signal(signal.SIGALRM, elapsed)
        signal.setitimer(signal.ITIMER_REAL, effective)
        return self

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("The bounded generation deadline elapsed")

    def __exit__(self, *_error):
        if not self.keep:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old_handler)
            remaining = max(0.0, self.old_delay - (time.monotonic() - self.started))
            if self.old_delay > 0 and remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining, self.old_interval)


def _load_core(expected: str | None):
    path = ROOT / CORE_RELATIVE
    raw = _read_regular(path, SOURCE_READ_CAP, expected)
    name = "_openquantum_degree_two_generator_core"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ValueError("Cannot construct the standalone source module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Compile the checked snapshot rather than asking a loader to reread it.
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, raw


def _input(args, core):
    if args.case == "c5":
        vertices, edges, signs = core.c5_input()
        file_pin = None
    elif args.case == "figure-eight":
        vertices, edges, signs = core.figure_eight_input()
        file_pin = None
    else:
        raw = _read_regular(args.graph_json, INPUT_READ_CAP, _expected_sha(args.input_sha256, "Input pin"))
        value = _json(raw)
        if type(value) is not dict or set(value) != {"schema", "vertices", "edges", "signs"} or value["schema"] != "openquantum.dqi.degree_two_input.v1":
            raise ValueError("Graph JSON requires its degree-two input schema, vertices, edges and signs")
        if type(value["vertices"]) is not int or not 5 <= value["vertices"] <= 65:
            raise ValueError("Graph vertex labels require a bounded integer vertex count")
        if type(value["edges"]) is not list or not 5 <= len(value["edges"]) <= 64 or type(value["signs"]) is not list or len(value["signs"]) != len(value["edges"]):
            raise ValueError("Graph edge and sign lists exceed their fixed input bounds")
        if any(type(edge) is not list or len(edge) != 2 or any(type(v) is not int for v in edge) for edge in value["edges"]) or any(type(v) is not int or v not in (0, 1) for v in value["signs"]):
            raise ValueError("Graph input requires integer endpoint pairs and binary integer signs")
        vertices = value["vertices"]
        edges = tuple(tuple(edge) if type(edge) is list else edge for edge in value["edges"])
        signs = tuple(value["signs"])
        file_pin = _pin(raw)
    if args.graph_json is None and args.input_sha256 is not None:
        raise ValueError("An input byte pin applies only to --graph-json")
    value = {"vertices": vertices, "edges": [list(edge) for edge in edges], "signs": list(signs)}
    return vertices, edges, signs, value, file_pin


def _fresh_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ValueError("Output directory must be fresh")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Output parent is not a directory")
    result = parent / path.name
    result.mkdir(mode=0o700)
    return result


class _OutputBudget:
    def __init__(self, cap: int):
        self.cap, self.total = _integer(cap, MAX_EXPORT_BYTES, "Output byte limit"), 0

    def charge(self, raw: bytes):
        if self.total + len(raw) > self.cap:
            raise ValueError("Combined source export exceeds its byte limit")
        self.total += len(raw)


def _open_exclusive(path: Path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    return os.fdopen(fd, "wb")


def _write_file(path: Path, raw: bytes, budget: _OutputBudget, check):
    check()
    budget.charge(raw)
    with _open_exclusive(path) as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    check()


def _qasm_record(record: dict, qubits: int, classical: int) -> bytes:
    gate = record.get("gate")
    wires = record.get("qubits")
    if type(gate) is not str or gate not in PRIMITIVES or type(wires) is not list or any(type(w) is not int or not 0 <= w < qubits for w in wires):
        raise ValueError("Malformed primitive record")
    arity = 2 if gate == "cx" else 1
    if gate == "barrier":
        if wires != list(range(qubits)) or set(record) != {"gate", "qubits"}:
            raise ValueError("Every stage barrier must name the full source register")
        return b"barrier q;\n"
    if len(wires) != arity or len(set(wires)) != len(wires):
        raise ValueError("Primitive arity or distinct-wire contract failed")
    args = ",".join(f"q[{w}]" for w in wires)
    if gate == "measure":
        cbits = record.get("classical")
        if set(record) != {"gate", "qubits", "classical"} or type(cbits) is not list or len(cbits) != 1 or type(cbits[0]) is not int or not 0 <= cbits[0] < classical:
            raise ValueError("Measurement must name one bounded classical bit")
        return f"measure {args} -> c[{cbits[0]}];\n".encode("ascii")
    if gate == "ry":
        params = record.get("parameters_hex")
        if set(record) != {"gate", "qubits", "parameters_hex"} or type(params) is not list or len(params) != 1 or type(params[0]) is not str or len(params[0]) > 80:
            raise ValueError("RY must retain one exact binary angle")
        angle = float.fromhex(params[0])
        if not math.isfinite(angle) or angle.hex() != params[0]:
            raise ValueError("RY angle must be finite and canonically encoded")
        text = format(angle, ".17g")
        if float(text).hex() != angle.hex():
            raise ValueError("QASM angle does not round-trip its binary source value")
        return f"ry({text}) {args};\n".encode("ascii")
    if set(record) != {"gate", "qubits"}:
        raise ValueError("Unexpected primitive fields")
    return f"{gate} {args};\n".encode("ascii")


def _generate(args, guard):
    own = _read_regular(Path(__file__), SOURCE_READ_CAP, _expected_sha(args.source_sha256, "Script pin"))
    core, core_raw = _load_core(_expected_sha(args.module_sha256, "Module pin"))
    guard.check()
    vertices, edges, signs, input_value, input_file_pin = _input(args, core)
    limits = core.DegreeTwoLimits(max_n=args.max_n, max_m=args.max_m,
                                 max_qubits=args.max_qubits, max_operations=args.max_operations,
                                 max_export_bytes=args.max_output_bytes,
                                 api_wall_seconds=args.time_limit_seconds)
    # Classical graph/code/lookup construction happens exactly once.
    plan = core.make_plan(vertices, edges, signs, limits, deadline=guard.deadline)
    ledger = core.plan_summary(plan)
    guard.check()
    output = _fresh_directory(args.output_dir)
    budget = _OutputBudget(args.max_output_bytes)
    ndjson_pin, qasm_pin = {"bytes": 0}, {"bytes": 0}
    ndjson_hash, qasm_hash = hashlib.sha256(), hashlib.sha256()
    actual = Counter()
    ranges, stage_index, stage_start, total = {}, 0, 0, 0
    header = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{ledger["source_wires"]}];\ncreg c[{ledger["n"]}];\n'.encode("ascii")
    with _open_exclusive(output / "operations.ndjson") as operations, _open_exclusive(output / "circuit.qasm") as qasm:
        budget.charge(header)
        qasm.write(header)
        qasm_hash.update(header)
        qasm_pin["bytes"] += len(header)
        # Both representations consume this same single iterator.
        for operation in core.iter_operations(plan, deadline=guard.deadline):
            guard.check()
            if total >= args.max_operations or stage_index >= len(STAGES):
                raise ValueError("Source stream exceeded its operation/stage limit")
            record = operation.record()
            qline = _qasm_record(record, ledger["source_wires"], ledger["n"])
            raw = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            if record["gate"] == "barrier":
                ranges[STAGES[stage_index]] = {"start": stage_start, "stop": total, "barrier": total}
                stage_index += 1
                stage_start = total + 1
            elif record["gate"] not in STAGE_GATES[STAGES[stage_index]]:
                raise ValueError("Primitive appeared outside its source stage")
            budget.charge(raw)
            budget.charge(qline)
            operations.write(raw)
            qasm.write(qline)
            ndjson_hash.update(raw)
            qasm_hash.update(qline)
            ndjson_pin["bytes"] += len(raw)
            qasm_pin["bytes"] += len(qline)
            actual[record["gate"]] += 1
            total += 1
        actual_counts = {name: actual[name] for name in ledger["source_counts"]}
        if stage_index != len(STAGES) or total != ledger["source_materialized_operations"] or actual_counts != ledger["source_counts"]:
            raise ValueError("Completed source stream disagrees with its preflight ledger")
        for stream in (operations, qasm):
            stream.flush()
            os.fsync(stream.fileno())
    guard.check()
    ndjson_pin["sha256"], qasm_pin["sha256"] = ndjson_hash.hexdigest(), qasm_hash.hexdigest()
    sources = {"script": {"path": SCRIPT_RELATIVE, **_pin(own)},
               "module": {"path": CORE_RELATIVE, **_pin(core_raw)}}
    ledger.update({"schema": "openquantum.dqi.degree_two_generation_ledger.v1",
                   "status": "generation_only_pending_invocation_acceptance", "degree": 2,
                   "stage_ranges": ranges, "observed_source_counts": actual_counts,
                   "observed_source_operations": total, "input": input_value, "sources": sources,
                   "requires_observed_cli_exit_zero": True, "simulation_performed": False,
                   "sampled_outcomes": False, "physical_resources_computed": False,
                   "formal_interval_certificate": False, "optimization_advantage_claim": False,
                   "semicircle_theorem_applied": False, "accepted_degree_one_producer_rerun": False})
    ledger_raw = _json_bytes(ledger)
    if len(ledger_raw) > SOURCE_READ_CAP:
        raise ValueError("Source ledger exceeds its bounded metadata allowance")
    _write_file(output / "ledger.json", ledger_raw, budget, guard.check)
    manifest = {"schema": "openquantum.dqi.degree_two_source_manifest.v1",
                "status": "complete_generation_component_pending_invocation_acceptance",
                "scope": "generation_only", "input": input_value,
                "input_file_pin": input_file_pin, "sources": sources,
                "files": {"operations.ndjson": ndjson_pin, "circuit.qasm": qasm_pin, "ledger.json": _pin(ledger_raw)},
                "limits": {"time_limit_seconds": args.time_limit_seconds, "max_n": args.max_n,
                           "max_m": args.max_m, "max_qubits": args.max_qubits,
                           "max_operations": args.max_operations, "max_output_bytes": args.max_output_bytes},
                "requires_observed_cli_exit_zero": True, "simulation_performed": False,
                "sampled_outcomes": False, "physical_resources_computed": False,
                "formal_interval_certificate": False, "optimization_advantage_claim": False,
                "publication_complete": False}
    manifest_raw = _json_bytes(manifest)
    if len(manifest_raw) > INPUT_READ_CAP:
        raise ValueError("Source manifest exceeds its bounded metadata allowance")
    _write_file(output / "manifest.json", manifest_raw, budget, guard.check)
    expected = {"operations.ndjson", "circuit.qasm", "ledger.json", "manifest.json"}
    seen = set()
    with os.scandir(output) as members:
        for member in members:
            guard.check()
            if member.name not in expected or len(seen) >= len(expected) or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("Unexpected file in the source export")
            seen.add(member.name)
    if seen != expected:
        raise ValueError("Source export has missing members")
    for name, recorded_pin in {**manifest["files"], "manifest.json": _pin(manifest_raw)}.items():
        guard.check()
        retained = _read_regular(output / name, args.max_output_bytes, recorded_pin["sha256"])
        if _pin(retained) != recorded_pin:
            raise ValueError("Retained source export changed before normal return")
    guard.check()
    return {"status": manifest["status"], "scope": "generation_only", "files": 4,
            "bytes": budget.total, "manifest_sha256": _sha(manifest_raw),
            "source_operations": total, "source_wires": ledger["source_wires"],
            "requires_observed_cli_exit_zero": True, "simulation_performed": False,
            "publication_complete": False}


def main(argv=None, *, _keep_process_alarm=False):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--case", choices=("c5", "figure-eight"))
    inputs.add_argument("--graph-json", type=Path)
    parser.add_argument("--input-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha256")
    parser.add_argument("--module-sha256")
    parser.add_argument("--max-n", type=int, default=64)
    parser.add_argument("--max-m", type=int, default=64)
    parser.add_argument("--max-qubits", type=int, default=192)
    parser.add_argument("--max-operations", type=int, default=200000)
    parser.add_argument("--max-output-bytes", type=int, default=MAX_EXPORT_BYTES)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        args.max_n = _integer(args.max_n, 64, "Gauge-variable limit")
        args.max_m = _integer(args.max_m, 64, "Edge limit")
        args.max_qubits = _integer(args.max_qubits, 192, "Wire limit")
        args.max_operations = _integer(args.max_operations, 200000, "Operation limit")
        args.max_output_bytes = _integer(args.max_output_bytes, MAX_EXPORT_BYTES, "Output byte limit")
        args.time_limit_seconds = _seconds(args.time_limit_seconds)
        with _HardDeadline(args.time_limit_seconds, _keep_process_alarm) as guard:
            result = _generate(args, guard)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed_generation_component", "error_class": type(error).__name__,
                          "reason": str(error)[:512], "retry": False, "simulation_performed": False,
                          "publication_complete": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(_keep_process_alarm=True))
