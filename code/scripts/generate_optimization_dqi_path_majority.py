#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded path-majority DQI program and component source, with opt-in.

The full program is generation-only. Its independent numerical scope belongs
only to the separate twelve-wire local decoder component validator. All counts
are source predictions until this producer emits and retains the streams. A
complete source export remains invocation-provisional until observed CLI exit
zero or ordinary API return. No state evolution, sampling, synthesis, physical
estimate, existing-evidence replay or public publication occurs here.

The scheduled POSIX alarm and cooperative checks require the reviewed outer
supervisor for strict whole-process and failure-output time bounds.
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
CORE_RELATIVE = "src/openquantum/optimization_dqi_path_majority.py"
SCRIPT_RELATIVE = "scripts/generate_optimization_dqi_path_majority.py"
SOURCE_READ_CAP = 512 * 1024
INPUT_READ_CAP = 64 * 1024
MAX_EXPORT_BYTES = 8 * 1024 * 1024
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


def _expected_sha(value, label: str) -> str:
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


def _load_core(expected: str):
    path = ROOT / CORE_RELATIVE
    raw = _read_regular(path, SOURCE_READ_CAP, expected)
    name = "_openquantum_path_majority_generator_core"
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



_INVOCATION_SPENT = False
COMPONENT_COUNTS = {"h": 68, "cx": 219, "t": 136, "tdg": 102}
OUTPUT_MEMBERS = frozenset({"program.qasm", "program.ndjson", "program-ledger.json",
                            "component.qasm", "component.ndjson", "component-ledger.json",
                            "manifest.json"})


def _input(args):
    raw = _read_regular(args.input_json, INPUT_READ_CAP, args.input_sha256)
    value = _json(raw)
    if type(value) is not dict or set(value) != {"k", "signs"}:
        raise ValueError("Input JSON requires exactly k and signs")
    if type(value["k"]) is not int or not 3 <= value["k"] <= args.max_k:
        raise ValueError("Path count must be an exact integer in [3,max-k]")
    signs = value["signs"]
    if type(signs) is not list or len(signs) != 5 * value["k"] or any(type(bit) is not int or bit not in (0, 1) for bit in signs):
        raise ValueError("Input requires exactly five binary integer signs per path")
    return value["k"], tuple(signs), value, _pin(raw)


def _component_ledger():
    # This independent structural contract has no producer-derived coherent input.
    return {"schema": "openquantum.dqi.path_majority_component_ledger.v1",
            "status": "generated_source_unvalidated", "qubits": 12,
            "syndrome_wires": [0, 1, 2, 3], "target_wires": [4, 5, 6, 7, 8],
            "scratch_wires": [9, 10, 11], "unitary_operations": 525,
            "ccx_blocks": 34, "cx_outside_ccx": 15,
            "primitive_counts": dict(COMPONENT_COUNTS),
            "promise_maximum_error_weight": 2,
            "total_extension_uses_terminal_syndrome": False,
            "arbitrary_target_xor": True, "clean_scratch_required": True,
            "full_program_validation": False}


def _export_stream(output, stem, iterator, wires, classical, budget, check,
                   operation_cap, *, stage_validation):
    actual, ranges = Counter(), {}
    stage_index, stage_start, total = 0, 0, 0
    qasm_header = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{wires}];\n'
    if classical:
        qasm_header += f'creg c[{classical}];\n'
    header = qasm_header.encode("ascii")
    nd_hash, qasm_hash = hashlib.sha256(), hashlib.sha256()
    nd_bytes, qasm_bytes = 0, len(header)
    with _open_exclusive(output / (stem + ".ndjson")) as ndjson, _open_exclusive(output / (stem + ".qasm")) as qasm:
        budget.charge(header); qasm.write(header); qasm_hash.update(header)
        for operation in iterator:
            check()
            if total >= operation_cap:
                raise ValueError("Source stream exceeded its exact operation allowance")
            record = operation.record()
            if type(record) is not dict:
                raise ValueError("Primitive serializer must return an ordinary dictionary")
            qline = _qasm_record(record, wires, classical)
            if stage_validation:
                if stage_index >= len(STAGES):
                    raise ValueError("Full source stream exceeded its seven stage boundaries")
                if record["gate"] == "barrier":
                    ranges[STAGES[stage_index]] = {"start": stage_start, "stop": total, "barrier": total}
                    stage_index += 1; stage_start = total + 1
                elif record["gate"] not in STAGE_GATES[STAGES[stage_index]]:
                    raise ValueError("Primitive appeared outside its declared full-program stage")
            elif record["gate"] not in COMPONENT_COUNTS:
                raise ValueError("Local component permits only H/CX/T/T-dagger primitives")
            raw = (json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
            if len(raw) > 8192:
                raise ValueError("One serialized primitive exceeds its byte allowance")
            budget.charge(raw); budget.charge(qline)
            ndjson.write(raw); qasm.write(qline)
            nd_hash.update(raw); qasm_hash.update(qline)
            nd_bytes += len(raw); qasm_bytes += len(qline)
            actual[record["gate"]] += 1; total += 1
        if stage_validation and stage_index != len(STAGES):
            raise ValueError("Full source stream did not complete all seven stage boundaries")
        for stream in (ndjson, qasm):
            stream.flush(); os.fsync(stream.fileno())
    check()
    return {stem + ".ndjson": {"bytes": nd_bytes, "sha256": nd_hash.hexdigest()},
            stem + ".qasm": {"bytes": qasm_bytes, "sha256": qasm_hash.hexdigest()}}, dict(actual), ranges, total


def _generate(args, guard):
    own = _read_regular(Path(__file__), SOURCE_READ_CAP, args.source_sha256)
    k, signs, input_value, input_file_pin = _input(args)
    core, core_raw = _load_core(args.module_sha256)
    guard.check()
    limits = core.PathMajorityLimits(max_k=args.max_k, max_qubits=args.max_qubits,
                                     max_operations=args.max_operations,
                                     max_export_bytes=args.max_output_bytes,
                                     api_wall_seconds=args.time_limit_seconds)
    plan = core.make_plan(k, signs, limits=limits, deadline=guard.deadline)
    ledger = core.plan_summary(plan)
    if type(ledger) is not dict or ledger.get("k") != k or ledger.get("source_wires") != 9 * k + 4 or ledger.get("n") != 4 * k + 1:
        raise ValueError("Core plan has an unexpected path-family coordinate contract")
    if type(ledger.get("source_materialized_operations")) is not int or not 1 <= ledger["source_materialized_operations"] <= args.max_operations:
        raise ValueError("Predicted full source operation count exceeds its allowance")
    output = _fresh_directory(args.output_dir)
    budget = _OutputBudget(args.max_output_bytes)
    files, actual, ranges, total = _export_stream(output, "program", core.iter_operations(plan, deadline=guard.deadline),
                                                 ledger["source_wires"], ledger["n"], budget, guard.check,
                                                 args.max_operations, stage_validation=True)
    observed = {name: actual.get(name, 0) for name in ledger["source_counts"]}
    if set(actual) - set(ledger["source_counts"]) or observed != ledger["source_counts"] or total != ledger["source_materialized_operations"]:
        raise ValueError("Retained full source disagrees with its preflight counts")
    component_files, component_actual, _, component_total = _export_stream(
        output, "component", core.iter_component_operations(deadline=guard.deadline), 12, 0,
        budget, guard.check, 525, stage_validation=False)
    if component_total != 525 or component_actual != COMPONENT_COUNTS:
        raise ValueError("Retained component must have exactly 34 CCX blocks and 15 outside CX gates")
    files.update(component_files)
    sources = {"script": {"path": SCRIPT_RELATIVE, **_pin(own)},
               "module": {"path": CORE_RELATIVE, **_pin(core_raw)}}
    ledger.update({"schema": "openquantum.dqi.path_majority_generation_ledger.v1",
                   "status": "generation_only_pending_invocation_acceptance", "input": input_value,
                   "sources": sources, "stage_ranges": ranges,
                   "observed_source_counts": observed, "observed_source_operations": total,
                   "observed_component_counts": dict(component_actual), "observed_component_operations": component_total,
                   "requires_observed_cli_exit_zero": True, "simulation_performed": False,
                   "component_numerically_validated": False, "full_program_validated": False,
                   "sampled_outcomes": False, "physical_resources_computed": False,
                   "formal_interval_certificate": False, "optimization_advantage_claim": False,
                   "old_degree_two_evidence_reexecuted": False, "publication_complete": False})
    ledger_raw, component_ledger_raw = _json_bytes(ledger), _json_bytes(_component_ledger())
    if len(ledger_raw) > SOURCE_READ_CAP or len(component_ledger_raw) > INPUT_READ_CAP:
        raise ValueError("Source ledger exceeds its bounded metadata allowance")
    _write_file(output / "program-ledger.json", ledger_raw, budget, guard.check)
    _write_file(output / "component-ledger.json", component_ledger_raw, budget, guard.check)
    files.update({"program-ledger.json": _pin(ledger_raw), "component-ledger.json": _pin(component_ledger_raw)})
    manifest = {"schema": "openquantum.dqi.path_majority_source_manifest.v1",
                "status": "complete_generation_component_pending_invocation_acceptance",
                "scope": "full_program_and_local_component_generation_only", "input": input_value,
                "input_file_pin": input_file_pin, "sources": sources, "files": files,
                "limits": {"time_limit_seconds": args.time_limit_seconds, "max_k": args.max_k,
                           "max_qubits": args.max_qubits, "max_operations": args.max_operations,
                           "max_output_bytes": args.max_output_bytes},
                "requires_observed_cli_exit_zero": True, "simulation_performed": False,
                "component_numerically_validated": False, "full_program_validated": False,
                "sampled_outcomes": False, "physical_resources_computed": False,
                "formal_interval_certificate": False, "optimization_advantage_claim": False,
                "old_degree_two_evidence_reexecuted": False, "publication_complete": False}
    manifest_raw = _json_bytes(manifest)
    if len(manifest_raw) > INPUT_READ_CAP:
        raise ValueError("Source manifest exceeds its bounded metadata allowance")
    _write_file(output / "manifest.json", manifest_raw, budget, guard.check)
    seen = set()
    with os.scandir(output) as members:
        for member in members:
            guard.check()
            if member.name in seen or len(seen) >= len(OUTPUT_MEMBERS) or member.name not in OUTPUT_MEMBERS or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("Source export contains an unexpected member")
            seen.add(member.name)
    if seen != OUTPUT_MEMBERS:
        raise ValueError("Source export has missing or extra members")
    for name, expected in {**files, "manifest.json": _pin(manifest_raw)}.items():
        guard.check()
        if _pin(_read_regular(output / name, args.max_output_bytes, expected["sha256"])) != expected:
            raise ValueError("Retained source bytes changed before normal return")
    for source in sources.values():
        guard.check()
        if _pin(_read_regular(ROOT / source["path"], SOURCE_READ_CAP, source["sha256"])) != {"bytes": source["bytes"], "sha256": source["sha256"]}:
            raise ValueError("Authenticated source snapshot changed before normal return")
    if _pin(_read_regular(args.input_json, INPUT_READ_CAP, args.input_sha256)) != input_file_pin:
        raise ValueError("Input snapshot changed before normal return")
    guard.check()
    return {"status": manifest["status"], "scope": manifest["scope"], "files": 7,
            "bytes": budget.total, "manifest_sha256": _sha(manifest_raw),
            "source_operations": total, "source_wires": ledger["source_wires"],
            "component_operations": component_total, "component_wires": 12,
            "requires_observed_cli_exit_zero": True, "simulation_performed": False,
            "component_numerically_validated": False, "full_program_validated": False,
            "retry": False, "publication_complete": False}


def main(argv=None, *, _keep_process_alarm=False):
    global _INVOCATION_SPENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--max-k", type=int, default=12)
    parser.add_argument("--max-qubits", type=int, default=112)
    parser.add_argument("--max-operations", type=int, default=20000)
    parser.add_argument("--max-output-bytes", type=int, default=MAX_EXPORT_BYTES)
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        if args.generate is not True:
            raise ValueError("Source emission requires explicit --generate opt-in")
        if _INVOCATION_SPENT:
            raise ValueError("This process already consumed its generation invocation")
        _INVOCATION_SPENT = True
        args.source_sha256 = _expected_sha(args.source_sha256, "Script pin")
        args.module_sha256 = _expected_sha(args.module_sha256, "Module pin")
        args.input_sha256 = _expected_sha(args.input_sha256, "Input pin")
        args.max_k = _integer(args.max_k, 12, "Path-count limit")
        args.max_qubits = _integer(args.max_qubits, 112, "Source-wire limit")
        args.max_operations = _integer(args.max_operations, 20000, "Operation limit")
        args.max_output_bytes = _integer(args.max_output_bytes, MAX_EXPORT_BYTES, "Output byte limit")
        args.time_limit_seconds = _seconds(args.time_limit_seconds)
        with _HardDeadline(args.time_limit_seconds, _keep_process_alarm) as guard:
            result = _generate(args, guard)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"schema": "openquantum.dqi.path_majority_failed_generation.v1",
                          "status": "failed_generation_component", "error_class": type(error).__name__,
                          "reason": str(error)[:512], "retry": False, "simulation_performed": False,
                          "component_numerically_validated": False, "full_program_validated": False,
                          "publication_complete": False}, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(_keep_process_alarm=True))
