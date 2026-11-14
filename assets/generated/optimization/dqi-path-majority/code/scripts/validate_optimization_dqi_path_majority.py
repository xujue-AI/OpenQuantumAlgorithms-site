#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate and opt in to one bounded local path-majority decoder check.

The only numerical source is four syndrome wires, five arbitrary target wires
and three initially clean scratch wires: one 4096-amplitude forward pass over
525 actual H/CX/T/T-dagger primitives. The 31-wire full degree-two program is
read as generation-only source; it is never imported or evolved here. Coherent
input weights and signed output reference are independently constructed in the
verifier. Default invocation rejects before any target module is imported.

A saved result requires observed CLI exit zero. Strict whole-process and
failure-output timing relies on the reviewed outer supervisor as well as the
scheduled POSIX alarm and cooperative checks below.
"""
from __future__ import annotations

import argparse
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
SCRIPT_RELATIVE = "scripts/validate_optimization_dqi_path_majority.py"
VERIFY_RELATIVE = "src/openquantum/optimization_dqi_path_majority_verify.py"
GENERATOR_RELATIVE = "scripts/generate_optimization_dqi_path_majority.py"
CORE_RELATIVE = "src/openquantum/optimization_dqi_path_majority.py"
SOURCE_READ_CAP = 512 * 1024
MANIFEST_READ_CAP = 64 * 1024
INPUT_BYTE_CAP = 8 * 1024 * 1024
DEFAULT_OUTPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 30.0
INPUT_MEMBERS = frozenset({"program.qasm", "program.ndjson", "program-ledger.json", "component.qasm", "component.ndjson", "component-ledger.json", "manifest.json"})
PRIMITIVES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z", "reset", "measure", "barrier"})
_INVOCATION_SPENT = False


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pin(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _expected_sha(value, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(label + " requires a lowercase SHA-256")
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
            raise TimeoutError("The bounded validation process deadline elapsed")
        signal.signal(signal.SIGALRM, elapsed)
        signal.setitimer(signal.ITIMER_REAL, effective)
        return self

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("The bounded validation deadline elapsed")

    def __exit__(self, *_error):
        if not self.keep:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self.old_handler)
            remaining = max(0.0, self.old_delay - (time.monotonic() - self.started))
            if self.old_delay > 0 and remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining, self.old_interval)


def _load_verifier(expected: str):
    path = ROOT / VERIFY_RELATIVE
    raw = _read_regular(path, SOURCE_READ_CAP, expected)
    name = "_openquantum_path_majority_cli_verifier"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None:
        raise ValueError("Cannot construct the standalone verifier module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module, raw


def _pin_shape(value, cap: int):
    if type(value) is not dict or set(value) != {"bytes", "sha256"}:
        raise ValueError("File identity must contain only bytes and sha256")
    if type(value["bytes"]) is not int or not 0 <= value["bytes"] <= cap:
        raise ValueError("Pinned file size exceeds its bound")
    _expected_sha(value["sha256"], "File pin")


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



def _fresh_directory(path: Path) -> Path:
    if path.exists() or path.is_symlink():
        raise ValueError("Validation output directory must be fresh")
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("Validation output parent is not a directory")
    result = parent / path.name
    result.mkdir(mode=0o700)
    return result


def _write_exclusive(path: Path, raw: bytes, check):
    check()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    check()


def _budget_snapshot(budget) -> dict:
    if budget is None:
        return {"verification_entered": False}
    try:
        value = budget.snapshot()
        if type(value) is not dict or len(_json_bytes(value)) > 16 * 1024:
            return {"snapshot_unavailable": True}
        return value
    except Exception:
        return {"snapshot_unavailable": True}



COMPONENT_COUNTS = {"h": 68, "cx": 219, "t": 136, "tdg": 102}
PROGRAM_COUNTS = {"h": 385, "cx": 1272, "t": 744, "tdg": 558, "x": 84,
                  "ry": 57, "z": 14, "reset": 31, "measure": 13, "barrier": 7}
FIXTURE_SIGNS = (1, 1, 1, 1, 0) + (1,) * 10
RESULT_FALSE_FLAGS = ("full_program_validated", "degree_two_preparation_validated",
                      "optimization_output_validated", "physical_resources_estimated",
                      "rotation_synthesis_validated", "formal_proof_reproduced",
                      "hardware_executed", "sampling_performed", "dense_operator_constructed",
                      "notebook_executed")


def _fixed_input(value):
    if type(value) is not dict or set(value) != {"k", "signs"} or type(value["k"]) is not int or value["k"] != 3:
        raise ValueError("This component invocation accepts the frozen k=3 source fixture")
    signs = value["signs"]
    if type(signs) is not list or any(type(bit) is not int for bit in signs) or tuple(signs) != FIXTURE_SIGNS:
        raise ValueError("The source fixture signs differ from their fixed path order")


def _source_binding(value, relative):
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"} or value["path"] != relative:
        raise ValueError("Source identity names an unexpected path or extra field")
    expected = {"bytes": value["bytes"], "sha256": value["sha256"]}
    _pin_shape(expected, SOURCE_READ_CAP)
    if _pin(_read_regular(ROOT / relative, SOURCE_READ_CAP, expected["sha256"])) != expected:
        raise ValueError("Authenticated source size does not match its saved snapshot")


def _strict_same_json(value, expected):
    """Exact JSON types prevent Boolean/integer aliases in fixed wire/flag roles."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(_strict_same_json(value[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(_strict_same_json(a, b) for a, b in zip(value, expected))
    return value == expected


def _component_ledger():
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


def _read_records(raw, qasm_raw, *, wires, classical, expected_count, expected_counts, component, check):
    lines = raw.splitlines()
    if len(lines) != expected_count:
        raise ValueError("Primitive stream has an unexpected exact record count")
    records, counts = [], {gate: 0 for gate in expected_counts}
    header = f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{wires}];\n'
    if classical:
        header += f'creg c[{classical}];\n'
    reconstructed = [header.encode("ascii")]
    for line in lines:
        check()
        if len(line) > 8192:
            raise ValueError("One primitive record exceeded its bounded byte length")
        record = _json(line)
        if type(record) is not dict:
            raise ValueError("Each primitive record must be an ordinary JSON object")
        qline = _qasm_record(record, wires, classical)
        gate = record["gate"]
        if gate not in counts or (component and gate not in COMPONENT_COUNTS):
            raise ValueError("Stream contains a gate outside its exact source role")
        counts[gate] += 1; reconstructed.append(qline); records.append(record)
    if counts != expected_counts or b"".join(reconstructed) != qasm_raw:
        raise ValueError("Source primitive counts or QASM/NDJSON equivalence failed")
    return records


def _input_files(directory, expected_manifest, check):
    if not stat.S_ISDIR(directory.lstat().st_mode):
        raise ValueError("Input must be an ordinary source-export directory")
    seen = set()
    with os.scandir(directory) as members:
        for member in members:
            check()
            if member.name in seen or len(seen) >= len(INPUT_MEMBERS) or member.name not in INPUT_MEMBERS or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("Source export contains an unexpected or nonregular member")
            seen.add(member.name)
    if seen != INPUT_MEMBERS:
        raise ValueError("Source export has missing or extra members")
    raw_manifest = _read_regular(directory / "manifest.json", MANIFEST_READ_CAP, expected_manifest)
    manifest = _json(raw_manifest)
    fields = {"schema", "status", "scope", "input", "input_file_pin", "sources", "files", "limits",
              "requires_observed_cli_exit_zero", "simulation_performed", "component_numerically_validated",
              "full_program_validated", "sampled_outcomes", "physical_resources_computed",
              "formal_interval_certificate", "optimization_advantage_claim", "old_degree_two_evidence_reexecuted",
              "publication_complete"}
    if type(manifest) is not dict or set(manifest) != fields or manifest["schema"] != "openquantum.dqi.path_majority_source_manifest.v1" or manifest["status"] != "complete_generation_component_pending_invocation_acceptance" or manifest["scope"] != "full_program_and_local_component_generation_only":
        raise ValueError("Generated source manifest schema/status/closed fields changed")
    if manifest["requires_observed_cli_exit_zero"] is not True:
        raise ValueError("Generated source must remain invocation-provisional")
    for flag in fields - {"schema", "status", "scope", "input", "input_file_pin", "sources", "files", "limits", "requires_observed_cli_exit_zero"}:
        if manifest[flag] is not False:
            raise ValueError("Generated source carries an unsupported execution claim")
    _fixed_input(manifest["input"])
    _pin_shape(manifest["input_file_pin"], 64 * 1024)
    limits = manifest["limits"]
    if type(limits) is not dict or set(limits) != {"time_limit_seconds", "max_k", "max_qubits", "max_operations", "max_output_bytes"}:
        raise ValueError("Generated source limits have an unexpected shape")
    _seconds(limits["time_limit_seconds"])
    for field, lower, upper in (("max_k", 3, 12), ("max_qubits", 31, 112), ("max_operations", 3165, 20000), ("max_output_bytes", 1, INPUT_BYTE_CAP)):
        if type(limits[field]) is not int or not lower <= limits[field] <= upper:
            raise ValueError("Generated source limit is outside its admitted range")
    files = manifest["files"]
    if type(files) is not dict or set(files) != INPUT_MEMBERS - {"manifest.json"}:
        raise ValueError("Generated source member pin set changed")
    payloads, total = {}, len(raw_manifest)
    for name in sorted(files):
        check(); _pin_shape(files[name], INPUT_BYTE_CAP)
        raw = _read_regular(directory / name, INPUT_BYTE_CAP, files[name]["sha256"])
        if _pin(raw) != files[name]:
            raise ValueError("Generated member size differs from its saved pin")
        total += len(raw)
        if total > INPUT_BYTE_CAP or total > limits["max_output_bytes"]:
            raise ValueError("Combined source input exceeds its bounded export limit")
        payloads[name] = raw
    sources = manifest["sources"]
    if type(sources) is not dict or set(sources) != {"script", "module"}:
        raise ValueError("Source manifest must bind exactly generator and core snapshots")
    _source_binding(sources["script"], GENERATOR_RELATIVE)
    _source_binding(sources["module"], CORE_RELATIVE)
    ledger = _json(payloads["program-ledger.json"])
    if type(ledger) is not dict or ledger.get("schema") != "openquantum.dqi.path_majority_generation_ledger.v1" or ledger.get("status") != "generation_only_pending_invocation_acceptance" or not _strict_same_json(ledger.get("input"), manifest["input"]) or not _strict_same_json(ledger.get("sources"), sources):
        raise ValueError("Full source ledger does not match the generation manifest")
    for flag in ("simulation_performed", "component_numerically_validated", "full_program_validated", "sampled_outcomes", "physical_resources_computed", "formal_interval_certificate", "optimization_advantage_claim", "old_degree_two_evidence_reexecuted", "publication_complete"):
        if ledger.get(flag) is not False:
            raise ValueError("Full source ledger carries an unsupported validation claim")
    for field, expected in (("source_wires", 31), ("n", 13), ("m", 15), ("k", 3), ("source_unitary_primitives", 3114), ("source_materialized_operations", 3165), ("observed_source_operations", 3165), ("observed_component_operations", 525)):
        if type(ledger.get(field)) is not int or ledger[field] != expected:
            raise ValueError("Full source ledger has an unexpected fixed-fixture count")
    if ledger.get("requires_observed_cli_exit_zero") is not True or not _strict_same_json(ledger.get("source_counts"), PROGRAM_COUNTS) or not _strict_same_json(ledger.get("observed_source_counts"), PROGRAM_COUNTS) or not _strict_same_json(ledger.get("observed_component_counts"), COMPONENT_COUNTS):
        raise ValueError("Full/component source counts do not match the frozen fixture")
    # Parse full source only; no generator/core import or full state allocation.
    _read_records(payloads["program.ndjson"], payloads["program.qasm"], wires=31, classical=13,
                  expected_count=3165, expected_counts=PROGRAM_COUNTS, component=False, check=check)
    component_ledger = _json(payloads["component-ledger.json"])
    if not _strict_same_json(component_ledger, _component_ledger()):
        raise ValueError("Component ledger differs from the exact independent local role")
    records = _read_records(payloads["component.ndjson"], payloads["component.qasm"], wires=12, classical=0,
                            expected_count=525, expected_counts=COMPONENT_COUNTS, component=True, check=check)
    check()
    return records, component_ledger, manifest, _sha(raw_manifest)


def _validate(args, guard, state):
    state["deadline"] = guard.deadline
    own = _read_regular(Path(__file__), SOURCE_READ_CAP, args.source_sha256)
    records, ledger, source_manifest, input_hash = _input_files(args.input_dir, args.input_manifest_sha256, guard.check)
    verifier, verifier_raw = _load_verifier(args.module_sha256)
    sources = {"script": {"path": SCRIPT_RELATIVE, **_pin(own)},
               "module": {"path": VERIFY_RELATIVE, **_pin(verifier_raw)}}
    state.update({"sources": sources, "input_manifest_sha256": input_hash})
    guard.check(); output = _fresh_directory(args.output_dir); state["output"] = output
    budget = verifier.ExecutionBudget(execution_enabled=True); state["budget"] = budget
    result = verifier.verify_component_records(records, ledger, allow_numerical_execution=True,
                                               budget=budget, check=guard.check)
    if type(result) is not dict or result.get("schema") != "openquantum.dqi.path_majority_component_validation.v1" or result.get("status") != "passed_component_only_numerical_check":
        raise ValueError("Verifier returned an unexpected component result")
    snapshot = _budget_snapshot(budget)
    expected_counts = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
                       "boolean_decoder_mappings": 512, "primitive_unitary_applications": 525,
                       "small_ccx_reference_cases": 0, "native_estimator_calls": 0}
    if not _strict_same_json(result.get("execution_budget"), snapshot) or any(type(snapshot.get(key)) is not int or snapshot[key] != count or type(result.get(key)) is not int or result[key] != count for key, count in expected_counts.items()) or snapshot.get("spent") is not True or snapshot.get("active") is not False or snapshot.get("complete") is not True:
        raise ValueError("Successful component requires complete exact once-only role counters")
    if result.get("state_wires") != 12 or type(result.get("state_wires")) is not int or result.get("state_amplitudes") != 4096 or type(result.get("state_amplitudes")) is not int or not _strict_same_json(result.get("primitive_counts"), COMPONENT_COUNTS):
        raise ValueError("Verifier result changed its fixed component shape")
    for flag in RESULT_FALSE_FLAGS:
        if result.get(flag) is not False:
            raise ValueError("Verifier returned an unsupported full-program or external execution claim")
    guard.check()
    result["cli_component"] = {"sources": sources, "source_input_manifest_sha256": input_hash,
                               "source_input_files": source_manifest["files"], "requires_observed_cli_exit_zero": True,
                               "elapsed_seconds_before_export": time.monotonic() - guard.started,
                               "time_limit_seconds": args.time_limit_seconds, "max_output_bytes": args.max_output_bytes,
                               "budget_snapshot": snapshot, "retry": False, "publication_complete": False}
    raw = _json_bytes(result)
    manifest = {"schema": "openquantum.dqi.path_majority_validation_manifest.v1",
                "status": "complete_validation_component_pending_invocation_acceptance", "scope": "local_component_only",
                "files": {"validation.json": _pin(raw)}, "sources": sources, "input_manifest_sha256": input_hash,
                "requires_observed_cli_exit_zero": True, "full_program_validated": False,
                "degree_two_preparation_validated": False, "optimization_output_validated": False,
                "physical_resources_computed": False, "sampled_outcomes": False,
                "formal_interval_certificate": False, "publication_complete": False}
    raw_manifest = _json_bytes(manifest)
    if len(raw_manifest) > MANIFEST_READ_CAP or len(raw) + len(raw_manifest) > args.max_output_bytes:
        raise ValueError("Combined validation output exceeds its byte cap")
    state["output_bytes"] = len(raw); _write_exclusive(output / "validation.json", raw, guard.check)
    state["output_bytes"] += len(raw_manifest); _write_exclusive(output / "manifest.json", raw_manifest, guard.check)
    expected = {"validation.json", "manifest.json"}; seen = set()
    with os.scandir(output) as members:
        for member in members:
            guard.check()
            if member.name in seen or len(seen) >= len(expected) or member.name not in expected or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("Validation output has an unexpected member")
            seen.add(member.name)
    if seen != expected:
        raise ValueError("Validation output has missing or extra members")
    for name, recorded in {"validation.json": _pin(raw), "manifest.json": _pin(raw_manifest)}.items():
        guard.check()
        if _pin(_read_regular(output / name, args.max_output_bytes, recorded["sha256"])) != recorded:
            raise ValueError("Retained component validation changed before normal return")
    for source in sources.values():
        _source_binding(source, source["path"])
    # Reauthenticate saved source inputs; do not rerun the verifier or producer.
    if _sha(_read_regular(args.input_dir / "manifest.json", MANIFEST_READ_CAP, args.input_manifest_sha256)) != input_hash:
        raise ValueError("Input manifest changed before normal return")
    for name, recorded in source_manifest["files"].items():
        guard.check()
        if _pin(_read_regular(args.input_dir / name, INPUT_BYTE_CAP, recorded["sha256"])) != recorded:
            raise ValueError("Input member changed before normal return")
    for label, relative in (("script", GENERATOR_RELATIVE), ("module", CORE_RELATIVE)):
        _source_binding(source_manifest["sources"][label], relative)
    guard.check()
    return {"status": manifest["status"], "scope": "local_component_only", "files": 2,
            "bytes": len(raw) + len(raw_manifest), "manifest_sha256": _sha(raw_manifest),
            "budget_snapshot": snapshot, "requires_observed_cli_exit_zero": True,
            "full_program_validated": False, "retry": False, "publication_complete": False}


def main(argv=None, *, _keep_process_alarm=False):
    global _INVOCATION_SPENT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-manifest-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--module-sha256", required=True)
    parser.add_argument("--time-limit-seconds", type=float, default=20.0)
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_OUTPUT_BYTES)
    args = parser.parse_args(argv)
    state = {"budget": None, "output": None, "output_bytes": 0}
    try:
        if args.execute is not True:
            raise ValueError("Component numerical validation requires explicit --execute opt-in")
        if _INVOCATION_SPENT:
            raise ValueError("This process already consumed its validation invocation")
        _INVOCATION_SPENT = True
        args.source_sha256 = _expected_sha(args.source_sha256, "Script pin")
        args.module_sha256 = _expected_sha(args.module_sha256, "Verifier pin")
        args.input_manifest_sha256 = _expected_sha(args.input_manifest_sha256, "Input manifest pin")
        args.time_limit_seconds = _seconds(args.time_limit_seconds)
        args.max_output_bytes = _integer(args.max_output_bytes, MAX_OUTPUT_BYTES, "Output byte limit")
        with _HardDeadline(args.time_limit_seconds, _keep_process_alarm) as guard:
            result = _validate(args, guard, state)
        print(json.dumps(result, sort_keys=True, allow_nan=False)); return 0
    except (Exception, KeyboardInterrupt) as error:
        failure = {"schema": "openquantum.dqi.path_majority_failed_invocation.v1",
                   "status": "failed_local_component", "error_class": type(error).__name__, "reason": str(error)[:512],
                   "budget_snapshot": _budget_snapshot(state["budget"]), "retry": False, "complete": False,
                   "full_program_validated": False, "physical_resources_computed": False,
                   "sampled_outcomes": False, "publication_complete": False}
        output = state.get("output")
        if output is not None:
            try:
                failed_raw = _json_bytes(failure)
                if len(failed_raw) <= 16 * 1024 and state["output_bytes"] + len(failed_raw) <= args.max_output_bytes:
                    def failure_check():
                        if time.monotonic() >= state.get("deadline", 0):
                            raise TimeoutError("No wall allowance remains for failure retention")
                    state["output_bytes"] += len(failed_raw)
                    _write_exclusive(output / "failed-invocation.json", failed_raw, failure_check)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True, allow_nan=False)); return 1


if __name__ == "__main__":
    raise SystemExit(main(_keep_process_alarm=True))
