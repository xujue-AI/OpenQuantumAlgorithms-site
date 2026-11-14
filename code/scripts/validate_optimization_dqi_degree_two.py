#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the opt-in, once-only bounded C5 degree-two DQI component check.

This wrapper authenticates the generated source files before loading the
standalone verifier. Its only numerical scope is one complete 12-wire pass,
512 Boolean decoder mappings and one small 8-by-8 CCX reference. A saved
component remains provisional until the caller observes this CLI exit zero.
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
SCRIPT_RELATIVE = "scripts/validate_optimization_dqi_degree_two.py"
VERIFY_RELATIVE = "src/openquantum/optimization_dqi_degree_two_verify.py"
GENERATOR_RELATIVE = "scripts/generate_optimization_dqi_degree_two.py"
CORE_RELATIVE = "src/openquantum/optimization_dqi_degree_two.py"
SOURCE_READ_CAP = 512 * 1024
MANIFEST_READ_CAP = 64 * 1024
INPUT_BYTE_CAP = 1024 * 1024
DEFAULT_OUTPUT_BYTES = 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 30.0
C5_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
C5_SIGNS = (1, 1, 1, 1, 1)
INPUT_MEMBERS = frozenset({"operations.ndjson", "circuit.qasm", "ledger.json", "manifest.json"})
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
    name = "_openquantum_degree_two_cli_verifier"
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


def _c5_input(value):
    if type(value) is not dict or set(value) != {"vertices", "edges", "signs"}:
        raise ValueError("The generated input must retain exactly its graph fields")
    if type(value["vertices"]) is not int or value["vertices"] != 5:
        raise ValueError("This validator accepts exactly five graph vertices")
    edges, signs = value["edges"], value["signs"]
    if type(edges) is not list or len(edges) != 5 or any(type(e) is not list or len(e) != 2 or any(type(v) is not int for v in e) for e in edges):
        raise ValueError("C5 edges must be exact ordered integer endpoint pairs")
    if tuple(tuple(e) for e in edges) != C5_EDGES:
        raise ValueError("The C5 edge order differs from the frozen input")
    if type(signs) is not list or len(signs) != 5 or any(type(v) is not int for v in signs) or tuple(signs) != C5_SIGNS:
        raise ValueError("The frozen C5 signs are all one")


def _source_binding(value, relative: str):
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"} or value["path"] != relative:
        raise ValueError("Generated source identity names an unexpected path")
    pin = {"bytes": value["bytes"], "sha256": value["sha256"]}
    _pin_shape(pin, SOURCE_READ_CAP)
    raw = _read_regular(ROOT / relative, SOURCE_READ_CAP, pin["sha256"])
    if _pin(raw) != pin:
        raise ValueError("Generated source size differs from its recorded snapshot")


def _qasm_record(record: dict) -> bytes:
    gate, wires = record.get("gate"), record.get("qubits")
    if type(gate) is not str or gate not in PRIMITIVES or type(wires) is not list or any(type(w) is not int or not 0 <= w < 12 for w in wires):
        raise ValueError("Malformed C5 primitive record")
    if gate == "barrier":
        if set(record) != {"gate", "qubits"} or wires != list(range(12)):
            raise ValueError("Each C5 stage barrier must name all twelve wires")
        return b"barrier q;\n"
    if len(wires) != (2 if gate == "cx" else 1) or len(set(wires)) != len(wires):
        raise ValueError("Primitive arity or distinct-wire contract failed")
    args = ",".join(f"q[{w}]" for w in wires)
    if gate == "measure":
        bits = record.get("classical")
        if set(record) != {"gate", "qubits", "classical"} or type(bits) is not list or len(bits) != 1 or type(bits[0]) is not int or not 0 <= bits[0] < 4:
            raise ValueError("Readout must name one of four gauge-variable bits")
        return f"measure {args} -> c[{bits[0]}];\n".encode("ascii")
    if gate == "ry":
        values = record.get("parameters_hex")
        if set(record) != {"gate", "qubits", "parameters_hex"} or type(values) is not list or len(values) != 1 or type(values[0]) is not str or len(values[0]) > 80:
            raise ValueError("RY requires one canonical binary source angle")
        angle = float.fromhex(values[0])
        if not math.isfinite(angle) or angle.hex() != values[0]:
            raise ValueError("RY angle is nonfinite or noncanonical")
        text = format(angle, ".17g")
        if float(text).hex() != angle.hex():
            raise ValueError("QASM angle does not retain the binary source value")
        return f"ry({text}) {args};\n".encode("ascii")
    if set(record) != {"gate", "qubits"}:
        raise ValueError("Unexpected C5 primitive fields")
    return f"{gate} {args};\n".encode("ascii")


def _input_files(directory: Path, expected_manifest: str, check):
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError("Input must be an ordinary source-export directory")
    seen = set()
    with os.scandir(directory) as members:
        for member in members:
            check()
            if member.name not in INPUT_MEMBERS or len(seen) >= len(INPUT_MEMBERS) or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("The C5 source export has an unexpected member")
            seen.add(member.name)
    if seen != INPUT_MEMBERS:
        raise ValueError("The C5 source export has missing or extra members")
    raw_manifest = _read_regular(directory / "manifest.json", MANIFEST_READ_CAP, expected_manifest)
    manifest = _json(raw_manifest)
    if type(manifest) is not dict or manifest.get("schema") != "openquantum.dqi.degree_two_source_manifest.v1" or manifest.get("status") != "complete_generation_component_pending_invocation_acceptance" or manifest.get("scope") != "generation_only":
        raise ValueError("Unexpected generated source manifest schema/status")
    if manifest.get("requires_observed_cli_exit_zero") is not True:
        raise ValueError("Source component must remain invocation-provisional")
    for flag in ("simulation_performed", "sampled_outcomes", "physical_resources_computed", "formal_interval_certificate", "optimization_advantage_claim", "publication_complete"):
        if manifest.get(flag) is not False:
            raise ValueError("Generated source has an unsupported execution claim")
    _c5_input(manifest.get("input"))
    files = manifest.get("files")
    if type(files) is not dict or set(files) != INPUT_MEMBERS - {"manifest.json"}:
        raise ValueError("Generated source manifest file set changed")
    payloads, total = {}, len(raw_manifest)
    for name in sorted(files):
        check()
        _pin_shape(files[name], INPUT_BYTE_CAP)
        raw = _read_regular(directory / name, INPUT_BYTE_CAP, files[name]["sha256"])
        if _pin(raw) != files[name]:
            raise ValueError("Generated source member size changed")
        total += len(raw)
        if total > INPUT_BYTE_CAP:
            raise ValueError("Combined C5 source input exceeds one MiB")
        payloads[name] = raw
    sources = manifest.get("sources")
    if type(sources) is not dict or set(sources) != {"script", "module"}:
        raise ValueError("Generated source must retain two source snapshots")
    # Byte checks only: neither generator nor core is imported here.
    _source_binding(sources["script"], GENERATOR_RELATIVE)
    _source_binding(sources["module"], CORE_RELATIVE)
    ledger = _json(payloads["ledger.json"])
    if type(ledger) is not dict or ledger.get("schema") != "openquantum.dqi.degree_two_generation_ledger.v1" or ledger.get("status") != "generation_only_pending_invocation_acceptance" or ledger.get("input") != manifest["input"] or ledger.get("sources") != sources:
        raise ValueError("Source ledger is not bound to its generation manifest")
    lines = payloads["operations.ndjson"].splitlines()
    if len(lines) != 1796:
        raise ValueError("The frozen C5 stream contains exactly 1796 source operations")
    records = []
    qasm = [b'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[12];\ncreg c[4];\n']
    for line in lines:
        check()
        if len(line) > 8192:
            raise ValueError("One primitive record exceeds its bounded read")
        record = _json(line)
        if type(record) is not dict:
            raise ValueError("Each primitive must be a serialized object")
        qasm.append(_qasm_record(record))
        records.append(record)
    if b"".join(qasm) != payloads["circuit.qasm"]:
        raise ValueError("QASM and NDJSON do not describe the same source sequence")
    check()
    return records, ledger, manifest, _sha(raw_manifest)


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


def _validate(args, guard, state):
    state["deadline"] = guard.deadline
    own = _read_regular(Path(__file__), SOURCE_READ_CAP, args.source_sha256)
    # Input byte/graph/QASM checks finish before entering a numerical role.
    records, ledger, source_manifest, input_hash = _input_files(args.input_dir, args.input_manifest_sha256, guard.check)
    verifier, verifier_raw = _load_verifier(args.module_sha256)
    sources = {"script": {"path": SCRIPT_RELATIVE, **_pin(own)},
               "module": {"path": VERIFY_RELATIVE, **_pin(verifier_raw)}}
    state.update({"sources": sources, "input_manifest_sha256": input_hash})
    guard.check()
    output = _fresh_directory(args.output_dir)
    state["output"] = output
    budget = verifier.ExecutionBudget(execution_enabled=True)
    state["budget"] = budget
    # The verifier manages begin/reserve/finish/abort itself; no pre-begin here.
    result = verifier.verify_c5_records(records, ledger, vertices=5, edges=C5_EDGES,
                                       signs=C5_SIGNS, allow_execution=True,
                                       budget=budget, check=guard.check)
    if type(result) is not dict or result.get("schema") != "openquantum.dqi.degree_two_validation.v1" or result.get("status") != "passed_c5_degree_two_component_pending_invocation_acceptance":
        raise ValueError("Verifier returned an unexpected component result")
    snapshot = _budget_snapshot(budget)
    expected_counts = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
                       "boolean_decoder_mappings": 512, "small_ccx_reference_cases": 1,
                       "primitive_unitary_applications": 1773}
    if result.get("counters") != snapshot or any(type(snapshot.get(key)) is not int or snapshot[key] != count for key, count in expected_counts.items()) or snapshot.get("spent") is not True or snapshot.get("active") is not False or snapshot.get("complete") is not True:
        raise ValueError("Successful verification requires complete exact role counts")
    for flag in ("auxiliary_postselection_performed", "global_phase_fitting_performed", "phase_normalization_performed",
                 "sampled_outcomes", "native_synthesis_executed", "physical_resources_computed", "qpu_execution",
                 "formal_interval_certificate", "formal_theorem_reproduced", "optimization_advantage_claim",
                 "figure_eight_evolution_performed", "figure_eight_state_evolved", "accepted_degree_one_producer_rerun",
                 "accepted_degree_one_operator_rerun", "old_degree_one_evidence_reexecuted", "publication_complete"):
        if result.get(flag) is not False:
            raise ValueError("Verifier returned an unsupported execution claim")
    for field in ("native_synthesis_calls", "direct_native_estimator_calls"):
        if type(result.get(field)) is not int or result[field] != 0:
            raise ValueError("No native synthesis or estimator entry is permitted")
    guard.check()
    result["cli_component"] = {"sources": sources, "source_input_manifest_sha256": input_hash,
                               "source_input_files": source_manifest["files"],
                               "requires_observed_cli_exit_zero": True,
                               "elapsed_seconds_before_export": time.monotonic() - guard.started,
                               "time_limit_seconds": args.time_limit_seconds,
                               "max_output_bytes": args.max_output_bytes,
                               "budget_snapshot": snapshot, "retry": False,
                               "publication_complete": False}
    raw = _json_bytes(result)
    manifest = {"schema": "openquantum.dqi.degree_two_validation_manifest.v1",
                "status": "complete_validation_component_pending_invocation_acceptance",
                "files": {"validation.json": _pin(raw)}, "sources": sources,
                "input_manifest_sha256": input_hash, "requires_observed_cli_exit_zero": True,
                "sampled_outcomes": False, "physical_resources_computed": False,
                "formal_interval_certificate": False, "publication_complete": False}
    raw_manifest = _json_bytes(manifest)
    if len(raw_manifest) > MANIFEST_READ_CAP or len(raw) + len(raw_manifest) > args.max_output_bytes:
        raise ValueError("Combined validation output exceeds its byte cap")
    state["output_bytes"] = len(raw)
    _write_exclusive(output / "validation.json", raw, guard.check)
    state["output_bytes"] += len(raw_manifest)
    _write_exclusive(output / "manifest.json", raw_manifest, guard.check)
    expected = {"validation.json", "manifest.json"}
    seen = set()
    with os.scandir(output) as members:
        for member in members:
            guard.check()
            if member.name not in expected or len(seen) >= len(expected) or not stat.S_ISREG(member.stat(follow_symlinks=False).st_mode):
                raise ValueError("Unexpected validation output member")
            seen.add(member.name)
    if seen != expected:
        raise ValueError("Validation output has missing members")
    for name, recorded_pin in {"validation.json": _pin(raw), "manifest.json": _pin(raw_manifest)}.items():
        guard.check()
        retained = _read_regular(output / name, args.max_output_bytes, recorded_pin["sha256"])
        if _pin(retained) != recorded_pin:
            raise ValueError("Retained validation output changed before normal return")
    guard.check()
    return {"status": manifest["status"], "files": 2, "bytes": len(raw) + len(raw_manifest),
            "manifest_sha256": _sha(raw_manifest), "budget_snapshot": _budget_snapshot(budget),
            "requires_observed_cli_exit_zero": True, "retry": False, "publication_complete": False}


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
            raise ValueError("Numerical validation requires explicit --execute opt-in")
        if _INVOCATION_SPENT:
            raise ValueError("This CLI process already consumed its validation invocation")
        _INVOCATION_SPENT = True
        args.source_sha256 = _expected_sha(args.source_sha256, "Script pin")
        args.module_sha256 = _expected_sha(args.module_sha256, "Verifier pin")
        args.input_manifest_sha256 = _expected_sha(args.input_manifest_sha256, "Input manifest pin")
        args.time_limit_seconds = _seconds(args.time_limit_seconds)
        args.max_output_bytes = _integer(args.max_output_bytes, MAX_OUTPUT_BYTES, "Output byte limit")
        with _HardDeadline(args.time_limit_seconds, _keep_process_alarm) as guard:
            result = _validate(args, guard, state)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        failure = {"schema": "openquantum.dqi.degree_two_failed_invocation.v1",
                   "status": "failed_c5_degree_two_component", "error_class": type(error).__name__,
                   "reason": str(error)[:512], "budget_snapshot": _budget_snapshot(state["budget"]),
                   "retry": False, "complete": False, "physical_resources_computed": False,
                   "sampled_outcomes": False, "publication_complete": False}
        output = state.get("output")
        if output is not None:
            try:
                raw = _json_bytes(failure)
                if len(raw) <= 16 * 1024 and state["output_bytes"] + len(raw) <= args.max_output_bytes:
                    # Failed operational evidence is retained; it is never a passed science file.
                    def failure_check():
                        if time.monotonic() >= state.get("deadline", 0):
                            raise TimeoutError("No validation wall allowance remains for failure retention")
                    state["output_bytes"] += len(raw)
                    _write_exclusive(output / "failed-invocation.json", raw, failure_check)
            except Exception:
                pass
        print(json.dumps(failure, sort_keys=True, allow_nan=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main(_keep_process_alarm=True))
