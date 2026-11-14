#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Verify saved RPE source/export evidence without importing a quantum package.

This checks bytes, a bounded primitive QASM recipe, ledgers, safe archive
membership and the retained failed/succeeded invocation distinction. It never
executes a producer, samples a phase, constructs an SDK circuit or computes a
physical estimate. An external projection-manifest pin is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile

SCHEMA = "openquantum.robust_qpe_resource_source_projection.v1"
RECORD_SCHEMA = "openquantum.robust_qpe_resource_source_execution_summary.v1"
DESCRIPTOR_SCHEMA = "openquantum.robust_qpe_resource_source_descriptor.v1"
MANIFEST = "projection-manifest.json"
DESCRIPTOR = "source-export-descriptor.json"
ARCHIVE = "source-export-download.zip"
RAW_CODE = (
    "scripts/generate_robust_qpe_resources.py", "src/openquantum/robust_qpe_resources.py",
    "src/openquantum/instrument_resources.py", "src/openquantum/hadamard_qpe.py",
    "src/openquantum/robust_qpe.py", "src/openquantum/__init__.py",
    "src/openquantum/grover.py", "requirements-lock.txt", "LICENSE",
)
EXTRA_CODE = (
    "tests/test_robust_qpe_resources.py", "tests/test_robust_qpe_resources_producer.py",
    "pyproject.toml", "scripts/verify_robust_qpe_resource_source_saved.py",
)
SOURCE_CHECK_NAMES = RAW_CODE + tuple(name for name in EXTRA_CODE if name != "pyproject.toml")
SCIENCE = ("source-macrocycle.qasm", "source-ledger.json") + tuple(f"source-fringe-{i}.qasm" for i in range(12))
EVIDENCE = ("initial-source-manifest.json", "repaired-source-manifest.json", "initial-producer.py", "producer-README.md")
FILES = ({"README.md", "LICENSE.txt", "references.bib", "source-export-record.json"}
         | {"code/" + p for p in RAW_CODE + EXTRA_CODE}
         | {"source/" + p for p in SCIENCE} | {"evidence/" + p for p in EVIDENCE})
ALL_FILES = FILES | {MANIFEST, DESCRIPTOR, ARCHIVE}
COUNTS = {"p": 3024, "cx": 2016, "h": 24, "x": 96, "sdg": 6, "reset": 108, "measure": 12}
OLD_PRODUCER_SHA = "f466be1d1ce7f022905103bb2f53dd2feb9e7cacb122835d52de4a196d1e2cfe"
NEW_PRODUCER_SHA = "5c5b34f3041a33d319232d60c5bd06067893f328a6645d1c38ed51860631e393"
TEST_HASHES = {
    "tests/test_robust_qpe_resources.py": "a77daf938d716bae17f4fe30dfc85e5b82e8f8971dcdc80ae09e2a84f3adcf77",
    "tests/test_robust_qpe_resources_producer.py": "384cd1ae4c837e893ac5d4f58558a58237008372640bf2682103309086cee1e8",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 8 * 1024 * 1024
MAX_FILES = 40


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def reject(value):
        raise ValueError("Nonfinite JSON scalar " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _name(name):
    _require(type(name) is str and name and "\\" not in name, "Invalid member name")
    p = PurePosixPath(name)
    _require(not p.is_absolute() and all(part not in {"", ".", ".."} for part in p.parts)
             and str(p) == name, "Noncanonical member path")
    return p


def _read(root, name, check, cap=MAX_BYTES):
    p = root / _name(name)
    for parent in [p, *p.parents]:
        if parent == root.parent:
            break
        _require(not parent.is_symlink(), "Member path is a symlink")
    check()
    before = p.stat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and before.st_size <= cap,
             "Member is not a bounded ordinary single-link file")
    with p.open("rb") as handle:
        raw = handle.read(cap + 1)
    after = p.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Member changed during read")
    _require(len(raw) == before.st_size <= cap, "Member byte cap changed")
    check()
    return raw


def _false(record, names):
    for name in names:
        _require(type(record.get(name)) is bool and record[name] is False, "Forbidden validation claim: " + name)


def _qasm(raw, bits):
    lines = raw.decode("ascii").splitlines()
    _require(lines[:4] == ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{bits}] c;", "qubit[9] q;"],
             "Unexpected QASM header/width")
    return [line for line in lines[4:] if line.strip()]


def _fringe(power, sine, bit):
    expected = [f"reset q[{i}];" for i in range(9)] + [f"x q[{i}];" for i in range(1, 9)] + ["h q[0];"]
    for _ in range(power):
        for wire in range(1, 9):
            expected += ["p(3*pi/80) q[0];", f"cx q[0], q[{wire}];", f"p(-3*pi/80) q[{wire}];",
                         f"cx q[0], q[{wire}];", f"p(3*pi/80) q[{wire}];"]
    if sine:
        expected += ["sdg q[0];"]
    return expected + ["h q[0];", f"c[{bit}] = measure q[0];"]


def _check_qasm_and_ledger(data):
    statements = []
    for index in range(12):
        power, sine = 1 << (index // 2), bool(index % 2)
        actual = _qasm(data[f"source/source-fringe-{index}.qasm"], 1)
        _require(actual == _fringe(power, sine, 0), "Saved fringe primitive recipe differs")
        statements += _fringe(power, sine, index)
    macro = _qasm(data["source/source-macrocycle.qasm"], 12)
    _require(macro == statements, "Saved macrocycle differs from its twelve fresh fringes")
    counts = Counter()
    for line in macro:
        if " = measure " in line:
            counts["measure"] += 1
        else:
            name = re.match(r"^([a-z]+)(?:\(|\s)", line)
            _require(name is not None, "Unknown primitive statement")
            counts[name.group(1)] += 1
    _require(dict(counts) == COUNTS and sum(counts.values()) == 5286, "Primitive source counts differ")
    ledger = _json(data["source/source-ledger.json"])
    expected = {"target_qubits": 8, "levels": 6, "shots_per_fringe": 400, "logical_qubits": 9,
                "distinct_experiment_circuits": 12, "fresh_preparations": 4800, "final_readouts": 4800,
                "controlled_U_calls": 50400, "maximum_coherent_base_calls": 32, "source_operations_per_macrocycle": 5286}
    for key, value in expected.items():
        _require(type(ledger.get(key)) is int and ledger[key] == value, "Source ledger scalar differs: " + key)
    _require(ledger.get("powers") == [1, 2, 4, 8, 16, 32]
             and all(type(value) is int for value in ledger["powers"]), "Source powers differ")
    _require(ledger.get("source_counts_per_macrocycle") == COUNTS, "Macrocycle ledger differs")
    _require(ledger.get("raw_all_trial_counts") == {k: 400*v for k, v in COUNTS.items()}, "Designed replay ledger differs")
    return dict(counts)


def _check_record(record):
    _require(record.get("schema") == RECORD_SCHEMA, "Unexpected execution-summary schema")
    _false(record, ("numerical_validation", "physical_estimation", "sampling", "phase_output", "formal_proof", "hardware", "complete_end_to_end_cost"))
    task = record.get("task", {})
    expected = {"n": 8, "levels": 6, "shots_per_fringe": 400, "macrocycle_wires": 9,
                "fringe_count": 12, "maximum_power": 32, "actual_macrocycle_primitives": 5286}
    for key, value in expected.items():
        _require(type(task.get(key)) is int and task[key] == value, "Unexpected source task: " + key)
    _require(task.get("phase_nominal") == .3 and task.get("actual_macrocycle_counts") == COUNTS,
             "Source phase/count recipe differs")
    _require(task.get("repeated_protocol_counts_status") == "design_only_not_executed_shots", "Replay confused with executed shots")
    limits = record.get("limits", {})
    for key, value in {"outer_wall_seconds": 180, "self_RSS_bytes": 268435456, "output_bytes": MAX_BYTES,
                       "output_files": 40, "capture_bytes": 262144}.items():
        _require(type(limits.get(key)) is int and limits[key] == value, "Frozen run limit differs: " + key)
    runs = record.get("runs")
    _require(type(runs) is list and len(runs) == 2, "Both original run outcomes are required")
    for index, (run, expected_id, complete, code, producer) in enumerate(zip(
            runs, ("initial", "repaired"), (False, True), (1, 0), (OLD_PRODUCER_SHA, NEW_PRODUCER_SHA), strict=True)):
        _require(run.get("id") == expected_id and type(run.get("whole_invocation_completed")) is bool
                 and run["whole_invocation_completed"] is complete, "Failed/succeeded source invocation distinction changed")
        _require(type(run.get("exit_code")) is int and run["exit_code"] == code, "Source exit code differs")
        _require(run.get("producer_sha256") == producer, "Source producer identity differs")
        _require(run.get("source_commit") == ("182fb7e33ae98c0de0a12084eff1d85efa49e06f" if index == 0 else
                                               "57d0c3c6ff732b7118ec37f6cf9b626443309ec7"), "Observed source commit differs")
        _require(type(run.get("final_self_RSS_guard_passed")) is bool
                 and run["final_self_RSS_guard_passed"] is complete, "RSS failure/success distinction differs")
        peak = run.get("recorded_peak_rss_bytes")
        _require(type(peak) is int and peak == (301023232 if index == 0 else 106037248), "Retained peak RSS differs")
        wall = run.get("outer_wall_seconds")
        _require(type(wall) in (int, float) and math.isfinite(wall) and 0 < wall < 180, "Invalid retained wall time")
        _require(run.get("raw_manifest_path") == f"evidence/{'initial' if index == 0 else 'repaired'}-source-manifest.json",
                 "Raw source manifest role differs")
    tests = record.get("test_evidence", {})
    _require(type(tests.get("collected")) is int and tests["collected"] == 79
             and type(tests.get("passed")) is int and tests["passed"] == 79, "Retained source-test count differs")
    _require(tests.get("tested_producer_sha256") == OLD_PRODUCER_SHA
             and tests.get("tested_source_commit") == "815955f2f7dfdd23623268b6784fceb949bc2e26"
             and tests.get("tests_sha256") == TEST_HASHES, "Tested source identity differs")
    _require(tests.get("not_rerun_after_buffer_repair") is True and tests.get("post_repair_static_review") is True,
             "Source-test execution versus static-review scope differs")
    return runs


def _raw_manifest(data, run, index):
    raw = _json(data[run["raw_manifest_path"]])
    _require(raw.get("mode") == "source-only" and raw.get("status") == "generation_only_no_phase_synthesis_or_physical_result",
             "Raw producer manifest is not source-only")
    _false(raw, ("compiled_fringe_check_executed", "physical_resource_estimate_computed", "sampling_performed",
                 "phase_output_computed", "complete_end_to_end_cost", "end_to_end_confidence_established",
                 "formal_proof_reproduced", "hardware_executed", "matched_improvement_computed", "retry"))
    for name in ("fixed_physical_task", "source_check", "synthesis", "synthesis_wall_seconds", "native_estimation_wall_seconds"):
        _require(raw.get(name) is None, "Raw source record contains a physical/numerical result")
    _require(raw.get("source_commit") == run.get("source_commit") and raw.get("source_dirty") is False, "Raw source ancestry differs")
    _require(raw.get("wait_derived_self_peak_rss_bytes") == run["recorded_peak_rss_bytes"], "Raw peak record differs")
    _require(raw.get("protected_generated_file_count") == 9922 and raw.get("protected_generated_unchanged") is True,
             "Protected-artifact comparison differs")
    _require(raw.get("protected_generated_digest") == "60707c03f3b71f3e4afce98887f86821a7b863731f4f65221974fad154edb9c3",
             "Protected-artifact digest differs")
    members = raw.get("artifact_sha256", {})
    _require(set(members) == set(SCIENCE) | {"code/" + p for p in RAW_CODE} | {"README.md"}, "Raw producer membership differs")
    for name, expected in members.items():
        mapped = ("source/" + name if name in SCIENCE else "evidence/producer-README.md" if name == "README.md" else
                  "evidence/initial-producer.py" if index == 0 and name == "code/scripts/generate_robust_qpe_resources.py" else name)
        _require(hashlib.sha256(data[mapped]).hexdigest() == expected, "Raw source member identity differs: " + name)


def verify_saved_source_export(bundle: Path, expected_manifest_sha256: str, source_root: Path | None = None,
                               max_seconds: int = 30):
    _require(type(expected_manifest_sha256) is str and SHA.fullmatch(expected_manifest_sha256), "An exact external manifest pin is required")
    _require(type(max_seconds) is int and 1 <= max_seconds <= 60, "Saved-verifier time must be 1..60 seconds")
    started = time.monotonic()
    def check():
        _require(time.monotonic() - started < max_seconds, "Saved source-verifier deadline exceeded")
    bundle = Path(bundle).absolute()
    _require(bundle.is_dir() and not bundle.is_symlink(), "Source packet root must be an ordinary directory")
    found = set()
    expected_dirs = {str(p) for n in ALL_FILES for p in PurePosixPath(n).parents if str(p) != "."}
    for path in bundle.rglob("*"):
        check()
        _require(not path.is_symlink(), "Source packet contains a symlink")
        name = str(path.relative_to(bundle))
        if path.is_dir():
            _require(name in expected_dirs, "Unexpected packet directory")
        else:
            _require(path.is_file(), "Nonordinary source member")
            found.add(name)
    _require(found == ALL_FILES and len(found) <= MAX_FILES, "Source packet membership is not closed")
    manifest_raw = _read(bundle, MANIFEST, check, 65536)
    _require(hashlib.sha256(manifest_raw).hexdigest() == expected_manifest_sha256, "Projection manifest pin differs")
    manifest = _json(manifest_raw)
    _require(manifest.get("schema") == SCHEMA and manifest.get("scope") == "source_export_only", "Unexpected projection schema/scope")
    _require(manifest.get("source_record") == "source-export-record.json", "Unexpected source record role")
    pins = manifest.get("files", {})
    _require(type(pins) is dict and set(pins) == FILES, "Projection file map differs")
    data = {MANIFEST: manifest_raw}
    for name, pin in pins.items():
        _require(type(pin) is dict and set(pin) == {"bytes", "sha256"} and type(pin["bytes"]) is int
                 and 0 <= pin["bytes"] <= MAX_BYTES and type(pin["sha256"]) is str and SHA.fullmatch(pin["sha256"]), "Invalid member byte/hash pin")
        raw = _read(bundle, name, check)
        _require(len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"], "Projection member bytes differ: " + name)
        data[name] = raw
    _require(sum(len(raw) for raw in data.values()) <= MAX_BYTES, "Expanded source packet exceeds byte cap")
    runs = _check_record(_json(data["source-export-record.json"]))
    qasm_counts = _check_qasm_and_ledger(data)
    for index, run in enumerate(runs):
        _raw_manifest(data, run, index)
    descriptor_raw = _read(bundle, DESCRIPTOR, check, 8192)
    descriptor = _json(descriptor_raw)
    _require(descriptor.get("schema") == DESCRIPTOR_SCHEMA and descriptor.get("scope") == "source_export_only", "Descriptor schema/scope differs")
    _false(descriptor, ("numerical_validation", "physical_estimation"))
    _require(descriptor.get("manifest") == {"path": MANIFEST, "bytes": len(manifest_raw), "sha256": expected_manifest_sha256}, "Descriptor manifest role differs")
    archive = _read(bundle, ARCHIVE, check)
    _require(descriptor.get("archive") == {"path": ARCHIVE, "bytes": len(archive), "sha256": hashlib.sha256(archive).hexdigest()}, "Archive descriptor differs")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        infos = zipped.infolist()
        _require(len(infos) == len(FILES) + 1 and len({i.filename for i in infos}) == len(infos)
                 and {i.filename for i in infos} == FILES | {MANIFEST}, "Archive membership is not closed")
        _require(sum(i.file_size for i in infos) <= MAX_BYTES, "Expanded archive exceeds byte cap")
        for info in infos:
            check(); _name(info.filename)
            _require(not info.is_dir() and not stat.S_ISLNK(info.external_attr >> 16) and not info.flag_bits & 1,
                     "Unsafe archive member")
            _require(zipped.read(info) == data[info.filename], "Archive member differs from pinned bytes")
    source_files = 0
    if source_root is not None:
        source_root = Path(source_root).absolute()
        # The original project configuration is a pinned download snapshot.
        # Aggregate example packages may use their established synthetic
        # pyproject; it is not a runtime input to this source-only generator.
        for name in SOURCE_CHECK_NAMES:
            raw = _read(source_root, name, check)
            _require(raw == data["code/" + name], "Source-root code/config differs: " + name)
            source_files += 1
    return {"status": "verified_saved_source_export_bytes_recipe_and_outcome_scope",
            "source_only_invocation_succeeded": True, "prior_source_only_invocation_succeeded": False,
            "algorithm_numerically_validated": False, "physical_resource_estimate_computed": False,
            "actual_macrocycle_operations": 5286, "actual_unitary_operations": 5166,
            "reset_operations": 108, "readout_operations": 12, "qasm_counts": qasm_counts,
            "source_files_checked": source_files, "bundle_members_checked": len(found), "zip_members_checked": len(FILES) + 1,
            "external_manifest_sha256": expected_manifest_sha256, "elapsed_seconds": time.monotonic() - started,
            "producer_or_test_invocations": 0, "quantum_package_imports": 0,
            "scope": "saved source/export only; no phase, numerical, synthesis, physical or notebook result"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--max-seconds", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(verify_saved_source_export(args.bundle, args.expected_manifest_sha256, args.source_root, args.max_seconds), sort_keys=True))


if __name__ == "__main__":
    main()
