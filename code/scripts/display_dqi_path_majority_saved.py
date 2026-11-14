#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Display the parent-accepted path-majority component from saved bytes only.

This source has not been invoked as part of public projection preparation.
It loads no scientific module, reconstructs no input/reference vector,
recomputes no numerical diagnostic and executes no circuit, test or notebook.
The accepted twelve-wire local component and generated thirty-one-wire source
retain separate scopes. CLI alarms are scheduled from the guarded main call;
they are not a certificate for shell/interpreter startup or future portability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import time


BASE = "assets/generated/optimization/dqi-path-majority"
MANIFEST = BASE + "/public-manifest.json"
SELF = "scripts/display_dqi_path_majority_saved.py"
SCHEMA = "openquantum.dqi.path_majority_public_projection.v1"
STATUS = "parent_accepted_component_and_generation_only"
ASSET_NAMES = frozenset({
    "generated/component-ledger.json", "generated/component.ndjson",
    "generated/component.qasm", "generated/program-ledger.json",
    "generated/program.ndjson", "generated/program.qasm", "generated/manifest.json",
    "validation/manifest.json", "validation/validation.json", "input-k3.json",
    "component-input-recipe.json", "accepted-components.json", "results-summary.json",
    "environment.json", "README.md", "companion-spec.json", "component-circuit.svg",
    "code/src/openquantum/optimization_dqi_path_majority.py",
    "code/src/openquantum/optimization_dqi_path_majority_verify.py",
    "code/scripts/generate_optimization_dqi_path_majority.py",
    "code/scripts/validate_optimization_dqi_path_majority.py",
    "code/tests/test_optimization_dqi_path_majority.py",
    "code/tests/test_optimization_dqi_path_majority_boundaries.py",
})
ASSETS = frozenset(BASE + "/" + name for name in ASSET_NAMES)
SOURCE_FILES = frozenset({
    "src/openquantum/optimization_dqi_path_majority.py",
    "src/openquantum/optimization_dqi_path_majority_verify.py",
    "scripts/generate_optimization_dqi_path_majority.py",
    "scripts/validate_optimization_dqi_path_majority.py",
    "tests/test_optimization_dqi_path_majority.py",
    "tests/test_optimization_dqi_path_majority_boundaries.py", SELF,
    "scripts/plot_dqi_path_majority_component_svg.py",
    "tests/test_dqi_path_majority_public_projection.py",
})
COUNTERS = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
            "primitive_unitary_applications": 525, "boolean_decoder_mappings": 512,
            "small_ccx_reference_cases": 0, "native_estimator_calls": 0}
CLAIMS_FALSE = ("full_program_validated", "degree_two_preparation_validated",
    "optimization_output_validated", "rotation_synthesis_validated",
    "physical_resources_estimated", "formal_proof_reproduced", "hardware_executed",
    "sampling_performed", "dense_operator_constructed", "notebook_executed",
    "global_phase_fitted", "state_renormalized", "auxiliary_postselection")
DIAGNOSTICS = ("max_signed_complex_entry_error", "auxiliary_leakage_probability",
               "preserved_syndrome_probability_error", "final_norm2",
               "tolerance_absolute", "tolerance_relative")
PUBLIC_FALSE = frozenset({"full_program_validated", "degree_two_preparation_validated",
    "optimization_output_validated", "rotation_synthesis_validated", "physical_resources_estimated",
    "formal_proof_reproduced", "hardware_executed", "sampling_performed", "dense_operator_constructed",
    "companion_notebook_executed", "portable_scientific_replay_validated", "publication_complete",
    "saved_display_executed"})
SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MAX_FILE = 2 << 20
MAX_MANIFEST = 64 << 10
MAX_TOTAL = 8 << 20
MAX_SECONDS = 5.0
MAX_OUTPUT = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seconds(value):
    _require(type(value) in (int, float) and math.isfinite(value)
             and 0 < value <= MAX_SECONDS, "Saved display limit must be finite and at most five seconds")
    return float(value)


def _digest(value):
    _require(type(value) is str and SHA.fullmatch(value), "An external lowercase manifest SHA-256 is required")
    return value


def _relative(value):
    _require(type(value) is str and len(value) <= 4096, "A bounded relative file name is required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and all(part not in ("", ".", "..") for part in path.parts),
             "A saved file name cannot redirect")
    return value


def _ordinary(path, *, directory=False):
    _require(isinstance(path, (str, Path)) and len(str(path)) <= 4096,
             "A bounded ordinary filesystem path is required")
    path = Path(path).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts,
             "A filesystem path is too deep or redirects")
    for parent in reversed(path.parents):
        _require(stat.S_ISDIR(parent.lstat().st_mode), "A parent is not an ordinary directory")
    if directory:
        _require(stat.S_ISDIR(path.lstat().st_mode), "A root is not an ordinary directory")
    return path


def _pin(value):
    _require(type(value) is dict and set(value) == {"bytes", "sha256"}
             and type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_FILE,
             "An exact bounded file pin is required")
    _digest(value["sha256"])
    return value


def _read(path, check, cap=MAX_FILE):
    path = _ordinary(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap,
             "Only a bounded ordinary saved file can be read")
    check()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                 | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(fd)
        _require(stat.S_ISREG(opened.st_mode), "A saved file changed kind while opening")
        chunks, count = [], 0
        while True:
            check()
            block = os.read(fd, min(65536, cap + 1 - count))
            if not block:
                break
            count += len(block)
            _require(count <= cap, "A saved file exceeded its byte bound")
            chunks.append(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    now = path.lstat()
    identity = lambda item: (item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns)
    _require(identity(before) == identity(opened) == identity(after) == identity(now)
             and count == after.st_size, "A saved file changed during its read")
    check()
    return b"".join(chunks)


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, "A saved JSON object has duplicate keys")
            value[key] = item
        return value
    def invalid_constant(_):
        raise ValueError("A saved JSON value is nonfinite")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                      parse_constant=invalid_constant)


def _inventory(root, check):
    expected = ASSET_NAMES | {"public-manifest.json"}
    directories = frozenset(str(parent) for name in expected
        for parent in PurePosixPath(name).parents if str(parent) != ".")
    seen, visited = set(), 0
    pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        check()
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                visited += 1
                _require(visited <= 40, "The public family has too many entries")
                name = prefix + entry.name
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    _require(name in directories, "The public family has an unknown directory")
                    pending.append((Path(entry.path), name + "/"))
                else:
                    _require(stat.S_ISREG(mode) and name in expected and name not in seen,
                             "The public family has an unknown or redirected member")
                    seen.add(name)
    _require(seen == expected, "The public family is incomplete")


def _integer(value, expected, message):
    _require(type(value) is int and value == expected, message)


def _same(value, expected):
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(_same(value[key], item) for key, item in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(_same(a, b) for a, b in zip(value, expected))
    return value == expected


def _selection(accepted, summary, raw, files):
    _require(_same(accepted.get("accepted_scope"), {"local_decoder_component": True,
        "generated_full_program_source": True, "full_algorithm": False}), "Parent acceptance scope differs")
    component = accepted.get("component")
    _require(type(component) is dict and component.get("scratch_initially_clean") is True,
             "The accepted target action requires clean scratch")
    for key, expected in {"wires": 12, "state_amplitudes": 4096,
        "complete_statevector_evolutions": 1, "primitive_unitary_applications": 525,
        "validator_boolean_mappings": 512, "pure_boolean_mappings": 512,
        "whole_job_local_mapping_iterations": 1024, "small_ccx_reference_cases": 0,
        "native_estimator_calls": 0, "syndromes": 16, "arbitrary_clause_targets": 32}.items():
        _integer(component.get(key), expected, "An accepted component count differs")
    for key, expected in {"syndrome_wires": [0, 1, 2, 3],
        "target_wires": [4, 5, 6, 7, 8], "scratch_wires": [9, 10, 11]}.items():
        _require(_same(component.get(key), expected), "The accepted wire layout differs")
    _require(_same(summary.get("component_counts"), component), "Selected component counts differ")
    tests = accepted.get("tests")
    _require(type(tests) is dict and tests.get("public_source_contains_private_control_tests") is False,
             "The public source must exclude operational control tests")
    for key, expected in {"passed": 259, "scientific_pure_checks": 184,
        "private_control_boundary_checks": 75, "global_weight_at_most_two_error_words": 121,
        "independent_classical_assignments_scanned": 8192}.items():
        _integer(tests.get(key), expected, "An accepted test scope differs")
    _require(_same(summary.get("test_counts"), tests), "Selected test counts differ")
    vectors = accepted.get("vectors_saved")
    _require(_same(vectors, {"final_state_pairs_saved": 4096,
        "input_and_reference_defined_by_pinned_source_recipe": True,
        "input_vector_separately_saved": False, "reference_vector_separately_saved": False,
        "reconstruction_performed_during_public_projection": False}), "Saved-vector scope differs")
    _require(_same(summary.get("vectors_saved"), vectors), "Selected vector scope differs")
    unsupported = accepted.get("unsupported_claims")
    _require(type(unsupported) is dict and set(unsupported) == PUBLIC_FALSE
             and all(value is False for value in unsupported.values()), "Unsupported acceptance claims differ")
    _require(_same(summary.get("unsupported_claims"), unsupported), "Selected unsupported claims differ")
    selected = {key: raw[key] for key in DIAGNOSTICS + ("global_phase_fitted",
        "state_renormalized", "auxiliary_postselection")}
    _require(_same(accepted.get("component_diagnostics"), selected)
             and _same(summary.get("component_diagnostics"), selected), "Selected stored diagnostics differ")
    _require(_same(summary.get("component_input_recipe"), raw.get("component_input_recipe")),
             "The selected input recipe differs")
    _require(_same(summary.get("validation_source_pin"), files[BASE + "/validation/validation.json"]),
             "The selected validation identity differs")
    generated = accepted.get("generated_source_counts")
    _require(type(generated) is dict and generated.get("generation_only") is True
             and generated.get("scope") == "generation_only", "Full program source must remain generation-only")
    for key, expected in {"quantum_wires": 31, "unitary_primitives": 3114,
        "source_records_including_barriers": 3165, "barrier_records": 7,
        "operational_events_excluding_barriers": 3158, "reset_events": 31,
        "readout_events": 13, "component_primitives": 525}.items():
        _integer(generated.get(key), expected, "An accepted source-count scope differs")
    _require(_same(summary.get("generated_source_counts"), generated), "Selected generated counts differ")
    _require(accepted.get("raw_records_preserved") is True
             and accepted.get("original_c5_evidence_modified") is False
             and accepted.get("surface_code_baseline_modified") is False
             and accepted.get("public_projection_scientific_work_reexecuted") is False,
             "Projection preservation claims differ")
    return component, tests, vectors, generated


def _result(value):
    _require(type(value) is dict and value.get("schema") == "openquantum.dqi.path_majority_component_validation.v1"
             and value.get("status") == "passed_component_only_numerical_check",
             "The saved result is not the accepted component check")
    for key, expected in COUNTERS.items():
        _integer(value.get(key), expected, "A saved validator counter differs")
    _integer(value.get("state_wires"), 12, "Only the twelve-wire component is accepted")
    _integer(value.get("state_amplitudes"), 4096, "The saved component shape differs")
    for key in CLAIMS_FALSE:
        _require(value.get(key) is False, "An unsupported validation claim is present")
    counts = value.get("primitive_counts")
    _require(type(counts) is dict and set(counts) == {"h", "cx", "t", "tdg"}, "Primitive counts differ")
    for key, expected in {"h": 68, "cx": 219, "t": 136, "tdg": 102}.items():
        _integer(counts[key], expected, "A primitive count differs")
    budget = value.get("execution_budget")
    _require(type(budget) is dict and budget.get("spent") is True
             and budget.get("complete") is True and budget.get("active") is False,
             "The saved once-only budget is incomplete")
    for key, expected in COUNTERS.items():
        _integer(budget.get(key), expected, "A saved budget counter differs")
    state = value.get("final_state")
    _require(type(state) is list and len(state) == 4096, "The final saved pairs are incomplete")
    for pair in state:
        _require(type(pair) is list and len(pair) == 2 and all(
            type(item) in (int, float) and math.isfinite(item) for item in pair),
            "A saved real/imaginary pair is malformed")
    for key in DIAGNOSTICS:
        item = value.get(key)
        _require(type(item) in (int, float) and math.isfinite(item), "A saved diagnostic is malformed")
    _require(type(value["tolerance_absolute"]) is float and value["tolerance_absolute"] == 2e-10
             and type(value["tolerance_relative"]) is float and value["tolerance_relative"] == 0.0,
             "Saved tolerances differ")
    return {key: value[key] for key in DIAGNOSTICS}


def display_saved_component(artifact_root, *, source_root=None,
                            expected_manifest_sha256, time_limit_seconds=5.0):
    """Authenticate and select stored component facts; never reconstruct them."""
    expected = _digest(expected_manifest_sha256)
    seconds = _seconds(time_limit_seconds)
    stop = time.monotonic() + seconds
    def check():
        if time.monotonic() >= stop:
            raise TimeoutError("Saved display deadline elapsed")
    artifact = _ordinary(artifact_root, directory=True)
    source = _ordinary(artifact_root if source_root is None else source_root, directory=True)
    raw = _read(artifact / MANIFEST, check, MAX_MANIFEST)
    _require(hashlib.sha256(raw).hexdigest() == expected, "The external manifest digest differs")
    manifest = _json(raw)
    _require(type(manifest) is dict and set(manifest) == {"schema", "status", "files", "source_files"}
             and manifest.get("schema") == SCHEMA
             and manifest.get("status") == STATUS, "The public manifest scope differs")
    files, sources = manifest.get("files"), manifest.get("source_files")
    _require(type(files) is dict and set(files) == ASSETS
             and type(sources) is dict and set(sources) == SOURCE_FILES,
             "The public file/source allowlists differ")
    _inventory(_ordinary(artifact / BASE, directory=True), check)
    total, saved = len(raw), {}
    for entries, root in ((files, artifact), (sources, source)):
        for relative, pin in entries.items():
            _relative(relative)
            _pin(pin)
            total += pin["bytes"]
            _require(total <= MAX_TOTAL, "The saved family exceeds its total byte bound")
            data = _read(root / relative, check)
            _require(len(data) == pin["bytes"] and hashlib.sha256(data).hexdigest() == pin["sha256"],
                     "A saved file does not match its exact pin")
            if relative.startswith(BASE + "/") and relative.endswith(".json"):
                saved[relative[len(BASE) + 1:]] = _json(data)
    accepted = saved["accepted-components.json"]
    _require(type(accepted) is dict
             and accepted.get("schema") == "openquantum.dqi.path_majority_public_acceptance.v1"
             and accepted.get("status") == STATUS, "Parent component acceptance is absent")
    summary = saved["results-summary.json"]
    _require(type(summary) is dict
             and summary.get("schema") == "openquantum.dqi.path_majority_public_results.v1"
             and summary.get("status") == STATUS,
             "The selected saved summary is absent")
    raw_result = saved["validation/validation.json"]
    diagnostics = _result(raw_result)
    component, tests, vectors, generated = _selection(accepted, summary, raw_result, files)
    check()
    return {"schema": "openquantum.dqi.path_majority_saved_display.v1",
            "status": "displayed_authenticated_saved_component",
            "scope": "twelve-wire local decoder only; thirty-one-wire source generation only",
            "manifest_sha256": expected, "asset_files": 24, "source_files": 9,
            "component_diagnostics": diagnostics, "validator_counters": dict(COUNTERS),
            "component_counts": component, "test_counts": tests,
            "vectors_saved": vectors, "generated_source_counts": generated,
            "scientific_work_reexecuted": False, "notebook_executed": False,
            "full_program_validated": False, "publication_complete": False}


def _write(raw):
    _require(len(raw) <= MAX_OUTPUT, "Saved display output exceeds its bound")
    offset = 0
    while offset < len(raw):
        count = os.write(1, raw[offset:])
        _require(count > 0, "Saved display output did not advance")
        offset += count


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--artifact-root")
    parser.add_argument("--source-root")
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--time-limit-seconds", type=float, default=5.0)
    args = parser.parse_args(argv)
    if args.display is not True:
        parser.error("Explicit --display is required before saved-file access")
    seconds = _seconds(args.time_limit_seconds)
    _require(hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer"),
             "The guarded CLI requires a POSIX scheduled alarm")
    _require(signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "The CLI requires no inherited alarm")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(_number, _frame):
        os._exit(124)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        result = display_saved_component(args.artifact_root, source_root=args.source_root,
            expected_manifest_sha256=args.manifest_sha256, time_limit_seconds=seconds)
        _write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        return 0
    except BaseException as error:
        try:
            _write((json.dumps({"status": "failed_saved_display", "error": type(error).__name__},
                              sort_keys=True) + "\n").encode("ascii"))
        except BaseException:
            os._exit(1)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
