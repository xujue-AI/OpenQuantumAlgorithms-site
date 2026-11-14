#!/usr/bin/env python3
"""Authenticate the accepted n=2/3 DQI resource evidence using saved data only.

This standard-library checker reads a fixed fourteen-file packet, verifies its
byte identities, and cross-checks recorded scalar/count contracts. It does not
import the producer, evolve a state, synthesize a gate, estimate a model, sample
an assignment, execute a notebook, or certify native replay or publication.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat


EXPECTED_MANIFEST_SHA256 = "6dc9ea6f5a44430e5dd83e8b796db27cd060e332aca85a2720a2510259be23ca"
SOURCE_COMMIT = "90b34b88b7da2f3846305b9ddd790f3845f74240"
ACCEPTED_STUDY_COMMIT = "0758b8694b213eb0ee670df053233ba9f5fe36d6"
INDEPENDENT_AUDIT_SHA256 = "d52b18719ea340982d304de97374619fea2a0b5e6946e7c49a9e8379c41712b5"
PARENT_ACCEPTANCE_SHA256 = "fffb3e0a58279b6507789650d6578d98dedf10197822b8344f1e98b7809e2c2e"
SCHEMA_PREFIX = "openquantum.dqi.structured_resource_"
STATUS = "accepted_conditional_fixed_d15_model"
SCIENTIFIC_FILES = frozenset({
    "n2-compiled.json", "n2-compiled.ndjson", "n3-compiled.json",
    "n3-compiled.ndjson", "native-words.json", "verification.json", "physical.json",
})
MEMBERS = SCIENTIFIC_FILES | {
    "results.json", "acceptance.json", "software.json", "README.md",
    "requirements-lock.txt", "LICENSE.txt",
}
ALL_FILES = MEMBERS | {"manifest.json"}
MAX_MANIFEST_BYTES = 65536
MAX_MEMBER_BYTES = 1 << 20
MAX_PACKET_BYTES = 1 << 20
SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
FRACTION_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)/[1-9][0-9]*\Z", re.ASCII)
PRIVATE_MARKERS = (
    b"/" + b"Users/", b"/private/" + b"tmp/", b"/" + b"tmp/",
    b"docs/" + b"coordination/", b"ssh" + b"://", b"PAGES_" + b"DEPLOY_KEY",
    b"BEGIN " + b"OPENSSH PRIVATE KEY", b"BEGIN " + b"RSA PRIVATE KEY",
    b"gh" + b"p_", b"github_" + b"pat_",
)
NATIVE_COUNTS = {
    "synthesis_native_calls": 8, "direct_native_trace_estimate_calls": 2,
    "factory_frontier_candidates": 0, "factory_frontier_queries": 0,
    "factory_pipeline_candidates": 0, "qdk_top_level_estimate_calls": 0,
    "query_scope": "top-level native entries only; internal native iterations not counted",
}
FALSE_FLAGS = (
    "qpu_execution", "sampled_outcomes", "physical_configuration_optimized",
    "simultaneous_case_success_claim",
)
SOURCE_FILES = frozenset({
    "scripts/generate_optimization_dqi_resources.py",
    "src/openquantum/optimization_dqi_resource_compile.py",
    "src/openquantum/optimization_dqi_resource_contract.py",
    "src/openquantum/optimization_dqi_resource_physical.py",
    "src/openquantum/optimization_dqi_resource_verify.py",
    "tests/test_optimization_dqi_resource_contract.py",
    "tests/test_optimization_dqi_resource_driver.py",
})
SOURCE_INPUTS = frozenset({
    "LICENSE", "requirements-lock.txt", "src/openquantum/instrument_resources.py",
    "src/openquantum/optimization_dqi_structured.py",
    "assets/generated/optimization/compiled-replay/physical-summary.json",
    "assets/generated/optimization/word-transport/baseline/physical-dag.json.gz",
    "assets/generated/optimization/dqi-structured/bundle.json",
    "assets/generated/optimization/dqi-structured/small/validated-small.json",
    "assets/generated/optimization/dqi-structured/small/n2/dqi.qasm",
    "assets/generated/optimization/dqi-structured/small/n2/ledger.json",
    "assets/generated/optimization/dqi-structured/small/n2/operations.ndjson",
    "assets/generated/optimization/dqi-structured/small/n3/dqi.qasm",
    "assets/generated/optimization/dqi-structured/small/n3/ledger.json",
    "assets/generated/optimization/dqi-structured/small/n3/operations.ndjson",
})
CONDITIONS = [
    "numerical error estimates bounding realization discrepancy",
    "frozen physical/controller/readiness scenario",
    "faithful assignment-bit transfer", "correct classical scoring and selection",
]
PROFILE = {
    "physical_error_rate": 0.0001, "physical_gate_time_ns": 100,
    "physical_measurement_time_ns": 500, "physical_initialization_time_ns": 100,
    "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
    "controller_failure_per_decision": 1e-15, "memory_prefactor": 0.1,
    "initialization_residual_prefactor": 0.1, "memory_threshold": 0.01,
    "factory_pipeline_failure_bound": 0.01, "factory_feedback_rounds_reserved": 3,
    "maximum_code_distance": 49,
}
CASES = {
    2: {"draws": 5, "wires": 6, "operations": 861, "t_per_draw": 186,
        "t_total": 930, "patches": 23, "data_qubits": 10327, "peak_qubits": 246950,
        "runtime_per_draw": 636118200, "runtime_total": 3180591000,
        "eta_ceiling": "1/2000", "ideal_miss": "1/1024",
        "availability": "93/100000", "pipeline_attempts": 2790},
    3: {"draws": 3, "wires": 9, "operations": 2604, "t_per_draw": 383,
        "t_total": 1149, "patches": 35, "data_qubits": 15715, "peak_qubits": 252338,
        "runtime_per_draw": 1952184000, "runtime_total": 5856552000,
        "eta_ceiling": "1/1200", "ideal_miss": "1/1000",
        "availability": "1149/1000000", "pipeline_attempts": 3447},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _sha_value(value):
    return isinstance(value, str) and SHA_PATTERN.fullmatch(value) is not None


def _exact_int(value, expected):
    return type(value) is int and value == expected


def _number(value, label):
    _require(type(value) in (int, float) and math.isfinite(value), label + " is not finite")
    return value


def _fraction(value, label):
    _require(isinstance(value, str) and len(value) <= 256
             and FRACTION_PATTERN.fullmatch(value) is not None,
             label + " is not a bounded canonical fraction")
    result = Fraction(value)
    _require(max(result.numerator.bit_length(), result.denominator.bit_length()) <= 512
             and str(result.numerator) + "/" + str(result.denominator) == value,
             label + " is not reduced or exceeds its bit bound")
    return result


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Non-finite JSON constant")


def _json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def _read(directory, name, limit):
    path = directory / name
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
             "Packet member is not a bounded ordinary file: " + name)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        _require(stat.S_ISREG(opened.st_mode)
                 and (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
                 "Packet member changed while opening: " + name)
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    _require(len(raw) == before.st_size <= limit
             and (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
             == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns),
             "Packet member changed or exceeded its read limit: " + name)
    _require(not any(marker in raw for marker in PRIVATE_MARKERS),
             "Private operational marker in public packet: " + name)
    return raw


def _inventory(directory):
    root = directory.lstat()
    _require(stat.S_ISDIR(root.st_mode), "Packet root must be an ordinary directory")
    names = set()
    with os.scandir(directory) as entries:
        for entry in entries:
            _require(entry.name in ALL_FILES and entry.name not in names
                     and entry.is_file(follow_symlinks=False) and not entry.is_symlink(),
                     "Packet contains an extra, symlink or nonregular entry")
            names.add(entry.name)
            _require(len(names) <= 14, "Packet file count exceeds fourteen")
    _require(names == ALL_FILES, "The fourteen-file saved packet is incomplete")


def _pin_map(record, names, label):
    _require(isinstance(record, dict) and set(record) == names, label + " names changed")
    for name, pin in record.items():
        _require(isinstance(pin, dict) and set(pin) == {"bytes", "sha256"}
                 and type(pin["bytes"]) is int and 0 < pin["bytes"] <= 32 * (1 << 20)
                 and _sha_value(pin["sha256"]), label + " pin is invalid: " + name)


def _native_counts(record):
    _require(isinstance(record, dict) and set(record) == set(NATIVE_COUNTS),
             "Aggregate native counter fields changed")
    for key, expected in NATIVE_COUNTS.items():
        _require((_exact_int(record[key], expected) if type(expected) is int
                  else type(record[key]) is str and record[key] == expected),
                 "Aggregate native count or scope changed: " + key)


def _cases(record, label):
    _require(isinstance(record, list) and len(record) == 2
             and all(isinstance(row, dict) for row in record), label + " must contain two cases")
    _require([row.get("n") for row in record] == [2, 3]
             and all(type(row["n"]) is int for row in record), label + " sizes/order changed")
    return {row["n"]: row for row in record}


def _check_compiled(raw, record, verification, n):
    expected = CASES[n]
    rows = [_json(line) for line in raw.splitlines()]
    _require(len(rows) == expected["operations"], "Compiled operation count changed")
    allowed = {"h", "x", "z", "s", "t", "tdg", "cx", "reset", "measure", "barrier"}
    counts = Counter()
    for row in rows:
        _require(type(row) is list and len(row) == 4 and row[0] in allowed
                 and type(row[1]) is list and type(row[2]) is list and row[3] == []
                 and all(type(q) is int and 0 <= q < expected["wires"] for q in row[1])
                 and all(type(c) is int and 0 <= c < n for c in row[2]),
                 "Opaque or malformed compiled operation")
        arity = expected["wires"] if row[0] == "barrier" else 2 if row[0] == "cx" else 1
        _require(len(row[1]) == arity and len(set(row[1])) == arity
                 and len(row[2]) == (1 if row[0] == "measure" else 0),
                 "Compiled operation arity changed")
        counts[row[0]] += 1
    mapping = [{"classical_bit": j, "wire": n + 1 + j} for j in range(n)]
    _require(rows[:expected["wires"]] == [["reset", [q], [], []] for q in range(expected["wires"])]
             and rows[-n-1:-1] == [["measure", [n+1+j], [j], []] for j in range(n)]
             and rows[-1] == ["barrier", list(range(expected["wires"])), [], []],
             "Initial reset or assignment readout chronology changed")
    _require(record["source_wires"] == expected["wires"] and record["n"] == n
             and record["compiled_counts"] == dict(counts)
             and verification["compiled_gate_counts"] == dict(counts)
             and verification["compiled_operation_count"] == len(rows)
             and record["source_readout_mapping"] == verification["readout_mapping"] == mapping
             and counts["reset"] == expected["wires"] and counts["measure"] == n
             and counts["t"] + counts["tdg"] == expected["t_per_draw"]
             and record["actual_t_tdg_per_draw"] == expected["t_per_draw"]
             and record["actual_t_tdg_whole_task"] == expected["t_total"]
             and record["fixed_complete_draws"] == expected["draws"],
             "Saved compiled count/readout summaries disagree")
    _require(record["validation_status"] == "words_checked_routing_and_compiled_state_pending"
             and verification["status"] == "passed_new_compiled_state_check"
             and verification["assignment_probability_scope"]
             == "unconditional marginal over every clause/workspace value",
             "Compiler-stage preservation or compiled acceptance status changed")
    for key in ("accepted_decoder_operator_rerun", "accepted_ideal_circuit_rerun",
                "cleanup_postselection_performed", "dense_full_circuit_operator_constructed",
                "formal_interval_certificate", "phase_normalization_performed", "sampled_outcomes"):
        _require(verification[key] is False, "Saved verification scope changed: " + key)
    _require(record["global_phase_hex"] == verification["global_phase_hex"], "Saved phase summaries disagree")
    exact_error = _fraction(record["whole_error_exact_fraction"], "Occurrence error")
    upper_error = Fraction.from_float(_number(record["whole_error"], "Occurrence float guard"))
    _require(exact_error == _fraction(verification["whole_error_sum_fraction"], "Verification error")
             and upper_error == Fraction.from_float(float.fromhex(verification["whole_error_upper_hex"]))
             and exact_error <= upper_error <= _fraction(expected["eta_ceiling"], "Eta ceiling")
             and record["realization_ceiling_per_draw"] == verification["per_draw_eta_fraction"]
             == expected["eta_ceiling"]
             and record["reference_padding_fraction"] == verification["reference_padding_fraction"]
             == "1/10000000000", "Saved realization guards disagree")
    return upper_error


def _check_physical(record, n):
    expected = CASES[n]
    scalar_fields = {
        "source_wires": expected["wires"], "compiled_operation_count": expected["operations"],
        "complete_draws": expected["draws"], "data_code_distance": 15, "factory_count": 1,
        "reserved_logical_patches": expected["patches"],
        "additional_patch_initializations_per_draw": expected["patches"]-expected["wires"],
        "assignment_readouts_per_draw": n, "source_initializations_per_draw": expected["wires"],
        "initializations_all_draws": expected["patches"]*expected["draws"],
        "data_workspace_physical_qubits": expected["data_qubits"], "factory_physical_qubits": 236623,
        "peak_physical_qubits": expected["peak_qubits"], "t_states_per_draw": expected["t_per_draw"],
        "t_states_all_draws": expected["t_total"], "runtime_per_draw_ns": expected["runtime_per_draw"],
        "worst_case_runtime_all_draws_ns": expected["runtime_total"],
    }
    for key, value in scalar_fields.items():
        _require(_exact_int(record[key], value), "Fixed physical scalar changed: " + key)
    _require(record["status"] == "computed_conditional_fixed_d15_model"
             and record["validated"] is True and record["hardware_guarantee"] is False
             and record["profile"] == PROFILE and record["native_error_composition"] == "UnionBound"
             and record["physical_failure_budget_exact"] == "1/400",
             "Fixed physical-model status/profile changed")
    supply = record["factory_supply"]
    _require(_exact_int(supply["full_pipeline_retry_cap_per_state"], 3)
             and _exact_int(supply["worst_case_pipeline_attempts"], expected["pipeline_attempts"])
             and supply["availability_failure_exact"] == expected["availability"]
             and supply["availability_budget_exact"] == "1/400"
             and supply["complete_pipeline_rejection_bound_exact"] == "1/100"
             and supply["complete_pipeline_attempts_independent_assumed"] is True
             and supply["native_per_round_requirement_is_whole_pipeline_certificate"] is False
             and _exact_int(supply["native_pipeline_time_ns"], 330000)
             and supply["native_output_error_hex"] == "0x1.88ea67f531f32p-56",
             "Complete factory availability/quality scope changed")
    count_per = record["macro_counts_per_draw"]
    count_all = record["macro_counts_all_draws"]
    _require(isinstance(count_per, dict) and set(count_per) == set(count_all)
             and all(type(v) is int and v > 0 for v in count_per.values())
             and all(_exact_int(count_all[k], expected["draws"]*v) for k, v in count_per.items())
             and count_per["initial_reset"] == expected["patches"] and count_per["measure"] == n,
             "Complete physical macro weights/readouts changed")
    ledger = record["macro_ledger_per_draw"]
    _require(isinstance(ledger, list) and len(ledger) <= 32
             and {row["operation"] for row in ledger} == set(count_per)
             and len(ledger) == len(count_per)
             and all(_exact_int(row["count"], count_per[row["operation"]])
                     and type(row["time_ns"]) is int and row["time_ns"] > 0 for row in ledger)
             and sum(row["count"]*row["time_ns"] for row in ledger) == expected["runtime_per_draw"],
             "Saved serial physical ledger/runtime disagree")
    failure = _fraction(record["native_physical_failure_all_draws_exact"], "Native physical failure")
    _require(failure == expected["draws"]*Fraction.from_float(float.fromhex(record["native_error_per_draw_hex"]))
             and failure <= Fraction(1, 400), "Native physical failure weighting changed")
    return failure


def _check_bounds(record, n, realization, physical):
    expected = CASES[n]
    components = {
        "ideal_miss_fraction": _fraction(expected["ideal_miss"], "Ideal miss"),
        "realization_failure_fraction": expected["draws"]*realization,
        "physical_failure_fraction": physical,
        "availability_failure_fraction": _fraction(expected["availability"], "Availability failure"),
    }
    _require(record["case_id"] == f"n{n}_optimum99" and record["objective_condition"] == "f(x)=n"
             and _exact_int(record["objective_target"], n)
             and record["conditional_on_faithful_bit_transfer_and_correct_scoring"] is True
             and record["formal_interval_certificate"] is False
             and record["success_target_fraction"] == "99/100", "Separate output task/conditions changed")
    for name, value in components.items():
        _require(_fraction(record[name], name) == value and 0 <= value <= Fraction(1, 400),
                 "Whole-task miss component changed: " + name)
    miss = sum(components.values(), Fraction(0))
    success = 1-miss
    _require(_fraction(record["miss_upper_bound_fraction"], "Exact miss") == miss
             and _fraction(record["success_lower_bound_fraction"], "Exact success") == success
             and success >= Fraction(99, 100), "Exact whole-task budget arithmetic changed")
    for key, exact, direction in (("miss_upper_bound", miss, "up"),
                                  ("success_lower_bound", success, "down")):
        observed = _number(record[key], key)
        value = Fraction.from_float(observed)
        _require((value >= exact if direction == "up" else value <= exact)
                 and abs(observed-float(exact)) <= 2*math.ulp(float(exact)),
                 "Recorded probability rounding is not outward: " + key)
    return {"n": n, "complete_draws": expected["draws"], "compiled_operations": expected["operations"],
            "t_states_all_draws": expected["t_total"], "peak_physical_qubits": expected["peak_qubits"],
            "modeled_runtime_all_draws_ns": expected["runtime_total"],
            "conditional_success_lower_bound_fraction": record["success_lower_bound_fraction"]}


def verify_saved_resources(directory):
    """Read/authenticate this exact saved packet; no scientific backend is called."""
    directory = Path(directory)
    _require(_sha_value(EXPECTED_MANIFEST_SHA256), "The saved gate has no reviewed manifest binding")
    _inventory(directory)
    manifest_raw = _read(directory, "manifest.json", MAX_MANIFEST_BYTES)
    _require(_sha(manifest_raw) == EXPECTED_MANIFEST_SHA256, "Public resource manifest digest changed")
    manifest = _json(manifest_raw)
    _require(manifest["schema"] == SCHEMA_PREFIX + "public_manifest.v1"
             and manifest["source_commit"] == SOURCE_COMMIT
             and manifest["accepted_study_commit"] == ACCEPTED_STUDY_COMMIT
             and manifest["independent_audit_sha256"] == INDEPENDENT_AUDIT_SHA256
             and set(manifest["scientific_members"]) == SCIENTIFIC_FILES
             and len(manifest["scientific_members"]) == 7, "Public manifest provenance/schema changed")
    _pin_map(manifest["files"], MEMBERS, "Public member")
    buffers = {}
    total = len(manifest_raw)
    for name in sorted(MEMBERS):
        raw = _read(directory, name, min(MAX_MEMBER_BYTES, manifest["files"][name]["bytes"]))
        _require(manifest["files"][name] == {"bytes": len(raw), "sha256": _sha(raw)},
                 "Public member digest changed: " + name)
        total += len(raw)
        _require(total <= MAX_PACKET_BYTES, "Saved packet exceeds its one-MiB cap")
        buffers[name] = raw
    results, acceptance, software = (_json(buffers[name]) for name in
                                     ("results.json", "acceptance.json", "software.json"))
    for record, schema in ((results, "results.v1"), (acceptance, "acceptance.v1"), (software, "software.v1")):
        _require(record["schema"] == SCHEMA_PREFIX + schema, "Public projection schema changed")
    _require(results["status"] == acceptance["status"] == STATUS, "Scoped accepted status changed")
    for record in (results, acceptance):
        for key in FALSE_FLAGS:
            _require(record[key] is False, "Public task scope changed: " + key)
        _native_counts(record["native_counts"])
        _require(record["source_commit"] == SOURCE_COMMIT
                 and record["accepted_study_commit"] == ACCEPTED_STUDY_COMMIT,
                 "Public projection source/study provenance changed")
        for key in ("formal_interval_certificate", "native_rerun_portability_certified", "publication_complete"):
            _require(record[key] is False, "Public projection overclaims: " + key)
    _require(results["accepted_study_not_reexecuted"] is True
             and results["frozen_inputs_verified"] is True and _exact_int(results["frozen_input_count"], 14)
             and results["conditional_on"] == acceptance["conditional_on"] == CONDITIONS
             and _exact_int(acceptance["frozen_input_count"], 14),
             "Saved study/input/conditional scope changed")
    _pin_map(acceptance["source_inputs"], SOURCE_INPUTS, "Recorded original input")
    _pin_map(acceptance["source_files"], SOURCE_FILES, "Recorded executed source")
    _pin_map(acceptance["scientific_files"], SCIENTIFIC_FILES, "Accepted scientific member")
    _require(acceptance["scientific_files"] == {name: manifest["files"][name] for name in SCIENTIFIC_FILES}
             and acceptance["independent_audit"]
             == {"verdict": "GO", "sha256": INDEPENDENT_AUDIT_SHA256, "bytes": 15874}
             and acceptance["parent_acceptance_sha256"] == PARENT_ACCEPTANCE_SHA256,
             "Accepted member/audit bindings changed")
    pure, native = acceptance["pure_tests"], acceptance["native_cli"]
    _require(_exact_int(pure["passed"], 83) and _exact_int(pure["exit_code"], 0)
             and _exact_int(pure["invocations"], 1) and pure["retry"] is False
             and _number(pure["pytest_reported_seconds"], "Pytest observation") == 0.15
             and 0 < _number(pure["caller_wall_seconds"], "Pure caller time") <= 10
             and _exact_int(native["exit_code"], 0) and _exact_int(native["invocations"], 1)
             and native["retry"] is False
             and 0 < _number(native["elapsed_seconds"], "Native caller time") <= 66,
             "Successful once-only caller observations changed")
    completion = acceptance["completion"]
    for key in ("capture_complete", "complete_control", "pipe_drain_complete", "retention_checked", "no_retry"):
        _require(completion[key] is True, "Saved completion is incomplete: " + key)
    _require(_exact_int(completion["worker_returncode"], 0), "Saved worker did not exit zero")
    cpu = acceptance["cpu_observation"]
    _require(_exact_int(cpu["rss_samples"], 80)
             and type(cpu["sampled_child_peak_rss_bytes"]) is int
             and 0 < cpu["sampled_child_peak_rss_bytes"] <= 1 << 30
             and 0 < _number(cpu["work_elapsed_before_retention"], "Saved work duration") <= 60
             and isinstance(cpu["rss_scope"], str) and bool(cpu["rss_scope"]),
             "CPU observation sampling/caps changed")
    _require(software["python"] == "3.12.14" and software["platform"] == "darwin"
             and software["source_commit"] == SOURCE_COMMIT
             and software["native_rerun_portability_certified"] is False,
             "Observed runtime identity changed")
    versions = {"qdk": "1.34.0", "qiskit": "2.6.2", "numpy": "2.5.4",
                "psutil": "7.2.2", "pytest": "9.2.0"}
    _require(set(software["packages"]) == set(versions)
             and all(software["packages"][key]["version"] == value for key, value in versions.items()),
             "Observed package versions changed")
    for field, name in (("physical_file", "physical.json"), ("verification_file", "verification.json")):
        _require(results[field] == {"path": name, **manifest["files"][name]}, "Public scientific reference changed")
    physical = _cases(_json(buffers["physical.json"]), "Physical report")
    verification = _cases(_json(buffers["verification.json"]), "Verification report")
    bounds = _cases(results["whole_task_output_bounds"], "Whole-task output bounds")
    words = _json(buffers["native-words.json"])
    _require(isinstance(words, list) and len(words) == 8
             and len({word["word_id"] for word in words}) == 8
             and all(word["validation"]["status"] == "passed_numerical_word_check"
                     and word["validation"]["formal_interval_certificate"] is False for word in words),
             "Saved exact-key word inventory changed")
    report_cases = []
    for n in (2, 3):
        compiled = _json(buffers[f"n{n}-compiled.json"])
        realization = _check_compiled(buffers[f"n{n}-compiled.ndjson"], compiled, verification[n], n)
        failure = _check_physical(physical[n], n)
        report_cases.append(_check_bounds(bounds[n], n, realization, failure))
    return {"schema": SCHEMA_PREFIX + "saved_gate_report.v1", "status": "passed_saved_evidence_check",
            "scope": "pinned saved bytes, schema, scalar fractions and count cross-checks only",
            "manifest_sha256": EXPECTED_MANIFEST_SHA256, "files": 14, "bytes": total,
            "source_commit": SOURCE_COMMIT, "accepted_study_commit": ACCEPTED_STUDY_COMMIT,
            "scientific_or_native_work_executed": False, "native_rerun_portability_certified": False,
            "publication_complete": False, "cases": report_cases}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Complete fourteen-file public saved-evidence directory")
    arguments = parser.parse_args(argv)
    try:
        result = verify_saved_resources(arguments.directory)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        print(json.dumps({"status": "failed_saved_evidence_check", "reason": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
