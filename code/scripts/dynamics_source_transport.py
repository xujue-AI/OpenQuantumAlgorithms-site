#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stdlib-only, static disclosure of the reviewed Linux source differences.

The public entry points describe one preserved source run. They authenticate
the canonical inputs supplied by the caller and reconstruct the declared
source outputs, but perform no circuit execution or resource estimation.
Canonical physical contracts remain authoritative throughout.
"""
from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
import re


REVIEWED_REPLAY_COMMIT = "497c5eeeb2af11a18cf3a59c5760a4dc03bc1fec"
NATIVE_PRODUCER_COMMIT = "f842df78eb01cedd1fb438d5fd45c16fe48c435d"
ORIGINAL_STUDY_SHA256 = "35078f9e54e137c2718791741041c2d7b1f850409d54e9f93ecf32d0c76f82fb"
ORIGINAL_CORE_INVENTORY_SHA256 = "477563fe99320a6d2711f2cde1b8d883bf29d900a6621469835e0f984f3dd6f0"
REPLAY_SOURCE_FILES = (
    "src/openquantum/dynamics_resources.py", "scripts/generate_dynamics_resources.py",
    "src/openquantum/hamiltonian_simulation.py", "src/openquantum/resource_estimation.py",
    "src/openquantum/simulation/qsp.py", "src/openquantum/simulation/qsp_evolution.py",
    "src/openquantum/simulation/lcu.py", "src/openquantum/simulation/amplification.py",
    "requirements-lock.txt", "src/openquantum/__init__.py", "src/openquantum/grover.py",
    "scripts/source_provenance.py", "LICENSE",
)
LOCKED_VERSIONS = {
    "python": "3.12.14", "qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4",
    "scipy": "1.18.1", "matplotlib": "3.11.2", "psutil": "7.2.2",
}
PROVENANCE_SCOPE = "Exact checkout root when Git is available; null for standalone packages. Content hashes identify actual source."
_product_parameters = {(n, 1., order, .05) for n in range(2, 7) for order in (1, 2)}
_product_parameters.update((3, time, order, .05) for time in (.5, 2.) for order in (1, 2))
_product_parameters.update((3, 1., order, error) for error in (.025, .1) for order in (1, 2))
CASE_IDS = tuple(f"pf{order}-n{n}-t{time:g}-eps{error:g}"
                 for n, time, order, error in sorted(_product_parameters)) + tuple(
    f"qsp-oaa{rounds}-n3-t1-eps0.05" for rounds in range(4)
) + ("qsp-retargeted-n2-t1-eps0.05", "qsp-retargeted-n4-t1-eps0.05")
EXPECTED_ROTATION_CHANGES = {**{f"qsp-oaa{r}-n3-t1-eps0.05": 3 + 6*r for r in range(4)},
                             "qsp-retargeted-n2-t1-eps0.05": 28,
                             "qsp-retargeted-n4-t1-eps0.05": 61}
EXPECTED_FLOAT_COUNTS = {"numerical_checks": 221, "phase_evidence": 87, "postselection_contract": 31}
REPORT_SCOPE = (
    "Static disclosure of the preserved Linux source run: fresh source was not byte reproduced. "
    "Canonical stored circuits and complete physical contracts remain separately authoritative."
)
ROTATION_SCOPE = (
    "For identical unsynthesized rotation streams, operator norm is bounded by half the sum of angle changes. "
    "Decimal-literal and binary64 interpretations are reported separately; this is no circuit, compiler, synthesis or hardware certificate."
)
CLAIMS = {
    "canonical_costs_authoritative": True,
    "canonical_circuit_validation_performed": False,
    "source_circuit_equivalence_claimed": False,
    "formal_certificate": False,
    "hardware_or_synthesis_certificate": False,
    "Native_estimates_run": False,
    "phase_fitter_run": False,
    "costs_repriced": False,
}
_REPORT_KEYS = {"schema_version", "status", "scope", "provenance", "canonical_cases_sha256",
                "reconstructed_source_cases_sha256", "summary", "cases", "claims"}
_CASE_KEYS = {"case_id", "canonical_case_sha256", "reconstructed_source_case_sha256",
              "frozen_postselection_contract_sha256", "float_changes", "qasm"}
_QASM_KEYS = {"canonical_sha256", "source_sha256", "byte_reproduced", "line_count", "rotation_changes",
              "decimal_angle_l1_exact", "decimal_rotation_product_bound_exact", "binary64_angle_l1_exact",
              "binary64_rotation_product_bound_exact", "scope"}
_ROTATION_KEYS = {"line", "opcode", "wire", "recorded_literal", "replayed_literal"}
_FLOAT_KEYS = {"path", "recorded", "replayed", "delta", "absolute_delta"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_DECIMAL = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_DECIMAL_RE = re.compile(_DECIMAL + r"\Z")
_PI_RE = re.compile(r"[+-]?(?:\d+\*)?pi(?:/[1-9]\d*)?\Z")
_ROTATION_RE = re.compile(r"(rx|ry|rz)\(([^()]+)\) q\[(\d+)\];\n\Z")
_SIMPLE_RE = re.compile(r"(x|h|t|tdg|sdg) q\[(\d+)\];\n\Z")
_CX_RE = re.compile(r"cx q\[(\d+)\], q\[(\d+)\];\n\Z")
_MEASURE_RE = re.compile(r"meas\[(\d+)\] = measure q\[(\d+)\];\n\Z")
_BIT_RE = re.compile(r"bit\[(\d+)\] meas;\n\Z")
_QUBIT_RE = re.compile(r"qubit\[(\d+)\] q;\n\Z")
MAX_JSON_NODES = 200000
MAX_JSON_DEPTH = 64
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_QASM_BYTES = 1024 * 1024
MAX_QASM_AGGREGATE_BYTES = 16 * 1024 * 1024
MAX_QASM_LINES = 100000
MAX_CHANGED_FLOATS = 512
MAX_CHANGED_ROTATIONS = 256

_PF_NUMERICAL = {
    "circuit_observable", "exact_observable", "observable_error", "operator_error",
    "source_to_lowered_operator_error", "state_infidelity", "state_vector_error",
}
_QSP_NUMERICAL = {
    "accepted_block_polynomial_error", "actual_input_acceptance_probability",
    "actual_minus_ideal_success_probability", "all_input_conditional_Pauli_bias_projection",
    "block_error_vs_ideal_amplitude_times_exact_U", "conditional_state_vector_error",
    "decoded_linear_map_operator_error", "finite_H_numerical_acceptance_input_dependence_span",
    "finite_H_numerical_maximum_acceptance_probability", "finite_H_numerical_minimum_acceptance_probability",
    "observable_conditional", "observable_error", "observable_exact", "seeded_full_state_lowering_error",
}
_RETARGETED_NUMERICAL = {
    "actual_input_acceptance_probability", "all_input_conditional_Pauli_bias_projection",
    "compiled_block_error_vs_exact_U_over4", "conditional_state_vector_error",
    "finite_H_numerical_maximum_acceptance_probability", "finite_H_numerical_minimum_acceptance_probability",
    "observable_conditional", "observable_error", "observable_exact", "seeded_full_state_lowering_error",
    "source_to_compiled_accepted_block_operator_error",
}
_POSTSELECTION_FLOATS = {
    "reference_acceptance_lower", "synthesis_model_error_per_attempt",
    "conditional_acceptance_lower_after_synthesis_model", "all_slots_no_acceptance_union_bound",
    "conditional_expected_stopped_attempts_all_slots",
}
_PHASE_SCALARS = {
    "numerically_reconstructed_coefficient_l1_discrepancy", "phase_fit_max_residual_10001_point_grid",
    "target_max_residual_10001_point_grid",
}
_COMMON_CASE_KEYS = {
    "J", "accepted_slots", "attempts_per_accepted_slot", "case_id", "compiled_depth", "compiled_gate_counts",
    "compiled_global_phase_radians", "complete_attempts", "exclusions", "h", "hbar", "initial_state",
    "input_preparation", "logical_qubits", "measurement", "method", "numerical_checks", "observable",
    "provenance", "qasm_global_phase_scope", "qasm_sha256", "source_depth", "source_gate_counts",
    "system_qubits", "task", "time",
}
_PF_CASE_KEYS = _COMMON_CASE_KEYS | {
    "acceptance_probability", "analytic_error", "boundary", "no_acceptance_failure_actual", "order", "steps", "terms",
}
_QSP_CASE_KEYS = _COMMON_CASE_KEYS | {"access", "phase_tau", "postselection_contract", "rounds", "size_scaling_scope"}
_RETARGETED_CASE_KEYS = _QSP_CASE_KEYS | {"phase_evidence"}
_TASK_KEYS = {
    "absolute_error", "algorithm_bias", "synthesis_bias", "sampling_error", "sampling_failure", "acceptance_failure",
    "physical_failure", "accepted_samples", "whole_task_failure_allowance", "certified_confidence_lower_bound",
}
_ACCESS_KEYS = {
    "forward_A_calls", "inverse_A_calls", "total_A_or_inverse_calls", "zero_ancilla_reflections",
    "uncontrolled_block_encoding_calls", "controlled_block_encoding_calls", "branch_conditioned_projector_phase_calls",
}
_POSTSELECTION_KEYS = _POSTSELECTION_FLOATS | {
    "conditional_synthesis_Pauli_bias_allowance", "attempts_per_accepted_slot", "accepted_slots", "maximum_complete_attempts",
    "physical_failure_allowance_per_complete_attempt", "certified_confidence_lower_bound", "conditional_confidence_lower_bound",
    "assumptions", "proof", "does_not_certify",
}
_PHASE_EVIDENCE_KEYS = {
    "cosine_phases", "sine_phases", "cosine", "sine", "constructor_tolerance", "phase_fit_method", "certified",
}
CANONICAL_ENVIRONMENT = {
    "python": LOCKED_VERSIONS["python"], "device": "CPU", "seed": 20261112,
    "packages": {key: LOCKED_VERSIONS[key] for key in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")},
}


def _require(condition, message):
    if not condition:
        raise ValueError("Dynamics source transport: " + message)


def _checked_json(value):
    # Count a conservative ensure_ascii serialization bound before any copy or
    # canonicalization. Every Unicode/control character needs at most twelve
    # bytes, and finite binary64/64-bit scalar representations fit32 bytes.
    stack, nodes, byte_bound = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, "JSON traversal limit exceeded")
        kind = type(item)
        if kind is dict:
            _require(len(item) <= MAX_JSON_NODES, "JSON dictionary limit exceeded")
            _require(all(type(k) is str and len(k) <= 512 for k in item), "bounded string keys required")
            byte_bound += 2 + len(item) + sum(3 + 12 * len(k) for k in item)
            _require(nodes + len(stack) + len(item) <= MAX_JSON_NODES, "JSON node limit exceeded")
            _require(byte_bound <= MAX_JSON_BYTES, "JSON byte budget exceeded")
            stack.extend((child, depth + 1) for child in item.values())
        elif kind is list:
            byte_bound += 2 + len(item)
            _require(nodes + len(stack) + len(item) <= MAX_JSON_NODES, "JSON node limit exceeded")
            _require(byte_bound <= MAX_JSON_BYTES, "JSON byte budget exceeded")
            stack.extend((child, depth + 1) for child in item)
        elif kind is float:
            _require(math.isfinite(item), "finite JSON numbers required")
            byte_bound += 32
        elif kind is int:
            _require(item.bit_length() <= 64, "bounded JSON integers required")
            byte_bound += 32
        elif kind is str:
            _require(len(item) <= 8192, "JSON string limit exceeded")
            byte_bound += 2 + 12 * len(item)
        else:
            _require(kind in (bool, type(None)), "JSON values only")
            byte_bound += 5
        _require(byte_bound <= MAX_JSON_BYTES, "JSON byte budget exceeded")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _same(first, second):
    return _canonical(first) == _canonical(second)


def _checked_sources(sources):
    _require(type(sources) is dict and set(sources) == set(REPLAY_SOURCE_FILES), "the thirteen replay sources are required")
    _require(all(type(v) is str and _SHA256.fullmatch(v) for v in sources.values()), "invalid replay source digest")
    return copy.deepcopy(sources)


def _checked_provenance(provenance, expected_source_hashes):
    _checked_json(provenance)
    keys = {"replay_source_commit", "replay_source_sha256", "recorded_producer_commit", "original_study_sha256",
            "original_core_inventory_sha256", "locked_versions", "architecture"}
    _require(type(provenance) is dict and set(provenance) == keys, "public provenance allowlist differs")
    expected = {
        "replay_source_commit": REVIEWED_REPLAY_COMMIT,
        "replay_source_sha256": _checked_sources(expected_source_hashes),
        "recorded_producer_commit": NATIVE_PRODUCER_COMMIT,
        "original_study_sha256": ORIGINAL_STUDY_SHA256,
        "original_core_inventory_sha256": ORIGINAL_CORE_INVENTORY_SHA256,
        "locked_versions": LOCKED_VERSIONS,
        "architecture": "x86_64",
    }
    _require(_same(provenance, expected), "reviewed public provenance differs")
    return copy.deepcopy(expected)


def normalize_canonical_case(case, expected_native_sources, expected_environment):
    """Pure snapshot normalization matching the reviewed canonical normalizer."""
    _checked_json(case)
    _checked_json(expected_native_sources)
    _checked_json(expected_environment)
    _require(type(case) is dict and case.get("case_id") in CASE_IDS, "unexpected canonical case")
    value = copy.deepcopy(case)
    provenance = value.get("provenance")
    _require(type(provenance) is dict and set(provenance) == {
        "source_commit", "source_revision_scope", "source_sha256", "environment"}, "invalid logical provenance")
    commit = provenance["source_commit"]
    _require(commit is None or type(commit) is str and _COMMIT.fullmatch(commit), "invalid canonical Git commit")
    _require(provenance["source_revision_scope"] == PROVENANCE_SCOPE, "canonical provenance scope differs")
    _require(_same(provenance["source_sha256"], expected_native_sources), "canonical source bindings differ")
    _require(_same(provenance["environment"], expected_environment), "canonical interpreter/package/device/seed differs")
    provenance.pop("source_commit")
    provenance.pop("source_sha256")
    _require(value.pop("physical_reports", None) == ["frozen-p-1e-3.json", "frozen-p-1e-4.json"],
             "canonical physical report roster differs")
    timings = value.pop("cpu_timings_seconds", None)
    keys = {"construction_and_lowering", "independent_dense_validation"} if value.get("method") == "product_formula" else {
        "construction_validation_lowering"}
    _require(type(timings) is dict and set(timings) == keys
             and all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in timings.values()),
             "invalid bounded canonical CPU timing metadata")
    return value


def _checked_cases(cases, *, require_full_study):
    _checked_json(cases)
    _require(type(cases) is list and 0 < len(cases) <= len(CASE_IDS), "bounded case list required")
    ids = []
    for case in cases:
        _require(type(case) is dict and case.get("case_id") in CASE_IDS, "unknown case identity")
        ids.append(case["case_id"])
        family = ("product_formula" if case["case_id"].startswith("pf") else
                  "QSP_retargeted" if "retargeted" in case["case_id"] else "QSP_oblivious_amplification")
        _require(case.get("method") == family, "case method differs from its identity")
        _require(type(case.get("qasm_sha256")) is str and _SHA256.fullmatch(case["qasm_sha256"]), "case QASM digest required")
        if require_full_study:
            _full_case_schema(case)
    _require(len(set(ids)) == len(ids), "duplicate case identity")
    if require_full_study:
        _require(tuple(ids) == CASE_IDS, "the reviewed twenty-four case roster/order is required")
    return copy.deepcopy(cases)


def _full_case_schema(case):
    """Require the complete controlled fields of the one recorded study."""
    family = case["method"]
    keys = _PF_CASE_KEYS if family == "product_formula" else _RETARGETED_CASE_KEYS if family == "QSP_retargeted" else _QSP_CASE_KEYS
    _require(set(case) == keys, "required complete canonical/source case fields differ")
    _require(_same(case["provenance"], {"source_revision_scope": PROVENANCE_SCOPE, "environment": CANONICAL_ENVIRONMENT}),
             "normalized controlled case provenance differs")
    for name in ("system_qubits", "logical_qubits", "accepted_slots", "attempts_per_accepted_slot", "complete_attempts"):
        _require(type(case[name]) is int and case[name] > 0, "positive controlled integer required: " + name)
    for name in ("source_depth", "compiled_depth"):
        _require(type(case[name]) is int and case[name] >= 0, "controlled integer depth required")
    _require(case["system_qubits"] <= case["logical_qubits"] <= 9, "bounded controlled qubit widths required")
    _require(case["complete_attempts"] == case["accepted_slots"] * case["attempts_per_accepted_slot"], "complete retry accounting differs")
    for name in ("source_gate_counts", "compiled_gate_counts"):
        counts = case[name]
        _require(type(counts) is dict and counts and all(type(v) is int and v >= 0 for v in counts.values()), "controlled gate counts required")
    task = case["task"]
    _require(type(task) is dict and set(task) == _TASK_KEYS, "complete fixed task fields required")
    _require(type(task["accepted_samples"]) is int and task["accepted_samples"] == case["accepted_slots"]
             and task["certified_confidence_lower_bound"] is None, "controlled task shots/certificate differ")
    _require(all(type(task[k]) in (int, float) and math.isfinite(task[k]) and task[k] > 0
                 for k in _TASK_KEYS - {"accepted_samples", "certified_confidence_lower_bound"}), "finite positive task allocations required")
    _require(type(case["numerical_checks"]) is dict, "numerical evidence dictionary required")
    if family == "product_formula":
        _require(type(case["order"]) is int and case["order"] in (1, 2) and type(case["steps"]) is int and case["steps"] > 0,
                 "controlled product-formula order/steps required")
        _require(type(case["analytic_error"]) is dict and type(case["terms"]) is list, "controlled product-formula evidence required")
        return
    _require(type(case["rounds"]) is int and 0 <= case["rounds"] <= 3, "controlled amplification rounds required")
    _require(type(case["phase_tau"]) in (int, float) and math.isfinite(case["phase_tau"]) and case["phase_tau"] > 0,
             "controlled positive phase tau required")
    access = case["access"]
    _require(type(access) is dict and set(access) == _ACCESS_KEYS and all(type(v) is int and v >= 0 for v in access.values()),
             "complete controlled access counts required")
    contract = case["postselection_contract"]
    _require(type(contract) is dict and set(contract) == _POSTSELECTION_KEYS, "complete frozen postselection contract required")
    for name, controlled in (("attempts_per_accepted_slot", "attempts_per_accepted_slot"), ("accepted_slots", "accepted_slots"),
                             ("maximum_complete_attempts", "complete_attempts")):
        _require(type(contract[name]) is int and contract[name] == case[controlled], "frozen contract retry accounting differs")
    _require(contract["certified_confidence_lower_bound"] is None, "frozen confidence certificate differs")
    if family == "QSP_retargeted":
        phases = case["phase_evidence"]
        _require(type(phases) is dict and set(phases) == _PHASE_EVIDENCE_KEYS, "complete frozen phase evidence required")
        for name, length in (("cosine_phases", 15), ("sine_phases", 16)):
            _require(type(phases[name]) is list and len(phases[name]) == length and all(type(v) is float and math.isfinite(v) for v in phases[name]),
                     "controlled frozen phase word required")
        _require(phases["certified"] is False and type(phases["constructor_tolerance"]) in (int, float)
                 and math.isfinite(phases["constructor_tolerance"]) and phases["constructor_tolerance"] > 0,
                 "controlled phase tolerance/certificate differs")


def _allowed_float_path(case_id, path):
    parts = path.split("/")
    if len(parts) == 3 and parts[0] == "" and parts[1] == "numerical_checks":
        allowed = (_PF_NUMERICAL if case_id.startswith("pf") else
                   _RETARGETED_NUMERICAL if "retargeted" in case_id else _QSP_NUMERICAL)
        return parts[2] in allowed
    if len(parts) == 4 and parts[:3] == ["", "numerical_checks", "actual_block_singular_values"]:
        return case_id.startswith("qsp-oaa") and parts[3] in {str(i) for i in range(8)}
    if len(parts) == 3 and parts[:2] == ["", "postselection_contract"]:
        if case_id.startswith("pf"):
            return False
        return parts[2] in _POSTSELECTION_FLOATS or (
            case_id == "qsp-oaa2-n3-t1-eps0.05" and parts[2] == "conditional_synthesis_Pauli_bias_allowance")
    if "retargeted" not in case_id or len(parts) not in (4, 5) or parts[:2] != ["", "phase_evidence"]:
        return False
    branch = parts[2]
    if branch not in ("cosine", "sine"):
        return False
    if len(parts) == 4:
        projection = "uniform_error_bound_with_numerical_reconstruction_guard" if branch == "cosine" else "conditional_numerical_uniform_error_projection"
        return parts[3] in _PHASE_SCALARS | {projection}
    return parts[3] in {"target_chebyshev_coefficients", "achieved_chebyshev_coefficients"} and parts[4] in {
        str(i) for i in range(15 if branch == "cosine" else 16)}


def _typed_float(value):
    _require(type(value) is float and math.isfinite(value), "finite floating diagnostic required")
    return {"type": "float", "value": value, "hex": value.hex()}


def _float_change(path, recorded, replayed):
    old_value, new_value = _typed_float(recorded), _typed_float(replayed)
    delta = replayed - recorded
    _require(math.isfinite(delta), "finite floating difference required")
    return {"path": path, "recorded": old_value, "replayed": new_value,
            "delta": delta, "absolute_delta": abs(delta)}


def _case_float_changes(recorded, replayed):
    _require(recorded["case_id"] == replayed["case_id"], "source case identity differs")
    case_id, changes = recorded["case_id"], []

    def walk(first, second, pointer):
        _require(type(first) is type(second), "source field type differs at " + pointer)
        if type(first) is dict:
            _require(set(first) == set(second), "source field keys differ at " + pointer)
            for key in sorted(first):
                _require("/" not in key and "~" not in key, "unambiguous case keys required")
                walk(first[key], second[key], pointer + "/" + key)
        elif type(first) is list:
            _require(len(first) == len(second), "source array shape differs at " + pointer)
            for index, (old, new) in enumerate(zip(first, second)):
                walk(old, new, pointer + "/" + str(index))
        elif not _same(first, second):
            if pointer == "/qasm_sha256":
                _require(type(first) is str and _SHA256.fullmatch(first) and _SHA256.fullmatch(second), "invalid source QASM digest")
                return
            _require(type(first) is float and _allowed_float_path(case_id, pointer), "controlled source field differs at " + pointer)
            changes.append(_float_change(pointer, first, second))
            _require(len(changes) <= MAX_CHANGED_FLOATS, "floating difference limit exceeded")

    walk(recorded, replayed, "")
    return changes


def _decimal_fraction(literal):
    _require(type(literal) is str and len(literal) <= 64 and _DECIMAL_RE.fullmatch(literal), "bounded decimal angle required")
    try:
        value = Decimal(literal)
        binary = float(literal)
        _require(value.is_finite() and math.isfinite(binary), "finite decimal angle required")
        _require(abs(value) <= 64, "bounded decimal angle required")
        _require(abs(value.as_tuple().exponent) <= 400, "decimal exponent limit exceeded")
        return Fraction(value), Fraction.from_float(binary)
    except (InvalidOperation, OverflowError, ValueError) as error:
        raise ValueError("Dynamics source transport: invalid decimal angle") from error


def _parse_qasm(payload):
    _require(type(payload) is bytes and 0 < len(payload) <= MAX_QASM_BYTES, "bounded QASM byte snapshot required")
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise ValueError("Dynamics source transport: ASCII QASM required") from error
    lines = text.splitlines(keepends=True)
    _require(5 <= len(lines) <= MAX_QASM_LINES and all(line.endswith("\n") and "\r" not in line and len(line) <= 256 for line in lines),
             "bounded LF QASM lines required")
    _require(lines[:2] == ["OPENQASM 3.0;\n", 'include "stdgates.inc";\n'], "canonical QASM header differs")
    bit, qubit = _BIT_RE.fullmatch(lines[2]), _QUBIT_RE.fullmatch(lines[3])
    _require(bit is not None and qubit is not None, "canonical register declarations required")
    width = int(qubit.group(1))
    _require(1 <= width <= 9 and int(bit.group(1)) == width, "bounded complete readout registers required")
    rotations, measured, barrier, counts = {}, set(), False, {}

    def wire(number):
        value = int(number)
        _require(0 <= value < width, "QASM wire outside its register")
        return value

    for number, line in enumerate(lines[4:], 5):
        match = _MEASURE_RE.fullmatch(line)
        if match:
            target, source = wire(match.group(1)), wire(match.group(2))
            _require(barrier and target == source and source not in measured, "complete ordered final readout required")
            _require(source == len(measured), "final readout order differs")
            measured.add(source)
            counts["measure"] = counts.get("measure", 0) + 1
            continue
        _require(not measured, "operations after final readout are forbidden")
        expected_barrier = "barrier " + ", ".join(f"q[{i}]" for i in range(width)) + ";\n"
        if line == expected_barrier:
            _require(not barrier, "one complete readout barrier required")
            barrier = True
            counts["barrier"] = 1
            continue
        _require(not barrier, "operations after the readout barrier are forbidden")
        match = _ROTATION_RE.fullmatch(line)
        if match:
            opcode, literal, target = match.group(1), match.group(2), wire(match.group(3))
            if _DECIMAL_RE.fullmatch(literal):
                _decimal_fraction(literal)
            else:
                _require(len(literal) <= 64 and _PI_RE.fullmatch(literal), "unknown symbolic angle grammar")
            rotations[number] = (opcode, literal, target)
        else:
            match = _SIMPLE_RE.fullmatch(line)
            if match:
                opcode = match.group(1)
                wire(match.group(2))
            else:
                match = _CX_RE.fullmatch(line)
                _require(match is not None, "unknown QASM line grammar")
                opcode = "cx"
                _require(wire(match.group(1)) != wire(match.group(2)), "distinct CX wires required")
        counts[opcode] = counts.get(opcode, 0) + 1
    _require(barrier and measured == set(range(width)), "all declared wires require final readout")
    return lines, rotations, width, counts


def _allowed_literal_change(case_id, opcode, old, new):
    pairs = []
    if case_id.startswith("qsp-oaa") and opcode == "rx":
        pairs = [("0.8759866069501495", "0.8759866069501493")]
    elif case_id == "qsp-retargeted-n2-t1-eps0.05" and opcode == "ry":
        pairs = [("1.141020895490369", "1.1410208954903687")]
    elif case_id == "qsp-retargeted-n4-t1-eps0.05":
        if opcode == "rx":
            pairs = [("0.909839425867432", "0.9098394258674319")]
        elif opcode == "ry":
            pairs = [("0.6154797086703874", "0.6154797086703873")]
            if (old, new) == ("0.6456549348382472", "0.6456549348382471"):
                return True
    return (old, new) in pairs or (old, new) in [("-" + a, "-" + b) for a, b in pairs]


def _qasm_transport(case_id, canonical, replay):
    old_lines, old_rotations, width, counts = _parse_qasm(canonical)
    new_lines, new_rotations, new_width, new_counts = _parse_qasm(replay)
    _require(len(old_lines) == len(new_lines) and width == new_width and counts == new_counts,
             "QASM structure/counts differ")
    changes, decimal_l1, binary_l1 = [], Fraction(0), Fraction(0)
    for number, (old_line, new_line) in enumerate(zip(old_lines, new_lines), 1):
        if old_line == new_line:
            continue
        _require(number in old_rotations and number in new_rotations, "changed nonrotation QASM line")
        opcode, old_literal, old_wire = old_rotations[number]
        new_opcode, new_literal, new_wire = new_rotations[number]
        _require(opcode in ("rx", "ry") and opcode == new_opcode and old_wire == new_wire, "changed QASM opcode/wire")
        _require(_allowed_literal_change(case_id, opcode, old_literal, new_literal), "unreviewed decimal rotation change")
        _require(old_line.replace("(" + old_literal + ")", "(" + new_literal + ")", 1) == new_line,
                 "QASM bytes outside the decimal literal differ")
        old_decimal, old_binary = _decimal_fraction(old_literal)
        new_decimal, new_binary = _decimal_fraction(new_literal)
        decimal_l1 += abs(new_decimal - old_decimal)
        binary_l1 += abs(new_binary - old_binary)
        changes.append({"line": number, "opcode": opcode, "wire": old_wire,
                        "recorded_literal": old_literal, "replayed_literal": new_literal})
        _require(len(changes) <= MAX_CHANGED_ROTATIONS, "changed rotation limit exceeded")
    return {
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_sha256": hashlib.sha256(replay).hexdigest(),
        "byte_reproduced": canonical == replay,
        "line_count": len(old_lines), "rotation_changes": changes,
        "decimal_angle_l1_exact": str(decimal_l1), "decimal_rotation_product_bound_exact": str(decimal_l1 / 2),
        "binary64_angle_l1_exact": str(binary_l1), "binary64_rotation_product_bound_exact": str(binary_l1 / 2),
        "scope": ROTATION_SCOPE,
    }


def _checked_qasm_map(qasm, cases):
    _require(type(qasm) is dict and set(qasm) == {case["case_id"] for case in cases}, "QASM case map differs")
    total = 0
    for case in cases:
        payload = qasm[case["case_id"]]
        _require(type(payload) is bytes and 0 < len(payload) <= MAX_QASM_BYTES, "bounded QASM snapshot required")
        total += len(payload)
        _require(total <= MAX_QASM_AGGREGATE_BYTES, "aggregate QASM size limit exceeded")
        _require(hashlib.sha256(payload).hexdigest() == case["qasm_sha256"], "case QASM snapshot binding differs")
        _, _, width, counts = _parse_qasm(payload)
        if "logical_qubits" in case:
            _require(type(case["logical_qubits"]) is int and case["logical_qubits"] == width, "QASM logical width differs")
        if "compiled_gate_counts" in case:
            _require(_same(case["compiled_gate_counts"], counts), "QASM compiled operation counts differ")
    return dict(qasm)


def _full_study_check(report):
    summary = report["summary"]
    _require(tuple(row["case_id"] for row in report["cases"]) == CASE_IDS, "reviewed production roster/order required")
    _require(summary == {"canonical_cases": 24, "float_changes": 339, "qasm_hash_changes": 6,
                         "changed_qasm_files": 6, "changed_rotation_lines": 137,
                         "float_changes_by_section": EXPECTED_FLOAT_COUNTS}, "reviewed one-run difference coverage differs")
    actual = {row["case_id"]: len(row["qasm"]["rotation_changes"]) for row in report["cases"] if row["qasm"]["rotation_changes"]}
    _require(actual == EXPECTED_ROTATION_CHANGES, "reviewed per-case rotation difference counts differ")


def _build_source_transport(canonical_cases, replay_cases, canonical_qasm, replay_qasm, provenance, *, require_full_study=False):
    """Internal bounded fixture helper; public builder requires the full study."""
    original = _checked_cases(canonical_cases, require_full_study=require_full_study)
    source = _checked_cases(replay_cases, require_full_study=require_full_study)
    _require([c["case_id"] for c in original] == [c["case_id"] for c in source], "source roster/order differs")
    public = _checked_provenance(provenance, provenance.get("replay_source_sha256") if type(provenance) is dict else None)
    old_qasm, new_qasm = _checked_qasm_map(canonical_qasm, original), _checked_qasm_map(replay_qasm, source)
    rows, by_section = [], {name: 0 for name in EXPECTED_FLOAT_COUNTS}
    for old, new in zip(original, source):
        case_id = old["case_id"]
        changes = _case_float_changes(old, new)
        for change in changes:
            by_section[change["path"].split("/")[1]] += 1
        rows.append({
            "case_id": case_id, "canonical_case_sha256": _digest(old),
            "reconstructed_source_case_sha256": _digest(new),
            "frozen_postselection_contract_sha256": _digest(old["postselection_contract"]) if "postselection_contract" in old else None,
            "float_changes": changes, "qasm": _qasm_transport(case_id, old_qasm[case_id], new_qasm[case_id]),
        })
    summary = {"canonical_cases": len(rows), "float_changes": sum(by_section.values()),
               "qasm_hash_changes": sum(old["qasm_sha256"] != new["qasm_sha256"] for old, new in zip(original, source)),
               "changed_qasm_files": sum(not row["qasm"]["byte_reproduced"] for row in rows),
               "changed_rotation_lines": sum(len(row["qasm"]["rotation_changes"]) for row in rows),
               "float_changes_by_section": by_section}
    _require(summary["float_changes"] or summary["changed_qasm_files"], "a source difference is required")
    _require(summary["qasm_hash_changes"] == summary["changed_qasm_files"], "source QASM hash difference coverage differs")
    _require(summary["float_changes"] <= MAX_CHANGED_FLOATS and summary["changed_rotation_lines"] <= MAX_CHANGED_ROTATIONS,
             "aggregate difference limit exceeded")
    report = {"schema_version": 1, "status": "source_not_byte_reproduced", "scope": REPORT_SCOPE,
              "provenance": public, "canonical_cases_sha256": _digest(original),
              "reconstructed_source_cases_sha256": _digest(source), "summary": summary, "cases": rows,
              "claims": copy.deepcopy(CLAIMS)}
    if require_full_study:
        _full_study_check(report)
    return report


def build_source_transport(canonical_cases, replay_cases, canonical_qasm, replay_qasm, provenance):
    """Build the sanitized disclosure for the preserved, reviewed Linux study."""
    return _build_source_transport(canonical_cases, replay_cases, canonical_qasm, replay_qasm, provenance, require_full_study=True)


def _pointer_parent(value, pointer):
    _require(type(pointer) is str and pointer.startswith("/") and len(pointer) <= 512, "bounded diagnostic pointer required")
    parts = pointer[1:].split("/")
    parent = value
    for key in parts[:-1]:
        if type(parent) is dict:
            _require(key in parent, "diagnostic pointer is absent")
            parent = parent[key]
        elif type(parent) is list:
            _require(key.isdigit() and str(int(key)) == key and int(key) < len(parent), "diagnostic array pointer differs")
            parent = parent[int(key)]
        else:
            raise ValueError("Dynamics source transport: diagnostic pointer does not reach a leaf")
    key = parts[-1]
    if type(parent) is dict:
        _require(key in parent, "diagnostic leaf is absent")
    else:
        _require(type(parent) is list and key.isdigit() and str(int(key)) == key and int(key) < len(parent), "diagnostic array leaf differs")
        key = int(key)
    return parent, key


def _apply_float_changes(case, changes):
    _require(type(changes) is list and len(changes) <= MAX_CHANGED_FLOATS, "bounded diagnostic changes required")
    result, seen = copy.deepcopy(case), set()
    for change in changes:
        _require(type(change) is dict and set(change) == _FLOAT_KEYS, "typed floating change schema differs")
        pointer = change["path"]
        _require(type(pointer) is str and len(pointer) <= 512, "bounded diagnostic path required")
        _require(pointer not in seen and _allowed_float_path(case["case_id"], pointer), "duplicate or unreviewed floating path")
        seen.add(pointer)
        parent, key = _pointer_parent(result, pointer)
        old = parent[key]
        _require(type(change["replayed"]) is dict and set(change["replayed"]) == {"type", "value", "hex"}, "typed replay value differs")
        new = change["replayed"]["value"]
        expected = _float_change(pointer, old, new)
        _require(_same(change, expected) and not _same(old, new), "floating values/hex/deltas differ")
        parent[key] = new
    return result


def _apply_qasm_changes(case_id, canonical, metadata):
    _require(type(metadata) is dict and set(metadata) == _QASM_KEYS, "QASM transport schema differs")
    lines, rotations, _, _ = _parse_qasm(canonical)
    changes = metadata["rotation_changes"]
    _require(type(changes) is list and len(changes) <= MAX_CHANGED_ROTATIONS, "bounded rotation changes required")
    result, seen = list(lines), set()
    for change in changes:
        _require(type(change) is dict and set(change) == _ROTATION_KEYS, "rotation replacement schema differs")
        number = change["line"]
        _require(type(number) is int and number in rotations and number not in seen, "duplicate or absent changed rotation line")
        seen.add(number)
        opcode, old_literal, wire = rotations[number]
        _require(type(change["wire"]) is int and (change["opcode"], change["recorded_literal"], change["wire"]) == (opcode, old_literal, wire),
                 "recorded rotation replacement differs")
        new_literal = change["replayed_literal"]
        _require(_allowed_literal_change(case_id, opcode, old_literal, new_literal), "unreviewed replacement literal")
        result[number - 1] = lines[number - 1].replace("(" + old_literal + ")", "(" + new_literal + ")", 1)
    replay = "".join(result).encode("ascii")
    _require(_same(metadata, _qasm_transport(case_id, canonical, replay)), "reconstructed QASM hashes/bounds/differences disagree")
    return replay


def _validate_source_transport(report, canonical_cases, canonical_qasm, expected_source_hashes, *, require_full_study=False):
    """Validate static reconstruction; this function does no circuit execution."""
    _checked_json(report)
    _require(type(report) is dict and set(report) == _REPORT_KEYS, "public report schema differs")
    _require(type(report["schema_version"]) is int and report["schema_version"] == 1
             and report["status"] == "source_not_byte_reproduced" and report["scope"] == REPORT_SCOPE
             and _same(report["claims"], CLAIMS), "report scope/claims differ")
    provenance = _checked_provenance(report["provenance"], expected_source_hashes)
    original = _checked_cases(canonical_cases, require_full_study=require_full_study)
    qasm = _checked_qasm_map(canonical_qasm, original)
    _require(type(report["cases"]) is list and len(report["cases"]) == len(original), "report case roster differs")
    source, source_qasm = [], {}
    for case, row in zip(original, report["cases"]):
        _require(type(row) is dict and set(row) == _CASE_KEYS and row["case_id"] == case["case_id"], "report case identity/schema differs")
        _require(row["canonical_case_sha256"] == _digest(case), "canonical case digest differs")
        frozen = _digest(case["postselection_contract"]) if "postselection_contract" in case else None
        _require(row["frozen_postselection_contract_sha256"] == frozen, "authoritative frozen contract differs")
        rebuilt = _apply_float_changes(case, row["float_changes"])
        payload = _apply_qasm_changes(case["case_id"], qasm[case["case_id"]], row["qasm"])
        rebuilt["qasm_sha256"] = hashlib.sha256(payload).hexdigest()
        _require(row["reconstructed_source_case_sha256"] == _digest(rebuilt), "reconstructed source case digest differs")
        source.append(rebuilt)
        source_qasm[case["case_id"]] = payload
    expected = _build_source_transport(original, source, qasm, source_qasm, provenance, require_full_study=require_full_study)
    _require(_same(report, expected), "static source report is incomplete or inconsistent")
    return {"status": "validated_static_source_transport_report", "source_status": "source_not_byte_reproduced",
            "report_digest_sha256": _digest(report), "canonical_circuit_validation_performed": False,
            "costs_repriced": False, **copy.deepcopy(report["summary"])}


def validate_source_transport(report, canonical_cases, canonical_qasm, expected_source_hashes):
    """Check the complete reviewed report against authenticated canonical data."""
    return _validate_source_transport(report, canonical_cases, canonical_qasm, expected_source_hashes, require_full_study=True)
