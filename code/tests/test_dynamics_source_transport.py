"""Synthetic transport evidence and mutations; no quantum/Native execution."""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "tested_dynamics_source_transport", ROOT / "scripts/dynamics_source_transport.py"
)
t = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(t)

CASE_ID = "qsp-oaa0-n3-t1-eps0.05"
OLD_ANGLE = "0.8759866069501495"
NEW_ANGLE = "0.8759866069501493"
FALSE_CLAIMS = (
    "canonical_circuit_validation_performed", "source_circuit_equivalence_claimed",
    "formal_certificate", "hardware_or_synthesis_certificate", "Native_estimates_run",
    "phase_fitter_run", "costs_repriced",
)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def canonical_sha(value):
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode())


def qasm(angle=OLD_ANGLE, *, width=8):
    return ("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
            + f"bit[{width}] meas;\nqubit[{width}] q;\n"
            + f"rx({angle}) q[{width-2}];\nry(0.5) q[{width-1}];\nrz(pi/4) q[0];\n"
            + "barrier " + ", ".join(f"q[{n}]" for n in range(width)) + ";\n"
            + "".join(f"meas[{n}] = measure q[{n}];\n" for n in range(width))).encode()


@pytest.fixture
def transport_inputs():
    original_qasm = {CASE_ID: qasm()}
    source_qasm = {CASE_ID: qasm(NEW_ANGLE)}
    old = {
        "case_id": CASE_ID, "method": "QSP_oblivious_amplification",
        "system_qubits": 3, "logical_qubits": 8, "rounds": 0,
        "complete_attempts": 100, "initial_state": "000", "observable": "Z(q0)",
        "source_gate_counts": {"h": 2}, "source_depth": 1,
        "compiled_gate_counts": {"rx": 1, "ry": 1, "rz": 1, "barrier": 1, "measure": 8},
        "compiled_depth": 4, "compiled_global_phase_radians": 0.0,
        "access": {"total_A_or_inverse_calls": 1},
        "numerical_checks": {"actual_input_acceptance_probability": 0.0625},
        "postselection_contract": {
            "reference_acceptance_lower": 0.062499999,
            "conditional_acceptance_lower_after_synthesis_model": 0.062499998,
            "synthesis_model_error_per_attempt": 0.000001,
            "attempt_cap_per_accepted_sample": 10,
            "maximum_complete_attempts": 100,
            "conditional_expected_stopped_attempts_all_slots": 90.0,
        },
        "qasm_sha256": sha(original_qasm[CASE_ID]),
    }
    new = copy.deepcopy(old)
    new["numerical_checks"]["actual_input_acceptance_probability"] = math.nextafter(0.0625, math.inf)
    new["qasm_sha256"] = sha(source_qasm[CASE_ID])
    sources = {name: "a" * 64 for name in t.REPLAY_SOURCE_FILES}
    provenance = {
        "replay_source_commit": t.REVIEWED_REPLAY_COMMIT,
        "replay_source_sha256": copy.deepcopy(sources),
        "recorded_producer_commit": t.NATIVE_PRODUCER_COMMIT,
        "original_study_sha256": t.ORIGINAL_STUDY_SHA256,
        "original_core_inventory_sha256": t.ORIGINAL_CORE_INVENTORY_SHA256,
        "locked_versions": copy.deepcopy(t.LOCKED_VERSIONS),
        "architecture": "x86_64",
    }
    return {"canonical": [old], "source": [new], "old_qasm": original_qasm,
            "new_qasm": source_qasm, "provenance": provenance, "sources": sources}


def build(inputs):
    return t._build_source_transport(
        inputs["canonical"], inputs["source"], inputs["old_qasm"], inputs["new_qasm"],
        inputs["provenance"], require_full_study=False,
    )


def validate(report, inputs):
    return t._validate_source_transport(
        report, inputs["canonical"], inputs["old_qasm"], inputs["sources"],
        require_full_study=False,
    )


def full_roster_inputs(compact):
    """Complete synthetic control schema with deliberately incomplete drift."""
    inputs = copy.deepcopy(compact)
    canonical, source, old_qasm, new_qasm = [], [], {}, {}
    for case_id in t.CASE_IDS:
        family = ("product_formula" if case_id.startswith("pf") else
                  "QSP_retargeted" if "retargeted" in case_id else "QSP_oblivious_amplification")
        system_width = int(case_id.split("-n", 1)[1].split("-", 1)[0])
        width = (system_width if family == "product_formula" else
                 6 if system_width == 2 else 9 if system_width == 4 else 8)
        old_qasm[case_id] = qasm(width=width)
        new_qasm[case_id] = qasm(NEW_ANGLE, width=width) if case_id == CASE_ID else qasm(width=width)
        task = {
            "absolute_error": 0.05, "algorithm_bias": 0.01, "synthesis_bias": 0.01,
            "sampling_error": 0.03, "sampling_failure": 0.004, "acceptance_failure": 0.001,
            "physical_failure": 0.005, "accepted_samples": 2,
            "whole_task_failure_allowance": 0.01, "certified_confidence_lower_bound": None,
        }
        old = {
            "case_id": case_id, "method": family, "J": 1.0, "h": 0.7, "hbar": 1.0, "time": 1.0,
            "accepted_slots": 2, "attempts_per_accepted_slot": 3, "complete_attempts": 6,
            "compiled_depth": 4, "compiled_gate_counts": {"rx": 1, "ry": 1, "rz": 1, "barrier": 1, "measure": width},
            "compiled_global_phase_radians": 0.0, "source_depth": 3,
            "source_gate_counts": {"rx": 1, "ry": 1, "rz": 1},
            "system_qubits": system_width, "logical_qubits": width, "initial_state": "0" * system_width,
            "input_preparation": "Synthetic clean-input placeholder; no state preparation executed.",
            "measurement": {"basis": "Z", "all_wires": width}, "observable": "Z(q0)",
            "exclusions": ["Synthetic schema fixture; no algorithm or resource validation."],
            "provenance": {"source_revision_scope": t.PROVENANCE_SCOPE,
                           "environment": copy.deepcopy(t.CANONICAL_ENVIRONMENT)},
            "qasm_global_phase_scope": "Synthetic measured trial only.",
            "qasm_sha256": sha(old_qasm[case_id]), "task": task,
            "numerical_checks": {"operator_error": 0.001} if family == "product_formula" else {
                "actual_input_acceptance_probability": 0.0625},
        }
        if family == "product_formula":
            old.update(acceptance_probability=1.0, analytic_error={"scope": "synthetic"},
                       boundary="Synthetic static-circuit boundary.", no_acceptance_failure_actual=0.0,
                       order=1 if case_id.startswith("pf1") else 2, steps=2,
                       terms=[[1.0, "Z" * system_width]])
        else:
            contract = {
                "reference_acceptance_lower": 0.062499999,
                "synthesis_model_error_per_attempt": 0.000001,
                "conditional_acceptance_lower_after_synthesis_model": 0.062499998,
                "all_slots_no_acceptance_union_bound": 0.0005,
                "conditional_expected_stopped_attempts_all_slots": 5.0,
                "conditional_synthesis_Pauli_bias_allowance": 0.01,
                "attempts_per_accepted_slot": 3, "accepted_slots": 2, "maximum_complete_attempts": 6,
                "physical_failure_allowance_per_complete_attempt": 0.005 / 6,
                "certified_confidence_lower_bound": None, "conditional_confidence_lower_bound": 0.99,
                "assumptions": ["Synthetic controlled accounting."],
                "proof": "Synthetic fixture; no scientific proof.",
                "does_not_certify": ["hardware", "synthesis"],
            }
            old.update(access={
                "forward_A_calls": 1, "inverse_A_calls": 0, "total_A_or_inverse_calls": 1,
                "zero_ancilla_reflections": 0, "uncontrolled_block_encoding_calls": 1,
                "controlled_block_encoding_calls": 0, "branch_conditioned_projector_phase_calls": 1,
            }, phase_tau=2.4 if system_width == 2 else 5.8 if system_width == 4 else 4.1,
                       postselection_contract=contract,
                       rounds=0 if family == "QSP_retargeted" else int(case_id[7]),
                       size_scaling_scope="Synthetic schema coverage only.")
            if family == "QSP_retargeted":
                old["phase_evidence"] = {
                    "cosine_phases": [0.1] * 15, "sine_phases": [0.2] * 16,
                    "cosine": {"scope": "synthetic"}, "sine": {"scope": "synthetic"},
                    "constructor_tolerance": 1e-8, "phase_fit_method": "Synthetic frozen-word placeholder.",
                    "certified": False,
                }
        new = copy.deepcopy(old)
        if case_id == CASE_ID:
            new["numerical_checks"]["actual_input_acceptance_probability"] = math.nextafter(0.0625, math.inf)
        new["qasm_sha256"] = sha(new_qasm[case_id])
        canonical.append(old)
        source.append(new)
    inputs.update(canonical=canonical, source=source, old_qasm=old_qasm, new_qasm=new_qasm)
    return inputs


def assert_full_schema_reaches_coverage_guard(inputs):
    """Establish the complete-schema baseline before mutating required controls."""
    report = build(inputs)
    validate(report, inputs)
    with pytest.raises(ValueError, match="reviewed one-run difference coverage"):
        t.build_source_transport(inputs["canonical"], inputs["source"], inputs["old_qasm"],
                                 inputs["new_qasm"], inputs["provenance"])
    return report


@pytest.mark.parametrize("character", ["x", "\U0001f9ea"], ids=["ASCII", "non-BMP"])
def test_aggregate_json_aliases_inside_the_two_mib_budget_are_accepted(character):
    assert t.MAX_JSON_BYTES == 2 * 1024 * 1024
    token = character * 1024
    payload = [token] * 170
    assert all(value is token for value in payload)
    assert len(json.dumps(payload, ensure_ascii=True, separators=(",", ":")).encode("ascii")) < 2 * 1024 * 1024
    assert t._checked_json(payload) is payload


@pytest.mark.parametrize("character", ["x", "\U0001f9ea"], ids=["ASCII", "non-BMP"])
@pytest.mark.parametrize("entry_point", ["build", "validate"])
def test_aggregate_json_budget_rejects_aliases_before_copy_or_serialization(transport_inputs, monkeypatch,
                                                                         character, entry_point):
    report = build(transport_inputs)
    token = character * 1024
    payload = [token] * 171
    assert all(value is token for value in payload)
    if entry_point == "build":
        transport_inputs["canonical"][0]["aggregate_payload"] = payload
        transport_inputs["source"][0]["aggregate_payload"] = payload
    else:
        report["aggregate_payload"] = payload

    def forbidden(*args, **kwargs):
        raise AssertionError("Oversized JSON reached copying or serialization")

    with monkeypatch.context() as guard:
        guard.setattr(t.copy, "deepcopy", forbidden)
        guard.setattr(t, "_canonical", forbidden)
        with pytest.raises(ValueError, match="JSON byte budget exceeded"):
            if entry_point == "build":
                build(transport_inputs)
            else:
                validate(report, transport_inputs)


def test_compact_roundtrip_records_drift_without_upgrading_its_scope(transport_inputs):
    inputs = transport_inputs
    before = copy.deepcopy(inputs)
    report = build(inputs)
    validate(report, inputs)
    assert inputs == before
    assert report["status"] == "source_not_byte_reproduced"
    assert all(report["claims"][claim] is False for claim in FALSE_CLAIMS)
    assert report["claims"]["canonical_costs_authoritative"] is True
    assert report["summary"]["canonical_cases"] == 1
    assert report["summary"]["float_changes"] == 1
    assert report["summary"]["qasm_hash_changes"] == 1
    assert report["summary"]["changed_rotation_lines"] == 1
    assert report["canonical_cases_sha256"] == canonical_sha(inputs["canonical"])
    assert report["reconstructed_source_cases_sha256"] == canonical_sha(inputs["source"])


def test_rotation_evidence_independently_reconstructs_bytes_and_exact_bounds(transport_inputs):
    report = build(transport_inputs)
    evidence = report["cases"][0]["qasm"]
    assert evidence["canonical_sha256"] == sha(transport_inputs["old_qasm"][CASE_ID])
    assert evidence["source_sha256"] == sha(transport_inputs["new_qasm"][CASE_ID])
    assert evidence["byte_reproduced"] is False
    assert evidence["line_count"] == 16
    assert evidence["rotation_changes"] == [{
        "line": 5, "opcode": "rx", "wire": 6,
        "recorded_literal": OLD_ANGLE, "replayed_literal": NEW_ANGLE,
    }]
    decimal_change = abs(Fraction(NEW_ANGLE) - Fraction(OLD_ANGLE))
    binary_change = abs(Fraction.from_float(float(NEW_ANGLE)) - Fraction.from_float(float(OLD_ANGLE)))
    assert Fraction(evidence["decimal_angle_l1_exact"]) == decimal_change
    assert Fraction(evidence["decimal_rotation_product_bound_exact"]) == decimal_change / 2
    assert Fraction(evidence["binary64_angle_l1_exact"]) == binary_change
    assert Fraction(evidence["binary64_rotation_product_bound_exact"]) == binary_change / 2


def test_float_evidence_preserves_types_hex_and_canonical_contract(transport_inputs):
    case = build(transport_inputs)["cases"][0]
    old = transport_inputs["canonical"][0]
    new = transport_inputs["source"][0]
    original = old["numerical_checks"]["actual_input_acceptance_probability"]
    replayed = new["numerical_checks"]["actual_input_acceptance_probability"]
    assert case["float_changes"] == [{
        "path": "/numerical_checks/actual_input_acceptance_probability",
        "recorded": {"type": "float", "value": original, "hex": original.hex()},
        "replayed": {"type": "float", "value": replayed, "hex": replayed.hex()},
        "delta": replayed - original, "absolute_delta": abs(replayed - original),
    }]
    assert case["canonical_case_sha256"] == canonical_sha(old)
    assert case["reconstructed_source_case_sha256"] == canonical_sha(new)
    assert case["frozen_postselection_contract_sha256"] == canonical_sha(old["postselection_contract"])


def test_replayed_synthesis_drift_does_not_replace_the_authoritative_contract(transport_inputs):
    old_contract = copy.deepcopy(transport_inputs["canonical"][0]["postselection_contract"])
    new_contract = transport_inputs["source"][0]["postselection_contract"]
    new_contract["synthesis_model_error_per_attempt"] = math.nextafter(
        new_contract["synthesis_model_error_per_attempt"], math.inf
    )
    report = build(transport_inputs)
    validate(report, transport_inputs)
    assert report["summary"]["float_changes"] == 2
    assert any(row["path"] == "/postselection_contract/synthesis_model_error_per_attempt"
               for row in report["cases"][0]["float_changes"])
    assert report["cases"][0]["frozen_postselection_contract_sha256"] == canonical_sha(old_contract)
    assert report["cases"][0]["frozen_postselection_contract_sha256"] != canonical_sha(new_contract)
    assert report["claims"]["canonical_costs_authoritative"] is True
    assert report["claims"]["costs_repriced"] is False
    assert transport_inputs["canonical"][0]["postselection_contract"] == old_contract


def test_public_builder_requires_the_complete_recorded_study(transport_inputs):
    build(transport_inputs)
    with pytest.raises(ValueError):
        t.build_source_transport(transport_inputs["canonical"], transport_inputs["source"],
                                 transport_inputs["old_qasm"], transport_inputs["new_qasm"],
                                 transport_inputs["provenance"])


def test_public_validator_rejects_an_otherwise_valid_compact_report(transport_inputs):
    report = build(transport_inputs)
    validate(report, transport_inputs)
    with pytest.raises(ValueError):
        t.validate_source_transport(report, transport_inputs["canonical"],
                                    transport_inputs["old_qasm"], transport_inputs["sources"])


@pytest.mark.parametrize("entry_point", ["build", "validate"])
def test_public_api_rejects_incomplete_coverage_even_with_all_twenty_four_cases(transport_inputs, entry_point):
    inputs = full_roster_inputs(transport_inputs)
    assert len(inputs["canonical"]) == 24
    assert tuple(case["case_id"] for case in inputs["canonical"]) == t.CASE_IDS
    report = build(inputs)
    validate(report, inputs)
    assert report["summary"]["float_changes"] == 1
    assert report["summary"]["changed_rotation_lines"] == 1
    with pytest.raises(ValueError, match="reviewed one-run difference coverage"):
        if entry_point == "build":
            t.build_source_transport(inputs["canonical"], inputs["source"], inputs["old_qasm"],
                                     inputs["new_qasm"], inputs["provenance"])
        else:
            t.validate_source_transport(report, inputs["canonical"], inputs["old_qasm"], inputs["sources"])


def test_public_api_keeps_the_full_roster_order_fixed(transport_inputs):
    inputs = full_roster_inputs(transport_inputs)
    inputs["canonical"].reverse()
    inputs["source"].reverse()
    build(inputs)
    with pytest.raises(ValueError, match="roster/order"):
        t.build_source_transport(inputs["canonical"], inputs["source"], inputs["old_qasm"],
                                 inputs["new_qasm"], inputs["provenance"])


@pytest.mark.parametrize("case_id,path,message", [
    ("pf1-n2-t1-eps0.05", "task", "required complete canonical/source case fields"),
    ("pf1-n2-t1-eps0.05", "complete_attempts", "required complete canonical/source case fields"),
    ("pf1-n2-t1-eps0.05", "measurement", "required complete canonical/source case fields"),
    (CASE_ID, "rounds", "required complete canonical/source case fields"),
    (CASE_ID, "phase_tau", "required complete canonical/source case fields"),
    (CASE_ID, "access", "required complete canonical/source case fields"),
    (CASE_ID, "postselection_contract", "required complete canonical/source case fields"),
    ("qsp-retargeted-n2-t1-eps0.05", "phase_evidence", "required complete canonical/source case fields"),
    ("pf1-n2-t1-eps0.05", "task/accepted_samples", "complete fixed task fields"),
    (CASE_ID, "access/total_A_or_inverse_calls", "complete controlled access counts"),
    (CASE_ID, "postselection_contract/maximum_complete_attempts", "complete frozen postselection contract"),
    ("qsp-retargeted-n2-t1-eps0.05", "phase_evidence/cosine_phases", "complete frozen phase evidence"),
])
def test_public_full_schema_requires_control_fields_even_when_both_snapshots_omit_them(transport_inputs,
                                                                                    case_id, path, message):
    inputs = full_roster_inputs(transport_inputs)
    assert_full_schema_reaches_coverage_guard(inputs)
    parts = path.split("/")
    for side in ("canonical", "source"):
        value = next(case for case in inputs[side] if case["case_id"] == case_id)
        for part in parts[:-1]:
            value = value[part]
        value.pop(parts[-1])
    with pytest.raises(ValueError, match=message):
        t.build_source_transport(inputs["canonical"], inputs["source"], inputs["old_qasm"],
                                 inputs["new_qasm"], inputs["provenance"])


@pytest.mark.parametrize("case_id,path,value,message", [
    ("pf1-n2-t1-eps0.05", "accepted_slots", True, "positive controlled integer"),
    ("pf1-n2-t1-eps0.05", "complete_attempts", 7, "complete retry accounting"),
    ("pf1-n2-t1-eps0.05", "source_depth", True, "controlled integer depth"),
    ("pf1-n2-t1-eps0.05", "source_gate_counts", {}, "controlled gate counts"),
    ("pf1-n2-t1-eps0.05", "task/accepted_samples", 3, "controlled task shots/certificate"),
    ("pf1-n2-t1-eps0.05", "task/sampling_failure", 0.0, "finite positive task allocations"),
    (CASE_ID, "rounds", 4, "controlled amplification rounds"),
    (CASE_ID, "phase_tau", 0.0, "controlled positive phase tau"),
    (CASE_ID, "access/total_A_or_inverse_calls", -1, "complete controlled access counts"),
    (CASE_ID, "postselection_contract/maximum_complete_attempts", 7, "frozen contract retry accounting"),
    ("qsp-retargeted-n2-t1-eps0.05", "phase_evidence/cosine_phases", [0.1] * 14, "controlled frozen phase word"),
    ("qsp-retargeted-n4-t1-eps0.05", "phase_evidence/sine_phases", [0.2] * 15, "controlled frozen phase word"),
    ("qsp-retargeted-n2-t1-eps0.05", "phase_evidence/cosine_phases", [0] * 15, "controlled frozen phase word"),
    ("qsp-retargeted-n2-t1-eps0.05", "phase_evidence/certified", True, "controlled phase tolerance/certificate"),
])
def test_public_full_schema_enforces_control_types_and_accounting(transport_inputs, case_id, path, value, message):
    inputs = full_roster_inputs(transport_inputs)
    assert_full_schema_reaches_coverage_guard(inputs)
    parts = path.split("/")
    for side in ("canonical", "source"):
        target = next(case for case in inputs[side] if case["case_id"] == case_id)
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = copy.deepcopy(value)
    with pytest.raises(ValueError, match=message):
        t.build_source_transport(inputs["canonical"], inputs["source"], inputs["old_qasm"],
                                 inputs["new_qasm"], inputs["provenance"])


@pytest.mark.parametrize("field", ["system_qubits", "rounds", "complete_attempts", "source_depth",
                                    "compiled_depth", "compiled_global_phase_radians"])
def test_builder_rejects_changes_outside_reviewed_float_paths(transport_inputs, field):
    transport_inputs["source"][0][field] += 1
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("value", [True, 1, None, "0.0625", math.inf, -math.inf, math.nan])
def test_builder_rejects_type_changes_and_nonfinite_diagnostics(transport_inputs, value):
    transport_inputs["source"][0]["numerical_checks"]["actual_input_acceptance_probability"] = value
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("field", ["attempt_cap_per_accepted_sample", "maximum_complete_attempts"])
def test_builder_never_transports_discrete_retry_decisions(transport_inputs, field):
    transport_inputs["source"][0]["postselection_contract"][field] += 1
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("change", ["opcode", "wire", "symbolic_angle", "readout", "extra_gate", "header", "barrier"])
def test_builder_rejects_any_structural_qasm_change(transport_inputs, change):
    payload = transport_inputs["new_qasm"][CASE_ID]
    replacements = {
        "opcode": (b"rx(", b"ry("), "wire": (b") q[6];", b") q[5];"),
        "symbolic_angle": (b"rz(pi/4)", b"rz(pi/8)"),
        "readout": (b"meas[0] = measure q[0]", b"meas[0] = measure q[1]"),
        "extra_gate": (b"ry(0.5) q[7];", b"ry(0.5) q[7];\nx q[0];"),
        "header": (b"qubit[8]", b"qubit[9]"),
        "barrier": (b"barrier q[0],", b"barrier q[1],"),
    }
    old, new = replacements[change]
    payload = payload.replace(old, new, 1)
    transport_inputs["new_qasm"][CASE_ID] = payload
    transport_inputs["source"][0]["qasm_sha256"] = sha(payload)
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("side", ["canonical", "source"])
def test_builder_authenticates_each_qasm_against_its_own_case(transport_inputs, side):
    transport_inputs[side][0]["qasm_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("side", ["canonical", "source"])
def test_duplicate_or_unknown_cases_cannot_be_transport_evidence(transport_inputs, side):
    transport_inputs[side].append(copy.deepcopy(transport_inputs[side][0]))
    with pytest.raises(ValueError):
        build(transport_inputs)


def test_unknown_case_is_rejected_even_when_both_sides_agree(transport_inputs):
    for key in ("canonical", "source"):
        transport_inputs[key][0]["case_id"] = "unreviewed-case"
    for key in ("old_qasm", "new_qasm"):
        transport_inputs[key]["unreviewed-case"] = transport_inputs[key].pop(CASE_ID)
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("side", ["old_qasm", "new_qasm"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_qasm_maps_cover_exactly_the_case_roster(transport_inputs, side, change):
    if change == "missing":
        transport_inputs[side].pop(CASE_ID)
    else:
        transport_inputs[side]["unreviewed-case"] = qasm()
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("field", ["replay_source_commit", "recorded_producer_commit",
                                    "original_study_sha256", "original_core_inventory_sha256",
                                    "architecture"])
def test_builder_checks_the_fixed_provenance_identity(transport_inputs, field):
    transport_inputs["provenance"][field] = "wrong"
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("change", ["extra_key", "missing_source", "invalid_digest", "version"])
def test_provenance_cannot_silently_change_scope_or_environment(transport_inputs, change):
    p = transport_inputs["provenance"]
    if change == "extra_key":
        p["source_equivalence_certified"] = True
    elif change == "missing_source":
        p["replay_source_sha256"].pop(next(iter(p["replay_source_sha256"])))
    elif change == "invalid_digest":
        p["replay_source_sha256"][next(iter(p["replay_source_sha256"]))] = "not-a-digest"
    else:
        p["locked_versions"][next(iter(p["locked_versions"]))] = "wrong-version"
    with pytest.raises(ValueError):
        build(transport_inputs)


@pytest.mark.parametrize("claim", FALSE_CLAIMS)
def test_validator_rejects_unearned_validation_or_execution_claims(transport_inputs, claim):
    report = build(transport_inputs)
    report["claims"][claim] = True
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


def test_validator_keeps_canonical_costs_authoritative(transport_inputs):
    report = build(transport_inputs)
    report["claims"]["canonical_costs_authoritative"] = False
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("value", [0, None, "false"])
def test_scope_claims_require_boolean_false(transport_inputs, value):
    report = build(transport_inputs)
    report["claims"]["source_circuit_equivalence_claimed"] = value
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


def test_authoritative_cost_claim_requires_boolean_true(transport_inputs):
    report = build(transport_inputs)
    report["claims"]["canonical_costs_authoritative"] = 1
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("location", ["report", "case", "qasm", "float_change", "rotation", "typed_float"])
def test_validator_rejects_undeclared_report_extensions(transport_inputs, location):
    report = build(transport_inputs)
    case = report["cases"][0]
    targets = {
        "report": report, "case": case, "qasm": case["qasm"],
        "float_change": case["float_changes"][0],
        "rotation": case["qasm"]["rotation_changes"][0],
        "typed_float": case["float_changes"][0]["replayed"],
    }
    targets[location]["unearned_certificate"] = True
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field", ["canonical_cases_sha256", "reconstructed_source_cases_sha256", "status", "scope"])
def test_validator_rejects_modified_report_identity(transport_inputs, field):
    report = build(transport_inputs)
    report[field] = "changed"
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field", ["canonical_case_sha256", "reconstructed_source_case_sha256",
                                    "frozen_postselection_contract_sha256", "case_id"])
def test_validator_rejects_modified_case_bindings(transport_inputs, field):
    report = build(transport_inputs)
    report["cases"][0][field] = "changed"
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("change", ["omit", "duplicate", "path", "old_hex", "new_hex", "new_value", "delta", "absolute_delta"])
def test_validator_checks_typed_float_transport_completely(transport_inputs, change):
    report = build(transport_inputs)
    changes = report["cases"][0]["float_changes"]
    if change == "omit":
        changes.clear()
    elif change == "duplicate":
        changes.append(copy.deepcopy(changes[0]))
    else:
        row = changes[0]
        if change == "path":
            row["path"] = "/complete_attempts"
        elif change == "old_hex":
            row["recorded"]["hex"] = (0.5).hex()
        elif change == "new_hex":
            row["replayed"]["hex"] = (0.5).hex()
        elif change == "new_value":
            row["replayed"]["value"] = 0.5
        else:
            row[change] = 0.5
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field", ["line", "opcode", "wire", "recorded_literal", "replayed_literal"])
def test_validator_reconstructs_qasm_instead_of_trusting_rotation_rows(transport_inputs, field):
    report = build(transport_inputs)
    row = report["cases"][0]["qasm"]["rotation_changes"][0]
    row[field] = {"line": 6, "opcode": "ry", "wire": 5,
                  "recorded_literal": "0.5", "replayed_literal": "0.4"}[field]
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field", ["canonical_sha256", "source_sha256", "line_count",
                                    "decimal_angle_l1_exact", "decimal_rotation_product_bound_exact",
                                    "binary64_angle_l1_exact", "binary64_rotation_product_bound_exact", "scope"])
def test_validator_checks_rotation_hashes_counts_and_rational_arithmetic(transport_inputs, field):
    report = build(transport_inputs)
    evidence = report["cases"][0]["qasm"]
    evidence[field] = 100 if field == "line_count" else "wrong"
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("change", ["omit", "duplicate", "byte_reproduced"])
def test_validator_cannot_hide_or_duplicate_rotation_changes(transport_inputs, change):
    report = build(transport_inputs)
    evidence = report["cases"][0]["qasm"]
    if change == "omit":
        evidence["rotation_changes"].clear()
    elif change == "duplicate":
        evidence["rotation_changes"].append(copy.deepcopy(evidence["rotation_changes"][0]))
    else:
        evidence["byte_reproduced"] = True
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field", ["canonical_cases", "float_changes", "qasm_hash_changes", "changed_rotation_lines"])
def test_validator_rederives_summary_counts(transport_inputs, field):
    report = build(transport_inputs)
    report["summary"][field] += 1
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


@pytest.mark.parametrize("field,value", [("changed_qasm_files", []), ("float_changes_by_section", {})])
def test_validator_rederives_summary_membership_and_sections(transport_inputs, field, value):
    report = build(transport_inputs)
    report["summary"][field] = value
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


def test_validator_rejects_changed_expected_sources(transport_inputs):
    report = build(transport_inputs)
    transport_inputs["sources"][next(iter(transport_inputs["sources"]))] = "b" * 64
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


def test_validator_binds_the_entire_frozen_contract(transport_inputs):
    report = build(transport_inputs)
    contract = transport_inputs["canonical"][0]["postselection_contract"]
    contract["synthesis_model_error_per_attempt"] = math.nextafter(contract["synthesis_model_error_per_attempt"], math.inf)
    with pytest.raises(ValueError):
        validate(report, transport_inputs)


def canonical_normalizer_fixture():
    sources = {"synthetic-source.py": "a" * 64}
    environment = {"python": "3.12.14", "device": "CPU", "seed": 20261112,
                   "packages": {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4",
                                "scipy": "1.18.1", "matplotlib": "3.11.2"}}
    case = {
        "case_id": CASE_ID, "method": "QSP_oblivious_amplification",
        "provenance": {
            "source_commit": t.NATIVE_PRODUCER_COMMIT,
            "source_revision_scope": "Exact checkout root when Git is available; null for standalone packages. Content hashes identify actual source.",
            "source_sha256": copy.deepcopy(sources), "environment": copy.deepcopy(environment),
        },
        "physical_reports": ["frozen-p-1e-3.json", "frozen-p-1e-4.json"],
        "cpu_timings_seconds": {"construction_validation_lowering": 1.25},
        "numerical_checks": {"actual_input_acceptance_probability": 0.0625},
        "postselection_contract": {"maximum_complete_attempts": 100, "synthesis_model_error_per_attempt": 0.000001},
        "qasm_sha256": "c" * 64, "unrelated_scientific_field": {"keep": 1.234},
    }
    return case, sources, environment


def test_canonical_normalization_preserves_every_scientific_field():
    case, sources, environment = canonical_normalizer_fixture()
    before = copy.deepcopy(case)
    normalized = t.normalize_canonical_case(case, sources, environment)
    expected = copy.deepcopy(case)
    expected.pop("physical_reports")
    expected.pop("cpu_timings_seconds")
    expected["provenance"].pop("source_commit")
    expected["provenance"].pop("source_sha256")
    assert normalized == expected
    assert case == before


def test_product_formula_normalization_retains_both_original_timing_keys():
    case, sources, environment = canonical_normalizer_fixture()
    case["case_id"] = "pf1-n2-t1-eps0.05"
    case["method"] = "product_formula"
    case["cpu_timings_seconds"] = {"construction_and_lowering": 0,
                                   "independent_dense_validation": 1.25}
    normalized = t.normalize_canonical_case(case, sources, environment)
    assert normalized["method"] == "product_formula"
    assert "cpu_timings_seconds" not in normalized
    assert normalized["postselection_contract"] == case["postselection_contract"]


def test_product_formula_normalization_rejects_a_replaced_timing_ledger():
    case, sources, environment = canonical_normalizer_fixture()
    case["case_id"] = "pf1-n2-t1-eps0.05"
    case["method"] = "product_formula"
    with pytest.raises(ValueError):
        t.normalize_canonical_case(case, sources, environment)


@pytest.mark.parametrize("commit", [None, "1" * 40])
def test_canonical_normalization_retains_the_original_valid_commit_domain(commit):
    case, sources, environment = canonical_normalizer_fixture()
    case["provenance"]["source_commit"] = commit
    normalized = t.normalize_canonical_case(case, sources, environment)
    assert "source_commit" not in normalized["provenance"]
    assert normalized["numerical_checks"] == case["numerical_checks"]


@pytest.mark.parametrize("timing", [-1.0, math.inf, math.nan, None, "1.25"])
def test_canonical_normalization_rejects_invalid_timings(timing):
    case, sources, environment = canonical_normalizer_fixture()
    case["cpu_timings_seconds"]["construction_validation_lowering"] = timing
    with pytest.raises(ValueError):
        t.normalize_canonical_case(case, sources, environment)


@pytest.mark.parametrize("change", ["source", "environment", "scope", "Git", "timing", "physical", "extra_provenance"])
def test_canonical_normalization_retains_original_metadata_guards(change):
    case, sources, environment = canonical_normalizer_fixture()
    if change == "source":
        case["provenance"]["source_sha256"]["synthetic-source.py"] = "b" * 64
    elif change == "environment":
        case["provenance"]["environment"]["seed"] = True
    elif change == "scope":
        case["provenance"]["source_revision_scope"] = "borrowed Git"
    elif change == "Git":
        case["provenance"]["source_commit"] = "invalid"
    elif change == "timing":
        case["cpu_timings_seconds"]["construction_validation_lowering"] = True
    elif change == "physical":
        case["physical_reports"] = []
    else:
        case["provenance"]["equivalence_certificate"] = True
    with pytest.raises(ValueError):
        t.normalize_canonical_case(case, sources, environment)
