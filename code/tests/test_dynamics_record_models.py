"""Saved dynamics arithmetic checks; no scientific imports or computation."""
from __future__ import annotations

import builtins
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "assets/generated/dynamics-resources"
spec = importlib.util.spec_from_file_location("dynamics_saved_model", ROOT / "scripts/dynamics_physical_records.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
PF = "pf2-n3-t1-eps0.05"
QSP = "qsp-oaa0-n3-t1-eps0.05"
RETARGETED = "qsp-retargeted-n4-t1-eps0.05"
PROFILE = "frozen-p-1e-4.json"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def snapshot(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file() and p.name != "resource-record-manifest.json"}


@pytest.fixture
def saved(tmp_path):
    target = tmp_path / "records"
    shutil.copytree(RECORDS, target)
    return target


def validate(root):
    return model.validate_physical_records(root, read(root / "manifest.json"))


def change_case(root, name, operation):
    study = read(root / "manifest.json")
    case = next(value for value in study["cases"] if value["case_id"] == name)
    operation(case)
    write(root / name / "logical-resources.json", case)
    write(root / "manifest.json", study)


def change_report(root, name, operation):
    path = root / name / PROFILE
    report = read(path)
    operation(report)
    write(path, report)


def test_complete_real_saved_study_is_checked_without_scientific_imports(monkeypatch):
    before = snapshot(RECORDS)
    imported = []
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        imported.append(name)
        if name.split(".")[0] in {"qdk", "qiskit", "numpy", "scipy", "matplotlib"}:
            raise AssertionError("Saved model attempted scientific import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = validate(RECORDS)
    assert result["status"] == "validated_frozen_conditional_dynamics_records"
    assert [result[name] for name in ("cases", "reports", "pareto_points", "csv_rows")] == [24, 48, 489, 144]
    assert result["native_estimates_run"] is False
    assert result["distance_domain"] == list(range(3, 42, 2))
    assert "trusted" in " ".join(result["limitations"])
    assert snapshot(RECORDS) == before and len(before) == 105
    result["profiles"][0]["physical_error_rate"] = 0
    assert model.PROFILES[0]["physical_error_rate"] == .001


def test_authorized_additive_sidecar_does_not_replace_any_original_record(saved):
    before = snapshot(saved)
    write(saved / "resource-record-manifest.json", {"owned_by_external_identity_gate": True})
    assert validate(saved)["native_estimates_run"] is False
    assert snapshot(saved) == before


@pytest.mark.parametrize("mutation", ["case_count", "query_count", "profile", "domain", "certification", "duplicate_case"])
def test_changed_top_level_frozen_study_is_rejected(saved, mutation):
    study = read(saved / "manifest.json")
    if mutation == "case_count": study["canonical_cases"] = 23
    elif mutation == "query_count": study["actual_physical_queries"] = 47
    elif mutation == "profile": study["profiles"][0]["physical_error_rate"] = .0005
    elif mutation == "domain": study["common_data_and_factory_distance_domain"].pop()
    elif mutation == "certification": study["certified_confidence_lower_bound"] = .99
    else: study["cases"][-1] = copy.deepcopy(study["cases"][0])
    write(saved / "manifest.json", study)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["steps", "source_count", "source_depth", "rational_bound", "rounding",
                                     "algorithm_bias", "sampling", "readout_count", "certificate"])
def test_product_formula_contract_cannot_be_mutated_consistently_in_both_ledgers(saved, mutation):
    def operation(case):
        if mutation == "steps": case["steps"] += 1
        elif mutation == "source_count": case["source_gate_counts"]["rz"] += 1
        elif mutation == "source_depth": case["source_depth"] += 1
        elif mutation == "rational_bound": case["analytic_error"]["product_formula_operator_bound_exact"] = "1/100"
        elif mutation == "rounding": case["analytic_error"]["source_angle_rounding_operator_bound_exact"] = "0"
        elif mutation == "algorithm_bias": case["task"]["algorithm_bias"] += .001
        elif mutation == "sampling": case["accepted_slots"] -= 1
        elif mutation == "readout_count": case["compiled_gate_counts"]["measure"] -= 1
        else: case["numerical_checks"]["certified"] = True
    change_case(saved, PF, operation)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["guarded_p", "synthesis_allowance", "cap", "executions", "access",
                                     "phase_word", "phase_type", "phase_tau", "lowering_guard", "phase_certificate"])
def test_qsp_frozen_phase_access_acceptance_and_retry_parameters_remain_exact(saved, mutation):
    name = RETARGETED if mutation.startswith("phase_") else QSP
    def operation(case):
        if mutation == "guarded_p":
            contract = case["postselection_contract"]
            contract["reference_acceptance_lower"] = math.nextafter(contract["reference_acceptance_lower"], math.inf)
        elif mutation == "synthesis_allowance": case["postselection_contract"]["synthesis_model_error_per_attempt"] *= 2
        elif mutation == "cap": case["attempts_per_accepted_slot"] -= 1
        elif mutation == "executions": case["complete_attempts"] -= 1
        elif mutation == "access": case["access"]["inverse_A_calls"] += 1
        elif mutation == "phase_word":
            words = case["phase_evidence"]["cosine_phases"]
            words[0] = math.nextafter(words[0], math.inf)
        elif mutation == "phase_type": case["phase_evidence"]["cosine"]["degree"] = 14.0
        elif mutation == "phase_tau": case["phase_evidence"]["sine"]["tau"] = 4.1
        elif mutation == "lowering_guard": case["numerical_checks"]["acceptance_guard_subtracted"] *= 2
        else: case["phase_evidence"]["certified"] = True
    change_case(saved, name, operation)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["budget", "profile", "factory_domain", "raw_trace", "synthesized_trace",
                                     "synthesis_ts", "runtime", "serial", "distance", "data_space", "peak_space",
                                     "factory_copies", "factory_states", "factory_space", "factory_error",
                                     "physical_subtraction", "physical_budget", "spacetime", "properties_type",
                                     "certificate", "embedded_qasm", "missing_point"])
def test_every_recorded_model_layer_rejects_tampering(saved, mutation):
    def operation(report):
        point = report["pareto_points"][0]
        factory = point["factories"]["T"]
        if mutation == "budget": report["budget"]["executions"] -= 1
        elif mutation == "profile": report["physical_profile"]["gate_time_ns"] *= 2
        elif mutation == "factory_domain": report["factory_surface_code_distance_domain"].pop()
        elif mutation == "raw_trace": report["canonical_application_trace"] = report["canonical_application_trace"].replace("RZ(0)", "RZ(9)", 1)
        elif mutation == "synthesized_trace": report["synthesized_application_trace"] = report["synthesized_application_trace"].replace("@resource_state(T,", "@resource_state(CCZ,", 1)
        elif mutation == "synthesis_ts": report["synthesis"]["fixed_num_ts_per_rotation"] += 1
        elif mutation == "runtime": point["runtime_per_execution_ns"] += 1
        elif mutation == "serial": point["serial_runtime_all_executions_ns"] += 1
        elif mutation == "distance": point["data_code_distance"] = 2
        elif mutation == "data_space": point["model_properties"]["PHYSICAL_COMPUTE_QUBITS"] += 1
        elif mutation == "peak_space": point["peak_physical_qubits"] += 1
        elif mutation == "factory_copies": factory["copies"] = 1
        elif mutation == "factory_states": factory["states"] += 1
        elif mutation == "factory_space": point["model_properties"]["PHYSICAL_FACTORY_QUBITS"] += 1
        elif mutation == "factory_error": factory["error_rate"] *= 2
        elif mutation == "physical_subtraction":
            point["physical_failure_model_per_execution"] = math.nextafter(point["physical_failure_model_per_execution"], math.inf)
        elif mutation == "physical_budget": point["physical_failure_union_bound_all_executions"] = .0051
        elif mutation == "spacetime": point["peak_times_capped_runtime_qubit_seconds"] += 1
        elif mutation == "properties_type": point["model_properties"]["ALGORITHM_COMPUTE_QUBITS"] = 3.0
        elif mutation == "certificate": report["synthesis"]["actual_synthesis_sequence_generated"] = True
        elif mutation == "embedded_qasm": report["qasm3"] += "\n"
        else: report["pareto_points"].pop()
    change_report(saved, PF, operation)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["measurement", "reset", "angle", "missing_barrier", "linked", "missing", "extra"])
def test_qasm_and_complete_original_file_boundary_are_checked(saved, mutation):
    path = saved / PF / "measured-trial.qasm"
    if mutation == "linked":
        path.unlink()
        path.symlink_to(RECORDS / PF / "measured-trial.qasm")
    elif mutation == "missing": path.unlink()
    elif mutation == "extra": (saved / "raw-private-log.txt").write_text("not a publication artifact")
    else:
        text = path.read_text()
        if mutation == "measurement": text = text.replace("meas[0] = measure q[0]", "meas[0] = measure q[1]")
        elif mutation == "reset": text = text.replace("cx q[0], q[1];", "reset q[0];", 1)
        elif mutation == "angle": text = text.replace("rx(", "rx(__import__('os')+", 1)
        else: text = "\n".join(line for line in text.splitlines() if not line.startswith("barrier ")) + "\n"
        path.write_text(text)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["row", "choice", "header", "duplicate"])
def test_all_144_csv_choices_are_reconstructed(saved, mutation):
    path = saved / "physical-summary.csv"
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if mutation == "row": rows[0]["peak_physical_qubits"] = str(int(rows[0]["peak_physical_qubits"]) + 1)
    elif mutation == "choice": rows[0]["choice"] = "globally_optimal"
    elif mutation == "header": fields = fields[:-1]
    else: rows.append(copy.deepcopy(rows[0]))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "1e999"])
def test_json_nonfinite_and_overflow_are_rejected(saved, token):
    path = saved / PF / PROFILE
    path.write_text(path.read_text().replace('"projection_wall_seconds":', '"projection_wall_seconds": ' + token + ', "unused":', 1))
    with pytest.raises(ValueError):
        validate(saved)


def test_json_duplicate_keys_are_rejected(saved):
    path = saved / PF / PROFILE
    path.write_text(path.read_text().replace('"schema_version": 1,', '"schema_version": 1, "schema_version": 1,', 1))
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("mutation", ["assumptions", "proof", "exclusions", "missing_certificate",
                                     "phase_scope", "phase_fit_claim"])
def test_conditional_claim_inputs_are_not_optional_or_replaceable(saved, mutation):
    name = RETARGETED if mutation.startswith("phase_") else QSP
    def operation(case):
        if mutation == "phase_scope": case["phase_evidence"]["cosine"]["verification_scope"] = "certified interval proof"
        elif mutation == "phase_fit_claim": case["phase_evidence"]["phase_fit_method"] = "certified optimal synthesis"
        elif mutation == "assumptions": case["postselection_contract"]["assumptions"] = []
        elif mutation == "proof": case["postselection_contract"]["proof"] = "hardware output confidence established"
        elif mutation == "exclusions": case["postselection_contract"]["does_not_certify"] = []
        else: case["postselection_contract"].pop("certified_confidence_lower_bound")
    change_case(saved, name, operation)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("field", ["model", "cost_semantics", "fit_domain", "unknown_field"])
def test_synthesis_mean_fit_disclosures_cannot_be_replaced(saved, field):
    def operation(report):
        report["synthesis"][field] = "certified per-angle worst-case bound"
    change_report(saved, PF, operation)
    with pytest.raises(ValueError):
        validate(saved)


@pytest.mark.parametrize("reader", ["json", "qasm", "hash"])
def test_actual_read_cap_rejects_a_file_that_grows_after_stat(tmp_path, monkeypatch, reader):
    path = tmp_path / "oversized.json"
    path.write_bytes(b" " * (model.MAX_BYTES + 1))
    original = Path.stat
    def lying_stat(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self == path:
            fields = list(result)
            fields[6] = 1
            return os.stat_result(fields)
        return result
    monkeypatch.setattr(Path, "stat", lying_stat)
    with pytest.raises(ValueError, match="actual record read"):
        if reader == "json": model._read_json(path)
        elif reader == "qasm": model._qasm(path, 3)
        else: model._sha(path)


def test_actual_aggregate_read_cap_is_not_only_a_stat_sum(saved, monkeypatch):
    original = Path.stat
    def lying_stat(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if self.is_relative_to(saved):
            fields = list(result)
            fields[6] = 0
            return os.stat_result(fields)
        return result
    monkeypatch.setattr(Path, "stat", lying_stat)
    monkeypatch.setattr(model, "MAX_TOTAL_BYTES", 1024)
    with pytest.raises(ValueError, match="actual aggregate record read"):
        validate(saved)


@pytest.mark.parametrize("kind", ["depth", "nodes"])
def test_json_structure_work_is_bounded(tmp_path, kind):
    path = tmp_path / "bounded.json"
    value = [0] * 200001
    if kind == "depth":
        value = 0
        for _ in range(45):
            value = [value]
    write(path, value)
    with pytest.raises(ValueError, match="traversal bound"):
        model._read_json(path)
