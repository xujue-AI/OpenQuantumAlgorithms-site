"""Saved-data and independent arithmetic tests; no scientific implementation runs."""
from __future__ import annotations

import ast
from copy import deepcopy
from decimal import Decimal
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("tail_resource_model_tests", ROOT / "scripts/tail_resource_model.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)
ERROR = model.ModelVerificationError


@pytest.fixture(scope="module")
def instrument():
    base = ROOT / "assets/generated/instrument-resources"
    reports = {task: json.loads((base / task / "report.json").read_text()) for task in model.INSTRUMENT_CASES}
    qasm = {task: (base / task / "clifford-t-instrument.qasm").read_text() for task in model.INSTRUMENT_CASES}
    return reports, qasm


@pytest.fixture(scope="module")
def complete():
    base = ROOT / "assets/generated/simulation/complete-resources"
    results = json.loads((base / "results.json").read_text())
    qasm = {task: (base / task / "clifford-t-instrument.qasm").read_text() for task in model.COMPLETE_CASES}
    return results, qasm


def _one_profile(instrument):
    reports, qasms = instrument
    return deepcopy(reports["damping-Z"]["physical_profiles"][0]), qasms["damping-Z"]


def _qasm(statements, n=1, bits=1):
    return f'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[{bits}] c;\nqubit[{n}] q;\n{statements}\n'


def _zero_t():
    # Hand-calculated d=3, n=1 schedule: init=13100, parity=39700 ns.
    # Three startup patches, one H, one measured readout and one trial dispatch:
    # runtime=99000 ns, 99 cycles, 51 physical data qubits.
    p = {
        "physical_error_rate": 0.0001, "physical_gate_time_ns": 100,
        "physical_measurement_time_ns": 500, "physical_initialization_time_ns": 100,
        "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
        "controller_failure_per_decision": 1e-15, "memory_prefactor": 0.1,
        "initialization_residual_prefactor": 0.1, "memory_threshold": 0.01,
        "factory_pipeline_failure_bound": 0.01, "factory_feedback_rounds_reserved": 3,
        "maximum_code_distance": 49,
    }
    path_error = (1 - (1 - 2*Fraction(1, 10000))**3)/2
    readout = float(3*path_error**2*(1 - path_error) + path_error**3)
    prep = 1e-5
    components = {
        "data_workspace_code_exposure": 0.00594,
        "factory_output_handoff_storage": 0.0,
        "logical_readout_scenario": 2*readout,
        "logical_initialization_scenario": 0.0001,
        "controller_scenario": 16e-15,
        "factory_output_quality_model": 0.0,
    }
    rows = [
        dict(operation="h", count=1, time_ns=10000, readouts=0, preparations=0, decisions=1, event_failure_model=1e-15),
        dict(operation="initial_reset", count=3, time_ns=13100, readouts=0, preparations=1, decisions=1, event_failure_model=prep + 1e-15),
        dict(operation="measure", count=1, time_ns=39700, readouts=1, preparations=2, decisions=3, event_failure_model=readout + 2*prep + 3e-15),
        dict(operation="trial_control", count=1, time_ns=10000, readouts=0, preparations=0, decisions=1, event_failure_model=1e-15),
    ]
    point = {
        "data_code_distance": 3, "reserved_logical_patches": 3,
        "peak_physical_qubits": 51, "data_workspace_physical_qubits": 51,
        "factory_physical_qubits": 0, "runtime_per_complete_attempt_ns": 99000,
        "worst_case_runtime_all_attempts_ns": 198000, "code_cycles_per_complete_attempt": 99,
        "full_physical_failure_model": sum(components.values()), "failure_components_all_attempts": components,
        "factory_supply": {
            "output_error": 0.0, "pipeline_time_ns": 0, "fixed_logical_distillation_distance": 3,
            "native_qec_crossing_prefactor": 0.03, "native_qec_threshold": 0.01,
            "native_code_cycle_time_ns": 1000, "full_pipeline_retry_cap_per_state": 0,
            "states_all_complete_attempts": 0, "worst_case_pipeline_attempts": 0,
            "availability_failure_union_bound": 0.0, "pipeline_feedback_rounds_reserved": 3,
            "extra_init_readout_feedback_dispatch_ns_per_pipeline": 41800,
            "extra_output_port_hold_ns_per_state": 0,
        },
        "macro_ledger_per_complete_attempt": rows, "accounting_trace_sha256": "0"*64,
    }
    return {
        "status": model.POSITIVE, "profile": p, "complete_algorithm_attempts": 2,
        "physical_failure_budget": 0.02, "factory_availability_budget": 0.0025,
        "limits": {"max_qubits": 5, "synthesized_operations": 100000},
        "synthesized_event_counts": {"reset": 1, "h": 1, "measure": 1},
        "workspace_initializations_per_attempt": 2, "pareto_points": [point], "feasible_candidate_count": 1,
    }, _qasm("reset q[0];\nh q[0];\nc[0] = measure q[0];")


def _negative(controller=0.0025, physical_budget=0.02):
    positive, qasm = _zero_t()
    profile = positive["profile"]
    profile["controller_failure_per_decision"] = controller
    floor = 16*controller
    return {
        "status": model.CONTROLLER_NEGATIVE, "profile": profile,
        "complete_algorithm_attempts": 2, "pareto_points": [], "all_feasible_points": [],
        "controller_model_diagnostic": {
            "controller_decisions_per_attempt": 8, "controller_decisions_all_attempts": 16,
            "controller_failure_per_decision_scenario": controller,
            "distance_independent_controller_model_contribution": floor,
            "physical_failure_budget": physical_budget, "budget_exceeded": floor > physical_budget,
            "factory_retry_cap": 0,
        },
    }, qasm


def test_canonical_instrument_scope_and_no_mutation(instrument):
    reports, qasm = instrument
    before = deepcopy(reports)
    outcome = model.verify_instrument_models(reports, qasm)
    assert outcome["saved_points_checked"] == 23
    assert outcome["feasible_candidates_attested"] == 859
    assert outcome["point_scope"] == "retained"
    assert len(outcome["profiles"]) == 8
    assert reports == before


def test_canonical_complete_frontiers_negatives_and_joins(complete):
    results, qasm = complete
    before = json.dumps(results, sort_keys=True)
    outcome = model.verify_complete_models(results, qasm)
    assert outcome["saved_points_checked"] == 1168
    assert outcome["positive_scenarios_checked"] == 14
    assert outcome["controller_negatives_checked"] == 6
    assert [(e["common_count"], e["pareto_count"]) for e in outcome["energy"]] == [(0, 0), (0, 0), (35, 3), (112, 3)]
    assert outcome["max_absolute_float_residual"] <= 1e-18
    assert json.dumps(results, sort_keys=True) == before


def test_module_imports_are_stdlib_and_verification_has_no_io(monkeypatch):
    tree = ast.parse((ROOT / "scripts/tail_resource_model.py").read_text())
    allowed = {"__future__", "collections", "decimal", "fractions", "math", "re"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert {entry.name for entry in node.names} <= allowed
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed
    scenario, qasm = _zero_t()
    def forbidden(*args, **kwargs):
        raise AssertionError("model attempted file I/O")
    monkeypatch.setattr("builtins.open", forbidden)
    assert model.verify_profile_model(scenario, qasm)["saved_points_checked"] == 1


@pytest.mark.parametrize("states,failure,budget,want", [
    (0, 0.01, 0.0025, 0),
    (1, 0.01, 0.0001, 2),
    (1, Decimal("0.01"), Decimal("0.00009999999999999999999"), 3),
    (1, Decimal("0.01"), Decimal("1e-32"), 16),
    (1277530, 0.01, 0.0025, 5),
])
def test_exact_factory_retry_thresholds(states, failure, budget, want):
    assert model.factory_retry_cap(states, failure, budget) == want


@pytest.mark.parametrize("states,failure,budget", [
    (True, 0.01, 0.0025), (1.0, 0.01, 0.0025), (-1, 0.01, 0.0025),
    (1, False, 0.0025), (1, math.nan, 0.0025), (1, math.inf, 0.0025),
    (1, 0.01, 0), (1, 0.01, True), (1, 0.01, Decimal("1e-33")),
    (10**400, 0.01, 0.0025),
])
def test_factory_retry_rejects_invalid_or_unsupported_values(states, failure, budget):
    with pytest.raises(ERROR):
        model.factory_retry_cap(states, failure, budget)


def test_reset_chronology_and_barriers():
    parsed = model.parse_compiled_qasm(_qasm("barrier q[0];\nreset q[0];\nh q[0];\nreset q[0];\nc[0] = measure q[0];"))
    assert parsed["event_counts"] == {"initial_reset": 3, "h": 1, "reset": 1, "measure": 1, "trial_control": 1}
    assert parsed["operation_counts"]["barrier"] == 1
    assert parsed["all_inputs_initialized_before_use"] is True


@pytest.mark.parametrize("first", ["h q[0];", "t q[0];", "c[0] = measure q[0];"])
def test_use_before_reset_does_not_fabricate_initialization(first):
    parsed = model.parse_compiled_qasm(_qasm(first + "\nreset q[0];\nc[0] = measure q[0];"))
    assert parsed["event_counts"]["initial_reset"] == 2
    assert parsed["event_counts"]["reset"] == 1
    assert parsed["all_inputs_initialized_before_use"] is False


@pytest.mark.parametrize("statement,n,bits", [
    ("cx q[0];", 1, 1), ("h q[0], q[1];", 2, 1), ("cx q[0], q[0];", 1, 1),
    ("cx q[0], q[2];", 3, 1), ("x q[1];", 1, 1), ("c[1] = measure q[0];", 1, 1),
    ("rz(0.1) q[0];", 1, 1), ("h q[0]; garbage", 1, 1),
    ("qubit[1] q;", 1, 1), ("barrier q[0], q[0];", 1, 1),
    ("barrier q[1];", 1, 1), ("reset q[0];", 9, 1), ("reset q[0];", 1, 0),
])
def test_qasm_rejects_out_of_model_statements(statement, n, bits):
    with pytest.raises(ERROR):
        model.parse_compiled_qasm(_qasm(statement, n, bits))


def test_independent_zero_t_hand_calculation():
    scenario, qasm = _zero_t()
    result = model.verify_profile_model(scenario, qasm)
    assert result["saved_points_checked"] == 1
    assert result["max_absolute_float_residual"] < 1e-17


@pytest.mark.parametrize("field,value", [
    ("data_code_distance", True), ("data_code_distance", 3.0), ("data_code_distance", 4),
    ("peak_physical_qubits", 51.0), ("runtime_per_complete_attempt_ns", 99001),
    ("worst_case_runtime_all_attempts_ns", 198001), ("code_cycles_per_complete_attempt", 100),
    ("factory_physical_qubits", -1), ("full_physical_failure_model", math.nan),
    ("accounting_trace_sha256", "0"*63),
])
def test_zero_t_integer_and_failure_mutations(field, value):
    scenario, qasm = _zero_t()
    scenario["pareto_points"][0][field] = value
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


@pytest.mark.parametrize("field,value", [
    ("output_error", 1e-20), ("pipeline_time_ns", 1), ("full_pipeline_retry_cap_per_state", 1),
    ("states_all_complete_attempts", 1), ("worst_case_pipeline_attempts", 1),
    ("availability_failure_union_bound", 1e-10), ("extra_output_port_hold_ns_per_state", 1),
    ("native_qec_crossing_prefactor", 0.03000000000001),
])
def test_zero_t_cannot_acquire_fabricated_factory(field, value):
    scenario, qasm = _zero_t()
    scenario["pareto_points"][0]["factory_supply"][field] = value
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


@pytest.mark.parametrize("field,value", [
    ("count", True), ("count", 2), ("time_ns", 10000.0),
    ("readouts", 1), ("preparations", 1), ("decisions", 2), ("event_failure_model", math.inf),
])
def test_each_macro_accounting_field_is_checked(field, value):
    scenario, qasm = _zero_t()
    scenario["pareto_points"][0]["macro_ledger_per_complete_attempt"][0][field] = value
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


@pytest.mark.parametrize("component", sorted(model._FAILURES))
def test_all_six_failure_components_are_rederived(component):
    scenario, qasm = _zero_t()
    scenario["pareto_points"][0]["failure_components_all_attempts"][component] += 1e-7
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


def test_retained_scope_does_not_claim_unsaved_candidate_reconstruction():
    scenario, qasm = _zero_t()
    scenario["feasible_candidate_count"] = 859
    result = model.verify_profile_model(scenario, qasm)
    assert result["saved_points_checked"] == 1
    assert result["feasible_candidates_attested"] == 859
    with pytest.raises(ERROR, match="all_feasible_points|controller"):
        model.verify_profile_model(scenario, qasm, point_scope="full")


def test_controller_negative_with_omitted_root_budgets_is_supported():
    scenario, qasm = _negative()
    result = model.verify_profile_model(scenario, qasm, point_scope="full", physical_budget=0.02, availability_budget=0.0025)
    assert result["status"] == model.CONTROLLER_NEGATIVE
    assert result["saved_points_checked"] == 0


@pytest.mark.parametrize("controller", [0.00125, 0.001, 0.0])
def test_controller_negative_requires_strict_excess(controller):
    scenario, qasm = _negative(controller)
    with pytest.raises(ERROR, match="strict"):
        model.verify_profile_model(scenario, qasm, point_scope="full", physical_budget=0.02, availability_budget=0.0025)


@pytest.mark.parametrize("field,value", [
    ("controller_decisions_per_attempt", 9), ("controller_decisions_all_attempts", 17),
    ("controller_failure_per_decision_scenario", 0.0026), ("physical_failure_budget", 0.021),
    ("budget_exceeded", 1), ("factory_retry_cap", 1),
    ("distance_independent_controller_model_contribution", 0.039),
])
def test_controller_diagnostic_cannot_change_saved_inputs(field, value):
    scenario, qasm = _negative()
    scenario["controller_model_diagnostic"][field] = value
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm, point_scope="full", physical_budget=0.02, availability_budget=0.0025)


@pytest.mark.parametrize("mutate", [
    lambda s: s["all_feasible_points"].append({}),
    lambda s: s["pareto_points"].append({}),
    lambda s: s.update(status="infeasible_declared_distance_factory_domain"),
])
def test_controller_negative_empty_lists_and_status_are_exact(mutate):
    scenario, qasm = _negative()
    mutate(scenario)
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm, point_scope="full", physical_budget=0.02, availability_budget=0.0025)


def test_t_wire_transport_and_successful_quality_are_rederived(instrument):
    scenario, qasm = _one_profile(instrument)
    match = re.search(r"t q\[(\d+)\];", qasm)
    wire = int(match[1])
    altered = qasm[:match.start()] + f"t q[{1-wire}];" + qasm[match.end():]
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, altered)
    point = scenario["pareto_points"][0]
    point["failure_components_all_attempts"]["factory_output_quality_model"] *= point["factory_supply"]["full_pipeline_retry_cap_per_state"]
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


def test_independent_float_tolerance_is_narrow_and_inputs_unchanged(instrument):
    scenario, qasm = _one_profile(instrument)
    field = "full_physical_failure_model"
    point = scenario["pareto_points"][0]
    original = point[field]
    point[field] *= 1 + 1e-12
    accepted = point[field]
    model.verify_profile_model(scenario, qasm)
    assert point[field] == accepted
    point[field] = original*(1 + 1e-8)
    with pytest.raises(ERROR):
        model.verify_profile_model(scenario, qasm)


@pytest.mark.parametrize("target", ["workload", "profile", "counts", "case_set", "energy_count", "energy_join", "energy_status"])
def test_complete_crosslinks_and_join_mutations(complete, target):
    original, qasm = complete
    data = deepcopy(original)
    case = data["cases"]["dynamics-n1-Z"]
    if target == "workload":
        case["prepared"]["complete_workload"]["complete_attempts"] += 1
    elif target == "profile":
        for value in data["profiles"].values():
            value["reaction_time_ns"] += 1
    elif target == "counts":
        case["prepared"]["complete_workload"]["full_clifford_t_counts_all_attempts"]["t"] += 1
    elif target == "case_set":
        del data["cases"]["dynamics-n2-Z"]
    elif target == "energy_count":
        data["energy"]["p1e-03-controller1e-18"]["common_feasible_candidate_count"] += 1
    elif target == "energy_join":
        data["energy"]["p1e-03-controller1e-18"]["pareto_points"][0]["group_points"]["X"]["factory_supply"]["full_pipeline_retry_cap_per_state"] += 1
    else:
        data["energy"]["p1e-03-controller1e-15"]["status"] = "computed_common_configuration_energy_model"
    with pytest.raises(ERROR):
        model.verify_complete_models(data, qasm)


def test_full_frontier_is_checked_against_every_saved_candidate(complete):
    data, qasm = complete
    scenario = deepcopy(data["cases"]["dynamics-n1-Z"]["scenarios"]["p1e-03-controller1e-18"])
    scenario["pareto_points"] = scenario["pareto_points"][1:]
    with pytest.raises(ERROR, match="frontier"):
        model.verify_profile_model(scenario, qasm["dynamics-n1-Z"], point_scope="full")


def test_duplicate_common_configuration_cannot_hide_a_ledger(complete):
    original, qasm = complete
    data = deepcopy(original)
    scenario = data["cases"]["thermal-n2-X"]["scenarios"]["p1e-03-controller1e-18"]
    scenario["all_feasible_points"].append(deepcopy(scenario["all_feasible_points"][0]))
    scenario["feasible_candidate_count"] += 1
    with pytest.raises(ERROR, match="duplicate"):
        model.verify_complete_models(data, qasm)


def test_duplicate_full_configuration_rejected_before_energy_join(complete):
    data, qasm = complete
    scenario = deepcopy(data["cases"]["dynamics-n1-Z"]["scenarios"]["p1e-03-controller1e-18"])
    scenario["all_feasible_points"].append(deepcopy(scenario["all_feasible_points"][0]))
    scenario["feasible_candidate_count"] += 1
    with pytest.raises(ERROR, match="duplicate"):
        model.verify_profile_model(scenario, qasm["dynamics-n1-Z"], point_scope="full")


@pytest.mark.parametrize("component", ["availability", "factory_output_handoff_storage", "factory_output_quality_model"])
def test_zero_t_exact_zeros_cannot_use_float_tolerance(component):
    scenario, qasm = _zero_t()
    point = scenario["pareto_points"][0]
    if component == "availability":
        point["factory_supply"]["availability_failure_union_bound"] = 1e-32
    else:
        point["failure_components_all_attempts"][component] = 1e-32
    with pytest.raises(ERROR, match="zero_T"):
        model.verify_profile_model(scenario, qasm)


@pytest.mark.parametrize("budget", ["physical_failure_budget", "factory_availability_budget"])
def test_positive_scenario_must_carry_its_own_budget(instrument, budget):
    scenario, qasm = _one_profile(instrument)
    physical = scenario["physical_failure_budget"]
    availability = scenario["factory_availability_budget"]
    del scenario[budget]
    with pytest.raises(ERROR, match="missing"):
        model.verify_profile_model(scenario, qasm, physical_budget=physical, availability_budget=availability)


def test_model_feasibility_cannot_borrow_native_float_tolerance():
    scenario, qasm = _zero_t()
    point = scenario["pareto_points"][0]
    scenario["physical_failure_budget"] = point["full_physical_failure_model"]*(1 - 1e-12)
    point["full_physical_failure_model"] = scenario["physical_failure_budget"]
    with pytest.raises(ERROR, match="independently recomputed physical budget"):
        model.verify_profile_model(scenario, qasm)
