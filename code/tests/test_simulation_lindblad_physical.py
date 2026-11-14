"""Fixed literal Lindblad resource slice: source evidence and separate budgets."""
from dataclasses import asdict, replace
from fractions import Fraction
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

from openquantum import instrument_resources as ir
from openquantum.simulation import lindblad_physical_workload as lp, lindblad_workload as lw


@pytest.fixture(scope="module")
def source():
    return lp.build_source()


def test_fixed_source_retains_all_components_input_resets_and_readout(source):
    assert asdict(lp.source_task()) == {"n_system": 2, "time": 1., "J": .6,
                                        "hx": .35, "rate0": .8, "other_rate": .5}
    assert lp.STEPS == 69 and lp.COMPLETE_TRIALS == 15756
    assert source.num_qubits == source.num_clbits == 4
    assert len(source.data) == 1320
    assert dict(source.count_ops()) == {"cx": 483, "rx": 276, "reset": 209,
                                        "cry": 207, "rz": 138, "measure": 4,
                                        "h": 2, "s": 1}
    assert [source.find_bit(item.qubits[0]).index for item in source.data[:2]] == [0, 1]
    assert source.metadata["environment_initializations"] == [138, 69]
    assert source.metadata["environment_discard_resets_before_reuse"] == [137, 68]
    assert [(source.find_bit(item.qubits[0]).index, source.find_bit(item.clbits[0]).index)
            for item in source.data if item.operation.name == "measure"] == [(0,0),(1,1),(2,2),(3,3)]
    assert source.metadata["postselection"] == "none"


@pytest.mark.parametrize("task", [lw.ChainTask(3), lw.ChainTask(4), lw.ChainTask(2, time=.5),
                                  lw.ChainTask(2, J=.5), lw.ChainTask(2, hx=.4),
                                  lw.ChainTask(2, rate0=.5), "n2", False])
def test_nonfixed_physical_tasks_reject(task):
    with pytest.raises(ValueError, match="Only literal"):
        lp.build_source(task)


def test_existing_density_source_matches_two_independent_classical_answers(source):
    checked = ir.simulate_trial(source, 0, {}, limits=lp.limits())
    task = lp.source_task()
    z0 = lw.on_wire(lw.Z, 0, 2)
    gate_mean = float(np.trace(z0@lw.gate_density(task, 69)).real)
    exact_mean = float(np.trace(z0@lw.reference_density(task)).real)
    assert checked["acceptance_probability"] == pytest.approx(1., abs=1e-12)
    assert checked["mean"] == pytest.approx(gate_mean, abs=1e-13)
    assert exact_mean == pytest.approx(.7911337403681944, abs=1e-13)
    assert checked["mean"] == pytest.approx(.791140004653029, abs=1e-13)
    assert abs(checked["mean"]-exact_mean) <= float(lw.uniform_split_bound(task, 69))
    assert lp.limits().max_qubits == 4
    assert lp.limits().source_operations == 20_000
    assert lp.limits().synthesized_operations == lp.limits().validation_operations == 500_000


def test_all_family_allocations_remain_exact_and_unused_cases_are_not_reallocated():
    family = lp.family_allocations()
    def value(record):
        return Fraction(record["numerator"], record["denominator"])
    assert family["implemented_physical_cases"] == [2]
    assert family["unused_cases_reallocated"] is False
    assert family["family_physical_confidence_implemented"] is False
    assert value(family["statistics_family_failure_exact"]) == Fraction(1,200)
    assert value(family["physical_family_failure_exact"]) == value(family["factory_family_failure_exact"]) == Fraction(1,400)
    assert value(family["total_conditional_family_failure_exact"]) == Fraction(1,100)
    for n, case in family["case_allocations"].items():
        assert value(case["statistical_failure_exact"]) == Fraction(1,600)
        assert value(case["physical_failure_exact"]) == value(case["factory_availability_failure_exact"]) == Fraction(1,1200)
        assert Fraction.from_float(case["physical_failure_budget"]) <= Fraction(1,1200)
        assert Fraction.from_float(case["factory_availability_budget"]) <= Fraction(1,1200)
        assert case["complete_trials"] == 15756
        assert case["physical_status"] == ("implemented_n2_slice" if n == "2" else "unimplemented_reserved_allocation")


def test_guarded_actual_binary64_error_not_requested_allowance_drives_contract():
    actual = math.nextafter(9e-6, math.inf)
    contract = lp.error_contract(actual)
    eta = Fraction.from_float(actual)+Fraction("1e-7")+Fraction("1e-7")
    exact = contract["whole_realization_error_exact"]
    assert Fraction(exact["numerator"], exact["denominator"]) == eta
    assert contract["requested_synthesis_allowance"] == 5e-5
    assert contract["guarded_actual_synthesis_error"] == actual
    assert Fraction.from_float(contract["realization_Z0_bias_upper_proposal"]) >= 2*eta
    assert contract["realization_Z0_bias_upper_proposal"] < 2*5e-5
    assert contract["absolute_Z0_error_upper_proposal"] < .05
    assert contract["representation_norm_reserve_scenario"] == 1e-7
    assert contract["SDK_control_routing_norm_reserve_scenario"] == 1e-7
    assert contract["full_representation_certificate"] is False
    assert contract["formal_synthesis_certificate"] is False
    assert contract["coherent_error_is_stochastic_failure"] is False


@pytest.mark.parametrize("actual,kwargs", [(True, {}), (-1e-6, {}), (float("nan"), {}),
                                           (1e-4, {}), (1e-6, {"sdk_reserve": .01}),
                                           (1e-6, {"representation_reserve": .01}),
                                           (1e-6, {"synthesis_allowance": 1e-10})])
def test_invalid_error_premises_reject(actual, kwargs):
    with pytest.raises(ValueError):
        lp.error_contract(actual, **kwargs)


def test_exact_H_angle_and_input_discrepancies_do_not_claim_whole_representation_certificate(source):
    record = lp.representation_record(source)
    gap = record["parameters"]["J"]["binary_minus_decimal_exact"]
    assert Fraction(gap["numerator"], gap["denominator"]) == Fraction.from_float(.6)-Fraction(".6")
    assert record["representation_reserve_is_certified"] is False
    component = record["Hamiltonian_angle_component_exact"]
    exact = Fraction(component["numerator"], component["denominator"])
    assert Fraction.from_float(record["Hamiltonian_angle_coherent_error_component_upper"]) >= exact
    assert record["Hamiltonian_angle_coherent_error_component_upper"] < 1e-7
    assert sum(row["occurrences"] for row in record["stored_source_angle_targets"] if row["gate"] == "cry") == 207
    assert sum(row["occurrences"] for row in record["stored_source_angle_targets"] if row["gate"] == "rx") == 276


def _fake_synthesized(source, error=8e-6):
    return ir.SynthesizedInstrument(source.copy(), source.copy(), source.copy(),
        {"preserve_full_unitary": True, "transpiler_qubits_initially_zero": False,
         "rotation_occurrences": 828, "limits": asdict(lp.limits()),
         "guarded_numerical_operator_error_upper_estimate": error})


def test_prepare_adapter_plumbs_actual_full_unitary_synthesis_and_complete_counts(source, monkeypatch):
    checked = ir.simulate_trial(source, 0, {}, limits=lp.limits())
    calls = []
    def synth(circuit, allowance, *, limits, preserve_full_unitary):
        calls.append((circuit, allowance, limits, preserve_full_unitary))
        return _fake_synthesized(circuit)
    monkeypatch.setattr(ir, "synthesize_instrument", synth)
    monkeypatch.setattr(ir, "simulate_trial", lambda *args, **kwargs: dict(checked))
    prepared = lp.prepare_workload()
    assert calls[0][1:] == (5e-5, lp.limits(), True)
    assert calls[0][0] == source
    assert prepared.contract["guarded_actual_synthesis_error"] == 8e-6
    assert prepared.workload["complete_trials"] == prepared.workload["complete_attempts"] == 15756
    assert prepared.workload["algorithm_retry_cap"] == 1
    assert prepared.workload["source_resets_all_trials"] == 3293004
    assert prepared.workload["source_readouts_all_trials"] == 63024
    assert prepared.workload["workspace_startup_initializations_per_trial"] == 11
    assert prepared.workload["workspace_startup_initializations_all_trials"] == 173316
    assert prepared.workload["deterministic_bias_multiplied_by_trials"] is False
    assert prepared.source_check["density_path"].startswith("existing four-wire")
    assert prepared.routed_check["source_reduced_output_trace_norm_residual"] == 0
    record = prepared.as_record()
    assert record["case_id"] == "lindblad-n2-r69-Z0"
    assert record["hardware_execution"] is record["formal_confidence_certificate"] is False
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize("change,match", [("rotation", "828"), ("fullunitary", "full-unitary"),
                                         ("limits", "limits"), ("events", "209 resets"),
                                         ("source", "literal schedule")])
def test_prepared_compiler_and_literal_event_contracts_are_checked(source, change, match):
    synthesized = _fake_synthesized(source)
    if change == "rotation":
        synthesized.report["rotation_occurrences"] = 760
    elif change == "fullunitary":
        synthesized.report["preserve_full_unitary"] = False
    elif change == "limits":
        synthesized.report["limits"] = asdict(ir.InstrumentLimits(max_qubits=8))
    elif change == "events":
        synthesized.circuit.data = [item for item in synthesized.circuit.data if item.operation.name != "reset"]
    else:
        synthesized.source.z(1)
    with pytest.raises((AssertionError, ValueError), match=match):
        lp.check_synthesized(synthesized)


def test_routing_mean_and_reduced_output_change_is_detected(source, monkeypatch):
    ideal = {"acceptance_probability": 1., "mean": .791140004653029,
             "reduced_before_output": np.diag([.8955700023265145, .1044299976734855])}
    changed = {**ideal, "reduced_before_output": np.diag([.9, .1])}
    sequence = iter((dict(ideal), dict(changed), dict(ideal)))
    monkeypatch.setattr(ir, "simulate_trial", lambda *args, **kwargs: next(sequence))
    with pytest.raises(AssertionError, match="Routing changed"):
        lp.check_synthesized(_fake_synthesized(source))


def test_postselection_or_synthesis_score_violation_is_detected(source, monkeypatch):
    ideal = {"acceptance_probability": 1., "mean": .791140004653029,
             "reduced_before_output": np.diag([.8955700023265145, .1044299976734855])}
    sequence = iter((dict(ideal), dict(ideal), {**ideal, "acceptance_probability": .99}))
    monkeypatch.setattr(ir, "simulate_trial", lambda *args, **kwargs: next(sequence))
    with pytest.raises(AssertionError, match="No postselection"):
        lp.check_synthesized(_fake_synthesized(source))
    sequence = iter((dict(ideal), dict(ideal), {**ideal, "mean": .8}))
    with pytest.raises(AssertionError, match="realization proposal"):
        lp.check_synthesized(_fake_synthesized(source))


def _fake_physical_prepared():
    circuit = QuantumCircuit(4, 4)
    circuit.reset(range(4))
    circuit.t(0)
    circuit.tdg(3)
    circuit.h(1)
    circuit.cx(0,1)
    circuit.measure(range(4), range(4))
    return SimpleNamespace(synthesized=ir.SynthesizedInstrument(circuit, circuit, circuit, {}))


def test_controller_diagnostic_charges_workspace_dispatch_T_wire_and_factory_retry():
    prepared = _fake_physical_prepared()
    profile = lp.fixed_profiles()[0]
    result = lp.controller_diagnostic(prepared, profile)
    cap = ir._factory_retry_cap(2*15756, .01, math.nextafter(float(Fraction(1,1200)), -math.inf))
    # 15 startup +4*3 readout +1 H +11 CX +1 dispatch, with wire0/wire3 T macros.
    expected = 15+12+1+11+1+(47+4*cap)+(47+33*3+1+4*cap)
    assert result["controller_decisions_per_trial"] == expected
    assert result["controller_decisions_all_trials"] == 15756*expected
    assert result["factory_retry_cap"] == cap
    assert result["distance_independent_controller_model_contribution"] == pytest.approx(15756*expected*1e-15)
    assert "not a measured hardware" in result["scope"]


def test_estimator_preserves_case_budgets_profiles_and_native_arguments(monkeypatch, tmp_path):
    calls = []
    def estimate(synthesized, attempts, **kwargs):
        calls.append((synthesized, attempts, kwargs))
        return {"status": "computed_conditional_instrument_surface_code_model", "pareto_points": [{"test": True}]}
    monkeypatch.setattr(ir, "estimate_instrument", estimate)
    prepared = _fake_physical_prepared()
    report = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[1], cache_dir=tmp_path)
    assert calls[0][1] == 15756
    assert calls[0][2]["include_all_feasible_points"] is True
    assert Fraction.from_float(calls[0][2]["physical_failure_budget"]) <= Fraction(1,1200)
    assert Fraction.from_float(calls[0][2]["factory_availability_budget"]) <= Fraction(1,1200)
    assert report["family_allocations"]["unused_cases_reallocated"] is False
    assert report["controller_model_diagnostic"]["budget_exceeded"] is False
    assert report["profile"] == asdict(ir.InstrumentProfile(physical_error_rate=1e-4))


@pytest.mark.parametrize("profile", [ir.InstrumentProfile(physical_error_rate=1e-4, controller_failure_per_decision=1e-18),
                                     ir.InstrumentProfile(physical_error_rate=1e-4, maximum_code_distance=3),
                                     ir.InstrumentProfile(physical_error_rate=5e-4), None])
def test_nonfrozen_physical_profiles_reject(profile, tmp_path):
    with pytest.raises(ValueError, match="two frozen"):
        lp.estimate_prepared(_fake_physical_prepared(), profile=profile, cache_dir=tmp_path)


def test_infeasible_results_remain_explicit_and_keep_all_attempts(monkeypatch, tmp_path):
    monkeypatch.setattr(lp, "controller_diagnostic", lambda *args: {"budget_exceeded": True})
    report = lp.estimate_prepared(_fake_physical_prepared(), profile=lp.fixed_profiles()[0], cache_dir=tmp_path)
    assert report["status"] == "infeasible_declared_controller_model"
    assert report["complete_algorithm_attempts"] == 15756
    assert report["pareto_points"] == []
    monkeypatch.setattr(lp, "controller_diagnostic", lambda *args: {"budget_exceeded": False})
    def fail(*args, **kwargs):
        raise RuntimeError("No feasible point in the declared distance/factory/scenario domain")
    monkeypatch.setattr(ir, "estimate_instrument", fail)
    report = lp.estimate_prepared(_fake_physical_prepared(), profile=lp.fixed_profiles()[1], cache_dir=tmp_path)
    assert report["status"] == "infeasible_declared_distance_factory_domain"
    assert report["family_allocations"]["implemented_physical_cases"] == [2]


def test_unknown_model_errors_and_runner_caps_are_not_silently_reclassified(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("factory candidate cap exceeded")
    monkeypatch.setattr(ir, "estimate_instrument", fail)
    with pytest.raises(RuntimeError, match="candidate cap"):
        lp.estimate_prepared(_fake_physical_prepared(), profile=lp.fixed_profiles()[1], cache_dir=tmp_path)


def test_wider_family_rejections_are_reproduced_without_synthesis_or_estimates(monkeypatch):
    monkeypatch.setattr(ir, "synthesize_instrument", lambda *args, **kwargs: pytest.fail("No wider synthesis"))
    monkeypatch.setattr(ir, "estimate_instrument", lambda *args, **kwargs: pytest.fail("No wider physical estimate"))
    records = lp.family_status_records()
    for n, r, instructions, resets in ((3,134,4435,673),(4,211,9934,1481)):
        record = records[str(n)]
        assert record["selected_steps"] == r
        assert record["raw_source_instructions"] == instructions
        assert record["source_counts"]["reset"] == resets
        assert record["status"] == "ensemble_branch_cap_exceeded"
        assert record["rejection_reproduced"] is True
        assert record["synthesis_performed"] is record["physical_estimation_performed"] is False
        assert record["family_allocation_reallocated"] is False


def test_skip_diagnostics_is_explicit_and_predeclared_caps_remain_bounded():
    records = lp.family_status_records(check_rejections=False)
    assert records["3"]["status"] == records["4"]["status"] == "ensemble_rejection_not_rechecked"
    caps = lp.run_caps()
    assert caps["physical_cases"] == caps["prepared_synthesis_stages"] == 1
    assert caps["native_rotation_synthesis_attempts_upper_bound"] == 3312
    assert "actual gridsynth_unitary calls" in caps["native_rotation_synthesis_attempts_scope"]
    assert "top-level instrument adapter loop" in caps["query_scope"]
    assert "internal native factory construction" in caps["query_scope"]
    assert caps["physical_profiles"] == 2
    assert caps["factory_queries_per_profile"] == caps["surface_code_queries_per_profile"] == 24
    assert caps["prepare_seconds"] == 600 and caps["parent_seconds"] == 1200
