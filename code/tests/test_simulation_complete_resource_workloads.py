"""Complete fixed workloads: independent references and separated error budgets."""
from dataclasses import asdict, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from openquantum import hamiltonian_simulation as hs, instrument_resources as ir
from openquantum.simulation import complete_resource_workloads as cr


@pytest.fixture(scope="module")
def tasks():
    return {f"{kind}-n{n}-{axis}": cr.build_complete_task(kind, n, axis)
            for kind, n, axis in cr.default_task_specs()}


def _measurement_events(circuit):
    return [(circuit.find_bit(item.qubits[0]).index, circuit.find_bit(item.clbits[0]).index)
            for item in circuit.data if item.operation.name == "measure"]


def _short_ct(task):
    circuit = QuantumCircuit(task.source.num_qubits, task.source.num_qubits)
    circuit.reset(range(circuit.num_qubits))
    circuit.h(0)
    circuit.t(0)
    circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    return circuit


def _fake_prepared(task, attempts=100):
    circuit = _short_ct(task)
    synthesized = ir.SynthesizedInstrument(task.source, task.source, circuit, {})
    return SimpleNamespace(task=task, synthesized=synthesized,
                           workload={"complete_attempts": attempts})


def test_fixed_instances_scores_complete_reset_and_measurement_mapping(tasks):
    assert cr.default_task_specs() == (("dynamics", 1, "Z"), ("dynamics", 2, "Z"),
                                      ("thermal", 1, "X"), ("thermal", 2, "X"), ("thermal", 2, "Z"))
    for task in tasks.values():
        n, width = task.n_system, task.source.num_qubits
        assert width == ({1: 3, 2: 6}[n] if task.kind == "dynamics" else {1: 4, 2: 8}[n])
        assert [task.source.find_bit(item.qubits[0]).index for item in task.source.data[:width]] == list(range(width))
        assert all(item.operation.name == "reset" for item in task.source.data[:width])
        events = _measurement_events(task.source)
        if task.kind == "dynamics":
            assert events == [(q, q) for q in list(range(n, width))+list(range(n))]
            assert task.outcome_clbits == (0,)
            assert task.accept_clbits == dict.fromkeys(range(n, width), 0)
            assert task.score_values == (1., -1.)
            assert task.source.count_ops()["reset"] == width
            assert task.source.count_ops()["measure"] == width
            assert task.as_record()["instance"] == {"time": .5, "J": 1., "h": .7, "initial_state": "zero"}
        else:
            success = task.core.metadata["n_success_ancillas"]
            assert events == [(2*n+q, q) for q in range(success)]+[(q, success+q) for q in range(n)]
            assert task.outcome_clbits == tuple(range(success, success+n))
            assert task.accept_clbits == dict.fromkeys(range(success), 0)
            assert task.source.count_ops()["reset"] == width+n
            assert task.source.count_ops()["measure"] == success+n
            reference_resets = [task.source.find_bit(item.qubits[0]).index
                                for item in task.source.data[width:] if item.operation.name == "reset"]
            assert reference_resets == list(range(n, 2*n))
            assert task.as_record()["instance"] == {"beta": 1.5, "J": .6, "h": .35,
                                                    "input": "Bell", "reference_discard": "unconditioned"}
    assert tasks["thermal-n1-X"].score_values == (.35, -.35)
    assert tasks["thermal-n2-X"].score_values == (.7, 0., 0., -.7)
    assert tasks["thermal-n2-Z"].score_values == (.6, -.6, -.6, .6)
    assert tasks["thermal-n1-X"].terms == ((.35, "X"),)


@pytest.mark.parametrize("kind,n,axis", [("dynamics", 0, None), ("thermal", 3, None),
                                       ("dynamics", True, None), ("other", 1, None),
                                       ("dynamics", 1, "X"), ("thermal", 1, "Z"),
                                       ("thermal", 2, "Y")])
def test_unsupported_fixed_task_domains_reject(kind, n, axis):
    with pytest.raises(ValueError):
        cr.build_complete_task(kind, n, axis)


@pytest.mark.parametrize("n", [1, 2])
def test_data_first_and_success_first_full_joint_distribution_equivalence(tasks, n):
    task = tasks[f"dynamics-n{n}-Z"]
    check = cr.dynamics_measurement_order_check(task)
    joint = Statevector.from_instruction(task.core).probabilities()
    assert check["status"] == "numerically_equivalent"
    assert check["formal_certificate"] is False
    assert check["maximum_absolute_joint_probability_difference"] <= 1e-14
    np.testing.assert_allclose(check["original_data_first_joint_probabilities"], joint, atol=1e-14)
    np.testing.assert_allclose(check["resource_success_first_joint_probabilities"], joint, atol=1e-14)
    # Retain all data bits, rather than checking only the requested Z0 mean.
    checked = ir.simulate_scored_trial(task.source, tuple(range(n)),
                                      tuple(float(1-2*(state & 1)) for state in range(2**n)),
                                      task.accept_clbits, limits=cr.resource_limits())
    accepted = joint[:2**n]
    assert checked["acceptance_probability"] == pytest.approx(float(accepted.sum()), abs=2e-14)
    np.testing.assert_allclose(checked["joint_probabilities"], accepted/accepted.sum(), atol=2e-14)
    assert checked["mean"] == pytest.approx(task.exact_mean, abs=4e-7)


def test_full_joint_measurement_check_detects_changed_classical_mapping(tasks):
    task = tasks["dynamics-n1-Z"]
    changed = task.source.copy()
    measurements = [index for index, item in enumerate(changed.data) if item.operation.name == "measure"]
    left, right = measurements[:2]
    first, second = changed.data[left], changed.data[right]
    changed.data[left] = first.replace(clbits=second.clbits)
    changed.data[right] = second.replace(clbits=first.clbits)
    with pytest.raises(AssertionError, match="full joint"):
        cr.dynamics_measurement_order_check(replace(task, source=changed))


def test_dynamics_reference_matches_independent_closed_form(tasks):
    time, J, h = .5, 1., .7
    radius = math.sqrt(J*J+4*h*h)
    expected_two = math.cos(radius*time)*math.cos(J*time)+(J/radius)*math.sin(radius*time)*math.sin(J*time)
    assert tasks["dynamics-n1-Z"].exact_mean == pytest.approx(math.cos(2*h*time), abs=2e-15)
    assert tasks["dynamics-n2-Z"].exact_mean == pytest.approx(expected_two, abs=2e-15)


def test_thermal_reference_matches_independent_closed_form_and_jensen_norm(tasks):
    beta, J, h = 1.5, .6, .35
    radius = math.sqrt(J*J+4*h*h)
    denominator = math.cosh(beta*radius)+math.cosh(beta*J)
    x = -(4*h*h/radius)*math.sinh(beta*radius)/denominator
    z = -(J*J/radius*math.sinh(beta*radius)+J*math.sinh(beta*J))/denominator
    assert tasks["thermal-n1-X"].exact_mean == pytest.approx(-h*math.tanh(beta*h), abs=2e-15)
    assert tasks["thermal-n2-X"].exact_mean == pytest.approx(x, abs=2e-15)
    assert tasks["thermal-n2-Z"].exact_mean == pytest.approx(z, abs=2e-15)
    assert x+z == pytest.approx(-(radius*math.sinh(beta*radius)+J*math.sinh(beta*J))/denominator, abs=2e-15)
    for n in (1, 2):
        task = tasks[f"thermal-n{n}-X"]
        H = sum((coefficient*hs.pauli_matrix(pauli) for coefficient, pauli in task.terms),
                np.zeros((2**n, 2**n), complex))
        assert np.trace(H) == 0
        assert all(np.trace(hs.pauli_matrix(pauli)) == 0 for _, pauli in task.terms)
        a = Fraction(task.source_error["exact_binary_input_parameter"])
        with localcontext() as context:
            context.prec = 90
            decimal_a = Decimal(a.numerator)/Decimal(a.denominator)
            assert Decimal(task.target_amplitude_lower_decimal) <= (-decimal_a).exp()/4
            assert Decimal(task.generic_target_amplitude_lower_decimal) <= (-2*decimal_a).exp()/4
        assert Decimal(task.target_amplitude_lower_decimal) > Decimal(task.generic_target_amplitude_lower_decimal)
        partition = 2*math.cosh(beta*h) if n == 1 else 2*denominator
        assert partition/(2**n) >= 1
        ideal_branch_norm_squared = math.exp(-2*float(a))*partition/(16*2**n)
        assert ideal_branch_norm_squared >= float(task.target_amplitude_lower_decimal)**2
        actual_branch = Statevector.from_instruction(task.core).data[:2**(2*n)]
        assert float(np.vdot(actual_branch, actual_branch).real) == pytest.approx(ideal_branch_norm_squared, abs=2e-8)


@pytest.mark.parametrize("case", ["thermal-n1-X", "thermal-n2-X", "thermal-n2-Z"])
def test_actual_complete_thermal_source_scores_and_reference_discard(tasks, case):
    task = tasks[case]
    checked = ir.simulate_scored_trial(task.source, task.outcome_clbits, task.score_values,
                                      task.accept_clbits, limits=cr.resource_limits())
    contract = cr.conditioning_contract(task)
    assert checked["acceptance_probability"] >= contract["acceptance_lower_proposal"]
    assert abs(checked["mean"]-task.exact_mean) <= contract["source_score_bias_upper"]+2e-12
    assert np.trace(checked["reduced_before_output"]).real == pytest.approx(1, abs=2e-13)
    assert checked["simulation"]["peak_branches"] <= 2**task.n_system


def test_statistics_retry_and_all_group_budgets_are_outward_and_bounded(tasks):
    accepted_expected = {"dynamics-n1-Z": 14855, "dynamics-n2-Z": 14855,
                         "thermal-n1-X": 1820, "thermal-n2-X": 32135, "thermal-n2-Z": 23609}
    attempts_expected = {"dynamics-n1-Z": 3594910, "dynamics-n2-Z": 3594910,
                         "thermal-n1-X": 653380, "thermal-n2-X": 61474255, "thermal-n2-Z": 44337702}
    for case, task in tasks.items():
        count = cr.accepted_sample_count(task.score_values, task.statistical_absolute_error,
                                         task.statistical_failure_budget)
        assert count == accepted_expected[case]
        with localcontext() as context:
            context.prec = 90
            width = Fraction.from_float(max(task.score_values))-Fraction.from_float(min(task.score_values))
            w = Decimal(width.numerator)/Decimal(width.denominator)
            estimate = w*w*(2/Decimal(str(task.statistical_failure_budget))).ln()/(2*Decimal(str(task.statistical_absolute_error))**2)
            assert count >= estimate > count-1
        contract = cr.conditioning_contract(task)
        retries = cr.retry_cap(contract["acceptance_lower_proposal"], count, task.acceptance_failure_budget)
        assert retries["complete_attempts"] == attempts_expected[case]
        assert retries["complete_attempts"] == count*retries["retry_cap_per_accepted_slot"]
        assert Decimal(retries["no_acceptance_failure_upper_decimal_all_slots"]) <= Decimal(str(task.acceptance_failure_budget))
        assert Decimal(retries["previous_cap_failure_lower_decimal_all_slots"]) > Decimal(str(task.acceptance_failure_budget))
        assert retries["minimality_proved_for_selected_lower_probability"] is True
        assert contract["absolute_score_error_upper_proposal"] <= .05/(2 if case.startswith("thermal-n2") else 1)
    n2 = [tasks["thermal-n2-X"], tasks["thermal-n2-Z"]]
    assert sum(task.statistical_absolute_error for task in n2) == .03
    for name in ("statistical_failure_budget", "acceptance_failure_budget",
                 "physical_failure_budget", "factory_availability_budget"):
        assert sum(getattr(task, name) for task in n2) == .0025
    assert sum(getattr(task, name) for task in n2 for name in
               ("statistical_failure_budget", "acceptance_failure_budget", "physical_failure_budget", "factory_availability_budget")) == .01


@pytest.mark.parametrize("probability,slots,budget", [(.5, 7, .01), (.1, 3, .005), (1., 19, .1)])
def test_retry_cap_against_exact_fraction_reference(probability, slots, budget):
    cap = cr.retry_cap(probability, slots, budget)["retry_cap_per_accepted_slot"]
    q = 1-Fraction.from_float(probability)
    assert slots*q**cap <= Fraction(str(budget))
    assert slots*q**(cap-1) > Fraction(str(budget))


def test_actual_synthesis_contract_uses_exact_binary64_bound_and_separates_request(tasks):
    task = tasks["thermal-n2-X"]
    actual = math.nextafter(1e-5, math.inf)
    contract = cr.conditioning_contract(task, guarded_synthesis_error_upper_estimate=actual)
    eta = Fraction.from_float(actual)+Fraction("1e-7")
    s = Fraction(Decimal(task.target_amplitude_lower_decimal))
    epsilon = Fraction.from_float(task.source_error["precompiler_source_encoded_total_error_upper"])
    width = Fraction.from_float(max(task.score_values))-Fraction.from_float(min(task.score_values))
    assert contract["synthesis_numerical_allowance"] == 5e-5
    assert contract["guarded_numerical_operator_error_upper_estimate"] == actual
    assert Fraction(Decimal(contract["conditioning_synthesis_error_upper_decimal"])) >= Fraction.from_float(actual)
    assert Fraction(Decimal(contract["whole_realization_allowance_scenario_decimal"])) >= eta
    assert Fraction.from_float(contract["whole_realization_allowance_scenario"]) >= eta
    assert Fraction(Decimal(contract["acceptance_lower_proposal_decimal"])) <= (s-epsilon-eta)**2
    assert Fraction.from_float(contract["acceptance_lower_proposal"]) <= (s-epsilon-eta)**2
    assert Fraction.from_float(contract["realization_score_bias_upper_proposal"]) >= 2*width*eta/(s-epsilon)
    preflight = cr.conditioning_contract(task)
    assert contract["acceptance_lower_proposal"] > preflight["acceptance_lower_proposal"]
    assert contract["realization_score_bias_upper_proposal"] < preflight["realization_score_bias_upper_proposal"]
    zero = cr.conditioning_contract(task, guarded_synthesis_error_upper_estimate=0., realization_reserve=0.)
    assert zero["whole_realization_allowance_scenario"] == 0
    assert zero["realization_score_bias_upper_proposal"] == 0


def test_source_and_actual_realization_conditioning_is_monotone(tasks):
    task = tasks["dynamics-n1-Z"]
    contract = cr.conditioning_contract(task, guarded_synthesis_error_upper_estimate=1e-5)
    more_synthesis = cr.conditioning_contract(task, guarded_synthesis_error_upper_estimate=2e-5)
    source = {**task.source_error, "encoded_operator_error_upper": 2*task.source_error["encoded_operator_error_upper"]}
    more_source = cr.conditioning_contract(replace(task, source_error=source), guarded_synthesis_error_upper_estimate=1e-5)
    assert more_synthesis["acceptance_lower_proposal"] < contract["acceptance_lower_proposal"]
    assert more_synthesis["realization_score_bias_upper_proposal"] > contract["realization_score_bias_upper_proposal"]
    assert more_source["acceptance_lower_proposal"] < contract["acceptance_lower_proposal"]
    assert more_source["source_score_bias_upper"] > contract["source_score_bias_upper"]
    assert more_source["realization_score_bias_upper_proposal"] > contract["realization_score_bias_upper_proposal"]
    broken = replace(task, source_error={**task.source_error, "encoded_operator_error_upper": .3})
    with pytest.raises(ValueError, match="acceptance"):
        cr.conditioning_contract(broken)


@pytest.mark.parametrize("kwargs", [{"synthesis_allowance": 0.}, {"synthesis_allowance": .2},
                                    {"realization_reserve": .02}, {"guarded_synthesis_error_upper_estimate": -1.},
                                    {"guarded_synthesis_error_upper_estimate": 1e-4},
                                    {"guarded_synthesis_error_upper_estimate": float("nan")}])
def test_invalid_conditioning_domain_rejects(tasks, kwargs):
    with pytest.raises(ValueError):
        cr.conditioning_contract(tasks["dynamics-n1-Z"], **kwargs)


@pytest.mark.parametrize("values,error,failure", [([], .03, .01), ([float("nan")], .03, .01),
                                                ([1., -1.], 0., .01), ([1., -1.], .03, 1.),
                                                ([1., -1.], 1e-300, .01)])
def test_invalid_statistical_domain_rejects(values, error, failure):
    with pytest.raises(ValueError):
        cr.accepted_sample_count(values, error, failure)


@pytest.mark.parametrize("probability,slots,failure", [(0., 1, .01), (1.1, 1, .01),
                                                     (.1, True, .01), (.1, 0, .01),
                                                     (.1, 1, 1.), (1e-300, 1, .01)])
def test_invalid_or_unmaterializable_retry_domain_rejects(probability, slots, failure):
    with pytest.raises(ValueError):
        cr.retry_cap(probability, slots, failure)


def test_synthesis_pipeline_uses_actual_guarded_estimate_and_integer_workload(tasks):
    task = tasks["dynamics-n1-Z"]
    actual_error = 2e-7
    synthesized = ir.SynthesizedInstrument(task.source, task.source, _short_ct(task),
                                          {"guarded_numerical_operator_error_upper_estimate": actual_error,
                                           "operator_error_allowance": 5e-5})
    check = {"mean": task.exact_mean, "acceptance_probability": .0625,
             "joint_probabilities": np.array([.9, .1]),
             "reduced_before_output": np.array([[.9, .1j], [-.1j, .1]])}
    equivalence = {"status": "numerically_equivalent"}
    with patch.object(ir, "synthesize_instrument", return_value=synthesized) as synthesize, \
         patch.object(ir, "simulate_scored_trial", side_effect=[dict(check), dict(check), dict(check)]) as simulate, \
         patch.object(cr, "dynamics_measurement_order_check", return_value=equivalence):
        prepared = cr.synthesize_task(task)
    synthesize.assert_called_once_with(task.source, 5e-5, limits=cr.resource_limits(), preserve_full_unitary=True)
    assert simulate.call_count == 3
    assert prepared.contract["guarded_numerical_operator_error_upper_estimate"] == actual_error
    assert prepared.contract["whole_realization_allowance_scenario"] < 3.000000000001e-7
    assert prepared.contract["synthesis_numerical_allowance"] == 5e-5
    assert prepared.source_check["measurement_order_numerical_equivalence"] == equivalence
    assert prepared.workload["complete_attempts"] == prepared.workload["accepted_slots"]*prepared.workload["retry_cap_per_accepted_slot"]
    assert len(prepared.synthesized.circuit.data) == len(synthesized.circuit.data)
    for name, count in synthesized.circuit.count_ops().items():
        assert prepared.workload["full_clifford_t_counts_all_attempts"][name] == count*prepared.workload["complete_attempts"]
    record = prepared.as_record()
    json.dumps(record, allow_nan=False)
    assert record["source_instrument_check"]["reduced_before_output"]["imag"][0][1] == .1
    assert record["hardware_workload_executed"] is False
    assert record["complete_workload"]["whole_workload_executed"] is False
    assert record["numerical_realization_certificate"] is False
    assert record["formal_confidence_certificate"] is False


def test_routed_joint_distribution_change_rejects_even_when_mean_and_acceptance_agree(tasks):
    task = tasks["thermal-n2-X"]
    synthesized = ir.SynthesizedInstrument(task.source, task.source, _short_ct(task),
                                          {"guarded_numerical_operator_error_upper_estimate": 1e-7})
    source = {"mean": task.exact_mean, "acceptance_probability": .02,
              "joint_probabilities": np.array([.25, .25, .25, .25])}
    routed = {**source, "joint_probabilities": np.array([.25, .4, .1, .25])}
    assert np.asarray(task.score_values) @ source["joint_probabilities"] == pytest.approx(
        np.asarray(task.score_values) @ routed["joint_probabilities"])
    with patch.object(ir, "synthesize_instrument", return_value=synthesized), \
         patch.object(ir, "simulate_scored_trial", side_effect=[source, routed, dict(source)]):
        with pytest.raises(AssertionError, match="routed"):
            cr.synthesize_task(task)


def test_controller_infeasibility_is_declared_before_native_search(tasks, tmp_path):
    prepared = _fake_prepared(tasks["dynamics-n1-Z"], 10**12)
    profile = ir.InstrumentProfile()
    diagnostic = cr.controller_model_floor(prepared, profile)
    cap = diagnostic["factory_retry_cap"]
    assert cap == 8
    # Eleven initialized data/workspace patches, three readouts, one H frame,
    # one trial dispatch and one T injection/transport with capped supply.
    assert diagnostic["controller_decisions_per_attempt"] == 69+4*cap
    assert diagnostic["distance_independent_controller_model_contribution"] == pytest.approx(.101)
    assert diagnostic["budget_exceeded"] is True
    with patch.object(ir, "estimate_instrument", side_effect=AssertionError("native search should not run")):
        result = cr.estimate_workload(prepared, profile=profile, cache_dir=tmp_path)
    assert result["status"] == "infeasible_declared_controller_model"
    assert result["profile"] == asdict(profile)
    assert result["all_feasible_points"] == []


def test_factory_domain_infeasibility_and_native_no_point_are_not_omitted(tasks, tmp_path):
    prepared = _fake_prepared(tasks["dynamics-n1-Z"], 10**12)
    profile = ir.InstrumentProfile(factory_pipeline_failure_bound=.9)
    with patch.object(ir, "estimate_instrument", side_effect=AssertionError("native search should not run")):
        result = cr.estimate_workload(prepared, profile=profile, cache_dir=tmp_path)
    assert result["status"] == "unsupported_factory_retry_model_domain"
    assert result["complete_algorithm_attempts"] == 10**12
    prepared = _fake_prepared(tasks["dynamics-n1-Z"])
    profile = ir.InstrumentProfile(controller_failure_per_decision=1e-18)
    with patch.object(ir, "estimate_instrument", side_effect=RuntimeError("No feasible point in the declared distance/factory/scenario domain")):
        result = cr.estimate_workload(prepared, profile=profile, cache_dir=tmp_path)
    assert result["status"] == "infeasible_declared_distance_factory_domain"
    assert result["all_feasible_points"] == []
    with patch.object(ir, "estimate_instrument", side_effect=RuntimeError("unexpected backend failure")):
        with pytest.raises(RuntimeError, match="unexpected"):
            cr.estimate_workload(prepared, profile=profile, cache_dir=tmp_path)


def test_native_estimator_gets_whole_attempt_count_allocations_and_full_candidates(tasks, tmp_path):
    prepared = _fake_prepared(tasks["thermal-n2-X"])
    profile = ir.InstrumentProfile(controller_failure_per_decision=1e-18)
    with patch.object(ir, "estimate_instrument", return_value={"status": "computed", "all_feasible_points": []}) as estimate:
        result = cr.estimate_workload(prepared, profile=profile, cache_dir=tmp_path)
    estimate.assert_called_once_with(prepared.synthesized, 100, profile=profile,
                                     physical_failure_budget=.00125, factory_availability_budget=.00125,
                                     cache_dir=tmp_path, include_all_feasible_points=True)
    assert result["controller_model_diagnostic"]["budget_exceeded"] is False


def _point(distance, peak, runtime, failure=.0003, availability=.0002):
    return {"data_code_distance": distance, "reserved_logical_patches": 31,
            "factory_physical_qubits": 1000, "peak_physical_qubits": peak,
            "worst_case_runtime_all_attempts_ns": runtime, "full_physical_failure_model": failure,
            "factory_supply": {"pipeline_time_ns": 5000, "output_error": 1e-14,
                               "availability_failure_union_bound": availability}}


def test_energy_uses_common_hardware_even_if_separate_pareto_pruning_loses_it():
    common_x, common_z = _point(7, 5000, 30), _point(7, 5200, 40)
    x_only = _point(9, 4000, 20)  # Dominates X's only common point in isolation.
    result = cr.combine_energy_profiles({"X": {"profile": {"p": .001}, "all_feasible_points": [common_x, x_only], "pareto_points": [x_only]},
                                        "Z": {"profile": {"p": .001}, "all_feasible_points": [common_z], "pareto_points": [common_z]}})
    assert result["status"] == "computed_common_configuration_energy_model"
    assert result["common_feasible_candidate_count"] == 1
    point = result["pareto_points"][0]
    assert point["data_code_distance"] == 7
    assert point["peak_physical_qubits"] == 5200
    assert point["worst_case_runtime_all_attempts_ns"] == 70
    assert point["full_physical_failure_model"] == pytest.approx(.0006)
    assert point["factory_availability_failure_model"] == pytest.approx(.0004)
    assert point["group_points"] == {"X": common_x, "Z": common_z}
    assert result["whole_workload_executed"] is False


def test_energy_join_requires_all_candidates_identical_profiles_and_both_groups():
    report = {"profile": {"p": .001}, "all_feasible_points": [_point(7, 5000, 30)]}
    with pytest.raises(ValueError, match="both"):
        cr.combine_energy_profiles({"X": report})
    with pytest.raises(ValueError, match="same"):
        cr.combine_energy_profiles({"X": report, "Z": {**report, "profile": {"p": .0001}}})
    with pytest.raises(ValueError, match="all feasible"):
        cr.combine_energy_profiles({"X": report, "Z": {"profile": report["profile"], "pareto_points": report["all_feasible_points"]}})
    disjoint = {**report, "all_feasible_points": [_point(9, 5000, 30)]}
    result = cr.combine_energy_profiles({"X": report, "Z": disjoint})
    assert result["status"] == "no_common_feasible_configuration"
    assert result["pareto_points"] == []
