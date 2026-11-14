# SPDX-License-Identifier: MIT
"""Source tests for the fixed robust-QPE quantum-schedule accounting boundary.

These tests do not request synthesis, native factory searches or physical
estimation. A fake estimator is used for adapter dispatch tests below.
"""
from dataclasses import FrozenInstanceError, asdict
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import pytest
from openquantum import robust_qpe_resources as resource


RAW_MACROCYCLE_COUNTS = {
    "p": 18, "cx": 12, "h": 8, "x": 4, "sdg": 2,
    "reset": 8, "measure": 4,
}


def copied_without(circuit, predicate):
    output = circuit.copy_empty_like()
    for index, item in enumerate(circuit.data):
        if not predicate(index, item):
            output.append(item.operation, item.qubits, item.clbits)
    return output


def test_task_freezes_one_physical_workload_and_exact_rational_promises():
    task = resource.RobustResourceTask()
    assert task.phase == .3
    assert task.target_qubits == 1
    assert task.levels == 2 and task.shots == 400
    assert task.additive_bound == Fraction(1, 50)
    assert task.statistical_failure == Fraction(1, 200)
    assert task.physical_failure_budget == task.factory_availability_budget == .0025
    assert task.operator_error_allowance == 1e-4
    with pytest.raises(FrozenInstanceError):
        task.shots = 399


@pytest.mark.parametrize("field,value", [
    ("phase", .125), ("phase", float("nan")), ("phase", True),
    ("target_qubits", 2), ("target_qubits", True), ("target_qubits", 1.0),
    ("levels", 3), ("levels", True), ("levels", 2.0),
    ("shots", 399), ("shots", True), ("shots", 400.0),
    ("additive_bound", Fraction(0)), ("additive_bound", .02),
    ("additive_bound", True),
    ("statistical_failure", Fraction(1, 100)),
    ("statistical_failure", .005), ("statistical_failure", True),
    ("physical_failure_budget", .005), ("physical_failure_budget", True),
    ("factory_availability_budget", .005),
    ("factory_availability_budget", True),
    ("operator_error_allowance", 1e-3), ("operator_error_allowance", True),
])
def test_task_rejects_implicit_scope_changes_and_bool_numeric_aliases(field, value):
    with pytest.raises((ValueError, TypeError)):
        resource.RobustResourceTask(**{field: value})


def test_fixed_ledger_charges_every_trial_including_earlier_powers():
    ledger = resource.design_ledger()
    assert ledger["logical_qubits"] == 2 and ledger["powers"] == [1, 2]
    assert ledger["shots_per_fringe"] == 400
    assert ledger["fresh_preparations"] == ledger["final_readouts"] == 1600
    assert ledger["controlled_U_calls"] == 2400
    assert ledger["maximum_coherent_base_calls"] == 2
    assert ledger["distinct_experiment_circuits"] == 4
    assert ledger["raw_all_trial_counts"] == {
        "p": 7200, "cx": 4800, "h": 3200, "x": 1600, "sdg": 800,
        "reset": 3200, "measure": 1600,
    }
    assert ledger["source_counts_per_macrocycle"] == RAW_MACROCYCLE_COUNTS
    assert ledger["source_operations_per_macrocycle"] == sum(RAW_MACROCYCLE_COUNTS.values())


@pytest.mark.parametrize("n,levels", [(1, 2), (2, 3), (8, 4), (16, 5), (32, 6)])
def test_generation_ledger_scales_actual_access_width_and_all_fresh_trials(n, levels):
    ledger = resource.design_ledger(target_qubits=n, levels=levels)
    powers = [1 << level for level in range(levels)]
    literal_calls = 400*sum(2*power for power in powers)
    trials = 400*len(powers)*2
    assert ledger["logical_qubits"] == n+1
    assert ledger["powers"] == powers
    assert ledger["controlled_U_calls"] == literal_calls
    assert ledger["fresh_preparations"] == ledger["final_readouts"] == trials
    assert ledger["raw_all_trial_counts"]["p"] == 3*n*literal_calls
    assert ledger["raw_all_trial_counts"]["cx"] == 2*n*literal_calls
    assert ledger["raw_all_trial_counts"]["reset"] == (n+1)*trials


@pytest.mark.parametrize("arguments", [
    {"target_qubits": 0}, {"target_qubits": 33}, {"target_qubits": True},
    {"target_qubits": 1.0}, {"levels": 1}, {"levels": 7},
    {"levels": True}, {"levels": 2.0}, {"shots": True},
    {"shots": 399}, {"shots": 400.0},
])
def test_generation_domain_is_separate_but_still_finite(arguments):
    with pytest.raises((ValueError, TypeError)):
        resource.design_ledger(**arguments)


@pytest.mark.parametrize("levels,shots,beta", [
    (2, 400, Fraction(1, 50)), (2, 1, Fraction(0)),
    (3, 400, Fraction(1, 50)), (6, 400, Fraction(0)),
    (6, 400, Fraction(3, 50)), (2, 25_000, Fraction(1, 50)),
])
def test_exact_confidence_law_matches_existing_robust_protocol(levels, shots, beta):
    from openquantum.robust_qpe import failure_bound
    bound = resource.rational_failure_bound(levels, shots, beta)
    assert type(bound) is Fraction and bound == failure_bound(levels, shots, beta)
    if (levels, shots, beta) == (2, 400, Fraction(1, 50)):
        assert bound <= resource.RobustResourceTask().statistical_failure


def test_complete_macrocycle_has_literal_access_and_four_fresh_records():
    task = resource.RobustResourceTask()
    macrocycle, trials = resource.build_macrocycle(task)
    assert macrocycle.num_qubits == 2 and macrocycle.num_clbits == 4
    assert len(trials) == 4
    assert {name: count for name, count in macrocycle.count_ops().items()
            if name != "barrier"} == RAW_MACROCYCLE_COUNTS
    expected_events = []
    for bit in range(4):
        expected_events.extend([("reset", (0,), ()), ("reset", (1,), ()),
                                ("measure", (0,), (bit,))])
    actual_events = [
        (item.operation.name,
         tuple(macrocycle.find_bit(wire).index for wire in item.qubits),
         tuple(macrocycle.find_bit(bit).index for bit in item.clbits))
        for item in macrocycle.data if item.operation.name in {"reset", "measure"}
    ]
    assert actual_events == expected_events
    for trial, power, basis in zip(trials, (1, 1, 2, 2), ("cos", "sin", "cos", "sin")):
        counts = trial.count_ops()
        assert trial.num_qubits == 2 and trial.num_clbits == 1
        assert counts["reset"] == 2 and counts["measure"] == 1
        assert counts["p"] == 3*power and counts["cx"] == 2*power
        assert counts["h"] == 2 and counts["x"] == 1
        assert counts.get("sdg", 0) == (basis == "sin")
        assert not any(item.operation.name in {"unitary", "cp", "if_else"}
                       for item in trial.data)
    resource.check_macrocycle_source(macrocycle, task)


@pytest.mark.parametrize("missing", ("reset", "p", "cx", "sdg", "measure"))
def test_source_guard_detects_missing_preparation_access_quadrature_or_readout(missing):
    task = resource.RobustResourceTask()
    macrocycle, _ = resource.build_macrocycle(task)
    removed = next(index for index, item in enumerate(macrocycle.data)
                   if item.operation.name == missing)
    corrupted = copied_without(macrocycle, lambda index, _: index == removed)
    with pytest.raises(ValueError):
        resource.check_macrocycle_source(corrupted, task)


def test_source_guard_detects_an_angle_change_even_when_counts_and_boundaries_match():
    task = resource.RobustResourceTask()
    macrocycle, _ = resource.build_macrocycle(task)
    corrupted = macrocycle.copy_empty_like()
    changed = False
    for item in macrocycle.data:
        operation = item.operation
        if not changed and operation.name == "p":
            operation = operation.copy()
            operation.params[0] += 1e-3
            changed = True
        corrupted.append(operation, item.qubits, item.clbits)
    assert dict(corrupted.count_ops()) == dict(macrocycle.count_ops())
    with pytest.raises(ValueError):
        resource.check_macrocycle_source(corrupted, task)


def test_splitter_retains_full_trials_and_reindexes_only_classical_outputs():
    task = resource.RobustResourceTask()
    macrocycle, original = resource.build_macrocycle(task)
    split = resource.split_measured_trials(macrocycle)
    assert len(split) == 4
    for extracted, expected in zip(split, original):
        assert extracted.num_qubits == 2 and extracted.num_clbits == 1
        def records(circuit):
            return [
                (item.operation.name,
                 tuple(circuit.find_bit(wire).index for wire in item.qubits),
                 tuple(circuit.find_bit(bit).index for bit in item.clbits),
                 tuple(item.operation.params))
                for item in circuit.data if item.operation.name != "barrier"
            ]
        assert records(extracted) == records(expected)


def test_splitter_rejects_reused_record_and_quantum_work_after_last_readout():
    macrocycle, _ = resource.build_macrocycle(resource.RobustResourceTask())
    reused = macrocycle.copy_empty_like()
    for item in macrocycle.data:
        if item.operation.name == "measure":
            reused.measure(reused.find_bit(item.qubits[0]).index, 0)
        else:
            reused.append(item.operation, item.qubits, item.clbits)
    with pytest.raises(ValueError):
        resource.split_measured_trials(reused)
    trailing = macrocycle.copy()
    trailing.x(1)
    with pytest.raises(ValueError):
        resource.split_measured_trials(trailing)


@pytest.mark.parametrize("missing", ("reset", "measure"))
def test_compiled_guard_rejects_missing_boundary_before_any_numerical_check(missing, monkeypatch):
    task = resource.RobustResourceTask()
    fake = fake_synthesized_macrocycle()
    removed = next(index for index, item in enumerate(fake.circuit.data)
                   if item.operation.name == missing)
    fake.circuit = copied_without(fake.circuit, lambda index, _: index == removed)
    def forbidden(_):
        raise AssertionError("A malformed compiled boundary reached density evolution")
    monkeypatch.setattr(resource, "_zero_probability_without_renormalization", forbidden)
    with pytest.raises((ValueError, TypeError)):
        resource.validate_compiled_macrocycle(fake, task)


@pytest.mark.parametrize("allowance", (None, True, -1e-4, float("nan"), 1e-3))
def test_compiled_synthesis_metadata_cannot_bypass_the_fringe_guard(allowance, monkeypatch):
    fake = fake_synthesized_macrocycle()
    fake.report["guarded_numerical_operator_error_upper_estimate"] = allowance
    def forbidden(_):
        raise AssertionError("An unsupported synthesis allowance reached density evolution")
    monkeypatch.setattr(resource, "_zero_probability_without_renormalization", forbidden)
    with pytest.raises(ValueError):
        resource.validate_compiled_macrocycle(fake, resource.RobustResourceTask())


def fake_synthesized_macrocycle():
    """Dispatch-only fixture, explicitly not a synthesized phase oracle."""
    source, _ = resource.build_macrocycle(resource.RobustResourceTask())
    compiled = source.copy_empty_like()
    for item in source.data:
        if item.operation.name == "p":
            compiled.t(compiled.find_bit(item.qubits[0]).index)
        else:
            compiled.append(item.operation, item.qubits, item.clbits)
    return SimpleNamespace(source=source, routed=source, circuit=compiled,
                           report={"guarded_numerical_operator_error_upper_estimate": 0.0})


def fake_native_report(profile, repetitions):
    point = {
        "data_code_distance": 11, "peak_physical_qubits": 1000,
        "reserved_logical_patches": 7, "factory_physical_qubits": 500,
        "runtime_per_complete_attempt_ns": 1000,
        "worst_case_runtime_all_attempts_ns": 1000*repetitions,
        "code_cycles_per_complete_attempt": 1,
        "full_physical_failure_model": .001,
        "failure_components_all_attempts": {"data_workspace_code_exposure": .001},
        "factory_supply": {"full_pipeline_retry_cap_per_state": 4,
                           "availability_failure_union_bound": .0001},
        "macro_ledger_per_complete_attempt": [],
    }
    return {
        "schema_version": 1,
        "status": "computed_conditional_instrument_surface_code_model",
        "profile": asdict(profile), "complete_algorithm_attempts": repetitions,
        "physical_failure_budget": .0025, "factory_availability_budget": .0025,
        "pareto_points": [point], "all_feasible_points": [deepcopy(point)],
        "assumptions": [], "excluded": [],
    }


def test_one_common_native_estimate_charges_four_trial_macrocycle_400_times(tmp_path):
    task = resource.RobustResourceTask()
    synthesized = fake_synthesized_macrocycle()
    calls = []
    preserved = {}
    def estimator(actual, repetitions, **kwargs):
        calls.append((actual, repetitions, kwargs))
        report = fake_native_report(kwargs["profile"], repetitions)
        preserved["report"] = report
        preserved["snapshot"] = deepcopy(report)
        return report
    report = resource.estimate_quantum_schedule(synthesized, task,
                                               cache_dir=tmp_path/"private-cache",
                                               estimator=estimator)
    assert len(calls) == 1
    actual, repetitions, kwargs = calls[0]
    assert actual is synthesized and repetitions == 400
    assert actual.source.count_ops()["measure"]*repetitions == 1600
    assert actual.source.count_ops()["reset"]*repetitions == 3200
    profile = kwargs["profile"]
    assert profile.physical_error_rate == 1e-4 and profile.maximum_code_distance == 15
    assert profile.physical_gate_time_ns == profile.physical_initialization_time_ns == 100
    assert profile.physical_measurement_time_ns == 500
    assert profile.code_cycle_time_ns == 1000 and profile.reaction_time_ns == 10_000
    assert kwargs["physical_failure_budget"] == kwargs["factory_availability_budget"] == .0025
    assert kwargs["include_all_feasible_points"] is True
    assert kwargs["cache_dir"] == tmp_path/"private-cache"
    assert report["complete_end_to_end_cost"] is False
    assert report["end_to_end_confidence_established"] is False
    assert report["compiled_fringe_check_executed"] is False
    assert report["comparison_improvement_computed"] is False
    classical = report["classical"]
    assert classical["complete_end_to_end_cost"] is False
    assert classical["end_to_end_confidence_established"] is False
    for section in ("classical_input_preparation", "count_aggregation", "rational_inference"):
        assert classical[section]["included_in_physical_runtime"] is False
        assert classical[section]["wall_time_measured"] is False
    counts = classical["count_aggregation"]
    assert counts["represented_readout_bits"] == 1600
    assert counts["count_registers"] == 4 and counts["bits_per_count"] == 9
    assert classical["rational_inference"]["phase_output_computed"] is False
    exclusions = classical["excluded_from_physical_totals"]
    for missing_cost in ("count accumulation", "classical record storage", "rational phase inference",
                         "quantum idle exposure", "experimental bias calibration", "controller area"):
        assert any(missing_cost in item for item in exclusions)
    assert preserved["report"] == preserved["snapshot"]


def test_native_report_must_retain_all_macrocycle_repetitions(tmp_path):
    calls = []
    def estimator(actual, repetitions, **kwargs):
        calls.append(repetitions)
        return fake_native_report(kwargs["profile"], repetitions-1)
    with pytest.raises(RuntimeError, match="repetition count"):
        resource.estimate_quantum_schedule(fake_synthesized_macrocycle(),
                                           resource.RobustResourceTask(),
                                           cache_dir=tmp_path/"private-cache", estimator=estimator)
    assert calls == [400]


def test_native_failure_is_not_retried_or_replaced_with_a_lighter_workload(tmp_path):
    calls = []
    def estimator(actual, repetitions, **kwargs):
        calls.append((actual, repetitions))
        raise RuntimeError("sentinel native failure")
    with pytest.raises(RuntimeError, match="sentinel native failure"):
        resource.estimate_quantum_schedule(fake_synthesized_macrocycle(),
                                           resource.RobustResourceTask(),
                                           cache_dir=tmp_path/"private-cache",
                                           estimator=estimator)
    assert len(calls) == 1 and calls[0][1] == 400
