"""Fixed n3 contracts and pipeline guards; no native synthesis/model jobs."""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction

from openquantum import instrument_resources as ir
from openquantum.simulation import lindblad_three_physical as lp


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    task = lp.source_task()
    gate = lp.lw.gate_density(task, lp.STEPS)
    return task, gate


def stub_synthesized(error=1e-6):
    """Metadata-plumbing fixture, never an actual compiled circuit result."""
    source = lp.build_source()
    stub = source.copy_empty_like()
    for item in source.data:
        name = item.operation.name
        wires = [source.find_bit(bit).index for bit in item.qubits]
        if name in {"reset", "measure", "h", "s"} or (name == "cx" and abs(wires[0]-wires[1]) == 1):
            stub.append(item.operation, item.qubits, item.clbits)
    stub.t(0)
    # Keep the fixture's T before final readouts, even though the checker is
    # mocked below: the synthetic schedule itself must have final readout.
    item = stub.data.pop()
    stub.data.insert(len(stub.data)-6, item)
    report = {"preserve_full_unitary": True, "transpiler_qubits_initially_zero": False,
              "limits": asdict(lp.limits()), "rotation_occurrences": 2680,
              "guarded_numerical_operator_error_upper_estimate": error,
              "clifford_t_counts": dict(stub.count_ops()), "routed_counts": dict(stub.count_ops()),
              "source_counts": dict(source.count_ops()), "actual_synthesis_sequences_generated": True}
    return ir.SynthesizedInstrument(source, stub.copy(), stub, report)


def checked_fake(rho):
    diagonal = np.diag(rho).real.copy()
    return {"acceptance_probability": 1., "mean": float(np.asarray(lp.score_values()) @ diagonal),
            "joint_probabilities": diagonal, "reduced_before_output": rho.copy(),
            "simulation": {"method": "explicit mocked pipeline fixture"}}


@pytest.fixture
def prepared(monkeypatch, reference):
    task, gate = reference
    synthesized = stub_synthesized()
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    return lp.check_synthesized(synthesized, task)


def test_fixed_task_and_literal_complete_schedule():
    task = lp.source_task()
    assert asdict(task) == {"n_system": 3, "time": 1., "J": .6, "hx": .35, "rate0": .8, "other_rate": .5}
    source = lp.build_source()
    assert len(source.data) == 4435
    assert source.num_qubits == source.num_clbits == 6
    assert source.count_ops()["reset"] == 673
    assert source.count_ops()["measure"] == 6
    assert source.metadata["environment_initializations"] == [268, 268, 134]
    assert source.metadata["environment_discard_resets_before_reuse"] == [267, 267, 133]
    assert [source.find_bit(item.clbits[0]).index for item in source.data if item.operation.name == "measure"] == list(range(6))


@pytest.mark.parametrize("task", [lp.lw.ChainTask(2), lp.lw.ChainTask(4), lp.lw.ChainTask(3, time=.5), object()])
def test_other_tasks_fail_closed(task):
    with pytest.raises(ValueError, match="Only literal n3"):
        lp.build_source(task)


def test_explicit_optin_limits_preserve_frozen_defaults():
    defaults = ir.InstrumentLimits()
    assert defaults.max_qubits == 5 and defaults.synthesized_operations == 100000
    bounds = lp.limits()
    assert bounds.max_qubits == 6 and bounds.source_operations == 20000
    assert bounds.synthesized_operations == bounds.validation_operations == 1000000
    density = lp.checker_limits()
    assert density.max_qubits == 6 and density.validation_operations == 1000000
    assert lp.run_caps()["native_rotation_synthesis_attempts_upper_bound"] == 10720
    assert lp.run_caps()["prepare_seconds"] == 1200
    assert lp.run_caps()["parent_seconds"] == 1800


def test_exact_uniform_policy_and_unchanged_statistics():
    task = lp.source_task()
    assert lp.lw.uniform_split_bound(task, 134) == Fraction(177147, 17956000)
    assert lp.lw.uniform_split_bound(task, 133) > Fraction(1, 100)
    assert lp.lw.output_contract()["complete_trials_per_case"] == lp.COMPLETE_TRIALS == 15756
    assert lp.score_values() == (1., -1., 1., -1., 1., -1., 1., -1.)


def test_family_failure_allocations_are_exact_downward_and_not_reassigned():
    family = lp.family_allocations()
    assert family["implemented_physical_cases"] == [2, 3]
    assert family["new_physical_cases"] == [3]
    assert family["preserved_prior_physical_cases"] == [2]
    assert family["unused_cases_reallocated"] is False
    assert family["family_physical_confidence_implemented"] is False
    for case in family["case_allocations"].values():
        assert case["statistical_failure_exact"] == {"numerator": 1, "denominator": 600}
        assert case["physical_failure_exact"] == case["factory_availability_failure_exact"] == {"numerator": 1, "denominator": 1200}
        assert Fraction.from_float(case["physical_failure_budget"]) <= Fraction(1, 1200)
        assert Fraction.from_float(case["factory_availability_budget"]) <= Fraction(1, 1200)
    assert family["case_allocations"]["2"]["physical_status"] == "preserved_prior_n2_result"
    assert family["case_allocations"]["4"]["physical_status"] == "unimplemented_reserved_allocation"


def test_actual_synthesis_error_uses_exact_binary64_and_separate_reserves():
    actual = 1.23456789012345e-6
    contract = lp.error_contract(actual)
    eta = Fraction.from_float(actual)+Fraction(1, 10000000)+Fraction(1, 10000000)
    assert contract["whole_realization_error_exact"] == {"numerator": eta.numerator, "denominator": eta.denominator}
    assert Fraction.from_float(contract["realization_Z0_bias_upper_proposal"]) >= 2*eta
    assert contract["splitting_bias_exact"] == {"numerator": 177147, "denominator": 17956000}
    assert contract["requested_synthesis_allowance"] == 5e-5
    assert contract["full_representation_certificate"] is False
    assert contract["formal_synthesis_certificate"] is False
    assert contract["coherent_error_is_stochastic_failure"] is False
    assert contract["reserve_is_inferred_from_density_agreement"] is False


@pytest.mark.parametrize("actual", [-1., float("nan"), float("inf"), True, 6e-5])
def test_invalid_synthesis_premises_reject(actual):
    with pytest.raises(ValueError):
        lp.error_contract(actual)


def test_representation_scope_does_not_turn_damping_functions_into_certificate():
    report = lp.representation_record()
    assert report["representation_reserve_is_certified"] is False
    assert "not supplied" in report["damping_angle_functions"]
    h_exact = report["Hamiltonian_angle_component_exact"]
    assert Fraction(h_exact["numerator"], h_exact["denominator"]) >= 0
    counts = Counter()
    for target in report["stored_source_angle_targets"]:
        counts[target["gate"]] += target["occurrences"]
    assert counts == {"rx": 804, "rz": 536, "cry": 670}


def test_check_pipeline_counts_all_trials_and_preserves_joint_density(prepared):
    workload = prepared.workload
    assert workload["complete_trials"] == workload["complete_attempts"] == 15756
    assert workload["algorithm_retry_cap"] == 1 and workload["postselection"] == "none"
    assert workload["source_resets_all_trials"] == 10603788
    assert workload["source_readouts_all_trials"] == 94536
    assert workload["workspace_startup_initializations_all_trials"] == 267852
    assert workload["logical_startup_initializations_all_trials"] == 362388
    assert workload["reserved_logical_patches"] == 23
    assert workload["source_first_wire_initializations_per_trial"] == 6
    assert workload["source_reuse_resets_per_trial"] == 667
    assert workload["data_output_clbits"] == [0, 1, 2]
    assert workload["ignored_environment_readout_clbits"] == [3, 4, 5]
    assert workload["unobserved_reset_histories_claimed_as_outputs"] is False
    for name, count in workload["primitive_counts_per_complete_trial"].items():
        assert workload["primitive_counts_all_trials"][name] == count*15756
    record = prepared.as_record()
    assert record["joint_system_correlations_preserved"] is True
    assert record["unobserved_reset_histories_are_measured_outputs"] is False
    assert len(record["source_instrument_check"]["joint_probabilities"]) == 8
    assert len(record["source_instrument_check"]["reduced_before_output"]["real"]) == 8
    assert record["hardware_execution"] is False


def test_prepare_constructor_passes_exact_fullunitary_optin(monkeypatch, reference):
    _, gate = reference
    calls = []
    def synthetic(source, allowance, **kwargs):
        calls.append((source, allowance, kwargs))
        return stub_synthesized(2e-6)
    monkeypatch.setattr(ir, "synthesize_instrument", synthetic)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    prepared = lp.prepare_workload()
    assert len(calls) == 1
    assert calls[0][1] == 5e-5
    assert calls[0][2] == {"limits": lp.limits(), "preserve_full_unitary": True}
    assert prepared.contract["guarded_actual_synthesis_error"] == 2e-6


@pytest.mark.parametrize("changed", ["clean_input", "rotations", "limits", "counts"])
def test_forged_compilation_metadata_fails(changed, monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    if changed == "clean_input":
        synth.report["transpiler_qubits_initially_zero"] = True
    elif changed == "rotations":
        synth.report["rotation_occurrences"] = 760
    elif changed == "limits":
        synth.report["limits"]["max_qubits"] = 8
    else:
        synth.report["clifford_t_counts"]["t"] += 1
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    with pytest.raises(AssertionError):
        lp.check_synthesized(synth)


def test_same_Z0_mean_cannot_hide_correlated_density_routing_change(monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    z1 = lp.lw.on_wire(lp.lw.Z, 1, 3)
    changed = z1@gate@z1
    assert checked_fake(changed)["mean"] == pytest.approx(checked_fake(gate)["mean"])
    assert np.allclose(np.diag(changed), np.diag(gate))
    def check(circuit, *args, **kwargs):
        return checked_fake(changed if circuit is synth.routed else gate)
    monkeypatch.setattr(lp.di, "simulate_density_trial", check)
    with pytest.raises(AssertionError, match="correlated three-system"):
        lp.check_synthesized(synth)


def test_only_full_three_output_density_is_accepted(monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    def marginalized(*args, **kwargs):
        result = checked_fake(gate)
        result["joint_probabilities"] = np.array([.5, .5])
        result["reduced_before_output"] = np.eye(2)/2
        return result
    monkeypatch.setattr(lp.di, "simulate_density_trial", marginalized)
    with pytest.raises(AssertionError, match="three correlated system outputs"):
        lp.check_synthesized(synth)


def test_controller_infeasibility_keeps_profile_and_never_native_searches(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(ir, "_event_counts", lambda circuit: Counter({"t_q5": 500000}))
    monkeypatch.setattr(ir, "estimate_instrument", lambda *args, **kwargs: pytest.fail("infeasible controller must not run native estimates"))
    for profile in lp.fixed_profiles():
        report = lp.estimate_prepared(prepared, profile=profile, cache_dir=tmp_path)
        assert report["status"] == "infeasible_declared_controller_model"
        assert report["pareto_points"] == report["all_feasible_points"] == []
        assert report["complete_algorithm_attempts"] == 15756
        assert report["controller_model_diagnostic"]["budget_exceeded"] is True
        assert report["family_allocations"]["unused_cases_reallocated"] is False


def test_native_distance_failure_is_retained(prepared, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("No feasible point in the declared distance/factory/scenario domain")
    monkeypatch.setattr(ir, "estimate_instrument", fail)
    report = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[0], cache_dir=tmp_path)
    assert report["status"] == "infeasible_declared_distance_factory_domain"
    assert report["physical_failure_budget"] == lp._downward(Fraction(1, 1200))


def test_native_adapter_uses_actual_trials_and_original_budgets(prepared, monkeypatch, tmp_path):
    calls = []
    def native(synthesized, count, **kwargs):
        calls.append((synthesized, count, kwargs))
        return {"status": "computed_conditional_instrument_surface_code_model", "pareto_points": [], "all_feasible_points": []}
    monkeypatch.setattr(ir, "estimate_instrument", native)
    result = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[1], cache_dir=tmp_path)
    assert calls[0][0] is prepared.synthesized and calls[0][1] == 15756
    assert calls[0][2]["physical_failure_budget"] == lp._downward(Fraction(1, 1200))
    assert calls[0][2]["factory_availability_budget"] == lp._downward(Fraction(1, 1200))
    assert calls[0][2]["include_all_feasible_points"] is True
    assert result["family_allocations"]["new_physical_cases"] == [3]


def test_changed_controller_scenario_is_not_substituted(prepared, tmp_path):
    with pytest.raises(ValueError, match="two frozen"):
        lp.estimate_prepared(prepared, profile=ir.InstrumentProfile(controller_failure_per_decision=1e-18), cache_dir=tmp_path)


def test_prior_n2_identity_and_n4_no_rerun_flags(monkeypatch):
    manifest = ROOT/"assets/generated/simulation/lindblad-physical/manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"
    monkeypatch.setattr(ir, "synthesize_instrument", lambda *args, **kwargs: pytest.fail("family status must not synthesize old or unimplemented cases"))
    records = lp.family_status_records(check_rejections=False)
    assert records["2"]["status"] == "preserved_prior_n2_result"
    assert records["2"]["regenerated"] is False
    assert records["3"]["selected_steps"] == 134
    assert records["4"]["status"] == "ensemble_rejection_not_rechecked"
    assert records["4"]["synthesis_performed"] is records["4"]["physical_estimation_performed"] is False


def test_n4_density_width_rejection_is_explicit_without_history_or_density_simulation(monkeypatch):
    calls = []
    validate = lp.di.validate_density_source
    def checked(source, *, limits):
        calls.append((source.num_qubits, len(source.data), limits))
        return validate(source, limits=limits)
    monkeypatch.setattr(lp.di, "validate_density_source", checked)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("n4 must not run density evolution"))
    monkeypatch.setattr(ir, "simulate_scored_trial", lambda *args, **kwargs: pytest.fail("this diagnostic excludes an ensemble rerun"))
    record = lp.family_status_records(check_rejections=False)["4"]
    assert calls == [(8, 9934, lp.checker_limits())]
    assert record["density_checker_status"] == "density_width_cap_exceeded"
    assert record["density_width_rejection_reproduced"] is True
    assert record["density_required_wires"] == 8 and record["density_supported_wires"] == 6
    assert "width" in record["density_checker_reason"]
    assert record["density_checker_limits"] == asdict(lp.checker_limits())
    assert record["physical_status"] == "unimplemented_reserved_allocation"
    assert record["synthesis_performed"] is record["physical_estimation_performed"] is False


@pytest.mark.parametrize("field", ["routed", "circuit"])
def test_duplicate_ignored_readout_target_is_rejected_before_density_checks(field, monkeypatch):
    synth = stub_synthesized()
    circuit = getattr(synth, field)
    for index, item in enumerate(circuit.data):
        if item.operation.name == "measure" and circuit.find_bit(item.clbits[0]).index == 3:
            circuit.data[index] = CircuitInstruction(item.operation, (circuit.qubits[0],), item.clbits)
            break
    else:
        pytest.fail("Fixture must contain the ignored c3 readout")
    assert circuit.count_ops()["measure"] == 6
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("invalid literal readout must not be simulated"))
    with pytest.raises(ValueError, match="all six physical wires exactly once"):
        lp.check_synthesized(synth)
