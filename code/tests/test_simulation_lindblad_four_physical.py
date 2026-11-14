"""Fixed n4 contracts and pipeline guards; no native synthesis/model jobs."""
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.quantum_info import Operator

from openquantum import instrument_resources as ir
from openquantum.simulation import lindblad_four_physical as lp


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def reference():
    task = lp.source_task()
    gate = lp.lw.gate_density(task, lp.STEPS)
    return task, gate


@lru_cache(maxsize=1)
def routed_template():
    return lp.cc.route_source(lp.build_source(), limits=lp.limits()).routed


def stub_synthesized(error=1e-6):
    """Metadata/accounting fixture, never an actual compiled circuit result."""
    source = lp.build_source()
    routed = routed_template().copy()
    stub = source.copy_empty_like()
    for item in source.data:
        name = item.operation.name
        wires = [source.find_bit(bit).index for bit in item.qubits]
        if name in {"reset", "measure", "h", "s"} or (name == "cx" and abs(wires[0]-wires[1]) == 1):
            stub.append(item.operation, item.qubits, item.clbits)
    stub.t(0)
    item = stub.data.pop()
    stub.data.insert(len(stub.data)-8, item)
    report = {"preserve_full_unitary": True, "transpiler_qubits_initially_zero": False,
              "compiler_domain": "literal-n4-explicit-opt-in-v1",
              "materialization_limits": asdict(lp.limits()), "rotation_occurrences": 5908,
              "operator_error_allowance": lp.SYNTHESIS_ALLOWANCE,
              "guarded_numerical_operator_error_upper_estimate": error,
              "clifford_t_counts": dict(stub.count_ops()), "routed_counts": dict(routed.count_ops()),
              "source_counts": dict(source.count_ops()), "actual_synthesis_sequences_generated": True,
              "actual_clifford_t_synthesis_performed": True, "derivation": lp.cc.derivation_record(),
              "source_global_phase": float(source.global_phase), "routed_global_phase": float(routed.global_phase),
              "compiled_global_phase": float(stub.global_phase),
              "source_final_measurement_wires": list(range(8)),
              "routed_final_measurement_wires": [routed.find_bit(item.qubits[0]).index for item in routed.data if item.operation.name == "measure"],
              "clifford_t_final_measurement_wires": list(range(8))}
    return lp.cc.SynthesizedFourInstrument(source, routed, stub, report)


def checked_fake(rho):
    diagonal = np.diag(rho).real.copy()
    return {"acceptance_probability": 1., "mean": float(np.asarray(lp.score_values()) @ diagonal),
            "joint_probabilities": diagonal, "reduced_before_output": rho.copy(),
            "unnormalized_trace_before_output": 1., "normalization_divisor": 1.,
            "normalization_is_postselection": False, "simulation": {"method": "explicit mocked pipeline fixture"}}


@pytest.fixture
def prepared(monkeypatch, reference):
    task, gate = reference
    synthesized = stub_synthesized()
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    return lp.check_synthesized(synthesized, task)


def test_fixed_task_and_literal_complete_schedule():
    task = lp.source_task()
    assert asdict(task) == {"n_system": 4, "time": 1., "J": .6, "hx": .35, "rate0": .8, "other_rate": .5}
    source = lp.build_source()
    assert len(source.data) == 9934
    assert source.num_qubits == source.num_clbits == 8
    assert source.count_ops()["reset"] == 1481
    assert source.count_ops()["measure"] == 8
    assert source.metadata["environment_initializations"] == [422, 422, 422, 211]
    assert source.metadata["environment_discard_resets_before_reuse"] == [421, 421, 421, 210]
    assert [source.find_bit(item.clbits[0]).index for item in source.data if item.operation.name == "measure"] == list(range(8))


@pytest.mark.parametrize("task", [lp.lw.ChainTask(2), lp.lw.ChainTask(3), lp.lw.ChainTask(4, time=.5), object()])
def test_other_tasks_fail_closed(task):
    with pytest.raises(ValueError, match="Only literal n4"):
        lp.build_source(task)


def test_explicit_optin_limits_preserve_frozen_defaults():
    defaults = ir.InstrumentLimits()
    assert defaults.max_qubits == 5 and defaults.synthesized_operations == 100000
    bounds = lp.limits()
    assert bounds.max_qubits == 8 and bounds.source_operations == 20000
    assert bounds.routed_operations == 100000 and bounds.synthesized_operations == 2000000
    density = lp.checker_limits()
    assert density.max_qubits == 8 and density.validation_operations == 2000000
    assert lp.run_caps()["native_rotation_synthesis_attempts_upper_bound"] == 23632
    assert lp.run_caps()["prepare_seconds"] == 1800
    assert lp.run_caps()["parent_seconds"] == 2400


def test_exact_uniform_policy_and_unchanged_statistics():
    task = lp.source_task()
    assert lp.lw.uniform_split_bound(task, 211) == Fraction(1331, 133563)
    assert lp.lw.uniform_split_bound(task, 210) > Fraction(1, 100)
    assert lp.lw.output_contract()["complete_trials_per_case"] == lp.COMPLETE_TRIALS == 15756
    assert lp.score_values() == tuple(1-2*(i&1) for i in range(16))


def test_family_failure_allocations_are_exact_downward_and_not_reassigned():
    family = lp.family_allocations()
    assert family["implemented_physical_cases"] == [2, 3]
    assert family["new_physical_cases"] == [4]
    assert family["preserved_prior_physical_cases"] == [2, 3]
    assert family["unused_cases_reallocated"] is False
    assert family["family_physical_confidence_implemented"] is False
    for case in family["case_allocations"].values():
        assert case["statistical_failure_exact"] == {"numerator": 1, "denominator": 600}
        assert case["physical_failure_exact"] == case["factory_availability_failure_exact"] == {"numerator": 1, "denominator": 1200}
        assert Fraction.from_float(case["physical_failure_budget"]) <= Fraction(1, 1200)
        assert Fraction.from_float(case["factory_availability_budget"]) <= Fraction(1, 1200)
    assert family["case_allocations"]["2"]["physical_status"] == "preserved_prior_n2_result"
    assert family["case_allocations"]["4"]["physical_status"] == "new_bounded_n4_assessment_not_feasibility_claim"


def test_actual_synthesis_error_uses_exact_binary64_and_separate_reserves():
    actual = 1.23456789012345e-6
    contract = lp.error_contract(actual)
    eta = Fraction.from_float(actual)+Fraction(1, 10000000)+Fraction(1, 10000000)
    assert contract["whole_realization_error_exact"] == {"numerator": eta.numerator, "denominator": eta.denominator}
    assert Fraction.from_float(contract["realization_Z0_bias_upper_proposal"]) >= 2*eta
    assert contract["splitting_bias_exact"] == {"numerator": 1331, "denominator": 133563}
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
    assert counts == {"rx": 1688, "rz": 1266, "cry": 1477}



def test_check_pipeline_counts_all_trials_and_preserves_joint_density(prepared):
    workload = prepared.workload
    assert workload["complete_trials"] == workload["complete_attempts"] == 15756
    assert workload["algorithm_retry_cap"] == 1 and workload["postselection"] == "none"
    assert workload["source_resets_all_trials"] == 23334636
    assert workload["source_readouts_all_trials"] == 126048
    assert workload["workspace_startup_initializations_all_trials"] == 362388
    assert workload["logical_startup_initializations_all_trials"] == 488436
    assert workload["reserved_logical_patches"] == 31
    assert workload["source_first_wire_initializations_per_trial"] == 8
    assert workload["source_reuse_resets_per_trial"] == 1473
    assert workload["environment_initializations_per_trial"] == [422, 422, 422, 211]
    assert workload["data_output_clbits"] == [0, 1, 2, 3]
    assert workload["ignored_environment_readout_clbits"] == [4, 5, 6, 7]
    assert workload["unobserved_reset_histories_claimed_as_outputs"] is False
    for name, count in workload["primitive_counts_per_complete_trial"].items():
        assert workload["primitive_counts_all_trials"][name] == count*15756
    record = prepared.as_record()
    assert record["joint_system_correlations_preserved"] is True
    assert len(record["source_instrument_check"]["joint_probabilities"]) == 16
    assert len(record["source_instrument_check"]["reduced_before_output"]["real"]) == 16
    assert record["hardware_execution"] is record["formal_confidence_certificate"] is False
    assert record["legacy_materialization_validation"]["legacy_synthesis_performed"] is False


def test_prepare_constructor_uses_separate_explicit_api(monkeypatch, reference):
    _, gate = reference
    calls = []
    def synthetic(source, allowance, **kwargs):
        calls.append((source, allowance, kwargs))
        return stub_synthesized(2e-6)
    monkeypatch.setattr(lp.cc, "synthesize_literal_four", synthetic)
    monkeypatch.setattr(ir, "synthesize_instrument", lambda *args, **kwargs: pytest.fail("new domain must not mutate the shared compiler"))
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    prepared = lp.prepare_workload()
    assert len(calls) == 1 and calls[0][1] == 5e-5
    assert calls[0][2] == {"limits": lp.limits()}
    assert prepared.contract["guarded_actual_synthesis_error"] == 2e-6


@pytest.mark.parametrize("kwargs", [{"synthesis_allowance": 1e-4}, {"representation_reserve": 2e-7}, {"sdk_reserve": 0}])
def test_fixed_requests_cannot_be_changed_to_force_a_result(kwargs, monkeypatch):
    monkeypatch.setattr(lp.cc, "synthesize_literal_four", lambda *args, **kw: pytest.fail("changed premises must reject before native"))
    with pytest.raises(ValueError, match="predeclared"):
        lp.prepare_workload(**kwargs)


@pytest.mark.parametrize("changed", ["clean_input", "rotations", "new_limits", "legacy_limits", "counts", "allowance", "performed", "derivation", "phase", "targets"])
def test_forged_compilation_metadata_fails(changed, monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    if changed == "clean_input":
        synth.report["transpiler_qubits_initially_zero"] = True
    elif changed == "rotations":
        synth.report["rotation_occurrences"] = 5907
    elif changed == "new_limits":
        synth.report["materialization_limits"]["synthesized_operations"] = 1_000_000
    elif changed == "legacy_limits":
        synth.report["limits"] = asdict(ir.InstrumentLimits(max_qubits=8))
    elif changed == "allowance":
        synth.report["operator_error_allowance"] = 1e-4
    elif changed == "performed":
        synth.report["actual_clifford_t_synthesis_performed"] = False
    elif changed == "derivation":
        synth.report["derivation"]["shared_IR_limits_modified"] = True
    elif changed == "phase":
        synth.report["compiled_global_phase"] += .01
    elif changed == "targets":
        synth.report["clifford_t_final_measurement_wires"][4] = 0
    else:
        synth.report["clifford_t_counts"]["t"] += 1
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: checked_fake(gate))
    with pytest.raises(AssertionError):
        lp.check_synthesized(synth)


def test_actual_rotation_counter_is_not_only_a_metadata_claim(monkeypatch):
    synth = stub_synthesized()
    index = next(i for i, item in enumerate(synth.routed.data) if item.operation.name == "rx")
    del synth.routed.data[index]
    synth.report["routed_counts"] = dict(synth.routed.count_ops())
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("bad actual rotation count must reject first"))
    with pytest.raises(AssertionError, match="5908"):
        lp.check_synthesized(synth)


def test_same_Z0_and_joint_diagonal_cannot_hide_routing_state_change(monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    z1 = lp.lw.on_wire(lp.lw.Z, 1, 4)
    changed = z1@gate@z1
    assert checked_fake(changed)["mean"] == pytest.approx(checked_fake(gate)["mean"])
    assert np.allclose(np.diag(changed), np.diag(gate))
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda circuit, *args, **kwargs: checked_fake(changed if circuit is synth.routed else gate))
    with pytest.raises(AssertionError, match="full correlated four-system"):
        lp.check_synthesized(synth)


def test_same_local_mean_cannot_hide_synthesized_full_state_error(monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    z1 = lp.lw.on_wire(lp.lw.Z, 1, 4)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda circuit, *args, **kwargs: checked_fake(z1@gate@z1 if circuit is synth.circuit else gate))
    with pytest.raises(AssertionError, match="correlated instrument"):
        lp.check_synthesized(synth)


def test_only_full_four_output_density_is_accepted(monkeypatch, reference):
    _, gate = reference
    synth = stub_synthesized()
    def marginalized(*args, **kwargs):
        result = checked_fake(gate)
        result["joint_probabilities"] = np.array([.5, .5])
        result["reduced_before_output"] = np.eye(2)/2
        return result
    monkeypatch.setattr(lp.di, "simulate_density_trial", marginalized)
    with pytest.raises(AssertionError, match="four correlated system outputs"):
        lp.check_synthesized(synth)


def test_raw_trace_guard_is_predeclared_and_normalization_disclosed(reference):
    _, gate = reference
    check = checked_fake(gate)
    check["acceptance_probability"] = check["unnormalized_trace_before_output"] = check["normalization_divisor"] = 1+7e-9
    lp._annotate_check(check)
    assert check["normalization_probability_is_postselection"] is False
    check["acceptance_probability"] = 1+1.1e-8
    with pytest.raises(AssertionError, match="raw trace"):
        lp._annotate_check(check)
    check = checked_fake(gate)
    check["normalization_divisor"] = 1.1
    with pytest.raises(AssertionError, match="divisor"):
        lp._annotate_check(check)


def test_source_preflight_is_real_full_correlated_cpu_check_without_native(monkeypatch):
    monkeypatch.setattr(lp.cc, "gridsynth_unitary", lambda *args, **kwargs: pytest.fail("preflight must not synthesize"))
    result = lp.source_preflight()
    record = result.as_record()
    assert len(result.source.data) == 9934 and len(result.routed.data) == 24510
    assert record["actual_routing"]["rotation_occurrences"] == 5908
    assert result.routed.count_ops()["cx"] == 14130
    assert np.shape(result.source_check["reduced_before_output"]) == (16, 16)
    assert len(result.source_check["joint_probabilities"]) == 16
    assert result.independent_reference["source_vs_gate_contracted_system_trace_norm_residual"] < 1e-10
    assert record["synthesis_performed"] is record["physical_estimation_performed"] is False
    assert record["output_clbits"] == [0, 1, 2, 3]
    assert abs(record["source_instrument_check"]["mean"]-.7910514390501098) < 1e-10
    witness = result.independent_reference["correlation_witness"]
    assert abs(witness["source_minus_product_Z0"]) < 1e-10
    assert witness["product_of_marginals_system_trace_norm_discrepancy"] > .1


def test_controller_ledger_uses_actual_wires_and_full_readiness_retry(prepared):
    profile = lp.fixed_profiles()[0]
    events = ir._event_counts(prepared.synthesized.circuit)
    assert events["initial_reset"] == 8 and events["reset"] == 1473
    states = sum(c for name, c in events.items() if name.startswith(("t_q", "tdg_q")))
    k = ir._factory_retry_cap(states*15756, .01, lp._downward(Fraction(1, 1200)))
    decisions = 7421 + 11*events["cx"]
    for name, count in events.items():
        if name in ir.CLIFFORD_T-{"cx", "t", "tdg"}:
            decisions += count
        elif name.startswith(("t_q", "tdg_q")):
            w = int(name.split("_q")[1])
            decisions += count*(47+33*w+4*k+int(name.startswith("tdg")))
    result = lp.controller_diagnostic(prepared, profile)
    assert result["controller_decisions_per_trial"] == decisions
    assert result["controller_decisions_all_trials"] == decisions*15756
    assert result["event_counts_per_trial"]["initial_reset"] == 31
    assert result["event_counts_per_trial"]["trial_control"] == 1
    assert result["factory_retry_cap"] == k
    assert result["required_magic_states_all_trials"] == states*15756
    assert result["maximum_allowed_controller_decisions_per_trial"] == 52889904


def test_controller_infeasibility_keeps_profiles_and_never_native_searches(prepared, monkeypatch, tmp_path):
    monkeypatch.setattr(ir, "_event_counts", lambda circuit: Counter({"initial_reset": 8, "reset": 1473, "measure": 8, "t_q7": 500000}))
    monkeypatch.setattr(ir, "estimate_instrument", lambda *args, **kwargs: pytest.fail("infeasible controller must not search"))
    for profile in lp.fixed_profiles():
        report = lp.estimate_prepared(prepared, profile=profile, cache_dir=tmp_path)
        assert report["status"] == "infeasible_declared_controller_model"
        assert report["pareto_points"] == report["all_feasible_points"] == []
        assert report["native_estimation_performed"] is False
        assert report["controller_model_diagnostic"]["budget_exceeded"] is True
        proposal = report["matched_task_implementation_proposal"]
        assert proposal["planned_initial_layouts"] == [[(j+k)%8 for j in range(8)] for k in range(8)]
        assert proposal["planned_candidate_native_attempts_total_upper_bound"] == 8*23632
        primary = proposal["primary_block_porting_recipe"]
        assert primary["extra_neighbor_CX_per_rewritten_block"].startswith("6*w")
        assert primary["free_initialization_or_quantum_occurrence_pruning"] is False
        assert primary["implementation_executed"] is primary["feasibility_guaranteed"] is False
        assert proposal["proposed_implementation_executed"] is proposal["feasibility_claimed"] is False
        assert proposal["controller_decisions_per_trial_reduction_required_at_least"] > 0


def test_native_adapter_uses_fresh_validated_view_and_original_budgets(prepared, monkeypatch, tmp_path):
    original = deepcopy(prepared.synthesized.report)
    calls = []
    def native(synthesized, count, **kwargs):
        calls.append((synthesized, count, kwargs))
        return {"status": "computed_conditional_instrument_surface_code_model", "pareto_points": [], "all_feasible_points": []}
    monkeypatch.setattr(ir, "estimate_instrument", native)
    result = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[1], cache_dir=tmp_path)
    assert calls[0][0] is not prepared.synthesized
    assert calls[0][0].circuit is prepared.synthesized.circuit and calls[0][1] == 15756
    assert calls[0][0].report["limits"] == asdict(ir.InstrumentLimits(max_qubits=8, source_operations=20000, synthesized_operations=1000000, validation_operations=1000000))
    assert prepared.synthesized.report == original and "limits" not in original
    assert calls[0][2]["physical_failure_budget"] == calls[0][2]["factory_availability_budget"] == lp._downward(Fraction(1, 1200))
    assert calls[0][2]["include_all_feasible_points"] is True
    assert result["native_estimation_performed"] is True


def test_native_distance_failure_is_retained(prepared, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise RuntimeError("No feasible point in the declared distance/factory/scenario domain")
    monkeypatch.setattr(ir, "estimate_instrument", fail)
    report = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[0], cache_dir=tmp_path)
    assert report["status"] == "infeasible_declared_distance_factory_domain"
    assert report["physical_failure_budget"] == lp._downward(Fraction(1, 1200))


def test_actual_over_legacy_operation_domain_never_enters_frozen_estimator(prepared, monkeypatch, tmp_path):
    # Bounded actual million-instruction materialization, not a forged limits
    # report or mocked length. The data only exercises the domain handoff.
    circuit = prepared.synthesized.circuit
    extra = QuantumCircuit(8)
    extra.h(0)
    item = extra.data[0]
    copied = CircuitInstruction(item.operation, (circuit.qubits[0],), ())
    before = list(circuit.data[:-8])
    final = list(circuit.data[-8:])
    circuit.data = before + [copied]*(1_000_001-len(before)-len(final)) + final
    prepared.synthesized.report["clifford_t_counts"] = dict(circuit.count_ops())
    assert len(circuit.data) == 1_000_001
    # Seal a synthetic trust-boundary fixture; this is not a scientific
    # verification claim about these repeated H operations.
    prepared.preparation_fingerprint = lp._preparation_fingerprint(prepared)
    monkeypatch.setattr(ir, "estimate_instrument", lambda *args, **kwargs: pytest.fail("over-domain CT must not be relabeled"))
    original = deepcopy(prepared.synthesized.report)
    report = lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[0], cache_dir=tmp_path)
    assert report["status"] == "unsupported_frozen_materialization_domain"
    assert report["native_estimation_performed"] is False
    assert prepared.synthesized.report == original and "limits" not in original
    assert report["matched_task_implementation_proposal"]["native_validation_adapter"]["planned_api"] == "estimate_verified_four"


def test_changed_controller_scenario_is_not_substituted(prepared, tmp_path):
    with pytest.raises(ValueError, match="two frozen"):
        lp.estimate_prepared(prepared, profile=ir.InstrumentProfile(controller_failure_per_decision=1e-18), cache_dir=tmp_path)


def test_prior_n2_n3_identity_and_no_legacy_native_reruns(monkeypatch):
    pairs = (("lindblad-physical", "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"),
             ("lindblad-three", "9dde563b53b186272edb724de1b61865d497a2d00ea3368e05c8162b01d837f7"))
    for directory, expected in pairs:
        assert hashlib.sha256((ROOT/f"assets/generated/simulation/{directory}/manifest.json").read_bytes()).hexdigest() == expected
    monkeypatch.setattr(ir, "synthesize_instrument", lambda *args, **kwargs: pytest.fail("family must not synthesize old cases"))
    monkeypatch.setattr(lp.cc, "synthesize_literal_four", lambda *args, **kwargs: pytest.fail("family status must not synthesize new case"))
    records = lp.family_status_records(check_rejections=False)
    assert records["2"]["regenerated"] is records["3"]["regenerated"] is False
    assert records["2"]["status"] == "preserved_prior_n2_result"
    assert records["3"]["status"] == "preserved_prior_n3_result"
    assert records["4"]["status"] == "ensemble_rejection_not_rechecked"
    assert records["4"]["legacy_synthesis_performed"] is records["4"]["legacy_physical_estimation_performed"] is False


def test_historical_six_width_rejection_is_kept_without_new_density_evolution(monkeypatch):
    calls = []
    validate = lp.legacy_di.validate_density_source
    def checked(source, *, limits):
        calls.append((source.num_qubits, len(source.data), limits.max_qubits))
        return validate(source, limits=limits)
    monkeypatch.setattr(lp.legacy_di, "validate_density_source", checked)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("legacy width diagnostic must not evolve density"))
    monkeypatch.setattr(ir, "simulate_scored_trial", lambda *args, **kwargs: pytest.fail("this diagnostic excludes an ensemble rerun"))
    record = lp.family_status_records(check_rejections=False)["4"]
    assert calls == [(8, 9934, 6)]
    assert record["density_checker_status"] == "density_width_cap_exceeded"
    assert record["density_width_rejection_reproduced"] is True
    assert record["density_required_wires"] == 8 and record["density_supported_wires"] == 6
    assert record["physical_status"] == "historical_unimplemented_reserved_allocation_preserved"


@pytest.mark.parametrize("field", ["routed", "circuit"])
def test_duplicate_ignored_readout_target_rejects_before_density(field, monkeypatch):
    synth = stub_synthesized()
    circuit = getattr(synth, field)
    target0 = next(item.qubits[0] for item in circuit.data if item.operation.name == "measure" and circuit.find_bit(item.clbits[0]).index == 0)
    index = next(i for i, item in enumerate(circuit.data) if item.operation.name == "measure" and circuit.find_bit(item.clbits[0]).index == 4)
    item = circuit.data[index]
    circuit.data[index] = CircuitInstruction(item.operation, (target0,), item.clbits)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("bad readout must reject before simulation"))
    with pytest.raises(ValueError, match="all eight physical wires exactly once"):
        lp.check_synthesized(synth)


@pytest.mark.parametrize("changed", ["phase", "source_angle", "contract", "workload", "report", "check", "unverified"])
def test_preparation_identity_rejects_mutation_before_controller_or_native(prepared, changed, monkeypatch, tmp_path):
    if changed == "phase":
        prepared.synthesized.circuit.global_phase += .01
    elif changed == "source_angle":
        index = next(i for i, item in enumerate(prepared.synthesized.source.data) if item.operation.name == "rx")
        item = prepared.synthesized.source.data[index]
        operation = item.operation.to_mutable()
        operation.params[0] += .0001
        prepared.synthesized.source.data[index] = CircuitInstruction(operation, item.qubits, item.clbits)
    elif changed == "contract":
        prepared.contract["target_absolute_error"] = .1
    elif changed == "workload":
        prepared.workload["complete_trials"] = 10
    elif changed == "report":
        prepared.synthesized.report["actual_synthesis_sequences_generated"] = False
    elif changed == "check":
        prepared.synthesized_check["mean"] += .001
    else:
        prepared.preparation_fingerprint = ""
    monkeypatch.setattr(lp, "controller_diagnostic", lambda *args, **kwargs: pytest.fail("mutated checkpoint must fail before accounting"))
    monkeypatch.setattr(ir, "estimate_instrument", lambda *args, **kwargs: pytest.fail("mutated checkpoint must fail before native"))
    with pytest.raises(ValueError, match="verified unmodified"):
        lp.estimate_prepared(prepared, profile=lp.fixed_profiles()[0], cache_dir=tmp_path)


def test_runner_execution_counter_attestation_does_not_change_scientific_identity(prepared):
    before = prepared.preparation_fingerprint
    prepared.synthesized.report["native_rotation_synthesis_attempts"] = 8
    prepared.synthesized.report["native_rotation_synthesis_attempts_upper_bound"] = 23632
    lp._require_verified_preparation(prepared)
    assert lp._preparation_fingerprint(prepared) == before


def test_task_subclass_cannot_override_the_literal_input_contract():
    class DifferentRates(lp.lw.ChainTask):
        @property
        def rates(self):
            return (.1, .1, .1, .1)
    with pytest.raises(ValueError, match="Only literal"):
        lp.build_source(DifferentRates(4))


def test_profile_subclass_cannot_override_fixed_model_methods(prepared, tmp_path):
    class DifferentPhysics(ir.InstrumentProfile):
        def patch_failure_per_cycle(self, distance):
            return 0.
    with pytest.raises(ValueError, match="two frozen"):
        lp.estimate_prepared(prepared, profile=DifferentPhysics(physical_error_rate=1e-3), cache_dir=tmp_path)


def test_shared_synthesized_type_does_not_impersonate_the_new_domain(monkeypatch):
    new = stub_synthesized()
    old = ir.SynthesizedInstrument(new.source, new.routed, new.circuit, new.report)
    monkeypatch.setattr(lp.di, "simulate_density_trial", lambda *args, **kwargs: pytest.fail("wrong domain must fail before simulation"))
    with pytest.raises(ValueError, match="separate-domain"):
        lp.check_synthesized(old)


@pytest.mark.parametrize("wire", [0, 2, 7])
def test_planned_porting_primitive_identity_retains_all_correlations(wire):
    # Check only the small block identity, not a future full-workload generator
    # or conditional physical feasibility. S maps wire0 to wirew; the actual
    # chronological circuit starts with its inverse to move data to wire0.
    direct = QuantumCircuit(8)
    direct.h(wire)
    direct.t(wire)
    direct.s(wire)
    ported = QuantumCircuit(8)
    def adjacent_swap(a, b):
        ported.cx(a, b)
        ported.cx(b, a)
        ported.cx(a, b)
    for q in range(wire-1, -1, -1):
        adjacent_swap(q, q+1)
    ported.h(0)
    ported.t(0)
    ported.s(0)
    for q in range(wire):
        adjacent_swap(q, q+1)
    assert ported.count_ops().get("cx", 0) == 6*wire
    assert ported.count_ops()["t"] == direct.count_ops()["t"] == 1
    assert np.linalg.norm(Operator(ported).data-Operator(direct).data, ord="nuc") < 1e-10
