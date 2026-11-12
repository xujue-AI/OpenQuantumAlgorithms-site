"""Independent repetition/metric checks for the bounded Grover scaling extension."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("grover_scaling", ROOT / "scripts/generate_grover_scaling.py")
scaling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scaling)


@pytest.mark.parametrize("success,allowance", [(1., .005), (.9453125, .005), (.6, .001), (.01, .005)])
def test_cap_is_minimal_and_expected_stopping_matches_survival_sum(success, allowance):
    contract = scaling.trial_contract(success, 0., allowance, .005)
    cap = contract["trial_cap"]
    assert (1-success)**cap <= allowance + 1e-14
    if cap > 1:
        assert (1-success)**(cap-1) > allowance
    independent_expectation = sum((1-success)**j for j in range(cap))
    assert contract["ideal_expected_stopped_trials"] == pytest.approx(independent_expectation)
    assert contract["physical_failure_allowance_per_trial"] * cap == pytest.approx(.005)


def test_original_three_qubit_contract_and_mean_are_preserved():
    contract = scaling.trial_contract(121/128, 0, .005, .005)
    assert contract["trial_cap"] == 2
    assert contract["ideal_expected_stopped_trials"] == pytest.approx(135/128)
    assert contract["configured_conditional_miss_probability"] == pytest.approx((7/128)**2)
    assert contract["physical_failure_allowance_per_trial"] == .0025


def test_synthesis_changes_success_without_double_counting_in_failure():
    contract = scaling.trial_contract(.96, .001, .005, .005)
    assert contract["configured_conditional_success_lower_per_trial"] == pytest.approx(.959)
    assert contract["configured_conditional_full_failure"] == pytest.approx(
        (.041)**contract["trial_cap"] + .005
    )
    assert contract["configured_conditional_expected_stopped_trials_upper"] >= contract["ideal_expected_stopped_trials"]
    assert contract["certified_confidence_lower_bound"] is None


@pytest.mark.parametrize("ideal,synthesis,max_trials", [(.001,0,1), (.5,.5,10), (0,0,10), (.5,-1,10), (1e-320,0,10_000)])
def test_unachievable_or_invalid_contract_is_not_silently_capped(ideal, synthesis, max_trials):
    with pytest.raises(ValueError):
        scaling.trial_contract(ideal, synthesis, .005, .005, max_trials=max_trials)


@pytest.mark.parametrize("n", range(2,9))
def test_checked_circuit_contains_complete_readout_and_expected_family(n):
    _, _, case = scaling.build_logical_case(n)
    target = int("".join("1" if j%2==0 else "0" for j in reversed(range(n))), 2)
    assert case["problem"]["marked"] == target
    assert case["problem"]["iterations"] == math.floor(math.pi/(4*math.asin(1/math.sqrt(2**n))))
    assert case["oracle_calls_per_trial"] == case["problem"]["iterations"]
    assert case["output_measurements_per_trial"] == n
    assert case["canonical_gate_counts"]["MEAS_Z"] == n
    assert case["statevector_checked"] and case["lowered_state_checked_up_to_global_phase"]
    assert case["classical_verification_time_s"] is None


def test_serial_peak_and_time_proxy_are_distinct_from_stopped_expectation():
    contract = scaling.trial_contract(121/128, 0, .005, .005)
    point = {"peak_physical_qubits":100, "runtime_per_execution_ns":10,
             "physical_failure_union_bound_all_executions":.004}
    value = scaling.add_execution_metrics(point, contract, 0)
    assert value["peak_physical_qubits"] == 100
    assert value["capped_quantum_model_runtime_ns"] == 20
    assert value["ideal_expected_quantum_model_runtime_ns"] == pytest.approx(10*135/128)
    assert value["peak_times_capped_runtime_qubit_ns"] == 2000
    assert value["conditional_full_failure_model"] == pytest.approx((7/128)**2+.004)
    assert value["certified_confidence_lower_bound"] is None


@pytest.mark.parametrize("n", [1,9,True,2.5])
def test_materialization_bounds_are_explicit(n):
    with pytest.raises(ValueError):
        scaling.equality_target(n)


def test_generated_reports_and_frozen_baseline_integrity():
    path = ROOT / "assets/generated/grover-scaling/grover-scaling.json"
    if not path.exists():
        pytest.skip("Generate the actual physical artifact before artifact-integrity checks")
    report = json.loads(path.read_text())
    assert report["status"] == "computed_conditional_model"
    assert report["sizes"] == list(range(2,9))
    assert report["frozen_files_unchanged"]
    assert report["frozen_file_sha256_before"] == report["frozen_file_sha256_after"]
    for name,digest in report["frozen_file_sha256_before"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    for case in report["cases"]:
        logical = case["logical"]
        n = logical["problem"]["n_qubits"]
        qasm = ROOT / f"assets/generated/grover-scaling/n{n}/circuit.qasm"
        assert hashlib.sha256(qasm.read_bytes()).hexdigest() == logical["qasm_sha256"]
        if n==3:
            assert logical["matches_frozen_n3_qasm"]
            assert logical["canonical_gate_counts"] == {"T":28,"MEAS_Z":3}
            assert case["trial_contract"]["trial_cap"] == 2
        for model in case["profiles"]:
            assert model["status"] == "computed_conditional_model"
            assert model["profile"]["gate_time_ns"] == 100
            assert model["profile"]["measurement_time_ns"] == 500
            assert model["profile"]["physical_error_rate"] in [.001,.0001]
            estimate=model["estimate"]
            assert not estimate["isa_graph_pruning"]
            assert not estimate["synthesis"]["actual_synthesis_sequence_generated"]
            assert not estimate["synthesis"]["instance_approximation_certificate_verified"]
            for point in estimate["pareto_points"]:
                assert point["conditional_full_failure_model"] <= .01 + 1e-12
                assert point["certified_confidence_lower_bound"] is None
                cap=case["trial_contract"]["trial_cap"]
                assert point["capped_quantum_model_runtime_ns"] == cap*point["runtime_per_execution_ns"]
                assert point["peak_times_capped_runtime_qubit_ns"] == point["peak_physical_qubits"]*point["capped_quantum_model_runtime_ns"]


@pytest.mark.parametrize("relative", ["assets/generated/resources", "assets/generated/resources/scaling", "assets/generated/grover"])
def test_scaling_output_cannot_overwrite_frozen_baseline_assets(relative):
    with pytest.raises(ValueError, match="overlap frozen"):
        scaling.guard_output_directory(ROOT / relative)
