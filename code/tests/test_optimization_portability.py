"""Independent triangle and asymmetric-Hamiltonian portability checks."""

import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from openquantum import optimization as opt

TRIANGLE_EDGES = ((0, 1), (1, 2), (0, 2))
H2_COEFFICIENTS = (0.8, 0.6, 0.4)


def _triangle_dense_state(gamma, beta):
    """Use diagonal classical costs and a tensor-product matrix exponential."""
    cuts = np.array([sum(((x >> u) & 1) != ((x >> v) & 1)
                         for u, v in TRIANGLE_EDGES) for x in range(8)])
    phased_state = np.exp(-1j * gamma * cuts) / math.sqrt(8)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    rotation = math.cos(beta) * np.eye(2) - 1j * math.sin(beta) * x
    mixer = np.kron(np.kron(rotation, rotation), rotation)
    return mixer @ phased_state, cuts


@pytest.mark.parametrize("gamma,beta", [(0.2, 0.1), (0.7, -0.3), (-0.6, 0.4), (math.asin(1 / math.sqrt(3)), math.asin(1 / math.sqrt(3)) / 2)])
def test_triangle_qaoa_matches_dense_evolution_and_closed_form(gamma, beta):
    reference, cuts = _triangle_dense_state(gamma, beta)
    circuit = opt.build_qaoa_circuit(3, TRIANGLE_EDGES, [gamma], [beta])
    state = Statevector.from_instruction(circuit)
    assert state.equiv(Statevector(reference), atol=1e-12)
    expected = (1.5 + 0.75 * math.sin(4 * beta) * math.sin(2 * gamma)
                - 1.5 * math.sin(2 * beta) ** 2 * math.sin(gamma) ** 2)
    assert np.abs(reference) ** 2 @ cuts == pytest.approx(expected, abs=1e-12)
    assert opt.qaoa_expectation(3, TRIANGLE_EDGES, [gamma], [beta]) == pytest.approx(expected, abs=1e-12)


def test_triangle_depth_one_circuit_attains_the_enumerated_optimum():
    gamma = math.asin(1 / math.sqrt(3))
    state, cuts = _triangle_dense_state(gamma, gamma / 2)
    np.testing.assert_array_equal(cuts, [0, 2, 2, 2, 2, 2, 2, 0])
    np.testing.assert_array_equal(opt.maxcut_values(3, TRIANGLE_EDGES), cuts)
    assert cuts.max() == 2
    np.testing.assert_allclose(np.abs(state) ** 2, [0, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 0], atol=1e-12)
    assert opt.qaoa_expectation(3, TRIANGLE_EDGES, [gamma], [gamma / 2]) == pytest.approx(2, abs=1e-12)


@pytest.mark.parametrize("coefficients", [H2_COEFFICIENTS, (1.0, -1.0, 0.4), (-0.7, -0.2, -0.3)])
def test_asymmetric_hamiltonian_matrix_preserves_wire_coefficient_order(coefficients):
    a, b, c = coefficients
    expected = np.array([[-a - b, 0, 0, c], [0, a - b, c, 0],
                         [0, c, -a + b, 0], [c, 0, 0, a + b]])
    np.testing.assert_allclose(opt.vqe_hamiltonian(coefficients=coefficients), expected, atol=1e-12)
    eigenvalues = sorted([-math.hypot(a + b, c), math.hypot(a + b, c),
                          -math.hypot(a - b, c), math.hypot(a - b, c)])
    np.testing.assert_allclose(np.linalg.eigvalsh(expected), eigenvalues, atol=1e-12)


@pytest.mark.parametrize("theta", [-1.1, 0.6, 2.2, -math.atan2(0.4, 1.4)])
def test_h2_energy_and_full_reference_ground_state(theta):
    assert opt.vqe_expectation(theta, coefficients=H2_COEFFICIENTS) == pytest.approx(
        -1.4 * math.cos(theta) + 0.4 * math.sin(theta), abs=1e-12)
    ground = np.linalg.eigvalsh(opt.vqe_hamiltonian(coefficients=H2_COEFFICIENTS))[0]
    assert ground == pytest.approx(-math.sqrt(2.12), abs=1e-12)
    assert opt.vqe_expectation(-math.atan2(0.4, 1.4), coefficients=H2_COEFFICIENTS) == pytest.approx(ground, abs=1e-12)


def test_generic_coefficients_do_not_make_the_even_ansatz_a_global_ground_state_solver():
    coefficients = (1.0, -1.0, 0.4)
    eigenvalues = np.linalg.eigvalsh(opt.vqe_hamiltonian(coefficients=coefficients))
    assert eigenvalues[0] == pytest.approx(-math.sqrt(4.16), abs=1e-12)
    even_minimum = opt.vqe_expectation(-math.pi / 2, coefficients=coefficients)
    assert even_minimum == pytest.approx(-0.4, abs=1e-12)
    assert even_minimum > eigenvalues[0]


@pytest.mark.parametrize("coefficients", [H2_COEFFICIENTS, (1.0, -1.0, -0.4)])
def test_sampled_asymmetric_energy_reconstructs_joint_group_covariance(coefficients):
    a, b, c = coefficients
    result = opt.sampled_vqe_objective([0.6], shots=512, seed=19,
                                      evaluation_index=2, coefficients=coefficients, stream_id=3)
    means, variances = [], []
    for group in result["groups"]:
        samples = []
        for bits, count in group["counts"].items():
            z0 = 1 - 2 * int(bits[-1])
            z1 = 1 - 2 * int(bits[-2])
            value = -a * z0 - b * z1 if group["group"] == "all_z" else c * z0 * z1
            samples.extend([value] * count)
        assert len(samples) == group["shots"] == 512
        mean, variance = float(np.mean(samples)), float(np.var(samples, ddof=1))
        assert group["estimate"] == pytest.approx(mean, abs=1e-12)
        assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
        means.append(mean)
        variances.append(variance / 512)
    assert result["objective_estimate"] == pytest.approx(sum(means), abs=1e-12)
    assert result["estimated_standard_error"] == pytest.approx(math.sqrt(sum(variances)), abs=1e-12)
    assert result["total_shots"] == 1024
    if a == -b:
        assert result["groups"][0]["sample_variance"] == 0


def test_new_instances_use_distinct_rng_stream_ids_from_the_original_examples():
    request = dict(shots=16, seed=7, evaluation_index=0)
    old_q = opt.sampled_qaoa_objective([0.2, 0.1], **request)
    new_q = opt.sampled_qaoa_objective([0.2, 0.1], n_qubits=3, edges=TRIANGLE_EDGES, stream_id=2, **request)
    old_v = opt.sampled_vqe_objective([0.6], **request)
    new_v = opt.sampled_vqe_objective([0.6], coefficients=H2_COEFFICIENTS, stream_id=3, **request)
    stream_keys = [tuple(group["rng"]["stream_key"]) for call in [old_q, new_q, old_v, new_v] for group in call["groups"]]
    assert len(stream_keys) == len(set(stream_keys))
    assert new_q["groups"][0]["rng"]["entropy"] == [7, 2]
    assert all(group["rng"]["entropy"] == [7, 3] for group in new_v["groups"])


@pytest.mark.parametrize("coefficients", [(0.8, 0.6), (0.8, 0.6, 0.4, 1.0), (math.nan, 0.6, 0.4), (0.8, math.inf, 0.4)])
def test_invalid_coefficient_lists_are_rejected(coefficients):
    with pytest.raises(ValueError):
        opt.vqe_hamiltonian(coefficients=coefficients)
    with pytest.raises(ValueError):
        opt.sampled_vqe_objective([0.6], shots=16, seed=7, evaluation_index=0, coefficients=coefficients)


@pytest.fixture(scope="module")
def portability_demo():
    return opt.run_portability_demo()


def test_portability_cases_hold_initialization_fixed_and_isolate_sampling_seed(portability_demo):
    assert portability_demo["qaoa"]["classical"]["values"] == [0, 2, 2, 2, 2, 2, 2, 0]
    assert portability_demo["qaoa"]["classical"]["optimum"] == 2
    assert portability_demo["vqe"]["coefficients"] == list(H2_COEFFICIENTS)
    assert portability_demo["vqe"]["reference"]["ground_energy"] == pytest.approx(-math.sqrt(2.12), abs=1e-12)
    for algorithm, initial in (("qaoa", [0.2, 0.1]), ("vqe", [0.6])):
        runs = portability_demo[algorithm]["runs"]
        assert [run["seed"] for run in runs] == [7, 19, 41]
        assert all(run["initial"] == initial for run in runs)
        assert all(run["iteration_count"] == 32 for run in runs)
        assert all(run["stop_reason"] == "fixed_schedule_exhausted" for run in runs)
        assert all(run["convergence_certified"] is False for run in runs)


def test_second_instance_training_decisions_replay_using_sampled_observations(portability_demo):
    assert portability_demo["exact_diagnostics_used_for_optimization"] is False
    for algorithm in ("qaoa", "vqe"):
        for run in portability_demo[algorithm]["runs"]:
            for step in run["iterations"]:
                incumbent = run["evaluations"][step["incumbent_evaluation"]]
                proposals = [run["evaluations"][index] for index in step["proposal_evaluations"]]
                best = min(proposals, key=lambda call: call["objective_estimate"])
                improvement = incumbent["objective_estimate"] - best["objective_estimate"]
                margin = 2 * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
                accepted = improvement > margin
                assert step["accepted"] == accepted
                assert step["best_proposal_evaluation"] == best["index"]
                assert step["sampled_improvement"] == pytest.approx(improvement, abs=1e-12)
                assert step["comparison_margin"] == pytest.approx(margin, abs=1e-12)
                expected_parameters = best["parameters"] if accepted else incumbent["parameters"]
                np.testing.assert_allclose(step["parameters_after"], expected_parameters, atol=1e-12)


def test_portability_exact_triangle_audits_do_not_change_sampled_training(monkeypatch):
    schedule = ((0.4, 256, 2),)
    monkeypatch.setattr(opt, "triangle_expectation_reference", lambda gamma, beta: -10000.0)
    first = opt.run_portability_demo(schedule=schedule)
    monkeypatch.setattr(opt, "triangle_expectation_reference", lambda gamma, beta: 10000.0)
    second = opt.run_portability_demo(schedule=schedule)
    for first_run, second_run in zip(first["qaoa"]["runs"], second["qaoa"]["runs"], strict=True):
        assert first_run["parameters"] == second_run["parameters"]
        assert [iteration["accepted"] for iteration in first_run["iterations"]] == [iteration["accepted"] for iteration in second_run["iterations"]]
        for first_call, second_call in zip(first_run["evaluations"], second_run["evaluations"], strict=True):
            assert first_call["objective_estimate"] == second_call["objective_estimate"]
            assert [group["counts"] for group in first_call["groups"]] == [group["counts"] for group in second_call["groups"]]
            assert first_call["exact_objective_audit"] != second_call["exact_objective_audit"]


def test_portability_holdout_audits_use_instance_coefficients_and_independent_streams(portability_demo):
    stream_keys = []
    for algorithm, stream_id in (("qaoa", 2), ("vqe", 3)):
        for run in portability_demo[algorithm]["runs"]:
            for family, calls in ((0, run["evaluations"]), (1, [run["validation"]])):
                for call_index, call in enumerate(calls):
                    parameters = call["parameters"]
                    if algorithm == "qaoa":
                        gamma, beta = parameters
                        exact = -(1.5 + 0.75 * math.sin(4 * beta) * math.sin(2 * gamma)
                                  - 1.5 * math.sin(2 * beta) ** 2 * math.sin(gamma) ** 2)
                    else:
                        exact = -1.4 * math.cos(parameters[0]) + 0.4 * math.sin(parameters[0])
                    assert call["exact_objective_audit"] == pytest.approx(exact, abs=1e-12)
                    for group_index, group in enumerate(call["groups"]):
                        expected = [run["seed"], stream_id, family, call_index, group_index]
                        assert group["rng"]["stream_key"] == expected
                        stream_keys.append(tuple(expected))
            validation = run["validation"]
            assert all(group["shots"] == 4096 for group in validation["groups"])
            assert validation["parameters"] == run["parameters"]
            assert validation["role"] == "independent_final_validation"
            expected_widths = [2] if algorithm == "qaoa" else [2.8, 0.8]
            assert [group["outcome_max"] - group["outcome_min"] for group in validation["groups"]] == pytest.approx(expected_widths)
            expected_radius = math.sqrt(sum(width**2 / 4096 for width in expected_widths) * math.log(40) / 2)
            assert validation["hoeffding_95_absolute_mean_radius"] == pytest.approx(expected_radius, abs=1e-12)
            assert validation["precision_target_supported_by_bound"] == (expected_radius <= 0.10)
            assert validation["estimation_error_from_exact"] == pytest.approx(
                validation["objective_estimate"] - validation["exact_objective_audit"], abs=1e-12)
            if algorithm == "qaoa":
                assert run["exact_gap_audit"] == pytest.approx(2 - run["exact_expected_cut_audit"], abs=1e-12)
                assert run["target_met_audit"] == (run["exact_expected_cut_audit"] >= 1.90)
            else:
                assert run["exact_gap_audit"] == pytest.approx(run["exact_energy_audit"] + math.sqrt(2.12), abs=1e-12)
                assert run["target_met_audit"] == (run["exact_gap_audit"] <= 0.03)
    assert len(stream_keys) == len(set(stream_keys)) == 1074


def test_second_instance_budget_and_logical_resources_cover_every_call(portability_demo):
    assert portability_demo["training_shots"] == 1016064
    assert portability_demo["validation_shots"] == 36864
    assert portability_demo["total_shots"] == 1052928
    assert portability_demo["physical_resource_estimate"] is None
    resources = opt.sampled_run_resources(portability_demo)
    assert resources["total_shots"] == 1052928
    for algorithm, shots, groups, calls in (("qaoa", 473856, 486, 483), ("vqe", 579072, 588, 291)):
        ledger = resources["algorithms"][algorithm]
        assert ledger["total_modeled_circuit_shots"] == shots
        assert ledger["measured_group_batches"] == groups
        assert ledger["objective_calls"] == calls
        assert ledger["independent_validation_calls"] == 3
        assert sum(entry["shots"] for entry in ledger["execution_inputs"]) == shots
        for gate, total in ledger["shot_weighted_gate_counts"].items():
            assert total == sum(entry["circuit_gate_counts"].get(gate, 0) * entry["shots"] for entry in ledger["execution_inputs"])
    qaoa = resources["algorithms"]["qaoa"]
    assert qaoa["total_modeled_qubit_readouts"] == 3 * 473856
    assert qaoa["shot_weighted_gate_counts"]["h"] == 3 * 473856
    assert qaoa["shot_weighted_gate_counts"]["cx"] == 6 * 473856
    vqe = resources["algorithms"]["vqe"]
    assert vqe["total_modeled_qubit_readouts"] == 2 * 579072
    assert vqe["shot_weighted_gate_counts"]["h"] == 579072


@pytest.mark.parametrize("stream_id", [-1, True])
def test_invalid_instance_stream_ids_are_rejected(stream_id):
    request = dict(shots=16, seed=7, evaluation_index=0, stream_id=stream_id)
    with pytest.raises(ValueError):
        opt.sampled_qaoa_objective([0.2, 0.1], n_qubits=3, edges=TRIANGLE_EDGES, **request)
    with pytest.raises(ValueError):
        opt.sampled_vqe_objective([0.6], coefficients=H2_COEFFICIENTS, **request)
