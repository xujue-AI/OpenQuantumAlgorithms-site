"""Independent checks for signs, bit ordering, grouping, and hybrid run records."""

import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

from openquantum.optimization import (
    CYCLE_EDGES,
    build_qaoa_circuit,
    build_vqe_circuit,
    compile_rotation_basis,
    logical_run_resources,
    maxcut_values,
    pattern_search,
    qaoa_expectation,
    run_demo,
    sample_qaoa,
    sample_vqe,
    vqe_expectation,
    vqe_hamiltonian,
)


def dense_qaoa_reference(n_qubits, edges, gammas, betas):
    """Exponentiate diagonal classical costs and a tensor-product mixer.

    This reference neither constructs a Qiskit circuit nor calls the project's
    cost enumeration. Integer bits independently define the classical input.
    """
    costs = np.asarray([
        sum(((index >> a) & 1) != ((index >> b) & 1) for a, b in edges)
        for index in range(2**n_qubits)
    ])
    state = np.full(2**n_qubits, 2.0 ** (-n_qubits / 2), dtype=complex)
    pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
    for gamma, beta in zip(gammas, betas, strict=True):
        state *= np.exp(-1j * gamma * costs)
        one_qubit_mixer = np.eye(2) * math.cos(beta) - 1j * math.sin(beta) * pauli_x
        mixer = np.array([[1]], dtype=complex)
        for _ in range(n_qubits):
            mixer = np.kron(mixer, one_qubit_mixer)
        state = mixer @ state
    return state, costs


@pytest.mark.parametrize("gamma,beta", [(0, 0), (0.2, 0.1), (-0.43, 0.29), (math.pi / 4, math.pi / 8)])
def test_qaoa_cycle_mean_matches_closed_form(gamma, beta):
    expected = 2 + math.sin(4 * beta) * math.sin(2 * gamma)
    assert qaoa_expectation(4, CYCLE_EDGES, [gamma], [beta]) == pytest.approx(expected, abs=1e-12)


def test_qaoa_cycle_best_cut_probability_is_not_its_mean_approximation_ratio():
    state = Statevector.from_instruction(build_qaoa_circuit(4, CYCLE_EDGES, [math.pi / 4], [math.pi / 8]))
    probabilities = state.probabilities()
    # The two alternating cuts are 0101 and 1010, and each costs four.
    assert probabilities[5] == pytest.approx(17 / 64, abs=1e-12)
    assert probabilities[10] == pytest.approx(17 / 64, abs=1e-12)
    assert probabilities[5] + probabilities[10] == pytest.approx(17 / 32, abs=1e-12)
    costs = maxcut_values(4, CYCLE_EDGES)
    assert probabilities[costs == 0].sum() == pytest.approx(1 / 32, abs=1e-12)
    assert probabilities[costs == 2].sum() == pytest.approx(14 / 32, abs=1e-12)
    assert probabilities @ costs == pytest.approx(3, abs=1e-12)
    assert costs.max() == 4


@pytest.mark.parametrize("n_qubits,edges", [(3, ((0, 1), (0, 2))), (4, ((0, 1), (1, 2), (1, 3)))])
def test_two_qaoa_layers_match_independent_dense_evolution(n_qubits, edges):
    gammas = [0.31, -0.72]
    betas = [-0.19, 0.42]
    reference, costs = dense_qaoa_reference(n_qubits, edges, gammas, betas)
    state = Statevector.from_instruction(build_qaoa_circuit(n_qubits, edges, gammas, betas))
    assert state.equiv(Statevector(reference), atol=1e-12)
    assert qaoa_expectation(n_qubits, edges, gammas, betas) == pytest.approx(
        np.abs(reference) ** 2 @ costs, abs=1e-12
    )


def test_classical_enumeration_uses_qubit_zero_as_the_lowest_integer_bit():
    # Only q0 and q2 matter: flipping q1 must leave the cut unchanged.
    np.testing.assert_array_equal(maxcut_values(3, [(0, 2)]), [0, 1, 0, 1, 1, 0, 1, 0])


@pytest.mark.parametrize("theta", [0.0, 0.6, -1.1, 2.2, -math.atan(0.25)])
def test_vqe_energy_matches_analytic_expression(theta):
    assert vqe_expectation(theta) == pytest.approx(-2 * math.cos(theta) + 0.5 * math.sin(theta), abs=1e-12)
    state = Statevector.from_instruction(build_vqe_circuit(theta)).data
    expected_state = np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])
    np.testing.assert_allclose(state, expected_state, atol=1e-12)


def test_vqe_matrix_and_ground_state_are_independently_diagonalized():
    # In q1 q0 order the even and odd parity blocks are visible by inspection.
    expected = np.array([[-2, 0, 0, 0.5], [0, 0, 0.5, 0], [0, 0.5, 0, 0], [0.5, 0, 0, 2]])
    np.testing.assert_allclose(vqe_hamiltonian(), expected, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(expected)
    np.testing.assert_allclose(eigenvalues, [-math.sqrt(4.25), -0.5, 0.5, math.sqrt(4.25)], atol=1e-12)
    assert vqe_expectation(-math.atan(0.25)) == pytest.approx(eigenvalues[0], abs=1e-12)


@pytest.mark.parametrize("theta", [-0.4, 0.6, 1.7])
def test_vqe_grouped_measurement_probabilities_and_energy(theta):
    z_probabilities = Statevector.from_instruction(build_vqe_circuit(theta, basis="z")).probabilities()
    x_probabilities = Statevector.from_instruction(build_vqe_circuit(theta, basis="x")).probabilities()
    np.testing.assert_allclose(z_probabilities, [math.cos(theta / 2) ** 2, 0, 0, math.sin(theta / 2) ** 2], atol=1e-12)
    np.testing.assert_allclose(x_probabilities, np.array([1, -1, -1, 1]) * math.sin(theta) / 4 + 0.25, atol=1e-12)
    # Z0 and Z1 share outcomes and are evaluated together, retaining covariance.
    z_group = np.array([-2, 0, 0, 2])
    x_group = np.array([0.5, -0.5, -0.5, 0.5])
    measured_energy = z_probabilities @ z_group + x_probabilities @ x_group
    assert measured_energy == pytest.approx(vqe_expectation(theta), abs=1e-12)


@pytest.mark.parametrize("circuit", [
    build_qaoa_circuit(4, CYCLE_EDGES, [0.31, -0.72], [-0.19, 0.42]),
    build_vqe_circuit(-0.4, basis="x"),
])
def test_explicit_compilation_and_qasm_preserve_unitary(circuit):
    compiled = compile_rotation_basis(circuit)
    assert Operator(circuit).equiv(Operator(compiled), atol=1e-12)
    assert set(compiled.count_ops()) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"}
    measured = compiled.copy()
    measured.measure_all()
    text = qasm3.dumps(measured)
    assert "OPENQASM 3.0" in text
    assert "measure" in text


def test_measured_circuits_include_readout_of_every_qubit():
    assert build_qaoa_circuit(4, CYCLE_EDGES, [0.2], [0.1], measure=True).count_ops()["measure"] == 4
    assert build_vqe_circuit(0.6, basis="x", measure=True).count_ops()["measure"] == 2


def test_circuit_generation_is_separate_from_bounded_simulation():
    circuit = build_qaoa_circuit(13, [(0, 12)], [0.2], [0.1])
    assert circuit.num_qubits == 13
    with pytest.raises(ValueError):
        qaoa_expectation(13, [(0, 12)], [0.2], [0.1], max_qubits=12)
    with pytest.raises(ValueError):
        maxcut_values(17, [(0, 16)], max_qubits=16)


@pytest.mark.parametrize("n_qubits,edges", [(0, []), (True, []), (2, [(0, 0)]), (2, [(0, 2)]), (2, [(0, 1), (1, 0)]), (2, [(False, 1)])])
def test_invalid_qaoa_graph_is_rejected(n_qubits, edges):
    with pytest.raises(ValueError):
        build_qaoa_circuit(n_qubits, edges, [0.2], [0.1])


@pytest.mark.parametrize("gammas,betas", [([0.2], []), ([math.nan], [0.1]), ([0.2], [math.inf])])
def test_invalid_qaoa_angles_are_rejected(gammas, betas):
    with pytest.raises(ValueError):
        build_qaoa_circuit(4, CYCLE_EDGES, gammas, betas)


@pytest.mark.parametrize("theta,basis", [(math.nan, None), (math.inf, None), (0.2, "y")])
def test_invalid_vqe_request_is_rejected(theta, basis):
    with pytest.raises(ValueError):
        build_vqe_circuit(theta, basis=basis)


def test_optimizer_records_every_actual_objective_call_and_reaches_vqe_ground_state():
    calls = []

    def objective(parameters):
        parameters = np.asarray(parameters).copy()
        value = -2 * math.cos(parameters[0]) + 0.5 * math.sin(parameters[0])
        calls.append((parameters, value))
        return value

    result = pattern_search(objective, [0.6])
    assert result["value"] - (-math.sqrt(4.25)) < 1e-9
    assert len(result["evaluations"]) == len(calls)
    for recorded, (parameters, value) in zip(result["evaluations"], calls, strict=True):
        np.testing.assert_allclose(recorded["parameters"], parameters, atol=1e-12)
        assert recorded["value"] == pytest.approx(value, abs=1e-12)
        assert recorded["seconds"] >= 0
    assert result["value"] == pytest.approx(min(value for _, value in calls), abs=1e-12)


def test_optimizer_is_deterministic_and_respects_iteration_bound():
    objective = lambda parameters: -math.cos(parameters[0])
    first = pattern_search(objective, [0.6], max_iterations=3)
    second = pattern_search(objective, [0.6], max_iterations=3)
    assert len(first["iterations"]) <= 3
    assert first["parameters"] == pytest.approx(second["parameters"], abs=0)
    assert [entry["value"] for entry in first["evaluations"]] == pytest.approx(
        [entry["value"] for entry in second["evaluations"]], abs=0
    )


def test_qaoa_sampling_records_joint_edge_outcomes_and_all_shots():
    result = sample_qaoa(math.pi / 4, math.pi / 8, shots=2048, seed=19)
    repeated = sample_qaoa(math.pi / 4, math.pi / 8, shots=2048, seed=19)
    assert result["counts"] == repeated["counts"]
    assert sum(result["counts"].values()) == result["shots"] == 2048
    samples = []
    for bits, count in result["counts"].items():
        bit_assignment = [int(bit) for bit in bits[::-1]]
        cut = sum(bit_assignment[a] != bit_assignment[b] for a, b in [(0, 1), (1, 2), (2, 3), (0, 3)])
        samples.extend([cut] * count)
    assert result["estimate"] == pytest.approx(np.mean(samples), abs=1e-12)
    assert result["sample_variance"] == pytest.approx(np.var(samples, ddof=1), abs=1e-12)
    assert result["optimal_cut_samples"] == samples.count(4)
    assert result["exact_expectation"] == pytest.approx(3, abs=1e-12)
    assert result["estimated_standard_error"] == pytest.approx(np.std(samples, ddof=1) / math.sqrt(2048), abs=1e-12)


def test_vqe_sampling_retains_covariance_and_uses_two_independent_groups():
    theta = -math.atan(0.25)
    result = sample_vqe(theta, shots=2048, seed=7)
    assert result["total_shots"] == 4096
    assert [group["group"] for group in result["groups"]] == ["all_z", "all_x"]
    assert result["groups"][0]["seed"] != result["groups"][1]["seed"]
    samples_by_group = []
    for group in result["groups"]:
        samples = []
        assert sum(group["counts"].values()) == 2048
        for bits, count in group["counts"].items():
            eigenvalues = [1 - 2 * int(bit) for bit in bits]
            value = -sum(eigenvalues) if group["group"] == "all_z" else 0.5 * math.prod(eigenvalues)
            samples.extend([value] * count)
        samples_by_group.append(samples)
        assert group["estimate"] == pytest.approx(np.mean(samples), abs=1e-12)
        assert group["sample_variance"] == pytest.approx(np.var(samples, ddof=1), abs=1e-12)
    assert result["estimate"] == pytest.approx(sum(np.mean(samples) for samples in samples_by_group), abs=1e-12)
    assert result["estimated_standard_error"] == pytest.approx(
        math.sqrt(sum(np.var(samples, ddof=1) / len(samples) for samples in samples_by_group)), abs=1e-12
    )
    assert result["exact_expectation"] == pytest.approx(-math.sqrt(4.25), abs=1e-12)


def test_one_shot_readout_has_no_fabricated_variance_estimate():
    assert sample_qaoa(0.2, 0.1, shots=1)["estimated_standard_error"] is None
    result = sample_vqe(0.6, shots=1)
    assert result["total_shots"] == 2
    assert result["estimated_standard_error"] is None
    assert all(group["sample_variance"] is None for group in result["groups"])


@pytest.fixture(scope="module")
def complete_demo_and_resources():
    demo = run_demo(shots=128, seeds=(7, 19))
    return demo, logical_run_resources(demo)


def test_full_demo_reports_all_predeclared_optimization_runs(complete_demo_and_resources):
    demo, _ = complete_demo_and_resources
    assert len(demo["qaoa"]["optimization_runs"]) == 3
    assert len(demo["vqe"]["optimization_runs"]) == 3
    assert demo["qaoa"]["classical"]["optimum"] == 4
    assert demo["qaoa"]["classical"]["optimal_bitstrings"] == ["0101", "1010"]
    assert demo["vqe"]["reference"]["ground_energy"] == pytest.approx(-math.sqrt(4.25), abs=1e-12)
    for algorithm, dimensions in (("qaoa", 2), ("vqe", 1)):
        for run in demo[algorithm]["optimization_runs"]:
            assert run["target_met"]
            assert run["shots"] == 0
            assert run["objective_evaluations"] == 1 + 2 * dimensions * run["iteration_count"]
            assert run["objective_evaluations"] == len(run["evaluations"])
            assert run["objective_wall_seconds"] == pytest.approx(sum(entry["seconds"] for entry in run["evaluations"]))
            assert run["optimizer_wall_seconds"] >= run["objective_wall_seconds"] >= 0
            assert run["stop_reason"] == "step_tolerance"
            assert run["final_step"] < run["settings"]["step_tolerance"]
            assert all(later["value"] <= earlier["value"] for earlier, later in zip(run["iterations"], run["iterations"][1:]))
            assert all(entry["shots"] == 0 for entry in run["evaluations"])
    qaoa = demo["qaoa"]["optimization_runs"][0]
    assert qaoa["achieved_expectation"] == pytest.approx(3, abs=1e-8)
    assert qaoa["approximation_ratio"] == pytest.approx(0.75, abs=1e-8)
    assert qaoa["optimal_cut_probability"] == pytest.approx(17 / 32, abs=1e-5)
    repetitions = qaoa["independent_trials_for_99_percent_one_optimum"]
    assert (1 - qaoa["optimal_cut_probability"]) ** repetitions <= 0.01
    assert (1 - qaoa["optimal_cut_probability"]) ** (repetitions - 1) > 0.01


def test_resource_ledger_accounts_for_all_objective_and_measurement_repetitions(complete_demo_and_resources):
    demo, resources = complete_demo_and_resources
    assert resources["physical_resource_estimate"] is None
    for algorithm in ("qaoa", "vqe"):
        ledger = resources["algorithms"][algorithm]
        runs = demo[algorithm]["optimization_runs"]
        number_of_evaluations = sum(run["objective_evaluations"] for run in runs)
        assert ledger["exact_objective_evaluations"] == number_of_evaluations
        assert len(ledger["exact_evaluation_inputs"]) == number_of_evaluations
        assert ledger["exact_statevector_shots"] == 0
        assert ledger["ancillas"] == 0
        assert ledger["optimizer_wall_seconds"] == pytest.approx(sum(run["optimizer_wall_seconds"] for run in runs))
        groups_per_experiment = 1 if algorithm == "qaoa" else 2
        expected_shots = 128 * 2 * groups_per_experiment
        assert ledger["held_out_measurement_circuit_shots"] == expected_shots
        assert ledger["measured_qubit_readouts"] == expected_shots * demo[algorithm]["n_qubits"]
        assert ledger["shot_weighted_logical_gate_counts"]["measure"] == ledger["measured_qubit_readouts"]
        assert sum(group["shots"] for group in ledger["readout_group_inputs"]) == expected_shots
        for gate, total in ledger["shot_weighted_logical_gate_counts"].items():
            assert total == sum(group["circuit_gate_counts"].get(gate, 0) * group["shots"] for group in ledger["readout_group_inputs"])
        for gate, total in ledger["exact_simulated_gate_counts"].items():
            assert total == sum(entry["circuit_gate_counts"].get(gate, 0) for entry in ledger["exact_evaluation_inputs"])
    # Preparation and basis-change gates are counted across every repetition.
    assert resources["algorithms"]["qaoa"]["shot_weighted_logical_gate_counts"]["h"] == 4 * 256
    assert resources["algorithms"]["vqe"]["shot_weighted_logical_gate_counts"]["h"] == 2 * 256
    assert resources["algorithms"]["vqe"]["shot_weighted_logical_gate_counts"]["ry"] == 512
