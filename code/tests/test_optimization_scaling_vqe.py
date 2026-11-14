"""Independent multi-spin ansatz, Hamiltonian, and grouped-readout checks."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from openquantum import optimization as opt


def _tensor_operator(n_qubits, operators):
    """Build matrices in |q[n-1]...q[0]> order without circuit utilities."""
    result = np.array([[1.0]])
    for qubit in reversed(range(n_qubits)):
        result = np.kron(result, operators.get(qubit, np.eye(2)))
    return result


def _independent_hamiltonian(n_qubits):
    z = np.diag([1.0, -1.0])
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = np.zeros((2**n_qubits, 2**n_qubits))
    for qubit in range(n_qubits):
        hamiltonian -= _tensor_operator(n_qubits, {qubit: z})
    for qubit in range(n_qubits - 1):
        hamiltonian += 0.5 * _tensor_operator(n_qubits, {qubit: x, qubit + 1: x})
    return hamiltonian


def _independent_ansatz(parameters):
    """Tensor RY states, then apply the classical CNOT bit permutations."""
    state = np.array([1.0])
    for theta in reversed(parameters):
        state = np.kron(state, [math.cos(theta / 2), math.sin(theta / 2)])
    state = np.asarray(state, dtype=complex)
    for control in range(len(parameters) - 1):
        result = np.empty_like(state)
        for index, amplitude in enumerate(state):
            destination = index ^ (1 << (control + 1)) if (index >> control) & 1 else index
            result[destination] = amplitude
        state = result
    return state


def _group_score(bits, group):
    eigenvalues = [1 - 2 * int(bit) for bit in bits[::-1]]
    if group == "all_z":
        return -sum(eigenvalues)
    return 0.5 * sum(eigenvalues[qubit] * eigenvalues[qubit + 1]
                     for qubit in range(len(eigenvalues) - 1))


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 6])
def test_spin_hamiltonian_matches_independent_pauli_tensor_products(n_qubits):
    expected = _independent_hamiltonian(n_qubits)
    actual = opt.spin_vqe_hamiltonian(n_qubits)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual.shape == (2**n_qubits, 2**n_qubits)
    np.testing.assert_allclose(actual, actual.T.conj(), atol=1e-12)
    # Diagonal scores and pair-flip edges expose coefficient/sign/bit mistakes.
    for index in range(2**n_qubits):
        assert actual[index, index] == pytest.approx(-n_qubits + 2 * index.bit_count())
        for qubit in range(n_qubits - 1):
            assert actual[index ^ (3 << qubit), index] == pytest.approx(0.5)


@pytest.mark.parametrize("parameters", [[0.21, -0.47], [0.21, -0.47, 0.62], [0.21, -0.47, 0.62, -0.32], [0.21, -0.47, 0.62, -0.32, 0.11, 0.84]])
def test_spin_ansatz_matches_independent_tensor_and_ordered_bit_permutations(parameters):
    n_qubits = len(parameters)
    expected = _independent_ansatz(parameters)
    actual = Statevector.from_instruction(opt.build_spin_vqe_circuit(n_qubits, parameters)).data
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.linalg.norm(actual) == pytest.approx(1, abs=1e-12)
    h = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    expected_x = _tensor_operator(n_qubits, {qubit: h for qubit in range(n_qubits)}) @ expected
    actual_x = Statevector.from_instruction(opt.build_spin_vqe_circuit(n_qubits, parameters, basis="x")).data
    np.testing.assert_allclose(actual_x, expected_x, atol=1e-12)


@pytest.mark.parametrize("parameters", [[0.21, -0.47], [0.21, -0.47, 0.62], [0.21, -0.47, 0.62, -0.32], [0.21, -0.47, 0.62, -0.32, 0.11, 0.84]])
def test_spin_reference_energy_ground_and_scalar_formula_are_independent(parameters):
    n_qubits = len(parameters)
    hamiltonian = _independent_hamiltonian(n_qubits)
    state = _independent_ansatz(parameters)
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    energy = float(np.vdot(state, hamiltonian @ state).real)
    reference = opt.spin_vqe_reference(n_qubits, parameters)
    np.testing.assert_allclose(reference["reference_state"], state, atol=1e-12)
    np.testing.assert_allclose(reference["eigenvalues"], eigenvalues, atol=1e-12)
    assert reference["ground_energy"] == pytest.approx(eigenvalues[0], abs=1e-12)
    assert reference["exact_energy"] == pytest.approx(energy, abs=1e-12)
    assert reference["energy_density_gap"] == pytest.approx((energy - eigenvalues[0]) / n_qubits, abs=1e-12)
    assert reference["scalar_energy"] == pytest.approx(energy, abs=1e-12)
    assert opt.spin_vqe_energy_formula(n_qubits, parameters) == pytest.approx(energy, abs=1e-12)
    assert reference["norm_error"] < 1e-12


def test_two_spin_subcase_recovers_the_original_even_parity_energy():
    theta = -0.3
    state = _independent_ansatz([theta, 0.0])
    np.testing.assert_allclose(state, [math.cos(theta / 2), 0, 0, math.sin(theta / 2)], atol=1e-12)
    energy = np.vdot(state, opt.spin_vqe_hamiltonian(2) @ state).real
    assert energy == pytest.approx(-2 * math.cos(theta) + 0.5 * math.sin(theta), abs=1e-12)


def test_zero_product_energy_does_not_replace_the_full_ground_reference():
    """A simple representable state can miss the prescribed density target."""
    n_qubits = 6
    reference = opt.spin_vqe_reference(n_qubits, [0.0] * n_qubits)
    independent_ground = np.linalg.eigvalsh(_independent_hamiltonian(n_qubits))[0]
    assert reference["exact_energy"] == pytest.approx(-n_qubits, abs=1e-12)
    assert reference["ground_energy"] == pytest.approx(independent_ground, abs=1e-12)
    assert reference["ground_energy"] < reference["exact_energy"]
    assert reference["energy_density_gap"] > 0.05


def test_postrun_reference_never_calls_the_circuit_or_statevector_backend(monkeypatch):
    parameters = [0.21, -0.47, 0.62, -0.32]
    expected_state = _independent_ansatz(parameters)
    expected_energy = np.vdot(expected_state, _independent_hamiltonian(4) @ expected_state).real

    def forbidden(*args, **kwargs):
        raise AssertionError("the post-run reference must be independent of circuit construction")

    monkeypatch.setattr(opt, "build_spin_vqe_circuit", forbidden)
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    reference = opt.spin_vqe_reference(4, parameters)
    np.testing.assert_allclose(reference["reference_state"], expected_state, atol=1e-12)
    assert reference["exact_energy"] == pytest.approx(expected_energy, abs=1e-12)


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 6])
def test_measured_ansatz_counts_every_preparation_basis_change_and_readout(n_qubits):
    parameters = [0.6] * n_qubits
    z = opt.build_spin_vqe_circuit(n_qubits, parameters, basis="z", measure=True)
    x = opt.build_spin_vqe_circuit(n_qubits, parameters, basis="x", measure=True)
    for circuit in (z, x):
        assert circuit.num_qubits == n_qubits
        assert circuit.count_ops()["ry"] == n_qubits
        assert circuit.count_ops()["cx"] == n_qubits - 1
        assert circuit.count_ops()["measure"] == n_qubits
    assert z.count_ops().get("h", 0) == 0
    assert x.count_ops()["h"] == n_qubits


@pytest.mark.parametrize("parameters", [[0.21, -0.47], [0.21, -0.47, 0.62], [0.21, -0.47, 0.62, -0.32, 0.11, 0.84]])
def test_sampled_spin_objective_reconstructs_joint_scores_and_variance(parameters):
    n_qubits = len(parameters)
    result = opt.sampled_spin_vqe_objective(parameters, n_qubits=n_qubits,
                                           shots=512, seed=19, evaluation_index=3,
                                           family=0, stream_id=63)
    assert [group["group"] for group in result["groups"]] == ["all_z", "all_x"]
    means, variance_of_means = [], []
    for group_index, group in enumerate(result["groups"]):
        samples = []
        for bits, count in group["counts"].items():
            assert len(bits) == n_qubits
            samples.extend([_group_score(bits, group["group"])] * count)
        assert len(samples) == group["shots"] == 512
        mean, variance = float(np.mean(samples)), float(np.var(samples, ddof=1))
        assert group["estimate"] == pytest.approx(mean, abs=1e-12)
        assert group["sample_variance"] == pytest.approx(variance, abs=1e-12)
        assert group["estimated_standard_error"] == pytest.approx(math.sqrt(variance / 512), abs=1e-12)
        assert group["rng"]["entropy"] == [19, 63]
        assert group["rng"]["spawn_key"] == [0, 3, group_index]
        means.append(mean)
        variance_of_means.append(variance / 512)
    assert result["objective_estimate"] == pytest.approx(sum(means), abs=1e-12)
    assert result["estimated_standard_error"] == pytest.approx(math.sqrt(sum(variance_of_means)), abs=1e-12)
    assert result["total_shots"] == 1024
    assert "exact_objective_audit" not in result
    assert all("exact_expectation" not in group for group in result["groups"])


def test_ghz_joint_scores_retain_covariance_between_all_terms():
    parameters = [math.pi / 2, 0.0, 0.0, 0.0]
    result = opt.sampled_spin_vqe_objective(parameters, n_qubits=4,
                                           shots=2048, seed=41, evaluation_index=2,
                                           family=0, stream_id=64)
    z_group, x_group = result["groups"]
    assert set(z_group["counts"]) <= {"0000", "1111"}
    for group in (z_group, x_group):
        samples = [_group_score(bits, group["group"])
                   for bits, count in group["counts"].items() for _ in range(count)]
        assert group["sample_variance"] == pytest.approx(np.var(samples, ddof=1), abs=1e-12)
    # This GHZ state has exact Z-sum variance16 and XX-sum variance1.25;
    # adding individual-term variances would incorrectly give4 and0.75.
    state_x = Statevector.from_instruction(opt.build_spin_vqe_circuit(4, parameters, basis="x")).probabilities()
    scores = np.array([_group_score(f"{index:04b}", "all_x") for index in range(16)])
    assert state_x @ scores == pytest.approx(0, abs=1e-12)
    assert state_x @ (scores**2) == pytest.approx(1.25, abs=1e-12)
    z_mean = z_group["estimate"]
    assert z_group["sample_variance"] == pytest.approx((16 - z_mean**2) * 2048 / 2047, abs=1e-12)


def test_sampled_spin_objective_never_consults_exact_energy_references(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sampled objective must not consult exact references")

    monkeypatch.setattr(opt, "spin_vqe_reference", forbidden)
    monkeypatch.setattr(opt, "spin_vqe_hamiltonian", forbidden)
    monkeypatch.setattr(opt, "spin_vqe_energy_formula", forbidden)
    result = opt.sampled_spin_vqe_objective([0.6, -0.2, 0.4], n_qubits=3,
                                           shots=64, seed=7, evaluation_index=0,
                                           family=0, stream_id=65)
    assert result["total_shots"] == 128


def test_generator_and_dense_sampling_have_separate_width_limits(monkeypatch):
    assert opt.build_spin_vqe_circuit(32, [0.6] * 32).num_qubits == 32
    with pytest.raises(ValueError):
        opt.build_spin_vqe_circuit(33, [0.6] * 33)

    def forbidden(*args, **kwargs):
        raise AssertionError("width preflight must happen before statevector allocation")

    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        opt.sampled_spin_vqe_objective([0.6] * 9, n_qubits=9,
                                       shots=16, seed=7, evaluation_index=0,
                                       family=0, stream_id=65)
    with pytest.raises(ValueError):
        opt.spin_vqe_hamiltonian(9)
    with pytest.raises(ValueError):
        opt.spin_vqe_hamiltonian(9, max_qubits=32)


def test_oversized_dense_reference_rejects_before_matrix_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("the dense-width preflight must precede matrix allocation")

    monkeypatch.setattr(np, "zeros", forbidden)
    monkeypatch.setattr(np, "kron", forbidden)
    with pytest.raises(ValueError):
        opt.spin_vqe_hamiltonian(9)
    with pytest.raises(ValueError):
        opt.spin_vqe_reference(9, [0.0] * 9, max_qubits=32)


@pytest.mark.parametrize("n_qubits,parameters", [(1, [0.6]), (True, [0.6]), (3, [0.6, 0.6]), (2, [0.6, math.nan])])
def test_invalid_width_and_parameter_requests_are_rejected(n_qubits, parameters):
    with pytest.raises(ValueError):
        opt.build_spin_vqe_circuit(n_qubits, parameters)


def test_all_saved_spin_runs_match_independent_ground_states_and_joint_holdouts():
    """Audit the recorded evidence without rerunning training or its references."""
    root = Path(__file__).resolve().parents[1]
    recorded = json.loads((root / "assets/generated/optimization/quality-scaling/runs.json").read_text())
    cases = [case for case in recorded["cases"] if case["algorithm"] == "vqe"]
    assert [case["n_qubits"] for case in cases] == [2, 3, 4, 6]
    checked = 0
    for case in cases:
        n_qubits = case["n_qubits"]
        assert case["target"]["density_gap_at_most"] == 0.05
        assert [run["seed"] for run in case["runs"]] == [7, 19, 41]
        hamiltonian = _independent_hamiltonian(n_qubits)
        ground_energy = float(np.linalg.eigvalsh(hamiltonian)[0])
        for run in case["runs"]:
            state = _independent_ansatz(run["parameters"])
            energy = float(np.vdot(state, hamiltonian @ state).real)
            density_gap = (energy - ground_energy) / n_qubits
            diagnostic = run["diagnostic"]
            assert diagnostic["ground_energy"] == pytest.approx(ground_energy, abs=1e-12, rel=0)
            assert diagnostic["exact_energy"] == pytest.approx(energy, abs=1e-12, rel=0)
            assert diagnostic["energy_density_gap"] == pytest.approx(density_gap, abs=1e-12, rel=0)
            assert run["target_met_audit"] is bool(density_gap <= 0.05)

            validation = run["validation"]
            assert validation["parameters"] == run["parameters"]
            assert [group["group"] for group in validation["groups"]] == ["all_z", "all_x"]
            means, variance_of_means = [], []
            for group in validation["groups"]:
                shots = group["shots"]
                assert shots == 16384
                assert sum(group["counts"].values()) == shots
                scores = {bits: _group_score(bits, group["group"]) for bits in group["counts"]}
                mean = sum(count * scores[bits] for bits, count in group["counts"].items()) / shots
                variance = sum(count * (scores[bits] - mean) ** 2
                               for bits, count in group["counts"].items()) / (shots - 1)
                assert group["estimate"] == pytest.approx(mean, abs=1e-12, rel=0)
                assert group["sample_variance"] == pytest.approx(variance, abs=1e-12, rel=0)
                assert group["estimated_standard_error"] == pytest.approx(
                    math.sqrt(variance / shots), abs=1e-12, rel=0)
                means.append(mean)
                variance_of_means.append(variance / shots)
            estimate = sum(means)
            standard_error = math.sqrt(sum(variance_of_means))
            assert validation["objective_estimate"] == pytest.approx(estimate, abs=1e-12, rel=0)
            assert validation["estimated_standard_error"] == pytest.approx(standard_error, abs=1e-12, rel=0)
            bound = validation["quality_bound"]
            assert bound["normalized_estimate"] == pytest.approx(
                (estimate - ground_energy) / n_qubits, abs=1e-12, rel=0)
            assert bound["normalized_standard_error"] == pytest.approx(
                standard_error / n_qubits, abs=1e-12, rel=0)
            checked += 1
    assert checked == 12
