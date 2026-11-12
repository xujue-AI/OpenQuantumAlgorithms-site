"""Check the actual dilation and measurement circuit against independent answers."""
import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import DensityMatrix, Operator, Statevector, partial_trace

from openquantum.simulation.open_systems import (
    INPUT_LABELS, PAULI_MATRICES, analytic_amplitude_damping, analytic_finite_time_damping, build_amplitude_damping,
    compare_damping, damping_dilation, damping_probability, independent_damping_circuit,
    input_density, kraus_operators, lower_to_rotation_basis, observable_shots,
    reduced_dilation_state, reduced_finite_time_state,
)


@pytest.mark.parametrize("p", [0, 0.2, 0.9, 1])
def test_unitary_dilation_blocks_are_the_kraus_operators(p):
    U = Operator(damping_dilation(p)).data
    K0, K1 = kraus_operators(p)
    np.testing.assert_allclose(U[:2, :2], K0, atol=1e-13)
    np.testing.assert_allclose(U[2:, :2], K1, atol=1e-13)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-13)
    np.testing.assert_allclose(K0.conj().T @ K0+K1.conj().T @ K1, np.eye(2), atol=1e-13)


@pytest.mark.parametrize("label", INPUT_LABELS)
@pytest.mark.parametrize("time", [0, 0.25, 1, 2, 40])
def test_prepared_inputs_match_kraus_with_trace_positivity_and_observables(label, time):
    result = compare_damping(0.8, time, label)
    assert result["trace_distance"] < 1e-12
    assert result["matrix_entry_max_error"] < 1e-12
    assert result["trace_real"] == pytest.approx(1, abs=1e-12)
    assert abs(result["trace_imag"]) < 1e-12
    assert result["minimum_eigenvalue"] > -1e-12
    assert result["maximum_observable_error"] < 1e-12


def test_arbitrary_mixed_inputs_and_semigroup_composition():
    rng = np.random.default_rng(42)
    for _ in range(5):
        M = rng.normal(size=(2, 2))+1j*rng.normal(size=(2, 2))
        rho = M @ M.conj().T
        rho /= np.trace(rho)
        p = damping_probability(0.8, 0.5)
        actual = reduced_dilation_state(rho, p)
        reference = analytic_amplitude_damping(rho, p)
        np.testing.assert_allclose(actual, reference, atol=1e-13)
        sequential = reduced_dilation_state(actual, p)
        single = reduced_dilation_state(rho, damping_probability(0.8, 1))
        np.testing.assert_allclose(sequential, single, atol=1e-13)


def test_finite_time_survival_is_not_lost_when_probability_rounds_to_one():
    assert damping_probability(1, 40) == 1
    result = compare_damping(1, 40, "plus")
    assert result["circuit_observables"]["X"] == pytest.approx(math.exp(-20), abs=1e-16)
    assert result["circuit_observables"]["X"] > 1e-9
    reference = analytic_finite_time_damping(input_density("plus"), 1, 40)
    actual = reduced_finite_time_state(input_density("plus"), 1, 40)
    np.testing.assert_allclose(actual, reference, atol=1e-16)


@pytest.mark.parametrize("observable", ["X", "Y", "Z"])
def test_final_basis_change_measures_requested_mean_and_discards_environment(observable):
    circuit = build_amplitude_damping(input_label="plus_i", observable=observable, reset=True)
    lowered = lower_to_rotation_basis(circuit)
    assert lowered.count_ops()["reset"] == 2
    assert lowered.count_ops()["measure"] == 2
    probabilities = DensityMatrix.from_instruction(circuit.remove_final_measurements(inplace=False)).probabilities()
    mean = float(sum((1 if i % 2 == 0 else -1) * value for i, value in enumerate(probabilities)))
    rho = analytic_amplitude_damping(input_density("plus_i"), damping_probability(0.8, 1))
    assert mean == pytest.approx(np.trace(rho @ PAULI_MATRICES[observable]).real, abs=1e-12)
    export = qasm3.dumps(lowered)
    assert "gate " not in export and "reset" in export and "measure" in export


def test_decomposition_preserves_actual_gate_semantics():
    circuit = build_amplitude_damping(input_label="plus_i")
    lowered = lower_to_rotation_basis(circuit)
    np.testing.assert_allclose(Operator(circuit).data, Operator(lowered).data, atol=1e-13)
    assert set(lowered.count_ops()) <= {"h", "s", "ry", "cx"}


def test_independent_layer_on_an_entangled_two_system_input():
    circuit = independent_damping_circuit(2, 0.8, 0.5)
    initial = np.zeros(16, dtype=complex)
    initial[0] = initial[3] = 1/math.sqrt(2)
    reduced = partial_trace(Statevector(initial).evolve(circuit), [2, 3]).data
    bell = np.zeros(4, dtype=complex)
    bell[0] = bell[3] = 1/math.sqrt(2)
    rho = np.outer(bell, bell.conj())
    kraus = kraus_operators(damping_probability(0.8, 0.5))
    reference = sum(np.kron(a, b) @ rho @ np.kron(a, b).conj().T for a in kraus for b in kraus)
    np.testing.assert_allclose(reduced, reference, atol=1e-13)


def test_generation_width_is_separate_from_dense_simulation():
    assert independent_damping_circuit(32).num_qubits == 64
    assert observable_shots() == 13315


@pytest.mark.parametrize("rate,time", [(True, 1), (-1, 1), (0.8, -1), (float("nan"), 1), (1, float("inf"))])
def test_invalid_rates_times_are_rejected(rate, time):
    with pytest.raises(ValueError):
        damping_probability(rate, time)


@pytest.mark.parametrize("p", [-0.1, 1.1, True, float("nan")])
def test_invalid_probabilities_are_rejected(p):
    with pytest.raises(ValueError):
        damping_dilation(p)


def test_invalid_density_and_preparation_are_rejected():
    for rho in [np.eye(3)/3, np.array([[1, 1j], [0, 0]]), np.diag([2, -1])]:
        with pytest.raises(ValueError):
            analytic_amplitude_damping(rho, 0.2)
    with pytest.raises(ValueError):
        independent_damping_circuit(33)
    with pytest.raises(ValueError):
        build_amplitude_damping(input_label="unknown")
    with pytest.raises(ValueError):
        build_amplitude_damping(observable="unknown")
    with pytest.raises(ValueError, match="caller-supplied"):
        independent_damping_circuit(reset=True)
