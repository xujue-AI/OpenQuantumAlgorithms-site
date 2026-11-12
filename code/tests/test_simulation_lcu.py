"""Independent Pauli-matrix checks of actual access and postselection circuits."""

import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.grover import decompose_to_rotation_basis
from openquantum.simulation.lcu import (
    build_lcu_attempt,
    build_pauli_block_encoding,
    build_pauli_lcu,
    build_taylor_lcu,
    postselected_lcu_evolution,
    postselection_attempts,
    taylor_pauli_terms,
)


MATRICES = {
    "I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
    "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1]),
}


def matrix(terms):
    dimension = 2 ** len(terms[0][1])
    result = np.zeros((dimension, dimension), dtype=complex)
    for coefficient, pauli in terms:
        product = np.array([[1]], dtype=complex)
        for letter in pauli:
            product = np.kron(product, MATRICES[letter])
        result += coefficient * product
    return result


def extracted_block(lcu):
    dimension = 2**lcu.n_system
    columns = []
    for basis in range(dimension):
        state = np.zeros(2**lcu.circuit.num_qubits, dtype=complex)
        state[basis] = 1
        columns.append(Statevector(state).evolve(lcu.circuit).data[:dimension])
    return np.column_stack(columns)


@pytest.mark.parametrize("terms", [
    [(1, "X")], [(2, "I"), (-0.7, "Y")],
    [(0.5j, "IX"), (-0.2, "YZ"), (1, "ZI")],
    [(1, "ZZI"), (1, "IZZ"), (0.7, "IIX"), (0.7, "IXI"), (0.7, "XII")],
])
def test_prepare_select_encode_the_actual_pauli_sum(terms):
    lcu = build_pauli_lcu(terms)
    np.testing.assert_allclose(extracted_block(lcu), matrix(terms) / lcu.normalization, atol=2e-12)
    if lcu.n_index:
        probabilities = Statevector.from_instruction(lcu.prepare).probabilities()
        expected = np.zeros(2**lcu.n_index)
        expected[:len(terms)] = [abs(c) / lcu.normalization for c, _ in terms]
        np.testing.assert_allclose(probabilities, expected, atol=1e-13)
    assert build_pauli_block_encoding(terms).metadata["normalization"] == lcu.normalization


def test_access_decomposition_preserves_coefficient_phases_and_exports():
    lcu = build_pauli_lcu([(1j, "IX"), (-0.5, "YZ"), (0.3, "ZI")])
    compiled = decompose_to_rotation_basis(lcu.circuit)
    assert Operator(compiled).equiv(Operator(lcu.circuit))
    export = qasm3.dumps(compiled)
    assert "OPENQASM 3.0" in export
    assert "gate " not in export


@pytest.mark.parametrize("degree", [0, 1, 2, 3, 6, 12])
def test_classically_aggregated_taylor_coefficients_match_dense_polynomial(degree):
    terms = [(1, "ZZI"), (1, "IZZ"), (0.7, "IIX"), (0.7, "IXI"), (0.7, "XII")]
    hamiltonian = matrix(terms)
    time = 0.25
    reference = sum(np.linalg.matrix_power(-1j * time * hamiltonian, k) / math.factorial(k)
                    for k in range(degree + 1))
    np.testing.assert_allclose(matrix(taylor_pauli_terms(terms, time, degree)), reference, atol=3e-14)


def test_segment_circuit_has_the_truncated_polynomial_block():
    terms = [(1, "ZZI"), (1, "IZZ"), (0.7, "IIX"), (0.7, "IXI"), (0.7, "XII")]
    lcu = build_taylor_lcu(terms, 0.25, 4)
    hamiltonian = matrix(terms)
    polynomial = sum(np.linalg.matrix_power(-0.25j * hamiltonian, k) / math.factorial(k)
                     for k in range(5))
    np.testing.assert_allclose(extracted_block(lcu), polynomial / lcu.normalization, atol=3e-12)


def test_fresh_ancilla_projection_and_restart_probability_match_independent_classical_answer():
    terms = [(1, "ZZI"), (1, "IZZ"), (0.7, "IIX"), (0.7, "IXI"), (0.7, "XII")]
    initial = np.zeros(8, dtype=complex)
    initial[0] = 1
    errors = []
    for segments in [2, 4, 8]:
        lcu = build_taylor_lcu(terms, 1 / segments, 4)
        actual, success, conditional = postselected_lcu_evolution(lcu, initial, segments)
        polynomial = matrix(taylor_pauli_terms(terms, 1 / segments, 4)) / lcu.normalization
        unnormalized = np.linalg.matrix_power(polynomial, segments) @ initial
        expected_success = np.vdot(unnormalized, unnormalized).real
        np.testing.assert_allclose(actual, unnormalized / math.sqrt(expected_success), atol=3e-12)
        assert success == pytest.approx(expected_success, abs=1e-12)
        assert math.prod(conditional) == pytest.approx(success)
        exact = expm(-1j * matrix(terms)) @ initial
        errors.append(np.linalg.norm(actual - exact))
        attempts = postselection_attempts(success)
        assert (1 - success)**attempts <= 0.01
        if attempts > 1:
            assert (1 - success)**(attempts - 1) > 0.01
    assert errors[2] < errors[1] < errors[0]
    assert errors[-1] < 0.001


@pytest.mark.parametrize("terms", [[], [(0, "X")], [(1, "X"), (1, "XX")],
    [(np.nan, "Z")], [(1, "A")], [(1e308, "X"), (1e308, "Z")]])
def test_invalid_access_inputs_are_rejected(terms):
    with pytest.raises(ValueError):
        build_pauli_lcu(terms)


@pytest.mark.parametrize("probability,target", [(0, 0.01), (1.1, 0.01), (0.5, 0), (0.5, 1)])
def test_invalid_acceptance_budget_is_rejected(probability, target):
    with pytest.raises(ValueError):
        postselection_attempts(probability, target)


def test_complete_attempt_measurement_reset_and_accepted_instrument():
    """Follow the all-zero instrument of the exported full schedule exactly."""
    lcu = build_taylor_lcu([(0.7, "X"), (0.4, "Z")], 0.5, 2)
    attempt = build_lcu_attempt(lcu, 2)
    assert attempt.count_ops()["measure"] == 2 * lcu.n_index + lcu.n_system
    assert attempt.count_ops()["reset"] == lcu.n_system + 2 * lcu.n_index
    state = Statevector.from_int(0, 2**attempt.num_qubits)
    success = 1.0
    for instruction in attempt.data:
        qubits = [attempt.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "measure":
            clbit = attempt.find_bit(instruction.clbits[0]).index
            if clbit >= 2 * lcu.n_index:
                continue  # Final data outcome is retained, rather than postselected.
            projected = state.data.copy()
            projected[np.arange(len(projected)) & (1 << qubits[0]) != 0] = 0
            probability = float(np.vdot(projected, projected).real)
            success *= probability
            state = Statevector(projected / np.sqrt(probability))
        elif instruction.operation.name == "reset":
            # Every reset is on an already-zero wire on this accepted branch.
            assert sum(state.probabilities()[np.arange(len(state.data)) & (1 << qubits[0]) != 0]) < 1e-12
        else:
            state = state.evolve(instruction.operation, qargs=qubits)
    polynomial = matrix(lcu.terms) / lcu.normalization
    reference = np.linalg.matrix_power(polynomial, 2) @ np.array([1, 0], dtype=complex)
    probability = float(np.vdot(reference, reference).real)
    assert success == pytest.approx(probability)
    np.testing.assert_allclose(state.data[:2], reference / np.sqrt(probability), atol=1e-12)


def test_taylor_rejects_a_nonhermitian_pauli_model():
    with pytest.raises(ValueError, match="must be real"):
        build_taylor_lcu([(1j, "X")], 1, 2)


def test_postselection_simulator_rejects_unbounded_segment_count():
    lcu = build_pauli_lcu([(1, "X")])
    with pytest.raises(ValueError, match="256"):
        postselected_lcu_evolution(lcu, np.array([1, 0]), 10**9)
