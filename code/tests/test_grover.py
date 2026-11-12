"""Check the oracle semantics, register ordering, and actual Grover behavior."""

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

from openquantum.grover import (
    ROTATION_BASIS,
    build_grover_circuit,
    decompose_to_rotation_basis,
    expected_success_probability,
    optimal_iterations,
    phase_oracle,
)


@pytest.mark.parametrize("n_qubits,marked", [(1, 0), (1, 1), (2, 1), (3, 5), (4, 0), (4, 15)])
def test_oracle_flips_exactly_the_selected_basis_state(n_qubits, marked):
    diagonal = np.ones(2**n_qubits)
    diagonal[marked] = -1
    np.testing.assert_allclose(Operator(phase_oracle(n_qubits, marked)).data, np.diag(diagonal), atol=1e-12)


@pytest.mark.parametrize("n_qubits,marked", [(1, 0), (2, 3), (3, 5), (4, 3), (5, 19), (6, 62)])
def test_simulated_success_matches_closed_form(n_qubits, marked):
    for iterations in [0, 1, optimal_iterations(n_qubits), optimal_iterations(n_qubits) + 1]:
        state = Statevector.from_instruction(build_grover_circuit(n_qubits, marked, iterations))
        assert state.probabilities()[marked] == pytest.approx(
            expected_success_probability(n_qubits, iterations), abs=1e-11
        )


def test_first_demo_and_decomposition_preserve_behavior():
    circuit = build_grover_circuit(3, 5)
    compiled = decompose_to_rotation_basis(circuit)
    assert optimal_iterations(3) == 2
    assert Statevector.from_instruction(circuit).probabilities_dict()["101"] == pytest.approx(0.9453125)
    assert set(compiled.count_ops()) <= set(ROTATION_BASIS)
    assert Operator(circuit).equiv(Operator(compiled))
    assert "OPENQASM 3.0" in qasm3.dumps(compiled)


@pytest.mark.parametrize("args", [(0, 0), (-1, 0), (2, -1), (2, 4), (True, 0), (2, True)])
def test_invalid_problem_is_rejected(args):
    with pytest.raises(ValueError):
        build_grover_circuit(*args)


@pytest.mark.parametrize("iterations", [-1, 0.5, True])
def test_invalid_iteration_count_is_rejected(iterations):
    with pytest.raises(ValueError):
        build_grover_circuit(3, 5, iterations)
