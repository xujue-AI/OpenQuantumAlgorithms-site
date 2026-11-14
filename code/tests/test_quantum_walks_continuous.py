# SPDX-License-Identifier: MIT
import math

import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from openquantum.quantum_walks_continuous import (
    MAX_DENSE_QUBITS, MAX_DISTRIBUTION_QUBITS, MAX_GENERATOR_QUBITS,
    MAX_STATEVECTOR_QUBITS, analytic_continuous_distribution,
    build_continuous_walk, continuous_observable_budget,
    continuous_position_probability, continuous_return_probability,
    hypercube_hamiltonian, validate_continuous_walk,
)


def independent_kronecker_hamiltonian(n, coupling):
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    result = np.zeros((2**n, 2**n), dtype=complex)
    for q in range(n):
        term = np.ones((1, 1), dtype=complex)
        for k in reversed(range(n)):
            term = np.kron(term, x if k == q else np.eye(2))
        result += coupling*term
    return result


@pytest.mark.parametrize("n,time,coupling,start", [
    (1, .37, 1., 0), (2, .53, -.7, 2), (3, -.28, 1.3, 5),
    (4, math.pi/4, 1., 9), (5, 0., 2., 31), (6, .21, .9, 17),
])
def test_actual_gates_match_independent_matrix_exponential(n, time, coupling, start):
    matrix = expm(-1j*time*independent_kronecker_hamiltonian(n, coupling))
    expected = matrix[:, np.arange(2**n)^start]
    actual = Operator(build_continuous_walk(n, time, coupling, start)).data
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0)
    np.testing.assert_allclose(analytic_continuous_distribution(n, time, coupling, start),
                               np.abs(matrix[:, start])**2, atol=2e-13, rtol=0)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_uniform_and_antipode_transfer_with_normalization(n):
    np.testing.assert_allclose(analytic_continuous_distribution(n, math.pi/4),
                               np.full(2**n, 1/2**n), atol=2e-15, rtol=0)
    np.testing.assert_allclose(analytic_continuous_distribution(n, math.pi*n/4, 1/n),
                               np.full(2**n, 1/2**n), atol=2e-15, rtol=0)
    assert continuous_position_probability(n, math.pi/2, (1 << n)-1) == pytest.approx(1.)
    assert continuous_return_probability(n, math.pi) == pytest.approx(1.)


def test_full_graph_reference_is_independent_bit_flip_adjacency():
    n, coupling = 4, -.8
    matrix = hypercube_hamiltonian(n, coupling)
    np.testing.assert_array_equal(matrix, independent_kronecker_hamiltonian(n, coupling))
    assert np.all(np.count_nonzero(matrix, axis=0) == n)
    np.testing.assert_allclose(matrix.conj().T, matrix)
    assert np.max(np.abs(np.linalg.eigvalsh(matrix))) == pytest.approx(n*abs(coupling))


def test_gate_preparation_and_final_measurement_map_are_charged():
    circuit = build_continuous_walk(5, .7, .4, 0b10110, measure=True)
    assert dict(circuit.count_ops()) == {"x": 3, "rx": 5, "measure": 5}
    assert circuit.num_qubits == 5
    mapping = [(circuit.find_bit(i.qubits[0]).index, circuit.find_bit(i.clbits[0]).index)
               for i in circuit.data if i.operation.name == "measure"]
    assert mapping == list(zip(range(5), range(5)))
    assert circuit.metadata["source_preparation_X_gates"] == 3
    assert not any(i.operation.name in {"unitary", "initialize", "reset"} for i in circuit.data)


def test_time_reversal_and_lowering_keep_the_unitary():
    forward = build_continuous_walk(4, .812, -.23)
    backward = build_continuous_walk(4, -.812, -.23)
    np.testing.assert_allclose(Operator(forward.compose(backward)).data, np.eye(16), atol=1e-14, rtol=0)
    from openquantum.resource_estimation import prepare_measured_circuit
    measured = build_continuous_walk(4, .812, -.23, 9, measure=True)
    lowered = prepare_measured_circuit(measured)
    np.testing.assert_allclose(Operator(lowered.remove_final_measurements(inplace=False)).data,
                               Operator(build_continuous_walk(4, .812, -.23, 9)).data, atol=1e-13, rtol=0)


def test_validation_reports_distinct_classical_bounds():
    result = validate_continuous_walk(4, .31, .8, 11)
    assert result["dense_operator_check_performed"]
    assert result["operator_max_absolute_error"] < 1e-13
    assert result["numerical_checks_passed"]
    assert result["state_max_absolute_error"] < 1e-13
    result = validate_continuous_walk(9, .31)
    assert not result["dense_operator_check_performed"]
    assert result["operator_max_absolute_error"] is None
    assert result["probability_max_absolute_error"] < 1e-13
    large_time = validate_continuous_walk(2, 80.)
    assert "dense_reference_exclusion" in large_time
    assert large_time["probability_max_absolute_error"] < 1e-13


def test_checker_rejects_a_disagreeing_reference(monkeypatch):
    import openquantum.quantum_walks_continuous as module
    monkeypatch.setattr(module, "analytic_continuous_distribution", lambda *args: np.array([1., 0.]))
    with pytest.raises(AssertionError, match="residual exceeds"):
        module.validate_continuous_walk(1, .7)


def test_large_finite_coupling_small_time_does_not_overflow_reference():
    result = validate_continuous_walk(2, 1e-308, coupling=1e308)
    assert result["dense_operator_check_performed"]
    assert result["operator_max_absolute_error"] < 1e-13


def test_bounded_scalar_contract_is_not_a_histogram_guarantee():
    budget = continuous_observable_budget()
    assert budget.outcome_min == 0 and budget.outcome_max == 1
    assert budget.algorithm_bias_bound == 0
    assert budget.absolute_error_target == pytest.approx(.04)
    assert budget.shots_required == math.ceil(math.log(400)/(2*.03**2))
    assert budget.as_contract()["certified_confidence_lower_bound"] is None
    assert budget.resource_budget().executions == budget.shots_required


@pytest.mark.parametrize("n,time,coupling,start", [
    (0, 1., 1., 0), (True, 1., 1., 0), (2.5, 1., 1., 0),
    (MAX_GENERATOR_QUBITS+1, 1., 1., 0), (2, True, 1., 0),
    (2, float("nan"), 1., 0), (2, 1., float("inf"), 0),
    (2, 100., 1e308, 0), (2, 1., 1., -1), (2, 1., 1., 4),
    (2, 1., 1., True),
])
def test_invalid_parameters_rejected_before_allocation(n, time, coupling, start):
    with pytest.raises(ValueError):
        build_continuous_walk(n, time, coupling, start)


def test_materialization_distribution_and_simulation_caps_are_distinct():
    assert build_continuous_walk(MAX_GENERATOR_QUBITS, .1).num_qubits == MAX_GENERATOR_QUBITS
    with pytest.raises(ValueError, match="full analytic"):
        analytic_continuous_distribution(MAX_DISTRIBUTION_QUBITS+1, .1)
    with pytest.raises(ValueError, match="statevector"):
        validate_continuous_walk(MAX_STATEVECTOR_QUBITS+1, .1)
    with pytest.raises(ValueError, match="dense adjacency"):
        hypercube_hamiltonian(MAX_DENSE_QUBITS+1)
    with pytest.raises(ValueError, match="outside"):
        continuous_position_probability(2, .1, 4)
    with pytest.raises(ValueError, match="measure"):
        build_continuous_walk(2, .1, measure="yes")
