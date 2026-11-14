"""Independent functional and coherent checks of actual ripple source gates."""
import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector, Operator

from openquantum.shor import decompose, simulate_phase_distribution
from openquantum.shor_ripple import (
    ripple_adder_gate, modular_adder_gate_ripple, controlled_multiplier_gate_ripple,
    build_order_finding_circuit_ripple, source_primitives,
    simulate_boolean, simulate_boolean_batch, sparse_statevector, simulate_order_finding_ripple,
)


def wrap(gate):
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, range(gate.num_qubits))
    return circuit


@pytest.mark.parametrize("width", [1, 2, 3, 4, 5])
def test_ripple_sum_restores_a_and_carry(width):
    dimension = 1 << width
    circuit = wrap(ripple_adder_gate(width))
    pairs = [(a, b, carry) for a in range(dimension) for b in range(dimension) for carry in [0, 1]]
    values = [a + (b << width) + (carry << (2 * width)) for a, b, carry in pairs]
    actual = simulate_boolean_batch(circuit, values)
    expected = [a + (((a + b + carry) % dimension) << width) + (carry << (2 * width)) for a, b, carry in pairs]
    assert actual == expected
    names = [name for name, _, _ in source_primitives(circuit)]
    assert names.count("ccx") == 2 * width
    assert names.count("cx") == 4 * width


@pytest.mark.parametrize("N,b,k", [(3, 0, 0), (3, 2, 0), (5, 4, 1), (7, 3, 2), (8, 7, 2), (15, 7, 2), (21, 20, 2), (33, 17, 2)])
def test_modular_add_and_inverse_clean_all_workspace(N, b, k):
    circuit = wrap(modular_adder_gate_ripple(N, b, k))
    pairs = [(control, value) for control in range(1 << k) for value in range(N)]
    values = [control + (value << k) for control, value in pairs]
    actual = simulate_boolean_batch(circuit, values)
    expected = [control + ((((value + b) % N) if control == (1 << k) - 1 else value) << k) for control, value in pairs]
    assert actual == expected
    assert simulate_boolean_batch(circuit.inverse(), actual) == values


@pytest.mark.parametrize("N,a", [(3, 2), (5, 2), (7, 3), (15, 2), (15, 7), (21, 2), (21, 5), (33, 2)])
def test_modular_multiplier_all_valid_inputs_and_inverse(N, a):
    circuit = wrap(controlled_multiplier_gate_ripple(N, a))
    pairs = [(c, x) for c in [0, 1] for x in range(N)]
    inputs = [c + (x << 1) for c, x in pairs]
    output = simulate_boolean_batch(circuit, inputs)
    assert output == [c + (((a * x % N) if c else x) << 1) for c, x in pairs]
    assert simulate_boolean_batch(circuit.inverse(), output) == inputs
    assert circuit.num_qubits == 3 * (N - 1).bit_length() + 5


def test_cuccaro_gates_against_independent_qiskit_unitary():
    circuit = wrap(ripple_adder_gate(2))
    actual = Operator(circuit).data
    expected = np.zeros_like(actual)
    for initial in range(actual.shape[1]):
        a, b, carry = initial & 3, (initial >> 2) & 3, (initial >> 4) & 1
        final = a + (((a + b + carry) & 3) << 2) + (carry << 4)
        expected[final, initial] = 1
    np.testing.assert_allclose(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("N,a", [(3, 2), (5, 2), (15, 2)])
def test_actual_coherent_control_relative_phase_and_work_cleanup(N, a):
    gate = controlled_multiplier_gate_ripple(N, a)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.h(0); circuit.s(0); circuit.x(1)
    circuit.append(gate, range(gate.num_qubits))
    actual = Statevector.from_instruction(circuit).data
    expected = np.zeros_like(actual)
    expected[2] = 1 / math.sqrt(2)
    expected[1 + (a << 1)] = 1j / math.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)


def test_small_dense_sparse_and_lowered_complex_amplitudes_agree():
    circuit = build_order_finding_circuit_ripple(3, 2, 2)
    sparse, _ = sparse_statevector(circuit)
    expected = np.zeros(1 << circuit.num_qubits, dtype=complex)
    for value, amplitude in sparse.items():
        expected[value] = amplitude
    actual = Statevector.from_instruction(circuit).data
    lowered = Statevector.from_instruction(decompose(circuit)).data
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
    # In-memory lowering includes the stored compiled global phase.
    np.testing.assert_allclose(lowered, actual, atol=5e-12, rtol=0)


@pytest.mark.parametrize("N,a,t", [(15, 2, 4), (15, 7, 4), (21, 2, 6)])
def test_full_sparse_phase_distribution_matches_independent_fourier_circuit(N, a, t):
    result = simulate_order_finding_ripple(N, a, t)
    old = simulate_phase_distribution(N, a, t)
    np.testing.assert_allclose(result["probabilities"], old["probabilities"], atol=1e-10, rtol=0)
    assert result["dirty_work_probability"] == 0
    assert result["maximum_probability_error"] < 1e-10


@pytest.mark.parametrize("N,b,k", [(2, 1, 0), (5, 5, 0), (5, -1, 0), (5, 1, 3), (True, 1, 0)])
def test_bad_modular_adder_inputs_rejected(N, b, k):
    with pytest.raises(ValueError):
        modular_adder_gate_ripple(N, b, k)


def test_generation_and_simulation_limits():
    with pytest.raises(ValueError, match="preflight"):
        build_order_finding_circuit_ripple(2047, 2, 22)
    large = build_order_finding_circuit_ripple(33, 2, 12)
    assert large.num_qubits == 34
    with pytest.raises(ValueError, match="sparse phase"):
        simulate_order_finding_ripple(33, 2, 12)
    hadamard = QuantumCircuit(1)
    hadamard.h(0)
    with pytest.raises(ValueError, match="Boolean"):
        simulate_boolean(hadamard, 0)
    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    with pytest.raises(ValueError, match="Opaque|opaque|dynamic"):
        source_primitives(measured)


def test_opaque_gate_cannot_impersonate_supported_primitive():
    circuit = QuantumCircuit(1)
    circuit.append(Gate("x", 1, []), [0])
    with pytest.raises(ValueError, match="opaque"):
        sparse_statevector(circuit)
