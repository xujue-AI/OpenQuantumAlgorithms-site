"""Functional clean-domain arithmetic and independent order-finding checks."""
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from openquantum.shor import (
    analytic_phase_distribution, build_order_finding_circuit, classical_order,
    continued_fraction_candidates, controlled_multiplier_gate, decompose,
    factors_from_order, modular_adder_gate, qft_gate, reconstruct_order,
    simulate_phase_distribution, success_contract, validate_instance,
)


@pytest.mark.parametrize("width", [1, 2, 3, 4])
def test_qft_matches_independent_complex_matrix(width):
    d = 1 << width
    reference = np.exp(2j * np.pi * np.outer(np.arange(d), np.arange(d)) / d) / np.sqrt(d)
    np.testing.assert_allclose(Operator(qft_gate(width)).data, reference, atol=4e-15, rtol=0)


@pytest.mark.parametrize("modulus,constant", [(3, 0), (3, 2), (5, 1), (5, 4), (15, 7), (21, 20)])
def test_modular_add_active_inactive_and_clean_flag(modulus, constant):
    gate = modular_adder_gate(modulus, constant, 2)
    values = range(modulus) if modulus < 16 else [0, 1, 10, 19, 20]
    for controls in range(4):
        for value in values:
            initial = controls + (value << 2)
            expected = controls + (((value + constant) % modulus if controls == 3 else value) << 2)
            state = Statevector.from_int(initial, 1 << gate.num_qubits).evolve(gate)
            assert abs(state.data[expected] - 1) < 3e-13
            assert abs(np.linalg.norm(state.data) - 1) < 3e-13


@pytest.mark.parametrize("modulus,multiplier", [(3, 2), (5, 2), (15, 2), (15, 7), (21, 2), (21, 5)])
def test_multiplier_semantics_and_all_work_cleanup(modulus, multiplier):
    gate = controlled_multiplier_gate(modulus, multiplier)
    values = range(modulus) if modulus <= 5 else [0, 1, 2, modulus // 2, modulus - 1]
    for control in [0, 1]:
        for value in values:
            expected_value = multiplier * value % modulus if control else value
            state = Statevector.from_int(control + (value << 1), 1 << gate.num_qubits).evolve(gate)
            assert abs(state.data[control + (expected_value << 1)] - 1) < 6e-13


def test_coherent_control_has_no_relative_global_phase():
    modulus, multiplier = 5, 2
    gate = controlled_multiplier_gate(modulus, multiplier)
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.h(0)
    circuit.x(1)
    circuit.append(gate, range(gate.num_qubits))
    actual = Statevector.from_instruction(circuit).data
    expected = np.zeros_like(actual)
    expected[2] = expected[5] = 1 / math.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=8e-14, rtol=0)


def test_lowering_preserves_small_arithmetic_unitary():
    circuit = QuantumCircuit(7)
    circuit.append(controlled_multiplier_gate(3, 2), range(7))
    compiled = decompose(circuit)
    assert set(compiled.count_ops()) <= {"h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"}
    assert Operator(circuit).equiv(Operator(compiled), atol=2e-12)


@pytest.mark.parametrize("modulus,base,phase_bits", [(3, 2, 2), (15, 2, 4), (15, 7, 5), (21, 2, 6)])
def test_full_circuit_probabilities_and_clean_work_match_independent_reference(modulus, base, phase_bits):
    result = simulate_phase_distribution(modulus, base, phase_bits)
    assert result["max_probability_error"] < 1e-10
    assert result["work_register_zero_probability"] > 1 - 1e-10
    expected_width = phase_bits + 2 * (modulus - 1).bit_length() + 2
    assert result["total_qubits"] == expected_width


@pytest.mark.parametrize("order,phase_bits", [(1, 3), (3, 5), (4, 4), (6, 6), (20, 3)])
def test_reference_distribution_normalized_nonnegative(order, phase_bits):
    probabilities = analytic_phase_distribution(order, phase_bits)
    assert abs(sum(probabilities) - 1) < 2e-13
    assert np.min(probabilities) >= 0


def test_geometric_reference_matches_direct_finite_sums():
    for phase_bits in [1, 3, 5]:
        dimension = 1 << phase_bits
        outcomes = np.arange(dimension)
        for order in [1, 2, 3, 4, 6, dimension + 3]:
            direct = np.zeros(dimension)
            for residue in range(min(order, dimension)):
                positions = np.arange(residue, dimension, order)
                amplitude = np.exp(-2j * math.pi * np.outer(outcomes, positions) / dimension).sum(axis=1) / dimension
                direct += np.abs(amplitude) ** 2
            np.testing.assert_allclose(analytic_phase_distribution(order, phase_bits), direct, atol=5e-15, rtol=0)


def test_continued_fractions_do_not_hide_order_enumeration():
    assert continued_fraction_candidates(4, 4, 15) == [4]
    assert reconstruct_order(4, 4, 15, 2) == 4
    assert reconstruct_order(12, 4, 15, 2) == 4
    assert reconstruct_order(8, 4, 15, 2) is None
    assert reconstruct_order(0, 4, 15, 2) is None
    assert factors_from_order(15, 2, 4) == [3, 5]
    assert factors_from_order(21, 2, 6) == [3, 7]
    assert factors_from_order(15, 14, 2) == []
    assert factors_from_order(7, 2, 3) == []


def test_repeat_contract_for_fixed_base():
    contract = success_contract(15, 2, 4)
    assert contract["ideal_factoring_success_probability_per_trial"] == pytest.approx(0.5)
    assert contract["serial_trials"] == 8
    assert (1 - contract["ideal_factoring_success_probability_per_trial"]) ** 8 <= 0.005
    second = success_contract(21, 2, 6)
    assert second["serial_trials"] == 16
    assert second["ideal_factoring_success_probability_per_trial"] == pytest.approx(0.28798913703656065)
    assert classical_order(21, 2) == 6


@pytest.mark.parametrize("modulus,base", [(2, 1), (15, 3), (15, 0), (15, 15), (True, 1), (8193, 2)])
def test_invalid_instances_rejected(modulus, base):
    with pytest.raises(ValueError):
        validate_instance(modulus, base)


@pytest.mark.parametrize("phase_bits", [0, True, 25])
def test_bad_precision_rejected(phase_bits):
    with pytest.raises(ValueError):
        build_order_finding_circuit(15, 2, phase_bits)


def test_materialization_and_simulation_guards_are_distinct():
    with pytest.raises(ValueError, match="arithmetic-work"):
        build_order_finding_circuit(2047, 2, 22)
    circuit = build_order_finding_circuit(33, 2, 12)
    assert circuit.num_qubits == 26
    with pytest.raises(ValueError, match="Statevector"):
        simulate_phase_distribution(33, 2, 12)
    measured = build_order_finding_circuit(measure=True)
    assert measured.count_ops()["measure"] == 4
    assert measured.num_clbits == 4
