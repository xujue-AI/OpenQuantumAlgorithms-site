"""Oblivious amplification: explicit reflection, exact-model signs, QSP robustness."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.hamiltonian_simulation import spin_terms, dense_hamiltonian, pauli_matrix
from openquantum.simulation.qsp_evolution import build_qsp_evolution
from openquantum.simulation.amplification import (
    zero_ancilla_reflection, build_oblivious_amplification, decompose_amplification,
    ideal_amplified_amplitude, ideal_oblivious_success, amplified_block_polynomial,
    projected_amplified_block, compare_oblivious_amplification, postselected_amplified_state,
)


def test_reflection_only_projects_zero_ancillas_and_leaves_unknown_data_identity():
    reflection = zero_ancilla_reflection(2, 4)
    expected = np.diag([-1]*4+[1]*12)
    np.testing.assert_allclose(Operator(reflection).data, expected, atol=1e-14)
    assert all(reflection.find_bit(q).index >= 2 for instruction in reflection.data for q in instruction.qubits)
    np.testing.assert_allclose(Operator(zero_ancilla_reflection(2, 2)).data, -np.eye(4), atol=1e-14)


@pytest.mark.parametrize("rounds", [0, 1, 2, 3, 4])
def test_exact_scaled_unitary_signs_query_counts_and_full_decomposition(rounds):
    # A=(ancilla RY) tensor(-iY), an explicit exact scaled-unitary block.
    source = QuantumCircuit(2)
    source.ry(2*np.arccos(.25), 1)
    source.y(0)
    source.global_phase = -np.pi/2
    source.metadata = {"n_system": 1}
    circuit = build_oblivious_amplification(source, rounds)
    unitary = -1j*pauli_matrix("Y")
    actual = Operator(circuit).data[:2, :2]
    np.testing.assert_allclose(actual, ideal_amplified_amplitude(rounds)*unitary, atol=2e-14)
    compiled = decompose_amplification(circuit)
    np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=5e-14)
    assert circuit.metadata["forward_A_calls"] == rounds+1
    assert circuit.metadata["inverse_A_calls"] == rounds
    assert circuit.metadata["total_A_or_inverse_calls"] == 2*rounds+1
    assert circuit.metadata["zero_ancilla_reflections"] == 2*rounds
    assert not circuit.metadata["input_state_knowledge_used_by_amplification"]
    state, probability = postselected_amplified_state(circuit, np.array([1, 1j])/np.sqrt(2))
    assert probability == pytest.approx(ideal_oblivious_success(rounds), abs=2e-14)
    np.testing.assert_allclose(state, unitary@np.array([1, 1j])/np.sqrt(2), atol=2e-14)


def test_negative_identity_block_keeps_its_global_phase_after_three_rounds():
    source = QuantumCircuit(2)
    source.ry(2*np.arccos(.25), 1)
    source.global_phase = np.pi
    source.metadata = {"n_system": 1}
    circuit = build_oblivious_amplification(source, 3)
    np.testing.assert_allclose(Operator(circuit).data[:2, :2], -(251/256)*np.eye(2), atol=2e-14)


@pytest.mark.parametrize("rounds", [0, 1, 2, 3])
def test_actual_noncommuting_qsp_evolution_and_general_block_polynomial(rounds):
    source = build_qsp_evolution(spin_terms(3))
    k = projected_amplified_block(source)
    circuit = build_oblivious_amplification(source, rounds)
    actual = projected_amplified_block(circuit)
    assert circuit.metadata["uncontrolled_block_encoding_calls"] == 14*(2*rounds+1)
    assert circuit.metadata["controlled_block_encoding_calls"] == 2*rounds+1
    np.testing.assert_allclose(actual, amplified_block_polynomial(k, rounds), atol=1e-12, rtol=0)
    result = compare_oblivious_amplification(circuit, dense_hamiltonian(3), np.eye(8)[:, 0], pauli_matrix("IIZ"))
    assert result["block_error_vs_ideal_amplitude_times_exact_U"] < 2e-8
    assert result["conditional_state_vector_error"] < 3e-8
    assert result["observable_error"] < 3e-8
    assert abs(result["actual_minus_ideal_success_probability"]) < 3e-8
    assert result["finite_H_numerical_acceptance_input_dependence_span"] < 5e-8


def test_y_and_negative_coefficients_with_unknown_complex_input():
    terms = [(1.1, "X"), (-.9, "Y"), (2.1, "Z")]
    source = build_qsp_evolution(terms)
    circuit = build_oblivious_amplification(source, 3)
    h = sum(c*pauli_matrix(p) for c, p in terms)
    initial = np.array([1, 1j])/np.sqrt(2)
    state, probability = postselected_amplified_state(circuit, initial)
    np.testing.assert_allclose(state, expm(-1j*h)@initial, atol=2e-8)
    assert probability == pytest.approx(ideal_oblivious_success(3), abs=3e-8)
    compiled = decompose_amplification(circuit)
    rng = np.random.default_rng(7)
    full_input = rng.normal(size=2**circuit.num_qubits)+1j*rng.normal(size=2**circuit.num_qubits)
    full_input /= np.linalg.norm(full_input)
    np.testing.assert_allclose(Statevector(full_input).evolve(compiled).data,
                               Statevector(full_input).evolve(circuit).data, atol=3e-12)
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"} & set(compiled.count_ops())


@pytest.mark.parametrize("rounds", [-1, True, 5, .5])
def test_round_bounds_reject_invalid_materialization(rounds):
    source = QuantumCircuit(2)
    source.metadata = {"n_system": 1}
    with pytest.raises(ValueError, match="rounds"):
        build_oblivious_amplification(source, rounds)


def test_opaque_or_measured_source_is_rejected_including_nested_control():
    opaque = QuantumCircuit(2)
    opaque.append(UnitaryGate(np.eye(4)), [0, 1])
    opaque.metadata = {"n_system": 1}
    with pytest.raises(ValueError, match="opaque"):
        build_oblivious_amplification(opaque, 1)
    nested = QuantumCircuit(3)
    nested.append(opaque.to_gate().control(1), [0, 1, 2])
    nested.metadata = {"n_system": 1}
    with pytest.raises(ValueError, match="opaque"):
        build_oblivious_amplification(nested, 1)
    measured = QuantumCircuit(2)
    measured.measure_all()
    measured.metadata = {"n_system": 1}
    with pytest.raises(ValueError, match="classical"):
        build_oblivious_amplification(measured, 1)


def test_generator_dense_and_statevector_bounds_are_separate():
    source = QuantumCircuit(13)
    source.metadata = {"n_system": 11}
    circuit = build_oblivious_amplification(source, 1)
    assert circuit.num_qubits == 13
    with pytest.raises(ValueError, match="dense"):
        projected_amplified_block(circuit)
    with pytest.raises(ValueError, match="statevector"):
        postselected_amplified_state(circuit, np.zeros(2048))
    with pytest.raises(ValueError, match="source_qubits"):
        build_oblivious_amplification(QuantumCircuit(25), 0, n_system=1)


def test_unbound_or_nonfinite_source_parameters_and_global_phase_are_rejected():
    for angle in [Parameter("theta"), float("nan"), float("inf")]:
        source = QuantumCircuit(2)
        source.rx(angle, 0)
        source.metadata = {"n_system": 1}
        with pytest.raises(ValueError, match="bound|finite"):
            build_oblivious_amplification(source, 1)
    source = QuantumCircuit(2)
    source.global_phase = float("nan")
    source.metadata = {"n_system": 1}
    with pytest.raises(ValueError, match="global phase"):
        build_oblivious_amplification(source, 1)
