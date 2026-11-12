"""Independent complex/QSP branch, spectral and decomposed-circuit validation."""
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import expm

from openquantum.hamiltonian_simulation import dense_hamiltonian, spin_terms, pauli_matrix, decompose
from openquantum.simulation.qsp import COSINE_PHASES, build_real_qsp_filter, qsp_real_response
from openquantum.simulation.qsp_evolution import (
    SINE_PHASES, build_qsp_evolution, complex_qsp_response, verify_sine_phases,
    evolution_error_contract, compare_qsp_evolution, postselected_evolution_state,
)


def test_frozen_sine_and_complex_phase_approximation_on_dense_interval():
    report = verify_sine_phases()
    assert report["phase_fit_max_residual_10001_point_grid"] < 3e-14
    assert report["target_max_residual_10001_point_grid"] < 5e-10
    grid = np.linspace(-1, 1, 10001)
    np.testing.assert_allclose(qsp_real_response(-grid, SINE_PHASES), -qsp_real_response(grid, SINE_PHASES), atol=2e-14)
    assert np.max(np.abs(complex_qsp_response(grid)-np.exp(-4.1j*grid)/4)) < 3e-9
    contract = evolution_error_contract()
    assert contract["complete_attempts_for_conditional_acceptance_projection"] == 72
    assert contract["conditional_numerical_U_over_4_operator_error_projection"] < 4e-9


def test_default_complex_evolution_block_state_operator_and_observable():
    circuit = build_qsp_evolution(spin_terms(3))
    initial = np.eye(8, dtype=complex)[:, 0]
    result = compare_qsp_evolution(circuit, dense_hamiltonian(3), initial, pauli_matrix("IIZ"))
    assert result["encoded_block_error"] < 3e-9
    assert result["decoded_linear_map_operator_error"] < 1.2e-8
    assert result["conditional_state_vector_error"] < 2e-8
    assert result["observable_error"] < 3e-8
    assert result["acceptance_probability"] == pytest.approx(1/16, abs=2e-9)
    assert circuit.num_qubits == 8
    assert circuit.metadata["uncontrolled_block_encoding_calls"] == 14
    assert circuit.metadata["controlled_block_encoding_calls"] == 1


def test_combined_circuit_matches_actual_cosine_sine_branches_for_y_model():
    terms = [(1.1, "X"), (-.9, "Y"), (2.1, "Z")]
    circuit = build_qsp_evolution(terms)
    cosine = Operator(build_real_qsp_filter(terms, COSINE_PHASES)).data[:2, :2]
    sine = Operator(build_real_qsp_filter(terms, SINE_PHASES)).data[:2, :2]
    actual = Operator(circuit).data[:2, :2]
    # This tests selector -i phase and all branch-global projector phases.
    np.testing.assert_allclose(actual, (cosine-1j*sine)/2, atol=3e-13)
    h = sum(c*pauli_matrix(p) for c, p in terms)
    np.testing.assert_allclose(actual, expm(-1j*h)/4, atol=3e-9)
    values, vectors = np.linalg.eigh(h/4.1)
    scalar_reference = (vectors*complex_qsp_response(values))@vectors.conj().T
    np.testing.assert_allclose(actual, scalar_reference, atol=3e-13)
    initial = np.array([1, 1j])/np.sqrt(2)
    state, probability = postselected_evolution_state(circuit, initial)
    np.testing.assert_allclose(state, expm(-1j*h)@initial, atol=2e-8)
    assert probability == pytest.approx(1/16, abs=2e-9)


def test_primitive_decomposition_preserves_controlled_globals_and_complex_branch():
    terms = [(1.0, "Y"), (-3.1, "X")]
    circuit = build_qsp_evolution(terms)
    compiled = decompose(circuit)
    np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=2e-12)
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"} & set(compiled.count_ops())


def test_phase_tau_is_not_silently_reused_for_wrong_time_or_hamiltonian():
    with pytest.raises(ValueError, match="normalization"):
        build_qsp_evolution([(1, "Y")], time=1)
    with pytest.raises(ValueError, match="phase_tau"):
        build_qsp_evolution([(1, "Y")], time=2.0, phase_tau=2.0)
    # Supplying matching physical time uses the frozen tau with another model.
    circuit = build_qsp_evolution([(1, "Y")], time=4.1)
    np.testing.assert_allclose(Operator(circuit).data[:2, :2], expm(-4.1j*pauli_matrix("Y"))/4, atol=3e-9)


@pytest.mark.parametrize("kwargs", [
    {"cosine_phases": []}, {"sine_phases": [0]},
    {"sine_phases": [0]*14}, {"cosine_phases": [0]*35},
    {"time": float("nan")}, {"phase_tau": float("inf")},
])
def test_bad_phase_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        build_qsp_evolution(spin_terms(3), **kwargs)


def test_term_and_dense_simulation_bounds_are_separate():
    with pytest.raises(ValueError, match="system"):
        build_qsp_evolution([(4.1, "X"*17)])
    with pytest.raises(ValueError, match="Pauli terms"):
        build_qsp_evolution([(4.1/33, "X")]*33)
    with pytest.raises(ValueError, match="real"):
        build_qsp_evolution([(4.1j, "X")])
    circuit = build_qsp_evolution([(4.1, "X"*7)])
    assert circuit.num_qubits == 9  # Generator can exceed dense-check allocation.
    with pytest.raises(ValueError, match="dense"):
        compare_qsp_evolution(circuit, np.eye(128), np.eye(128)[:, 0], np.eye(128))
    with pytest.raises(ValueError, match="normalized"):
        postselected_evolution_state(build_qsp_evolution(spin_terms(3)), np.ones(8))
