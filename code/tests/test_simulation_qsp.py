"""Independent spectral and circuit checks for the real QSP primitive."""
import numpy as np
import pytest
from qiskit.quantum_info import Operator
from scipy.linalg import cosm

from openquantum.hamiltonian_simulation import dense_hamiltonian, spin_terms, decompose
from openquantum.simulation.lcu import build_pauli_block_encoding
from openquantum.simulation.qsp import (
    COSINE_PHASES, build_real_qsp_filter, build_qubitization_walk,
    qsp_real_response, verify_cosine_phases, postselected_filter_state,
)


def test_frozen_phase_sequence_matches_cosine_across_interval():
    report = verify_cosine_phases()
    assert report["phase_fit_max_residual_10001_point_grid"] < 1e-12
    assert report["target_max_residual_10001_point_grid"] < 4e-9
    assert report["uniform_error_bound_with_numerical_reconstruction_guard"] < 7e-9
    x = np.linspace(-1, 1, 121)
    assert np.max(np.abs(qsp_real_response(x)-qsp_real_response(-x))) < 1e-12


def test_actual_qsp_cosine_block_for_noncommuting_spin_chain():
    circuit = build_real_qsp_filter(spin_terms(3))
    matrix = Operator(circuit).data
    h = dense_hamiltonian(3)
    expected = cosm(h)/2
    np.testing.assert_allclose(matrix[:8, :8], expected, atol=7e-9, rtol=0)
    np.testing.assert_allclose(matrix.conj().T@matrix, np.eye(len(matrix)), atol=2e-12)
    initial = np.eye(8, dtype=complex)[:, 0]
    state, success = postselected_filter_state(circuit, initial)
    branch = expected@initial
    np.testing.assert_allclose(state, branch/np.linalg.norm(branch), atol=3e-8)
    assert success == pytest.approx(np.linalg.norm(branch)**2, abs=7e-9)
    assert circuit.metadata["block_encoding_calls"] == 14
    assert circuit.metadata["n_flag"] == 1


def test_qubitization_walk_has_correct_two_dimensional_eigenphase_sector():
    terms = spin_terms(2)
    block = build_pauli_block_encoding(terms)
    b = Operator(block).data
    walk = Operator(build_qubitization_walk(terms)).data
    h = dense_hamiltonian(2)
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    alpha = block.metadata["normalization"]
    for k, value in enumerate(eigenvalues):
        x = value/alpha
        v = np.zeros(len(b), dtype=complex)
        v[:4] = eigenvectors[:, k]
        w = (b@v-x*v)/np.sqrt(1-x*x)
        basis = np.array([v, w]).T
        expected = np.array([[x, np.sqrt(1-x*x)], [-np.sqrt(1-x*x), x]])
        np.testing.assert_allclose(basis.conj().T@walk@basis, expected, atol=2e-12)
        np.testing.assert_allclose(np.sort_complex(np.linalg.eigvals(expected)),
                                   np.sort_complex(np.exp(1j*np.array([-1, 1])*np.arccos(x))), atol=2e-12)


def test_projector_phase_globals_and_negative_real_coefficients():
    # Degree0 separates phase-global correctness from access-circuit behavior.
    circuit = build_real_qsp_filter([(1, "X"), (-.4, "Z")], [0.31])
    np.testing.assert_allclose(Operator(circuit).data[:2, :2], np.cos(.31)*np.eye(2), atol=1e-12)
    phases = [0.2, -.4]
    circuit = build_real_qsp_filter([(1, "X"), (-.4, "Z")], phases)
    h = np.array([[-.4, 1], [1, .4]])/1.4
    values, vectors = np.linalg.eigh(h)
    expected = (vectors*qsp_real_response(values, phases))@vectors.conj().T
    np.testing.assert_allclose(Operator(circuit).data[:2, :2], expected, atol=1e-12)


def test_decomposed_qsp_circuit_has_no_opaque_hamiltonian_or_unitary_gate():
    circuit = build_real_qsp_filter(spin_terms(1), COSINE_PHASES[:3])
    compiled = decompose(circuit)
    np.testing.assert_allclose(Operator(compiled).data, Operator(circuit).data, atol=2e-12)
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"} & set(compiled.count_ops())


@pytest.mark.parametrize("phases", [[], [float("nan")], [0]*66])
def test_invalid_phase_lists_rejected(phases):
    with pytest.raises(ValueError):
        build_real_qsp_filter([(1, "X")], phases)


def test_invalid_hermitian_coefficient_and_signal_rejected():
    with pytest.raises(ValueError, match="real"):
        build_real_qsp_filter([(1j, "X")])
    with pytest.raises(ValueError, match="signal"):
        qsp_real_response([1.001])
