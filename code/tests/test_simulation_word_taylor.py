"""Actual word access, independent matrix references and complete instruments."""
import math
from unittest.mock import patch
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

from openquantum.hamiltonian_simulation import pauli_matrix
from openquantum.simulation import word_taylor as wt
from openquantum.simulation.amplification import amplified_block_polynomial


@pytest.mark.parametrize("terms,time,degree", [
    ([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], .25, 3),
    ([(.4, "X"), (-.3, "Y"), (.2, "Z")], -.4, 3),
    ([(-.5, "Y")], .5, 3),
    ([(.6, "Z"), (.4, "X")], 0, 3),
    ([(.6, "Z"), (.4, "X")], .25, 0),
])
def test_actual_zero_word_block_matches_independent_matrix_taylor(terms, time, degree):
    access = wt.build_word_taylor(terms, time, degree)
    h, polynomial = wt.matrix_taylor_reference(access)
    direct = sum(np.linalg.matrix_power(-1j*time*h, k)/math.factorial(k) for k in range(degree+1))
    np.testing.assert_allclose(polynomial, direct, atol=1e-13)
    np.testing.assert_allclose(wt.projected_word_block(access.circuit), direct/access.normalization, atol=4e-12)
    assert access.circuit.num_qubits == access.n_system+degree.bit_length()+degree*(len(terms)-1).bit_length()
    assert access.circuit.metadata["word_enumeration_performed"] is False


def test_degree_and_all_independent_terms_are_prepared_with_correct_weights():
    access = wt.build_word_taylor([(.6, "ZZ"), (-.35, "IX"), (.35, "XI")], .25, 3)
    degree = Statevector.from_instruction(access.prepare_degree).probabilities()
    np.testing.assert_allclose(degree, access.degree_probabilities, atol=2e-14)
    term = Statevector.from_instruction(access.prepare_term).probabilities()
    expected_term = np.r_[access.term_probabilities, 0]
    np.testing.assert_allclose(term, expected_term, atol=2e-14)
    joint = Statevector.from_instruction(access.prepare).probabilities()
    expected_joint = access.degree_probabilities
    for _ in range(access.degree):
        expected_joint = np.kron(expected_term, expected_joint)
    np.testing.assert_allclose(joint, expected_joint, atol=2e-14)


def test_select_chronology_preserves_noncommuting_word_order_and_inactive_slots():
    access = wt.build_word_taylor([(1, "X"), (1, "Y")], .2, 2)
    # degree2, slot0=X, slot1=Y -> (-i)^2 YX=+iZ.
    ancilla_value = 2+(0 << 2)+(1 << 3)
    for basis in [0, 1]:
        full_basis = basis+(ancilla_value << 1)
        state = Statevector.from_int(full_basis, 2**access.circuit.num_qubits).evolve(access.select)
        assert state.data[full_basis] == pytest.approx(1j*(-1)**basis, abs=1e-12)
        assert np.count_nonzero(abs(state.data)>1e-12) == 1
    # degree0 ignores both prepared word registers, even if labels are nonzero.
    ancilla_value = (1 << 2)+(1 << 3)
    state = Statevector.from_int(ancilla_value << 1, 2**access.circuit.num_qubits).evolve(access.select)
    assert state.data[ancilla_value << 1] == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize("degree_label,slots,expected", [
    (1, (0, 1), 1j*np.eye(2)),              # Active negative identity; inactive negative Y ignored.
    (1, (1, 0), 1j*pauli_matrix("Y")),      # Active negative Y; inactive negative identity ignored.
    (1, (2, 1), -1j*pauli_matrix("Z")),     # Inactive negative Y must not flip this positive active branch.
    (2, (1, 2), pauli_matrix("Z")@pauli_matrix("Y")),  # Signs and chronological ZY phase.
    (0, (0, 1), np.eye(2)),                # No active negative coefficient.
    (3, (0, 1), np.eye(2)),                # Padded degree applies identity.
    (1, (3, 1), -1j*np.eye(2)),            # Padded active index identity; valid degree phase remains.
])
def test_direct_select_sign_identity_inactive_and_padded_branches(degree_label, slots, expected):
    access = wt.build_word_taylor([(-.4, "I"), (-.3, "Y"), (.2, "Z")], .2, 2)
    ancilla_value = degree_label+(slots[0] << access.n_degree)+(slots[1] << (access.n_degree+access.n_term))
    start = ancilla_value*2
    for basis in range(2):
        actual = Statevector.from_int(start+basis, 2**access.circuit.num_qubits).evolve(access.select).data
        reference = np.zeros_like(actual)
        reference[start:start+2] = expected[:, basis]
        np.testing.assert_allclose(actual, reference, atol=2e-12)


def test_no_pauli_aggregation_or_exponential_word_enumeration_is_called():
    with patch("openquantum.simulation.lcu.taylor_pauli_terms", side_effect=AssertionError("aggregation called")), patch("openquantum.simulation.lcu.build_pauli_lcu", side_effect=AssertionError("flat LCU called")):
        access = wt.build_word_taylor([(1, "X"), (.8, "Y"), (.6, "Z"), (.4, "I"), (.2, "X")], .1, 7)
    assert access.circuit.metadata["SELECT_slot_degree_term_predicates"] == 5*7*8//2
    assert access.circuit.metadata["degree_coefficient_count"] == 8
    assert access.circuit.metadata["term_coefficient_count"] == 5
    assert access.circuit.size() < 10000
    assert access.circuit.num_qubits == 1+3+7*3
    with pytest.raises(ValueError, match="12 total wires"):
        wt.projected_word_block(access.circuit)


@pytest.mark.parametrize("segments", [1, 2, 4])
@pytest.mark.parametrize("rounds", [0, 1])
def test_actual_segment_and_repeated_instrument_match_general_amplification_polynomial(segments, rounds):
    access = wt.build_word_taylor([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], 1/segments, 3)
    check = wt.compare_word_dynamics(access, segments, rounds)
    assert check["actual_source_block_error_vs_matrix_taylor"] < 5e-12
    assert check["actual_amplified_block_error_vs_general_polynomial"] < 5e-12
    assert check["conditional_state_error_vs_independent_polynomial"] < 5e-12
    assert check["full_attempt_acceptance_probability"] == pytest.approx(check["independent_full_attempt_acceptance_probability"], abs=5e-12)
    assert check["acceptance_input_dependence_span"] > 0
    assert not check["acceptance_certificate"]


def test_overrotation_is_reported_without_a_false_success_improvement():
    access = wt.build_word_taylor([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], .25, 3)
    plain = wt.compare_word_dynamics(access, 4, 0)
    amplified = wt.compare_word_dynamics(access, 4, 1)
    assert plain["conditional_state_vector_error"] < .00025
    assert amplified["full_attempt_acceptance_probability"] < plain["full_attempt_acceptance_probability"]
    assert amplified["conditional_state_vector_error"] > plain["conditional_state_vector_error"]


@pytest.mark.parametrize("segments", [2, 4])
def test_actual_normalization2_zero_lcu_and_all_ancilla_oaa(segments):
    access = wt.build_word_taylor([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], 1/segments, 3)
    source = wt.padded_word_source(access)
    assert source.num_qubits == 12
    _, polynomial = wt.matrix_taylor_reference(access)
    np.testing.assert_allclose(wt.projected_word_block(source), polynomial/2, atol=2e-12)
    check = wt.compare_word_dynamics(access, segments, 1, padding_to_two=True)
    assert check["actual_amplified_block_error_vs_general_polynomial"] < 2e-12
    assert check["conditional_state_error_vs_independent_polynomial"] < 2e-12
    assert check["full_attempt_acceptance_probability"] > .9999
    assert check["full_attempt_acceptance_probability"] < 1
    if segments == 4:
        assert check["conditional_state_vector_error"] < .00007
    attempt = wt.build_word_attempt(access, segments, 1, padding_to_two=True)
    assert attempt.count_ops()["measure"] == 2+segments*10
    assert attempt.count_ops()["reset"] == 2+segments*10


def test_padding_rejects_s_greater_than_two_and_handles_zero_time_identity():
    with pytest.raises(ValueError, match="s<=2"):
        wt.padded_word_source(wt.build_word_taylor([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], 1, 3))
    access = wt.build_word_taylor([(.5, "X"), (-.5, "Y")], 0, 2)
    source = wt.padded_word_source(access)
    np.testing.assert_allclose(wt.projected_word_block(source), np.eye(2)/2, atol=1e-12)
    amplified = wt.amplified_word_segment(access, 1, padding_to_two=True)
    np.testing.assert_allclose(wt.projected_word_block(amplified), np.eye(2), atol=1e-12)


def test_full_register_phase_preservation_without_initial_clean_wire_assumptions():
    access = wt.build_word_taylor([(.6, "ZZ"), (.35, "IX"), (.35, "XI")], .25, 3)
    source = wt.amplified_word_segment(access, 1)
    compiled = wt.decompose_word_circuit(source)
    rng = np.random.default_rng(7)
    initial = rng.normal(size=2**source.num_qubits)+1j*rng.normal(size=2**source.num_qubits)
    initial /= np.linalg.norm(initial)
    np.testing.assert_allclose(Statevector(initial).evolve(source).data,
                               Statevector(initial).evolve(compiled).data, atol=3e-12)


def test_unknown_complex_input_and_noncommuting_negative_y_hamiltonian():
    access = wt.build_word_taylor([(.4, "X"), (-.3, "Y"), (.2, "Z")], .2, 3)
    initial = np.array([1, 1j])/np.sqrt(2)
    result = wt.compare_word_dynamics(access, 2, 1, initial, pauli_matrix("X"))
    assert result["actual_amplified_block_error_vs_general_polynomial"] < 4e-12
    assert result["conditional_state_error_vs_independent_polynomial"] < 4e-12
    assert result["conditional_state_vector_error"] < 1e-4


def test_primitive_lowering_preserves_relative_and_global_phases():
    access = wt.build_word_taylor([(.4, "X"), (-.3, "Y")], -.3, 2)
    source = wt.amplified_word_segment(access, 1)
    compiled = wt.decompose_word_circuit(source)
    # Total5 wires permits complete Operator equality, not global-phase alignment.
    np.testing.assert_allclose(Operator(compiled).data, Operator(source).data, atol=4e-12)
    qasm = qasm3.dumps(compiled)
    assert "OPENQASM 3.0" in qasm
    assert "gate " not in qasm
    assert not {"unitary", "hamiltonian", "initialize", "state_preparation"}&set(compiled.count_ops())


def test_complete_reset_readout_schedule_has_the_correct_accepted_instrument():
    access = wt.build_word_taylor([(.4, "X"), (-.3, "Z")], .2, 1)
    attempt = wt.build_word_attempt(access, 2, 1)
    n_ancillas = attempt.num_qubits-access.n_system
    assert attempt.count_ops()["reset"] == access.n_system+2*n_ancillas
    assert attempt.count_ops()["measure"] == access.n_system+2*n_ancillas
    state = Statevector.from_int(0, 2**attempt.num_qubits)
    success = 1.0
    indices = np.arange(len(state.data))
    for instruction in attempt.data:
        qubits = [attempt.find_bit(q).index for q in instruction.qubits]
        if instruction.operation.name == "measure":
            if attempt.find_bit(instruction.clbits[0]).index >= 2*n_ancillas:
                continue
            projected = state.data.copy()
            projected[(indices&(1 << qubits[0])) != 0] = 0
            probability = float(np.vdot(projected, projected).real)
            state = Statevector(projected/math.sqrt(probability))
            success *= probability
        elif instruction.operation.name == "reset":
            assert state.probabilities()[(indices&(1 << qubits[0])) != 0].sum() < 1e-12
        else:
            state = state.evolve(instruction.operation, qargs=qubits)
    _, polynomial = wt.matrix_taylor_reference(access)
    block = amplified_block_polynomial(polynomial/access.normalization, 1)
    branch = (block@block)@np.array([1, 0], dtype=complex)
    expected_success = float(np.vdot(branch, branch).real)
    assert success == pytest.approx(expected_success, abs=4e-12)
    np.testing.assert_allclose(state.data[:2], branch/math.sqrt(expected_success), atol=4e-12)
    compiled = wt.decompose_word_circuit(attempt)
    assert compiled.count_ops()["reset"] == attempt.count_ops()["reset"]


@pytest.mark.parametrize("kwargs", [
    {"degree": -1}, {"degree": True}, {"degree": 9}, {"time": np.inf},
    {"time": True}, {"time": 100},
])
def test_invalid_size_time_and_degree_bounds(kwargs):
    with pytest.raises(ValueError):
        wt.build_word_taylor([(1, "X")], **kwargs)


@pytest.mark.parametrize("terms", [[], [(0, "X")], [(1j, "X")], [(True, "X")],
                                       [(np.nan, "Z")], [(1, "A")], [(1, "X"), (1, "ZZ")]])
def test_invalid_hamiltonian_representation_is_rejected(terms):
    with pytest.raises(ValueError):
        wt.build_word_taylor(terms)


def test_no_sim_regeneration_removes_stale_managed_curves_scaling_and_exports(tmp_path):
    script = Path(__file__).resolve().parents[1]/"scripts/generate_simulation_word_taylor.py"
    spec = importlib.util.spec_from_file_location("generate_word_taylor_under_test", script)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    generator.generate(tmp_path, n=1, degree=1, segments=(1,), rounds=(0,), include_scaling=True)
    assert (tmp_path/"convergence.csv").is_file()
    assert (tmp_path/"scaling.csv").is_file()
    assert (tmp_path/"access-circuit.svg").is_file()
    unrelated = tmp_path/"reader-notes.txt"
    unrelated.write_text("keep this unrelated file")
    result = generator.generate(tmp_path, n=8, degree=3, segments=(1,), rounds=(0,),
                                simulate=False, include_padding=False, include_scaling=False)
    assert result["records"][0]["simulation"]["performed"] is False
    for name in ("convergence.csv", "convergence.svg", "scaling.csv", "access-circuit.svg"):
        assert not (tmp_path/name).exists()
    assert unrelated.read_text() == "keep this unrelated file"
    assert [path.name for path in tmp_path.glob("*.qasm")] == ["degree3-segments1-rounds0.qasm"]
