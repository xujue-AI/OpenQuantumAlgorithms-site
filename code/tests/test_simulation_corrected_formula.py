"""Unexecuted independent checks for the frozen disjoint-pair CPF2 prototype."""
from fractions import Fraction
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from scipy.linalg import expm

from openquantum.simulation.corrected_product_formula import (
    PairModel, append_yz_rotation, canonical_cases, compare_case, complete_trial,
    evolution_circuit, independent_exact_dynamics, independent_formula,
    independent_hamiltonian, independent_prepared_state, logical_ledger,
    preparation_circuit, rational, request,
)


def independent_paulis():
    z = np.diag([1.0, -1.0]).astype(complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return np.kron(z, z), np.kron(np.eye(2), x), np.kron(z, y)


@pytest.mark.parametrize("theta", [Fraction(0), Fraction(1, 3), Fraction(-7, 11)])
def test_literal_yz_rotation_has_full_complex_pauli_semantics(theta):
    circuit = QuantumCircuit(2)
    append_yz_rotation(circuit, 0, 1, theta)
    _, _, yz = independent_paulis()
    expected = math.cos(float(theta)/2)*np.eye(4)-1j*math.sin(float(theta)/2)*yz
    np.testing.assert_allclose(Operator(circuit).data, expected, atol=2e-13, rtol=0)
    assert dict(circuit.count_ops()) == {"h": 2, "cx": 2, "sdg": 1, "rz": 1, "s": 1}


@pytest.mark.parametrize("alpha", ["1", "1/5", "1/20"])
@pytest.mark.parametrize("repetitions", [1, 4])
def test_actual_corrector_conjugation_matches_independent_dense_exponentials(alpha, repetitions):
    model = PairModel(1, alpha=alpha)
    aa, xx, yz = independent_paulis()
    delta = 1/repetitions
    b = float(Fraction(alpha))
    c = 1j*b*delta**2*yz/12
    step = expm(-1j*delta*aa/2) @ expm(-1j*b*delta*xx) @ expm(-1j*delta*aa/2)
    expected = expm(c) @ np.linalg.matrix_power(step, repetitions) @ expm(-c)
    np.testing.assert_allclose(Operator(evolution_circuit(model, "1", repetitions)).data,
                               expected, atol=5e-12, rtol=0)


def test_missing_and_wrong_sign_boundaries_are_detected_on_frozen_noncommuting_case():
    model = PairModel(1, alpha="1/5")
    correct = Operator(evolution_circuit(model, "1", 1, "cpf2")).data
    missing = Operator(evolution_circuit(model, "1", 1, "strang")).data
    wrong = QuantumCircuit(2)
    append_yz_rotation(wrong, 0, 1, Fraction(-1, 30))
    wrong.compose(evolution_circuit(model, "1", 1, "strang"), inplace=True)
    append_yz_rotation(wrong, 0, 1, Fraction(1, 30))
    expected = independent_formula(model, "1", 1)
    np.testing.assert_allclose(correct, expected, atol=5e-12, rtol=0)
    assert np.max(np.abs(missing-expected)) > 1e-6
    assert np.max(np.abs(Operator(wrong).data-expected)) > 1e-6


def test_heterogeneous_signed_two_pair_convention_and_entangled_input():
    model = PairModel(2, a=("1", "1/2"), b=("1", "-3/4"), alpha="1/5")
    actual = Operator(evolution_circuit(model, "1", 2)).data
    reference = independent_formula(model, "1", 2)
    np.testing.assert_allclose(actual, reference, atol=5e-12, rtol=0)
    state = np.zeros(16, dtype=complex)
    state[[0, 3, 12, 15]] = [1, 1j, -1, -1j]
    state /= 2
    np.testing.assert_allclose(actual @ state, reference @ state, atol=5e-12, rtol=0)
    exact = independent_exact_dynamics(model, "1", 2)
    np.testing.assert_allclose(exact, expm(-1j*independent_hamiltonian(model)), atol=5e-12, rtol=0)


@pytest.mark.parametrize("time", ["0", "-1"])
def test_zero_and_negative_time_preserve_formula_and_inverse(time):
    model = PairModel(1, alpha="1/5")
    actual = Operator(evolution_circuit(model, time, 4)).data
    reference = independent_formula(model, time, 4)
    np.testing.assert_allclose(actual, reference, atol=5e-12, rtol=0)
    if time == "0":
        np.testing.assert_allclose(actual, np.eye(4), atol=5e-12, rtol=0)
    else:
        positive = Operator(evolution_circuit(model, "1", 4)).data
        np.testing.assert_allclose(actual, positive.conj().T, atol=5e-12, rtol=0)


def test_reset_preparation_readout_and_closed_form_answer():
    model = PairModel(2, alpha="1/5")
    prepared = Statevector.from_instruction(preparation_circuit(model)).data
    np.testing.assert_allclose(prepared, independent_prepared_state(model), atol=2e-13, rtol=0)
    trial = complete_trial(model, "1", 2)
    measurements = [(trial.find_bit(i.qubits[0]).index, trial.find_bit(i.clbits[0]).index)
                    for i in trial.data if i.operation.name == "measure"]
    assert measurements == [(i, i) for i in range(4)]
    density = DensityMatrix(np.eye(16)/16).evolve(trial.remove_final_measurements(inplace=False)).data
    state = independent_formula(model, "1", 2) @ independent_prepared_state(model)
    np.testing.assert_allclose(density, np.outer(state, state.conj()), atol=5e-12, rtol=0)
    report = compare_case(model, "1", 2)
    assert report["independent_expm_max_entry_discrepancy"] < 5e-12
    for row in report["methods"].values():
        assert row["operator_formula_max_entry_discrepancy"] < 5e-12
        assert row["reset_density_max_entry_discrepancy"] < 5e-12


def test_source_ledger_includes_both_boundaries_zero_angle_calls_and_complete_trial():
    for alpha in ("0", "1/5"):
        model = PairModel(2, alpha=alpha)
        row = logical_ledger(model, "1", 4)
        assert row["source_operations"] == (7*4+21)*2
        assert row["primitive_counts"]["cx"] == (4*4+4)*2
        assert row["continuous_rotation_calls"] == (3*4+2)*2
        assert row["primitive_counts"]["reset"] == row["primitive_counts"]["measure"] == 4
        assert Fraction(row["angle_conversion_ideal_gate_norm_bound_exact"]) >= 0


@pytest.mark.parametrize("value", [True, "nan", "1"*257, "1e1000000000", "1E-1000000000", 1 << 257, Fraction(1, 1 << 257)])
def test_scalar_guards_reject_unbounded_input_before_science(value):
    with pytest.raises(ValueError):
        rational(value, "test")


def test_separate_generation_and_simulation_guards():
    with pytest.raises(ValueError):
        PairModel(0)
    with pytest.raises(ValueError):
        PairModel(2, a=("1",))
    with pytest.raises(ValueError):
        request(PairModel(32), "1", 1024)
    with pytest.raises(ValueError):
        request(PairModel(4), "1", 4, dense=True)
    with pytest.raises(ValueError):
        request(PairModel(1), "1", 17, dense=True)
    assert request(PairModel(32), "1", 8)[2] == (7*8+21)*32


def test_canonical_case_vector_is_frozen_and_contains_controls():
    cases = canonical_cases()
    assert len(cases) == 32
    assert {str(model.alpha) for model, _, _ in cases} == {"0", "1", "1/5", "1/20"}
    assert {str(time) for _, time, _ in cases} == {"-1", "0", "1"}
    assert max(model.qubits for model, _, _ in cases) == 4
