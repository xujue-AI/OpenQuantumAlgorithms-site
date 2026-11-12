"""Check gate semantics, independent evolution, convergence and boundedness."""
import numpy as np
import pytest
from qiskit import qasm3
from qiskit.quantum_info import Operator

from openquantum.hamiltonian_simulation import (
    compare_product_formula, decompose, dense_hamiltonian, exact_evolution,
    observable_shots, product_formula_circuit, spin_terms,
)


def test_spin_representation_and_bit_order():
    assert spin_terms(3) == [(1.0, "IZZ"), (1.0, "ZZI"),
                             (0.7, "IIX"), (0.7, "IXI"), (0.7, "XII")]
    H = dense_hamiltonian()
    assert np.allclose(H, H.conj().T)
    assert H[0, 0] == 2 and H[1, 0] == 0.7


@pytest.mark.parametrize("order", [1, 2])
def test_commuting_cases_are_exact(order):
    for J, h in [(1, 0), (0, 0.7), (0, 0)]:
        actual = Operator(product_formula_circuit(3, 0.71, 3, order, J, h)).data
        assert np.allclose(actual, exact_evolution(3, 0.71, J, h), atol=1e-12)


def test_second_order_improves_operator_and_state():
    first = [compare_product_formula(steps=r, order=1) for r in [8, 16, 32]]
    second = [compare_product_formula(steps=r, order=2) for r in [8, 16, 32]]
    for key in ["operator_error", "state_vector_error"]:
        assert all(a[key] > b[key] for a, b in zip(first, second))
        assert second[0][key] > second[1][key] > second[2][key]
    assert 1.8 < first[1]["operator_error"] / first[2]["operator_error"] < 2.2
    assert 3.8 < second[1]["operator_error"] / second[2]["operator_error"] < 4.2
    assert second[-1]["operator_error"] < 0.005


def test_diagonal_observable_can_hide_boundary_phase_error():
    # Initial |000> is a ZZ eigenstate and final ZZ phases commute with Z(q0).
    # Hence these two splittings have equal Z probabilities, despite different
    # operator and state errors; no observable-only validation is sufficient.
    for r in [1, 2, 8, 32]:
        first = compare_product_formula(steps=r, order=1)
        second = compare_product_formula(steps=r, order=2)
        assert first["circuit_observable"] == pytest.approx(second["circuit_observable"], abs=1e-12)


def test_decomposition_and_measured_export():
    source = product_formula_circuit(3, 1, 3, 2)
    compiled = decompose(source)
    assert np.allclose(Operator(source).data, Operator(compiled).data, atol=1e-12)
    measured = decompose(product_formula_circuit(3, 1, 3, 2, measure=True))
    text = qasm3.dumps(measured)
    assert "measure" in text and "gate " not in text
    assert measured.count_ops()["measure"] == 3


def test_basis_preparation_and_backward_time():
    prepared = product_formula_circuit(3, 0, 1, initial_basis=5)
    assert np.argmax(abs(Operator(prepared).data[:, 0])) == 5
    forward = Operator(product_formula_circuit(3, 0.43, 4, 2)).data
    backward = Operator(product_formula_circuit(3, -0.43, 4, 2)).data
    assert np.allclose(backward @ forward, np.eye(8), atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"n_qubits": True}, {"n_qubits": 65}, {"steps": 0},
    {"steps": True}, {"steps": 100_000}, {"order": 3}, {"order": 2.0}, {"time": float("nan")},
    {"h": float("inf")}, {"initial_basis": 8}])
def test_invalid_or_unbounded_construction(kwargs):
    with pytest.raises(ValueError):
        product_formula_circuit(**kwargs)


def test_generation_is_separate_from_dense_validation():
    assert product_formula_circuit(n_qubits=32, steps=2).num_qubits == 32
    with pytest.raises(ValueError, match="dense"):
        dense_hamiltonian(9)


def test_zero_hamiltonian_does_not_bypass_loop_bound():
    with pytest.raises(ValueError, match="steps"):
        product_formula_circuit(n_qubits=1, J=0, h=0, steps=10**9)


def test_statistical_budget():
    assert observable_shots() == 13315
    with pytest.raises(ValueError):
        observable_shots(statistical_failure=0)
