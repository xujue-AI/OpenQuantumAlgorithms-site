# SPDX-License-Identifier: MIT
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from openquantum.linear_systems_qsvt import (
    DEFAULT_TERMS, INVERSE_PHASES, build_inverse_qsvt, compare_inverse_qsvt,
    fit_inverse_phases, inverse_polynomial, pauli_word_inverse_terms,
    qsvt_measured_trial, reciprocal_interval_error_bound, verify_inverse_phases,
)


def test_cubic_algebra_and_global_bound():
    grid = np.linspace(-1, 1, 10001)
    assert np.max(np.abs(inverse_polynomial(grid))) <= 5 * math.sqrt(15) / 36 + 1e-12
    values = np.array([-1, -0.5, 0.5, 1])
    np.testing.assert_allclose(inverse_polynomial(values), 0.25 / values, atol=1e-15)
    np.testing.assert_allclose(inverse_polynomial(-grid), -inverse_polynomial(grid), atol=1e-15)


@pytest.mark.parametrize("delta", [0.1, 0.5, 0.8, 1.0])
def test_interval_bound_does_not_claim_arbitrary_inverse(delta):
    grid = np.linspace(delta, 1, 10001)
    error = np.max(np.abs(inverse_polynomial(grid) - 0.25 / grid))
    assert error <= reciprocal_interval_error_bound(delta) + 1e-14
    assert reciprocal_interval_error_bound(0.5) > 0.18


def test_frozen_phases_and_bounded_reproduction():
    report = verify_inverse_phases()
    assert report["coefficient_l1_discrepancy"] < 1e-12
    assert report["dense_grid_max_phase_error"] < 1e-12
    phases, evidence = fit_inverse_phases()
    assert evidence["successful_restart_zero_based"] == 2
    assert verify_inverse_phases(phases)["dense_grid_max_phase_error"] < 1e-12


@pytest.mark.parametrize("terms", [DEFAULT_TERMS, ((0.7,"I"),(0.2,"Y"),(0.1,"Z")), ((0.75,"II"),(-0.25,"XY")), ((-0.75,"I"),(0.25,"X"))])
def test_actual_accepted_block_matches_independent_matrix_polynomial(terms):
    circuit = build_inverse_qsvt(terms)
    matrix = SparsePauliOp.from_list([(word,c) for c,word in terms]).to_matrix() / sum(abs(c) for c, _ in terms)
    target = (5 * matrix - 4 * matrix @ matrix @ matrix) / 4
    width = len(terms[0][1])
    block = Operator(circuit).data[: 2**width, : 2**width]
    np.testing.assert_allclose(block, target, atol=3e-14, rtol=0)


def test_default_matches_classical_solution_and_analytic_probability():
    result = compare_inverse_qsvt()
    assert result["acceptance_probability"] == pytest.approx(5 / 32, abs=1e-14)
    assert result["accepted_branch_error_vs_cubic"] < 1e-14
    assert result["solution_state_phase_invariant_error"] < 1e-14
    assert result["z_actual"] == pytest.approx(0.8, abs=1e-14)
    np.testing.assert_allclose(result["solution_state_real"], np.array([3, 1]) / math.sqrt(10), atol=1e-14)


@pytest.mark.parametrize("word", ["X", "Y", "ZX", "YY", "XYZ"])
def test_structured_pauli_family_and_actual_preparation(word):
    prep = QuantumCircuit(len(word))
    prep.h(0)
    prep.s(0)
    result = compare_inverse_qsvt(pauli_word_inverse_terms(word), prep)
    assert result["exact_spectral_contract_numerically_met"]
    assert result["solution_state_phase_invariant_error"] < 1e-13
    assert result["accepted_branch_error_vs_cubic"] < 1e-13


@pytest.mark.parametrize("basis", ["X", "Y", "Z"])
def test_readout_contains_all_work_and_success_ancillas(basis):
    circuit = qsvt_measured_trial(observable=basis)
    assert circuit.count_ops()["measure"] == circuit.num_qubits == 3
    assert circuit.metadata["block_encoding_calls"] == 3
    assert circuit.metadata["projector_phase_calls"] == 4
    compiled = transpile(circuit, basis_gates=["h","x","z","s","sdg","t","tdg","rx","ry","rz","cx"], optimization_level=1, seed_transpiler=7)
    assert not set(compiled.count_ops()) - {"h","x","z","s","sdg","t","tdg","rx","ry","rz","cx","measure","barrier"}
    unitary = circuit.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(unitary).data[:2]
    probability = float(np.vdot(state,state).real)
    actual = (abs(state[0])**2-abs(state[1])**2)/probability
    expected = {"X":0.6,"Y":0.0,"Z":0.8}[basis]
    assert actual == pytest.approx(expected, abs=1e-13)


def test_width_generation_is_separate_from_dense_simulation():
    terms = pauli_word_inverse_terms("X"*24)
    circuit = build_inverse_qsvt(terms)
    assert circuit.num_qubits == 26
    with pytest.raises(ValueError,match="6 data"):
        compare_inverse_qsvt(terms)


@pytest.mark.parametrize("terms", [(), ((1,""),), ((1j,"X"),), ((float("inf"),"X"),), ((1,"A"),), ((0,"X"),), ((1,"X"),(1,"ZZ"))])
def test_invalid_matrix_access_rejected(terms):
    with pytest.raises(ValueError):
        build_inverse_qsvt(terms)


def test_preparation_and_other_invalid_inputs_rejected():
    prep=QuantumCircuit(1);prep.reset(0)
    with pytest.raises(ValueError,match="unitary"):
        compare_inverse_qsvt(preparation=prep)
    with pytest.raises(ValueError):
        build_inverse_qsvt(phases=INVERSE_PHASES[:-1])
    with pytest.raises(ValueError):
        qsvt_measured_trial(observable="XX")
    with pytest.raises(ValueError):
        pauli_word_inverse_terms("II")
    with pytest.raises(ValueError):
        reciprocal_interval_error_bound(0)
    with pytest.raises(ValueError):
        inverse_polynomial(1.1)


def test_generic_spectrum_is_labeled_as_cubic_not_exact_inverse():
    result=compare_inverse_qsvt(((0.8,"I"),(0.2,"X")))
    assert not result["exact_spectral_contract_numerically_met"]
    assert result["accepted_branch_error_vs_cubic"] < 1e-13
    assert result["solution_state_trace_distance"] > 0.03


@pytest.mark.parametrize("kind", ["reset", "measurement", "feedback", "nan", "infinite_global_phase"])
def test_nested_preparation_operations_and_invalid_angles_are_rejected(kind):
    inner=QuantumCircuit(1,1) if kind in {"measurement","feedback"} else QuantumCircuit(1)
    if kind=="reset":
        inner.reset(0)
    elif kind=="measurement":
        inner.measure(0,0)
    elif kind=="feedback":
        with inner.if_test((inner.clbits[0], True)):
            inner.x(0)
    elif kind=="nan":
        inner.ry(float("nan"),0)
    else:
        inner.global_phase=float("inf")
    # A wrapped definition is inspected even if its outer instruction has a
    # harmless name. Ordinary classical metadata rejection may act first.
    from qiskit.circuit import Instruction
    wrapped=Instruction("hidden-preparation",1,0,[])
    wrapped.definition=inner
    prep=QuantumCircuit(1)
    prep.append(wrapped,[0])
    with pytest.raises(ValueError,match="preparation"):
        qsvt_measured_trial(preparation=prep)


def test_top_level_nonfinite_preparation_angle_is_rejected():
    prep=QuantumCircuit(1)
    prep.rz(float("nan"),0)
    with pytest.raises(ValueError,match="finite"):
        qsvt_measured_trial(preparation=prep)


def test_nested_bound_unitary_preparation_is_allowed():
    inner=QuantumCircuit(1,name="complex-input")
    inner.h(0)
    inner.rz(0.37,0)
    prep=QuantumCircuit(1)
    prep.append(inner.to_gate(),[0])
    result=compare_inverse_qsvt(preparation=prep)
    assert result["solution_state_trace_distance"] < 1e-13


def test_opaque_non_gate_preparation_instruction_is_rejected():
    from qiskit.circuit import Instruction
    prep=QuantumCircuit(1)
    prep.append(Instruction("unverified-operation",1,0,[]),[0])
    with pytest.raises(ValueError,match="unitary"):
        qsvt_measured_trial(preparation=prep)
