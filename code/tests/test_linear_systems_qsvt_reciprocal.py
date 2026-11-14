# SPDX-License-Identifier: MIT
from fractions import Fraction
import math
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation,UnitaryGate
from qiskit.quantum_info import Operator,Statevector

import openquantum.linear_systems_qsvt_reciprocal as inverse
from openquantum.reciprocal_phase_certificate import certify_real_qsp_phases
from openquantum.simulation.qsp import qsp_real_response


def independent_matrix(terms):
    matrices = {"I":np.eye(2),"X":np.array([[0,1],[1,0]]),
                "Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}
    result = np.zeros((1<<len(terms[0][1]),)*2,dtype=complex)
    alpha = sum(abs(float(c)) for c,w in terms)
    for coefficient,word in terms:
        matrix = np.array([[1]],complex)
        for letter in word: matrix = np.kron(matrix,matrices[letter])
        result += float(coefficient)/alpha*matrix
    return result


@pytest.mark.parametrize("k",range(32))
def test_all_supported_polynomials_phases_and_exact_bounds(k):
    model = inverse.ReciprocalPolynomial(k,Fraction(1,2))
    assert model.degree==2*k+1
    assert model.m*model.scale**2<=1
    assert model.relative_tail==Fraction(3,4)**(k+1)
    coefficients = inverse.exact_chebyshev_coefficients(k)
    assert len(coefficients)==2*k+2
    assert all(c==0 for c in coefficients[::2])
    phases,record = inverse.fit_reciprocal_phases(k)
    assert record["function_evaluations"]<=500
    certificate = certify_real_qsp_phases(phases,coefficients)
    assert certificate.realization_error_bound<Fraction(1,10**11)
    x = np.linspace(-1,1,301)
    expected = inverse.reciprocal_polynomial(x,k)
    assert max(abs(expected))<=1+1e-14
    assert np.max(abs(qsp_real_response(x,phases)-expected))<2e-12
    interval = x[abs(x)>=.5]
    assert np.max(abs(inverse.reciprocal_polynomial(interval,k)-float(model.scale)/interval))<=float(model.scale*model.relative_tail/model.delta)+1e-14


def test_exact_default_degree_selection_and_cost():
    model = inverse.ReciprocalPolynomial()
    assert inverse.choose_k()==16
    assert inverse.choose_k("1/2","1/50")==13
    assert model.scale==Fraction(1,5)
    assert model.success_lower_bound==Fraction(1,25)
    assert model.pauli_bias_bound<Fraction(1,100)
    assert inverse.ReciprocalPolynomial(15).pauli_bias_bound>Fraction(1,100)
    contract = inverse.build_reciprocal_qsvt().metadata
    p = Fraction(contract["retry_success_lower_bound_exact"])
    assert 0<p<Fraction(1,25)
    assert contract["phase_adjusted_Pauli_bias_bound"]<.01
    workload = inverse.observable_workload(p)
    assert (workload["accepted_samples"],workload["retry_cap_per_slot"],workload["total_capped_attempts"])==(13811,403,5565833)
    assert workload["hoeffding_bound_certified"]
    assert workload["verified_union_empty_slot_bound"]<.001
    assert Fraction.from_float(workload["uncapped_expected_attempts_upper"])>=13811/p


@pytest.mark.parametrize("terms",[
    inverse.DEFAULT_TERMS,((7,"ZI"),(2,"IZ"),(1,"XX")),
    ((7,"YI"),(2,"IX"),(1,"ZZ")),((3,"I"),(-1,"Y")),
    inverse.active_chain_terms(3),
])
def test_actual_complex_prepared_branches_and_signed_inverse(terms):
    n = len(terms[0][1])
    prep = inverse.default_preparation(n)
    b = Statevector.from_instruction(prep).data
    circuit = inverse.prepared_reciprocal(terms,preparation=prep)
    matrix = independent_matrix(terms)
    values,vectors = np.linalg.eigh(matrix)
    expected = vectors@(inverse.reciprocal_polynomial(values)*(vectors.conj().T@b))
    branch = Statevector.from_instruction(circuit).data[:1<<n]
    assert np.linalg.norm(branch-expected)<1e-12
    result = inverse.compare_reciprocal(terms,preparation=prep)
    assert result["spectral_promise_numerically_met"]
    assert result["solution_trace_distance"]<=result["analytic_trace_bound"]+1e-11
    assert result["z_absolute_bias"]<=result["analytic_Pauli_bias_bound"]+1e-11
    assert result["success_probability"]>=result["analytic_success_lower_bound"]-1e-11


@pytest.mark.parametrize("observable",["X","Y","Z"])
def test_full_actual_readout_and_coherent_lowering(observable):
    terms = ((3,"I"),(-1,"Y"))
    trial = inverse.measured_reciprocal_trial(terms,k=2,observable=observable)
    measured = trial.remove_final_measurements(inplace=False)
    state = Statevector.from_instruction(measured).data
    n = 1
    branch = state[:1<<n]
    actual = (abs(branch[0])**2-abs(branch[1])**2)/np.linalg.norm(branch)**2
    source = inverse.prepared_reciprocal(terms,k=2)
    branch_source = Statevector.from_instruction(source).data[:2]
    branch_source /= np.linalg.norm(branch_source)
    pauli = {"X":np.array([[0,1],[1,0]]),"Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}[observable]
    assert abs(actual-np.vdot(branch_source,pauli@branch_source).real)<1e-12
    lowered = inverse.lower_reciprocal(measured)
    assert Operator(source).num_qubits==trial.num_qubits
    assert np.linalg.norm(Operator(lowered).data-Operator(measured).data)<1e-11
    assert trial.count_ops()["measure"]==trial.num_qubits


def test_jacobian_independent_central_difference():
    phases = np.array(inverse.DEFAULT_PHASES)
    x = np.linspace(-.99,.99,17)
    response,jacobian = inverse.response_and_jacobian(x,phases)
    assert np.max(abs(response-qsp_real_response(x,phases)))<1e-14
    for j in (0,1,7,16,33):
        change = np.zeros(len(phases));change[j]=1e-6
        finite = (qsp_real_response(x,phases+change)-qsp_real_response(x,phases-change))/(2e-6)
        assert np.max(abs(finite-jacobian[:,j]))<1e-8


@pytest.mark.parametrize("n",[1,2,3,4,5])
def test_every_chain_spin_active_with_analytic_gap(n):
    terms = inverse.active_chain_terms(n)
    assert len(terms)==2*n
    assert sum(abs(c) for c,w in terms)==4*(2*n-1)
    values = np.linalg.eigvalsh(independent_matrix(terms))
    assert min(values)>=.5-1e-12 and max(values)<=1+1e-12
    for index in range(n):
        assert any(word[index]!="I" for c,word in terms)


def test_custom_wrong_phases_and_wrong_certificate_fail():
    with pytest.raises(ValueError,match="realization"):
        inverse.build_reciprocal_qsvt(phases=[0]*34)
    certificate = certify_real_qsp_phases((0.,0.),(Fraction(0),Fraction(1)))
    with pytest.raises(ValueError,match="match"):
        inverse.phase_adjusted_bounds(inverse.ReciprocalPolynomial(),certificate)


def test_bad_spectral_promise_rejected_and_large_reference_separate():
    with pytest.raises(ValueError,match="promise"):
        inverse.compare_reciprocal(((99,"I"),(98,"X")))
    # Generation never diagonalizes; an unsimulated promise is labeled assumed.
    circuit = inverse.build_reciprocal_qsvt(inverse.active_chain_terms(12),k=1)
    assert circuit.metadata["spectral_promise_verified"] is False
    with pytest.raises(ValueError,match="limited"):
        inverse.compare_reciprocal(inverse.active_chain_terms(12),k=1)


@pytest.mark.parametrize("k",[True,-1,32,1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError): inverse.ReciprocalPolynomial(k)


@pytest.mark.parametrize("delta",[0,-1,2,True,"nan","1e-9999"])
def test_invalid_gap(delta):
    with pytest.raises(ValueError): inverse.choose_k(delta)


def test_impossible_accuracy_rejected_without_huge_power():
    with pytest.raises(ValueError,match="bounded"):
        inverse.choose_k("1/1000000","1/1000000")
    with pytest.raises(ValueError): inverse.active_chain_terms(25)
    with pytest.raises(ValueError): inverse.build_reciprocal_qsvt(((1,"X"*25),))
    with pytest.raises(ValueError): inverse.build_reciprocal_qsvt(((1e308,"I"),(1e308,"X")))


@pytest.mark.parametrize("kind",["unitary","loader","hidden_reset"])
def test_dense_and_nonunitary_preparation_rejected(kind):
    prep = QuantumCircuit(2)
    if kind=="unitary": prep.append(UnitaryGate(np.eye(4)),range(2))
    elif kind=="loader": prep.append(StatePreparation([1.,0.,0.,0.]),range(2))
    else:
        leaf = QuantumCircuit(2);leaf.reset(0)
        prep.append(leaf.to_instruction(),range(2))
    with pytest.raises(ValueError): inverse.prepared_reciprocal(preparation=prep)


def test_nested_preparation_counts_expanded_multiplicity(monkeypatch):
    leaf = QuantumCircuit(2)
    for _ in range(11): leaf.h(0)
    prep = QuantumCircuit(2)
    gate = leaf.to_gate()
    for _ in range(11): prep.append(gate,range(2))
    monkeypatch.setattr(inverse,"MAX_SOURCE_OPERATIONS",100)
    with pytest.raises(ValueError,match="expanded"):
        inverse.prepared_reciprocal(((1,"II"),),k=0,preparation=prep)


def test_sampling_boundary_exact_certificate_and_probability_edges():
    failure = Fraction("0.164169997247797590339057348934")
    result = inverse.observable_workload(1,Fraction(1,2),failure)
    assert result["accepted_samples"]==21
    assert result["retry_cap_per_slot"]==1
    near_one = Fraction((1<<100)-1,1<<100)
    result = inverse.observable_workload(near_one)
    assert result["retry_cap_per_slot"]==1
    with pytest.raises(ValueError,match="128-bit"):
        inverse.observable_workload(Fraction(1,1<<200))
    with pytest.raises(ValueError,match="100000"):
        inverse.observable_workload(Fraction(1,10**8))
    with pytest.raises(ValueError): inverse.observable_workload(sampling_error=0)


def test_export_provenance_never_borrows_enclosing_head(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("reciprocal_generator",root/"scripts/generate_linear_systems_qsvt_reciprocal.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.source_revision(root/"algorithms") is None
    assert module.source_revision(tmp_path) is None
    revision = module.source_revision(root)
    assert revision is None or len(revision)==40
