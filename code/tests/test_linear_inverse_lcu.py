# SPDX-License-Identifier: MIT
"""Independent inverse-polynomial, gate, conditioning and amplification checks."""
from fractions import Fraction
import math
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from openquantum.linear_inverse_lcu import (
    DEFAULT_TERMS, build_neumann_lcu, compare_inverse, default_preparation,
    degree_for_pauli_bias, measured_inverse_trial, neumann_coefficients,
    prepared_inverse, structured_inverse_terms, zero_reflection,
)
from openquantum.linear_systems import decompose

PAULIS={"I":np.eye(2),"X":np.array([[0,1],[1,0]]),
        "Y":np.array([[0,-1j],[1j,0]]),"Z":np.diag([1,-1])}


def test_tiny_gap_uses_exact_bias_target_and_minimal_bounded_degree():
    gap=Fraction(1,10**18)
    allowance=math.nextafter(2.,0.)
    target=Fraction(str(allowance))
    threshold=2*target/(2+target)
    assert (1-gap)**50>threshold
    assert (1-gap)**51<=threshold
    assert degree_for_pauli_bias(gap,allowance)==50


def test_tiny_gap_with_strict_accuracy_reports_degree_cap():
    with pytest.raises(ValueError,match="degree64"):
        degree_for_pauli_bias(Fraction(1,10**18),1.)


def matrix(terms):
    result=np.zeros((1<<len(terms[0][1]),)*2,complex)
    for coefficient,word in terms:
        p=np.array([[1]],complex)
        for char in word:p=np.kron(p,PAULIS[char])
        result+=float(Fraction(str(coefficient)))*p
    return result


@pytest.mark.parametrize("degree",[0,1,2,4,8,12,32,64])
@pytest.mark.parametrize("terms",[
    DEFAULT_TERMS, ((1,"I"),("1/4","Y")),
    (("4/5","II"),("-1/10","XY"),("1/20","YZ"),("1/25","ZX")),
])
def test_exact_pauli_coefficients_match_independent_matrix_polynomial(terms,degree):
    model=neumann_coefficients(terms,degree)
    A=matrix(terms);B=A/float(model.alpha);C=np.eye(len(A))-B
    expected=np.eye(len(A),dtype=complex);power=expected.copy()
    for _ in range(degree):power=power@C;expected+=power
    expected*=float(model.delta)
    np.testing.assert_allclose(matrix(model.coefficients),expected,atol=3e-15,rtol=0)
    err=np.linalg.norm(expected-float(model.delta)*np.linalg.inv(B),ord=2)
    assert err<=float(model.tail_bound)+4e-15
    assert model.delta<=model.normalization<=1-model.tail_bound
    assert all(isinstance(coefficient,Fraction) for coefficient,_ in model.coefficients)


@pytest.mark.parametrize("degree",[0,1,2,4,12])
def test_actual_lcu_block_matches_polynomial_with_complete_access(degree):
    model=neumann_coefficients(DEFAULT_TERMS,degree);circuit=build_neumann_lcu(DEFAULT_TERMS,degree)
    block=Operator(circuit).data[:4,:4]
    np.testing.assert_allclose(block,matrix(model.coefficients)/float(model.normalization),atol=4e-14,rtol=0)
    assert circuit.metadata["analytical_scope"].startswith("exact rational")
    assert circuit.metadata["coefficient_conversion_l1_error_bound"]<1e-15


@pytest.mark.parametrize("gap",["1/2","1/4","3/4"])
@pytest.mark.parametrize("degree",[2,8,16])
def test_complex_input_inverse_matches_bounds_over_changed_gap(gap,degree):
    terms=structured_inverse_terms(2,gap)
    result=compare_inverse(terms,degree,default_preparation(2))
    assert len(set(np.round(result["matrix_eigenvalues"],10)))==4
    assert result["scaled_inverse_operator_error"]<=result["analytic_operator_error_bound"]+1e-14
    assert result["conditional_state_trace_distance_to_classical_solution"]<=result["analytic_conditional_trace_distance_bound"]+1e-13
    assert result["z_absolute_bias"]<=result["analytic_conditional_Pauli_bias_bound"]+1e-13
    assert result["actual_condition_number"]<=1/float(Fraction(gap))


@pytest.mark.parametrize("rounds",[0,1,2,3,4])
def test_known_input_amplification_preserves_inverse_direction(rounds):
    result=compare_inverse(DEFAULT_TERMS,8,default_preparation(),rounds)
    assert result["success_probability"]==pytest.approx(result["amplification_prediction"],abs=2e-13)
    assert result["conditional_state_trace_distance_to_unamplified_polynomial"]<2e-13
    circuit=prepared_inverse(DEFAULT_TERMS,8,default_preparation(),rounds)
    assert circuit.metadata["full_source_calls"]==2*rounds+1
    assert circuit.metadata["initial_zero_reflections"]==rounds
    assert circuit.metadata["input_preparation_counts"]=={"ry":2,"rz":1}


@pytest.mark.parametrize("observable",["X","Y","Z"])
def test_complete_basis_readout_matches_independent_target(observable):
    prep=default_preparation();model=neumann_coefficients(DEFAULT_TERMS,12)
    circuit=measured_inverse_trial(DEFAULT_TERMS,12,prep,1,observable)
    state=Statevector.from_instruction(circuit.remove_final_measurements(inplace=False)).data[:4]
    state/=np.linalg.norm(state)
    b=Statevector.from_instruction(prep).data
    target=matrix(model.coefficients)@b;target/=np.linalg.norm(target)
    expected=np.vdot(target,np.kron(np.eye(2),PAULIS[observable])@target).real
    assert sum(abs(state[i])**2*(1-2*(i&1)) for i in range(4))==pytest.approx(expected,abs=2e-13)
    assert circuit.count_ops()["measure"]==circuit.num_qubits


@pytest.mark.parametrize("targets",[[],[0],[1,2],[0,1,2]])
def test_reflections_are_correct_without_dense_oracle(targets):
    actual=Operator(zero_reflection(3,targets)).data
    expected=np.diag([-1 if all(not((index>>q)&1) for q in targets) else 1 for index in range(8)])
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)


@pytest.mark.parametrize("gap,allowance",[("1/2",.001),("1/4",.005),("3/4",.0001),(1,.001)])
def test_automatic_degree_meets_exact_uniform_pauli_bias(gap,allowance):
    degree=degree_for_pauli_bias(gap,allowance)
    E=(1-Fraction(gap))**(degree+1)
    assert 2*E/(2-E)<=Fraction(str(allowance))
    if degree:
        old=(1-Fraction(gap))**degree
        assert 2*old/(2-old)>Fraction(str(allowance))


def test_scalar_inverse_and_decomposition_boundary():
    model=neumann_coefficients(((3,"I"),),8)
    assert model.delta==1 and model.tail_bound==0 and model.coefficients==((Fraction(1),"I"),)
    c=prepared_inverse(((3,"I"),),8,rounds=2)
    assert abs(Statevector.from_instruction(c).data[0])**2==pytest.approx(1)
    original=prepared_inverse(DEFAULT_TERMS,4,default_preparation(),1)
    assert Operator(original).equiv(Operator(decompose(original)),atol=3e-13)


def test_generation_and_bounded_validation_are_separate():
    c=build_neumann_lcu(structured_inverse_terms(24),12)
    assert c.num_qubits==27
    with pytest.raises(ValueError,match="capped"):compare_inverse(structured_inverse_terms(8),8)
    with pytest.raises(ValueError,match="degree64"):degree_for_pauli_bias("1/100",.001)
    with pytest.raises(ValueError,match="degree64"):degree_for_pauli_bias("1/100000000",.001)
    with pytest.raises(ValueError):zero_reflection(100,[])


@pytest.mark.parametrize("terms",[
    ((1,"X"),), (("1/2","I"),("1/2","X")), ((-1,"I"),),
    ((True,"I"),), ((float("nan"),"I"),), ((1j,"I"),),
    ((1,"I"),(1,"XX")), ((1,"Q"),), (("1e1000","I"),),
])
def test_invalid_matrix_access_rejected(terms):
    with pytest.raises(ValueError):neumann_coefficients(terms)


def test_degree_round_and_nonunitary_input_guards():
    with pytest.raises(ValueError):neumann_coefficients(degree=65)
    with pytest.raises(ValueError):neumann_coefficients(degree=True)
    with pytest.raises(ValueError):prepared_inverse(rounds=5)
    bad=QuantumCircuit(2);bad.reset(0)
    with pytest.raises(ValueError):prepared_inverse(preparation=bad)


def test_rational_literal_and_derived_precision_guards_precede_large_arithmetic():
    with pytest.raises(ValueError,match="exponents"):
        neumann_coefficients((("1e-100000000","I"),),0)
    with pytest.raises(ValueError,match="literal"):
        neumann_coefficients((("1"*193,"I"),),0)
    with pytest.raises(ValueError,match="512-bit"):
        neumann_coefficients(((Fraction(1,1<<1000),"I"),),0)
    tiny_gap=Fraction((1<<200)-1,1<<200)
    with pytest.raises(ValueError,match="128-bit"):
        neumann_coefficients(((1,"I"),(tiny_gap,"X")),0)
    huge_norm=Fraction(1<<511,1)
    model=neumann_coefficients(((huge_norm,"I"),),1)
    assert model.as_metadata()["promised_condition_upper"]==1
    assert compare_inverse(((Fraction(1,1<<511),"I"),),1)["classical_residual_norm"]<1e-13
    unresolved=Fraction((1<<100)-1,1<<100)
    with pytest.raises(ValueError,match="numerically resolved"):
        compare_inverse(((1,"I"),(unresolved,"X")),0)
    for scale in ("1e128","1e-128"):
        result=compare_inverse(((scale,"I"),(str(Fraction(scale)/4),"Y")),4)
        assert math.isfinite(result["actual_condition_number"])
        assert math.isfinite(result["conditional_state_trace_distance_to_classical_solution"])
        assert result["classical_residual_norm"]<1e-13
