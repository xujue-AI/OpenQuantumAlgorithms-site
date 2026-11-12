# SPDX-License-Identifier: MIT
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from openquantum.qft_qpe import (build_qft, qft_matrix, qft_omission_bound,
    build_qpe, build_iterative_qpe, phase_gate_example, qpe_distribution,
    simulate_qpe, dynamic_distribution, lower_circuit, median_repetitions,
    qpe_ledger)


@pytest.mark.parametrize("n", range(1, 7))
def test_qft_matches_independent_matrix(n):
    actual = Operator(build_qft(n)).data
    np.testing.assert_allclose(actual, qft_matrix(n), atol=1e-13, rtol=0)
    inverse = Operator(build_qft(n, inverse=True)).data
    np.testing.assert_allclose(inverse @ actual, np.eye(2**n), atol=1e-13, rtol=0)


@pytest.mark.parametrize("n,cutoff", [(n, b) for n in range(1, 6) for b in range(n)])
def test_approximation_bound_and_gate_omissions(n, cutoff):
    approximation = build_qft(n, max_distance=cutoff)
    expected_cp = sum(n - d for d in range(1, cutoff + 1))
    assert approximation.count_ops().get("cp", 0) == expected_cp
    error = np.linalg.norm(Operator(approximation).data - qft_matrix(n), ord=2)
    assert error <= qft_omission_bound(n, cutoff) + 1e-12
    inverse_error = np.linalg.norm(Operator(build_qft(n,inverse=True,max_distance=cutoff)).data-qft_matrix(n).conj().T, ord=2)
    assert inverse_error <= qft_omission_bound(n, cutoff) + 1e-12


@pytest.mark.parametrize("m,phase", [(m,p) for m in range(1,6) for p in (0.,.125,.3,.5,.875,.9999)])
def test_static_qpe_matches_dirichlet_distribution(m,phase):
    unitary, preparation = phase_gate_example(phase)
    actual = simulate_qpe(build_qpe(m, unitary, preparation), m)
    np.testing.assert_allclose(actual, qpe_distribution(phase,m), atol=2e-13, rtol=0)


@pytest.mark.parametrize("m,phase", [(m,p) for m in range(1,6) for p in (0.,.15625,.3,.5,.96875)])
def test_iterative_feedback_actual_instrument(m,phase):
    unitary, preparation = phase_gate_example(phase)
    instrument=build_iterative_qpe(m,unitary,preparation)
    actual=dynamic_distribution(instrument)
    np.testing.assert_allclose(actual,qpe_distribution(phase,m),atol=2e-13,rtol=0)
    compiled=lower_circuit(instrument)
    np.testing.assert_allclose(dynamic_distribution(compiled),actual,atol=2e-13,rtol=0)
    assert instrument.count_ops()["reset"]==m
    assert instrument.count_ops()["if_else"]==m*(m-1)//2 if m>1 else "if_else" not in instrument.count_ops()


def test_general_preparation_unitary_and_coherent_global_phase():
    # |+i> is the +1 eigenstate of Y; RY(theta) eigenphase is -theta/(4pi).
    unitary=QuantumCircuit(1)
    unitary.ry(-4*math.pi*.3,0)
    unitary.global_phase=2*math.pi*.125
    preparation=QuantumCircuit(1)
    preparation.h(0); preparation.s(0)
    expected=qpe_distribution(.425,4)
    np.testing.assert_allclose(simulate_qpe(build_qpe(4,unitary,preparation),4),expected,atol=2e-13,rtol=0)
    np.testing.assert_allclose(dynamic_distribution(build_iterative_qpe(4,unitary,preparation)),expected,atol=2e-13,rtol=0)


def test_non_eigenstate_is_phase_mixture():
    unitary, _ = phase_gate_example(.3)
    preparation=QuantumCircuit(1); preparation.h(0)
    expected=.5*qpe_distribution(0,4)+.5*qpe_distribution(.3,4)
    np.testing.assert_allclose(simulate_qpe(build_qpe(4,unitary,preparation),4),expected,atol=2e-13,rtol=0)
    np.testing.assert_allclose(dynamic_distribution(build_iterative_qpe(4,unitary,preparation)),expected,atol=2e-13,rtol=0)


def test_static_lowering_preserves_control_global_phase():
    unitary,preparation=phase_gate_example(.31)
    unitary.global_phase=.17
    source=build_qpe(3,unitary,preparation)
    compiled=lower_circuit(source)
    np.testing.assert_allclose(Operator(source).data,Operator(compiled).data,atol=2e-13,rtol=0)


@pytest.mark.parametrize("n", [0,-1,65,True,1.5])
def test_invalid_qft_size(n):
    with pytest.raises(ValueError): build_qft(n)


@pytest.mark.parametrize("m", [0,-1,17,True,1.5])
def test_invalid_phase_bits(m):
    u,p=phase_gate_example(.3)
    with pytest.raises(ValueError): build_qpe(m,u,p)


@pytest.mark.parametrize("phase", [-.1,1.,float("nan"),float("inf"),True])
def test_invalid_phase(phase):
    with pytest.raises(ValueError): qpe_distribution(phase,3)


@pytest.mark.parametrize("operation", ["unitary","measure","reset","initialize"])
def test_reject_opaque_or_dynamic_access(operation):
    u=QuantumCircuit(1,1 if operation=="measure" else 0)
    if operation=="unitary": u.unitary(np.eye(2),[0])
    elif operation=="measure": u.measure(0,0)
    elif operation=="reset": u.reset(0)
    else: u.initialize([1,0],[0])
    with pytest.raises(ValueError): build_qpe(2,u,QuantumCircuit(1))


def test_generator_and_simulation_limits_are_separate():
    u,p=phase_gate_example(.3)
    circuit=build_qpe(14,u,p)
    assert circuit.num_qubits==15
    with pytest.raises(ValueError): simulate_qpe(circuit,14)
    wider=QuantumCircuit(20); wider.h(range(20))
    with pytest.raises(ValueError): build_qpe(16,wider,QuantumCircuit(20))
    with pytest.raises(ValueError): dynamic_distribution(build_iterative_qpe(9,u,p))


def test_source_ledger_and_median_confidence():
    u,p=phase_gate_example(.3)
    ledger=qpe_ledger(4,u,p)
    assert ledger["controlled_unitary_calls_per_execution"]==15
    assert ledger["input_preparations_per_execution"]==1
    assert ledger["decomposed_gate_counts"]["measure"]==4
    reps=median_repetitions(.005)
    q=1-8/math.pi**2
    tail=sum(math.comb(reps,k)*q**k*(1-q)**(reps-k) for k in range((reps+1)//2,reps+1))
    assert reps%2==1 and tail<=.005
    assert median_repetitions(.005,0)==1


@pytest.mark.parametrize("target_qubits", [1,2,4,8])
def test_active_product_phase_scaling_has_declared_eigenphase(target_qubits):
    from openquantum.qft_qpe import phase_product_example
    u,p=phase_product_example(target_qubits,.3)
    actual=simulate_qpe(build_qpe(3,u,p),3)
    np.testing.assert_allclose(actual,qpe_distribution(.3,3),atol=2e-13,rtol=0)
