"""Independent truth-table and coherent clean-workspace checks for SAT search."""
import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from openquantum.grover_sat import (
    CNF, amplification_probability, build_cnf_phase_oracle,
    build_sat_grover_circuit, compute_clause_register, exhaustive_solutions,
    replay_recorded_attempts, verify_candidate, build_amplitude_amplification_circuit,
    preflight_sat_circuit,
)

CORPUS = [
    (CNF(1, ((1,),)), {1}),
    (CNF(1, ((-1,),)), {0}),
    (CNF(2, ()), {0,1,2,3}),
    (CNF(2, ((),)), set()),
    (CNF(2, ((1,-1),)), {0,1,2,3}),
    (CNF(2, ((2,2),)), {2,3}),
    (CNF(2, ((1,-2),)), {0,1,3}),
    (CNF(3, ((1,2),(-1,3),(-2,-3))), {2,5}),
    (CNF(3, ((1,),(2,),(3,))), {7}),
    (CNF(4, ((1,2),(-1,3),(-2,-3),(4,))), {10,13}),
]


@pytest.mark.parametrize("cnf,solutions", CORPUS)
def test_classical_verifier_agrees_with_independent_expected_assignments(cnf, solutions):
    assert set(exhaustive_solutions(cnf)) == solutions
    for assignment in range(2**cnf.n_variables):
        assert cnf.satisfies(assignment) == (assignment in solutions)
        result=verify_candidate(cnf,assignment)
        assert result["verified"] == (assignment in solutions)
        assert result["classical_verification_cpu_ns"] >= 0


@pytest.mark.parametrize("cnf,solutions", CORPUS)
def test_oracle_relative_phase_and_full_uncompute_on_coherent_data(cnf, solutions):
    n=cnf.n_variables
    rng=np.random.default_rng(391)
    data=rng.normal(size=2**n)+1j*rng.normal(size=2**n)
    data/=np.linalg.norm(data)
    initial=np.zeros(2**cnf.num_qubits,dtype=complex)
    initial[:2**n]=data
    expected=initial.copy()
    for assignment in solutions:
        expected[assignment]*=-1
    oracle=build_cnf_phase_oracle(cnf)
    actual=Statevector(initial).evolve(oracle)
    # Direct amplitudes retain relative phase; basis-wise equiv would miss it.
    np.testing.assert_allclose(actual.data,expected,atol=1e-12,rtol=0)
    assert np.sum(actual.probabilities()[:2**n]) == pytest.approx(1)
    assert oracle.metadata["clause_qubits"] == len(cnf.clauses)


def test_computed_clause_bits_match_literal_truths_and_inverse():
    cnf=CNF(3,((1,2),(-1,3),(2,2),(1,-1),()))
    compute=compute_clause_register(cnf)
    for assignment in range(8):
        bits={i+1:bool(assignment&(1<<i)) for i in range(3)}
        truth=[any(bits[abs(l)] if l>0 else not bits[abs(l)] for l in clause) for clause in cnf.clauses]
        clause_register=sum(int(value)<<i for i,value in enumerate(truth))
        initial=Statevector.from_int(assignment,2**cnf.num_qubits)
        actual=initial.evolve(compute)
        expected=Statevector.from_int(assignment+(clause_register<<3),2**cnf.num_qubits)
        np.testing.assert_allclose(actual.data,expected.data,atol=1e-12)
        np.testing.assert_allclose(actual.evolve(compute.inverse()).data,initial.data,atol=1e-12)


def test_dirty_scratch_is_not_the_clean_oracle_contract():
    cnf=CNF(1,((1,),))
    oracle=build_cnf_phase_oracle(cnf)
    # Data x=0, scratch superposition: scratch-dependent relative phase changes.
    initial=np.array([1,0,1,0],dtype=complex)/math.sqrt(2)
    expected=np.array([1,0,-1,0],dtype=complex)/math.sqrt(2)
    np.testing.assert_allclose(Statevector(initial).evolve(oracle).data,expected,atol=1e-12)
    assert "start in zero" in oracle.metadata["workspace_promise"]


@pytest.mark.parametrize("cnf,solutions", CORPUS)
def test_multiple_marked_amplification_matches_count_reference(cnf,solutions):
    for iterations in range(5):
        circuit=build_sat_grover_circuit(cnf,iterations)
        state=Statevector.from_instruction(circuit)
        probabilities=state.probabilities()
        success=sum(probabilities[x] for x in solutions)
        expected=amplification_probability(cnf.n_variables,len(solutions),iterations)
        assert success == pytest.approx(expected,abs=1e-11)
        assert sum(probabilities[:2**cnf.n_variables]) == pytest.approx(1,abs=1e-11)
        assert not circuit.metadata["known_marked_count_used"]


@pytest.mark.parametrize("args", [
    (0,((1,),)), (True,()), (2,((0,),)), (2,((True,),)),
    (2,((.5,),)), (2,((3,),)), (2,((-3,),)), (2,("bad",)),
])
def test_invalid_literal_contracts_are_rejected(args):
    with pytest.raises(ValueError):
        CNF(*args)


def test_scalable_generator_and_bounded_diagnostic_are_separate():
    cnf=CNF(10,((1,-2,3),(-3,4),(5,-6),(7,8),(-9,10),(2,)))
    circuit=build_sat_grover_circuit(cnf,2,measure=True)
    assert circuit.num_qubits==23
    assert cnf.num_clean_work_qubits==7
    assert circuit.num_clbits==10
    assert circuit.count_ops().get("measure")==10
    assert not any(item.operation.name in {"unitary","initialize","reset"} for item in circuit.data)
    with pytest.raises(ValueError,match="variable bound"):
        exhaustive_solutions(cnf,max_variables=8)


def test_replay_banks_deferred_measurements_and_product_distribution():
    cnf=CNF(2,((1,),(2,)))
    replay=replay_recorded_attempts(cnf,[0,1])
    assert replay.num_qubits==8 and replay.num_clbits==4
    seen_measure=False
    for item in replay.data:
        if item.operation.name=="measure":
            seen_measure=True
        else:
            assert not seen_measure
            if item.operation.name!="barrier":
                banks={replay.find_bit(q).index//cnf.num_qubits for q in item.qubits}
                assert len(banks)==1
    mapping=[(replay.find_bit(item.qubits[0]).index,replay.find_bit(item.clbits[0]).index)
             for item in replay.data if item.operation.name=="measure"]
    assert mapping==[(0,0),(1,1),(4,2),(5,3)]
    first=Statevector.from_instruction(build_sat_grover_circuit(cnf,0)).probabilities(qargs=[0,1])
    second=Statevector.from_instruction(build_sat_grover_circuit(cnf,1)).probabilities(qargs=[0,1])
    joint=Statevector.from_instruction(replay.remove_final_measurements(inplace=False)).probabilities(qargs=[0,1,4,5])
    np.testing.assert_allclose(joint,np.kron(second,first),atol=1e-12)
    with pytest.raises(ValueError):
        replay_recorded_attempts(cnf,[])


@pytest.mark.parametrize("cnf", [
    CNF(5,((1,-2,3,-4,5),)),
    CNF(2,((1,-1),(2,-2),(1,),(2,),(1,2))),
    CNF(5,()),
])
def test_long_conjunction_and_diffuser_restore_clean_pool_coherently(cnf):
    rng=np.random.default_rng(442)
    data=rng.normal(size=2**cnf.n_variables)+1j*rng.normal(size=2**cnf.n_variables)
    data/=np.linalg.norm(data)
    initial=np.zeros(2**cnf.num_qubits,dtype=complex)
    initial[:len(data)]=data
    expected=initial.copy()
    for x in range(len(data)):
        if cnf.satisfies(x):
            expected[x]*=-1
    oracle=build_cnf_phase_oracle(cnf)
    assert not any(item.operation.name=="mcx" for item in oracle.data)
    np.testing.assert_allclose(Statevector(initial).evolve(oracle).data,expected,atol=1e-12,rtol=0)
    # Independent full-state iteration, retaining the implemented global sign.
    trial=build_sat_grover_circuit(cnf,2)
    reference=np.ones(len(data),dtype=complex)/math.sqrt(len(data))
    signs=np.array([-1 if cnf.satisfies(x) else 1 for x in range(len(data))])
    for _ in range(2):
        reference*=signs
        reference-=2*np.mean(reference)
    target=np.zeros(2**cnf.num_qubits,dtype=complex)
    target[:len(data)]=reference
    np.testing.assert_allclose(Statevector.from_instruction(trial).data,target,atol=1e-12,rtol=0)


def test_generic_oracle_gets_clean_diffuser_work_and_preserves_global_phase():
    from qiskit import QuantumCircuit
    oracle=QuantumCircuit(5)
    oracle.z(0)
    trial=build_amplitude_amplification_circuit(oracle,5,1)
    assert trial.num_qubits==7 and trial.metadata["added_diffuser_work_qubits"]==2
    expected=np.zeros(128,dtype=complex)
    expected[:32]=[(-1 if x&1 else 1)/math.sqrt(32) for x in range(32)]
    np.testing.assert_allclose(Statevector.from_instruction(trial).data,expected,atol=1e-12,rtol=0)


def test_linear_chain_materialization_is_bounded_separately_from_search():
    cnf=CNF(15,(tuple(range(1,16)),))
    info=preflight_sat_circuit(cnf,2)
    circuit=build_sat_grover_circuit(cnf,2,measure=True)
    assert cnf.num_qubits==29
    assert set(circuit.count_ops())<={"h","x","z","cx","ccx","measure"}
    assert info["estimated_primitive_operations"]<4000
    with pytest.raises(ValueError,match="preflight"):
        preflight_sat_circuit(cnf,1000)
    with pytest.raises(ValueError,match="32-qubit"):
        preflight_sat_circuit(CNF(20,(tuple(range(1,21)),)),1)
