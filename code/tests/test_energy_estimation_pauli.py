import math
from itertools import product
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator,Statevector
from scipy.linalg import expm
from openquantum import energy_estimation_pauli as pauli
from openquantum.energy_estimation import H2_STO3G,reference_matrix
from openquantum.qft_qpe import build_qpe


@pytest.mark.parametrize('label',['I','X','Y','Z','IX','YZ','XYZ','YXI','YYYY','XXYY'])
def test_actual_pauli_exponential_has_exact_matrix_semantics(label):
    c=.31;t=-.7
    matrix=pauli.pauli_matrix(label)
    expected=math.cos(c*t)*np.eye(matrix.shape[0])-1j*math.sin(c*t)*matrix
    np.testing.assert_allclose(Operator(pauli.pauli_exponential(c,label,t)).data,expected,atol=2e-15,rtol=0)


def test_identity_phase_is_preserved_when_the_access_is_controlled():
    unitary=pauli.pauli_exponential(.43,'II',.7)
    expected=np.eye(8,dtype=complex)
    expected[np.ix_([1,3,5,7],[1,3,5,7])]*=np.exp(-1j*.43*.7)
    np.testing.assert_allclose(Operator(unitary.to_gate().control(1)).data,expected,atol=3e-15,rtol=0)


@pytest.mark.parametrize('order',[1,2])
@pytest.mark.parametrize('steps',[1,2,4,8])
def test_product_formula_is_independently_reconstructed_and_bound_exceeds_error(order,steps):
    terms=((.3,'XI'),(-.7,'YY'),(.2,'IZ'),(.1,'XY'),(-.11,'II'))
    t=.9
    actual=Operator(pauli.product_formula_unitary(terms,time=t,steps=steps,order=order)).data
    reference=pauli.reference_product_formula(terms,time=t,steps=steps,order=order)
    exact=expm(-1j*t*pauli.reference_hamiltonian(terms))
    np.testing.assert_allclose(actual,reference,atol=2e-14,rtol=0)
    error=float(np.linalg.norm(actual-exact,2))
    bound=pauli.product_formula_error_bound(terms,time=t,steps=steps,order=order)
    assert error<=bound['analytic_operator_error_bound']+2e-14
    assert bound['includes_floating_gate_or_synthesis_error'] is False


@pytest.mark.parametrize('seed',range(10))
def test_nested_commutator_norms_hold_for_noncommuting_random_pauli_models(seed):
    rng=np.random.default_rng(seed)
    labels=['XI','YX','IZ','ZZ','XY']
    terms=tuple((float(c),p) for c,p in zip(rng.uniform(-.7,.7,5),labels,strict=True))
    t=.1+seed/31
    for order in [1,2]:
        actual=Operator(pauli.product_formula_unitary(terms,time=t,steps=2,order=order)).data
        exact=expm(-1j*t*pauli.reference_hamiltonian(terms))
        error=np.linalg.norm(actual-exact,2)
        bound=pauli.product_formula_error_bound(terms,time=t,steps=2,order=order)
        assert error<=bound['analytic_operator_error_bound']+3e-14


@pytest.mark.parametrize('order',[1,2])
def test_commuting_diagonal_terms_have_zero_analytic_error_bound(order):
    terms=((.1,'II'),(.3,'IZ'),(-.7,'ZI'),(.2,'ZZ'))
    expected=expm(-1j*.9*pauli.reference_hamiltonian(terms))
    actual=Operator(pauli.product_formula_unitary(terms,time=.9,steps=3,order=order)).data
    np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0)
    assert pauli.product_formula_error_bound(terms,time=.9,steps=3,order=order)['analytic_operator_error_bound']==0


def test_symmetric_schedule_inverse_and_empirical_second_order_rate():
    terms=((.7,'X'),(.3,'Z'))
    actual=Operator(pauli.product_formula_unitary(terms,time=.9,steps=4,order=2)).data
    reverse=Operator(pauli.product_formula_unitary(terms,time=-.9,steps=4,order=2)).data
    np.testing.assert_allclose(reverse,actual.conj().T,atol=1e-14,rtol=0)
    exact=expm(-1j*.9*pauli.reference_hamiltonian(terms))
    errors=[np.linalg.norm(Operator(pauli.product_formula_unitary(terms,time=.9,steps=r,order=2)).data-exact,2) for r in [4,8,16]]
    assert 3.95<errors[0]/errors[1]<4.05 and 3.95<errors[1]/errors[2]<4.05


def test_full_jw_fixture_matches_reduced_physical_sector_with_nuclear_offset():
    electronic=pauli.reference_hamiltonian(pauli.h2_jw_terms(include_nuclear=False))
    restricted=electronic[np.ix_(pauli.H2_SECTOR_INDICES,pauli.H2_SECTOR_INDICES)]
    total=restricted+pauli.H2_NUCLEAR_REPULSION*np.eye(4)
    np.testing.assert_allclose(total,reference_matrix(H2_STO3G),atol=1e-15,rtol=0)
    assert total[1,1].real==pytest.approx(-1.1169989967540035,abs=2e-14)
    assert np.linalg.eigvalsh(total)[0]==pytest.approx(-1.1373060357533995,abs=2e-14)
    assert pauli.H2_JW_PROVENANCE['fresh_scf_or_mapping_executed_here'] is False


@pytest.mark.parametrize('order,steps',[(1,1),(1,4),(2,1),(2,4),(2,8)])
def test_actual_qpe_measures_approximate_unitary_and_keeps_H_accuracy_separate(order,steps):
    result=pauli.compare_pauli_energy_qpe(pauli.h2_jw_terms(),order=order,steps=steps,sector_indices=pauli.H2_SECTOR_INDICES)
    assert result['distribution_max_error']<5e-11
    assert result['access_gate_vs_product_reference_error']<1e-11
    assert result['access_operator_error_against_exact_H']<=result['analytic_product_formula_bound']['analytic_operator_error_bound']+1e-11
    assert result['ground_energy_bias_numeric']<=result['spectral_neighbourhood_energy_bound_ideal_angles']+1e-12
    assert result['ground_confidence_guarantee'] is None
    assert result['true_H_hf_ground_overlap_reference']==pytest.approx(.9875597343674857,abs=1e-13)
    assert result['phase_grid_energy_width']> .9


def test_removed_coherent_identity_phase_changes_qpe_output():
    terms=pauli.h2_jw_terms();unitary=pauli.build_pauli_energy_unitary(terms,steps=1)
    wrong=unitary.copy();wrong.global_phase=0
    preparation=QuantumCircuit(4);preparation.x([0,2])
    correct=Statevector.from_instruction(build_qpe(3,unitary,preparation)).probabilities(qargs=[0,1,2])
    shifted=Statevector.from_instruction(build_qpe(3,wrong,preparation)).probabilities(qargs=[0,1,2])
    assert np.max(np.abs(correct-shifted))>.3


def test_identity_offset_changes_reported_energy_without_altering_encoded_phase():
    original=pauli.h2_jw_terms();changed=list(original)
    changed[0]=(changed[0][0]+.7,changed[0][1])
    np.testing.assert_allclose(Operator(pauli.build_pauli_energy_unitary(original,steps=1)).data,
                               Operator(pauli.build_pauli_energy_unitary(changed,steps=1)).data,atol=1e-13,rtol=0)
    assert pauli.energy_encoding(changed)['shift']-pauli.energy_encoding(original)['shift']==pytest.approx(.7)


def test_qasm_ready_lowering_retains_readout_and_equivalent_probabilities():
    source=pauli.build_pauli_energy_qpe(pauli.h2_jw_terms(),steps=1,phase_bits=2)
    lowered=pauli.lower_circuit(source)
    assert set(lowered.count_ops())<=set(pauli.ROTATION_BASIS)|{'measure','barrier'}
    assert lowered.count_ops()['measure']==2
    original_map=[(source.find_bit(i.qubits[0]).index,source.find_bit(i.clbits[0]).index) for i in source.data if i.operation.name=='measure']
    lowered_map=[(lowered.find_bit(i.qubits[0]).index,lowered.find_bit(i.clbits[0]).index) for i in lowered.data if i.operation.name=='measure']
    assert original_map==lowered_map==[(0,0),(1,1)]
    actual=Statevector.from_instruction(source.remove_final_measurements(inplace=False)).probabilities(qargs=[0,1])
    expected=Statevector.from_instruction(lowered.remove_final_measurements(inplace=False)).probabilities(qargs=[0,1])
    np.testing.assert_allclose(actual,expected,atol=5e-12,rtol=0)


def test_bound_fallback_for_large_term_lists_is_explicit_and_finite():
    labels=[format(i,'07b').translate(str.maketrans({'0':'I','1':'Z'})) for i in range(1,66)]
    terms=tuple((.01,p) for p in labels)
    result=pauli.product_formula_error_bound(terms,time=.4,steps=2,order=2)
    assert 'nested bound not computed' in result['method']
    assert result['analytic_operator_error_bound']==0


def test_large_noncommuting_list_obeys_conservative_symmetric_fallback():
    labels=[''.join(label) for label in product('IXYZ',repeat=4) if ''.join(label)!='IIII'][:65]
    terms=tuple(((-1)**i*(i%5+1)/100,p) for i,p in enumerate(labels))
    actual=Operator(pauli.product_formula_unitary(terms,time=.31,steps=3,order=2)).data
    exact=expm(-.31j*pauli.reference_hamiltonian(terms))
    error=np.linalg.norm(actual-exact,2)
    result=pauli.product_formula_error_bound(terms,time=.31,steps=3,order=2)
    assert 'nested bound not computed' in result['method']
    assert 1e-8<error<=result['analytic_operator_error_bound']+3e-14


def test_duplicate_terms_are_aggregated_without_tolerance_pruning():
    assert pauli.normalize_terms(((.125,'X'),(.25,'X'),(-.375,'X'),(.2,'Y')))==((.2,'Y'),)
    near=pauli.normalize_terms(((.1,'X'),(.2,'X'),(-.3,'X')))
    assert near[0][0]!=0 and near[0][1]=='X'
    zero=pauli.normalize_terms(((0.,'XX'),))
    assert zero==((0.,'II'),)
    assert len(pauli.product_formula_unitary(zero,time=1).data)==0


@pytest.mark.parametrize('terms',[[],[(True,'X')],[(1j,'X')],[(float('nan'),'X')],[(1,'A')],[(1,'X'),(1,'YY')],[(1,'X'*25)],[(.1,'I')]*257])
def test_invalid_inputs_rejected_before_circuit_work(terms):
    with pytest.raises(ValueError):pauli.normalize_terms(terms)


@pytest.mark.parametrize('kwargs',[{'steps':0},{'steps':True},{'steps':4097},{'order':3},{'time':float('inf')},{'time':101}])
def test_product_formula_bounds_are_separate_from_reference_limits(kwargs):
    options={'time':.2};options.update(kwargs)
    with pytest.raises(ValueError):pauli.product_formula_unitary(((.3,'X'),(.2,'Z')),**options)


def test_dense_references_reject_large_width_before_matrix_allocation(monkeypatch):
    def poisoned(*a,**kw):raise AssertionError('dense allocation must not happen')
    monkeypatch.setattr(pauli.np,'eye',poisoned)
    with pytest.raises(ValueError):pauli.reference_product_formula(((.1,'X'*7),),time=.1)
    with pytest.raises(ValueError):pauli.reference_hamiltonian(((.1,'X'*7),))
    with pytest.raises(ValueError):pauli.compare_pauli_energy_qpe(((.1,'X'*7),),initial_basis=0)


def test_primitive_qpe_preflight_rejects_unbounded_precision():
    with pytest.raises(ValueError,match='preflight'):
        pauli.build_pauli_energy_qpe(pauli.h2_jw_terms(),steps=8,phase_bits=10)


@pytest.mark.parametrize('sector',[[],[1,1],[True,2],[16,5],[6,10,9]])
def test_invalid_or_noncontaining_sector_is_rejected(sector):
    with pytest.raises(ValueError):pauli.compare_pauli_energy_qpe(pauli.h2_jw_terms(),steps=1,sector_indices=sector)


@pytest.mark.parametrize('label',['I','X','Y','Z','XYZ','YYYY'])
def test_termwise_control_cancels_data_operations_on_inactive_branch(label):
    c=-.27;t=.43
    matrix=math.cos(c*t)*np.eye(2**len(label))-1j*math.sin(c*t)*pauli.pauli_matrix(label)
    expected=np.eye(2**(len(label)+1),dtype=complex)
    selected=list(range(1,expected.shape[0],2))
    expected[np.ix_(selected,selected)]=matrix
    np.testing.assert_allclose(Operator(pauli.controlled_pauli_exponential(c,label,t)).data,
                               expected,atol=4e-15,rtol=0)


@pytest.mark.parametrize('order',[1,2])
def test_termwise_full_access_keeps_global_identity_and_matches_lifted_unitary(order):
    terms=pauli.h2_jw_terms()
    exact=Operator(pauli.build_pauli_energy_unitary(terms,steps=2,order=order)).data
    expected=np.eye(32,dtype=complex);expected[np.ix_(list(range(1,32,2)),list(range(1,32,2)))]=exact
    np.testing.assert_allclose(Operator(pauli.controlled_pauli_energy_access(terms,steps=2,order=order)).data,
                               expected,atol=3e-14,rtol=0)


def test_termwise_and_lifted_qpe_have_same_output_with_different_primitive_costs():
    terms=pauli.h2_jw_terms()
    efficient=pauli.build_pauli_energy_qpe(terms,steps=2,phase_bits=3,measure=False)
    lifted=pauli.build_pauli_energy_qpe(terms,steps=2,phase_bits=3,measure=False,control_strategy='lift-all-gates')
    p=Statevector.from_instruction(efficient).probabilities(qargs=[0,1,2])
    q=Statevector.from_instruction(lifted).probabilities(qargs=[0,1,2])
    np.testing.assert_allclose(p,q,atol=1e-12,rtol=0)
    assert len(pauli.lower_circuit(efficient).data)<len(pauli.lower_circuit(lifted).data)/4
    with pytest.raises(ValueError):pauli.build_pauli_energy_qpe(terms,control_strategy='unrecognized')


def test_accidental_product_formula_sector_invariance_does_not_validate_H_sector():
    terms=((1.,'IX'),(1.,'IY'),(-1.22273057506641,'ZX'),(-.6432833515501427,'ZY'))
    encoding=pauli.energy_encoding(terms);tau=2*math.pi/encoding['span']
    unitary=pauli.reference_product_formula(terms,time=-tau,steps=1,order=1)
    hamiltonian=pauli.reference_hamiltonian(terms)
    assert abs(unitary[1,0])<1e-12
    assert abs(hamiltonian[1,0])>.4
    with pytest.raises(ValueError,match='Hamiltonian does not preserve'):
        pauli.compare_pauli_energy_qpe(terms,phase_bits=2,steps=1,order=1,initial_basis=0,sector_indices=(0,))


def test_ideal_spectral_sample_contract_does_not_claim_ground_selection():
    terms=pauli.h2_jw_terms()
    contract=pauli.spectral_sample_contract(terms,phase_bits=3,steps=8)
    assert contract['analytic_ideal_angle_contract_applies']
    assert contract['one_shot_ideal_success_lower_bound']==pytest.approx(8/math.pi**2)
    assert contract['ground_energy_selection_guarantee'] is None
    result=pauli.compare_pauli_energy_qpe(terms,phase_bits=3,steps=8,sector_indices=pauli.H2_SECTOR_INDICES)
    eigen=np.linalg.eigvalsh(pauli.reference_hamiltonian(terms)[np.ix_(pauli.H2_SECTOR_INDICES,pauli.H2_SECTOR_INDICES)])
    encoding=pauli.energy_encoding(terms);energies=encoding['shift']+encoding['span']*np.arange(8)/8
    selected=np.min(np.abs(energies[:,None]-eigen[None,:]),axis=1)<=contract['energy_error_bound_ideal']
    assert np.asarray(result['phase_distribution'])[selected].sum()>=8/math.pi**2-1e-12
    coarse=pauli.spectral_sample_contract(terms,phase_bits=2,steps=1)
    assert not coarse['analytic_ideal_angle_contract_applies']
    assert coarse['energy_error_bound_ideal'] is None
