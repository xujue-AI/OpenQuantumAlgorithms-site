import math
import numpy as np
import pytest
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from openquantum import energy_estimation as energy


@pytest.mark.parametrize('time',[-3,-.4,0,.2,1,4])
def test_h2_gate_access_agrees_with_independent_matrix_exponential(time):
    actual=Operator(energy.dimer_evolution(energy.H2_STO3G,time)).data
    expected=expm(-1j*time*energy.reference_matrix(energy.H2_STO3G))
    np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)


@pytest.mark.parametrize('seed',range(12))
def test_general_parity_blocks_and_negative_coefficients(seed):
    values=np.random.default_rng(seed).uniform(-1,1,6)
    model=energy.DimerHamiltonian(*values)
    time=.13+seed/17
    matrix=energy.reference_matrix(model)
    np.testing.assert_allclose(Operator(energy.dimer_evolution(model,time)).data,
                               expm(-1j*time*matrix),atol=2e-14,rtol=0)
    np.testing.assert_allclose(model.analytic_spectrum_and_hf_overlap()['eigenvalues'],
                               np.linalg.eigvalsh(matrix),atol=1e-14,rtol=0)


@pytest.mark.parametrize('dimers',[1,2,3])
def test_every_dimer_is_active_and_encoded_identity_phase_is_coherent(dimers):
    model=energy.H2_STO3G
    encoding=energy.energy_encoding(model,dimers)
    matrix=energy.reference_matrix(model,dimers)
    expected=expm(1j*encoding.time*(matrix-encoding.shift*np.eye(matrix.shape[0])))
    actual=Operator(energy.build_energy_unitary(model,dimers)).data
    np.testing.assert_allclose(actual,expected,atol=4e-14,rtol=0)
    if dimers==1:
        controlled=Operator(energy.build_energy_unitary(model,dimers).to_gate().control(1)).data
        expected_controlled=np.eye(8,dtype=complex)
        expected_controlled[np.ix_([1,3,5,7],[1,3,5,7])]=expected
        np.testing.assert_allclose(controlled,expected_controlled,atol=4e-14,rtol=0)
    preparation=Statevector.from_instruction(energy.hartree_fock_preparation(dimers))
    target=sum(1<<(2*d) for d in range(dimers))
    assert preparation.probabilities()[target]==pytest.approx(1.)
    phases=encoding.phase(np.linalg.eigvalsh(matrix))
    assert phases.min()>=.25-1e-14 and phases.max()<=.75+1e-14


def test_hf_overlap_and_nuclear_offset_match_independent_diagonalization():
    model=energy.H2_STO3G
    matrix=energy.reference_matrix(model)
    eig,vec=np.linalg.eigh(matrix)
    spec=model.analytic_spectrum_and_hf_overlap()
    assert eig[0]==pytest.approx(-1.1373060357534,abs=1e-13)
    assert abs(vec[1,0])**2==pytest.approx(spec['hartree_fock_ground_overlap'])
    electronic=energy.DimerHamiltonian(model.electronic_identity,model.z0,model.z1,model.zz,model.xx)
    np.testing.assert_allclose(matrix-energy.reference_matrix(electronic),
                               model.nuclear_repulsion*np.eye(4),atol=1e-15,rtol=0)
    assert energy.H2_FIXTURE_PROVENANCE['fresh_scf_or_mapping_executed_here'] is False


@pytest.mark.parametrize('phase_bits,rounds',[(2,1),(3,1),(4,1),(2,3),(3,3)])
def test_full_gate_circuit_matches_retained_target_joint_spectral_law(phase_bits,rounds):
    actual=energy.simulate_complete_group(energy.H2_STO3G,phase_bits=phase_bits,rounds=rounds)
    expected=energy.joint_reference_distribution(energy.H2_STO3G,phase_bits=phase_bits,rounds=rounds)
    np.testing.assert_allclose(actual,expected,atol=3e-14,rtol=0)


def test_fresh_hf_marginal_product_is_not_retained_target_joint_law():
    model=energy.DimerHamiltonian(0,0,0,0,.4)
    reference=energy.reference_spectral_law(model,phase_bits=3)
    correct=energy.joint_reference_distribution(model,phase_bits=3,rounds=3)
    wrong=np.kron(reference['marginal'],np.kron(reference['marginal'],reference['marginal']))
    assert np.max(np.abs(correct-wrong))>.05
    np.testing.assert_allclose(energy.simulate_complete_group(model,phase_bits=3,rounds=3),
                               correct,atol=1e-13,rtol=0)


@pytest.mark.parametrize('dimers',[1,2,3,6,12])
def test_confidence_contract_charges_group_preparation_and_median_rounds(dimers):
    contract=energy.ground_energy_contract(energy.H2_STO3G,dimers,phase_bits=5)
    groups=contract['independent_groups'];rounds=contract['qpe_rounds_per_retained_target']
    overlap=contract['ground_overlap_lower_bound']
    assert contract['ideal_failure_union_bound']<=contract['requested_ideal_failure']
    assert contract['no_ground_group_failure_bound']==pytest.approx((1-overlap)**groups)
    if groups>1:assert (1-overlap)**(groups-1)>contract['requested_ideal_failure']/2
    assert contract['total_qpe_rounds']==groups*rounds
    assert contract['total_base_U_equivalent_calls']==groups*rounds*31
    assert rounds%2==1
    assert groups*energy.majority_failure(rounds)<=contract['requested_ideal_failure']/2
    assert contract['hardware_or_synthesis_confidence_certified'] is False
    assert overlap==pytest.approx(energy.H2_STO3G.analytic_spectrum_and_hf_overlap()['hartree_fock_ground_overlap']**dimers)


@pytest.mark.parametrize('phase_bits',[2,3,4,5])
def test_exact_reference_success_exceeds_theory_and_minimum_keeps_outlier_accounting(phase_bits):
    contract=energy.ground_energy_contract(energy.H2_STO3G,phase_bits=phase_bits)
    ref=energy.robust_energy_reference(energy.H2_STO3G,phase_bits=phase_bits,
        rounds=contract['qpe_rounds_per_retained_target'],groups=contract['independent_groups'])
    assert ref['ideal_success_within_grid_error']>=contract['ideal_confidence_lower_bound']-1e-12
    assert ref['probabilities'].sum()==pytest.approx(1)
    assert np.min(ref['probabilities'])>=0
    assert ref['error_tolerance']==pytest.approx(contract['energy_absolute_error_ideal'])


def test_actual_sampler_is_seeded_retains_target_and_does_not_use_reference(monkeypatch):
    expected=energy.sample_retained_target_groups(energy.H2_STO3G,phase_bits=3,rounds=3,groups=3,seed=17)
    def poisoned(*a,**kw):raise AssertionError('reference must not construct samples')
    monkeypatch.setattr(energy,'reference_matrix',poisoned)
    monkeypatch.setattr(energy,'reference_spectral_law',poisoned)
    actual=energy.sample_retained_target_groups(energy.H2_STO3G,phase_bits=3,rounds=3,groups=3,seed=17)
    assert actual==expected
    assert actual['simulated_width']==5
    assert actual['total_qpe_rounds']==9
    assert actual['estimate']==min(g['median_energy'] for g in actual['groups'])
    for g in actual['groups']:assert g['median_outcome']==int(np.median(g['outcomes']))


def test_lowering_preserves_joint_probabilities_and_readout_map():
    source=energy.build_retained_target_qpe(energy.H2_STO3G,phase_bits=2,rounds=3,measure=False)
    lowered=energy.lower_circuit(source)
    np.testing.assert_allclose(Statevector.from_instruction(source).data,
                               Statevector.from_instruction(lowered).data,atol=1e-13,rtol=0)
    measured=energy.build_retained_target_qpe(energy.H2_STO3G,phase_bits=2,rounds=3)
    assert measured.num_qubits==8 and measured.num_clbits==6
    assert measured.count_ops()['x']==1
    assert measured.count_ops()['measure']==6
    assert measured.metadata['controlled_unitary_calls']==9
    assert measured.metadata['hf_preparations']==1
    actual_map=[(measured.find_bit(item.qubits[0]).index,measured.find_bit(item.clbits[0]).index)
                for item in measured.data if item.operation.name=='measure']
    assert actual_map==[(i,i) for i in range(6)]


@pytest.mark.parametrize('args',[
 {'dimers':0},{'dimers':13},{'dimers':True},{'phase_bits':1},{'phase_bits':13},
 {'rounds':2},{'rounds':33},{'rounds':True},{'measure':1}, {'preparation':1},
 {'dimers':12,'phase_bits':12,'rounds':31},
])
def test_generator_guards_invalid_or_unbounded_work(args):
    with pytest.raises(ValueError):energy.build_retained_target_qpe(energy.H2_STO3G,**args)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),1e7])
def test_invalid_coefficients_fail_before_gate_construction(value):
    with pytest.raises(ValueError):energy.DimerHamiltonian(value,0,0,0,0)


def test_wrong_sector_ground_or_unproved_overlap_is_not_silently_retried():
    model=energy.DimerHamiltonian(0,0,0,-2,.1)
    assert model.analytic_spectrum_and_hf_overlap()['hartree_fock_ground_overlap']==0
    with pytest.raises(ValueError,match='overlap'):energy.ground_energy_contract(model)
    with pytest.raises(ValueError,match='exceeds'):energy.ground_energy_contract(energy.H2_STO3G,overlap_lower_bound=1)


def test_constant_model_has_exact_midpoint_energy_and_full_ground_overlap():
    model=energy.DimerHamiltonian(.3,0,0,0,0,.7)
    encoding=energy.energy_encoding(model)
    assert encoding.phase(1)==pytest.approx(.5)
    spec=model.analytic_spectrum_and_hf_overlap()
    assert spec['ground_energy']==pytest.approx(1.)
    assert spec['hartree_fock_ground_overlap']==1.
    reference=energy.reference_spectral_law(model,phase_bits=3)
    assert reference['marginal'][4]==pytest.approx(1)


def test_recycled_measure_reset_instrument_matches_static_deferred_joint_law():
    from openquantum.qft_qpe import dynamic_distribution
    instrument=energy.build_recycled_clock_qpe(energy.H2_STO3G,phase_bits=2,rounds=3)
    assert instrument.num_qubits==4 and instrument.num_clbits==6
    assert instrument.count_ops()['reset']==8
    assert instrument.count_ops()['measure']==6
    assert instrument.metadata['hf_preparations']==1
    np.testing.assert_allclose(dynamic_distribution(instrument),
        energy.simulate_complete_group(energy.H2_STO3G,phase_bits=2,rounds=3),atol=1e-13,rtol=0)
    lowered=energy.lower_circuit(instrument)
    np.testing.assert_allclose(dynamic_distribution(lowered),
        energy.joint_reference_distribution(energy.H2_STO3G,phase_bits=2,rounds=3),atol=1e-13,rtol=0)
    from openquantum.resource_estimation import prepare_measured_circuit
    with pytest.raises(ValueError,match='reset'):
        prepare_measured_circuit(instrument)


@pytest.mark.parametrize('power',[1,2,4,8,32,256,2048])
def test_analytic_power_access_retains_full_coherent_unitary(power):
    model=energy.H2_STO3G
    reference=Operator(energy.build_energy_unitary(model)).data
    actual=Operator(energy.build_energy_unitary(model,power=power)).data
    np.testing.assert_allclose(actual,np.linalg.matrix_power(reference,power),atol=4e-12,rtol=0)
    controlled=Operator(energy.build_energy_unitary(model,power=power).to_gate().control(1)).data
    expected=np.eye(8,dtype=complex)
    expected[np.ix_([1,3,5,7],[1,3,5,7])]=np.linalg.matrix_power(reference,power)
    np.testing.assert_allclose(controlled,expected,atol=4e-12,rtol=0)


def test_analytic_power_qpe_matches_repeated_access_at_identical_task_contract():
    repeated=energy.build_retained_target_qpe(energy.H2_STO3G,phase_bits=3,rounds=3,measure=False)
    powered=energy.build_retained_target_qpe(energy.H2_STO3G,phase_bits=3,rounds=3,measure=False,analytic_powers=True)
    np.testing.assert_allclose(Statevector.from_instruction(repeated).data,
                               Statevector.from_instruction(powered).data,atol=2e-13,rtol=0)
    assert repeated.num_qubits==powered.num_qubits
    assert repeated.metadata['base_U_equivalent_calls']==powered.metadata['base_U_equivalent_calls']==21
    assert powered.metadata['controlled_power_circuits']==9
    assert repeated.metadata['controlled_power_circuits']==21
    assert len(energy.lower_circuit(powered).data)<len(energy.lower_circuit(repeated).data)
