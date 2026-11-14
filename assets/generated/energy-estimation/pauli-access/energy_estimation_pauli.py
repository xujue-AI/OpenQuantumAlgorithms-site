# SPDX-License-Identifier: MIT
"""Generic explicit Pauli product formulas and energy phase estimation.

Every term exponential uses basis rotations, a CX parity ladder, RZ and
uncomputation. Dense matrices are bounded independent references only. The
stored four-spin-orbital H2 input is credited to Qiskit Nature; no SCF or
fermion-mapping calculation is silently rerun or treated as free access.
"""
from __future__ import annotations

from fractions import Fraction
import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm, schur

if __package__:
    from .qft_qpe import build_qpe,build_qft
else:
    from qft_qpe import build_qpe,build_qft

ROTATION_BASIS = ['h','x','z','s','sdg','t','tdg','rx','ry','rz','cx']
MAX_GENERATOR_QUBITS = 24
MAX_INPUT_TERMS = 256
MAX_STEPS = 4096
MAX_SOURCE_OPERATIONS = 100_000
MAX_PRIMITIVE_OPERATIONS = 500_000
MAX_PHASE_BITS = 10
MAX_REFERENCE_QUBITS = 6
MAX_STATEVECTOR_QUBITS = 12
MAX_SECOND_ORDER_BOUND_TERMS = 64

# The coefficients are exact stored stdout strings, not independently rerun
# integrals: docs/tutorials/06_qubit_mappers.ipynb zero-based cell 6, Qiskit Nature
# tag 0.7.2 commit d86d4a0306f8f4af8918c9eb9e3eae4622192f45. Qiskit labels q3 q2 q1 q0.
H2_JW_ELECTRONIC = (
    (-0.8105479805373283,'IIII'), (0.17218393261915543,'IIIZ'),
    (-0.2257534922240237,'IIZI'), (0.12091263261776633,'IIZZ'),
    (0.17218393261915543,'IZII'), (0.16892753870087907,'IZIZ'),
    (0.045232799946057826,'YYYY'), (0.045232799946057826,'XXYY'),
    (0.045232799946057826,'YYXX'), (0.045232799946057826,'XXXX'),
    (-0.22575349222402363,'ZIII'), (0.1661454325638241,'ZIIZ'),
    (0.16614543256382408,'IZZI'), (0.1746434306830045,'ZIZI'),
    (0.12091263261776633,'ZZII'),
)
H2_NUCLEAR_REPULSION = 0.7199689944489797
H2_SECTOR_INDICES = (6,5,10,9)
H2_HF_BASIS = 5
H2_JW_PROVENANCE = {
    'molecule':'H2','bond_length_angstrom':.735,'basis':'sto3g','charge':0,'spin':0,'method':'RHF',
    'spin_orbital_order':'alpha block then beta block','mapper':'JordanWignerMapper()',
    'pauli_label_order':'q3 q2 q1 q0','hf_occupation':[1,0,1,0],'hf_basis_integer':5,
    'fixed_alpha_beta_particle_numbers':[1,1],
    'source_repository':'qiskit-community/qiskit-nature','fixture_checkout_tag':'0.7.2',
    'fixture_checkout_commit':'d86d4a0306f8f4af8918c9eb9e3eae4622192f45',
    'source_notebook':'docs/tutorials/06_qubit_mappers.ipynb','source_stdout_cell':6,
    'source_notebook_sha256':'560c1b4c4863cabef7b2cf8b2d5975695bfc0782fd1cbed2156aed6b778374bd',
    'source_notebook_cell_indexing':'zero-based','fixture_release_date':'2024-02-23',
    'nuclear_source_notebook':'docs/tutorials/01_electronic_structure.ipynb','nuclear_source_stdout_cell':12,
    'nuclear_source_notebook_sha256':'d22a8646af6af5d83218c9351832e2b0f3e8dd21f6ae15896b205ab35d0bb33f',
    'stored_execution_qiskit_nature_version':'0.6.0','stored_execution_python_version':'3.9.16',
    'stored_execution_date':'2023-04-06','mapped_space_is_singlet_only':False,
    'pyscf_version_in_stored_execution':None,'fresh_scf_or_mapping_executed_here':False,
    'upstream_license':'Apache-2.0','nuclear_energy_is_separate_from_electronic_operator':True,
}


def _integer(value,name,minimum,maximum):
    if isinstance(value,bool) or not isinstance(value,int) or not minimum<=value<=maximum:
        raise ValueError(f'{name} must be an integer in [{minimum}, {maximum}]')
    return value


def _real(value,name,limit=1e4):
    if isinstance(value,bool) or not isinstance(value,Real) or not math.isfinite(value) or abs(value)>limit:
        raise ValueError(f'{name} must be finite, real and at most {limit} in magnitude')
    return float(value)


def normalize_terms(terms:Sequence[tuple[float,str]]):
    """Aggregate equal labels with math.fsum, preserving first occurrence.

    The resulting binary64 coefficients define the normalized input model.
    Analytic bounds do not include roundoff from an incoming duplicate-term
    representation. The verified H2 fixture has unique labels.
    """
    if not isinstance(terms,Sequence) or not 1<=len(terms)<=MAX_INPUT_TERMS:
        raise ValueError(f'supply 1 to {MAX_INPUT_TERMS} explicit Pauli terms')
    width=None; grouped={}
    for item in terms:
        if not isinstance(item,(tuple,list)) or len(item)!=2:
            raise ValueError('each term must be (real coefficient, Pauli label)')
        coefficient,label=item
        coefficient=_real(coefficient,'coefficient')
        if not isinstance(label,str) or not label or any(p not in 'IXYZ' for p in label):
            raise ValueError('labels must be nonempty I/X/Y/Z strings')
        if width is None: width=len(label)
        if len(label)!=width or width>MAX_GENERATOR_QUBITS:
            raise ValueError('Pauli labels must share a width at most24')
        grouped.setdefault(label,[]).append(coefficient)
    result=tuple((math.fsum(values),label) for label,values in grouped.items() if math.fsum(values)!=0)
    for coefficient,_ in result:_real(coefficient,'aggregate coefficient')
    return result if result else ((0.,'I'*width),)


def h2_jw_terms(*,include_nuclear=True):
    if type(include_nuclear) is not bool:raise ValueError('include_nuclear must be boolean')
    terms=list(H2_JW_ELECTRONIC)
    if include_nuclear:
        coefficient,label=terms[0]
        terms[0]=(coefficient+H2_NUCLEAR_REPULSION,label)
    return tuple(terms)


def pauli_exponential(coefficient,label,time):
    """Explicit exp(-i coefficient*time*P); identity is a coherent global phase."""
    coefficient,label=normalize_terms(((coefficient,label),))[0]
    time=_real(time,'time',100.)
    circuit=QuantumCircuit(len(label),name=f'exp({label})')
    if coefficient==0 or time==0:return circuit
    active=[wire for wire,p in enumerate(reversed(label)) if p!='I']
    if not active:
        circuit.global_phase=-coefficient*time
        return circuit
    for wire,p in enumerate(reversed(label)):
        if p=='X':circuit.h(wire)
        elif p=='Y':circuit.sdg(wire);circuit.h(wire)
    target=active[-1]
    for control in active[:-1]:circuit.cx(control,target)
    circuit.rz(2*coefficient*time,target)
    for control in reversed(active[:-1]):circuit.cx(control,target)
    for wire,p in enumerate(reversed(label)):
        if p=='X':circuit.h(wire)
        elif p=='Y':circuit.h(wire);circuit.s(wire)
    return circuit


def _term_gate_count(label):
    active=[p for p in label if p!='I']
    return (2*sum(1 if p=='X' else 2 if p=='Y' else 0 for p in active)
            +2*max(0,len(active)-1)+(1 if active else 0))


def product_formula_unitary(terms,*,time,steps=8,order=2):
    """A materialized Lie or symmetric Pauli product formula, without preparation."""
    terms=normalize_terms(terms);time=_real(time,'time',100.)
    _integer(steps,'steps',1,MAX_STEPS)
    _integer(order,'order',1,2)
    identity='I'*len(terms[0][1])
    scalar=sum(c for c,p in terms if p==identity)
    nonidentity=[(c,p) for c,p in terms if p!=identity]
    schedule=nonidentity if order==1 else [*((c/2,p) for c,p in nonidentity[:-1]),
             *nonidentity[-1:], *((c/2,p) for c,p in reversed(nonidentity[:-1]))]
    preflight=steps*sum(_term_gate_count(label) for _,label in schedule)
    if preflight>MAX_SOURCE_OPERATIONS:
        raise ValueError('Pauli source gate materialization exceeds limit')
    circuit=QuantumCircuit(len(identity),name=f'PF{order} r={steps}')
    circuit.global_phase=-scalar*time
    for _ in range(steps):
        for coefficient,label in schedule:
            circuit.compose(pauli_exponential(coefficient,label,time/steps),inplace=True)
    circuit.metadata={'steps':steps,'order':order,'time':time,'term_count':len(terms),
                      'source_operation_preflight':preflight,'identity_coefficient':scalar,
                      'actual_pauli_exponential_calls':steps*len(schedule)}
    return circuit


def _anti(left,right):
    return sum(a!='I' and b!='I' and a!=b for a,b in zip(left,right,strict=True))%2==1


def _product_label(left,right):
    """Pauli product label; scalar phase is irrelevant to commutator norm."""
    lookup={('X','Y'):'Z',('Y','X'):'Z',('Y','Z'):'X',('Z','Y'):'X',('Z','X'):'Y',('X','Z'):'Y'}
    return ''.join(b if a=='I' else a if b=='I' else 'I' if a==b else lookup[(a,b)]
                   for a,b in zip(left,right,strict=True))


def _upper_float(value:Fraction):
    result=float(value)
    return math.nextafter(result,math.inf) if Fraction.from_float(result)<value else result


def product_formula_error_bound(terms,*,time,steps,order):
    """Ideal-angle operator bound with exact binary64-coefficient rational sums.

    Lie: t²/(2r) sum_{j<k}||[Hj,Hk]||.
    Symmetric: t³/r² sum_j (sum_{k,l>j}||[Hk,[Hl,Hj]]||/12
                              +sum_{k>j}||[Hj,[Hj,Hk]]||/24).
    Triangle-expanded Pauli commutator norms avoid a dense matrix. For more
    than64 nonidentity terms, order2 uses a valid coarser O(1/r) two-half Lie
    bound, explicitly labeled. Numerical gate/angle and synthesis errors are
    outside this analytic product-formula bound.
    """
    terms=normalize_terms(terms);time=_real(time,'time',100.)
    _integer(steps,'steps',1,MAX_STEPS);_integer(order,'order',1,2)
    terms=[(Fraction.from_float(c),p) for c,p in terms if set(p)!={'I'} and c]
    t=abs(Fraction.from_float(time))
    pair=sum((2*abs(c*d) for j,(c,p) in enumerate(terms) for d,q in terms[j+1:] if _anti(p,q)),Fraction())
    if order==1:
        raw=t*t*pair/(2*steps);method='Lie commutator sum, O(t²/r)'
    elif len(terms)>MAX_SECOND_ORDER_BOUND_TERMS:
        raw=t*t*pair/(4*steps);method='two half-step Lie bounds, conservative O(t²/r); nested bound not computed'
    else:
        first=Fraction();second=Fraction()
        for j,(cj,pj) in enumerate(terms):
            tail=terms[j+1:]
            for ck,pk in tail:
                for cl,pl in tail:
                    if _anti(pl,pj) and _anti(pk,_product_label(pl,pj)):
                        first+=4*abs(ck*cl*cj)
                if _anti(pj,pk):second+=4*abs(cj*cj*ck)
        raw=t*t*t*(first/Fraction(12)+second/Fraction(24))/(steps*steps)
        method='symmetric nested-commutator triangle sum, O(t³/r²)'
    return {'analytic_operator_error_bound':min(2.,_upper_float(raw)),
            'uncapped_analytic_bound':_upper_float(raw),'method':method,
            'sum_commutator_norms':_upper_float(pair),'coefficient_sum_arithmetic':'exact Fraction of normalized binary64 input; incoming duplicate aggregation roundoff excluded; rounded upward on float export',
            'includes_floating_gate_or_synthesis_error':False,'formal_proof_artifact':None}


def energy_encoding(terms):
    terms=normalize_terms(terms);identity='I'*len(terms[0][1])
    center=math.fsum(c for c,p in terms if p==identity)
    beta=_upper_float(sum((abs(Fraction.from_float(c)) for c,p in terms if p!=identity),Fraction()))
    if beta==0:return {'shift':center-.5,'span':1.,'center':center,'pauli_norm_bound':0.}
    return {'shift':center-2*beta,'span':4*beta,'center':center,'pauli_norm_bound':beta}


def spectral_sample_contract(terms,*,phase_bits=3,steps=8,order=2):
    """Ideal one-shot accuracy near some true-H eigenvalue, not ground selection.

    Normal-unitary spectral residuals give chord distance<=eta, hence phase
    distance<=asin(eta/2)/pi. The true spectrum is centered in[1/4,3/4]. If
    the perturbation leaves a full1/M margin to the scalar branch cut, QPE's
    two-neighbour-bin event retains its8/pi² probability bound. The statement
    excludes floating gate errors and any physical/synthesis implementation.
    """
    _integer(phase_bits,'phase_bits',2,MAX_PHASE_BITS)
    encoding=energy_encoding(terms)
    bound=product_formula_error_bound(terms,time=-2*math.pi/encoding['span'],steps=steps,order=order)
    eta=bound['analytic_operator_error_bound']
    phase_bias=math.asin(eta/2)/math.pi if eta<2 else .5
    applies=phase_bias<=.25-1/2**phase_bits
    return {'kind':'one energy sample near some supplied-H eigenvalue',
            'analytic_ideal_angle_contract_applies':applies,
            'phase_grid_error':1/2**phase_bits,'spectral_phase_bias_bound':phase_bias,
            'energy_error_bound_ideal':encoding['span']*(1/2**phase_bits+phase_bias) if applies else None,
            'one_shot_ideal_success_lower_bound':8/math.pi**2 if applies else None,
            'ground_energy_selection_guarantee':None,'synthesis_or_physical_accuracy_certified':False,
            'branch_margin_condition':'spectral phase perturbation<=1/4-1/M',
            'exclusions':['floating coefficient/gate arithmetic beyond fixed normalized input',
                          'concrete gate synthesis errors','physical faults','ground-state overlap/selection']}


def build_pauli_energy_unitary(terms,*,steps=8,order=2):
    terms=normalize_terms(terms);encoding=energy_encoding(terms)
    tau=2*math.pi/encoding['span']
    circuit=product_formula_unitary(terms,time=-tau,steps=steps,order=order)
    circuit.global_phase-=tau*encoding['shift']
    circuit.metadata.update({'energy_encoding':encoding,'evolution_time_for_positive_phase':-tau})
    return circuit


def controlled_pauli_exponential(coefficient,label,time):
    """C(exp(-ictP)): data basis/parity gates run unconditionally and uncompute.

    Only the central RZ is controlled. On control0 the data gates cancel;
    controlling every data-basis gate is unnecessary. Identity terms apply P
    to the control, preserving the otherwise-global relative phase.
    """
    coefficient,label=normalize_terms(((coefficient,label),))[0]
    time=_real(time,'time',100.)
    circuit=QuantumCircuit(len(label)+1,name=f'c-exp({label})')
    if coefficient==0 or time==0:return circuit
    active=[wire+1 for wire,p in enumerate(reversed(label)) if p!='I']
    if not active:
        circuit.p(-coefficient*time,0)
        return circuit
    for wire,p in enumerate(reversed(label)):
        if p=='X':circuit.h(wire+1)
        elif p=='Y':circuit.sdg(wire+1);circuit.h(wire+1)
    target=active[-1]
    for control in active[:-1]:circuit.cx(control,target)
    circuit.crz(2*coefficient*time,0,target)
    for control in reversed(active[:-1]):circuit.cx(control,target)
    for wire,p in enumerate(reversed(label)):
        if p=='X':circuit.h(wire+1)
        elif p=='Y':circuit.h(wire+1);circuit.s(wire+1)
    return circuit


def controlled_product_formula_unitary(terms,*,time,steps=8,order=2):
    """Explicit controlled access using conditional term rotations."""
    terms=normalize_terms(terms);time=_real(time,'time',100.)
    _integer(steps,'steps',1,MAX_STEPS);_integer(order,'order',1,2)
    width=len(terms[0][1]);identity='I'*width
    scalar=math.fsum(c for c,p in terms if p==identity)
    nonidentity=[(c,p) for c,p in terms if p!=identity]
    schedule=nonidentity if order==1 else [*((c/2,p) for c,p in nonidentity[:-1]),
             *nonidentity[-1:],*((c/2,p) for c,p in reversed(nonidentity[:-1]))]
    preflight=steps*sum(_term_gate_count(p)+3 for _,p in schedule)+1
    if preflight>MAX_SOURCE_OPERATIONS:
        raise ValueError('controlled Pauli source materialization exceeds limit')
    circuit=QuantumCircuit(width+1,name=f'cPF{order} r={steps}')
    if scalar*time:circuit.p(-scalar*time,0)
    for _ in range(steps):
        for coefficient,label in schedule:
            circuit.compose(controlled_pauli_exponential(coefficient,label,time/steps),inplace=True)
    circuit.metadata={'steps':steps,'order':order,'time':time,'source_operation_preflight':preflight,
                      'actual_pauli_exponential_calls':steps*len(schedule),
                      'control_strategy':'termwise conditional central rotations; basis/parity gates unconditionally uncomputed'}
    return circuit


def controlled_pauli_energy_access(terms,*,steps=8,order=2):
    terms=normalize_terms(terms);encoding=energy_encoding(terms)
    tau=2*math.pi/encoding['span']
    circuit=controlled_product_formula_unitary(terms,time=-tau,steps=steps,order=order)
    if tau*encoding['shift']:circuit.p(-tau*encoding['shift'],0)
    return circuit


def build_pauli_energy_qpe(terms,*,phase_bits=3,steps=8,order=2,initial_basis=H2_HF_BASIS,measure=True,control_strategy='termwise'):
    terms=normalize_terms(terms);n=len(terms[0][1])
    _integer(phase_bits,'phase_bits',2,MAX_PHASE_BITS)
    _integer(initial_basis,'initial_basis',0,2**n-1)
    if type(measure) is not bool:raise ValueError('measure must be boolean')
    if control_strategy not in ('termwise','lift-all-gates'):
        raise ValueError('control_strategy must be termwise or lift-all-gates')
    unitary=build_pauli_energy_unitary(terms,steps=steps,order=order)
    controlled=(controlled_pauli_energy_access(terms,steps=steps,order=order).to_gate(label=f'PF{order} r={steps}')
                if control_strategy=='termwise' else unitary.to_gate().control(1))
    probe=QuantumCircuit(n+1);probe.append(controlled,range(n+1))
    controlled_count=len(transpile(probe,basis_gates=ROTATION_BASIS,optimization_level=1,seed_transpiler=7).data)
    preflight=(2**phase_bits-1)*controlled_count+6*phase_bits**2+n
    if preflight>MAX_PRIMITIVE_OPERATIONS:raise ValueError('controlled Pauli QPE expansion exceeds primitive preflight')
    circuit=QuantumCircuit(n+phase_bits,phase_bits if measure else 0,name='Pauli energy QPE')
    for wire in range(n):
        if (initial_basis>>wire)&1:circuit.x(phase_bits+wire)
    targets=list(range(phase_bits,phase_bits+n));circuit.h(range(phase_bits))
    for bit in range(phase_bits):
        for _ in range(2**bit):circuit.append(controlled,[bit,*targets])
    circuit.compose(build_qft(phase_bits,inverse=True),range(phase_bits),inplace=True)
    if measure:circuit.measure(range(phase_bits),range(phase_bits))
    circuit.metadata={'phase_bits':phase_bits,'data_qubits':n,'initial_basis':initial_basis,
                      'unitary_calls':2**phase_bits-1,'unitary_source_metadata':unitary.metadata,
                      'controlled_access_primitive_preflight':preflight,
                      'energy_encoding':energy_encoding(terms),'ground_confidence_guarantee':None,
                      'control_strategy':control_strategy,
                      'output_contract':'one sample of approximate-unitary eigenenergy distribution'}
    return circuit


def lower_circuit(circuit):
    lowered=transpile(circuit,basis_gates=ROTATION_BASIS,optimization_level=1,seed_transpiler=7)
    if len(lowered.data)>MAX_PRIMITIVE_OPERATIONS:raise ValueError('primitive export cap exceeded')
    if set(lowered.count_ops())-set(ROTATION_BASIS)-{'measure','barrier'}:
        raise RuntimeError('opaque operation remains after primitive lowering')
    return lowered


def pauli_matrix(label):
    if not label or len(label)>MAX_REFERENCE_QUBITS or any(p not in 'IXYZ' for p in label):
        raise ValueError('dense Pauli reference is limited to6 data wires')
    matrices={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Y':np.array([[0,-1j],[1j,0]]),'Z':np.diag([1,-1])}
    result=np.ones((1,1),complex)
    for p in label:result=np.kron(result,matrices[p])
    return result


def reference_hamiltonian(terms):
    terms=normalize_terms(terms)
    if len(terms[0][1])>MAX_REFERENCE_QUBITS:raise ValueError('dense Hamiltonian reference exceeds6 data wires')
    size=2**len(terms[0][1])
    return sum((c*pauli_matrix(p) for c,p in terms),np.zeros((size,size),complex))


def reference_product_formula(terms,*,time,steps=8,order=2):
    """Independent analytic Pauli matrix products in chronological circuit order."""
    terms=normalize_terms(terms);_real(time,'time',100.)
    if len(terms[0][1])>MAX_REFERENCE_QUBITS:
        raise ValueError('dense product reference exceeds6 data wires')
    _integer(steps,'steps',1,MAX_STEPS);_integer(order,'order',1,2)
    size=2**len(terms[0][1]);identity=np.eye(size,dtype=complex)
    scalar=math.fsum(c for c,p in terms if set(p)=={'I'})
    nonidentity=[(c,p) for c,p in terms if set(p)!={'I'}]
    schedule=nonidentity if order==1 else [*((c/2,p) for c,p in nonidentity[:-1]),
             *nonidentity[-1:],*((c/2,p) for c,p in reversed(nonidentity[:-1]))]
    one=identity.copy()
    for c,p in schedule:
        angle=c*time/steps
        one=(math.cos(angle)*identity-1j*math.sin(angle)*pauli_matrix(p))@one
    return np.exp(-1j*scalar*time)*np.linalg.matrix_power(one,steps)


def _phase_laws(phases,phase_bits):
    size=2**phase_bits;indices=np.arange(size)
    laws=np.array([np.abs(np.exp(2j*np.pi*np.outer(indices,phase-indices/size)).sum(axis=0)/size)**2
                   for phase in phases])
    return laws/laws.sum(axis=1,keepdims=True)


def compare_pauli_energy_qpe(terms,*,phase_bits=3,steps=8,order=2,initial_basis=H2_HF_BASIS,sector_indices=None,control_strategy='termwise'):
    """Circuit vs independent approximate-V spectral law; H error kept separate."""
    terms=normalize_terms(terms);n=len(terms[0][1]);encoding=energy_encoding(terms)
    _integer(phase_bits,'phase_bits',2,MAX_PHASE_BITS)
    if n>MAX_REFERENCE_QUBITS or n+phase_bits>MAX_STATEVECTOR_QUBITS:
        raise ValueError('bounded reference/statevector width exceeded')
    full_hamiltonian=reference_hamiltonian(terms)
    circuit=build_pauli_energy_qpe(terms,phase_bits=phase_bits,steps=steps,order=order,initial_basis=initial_basis,
                                 measure=False,control_strategy=control_strategy)
    if circuit.num_qubits>MAX_STATEVECTOR_QUBITS:raise ValueError('QPE statevector check exceeds12 total wires')
    actual_distribution=Statevector.from_instruction(circuit).probabilities(qargs=list(range(phase_bits)))
    tau=2*math.pi/encoding['span']
    reference=reference_product_formula(terms,time=-tau,steps=steps,order=order)*np.exp(-1j*tau*encoding['shift'])
    actual_u=Operator(build_pauli_energy_unitary(terms,steps=steps,order=order)).data
    exact=expm(1j*tau*(full_hamiltonian-encoding['shift']*np.eye(reference.shape[0])))
    operator_error=float(np.linalg.norm(actual_u-exact,2))
    spectrum_reference=reference
    hf_index=initial_basis
    sector_invariance=None
    if sector_indices is not None:
        sector_indices=tuple(sector_indices)
        if (not sector_indices or len(set(sector_indices))!=len(sector_indices)
                or any(type(i) is not int or not 0<=i<reference.shape[0] for i in sector_indices)):
            raise ValueError('reference sector requires unique valid basis indices')
        if initial_basis not in sector_indices:raise ValueError('input basis is outside reference sector')
        outside=[i for i in range(reference.shape[0]) if i not in sector_indices]
        unitary_leakage=float(np.linalg.norm(reference[np.ix_(outside,sector_indices)],2)) if outside else 0.
        hamiltonian_leakage=float(np.linalg.norm(full_hamiltonian[np.ix_(outside,sector_indices)],2)) if outside else 0.
        if hamiltonian_leakage>1e-12:
            raise ValueError('Hamiltonian does not preserve selected particle sector')
        if unitary_leakage>1e-12:
            raise ValueError('approximate unitary does not preserve selected particle sector')
        sector_invariance={'hamiltonian_leakage_norm':hamiltonian_leakage,'approximate_unitary_leakage_norm':unitary_leakage,
                           'numerical_validation_tolerance':1e-12,'formal_invariance_proof':False}
        spectrum_reference=reference[np.ix_(sector_indices,sector_indices)]
        hf_index=sector_indices.index(initial_basis)
    triangular,vectors=schur(spectrum_reference,output='complex')
    if np.linalg.norm(triangular-np.diag(np.diag(triangular)),2)>1e-10:
        raise RuntimeError('normal-unitary Schur reference failed diagonal check')
    phases=np.mod(np.angle(np.diag(triangular))/(2*math.pi),1.)
    weights=np.abs(vectors[hf_index,:])**2
    expected=weights@_phase_laws(phases,phase_bits)
    expected/=expected.sum()
    h=full_hamiltonian
    if sector_indices is not None:h=h[np.ix_(sector_indices,sector_indices)]
    eigenvalues,eigenvectors=np.linalg.eigh(h)
    energies=encoding['shift']+encoding['span']*phases
    approximate_ground=float(energies.min())
    bound=product_formula_error_bound(terms,time=-tau,steps=steps,order=order)
    eta=bound['analytic_operator_error_bound']
    energy_neighbourhood_bound=(encoding['span']/math.pi*math.asin(eta/2) if eta<math.sqrt(2) else None)
    ground_cluster_tolerance=1e-10*(1+max(abs(approximate_ground),abs(float(eigenvalues[0]))))
    true_overlap=float((np.abs(eigenvectors[hf_index,eigenvalues<=eigenvalues[0]+ground_cluster_tolerance])**2).sum())
    approx_overlap=float(weights[energies<=approximate_ground+ground_cluster_tolerance].sum())
    return {'phase_distribution':actual_distribution.tolist(),'reference_approximate_phase_distribution':expected.tolist(),
            'distribution_max_error':float(np.max(np.abs(actual_distribution-expected))),
            'access_gate_vs_product_reference_error':float(np.linalg.norm(actual_u-reference,2)),
            'access_operator_error_against_exact_H':operator_error,'analytic_product_formula_bound':bound,
            'exact_energy_ground_reference':float(eigenvalues[0]),'approximate_unitary_ground_energy_reference':approximate_ground,
            'ground_energy_bias_numeric':abs(approximate_ground-float(eigenvalues[0])),
            'spectral_neighbourhood_energy_bound_ideal_angles':energy_neighbourhood_bound,
            'true_H_hf_ground_overlap_reference':true_overlap,'approximate_V_hf_ground_weight_reference':approx_overlap,
            'numerical_ground_subspace_energy_tolerance':ground_cluster_tolerance,
            'phase_grid_energy_width':encoding['span']/2**phase_bits,'energy_encoding':encoding,
            'phase_eigenvalues_reference':phases.tolist(),'phase_eigenweights_reference':weights.tolist(),
            'sector_indices_reference':list(sector_indices) if sector_indices is not None else None,
            'sector_invariance_reference':sector_invariance,
            'ground_confidence_guarantee':None,'reference_method':'independent Pauli matrix factors and unitary Schur spectral law',
            'control_strategy':control_strategy,
            'spectral_sample_contract':spectral_sample_contract(terms,phase_bits=phase_bits,steps=steps,order=order),
            'simulation_width':circuit.num_qubits,'statevector_performed':True}
