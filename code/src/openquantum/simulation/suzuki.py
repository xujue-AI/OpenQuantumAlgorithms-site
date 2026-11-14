# SPDX-License-Identifier: MIT
"""Original order2/4/6 Suzuki circuits with bounded scalar-parameter accounting.

Supported family: real-coefficient open-chain ZZ interactions, local Z fields
and local X fields. Dense references are separate and stop at three qubits.
No frozen Lie/Strang implementation is modified or called by the generator.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm

MAX_GENERATOR_QUBITS = 64
MAX_DENSE_QUBITS = 3
MAX_DENSE_BASE_BLOCKS = 512
# Evolution-body operations only; preparation/readout add at most3/6 ops.
MAX_DENSE_PRIMITIVE_OPERATIONS = 10_000
MAX_REPETITIONS = 4096
MAX_BASE_BLOCKS = 16384
MAX_PRIMITIVE_OPERATIONS = 100_000
MAX_RATIONAL_BITS = 4096
ROOT_BISECTION_BITS = 160
SUPPORTED_ORDERS = (2, 4, 6)
PRIMITIVES = frozenset({'rx', 'rz', 'cx', 'x', 'reset', 'measure'})


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'{name} must be a bounded integer >= {minimum}')
    return value


def rational(value, name):
    if isinstance(value, bool) or not isinstance(value, (Real, Fraction, str)):
        raise ValueError(f'{name} must be finite real; complex coefficients are unsupported')
    try:
        result = value if type(value) is Fraction else Fraction(str(value))
        real = float(result)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f'{name} must be finite real') from error
    if not math.isfinite(real) or result.numerator.bit_length() > MAX_RATIONAL_BITS or result.denominator.bit_length() > MAX_RATIONAL_BITS:
        raise ValueError(f'{name} exceeds finite rational representation limits')
    return result


@dataclass(frozen=True)
class SpinChain:
    qubits: int
    coupling: object = '1'
    x_field: object = '7/10'
    z_field: object = '0'

    def __post_init__(self):
        integer(self.qubits, 'qubits', 1, MAX_GENERATOR_QUBITS)
        for field in ('coupling', 'x_field', 'z_field'):
            object.__setattr__(self, field, rational(getattr(self, field), field))
        rational(self.l1_norm, 'Pauli l1 norm')

    @property
    def l1_norm(self):
        return (self.qubits-1)*abs(self.coupling) + self.qubits*(abs(self.x_field)+abs(self.z_field))

    @property
    def diagonal_terms(self):
        terms = []
        if self.coupling:
            for wire in range(self.qubits-1):
                label = ['I']*self.qubits
                label[self.qubits-1-wire] = label[self.qubits-2-wire] = 'Z'
                terms.append((self.coupling, ''.join(label)))
        if self.z_field:
            for wire in range(self.qubits):
                label = ['I']*self.qubits; label[self.qubits-1-wire] = 'Z'
                terms.append((self.z_field, ''.join(label)))
        return tuple(terms)

    @property
    def transverse_terms(self):
        if not self.x_field:
            return ()
        terms = []
        for wire in range(self.qubits):
            label = ['I']*self.qubits; label[self.qubits-1-wire] = 'X'
            terms.append((self.x_field, ''.join(label)))
        return tuple(terms)

    def as_dict(self):
        return {'qubits': self.qubits, 'coupling_exact': str(self.coupling), 'x_field_exact': str(self.x_field),
                'z_field_exact': str(self.z_field), 'pauli_l1_norm_exact': str(self.l1_norm),
                'diagonal_terms': [(str(c), p) for c,p in self.diagonal_terms],
                'transverse_terms': [(str(c), p) for c,p in self.transverse_terms],
                'boundary': 'open', 'hbar': 1,
                'normalization': 'No rescaling of H or time. The Pauli l1 norm is a norm upper bound, not a block-encoding normalization.',
                'coefficient_semantics': 'Exact Fraction(str(value)); real ZZ/Z/X family only. Binary64 gates and numerical matrices are separate representations.'}


def validated_order(order):
    integer(order, 'order', 2)
    if order not in SUPPORTED_ORDERS:
        raise ValueError('supported Suzuki orders are2,4,6; odd and other orders are unsupported')
    return order


def interval_product(left, right):
    values = [a*b for a in left for b in right]
    return min(values), max(values)


@lru_cache(maxsize=2)
def suzuki_coefficient(order):
    validated_order(order)
    if order == 2:
        raise ValueError('order2 is the base formula and has no recursive coefficient')
    degree = order-1
    lower, upper = Fraction(1), Fraction(2)
    # Exact dyadic arithmetic proves the enclosure of4^(1/degree).
    for _ in range(ROOT_BISECTION_BITS):
        middle = (lower+upper)/2
        if middle**degree <= 4:
            lower = middle
        else:
            upper = middle
    assert lower**degree <= 4 <= upper**degree and upper < 4
    ideal = (1/(4-lower), 1/(4-upper))
    # Choose the binary64 coefficient from a deterministic exact-rational
    # midpoint. Qiskit's separate reference uses its own libm reduction.
    actual = float((ideal[0]+ideal[1])/2)
    middle_actual = 1-4*actual
    middle_ideal = (1-4*ideal[1], 1-4*ideal[0])
    return actual, middle_actual, ideal, middle_ideal, (lower, upper)


@dataclass(frozen=True)
class Leaf:
    binary64: float
    ideal_lower: Fraction
    ideal_upper: Fraction

    @property
    def source_fraction(self):
        return Fraction.from_float(self.binary64)

    @property
    def coefficient_error_bound(self):
        return max(abs(self.source_fraction-self.ideal_lower), abs(self.source_fraction-self.ideal_upper))


@lru_cache(maxsize=3)
def normalized_leaves(order):
    """One macro-step's chronological S2 leaves, including every negative one."""
    validated_order(order)
    def expand(level, binary, enclosure):
        if level == 2:
            return (Leaf(binary, enclosure[0], enclosure[1]),)
        p, q, p_interval, q_interval, _ = suzuki_coefficient(level)
        outer = expand(level-2, binary*p, interval_product(enclosure, p_interval))
        middle = expand(level-2, binary*q, interval_product(enclosure, q_interval))
        return outer+outer+middle+outer+outer
    result = expand(order, 1.0, (Fraction(1), Fraction(1)))
    assert len(result) == 5**(order//2-1)
    return result


def request(chain, time, repetitions, order, initial_basis=0):
    if not isinstance(chain, SpinChain):
        raise ValueError('expected the supported real SpinChain family')
    validated_order(order)
    integer(repetitions, 'repetitions', 1, MAX_REPETITIONS)
    integer(initial_basis, 'initial_basis', 0, 2**chain.qubits-1)
    time = rational(time, 'time')
    leaves_per_step = 5**(order//2-1)
    blocks = repetitions*leaves_per_step
    if blocks > MAX_BASE_BLOCKS:
        raise ValueError('base Strang-block materialization exceeds its guard')
    per_block = (6*(chain.qubits-1) if chain.coupling else 0) + (2*chain.qubits if chain.z_field else 0) + (chain.qubits if chain.x_field else 0)
    operations = blocks*per_block+2*chain.qubits+initial_basis.bit_count()
    if operations > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError('primitive-operation materialization exceeds100000')
    leaves = normalized_leaves(order)
    for leaf in leaves:
        dt = time/repetitions*leaf.source_fraction
        for coefficient in ((chain.coupling,) if chain.qubits>1 else ()) + (chain.z_field, 2*chain.x_field):
            if abs(coefficient*dt) > 100_000:
                raise ValueError('source rotation angle exceeds materialization guard')
    return time, leaves


def _diagonal_half(circuit, chain, dt, reverse=False):
    edges = range(chain.qubits-1)
    wires = range(chain.qubits)
    if reverse:
        # Diagonal terms commute, but reverse the actual expansion explicitly.
        if chain.z_field:
            for wire in reversed(wires):
                circuit.rz(float(chain.z_field*dt), wire)
        if chain.coupling:
            for wire in reversed(edges):
                circuit.cx(wire,wire+1); circuit.rz(float(chain.coupling*dt),wire+1); circuit.cx(wire,wire+1)
    else:
        if chain.coupling:
            for wire in edges:
                circuit.cx(wire,wire+1); circuit.rz(float(chain.coupling*dt),wire+1); circuit.cx(wire,wire+1)
        if chain.z_field:
            for wire in wires:
                circuit.rz(float(chain.z_field*dt),wire)


def suzuki_circuit(chain, time='1', repetitions=1, order=4, *, initial_basis=0, measure=False):
    """Direct RX/RZ/CX recursion; matrices are never allocated by this function."""
    if not isinstance(chain, SpinChain) or not isinstance(measure, bool):
        raise ValueError('expected a SpinChain and Boolean measurement choice')
    time, leaves = request(chain,time,repetitions,order,initial_basis)
    circuit = QuantumCircuit(chain.qubits, chain.qubits if measure else 0, name=f'Suzuki{order} r={repetitions}')
    if measure:
        circuit.reset(range(chain.qubits))
    for wire in range(chain.qubits):
        if initial_basis>>wire&1:
            circuit.x(wire)
    for _ in range(repetitions):
        for leaf in leaves:
            dt = time/repetitions*leaf.source_fraction
            _diagonal_half(circuit,chain,dt)
            if chain.x_field:
                for wire in range(chain.qubits):
                    circuit.rx(float(2*chain.x_field*dt),wire)
            _diagonal_half(circuit,chain,dt,reverse=True)
    if measure:
        circuit.measure(range(chain.qubits),range(chain.qubits))
    return circuit


def parameter_rounding(chain, time, repetitions, order):
    """Exact scalar enclosures, distinct from the formula's splitting error."""
    time, leaves = request(chain,time,repetitions,order)
    scalar = Fraction(0)
    angle_rows = []
    for leaf in leaves:
        dt = time/repetitions*leaf.source_fraction
        angle_bound = Fraction(0)
        rotation_records = []
        for label, coefficient, count in (
            ('ZZ half layers',chain.coupling,2*(chain.qubits-1) if chain.coupling else 0),
            ('Z half layers',chain.z_field,2*chain.qubits if chain.z_field else 0),
            ('X full layers',2*chain.x_field,chain.qubits if chain.x_field else 0)):
            if count == 0:
                rotation_records.append({'layer':label,'occurrences_in_one_base_block':0,'angle_target_exact':None,'binary64_angle_hex':None,'per_rotation_operator_conversion_bound_exact':'0'})
                continue
            target = coefficient*dt
            actual = float(target)
            norm_bound = abs(Fraction.from_float(actual)-target)/2
            angle_bound += count*norm_bound
            rotation_records.append({'layer':label,'occurrences_in_one_base_block':count,'angle_target_exact':str(target),'binary64_angle_hex':actual.hex(),'per_rotation_operator_conversion_bound_exact':str(norm_bound)})
        scalar += repetitions*angle_bound
        angle_rows.append({'normalized_coefficient_binary64_hex':leaf.binary64.hex(),
                           'normalized_coefficient_source_exact':str(leaf.source_fraction),
                           'ideal_coefficient_enclosure':[str(leaf.ideal_lower),str(leaf.ideal_upper)],
                           'coefficient_absolute_error_bound_exact':str(leaf.coefficient_error_bound),
                           'source_substep_time_exact':str(dt),
                           'one_occurrence_gate_angle_conversion_bound_exact':str(angle_bound),
                           'rotation_parameters':rotation_records})
    coefficient_bound = chain.l1_norm*abs(time)*sum((leaf.coefficient_error_bound for leaf in leaves),Fraction(0))
    actual_sum = sum((leaf.source_fraction for leaf in leaves),Fraction(0))
    absolute = sum((abs(leaf.source_fraction) for leaf in leaves),Fraction(0))
    total = coefficient_bound+scalar
    return {'root_bisection_bits':ROOT_BISECTION_BITS,'normalized_leaves_per_macro_step':len(leaves),
            'normalized_signed_sum_exact_binary64':str(actual_sum),
            'normalized_signed_sum_defect_exact':str(abs(actual_sum-1)),
            'normalized_absolute_sum_exact_binary64':str(absolute),
            'actual_signed_substep_time_sum_exact':str(time*actual_sum),
            'actual_absolute_substep_time_sum_exact':str(abs(time)*absolute),
            'negative_normalized_leaves_per_step':sum(leaf.binary64<0 for leaf in leaves),
            'negative_actual_substeps':sum(time*leaf.source_fraction<0 for leaf in leaves)*repetitions,
            'coefficient_approximation_operator_bound_exact':str(coefficient_bound),
            'gate_angle_conversion_operator_bound_exact':str(scalar),
            'total_source_parameter_operator_bound_exact':str(total),
            'total_source_parameter_operator_bound_upper_float':math.nextafter(float(total),math.inf),
            'scope':'Exact rational bounds on coefficient/angle representation in the ideal gate model, relative to mathematical Suzuki roots; no bound on algorithmic splitting, numerical simulation, synthesis or physical errors. No finite-coefficient order certificate.',
            'leaf_parameter_records_one_macro_step':angle_rows}


def logical_ledger(chain,time='1',repetitions=1,order=4,*,initial_basis=0):
    circuit=suzuki_circuit(chain,time,repetitions,order,initial_basis=initial_basis,measure=True)
    counts={key:int(value) for key,value in circuit.count_ops().items()}
    return {'hamiltonian':chain.as_dict(),'time_exact':str(rational(time,'time')),'repetitions':repetitions,'order':order,
            'initial_basis':initial_basis,'system_qubits':chain.qubits,'ancillas':0,'logical_qubits':chain.qubits,
            'complete_measured_primitive_counts':counts,'arbitrary_rotations':counts.get('rx',0)+counts.get('rz',0),
            'source_dependency_depth':circuit.depth(),'base_strang_blocks':repetitions*5**(order//2-1),
            'source_parameter_rounding':parameter_rounding(chain,time,repetitions,order),
            'resource_kind':'Complete deterministic measured trial before synthesis/fault tolerance; negative substeps retain all gates',
            'physical_resource_estimate':None,'measurement':'All n wires inZ; bit0 scores +1/-1',
            'input_access':'Explicit real Pauli coefficients and local gates; no opaque Hamiltonian access or external data-loading oracle',
            'exclusions':['Physical reset/readout/controller/routing/idle/factory time and faults',
                          'Concrete rotation synthesis and its coherent error certificate','Unmeasured classical hardware input/control/postprocessing latency'],
            'planned_sampling_contract':{'fresh_repreparations':True,'observable':'Z0','epsilon_stat':0.05,'eta_stat':0.05,
                                        'required_trials_hoeffding':math.ceil(2*math.log(2/0.05)/0.05**2),
                                        'status':'Analytic independent-readout proposal; budgeted trials are not executed and algorithm/synthesis/physical allocations are not supplied'}}


def _dense_guard(chain):
    if not isinstance(chain,SpinChain) or chain.qubits>MAX_DENSE_QUBITS:
        raise ValueError('dense references stop at three qubits before allocation')


def _dense_request(chain,time,repetitions,order):
    _dense_guard(chain)
    time,leaves=request(chain,time,repetitions,order)
    if repetitions*len(leaves)>MAX_DENSE_BASE_BLOCKS:
        raise ValueError('dense recursive workload exceeds512base blocks')
    per_block=(6*(chain.qubits-1) if chain.coupling else 0)+(2*chain.qubits if chain.z_field else 0)+(chain.qubits if chain.x_field else 0)
    if repetitions*len(leaves)*per_block>MAX_DENSE_PRIMITIVE_OPERATIONS:
        raise ValueError('dense evolution-body workload exceeds10000operations')
    return time,leaves


def pauli_matrix(label):
    if not isinstance(label,str) or not 1<=len(label)<=MAX_DENSE_QUBITS or any(c not in 'IXZ' for c in label):
        raise ValueError('bounded real X/Z dense reference expected')
    matrices={'I':np.eye(2),'X':np.array([[0,1],[1,0]]),'Z':np.diag([1,-1])}
    result=np.array([[1+0j]])
    for character in label:
        result=np.kron(result,matrices[character])
    return result


def dense_blocks(chain):
    _dense_guard(chain)
    dimension=2**chain.qubits
    a=np.zeros((dimension,dimension),complex); b=np.zeros_like(a)
    for coefficient,label in chain.diagonal_terms:
        a+=float(coefficient)*pauli_matrix(label)
    for coefficient,label in chain.transverse_terms:
        b+=float(coefficient)*pauli_matrix(label)
    return a,b


def dense_product(chain,time,repetitions,order):
    time,leaves=_dense_request(chain,time,repetitions,order)
    a,b=dense_blocks(chain);result=np.eye(2**chain.qubits,dtype=complex)
    for _ in range(repetitions):
        for leaf in leaves:
            dt=float(time/repetitions*leaf.source_fraction)
            half=expm(-1j*a*dt/2); full=expm(-1j*b*dt)
            result=half@full@half@result
    return result


def qiskit_reference(chain,time,repetitions,order):
    time,_=_dense_request(chain,time,repetitions,order)
    groups=[SparsePauliOp.from_list([(label,float(coefficient)) for coefficient,label in terms])
            for terms in (chain.diagonal_terms,chain.transverse_terms) if terms]
    if not groups:
        groups=[SparsePauliOp.from_list([('I'*chain.qubits,0.0)])]
    evolution=PauliEvolutionGate(groups,time=float(time))
    return SuzukiTrotter(order=order,reps=repetitions,insert_barriers=False,preserve_order=True).synthesize(evolution)


def compare_case(chain,time='1',repetitions=1,order=4,*,initial_basis=0):
    _dense_request(chain,time,repetitions,order)
    integer(initial_basis,'initial_basis',0,2**chain.qubits-1)
    unitary=suzuki_circuit(chain,time,repetitions,order)
    actual=Operator(unitary).data
    independent=dense_product(chain,time,repetitions,order)
    reference=qiskit_reference(chain,time,repetitions,order)
    reference_matrix=Operator(reference).data
    a,b=dense_blocks(chain);target=expm(-1j*float(rational(time,'time'))*(a+b))
    source_error=float(np.linalg.norm(actual-independent,2)); package_error=float(np.linalg.norm(actual-reference_matrix,2))
    if source_error>1e-10 or package_error>1e-10:
        raise AssertionError('actual recursion failed independent product/reference comparison')
    prepared=suzuki_circuit(chain,time,repetitions,order,initial_basis=initial_basis)
    state=Statevector.from_instruction(prepared).data; target_state=target[:,initial_basis]
    measured=suzuki_circuit(chain,time,repetitions,order,initial_basis=initial_basis,measure=True)
    prefix=measured.remove_final_measurements(inplace=False)
    density=DensityMatrix.from_instruction(prefix).data
    reset_error=float(np.linalg.norm(density-np.outer(state,state.conj()),2))
    if reset_error>1e-10:
        raise AssertionError('actual reset/preparation prefix failed independent state check')
    observable=pauli_matrix('I'*(chain.qubits-1)+'Z')
    actual_z=float(np.vdot(state,observable@state).real); target_z=float(np.vdot(target_state,observable@target_state).real)
    return {'qubits':chain.qubits,'order':order,'repetitions':repetitions,'time_exact':str(rational(time,'time')),
            'source_to_independent_dense_product_operator_error':source_error,
            'source_to_qiskit_suzuki_operator_error':package_error,'reset_prefix_density_error':reset_error,
            'operator_error_vs_exact_evolution':float(np.linalg.norm(actual-target,2)),
            'state_vector_error_vs_exact_evolution':float(np.linalg.norm(state-target_state)),
            'state_infidelity':float(max(0,1-abs(np.vdot(target_state,state))**2)),
            'circuit_z0':actual_z,'exact_z0':target_z,'observable_bias':abs(actual_z-target_z),
            'qiskit_reference_primitive_counts':{name:int(count) for name,count in reference.count_ops().items()},
            'qiskit_reference_grouping':'Explicit nonempty commuting[A,B] groups; zero blocks omitted as a declared commuting-limit shortcut; source retains its S2 template.',
            'verified_tolerance':1e-10,'formal_certificate':False,
            'scope':'Bounded ideal numerical circuit/product/matrix-exponential comparisons; source parameter enclosures do not certify the large-system splitting error or a physical improvement'}
