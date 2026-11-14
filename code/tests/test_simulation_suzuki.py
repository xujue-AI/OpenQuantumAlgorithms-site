# SPDX-License-Identifier: MIT
"""Independent recursive evolution, signed coefficient and source-bound checks."""
from fractions import Fraction
import importlib.metadata
import math

import numpy as np
import pytest
from qiskit import qasm3
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import DensityMatrix, Operator, SparsePauliOp, Statevector
from qiskit.synthesis import SuzukiTrotter
from scipy.linalg import expm

from openquantum.hamiltonian_simulation import product_formula_circuit
from openquantum.simulation import suzuki as s


def independent_blocks(chain):
    """Computational-bit action, independent of Kronecker/Pauli helpers."""
    dimension = 2**chain.qubits
    diagonal = np.zeros((dimension, dimension), complex)
    transverse = np.zeros((dimension, dimension), complex)
    for basis in range(dimension):
        z_values = [1 - 2*((basis >> wire) & 1) for wire in range(chain.qubits)]
        diagonal[basis, basis] = float(chain.coupling)*sum(z_values[j]*z_values[j+1] for j in range(chain.qubits-1)) + float(chain.z_field)*sum(z_values)
        for wire in range(chain.qubits):
            transverse[basis ^ (1 << wire), basis] += float(chain.x_field)
    return diagonal, transverse


def recursive_dense_step(a, b, time, order):
    if order == 2:
        half = expm(-1j * time * a / 2)
        return half @ expm(-1j * time * b) @ half
    p = 1/(4 - math.pow(4, 1/(order-1)))
    result = np.eye(a.shape[0], dtype=complex)
    for coefficient in (p, p, 1-4*p, p, p):
        result = recursive_dense_step(a, b, time*coefficient, order-2) @ result
    return result


def recursive_dense_evolution(chain, time, repetitions, order):
    a, b = independent_blocks(chain)
    return np.linalg.matrix_power(recursive_dense_step(a, b, float(time)/repetitions, order), repetitions)


def close(actual, expected, tolerance=3e-12):
    assert np.allclose(actual, expected, atol=tolerance, rtol=0)


def exact_bit_pauli(label):
    dimension = 2**len(label)
    result = np.zeros((dimension, dimension), complex)
    for basis in range(dimension):
        target, phase = basis, 1
        for wire, character in enumerate(reversed(label)):
            if character == 'X':
                target ^= 1 << wire
            elif character == 'Z' and (basis >> wire) & 1:
                phase *= -1
        result[target, basis] = phase
    return result


def grouped_qiskit_reference(chain, time, repetitions, order):
    groups = []
    diagonal = []
    transverse = []
    for wire in range(chain.qubits-1):
        if chain.coupling:
            label = ['I']*chain.qubits
            label[chain.qubits-1-wire] = label[chain.qubits-2-wire] = 'Z'
            diagonal.append((''.join(label), float(chain.coupling)))
    for wire in range(chain.qubits):
        if chain.z_field:
            label = ['I']*chain.qubits; label[chain.qubits-1-wire] = 'Z'
            diagonal.append((''.join(label), float(chain.z_field)))
        if chain.x_field:
            label = ['I']*chain.qubits; label[chain.qubits-1-wire] = 'X'
            transverse.append((''.join(label), float(chain.x_field)))
    for terms in (diagonal, transverse):
        if terms:
            groups.append(SparsePauliOp.from_list(terms))
    if not groups:
        groups.append(SparsePauliOp.from_list([('I'*chain.qubits, 0)]))
    gate = PauliEvolutionGate(groups, time=float(time))
    return SuzukiTrotter(order=order, reps=repetitions, preserve_order=True, insert_barriers=False).synthesize(gate)


@pytest.mark.parametrize('qubits', [1, 2, 3])
@pytest.mark.parametrize('order', [2, 4, 6])
def test_actual_chronology_matches_independent_recursive_dense_evolution(qubits, order):
    # A nonzero Z field makes even the one-spin example noncommuting.
    chain = s.SpinChain(qubits, coupling='-3/5', x_field='7/10', z_field='1/5')
    circuit = s.suzuki_circuit(chain, '0.47', 3, order)
    close(Operator(circuit).data, recursive_dense_evolution(chain, 0.47, 3, order))
    assert set(circuit.count_ops()) <= s.PRIMITIVES
    assert not any(item.operation.name in {'unitary', 'PauliEvolution', 'hamiltonian'} for item in circuit.data)


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('coupling,x_field,z_field', [('1', '7/10', '0'), ('-1', '-3/10', '1/5'), ('0', '-3/5', '-2/5')])
def test_negative_time_and_couplings_keep_correct_reverse_evolution(order, coupling, x_field, z_field):
    chain = s.SpinChain(2, coupling, x_field, z_field)
    forward = Operator(s.suzuki_circuit(chain, '0.53', 2, order)).data
    backward = Operator(s.suzuki_circuit(chain, '-0.53', 2, order)).data
    close(backward, recursive_dense_evolution(chain, -0.53, 2, order))
    close(backward @ forward, np.eye(4))


@pytest.mark.parametrize('order,leaves,negative', [(2, 1, 0), (4, 5, 1), (6, 25, 8)])
def test_signed_leaf_inventory_and_negative_substeps_are_preserved(order, leaves, negative):
    normalized = s.normalized_leaves(order)
    assert len(normalized) == leaves
    assert sum(leaf.binary64 < 0 for leaf in normalized) == negative
    assert [leaf.source_fraction for leaf in normalized] == [leaf.source_fraction for leaf in normalized[::-1]]
    positive = s.parameter_rounding(s.SpinChain(2), '1', 3, order)
    backward = s.parameter_rounding(s.SpinChain(2), '-1', 3, order)
    assert positive['negative_actual_substeps'] == 3*negative
    assert backward['negative_actual_substeps'] == 3*(leaves-negative)
    if order > 2:
        assert positive['negative_actual_substeps'] > 0
    circuit = s.suzuki_circuit(s.SpinChain(2), '1', 3, order)
    expected_rx = leaves*3*2
    assert circuit.count_ops()['rx'] == expected_rx
    assert circuit.count_ops()['cx'] == leaves*3*4
    assert circuit.count_ops()['rz'] == leaves*3*2
    assert sum(item.operation.name == 'rx' and float(item.operation.params[0]) < 0 for item in circuit.data) == negative*3*2
    assert sum(item.operation.name == 'rz' and float(item.operation.params[0]) < 0 for item in circuit.data) == negative*3*2


@pytest.mark.parametrize('order', [4, 6])
def test_exact_dyadic_root_and_recursive_coefficient_enclosures(order):
    actual, middle, p_interval, q_interval, root_interval = s.suzuki_coefficient(order)
    lower, upper = root_interval
    degree = order-1
    assert type(lower) is Fraction and type(upper) is Fraction
    assert lower**degree <= 4 <= upper**degree
    assert upper-lower == Fraction(1, 2**s.ROOT_BISECTION_BITS)
    assert 1 < lower < upper < 2
    assert p_interval == (1/(4-lower), 1/(4-upper))
    assert q_interval == (1-4*p_interval[1], 1-4*p_interval[0])
    assert p_interval[0] <= p_interval[1] and q_interval[0] <= q_interval[1] < 0
    assert 0 < actual < 0.5 and middle < 0
    assert actual == pytest.approx(1/(4-math.pow(4, 1/degree)), abs=2e-15)
    for leaf in s.normalized_leaves(order):
        assert leaf.ideal_lower <= leaf.ideal_upper
        exact = Fraction.from_float(leaf.binary64)
        assert leaf.source_fraction == exact
        assert leaf.coefficient_error_bound == max(abs(exact-leaf.ideal_lower), abs(exact-leaf.ideal_upper))
    normalized = s.normalized_leaves(order)
    assert sum(leaf.ideal_lower for leaf in normalized) <= 1 <= sum(leaf.ideal_upper for leaf in normalized)


@pytest.mark.parametrize('left,right,expected', [
    ((Fraction(1), Fraction(3)), (Fraction(2), Fraction(4)), (Fraction(2), Fraction(12))),
    ((Fraction(1), Fraction(3)), (Fraction(-4), Fraction(-2)), (Fraction(-12), Fraction(-2))),
    ((Fraction(-3), Fraction(-1)), (Fraction(-4), Fraction(-2)), (Fraction(2), Fraction(12))),
    ((Fraction(-2), Fraction(3)), (Fraction(-4), Fraction(5)), (Fraction(-12), Fraction(15))),
])
def test_interval_multiplication_uses_all_signed_cartesian_corners(left, right, expected):
    assert s.interval_product(left, right) == expected


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('time', ['1', '-2/3'])
def test_exact_scalar_and_gate_angle_bound_accounting(order, time):
    chain = s.SpinChain(3, '-1/3', '7/10', '1/5')
    repetitions = 4
    report = s.parameter_rounding(chain, time, repetitions, order)
    leaves = s.normalized_leaves(order)
    coefficient_error_sum = Fraction(0)
    gate_error_sum = Fraction(0)
    for leaf, row in zip(leaves, report['leaf_parameter_records_one_macro_step']):
        source = Fraction.from_float(float.fromhex(row['normalized_coefficient_binary64_hex']))
        assert source == leaf.source_fraction == Fraction(row['normalized_coefficient_source_exact'])
        low, high = map(Fraction, row['ideal_coefficient_enclosure'])
        coefficient_error = max(abs(source-low), abs(source-high))
        assert Fraction(row['coefficient_absolute_error_bound_exact']) == coefficient_error
        coefficient_error_sum += coefficient_error
        dt = Fraction(time)*source/repetitions
        assert Fraction(row['source_substep_time_exact']) == dt
        block_error = Fraction(0)
        targets = [(chain.coupling, 2*(chain.qubits-1)), (chain.z_field, 2*chain.qubits), (2*chain.x_field, chain.qubits)]
        for rotation, (coefficient, count) in zip(row['rotation_parameters'], targets):
            target = coefficient*dt
            assert Fraction(rotation['angle_target_exact']) == target
            assert rotation['occurrences_in_one_base_block'] == count
            actual = Fraction.from_float(float.fromhex(rotation['binary64_angle_hex']))
            error = abs(actual-target)/2
            assert Fraction(rotation['per_rotation_operator_conversion_bound_exact']) == error
            block_error += count*error
        assert Fraction(row['one_occurrence_gate_angle_conversion_bound_exact']) == block_error
        gate_error_sum += repetitions*block_error
    expected_coefficient = chain.l1_norm*abs(Fraction(time))*coefficient_error_sum
    assert Fraction(report['coefficient_approximation_operator_bound_exact']) == expected_coefficient
    assert Fraction(report['gate_angle_conversion_operator_bound_exact']) == gate_error_sum
    total = expected_coefficient+gate_error_sum
    assert Fraction(report['total_source_parameter_operator_bound_exact']) == total
    assert Fraction.from_float(report['total_source_parameter_operator_bound_upper_float']) >= total
    signed = sum((leaf.source_fraction for leaf in leaves), Fraction(0))
    absolute = sum((abs(leaf.source_fraction) for leaf in leaves), Fraction(0))
    assert Fraction(report['normalized_signed_sum_exact_binary64']) == signed
    assert Fraction(report['normalized_signed_sum_defect_exact']) == abs(signed-1)
    assert abs(signed-1) <= coefficient_error_sum
    assert Fraction(report['actual_signed_substep_time_sum_exact']) == Fraction(time)*signed
    assert Fraction(report['actual_absolute_substep_time_sum_exact']) == abs(Fraction(time))*absolute
    assert 'no bound on algorithmic splitting' in report['scope']
    assert 'No finite-coefficient order certificate' in report['scope']


@pytest.mark.parametrize('qubits', [1, 2, 3])
@pytest.mark.parametrize('time,coupling,x_field', [(0.37, 1, 0.7), (-0.41, -0.6, -0.3)])
def test_order_two_matches_unchanged_legacy_strang(qubits, time, coupling, x_field):
    chain = s.SpinChain(qubits, coupling, x_field, 0)
    actual = Operator(s.suzuki_circuit(chain, time, 3, 2)).data
    legacy = Operator(product_formula_circuit(qubits, time, 3, 2, coupling, x_field)).data
    close(actual, legacy)


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('chain', [s.SpinChain(1, '1', '7/10', '0'), s.SpinChain(2, '-3/5', '7/10', '1/5')])
def test_grouped_pinned_qiskit_reference_preserves_ab_contract(order, chain):
    assert importlib.metadata.version('qiskit') == '2.6.2'
    actual = Operator(s.suzuki_circuit(chain, '0.61', 2, order)).data
    reference = Operator(grouped_qiskit_reference(chain, 0.61, 2, order)).data
    close(actual, reference, 1e-11)
    close(Operator(s.qiskit_reference(chain, '0.61', 2, order)).data, reference, 1e-11)


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('chain', [s.SpinChain(1, '1', '7/10', '0'), s.SpinChain(2, '1', '0', '-1/5'), s.SpinChain(2, '0', '0', '0')])
def test_commuting_and_zero_hamiltonian_limits(order, chain):
    a, b = independent_blocks(chain)
    actual = Operator(s.suzuki_circuit(chain, '0.53', 4, order)).data
    close(actual, expm(-1j*0.53*(a+b)), 5e-12)


def test_bits_basis_preparation_resets_and_complete_readout():
    chain = s.SpinChain(3, '-1/2', '7/10', '1/5')
    assert chain.diagonal_terms == ((Fraction(-1, 2), 'IZZ'), (Fraction(-1, 2), 'ZZI'), (Fraction(1, 5), 'IIZ'), (Fraction(1, 5), 'IZI'), (Fraction(1, 5), 'ZII'))
    assert chain.transverse_terms == ((Fraction(7, 10), 'IIX'), (Fraction(7, 10), 'IXI'), (Fraction(7, 10), 'XII'))
    for label in ('IIX', 'XII', 'ZII', 'IIZ', 'ZZI'):
        close(s.pauli_matrix(label), exact_bit_pauli(label))
    prepared = s.suzuki_circuit(chain, '0.29', 1, 4, initial_basis=1)
    target = recursive_dense_evolution(chain, 0.29, 1, 4)[:, 1]
    close(Statevector.from_instruction(prepared).data, target)
    measured = s.suzuki_circuit(chain, '0.29', 1, 4, initial_basis=1, measure=True)
    counts = dict(measured.count_ops())
    assert counts['reset'] == counts['measure'] == 3 and counts['x'] == 1
    assert measured.data[3].operation.name == 'x'
    assert measured.find_bit(measured.data[3].qubits[0]).index == 0
    assert all(item.operation.name == 'reset' for item in measured.data[:3])
    assert all(item.operation.name == 'measure' for item in measured.data[-3:])
    assert set(counts) <= s.PRIMITIVES
    prefix = measured.remove_final_measurements(inplace=False)
    dirty_input = DensityMatrix(np.eye(8)/8)
    close(dirty_input.evolve(prefix).data, np.outer(target, target.conj()))
    ledger = s.logical_ledger(chain, '0.29', 1, 4, initial_basis=1)
    assert ledger['complete_measured_primitive_counts'] == counts
    assert ledger['base_strang_blocks'] == 5
    assert ledger['arbitrary_rotations'] == counts.get('rx', 0)+counts.get('rz', 0)
    assert ledger['physical_resource_estimate'] is None
    exported = qasm3.dumps(measured)
    assert 'reset' in exported and 'measure' in exported and 'unitary' not in exported


def test_fixed_case_empirical_convergence_is_finite_and_scoped():
    chain = s.SpinChain(2, '1', '7/10', '0')
    errors = {}
    for order in (2, 4, 6):
        group = [s.compare_case(chain, '1', repetitions, order) for repetitions in (1, 2, 4, 8)]
        errors[order] = [item['operator_error_vs_exact_evolution'] for item in group]
        for item in group:
            assert math.isfinite(item['operator_error_vs_exact_evolution'])
            assert item['source_to_independent_dense_product_operator_error'] < 1e-10
            assert item['source_to_qiskit_suzuki_operator_error'] < 1e-10
            assert item['reset_prefix_density_error'] < 1e-10
        assert errors[order][-1] < errors[order][0]/4
    assert errors[6][-1] < errors[4][-1] < errors[2][-1]
    # These are fixed-case numerical observations, not a universal step/order
    # or requested-accuracy certificate for future generated circuits.


@pytest.mark.parametrize('order', [False, 0, 1, 3, 5, 8, 4.0, '4'])
def test_odd_other_and_untyped_orders_rejected(order):
    with pytest.raises(ValueError):
        s.suzuki_circuit(s.SpinChain(1), order=order)


@pytest.mark.parametrize('kwargs', [{'qubits': True}, {'qubits': 0}, {'qubits': 65}, {'coupling': complex(1, 0)}, {'x_field': complex(0, 1)}, {'z_field': float('nan')}, {'x_field': float('inf')}, {'z_field': True}, {'coupling': Fraction(1, 2**4096)}])
def test_nonreal_or_unbounded_spin_family_rejected(kwargs):
    inputs = {'qubits': 2}; inputs.update(kwargs)
    with pytest.raises(ValueError):
        s.SpinChain(**inputs)


@pytest.mark.parametrize('kwargs', [{'repetitions': 0}, {'repetitions': True}, {'repetitions': 4097}, {'time': True}, {'time': complex(1, 0)}, {'time': float('nan')}, {'time': '1e308'}, {'initial_basis': 4}, {'initial_basis': True}, {'measure': 1}])
def test_invalid_request_arguments_rejected(kwargs):
    with pytest.raises(ValueError):
        s.suzuki_circuit(s.SpinChain(2), **kwargs)


def test_unsupported_hamiltonian_object_and_y_reference_rejected():
    with pytest.raises(ValueError):
        s.suzuki_circuit({'qubits': 1, 'x_field': 1, 'y_field': 1})
    with pytest.raises(ValueError):
        s.pauli_matrix('Y')


def test_base_block_and_repetition_guards_even_at_zero_hamiltonian():
    chain = s.SpinChain(1, 0, 0, 0)
    _, leaves = s.request(chain, 0, s.MAX_REPETITIONS, 2)
    assert len(leaves) == 1
    for order in (4, 6):
        factor = 5**(order//2-1)
        boundary = s.MAX_BASE_BLOCKS//factor
        s.request(chain, 0, boundary, order)
        with pytest.raises(ValueError, match='base Strang-block'):
            s.request(chain, 0, boundary+1, order)
    with pytest.raises(ValueError, match='repetitions'):
        s.request(chain, 0, s.MAX_REPETITIONS+1, 2)


def test_primitive_guard_before_quantum_circuit_creation(monkeypatch):
    chain = s.SpinChain(64, 1, 0, 0)
    # r264 gives99920 complete source operations; r265 gives100298.
    s.request(chain, 0, 264, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError('QuantumCircuit called before operation guard')
    monkeypatch.setattr(s, 'QuantumCircuit', forbidden)
    with pytest.raises(ValueError, match='primitive-operation'):
        s.suzuki_circuit(chain, 0, 265, 2)
    with pytest.raises(ValueError, match='primitive-operation'):
        s.suzuki_circuit(s.SpinChain(64), '1', 10, 6)


def test_dense_guards_before_matrix_or_reference_allocation(monkeypatch):
    chain = s.SpinChain(4)
    def forbidden(*args, **kwargs):
        raise AssertionError('matrix/reference allocated before dense guard')
    monkeypatch.setattr(s.np, 'zeros', forbidden)
    monkeypatch.setattr(s.np, 'eye', forbidden)
    monkeypatch.setattr(s, 'SparsePauliOp', forbidden)
    for function, args in ((s.dense_blocks, (chain,)), (s.dense_product, (chain, 1, 1, 4)), (s.qiskit_reference, (chain, 1, 1, 4)), (s.compare_case, (chain, 1, 1, 4))):
        with pytest.raises(ValueError, match='dense'):
            function(*args)


def test_source_generator_is_separate_from_bounded_dense_simulation():
    chain = s.SpinChain(32)
    circuit = s.suzuki_circuit(chain, '0.1', 1, 2, measure=True)
    assert circuit.num_qubits == 32
    assert circuit.count_ops()['reset'] == circuit.count_ops()['measure'] == 32
    with pytest.raises(ValueError, match='dense'):
        s.compare_case(chain, '0.1', 1, 2)


@pytest.mark.parametrize('order', [2, 4, 6])
def test_unused_layer_parameters_are_not_converted_into_nonexistent_angles(order):
    chain = s.SpinChain(1, coupling='1e308', x_field=0, z_field=0)
    report = s.parameter_rounding(chain, '1e308', 1, order)
    assert Fraction(report['total_source_parameter_operator_bound_exact']) == 0
    for leaf in report['leaf_parameter_records_one_macro_step']:
        for rotation in leaf['rotation_parameters']:
            assert rotation['occurrences_in_one_base_block'] == 0
            assert rotation['angle_target_exact'] is None
            assert rotation['binary64_angle_hex'] is None
            assert Fraction(rotation['per_rotation_operator_conversion_bound_exact']) == 0
    ledger = s.logical_ledger(chain, '1e308', 1, order)
    assert ledger['complete_measured_primitive_counts'] == {'reset': 1, 'measure': 1}


@pytest.mark.parametrize('order', [2, 4, 6])
@pytest.mark.parametrize('function', [s.dense_product, s.qiskit_reference, s.compare_case])
def test_dense_base_block_guard_even_for_zero_hamiltonian_before_allocation(monkeypatch, order, function):
    chain = s.SpinChain(1, 0, 0, 0)
    factor = 5**(order//2-1)
    boundary = s.MAX_DENSE_BASE_BLOCKS//factor
    s._dense_request(chain, 0, boundary, order)
    def forbidden(*args, **kwargs):
        raise AssertionError('dense/circuit allocation reached before workload guard')
    monkeypatch.setattr(s.np, 'zeros', forbidden)
    monkeypatch.setattr(s.np, 'eye', forbidden)
    monkeypatch.setattr(s, 'QuantumCircuit', forbidden)
    monkeypatch.setattr(s, 'SparsePauliOp', forbidden)
    with pytest.raises(ValueError, match='dense recursive workload'):
        function(chain, 0, boundary+1, order)


@pytest.mark.parametrize('function', [s.dense_product, s.qiskit_reference, s.compare_case])
def test_dense_evolution_body_operation_cap_before_allocation(monkeypatch, function):
    chain = s.SpinChain(3, 1, '7/10', '1/5')
    # Each S2 body has21 operations: r476 fits9996; r477 exceeds10000.
    s._dense_request(chain, '0.01', 476, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError('dense/circuit allocation reached before primitive guard')
    monkeypatch.setattr(s.np, 'zeros', forbidden)
    monkeypatch.setattr(s.np, 'eye', forbidden)
    monkeypatch.setattr(s, 'QuantumCircuit', forbidden)
    monkeypatch.setattr(s, 'SparsePauliOp', forbidden)
    with pytest.raises(ValueError, match='dense evolution-body workload'):
        function(chain, '0.01', 477, 2)
