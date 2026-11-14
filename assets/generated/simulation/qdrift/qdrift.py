# SPDX-License-Identifier: MIT
"""Primitive qDRIFT circuits, exact rational sampling and bounded channel checks.

This module implements the original construction. Qiskit supplies primitive
circuit containers/simulation; all dense references below are separate small
classical calculations. Decimal strings (including str(float)) specify exact
rational coefficients, while gates and dense matrices use binary64 arithmetic.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
import random
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from scipy.linalg import expm

MAX_GENERATOR_QUBITS = 64
MAX_TERMS = 256
MAX_EXPONENTIALS = 512
MAX_PRIMITIVE_GATES = 100_000
MAX_WEIGHT_BITS = 4096
MAX_DENSE_QUBITS = 3
MAX_ENSEMBLE_TRAJECTORIES = 64
PRIMITIVES = frozenset({'h', 's', 'sdg', 'x', 'rz', 'cx', 'reset', 'measure'})


def integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} exceeds {maximum}')
    return value


def rational(value, name):
    if isinstance(value, bool) or not isinstance(value, (Real, Fraction, str)):
        raise ValueError(f'{name} must be finite and real or a decimal/rational string')
    try:
        result = value if isinstance(value, Fraction) else Fraction(str(value))
        converted = float(result)
    except (ValueError, ZeroDivisionError, OverflowError) as error:
        raise ValueError(f'{name} must be finite and real') from error
    if not math.isfinite(converted) or result.numerator.bit_length() > MAX_WEIGHT_BITS or result.denominator.bit_length() > MAX_WEIGHT_BITS:
        raise ValueError(f'{name} exceeds the finite rational representation guard')
    return result


@dataclass(frozen=True)
class Hamiltonian:
    """Canonical ordered Pauli terms, combined exactly before normalization."""
    qubits: int
    terms: tuple[tuple[Fraction, str], ...]
    l1_norm: Fraction
    integer_weights: tuple[int, ...]
    cumulative_weights: tuple[int, ...]
    total_weight: int

    def __post_init__(self):
        integer(self.qubits, 'qubits', 1, MAX_GENERATOR_QUBITS)
        if not isinstance(self.terms, tuple) or len(self.terms) > MAX_TERMS:
            raise ValueError('invalid canonical term inventory')
        if type(self.l1_norm) is not Fraction or not isinstance(self.integer_weights, tuple) or not isinstance(self.cumulative_weights, tuple):
            raise ValueError('canonical probabilities require immutable exact metadata')
        rational(self.l1_norm, 'coefficient l1 norm')
        labels = []
        for pair in self.terms:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError('canonical terms must be immutable pairs')
            coefficient, label = pair
            if not isinstance(coefficient, Fraction) or not coefficient or not isinstance(label, str) or len(label) != self.qubits or any(c not in 'IXYZ' for c in label):
                raise ValueError('invalid canonical term')
            rational(coefficient, 'coefficient')
            labels.append(label)
        if len(set(labels)) != len(labels) or self.l1_norm != sum((abs(c) for c, _ in self.terms), Fraction(0)):
            raise ValueError('inconsistent canonical normalization')
        if len(self.integer_weights) != len(self.terms) or len(self.cumulative_weights) != len(self.terms):
            raise ValueError('inconsistent categorical inventory')
        running = 0
        for index, weight in enumerate(self.integer_weights):
            integer(weight, 'integer weight', 1)
            running += weight
            integer(self.cumulative_weights[index], 'cumulative weight', 1)
            if self.cumulative_weights[index] != running:
                raise ValueError('inconsistent cumulative weights')
        integer(self.total_weight, 'total weight', 0)
        if running != self.total_weight or running.bit_length() > MAX_WEIGHT_BITS:
            raise ValueError('inconsistent categorical total')
        for (coefficient, _), weight in zip(self.terms, self.integer_weights):
            if Fraction(weight, running) != abs(coefficient) / self.l1_norm:
                raise ValueError('categorical probabilities disagree with exact coefficients')

    @classmethod
    def from_terms(cls, qubits: int, terms: Iterable[tuple[object, str]]):
        integer(qubits, 'qubits', 1, MAX_GENERATOR_QUBITS)
        combined = {}
        for count, pair in enumerate(terms, 1):
            if count > MAX_TERMS:
                raise ValueError(f'input contains more than {MAX_TERMS} terms')
            try:
                coefficient, label = pair
            except (TypeError, ValueError) as error:
                raise ValueError('each term must contain a coefficient and a Pauli label') from error
            if not isinstance(label, str) or len(label) != qubits or any(c not in 'IXYZ' for c in label):
                raise ValueError('Pauli labels must match the qubit count and use I/X/Y/Z')
            combined[label] = combined.get(label, Fraction(0)) + rational(coefficient, 'coefficient')
        canonical = tuple((value, label) for label, value in combined.items() if value)
        l1 = sum((abs(value) for value, _ in canonical), Fraction(0))
        rational(l1, 'coefficient l1 norm')
        denominator = 1
        for value, _ in canonical:
            denominator = math.lcm(denominator, value.denominator)
            if denominator.bit_length() > MAX_WEIGHT_BITS:
                raise ValueError('categorical denominator exceeds its bit guard')
        weights = tuple(int(abs(value) * denominator) for value, _ in canonical)
        common = math.gcd(*weights) if weights else 1
        weights = tuple(weight // common for weight in weights)
        total = sum(weights)
        if total.bit_length() > MAX_WEIGHT_BITS:
            raise ValueError('categorical total exceeds its bit guard')
        cumulative = []
        running = 0
        for weight in weights:
            running += weight
            cumulative.append(running)
        return cls(qubits, canonical, l1, weights, tuple(cumulative), total)

    def as_dict(self):
        return {
            'qubits': self.qubits,
            'terms': [{'coefficient_exact': str(c), 'pauli_qiskit_order': p} for c, p in self.terms],
            'lambda_exact': str(self.l1_norm),
            'probabilities_exact': [str(abs(c) / self.l1_norm) for c, _ in self.terms],
            'integer_weights': list(self.integer_weights),
            'total_integer_weight': self.total_weight,
            'coefficient_semantics': 'Exact Fraction(str(value)); repeated labels combined exactly; zero coefficients omitted. Identity terms are retained and carry global phase.',
        }



def chain_hamiltonian(qubits: int, coupling='1', x_field='7/10', z_field='0'):
    """Open chain with explicit local terms; no matrix construction."""
    integer(qubits, 'qubits', 1, MAX_GENERATOR_QUBITS)
    coupling, x_field, z_field = (rational(v, n) for v, n in ((coupling, 'coupling'), (x_field, 'x_field'), (z_field, 'z_field')))
    terms = []
    for wire in range(qubits - 1):
        label = ['I'] * qubits
        label[qubits - 1 - wire] = label[qubits - 2 - wire] = 'Z'
        terms.append((coupling, ''.join(label)))
    for wire in range(qubits):
        for coefficient, letter in ((x_field, 'X'), (z_field, 'Z')):
            label = ['I'] * qubits
            label[qubits - 1 - wire] = letter
            terms.append((coefficient, ''.join(label)))
    return Hamiltonian.from_terms(qubits, terms)


def term_from_ticket(hamiltonian: Hamiltonian, ticket: int) -> int:
    if not hamiltonian.terms:
        raise ValueError('zero Hamiltonian has no categorical draw')
    integer(ticket, 'ticket', 0, hamiltonian.total_weight - 1)
    return bisect_right(hamiltonian.cumulative_weights, ticket)


def sample_schedule(hamiltonian: Hamiltonian, steps: int, seed: int) -> tuple[int, ...]:
    integer(steps, 'steps', 1, MAX_EXPONENTIALS)
    integer(seed, 'seed', 0, 2**64 - 1)
    if not hamiltonian.terms:
        return ()
    rng = random.Random(seed)
    return tuple(term_from_ticket(hamiltonian, rng.randrange(hamiltonian.total_weight)) for _ in range(steps))


def paired_seeds(master_seed: int, trajectories: int) -> tuple[tuple[int, int], ...]:
    """Distinct recorded PRNG seeds; ideal-i.i.d. assumptions are separate."""
    integer(master_seed, 'master_seed', 0, 2**64 - 1)
    integer(trajectories, 'trajectories', 1, MAX_ENSEMBLE_TRAJECTORIES)
    rng = random.Random(master_seed)
    used = set()
    values = []
    for _ in range(8 * trajectories):
        seed = rng.getrandbits(64)
        if seed not in used:
            used.add(seed)
            values.append(seed)
        if len(values) == 2 * trajectories:
            return tuple(zip(values[::2], values[1::2]))
    raise RuntimeError('distinct-seed derivation exhausted its finite candidate cap')


def _validated_schedule(hamiltonian, steps, seed, schedule):
    integer(steps, 'steps', 1, MAX_EXPONENTIALS)
    integer(seed, 'seed', 0, 2**64 - 1)
    result = sample_schedule(hamiltonian, steps, seed) if schedule is None else tuple(schedule)
    if len(result) != (steps if hamiltonian.terms else 0):
        raise ValueError('schedule length does not match the complete trial')
    for index in result:
        integer(index, 'term index', 0, len(hamiltonian.terms) - 1)
    return result


def _angles(hamiltonian, time, steps):
    exact = hamiltonian.l1_norm * rational(time, 'time') / steps
    # Bound huge-angle reduction independently from width and circuit size.
    if abs(exact) > 100_000:
        raise ValueError('per-exponential angle exceeds the materialization guard')
    angle = float(2 * exact)
    rounding = abs(Fraction.from_float(angle) - 2 * exact) / 2
    return exact, angle, rounding


def _pauli_exponential(circuit, label, angle):
    active = [(wire, character) for wire, character in enumerate(reversed(label)) if character != 'I']
    if not active:
        circuit.global_phase = float(circuit.global_phase) - angle / 2
        return
    for wire, character in active:
        if character == 'X':
            circuit.h(wire)
        elif character == 'Y':
            circuit.sdg(wire)
            circuit.h(wire)
    pivot = active[-1][0]
    for wire, _ in active[:-1]:
        circuit.cx(wire, pivot)
    circuit.rz(angle, pivot)
    for wire, _ in reversed(active[:-1]):
        circuit.cx(wire, pivot)
    for wire, character in reversed(active):
        if character == 'X':
            circuit.h(wire)
        elif character == 'Y':
            circuit.h(wire)
            circuit.s(wire)


def qdrift_circuit(hamiltonian: Hamiltonian, time=0.5, steps=8, seed=7, *, initial_basis=0, measure=False, schedule=None):
    """An actual complete trial in primitive gates, no Hamiltonian matrices.

    Measured trials reset all wires before basis preparation and measure each
    wire at the end. Unit-only variants omit resets/readout for Operator checks.
    """
    integer(initial_basis, 'initial_basis', 0, 2**hamiltonian.qubits - 1)
    if not isinstance(measure, bool):
        raise ValueError('measure must be Boolean')
    selected = _validated_schedule(hamiltonian, steps, seed, schedule)
    _, magnitude, _ = _angles(hamiltonian, time, steps)
    estimate = 2 * hamiltonian.qubits + initial_basis.bit_count()
    for index in selected:
        _, label = hamiltonian.terms[index]
        support = sum(character != 'I' for character in label)
        estimate += 2 * sum(character == 'X' for character in label) + 4 * sum(character == 'Y' for character in label)
        estimate += (1 + 2 * (support - 1)) if support else 0
    if estimate > MAX_PRIMITIVE_GATES:
        raise ValueError('primitive materialization exceeds its operation guard')
    circuit = QuantumCircuit(hamiltonian.qubits, hamiltonian.qubits if measure else 0, name=f'qDRIFT N={steps}')
    if measure:
        circuit.reset(range(hamiltonian.qubits))
    for wire in range(hamiltonian.qubits):
        if initial_basis >> wire & 1:
            circuit.x(wire)
    for index in selected:
        coefficient, label = hamiltonian.terms[index]
        _pauli_exponential(circuit, label, magnitude if coefficient > 0 else -magnitude)
    if measure:
        circuit.measure(range(hamiltonian.qubits), range(hamiltonian.qubits))
    return circuit


def logical_ledger(hamiltonian, time, steps, seed, *, initial_basis=0, schedule=None):
    selected = _validated_schedule(hamiltonian, steps, seed, schedule)
    circuit = qdrift_circuit(hamiltonian, time, steps, seed, initial_basis=initial_basis, measure=True, schedule=selected)
    _, angle, rounding = _angles(hamiltonian, time, steps)
    counts = {name: int(count) for name, count in circuit.count_ops().items() if name != 'barrier'}
    return {
        'hamiltonian': hamiltonian.as_dict(), 'time_exact': str(rational(time, 'time')),
        'steps_requested': steps, 'categorical_draws': len(selected), 'schedule_indices': list(selected),
        'trajectory_seed': seed, 'initial_basis': initial_basis, 'system_qubits': hamiltonian.qubits,
        'ancillas': 0, 'logical_qubits': hamiltonian.qubits,
        'complete_primitive_counts': counts, 'source_dependency_depth': circuit.depth(),
        'arbitrary_rotations': counts.get('rz', 0), 'magnitude_rz_angle_hex': angle.hex(),
        'per_exponential_scalar_angle_conversion_operator_bound_exact': str(rounding),
        'trajectory_scalar_angle_conversion_operator_bound_exact': str(rounding * len(selected)),
        'angle_bound_scope': 'Sum of binary64 scalar angle conversions only; not a full compiler/synthesis certificate. Qiskit global-phase accumulation/reduction for identity terms is unbounded by this field and irrelevant to the measured channel.',
        'global_phase_radians': float(circuit.global_phase),
        'resource_kind': 'Complete measured trial before rotation synthesis and error correction',
        'physical_resource_estimate': None,
        'excluded_costs': ['Physical reset, decoder, routing, factory and gate durations/faults', 'Concrete Clifford+T synthesis and its coherent error certificate', 'Classical input/sampling/scoring hardware time outside recorded CPU generation/check measurements'],
        'qasm_phase_scope': 'Measured-trial specification. Qiskit exporter may omit overall phase; restore the recorded phase before treating this as a coherently controlled oracle.',
    }


def _dense_guard(hamiltonian):
    if hamiltonian.qubits > MAX_DENSE_QUBITS:
        raise ValueError(f'dense references are limited to {MAX_DENSE_QUBITS} qubits before allocation')


def pauli_matrix(label):
    if not isinstance(label, str) or not 1 <= len(label) <= MAX_DENSE_QUBITS or any(c not in 'IXYZ' for c in label):
        raise ValueError('invalid bounded dense Pauli label')
    matrices = {'I': np.eye(2), 'X': np.array([[0, 1], [1, 0]]), 'Y': np.array([[0, -1j], [1j, 0]]), 'Z': np.diag([1, -1])}
    result = np.array([[1.0 + 0j]])
    for character in label:
        result = np.kron(result, matrices[character])
    return result


def dense_hamiltonian(hamiltonian):
    _dense_guard(hamiltonian)
    result = np.zeros((2**hamiltonian.qubits, 2**hamiltonian.qubits), complex)
    for coefficient, label in hamiltonian.terms:
        result += float(coefficient) * pauli_matrix(label)
    return result


def dense_trajectory(hamiltonian, time, steps, selected):
    _dense_guard(hamiltonian)
    selected = _validated_schedule(hamiltonian, steps, 0, selected)
    magnitude, _, _ = _angles(hamiltonian, time, steps)
    identity = np.eye(2**hamiltonian.qubits, dtype=complex)
    result = identity.copy()
    for index in selected:
        coefficient, label = hamiltonian.terms[index]
        phase = float(magnitude if coefficient > 0 else -magnitude)
        # P^2=I gives an independent elementary exponential, without Qiskit.
        gate = math.cos(phase) * identity - 1j * math.sin(phase) * pauli_matrix(label)
        result = gate @ result
    return result


def unitary_channel(unitary):
    if not isinstance(unitary, np.ndarray) or unitary.ndim != 2 or unitary.shape[0] != unitary.shape[1] or not 1 <= unitary.shape[0] <= 2**MAX_DENSE_QUBITS or not np.all(np.isfinite(unitary)):
        raise ValueError('unitary must be a finite bounded square matrix')
    if np.linalg.norm(unitary.conj().T @ unitary - np.eye(unitary.shape[0]), 2) > 1e-10:
        raise ValueError('unitary reference fails numerical unitarity')
    return np.kron(unitary.conjugate(), unitary)


def linear_channel_action(channel, matrix):
    """Column-vectorized action, also usable on spanning matrix units E_ab."""
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 2 or not isinstance(channel, np.ndarray) or channel.ndim != 2:
        raise ValueError('channel and matrix must be two-dimensional arrays')
    dimension = matrix.shape[0]
    if not 1 <= dimension <= 2**MAX_DENSE_QUBITS or matrix.shape != (dimension, dimension) or channel.shape != (dimension**2, dimension**2):
        raise ValueError('channel/matrix dimensions do not match')
    if not np.all(np.isfinite(channel)) or not np.all(np.isfinite(matrix)):
        raise ValueError('channel and matrix must be finite')
    return (channel @ matrix.reshape(-1, order='F')).reshape((dimension, dimension), order='F')


def averaged_channel(hamiltonian, time, steps):
    """Numerical finite mixture S^N, not a sampled-unitary or diamond metric."""
    _dense_guard(hamiltonian)
    integer(steps, 'steps', 1, MAX_EXPONENTIALS)
    dimension = 2**hamiltonian.qubits
    if not hamiltonian.terms:
        rational(time, 'time')
        return np.eye(dimension**2, dtype=complex)
    elementary = np.zeros((dimension**2, dimension**2), complex)
    for index, (coefficient, _) in enumerate(hamiltonian.terms):
        # Construct one elementary unitary before taking the channel power.
        magnitude, _, _ = _angles(hamiltonian, time, steps)
        phase = float(magnitude if coefficient > 0 else -magnitude)
        gate = math.cos(phase) * np.eye(dimension) - 1j * math.sin(phase) * pauli_matrix(hamiltonian.terms[index][1])
        elementary += float(abs(coefficient) / hamiltonian.l1_norm) * unitary_channel(gate)
    return np.linalg.matrix_power(elementary, steps)


def normalized_choi(channel, dimension):
    integer(dimension, 'dimension', 1, 2**MAX_DENSE_QUBITS)
    if not isinstance(channel, np.ndarray) or channel.shape != (dimension**2, dimension**2) or not np.all(np.isfinite(channel)):
        raise ValueError('invalid bounded channel matrix')
    result = np.zeros((dimension**2, dimension**2), complex)
    for a in range(dimension):
        for b in range(dimension):
            matrix = np.zeros((dimension, dimension), complex)
            matrix[a, b] = 1
            result[a*dimension:(a+1)*dimension, b*dimension:(b+1)*dimension] = linear_channel_action(channel, matrix) / dimension
    return result


def channel_diagnostics(channel, dimension):
    choi = normalized_choi(channel, dimension)
    tp_error = 0.0
    hermitian_error = 0.0
    for a in range(dimension):
        for b in range(dimension):
            basis = np.zeros((dimension, dimension), complex)
            basis[a, b] = 1
            image = linear_channel_action(channel, basis)
            transpose = np.zeros((dimension, dimension), complex)
            transpose[b, a] = 1
            hermitian_error = max(hermitian_error, float(np.linalg.norm(image.conj().T - linear_channel_action(channel, transpose))))
            tp_error = max(tp_error, float(abs(np.trace(image) - (1 if a == b else 0))))
    return {'spanning_matrix_units': dimension**2, 'trace_preservation_max_error': tp_error,
            'hermiticity_preservation_max_error': hermitian_error,
            'unitality_error': float(np.linalg.norm(linear_channel_action(channel, np.eye(dimension)) - np.eye(dimension))),
            'normalized_choi_trace': float(np.trace(choi).real),
            'normalized_choi_hermitian_error': float(np.linalg.norm(choi - choi.conj().T)),
            'normalized_choi_min_eigenvalue': float(np.linalg.eigvalsh((choi + choi.conj().T)/2).min()),
            'scope': 'Numerical diagnostics of an analytically CP/TP mixture of ideal unitaries; no formal or interval certificate'}


def channel_comparison(hamiltonian, time, steps, *, initial_basis=0):
    _dense_guard(hamiltonian)
    integer(initial_basis, 'initial_basis', 0, 2**hamiltonian.qubits - 1)
    dimension = 2**hamiltonian.qubits
    channel = averaged_channel(hamiltonian, time, steps)
    exact_unitary = expm(-1j * float(rational(time, 'time')) * dense_hamiltonian(hamiltonian))
    exact_channel = unitary_channel(exact_unitary)
    initial = np.zeros((dimension, dimension), complex)
    initial[initial_basis, initial_basis] = 1
    final = linear_channel_action(channel, initial)
    target = exact_unitary @ initial @ exact_unitary.conj().T
    observable = pauli_matrix('I' * (hamiltonian.qubits - 1) + 'Z')
    return {
        'steps': steps, 'time_exact': str(rational(time, 'time')),
        'hs_induced_superoperator_error': float(np.linalg.norm(channel - exact_channel, 2)),
        'normalized_choi_trace_distance': float(np.linalg.norm(normalized_choi(channel - exact_channel, dimension), ord='nuc') / 2),
        'fixed_input_trace_distance': float(np.linalg.norm(final - target, ord='nuc') / 2),
        'ensemble_observable': float(np.trace(observable @ final).real),
        'exact_observable': float(np.trace(observable @ target).real),
        'observable_bias': float(abs(np.trace(observable @ (final - target)).real)),
        'metric_scope': 'HS-induced2->2 superoperator norm, normalized Choi and fixed-input diagnostics; none is a computed diamond norm or certified floating-point bound',
        'channel_diagnostics': channel_diagnostics(channel, dimension),
    }


def empirical_study(hamiltonian, time, steps, trajectories=32, master_seed=20261113, *, initial_basis=0):
    """One fresh trajectory and one ideal classically sampled bit per trial."""
    _dense_guard(hamiltonian)
    integer(initial_basis, 'initial_basis', 0, 2**hamiltonian.qubits - 1)
    pairs = paired_seeds(master_seed, trajectories)
    records = []
    dimension = 2**hamiltonian.qubits
    empirical_channel = np.zeros((dimension**2, dimension**2), complex)
    total_counts = {}
    for trajectory_seed, readout_seed in pairs:
        selected = sample_schedule(hamiltonian, steps, trajectory_seed)
        circuit = qdrift_circuit(hamiltonian, time, steps, trajectory_seed, initial_basis=initial_basis, schedule=selected)
        actual = Operator(qdrift_circuit(hamiltonian, time, steps, trajectory_seed, schedule=selected)).data
        independent = dense_trajectory(hamiltonian, time, steps, selected)
        error = float(np.linalg.norm(actual - independent, 2))
        if error > 1e-11:
            raise AssertionError('actual primitive trajectory failed independent dense product check')
        state = Statevector.from_instruction(circuit).data
        reference_state = independent[:, initial_basis]
        state_error = float(np.linalg.norm(state - reference_state))
        if state_error > 1e-11:
            raise AssertionError('actual prepared state failed independent reference')
        measured_prefix = qdrift_circuit(hamiltonian, time, steps, trajectory_seed, initial_basis=initial_basis, measure=True, schedule=selected).remove_final_measurements(inplace=False)
        prefix_density = DensityMatrix.from_instruction(measured_prefix).data
        reset_error = float(np.linalg.norm(prefix_density - np.outer(state, state.conj()), 2))
        if reset_error > 1e-11:
            raise AssertionError('actual reset/preparation prefix failed its independent pure-state check')
        probabilities = np.abs(state)**2
        probability_norm = float(probabilities.sum())
        if abs(probability_norm - 1) > 1e-11:
            raise AssertionError('invalid state probability normalization')
        probabilities = probabilities / probability_norm
        p_zero = float(np.sum(probabilities[::2]))
        if not -1e-12 <= p_zero <= 1 + 1e-12:
            raise AssertionError('invalid readout probability')
        p_zero = min(1.0, max(0.0, p_zero))
        cumulative = np.cumsum(probabilities)
        cumulative[-1] = 1.0
        outcome = int(np.searchsorted(cumulative, random.Random(readout_seed).random(), side='right'))
        sample = 1 if outcome & 1 == 0 else -1
        ledger = logical_ledger(hamiltonian, time, steps, trajectory_seed, initial_basis=initial_basis, schedule=selected)
        for name, count in ledger['complete_primitive_counts'].items():
            total_counts[name] = total_counts.get(name, 0) + count
        records.append({'trajectory_seed': trajectory_seed, 'readout_seed': readout_seed,
                        'schedule_indices': list(selected), 'source_to_independent_operator_error': error,
                        'source_to_independent_state_error': state_error,
                        'actual_reset_prefix_density_error': reset_error, 'state_probability_norm_before_normalization': probability_norm,
                        'classically_sampled_basis_outcome': outcome, 'readout_bitstring_qiskit_order': format(outcome, f'0{hamiltonian.qubits}b'),
                        'circuit_z0_expectation': 2*p_zero - 1, 'classically_sampled_z0': sample,
                        'logical_ledger': ledger})
        empirical_channel += unitary_channel(actual) / trajectories
    mean = sum(r['classically_sampled_z0'] for r in records) / trajectories
    expected = sum(r['circuit_z0_expectation'] for r in records) / trajectories
    averaged = averaged_channel(hamiltonian, time, steps)
    comparison = channel_comparison(hamiltonian, time, steps, initial_basis=initial_basis)
    return {
        'hamiltonian': hamiltonian.as_dict(), 'time_exact': str(rational(time, 'time')), 'steps': steps,
        'initial_basis': initial_basis, 'master_seed': master_seed, 'trajectories': trajectories,
        'sample_kind': 'Seeded pseudorandom classical simulation; one final bit sampled from each freshly compiled ideal trajectory; no QPU execution',
        'randomness_scope': 'Distinct PRNG seeds record reproducible streams, not a proof of ideal independence. Channel and confidence statements assume independent ideal categorical draws and readout.',
        'records': records, 'sample_mean': mean, 'mean_of_fixed_trajectory_expectations': expected,
        'empirical_to_ensemble_hs_superoperator_error': float(np.linalg.norm(empirical_channel - averaged, 2)),
        'ensemble_reference': comparison,
        'complete_workload': {'fresh_preparations': trajectories, 'term_draws': len(hamiltonian.terms) and steps*trajectories,
                              'primitive_counts': total_counts, 'zero_state_resets': hamiltonian.qubits*trajectories,
                              'final_measurements': hamiltonian.qubits*trajectories, 'readout_samples_used': trajectories,
                              'rational_weight_table_builds': 1, 'classical_objective_scores': trajectories},
        'ideal_iid_hoeffding': {'failure_probability': 0.05, 'absolute_radius': math.sqrt(2*math.log(2/0.05)/trajectories),
                              'target': 'The averaged-channel Z0 mean, not the expectation of one fixed trajectory',
                              'scope': 'Analytic iid +/-1 bound; displayed PRNG sample does not validate coverage'},
        'physical_resource_estimate': None,
    }
