# SPDX-License-Identifier: MIT
"""Structured Pauli-word Taylor access without enumerating Hamiltonian words.

PREPARE loads K+1 degree weights and K copies of L term weights. SELECT
enumerates only (slot, active-degree, term) predicates and applies their signed
Pauli gates. Its projected block is T_K(-itH)/s, with the unaggregated Taylor
normalization. Matrix exponentiation appears only in bounded check helpers.
The original source circuits use Qiskit 2.6.2 (Apache-2.0); this module is MIT.
Method: Berry et al.2015, doi:10.1103/PhysRevLett.114.090502.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

if __package__:
    from ..hamiltonian_simulation import ROTATION_BASIS, pauli_matrix
    from .lcu import _prepare_probabilities, postselection_attempts
    from .amplification import (build_oblivious_amplification,
                               amplified_block_polynomial, postselected_amplified_state)
else:
    from hamiltonian_simulation import ROTATION_BASIS, pauli_matrix
    from lcu import _prepare_probabilities, postselection_attempts
    from amplification import (build_oblivious_amplification,
                               amplified_block_polynomial, postselected_amplified_state)

MAX_SYSTEM_QUBITS = 32
MAX_TERMS = 64
MAX_DEGREE = 8
MAX_GENERATOR_QUBITS = 96
MAX_SELECT_PREDICATES = 4096
MAX_SOURCE_OPERATIONS = 100000
MAX_PRIMITIVE_OPERATIONS = 250000
MAX_STATEVECTOR_QUBITS = 12
MAX_REFERENCE_SYSTEM_QUBITS = 6
MAX_ATTEMPT_SEGMENTS = 32


@dataclass
class WordTaylor:
    terms: tuple[tuple[float, str], ...]
    time: float
    degree: int
    alpha: float
    normalization: float
    n_system: int
    n_degree: int
    n_term: int
    degree_probabilities: np.ndarray
    term_probabilities: np.ndarray
    prepare_degree: QuantumCircuit
    prepare_term: QuantumCircuit
    prepare: QuantumCircuit
    select: QuantumCircuit
    circuit: QuantumCircuit


def _integer(value, name: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer from {lower} to {upper}")
    return value


def _terms(terms: Sequence[tuple[float, str]]) -> tuple[tuple[float, str], ...]:
    result = []
    width = None
    for coefficient, label in terms:
        if isinstance(coefficient, bool):
            raise ValueError("coefficients must be finite real scalars")
        number = complex(coefficient)
        if number.imag != 0 or not math.isfinite(number.real):
            raise ValueError("Hamiltonian coefficients must be finite real scalars")
        if not isinstance(label, str) or not label or set(label)-set("IXYZ"):
            raise ValueError("Pauli labels must be nonempty strings over IXYZ")
        if width is None:
            width = len(label)
        if len(label) != width:
            raise ValueError("Pauli labels must all have the same width")
        if number.real != 0:
            result.append((number.real, label))
    if not result:
        raise ValueError("at least one nonzero Hamiltonian coefficient is required")
    _integer(width, "system qubits", 1, MAX_SYSTEM_QUBITS)
    if len(result) > MAX_TERMS:
        raise ValueError(f"explicit term access is bounded to {MAX_TERMS} terms")
    return tuple(result)


def _pattern_zeros(wires, value):
    return [q for bit, q in enumerate(wires) if not ((value >> bit) & 1)]


def _flip(circuit, wires):
    for wire in wires:
        circuit.x(wire)


def _phase_on_ones(circuit, controls, angle):
    if not controls:
        circuit.global_phase += angle
    elif len(controls) == 1:
        circuit.p(angle, controls[0])
    else:
        circuit.mcp(angle, controls[:-1], controls[-1])


def _controlled_pauli(circuit, controls, target, letter):
    if letter == "I":
        return
    if not controls:
        getattr(circuit, letter.lower())(target)
        return
    if letter == "Y":
        circuit.sdg(target)
    elif letter == "Z":
        circuit.h(target)
    circuit.mcx(controls, target)
    if letter == "Y":
        circuit.s(target)
    elif letter == "Z":
        circuit.h(target)


def _populate_word_select(select, terms, time, degree, n_system, n_degree,
                          n_term, extra_controls=()):
    """Write the same word predicates with optional already-one guard controls."""
    degree_wires = list(range(n_system, n_system+n_degree))
    time_sign = -1 if time < 0 else 1
    for k in range(1, degree+1):
        zeros = _pattern_zeros(degree_wires, k)
        _flip(select, zeros)
        _phase_on_ones(select, degree_wires+list(extra_controls), -time_sign*k*math.pi/2)
        _flip(select, zeros)
    negative_phase_calls = 0
    controlled_pauli_calls = 0
    for slot in range(degree):
        term_wires = list(range(n_system+n_degree+slot*n_term,
                                n_system+n_degree+(slot+1)*n_term))
        controls = degree_wires+term_wires+list(extra_controls)
        for k in range(slot+1, degree+1):
            for j, (coefficient, pauli) in enumerate(terms):
                zeros = _pattern_zeros(degree_wires, k)+_pattern_zeros(term_wires, j)
                _flip(select, zeros)
                if coefficient < 0:
                    _phase_on_ones(select, controls, math.pi)
                    negative_phase_calls += 1
                for target, letter in enumerate(reversed(pauli)):
                    _controlled_pauli(select, controls, target, letter)
                    controlled_pauli_calls += letter != "I"
                _flip(select, zeros)
    return negative_phase_calls, controlled_pauli_calls


def build_word_taylor(terms: Sequence[tuple[float, str]], time: float = .25,
                      degree: int = 3) -> WordTaylor:
    """Construct polynomially many defined access gates, independently of simulation.

    Register order is data, degree, slot0-term, ... slot(K-1)-term. The degree
    register has ceil(log2(K+1)) wires; each slot has ceil(log2 L). All K term
    registers are prepared even for inactive slots. Their projected sums are1.
    The source predicate count is K(K+1)L/2, not sum L**k. Controls use explicit
    zero-pattern X gates and controlled Pauli/phase definitions. Primitive
    lowering has a separate cap; source-operation counts are not T estimates.
    """
    terms = _terms(terms)
    _integer(degree, "degree", 0, MAX_DEGREE)
    if isinstance(time, bool):
        raise ValueError("time must be a finite real scalar")
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    alpha = float(sum(abs(c) for c, _ in terms))
    scale = alpha*abs(time)
    if not math.isfinite(alpha) or not math.isfinite(scale) or scale > 20:
        raise ValueError("alpha must be finite and alpha*abs(time) must not exceed20")
    weights = np.array([scale**k/math.factorial(k) for k in range(degree+1)])
    normalization = float(weights.sum())
    degree_probabilities = weights/normalization
    term_probabilities = np.array([abs(c)/alpha for c, _ in terms])
    n_system = len(terms[0][1])
    n_degree = degree.bit_length()
    n_term = (len(terms)-1).bit_length()
    n_ancillas = n_degree+degree*n_term
    n_total = n_system+n_ancillas
    _integer(n_total, "generator wires", 1, MAX_GENERATOR_QUBITS)
    predicates = degree*(degree+1)//2*len(terms)
    if predicates > MAX_SELECT_PREDICATES:
        raise ValueError("SELECT predicate materialization bound exceeded")
    prepare_degree = _prepare_probabilities(degree_probabilities, n_degree)
    prepare_degree.name = "PREPARE degree"
    prepare_term = _prepare_probabilities(term_probabilities, n_term)
    prepare_term.name = "PREPARE term"
    prepare = QuantumCircuit(n_ancillas, name="PREPARE words")
    prepare.compose(prepare_degree, qubits=list(range(n_degree)), inplace=True)
    for slot in range(degree):
        wires = list(range(n_degree+slot*n_term, n_degree+(slot+1)*n_term))
        prepare.compose(prepare_term, qubits=wires, inplace=True)
    select = QuantumCircuit(n_total, name="SELECT words")
    # At time0, phases must still define a unitary on unreachable degrees.
    negative_phase_calls, controlled_pauli_calls = _populate_word_select(
        select, terms, time, degree, n_system, n_degree, n_term)
    circuit = QuantumCircuit(n_total, name="Structured word Taylor")
    ancillas = list(range(n_system, n_total))
    circuit.compose(prepare, qubits=ancillas, inplace=True)
    circuit.compose(select, inplace=True)
    circuit.compose(prepare.inverse(), qubits=ancillas, inplace=True)
    if circuit.size() > MAX_SOURCE_OPERATIONS:
        raise ValueError("source instruction materialization bound exceeded")
    circuit.metadata = {
        "n_system": n_system, "time": time, "degree": degree,
        "alpha": alpha, "accepted_branch_normalization": normalization,
        "n_degree": n_degree, "n_term": n_term, "term_registers": degree,
        "n_ancillas": n_ancillas, "degree_coefficient_count": degree+1,
        "term_coefficient_count": len(terms), "word_enumeration_performed": False,
        "SELECT_slot_degree_term_predicates": predicates,
        "SELECT_signed_Pauli_gates": controlled_pauli_calls,
        "SELECT_negative_coefficient_phase_calls": negative_phase_calls,
        "SELECT_degree_phase_calls": degree,
        "PREPARE_degree_calls": 1, "PREPARE_term_calls": degree,
        "UNPREPARE_degree_calls": 1, "UNPREPARE_term_calls": degree,
        "PREPARE_total_calls": 1, "SELECT_total_calls": 1,
        "UNPREPARE_total_calls": 1,
        "source_instruction_count": circuit.size(),
        "register_order": "data, degree, slot0 term, ..., slot(K-1) term",
        "inactive_terms": "prepared independently and not selected; their projected weights sum to1",
        "primitive_rotations_synthesized": False,
    }
    return WordTaylor(terms, time, degree, alpha, normalization, n_system,
                      n_degree, n_term, degree_probabilities, term_probabilities,
                      prepare_degree, prepare_term, prepare, select, circuit)


def padded_word_source(access: WordTaylor) -> QuantumCircuit:
    """Actual zero-LCU padding to normalization2, requiring the native s<=2.

    A signal flag prepares probability s/2 in its0 branch, which controls word
    SELECT. The other branch applies Z to a sign qubit prepared in|+>, giving
    zero overlap. All word, signal and sign preparations are uncomputed. Thus
    the zero-ancilla block is T/2. The controlled SELECT gate has an actual
    circuit definition; neither a Hamiltonian matrix nor dense gate is used.
    """
    if access.normalization > 2:
        raise ValueError("normalization2 padding requires native Taylor normalization s<=2")
    original_width = access.circuit.num_qubits
    _integer(original_width+2, "padded generator wires", 1, MAX_GENERATOR_QUBITS)
    signal, sign = original_width, original_width+1
    source = QuantumCircuit(original_width+2, name="Word Taylor normalization2")
    word_ancillas = list(range(access.n_system, original_width))
    source.compose(access.prepare, qubits=word_ancillas, inplace=True)
    angle = 2*math.acos(math.sqrt(min(1.0, access.normalization/2)))
    source.ry(angle, signal)
    source.h(sign)
    source.x(signal)
    _populate_word_select(source, access.terms, access.time, access.degree,
                          access.n_system, access.n_degree, access.n_term,
                          extra_controls=(signal,))
    source.x(signal)
    source.cz(signal, sign)
    source.h(sign)
    source.ry(-angle, signal)
    source.compose(access.prepare.inverse(), qubits=word_ancillas, inplace=True)
    source.metadata = {
        **access.circuit.metadata, "accepted_branch_normalization": 2.0,
        "native_Taylor_normalization": access.normalization,
        "padding_signal_probability_zero": access.normalization/2,
        "zero_LCU_padding": True, "padding_ancillas": 2,
        "n_ancillas": source.num_qubits-access.n_system,
        "padding_implementation": "signal0 controls actual word SELECT; signal1 applies Z to sign|+>; all uncomputed",
        "controlled_SELECT_implementation": "extra signal guard on actual degree phases and Pauli predicates",
    }
    return source


def amplified_word_segment(access: WordTaylor, rounds: int = 0,
                           *, padding_to_two: bool = False) -> QuantumCircuit:
    """Actual oblivious access around ALL degree/word ancillas, no data reflection."""
    _integer(rounds, "rounds", 0, 4)
    source = padded_word_source(access) if padding_to_two else access.circuit
    if rounds == 0:
        return source.copy()
    return build_oblivious_amplification(source, rounds, n_system=access.n_system)


def build_word_attempt(access: WordTaylor, segments: int = 4, rounds: int = 0,
                       *, initial_basis: int = 0, padding_to_two: bool = False) -> QuantumCircuit:
    """Explicit all-segment attempt with resets, ancillary readout and final data Z.

    All segments execute, including after a nonzero label outcome. Accept only
    all-zero ancillary outcomes. The initial computational basis input is
    re-prepared after reset for every retry; an unknown specimen is not copied.
    """
    _integer(segments, "segments", 1, MAX_ATTEMPT_SEGMENTS)
    _integer(initial_basis, "initial_basis", 0, 2**access.n_system-1)
    source = amplified_word_segment(access, rounds, padding_to_two=padding_to_two)
    n_ancillas = source.num_qubits-access.n_system
    if segments*source.size() > MAX_SOURCE_OPERATIONS:
        raise ValueError("full attempt source-operation bound exceeded")
    attempt = QuantumCircuit(source.num_qubits, segments*n_ancillas+access.n_system,
                             name="Word Taylor measured attempt")
    attempt.reset(range(source.num_qubits))
    for q in range(access.n_system):
        if (initial_basis >> q)&1:
            attempt.x(q)
    ancillas = list(range(access.n_system, source.num_qubits))
    for segment in range(segments):
        attempt.compose(source, inplace=True)
        if ancillas:
            attempt.measure(ancillas, list(range(segment*n_ancillas, (segment+1)*n_ancillas)))
            if segment+1 < segments:
                attempt.reset(ancillas)
    attempt.measure(list(range(access.n_system)),
                    list(range(segments*n_ancillas, segments*n_ancillas+access.n_system)))
    factor = 2*rounds+1
    attempt.metadata = {
        **source.metadata, "segments": segments, "initial_basis": initial_basis,
        "forward_A_calls_per_attempt": segments*(rounds+1),
        "inverse_A_calls_per_attempt": segments*rounds,
        "total_A_or_inverse_calls_per_attempt": segments*factor,
        "zero_ancilla_reflections_per_attempt": segments*2*rounds,
        "degree_prepare_or_inverse_calls_per_attempt": 2*segments*factor,
        "term_prepare_or_inverse_calls_per_attempt": 2*segments*factor*access.degree,
        "SELECT_or_inverse_calls_per_attempt": segments*factor,
        "SELECT_predicates_or_inverse_per_attempt": factor*segments*access.circuit.metadata["SELECT_slot_degree_term_predicates"],
        "SELECT_signed_Paulis_or_inverse_per_attempt": factor*segments*access.circuit.metadata["SELECT_signed_Pauli_gates"],
        "padding_signal_and_sign_prepare_or_inverse_calls_per_attempt": 4*segments*factor if padding_to_two else 0,
        "padding_zero_LCU_CZ_or_inverse_calls_per_attempt": segments*factor if padding_to_two else 0,
        "acceptance": "all measured degree+word ancillary bits are zero",
        "schedule": "run all segments; reset ancillary wires between segments; final Z data readout",
    }
    return attempt


def decompose_word_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Primitive export with separate operation cap; resets/readout retained."""
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS,
                         optimization_level=1, seed_transpiler=7,
                         qubits_initially_zero=False)
    unexpected = set(compiled.count_ops())-set(ROTATION_BASIS)-{"measure", "reset", "barrier"}
    if unexpected:
        raise ValueError(f"unresolved primitive operations: {unexpected}")
    if compiled.size() > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("primitive export operation limit exceeded")
    return compiled


def projected_word_block(circuit: QuantumCircuit) -> np.ndarray:
    """Bounded projected-basis statevector check; not a generator or dense gate."""
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError("projected-basis statevector check limited to12 total wires")
    n_system = circuit.metadata["n_system"]
    if n_system > MAX_REFERENCE_SYSTEM_QUBITS:
        raise ValueError("projected block reference limited to6 system wires")
    dimension = 2**n_system
    columns = []
    for basis in range(dimension):
        full = np.zeros(2**circuit.num_qubits, dtype=complex)
        full[basis] = 1
        columns.append(Statevector(full).evolve(circuit).data[:dimension])
    return np.column_stack(columns)


def matrix_taylor_reference(access: WordTaylor) -> tuple[np.ndarray, np.ndarray]:
    """Independent small Kronecker H and matrix Taylor recurrence; no Pauli words."""
    if access.n_system > MAX_REFERENCE_SYSTEM_QUBITS:
        raise ValueError("classical references limited to6 system wires")
    h = sum(c*pauli_matrix(p) for c, p in access.terms)
    power = np.eye(len(h), dtype=complex)
    polynomial = power.copy()
    for k in range(1, access.degree+1):
        power = (-1j*access.time/k)*(h@power)
        polynomial += power
    return h, polynomial


def compare_word_dynamics(access: WordTaylor, segments: int = 4, rounds: int = 0,
                          initial_state: np.ndarray | None = None,
                          observable: np.ndarray | None = None,
                          *, padding_to_two: bool = False) -> dict:
    """Actual block/instrument versus independent Taylor and exponential answers.

    OAA reference is K p_r(K†K); no scaled-unitary success formula is assumed.
    Conditional errors retain phase. Numerical min/max acceptance are from the
    whole-attempt block singular values and are not a synthesis certificate.
    """
    _integer(segments, "segments", 1, MAX_ATTEMPT_SEGMENTS)
    _integer(rounds, "rounds", 0, 4)
    source = padded_word_source(access) if padding_to_two else access.circuit
    source_block = projected_word_block(source)
    h, polynomial = matrix_taylor_reference(access)
    normalization = 2.0 if padding_to_two else access.normalization
    source_reference = polynomial/normalization
    source_error = float(np.linalg.norm(source_block-source_reference, 2))
    segment = source if rounds == 0 else build_oblivious_amplification(source, rounds, n_system=access.n_system)
    actual_block = source_block if rounds == 0 else projected_word_block(segment)
    reference_block = amplified_block_polynomial(source_reference, rounds)
    dimension = 2**access.n_system
    if initial_state is None:
        initial_state = np.eye(dimension, dtype=complex)[:, 0]
    initial_state = np.asarray(initial_state, dtype=complex)
    if initial_state.shape != (dimension,) or not np.all(np.isfinite(initial_state)) or not np.isclose(np.vdot(initial_state, initial_state).real, 1, atol=1e-12, rtol=0):
        raise ValueError("initial state must be finite, normalized and have the system dimension")
    if observable is None:
        observable = pauli_matrix("I"*(access.n_system-1)+"Z")
    observable = np.asarray(observable, dtype=complex)
    if observable.shape != h.shape or not np.all(np.isfinite(observable)) or not np.allclose(observable, observable.conj().T, atol=1e-12, rtol=0):
        raise ValueError("observable must be finite Hermitian with the system dimension")
    state = initial_state.copy()
    probabilities = []
    for _ in range(segments):
        state, probability = postselected_amplified_state(segment, state)
        probabilities.append(probability)
    success = math.prod(probabilities)
    full_block = np.linalg.matrix_power(actual_block, segments)
    independent_block = np.linalg.matrix_power(reference_block, segments)
    independent_branch = independent_block@initial_state
    independent_success = float(np.vdot(independent_branch, independent_branch).real)
    target = expm(-1j*segments*access.time*h)
    reference_state = target@initial_state
    exact_observable = float(np.vdot(reference_state, observable@reference_state).real)
    actual_observable = float(np.vdot(state, observable@state).real)
    singular_values = np.linalg.svd(full_block, compute_uv=False)
    return {
        "segments": segments, "degree": access.degree, "rounds_per_segment": rounds,
        "total_time": segments*access.time, "source_normalization_per_segment": access.normalization,
        "accepted_branch_normalization_per_segment": normalization, "normalization2_padding": padding_to_two,
        "actual_source_block_error_vs_matrix_taylor": source_error,
        "actual_amplified_block_error_vs_general_polynomial": float(np.linalg.norm(actual_block-reference_block, 2)),
        "actual_whole_attempt_accepted_map_operator_error_vs_exact_U": float(np.linalg.norm(full_block-target, 2)),
        "unnormalized_Taylor_operator_error_full_time": float(np.linalg.norm(np.linalg.matrix_power(polynomial, segments)-target, 2)),
        "conditional_state_vector_error": float(np.linalg.norm(state-reference_state)),
        "conditional_state_infidelity": float(max(0, 1-abs(np.vdot(reference_state, state))**2)),
        "conditional_state_error_vs_independent_polynomial": float(np.linalg.norm(state-independent_branch/math.sqrt(independent_success))),
        "actual_observable": actual_observable, "exact_observable": exact_observable,
        "observable_error": abs(actual_observable-exact_observable),
        "full_attempt_acceptance_probability": success,
        "independent_full_attempt_acceptance_probability": independent_success,
        "per_segment_conditional_probabilities": probabilities,
        "finite_H_numerical_minimum_acceptance": float(singular_values[-1]**2),
        "finite_H_numerical_maximum_acceptance": float(singular_values[0]**2),
        "acceptance_input_dependence_span": float(singular_values[0]**2-singular_values[-1]**2),
        "full_attempts_for_one_accepted_trial_99_percent": postselection_attempts(min(1.0, success), .01),
        "acceptance_certificate": False,
        "success_formula": "actual instrument and K p_r(K†K); not assumed sin((2r+1)asin(1/s))²",
        "conditional_state_scope": "accepted state before final data Z readout",
        "physical_resource_estimate": None,
    }
