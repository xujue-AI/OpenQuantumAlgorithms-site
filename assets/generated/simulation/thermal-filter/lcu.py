# SPDX-License-Identifier: MIT
"""Explicit Pauli LCU access circuits and bounded postselected Taylor evolution.

The original PREPARE circuit uses a binary tree of controlled RY rotations.
SELECT uses label-controlled X/Y/Z and coefficient phases. No dense unitary or
opaque Hamiltonian-evolution gate appears in the generated circuit. This is a
pedagogical, classically aggregated Taylor LCU, not the asymptotic segmented
algorithm with oblivious amplitude amplification of Berry et al. (2015),
doi:10.1103/PhysRevLett.114.090502. Qiskit is Apache-2.0; this module is MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector


MAX_SYSTEM_QUBITS = 32
MAX_LCU_TERMS = 256
MAX_TAYLOR_DEGREE = 12
MAX_STATEVECTOR_QUBITS = 12

PauliTerms = Sequence[tuple[complex, str]]


@dataclass
class PauliLCU:
    """PREPARE/SELECT access with data qubits first and label ancillas last."""

    terms: tuple[tuple[complex, str], ...]
    normalization: float
    n_system: int
    n_index: int
    prepare: QuantumCircuit
    select: QuantumCircuit
    circuit: QuantumCircuit


def _validate_terms(terms: PauliTerms) -> tuple[tuple[complex, str], ...]:
    clean = []
    n_system = None
    for coefficient, pauli in terms:
        coefficient = complex(coefficient)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("Pauli coefficients must be finite")
        if not isinstance(pauli, str) or not pauli or set(pauli) - set("IXYZ"):
            raise ValueError("Pauli labels must be nonempty strings over IXYZ")
        if n_system is None:
            n_system = len(pauli)
        if len(pauli) != n_system:
            raise ValueError("Pauli labels must have the same length")
        if coefficient != 0:
            clean.append((coefficient, pauli))
    if not clean:
        raise ValueError("LCU must contain a nonzero term")
    if n_system > MAX_SYSTEM_QUBITS:
        raise ValueError(f"Circuit generation is bounded to {MAX_SYSTEM_QUBITS} system qubits")
    if len(clean) > MAX_LCU_TERMS:
        raise ValueError(f"Explicit label SELECT is bounded to {MAX_LCU_TERMS} terms")
    return tuple(clean)


def _prepare_probabilities(probabilities: np.ndarray, n_index: int) -> QuantumCircuit:
    """Prepare positive amplitudes with high-to-low binary-tree rotations."""
    circuit = QuantumCircuit(n_index, name="PREPARE")
    padded = np.pad(probabilities, (0, 2**n_index - len(probabilities)))
    for depth in range(n_index):
        target = n_index - depth - 1
        controls = list(range(n_index - 1, target, -1))
        width = 2 ** (target + 1)
        for prefix in range(2**depth):
            start = prefix * width
            left = float(padded[start : start + width // 2].sum())
            right = float(padded[start + width // 2 : start + width].sum())
            if right == 0:
                continue
            theta = 2 * math.atan2(math.sqrt(right), math.sqrt(left))
            zero_controls = [q for k, q in enumerate(controls) if not ((prefix >> (depth - k - 1)) & 1)]
            for q in zero_controls:
                circuit.x(q)
            if controls:
                circuit.append(RYGate(theta).control(len(controls), annotated=False), controls + [target])
            else:
                circuit.ry(theta, target)
            for q in zero_controls:
                circuit.x(q)
    return circuit


def build_pauli_lcu(terms: PauliTerms) -> PauliLCU:
    """Encode sum(c_j P_j)/s with explicit access, s=sum(abs(c_j)).

    Labels use Qiskit's displayed big-endian Pauli convention: the rightmost
    character acts on data q[0]. All system qubits precede the index register,
    so the top-left 2**n_system block is the zero-index postselection branch.
    Unused padded labels apply identity. Coefficient loading is classical and
    all floating RY/phase rotations still require coherent synthesis.
    """
    terms = _validate_terms(terms)
    n_system = len(terms[0][1])
    n_index = (len(terms) - 1).bit_length()
    normalization = float(sum(abs(c) for c, _ in terms))
    if not math.isfinite(normalization) or normalization <= 0:
        raise ValueError("LCU normalization must be finite and positive")
    probabilities = np.array([abs(c) / normalization for c, _ in terms])
    prepare = _prepare_probabilities(probabilities, n_index)
    select = QuantumCircuit(n_system + n_index, name="SELECT")
    index = list(range(n_system, n_system + n_index))
    for label, (coefficient, pauli) in enumerate(terms):
        zeros = [q for k, q in enumerate(index) if not ((label >> k) & 1)]
        for q in zeros:
            select.x(q)
        angle = float(np.angle(coefficient))
        if angle != 0:
            if not index:
                select.global_phase += angle
            elif len(index) == 1:
                select.p(angle, index[0])
            else:
                select.mcp(angle, index[:-1], index[-1])
        for target, letter in enumerate(reversed(pauli)):
            if letter == "I":
                continue
            if not index:
                getattr(select, letter.lower())(target)
                continue
            if letter == "Y":
                select.sdg(target)
            elif letter == "Z":
                select.h(target)
            select.mcx(index, target)
            if letter == "Y":
                select.s(target)
            elif letter == "Z":
                select.h(target)
        for q in zeros:
            select.x(q)
    circuit = QuantumCircuit(n_system + n_index, name="Pauli LCU")
    circuit.compose(prepare, qubits=index, inplace=True)
    circuit.compose(select, inplace=True)
    circuit.compose(prepare.inverse(), qubits=index, inplace=True)
    circuit.metadata = {
        "normalization": normalization,
        "n_system": n_system,
        "n_index": n_index,
        "term_count": len(terms),
        "construction": "controlled-rotation PREPARE; indexed Pauli SELECT; PREPARE inverse",
        "rotations_synthesized": False,
    }
    return PauliLCU(terms, normalization, n_system, n_index, prepare, select, circuit)


def build_pauli_block_encoding(terms: PauliTerms) -> QuantumCircuit:
    """Return the access circuit; normalization/register sizes are in metadata."""
    return build_pauli_lcu(terms).circuit


_PAULI_PRODUCTS = {
    ("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
    ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
    ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y"),
}


def _multiply_paulis(left: str, right: str) -> tuple[complex, str]:
    phase = 1 + 0j
    result = []
    for a, b in zip(left, right, strict=True):
        if a == "I":
            result.append(b)
        elif b == "I":
            result.append(a)
        elif a == b:
            result.append("I")
        else:
            local_phase, letter = _PAULI_PRODUCTS[(a, b)]
            phase *= local_phase
            result.append(letter)
    return phase, "".join(result)


def taylor_pauli_terms(terms: PauliTerms, time: float, degree: int) -> tuple[tuple[complex, str], ...]:
    """Classically aggregate sum_{k=0}^K (-itH)^k/k! in the Pauli basis.

    This bounded preprocessing can grow exponentially with system size. It
    intentionally differs from a scalable word-indexed Taylor oracle. No
    coefficient is dropped by a numerical tolerance: only exact zeros are
    removed. IEEE floating arithmetic remains a representation limitation.
    """
    terms = _validate_terms(terms)
    if any(coefficient.imag != 0 for coefficient, _ in terms):
        raise ValueError("Taylor Hamiltonian Pauli coefficients must be real")
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= MAX_TAYLOR_DEGREE:
        raise ValueError(f"degree must be an integer from 0 to {MAX_TAYLOR_DEGREE}")
    if isinstance(time, bool) or not math.isfinite(time):
        raise ValueError("time must be finite")
    n_system = len(terms[0][1])
    identity = "I" * n_system
    power = {identity: 1 + 0j}
    polynomial = dict(power)
    for k in range(1, degree + 1):
        next_power = {}
        for label, coefficient in power.items():
            for h, pauli in terms:
                phase, product = _multiply_paulis(pauli, label)
                next_power[product] = next_power.get(product, 0j) + coefficient * h * phase * (-1j * time / k)
        power = {p: c for p, c in next_power.items() if c != 0}
        if len(power) > MAX_LCU_TERMS:
            raise ValueError("Classical Pauli expansion exceeds the explicit term bound")
        for label, coefficient in power.items():
            polynomial[label] = polynomial.get(label, 0j) + coefficient
        if len(polynomial) > MAX_LCU_TERMS:
            raise ValueError("Classical Pauli expansion exceeds the explicit term bound")
    return tuple((coefficient, label) for label, coefficient in sorted(polynomial.items()) if coefficient != 0)


def build_taylor_lcu(terms: PauliTerms, time: float, degree: int) -> PauliLCU:
    """One postselected Taylor segment with all access/uncomputation included."""
    return build_pauli_lcu(taylor_pauli_terms(terms, time, degree))


def build_lcu_attempt(lcu: PauliLCU, segments: int, *, initial_basis: int = 0) -> QuantumCircuit:
    """A complete basis-input attempt, label measurements, reset and Z readout.

    Labels are measured after each segment and reset before their next use.
    All segments run, even after failure; the reader accepts only outcomes with
    all label bits zero. This gives an explicit worst-case schedule without a
    classical early-abort optimization. Each complete retry starts from reset
    data and label wires. Final measurements on every data wire yield an
    accepted measured trial, with bits supporting computational-Z observables.
    """
    if isinstance(segments, bool) or not isinstance(segments, int) or not 1 <= segments <= 256:
        raise ValueError("segments must be an integer from 1 to 256")
    if isinstance(initial_basis, bool) or not isinstance(initial_basis, int) or not 0 <= initial_basis < 2**lcu.n_system:
        raise ValueError("initial_basis must index a computational basis state")
    circuit = QuantumCircuit(lcu.n_system + lcu.n_index,
                             segments * lcu.n_index + lcu.n_system,
                             name="Postselected Taylor attempt")
    circuit.reset(range(circuit.num_qubits))
    for q in range(lcu.n_system):
        if (initial_basis >> q) & 1:
            circuit.x(q)
    index = list(range(lcu.n_system, circuit.num_qubits))
    for segment in range(segments):
        circuit.compose(lcu.circuit, inplace=True)
        if index:
            circuit.measure(index, list(range(segment * lcu.n_index, (segment + 1) * lcu.n_index)))
            if segment < segments - 1:
                circuit.reset(index)
    circuit.measure(list(range(lcu.n_system)),
                    list(range(segments * lcu.n_index, segments * lcu.n_index + lcu.n_system)))
    circuit.metadata = dict(lcu.circuit.metadata or {}, segments=segments,
                           acceptance="all segment label measurement bits are zero; output is a measured system trial",
                           initial_basis=initial_basis,
                           schedule="all segments run; failed attempts are discarded after readout")
    return circuit


def postselected_lcu_evolution(
    lcu: PauliLCU, initial_state: np.ndarray, segments: int = 1,
) -> tuple[np.ndarray, float, list[float]]:
    """Simulate fresh-zero label access and condition on success each segment.

    Returns the conditional data state before final system readout, the probability that a complete attempt passes
    every label measurement, and per-segment conditional probabilities. This
    explicitly simulates the circuit each segment; it is not a dense matrix
    quantum implementation. Reset/measurement are accounted separately because
    this ideal simulator projects the successful branch deterministically.
    """
    if isinstance(segments, bool) or not isinstance(segments, int) or not 1 <= segments <= 256:
        raise ValueError("segments must be an integer from 1 to 256")
    if lcu.circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"Statevector simulation is bounded to {MAX_STATEVECTOR_QUBITS} total qubits")
    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (2**lcu.n_system,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state has the wrong dimension or nonfinite entries")
    if not np.isclose(np.vdot(state, state).real, 1, atol=1e-12, rtol=0):
        raise ValueError("initial_state must be normalized")
    probabilities = []
    for _ in range(segments):
        full = np.zeros(2**lcu.circuit.num_qubits, dtype=complex)
        full[: len(state)] = state
        output = Statevector(full).evolve(lcu.circuit).data[: len(state)]
        probability = float(np.vdot(output, output).real)
        if probability <= 0:
            raise ValueError("Postselection has zero success probability")
        probabilities.append(probability)
        state = output / math.sqrt(probability)
    return state, math.prod(probabilities), probabilities


def postselection_attempts(success_probability: float, failure_target: float = 0.01) -> int:
    """Independent complete restarts to obtain one accepted trial.

    This is an acceptance budget. The caller's measurement schedule determines
    whether its output is a pre-readout state or a measured trial; exported
    build_lcu_attempt circuits measure every system wire.

    Acceptance is distinct from physical faults and from observable sampling.
    Each attempt re-prepares the input and uses fresh/measured label ancillas;
    the worst-case circuit ledger is attempts*segments complete segment calls.
    """
    if not math.isfinite(success_probability) or not 0 < success_probability <= 1:
        raise ValueError("success_probability must be in (0, 1]")
    if not math.isfinite(failure_target) or not 0 < failure_target < 1:
        raise ValueError("failure_target must be in (0, 1)")
    if success_probability == 1:
        return 1
    return max(1, math.ceil(math.log(failure_target) / math.log1p(-success_probability)))
