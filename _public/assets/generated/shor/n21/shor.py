"""Bounded order finding with explicit exact-angle Fourier modular arithmetic.

This original pedagogical implementation uses no dense arithmetic matrices,
permutation lookup tables or opaque simulator arithmetic. Arithmetic correctness
is specified on x < N, an accumulator initially zero and a clean borrow flag.
The reversible circuit also has a unitary extension outside that valid domain;
no modular-arithmetic interpretation of that extension is claimed.
"""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

ROTATION_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")
MAX_INTEGER_BITS = 12
MAX_PHASE_BITS = 24
MAX_SIM_QUBITS = 18
MAX_PRIMITIVE_GATES = 500_000


def _integer(name: str, value: int, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum or 'unbounded'}]")
    return value


def validate_instance(modulus: int, base: int) -> int:
    _integer("modulus", modulus, 3)
    n = (modulus - 1).bit_length()
    if n > MAX_INTEGER_BITS:
        raise ValueError(f"At most {MAX_INTEGER_BITS} integer bits are supported")
    _integer("base", base, 1, modulus - 1)
    if math.gcd(base, modulus) != 1:
        raise ValueError("base must be coprime to modulus")
    return n


@lru_cache(maxsize=64)
def qft_gate(width: int):
    """Positive-exponent little-endian QFT, including final bit reversal."""
    _integer("width", width, 1, MAX_PHASE_BITS)
    circuit = QuantumCircuit(width, name=f"QFT{width}")
    for target in reversed(range(width)):
        circuit.h(target)
        for control in reversed(range(target)):
            circuit.cp(math.pi / 2 ** (target - control), control, target)
    for low in range(width // 2):
        circuit.swap(low, width - low - 1)
    return circuit.to_gate()


def _fourier_add(circuit: QuantumCircuit, constant: int, wires: list[int], controls: list[int]) -> None:
    """Add a classical constant modulo 2**width in the Fourier basis."""
    dimension = 1 << len(wires)
    for bit, wire in enumerate(wires):
        numerator = (constant * (1 << bit)) % dimension
        if numerator:
            angle = 2 * math.pi * numerator / dimension
            if controls:
                circuit.mcp(angle, controls, wire)
            else:
                circuit.p(angle, wire)


@lru_cache(maxsize=256)
def modular_adder_gate(modulus: int, constant: int, num_controls: int = 0):
    """Controlled y <- y+b mod N; y<N, high bit and borrow flag initially 0.

    Wire order: controls, n+1 accumulator bits, borrow flag. Subtraction of N
    and the borrow-cleanup work run even on inactive control branches. They
    cancel there on the valid y<N domain. This avoids controlling entire QFTs.
    """
    _integer("modulus", modulus, 3)
    n = (modulus - 1).bit_length()
    if n > MAX_INTEGER_BITS:
        raise ValueError("Modular adder exceeds integer-width limit")
    _integer("constant", constant, 0, modulus - 1)
    _integer("num_controls", num_controls, 0, 2)
    m = n + 1
    circuit = QuantumCircuit(num_controls + m + 1, name=f"ADD{constant}%{modulus}")
    controls = list(range(num_controls))
    y = list(range(num_controls, num_controls + m))
    flag = num_controls + m
    if constant == 0:
        return circuit.to_gate()
    fourier = qft_gate(m)
    circuit.append(fourier, y)
    _fourier_add(circuit, constant, y, controls)
    _fourier_add(circuit, -modulus, y, [])
    circuit.append(fourier.inverse(), y)
    circuit.cx(y[-1], flag)
    circuit.append(fourier, y)
    _fourier_add(circuit, modulus, y, [flag])
    _fourier_add(circuit, -constant, y, controls)
    circuit.append(fourier.inverse(), y)
    circuit.x(y[-1])
    circuit.cx(y[-1], flag)
    circuit.x(y[-1])
    circuit.append(fourier, y)
    _fourier_add(circuit, constant, y, controls)
    circuit.append(fourier.inverse(), y)
    return circuit.to_gate()


@lru_cache(maxsize=128)
def controlled_multiplier_gate(modulus: int, multiplier: int):
    """|c,x,0,0> -> |c, (a*x mod N if c else x),0,0>, for x<N.

    Layout is control; n x bits; n+1 accumulator bits; one borrow flag.
    Compute using n binary-weighted modular additions, controlled swaps, then
    uncompute using the inverse multiplier. All arithmetic gates are defined.
    """
    n = validate_instance(modulus, multiplier)
    circuit = QuantumCircuit(2 * n + 3, name=f"cM{multiplier}")
    control = 0
    x = list(range(1, n + 1))
    y = list(range(n + 1, 2 * n + 2))
    flag = 2 * n + 2
    if multiplier == 1:
        return circuit.to_gate()
    for bit, wire in enumerate(x):
        constant = multiplier * pow(2, bit, modulus) % modulus
        circuit.append(modular_adder_gate(modulus, constant, 2), [control, wire, *y, flag])
    for first, second in zip(x, y[:n]):
        circuit.cswap(control, first, second)
    inverse = pow(multiplier, -1, modulus)
    for bit, wire in reversed(list(enumerate(x))):
        constant = inverse * pow(2, bit, modulus) % modulus
        circuit.append(modular_adder_gate(modulus, constant, 2).inverse(), [control, wire, *y, flag])
    return circuit.to_gate()


def build_order_finding_circuit(modulus: int = 15, base: int = 2, phase_bits: int = 4, *, measure: bool = False) -> QuantumCircuit:
    """Build the real arithmetic circuit; total width t+2n+2.

    phase_bits=4 is a bounded N=15 demonstration, not the generic 2n-bit
    precision convention. Classical pow only chooses circuit constants.
    """
    n = validate_instance(modulus, base)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if 100 * n * phase_bits * (n + 1) ** 2 > MAX_PRIMITIVE_GATES:
        raise ValueError("Conservative arithmetic-work guard rejects this materialized circuit; reduce width/precision")
    circuit = QuantumCircuit(phase_bits + 2 * n + 2, phase_bits if measure else 0, name=f"order({base},{modulus})")
    phase = list(range(phase_bits))
    x = list(range(phase_bits, phase_bits + n))
    y = list(range(phase_bits + n, phase_bits + 2 * n + 1))
    flag = phase_bits + 2 * n + 1
    circuit.x(x[0])
    circuit.h(phase)
    for bit, control in enumerate(phase):
        multiplier = pow(base, 1 << bit, modulus)
        if multiplier != 1:
            circuit.append(controlled_multiplier_gate(modulus, multiplier), [control, *x, *y, flag])
    circuit.append(qft_gate(phase_bits).inverse(), phase)
    if measure:
        circuit.measure(phase, range(phase_bits))
    return circuit


def decompose(circuit: QuantumCircuit) -> QuantumCircuit:
    compiled = transpile(circuit, basis_gates=list(ROTATION_BASIS), optimization_level=1, seed_transpiler=7)
    if sum(compiled.count_ops().values()) > MAX_PRIMITIVE_GATES:
        raise ValueError("Expanded circuit exceeds the bounded primitive export limit")
    return compiled


def classical_order(modulus: int, base: int) -> int:
    """Bounded independent reference; deliberately not used to build gates."""
    validate_instance(modulus, base)
    if modulus > 4095:
        raise ValueError("Classical enumeration reference is bounded")
    value = 1
    for exponent in range(1, modulus):
        value = value * base % modulus
        if value == 1:
            return exponent
    raise AssertionError("Coprime base should have finite order")


def analytic_phase_distribution(order: int, phase_bits: int) -> np.ndarray:
    """Independent Fourier residue-class sums, evaluated geometrically.

    Residue classes have only two possible lengths. Summing their geometric
    series needs O(2**phase_bits) memory rather than a dense Fourier matrix.
    """
    _integer("order", order, 1)
    _integer("phase_bits", phase_bits, 1, 12)
    dimension = 1 << phase_bits
    if order > 4095:
        raise ValueError("Reference order is bounded")
    remainders = (np.arange(dimension) * order) % dimension
    singular = remainders == 0
    angles = -2j * math.pi * remainders / dimension
    quotient, remainder = divmod(dimension, order)
    def squared_sum(length: int) -> np.ndarray:
        amplitude = np.empty(dimension, dtype=complex)
        amplitude[singular] = length
        amplitude[~singular] = np.expm1(length * angles[~singular]) / np.expm1(angles[~singular])
        return np.abs(amplitude) ** 2
    return ((order - remainder) * squared_sum(quotient) + remainder * squared_sum(quotient + 1)) / dimension**2


def simulate_phase_distribution(modulus: int, base: int, phase_bits: int) -> dict[str, Any]:
    circuit = build_order_finding_circuit(modulus, base, phase_bits)
    if circuit.num_qubits > MAX_SIM_QUBITS:
        raise ValueError(f"Statevector checks are capped at {MAX_SIM_QUBITS} total qubits")
    state = Statevector.from_instruction(circuit)
    actual = state.probabilities(list(range(phase_bits)))
    order = classical_order(modulus, base)
    expected = analytic_phase_distribution(order, phase_bits)
    n = (modulus - 1).bit_length()
    auxiliary = list(range(phase_bits + n, circuit.num_qubits))
    clean = float(state.probabilities(auxiliary)[0])
    error = float(np.max(np.abs(actual - expected)))
    if error > 1e-10 or abs(clean - 1) > 1e-10:
        raise AssertionError("Arithmetic circuit disagrees with independent Fourier reference or leaves dirty work wires")
    return {"probabilities": actual.tolist(), "reference_probabilities": expected.tolist(), "max_probability_error": error,
            "work_register_zero_probability": clean, "classical_reference_order": order, "total_qubits": circuit.num_qubits}


def continued_fraction_candidates(outcome: int, phase_bits: int, modulus: int) -> list[int]:
    """Convergent denominators only; no hidden classical scan over orders."""
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    _integer("outcome", outcome, 0, (1 << phase_bits) - 1)
    _integer("modulus", modulus, 3)
    numerator, denominator = outcome, 1 << phase_bits
    p_previous, p_current, q_previous, q_current = 0, 1, 1, 0
    candidates = []
    while denominator:
        quotient, remainder = divmod(numerator, denominator)
        p_previous, p_current = p_current, quotient * p_current + p_previous
        q_previous, q_current = q_current, quotient * q_current + q_previous
        if 1 < q_current < modulus:
            candidates.append(q_current)
        numerator, denominator = denominator, remainder
    return candidates


def reconstruct_order(outcome: int, phase_bits: int, modulus: int, base: int) -> int | None:
    """Verify a sampled continued-fraction denominator by modular exponentiation.

    A verified denominator is a period, potentially a multiple of the minimal
    order when finite precision yields an unintended convergent. Factoring
    subsequently verifies its gcd outputs; no classical trial division or
    hidden order enumeration is performed here.
    """
    validate_instance(modulus, base)
    for candidate in continued_fraction_candidates(outcome, phase_bits, modulus):
        if pow(base, candidate, modulus) != 1:
            continue
        return candidate
    return None


def factors_from_order(modulus: int, base: int, order: int | None) -> list[int]:
    validate_instance(modulus, base)
    if order is None:
        return []
    _integer("order", order, 1)
    if order % 2 or pow(base, order, modulus) != 1:
        return []
    halfway = pow(base, order // 2, modulus)
    factors = sorted({math.gcd(halfway - 1, modulus), math.gcd(halfway + 1, modulus)} - {1, modulus})
    return factors


def success_contract(modulus: int, base: int, phase_bits: int, failure_allowance: float = 0.005) -> dict[str, Any]:
    """Tiny fixed-instance analytic sampling calculation; not generic Shor success."""
    if not 0 < failure_allowance < 1:
        raise ValueError("failure_allowance must lie in (0,1)")
    order = classical_order(modulus, base)
    distribution = analytic_phase_distribution(order, phase_bits)
    records = []
    successful_probability = 0.0
    for outcome, probability in enumerate(distribution):
        recovered = reconstruct_order(outcome, phase_bits, modulus, base)
        factors = factors_from_order(modulus, base, recovered)
        if factors:
            successful_probability += float(probability)
        if probability > 1e-14:
            records.append({"outcome": outcome, "probability": float(probability), "recovered_order": recovered, "factors": factors})
    if successful_probability <= 0:
        trials = None
    elif successful_probability >= 1 - 1e-14:
        trials = 1
    else:
        trials = math.ceil(math.log(failure_allowance) / math.log1p(-successful_probability))
    return {"fixed_modulus": modulus, "fixed_base": base, "phase_bits": phase_bits, "reference_order": order,
            "ideal_factoring_success_probability_per_trial": successful_probability, "ideal_sampling_failure_allowance": failure_allowance,
            "serial_trials": trials, "records": records, "scope": "fixed chosen base; no random-base or generic factoring guarantee",
            "classical_reference_used_only_for_validation_and_sampling_contract": True}
