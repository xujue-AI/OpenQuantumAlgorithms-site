# SPDX-License-Identifier: MIT
"""Original, unvalidated first symplectic CPF2 prototype for disjoint pairs.

Paper contract: arXiv:2409.08265v4, Eqs.25/26/30 and boundary identity Eq.6.
A=sum a_j Z_(2j)Z_(2j+1), B=alpha sum b_j X_(2j), delta=t/r.
C=-(-i delta)^2[A,B]/24=i alpha delta^2 sum a_j b_j Y_(2j)Z_(2j+1)/12.
The chronological circuit runs exp(-C), S2(delta)^r, exp(C).
This is a given-formula construction, not a generic corrector compiler.
Numerical checks and all publication claims remain pending at source freeze.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
import re
from typing import Iterator

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector
from scipy.linalg import expm

MAX_PAIRS = 32
MAX_REPETITIONS = 1024
MAX_SOURCE_OPERATIONS = 100_000
MAX_DENSE_PAIRS = 2
MAX_DENSE_REPETITIONS = 16
MAX_DENSE_SOURCE_OPERATIONS = 2_000
MAX_RATIONAL_BITS = 256
MAX_SCALAR_TEXT = 256
METHODS = ("strang", "cpf2")
PRIMITIVES = frozenset({"h", "s", "sdg", "cx", "rz", "rx", "reset", "measure"})


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def rational(value, name, magnitude=4):
    """Exact Fraction(str(value)); float arguments name decimal representations."""
    if isinstance(value, bool) or not isinstance(value, (Real, Fraction, str)):
        raise ValueError(f"{name} must be finite real")
    if isinstance(value, str) and len(value) > MAX_SCALAR_TEXT:
        raise ValueError(f"{name} exceeds the scalar-text bound before parsing")
    if isinstance(value, str) and not re.fullmatch(r"[+-]?(?:\d+/\d+|\d+(?:\.\d*)?|\.\d+)", value.strip()):
        raise ValueError(f"{name} must use bounded decimal or integer/fraction text; exponent notation is unsupported")
    if isinstance(value, int) and value.bit_length() > MAX_RATIONAL_BITS:
        raise ValueError(f"{name} exceeds the integer-bit bound before conversion")
    if isinstance(value, Fraction) and max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise ValueError(f"{name} exceeds the fraction-bit bound before conversion")
    try:
        result = value if type(value) is Fraction else Fraction(str(value))
        if not math.isfinite(float(result)):
            raise ValueError
    except (ValueError, OverflowError, ZeroDivisionError) as error:
        raise ValueError(f"{name} must be finite real") from error
    if abs(result) > magnitude or max(result.numerator.bit_length(), result.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise ValueError(f"{name} exceeds the rational or magnitude bound")
    return result


def _coefficients(values, pairs, name):
    if isinstance(values, (tuple, list)):
        if len(values) != pairs:
            raise ValueError(f"{name} must have exactly one coefficient per pair")
        return tuple(rational(v, name) for v in values)
    return (rational(values, name),) * pairs


@dataclass(frozen=True)
class PairModel:
    pairs: int
    a: object = "1"
    b: object = "1"
    alpha: object = "1/5"

    def __post_init__(self):
        _integer(self.pairs, "pairs", 1, MAX_PAIRS)
        object.__setattr__(self, "a", _coefficients(self.a, self.pairs, "a"))
        object.__setattr__(self, "b", _coefficients(self.b, self.pairs, "b"))
        alpha = rational(self.alpha, "alpha", 1)
        if alpha < 0:
            raise ValueError("alpha must be nonnegative; signed terms belong in b")
        object.__setattr__(self, "alpha", alpha)

    @property
    def qubits(self):
        return 2 * self.pairs

    def as_dict(self):
        norm_a = sum(abs(v) for v in self.a)
        norm_b0 = sum(abs(v) for v in self.b)
        return {"pairs": self.pairs, "qubits": self.qubits,
                "a_exact": list(map(str, self.a)), "b_exact": list(map(str, self.b)),
                "alpha_exact": str(self.alpha), "A_norm_exact": str(norm_a),
                "B0_norm_exact": str(norm_b0),
                "perturbation_norm_ratio_exact": str(self.alpha * norm_b0 / norm_a) if norm_a else None,
                "Hamiltonian": "sum_j a_j Z_(2j)Z_(2j+1) + alpha*b_j X_(2j)",
                "SDK_pair_label": "ZY means Y on first q[2j], Z on partner q[2j+1]",
                "scope": "Noninteracting disjoint pairs; classically factorized toy, not an application advantage."}


def request(model, time="1", repetitions=4, method="cpf2", dense=False):
    if not isinstance(model, PairModel):
        raise ValueError("model must be PairModel")
    if method not in METHODS:
        raise ValueError("method must be strang or cpf2")
    duration = rational(time, "time", 4)
    _integer(repetitions, "repetitions", 1, MAX_REPETITIONS)
    predicted = model.pairs * (7 * repetitions + (21 if method == "cpf2" else 7))
    if predicted > MAX_SOURCE_OPERATIONS:
        raise ValueError("complete source-operation cap exceeded before construction")
    if dense and (model.pairs > MAX_DENSE_PAIRS or repetitions > MAX_DENSE_REPETITIONS or predicted > MAX_DENSE_SOURCE_OPERATIONS):
        raise ValueError("dense checks exceed the separate two-pair/four-wire bounds")
    return duration, repetitions, predicted


def append_yz_rotation(circuit, first, partner, theta):
    """Literal exp(-i theta Y_first Z_partner /2), with exact Clifford frames."""
    theta = rational(theta, "Pauli rotation angle", 64)
    circuit.sdg(first)
    circuit.h(first)
    circuit.cx(first, partner)
    circuit.rz(float(theta), partner)
    circuit.cx(first, partner)
    circuit.h(first)
    circuit.s(first)


def _append_boundary(circuit, model, delta, sign):
    for j, (a, b) in enumerate(zip(model.a, model.b)):
        append_yz_rotation(circuit, 2*j, 2*j+1, sign * model.alpha * a * b * delta**2 / 6)


def evolution_circuit(model, time="1", repetitions=4, method="cpf2"):
    duration, repetitions, _ = request(model, time, repetitions, method)
    delta = duration / repetitions
    circuit = QuantumCircuit(model.qubits, name=method)
    if method == "cpf2":
        _append_boundary(circuit, model, delta, +1)  # chronological exp(-C)
    for _ in range(repetitions):
        for j, (a, b) in enumerate(zip(model.a, model.b)):
            first, partner = 2*j, 2*j+1
            circuit.cx(first, partner)
            circuit.rz(float(a * delta), partner)
            circuit.cx(first, partner)
            circuit.rx(float(2 * model.alpha * b * delta), first)
            circuit.cx(first, partner)
            circuit.rz(float(a * delta), partner)
            circuit.cx(first, partner)
    if method == "cpf2":
        _append_boundary(circuit, model, delta, -1)  # chronological exp(C)
    return circuit


def preparation_circuit(model):
    circuit = QuantumCircuit(model.qubits)
    circuit.h(range(model.qubits))
    for first in range(0, model.qubits, 2):
        circuit.s(first)
    return circuit


def complete_trial(model, time="1", repetitions=4, method="cpf2"):
    request(model, time, repetitions, method)
    circuit = QuantumCircuit(model.qubits, model.qubits)
    circuit.reset(range(model.qubits))
    circuit.compose(preparation_circuit(model), inplace=True)
    circuit.compose(evolution_circuit(model, time, repetitions, method), inplace=True)
    circuit.measure(range(model.qubits), range(model.qubits))
    return circuit


def rotation_targets(model, time="1", repetitions=4, method="cpf2") -> Iterator[Fraction]:
    duration, repetitions, _ = request(model, time, repetitions, method)
    delta = duration/repetitions
    if method == "cpf2":
        for a, b in zip(model.a, model.b):
            yield model.alpha * a * b * delta**2 / 6
    for _ in range(repetitions):
        for a, b in zip(model.a, model.b):
            yield a * delta
            yield 2 * model.alpha * b * delta
            yield a * delta
    if method == "cpf2":
        for a, b in zip(model.a, model.b):
            yield -model.alpha * a * b * delta**2 / 6


def logical_ledger(model, time="1", repetitions=4, method="cpf2"):
    duration, repetitions, predicted = request(model, time, repetitions, method)
    trial = complete_trial(model, duration, repetitions, method)
    counts = dict(trial.count_ops())
    if set(counts) - PRIMITIVES or len(trial.data) != predicted:
        raise ValueError("source ledger differs from the literal primitive construction")
    angles = list(rotation_targets(model, duration, repetitions, method))
    bound = sum((abs(Fraction.from_float(float(v))-v)/2 for v in angles), Fraction(0))
    return {"method": method, "model": model.as_dict(), "time_exact": str(duration),
            "repetitions": repetitions, "delta_exact": str(duration/repetitions),
            "source_operations": len(trial.data), "predicted_source_operations": predicted,
            "primitive_counts": counts, "continuous_rotation_calls": len(angles),
            "dependency_depth": trial.depth(), "global_phase": float(trial.global_phase),
            "angle_conversion_ideal_gate_norm_bound_exact": str(bound),
            "bound_scope": "Angle conversion only; splitting, SDK arithmetic, synthesis and physical errors are separate.",
            "included": ["all initial resets", "H/S known-input preparation", "both corrector boundaries for cpf2", "all Strang steps", "all Z readout bits"],
            "excluded": ["Clifford+T synthesis", "physical code/factory/routing", "hardware calibration", "finite-shot sampling", "classical postprocessing latency"],
            "physical_estimate": None, "formal_proof": None,
            "generator_bounds": {"pairs": MAX_PAIRS, "repetitions": MAX_REPETITIONS, "source_operations": MAX_SOURCE_OPERATIONS},
            "dense_bounds": {"pairs": MAX_DENSE_PAIRS, "qubits": 2*MAX_DENSE_PAIRS, "repetitions": MAX_DENSE_REPETITIONS, "source_operations": MAX_DENSE_SOURCE_OPERATIONS}}


def _pair_matrices(a, b, alpha):
    # Independent classical basis is |partner first>; no quantum matrix gate.
    z = np.array([[1, 0], [0, -1]], dtype=complex)
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    p, q, r = np.kron(z, z), np.kron(np.eye(2), x), np.kron(z, y)
    return float(a)*p, float(alpha*b)*q, p, q, r


def _tensor_pairs(pair_operators):
    result = np.ones((1, 1), dtype=complex)
    for value in reversed(pair_operators):
        result = np.kron(result, value)
    return result


def independent_formula(model, time="1", repetitions=4, method="cpf2"):
    """Closed-form four-by-four oracle, separate from circuit gate application."""
    duration, repetitions, _ = request(model, time, repetitions, method, dense=True)
    delta = float(duration / repetitions)
    pair_operators = []
    for a, b in zip(model.a, model.b):
        _, _, p, q, r = _pair_matrices(a, b, model.alpha)
        u, beta = float(a)*delta, float(model.alpha*b)*delta
        step = math.cos(beta)*math.cos(u)*np.eye(4)-1j*(math.sin(beta)*q+math.cos(beta)*math.sin(u)*p)
        result = np.linalg.matrix_power(step, repetitions)
        if method == "cpf2":
            c = float(model.alpha*a*b)*delta**2/12
            positive = math.cos(c)*np.eye(4)+1j*math.sin(c)*r
            result = positive @ result @ positive.conj().T
        pair_operators.append(result)
    return _tensor_pairs(pair_operators)


def independent_exact_dynamics(model, time="1", repetitions=4):
    duration, _, _ = request(model, time, repetitions, dense=True)
    pairs = []
    for a, b in zip(model.a, model.b):
        aa, bb, _, _, _ = _pair_matrices(a, b, model.alpha)
        rho = math.hypot(float(a), float(model.alpha*b))
        pairs.append(np.eye(4, dtype=complex) if rho == 0 else math.cos(rho*float(duration))*np.eye(4)-1j*math.sin(rho*float(duration))*(aa+bb)/rho)
    return _tensor_pairs(pairs)


def independent_hamiltonian(model):
    request(model, "1", 1, dense=True)
    dimension = 1 << model.qubits
    result = np.zeros((dimension, dimension), dtype=complex)
    for basis in range(dimension):
        for j, (a, b) in enumerate(zip(model.a, model.b)):
            first, partner = 2*j, 2*j+1
            parity = ((basis >> first) ^ (basis >> partner)) & 1
            result[basis, basis] += float(a) * (-1 if parity else 1)
            result[basis ^ (1 << first), basis] += float(model.alpha*b)
    return result


def independent_prepared_state(model):
    request(model, "1", 1, dense=True)
    dimension = 1 << model.qubits
    return np.array([(1j)**sum((basis >> first) & 1 for first in range(0, model.qubits, 2)) / math.sqrt(dimension)
                     for basis in range(dimension)], dtype=complex)


def compare_case(model, time="1", repetitions=4):
    duration, _, _ = request(model, time, repetitions, dense=True)
    exact = independent_exact_dynamics(model, duration, repetitions)
    exact_expm = expm(-1j*float(duration)*independent_hamiltonian(model))
    prepared = independent_prepared_state(model)
    signs = np.array([1 if (basis & 1) == 0 else -1 for basis in range(1 << model.qubits)])
    exact_state = exact @ prepared
    rows = {}
    for method in METHODS:
        circuit = evolution_circuit(model, duration, repetitions, method)
        actual = Operator(circuit).data
        formula = independent_formula(model, duration, repetitions, method)
        prefix = preparation_circuit(model)
        prefix.compose(circuit, inplace=True)
        state = Statevector.from_instruction(prefix).data
        trial = complete_trial(model, duration, repetitions, method)
        reset_body = trial.remove_final_measurements(inplace=False)
        density = DensityMatrix(np.eye(1 << model.qubits)/(1 << model.qubits)).evolve(reset_body).data
        expected_density = np.outer(state, state.conj())
        measured_probabilities = np.real(np.diag(density))
        rows[method] = {"operator_formula_max_entry_discrepancy": float(np.max(np.abs(actual-formula))),
                        "operator_exact_dynamics_error": float(np.linalg.norm(actual-exact, ord=2)),
                        "state_preparation_max_entry_discrepancy": float(np.max(np.abs(state-actual@prepared))),
                        "reset_density_max_entry_discrepancy": float(np.max(np.abs(density-expected_density))),
                        "state_vector_error": float(np.linalg.norm(state-exact_state)),
                        "state_infidelity": float(max(0.0, 1-abs(np.vdot(exact_state, state))**2)),
                        "Z0_mean": float(signs @ measured_probabilities),
                        "Z0_bias": float(abs(signs@measured_probabilities-signs@abs(exact_state)**2)),
                        "probabilities": measured_probabilities.tolist(),
                        "ledger": logical_ledger(model, duration, repetitions, method)}
    return {"model": model.as_dict(), "time_exact": str(duration), "repetitions": repetitions,
            "independent_expm_max_entry_discrepancy": float(np.max(np.abs(exact-exact_expm))),
            "methods": rows, "scope": "Ideal full complex operators and reset/readout laws, no phase alignment, hardware, shots or formal certificate."}


def negative_control_contrasts():
    """Frozen one-pair numerical-control proposal, never tuned from results."""
    model = PairModel(1, alpha="1/5")
    reference = independent_formula(model, "1", 1, "cpf2")
    correct = Operator(evolution_circuit(model, "1", 1, "cpf2")).data
    missing = Operator(evolution_circuit(model, "1", 1, "strang")).data
    wrong = QuantumCircuit(2)
    append_yz_rotation(wrong, 0, 1, Fraction(-1, 30))
    wrong.compose(evolution_circuit(model, "1", 1, "strang"), inplace=True)
    append_yz_rotation(wrong, 0, 1, Fraction(1, 30))
    return {"model": model.as_dict(), "time_exact": "1", "repetitions": 1,
            "correct_formula_max_entry_discrepancy": float(np.max(np.abs(correct-reference))),
            "missing_boundaries_max_entry_contrast": float(np.max(np.abs(missing-reference))),
            "wrong_sign_boundaries_max_entry_contrast": float(np.max(np.abs(Operator(wrong).data-reference))),
            "scope": "Numerical full complex operator contrasts; no sampled or hardware measurements."}


def canonical_cases():
    """Frozen 32-case proposal; no result-dependent tuning or phase fitting."""
    cases = [(PairModel(m, alpha=alpha), "1", r)
             for m in (1, 2) for alpha in ("1", "1/5", "1/20") for r in (1, 2, 4, 8)]
    cases.extend((PairModel(m, alpha="1/5"), "-1", r) for m in (1, 2) for r in (1, 4))
    cases.extend((PairModel(m, alpha="1/5"), "0", 1) for m in (1, 2))
    cases.append((PairModel(1, alpha="0"), "1", 4))
    cases.append((PairModel(2, a=("1", "1/2"), b=("1", "-3/4"), alpha="1/5"), "1", 2))
    return tuple(cases)
