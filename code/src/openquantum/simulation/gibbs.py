# SPDX-License-Identifier: MIT
"""Bounded diagonal Gibbs purification with explicit positive-amplitude loading.

This original circuit classically enumerates all diagonal energies and loads
their Boltzmann amplitudes with controlled RY gates. The exhaustive classical
input, O(n*2**n) preprocessing and expanded circuit are not a scalable Gibbs algorithm.
No dense matrix, diagonalization, Initialize or StatePreparation is a gate.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector, partial_trace
from scipy.linalg import expm

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_PREPARATION_QUBITS = 8
MAX_DENSE_SYSTEM_QUBITS = 6
MAX_SCALED_ENERGY_RANGE = 700.0


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _energies(energies: Sequence[float]) -> tuple[np.ndarray, int]:
    values = np.asarray(energies)
    if values.ndim != 1 or np.iscomplexobj(values) or values.dtype.kind not in "fiu":
        raise ValueError("energies must be a one-dimensional real array")
    values = values.astype(float)
    if len(values) < 2 or len(values) & (len(values) - 1) or not np.all(np.isfinite(values)):
        raise ValueError("energies need a finite power-of-two length of at least two")
    n = (len(values) - 1).bit_length()
    if n > MAX_PREPARATION_QUBITS:
        raise ValueError(f"explicit preparation is bounded to {MAX_PREPARATION_QUBITS} system qubits")
    return values, n


def diagonal_ising_energies(n_qubits: int = 2, J: float = 0.6,
                           fields: Sequence[float] = (0.35, -0.2)) -> np.ndarray:
    """Enumerate open-chain sum J Z_i Z_(i+1) + sum h_i Z_i in bit order.

    Entry x is E_x with q0 the least significant bit. This is explicit
    classical coefficient preparation, O(n*2**n) time and O(2**n) storage.
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or not 1 <= n_qubits <= MAX_PREPARATION_QUBITS:
        raise ValueError(f"n_qubits must be an integer from 1 to {MAX_PREPARATION_QUBITS}")
    J = _real(J, "J")
    fields = tuple(_real(h, "field") for h in fields)
    if len(fields) != n_qubits:
        raise ValueError("one field coefficient is required per system qubit")
    entries = []
    for x in range(2**n_qubits):
        z = [1 - 2 * ((x >> i) & 1) for i in range(n_qubits)]
        entries.append(J * sum(z[i] * z[i+1] for i in range(n_qubits-1))
                       + sum(h * v for h, v in zip(fields, z, strict=True)))
    result = np.array(entries, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("classical energy enumeration overflowed")
    return result


def gibbs_probabilities(energies: Sequence[float], beta: float) -> np.ndarray:
    """Classical normalized weights; no quantum algorithm or matrix gate.

    Use a ground-energy shift for stable weights. Reject scaled ranges above
    700 rather than silently dropping cold-state weights by underflow.
    """
    energies, _ = _energies(energies)
    beta = _real(beta, "beta")
    if beta < 0:
        raise ValueError("beta must be nonnegative in this thermal example")
    if beta == 0:
        return np.full(len(energies), 1/len(energies))
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = beta * (energies - energies.min())
    if not np.all(np.isfinite(shifted)) or shifted.max() > MAX_SCALED_ENERGY_RANGE:
        raise ValueError("beta times the energy range exceeds the floating preparation limit of 700")
    weights = np.exp(-shifted)
    return weights / weights.sum()


def positive_amplitude_circuit(probabilities: Sequence[float]) -> QuantumCircuit:
    """Original binary-tree amplitude loading; all controls are explicit gates."""
    probabilities, n = _energies(probabilities)
    if np.any(probabilities < 0) or not math.isclose(float(probabilities.sum()), 1, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("probabilities must be nonnegative and normalized")
    circuit = QuantumCircuit(n, name="Boltzmann PREPARE")
    for depth in range(n):
        target = n - depth - 1
        controls = list(range(n-1, target, -1))
        width = 2**(target+1)
        for prefix in range(2**depth):
            start = prefix * width
            left = float(probabilities[start:start+width//2].sum())
            right = float(probabilities[start+width//2:start+width].sum())
            if right == 0:
                continue
            angle = 2 * math.atan2(math.sqrt(right), math.sqrt(left))
            zeros = [q for k, q in enumerate(controls) if not ((prefix >> (depth-k-1)) & 1)]
            for q in zeros:
                circuit.x(q)
            if controls:
                circuit.append(RYGate(angle).control(len(controls), annotated=False), controls+[target])
            else:
                circuit.ry(angle, target)
            for q in zeros:
                circuit.x(q)
    return circuit


def diagonal_gibbs_purification(energies: Sequence[float], beta: float,
                               *, measure: bool = False) -> QuantumCircuit:
    """Prepare sum_x sqrt(p_x)|x>_system|x>_reference from all-zero wires.

    System wires q[0:n] precede reference q[n:2n]. Discarding the reference
    leaves the Gibbs density operator. With measure=True only system wires
    are measured, yielding one thermal sample; reference wires are discarded.
    Fresh preparation of all 2n wires is required on every repeated trial.
    """
    energies, n = _energies(energies)
    probabilities = gibbs_probabilities(energies, beta)
    circuit = QuantumCircuit(2*n, n if measure else 0, name="Diagonal Gibbs purification")
    circuit.compose(positive_amplitude_circuit(probabilities), qubits=list(range(n)), inplace=True)
    for i in range(n):
        circuit.cx(i, n+i)
    if measure:
        circuit.measure(list(range(n)), list(range(n)))
    circuit.metadata = {
        "n_system": n, "n_reference": n, "beta": float(beta),
        "energies": energies.tolist(), "probabilities": probabilities.tolist(),
        "preprocessing": "O(n*2**n) energy enumeration and prefix-mass/angle loading; O(2**n) normalization/storage",
        "prepared_output": "pure purification; reduced system state after reference discard is Gibbs",
        "measured_output": "one thermal system bitstring; reference discarded" if measure else None,
        "initialization": "all 2n wires assumed reset to zero before each trial; reset time unmeasured",
        "postselection": "none; ideal preparation succeeds deterministically",
        "rotations_synthesized": False,
    }
    return circuit


def decompose_gibbs(circuit: QuantumCircuit) -> QuantumCircuit:
    """Primitive rotation-basis lowering; arbitrary rotations remain unsynthesized."""
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS, optimization_level=1, seed_transpiler=7)
    unknown = set(compiled.count_ops()) - set(ROTATION_BASIS) - {"measure", "reset", "barrier"}
    if unknown:
        raise ValueError(f"unresolved gates: {unknown}")
    return compiled


def build_gibbs_trial(energies: Sequence[float], beta: float) -> QuantumCircuit:
    """Reset all wires, purify, measure the system and discard the reference.

    The output is a measured thermal trial, not a retained Gibbs state. The
    n reference wires are ignored without conditioning and reset on the next
    fresh trial. All 2n initial reset instructions are part of this circuit.
    """
    energies, n = _energies(energies)
    circuit = QuantumCircuit(2*n, n, name="Measured Gibbs trial")
    circuit.reset(range(2*n))
    preparation = diagonal_gibbs_purification(energies, beta, measure=True)
    circuit.compose(preparation, inplace=True)
    circuit.metadata = {**preparation.metadata, "initial_reset_instructions": 2*n,
                        "discarded_reference_wires": n, "output": "measured thermal trial"}
    return circuit


def build_gibbs_mixed_preparation(energies: Sequence[float], beta: float) -> QuantumCircuit:
    """Reset all wires, purify and reset the reference without conditioning.

    This explicit quantum-channel schedule leaves a retained thermal system
    and a zero reference register. Reset is irreversible; uncomputing CX would
    instead restore a coherent amplitude state and would not prepare Gibbs.
    """
    energies, n = _energies(energies)
    circuit = QuantumCircuit(2*n, name="Retained mixed Gibbs state")
    circuit.reset(range(2*n))
    preparation = diagonal_gibbs_purification(energies, beta)
    circuit.compose(preparation, inplace=True)
    circuit.reset(list(range(n, 2*n)))
    circuit.metadata = {**preparation.metadata, "initial_reset_instructions": 2*n,
                        "reference_reset_instructions": n, "output": "retained mixed system state"}
    return circuit


def reduced_gibbs_state(energies: Sequence[float], beta: float) -> np.ndarray:
    """Bounded simulation of the actual purification followed by partial trace."""
    energies, n = _energies(energies)
    if n > MAX_DENSE_SYSTEM_QUBITS:
        raise ValueError(f"reduced-state validation is bounded to {MAX_DENSE_SYSTEM_QUBITS} system qubits")
    state = Statevector.from_instruction(diagonal_gibbs_purification(energies, beta))
    return partial_trace(state, list(range(n, 2*n))).data


def exact_diagonal_gibbs(energies: Sequence[float], beta: float) -> np.ndarray:
    """Independent bounded classical matrix-exponential reference, never a gate."""
    energies, n = _energies(energies)
    if n > MAX_DENSE_SYSTEM_QUBITS:
        raise ValueError(f"dense reference is bounded to {MAX_DENSE_SYSTEM_QUBITS} system qubits")
    beta = _real(beta, "beta")
    gibbs_probabilities(energies, beta)  # validates the declared numeric domain
    if beta == 0:
        return np.eye(len(energies))/len(energies)
    operator = expm(-beta * np.diag(energies - energies.min()))
    return operator / np.trace(operator)


def thermal_observables(energies: Sequence[float], density: np.ndarray) -> dict[str, float]:
    """Energy and diagonal Z correlations for the state; not a sampling run."""
    energies, n = _energies(energies)
    density = np.asarray(density)
    if density.shape != (2**n, 2**n) or not np.all(np.isfinite(density)):
        raise ValueError("density matrix dimension differs from the energy list")
    probabilities = np.diag(density).real
    values = {"energy": float(np.dot(energies, probabilities))}
    for i in range(n):
        diagonal = np.array([1 - 2*((x >> i) & 1) for x in range(2**n)])
        values[f"Z{i}"] = float(np.dot(diagonal, probabilities))
    if n > 1:
        values["Z0Z1"] = float(sum((1-2*(x&1))*(1-2*((x>>1)&1))*p for x, p in enumerate(probabilities)))
    return values


def gibbs_check(energies: Sequence[float], beta: float) -> dict:
    """Report actual reduced-state error, positivity, trace and thermal means."""
    energies, n = _energies(energies)
    actual, expected = reduced_gibbs_state(energies, beta), exact_diagonal_gibbs(energies, beta)
    difference = actual - expected
    return {"beta": float(beta), "system_qubits": n,
            "trace_distance": float(np.linalg.svd(difference, compute_uv=False).sum()/2),
            "max_entry_error": float(np.max(np.abs(difference))),
            "trace_real": float(np.trace(actual).real),
            "minimum_eigenvalue": float(np.linalg.eigvalsh(actual).min()),
            "purification_success_probability": 1.0,
            "actual_observables": thermal_observables(energies, actual),
            "reference_observables": thermal_observables(energies, expected)}


def bounded_mean_shots(error: float, failure_probability: float,
                       outcome_min: float = -1.0, outcome_max: float = 1.0) -> int:
    """Hoeffding count for one fixed iid bounded scalar mean only."""
    error, delta = _real(error, "error"), _real(failure_probability, "failure_probability")
    low, high = _real(outcome_min, "outcome_min"), _real(outcome_max, "outcome_max")
    if error <= 0 or not 0 < delta < 1 or high <= low:
        raise ValueError("need positive error, 0 < failure_probability < 1 and positive outcome width")
    if not math.isfinite(high-low):
        raise ValueError("the outcome width must be finite")
    # Logarithms avoid squaring overflow, squared-error underflow and 2/delta
    # overflow. This bounded implementation rejects an unrepresentable count.
    log_count = 2*(math.log(high-low)-math.log(error)) + math.log(math.log(2)-math.log(delta)) - math.log(2)
    if log_count >= math.log(np.finfo(float).max)-1e-12:
        raise ValueError("shot count exceeds the supported floating budget domain")
    return max(1, math.ceil(math.exp(log_count)))
