# SPDX-License-Identifier: MIT
"""Explicit amplitude-damping dilations and separate bounded channel checks.

No Kraus matrix is appended to a quantum circuit. The circuit uses CRY and
CX with an environment wire initialized to zero; controlled rotations are
lowered to primitive gates before export/counting. The analytic/Kraus routines
are independent classical answers, limited to a single-qubit density matrix.
Qiskit orders computational basis bits environment then system for this demo.
Upstream Qiskit core2.6.2 is Apache-2.0:
https://github.com/Qiskit/qiskit/tree/2.6.2
The exact RY/CRY convention is in circuit/library/standard_gates/ry.py.
GKSL model context: Lindblad1976, doi:10.1007/BF01608499; the circuit and
finite-time algebra here are original, not copied from that paper.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_SYSTEM_QUBITS = 32
MAX_DENSE_TOTAL_QUBITS = 2
INPUT_LABELS = ("zero", "one", "plus", "plus_i")
PAULI_MATRICES = {
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite, real and nonnegative")
    return float(value)


def _probability(value: float) -> float:
    value = _nonnegative(value, "probability")
    if value > 1:
        raise ValueError("probability must lie in [0, 1]")
    return value


def damping_probability(rate: float, time: float) -> float:
    """p=1-exp(-rate*time), stable near zero; rate is in inverse time units."""
    rate, time = _nonnegative(rate, "rate"), _nonnegative(time, "time")
    return -math.expm1(-rate * time)


def damping_parameters(rate: float, time: float) -> dict[str, float]:
    """Keep finite-time survival separately when p rounds to one.

    atan2 avoids reconstructing a small coherence from sqrt(1-rounded_p).
    The final floating rotation angle still has finite precision near pi.
    """
    rate, time = _nonnegative(rate, "rate"), _nonnegative(time, "time")
    tau = rate * time
    probability = -math.expm1(-tau)
    survival = math.exp(-tau/2)
    return {"probability": probability, "population_survival": math.exp(-tau),
            "coherence_survival": survival,
            "rotation_angle": 2 * math.atan2(math.sqrt(probability), survival)}


def prepare_input(circuit: QuantumCircuit, wire: int, label: str) -> None:
    """Gate preparation from |0>: |0>, |1>, |+>, or |+i>."""
    if label not in INPUT_LABELS:
        raise ValueError(f"input label must be one of {INPUT_LABELS}")
    if label == "one":
        circuit.x(wire)
    elif label in ("plus", "plus_i"):
        circuit.h(wire)
        if label == "plus_i":
            circuit.s(wire)


def _append_damping_angle(circuit: QuantumCircuit, system: int,
                           environment: int, angle: float) -> None:
    if system == environment or not 0 <= system < circuit.num_qubits or not 0 <= environment < circuit.num_qubits:
        raise ValueError("system and environment must be distinct circuit wires")
    circuit.cry(angle, system, environment)
    circuit.cx(environment, system)


def append_damping_dilation(circuit: QuantumCircuit, system: int,
                            environment: int, probability: float) -> None:
    """Append CRY(system->environment) then CX(environment->system).

    Environment input must be |0>. The system channel is obtained by discarding
    that environment, never by postselecting one of its measurement outcomes.
    """
    probability = _probability(probability)
    angle = 2 * math.asin(math.sqrt(probability))
    _append_damping_angle(circuit, system, environment, angle)


def damping_dilation(probability: float) -> QuantumCircuit:
    """Two-qubit access circuit: system q0 and zero-input environment q1."""
    circuit = QuantumCircuit(2, name="Amplitude damping dilation")
    append_damping_dilation(circuit, 0, 1, probability)
    return circuit


def finite_time_dilation(rate: float, time: float) -> QuantumCircuit:
    """Two-qubit finite-time channel using its stable survival-aware angle."""
    circuit = QuantumCircuit(2, name="Finite-time damping dilation")
    _append_damping_angle(circuit, 0, 1, damping_parameters(rate, time)["rotation_angle"])
    return circuit


def independent_damping_circuit(n_system: int = 1, rate: float = 0.8,
                                time: float = 1.0, *,
                                inputs: Sequence[str] | None = None,
                                observable: str | None = None,
                                reset: bool = False) -> QuantumCircuit:
    """Generate an explicit tensor product of independent damping channels.

    System wires are q[0:n], environment wires q[n:2n]. ``inputs=None`` leaves
    the quantum system input to the caller and does not prepare arbitrary data.
    Named product inputs are prepared with actual X/H/S gates. A requested
    observable adds its basis changes and measures every wire; environment
    bits are ignored, so this realizes the unconditional channel. ``reset``
    explicitly initializes all wires and is used in exported measured trials.
    It requires named inputs, since resetting a caller-supplied unknown system
    state would erase that input rather than apply the channel to it.
    Generation is capped at32 system qubits independently of simulation.
    """
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 1 <= n_system <= MAX_SYSTEM_QUBITS:
        raise ValueError(f"n_system must be an integer from 1 to {MAX_SYSTEM_QUBITS}")
    parameters = damping_parameters(rate, time)
    if reset and inputs is None:
        raise ValueError("reset=True requires named prepared inputs; do not erase a caller-supplied state")
    if inputs is not None and (len(inputs) != n_system or any(label not in INPUT_LABELS for label in inputs)):
        raise ValueError("inputs must contain one supported label per system wire")
    if observable is not None and observable not in PAULI_MATRICES:
        raise ValueError("observable must be X, Y, Z, or None")
    circuit = QuantumCircuit(2 * n_system, 2 * n_system if observable else 0,
                             name="Independent amplitude damping")
    if reset:
        circuit.reset(range(circuit.num_qubits))
    if inputs is not None:
        for wire, label in enumerate(inputs):
            prepare_input(circuit, wire, label)
    for system in range(n_system):
        _append_damping_angle(circuit, system, n_system + system, parameters["rotation_angle"])
    if observable:
        for system in range(n_system):
            if observable == "Y":
                circuit.sdg(system)
            if observable in ("X", "Y"):
                circuit.h(system)
        circuit.measure(range(circuit.num_qubits), range(circuit.num_qubits))
    circuit.metadata = {
        "n_system": n_system, "n_environment": n_system,
        "rate": float(rate), "time": float(time), "damping_probability": parameters["probability"],
        "population_survival": parameters["population_survival"],
        "coherence_survival": parameters["coherence_survival"],
        "rotation_angle": parameters["rotation_angle"],
        "environment": "zero input; trace/discard or ignore all environment measurement bits",
        "postselection": "none; every trial contributes to the requested system observable",
        "input_preparation": "caller supplied" if inputs is None else list(inputs),
        "observable": observable, "explicit_initial_resets": bool(reset),
    }
    return circuit


def build_amplitude_damping(rate: float = 0.8, time: float = 1.0,
                            input_label: str = "plus", *,
                            observable: str | None = None,
                            reset: bool = False) -> QuantumCircuit:
    return independent_damping_circuit(1, rate, time, inputs=(input_label,),
                                       observable=observable, reset=reset)


def lower_to_rotation_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Bound and lower actual controlled rotations; reset/measure are retained."""
    if circuit.num_parameters:
        raise ValueError("all circuit parameters must be bound")
    if circuit.num_qubits > 2 * MAX_SYSTEM_QUBITS:
        raise ValueError("circuit exceeds the bounded generation width")
    result = transpile(circuit, basis_gates=ROTATION_BASIS,
                       optimization_level=1, seed_transpiler=7)
    unsupported = set(result.count_ops()) - set(ROTATION_BASIS) - {"reset", "measure", "barrier"}
    if unsupported:
        raise ValueError(f"opaque operations remain after lowering: {unsupported}")
    return result


def input_density(label: str) -> np.ndarray:
    """Analytic input states for classical checks; not a circuit gate."""
    vectors = {
        "zero": np.array([1, 0], dtype=complex),
        "one": np.array([0, 1], dtype=complex),
        "plus": np.array([1, 1], dtype=complex) / math.sqrt(2),
        "plus_i": np.array([1, 1j], dtype=complex) / math.sqrt(2),
    }
    if label not in vectors:
        raise ValueError("unsupported analytic input label")
    return np.outer(vectors[label], vectors[label].conj())


def _density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2) or not np.all(np.isfinite(rho)):
        raise ValueError("single-qubit density must be finite and 2x2")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0) or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0):
        raise ValueError("density must be Hermitian and have trace one")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("density must be positive semidefinite")
    return rho


def kraus_operators(probability: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent classical reference; these matrices are never circuit gates."""
    p = _probability(probability)
    return np.diag([1, math.sqrt(1-p)]).astype(complex), np.array([[0, math.sqrt(p)], [0, 0]], dtype=complex)


def analytic_amplitude_damping(rho: np.ndarray, probability: float) -> np.ndarray:
    rho = _density(rho)
    K0, K1 = kraus_operators(probability)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T


def analytic_finite_time_damping(rho: np.ndarray, rate: float, time: float) -> np.ndarray:
    """Independent finite-time density formula, without sqrt(1-rounded_p)."""
    rho = _density(rho)
    rate, time = _nonnegative(rate, "rate"), _nonnegative(time, "time")
    tau = rate * time
    q = math.exp(-tau)
    survival = math.exp(-tau/2)
    p = -math.expm1(-tau)
    return np.array([[rho[0, 0]+p*rho[1, 1], survival*rho[0, 1]],
                     [survival*rho[1, 0], q*rho[1, 1]]], dtype=complex)


def reduced_dilation_state(rho: np.ndarray, probability: float) -> np.ndarray:
    """Evaluate the actual two-qubit gate circuit on a supplied density input.

    Supplying a density matrix initializes this bounded classical simulator,
    not a quantum state-preparation implementation for arbitrary mixed inputs.
    """
    rho = _density(rho)
    environment_zero = np.array([[1, 0], [0, 0]], dtype=complex)
    joint = DensityMatrix(np.kron(environment_zero, rho)).evolve(damping_dilation(probability))
    return partial_trace(joint, [1]).data


def reduced_finite_time_state(rho: np.ndarray, rate: float, time: float) -> np.ndarray:
    """Bounded actual gate check for the survival-aware finite-time channel."""
    rho = _density(rho)
    environment_zero = np.array([[1, 0], [0, 0]], dtype=complex)
    joint = DensityMatrix(np.kron(environment_zero, rho)).evolve(finite_time_dilation(rate, time))
    return partial_trace(joint, [1]).data


def simulate_prepared_damping(rate: float = 0.8, time: float = 1.0,
                              input_label: str = "plus") -> np.ndarray:
    """Checked gate-based input preparation followed by explicit dilation."""
    circuit = build_amplitude_damping(rate, time, input_label)
    if circuit.num_qubits > MAX_DENSE_TOTAL_QUBITS:
        raise ValueError("dense simulation width exceeded")
    return partial_trace(Statevector.from_instruction(circuit), [1]).data


def compare_damping(rate: float = 0.8, time: float = 1.0,
                    input_label: str = "plus") -> dict:
    parameters = damping_parameters(rate, time)
    actual = simulate_prepared_damping(rate, time, input_label)
    reference = analytic_finite_time_damping(input_density(input_label), rate, time)
    difference = actual-reference
    actual_means = {label: float(np.trace(actual @ P).real) for label, P in PAULI_MATRICES.items()}
    reference_means = {label: float(np.trace(reference @ P).real) for label, P in PAULI_MATRICES.items()}
    return {
        "input": input_label, "rate": rate, "time": time, "damping_probability": parameters["probability"],
        "population_survival": parameters["population_survival"],
        "coherence_survival": parameters["coherence_survival"],
        "trace_distance": float(np.linalg.norm(difference, ord="nuc") / 2),
        "matrix_entry_max_error": float(np.max(abs(difference))),
        "trace_real": float(np.trace(actual).real), "trace_imag": float(np.trace(actual).imag),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(actual).min()),
        "circuit_observables": actual_means, "analytic_observables": reference_means,
        "maximum_observable_error": max(abs(actual_means[label]-reference_means[label]) for label in PAULI_MATRICES),
    }


def observable_shots(error: float = 0.03, failure: float = 0.005) -> int:
    """Hoeffding budget for one fixed iid +/-1 observable, not tomography."""
    error, failure = _nonnegative(error, "error"), _nonnegative(failure, "failure")
    if not 0 < error < 2 or not 0 < failure < 1:
        raise ValueError("need 0<error<2 and 0<failure<1")
    return math.ceil(2 * math.log(2/failure) / error**2)
