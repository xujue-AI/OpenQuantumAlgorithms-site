# SPDX-License-Identifier: MIT
"""Explicit spin-chain product formulas and bounded independent classical checks.

The circuit generator never constructs a Hamiltonian matrix. Dense references
are separate functions limited to eight qubits, and are not quantum circuits.
Qiskit strings and basis indices order wires q[n-1] ... q[0].
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_DENSE_QUBITS = 8
MAX_GENERATOR_QUBITS = 64
MAX_TERM_EXPONENTIALS = 200_000
MAX_STEPS = 20_000


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def spin_terms(n_qubits: int, J: float = 1.0, h: float = 0.7) -> list[tuple[float, str]]:
    """Open-boundary H = J sum ZZ + h sum X; omit zero terms."""
    _integer(n_qubits, "n_qubits")
    if n_qubits > MAX_GENERATOR_QUBITS:
        raise ValueError(f"generation is limited to {MAX_GENERATOR_QUBITS} qubits")
    J, h = _finite(J, "J"), _finite(h, "h")
    terms = []
    for i in range(n_qubits - 1):
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - i] = pauli[n_qubits - 2 - i] = "Z"
        if J:
            terms.append((J, "".join(pauli)))
    for i in range(n_qubits):
        pauli = ["I"] * n_qubits
        pauli[n_qubits - 1 - i] = "X"
        if h:
            terms.append((h, "".join(pauli)))
    return terms


def _zz_layer(circuit: QuantumCircuit, n: int, angle: float) -> None:
    if angle:
        for i in range(n - 1):
            circuit.cx(i, i + 1)
            circuit.rz(angle, i + 1)
            circuit.cx(i, i + 1)


def _x_layer(circuit: QuantumCircuit, n: int, angle: float) -> None:
    if angle:
        for i in range(n):
            circuit.rx(angle, i)


def product_formula_circuit(n_qubits: int = 3, time: float = 1.0, steps: int = 16,
                            order: int = 2, J: float = 1.0, h: float = 0.7,
                            *, initial_basis: int = 0, measure: bool = False) -> QuantumCircuit:
    """Prepare a basis state and evolve using chronological ZZ then X layers.

    Order 1 applies full ZZ then full X in every step; order 2 applies half ZZ,
    full X, half ZZ. Each commuting layer is an explicit RX or CX-RZ-CX circuit.
    Construction is bounded by 64 qubits, 20,000 steps and 200,000 term exponentials, apart
    from later transpilation overhead. There is no statevector allocation here.
    """
    terms = spin_terms(n_qubits, J, h)
    time, J, h = _finite(time, "time"), _finite(J, "J"), _finite(h, "h")
    _integer(steps, "steps")
    if steps > MAX_STEPS:
        raise ValueError(f"construction is limited to {MAX_STEPS} steps even for zero evolution")
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    _integer(initial_basis, "initial_basis", 0)
    if initial_basis >= 2**n_qubits:
        raise ValueError("initial_basis is outside the register")
    count = steps * (len(terms) + (n_qubits - 1 if order == 2 and J else 0))
    if count > MAX_TERM_EXPONENTIALS:
        raise ValueError(f"construction exceeds {MAX_TERM_EXPONENTIALS} term exponentials")
    circuit = QuantumCircuit(n_qubits, name=f"PF{order} r={steps}")
    for i in range(n_qubits):
        if (initial_basis >> i) & 1:
            circuit.x(i)
    dt = time / steps
    for _ in range(steps):
        if order == 1:
            _zz_layer(circuit, n_qubits, 2 * J * dt)
            _x_layer(circuit, n_qubits, 2 * h * dt)
        else:
            _zz_layer(circuit, n_qubits, J * dt)
            _x_layer(circuit, n_qubits, 2 * h * dt)
            _zz_layer(circuit, n_qubits, J * dt)
    if measure:
        circuit.measure_all()
    return circuit


def decompose(circuit: QuantumCircuit) -> QuantumCircuit:
    """Primitive rotation basis, deterministic seed; arbitrary rotations remain."""
    result = transpile(circuit, basis_gates=ROTATION_BASIS,
                       optimization_level=1, seed_transpiler=7)
    unsupported = set(result.count_ops()) - set(ROTATION_BASIS) - {"measure", "barrier"}
    if unsupported:
        raise ValueError(f"unsupported gates after decomposition: {unsupported}")
    return result


def pauli_matrix(label: str) -> np.ndarray:
    """Independent Kronecker reference, rather than a circuit-library matrix."""
    matrices = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
                "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    if not label or len(label) > MAX_DENSE_QUBITS or any(p not in matrices for p in label):
        raise ValueError("dense Pauli reference needs 1–8 I/X/Y/Z characters")
    result = np.array([[1.0 + 0j]])
    for p in label:
        result = np.kron(result, matrices[p])
    return result


def dense_hamiltonian(n_qubits: int = 3, J: float = 1.0, h: float = 0.7) -> np.ndarray:
    """Bounded independent classical answer; never used by the generator."""
    _integer(n_qubits, "n_qubits")
    if n_qubits > MAX_DENSE_QUBITS:
        raise ValueError(f"dense validation is limited to {MAX_DENSE_QUBITS} qubits")
    H = np.zeros((2**n_qubits, 2**n_qubits), dtype=complex)
    for coefficient, pauli in spin_terms(n_qubits, J, h):
        H += coefficient * pauli_matrix(pauli)
    return H


def exact_evolution(n_qubits: int = 3, time: float = 1.0,
                    J: float = 1.0, h: float = 0.7) -> np.ndarray:
    return expm(-1j * _finite(time, "time") * dense_hamiltonian(n_qubits, J, h))


def compare_product_formula(n_qubits: int = 3, time: float = 1.0, steps: int = 16,
                            order: int = 2, J: float = 1.0, h: float = 0.7) -> dict:
    """Spectral operator error, vector error, infidelity and Z(q0) expectation."""
    exact = exact_evolution(n_qubits, time, J, h)
    circuit = product_formula_circuit(n_qubits, time, steps, order, J, h)
    actual = Operator(circuit).data
    state, reference = Statevector.from_instruction(circuit).data, exact[:, 0]
    observable = pauli_matrix("I" * (n_qubits - 1) + "Z")
    expected_z = float(np.vdot(reference, observable @ reference).real)
    observed_z = float(np.vdot(state, observable @ state).real)
    return {"order": order, "steps": steps,
            "operator_error": float(np.linalg.norm(actual - exact, 2)),
            "state_vector_error": float(np.linalg.norm(state - reference)),
            "state_infidelity": float(max(0.0, 1 - abs(np.vdot(reference, state))**2)),
            "observable": "Z(q0)", "exact_observable": expected_z,
            "circuit_observable": observed_z, "observable_error": abs(observed_z - expected_z)}


def observable_shots(statistical_error: float = 0.03, statistical_failure: float = 0.005) -> int:
    """Hoeffding for independent +/-1 outcomes: N >= 2 log(2/delta)/epsilon²."""
    epsilon = _finite(statistical_error, "statistical_error")
    delta = _finite(statistical_failure, "statistical_failure")
    if not 0 < epsilon < 2 or not 0 < delta < 1:
        raise ValueError("need 0 < statistical_error < 2 and 0 < statistical_failure < 1")
    return math.ceil(2 * math.log(2 / delta) / epsilon**2)
