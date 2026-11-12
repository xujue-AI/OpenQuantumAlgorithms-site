# SPDX-License-Identifier: MIT
"""Explicit HHL circuits for a dyadic, commuting X-projector matrix family.

This structured teaching family can also be solved efficiently classically.
Controlled evolution is built from phase and controlled-RX gates; no dense
matrix, inverse, evolution, or solution state is supplied to a quantum gate.
The reciprocal rotation is an explicit phase-label lookup, exponential in
clock precision. General sparse-matrix access and reciprocal arithmetic are
separate algorithms, not supplied by this generator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Statevector

MAX_SYSTEM_QUBITS = 24
MAX_PHASE_QUBITS = 8
MAX_STATEVECTOR_QUBITS = 12
MAX_DENSE_SYSTEM_QUBITS = 8
MAX_PRIMITIVE_OPERATIONS = 200_000
ROTATION_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")


def _integer(name: str, value: int, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True)
class GridXSystem:
    """A=offset/D I + sum_j weight_j/D (I-X_j)/2, D=2**phase_qubits.

    All eigenvalues are nonzero exact phase-grid points strictly below one.
    ``weights`` are little-endian, acting on data q[j]. Zero weights permit
    spectators. A product of RY gates provides an explicit right-hand side.
    """
    weights: tuple[int, ...] = (1,)
    offset: int = 1
    phase_qubits: int = 2

    def __post_init__(self):
        object.__setattr__(self, "weights", tuple(self.weights))
        _integer("phase_qubits", self.phase_qubits, 1, MAX_PHASE_QUBITS)
        _integer("system width", len(self.weights), 1, MAX_SYSTEM_QUBITS)
        _integer("offset", self.offset, 1, self.denominator-1)
        for value in self.weights:
            _integer("weight", value, 0, self.denominator-1)
        if self.offset + sum(self.weights) >= self.denominator:
            raise ValueError("All matrix eigenvalues must be strictly below one to avoid QPE phase aliasing")

    @property
    def denominator(self) -> int:
        return 1 << self.phase_qubits

    @property
    def n_system(self) -> int:
        return len(self.weights)

    @property
    def lambda_min(self) -> float:
        return self.offset/self.denominator

    @property
    def lambda_max(self) -> float:
        return (self.offset+sum(self.weights))/self.denominator

    @property
    def condition_number(self) -> float:
        return (self.offset+sum(self.weights))/self.offset

    @property
    def phase_labels(self) -> tuple[int, ...]:
        labels = {self.offset}
        for weight in self.weights:
            labels |= {label+weight for label in labels}
        return tuple(sorted(labels))


def input_preparation(n_system: int, angles: Sequence[float] | None = None) -> QuantumCircuit:
    _integer("system width", n_system, 1, MAX_SYSTEM_QUBITS)
    values = (0.0,)*n_system if angles is None else tuple(float(angle) for angle in angles)
    if len(values) != n_system or not all(map(math.isfinite, values)):
        raise ValueError("Supply one finite RY preparation angle per system qubit")
    circuit = QuantumCircuit(n_system, name="Prepare b")
    for qubit, angle in enumerate(values):
        if angle:
            circuit.ry(angle, qubit)
    circuit.metadata = {"angles": list(values), "construction": "product RY state, from clean zero input"}
    return circuit


def qft_circuit(width: int, *, inverse: bool = False) -> QuantumCircuit:
    """Original exact textbook QFT, with explicit controlled phases and swaps."""
    _integer("QFT width", width, 1, MAX_PHASE_QUBITS)
    circuit = QuantumCircuit(width, name="QFT")
    for target in reversed(range(width)):
        circuit.h(target)
        for control in reversed(range(target)):
            circuit.cp(math.pi/(1 << (target-control)), target, control)
    for qubit in range(width//2):
        circuit.swap(qubit, width-qubit-1)
    return circuit.inverse() if inverse else circuit


def controlled_evolution(system: GridXSystem, power: int) -> QuantumCircuit:
    """Controlled exp(2 pi i A power), control first, then data wires.

    Analytic fast-forwarding is possible because the given terms commute.
    The control phase includes the identity component of A; omitting it would
    shift every phase estimate. Controlled RX angles are reduced modulo 4pi,
    preserving their control-dependent global phase.
    """
    _integer("power", power, 1, 1 << (MAX_PHASE_QUBITS-1))
    circuit = QuantumCircuit(system.n_system+1, name=f"ctrl U^{power}")
    numerator = ((2*system.offset + sum(system.weights))*power) % (2*system.denominator)
    if numerator:
        circuit.p(math.pi*numerator/system.denominator, 0)
    for target, weight in enumerate(system.weights, start=1):
        numerator = (weight*power) % (2*system.denominator)
        if numerator:
            circuit.crx(2*math.pi*numerator/system.denominator, 0, target)
    return circuit


def phase_estimation(system: GridXSystem) -> QuantumCircuit:
    """Coherent exact-grid QPE, data first, clock last, no measurements."""
    n = system.n_system
    clock = list(range(n, n+system.phase_qubits))
    circuit = QuantumCircuit(n+system.phase_qubits, name="Exact-grid QPE")
    circuit.h(clock)
    for k, control in enumerate(clock):
        circuit.compose(controlled_evolution(system, 1 << k), [control]+list(range(n)), inplace=True)
    circuit.compose(qft_circuit(system.phase_qubits, inverse=True), clock, inplace=True)
    circuit.metadata = {"phase_qubits": system.phase_qubits, "phase_denominator": system.denominator}
    return circuit


def reciprocal_rotation(system: GridXSystem) -> QuantumCircuit:
    """Explicit equality-conditioned RY(2 asin(offset/y)) for reachable labels.

    Clock wires precede the success flag. Other labels apply identity. This
    finite lookup is valid for the specified exact-grid family; it does not
    approximate a general eigenvalue arithmetic oracle.
    """
    m = system.phase_qubits
    circuit = QuantumCircuit(m+1, name="Reciprocal rotation")
    for label in system.phase_labels:
        zeros = [k for k in range(m) if not ((label >> k) & 1)]
        if zeros:
            circuit.x(zeros)
        theta = 2*math.asin(system.offset/label)
        circuit.append(RYGate(theta).control(m, annotated=False), list(range(m))+[m])
        if zeros:
            circuit.x(zeros)
    circuit.metadata = {"phase_labels": list(system.phase_labels), "C": system.lambda_min,
                        "rotations": len(system.phase_labels), "lookup_scaling": "at most 2**phase_qubits labels"}
    return circuit


def build_hhl(system: GridXSystem = GridXSystem(), *, angles: Sequence[float] | None = None) -> QuantumCircuit:
    """Prepare b; QPE; reciprocal rotation; inverse QPE. No opaque gates."""
    n, m = system.n_system, system.phase_qubits
    circuit = QuantumCircuit(n+m+1, name="Structured HHL")
    circuit.compose(input_preparation(n, angles), list(range(n)), inplace=True)
    qpe = phase_estimation(system)
    circuit.compose(qpe, list(range(n+m)), inplace=True)
    circuit.compose(reciprocal_rotation(system), list(range(n, n+m+1)), inplace=True)
    circuit.compose(qpe.inverse(), list(range(n+m)), inplace=True)
    circuit.metadata = {
        "n_system": n, "phase_qubits": m, "flag_qubit": n+m,
        "weights": list(system.weights), "offset": system.offset, "denominator": system.denominator,
        "lambda_min": system.lambda_min, "lambda_max": system.lambda_max, "kappa": system.condition_number,
        "C": system.lambda_min, "phase_labels": list(system.phase_labels),
        "postselection": "phase register all zero and success flag one",
        "matrix_access": "explicit commuting X-projector evolution with analytic fast-forwarding",
        "input_preparation": input_preparation(n, angles).metadata,
        "amplitude_amplification": "not performed; independent fresh trials are capped",
        "rotation_synthesis": "analytic angles; no concrete Clifford+T synthesis certificate",
    }
    return circuit


def measured_hhl_trial(system: GridXSystem = GridXSystem(), *, angles=None,
                       observable: str = "Z", target: int = 0) -> QuantumCircuit:
    """Full fresh trial with explicit Pauli basis change and final all-wire readout."""
    if observable not in {"X", "Y", "Z"}:
        raise ValueError("observable must be X, Y or Z")
    _integer("observable target", target, 0, system.n_system-1)
    circuit = build_hhl(system, angles=angles)
    if observable == "X":
        circuit.h(target)
    elif observable == "Y":
        circuit.sdg(target)
        circuit.h(target)
    circuit.measure_all()
    circuit.metadata.update({"observable": f"{observable}(q{target})",
                             "measurement": "all clock/flag/data wires at end; rejected branches pay every gate"})
    return circuit


def decompose(circuit: QuantumCircuit) -> QuantumCircuit:
    compiled = transpile(circuit, basis_gates=list(ROTATION_BASIS), optimization_level=1, seed_transpiler=7)
    if len(compiled.data) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError(f"Expanded primitive circuit exceeds {MAX_PRIMITIVE_OPERATIONS} operations")
    if any(item.operation.name not in {*ROTATION_BASIS, "measure", "barrier"} for item in compiled.data):
        raise RuntimeError("An opaque operation survived circuit lowering")
    return compiled


def dense_matrix(system: GridXSystem) -> np.ndarray:
    """Bounded independent classical matrix construction, never used to make gates."""
    if system.n_system > MAX_DENSE_SYSTEM_QUBITS:
        raise ValueError(f"Dense matrix reference capped at {MAX_DENSE_SYSTEM_QUBITS} system qubits")
    size = 1 << system.n_system
    matrix = np.eye(size)*((system.offset+sum(system.weights)/2)/system.denominator)
    indices = np.arange(size)
    for qubit, weight in enumerate(system.weights):
        matrix[indices, indices ^ (1 << qubit)] -= weight/(2*system.denominator)
    return matrix


def compare_hhl(system: GridXSystem = GridXSystem(), *, angles=None) -> dict:
    circuit = build_hhl(system, angles=angles)
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"Statevector validation capped at {MAX_STATEVECTOR_QUBITS} total qubits")
    b = Statevector.from_instruction(input_preparation(system.n_system, angles)).data
    exact = np.linalg.solve(dense_matrix(system), b)
    exact_norm = float(np.linalg.norm(exact))
    target = exact/exact_norm
    data = Statevector.from_instruction(circuit).data
    start = 1 << (system.n_system+system.phase_qubits)
    branch = data[start:start+(1 << system.n_system)]
    success = float(np.vdot(branch, branch).real)
    accepted = branch/math.sqrt(success)
    overlap = abs(np.vdot(target, accepted))
    distance = float(np.linalg.norm(np.outer(accepted, accepted.conj())-np.outer(target, target.conj()), ord=2))
    phase_success = float(sum(abs(v)**2 for index, v in enumerate(data)
                              if (index >> (system.n_system+system.phase_qubits)) & 1))
    signs = 1-2*((np.arange(len(target)) >> 0) & 1)
    return {
        "performed": True, "method": "Qiskit Statevector versus NumPy solve of independently assembled A",
        "target_state_real": target.real.tolist(), "target_state_imag": target.imag.tolist(),
        "accepted_state_real": accepted.real.tolist(), "accepted_state_imag": accepted.imag.tolist(),
        "success_probability": success, "analytic_success_probability": system.lambda_min**2*exact_norm**2,
        "clock_leakage_given_flag_probability": max(0.0, phase_success-success),
        "trace_distance_to_normalized_classical_solution": distance,
        "state_fidelity": float(min(1.0, overlap**2)),
        "classical_solution_norm": exact_norm,
        "classical_residual_norm": float(np.linalg.norm(dense_matrix(system)@exact-b)),
        "observable": "Z(q0)", "quantum_observable": float(np.dot(abs(accepted)**2, signs)),
        "classical_observable": float(np.dot(abs(target)**2, signs)),
        "algorithm_precision_scope": "exact dyadic eigenphases analytically; reported discrepancies are floating numerical checks",
    }


def accepted_observable_workload(success: float, *, statistical_error: float = .03,
                                 statistical_failure: float = .005, acceptance_failure: float = .005) -> dict:
    """Unexecuted IID +/-1 mean proposal, with a separate per-slot retry cap."""
    if not math.isfinite(success) or not 0 < success <= 1:
        raise ValueError("success must lie in (0,1]")
    if not math.isfinite(statistical_error) or statistical_error <= 0:
        raise ValueError("statistical_error must be positive")
    for failure in (statistical_failure, acceptance_failure):
        if not math.isfinite(failure) or not 0 < failure < 1:
            raise ValueError("failure allowances must lie in (0,1)")
    shots = max(1, math.ceil(2*math.log(2/statistical_failure)/statistical_error**2))
    cap = 1 if success == 1 else max(1, math.ceil(math.log(acceptance_failure/shots)/math.log1p(-success)))
    return {"status": "unexecuted numerical-ideal workload proposal", "accepted_samples": shots,
            "statistical_error": statistical_error, "statistical_failure": statistical_failure,
            "acceptance_failure": acceptance_failure, "success_probability_used": success,
            "attempt_cap_per_accepted_slot": cap, "total_capped_attempts": shots*cap,
            "numerical_expected_attempts_until_success_without_cap": shots/success,
            "assumptions": ["IID fresh prepared inputs", "ideal numerical acceptance remains applicable",
                            "one fixed Pauli mean", "all failures pay full attempt schedule"],
            "certified_synthesis_or_physical_success_guarantee": False}


def analytic_success_probability(system: GridXSystem, *, angles=None) -> float:
    """Independent classical spectral-distribution calculation, O(n 2**m).

    A product RY input has X=-1 probability (1-sin(theta_j))/2. Dynamic
    convolution aggregates these known eigenvalue probabilities without
    forming an exponentially large matrix or statevector.
    """
    prep = input_preparation(system.n_system, angles)
    probabilities = {system.offset: 1.0}
    for weight, angle in zip(system.weights, prep.metadata["angles"]):
        minus_probability = (1-math.sin(angle))/2
        updated = {}
        for label, probability in probabilities.items():
            updated[label] = updated.get(label,0.0) + probability*(1-minus_probability)
            updated[label+weight] = updated.get(label+weight,0.0) + probability*minus_probability
        probabilities = updated
    probability = float(sum(value*(system.offset/label)**2 for label,value in probabilities.items()))
    return min(1.0, max(0.0, probability))  # floating roundoff at deterministic success
