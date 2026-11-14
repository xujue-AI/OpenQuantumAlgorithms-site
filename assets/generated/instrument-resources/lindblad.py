# SPDX-License-Identifier: MIT
"""Positive-time Lie/Strang circuits for one Hamiltonian plus damping jump.

Hamiltonian access is an analytic RY-RZ-RY circuit for hx X+hz Z. Each
dissipative substep uses the explicit survival-aware environment dilation from
open_systems.py and resets that environment before reuse. No dense unitary or
Liouvillian exponential is appended as a quantum operation. Dense 2x2/4x4
references are separate bounded classical checks.

Qiskit core 2.6.2 is Apache-2.0: https://github.com/Qiskit/qiskit/tree/2.6.2
SciPy 1.18.1 is BSD-3-Clause: https://github.com/scipy/scipy/tree/v1.18.1
GKSL context: Lindblad1976, doi:10.1007/BF01608499. Liouvillian product-formula
context: Kliesch et al.2011, doi:10.1103/PhysRevLett.107.120501, with its
2012 erratum doi:10.1103/PhysRevLett.109.119904. The circuits,
matrix conventions and explanatory algebra here are original MIT code.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.linalg import expm

if __package__:
    from .open_systems import (
        INPUT_LABELS, PAULI_MATRICES, ROTATION_BASIS, finite_time_dilation,
        input_density, lower_to_rotation_basis, observable_shots, prepare_input,
    )
else:
    from open_systems import (
        INPUT_LABELS, PAULI_MATRICES, ROTATION_BASIS, finite_time_dilation,
        input_density, lower_to_rotation_basis, observable_shots, prepare_input,
    )

MAX_STEPS = 4096
MAX_PRIMITIVE_OPERATIONS = 60_000
MAX_SIMULATION_STEPS = 256
MAX_VALIDATION_POINTS = 16
MAX_REFERENCE_EVOLUTION_SCALE = 100.0


def _real(value: float, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _problem(time: float, rate: float, hx: float, hz: float) -> tuple[float, float, float, float]:
    time, rate = _real(time, "time", nonnegative=True), _real(rate, "rate", nonnegative=True)
    hx, hz = _real(hx, "hx"), _real(hz, "hz")
    if not math.isfinite(math.hypot(hx, hz)):
        raise ValueError("Hamiltonian coefficient norm must be finite")
    return time, rate, hx, hz


def _steps(steps: int, order: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
        raise ValueError(f"steps must be an integer from 1 to {MAX_STEPS}")
    if isinstance(order, bool) or not isinstance(order, int) or order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if steps * (12 if order == 2 else 9) + 10 > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("estimated primitive materialization exceeds its explicit bound")


def append_exact_hamiltonian(circuit: QuantumCircuit, system: int, time: float,
                             hx: float = 0.7, hz: float = 0.3) -> None:
    """Exact analytic Euler circuit for exp[-i time(hx X+hz Z)].

    theta=atan2(hx,hz), omega=hypot(hx,hz). The matrix is
    RY(theta)RZ(2omega*time)RY(-theta), so chronological gate order starts
    with -theta. There is no inner Hamiltonian product-formula approximation.
    """
    time, _, hx, hz = _problem(time, 0, hx, hz)
    if isinstance(system, bool) or not isinstance(system, int) or not 0 <= system < circuit.num_qubits:
        raise ValueError("system must be a valid circuit wire")
    omega = math.hypot(hx, hz)
    if time == 0 or omega == 0:
        return
    angle = 2 * (omega * time)
    if not math.isfinite(angle):
        raise ValueError("Hamiltonian rotation angle overflowed finite arithmetic")
    theta = math.atan2(hx, hz)
    circuit.ry(-theta, system)
    circuit.rz(angle, system)
    circuit.ry(theta, system)


def build_lindblad_circuit(time: float = 1.0, rate: float = 0.8,
                           hx: float = 0.7, hz: float = 0.3,
                           steps: int = 32, order: int = 2,
                           input_label: str | None = "plus_i", *,
                           observable: str | None = None,
                           initial_system_reset: bool = True) -> QuantumCircuit:
    """Explicit two-wire Lie/Strang simulation and optional measured trial.

    q0 is the system, q1 the environment. Every damping call resets q1,
    discarding previous environment correlations and supplying a fresh zero.
    Positive-time schedules: order1 H(dt),D(dt); order2 H(dt/2),D(dt),H(dt/2).
    Unknown caller input uses input_label=None and initial_system_reset=False.
    Measured exports read both wires and ignore every environment outcome.
    """
    time, rate, hx, hz = _problem(time, rate, hx, hz)
    _steps(steps, order)
    if input_label is not None and input_label not in INPUT_LABELS:
        raise ValueError("unsupported gate-prepared input label")
    if input_label is None and initial_system_reset:
        raise ValueError("do not reset an unknown caller-supplied system input")
    if observable is not None and observable not in PAULI_MATRICES:
        raise ValueError("observable must be X, Y, Z or None")
    circuit = QuantumCircuit(2, 2 if observable is not None else 0,
                             name=f"Lindblad PF{order} r={steps}")
    if initial_system_reset:
        circuit.reset(0)
    if input_label is not None:
        prepare_input(circuit, 0, input_label)
    dt = time / steps
    damping = finite_time_dilation(rate, dt)
    for _ in range(steps):
        append_exact_hamiltonian(circuit, 0, dt if order == 1 else dt/2, hx, hz)
        circuit.reset(1)
        circuit.compose(damping, qubits=[0, 1], inplace=True)
        if order == 2:
            append_exact_hamiltonian(circuit, 0, dt/2, hx, hz)
    if observable is not None:
        if observable == "Y":
            circuit.sdg(0)
        if observable in ("X", "Y"):
            circuit.h(0)
        circuit.measure([0, 1], [0, 1])
    circuit.metadata = {
        "system_qubit": 0, "environment_qubit": 1, "time": time,
        "rate": rate, "hx": hx, "hz": hz, "steps": steps, "order": order,
        "chronological_step": "H(dt), D(dt)" if order == 1 else "H(dt/2), D(dt), H(dt/2)",
        "hamiltonian_access": "analytic RY(-theta), RZ(2omega*subtime), RY(theta)",
        "inner_hamiltonian_approximation": "none in exact-angle model",
        "damping_access": "survival-aware CRY and reverse CX; fresh environment reset each step",
        "environment_initializations": steps, "environment_discards_before_reuse": steps-1,
        "final_environment_handling": "trace/discard for retained state; Z measurement and ignored bit for measured trial",
        "postselection": "none", "input_preparation": input_label or "caller supplied",
        "observable": observable, "initial_system_reset": bool(initial_system_reset),
    }
    return circuit


def _density(rho: np.ndarray) -> np.ndarray:
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (2, 2) or not np.all(np.isfinite(rho)):
        raise ValueError("reference density must be finite and 2x2")
    if not np.allclose(rho, rho.conj().T, atol=1e-12, rtol=0) or not np.isclose(np.trace(rho), 1, atol=1e-12, rtol=0):
        raise ValueError("reference density must be Hermitian with trace one")
    if np.linalg.eigvalsh(rho).min() < -1e-12:
        raise ValueError("reference density must be positive semidefinite")
    return rho


def liouvillian(rate: float = 0.8, hx: float = 0.7, hz: float = 0.3) -> np.ndarray:
    """Independent classical column-vectorized GKSL matrix, dimensions4x4.

    vec(A rho B)=(B.T tensor A)vec(rho). The convention uses
    L rho L† -1/2{L†L,rho}, not a doubled dissipator.
    """
    _, rate, hx, hz = _problem(0, rate, hx, hz)
    if abs(hz) > np.finfo(float).max/2:
        raise ValueError("raw Liouvillian entries would overflow finite arithmetic")
    identity = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    H = hx * X + hz * Z
    # The lowering-jump dissipator is linear in rate. Assemble it directly
    # rather than squaring sqrt(rate), which can overflow for a raw large L.
    lowering_dissipator = rate * np.array([
        [0, 0, 0, 1], [0, -0.5, 0, 0],
        [0, 0, -0.5, 0], [0, 0, 0, -1],
    ], dtype=complex)
    result = -1j * (np.kron(identity, H)-np.kron(H.T, identity)) + lowering_dissipator
    if not np.all(np.isfinite(result)):
        raise ValueError("raw Liouvillian entries overflowed finite arithmetic")
    return result


def exact_channel(time: float = 1.0, rate: float = 0.8,
                   hx: float = 0.7, hz: float = 0.3) -> np.ndarray:
    """Bounded independent dense reference; never a quantum circuit gate."""
    time, rate, hx, hz = _problem(time, rate, hx, hz)
    if time == 0:
        return np.eye(4, dtype=complex)
    scaled_rate, scaled_hx, scaled_hz = time * rate, time * hx, time * hz
    if max(scaled_rate, 2 * math.hypot(scaled_hx, scaled_hz)) > MAX_REFERENCE_EVOLUTION_SCALE:
        raise ValueError("dense reference evolution scale exceeds100")
    # Form t*L from scaled coefficients before building the matrix, avoiding
    # huge intermediate entries when finite rate*time or H*time is small.
    return expm(liouvillian(scaled_rate, scaled_hx, scaled_hz))


def exact_density(rho: np.ndarray, time: float = 1.0, rate: float = 0.8,
                   hx: float = 0.7, hz: float = 0.3) -> np.ndarray:
    rho = _density(rho)
    return (exact_channel(time, rate, hx, hz) @ rho.reshape(4, order="F")).reshape((2, 2), order="F")


def simulate_lindblad(time: float = 1.0, rate: float = 0.8,
                      hx: float = 0.7, hz: float = 0.3,
                      steps: int = 32, order: int = 2,
                      input_label: str = "plus_i") -> np.ndarray:
    """Actual gate/reset DensityMatrix evolution, followed by environment trace."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"dense circuit simulation is capped at {MAX_SIMULATION_STEPS} steps")
    circuit = build_lindblad_circuit(time, rate, hx, hz, steps, order, input_label)
    return partial_trace(DensityMatrix.from_instruction(circuit), [1]).data


def simulate_input_density(rho: np.ndarray, time: float = 1.0, rate: float = 0.8,
                           hx: float = 0.7, hz: float = 0.3,
                           steps: int = 32, order: int = 2) -> np.ndarray:
    """Check a supplied mixed density input; its simulator loading is classical."""
    rho = _density(rho)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps > MAX_SIMULATION_STEPS:
        raise ValueError(f"dense circuit simulation is capped at {MAX_SIMULATION_STEPS} steps")
    circuit = build_lindblad_circuit(time, rate, hx, hz, steps, order, None,
                                    initial_system_reset=False)
    environment_zero = np.array([[1, 0], [0, 0]], dtype=complex)
    joint = DensityMatrix(np.kron(environment_zero, rho)).evolve(circuit)
    return partial_trace(joint, [1]).data


def reconstructed_channel(states: dict[str, np.ndarray]) -> np.ndarray:
    """Reconstruct the column-vectorized channel from four physical input runs."""
    F00, F11 = states["zero"], states["one"]
    symmetric = 2 * states["plus"]-F00-F11
    y_part = 2 * states["plus_i"]-F00-F11
    F01 = (symmetric+1j*y_part)/2
    F10 = (symmetric-1j*y_part)/2
    return np.column_stack([F00.reshape(4, order="F"), F10.reshape(4, order="F"),
                            F01.reshape(4, order="F"), F11.reshape(4, order="F")])


def compare_lindblad(time: float = 1.0, rate: float = 0.8,
                     hx: float = 0.7, hz: float = 0.3,
                     steps: int = 32, order: int = 2,
                     input_label: str = "plus_i") -> dict:
    if input_label not in INPUT_LABELS:
        raise ValueError("unsupported gate-prepared input label")
    states = {label: simulate_lindblad(time, rate, hx, hz, steps, order, label) for label in INPUT_LABELS}
    reference_channel = exact_channel(time, rate, hx, hz)
    actual = states[input_label]
    reference = (reference_channel @ input_density(input_label).reshape(4, order="F")).reshape((2, 2), order="F")
    channel = reconstructed_channel(states)
    actual_means = {name: float(np.trace(actual @ P).real) for name, P in PAULI_MATRICES.items()}
    reference_means = {name: float(np.trace(reference @ P).real) for name, P in PAULI_MATRICES.items()}
    return {
        "order": order, "steps": steps, "time": time, "rate": rate, "hx": hx, "hz": hz,
        "input": input_label, "trace_distance": float(np.linalg.norm(actual-reference, ord="nuc")/2),
        "superoperator_spectral_error": float(np.linalg.norm(channel-reference_channel, 2)),
        "superoperator_metric": "column-vectorized2->2 induced Hilbert-Schmidt norm; not diamond norm",
        "trace_real": float(np.trace(actual).real), "trace_imag": float(np.trace(actual).imag),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(actual).min()),
        "circuit_observables": actual_means, "exact_observables": reference_means,
        "observable_errors": {name: abs(actual_means[name]-reference_means[name]) for name in PAULI_MATRICES},
        "actual_density_real": actual.real.tolist(), "actual_density_imag": actual.imag.tolist(),
        "inner_hamiltonian_approximation": "zero for the analytic exact-angle substep; no inner Trotter product",
    }
