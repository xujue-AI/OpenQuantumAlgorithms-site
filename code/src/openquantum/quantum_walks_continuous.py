# SPDX-License-Identifier: MIT
"""Explicit continuous-time hypercube walks with bounded classical checks.

The n-dimensional hypercube adjacency is sum_i X_i in the computational
basis. These terms commute, so n RX gates implement its evolution exactly
in the analytic-angle model. No graph matrix is used to construct a gate.
Qiskit wire q[0] is the least significant position bit.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

MAX_GENERATOR_QUBITS = 64
MAX_DISTRIBUTION_QUBITS = 16
MAX_STATEVECTOR_QUBITS = 12
MAX_DENSE_QUBITS = 8
MAX_DENSE_EXPOSURE = 100.0
VALIDATION_TOLERANCE = 1e-12


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _parameters(n: int, time: float, coupling: float, initial_position: int) -> tuple[int, float, float, int, float]:
    _integer(n, "n", 1)
    if n > MAX_GENERATOR_QUBITS:
        raise ValueError(f"generation is limited to {MAX_GENERATOR_QUBITS} position qubits")
    _integer(initial_position, "initial_position")
    if initial_position >= 1 << n:
        raise ValueError("initial_position is outside the position register")
    time, coupling = _finite(time, "time"), _finite(coupling, "coupling")
    exposure = coupling * time
    if not math.isfinite(exposure) or not math.isfinite(2 * exposure):
        raise ValueError("the rotation angle 2*coupling*time must be representable and finite")
    return n, time, coupling, initial_position, exposure


def build_continuous_walk(n: int, time: float, coupling: float = 1.0,
                          initial_position: int = 0, *, measure: bool = False) -> QuantumCircuit:
    """Prepare |initial_position> and apply RX(2*coupling*time) on every bit.

    ``H=coupling*sum_i X_i`` is the unnormalized weighted hypercube adjacency.
    Choosing coupling=1/n gives the normalized adjacency convention. The
    initial clean-zero state is assumed; its device reset/timing is excluded.
    Generation uses at most 64 position qubits and has no dense allocation.
    """
    n, time, coupling, initial_position, exposure = _parameters(n, time, coupling, initial_position)
    if not isinstance(measure, bool):
        raise ValueError("measure must be a boolean")
    circuit = QuantumCircuit(n, n if measure else 0, name="hypercube CT walk")
    for q in range(n):
        if (initial_position >> q) & 1:
            circuit.x(q)
    if exposure:
        for q in range(n):
            circuit.rx(2 * exposure, q)
    if measure:
        circuit.measure(range(n), range(n))
    circuit.metadata = {
        "algorithm": "continuous_time_hypercube_walk",
        "position_qubits": n,
        "vertices": 1 << n,
        "time": time,
        "coupling": coupling,
        "initial_position": initial_position,
        "scaled_time_Jt": exposure,
        "hamiltonian": "H=J*sum_i X_i",
        "normalization": "edge weight J; graph degree n; normalized adjacency uses J=1/n",
        "source_preparation_X_gates": initial_position.bit_count(),
        "source_RX_rotations": n if exposure else 0,
        "final_position_measurements": n if measure else 0,
        "bit_order": "q[n-1]...q[0], q[0] least significant",
        "approximation": "exact commuting factorization with floating analytic angles; no synthesis certificate",
        "included": ["computational-basis X preparation", "all RX term evolutions", "requested final position measurements"],
        "excluded": ["clean-state initialization timing/faults", "physical angle synthesis", "classical decoding/control latency"],
    }
    return circuit


def continuous_position_probability(n: int, time: float, position: int,
                                     coupling: float = 1.0, initial_position: int = 0) -> float:
    """Analytic probability for one specified vertex, without a statevector."""
    n, _, _, initial_position, exposure = _parameters(n, time, coupling, initial_position)
    _integer(position, "position")
    if position >= 1 << n:
        raise ValueError("position is outside the position register")
    h = (position ^ initial_position).bit_count()
    flipped, retained = math.sin(exposure)**2, math.cos(exposure)**2
    return flipped**h * retained**(n-h)


def continuous_return_probability(n: int, time: float, coupling: float = 1.0) -> float:
    """Return to the start; independent of which computational vertex starts."""
    return continuous_position_probability(n, time, 0, coupling=coupling, initial_position=0)


def analytic_continuous_distribution(n: int, time: float, coupling: float = 1.0,
                                     initial_position: int = 0) -> np.ndarray:
    """Full probability vector; classical allocation is capped at 16 bits."""
    n, _, _, initial_position, exposure = _parameters(n, time, coupling, initial_position)
    if n > MAX_DISTRIBUTION_QUBITS:
        raise ValueError(f"full analytic distributions are limited to {MAX_DISTRIBUTION_QUBITS} position qubits")
    flipped, retained = math.sin(exposure)**2, math.cos(exposure)**2
    return np.array([flipped**((x^initial_position).bit_count())
                     * retained**(n-(x^initial_position).bit_count()) for x in range(1 << n)])


def hypercube_hamiltonian(n: int, coupling: float = 1.0) -> np.ndarray:
    """Independent dense weighted adjacency, built by bit-flip edges only.

    This is a reference object, never a circuit construction. Distinct bit
    flips are distinct neighbors, so each adjacency entry has weight J.
    """
    _integer(n, "n", 1)
    if n > MAX_DENSE_QUBITS:
        raise ValueError(f"dense adjacency is limited to {MAX_DENSE_QUBITS} position qubits")
    coupling = _finite(coupling, "coupling")
    matrix = np.zeros((1 << n, 1 << n), dtype=complex)
    for x in range(1 << n):
        for q in range(n):
            matrix[x ^ (1 << q), x] = coupling
    return matrix


def validate_continuous_walk(n: int = 3, time: float = math.pi/4,
                             coupling: float = 1.0, initial_position: int = 0) -> dict:
    """Run an actual statevector and, when bounded, an independent expm check.

    Dense evolution is formed from the time-scaled adjacency, avoiding an
    overflowing unscaled Hamiltonian in valid small-time/large-J examples.
    Dense reference exposure n*abs(J*t) is bounded separately from generation.
    Numerical checks do not certify arbitrary floating-angle synthesis.
    """
    n, _, _, initial_position, exposure = _parameters(n, time, coupling, initial_position)
    if n > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"statevector checking is limited to {MAX_STATEVECTOR_QUBITS} position qubits")
    circuit = build_continuous_walk(n, time, coupling, initial_position)
    state = np.asarray(Statevector.from_instruction(circuit).data)
    actual = np.abs(state)**2
    reference = analytic_continuous_distribution(n, time, coupling, initial_position)
    result = {
        "performed": True,
        "statevector_method": "Qiskit circuit with actual X/RX gates",
        "reference_method": "Hamming-distance product probabilities; bounded bit-flip adjacency expm",
        "probabilities": actual.tolist(),
        "reference_probabilities": reference.tolist(),
        "probability_max_absolute_error": float(np.max(np.abs(actual-reference))),
        "return_probability": float(actual[initial_position]),
        "analytic_return_probability": continuous_return_probability(n, time, coupling),
        "probability_sum": float(actual.sum()),
        "dense_operator_check_performed": False,
        "operator_max_absolute_error": None,
        "state_max_absolute_error": None,
        "scope": "bounded numerical circuit checks; not formal proof or angle-synthesis certificate",
        "numerical_tolerance": VALIDATION_TOLERANCE,
    }
    if n <= MAX_DENSE_QUBITS:
        if n*abs(exposure) > MAX_DENSE_EXPOSURE:
            result["dense_reference_exclusion"] = "scaled dense exposure exceeds 100; analytic distribution check retained"
        else:
            unitary = expm(-1j * hypercube_hamiltonian(n, exposure))
            permutation = np.arange(1 << n) ^ initial_position
            prepared_unitary = unitary[:, permutation]
            result.update({
                "dense_operator_check_performed": True,
                "operator_max_absolute_error": float(np.max(np.abs(Operator(circuit).data-prepared_unitary))),
                "state_max_absolute_error": float(np.max(np.abs(state-unitary[:, initial_position]))),
            })
    residuals = [result["probability_max_absolute_error"], abs(result["probability_sum"]-1)]
    if result["dense_operator_check_performed"]:
        residuals += [result["operator_max_absolute_error"], result["state_max_absolute_error"]]
    if not all(math.isfinite(value) and value <= VALIDATION_TOLERANCE for value in residuals):
        raise AssertionError("continuous-walk circuit/reference residual exceeds the declared 1e-12 tolerance")
    result["numerical_checks_passed"] = True
    return result


def continuous_observable_budget(*, sampling_error: float = .03,
                                 synthesis_bias: float = .01,
                                 sampling_failure: float = .005,
                                 physical_failure: float = .005):
    """One fixed vertex-return indicator, with separate conditional budgets.

    Sampling is Bernoulli in [0,1]. Ideal factorization bias is zero. The
    synthesis allowance is conditional on the shared helper's model scalar
    bounding distribution variation, not a certified output-error bound.
    """
    from .resource_estimation import BoundedObservableBudget
    return BoundedObservableBudget(
        absolute_error_target=sampling_error+synthesis_bias,
        algorithm_bias_bound=0.,
        synthesis_bias_allowance=synthesis_bias,
        sampling_error_allowance=sampling_error,
        sampling_failure_probability=sampling_failure,
        physical_failure_probability=physical_failure,
        outcome_min=0., outcome_max=1.,
    )
