# SPDX-License-Identifier: MIT
"""Explicit repeated-access Hadamard QPE and bounded dyadic phase recovery.

The inference theorem is conditional on exact eigenstate preparation, exact
controlled powers, independent ideal trials and exact-real angular inference.
SDK realization and the atan2 implementation are checked numerically, not
certified. No dense gate, analytical power shortcut or native estimator is used.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from fractions import Fraction
import math
from itertools import islice
from typing import Iterable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

MAX_TARGET_QUBITS = 32
MAX_LEVELS = 16
MAX_POWER = 1 << (MAX_LEVELS - 1)
MAX_CIRCUIT_OPERATIONS = 200_000
MAX_SCHEDULE_OPERATIONS = 200_000
MAX_EXPERIMENT_CIRCUITS = 64
MAX_MODELED_DRAWS = 100_000
MAX_DENSE_TARGET_QUBITS = 3
MAX_SHOTS = 50_000
EXP_TAYLOR_DEGREE = 128


def _integer(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _phase(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError("phase must be finite and in [0, 1)")
    return float(value)


def _probability(value: Fraction | float) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("failure target must be in (0, 1)")
    try:
        target = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError("failure target must be in (0, 1)") from error
    if not 0 < target < 1:
        raise ValueError("failure target must be in (0, 1)")
    return target


def _unit_phase(value: float) -> float:
    normalized = value % 1
    return 0.0 if normalized >= 1 else normalized


def circular_distance(a: float, b: float) -> float:
    if not math.isfinite(a) or not math.isfinite(b):
        raise ValueError("circular inputs must be finite")
    return abs((a - b + .5) % 1 - .5)


def exp_lower_bound(x: Fraction) -> Fraction:
    """Positive finite Taylor sum, an exact rational lower bound to exp(x)."""
    if not isinstance(x, Fraction) or x < 0:
        raise ValueError("x must be a nonnegative Fraction")
    total = term = Fraction(1)
    for j in range(1, EXP_TAYLOR_DEGREE + 1):
        term *= x / j
        total += term
    return total


def ideal_failure_bound(levels: int, shots: int) -> Fraction:
    """Conservative rational upper bound to 4 L exp(-shots/32), capped at one.

    Hoeffding is applied separately to each of the 2L sample proportions with
    deviation 1/8. This is an ideal probability bound, not a numerical proof of
    atan2, synthesis or a fault-tolerant hardware confidence bound.
    """
    _integer("levels", levels, 1, MAX_LEVELS)
    _integer("shots", shots, 1, MAX_SHOTS)
    return min(Fraction(1), Fraction(4 * levels) / exp_lower_bound(Fraction(shots, 32)))


def shots_for_failure(levels: int, failure_target: Fraction | float) -> int:
    """Choose the least permitted m satisfying the exact rational bound."""
    _integer("levels", levels, 1, MAX_LEVELS)
    target = _probability(failure_target)
    upper = min(MAX_SHOTS, MAX_MODELED_DRAWS // (2 * levels))
    if ideal_failure_bound(levels, upper) > target:
        raise ValueError("Requested ideal confidence exceeds the sampled-work guard")
    low, high = 1, upper
    while low < high:
        middle = (low + high) // 2
        if ideal_failure_bound(levels, middle) <= target:
            high = middle
        else:
            low = middle + 1
    return low


def base_phase_angle(target_qubits: int, phase: float) -> float:
    _integer("target_qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    return (2 * math.pi * _phase(phase)) / target_qubits


def actual_angle_phase(target_qubits: int, phase: float) -> float:
    """Binary64 diagnostic of the eigenphase from the actual stored P angles."""
    return _unit_phase(target_qubits * base_phase_angle(target_qubits, phase) / (2 * math.pi))


def source_operation_count(target_qubits: int, power: int, basis: str, *, measure: bool = True, initialize: bool = True) -> int:
    _integer("target_qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    _integer("power", power, 1, MAX_POWER)
    if basis not in {"cos", "sin"}:
        raise ValueError("basis must be cos or sin")
    if not isinstance(measure, bool) or not isinstance(initialize, bool):
        raise ValueError("measure and initialize must be booleans")
    return 5 * target_qubits * power + target_qubits + 2 + (basis == "sin") + measure + initialize * (target_qubits + 1)


def build_experiment(target_qubits: int, phase: float, power: int, basis: str, *, measure: bool = True, initialize: bool = True) -> QuantumCircuit:
    """Fresh product |1> input, k literal controlled U calls and one readout.

    U is tensor_i P(theta), theta=(2pi*phase)/n. Every controlled P(theta)
    uses P(theta/2) on the control; CX; P(-theta/2); CX; P(theta/2) on the data.
    Exact binary halving retains the controlled scalar phase. There is no
    CP(k*theta) shortcut. Default initial resets model fresh trials explicitly.
    """
    count = source_operation_count(target_qubits, power, basis, measure=measure, initialize=initialize)
    if count > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("Materialized experiment exceeds its primitive-operation cap")
    theta = base_phase_angle(target_qubits, phase)
    circuit = QuantumCircuit(target_qubits + 1, 1 if measure else 0, name=f"Hadamard-{basis}-k{power}")
    if initialize:
        circuit.reset(range(target_qubits + 1))
    circuit.x(range(1, target_qubits + 1))
    circuit.h(0)
    for _ in range(power):
        for data in range(1, target_qubits + 1):
            circuit.p(theta / 2, 0)
            circuit.cx(0, data)
            circuit.p(-theta / 2, data)
            circuit.cx(0, data)
            circuit.p(theta / 2, data)
    if basis == "sin":
        circuit.sdg(0)
    circuit.h(0)
    if measure:
        circuit.measure(0, 0)
    if len(circuit.data) != count:
        raise AssertionError("Actual materialization differs from source accounting")
    return circuit


def build_schedule(target_qubits: int, phase: float, levels: int) -> list[QuantumCircuit]:
    _integer("levels", levels, 1, MAX_LEVELS)
    count = sum(source_operation_count(target_qubits, 1 << j, basis) for j in range(levels) for basis in ("cos", "sin"))
    if 2 * levels > MAX_EXPERIMENT_CIRCUITS or count > MAX_SCHEDULE_OPERATIONS:
        raise ValueError("Experiment schedule exceeds circuit or operation guards")
    _phase(phase)
    return [build_experiment(target_qubits, phase, 1 << j, basis) for j in range(levels) for basis in ("cos", "sin")]


def lower_experiment(circuit: QuantumCircuit) -> QuantumCircuit:
    """RZ/CX lowering with actual resets/readout; rotations are unsynthesized."""
    if not isinstance(circuit, QuantumCircuit) or len(circuit.data) > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("Bounded QuantumCircuit required")
    lowered = transpile(circuit, basis_gates=["x", "h", "sdg", "cx", "rz", "reset", "measure"], optimization_level=0, seed_transpiler=7)
    if len(lowered.data) > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("Lowered experiment exceeds its primitive-operation cap")
    return lowered


def analytic_zero_probability(target_qubits: int, phase: float, power: int, basis: str) -> float:
    source_operation_count(target_qubits, power, basis, measure=False, initialize=False)
    angle = power * target_qubits * base_phase_angle(target_qubits, phase)
    signal = math.cos(angle) if basis == "cos" else math.sin(angle)
    return (1 + signal) / 2


def dense_experiment_reference(target_qubits: int, phase: float, power: int, basis: str) -> np.ndarray:
    """Independent block-matrix expression of the complete preparation/body."""
    _integer("target_qubits", target_qubits, 1, MAX_DENSE_TARGET_QUBITS)
    source_operation_count(target_qubits, power, basis, measure=False, initialize=False)
    angle = base_phase_angle(target_qubits, phase)
    data_dim = 1 << target_qubits
    h = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    h_control = np.kron(np.eye(data_dim), h)
    prep = np.zeros((2 * data_dim, 2 * data_dim), complex)
    for index in range(2 * data_dim):
        prep[index ^ (2 * (data_dim - 1)), index] = 1
    diagonal = [np.exp(1j * power * angle * (index >> 1).bit_count()) if index & 1 else 1 for index in range(2 * data_dim)]
    offset = np.kron(np.eye(data_dim), np.diag([1, -1j])) if basis == "sin" else np.eye(2 * data_dim)
    return h_control @ offset @ np.diag(diagonal) @ h_control @ prep


def check_experiment(target_qubits: int, phase: float, power: int, basis: str) -> dict:
    _integer("target_qubits", target_qubits, 1, MAX_DENSE_TARGET_QUBITS)
    body = build_experiment(target_qubits, phase, power, basis, measure=False, initialize=False)
    lowered = lower_experiment(body)
    state = Statevector.from_instruction(body)
    probability = float(state.probabilities(qargs=[0])[0])
    expected = analytic_zero_probability(target_qubits, phase, power, basis)
    matrix = Operator(body).data
    reference = dense_experiment_reference(target_qubits, phase, power, basis)
    source_error = float(np.max(abs(matrix - reference)))
    lower_error = float(np.max(abs(Operator(lowered).data - matrix)))
    if abs(probability - expected) > 2e-11 or source_error > 2e-11 or lower_error > 2e-11:
        raise AssertionError("Hadamard experiment failed independent probability/operator checks")
    return {"performed": True, "zero_probability": probability, "analytic_zero_probability": expected,
            "probability_absolute_error": abs(probability - expected), "full_operator_max_error": source_error,
            "source_to_lowered_operator_max_error": lower_error, "certified": False}


@dataclass(frozen=True)
class FringeCounts:
    power: int
    basis: str
    zeros: int
    shots: int

    def validate(self) -> None:
        _integer("power", self.power, 1, MAX_POWER)
        if self.power & (self.power - 1) or self.basis not in {"cos", "sin"}:
            raise ValueError("Recovery requires dyadic powers and cos/sin labels")
        _integer("shots", self.shots, 1, MAX_SHOTS)
        _integer("zeros", self.zeros, 0, self.shots)


def recover_phase(counts: Iterable[FringeCounts]) -> dict:
    """Nearest circular dyadic lifts of sampled sine/cosine angles.

    No unknown phase, true order or spectrum is supplied to the recovery code.
    Numerical consistency is diagnostic and cannot certify the good event.
    """
    try:
        records = list(islice(counts, 2 * MAX_LEVELS + 1))
    except TypeError as error:
        raise ValueError("Recovery requires a bounded iterable of FringeCounts") from error
    if not 2 <= len(records) <= 2 * MAX_LEVELS:
        raise ValueError("A bounded complete sin/cos schedule is required")
    if not all(isinstance(r, FringeCounts) for r in records):
        raise ValueError("Recovery requires FringeCounts records")
    for record in records:
        record.validate()
    if len({(r.power, r.basis) for r in records}) != len(records):
        raise ValueError("Duplicate fringe record")
    levels = len(records) // 2
    expected = {(1 << j, b) for j in range(levels) for b in ("cos", "sin")}
    if len(records) % 2 or {(r.power, r.basis) for r in records} != expected:
        raise ValueError("Missing or unordered-power fringe schedule")
    shots = {r.shots for r in records}
    if len(shots) != 1 or sum(r.shots for r in records) > MAX_MODELED_DRAWS:
        raise ValueError("Uniform bounded shot counts are required by this protocol")
    m = next(iter(shots))
    lookup = {(r.power, r.basis): r for r in records}
    previous = None
    trace = []
    for j in range(levels):
        k = 1 << j
        c = 2 * lookup[k, "cos"].zeros / m - 1
        s = 2 * lookup[k, "sin"].zeros / m - 1
        if c == 0 and s == 0:
            return {"status": "ambiguous_zero_signal", "phase_estimate": None, "trace": trace,
                    "ideal_failure_upper_bound_exact": str(ideal_failure_bound(levels, m)),
                    "confidence_scope": "conditional ideal good-event failure bound; no numerical/hardware certificate",
                    "ideal_phase_radius_exact": str(Fraction(1, 8 * (1 << (levels - 1)))),
                    "numerical_inference_certified": False, "physical_confidence_certified": False,
                    "counts": [asdict(r) for r in records]}
        theta = _unit_phase(math.atan2(s, c) / (2 * math.pi))
        lift = 0 if previous is None else math.floor(k * previous - theta + .5) % k
        estimate = _unit_phase((theta + lift) / k)
        trace.append({"power": k, "cos_signal": c, "sin_signal": s, "angle_cycles": theta,
                      "lift_integer": lift, "phase_estimate": estimate})
        previous = estimate
    return {"status": "recovered_numerically", "phase_estimate": previous,
            "ideal_phase_radius_exact": str(Fraction(1, 8 * (1 << (levels - 1)))),
            "ideal_failure_upper_bound_exact": str(ideal_failure_bound(levels, m)),
            "confidence_scope": "exact eigenstate, exact controlled powers, IID fresh ideal trials, exact-real angle/lift inference",
            "numerical_inference_certified": False, "physical_confidence_certified": False,
            "trace": trace, "counts": [asdict(r) for r in records]}


def sample_counts(probabilities: list[float], levels: int, shots: int, seed: int, *, stream: int = 0) -> list[FringeCounts]:
    _integer("levels", levels, 1, MAX_LEVELS)
    _integer("shots", shots, 1, MAX_SHOTS)
    _integer("seed", seed, 0, (1 << 63) - 1)
    _integer("stream", stream, 0, (1 << 31) - 1)
    if len(probabilities) != 2 * levels or 2 * levels * shots > MAX_MODELED_DRAWS:
        raise ValueError("Probability schedule or modeled draw guard failed")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Finite Bernoulli probabilities are required")
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    return [FringeCounts(1 << (i // 2), "cos" if i % 2 == 0 else "sin", int(rng.binomial(shots, p)), shots)
            for i, p in enumerate(probabilities)]


def protocol_ledger(target_qubits: int, levels: int, shots: int) -> dict:
    _integer("target_qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    _integer("levels", levels, 1, MAX_LEVELS)
    _integer("shots", shots, 1, MAX_SHOTS)
    if 2 * levels * shots > MAX_MODELED_DRAWS:
        raise ValueError("Modeled fresh-trial guard exceeded")
    calls = 2 * shots * ((1 << levels) - 1)
    trials = 2 * levels * shots
    counts = {"p": 3 * target_qubits * calls, "cx": 2 * target_qubits * calls,
              "h": 2 * trials, "x": target_qubits * trials, "sdg": levels * shots,
              "reset": (target_qubits + 1) * trials, "measure": trials}
    return {"target_qubits": target_qubits, "logical_qubits": target_qubits + 1,
            "levels": levels, "shots_per_fringe": shots, "distinct_experiment_circuits": 2 * levels,
            "complete_fresh_preparations": trials, "maximum_controlled_U_power": 1 << (levels - 1),
            "shot_weighted_controlled_U_calls": calls, "source_gate_counts_all_trials": counts,
            "ideal_phase_radius_exact": str(Fraction(1, 8 * (1 << (levels - 1)))),
            "ideal_failure_upper_bound_exact": str(ideal_failure_bound(levels, shots)),
            "controlled_power_scope": "literal repeated controlled product-P calls; no analytic power shortcut",
            "interrogation_time_scope": "dimensionless base-U calls; not physical wall time",
            "initialization_scope": "actual ideal resets then n X gates for every fresh measured trial",
            "physical_estimate_performed": False, "synthesis_performed": False,
            "exclusions": ["physical zero initialization/reset error and time", "rotation synthesis and hardware faults",
                           "routing/factory/controller schedules", "classical compilation/communication latency"],
            "reference_scope": "commuting product phase gates on |1>^n are classically easy; no application advantage"}
