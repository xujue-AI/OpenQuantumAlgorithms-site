# SPDX-License-Identifier: MIT
"""Original bounded circular-grid Bayesian inference from explicit QPE fringes.

This is a discrete single-phase model and a fixed-length greedy experiment
policy. Its credible cell arc is a posterior summary, not a frequentist,
continuous-phase, synthesis or noisy-hardware confidence certificate.
"""
from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Measure, Reset
from qiskit.circuit.library import CXGate, HGate, PhaseGate, RZGate, XGate
from qiskit.quantum_info import Operator, Statevector

MAX_GRID_SIZE = 4096
MAX_TARGET_QUBITS = 32
MAX_DENSE_TARGET_QUBITS = 3
MAX_POWER = 128
MAX_STEPS = 128
MAX_CIRCUIT_OPERATIONS = 200_000
MAX_TRANSCRIPT_OPERATIONS = 1_000_000
MAX_CANDIDATE_POWERS = 8
MAX_CANDIDATE_OFFSETS = 8
DEFAULT_POWERS = (1, 2, 4, 8, 16, 32)
DEFAULT_OFFSETS = (0.0, math.pi / 2, math.pi, 3 * math.pi / 2)
UTILITY_TIE_TOLERANCE = 1e-12
RESULTANT_AMBIGUITY_THRESHOLD = 1e-12


def _integer(name: str, value: int, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _finite(name: str, value: float, low: float, high: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real in [{low}, {high})")
    if isinstance(value, np.floating) and np.dtype(type(value)).itemsize > 8:
        raise ValueError(f"{name} requires binary64 or narrower input")
    try:
        result = float(value)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} is not a representable finite real") from error
    if not math.isfinite(result) or not low <= result < high:
        raise ValueError(f"{name} must be a finite real in [{low}, {high})")
    return result


def _array(name: str, values: Sequence[float]) -> np.ndarray:
    if not isinstance(values, (list, tuple, np.ndarray)):
        raise ValueError(f"{name} requires a bounded numerical vector")
    if isinstance(values, np.ndarray) and values.ndim != 1:
        raise ValueError(f"{name} requires a one-dimensional real numerical vector")
    if not 1 <= len(values) <= MAX_GRID_SIZE:
        raise ValueError(f"{name} length must be in [1, {MAX_GRID_SIZE}]")
    if isinstance(values, np.ndarray):
        if values.ndim != 1 or values.dtype.kind not in "fiu" or values.dtype.itemsize > 8:
            raise ValueError(f"{name} requires a one-dimensional real numerical vector")
    elif any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating))
             or (isinstance(v, np.floating) and np.dtype(type(v)).itemsize > 8) for v in values):
        raise ValueError(f"{name} requires finite real entries, not booleans")
    try:
        array = np.asarray(values, dtype=float)
    except (OverflowError, ValueError, TypeError) as error:
        raise ValueError(f"{name} contains an unrepresentable real entry") from error
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} requires a finite numerical vector")
    return array


def phase_grid(size: int = 512) -> np.ndarray:
    """Exactly represented dyadic atoms i/G on the phase circle."""
    _integer("grid size", size, 16, MAX_GRID_SIZE)
    if size & (size - 1):
        raise ValueError("Grid size must be a power of two")
    return np.arange(size, dtype=float) / size


def _grid(grid: Sequence[float]) -> np.ndarray:
    result = _array("grid", grid)
    expected = phase_grid(len(result))
    if not np.array_equal(result, expected):
        raise ValueError("Grid must contain the ordered dyadic atoms i/G")
    return result


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Normalize finite nonnegative weights while retaining zero support.

    Unsupported positive-support underflow is rejected rather than silently
    clipping probabilities or introducing a replacement prior.
    """
    values = _array("weights", weights)
    if np.any(values < 0) or not np.any(values > 0):
        raise ValueError("Weights must be nonnegative with positive total mass")
    scaled = values / float(np.max(values))
    if np.any((values > 0) & (scaled == 0)):
        raise ValueError("Positive prior support underflows during normalization")
    total = math.fsum(float(v) for v in scaled)
    result = scaled / total
    if np.any((scaled > 0) & (result == 0)) or not np.all(np.isfinite(result)):
        raise ValueError("Positive posterior support is not representable")
    return result


def likelihood(grid: Sequence[float], power: int, offset: float) -> np.ndarray:
    """Ideal zero-readout probabilities for the finite single-phase model."""
    points = _grid(grid)
    _integer("power", power, 1, MAX_POWER)
    beta = _finite("offset", offset, 0, 2 * math.pi)
    result = (1 + np.cos(2 * math.pi * power * points - beta)) / 2
    if np.any(result < 0) or np.any(result > 1) or not np.all(np.isfinite(result)):
        raise ValueError("Likelihood evaluation left its probability domain")
    return result


def _weighted_branch(weights: np.ndarray, probabilities: np.ndarray) -> tuple[float, np.ndarray | None]:
    raw = weights * probabilities
    if np.any((weights > 0) & (probabilities > 0) & (raw == 0)):
        raise ValueError("Positive likelihood support underflows")
    raw_evidence = math.fsum(float(v) for v in raw)
    if raw_evidence == 0:
        return 0.0, None
    evidence = raw_evidence / math.fsum(float(v) for v in weights)
    if not math.isfinite(evidence) or not 0 <= evidence <= 1:
        raise ValueError("Predictive evidence is not a finite normalized probability")
    return evidence, normalize_weights(raw)


def update_posterior(weights: Sequence[float], grid: Sequence[float], power: int,
                     offset: float, outcome: int) -> np.ndarray:
    """One Bayes update; an impossible observed result is a model contradiction."""
    prior = normalize_weights(weights)
    points = _grid(grid)
    if len(prior) != len(points):
        raise ValueError("Prior and grid lengths differ")
    _integer("outcome", outcome, 0, 1)
    zero = likelihood(points, power, offset)
    _, posterior = _weighted_branch(prior, zero if outcome == 0 else 1 - zero)
    if posterior is None:
        raise ValueError("Observed outcome has zero evidence in the supplied grid prior")
    return posterior


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0]
    return -math.fsum(float(v) for v in positive * np.log(positive))


def _unit_phase(value: float) -> float:
    normalized = value % 1
    return 0.0 if normalized >= 1 else normalized


def choose_experiment(weights: Sequence[float], grid: Sequence[float],
                      powers: Sequence[int] = DEFAULT_POWERS,
                      offsets: Sequence[float] = DEFAULT_OFFSETS) -> dict:
    """Greedy expected-entropy experiment, using only the current posterior.

    The fixed 1e-12-nat tie policy prefers the earliest supplied experiment.
    The default order prefers shorter powers in ties. This is a bounded
    numerical policy, not a claim of resource-optimal or noise-robust design.
    """
    prior = normalize_weights(weights)
    points = _grid(grid)
    if len(prior) != len(points):
        raise ValueError("Prior and grid lengths differ")
    if not isinstance(powers, (tuple, list)) or not 1 <= len(powers) <= MAX_CANDIDATE_POWERS:
        raise ValueError("A bounded power schedule is required")
    for power in powers:
        _integer("candidate power", power, 1, MAX_POWER)
        if power & (power - 1):
            raise ValueError("Candidate powers must be dyadic")
    if len(set(powers)) != len(powers) or len(points) < 4 * max(powers):
        raise ValueError("Powers must be unique and grid size must be at least four times the maximum power")
    if not isinstance(offsets, (tuple, list)) or not 1 <= len(offsets) <= MAX_CANDIDATE_OFFSETS:
        raise ValueError("A bounded offset schedule is required")
    betas = tuple(_finite("candidate offset", v, 0, 2 * math.pi) for v in offsets)
    if any(2 * (v / 2) != v for v in betas):
        raise ValueError("Candidate offsets must be exactly binary64-halvable")
    if len(set(betas)) != len(betas):
        raise ValueError("Offsets must be unique")
    entropy = _entropy(prior)
    best = None
    for power in powers:
        for beta in betas:
            zero = likelihood(points, power, beta)
            p0, posterior0 = _weighted_branch(prior, zero)
            p1, posterior1 = _weighted_branch(prior, 1 - zero)
            expected = (p0 * _entropy(posterior0) if posterior0 is not None else 0.0)
            expected += (p1 * _entropy(posterior1) if posterior1 is not None else 0.0)
            gain = entropy - expected
            if not math.isfinite(gain) or abs(p0 + p1 - 1) > 1e-12 or gain < -1e-12:
                raise ValueError("Predictive entropy calculation failed its numerical checks")
            candidate = {"power": power, "offset": beta, "expected_information_gain": gain,
                         "predictive_zero_probability": p0}
            if best is None or gain > best["expected_information_gain"] + UTILITY_TIE_TOLERANCE:
                best = candidate
    return dict(best, candidates_evaluated=len(powers) * len(betas),
                utility_tie_tolerance_nats=UTILITY_TIE_TOLERANCE,
                policy_scope="finite-grid greedy expected entropy; fixed schedule-order ties; not resource optimal")


def posterior_summary(weights: Sequence[float], grid: Sequence[float], mass: float = .95) -> dict:
    """MAP, ambiguous circular mean and shortest credible GRID-CELL arc.

    The selected consecutive atoms carry at least the requested posterior
    mass. Half-cell endpoints prevent a point-mass prior from being presented
    as continuous zero-width resolution. This is not a disjoint HPD set.
    """
    values = normalize_weights(weights)
    points = _grid(grid)
    if len(values) != len(points):
        raise ValueError("Posterior and grid lengths differ")
    q = _finite("credible mass", mass, 0, 1)
    if q == 0:
        raise ValueError("Credible mass must lie strictly between zero and one")
    count = len(values)
    doubled = np.concatenate((values, values))
    total = math.fsum(float(v) for v in values)
    best = None
    end = 0
    for start in range(count):
        # Nonnegative masses make the minimal right endpoint monotone as the
        # left endpoint advances. At most 2G extensions and G final fsum calls
        # each visit at most G values: an explicitly bounded O(G^2) scan.
        end = max(end, start + 1)
        covered = math.fsum(float(v) for v in doubled[start:end]) / total
        while end < start + count and covered < q:
            end += 1
            covered = math.fsum(float(v) for v in doubled[start:end]) / total
        if covered < q:
            raise ValueError("Normalized posterior cannot represent the requested mass")
        candidate = (end - start, start, covered)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    width_nodes, start_index, covered = best
    start = ((start_index - .5) / count) % 1
    end = ((start_index + width_nodes - .5) / count) % 1
    vector = sum(complex(w) * complex(np.exp(2j * math.pi * x)) for w, x in zip(values, points))
    resultant = abs(vector)
    mean = None if resultant <= RESULTANT_AMBIGUITY_THRESHOLD else _unit_phase(math.atan2(vector.imag, vector.real) / (2 * math.pi))
    return {"map_phase": float(points[int(np.argmax(values))]), "mean_phase": mean,
            "resultant_length": resultant, "grid_spacing": 1 / count,
            "credible_arc": {"start": start, "end": end, "width": width_nodes / count,
                             "mass": covered, "target_mass": q, "node_count": width_nodes,
                             "wraps": width_nodes < count and end <= start, "full_circle": width_nodes == count},
            "credible_scope": "discrete posterior grid-cell arc, not continuous or frequentist coverage; not a disjoint HPD set",
            "certified": False}


def base_phase_angle(target_qubits: int, phase: float) -> float:
    _integer("target qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    phi = _finite("phase", phase, 0, 1)
    theta = 2 * math.pi * phi / target_qubits
    if (phi != 0 and theta == 0) or 2 * (theta / 2) != theta:
        raise ValueError("Stored base-P angle must be nonzero and exactly binary64-halvable unless phase is zero")
    return theta


def actual_angle_phase(target_qubits: int, phase: float) -> float:
    return _unit_phase(target_qubits * base_phase_angle(target_qubits, phase) / (2 * math.pi))


def source_operation_count(target_qubits: int, power: int, offset: float,
                           *, measure: bool = True, initialize: bool = True) -> int:
    _integer("target qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    _integer("power", power, 1, MAX_POWER)
    beta = _finite("offset", offset, 0, 2 * math.pi)
    if 2 * (beta / 2) != beta:
        raise ValueError("Stored offset must be exactly binary64-halvable")
    if not isinstance(measure, bool) or not isinstance(initialize, bool):
        raise ValueError("Measurement and initialization flags must be booleans")
    return 5 * target_qubits * power + target_qubits + 3 + measure + initialize * (target_qubits + 1)


def build_experiment(target_qubits: int, phase: float, power: int, offset: float,
                     *, measure: bool = True, initialize: bool = True) -> QuantumCircuit:
    """Fresh |1> product input, literal repeated controlled U and one readout."""
    count = source_operation_count(target_qubits, power, offset, measure=measure, initialize=initialize)
    if count > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("Experiment exceeds its primitive-operation cap")
    theta = base_phase_angle(target_qubits, phase)
    circuit = QuantumCircuit(target_qubits + 1, 1 if measure else 0, name=f"Bayes fringe k{power}")
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
    circuit.p(-float(offset), 0)
    circuit.h(0)
    if measure:
        circuit.measure(0, 0)
    if len(circuit.data) != count:
        raise AssertionError("Literal construction differs from source accounting")
    return circuit


def lower_experiment(circuit: QuantumCircuit) -> QuantumCircuit:
    if not isinstance(circuit, QuantumCircuit) or len(circuit.data) > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("A bounded circuit is required")
    if not 1 <= circuit.num_qubits <= MAX_TARGET_QUBITS + 1 or circuit.num_clbits > 1:
        raise ValueError("Lowering exceeds the experiment wire/bit bounds")
    _finite("circuit global phase", circuit.global_phase, -1000, 1000)
    classes = {"x": {XGate, type(XGate())}, "h": {HGate, type(HGate())},
               "p": {PhaseGate}, "rz": {RZGate}, "cx": {CXGate, type(CXGate())},
               "reset": {Reset, type(Reset())}, "measure": {Measure, type(Measure())}}
    for item in circuit.data:
        operation = item.operation
        name = operation.name
        if name not in classes or type(operation) not in classes[name]:
            raise ValueError("Only actual standard experiment primitives may be lowered")
        expected_qubits = 2 if name == "cx" else 1
        expected_bits = 1 if name == "measure" else 0
        expected_params = 1 if name in {"p", "rz"} else 0
        if (len(item.qubits) != expected_qubits or len(item.clbits) != expected_bits
                or len(operation.params) != expected_params or len(set(item.qubits)) != expected_qubits):
            raise ValueError("Primitive operand/parameter shape is invalid")
        for parameter in operation.params:
            _finite("primitive angle", parameter, -1000, 1000)
    lowered = transpile(circuit, basis_gates=["x", "h", "cx", "rz", "reset", "measure"],
                        optimization_level=0, seed_transpiler=7)
    if len(lowered.data) > MAX_CIRCUIT_OPERATIONS:
        raise ValueError("Lowered circuit exceeds its operation cap")
    return lowered


def analytic_zero_probability(target_qubits: int, phase: float, power: int, offset: float) -> float:
    source_operation_count(target_qubits, power, offset, measure=False, initialize=False)
    theta = base_phase_angle(target_qubits, phase)
    return (1 + math.cos(power * target_qubits * theta - float(offset))) / 2


def check_experiment(target_qubits: int, phase: float, power: int, offset: float) -> dict:
    """Independent complete operator and scalar probability for n<=3."""
    _integer("dense target qubits", target_qubits, 1, MAX_DENSE_TARGET_QUBITS)
    body = build_experiment(target_qubits, phase, power, offset, measure=False, initialize=False)
    dim = 1 << target_qubits
    h = np.array([[1, 1], [1, -1]], complex) / math.sqrt(2)
    control_h = np.kron(np.eye(dim), h)
    prep = np.zeros((2 * dim, 2 * dim), complex)
    for index in range(2 * dim):
        prep[index ^ (2 * (dim - 1)), index] = 1
    theta = base_phase_angle(target_qubits, phase)
    diagonal = [np.exp(1j * power * theta * (index >> 1).bit_count()) if index & 1 else 1 for index in range(2 * dim)]
    offset_gate = np.kron(np.eye(dim), np.diag([1, np.exp(-1j * float(offset))]))
    reference = control_h @ offset_gate @ np.diag(diagonal) @ control_h @ prep
    matrix = Operator(body).data
    lowered = lower_experiment(body)
    raw_probabilities = Statevector.from_instruction(body).probabilities(qargs=[0])
    probability_sum = math.fsum(float(v) for v in raw_probabilities)
    if (not np.all(np.isfinite(raw_probabilities)) or np.any(raw_probabilities < 0)
            or abs(probability_sum - 1) > 2e-11):
        raise AssertionError("Raw Born probabilities failed their independent normalization guard")
    # Normalize only the checked floating-point state norm, retaining both raw
    # values. Independent full-operator checks still reject gross deviations.
    probability = float(raw_probabilities[0]) / probability_sum
    expected = analytic_zero_probability(target_qubits, phase, power, offset)
    operator_error = float(np.max(abs(matrix - reference)))
    lower_error = float(np.max(abs(Operator(lowered).data - matrix)))
    if operator_error > 2e-11 or lower_error > 2e-11 or abs(probability - expected) > 2e-11:
        raise AssertionError("Bayesian experiment failed independent operator/probability checks")
    return {"performed": True, "zero_probability": probability, "analytic_zero_probability": expected,
            "raw_zero_probability": float(raw_probabilities[0]), "raw_probability_sum": probability_sum,
            "born_normalization_correction": abs(probability - float(raw_probabilities[0])),
            "born_scope": "normalized checked binary64 statevector; raw norm and independent operator guards retained",
            "probability_absolute_error": abs(probability - expected), "full_operator_max_error": operator_error,
            "source_to_lowered_operator_max_error": lower_error, "certified": False}


def transcript_ledger(target_qubits: int, powers: Sequence[int], *, grid_size: int,
                      candidates: int = 24, preflight_policy_evaluations: int = 0) -> dict:
    _integer("target qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    phase_grid(grid_size)
    _integer("policy candidate count", candidates, 1, MAX_CANDIDATE_POWERS * MAX_CANDIDATE_OFFSETS)
    _integer("preflight policy evaluations", preflight_policy_evaluations, 0, 1)
    if not isinstance(powers, (tuple, list)) or not 1 <= len(powers) <= MAX_STEPS:
        raise ValueError("A bounded fixed-length transcript is required")
    for power in powers:
        _integer("transcript power", power, 1, MAX_POWER)
    calls, steps = sum(powers), len(powers)
    counts = {"p": 3 * target_qubits * calls + steps, "cx": 2 * target_qubits * calls,
              "h": 2 * steps, "x": target_qubits * steps, "reset": (target_qubits + 1) * steps,
              "measure": steps}
    total = sum(counts.values())
    if total > MAX_TRANSCRIPT_OPERATIONS:
        raise ValueError("Transcript exceeds its complete source-operation cap")
    return {"logical_qubits": target_qubits + 1, "complete_fresh_preparations": steps,
            "shot_weighted_controlled_U_calls": calls, "maximum_controlled_U_power": max(powers),
            "source_gate_counts_all_trials": counts, "source_operations_all_trials": total,
            "classical_posterior_updates": steps,
            "classical_policy_calls": steps + preflight_policy_evaluations,
            "classical_preflight_policy_calls": preflight_policy_evaluations,
            "classical_candidate_likelihood_values": (steps + preflight_policy_evaluations) * grid_size * candidates,
            "classical_observation_likelihood_values": steps * grid_size,
            "classical_credible_arc_summaries": 1, "credible_arc_fsum_value_visit_upper_bound": 3 * grid_size * grid_size,
            "classical_complexity_scope": "O(N G C) policy and updates; final grid-cell arc O(G^2), at most 3G fsum windows of at most G values",
            "grid_cells": grid_size, "policy_candidates_per_step": candidates,
            "readout_bits": steps, "physical_estimate_performed": False,
            "source_scope": "complete recorded fixed-length adaptive path, including resets/input/powers/offset/readout; P counts are unsynthesized",
            "excluded": ["rotation synthesis and its certificate", "physical factories/code distance/runtime/faults", "controller and decoder latency", "generic application state preparation", "continuous-grid error certificate"]}


def run_adaptive(phase: float, *, target_qubits: int = 1, steps: int = 48,
                 grid_size: int = 512, seed: int = 7,
                 powers: Sequence[int] = DEFAULT_POWERS,
                 offsets: Sequence[float] = DEFAULT_OFFSETS) -> dict:
    """Classical ideal draws from checked circuits; no QPU execution.

    All requested observations are charged. There is no success-triggered
    early stop and no theorem that the final credible arc covers the phase.
    """
    _integer("dense target qubits", target_qubits, 1, MAX_DENSE_TARGET_QUBITS)
    _integer("steps", steps, 1, MAX_STEPS)
    _integer("seed", seed, 0, (1 << 63) - 1)
    nominal = _finite("phase", phase, 0, 1)
    true_phase = actual_angle_phase(target_qubits, nominal)
    points = phase_grid(grid_size)
    prior = np.ones(grid_size) / grid_size
    # Validate the policy and maximum complete workload before simulation.
    preflight_start = time.perf_counter()
    choose_experiment(prior, points, powers, offsets)
    preflight_seconds = time.perf_counter() - preflight_start
    transcript_ledger(target_qubits, [max(powers)] * steps, grid_size=grid_size,
                      candidates=len(powers) * len(offsets))
    rng = np.random.default_rng(seed)
    cache, trace = {}, []
    policy_seconds = check_seconds = update_seconds = 0.0
    for index in range(steps):
        start = time.perf_counter()
        choice = choose_experiment(prior, points, powers, offsets)
        policy_seconds += time.perf_counter() - start
        key = (choice["power"], choice["offset"])
        if key not in cache:
            start = time.perf_counter()
            cache[key] = check_experiment(target_qubits, nominal, *key)
            check_seconds += time.perf_counter() - start
        probability = cache[key]["zero_probability"]
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise AssertionError("Checked sampling probability is outside [0, 1]")
        draw = float(rng.random())
        outcome = 0 if draw < probability else 1
        start = time.perf_counter()
        prior = update_posterior(prior, points, *key, outcome)
        update_seconds += time.perf_counter() - start
        trace.append({"step": index, **choice, "outcome": outcome, "ideal_zero_probability": probability,
                      "posterior_weights": prior.tolist()})
    summary_start = time.perf_counter()
    summary = posterior_summary(prior, points)
    summary_seconds = time.perf_counter() - summary_start
    estimate = summary["mean_phase"] if summary["mean_phase"] is not None else summary["map_phase"]
    error = abs((estimate - true_phase + .5) % 1 - .5)
    return {"status": "checked_ideal_single_phase_fixed_length_grid_bayes", "nominal_phase": nominal,
            "actual_angle_phase_reference": true_phase, "target_qubits": target_qubits, "seed": seed,
            "grid": points.tolist(), "final_posterior": prior.tolist(), "trace": trace,
            "summary": summary, "estimate": estimate, "circular_error": error,
            "ledger": transcript_ledger(target_qubits, [r["power"] for r in trace], grid_size=grid_size,
                                        candidates=len(powers) * len(offsets), preflight_policy_evaluations=1),
            "checked_experiments": [{"power": k, "offset": beta, **check} for (k, beta), check in cache.items()],
            "classical_timing_seconds": {"policy_selections": policy_seconds, "preflight_policy": preflight_seconds,
                                         "policy_total": policy_seconds + preflight_seconds, "posterior_updates": update_seconds,
                                         "final_posterior_summary": summary_seconds,
                                         "circuit_and_independent_checks": check_seconds},
            "empirical_scope": "one prescribed seeded ideal-model transcript; errors retained without coverage or convergence claim",
            "physical_execution": False, "uniform_confidence_certificate": False,
            "mixed_spectral_input_supported": False}
