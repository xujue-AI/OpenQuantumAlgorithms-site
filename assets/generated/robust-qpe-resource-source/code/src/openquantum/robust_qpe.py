# SPDX-License-Identifier: MIT
"""Fresh-preparation robust QPE with exact rational circular decisions.

This original conservative protocol has a bounded additive probability model,
not a physical fault-tolerance claim. Circuit construction and independent
matrix checks reuse the project's explicit product-P Hadamard access. The
2015 RPE error-event bound is not used. No true phase enters recovery.
"""
from __future__ import annotations

from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from itertools import islice
import math
from typing import Iterable

if __package__:
    from .hadamard_qpe import (
        FringeCounts, MAX_LEVELS, MAX_SHOTS, MAX_MODELED_DRAWS,
        MAX_TARGET_QUBITS, MAX_DENSE_TARGET_QUBITS, MAX_EXPERIMENT_CIRCUITS,
        MAX_CIRCUIT_OPERATIONS, MAX_SCHEDULE_OPERATIONS,
        actual_angle_phase, analytic_zero_probability, base_phase_angle,
        build_experiment, build_schedule, check_experiment, circular_distance,
        dense_experiment_reference, lower_experiment, sample_counts,
        source_operation_count,
    )
else:
    from hadamard_qpe import (
        FringeCounts, MAX_LEVELS, MAX_SHOTS, MAX_MODELED_DRAWS,
        MAX_TARGET_QUBITS, MAX_DENSE_TARGET_QUBITS, MAX_EXPERIMENT_CIRCUITS,
        MAX_CIRCUIT_OPERATIONS, MAX_SCHEDULE_OPERATIONS,
        actual_angle_phase, analytic_zero_probability, base_phase_angle,
        build_experiment, build_schedule, check_experiment, circular_distance,
        dense_experiment_reference, lower_experiment, sample_counts,
        source_operation_count,
    )

FRINGE_RADIUS = Fraction(1, 12)
MAX_ANGLE_ENCLOSURE_WIDTH = Fraction(1, 96)
ATAN_TERMS = 32
EXP_TERMS = 128


def _integer(name: str, value: int, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _fraction(name: str, value: Fraction | float) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (Fraction, int, float)):
        raise ValueError(f"{name} must be a finite bounded rational scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    result = value if isinstance(value, Fraction) else Fraction(str(value))
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 256:
        raise ValueError(f"{name} exceeds the rational-input cap")
    return result


def sampling_tolerance(additive_bound: Fraction | float) -> Fraction:
    beta = _fraction("additive_bound", additive_bound)
    if not 0 <= beta < Fraction(1, 8):
        raise ValueError("Conservative protocol requires 0 <= additive_bound < 1/8")
    return Fraction(1, 8) - beta


def exp_lower_bound(x: Fraction) -> Fraction:
    """Exact positive Taylor lower bound to exp(x); never round a ceiling."""
    if not isinstance(x, Fraction) or x < 0:
        raise ValueError("Nonnegative Fraction required")
    total = term = Fraction(1)
    for j in range(1, EXP_TERMS + 1):
        term *= x / j
        total += term
    return total


def failure_bound(levels: int, shots: int, additive_bound: Fraction | float) -> Fraction:
    """Upper bound to 4L exp(-2m(1/8-beta)^2), including inconclusive output."""
    _integer("levels", levels, 1, MAX_LEVELS)
    _integer("shots", shots, 1, MAX_SHOTS)
    t = sampling_tolerance(additive_bound)
    return min(Fraction(1), Fraction(4 * levels) / exp_lower_bound(2 * shots * t * t))


def shots_for_failure(levels: int, failure_target: Fraction | float,
                      additive_bound: Fraction | float) -> int:
    _integer("levels", levels, 1, MAX_LEVELS)
    delta = _fraction("failure_target", failure_target)
    if not 0 < delta < 1:
        raise ValueError("failure_target must be in (0,1)")
    sampling_tolerance(additive_bound)
    upper = min(MAX_SHOTS, MAX_MODELED_DRAWS // (2 * levels))
    if failure_bound(levels, upper, additive_bound) > delta:
        raise ValueError("Confidence request exceeds the finite modeled-trial cap")
    low, high = 1, upper
    while low < high:
        middle = (low + high) // 2
        if failure_bound(levels, middle, additive_bound) <= delta:
            high = middle
        else:
            low = middle + 1
    return low


def _small_atan_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    """Alternating series enclosure for |x| <= 1/2, in radians."""
    if abs(x) > Fraction(1, 2):
        raise ValueError("Small-atan range exceeded")
    if x < 0:
        low, high = _small_atan_interval(-x)
        return -high, -low
    total = Fraction(0)
    term = x
    for j in range(ATAN_TERMS):
        total += term / (2 * j + 1)
        term *= -x * x
    other = total + term / (2 * ATAN_TERMS + 1)
    return min(total, other), max(total, other)


@lru_cache(maxsize=1)
def pi_interval() -> tuple[Fraction, Fraction]:
    """Machin identity pi = 16 atan(1/5) - 4 atan(1/239)."""
    a, b = _small_atan_interval(Fraction(1, 5))
    c, d = _small_atan_interval(Fraction(1, 239))
    return 16 * a - 4 * d, 16 * b - 4 * c


def _atan_unit_interval(x: Fraction) -> tuple[Fraction, Fraction]:
    if not 0 <= x <= 1:
        raise ValueError("Unit-atan range exceeded")
    if x <= Fraction(1, 2):
        return _small_atan_interval(x)
    # atan x = pi/4 + atan((x-1)/(x+1)); transformed |x| <= 1/3.
    a, b = _small_atan_interval((x - 1) / (x + 1))
    p, q = pi_interval()
    return p / 4 + a, q / 4 + b


def angle_interval(cos_signal: Fraction, sin_signal: Fraction) -> tuple[Fraction, Fraction]:
    """Exact enclosure of atan2(s,c)/(2pi) modulo one; zero is undefined.

    Fractions come directly from bounded counts. Octants and all comparisons
    are exact; no binary64 atan2 or rounded pi enters a lift decision.
    """
    c, s = cos_signal, sin_signal
    if not isinstance(c, Fraction) or not isinstance(s, Fraction):
        raise ValueError("Exact count-derived Fractions required")
    if any(max(v.numerator.bit_length(), v.denominator.bit_length()) > 256 or abs(v) > 1 for v in (c, s)):
        raise ValueError("Signals exceed the bounded count range")
    if c == s == 0:
        raise ValueError("Zero signal has no angle")
    if s == 0:
        v = Fraction(0) if c > 0 else Fraction(1, 2)
        return v, v
    if c == 0:
        v = Fraction(1, 4) if s > 0 else Fraction(3, 4)
        return v, v
    x = min(abs(c), abs(s)) / max(abs(c), abs(s))
    a, b = _atan_unit_interval(x)
    p, q = pi_interval()
    low, high = a / (2 * q), b / (2 * p)
    if abs(c) < abs(s):
        low, high = Fraction(1, 4) - high, Fraction(1, 4) - low
    if c < 0 and s > 0:
        low, high = Fraction(1, 2) - high, Fraction(1, 2) - low
    elif c < 0 and s < 0:
        low, high = Fraction(1, 2) + low, Fraction(1, 2) + high
    elif c > 0 and s < 0:
        low, high = 1 - high, 1 - low
    if not 0 <= low <= high <= 1 or high - low > MAX_ANGLE_ENCLOSURE_WIDTH:
        raise ArithmeticError("Angle enclosure failed its conservative width guard")
    return low, high


def exact_lift_interval(previous: Fraction, power: int,
                        angle: tuple[Fraction, Fraction]) -> int | None:
    """Return a unique nearest lift for every angle in its exact enclosure."""
    low, high = angle
    dlow, dhigh = power * previous - high, power * previous - low
    lift = (dlow + dhigh) // 2  # floor of the midpoint; adjust by exact half.
    if (dlow + dhigh) / 2 - lift >= Fraction(1, 2):
        lift += 1
    return lift if lift - Fraction(1, 2) < dlow and dhigh < lift + Fraction(1, 2) else None


def circular_arc(center: Fraction, radius: Fraction) -> list[list[str]]:
    """Split a circular closed confidence arc into exact intervals in [0,1]."""
    low, high = center - radius, center + radius
    if low < 0:
        return [["0", str(high)], [str(1 + low), "1"]]
    if high > 1:
        return [["0", str(high - 1)], [str(low), "1"]]
    return [[str(low), str(high)]]


def recover_phase(counts: Iterable[FringeCounts], additive_bound: Fraction | float) -> dict:
    """Counts-only robust circular recovery; retain every record on failure."""
    beta = _fraction("additive_bound", additive_bound)
    sampling_tolerance(beta)
    try:
        records = list(islice(counts, 2 * MAX_LEVELS + 1))
    except TypeError as error:
        raise ValueError("Bounded FringeCounts iterable required") from error
    if not 2 <= len(records) <= 2 * MAX_LEVELS or len(records) % 2:
        raise ValueError("Complete bounded sine/cosine schedule required")
    if not all(isinstance(r, FringeCounts) for r in records):
        raise ValueError("FringeCounts records required")
    for r in records:
        r.validate()
    levels = len(records) // 2
    expected = {(1 << j, b) for j in range(levels) for b in ("cos", "sin")}
    if {(r.power, r.basis) for r in records} != expected or len(set((r.power, r.basis) for r in records)) != len(records):
        raise ValueError("Duplicate, missing or nonconsecutive dyadic record")
    if len({r.shots for r in records}) != 1 or sum(r.shots for r in records) > MAX_MODELED_DRAWS:
        raise ValueError("Uniform bounded shot schedule required")
    m = records[0].shots
    lookup = {(r.power, r.basis): r for r in records}
    radius = FRINGE_RADIUS / (1 << (levels - 1))
    result = {"status": "recovered", "phase_estimate": None, "phase_estimate_exact": None,
              "confidence_arc_exact": None, "phase_radius_exact": str(radius),
              "failure_upper_bound_exact": str(failure_bound(levels, m, beta)),
              "additive_probability_bound_exact": str(beta),
              "confidence_scope": "uniform mathematical bound conditional on a single ideal eigenphase, independent Bernoulli trials and a valid per-fringe additive probability bound",
              "inference_arithmetic": "exact rational elementary angle enclosures and lift decisions; no machine-checked proof",
              "physical_confidence_certified": False, "counts": [asdict(r) for r in records], "trace": []}
    previous = None
    for j in range(levels):
        k = 1 << j
        c = Fraction(2 * lookup[k, "cos"].zeros - m, m)
        s = Fraction(2 * lookup[k, "sin"].zeros - m, m)
        item = {"power": k, "cos_signal_exact": str(c), "sin_signal_exact": str(s)}
        result["trace"].append(item)
        if c == s == 0:
            result["status"] = "inconclusive_zero_signal"
            return result
        try:
            low, high = angle_interval(c, s)
        except ArithmeticError:
            result["status"] = "inconclusive_angle_enclosure"
            return result
        item["angle_interval_exact"] = [str(low), str(high)]
        lift = 0 if previous is None else exact_lift_interval(previous, k, (low, high))
        if lift is None:
            result["status"] = "inconclusive_lift_boundary"
            return result
        previous = ((low + high) / 2 + lift) / k % 1
        item.update(lift_integer=lift, phase_estimate_exact=str(previous),
                    phase_estimate=float(previous))
    result.update(phase_estimate=float(previous), phase_estimate_exact=str(previous),
                  confidence_arc_exact=circular_arc(previous, radius))
    return result


def readout_zero_probability(ideal_zero: float, zero_to_one: Fraction | float,
                             one_to_zero: Fraction | float) -> float:
    """Classical confusion channel q0=(1-r01)p0+r10(1-p0)."""
    if isinstance(ideal_zero, bool) or not isinstance(ideal_zero, (float, int)) or not math.isfinite(ideal_zero) or not 0 <= ideal_zero <= 1:
        raise ValueError("Finite Bernoulli probability required")
    a, b = _fraction("zero_to_one", zero_to_one), _fraction("one_to_zero", one_to_zero)
    if not 0 <= a <= 1 or not 0 <= b <= 1:
        raise ValueError("Readout flip probabilities must be in [0,1]")
    return (1 - float(a)) * ideal_zero + float(b) * (1 - ideal_zero)


def protocol_ledger(target_qubits: int, levels: int, shots: int,
                    additive_bound: Fraction | float) -> dict:
    _integer("target_qubits", target_qubits, 1, MAX_TARGET_QUBITS)
    _integer("levels", levels, 1, MAX_LEVELS)
    _integer("shots", shots, 1, MAX_SHOTS)
    t = sampling_tolerance(additive_bound)
    trials = 2 * levels * shots
    if trials > MAX_MODELED_DRAWS:
        raise ValueError("Modeled-trial cap exceeded")
    calls = 2 * shots * ((1 << levels) - 1)
    gates = {"p": 3 * target_qubits * calls, "cx": 2 * target_qubits * calls,
             "h": 2 * trials, "x": target_qubits * trials, "sdg": levels * shots,
             "reset": (target_qubits + 1) * trials, "measure": trials}
    return {"target_qubits": target_qubits, "logical_qubits": target_qubits + 1,
            "levels": levels, "shots_per_fringe": shots, "distinct_experiment_circuits": 2 * levels,
            "complete_fresh_preparations": trials, "readout_bits": trials,
            "maximum_controlled_U_power": 1 << (levels - 1),
            "shot_weighted_controlled_U_calls": calls, "source_gate_counts_all_trials": gates,
            "raw_continuous_phase_rotations": gates["p"],
            "sampling_tolerance_exact": str(t), "phase_radius_exact": str(FRINGE_RADIUS / (1 << (levels - 1))),
            "failure_upper_bound_exact": str(failure_bound(levels, shots, additive_bound)),
            "classical_work": {"count_accumulation": "O(2Lm) represented bit observations; binomial simulation draws 2L aggregates",
                               "inference": "O(L) fixed-degree rational angle/lift operations; arithmetic bit cost grows with count precision",
                               "storage": "2L integer count batches plus all L exact interval/lift decisions"},
            "access_scope": "literal controlled product-P calls on freshly X-prepared |1>^n; classically easy",
            "synthesis_performed": False, "physical_estimate_performed": False,
            "physical_exclusions": ["rotation synthesis sequences and budgets", "physical preparation/reset/readout errors and timings",
                                    "routing, factories, code distance, controller and communication scheduling"],
            "calibration_scope": "supplied bound, no calibration experiment is executed; external calibration work is additional",
            "formal_proof_reproduced": False}
