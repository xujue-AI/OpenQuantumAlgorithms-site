"""Bounded random-base factoring with actual recycled-control order circuits.

Quantum trials evolve the defined modular-arithmetic gates and explicitly
project each ancilla measurement. Classical order enumeration is not used to
construct or sample a trial. General success certificates are not inferred from
the chosen precision or from a seeded successful demonstration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from openquantum.shor import (
    MAX_INTEGER_BITS, MAX_PHASE_BITS, MAX_PRIMITIVE_GATES, MAX_SIM_QUBITS,
    ROTATION_BASIS, controlled_multiplier_gate, factors_from_order,
    reconstruct_order, validate_instance,
)

MAX_DRIVER_INPUT = (1 << MAX_INTEGER_BITS) - 1
MAX_VALIDATION_PHASE_BITS = 6
MAX_VALIDATION_AMPLITUDES = 1 << 22


def _input(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 2 <= value <= MAX_DRIVER_INPUT:
        raise ValueError(f"modulus must be an integer in [2,{MAX_DRIVER_INPUT}]")
    return value


def integer_nth_root(value: int, degree: int) -> int:
    """Exact floor root using integer binary search, never floating roots."""
    _input(value)
    if isinstance(degree, bool) or not isinstance(degree, int) or not 2 <= degree <= MAX_INTEGER_BITS:
        raise ValueError("degree must be an integer from2 to the bounded bit width")
    low, high = 1, 1 << math.ceil(value.bit_length() / degree)
    while low < high:
        middle = (low + high + 1) // 2
        if pow(middle, degree) <= value:
            low = middle
        else:
            high = middle - 1
    return low


def perfect_power(value: int) -> tuple[int, int] | None:
    _input(value)
    for degree in range(value.bit_length(), 1, -1):
        root = integer_nth_root(value, degree)
        if root > 1 and pow(root, degree) == value:
            return root, degree
    return None


def is_prime_bounded(value: int) -> bool:
    """Deterministic two-base Miller-Rabin in this explicitly bounded domain.

    This API's maximum4095 is checked exhaustively against an independent
    sieve in the dedicated test suite; no unbounded primality claim is made.
    """
    _input(value)
    if value in (2, 3):
        return True
    if value % 2 == 0:
        return False
    odd, power = value - 1, 0
    while odd % 2 == 0:
        odd //= 2
        power += 1
    for base in (2, 3):
        if base >= value:
            continue
        residue = pow(base, odd, value)
        if residue in (1, value - 1):
            continue
        for _ in range(power - 1):
            residue = residue * residue % value
            if residue == value - 1:
                break
        else:
            return False
    return True


def preliminary_checks(modulus: int) -> dict[str, Any]:
    _input(modulus)
    if modulus in (2, 3):
        return {"decision": "prime", "checks": ["small_prime"], "factors": []}
    if modulus % 2 == 0:
        return {"decision": "classical_factor", "checks": ["even"], "factors": [2, modulus // 2]}
    power = perfect_power(modulus)
    if power:
        root, degree = power
        return {"decision": "classical_factor", "checks": ["even", "exact_perfect_power"],
                "perfect_power": {"base": root, "exponent": degree}, "factors": sorted([root, modulus // root])}
    if is_prime_bounded(modulus):
        return {"decision": "prime", "checks": ["even", "exact_perfect_power", "bounded_miller_rabin_2_3"], "factors": []}
    return {"decision": "quantum_needed", "checks": ["even", "exact_perfect_power", "bounded_miller_rabin_2_3"], "factors": []}


def recommended_phase_bits(modulus: int) -> int:
    """2n Fourier precision; a resolution rule, not a success certificate."""
    _input(modulus)
    return 2 * (modulus - 1).bit_length()


def _quantum_contract(modulus: int, base: int, phase_bits: int | None) -> tuple[int, int, int]:
    n = validate_instance(modulus, base)
    t = recommended_phase_bits(modulus) if phase_bits is None else phase_bits
    if isinstance(t, bool) or not isinstance(t, int) or not 1 <= t <= MAX_PHASE_BITS:
        raise ValueError("phase_bits outside bounded precision range")
    if 100 * n * t * (n + 1) ** 2 > MAX_PRIMITIVE_GATES:
        raise ValueError("Recycled-control arithmetic-work preflight exceeded")
    return n, t, 2 * n + 3


def build_iterative_order_circuit(modulus: int, base: int, phase_bits: int | None = None, *, recorded_bits: tuple[int, ...] | None = None) -> QuantumCircuit:
    """Actual measurement/feedback/reset circuit or one bound observed path.

    Wire0 is recycled control, then n input bits, n+1 accumulator and flag.
    All wires reset at the beginning of each complete trial; the control resets
    t-1 additional times. Output bits arrive least-significant first.
    """
    n, t, width = _quantum_contract(modulus, base, phase_bits)
    if recorded_bits is not None and (len(recorded_bits) != t or any(bit not in (0, 1) for bit in recorded_bits)):
        raise ValueError("recorded_bits must specify exactly one binary trajectory")
    circuit = QuantumCircuit(width, t, name=f"iterative_order_{base}_{modulus}")
    circuit.reset(range(width))
    circuit.x(1)
    for output_bit in range(t):
        if output_bit:
            circuit.reset(0)
        circuit.h(0)
        multiplier = pow(base, 1 << (t - 1 - output_bit), modulus)
        if multiplier != 1:
            circuit.append(controlled_multiplier_gate(modulus, multiplier), range(width))
        for previous in range(output_bit):
            angle = -math.pi / 2 ** (output_bit - previous)
            if recorded_bits is None:
                with circuit.if_test((circuit.clbits[previous], 1)):
                    circuit.p(angle, 0)
            elif recorded_bits[previous]:
                circuit.p(angle, 0)
        circuit.h(0)
        circuit.measure(0, output_bit)
    return circuit


def _evolve_step(state: Statevector, modulus: int, base: int, t: int, bits: tuple[int, ...]) -> Statevector:
    step = QuantumCircuit(state.num_qubits)
    if bits:
        # Previous measurement's selected value is known in this trajectory.
        if bits[-1]:
            step.x(0)
    step.h(0)
    multiplier = pow(base, 1 << (t - 1 - len(bits)), modulus)
    if multiplier != 1:
        step.append(controlled_multiplier_gate(modulus, multiplier), range(state.num_qubits))
    for previous, bit in enumerate(bits):
        if bit:
            step.p(-math.pi / 2 ** (len(bits) - previous), 0)
    step.h(0)
    return state.evolve(step)


def _measure_probabilities(state: Statevector) -> tuple[float, float, float]:
    raw0 = float(np.vdot(state.data[::2], state.data[::2]).real)
    raw1 = float(np.vdot(state.data[1::2], state.data[1::2]).real)
    norm = raw0 + raw1
    if abs(norm - 1) > 1e-10:
        raise AssertionError("Unitary floating-point trajectory lost normalization beyond tolerance")
    return raw0 / norm, raw1 / norm, norm


def _collapse(state: Statevector, bit: int) -> Statevector:
    raw_probability = float(np.vdot(state.data[bit::2], state.data[bit::2]).real)
    if raw_probability <= 0:
        raise AssertionError("Cannot select a zero-probability measurement branch")
    output = np.zeros_like(state.data)
    output[bit::2] = state.data[bit::2] / math.sqrt(raw_probability)
    return Statevector(output)


def sample_iterative_order(modulus: int, base: int, phase_bits: int | None = None, *, rng: np.random.Generator) -> dict[str, Any]:
    n, t, width = _quantum_contract(modulus, base, phase_bits)
    if width > MAX_SIM_QUBITS:
        raise ValueError(f"Recycled-control statevector width exceeds {MAX_SIM_QUBITS}")
    state = Statevector.from_int(2, 1 << width)  # all-wire reset followed by X on input bit0
    bits: tuple[int, ...] = ()
    records = []
    joint = 1.0
    for output_bit in range(t):
        state = _evolve_step(state, modulus, base, t, bits)
        p0, p1, raw_norm = _measure_probabilities(state)
        bit = int(rng.random() >= p0)
        selected_probability = p1 if bit else p0
        clean = float(state.probabilities(list(range(n + 1, width)))[0]) / raw_norm
        if clean < 1 - 1e-10:
            raise AssertionError("Arithmetic accumulator or flag did not uncompute")
        records.append({"output_bit": output_bit, "power_bit": t - 1 - output_bit,
                        "multiplier": pow(base, 1 << (t - 1 - output_bit), modulus),
                        "previous_bits": list(bits), "feedback_angles": [-math.pi / 2 ** (output_bit - j) for j, old in enumerate(bits) if old],
                        "measured_bit": bit, "conditional_probability_zero": p0,
                        "selected_conditional_probability": selected_probability,
                        "raw_norm_before_probability_normalization": raw_norm,
                        "work_register_zero_probability": clean})
        joint *= selected_probability
        state = _collapse(state, bit)
        bits += (bit,)
    outcome = sum(bit << index for index, bit in enumerate(bits))
    return {"modulus": modulus, "base": base, "phase_bits": t, "total_qubits": width,
            "bits_lsb_first": list(bits), "outcome": outcome, "selected_joint_probability": joint,
            "steps": records, "quantum_execution": "ideal statevector evolution of actual arithmetic; explicit sampled projection and recycling",
            "floating_point_norm_tolerance": 1e-10, "order_reference_used": False}


def iterative_distribution(modulus: int, base: int, phase_bits: int, *, max_phase_bits: int = 6) -> np.ndarray:
    """Independent bounded branch enumeration of the actual adaptive circuit.

    Positive branches are retained without probability truncation. Exactly-zero
    branches have zero output mass. This verifier is not the driver's sampler.
    """
    _, t, width = _quantum_contract(modulus, base, phase_bits)
    if isinstance(max_phase_bits, bool) or not isinstance(max_phase_bits, int) or not 1 <= max_phase_bits <= MAX_VALIDATION_PHASE_BITS:
        raise ValueError("Validation precision cannot exceed immutable six-bit cap")
    if t > max_phase_bits or width > MAX_SIM_QUBITS:
        raise ValueError("Adaptive branch enumeration exceeds explicit validation limit")
    if (1 << (width + t)) > MAX_VALIDATION_AMPLITUDES:
        raise ValueError("Adaptive branch-amplitude allocation exceeds immutable memory cap")
    branches = {(): (1.0, Statevector.from_int(2, 1 << width))}
    for _ in range(t):
        following = {}
        for bits, (mass, state) in branches.items():
            evolved = _evolve_step(state, modulus, base, t, bits)
            p0, p1, _ = _measure_probabilities(evolved)
            for bit, probability in enumerate((p0, p1)):
                if probability > 0:
                    following[bits + (bit,)] = (mass * probability, _collapse(evolved, bit))
        branches = following
    output = np.zeros(1 << t)
    for bits, (mass, _) in branches.items():
        output[sum(bit << index for index, bit in enumerate(bits))] = mass
    return output


def recorded_path_ledger(modulus: int, base: int, bits: tuple[int, ...]) -> dict[str, Any]:
    circuit = build_iterative_order_circuit(modulus, base, len(bits), recorded_bits=bits)
    compiled = transpile(circuit, basis_gates=list(ROTATION_BASIS), optimization_level=1, seed_transpiler=7)
    counts = dict(compiled.count_ops())
    if set(counts) - {*ROTATION_BASIS, "measure", "reset"}:
        raise AssertionError("Recorded path left opaque operations")
    if sum(counts.values()) > MAX_PRIMITIVE_GATES:
        raise ValueError("Recorded adaptive path exceeds primitive export bound")
    return {"total_logical_qubits": compiled.num_qubits, "primitive_gate_counts": counts,
            "primitive_operations": sum(counts.values()), "dependency_depth": compiled.depth(),
            "measurements": len(bits), "initial_all_wire_resets": circuit.num_qubits,
            "recycling_resets": len(bits) - 1, "executed_feedback_phases": sum(bit * (len(bits) - index - 1) for index, bit in enumerate(bits)),
            "conditional_checks_in_dynamic_program": len(bits) * (len(bits) - 1) // 2,
            "scope": "entire observed ideal trajectory including resets/readouts; no hardware timing or physical projection",
            "compiled_global_phase_radians": float(compiled.global_phase)}


@dataclass(frozen=True)
class FactoringConfig:
    seed: int = 20261112
    max_base_draws: int = 8
    shots_per_coprime_base: int = 2
    max_quantum_trials: int = 16
    phase_bits: int | None = None

    def __post_init__(self):
        for name, maximum in (("max_base_draws", 16), ("shots_per_coprime_base", 8), ("max_quantum_trials", 32)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from1 to{maximum}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.phase_bits is not None and (isinstance(self.phase_bits, bool) or not isinstance(self.phase_bits, int) or not 1 <= self.phase_bits <= MAX_PHASE_BITS):
            raise ValueError("phase_bits must be an integer in the bounded range")


def factor_with_quantum_trials(modulus: int, *, config: FactoringConfig = FactoringConfig(), base_sequence: tuple[int, ...] | None = None) -> dict[str, Any]:
    """Return a verified nontrivial factor split or explicit bounded failure.

    Random bases are sampled uniformly with replacement from2..N-1. Optional
    base_sequence is a labeled replay/testing mode, not random-base evidence.
    One failed quantum outcome consumes a complete trial. The returned trace
    stops when a verified factor, a resource limit, or an explicit cap is met.
    """
    _input(modulus)
    if base_sequence is not None and any(isinstance(a, bool) or not isinstance(a, int) or not 2 <= a < modulus for a in base_sequence):
        raise ValueError("replay bases must be integers from2 throughN-1")
    started = time.perf_counter()
    pre_started = time.perf_counter()
    preliminary = preliminary_checks(modulus)
    pre_seconds = time.perf_counter() - pre_started
    t = recommended_phase_bits(modulus) if config.phase_bits is None else config.phase_bits
    result = {"modulus": modulus, "config": asdict(config), "phase_bits": t,
              "precision_kind": "standard_2n_resolution" if t == recommended_phase_bits(modulus) else "explicit_nonstandard_precision",
              "base_sampling": "uniform_with_replacement_2_through_N_minus_1" if base_sequence is None else "explicit_replay_sequence",
              "preliminary_checks": preliminary, "trace": [], "base_draws": 0, "quantum_trials": 0,
              "factors": preliminary["factors"], "status": preliminary["decision"],
              "generic_success_certificate": None, "quantum_emulation_wall_seconds": 0.0,
              "classical_postprocessing_wall_seconds": 0.0, "logical_compilation_wall_seconds": 0.0,
              "classical_base_selection_and_gcd_wall_seconds": 0.0,
              "classical_precheck_wall_seconds": pre_seconds,
              "executed_quantum_primitive_counts": {}, "executed_quantum_primitive_operations": 0, "peak_logical_qubits": 0,
              "resource_estimate_status": "actual ideal trajectory logical counts; adaptive reset/feedback physical projection pending"}
    if preliminary["decision"] != "quantum_needed":
        result["wall_seconds"] = time.perf_counter() - started
        return result
    rng = np.random.default_rng(config.seed)
    result["status"] = "exhausted_base_draws"
    for draw in range(config.max_base_draws):
        base_started = time.perf_counter()
        if base_sequence is not None:
            if draw >= len(base_sequence):
                result["status"] = "exhausted_replay_sequence"
                break
            base = base_sequence[draw]
        else:
            base = int(rng.integers(2, modulus))
        result["base_draws"] += 1
        divisor = math.gcd(base, modulus)
        result["classical_base_selection_and_gcd_wall_seconds"] += time.perf_counter() - base_started
        draw_record = {"kind": "base_draw", "draw_index": draw, "base": base, "gcd": divisor, "quantum_attempts": []}
        result["trace"].append(draw_record)
        if divisor > 1:
            result["factors"] = sorted([divisor, modulus // divisor])
            result["status"] = "factored_classical_gcd"
            break
        for shot in range(config.shots_per_coprime_base):
            if result["quantum_trials"] >= config.max_quantum_trials:
                result["status"] = "exhausted_quantum_trials"
                break
            quantum_started = time.perf_counter()
            try:
                attempt = sample_iterative_order(modulus, base, t, rng=rng)
            except ValueError as error:
                result["status"] = "quantum_resource_limit"
                draw_record["resource_limit"] = str(error)
                break
            result["quantum_emulation_wall_seconds"] += time.perf_counter() - quantum_started
            result["quantum_trials"] += 1
            post_started = time.perf_counter()
            period = reconstruct_order(attempt["outcome"], t, modulus, base)
            factors = factors_from_order(modulus, base, period)
            assert all(1 < factor < modulus and modulus % factor == 0 for factor in factors)
            result["classical_postprocessing_wall_seconds"] += time.perf_counter() - post_started
            compile_started = time.perf_counter()
            attempt["logical_ledger"] = recorded_path_ledger(modulus, base, tuple(attempt["bits_lsb_first"]))
            result["logical_compilation_wall_seconds"] += time.perf_counter() - compile_started
            attempt.update({"shot_for_base": shot, "verified_period_candidate": period, "factors": factors,
                            "failure_reason": None if factors else ("no_verified_period" if period is None else "period_did_not_yield_nontrivial_factors")})
            draw_record["quantum_attempts"].append(attempt)
            if factors:
                result["factors"] = sorted([factors[0], modulus // factors[0]])
                result["status"] = "factored_quantum_trial"
                break
        if result["status"] in {"factored_quantum_trial", "quantum_resource_limit", "exhausted_quantum_trials"}:
            break
    aggregate: dict[str, int] = {}
    for draw in result["trace"]:
        for attempt in draw["quantum_attempts"]:
            for name, count in attempt["logical_ledger"]["primitive_gate_counts"].items():
                aggregate[name] = aggregate.get(name, 0) + count
    result["executed_quantum_primitive_counts"] = aggregate
    result["executed_quantum_primitive_operations"] = sum(aggregate.values())
    result["peak_logical_qubits"] = max((attempt["total_qubits"] for draw in result["trace"] for attempt in draw["quantum_attempts"]), default=0)
    result["wall_seconds"] = time.perf_counter() - started
    return result


def numerical_policy_audit(modulus: int, config: FactoringConfig) -> dict[str, Any]:
    """Audit-only ideal finite-policy calculation, including a censored trial cap.

    This calls an independently enumerated small-instance order reference. It
    is never called by the actual driver, and supplies no formal or physical
    confidence certificate. Reference memory and classical work are bounded.
    """
    _input(modulus)
    if modulus > 63:
        raise ValueError("Numerical policy audit is capped at modulus63")
    preliminary = preliminary_checks(modulus)
    common = {"modulus": modulus, "config": asdict(config), "is_certified_confidence_bound": False,
              "scope": "independent numerical ideal reference for uniform random bases; not a hardware confidence or algorithm execution",
              "assumptions": ["fresh independent reset/preparation for every quantum trial", "exact arithmetic and analytic-angle measurement/feedback",
                              "uniform independent base draws with replacement from2 throughN-1", "same continued-fraction verification and global cap policy as driver"],
              "reference_order_is_audit_only": True}
    if preliminary["decision"] == "prime":
        return {**common, "status": "prime_no_factoring_task", "ideal_success_probability": None}
    if preliminary["decision"] == "classical_factor":
        return {**common, "status": "deterministic_classical_factor", "ideal_success_probability": 1.0, "ideal_failure_probability": 0.0}
    t = recommended_phase_bits(modulus) if config.phase_bits is None else config.phase_bits
    if t > 12:
        raise ValueError("Policy audit Fourier reference precision capped at12bits")
    from openquantum.shor import success_contract
    base_records = []
    for base in range(2, modulus):
        divisor = math.gcd(base, modulus)
        if divisor > 1:
            base_records.append({"base": base, "gcd": divisor, "classical_factor_success": True, "quantum_factor_probability": None})
        else:
            probability = success_contract(modulus, base, t)["ideal_factoring_success_probability_per_trial"]
            if not 0 <= probability <= 1:
                raise AssertionError("Independent reference probability outside unit interval")
            base_records.append({"base": base, "gcd": 1, "classical_factor_success": False, "quantum_factor_probability": probability})
    @lru_cache(maxsize=None)
    def failure(bases_remaining: int, trials_remaining: int) -> float:
        if bases_remaining == 0:
            return 1.0
        total = 0.0
        for record in base_records:
            if record["classical_factor_success"]:
                continue
            probability = record["quantum_factor_probability"]
            if trials_remaining < config.shots_per_coprime_base:
                total += (1 - probability) ** trials_remaining
            else:
                total += (1 - probability) ** config.shots_per_coprime_base * failure(
                    bases_remaining - 1, trials_remaining - config.shots_per_coprime_base)
        return total / len(base_records)
    failed = failure(config.max_base_draws, config.max_quantum_trials)
    return {**common, "status": "computed_numerical_ideal_policy", "phase_bits": t,
            "ideal_failure_probability": failed, "ideal_success_probability": 1 - failed,
            "base_records": base_records, "policy_recurrence": "average coprime failure**used_shots times continuation; early quantum cap is terminal except next base's prequantum gcd",
            "global_quantum_cap_censoring_included": True}


def prime_factorization(modulus: int, *, config: FactoringConfig = FactoringConfig()) -> dict[str, Any]:
    """Recursively finish verified splits, respecting one global quantum cap.

    Each component has a separately recorded derived seed and the same base
    policy. An exhausted/unsupported component remains explicitly unresolved.
    No combined success theorem is inferred from component execution traces.
    """
    _input(modulus)
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    pending = [modulus]
    primes: list[int] = []
    trace = []
    quantum_trials = 0
    primitive_counts: dict[str, int] = {}
    status = "complete_prime_factorization"
    while pending:
        if len(trace) >= 2 * MAX_INTEGER_BITS - 1:
            status = "component_limit"
            break
        component = pending.pop()
        preliminary = preliminary_checks(component)
        remaining = config.max_quantum_trials - quantum_trials
        if remaining <= 0 and preliminary["decision"] == "quantum_needed":
            pending.append(component)
            status = "exhausted_global_quantum_trials"
            break
        child_seed = int(rng.integers(0, 1 << 32))
        child = FactoringConfig(seed=child_seed, max_base_draws=config.max_base_draws,
                               shots_per_coprime_base=config.shots_per_coprime_base,
                               max_quantum_trials=max(1, remaining), phase_bits=config.phase_bits)
        run = factor_with_quantum_trials(component, config=child)
        trace.append(run)
        quantum_trials += run["quantum_trials"]
        for name, count in run["executed_quantum_primitive_counts"].items():
            primitive_counts[name] = primitive_counts.get(name, 0) + count
        if run["status"] == "prime":
            primes.append(component)
        elif run["factors"]:
            first, second = run["factors"]
            if first * second != component or not 1 < first < component or not 1 < second < component:
                raise AssertionError("Invalid nontrivial split in recursive wrapper")
            pending.extend((first, second))
        else:
            pending.append(component)
            status = "unresolved_component"
            break
    if math.prod(primes) * math.prod(pending) != modulus:
        raise AssertionError("Prime/unresolved product invariant violated")
    if not all(is_prime_bounded(prime) for prime in primes):
        raise AssertionError("Recursive wrapper mislabeled a completed prime")
    return {"modulus": modulus, "config": asdict(config), "status": status,
            "prime_factors": sorted(primes), "unresolved_components": sorted(pending), "component_runs": trace,
            "quantum_trials": quantum_trials, "base_draws": sum(run["base_draws"] for run in trace),
            "executed_quantum_primitive_counts": primitive_counts, "executed_quantum_primitive_operations": sum(primitive_counts.values()),
            "peak_logical_qubits": max((run["peak_logical_qubits"] for run in trace), default=0),
            "component_call_limit": 2 * MAX_INTEGER_BITS - 1,
            "global_quantum_cap": config.max_quantum_trials, "generic_success_certificate": None,
            "wall_seconds": time.perf_counter() - started}
