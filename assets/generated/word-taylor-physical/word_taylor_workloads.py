# SPDX-License-Identifier: MIT
"""Accuracy-preserving logical workloads for explicit structured word-Taylor.

Exact Fraction bounds describe an ideal access model for the recorded binary64
coefficients and segment duration. A separate exact exposure bound connects that
model with the stated rational chain. Probability/angle construction, compiler
arithmetic, Clifford+T synthesis and physical faults are not certified here.
Actual tiny gate circuits have independent numerical checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from numbers import Real
from typing import Any

MAX_SYSTEM_QUBITS = 6
MAX_SEGMENTS = 32
MAX_DEGREE = 8
MAX_ATTEMPTS_PER_SLOT = 1_000_000
MAX_TOTAL_ATTEMPTS = 10**12
MAX_CHECK_TOTAL_QUBITS = 12
MAX_CHECK_SYSTEM_QUBITS = 3
LOG_SERIES_TERMS = 64
RETRY_LOG_LOWER_TERMS = 8
MODES = ("native", "normalization2-oaa1")
BOUND_SOURCE = {
    "method": "Simulating Hamiltonian Dynamics with a Truncated Taylor Series",
    "doi": "10.1103/PhysRevLett.114.090502",
    "arxiv": "https://arxiv.org/abs/1412.4687",
    "access": "Existing original degree/term PREPARE, signed SELECT and unprepare",
    "bound_derivation": "Original exact-rational tail, contraction/composition and normalization ledger",
    "oaa_identity": "One round has block 3F-4FF†F; padding F=T/2 gives 3T/2-TT†T/2",
}


def rational(value: Real, name: str) -> Fraction:
    """Accept bounded exact fractions or the displayed decimal of a real."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    if isinstance(value, Fraction):
        result = value
    else:
        try:
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            result = Fraction(str(value))
        except (OverflowError, TypeError, ValueError):
            raise ValueError(f"{name} must be finite") from None
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > 256:
        raise ValueError(f"{name} exceeds the 256-bit rational bound")
    return result


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def fraction_record(value: Fraction) -> dict[str, Any]:
    large = max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 12_000
    encode = hex if large else str
    return {"numerator": encode(value.numerator), "denominator": encode(value.denominator),
            "integer_encoding": "hexadecimal" if large else "decimal",
            "approximate_decimal": float(value)}


def dyadic_lower(value: Fraction, bits: int = 64) -> Fraction:
    """Downward dyadic rounding used only for a conservative, smaller retry bound."""
    if not isinstance(value, Fraction) or not 0 <= value <= 1:
        raise ValueError("probability must be an exact Fraction in [0,1]")
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 256:
        raise ValueError("dyadic precision must lie in [1,256]")
    scale = 1 << bits
    return Fraction(value.numerator*scale//value.denominator, scale)


def _small_log_upper(value: Fraction, terms: int) -> Fraction:
    z = (value-1)/(value+1)
    partial = sum((2*z**(2*j+1)/Fraction(2*j+1) for j in range(terms)), Fraction())
    tail = 2*z**(2*terms+1)/(Fraction(2*terms+1)*(1-z*z))
    return partial+tail


def log_upper(value: Fraction, terms: int = LOG_SERIES_TERMS) -> Fraction:
    """Exact rational upper bound for log(value), for 1 <= value <= 10**18."""
    value = rational(value, "log argument")
    if not 1 <= value <= 10**18:
        raise ValueError("log argument must lie in [1,10**18]")
    if isinstance(terms, bool) or not isinstance(terms, int) or not 1 <= terms <= 128:
        raise ValueError("log series length must be an integer in [1,128]")
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    power = Fraction(2)**exponent
    while power > value:
        exponent -= 1
        power /= 2
    while 2*power <= value:
        exponent += 1
        power *= 2
    reduced = value/power
    return exponent*_small_log_upper(Fraction(2), terms)+_small_log_upper(reduced, terms)


def log_lower(value: Fraction, terms: int = RETRY_LOG_LOWER_TERMS) -> Fraction:
    """Exact lower log bound for internally generated, bounded rational ratios."""
    if not isinstance(value, Fraction) or not 1 <= value <= 2**64:
        raise ValueError("log argument must be a Fraction in [1,2**64]")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 16_384:
        raise ValueError("internal log ratio exceeds its 16384-bit bound")
    if isinstance(terms, bool) or not isinstance(terms, int) or not 1 <= terms <= 32:
        raise ValueError("lower-log series length must be an integer in [1,32]")
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    power = Fraction(2)**exponent
    while power > value:
        exponent -= 1
        power /= 2
    while 2*power <= value:
        exponent += 1
        power *= 2
    def partial(argument):
        z = (argument-1)/(argument+1)
        return sum((2*z**(2*j+1)/Fraction(2*j+1) for j in range(terms)), Fraction())
    return exponent*partial(Fraction(2))+partial(value/power)


def taylor_tail_bound(x: Fraction, degree: int) -> Fraction:
    """Bound sum_{j>K} x**j/j! by a geometric majorant, without exp subtraction."""
    x = rational(x, "exposure")
    if not 0 <= x <= Fraction(1, 2):
        raise ValueError("exposure must lie in [0,1/2]")
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= MAX_DEGREE:
        raise ValueError("degree must be an integer in [0,8]")
    leading = x**(degree+1)/math.factorial(degree+1)
    return leading/(1-x/Fraction(degree+2))


def chain_terms(n: int, J: Real = Fraction(3, 5),
                h: Real = Fraction(7, 20)) -> tuple[tuple[Fraction, str], ...]:
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_SYSTEM_QUBITS:
        raise ValueError("system width must be an integer in [1,6]")
    J, h = rational(J, "J"), rational(h, "h")
    if abs(J) > 2 or abs(h) > 2:
        raise ValueError("local couplings must have magnitude at most two")
    terms = []
    for wire in range(n-1):
        letters = ["I"]*n
        letters[wire] = letters[wire+1] = "Z"
        if J:
            terms.append((J, "".join(reversed(letters))))
    for wire in range(n):
        letters = ["I"]*n
        letters[wire] = "X"
        if h:
            terms.append((h, "".join(reversed(letters))))
    if not terms:
        raise ValueError("the chosen chain must contain a nonzero term")
    return tuple(terms)


def mode_bounds(x: Fraction, degree: int, segments: int, mode: str,
                target_exposure_error: Fraction = Fraction()) -> dict[str, Fraction]:
    """Uniform ideal bounds; unsafe candidates return zero acceptance."""
    x = rational(x, "exposure")
    if mode not in MODES:
        raise ValueError("unsupported access mode")
    if isinstance(segments, bool) or not isinstance(segments, int) or not 1 <= segments <= MAX_SEGMENTS:
        raise ValueError("segments must be an integer in [1,32]")
    target_exposure_error = rational(target_exposure_error, "target exposure error")
    if target_exposure_error < 0:
        raise ValueError("target exposure error cannot be negative")
    tail = taylor_tail_bound(x, degree)
    normalization = sum((x**k/math.factorial(k) for k in range(degree+1)), Fraction())
    if normalization > 2:
        raise ValueError("normalization-two padding precondition is not met")
    if mode == "native":
        segment_error = tail
        approximation_error = (1+tail)**segments-1
        if approximation_error < 1:
            acceptance = ((1-approximation_error)/normalization**segments)**2
        else:
            acceptance = Fraction()
    else:
        # Write T=U(I+F), ||F||<=tail.  The linear term is (F-F†)/2;
        # quadratic terms have total bound 3 tail²/2, cubic tail³/2.
        segment_error = tail*(1+Fraction(3, 2)*tail+Fraction(1, 2)*tail*tail)
        # The amplified block is a projection of a unitary, hence a contraction.
        approximation_error = segments*segment_error
        acceptance = (1-segment_error)**(2*segments) if segment_error < 1 else Fraction()
    full_error = approximation_error+target_exposure_error
    bias = 4*full_error/(1-full_error) if full_error < 1 else Fraction(4)
    return {"segment_tail": tail, "native_taylor_normalization": normalization,
            "segment_comparison_operator_error": segment_error, "composition_comparison_error": approximation_error,
            "target_exposure_error": target_exposure_error, "full_comparison_operator_error": full_error,
            "acceptance_lower": acceptance, "conditional_pauli_bias": bias}


@dataclass(frozen=True)
class WordTaylorPlan:
    n_system: int
    J: Fraction
    h: Fraction
    total_time: Fraction
    total_accuracy: Fraction
    mode: str
    terms: tuple[tuple[Fraction, str], ...]
    binary_terms: tuple[tuple[Fraction, str], ...]
    binary_segment_time: Fraction
    alpha: Fraction
    exposure: Fraction
    segments: int
    degree: int
    bounds: dict[str, Fraction]
    statistical_error: Fraction
    statistical_failure: Fraction
    acceptance_failure: Fraction
    shots: int
    attempts_per_slot: int

    @property
    def full_attempts(self) -> int:
        return self.shots*self.attempts_per_slot

    def record(self) -> dict[str, Any]:
        return {
            "n": self.n_system, "mode": self.mode, "degree": self.degree,
            "segments": self.segments, "J": fraction_record(self.J),
            "h": fraction_record(self.h), "total_time": fraction_record(self.total_time),
            "total_accuracy": fraction_record(self.total_accuracy),
            "alpha_binary_coefficients": fraction_record(self.alpha),
            "segment_exposure": fraction_record(self.exposure),
            "binary_segment_time": fraction_record(self.binary_segment_time),
            "terms": [{"coefficient": fraction_record(c), "pauli": p,
                       "binary_coefficient": fraction_record(bc), "binary_hex": float(bc).hex()}
                      for (c, p), (bc, _) in zip(self.terms, self.binary_terms)],
            "uniform_ideal_bounds": {k: fraction_record(v) for k, v in self.bounds.items()},
            "operator_comparison_map": (
                "Native operator errors compare unnormalized T_K and T_K^r with U; "
                "the accepted maps are T_K/s_K and T_K^r/s_K^r. These are not error bounds "
                "for the attenuated accepted maps versus U."
                if self.mode == "native" else
                "OAA operator errors compare the actual ideal accepted B and B^r with U; "
                "B=3T_K/2-T_K T_K† T_K/2."
            ),
            "pre_amplification_source_block_normalization_per_segment": fraction_record(
                self.bounds["native_taylor_normalization"] if self.mode == "native" else Fraction(2)),
            "error_allocation": {
                "ideal_algorithm_bias": fraction_record(self.total_accuracy/5),
                "reserved_synthesis_bias_unimplemented": fraction_record(self.total_accuracy/5),
                "statistical_error": fraction_record(self.statistical_error),
                "statistical_failure": fraction_record(self.statistical_failure),
                "acceptance_failure": fraction_record(self.acceptance_failure),
                "physical_fault_allowance_unmodeled": fraction_record(Fraction(1, 200)),
                "ideal_output_error_bound_if_exact_access": fraction_record(4*self.total_accuracy/5),
                "ideal_bad_output_or_cap_exhaustion_failure_bound_if_exact_access": fraction_record(
                    self.statistical_failure+self.acceptance_failure),
                "hardware_combined_guarantee": False,
            },
            "workload": {"accepted_slots": self.shots, "full_attempt_cap_per_slot": self.attempts_per_slot,
                         "charged_complete_attempts": self.full_attempts,
                         "retry_probability_lower_rounded_down_64_bits": fraction_record(
                             dyadic_lower(self.bounds["acceptance_lower"])),
                         "uncapped_expected_attempts_upper_bound": fraction_record(
                             Fraction(self.shots)/self.bounds["acceptance_lower"]),
                         "sampling_performed": False,
                         "retry_policy": "All segments run on every attempt; fresh basis input for every retry",
                         "retry_cap_rule": "Round p_lower down to 64 dyadic bits, then ceil(log_upper(M/delta_acceptance)/log_lower(1/(1-p_lower))); p_lower=1 uses one attempt",
                         "log_upper_series_terms": LOG_SERIES_TERMS,
                         "retry_log_lower_series_terms": RETRY_LOG_LOWER_TERMS},
            "policy": "Smallest r enforcing exact binary exposure <=1/2; smallest K<=8 meeting ideal Pauli-bias allocation",
            "guarantee_scope": "Any normalized input for ideal exact access at the binary constants, with an exact rational target-exposure correction",
            "implemented_trial_input": "Fresh all-zero basis input; it is not a protocol for copying an unknown specimen",
            "not_certified": ["floating normalization/probability/angle construction", "primitive compiler arithmetic",
                              "rotation synthesis", "physical gates/reset/readout", "hardware confidence"],
            "bound_source": BOUND_SOURCE,
        }


def plan_word_taylor(n: int, total_time: Real = Fraction(1),
                     accuracy: Real = Fraction(1, 20), mode: str = "native",
                     J: Real = Fraction(3, 5), h: Real = Fraction(7, 20)) -> WordTaylorPlan:
    terms = chain_terms(n, J, h)
    time = rational(total_time, "time")
    accuracy = rational(accuracy, "accuracy")
    if abs(time) > 2 or not Fraction(1, 200) <= accuracy <= Fraction(1, 5):
        raise ValueError("time magnitude must be <=2 and accuracy must lie in [.005,.2]")
    if mode not in MODES:
        raise ValueError("unsupported access mode")
    binary_terms = tuple((Fraction.from_float(float(c)), p) for c, p in terms)
    alpha = sum((abs(c) for c, _ in binary_terms), Fraction())
    segments = 1
    while segments <= MAX_SEGMENTS:
        segment_time = Fraction.from_float(float(time/segments))
        exposure = alpha*abs(segment_time)
        if exposure <= Fraction(1, 2):
            break
        segments += 1
    if segments > MAX_SEGMENTS:
        raise ValueError("the segment policy exceeds its 32-segment cap")
    exposure_error = sum((abs(segments*segment_time*bc-time*c)
                          for (c, _), (bc, _) in zip(terms, binary_terms)), Fraction())
    for degree in range(MAX_DEGREE+1):
        bounds = mode_bounds(exposure, degree, segments, mode, exposure_error)
        if bounds["acceptance_lower"] > 0 and bounds["conditional_pauli_bias"] <= accuracy/5:
            break
    else:
        raise ValueError("no degree through eight meets the stated ideal bias allocation")
    statistical_error = 3*accuracy/5
    statistical_failure = Fraction(1, 250)
    acceptance_failure = Fraction(1, 1000)
    shots = ceil_fraction(2*log_upper(2/statistical_failure)/statistical_error**2)
    p = bounds["acceptance_lower"]
    retry_p = dyadic_lower(p)
    if retry_p == 0:
        raise ValueError("acceptance bound is below the explicit retry arithmetic resolution")
    attempts = 1 if p == 1 else ceil_fraction(
        log_upper(Fraction(shots)/acceptance_failure)/log_lower(1/(1-retry_p)))
    if attempts > MAX_ATTEMPTS_PER_SLOT or shots*attempts > MAX_TOTAL_ATTEMPTS:
        raise ValueError("the full retry/shot workload exceeds its bounded integer cap")
    return WordTaylorPlan(n, rational(J, "J"), rational(h, "h"), time, accuracy, mode,
                         terms, binary_terms, segment_time, alpha, exposure, segments, degree,
                         bounds, statistical_error, statistical_failure, acceptance_failure, shots, attempts)


def build_planned_attempt(plan: WordTaylorPlan):
    """Reuse the actual degree/term access and full measured schedule."""
    if __package__:
        from . import word_taylor as wt
    else:
        import word_taylor as wt
    access = wt.build_word_taylor([(float(c), p) for c, p in plan.binary_terms],
                                 float(plan.binary_segment_time), plan.degree)
    padded = plan.mode == "normalization2-oaa1"
    segment = wt.amplified_word_segment(access, int(padded), padding_to_two=padded)
    attempt = wt.build_word_attempt(access, plan.segments, int(padded), padding_to_two=padded)
    return access, segment, attempt


def primitive_ledger(plan: WordTaylorPlan, attempt, compiled) -> dict[str, Any]:
    counts = dict(compiled.count_ops())
    return {
        "logical_qubits": compiled.num_qubits, "classical_readout_bits": compiled.num_clbits,
        "source_instructions": attempt.size(), "source_dependency_depth": attempt.depth(),
        "primitive_gate_counts_per_attempt": counts, "primitive_operations_per_attempt": compiled.size(),
        "primitive_dependency_depth_per_attempt": compiled.depth(),
        "primitive_gate_counts_full_cap": {k: v*plan.full_attempts for k, v in counts.items()},
        "primitive_operations_full_cap": compiled.size()*plan.full_attempts,
        "serial_dependency_layers_full_cap": compiled.depth()*plan.full_attempts,
        "exact_T_Tdag_per_attempt": counts.get("t", 0)+counts.get("tdg", 0),
        "arbitrary_rotations_per_attempt": sum(counts.get(k, 0) for k in ("rx", "ry", "rz")),
        "source_access_and_schedule_metadata": attempt.metadata,
        "synthesis": {"performed": False, "T_cost_inferred_from_rotations": False},
        "physical_resource_estimate": None,
        "physical_scope": "The current small instrument model does not cover these general-width measured Taylor schedules",
        "time_scope": "Dependency layers and measured CPU generation time are not fault-tolerant runtime",
        "included": ["all input resets", "degree and term PREPARE/unprepare",
                     "actual signed SELECT", "actual forward/inverse OAA calls where selected",
                     "all ancilla reflections and zero-LCU padding where selected",
                     "every segment ancilla measurement/reset", "final system Z readout",
                     "all rejected complete attempts and capped accepted-shot workload"],
        "excluded": ["rotation synthesis", "physical fault/timing/routing/factory/code model",
                     "classical coefficient/readout wall time", "executed sampling"],
    }


def _independent_chain_matrix(plan: WordTaylorPlan, binary: bool):
    import numpy as np
    dimension = 2**plan.n_system
    result = np.zeros((dimension, dimension), complex)
    J = float(Fraction.from_float(float(plan.J)) if binary else plan.J)
    h = float(Fraction.from_float(float(plan.h)) if binary else plan.h)
    for basis in range(dimension):
        for wire in range(plan.n_system):
            result[basis ^ (1 << wire), basis] += h
        result[basis, basis] += sum(J*(1 if ((basis >> wire) & 1) == ((basis >> (wire+1)) & 1) else -1)
                                   for wire in range(plan.n_system-1))
    return result


def accepted_measured_probabilities(circuit, n_system: int):
    """Tiny pure-state all-zero acceptance branch of the actual measured schedule.

    Rejected branches are charged but not enumerated. Initial/reset wires are
    checked to be zero on the retained history. Final data readouts are deferred
    only after verifying that no later quantum operation touches that data wire.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    if circuit.num_qubits > MAX_CHECK_TOTAL_QUBITS or n_system > MAX_CHECK_SYSTEM_QUBITS:
        raise ValueError("complete measured checks are capped at 12 total/3 data wires")
    data_readouts = {}
    for index, item in enumerate(circuit.data):
        if item.operation.name == "measure":
            wire = circuit.find_bit(item.qubits[0]).index
            classical = circuit.find_bit(item.clbits[0]).index
            if wire < n_system:
                if wire in data_readouts:
                    raise ValueError("duplicate data readout")
                data_readouts[wire] = classical
                for later in circuit.data[index+1:]:
                    if later.operation.name not in {"measure", "barrier"} and item.qubits[0] in later.qubits:
                        raise ValueError("data measurement cannot be safely deferred")
    if set(data_readouts) != set(range(n_system)):
        raise ValueError("every data wire needs its final readout")
    if len(set(data_readouts.values())) != n_system:
        raise ValueError("data readouts need distinct classical targets")
    state = Statevector.from_int(0, 2**circuit.num_qubits)
    indices = np.arange(len(state.data))
    chunk = QuantumCircuit(circuit.num_qubits)
    seen_acceptance_bits = set()
    def flush():
        nonlocal state, chunk
        if chunk.data:
            state = state.evolve(chunk)
            chunk = QuantumCircuit(circuit.num_qubits)
    for item in circuit.data:
        name = item.operation.name
        if name in {"reset", "measure"}:
            flush()
            wire = circuit.find_bit(item.qubits[0]).index
            if name == "reset":
                if np.vdot(state.data[(indices >> wire) & 1 == 1],
                           state.data[(indices >> wire) & 1 == 1]).real > 1e-20:
                    raise ValueError("reset is not on a zero wire on this accepted history")
            elif wire >= n_system:
                classical = circuit.find_bit(item.clbits[0]).index
                if classical in seen_acceptance_bits or classical in data_readouts.values():
                    raise ValueError("ancillary outcomes need distinct acceptance bits")
                seen_acceptance_bits.add(classical)
                kept = state.data.copy()
                kept[(indices >> wire) & 1 == 1] = 0
                state = Statevector(kept)
        elif name != "barrier":
            chunk.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
    flush()
    probabilities = np.bincount(indices % (2**n_system), weights=np.abs(state.data)**2,
                                minlength=2**n_system)
    return probabilities, {"acceptance_bits_checked": len(seen_acceptance_bits),
                           "data_readout_map": data_readouts,
                           "rejected_branches_enumerated": False,
                           "method": "Unnormalized all-zero ancillary branch of actual primitive events"}


def check_tiny_circuit(plan: WordTaylorPlan, access, segment, compiled) -> dict[str, Any]:
    import numpy as np
    from qiskit.quantum_info import Statevector
    from scipy.linalg import expm
    if __package__:
        from . import word_taylor as wt
    else:
        import word_taylor as wt
    if segment.num_qubits > MAX_CHECK_TOTAL_QUBITS or plan.n_system > MAX_CHECK_SYSTEM_QUBITS:
        return {"performed": False, "reason": "12-total/3-data-wire independent-check cap"}
    actual = wt.projected_word_block(segment)
    h_binary = _independent_chain_matrix(plan, True)
    h_target = _independent_chain_matrix(plan, False)
    polynomial = np.eye(len(h_binary), dtype=complex)
    term = polynomial.copy()
    for k in range(1, plan.degree+1):
        term = (-1j*float(plan.binary_segment_time)/k)*(h_binary@term)
        polynomial += term
    reference_segment = polynomial/access.normalization if plan.mode == "native" else (
        1.5*polynomial-.5*polynomial@polynomial.conj().T@polynomial)
    segment_residual = float(np.linalg.norm(actual-reference_segment, 2))
    if segment_residual > 5e-10:
        raise AssertionError("actual structured access differs from independent polynomial")
    full = np.linalg.matrix_power(actual, plan.segments)
    reference_full = np.linalg.matrix_power(reference_segment, plan.segments)
    target = expm(-1j*float(plan.total_time)*h_target)
    eigenvalues, vectors = np.linalg.eigh(h_target)
    second_target = (vectors*np.exp(-1j*float(plan.total_time)*eigenvalues))@vectors.conj().T
    if np.linalg.norm(target-second_target, 2) > 5e-12:
        raise AssertionError("independent dense evolution references disagree")
    basis = np.eye(len(target), dtype=complex)[:, 0]
    branch = full@basis
    p = float(np.vdot(branch, branch).real)
    state = branch/math.sqrt(p)
    exact_state = target@basis
    signs = 1-2*(np.arange(len(state)) & 1)
    actual_mean = float(np.dot(signs, np.abs(state)**2))
    exact_mean = float(np.dot(signs, np.abs(exact_state)**2))
    minimum_acceptance = float(np.linalg.svd(full, compute_uv=False)[-1]**2)
    if minimum_acceptance+5e-10 < float(plan.bounds["acceptance_lower"]):
        raise AssertionError("tiny numerical acceptance violates the ideal-access bound beyond its guard")
    if abs(actual_mean-exact_mean) > float(plan.bounds["conditional_pauli_bias"])+5e-10:
        raise AssertionError("tiny numerical observable violates the ideal-access bound beyond its guard")
    measured, measured_scope = accepted_measured_probabilities(compiled, plan.n_system)
    measured_residual = float(np.max(np.abs(measured-np.abs(reference_full@basis)**2)))
    if measured_residual > 5e-10:
        raise AssertionError("complete measured/reset branch differs from polynomial chronology")
    rng = np.random.default_rng(20261112)
    random_input = rng.normal(size=2**segment.num_qubits)+1j*rng.normal(size=2**segment.num_qubits)
    random_input /= np.linalg.norm(random_input)
    lowered = wt.decompose_word_circuit(segment)
    lowering_residual = float(np.linalg.norm(Statevector(random_input).evolve(segment).data-
                                             Statevector(random_input).evolve(lowered).data))
    if lowering_residual > 5e-10:
        raise AssertionError("coherent primitive lowering changes the full-register state")
    return {"performed": True, "source_block_residual_vs_independent_polynomial": segment_residual,
            "complete_measured_branch_probability_residual": measured_residual,
            "coherent_lowering_random_full_register_residual": lowering_residual,
            "independent_expm_eigh_residual": float(np.linalg.norm(target-second_target, 2)),
            "actual_known_input_acceptance": p, "actual_known_input_Z_mean": actual_mean,
            "independent_target_Z_mean": exact_mean,
            "actual_known_input_Z_bias": abs(actual_mean-exact_mean),
            "actual_full_map_minimum_acceptance": minimum_acceptance,
            "numerical_probability_guard": 5e-10, "seed": 20261112,
            "uniform_hardware_certificate": False,
            "measured_check": measured_scope,
            "scope": "Tiny numerical gate/access/instrument check; analytic bounds remain an ideal-access statement"}
