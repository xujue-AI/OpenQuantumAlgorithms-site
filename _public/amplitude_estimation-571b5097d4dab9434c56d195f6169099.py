"""Explicit threshold counting with phase-estimation amplitude estimation.

All gates are built from Hadamard, X, controlled-X and phase operations.
The threshold is known, so this teaching instance has a trivial classical
answer. The generator is reusable; it does not establish application advantage.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


MAX_DATA_QUBITS = 20
MAX_PHASE_BITS = 10
MAX_SOURCE_OPERATIONS = 250_000
MAX_SIM_QUBITS = 12
SINGLE_RUN_SUCCESS_LOWER_BOUND = 8 / math.pi**2


def _integer(name: str, value: int, low: int, high: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _problem(n_data: int, threshold: int) -> None:
    _integer("n_data", n_data, 1, MAX_DATA_QUBITS)
    _integer("threshold", threshold, 0, 2**n_data)


def threshold_predicate(n_data: int, threshold: int) -> QuantumCircuit:
    """Compute flag XOR [x < threshold], preserving every data bit.

    q[0] is the least significant data bit; q[n_data] is the flag. At most
    n_data disjoint prefix terms are used, with no truth-table enumeration
    or comparator work register. Each MCX is decomposed during export.
    """
    _problem(n_data, threshold)
    result = QuantumCircuit(n_data + 1, name=f"x<{threshold}")
    flag = n_data
    if threshold == 2**n_data:
        result.x(flag)
        return result
    for bit in range(n_data - 1, -1, -1):
        if not ((threshold >> bit) & 1):
            continue
        controls = list(range(bit, n_data))
        # The first differing bit is 0 whereas the threshold bit is 1;
        # more significant bits must equal the threshold prefix.
        zero_controls = [bit] + [j for j in range(bit + 1, n_data) if not ((threshold >> j) & 1)]
        for wire in zero_controls:
            result.x(wire)
        if len(controls) == 1:
            result.cx(controls[0], flag)
        else:
            result.mcx(controls, flag)
        for wire in reversed(zero_controls):
            result.x(wire)
    return result


def threshold_preparation(n_data: int, threshold: int) -> QuantumCircuit:
    """A|0> = uniform sum of |x>|[x<threshold]> with a=threshold/2**n."""
    _problem(n_data, threshold)
    result = QuantumCircuit(n_data + 1, name="A")
    result.h(range(n_data))
    result.compose(threshold_predicate(n_data, threshold), inplace=True)
    return result


def zero_reflection(n_qubits: int) -> QuantumCircuit:
    """S0=I-2|0><0|, with its exact global phase retained."""
    _integer("n_qubits", n_qubits, 1, MAX_DATA_QUBITS + 1)
    result = QuantumCircuit(n_qubits, name="S0")
    result.x(range(n_qubits))
    if n_qubits == 1:
        result.z(0)
    else:
        result.h(n_qubits - 1)
        result.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        result.h(n_qubits - 1)
    result.x(range(n_qubits))
    return result


def threshold_grover_iterate(n_data: int, threshold: int) -> QuantumCircuit:
    """Q=-A S0 A† Sgood, eigenphases ±asin(sqrt(a))/pi.

    The minus sign is essential once Q is controlled. Dropping it shifts
    the phase-estimation outcome and changes sin²(pi*y/M).
    """
    preparation = threshold_preparation(n_data, threshold)
    result = QuantumCircuit(n_data + 1, name="Q")
    result.z(n_data)  # Sgood reflects the flag=1 subspace.
    result.compose(preparation.inverse(), inplace=True)
    result.compose(zero_reflection(n_data + 1), inplace=True)
    result.compose(preparation, inplace=True)
    result.global_phase += math.pi
    return result


def build_amplitude_estimation(n_data: int, threshold: int, phase_bits: int, *, measure: bool = False) -> QuantumCircuit:
    """Prepare A, apply explicitly repeated controlled Q, inverse QFT, readout.

    Construction is bounded by repeated source instructions with each MCX
    counted atomically. Nested control/lowering can cost more before the final
    primitive cap. Simulation is a separate, stricter operation.
    """
    _problem(n_data, threshold)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    preparation = threshold_preparation(n_data, threshold)
    iterate = threshold_grover_iterate(n_data, threshold)
    calls = 2**phase_bits - 1
    repeated_source_instructions = len(preparation.data) + calls * (len(iterate.data) + 1)
    if repeated_source_instructions > MAX_SOURCE_OPERATIONS:
        raise ValueError("Repeated source instructions exceed the amplitude-estimation generation cap")
    from openquantum.qft_qpe import build_qpe

    return build_qpe(phase_bits, iterate, preparation, measure=measure, controlled_label="Q")


def decode_amplitude(outcome: int, phase_bits: int) -> float:
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    _integer("outcome", outcome, 0, 2**phase_bits - 1)
    return math.sin(math.pi * outcome / 2**phase_bits) ** 2


def analytic_amplitude_distribution(amplitude: float, phase_bits: int) -> np.ndarray:
    """Equal mixture of the two independent geometric-sum QPE laws."""
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("amplitude must be finite and in [0, 1]")
    from openquantum.qft_qpe import qpe_distribution

    phase = math.asin(math.sqrt(amplitude)) / math.pi
    return (qpe_distribution(phase, phase_bits) + qpe_distribution((-phase) % 1, phase_bits)) / 2


def simulate_amplitude_estimation(n_data: int, threshold: int, phase_bits: int) -> np.ndarray:
    """Bounded ideal statevector marginal, compared by callers to analytic law."""
    _problem(n_data, threshold)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if n_data + 1 + phase_bits > MAX_SIM_QUBITS:
        raise ValueError(f"Statevector simulation is capped at {MAX_SIM_QUBITS} total qubits")
    circuit = build_amplitude_estimation(n_data, threshold, phase_bits)
    return Statevector.from_instruction(circuit).probabilities(qargs=list(range(phase_bits)))


def uniform_amplitude_error_bound(phase_bits: int) -> float:
    """BHMT Theorem 12 bound with sqrt(a(1-a)) <= 1/2, clipped to one."""
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    size = 2**phase_bits
    return min(1.0, math.pi / size + math.pi**2 / size**2)


def median_failure_bound(repetitions: int) -> float:
    """Binomial majority tail assuming independent complete ideal runs."""
    _integer("repetitions", repetitions, 1, 501)
    if repetitions % 2 == 0:
        raise ValueError("Median amplification requires an odd number of repetitions")
    p = SINGLE_RUN_SUCCESS_LOWER_BOUND
    return sum(math.comb(repetitions, failures) * (1 - p)**failures * p**(repetitions - failures)
               for failures in range((repetitions + 1) // 2, repetitions + 1))


def repetitions_for_failure(failure_probability: float) -> int:
    if not math.isfinite(failure_probability) or not 0 < failure_probability < 1:
        raise ValueError("failure_probability must be in (0, 1)")
    for count in range(1, 502, 2):
        if median_failure_bound(count) <= failure_probability:
            return count
    raise ValueError("Requested confidence exceeds the supported 501-run median cap")


def exact_median_interval_failure(amplitude: float, phase_bits: int, repetitions: int,
                                  distribution: np.ndarray) -> float:
    """Exact ideal median failure for this supplied discrete output law.

    Failure below and above the symmetric BHMT uniform interval are disjoint.
    This computes their binomial majority tails, without using a normal fit.
    It does not certify a hardware law or reuse empirical samples as proof.
    """
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    _integer("repetitions", repetitions, 1, 501)
    if repetitions % 2 == 0:
        raise ValueError("An odd number of repetitions is required")
    if not math.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError("amplitude must be in [0, 1]")
    values = np.asarray(distribution, dtype=float)
    if values.shape != (2**phase_bits,) or not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1, atol=1e-10, rtol=0):
        raise ValueError("distribution must be a normalized nonnegative phase probability vector")
    values = values / values.sum()
    estimates = np.array([decode_amplitude(y, phase_bits) for y in range(2**phase_bits)])
    epsilon = uniform_amplitude_error_bound(phase_bits)
    failure = 0.0
    for probability in (float(values[estimates < amplitude - epsilon].sum()),
                        float(values[estimates > amplitude + epsilon].sum())):
        failure += sum(math.comb(repetitions, k) * probability**k * (1 - probability)**(repetitions - k)
                       for k in range((repetitions + 1) // 2, repetitions + 1))
    return failure


def amplitude_workload(n_data: int, threshold: int, phase_bits: int, *, failure_probability: float = 0.01) -> dict[str, Any]:
    """Ideal precision/repetition contract and exact expanded source call counts.

    This is an unexecuted physical-workload proposal. It must not be confused
    with the one exact-statevector validation performed by the generator.
    """
    _problem(n_data, threshold)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    repetitions = repetitions_for_failure(failure_probability)
    q_calls = 2**phase_bits - 1
    epsilon = uniform_amplitude_error_bound(phase_bits)
    classical_shots = max(1, math.ceil(math.log(2 / failure_probability) / (2 * epsilon**2)))
    return {
        "status": "ideal_workload_proposal; complete repeated hardware execution not performed",
        "input": {"data_qubits": n_data, "threshold": threshold, "exact_amplitude": threshold / 2**n_data},
        "phase_bits": phase_bits,
        "absolute_amplitude_error_bound": epsilon,
        "single_run_success_lower_bound": SINGLE_RUN_SUCCESS_LOWER_BOUND,
        "confidence_source": "BHMT Theorem 12 plus independent odd-median binomial majority tail",
        "repetitions": repetitions,
        "ideal_median_failure_bound": median_failure_bound(repetitions),
        "requested_ideal_failure_probability": failure_probability,
        "per_run": {
            "logical_qubits": n_data + 1 + phase_bits,
            "uncontrolled_preparation_A_calls": 1,
            "controlled_Q_calls": q_calls,
            "controlled_A_calls": q_calls,
            "controlled_A_inverse_calls": q_calls,
            "threshold_predicate_or_inverse_calls": 1 + 2 * q_calls,
            "controlled_good_reflections": q_calls,
            "controlled_zero_reflections": q_calls,
            "controlled_global_minus_signs": q_calls,
            "inverse_QFT_calls": 1,
            "final_phase_measurements": phase_bits,
        },
        "all_runs": {
            "controlled_Q_calls": repetitions * q_calls,
            "threshold_predicate_or_inverse_calls": repetitions * (1 + 2 * q_calls),
            "final_phase_measurements": repetitions * phase_bits,
            "classical_postprocessing": "decode sin²(pi*y/M), sort R scalar estimates, select median",
        },
        "classical_same_preparation_comparison": {
            "method": "independent Bernoulli flag samples with Hoeffding bound",
            "shots_required": classical_shots,
            "absolute_error_target": epsilon,
            "failure_probability": failure_probability,
            "preparation_A_calls": classical_shots,
            "final_flag_measurements": classical_shots,
            "timing_executed": False,
            "classical_shortcut": "threshold/2**n is known exactly; classical sampling comparison is pedagogical",
        },
        "synthesis_error_certificate": None,
        "physical_failure_probability": None,
        "exclusions": ["actual Clifford+T synthesis sequence and approximation certificate", "hardware reset/initialization faults", "physical code/factory/timing model", "classical compilation and sorting timing", "external data loading for a practical predicate"],
        "not_claimed": ["quantum advantage on a known threshold", "fault-tolerant or hardware success guarantee", "iterative confidence-interval amplitude estimation"],
    }
