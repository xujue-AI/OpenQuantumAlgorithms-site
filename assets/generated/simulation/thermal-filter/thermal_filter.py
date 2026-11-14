# SPDX-License-Identifier: MIT
"""Bounded noncommuting thermal filtering with explicit Pauli LCU access.

An imaginary-time Taylor polynomial is built by Pauli multiplication, then
loaded into actual PREPARE/SELECT/unprepare circuits. No matrix exponential
or eigenvector produces a quantum gate. Dense matrices are independent tiny
references. Classical Pauli aggregation can be exponential; this is not a
scalable Gibbs oracle or an amplified Gibbs-preparation algorithm.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, partial_trace
from scipy.linalg import expm

if __package__:
    from .lcu import PauliLCU, build_pauli_lcu, postselection_attempts
else:
    from lcu import PauliLCU, build_pauli_lcu, postselection_attempts

DEFAULT_TERMS = ((0.6, "Z"), (0.4, "X"))
ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_SYSTEM_QUBITS = 4
MAX_INPUT_TERMS = 32
MAX_AGGREGATED_TERMS = 256
MAX_TAYLOR_DEGREE = 12
MAX_STATEVECTOR_QUBITS = 12
MAX_SCALED_BETA_ALPHA = 10.0


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _model(terms: Sequence[tuple[float, str]], beta: float) -> tuple[tuple[tuple[float, str], ...], float, int]:
    beta = _real(beta, "beta")
    if beta < 0:
        raise ValueError("beta must be nonnegative")
    values = []
    n = None
    for coefficient, label in terms:
        coefficient = complex(coefficient)
        if not math.isfinite(coefficient.real) or coefficient.imag != 0:
            raise ValueError("Hamiltonian Pauli coefficients must be finite and real")
        if not isinstance(label, str) or not label or set(label)-set("IXYZ"):
            raise ValueError("Pauli labels must contain only I/X/Y/Z")
        if len(label) > MAX_SYSTEM_QUBITS:
            raise ValueError(f"explicit construction is bounded to {MAX_SYSTEM_QUBITS} system qubits")
        if n is None:
            n = len(label)
        if len(label) != n:
            raise ValueError("all Pauli labels must have the same system width")
        values.append((float(coefficient.real), label))
        if len(values) > MAX_INPUT_TERMS:
            raise ValueError(f"need 1 to {MAX_INPUT_TERMS} explicit Hamiltonian terms")
    if not values or len(values) > MAX_INPUT_TERMS:
        raise ValueError(f"need 1 to {MAX_INPUT_TERMS} explicit Hamiltonian terms")
    if n > MAX_SYSTEM_QUBITS:
        raise ValueError(f"explicit construction is bounded to {MAX_SYSTEM_QUBITS} system qubits")
    alpha = sum(abs(c) for c, _ in values)
    scaled = (beta/2)*alpha
    if not math.isfinite(alpha) or not math.isfinite(scaled) or scaled > MAX_SCALED_BETA_ALPHA:
        raise ValueError("beta*sum(abs(coefficients))/2 exceeds the declared finite domain of 10")
    return tuple(values), beta, n


_PRODUCTS = {("X", "Y"): (1j, "Z"), ("Y", "X"): (-1j, "Z"),
             ("Y", "Z"): (1j, "X"), ("Z", "Y"): (-1j, "X"),
             ("Z", "X"): (1j, "Y"), ("X", "Z"): (-1j, "Y")}


def _multiply(a: str, b: str) -> tuple[complex, str]:
    phase, label = 1+0j, []
    for x, y in zip(a, b, strict=True):
        if x == "I":
            label.append(y)
        elif y == "I":
            label.append(x)
        elif x == y:
            label.append("I")
        else:
            local, product = _PRODUCTS[(x, y)]
            phase *= local
            label.append(product)
    return phase, "".join(label)


def thermal_pauli_terms(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> tuple[tuple[complex, str], ...]:
    """Aggregate sum_k (-beta H/2)^k/k! algebraically, including all phases.

    This preprocessing can grow exponentially. Each Hermitian power uses
    (H P + P H)/2: P is a polynomial in H, so this is H P, while anticommute
    Pauli pairs cancel algebraically before floating arithmetic. No tolerance
    pruning is used. All remaining coefficient signs enter indexed SELECT.
    """
    terms, beta, n = _model(terms, beta)
    if isinstance(degree, bool) or not isinstance(degree, int) or not 0 <= degree <= MAX_TAYLOR_DEGREE:
        raise ValueError(f"degree must be an integer from 0 to {MAX_TAYLOR_DEGREE}")
    power = {"I"*n: 1+0j}
    polynomial = dict(power)
    for k in range(1, degree+1):
        next_power = {}
        for p, value in power.items():
            for h, q in terms:
                phase, product = _multiply(q, p)
                reverse_phase, reverse_product = _multiply(p, q)
                if product != reverse_product:
                    raise AssertionError("Pauli products disagree up to phase")
                symmetric_phase = (phase+reverse_phase)/2
                if symmetric_phase:
                    next_power[product] = next_power.get(product, 0j) + value*h*symmetric_phase*(-beta/(2*k))
        power = {p: c for p, c in next_power.items() if c != 0}
        for p, c in power.items():
            polynomial[p] = polynomial.get(p, 0j) + c
        if len(power) > MAX_AGGREGATED_TERMS or len(polynomial) > MAX_AGGREGATED_TERMS:
            raise ValueError("Pauli aggregation exceeds the 256-term bound")
        if any(not math.isfinite(c.real) or not math.isfinite(c.imag) for c in polynomial.values()):
            raise ValueError("imaginary-time coefficient aggregation overflowed")
    result = tuple((c, p) for p, c in sorted(polynomial.items()) if c != 0)
    if not result:
        raise ValueError("Taylor filter is zero and has no accepted branch")
    return result


def build_thermal_lcu(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> PauliLCU:
    return build_pauli_lcu(thermal_pauli_terms(terms, beta, degree))


def build_thermal_filter(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> QuantumCircuit:
    """Prepare Bell pairs and apply actual LCU to the system, labels initially 0.

    Layout: system first, equally sized reference next, label register last.
    Projecting all label wires onto zero leaves a conditional purification.
    The reference must be discarded/reset without conditioning for mixed Gibbs
    output; the measured-trial wrapper performs that reset explicitly.
    """
    model, beta, n = _model(terms, beta)
    lcu = build_thermal_lcu(model, beta, degree)
    circuit = QuantumCircuit(2*n+lcu.n_index, name="Imaginary-time thermal filter")
    for q in range(n):
        circuit.h(q)
        circuit.cx(q, n+q)
    labels = list(range(2*n, circuit.num_qubits))
    circuit.compose(lcu.circuit, qubits=list(range(n))+labels, inplace=True)
    circuit.metadata = {
        "n_system": n, "n_reference": n, "n_index": lcu.n_index,
        "beta": beta, "degree": degree, "normalization": lcu.normalization,
        "Hamiltonian_terms": model, "aggregated_terms": len(lcu.terms),
        "postselection": "all label measurement bits zero",
        "prepared_output": "conditional system-reference purification before final system readout",
        "classical_preprocessing": "bounded Pauli multiplication/aggregation; potentially exponential",
        "amplitude_amplification": "not implemented", "rotations_synthesized": False,
    }
    return circuit


def build_thermal_trial(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6,
                        measurement: str = "Z") -> QuantumCircuit:
    """Full reset/access/label measurement/reference reset/system-readout trial.

    Accepted outcomes have all labels zero. Reference reset is unconditioned;
    every branch runs the full schedule and failed measured trials are discarded.
    X or Y basis rotations are actual gates; Z needs no final basis change.
    """
    core = build_thermal_filter(terms, beta, degree)
    n, m = core.metadata["n_system"], core.metadata["n_index"]
    if measurement not in {"X", "Y", "Z"}:
        raise ValueError("measurement must be X, Y or Z on every system wire")
    circuit = QuantumCircuit(core.num_qubits, m+n, name=f"Thermal {measurement} measured trial")
    circuit.reset(range(core.num_qubits))
    circuit.compose(core, inplace=True)
    if m:
        circuit.measure(list(range(2*n, core.num_qubits)), list(range(m)))
    circuit.reset(list(range(n, 2*n)))
    if measurement == "Y":
        circuit.sdg(range(n))
    if measurement != "Z":
        circuit.h(range(n))
    circuit.measure(list(range(n)), list(range(m, m+n)))
    circuit.metadata = {**core.metadata, "measurement_basis": measurement,
                        "output": "accepted measured thermal trial, reference reset without conditioning",
                        "initial_reset_instructions": core.num_qubits, "reference_reset_instructions": n,
                        "label_measurements": m, "system_measurements": n,
                        "schedule": "all branches run fully; only zero-label trials retained"}
    return circuit


def decompose_thermal_filter(circuit: QuantumCircuit) -> QuantumCircuit:
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS, optimization_level=1, seed_transpiler=7)
    unknown = set(compiled.count_ops()) - set(ROTATION_BASIS) - {"reset", "measure", "barrier"}
    if unknown:
        raise ValueError(f"unresolved operations: {unknown}")
    return compiled


def _pauli_matrix(label: str) -> np.ndarray:
    local = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
             "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    result = np.array([[1+0j]])
    for p in label:
        result = np.kron(result, local[p])
    return result


def dense_hamiltonian(terms=DEFAULT_TERMS, beta: float = 1.25) -> np.ndarray:
    """Independent bounded Kronecker reference; no matrix defines a circuit."""
    terms, _, n = _model(terms, beta)
    result = np.zeros((2**n, 2**n), dtype=complex)
    for coefficient, label in terms:
        result += coefficient*_pauli_matrix(label)
    return result


def matrix_thermal_polynomial(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> np.ndarray:
    """Independent matrix recurrence for checking the Pauli-aggregate circuit."""
    thermal_pauli_terms(terms, beta, degree)  # validates degree/domain, not used for coefficients
    H = dense_hamiltonian(terms, beta)
    result = np.eye(len(H), dtype=complex)
    power = result.copy()
    for k in range(1, degree+1):
        power = (-beta/(2*k))*(H @ power)
        result += power
    return result


def exact_gibbs_state(terms=DEFAULT_TERMS, beta: float = 1.25) -> np.ndarray:
    H = dense_hamiltonian(terms, beta)
    exponential = expm(-beta*H)
    return exponential/np.trace(exponential)


def actual_filter_block(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> np.ndarray:
    lcu = build_thermal_lcu(terms, beta, degree)
    if 2*lcu.n_system+lcu.n_index > MAX_STATEVECTOR_QUBITS:
        raise ValueError("thermal circuit validation exceeds 12 total qubits")
    d = 2**lcu.n_system
    columns = []
    for x in range(d):
        initial = np.zeros(2**lcu.circuit.num_qubits, dtype=complex)
        initial[x] = 1
        columns.append(Statevector(initial).evolve(lcu.circuit).data[:d])
    return np.column_stack(columns)


def postselected_thermal_purification(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> tuple[np.ndarray, float]:
    core = build_thermal_filter(terms, beta, degree)
    if core.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError("thermal circuit validation exceeds 12 total qubits")
    n = core.metadata["n_system"]
    branch = Statevector.from_instruction(core).data[:2**(2*n)]
    probability = float(np.vdot(branch, branch).real)
    if probability <= 1e-30 or probability > 1+1e-12:
        raise ValueError("thermal filter has zero or invalid numerical acceptance")
    return branch/math.sqrt(probability), min(probability, 1.0)


def _observables(state: np.ndarray, terms) -> dict[str, float]:
    n = len(terms[0][1])
    values = {"energy": float(np.trace(state @ dense_hamiltonian(terms, 0)).real)}
    for axis in ("X", "Y", "Z"):
        values[axis+"0"] = float(np.trace(state @ _pauli_matrix("I"*(n-1)+axis)).real)
    return values


def thermal_filter_check(terms=DEFAULT_TERMS, beta: float = 1.25, degree: int = 6) -> dict:
    terms, beta, n = _model(terms, beta)
    lcu = build_thermal_lcu(terms, beta, degree)
    polynomial = matrix_thermal_polynomial(terms, beta, degree)
    actual_block = actual_filter_block(terms, beta, degree)
    purified, probability = postselected_thermal_purification(terms, beta, degree)
    expected_probability = float(np.trace(polynomial.conj().T @ polynomial).real/(len(polynomial)*lcu.normalization**2))
    branch_reference = polynomial.reshape(-1, order="F") / math.sqrt(len(polynomial))
    branch_reference /= np.linalg.norm(branch_reference)
    actual_state = partial_trace(Statevector(purified), list(range(n, 2*n))).data
    polynomial_state = polynomial @ polynomial.conj().T
    polynomial_state /= np.trace(polynomial_state)
    exact_state = exact_gibbs_state(terms, beta)
    difference = actual_state-exact_state
    exact_filter = expm(-beta*dense_hamiltonian(terms, beta)/2)
    return {
        "beta": beta, "degree": degree, "system_qubits": n, "logical_qubits": 2*n+lcu.n_index,
        "normalization": lcu.normalization, "aggregated_terms": len(lcu.terms),
        "actual_access_block_spectral_error": float(np.linalg.norm(actual_block-polynomial/lcu.normalization, 2)),
        "polynomial_filter_operator_error": float(np.linalg.norm(polynomial-exact_filter, 2)),
        "conditional_purification_vector_error_vs_polynomial": float(np.linalg.norm(purified-branch_reference)),
        "actual_reduced_state_error_vs_polynomial": float(np.max(np.abs(actual_state-polynomial_state))),
        "gibbs_trace_distance": float(np.linalg.svd(difference, compute_uv=False).sum()/2),
        "gibbs_max_entry_error": float(np.max(np.abs(difference))),
        "trace_real": float(np.trace(actual_state).real), "minimum_eigenvalue": float(np.linalg.eigvalsh(actual_state).min()),
        "acceptance_probability": probability, "independent_acceptance_probability": expected_probability,
        "complete_trials_for_one_measured_output_99_percent": postselection_attempts(probability, .01),
        "actual_observables": _observables(actual_state, terms), "exact_observables": _observables(exact_state, terms),
        "state_stage": "conditional state before final system readout; references traced out without conditioning",
        "actual_reduced_state_real": actual_state.real.tolist(), "actual_reduced_state_imag": actual_state.imag.tolist(),
    }


def conditioned_synthesis_bounds(acceptance_probability: float, epsilon: float = 1e-4) -> dict:
    """Hypothetical bound from certified coherent Bell+LCU unitary norm error.

    This does not assert synthesis exists. A QDK mean-fit scalar is not a
    certified operator-norm error. Conditioning can amplify coherent error.
    """
    p, epsilon = _real(acceptance_probability, "acceptance_probability"), _real(epsilon, "epsilon")
    if not 0 < p <= 1 or not 0 <= epsilon < math.sqrt(p):
        raise ValueError("need 0<p<=1 and 0<=epsilon<sqrt(p)")
    distance = min(1.0, 2*epsilon/math.sqrt(p))
    return {"full_circuit_operator_error_allowance": epsilon,
            "conditional_state_trace_distance_bound": distance,
            "single_pauli_mean_bias_bound": 2*distance,
            "perturbed_acceptance_lower_bound": (math.sqrt(p)-epsilon)**2,
            "status": "hypothetical only; no certified synthesis implemented",
            "assumption": "certified coherent Bell+LCU operator-norm bound, projected branch then normalized",
            "bound_scope": "coherent unitary only; irreversible initialization, reference reset and readout faults are separate"}


def _pauli_shots(error: float, failure: float) -> int:
    error, failure = _real(error, "statistical_error"), _real(failure, "failure_probability")
    if error <= 0 or not 0 < failure < 1:
        raise ValueError("statistical error must be positive and failure in (0,1)")
    log_count = math.log(2)+math.log(math.log(2)-math.log(failure))-2*math.log(error)
    if log_count >= math.log(np.finfo(float).max)-1e-12:
        raise ValueError("shot count exceeds the floating budget domain")
    return max(1, math.ceil(math.exp(log_count)))


def expected_trials_stopped_at_cap(acceptance_probability: float, cap: int) -> float:
    """E[min(T, cap)] for an iid geometric trial count T, including exhaustion.

    This is an unconditional stopped expectation for one requested slot, not
    the unlimited 1/p expectation or an expectation conditional on acceptance.
    log1p/expm1 retain the finite-cap limit when tiny p rounds 1-p to one.
    """
    p = _real(acceptance_probability, "acceptance_probability")
    if not 0 < p <= 1 or isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("need 0<p<=1 and a positive integer complete-trial cap")
    if p == 1:
        return 1.0
    return -math.expm1(cap*math.log1p(-p))/p


def thermal_energy_workload(acceptance_probability: float, terms=DEFAULT_TERMS,
                            statistical_error: float = .03, statistical_failure: float = .005,
                            acceptance_failure: float = .005, physical_failure: float = .005) -> dict:
    """Separate measured Pauli groups for a one-qubit noncommuting energy mean.

    No single Z-basis bitstring supplies a noncommuting X/Z energy. For each
    nonzero Pauli group, equal accepted-shot error statistical_error/sum|c|
    guarantees the weighted statistical error allocation by a triangle bound.
    The returned trial caps and expectations are unexecuted resource proposals.
    Every requested slot is attempted even if another slot exhausts its cap.
    """
    terms, _, n = _model(terms, 0)
    p = _real(acceptance_probability, "acceptance_probability")
    if n != 1 or not 0 < p <= 1:
        raise ValueError("workload helper covers one-qubit means with 0<p<=1")
    coefficients = {}
    identity = 0.0
    for coefficient, axis in terms:
        if axis == "I":
            identity += coefficient
        else:
            coefficients[axis] = coefficients.get(axis, 0.0)+coefficient
    coefficients = {axis: c for axis, c in coefficients.items() if c != 0}
    weight = sum(abs(c) for c in coefficients.values())
    error = _real(statistical_error, "statistical_error")
    sf, af, pf = (_real(x, name) for x, name in ((statistical_failure, "statistical_failure"),
                                                 (acceptance_failure, "acceptance_failure"), (physical_failure, "physical_failure")))
    if error <= 0 or not all(0 < x < 1 for x in (sf, af, pf)) or sf+af+pf >= 1:
        raise ValueError("need positive statistical error and separate failure allowances summing below one")
    if not coefficients:
        return {"status": "known constant energy; no sampling required", "identity_offset": identity,
                "groups": [], "total_accepted_shots": 0, "worst_case_complete_trials": 0,
                "expected_complete_trials": 0, "expected_complete_trials_uncapped": 0,
                "expected_complete_trials_stopped_at_cap": 0,
                "expected_complete_trials_scope": "legacy alias for uncapped geometric expectation M/p"}
    shots = _pauli_shots(error/weight, sf/len(coefficients))
    total = len(coefficients)*shots
    per_shot_acceptance_failure = af/total
    if per_shot_acceptance_failure == 0:
        raise ValueError("per-shot acceptance allocation underflowed")
    cap = postselection_attempts(p, per_shot_acceptance_failure)
    stopped_per_slot = expected_trials_stopped_at_cap(p, cap)
    groups = [{"axis": axis, "coefficient": c, "accepted_shots": shots,
               "pauli_statistical_error": error/weight, "statistical_failure": sf/len(coefficients),
               "complete_trial_cap_per_accepted_shot": cap,
               "worst_case_complete_trials": shots*cap, "expected_complete_trials": shots/p,
               "expected_complete_trials_uncapped": shots/p,
               "expected_complete_trials_stopped_at_cap": shots*stopped_per_slot,
               "expected_complete_trials_scope": "legacy alias for uncapped geometric expectation M/p"}
              for axis, c in sorted(coefficients.items())]
    return {"status": "unexecuted grouped noncommuting energy-mean workload proposal",
            "identity_offset": identity, "groups": groups, "total_accepted_shots": total,
            "statistical_energy_error_allowance": error, "statistical_failure_probability": sf,
            "acceptance_failure_probability": af, "physical_failure_probability": pf,
            "worst_case_complete_trials": total*cap, "expected_complete_trials": total/p,
            "expected_complete_trials_uncapped": total/p,
            "expected_complete_trials_stopped_at_cap": total*stopped_per_slot,
            "expected_complete_trials_scope": "legacy alias for uncapped geometric expectation M/p",
            "stopped_expectation_scope": "M*(1-(1-p)^R)/p; every requested slot stops on acceptance or cap R, including exhausted slots; all M slots are attempted, without early abort or conditioning on workload success",
            "physical_failure_per_complete_trial_proposal": pf/(total*cap),
            "sampling_performed": False,
            "scope": "separate Pauli-basis accepted iid samples; every complete retry re-prepares Bell input/access and runs final readout"}
