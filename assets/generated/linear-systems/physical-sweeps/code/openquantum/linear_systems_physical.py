# SPDX-License-Identifier: MIT
"""Bounded conditional physical sweeps for explicit HHL and Neumann LCU.

The output is one normalized-solution Z(q0) mean at each declared instance.
Every attempted trial includes preparation, actual access, uncomputation and
all-wire readout. The summed QDK rotation scalar is used only hypothetically
as a complete coherent operator-norm error for postselection conditioning.
No concrete synthesis, numerical-angle or hardware confidence certificate is
provided. Reference matrices and solves never construct a quantum gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .linear_systems import GridXSystem, accepted_observable_workload, decompose, measured_hhl_trial
from .linear_inverse_lcu import default_preparation, degree_for_pauli_bias, measured_inverse_trial, neumann_coefficients, structured_inverse_terms
from .resource_estimation import CircuitResourceBudget, SurfaceCodeProfile

MAX_QUERIES = 13
MAX_QUERY_WIRES = 8
MAX_QUERY_PRIMITIVES = 4_000
PROFILE = SurfaceCodeProfile()
CONTRACT = {
    "output": "one Z(q0) mean of the normalized A^-1 b solution at each fixed instance",
    "requested_absolute_error": .05,
    "algorithm_bias_allowance": .01,
    "postselection_adjusted_synthesis_bias_allowance": .01,
    "sampling_error": .03,
    "statistical_failure": .005,
    "no_acceptance_failure": .005,
    "whole_workload_physical_failure": .005,
    "requested_failure_probability": .015,
    "per_attempt_synthesis_model_scalar": 1e-6,
    "neumann_uniform_polynomial_Pauli_bias_target": .001,
    "certified_confidence_lower_bound": None,
    "hardware_output_accuracy_certified": False,
    "scope": "each case is a separate requested task, not a simultaneous guarantee over the sweep",
    "conditioning_assumption": "Hypothetically the 1e-6 per-attempt scalar bounds the entire coherent pre-readout unitary error, including angle/loading/compilation errors. QDK does not certify this assumption.",
    "mean_fit_limitation": "A mean-fit rotation scalar is not an operator-norm bound or synthesis runtime-tail guarantee; conditional compiled-output fields are proposals.",
}


@dataclass(frozen=True)
class SweepCase:
    case_id: str
    method: str
    n: int
    slices: tuple[str, ...]
    offset: int | None = None
    weights: tuple[int, ...] = ()
    phase_qubits: int | None = None
    gap: str | None = None
    degree: int | None = None

    def instance(self) -> dict:
        if self.method == "hhl":
            return {
                "method": self.method, "system_qubits": self.n,
                "matrix": "A=offset/D I+sum_j weights[j]/D (I-X_j)/2",
                "offset": self.offset, "weights_little_endian": list(self.weights),
                "phase_qubits": self.phase_qubits, "D": 1 << self.phase_qubits,
                "input": "all-zero computational basis state; no additional b-preparation gates",
                "observable": "normalized-solution Z(q0)",
            }
        return {
            "method": self.method, "system_qubits": self.n,
            "input_terms_exact": [[str(c), p] for c, p in structured_inverse_terms(self.n, self.gap)],
            "gap_promised_exact": self.gap, "degree": self.degree,
            "input": "RY(0.63) then RZ(0.41) on q0; RY(0.37) on q(n-1); other inputs zero",
            "amplification_rounds": 0, "observable": "normalized-solution Z(q0)",
        }


def _declared_cases() -> tuple[SweepCase, ...]:
    cases = [SweepCase(f"hhl-size-n{n}", "hhl", n, ("hhl-size",),
                       1, (1,) * n, math.ceil(math.log2(n + 2))) for n in (1, 2, 3, 4)]
    cases += [SweepCase(f"hhl-condition-k{k}", "hhl", 2, ("hhl-conditioning",),
                        2, (k - 1, k - 1), 2 + int(math.log2(k))) for k in (2, 4, 8)]
    degree = degree_for_pauli_bias("1/2", .001)
    cases += [SweepCase(f"neumann-size-n{n}", "neumann-lcu", n,
                        ("neumann-size", "neumann-conditioning") if n == 2 else ("neumann-size",),
                        gap="1/2", degree=degree) for n in (2, 3, 4)]
    cases += [SweepCase(f"neumann-gap-{suffix}", "neumann-lcu", 2, ("neumann-conditioning",),
                        gap=gap, degree=degree_for_pauli_bias(gap, .001))
              for suffix, gap in (("3-4", "3/4"), ("1-4", "1/4"), ("1-8", "1/8"))]
    return tuple(cases)


CASES = _declared_cases()
SLICES = {
    "hhl-size": tuple(f"hhl-size-n{n}" for n in (1, 2, 3, 4)),
    "hhl-conditioning": tuple(f"hhl-condition-k{k}" for k in (2, 4, 8)),
    "neumann-size": tuple(f"neumann-size-n{n}" for n in (2, 3, 4)),
    "neumann-conditioning": ("neumann-gap-3-4", "neumann-size-n2", "neumann-gap-1-4", "neumann-gap-1-8"),
}


def _declared(case: SweepCase) -> None:
    if case not in CASES:
        raise ValueError("Only the 13 predeclared size/conditioning cases may be generated or estimated")


def _upper_float(value: Fraction) -> float:
    result = float(value)
    return result if Fraction(result) >= value else math.nextafter(result, math.inf)


def _lower_float(value: Fraction) -> float:
    result = float(value)
    return result if Fraction(result) <= value else math.nextafter(result, -math.inf)


def _sqrt_lower(value: Fraction) -> Fraction:
    result = math.sqrt(float(value))
    while Fraction(result) ** 2 > value:
        result = math.nextafter(result, 0.0)
    return Fraction(result)


def conditioned_workload(probability_lower: Fraction, algorithm_bias: float) -> dict:
    """Conservative numerical export of a hypothetical coherent-error contract.

    Rational comparisons verify the retry union bound after converting its
    probability downward. The physical/synthesis assumptions remain external.
    """
    probability_lower = Fraction(probability_lower)
    if not 0 < probability_lower <= 1 or not math.isfinite(algorithm_bias) or not 0 <= algorithm_bias <= .01:
        raise ValueError("A positive probability and algorithm bias within the fixed .01 allocation are required")
    eta = Fraction("0.000001")
    root_lower = _sqrt_lower(probability_lower)
    if root_lower <= eta:
        raise ValueError("The hypothetical synthesis scalar is too large for postselection")
    conditioned_probability = (root_lower - eta) ** 2
    exported_probability = _lower_float(conditioned_probability)
    synthesis_bias = _upper_float(min(Fraction(2), 4 * eta / root_lower))
    if synthesis_bias > CONTRACT["postselection_adjusted_synthesis_bias_allowance"]:
        raise ValueError("Postselection-adjusted synthesis exceeds its fixed .01 allocation")
    workload = accepted_observable_workload(
        exported_probability, statistical_error=.03, statistical_failure=.005, acceptance_failure=.005)
    samples = workload["accepted_samples"]
    cap = workload["attempt_cap_per_accepted_slot"]
    q = 1 - Fraction(exported_probability)
    while samples * q ** cap > Fraction("0.005"):
        cap += 1
    workload.update({
        "status": "unexecuted capped compiled-output proposal, conditional on the coherent-eta and hardware assumptions",
        "attempt_cap_per_accepted_slot": cap, "total_capped_attempts": samples * cap,
        "no_acceptance_union_bound": _upper_float(samples * q ** cap),
        "probability_scope": "one lower bound used for every retry; never substituted by a small-case numerical acceptance",
        "coherent_eta_candidate": float(eta), "coherent_eta_certificate_available": False,
        "physical_attempts_executed": 0,
        "assumptions": ["IID fresh preparations of the declared b state",
                        "the uniform exact-rational/promise acceptance lower bound holds for the ideal circuit",
                        CONTRACT["conditioning_assumption"],
                        "one fixed normalized-solution Pauli mean; rejected attempts pay the full schedule"],
        "conditional_expected_attempts_upper_bound_without_cap": _upper_float(Fraction(samples) / Fraction(exported_probability)),
    })
    workload.pop("numerical_expected_attempts_until_success_without_cap", None)
    total = _upper_float(Fraction(algorithm_bias) + Fraction(synthesis_bias) + Fraction("0.03"))
    if total > CONTRACT["requested_absolute_error"]:
        raise ValueError("The conditional compiled-output allocation exceeds .05")
    return {
        "ideal_probability_lower_bound": _lower_float(probability_lower),
        "ideal_probability_lower_bound_exact": str(probability_lower),
        "compiled_model_probability_lower_bound": exported_probability,
        "compiled_model_probability_lower_bound_exact": str(Fraction(exported_probability)),
        "algorithm_bias_bound": algorithm_bias,
        "postselection_adjusted_synthesis_bias_bound": synthesis_bias,
        "conditional_total_error_upper_bound": total,
        "ideal_algorithm_plus_sampling_error_upper_bound": _upper_float(Fraction(algorithm_bias) + Fraction("0.03")),
        "workload": workload,
        "conditional_compiled_output_scope": CONTRACT["conditioning_assumption"],
        "hardware_output_accuracy_certified": False,
    }


def query_preflight(source: QuantumCircuit) -> QuantumCircuit:
    if source.num_qubits > MAX_QUERY_WIRES:
        raise ValueError("Physical sweeps are capped at eight logical wires")
    compiled = decompose(source)
    if len(compiled.data) - compiled.count_ops().get("barrier", 0) > MAX_QUERY_PRIMITIVES:
        raise ValueError("Physical sweeps are capped at 4000 primitive operations per complete trial")
    return compiled


def build_case(case: SweepCase) -> tuple[QuantumCircuit, dict]:
    _declared(case)
    if case.method == "hhl":
        system = GridXSystem(case.weights, case.offset, case.phase_qubits)
        source = measured_hhl_trial(system, observable="Z", target=0)
        probability_lower = Fraction(system.offset, system.offset + sum(system.weights)) ** 2
        algorithm_bias = 0.0
        model = {"condition_number": system.condition_number, "lambda_min": system.lambda_min,
                 "lambda_max": system.lambda_max, "phase_lookup_labels": list(system.phase_labels),
                 "input_preparation_counts": {}, "algorithm_precision": "exact dyadic phases and explicit reciprocal lookup under ideal exact angles"}
    else:
        terms = structured_inverse_terms(case.n, case.gap)
        polynomial = neumann_coefficients(terms, case.degree)
        source = measured_inverse_trial(terms, case.degree, default_preparation(case.n), rounds=0, observable="Z")
        probability_lower = (polynomial.delta / polynomial.normalization) ** 2
        rounding = Fraction(source.metadata["coefficient_conversion_l1_error_bound_exact"])
        # A scalar LCU normalization cancels in the conditional state. The
        # coefficient-only perturbation contributes at most 2*rounding/delta
        # to a Pauli mean, in an otherwise exact-angle gate model.
        algorithm_exact = 2 * polynomial.tail_bound / (2 - polynomial.tail_bound)
        conversion_bias = 2 * rounding / polynomial.delta
        algorithm_bias = _upper_float(algorithm_exact + conversion_bias)
        model = {**polynomial.as_metadata(),
                 "condition_number": None,
                 "polynomial_Pauli_bias_bound": _upper_float(algorithm_exact),
                 "coefficient_conversion_Pauli_bias_bound": _upper_float(conversion_bias),
                 "coefficient_conversion_l1_error_bound_exact": str(rounding),
                 "input_preparation_counts": dict(default_preparation(case.n).count_ops()),
                 "amplification_rounds": 0, "full_source_calls": 1,
                 "algorithm_precision": "uniform rational Neumann bound plus coefficient-to-float correction; angle arithmetic and synthesis remain uncertified"}
    compiled = query_preflight(source)
    counts = dict(compiled.count_ops())
    ledger = {
        "schema_version": 1, "case_id": case.case_id, "method": case.method,
        "instance": case.instance(), "slices": list(case.slices),
        "system_qubits": case.n, "matrix_dimension": 1 << case.n,
        "logical_qubits": compiled.num_qubits, "model": model,
        "source_counts": dict(source.count_ops()), "primitive_counts": counts,
        "primitive_operations": len(compiled.data) - counts.get("barrier", 0),
        "dependency_depth": compiled.depth(), "final_measurements_per_attempt": counts.get("measure", 0),
        "input_preparation_counts": model["input_preparation_counts"],
        "compiled_global_phase_radians": float(compiled.global_phase),
        "qasm_scope": "complete measured trials; pinned exporter may omit an overall phase; not a coherently controlled oracle specification",
        "contract": dict(CONTRACT), **conditioned_workload(probability_lower, algorithm_bias),
        "included": ["actual b preparation (zero additional gates for HHL's known basis input)",
                     "all explicit controlled HHL evolution/QPE/reciprocal/inverse QPE, or Neumann PREPARE/SELECT/unprepare",
                     "one unamplified source call per attempt", "all final label/clock/flag/data readouts",
                     "every rejected complete attempt and every required accepted sample"],
        "excluded": ["fresh clean logical-zero initialization/reset timing and faults",
                     "classical coefficient/lookup construction, compilation, reference solving and readout communication/processing",
                     "hardware-specific routing and decoder realization", "general matrix or arbitrary vector loading",
                     "concrete Clifford+T synthesis sequence, coherent-error certificate and synthesis runtime-tail guarantee"],
        "physical_estimation_status": "not yet queried",
    }
    attempts = ledger["workload"]["total_capped_attempts"]
    ledger["capped_primitive_counts"] = {name: count * attempts for name, count in counts.items()}
    ledger["capped_measurements"] = counts.get("measure", 0) * attempts
    return source, ledger


def physical_budget(ledger: dict) -> CircuitResourceBudget:
    return CircuitResourceBudget(1e-6, .005, ledger["workload"]["total_capped_attempts"])


def _pauli_matrix(word: str) -> np.ndarray:
    matrices = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
                "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    matrix = np.ones((1, 1), dtype=complex)
    for letter in word:
        matrix = np.kron(matrix, matrices[letter])
    return matrix


def _reference(case: SweepCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    n = case.n
    if case.method == "hhl":
        D = 1 << case.phase_qubits
        matrix = np.eye(1 << n, dtype=complex) * (case.offset + sum(case.weights) / 2) / D
        indices = np.arange(1 << n)
        for wire, weight in enumerate(case.weights):
            matrix[indices, indices ^ (1 << wire)] -= weight / (2 * D)
        b = np.zeros(1 << n, dtype=complex); b[0] = 1
        expected = case.offset / D * np.linalg.solve(matrix, b)
        metadata = {"reference_equation": "A x=b; expected accepted branch C x", "normalization_alpha": 1.0}
    else:
        terms = structured_inverse_terms(n, case.gap)
        matrix = sum(float(c) * _pauli_matrix(word) for c, word in terms)
        vectors = [np.array([1.0, 0.0], dtype=complex) for _ in range(n)]
        vectors[0] = np.array([math.cos(.63 / 2) * np.exp(-.41j / 2), math.sin(.63 / 2) * np.exp(.41j / 2)])
        vectors[-1] = np.array([math.cos(.37 / 2), math.sin(.37 / 2)], dtype=complex)
        b = np.ones(1, dtype=complex)
        for vector in reversed(vectors): b = np.kron(b, vector)
        polynomial = neumann_coefficients(terms, case.degree)
        B = matrix / float(polynomial.alpha)
        complement = np.eye(1 << n) - B
        power = np.eye(1 << n, dtype=complex); T = power.copy()
        for _ in range(case.degree): power = power @ complement; T += power
        T *= float(polynomial.delta)
        expected = T @ b / float(polynomial.normalization)
        metadata = {"reference_equation": "A x=b; expected branch T_K(A/alpha)b/s assembled by independent dense matrix powers",
                    "normalization_alpha": float(polynomial.alpha)}
    identity = {"instance": case.instance(), "observable": CONTRACT["output"], "matrix_and_input_construction": "independent Kronecker-X/Pauli arrays and analytic one-qubit input amplitudes"}
    metadata["reference_identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    metadata["matrix_bytes_sha256"] = hashlib.sha256(np.asarray(matrix, dtype="<c16").tobytes()).hexdigest()
    metadata["input_bytes_sha256"] = hashlib.sha256(np.asarray(b, dtype="<c16").tobytes()).hexdigest()
    return matrix, b, expected, metadata


def validate_case(case: SweepCase, source: QuantumCircuit) -> dict:
    _declared(case)
    compiled = query_preflight(source)
    matrix, b, expected, metadata = _reference(case)
    solution = np.linalg.solve(matrix, b)
    target = solution / np.linalg.norm(solution)
    actual = Statevector.from_instruction(source.remove_final_measurements(inplace=False)).data
    lowered = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False)).data
    start = 1 << (case.n + case.phase_qubits) if case.method == "hhl" else 0
    branch = actual[start:start + (1 << case.n)]
    success = float(np.vdot(branch, branch).real)
    accepted = branch / math.sqrt(success)
    expected_probability = float(np.vdot(expected, expected).real)
    expected_state = expected / math.sqrt(expected_probability)
    distance = float(np.linalg.norm(np.outer(accepted, accepted.conj()) - np.outer(target, target.conj()), 2))
    branch_distance = float(np.linalg.norm(np.outer(accepted, accepted.conj()) - np.outer(expected_state, expected_state.conj()), 2))
    signs = 1 - 2 * (np.arange(1 << case.n) & 1)
    z = float(np.dot(abs(accepted) ** 2, signs))
    z_reference = float(np.dot(abs(target) ** 2, signs))
    lowering_error = float(np.max(np.abs(abs(actual) ** 2 - abs(lowered) ** 2)))
    _, ledger = build_case(case)
    if abs(success - expected_probability) > 1e-10 or branch_distance > 1e-10 or lowering_error > 1e-10:
        raise AssertionError("Actual complete-trial gates do not match the independent branch or primitive lowering")
    if abs(z - z_reference) > ledger["algorithm_bias_bound"] + 2e-10:
        raise AssertionError("Numerical normalized-Z output exceeds the declared ideal algorithm bound")
    if success + 2e-10 < ledger["ideal_probability_lower_bound"]:
        raise AssertionError("Numerical success violates the uniform promise lower bound")
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        "performed": True, "method": "complete Qiskit gate statevectors versus independently assembled dense matrix powers/solve and analytic input amplitudes",
        **metadata, "matrix_eigenvalues": eigenvalues.tolist(),
        "actual_condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "reference_classical_residual_norm": float(np.linalg.norm(matrix @ solution - b)),
        "source_success_probability": success, "independent_expected_probability": expected_probability,
        "conditional_z_mean": z, "normalized_classical_solution_z_mean": z_reference,
        "z_absolute_bias": abs(z - z_reference), "conditional_state_trace_distance": distance,
        "expected_polynomial_branch_trace_distance": branch_distance,
        "source_primitive_joint_probability_max_error": lowering_error,
        "reference_solution_real": target.real.tolist(), "reference_solution_imag": target.imag.tolist(),
        "physical_attempts_executed": 0,
        "scope": "bounded ideal-angle numerical verification; no synthesized operator-error or hardware confidence certificate",
    }
