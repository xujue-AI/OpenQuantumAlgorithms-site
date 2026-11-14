# SPDX-License-Identifier: MIT
"""Explicit QSVT inverse for a declared two-eigenvalue Pauli-word family.

An original spectrum-matched odd cubic is implemented using the project's
PREPARE/SELECT/unprepare and real-QSP circuits. It exactly represents c/x at
x=1/2 and 1; it is NOT a general high-accuracy reciprocal polynomial. The
accepted block is p(A/alpha), p(x)=(5x-4x^3)/4. Matrix assembly and classical
inversion occur only in bounded independent checks, never gate generation.

Qiskit is Apache-2.0. NumPy/SciPy are BSD-3-Clause. This module is MIT.
Quantum singular value transformation: Gilyen et al., arXiv:1806.01838,
doi:10.1145/3313276.3316366, Theorem17/Corollary18 (Hermitian case).
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import least_squares

from .simulation.lcu import PauliTerms
from .simulation.qsp import build_real_qsp_filter, qsp_real_response

MAX_GENERATED_SYSTEM_QUBITS = 24
MAX_INPUT_TERMS = 32
MAX_VALIDATED_SYSTEM_QUBITS = 6
MAX_STATEVECTOR_QUBITS = 12
DEFAULT_TERMS = ((0.75, "I"), (-0.25, "X"))
INVERSE_SCALE = 0.25
INVERSE_CHEBYSHEV_COEFFICIENTS = (0.0, 0.5, 0.0, -0.25)
INVERSE_PHASES = (
    0.043750941193310945, 0.4773037119407437,
    0.5955800476346843, 0.20148137088407908,
)


def inverse_polynomial(x):
    """Spectrum-matched cubic, odd and bounded by 5sqrt(15)/36 on [-1,1]."""
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
        raise ValueError("signal values must be finite and in [-1,1]")
    return (5 * values - 4 * values**3) / 4


def reciprocal_interval_error_bound(delta: float) -> float:
    """Exact scalar maximum of |p(x)-1/(4x)| on delta<=|x|<=1.

    This can be large: the cubic has no arbitrary-precision promise on an
    interval. Evaluating the endpoints and the sole positive stationary point
    of the error (squared x=(5+sqrt(73))/24) yields its exact extremum, up to
    ordinary floating evaluation. No interval arithmetic certificate is used.
    """
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < delta <= 1:
        raise ValueError("delta must lie in (0,1]")
    critical = math.sqrt((5 + math.sqrt(73)) / 24)
    points = [delta, 1.0]
    if delta <= critical:
        points.append(critical)
    return max(abs(float(inverse_polynomial(x)) - 0.25 / x) for x in points)


def verify_inverse_phases(phases: Sequence[float] = INVERSE_PHASES) -> dict:
    """Reconstruct degree3 coefficients and check 10,001 scalar values.

    A 1e-12 floating reconstruction guard is disclosed, not formally proved.
    The polynomial norm and exact inverse values at +/-1/2,+/-1 follow from
    the explicit cubic; phase/circuit agreement is numerical evidence.
    """
    phases = tuple(float(p) for p in phases)
    if len(phases) != 4 or not all(map(math.isfinite, phases)):
        raise ValueError("inverse cubic requires four finite QSP phases")
    nodes = np.cos(np.pi * (np.arange(32) + 0.5) / 32)
    achieved = chebfit(nodes, qsp_real_response(nodes, phases), 3)
    discrepancy = float(np.sum(np.abs(achieved - INVERSE_CHEBYSHEV_COEFFICIENTS)))
    grid = np.linspace(-1, 1, 10001)
    residual = float(np.max(np.abs(qsp_real_response(grid, phases) - inverse_polynomial(grid))))
    return {
        "degree": 3, "phases": list(phases), "scale": INVERSE_SCALE,
        "target": "p(x)=(5*x-4*x**3)/4; c/x exactly at x=+/-0.5,+/-1",
        "target_chebyshev_coefficients": list(INVERSE_CHEBYSHEV_COEFFICIENTS),
        "achieved_chebyshev_coefficients": achieved.tolist(),
        "coefficient_l1_discrepancy": discrepancy,
        "dense_grid_max_phase_error": residual,
        "uniform_phase_discrepancy_with_numerical_guard": discrepancy + 1e-12,
        "floating_reconstruction_guard": 1e-12,
        "analytic_polynomial_max_magnitude": 5 * math.sqrt(15) / 36,
        "reciprocal_max_error_interval_0_5_to_1": reciprocal_interval_error_bound(0.5),
        "spectral_contract": "normalized spectrum contained in {-1,-0.5,0.5,1} for exact inversion",
        "verification_scope": "analytic cubic identities plus numerical phase/coefficient/grid checks; no formal or interval proof",
    }


def fit_inverse_phases() -> tuple[tuple[float, ...], dict]:
    """Reproduce frozen cubic phases with seed29,64nodes,8 capped restarts."""
    nodes = np.cos(np.pi * (np.arange(64) + 0.5) / 64)
    target = inverse_polynomial(nodes)
    rng = np.random.default_rng(29)
    for attempt in range(8):
        initial = rng.normal(0, 0.2, 4)
        initial[0] = initial[-1] = np.pi / 4
        result = least_squares(
            lambda p: qsp_real_response(nodes, p) - target, initial,
            ftol=1e-13, gtol=1e-13, xtol=1e-13, max_nfev=2000,
        )
        residual = float(np.max(np.abs(qsp_real_response(nodes, result.x) - target)))
        if residual < 5e-12:
            return tuple(float(p) for p in result.x), {
                "method": "scipy.optimize.least_squares", "seed": 29,
                "nodes": 64, "successful_restart_zero_based": attempt,
                "function_evaluations_successful_restart": result.nfev,
                "maximum_restarts": 8, "maximum_function_evaluations_per_restart": 2000,
                "node_residual": residual, "ftol": 1e-13, "gtol": 1e-13, "xtol": 1e-13,
            }
    raise RuntimeError("bounded cubic phase fitting did not meet its 5e-12 node criterion")


def pauli_word_inverse_terms(pauli_word: str = "X") -> tuple[tuple[float, str], ...]:
    """A/alpha=.75I-.25P, for an explicitly supplied nonidentity Pauli word P.

    This structured family has eigenvalues1/2,1 and condition number2.
    The coefficient list and controlled Pauli implementation scale with word
    width. It is not an arbitrary matrix loader or a general linear solver.
    """
    if not isinstance(pauli_word, str) or not pauli_word or set(pauli_word) - set("IXYZ"):
        raise ValueError("Pauli word must be a nonempty string over IXYZ")
    if len(pauli_word) > MAX_GENERATED_SYSTEM_QUBITS or set(pauli_word) == {"I"}:
        raise ValueError("supply a nonidentity Pauli word of at most24 system qubits")
    return ((0.75, "I" * len(pauli_word)), (-0.25, pauli_word))


def _validated_terms(terms: PauliTerms) -> tuple[tuple[float, str], ...]:
    terms = tuple(terms)
    if not terms or len(terms) > MAX_INPUT_TERMS:
        raise ValueError(f"supply1 to {MAX_INPUT_TERMS} explicit Pauli terms")
    width = len(terms[0][1]) if isinstance(terms[0][1], str) else 0
    if not 1 <= width <= MAX_GENERATED_SYSTEM_QUBITS:
        raise ValueError("system width must lie between1 and24")
    clean = []
    for coefficient, word in terms:
        value = complex(coefficient)
        if not math.isfinite(value.real) or value.imag != 0:
            raise ValueError("QSVT requires finite real coefficients")
        if not isinstance(word, str) or len(word) != width or set(word) - set("IXYZ"):
            raise ValueError("Pauli labels must have equal width and use IXYZ")
        clean.append((float(value.real), word))
    if not any(coefficient for coefficient, _ in clean):
        raise ValueError("matrix access must have a nonzero coefficient")
    return tuple(clean)


def build_inverse_qsvt(
    terms: PauliTerms = DEFAULT_TERMS,
    phases: Sequence[float] = INVERSE_PHASES,
) -> QuantumCircuit:
    """Generate actual cubic QSVT gates; data,label,real-part flag ordering.

    The accepted block is p(A/alpha), alpha=sum|a_j|. Arbitrary supplied terms
    produce this polynomial, not necessarily an inverse: exact inversion
    requires the explicitly stated normalized-spectrum contract. No matrix
    eigenvalues, classical solution or dense unitary constructs any gate.
    """
    terms = _validated_terms(terms)
    phases = tuple(float(p) for p in phases)
    if len(phases) != 4 or not all(map(math.isfinite, phases)):
        raise ValueError("inverse cubic requires four finite phases")
    circuit = build_real_qsp_filter(terms, phases)
    circuit.name = "Cubic QSVT inverse"
    circuit.metadata.update({
        "algorithm": "Hermitian QSVT via real QSP with explicit Pauli LCU access",
        "inverse_scale": INVERSE_SCALE, "physical_matrix_inverse_multiplier": circuit.metadata["normalization"] * INVERSE_SCALE, "polynomial": "(5*x-4*x**3)/4",
        "exact_inverse_normalized_spectral_set": [-1.0, -0.5, 0.5, 1.0],
        "inverse_scope": "spectrum-matched; arbitrary terms only receive the stated cubic filter",
        "postselection": "label and real-part flag all zero; final data readout unconditioned on data value",
        "status": "implemented spectrum-matched inverse; unsynthesized numerical phase circuit",
    })
    return circuit


def _preparation(preparation: QuantumCircuit | None, width: int) -> QuantumCircuit:
    if preparation is None:
        return QuantumCircuit(width, name="prepare b=0")
    if preparation.num_qubits != width or preparation.num_clbits or preparation.num_parameters:
        raise ValueError("preparation must be a bound unitary circuit on the system register")
    forbidden = {
        "measure", "reset", "initialize", "if_else", "while_loop", "for_loop",
        "switch_case", "store", "break_loop", "continue_loop",
    }
    checked_definitions: set[int] = set()

    def check_definition(definition: QuantumCircuit) -> None:
        # Circuit identity guards repeated definitions; descendants are still
        # checked once. No hidden classical register or symbolic angle is allowed.
        if id(definition) in checked_definitions:
            return
        checked_definitions.add(id(definition))
        if definition.num_clbits or definition.num_parameters:
            raise ValueError("preparation must be a bound unitary circuit without classical bits")
        try:
            if not math.isfinite(float(definition.global_phase)):
                raise ValueError("nonfinite preparation global phase")
        except (TypeError, ValueError) as error:
            raise ValueError("preparation global phase must be finite and bound") from error
        for item in definition.data:
            operation = item.operation
            if (operation.name in forbidden or operation.num_clbits
                    or getattr(operation, "condition", None) is not None
                    or hasattr(operation, "blocks")):
                raise ValueError("preparation must use unitary gates without resets, measurements or feedback")
            for parameter in operation.params:
                try:
                    numeric = np.asarray(parameter, dtype=complex)
                except (TypeError, ValueError) as error:
                    raise ValueError("preparation parameters must be finite and numeric") from error
                if not np.all(np.isfinite(numeric)):
                    raise ValueError("preparation parameters must be finite and numeric")
            nested = operation.definition
            if nested is not None:
                check_definition(nested)
            elif not isinstance(operation, Gate) and operation.name != "barrier":
                raise ValueError("preparation contains an opaque instruction without unitary gate semantics")

    check_definition(preparation)
    return preparation


def qsvt_prepared_circuit(
    terms: PauliTerms = DEFAULT_TERMS,
    preparation: QuantumCircuit | None = None,
) -> QuantumCircuit:
    filter_circuit = build_inverse_qsvt(terms)
    width = filter_circuit.metadata["n_system"]
    circuit = QuantumCircuit(filter_circuit.num_qubits, name="Prepare and QSVT invert")
    circuit.compose(_preparation(preparation, width), qubits=list(range(width)), inplace=True)
    circuit.compose(filter_circuit, inplace=True)
    circuit.metadata = {**filter_circuit.metadata, "input_preparation": "explicit circuit acting on all-zero data input"}
    return circuit


def qsvt_measured_trial(
    terms: PauliTerms = DEFAULT_TERMS,
    preparation: QuantumCircuit | None = None,
    observable: str = "Z",
) -> QuantumCircuit:
    """Complete fixed X/Y/Z(q0) readout trial including actual basis gates.

    All data and ancillas are measured. Failed label/flag trials still pay all
    work. Initial all-zero qubits are assumed; reset timing/faults are outside
    this static circuit. Each retry needs fresh re-preparation of b.
    """
    if observable not in {"X", "Y", "Z"}:
        raise ValueError("observable must be X,Y orZ on q0")
    circuit = qsvt_prepared_circuit(terms, preparation)
    if observable == "X":
        circuit.h(0)
    elif observable == "Y":
        circuit.sdg(0)
        circuit.h(0)
    circuit.measure_all()
    circuit.metadata.update({"observable": f"{observable}(q0)", "success_predicate": "all measured bits at indices>=n_system are zero"})
    return circuit


def compare_inverse_qsvt(
    terms: PauliTerms = DEFAULT_TERMS,
    preparation: QuantumCircuit | None = None,
) -> dict:
    """Bounded statevector/cubic checks versus an independent classical solve."""
    circuit = qsvt_prepared_circuit(terms, preparation)
    width = circuit.metadata["n_system"]
    if width > MAX_VALIDATED_SYSTEM_QUBITS or circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError("independent dense checks limited to6 data and12 total qubits")
    b = Statevector.from_instruction(_preparation(preparation, width)).data
    matrix = SparsePauliOp.from_list([(word, coefficient) for coefficient, word in _validated_terms(terms)]).to_matrix()
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(np.abs(eigenvalues)) < 1e-12:
        raise ValueError("classical inverse comparison requires a numerically nonsingular matrix")
    alpha = circuit.metadata["normalization"]
    normalized = matrix / alpha
    ideal_block = (5 * normalized - 4 * normalized @ normalized @ normalized) / 4
    branch = Statevector.from_instruction(circuit).data[: 2**width]
    probability = float(np.vdot(branch, branch).real)
    if probability <= 1e-30:
        raise ValueError("filter has zero numerical acceptance probability")
    state = branch / math.sqrt(probability)
    target = np.linalg.solve(matrix, b)
    target /= np.linalg.norm(target)
    fidelity = min(1.0, float(abs(np.vdot(target, state))**2))
    bit_zero = np.arange(len(state)) % 2 == 0
    z_actual = float(2 * np.sum(np.abs(state[bit_zero])**2) - 1)
    z_target = float(2 * np.sum(np.abs(target[bit_zero])**2) - 1)
    return {
        "logical_qubits": circuit.num_qubits, "system_qubits": width,
        "matrix_eigenvalues": eigenvalues.tolist(), "normalization": alpha,
        "normalized_eigenvalues": (eigenvalues / alpha).tolist(),
        "condition_number": float(np.max(np.abs(eigenvalues)) / np.min(np.abs(eigenvalues))),
        "acceptance_probability": probability,
        "ideal_cubic_acceptance_probability": float(np.linalg.norm(ideal_block @ b)**2),
        "accepted_branch_error_vs_cubic": float(np.linalg.norm(branch - ideal_block @ b)),
        "solution_state_fidelity": fidelity, "solution_state_trace_distance": float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(np.outer(state, state.conj()) - np.outer(target, target.conj()))))),
        "solution_state_phase_invariant_error": float(np.linalg.norm(state - target * np.exp(1j * np.angle(np.vdot(target, state))))),
        "z_actual": z_actual, "z_classical_solution": z_target, "z_absolute_bias": abs(z_actual - z_target),
        "solution_state_real": state.real.tolist(), "solution_state_imag": state.imag.tolist(),
        "classical_solution_state_real": target.real.tolist(), "classical_solution_state_imag": target.imag.tolist(),
        "exact_spectral_contract_numerically_met": bool(np.all(np.min(np.abs((eigenvalues / alpha)[:, None] - np.array([-1, -0.5, 0.5, 1])[None, :]), axis=1) < 1e-12)),
        "validation_scope": "bounded ideal gates and numerical arithmetic; no synthesis or physical error certificate",
    }
