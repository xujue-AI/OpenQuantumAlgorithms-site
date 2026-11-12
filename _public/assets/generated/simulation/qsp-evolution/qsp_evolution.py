# SPDX-License-Identifier: MIT
"""Explicit postselected complex Hamiltonian evolution from two real QSP filters.

This original circuit coherently selects cosine/2 and sine/2 filters, includes
phase -i on the sine selector branch, and postselects the selector, real-part
flag and all block-encoding labels. Its accepted block approximates U(t)/4;
it is not deterministic evolution and includes no amplitude amplification.
QSP convention and primary source: Gilyen et al., arXiv:1806.01838 (2019),
Theorem17/Corollary18; Low and Chuang, doi:10.1103/PhysRevLett.118.010501
and doi:10.22331/q-2019-07-12-163. Qiskit Apache-2.0; original module MIT.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.special import gammaln, jv

if __package__:
    from .lcu import PauliTerms, build_pauli_block_encoding, postselection_attempts
    from .qsp import COSINE_PHASES, COSINE_TAU, qsp_real_response, verify_cosine_phases
else:  # Public standalone modules are downloaded together.
    from lcu import PauliTerms, build_pauli_block_encoding, postselection_attempts
    from qsp import COSINE_PHASES, COSINE_TAU, qsp_real_response, verify_cosine_phases

SINE_PHASES = (
    0.6636268934719792, -0.40191197243442534, 0.07286898272122097,
    0.22598774682915196, -0.16382791833915136, 0.5916040324107393,
    0.29322241131470433, 0.9707421541806565, -0.8835264500080798,
    -0.3345221124280212, -0.5486763824416628, 0.6322772303906963,
    0.10343217754987019, 0.3998112269864938, -0.18961297692674375,
    0.5608110687183081,
)
MAX_SYSTEM_QUBITS = 16
MAX_TERMS = 32
MAX_PHASE_DEGREE = 32
MAX_SOURCE_OPERATIONS = 50000
MAX_DENSE_TOTAL_QUBITS = 8
MAX_STATEVECTOR_QUBITS = 12


def _phase_values(phases: Sequence[float], parity: int) -> tuple[float, ...]:
    values = tuple(float(p) for p in phases)
    degree = len(values)-1
    if degree < 0 or degree > MAX_PHASE_DEGREE or degree % 2 != parity or not all(map(math.isfinite, values)):
        raise ValueError("phase degree/parity or finite-value limit invalid")
    return values


def sine_chebyshev_coefficients(tau: float = COSINE_TAU, degree: int = 15) -> np.ndarray:
    """Jacobi-Anger sine/2 polynomial; coefficient of odd T_k is +/-J_k(tau)."""
    if isinstance(tau, bool) or not math.isfinite(tau):
        raise ValueError("tau must be finite and real")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1 or degree % 2 != 1:
        raise ValueError("sine degree must be a positive odd integer")
    coefficients = np.zeros(degree+1)
    for k in range(1, degree+1, 2):
        coefficients[k] = (-1)**((k-1)//2)*jv(k, tau)
    return coefficients


def sine_tail_bound(tau: float = COSINE_TAU, degree: int = 15) -> float:
    """Absolute-series/geometric upper bound on the infinite odd Bessel tail."""
    sine_chebyshev_coefficients(tau, degree)
    a = abs(tau)/2
    if a == 0:
        return 0.0
    n = degree+2
    ratio = a*a/((n+1)*(n+2))
    if ratio >= 1:
        raise ValueError("degree too small for geometric tail bound")
    first = math.exp(n*math.log(a)-gammaln(n+1)+a*a/(n+1))
    return first/(1-ratio)


def verify_sine_phases(phases: Sequence[float] = SINE_PHASES, tau: float = COSINE_TAU) -> dict:
    """Dense grid and numerical coefficient reconstruction, not formal intervals."""
    phases = _phase_values(phases, 1)
    degree = len(phases)-1
    target = sine_chebyshev_coefficients(tau, degree)
    nodes = np.cos(np.pi*(np.arange(128)+.5)/128)
    achieved = chebfit(nodes, qsp_real_response(nodes, phases), degree)
    discrepancy = float(np.sum(np.abs(achieved-target)))
    grid = np.linspace(-1, 1, 10001)
    actual = qsp_real_response(grid, phases)
    truncation = sine_tail_bound(tau, degree)
    return {
        "degree": degree, "tau": tau, "target": "sin(tau*x)/2 on [-1,1]",
        "target_chebyshev_coefficients": target.tolist(), "achieved_chebyshev_coefficients": achieved.tolist(),
        "phase_fit_max_residual_10001_point_grid": float(np.max(np.abs(actual-chebval(grid, target)))),
        "target_max_residual_10001_point_grid": float(np.max(np.abs(actual-np.sin(tau*grid)/2))),
        "numerically_reconstructed_coefficient_l1_discrepancy": discrepancy,
        "analytic_chebyshev_truncation_bound": truncation,
        "coefficient_reconstruction_guard": 1e-12,
        "conditional_numerical_uniform_error_projection": discrepancy+truncation+1e-12,
        "certified": False,
        "verification_scope": "analytic tail bound plus floating coefficient reconstruction and dense-grid check; no formal interval proof",
    }


def fit_sine_phases(tau: float = COSINE_TAU, degree: int = 15) -> tuple[tuple[float, ...], dict]:
    """Bounded optional classical phase reproduction: seed31, <=8x4000 evaluations."""
    if degree > 17:
        raise ValueError("sine phase optimization bounded to degree17")
    target_coefficients = sine_chebyshev_coefficients(tau, degree)
    nodes = np.cos(np.pi*(np.arange(80)+.5)/80)
    target = chebval(nodes, target_coefficients)
    rng = np.random.default_rng(31)
    for attempt in range(8):
        initial = rng.normal(0, .2, degree+1)
        initial[0] = initial[-1] = np.pi/4
        result = least_squares(lambda p: qsp_real_response(nodes, p)-target, initial,
                               ftol=1e-13, gtol=1e-13, xtol=1e-13, max_nfev=4000)
        residual = float(np.max(np.abs(qsp_real_response(nodes, result.x)-target)))
        if residual < 5e-12:
            return tuple(result.x), {
                "method": "scipy.optimize.least_squares", "seed": 31,
                "fit_nodes": 80, "successful_restart_zero_based": attempt,
                "function_evaluations_successful_restart": result.nfev,
                "node_residual": residual, "maximum_restarts": 8,
                "maximum_function_evaluations_per_restart": 4000,
                "ftol": 1e-13, "gtol": 1e-13, "xtol": 1e-13,
            }
    raise RuntimeError("bounded sine phase fit failed 5e-12 node-residual criterion")


def complex_qsp_response(x, cosine_phases: Sequence[float] = COSINE_PHASES,
                         sine_phases: Sequence[float] = SINE_PHASES) -> np.ndarray:
    """Scalar encoded block (real(Pc)-i real(Ps))/2, target exp(-i tau x)/4."""
    cosine_phases = _phase_values(cosine_phases, 0)
    sine_phases = _phase_values(sine_phases, 1)
    return (qsp_real_response(x, cosine_phases)-1j*qsp_real_response(x, sine_phases))/2


def _branch_signed_phase(circuit: QuantumCircuit, index: list[int], flag: int,
                         selector: int, phi: float, selector_value: int):
    """Conditional exp(i phi (2Pi-I)Zflag); BOTH phase terms are controlled."""
    if selector_value == 0:
        circuit.x(selector)
    circuit.append(RZGate(2*phi).control(1, annotated=False), [selector, flag])
    if index:
        circuit.x(index)
    circuit.append(RZGate(-4*phi).control(len(index)+1, annotated=False), index+[selector, flag])
    if index:
        circuit.x(index)
    if selector_value == 0:
        circuit.x(selector)


def build_qsp_evolution(terms: PauliTerms, time: float = 1.0, *,
                        cosine_phases: Sequence[float] = COSINE_PHASES,
                        sine_phases: Sequence[float] = SINE_PHASES,
                        phase_tau: float = COSINE_TAU,
                        phase_verification_tolerance: float = 1e-8) -> QuantumCircuit:
    """Parameterized explicit complex QSP evolution block, no dense quantum gate.

    The supplied phase target must satisfy phase_tau=alpha*time, where
    alpha=sum(abs(real Pauli coefficients)). Frozen phases target4.1 only.
    Cosine degree must be even and sine degree exactly one higher. Share the
    common B calls, then apply one selector-controlled B and sine endpoint.
    Data precedes index, real-part flag, and selector. All ancillas must be
    zero at readout to accept U(t)/4. Synthesis and physical faults excluded.
    """
    terms = tuple(terms)
    if not terms or len(terms) > MAX_TERMS:
        raise ValueError(f"materialization bounded to1..{MAX_TERMS} Pauli terms")
    if len(terms[0][1]) > MAX_SYSTEM_QUBITS:
        raise ValueError(f"materialization bounded to{MAX_SYSTEM_QUBITS} system qubits")
    if any(complex(c).imag != 0 for c, _ in terms):
        raise ValueError("evolution requires real Pauli coefficients")
    if isinstance(time, bool) or not math.isfinite(time) or isinstance(phase_tau, bool) or not math.isfinite(phase_tau):
        raise ValueError("time and phase_tau must be finite real values")
    alpha = float(sum(abs(complex(c)) for c, _ in terms))
    if not math.isclose(alpha*time, phase_tau, abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("phase_tau must match normalization alpha times physical time")
    cosine_phases = _phase_values(cosine_phases, 0)
    sine_phases = _phase_values(sine_phases, 1)
    degree_c, degree_s = len(cosine_phases)-1, len(sine_phases)-1
    if degree_s != degree_c+1:
        raise ValueError("shared-query circuit requires sine degree=cosine degree+1")
    if not math.isfinite(phase_verification_tolerance) or phase_verification_tolerance <= 0:
        raise ValueError("phase verification tolerance must be positive and finite")
    cosine_check = verify_cosine_phases(cosine_phases, phase_tau)
    sine_check = verify_sine_phases(sine_phases, phase_tau)
    if max(cosine_check["target_max_residual_10001_point_grid"],
           sine_check["target_max_residual_10001_point_grid"]) > phase_verification_tolerance:
        raise ValueError("supplied phases fail dense-grid verification for the requested phase_tau")
    block = build_pauli_block_encoding(terms)
    if (degree_s+1)*(block.size()+20) > MAX_SOURCE_OPERATIONS:
        raise ValueError("source operation materialization budget exceeded")
    n_system, n_index = block.metadata["n_system"], block.metadata["n_index"]
    flag, selector = block.num_qubits, block.num_qubits+1
    index = list(range(n_system, flag))
    circuit = QuantumCircuit(block.num_qubits+2, name="Postselected QSP evolution")
    circuit.h(flag)
    circuit.h(selector)
    for j in range(degree_c+1):
        _branch_signed_phase(circuit, index, flag, selector, cosine_phases[degree_c-j], 0)
        _branch_signed_phase(circuit, index, flag, selector, sine_phases[degree_s-j], 1)
        if j < degree_c:
            circuit.compose(block, qubits=list(range(flag)), inplace=True)
    circuit.append(block.to_gate(label="controlled PREPARE SELECT unprepare").control(1, annotated=False),
                   [selector]+list(range(flag)))
    _branch_signed_phase(circuit, index, flag, selector, sine_phases[0], 1)
    circuit.h(flag)
    circuit.sdg(selector)
    circuit.h(selector)
    circuit.metadata = {
        **block.metadata, "n_flag": 1, "n_selector": 1,
        "time": float(time), "phase_tau": float(phase_tau),
        "phase_verification": {"performed": True, "certified": False,
                               "method": "10001-point scalar grid against requested cos/sin target",
                               "tolerance": phase_verification_tolerance},
        "cosine_degree": degree_c, "sine_degree": degree_s,
        "cosine_phases": list(cosine_phases), "sine_phases": list(sine_phases),
        "uncontrolled_block_encoding_calls": degree_c, "controlled_block_encoding_calls": 1,
        "branch_conditioned_projector_phase_calls": len(cosine_phases)+len(sine_phases),
        "encoded_operator": "(real Pc(H/alpha)-i real Ps(H/alpha))/2; target exp(-iHt)/4",
        "accepted_branch_normalization": 4,
        "postselection": "all index bits, real-part flag, selector are0",
        "amplitude_amplification": "not implemented",
        "rotations_synthesized": False,
        "status": "explicit postselected complex evolution; deterministic amplified evolution not implemented",
    }
    return circuit


def evolution_error_contract(cosine_phases: Sequence[float] = COSINE_PHASES,
                             sine_phases: Sequence[float] = SINE_PHASES,
                             tau: float = COSINE_TAU, failure_target: float = .01) -> dict:
    """Conditional numerical projections, acceptance and pending synthesis/faults."""
    cosine = verify_cosine_phases(cosine_phases, tau)
    cosine["conditional_numerical_uniform_error_projection"] = cosine.pop("uniform_error_bound_with_numerical_reconstruction_guard")
    cosine["certified"] = False
    sine = verify_sine_phases(sine_phases, tau)
    block_bound = (cosine["conditional_numerical_uniform_error_projection"]+
                   sine["conditional_numerical_uniform_error_projection"])/2
    if block_bound >= .25:
        raise ValueError("polynomial accuracy cannot ensure nonzero uniform acceptance")
    minimum_probability = (.25-block_bound)**2
    return {
        "cosine_filter": cosine, "sine_filter": sine,
        "conditional_numerical_U_over_4_operator_error_projection": block_bound,
        "conditional_numerical_4K_minus_U_operator_error_projection": 4*block_bound,
        "conditional_numerical_acceptance_probability_projection": minimum_probability,
        "ideal_exact_scaled_unitary_acceptance_probability": 1/16,
        "complete_attempts_for_conditional_acceptance_projection": postselection_attempts(minimum_probability, failure_target),
        "acceptance_failure_target": failure_target,
        "coherent_synthesis_error": "excluded; all primitive continuous rotations remain unsynthesized",
        "physical_faults": "excluded; separate physical-fault budget is pending",
        "statistical_error": "distinct from approximation and acceptance; observable shots accounted separately",
        "certified": False,
        "accuracy_bound_scope": "conditional numerical projections use analytic Bessel tails plus assumed floating coefficient guard; no certified coefficient, gate or acceptance bound",
    }


def postselected_evolution_state(circuit: QuantumCircuit, initial_state: np.ndarray) -> tuple[np.ndarray, float]:
    """Ideal circuit statevector and accepted branch before final system readout."""
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"statevector bounded to{MAX_STATEVECTOR_QUBITS} total qubits")
    n_system = circuit.metadata["n_system"]
    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (2**n_system,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state has invalid dimension or entries")
    if not np.isclose(np.vdot(state, state).real, 1, atol=1e-12, rtol=0):
        raise ValueError("initial_state must be normalized")
    full = np.zeros(2**circuit.num_qubits, dtype=complex)
    full[:len(state)] = state
    branch = Statevector(full).evolve(circuit).data[:len(state)]
    probability = float(np.vdot(branch, branch).real)
    if probability <= 1e-30:
        raise ValueError("postselection probability is numerically zero")
    return branch/math.sqrt(probability), probability


def compare_qsp_evolution(circuit: QuantumCircuit, hamiltonian: np.ndarray,
                          initial_state: np.ndarray, observable: np.ndarray) -> dict:
    """Bounded classical check; conditional states are before final system readout."""
    if circuit.num_qubits > MAX_DENSE_TOTAL_QUBITS:
        raise ValueError(f"dense Operator checks bounded to{MAX_DENSE_TOTAL_QUBITS} total qubits")
    dimension = 2**circuit.metadata["n_system"]
    h = np.asarray(hamiltonian, dtype=complex)
    obs = np.asarray(observable, dtype=complex)
    if h.shape != (dimension, dimension) or obs.shape != h.shape:
        raise ValueError("classical matrix dimensions invalid")
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(obs)):
        raise ValueError("classical matrix entries must be finite")
    if not np.allclose(h, h.conj().T, atol=1e-12, rtol=0) or not np.allclose(obs, obs.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian and observable must be Hermitian")
    actual = Operator(circuit).data[:dimension, :dimension]
    target = expm(-1j*circuit.metadata["time"]*h)
    state, probability = postselected_evolution_state(circuit, initial_state)
    reference = target@initial_state
    z_actual = float(np.vdot(state, obs@state).real)
    z_exact = float(np.vdot(reference, obs@reference).real)
    # 4K is the unnormalized approximate linear map. Renormalized postselected
    # data evolution is nonlinear when acceptance has input-state dependence.
    return {
        "encoded_block_error": float(np.linalg.norm(actual-target/4, 2)),
        "decoded_linear_map_operator_error": float(np.linalg.norm(4*actual-target, 2)),
        "conditional_state_check_scope": "accepted data state before final system readout",
        "conditional_state_vector_error": float(np.linalg.norm(state-reference)),
        "conditional_state_infidelity": float(max(0.0, 1-abs(np.vdot(reference, state))**2)),
        "observable_exact": z_exact, "observable_conditional": z_actual,
        "observable_error": abs(z_actual-z_exact), "acceptance_probability": probability,
        "accepted_block_singular_values": np.linalg.svd(actual, compute_uv=False).tolist(),
        "finite_H_numerical_minimum_acceptance_probability": float(np.linalg.svd(actual, compute_uv=False)[-1]**2),
        "finite_H_numerical_maximum_acceptance_probability": float(np.linalg.svd(actual, compute_uv=False)[0]**2),
        "acceptance_certified": False,

    }
