# SPDX-License-Identifier: MIT
"""Original bounded QSP cosine-filter and qubitization circuits.

This is a checked polynomial primitive, not complete Hamiltonian evolution.
Convention: D(phi)=exp(i phi (2 Pi_0-I)); V=D(phi_0) B ... B D(phi_d)
for a Hermitian involution B that block-encodes A=H/alpha. An extra flag
extracts Re(P(A)) by coherently combining the +phi and -phi sequences.
See Gilyen, Su, Low, Wiebe (2019), arXiv:1806.01838, Theorem 17 and
Corollary 18, doi:10.1145/3313276.3316366. Qiskit is Apache-2.0; this
module and frozen numerical phases are original MIT implementation assets.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate
from qiskit.quantum_info import Statevector
from scipy.optimize import least_squares
from scipy.special import gammaln, jv

if __package__:
    from .lcu import PauliTerms, build_pauli_block_encoding
else:  # Standalone public downloads keep qsp.py and lcu.py together.
    from lcu import PauliTerms, build_pauli_block_encoding

MAX_QSP_DEGREE = 64
MAX_STATEVECTOR_QUBITS = 12
COSINE_TAU = 4.1
COSINE_PHASES = (
    0.877901529533427, 0.2162140998463567, 0.16749017740701486,
    -0.45204129180680813, -0.37672022483622003, -0.5504438599172469,
    -0.4467429640660551, 0.5605234774971832, 0.33433609203503734,
    -0.08395243905459, -0.2891623970772259, 0.38417736673036723,
    0.3116699753749945, 0.3330838865835369, 0.8759866069501494,
)


def _phases(phases: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(p) for p in phases)
    if not values or len(values) > MAX_QSP_DEGREE + 1 or not all(map(math.isfinite, values)):
        raise ValueError(f"phases must contain 1 to {MAX_QSP_DEGREE+1} finite values")
    return values


def _hermitian_block(terms: PauliTerms) -> QuantumCircuit:
    terms = tuple(terms)
    if any(abs(complex(c).imag) > 0 for c, _ in terms):
        raise ValueError("QSP reflection requires real Pauli coefficients")
    # Real coefficient signs give +/-Pauli in SELECT; padded branches are I.
    # Consequently SELECT and B are Hermitian involutions.
    return build_pauli_block_encoding(terms)


def qsp_real_response(x, phases: Sequence[float] = COSINE_PHASES) -> np.ndarray:
    """Independent 2x2 signal-model polynomial, for scalar or array x in [-1,1]."""
    phases = _phases(phases)
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(np.abs(values) > 1):
        raise ValueError("signal values must be finite and in [-1,1]")
    shape = values.shape
    flat = values.reshape(-1)
    sine = np.sqrt(np.maximum(0, 1-flat*flat))
    signal = np.array([[flat, sine], [sine, -flat]]).transpose(2, 0, 1).astype(complex)
    response = np.tile(np.eye(2, dtype=complex), (len(flat), 1, 1))
    for i, phi in enumerate(phases):
        response[:, :, 0] *= np.exp(1j*phi)
        response[:, :, 1] *= np.exp(-1j*phi)
        if i < len(phases)-1:
            response = response @ signal
    return response[:, 0, 0].real.reshape(shape)


def cosine_chebyshev_coefficients(tau: float = COSINE_TAU, degree: int = 14) -> np.ndarray:
    """Truncated Jacobi-Anger expansion of cos(tau*x)/2, no quantum matrix gate."""
    if not math.isfinite(tau):
        raise ValueError("tau must be finite")
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0 or degree % 2:
        raise ValueError("cosine degree must be a nonnegative even integer")
    coefficients = np.zeros(degree+1)
    coefficients[0] = jv(0, tau)/2
    for k in range(2, degree+1, 2):
        coefficients[k] = (-1)**(k//2)*jv(k, tau)
    return coefficients


def cosine_tail_bound(tau: float = COSINE_TAU, degree: int = 14) -> float:
    """Analytic uniform bound on omitted even Bessel coefficients.

    From the absolute Bessel series, |J_n(tau)| <= a^n/n! exp(a^2/(n+1)),
    a=|tau|/2. Ratios of successive even upper bounds are at most
    a^2/((n+1)(n+2)); a geometric series bounds the infinite omitted tail.
    """
    cosine_chebyshev_coefficients(tau, degree)  # validation
    a = abs(tau)/2
    if a == 0:
        return 0.0
    n = degree+2
    ratio = a*a/((n+1)*(n+2))
    if ratio >= 1:
        raise ValueError("degree is too small for this geometric tail bound")
    first_bound = math.exp(n*math.log(a)-gammaln(n+1)+a*a/(n+1))
    return first_bound/(1-ratio)


def verify_cosine_phases(phases: Sequence[float] = COSINE_PHASES, tau: float = COSINE_TAU) -> dict:
    """Bounded numerical verification; coefficient reconstruction is not a formal proof."""
    phases = _phases(phases)
    degree = len(phases)-1
    target = cosine_chebyshev_coefficients(tau, degree)
    nodes = np.cos(np.pi*(np.arange(128)+0.5)/128)
    achieved = chebfit(nodes, qsp_real_response(nodes, phases), degree)
    discrepancy = float(np.sum(np.abs(achieved-target)))
    grid = np.linspace(-1, 1, 10001)
    phase_residual = float(np.max(np.abs(qsp_real_response(grid, phases)-chebval(grid, target))))
    total_residual = float(np.max(np.abs(qsp_real_response(grid, phases)-np.cos(tau*grid)/2)))
    truncation_bound = cosine_tail_bound(tau, degree)
    return {
        "degree": degree, "tau": tau,
        "target": "cos(tau*x)/2 on [-1,1]",
        "target_chebyshev_coefficients": target.tolist(),
        "achieved_chebyshev_coefficients": achieved.tolist(),
        "phase_fit_max_residual_10001_point_grid": phase_residual,
        "target_max_residual_10001_point_grid": total_residual,
        "numerically_reconstructed_coefficient_l1_discrepancy": discrepancy,
        "analytic_chebyshev_truncation_bound": truncation_bound,
        "uniform_error_bound_with_numerical_reconstruction_guard": discrepancy+truncation_bound+1e-12,
        "coefficient_reconstruction_guard": 1e-12,
        "verification_scope": "analytic Bessel-tail bound plus floating coefficient reconstruction and dense-grid checks; no formal interval proof",
    }


def fit_cosine_phases(tau: float = COSINE_TAU, degree: int = 14) -> tuple[tuple[float, ...], dict]:
    """Optional bounded reproduction of the frozen numerical phase fit.

    Uses existing SciPy, deterministic seed17, 80 Chebyshev nodes, at most
    8 restarts and 4000 function evaluations per restart. Failure raises rather
    than returning phases with an unsupported accuracy claim. Frozen phases
    can be verified without this slower optional classical preprocessing.
    """
    if degree > 16:
        raise ValueError("numerical phase fitting is bounded to degree16")
    target_coefficients = cosine_chebyshev_coefficients(tau, degree)
    nodes = np.cos(np.pi*(np.arange(80)+0.5)/80)
    target = chebval(nodes, target_coefficients)
    rng = np.random.default_rng(17)
    for attempt in range(8):
        initial = rng.normal(0, 0.2, degree+1)
        initial[0] = initial[-1] = np.pi/4
        result = least_squares(
            lambda p: qsp_real_response(nodes, p)-target, initial,
            ftol=1e-13, gtol=1e-13, xtol=1e-13, max_nfev=4000,
        )
        residual = float(np.max(np.abs(qsp_real_response(nodes, result.x)-target)))
        if residual < 5e-12:
            return tuple(result.x), {
                "method": "scipy.optimize.least_squares", "seed": 17,
                "fit_nodes": 80, "successful_restart_zero_based": attempt,
                "function_evaluations_successful_restart": result.nfev,
                "node_residual": residual, "maximum_restarts": 8,
                "maximum_function_evaluations_per_restart": 4000,
                "ftol": 1e-13, "gtol": 1e-13, "xtol": 1e-13,
            }
    raise RuntimeError("bounded phase fit failed the 5e-12 node-residual criterion")


def _signed_projector_phase(circuit: QuantumCircuit, index: list[int], flag: int, phi: float):
    """exp(i phi (2Pi-I)Zflag); preserve both branch-global phases exactly."""
    circuit.rz(2*phi, flag)
    if not index:
        circuit.rz(-4*phi, flag)
        return
    circuit.x(index)
    circuit.append(RZGate(-4*phi).control(len(index), annotated=False), index+[flag])
    circuit.x(index)


def build_real_qsp_filter(terms: PauliTerms, phases: Sequence[float] = COSINE_PHASES) -> QuantumCircuit:
    """Real QSP polynomial block, data then label then one flag qubit.

    All degree-many PREPARE/SELECT/PREPARE-inverse calls are shared between
    the +phase and -phase branches. Postselect label=0 AND flag=0 to obtain
    Re(P(H/alpha)). Both flag H gates and every controlled phase are present.
    Arbitrary rotations are unsynthesized. No amplitude amplification included.
    """
    phases = _phases(phases)
    block = _hermitian_block(terms)
    n_system, n_index = block.metadata["n_system"], block.metadata["n_index"]
    circuit = QuantumCircuit(block.num_qubits+1, name="Real QSP filter")
    flag = block.num_qubits
    index = list(range(n_system, flag))
    circuit.h(flag)
    for i, phi in enumerate(reversed(phases)):
        _signed_projector_phase(circuit, index, flag, phi)
        if i < len(phases)-1:
            circuit.compose(block, qubits=list(range(flag)), inplace=True)
    circuit.h(flag)
    circuit.metadata = {
        **block.metadata, "degree": len(phases)-1, "n_flag": 1,
        "block_encoding_calls": len(phases)-1, "projector_phase_calls": len(phases),
        "phases": list(phases), "encoded_function": "real part of QSP polynomial",
        "phase_convention": "D(phi)=exp(i phi(2Pi_0-I)); V=D(phi0)B...BD(phid)",
        "postselection": "all label ancillas and flag measured zero",
        "amplitude_amplification": "not implemented",
        "status": "verified polynomial primitive; full complex time evolution planned",
    }
    return circuit


def build_qubitization_walk(terms: PauliTerms) -> QuantumCircuit:
    """Actual walk Q=(2Pi_0-I)B with eigenphases +/-acos(lambda/alpha)."""
    block = _hermitian_block(terms)
    circuit = block.copy()
    index = list(range(block.metadata["n_system"], block.num_qubits))
    if index:
        circuit.global_phase += np.pi
        circuit.x(index)
        if len(index) == 1:
            circuit.p(np.pi, index[0])
        else:
            circuit.mcp(np.pi, index[:-1], index[-1])
        circuit.x(index)
    circuit.name = "Qubitization walk"
    circuit.metadata = {**block.metadata, "walk": "(2Pi_0-I)B", "block_encoding_calls": 1}
    return circuit


def postselected_filter_state(circuit: QuantumCircuit, initial_state: np.ndarray) -> tuple[np.ndarray, float]:
    """Bounded ideal circuit simulation followed by zero-ancilla projection."""
    if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError(f"simulation is bounded to {MAX_STATEVECTOR_QUBITS} total qubits")
    n_system = circuit.metadata["n_system"]
    state = np.asarray(initial_state, dtype=complex)
    if state.shape != (2**n_system,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state dimension or entries invalid")
    if not np.isclose(np.vdot(state, state).real, 1, atol=1e-12, rtol=0):
        raise ValueError("initial_state must be normalized")
    padded = np.zeros(2**circuit.num_qubits, dtype=complex)
    padded[:len(state)] = state
    branch = Statevector(padded).evolve(circuit).data[:len(state)]
    probability = float(np.vdot(branch, branch).real)
    if probability <= 1e-30:
        raise ValueError("filter has zero numerical acceptance probability")
    return branch/math.sqrt(probability), probability
