# SPDX-License-Identifier: MIT
"""Bounded original real-QSP phase generation with an analytic Jacobian.

Signal convention: B(x)=[[x,sqrt(1-x*x)],[sqrt(1-x*x),-x]],
D(phi)=diag(exp(i phi),exp(-i phi)), V=D(phi0)B...BD(phid).
The phase fit targets Re(V[0,0]) and is numerical evidence, not an interval
coefficient, access, angle-rounding or rotation-synthesis certificate.

A palindromic parameterization of the usual W(x)=[[x,i*s],[i*s,x]] is
converted into the stated B convention. The conversion is derived explicitly
below; it avoids the rank-deficient all-zero initialization of unrestricted
B-convention phases. No upstream phase package or frozen phase table is used.
Original code is MIT; SciPy is BSD-3-Clause. QSP context: Gilyen et al. (2019),
DOI: 10.1145/3313276.3316366, and Low/Chuang (2017),
DOI: 10.1103/PhysRevLett.118.010501. This module performs bounded CPU algebra;
it constructs no quantum matrix gate and imports no Qiskit circuit simulator.
"""
from __future__ import annotations

import math
from numbers import Real
import time
from typing import Sequence

import numpy as np
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.optimize import least_squares

MAX_PHASE_DEGREE = 32
MAX_SIGNAL_POINTS = 16385
MAX_FIT_NODES = 256
MAX_RESTARTS = 8
MAX_FUNCTION_EVALUATIONS = 5000
MAX_FIT_SECONDS = 120.0
DEFAULT_GRID_POINTS = 4097
TARGET_MAGNITUDE_GUARD = 0.75
PHASE_CONVENTION = "D(phi)=exp(i phi Z); V=D(phi0)B...BD(phid); B=[[x,sqrt(1-x^2)],[sqrt(1-x^2),-x]]; target real V00"


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return int(value)


def _real(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    return float(value)


def _phase_values(phases: Sequence[float]) -> np.ndarray:
    values = tuple(phases)
    if not 1 <= len(values) <= MAX_PHASE_DEGREE+1:
        raise ValueError(f"phase count must be 1..{MAX_PHASE_DEGREE+1}")
    return np.array([_real(value, "phase") for value in values])


def _signals(x) -> tuple[np.ndarray, tuple[int, ...]]:
    try:
        values = np.asarray(x)
        if np.iscomplexobj(values):
            raise ValueError("signals must be real values in [-1,1]")
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("signals must be real values in [-1,1]") from None
    if values.size > MAX_SIGNAL_POINTS or not np.all(np.isfinite(values)) or np.any(abs(values) > 1):
        raise ValueError(f"signals require at most {MAX_SIGNAL_POINTS} finite values in [-1,1]")
    return values.reshape(-1), values.shape


def _signal_matrices(x: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.maximum(0.0, 1-x*x))
    return np.array([[x, s], [s, -x]]).transpose(2, 0, 1).astype(complex)


def _response(x: np.ndarray, phases: np.ndarray) -> np.ndarray:
    signal = _signal_matrices(x)
    matrix = np.broadcast_to(np.eye(2, dtype=complex), (len(x), 2, 2)).copy()
    for j, phase in enumerate(phases):
        matrix[:, :, 0] *= np.exp(1j*phase)
        matrix[:, :, 1] *= np.exp(-1j*phase)
        if j < len(phases)-1:
            matrix = matrix@signal
    return matrix[:, 0, 0].real


def _response_and_jacobian(x: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    signal = _signal_matrices(x)
    count = len(phases)
    identity = np.broadcast_to(np.eye(2, dtype=complex), (len(x), 2, 2)).copy()
    factors = []
    prefixes = [identity]
    for j, phase in enumerate(phases):
        diagonal = np.diag([np.exp(1j*phase), np.exp(-1j*phase)])
        factor = diagonal@signal if j < count-1 else np.broadcast_to(diagonal, identity.shape)
        factors.append(factor)
        prefixes.append(prefixes[-1]@factor)
    suffixes = [None]*(count+1)
    suffixes[count] = identity
    for j in reversed(range(count)):
        suffixes[j] = factors[j]@suffixes[j+1]
    jacobian = np.empty((len(x), count))
    for j, factor in enumerate(factors):
        derivative = factor.copy()
        derivative[:, 0, :] *= 1j
        derivative[:, 1, :] *= -1j
        jacobian[:, j] = (prefixes[j]@derivative@suffixes[j+1])[:, 0, 0].real
    return prefixes[-1][:, 0, 0].real, jacobian


def qsp_response_and_jacobian(x, phases: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Real signal response and exact-algebra phase derivative, floating values.

    The response has x's shape; the Jacobian appends one phase axis. The
    derivative follows the ordered matrix product without phase alignment.
    """
    values, shape = _signals(x)
    phases = _phase_values(phases)
    response, jacobian = _response_and_jacobian(values, phases)
    return response.reshape(shape), jacobian.reshape(shape+(len(phases),))


def real_qsp_response(x, phases: Sequence[float]) -> np.ndarray:
    """Evaluate only the bounded scalar response, with no circuit simulation."""
    values, shape = _signals(x)
    return _response(values, _phase_values(phases)).reshape(shape)


def _coefficients(coefficients: Sequence[float]) -> np.ndarray:
    values = tuple(coefficients)
    if not 1 <= len(values) <= MAX_PHASE_DEGREE+1:
        raise ValueError(f"coefficient count must be 1..{MAX_PHASE_DEGREE+1}; trailing zero degree padding is retained")
    result = np.array([_real(value, "coefficient") for value in values])
    parity = (len(result)-1)%2
    if np.any(result[1-parity::2] != 0):
        raise ValueError("coefficients must have the definite parity of their requested degree")
    return result


def symmetric_qsp_phases(parameters: Sequence[float], degree: int) -> np.ndarray:
    """Map reduced palindromic W phases to this module's real B convention.

    W=-i D(-pi/4) B D(3pi/4). Therefore the base conversion gives
    V_B=i**degree U_W. An additional first phase pi/2 for even degree
    makes Re(V_B)=(-1)**(floor(degree/2)+1) Im(U_W), for both parities.
    Each theta_j is repeated at its mirrored position; the even center is
    used once. This is a parameterization, not a phase-error certificate.
    """
    _integer(degree, "degree", 1, MAX_PHASE_DEGREE)
    values = tuple(parameters)
    if len(values) != degree//2+1:
        raise ValueError("symmetric parameter count must be floor(degree/2)+1")
    reduced = np.array([_real(v, "symmetric parameter") for v in values])
    mirror = np.minimum(np.arange(degree+1), degree-np.arange(degree+1))
    phases = reduced[mirror]
    phases[0] += -np.pi/4 + (np.pi/2 if degree%2 == 0 else 0.0)
    phases[1:-1] += np.pi/2
    phases[-1] += 3*np.pi/4
    return phases


class _SymmetricModel:
    def __init__(self, nodes: np.ndarray, degree: int, deadline: float):
        self.nodes, self.degree, self.deadline = nodes, degree, deadline
        self.parameters = self.response = self.jacobian = None

    def evaluate(self, parameters: np.ndarray):
        if time.perf_counter() > self.deadline:
            raise RuntimeError("bounded QSP phase fitting exceeded its elapsed-time limit")
        if self.parameters is None or not np.array_equal(self.parameters, parameters):
            phases = symmetric_qsp_phases(parameters, self.degree)
            self.response, full_jacobian = _response_and_jacobian(self.nodes, phases)
            self.jacobian = np.zeros((len(self.nodes), len(parameters)))
            for j in range(self.degree+1):
                self.jacobian[:, min(j, self.degree-j)] += full_jacobian[:, j]
            self.parameters = parameters.copy()
        return self.response, self.jacobian


def fit_real_qsp_phases(coefficients: Sequence[float], *, tolerance: float = 1e-10,
                        seed: int = 17, max_restarts: int = 4,
                        max_nfev: int = 1500, max_seconds: float = 30.0,
                        fit_nodes: int | None = None,
                        grid_points: int = DEFAULT_GRID_POINTS) -> tuple[tuple[float, ...], dict]:
    """Generate phases for a supplied real Chebyshev polynomial, fail closed.

    Array length sets degree; zero padding is never trimmed. Incompatible
    parity coefficients must be exactly zero. Intended targets have magnitude
    at most 1/2. A sampled 0.75 guard rejects clearly unsupported targets but is
    explicitly not a rigorous supremum test. Independent coefficient/interval
    validation belongs to the caller. All requested limits have fixed ceilings.

    The optimizer uses reduced symmetric phases and an analytic prefix/suffix
    Jacobian. First initialization is the linear Chebyshev solution; bounded
    seeded perturbations are fallback restarts. Constant even and linear odd
    targets (including zero targets) have direct phases. Accepted fitting must
    satisfy both independent dense-grid and fit-node residual checks. Neither
    check certifies coefficient arithmetic, ideal target truncation, phases,
    block access, compiled angles or Clifford+T synthesis.
    """
    coefficients = _coefficients(coefficients)
    degree = len(coefficients)-1
    tolerance = _real(tolerance, "tolerance")
    if not 1e-13 <= tolerance <= 1e-3:
        raise ValueError("tolerance must be in [1e-13,1e-3] for this bounded float fitter")
    seed = _integer(seed, "seed", 0, 2**32-1)
    max_restarts = _integer(max_restarts, "max_restarts", 1, MAX_RESTARTS)
    max_nfev = _integer(max_nfev, "max_nfev", 1, MAX_FUNCTION_EVALUATIONS)
    max_seconds = _real(max_seconds, "max_seconds")
    if not 0 < max_seconds <= MAX_FIT_SECONDS:
        raise ValueError(f"max_seconds must be positive and at most{MAX_FIT_SECONDS}")
    fit_nodes = max(64, 2*(degree+1)) if fit_nodes is None else _integer(fit_nodes, "fit_nodes", degree+1, MAX_FIT_NODES)
    grid_points = _integer(grid_points, "grid_points", max(33, degree+1), MAX_SIGNAL_POINTS)
    nodes = np.cos(np.pi*(np.arange(fit_nodes)+0.5)/fit_nodes)
    grid = np.linspace(-1, 1, grid_points)
    target, grid_target = chebval(nodes, coefficients), chebval(grid, coefficients)
    sampled_magnitude = float(max(np.max(abs(target)), np.max(abs(grid_target))))
    if sampled_magnitude > TARGET_MAGNITUDE_GUARD:
        raise ValueError("target exceeds the bounded fitter's sampled 0.75 magnitude guard")
    started = time.perf_counter()
    deadline = started+max_seconds
    method = "analytic constant/linear phases"
    evaluations = total_evaluations = 0
    restart = None
    parity = degree%2
    if np.count_nonzero(coefficients[parity+2:]) == 0:
        phases = np.zeros(degree+1)
        phases[0] = math.acos(float(coefficients[parity]))
    else:
        method = "scipy.optimize.least_squares; palindromic phases; analytic prefix/suffix Jacobian"
        sign = (-1)**(degree//2+1)
        initial = np.array([sign*coefficients[degree-2*j]/(1 if j == degree-j else 2)
                            for j in range(degree//2+1)])
        rng = np.random.default_rng(seed)
        phases = None
        for restart in range(max_restarts):
            guess = initial if restart == 0 else initial+rng.normal(0, 0.03*restart, len(initial))
            model = _SymmetricModel(nodes, degree, deadline)
            result = least_squares(lambda p: model.evaluate(p)[0]-target,
                                   guess, jac=lambda p: model.evaluate(p)[1],
                                   ftol=1e-13, gtol=1e-13, xtol=1e-13,
                                   max_nfev=max_nfev)
            evaluations = int(result.nfev)
            total_evaluations += evaluations
            candidate = symmetric_qsp_phases(result.x, degree)
            node_error = float(np.max(abs(_response(nodes, candidate)-target)))
            grid_error = float(np.max(abs(_response(grid, candidate)-grid_target)))
            if max(node_error, grid_error) <= tolerance:
                phases = candidate
                break
        if phases is None:
            raise RuntimeError(f"bounded QSP phase fitting failed residual tolerance {tolerance:g} after {max_restarts} restarts and {total_evaluations} evaluations")
    if time.perf_counter() > deadline:
        raise RuntimeError("bounded QSP phase fitting exceeded its elapsed-time limit")
    actual_nodes = _response(nodes, phases)
    actual_grid = _response(grid, phases)
    node_error = float(np.max(abs(actual_nodes-target)))
    grid_error = float(np.max(abs(actual_grid-grid_target)))
    if max(node_error, grid_error) > tolerance:
        raise RuntimeError("direct QSP phases failed independent numerical residual checks")
    reconstructed = chebfit(nodes, actual_nodes, degree)
    completed = time.perf_counter()
    if completed > deadline:
        raise RuntimeError("bounded QSP phase fitting exceeded its elapsed-time limit during final validation")
    report = {
        "degree": degree, "parity": parity, "method": method, "seed": seed,
        "target_chebyshev_coefficients": coefficients.tolist(),
        "fit_nodes": fit_nodes, "dense_grid_points": grid_points,
        "fit_tolerance": tolerance, "successful_restart_zero_based": restart,
        "function_evaluations": evaluations, "total_function_evaluations": total_evaluations,
        "maximum_restarts": max_restarts, "maximum_function_evaluations_per_restart": max_nfev,
        "elapsed_time_limit_seconds": max_seconds, "wall_seconds": completed-started,
        "node_max_residual": node_error, "grid_max_residual": grid_error,
        "numerically_reconstructed_coefficient_l1_discrepancy": float(np.sum(abs(reconstructed-coefficients))),
        "numerically_reconstructed_chebyshev_coefficients": reconstructed.tolist(),
        "sampled_target_magnitude": sampled_magnitude,
        "sampled_target_magnitude_guard": TARGET_MAGNITUDE_GUARD,
        "phase_convention": PHASE_CONVENTION,
        "initialization": "linear Chebyshev solution in reduced palindromic W phases, mapped to B convention",
        "jacobian": "analytic ordered-product prefix/suffix derivatives with mirrored parameter columns summed",
        "certified": False,
        "validation_scope": "floating scalar phase fit and sampled residuals only; independent analytic/interval coefficient validation required",
        "exclusions": ["target truncation and coefficient certificates", "phase/angle rounding certificates",
                       "Hamiltonian coefficient/access rounding", "circuit compilation and rotation synthesis", "physical faults"],
    }
    return tuple(float(p) for p in phases), report
