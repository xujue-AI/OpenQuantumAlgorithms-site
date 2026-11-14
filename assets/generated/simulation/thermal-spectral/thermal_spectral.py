# SPDX-License-Identifier: MIT
"""Retargetable thermal QSP from explicit Pauli access and a Bell input.

For a=beta*alpha/2, the even/odd real filters target
e^-a cosh(a*x)/2 and -e^-a sinh(a*x)/2. Their coherent real sum
encodes exp[-a(x+1)]/4. The scalar certificate concerns exact ideal
H/alpha access and the stored binary64 phases; implemented loading,
compilation, synthesis and instrument faults have separate contracts.
No Pauli-power aggregation, eigenvector loading or dense matrix gate is used.
Original MIT code; Qiskit2.6.2 Apache-2.0, SciPy1.18.1 BSD-3-Clause.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from decimal import Context, Decimal, localcontext, ROUND_CEILING, ROUND_FLOOR
from fractions import Fraction
import math
from numbers import Real
from typing import Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector, partial_trace
from scipy.linalg import expm
from scipy.special import gammaln, ive

if __package__:
    from .retargetable_qsp import build_two_parity_filter, build_hermitian_pauli_access
    from .lcu import postselection_attempts
else:
    from retargetable_qsp import build_two_parity_filter, build_hermitian_pauli_access
    from lcu import postselection_attempts

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_SYSTEM_QUBITS = 8
MAX_TERMS = 32
MAX_TOTAL_QUBITS = 24
MAX_PHASE_DEGREE = 32
MAX_PARAMETER = 16.0
MAX_SOURCE_OPERATIONS = 50_000
MAX_PRIMITIVE_OPERATIONS = 250_000
MAX_STATEVECTOR_QUBITS = 12
MAX_OPERATOR_QUBITS = 8
MAX_REFERENCE_SYSTEM_QUBITS = 6
MIN_FILTER_ERROR = 1e-10
MAX_FILTER_ERROR = 0.1
MAX_WORKLOAD_ATTEMPTS = 10**12


@dataclass(frozen=True)
class ThermalPhasePlan:
    parameter: float
    requested_filter_error: float
    requested_gibbs_error: float | None
    effective_filter_target: float
    even_degree: int
    odd_degree: int
    even_phases: tuple[float, ...]
    odd_phases: tuple[float, ...]
    even_coefficients: tuple[float, ...]
    odd_coefficients: tuple[float, ...]
    even_fit: dict
    odd_fit: dict
    even_certificate: dict
    odd_certificate: dict
    ideal_encoded_polynomial_error_upper: float
    ideal_decoded_filter_error_upper: float
    ideal_shifted_filter_gibbs_error_upper: float
    certificate_scope: str

    def as_dict(self) -> dict:
        return asdict(self)


def _real(value, name: str, *, nonnegative=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return float(value)


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")
    return value


def _float_upper(value: Fraction | Decimal) -> float:
    if value == 0:
        return 0.0
    result = float(value)
    exact_result = Fraction.from_float(result) if isinstance(value, Fraction) else Decimal.from_float(result)
    return math.nextafter(result, math.inf) if exact_result < value else result


def _float_lower(value: Decimal) -> float:
    result = float(value)
    return math.nextafter(result, -math.inf) if Decimal.from_float(result) > value else result


def _fraction_record(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def spin_chain_terms(n_system: int = 2, J: float = 0.6, h: float = 0.35) -> tuple[tuple[float, str], ...]:
    """Fixed-local-coupling open-chain inputs, without a matrix allocation."""
    _integer(n_system, "n_system", 1, MAX_SYSTEM_QUBITS)
    J, h = _real(J, "J"), _real(h, "h")
    result = []
    for q in range(n_system-1):
        label = ["I"]*n_system
        label[-1-q] = label[-2-q] = "Z"
        if J:
            result.append((J, "".join(label)))
    for q in range(n_system):
        label = ["I"]*n_system
        label[-1-q] = "X"
        if h:
            result.append((h, "".join(label)))
    if not result:
        raise ValueError("the explicit access requires a nonzero Hamiltonian")
    return tuple(result)


def _model(terms, beta: float):
    beta = _real(beta, "beta", nonnegative=True)
    values = tuple(terms)
    if not 1 <= len(values) <= MAX_TERMS:
        raise ValueError(f"explicit input requires1..{MAX_TERMS} Pauli terms")
    result, n = [], None
    for coefficient, label in values:
        coefficient = _real(coefficient, "Pauli coefficient")
        if not isinstance(label, str) or not label or set(label)-set("IXYZ"):
            raise ValueError("Pauli labels must be nonempty IXYZ strings")
        n = len(label) if n is None else n
        if len(label) != n:
            raise ValueError("Pauli labels must have the same width")
        if coefficient:
            result.append((coefficient, label))
    if not result or n > MAX_SYSTEM_QUBITS:
        raise ValueError("nonzero Hamiltonian or thermal generator width invalid")
    alpha = sum((Fraction.from_float(abs(c)) for c, _ in result), Fraction())
    try:
        alpha_float = float(alpha)
    except OverflowError:
        raise ValueError("Hamiltonian coefficient normalization is not representable") from None
    if not math.isfinite(alpha_float) or alpha <= 0:
        raise ValueError("Hamiltonian coefficient normalization is not representable")
    a_exact = Fraction.from_float(beta)*alpha/2
    if a_exact > Fraction.from_float(MAX_PARAMETER):
        raise ValueError("beta times exact coefficient normalization exceeds parameter cap")
    return tuple(result), beta, n, alpha, a_exact


def thermal_coefficients(parameter: float, even_degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Floating scaled-Bessel fit targets; certificate encloses the target independently."""
    a = _real(parameter, "parameter", nonnegative=True)
    _integer(even_degree, "even_degree", 0, MAX_PHASE_DEGREE-1)
    if a > MAX_PARAMETER or even_degree % 2:
        raise ValueError("thermal parameter or even phase degree invalid")
    even, odd = np.zeros(even_degree+1), np.zeros(even_degree+2)
    even[0] = ive(0, a)/2
    for k in range(2, even_degree+1, 2):
        even[k] = ive(k, a)
    for k in range(1, even_degree+2, 2):
        odd[k] = -ive(k, a)
    return even, odd


def _tail_preview(parameter: float, degree: int) -> float:
    """Floating evaluation of an analytic omitted-parity scaled-I tail bound."""
    if parameter == 0:
        return 0.0
    b, k = parameter/2, degree+2
    ratio = b*b/((k+1)*(k+2))
    if ratio >= 1:
        return math.inf
    return math.exp(-parameter+k*math.log(b)-gammaln(k+1)+b*b/(k+1))/(1-ratio)


def choose_thermal_degree(parameter: float, filter_error: float = 1e-6) -> int:
    a = _real(parameter, "parameter", nonnegative=True)
    error = _real(filter_error, "filter_error")
    if a > MAX_PARAMETER or not MIN_FILTER_ERROR <= error <= MAX_FILTER_ERROR:
        raise ValueError("thermal parameter or requested filter-error cap invalid")
    for degree in range(0, MAX_PHASE_DEGREE, 2):
        if max(_tail_preview(a, degree), _tail_preview(a, degree+1)) <= error/8:
            return degree
    raise ValueError("requested thermal truncation exceeds phase-degree support")


def ideal_conditioning_bounds(parameter_exact: Fraction, encoded_error_upper: Fraction,
                              *, precision: int = 80) -> dict:
    """Outward Decimal bounds for exact ideal signal access; no loading certificate.

    Decimal exp is correctly rounded. One adjacent representable Decimal on
    each side encloses it; directed division/squaring then preserves bounds.
    The target Bell branch has norm at least exp(-2a)/4.
    """
    _integer(precision, "precision", 40, 120)
    if (not isinstance(parameter_exact, Fraction) or not isinstance(encoded_error_upper, Fraction)
            or parameter_exact < 0 or parameter_exact > Fraction.from_float(MAX_PARAMETER)
            or encoded_error_upper < 0 or encoded_error_upper > 1):
        raise ValueError("conditioning requires nonnegative parameter/error")
    with localcontext(Context(prec=precision, rounding=ROUND_CEILING)) as ctx:
        a_upper = Decimal(parameter_exact.numerator)/Decimal(parameter_exact.denominator)
        delta = Decimal(encoded_error_upper.numerator)/Decimal(encoded_error_upper.denominator)
        ctx.rounding = ROUND_FLOOR
        x_lower = ctx.multiply(Decimal(-2), a_upper)
        e_lower = x_lower.exp().next_minus()
        ctx.rounding = ROUND_FLOOR
        m_lower = e_lower/Decimal(4)
        gap = m_lower-delta
        p_lower = gap*gap if gap > 0 else Decimal(0)
        ctx.rounding = ROUND_CEILING
        distance = min(Decimal(1), 2*delta/m_lower)
        return {
            "ideal_target_Bell_branch_amplitude_lower": _float_lower(m_lower),
            "ideal_target_Bell_branch_amplitude_lower_decimal": str(m_lower),
            "ideal_approximate_filter_acceptance_lower": max(0.0, _float_lower(p_lower)),
            "ideal_approximate_filter_acceptance_lower_decimal": str(p_lower),
            "ideal_Gibbs_trace_distance_upper": min(1.0, _float_upper(distance)),
            "ideal_Gibbs_trace_distance_upper_decimal": str(distance),
            "precision": precision, "outward_evaluation": True,
            "scope": "exact declared logical H/alpha_exact signal and stored exact angles; this conditioning evaluation includes the supplied error bound, with nested-control realization/compilation/synthesis/instrument faults separate",
        }


def generate_thermal_phases(parameter: float, filter_error: float = 1e-6,
                            gibbs_error: float | None = 1e-4, *, seed: int = 53) -> ThermalPhasePlan:
    """Actually fit and interval-validate each requested thermal parameter."""
    if __package__:
        from .qsp_phase_generation import fit_real_qsp_phases
        from .qsp_certificate import certify_real_qsp
    else:
        from qsp_phase_generation import fit_real_qsp_phases
        from qsp_certificate import certify_real_qsp
    a = _real(parameter, "parameter", nonnegative=True)
    error = _real(filter_error, "filter_error")
    if not MIN_FILTER_ERROR <= error <= MAX_FILTER_ERROR:
        raise ValueError("requested filter-error cap invalid")
    if gibbs_error is not None:
        gibbs_error = _real(gibbs_error, "gibbs_error")
        if not 1e-6 <= gibbs_error <= 0.1:
            raise ValueError("requested Gibbs trace-distance target must be1e-6..0.1 or None")
        error = min(error, gibbs_error*math.exp(-2*a)/4)
    degree = choose_thermal_degree(a, error)
    even, odd = thermal_coefficients(a, degree)
    fit_tolerance = min(1e-10, error/64)
    pe, fe = fit_real_qsp_phases(even, tolerance=fit_tolerance, seed=seed)
    po, fo = fit_real_qsp_phases(odd, tolerance=fit_tolerance, seed=seed+1)
    pe = tuple(math.remainder(p, 2*math.pi) for p in pe)
    po = tuple(math.remainder(p, 2*math.pi) for p in po)
    ce = certify_real_qsp(pe, target_kind="thermal_even", parameter=a)
    co = certify_real_qsp(po, target_kind="thermal_odd", parameter=a)
    delta = (Fraction.from_float(ce["uniform_error_upper"])+Fraction.from_float(co["uniform_error_upper"]))/2
    decoded = _float_upper(4*delta)
    bounds = ideal_conditioning_bounds(Fraction.from_float(a), delta)
    if decoded > error:
        raise ValueError("generated thermal phases fail the rigorous decoded-filter allowance")
    if gibbs_error is not None and bounds["ideal_Gibbs_trace_distance_upper"] > gibbs_error:
        raise ValueError("generated thermal phases fail the ideal-access Gibbs allowance")
    return ThermalPhasePlan(a, filter_error, gibbs_error, error, degree, degree+1,
                            tuple(pe), tuple(po), tuple(even), tuple(odd), fe, fo, ce, co,
                            _float_upper(delta), decoded, bounds["ideal_Gibbs_trace_distance_upper"],
                            bounds["scope"])


def audit_thermal_phase_plan(plan: ThermalPhasePlan) -> ThermalPhasePlan:
    """Revalidate actual supplied phase values rather than trusting saved report fields."""
    if not isinstance(plan, ThermalPhasePlan):
        raise ValueError("plan must be a ThermalPhasePlan")
    if __package__:
        from .qsp_certificate import certify_real_qsp
    else:
        from qsp_certificate import certify_real_qsp
    a = _real(plan.parameter, "plan parameter", nonnegative=True)
    error = _real(plan.effective_filter_target, "plan effective filter target")
    requested_filter_error = _real(plan.requested_filter_error, "saved filter error")
    if not MIN_FILTER_ERROR <= requested_filter_error <= MAX_FILTER_ERROR or error > requested_filter_error:
        raise ValueError("saved effective allowance exceeds requested filter error")
    if a > MAX_PARAMETER or not MIN_FILTER_ERROR <= error <= MAX_FILTER_ERROR:
        raise ValueError("saved plan parameter/error exceeds the bounded domain")
    for phases, degree, parity in ((plan.even_phases, plan.even_degree, 0), (plan.odd_phases, plan.odd_degree, 1)):
        _integer(degree, "saved phase degree", 0, MAX_PHASE_DEGREE)
        if degree % 2 != parity or len(phases) != degree+1:
            raise ValueError("saved phase degree/parity does not match the stored values")
        if any(abs(_real(p, "saved phase")) > math.pi for p in phases):
            raise ValueError("saved phases must be wrapped into[-pi,pi]")
    if plan.odd_degree != plan.even_degree+1:
        raise ValueError("saved parity degrees must be adjacent")
    ce = certify_real_qsp(plan.even_phases, target_kind="thermal_even", parameter=a)
    co = certify_real_qsp(plan.odd_phases, target_kind="thermal_odd", parameter=a)
    delta = (Fraction.from_float(ce["uniform_error_upper"])+Fraction.from_float(co["uniform_error_upper"]))/2
    bounds = ideal_conditioning_bounds(Fraction.from_float(a), delta)
    if _float_upper(4*delta) > error:
        raise ValueError("saved phases fail the interval decoded-filter allowance")
    if plan.requested_gibbs_error is not None:
        gibbs_error = _real(plan.requested_gibbs_error, "saved Gibbs error")
        if not 1e-6 <= gibbs_error <= 0.1 or bounds["ideal_Gibbs_trace_distance_upper"] > gibbs_error:
            raise ValueError("saved phases fail the interval ideal Gibbs allowance")
    return replace(plan, even_certificate=ce, odd_certificate=co,
                   ideal_encoded_polynomial_error_upper=_float_upper(delta),
                   ideal_decoded_filter_error_upper=_float_upper(4*delta),
                   ideal_shifted_filter_gibbs_error_upper=bounds["ideal_Gibbs_trace_distance_upper"])


def thermal_error_contract(terms, beta: float, plan: ThermalPhasePlan) -> dict:
    """Exact binary-input parameter rounding plus explicitly excluded access errors."""
    terms, beta, n, alpha, a_exact = _model(terms, beta)
    a_float = Fraction.from_float(plan.parameter)
    if float(a_exact) != plan.parameter:
        raise ValueError("thermal phase plan does not match requested beta and Hamiltonian")
    parameter_error = abs(a_float-a_exact)
    polynomial = (Fraction.from_float(plan.even_certificate["uniform_error_upper"])+
                  Fraction.from_float(plan.odd_certificate["uniform_error_upper"]))/2
    combined = polynomial+parameter_error/2
    bounds = ideal_conditioning_bounds(a_exact, combined)
    if plan.requested_gibbs_error is not None and bounds["ideal_Gibbs_trace_distance_upper"] > plan.requested_gibbs_error:
        raise ValueError("parameter rounding exceeds requested ideal Gibbs allowance")
    if __package__:
        from .qsp_certificate import certify_prepared_pauli_weights
    else:
        from qsp_certificate import certify_prepared_pauli_weights
    preparation, _, _ = build_hermitian_pauli_access(terms, return_components=True)
    loading = certify_prepared_pauli_weights(terms, preparation)
    access_error = a_exact*Fraction.from_float(loading["weight_l1_error_upper"])/4
    source_error = combined+access_error
    if 4*source_error > Fraction.from_float(float(plan.requested_filter_error)):
        raise ValueError("certified loading/parameter error exceeds requested declared-logical filter allowance")
    source_bounds = ideal_conditioning_bounds(a_exact, source_error)
    if plan.requested_gibbs_error is not None and source_bounds["ideal_Gibbs_trace_distance_upper"] > plan.requested_gibbs_error:
        raise ValueError("certified loading/parameter error exceeds the requested precompiler Gibbs allowance")
    return {
        "exact_binary_input_alpha": _fraction_record(alpha), "alpha_float_for_display": float(alpha),
        "beta": beta, "exact_binary_input_parameter": _fraction_record(a_exact),
        "generated_phase_parameter": plan.parameter,
        "parameter_rounding_absolute_upper": _float_upper(parameter_error),
        "encoded_filter_parameter_rounding_error_upper": _float_upper(parameter_error/2),
        "ideal_encoded_total_error_upper": _float_upper(combined),
        "ideal_decoded_total_error_upper": _float_upper(4*combined),
        **bounds,
        "precompiler_source_encoded_total_error_upper": _float_upper(source_error),
        "precompiler_source_decoded_total_error_upper": _float_upper(4*source_error),
        "precompiler_source_acceptance_lower": source_bounds["ideal_approximate_filter_acceptance_lower"],
        "precompiler_source_acceptance_lower_decimal": source_bounds["ideal_approximate_filter_acceptance_lower_decimal"],
        "precompiler_source_Gibbs_trace_distance_upper": source_bounds["ideal_Gibbs_trace_distance_upper"],
        "precompiler_source_Gibbs_trace_distance_upper_decimal": source_bounds["ideal_Gibbs_trace_distance_upper_decimal"],
        "loading_weight_certificate": loading,
        "encoded_access_loading_error_upper": _float_upper(access_error),
        "normalization_shift": "F=exp[-beta*(H+alpha_exact I)/2], encoded F/4; positive scalar shift cancels in Gibbs density",
        "spectral_domain": "H/alpha_exact in[-1,1] for exact binary64 input coefficients; alpha_exact is an exact Fraction sum",
        "implemented_loading_rounding_certified": True,
        "implemented_access_contract": "The interval weight certificate encloses declared-logical exact RY/controlled-RY probabilities from stored angles and padded labels. Exact signed SELECT makes Bhat Hermitian/involutory in that logical model. Duhamel for shifted positive exponential contractions gives decoded access error<=a_exact*weight_l1. SDK nested-control/MCX realization, flat compiler/synthesis and instruments remain separate.",
        "phase_arithmetic": "stored phases are binary64 and are treated as exact real angles by the coefficient certificate; 2phi and4phi are exact binary powers of two in this bounded domain; nested-decomposition and compiler-created angles are separate",
        "precompiler_source_scope": "rigorous coefficient/weight/parameter bound for exact declared logical Cliffords, controlled gates and stored angles; Bell H/CX gates are exact in this model. This does not certify Qiskit nested controlled-gate/MCX definitions, flat compilation or fault-tolerant synthesis",
        "coherent_synthesis": "not performed; require additional norm bounds for compiler transformations and fault-tolerant synthesis before physical/postselection projections",
        "instrument_contract": "initial resets, success measurements, unconditioned reference discard/reset and final readout are outside the coherent scalar certificate",
        "input_representation": "finite exact binary64 Pauli coefficient list; original physical-model representation error excluded",
    }


def build_thermal_spectral_filter(terms=None, beta: float = 1.25, *,
                                  filter_error: float = 1e-6, gibbs_error: float | None = 1e-4,
                                  plan: ThermalPhasePlan | None = None) -> QuantumCircuit:
    """Actual Bell preparation and real two-parity QSP, no coefficient aggregation."""
    terms = spin_chain_terms(2) if terms is None else terms
    model, beta, n, alpha, a_exact = _model(terms, beta)
    requested_filter_error = _real(filter_error, "filter_error")
    if not MIN_FILTER_ERROR <= requested_filter_error <= MAX_FILTER_ERROR:
        raise ValueError("requested filter-error cap invalid")
    if gibbs_error is not None:
        gibbs_error = _real(gibbs_error, "gibbs_error")
        if not 1e-6 <= gibbs_error <= 0.1:
            raise ValueError("requested Gibbs trace-distance target must be1e-6..0.1 or None")
    plan = generate_thermal_phases(float(a_exact), filter_error, gibbs_error) if plan is None else audit_thermal_phase_plan(plan)
    if plan.requested_filter_error > requested_filter_error:
        raise ValueError("supplied phase plan has a looser filter allowance than the caller requested")
    if gibbs_error is not None and (plan.requested_gibbs_error is None or plan.requested_gibbs_error > gibbs_error):
        raise ValueError("supplied phase plan has a looser Gibbs allowance than the caller requested")
    contract = thermal_error_contract(model, beta, plan)
    access = build_two_parity_filter(model, plan.even_phases, plan.odd_phases, odd_branch_phase=0.0)
    total = access.num_qubits+n
    if total > MAX_TOTAL_QUBITS or access.size()+2*n > MAX_SOURCE_OPERATIONS:
        raise ValueError("thermal source materialization exceeds width/operation cap")
    core = QuantumCircuit(total, name="Thermal spectral QSP preparation")
    core.h(range(n))
    for q in range(n):
        core.cx(q, n+q)
    core.compose(access, qubits=list(range(n))+list(range(2*n, total)), inplace=True)
    core.metadata = {
        **access.metadata, "n_reference": n, "beta": beta, "parameter": plan.parameter,
        "Hamiltonian_terms": model, "n_success_ancillas": total-2*n,
        "success_qubits": list(range(2*n, total)), "phase_plan": plan.as_dict(),
        "error_contract": contract, "normalization_exact": _fraction_record(alpha),
        "encoded_operator": "polynomial real-parity filter; target exp[-beta*(H+alpha_exact I)/2]/4, with separate declared-logical bound and numerical SDK realization checks",
        "output": "conditional system-reference purification before final system readout",
        "classical_preprocessing": "scaled-Bessel coefficients and bounded actual phase fit; no Hamiltonian-word or Pauli-power aggregation",
        "initialization": "coherent core assumes all wires zero; complete trial explicitly resets them",
        "amplitude_amplification": "not included", "rotations_synthesized": False,
    }
    return core


def build_thermal_spectral_trial(core: QuantumCircuit, measurement: str = "Z") -> QuantumCircuit:
    """Complete reset/preparation/success/readout schedule, charging every rejection."""
    if measurement not in {"X", "Y", "Z"}:
        raise ValueError("measurement must be X, Y or Z on every system wire")
    n, success = core.metadata["n_system"], core.metadata["success_qubits"]
    if success != list(range(2*n, core.num_qubits)) or core.num_clbits:
        raise ValueError("expected coherent thermal source with final success register")
    circuit = QuantumCircuit(core.num_qubits, len(success)+n, name=f"Thermal spectral {measurement} trial")
    circuit.reset(range(core.num_qubits))
    circuit.compose(core, inplace=True)
    circuit.measure(success, range(len(success)))
    circuit.reset(range(n, 2*n))
    if measurement == "Y":
        circuit.sdg(range(n))
    if measurement != "Z":
        circuit.h(range(n))
    circuit.measure(range(n), range(len(success), len(success)+n))
    circuit.metadata = {
        **core.metadata, "measurement_basis": measurement,
        "initial_reset_instructions": core.num_qubits, "reference_reset_instructions": n,
        "success_measurement_instructions": len(success), "system_measurement_instructions": n,
        "output": "accepted measured thermal trial; reference reset without conditioning",
        "schedule": "all outcomes run the full circuit; accept zero success bits and keep system bits",
    }
    return circuit


def lower_thermal_spectral(circuit: QuantumCircuit) -> QuantumCircuit:
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS, optimization_level=1,
                         seed_transpiler=7, qubits_initially_zero=False)
    if compiled.size() > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("thermal spectral primitive export cap exceeded")
    unsupported = set(compiled.count_ops())-set(ROTATION_BASIS)-{"reset", "measure", "barrier"}
    if unsupported:
        raise ValueError(f"unresolved primitive operations: {unsupported}")
    return compiled


def _pauli(label: str) -> np.ndarray:
    matrices = {"I": np.eye(2), "X": np.array([[0,1],[1,0]]),
                "Y": np.array([[0,-1j],[1j,0]]), "Z": np.diag([1,-1])}
    result = np.array([[1+0j]])
    for letter in label:
        result = np.kron(result, matrices[letter])
    return result


def thermal_spectral_check(core: QuantumCircuit, *, operator_check: bool = True) -> dict:
    """Bounded actual branch/Gibbs checks; matrix references never construct gates."""
    if core.num_qubits > MAX_STATEVECTOR_QUBITS:
        raise ValueError("thermal statevector validation exceeds12 total wires")
    n = core.metadata["n_system"]
    if n > MAX_REFERENCE_SYSTEM_QUBITS:
        raise ValueError("dense thermal reference exceeds system cap")
    model, beta, _, alpha, _ = _model(core.metadata["Hamiltonian_terms"], core.metadata["beta"])
    dimension = 2**n
    # Scale exact binary-input coefficients before floating matrix formation.
    # H+alpha*I can overflow even when beta*alpha and the target are bounded.
    beta_fraction = Fraction.from_float(beta)
    scaled_h = sum((float(beta_fraction*Fraction.from_float(c))*_pauli(p) for c,p in model),
                   np.zeros((dimension, dimension), dtype=complex))
    normalized_h = sum((float(Fraction.from_float(c)/alpha)*_pauli(p) for c,p in model),
                       np.zeros((dimension, dimension), dtype=complex))
    exponential = expm(-scaled_h)
    gibbs = exponential/np.trace(exponential)
    shifted = expm(-float(beta_fraction*alpha/2)*(normalized_h+np.eye(dimension)))
    reference = shifted.reshape(-1, order="F")/(4*math.sqrt(dimension))
    branch = Statevector.from_instruction(core).data[:2**(2*n)]
    probability = float(np.vdot(branch, branch).real)
    if not 1e-28 < probability <= 1+1e-12:
        raise ValueError("thermal conditional state has zero or invalid numerical acceptance")
    density = partial_trace(Statevector(branch/math.sqrt(probability)), range(n,2*n)).data
    difference = density-gibbs
    # Undo the known Bell preparation before the provided core, then extract
    # its actual data-input/zero-reference/zero-success block. Rebuilding an
    # access from metadata would miss modifications to the supplied circuit.
    block_error = None
    if operator_check:
        if core.num_qubits-n > MAX_OPERATOR_QUBITS:
            raise ValueError("thermal filter Operator check exceeds8 total wires; useoperator_check=False")
        actual_access = QuantumCircuit(core.num_qubits)
        for q in range(n):
            actual_access.cx(q,n+q)
        actual_access.h(range(n))
        actual_access.compose(core,inplace=True)
        columns = []
        for basis in range(dimension):
            initial = np.zeros(2**core.num_qubits,dtype=complex)
            initial[basis] = 1
            columns.append(Statevector(initial).evolve(actual_access).data[:dimension])
        block = np.column_stack(columns)
        block_error = float(np.linalg.norm(block-shifted/4,2))
    actual_energy = math.fsum(c*float(np.trace(density@_pauli(p)).real) for c,p in model)
    exact_energy = math.fsum(c*float(np.trace(gibbs@_pauli(p)).real) for c,p in model)
    return {
        "n_system": n, "logical_qubits": core.num_qubits, "beta": beta,
        "parameter": core.metadata["parameter"], "even_degree": core.metadata["even_degree"],
        "odd_degree": core.metadata["odd_degree"], "operator_validation_performed": operator_check,
        "operator_validation_scope": "all data basis inputs of the supplied core after coherent inverse Bell preparation, with zero reference/success projection; bounded filter-block check, not full total-wire unitary proof" if operator_check else "unperformed",
        "actual_encoded_filter_operator_error": block_error,
        "actual_Bell_branch_vector_error": float(np.linalg.norm(branch-reference)),
        "actual_acceptance_probability": min(1.0,probability),
        "exact_shifted_filter_acceptance_probability": float(np.vdot(reference,reference).real),
        "gibbs_trace_distance": float(np.linalg.svd(difference,compute_uv=False).sum()/2),
        "gibbs_max_entry_error": float(np.max(abs(difference))),
        "trace_real": float(np.trace(density).real), "minimum_eigenvalue": float(np.linalg.eigvalsh(density).min()),
        "actual_energy": actual_energy, "exact_gibbs_energy": exact_energy,
        "energy_error": abs(actual_energy-exact_energy),
        "actual_density_real": density.real.tolist(), "actual_density_imag": density.imag.tolist(),
        "complete_trials_for_one_accepted_measured_output_99_percent": postselection_attempts(min(1.0,probability),.01),
        "state_stage": "conditional state before final system readout; reference traced without conditioning",
        "actual_circuit_error_certificate": False,
        "scope": "numerical SDK gate/branch/Gibbs check; declared-logical coefficient/weight/parameter certificate is separate from nested-control realization, compilation, synthesis and instrument faults",
    }


def energy_workload(probability: float, n_system: int = 2, J: float = 0.6, h: float = 0.35,
                    statistical_error: float = .03, statistical_failure: float = .005,
                    acceptance_failure: float = .005) -> dict:
    """Grouped open-chain energy proposal; accepted trials may have correlated bits."""
    _integer(n_system, "n_system", 1, MAX_SYSTEM_QUBITS)
    p = _real(probability, "probability")
    J, h = _real(J, "J"), _real(h, "h")
    error = _real(statistical_error, "statistical_error")
    sf, af = _real(statistical_failure, "statistical_failure"), _real(acceptance_failure,"acceptance_failure")
    if not 0 < p <= 1 or error <= 0 or not 0 < sf < 1 or not 0 < af < 1 or sf+af >= 1:
        raise ValueError("invalid sampling/acceptance budget")
    widths = {axis:width for axis,width in (("X",2*n_system*abs(h)),("Z",2*(n_system-1)*abs(J))) if width}
    if not widths:
        return {"groups":{},"total_accepted_trials":0,"complete_attempt_cap":0,"sampling_performed":False}
    groups = {}
    for axis,width in widths.items():
        log_count = 2*math.log(width)+math.log(math.log(2*len(widths)/sf))-math.log(2)-2*math.log(error/len(widths))
        if log_count > math.log(MAX_WORKLOAD_ATTEMPTS):
            raise ValueError("accepted-shot budget exceeds explicit cap")
        groups[axis] = {"score_width":width,"accepted_trials":max(1,math.ceil(math.exp(log_count))),
                        "allocated_absolute_error":error/len(widths),"statistical_failure":sf/len(widths)}
    total = sum(g["accepted_trials"] for g in groups.values())
    retries = postselection_attempts(p,af/total)
    if total*retries > MAX_WORKLOAD_ATTEMPTS:
        raise ValueError("complete accepted-energy workload exceeds explicit cap")
    return {"groups":groups,"total_accepted_trials":total,"retry_cap_per_slot":retries,
            "complete_attempt_cap":total*retries,"expected_complete_attempts":total/p,
            "statistical_failure":sf,"acceptance_failure":af,"sampling_performed":False,
            "acceptance_model":"numerical actual ideal-circuit probability for fixed model; not a hardware certificate",
            "protocol":"fresh complete attempts for every accepted slot; rejected trials charged fully",
            "physical_failure_allowance":None}
