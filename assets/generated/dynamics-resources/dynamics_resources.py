# SPDX-License-Identifier: MIT
"""Explicit error/workload contracts for bounded static dynamics resource studies.

The spin-chain commutator coefficients are exact rational l1 bounds. QDK's
synthesis scalar is only an empirical cost model, not a channel certificate.
Postselection claims are explicitly conditional on that scalar bounding total
variation of the complete final joint (data, ancillary outcome) distribution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from numbers import Real

from qiskit import QuantumCircuit

MAX_SYSTEM_QUBITS = 64
MAX_STEPS = 20_000
MAX_ATTEMPTS_PER_SLOT = 1_000_000
MAX_PHASE_FIT_SIGNAL_VALUES = 10_001

BOUND_SOURCE = {
    "paper": "Theory of Trotter Error with Commutator Scaling",
    "doi": "10.1103/PhysRevX.11.011020",
    "arxiv": "https://arxiv.org/abs/1912.08854v3",
    "PDF_sha256": "c76e2980dee799253ac3a5fa6176ac9005f78c844c1243dc2e579433137c7965",
    "equations": "Sec. 5.1, Eqs. (143) and (150), PDF pages 38/39, arXiv v3 (3 February 2021)",
    "step_composition": "Unitary telescoping over r equal-duration steps",
}
DISTANCE_DOMAIN = tuple(range(3, 42, 2))

# Original numerical phase solutions. They retarget the same degree-14/15
# construction to fixed local couplings and t=1; they are not certificates.
RETARGETED_PHASES = {
    2: {
        "tau": 2.4, "constructor_tolerance": 1e-8,
        "cosine": (1.5095453382308335, 0.12450914609070969, 0.08900796123225115, -0.26519182587457535, -0.8483246360474818, -0.5493325876775899, -0.04734630889933234, 0.38534336252098517, 0.11799332499378123, -0.15521598755842278, -0.1612524924308159, 0.0337550177311136, 0.11150239513024923, 0.8767356241867994, 0.7266747184070741),
        "sine": (0.5217097950176859, -0.4696953574740585, 0.05088442937698719, 0.05164822160080615, -0.06595061142587275, 0.597493180611017, 0.17502649397688486, 0.6147523942101356, -0.9701312686621334, -0.09461966678507572, -0.4518240147390462, 0.4832706738096487, 0.007275549814451783, 0.3959545546102542, -0.02453438642710989, 0.40503050311070815),
    },
    4: {
        "tau": 5.8, "constructor_tolerance": 1e-5,
        "cosine": (0.7691511669005053, 0.34078985824366403, 0.25264716543713367, -0.8090070326686101, -0.45069238689313723, -0.6334959578119661, -1.0411031835640414, 0.5136937650523742, 0.6776190730773453, -0.18470749012375232, -0.5209251892661454, 0.1930183661597749, 0.4831245748611812, 0.6121707253429074, 0.909839425867432),
        "sine": (1.0932060122023792, -0.4305707303269108, 0.09396540278016041, 0.49548975270048334, -0.34157201125313846, 0.9479366047160769, 0.3983175608314927, 1.5785869320125472, -1.0442244133337926, -0.5944451384360948, -1.443496019688043, 0.6354207450852886, 0.18442694227476383, 0.39838183009243405, -0.3766327294029219, 0.21044853994485918),
    },
}


def qsp_response_jacobian(values, phases):
    """Numerical real Hermitian-signal response and analytic phase derivative."""
    import numpy as np
    x = np.asarray(values, dtype=float)
    phases = np.asarray(phases, dtype=float)
    if (x.ndim != 1 or not 1 <= x.size <= MAX_PHASE_FIT_SIGNAL_VALUES
            or phases.ndim != 1 or not 1 <= len(phases) <= 18):
        raise ValueError("Bounded one-dimensional values and phase arrays required")
    if not np.all(np.isfinite(x)) or np.any(np.abs(x) > 1) or not np.all(np.isfinite(phases)):
        raise ValueError("Finite phases and signal values in [-1,1] required")
    degree = len(phases) - 1
    sine = np.sqrt(np.maximum(0, 1 - x*x))
    signal = np.array([[x, sine], [sine, -x]]).transpose(2, 0, 1).astype(complex)
    right = [None] * (degree + 1)
    right[degree] = np.tile(np.eye(2, dtype=complex), (len(x), 1, 1))
    for i in range(degree - 1, -1, -1):
        value = right[i+1].copy()
        value[:, 0, :] *= np.exp(1j*phases[i+1])
        value[:, 1, :] *= np.exp(-1j*phases[i+1])
        right[i] = signal @ value
    response = np.tile(np.eye(2, dtype=complex), (len(x), 1, 1))
    jacobian = np.empty((len(x), degree + 1))
    for i, phi in enumerate(phases):
        response[:, :, 0] *= np.exp(1j*phi)
        response[:, :, 1] *= np.exp(-1j*phi)
        jacobian[:, i] = np.real(1j*response[:, 0, 0]*right[i][:, 0, 0]
                                - 1j*response[:, 0, 1]*right[i][:, 1, 0])
        if i < degree:
            response = response @ signal
    return response[:, 0, 0].real, jacobian


def fit_retargeted_phases(n: int, *, component_timeout_seconds: float = 60):
    """Bounded optional continuation from frozen tau4.1; no interval proof.

    Canonical circuit generation uses stored phases, not a fit. This routine
    reproduces their deterministic 0.2-or-smaller continuation steps with
    analytic derivatives and at most 500 evaluations per step/component.
    """
    import time
    import numpy as np
    from scipy.optimize import least_squares
    from numpy.polynomial.chebyshev import chebval
    from .simulation.qsp import COSINE_PHASES, cosine_chebyshev_coefficients, verify_cosine_phases
    from .simulation.qsp_evolution import SINE_PHASES, sine_chebyshev_coefficients, verify_sine_phases
    if isinstance(n, bool) or not isinstance(n, int) or n not in RETARGETED_PHASES:
        raise ValueError("Retargeted reproduction is bounded to n=2 or n=4")
    if (isinstance(component_timeout_seconds, bool) or not isinstance(component_timeout_seconds, Real)
            or not math.isfinite(component_timeout_seconds) or not 0 < component_timeout_seconds <= 60):
        raise ValueError("Component timeout must be positive and at most 60 seconds")
    tau = RETARGETED_PHASES[n]["tau"]
    result = {"tau": tau, "certified": False,
              "method": "analytic-Jacobian continuation from frozen tau4.1",
              "maximum_step_size": .2, "maximum_evaluations_per_step": 500}
    for kind, initial, coefficients, verify in (
        ("cosine", COSINE_PHASES, cosine_chebyshev_coefficients, verify_cosine_phases),
        ("sine", SINE_PHASES, sine_chebyshev_coefficients, verify_sine_phases),
    ):
        phases = np.array(initial)
        degree = len(phases) - 1
        nodes = np.cos(np.pi*(np.arange(2*degree+5)+.5)/(2*degree+5))
        started = time.perf_counter()
        evaluations = 0
        for current in np.linspace(4.1, tau, math.ceil(abs(tau-4.1)/.2)+1)[1:]:
            target = chebval(nodes, coefficients(float(current), degree))
            def evaluate(candidate):
                if time.perf_counter()-started > component_timeout_seconds:
                    raise TimeoutError("Bounded phase continuation exceeded its component timeout")
                return qsp_response_jacobian(nodes, candidate)
            fit = least_squares(lambda p: evaluate(p)[0]-target, phases,
                                jac=lambda p: evaluate(p)[1], max_nfev=500,
                                ftol=1e-13, xtol=1e-13, gtol=1e-13)
            phases = fit.x
            evaluations += fit.nfev
            if np.max(np.abs(qsp_response_jacobian(nodes, phases)[0]-target)) > 1e-8:
                raise RuntimeError("Bounded continuation failed its node-residual criterion")
        verification = {**verify(tuple(phases), tau), "certified": False}
        elapsed = time.perf_counter()-started
        if elapsed > component_timeout_seconds:
            raise TimeoutError("Bounded phase continuation exceeded its component timeout during verification")
        result[kind] = {"phases": phases.tolist(), "verification": verification,
                        "evaluations": evaluations, "wall_seconds": elapsed}
    return result


def _integer(value, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def rational(value: Real, name: str = "value") -> Fraction:
    """Use an exact supplied Fraction or the stated finite decimal of a real."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and real")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite and real")
    return value if isinstance(value, Fraction) else Fraction(str(value))


def commutator_l1_bounds(n: int, J: Real = 1, h: Real = Fraction(7, 10)) -> dict[str, Fraction]:
    """Exact Pauli l1 norms for A=J sum ZZ and B=h sum X on an open chain.

    AAB = 4 J² h sum_v deg(v) X_v + 8 J² h sum_interior Z X Z.
    BBA = 8 J h² sum_edges (ZZ - YY). Distinct Pauli strings cannot
    cancel their l1 coefficients. Every Pauli has spectral norm one.
    """
    _integer(n, "n", 1, MAX_SYSTEM_QUBITS)
    J, h = rational(J, "J"), rational(h, "h")
    if n == 1:
        return {"AB": Fraction(0), "AAB": Fraction(0), "BBA": Fraction(0)}
    return {
        "AB": 4 * abs(J * h) * (n - 1),
        "AAB": 8 * abs(J * J * h) * (2 * n - 3),
        "BBA": 16 * abs(J * h * h) * (n - 1),
    }


def product_formula_error_bound(n: int, time: Real, steps: int, order: int,
                                J: Real = 1, h: Real = Fraction(7, 10)) -> Fraction:
    """Rigorous intended-angle spectral bound, not a synthesis/compile bound."""
    _integer(steps, "steps", 1, MAX_STEPS)
    _integer(order, "order", 1, 2)
    t = abs(rational(time, "time"))
    c = commutator_l1_bounds(n, J, h)
    if order == 1:
        return t * t * c["AB"] / (2 * steps)
    return t**3 * (c["AAB"] / 24 + c["BBA"] / 12) / steps**2


def choose_steps(n: int, time: Real, order: int, operator_allowance: Real,
                 J: Real = 1, h: Real = Fraction(7, 10)) -> int:
    """Smallest integer meeting the rational ideal-angle bound, with a hard cap."""
    allowance = rational(operator_allowance, "operator_allowance")
    if allowance <= 0:
        raise ValueError("operator_allowance must be positive")
    _integer(order, "order", 1, 2)
    ratio = product_formula_error_bound(n, time, 1, order, J, h) / allowance
    if order == 1:
        steps = max(1, -(-ratio.numerator // ratio.denominator))
    else:
        steps = math.isqrt(ratio.numerator // ratio.denominator)
        if Fraction(steps * steps) < ratio:
            steps += 1
        steps = max(1, steps)
    if steps > MAX_STEPS:
        raise ValueError("The requested analytic accuracy exceeds the bounded step budget")
    return steps


def source_angle_rounding_bound(circuit: QuantumCircuit, *, n: int, time: Real,
                                steps: int, order: int, J: Real = 1,
                                h: Real = Fraction(7, 10)) -> Fraction:
    """Telescoping norm bound for original RX/RZ binary64 angles versus rationals.

    The generated gate circuit must still be in its original per-step form.
    This does not certify subsequent floating transpilation or synthesis.
    """
    _integer(n, "n", 1, MAX_SYSTEM_QUBITS)
    _integer(steps, "steps", 1, MAX_STEPS)
    _integer(order, "order", 1, 2)
    t, J, h = rational(time), rational(J), rational(h)
    if circuit.num_qubits != n:
        raise ValueError("Source width disagrees with the declared spin chain")
    counts = circuit.count_ops()
    expected_rx = n * steps if h * t else 0
    expected_rz = (n - 1) * steps * (2 if order == 2 else 1) if J * t else 0
    if counts.get("rx", 0) != expected_rx or counts.get("rz", 0) != expected_rz or counts.get("cx", 0) != 2 * expected_rz:
        raise ValueError("Source operation counts do not match the declared original product formula")
    expected = {"rx": 2 * h * t / steps, "rz": (2 if order == 1 else 1) * J * t / steps}
    total = Fraction(0)
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in expected:
            theta = float(instruction.operation.params[0])
            total += abs(Fraction.from_float(theta) - expected[name]) / 2
        elif name not in {"x", "cx", "barrier", "measure"}:
            raise ValueError(f"Unexpected source operation for angle-rounding check: {name}")
    return total


@dataclass(frozen=True)
class DynamicsTask:
    """One fixed Pauli-Z mean; disjoint bias and stochastic failure allocations."""
    absolute_error: float = .05
    algorithm_bias: float = .01
    synthesis_bias: float = .01
    sampling_error: float = .03
    sampling_failure: float = .004
    acceptance_failure: float = .001
    physical_failure: float = .005

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not math.isclose(self.algorithm_bias + self.synthesis_bias + self.sampling_error,
                            self.absolute_error, rel_tol=1e-12):
            raise ValueError("Bias and sampling allocations must add to the requested accuracy")
        if self.sampling_failure + self.acceptance_failure + self.physical_failure >= 1:
            raise ValueError("Whole-task failure allocations must total less than one")
        if self.absolute_error > 2:
            raise ValueError("The requested Pauli-mean error must not exceed its outcome width")

    @classmethod
    def at_accuracy(cls, accuracy: float) -> "DynamicsTask":
        e = rational(accuracy, "accuracy")
        return cls(float(e), float(e / 5), float(e / 5), float(3 * e / 5))

    @property
    def shots(self) -> int:
        return math.ceil(2 * math.log(2 / self.sampling_failure) / self.sampling_error**2)

    @property
    def failure_allowance(self) -> float:
        return self.sampling_failure + self.acceptance_failure + self.physical_failure

    def as_dict(self):
        return {**asdict(self), "accepted_samples": self.shots,
                "whole_task_failure_allowance": self.failure_allowance,
                "certified_confidence_lower_bound": None}


def attempts_per_slot(probability: float, slots: int, failure: float) -> int:
    """Minimal complete-attempt cap for each of all required accepted slots."""
    _integer(slots, "slots", 1, 1_000_000_000)
    if isinstance(probability, bool) or not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError("probability must lie in (0, 1]")
    if isinstance(failure, bool) or not math.isfinite(failure) or not 0 < failure < 1:
        raise ValueError("failure must lie in (0, 1)")
    if probability == 1:
        return 1
    target = math.log(failure / slots)
    log_miss = math.log1p(-probability)
    cap = max(1, math.ceil(target / log_miss))
    while cap * log_miss > target:
        cap += 1
    while cap > 1 and (cap - 1) * log_miss <= target:
        cap -= 1
    if cap > MAX_ATTEMPTS_PER_SLOT:
        raise ValueError("Required complete-attempt cap exceeds the bounded workload")
    return cap


def postselected_contract(reference_acceptance_lower: float, task: DynamicsTask) -> dict:
    """Conditional measured-law cost proposal, retaining joint-distribution scope.

    If TV(P,Q)<=epsilon for the FULL final joint law and p=P(E)>epsilon,
    Q(E)>=p-epsilon and TV(P|E,Q|E)<=epsilon/p. A +/-1 mean thus
    changes by at most 2epsilon/p. The native mean fit does not prove TV.
    """
    p = reference_acceptance_lower
    if isinstance(p, bool) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("reference_acceptance_lower must lie in (0, 1]")
    epsilon = task.synthesis_bias * p / 2
    if epsilon >= p:
        raise ValueError("Synthesis proposal cannot preserve a nonzero acceptance bound")
    lower = p - epsilon
    cap = attempts_per_slot(lower, task.shots, task.acceptance_failure)
    miss = (1 - lower)**cap
    expected = -math.expm1(cap * math.log1p(-lower)) / lower if lower < 1 else 1.0
    return {
        "reference_acceptance_lower": p,
        "synthesis_model_error_per_attempt": epsilon,
        "conditional_acceptance_lower_after_synthesis_model": lower,
        "conditional_synthesis_Pauli_bias_allowance": 2 * epsilon / p,
        "attempts_per_accepted_slot": cap,
        "accepted_slots": task.shots,
        "maximum_complete_attempts": cap * task.shots,
        "all_slots_no_acceptance_union_bound": task.shots * miss,
        "conditional_expected_stopped_attempts_all_slots": expected * task.shots,
        "physical_failure_allowance_per_complete_attempt": task.physical_failure / (cap * task.shots),
        "certified_confidence_lower_bound": None,
        "conditional_confidence_lower_bound": 1 - task.failure_allowance,
        "assumptions": [
            "The stated finite-H numerical minimum acceptance is a valid lower bound; it is not formally certified",
            "The QDK mean-fit scalar upper-bounds total variation of the complete final joint data/ancilla distribution",
            "The independently declared ideal algorithm bias bound applies to each conditional Pauli mean",
            "Every independent trial starts with newly supplied clean zero qubits",
            "The declared physical-fault model and classical acceptance/readout interpretation are correct",
        ],
        "proof": "For event E, restricted l1 difference a and complement b satisfy |p-q|<=b; conditional TV<=(a+|p-q|)/(2p)<=TV(P,Q)/p",
        "does_not_certify": ["concrete synthesis", "operator/diamond error", "runtime tails", "fresh-state/reset hardware cost"],
    }


def estimate_static_dynamics_resources(circuit: QuantumCircuit, *, budget, profile,
                                       cache_dir, observable_budget=None,
                                       distance_domain=DISTANCE_DOMAIN):
    """Use an explicit common, bounded data AND factory distance domain.

    This is the same audited native PSSPC/SurfaceCode/RoundBasedFactory model
    as the shared adapter. Its default data and factory code domains stop at
    distance 25, which did not admit the complete unamplified-QSP workload at
    physical error 1e-3. No task/hardware/failure allowance is relaxed here.
    The study uses odd distances 3..41 for every method, both data and factory
    codes, and records that change. The frozen Grover report is untouched.
    """
    import hashlib
    import importlib.metadata
    from pathlib import Path
    from qiskit import qasm3
    from .resource_estimation import (
        AUDITED_SOURCE_COMMIT, PINNED_QDK_VERSION, SOURCES,
        choose_num_ts_per_rotation, prepare_measured_circuit, synthesis_model_error,
    )
    from qdk.qre import ErrorComposition, LatticeSurgery, PSSPC, estimate, instruction_name, property_name
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.instruction_ids import LATTICE_SURGERY, RZ
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode

    distances = tuple(distance_domain)
    if (not distances or len(distances) > 30 or len(set(distances)) != len(distances)
            or any(isinstance(d, bool) or not isinstance(d, int) or d < 3 or d > 61 or d % 2 == 0 for d in distances)):
        raise ValueError("Use at most 30 distinct odd code distances in [3,61]")
    if importlib.metadata.version("qdk") != PINNED_QDK_VERSION:
        raise RuntimeError("The audited QDK version is required")
    if observable_budget is not None and budget != observable_budget.resource_budget():
        raise ValueError("Budget and bounded-observable allocations differ")
    compiled = prepare_measured_circuit(circuit)
    qasm = qasm3.dumps(compiled)
    app = OpenQASMApplication(qasm)
    raw = next(app.enumerate_traces())
    counts = {instruction_name(key): count for key, count in raw.gate_counts.items()}
    if set(counts) - {"RZ", "T", "CCX", "MEAS_Z"} or raw.base_error != 0:
        raise ValueError("Unaudited canonical trace or pre-existing base error")
    rotations = raw.gate_counts.get(RZ, 0)
    ts = choose_num_ts_per_rotation(rotations, budget.synthesis_model_error_per_execution)
    synthesized = PSSPC(num_ts_per_rotation=ts).transform(raw)
    if synthesized is None:
        raise RuntimeError("PSSPC did not produce a transformed trace")
    base = float(synthesized.base_error)
    if not math.isclose(base, rotations * synthesis_model_error(ts), rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("Native synthesis semantics differ from the audited fit")
    # QDK's dataclass enumerator treats a LIST override as a domain; a tuple
    # would be a fixed field value. Keep this explicit and tested.
    code_query = SurfaceCode.q(distance=list(distances))
    results = estimate(
        app, GateBased(error_rate=profile.physical_error_rate, gate_time=profile.gate_time_ns,
                       measurement_time=profile.measurement_time_ns),
        isa_query=code_query * RoundBasedFactory.q(code_query=SurfaceCode.q(distance=list(distances)), cache_dir=Path(cache_dir)),
        trace_query=PSSPC.q(num_ts_per_rotation=ts) * LatticeSurgery.q(),
        max_error=base + budget.physical_failure_per_execution,
        composition=ErrorComposition.UnionBound, use_graph=False,
    )
    if not len(results):
        raise RuntimeError("No feasible point in the explicit bounded dynamics distance/factory domain")
    points = []
    for entry in results:
        physical = float(entry.error) - base
        if physical < -1e-12 or physical > budget.physical_failure_per_execution + 1e-12:
            raise RuntimeError("The native physical contribution violates the capped budget")
        physical = max(0., physical)
        points.append({
            "peak_physical_qubits": int(entry.qubits),
            "data_code_distance": int(entry.source[LATTICE_SURGERY].transform.distance),
            "runtime_per_execution_ns": float(entry.runtime),
            "serial_runtime_all_executions_ns": float(entry.runtime) * budget.executions,
            "qdk_combined_error_score_per_execution": float(entry.error),
            "synthesis_model_error_per_execution": base,
            "physical_failure_model_per_execution": physical,
            "physical_failure_union_bound_all_executions": physical * budget.executions,
            "factories": {instruction_name(key): {
                "copies": value.copies, "runs": value.runs, "states": value.states,
                "error_rate": value.error_rate,
            } for key, value in entry.factories.items()},
            "model_properties": {property_name(key): value for key, value in entry.properties.items()},
            "instruction_source_graph": str(entry.source),
        })
    return {
        "schema_version": 1, "status": "computed_conditional_model",
        "resource_estimate_kind": "conditional_mean_fit_resource_projection",
        "qdk_version": PINNED_QDK_VERSION, "qiskit_version": importlib.metadata.version("qiskit"),
        "audited_source_commit": AUDITED_SOURCE_COMMIT,
        "source_commit_is_package_release_commit": False,
        "qasm3": qasm, "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "source_gate_counts": dict(circuit.count_ops()), "decomposed_gate_counts": dict(compiled.count_ops()),
        "canonical_gate_counts": counts, "canonical_application_trace": str(raw),
        "synthesized_application_trace": str(synthesized),
        "synthesis": {
            "model": "Audited QDK PSSPC mixed-fallback mean fit for random angles",
            "cost_semantics": "Expected fallback work, not a worst-case or per-angle certificate",
            "fit_domain": "Per-rotation modeled accuracy < 1e-4",
            "num_arbitrary_rotations": rotations, "fixed_num_ts_per_rotation": ts,
            "model_error_per_rotation": synthesis_model_error(ts) if rotations else 0.,
            "model_error_per_execution": base,
            "actual_synthesis_sequence_generated": False,
            "instance_approximation_certificate_verified": False,
        },
        "budget": asdict(budget), "physical_profile": asdict(profile),
        "observable_contract": observable_budget.as_contract() if observable_budget else None,
        "error_composition": "UnionBound; synthesis base and physical faults are explicitly separated",
        "isa_graph_pruning": False,
        "surface_code_data_distance_domain": list(distances),
        "factory_surface_code_distance_domain": list(distances),
        "native_default_distance_domain": list(range(3, 26, 2)),
        "distance_domain_scope": "A common bounded extended search for all dynamics methods; unchanged physical/code formulas and task budgets; not the frozen Grover search domain",
        "pareto_completeness": "Enumerated listed domain only; not a global optimality certificate",
        "runtime_semantics": "Capped circuit attempts times QDK mean-fit per-attempt model time, not a stochastic synthesis runtime-tail bound",
        "limitations": [
            "No synthesis/channel error certificate; empirical mean-fit costs are conditional",
            "QDK abstracts Clifford scheduling rather than retaining an exact routed schedule",
            "Physical reliability is conditional on the hardware/QEC/factory model",
            "Clean-state initialization/reset and classical communication/control timing are excluded",
        ],
        "sources": list(SOURCES), "pareto_points": points,
    }
