# SPDX-License-Identifier: MIT
"""Bounded complete QSP/thermal workloads for the serialized instrument model.

This module reuses the explicit source algorithms and instrument backend. It
does not define another surface-code model, omit measurement/reset events, or
materialize repeated quantum circuits. Source intervals, guarded numerical
realization, accepted-shot statistics and physical-model scenarios are separate.
Only the specified one-/two-spin tasks are supported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.linalg import expm

if __package__:
    from .. import instrument_resources as ir
    from .. import hamiltonian_simulation as hs
    from . import retargetable_qsp as rq, thermal_spectral as ts
    from .qsp_certificate import DecimalInterval
else:
    import instrument_resources as ir
    import hamiltonian_simulation as hs
    import retargetable_qsp as rq
    import thermal_spectral as ts
    from qsp_certificate import DecimalInterval

MAX_SYSTEM_QUBITS = 2
MAX_COMPLETE_ATTEMPTS = 10**12
NUMERICAL_CHECK_GUARD = 1e-10
DEFAULT_SYNTHESIS_ALLOWANCE = 5e-5
DEFAULT_REALIZATION_RESERVE = 1e-7


def _finite(value, name: str, *, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and real")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(f"{name} outside nonnegative domain")
    return float(value)


def _upper(value: Decimal | Fraction) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("bound not representable by finite binary64")
    exact = Fraction.from_float(number) if isinstance(value, Fraction) else Decimal.from_float(number)
    return math.nextafter(number, math.inf) if exact < value else number


def _lower(value: Decimal) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("bound not representable by finite binary64")
    return math.nextafter(number, -math.inf) if Decimal.from_float(number) > value else number


def _decimal_upper(value: Fraction, precision=80) -> Decimal:
    return Context(prec=precision, rounding=ROUND_CEILING).divide(Decimal(value.numerator), Decimal(value.denominator))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


@dataclass
class CompleteResourceTask:
    case_id: str
    kind: str
    n_system: int
    measurement: str
    terms: tuple[tuple[float, str], ...]
    core: QuantumCircuit
    source: QuantumCircuit
    outcome_clbits: tuple[int, ...]
    accept_clbits: dict[int, int]
    score_values: tuple[float, ...]
    exact_mean: float
    source_error: dict[str, Any]
    target_amplitude_lower_decimal: str
    generic_target_amplitude_lower_decimal: str
    target_bound_method: str
    statistical_absolute_error: float
    statistical_failure_budget: float
    acceptance_failure_budget: float
    physical_failure_budget: float
    factory_availability_budget: float

    def as_record(self) -> dict:
        return {"case_id": self.case_id, "kind": self.kind, "n_system": self.n_system,
                "measurement": self.measurement, "Hamiltonian_terms": self.terms,
                "instance": ({"time": .5, "J": 1., "h": .7, "initial_state": "zero"}
                             if self.kind == "dynamics" else
                             {"beta": 1.5, "J": .6, "h": .35, "input": "Bell",
                              "reference_discard": "unconditioned"}),
                "source_error_contract": self.source_error,
                "source_logical_qubits": self.source.num_qubits,
                "source_counts": dict(self.source.count_ops()),
                "outcome_clbits": self.outcome_clbits, "accept_clbits": self.accept_clbits,
                "score_values": self.score_values, "score_bit_order": "first outcome clbit is least significant",
                "exact_reference_mean": self.exact_mean,
                "target_amplitude_lower_decimal": self.target_amplitude_lower_decimal,
                "generic_target_amplitude_lower_decimal": self.generic_target_amplitude_lower_decimal,
                "target_bound_method": self.target_bound_method,
                "statistical_absolute_error": self.statistical_absolute_error,
                "statistical_failure_budget": self.statistical_failure_budget,
                "acceptance_failure_budget": self.acceptance_failure_budget,
                "physical_failure_budget": self.physical_failure_budget,
                "factory_availability_budget": self.factory_availability_budget,
                "output": "one fixed accepted scalar score; state is defined before final data readout",
                "instrument_protocol": "fresh initialization; complete coherent source; all declared success measurements; reference discard if thermal; full final readout; every rejected trial charged"}


@dataclass
class PreparedResourceWorkload:
    task: CompleteResourceTask
    synthesized: Any
    source_check: dict
    routed_check: dict
    synthesized_check: dict
    contract: dict
    workload: dict

    def as_record(self) -> dict:
        return _jsonable({**self.task.as_record(), "actual_synthesis": self.synthesized.report,
                          "source_instrument_check": self.source_check,
                          "routed_instrument_check": self.routed_check,
                          "clifford_t_instrument_check": self.synthesized_check,
                          "error_and_conditioning_contract": self.contract,
                          "complete_workload": self.workload,
                          "full_clifford_t_counts": dict(self.synthesized.circuit.count_ops()),
                          "numerical_realization_certificate": False,
                          "formal_confidence_certificate": False,
                          "hardware_workload_executed": False})


def default_task_specs() -> tuple[tuple[str, int, str], ...]:
    """Four problem settings, five measured group circuits; n1 has no ZZ term."""
    return (("dynamics", 1, "Z"), ("dynamics", 2, "Z"),
            ("thermal", 1, "X"), ("thermal", 2, "X"), ("thermal", 2, "Z"))


def _exact_thermal_score(terms, beta: float, measurement: str, n: int) -> float:
    h = sum((c*hs.pauli_matrix(p) for c, p in terms), np.zeros((2**n, 2**n), complex))
    gibbs = expm(-beta*h)
    gibbs /= np.trace(gibbs)
    score_terms = [(c, p) for c, p in terms if ("X" in p if measurement == "X" else "Z" in p)]
    return math.fsum(c*float(np.trace(gibbs@hs.pauli_matrix(p)).real) for c, p in score_terms)


def build_complete_task(kind: str, n_system: int, measurement: str | None = None) -> CompleteResourceTask:
    """Construct fixed n1/n2 complete trials, without estimating or synthesizing."""
    if isinstance(n_system, bool) or not isinstance(n_system, int) or not 1 <= n_system <= MAX_SYSTEM_QUBITS:
        raise ValueError("complete workload study is bounded to one or two systems")
    if kind not in {"dynamics", "thermal"}:
        raise ValueError("only the fixed dynamics and thermal tasks are supported")
    n = n_system
    if kind == "dynamics":
        measurement = "Z" if measurement is None else measurement
        if measurement != "Z":
            raise ValueError("fixed dynamics task measures Z0")
        terms = tuple(hs.spin_terms(n, J=1., h=.7))
        core = rq.build_retargetable_evolution(terms, time=.5, decoded_error=1e-5)
        trial = QuantumCircuit(core.num_qubits, core.num_qubits, name=f"Complete dynamics n{n} Z0")
        trial.reset(range(core.num_qubits))
        trial.compose(core, inplace=True)
        # Joint final measurements commute. Record success before data so the
        # checker never invents a postselection after reading its output.
        trial.measure(range(n, core.num_qubits), range(n, core.num_qubits))
        trial.measure(range(n), range(n))
        trial.metadata = {**core.metadata, "measurement_order": "success then all data; equivalent joint distribution to original data-first final readout"}
        reference = hs.exact_evolution(n, time=.5, J=1., h=.7)[:, 0]
        z = hs.pauli_matrix("I"*(n-1)+"Z")
        exact = float(np.vdot(reference, z@reference).real)
        source_error = core.metadata["logical_source_certificate"]
        outcome, accept, scores = (0,), dict.fromkeys(range(n, core.num_qubits), 0), (1., -1.)
        target_lower = generic_lower = str(Decimal(".25"))
        bound_method = "exact scaled unitary U/4 has target branch norm1/4 for any normalized input"
        group_count = 1
    else:
        measurement = "X" if measurement is None else measurement
        if measurement not in {"X", "Z"} or (n == 1 and measurement == "Z"):
            raise ValueError("thermal n1 has only the nonzero X group; n2 has X and ZZ groups")
        terms = ts.spin_chain_terms(n, J=.6, h=.35)
        core = ts.build_thermal_spectral_filter(terms, beta=1.5, filter_error=1e-6, gibbs_error=1e-4)
        trial = ts.build_thermal_spectral_trial(core, measurement)
        success_count = core.metadata["n_success_ancillas"]
        outcome, accept = tuple(range(success_count, success_count+n)), dict.fromkeys(range(success_count), 0)
        scores = tuple((.35*sum(1-2*((value>>q)&1) for q in range(n)) if measurement == "X" else
                        .6*sum((1-2*((value>>q)&1))*(1-2*((value>>(q+1))&1)) for q in range(n-1)))
                       for value in range(2**n))
        exact = _exact_thermal_score(terms, 1.5, measurement, n)
        source_error = core.metadata["error_contract"]
        a = Fraction(source_error["exact_binary_input_parameter"])
        generic_lower = ts.ideal_conditioning_bounds(a, Fraction())["ideal_target_Bell_branch_amplitude_lower_decimal"]
        if any(set(label) <= {"I"} for _, label in terms):
            raise ValueError("fixed thermal Jensen bound requires an exactly traceless Pauli input")
        # Tr exp(-beta H)/d >= exp(-beta TrH/d)=1, since every listed Pauli
        # has trace zero. For F/4 on Bell, s>=exp(-a)/4. No empirical p is used.
        target_lower = ts.ideal_conditioning_bounds(a/2, Fraction())["ideal_target_Bell_branch_amplitude_lower_decimal"]
        bound_method = "known Bell input and exact trace-zero Pauli list; Jensen Tr exp(-beta H)/d>=1 gives s>=exp(-a_exact)/4; generic exp(-2a_exact)/4 retained separately"
        group_count = 1 if n == 1 else 2
    return CompleteResourceTask(f"{kind}-n{n}-{measurement}", kind, n, measurement, terms, core, trial,
                                outcome, accept, scores, exact, source_error, target_lower, generic_lower,
                                bound_method, .03/group_count, .0025/group_count, .0025/group_count,
                                .0025/group_count, .0025/group_count)


def accepted_sample_count(score_values, statistical_error: float, failure_budget: float) -> int:
    """Outward Hoeffding count: exact binary64 score width, decimal budget inputs."""
    values = tuple(float(v) for v in score_values)
    if not values or not all(map(math.isfinite, values)):
        raise ValueError("finite nonempty scalar-score table required")
    error = _finite(statistical_error, "statistical error", positive=True)
    failure = _finite(failure_budget, "statistical failure", positive=True)
    if failure >= 1:
        raise ValueError("statistical failure must be below1")
    width = Fraction.from_float(max(values))-Fraction.from_float(min(values))
    if width == 0:
        return 0
    with localcontext(Context(prec=80, rounding=ROUND_CEILING)) as context:
        e = Fraction(str(error))
        f = Fraction(str(failure))
        logarithm_argument = Decimal(2*f.denominator)/Decimal(f.numerator)
        logarithm_upper = logarithm_argument.ln().next_plus()
        factor = width*width/(2*e*e)
        factor_upper = Decimal(factor.numerator)/Decimal(factor.denominator)
        estimate = factor_upper*logarithm_upper
        count = int(estimate.to_integral_value(rounding=ROUND_CEILING))
    if not 1 <= count <= MAX_COMPLETE_ATTEMPTS:
        raise ValueError("accepted sample count exceeds the bounded workload domain")
    return count


def retry_cap(probability_lower: float, accepted_slots: int, failure_budget: float) -> dict:
    """Least outward-certified cap; repeated interval powers never build N trials."""
    p = _finite(probability_lower, "acceptance lower", positive=True)
    failure = _finite(failure_budget, "acceptance failure", positive=True)
    if p > 1 or failure >= 1 or isinstance(accepted_slots, bool) or not isinstance(accepted_slots, int) or accepted_slots < 1:
        raise ValueError("invalid accepted-slot/retry contract")
    maximum = MAX_COMPLETE_ATTEMPTS//accepted_slots
    if maximum < 1:
        raise ValueError("whole workload exceeds complete-attempt cap")
    q = DecimalInterval.point(1)-DecimalInterval.point(p)
    budget = DecimalInterval.rational(Fraction(str(failure)))
    def bounds(cap):
        return q.power(cap)*accepted_slots
    high = 1
    while bounds(high).hi > budget.lo:
        if high == maximum:
            raise ValueError("acceptance retry workload exceeds complete-attempt cap")
        high = min(maximum, 2*high)
    low = 0
    while high-low > 1:
        middle = (low+high)//2
        if bounds(middle).hi <= budget.lo:
            high = middle
        else:
            low = middle
    final, previous = bounds(high), bounds(high-1)
    return {"retry_cap_per_accepted_slot": high, "complete_attempts": high*accepted_slots,
            "no_acceptance_failure_upper_decimal_all_slots": str(final.hi),
            "previous_cap_failure_lower_decimal_all_slots": str(previous.lo),
            "minimality_proved_for_selected_lower_probability": previous.lo > budget.hi,
            "arithmetic": "exact binary64 lower probability; outward Decimal interval repeated-squaring and bounded doubling/binary search",
            "probability_scope": "source/numerical-realization conditional lower, not empirical acceptance or a hardware certificate"}


def conditioning_contract(task: CompleteResourceTask, synthesis_allowance: float = DEFAULT_SYNTHESIS_ALLOWANCE,
                          realization_reserve: float = DEFAULT_REALIZATION_RESERVE, *,
                          guarded_synthesis_error_upper_estimate: float | None = None) -> dict:
    """Staged source/realization score bias and conservative accepted-branch p."""
    synthesis = _finite(synthesis_allowance, "synthesis allowance", positive=True)
    reserve = _finite(realization_reserve, "nested/routing reserve")
    if not 1e-9 <= synthesis <= .1 or reserve > .01:
        raise ValueError("realization allowance exceeds this bounded study")
    actual = (None if guarded_synthesis_error_upper_estimate is None else
              _finite(guarded_synthesis_error_upper_estimate, "guarded synthesis error estimate"))
    if actual is not None and actual > synthesis:
        raise ValueError("guarded synthesis error estimate exceeds its requested allowance")
    source_value = (task.source_error["encoded_operator_error_upper"] if task.kind == "dynamics" else
                    task.source_error["precompiler_source_encoded_total_error_upper"])
    epsilon = Fraction.from_float(float(source_value))
    # The requested decimal allowance supplies the preflight contract. After
    # synthesis, the guarded reported binary64 sum itself is the error premise;
    # converting it through its shorter decimal spelling may round inward.
    synthesis_bound = Fraction(str(synthesis)) if actual is None else Fraction.from_float(actual)
    eta = synthesis_bound+Fraction(str(reserve))
    s = Fraction(Decimal(task.target_amplitude_lower_decimal))
    generic_s = Fraction(Decimal(task.generic_target_amplitude_lower_decimal))
    if epsilon+eta >= s:
        raise ValueError("combined source/realization error cannot support nonzero acceptance")
    width = Fraction.from_float(max(task.score_values))-Fraction.from_float(min(task.score_values))
    source_bias = 2*width*epsilon/s
    realization_bias = 2*width*eta/(s-epsilon)
    if source_bias > Fraction(".01")/(1 if task.kind == "dynamics" or task.n_system == 1 else 2):
        raise ValueError("source score bias exceeds its allocated task allowance")
    # Per-group allowance halves for thermal n2, so the energy sum stays<=.01.
    if realization_bias > Fraction(".01")/(1 if task.kind == "dynamics" or task.n_system == 1 else 2):
        raise ValueError("realization score bias exceeds its allocated task allowance")
    probability = (s-epsilon-eta)**2
    generic_probability = max(Fraction(), generic_s-epsilon-eta)**2
    return {"source_encoded_error_upper": _upper(epsilon),
            "source_encoded_error_upper_decimal": str(_decimal_upper(epsilon)),
            "synthesis_numerical_allowance": synthesis, "nested_control_routing_allowance_scenario": reserve,
            "guarded_numerical_operator_error_upper_estimate": actual,
            "conditioning_synthesis_error_upper": _upper(synthesis_bound),
            "conditioning_synthesis_error_upper_decimal": str(_decimal_upper(synthesis_bound)),
            "conditioning_synthesis_error_source": ("requested allowance before synthesis" if actual is None else
                                                    "guarded actual synthesis estimate interpreted as exact binary64"),
            "whole_realization_allowance_scenario": _upper(eta),
            "whole_realization_allowance_scenario_decimal": str(_decimal_upper(eta)),
            "target_branch_amplitude_lower_decimal": task.target_amplitude_lower_decimal,
            "source_branch_amplitude_lower_decimal": str(Context(prec=80, rounding=ROUND_FLOOR).divide(Decimal((s-epsilon).numerator), Decimal((s-epsilon).denominator))),
            "acceptance_lower_proposal": _lower(Context(prec=80, rounding=ROUND_FLOOR).divide(Decimal(probability.numerator), Decimal(probability.denominator))),
            "acceptance_lower_proposal_decimal": str(Context(prec=80, rounding=ROUND_FLOOR).divide(Decimal(probability.numerator), Decimal(probability.denominator))),
            "generic_acceptance_lower_control": _lower(Context(prec=80, rounding=ROUND_FLOOR).divide(Decimal(generic_probability.numerator), Decimal(generic_probability.denominator))),
            "source_score_bias_upper": _upper(source_bias),
            "realization_score_bias_upper_proposal": _upper(realization_bias),
            "absolute_score_error_upper_proposal": _upper(source_bias+realization_bias+Fraction(str(task.statistical_absolute_error))),
            "target_absolute_error": .05,
            "source_scope": task.source_error.get("scope", task.source_error.get("precompiler_source_scope")),
            "target_bound_method": task.target_bound_method,
            "realization_status": "conditional numerical synthesis/nested-control/routing premise; not an interval or hardware certificate",
            "realization_assumption": "The guarded actual per-rotation numerical error sum (or requested allowance at preflight) bounds synthesis, and the separate reserve bounds SDK nested/control/routing realization. Agreement between nominal and routed circuits alone does not prove the latter.",
            "instrument_assumption": "Reset/readout have their declared ideal CPTP/instrument meaning for numerical checks; physical failures are modeled separately",
            "coherent_error_is_stochastic_failure": False}


def resource_limits():
    return ir.InstrumentLimits(max_qubits=8, source_operations=20_000,
                               synthesized_operations=500_000, validation_operations=500_000)


def dynamics_measurement_order_check(task: CompleteResourceTask) -> dict:
    """Compare both full final classical distributions with a core Statevector.

    Every wire is initialized in zero and measured once in Z. Final readouts
    commute; each actual measurement mapping is applied to its independently
    evolved coherent prefix instead of comparing just the accepted Z0 mean.
    """
    if task.kind != "dynamics":
        raise ValueError("measurement-order comparison requires a dynamics task")
    original = rq.build_measured_trial(task.core)
    def joint(circuit):
        prefix = QuantumCircuit(circuit.num_qubits)
        prefix.global_phase = circuit.global_phase
        mapping, measured = {}, False
        for item in circuit.data:
            name = item.operation.name
            if name == "barrier":
                continue
            if name == "reset":
                if prefix.data or measured:
                    raise ValueError("dynamics comparison permits only fresh initial resets")
            elif name == "measure":
                measured = True
                bit = circuit.find_bit(item.clbits[0]).index
                wire = circuit.find_bit(item.qubits[0]).index
                if bit in mapping:
                    raise ValueError("dynamics comparison requires one readout per classical bit")
                mapping[bit] = wire
            else:
                if measured:
                    raise ValueError("dynamics comparison requires only final commuting readouts")
                prefix.append(item.operation, [circuit.find_bit(q).index for q in item.qubits])
        if set(mapping) != set(range(circuit.num_qubits)) or set(mapping.values()) != set(range(circuit.num_qubits)):
            raise ValueError("dynamics comparison requires complete one-to-one final readout")
        quantum = Statevector.from_instruction(prefix).probabilities()
        classical = np.zeros_like(quantum)
        for state, probability in enumerate(quantum):
            word = sum(((state >> wire) & 1) << bit for bit, wire in mapping.items())
            classical[word] += probability
        return classical
    data_first, success_first = joint(original), joint(task.source)
    residual = float(np.max(np.abs(data_first-success_first)))
    if residual > NUMERICAL_CHECK_GUARD:
        raise AssertionError("reordered dynamics readout changed the full joint distribution")
    return {"status": "numerically_equivalent", "maximum_absolute_joint_probability_difference": residual,
            "original_data_first_joint_probabilities": data_first,
            "resource_success_first_joint_probabilities": success_first,
            "bit_order": "classical bit0 is least significant; all data and success bits retained",
            "method": "independent coherent-prefix Statevector probabilities mapped through each complete final measurement schedule",
            "formal_certificate": False}


def synthesize_task(task: CompleteResourceTask, synthesis_allowance: float = DEFAULT_SYNTHESIS_ALLOWANCE,
                    realization_reserve: float = DEFAULT_REALIZATION_RESERVE, *, limits=None) -> PreparedResourceWorkload:
    """Materialize one actual CT trial and check its score; no physical run."""
    limits = resource_limits() if limits is None else limits
    conditioning_contract(task, synthesis_allowance, realization_reserve)
    synthesized = ir.synthesize_instrument(task.source, synthesis_allowance, limits=limits,
                                           preserve_full_unitary=True)
    actual_synthesis_error = synthesized.report["guarded_numerical_operator_error_upper_estimate"]
    if actual_synthesis_error > synthesis_allowance:
        raise AssertionError("actual synthesis exceeded guarded numerical allowance")
    contract = conditioning_contract(task, synthesis_allowance, realization_reserve,
                                     guarded_synthesis_error_upper_estimate=actual_synthesis_error)
    checks = [ir.simulate_scored_trial(circuit, task.outcome_clbits, task.score_values, task.accept_clbits, limits=limits)
              for circuit in (task.source, synthesized.routed, synthesized.circuit)]
    source, routed, actual = checks
    if task.kind == "dynamics":
        source["measurement_order_numerical_equivalence"] = dynamics_measurement_order_check(task)
    joint_routing_residual = float(np.max(np.abs(np.asarray(source["joint_probabilities"])-
                                                np.asarray(routed["joint_probabilities"]))))
    routed["source_joint_probability_maximum_absolute_difference"] = joint_routing_residual
    if (abs(source["mean"]-routed["mean"]) > NUMERICAL_CHECK_GUARD or
            abs(source["acceptance_probability"]-routed["acceptance_probability"]) > NUMERICAL_CHECK_GUARD or
            joint_routing_residual > NUMERICAL_CHECK_GUARD):
        raise AssertionError("routed instrument changed the supplied accepted score")
    if actual["acceptance_probability"] < contract["acceptance_lower_proposal"]-NUMERICAL_CHECK_GUARD:
        raise AssertionError("actual instrument fell below conditional acceptance proposal")
    if abs(actual["mean"]-task.exact_mean) > contract["source_score_bias_upper"]+contract["realization_score_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD:
        raise AssertionError("actual score exceeds the separated conditional source/realization proposal")
    accepted = accepted_sample_count(task.score_values, task.statistical_absolute_error, task.statistical_failure_budget)
    workload = retry_cap(contract["acceptance_lower_proposal"], accepted, task.acceptance_failure_budget)
    generic = retry_cap(contract["generic_acceptance_lower_control"], accepted, task.acceptance_failure_budget)
    counts = {name: int(count) for name, count in synthesized.circuit.count_ops().items()}
    workload.update({"accepted_slots": accepted, "generic_target_bound_control_attempts": generic["complete_attempts"],
                     "full_clifford_t_counts_all_attempts": {name: count*workload["complete_attempts"] for name, count in counts.items()},
                     "statistics": "Hoeffding for iid complete bounded scalar-score trials; correlations among bits within a trial allowed",
                     "conditional_failure_budget_sum": task.statistical_failure_budget+task.acceptance_failure_budget+task.physical_failure_budget+task.factory_availability_budget,
                     "statistical_failure_budget": task.statistical_failure_budget,
                     "acceptance_failure_budget": task.acceptance_failure_budget,
                     "physical_failure_budget": task.physical_failure_budget,
                     "factory_availability_budget": task.factory_availability_budget,
                     "whole_workload_executed": False,
                     "retry_protocol": "each accepted slot restarts a complete fresh preparation; all rejected schedules run fully"})
    return PreparedResourceWorkload(task, synthesized, source, routed, actual, contract, workload)


def controller_model_floor(prepared: PreparedResourceWorkload, profile) -> dict:
    """Distance-independent model contribution; not a real hardware lower bound."""
    counts = ir._event_counts(prepared.synthesized.circuit)
    counts["trial_control"] += 1
    n = prepared.synthesized.circuit.num_qubits
    counts["initial_reset"] += 3*n-1
    states = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    attempts = prepared.workload["complete_attempts"]
    cap = ir._factory_retry_cap(states*attempts, profile.factory_pipeline_failure_bound,
                                prepared.task.factory_availability_budget)
    decisions = sum(count*ir._costs(profile, 3, name, 0, cap)["decisions"] for name, count in counts.items())
    contribution = decisions*attempts*profile.controller_failure_per_decision
    return {"controller_decisions_per_attempt": decisions, "controller_decisions_all_attempts": decisions*attempts,
            "controller_failure_per_decision_scenario": profile.controller_failure_per_decision,
            "distance_independent_controller_model_contribution": contribution,
            "physical_failure_budget": prepared.task.physical_failure_budget,
            "budget_exceeded": contribution > prepared.task.physical_failure_budget,
            "factory_retry_cap": cap,
            "scope": "additive contribution within the declared engineering model; not a hardware fault lower bound or calibration"}


def estimate_workload(prepared: PreparedResourceWorkload, *, profile, cache_dir: Path,
                      include_all_feasible_points: bool = True) -> dict:
    """Reuse serialized model; infeasibility is explicit, never dropped work."""
    try:
        diagnostic = controller_model_floor(prepared, profile)
    except ValueError as error:
        if "retry cap" not in str(error):
            raise
        return {"status": "unsupported_factory_retry_model_domain", "profile": asdict(profile),
                "pareto_points": [], "all_feasible_points": [], "reason": str(error),
                "complete_algorithm_attempts": prepared.workload["complete_attempts"],
                "comparison_scope": "bounded serialized instrument model; no unsupported work omitted"}
    if diagnostic["budget_exceeded"]:
        return {"status": "infeasible_declared_controller_model", "profile": asdict(profile), "controller_model_diagnostic": diagnostic,
                "pareto_points": [], "all_feasible_points": [],
                "reason": "distance-independent controller contribution already exceeds the allocated physical budget",
                "complete_algorithm_attempts": prepared.workload["complete_attempts"],
                "comparison_scope": "separate serialized instrument scenario; frozen Grover/PSSPC model unchanged"}
    try:
        result = ir.estimate_instrument(prepared.synthesized, prepared.workload["complete_attempts"], profile=profile,
                                        physical_failure_budget=prepared.task.physical_failure_budget,
                                        factory_availability_budget=prepared.task.factory_availability_budget,
                                        cache_dir=Path(cache_dir), include_all_feasible_points=include_all_feasible_points)
    except RuntimeError as error:
        if "No feasible point in the declared" not in str(error):
            raise
        return {"status": "infeasible_declared_distance_factory_domain", "profile": asdict(profile), "controller_model_diagnostic": diagnostic,
                "pareto_points": [], "all_feasible_points": [], "reason": str(error),
                "complete_algorithm_attempts": prepared.workload["complete_attempts"],
                "comparison_scope": "separate serialized instrument scenario; no unsupported work is omitted"}
    return {**result, "controller_model_diagnostic": diagnostic}


def combine_energy_profiles(group_results: dict[str, dict]) -> dict:
    """Join only common code/factory candidates for one serial energy workload."""
    if set(group_results) != {"X", "Z"}:
        raise ValueError("two-spin energy requires both X and Z resource reports")
    if group_results["X"].get("profile") != group_results["Z"].get("profile"):
        raise ValueError("energy groups must use the same physical/controller scenario")
    if any("all_feasible_points" not in result for result in group_results.values()):
        raise ValueError("energy groups require all feasible candidates before common-configuration selection")
    def key(point):
        supply = point["factory_supply"]
        return (point["data_code_distance"], point["reserved_logical_patches"], point["factory_physical_qubits"],
                supply["pipeline_time_ns"], supply["output_error"])
    candidates = {}
    for axis, result in group_results.items():
        values = result["all_feasible_points"]
        candidates[axis] = {key(point): point for point in values}
    points = []
    for configuration in sorted(set(candidates["X"]) & set(candidates["Z"])):
        x, z = candidates["X"][configuration], candidates["Z"][configuration]
        points.append({"data_code_distance": configuration[0], "reserved_logical_patches": configuration[1],
                       "factory_physical_qubits": configuration[2], "factory_pipeline_time_ns": configuration[3],
                       "factory_output_error": configuration[4],
                       "peak_physical_qubits": max(x["peak_physical_qubits"], z["peak_physical_qubits"]),
                       "worst_case_runtime_all_attempts_ns": x["worst_case_runtime_all_attempts_ns"]+z["worst_case_runtime_all_attempts_ns"],
                       "full_physical_failure_model": x["full_physical_failure_model"]+z["full_physical_failure_model"],
                       "factory_availability_failure_model": x["factory_supply"]["availability_failure_union_bound"]+z["factory_supply"]["availability_failure_union_bound"],
                       "group_points": {"X": x, "Z": z}})
    pareto = []
    for point in sorted(points, key=lambda p: (p["peak_physical_qubits"], p["worst_case_runtime_all_attempts_ns"])):
        if not pareto or point["worst_case_runtime_all_attempts_ns"] < pareto[-1]["worst_case_runtime_all_attempts_ns"]:
            pareto.append(point)
    return {"status": "computed_common_configuration_energy_model" if points else "no_common_feasible_configuration",
            "pareto_points": pareto, "common_feasible_candidate_count": len(points),
            "selection_scope": "same distance, reserved patches, factory size, supply time and output error; serial X then Z runtime summed, peak max; group retry caps are counted controller protocols, not hardware reconfiguration",
            "whole_workload_executed": False, "formal_confidence_certificate": False}
