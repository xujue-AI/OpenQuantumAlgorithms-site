"""Pure-data checks of the two frozen, saved CI-tail resource models.

Callers authenticate JSON/QASM and source provenance before calling this module.
There is no file I/O, project/scientific import, synthesis, or Native estimation.
Factory properties, profiles, budgets and workload probabilities remain saved
inputs. Complete Pareto reconstruction covers stored feasible ledgers, not fresh
factory-search completeness; the instrument bundle retains only Pareto records.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from fractions import Fraction
import math
import re


REL_TOL = 1e-10
ABS_TOL = 1e-30
INSTRUMENT_CASES = ("damping-Z", "lindblad-Z", "thermal-X", "thermal-Z")
COMPLETE_CASES = ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-X", "thermal-n2-Z")
POSITIVE = "computed_conditional_instrument_surface_code_model"
CONTROLLER_NEGATIVE = "infeasible_declared_controller_model"
_DEFAULT_PROFILE = {
    "physical_error_rate": 0.0001, "physical_gate_time_ns": 100,
    "physical_measurement_time_ns": 500, "physical_initialization_time_ns": 100,
    "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
    "controller_failure_per_decision": 1e-15, "memory_prefactor": 0.1,
    "initialization_residual_prefactor": 0.1, "memory_threshold": 0.01,
    "factory_pipeline_failure_bound": 0.01, "factory_feedback_rounds_reserved": 3,
    "maximum_code_distance": 49,
}
_COMPLETE_PROFILES = {
    f"p{p:.0e}-controller{c:.0e}": {**_DEFAULT_PROFILE, "physical_error_rate": p,
                                   "controller_failure_per_decision": c}
    for c in (1e-15, 1e-18) for p in (0.001, 0.0001)
}
_FAILURES = {
    "data_workspace_code_exposure", "factory_output_handoff_storage",
    "logical_readout_scenario", "logical_initialization_scenario",
    "controller_scenario", "factory_output_quality_model",
}
_MACRO_FIELDS = {"operation", "count", "time_ns", "readouts", "preparations", "decisions", "event_failure_model"}


class ModelVerificationError(ValueError):
    """An authenticated saved record does not satisfy its declared model."""


def _fail(label, message):
    raise ModelVerificationError(f"{label}: {message}")


def _dict(value, label):
    if type(value) is not dict:
        _fail(label, "expected a plain JSON object")
    return value


def _list(value, label):
    if type(value) is not list:
        _fail(label, "expected a plain JSON array")
    return value


def _field(value, name, label):
    _dict(value, label)
    if name not in value:
        _fail(label, f"missing {name}")
    return value[name]


def _integer(value, label, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        _fail(label, "integer outside the declared domain")
    return value


def _number(value, label, minimum=0.0, maximum=None):
    if type(value) not in (int, float, Decimal):
        _fail(label, "expected a finite numeric value, excluding Boolean/string values")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail(label, "nonfinite numeric value")
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        _fail(label, "numeric value outside the declared domain")
    return number


def _exact(left, right):
    """Stored-to-stored equality must not conflate bool/int or int/float."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if type(left) is list:
        return len(left) == len(right) and all(_exact(a, b) for a, b in zip(left, right))
    return left == right


class _Checks:
    def __init__(self):
        self.count = 0
        self.max_residual = 0.0

    def equal(self, actual, expected, label):
        self.count += 1
        if not _exact(actual, expected):
            _fail(label, "exact saved/model equality failed")

    def near(self, actual, expected, label):
        self.count += 1
        actual = _number(actual, label)
        expected = _number(expected, label + ".recomputed")
        self.max_residual = max(self.max_residual, abs(actual - expected))
        if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            _fail(label, "independent model arithmetic differs from the saved value")


def factory_retry_cap(total_states, failure_probability, budget):
    """Smallest cap in 1..16 using exact decimal interpretations; zero for no T."""
    _integer(total_states, "factory.total_states")
    _number(failure_probability, "factory.failure_probability", maximum=1)
    _number(budget, "factory.availability_budget", maximum=1)
    failure, allowance = Fraction(str(failure_probability)), Fraction(str(budget))
    if not 0 < failure < 1 or not 0 < allowance < 1:
        _fail("factory.retry", "failure probability and budget must lie strictly in (0,1)")
    if total_states == 0:
        return 0
    bound = Fraction(total_states)
    for cap in range(1, 17):
        bound *= failure
        if bound <= allowance:
            return cap
    _fail("factory.retry", "factory retry cap exceeds the bounded model domain")


def parse_compiled_qasm(qasm, *, max_qubits=8, operation_limit=1_000_000):
    """Parse the saved flat Clifford+T instrument, retaining reset chronology.

    Barriers are counted in primitive counts but never activate/reset a wire.
    This parser is deliberately not a general OpenQASM interpreter.
    """
    _integer(max_qubits, "qasm.max_qubits", 1, 8)
    _integer(operation_limit, "qasm.operation_limit", 1, 1_000_000)
    if type(qasm) is not str or len(qasm) > 16 * 1024 * 1024:
        _fail("qasm", "expected bounded QASM text")
    lines = [line.strip() for line in qasm.splitlines() if line.strip()]
    if len(lines) < 5 or lines[:2] != ["OPENQASM 3.0;", 'include "stdgates.inc";']:
        _fail("qasm.header", "expected the saved OpenQASM 3 header")
    classical = re.fullmatch(r"bit\[(\d+)\] c;", lines[2])
    quantum = re.fullmatch(r"qubit\[(\d+)\] q;", lines[3])
    if classical is None or quantum is None:
        _fail("qasm.registers", "expected one declared c and q register")
    try:
        bits, n = int(classical[1]), int(quantum[1])
    except ValueError:
        _fail("qasm.registers", "register size is not a bounded integer")
    _integer(bits, "qasm.classical_bits", 1, operation_limit)
    _integer(n, "qasm.logical_qubits", 1, max_qubits)
    if len(lines) - 4 > operation_limit:
        _fail("qasm.operations", "compiled operation bound exceeded")
    active, initialized = set(), set()
    initializations_before_use = True
    counts, plain = Counter(), Counter()
    for index, line in enumerate(lines[4:], 5):
        label = f"qasm.line{index}"
        barrier = re.fullmatch(r"barrier (q\[\d+\](?:, q\[\d+\])*)\;", line)
        measured = re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line)
        gate = re.fullmatch(r"(h|x|y|z|s|sdg|t|tdg|cx|reset) q\[(\d+)\](?:, q\[(\d+)\])?;", line)
        if barrier:
            wires = [int(w) for w in re.findall(r"q\[(\d+)\]", barrier[1])]
            if len(set(wires)) != len(wires) or any(w >= n for w in wires):
                _fail(label, "invalid barrier wires")
            plain["barrier"] += 1
            continue
        if measured:
            if int(measured[1]) >= bits:
                _fail(label, "measurement bit is outside the declared register")
            name, wires = "measure", [int(measured[2])]
        elif gate:
            name = gate[1]
            wires = [int(gate[2])] + ([] if gate[3] is None else [int(gate[3])])
            if len(wires) != (2 if name == "cx" else 1):
                _fail(label, "gate arity differs from its canonical primitive")
        else:
            _fail(label, "unsupported or malformed compiled statement")
        if any(w >= n for w in wires) or len(set(wires)) != len(wires):
            _fail(label, "invalid quantum wire")
        if name == "cx" and abs(wires[0] - wires[1]) != 1:
            _fail(label, "the serialized routing model requires adjacent CX")
        plain[name] += 1
        if name == "reset":
            counts["initial_reset" if wires[0] not in active else "reset"] += 1
            initialized.update(wires)
        else:
            initializations_before_use &= set(wires) <= initialized
            counts[f"{name}_q{wires[0]}" if name in ("t", "tdg") else name] += 1
        active.update(wires)
    counts["trial_control"] += 1
    counts["initial_reset"] += 3*n - 1
    return {"logical_qubits": n, "classical_bits": bits, "event_counts": dict(counts),
            "operation_counts": dict(plain), "operation_count": len(lines) - 4,
            "all_inputs_initialized_before_use": bool(initializations_before_use and initialized == set(range(n)))}


def _profile(value, label):
    p = _dict(value, label)
    if set(p) != set(_DEFAULT_PROFILE):
        _fail(label, "profile fields differ from the declared model")
    out = {}
    for name in ("physical_gate_time_ns", "physical_measurement_time_ns", "physical_initialization_time_ns",
                 "code_cycle_time_ns", "reaction_time_ns"):
        out[name] = _integer(p[name], label + "." + name, 1)
    out["maximum_code_distance"] = _integer(p["maximum_code_distance"], label + ".maximum_code_distance", 3, 61)
    if out["maximum_code_distance"] % 2 == 0:
        _fail(label, "maximum code distance must be odd")
    out["factory_feedback_rounds_reserved"] = _integer(p["factory_feedback_rounds_reserved"], label + ".rounds", 3)
    for name in set(p) - set(out):
        out[name] = _number(p[name], label + "." + name)
    if not 0 < out["physical_error_rate"] < out["memory_threshold"] <= 0.01:
        _fail(label, "physical error must be below the declared threshold")
    if not 0 <= out["controller_failure_per_decision"] < 1 or not 0.01 <= out["factory_pipeline_failure_bound"] < 1:
        _fail(label, "invalid controller/factory failure probability")
    if out["memory_prefactor"] <= 0:
        _fail(label, "memory prefactor must be positive")
    if out["code_cycle_time_ns"] < 5*out["physical_gate_time_ns"] + out["physical_measurement_time_ns"]:
        _fail(label, "cycle is shorter than declared syndrome extraction")
    return out


def _costs(p, d, name, factory_ns, cap):
    dt, reaction = d*p["code_cycle_time_ns"], p["reaction_time_ns"]
    init = p["physical_initialization_time_ns"] + dt + reaction
    destructive = p["physical_measurement_time_ns"] + reaction
    parity = 2*init + dt + destructive
    cnot = 3*parity + 2*reaction
    if name == "initial_reset":
        values = init, 0, 1, 1
    elif name == "reset":
        values = parity + 2*reaction + dt, 1, 2, 5
    elif name == "measure":
        values = parity, 1, 2, 3
    elif name == "cx":
        values = cnot, 3, 6, 11
    elif name.startswith(("t_q", "tdg_q")):
        k, inverse = 3*(int(name.split("_q")[1]) + 1) + 1, int(name.startswith("tdg"))
        overhead = p["physical_initialization_time_ns"] + p["physical_measurement_time_ns"] + reaction
        rounds = p["factory_feedback_rounds_reserved"]
        supply = cap*(factory_ns + rounds*overhead + reaction)
        values = (supply + k*cnot + destructive + reaction + init + inverse*reaction,
                  3*k + 1, 6*k + 1, 11*k + 3 + inverse + cap*(rounds + 1))
    elif name in ("trial_control", "h", "x", "y", "z", "s", "sdg"):
        values = reaction, 0, 0, 1
    else:
        _fail("macro", "unsupported event name")
    return dict(zip(("time_ns", "readouts", "preparations", "decisions"), values))


def _pareto(points, time_key):
    selected = []
    for point in sorted(points, key=lambda v: (v["peak_physical_qubits"], v[time_key])):
        if not selected or point[time_key] < selected[-1][time_key]:
            selected.append(point)
    return selected


def _point(point, p, raw_p, parsed, attempts, physical_budget, availability_budget, check, label):
    point = _dict(point, label)
    n, counts = parsed["logical_qubits"], parsed["event_counts"]
    d = _integer(_field(point, "data_code_distance", label), label + ".distance", 3, p["maximum_code_distance"])
    if d % 2 == 0:
        _fail(label, "data distance must be odd")
    patches = 4*n - 1
    check.equal(_field(point, "reserved_logical_patches", label), patches, label + ".patches")
    check.equal(_field(point, "data_workspace_physical_qubits", label), patches*(2*d*d - 1), label + ".data_qubits")
    factory_qubits = _integer(_field(point, "factory_physical_qubits", label), label + ".factory_qubits")
    check.equal(_field(point, "peak_physical_qubits", label), patches*(2*d*d - 1) + factory_qubits, label + ".peak_qubits")
    supply = _dict(_field(point, "factory_supply", label), label + ".factory")
    factory_ns = _integer(_field(supply, "pipeline_time_ns", label), label + ".factory_time")
    factory_error = _number(_field(supply, "output_error", label), label + ".factory_error", maximum=1)
    t_per_attempt = sum(v for k, v in counts.items() if k.startswith(("t_q", "tdg_q")))
    total_t = t_per_attempt*attempts
    cap = factory_retry_cap(total_t, raw_p["factory_pipeline_failure_bound"], availability_budget)
    for key, expected in {
        "states_all_complete_attempts": total_t, "full_pipeline_retry_cap_per_state": cap,
        "worst_case_pipeline_attempts": total_t*cap, "fixed_logical_distillation_distance": d,
        "pipeline_feedback_rounds_reserved": p["factory_feedback_rounds_reserved"],
        "native_code_cycle_time_ns": p["code_cycle_time_ns"],
    }.items():
        check.equal(_field(supply, key, label), expected, label + ".factory." + key)
    check.equal(_field(supply, "native_qec_crossing_prefactor", label), 0.03, label + ".factory.prefactor")
    check.equal(_field(supply, "native_qec_threshold", label), raw_p["memory_threshold"], label + ".factory.threshold")
    availability = total_t*Fraction(str(raw_p["factory_pipeline_failure_bound"]))**cap
    check.near(_field(supply, "availability_failure_union_bound", label), float(availability), label + ".availability")
    if availability > Fraction(str(availability_budget)):
        _fail(label, "factory availability budget exceeded")
    if factory_error*total_t > float(physical_budget)*0.75 or factory_error >= 7.1*p["physical_error_rate"]:
        _fail(label, "factory output quality violates its recorded constraints")
    if total_t == 0:
        for value, expected, name in ((factory_error, 0.0, "error"), (factory_ns, 0, "time"), (factory_qubits, 0, "qubits")):
            check.equal(value, expected, label + ".zero_T_factory_" + name)
        check.equal(_number(supply["availability_failure_union_bound"], label), 0.0, label + ".zero_T_availability")
    elif factory_ns <= 0 or factory_qubits <= 0:
        _fail(label, "positive T workload requires a recorded factory")
    q = -math.expm1(d*math.log1p(-2*p["physical_error_rate"]))/2
    readout = math.fsum(math.comb(d, k)*q**k*(1 - q)**(d - k) for k in range((d + 1)//2, d + 1))
    memory = p["memory_prefactor"]*(p["physical_error_rate"]/p["memory_threshold"])**((d + 1)//2)
    preparation = p["initialization_residual_prefactor"]*(p["physical_error_rate"]/p["memory_threshold"])**((d + 1)//2)
    rows = _list(_field(point, "macro_ledger_per_complete_attempt", label), label + ".macros")
    if len(rows) != len(counts):
        _fail(label, "macro list does not contain exactly the compiled events")
    check.equal([_field(r, "operation", label) for r in rows], sorted(counts), label + ".macro_names")
    runtime = readouts = preparations = decisions = 0
    event_failure_terms = []
    for row in rows:
        name = row["operation"]
        if set(row) != _MACRO_FIELDS:
            _fail(label, "macro fields differ from the declared model")
        count = _integer(row["count"], label + ".macro_count", 1)
        check.equal(count, counts[name], label + ".count." + name)
        cost = _costs(p, d, name, factory_ns, cap)
        for key, expected in cost.items():
            check.equal(row[key], expected, label + ".macro." + name + "." + key)
        event_error = cost["readouts"]*readout + cost["preparations"]*preparation + cost["decisions"]*p["controller_failure_per_decision"]
        check.near(row["event_failure_model"], event_error, label + ".event_error." + name)
        event_failure_terms.append(count*event_error)
        runtime += count*cost["time_ns"]
        readouts += count*cost["readouts"]
        preparations += count*cost["preparations"]
        decisions += count*cost["decisions"]
    cycles = -(-runtime//p["code_cycle_time_ns"])
    for key, expected in {"runtime_per_complete_attempt_ns": runtime, "worst_case_runtime_all_attempts_ns": runtime*attempts,
                          "code_cycles_per_complete_attempt": cycles}.items():
        check.equal(_field(point, key, label), expected, label + "." + key)
    overhead = p["physical_initialization_time_ns"] + p["physical_measurement_time_ns"] + p["reaction_time_ns"]
    dispatch = p["factory_feedback_rounds_reserved"]*overhead + p["reaction_time_ns"]
    hold = dispatch + 3*_costs(p, d, "cx", factory_ns, cap)["time_ns"] if total_t else 0
    check.equal(_field(supply, "extra_init_readout_feedback_dispatch_ns_per_pipeline", label), dispatch, label + ".dispatch")
    check.equal(_field(supply, "extra_output_port_hold_ns_per_state", label), hold, label + ".port_hold")
    # Retain the producer's accumulation order and its independent prefilter.
    # Native ledger agreement has a tolerance; feasibility never borrows it.
    memory_per_attempt = patches*cycles*memory
    port_per_attempt = t_per_attempt*(-(-hold//p["code_cycle_time_ns"]))*memory
    expected_per_attempt = memory_per_attempt + port_per_attempt + sum(event_failure_terms) + t_per_attempt*factory_error
    if expected_per_attempt*attempts > float(physical_budget):
        _fail(label, "independently recomputed physical budget exceeded")
    components = {
        "data_workspace_code_exposure": patches*cycles*memory*attempts,
        "factory_output_handoff_storage": t_per_attempt*(-(-hold//p["code_cycle_time_ns"]))*memory*attempts,
        "logical_readout_scenario": readouts*readout*attempts,
        "logical_initialization_scenario": preparations*preparation*attempts,
        "controller_scenario": decisions*p["controller_failure_per_decision"]*attempts,
        "factory_output_quality_model": total_t*factory_error,
    }
    saved_components = _dict(_field(point, "failure_components_all_attempts", label), label + ".failures")
    if set(saved_components) != _FAILURES:
        _fail(label, "expected exactly six failure components")
    for key, expected in components.items():
        check.near(saved_components[key], expected, label + ".failure." + key)
    if total_t == 0:
        for key in ("factory_output_handoff_storage", "factory_output_quality_model"):
            check.equal(_number(saved_components[key], label), 0.0, label + ".zero_T_failure." + key)
    saved_failure = _number(_field(point, "full_physical_failure_model", label), label + ".full_failure")
    check.near(saved_failure, sum(components.values()), label + ".failure_sum")
    if saved_failure > float(physical_budget):
        _fail(label, "physical budget exceeded")
    digest = _field(point, "accounting_trace_sha256", label)
    if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        _fail(label, "expected a saved Trace digest; raw Trace authentication is separate")


def _verify_profile(scenario, parsed, *, point_scope, expected_attempts, physical_budget, availability_budget, check, label):
    scenario = _dict(scenario, label)
    if point_scope not in ("retained", "full"):
        _fail(label, "point_scope must be retained or full")
    raw_p = _field(scenario, "profile", label)
    p = _profile(raw_p, label + ".profile")
    attempts = _integer(_field(scenario, "complete_algorithm_attempts", label), label + ".attempts", 1)
    if expected_attempts is not None:
        _integer(expected_attempts, label + ".expected_attempts", 1)
        check.equal(attempts, expected_attempts, label + ".attempt_binding")
    for name, supplied in (("physical_failure_budget", physical_budget), ("factory_availability_budget", availability_budget)):
        if supplied is None:
            supplied = _field(scenario, name, label)
        value = _number(supplied, label + "." + name, maximum=1)
        if not 0 < value < 1:
            _fail(label, "separate budgets must lie strictly in (0,1)")
        if name in scenario:
            check.equal(scenario[name], supplied, label + ".budget_binding." + name)
        if name == "physical_failure_budget":
            physical_budget = supplied
        else:
            availability_budget = supplied
    if not parsed["all_inputs_initialized_before_use"] or "measure" not in parsed["operation_counts"]:
        _fail(label, "complete instrument must initialize every used input and include readout")
    cap = factory_retry_cap(attempts*sum(v for k, v in parsed["event_counts"].items() if k.startswith(("t_q", "tdg_q"))),
                            raw_p["factory_pipeline_failure_bound"], availability_budget)
    decisions = sum(v*_costs(p, 3, k, 0, cap)["decisions"] for k, v in parsed["event_counts"].items())
    floor = decisions*attempts*p["controller_failure_per_decision"]
    if "controller_model_diagnostic" in scenario:
        diagnostic = _dict(scenario["controller_model_diagnostic"], label + ".controller")
        for key, expected in {"controller_decisions_per_attempt": decisions, "controller_decisions_all_attempts": decisions*attempts,
                              "controller_failure_per_decision_scenario": raw_p["controller_failure_per_decision"],
                              "physical_failure_budget": physical_budget, "budget_exceeded": floor > float(physical_budget),
                              "factory_retry_cap": cap}.items():
            check.equal(_field(diagnostic, key, label), expected, label + ".controller." + key)
        check.near(_field(diagnostic, "distance_independent_controller_model_contribution", label), floor, label + ".controller.floor")
    elif point_scope == "full":
        _fail(label, "complete scenarios require the saved controller diagnostic")
    status = _field(scenario, "status", label)
    pareto = _list(_field(scenario, "pareto_points", label), label + ".pareto")
    if status == CONTROLLER_NEGATIVE:
        if point_scope != "full" or floor <= float(physical_budget):
            _fail(label, "controller-negative status requires a strict saved-model budget excess")
        check.equal(pareto, [], label + ".negative_pareto")
        check.equal(_field(scenario, "all_feasible_points", label), [], label + ".negative_points")
        return {"status": status, "saved_points_checked": 0, "feasible_candidates_attested": 0}
    if status != POSITIVE or floor > float(physical_budget):
        _fail(label, "unsupported status or controller-infeasible positive scenario")
    for name in ("physical_failure_budget", "factory_availability_budget"):
        _field(scenario, name, label)  # Only controller-negative records omit these.
    limits = _dict(_field(scenario, "limits", label), label + ".limits")
    width = _integer(_field(limits, "max_qubits", label), label + ".max_qubits", 1, 8)
    operations = _integer(_field(limits, "synthesized_operations", label), label + ".max_operations", 1, 1_000_000)
    if parsed["logical_qubits"] > width or parsed["operation_count"] > operations:
        _fail(label, "compiled instrument exceeds saved width/operation limits")
    check.equal(_field(scenario, "synthesized_event_counts", label), parsed["operation_counts"], label + ".compiled_counts")
    check.equal(_field(scenario, "workspace_initializations_per_attempt", label), 3*parsed["logical_qubits"] - 1, label + ".workspace_resets")
    points = pareto if point_scope == "retained" else _list(_field(scenario, "all_feasible_points", label), label + ".all_points")
    if not points or len(points) > 5000:
        _fail(label, "positive scenario needs bounded saved points")
    feasible = _integer(_field(scenario, "feasible_candidate_count", label), label + ".feasible_count", 1)
    if point_scope == "full":
        check.equal(feasible, len(points), label + ".full_feasible_count")
    elif feasible < len(points):
        _fail(label, "attested candidate count is smaller than retained points")
    for i, point in enumerate(points):
        _point(point, p, raw_p, parsed, attempts, physical_budget, availability_budget, check, f"{label}.point{i}")
    if point_scope == "full" and len({_configuration(point) for point in points}) != len(points):
        _fail(label, "duplicate full factory configuration would hide a saved ledger")
    check.equal(pareto, _pareto(points, "runtime_per_complete_attempt_ns"), label + ".frontier")
    return {"status": status, "saved_points_checked": len(points), "feasible_candidates_attested": feasible}


def verify_profile_model(scenario, compiled_qasm, *, point_scope="retained", expected_attempts=None,
                         physical_budget=None, availability_budget=None):
    """Check one authenticated profile; full scope includes negatives/frontier."""
    check = _Checks()
    result = _verify_profile(scenario, parse_compiled_qasm(compiled_qasm), point_scope=point_scope,
                             expected_attempts=expected_attempts, physical_budget=physical_budget,
                             availability_budget=availability_budget, check=check, label="scenario")
    return {**result, "checks": check.count, "max_absolute_float_residual": check.max_residual,
            "point_scope": point_scope}


def verify_instrument_models(reports, compiled_qasm):
    """Verify four frozen tasks/eight profiles; unsaved candidates stay attested."""
    reports, compiled_qasm = _dict(reports, "reports"), _dict(compiled_qasm, "compiled_qasm")
    if set(reports) != set(INSTRUMENT_CASES) or set(compiled_qasm) != set(INSTRUMENT_CASES):
        _fail("instrument.tasks", "expected all four frozen task records and QASMs")
    check, summaries = _Checks(), []
    for task in INSTRUMENT_CASES:
        report = _dict(reports[task], task)
        check.equal(_field(report, "task", task), task, task + ".identity")
        workload = _dict(_field(report, "workload", task), task + ".workload")
        attempts = _integer(_field(workload, "complete_attempts", task), task + ".attempts", 1)
        slots = _integer(_field(workload, "accepted_samples", task), task + ".accepted_samples", 1)
        retry = _integer(_field(workload, "retry_cap_per_accepted_sample", task), task + ".algorithm_retry", 1)
        check.equal(attempts, slots*retry, task + ".complete_attempts")
        parsed = parse_compiled_qasm(compiled_qasm[task])
        profiles = _list(_field(report, "physical_profiles", task), task + ".profiles")
        if len(profiles) != 2:
            _fail(task, "expected the two frozen physical scenarios")
        for index, scenario in enumerate(profiles):
            expected = {**_DEFAULT_PROFILE, "physical_error_rate": (0.001, 0.0001)[index]}
            check.equal(_field(scenario, "profile", task), expected, task + ".frozen_profile")
            result = _verify_profile(scenario, parsed, point_scope="retained", expected_attempts=attempts,
                                     physical_budget=_field(workload, "physical_failure_budget", task),
                                     availability_budget=_field(workload, "factory_availability_budget", task),
                                     check=check, label=f"{task}.profile{index}")
            summaries.append({"task": task, "physical_error": expected["physical_error_rate"], **result})
    return {"status": "verified_saved_physical_model", "point_scope": "retained",
            "saved_points_checked": sum(s["saved_points_checked"] for s in summaries),
            "feasible_candidates_attested": sum(s["feasible_candidates_attested"] for s in summaries),
            "profiles": summaries, "checks": check.count, "max_absolute_float_residual": check.max_residual}


def _configuration(point):
    supply = point["factory_supply"]
    return (point["data_code_distance"], point["reserved_logical_patches"], point["factory_physical_qubits"],
            supply["pipeline_time_ns"], supply["output_error"])


def _energy(result, x, z, check, label):
    check.equal(x["profile"], z["profile"], label + ".profiles")
    groups = []
    for group in (x, z):
        points = group["all_feasible_points"]
        mapping = {_configuration(point): point for point in points}
        if len(mapping) != len(points):
            _fail(label, "duplicate full factory configuration would hide a group ledger")
        groups.append(mapping)
    xx, zz = groups
    joined = []
    for key in sorted(xx.keys() & zz.keys()):
        a, b = xx[key], zz[key]
        joined.append({"data_code_distance": key[0], "reserved_logical_patches": key[1], "factory_physical_qubits": key[2],
                       "factory_pipeline_time_ns": key[3], "factory_output_error": key[4],
                       "peak_physical_qubits": max(a["peak_physical_qubits"], b["peak_physical_qubits"]),
                       "worst_case_runtime_all_attempts_ns": a["worst_case_runtime_all_attempts_ns"] + b["worst_case_runtime_all_attempts_ns"],
                       "full_physical_failure_model": a["full_physical_failure_model"] + b["full_physical_failure_model"],
                       "factory_availability_failure_model": a["factory_supply"]["availability_failure_union_bound"] + b["factory_supply"]["availability_failure_union_bound"],
                       "group_points": {"X": a, "Z": b}})
    frontier = _pareto(joined, "worst_case_runtime_all_attempts_ns")
    check.equal(_field(result, "common_feasible_candidate_count", label), len(joined), label + ".common_count")
    check.equal(_field(result, "pareto_points", label), frontier, label + ".frontier")
    check.equal(_field(result, "status", label), "computed_common_configuration_energy_model" if joined else "no_common_feasible_configuration", label + ".status")
    for field in ("whole_workload_executed", "formal_confidence_certificate"):
        check.equal(_field(result, field, label), False, label + "." + field)
    return {"common_count": len(joined), "pareto_count": len(frontier)}


def verify_complete_models(results, compiled_qasm):
    """Verify all stored complete candidates, controller negatives and energy."""
    results, compiled_qasm = _dict(results, "results"), _dict(compiled_qasm, "compiled_qasm")
    cases, profiles = _dict(_field(results, "cases", "results"), "cases"), _dict(_field(results, "profiles", "results"), "profiles")
    if set(cases) != set(COMPLETE_CASES) or set(compiled_qasm) != set(COMPLETE_CASES):
        _fail("complete.cases", "expected all five frozen complete instruments")
    check, summaries, energy_summaries = _Checks(), [], []
    check.equal(profiles, _COMPLETE_PROFILES, "complete.frozen_profiles")
    check.equal(_field(results, "schema_version", "results"), 1, "complete.schema")
    for field in ("repeated_workloads_executed", "formal_confidence_certificate"):
        check.equal(_field(results, field, "results"), False, "complete." + field)
    for task in COMPLETE_CASES:
        case = _dict(cases[task], task)
        prepared = _dict(_field(case, "prepared", task), task + ".prepared")
        check.equal(_field(prepared, "case_id", task), task, task + ".identity")
        workload = _dict(_field(prepared, "complete_workload", task), task + ".workload")
        attempts = _integer(_field(workload, "complete_attempts", task), task + ".attempts", 1)
        slots = _integer(_field(workload, "accepted_slots", task), task + ".slots", 1)
        retry = _integer(_field(workload, "retry_cap_per_accepted_slot", task), task + ".retry", 1)
        check.equal(attempts, slots*retry, task + ".complete_attempts")
        parsed = parse_compiled_qasm(compiled_qasm[task])
        check.equal(_field(prepared, "source_logical_qubits", task), parsed["logical_qubits"], task + ".width")
        check.equal(_field(prepared, "full_clifford_t_counts", task), parsed["operation_counts"], task + ".prepared_counts")
        check.equal(_field(workload, "full_clifford_t_counts_all_attempts", task),
                    {k: v*attempts for k, v in parsed["operation_counts"].items()}, task + ".workload_counts")
        for field in ("physical_failure_budget", "factory_availability_budget"):
            check.equal(_field(workload, field, task), _field(prepared, field, task), task + ".budget_binding." + field)
        scenarios = _dict(_field(case, "scenarios", task), task + ".scenarios")
        if set(scenarios) != set(profiles):
            _fail(task, "missing or extra complete physical/controller scenarios")
        for name, scenario in scenarios.items():
            check.equal(_field(scenario, "profile", task), profiles[name], task + ".profile_binding." + name)
            summary = _verify_profile(scenario, parsed, point_scope="full", expected_attempts=attempts,
                                      physical_budget=prepared["physical_failure_budget"], availability_budget=prepared["factory_availability_budget"],
                                      check=check, label=f"{task}.{name}")
            summaries.append({"task": task, "scenario": name, **summary})
    energy = _dict(_field(results, "energy", "results"), "energy")
    if set(energy) != set(profiles):
        _fail("energy", "missing or extra physical/controller scenarios")
    for name, result in energy.items():
        summary = _energy(result, cases["thermal-n2-X"]["scenarios"][name], cases["thermal-n2-Z"]["scenarios"][name], check, "energy." + name)
        energy_summaries.append({"scenario": name, **summary})
    return {"status": "verified_saved_physical_model", "point_scope": "all_stored_feasible_points",
            "saved_points_checked": sum(s["saved_points_checked"] for s in summaries),
            "positive_scenarios_checked": sum(s["status"] == POSITIVE for s in summaries),
            "controller_negatives_checked": sum(s["status"] == CONTROLLER_NEGATIVE for s in summaries),
            "scenarios": summaries, "energy": energy_summaries,
            "checks": check.count, "max_absolute_float_residual": check.max_residual}
