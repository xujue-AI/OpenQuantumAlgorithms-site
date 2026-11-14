# SPDX-License-Identifier: MIT
"""Bounded source and conditional quantum-schedule costs for robust QPE.

This is a separate adapter, never an extension of the frozen feedback task.
One four-fringe macrocycle runs 400 times on a machine selected before readout.
The source preparation, controlled access, resets, measurements and factory
supply belong to the quantum schedule. Classical count accumulation and exact
rational inference have known operation scaling but unmeasured latency and
are excluded from its physical totals. No end-to-end confidence is certified.

This file's module body uses only the standard library; the existing package
initializer can still load its original dependencies. Circuit construction,
synthesis, small density checks and native estimation occur in explicit functions.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path
from typing import Any, Callable

MAX_SOURCE_OPERATIONS = 1000
MAX_COMPILED_OPERATIONS = 100_000
MAX_GENERATED_OPERATIONS = 200_000
MAX_CODE_DISTANCE = 15
PHYSICAL_TRIALS = 4
SOURCE_NAMES = {"reset", "x", "h", "p", "cx", "sdg", "measure"}
COMPILED_NAMES = {"reset", "measure", "barrier", "h", "x", "y", "z",
                  "s", "sdg", "t", "tdg", "cx"}


def _fixed_integer(name: str, value: int, expected: int) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} is frozen to {expected}")


def _bounded_integer(name: str, value: int, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def _fixed_fraction(name: str, value: Fraction, expected: Fraction) -> None:
    if type(value) is not Fraction or value != expected:
        raise ValueError(f"{name} requires the exact frozen Fraction {expected}")


def _fixed_real(name: str, value: float, expected: float) -> None:
    if type(value) not in (float, int) or not math.isfinite(value) or value != expected:
        raise ValueError(f"{name} is frozen to {expected}")


def rational_failure_bound(levels: int, shots: int, additive_bound: Fraction) -> Fraction:
    """The existing conservative law, evaluated with its exact 128-term bound.

    This arithmetic does not import a circuit package or infer an unknown phase.
    The method and its assumptions remain those of robust_qpe.py; independent
    tests must check exact agreement before a new record is accepted.
    """
    _bounded_integer("levels", levels, 2, 6)
    _bounded_integer("shots", shots, 1, 25_000)
    if type(additive_bound) is not Fraction or not 0 <= additive_bound < Fraction(1, 8):
        raise ValueError("additive_bound must be an exact Fraction in [0,1/8)")
    x = 2 * shots * (Fraction(1, 8) - additive_bound) ** 2
    total = term = Fraction(1)
    for index in range(1, 129):
        term *= x / index
        total += term
    return min(Fraction(1), Fraction(4 * levels) / total)


@dataclass(frozen=True)
class RobustResourceTask:
    phase: float = .3
    target_qubits: int = 1
    levels: int = 2
    shots: int = 400
    additive_bound: Fraction = Fraction(1, 50)
    statistical_failure: Fraction = Fraction(1, 200)
    physical_failure_budget: float = .0025
    factory_availability_budget: float = .0025
    operator_error_allowance: float = 1e-4

    def __post_init__(self) -> None:
        _fixed_real("phase", self.phase, .3)
        _fixed_integer("target_qubits", self.target_qubits, 1)
        _fixed_integer("levels", self.levels, 2)
        _fixed_integer("shots", self.shots, 400)
        _fixed_fraction("additive_bound", self.additive_bound, Fraction(1, 50))
        _fixed_fraction("statistical_failure", self.statistical_failure, Fraction(1, 200))
        _fixed_real("physical_failure_budget", self.physical_failure_budget, .0025)
        _fixed_real("factory_availability_budget", self.factory_availability_budget, .0025)
        _fixed_real("operator_error_allowance", self.operator_error_allowance, 1e-4)

    def contract(self) -> dict[str, Any]:
        bound = rational_failure_bound(self.levels, self.shots, self.additive_bound)
        if bound > self.statistical_failure:
            raise ValueError("The fixed trial count does not meet the exact statistical gate")
        return {
            "schema": "openquantum.robust_qpe_resource_task.v1",
            "status": "fixed_task_contract_not_an_execution_result",
            "phase_nominal": self.phase,
            "stored_base_angle_hex": float(2 * math.pi * self.phase).hex(),
            "target_qubits": 1, "levels": 2, "shots_per_fringe": 400,
            "requested_circular_phase_error": "1/16",
            "conditional_inference_radius": "1/24",
            "common_scalar_comparison_phase_domain": ["1/16", "15/16"],
            "additive_probability_promise": str(self.additive_bound),
            "exact_statistical_upper_bound": str(bound),
            "statistical_failure_allocation": str(self.statistical_failure),
            "physical_failure_budget_all_trials": self.physical_failure_budget,
            "factory_availability_budget_all_trials": self.factory_availability_budget,
            "requested_complete_failure_target": "1/100",
            "operator_error_allowance_per_macrocycle": self.operator_error_allowance,
            "synthetic_readout_confusion_applied": False,
            "independence_promise": "fresh IID Bernoulli observations within each of four fixed fringes",
            "bias_calibrated": False, "uniform_compiled_bias_certified": False,
            "hardware_confidence_certified": False,
            "phase_entering_inference": "unknown phase of the actual stored-angle U; true phase is never decoder input",
        }


def design_ledger(target_qubits: int = 1, levels: int = 2, shots: int = 400) -> dict[str, Any]:
    """Finite product-P constructor design counts, not numerical/physical evidence."""
    _bounded_integer("target_qubits", target_qubits, 1, 32)
    _bounded_integer("levels", levels, 2, 6)
    _fixed_integer("shots", shots, 400)
    trials = 2 * levels * shots
    calls = 2 * shots * ((1 << levels) - 1)
    raw = {"p": 3 * target_qubits * calls, "cx": 2 * target_qubits * calls,
           "h": 2 * trials, "x": target_qubits * trials, "sdg": levels * shots,
           "reset": (target_qubits + 1) * trials, "measure": trials}
    macro = {name: count // shots for name, count in raw.items()}
    if sum(macro.values()) > MAX_GENERATED_OPERATIONS:
        raise ValueError("Generated macrocycle exceeds its primitive-operation cap")
    return {
        "target_qubits": target_qubits, "levels": levels, "shots_per_fringe": shots,
        "logical_qubits": target_qubits + 1,
        "powers": [1 << j for j in range(levels)],
        "distinct_experiment_circuits": 2 * levels,
        "fresh_preparations": trials, "final_readouts": trials,
        "controlled_U_calls": calls, "maximum_coherent_base_calls": 1 << (levels - 1),
        "raw_all_trial_counts": raw, "source_counts_per_macrocycle": macro,
        "source_operations_per_macrocycle": sum(macro.values()),
        "source_status": "constructor_design_counts; generation does not supply phase, synthesis or physical validation",
    }


def classical_exclusions(task: RobustResourceTask) -> dict[str, Any]:
    """Explicitly separate classical preparation/decoding from physical totals."""
    return {
        "classical_input_preparation": {
            "included_in_physical_runtime": False,
            "operation_scaling": "O(n) coefficient/angle preparation and O(n*2**L) primitive source construction; offline",
            "wall_time_measured": False,
            "external_input_loading": "none in the explicitly specified classically easy product-P family",
            "quantum_input_preparation": "all n X gates and n+1 resets per fresh trial are included",
        },
        "count_aggregation": {
            "included_in_physical_runtime": False,
            "represented_readout_bits": 2 * task.levels * task.shots,
            "count_registers": 2 * task.levels,
            "bits_per_count": task.shots.bit_length(),
            "operation_scaling": "O(M) count increments; O(M*log(m)) elementary bit-work upper bound",
            "wall_time_measured": False,
        },
        "rational_inference": {
            "included_in_physical_runtime": False,
            "operation_scaling": "O(L) fixed-degree rational angle/lift operations; bigint cost depends on count precision",
            "arctangent_terms": 32, "confidence_exponential_terms": 128,
            "wall_time_measured": False, "phase_output_computed": False,
            "controller_ISA_calibrated": False,
        },
        "excluded_from_physical_totals": [
            "classical angle/circuit/synthesis preprocessing latency",
            "per-observation count accumulation and reliable classical record storage",
            "rational phase inference, inconclusive decisions and final interval encoding",
            "communication and any quantum idle exposure caused by excluded classical processing",
            "experimental bias calibration, decoder/controller throughput and controller area",
        ],
        "complete_end_to_end_cost": False,
        "end_to_end_confidence_established": False,
    }


def build_generation_macrocycle(target_qubits: int = 1, levels: int = 2):
    """Bounded actual primitive construction; no simulation/synthesis is invoked."""
    from qiskit import QuantumCircuit
    from .hadamard_qpe import build_schedule
    ledger = design_ledger(target_qubits, levels)
    trials = tuple(build_schedule(target_qubits, .3, levels))
    macro = QuantumCircuit(target_qubits + 1, 2 * levels, name="fresh-fringe robust QPE macrocycle")
    for bit, trial in enumerate(trials):
        macro.compose(trial, qubits=list(range(target_qubits + 1)), clbits=[bit], inplace=True)
    if dict(macro.count_ops()) != ledger["source_counts_per_macrocycle"]:
        raise AssertionError("Actual macrocycle differs from literal design counts")
    if len(macro.data) != ledger["source_operations_per_macrocycle"]:
        raise AssertionError("Actual macrocycle differs from its operation cap")
    macro.metadata = {"fringe_order": [{"power": 1 << j, "basis": basis, "record_bit": 2*j+i}
                      for j in range(levels) for i, basis in enumerate(("cos", "sin"))],
                      "each_trial_initializes_all_input_wires": True,
                      "repetition_count": 400, "generation_only": True}
    return macro, trials


def build_macrocycle(task: RobustResourceTask):
    if not isinstance(task, RobustResourceTask):
        raise ValueError("Use the exact RobustResourceTask")
    task.contract()
    macro, trials = build_generation_macrocycle(task.target_qubits, task.levels)
    check_macrocycle_source(macro, task)
    return macro, trials


def _events(circuit) -> list[list[Any]]:
    return [[item.operation.name,
             [circuit.find_bit(q).index for q in item.qubits],
             [circuit.find_bit(c).index for c in item.clbits]]
            for item in circuit.data if item.operation.name in {"reset", "measure"}]


def _signature(circuit) -> list[tuple[Any, ...]]:
    return [(item.operation.name,
             tuple(circuit.find_bit(q).index for q in item.qubits),
             tuple(circuit.find_bit(c).index for c in item.clbits),
             tuple(float(angle).hex() for angle in item.operation.params))
            for item in circuit.data if item.operation.name != "barrier"]


def check_macrocycle_source(circuit, task: RobustResourceTask) -> dict[str, Any]:
    """Check actual source signatures, including the literal access angles."""
    if not isinstance(task, RobustResourceTask) or circuit.num_qubits != 2 or circuit.num_clbits != 4:
        raise ValueError("The physical source requires two wires and four record bits")
    if circuit.num_parameters or len(circuit.data) > MAX_SOURCE_OPERATIONS:
        raise ValueError("Unbound or oversized macrocycle source")
    theta = 2 * math.pi * task.phase
    expected = []
    def add(name, wires, bits=(), parameters=()):
        expected.append((name, tuple(wires), tuple(bits), tuple(float(a).hex() for a in parameters)))
    for bit, (power, basis) in enumerate(((1, "cos"), (1, "sin"), (2, "cos"), (2, "sin"))):
        add("reset", (0,)); add("reset", (1,)); add("x", (1,)); add("h", (0,))
        for _ in range(power):
            add("p", (0,), parameters=(theta/2,)); add("cx", (0, 1))
            add("p", (1,), parameters=(-theta/2,)); add("cx", (0, 1))
            add("p", (1,), parameters=(theta/2,))
        if basis == "sin":
            add("sdg", (0,))
        add("h", (0,)); add("measure", (0,), (bit,))
    if _signature(circuit) != expected or float(circuit.global_phase) != 0.0:
        raise ValueError("Source reset/readout/access signature differs from the fixed task")
    ledger = design_ledger()
    counts = dict(circuit.count_ops())
    if counts != ledger["source_counts_per_macrocycle"]:
        raise ValueError("Source counts differ from the full fresh-trial schedule")
    return {"source_counts": counts, "whole_protocol_counts": ledger["raw_all_trial_counts"],
            "events": _events(circuit), "controlled_base_calls_per_macrocycle": 6,
            "repetitions": 400, "source_signature_checked": True}


def split_measured_trials(circuit):
    """Preserve four reset/readout boundaries, with no joint-transcript expansion.

    The whole macrocycle global phase stays in its original export. A trial's
    scalar phase is not assigned or fitted by this output-probability checker.
    """
    from qiskit import QuantumCircuit
    if circuit.num_qubits != 2 or circuit.num_clbits != 4 or circuit.num_parameters:
        raise ValueError("Split only the fixed two-wire/four-record macrocycle")
    if len(circuit.data) > MAX_COMPILED_OPERATIONS:
        raise ValueError("Compiled macrocycle exceeds its operation cap")
    trials, current, written = [], QuantumCircuit(2, 1), []
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        wires = [circuit.find_bit(q).index for q in item.qubits]
        bits = [circuit.find_bit(c).index for c in item.clbits]
        if len(trials) == PHYSICAL_TRIALS:
            raise ValueError("Quantum work after the last measurement is unsupported")
        if name == "measure":
            if wires != [0] or bits != [len(trials)] or bits[0] in written:
                raise ValueError("Every fringe needs its own ordered readout bit")
            current.append(item.operation, wires, [0])
            resets = [(current.find_bit(g.qubits[0]).index) for g in current.data if g.operation.name == "reset"]
            if resets != [0, 1]:
                raise ValueError("Every fringe must reset both input wires in order")
            if [g.operation.name for g in current.data[:2]] != ["reset", "reset"]:
                raise ValueError("Each fresh trial must begin with the two resets")
            written.append(bits[0]); trials.append(current); current = QuantumCircuit(2, 1)
        else:
            if bits or name in {"if_else", "while_loop", "for_loop", "switch_case", "store", "initialize"}:
                raise ValueError("Adaptive or opaque work is outside this fixed macrocycle")
            current.append(item.operation, wires, [])
    if len(trials) != PHYSICAL_TRIALS or len(current.data):
        raise ValueError("Missing readout or trailing quantum work in macrocycle")
    return tuple(trials)


def _compiled_frame(synthesized, task: RobustResourceTask):
    source_check = check_macrocycle_source(synthesized.source, task)
    circuit = synthesized.circuit
    if len(circuit.data) > MAX_COMPILED_OPERATIONS or set(circuit.count_ops()) - COMPILED_NAMES:
        raise ValueError("Compiled source must be bounded actual Clifford+T with reset/readout")
    if _events(circuit) != source_check["events"]:
        raise ValueError("Compilation changed the reset/readout schedule")
    eta = synthesized.report.get("guarded_numerical_operator_error_upper_estimate")
    if type(eta) not in (int, float) or not math.isfinite(eta) or not 0 <= eta <= task.operator_error_allowance:
        raise ValueError("Missing or excessive guarded numerical synthesis allowance")
    if 2 * eta > float(task.additive_bound):
        raise ValueError("Synthesis proposal exceeds the available per-fringe bias margin")
    return split_measured_trials(circuit), float(eta)


def _zero_probability_without_renormalization(trial) -> tuple[float, float]:
    """One small CPTP density evolution; no sampling, fitting or normalization."""
    import numpy as np
    from qiskit.quantum_info import DensityMatrix, Kraus
    rho = DensityMatrix.from_int(0, 4)
    reset = Kraus([np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex)])
    measured = False
    for item in trial.data:
        wires = [trial.find_bit(q).index for q in item.qubits]
        name = item.operation.name
        if name == "reset":
            rho = rho.evolve(reset, wires)
        elif name == "measure":
            if measured or wires != [0]:
                raise ValueError("A trial must have one final control readout")
            measured = True
        elif name != "barrier":
            if measured:
                raise ValueError("Quantum work after readout is unsupported")
            rho = rho.evolve(item.operation, wires)
    if not measured:
        raise ValueError("The compiled trial must include its final measurement")
    trace = complex(np.trace(rho.data))
    zero = complex(rho.data[0, 0] + rho.data[2, 2])
    if abs(trace.imag) > 1e-10 or abs(trace.real - 1) > 1e-10 or abs(zero.imag) > 1e-10:
        raise RuntimeError("Density trace/readout is outside the fixed numerical guard")
    if not -1e-10 <= zero.real <= 1 + 1e-10:
        raise RuntimeError("Raw readout probability is outside its numerical domain")
    return float(zero.real), float(abs(trace.real - 1))


def validate_compiled_macrocycle(synthesized, task: RobustResourceTask) -> dict[str, Any]:
    trials, eta = _compiled_frame(synthesized, task)
    theta = 2 * math.pi * task.phase
    rows = []
    for trial, (power, basis) in zip(trials, ((1, "cos"), (1, "sin"), (2, "cos"), (2, "sin")), strict=True):
        observed, trace_error = _zero_probability_without_renormalization(trial)
        ideal = (1 + (math.cos(power * theta) if basis == "cos" else math.sin(power * theta))) / 2
        error = abs(observed - ideal)
        if error > 2 * eta + 1e-10:
            raise RuntimeError("Compiled fringe differs from independent stored-angle answer")
        rows.append({"power": power, "basis": basis, "compiled_zero_probability": observed,
                     "independent_zero_probability": ideal, "absolute_probability_error": error,
                     "raw_density_trace_error": trace_error})
    return {"status": "passed_bounded_compiled_fringe_numerical_check",
            "compiled_trial_density_evolutions": 4, "joint_shot_transcripts_enumerated": 0,
            "rows": rows, "maximum_probability_error": max(r["absolute_probability_error"] for r in rows),
            "guarded_numerical_macrocycle_operator_error": eta,
            "conservative_per_fringe_probability_error_proposal": 2 * eta,
            "independent_answer": "two-by-two diagonal U and sine/cosine fringe at the actual stored base angle",
            "probabilities_renormalized": False, "global_phase_fitted": False,
            "uniform_bias_certified": False, "hardware_or_formal_confidence_certified": False,
            "sampling_performed": False, "phase_inference_performed": False}


def synthesize_macrocycle(source, task: RobustResourceTask):
    from .instrument_resources import InstrumentLimits, synthesize_instrument
    check_macrocycle_source(source, task)
    limits = InstrumentLimits(max_qubits=2, source_operations=MAX_SOURCE_OPERATIONS,
                              synthesized_operations=MAX_COMPILED_OPERATIONS,
                              validation_operations=MAX_COMPILED_OPERATIONS, validation_branches=16)
    return synthesize_instrument(source, task.operator_error_allowance,
                                 limits=limits, preserve_full_unitary=True)


def estimate_quantum_schedule(synthesized, task: RobustResourceTask, *,
                              cache_dir: Path, estimator: Callable | None = None) -> dict[str, Any]:
    """One common complete macrocycle estimator invocation, not four minima."""
    from .instrument_resources import InstrumentProfile, estimate_instrument
    _compiled_frame(synthesized, task)
    profile = InstrumentProfile(physical_error_rate=1e-4, maximum_code_distance=MAX_CODE_DISTANCE)
    estimate = estimate_instrument if estimator is None else estimator
    resource = estimate(synthesized, 400, profile=profile,
                        physical_failure_budget=task.physical_failure_budget,
                        factory_availability_budget=task.factory_availability_budget,
                        cache_dir=Path(cache_dir), include_all_feasible_points=True)
    if not isinstance(resource, dict) or resource.get("complete_algorithm_attempts") != 400:
        raise RuntimeError("Native report lost the full macrocycle repetition count")
    resource = deepcopy(resource)
    resource.update({
        "status": "computed_conditional_robust_qpe_quantum_schedule_surface_code_model",
        "task": task.contract(), "design_ledger": design_ledger(),
        "macrocycle_repetitions": 400,
        "fresh_trials_all_macrocycles": 1600,
        "native_field_semantics": "complete_algorithm_attempts counts 400 macrocycles; they are one requested robust inference protocol",
        "common_provisioning": "one preselected distance/factory/retry cap per Pareto point, for all four fringes and 400 macrocycles",
        "workspace_policy": "all input wires reset per fringe; extra reserved workspace initialized per macrocycle and protected through all four fringes",
        "classical": classical_exclusions(task),
        "complete_end_to_end_cost": False, "end_to_end_confidence_established": False,
        "compiled_fringe_check_executed": False,
        "cost_scope": "complete repeated quantum macrocycle only; excluded classical latency and its idle exposure are not zero",
        "comparison_improvement_computed": False,
        "native_call_scope": "one instrument invocation, seven odd-distance factory searches; supply/candidate internals not runtime-profiled here",
    })
    return resource
