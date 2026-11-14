# SPDX-License-Identifier: MIT
"""One bounded literal Lindblad workload using the frozen instrument backend.

Only n2/r69/Z0 is physically estimated.  The n3/n4 family allocations remain
reserved and their literal ensemble checks explicitly reject.  Density checks,
guarded synthesis, representation/SDK premises and physical model assumptions
are separate; the repeated populations and hardware schedules are unexecuted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

if __package__:
    from .. import instrument_resources as ir
    from . import lindblad_workload as lw
    from .two_spin_lindblad import build_chain_circuit
else:
    import instrument_resources as ir
    import lindblad_workload as lw
    from two_spin_lindblad import build_chain_circuit

CASE_ID = "lindblad-n2-r69-Z0"
STEPS = 69
COMPLETE_TRIALS = 15_756
SYNTHESIS_ALLOWANCE = 5e-5
REPRESENTATION_NORM_RESERVE = 1e-7
SDK_NORM_RESERVE = 1e-7
NUMERICAL_CHECK_GUARD = 1e-10
STATISTICAL_CASE_FAILURE = Fraction(1, 600)
PHYSICAL_CASE_FAILURE = Fraction(1, 1200)
FACTORY_CASE_FAILURE = Fraction(1, 1200)


def limits() -> ir.InstrumentLimits:
    """Retain the four-wire density path, rather than the wider ensemble path."""
    return ir.InstrumentLimits(max_qubits=4, source_operations=20_000,
                               synthesized_operations=500_000,
                               validation_operations=500_000, validation_branches=256)


def run_caps() -> dict:
    """Predeclared limits enforced by this adapter and the bounded runner."""
    return {"physical_cases": 1, "prepared_synthesis_stages": 1,
            "native_rotation_synthesis_attempts_upper_bound": 4*828,
            "native_rotation_synthesis_attempts_scope": "actual gridsynth_unitary calls including up to four attempts per unique bound rotation target; at most 828 targets",
            "physical_profiles": 2, "source_instructions": 20_000,
            "routed_and_synthesized_instructions": 500_000,
            "density_validation_instructions": 500_000, "density_wires": 4,
            "unused_family_ensemble_branches": 256,
            "family_diagnostic_seconds": 60, "prepare_seconds": 600,
            "physical_seconds_per_profile": 180, "parent_seconds": 1200,
            "factory_queries_per_profile": 24, "surface_code_queries_per_profile": 24,
            "factory_candidates_per_query": 1000, "factory_candidates_per_profile": 24_000,
            "managed_json_qasm_bytes": 32*1024*1024,
            "private_checkpoint_bytes": 64*1024*1024,
            "query_scope": "top-level instrument adapter loop over odd distances 3..49: one factory and one surface-code query per distance; internal native factory construction is timeout-bounded and not included in these top-level query counts; no candidate truncation"}


def _downward(value: Fraction) -> float:
    number = float(value)
    return math.nextafter(number, -math.inf) if Fraction.from_float(number) > value else number


def _upward(value: Fraction) -> float:
    number = float(value)
    return math.nextafter(number, math.inf) if Fraction.from_float(number) < value else number


def _fraction_record(value: Fraction) -> dict:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def source_task() -> lw.ChainTask:
    """The single implemented task; no parameter sweep is accepted."""
    task = lw.ChainTask(2)
    if lw.select_steps(task) != STEPS or lw.output_contract()["complete_trials_per_case"] != COMPLETE_TRIALS:
        raise RuntimeError("The integrated uniform-accuracy task contract changed")
    return task


def _fixed_task(task: lw.ChainTask | None) -> lw.ChainTask:
    fixed = source_task()
    if task is None:
        return fixed
    if not isinstance(task, lw.ChainTask) or asdict(task) != asdict(fixed):
        raise ValueError("Only literal n2/r69/Z0 at the fixed integrated parameters is implemented")
    return task


def _events(circuit: QuantumCircuit) -> list[tuple[str, int | None]]:
    return [(item.operation.name,
             circuit.find_bit(item.clbits[0]).index if item.clbits else None)
            for item in circuit.data if item.operation.name in {"reset", "measure"}]


def _complete_trial(circuit: QuantumCircuit) -> None:
    if circuit.num_qubits != 4 or circuit.num_clbits != 4:
        raise ValueError("The literal trial requires exactly four quantum and classical wires")
    counts = circuit.count_ops()
    if counts.get("reset", 0) != 209 or counts.get("measure", 0) != 4:
        raise ValueError("The literal complete trial must retain all 209 resets and four readouts")
    outputs = [circuit.find_bit(item.clbits[0]).index for item in circuit.data
               if item.operation.name == "measure"]
    if outputs != [0, 1, 2, 3]:
        raise ValueError("Complete readout must retain the declared classical bit order")


def build_source(task: lw.ChainTask | None = None) -> QuantumCircuit:
    task = _fixed_task(task)
    source = build_chain_circuit(2, task.time, task.J, task.hx, task.rates,
                                 STEPS, 2, "product", observable="Z")
    _complete_trial(source)
    if len(source.data) != 1320:
        raise AssertionError("Literal source schedule changed")
    return source


def family_allocations() -> dict:
    cases = {str(n): {"selected_steps": r, "complete_trials": COMPLETE_TRIALS,
                      "statistical_failure_exact": _fraction_record(STATISTICAL_CASE_FAILURE),
                      "physical_failure_exact": _fraction_record(PHYSICAL_CASE_FAILURE),
                      "factory_availability_failure_exact": _fraction_record(FACTORY_CASE_FAILURE),
                      "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
                      "factory_availability_budget": _downward(FACTORY_CASE_FAILURE),
                      "physical_status": "implemented_n2_slice" if n == 2 else "unimplemented_reserved_allocation"}
             for n, r in ((2, 69), (3, 134), (4, 211))}
    return {"declared_family": [2, 3, 4], "implemented_physical_cases": [2],
            "statistics_family_failure_exact": _fraction_record(Fraction(1, 200)),
            "physical_family_failure_exact": _fraction_record(Fraction(1, 400)),
            "factory_family_failure_exact": _fraction_record(Fraction(1, 400)),
            "total_conditional_family_failure_exact": _fraction_record(Fraction(1, 100)),
            "case_allocations": cases, "unused_cases_reallocated": False,
            "family_physical_confidence_implemented": False,
            "scope": "one executed numerical source/compiled check and conditional physical model slice; n3/n4 allocations remain reserved"}


def representation_record(source: QuantumCircuit | None = None, *, reserve: float = REPRESENTATION_NORM_RESERVE) -> dict:
    """Exact input/H-angle discrepancies; the remaining reserve is a premise."""
    source = build_source() if source is None else source
    task = source_task()
    parameters = {}
    for name, value in asdict(task).items():
        if name == "n_system":
            continue
        decimal = Fraction(str(value))
        binary = Fraction.from_float(value)
        parameters[name] = {"supplied_binary64_hex": value.hex(),
                            "ideal_decimal": str(value),
                            "binary_minus_decimal_exact": _fraction_record(binary-decimal)}
    ideal = {"rx": Fraction(".35")/STEPS, "rz": Fraction(".6")/STEPS}
    h_error = Fraction()
    angles = {}
    for item in source.data:
        if item.operation.name not in {"rx", "rz", "cry"}:
            continue
        name, angle = item.operation.name, float(item.operation.params[0])
        key = (name, angle.hex())
        if key not in angles:
            angles[key] = {"gate": name, "stored_angle_hex": angle.hex(), "occurrences": 0}
            if name in ideal:
                difference = abs(Fraction.from_float(angle)-ideal[name])
                angles[key]["ideal_decimal_H_angle_exact"] = _fraction_record(ideal[name])
                angles[key]["absolute_H_angle_difference_exact"] = _fraction_record(difference)
        angles[key]["occurrences"] += 1
        if name in ideal:
            h_error += abs(Fraction.from_float(angle)-ideal[name])/2
    return {"parameters": parameters, "stored_source_angle_targets": list(angles.values()),
            "Hamiltonian_angle_coherent_error_component_upper": _upward(h_error),
            "Hamiltonian_angle_component_exact": _fraction_record(h_error),
            "representation_norm_reserve_scenario": reserve,
            "damping_angle_functions": "stored binary64 expm1/exp/sqrt/atan2 values; exact decimal damping-angle enclosure is not supplied",
            "certificate_scope": "exact rational binary-input and RX/RZ-angle discrepancies only; not a whole-circuit representation certificate",
            "representation_reserve_is_certified": False}


def error_contract(actual_synthesis_error: float, *, synthesis_allowance: float = SYNTHESIS_ALLOWANCE,
                   representation_reserve: float = REPRESENTATION_NORM_RESERVE,
                   sdk_reserve: float = SDK_NORM_RESERVE) -> dict:
    for name, value in (("actual synthesis", actual_synthesis_error), ("synthesis allowance", synthesis_allowance),
                        ("representation reserve", representation_reserve), ("SDK reserve", sdk_reserve)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 1e-9 <= synthesis_allowance <= .1 or actual_synthesis_error > synthesis_allowance:
        raise ValueError("Guarded synthesis estimate exceeds the requested allowance/domain")
    if representation_reserve > .001 or sdk_reserve > .001:
        raise ValueError("Representation/SDK reserve exceeds the fixed bounded study")
    eta = Fraction.from_float(float(actual_synthesis_error))+Fraction(str(representation_reserve))+Fraction(str(sdk_reserve))
    realization_bias = 2*eta
    if realization_bias > Fraction(1, 100):
        raise ValueError("Unconditional realization bias exceeds its .01 allowance")
    splitting = lw.uniform_split_bound(source_task(), STEPS)
    if splitting > Fraction(1, 100):
        raise AssertionError("Selected literal schedule violates its uniform splitting allowance")
    return {"splitting_bias_allowance": .01, "splitting_bias_upper": _upward(splitting),
            "splitting_bias_exact": _fraction_record(splitting), "realization_bias_allowance": .01,
            "statistical_absolute_error": .03, "target_absolute_error": .05,
            "requested_synthesis_allowance": synthesis_allowance,
            "guarded_actual_synthesis_error": actual_synthesis_error,
            "representation_norm_reserve_scenario": representation_reserve,
            "SDK_control_routing_norm_reserve_scenario": sdk_reserve,
            "whole_realization_error_scenario": _upward(eta), "whole_realization_error_exact": _fraction_record(eta),
            "realization_Z0_bias_upper_proposal": _upward(realization_bias),
            "absolute_Z0_error_upper_proposal": _upward(splitting+realization_bias+Fraction(3, 100)),
            "realization_bound_rule": "unconditional Z0 bias <=2 eta by coherent-segment telescoping and ideal reset/CPTP contraction; deterministic bias is not multiplied by trials",
            "realization_assumptions": ["guarded actual numerical matrix errors bound synthesis",
                                        "the separate representation reserve bounds decimal/binary sub-time and damping-angle realization",
                                        "the separate SDK reserve bounds control definitions and routing realization"],
            "reserve_is_inferred_from_density_agreement": False, "coherent_error_is_stochastic_failure": False,
            "full_representation_certificate": False, "formal_synthesis_certificate": False,
            "hardware_confidence_certificate": False}


@dataclass
class PreparedWorkload:
    task: lw.ChainTask
    synthesized: Any
    source_check: dict
    routed_check: dict
    synthesized_check: dict
    independent_reference: dict
    contract: dict
    workload: dict

    def as_record(self) -> dict:
        return _jsonable({"case_id": CASE_ID, "task": asdict(self.task), "selected_steps": STEPS,
                          "source_counts": dict(self.synthesized.source.count_ops()),
                          "actual_synthesis": self.synthesized.report,
                          "source_instrument_check": self.source_check,
                          "routed_instrument_check": self.routed_check,
                          "clifford_t_instrument_check": self.synthesized_check,
                          "independent_reference": self.independent_reference,
                          "error_contract": self.contract, "complete_workload": self.workload,
                          "representation_check": representation_record(self.synthesized.source,
                              reserve=self.contract["representation_norm_reserve_scenario"]),
                          "limits": asdict(limits()), "run_caps": run_caps(),
                          "family_allocations": family_allocations(), "hardware_execution": False,
                          "formal_confidence_certificate": False})


def check_synthesized(synthesized, task: lw.ChainTask | None = None, *,
                      synthesis_allowance: float = SYNTHESIS_ALLOWANCE,
                      representation_reserve: float = REPRESENTATION_NORM_RESERVE,
                      sdk_reserve: float = SDK_NORM_RESERVE) -> PreparedWorkload:
    task = _fixed_task(task)
    source, routed, circuit = synthesized.source, synthesized.routed, synthesized.circuit
    _complete_trial(source)
    if source != build_source(task):
        raise AssertionError("Synthesis source differs from the fixed literal schedule")
    for value in (routed, circuit):
        _complete_trial(value)
        if _events(value) != _events(source):
            raise AssertionError("Compilation changed the literal reset/readout event order")
    if synthesized.report.get("preserve_full_unitary") is not True or synthesized.report.get("transpiler_qubits_initially_zero") is not False:
        raise AssertionError("This workload requires actual full-unitary compiler lowering")
    if synthesized.report.get("limits") != asdict(limits()):
        raise AssertionError("The prepared source must retain the declared four-wire instruction limits")
    if synthesized.report.get("rotation_occurrences") != 828:
        raise AssertionError("Full-unitary level0 lowering must allocate all 828 literal rotations")
    actual_error = synthesized.report["guarded_numerical_operator_error_upper_estimate"]
    contract = error_contract(actual_error, synthesis_allowance=synthesis_allowance,
                              representation_reserve=representation_reserve, sdk_reserve=sdk_reserve)
    checks = [ir.simulate_trial(value, 0, {}, limits=limits()) for value in (source, routed, circuit)]
    original, routing, actual = checks
    for check in checks:
        if abs(check["acceptance_probability"]-1) > NUMERICAL_CHECK_GUARD:
            raise AssertionError("No postselection is allowed; trace must remain one numerically")
        check["density_path"] = "existing four-wire density/CPTP checker; no ensemble-history truncation"
        check["normalization_probability_is_postselection"] = False
    routing_residual = float(np.linalg.norm(original["reduced_before_output"]-routing["reduced_before_output"], ord="nuc"))
    routing["source_reduced_output_trace_norm_residual"] = routing_residual
    if abs(original["mean"]-routing["mean"]) > NUMERICAL_CHECK_GUARD or routing_residual > NUMERICAL_CHECK_GUARD:
        raise AssertionError("Routing changed the logical Z0 instrument")
    realized_mean_bias = abs(original["mean"]-actual["mean"])
    if realized_mean_bias > contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD:
        raise AssertionError("Synthesized Z0 instrument exceeds the separated realization proposal")
    exact = lw.reference_density(task)
    gate = lw.gate_density(task, STEPS)
    observable = lw.on_wire(lw.Z, 0, 2)
    exact_mean = float(np.trace(observable@exact).real)
    gate_mean = float(np.trace(observable@gate).real)
    if abs(original["mean"]-gate_mean) > NUMERICAL_CHECK_GUARD:
        raise AssertionError("Literal density instrument disagrees with gate-contracted reference")
    if abs(actual["mean"]-exact_mean) > contract["splitting_bias_upper"]+contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD:
        raise AssertionError("Actual Z0 answer exceeds the declared conditional bias contract")
    reference = {"exact_GKSL_Z0": exact_mean, "gate_contracted_Z0": gate_mean,
                 "source_bias_numerical": abs(original["mean"]-exact_mean),
                 "source_vs_gate_contracted_Z0_residual": abs(original["mean"]-gate_mean),
                 "actual_clifford_t_Z0_bias_numerical": abs(actual["mean"]-exact_mean),
                 "source_vs_synthesized_Z0_bias_numerical": realized_mean_bias,
                 "scope": "bounded independent full GKSL and gate-derived reduced channels; not a representation or compiler norm certificate"}
    counts = {name: int(count) for name, count in circuit.count_ops().items() if name != "barrier"}
    allocations = family_allocations()["case_allocations"]["2"]
    workload = {"complete_trials": COMPLETE_TRIALS, "complete_attempts": COMPLETE_TRIALS,
                "algorithm_retry_cap": 1, "postselection": "none", "whole_workload_executed": False,
                "primitive_counts_per_complete_trial": counts,
                "primitive_counts_all_trials": {name: count*COMPLETE_TRIALS for name, count in counts.items()},
                "reset_per_trial": 209, "readout_per_trial": 4,
                "source_resets_all_trials": 209*COMPLETE_TRIALS,
                "source_readouts_all_trials": 4*COMPLETE_TRIALS,
                "workspace_startup_initializations_per_trial": 11,
                "workspace_startup_initializations_all_trials": 11*COMPLETE_TRIALS,
                "system_initializations_per_trial": 2,
                "environment_initializations_per_trial": [138, 69],
                "discard_resets_before_environment_reuse_per_trial": [137, 68],
                "statistical_failure_exact": _fraction_record(STATISTICAL_CASE_FAILURE),
                "physical_failure_budget": allocations["physical_failure_budget"],
                "factory_availability_budget": allocations["factory_availability_budget"],
                "physical_failure_exact": _fraction_record(PHYSICAL_CASE_FAILURE),
                "factory_availability_failure_exact": _fraction_record(FACTORY_CASE_FAILURE),
                "conditional_case_failure_exact": _fraction_record(Fraction(1, 300)),
                "family_allocations_reallocated": False,
                "trial_protocol": "fresh complete product preparation; every literal reset and all system/environment readouts charged",
                "deterministic_bias_multiplied_by_trials": False}
    return PreparedWorkload(task, synthesized, original, routing, actual, reference, contract, workload)


def prepare_workload(task: lw.ChainTask | None = None, *, synthesis_allowance: float = SYNTHESIS_ALLOWANCE,
                     representation_reserve: float = REPRESENTATION_NORM_RESERVE,
                     sdk_reserve: float = SDK_NORM_RESERVE) -> PreparedWorkload:
    task = _fixed_task(task)
    error_contract(synthesis_allowance, synthesis_allowance=synthesis_allowance,
                   representation_reserve=representation_reserve, sdk_reserve=sdk_reserve)
    synthesized = ir.synthesize_instrument(build_source(task), synthesis_allowance,
                                           limits=limits(), preserve_full_unitary=True)
    return check_synthesized(synthesized, task, synthesis_allowance=synthesis_allowance,
                             representation_reserve=representation_reserve, sdk_reserve=sdk_reserve)


def fixed_profiles() -> tuple[ir.InstrumentProfile, ir.InstrumentProfile]:
    return tuple(ir.InstrumentProfile(physical_error_rate=value) for value in (1e-3, 1e-4))


def controller_diagnostic(prepared: PreparedWorkload, profile: ir.InstrumentProfile) -> dict:
    counts = ir._event_counts(prepared.synthesized.circuit)
    counts["trial_control"] += 1
    counts["initial_reset"] += 11
    states = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    cap = ir._factory_retry_cap(states*COMPLETE_TRIALS, profile.factory_pipeline_failure_bound,
                                _downward(FACTORY_CASE_FAILURE))
    decisions = sum(count*ir._costs(profile, 3, name, 0, cap)["decisions"] for name, count in counts.items())
    contribution = decisions*COMPLETE_TRIALS*profile.controller_failure_per_decision
    return {"controller_decisions_per_trial": decisions,
            "controller_decisions_all_trials": decisions*COMPLETE_TRIALS,
            "controller_failure_per_decision_scenario": profile.controller_failure_per_decision,
            "distance_independent_controller_model_contribution": contribution,
            "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
            "budget_exceeded": contribution > _downward(PHYSICAL_CASE_FAILURE),
            "factory_retry_cap": cap,
            "scope": "additive serialized engineering-model contribution; not a measured hardware fault lower bound"}


def estimate_prepared(prepared: PreparedWorkload, *, profile: ir.InstrumentProfile,
                      cache_dir: Path, include_all_feasible_points: bool = True) -> dict:
    if not isinstance(profile, ir.InstrumentProfile) or asdict(profile) not in [asdict(value) for value in fixed_profiles()]:
        raise ValueError("Only the two frozen physical error profiles with the default controller are supported")
    common = {"case_id": CASE_ID, "profile": asdict(profile), "complete_algorithm_attempts": COMPLETE_TRIALS,
              "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
              "factory_availability_budget": _downward(FACTORY_CASE_FAILURE),
              "family_allocations": family_allocations(), "run_caps": run_caps(),
              "comparison_scope": "same fixed literal n2 workload; separate serialized scenario, frozen Grover/PSSPC unchanged"}
    try:
        diagnostic = controller_diagnostic(prepared, profile)
    except ValueError as error:
        if "retry cap" not in str(error):
            raise
        return {**common, "status": "unsupported_factory_retry_model_domain", "pareto_points": [],
                "all_feasible_points": [], "reason": str(error)}
    if diagnostic["budget_exceeded"]:
        return {**common, "status": "infeasible_declared_controller_model", "pareto_points": [],
                "all_feasible_points": [], "controller_model_diagnostic": diagnostic,
                "reason": "distance-independent controller contribution exceeds the unchanged allocated physical budget"}
    try:
        result = ir.estimate_instrument(prepared.synthesized, COMPLETE_TRIALS, profile=profile,
                                        physical_failure_budget=_downward(PHYSICAL_CASE_FAILURE),
                                        factory_availability_budget=_downward(FACTORY_CASE_FAILURE),
                                        cache_dir=Path(cache_dir), include_all_feasible_points=include_all_feasible_points)
    except RuntimeError as error:
        if "No feasible point in the declared" not in str(error):
            raise
        return {**common, "status": "infeasible_declared_distance_factory_domain", "pareto_points": [],
                "all_feasible_points": [], "controller_model_diagnostic": diagnostic, "reason": str(error)}
    return {**result, **common, "controller_model_diagnostic": diagnostic}


def family_status_records(*, check_rejections: bool = True) -> dict:
    """Wider source-only rejection evidence; never synthesize or estimate them."""
    records = {"2": {"n_system": 2, "selected_steps": STEPS,
                      "status": "implemented_four_wire_density_slice",
                      "complete_trials": COMPLETE_TRIALS, "physical_confidence_certificate": False}}
    ensemble_limits = ir.InstrumentLimits(max_qubits=8, source_operations=20_000,
                                          synthesized_operations=500_000,
                                          validation_operations=500_000, validation_branches=256)
    for n, r in ((3, 134), (4, 211)):
        task = lw.ChainTask(n)
        source = build_chain_circuit(n, task.time, task.J, task.hx, task.rates,
                                     r, 2, "product", observable="Z")
        record = {"n_system": n, "selected_steps": r, "complete_trials": COMPLETE_TRIALS,
                  "raw_source_instructions": len(source.data), "source_counts": dict(source.count_ops()),
                  "limits": asdict(ensemble_limits), "physical_status": "unimplemented_reserved_allocation",
                  "synthesis_performed": False, "physical_estimation_performed": False,
                  "family_allocation_reallocated": False,
                  "scope": "literal fixed-input source-only CPU ensemble check; no width-only physical extrapolation"}
        if check_rejections:
            try:
                ir.simulate_scored_trial(source, (0,), (1., -1.), {}, limits=ensemble_limits)
            except ValueError as error:
                if "branch cap" not in str(error):
                    raise
                record.update(status="ensemble_branch_cap_exceeded", reason=str(error), rejection_reproduced=True)
            else:
                raise AssertionError("The selected wider literal schedule unexpectedly passed its frozen ensemble cap")
        else:
            record.update(status="ensemble_rejection_not_rechecked", rejection_reproduced=False)
        records[str(n)] = record
    return records
