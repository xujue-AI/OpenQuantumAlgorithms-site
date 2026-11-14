# SPDX-License-Identifier: MIT
"""Bounded literal four-system Lindblad physical workload.

Only the new n4/r211/Z0 workload is synthesized and assessed here. The prior
n2/n3 results and historical n4 rejection records are preserved. Density checks,
guarded synthesis, representation/SDK premises and physical model assumptions
are separate; the repeated populations and hardware schedules are unexecuted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

if __package__:
    from .. import instrument_resources as ir
    from . import lindblad_workload as lw
    from . import lindblad_four_density as di
    from . import lindblad_four_compile as cc
    from . import bounded_density_instrument as legacy_di
    from .two_spin_lindblad import build_chain_circuit
else:
    import instrument_resources as ir
    import lindblad_workload as lw
    import lindblad_four_density as di
    import lindblad_four_compile as cc
    import bounded_density_instrument as legacy_di
    from two_spin_lindblad import build_chain_circuit

CASE_ID = "lindblad-n4-r211-Z0"
STEPS = 211
COMPLETE_TRIALS = 15_756
SYNTHESIS_ALLOWANCE = 5e-5
REPRESENTATION_NORM_RESERVE = 1e-7
SDK_NORM_RESERVE = 1e-7
NUMERICAL_CHECK_GUARD = 1e-10
NUMERICAL_STATE_GUARD = 1e-8
STATISTICAL_CASE_FAILURE = Fraction(1, 600)
PHYSICAL_CASE_FAILURE = Fraction(1, 1200)
FACTORY_CASE_FAILURE = Fraction(1, 1200)
EXECUTION_ONLY_REPORT_FIELDS = frozenset({"native_rotation_synthesis_attempts",
                                         "native_rotation_synthesis_attempts_upper_bound"})


def limits() -> cc.FourCompilerLimits:
    """Separate explicit literal compiler domain; no shared limit is widened."""
    return cc.FourCompilerLimits(8, 20_000, 100_000, 2_000_000)


def checker_limits() -> di.EightWireDensityLimits:
    """The separate checker requires an explicit eight-wire opt-in."""
    return di.EightWireDensityLimits(8, 20_000, 2_000_000)


def score_values() -> tuple[float, ...]:
    """One Z0 score from the complete correlated four-system joint outcome."""
    return tuple(1.0 if value & 1 == 0 else -1.0 for value in range(16))


def run_caps() -> dict:
    """Predeclared limits enforced by this adapter and the bounded runner."""
    return {"physical_cases": 1, "prepared_synthesis_stages": 1,
            "native_rotation_synthesis_attempts_upper_bound": 4*5908,
            "native_rotation_synthesis_attempts_scope": "actual new-compiler gridsynth_unitary calls including up to four attempts per unique bound rotation target; at most 5908 targets",
            "physical_profiles": 2, "source_instructions": 20_000,
            "routed_instructions": 100_000, "synthesized_instructions": 2_000_000,
            "routed_and_synthesized_instructions": 2_000_000,
            "density_validation_instructions": 2_000_000, "density_wires": 8,
            "density_matrix_entries": 65536, "density_matrix_bytes": 1048576,
            "density_storage_scope": "largest retained eight-wire complex128 density matrix; not a measured process peak-memory claim",
            "unused_family_ensemble_branches": 256,
            "family_diagnostic_seconds": 180, "source_preflight_seconds": 180,
            "diagnostic_aggregate_seconds": 180,
            "prepare_seconds": 1800,
            "physical_seconds_per_profile": 180, "parent_seconds": 2400,
            "factory_queries_per_profile": 24, "surface_code_queries_per_profile": 24,
            "factory_candidates_per_query": 1000, "factory_candidates_per_profile": 24_000,
            "managed_json_qasm_bytes": 32*1024*1024,
            "private_checkpoint_bytes": 256*1024*1024,
            "nominal_state_trace_guard": NUMERICAL_STATE_GUARD,
            "nominal_density_comparison_guard": NUMERICAL_CHECK_GUARD,
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
    task = lw.ChainTask(4)
    if lw.select_steps(task) != STEPS or lw.output_contract()["complete_trials_per_case"] != COMPLETE_TRIALS:
        raise RuntimeError("The integrated uniform-accuracy task contract changed")
    return task


def _fixed_task(task: lw.ChainTask | None) -> lw.ChainTask:
    fixed = source_task()
    if task is None:
        return fixed
    if type(task) is not lw.ChainTask or asdict(task) != asdict(fixed):
        raise ValueError("Only literal n4/r211/Z0 at the fixed integrated parameters is implemented")
    return task


def _events(circuit: QuantumCircuit) -> list[tuple[str, int | None]]:
    return [(item.operation.name,
             circuit.find_bit(item.clbits[0]).index if item.clbits else None)
            for item in circuit.data if item.operation.name in {"reset", "measure"}]


def _complete_trial(circuit: QuantumCircuit) -> None:
    if circuit.num_qubits != 8 or circuit.num_clbits != 8:
        raise ValueError("The literal trial requires exactly eight quantum and classical wires")
    counts = circuit.count_ops()
    if counts.get("reset", 0) != 1481 or counts.get("measure", 0) != 8:
        raise ValueError("The literal complete trial must retain all 1481 resets and eight readouts")
    outputs = [circuit.find_bit(item.clbits[0]).index for item in circuit.data
               if item.operation.name == "measure"]
    if outputs != list(range(8)):
        raise ValueError("Complete readout must retain the declared classical bit order")
    measured_wires = [circuit.find_bit(item.qubits[0]).index for item in circuit.data
                      if item.operation.name == "measure"]
    if len(measured_wires) != 8 or set(measured_wires) != set(range(8)):
        raise ValueError("Complete final readout must measure all eight physical wires exactly once")


def build_source(task: lw.ChainTask | None = None) -> QuantumCircuit:
    task = _fixed_task(task)
    source = build_chain_circuit(4, task.time, task.J, task.hx, task.rates,
                                 STEPS, 2, "product", observable="Z")
    _complete_trial(source)
    if len(source.data) != 9934:
        raise AssertionError("Literal source schedule changed")
    return source


def family_allocations() -> dict:
    cases = {str(n): {"selected_steps": r, "complete_trials": COMPLETE_TRIALS,
                      "statistical_failure_exact": _fraction_record(STATISTICAL_CASE_FAILURE),
                      "physical_failure_exact": _fraction_record(PHYSICAL_CASE_FAILURE),
                      "factory_availability_failure_exact": _fraction_record(FACTORY_CASE_FAILURE),
                      "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
                      "factory_availability_budget": _downward(FACTORY_CASE_FAILURE),
                      "physical_status": ("preserved_prior_n2_result" if n == 2 else
                                          "preserved_prior_n3_result" if n == 3 else
                                          "new_bounded_n4_assessment_not_feasibility_claim")}
             for n, r in ((2, 69), (3, 134), (4, 211))}
    return {"declared_family": [2, 3, 4], "implemented_physical_cases": [2, 3],
            "new_physical_cases": [4], "preserved_prior_physical_cases": [2, 3],
            "statistics_family_failure_exact": _fraction_record(Fraction(1, 200)),
            "physical_family_failure_exact": _fraction_record(Fraction(1, 400)),
            "factory_family_failure_exact": _fraction_record(Fraction(1, 400)),
            "total_conditional_family_failure_exact": _fraction_record(Fraction(1, 100)),
            "case_allocations": cases, "unused_cases_reallocated": False,
            "family_physical_confidence_implemented": False,
            "scope": "one new n4 bounded numerical/compiler and fixed-model assessment; n2/n3 prior results and all exact family allocations are preserved; no full-family hardware-confidence claim"}


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


def _fixed_error_premises(synthesis_allowance, representation_reserve, sdk_reserve):
    if (synthesis_allowance != SYNTHESIS_ALLOWANCE or
            representation_reserve != REPRESENTATION_NORM_RESERVE or sdk_reserve != SDK_NORM_RESERVE):
        raise ValueError("The literal task retains the predeclared synthesis/representation/SDK requests")


def _annotate_check(check: dict) -> None:
    trace = check["acceptance_probability"]
    if not math.isfinite(trace) or abs(trace-1) > NUMERICAL_STATE_GUARD:
        raise AssertionError("No postselection is allowed; raw trace must remain one within the predeclared state guard")
    if check.get("normalization_is_postselection") is not False:
        raise AssertionError("The literal task has no algorithm postselection")
    if (check.get("unnormalized_trace_before_output") != trace or
            check.get("normalization_divisor") != trace):
        raise AssertionError("The raw trace and output-only normalization divisor must remain disclosed consistently")
    check["density_path"] = "explicit eight-wire density/CPTP checker; full four-system correlations, no reset-history outputs"
    check["normalization_probability_is_postselection"] = False
    check["nominal_state_trace_guard"] = NUMERICAL_STATE_GUARD
    check["nominal_density_comparison_guard"] = NUMERICAL_CHECK_GUARD
    rho = check["reduced_before_output"]
    probabilities = check["joint_probabilities"]
    if np.shape(probabilities) != (16,) or np.shape(rho) != (16, 16):
        raise AssertionError("All four correlated system outputs must remain in the joint density check")
    if (not np.all(np.isfinite(rho)) or not np.all(np.isfinite(probabilities)) or
            abs(complex(np.trace(rho))-1) > NUMERICAL_CHECK_GUARD or
            abs(float(np.sum(probabilities))-1) > NUMERICAL_CHECK_GUARD):
        raise AssertionError("The normalized reduced system and joint law must retain trace one numerically")
    if float(np.max(np.abs(np.diag(rho).real-probabilities))) > NUMERICAL_CHECK_GUARD:
        raise AssertionError("Joint probabilities must agree with the full retained density diagonal")
    if np.linalg.norm(rho-rho.conj().T, ord="nuc") > NUMERICAL_STATE_GUARD:
        raise AssertionError("The retained system density violates its predeclared Hermiticity guard")
    if min(np.linalg.eigvalsh((rho+rho.conj().T)/2)) < -NUMERICAL_STATE_GUARD:
        raise AssertionError("The retained system density violates its predeclared positivity guard")


def _routing_agreement(original: dict, routing: dict) -> None:
    residual = float(np.linalg.norm(original["reduced_before_output"]-routing["reduced_before_output"], ord="nuc"))
    joint = float(np.max(np.abs(original["joint_probabilities"]-routing["joint_probabilities"])))
    routing["source_reduced_output_trace_norm_residual"] = residual
    routing["source_joint_probability_maximum_absolute_residual"] = joint
    if (abs(original["mean"]-routing["mean"]) > NUMERICAL_CHECK_GUARD or
            residual > NUMERICAL_CHECK_GUARD or joint > NUMERICAL_CHECK_GUARD):
        raise AssertionError("Routing changed the full correlated four-system instrument")


def _correlation_witness(rho: np.ndarray) -> dict:
    singles = []
    for q in range(4):
        single = np.zeros((2, 2), dtype=complex)
        others = [k for k in range(4) if k != q]
        for rest in range(8):
            base = sum(((rest >> k) & 1) << wire for k, wire in enumerate(others))
            for a in (0, 1):
                for b in (0, 1):
                    single[a, b] += rho[base | (a << q), base | (b << q)]
        singles.append(single)
    product = singles[3]
    for q in (2, 1, 0):
        product = np.kron(product, singles[q])
    zz = []
    for q in range(3):
        observable = lw.on_wire(lw.Z, q, 4) @ lw.on_wire(lw.Z, q+1, 4)
        zz.append({"systems": [q, q+1], "source_ZZ": float(np.trace(observable@rho).real),
                   "product_of_marginals_ZZ": float(np.trace(observable@product).real)})
    z0 = lw.on_wire(lw.Z, 0, 4)
    return {"adjacent_ZZ": zz,
            "product_of_marginals_system_trace_norm_discrepancy": float(np.linalg.norm(rho-product, ord="nuc")),
            "source_minus_product_Z0": float(np.trace(z0@(rho-product)).real),
            "product_replaces_the_actual_state": False,
            "scope": "diagnostic explaining why one local mean cannot validate the interacting joint channel"}


def _reference_record(task: lw.ChainTask, original: dict, actual: dict | None = None) -> dict:
    exact = lw.reference_density(task)
    gate = lw.gate_density(task, STEPS)
    observable = lw.on_wire(lw.Z, 0, 4)
    exact_mean = float(np.trace(observable@exact).real)
    gate_mean = float(np.trace(observable@gate).real)
    residual = float(np.linalg.norm(original["reduced_before_output"]-gate, ord="nuc"))
    if abs(original["mean"]-gate_mean) > NUMERICAL_CHECK_GUARD or residual > NUMERICAL_CHECK_GUARD:
        raise AssertionError("Literal density instrument disagrees with the independent gate-derived full state")
    source_exact = float(np.linalg.norm(original["reduced_before_output"]-exact, ord="nuc"))
    splitting = _upward(lw.uniform_split_bound(task, STEPS))
    if source_exact > splitting+NUMERICAL_CHECK_GUARD:
        raise AssertionError("Literal source full density exceeds its uniform splitting proposal")
    reference = {"exact_GKSL_Z0": exact_mean, "gate_contracted_Z0": gate_mean,
                 "source_bias_numerical": abs(original["mean"]-exact_mean),
                 "source_vs_GKSL_system_trace_norm_residual": source_exact,
                 "source_vs_gate_contracted_Z0_residual": abs(original["mean"]-gate_mean),
                 "source_vs_gate_contracted_system_trace_norm_residual": residual,
                 "exact_GKSL_system_density": exact, "gate_contracted_system_density": gate,
                 "correlation_witness": _correlation_witness(original["reduced_before_output"]),
                 "scope": "bounded independent full GKSL and gate-derived reduced channels; not a representation/compiler-norm certificate"}
    if actual is not None:
        reference.update(source_vs_synthesized_system_trace_norm_residual=float(np.linalg.norm(original["reduced_before_output"]-actual["reduced_before_output"], ord="nuc")),
                         actual_vs_GKSL_system_trace_norm_residual=float(np.linalg.norm(actual["reduced_before_output"]-exact, ord="nuc")),
                         actual_clifford_t_Z0_bias_numerical=abs(actual["mean"]-exact_mean),
                         source_vs_synthesized_Z0_bias_numerical=abs(original["mean"]-actual["mean"]))
    return reference


@dataclass
class SourcePreflight:
    task: lw.ChainTask
    source: QuantumCircuit
    lowered: QuantumCircuit
    routed: QuantumCircuit
    source_check: dict
    routed_check: dict
    independent_reference: dict
    routing_report: dict

    def as_record(self) -> dict:
        return _jsonable({"case_id": CASE_ID, "task": asdict(self.task), "selected_steps": STEPS,
                          "status": "checked_literal_source_and_routed_rotations",
                          "source_counts": dict(self.source.count_ops()), "actual_routing": self.routing_report,
                          "source_instrument_check": self.source_check, "routed_instrument_check": self.routed_check,
                          "independent_reference": self.independent_reference,
                          "checker_limits": asdict(checker_limits()), "materialization_limits": asdict(limits()),
                          "limits": asdict(limits()), "output_clbits": [0, 1, 2, 3], "score_values": score_values(),
                          "representation_check": representation_record(self.source), "run_caps": run_caps(),
                          "family_allocations": family_allocations(), "synthesis_performed": False,
                          "physical_estimation_performed": False, "joint_system_correlations_preserved": True,
                          "unobserved_reset_histories_are_measured_outputs": False,
                          "hardware_execution": False, "formal_confidence_certificate": False})


def source_preflight(task: lw.ChainTask | None = None) -> SourcePreflight:
    """Run only bounded source/routing/full-density checks, never gridsynth."""
    task = _fixed_task(task)
    source = build_source(task)
    di.validate_density_source(source, limits=checker_limits())
    routed = cc.route_source(source, limits=limits())
    _complete_trial(routed.routed)
    if _events(routed.routed) != _events(source):
        raise AssertionError("Routing changed the literal reset/readout chronology")
    checks = [di.simulate_density_trial(value, (0, 1, 2, 3), score_values(), {}, limits=checker_limits())
              for value in (source, routed.routed)]
    for check in checks:
        _annotate_check(check)
    _routing_agreement(*checks)
    reference = _reference_record(task, checks[0])
    return SourcePreflight(task, source, routed.lowered, routed.routed, *checks, reference, routed.report)


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
    legacy_materialization_validation: dict
    preparation_fingerprint: str = ""

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
                          "materialization_limits": asdict(limits()), "limits": asdict(limits()),
                          "checker_limits": asdict(checker_limits()),
                          "legacy_materialization_validation": self.legacy_materialization_validation,
                          "preparation_fingerprint": self.preparation_fingerprint,
                          "execution_report_fields_excluded_from_preparation_fingerprint": sorted(EXECUTION_ONLY_REPORT_FIELDS),
                          "output_clbits": [0, 1, 2, 3], "score_values": score_values(),
                          "joint_system_correlations_preserved": True,
                          "unobserved_reset_histories_are_measured_outputs": False,
                          "run_caps": run_caps(), "family_allocations": family_allocations(),
                          "hardware_execution": False, "formal_confidence_certificate": False})


def _preparation_fingerprint(prepared: PreparedWorkload) -> str:
    """Bind completed checks to bounded actual operations and immutable evidence.

    This is an integrity attestation for the local verified checkpoint, not a
    mathematical certificate. Only two runner-added execution counters are
    excluded; physical/scientific report data and all returned checks are bound.
    """
    if type(prepared) is not PreparedWorkload or type(prepared.synthesized) is not cc.SynthesizedFourInstrument:
        raise ValueError("A verified literal PreparedWorkload is required")
    digest = hashlib.sha256()
    for name, circuit, maximum in (("source", prepared.synthesized.source, 20_000),
                                   ("routed", prepared.synthesized.routed, 100_000),
                                   ("clifford_t", prepared.synthesized.circuit, 2_000_000)):
        if not isinstance(circuit, QuantumCircuit) or len(circuit.data) > maximum:
            raise ValueError("Prepared circuit identity exceeds the predeclared operation domain")
        if circuit.num_qubits != 8 or circuit.num_clbits != 8:
            raise ValueError("Prepared circuit identity requires eight quantum and classical wires")
        phase = float(circuit.global_phase)
        if not math.isfinite(phase):
            raise ValueError("Prepared coherent phases must remain finite")
        layout = circuit.layout
        layout_record = None if layout is None else {"initial": layout.initial_index_layout(),
                                                     "final": layout.final_index_layout()}
        header = {"stage": name, "instructions": len(circuit.data), "name": circuit.name,
                  "qubits": circuit.num_qubits, "clbits": circuit.num_clbits,
                  "qregs": [(r.name, r.size) for r in circuit.qregs],
                  "cregs": [(r.name, r.size) for r in circuit.cregs],
                  "global_phase_hex": phase.hex(), "layout": layout_record}
        digest.update(json.dumps(header, sort_keys=True, allow_nan=False, separators=(",", ":")).encode())
        operation_tokens = {}
        for item in circuit.data:
            operation = item.operation
            parameters = tuple(float(value).hex() for value in operation.params)
            token = (type(operation).__module__, type(operation).__qualname__,
                     operation.base_class.__module__, operation.base_class.__qualname__,
                     operation.name, operation.num_qubits, operation.num_clbits,
                     getattr(operation, "ctrl_state", None), getattr(operation, "num_ctrl_qubits", None),
                     parameters, str(getattr(operation, "condition", None)))
            encoded = operation_tokens.get(token)
            if encoded is None:
                encoded = json.dumps(token, allow_nan=False, separators=(",", ":")).encode()
                if len(operation_tokens) < 256:
                    operation_tokens[token] = encoded
            quantum = [circuit.find_bit(bit).index for bit in item.qubits]
            classical = [circuit.find_bit(bit).index for bit in item.clbits]
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            digest.update(bytes([len(quantum), *quantum, len(classical), *classical]))
    evidence = {"task": asdict(prepared.task),
                "report": {key: value for key, value in prepared.synthesized.report.items()
                           if key not in EXECUTION_ONLY_REPORT_FIELDS},
                "source_check": prepared.source_check, "routed_check": prepared.routed_check,
                "synthesized_check": prepared.synthesized_check,
                "independent_reference": prepared.independent_reference,
                "contract": prepared.contract, "workload": prepared.workload,
                "legacy_materialization_validation": prepared.legacy_materialization_validation}
    raw = json.dumps(_jsonable(evidence), sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    if len(raw) > run_caps()["managed_json_qasm_bytes"]:
        raise ValueError("Prepared scientific evidence exceeds its bounded JSON domain")
    digest.update(raw)
    return digest.hexdigest()


def _require_verified_preparation(prepared: PreparedWorkload) -> None:
    if (type(prepared) is not PreparedWorkload or not isinstance(prepared.preparation_fingerprint, str) or
            len(prepared.preparation_fingerprint) != 64 or
            prepared.preparation_fingerprint != _preparation_fingerprint(prepared)):
        raise ValueError("A verified unmodified PreparedWorkload and matching preparation identity are required")


def check_synthesized(synthesized, task: lw.ChainTask | None = None, *,
                      synthesis_allowance: float = SYNTHESIS_ALLOWANCE,
                      representation_reserve: float = REPRESENTATION_NORM_RESERVE,
                      sdk_reserve: float = SDK_NORM_RESERVE) -> PreparedWorkload:
    _fixed_error_premises(synthesis_allowance, representation_reserve, sdk_reserve)
    if type(synthesized) is not cc.SynthesizedFourInstrument:
        raise ValueError("An actual separate-domain SynthesizedFourInstrument is required")
    task = _fixed_task(task)
    source, routed, circuit = synthesized.source, synthesized.routed, synthesized.circuit
    _complete_trial(source)
    if source != build_source(task):
        raise AssertionError("Synthesis source differs from the fixed literal schedule")
    for value in (routed, circuit):
        _complete_trial(value)
        if _events(value) != _events(source):
            raise AssertionError("Compilation changed the literal reset/readout event order")
    report = synthesized.report
    if report.get("compiler_domain") != "literal-n4-explicit-opt-in-v1" or report.get("materialization_limits") != asdict(limits()):
        raise AssertionError("Prepared n4 must retain its separate explicit compiler domain")
    if "limits" in report:
        raise AssertionError("The n4 compiler report must not impersonate shared InstrumentLimits")
    if (report.get("actual_clifford_t_synthesis_performed") is not True or
            report.get("actual_synthesis_sequences_generated") is not True):
        raise AssertionError("Actual Clifford+T synthesis must be marked completed before validation")
    if report.get("derivation") != cc.derivation_record():
        raise AssertionError("The compiler derivation must match the exact frozen source")
    if report.get("preserve_full_unitary") is not True or report.get("transpiler_qubits_initially_zero") is not False:
        raise AssertionError("This workload requires actual full-unitary compiler lowering")
    actual_rotations = sum(item.operation.name in {"rx", "ry", "rz"} for item in routed.data)
    if report.get("rotation_occurrences") != 5908 or actual_rotations != 5908:
        raise AssertionError("Full-unitary level0 lowering must allocate all 5908 literal rotations")
    if report.get("operator_error_allowance") != SYNTHESIS_ALLOWANCE:
        raise AssertionError("The actual compiler request must retain the predeclared synthesis allowance")
    if (report.get("source_counts") != dict(source.count_ops()) or
            report.get("routed_counts") != dict(routed.count_ops()) or
            report.get("clifford_t_counts") != dict(circuit.count_ops())):
        raise AssertionError("Actual source/routed/synthesized counts differ from the saved report")
    if (len(routed.data) > limits().routed_operations or len(circuit.data) > limits().synthesized_operations or
            set(circuit.count_ops()) - (ir.CLIFFORD_T | {"reset", "measure", "barrier"})):
        raise AssertionError("Actual trial must fit its explicit caps and Clifford+T instrument basis")
    cc._flat_basis_check(routed, rotations=True)
    cc._flat_basis_check(circuit, rotations=False)
    for name, value in (("source", source), ("routed", routed), ("clifford_t", circuit)):
        wire_key = f"{name}_final_measurement_wires"
        measured = [value.find_bit(item.qubits[0]).index for item in value.data if item.operation.name == "measure"]
        if report.get(wire_key) != measured:
            raise AssertionError("Reported final readout targets differ from the actual complete trial")
    if (report.get("source_global_phase") != float(source.global_phase) or
            report.get("routed_global_phase") != float(routed.global_phase) or
            report.get("compiled_global_phase") != float(circuit.global_phase)):
        raise AssertionError("Reported coherent-segment phases differ from the in-memory circuit phases")
    for value in (routed, circuit):
        for item in value.data:
            if item.operation.name == "cx":
                wires = [value.find_bit(bit).index for bit in item.qubits]
                if abs(wires[0]-wires[1]) != 1:
                    raise AssertionError("The actual routed schedule must contain neighboring CNOTs")
    actual_error = report["guarded_numerical_operator_error_upper_estimate"]
    contract = error_contract(actual_error)
    checks = [di.simulate_density_trial(value, (0, 1, 2, 3), score_values(), {}, limits=checker_limits())
              for value in (source, routed, circuit)]
    for check in checks:
        _annotate_check(check)
    original, routing, actual = checks
    _routing_agreement(original, routing)
    reference = _reference_record(task, original, actual)
    if (reference["source_vs_synthesized_Z0_bias_numerical"] > contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD or
            reference["source_vs_synthesized_system_trace_norm_residual"] > contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD):
        raise AssertionError("Synthesized correlated instrument exceeds the separated realization proposal")
    if (reference["actual_clifford_t_Z0_bias_numerical"] > contract["splitting_bias_upper"]+contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD or
            reference["actual_vs_GKSL_system_trace_norm_residual"] > contract["splitting_bias_upper"]+contract["realization_Z0_bias_upper_proposal"]+NUMERICAL_CHECK_GUARD):
        raise AssertionError("Actual full system answer exceeds the declared conditional bias contract")
    counts = {name: int(count) for name, count in circuit.count_ops().items() if name != "barrier"}
    allocations = family_allocations()["case_allocations"]["4"]
    workload = {"complete_trials": COMPLETE_TRIALS, "complete_attempts": COMPLETE_TRIALS,
                "algorithm_retry_cap": 1, "postselection": "none", "whole_workload_executed": False,
                "primitive_counts_per_complete_trial": counts,
                "primitive_counts_all_trials": {name: count*COMPLETE_TRIALS for name, count in counts.items()},
                "reset_per_trial": 1481, "readout_per_trial": 8,
                "source_resets_all_trials": 1481*COMPLETE_TRIALS,
                "source_readouts_all_trials": 8*COMPLETE_TRIALS,
                "workspace_startup_initializations_per_trial": 23,
                "workspace_startup_initializations_all_trials": 23*COMPLETE_TRIALS,
                "system_initializations_per_trial": 4,
                "source_first_wire_initializations_per_trial": 8, "source_reuse_resets_per_trial": 1473,
                "reserved_logical_patches": 31,
                "logical_startup_initializations_all_trials": 31*COMPLETE_TRIALS,
                "environment_initializations_per_trial": [422, 422, 422, 211],
                "discard_resets_before_environment_reuse_per_trial": [421, 421, 421, 210],
                "statistical_failure_exact": _fraction_record(STATISTICAL_CASE_FAILURE),
                "physical_failure_budget": allocations["physical_failure_budget"],
                "factory_availability_budget": allocations["factory_availability_budget"],
                "physical_failure_exact": _fraction_record(PHYSICAL_CASE_FAILURE),
                "factory_availability_failure_exact": _fraction_record(FACTORY_CASE_FAILURE),
                "conditional_case_failure_exact": _fraction_record(Fraction(1, 300)),
                "family_allocations_reallocated": False,
                "data_output_clbits": [0, 1, 2, 3], "ignored_environment_readout_clbits": [4, 5, 6, 7],
                "unobserved_reset_histories_claimed_as_outputs": False,
                "joint_output_score_protocol": "retain all four system outcome correlations; score depends only on the first bit",
                "trial_protocol": "fresh complete product preparation; every literal reset and all system/environment readouts charged",
                "deterministic_bias_multiplied_by_trials": False}
    prepared = PreparedWorkload(task, synthesized, original, routing, actual, reference, contract, workload,
                                cc.legacy_cap_check(circuit))
    prepared.preparation_fingerprint = _preparation_fingerprint(prepared)
    return prepared


def prepare_workload(task: lw.ChainTask | None = None, *, synthesis_allowance: float = SYNTHESIS_ALLOWANCE,
                     representation_reserve: float = REPRESENTATION_NORM_RESERVE,
                     sdk_reserve: float = SDK_NORM_RESERVE) -> PreparedWorkload:
    _fixed_error_premises(synthesis_allowance, representation_reserve, sdk_reserve)
    task = _fixed_task(task)
    source = build_source(task)
    di.validate_density_source(source, limits=checker_limits())
    synthesized = cc.synthesize_literal_four(source, SYNTHESIS_ALLOWANCE, limits=limits())
    return check_synthesized(synthesized, task)


def fixed_profiles() -> tuple[ir.InstrumentProfile, ir.InstrumentProfile]:
    return tuple(ir.InstrumentProfile(physical_error_rate=value) for value in (1e-3, 1e-4))


def controller_diagnostic(prepared: PreparedWorkload, profile: ir.InstrumentProfile) -> dict:
    counts = ir._event_counts(prepared.synthesized.circuit)
    counts["trial_control"] += 1
    counts["initial_reset"] += 23
    states = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    cap = ir._factory_retry_cap(states*COMPLETE_TRIALS, profile.factory_pipeline_failure_bound,
                                _downward(FACTORY_CASE_FAILURE))
    decisions = sum(count*ir._costs(profile, 3, name, 0, cap)["decisions"] for name, count in counts.items())
    contribution = decisions*COMPLETE_TRIALS*profile.controller_failure_per_decision
    return {"event_counts_per_trial": dict(counts), "required_magic_states_per_trial": states,
            "required_magic_states_all_trials": states*COMPLETE_TRIALS,
            "controller_decisions_per_trial": decisions,
            "controller_decisions_all_trials": decisions*COMPLETE_TRIALS,
            "controller_failure_per_decision_scenario": profile.controller_failure_per_decision,
            "distance_independent_controller_model_contribution": contribution,
            "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
            "maximum_allowed_controller_decisions_per_trial": math.floor(Fraction.from_float(_downward(PHYSICAL_CASE_FAILURE))/(COMPLETE_TRIALS*Fraction.from_float(profile.controller_failure_per_decision))) if profile.controller_failure_per_decision else None,
            "budget_exceeded": contribution > _downward(PHYSICAL_CASE_FAILURE), "factory_retry_cap": cap,
            "factory_readiness_failure_union_bound": float(Fraction(str(profile.factory_pipeline_failure_bound))**cap*states*COMPLETE_TRIALS),
            "scope": "additive serialized engineering-model contribution; not a measured hardware fault lower bound"}


def matched_task_proposal(diagnostic: dict, *, legacy_rejected: bool) -> dict:
    return {"status": "planned_matched_task_implementation_proposal", "task_changed": False,
            "accuracy_allocations_hardware_geometry_controller_changed": False,
            "proposal": "Build the explicit bounded estimate_verified_four validation adapter, then amortize data movement over each verified rotation block: move its data wire to wire0 with adjacent SWAPs, execute that same actual Clifford+T block at0, and reverse the SWAP chain. Preserve the baseline and validate the complete rewritten instrument before any native physical claim. A secondary bounded comparison may use the eight predeclared cyclic initial layouts.",
            "legacy_materialization_adapter_needed": legacy_rejected,
            "controller_decisions_per_trial_before": diagnostic["controller_decisions_per_trial"],
            "controller_decisions_per_trial_required_at_most": diagnostic["maximum_allowed_controller_decisions_per_trial"],
            "controller_decisions_per_trial_reduction_required_at_least": max(0, diagnostic["controller_decisions_per_trial"]-diagnostic["maximum_allowed_controller_decisions_per_trial"]),
            "candidate_layout_count_cap": 8,
            "planned_initial_layouts": [[(j+k) % 8 for j in range(8)] for k in range(8)],
            "identity_layout_is_existing_candidate": True,
            "planned_candidate_native_attempts_per_stage_upper_bound": 23632,
            "planned_candidate_native_attempts_total_upper_bound": 8*23632,
            "planned_candidate_stage_caps_seconds": {"source_validation": 180, "prepare_and_full_density": 1800, "physical_profile": 180, "parent": 2400},
            "planned_physical_profiles_per_candidate": 2,
            "planned_parent_seconds_per_candidate": 2400,
            "planned_aggregate_parent_seconds_upper_bound": 8*2400,
            "primary_block_porting_recipe": {
                "planned_generator_api": "port_verified_rotation_blocks_to_zero",
                "input": "verified baseline source/routed circuit plus the eight actual native blocks, their hashes and recorded in-memory phases",
                "rewrite_identity": "S R0 S^-1 = Rw, defining S as the adjacent SWAP chain taking data from wire0 to wirew; chronologically apply S^-1, R0, S",
                "rewrite_condition": "all physical wires0..w already have explicit initialization before this block; otherwise retain the original early block",
                "extra_neighbor_CX_per_rewritten_block": "6*w: three CX per adjacent SWAP, w outward and w restoring SWAPs",
                "factory_port_and_geometry": "unchanged wire0 interface on the same eight-wire bidirectional line and31 reserved patches",
                "approximants_error_allocation_and_phase": "reuse each exact verified native sequence and its phase; preserve all5908 occurrences and the same5e-5 whole-trial request",
                "source_events_and_register_mapping": "retain every preparation/reset/readout and restore all moved registers before the next source event",
                "free_initialization_or_quantum_occurrence_pruning": False,
                "native_synthesis_attempts_if_verified_blocks_reused": 0,
                "full_validation_required": "raw/normalized full eight-wire instrument and16x16/16-outcome system checks, actual added CX/T-wire/controller ledger, complete repetitions/factory retries and both frozen native profiles",
                "stage_caps_seconds": {"source_validation": 180, "rewrite_and_full_validation": 1800, "physical_profile": 180, "parent": 2400},
                "operation_limits": {"source": 20000, "preserved_baseline_routed_reference": 100000, "ported_clifford_t_and_validation": 2000000},
                "routed_limit_scope": "the100k limit applies to the preserved baseline routed reference; no new ported rotation-stage count is claimed, and every added CX belongs to the2m-capped ported CT trial",
                "implementation_executed": False, "feasibility_guaranteed": False},
            "native_validation_adapter": {"domain": "mandatory explicit eight-wire/source20k/routed100k/CT2m validation",
                "planned_api": "estimate_verified_four",
                "preserve_original_compiler_report": True, "shared_defaults_or_formulas_changed": False,
                "entry_checks": "complete actual counts, flat Clifford+T basis, all eight readouts/reset chronology, neighboring CX, raw/normalized full correlated density and exact budgets before native Trace"},
            "acceptance_criterion": "actual controller count within the unchanged share and complete physical/readiness candidate sums within their separate1/1200 budgets at odd distances3..49; controller improvement alone is insufficient",
            "validation_required": ["full raw and normalized correlated instrument agreement and unchanged accuracy premises",
                                    "actual source/oracle/preparation/synthesis/readout counts and T-wire positions",
                                    "all repetitions, transport, injection, controller and full capped factory retries",
                                    "independent audit of all conditional physical candidates under the frozen profiles"],
            "proposed_implementation_executed": False, "feasibility_claimed": False}


def estimate_prepared(prepared: PreparedWorkload, *, profile: ir.InstrumentProfile,
                      cache_dir: Path, include_all_feasible_points: bool = True) -> dict:
    if type(profile) is not ir.InstrumentProfile or asdict(profile) not in [asdict(value) for value in fixed_profiles()]:
        raise ValueError("Only the two frozen physical error profiles with the default controller are supported")
    _require_verified_preparation(prepared)
    common = {"case_id": CASE_ID, "profile": asdict(profile), "complete_algorithm_attempts": COMPLETE_TRIALS,
              "physical_failure_budget": _downward(PHYSICAL_CASE_FAILURE),
              "factory_availability_budget": _downward(FACTORY_CASE_FAILURE),
              "family_allocations": family_allocations(), "run_caps": run_caps(),
              "legacy_materialization_validation": prepared.legacy_materialization_validation,
              "new_compiler_domain": prepared.synthesized.report["compiler_domain"],
              "comparison_scope": "same fixed literal n4 task; unchanged serialized engineering scenario and frozen Grover/PSSPC baseline"}
    try:
        diagnostic = controller_diagnostic(prepared, profile)
    except ValueError as error:
        if "retry cap" not in str(error):
            raise
        return {**common, "status": "unsupported_factory_retry_model_domain", "pareto_points": [],
                "all_feasible_points": [], "reason": str(error), "native_estimation_performed": False}
    legacy_rejected = len(prepared.synthesized.circuit.data) > 1_000_000
    if diagnostic["budget_exceeded"]:
        return {**common, "status": "infeasible_declared_controller_model", "pareto_points": [],
                "all_feasible_points": [], "controller_model_diagnostic": diagnostic,
                "matched_task_implementation_proposal": matched_task_proposal(diagnostic, legacy_rejected=legacy_rejected),
                "native_estimation_performed": False,
                "reason": "distance-independent controller contribution exceeds the unchanged allocated physical budget"}
    if legacy_rejected:
        return {**common, "status": "unsupported_frozen_materialization_domain", "pareto_points": [],
                "all_feasible_points": [], "controller_model_diagnostic": diagnostic,
                "matched_task_implementation_proposal": matched_task_proposal(diagnostic, legacy_rejected=True),
                "native_estimation_performed": False,
                "reason": "the complete actual n4 trial exceeds the frozen shared estimator's one-million-operation validation domain"}
    # A separate attested view is allowed only after the actual trial fits every
    # unchanged shared limit. Preserve the new compiler report and original
    # materialization limits verbatim; never relabel a >1m schedule.
    bounds = ir.InstrumentLimits(max_qubits=8, source_operations=20_000,
                                 synthesized_operations=1_000_000, validation_operations=1_000_000,
                                 validation_branches=256)
    ir._source_check(prepared.synthesized.source, operation_limit=bounds.source_operations, max_qubits=bounds.max_qubits)
    ir._source_check(prepared.synthesized.routed, operation_limit=bounds.synthesized_operations, max_qubits=bounds.max_qubits)
    ir._source_check(prepared.synthesized.circuit, operation_limit=bounds.synthesized_operations, max_qubits=bounds.max_qubits)
    view_report = {**prepared.synthesized.report, "limits": asdict(bounds),
                   "validated_legacy_view": {"source_report_preserved": True,
                       "actual_source_routed_clifford_t_within_shared_limits": True,
                       "scope": "fresh validated shared-domain view; original new-domain report is unchanged"}}
    view = ir.SynthesizedInstrument(prepared.synthesized.source, prepared.synthesized.routed,
                                    prepared.synthesized.circuit, view_report)
    try:
        result = ir.estimate_instrument(view, COMPLETE_TRIALS, profile=profile,
                                        physical_failure_budget=_downward(PHYSICAL_CASE_FAILURE),
                                        factory_availability_budget=_downward(FACTORY_CASE_FAILURE),
                                        cache_dir=Path(cache_dir), include_all_feasible_points=include_all_feasible_points)
    except RuntimeError as error:
        if "No feasible point in the declared" not in str(error):
            raise
        return {**common, "status": "infeasible_declared_distance_factory_domain", "pareto_points": [],
                "all_feasible_points": [], "controller_model_diagnostic": diagnostic,
                "native_estimation_performed": True, "validated_legacy_view": view_report["validated_legacy_view"], "reason": str(error)}
    return {**result, **common, "controller_model_diagnostic": diagnostic,
            "native_estimation_performed": True, "validated_legacy_view": view_report["validated_legacy_view"]}


def family_status_records(*, check_rejections: bool = True) -> dict:
    """Keep n2/n3 native results and historical n4 negative API labels unchanged."""
    records = {
        "2": {"n_system": 2, "selected_steps": 69, "complete_trials": COMPLETE_TRIALS,
              "status": "preserved_prior_n2_result", "physical_status": "prior_result_preserved",
              "regenerated": False, "source_or_synthesis_rerun": False,
              "prior_manifest_path": "assets/generated/simulation/lindblad-physical/manifest.json",
              "prior_manifest_sha256": "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"},
        "3": {"n_system": 3, "selected_steps": 134, "complete_trials": COMPLETE_TRIALS,
              "status": "preserved_prior_n3_result", "physical_status": "prior_result_preserved",
              "regenerated": False, "source_or_synthesis_rerun": False,
              "prior_manifest_path": "assets/generated/simulation/lindblad-three/manifest.json",
              "prior_manifest_sha256": "9dde563b53b186272edb724de1b61865d497a2d00ea3368e05c8162b01d837f7"},
    }
    source = build_source()
    ensemble_limits = ir.InstrumentLimits(max_qubits=8, source_operations=20_000,
                                          synthesized_operations=1_000_000,
                                          validation_operations=1_000_000, validation_branches=256)
    old_density_bounds = legacy_di.DenseInstrumentLimits(max_qubits=6, source_operations=20_000,
                                                         validation_operations=1_000_000)
    record = {"n_system": 4, "selected_steps": STEPS, "complete_trials": COMPLETE_TRIALS,
              "raw_source_instructions": len(source.data), "source_counts": dict(source.count_ops()),
              "limits": asdict(ensemble_limits),
              "physical_status": "historical_unimplemented_reserved_allocation_preserved",
              "new_n4_assessment_status": "separate_explicit_eight_wire_assessment",
              "legacy_synthesis_performed": False, "legacy_physical_estimation_performed": False,
              "family_allocation_reallocated": False,
              "status_scope": "historical rejected APIs; new n4 source/synthesis/model assessment is recorded separately"}
    try:
        legacy_di.validate_density_source(source, limits=old_density_bounds)
    except ValueError as error:
        if "width" not in str(error):
            raise
        record.update(density_checker_status="density_width_cap_exceeded", density_checker_reason=str(error),
                      density_checker_limits=asdict(old_density_bounds), density_required_wires=8,
                      density_supported_wires=6, density_width_rejection_reproduced=True)
    else:
        raise AssertionError("The n4 literal source unexpectedly passed the historical six-wire density cap")
    if check_rejections:
        try:
            ir.simulate_scored_trial(source, (0,), (1., -1.), {}, limits=ensemble_limits)
        except ValueError as error:
            if "branch cap" not in str(error):
                raise
            record.update(status="ensemble_branch_cap_exceeded", reason=str(error), rejection_reproduced=True)
        else:
            raise AssertionError("The n4 literal schedule unexpectedly passed its frozen ensemble cap")
    else:
        record.update(status="ensemble_rejection_not_rechecked", rejection_reproduced=False)
    records["4"] = record
    return records
