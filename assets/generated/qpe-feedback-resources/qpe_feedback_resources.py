# SPDX-License-Identifier: MIT
"""Bounded, causal feedback-QPE synthesis and common physical provisioning.

The executable instrument and the all-corrections accounting envelope are
different artifacts. Only local, previously measured single-bit corrections
are supported. No machine is selected after an outcome is observed.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Clbit, IfElseOp
from qiskit.quantum_info import Statevector

from .instrument_resources import (CLIFFORD_T, InstrumentLimits, InstrumentProfile,
                                   SynthesizedInstrument, _ceil_div, _event_counts,
                                   _pareto, estimate_instrument, synthesize_instrument)
from .qft_qpe import build_iterative_qpe, build_qpe, phase_gate_example

LIMITS = InstrumentLimits(max_qubits=5, source_operations=10_000,
                          synthesized_operations=100_000,
                          validation_operations=100_000, validation_branches=256)
MAX_FEEDBACK_BITS = 4
MARKER = "qpe_feedback_correction_"


@dataclass(frozen=True)
class PhaseTask:
    phase: float = .3
    phase_bits: int = 3
    repetitions: int = 15
    operator_error_allowance: float = 2e-5
    physical_failure_budget: float = .0025
    factory_availability_budget: float = .0025

    def __post_init__(self):
        phase_gate_example(self.phase)
        if isinstance(self.phase_bits, bool) or not isinstance(self.phase_bits, int) or self.phase_bits not in (3, 4):
            raise ValueError("Physical comparison is bounded to m=3 or4, n=1")
        if self.repetitions != 15 or not isinstance(self.repetitions, int) or isinstance(self.repetitions, bool):
            raise ValueError("This frozen comparison charges15 complete independent runs")
        tolerance = 2.0**-self.phase_bits
        if not tolerance <= self.phase <= 1-tolerance:
            raise ValueError("Scalar-median good interval must not cross the phase branch cut")
        if self.operator_error_allowance != 2e-5 or self.physical_failure_budget != .0025 or self.factory_availability_budget != .0025:
            raise ValueError("Use the frozen synthesis/physical/availability budgets")

    def contract(self):
        bad = 1-8/math.pi**2
        tail = math.fsum(math.comb(15, k)*bad**k*(1-bad)**(15-k) for k in range(8, 16))
        return {**asdict(self), "target_qubits": 1,
                "output": "scalar median of15 decoded y/2**m estimates",
                "absolute_phase_error_target": 2.0**-self.phase_bits,
                "phase_branch_cut_condition": "2**(-m) <= phi <=1-2**(-m)",
                "ideal_single_run_good_probability_lower_bound": 8/math.pi**2,
                "ideal_majority_failure_upper_bound": tail,
                "synthesis_full_workload_tv_upper_estimate": 15*self.operator_error_allowance,
                "conditional_failure_upper_estimate": tail+15*self.operator_error_allowance+.005,
                "confidence_status": "ideal theorem plus numerical synthesis and conditional physical scenarios; no hardware/formal confidence certificate",
                "fresh_complete_preparations": 15,
                "stored_transcript_bits": 15*self.phase_bits,
                "controlled_unitary_calls_per_run": 2**self.phase_bits-1,
                "median_controller_decisions_all_runs": median_controller_decisions(self.phase_bits),
                "classical_processing": "15*m bit decoding slots,105 fixed bubble-network comparators each with m bit-comparison plus3 word-write slots, and1 median selection; every slot uses the declared controller scenario"}


def median_controller_decisions(phase_bits):
    return 15*phase_bits+105*(phase_bits+3)+1


def decode_median_network(transcripts, phase_bits):
    """Fixed105-comparator classical network, padding all m comparison bits."""
    if len(transcripts) != 15 or any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < 2**phase_bits for y in transcripts):
        raise ValueError("Median requires15 valid m-bit transcript integers")
    values = list(transcripts)
    for high in range(14, 0, -1):
        for index in range(high):
            greater, equal = False, True
            for bit in reversed(range(phase_bits)):
                a, b = (values[index] >> bit) & 1, (values[index+1] >> bit) & 1
                greater = greater or (equal and a > b)
                equal = equal and a == b
            temporary = values[index]
            values[index] = values[index+1] if greater else values[index]
            values[index+1] = temporary if greater else values[index+1]
    return values[7]/2**phase_bits


def build_sources(task: PhaseTask):
    """Fresh explicit initialization followed by the unchanged QPE generators."""
    unitary, preparation = phase_gate_example(task.phase)
    result = {}
    for name, body in (("textbook", build_qpe(task.phase_bits, unitary, preparation, measure=True)),
                       ("feedback", build_iterative_qpe(task.phase_bits, unitary, preparation))):
        source = body.copy_empty_like(name=f"complete {name} QPE run")
        source.reset(range(source.num_qubits))
        source.compose(body, inplace=True)
        result[name] = source
    return result


def correction_envelope(source: QuantumCircuit):
    """Flatten only one-wire corrections, retaining their independent barriers.

    This envelope is accounting input, not an executable substitute for QPE.
    Every predicate must refer to a prior, uniquely written measurement bit.
    """
    if source.num_qubits != 2 or not 1 <= source.num_clbits <= MAX_FEEDBACK_BITS:
        raise ValueError("Feedback synthesis supports exactly two wires and at most4 output bits")
    if len(source.data) > LIMITS.source_operations or source.num_parameters:
        raise ValueError("Feedback source exceeds its bounded materialization domain")
    envelope = source.copy_empty_like(name="all-corrections accounting envelope")
    envelope.global_phase = source.global_phase
    records, written = [], set()
    for item in source.data:
        op = item.operation
        if op.name == "measure":
            bit = source.find_bit(item.clbits[0]).index
            if bit in written:
                raise ValueError("Measurement bits may not be overwritten")
            written.add(bit)
        if op.name == "if_else":
            if op.base_class is not IfElseOp or len(op.blocks) != 1:
                raise ValueError("Only canonical one-arm single-bit feedback is supported")
            bit, value = op.condition
            if not isinstance(bit, Clbit) or bit not in source.clbits or value != 1:
                raise ValueError("Only previously recorded single-bit equality to1 is supported")
            index = source.find_bit(bit).index
            if index not in written:
                raise ValueError("Feedback predicate precedes its measurement")
            block = op.blocks[0]
            if len(item.qubits) != 1 or source.find_bit(item.qubits[0]).index != 0:
                raise ValueError("Only local phase-ancilla corrections are supported")
            if any(g.operation.name not in CLIFFORD_T | {"p", "rx", "ry", "rz"} or
                   len(g.qubits) != 1 or g.clbits for g in block.data):
                raise ValueError("Correction bodies must be explicit local unitaries")
            label = f"{MARKER}{len(records)}"
            envelope.barrier(label=label+"_begin")
            envelope.compose(block, qubits=[0], clbits=list(item.clbits), inplace=True)
            envelope.barrier(label=label+"_end")
            records.append({"marker": label, "condition_clbit": index,
                            "condition_value": 1, "source_counts": dict(block.count_ops())})
        else:
            if op.name in {"while_loop", "for_loop", "switch_case", "store", "initialize", "delay"} or getattr(op, "condition", None) is not None:
                raise ValueError("Unsupported source control flow")
            envelope.append(op, item.qubits, item.clbits)
    if not records:
        raise ValueError("A feedback instrument must contain an actual conditional correction")
    return envelope, records


def reconstruct_feedback(synthesized: SynthesizedInstrument, records):
    """Restore actual conditional bodies from the independently compiled slots."""
    compiled = synthesized.circuit
    if compiled.num_qubits != 2:
        raise ValueError("A common two-wire layout is required")
    output = compiled.copy_empty_like(name="actual Clifford+T feedback QPE")
    output.global_phase = compiled.global_phase
    expected = {r["marker"]: r for r in records}
    seen, active, body, recovered = set(), None, [], []
    for item in compiled.data:
        op = item.operation
        label = op.label if op.name == "barrier" else None
        if label and label.startswith(MARKER):
            marker, suffix = label.rsplit("_", 1)
            if marker not in expected or suffix not in {"begin", "end"}:
                raise ValueError("Malformed synthesis marker")
            if suffix == "begin":
                if active is not None or marker in seen:
                    raise ValueError("Nested or duplicate correction synthesis markers")
                active, body = marker, []
            else:
                if marker != active:
                    raise ValueError("Correction synthesis markers are unbalanced")
                record = expected[marker]
                with output.if_test((output.clbits[record["condition_clbit"]], 1)):
                    for gate in body:
                        output.append(gate.operation, gate.qubits, gate.clbits)
                counts = Counter(g.operation.name for g in body)
                recovered.append({**record, "clifford_t_counts": dict(counts),
                                  "event_counts": {("t_q0" if k == "t" else "tdg_q0" if k == "tdg" else k): v for k, v in counts.items()}})
                seen.add(marker)
                active, body = None, []
            continue
        if active is not None:
            if op.name == "barrier":
                continue
            if op.name not in CLIFFORD_T or len(item.qubits) != 1 or item.clbits or compiled.find_bit(item.qubits[0]).index != 0:
                raise ValueError("Compiled correction changed the common layout or instrument")
            body.append(item)
        else:
            output.append(op, item.qubits, item.clbits)
    if active is not None or seen != set(expected):
        raise ValueError("Synthesis lost a correction marker")
    def events(circuit):
        return [(g.operation.name, circuit.find_bit(g.qubits[0]).index,
                 circuit.find_bit(g.clbits[0]).index if g.clbits else None)
                for g in circuit.data if g.operation.name in {"reset", "measure"}]
    if events(output) != events(synthesized.source):
        raise ValueError("Compilation changed reset/readout boundaries or fixed wire mapping")
    return output, recovered


def synthesize_feedback(source, allowance=2e-5):
    envelope, records = correction_envelope(source)
    synthesized = synthesize_instrument(envelope, allowance, limits=LIMITS,
                                       preserve_full_unitary=True)
    actual, recovered = reconstruct_feedback(synthesized, records)
    return synthesized, actual, recovered


def measured_distribution(circuit: QuantumCircuit):
    """Bounded actual instrument enumeration with no nonzero-branch pruning.

    Canonical reset projects both outcomes and applies X to outcome1. The
    target remains in every branch throughout preparation/control/readout.
    All branches are kept unless their computed probability is exactly zero.
    """
    if not 1 <= circuit.num_qubits <= LIMITS.max_qubits or not 1 <= circuit.num_clbits <= MAX_FEEDBACK_BITS:
        raise ValueError("Measured validation exceeds its width/record cap")
    def expanded(block):
        return sum(1+sum(expanded(b) for b in g.operation.blocks) if g.operation.name == "if_else" else 1 for g in block.data)
    if expanded(circuit) > LIMITS.validation_operations:
        raise ValueError("Measured validation exceeds its complete operation cap")
    branches = [(Statevector.from_int(0, 2**circuit.num_qubits), (0,)*circuit.num_clbits)]
    indices = np.arange(2**circuit.num_qubits)
    def run(block, qmap, cmap, incoming):
        current = incoming
        for item in block.data:
            op = item.operation
            qs = [qmap[block.find_bit(q).index] for q in item.qubits]
            cs = [cmap[block.find_bit(c).index] for c in item.clbits]
            following = []
            for state, record in current:
                if op.name == "barrier":
                    following.append((state, record))
                elif op.name == "if_else":
                    bit, value = op.condition
                    chosen = 0 if record[cmap[block.find_bit(bit).index]] == value else 1
                    if chosen < len(op.blocks):
                        following.extend(run(op.blocks[chosen], qs, cs, [(state, record)]))
                    else:
                        following.append((state, record))
                elif op.name in {"measure", "reset"}:
                    for outcome in (0, 1):
                        vector = state.data.copy()
                        vector[((indices >> qs[0]) & 1) != outcome] = 0
                        if not np.vdot(vector, vector).real > 0:
                            continue
                        if op.name == "reset" and outcome:
                            vector = vector[indices ^ (1 << qs[0])]
                        updated = list(record)
                        if op.name == "measure":
                            updated[cs[0]] = outcome
                        following.append((Statevector(vector), tuple(updated)))
                else:
                    if cs or getattr(op, "condition", None) is not None:
                        raise ValueError("Unsupported classical operation in validation")
                    following.append((state.evolve(op, qs), record))
            current = following
            if len(current) > LIMITS.validation_branches:
                raise ValueError("Actual instrument branches exceed256")
        return current
    branches = run(circuit, list(range(circuit.num_qubits)), list(range(circuit.num_clbits)), branches)
    probabilities = np.zeros(2**circuit.num_clbits)
    for state, record in branches:
        probabilities[sum(b << i for i, b in enumerate(record))] += float(np.vdot(state.data, state.data).real)
    if not math.isclose(float(probabilities.sum()), 1, abs_tol=1e-10):
        raise AssertionError("Actual branch probability mass was not preserved")
    return probabilities


def add_feedback_exposure(resource, predicate_count, *, postprocessing_decisions=0):
    """Add supported controller macros and *full-device* idle exposure.

    Base native queries contain every optional synthesized gate. Adding time
    and error can only remove feasible machines. The all-feasible candidate
    search is therefore sufficient; no branch-specific native query is needed.
    """
    if isinstance(predicate_count, bool) or not isinstance(predicate_count, int) or predicate_count < 0:
        raise ValueError("Predicate count must be a nonnegative integer")
    if isinstance(postprocessing_decisions, bool) or not isinstance(postprocessing_decisions, int) or postprocessing_decisions < 0:
        raise ValueError("Postprocessing decision count must be a nonnegative integer")
    report = deepcopy(resource)
    profile = InstrumentProfile(**report["profile"])
    attempts = report["complete_algorithm_attempts"]
    candidates = report.pop("all_feasible_points", None)
    if candidates is None:
        raise ValueError("Common provisioning requires the complete feasible candidate set")
    feasible = []
    for point in candidates:
        elapsed = predicate_count*profile.reaction_time_ns
        runtime = point["runtime_per_complete_attempt_ns"]+elapsed
        cycles = _ceil_div(runtime, profile.code_cycle_time_ns)
        memory = point["reserved_logical_patches"]*(cycles-point["code_cycles_per_complete_attempt"])*profile.patch_failure_per_cycle(point["data_code_distance"])*attempts
        controller = predicate_count*profile.controller_failure_per_decision*attempts
        point["runtime_per_complete_attempt_ns"] = runtime
        post_time = postprocessing_decisions*profile.reaction_time_ns
        point["worst_case_runtime_all_attempts_ns"] = runtime*attempts+post_time
        point["code_cycles_per_complete_attempt"] = cycles
        components = point["failure_components_all_attempts"]
        components["data_workspace_code_exposure"] += memory
        components["controller_scenario"] += controller
        post_memory = point["reserved_logical_patches"]*_ceil_div(post_time, profile.code_cycle_time_ns)*profile.patch_failure_per_cycle(point["data_code_distance"])
        components["data_workspace_code_exposure"] += post_memory
        components["controller_scenario"] += postprocessing_decisions*profile.controller_failure_per_decision
        point["full_physical_failure_model"] = math.fsum(components.values())
        point["macro_ledger_per_complete_attempt"].append({"operation": "feedback_control", "count": predicate_count,
            "time_ns": profile.reaction_time_ns, "readouts": 0, "preparations": 0,
            "decisions": 1, "event_failure_model": profile.controller_failure_per_decision})
        point["feedback_adapter"] = {"predicate_count": predicate_count,
            "added_all_device_idle_time_ns_per_attempt": elapsed,
            "added_code_exposure_all_attempts": memory, "added_controller_failure_all_attempts": controller,
            "backend": "exact additive existing trial_control macro and ceil-cycle exposure after QDK direct envelope Trace"}
        point["median_postprocessing_all_runs"] = {
            "controller_decisions": postprocessing_decisions, "time_ns": post_time,
            "code_exposure_failure_model": post_memory,
            "controller_failure_model": postprocessing_decisions*profile.controller_failure_per_decision,
            "scope": "one fixed classical median network after all runs; all reserved patches remain charged during processing"}
        point["base_envelope_accounting_trace_sha256"] = point.pop("accounting_trace_sha256", None)
        point["native_backend"] = "base QDK direct instrument/envelope Trace plus explicit additive source-controller/median/idle ledger; augmented schedule is not the base native trace"
        point["augmented_macro_ledger_sha256"] = hashlib.sha256(json.dumps(
            point["macro_ledger_per_complete_attempt"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        point["common_factory_configuration"] = {
            "data_code_distance": point["data_code_distance"],
            "factory_physical_qubits": point.get("factory_physical_qubits"),
            "factory_supply": deepcopy(point["factory_supply"]),
            "qdk_version": "1.34.0", "physical_profile": asdict(profile),
            "scope": "complete effective configuration of this serialized factory model; fixed before any transcript"}
        if point["full_physical_failure_model"] <= report["physical_failure_budget"]:
            feasible.append(point)
    if not feasible:
        raise RuntimeError("No common envelope machine survives explicit feedback exposure")
    report["pareto_points"] = _pareto(feasible)
    report["feasible_candidate_count"] = len(feasible)
    report["source_feedback_predicates_per_run"] = predicate_count
    report["status"] = ("computed_conditional_common_feedback_envelope_surface_code_model" if predicate_count
                        else "computed_conditional_complete_static_qpe_surface_code_model")
    if predicate_count:
        report["unsupported"] = ["arbitrary branching, multiwire corrections, loops, adaptive stopping and target replacement"]
        report["assumptions"].extend([
        "Each predicate, including false outcomes, uses one charged controller reaction and decision",
        "Every optional correction owns a worst-case padded slot; false bodies idle all reserved data/workspace patches",
        "One fixed distance, complete factory configuration and full-pipeline retry cap cover every possible transcript",
        "On early factory success the state is immediately injected and remaining supply time is padded afterward; no uncharged magic-port holding is assumed",
        "If a required factory state is unavailable after its common cap, the complete task fails under the separately budgeted availability event",
            "The all-corrections envelope overcounts optional injection/readout/controller faults, bounding every supported branch"])
    if postprocessing_decisions:
        report["assumptions"].append("The fixed m-bit105-comparator median network uses one declared reaction/decision slot per bit comparison or word write; one postprocessing pass is charged after all runs with reserved-patch code exposure")
        report["assumptions"].append("Classical transcript storage, Boolean comparison logic and dispatch reliability are included in the assumed per-decision controller failure scenario; no independent idle classical-memory or controller-hardware calibration is supplied")
    report["excluded"] = [item.replace("and final statistical reporting wall time", "and presentation-only final reporting wall time") for item in report.get("excluded", [])]
    return report


def estimate_feedback(envelope, corrections, task, *, profile, cache_dir: Path):
    if envelope.circuit.num_clbits != task.phase_bits:
        raise ValueError("Task and compiled envelope must share the same output record domain")
    base = estimate_instrument(envelope, task.repetitions, profile=profile,
        physical_failure_budget=task.physical_failure_budget,
        factory_availability_budget=task.factory_availability_budget,
        cache_dir=cache_dir, include_all_feasible_points=True)
    report = add_feedback_exposure(base, len(corrections), postprocessing_decisions=median_controller_decisions(task.phase_bits))
    report["branch_coverage"] = {"allowed_transcripts": list(range(2**task.phase_bits)),
        "branch_pruning_for_provisioning": False, "all_true_transcript": 2**task.phase_bits-1,
        "uniform_synthesis_error_upper_estimate": envelope.report["guarded_numerical_operator_error_upper_estimate"],
        "conditional_corrections": corrections,
        "worst_case_counts_per_complete_run": dict(_event_counts(envelope.circuit)),
        "expected_cost": "not reported; fixed padded schedule has the same reserved wall time for every transcript"}
    return report


def branch_event_counts(envelope, corrections, phase_bits):
    """Cover *all* records, including zero-probability ideal transcripts."""
    maximum = _event_counts(envelope.circuit)
    result = []
    for transcript in range(2**phase_bits):
        counts = maximum.copy()
        selected = []
        for correction in corrections:
            if (transcript >> correction["condition_clbit"]) & 1:
                selected.append(correction["marker"])
            else:
                counts.subtract(correction["event_counts"])
        if any(v < 0 or v > maximum[k] for k, v in counts.items()):
            raise AssertionError("A branch escaped the common componentwise envelope")
        result.append({"transcript": transcript, "selected_corrections": selected,
                       "event_counts": {k: v for k, v in counts.items() if v},
                       "feedback_control": len(corrections)})
    return result
