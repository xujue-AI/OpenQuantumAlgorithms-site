# SPDX-License-Identifier: MIT
"""Conditional serialized-instrument baseline for two one-spin Taylor tasks.

The existing five-wire instrument model is reused unchanged. Actual synthesis
sequences, complete readout laws and bounded native profile searches are saved.
Numerical operator guards and physical/controller/interface models are
conditional assumptions, not interval or hardware-success certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
import multiprocessing as mp
from numbers import Real
from pathlib import Path
import time
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

if __package__:
    from . import instrument_resources as instruments
    from .simulation import word_taylor_workloads as workloads
else:
    import instrument_resources as instruments
    import word_taylor_workloads as workloads

CASES = {"size-n1-native": "native",
         "size-n1-normalization2-oaa1": "normalization2-oaa1"}
PHYSICAL_ERROR_PROFILES = (1e-3, 1e-4)
NUMERICAL_ROUTED_ACCEPTED_GUARD = Fraction(1, 10**10)
NUMERICAL_OPERATOR_GUARD = 1e-10
PHYSICAL_FAILURE_BUDGET = Fraction(1, 400)
FACTORY_READINESS_BUDGET = Fraction(1, 400)
DEFAULT_SYNTHESIS_TIMEOUT = 60.0
DEFAULT_QUERY_TIMEOUT = 60.0


@dataclass(frozen=True)
class TaylorPhysicalContract:
    case_id: str
    plan: Any
    accepted_amplitude_lower: Fraction
    total_norm_allowance: Fraction
    synthesis_allowance_float: float
    post_synthesis_acceptance_lower: Fraction
    retry_cap: int

    @property
    def complete_attempts(self) -> int:
        return self.plan.shots*self.retry_cap

    def record(self):
        frac = workloads.fraction_record
        return {
            "case_id": self.case_id, "original_ideal_plan": self.plan.record(),
            "original_ideal_retry_cap_not_used_for_physical_cost": self.plan.attempts_per_slot,
            "accepted_amplitude_lower_before_synthesis": frac(self.accepted_amplitude_lower),
            "complete_attempt_accepted_map_norm_allowance": frac(self.total_norm_allowance),
            "routed_accepted_map_numerical_construction_guard": frac(NUMERICAL_ROUTED_ACCEPTED_GUARD),
            "actual_requested_synthesis_operator_allowance": self.synthesis_allowance_float,
            "actual_requested_synthesis_operator_allowance_binary": frac(
                Fraction.from_float(self.synthesis_allowance_float)),
            "post_synthesis_acceptance_lower_if_guards_hold": frac(self.post_synthesis_acceptance_lower),
            "reserved_synthesis_conditional_Pauli_bias": frac(self.plan.total_accuracy/5),
            "accepted_slots": self.plan.shots, "synthesis_adjusted_retry_cap": self.retry_cap,
            "charged_complete_algorithm_attempts": self.complete_attempts,
            "failure_allocation": {
                "statistical": frac(self.plan.statistical_failure),
                "no_acceptance_across_slots": frac(self.plan.acceptance_failure),
                "physical_faults": frac(PHYSICAL_FAILURE_BUDGET),
                "factory_readiness": frac(FACTORY_READINESS_BUDGET),
                "total": frac(self.plan.statistical_failure+self.plan.acceptance_failure+
                              PHYSICAL_FAILURE_BUDGET+FACTORY_READINESS_BUDGET),
            },
            "accuracy_allocation": {"algorithm_bias": frac(self.plan.total_accuracy/5),
                                    "synthesis_bias": frac(self.plan.total_accuracy/5),
                                    "statistical_error": frac(self.plan.statistical_error),
                                    "total": frac(self.plan.total_accuracy)},
            "synthesis_contract": "If routed accepted-block construction error <= guard and coherent synthesis error <= its allowance, conditional bias <=4 eta/(a-eta) and acceptance >=(a-eta)^2.",
            "guard_scope": "The final routed native/OAA accepted block is checked directly, including routing and floating preparation; its numerical guard is an assumption.",
            "no_error_double_counting": "Synthesis is one per-attempt mean-bias allocation, not a stochastic failure and not multiplied by accepted slots.",
            "hardware_confidence_certified": False, "sampling_performed": False,
        }


def physical_contract(case_id: str) -> TaylorPhysicalContract:
    if case_id not in CASES:
        raise ValueError("only the two predeclared one-system Taylor cases are supported")
    plan = workloads.plan_word_taylor(1, total_time=1, accuracy=Fraction(1, 20), mode=CASES[case_id])
    if plan.segments != 1:
        raise RuntimeError("the fixed one-spin contract changed its one-segment schedule")
    if plan.mode == "native":
        amplitude = ((1-plan.bounds["composition_comparison_error"]) /
                     plan.bounds["native_taylor_normalization"]**plan.segments)
    else:
        amplitude = (1-plan.bounds["segment_comparison_operator_error"])**plan.segments
    synthesis_bias = plan.total_accuracy/5
    total = synthesis_bias*amplitude/(4+synthesis_bias)
    request_exact = total-NUMERICAL_ROUTED_ACCEPTED_GUARD
    request = float(request_exact)
    if Fraction.from_float(request) > request_exact:
        request = math.nextafter(request, 0)
    if request <= 0:
        raise RuntimeError("the numerical construction reserve exhausts synthesis allowance")
    p = (amplitude-total)**2
    retry_p = workloads.dyadic_lower(p)
    cap = workloads.ceil_fraction(workloads.log_upper(Fraction(plan.shots)/plan.acceptance_failure)/
                                  workloads.log_lower(1/(1-retry_p)))
    return TaylorPhysicalContract(case_id, plan, amplitude, total, request, p, cap)


def _prefix_and_output_order(circuit: QuantumCircuit):
    """Validate r=1 initial-reset/final-readout law and restore classical output order."""
    n = circuit.num_qubits
    if n not in (3, 5) or circuit.num_clbits != n:
        raise ValueError("the baseline needs three or five wires with complete final readout")
    initialized, measured = set(), {}
    started = False
    prefix = QuantumCircuit(n)
    prefix.global_phase = circuit.global_phase
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        wires = [circuit.find_bit(q).index for q in item.qubits]
        if name == "reset":
            if started or measured:
                raise ValueError("this fixed one-segment law requires only initial resets")
            initialized.update(wires)
        elif name == "measure":
            if initialized != set(range(n)):
                raise ValueError("every reserved input wire must be explicitly initialized")
            wire, classical = wires[0], circuit.find_bit(item.clbits[0]).index
            if wire in measured or classical in measured.values():
                raise ValueError("final readout must be bijective in quantum and classical wires")
            measured[wire] = classical
        else:
            started = True
            if measured or initialized != set(range(n)):
                raise ValueError("quantum work must lie after initialization and before all final measurements")
            prefix.append(item.operation, wires)
    if set(measured) != set(range(n)) or set(measured.values()) != set(range(n)):
        raise ValueError("complete final readout is required")
    rows = np.empty(2**n, dtype=int)
    for basis in range(2**n):
        classical = sum(((basis >> wire) & 1) << bit for wire, bit in measured.items())
        rows[classical] = basis
    return prefix, rows, measured


def semantic_unitary(circuit: QuantumCircuit):
    """Coherent body after initialization, not the reset channel on arbitrary inputs."""
    prefix, rows, mapping = _prefix_and_output_order(circuit)
    return Operator(prefix).data[rows], mapping


def full_measured_law(circuit: QuantumCircuit):
    unitary, mapping = semantic_unitary(circuit)
    law = np.abs(unitary[:, 0])**2
    if abs(float(law.sum())-1) > NUMERICAL_OPERATOR_GUARD:
        raise RuntimeError("complete measured law is not normalized within the numerical guard")
    return law, mapping


def _ideal_accepted_block(contract: TaylorPhysicalContract):
    coefficient = float(contract.plan.binary_terms[0][0])
    duration = float(contract.plan.binary_segment_time)
    X = np.array([[0, 1], [1, 0]], complex)
    H = coefficient*X
    term = np.eye(2, dtype=complex)
    polynomial = term.copy()
    for k in range(1, contract.plan.degree+1):
        term = (-1j*duration/k)*(H@term)
        polynomial += term
    if contract.plan.mode == "native":
        return polynomial/float(contract.plan.bounds["native_taylor_normalization"])
    return 1.5*polynomial-.5*polynomial@polynomial.conj().T@polynomial


def check_instrument_laws(contract: TaylorPhysicalContract, synthesized):
    """Complete source/routed/Clifford+T laws, including every failed outcome."""
    source, source_map = semantic_unitary(synthesized.source)
    routed, routed_map = semantic_unitary(synthesized.routed)
    clifford, clifford_map = semantic_unitary(synthesized.circuit)
    source_law, routed_law, final_law = (np.abs(matrix[:, 0])**2 for matrix in (source, routed, clifford))
    routing_operator_error = float(np.linalg.norm(source-routed, 2))
    synthesis_operator_error = float(np.linalg.norm(routed-clifford, 2))
    if routing_operator_error > NUMERICAL_OPERATOR_GUARD:
        raise RuntimeError("routed quantum law differs after restoring logical output labels")
    guarded_synthesis = float(synthesized.report["guarded_numerical_operator_error_upper_estimate"])
    if synthesis_operator_error > guarded_synthesis+NUMERICAL_OPERATOR_GUARD:
        raise RuntimeError("full coherent synthesis exceeds its accumulated numerical gate bound")
    output_bit = synthesized.source.num_clbits-1
    accepted_rows = [0, 1 << output_bit]
    accepted_columns = [0, 1]  # Logical system q0; initial layout is fixed identity.
    reference = _ideal_accepted_block(contract)
    source_block = source[np.ix_(accepted_rows, accepted_columns)]
    routed_block = routed[np.ix_(accepted_rows, accepted_columns)]
    source_error = float(np.linalg.norm(source_block-reference, 2))
    routed_error = float(np.linalg.norm(routed_block-reference, 2))
    if max(source_error, routed_error) > float(NUMERICAL_ROUTED_ACCEPTED_GUARD):
        raise RuntimeError("final source/routed accepted block fails its numerical construction guard")
    acceptance = dict.fromkeys(range(output_bit), 0)
    checks = [instruments.simulate_trial(circuit, output_bit, acceptance)
              for circuit in (synthesized.source, synthesized.routed, synthesized.circuit)]
    names = ("source", "routed", "clifford_t")
    law_records = {}
    for name, law, result in zip(names, (source_law, routed_law, final_law), checks):
        p = float(sum(law[row] for row in accepted_rows))
        mean = float((law[0]-law[1 << output_bit])/p)
        if abs(p-result["acceptance_probability"]) > NUMERICAL_OPERATOR_GUARD or abs(mean-result["mean"]) > NUMERICAL_OPERATOR_GUARD:
            raise RuntimeError("pure-state complete law and independent density/instrument checker disagree")
        law_records[name] = {"outcome_probabilities": law.tolist(),
                             "acceptance_probability": p, "conditional_Z_mean": mean,
                             "normalization_residual": abs(float(law.sum())-1),
                             "failed_attempt_probability": 1-p}
    bias = abs(checks[-1]["mean"]-checks[0]["mean"])
    if bias > float(contract.plan.total_accuracy/5)+NUMERICAL_OPERATOR_GUARD:
        raise RuntimeError("actual synthesized accepted mean exceeds the reserved synthesis bias")
    if checks[-1]["acceptance_probability"]+NUMERICAL_OPERATOR_GUARD < float(contract.post_synthesis_acceptance_lower):
        raise RuntimeError("synthesized acceptance is below the conditional guarded lower bound")
    return {"complete_measured_laws": law_records,
            "classical_readout_maps": {"source": source_map, "routed": routed_map, "clifford_t": clifford_map},
            "source_routed_semantic_operator_residual": routing_operator_error,
            "full_coherent_synthesis_operator_residual": synthesis_operator_error,
            "source_accepted_block_residual_vs_ideal": source_error,
            "routed_accepted_block_residual_vs_ideal": routed_error,
            "full_joint_law_total_variation_after_synthesis": float(np.sum(np.abs(source_law-final_law))/2),
            "actual_synthesis_conditional_Z_bias": bias,
            "independent_exact_target_Z_mean": math.cos(0.7),
            "actual_synthesized_target_Z_bias": abs(checks[-1]["mean"]-math.cos(0.7)),
            "all_failed_outcomes_included": True,
            "semantic_unitary_scope": "Coherent body after explicit initialization; the full reset channel on arbitrary inputs is not represented by a unitary",
            "guard_is_formal_interval_certificate": False,
            "law_scope": "Actual initialized one-segment source/body and complete final readout; routing output permutation restored"}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return {"real": value.real.tolist(), "imag": value.imag.tolist()} if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _synthesis_worker(source_path, prefix, allowance):
    with Path(source_path).open("rb") as handle:
        source = qpy.load(handle)[0]
    result = instruments.synthesize_instrument(source, allowance)
    with Path(str(prefix)+".qpy").open("wb") as handle:
        qpy.dump([result.source, result.routed, result.circuit], handle)
    Path(str(prefix)+".json").write_text(json.dumps(_jsonable(result.report), allow_nan=False))


def _estimate_worker(circuit_path, report_path, destination, attempts, profile_value, cache_dir):
    with Path(circuit_path).open("rb") as handle:
        source, routed, circuit = qpy.load(handle)
    report = json.loads(Path(report_path).read_text())
    synthesized = instruments.SynthesizedInstrument(source, routed, circuit, report)
    physical = instruments.estimate_instrument(
        synthesized, attempts,
        profile=instruments.InstrumentProfile(physical_error_rate=profile_value),
        physical_failure_budget=float(PHYSICAL_FAILURE_BUDGET),
        factory_availability_budget=float(FACTORY_READINESS_BUDGET),
        cache_dir=Path(cache_dir))
    Path(destination).write_text(json.dumps(_jsonable(physical), allow_nan=False))


def validate_timeout(timeout_seconds):
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, Real)
            or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
        raise ValueError("bounded worker timeout must lie in (0,60] seconds")
    return float(timeout_seconds)


def _bounded(target, arguments, timeout_seconds):
    timeout_seconds = validate_timeout(timeout_seconds)
    process = mp.get_context("spawn").Process(target=target, args=arguments)
    started = time.perf_counter()
    try:
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            raise TimeoutError("bounded native synthesis/resource worker exceeded its deadline")
        if process.exitcode != 0:
            raise RuntimeError("bounded native synthesis/resource worker failed")
        return time.perf_counter()-started
    finally:
        # Also covers KeyboardInterrupt and exceptions during join/start.
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join(1)
            process.close()


def bounded_synthesis(source, private_folder: Path, allowance: float,
                      timeout_seconds=DEFAULT_SYNTHESIS_TIMEOUT):
    private_folder.mkdir(parents=True, exist_ok=True)
    source_path = private_folder/"source.qpy"
    prefix = private_folder/"synthesized"
    with source_path.open("wb") as handle:
        qpy.dump(source, handle)
    elapsed = _bounded(_synthesis_worker, (source_path, prefix, allowance), timeout_seconds)
    with Path(str(prefix)+".qpy").open("rb") as handle:
        circuits = qpy.load(handle)
    report = json.loads(Path(str(prefix)+".json").read_text())
    return instruments.SynthesizedInstrument(*circuits, report), elapsed


def bounded_estimate(synthesized, private_folder: Path, attempts: int,
                     physical_error: float, cache_dir: Path,
                     timeout_seconds=DEFAULT_QUERY_TIMEOUT):
    if physical_error not in PHYSICAL_ERROR_PROFILES:
        raise ValueError("only the two predeclared hardware profiles are supported")
    private_folder.mkdir(parents=True, exist_ok=True)
    circuit_path, report_path, destination = (private_folder/name for name in
                                             ("estimate-input.qpy", "estimate-input.json", "physical.json"))
    with circuit_path.open("wb") as handle:
        qpy.dump([synthesized.source, synthesized.routed, synthesized.circuit], handle)
    report_path.write_text(json.dumps(_jsonable(synthesized.report), allow_nan=False))
    elapsed = _bounded(_estimate_worker, (circuit_path, report_path, destination, attempts,
                                         physical_error, cache_dir), timeout_seconds)
    return json.loads(destination.read_text()), elapsed
