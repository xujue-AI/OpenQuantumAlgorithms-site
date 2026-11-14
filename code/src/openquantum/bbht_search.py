# SPDX-License-Identifier: MIT
"""Count-agnostic bounded BBHT-style search with explicit ideal trial traces.

An arbitrary finite attempt/query cap gives an inconclusive result, not UNSAT.
The simulator is bounded and used only to sample outcomes. It does not count
satisfying assignments or feed a marked count into the iteration schedule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import qasm3

from .grover_sat import (
    CNF, MAX_PRIMITIVE_OPERATIONS, build_sat_grover_circuit,
    circuit_fingerprint, preflight_sat_circuit, verify_candidate,
)
from .resource_estimation import prepare_measured_circuit


@dataclass(frozen=True)
class BBHTConfig:
    seed: int = 7
    max_attempts: int = 16
    max_oracle_calls: int = 32
    growth: float = 1.2
    max_simulation_qubits: int = 14
    max_wall_seconds: float | None = 30.0

    def __post_init__(self) -> None:
        for name in ("seed", "max_attempts", "max_oracle_calls", "max_simulation_qubits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0 or not 1 <= self.max_attempts <= 128 or self.max_oracle_calls < 0:
            raise ValueError("Require nonnegative seed/query cap and 1..128 attempts")
        if isinstance(self.growth, bool) or not isinstance(self.growth, (int, float)) or not math.isfinite(self.growth) or not 1 < self.growth < 4/3:
            raise ValueError("BBHT growth must lie strictly between one and four thirds")
        if not 1 <= self.max_simulation_qubits <= 16:
            raise ValueError("CPU simulation bound must lie in 1..16 total qubits")
        if self.max_wall_seconds is not None and (
            isinstance(self.max_wall_seconds, bool) or not isinstance(self.max_wall_seconds, (int, float))
            or not math.isfinite(self.max_wall_seconds) or self.max_wall_seconds <= 0
        ):
            raise ValueError("Soft wall budget must be finite and positive, or None")


def run_bbht(cnf: CNF, config: BBHTConfig = BBHTConfig()) -> dict:
    """Run the bounded unknown-count schedule and verify every sampled candidate."""
    if cnf.num_qubits > config.max_simulation_qubits:
        raise ValueError("Data plus clause/conjunction workspace exceeds the explicit simulation bound")
    wall_started = time.perf_counter()
    rng = np.random.default_rng(config.seed)
    m = 1.0
    maximum_window = math.sqrt(2**cnf.n_variables)
    used_queries = 0
    attempts = []
    result = {
        "algorithm": "Bounded BBHT-style unknown-solution-count search",
        "cnf": cnf.as_dict(), "config": asdict(config),
        "marked_count_used_by_search": False,
        "sampling_method": "NumPy Generator choice from exact ideal Statevector; CPU simulation, not hardware",
        "status": "inconclusive", "termination_reason": "attempt_cap",
        "verified_assignment": None, "attempts": attempts,
        "search_failure_confidence": None,
        "wall_budget_policy": "Soft budget checked between attempts; an in-progress trial may overrun",
        "unsat_certificate": None,
        "limitations": [
            "An arbitrary attempt/query cap does not certify absence of a satisfying assignment",
            "Every trial starts from fresh clean logical zero; adaptive physical reset/feedback timing is not simulated",
            "Ideal statevector simulation is exponential in data plus scratch width and is explicitly bounded",
            "No free exact marked count or large-instance classical exhaustive solve is used by this schedule",
            "Recorded quantum-prefix replay costs are not a global adaptive runtime/success bound",
        ],
    }
    for number in range(1, config.max_attempts+1):
        if config.max_wall_seconds is not None and time.perf_counter()-wall_started >= config.max_wall_seconds:
            result["termination_reason"] = "soft_wall_cap_between_attempts"
            break
        window = max(1, math.ceil(m))
        count = int(rng.integers(0, window))
        if used_queries+count > config.max_oracle_calls:
            result["termination_reason"] = "oracle_call_cap_before_proposed_attempt"
            result["unexecuted_budget_proposal"] = {
                "attempt": number, "window_real": m, "window_integer_cardinality": window,
                "proposed_iterations": count, "remaining_oracle_calls": config.max_oracle_calls-used_queries,
                "policy": "Stop without executing; sampled iterations were not clipped",
            }
            break
        try:
            preflight = preflight_sat_circuit(cnf, count)
        except ValueError as exc:
            result["termination_reason"] = "materialization_cap_before_proposed_attempt"
            result["unexecuted_materialization_proposal"] = {
                "attempt": number, "proposed_iterations": count, "reason": str(exc),
                "policy": "Stop without constructing/simulating or clipping the proposed trial",
            }
            break
        started = time.perf_counter_ns()
        circuit = build_sat_grover_circuit(cnf, count, measure=True)
        generation_ns = time.perf_counter_ns()-started
        started = time.perf_counter_ns()
        compiled = prepare_measured_circuit(circuit, optimization_level=0)
        lowering_ns = time.perf_counter_ns()-started
        if compiled.num_qubits > config.max_simulation_qubits or len(compiled.data) > MAX_PRIMITIVE_OPERATIONS:
            result["termination_reason"] = "lowered_circuit_cap_before_simulation"
            result["unexecuted_materialization_proposal"] = {
                "attempt": number, "proposed_iterations": count,
                "lowered_qubits": compiled.num_qubits, "lowered_operations": len(compiled.data),
                "policy": "Circuit compiled but quantum trial not simulated/executed; not charged as an executed oracle query",
            }
            break
        # Primitive one/two-qubit gates avoid a dense high-control MCX matrix.
        started = time.perf_counter_ns()
        state = Statevector.from_instruction(compiled.remove_final_measurements(inplace=False))
        full_probabilities = state.probabilities()
        clean_probability = float(np.sum(full_probabilities[:2**cnf.n_variables]))
        if abs(clean_probability-1) > 1e-10:
            raise RuntimeError("Ideal clause/conjunction scratch did not uncompute to zero")
        probabilities = state.probabilities(qargs=list(range(cnf.n_variables)))
        candidate = int(rng.choice(2**cnf.n_variables, p=probabilities/np.sum(probabilities)))
        simulation_ns = time.perf_counter_ns()-started
        verification = verify_candidate(cnf, candidate)
        used_queries += count
        attempts.append({
            "attempt": number, "window_real": m, "window_integer_cardinality": window,
            "iterations": count, "oracle_calls": count, "cumulative_oracle_calls": used_queries,
            "candidate": candidate, "candidate_bitstring": f"{candidate:0{cnf.n_variables}b}",
            **verification,
            "source_circuit_sha256": circuit_fingerprint(circuit),
            "source_gate_counts": dict(circuit.count_ops()),
            "simulated_primitive_gate_counts": dict(compiled.count_ops()),
            "simulation_input_qasm_sha256": hashlib.sha256(qasm3.dumps(compiled).encode()).hexdigest(),
            "simulation_input_kind": "Actual source trial lowered to primitive gates before bounded Statevector simulation",
            "materialization_preflight": preflight,
            "declared_qubits": circuit.num_qubits,
            "data_qubits": cnf.n_variables, "clause_scratch_qubits": cnf.num_clause_qubits,
            "clean_conjunction_work_qubits": cnf.num_clean_work_qubits,
            "data_measurements": cnf.n_variables,
            "clean_scratch_probability": clean_probability,
            "circuit_generation_cpu_ns": generation_ns,
            "circuit_lowering_cpu_ns": lowering_ns,
            "ideal_simulation_cpu_ns": simulation_ns,
        })
        if verification["verified"]:
            result["status"] = "found"
            result["termination_reason"] = "classically_verified_candidate"
            result["verified_assignment"] = candidate
            break
        m = min(config.growth*m, maximum_window)
    result["elapsed_wall_s"] = time.perf_counter()-wall_started
    result["executed_attempts"] = len(attempts)
    result["actual_oracle_calls"] = used_queries
    result["classical_verification_calls"] = len(attempts)
    result["classical_literal_checks"] = sum(x["literals_checked"] for x in attempts)
    result["classical_clause_checks"] = sum(x["clauses_checked"] for x in attempts)
    result["classical_verification_cpu_ns"] = sum(x["classical_verification_cpu_ns"] for x in attempts)
    return result
