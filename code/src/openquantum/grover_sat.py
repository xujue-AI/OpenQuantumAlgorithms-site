# SPDX-License-Identifier: MIT
"""Reversible CNF predicates and general amplitude-amplification circuits.

Literal 1 denotes x[0], the least significant assignment bit. The CNF phase
oracle requires its clause workspace to start in zero. It computes every
clause, phases the all-true clause register, and reverses the full computation.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Iterable

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

MAX_GENERATED_QUBITS = 32
MAX_ESTIMATED_PRIMITIVE_OPERATIONS = 200_000
MAX_PRIMITIVE_OPERATIONS = 250_000


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("iterations must be a nonnegative integer")


@dataclass(frozen=True)
class CNF:
    n_variables: int
    clauses: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        _positive_integer(self.n_variables, "n_variables")
        if isinstance(self.clauses, (str, bytes)):
            raise ValueError("clauses must be sequences of literal sequences")
        original = []
        for clause in self.clauses:
            if isinstance(clause, (str, bytes)):
                raise ValueError("each clause must be a sequence of integers")
            literals = tuple(clause)
            for literal in literals:
                if isinstance(literal, bool) or not isinstance(literal, int) or literal == 0:
                    raise ValueError("literals must be nonzero signed integers")
                if abs(literal) > self.n_variables:
                    raise ValueError("literal variable exceeds n_variables")
            original.append(literals)
        object.__setattr__(self, "clauses", tuple(original))

    @property
    def canonical_clauses(self) -> tuple[tuple[int, ...], ...]:
        """Remove duplicate literals without changing the predicate."""
        return tuple(tuple(dict.fromkeys(clause)) for clause in self.clauses)

    @property
    def num_clause_qubits(self) -> int:
        return len(self.clauses)

    @property
    def num_clean_work_qubits(self) -> int:
        """Reusable conjunction work for clauses, marking, and diffusion."""
        clause_work = max((max(len(clause)-2, 0)
            for clause in self.canonical_clauses
            if not any(-literal in clause for literal in clause)), default=0)
        return max(clause_work, self.num_clause_qubits-3, self.n_variables-3, 0)

    @property
    def num_qubits(self) -> int:
        return self.n_variables + self.num_clause_qubits + self.num_clean_work_qubits

    def satisfies(self, assignment: int) -> bool:
        if isinstance(assignment, bool) or not isinstance(assignment, int) or not 0 <= assignment < 2**self.n_variables:
            raise ValueError("assignment must be an integer in the declared search space")
        return all(any(
            bool((assignment >> (abs(literal)-1)) & 1) == (literal > 0)
            for literal in clause
        ) for clause in self.clauses)

    def as_dict(self) -> dict:
        return {
            "n_variables": self.n_variables,
            "clauses": [list(clause) for clause in self.clauses],
            "canonical_clauses": [list(clause) for clause in self.canonical_clauses],
            "literal_convention": "Signed one-based variable identifiers; variable 1 is x[0]/least-significant bit",
            "oracle_normalization": "Duplicate literals removed; tautological and empty clauses handled explicitly",
            "clause_qubits": self.num_clause_qubits,
            "clean_work_qubits": self.num_clean_work_qubits,
            "total_circuit_qubits": self.num_qubits,
            "multi_control_construction": "Exact CCX conjunction chain with clean reusable work; full uncompute",
        }

    @classmethod
    def from_dict(cls, value: dict) -> "CNF":
        return cls(value["n_variables"], value["clauses"])


def verify_candidate(cnf: CNF, assignment: int) -> dict:
    """Classical verification, with actual short-circuit work and CPU timing."""
    if isinstance(assignment, bool) or not isinstance(assignment, int) or not 0 <= assignment < 2**cnf.n_variables:
        raise ValueError("Candidate lies outside the declared search space")
    started = time.perf_counter_ns()
    literal_checks = 0
    clause_checks = 0
    valid = True
    for clause in cnf.clauses:
        clause_checks += 1
        clause_true = False
        for literal in clause:
            literal_checks += 1
            if bool((assignment >> (abs(literal)-1)) & 1) == (literal > 0):
                clause_true = True
                break
        if not clause_true:
            valid = False
            break
    return {
        "verified": valid, "clauses_checked": clause_checks,
        "literals_checked": literal_checks,
        "classical_verification_cpu_ns": time.perf_counter_ns() - started,
    }


def exhaustive_solutions(cnf: CNF, *, max_variables: int = 8) -> list[int]:
    """A separately charged, bounded reference diagnostic; search never calls it."""
    if cnf.n_variables > max_variables:
        raise ValueError("Exhaustive diagnostic exceeds its explicit variable bound")
    return [value for value in range(2**cnf.n_variables) if cnf.satisfies(value)]


def _clean_controlled_x(circuit: QuantumCircuit, controls: list[int], target: int, work: list[int]) -> None:
    """Exact k-controlled X using max(k-2,0) clean qubits, restored to zero."""
    count = len(controls)
    if count == 0:
        circuit.x(target)
    elif count == 1:
        circuit.cx(controls[0], target)
    elif count == 2:
        circuit.ccx(*controls, target)
    else:
        if len(work) < count-2 or set(work[:count-2]) & {*controls, target}:
            raise ValueError("Controlled X needs distinct clean conjunction work")
        circuit.ccx(controls[0], controls[1], work[0])
        for index in range(2, count-1):
            circuit.ccx(work[index-2], controls[index], work[index-1])
        circuit.ccx(work[count-3], controls[-1], target)
        for index in reversed(range(2, count-1)):
            circuit.ccx(work[index-2], controls[index], work[index-1])
        circuit.ccx(controls[0], controls[1], work[0])


def _phase_all_ones(circuit: QuantumCircuit, qubits: list[int], work: list[int]) -> None:
    if not qubits:
        circuit.global_phase += math.pi
    elif len(qubits) == 1:
        circuit.z(qubits[0])
    else:
        target = qubits[-1]
        circuit.h(target)
        _clean_controlled_x(circuit, qubits[:-1], target, work)
        circuit.h(target)


def estimate_sat_primitive_operations(cnf: CNF, iterations: int, *, measure: bool = True) -> int:
    """Conservative expansion preflight, not a timing or synthesis certificate.

    A CCX lowers to at most 15 Clifford+T primitives in the pinned basis.
    Source circuits use linear conjunction chains rather than opaque MCX.
    A post-transpilation cap separately checks the actual exported operations.
    """
    _iterations(iterations)
    if cnf.num_qubits > MAX_GENERATED_QUBITS:
        raise ValueError("SAT circuit exceeds the 32-qubit source-generation limit")

    def controlled_x(controls: int) -> int:
        return 1 if controls < 2 else 15 * (2*controls-3)

    def phase(bits: int) -> int:
        return 0 if bits == 0 else 1 if bits == 1 else 2 + controlled_x(bits - 1)

    compute = 0
    for clause in cnf.canonical_clauses:
        if not clause:
            continue
        if any(-literal in clause for literal in clause):
            compute += 1
        else:
            compute += 1 + 2 * sum(literal > 0 for literal in clause) + controlled_x(len(clause))
    oracle = 2 * compute + phase(cnf.num_clause_qubits)
    diffuser = 4 * cnf.n_variables + phase(cnf.n_variables)
    return cnf.n_variables + iterations * (oracle + diffuser) + (cnf.n_variables if measure else 0)


def preflight_sat_circuit(cnf: CNF, iterations: int, *, measure: bool = True) -> dict:
    estimate = estimate_sat_primitive_operations(cnf, iterations, measure=measure)
    if estimate > MAX_ESTIMATED_PRIMITIVE_OPERATIONS:
        raise ValueError("SAT circuit exceeds the conservative primitive-expansion preflight")
    return {
        "estimated_primitive_operations": estimate,
        "maximum_estimated_primitive_operations": MAX_ESTIMATED_PRIMITIVE_OPERATIONS,
        "maximum_actual_primitive_operations": MAX_PRIMITIVE_OPERATIONS,
        "estimate_scope": "Linear clean-work CCX-chain expansion preflight; actual lowering checked separately, no compiler memory/time certificate",
    }


def compute_clause_register(cnf: CNF) -> QuantumCircuit:
    """Toggle clause[i] by the OR of that clause's literals, for every clause."""
    data = QuantumRegister(cnf.n_variables, "x")
    registers = [data]
    if cnf.num_clause_qubits:
        registers.append(QuantumRegister(cnf.num_clause_qubits, "clause"))
    if cnf.num_clean_work_qubits:
        registers.append(QuantumRegister(cnf.num_clean_work_qubits, "work"))
    circuit = QuantumCircuit(*registers, name="compute clauses")
    work = list(range(cnf.n_variables+cnf.num_clause_qubits, cnf.num_qubits))
    for index, clause in enumerate(cnf.canonical_clauses):
        target = cnf.n_variables + index
        if not clause:
            continue
        if any(-literal in clause for literal in clause):
            circuit.x(target)
            continue
        # OR(literals) = 1 XOR AND(all literals false).
        circuit.x(target)
        masked = [literal-1 for literal in clause if literal > 0]
        for qubit in masked:
            circuit.x(qubit)
        _clean_controlled_x(circuit, [abs(literal)-1 for literal in clause], target, work)
        for qubit in masked:
            circuit.x(qubit)
    return circuit


def build_cnf_phase_oracle(cnf: CNF) -> QuantumCircuit:
    """Implement (-1)^CNF(x) on the clean-clause-workspace subspace.

    With the conjunction pool clean but dirty clause register s, the phase is
    AND_i(s_i XOR clause_i(x)). Dirty conjunction work violates both contracts.
    Scratch restoration alone is therefore not the intended oracle contract.
    """
    compute = compute_clause_register(cnf)
    oracle = QuantumCircuit(*compute.qregs, name="CNF phase oracle")
    oracle.compose(compute, inplace=True)
    _phase_all_ones(oracle, list(range(cnf.n_variables, cnf.n_variables+cnf.num_clause_qubits)),
                    list(range(cnf.n_variables+cnf.num_clause_qubits, cnf.num_qubits)))
    oracle.compose(compute.inverse(), inplace=True)
    oracle.metadata = {
        "cnf": cnf.as_dict(), "search_qubits": cnf.n_variables,
        "clause_qubits": cnf.num_clause_qubits,
        "clean_work_qubits": cnf.num_clean_work_qubits,
        "workspace_promise": "All clause and conjunction work qubits start in zero and return to zero",
        "construction": "Compute every clause with exact CCX chains; phase all true; full inverse/uncompute",
    }
    return oracle


def build_amplitude_amplification_circuit(
    oracle: QuantumCircuit, n_search_qubits: int, iterations: int, *, measure: bool = False,
) -> QuantumCircuit:
    """Use any valid clean-workspace phase oracle, without knowing marked count."""
    _positive_integer(n_search_qubits, "n_search_qubits")
    _iterations(iterations)
    if n_search_qubits > oracle.num_qubits or oracle.num_clbits or oracle.num_parameters:
        raise ValueError("Oracle must be bound, quantum-only, and contain the search register")
    if any(item.operation.name in {"measure", "reset", "initialize", "if_else", "while_loop", "for_loop"} for item in oracle.data):
        raise ValueError("Oracle must be coherent and contain no measurements/reset/dynamic control")
    circuit = QuantumCircuit(*oracle.qregs)
    required_work = max(n_search_qubits-3, 0)
    deficit = max(required_work-(oracle.num_qubits-n_search_qubits), 0)
    if deficit:
        circuit.add_register(QuantumRegister(deficit, "diffuser_work"))
    data = list(range(n_search_qubits))
    work = list(range(n_search_qubits, circuit.num_qubits))
    circuit.h(data)
    for _ in range(iterations):
        circuit.compose(oracle, qubits=list(range(oracle.num_qubits)), inplace=True)
        circuit.h(data)
        circuit.x(data)
        _phase_all_ones(circuit, data, work)
        circuit.x(data)
        circuit.h(data)
    if measure:
        bits = ClassicalRegister(n_search_qubits, "assignment")
        circuit.add_register(bits)
        circuit.measure(data, bits)
    circuit.metadata = {
        "iterations": iterations, "search_qubits": n_search_qubits,
        "workspace_qubits": circuit.num_qubits-n_search_qubits,
        "added_diffuser_work_qubits": deficit,
        "diffuser_convention": "I-2|uniform><uniform|; minus the usual reflection, a per-iteration global sign",
        "known_marked_count_used": False,
        "workspace_promise": "Oracle workspace starts clean and is fully restored each iteration",
        "zero_initialization": "Initial logical zero is an explicit circuit/model assumption",
    }
    return circuit


def build_sat_grover_circuit(cnf: CNF, iterations: int, *, measure: bool = False) -> QuantumCircuit:
    return build_amplitude_amplification_circuit(
        build_cnf_phase_oracle(cnf), cnf.n_variables, iterations, measure=measure
    )


def amplification_probability(n_variables: int, marked_count: int, iterations: int) -> float:
    """Ideal reference formula when the marked count is supplied independently."""
    _positive_integer(n_variables, "n_variables")
    _iterations(iterations)
    if isinstance(marked_count, bool) or not isinstance(marked_count, int) or not 0 <= marked_count <= 2**n_variables:
        raise ValueError("marked_count must be an integer in the search-space range")
    if marked_count == 0:
        return 0.0
    if marked_count == 2**n_variables:
        return 1.0
    theta = math.asin(math.sqrt(marked_count / (2**n_variables)))
    return math.sin((2*iterations+1)*theta)**2


def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    """Stable structural identity; exact SDK version is recorded separately."""
    operations = []
    for item in circuit.data:
        operations.append({
            "name": item.operation.name,
            "parameters": [float(p) for p in item.operation.params],
            "qubits": [circuit.find_bit(q).index for q in item.qubits],
            "clbits": [circuit.find_bit(c).index for c in item.clbits],
            "control_state": getattr(item.operation, "ctrl_state", None),
        })
    value = {
        "qubits": circuit.num_qubits, "clbits": circuit.num_clbits,
        "global_phase": float(circuit.global_phase), "operations": operations,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replay_recorded_attempts(cnf: CNF, iterations: Iterable[int], *, measure: bool = True) -> QuantumCircuit:
    """Fixed chosen-iteration replay with fresh disjoint banks, not adaptive BBHT.

    Independent-bank measurements can be deferred to the end. This trades memory
    for avoiding unsupported reset/feedback. Replay outcomes are fresh, not forced
    to match a previously recorded trace.
    """
    values = tuple(iterations)
    if not values:
        raise ValueError("Replay requires at least one recorded executed attempt")
    for value in values:
        _iterations(value)
    bank_width = cnf.num_qubits
    circuit = QuantumCircuit(len(values)*bank_width)
    for bank, count in enumerate(values):
        trial = build_sat_grover_circuit(cnf, count)
        qubits = list(range(bank*bank_width, (bank+1)*bank_width))
        circuit.compose(trial, qubits=qubits, inplace=True)
        circuit.barrier(*qubits)
    if measure:
        outputs = ClassicalRegister(len(values)*cnf.n_variables, "assignments")
        circuit.add_register(outputs)
        for bank in range(len(values)):
            for variable in range(cnf.n_variables):
                circuit.measure(bank*bank_width+variable, outputs[bank*cnf.n_variables+variable])
    circuit.metadata = {
        "scope": "Fixed recorded chosen-iteration prefix replay, not the adaptive search",
        "banks": len(values), "bank_width": bank_width, "attempt_iterations": list(values),
        "search_qubits_per_bank": cnf.n_variables,
        "scratch_qubits_per_bank": cnf.num_clause_qubits+cnf.num_clean_work_qubits,
        "clause_qubits_per_bank": cnf.num_clause_qubits,
        "clean_work_qubits_per_bank": cnf.num_clean_work_qubits,
        "initialization_assumption": "Every fresh bank starts in logical zero; no separate physical initialization protocol is calibrated",
        "measurement_policy": "All independent data-bank measurements deferred to the end",
        "runtime_limit": "QDK canonical counts/PSSPC abstract the source gate DAG; replay model time is not a validated gate schedule or serial adaptive search wall time",
    }
    return circuit
