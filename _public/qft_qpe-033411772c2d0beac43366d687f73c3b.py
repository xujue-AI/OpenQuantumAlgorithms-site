# SPDX-License-Identifier: MIT
"""Explicit Fourier and phase-estimation circuits with bounded reference checks.

No dense unitary is used to construct an algorithm circuit. Controlled powers
are repeated supplied circuit calls, including their coherent global phase.
The iterative circuit includes actual feedback and reset; its branch evaluator
is a bounded ideal instrument check, not a physical simulator or estimator.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]
MAX_GENERATOR_QUBITS = 64
MAX_PHASE_BITS = 16
MAX_SOURCE_OPERATIONS = 200_000
MAX_PRIMITIVE_OPERATIONS = 1_000_000


def _integer(name: str, value: int, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")


def build_qft(n_qubits: int, *, inverse: bool = False, max_distance: int | None = None) -> QuantumCircuit:
    """Positive-sign QFT including swaps; optionally omit small controlled phases.

    q[0] is the least-significant bit. max_distance=b retains CP(pi/2**d)
    for wire separation d<=b. b=0 retains H and swaps only. This is a
    circuit-approximation parameter, not a stochastic error probability.
    """
    _integer("n_qubits", n_qubits, 1, MAX_GENERATOR_QUBITS)
    if not isinstance(inverse, bool):
        raise ValueError("inverse must be boolean")
    if max_distance is not None:
        _integer("max_distance", max_distance, 0, n_qubits - 1)
    cutoff = n_qubits - 1 if max_distance is None else max_distance
    circuit = QuantumCircuit(n_qubits, name="QFT")
    for target in reversed(range(n_qubits)):
        circuit.h(target)
        for control in reversed(range(target)):
            distance = target - control
            if distance <= cutoff:
                circuit.cp(math.pi / 2**distance, control, target)
    for low in range(n_qubits // 2):
        circuit.swap(low, n_qubits - low - 1)
    return circuit.inverse() if inverse else circuit


def qft_matrix(n_qubits: int) -> np.ndarray:
    """Independent matrix definition, restricted to <=8 qubits."""
    _integer("n_qubits", n_qubits, 1, 8)
    size = 2**n_qubits
    index = np.arange(size)
    return np.exp(2j * np.pi * np.outer(index, index) / size) / math.sqrt(size)


def qft_omission_bound(n_qubits: int, max_distance: int | None) -> float:
    """Telescoping spectral-norm bound from omitted CP gates (at most 2)."""
    _integer("n_qubits", n_qubits, 1, MAX_GENERATOR_QUBITS)
    if max_distance is None:
        return 0.0
    _integer("max_distance", max_distance, 0, n_qubits - 1)
    return min(2.0, sum((n_qubits - d) * math.pi / 2**d
                        for d in range(max_distance + 1, n_qubits)))


def _expanded_count(circuit: QuantumCircuit, *, depth: int = 0) -> int:
    if depth > 24:
        raise ValueError("Nested gate expansion exceeds validation depth 24")
    if circuit.num_clbits or circuit.num_parameters:
        raise ValueError("Supply a bound, coherent circuit without classical bits")
    if not math.isfinite(float(circuit.global_phase)):
        raise ValueError("Global phase must be finite")
    count = 0
    primitive = set(ROTATION_BASIS) | {"p", "cp", "cy", "cz", "ch", "swap", "ccx", "mcx", "cs", "csdg", "crx", "cry", "crz"}
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"unitary", "initialize", "reset", "measure", "if_else", "while_loop", "for_loop", "switch_case", "store", "barrier", "delay"}:
            raise ValueError(f"Supply explicit unitary gates, found {operation.name}")
        for parameter in operation.params:
            try:
                if not math.isfinite(float(parameter)):
                    raise ValueError("Gate parameters must be finite")
            except (TypeError, ValueError) as error:
                raise ValueError("Gate parameters must be bound finite scalars") from error
        if operation.name in primitive:
            count += 1
        elif operation.definition is not None:
            count += _expanded_count(operation.definition, depth=depth + 1)
        else:
            raise ValueError(f"Opaque gate is unsupported: {operation.name}")
        if count > MAX_SOURCE_OPERATIONS:
            raise ValueError("Expanded supplied circuit exceeds source operation cap")
    return count


def _qpe_inputs(phase_bits: int, unitary: QuantumCircuit, preparation: QuantumCircuit) -> tuple[int, int, int]:
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if not isinstance(unitary, QuantumCircuit) or not isinstance(preparation, QuantumCircuit):
        raise ValueError("unitary and preparation must be QuantumCircuit objects")
    if unitary.num_qubits < 1 or unitary.num_qubits != preparation.num_qubits:
        raise ValueError("Unitary and preparation must share a positive target width")
    if unitary.num_qubits + phase_bits > MAX_GENERATOR_QUBITS:
        raise ValueError("Total circuit width exceeds generation cap")
    unitary_count = _expanded_count(unitary)
    prep_count = _expanded_count(preparation)
    source_upper = prep_count + (2**phase_bits - 1) * max(1, unitary_count) + phase_bits**2 + 3 * phase_bits
    if source_upper > MAX_SOURCE_OPERATIONS:
        raise ValueError("Repeated controlled access exceeds source operation cap")
    return unitary_count, prep_count, source_upper


def build_qpe(phase_bits: int, unitary: QuantumCircuit, preparation: QuantumCircuit, *, measure: bool = False, controlled_label: str = "U") -> QuantumCircuit:
    """Textbook QPE with repeated controlled U calls and an exact inverse QFT.

    Input preparation is applied to the target exactly once. QPE measures the
    first phase_bits qubits; y/2**phase_bits is the phase estimate. The caller
    supplies eigenstate preparation and must separately establish its overlap.
    """
    _qpe_inputs(phase_bits, unitary, preparation)
    if not isinstance(measure, bool):
        raise ValueError("measure must be boolean")
    target_width = unitary.num_qubits
    circuit = QuantumCircuit(phase_bits + target_width, phase_bits if measure else 0, name="QPE")
    targets = list(range(phase_bits, phase_bits + target_width))
    circuit.compose(preparation, targets, inplace=True)
    controlled = unitary.to_gate(label=controlled_label).control(1)
    circuit.h(range(phase_bits))
    for bit in range(phase_bits):
        for _ in range(2**bit):
            circuit.append(controlled, [bit, *targets])
    circuit.compose(build_qft(phase_bits, inverse=True), range(phase_bits), inplace=True)
    if measure:
        circuit.measure(range(phase_bits), range(phase_bits))
    return circuit


def build_iterative_qpe(phase_bits: int, unitary: QuantumCircuit, preparation: QuantumCircuit) -> QuantumCircuit:
    """One phase ancilla, decreasing powers, actual conditional phases and reset.

    Round r stores the r-th least-significant output bit in c[r]. Each
    previous c[l]=1 triggers P(-2pi/2**(r-l+1)) on the phase ancilla.
    The same target eigenstate survives each round in the ideal model.
    """
    _qpe_inputs(phase_bits, unitary, preparation)
    circuit = QuantumCircuit(1 + unitary.num_qubits, phase_bits, name="Iterative QPE")
    targets = list(range(1, 1 + unitary.num_qubits))
    circuit.compose(preparation, targets, inplace=True)
    controlled = unitary.to_gate(label="U").control(1)
    for round_index in range(phase_bits):
        circuit.reset(0)
        circuit.h(0)
        for _ in range(2**(phase_bits - round_index - 1)):
            circuit.append(controlled, [0, *targets])
        for previous in range(round_index):
            with circuit.if_test((circuit.clbits[previous], 1)):
                circuit.p(-2 * math.pi / 2**(round_index - previous + 1), 0)
        circuit.h(0)
        circuit.measure(0, round_index)
    return circuit


def phase_gate_example(phase: float) -> tuple[QuantumCircuit, QuantumCircuit]:
    """Known-phase pedagogical U=P(2pi phi), eigenstate |1> prepared by X."""
    if isinstance(phase, bool) or not isinstance(phase, (float, int)) or not math.isfinite(phase) or not 0 <= phase < 1:
        raise ValueError("phase must be a finite number in [0,1)")
    unitary = QuantumCircuit(1, name="P(2pi phi)")
    unitary.p(2 * math.pi * phase, 0)
    preparation = QuantumCircuit(1, name="Prepare |1>")
    preparation.x(0)
    return unitary, preparation


def qpe_distribution(phase: float, phase_bits: int) -> np.ndarray:
    """Independent Dirichlet-kernel distribution for a prepared eigenstate.

    sinc ratios evaluate the removable singularity at integer phases without
    cancellation. The output is normalized to remove roundoff only.
    """
    phase_gate_example(phase)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    size = 2**phase_bits
    delta = phase - np.arange(size) / size
    delta = (delta + 0.5) % 1 - 0.5
    probabilities = (np.sinc(size * delta) / np.sinc(delta))**2
    return probabilities / probabilities.sum()


def simulate_qpe(circuit: QuantumCircuit, phase_bits: int) -> np.ndarray:
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if circuit.num_qubits > 14 or circuit.num_clbits or phase_bits > circuit.num_qubits:
        raise ValueError("Static ideal simulation needs an unmeasured circuit of at most 14 qubits")
    return Statevector.from_instruction(circuit).probabilities(qargs=list(range(phase_bits)))


def lower_circuit(circuit: QuantumCircuit) -> QuantumCircuit:
    """Lower static or feedback circuits; preserve blocks and bound export size.

    Continuous rotations are unsynthesized; a lowered circuit is not a
    Clifford+T implementation. Dynamic feedback/reset needs its own estimator.
    """
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS, optimization_level=1, seed_transpiler=7)
    def count(block: QuantumCircuit) -> int:
        total = 0
        for item in block.data:
            if item.operation.name == "if_else":
                total += max(count(part) for part in item.operation.blocks if part is not None)
            elif item.operation.name in set(ROTATION_BASIS) | {"measure", "reset", "barrier"}:
                total += 1
            else:
                raise RuntimeError(f"Unsupported operation after lowering: {item.operation.name}")
        return total
    if count(compiled) > MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("Lowered primitive circuit exceeds export cap")
    return compiled


def dynamic_distribution(circuit: QuantumCircuit) -> np.ndarray:
    """Enumerate actual ideal measurement/reset/if_else branches (<=8 wires/bits).

    Numerical zero branches below 1e-15 are dropped. This evaluates the
    supplied instrument, including feedback gates; it does not infer a QPE
    distribution from the known phase formula.
    """
    if circuit.num_qubits > 8 or circuit.num_clbits > 8 or len(circuit.data) > 4096:
        raise ValueError("Dynamic instrument check is capped at 8 wires/bits and 4096 top-level operations")
    branches = [(1.0, Statevector.from_int(0, 2**circuit.num_qubits), tuple(0 for _ in circuit.clbits))]
    def project(state: Statevector, qubit: int, outcome: int) -> tuple[float, Statevector | None]:
        values = state.data.copy()
        mask = (np.arange(values.size) >> qubit) & 1
        values[mask != outcome] = 0
        probability = float(np.vdot(values, values).real)
        return probability, Statevector(values / math.sqrt(probability)) if probability > 1e-15 else None
    def run(block: QuantumCircuit, qmap: list[int], cmap: list[int], incoming):
        current = incoming
        for item in block.data:
            operation = item.operation
            qargs = [qmap[block.find_bit(q).index] for q in item.qubits]
            cargs = [cmap[block.find_bit(c).index] for c in item.clbits]
            next_branches = []
            for weight, state, classical in current:
                if operation.name == "if_else":
                    condition_bit, condition_value = operation.condition
                    if condition_bit not in block.clbits:
                        raise ValueError("Only single-bit equality feedback is supported")
                    condition_index = cmap[block.find_bit(condition_bit).index]
                    selected = 0 if classical[condition_index] == condition_value else 1
                    chosen = operation.blocks[selected] if selected < len(operation.blocks) else None
                    if chosen is None:
                        next_branches.append((weight, state, classical))
                    else:
                        next_branches.extend(run(chosen, qargs, cargs, [(weight, state, classical)]))
                elif operation.name in {"measure", "reset"}:
                    for outcome in (0, 1):
                        probability, collapsed = project(state, qargs[0], outcome)
                        if collapsed is None:
                            continue
                        new_classical = list(classical)
                        if operation.name == "measure":
                            new_classical[cargs[0]] = outcome
                        elif outcome:
                            xgate = QuantumCircuit(1)
                            xgate.x(0)
                            collapsed = collapsed.evolve(xgate, qargs)
                        next_branches.append((weight * probability, collapsed, tuple(new_classical)))
                elif operation.name == "barrier":
                    next_branches.append((weight, state, classical))
                else:
                    if item.clbits or getattr(operation, "condition", None) is not None:
                        raise ValueError("Unsupported classical operation in instrument")
                    next_branches.append((weight, state.evolve(operation, qargs), classical))
            current = next_branches
            if len(current) > 1024:
                raise ValueError("Instrument enumeration exceeds 1024 branches")
        return current
    branches = run(circuit, list(range(circuit.num_qubits)), list(range(circuit.num_clbits)), branches)
    probabilities = np.zeros(2**circuit.num_clbits)
    for weight, _, classical in branches:
        probabilities[sum(value << index for index, value in enumerate(classical))] += weight
    if not math.isclose(float(probabilities.sum()), 1, abs_tol=1e-12):
        raise AssertionError("Instrument branches lost probability mass")
    return probabilities


def median_repetitions(failure_probability: float, single_failure_bound: float = 1 - 8 / math.pi**2) -> int:
    """Odd independent repetitions whose majority-failure binomial tail meets δ.

    Scalar phase median requires phi away from the [0,1) branch cut by the
    target tolerance. A circular phase median is not implemented here.
    """
    if not 0 < failure_probability < 1 or not 0 <= single_failure_bound < 0.5:
        raise ValueError("Failure target must be in (0,1), single failure in [0,.5)")
    for repetitions in range(1, 1001, 2):
        tail = sum(math.comb(repetitions, bad) * single_failure_bound**bad * (1-single_failure_bound)**(repetitions-bad)
                   for bad in range((repetitions+1)//2, repetitions+1))
        if tail <= failure_probability:
            return repetitions
    raise ValueError("Repetition target exceeds bounded search")


def qpe_ledger(phase_bits: int, unitary: QuantumCircuit, preparation: QuantumCircuit, *, iterative: bool = False) -> dict:
    unitary_count, prep_count, upper = _qpe_inputs(phase_bits, unitary, preparation)
    circuit = build_iterative_qpe(phase_bits, unitary, preparation) if iterative else build_qpe(phase_bits, unitary, preparation, measure=True)
    compiled = lower_circuit(circuit)
    def counts(block):
        result = defaultdict(int)
        for item in block.data:
            if item.operation.name == "if_else":
                result["conditional_blocks"] += 1
                for name, value in counts(item.operation.blocks[0]).items():
                    result[f"conditional_{name}"] += value
            else:
                result[item.operation.name] += 1
        return dict(result)
    return {
        "phase_bits": phase_bits, "target_qubits": unitary.num_qubits,
        "logical_qubits": circuit.num_qubits, "classical_output_bits": phase_bits,
        "input_preparations_per_execution": 1, "controlled_unitary_calls_per_execution": 2**phase_bits-1,
        "controlled_powers": [2**j for j in range(phase_bits)],
        "supplied_unitary_source_operations": unitary_count,
        "supplied_preparation_source_operations": prep_count,
        "source_operation_upper_guard": upper,
        "source_count_semantics": "recognized CP/MCX/CCX gates count as single source operations; this guard does not bound nested transpilation work",
        "decomposed_gate_counts": counts(compiled),
        "decomposed_dependency_depth": compiled.depth(),
        "depth_semantics": "circuit dependencies; no physical feed-forward latency or scheduling model",
        "variant": "iterative feedback" if iterative else "textbook QPE",
        "rotation_synthesis_performed": False,
        "scope": "all supplied preparation, controlled access, Fourier/feedback gates, reset and final phase readout",
        "excluded": ["construction of application unitary and eigenstate beyond supplied circuits", "classical compilation/phase decoding runtime", "physical initialization faults", "physical execution unless separately projected"],
    }


def phase_product_example(target_qubits: int, phase: float = 0.3) -> tuple[QuantumCircuit, QuantumCircuit]:
    """n active P gates with eigenstate |1>^n, total eigenphase phi.

    Every target is acted on. The known equal-angle decomposition is a
    transparent workload for n-scaling, not an application speedup example.
    """
    _integer("target_qubits", target_qubits, 1, MAX_GENERATOR_QUBITS - 1)
    phase_gate_example(phase)
    unitary = QuantumCircuit(target_qubits, name="Product phase U")
    preparation = QuantumCircuit(target_qubits, name="Prepare |1>^n")
    for qubit in range(target_qubits):
        unitary.p(2 * math.pi * phase / target_qubits, qubit)
        preparation.x(qubit)
    return unitary, preparation
