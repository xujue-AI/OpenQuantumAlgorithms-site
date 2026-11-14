# SPDX-License-Identifier: MIT
"""Grover search with an explicit, parameterized single-target phase oracle.

The known target is a pedagogical oracle. It is not a meaningful classical
search benchmark: a production problem must supply its own reversible predicate.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit, transpile


ROTATION_BASIS = ["h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx"]


def _validate_problem(n_qubits: int, marked: int) -> None:
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    if isinstance(marked, bool) or not isinstance(marked, int) or not 0 <= marked < 2**n_qubits:
        raise ValueError("marked must be an integer in [0, 2**n_qubits)")


def optimal_iterations(n_qubits: int) -> int:
    """Choose the first integer nearest the initial success-probability peak.

    Assumes exactly one marked state. The two-qubit case has one iteration;
    the three-qubit case has two. Qubit count is not a simulation limit.
    """
    _validate_problem(n_qubits, 0)
    theta = math.asin(2.0 ** (-n_qubits / 2))
    return math.floor(math.pi / (4 * theta))


def expected_success_probability(n_qubits: int, iterations: int) -> float:
    """Ideal single-target probability sin²((2r + 1) asin(1/sqrt(N)))."""
    _validate_problem(n_qubits, 0)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a non-negative integer")
    theta = math.asin(2.0 ** (-n_qubits / 2))
    return math.sin((2 * iterations + 1) * theta) ** 2


def _phase_flip_all_ones(circuit: QuantumCircuit, n_qubits: int) -> None:
    if n_qubits == 1:
        circuit.z(0)
    else:
        target = n_qubits - 1
        circuit.h(target)
        circuit.mcx(list(range(target)), target)
        circuit.h(target)


def phase_oracle(n_qubits: int, marked: int) -> QuantumCircuit:
    """Flip only |marked⟩, exposing X and MCX gates rather than a dense matrix.

    Qiskit numbers q[0] as the least significant bit. Displayed bit strings
    therefore read q[n-1] ... q[0], matching Python's binary integer notation.
    """
    _validate_problem(n_qubits, marked)
    circuit = QuantumCircuit(n_qubits, name=f"Oracle {marked:0{n_qubits}b}")
    zero_bits = [i for i in range(n_qubits) if not ((marked >> i) & 1)]
    for i in zero_bits:
        circuit.x(i)
    _phase_flip_all_ones(circuit, n_qubits)
    for i in zero_bits:
        circuit.x(i)
    return circuit


def build_grover_circuit(
    n_qubits: int,
    marked: int,
    iterations: int | None = None,
    *,
    measure: bool = False,
) -> QuantumCircuit:
    """Generate a Grover circuit for any register size and single marked value.

    The diffuser differs from 2|s><s|-I by an irrelevant global phase.
    MCX is an actual Qiskit operation; decompose it before gate-level costing.
    Large circuit generation can still be expensive even without simulation.
    """
    _validate_problem(n_qubits, marked)
    if iterations is None:
        iterations = optimal_iterations(n_qubits)
    expected_success_probability(n_qubits, iterations)  # validates iterations
    circuit = QuantumCircuit(n_qubits)
    circuit.h(range(n_qubits))
    oracle = phase_oracle(n_qubits, marked)
    for _ in range(iterations):
        circuit.compose(oracle, inplace=True)
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        _phase_flip_all_ones(circuit, n_qubits)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))
    if measure:
        circuit.measure_all()
    return circuit


def decompose_to_rotation_basis(circuit: QuantumCircuit) -> QuantumCircuit:
    """Produce an explicit supported gate circuit for export and logical counts.

    Arbitrary rotations remain unsynthesized. These counts are not T counts
    and are not a physical or fault-tolerant resource estimate.
    """
    compiled = transpile(
        circuit,
        basis_gates=ROTATION_BASIS,
        optimization_level=1,
        seed_transpiler=7,
    )
    unsupported = set(compiled.count_ops()) - set(ROTATION_BASIS) - {"measure", "barrier"}
    if unsupported:
        raise RuntimeError(f"Circuit contains unsupported operations: {unsupported}")
    return compiled
