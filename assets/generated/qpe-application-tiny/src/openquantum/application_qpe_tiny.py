# SPDX-License-Identifier: MIT
"""Private, unexecuted source for a frozen tiny application-QPE tranche.

H = -Z0 - Z1 + X0 X1/4; input is actually reset to |00>.  The base
controlled U = -exp(-i H pi/3) is written as 39 primitive instructions.
No dense gate, eigenstate loader, analytical-power algorithm shortcut,
transpiler, inference, synthesis or resource estimator is used here.

Authoring this module supplies no numerical evidence.  Only a separately
approved bounded driver may import it.  Full m6/R41/K2 and physical work
remain outside this fixed m2/R2, powers 1/2/4 source contract.
"""
from __future__ import annotations

from collections import Counter
import math

from qiskit import QuantumCircuit

CONTRACT_SHA256 = "2d5d9a7f60503c65ab09a433679da9449fb805d59110808e778be7cba59164b3"
MAX_QUBITS = 6
MAX_CLASSICAL_BITS = 4
MAX_RETAINED_RECORDS = 16
MAX_EXPANDED_INSTRUCTIONS = 10_000
ADMITTED_POWERS = (1, 2, 4)
PRIMITIVES = frozenset({"h", "x", "z", "sdg", "p", "ry", "rz", "cx", "reset", "measure"})


def _parameters() -> tuple[float, float, float]:
    """Future binary64 realization; no interval/synthesis certificate implied."""
    return math.sqrt(65) / 4, math.atan(1 / 8), math.pi / 3


def _cc_rotation(circuit: QuantumCircuit, axis: str, angle: float,
                 parity: int, clock: int, target: int) -> None:
    """Chronological Gray-code quarter rotations, selecting parity=clock=1.

    Net angle is angle*(1-(-1)^parity)*(1-(-1)^clock)/4.  Target X
    flips cancel.  Only Y/Z rotation axes are admitted.
    """
    if axis not in {"y", "z"} or not math.isfinite(angle):
        raise ValueError("Internal CC rotation must have a finite Y/Z angle")
    rotate = circuit.ry if axis == "y" else circuit.rz
    rotate(angle / 4, target)
    circuit.cx(parity, target)
    rotate(-angle / 4, target)
    circuit.cx(clock, target)
    rotate(angle / 4, target)
    circuit.cx(parity, target)
    rotate(-angle / 4, target)
    circuit.cx(clock, target)


def _append_base(circuit: QuantumCircuit, clock: int, data0: int, data1: int,
                 *, reference_z: bool = True, odd_evolution: bool = True) -> None:
    """Append actual access; local map is clock/data0/data1, in that order."""
    if len({clock, data0, data1}) != 3 or any(
        isinstance(wire, bool) or not isinstance(wire, int)
        or not 0 <= wire < circuit.num_qubits for wire in (clock, data0, data1)
    ):
        raise ValueError("Distinct in-range internal wires required")
    radius, theta, tau = _parameters()
    circuit.cx(data0, data1)
    # Open the even-parity control once around all three even rotations.
    circuit.x(data1)
    _cc_rotation(circuit, "y", theta, data1, clock, data0)
    _cc_rotation(circuit, "z", -2 * radius * tau, data1, clock, data0)
    _cc_rotation(circuit, "y", -theta, data1, clock, data0)
    circuit.x(data1)
    # H changes are unconditioned and cancel when clock=0 or parity=0.
    circuit.h(data0)
    if odd_evolution:
        _cc_rotation(circuit, "z", tau / 2, data1, clock, data0)
    circuit.h(data0)
    circuit.cx(data0, data1)
    if reference_z:
        circuit.z(clock)


def build_controlled_kernel() -> QuantumCircuit:
    """Three-wire, 39-instruction full controlled propagator, including odd sector."""
    circuit = QuantumCircuit(3, name="application-qpe-controlled-U")
    _append_base(circuit, 0, 1, 2)
    ledger = source_ledger(circuit)
    expected = {"ry": 8, "rz": 8, "cx": 18, "h": 2, "x": 2, "z": 1}
    if ledger["unitary_by_name"] != expected or ledger["expanded_instructions"] != 39:
        raise AssertionError("Frozen base kernel must contain exactly 39 primitive instructions")
    return circuit


def build_kernel_without_reference_z() -> QuantumCircuit:
    """Deliberately wrong 38-instruction negative control; never an access option."""
    circuit = QuantumCircuit(3, name="negative-missing-reference-Z")
    _append_base(circuit, 0, 1, 2, reference_z=False)
    source_ledger(circuit)
    return circuit


def build_kernel_without_odd_evolution() -> QuantumCircuit:
    """Deliberately wrong full-operator negative control, invisible to |00> input."""
    circuit = QuantumCircuit(3, name="negative-missing-odd-sector")
    _append_base(circuit, 0, 1, 2, odd_evolution=False)
    source_ledger(circuit)
    return circuit


def _repeat_base(circuit: QuantumCircuit, clock: int, data0: int,
                 data1: int, power: int) -> None:
    if isinstance(power, bool) or not isinstance(power, int) or power not in ADMITTED_POWERS:
        raise ValueError("Only literal powers 1, 2 and 4 belong to this tranche")
    for _ in range(power):
        _append_base(circuit, clock, data0, data1)


def _iqft2(circuit: QuantumCircuit, low: int, high: int) -> None:
    """Exact mathematical inverse F4, primitive CP and SWAP decompositions.

    Wire low is the least-significant output bit.  Each controlled P(lambda)
    uses P(lambda/2) on control, CX, P(-lambda/2) on target, CX, P(lambda/2)
    on target, preserving the controlled scalar phase.
    """
    circuit.cx(low, high)
    circuit.cx(high, low)
    circuit.cx(low, high)
    circuit.h(low)
    angle = -math.pi / 2
    circuit.p(angle / 2, low)
    circuit.cx(low, high)
    circuit.p(-angle / 2, high)
    circuit.cx(low, high)
    circuit.p(angle / 2, high)
    circuit.h(high)


def _round2(circuit: QuantumCircuit, low: int, high: int,
            data0: int, data1: int) -> None:
    circuit.h(low)
    circuit.h(high)
    _repeat_base(circuit, low, data0, data1, 1)
    _repeat_base(circuit, high, data0, data1, 2)
    _iqft2(circuit, low, high)


def build_reduced_deferred() -> QuantumCircuit:
    """Six actual reset wires, two retained-target rounds, final four-bit readout.

    q0/q1 and q2/q3 are round clocks; q4/q5 are data.  All clock readout is
    deferred.  Record integer is y0 + 4*y1 with y0=c0+2c1, y1=c2+2c3.
    """
    circuit = QuantumCircuit(6, 4, name="m2-r2-deferred-retained")
    circuit.reset(range(6))
    _round2(circuit, 0, 1, 4, 5)
    _round2(circuit, 2, 3, 4, 5)
    circuit.measure(range(4), range(4))
    source_ledger(circuit)
    return circuit


def _reused_textbook(*, reprepare: bool) -> QuantumCircuit:
    circuit = QuantumCircuit(4, 4, name="m2-r2-reprepared-negative" if reprepare
                             else "m2-r2-reused-retained")
    circuit.reset([2, 3])
    for round_index in range(2):
        if reprepare and round_index == 1:
            circuit.reset([2, 3])
        circuit.reset([0, 1])
        _round2(circuit, 0, 1, 2, 3)
        circuit.measure(0, 2 * round_index)
        circuit.measure(1, 2 * round_index + 1)
    source_ledger(circuit)
    return circuit


def build_reduced_textbook() -> QuantumCircuit:
    """Four-wire clock reset/reuse; target retained through both rounds."""
    return _reused_textbook(reprepare=False)


def build_reprepared_textbook() -> QuantumCircuit:
    """Actual target-reset negative control: product of mixed marginals."""
    return _reused_textbook(reprepare=True)


def build_reduced_feedback() -> QuantumCircuit:
    """Three wires, four recorded bits, exact prior-bit feedback and actual resets."""
    circuit = QuantumCircuit(3, 4, name="m2-r2-feedback-retained")
    circuit.reset([1, 2])
    for round_index in range(2):
        for bit_index, power in enumerate((2, 1)):
            circuit.reset(0)
            circuit.h(0)
            _repeat_base(circuit, 0, 1, 2, power)
            if bit_index == 1:
                with circuit.if_test((circuit.clbits[2 * round_index], 1)):
                    circuit.sdg(0)  # P(-pi/2), exact ideal Clifford correction.
            circuit.h(0)
            circuit.measure(0, 2 * round_index + bit_index)
    source_ledger(circuit)
    return circuit


def build_fresh_experiment(power: int, basis: str) -> QuantumCircuit:
    """One actual fresh |00> preparation and one Hadamard-fringe readout."""
    if isinstance(power, bool) or not isinstance(power, int) or power not in ADMITTED_POWERS:
        raise ValueError("Fixed tranche admits powers 1, 2 and 4 only")
    if basis not in {"cos", "sin"}:
        raise ValueError("Quadrature must be cos or sin")
    circuit = QuantumCircuit(3, 1, name=f"fresh-k{power}-{basis}")
    circuit.reset(range(3))
    circuit.h(0)
    _repeat_base(circuit, 0, 1, 2, power)
    if basis == "sin":
        circuit.sdg(0)
    circuit.h(0)
    circuit.measure(0, 0)
    source_ledger(circuit)
    return circuit


def source_ledger(circuit: QuantumCircuit) -> dict:
    """Strict expanded source guard; conditional bodies use a common upper envelope.

    This is a logical source count, not synthesis, physical work or scheduling.
    Opaque gates, loops, arbitrary conditions and unbound parameters are refused.
    """
    if not isinstance(circuit, QuantumCircuit) or circuit.num_qubits > MAX_QUBITS \
            or circuit.num_clbits > MAX_CLASSICAL_BITS or circuit.num_parameters:
        raise ValueError("Fixed tranche permits <=6 wires and <=4 classical bits")
    if circuit.num_qubits < 1 or (1 << circuit.num_clbits) > MAX_RETAINED_RECORDS:
        raise ValueError("Invalid width or retained-record bound")
    if float(circuit.global_phase) != 0:
        raise ValueError("Only the explicit zero-global-phase source representation is admitted")

    def visit(block: QuantumCircuit, depth: int = 0) -> tuple[Counter, int]:
        if depth > 1:
            raise ValueError("Only one level of single-bit feedback is admitted")
        count = Counter()
        decisions = 0
        for item in block.data:
            operation = item.operation
            if operation.name == "if_else":
                condition = operation.condition
                if not isinstance(condition, tuple) or len(condition) != 2 \
                        or condition[1] != 1 or condition[0] not in block.clbits:
                    raise ValueError("Feedback must compare a stored bit with one")
                if len(operation.blocks) != 1:
                    raise ValueError("Only a true-arm correction and empty false arm are admitted")
                branch_count, nested_decisions = visit(operation.blocks[0], depth + 1)
                if nested_decisions or branch_count != Counter({"sdg": 1}):
                    raise ValueError("Only the frozen one-S-dagger correction is admitted")
                count.update(branch_count)
                decisions += 1
            elif operation.name in PRIMITIVES:
                for parameter in operation.params:
                    if not math.isfinite(float(parameter)):
                        raise ValueError("Source parameters must be bound finite numbers")
                count[operation.name] += 1
            else:
                raise ValueError(f"Forbidden/opaque source operation: {operation.name}")
            if sum(count.values()) + decisions > MAX_EXPANDED_INSTRUCTIONS:
                raise ValueError("Expanded source exceeds 10000-instruction case cap")
        return count, decisions

    counts, decisions = visit(circuit)
    unitary = {name: counts[name] for name in sorted(counts) if name not in {"measure", "reset"}}
    return {
        "qubits": circuit.num_qubits,
        "classical_bits": circuit.num_clbits,
        "maximum_retained_records": 1 << circuit.num_clbits,
        "expanded_instructions": sum(counts.values()) + decisions,
        "unitary_instructions_upper_bound": sum(unitary.values()),
        "unitary_by_name": unitary,
        "resets": counts["reset"],
        "readouts": counts["measure"],
        "feedback_decisions": decisions,
        "scope": "unexecuted bounded source; conditional corrections counted in common upper envelope",
    }
