"""Original ripple-carry modular arithmetic with explicit clean work registers.

Uses a Cuccaro-style MAJ/UMA carry chain, not its optimized endpoint variant.
All arithmetic operations are X, CX, CCX and controlled SWAP. The valid-domain
contract is y<N and clean constant/carry/borrow workspace; the circuit has an
unspecified reversible extension on dirty/invalid inputs.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Iterator, Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    XGate, CXGate, CCXGate, SwapGate, CSwapGate, HGate, PhaseGate, CPhaseGate,
    ZGate, SGate, SdgGate, TGate, TdgGate,
)

from .shor import (
    MAX_INTEGER_BITS, MAX_PHASE_BITS, MAX_PRIMITIVE_GATES,
    validate_instance, qft_gate, decompose, analytic_phase_distribution, classical_order,
)

MAX_STATEVECTOR_QUBITS = 20
MAX_SPARSE_PHASE_BITS = 8
MAX_SPARSE_STATES = 8192
MAX_SPARSE_GATE_APPLICATIONS = 4_000_000
MAX_SOURCE_PRIMITIVES = 100_000
MAX_SOURCE_QUBITS = 64
MAX_DENSE_GATE_AMPLITUDE_WORK = 1 << 32
BOOLEAN_GATES = {"x", "cx", "ccx", "swap", "cswap"}
SOURCE_GATES = BOOLEAN_GATES | {"h", "p", "cp", "z", "s", "sdg", "t", "tdg"}
SOURCE_GATE_CLASSES = {"x": XGate, "cx": CXGate, "ccx": CCXGate,
                       "swap": SwapGate, "cswap": CSwapGate, "h": HGate,
                       "p": PhaseGate, "cp": CPhaseGate, "z": ZGate,
                       "s": SGate, "sdg": SdgGate, "t": TGate, "tdg": TdgGate}


def _integer(name: str, value: int, minimum: int, maximum: int):
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum},{maximum}]")


@lru_cache(maxsize=32)
def ripple_adder_gate(width: int):
    """|a,b,c> -> |a,(a+b+c) mod2**width,c>, including restored carry-in.

    Layout: width a bits; width b bits; one carry-in. No carry-out is exposed.
    MAJ/UMA endpoints are intentionally unoptimized: 2m CCX and4m CX.
    """
    _integer("width", width, 1, MAX_INTEGER_BITS + 1)
    circuit = QuantumCircuit(2 * width + 1, name=f"rADD{width}")
    a, b, carry = list(range(width)), list(range(width, 2 * width)), 2 * width
    for bit in range(width):
        previous = carry if bit == 0 else a[bit - 1]
        circuit.cx(a[bit], b[bit])
        circuit.cx(a[bit], previous)
        circuit.ccx(b[bit], previous, a[bit])
    for bit in reversed(range(width)):
        previous = carry if bit == 0 else a[bit - 1]
        circuit.ccx(b[bit], previous, a[bit])
        circuit.cx(a[bit], previous)
        circuit.cx(previous, b[bit])
    return circuit.to_gate()


def _constant_add(circuit: QuantumCircuit, constant: int, y: list[int], register: list[int], carry: int, controls: list[int]):
    """Prepare the controlled constant, ripple-add, then unprepare it."""
    value = constant % (1 << len(y))
    if value == 0:
        return
    bits = [register[bit] for bit in range(len(register)) if (value >> bit) & 1]
    for wire in bits:
        if len(controls) == 2:
            circuit.ccx(controls[0], controls[1], wire)
        elif len(controls) == 1:
            circuit.cx(controls[0], wire)
        else:
            circuit.x(wire)
    circuit.append(ripple_adder_gate(len(y)), [*register, *y, carry])
    for wire in reversed(bits):
        if len(controls) == 2:
            circuit.ccx(controls[0], controls[1], wire)
        elif len(controls) == 1:
            circuit.cx(controls[0], wire)
        else:
            circuit.x(wire)


@lru_cache(maxsize=256)
def modular_adder_gate_ripple(modulus: int, constant: int, num_controls: int = 0):
    """Controlled y<-y+b modN. Layout controls;y(n+1);borrow;constant(n+1);carry.

    All constant bits, carry and borrow start/end zero, and y<N. Subtraction
    of N runs even on an inactive control branch; the borrow restoration and
    cleanup cancel there. No controlled opaque adder is required.
    """
    _integer("modulus", modulus, 3, 1 << MAX_INTEGER_BITS)
    n = (modulus - 1).bit_length()
    _integer("constant", constant, 0, modulus - 1)
    _integer("num_controls", num_controls, 0, 2)
    m = n + 1
    circuit = QuantumCircuit(num_controls + 2 * m + 2, name=f"rMod{constant}")
    controls = list(range(num_controls))
    y = list(range(num_controls, num_controls + m))
    flag = num_controls + m
    register = list(range(flag + 1, flag + 1 + m))
    carry = circuit.num_qubits - 1
    if constant == 0:
        return circuit.to_gate()
    _constant_add(circuit, constant, y, register, carry, controls)
    _constant_add(circuit, -modulus, y, register, carry, [])
    circuit.cx(y[-1], flag)
    _constant_add(circuit, modulus, y, register, carry, [flag])
    _constant_add(circuit, -constant, y, register, carry, controls)
    circuit.x(y[-1]); circuit.cx(y[-1], flag); circuit.x(y[-1])
    _constant_add(circuit, constant, y, register, carry, controls)
    return circuit.to_gate()


@lru_cache(maxsize=128)
def controlled_multiplier_gate_ripple(modulus: int, multiplier: int):
    """Control;x(n);y(n+1);borrow;constant(n+1);carry, total3n+5 wires.

    Valid x<N, all work zero. Compute weighted additions, controlled swap,
    then inverse-multiplier subtraction clean the old input and all work.
    """
    n = validate_instance(modulus, multiplier)
    circuit = QuantumCircuit(3 * n + 5, name=f"rM{multiplier}")
    control = 0
    x = list(range(1, n + 1))
    y = list(range(n + 1, 2 * n + 2))
    flag = 2 * n + 2
    register = list(range(flag + 1, flag + n + 2))
    carry = circuit.num_qubits - 1
    if multiplier == 1:
        return circuit.to_gate()
    for bit, wire in enumerate(x):
        value = multiplier * pow(2, bit, modulus) % modulus
        circuit.append(modular_adder_gate_ripple(modulus, value, 2), [control, wire, *y, flag, *register, carry])
    for first, second in zip(x, y[:n]):
        circuit.cswap(control, first, second)
    inverse = pow(multiplier, -1, modulus)
    for bit, wire in reversed(list(enumerate(x))):
        value = inverse * pow(2, bit, modulus) % modulus
        circuit.append(modular_adder_gate_ripple(modulus, value, 2).inverse(), [control, wire, *y, flag, *register, carry])
    return circuit.to_gate()


def build_order_finding_circuit_ripple(modulus: int = 15, base: int = 2, phase_bits: int = 4, *, measure: bool = False):
    n = validate_instance(modulus, base)
    _integer("phase_bits", phase_bits, 1, MAX_PHASE_BITS)
    if 650 * n * phase_bits * (n + 1) + 10 * phase_bits**2 > MAX_PRIMITIVE_GATES:
        raise ValueError("Ripple arithmetic preflight rejects this materialization")
    circuit = QuantumCircuit(phase_bits + 3 * n + 4, phase_bits if measure else 0, name=f"rOrder{modulus}")
    phase = list(range(phase_bits))
    x = list(range(phase_bits, phase_bits + n))
    work = list(range(phase_bits + n, circuit.num_qubits))
    circuit.x(x[0]); circuit.h(phase)
    for bit, wire in enumerate(phase):
        multiplier = pow(base, 1 << bit, modulus)
        if multiplier != 1:
            circuit.append(controlled_multiplier_gate_ripple(modulus, multiplier), [wire, *x, *work])
    circuit.append(qft_gate(phase_bits).inverse(), phase)
    if measure:
        circuit.measure(phase, range(phase_bits))
    return circuit


def source_primitives(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    """Expand only defined source gates, preserving globals, with a fixed cap."""
    if circuit.num_parameters:
        raise ValueError("Bind source parameters")
    if circuit.num_qubits > MAX_SOURCE_QUBITS:
        raise ValueError("Source qubit cap exceeded")
    result = []
    def visit(current, mapping, depth):
        if depth > 16:
            raise ValueError("Source decomposition is too deeply nested")
        phase = float(current.global_phase)
        if not math.isfinite(phase):
            raise ValueError("Source global phase must be finite")
        if phase:
            result.append(("gphase", (), (phase,)))
        for item in current.data:
            name = item.operation.name
            wires = tuple(mapping[current.find_bit(bit).index] for bit in item.qubits)
            if name in SOURCE_GATES and item.operation.base_class is SOURCE_GATE_CLASSES[name]:
                parameters = tuple(float(parameter) for parameter in item.operation.params)
                if any(not math.isfinite(value) for value in parameters):
                    raise ValueError("Source gate angles must be finite")
                result.append((name, wires, parameters))
            elif item.operation.definition is not None:
                visit(item.operation.definition, wires, depth + 1)
            else:
                raise ValueError(f"Unsupported opaque/dynamic source operation {name}")
            if len(result) > MAX_SOURCE_PRIMITIVES:
                raise ValueError("Source primitive cap exceeded")
    visit(circuit, tuple(range(circuit.num_qubits)), 0)
    return result


def _boolean_step(value: int, name: str, wires: tuple[int, ...]) -> int:
    if name == "x":
        return value ^ (1 << wires[0])
    if name == "cx":
        return value ^ (1 << wires[1]) if (value >> wires[0]) & 1 else value
    if name == "ccx":
        return value ^ (1 << wires[2]) if (value >> wires[0]) & (value >> wires[1]) & 1 else value
    if name == "cswap" and not ((value >> wires[0]) & 1):
        return value
    left, right = wires[-2:]
    return value ^ ((1 << left) | (1 << right)) if ((value >> left) ^ (value >> right)) & 1 else value


def simulate_boolean(circuit: QuantumCircuit, value: int) -> int:
    """Exact classical trajectory of X/CX/CCX/SWAP gates; no phase theorem."""
    _integer("input basis value", value, 0, (1 << circuit.num_qubits) - 1)
    for name, wires, _ in source_primitives(circuit):
        if name not in BOOLEAN_GATES:
            raise ValueError("Boolean simulation accepts only explicit Boolean gates")
        value = _boolean_step(value, name, wires)
    return value


def simulate_boolean_batch(circuit: QuantumCircuit, values: list[int]) -> list[int]:
    """Reuse one explicit source lowering for a bounded list of basis inputs."""
    if len(values) > 4096:
        raise ValueError("Boolean batch input cap exceeded")
    operations = source_primitives(circuit)
    if len(operations) * len(values) > MAX_SPARSE_GATE_APPLICATIONS:
        raise ValueError("Boolean gate-input work cap exceeded")
    for value in values:
        _integer("input basis value", value, 0, (1 << circuit.num_qubits) - 1)
    current = list(values)
    for name, wires, _ in operations:
        if name not in BOOLEAN_GATES:
            raise ValueError("Boolean simulation accepts only explicit Boolean gates")
        current = [_boolean_step(value, name, wires) for value in current]
    return current


def sparse_statevector(circuit: QuantumCircuit) -> tuple[dict[int, complex], dict[str, int]]:
    """Bounded independent source-gate simulator of complex pure amplitudes.

    H/phase gates are actually applied, as well as every Boolean primitive.
    No amplitudes are truncated. Refuses excessive states or gate-state work.
    This is CPU sparse simulation, not a quantum device execution.
    """
    amplitudes = {0: 1 + 0j}
    work, peak = 0, 1
    for name, wires, params in source_primitives(circuit):
        work += len(amplitudes)
        if work > MAX_SPARSE_GATE_APPLICATIONS:
            raise ValueError("Sparse gate-state work cap exceeded")
        updated = {}
        if name in BOOLEAN_GATES:
            for value, amplitude in amplitudes.items():
                target = _boolean_step(value, name, wires)
                updated[target] = updated.get(target, 0) + amplitude
        elif name == "h":
            mask = 1 << wires[0]
            for value, amplitude in amplitudes.items():
                zero = value & ~mask
                updated[zero] = updated.get(zero, 0) + amplitude / math.sqrt(2)
                updated[zero | mask] = updated.get(zero | mask, 0) + (-amplitude if value & mask else amplitude) / math.sqrt(2)
        else:
            angle = params[0] if name in {"p", "cp", "gphase"} else {"z": math.pi, "s": math.pi / 2, "sdg": -math.pi / 2, "t": math.pi / 4, "tdg": -math.pi / 4}[name]
            for value, amplitude in amplitudes.items():
                active = all((value >> wire) & 1 for wire in wires)
                updated[value] = amplitude * np.exp(1j * angle) if active else amplitude
        amplitudes = {value: amplitude for value, amplitude in updated.items() if amplitude != 0}
        peak = max(peak, len(amplitudes))
        if len(amplitudes) > MAX_SPARSE_STATES:
            raise ValueError("Sparse amplitude-state cap exceeded")
    return amplitudes, {"peak_nonzero_amplitudes": peak, "source_gate_state_applications": work}


def simulate_order_finding_ripple(modulus: int, base: int, phase_bits: int, *, qiskit: bool = False) -> dict[str, Any]:
    _integer("sparse phase bits", phase_bits, 1, MAX_SPARSE_PHASE_BITS)
    circuit = build_order_finding_circuit_ripple(modulus, base, phase_bits)
    amplitudes, metrics = sparse_statevector(circuit)
    probabilities = np.zeros(1 << phase_bits)
    n = (modulus - 1).bit_length()
    dirty_probability = 0.0
    for value, amplitude in amplitudes.items():
        weight = abs(amplitude) ** 2
        probabilities[value & ((1 << phase_bits) - 1)] += weight
        if value >> (phase_bits + n):
            dirty_probability += weight
    expected = analytic_phase_distribution(classical_order(modulus, base), phase_bits)
    discrepancy = float(np.max(np.abs(probabilities - expected)))
    if discrepancy > 1e-10 or dirty_probability > 1e-10 or abs(float(probabilities.sum()) - 1) > 1e-10:
        raise AssertionError("Ripple circuit disagrees with independent order reference or leaves dirty work")
    result = {"performed": True, "method": "bounded source-gate complex sparse statevector; actual H/phase/Boolean primitives",
              "modulus": modulus, "base": base, "phase_bits": phase_bits, "total_qubits": circuit.num_qubits,
              "probabilities": probabilities.tolist(), "reference_probabilities": expected.tolist(),
              "maximum_probability_error": discrepancy, "dirty_work_probability": dirty_probability,
              "norm": float(probabilities.sum()), "qiskit_dense_check_performed": False, **metrics}
    if qiskit:
        if circuit.num_qubits > MAX_STATEVECTOR_QUBITS:
            raise ValueError("Qiskit statevector qubit cap exceeded")
        if len(source_primitives(circuit)) * (1 << circuit.num_qubits) > MAX_DENSE_GATE_AMPLITUDE_WORK:
            raise ValueError("Qiskit dense gate-amplitude work cap exceeded")
        actual = Statevector.from_instruction(circuit).data
        reference = np.zeros_like(actual)
        for value, amplitude in amplitudes.items():
            reference[value] = amplitude
        error = float(np.max(np.abs(actual - reference)))
        if error > 1e-10:
            raise AssertionError("Independent sparse source simulator disagrees with Qiskit complex amplitudes")
        result["qiskit_dense_check_performed"] = True
        result["maximum_qiskit_complex_amplitude_error"] = error
    return result
