"""Original measured-AND CCX replacement and ripple arithmetic.

The clean-helper construction follows Gidney's temporary logical-AND idea.
It is a drop-in channel replacement for CCX, not the paper's separately
optimized 4w-4 T adder.  Literal Clifford+T gates, measurements and feedback
are emitted.  Dense simulation and exact sparse cyclotomic simulation have
separate bounds; neither is implied by circuit materialization.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import cmath
import math
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import Clbit, Measure, Reset, Barrier
from qiskit.circuit.controlflow import IfElseOp
from qiskit.circuit.library import (
    HGate, XGate, ZGate, SGate, SdgGate, TGate, TdgGate,
    CXGate, CCXGate, SwapGate, CSwapGate, PhaseGate, CPhaseGate,
    UnitaryGate, Initialize, StatePreparation,
)
from qiskit.quantum_info import DensityMatrix, Operator

MAX_WIDTH = 31
MAX_QUBITS = 64
MAX_CLASSICAL_BITS = 64
MAX_SOURCE_EVENTS = 500_000
MAX_SOURCE_INPUT_EVENTS = 100_000
MAX_DENSE_QUBITS = 10
MAX_DENSE_EVENT_MATRIX_WORK = 1_200_000_000
MAX_EXACT_PHASE_BITS = 4
MAX_EXACT_STATES = 8192
MAX_EXACT_GATE_STATE_WORK = 20_000_000
MAX_COEFFICIENT_BITS = 4096

_CLASSES = {
    "h": HGate, "x": XGate, "z": ZGate, "s": SGate, "sdg": SdgGate,
    "t": TGate, "tdg": TdgGate, "cx": CXGate, "ccx": CCXGate,
    "swap": SwapGate, "cswap": CSwapGate, "p": PhaseGate, "cp": CPhaseGate,
}
_ARITY = {"h": 1, "x": 1, "z": 1, "s": 1, "sdg": 1, "t": 1,
          "tdg": 1, "p": 1, "cx": 2, "ccx": 3, "swap": 2,
          "cswap": 3, "cp": 2, "reset": 1, "measure": 1}


def _integer(name: str, value: int, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def _clear_mode(value: str) -> str:
    if type(value) is not str or value not in {"feedback_x", "reset"}:
        raise ValueError("helper clearing must be feedback_x or reset")
    return value


def append_temporary_and(circuit: QuantumCircuit, a: int, b: int, helper: int) -> None:
    """Exactly |a,b,0> -> |a,b,a&b>; no dirty-helper promise.

    The final Sdg removes the i on the clean |11,0> branch of the
    four-T relative-phase construction. No opaque RCCX instruction is used.
    """
    if len({a, b, helper}) != 3:
        raise ValueError("AND wires must be distinct")
    for wire in (a, b, helper):
        _integer("AND wire", wire, 0, circuit.num_qubits - 1)
    circuit.h(helper); circuit.t(helper)
    circuit.cx(b, helper); circuit.tdg(helper)
    circuit.cx(a, helper); circuit.t(helper)
    circuit.cx(b, helper); circuit.tdg(helper)
    circuit.h(helper); circuit.sdg(helper)


def build_temporary_and_compute() -> QuantumCircuit:
    circuit = QuantumCircuit(3, name="clean four-T AND")
    append_temporary_and(circuit, 0, 1, 2)
    circuit.metadata = {"kind": "clean_temporary_and_compute", "helper": 2,
                        "valid_input": "helper zero; data arbitrary",
                        "dirty_helper_extension": "not the CCX contract"}
    return circuit


def append_measured_ccx(circuit: QuantumCircuit, a: int, b: int, target: int,
                        helper: int, feedback: Clbit, *, clear_mode="feedback_x") -> None:
    """CCX on arbitrary data with a clean helper returned to zero.

    Both measured Kraus branches are CCX/sqrt(2). The classical bit can be
    reused only after this conditional correction completes.
    """
    _clear_mode(clear_mode)
    if len({a, b, target, helper}) != 4:
        raise ValueError("measured CCX wires must be distinct")
    for wire in (a, b, target, helper):
        _integer("measured CCX wire", wire, 0, circuit.num_qubits - 1)
    if feedback not in circuit.clbits:
        raise ValueError("feedback bit must belong to the circuit")
    append_temporary_and(circuit, a, b, helper)
    circuit.cx(helper, target)
    circuit.h(helper)
    circuit.measure(helper, feedback)
    with circuit.if_test((feedback, 1)):
        circuit.h(b); circuit.cx(a, b); circuit.h(b)
        if clear_mode == "feedback_x":
            circuit.x(helper)
    if clear_mode == "reset":
        circuit.reset(helper)


def build_measured_ccx(*, clear_mode="feedback_x", initialize_helper=True) -> QuantumCircuit:
    if type(initialize_helper) is not bool:
        raise ValueError("initialize_helper must be Boolean")
    _clear_mode(clear_mode)
    circuit = QuantumCircuit(4, 1, name=f"measured CCX ({clear_mode})")
    if initialize_helper:
        circuit.reset(3)
    append_measured_ccx(circuit, 0, 1, 2, 3, circuit.clbits[0], clear_mode=clear_mode)
    circuit.metadata = {"kind": "measured_ccx", "helper": 3, "feedback_clbit": 0,
                        "helper_clear_mode": clear_mode,
                        "initial_helper_reset": initialize_helper,
                        "ideal_branch_probability": "1/2",
                        "input_data_contract": "all three data qubits arbitrary, including external entanglement",
                        "physical_status": "not estimated; measurement/feedback/reset assumptions separate"}
    return circuit


def append_seven_t_ccx(circuit: QuantumCircuit, a: int, b: int, target: int) -> None:
    """Literal standard exact CCX decomposition; seven T/Tdg, no helper."""
    if len({a, b, target}) != 3:
        raise ValueError("CCX wires must be distinct")
    for wire in (a, b, target):
        _integer("CCX wire", wire, 0, circuit.num_qubits - 1)
    circuit.h(target); circuit.cx(b, target); circuit.tdg(target)
    circuit.cx(a, target); circuit.t(target)
    circuit.cx(b, target); circuit.tdg(target)
    circuit.cx(a, target); circuit.t(b); circuit.t(target); circuit.h(target)
    circuit.cx(a, b); circuit.t(a); circuit.tdg(b); circuit.cx(a, b)


def _static_source(circuit: QuantumCircuit) -> list[tuple[str, tuple[int, ...], tuple[float, ...]]]:
    if circuit.num_clbits or circuit.num_parameters or circuit.num_qubits > MAX_QUBITS - 1:
        raise ValueError("replacement accepts bound, unitary sources up to 63 qubits")
    result = []
    def visit(current, mapping, depth):
        if depth > 16:
            raise ValueError("source decomposition exceeds depth 16")
        phase = float(current.global_phase)
        if not math.isfinite(phase):
            raise ValueError("nonfinite source global phase")
        if phase:
            result.append(("gphase", (), (phase,)))
        for item in current.data:
            op = item.operation
            name = op.name
            wires = tuple(mapping[current.find_bit(q).index] for q in item.qubits)
            if isinstance(op, (UnitaryGate, Initialize, StatePreparation)):
                raise ValueError("opaque matrix/state source is not accepted")
            if name in _CLASSES and op.base_class is _CLASSES[name]:
                if len(wires) != _ARITY[name] or len(set(wires)) != len(wires):
                    raise ValueError("invalid source gate arity")
                parameters = tuple(float(p) for p in op.params)
                if any(not math.isfinite(p) for p in parameters):
                    raise ValueError("nonfinite source parameter")
                result.append((name, wires, parameters))
            elif name in {"unitary", "initialize", "state_preparation"}:
                raise ValueError("opaque matrix/state source is not accepted")
            elif op.definition is not None:
                visit(op.definition, wires, depth + 1)
            else:
                raise ValueError(f"unsupported source operation {name}")
            if len(result) > MAX_SOURCE_INPUT_EVENTS:
                raise ValueError("static source event limit exceeded")
    visit(circuit, tuple(range(circuit.num_qubits)), 0)
    return result


def _leaf(circuit, name, wires, params):
    if name == "gphase":
        circuit.global_phase += params[0]
    elif name == "p":
        circuit.p(params[0], wires[0])
    elif name == "cp":
        circuit.cp(params[0], *wires)
    else:
        getattr(circuit, name)(*wires)


def replace_ccx(source: QuantumCircuit, *, method="measured", clear_mode="feedback_x") -> QuantumCircuit:
    """Expand static arithmetic first, then replace each exact CCX channel.

    This is not a Gate with an automatic inverse/control operation. Apply
    static inverse/control before this transformation. Other source phase
    gates are preserved and reported separately from literal T counts.
    """
    if type(method) is not str or method not in {"static", "measured"}:
        raise ValueError("method must be static or measured")
    _clear_mode(clear_mode)
    primitives = _static_source(source)
    ccx_count = sum(name in {"ccx", "cswap"} for name, _, _ in primitives)
    estimate = sum({"ccx": 17 if method == "measured" else 15,
                    "cswap": 19 if method == "measured" else 17,
                    "swap": 3}.get(name, 1) for name, _, _ in primitives) + (method == "measured")
    if estimate > MAX_SOURCE_EVENTS:
        raise ValueError("replacement primitive preflight exceeds 500000")
    measured = method == "measured"
    circuit = QuantumCircuit(source.num_qubits + measured, int(measured), name=f"{source.name} {method} AND")
    helper = source.num_qubits if measured else None
    if measured:
        circuit.reset(helper)
    def ccx(a, b, t):
        if measured:
            append_measured_ccx(circuit, a, b, t, helper, circuit.clbits[0], clear_mode=clear_mode)
        else:
            append_seven_t_ccx(circuit, a, b, t)
    for name, wires, parameters in primitives:
        if name == "ccx":
            ccx(*wires)
        elif name == "cswap":
            c, a, b = wires
            circuit.cx(a, b); ccx(c, b, a); circuit.cx(a, b)
        elif name == "swap":
            a, b = wires
            circuit.cx(a, b); circuit.cx(b, a); circuit.cx(a, b)
        else:
            _leaf(circuit, name, wires, parameters)
    circuit.metadata = {
        "kind": "measured_and_replacement" if measured else "seven_t_baseline",
        "source_name": source.name, "source_qubits": source.num_qubits,
        "source_static_events": len(primitives), "replaced_ccx_including_cswap": ccx_count,
        "helper": helper, "feedback_clbit": 0 if measured else None,
        "helper_clear_mode": clear_mode if measured else None,
        "ideal_branch_probability": "1/2" if measured else None,
        "contract": "same ideal data channel as supplied static source; arithmetic input domain is separate",
        "controlled_oracle_status": "not a dynamically controlled Gate; lower static controls first",
        "physical_estimation_status": "not computed",
    }
    if len(primitive_events(circuit)) > MAX_SOURCE_EVENTS:
        raise ValueError("expanded replacement event limit exceeded")
    return circuit


def ripple_source(width: int) -> QuantumCircuit:
    """Original unoptimized MAJ/UMA |a,b,c> -> |a,b+a+c mod2**w,c>."""
    _integer("width", width, 1, MAX_WIDTH)
    circuit = QuantumCircuit(2 * width + 1, name=f"ripple{width}")
    a, b, carry = list(range(width)), list(range(width, 2 * width)), 2 * width
    for bit in range(width):
        previous = carry if bit == 0 else a[bit - 1]
        circuit.cx(a[bit], b[bit]); circuit.cx(a[bit], previous)
        circuit.ccx(b[bit], previous, a[bit])
    for bit in reversed(range(width)):
        previous = carry if bit == 0 else a[bit - 1]
        circuit.ccx(b[bit], previous, a[bit])
        circuit.cx(a[bit], previous); circuit.cx(previous, b[bit])
    circuit.metadata = {"kind": "unoptimized_MAJ_UMA", "width": width,
                        "ccx": 2 * width, "cx": 4 * width}
    return circuit


def build_ripple_core(width: int, *, method="measured", clear_mode="feedback_x") -> QuantumCircuit:
    source = ripple_source(width)
    circuit = replace_ccx(source, method=method, clear_mode=clear_mode)
    circuit.metadata.update(kind="measured_ripple_core" if method == "measured" else "static_ripple_core",
                            width=width, data_layout={"a": list(range(width)),
                            "b": list(range(width, 2 * width)), "carry": 2 * width},
                            algorithm_variant="drop-in measured-CCX; not the optimized 4w-4 adder")
    return circuit


def build_addition_demo(width=2, a=3, b=1, carry=0, *, method="measured", clear_mode="feedback_x", readout=True):
    _integer("width", width, 1, MAX_WIDTH)
    _integer("a", a, 0, (1 << width) - 1); _integer("b", b, 0, (1 << width) - 1)
    _integer("carry", carry, 0, 1)
    if type(readout) is not bool:
        raise ValueError("readout must be Boolean")
    core = build_ripple_core(width, method=method, clear_mode=clear_mode)
    data_qubits = 2 * width + 1
    feedback_bits = core.num_clbits
    circuit = QuantumCircuit(core.num_qubits, feedback_bits + (data_qubits if readout else 0), name=f"addition{width} {method}")
    for q in range(data_qubits):
        circuit.reset(q)
    initial = a | (b << width) | (carry << (2 * width))
    for q in range(data_qubits):
        if (initial >> q) & 1:
            circuit.x(q)
    circuit.compose(core, qubits=range(core.num_qubits), clbits=range(feedback_bits), inplace=True)
    if readout:
        circuit.measure(range(data_qubits), range(feedback_bits, feedback_bits + data_qubits))
    circuit.metadata = {**core.metadata, "kind": "complete_addition_demo", "a": a, "b": b,
                        "carry": carry, "input_basis": initial,
                        "expected_output_basis": a | (((a + b + carry) % (1 << width)) << width) | (carry << (2 * width)),
                        "data_qubits": data_qubits, "readout": readout,
                        "readout_clbits": list(range(feedback_bits, feedback_bits + data_qubits)) if readout else [],
                        "preparation": "explicit data reset and X; measured helper reset once",
                        "repetitions": 1, "sampling": "none required for the ideal deterministic addition output"}
    return circuit


def build_order_finding_measured(modulus=15, base=2, phase_bits=4, *, method="measured", clear_mode="feedback_x", readout=True):
    """Transform existing explicit ripple arithmetic; QFT phases are retained."""
    if type(readout) is not bool:
        raise ValueError("readout must be Boolean")
    from .shor_ripple import build_order_finding_circuit_ripple
    source = build_order_finding_circuit_ripple(modulus, base, phase_bits, measure=False)
    circuit = replace_ccx(source, method=method, clear_mode=clear_mode)
    if readout:
        circuit.add_register(ClassicalRegister(phase_bits, "phase_out"))
        offset = int(method == "measured")
        circuit.measure(range(phase_bits), range(offset, offset + phase_bits))
    circuit.metadata.update(kind="measured_order_finding" if method == "measured" else "static_order_finding",
                            modulus=modulus, base=base, phase_bits=phase_bits,
                            integer_bits=(modulus - 1).bit_length(), readout=readout,
                            data_qubits=source.num_qubits,
                            readout_clbits=list(range(int(method == "measured"), circuit.num_clbits)) if readout else [],
                            source_arithmetic="existing clean-work ripple modular powers",
                            phase_synthesis_status="P/CP preserved; total synthesized T cost not computed")
    return circuit


@dataclass(frozen=True)
class Event:
    name: str
    qubits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()
    clbits: tuple[int, ...] = ()
    condition: tuple[int, int] | None = None


def primitive_events(circuit: QuantumCircuit) -> list[Event]:
    """Read actual primitives and one-bit if bodies, never metadata or caches."""
    if circuit.num_qubits > MAX_QUBITS or circuit.num_clbits > MAX_CLASSICAL_BITS or circuit.num_parameters:
        raise ValueError("bound instrument qubit limit exceeded")
    events: list[Event] = []
    def visit(current, qmap, cmap, condition=None, depth=0):
        if depth > 8:
            raise ValueError("instrument nesting exceeds eight")
        phase = float(current.global_phase)
        if not math.isfinite(phase):
            raise ValueError("nonfinite instrument global phase")
        if phase:
            events.append(Event("gphase", params=(phase,), condition=condition))
        for item in current.data:
            op = item.operation
            q = tuple(qmap[current.find_bit(bit).index] for bit in item.qubits)
            c = tuple(cmap[current.find_bit(bit).index] for bit in item.clbits)
            if isinstance(op, IfElseOp):
                if condition is not None or not isinstance(op.condition, tuple) or len(op.condition) != 2:
                    raise ValueError("only unnested one-bit feedback is supported")
                bit, value = op.condition
                if not isinstance(bit, Clbit) or type(value) not in {int, bool} or value != 1:
                    raise ValueError("feedback must test one classical bit for one")
                index = cmap[current.find_bit(bit).index]
                blocks = op.blocks
                if len(blocks) > 2 or (len(blocks) == 2 and
                    (blocks[1].data or float(blocks[1].global_phase) != 0)):
                    raise ValueError("false feedback body must be absent or empty")
                visit(blocks[0], q, c, (index, 1), depth + 1)
                events.append(Event("end_if", clbits=(index,)))
            elif op.name in _CLASSES and op.base_class is _CLASSES[op.name]:
                if op.name in {"ccx", "swap", "cswap"}:
                    raise ValueError("instrument must already contain literal decomposed gates")
                if len(q) != _ARITY[op.name] or len(set(q)) != len(q):
                    raise ValueError("instrument gate arity mismatch")
                params = tuple(float(p) for p in op.params)
                if any(not math.isfinite(p) for p in params):
                    raise ValueError("nonfinite instrument angle")
                events.append(Event(op.name, q, params, condition=condition))
            elif ((op.name == "reset" and op.base_class is Reset)
                  or (op.name == "measure" and op.base_class is Measure)) and len(q) == 1:
                if op.name == "measure" and len(c) != 1:
                    raise ValueError("measure needs one classical bit")
                events.append(Event(op.name, q, clbits=c, condition=condition))
            elif op.name == "barrier" and op.base_class is Barrier:
                events.append(Event("barrier", q, condition=condition))
            else:
                raise ValueError(f"nonliteral instrument operation {op.name}")
            if len(events) > MAX_SOURCE_EVENTS:
                raise ValueError("instrument event limit exceeded")
    visit(circuit, tuple(range(circuit.num_qubits)), tuple(range(circuit.num_clbits)))
    return events


def logical_counts(circuit: QuantumCircuit) -> dict[str, Any]:
    events = primitive_events(circuit)
    base = Counter(e.name for e in events if e.condition is None and e.name not in {"gphase", "end_if", "barrier"})
    controlled = Counter(e.name for e in events if e.condition is not None and e.name not in {"gphase", "barrier"})
    maximum = base + controlled
    decisions = sum(e.name == "end_if" for e in events)
    def depth(run_conditional):
        last = {}
        for event in events:
            if event.name == "gphase" or (event.condition is not None and not run_conditional):
                continue
            keys = [("q", q) for q in event.qubits] + [("c", c) for c in event.clbits]
            if event.condition is not None:
                keys.append(("c", event.condition[0]))
            if not keys:
                continue
            level = max((last.get(key, 0) for key in keys), default=0) + 1
            for key in keys:
                last[key] = level
        return max(last.values(), default=0)
    return {"logical_qubits": circuit.num_qubits, "classical_bits": circuit.num_clbits,
            "unconditional_primitive_counts": dict(sorted(base.items())),
            "all_one_feedback_extra_counts": dict(sorted(controlled.items())),
            "all_zero_feedback_primitive_count": sum(base.values()),
            "all_one_feedback_primitive_count": sum(maximum.values()),
            "literal_t_or_tdg": maximum.get("t", 0) + maximum.get("tdg", 0),
            "unlowered_phase_operations": maximum.get("p", 0) + maximum.get("cp", 0),
            "classical_feedback_decisions": decisions,
            "dependency_depth_all_zero_feedback": depth(False),
            "dependency_depth_all_one_feedback": depth(True),
            "depth_scope": "conservative serial source/decision dependency depth, not an optimized parallel DAG or physical cycles",
            "included": ["literal gates", "explicit preparation/reset", "measured helper", "conditional correction/clear", "requested final readout"],
            "excluded_physical": ["code distance/factories", "routing/decoder", "calibrated measurement/reset/feedback faults and time", "P/CP synthesis if present"],
            "physical_estimation_status": "not computed"}


def addition_scaling(width: int, a=0, b=0, carry=0) -> dict[str, Any]:
    """Exact fixed-size integer count formulas; never materializes a circuit."""
    _integer("width", width, 1, 1_000_000)
    for name, value in (("a", a), ("b", b)):
        if type(value) is not int or value < 0 or value.bit_length() > width:
            raise ValueError(f"{name} must be a nonnegative integer fitting width")
    _integer("carry", carry, 0, 1)
    weight = a.bit_count() + b.bit_count() + carry
    return {"width": width, "input_x": weight,
            "static": {"qubits": 2 * width + 1, "literal_t_or_tdg": 14 * width,
                       "complete_primitives": 38 * width + 2 + weight},
            "measured_feedback_x": {"qubits": 2 * width + 2, "feedback_bits": 1,
                       "literal_t_or_tdg": 8 * width, "feedback_decisions": 2 * width,
                       "complete_primitives_all_zero": 34 * width + 3 + weight,
                       "complete_primitives_all_one": 42 * width + 3 + weight,
                       "ideal_expected_primitive_numerator": 38 * width + 3 + weight,
                       "expected_scope": "uniform ideal gadget outcomes; not physical runtime"},
            "materialization_performed": False, "accuracy_inherited": False,
            "physical_estimation_status": "not computed"}


def _gate(event: Event):
    if event.name == "p":
        return PhaseGate(event.params[0])
    if event.name == "cp":
        return CPhaseGate(event.params[0])
    return _CLASSES[event.name]()


def density_output(circuit: QuantumCircuit, *, initial_state=None) -> tuple[np.ndarray, dict[str, Any]]:
    """Bounded independent branch-density interpretation of actual events.

    Classical writes overwrite the reusable bit. Terminal readout is reported
    from its commuting projectors; it is not expanded into exponentially many
    collapsed density matrices.
    """
    n = circuit.num_qubits
    if n > MAX_DENSE_QUBITS:
        raise ValueError("dense instrument simulation exceeds ten qubits")
    events = primitive_events(circuit)
    dimension = 1 << n
    if len(events) * dimension * dimension > MAX_DENSE_EVENT_MATRIX_WORK:
        raise ValueError("dense event/matrix-work preflight exceeded")
    if initial_state is None:
        state = np.zeros(dimension, complex); state[0] = 1
    else:
        if len(initial_state) != dimension:
            raise ValueError("initial vector dimension mismatch")
        if isinstance(initial_state, np.ndarray):
            if initial_state.ndim != 1 or initial_state.shape != (dimension,) or initial_state.dtype.kind not in "biufc":
                raise ValueError("initial vector must be a one-dimensional numeric array")
        elif isinstance(initial_state, (list, tuple)):
            if any(not isinstance(x, (int, float, complex, np.number)) for x in initial_state):
                raise ValueError("initial vector must contain numeric scalars")
        else:
            raise ValueError("initial vector must be a bounded list, tuple or numeric array")
        state = np.asarray(initial_state, dtype=complex)
        if state.ndim != 1 or state.shape != (dimension,):
            raise ValueError("initial vector must be one-dimensional")
        if not np.all(np.isfinite(state)) or abs(float(np.vdot(state, state).real) - 1) > 1e-12:
            raise ValueError("initial vector must be finite and normalized")
    branches = {0: DensityMatrix(state)}
    rows = np.arange(dimension)
    peak = 1
    terminal = []
    for index, event in enumerate(events):
        if event.name == "measure" and all(e.name == "measure" and e.condition is None for e in events[index:]):
            terminal = events[index:]
            break
        if event.name in {"end_if", "barrier", "gphase"}:
            continue
        updated = {}
        for bits, density in branches.items():
            active = event.condition is None or ((bits >> event.condition[0]) & 1) == event.condition[1]
            if not active:
                updated[bits] = updated.get(bits, 0) + density.data
            elif event.name == "measure":
                wire, bit = event.qubits[0], event.clbits[0]
                selected = (rows >> wire) & 1
                for outcome in (0, 1):
                    matrix = density.data.copy()
                    matrix[selected != outcome, :] = 0; matrix[:, selected != outcome] = 0
                    key = (bits & ~(1 << bit)) | (outcome << bit)
                    updated[key] = updated.get(key, 0) + matrix
            elif event.name == "reset":
                matrix = density.reset(list(event.qubits)).data
                updated[bits] = updated.get(bits, 0) + matrix
            else:
                matrix = density.evolve(_gate(event), qargs=list(event.qubits)).data
                updated[bits] = updated.get(bits, 0) + matrix
        branches = {key: DensityMatrix(matrix) for key, matrix in updated.items()}
        peak = max(peak, len(branches))
        if peak > 2:
            raise ValueError("density interpreter supports one reusable feedback bit only")
    matrix = sum((value.data for value in branches.values()), np.zeros((dimension, dimension), complex))
    trace = complex(np.trace(matrix))
    if abs(trace - 1) > 1e-10 or np.max(np.abs(matrix - matrix.conj().T)) > 1e-10:
        raise AssertionError("raw density trace/Hermiticity guard failed")
    distribution = {}
    if terminal:
        for bits, density in branches.items():
            for value, probability in enumerate(np.real(np.diag(density.data))):
                key = bits
                for event in terminal:
                    bit = event.clbits[0]
                    key = (key & ~(1 << bit)) | (((value >> event.qubits[0]) & 1) << bit)
                distribution[key] = distribution.get(key, 0.0) + float(probability)
    return matrix, {"method": "actual primitive branch-density interpretation",
                    "qubits": n, "primitive_events": len(events), "maximum_classical_branches": peak,
                    "trace_real": trace.real, "trace_imag": trace.imag,
                    "terminal_readout_distribution": distribution,
                    "terminal_measurement_scope": "full final classical-register joint distribution, retaining feedback and overwriting written bits; commuting projectors of each prefix branch",
                    "floating_guard": 1e-10, "formal_proof": False}


@dataclass(frozen=True)
class Cyclotomic16:
    """Exact Q(zeta16) value with integer coefficients divided by 2**power.

    zeta16**8=-1. This computational representation is not a proof kernel.
    It prevents tiny floating garbage from exploding sparse support.
    """
    coefficients: tuple[int, ...]
    power: int = 0

    def __post_init__(self):
        if type(self.coefficients) is not tuple or len(self.coefficients) != 8 or any(type(x) is not int for x in self.coefficients):
            raise ValueError("cyclotomic value needs eight integer coefficients")
        _integer("denominator power", self.power, 0, MAX_COEFFICIENT_BITS)
        if any(abs(x).bit_length() > MAX_COEFFICIENT_BITS for x in self.coefficients):
            raise ValueError("cyclotomic coefficient cap exceeded")
        values, power = self.coefficients, self.power
        if not any(values):
            power = 0
        else:
            while power and all(x % 2 == 0 for x in values):
                values = tuple(x // 2 for x in values); power -= 1
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "power", power)

    @classmethod
    def one(cls):
        return cls((1, 0, 0, 0, 0, 0, 0, 0))

    @classmethod
    def zero(cls):
        return cls((0,) * 8)

    def __bool__(self):
        return any(self.coefficients)

    def __neg__(self):
        return Cyclotomic16(tuple(-x for x in self.coefficients), self.power)

    def __add__(self, other):
        power = max(self.power, other.power)
        left, right = power - self.power, power - other.power
        return Cyclotomic16(tuple((a << left) + (b << right) for a, b in zip(self.coefficients, other.coefficients)), power)

    def __sub__(self, other):
        return self + -other

    def shifted(self, exponent: int):
        _integer("root exponent", exponent, -(1 << 20), 1 << 20)
        exponent %= 16
        values = [0] * 8
        for index, value in enumerate(self.coefficients):
            quotient, remainder = divmod(index + exponent, 8)
            values[remainder] += value if quotient % 2 == 0 else -value
        return Cyclotomic16(tuple(values), self.power)

    def inv_sqrt2(self):
        value = self.shifted(2) - self.shifted(6)
        return Cyclotomic16(value.coefficients, value.power + 1)

    def sqrt2(self):
        return self.shifted(2) - self.shifted(6)

    def conjugate(self):
        result = Cyclotomic16.zero()
        for index, coefficient in enumerate(self.coefficients):
            monomial = Cyclotomic16((coefficient,) + (0,) * 7, self.power).shifted(-index)
            result = result + monomial
        return result

    def __mul__(self, other):
        values = [0] * 8
        for i, a in enumerate(self.coefficients):
            for j, b in enumerate(other.coefficients):
                quotient, remainder = divmod(i + j, 8)
                values[remainder] += a * b if quotient % 2 == 0 else -a * b
        return Cyclotomic16(tuple(values), self.power + other.power)

    def as_complex(self):
        value = sum((coefficient / (1 << self.power)) * cmath.exp(1j * math.pi * index / 8)
                    for index, coefficient in enumerate(self.coefficients))
        if self and value == 0:
            raise ValueError("nonzero exact amplitude underflows in floating display")
        return value


def _field_phase(angle: float) -> int:
    unit = math.pi / 8
    quotient = angle / unit
    if not math.isfinite(quotient) or abs(quotient) > (1 << 20):
        raise ValueError("exact phase-index bound exceeded")
    index = round(quotient)
    if angle != index * unit:
        raise ValueError("exact sparse simulation accepts only integer pi/8 phases")
    return index


def _field_apply(amplitudes: dict[int, Cyclotomic16], event: Event):
    if event.name == "gphase":
        exponent = _field_phase(event.params[0])
        return {value: amp.shifted(exponent) for value, amp in amplitudes.items()}
    if event.name == "barrier":
        return amplitudes
    updated = {}
    if event.name == "h":
        mask = 1 << event.qubits[0]
        for value, amplitude in amplitudes.items():
            zero = value & ~mask; half = amplitude.inv_sqrt2()
            updated[zero] = updated.get(zero, Cyclotomic16.zero()) + half
            updated[zero | mask] = updated.get(zero | mask, Cyclotomic16.zero()) + (-half if value & mask else half)
    elif event.name in {"x", "cx"}:
        for value, amplitude in amplitudes.items():
            active = event.name == "x" or ((value >> event.qubits[0]) & 1)
            target = value ^ (1 << event.qubits[-1]) if active else value
            updated[target] = updated.get(target, Cyclotomic16.zero()) + amplitude
    elif event.name in {"z", "s", "sdg", "t", "tdg", "p", "cp"}:
        exponent = _field_phase(event.params[0]) if event.name in {"p", "cp"} else {"z": 8, "s": 4, "sdg": -4, "t": 2, "tdg": -2}[event.name]
        for value, amplitude in amplitudes.items():
            active = all((value >> wire) & 1 for wire in event.qubits)
            updated[value] = amplitude.shifted(exponent) if active else amplitude
    elif event.name == "reset":
        wire = event.qubits[0]
        outcomes = {(value >> wire) & 1 for value in amplitudes}
        if len(outcomes) > 1:
            raise ValueError("exact pure-state reset needs a known basis-valued wire")
        for value, amplitude in amplitudes.items():
            target = value & ~(1 << wire)
            updated[target] = updated.get(target, Cyclotomic16.zero()) + amplitude
    else:
        raise ValueError(f"unsupported exact event {event.name}")
    return {value: amplitude for value, amplitude in updated.items() if amplitude}


def exact_sparse_output(circuit: QuantumCircuit) -> tuple[dict[int, Cyclotomic16], dict[str, Any]]:
    """Apply every literal gate and both feedback branches in Q(zeta16).

    Branches merge only after exact equality and helper restoration are checked.
    No amplitudes are truncated and no CCX oracle replaces the gadget gates.
    """
    events = primitive_events(circuit)
    metadata = circuit.metadata or {}
    helper, feedback = metadata.get("helper"), metadata.get("feedback_clbit")
    if helper is not None:
        _integer("declared helper", helper, 0, circuit.num_qubits - 1)
    if any(event.name == "end_if" for event in events):
        _integer("declared helper", helper, 0, circuit.num_qubits - 1)
        _integer("declared feedback bit", feedback, 0, circuit.num_clbits - 1)
        _clear_mode(metadata.get("helper_clear_mode"))
    amplitudes = {0: Cyclotomic16.one()}
    branches = None
    work, peak, merged = 0, 1, 0
    terminal = []
    for index, event in enumerate(events):
        if event.name == "measure" and all(e.name == "measure" and e.condition is None for e in events[index:]):
            terminal = events[index:]; break
        current_size = sum(len(branch) for branch in branches.values()) if branches is not None else len(amplitudes)
        work += current_size
        if work > MAX_EXACT_GATE_STATE_WORK:
            raise ValueError("exact sparse gate-state work cap exceeded")
        if event.name == "measure":
            if branches is not None or event.qubits != (helper,) or event.clbits != (feedback,):
                raise ValueError("exact feedback interpreter requires the declared reusable helper/bit")
            branches = {m: {value: amp for value, amp in amplitudes.items() if ((value >> helper) & 1) == m} for m in (0, 1)}
        elif event.name == "end_if":
            if branches is None or event.clbits != (feedback,):
                raise ValueError("unexpected feedback boundary")
            # Literal reset mode clears outside the if; delay equality to it.
            if metadata.get("helper_clear_mode") == "reset":
                continue
            if branches[0] != branches[1] or any((value >> helper) & 1 for value in branches[0]):
                raise AssertionError("actual corrected branches differ or leave a dirty helper")
            amplitudes = {value: amp.sqrt2() for value, amp in branches[0].items()}
            branches = None; merged += 1
        elif branches is not None:
            for outcome in (0, 1):
                if event.condition is None or outcome == event.condition[1]:
                    branches[outcome] = _field_apply(branches[outcome], event)
            if event.name == "reset" and metadata.get("helper_clear_mode") == "reset":
                if branches[0] != branches[1] or any((value >> helper) & 1 for value in branches[0]):
                    raise AssertionError("literal reset branches differ")
                amplitudes = {value: amp.sqrt2() for value, amp in branches[0].items()}
                branches = None; merged += 1
        else:
            if event.condition is not None:
                raise ValueError("feedback without a measured branch")
            amplitudes = _field_apply(amplitudes, event)
        peak = max(peak, sum(len(branch) for branch in branches.values()) if branches is not None else len(amplitudes))
        if peak > MAX_EXACT_STATES:
            raise ValueError("exact sparse state-support cap exceeded")
    if branches is not None:
        raise ValueError("unclosed measured helper branch")
    norm = sum((amp.conjugate() * amp for amp in amplitudes.values()), Cyclotomic16.zero())
    if norm != Cyclotomic16.one():
        raise AssertionError("exact sparse norm is not one")
    if helper is not None and any((value >> helper) & 1 for value in amplitudes):
        raise AssertionError("exact sparse helper is dirty")
    return amplitudes, {"method": "actual primitive exact Q(zeta16) sparse interpretation; both branches checked before merging",
                        "peak_amplitude_support": peak, "gate_state_applications": work,
                        "merged_uniform_feedback_pairs": merged, "norm_exactly_one": True,
                        "amplitude_truncation": False, "terminal_measurements": len(terminal),
                        "formal_proof": False,
                        "scope": "intended dyadic pi/8 logical-angle model recognized by exact Python spelling; not a binary-angle realization enclosure, general proof kernel or noise simulation"}


def gadget_checks() -> dict[str, Any]:
    prefix = QuantumCircuit(4)
    append_temporary_and(prefix, 0, 1, 3)
    prefix.cx(3, 2); prefix.h(3)
    actual = Operator(prefix).data
    expected = np.zeros((8, 8), complex)
    for value in range(8):
        output = value ^ 4 if (value & 1) and (value & 2) else value
        expected[output, value] = 1
    correction = np.diag([(-1 if (value & 1) and (value & 2) else 1) for value in range(8)])
    kraus = [actual[:8, :8], correction @ actual[8:, :8]]
    error = max(float(np.max(np.abs(block - expected / math.sqrt(2)))) for block in kraus)
    completeness = float(np.max(np.abs(sum(block.conj().T @ block for block in kraus) - np.eye(8))))
    if error > 1e-10 or completeness > 1e-10:
        raise AssertionError("measured CCX Kraus or completeness check failed")
    return {"kraus_columns": 8, "outcomes": 2, "max_kraus_error": error,
            "trace_preservation_error": completeness, "reference": "independent classical CCX permutation",
            "guarantee_scope": "bounded numerical all-data channel check; symbolic derivation separate",
            "physical_or_formal_certificate": False}


def check_addition(width, a, b, carry=0, *, method="measured", clear_mode="feedback_x"):
    _integer("density addition width", width, 1, 3)
    circuit = build_addition_demo(width, a, b, carry, method=method, clear_mode=clear_mode, readout=False)
    density, info = density_output(circuit)
    expected_value = circuit.metadata["expected_output_basis"]
    expected = np.zeros_like(density); expected[expected_value, expected_value] = 1
    discrepancy = float(np.max(np.abs(density - expected)))
    if discrepancy > 1e-10:
        raise AssertionError("actual addition differs from independent integer arithmetic")
    amplitudes, exact = exact_sparse_output(circuit)
    if amplitudes != {expected_value: Cyclotomic16.one()}:
        raise AssertionError("exact source addition differs from integer arithmetic")
    return {"width": width, "a": a, "b": b, "carry": carry, "method": method,
            "expected_output_basis": expected_value, "max_density_error": discrepancy,
            "density": info, "exact_sparse": exact,
            "reference": "a retained; b=(a+b+carry)mod2**width; carry retained"}


def check_order_finding(modulus=15, base=2, phase_bits=4, *, clear_mode="feedback_x"):
    _integer("exact phase bits", phase_bits, 1, MAX_EXACT_PHASE_BITS)
    _integer("exact modulus", modulus, 3, 33)
    from .shor import analytic_phase_distribution, classical_order
    circuit = build_order_finding_measured(modulus, base, phase_bits, clear_mode=clear_mode, readout=False)
    amplitudes, details = exact_sparse_output(circuit)
    probabilities = np.zeros(1 << phase_bits)
    n = (modulus - 1).bit_length()
    dirty_probability = 0.0
    for value, amplitude in amplitudes.items():
        probability = abs(amplitude.as_complex()) ** 2
        probabilities[value & ((1 << phase_bits) - 1)] += probability
        if value >> (phase_bits + n):
            dirty_probability += probability
    expected = analytic_phase_distribution(classical_order(modulus, base), phase_bits)
    error = float(np.max(np.abs(probabilities - expected)))
    if error > 1e-10 or dirty_probability > 1e-10 or abs(float(probabilities.sum()) - 1) > 1e-10:
        raise AssertionError("measured arithmetic order circuit fails independent phase/work checks")
    return {"modulus": modulus, "base": base, "phase_bits": phase_bits,
            "probabilities": probabilities.tolist(), "reference_probabilities": expected.tolist(),
            "max_probability_error": error, "dirty_work_probability": dirty_probability,
            "exact_sparse": details, "counts": logical_counts(circuit),
            "reference_scope": "independent classical order/Fourier distribution after circuit construction",
            "physical_estimation_status": "not computed; QFT phase synthesis and feedback costs remain separate"}
