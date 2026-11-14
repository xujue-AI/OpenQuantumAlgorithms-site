"""Explicit, conditional surface-code accounting for complete measured instruments.

The source reset/measurement schedule is retained. Actual ancilla-free
Ross--Selinger Clifford+T approximants replace rotations. A separate serialized
patch model charges initialization, nondestructive reset, readout, classical
feedback, routing, magic-state injection and capped factory supply retries.
It uses direct QDK Trace accounting, never the reset-dropping import/PSSPC path.
This is an engineering scenario with numerical synthesis checks, not a hardware
or formal fault-tolerance certificate and not the frozen Grover baseline.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
from fractions import Fraction
import hashlib
import importlib.metadata
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qasm3, transpile
from qiskit.circuit import Barrier, Measure, Reset
from qiskit.circuit.library import (CXGate, CZGate, HGate, IGate, PhaseGate, RXGate,
                                   RYGate, RZGate, SGate, SdgGate, TGate, TdgGate,
                                   UGate, XGate, YGate, ZGate)
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, Statevector, partial_trace
from qiskit.synthesis.discrete_basis.ross_selinger import gridsynth_unitary
from qiskit.transpiler import CouplingMap

ROTATION_BASIS = ("h", "x", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx")
CLIFFORD_T = {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx"}
_DIRECT_UNITARY_CLASSES = {CXGate, CZGate, HGate, IGate, PhaseGate, RXGate, RYGate,
                           RZGate, SGate, SdgGate, TGate, TdgGate, UGate, XGate,
                           YGate, ZGate}
MAX_QUBITS = 5
MAX_SOURCE_OPERATIONS = 10_000
MAX_SYNTHESIZED_OPERATIONS = 100_000
QDK_SOURCE = "81de0d2e5347e1e9f10e05066320fffdb3b7058c"
NUMERICAL_GATE_GUARD = 1e-12
MAX_FACTORY_RETRIES = 16


@dataclass(frozen=True)
class InstrumentLimits:
    """Explicit opt-in bounds; existing five-wire defaults remain unchanged.

    Eight wires is a bounded validation/materialization domain, not an
    arbitrary-size instrument or physical-model extrapolation. The ensemble
    branch cap may reject even a small schedule with repeated discarded events.
    """
    max_qubits: int = MAX_QUBITS
    source_operations: int = MAX_SOURCE_OPERATIONS
    synthesized_operations: int = MAX_SYNTHESIZED_OPERATIONS
    validation_operations: int = MAX_SYNTHESIZED_OPERATIONS
    validation_branches: int = 256

    def __post_init__(self):
        ceilings = {"max_qubits": 8, "source_operations": 100_000,
                    "synthesized_operations": 1_000_000,
                    "validation_operations": 1_000_000, "validation_branches": 256}
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in1..{maximum}")


def _limits(value: InstrumentLimits) -> InstrumentLimits:
    if not isinstance(value, InstrumentLimits):
        raise ValueError("limits must be an InstrumentLimits instance")
    return value


@dataclass(frozen=True)
class InstrumentProfile:
    physical_error_rate: float = 1e-4
    physical_gate_time_ns: int = 100
    physical_measurement_time_ns: int = 500
    physical_initialization_time_ns: int = 100
    code_cycle_time_ns: int = 1000
    reaction_time_ns: int = 10_000
    controller_failure_per_decision: float = 1e-15
    memory_prefactor: float = 0.1
    initialization_residual_prefactor: float = 0.1
    memory_threshold: float = 0.01
    factory_pipeline_failure_bound: float = 0.01
    factory_feedback_rounds_reserved: int = 3
    maximum_code_distance: int = 49

    def __post_init__(self):
        if not 0 < self.physical_error_rate < self.memory_threshold <= 0.01:
            raise ValueError("Physical error must be below the declared surface-code threshold")
        for name in ("physical_gate_time_ns", "physical_measurement_time_ns", "physical_initialization_time_ns", "code_cycle_time_ns", "reaction_time_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be positive integer nanoseconds")
        if not 0 <= self.controller_failure_per_decision < 1 or not 0 < self.factory_pipeline_failure_bound < 1:
            raise ValueError("Invalid controller/factory failure scenario")
        if not math.isfinite(self.memory_prefactor) or self.memory_prefactor <= 0:
            raise ValueError("Memory prefactor must be positive and finite")
        if not math.isfinite(self.initialization_residual_prefactor) or self.initialization_residual_prefactor < 0:
            raise ValueError("Initialization residual prefactor must be finite and nonnegative")
        if self.factory_pipeline_failure_bound < .01:
            raise ValueError("Factory readiness cannot improve the native 1% pipeline requirement without a new model")
        if not isinstance(self.factory_feedback_rounds_reserved, int) or self.factory_feedback_rounds_reserved < 3:
            raise ValueError("Reserve at least three factory feedback rounds")
        if self.code_cycle_time_ns < 5*self.physical_gate_time_ns+self.physical_measurement_time_ns:
            raise ValueError("Code cycle is shorter than the declared syndrome extraction depth")
        if not isinstance(self.maximum_code_distance, int) or not 3 <= self.maximum_code_distance <= 61 or self.maximum_code_distance % 2 == 0:
            raise ValueError("Maximum code distance must be odd, between 3 and 61")

    def patch_failure_per_cycle(self, distance: int) -> float:
        return self.memory_prefactor * (self.physical_error_rate / self.memory_threshold) ** ((distance + 1) // 2)

    def initialization_failure_scenario(self, distance: int) -> float:
        """Separate residual encoder assumption, in addition to code exposure."""
        return self.initialization_residual_prefactor * (self.physical_error_rate / self.memory_threshold) ** ((distance + 1) // 2)


@dataclass
class SynthesizedInstrument:
    source: QuantumCircuit
    routed: QuantumCircuit
    circuit: QuantumCircuit
    report: dict[str, Any]


def reset_kraus() -> Kraus:
    """Z measurement followed by X iff outcome=1, with the outcome discarded."""
    return Kraus([np.array([[1, 0], [0, 0]], complex), np.array([[0, 1], [0, 0]], complex)])


def t_injection_kraus(inverse: bool = False) -> Kraus:
    """Ideal CNOT + magic-port Z readout + conditional S (or S-dagger)."""
    sign = -1 if inverse else 1
    magic = np.array([1, np.exp(sign * 1j * np.pi / 4)]) / math.sqrt(2)
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)  # Data q0, magic port q1.
    isometry = Operator(circuit).data @ np.kron(magic.reshape(2, 1), np.eye(2))
    correction = np.diag([1, np.exp(sign * 1j * np.pi / 2)])
    return Kraus([isometry[:2], correction @ isometry[2:]])


def _source_check(circuit: QuantumCircuit, operation_limit: int = MAX_SOURCE_OPERATIONS,
                  *, max_qubits: int = MAX_QUBITS) -> None:
    if not 1 <= circuit.num_qubits <= max_qubits or len(circuit.data) > operation_limit:
        raise ValueError("Instrument source exceeds its explicit width/operation bound")
    if circuit.num_parameters:
        raise ValueError("Bind all parameters before instrument resource estimation")
    if not any(item.operation.name == "measure" for item in circuit.data):
        raise ValueError("The complete trial must include output measurements")
    initialized = set()
    for item in circuit.data:
        if item.operation.name in {"if_else", "while_loop", "for_loop", "switch_case", "store", "initialize"} or getattr(item.operation, "condition", None) is not None:
            raise ValueError("Only fixed full-attempt schedules are supported; conditional reset/T corrections are explicit model macros")
        if item.operation.name == "barrier":
            if item.operation.base_class is not Barrier:
                raise ValueError("Only canonical barrier instructions may be ignored")
            continue
        if item.operation.name == "reset" and (item.operation.base_class is not Reset
                or item.operation.num_qubits != 1 or item.operation.num_clbits != 0):
            raise ValueError("Only canonical single-wire reset instructions are supported")
        if item.operation.name == "measure" and (item.operation.base_class is not Measure
                or item.operation.num_qubits != 1 or item.operation.num_clbits != 1):
            raise ValueError("Only canonical single-wire measurement instructions are supported")
        wires = {circuit.find_bit(bit).index for bit in item.qubits}
        if item.operation.name == "reset":
            initialized.update(wires)
        elif not wires <= initialized:
            raise ValueError("Complete trials must explicitly initialize every used system/environment/label wire")
    if initialized != set(range(circuit.num_qubits)):
        raise ValueError("Every reserved input wire needs an explicit initial reset")


def synthesize_instrument(circuit: QuantumCircuit, operator_error_allowance: float = 1e-4,
                          *, limits: InstrumentLimits = InstrumentLimits(),
                          preserve_full_unitary: bool = False) -> SynthesizedInstrument:
    """Produce an actual Clifford+T circuit without dropping resets/readout.

    Routing is on a bidirectional logical line. Approximation sequences are
    cached only for exactly identical bound gates. The sum of guarded numerical
    Frobenius errors upper-estimates the coherent operator error; this numerical
    check is not a formal interval certificate. There is no mean-fit T proxy.
    """
    limits = _limits(limits)
    if not isinstance(preserve_full_unitary, bool):
        raise ValueError("preserve_full_unitary must be Boolean")
    _source_check(circuit, operation_limit=limits.source_operations, max_qubits=limits.max_qubits)
    if not math.isfinite(operator_error_allowance) or not 1e-9 <= operator_error_allowance <= 0.1:
        raise ValueError("Synthesis allowance must lie in the bounded numerical domain [1e-9, .1]")
    # Compiler scheduling may otherwise move commuting readouts across the
    # declared acceptance/output boundary. Barriers preserve the source event
    # order; they do not add cost to our already whole-device serial schedule.
    scheduled = circuit.copy_empty_like()
    for item in circuit.data:
        event = item.operation.name in {"reset", "measure"}
        if event:
            scheduled.barrier()
        scheduled.append(item.operation, item.qubits, item.clbits)
        if event:
            scheduled.barrier()
    # The default retains the published compiler setting. Callers needing the
    # whole coherent action explicitly disable clean-input assumptions.
    initially_zero = not preserve_full_unitary
    lowered = transpile(scheduled, basis_gates=list(ROTATION_BASIS), optimization_level=0,
                        seed_transpiler=7, qubits_initially_zero=initially_zero)
    routed = transpile(lowered, basis_gates=list(ROTATION_BASIS),
                       coupling_map=CouplingMap.from_line(circuit.num_qubits, bidirectional=True),
                       initial_layout=list(range(circuit.num_qubits)), routing_method="sabre",
                       optimization_level=0, seed_transpiler=7,
                       qubits_initially_zero=initially_zero)
    if len(routed.data) > limits.synthesized_operations:
        raise ValueError("Routed instrument exceeds the explicit synthesized-operation cap")
    rotations = sum(item.operation.name in {"rx", "ry", "rz"} for item in routed.data)
    per_rotation = operator_error_allowance / max(1, rotations)
    if per_rotation <= 4 * NUMERICAL_GATE_GUARD:
        raise ValueError("Requested error is below the guarded numerical synthesis resolution")
    output = QuantumCircuit(*routed.qregs, *routed.cregs, name="Clifford+T complete instrument")
    output.global_phase = routed.global_phase
    approximants = {}
    errors = []
    for item in routed.data:
        name = item.operation.name
        if name in {"rx", "ry", "rz"}:
            angle = float(item.operation.params[0])
            if not math.isfinite(angle):
                raise ValueError("Rotation angles must be finite")
            key = (name, angle.hex())
            if key not in approximants:
                matrix = np.asarray(item.operation.to_matrix(), dtype=complex)
                epsilon = per_rotation / 4
                for _ in range(4):
                    approximate = gridsynth_unitary(matrix, epsilon)
                    error = float(np.linalg.norm(Operator(approximate).data - matrix, "fro")) + NUMERICAL_GATE_GUARD
                    if error <= per_rotation:
                        break
                    epsilon /= 2
                else:
                    raise RuntimeError("Actual synthesis did not meet its guarded per-gate allowance")
                if set(approximate.count_ops()) - CLIFFORD_T:
                    raise RuntimeError("Synthesis returned unsupported non-Clifford+T operations")
                approximants[key] = (approximate, error, epsilon)
            approximate, error, _ = approximants[key]
            output.compose(approximate, qubits=list(item.qubits), inplace=True)
            errors.append(error)
        else:
            if name not in CLIFFORD_T | {"reset", "measure", "barrier"}:
                raise ValueError(f"Opaque operation remains: {name}")
            output.append(item.operation, item.qubits, item.clbits)
        if len(output.data) > limits.synthesized_operations:
            raise ValueError("Clifford+T expansion exceeds its bounded operation count")
    for name in ("reset", "measure"):
        if output.count_ops().get(name, 0) != circuit.count_ops().get(name, 0):
            raise RuntimeError("Routing/synthesis changed the instrument event count")
    def event_order(value):
        return [(item.operation.name, value.find_bit(item.clbits[0]).index if item.clbits else None)
                for item in value.data if item.operation.name in {"reset", "measure"}]
    if event_order(output) != event_order(circuit):
        raise RuntimeError("Routing/synthesis changed the reset/measurement event order")
    for item in output.data:
        if item.operation.name == "cx":
            a, b = (output.find_bit(bit).index for bit in item.qubits)
            if abs(a-b) != 1:
                raise RuntimeError("A non-neighbor CNOT remains after logical routing")
    bound = math.fsum(errors)
    if bound > operator_error_allowance:
        raise RuntimeError("Guarded total synthesis error exceeded its allocation")
    report = {
        "qiskit_version": importlib.metadata.version("qiskit"),
        "limits": asdict(limits),
        "preserve_full_unitary": preserve_full_unitary,
        "transpiler_qubits_initially_zero": initially_zero,
        "method": "actual ancilla-free Ross--Selinger gridsynth_unitary",
        "native_synthesis_backend": {
            "repository": "https://github.com/qiskit-community/rsgridsynth",
            "qiskit_262_cargo_lock_version": "0.2.2",
            "upstream_tag_commit": "0e94e5638c42624f4ec358f5eab079942462d2bf",
            "cargo_checksum": "133fdfbe9489a7c448e35e82c8443c9097ed6b5c22a492c3c78c19a472cacfad",
            "license": "MIT",
            "identity_scope": "Qiskit 2.6.2 Cargo.lock dependency; installed wheel exposes no direct native-crate version certificate",
            "seed": None,
            "seed_note": "gridsynth_unitary exposes no seed argument; exact generated sequences and hashes are saved",
        },
        "routing": "bidirectional logical line; fixed initial layout; SABRE seed 7; optimization level 0",
        "event_order": "all-wire compiler barriers preserve every source reset/measurement boundary; barriers have zero additional modeled serial cost",
        "source_counts": dict(circuit.count_ops()), "routed_counts": dict(routed.count_ops()),
        "clifford_t_counts": dict(output.count_ops()), "unique_rotation_targets": len(approximants),
        "rotation_occurrences": rotations, "operator_error_allowance": operator_error_allowance,
        "guarded_numerical_operator_error_upper_estimate": bound,
        "per_matrix_frobenius_guard": NUMERICAL_GATE_GUARD,
        "formal_interval_certificate": False, "actual_synthesis_sequences_generated": True,
        "approximants": [{"gate": key[0], "angle_hex": key[1], "epsilon_requested": value[2],
                          "guarded_frobenius_error": value[1], "gate_counts": dict(value[0].count_ops()),
                          "global_phase": float(value[0].global_phase)} for key, value in approximants.items()],
        "compiled_global_phase": float(output.global_phase),
        "qasm_export_scope": "complete measured trials; exporter omits the overall phase, so not a phase-preserving controlled-oracle specification",
    }
    output.metadata = {"instrument_synthesis": report}
    return SynthesizedInstrument(circuit.copy(), routed, output, report)


def simulate_trial(circuit: QuantumCircuit, outcome_clbit: int,
                   accept_clbits: dict[int, int] | None = None,
                   *, limits: InstrumentLimits = InstrumentLimits()) -> dict[str, Any]:
    """Exact small density/instrument check; resets are CPTP and labels projected.

    Acceptance measurements must precede output readout. Other measured outcomes
    are discarded by dephasing, not silently chosen. No conditional unitary is
    inferred. Every trial's full schedule is counted separately by the model.
    """
    limits = _limits(limits)
    if limits.max_qubits > MAX_QUBITS:
        checked = simulate_scored_trial(circuit, (outcome_clbit,), (1.0, -1.0),
                                        accept_clbits, limits=limits)
        checked["output_kind"] = "one accepted or unconditional measured +/-1 mean"
        return checked
    _source_check(circuit, operation_limit=limits.validation_operations, max_qubits=limits.max_qubits)
    accept_clbits = dict(accept_clbits or {})
    if outcome_clbit in accept_clbits or any(value not in (0, 1) for value in accept_clbits.values()):
        raise ValueError("Acceptance and output bits must be distinct")
    rho = DensityMatrix.from_int(0, 2**circuit.num_qubits)
    projections = [np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)]
    written, accepted = set(), set()
    outcome = None
    reduced = None
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if name == "barrier":
            continue
        if name == "reset":
            rho = rho.evolve(reset_kraus(), wires)
        elif name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in written:
                raise ValueError("Measurement-bit reuse requires a richer classical record model")
            written.add(bit)
            if bit in accept_clbits:
                if outcome is not None:
                    raise ValueError("Acceptance must be recorded before output measurement")
                rho = rho.evolve(Operator(projections[accept_clbits[bit]]), wires)
                accepted.add(bit)
            else:
                if bit == outcome_clbit:
                    if accepted != set(accept_clbits):
                        raise ValueError("Output precedes a required acceptance measurement")
                    probability = float(np.trace(rho.data).real)
                    if probability <= 0:
                        raise ValueError("The accepted branch has zero probability")
                    reduced = partial_trace(rho, [q for q in range(circuit.num_qubits) if q != wires[0]]).data / probability
                    outcome = float(np.trace(np.diag([1, -1]) @ reduced).real)
                rho = rho.evolve(Kraus(projections), wires)
        else:
            if outcome is not None:
                raise ValueError("Quantum work after output readout is outside this trial checker")
            rho = rho.evolve(item.operation, wires)
    if outcome is None or accepted != set(accept_clbits):
        raise ValueError("Required output/acceptance measurements are missing")
    return {"acceptance_probability": float(np.trace(rho.data).real), "mean": outcome,
            "reduced_before_output": reduced, "output_kind": "one accepted or unconditional measured +/-1 mean"}


def simulate_scored_trial(circuit: QuantumCircuit, outcome_clbits,
                          score_values, accept_clbits: dict[int, int] | None = None,
                          *, limits: InstrumentLimits = InstrumentLimits()) -> dict[str, Any]:
    """Bounded joint-output instrument validation by an unnormalized ensemble.

    Reset and discarded readouts split Kraus branches; accepted labels project
    only the declared outcome. No nonzero branch is pruned. Output probabilities
    use outcome_clbits tuple order, first bit least significant. The full fixed
    schedule is checked, including measurements after the first output; no
    quantum operation may follow that first output. Hardware/instrument faults
    and the physical serialized scenario are not simulated here.
    """
    limits = _limits(limits)
    _source_check(circuit, operation_limit=limits.validation_operations, max_qubits=limits.max_qubits)
    outputs = tuple(outcome_clbits)
    acceptance = dict(accept_clbits or {})
    def valid_bit(bit):
        return not isinstance(bit, bool) and isinstance(bit, int) and 0 <= bit < circuit.num_clbits
    if (not outputs or len(outputs) > circuit.num_qubits or len(set(outputs)) != len(outputs) or not all(valid_bit(bit) for bit in outputs)
            or any(not valid_bit(bit) for bit in acceptance)):
        raise ValueError("Output/acceptance bits must be distinct valid classical indices")
    if set(outputs) & set(acceptance) or any(value not in (0, 1) for value in acceptance.values()):
        raise ValueError("Acceptance and output bits must be distinct binary conditions")
    scores = np.asarray(tuple(score_values), dtype=float)
    if scores.shape != (2**len(outputs),) or not np.all(np.isfinite(scores)):
        raise ValueError("Finite score table must have2**len(outcome_clbits) entries")
    # Statevector.evolve may sample an embedded reset/measurement. Such hidden
    # instruments must never bypass this explicit Kraus schedule.
    # Canonical matrix-backed primitives cost one operation. For other gate
    # definitions, cache a conservative expanded cost but charge that cost at
    # every occurrence; a small shared definition graph can represent huge work.
    checked_operations, active_operations, definition_items = {}, set(), 0
    def unitary_definition(operation, depth=0):
        nonlocal definition_items
        if (operation.name in {"reset", "measure", "initialize", "if_else", "while_loop", "for_loop", "switch_case", "store"}
                or operation.num_clbits or getattr(operation, "condition", None) is not None):
            raise ValueError("Hidden or conditional instrument operations are outside the explicit schedule")
        if depth > 32:
            raise ValueError("Nested unitary definition depth exceeds the validation cap")
        if operation.base_class in _DIRECT_UNITARY_CLASSES:
            return 1, 0
        identity = id(operation)
        if identity in active_operations:
            raise ValueError("Cyclic unitary definitions are outside the validation domain")
        if identity in checked_operations:
            cost, height = checked_operations[identity]
            if depth+height > 32:
                raise ValueError("Nested unitary definition depth exceeds the validation cap")
            return cost, height
        active_operations.add(identity)
        cost, height = 1, 0
        definition = operation.definition
        if definition is not None:
            definition_items += len(definition.data)
            if definition_items > limits.validation_operations:
                raise ValueError("Nested unitary definitions exceed the validation operation cap")
            for nested in definition.data:
                nested_cost, nested_height = unitary_definition(nested.operation, depth+1)
                cost += nested_cost
                height = max(height, nested_height+1)
                if cost > limits.validation_operations:
                    raise ValueError("Expanded unitary definitions exceed the validation operation cap")
        active_operations.remove(identity)
        checked_operations[identity] = cost, height
        return cost, height
    # Preflight the complete schedule, never just the favorable accepted path.
    measured, accepted, mapping = set(), set(), {}
    output_started = False
    validation_work = 0
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        cost = 1
        if name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            wire = circuit.find_bit(item.qubits[0]).index
            if bit in measured:
                raise ValueError("Measurement-bit reuse requires a richer classical record model")
            measured.add(bit)
            if bit in acceptance:
                if output_started:
                    raise ValueError("Acceptance must be recorded before output measurement")
                accepted.add(bit)
            if bit in outputs:
                if accepted != set(acceptance):
                    raise ValueError("Output precedes a required acceptance measurement")
                output_started = True
                mapping[bit] = wire
        elif output_started:
            raise ValueError("Quantum work after output readout is outside this trial checker")
        elif name != "reset":
            cost, _ = unitary_definition(item.operation)
        validation_work += cost
        if validation_work > limits.validation_operations:
            raise ValueError("Expanded unitary occurrences exceed the validation operation cap")
    if set(mapping) != set(outputs) or accepted != set(acceptance):
        raise ValueError("Required output/acceptance measurements are missing")
    output_wires = tuple(mapping[bit] for bit in outputs)
    if len(set(output_wires)) != len(output_wires):
        raise ValueError("Joint outputs must measure distinct quantum wires")

    dimension = 2**circuit.num_qubits
    initial = np.zeros(dimension, dtype=complex)
    initial[0] = 1
    vectors = [initial]
    peak_branches = 1
    indices = np.arange(dimension)
    chunk = QuantumCircuit(circuit.num_qubits)
    def flush():
        nonlocal vectors, chunk
        if chunk.data:
            vectors = [Statevector(vector).evolve(chunk).data for vector in vectors]
            chunk = QuantumCircuit(circuit.num_qubits)
    def split(wire, reset=False):
        nonlocal vectors, peak_branches
        mask = 1 << wire
        zero_indices = indices[(indices & mask) == 0]
        one_indices = zero_indices | mask
        expanded = []
        for vector in vectors:
            zero_branch = np.zeros(dimension, complex)
            zero_branch[zero_indices] = vector[zero_indices]
            if np.any(zero_branch != 0):
                expanded.append(zero_branch)
            one_branch = np.zeros(dimension, complex)
            one_branch[zero_indices if reset else one_indices] = vector[one_indices]
            if np.any(one_branch != 0):
                expanded.append(one_branch)
            if len(expanded) > limits.validation_branches:
                raise ValueError("Instrument ensemble branch cap exceeded; no branches were discarded")
        vectors = expanded
        peak_branches = max(peak_branches, len(vectors))
    def project(wire, value):
        nonlocal vectors
        keep = ((indices >> wire) & 1) == value
        selected = []
        for vector in vectors:
            branch = np.where(keep, vector, 0)
            if np.any(branch != 0):
                selected.append(branch)
        vectors = selected
    reduced = probabilities = probability = None
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if name == "barrier":
            continue
        if name not in {"reset", "measure"}:
            chunk.append(item.operation, wires)
            continue
        flush()
        if name == "reset":
            split(wires[0], reset=True)
            continue
        bit = circuit.find_bit(item.clbits[0]).index
        if bit in acceptance:
            project(wires[0], acceptance[bit])
        elif bit in outputs:
            if reduced is None:
                probability = math.fsum(float(np.vdot(v, v).real) for v in vectors)
                if not math.isfinite(probability) or probability <= 0:
                    raise ValueError("The accepted branch has zero or unrepresentable probability")
                other_wires = tuple(q for q in range(circuit.num_qubits) if q not in output_wires)
                output_index = np.zeros(dimension, dtype=int)
                other_index = np.zeros(dimension, dtype=int)
                for k, q in enumerate(output_wires):
                    output_index |= ((indices >> q) & 1) << k
                for k, q in enumerate(other_wires):
                    other_index |= ((indices >> q) & 1) << k
                reduced = np.zeros((2**len(outputs), 2**len(outputs)), complex)
                for vector in vectors:
                    amplitudes = np.zeros((2**len(outputs), 2**len(other_wires)), complex)
                    amplitudes[output_index, other_index] = vector
                    reduced += amplitudes @ amplitudes.conj().T
                reduced /= probability
                probabilities = np.diag(reduced).real.copy()
            # Preflight established that only commuting final measurements
            # remain. Their joint law was captured before the first output.
        elif reduced is None:
            split(wires[0])
    if reduced is None:
        raise ValueError("Required output readout was not reached")
    return {
        "acceptance_probability": probability, "mean": float(scores @ probabilities),
        "joint_probabilities": probabilities, "reduced_before_output": reduced,
        "outcome_clbits": list(outputs), "outcome_wires": list(output_wires),
        "score_values": scores, "score_range": [float(min(scores)), float(max(scores))],
        "output_kind": "one accepted or unconditional bounded scalar score from joint final readout",
        "simulation": {"method": "unnormalized pure-state Kraus ensemble; exact declared logical operations evaluated numerically",
                       "peak_branches": peak_branches, "limits": asdict(limits),
                       "conservative_expanded_operations": validation_work,
                       "final_readout_handling": "joint law captured before first output; all remaining commuting readouts are validated and counted, without constructing an unused post-readout ensemble",
                       "scope": "bounded fixed schedule; no nonzero branch pruning; not a hardware or arbitrary-size instrument simulation"},
    }


def majority_path_error(distance: int, physical_error: float) -> float:
    """Simple independent-readout scenario: majority of d disjoint length-d paths.

    This is an explicit decoder assumption, not a fit or optimized surface-code
    decoder. Perfect prior code state is conditional on no counted logical fault.
    """
    if isinstance(distance, bool) or not isinstance(distance, int) or distance < 3 or distance % 2 == 0 or not 0 <= physical_error < .5:
        raise ValueError("Use an odd distance >=3 and independent readout probability in [0,.5)")
    parity_error = -math.expm1(distance * math.log1p(-2*physical_error)) / 2
    return math.fsum(math.comb(distance, k) * parity_error**k * (1-parity_error)**(distance-k)
                     for k in range((distance+1)//2, distance+1))


def _event_counts(circuit: QuantumCircuit) -> Counter:
    counts, active = Counter(), set()
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        wire = circuit.find_bit(item.qubits[0]).index
        if name == "reset":
            counts["initial_reset" if wire not in active else "reset"] += 1
        elif name in {"t", "tdg"}:
            counts[f"{name}_q{wire}"] += 1
        else:
            counts[name] += 1
        active.update(circuit.find_bit(bit).index for bit in item.qubits)
    return counts


def _costs(profile: InstrumentProfile, distance: int, name: str, factory_ns: int,
           factory_retry_cap: int) -> dict[str, int]:
    dtime = distance * profile.code_cycle_time_ns
    init = profile.physical_initialization_time_ns + dtime + profile.reaction_time_ns
    destructive = profile.physical_measurement_time_ns + profile.reaction_time_ns
    # Deliberately separate scratch preparation/recycling rather than overlap it.
    parity = 2*init + dtime + destructive
    cnot = 3*parity + 2*profile.reaction_time_ns
    if name == "initial_reset":
        return dict(time_ns=init, readouts=0, preparations=1, decisions=1)
    if name == "reset":
        return dict(time_ns=parity+2*profile.reaction_time_ns+dtime, readouts=1, preparations=2, decisions=5)
    if name == "measure":
        return dict(time_ns=parity, readouts=1, preparations=2, decisions=3)
    if name == "cx":
        return dict(time_ns=cnot, readouts=3, preparations=6, decisions=11)
    if name == "trial_control":
        return dict(time_ns=profile.reaction_time_ns, readouts=0, preparations=0, decisions=1)
    if name.startswith(("t_q", "tdg_q")):
        wire = int(name.split("_q")[1])
        # Factory port -> data-aligned workspace: (wire+1) neighboring SWAPs.
        cnots = 3*(wire+1)+1  # Plus the actual injection CNOT.
        inverse_frame = int(name.startswith("tdg"))
        round_overhead = profile.physical_initialization_time_ns + profile.physical_measurement_time_ns + profile.reaction_time_ns
        supply = factory_retry_cap * (factory_ns + profile.factory_feedback_rounds_reserved*round_overhead + profile.reaction_time_ns)
        return dict(time_ns=supply + cnots*cnot + destructive + profile.reaction_time_ns + init + inverse_frame*profile.reaction_time_ns,
                    readouts=3*cnots+1, preparations=6*cnots+1,
                    decisions=11*cnots+3+inverse_frame+factory_retry_cap*(profile.factory_feedback_rounds_reserved+1))
    if name in CLIFFORD_T - {"t", "tdg", "cx"}:
        # Local Clifford frames are tracked; their controller work is charged.
        return dict(time_ns=profile.reaction_time_ns, readouts=0, preparations=0, decisions=1)
    raise ValueError(f"Unsupported accounting operation {name}")


def _pareto(points: list[dict]) -> list[dict]:
    selected = []
    for point in sorted(points, key=lambda p: (p["peak_physical_qubits"], p["runtime_per_complete_attempt_ns"])):
        if not selected or point["runtime_per_complete_attempt_ns"] < selected[-1]["runtime_per_complete_attempt_ns"]:
            selected.append(point)
    return selected


def _ceil_div(numerator: int, denominator: int) -> int:
    """Integer ceiling, preserving nanosecond precision for long schedules."""
    return -(-numerator // denominator)


def _factory_retry_cap(total_states: int, failure_probability: float, budget: float) -> int:
    """Smallest cap <=16 using exact decimal interpretations of the inputs.

    Logarithms can underflow in budget/count or round across an integer cap.
    This bounded rational search keeps the stated union bound exact, without
    imposing a narrower count limit on the circuit's workload.
    """
    if isinstance(total_states, bool) or not isinstance(total_states, int) or total_states < 0:
        raise ValueError("Magic-state count must be a nonnegative integer")
    failure, allowance = Fraction(str(failure_probability)), Fraction(str(budget))
    if not 0 < failure < 1 or not 0 < allowance < 1:
        raise ValueError("Factory failure probability and budget must lie in (0,1)")
    if total_states == 0:
        return 0
    bound = Fraction(total_states)
    for cap in range(1, MAX_FACTORY_RETRIES+1):
        bound *= failure
        if bound <= allowance:
            return cap
    raise ValueError("Factory retry cap exceeds this bounded model domain")


def estimate_instrument(synthesized: SynthesizedInstrument, total_attempts: int, *,
                        profile: InstrumentProfile = InstrumentProfile(),
                        physical_failure_budget: float = .0025,
                        factory_availability_budget: float = .0025,
                        cache_dir: Path,
                        include_all_feasible_points: bool = False) -> dict[str, Any]:
    """Direct QDK accounting of a fully serialized, explicit instrument model.

    Native factory supply is the only reused QDK calibration. Its output error
    and its separate 1% pipeline rejection requirement are never confused.
    All rejected algorithm/factory attempts and idle code-cycle exposure are
    charged. No factory-memory error is added again to its supplied output model.
    include_all_feasible_points exposes the existing bounded candidate search
    for shared-configuration multi-group accounting; it changes no scenario or
    Pareto rule and is absent from the default report.
    """
    if importlib.metadata.version("qdk") != "1.34.0":
        raise RuntimeError("This native accounting interface was audited with QDK 1.34.0")
    if isinstance(total_attempts, bool) or not isinstance(total_attempts, int) or total_attempts < 1:
        raise ValueError("total_attempts must count a positive capped complete schedule")
    if not 0 < physical_failure_budget < 1 or not 0 < factory_availability_budget < 1:
        raise ValueError("Separate physical and factory-availability budgets are required")
    if not isinstance(include_all_feasible_points, bool):
        raise ValueError("include_all_feasible_points must be Boolean")
    # Retain the constructor's explicit domain rather than extrapolating a
    # forged or legacy report to arbitrary width. Legacy reports keep five wires.
    bound_fields = synthesized.report.get("limits")
    bounds = InstrumentLimits(**bound_fields) if isinstance(bound_fields, dict) else InstrumentLimits()
    _source_check(synthesized.source, operation_limit=bounds.source_operations, max_qubits=bounds.max_qubits)
    _source_check(synthesized.circuit, operation_limit=bounds.synthesized_operations, max_qubits=bounds.max_qubits)
    from qdk.qre import ErrorComposition, LOGICAL, Trace, constant_function, linear_function
    from qdk.qre.instruction_ids import T
    from qdk.qre.models import GateBased, RoundBasedFactory, SurfaceCode

    circuit = synthesized.circuit
    counts = _event_counts(circuit)
    counts["trial_control"] += 1
    n = circuit.num_qubits
    # Two interleaved rows for data/bridges and transport/bridges + handoff bridge.
    patches = 4*n-1
    counts["initial_reset"] += patches-n  # Explicit startup of reserved workspace.
    t_count = sum(count for name, count in counts.items() if name.startswith(("t_q", "tdg_q")))
    total_t = t_count*total_attempts
    factory_retry_cap = _factory_retry_cap(total_t, profile.factory_pipeline_failure_bound,
                                          factory_availability_budget)
    availability_failure = float(total_t*Fraction(str(profile.factory_pipeline_failure_bound))**factory_retry_cap)
    points = []
    for distance in range(3, profile.maximum_code_distance+1, 2):
        ctx = GateBased(error_rate=profile.physical_error_rate, gate_time=profile.physical_gate_time_ns,
                        measurement_time=profile.physical_measurement_time_ns).context()
        supplies = (list(RoundBasedFactory.q(code_query=SurfaceCode.q(distance=distance,
                                                                   code_cycle_override=profile.code_cycle_time_ns,
                                                                   error_correction_threshold=profile.memory_threshold),
                                           cache_dir=Path(cache_dir)).enumerate(ctx)) if t_count else [None])
        for supply in supplies:
            factory = supply[T] if supply is not None else None
            factory_error = float(factory.expect_error_rate()) if factory is not None else 0.0
            if factory_error*total_t > physical_failure_budget*.75:
                continue
            # This strict quality requirement also excludes a physical final stage.
            if factory is not None and factory_error >= 7.1*profile.physical_error_rate:
                continue
            factory_ns = int(factory.expect_time()) if factory is not None else 0
            patch_size = 2*distance**2-1
            readout_error = majority_path_error(distance, profile.physical_error_rate)
            preparation_error = profile.initialization_failure_scenario(distance)
            rows, nodes = [], []
            trace = Trace(patches)
            for index, (name, count) in enumerate(sorted(counts.items())):
                cost = _costs(profile, distance, name, factory_ns, factory_retry_cap)
                event_error = cost["readouts"]*readout_error + cost["preparations"]*preparation_error + cost["decisions"]*profile.controller_failure_per_decision
                instruction_id = 2**40+index
                nodes.append(ctx.add_instruction(instruction_id, encoding=LOGICAL, arity=None,
                                                 time=constant_function(cost["time_ns"]),
                                                 space=linear_function(patch_size), error_rate=constant_function(event_error)))
                # Whole-device reservation enforces the declared serial schedule.
                block = trace.add_block(repetitions=count)
                block.add_operation(instruction_id, list(range(patches)))
                rows.append(dict(operation=name, count=count, **cost, event_failure_model=event_error))
            if supply is not None:
                nodes.append(supply.node_index(T))
                trace.increment_resource_state(T, t_count)
            isa = ctx.make_isa(*nodes)
            runtime = sum(row["count"]*row["time_ns"] for row in rows)
            cycles = _ceil_div(runtime, profile.code_cycle_time_ns)
            memory_error = patches*cycles*profile.patch_failure_per_cycle(distance)
            # Extra output-port residence outside the native factory supply time:
            # reserved round overhead and the first transport SWAP into W0.
            round_overhead = profile.physical_initialization_time_ns + profile.physical_measurement_time_ns + profile.reaction_time_ns
            port_hold_ns = (profile.factory_feedback_rounds_reserved*round_overhead + profile.reaction_time_ns
                            + 3*_costs(profile, distance, "cx", factory_ns, factory_retry_cap)["time_ns"]) if t_count else 0
            port_hold_error = t_count*_ceil_div(port_hold_ns, profile.code_cycle_time_ns)*profile.patch_failure_per_cycle(distance)
            trace.increment_base_error(memory_error + port_hold_error)
            expected_error = memory_error + port_hold_error + sum(row["count"]*row["event_failure_model"] for row in rows) + t_count*factory_error
            if expected_error*total_attempts > physical_failure_budget:
                continue
            result = trace.estimate(isa, max_error=physical_failure_budget/total_attempts,
                                    composition=ErrorComposition.UnionBound)
            if result is None:
                continue
            if int(result.runtime) != runtime or not math.isclose(float(result.error), expected_error, rel_tol=1e-10, abs_tol=1e-30):
                raise RuntimeError("Native direct accounting differs from the explicit ledger")
            native_factory = result.factories.get(T)
            if native_factory is not None and native_factory.copies != 1:
                raise RuntimeError("Serialized just-in-time model unexpectedly duplicated factories")
            points.append({
                "data_code_distance": distance, "reserved_logical_patches": patches,
                "peak_physical_qubits": int(result.qubits), "data_workspace_physical_qubits": patches*patch_size,
                "factory_physical_qubits": int(factory.expect_space()) if factory is not None else 0,
                "runtime_per_complete_attempt_ns": runtime, "worst_case_runtime_all_attempts_ns": runtime*total_attempts,
                "code_cycles_per_complete_attempt": cycles,
                "full_physical_failure_model": float(result.error)*total_attempts,
                "failure_components_all_attempts": {
                    "data_workspace_code_exposure": memory_error*total_attempts,
                    "factory_output_handoff_storage": port_hold_error*total_attempts,
                    "logical_readout_scenario": sum(r["count"]*r["readouts"] for r in rows)*readout_error*total_attempts,
                    "logical_initialization_scenario": sum(r["count"]*r["preparations"] for r in rows)*preparation_error*total_attempts,
                    "controller_scenario": sum(r["count"]*r["decisions"] for r in rows)*profile.controller_failure_per_decision*total_attempts,
                    "factory_output_quality_model": total_t*factory_error,
                },
                "factory_supply": {"output_error": factory_error, "pipeline_time_ns": factory_ns,
                                   "fixed_logical_distillation_distance": distance,
                                   "native_qec_crossing_prefactor": 0.03,
                                   "native_qec_threshold": profile.memory_threshold,
                                   "native_code_cycle_time_ns": profile.code_cycle_time_ns,
                                   "quality_calibration_scope": "QDK factory model; distinct from the Litinski per-cycle data-exposure fit",
                                   "full_pipeline_retry_cap_per_state": factory_retry_cap,
                                   "states_all_complete_attempts": total_t,
                                   "worst_case_pipeline_attempts": total_t*factory_retry_cap,
                                   "availability_failure_union_bound": availability_failure,
                                   "retry_cap_arithmetic": "exact Fraction search over decimal input strings; smallest cap in 1..16",
                                   "pipeline_feedback_rounds_reserved": profile.factory_feedback_rounds_reserved,
                                   "extra_init_readout_feedback_dispatch_ns_per_pipeline": profile.factory_feedback_rounds_reserved*round_overhead + profile.reaction_time_ns,
                                   "extra_output_port_hold_ns_per_state": port_hold_ns},
                "macro_ledger_per_complete_attempt": rows,
                "accounting_trace_sha256": hashlib.sha256(trace.to_json().encode()).hexdigest(),
                "native_backend": "QDK direct Trace + explicit custom ISA; no import, PSSPC, LatticeSurgery or branch sampling",
            })
    if not points:
        raise RuntimeError("No feasible point in the declared distance/factory/scenario domain")
    return {
        "schema_version": 1, "status": "computed_conditional_instrument_surface_code_model",
        "limits": asdict(bounds),
        **({"all_feasible_points": points} if include_all_feasible_points else {}),
        "profile": asdict(profile), "complete_algorithm_attempts": total_attempts,
        "physical_failure_budget": physical_failure_budget, "factory_availability_budget": factory_availability_budget,
        "source_event_counts": dict(synthesized.source.count_ops()),
        "synthesized_event_counts": dict(circuit.count_ops()),
        "workspace_initializations_per_attempt": patches-n,
        "actual_synthesis": synthesized.report,
        "pareto_points": _pareto(points), "feasible_candidate_count": len(points),
        "comparison_scope": "separate serialized instrument model; not the frozen Grover/PSSPC baseline",
        "assumptions": [
            "All complete algorithm attempts run the entire supplied schedule, including rejected trials",
            "Data/workspace logical faults use Litinski's per-patch per-cycle scenario, homogeneously during active/idle/reset operations",
            "Readout uses a majority-of-disjoint-paths independent-error scenario; not an optimized decoder calibration",
            "Residual initialization error has a separately assumed code-distance prefactor in addition to code-volume exposure; it is not inferred from readout or hardware-calibrated",
            "Local Clifford frames are tracked with charged controller work; adjacent CNOT uses three protected Pauli-measurement slots supporting tracked Pauli bases",
            "Tracked Pauli bases, axis orientation at the magic port and protected parity geometry are conditional interface assumptions, not a certified lattice layout; T-dagger adds a charged inverse Clifford-frame decision",
            "Scratch initialization and recycling are charged separately; no favorable overlap is assumed",
            "One factory supplies states just in time; transport uses explicit neighboring logical SWAPs and injection uses CNOT/readout/conditional S plus port reinitialization",
            "QDK native factory output quality is conditional on its model; final logical stages use the declared distance, and compatible handoff is assumed",
            "Factory output quality already incorporates its quantum operations; no duplicate factory-memory fault term is added",
            "Additional output-port residence outside native supply time is charged separately; native factory init/readout timing remains a leading-order macro plus a declared round overhead reserve",
            "Independent complete factory retries use the QDK source-level 1% pipeline failure requirement; up to three feedback delays/decisions are reserved per pipeline",
            "Factory controller decisions are aggregate round-level decisions, not a count of every syndrome bit operation",
            "One trial accept/retry/counter dispatch interval is charged per complete algorithm attempt, and one retry dispatch interval per complete factory attempt",
            "Controller reliability/latency and logical macro geometry are scenario parameters, not measured hardware calibration",
            "Guarded floating synthesis and small instrument checks are numerical evidence, not formal interval certificates",
        ],
        "excluded": ["offline classical coefficient/circuit/synthesis preprocessing and final statistical reporting wall time", "factory/data interface layout certification", "measured hardware and decoder throughput validation"],
        "unsupported": ["source-level arbitrary if/else or adaptive quantum feedback; only fixed source schedules plus reset/T-correction and factory/trial dispatch macros are modeled"],
        "sources": [
            "https://quantum-journal.org/papers/q-2019-03-05-128/",
            "https://arxiv.org/abs/1111.4022", "https://arxiv.org/abs/1403.2975",
            "https://github.com/Qiskit/qiskit/blob/2.6.2/qiskit/synthesis/discrete_basis/ross_selinger.py",
            "https://github.com/qiskit-community/rsgridsynth/tree/v0.2.2",
            f"https://github.com/microsoft/qdk/blob/{QDK_SOURCE}/source/qdk_package/qdk/qre/models/factories/_round_based.py",
            f"https://github.com/microsoft/qdk/blob/{QDK_SOURCE}/source/qre/src/trace.rs",
        ],
        "reference_keys": ["resource_litinski_2019", "resource_horsman_2012", "resource_ross_selinger_2014", "resource_qiskit_262", "resource_rsgridsynth_022", "resource_qdk_134"],
    }
