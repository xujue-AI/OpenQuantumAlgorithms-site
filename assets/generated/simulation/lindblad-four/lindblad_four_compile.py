# SPDX-License-Identifier: MIT
"""Explicitly bounded Clifford+T materialization for the literal n4 increment.

This worker-owned domain preserves the full-unitary Qiskit 2.6.2 level0/SABRE
seed7 and Ross--Selinger policy of the frozen instrument adapter.  It does not
change that adapter's defaults or hard one-million-operation ceiling.  No
physical estimate, density approximation or controlled-oracle specification
is supplied here.  The complete density check belongs to the caller.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (CXGate, HGate, RXGate, RYGate, RZGate, SGate,
                                   SdgGate, TGate, TdgGate, XGate, YGate, ZGate)
from qiskit.quantum_info import Operator
from qiskit.transpiler import CouplingMap

if __package__:
    from .. import instrument_resources as ir
    from . import lindblad_four_density as di
else:
    import instrument_resources as ir
    import lindblad_four_density as di

ROTATION_BASIS = ir.ROTATION_BASIS
CLIFFORD_T = ir.CLIFFORD_T
NUMERICAL_GATE_GUARD = ir.NUMERICAL_GATE_GUARD
gridsynth_unitary = ir.gridsynth_unitary  # Only this new alias is runner-guarded.
COMPILER_DOMAIN = "literal-n4-explicit-opt-in-v1"
FIXED_SYNTHESIS_ALLOWANCE = 5e-5
FROZEN_IR_SOURCE_SHA256 = "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
LEGACY_OPERATION_CAP = 1_000_000
_PRIMITIVE_BASES = {"cx": CXGate, "h": HGate, "x": XGate, "y": YGate,
                    "z": ZGate, "s": SGate, "sdg": SdgGate,
                    "t": TGate, "tdg": TdgGate,
                    "rx": RXGate, "ry": RYGate, "rz": RZGate}


@dataclass(frozen=True)
class FourCompilerLimits:
    """Required limits for this separate eight-wire materialization domain."""

    max_qubits: int
    source_operations: int
    routed_operations: int
    synthesized_operations: int

    def __post_init__(self):
        if isinstance(self.max_qubits, bool) or not isinstance(self.max_qubits, int) or self.max_qubits != 8:
            raise ValueError("The literal-four compiler requires an explicit eight-wire domain")
        ceilings = {"source_operations": 100_000, "routed_operations": 100_000,
                    "synthesized_operations": 2_000_000}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in1..{ceiling}")


class CompilerLimitExceeded(ValueError):
    """A predeclared cap rejected the whole compiler stage, never a partial trial."""

    def __init__(self, stage: str, cap: int, actual_operations: int,
                 *, completed_operations: int = 0, next_segment_operations: int = 0):
        self.stage, self.cap, self.actual_operations = stage, cap, actual_operations
        self.completed_operations = completed_operations
        self.next_segment_operations = next_segment_operations
        super().__init__(f"{stage} exceeds its explicit operation cap {cap}: {actual_operations}")

    def as_record(self) -> dict:
        return {"status": "uncomputed_compiler_operation_cap", "stage": self.stage,
                "cap": self.cap, "actual_operations": self.actual_operations,
                "completed_operations": self.completed_operations,
                "next_segment_operations": self.next_segment_operations,
                "partial_trial_is_validated": False}


@dataclass
class RoutedFourInstrument:
    source: QuantumCircuit
    lowered: QuantumCircuit
    routed: QuantumCircuit
    report: dict[str, Any]


@dataclass
class SynthesizedFourInstrument:
    source: QuantumCircuit
    routed: QuantumCircuit
    circuit: QuantumCircuit
    report: dict[str, Any]


def _limits(value) -> FourCompilerLimits:
    if type(value) is not FourCompilerLimits:
        raise ValueError("limits must be an explicit FourCompilerLimits instance")
    # Revalidate even a dataclass mutated through object.__setattr__.
    return FourCompilerLimits(value.max_qubits, value.source_operations,
                              value.routed_operations, value.synthesized_operations)


def derivation_record() -> dict:
    source_hash = hashlib.sha256(Path(ir.__file__).read_bytes()).hexdigest()
    if source_hash != FROZEN_IR_SOURCE_SHA256:
        raise ValueError("The frozen shared instrument source differs from the predeclared derivation")
    return {"frozen_ir_source_sha256": source_hash,
            "frozen_ir_synthesis_function_sha256": hashlib.sha256(
                inspect.getsource(ir.synthesize_instrument).encode()).hexdigest(),
            "scope": "full-unitary routing and numerical rotation policy copied into a separate explicit materialization domain; shared defaults and physical formulas unchanged",
            "shared_IR_limits_modified": False,
            "shared_IR_report_limits_forged": False}


def _events(circuit: QuantumCircuit) -> list[tuple[str, int | None]]:
    return [(item.operation.name,
             circuit.find_bit(item.clbits[0]).index if item.clbits else None)
            for item in circuit.data if item.operation.name in {"reset", "measure"}]


def _readout_wires(circuit: QuantumCircuit) -> list[int]:
    return [circuit.find_bit(item.qubits[0]).index for item in circuit.data
            if item.operation.name == "measure"]


def _complete_readout(circuit: QuantumCircuit) -> None:
    if circuit.num_qubits != 8 or circuit.num_clbits != 8:
        raise ValueError("The literal-four compiler requires exactly eight quantum and classical wires")
    bits = [circuit.find_bit(item.clbits[0]).index for item in circuit.data
            if item.operation.name == "measure"]
    wires = _readout_wires(circuit)
    if bits != list(range(8)) or len(wires) != 8 or set(wires) != set(range(8)):
        raise ValueError("Complete readout must measure every physical wire once in c0..c7 order")
    output_started = False
    for item in circuit.data:
        if item.operation.name == "measure":
            output_started = True
        elif output_started and item.operation.name != "barrier":
            raise ValueError("The fixed compiler domain permits no quantum work after final readout begins")


def _matrix_checks(operation, checked: dict) -> None:
    """Bounded leaf checks complement the density module's recursive cost guard."""
    identity = id(operation)
    if identity in checked:
        return
    checked[identity] = operation  # Keep references alive, avoiding object-id reuse.
    direct = di._is_direct_matrix(operation)
    definition = None if direct else operation.definition
    if definition is not None:
        if definition.num_parameters:
            raise ValueError("Every nested unitary parameter must be bound")
        if not math.isfinite(float(definition.global_phase)):
            raise ValueError("Nested unitary phases must be finite")
        for nested in definition.data:
            _matrix_checks(nested.operation, checked)
        return
    try:
        matrix = np.asarray(operation.to_matrix(), dtype=complex)
    except Exception as error:
        raise ValueError("Unitary leaves must supply a bound finite matrix") from error
    dimension = 2**operation.num_qubits
    if not 1 <= operation.num_qubits <= 8 or matrix.shape != (dimension, dimension):
        raise ValueError("Unitary leaf matrix has inconsistent bounded arity")
    if not np.all(np.isfinite(matrix)) or not np.allclose(
            matrix.conj().T @ matrix, np.eye(dimension), atol=di.NUMERICAL_MATRIX_GUARD, rtol=0):
        raise ValueError("Unitary leaf matrix must be finite and numerically unitary")


def _source_guard(circuit: QuantumCircuit, operation_limit: int) -> int:
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("A bound QuantumCircuit is required")
    if len(circuit.data) > operation_limit:
        raise CompilerLimitExceeded("source_or_compiled_schedule", operation_limit, len(circuit.data))
    bounds = di.EightWireDensityLimits(8, min(operation_limit, 100_000), operation_limit)
    di._source_check(circuit, bounds, operation_limit)
    _complete_readout(circuit)
    if not math.isfinite(float(circuit.global_phase)):
        raise ValueError("The complete source phase must be finite")
    cost = di._unitary_cost_checker(bounds)
    expanded, checked = 0, {}
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        amount = 1
        if name not in {"reset", "measure"}:
            amount, _ = cost(item.operation)
            _matrix_checks(item.operation, checked)
        expanded += amount
        if expanded > operation_limit:
            raise CompilerLimitExceeded("expanded_unitary_source", operation_limit, expanded)
    return expanded


def _neighbor_check(circuit: QuantumCircuit) -> None:
    for item in circuit.data:
        if item.operation.name == "cx":
            a, b = (circuit.find_bit(bit).index for bit in item.qubits)
            if abs(a-b) != 1:
                raise ValueError("A non-neighbor CNOT remains after the fixed line routing")


def _primitive_check(operation, *, rotations: bool) -> None:
    """Names alone must never hide definitions in flat primitive accounting."""
    name = operation.name
    allowed = set(ROTATION_BASIS) if rotations else CLIFFORD_T
    if (name not in allowed or operation.base_class is not _PRIMITIVE_BASES.get(name)
            or not di._is_direct_matrix(operation) or operation.num_clbits
            or getattr(operation, "condition", None) is not None):
        raise ValueError("Compiler output must use canonical flat primitive gates")
    if operation.num_qubits != (2 if name == "cx" else 1):
        raise ValueError("Primitive gate has inconsistent quantum arity")
    if name == "cx" and getattr(operation, "ctrl_state", 1) != 1:
        raise ValueError("Open-control CNOTs must be explicitly lowered before primitive accounting")
    if name in {"rx", "ry", "rz"}:
        if len(operation.params) != 1 or not math.isfinite(float(operation.params[0])):
            raise ValueError("Primitive rotations require one finite bound angle")
    elif operation.params:
        raise ValueError("Clifford+T primitives must have no hidden parameters")


def _flat_basis_check(circuit: QuantumCircuit, *, rotations: bool) -> None:
    for item in circuit.data:
        if item.operation.name not in {"reset", "measure", "barrier"}:
            _primitive_check(item.operation, rotations=rotations)


def route_source(source: QuantumCircuit, *, limits: FourCompilerLimits) -> RoutedFourInstrument:
    """Run the actual frozen full-unitary policy in the new explicit domain."""
    limits = _limits(limits)
    derivation = derivation_record()
    expanded = _source_guard(source, limits.source_operations)
    scheduled = source.copy_empty_like()
    for item in source.data:
        event = item.operation.name in {"reset", "measure"}
        if event:
            scheduled.barrier()
        scheduled.append(item.operation, item.qubits, item.clbits)
        if event:
            scheduled.barrier()
    lowered = transpile(scheduled, basis_gates=list(ROTATION_BASIS), optimization_level=0,
                        seed_transpiler=7, qubits_initially_zero=False)
    if len(lowered.data) > limits.routed_operations:
        raise CompilerLimitExceeded("lowered_rotation_schedule", limits.routed_operations, len(lowered.data))
    _flat_basis_check(lowered, rotations=True)
    routed = transpile(lowered, basis_gates=list(ROTATION_BASIS),
                       coupling_map=CouplingMap.from_line(8, bidirectional=True),
                       initial_layout=list(range(8)), routing_method="sabre",
                       optimization_level=0, seed_transpiler=7, qubits_initially_zero=False)
    if len(routed.data) > limits.routed_operations:
        raise CompilerLimitExceeded("routed_rotation_schedule", limits.routed_operations, len(routed.data))
    _source_guard(routed, limits.routed_operations)
    _flat_basis_check(routed, rotations=True)
    _neighbor_check(routed)
    if _events(routed) != _events(source):
        raise ValueError("Routing changed the reset/readout event order")
    for name in ("reset", "measure"):
        if routed.count_ops().get(name, 0) != source.count_ops().get(name, 0):
            raise ValueError("Routing changed the explicit instrument event count")
    report = {"compiler_domain": COMPILER_DOMAIN, "materialization_limits": asdict(limits),
              "derivation": derivation, "qiskit_version": importlib.metadata.version("qiskit"),
              "preserve_full_unitary": True, "transpiler_qubits_initially_zero": False,
              "routing": "bidirectional logical line; fixed initial layout; SABRE seed 7; optimization level 0",
              "event_order": "all-wire barriers preserve source reset/readout chronology; no extra modeled serial cost",
              "source_counts": dict(source.count_ops()), "lowered_counts": dict(lowered.count_ops()),
              "routed_counts": dict(routed.count_ops()),
              "source_expanded_operations_preflight": expanded,
              "rotation_occurrences": sum(item.operation.name in {"rx", "ry", "rz"} for item in routed.data),
              "source_final_measurement_wires": _readout_wires(source),
              "routed_final_measurement_wires": _readout_wires(routed),
              "source_global_phase": float(source.global_phase),
              "routed_global_phase": float(routed.global_phase),
              "actual_clifford_t_synthesis_performed": False,
              "physical_estimation_performed": False}
    return RoutedFourInstrument(source.copy(), lowered, routed, report)


def _approximant_check(approximate: QuantumCircuit) -> None:
    if (not isinstance(approximate, QuantumCircuit) or approximate.num_qubits != 1
            or approximate.num_clbits or approximate.num_parameters):
        raise ValueError("Native synthesis must return a bound ancilla-free one-wire unitary")
    if not math.isfinite(float(approximate.global_phase)):
        raise ValueError("Native approximant phase must be finite")
    if set(approximate.count_ops()) - CLIFFORD_T:
        raise ValueError("Native synthesis returned unsupported non-Clifford+T operations")
    bounds = di.EightWireDensityLimits(8, 100_000, 2_000_000)
    cost, checked = di._unitary_cost_checker(bounds), {}
    amount = 0
    for item in approximate.data:
        if (len(item.qubits) != item.operation.num_qubits or item.clbits
                or len(set(item.qubits)) != len(item.qubits)):
            raise ValueError("Native approximant operands have inconsistent arity")
        _primitive_check(item.operation, rotations=False)
        value, _ = cost(item.operation)
        amount += value
        if amount > 2_000_000:
            raise CompilerLimitExceeded("native_approximant_expansion", 2_000_000, amount)
        _matrix_checks(item.operation, checked)


def synthesize_literal_four(source: QuantumCircuit, operator_error_allowance: float,
                            *, limits: FourCompilerLimits) -> SynthesizedFourInstrument:
    """Emit every actual CT occurrence under the unchanged fixed error request."""
    limits = _limits(limits)
    if (isinstance(operator_error_allowance, bool) or not isinstance(operator_error_allowance, (float, int))
            or not math.isfinite(operator_error_allowance)
            or operator_error_allowance != FIXED_SYNTHESIS_ALLOWANCE):
        raise ValueError("The literal-four compiler requires the fixed finite 5e-5 synthesis allowance")
    prepared = route_source(source, limits=limits)
    routed, rotations = prepared.routed, prepared.report["rotation_occurrences"]
    per_rotation = operator_error_allowance / max(1, rotations)
    if per_rotation <= 4*NUMERICAL_GATE_GUARD:
        raise ValueError("Requested error is below the guarded numerical synthesis resolution")
    output = QuantumCircuit(*routed.qregs, *routed.cregs, name="Clifford+T literal four instrument")
    output.global_phase = routed.global_phase
    approximants, errors = {}, []
    for item in routed.data:
        name = item.operation.name
        if name in {"rx", "ry", "rz"}:
            angle = float(item.operation.params[0])
            if not math.isfinite(angle):
                raise ValueError("Rotation angles must be finite")
            key = (name, angle.hex())
            if key not in approximants:
                matrix = np.asarray(item.operation.to_matrix(), dtype=complex)
                epsilon = per_rotation/4
                for _ in range(4):
                    approximate = gridsynth_unitary(matrix, epsilon)
                    _approximant_check(approximate)
                    error = float(np.linalg.norm(Operator(approximate).data-matrix, "fro"))+NUMERICAL_GATE_GUARD
                    if not math.isfinite(error):
                        raise ValueError("Native approximant produced a nonfinite guarded matrix error")
                    if error <= per_rotation:
                        break
                    epsilon /= 2
                else:
                    raise RuntimeError("Actual synthesis did not meet its guarded per-gate allowance")
                approximants[key] = approximate, error, epsilon
            approximate, error, _ = approximants[key]
            segment = len(approximate.data)
            if len(output.data)+segment > limits.synthesized_operations:
                raise CompilerLimitExceeded("clifford_t_schedule", limits.synthesized_operations,
                    len(output.data)+segment, completed_operations=len(output.data), next_segment_operations=segment)
            output.compose(approximate, qubits=list(item.qubits), inplace=True)
            errors.append(error)
        else:
            if name not in CLIFFORD_T | {"reset", "measure", "barrier"}:
                raise ValueError(f"Opaque operation remains after lowering: {name}")
            if len(output.data)+1 > limits.synthesized_operations:
                raise CompilerLimitExceeded("clifford_t_schedule", limits.synthesized_operations,
                    len(output.data)+1, completed_operations=len(output.data), next_segment_operations=1)
            output.append(item.operation, item.qubits, item.clbits)
    _source_guard(output, limits.synthesized_operations)
    _flat_basis_check(output, rotations=False)
    _neighbor_check(output)
    if _events(output) != _events(source):
        raise ValueError("Synthesis changed the reset/readout event order")
    for name in ("reset", "measure"):
        if output.count_ops().get(name, 0) != source.count_ops().get(name, 0):
            raise ValueError("Synthesis changed the instrument event count")
    bound = math.fsum(errors)
    if not math.isfinite(bound) or bound > operator_error_allowance:
        raise RuntimeError("Guarded total synthesis error exceeded its fixed allocation")
    report = {**prepared.report, "method": "actual ancilla-free Ross--Selinger gridsynth_unitary",
              "actual_clifford_t_synthesis_performed": True,
              "actual_synthesis_sequences_generated": True,
              "clifford_t_counts": dict(output.count_ops()), "unique_rotation_targets": len(approximants),
              "operator_error_allowance": operator_error_allowance,
              "guarded_numerical_operator_error_upper_estimate": bound,
              "per_matrix_frobenius_guard": NUMERICAL_GATE_GUARD,
              "formal_interval_certificate": False,
              "native_synthesis_backend": {
                  "repository": "https://github.com/qiskit-community/rsgridsynth",
                  "qiskit_262_cargo_lock_version": "0.2.2",
                  "upstream_tag_commit": "0e94e5638c42624f4ec358f5eab079942462d2bf",
                  "cargo_checksum": "133fdfbe9489a7c448e35e82c8443c9097ed6b5c22a492c3c78c19a472cacfad",
                  "license": "MIT", "seed": None,
                  "identity_scope": "Qiskit 2.6.2 Cargo.lock; installed wheel supplies no direct native-crate certificate",
                  "seed_note": "gridsynth_unitary exposes no seed argument; saved actual sequences/hashes identify realization"},
              "approximants": [{"gate": key[0], "angle_hex": key[1], "epsilon_requested": value[2],
                  "guarded_frobenius_error": value[1], "gate_counts": dict(value[0].count_ops()),
                  "global_phase": float(value[0].global_phase)} for key, value in approximants.items()],
              "compiled_global_phase": float(output.global_phase),
              "clifford_t_final_measurement_wires": _readout_wires(output),
              "qasm_export_scope": "complete measured instruments; overall phase is omitted by exporter, not a controlled-oracle specification"}
    output.metadata = {"literal_four_synthesis": report}
    return SynthesizedFourInstrument(prepared.source, routed, output, report)


def legacy_cap_check(circuit: QuantumCircuit) -> dict:
    """Check an actual complete CT trial against the unchanged frozen guard.

    This executes validation, not legacy synthesis or physical estimation.
    No limits report is rewritten to gain entry to the old estimator.
    """
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("An actual complete QuantumCircuit is required")
    _source_guard(circuit, 2_000_000)
    _flat_basis_check(circuit, rotations=False)
    if set(circuit.count_ops()) - (CLIFFORD_T | {"reset", "measure", "barrier"}):
        raise ValueError("The legacy check requires an actual Clifford+T instrument")
    record = {"legacy_operation_cap": LEGACY_OPERATION_CAP,
              "actual_operations": len(circuit.data), "legacy_synthesis_performed": False,
              "legacy_physical_estimation_performed": False,
              "validation_guard": "frozen instrument_resources._source_check max8/op1000000",
              "scope": "actual completed new-domain circuit tested against the unchanged old guard"}
    try:
        ir._source_check(circuit, operation_limit=LEGACY_OPERATION_CAP, max_qubits=8)
    except ValueError as error:
        status = "rejected_actual_clifford_t_operations" if len(circuit.data) > LEGACY_OPERATION_CAP else "rejected_frozen_validation_guard"
        return {**record, "status": status, "reason": str(error), "rejection_reproduced": True}
    return {**record, "status": "passed_frozen_operation_guard", "rejection_reproduced": False}
