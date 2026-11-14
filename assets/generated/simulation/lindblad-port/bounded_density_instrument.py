# SPDX-License-Identifier: MIT
"""Bounded fixed-schedule density instruments with correlated final outputs.

This is an opt-in numerical checker, not a quantum circuit transformation or a
new physical model.  A reset is the unobserved sum of its two Kraus terms;
discarded readouts dephase.  The entire density matrix is retained, so reset
histories never become invented measured bits or a pruned trajectory ensemble.
Only final commuting output readouts and earlier explicit acceptance are
supported.  Qiskit 2.6.2 matrices use the first supplied wire as local bit0.
Original MIT implementation; Qiskit core is Apache-2.0.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import OrderedDict
from collections.abc import Mapping
import math
from numbers import Real

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Measure, Reset
from qiskit.circuit.library import (CRYGate, CXGate, CZGate, HGate, IGate,
                                   PhaseGate, RXGate, RYGate, RZGate, SGate,
                                   SdgGate, SwapGate, TGate, TdgGate, UGate,
                                   UnitaryGate, XGate, YGate, ZGate)

MAX_DENSITY_QUBITS = 6
MAX_SOURCE_OPERATIONS = 100_000
MAX_VALIDATION_OPERATIONS = 1_000_000
MAX_DEFINITION_DEPTH = 32
MAX_MATRIX_CACHE_BYTES = 16*1024*1024
NUMERICAL_MATRIX_GUARD = 1e-10
NUMERICAL_STATE_GUARD = 1e-8
_PRIMITIVES = {CXGate, CZGate, HGate, IGate, PhaseGate, RXGate, RYGate,
               RZGate, SGate, SdgGate, SwapGate, TGate, TdgGate, UGate,
               UnitaryGate, XGate, YGate, ZGate}
_DIRECT_MATRICES = _PRIMITIVES | {CRYGate}
_INSTRUMENT_NAMES = {"reset", "measure", "initialize", "if_else", "while_loop",
                     "for_loop", "switch_case", "store"}


@dataclass(frozen=True)
class DenseInstrumentLimits:
    """The old five-wire/instruction defaults require explicit larger opt-in.

    source_operations applies to validate_density_source(), not to a fully
    compiled schedule.  simulate_density_trial() uses validation_operations for
    both top-level and conservatively expanded occurrences.  Width6 is a hard
    limit: the maximum dense state has64*64 complex128 entries (64KiB).
    """
    max_qubits: int = 5
    source_operations: int = 10_000
    validation_operations: int = 100_000
    matrix_cache_bytes: int = 1024*1024

    def __post_init__(self):
        ceilings = {"max_qubits": MAX_DENSITY_QUBITS,
                    "source_operations": MAX_SOURCE_OPERATIONS,
                    "validation_operations": MAX_VALIDATION_OPERATIONS,
                    "matrix_cache_bytes": MAX_MATRIX_CACHE_BYTES}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in1..{ceiling}")


def _limits(value):
    if not isinstance(value, DenseInstrumentLimits):
        raise ValueError("limits must be DenseInstrumentLimits")
    return value


def _canonical_event(item, name):
    expected = {"reset": Reset, "measure": Measure, "barrier": Barrier}[name]
    operation = item.operation
    if operation.base_class is not expected or not isinstance(operation, expected):
        raise ValueError("Reset, measure and barrier must be canonical explicit instructions")
    if name == "reset" and (operation.num_qubits != 1 or len(item.qubits) != 1 or operation.num_clbits or item.clbits):
        raise ValueError("Reset must have one quantum wire and no classical output")
    if name == "measure" and (operation.num_qubits != 1 or len(item.qubits) != 1 or operation.num_clbits != 1 or len(item.clbits) != 1):
        raise ValueError("Measure must have one quantum wire and one classical output")
    if name == "barrier" and (operation.num_clbits or item.clbits or operation.num_qubits != len(item.qubits)):
        raise ValueError("Barrier must have only its declared quantum wires")


def _source_check(circuit, limits, operation_limit):
    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("A bound QuantumCircuit is required")
    if not 1 <= circuit.num_qubits <= limits.max_qubits or len(circuit.data) > operation_limit:
        raise ValueError("Density instrument exceeds its explicit width/operation bound")
    if circuit.num_parameters:
        raise ValueError("Bind every parameter before density validation")
    initialized = set()
    for item in circuit.data:
        operation, name = item.operation, item.operation.name
        if operation.num_qubits != len(item.qubits) or operation.num_clbits != len(item.clbits) or len(set(item.qubits)) != len(item.qubits):
            raise ValueError("Instruction arity or quantum wire multiplicity is inconsistent")
        if name in {"reset", "measure", "barrier"}:
            _canonical_event(item, name)
        if getattr(operation, "condition", None) is not None or name in _INSTRUMENT_NAMES - {"reset", "measure"}:
            raise ValueError("Only fixed explicit reset/readout schedules are supported")
        if name == "barrier":
            continue
        wires = {circuit.find_bit(bit).index for bit in item.qubits}
        if name == "reset":
            initialized.update(wires)
        elif not wires <= initialized:
            raise ValueError("Every used wire requires an explicit initial reset before use")
    if initialized != set(range(circuit.num_qubits)):
        raise ValueError("Every reserved input wire requires an explicit initial reset")


def validate_density_source(circuit: QuantumCircuit, *, limits: DenseInstrumentLimits = DenseInstrumentLimits()) -> None:
    """Apply the independent raw-source ceiling, without running a simulator."""
    limits = _limits(limits)
    _source_check(circuit, limits, limits.source_operations)


def _unitary_cost_checker(limits):
    cache, active = {}, set()
    definition_items = 0
    def cost(operation, depth=0):
        nonlocal definition_items
        if operation.name in _INSTRUMENT_NAMES or operation.num_clbits or getattr(operation, "condition", None) is not None:
            raise ValueError("Hidden or conditional instruments are outside the explicit schedule")
        if operation.name == "barrier":
            raise ValueError("Nested barriers must remain explicit schedule instructions")
        if depth > MAX_DEFINITION_DEPTH:
            raise ValueError("Nested unitary definition depth exceeds its cap")
        base = operation.base_class
        if base in _PRIMITIVES and isinstance(operation, base):
            return 1, 0
        identity = id(operation)
        if identity in active:
            raise ValueError("Cyclic unitary definitions are outside the density domain")
        if identity in cache:
            _, amount, height = cache[identity]
            if depth+height > MAX_DEFINITION_DEPTH:
                raise ValueError("Nested unitary definition depth exceeds its cap")
            return amount, height
        active.add(identity)
        amount, height = 1, 0
        definition = operation.definition
        if definition is not None:
            if definition.num_qubits != operation.num_qubits or definition.num_clbits:
                raise ValueError("Unitary definition has inconsistent quantum/classical arity")
            definition_items += len(definition.data)
            if definition_items > limits.validation_operations:
                raise ValueError("Unitary definition graph exceeds the validation operation cap")
            for nested in definition.data:
                if (nested.operation.num_qubits != len(nested.qubits) or
                        nested.operation.num_clbits != len(nested.clbits) or
                        len(set(nested.qubits)) != len(nested.qubits)):
                    raise ValueError("Nested instruction arity or quantum wire multiplicity is inconsistent")
                try:
                    for q in nested.qubits:
                        definition.find_bit(q)
                except Exception as error:
                    raise ValueError("Nested instruction refers to a wire outside its definition") from error
                nested_amount, nested_height = cost(nested.operation, depth+1)
                amount += nested_amount
                height = max(height, nested_height+1)
                if amount > limits.validation_operations:
                    raise ValueError("Expanded unitary definition exceeds the validation operation cap")
            if base not in _DIRECT_MATRICES or not isinstance(operation, base):
                try:
                    alternate = operation.to_matrix()
                except Exception:
                    alternate = None
                if alternate is not None:
                    raise ValueError("Noncanonical defined gate also supplies a matrix; ambiguous representations are unsupported")
        elif not callable(getattr(operation, "to_matrix", None)):
            raise ValueError("Opaque operation has neither a matrix nor a bounded unitary definition")
        active.remove(identity)
        # Keep the operation alive with its identity entry; bounded shared
        # definitions must not collide after Python reuses an object id.
        cache[identity] = operation, amount, height
        return amount, height
    return cost


def _groups(wires, width):
    """Local little-endian index groups, with the remaining wires summed later."""
    rest = tuple(q for q in range(width) if q not in wires)
    local = np.zeros(2**len(wires), dtype=int)
    outside = np.zeros(2**len(rest), dtype=int)
    for k, q in enumerate(wires):
        local |= ((np.arange(len(local)) >> k) & 1) << q
    for k, q in enumerate(rest):
        outside |= ((np.arange(len(outside)) >> k) & 1) << q
    return local[:, None] | outside[None, :]


def _conjugate(rho, matrix, groups):
    rows = np.einsum("ab,bjk->ajk", matrix, rho[groups, :], optimize=False)
    result = np.empty_like(rho)
    result[groups, :] = rows
    columns = np.einsum("ab,dbk->dak", matrix.conj(), result[:, groups], optimize=False)
    result[:, groups] = columns
    return result


def simulate_density_trial(circuit: QuantumCircuit, outcome_clbits, score_values,
                           accept_clbits: dict[int, int] | None = None, *,
                           limits: DenseInstrumentLimits = DenseInstrumentLimits(),
                           batch_single_qubit: bool = True) -> dict:
    """Evaluate a bounded fixed instrument, preserving correlated output density.

    Every operation is preflighted/countable before execution.  A verifier-only
    local-matrix product batches 1q gates; it neither alters/export quantum
    circuits nor removes any physical operation.  Full density/Kraus evolution
    has no reset-history or trajectory-branch ceiling.
    """
    limits = _limits(limits)
    if not isinstance(batch_single_qubit, bool):
        raise ValueError("batch_single_qubit must be Boolean")
    _source_check(circuit, limits, limits.validation_operations)
    try:
        output_count = len(outcome_clbits)
    except TypeError as error:
        raise ValueError("Output indices must be a finite bounded sequence") from error
    if not 1 <= output_count <= circuit.num_qubits:
        raise ValueError("Output count exceeds the bounded quantum width")
    outputs = tuple(outcome_clbits)
    if accept_clbits is not None and not isinstance(accept_clbits, Mapping):
        raise ValueError("Acceptance must be an explicit finite classical-bit mapping")
    if accept_clbits is not None and len(accept_clbits) > circuit.num_clbits:
        raise ValueError("Acceptance mapping exceeds the declared classical register")
    acceptance = dict(accept_clbits or {})
    def valid_bit(bit):
        return not isinstance(bit, bool) and isinstance(bit, int) and 0 <= bit < circuit.num_clbits
    if not outputs or len(outputs) > circuit.num_qubits or len(set(outputs)) != len(outputs) or not all(valid_bit(bit) for bit in outputs) or any(not valid_bit(bit) for bit in acceptance):
        raise ValueError("Output and acceptance indices must be valid distinct classical bits")
    if set(outputs) & set(acceptance) or any(isinstance(value, bool) or value not in (0, 1) for value in acceptance.values()):
        raise ValueError("Acceptance and output bits must be distinct binary conditions")
    try:
        score_count = len(score_values)
    except TypeError as error:
        raise ValueError("Score values must be a finite bounded sequence") from error
    try:
        valid_scores = score_count == 2**len(outputs) and all(isinstance(value, Real) and math.isfinite(value) for value in score_values)
    except (OverflowError, ValueError) as error:
        raise ValueError("Score values must have finite real binary64 representations") from error
    if not valid_scores:
        raise ValueError("A finite scalar-score table of2**len(outcome_clbits) entries is required")
    scores = np.asarray(score_values, dtype=float)
    if scores.shape != (2**len(outputs),) or not np.all(np.isfinite(scores)):
        raise ValueError("A finite scalar-score table of2**len(outcome_clbits) entries is required")
    cost = _unitary_cost_checker(limits)
    mapping, measured, accepted = {}, set(), set()
    output_started, expanded_work = False, 0
    for item in circuit.data:
        name = item.operation.name
        amount = 1
        if name == "barrier":
            continue
        if name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in measured:
                raise ValueError("Measurement-bit reuse needs a richer classical history model")
            measured.add(bit)
            if bit in acceptance:
                if output_started:
                    raise ValueError("Acceptance must precede output readout")
                accepted.add(bit)
            if bit in outputs:
                if accepted != set(acceptance):
                    raise ValueError("Output precedes a required acceptance measurement")
                output_started = True
                mapping[bit] = circuit.find_bit(item.qubits[0]).index
        elif output_started:
            raise ValueError("Quantum work after output readout is outside this density checker")
        elif name != "reset":
            amount, _ = cost(item.operation)
        expanded_work += amount
        if expanded_work > limits.validation_operations:
            raise ValueError("Expanded unitary occurrences exceed the validation operation cap")
    if set(mapping) != set(outputs) or accepted != set(acceptance):
        raise ValueError("Required output/acceptance readouts are missing")
    output_wires = tuple(mapping[bit] for bit in outputs)
    if len(set(output_wires)) != len(output_wires):
        raise ValueError("Joint outputs must measure distinct quantum wires")

    width, dimension = circuit.num_qubits, 2**circuit.num_qubits
    rho = np.zeros((dimension, dimension), dtype=complex)
    rho[0, 0] = 1
    index = np.arange(dimension)
    groups = {tuple([q]): _groups((q,), width) for q in range(width)}
    matrices, pending = OrderedDict(), {}
    cache_bytes = peak_cache_bytes = cache_evictions = cache_skips = max_matrix_bytes = 0
    updates, instruments, unitary_occurrences = 0, {"reset": 0, "measure": 0}, 0
    def flush(wires=None):
        nonlocal rho, updates
        for q in tuple(pending) if wires is None else tuple(q for q in wires if q in pending):
            rho = _conjugate(rho, pending.pop(q), groups[(q,)])
            updates += 1
    def matrix_for(operation):
        nonlocal cache_bytes, peak_cache_bytes, cache_evictions, cache_skips, max_matrix_bytes
        base = operation.base_class
        standard = base in _DIRECT_MATRICES and base is not UnitaryGate and isinstance(operation, base)
        key = (base, operation.name, getattr(operation, "ctrl_state", None),
               getattr(operation, "num_ctrl_qubits", None),
               tuple(float(value).hex() for value in operation.params)) if standard else None
        if key is None or key not in matrices:
            try:
                matrix = np.asarray(operation.to_matrix(), dtype=complex)
            except Exception as error:
                raise ValueError("Unitary leaf must supply a finite matrix") from error
            local_dimension = 2**operation.num_qubits
            if matrix.shape != (local_dimension, local_dimension) or not np.all(np.isfinite(matrix)):
                raise ValueError("Unitary matrix has invalid shape or nonfinite entries")
            max_matrix_bytes = max(max_matrix_bytes, matrix.nbytes)
            if not np.allclose(matrix.conj().T@matrix, np.eye(local_dimension), atol=NUMERICAL_MATRIX_GUARD, rtol=0):
                raise ValueError("Operation matrix is not unitary within the numerical guard")
            if key is None or matrix.nbytes > limits.matrix_cache_bytes:
                cache_skips += 1
                return matrix
            while cache_bytes+matrix.nbytes > limits.matrix_cache_bytes:
                _, evicted = matrices.popitem(last=False)
                cache_bytes -= evicted.nbytes
                cache_evictions += 1
            matrices[key] = matrix
            cache_bytes += matrix.nbytes
            peak_cache_bytes = max(peak_cache_bytes, cache_bytes)
        else:
            matrices.move_to_end(key)
        return matrices[key]
    def evolve(operation, wires):
        nonlocal rho, updates, unitary_occurrences
        base = operation.base_class
        direct = base in _DIRECT_MATRICES and isinstance(operation, base)
        definition = operation.definition if not direct else None
        if definition is not None:
            for nested in definition.data:
                nested_wires = tuple(wires[definition.find_bit(q).index] for q in nested.qubits)
                evolve(nested.operation, nested_wires)
            return
        matrix = matrix_for(operation)
        unitary_occurrences += 1
        if len(wires) == 1 and batch_single_qubit:
            q = wires[0]
            pending[q] = matrix @ pending[q] if q in pending else matrix.copy()
        else:
            flush(wires if batch_single_qubit else None)
            if wires not in groups and len(wires) <= 2:
                groups[wires] = _groups(wires, width)
            index_groups = groups[wires] if wires in groups else _groups(wires, width)
            rho = _conjugate(rho, matrix, index_groups)
            updates += 1
    probability = reduced = probabilities = None
    for item in circuit.data:
        name = item.operation.name
        wires = tuple(circuit.find_bit(bit).index for bit in item.qubits)
        if name == "barrier":
            flush()
            continue
        if name not in {"reset", "measure"}:
            evolve(item.operation, wires)
            continue
        flush()
        instruments[name] += 1
        q = wires[0]
        zero = index[(index & (1 << q)) == 0]
        one = zero | (1 << q)
        if name == "reset":
            reset = np.zeros_like(rho)
            reset[np.ix_(zero, zero)] = rho[np.ix_(zero, zero)] + rho[np.ix_(one, one)]
            rho = reset
            continue
        bit = circuit.find_bit(item.clbits[0]).index
        if bit in acceptance:
            keep = one if acceptance[bit] else zero
            projected = np.zeros_like(rho)
            projected[np.ix_(keep, keep)] = rho[np.ix_(keep, keep)]
            rho = projected
        else:
            if bit in outputs and reduced is None:
                probability = float(np.trace(rho).real)
                if not math.isfinite(probability) or probability <= 0:
                    raise ValueError("Accepted density has zero or unrepresentable probability")
                if probability > 1+NUMERICAL_STATE_GUARD or (not acceptance and abs(probability-1) > NUMERICAL_STATE_GUARD):
                    raise ValueError("Density trace violates the numerical instrument probability guard")
                permutation = _groups(output_wires, width).reshape(-1)
                joint = rho[np.ix_(permutation, permutation)].reshape(2**len(outputs), 2**(width-len(outputs)), 2**len(outputs), 2**(width-len(outputs)))
                reduced = np.einsum("axbx->ab", joint)/probability
                probabilities = np.diag(reduced).real.copy()
                if not np.all(np.isfinite(reduced)) or min(probabilities) < -NUMERICAL_MATRIX_GUARD:
                    raise ValueError("Reduced output density is invalid numerically")
            rho[np.ix_(zero, one)] = 0
            rho[np.ix_(one, zero)] = 0
    if reduced is None:
        raise ValueError("Required output readout was not reached")
    return {"acceptance_probability": probability, "mean": float(scores@probabilities),
            "joint_probabilities": probabilities, "reduced_before_output": reduced,
            "outcome_clbits": list(outputs), "outcome_wires": list(output_wires),
            "score_values": scores, "score_range": [float(min(scores)), float(max(scores))],
            "output_kind": "one explicit bounded scalar score from correlated final readouts",
            "simulation": {"method": "full density matrix; reset Kraus terms summed without outcome recording",
                           "limits": asdict(limits), "density_shape": [dimension, dimension],
                           "density_state_bytes": rho.nbytes, "top_level_instructions": len(circuit.data),
                           "matrix_cache_bytes_limit": limits.matrix_cache_bytes,
                           "peak_matrix_cache_bytes": peak_cache_bytes,
                           "matrix_cache_evictions": cache_evictions, "matrix_cache_skipped_targets": cache_skips,
                           "maximum_unitary_matrix_bytes": max_matrix_bytes,
                           "numerical_state_guard": NUMERICAL_STATE_GUARD,
                           "index_group_cache_bytes": sum(group.nbytes for group in groups.values()),
                           "conservative_expanded_operations": expanded_work,
                           "unitary_leaf_occurrences": unitary_occurrences,
                           "density_unitary_updates": updates, "explicit_instrument_counts": instruments,
                           "single_qubit_matrix_batching": batch_single_qubit,
                           "batching_scope": "classical verification only; no quantum operation removed or export rewritten",
                           "reset_history_measured_outputs": False, "reset_history_classical_bits_created": 0,
                           "trajectory_branch_pruning": False,
                           "final_readout_handling": "joint law captured before first output; every later commuting readout validated and dephased",
                           "scope": "bounded nominal numerical instrument, not a physical, compiler-norm or arbitrary-width certificate"}}
