"""Source-only data shuttles for fixed, saved four-wire Clifford+T words.

This module never synthesizes rotations or invokes a physical estimator.  The
candidate changes where a contiguous one-wire word is performed, while keeping
its exact gates, phase, initialization and final readout contract.  Adjacent
data SWAPs are materialized as three CX gates and reversed before crossing a
source boundary.  Arbitrary spectator data states may be entangled; they are
not assumed to be clean transport ancillas.

The frozen instrument model supplies each T with a fresh magic state.  Moving
a word to data wire zero reduces that state's one-way transport, rather than
removing an uncharged return trip.  The counts below are exact logical/macro
inputs, not a completed native physical estimate or a certified code layout.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from typing import Any


CANDIDATE_ID = "qaoa-word-data-shuttle-v1"
NUM_QUBITS = 4
MAX_OPERATIONS_PER_CLASS = 10_000
MAX_WORKLOAD_OPERATIONS = 540_000
EXPECTED_CLASSES = 54
EXPECTED_GROUPS = 110
EXPECTED_PREPARATIONS = 82_032
LOCAL_CLIFFORD_T = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg"})
MAGIC_GATES = frozenset({"t", "tdg"})


def operation_digest(operations: list) -> str:
    """Hash the exact ordered primitive-list representation used by this module."""
    return hashlib.sha256(json.dumps(operations, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _copy_primitives(operations: list, num_qubits: int) -> list:
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits != NUM_QUBITS:
        raise ValueError("This candidate is bounded to exactly four data wires")
    if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_OPERATIONS_PER_CLASS:
        raise ValueError("Saved operations must be a nonempty list within the per-class cap")
    for operation in operations:
        if not isinstance(operation, list) or len(operation) != 4:
            raise ValueError("Every saved primitive must have four list-format fields")
        name, qubits, clbits, params = operation
        if not isinstance(name, str) or any(not isinstance(v, list) for v in (qubits, clbits, params)):
            raise ValueError("Primitive names and wire/parameter lists must retain their saved types")
        if any(isinstance(v, bool) or not isinstance(v, int) or not 0 <= v < NUM_QUBITS
               for v in qubits + clbits):
            raise ValueError("Primitive wire indices must be integers in 0..3")
        if params:
            raise ValueError("The candidate accepts saved Clifford+T primitives without parameters")
        if name in LOCAL_CLIFFORD_T | {"reset"}:
            if len(qubits) != 1 or clbits:
                raise ValueError("A local gate or reset requires one quantum wire and no classical destination")
        elif name == "cx":
            if len(qubits) != 2 or len(set(qubits)) != 2 or clbits:
                raise ValueError("CX requires two distinct quantum wires and no classical destination")
            if abs(qubits[0] - qubits[1]) != 1:
                raise ValueError("The saved CX must use the frozen adjacent data-line domain")
        elif name == "measure":
            if len(qubits) != 1 or len(clbits) != 1:
                raise ValueError("Readout requires one quantum wire and one classical destination")
        elif name == "barrier":
            if clbits or len(set(qubits)) != len(qubits):
                raise ValueError("A barrier retains distinct quantum wires and no classical destination")
        else:
            raise ValueError("Unsupported or unsynthesized operation in saved Clifford+T stream")
    return deepcopy(operations)


def _check_instrument_boundaries(operations: list) -> tuple[list, list]:
    # Frozen lowering retains barriers on both sides of reset/readout events.
    # Check chronology without interpreting those barriers as quantum work,
    # while leaving every barrier in the actual stream and word partition.
    chronology = [operation for operation in operations if operation[0] != "barrier"]
    canonical_resets = [["reset", [wire], [], []] for wire in range(NUM_QUBITS)]
    if len(chronology) < 2 * NUM_QUBITS or chronology[:NUM_QUBITS] != canonical_resets:
        raise ValueError("The saved instrument requires canonical initial resets on all four data wires")
    readouts = chronology[-NUM_QUBITS:]
    if any(item[0] != "measure" for item in readouts):
        raise ValueError("The saved instrument must end with its four source readouts")
    mapping = [[item[1][0], item[2][0]] for item in readouts]
    if ({row[0] for row in mapping} != set(range(NUM_QUBITS))
            or {row[1] for row in mapping} != set(range(NUM_QUBITS))):
        raise ValueError("Final readout wires and classical destinations must both be bijections")
    if any(item[0] in {"reset", "measure"} for item in chronology[NUM_QUBITS:-NUM_QUBITS]):
        raise ValueError("Interior reset or readout is outside the predeclared shuttle domain")
    return list(range(NUM_QUBITS)), mapping


def _event_counts(operations: list) -> dict[str, int]:
    """Match the frozen serializer's source events, including T-wire locations."""
    counts: Counter[str] = Counter()
    active: set[int] = set()
    for name, qubits, _, _ in operations:
        if name == "barrier":
            continue
        wire = qubits[0]
        if name == "reset":
            counts["initial_reset" if wire not in active else "reset"] += 1
        elif name in MAGIC_GATES:
            counts[f"{name}_q{wire}"] += 1
        else:
            counts[name] += 1
        active.update(qubits)
    return dict(sorted(counts.items()))


def _swap_operations(a: int, b: int) -> list:
    return [["cx", [a, b], [], []], ["cx", [b, a], [], []], ["cx", [a, b], [], []]]


def _magic_transport_swaps(events: dict[str, int]) -> int:
    return sum((wire + 1) * events.get(f"{name}_q{wire}", 0)
               for name in MAGIC_GATES for wire in range(NUM_QUBITS))


def build_candidate(operations: list, num_qubits: int = NUM_QUBITS) -> dict[str, Any]:
    """Materialize the single predeclared word data-shuttle candidate.

    A word is a maximal contiguous sequence of local Clifford+T gates on the
    same data wire.  Every other wire, CX, reset, readout and barrier terminates
    it.  Only words on wire ``w > 0`` containing at least three T/T-dagger gates
    are shuttled.  Their exact gate order is applied on wire zero, between the
    adjacent data SWAP path ``w -> 0`` and its inverse.

    With ``k`` magic gates, the baseline delivers ``k*(w+1)`` magic-transport
    SWAPs.  The candidate still delivers ``k`` fresh states to W0 and adds
    ``2*w`` data SWAPs, saving ``w*(k-2)`` SWAPs or ``3*w*(k-2)`` CX macros.
    No factory supply, injection, reset, correction or source gate is removed.
    """
    original = _copy_primitives(operations, num_qubits)
    reset_order, measurement_map = _check_instrument_boundaries(original)
    output: list = []
    segments: list[dict[str, Any]] = []
    index = 0
    added_cx = 0
    removed_magic_cx = 0
    data_swaps = 0
    while index < len(original):
        current = original[index]
        if current[0] not in LOCAL_CLIFFORD_T:
            output.append(deepcopy(current))
            index += 1
            continue
        wire = current[1][0]
        stop = index + 1
        while (stop < len(original) and original[stop][0] in LOCAL_CLIFFORD_T
               and original[stop][1] == [wire]):
            stop += 1
        word = original[index:stop]
        t_count = sum(item[0] == "t" for item in word)
        tdg_count = sum(item[0] == "tdg" for item in word)
        magic_count = t_count + tdg_count
        shuttled = wire > 0 and magic_count >= 3
        output_start = len(output)
        forward = [[right, right - 1] for right in range(wire, 0, -1)] if shuttled else []
        reverse = list(reversed(forward))
        rewired = [[name, [0], deepcopy(clbits), deepcopy(params)]
                   for name, _, clbits, params in word] if shuttled else deepcopy(word)
        if shuttled:
            for a, b in forward:
                output.extend(_swap_operations(a, b))
        output.extend(rewired)
        if shuttled:
            for a, b in reverse:
                output.extend(_swap_operations(a, b))
            added_cx += 6 * wire
            removed_magic_cx += 3 * wire * magic_count
            data_swaps += 2 * wire
        # This certificate is an exact structural statement.  Independent
        # operator/density validation is supplied separately before execution.
        segments.append({
            "source_start": index, "source_stop": stop,
            "candidate_start": output_start, "candidate_stop": len(output),
            "source_wire": wire, "candidate_word_wire": 0 if shuttled else wire,
            "local_operations": len(word), "t_count": t_count,
            "tdg_count": tdg_count, "magic_count": magic_count,
            "shuttled": shuttled,
            "selection_rule": "wire > 0 and T/T-dagger count >= 3",
            "initialized_data_path": list(range(wire, -1, -1)) if shuttled else [],
            "spectator_zero_assumption": False,
            "forward_swap_edges": deepcopy(forward), "reverse_swap_edges": deepcopy(reverse),
            "source_word_sha256": operation_digest(word),
            "rewired_word_sha256": operation_digest(rewired),
            "candidate_segment_sha256": operation_digest(output[output_start:]),
            "added_data_transport_swaps": 2 * wire if shuttled else 0,
            "removed_magic_transport_swaps": wire * magic_count if shuttled else 0,
            "net_removed_transport_swaps": wire * (magic_count - 2) if shuttled else 0,
            "added_data_transport_cx": 6 * wire if shuttled else 0,
            "removed_magic_transport_cx": 3 * wire * magic_count if shuttled else 0,
            "net_removed_cx": 3 * wire * (magic_count - 2) if shuttled else 0,
            "boundary_data_permutation": list(range(NUM_QUBITS)),
        })
        index = stop
    if len(output) > MAX_OPERATIONS_PER_CLASS:
        raise ValueError("Candidate materialization exceeds the frozen per-class operation cap")
    # Revalidate explicit gates and event boundaries after adding both routes.
    _copy_primitives(output, num_qubits)
    if _check_instrument_boundaries(output) != (reset_order, measurement_map):
        raise RuntimeError("Word transport changed source initialization or readout boundaries")
    baseline_events = _event_counts(original)
    candidate_events = _event_counts(output)
    baseline_magic_swaps = _magic_transport_swaps(baseline_events)
    candidate_magic_swaps = _magic_transport_swaps(candidate_events)
    if (candidate_events.get("cx", 0) - baseline_events.get("cx", 0) != added_cx
            or 3 * (baseline_magic_swaps - candidate_magic_swaps) != removed_magic_cx
            or removed_magic_cx - added_cx < 0):
        raise RuntimeError("The exact source and transport ledgers failed to reconcile")
    for gate in LOCAL_CLIFFORD_T - MAGIC_GATES | {"initial_reset", "measure"}:
        if baseline_events.get(gate, 0) != candidate_events.get(gate, 0):
            raise RuntimeError("A source event changed outside the predeclared word routing")
    for gate in MAGIC_GATES:
        if sum(baseline_events.get(f"{gate}_q{wire}", 0) for wire in range(NUM_QUBITS)) != sum(
                candidate_events.get(f"{gate}_q{wire}", 0) for wire in range(NUM_QUBITS)):
            raise RuntimeError("The candidate changed actual T/T-dagger demand")
    return {
        "candidate_id": CANDIDATE_ID,
        "status": "source_only_not_physically_executed",
        "logical_qubits": NUM_QUBITS,
        "operations": output, "segments": segments,
        "source_operations_sha256": operation_digest(original),
        "candidate_operations_sha256": operation_digest(output),
        "baseline_event_counts": baseline_events, "candidate_event_counts": candidate_events,
        "source_operations": len(original), "candidate_operations": len(output),
        "baseline_magic_transport_swaps": baseline_magic_swaps,
        "candidate_magic_transport_swaps": candidate_magic_swaps,
        "added_data_transport_swaps": data_swaps,
        "added_data_transport_cx": added_cx,
        "removed_magic_transport_cx": removed_magic_cx,
        "net_removed_transport_swaps": baseline_magic_swaps - candidate_magic_swaps - data_swaps,
        "net_removed_cx": removed_magic_cx - added_cx,
        "changed": any(segment["shuttled"] for segment in segments),
        "reset_wire_order": reset_order, "measurement_map": measurement_map,
        "source_phase_rule": "retain the exact saved global phase; SWAP conjugation adds no phase",
        "initialization_scope": "all data-path wires have explicit source initialization; arbitrary spectator states are retained",
        "physical_execution_performed": False,
    }
