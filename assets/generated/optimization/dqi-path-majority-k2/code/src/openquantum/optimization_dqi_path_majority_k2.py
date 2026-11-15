# SPDX-License-Identifier: MIT
"""Unexecuted fixed k=2 source profile for a complete degree-two DQI draw.

This module is a new isolated profile. It neither imports nor changes the
accepted k>=3 producer, its twelve-wire component verifier or their evidence.
The preparation, full phase-exact CCX word and clean majority construction
derive from the project's original MIT source
``optimization_dqi_path_majority.py`` at scientific commit
7a4b1cdaa1354014886f87d14149bd3d2d5edb63, SHA-256
ba3015dbdf35fe3ccec6a79503f2e911b735c2ea716969f690eb5fbc5e3c61bf.
The associated method reference is DOI 10.1038/s41586-025-09527-5; no paper
text, figure, author benchmark or opaque oracle is copied here.

Only the mixed-parity two-path input is admitted. Source construction includes
22 initial resets, actual counter-controlled degree-two preparation and counter
erasure, sign Z, sparse incidence CX, two clean path decoders, nine assignment
Hadamards, nine readouts and seven full-wire barriers. The prospective count is
2035 unitary primitives and 2073 total records. Counts and source materialization
are not evidence that this full program has been simulated or validated.

No state, reference, probability, sample or quantum/native backend is allocated.
Only source emission evaluates scalar ideal RY angles. All 37 RY occurrences
remain unsynthesized. There is no runtime-readiness, full-program, physical,
formal-proof, advantage or generic useful-output/repetition claim. Source limits
and cooperative deadlines do not certify caller retention, process RSS,
startup/shutdown or completion inside a prospective outer 360-second ceiling.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping


K = 2
N = 10
M = 10
N_VARIABLES = 9
Q = 22
SIGNS = (1, 1, 0, 0, 0, 1, 0, 0, 0, 0)
PATHS = ((0, 2, 3, 4, 5, 1), (0, 6, 7, 8, 9, 1))
EDGES = ((0, 2), (2, 3), (3, 4), (4, 5), (5, 1),
         (0, 6), (6, 7), (7, 8), (8, 9), (9, 1))
CLAUSE_WIRES = tuple(range(10))
ASSIGNMENT_WIRES = tuple(range(10, 19))
WORKSPACE_WIRES = (19, 20, 21)
ALL_WIRES = tuple(range(22))
COEFFICIENTS = (10, 10, 2)
NORMALIZATION_SQUARED = 1280
PRIMITIVES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z",
                        "reset", "measure", "barrier"})
UNITARIES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z"})
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
EXPECTED_COUNTS = MappingProxyType({"h": 253, "cx": 834, "t": 488,
    "tdg": 366, "x": 54, "ry": 37, "z": 3, "reset": 22,
    "measure": 9, "barrier": 7})
EXPECTED_UNITARIES = 2035
EXPECTED_RECORDS = 2073
EXPECTED_OPERATIONAL_EVENTS = 2066
STAGE_RANGES = (("reset", 0, 23), ("preparation", 23, 979),
    ("sign", 979, 983), ("syndrome", 983, 1002),
    ("decoder", 1002, 2053), ("hadamard", 2053, 2063),
    ("readout", 2063, 2073))
STAGE_UNITARIES = MappingProxyType({"reset": 0, "preparation": 955,
    "sign": 3, "syndrome": 18, "decoder": 1050, "hadamard": 9,
    "readout": 0})
MAX_SOURCE_EXPORT_BYTES = 1 << 20
MAX_API_SECONDS = 360.0
SOURCE_STATUS = "source_only_unexecuted_fixed_k2_profile"
QASM_HEADER = b'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[22] q;\nbit[9] c;\n'
_PLAN_TOKEN = object()


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " requires a finite int or float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(label + " requires a finite binary representation") from error
    if not math.isfinite(result):
        raise ValueError(label + " requires a finite binary representation")
    return result


def _wires(wires: object) -> tuple[int, ...]:
    if (type(wires) is not tuple or len(wires) > Q
            or any(type(w) is not int or not 0 <= w < Q for w in wires)
            or len(set(wires)) != len(wires)):
        raise ValueError("Wires must be distinct exact integer indices in [0,22)")
    return wires


@dataclass(frozen=True, slots=True)
class SourceLimits:
    """Fixed source caps can be tightened but cannot enlarge this profile."""

    max_qubits: int = Q
    max_operations: int = EXPECTED_RECORDS
    max_export_bytes: int = MAX_SOURCE_EXPORT_BYTES
    api_wall_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name, maximum in (("max_qubits", Q),
                              ("max_operations", EXPECTED_RECORDS),
                              ("max_export_bytes", MAX_SOURCE_EXPORT_BYTES)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(name + " must be a positive exact integer within its source cap")
        seconds = _finite(self.api_wall_seconds, "Source API seconds")
        if not 0 < seconds <= MAX_API_SECONDS:
            raise ValueError("Source API seconds must lie in (0,360]")
        object.__setattr__(self, "api_wall_seconds", seconds)


@dataclass(frozen=True, slots=True)
class Operation:
    """An explicit primitive/event with the same closed record vocabulary."""

    name: str
    qubits: tuple[int, ...]
    classical: tuple[int, ...] = ()
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in PRIMITIVES:
            raise ValueError("Unknown primitive/event")
        wires = _wires(self.qubits)
        if self.name == "barrier":
            if wires != ALL_WIRES:
                raise ValueError("The fixed profile uses only full-wire barriers")
        elif len(wires) != (2 if self.name == "cx" else 1):
            raise ValueError("Incorrect primitive arity")
        if (type(self.classical) is not tuple
                or len(self.classical) != (1 if self.name == "measure" else 0)
                or any(type(c) is not int or not 0 <= c < N_VARIABLES for c in self.classical)):
            raise ValueError("Only measurement has one exact classical index in [0,9)")
        if type(self.parameters) is not tuple or len(self.parameters) != (1 if self.name == "ry" else 0):
            raise ValueError("Only RY has one binary angle")
        values = tuple(_finite(p, "Primitive angle") for p in self.parameters)
        object.__setattr__(self, "parameters", values)

    def record(self) -> dict:
        if type(self) is not Operation:
            raise ValueError("Record serialization requires this profile's Operation")
        Operation.__post_init__(self)
        result = {"gate": self.name, "qubits": list(self.qubits)}
        if self.classical:
            result["classical"] = list(self.classical)
        if self.parameters:
            result["parameters_hex"] = [p.hex() for p in self.parameters]
        return result


@dataclass(frozen=True, slots=True, init=False)
class FixedK2Plan:
    """Fixed immutable source metadata; integrity is not execution authority."""

    k: int
    N: int
    m: int
    n: int
    source_wires: int
    paths: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, int], ...]
    signs: tuple[int, ...]
    incidence_masks: tuple[int, ...]
    per_pattern_coefficients: tuple[int, ...]
    normalization_squared: int
    rotation_ratios: tuple[tuple[int, int, int, int], ...]
    source_counts: Mapping[str, int]
    source_unitary_primitives: int
    source_materialized_operations: int
    limits: SourceLimits
    _marker: object
    _integrity_sha256: str


def fixture_input() -> dict:
    """Return the only admitted input, with fresh JSON containers."""
    return {"k": 2, "signs": list(SIGNS)}


def _fixed_input(value: object) -> None:
    if (type(value) is not dict or set(value) != {"k", "signs"}
            or type(value["k"]) is not int or value["k"] != K
            or type(value["signs"]) is not list or len(value["signs"]) != M
            or any(type(x) is not int or x not in (0, 1) for x in value["signs"])
            or tuple(value["signs"]) != SIGNS):
        raise ValueError("Only the exact fixed k=2 mixed-parity input is admitted")


def _branch_mass(remaining: int, prefix_weight: int) -> int:
    """Integer branch masses for per-pattern coefficients (10,10,2)."""
    if prefix_weight == 0:
        return 100 * (1 + remaining) + 2 * remaining * (remaining - 1)
    if prefix_weight == 1:
        return 100 + 4 * remaining
    if prefix_weight == 2:
        return 4
    return 0


def _branch_ratio(remaining: int, prefix_weight: int) -> tuple[int, int]:
    numerator = _branch_mass(remaining - 1, prefix_weight + 1)
    denominator = _branch_mass(remaining, prefix_weight)
    if not 0 <= numerator <= denominator or denominator <= 0:
        raise ValueError("Invalid fixed preparation branch mass")
    common = math.gcd(numerator, denominator)
    return numerator // common, denominator // common


def _ratios() -> tuple[tuple[int, int, int, int], ...]:
    rows = []
    for i in range(M):
        for prefix_weight in ((0,) if i == 0 else (0, 1)):
            numerator, denominator = _branch_ratio(M - i, prefix_weight)
            rows.append((i, prefix_weight, numerator, denominator))
    return tuple(rows)


def _incidence_masks() -> tuple[int, ...]:
    result = []
    for u, v in EDGES:
        mask = 0
        for vertex in (u, v):
            if vertex:
                mask ^= 1 << (vertex - 1)
        result.append(mask)
    return tuple(result)


def _payload(plan: FixedK2Plan) -> dict:
    return {"k": plan.k, "N": plan.N, "m": plan.m, "n": plan.n,
        "source_wires": plan.source_wires, "paths": plan.paths,
        "edges": plan.edges, "signs": plan.signs,
        "incidence_masks": plan.incidence_masks,
        "per_pattern_coefficients": plan.per_pattern_coefficients,
        "normalization_squared": plan.normalization_squared,
        "rotation_ratios": plan.rotation_ratios,
        "source_counts": dict(plan.source_counts),
        "source_unitary_primitives": plan.source_unitary_primitives,
        "source_materialized_operations": plan.source_materialized_operations,
        "limits": {name: getattr(plan.limits, name) for name in
            ("max_qubits", "max_operations", "max_export_bytes", "api_wall_seconds")}}


def _digest(plan: FixedK2Plan) -> str:
    raw = json.dumps(_payload(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _limits(value: SourceLimits | None) -> SourceLimits:
    if value is None:
        value = SourceLimits()
    if type(value) is not SourceLimits:
        raise ValueError("Limits require this fixed profile's SourceLimits")
    SourceLimits.__post_init__(value)
    if value.max_qubits < Q or value.max_operations < EXPECTED_RECORDS:
        raise ValueError("The fixed complete source does not fit these tightened limits")
    return value


def make_plan(input_value: object = None, *, limits: SourceLimits | None = None) -> FixedK2Plan:
    """Prepare integer source metadata only; no RY/state/reference evaluation."""
    _fixed_input(fixture_input() if input_value is None else input_value)
    bounds = _limits(limits)
    plan = object.__new__(FixedK2Plan)
    values = {"k": K, "N": N, "m": M, "n": N_VARIABLES, "source_wires": Q,
        "paths": PATHS, "edges": EDGES, "signs": SIGNS,
        "incidence_masks": _incidence_masks(), "per_pattern_coefficients": COEFFICIENTS,
        "normalization_squared": NORMALIZATION_SQUARED, "rotation_ratios": _ratios(),
        "source_counts": EXPECTED_COUNTS, "source_unitary_primitives": EXPECTED_UNITARIES,
        "source_materialized_operations": EXPECTED_RECORDS, "limits": bounds,
        "_marker": _PLAN_TOKEN}
    for name, value in values.items():
        object.__setattr__(plan, name, value)
    object.__setattr__(plan, "_integrity_sha256", _digest(plan))
    return plan


def _require_plan(plan: object) -> FixedK2Plan:
    if type(plan) is not FixedK2Plan or getattr(plan, "_marker", None) is not _PLAN_TOKEN:
        raise ValueError("A fixed plan from make_plan is required")
    _limits(plan.limits)
    if (any(type(getattr(plan, name)) is not int or getattr(plan, name) != expected for name, expected in
            (("k", K), ("N", N), ("m", M), ("n", N_VARIABLES), ("source_wires", Q),
             ("normalization_squared", NORMALIZATION_SQUARED),
             ("source_unitary_primitives", EXPECTED_UNITARIES),
             ("source_materialized_operations", EXPECTED_RECORDS)))
            or plan.paths != PATHS or plan.edges != EDGES or plan.signs != SIGNS
            or plan.incidence_masks != _incidence_masks()
            or plan.per_pattern_coefficients != COEFFICIENTS or plan.rotation_ratios != _ratios()
            or type(plan.source_counts) is not MappingProxyType
            or dict(plan.source_counts) != dict(EXPECTED_COUNTS)
            or any(type(x) is not int for x in plan.source_counts.values())
            or type(plan._integrity_sha256) is not str or plan._integrity_sha256 != _digest(plan)):
        raise ValueError("The fixed source plan was modified or is malformed")
    return plan


def _deadline(limits: SourceLimits, deadline: object = None) -> float:
    now = time.monotonic()
    stop = now + limits.api_wall_seconds
    if deadline is not None:
        stop = min(stop, _finite(deadline, "Absolute source deadline"))
    if time.monotonic() >= stop:
        raise TimeoutError("The source deadline has elapsed")
    return stop


def _check(stop: float) -> None:
    if time.monotonic() >= stop:
        raise TimeoutError("The source deadline has elapsed")


def plan_summary(plan: FixedK2Plan) -> dict:
    """Expose source predictions and exclusions, never an execution receipt."""
    plan = _require_plan(plan)
    result = _payload(plan)
    result.update({"schema": "openquantum.dqi.path_majority_k2_source_plan.v1",
        "status": SOURCE_STATUS, "input": fixture_input(),
        "assignment_wires": list(ASSIGNMENT_WIRES), "clause_wires": list(CLAUSE_WIRES),
        "workspace_wires": list(WORKSPACE_WIRES), "gauge_fixed_vertex": 0,
        "gauge_bit_vertex_offset": 1, "least_significant_wire": 0,
        "readout_map": [{"wire": wire, "classical": j} for j, wire in enumerate(ASSIGNMENT_WIRES)],
        "global_weight_promise": 2, "promised_error_words": 56,
        "cycle_dimension": 1, "code_distance": 10,
        "source_operational_events": EXPECTED_OPERATIONAL_EVENTS,
        "stage_ranges_including_final_barriers": [{"stage": stage, "start": start,
            "stop": stop, "barrier_index": stop - 1} for stage, start, stop in STAGE_RANGES],
        "stage_unitary_counts": dict(STAGE_UNITARIES),
        "preparation_ccx_blocks": 54, "decoder_ccx_blocks": 68,
        "phase_exact_ccx_blocks": 122, "primitive_ry_occurrences": 37,
        "source_materialized": False, "source_generated_in_this_record": False,
        "source_validation_executed": False, "full_program_validated": False,
        "statevector_evolutions": 0, "preparation_validated": False,
        "optimization_output_validated": False, "sampling_performed": False,
        "rotation_synthesis_validated": False, "physical_resources_estimated": False,
        "formal_proof_reproduced": False, "runtime_readiness_confirmed": False,
        "new_execution_authorized": False,
        "prospective_full_job_limits": {"primitive_applications": 2035,
            "buffer_bytes": 256 << 20, "sampled_rss_bytes": 1 << 30,
            "cpu_seconds": 300, "wall_seconds": 360,
            "reduction_scan_counts": "record actual scans separately from five checkpoint stages"}})
    return result


def _ccx(a: int, b: int, target: int) -> tuple[Operation, ...]:
    _wires((a, b, target))
    return (Operation("h", (target,)), Operation("cx", (b, target)),
        Operation("tdg", (target,)), Operation("cx", (a, target)),
        Operation("t", (target,)), Operation("cx", (b, target)),
        Operation("tdg", (target,)), Operation("cx", (a, target)),
        Operation("t", (b,)), Operation("t", (target,)),
        Operation("h", (target,)), Operation("cx", (a, b)),
        Operation("t", (a,)), Operation("tdg", (b,)), Operation("cx", (a, b)))


def _controlled_ry(control: int, target: int, angle: float) -> tuple[Operation, ...]:
    _wires((control, target))
    value = _finite(angle, "Controlled RY angle")
    half = value / 2
    if 2 * half != value:
        raise ValueError("Controlled RY loses information on binary halving")
    return (Operation("ry", (target,), parameters=(half,)), Operation("cx", (control, target)),
        Operation("ry", (target,), parameters=(-half,)), Operation("cx", (control, target)))


def _angle(numerator: int, denominator: int) -> float:
    """Unsynthesized scalar angle evaluation occurs only during source emission."""
    value = 2 * math.asin(math.sqrt(numerator / denominator))
    if not math.isfinite(value) or 2 * (value / 2) != value:
        raise ValueError("RY angle must be finite and survive binary halving")
    return value


def _counter_rotation(target: int, prefix_weight: int, angle: float) -> Iterator[Operation]:
    c0, c1, flag = WORKSPACE_WIRES
    negatives = (c0, c1) if prefix_weight == 0 else (c1,)
    for wire in negatives:
        yield Operation("x", (wire,))
    yield from _ccx(c0, c1, flag)
    yield from _controlled_ry(flag, target, angle)
    yield from _ccx(c0, c1, flag)
    for wire in reversed(negatives):
        yield Operation("x", (wire,))


def _counter_increment(clause: int, *, inverse: bool = False) -> Iterator[Operation]:
    c0, c1, _ = WORKSPACE_WIRES
    if inverse:
        yield Operation("cx", (clause, c0))
        yield from _ccx(clause, c0, c1)
    else:
        yield from _ccx(clause, c0, c1)
        yield Operation("cx", (clause, c0))


def _majority_words(syndrome: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    first, second, flag = WORKSPACE_WIRES
    words = []
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        words.extend(((syndrome[a], syndrome[b], first), (first, syndrome[c], flag),
                      (syndrome[a], syndrome[b], first)))
    words.extend(((syndrome[0], syndrome[1], first), (first, syndrome[2], second),
        (second, syndrome[3], flag), (first, syndrome[2], second),
        (syndrome[0], syndrome[1], first)))
    return tuple(words)


def _decoder(path: int) -> Iterator[Operation]:
    syndrome = tuple(11 + 4 * path + j for j in range(4))
    targets = tuple(5 * path + j for j in range(5))
    prefix = tuple((syndrome[j], syndrome[j + 1]) for j in range(3))
    for control, target in prefix:
        yield Operation("cx", (control, target))
    words = _majority_words(syndrome)
    for a, b, target in words:
        yield from _ccx(a, b, target)
    flag = WORKSPACE_WIRES[2]
    for target in targets:
        yield Operation("cx", (flag, target))
    for j, control in enumerate(syndrome):
        yield Operation("cx", (control, targets[j + 1]))
    for a, b, target in reversed(words):
        yield from _ccx(a, b, target)
    for control, target in reversed(prefix):
        yield Operation("cx", (control, target))


def _body(angles: dict[tuple[int, int], float]) -> Iterator[Operation]:
    for wire in ALL_WIRES:
        yield Operation("reset", (wire,))
    yield Operation("barrier", ALL_WIRES)
    for i in range(M):
        if i == 0:
            yield Operation("ry", (i,), parameters=(angles[(i, 0)],))
        else:
            yield from _counter_rotation(i, 0, angles[(i, 0)])
            yield from _counter_rotation(i, 1, angles[(i, 1)])
        if i < M - 1:
            yield from _counter_increment(i)
    for i in range(M - 2, -1, -1):
        yield from _counter_increment(i, inverse=True)
    yield Operation("barrier", ALL_WIRES)
    for i, sign in enumerate(SIGNS):
        if sign:
            yield Operation("z", (i,))
    yield Operation("barrier", ALL_WIRES)
    for i, (u, v) in enumerate(EDGES):
        for vertex in sorted((u, v)):
            if vertex:
                yield Operation("cx", (i, 9 + vertex))
    yield Operation("barrier", ALL_WIRES)
    for path in range(K):
        yield from _decoder(path)
    yield Operation("barrier", ALL_WIRES)
    for wire in ASSIGNMENT_WIRES:
        yield Operation("h", (wire,))
    yield Operation("barrier", ALL_WIRES)
    for j, wire in enumerate(ASSIGNMENT_WIRES):
        yield Operation("measure", (wire,), classical=(j,))
    yield Operation("barrier", ALL_WIRES)


def iter_operations(plan: FixedK2Plan, *, deadline: object = None) -> Iterator[Operation]:
    """Emit one complete fixed source; the count is prospective until consumed."""
    plan = _require_plan(plan)
    stop = _deadline(plan.limits, deadline)
    angles = {}
    for i, prefix_weight, numerator, denominator in plan.rotation_ratios:
        _check(stop)
        angles[(i, prefix_weight)] = _angle(numerator, denominator)
    observed = {name: 0 for name in PRIMITIVES}
    total, stage = 0, 0
    stage_unitaries = {name: 0 for name in STAGES}
    for operation in _body(angles):
        _check(stop)
        if total >= plan.limits.max_operations or stage >= len(STAGE_RANGES):
            raise ValueError("Complete source exceeded its fixed record cap")
        stage_name, start, stop_index = STAGE_RANGES[stage]
        if total < start or total >= stop_index:
            raise ValueError("Source stage record boundary differs")
        if (operation.name == "barrier") != (total == stop_index - 1):
            raise ValueError("Source barrier is not at the exact stage end")
        observed[operation.name] += 1
        if operation.name in UNITARIES:
            stage_unitaries[stage_name] += 1
        total += 1
        if operation.name == "barrier":
            stage += 1
        yield operation
    _check(stop)
    if (total != EXPECTED_RECORDS or stage != len(STAGES)
            or observed != dict(EXPECTED_COUNTS) or stage_unitaries != dict(STAGE_UNITARIES)):
        raise ValueError("Complete source consumption differs from the fixed prediction")


def iter_ndjson(plan: FixedK2Plan, *, deadline: object = None) -> Iterator[bytes]:
    """Yield bounded ASCII NDJSON without file, state or reference allocation."""
    plan = _require_plan(plan)
    used = 0
    for operation in iter_operations(plan, deadline=deadline):
        raw = (json.dumps(operation.record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("ascii")
        used += len(raw)
        if used > plan.limits.max_export_bytes:
            raise ValueError("NDJSON source exceeded the fixed byte cap")
        yield raw


def _record_operation(record: object) -> Operation:
    if type(record) is not dict or type(record.get("gate")) is not str:
        raise ValueError("QASM requires an exact primitive record")
    name = record["gate"]
    expected = {"gate", "qubits"}
    if name == "ry":
        expected.add("parameters_hex")
    if name == "measure":
        expected.add("classical")
    if (set(record) != expected or type(record["qubits"]) is not list
            or len(record["qubits"]) > Q):
        raise ValueError("Primitive record fields differ from the closed schema")
    classical, parameters = (), ()
    if name == "measure":
        if type(record["classical"]) is not list or len(record["classical"]) != 1:
            raise ValueError("Measurement classical fields must be an ordinary list")
        classical = tuple(record["classical"])
    if name == "ry":
        rows = record["parameters_hex"]
        if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not str or len(rows[0]) > 64:
            raise ValueError("RY requires one bounded canonical binary-hex angle")
        try:
            angle = float.fromhex(rows[0])
        except (OverflowError, ValueError) as error:
            raise ValueError("Malformed binary-hex angle") from error
        if not math.isfinite(angle) or angle.hex() != rows[0]:
            raise ValueError("RY binary-hex angle is not canonical and finite")
        parameters = (angle,)
    return Operation(name, tuple(record["qubits"]), classical, parameters)


def qasm_record(record: object) -> bytes:
    """Serialize one stored primitive record; do not emit a second gate source."""
    operation = _record_operation(record)
    name, wires = operation.name, operation.qubits
    operands = ", ".join("q[" + str(w) + "]" for w in wires)
    if name == "measure":
        line = "c[" + str(operation.classical[0]) + "] = measure " + operands + ";\n"
    elif name == "ry":
        angle = operation.parameters[0]
        decimal = repr(angle)
        if float(decimal).hex() != angle.hex():
            raise ValueError("QASM decimal does not roundtrip the binary RY angle")
        line = "ry(" + decimal + ") " + operands + ";\n"
    elif name == "tdg":
        line = "inv @ t " + operands + ";\n"
    else:
        line = name + " " + operands + ";\n"
    return line.encode("ascii")


def iter_qasm_records(records: Iterable[dict], *, deadline: object = None,
                      max_export_bytes: int = MAX_SOURCE_EXPORT_BYTES) -> Iterator[bytes]:
    """Export a single retained full source without new angle/gate generation."""
    if type(max_export_bytes) is not int or not 0 < max_export_bytes <= MAX_SOURCE_EXPORT_BYTES:
        raise ValueError("QASM byte cap must be an exact bounded positive integer")
    stop = _deadline(SourceLimits(max_export_bytes=max_export_bytes), deadline)
    used = len(QASM_HEADER)
    if used > max_export_bytes:
        raise ValueError("QASM header exceeds the byte cap")
    yield QASM_HEADER
    counts = {name: 0 for name in PRIMITIVES}
    total = 0
    for record in records:
        _check(stop)
        if total >= EXPECTED_RECORDS:
            raise ValueError("QASM source exceeds the fixed record cap")
        operation = _record_operation(record)
        counts[operation.name] += 1
        total += 1
        raw = qasm_record(record)
        used += len(raw)
        if used > max_export_bytes:
            raise ValueError("QASM source exceeds the byte cap")
        yield raw
    _check(stop)
    if total != EXPECTED_RECORDS or counts != dict(EXPECTED_COUNTS):
        raise ValueError("QASM export did not consume the complete fixed source")


def iter_qasm(plan: FixedK2Plan, *, deadline: object = None) -> Iterator[bytes]:
    """Convenience one-source QASM emission; a driver can reuse stored records."""
    plan = _require_plan(plan)
    stop = _deadline(plan.limits, deadline)
    records = (operation.record() for operation in iter_operations(plan, deadline=stop))
    yield from iter_qasm_records(records, deadline=stop, max_export_bytes=plan.limits.max_export_bytes)
