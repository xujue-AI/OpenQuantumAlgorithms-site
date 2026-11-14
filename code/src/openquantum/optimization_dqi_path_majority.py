# SPDX-License-Identifier: MIT
"""Bounded source for degree-two DQI with a path-local majority decoder.

This original implementation is source-only and unexecuted. It extends the
project's degree-two preparation and full phase-exact CCX construction; it
copies no paper figure, benchmark implementation or survey text. The DQI
reference is DOI 10.1038/s41586-025-09527-5. The family consists of k >= 3
internally vertex-disjoint five-edge paths between root 0 and terminal 1.
It has N=2+4k vertices, m=5k clauses and n=1+4k gauge variables. Vertex 0 is
fixed classically. The cycle-code distance is ten, but recovery is promised
only for global error weight at most two.

The total local decoder ignores the terminal syndrome, leaves all syndrome
wires unchanged, XORs the same five-bit reconstruction into arbitrary clause
targets, and restores three scratch wires initially in zero. Dirty-scratch
target behavior is not promised. Its off-promise extension can be nonzero and
can disagree with the terminal syndrome. In particular, error (0,1,1,1,0)
has internal syndrome (1,0,0,1) and returns complement (1,0,0,0,1), whose
terminal-edge parity differs. Distance ten supplies no every-neighbor claim.

The local source has 17 logical CCX blocks computing its flag, 17 reversing
that computation, and 15 other CX gates. Each CCX emits the complete exact
15-primitive H/CX/T/T-dagger word: 525 predicted primitives per path. These
are source predictions until emission completes; they are neither measured
timings nor synthesized or physical resources. Full program source includes
every reset, degree-two preparation, input sign, syndrome CX, decoder,
assignment Hadamard and readout. The k=3 full program has 31 wires and stays
generation-only; the separate local component has exactly 12 wires.

No quantum SDK, quantum state/operator, sample, optimizer, synthesis tool,
native estimator, GPU or remote job is used here. make_plan uses O(m)
bounded-word preprocessing at the hard n <= 49 ceiling. Its retained dense
n-bit incidence masks occupy O(mn) bits; integrity serialization adds bit/byte
work outside that bounded-word model. Linear emitted gate counts do not
establish linear producer bit/byte complexity. Syndrome gates themselves iterate the
two sparse edge endpoints directly. Only iter_operations evaluates scalar
unsynthesized RY angles. Their finite/lossless-halving checks are not error
certificates. API deadlines are cooperative, include consumer pauses and do
not replace an enclosing complete-process CLI guard. Source-byte limits do
not bound caller retention or process RSS. No generic success/repetition,
hardness, advantage, full-program validation or formal proof is claimed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Iterator, Mapping


MAX_K = 12
MAX_QUBITS = 112
MAX_CLASSICAL_BITS = 49
MAX_OPERATIONS = 20_000
MAX_EXPORT_BYTES = 8 * 1024 * 1024
MAX_API_WALL_SECONDS = 30.0
SOURCE_STATUS = "source_only_no_execution_evidence"
PRIMITIVES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z",
                        "reset", "measure", "barrier"})
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
_PLAN_MARKER = object()


def _finite_real(value: int | float, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(label + " must be a finite int or float")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(label + " must have a finite binary representation") from error
    if not math.isfinite(result):
        raise ValueError(label + " must have a finite binary representation")
    return result


def _wires(*wires: int) -> None:
    if (any(type(w) is not int or not 0 <= w < MAX_QUBITS for w in wires)
            or len(set(wires)) != len(wires)):
        raise ValueError("Quantum wires must be distinct bounded integer indices")


def _bits(value: object, length: int, label: str) -> tuple[int, ...]:
    if (type(value) is not tuple or len(value) != length
            or any(type(bit) is not int or bit not in (0, 1) for bit in value)):
        raise ValueError(label + " must be an exact bounded tuple of integer bits")
    return value


@dataclass(frozen=True, slots=True)
class PathMajorityLimits:
    """Source ceilings may be tightened, never enlarged by a caller."""

    max_k: int = MAX_K
    max_qubits: int = MAX_QUBITS
    max_operations: int = MAX_OPERATIONS
    max_export_bytes: int = MAX_EXPORT_BYTES
    api_wall_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name, hard in (("max_k", MAX_K), ("max_qubits", MAX_QUBITS),
                           ("max_operations", MAX_OPERATIONS),
                           ("max_export_bytes", MAX_EXPORT_BYTES)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= hard:
                raise ValueError(name + " must be a positive integer within its hard ceiling")
        allowance = _finite_real(self.api_wall_seconds, "API wall allowance")
        if not 0 < allowance <= MAX_API_WALL_SECONDS:
            raise ValueError("API wall allowance must lie in (0,30]")
        object.__setattr__(self, "api_wall_seconds", allowance)


@dataclass(frozen=True, slots=True)
class Operation:
    """One explicit primitive with strict event, arity and scalar guards."""

    name: str
    qubits: tuple[int, ...]
    classical: tuple[int, ...] = ()
    parameters: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.name) is not str or self.name not in PRIMITIVES:
            raise ValueError("Operation must name a listed explicit primitive")
        if type(self.qubits) is not tuple or len(self.qubits) > MAX_QUBITS:
            raise ValueError("Quantum wires must be a bounded tuple")
        _wires(*self.qubits)
        if self.name == "barrier":
            if not self.qubits:
                raise ValueError("A barrier must name at least one quantum wire")
        elif len(self.qubits) != (2 if self.name == "cx" else 1):
            raise ValueError("Primitive arity is incorrect")
        if (type(self.classical) is not tuple
                or len(self.classical) != (1 if self.name == "measure" else 0)
                or any(type(c) is not int or not 0 <= c < MAX_CLASSICAL_BITS for c in self.classical)):
            raise ValueError("Only measurement names one bounded classical target")
        if type(self.parameters) is not tuple or len(self.parameters) != (1 if self.name == "ry" else 0):
            raise ValueError("Only RY has one angle parameter")
        values = tuple(_finite_real(p, "Rotation parameter") for p in self.parameters)
        object.__setattr__(self, "parameters", values)

    def record(self) -> dict:
        if type(self) is not Operation:
            raise ValueError("Record serialization requires an Operation")
        Operation.__post_init__(self)
        value = {"gate": self.name, "qubits": list(self.qubits)}
        if self.classical:
            value["classical"] = list(self.classical)
        if self.parameters:
            value["parameters_hex"] = [p.hex() for p in self.parameters]
        return value


@dataclass(frozen=True, slots=True, init=False)
class PathMajorityPlan:
    """Immutable prepared source plan, obtained only through make_plan.

    Integrity detects accidental mutation. It is not a trust boundary against
    code that can replace this module or its marker/hash implementation.
    """

    k: int
    N: int
    n: int
    m: int
    edges: tuple[tuple[int, int], ...]
    signs: tuple[int, ...]
    incidence_masks: tuple[int, ...]
    shared_clean_workspace: int
    source_wires: int
    normalization_squared: int
    per_pattern_coefficients: tuple[int, int, int]
    rotation_ratios: tuple[tuple[int, int, int, int], ...]
    source_counts: Mapping[str, int]
    source_unitary_primitives: int
    source_materialized_operations: int
    preprocessing_counts: Mapping[str, int]
    limits: PathMajorityLimits
    _marker: object
    _integrity_sha256: str


def _checked_limits(limits: PathMajorityLimits | None) -> PathMajorityLimits:
    if limits is None:
        return PathMajorityLimits()
    if type(limits) is not PathMajorityLimits:
        raise ValueError("Limits must be a PathMajorityLimits instance")
    try:
        PathMajorityLimits.__post_init__(limits)
    except AttributeError as error:
        raise ValueError("The limit object is incomplete") from error
    return limits


def _checked_k(k: object, limits: PathMajorityLimits) -> int:
    if type(k) is not int or not 3 <= k <= limits.max_k:
        raise ValueError("The family requires integer k in 3..max_k")
    return k


def _deadline_for(limits: PathMajorityLimits, deadline: int | float | None) -> float:
    bounded = time.monotonic() + limits.api_wall_seconds
    return bounded if deadline is None else min(bounded, _finite_real(deadline, "Absolute deadline"))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("The cooperative path-majority source deadline elapsed")


def family_edges(k: int) -> tuple[tuple[int, int], ...]:
    """Canonical bounded graph only; root 0, terminal 1, four internals/path."""
    value = _checked_k(k, PathMajorityLimits())
    result = []
    for path in range(value):
        first = 2 + 4 * path
        result.extend(((0, first), (first, first + 1),
                       (first + 1, first + 2), (first + 2, first + 3),
                       (first + 3, 1)))
    return tuple(result)


def _mass(m: int, remaining: int, prefix_weight: int) -> int:
    """Exact branch mass for coefficients (m,m,2), no angle evaluation."""
    if prefix_weight == 0:
        return m * m * (1 + remaining) + 2 * remaining * (remaining - 1)
    if prefix_weight == 1:
        return m * m + 4 * remaining
    if prefix_weight == 2:
        return 4
    return 0


def _ratio(m: int, remaining: int, prefix_weight: int) -> tuple[int, int]:
    numerator = _mass(m, remaining - 1, prefix_weight + 1)
    denominator = _mass(m, remaining, prefix_weight)
    if not 0 <= numerator <= denominator or denominator <= 0:
        raise ValueError("Preparation branch mass is invalid")
    divisor = math.gcd(numerator, denominator)
    return numerator // divisor, denominator // divisor


def _predicted_counts(k: int, m: int, n: int, qubits: int, signs: tuple[int, ...]) -> dict[str, int]:
    ccx_blocks = 6 * (m - 1) + 34 * k
    return {"h": 2 * ccx_blocks + n,
            "cx": 6 * ccx_blocks + 6 * (m - 1) + 24 * k,
            "t": 4 * ccx_blocks, "tdg": 3 * ccx_blocks,
            "x": 6 * (m - 1), "ry": 4 * m - 3,
            "z": sum(signs), "reset": qubits, "measure": n,
            "barrier": len(STAGES)}


def _plan_payload(plan: PathMajorityPlan) -> dict:
    return {"k": plan.k, "N": plan.N, "n": plan.n, "m": plan.m,
            "edges": plan.edges, "signs": plan.signs,
            "incidence_masks": plan.incidence_masks,
            "shared_clean_workspace": plan.shared_clean_workspace,
            "source_wires": plan.source_wires,
            "normalization_squared": plan.normalization_squared,
            "per_pattern_coefficients": plan.per_pattern_coefficients,
            "rotation_ratios": plan.rotation_ratios,
            "source_counts": dict(plan.source_counts),
            "source_unitary_primitives": plan.source_unitary_primitives,
            "source_materialized_operations": plan.source_materialized_operations,
            "preprocessing_counts": dict(plan.preprocessing_counts),
            "limits": {name: getattr(plan.limits, name)
                       for name in ("max_k", "max_qubits", "max_operations",
                                    "max_export_bytes", "api_wall_seconds")}}


def _plan_digest(plan: PathMajorityPlan) -> str:
    raw = json.dumps(_plan_payload(plan), sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _require_plan(plan: PathMajorityPlan) -> PathMajorityPlan:
    if type(plan) is not PathMajorityPlan:
        raise ValueError("Source emission requires a prepared PathMajorityPlan")
    try:
        ceiling = _checked_limits(plan.limits)
        paths = _checked_k(plan.k, ceiling)
        m, n = 5 * paths, 1 + 4 * paths
        qubits = m + n + 3
        if (any(type(getattr(plan, name)) is not int for name in
                ("N", "n", "m", "source_wires", "shared_clean_workspace",
                 "normalization_squared", "source_unitary_primitives",
                 "source_materialized_operations"))
                or (plan.N, plan.n, plan.m, plan.source_wires,
                    plan.shared_clean_workspace) != (n + 1, n, m, qubits, 3)
                or qubits > ceiling.max_qubits):
            raise ValueError("The prepared plan coordinates are invalid")
        _bits(plan.signs, m, "Prepared signs")
        if (type(plan.edges) is not tuple or len(plan.edges) != m
                or any(type(edge) is not tuple or len(edge) != 2
                       or any(type(vertex) is not int or not 0 <= vertex < plan.N
                              for vertex in edge) for edge in plan.edges)
                or plan.edges != family_edges(paths)):
            raise ValueError("The prepared graph is not the ordered path family")
        if (type(plan.incidence_masks) is not tuple or len(plan.incidence_masks) != m
                or any(type(mask) is not int or not 0 < mask < (1 << n)
                       for mask in plan.incidence_masks)):
            raise ValueError("The prepared incidence masks are invalid")
        if (type(plan.rotation_ratios) is not tuple or len(plan.rotation_ratios) != 2 * m - 1
                or any(type(row) is not tuple or len(row) != 4
                       or any(type(value) is not int for value in row)
                       or not 0 <= row[0] < m or row[1] not in (0, 1)
                       or not 0 <= row[2] <= row[3] <= 1_000_000 or row[3] == 0
                       for row in plan.rotation_ratios)):
            raise ValueError("The prepared rotation ratios are invalid")
        if (type(plan.source_counts) is not MappingProxyType
                or len(plan.source_counts) != len(PRIMITIVES)
                or frozenset(plan.source_counts) != PRIMITIVES
                or any(type(value) is not int or not 0 <= value <= MAX_OPERATIONS
                       for value in plan.source_counts.values())
                or dict(plan.source_counts) != _predicted_counts(paths, m, n, qubits, plan.signs)):
            raise ValueError("The prepared source counts are invalid")
        if (type(plan.preprocessing_counts) is not MappingProxyType
                or len(plan.preprocessing_counts) != 4
                or dict(plan.preprocessing_counts) != {"edges": m, "incidence_masks": m,
                    "rotation_ratios": 2 * m - 1, "decoder_lookup_entries": 0}
                or any(type(value) is not int for value in plan.preprocessing_counts.values())):
            raise ValueError("The prepared preprocessing counts are invalid")
        materialized = sum(plan.source_counts.values())
        if (materialized > ceiling.max_operations
                or plan.source_materialized_operations != materialized
                or plan.source_unitary_primitives != materialized - qubits - n - len(STAGES)
                or plan.normalization_squared != m ** 3 + 3 * m ** 2 - 2 * m
                or type(plan.per_pattern_coefficients) is not tuple
                or len(plan.per_pattern_coefficients) != 3
                or any(type(value) is not int for value in plan.per_pattern_coefficients)
                or plan.per_pattern_coefficients != (m, m, 2)):
            raise ValueError("The prepared source accounting is invalid")
        if type(plan._integrity_sha256) is not str or len(plan._integrity_sha256) != 64:
            raise ValueError("The prepared plan integrity identity is invalid")
        if plan._marker is not _PLAN_MARKER or _plan_digest(plan) != plan._integrity_sha256:
            raise ValueError("The prepared plan integrity check failed")
    except (AttributeError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError("The prepared plan is incomplete or malformed") from error
    return plan


def make_plan(k: int, signs: tuple[int, ...], *, limits: PathMajorityLimits | None = None,
              deadline: int | float | None = None) -> PathMajorityPlan:
    """Prepare the canonical family in O(m) bounded-word graph operations.

    No singleton/pair lookup, source emission, scalar angle or quantum array
    is constructed. Retained n-bit incidence masks occupy O(mn) bits; integrity
    serialization adds bit/byte work beyond the bounded-word n <= 49 ceiling. Sign bits are
    explicit edge-equation targets, not weights.
    """
    ceiling = _checked_limits(limits)
    paths = _checked_k(k, ceiling)
    edge_count = 5 * paths
    targets = _bits(signs, edge_count, "Signs")
    stop = _deadline_for(ceiling, deadline)
    _check_deadline(stop)
    vertices = 2 + 4 * paths
    variables = vertices - 1
    qubits = edge_count + variables + 3
    if qubits > ceiling.max_qubits:
        raise ValueError("The full source wire count exceeds its bound")
    edges = family_edges(paths)
    incidence = []
    for u, v in edges:
        _check_deadline(stop)
        incidence.append((0 if u == 0 else 1 << (u - 1))
                         ^ (0 if v == 0 else 1 << (v - 1)))
    ratios = []
    for position in range(edge_count):
        _check_deadline(stop)
        for weight in ((0,) if position == 0 else (0, 1)):
            numerator, denominator = _ratio(edge_count, edge_count - position, weight)
            ratios.append((position, weight, numerator, denominator))
    counts = _predicted_counts(paths, edge_count, variables, qubits, targets)
    materialized = sum(counts.values())
    if materialized > ceiling.max_operations:
        raise ValueError("The predicted source operation count exceeds its bound")
    unitary = materialized - counts["reset"] - counts["measure"] - counts["barrier"]
    plan = object.__new__(PathMajorityPlan)
    values = {"k": paths, "N": vertices, "n": variables, "m": edge_count,
              "edges": edges, "signs": targets, "incidence_masks": tuple(incidence),
              "shared_clean_workspace": 3, "source_wires": qubits,
              "normalization_squared": edge_count ** 3 + 3 * edge_count ** 2 - 2 * edge_count,
              "per_pattern_coefficients": (edge_count, edge_count, 2),
              "rotation_ratios": tuple(ratios), "source_counts": MappingProxyType(counts),
              "source_unitary_primitives": unitary,
              "source_materialized_operations": materialized,
              "preprocessing_counts": MappingProxyType({"edges": edge_count,
                    "incidence_masks": edge_count, "rotation_ratios": 2 * edge_count - 1,
                    "decoder_lookup_entries": 0}),
              "limits": ceiling, "_marker": _PLAN_MARKER}
    for name, value in values.items():
        object.__setattr__(plan, name, value)
    object.__setattr__(plan, "_integrity_sha256", _plan_digest(plan))
    _check_deadline(stop)
    return plan


def component_summary() -> dict:
    """Prediction only; this does not emit or validate the local component."""
    return {"schema": "openquantum.dqi.path_majority_component_ledger.v1",
            "status": SOURCE_STATUS, "source_wires": 12,
            "syndrome_wires": [0, 1, 2, 3], "target_wires": [4, 5, 6, 7, 8],
            "scratch_wires": [9, 10, 11], "chain_wires": [9, 10], "flag_wire": 11,
            "source_counts": {"h": 68, "cx": 219, "t": 136, "tdg": 102},
            "source_unitary_primitives": 525, "source_materialized_operations": 525,
            "ccx_blocks_compute": 17, "ccx_blocks_uncompute": 17,
            "cx_outside_ccx": 15, "clean_scratch_required": True,
            "arbitrary_target_xor": True, "syndrome_restored": True,
            "promise": "global_error_weight_at_most_two",
            "total_extension": "local_majority_ignores_terminal",
            "source_prediction_only": True, "component_executed": False,
            "scientific_validated": False, "full_program_validated": False,
            "rotation_synthesis_available": False, "physical_resources_available": False}


def plan_summary(prepared_plan: PathMajorityPlan) -> dict:
    """JSON-friendly source predictions; no emitted or validated claim."""
    plan = _require_plan(prepared_plan)
    return {"schema": "openquantum.dqi.path_majority_plan.v1", "status": SOURCE_STATUS,
            "k": plan.k, "N": plan.N, "n": plan.n, "m": plan.m,
            "edges": [list(edge) for edge in plan.edges], "signs": list(plan.signs),
            "source_wires": plan.source_wires,
            "shared_clean_workspace": plan.shared_clean_workspace,
            "per_pattern_coefficients": list(plan.per_pattern_coefficients),
            "normalization_squared": plan.normalization_squared,
            "rotation_ratios": [list(row) for row in plan.rotation_ratios],
            "source_counts": dict(plan.source_counts),
            "source_unitary_primitives": plan.source_unitary_primitives,
            "source_materialized_operations": plan.source_materialized_operations,
            "preprocessing_counts": dict(plan.preprocessing_counts),
            "decoder": {"scope": "weight_at_most_two",
                "total_extension": "local_majority_ignores_terminal",
                "ccx_blocks_per_path": 34, "cx_outside_ccx_per_path": 15,
                "predicted_primitives_per_path": 525, "source_prediction_only": True},
            "graph": {"root": 0, "terminal": 1, "path_edges": 5,
                "girth": 10, "cycle_space_dimension": plan.k - 1,
                "every_neighbor_recovery_claimed": False},
            "scaling": {"fixed_path_length": 5, "fixed_degree": 2,
                "source_preparation": "O(m)", "source_decoder": "O(m)",
                "source_graph_preprocessing": "O(m) bounded-word operations; O(mn) mask bits",
                "source_graph_mask_storage": "O(mn) bits",
                "producer_bit_byte_work": "not identified with linear emitted gate counts",
                "operation_model": "fixed path length and degree; gate counts separate from producer bit work",
                "source_quantum_wires": "O(m+n)",
                "classical_comparator": "O(m)", "decoder_lookup_entries": 0},
            "repetitions": {"complete_draws_represented": 1,
                "generic_success_probability_available": False,
                "generic_repetition_bound_available": False},
            "exclusions": {"rotation_synthesis": True, "physical_estimation": True,
                "full_program_simulation": True, "sampling": True,
                "optimization_advantage": True, "formal_proof": True},
            "source_prediction_only": True, "scientific_validated": False,
            "full_program_validated": False, "publication_complete": False}


def exact_ccx_operations(control0: int, control1: int, target: int) -> tuple[Operation, ...]:
    """Project-original complete phase-exact CCX; never a relative-phase AND."""
    _wires(control0, control1, target)
    return (Operation("h", (target,)), Operation("cx", (control1, target)),
            Operation("tdg", (target,)), Operation("cx", (control0, target)),
            Operation("t", (target,)), Operation("cx", (control1, target)),
            Operation("tdg", (target,)), Operation("cx", (control0, target)),
            Operation("t", (control1,)), Operation("t", (target,)),
            Operation("h", (target,)), Operation("cx", (control0, control1)),
            Operation("t", (control0,)), Operation("tdg", (control1,)),
            Operation("cx", (control0, control1)))


def controlled_ry_operations(control: int, target: int, angle: int | float) -> tuple[Operation, ...]:
    _wires(control, target)
    value = _finite_real(angle, "Controlled-RY angle")
    half = value / 2
    if 2 * half != value:
        raise ValueError("Controlled-RY angle loses information on binary halving")
    return (Operation("ry", (target,), parameters=(half,)), Operation("cx", (control, target)),
            Operation("ry", (target,), parameters=(-half,)), Operation("cx", (control, target)))


def _angle_from_ratio(numerator: int, denominator: int) -> float:
    """Scalar unsynthesized binary RY evaluation, called only by emission."""
    angle = 2 * math.asin(math.sqrt(numerator / denominator))
    if not math.isfinite(angle) or 2 * (angle / 2) != angle:
        raise ValueError("An RY angle must be finite and survive lossless binary halving")
    return angle


def _counter_rotation(c0: int, c1: int, flag: int, target: int,
                      counter_value: int, angle: float) -> Iterator[Operation]:
    negatives = (c0, c1) if counter_value == 0 else (c1,)
    for wire in negatives:
        yield Operation("x", (wire,))
    yield from exact_ccx_operations(c0, c1, flag)
    yield from controlled_ry_operations(flag, target, angle)
    yield from exact_ccx_operations(c0, c1, flag)
    for wire in reversed(negatives):
        yield Operation("x", (wire,))


def _counter_increment(clause: int, c0: int, c1: int, *, inverse: bool = False) -> Iterator[Operation]:
    if inverse:
        yield Operation("cx", (clause, c0))
        yield from exact_ccx_operations(clause, c0, c1)
    else:
        yield from exact_ccx_operations(clause, c0, c1)
        yield Operation("cx", (clause, c0))


def _majority_compute_words(syndrome: tuple[int, ...], scratch: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    first, second, flag = scratch
    triples = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    result = []
    for a, b, c in triples:
        result.extend(((syndrome[a], syndrome[b], first),
                       (first, syndrome[c], flag),
                       (syndrome[a], syndrome[b], first)))
    result.extend(((syndrome[0], syndrome[1], first),
                   (first, syndrome[2], second), (second, syndrome[3], flag),
                   (first, syndrome[2], second), (syndrome[0], syndrome[1], first)))
    return tuple(result)


def _path_decoder(syndrome: tuple[int, ...], targets: tuple[int, ...],
                  scratch: tuple[int, ...]) -> Iterator[Operation]:
    """Total XOR extension; terminal is absent and target never controls work."""
    prefix = tuple((syndrome[j], syndrome[j + 1]) for j in range(3))
    for control, target in prefix:
        yield Operation("cx", (control, target))
    words = _majority_compute_words(syndrome, scratch)
    for a, b, target in words:
        yield from exact_ccx_operations(a, b, target)
    flag = scratch[2]
    for target in targets:
        yield Operation("cx", (flag, target))
    for j, control in enumerate(syndrome):
        yield Operation("cx", (control, targets[j + 1]))
    for a, b, target in reversed(words):
        yield from exact_ccx_operations(a, b, target)
    for control, target in reversed(prefix):
        yield Operation("cx", (control, target))


def iter_component_operations(*, deadline: int | float | None = None) -> Iterator[Operation]:
    """Emit only the fixed 12-wire local block; not a preparation or verifier.

    No RY, reset, measurement or barrier is emitted. Full consumption checks
    predicted counts; caller retention and complete-invocation acceptance are
    separate. Source emission itself supplies no coherent validation.
    """
    stop = _deadline_for(PathMajorityLimits(), deadline)
    _check_deadline(stop)
    counts = {"h": 0, "cx": 0, "t": 0, "tdg": 0}
    total = 0
    for operation in _path_decoder((0, 1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11)):
        _check_deadline(stop)
        total += 1
        if total > 525 or operation.name not in counts or any(w >= 12 for w in operation.qubits):
            raise ValueError("The local component exceeded its fixed source envelope")
        counts[operation.name] += 1
        yield operation
    _check_deadline(stop)
    if total != 525 or counts != {"h": 68, "cx": 219, "t": 136, "tdg": 102}:
        raise ValueError("The local component differs from its source prediction")


def _body(plan: PathMajorityPlan, angles: dict[tuple[int, int], float]) -> Iterator[Operation]:
    all_wires = tuple(range(plan.source_wires))
    assignments = tuple(range(plan.m, plan.m + plan.n))
    scratch = tuple(range(plan.m + plan.n, plan.source_wires))
    c0, c1, flag = scratch
    for wire in all_wires:
        yield Operation("reset", (wire,))
    yield Operation("barrier", all_wires)
    for i in range(plan.m):
        if i == 0:
            yield Operation("ry", (i,), parameters=(angles[(i, 0)],))
        else:
            yield from _counter_rotation(c0, c1, flag, i, 0, angles[(i, 0)])
            yield from _counter_rotation(c0, c1, flag, i, 1, angles[(i, 1)])
        if i < plan.m - 1:
            yield from _counter_increment(i, c0, c1)
    for i in range(plan.m - 2, -1, -1):
        yield from _counter_increment(i, c0, c1, inverse=True)
    yield Operation("barrier", all_wires)
    for i, sign in enumerate(plan.signs):
        if sign:
            yield Operation("z", (i,))
    yield Operation("barrier", all_wires)
    for i, edge in enumerate(plan.edges):
        for vertex in sorted(edge):
            if vertex != 0:
                yield Operation("cx", (i, assignments[vertex - 1]))
    yield Operation("barrier", all_wires)
    for path in range(plan.k):
        local_syndrome = tuple(assignments[1 + 4 * path + j] for j in range(4))
        local_targets = tuple(5 * path + j for j in range(5))
        yield from _path_decoder(local_syndrome, local_targets, scratch)
    yield Operation("barrier", all_wires)
    for wire in assignments:
        yield Operation("h", (wire,))
    yield Operation("barrier", all_wires)
    for j, wire in enumerate(assignments):
        yield Operation("measure", (wire,), classical=(j,))
    yield Operation("barrier", all_wires)


def iter_operations(prepared_plan: PathMajorityPlan, *,
                    deadline: int | float | None = None) -> Iterator[Operation]:
    """Emit one bounded full-program draw; full program remains unsimulated."""
    plan = _require_plan(prepared_plan)
    stop = _deadline_for(plan.limits, deadline)
    _check_deadline(stop)
    angles = {}
    for i, j, numerator, denominator in plan.rotation_ratios:
        _check_deadline(stop)
        angles[(i, j)] = _angle_from_ratio(numerator, denominator)
    total = 0
    observed = {name: 0 for name in PRIMITIVES}
    for operation in _body(plan, angles):
        _check_deadline(stop)
        total += 1
        if total > plan.limits.max_operations:
            raise ValueError("Emitted full-program operations exceeded their bound")
        if any(w >= plan.source_wires for w in operation.qubits):
            raise ValueError("Emitted operation names a wire outside this plan")
        observed[operation.name] += 1
        yield operation
    _check_deadline(stop)
    if total != plan.source_materialized_operations or observed != dict(plan.source_counts):
        raise ValueError("Full-program emission differs from its analytic source preflight")


def iter_ndjson(prepared_plan: PathMajorityPlan, *,
                deadline: int | float | None = None) -> Iterator[bytes]:
    """Bounded full-program serialization, with no file or quantum-state IO."""
    plan = _require_plan(prepared_plan)
    byte_count = 0
    for operation in iter_operations(plan, deadline=deadline):
        raw = (json.dumps(operation.record(), sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("ascii")
        if byte_count + len(raw) > plan.limits.max_export_bytes:
            raise ValueError("Serialized primitive source exceeded its byte cap")
        byte_count += len(raw)
        yield raw


def fixture_input() -> tuple[int, tuple[int, ...]]:
    """Mixed path-parity k=3 input; no generation or objective evaluation."""
    return 3, (1, 1, 1, 1, 0) + (1,) * 10


def decode_path_bits(syndrome: tuple[int, ...]) -> tuple[int, ...]:
    """Classical total extension for four internal syndrome bits only.

    It is a specification helper, not evidence for the emitted gate word.
    Coherent validation must independently build its reference and retain
    nonuniform signed arbitrary-target amplitudes.
    """
    bits = _bits(syndrome, 4, "Internal syndrome")
    prefix = [0]
    for bit in bits:
        prefix.append(prefix[-1] ^ bit)
    first_edge = int(sum(prefix) >= 3)
    return tuple(first_edge ^ bit for bit in prefix)


def reduced_incidence_syndrome(k: int, error_bits: tuple[int, ...]) -> int:
    """Explicit graph syndrome; sink is bit zero, internal vertices follow."""
    paths = _checked_k(k, PathMajorityLimits())
    errors = _bits(error_bits, 5 * paths, "Error word")
    syndrome = 0
    for bit, (u, v) in zip(errors, family_edges(paths)):
        if bit:
            if u != 0:
                syndrome ^= 1 << (u - 1)
            if v != 0:
                syndrome ^= 1 << (v - 1)
    return syndrome


def decode_syndrome(k: int, syndrome_bits: tuple[int, ...]) -> tuple[int, ...]:
    """Total graph extension, ignoring and preserving the terminal coordinate."""
    paths = _checked_k(k, PathMajorityLimits())
    bits = _bits(syndrome_bits, 1 + 4 * paths, "Reduced incidence syndrome")
    result = []
    for path in range(paths):
        result.extend(decode_path_bits(bits[1 + 4 * path:5 + 4 * path]))
    return tuple(result)


def classical_optimum(k: int, signs: tuple[int, ...], *,
                      edges: tuple[tuple[int, int], ...] | None = None) -> dict:
    """Independent O(m) parity/sign optimum with explicit assignment scoring.

    This comparator consumes the actual edge equations and never calls the
    quantum decoder, plan, preparation, primitive emitter or a quantum SDK.
    It tries the two terminal choices analytically. A path of mismatched sign
    parity has one necessary and achievable violation, placed at its first
    edge. Terminal zero wins a parity-count tie. It proves no advantage.
    """
    paths = _checked_k(k, PathMajorityLimits())
    targets = _bits(signs, 5 * paths, "Signs")
    canonical = family_edges(paths)
    if edges is None:
        equations = canonical
    elif (type(edges) is not tuple or len(edges) != 5 * paths
          or any(type(edge) is not tuple or len(edge) != 2
                 or any(type(vertex) is not int for vertex in edge) for edge in edges)
          or edges != canonical):
        raise ValueError("Comparator edges must be the exact ordered five-edge path family")
    else:
        equations = edges
    parities = []
    for path in range(paths):
        parity = 0
        for edge in range(5):
            parity ^= targets[5 * path + edge]
        parities.append(parity)
    odd = sum(parities)
    even = paths - odd
    terminal = 1 if odd > even else 0
    vertices = 2 + 4 * paths
    assignment = [0] * vertices
    assignment[1] = terminal
    for path in range(paths):
        current = 0
        mismatch = parities[path] != terminal
        for edge in range(4):
            u, v = equations[5 * path + edge]
            if assignment[u] != current:
                raise ValueError("Comparator path propagation coordinates are inconsistent")
            current ^= targets[5 * path + edge] ^ int(mismatch and edge == 0)
            assignment[v] = current
    score = 0
    for (u, v), target in zip(equations, targets):
        score += int((assignment[u] ^ assignment[v]) == target)
    optimum = 5 * paths - min(even, odd)
    if score != optimum:
        raise ValueError("Independently scored comparator assignment disagrees with its optimum")
    return {"schema": "openquantum.dqi.path_majority_classical_optimum.v1",
            "status": "analytic_classical_comparator_no_quantum_advantage",
            "k": paths, "N": vertices, "n": vertices - 1, "m": 5 * paths,
            "edges": equations, "signs": targets, "assignment": tuple(assignment),
            "gauge_assignment": tuple(assignment[1:]), "terminal_bit": terminal,
            "score": score, "minimum_violations": min(even, odd),
            "path_parities": tuple(parities), "parity_counts": {"even": even, "odd": odd},
            "classical_work": {"path_parity_xors": 5 * paths,
                "assignment_internal_steps": 4 * paths, "independent_edge_scores": 5 * paths,
                "full_vertex_output_writes": vertices}, "no_advantage_claim": True}
