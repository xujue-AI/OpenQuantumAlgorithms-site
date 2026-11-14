# SPDX-License-Identifier: MIT
"""Bounded degree-two DQI generator for binary graph cycle codes.

This original source implements the accepted gate recipe; no execution or
numerical validation is implied by its presence. It imports no quantum SDK,
allocates no quantum state/operator array, samples no outcome and invokes no
synthesis, estimator or remote job. The Nature DQI formulation is referenced
by DOI 10.1038/s41586-025-09527-5; no authors' benchmark code or figure is used.

Graph vertices N, edge clauses m and gauge variables n=N-1 are distinct.
Vertex zero is fixed classically. Ordered edges and signs are preserved.
Connected cyclic simple graphs must have girth at least five. Independent
length-two-wedge and weight-at-most-two syndrome checks enforce that promise.
The sparse decoder list stores an n-bit syndrome plus one or two edge indices;
it is a classical compilation list, not a free coherent lookup.

Preparation assigns per-pattern amplitudes (m,m,2) at weights zero, one and
two. A two-bit prefix counter and clean flag are uncomputed. The decoder
XORs its total extension into arbitrary clause targets with clean workspace;
dirty-scratch target behavior is not promised. Every CCX expands into the
project's complete phase-exact fifteen-primitive sequence. RY angles remain
unsynthesized binary floats; finite/halving guards are not error certificates.

Fixed degree two gives O(m**2*n) logical generation before synthesis. The
generator does not supply generic optimum probabilities, repetition bounds,
hardness, physical resources or a growing-degree decoder. C5/figure-eight
constructors are inputs, not validation evidence. The separate C5 ideal draw
planner refers only to its declared analytic reference, not to measured gates.

make_plan performs integer/graph preprocessing only, once. Its immutable plan
is reused by emission. iter_operations alone evaluates scalar rotation angles.
Wire/count/NDJSON-byte limits are separate from caller-retained memory/RSS.
API deadlines are cooperative and include consumer pauses; a CLI must supply
its own complete-invocation/retention acceptance and hard process guard.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import time
from types import MappingProxyType
from typing import Iterator, Mapping


MAX_N = 64
MAX_M = 64
MAX_QUBITS = 192
MAX_OPERATIONS = 200_000
MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_API_WALL_SECONDS = 30.0
MAX_IDEAL_DRAWS = 1024
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
        raise ValueError("Quantum wires must be distinct integer indices in 0..191")


@dataclass(frozen=True, slots=True)
class DegreeTwoLimits:
    """Hard source ceilings; callers may tighten them, not enlarge them."""

    max_n: int = MAX_N
    max_m: int = MAX_M
    max_qubits: int = MAX_QUBITS
    max_operations: int = MAX_OPERATIONS
    max_export_bytes: int = MAX_EXPORT_BYTES
    api_wall_seconds: float = 10.0

    def __post_init__(self) -> None:
        for name, hard in (("max_n", MAX_N), ("max_m", MAX_M),
                           ("max_qubits", MAX_QUBITS), ("max_operations", MAX_OPERATIONS),
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
    """One explicit primitive, with strict arity/event/parameter guards."""

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
                or any(type(c) is not int or not 0 <= c < MAX_N for c in self.classical)):
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
class DegreeTwoPlan:
    """Immutable prepared compilation plan; obtain it through make_plan only.

    Nested entries are tuples or read-only mappings. The integrity check detects
    accidental mutation; it is not a cryptographic trust boundary against code
    that can replace this module. It does not reconstruct the lookup list.
    """

    N: int
    n: int
    m: int
    edges: tuple[tuple[int, int], ...]
    signs: tuple[int, ...]
    incidence_masks: tuple[int, ...]
    lookup_entries: tuple[tuple[int, tuple[int, ...]], ...]
    shared_clean_workspace: int
    source_wires: int
    normalization_squared: int
    per_pattern_coefficients: tuple[int, int, int]
    source_counts: Mapping[str, int]
    source_unitary_primitives: int
    source_materialized_operations: int
    preprocessing_counts: Mapping[str, int]
    rotation_ratios: tuple[tuple[int, int, int, int], ...]
    limits: DegreeTwoLimits
    _marker: object
    _integrity_sha256: str


def _checked_limits(limits: DegreeTwoLimits | None) -> DegreeTwoLimits:
    if limits is None:
        return DegreeTwoLimits()
    if type(limits) is not DegreeTwoLimits:
        raise ValueError("Limits must be a DegreeTwoLimits instance")
    try:
        DegreeTwoLimits.__post_init__(limits)
    except AttributeError as error:
        raise ValueError("The limit object is incomplete") from error
    return limits


def _deadline_for(limits: DegreeTwoLimits, deadline: int | float | None) -> float:
    bounded = time.monotonic() + limits.api_wall_seconds
    return bounded if deadline is None else min(bounded, _finite_real(deadline, "Absolute deadline"))


def _check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("The cooperative degree-two source deadline elapsed")


def _mass(m: int, r: int, j: int) -> int:
    """Integer branch mass for coefficients (m,m,2); no angle evaluation."""
    if j == 0:
        return m * m * (1 + r) + 2 * r * (r - 1)
    if j == 1:
        return m * m + 4 * r
    if j == 2:
        return 4
    return 0


def _ratio(m: int, r: int, j: int) -> tuple[int, int]:
    numerator, denominator = _mass(m, r - 1, j + 1), _mass(m, r, j)
    if denominator <= 0 or not 0 < numerator <= denominator:
        raise ValueError("Invalid integer preparation branch mass")
    common = math.gcd(numerator, denominator)
    return numerator // common, denominator // common


def _plan_payload(plan: DegreeTwoPlan) -> dict:
    limits = plan.limits
    return {"N": plan.N, "n": plan.n, "m": plan.m, "edges": plan.edges,
            "signs": plan.signs, "incidence_masks": plan.incidence_masks,
            "lookup_entries": plan.lookup_entries,
            "shared_clean_workspace": plan.shared_clean_workspace,
            "source_wires": plan.source_wires,
            "normalization_squared": plan.normalization_squared,
            "per_pattern_coefficients": plan.per_pattern_coefficients,
            "source_counts": dict(plan.source_counts),
            "source_unitary_primitives": plan.source_unitary_primitives,
            "source_materialized_operations": plan.source_materialized_operations,
            "preprocessing_counts": dict(plan.preprocessing_counts),
            "rotation_ratios": plan.rotation_ratios,
            "limits": {name: getattr(limits, name) for name in
                       ("max_n", "max_m", "max_qubits", "max_operations",
                        "max_export_bytes", "api_wall_seconds")}}


def _plan_digest(plan: DegreeTwoPlan) -> str:
    raw = json.dumps(_plan_payload(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _require_plan(plan: DegreeTwoPlan) -> DegreeTwoPlan:
    if type(plan) is not DegreeTwoPlan or getattr(plan, "_marker", None) is not _PLAN_MARKER:
        raise ValueError("A prepared DegreeTwoPlan from make_plan is required")
    limits = _checked_limits(plan.limits)
    if (type(plan.N) is not int or type(plan.n) is not int or type(plan.m) is not int
            or not 5 <= plan.N <= limits.max_n + 1 or plan.n != plan.N - 1
            or not plan.N <= plan.m <= limits.max_m
            or plan.shared_clean_workspace != max(3, plan.n - 2)
            or plan.source_wires != plan.m + plan.n + plan.shared_clean_workspace
            or not 1 <= plan.source_wires <= limits.max_qubits
            or type(plan.source_unitary_primitives) is not int
            or not 1 <= plan.source_unitary_primitives <= limits.max_operations
            or plan.normalization_squared != plan.m**3 + 3 * plan.m**2 - 2 * plan.m
            or plan.per_pattern_coefficients != (plan.m, plan.m, 2)
            or type(plan.source_materialized_operations) is not int
            or not 1 <= plan.source_materialized_operations <= limits.max_operations):
        raise ValueError("Prepared plan dimensions or bounds changed")
    entries = plan.m + plan.m * (plan.m - 1) // 2
    for value, length in ((plan.edges, plan.m), (plan.signs, plan.m),
                          (plan.incidence_masks, plan.m), (plan.lookup_entries, entries),
                          (plan.rotation_ratios, 2 * plan.m - 1)):
        if type(value) is not tuple or len(value) != length:
            raise ValueError("Prepared plan tuple shape changed")
    if (type(plan.source_counts) is not MappingProxyType or set(plan.source_counts) != PRIMITIVES
            or any(type(v) is not int or not 0 <= v <= MAX_OPERATIONS for v in plan.source_counts.values())
            or type(plan.preprocessing_counts) is not MappingProxyType
            or len(plan.preprocessing_counts) != 16
            or any(type(k) is not str or type(v) is not int or not 0 <= v <= MAX_OPERATIONS
                   for k, v in plan.preprocessing_counts.items())):
        raise ValueError("Prepared plan count metadata changed")
    for edge in plan.edges:
        if (type(edge) is not tuple or len(edge) != 2
                or any(type(v) is not int or not 0 <= v < plan.N for v in edge)):
            raise ValueError("Prepared edge data changed")
    if (any(type(v) is not int or v not in (0, 1) for v in plan.signs)
            or any(type(s) is not int or not 0 < s < 1 << plan.n for s in plan.incidence_masks)):
        raise ValueError("Prepared sign/incidence data changed")
    last = 0
    for entry in plan.lookup_entries:
        if type(entry) is not tuple or len(entry) != 2:
            raise ValueError("Prepared lookup entry shape changed")
        syndrome, targets = entry
        if (type(syndrome) is not int or not last < syndrome < 1 << plan.n
                or type(targets) is not tuple or len(targets) not in (1, 2)
                or any(type(t) is not int or not 0 <= t < plan.m for t in targets)
                or (len(targets) == 2 and targets[0] >= targets[1])):
            raise ValueError("Prepared lookup entry changed")
        last = syndrome
    for item in plan.rotation_ratios:
        if (type(item) is not tuple or len(item) != 4 or any(type(v) is not int for v in item)
                or not 0 <= item[0] < plan.m or item[1] not in (0, 1)
                or not 0 < item[2] <= item[3] <= plan.normalization_squared):
            raise ValueError("Prepared integer rotation ratio changed")
    if type(plan._integrity_sha256) is not str or len(plan._integrity_sha256) != 64 or _plan_digest(plan) != plan._integrity_sha256:
        raise ValueError("Prepared plan integrity changed")
    return plan


def make_plan(N: int, edges: tuple[tuple[int, int], ...], signs: tuple[int, ...],
              limits: DegreeTwoLimits | None = None, *,
              deadline: int | float | None = None) -> DegreeTwoPlan:
    """Prepare one immutable classical compilation plan, with no RY evaluation.

    Graph checks and lookup construction are performed once. Edge/sign order
    is preserved. Sparse lookup entries are sorted by syndrome mask. Sorting
    comparisons and integer bit work are separate classical costs, not free
    coherent operations or measured runtime.
    """
    bounded = _checked_limits(limits)
    stop = _deadline_for(bounded, deadline)
    _check_deadline(stop)
    if type(N) is not int or not 5 <= N <= bounded.max_n + 1:
        raise ValueError("Require 5..max_n+1 explicitly labeled graph vertices")
    if type(edges) is not tuple or not N <= len(edges) <= bounded.max_m:
        raise ValueError("Require a bounded cyclic edge tuple with m>=N")
    m, n = len(edges), N - 1
    if (type(signs) is not tuple or len(signs) != m
            or any(type(v) is not int or v not in (0, 1) for v in signs)):
        raise ValueError("One ordered binary sign is required per edge")
    adjacency = [[] for _ in range(N)]
    degrees = [0] * N
    canonical, rows = [], []
    for edge in edges:
        _check_deadline(stop)
        if (type(edge) is not tuple or len(edge) != 2
                or any(type(v) is not int or not 0 <= v < N for v in edge)):
            raise ValueError("An edge must contain two bounded integer vertex labels")
        u, v = edge
        if u == v:
            raise ValueError("Graph loops are excluded")
        canonical.append((min(u, v), max(u, v)))
        adjacency[u].append(v)
        adjacency[v].append(u)
        degrees[u] += 1
        degrees[v] += 1
        rows.append((0 if u == 0 else 1 << (u - 1)) ^ (0 if v == 0 else 1 << (v - 1)))
    canonical.sort()
    if any(canonical[i - 1] == canonical[i] for i in range(1, m)):
        raise ValueError("Duplicate undirected edges are excluded")
    reached, stack = {0}, [0]
    while stack:
        _check_deadline(stop)
        for neighbor in adjacency[stack.pop()]:
            if neighbor not in reached:
                reached.add(neighbor)
                stack.append(neighbor)
    if len(reached) != N:
        raise ValueError("The graph must be connected to reference vertex zero")

    # Independent geometric test: triangles have adjacent wedge endpoints;
    # four-cycles have identical endpoint pairs with distinct middle vertices.
    wedges = []
    for center, neighbors in enumerate(adjacency):
        ordered = sorted(neighbors)
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                _check_deadline(stop)
                endpoints = (ordered[i], ordered[j])
                index = bisect_left(canonical, endpoints)
                if index < m and canonical[index] == endpoints:
                    raise ValueError("A triangle violates the girth-five promise")
                wedges.append((endpoints[0], endpoints[1], center))
    wedges.sort()
    for i in range(1, len(wedges)):
        if wedges[i - 1][:2] == wedges[i][:2] and wedges[i - 1][2] != wedges[i][2]:
            raise ValueError("A four-cycle violates the girth-five promise")
    _check_deadline(stop)

    # Independent code-domain check, including the vacuum. Sorting, rather
    # than a dense syndrome table, retains masks with one/two edge indices.
    candidates = [(0, ())]
    candidates.extend((syndrome, (i,)) for i, syndrome in enumerate(rows))
    for i in range(m):
        for j in range(i + 1, m):
            _check_deadline(stop)
            candidates.append((rows[i] ^ rows[j], (i, j)))
    candidates.sort()
    if any(candidates[i - 1][0] == candidates[i][0] for i in range(1, len(candidates))):
        raise ValueError("Vacuum/singleton/pair syndrome collision")
    lookup = tuple(candidates[1:])
    if candidates[0] != (0, ()):
        raise ValueError("Vacuum syndrome ordering changed")

    workspace = max(3, n - 2)
    Q = m + n + workspace
    if Q > bounded.max_qubits:
        raise ValueError("Logical source width exceeds its bound")
    pairs = m * (m - 1) // 2
    prep_ccx = 6 * (m - 1)
    decoder_ccx = m * (2 * n - 3) + pairs * (2 * n - 2)
    all_ccx = prep_ccx + decoder_ccx
    access_cx = 2 * m - degrees[0]
    pair_weight_sum = sum(d * (m - d) for d in degrees[1:])
    decoder_x = 2 * (n * (m + pairs) - access_cx - pair_weight_sum)
    counts = {"h": 2 * all_ccx + n, "cx": 6 * all_ccx + 6 * (m - 1) + access_cx,
              "t": 4 * all_ccx, "tdg": 3 * all_ccx,
              "x": 6 * (m - 1) + decoder_x, "ry": 4 * m - 3, "z": sum(signs),
              "reset": Q, "measure": n, "barrier": len(STAGES)}
    U = sum(counts[name] for name in ("h", "cx", "t", "tdg", "x", "ry", "z"))
    M = U + Q + n + len(STAGES)
    if M > bounded.max_operations:
        raise ValueError("Complete materialized source count exceeds its bound")
    ratios = [(0, 0, *_ratio(m, m, 0))]
    for i in range(1, m):
        _check_deadline(stop)
        ratios.extend((i, j, *_ratio(m, m - i, j)) for j in (0, 1))
    preprocessing = {"graph_vertices": N, "ordered_edges": m, "edge_endpoint_records": 2 * m,
                     "sign_records": m, "sorted_edge_entries": m,
                     "adjacent_edge_duplicate_checks": m - 1,
                     "connectivity_vertices_visited": N, "connectivity_adjacency_entries": 2 * m,
                     "wedge_entries": len(wedges), "triangle_membership_queries": len(wedges),
                     "adjacent_wedge_endpoint_checks": max(0, len(wedges) - 1),
                     "incidence_masks": m, "D2_masks_including_vacuum": len(candidates),
                     "adjacent_D2_collision_checks": len(candidates) - 1,
                     "lookup_entries": len(lookup), "integer_rotation_ratios": len(ratios)}
    values = {"N": N, "n": n, "m": m, "edges": edges, "signs": signs,
              "incidence_masks": tuple(rows), "lookup_entries": lookup,
              "shared_clean_workspace": workspace, "source_wires": Q,
              "normalization_squared": m**3 + 3 * m**2 - 2 * m,
              "per_pattern_coefficients": (m, m, 2),
              "source_counts": MappingProxyType(counts), "source_unitary_primitives": U,
              "source_materialized_operations": M,
              "preprocessing_counts": MappingProxyType(preprocessing),
              "rotation_ratios": tuple(ratios), "limits": bounded, "_marker": _PLAN_MARKER}
    plan = object.__new__(DegreeTwoPlan)
    for name, value in values.items():
        object.__setattr__(plan, name, value)
    object.__setattr__(plan, "_integrity_sha256", _plan_digest(plan))
    _check_deadline(stop)
    return plan


def plan_summary(prepared_plan: DegreeTwoPlan) -> dict:
    """Describe a prepared plan without rebuilding it or evaluating angles."""
    plan = _require_plan(prepared_plan)
    return {"status": SOURCE_STATUS, "degree": 2, "N": plan.N, "n": plan.n, "m": plan.m,
            "reference_vertex": 0, "edges": [list(edge) for edge in plan.edges],
            "signs": list(plan.signs), "incidence_masks": list(plan.incidence_masks),
            "source_wires": plan.source_wires, "shared_clean_workspace": plan.shared_clean_workspace,
            "normalization_squared": plan.normalization_squared,
            "per_pattern_coefficients": list(plan.per_pattern_coefficients),
            "source_counts": dict(plan.source_counts),
            "source_unitary_primitives": plan.source_unitary_primitives,
            "source_materialized_operations": plan.source_materialized_operations,
            "preprocessing_counts": dict(plan.preprocessing_counts),
            "lookup_entries": [{"syndrome_mask": s, "edge_indices": list(t)} for s, t in plan.lookup_entries],
            "prepared_support_size": 1 + len(plan.lookup_entries),
            "rotation_sine_squared": [{"position": i, "counter": j, "fraction": str(a) + "/" + str(b)}
                                     for i, j, a, b in plan.rotation_ratios],
            "plan_integrity_sha256": plan._integrity_sha256,
            "assignment_bit_order": "integer bit j is x_(j+1); fixed reference x_0=0",
            "cycle_space_dimension": plan.m - plan.n, "minimum_distance_lower_bound": 5,
            "promise_checks": {"simple_connected_cyclic": True, "triangle_four_cycle_wedges": True,
                               "independent_D2_collisions": True},
            "classical_preprocessing_scope": "explicit input/check/lookup/ratio counts; sorting comparisons, bit costs and time/RSS are not measured",
            "lookup_storage_scope": "n-bit mask plus one/two edge indices; no dense syndrome table or free coherent lookup",
            "angle_scope": "integer ratios prepared; binary RY evaluation occurs only during emission; synthesis/error certificates absent",
            "decoder_promise": "all weight<=2 errors; arbitrary clause target with initially clean scratch",
            "off_promise_extension": "zero for unmatched syndromes", "dirty_workspace_target_law": "not promised",
            "quality_scope": "no generic or figure-eight optimum-probability/repetition bound",
            "physical_resources_computed": False, "sampled_outcomes": False}


def c5_ideal_draw_plan(miss_target: Fraction, max_draws: int = MAX_IDEAL_DRAWS) -> dict:
    """Pure exact planner for the fixed C5 analytic ideal reference only.

    No gate, angle, state or hardware work is performed. The declared p=18/19
    is the independent recipe reference, not an observed circuit probability.
    Other graphs and the figure-eight do not inherit this planner's guarantee.
    """
    if (type(miss_target) is not Fraction or not 0 < miss_target < 1
            or miss_target.numerator.bit_length() > 256 or miss_target.denominator.bit_length() > 256):
        raise ValueError("Require a native Fraction in (0,1) with numerator/denominator at most 256 bits")
    if type(max_draws) is not int or not 1 <= max_draws <= MAX_IDEAL_DRAWS:
        raise ValueError("Ideal draw cap must be an integer in 1..1024")
    failure = Fraction(1)
    for draws in range(1, max_draws + 1):
        failure *= Fraction(1, 19)
        if failure <= miss_target:
            return {"draws": draws, "failure": failure, "success": 1 - failure,
                    "target": miss_target, "p_opt": Fraction(18, 19), "max_draws": max_draws,
                    "scope": "fixed_C5_analytic_ideal_reference_only", "sampled_outcomes": False,
                    "quantum_execution": False, "physical_resources_computed": False, "validated": False}
    raise ValueError("The fixed C5 ideal miss target is infeasible within the supplied draw cap")


def exact_ccx_operations(control0: int, control1: int, target: int) -> tuple[Operation, ...]:
    """Full fifteen-primitive CCX, with no relative-phase substitution."""
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
    """Two RY/two CX controlled rotation with finite, lossless halving guard."""
    _wires(control, target)
    value = _finite_real(angle, "Controlled-RY angle")
    half = value / 2
    if 2 * half != value:
        raise ValueError("Controlled-RY angle loses information on binary halving")
    return (Operation("ry", (target,), parameters=(half,)), Operation("cx", (control, target)),
            Operation("ry", (target,), parameters=(-half,)), Operation("cx", (control, target)))


def _angle_from_ratio(numerator: int, denominator: int) -> float:
    """Scalar unsynthesized RY evaluation; only emission calls this helper."""
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


def _counter_increment(clause: int, c0: int, c1: int, inverse: bool = False) -> Iterator[Operation]:
    if inverse:
        yield Operation("cx", (clause, c0))
        yield from exact_ccx_operations(clause, c0, c1)
    else:
        yield from exact_ccx_operations(clause, c0, c1)
        yield Operation("cx", (clause, c0))


def _predicate_toggle(controls: tuple[int, ...], syndrome: int,
                      targets: tuple[int, ...], workspace: tuple[int, ...]) -> Iterator[Operation]:
    negatives = tuple(controls[j] for j in range(len(controls)) if not (syndrome >> j) & 1)
    for wire in negatives:
        yield Operation("x", (wire,))
    chain = [(controls[0], controls[1], workspace[0])]
    chain.extend((workspace[j - 2], controls[j], workspace[j - 1])
                 for j in range(2, len(controls) - 1))
    for a, b, target in chain:
        yield from exact_ccx_operations(a, b, target)
    for target in targets:
        yield from exact_ccx_operations(workspace[len(controls) - 3], controls[-1], target)
    for a, b, target in reversed(chain):
        yield from exact_ccx_operations(a, b, target)
    for wire in reversed(negatives):
        yield Operation("x", (wire,))


def _body(plan: DegreeTwoPlan, angles: dict[tuple[int, int], float]) -> Iterator[Operation]:
    all_wires = tuple(range(plan.source_wires))
    assignments = tuple(range(plan.m, plan.m + plan.n))
    scratch = tuple(range(plan.m + plan.n, plan.source_wires))
    c0, c1, flag = scratch[:3]
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
    for i, syndrome in enumerate(plan.incidence_masks):
        for j, target in enumerate(assignments):
            if (syndrome >> j) & 1:
                yield Operation("cx", (i, target))
    yield Operation("barrier", all_wires)
    for syndrome, targets in plan.lookup_entries:
        yield from _predicate_toggle(assignments, syndrome, targets, scratch[:plan.n - 2])
    yield Operation("barrier", all_wires)
    for wire in assignments:
        yield Operation("h", (wire,))
    yield Operation("barrier", all_wires)
    for j, wire in enumerate(assignments):
        yield Operation("measure", (wire,), classical=(j,))
    yield Operation("barrier", all_wires)


def iter_operations(prepared_plan: DegreeTwoPlan, *,
                    deadline: int | float | None = None) -> Iterator[Operation]:
    """Emit one complete draw from a prepared plan without rebuilding lookup.

    Preflight runs on first iteration before any operation is emitted. A
    partially consumed iterator is not a complete source export. Scalar angle
    evaluation is confined to this function's invocation; no quantum arrays,
    measurements, simulation or native work are executed.
    """
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
            raise ValueError("Emitted source operations exceeded their bound")
        if any(w >= plan.source_wires for w in operation.qubits):
            raise ValueError("Emitted operation names a wire outside this plan")
        observed[operation.name] += 1
        yield operation
    _check_deadline(stop)
    if total != plan.source_materialized_operations or observed != dict(plan.source_counts):
        raise ValueError("Emitted source ledger differs from its analytic preflight")


def iter_ndjson(prepared_plan: DegreeTwoPlan, *,
                deadline: int | float | None = None) -> Iterator[bytes]:
    """Bounded primitive NDJSON serialization only, without file IO."""
    plan = _require_plan(prepared_plan)
    byte_count = 0
    for operation in iter_operations(plan, deadline=deadline):
        raw = (json.dumps(operation.record(), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")
        if byte_count + len(raw) > plan.limits.max_export_bytes:
            raise ValueError("Serialized primitive source exceeded its byte cap")
        byte_count += len(raw)
        yield raw


def c5_input() -> tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Fixed frustrated C5 input, with no generation or reference evaluation."""
    return 5, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)), (1, 1, 1, 1, 1)


def figure_eight_input() -> tuple[int, tuple[tuple[int, int], ...], tuple[int, ...]]:
    """Generation-only two-cycle input; it implies no 24-wire simulation."""
    return 9, ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
               (0, 5), (5, 6), (6, 7), (7, 8), (8, 0)), (1,) * 10
