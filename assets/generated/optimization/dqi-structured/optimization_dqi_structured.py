# SPDX-License-Identifier: MIT
"""Original degree-one DQI primitives for B=[I_n; ones], with clean workspace.

This module is a bounded, standard-library circuit generator. It imports no
quantum simulator, samples nothing and makes no execution or physical-resource
claim. The Nature DQI paper (DOI 10.1038/s41586-025-09527-5) is a formulation
reference; the authors' benchmark code was not adopted.

The prepared clause domain is the vacuum plus all n+1 singletons. Its syndrome
map is injective for n>=2. The decoder has n+1 equality predicates, rather than
a 2**n table. For initially zero workspace it XORs the specified decoder output
into arbitrary clause targets and restores workspace. The primitive sequence
is unitary on dirty workspace too, but for n>=3 its target action depends on
that initial workspace. It is separate from the old default's stronger dirty
decoder law and its 377/396 source ledger.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Iterator


MAX_N = 64
MAX_QUBITS = 192
MAX_OPERATIONS = 200_000
MAX_DRAWS = 1024
PRIMITIVES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z",
                        "reset", "measure", "barrier"})
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
SOURCE_STATUS = "source_only_execution_evidence_external"


@dataclass(frozen=True)
class StructuredDQILimits:
    """Materialization limits only; a simulator must impose separate limits."""

    max_n: int = MAX_N
    max_qubits: int = MAX_QUBITS
    max_operations: int = MAX_OPERATIONS

    def __post_init__(self):
        for name, ceiling in (("max_n", MAX_N), ("max_qubits", MAX_QUBITS),
                              ("max_operations", MAX_OPERATIONS)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def _limits(value: StructuredDQILimits | None) -> StructuredDQILimits:
    if value is None:
        return StructuredDQILimits()
    if not isinstance(value, StructuredDQILimits):
        raise ValueError("limits must be a StructuredDQILimits instance")
    return value


def _n(value: int, limits: StructuredDQILimits | None = None) -> int:
    limits = _limits(limits)
    if type(value) is not int or not 2 <= value <= limits.max_n:
        raise ValueError("The degree-one structured family requires integer n in 2..max_n")
    if 3 * value > limits.max_qubits or 48 * value * value - 20 * value - 36 > limits.max_operations:
        raise ValueError("The complete initialized draw exceeds its materialization limits")
    return value


def _wires(*values: int):
    if len(values) > MAX_QUBITS:
        raise ValueError("Primitive wire list exceeds its fixed cap")
    if any(type(value) is not int or not 0 <= value < MAX_QUBITS for value in values):
        raise ValueError("Primitive wires must be integer indices in 0..191")
    if len(set(values)) != len(values):
        raise ValueError("Primitive wires must be distinct")


@dataclass(frozen=True)
class Operation:
    """One explicit primitive; parameter values are unsynthesized binary floats."""

    name: str
    qubits: tuple[int, ...]
    classical: tuple[int, ...] = ()
    parameters: tuple[float, ...] = ()

    def __post_init__(self):
        if type(self.name) is not str or self.name not in PRIMITIVES or type(self.qubits) is not tuple:
            raise ValueError("An Operation requires a listed primitive and tuple of quantum wires")
        if len(self.qubits) > MAX_QUBITS:
            raise ValueError("Primitive quantum wire list exceeds its fixed cap")
        _wires(*self.qubits)
        arity = 2 if self.name == "cx" else 1
        if self.name == "barrier":
            if not self.qubits:
                raise ValueError("A barrier must name its quantum wires")
        elif len(self.qubits) != arity:
            raise ValueError("Primitive quantum arity changed")
        if (type(self.classical) is not tuple or len(self.classical) > 1
                or any(type(c) is not int or not 0 <= c < MAX_N for c in self.classical)):
            raise ValueError("Classical wires must be integer indices in 0..63")
        if len(self.classical) != (1 if self.name == "measure" else 0):
            raise ValueError("Only readout names one classical target")
        if type(self.parameters) is not tuple or len(self.parameters) != (1 if self.name == "ry" else 0):
            raise ValueError("Only RY names one angle")
        if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
               for p in self.parameters):
            raise ValueError("Rotation parameters must be finite real numbers")
        object.__setattr__(self, "parameters", tuple(float(p) for p in self.parameters))


def exact_ccx_operations(control0: int, control1: int, target: int) -> tuple[Operation, ...]:
    """The existing exact 15-primitive CCX sequence, with no relative phase."""
    _wires(control0, control1, target)
    return (Operation("h", (target,)), Operation("cx", (control1, target)),
            Operation("tdg", (target,)), Operation("cx", (control0, target)),
            Operation("t", (target,)), Operation("cx", (control1, target)),
            Operation("tdg", (target,)), Operation("cx", (control0, target)),
            Operation("t", (control1,)), Operation("t", (target,)),
            Operation("h", (target,)), Operation("cx", (control0, control1)),
            Operation("t", (control0,)), Operation("tdg", (control1,)),
            Operation("cx", (control0, control1)))


def controlled_ry_operations(control: int, target: int, angle: float) -> tuple[Operation, ...]:
    """Two RY and two CX primitives; reject lossy halving before any operation."""
    _wires(control, target)
    if isinstance(angle, bool) or not isinstance(angle, (int, float)):
        raise ValueError("The controlled-RY angle must be a finite real number")
    try:
        finite = math.isfinite(angle)
    except OverflowError as error:
        raise ValueError("The controlled-RY angle must be representable as a finite real number") from error
    if not finite:
        raise ValueError("The controlled-RY angle must be a finite real number")
    if 2 * (angle / 2) != angle:
        raise ValueError("The controlled-RY angle must survive halving exactly")
    return (Operation("ry", (target,), parameters=(angle / 2,)),
            Operation("cx", (control, target)),
            Operation("ry", (target,), parameters=(-angle / 2,)),
            Operation("cx", (control, target)))


def clean_mcx_operations(controls: tuple[int, ...], target: int,
                         workspace: tuple[int, ...]) -> Iterator[Operation]:
    """Conjunction chain, target toggle, inverse chain; workspace starts zero.

    For n>=3 and dirty initial workspace, computed w'_0=w_0 XOR c_0*c_1 and
    w'_j=w_j XOR w'_(j-1)*c_(j+1) yield target toggle w'_(n-3)*c_(n-1).
    Reversing the chain restores workspace but not the clean target law. With
    two controls there is no conjunction workspace and the toggle is c_0*c_1.
    """
    if (type(controls) is not tuple or type(workspace) is not tuple
            or not 2 <= len(controls) <= MAX_N or len(workspace) > MAX_N - 1):
        raise ValueError("An MCX requires 2..64 tuple controls and tuple workspace")
    if len(workspace) < len(controls) - 2:
        raise ValueError("The MCX conjunction chain requires n-2 clean workspace wires")
    _wires(*controls, target, *workspace)
    def sequence():
        if len(controls) == 2:
            yield from exact_ccx_operations(controls[0], controls[1], target)
            return
        yield from exact_ccx_operations(controls[0], controls[1], workspace[0])
        for index in range(1, len(controls) - 2):
            yield from exact_ccx_operations(workspace[index - 1], controls[index + 1], workspace[index])
        yield from exact_ccx_operations(workspace[len(controls) - 3], controls[-1], target)
        for index in reversed(range(1, len(controls) - 2)):
            yield from exact_ccx_operations(workspace[index - 1], controls[index + 1], workspace[index])
        yield from exact_ccx_operations(controls[0], controls[1], workspace[0])
    return sequence()


def register_map(n: int) -> dict:
    n = _n(n)
    return {"clause": list(range(n + 1)), "assignment": list(range(n + 1, 2 * n + 1)),
            "workspace": list(range(2 * n + 1, 3 * n)), "classical": list(range(n)),
            "total_qubits": 3 * n}


def family_spec(n: int) -> dict:
    n = _n(n)
    return {"family": "binary_identity_plus_all_ones_degree_one", "n": n, "m": n + 1, "ell": 1,
            "matrix": [[int(i == j) for j in range(n)] for i in range(n)] + [[1] * n],
            "signs": [0] * n + [1], "bit_order": "integer bit j is x_j; display x_(n-1)..x_0",
            "prepared_domain": "vacuum and every clause singleton", "domain_size": n + 2,
            "kernel_minimum_distance": n + 1, "workspace_input": "all n-1 workspace wires zero",
            "decoder_scope": "clean-workspace XOR on arbitrary clause targets; dirty target dependence for n>=3"}


def decode_syndrome(n: int, syndrome: int) -> int:
    """Total structured classical decoder specification, without a syndrome table."""
    n = _n(n)
    if type(syndrome) is not int or not 0 <= syndrome < (1 << n):
        raise ValueError("Syndrome exceeds its n-bit register")
    if syndrome == (1 << n) - 1:
        return 1 << n
    if syndrome and syndrome & (syndrome - 1) == 0:
        return 1 << (syndrome.bit_length() - 1)
    return 0


def objective(n: int, assignment: int) -> int:
    n = _n(n)
    if type(assignment) is not int or not 0 <= assignment < (1 << n):
        raise ValueError("Assignment exceeds its n-bit register")
    weight = assignment.bit_count()
    return n - weight + (weight & 1)


def analytic_ledger(n: int) -> dict:
    """Exact source prediction, independently checked against emitted primitives later."""
    n = _n(n)
    counts = {"h": 6 * n * n - 3 * n - 6, "cx": 18 * n * n - 8 * n - 18,
              "t": 12 * n * n - 8 * n - 12, "tdg": 9 * n * n - 6 * n - 9,
              "x": 3 * n * n - n, "ry": 2 * n + 1, "z": 1,
              "reset": 3 * n, "measure": n, "barrier": 7}
    lengths = {"reset": 3 * n, "preparation": 16 * n * n - 10 * n + 1,
               "sign": 1, "syndrome": 2 * n, "decoder": 32 * n * n - 17 * n - 45,
               "hadamard": n, "readout": n}
    cursor, intervals = 0, {}
    for name in STAGES:
        stop = cursor + lengths[name]
        intervals[name] = {"start": cursor, "stop": stop, "barrier_index": stop, "operations": lengths[name]}
        cursor = stop + 1
    return {"schema": "openquantum.dqi.structured_source_ledger.v1", "n": n, "m": n + 1, "ell": 1,
            "total_qubits": 3 * n, "classical_readout_bits": n, "counts": counts,
            "unitary_gates": 48 * n * n - 24 * n - 43,
            "unitary_serial_layer_upper_bound": 48 * n * n - 24 * n - 43,
            "schedule_scope": "all unitary primitives serial; parallel/routed/physical scheduling pending",
            "events_without_barriers": 48 * n * n - 20 * n - 43,
            "source_operations": 48 * n * n - 20 * n - 36,
            "unsynthesized_ry": 2 * n + 1, "explicit_t_tdg": 21 * n * n - 14 * n - 21,
            "stage_intervals": intervals,
            "workspace": {"allocated_clean": n - 1, "preparation_peak_used": n - 1,
                          "decoder_peak_used": n - 2, "serial_reuse": True,
                          "restored_after_each_predicate": True, "clean_target_law_requires_zero_input": True},
            "ccx_blocks": {"preparation": n * (n - 1), "decoder": (n + 1) * (2 * n - 3)},
            "accounting_scope": "one complete initialized draw, including preparation/sign/syndrome/decoder/uncompute/reset/readout",
            "exclusions": ["all RY synthesis and approximation allocation", "routing/scheduling and physical faults",
                           "classical compilation, angle generation and input delivery time",
                           "classical objective scoring and output selection time"],
            "classical_work": {"family_input": "n specifies fixed B/v; an explicit B dump takes O(n^2) bits",
                               "angle_generation": "O(n) classical coefficient/angle calculations, time unmeasured",
                               "primitive_compilation": "O(n^2) emitted primitives, time separate from quantum runtime",
                               "scoring_per_assignment": "O(n) bit/parity work and output selection, time unmeasured",
                               "hardware_input_delivery": "not measured or represented as a generic loading oracle"},
            "physical_estimate_status": "pending", "execution_evidence": "external; formulas alone are not validation"}


def _zero_prefix_operations(controls: tuple[int, ...], target: int, angle: float,
                            workspace: tuple[int, ...]) -> Iterator[Operation]:
    _wires(*controls, target, *workspace)
    # Validate before yielding X conjugations, so rejection cannot partly mutate a consumer.
    rotation = controlled_ry_operations(controls[0] if len(controls) == 1 else workspace[len(controls) - 2],
                                       target, angle) if controls else (Operation("ry", (target,), parameters=(angle,)),)
    def sequence():
        for wire in controls:
            yield Operation("x", (wire,))
        if len(controls) >= 2:
            yield from exact_ccx_operations(controls[0], controls[1], workspace[0])
            for index in range(1, len(controls) - 1):
                yield from exact_ccx_operations(workspace[index - 1], controls[index + 1], workspace[index])
        yield from rotation
        if len(controls) >= 2:
            for index in reversed(range(1, len(controls) - 1)):
                yield from exact_ccx_operations(workspace[index - 1], controls[index + 1], workspace[index])
            yield from exact_ccx_operations(controls[0], controls[1], workspace[0])
        for wire in reversed(controls):
            yield Operation("x", (wire,))
    return sequence()


def iter_stage_operations(n: int, stage: str, limits: StructuredDQILimits | None = None) -> Iterator[Operation]:
    """Return one bounded stage without barriers; reject bounds before iteration."""
    n = _n(n, limits)
    if type(stage) is not str or stage not in STAGES:
        raise ValueError("Unknown structured DQI stage")
    registers = register_map(n)
    clause, assignment = tuple(registers["clause"]), tuple(registers["assignment"])
    workspace = tuple(registers["workspace"])
    def sequence():
        if stage == "reset":
            for wire in range(3 * n):
                yield Operation("reset", (wire,))
        elif stage == "preparation":
            for index, target in enumerate(clause):
                angle = 2 * math.asin(1 / math.sqrt(n + 2 - index))
                yield from _zero_prefix_operations(clause[:index], target, angle, workspace)
        elif stage == "sign":
            yield Operation("z", (clause[n],))
        elif stage == "syndrome":
            for index, target in enumerate(assignment):
                yield Operation("cx", (clause[index], target))
            for target in assignment:
                yield Operation("cx", (clause[n], target))
        elif stage == "decoder":
            for index, target in enumerate(clause):
                negative = tuple(wire for j, wire in enumerate(assignment) if index < n and j != index)
                for wire in negative:
                    yield Operation("x", (wire,))
                yield from clean_mcx_operations(assignment, target, workspace)
                for wire in reversed(negative):
                    yield Operation("x", (wire,))
        elif stage == "hadamard":
            for wire in assignment:
                yield Operation("h", (wire,))
        else:
            for bit, wire in enumerate(assignment):
                yield Operation("measure", (wire,), classical=(bit,))
    return sequence()


def iter_operations(n: int, limits: StructuredDQILimits | None = None) -> Iterator[Operation]:
    """Stream the complete draw; no operation list or quantum arrays are allocated."""
    limits = _limits(limits)
    n = _n(n, limits)
    def sequence():
        for stage in STAGES:
            yield from iter_stage_operations(n, stage, limits)
            yield Operation("barrier", tuple(range(3 * n)))
    return sequence()


@dataclass(frozen=True)
class StructuredDQIProgram:
    n: int
    limits: StructuredDQILimits = StructuredDQILimits()

    def __post_init__(self):
        _n(self.n, self.limits)

    @property
    def registers(self) -> dict:
        return register_map(self.n)

    def operations(self) -> Iterator[Operation]:
        return iter_operations(self.n, self.limits)

    def stage_operations(self, stage: str) -> Iterator[Operation]:
        return iter_stage_operations(self.n, stage, self.limits)

    def ledger(self) -> dict:
        return analytic_ledger(self.n)

    def __iter__(self) -> Iterator[Operation]:
        return self.operations()


def structured_program(n: int, limits: StructuredDQILimits | None = None) -> StructuredDQIProgram:
    return StructuredDQIProgram(n, _limits(limits))


def _fraction(value: Fraction | str) -> Fraction:
    if type(value) is str:
        if len(value) > 64 or re.fullmatch(r"[0-9]+/[0-9]+", value) is None:
            raise ValueError("The exact failure target must be at most 64 ASCII digits/slash, without exponents")
        if not any(character != "0" for character in value.split("/", 1)[1]):
            raise ValueError("The exact failure target denominator must be positive")
        value = Fraction(value)
    if type(value) is not Fraction:
        raise ValueError("delta must be an exact Fraction or bounded fraction string strictly between zero and one")
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 256:
        raise ValueError("The exact failure target exceeds its integer-size cap")
    if not 0 < value < 1:
        raise ValueError("delta must be strictly between zero and one")
    return value


def repetition_plan(n: int, delta: Fraction | str = Fraction(1, 100), max_draws: int = MAX_DRAWS) -> dict:
    """Capped exact ideal accounting; never sample or calculate an unbounded draw list."""
    n = _n(n)
    delta = _fraction(delta)
    if type(max_draws) is not int or not 1 <= max_draws <= MAX_DRAWS:
        raise ValueError("max_draws must be an integer in 1..1024")
    good = Fraction((n + 1) * n * n, (1 << n) * (n + 2))
    failure, required = Fraction(1), None
    for draws in range(1, max_draws + 1):
        failure *= 1 - good
        if failure <= delta:
            required = draws
            break
    charged = required if required is not None else max_draws
    ledger = analytic_ledger(n)
    return {"scope": "exact ideal independent-draw accounting only; no draws executed or sampled",
            "status": "feasible" if required is not None else "cap_exceeded",
            "optimal_probability": {"numerator": good.numerator, "denominator": good.denominator},
            "requested_failure": {"numerator": delta.numerator, "denominator": delta.denominator},
            "max_draws": max_draws, "required_draws": required, "quality_feasible": required is not None,
            "minimum_required_draws_lower_bound": required if required is not None else max_draws + 1,
            "draws_charged": charged,
            "failure_probability_after_charged_draws": {"numerator_hex": format(failure.numerator, "x"),
                                                       "denominator_hex": format(failure.denominator, "x")},
            "serial_peak_qubits": 3 * n, "complete_unitary_gates": charged * ledger["unitary_gates"],
            "complete_source_operations": charged * ledger["source_operations"],
            "complete_counts": {name: charged * count for name, count in ledger["counts"].items()},
            "classical_scoring": "n+1 binary constraints and output selection per draw; CPU time unmeasured",
            "quality_limit": "ideal law only; approximate/synthesized/physical draws need a justified probability lower bound",
            "classical_baseline": "the all-zero assignment is an immediate optimum; no optimization advantage"}


def qasm_header(n: int) -> str:
    n = _n(n)
    return f'OPENQASM 3.0;\ninclude "stdgates.inc";\nbit[{n}] c;\nqubit[{3 * n}] q;\n'


def operation_to_record(operation: Operation) -> list:
    if not isinstance(operation, Operation):
        raise ValueError("Expected a structured DQI Operation")
    return [operation.name, list(operation.qubits), list(operation.classical), list(operation.parameters)]


def operation_to_qasm(operation: Operation) -> str:
    if not isinstance(operation, Operation):
        raise ValueError("Expected a structured DQI Operation")
    wires = ", ".join(f"q[{wire}]" for wire in operation.qubits)
    if operation.name == "measure":
        return f"c[{operation.classical[0]}] = measure {wires};\n"
    angle = f"({operation.parameters[0]!r})" if operation.parameters else ""
    return f"{operation.name}{angle} {wires};\n"
