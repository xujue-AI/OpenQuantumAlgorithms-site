# SPDX-License-Identifier: MIT
"""Independent, opt-in checks of the fixed twelve-wire degree-two C5 circuit.

This source is prepared but has not been numerically executed in this increment.
It imports no producer or quantum SDK.  Its only complete circuit evolution is
one in-place 4096-amplitude forward pass, with diagnostics at barriers in that
same pass.  Decoder coverage instead uses 512 Boolean mappings and one small
8-by-8 phase-exact CCX reference.  No full circuit operator is constructed.

All state/matrix comparisons are floating diagnostics, not interval or hardware
certificates.  Assignment probabilities include every clause/workspace value;
there is no auxiliary postselection, normalization or global-phase fitting.
"""

from __future__ import annotations

from collections import Counter
import cmath
from fractions import Fraction
import math
from typing import Callable


SOURCE_STATUS = "prepared_source_no_numerical_execution"
RESULT_SCHEMA = "openquantum.dqi.degree_two_validation.v1"
RESULT_STATUS = "passed_c5_degree_two_component_pending_invocation_acceptance"
MAX_RECORDS = 2_000
STATE_WIRES = 12
STATE_AMPLITUDES = 4_096
SMALL_MATRIX_DIMENSION = 8
MAX_QUANTUM_SOURCE_BYTES = 32 << 20
ABSOLUTE_TOLERANCE = 2e-10
RELATIVE_TOLERANCE = 0.0
C5_EDGES = ((0, 1), (1, 2), (2, 3), (3, 4), (4, 0))
C5_SIGNS = (1, 1, 1, 1, 1)
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
COUNTS = {"h": 222, "cx": 686, "t": 436, "tdg": 327, "x": 80,
          "ry": 17, "z": 5, "reset": 12, "measure": 4, "barrier": 7}
STAGE_LENGTHS = {"reset": 12, "preparation": 425, "sign": 5,
                 "syndrome": 8, "decoder": 1331, "hadamard": 4, "readout": 4}
SIGNED_NUMERATORS = (0, -1, -1, -1, -1, 6, -1, -1, -1, 6, 6, 6, -1, 6, -1, -1)
UNITARY_NAMES = frozenset({"h", "x", "z", "t", "tdg", "ry", "cx"})
_VERIFICATION_INVOCATION_SPENT = False


class DegreeTwoVerificationError(ValueError):
    """A rejected guard, malformed source, or failed independent comparison."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DegreeTwoVerificationError(message)


def _integer(value: object, lower: int, upper: int, label: str) -> int:
    _require(type(value) is int and lower <= value <= upper,
             label + " must be a bounded integer")
    return value


def _sequence(value: object, maximum: int, label: str) -> list | tuple:
    _require(type(value) in (list, tuple) and len(value) <= maximum,
             label + " must be a bounded list or tuple")
    return value


def _check(callback: Callable[[], None]) -> None:
    _require(callback() is None, "A deadline callback must return None")


def validate_quantum_shape(role: str, *, wires=None, amplitudes=None,
                           matrix_dimension=None) -> None:
    """Pure shape guard; this function allocates no state or matrix."""
    if role == "full_state":
        _require(type(wires) is int and wires == STATE_WIRES
                 and type(amplitudes) is int and amplitudes == STATE_AMPLITUDES
                 and matrix_dimension is None,
                 "Only one twelve-wire/4096-amplitude complete state is supported")
    elif role == "small_ccx":
        _require(type(wires) is int and wires == 3
                 and type(matrix_dimension) is int and matrix_dimension == 8
                 and amplitudes is None,
                 "Only the three-wire 8-by-8 CCX reference is supported")
    else:
        raise DegreeTwoVerificationError("Dense circuit operators and other quantum roles are forbidden")


class ExecutionBudget:
    """Pure precharged counters; reservation methods perform no science.

    An authorized verifier enters once and charges each role before its work.
    Failed/aborted budgets remain spent.  Counts are charged entries rather
    than evidence of success; only a passed component and complete=True mean
    that every comparison returned normally.
    """

    __slots__ = ("_enabled", "_entered", "_active", "_complete", "_counts")

    def __init__(self, execution_enabled: bool = False):
        _require(type(execution_enabled) is bool, "Execution enablement must be Boolean")
        self._enabled = execution_enabled
        self._entered = self._active = self._complete = False
        self._counts = {"verification_invocations": 0,
                        "complete_statevector_evolutions": 0,
                        "boolean_decoder_mappings": 0,
                        "small_ccx_reference_cases": 0,
                        "primitive_unitary_applications": 0}

    def begin(self) -> None:
        _require(self._enabled, "Numerical verification is disabled")
        _require(not self._entered, "The verification budget is already spent")
        self._entered = self._active = True
        self._counts["verification_invocations"] = 1

    def _charge(self, key: str, maximum: int) -> None:
        _require(self._active, "A reservation requires the active verification budget")
        _require(self._counts[key] < maximum, "The fixed " + key + " ceiling was reached")
        self._counts[key] += 1

    def reserve_full_evolution(self, *, wires=12, amplitudes=4096) -> None:
        validate_quantum_shape("full_state", wires=wires, amplitudes=amplitudes)
        self._charge("complete_statevector_evolutions", 1)

    def reserve_boolean_mapping(self) -> None:
        self._charge("boolean_decoder_mappings", 512)

    def reserve_ccx_reference(self, *, matrix_dimension=8) -> None:
        validate_quantum_shape("small_ccx", wires=3, matrix_dimension=matrix_dimension)
        self._charge("small_ccx_reference_cases", 1)

    def charge_unitary_gate(self) -> None:
        self._charge("primitive_unitary_applications", 1773)

    def finish(self) -> None:
        _require(self._active, "No active verification can finish")
        _require(self._counts == {"verification_invocations": 1,
                                 "complete_statevector_evolutions": 1,
                                 "boolean_decoder_mappings": 512,
                                 "small_ccx_reference_cases": 1,
                                 "primitive_unitary_applications": 1773},
                 "The complete verification did not consume its exact roles")
        self._active = False
        self._complete = True

    def abort(self) -> None:
        self._active = False
        self._complete = False

    def snapshot(self) -> dict:
        return {**self._counts, "spent": self._entered, "active": self._active,
                "complete": self._complete,
                "counter_scope": "charged entries; successful component and complete flag required"}


class _QuantumEnvelope:
    """Conservative source-level accounting, separate from whole-process RSS."""

    def __init__(self):
        self.live = {}
        self.total = self.peak = 0

    def reserve(self, name: str, count: int, bytes_per_element: int) -> None:
        _integer(count, 0, STATE_AMPLITUDES, "Quantum element count")
        _require(name not in self.live, "Quantum buffer reservation was duplicated")
        _integer(bytes_per_element, 1, 256, "Quantum element envelope")
        amount = count * bytes_per_element + 4096
        _require(self.total + amount <= MAX_QUANTUM_SOURCE_BYTES,
                 "The 32 MiB quantum source envelope was exceeded")
        self.live[name] = amount
        self.total += amount
        self.peak = max(self.peak, self.total)

    def release(self, name: str) -> None:
        _require(name in self.live, "Unknown quantum buffer reservation")
        self.total -= self.live.pop(name)


def validate_primitive_record(value: object) -> tuple:
    """Pure serialized-record checks; no trigonometry or quantum work."""
    _require(type(value) is dict and set(value) <= {"gate", "qubits", "classical", "parameters_hex"}
             and {"gate", "qubits"} <= set(value), "A primitive record has unknown or missing fields")
    name = value["gate"]
    _require(type(name) is str and name in UNITARY_NAMES | {"reset", "measure", "barrier"},
             "An opaque or unsupported primitive is forbidden")
    raw_qs = _sequence(value["qubits"], 12, "Quantum operands")
    qs = tuple(_integer(q, 0, 11, "Quantum wire") for q in raw_qs)
    _require(len(set(qs)) == len(qs), "Quantum operands must be distinct")
    raw_cs = _sequence(value.get("classical", ()), 1, "Classical operands")
    cs = tuple(_integer(c, 0, 3, "Classical bit") for c in raw_cs)
    raw_angles = _sequence(value.get("parameters_hex", ()), 1, "RY parameters")
    angles = []
    for text in raw_angles:
        _require(type(text) is str and 1 <= len(text) <= 80, "An angle requires bounded hexadecimal text")
        try:
            angle = float.fromhex(text)
        except (ValueError, OverflowError) as error:
            raise DegreeTwoVerificationError("An angle is not binary hexadecimal") from error
        _require(math.isfinite(angle) and angle.hex() == text and abs(angle) <= math.pi,
                 "A source angle must be finite, canonical and bounded by pi")
        angles.append(angle)
    if name == "barrier":
        _require(qs == tuple(range(12)) and not cs and not angles,
                 "A source barrier must cover all twelve wires")
    elif name == "cx":
        _require(len(qs) == 2 and not cs and not angles, "CX requires two wires only")
    elif name == "measure":
        _require(len(qs) == 1 and len(cs) == 1 and not angles, "A readout requires one wire and classical bit")
    elif name == "ry":
        _require(len(qs) == 1 and not cs and len(angles) == 1, "RY requires one wire and one angle")
    else:
        _require(len(qs) == 1 and not cs and not angles, "A one-qubit primitive requires one wire only")
    return name, qs, cs, tuple(angles)


def ideal_draw_contract() -> dict:
    """Pure fixed-instance analytic contract, not execution or hardware evidence."""
    return {"objective_condition": "f(x)=4", "ideal_success_target_fraction": "99/100",
            "ideal_optimum_probability_fraction": "18/19", "ideal_draws_for_99": 2,
            "ideal_miss_two_draws_fraction": "1/361", "prospective_complete_task_draws": 3,
            "ideal_miss_three_draws_fraction": "1/6859", "prospective_realization_ceiling_per_draw": "1/1200",
            "source_explicit_t_tdg_three_draws": 2289, "physical_budget_met": False,
            "factory_availability_budget_met": False, "draws_sampled": False}


def _c5_input(vertices, edges, signs) -> tuple:
    _require(type(vertices) is int and vertices == 5, "Only the five-vertex C5 input is supported")
    raw_edges = _sequence(edges, 5, "Edges")
    _require(len(raw_edges) == 5, "C5 has exactly five ordered edges")
    normalized_edges = []
    for edge in raw_edges:
        pair = _sequence(edge, 2, "Edge")
        _require(len(pair) == 2, "An edge requires two vertex labels")
        normalized_edges.append(tuple(_integer(v, 0, 4, "Vertex") for v in pair))
    raw_signs = _sequence(signs, 5, "Signs")
    normalized_signs = tuple(_integer(v, 0, 1, "Target bit") for v in raw_signs)
    _require(tuple(normalized_edges) == C5_EDGES and normalized_signs == C5_SIGNS,
             "The frozen edge/target order differs from the fixed C5 input")
    return tuple(normalized_edges), normalized_signs


def _ledger(ledger: object, operations: tuple, edges: tuple, signs: tuple) -> dict:
    _require(type(ledger) is dict, "The source ledger must be a dictionary")
    required = {"N": 5, "n": 4, "m": 5, "source_wires": 12,
                "shared_clean_workspace": 3, "source_unitary_primitives": 1773,
                "source_materialized_operations": 1796, "normalization_squared": 190}
    for key, expected in required.items():
        _require(type(ledger.get(key)) is int and ledger[key] == expected,
                 "The fixed source ledger changed: " + key)
    coefficients = _sequence(ledger.get("per_pattern_coefficients"), 3, "Coefficients")
    _require(tuple(coefficients) == (5, 5, 2) and all(type(v) is int for v in coefficients),
             "The degree-two coefficient contract changed")
    _c5_input(ledger.get("N"), ledger.get("edges"), ledger.get("signs"))
    counts = ledger.get("source_counts")
    _require(type(counts) is dict and set(counts) == set(COUNTS)
             and all(type(counts[k]) is int and counts[k] == COUNTS[k] for k in COUNTS),
             "The source primitive ledger changed")
    observed = dict(Counter(op[0] for op in operations))
    _require(observed == COUNTS, "Actual serialized gates disagree with the complete source ledger")
    return observed


def _segments(operations: tuple) -> tuple[dict, dict]:
    result, spans = {}, {}
    start = 0
    count = 0
    for index, operation in enumerate(operations):
        if operation[0] == "barrier":
            _require(count < 7, "Too many source stage barriers")
            name = STAGES[count]
            result[name] = operations[start:index]
            spans[name] = {"start": start, "stop": index, "barrier_index": index}
            _require(index - start == STAGE_LENGTHS[name], "A stage length changed: " + name)
            start = index + 1
            count += 1
    _require(count == 7 and start == len(operations), "The source does not end at its seventh barrier")
    _require(result["reset"] == tuple(("reset", (q,), (), ()) for q in range(12)),
             "All twelve initial resets must be explicit and ordered")
    _require(result["sign"] == tuple(("z", (q,), (), ()) for q in range(5)), "The five sign Zs changed")
    _require(result["hadamard"] == tuple(("h", (q,), (), ()) for q in range(5, 9)), "Final assignment Hadamards changed")
    _require(result["readout"] == tuple(("measure", (5 + j,), (j,), ()) for j in range(4)),
             "All assignment readouts and their bit order must be explicit")
    return result, spans


def _ccx_word(a: int, b: int, target: int) -> tuple:
    """Independent literal phase-exact fifteen-primitive reference."""
    return (("h", (target,), (), ()), ("cx", (b, target), (), ()),
            ("tdg", (target,), (), ()), ("cx", (a, target), (), ()),
            ("t", (target,), (), ()), ("cx", (b, target), (), ()),
            ("tdg", (target,), (), ()), ("cx", (a, target), (), ()),
            ("t", (b,), (), ()), ("t", (target,), (), ()),
            ("h", (target,), (), ()), ("cx", (a, b), (), ()),
            ("t", (a,), (), ()), ("tdg", (b,), (), ()), ("cx", (a, b), (), ()))


def _consume(sequence: tuple, position: int, expected: tuple) -> int:
    _require(sequence[position:position + len(expected)] == expected,
             "An explicit counter/decoder primitive pattern changed")
    return position + len(expected)


def _preparation_structure(sequence: tuple) -> dict:
    position = 0
    angle_blocks = 0
    for clause in range(5):
        if clause == 0:
            _require(sequence[position][:3] == ("ry", (0,), ()) and sequence[position][3][0] > 0,
                     "The first preparation rotation must be direct and positive")
            position += 1
            angle_blocks += 1
        else:
            for prefix_weight in (0, 1):
                negatives = (9, 10) if prefix_weight == 0 else (10,)
                position = _consume(sequence, position, tuple(("x", (q,), (), ()) for q in negatives))
                position = _consume(sequence, position, _ccx_word(9, 10, 11))
                first = sequence[position]
                _require(first[:3] == ("ry", (clause,), ()) and len(first[3]) == 1 and first[3][0] > 0,
                         "A controlled preparation half-angle is malformed")
                half = first[3][0]
                position = _consume(sequence, position,
                                    (first, ("cx", (11, clause), (), ()),
                                     ("ry", (clause,), (), (-half,)), ("cx", (11, clause), (), ())))
                position = _consume(sequence, position, _ccx_word(9, 10, 11))
                position = _consume(sequence, position, tuple(("x", (q,), (), ()) for q in reversed(negatives)))
                angle_blocks += 1
        if clause < 4:
            position = _consume(sequence, position, _ccx_word(clause, 9, 10))
            position = _consume(sequence, position, (("cx", (clause, 9), (), ()),))
    for clause in range(3, -1, -1):
        position = _consume(sequence, position, (("cx", (clause, 9), (), ()),))
        position = _consume(sequence, position, _ccx_word(clause, 9, 10))
    _require(position == len(sequence) and angle_blocks == 9, "Prefix-counter preparation is incomplete")
    return {"direct_rotations": 1, "counter_value_rotations": 8,
            "forward_prefix_increments": 4, "inverse_prefix_increments": 4,
            "source_ry_occurrences": 17, "counter_bits": [9, 10], "comparison_flag": 11,
            "last_counter_update_omitted": True, "cleanup_checks_are_unconditional": True}


def _independent_support(edges: tuple, signs: tuple, check: Callable) -> tuple:
    supports = [()]
    supports.extend((i,) for i in range(5))
    supports.extend((i, j) for i in range(5) for j in range(i + 1, 5))
    entries = []
    decoder = {}
    for support in supports:
        _check(check)
        boundary = [0] * 5
        phase = 0
        word = 0
        for edge_index in support:
            u, v = edges[edge_index]
            boundary[u] ^= 1
            boundary[v] ^= 1
            phase ^= signs[edge_index]
            word |= 1 << edge_index
        syndrome = sum(bit << j for j, bit in enumerate(boundary[1:]))
        _require(syndrome not in decoder, "The independently formed D2 syndrome map collided")
        coefficient = 2 if len(support) == 2 else 5
        signed_coefficient = -coefficient if phase else coefficient
        entries.append((word, syndrome, coefficient, signed_coefficient, support))
        decoder[syndrome] = word
    _require(len(entries) == 16 and sum(entry[2] ** 2 for entry in entries) == 190,
             "The independent prepared-support normalization changed")
    return tuple(entries), decoder


def _source_access_and_lookup(segments: dict, ledger: dict, edges: tuple,
                              entries: tuple) -> None:
    access = []
    for edge_index, edge in enumerate(edges):
        for vertex in edge:
            if vertex != 0:
                access.append(("cx", (edge_index, 5 + vertex - 1), (), ()))
    _require(segments["syndrome"] == tuple(access), "The explicit incidence-access CXs changed")
    lookup = _sequence(ledger.get("lookup_entries"), 15, "Sparse decoder lookup")
    expected = [{"syndrome_mask": syndrome, "edge_indices": list(support)}
                for word, syndrome, coefficient, phase, support in sorted(entries, key=lambda x: x[1])
                if support]
    normalized = []
    for row in lookup:
        _require(type(row) is dict and set(row) == {"syndrome_mask", "edge_indices"},
                 "A decoder compilation entry must be a sparse mask/support pair")
        mask = _integer(row["syndrome_mask"], 1, 15, "Syndrome mask")
        support = _sequence(row["edge_indices"], 2, "Edge support")
        _require(1 <= len(support) <= 2, "A nonzero D2 lookup entry has one or two edges")
        indices = [_integer(i, 0, 4, "Edge index") for i in support]
        _require(indices == sorted(set(indices)), "Sparse support indices must be distinct and ordered")
        normalized.append({"syndrome_mask": mask, "edge_indices": indices})
    _require(normalized == expected, "The source lookup disagrees with independent edge boundaries")


def _decoder_logic(sequence: tuple, entries: tuple) -> tuple:
    parsed = []
    position = 0
    while position < len(sequence):
        operation = sequence[position]
        if operation[0] == "x":
            parsed.append(("x", operation[1]))
            position += 1
        else:
            _require(operation[0] == "h" and position + 15 <= len(sequence),
                     "The decoder contains an opaque or incomplete nonlinear block")
            target = operation[1][0]
            second, fourth = sequence[position + 1], sequence[position + 3]
            _require(second[0] == fourth[0] == "cx" and second[1][1] == fourth[1][1] == target,
                     "The decoder Toffoli operands cannot be reconstructed")
            a, b = fourth[1][0], second[1][0]
            _require(len({a, b, target}) == 3, "A decoder Toffoli aliases operands")
            position = _consume(sequence, position, _ccx_word(a, b, target))
            parsed.append(("ccx", (a, b, target)))
    expected = []
    for word, syndrome, coefficient, phase, support in sorted(entries, key=lambda x: x[1]):
        if not support:
            continue
        negatives = tuple(5 + j for j in range(4) if not (syndrome >> j) & 1)
        expected.extend(("x", (q,)) for q in negatives)
        chain = ((5, 6, 9), (9, 7, 10))
        expected.extend(("ccx", block) for block in chain)
        expected.extend(("ccx", (10, 8, target)) for target in support)
        expected.extend(("ccx", block) for block in reversed(chain))
        expected.extend(("x", (q,)) for q in reversed(negatives))
    _require(tuple(parsed) == tuple(expected), "The decoder comparison/toggle/cleanup recipe changed")
    _require(sum(name == "ccx" for name, operands in parsed) == 85
             and sum(name == "x" for name, operands in parsed) == 56,
             "The decoder logical nonlinear/negative-control ledger changed")
    return tuple(parsed)


def _boolean_decoder_checks(logic: tuple, decoder: dict, budget: ExecutionBudget,
                             check: Callable) -> dict:
    for syndrome in range(16):
        for target in range(32):
            _check(check)
            budget.reserve_boolean_mapping()
            basis = target | (syndrome << 5)
            for name, operands in logic:
                if name == "x":
                    basis ^= 1 << operands[0]
                else:
                    a, b, bit = operands
                    if (basis >> a) & 1 and (basis >> b) & 1:
                        basis ^= 1 << bit
            expected = (target ^ decoder[syndrome]) | (syndrome << 5)
            _require(basis == expected, "An arbitrary-target clean decoder basis mapping failed")
    return {"status": "passed_boolean_clean_arbitrary_target_decoder",
            "mappings": 512, "syndromes": 16, "clause_targets": 32,
            "clean_scratch_inputs_per_mapping": 1, "scratch_bits": 3,
            "initially_dirty_scratch_target_law_promised": False,
            "full_statevector_runs_for_decoder_mappings": 0}


def _one_matrix(name: str, angles: tuple) -> tuple:
    if name == "h":
        r = math.sqrt(0.5)
        return ((r, r), (r, -r))
    if name == "x":
        return ((0j, 1 + 0j), (1 + 0j, 0j))
    if name == "z":
        return ((1 + 0j, 0j), (0j, -1 + 0j))
    if name in ("t", "tdg"):
        phase = cmath.exp((1j if name == "t" else -1j) * math.pi / 4)
        return ((1 + 0j, 0j), (0j, phase))
    if name == "ry":
        c, s = math.cos(angles[0] / 2), math.sin(angles[0] / 2)
        return ((c, -s), (s, c))
    raise DegreeTwoVerificationError("No independent local matrix for this gate")


def _small_ccx_reference(budget: ExecutionBudget, envelope: _QuantumEnvelope,
                         check: Callable) -> dict:
    budget.reserve_ccx_reference(matrix_dimension=8)
    envelope.reserve("small_ccx_matrix", 64, 128)
    matrix = [[0j] * 8 for _ in range(8)]
    for index in range(8):
        matrix[index][index] = 1 + 0j
    for name, qs, cs, angles in _ccx_word(0, 1, 2):
        _check(check)
        if name == "cx":
            control, target = qs
            for row in range(8):
                if (row >> control) & 1 and not (row >> target) & 1:
                    other = row | (1 << target)
                    matrix[row], matrix[other] = matrix[other], matrix[row]
        else:
            local = _one_matrix(name, angles)
            bit = 1 << qs[0]
            for row in range(8):
                if not row & bit:
                    other = row | bit
                    for column in range(8):
                        a, b = matrix[row][column], matrix[other][column]
                        matrix[row][column] = local[0][0] * a + local[0][1] * b
                        matrix[other][column] = local[1][0] * a + local[1][1] * b
    error = 0.0
    for row in range(8):
        _check(check)
        for column in range(8):
            target_row = column ^ 4 if column & 1 and column & 2 else column
            expected = 1 + 0j if row == target_row else 0j
            value = matrix[row][column]
            _require(math.isfinite(value.real) and math.isfinite(value.imag), "Small CCX matrix became nonfinite")
            error = max(error, abs(value - expected))
    _require(error <= ABSOLUTE_TOLERANCE, "The phase-exact CCX reference failed")
    envelope.reserve("small_ccx_output_pairs", 64, 256)
    pairs = [[[z.real, z.imag] for z in row] for row in matrix]
    del matrix
    envelope.release("small_ccx_matrix")
    return {"status": "passed_small_phase_exact_ccx_reference", "matrix_dimension": 8,
            "basis_columns": 8, "primitive_gates": 15, "max_entry_discrepancy": error,
            "matrix_real_imag": pairs, "global_phase_fitting_performed": False}


def _walsh_reference(entries: tuple, edges: tuple, signs: tuple, check: Callable) -> tuple:
    numerators, objectives, probabilities = [], [], []
    for assignment in range(16):
        _check(check)
        character_sum = sum(signed * (-1 if (assignment & syndrome).bit_count() & 1 else 1)
                            for word, syndrome, coefficient, signed, support in entries)
        _require(character_sum % 4 == 0, "The independently summed C5 character numerator changed")
        numerator = character_sum // 4
        vertex_values = [0] + [(assignment >> j) & 1 for j in range(4)]
        objective = sum((vertex_values[u] ^ vertex_values[v]) == target
                        for (u, v), target in zip(edges, signs))
        numerators.append(numerator)
        objectives.append(objective)
        probabilities.append(Fraction(character_sum * character_sum, 16 * 190))
    _require(tuple(numerators) == SIGNED_NUMERATORS and sum(probabilities) == 1,
             "The direct signed-Walsh reference disagrees with the independent recipe")
    optimum = max(objectives)
    p_opt = sum((p for p, value in zip(probabilities, objectives) if value == optimum), Fraction())
    expected = sum((p * value for p, value in zip(probabilities, objectives)), Fraction())
    _require(optimum == 4 and p_opt == Fraction(18, 19) and expected == Fraction(74, 19)
             and objectives[0] == 0 and objectives[5] == 4,
             "The independent edge-based objective/quality reference changed")
    return tuple(numerators), tuple(objectives), tuple(probabilities), p_opt, expected


def _apply_in_place(state: list, operation: tuple, check: Callable) -> None:
    name, qs, cs, angles = operation
    if name == "cx":
        control, target = qs
        bit = 1 << target
        for index in range(STATE_AMPLITUDES):
            if index % 256 == 0:
                _check(check)
            if (index >> control) & 1 and not index & bit:
                other = index | bit
                state[index], state[other] = state[other], state[index]
        return
    local = _one_matrix(name, angles)
    bit = 1 << qs[0]
    for index in range(STATE_AMPLITUDES):
        if index % 256 == 0:
            _check(check)
        if not index & bit:
            other = index | bit
            a, b = state[index], state[other]
            state[index] = local[0][0] * a + local[0][1] * b
            state[other] = local[1][0] * a + local[1][1] * b


def _squares(state: list, predicate: Callable[[int], bool], check: Callable):
    for index, value in enumerate(state):
        if index % 256 == 0:
            _check(check)
        _require(math.isfinite(value.real) and math.isfinite(value.imag), "The complete state became nonfinite")
        if predicate(index):
            yield value.real * value.real + value.imag * value.imag


def _checkpoint(state: list, reference: dict, stage: str, check: Callable) -> dict:
    normalizer = math.sqrt(190)
    maximum = imaginary = 0.0
    terms = []
    # A bounded list of 4096 scalar discrepancies is covered by the source
    # envelope's diagnostic reservation. It is not another quantum evolution.
    for index, value in enumerate(state):
        if index % 256 == 0:
            _check(check)
        _require(math.isfinite(value.real) and math.isfinite(value.imag), "A checkpoint state is nonfinite")
        difference = value - reference.get(index, 0) / normalizer
        maximum = max(maximum, abs(difference))
        imaginary = max(imaginary, abs(value.imag))
        terms.append(difference.real * difference.real + difference.imag * difference.imag)
    l2 = math.sqrt(math.fsum(terms))
    norm = math.fsum(_squares(state, lambda index: True, check))
    if stage in ("preparation", "sign"):
        auxiliary = lambda index: index >> 5 != 0
    elif stage == "syndrome":
        auxiliary = lambda index: index >> 9 != 0
    else:
        auxiliary = lambda index: bool(index & 31) or index >> 9 != 0
    leakage = math.fsum(_squares(state, auxiliary, check))
    _require(maximum <= ABSOLUTE_TOLERANCE and l2 <= ABSOLUTE_TOLERANCE
             and abs(norm - 1) <= ABSOLUTE_TOLERANCE and leakage <= ABSOLUTE_TOLERANCE,
             "The unconditional full-complex " + stage + " checkpoint failed")
    result = {"norm_squared": norm, "max_state_entry_discrepancy": maximum,
              "full_state_l2_discrepancy": l2, "max_imaginary_component": imaginary,
              "auxiliary_leakage_probability": leakage, "postselection_performed": False}
    if stage == "preparation":
        above_two = math.fsum(_squares(state, lambda index: (index & 31).bit_count() > 2, check))
        _require(above_two <= ABSOLUTE_TOLERANCE, "Prepared amplitude escaped D2")
        result["weight_above_two_probability"] = above_two
        result["expected_prepared_support_size"] = 16
        result["counter_and_flag_zero_before_sign"] = True
    if stage in ("decoder", "final"):
        result["clause_workspace_leakage_probability"] = leakage
    return result


def verify_c5_records(records, ledger, *, vertices=5, edges=C5_EDGES,
                      signs=C5_SIGNS, allow_execution=False, budget=None,
                      check=None) -> dict:
    """Perform the one authorized fixed C5 component; default execution is off.

    Caller-side file/hash/CLI acceptance is separate. The mandatory callback
    supplies a cooperative deadline, backed by the future CLI's hard watchdog.
    A failed entry consumes both its module slot and budget; no reset is exposed.
    """
    global _VERIFICATION_INVOCATION_SPENT
    _require(type(allow_execution) is bool and allow_execution, "Numerical verification requires explicit opt-in")
    _require(callable(check), "Numerical verification requires a deadline callback")
    _require(type(budget) is ExecutionBudget, "The exact ExecutionBudget type is required")
    _require(not _VERIFICATION_INVOCATION_SPENT, "This process has already spent its C5 verification entry")
    _VERIFICATION_INVOCATION_SPENT = True
    try:
        budget.begin()
        _check(check)
        bounded_records = _sequence(records, MAX_RECORDS, "Source records")
        _require(len(bounded_records) == 1796, "Only the complete fixed C5 source stream is accepted")
        operations = tuple(validate_primitive_record(record) for record in bounded_records)
        edges, signs = _c5_input(vertices, edges, signs)
        source_counts = _ledger(ledger, operations, edges, signs)
        segments, spans = _segments(operations)
        preparation_structure = _preparation_structure(segments["preparation"])
        entries, decoder = _independent_support(edges, signs, check)
        _source_access_and_lookup(segments, ledger, edges, entries)
        decoder_logic = _decoder_logic(segments["decoder"], entries)
        envelope = _QuantumEnvelope()
        envelope.reserve("local_coefficients", 64, 128)
        small_reference = _small_ccx_reference(budget, envelope, check)
        boolean_report = _boolean_decoder_checks(decoder_logic, decoder, budget, check)
        envelope.reserve("walsh_reference_scalars", 64, 128)
        numerators, objectives, reference_probabilities, exact_p, exact_mean = _walsh_reference(entries, edges, signs, check)
        envelope.reserve("reference_maps", 80, 128)
        reference_maps = {
            "preparation": {word: coefficient for word, syndrome, coefficient, phase, support in entries},
            "sign": {word: phase for word, syndrome, coefficient, phase, support in entries},
            "syndrome": {word | (syndrome << 5): phase for word, syndrome, coefficient, phase, support in entries},
            "decoder": {syndrome << 5: phase for word, syndrome, coefficient, phase, support in entries},
            "final": {assignment << 5: numerator for assignment, numerator in enumerate(numerators)},
        }
        envelope.reserve("diagnostic_terms", STATE_AMPLITUDES, 64)
        budget.reserve_full_evolution(wires=12, amplitudes=4096)
        envelope.reserve("complete_state", STATE_AMPLITUDES, 128)
        state = [0j] * STATE_AMPLITUDES
        state[0] = 1 + 0j
        checkpoints = {}
        barriers = {span["barrier_index"]: name for name, span in spans.items()}
        for index, operation in enumerate(operations):
            _check(check)
            name = operation[0]
            if name in UNITARY_NAMES:
                budget.charge_unitary_gate()
                _apply_in_place(state, operation, check)
            elif name == "barrier":
                stage = barriers[index]
                key = "final" if stage == "hadamard" else stage
                if key in reference_maps:
                    checkpoints[key] = _checkpoint(state, reference_maps[key], key, check)
            # The fixed zero-input reset law and terminal computational
            # readout are checked structurally. No reset branches or outcomes
            # are generated, discarded or postselected.
        _require(set(checkpoints) == set(reference_maps), "Complete-pass checkpoints are missing")
        envelope.reserve("assignment_probabilities", 16, 64)
        probabilities = [math.fsum(_squares(state, lambda index, x=assignment: ((index >> 5) & 15) == x, check))
                         for assignment in range(16)]
        probability_sum = math.fsum(probabilities)
        total_variation = 0.5 * math.fsum(abs(p - float(reference))
                                        for p, reference in zip(probabilities, reference_probabilities))
        p_opt = math.fsum(p for p, objective in zip(probabilities, objectives) if objective == 4)
        observed_mean = math.fsum(p * objective for p, objective in zip(probabilities, objectives))
        _require(abs(probability_sum - 1) <= ABSOLUTE_TOLERANCE
                 and total_variation <= ABSOLUTE_TOLERANCE
                 and abs(p_opt - float(exact_p)) <= ABSOLUTE_TOLERANCE
                 and abs(observed_mean - float(exact_mean)) <= ABSOLUTE_TOLERANCE,
                 "Unconditional assignment/objective comparisons failed")
        envelope.reserve("final_state_output_pairs", STATE_AMPLITUDES, 256)
        final_pairs = [[value.real, value.imag] for value in state]
        outcomes = []
        for assignment in range(16):
            _check(check)
            value = state[assignment << 5]
            outcomes.append({"assignment_integer": assignment, "display_bits": format(assignment, "04b"),
                             "rooted_vertex_values": [0] + [(assignment >> j) & 1 for j in range(4)],
                             "objective": objectives[assignment],
                             "signed_reference_numerator": numerators[assignment],
                             "reference_denominator_squared": 190,
                             "reference_probability_fraction": str(reference_probabilities[assignment]),
                             "clean_auxiliary_amplitude_real_imag": [value.real, value.imag],
                             "unconditional_probability": probabilities[assignment]})
        _check(check)
        budget.finish()
        final = checkpoints["final"]
        return {"schema": RESULT_SCHEMA, "status": RESULT_STATUS,
                "input": {"vertices": 5, "reference_vertex": 0, "n": 4, "m": 5,
                          "edges": [list(edge) for edge in edges], "signs": list(signs),
                          "gauge_bit_order": "integer bit j is x_(j+1); displayed x4x3x2x1"},
                "source_counts": source_counts, "source_unitary_primitives": 1773,
                "source_materialized_operations": 1796, "source_wires": 12,
                "shared_clean_workspace": 3, "state_amplitudes": 4096,
                "normalization_squared": 190, "prepared_support_size": 16,
                "preparation_structure": preparation_structure,
                "boolean_decoder_check": boolean_report, "small_ccx_reference": small_reference,
                "checkpoints": checkpoints, "stage_ranges": spans,
                "full_state_real_imag": final_pairs, "outcomes": outcomes,
                "unconditional_assignment_probabilities": probabilities,
                "assignment_probability_scope": "unconditional marginal over every clause and workspace value",
                "assignment_probability_sum": probability_sum,
                "assignment_total_variation": total_variation,
                "optimum_probability": p_opt, "independent_optimum_probability_fraction": str(exact_p),
                "expected_objective": observed_mean, "independent_expected_objective_fraction": str(exact_mean),
                "max_state_entry_discrepancy": final["max_state_entry_discrepancy"],
                "full_state_l2_discrepancy": final["full_state_l2_discrepancy"],
                "clause_workspace_leakage_probability": final["clause_workspace_leakage_probability"],
                "numerical_tolerance": {"absolute": ABSOLUTE_TOLERANCE,
                                        "relative": RELATIVE_TOLERANCE,
                                        "scope": "floating matrix/state diagnostics; no interval certificate"},
                "quantum_storage_bound_bytes": envelope.peak,
                "quantum_storage_cap_bytes": MAX_QUANTUM_SOURCE_BYTES,
                "quantum_storage_scope": "conservative stdlib state, small matrix, reference, diagnostic and saved-pair envelopes; RSS separate",
                "counters": budget.snapshot(), "output_contract": ideal_draw_contract(),
                "classical_baseline": {"assignment_integer": 5, "display_bits": "0101", "score": 4,
                                       "all_zero_score": 0, "search_required": False,
                                       "hardware_timing_measured": False},
                "initial_resets_checked_structurally": 12, "assignment_readouts_checked_structurally": 4,
                "auxiliary_postselection_performed": False, "global_phase_fitting_performed": False,
                "phase_normalization_performed": False, "sampled_outcomes": False,
                "native_synthesis_executed": False, "native_synthesis_calls": 0,
                "direct_native_estimator_calls": 0, "physical_resources_computed": False,
                "qpu_execution": False, "formal_interval_certificate": False,
                "formal_theorem_reproduced": False, "optimization_advantage_claim": False,
                "figure_eight_evolution_performed": False, "figure_eight_state_evolved": False,
                "accepted_degree_one_producer_rerun": False, "accepted_degree_one_operator_rerun": False,
                "old_degree_one_evidence_reexecuted": False, "publication_complete": False,
                "invocation_acceptance_required": True,
                "acceptance_scope": "component checks only; external CLI exit and independently audited invocation remain required"}
    except BaseException:
        budget.abort()
        raise
