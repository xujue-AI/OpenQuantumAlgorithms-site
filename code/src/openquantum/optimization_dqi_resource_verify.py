"""Independent bounded checks for new compiled structured-DQI circuits.

This source has not been executed as part of the resource increment.  It
imports only the standard library and neither synthesizes gates nor estimates
resources.  Its small state engine accepts only the two fixed n=2/3 compiled
cases.  It does not invoke the accepted ideal producer or decoder operator.

All matrix/state diagnostics are floating numerical estimates with explicit
guards, not interval certificates, formal proofs, or hardware guarantees.
Global phase is retained; assignment probabilities include every ancilla
value rather than postselecting a clean workspace.
"""

from __future__ import annotations

from collections import Counter
import cmath
from fractions import Fraction
import math
from typing import Callable, Sequence


MAX_WORD_OPERATIONS = 2_000
MAX_COMPILED_OPERATIONS = 20_000
MAX_QUBITS = 9
MAX_STATE_AMPLITUDES = 512
MAX_ROUTED_CX_OPERATIONS = 43
MAX_QUANTUM_ARRAY_BYTES = 32 << 20
MATRIX_GUARD = 1e-12
REFERENCE_PADDING = Fraction(1, 10_000_000_000)
WORD_ERROR_CEILING = Fraction(1, 20_000)
ETA_BY_N = {2: Fraction(1, 2_000), 3: Fraction(1, 1_200)}
ROTATIONS_BY_N = {2: 5, 3: 7}
SINGLE_QUBIT_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg"})
ANGLE_HEXES = frozenset({
    "0x1.0c152382d7366p+0",
    "0x1.3b2028082e8d5p-1",
    "-0x1.3b2028082e8d5p-1",
    "0x1.921fb54442d17p-1",
    "-0x1.921fb54442d17p-1",
    "0x1.dac670561bb4fp-1",
    "0x1.0c152382d7366p-1",
    "-0x1.0c152382d7366p-1",
})


def _check(callback: Callable[[], None] | None) -> None:
    if callback is not None:
        callback()


def _callback(value: Callable[[], None] | None) -> Callable[[], None] | None:
    if value is not None and not callable(value):
        raise ValueError("A cooperative deadline check must be callable")
    return value


def _integer(value: object, lower: int, upper: int, label: str) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{label} must be an integer in {lower}..{upper}")
    return value


def _number(value: object, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{label} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{label} must be a finite real number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _case_n(value: object) -> int:
    if type(value) is not int or value not in (2, 3):
        raise ValueError("Only the new compiled n=2/3 cases are supported")
    return value


def _sequence(value: object, maximum: int, label: str) -> Sequence:
    if type(value) not in (list, tuple) or len(value) > maximum:
        raise ValueError(f"{label} must be a list/tuple of at most {maximum} entries")
    return value


def _operation(value: object, wires: int, classical_bits: int,
               *, word: bool = False, routing: bool = False) -> tuple:
    entry = _sequence(value, 4, "Operation")
    if len(entry) != 4:
        raise ValueError("An operation must have exactly four fields")
    name, qargs, cargs, params = entry
    if type(name) is not str or not 1 <= len(name) <= 16:
        raise ValueError("An operation name must be a bounded string")
    qargs = _sequence(qargs, wires, "Quantum operands")
    cargs = _sequence(cargs, classical_bits, "Classical operands")
    params = _sequence(params, 0, "Compiled parameters")
    qs = tuple(_integer(q, 0, wires-1, "Wire") for q in qargs)
    cs = tuple(_integer(c, 0, classical_bits-1, "Classical bit") for c in cargs)
    if len(set(qs)) != len(qs) or len(set(cs)) != len(cs):
        raise ValueError("Operation operands must be distinct")
    if routing:
        if name != "cx" or len(qs) != 2 or cs:
            raise ValueError("A routing block may contain only CX primitives")
    elif word:
        if name not in SINGLE_QUBIT_GATES or qs != (0,) or cs:
            raise ValueError("A synthesized word must use only one-qubit Clifford+T primitives")
    elif name in SINGLE_QUBIT_GATES or name == "reset":
        if len(qs) != 1 or cs:
            raise ValueError("A one-qubit primitive/reset needs one wire and no classical bit")
    elif name == "cx":
        if len(qs) != 2 or cs:
            raise ValueError("CX needs two distinct wires and no classical bit")
        if abs(qs[0]-qs[1]) != 1:
            raise ValueError("A compiled circuit must contain only neighboring CX gates")
    elif name == "measure":
        if len(qs) != 1 or len(cs) != 1:
            raise ValueError("A readout needs one wire and one classical bit")
    elif name == "barrier":
        if qs != tuple(range(wires)) or cs:
            raise ValueError("Retained source barriers must cover all source wires")
    else:
        raise ValueError(f"Unsupported or unsynthesized operation: {name}")
    return name, qs, cs, tuple(params)


def _matrix(name: str) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Independent primitive matrices; no SDK matrix method is consulted."""
    if name == "h":
        r = math.sqrt(0.5)
        return ((r, r), (r, -r))
    if name == "x":
        return ((0j, 1+0j), (1+0j, 0j))
    if name == "y":
        return ((0j, -1j), (1j, 0j))
    if name == "z":
        return ((1+0j, 0j), (0j, -1+0j))
    if name == "s":
        return ((1+0j, 0j), (0j, 1j))
    if name == "sdg":
        return ((1+0j, 0j), (0j, -1j))
    if name == "t":
        return ((1+0j, 0j), (0j, cmath.exp(1j*math.pi/4)))
    if name == "tdg":
        return ((1+0j, 0j), (0j, cmath.exp(-1j*math.pi/4)))
    raise ValueError("No independent matrix for this primitive")


def _matmul(left: tuple, right: tuple) -> tuple:
    result = tuple(tuple(left[i][0]*right[0][j] + left[i][1]*right[1][j]
                         for j in range(2)) for i in range(2))
    if not all(math.isfinite(z.real) and math.isfinite(z.imag)
               for row in result for z in row):
        raise ValueError("Matrix reconstruction became nonfinite")
    return result


def _norm(values) -> float:
    squares = (z.real*z.real + z.imag*z.imag for z in values)
    result = math.sqrt(math.fsum(squares))
    if not math.isfinite(result):
        raise ValueError("A numerical norm became nonfinite")
    return result


def _complex_json(value: complex) -> list[float]:
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("A retained amplitude became nonfinite")
    return [float(value.real), float(value.imag)]


def validate_word(ops: object, phase: object, angle_hex: object, *,
                  check: Callable[[], None] | None = None) -> dict:
    """Reconstruct one actual word including its native global phase.

    The only eligible angle keys are the eight accepted binary RY values.
    A caller must preserve this word and phase rather than synthesize a
    replacement.  This function contains no synthesis or estimator entry.
    """
    check = _callback(check)
    _check(check)
    entries = _sequence(ops, MAX_WORD_OPERATIONS, "Synthesized word")
    if type(angle_hex) is not str or angle_hex not in ANGLE_HEXES:
        raise ValueError("The word target must be one of the eight fixed binary angles")
    angle = float.fromhex(angle_hex)
    if not math.isfinite(angle) or angle.hex() != angle_hex:
        raise ValueError("The RY target must have an exact canonical finite binary value")
    phase_value = _number(phase, "Word global phase")
    matrix = ((1+0j, 0j), (0j, 1+0j))
    counts: Counter = Counter()
    for index, value in enumerate(entries):
        if index % 128 == 0:
            _check(check)
        name, _, _, _ = _operation(value, 1, 0, word=True)
        matrix = _matmul(_matrix(name), matrix)
        counts[name] += 1
    factor = cmath.exp(1j*phase_value)
    matrix = tuple(tuple(factor*z for z in row) for row in matrix)
    half = angle/2
    target = ((math.cos(half)+0j, -math.sin(half)+0j),
              (math.sin(half)+0j, math.cos(half)+0j))
    raw_error = _norm(matrix[i][j]-target[i][j]
                      for i in range(2) for j in range(2))
    guarded_error = raw_error + MATRIX_GUARD
    if not math.isfinite(guarded_error) or Fraction.from_float(guarded_error) > WORD_ERROR_CEILING:
        raise ValueError("Phase-inclusive word discrepancy exceeds the guarded RY allowance")
    _check(check)
    return {
        "status": "passed_numerical_word_check",
        "gate": "ry",
        "angle_hex": angle_hex,
        "global_phase_hex": phase_value.hex(),
        "operation_count": len(entries),
        "gate_counts": dict(sorted(counts.items())),
        "matrix_dimension": 2,
        "phase_inclusive_matrix": [[_complex_json(z) for z in row] for row in matrix],
        "analytic_ry_matrix": [[_complex_json(z) for z in row] for row in target],
        "raw_frobenius_error": raw_error,
        "per_matrix_numerical_guard": MATRIX_GUARD,
        "guarded_frobenius_error": guarded_error,
        "guarded_ceiling_fraction": str(WORD_ERROR_CEILING),
        "formal_interval_certificate": False,
        "phase_normalization_performed": False,
    }


def verify_routed_cx_block(wires: object, source_control: object,
                           source_target: object, routed_ops: object, *,
                           check: Callable[[], None] | None = None) -> dict:
    """Check a newly routed CX block by exact bounded Boolean action.

    CX primitives have no basis-dependent phase.  The check therefore needs
    no dense quantum operator, and is separate from the accepted decoder law.
    """
    check = _callback(check)
    _check(check)
    if type(wires) is not int or wires not in (6, 9):
        raise ValueError("Routing checks accept only the six/nine-wire resource cases")
    control = _integer(source_control, 0, wires-1, "Source control")
    target = _integer(source_target, 0, wires-1, "Source target")
    if control == target:
        raise ValueError("A source CX needs distinct control and target")
    entries = _sequence(routed_ops, MAX_ROUTED_CX_OPERATIONS, "Routed CX block")
    if not entries:
        raise ValueError("A routed CX block must contain its actual CX primitives")
    operations = tuple(_operation(op, wires, 0, routing=True) for op in entries)
    if any(abs(q[0]-q[1]) != 1 for _, q, _, _ in operations):
        raise ValueError("A routing block must use neighboring CX primitives")
    for basis in range(1 << wires):
        if basis % 128 == 0:
            _check(check)
        observed = basis
        for _, qs, _, _ in operations:
            if (observed >> qs[0]) & 1:
                observed ^= 1 << qs[1]
        expected = basis ^ (1 << target) if (basis >> control) & 1 else basis
        if observed != expected:
            raise ValueError("A routed CX block changes the source Boolean action or placement")
    _check(check)
    return {
        "status": "passed_routing_permutation_check",
        "source_control": control,
        "source_target": target,
        "source_wires": wires,
        "basis_columns_checked": 1 << wires,
        "primitive_cx_count": len(operations),
        "dense_quantum_operator_constructed": False,
        "scope": "new routing block, not the accepted decoder or ideal circuit",
    }


def _readout_map(value: object, n: int) -> tuple[tuple[int, int], ...]:
    entries = _sequence(value, n, "Assignment readout mapping")
    if len(entries) != n:
        raise ValueError("Every assignment wire must have exactly one readout mapping")
    result = []
    for entry in entries:
        if type(entry) is dict:
            if len(entry) != 2 or set(entry) != {"wire", "classical_bit"}:
                raise ValueError("A readout mapping must have exactly wire/classical_bit fields")
            wire, bit = entry["wire"], entry["classical_bit"]
        else:
            pair = _sequence(entry, 2, "Readout mapping pair")
            if len(pair) != 2:
                raise ValueError("A readout mapping pair must have length two")
            wire, bit = pair
        result.append((_integer(wire, 0, 3*n-1, "Readout wire"),
                       _integer(bit, 0, n-1, "Readout bit")))
    expected = tuple((n+1+j, j) for j in range(n))
    if tuple(result) != expected:
        raise ValueError("The fixed assignment readout order or bit mapping changed")
    return expected


def _schedule(n: int, compiled_ops: object, mapping: tuple,
              check: Callable[[], None] | None) -> tuple[tuple, ...]:
    entries = _sequence(compiled_ops, MAX_COMPILED_OPERATIONS, "Compiled circuit")
    if not entries:
        raise ValueError("A compiled complete draw cannot be empty")
    wires = 3*n
    operations = []
    resets, measurements = [], []
    barriers = 0
    started, terminal = False, False
    for index, entry in enumerate(entries):
        if index % 128 == 0:
            _check(check)
        operation = _operation(entry, wires, n)
        name, qs, cs, _ = operation
        if name == "barrier":
            barriers += 1
        elif name == "reset":
            if started or terminal or qs[0] != len(resets):
                raise ValueError("Only ordered all-wire initial resets are allowed")
            resets.append(qs[0])
        elif name == "measure":
            if not started or len(resets) != wires:
                raise ValueError("Readout requires the complete initialized coherent draw")
            terminal = True
            measurements.append((qs[0], cs[0]))
        else:
            if terminal or len(resets) != wires:
                raise ValueError("The coherent circuit must follow all resets and precede all readouts")
            started = True
        operations.append(operation)
    if tuple(resets) != tuple(range(wires)) or tuple(measurements) != mapping:
        raise ValueError("The complete reset/readout schedule differs from the fixed source contract")
    if barriers != 7:
        raise ValueError("All seven source barriers must be retained without additions")
    return tuple(operations)


def _whole_error(n: int, value: object, rotation_errors: object) -> tuple[float, Fraction]:
    error = _number(value, "Whole-case guarded discrepancy")
    if error < 0:
        raise ValueError("A whole-case discrepancy must be nonnegative")
    entries = _sequence(rotation_errors, ROTATIONS_BY_N[n], "Per-occurrence rotation errors")
    if len(entries) != ROTATIONS_BY_N[n]:
        raise ValueError("Every one of the five/seven RY occurrences needs its guarded error")
    total = REFERENCE_PADDING
    for item in entries:
        guarded = _number(item, "Per-occurrence guarded discrepancy")
        if guarded < MATRIX_GUARD or Fraction.from_float(guarded) > WORD_ERROR_CEILING:
            raise ValueError("A per-occurrence word error lies outside its guarded allowance")
        total += Fraction.from_float(guarded)
    offered = Fraction.from_float(error)
    if total > ETA_BY_N[n] or offered < total or offered > ETA_BY_N[n]:
        raise ValueError("The whole-case error sum plus reference padding does not fit eta")
    return error, total


def _reference(n: int) -> tuple[list[complex], tuple[int, ...], tuple[int, ...], int]:
    """Direct fixed-family equations and Walsh sum, independently constructed."""
    wires = 3*n
    rows = tuple(1 << j for j in range(n)) + ((1 << n)-1,)
    signs = (0,)*n + (1,)
    clauses = (0,) + tuple(1 << j for j in range(n+1))
    denominator_squared = (1 << n)*(n+2)
    denominator = math.sqrt(denominator_squared)
    state = [0j] * (1 << wires)
    numerators, scores = [], []
    for assignment in range(1 << n):
        score = sum(((row & assignment).bit_count() & 1) == sign
                    for row, sign in zip(rows, signs))
        numerator = 0
        for clause in clauses:
            syndrome, sign_bit = 0, 0
            for j, (row, sign) in enumerate(zip(rows, signs)):
                if (clause >> j) & 1:
                    syndrome ^= row
                    sign_bit ^= sign
            character_bit = (syndrome & assignment).bit_count() & 1
            numerator += -1 if sign_bit ^ character_bit else 1
        state[assignment << (n+1)] = complex(numerator/denominator, 0)
        numerators.append(numerator)
        scores.append(score)
    if sum(k*k for k in numerators) != denominator_squared or max(scores) != n:
        raise ValueError("The independent finite reference is inconsistent")
    return state, tuple(numerators), tuple(scores), denominator_squared


def _apply_one(state: list[complex], wire: int, matrix: tuple) -> None:
    stride = 1 << wire
    for start in range(0, len(state), 2*stride):
        for low in range(start, start+stride):
            high = low+stride
            a, b = state[low], state[high]
            state[low] = matrix[0][0]*a + matrix[0][1]*b
            state[high] = matrix[1][0]*a + matrix[1][1]*b


def _apply_cx(state: list[complex], control: int, target: int) -> None:
    control_mask, target_mask = 1 << control, 1 << target
    for low in range(len(state)):
        if low & control_mask and not low & target_mask:
            high = low | target_mask
            state[low], state[high] = state[high], state[low]


def verify_compiled_case(n: object, compiled_ops: object, phase: object,
                         whole_error: object, readoutmap: object, *,
                         rotation_errors: object,
                         check: Callable[[], None] | None = None) -> dict:
    """Verify only the newly compiled complete draw on its fixed zero input.

    ``rotation_errors`` contains each occurrence's independently checked word
    error.  Their exact binary-float rational sum plus decimal reference
    padding must fit eta; a forged smaller total cannot bypass that check.
    Only the compiled circuit evolves.  The reference is constructed from
    equations/Walsh characters, not from the accepted circuit implementation.
    """
    n = _case_n(n)
    check = _callback(check)
    _check(check)
    phase_value = _number(phase, "Compiled global phase")
    error, exact_sum = _whole_error(n, whole_error, rotation_errors)
    mapping = _readout_map(readoutmap, n)
    operations = _schedule(n, compiled_ops, mapping, check)
    wires = 3*n
    dimension = 1 << wires
    if wires > MAX_QUBITS or dimension > MAX_STATE_AMPLITUDES:
        raise ValueError("The compiled-state check exceeds its fixed small domain")
    # The conservative logical-cell envelope also includes probability terms
    # and retained real/imaginary pairs.  It concerns this independent stdlib
    # engine, not SDK/native buffers, exact allocator occupancy, or whole RSS.
    quantum_storage_bound = (12*dimension + 128)*128
    if quantum_storage_bound > MAX_QUANTUM_ARRAY_BYTES:
        raise ValueError("The independent engine exceeds its quantum-array envelope")
    reference, numerators, scores, denominator_squared = _reference(n)
    state = [0j] * dimension
    state[0] = 1+0j
    for index, (name, qs, _, _) in enumerate(operations):
        if index % 128 == 0:
            _check(check)
        if name == "cx":
            _apply_cx(state, qs[0], qs[1])
        elif name in SINGLE_QUBIT_GATES:
            _apply_one(state, qs[0], _matrix(name))
        # The schedule check proved that resets prepare the zero input and
        # all readouts are terminal.  Neither event is passed to a sampler.
    phase_factor = cmath.exp(1j*phase_value)
    for index in range(dimension):
        state[index] *= phase_factor
        if not math.isfinite(state[index].real) or not math.isfinite(state[index].imag):
            raise ValueError("A compiled amplitude became nonfinite")
    norm_squared = math.fsum(z.real*z.real + z.imag*z.imag for z in state)
    if abs(norm_squared-1) > float(REFERENCE_PADDING):
        raise ValueError("The new compiled vector has excessive numerical norm drift")
    distance = _norm(state[j]-reference[j] for j in range(dimension))
    if distance > error:
        raise ValueError("The phase-inclusive full-state discrepancy exceeds the whole-case allowance")
    max_entry_error = max(abs(state[j]-reference[j]) for j in range(dimension))
    mask = ((1 << n)-1) << (n+1)
    all_wires_mask = (1 << wires)-1
    ancilla_mask = all_wires_mask ^ mask
    probabilities = [0.0] * (1 << n)
    mass_parts = [[] for _ in range(1 << n)]
    leakage_parts = []
    for basis, amplitude in enumerate(state):
        probability = amplitude.real*amplitude.real + amplitude.imag*amplitude.imag
        assignment = (basis >> (n+1)) & ((1 << n)-1)
        mass_parts[assignment].append(probability)
        if basis & ancilla_mask:
            leakage_parts.append(probability)
    for assignment in range(1 << n):
        probabilities[assignment] = math.fsum(mass_parts[assignment])
    leakage = math.fsum(leakage_parts)
    if leakage > error*error:
        raise ValueError("Clause/workspace leakage exceeds the whole-state error allowance")
    reference_probabilities = tuple(Fraction(k*k, denominator_squared) for k in numerators)
    total_variation = 0.5*math.fsum(abs(probabilities[j]-float(reference_probabilities[j]))
                                  for j in range(1 << n))
    if total_variation > error:
        raise ValueError("Unconditional assignment total variation exceeds the whole-case allowance")
    optimum_probability = math.fsum(probabilities[j] for j, score in enumerate(scores) if score == n)
    reference_optimum = sum((reference_probabilities[j] for j, score in enumerate(scores) if score == n), Fraction())
    if abs(optimum_probability-float(reference_optimum)) > error:
        raise ValueError("The unconditional optimum event exceeds the whole-case discrepancy allowance")
    expected_objective = math.fsum(probabilities[j]*scores[j] for j in range(1 << n))
    reference_objective = sum((reference_probabilities[j]*scores[j] for j in range(1 << n)), Fraction())
    if abs(expected_objective-float(reference_objective)) > n*error:
        raise ValueError("The objective expectation exceeds its bounded-observable discrepancy")
    _check(check)
    return {
        "status": "passed_new_compiled_state_check",
        "n": n,
        "source_wires": wires,
        "state_amplitudes": dimension,
        "compiled_operation_count": len(operations),
        "compiled_gate_counts": dict(sorted(Counter(op[0] for op in operations).items())),
        "global_phase_hex": phase_value.hex(),
        "phase_normalization_performed": False,
        "whole_error_upper_hex": error.hex(),
        "whole_error_sum_fraction": str(exact_sum),
        "reference_padding_fraction": str(REFERENCE_PADDING),
        "per_draw_eta_fraction": str(ETA_BY_N[n]),
        "full_state_l2_discrepancy": distance,
        "max_state_entry_discrepancy": max_entry_error,
        "compiled_norm_squared": norm_squared,
        "clause_workspace_leakage_probability": leakage,
        "assignment_probability_scope": "unconditional marginal over every clause/workspace value",
        "cleanup_postselection_performed": False,
        "assignment_total_variation": total_variation,
        "optimum_probability": optimum_probability,
        "independent_optimum_probability_fraction": str(reference_optimum),
        "expected_objective": expected_objective,
        "independent_expected_objective_fraction": str(reference_objective),
        "readout_mapping": [{"wire": wire, "classical_bit": bit} for wire, bit in mapping],
        "outcomes": [{
            "assignment": j,
            "display_bits": format(j, f"0{n}b"),
            "objective": scores[j],
            "signed_reference_numerator": numerators[j],
            "reference_denominator_squared": denominator_squared,
            "reference_probability_fraction": str(reference_probabilities[j]),
            "compiled_unconditional_probability": probabilities[j],
            "compiled_clean_workspace_amplitude": _complex_json(state[j << (n+1)]),
        } for j in range(1 << n)],
        "compiled_state_real_imag": [_complex_json(z) for z in state],
        "independent_reference_state_real_imag": [_complex_json(z) for z in reference],
        "quantum_storage_bound_bytes": quantum_storage_bound,
        "quantum_storage_scope": "bounded stdlib vectors, probability terms and retained scalar pairs; native buffers/RSS separate",
        "accepted_ideal_circuit_rerun": False,
        "accepted_decoder_operator_rerun": False,
        "sampled_outcomes": False,
        "dense_full_circuit_operator_constructed": False,
        "formal_interval_certificate": False,
    }
