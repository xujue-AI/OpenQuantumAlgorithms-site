"""Bounded serialized compilation of the two accepted structured DQI circuits.

This module imports only the standard library. It neither imports Qiskit nor
calls a native synthesizer by default. The ordinary driver must inject its
precharged, watchdog-bound native callback and an independent phase-inclusive
word validator after the exact execution source/environment/argv review.

Native words are unknown until that separate authorized pass. A per-word
check is insufficient: every occurrence contributes to the complete-case
guard, with the fixed reference padding, before a compiled case is returned.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable


ANGLE_HEXES = (
    "0x1.0c152382d7366p+0", "0x1.3b2028082e8d5p-1",
    "-0x1.3b2028082e8d5p-1", "0x1.921fb54442d17p-1",
    "-0x1.921fb54442d17p-1", "0x1.dac670561bb4fp-1",
    "0x1.0c152382d7366p-1", "-0x1.0c152382d7366p-1",
)
EPSILON_FRACTION = "1/80000"
EPSILON_HEX = "0x1.a36e2eb1c432dp-17"
WORD_ERROR_FRACTION = "1/20000"
MATRIX_GUARD = 1e-12
REFERENCE_PADDING = Fraction(1, 10_000_000_000)
SINGLE_GATES = frozenset({"h", "x", "y", "z", "s", "sdg", "t", "tdg"})
SOURCE_GATES = frozenset({"h", "x", "z", "t", "tdg", "cx", "ry", "reset", "measure", "barrier"})
DEFAULT_LIMITS = MappingProxyType({
    "source_qubits": 9, "source_operations_per_case": 336,
    "synthesis_native_calls": 8, "unique_ry_targets": 8,
    "synthesized_word_operations": 2000, "unique_word_cache_operations": 16000,
    "compiled_operations_per_case": 20000, "compiled_operations_total": 40000,
})
SOURCE_COUNTS = {
    2: {"barrier": 7, "cx": 38, "h": 12, "measure": 2, "reset": 6,
        "ry": 5, "t": 20, "tdg": 15, "x": 10, "z": 1},
    3: {"barrier": 7, "cx": 120, "h": 39, "measure": 3, "reset": 9,
        "ry": 7, "t": 72, "tdg": 54, "x": 24, "z": 1},
}
SOURCE_OPERATIONS_SHA256 = {
    2: "6fa145372a48336ce4761ae142c16a3da50235f2355268c10823cbb45ed9ca8a",
    3: "3a9a0ddfefd2b4f468100458e784012593a2404691e3f1b53bd318b409b2cfaf",
}


class CompilationError(ValueError):
    """An input, native word or complete compiled case violates the fixed scope."""


class NativeCompilationDisabled(PermissionError):
    """Source availability does not authorize a native compilation pass."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompilationError(message)


def _int(value, name: str, lower: int, upper: int) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool)
             and lower <= value <= upper, f"{name} must be an integer in {lower}..{upper}")
    return value


def _finite(value, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool),
             f"{name} must be a finite number")
    try:
        number = float(value)
    except (ValueError, OverflowError) as error:
        raise CompilationError(f"{name} is outside the finite numeric range") from error
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _hex_float(value, name: str) -> float:
    _require(isinstance(value, str) and value.isascii() and len(value) <= 64,
             f"{name} must be a bounded canonical hexadecimal float")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as error:
        raise CompilationError(f"{name} is not a finite hexadecimal float") from error
    _require(math.isfinite(number) and number.hex() == value,
             f"{name} must preserve exact finite float bits")
    return number


def _limits(value: dict | None) -> dict:
    if value is None:
        return dict(DEFAULT_LIMITS)
    _require(isinstance(value, dict), "Compilation limits must be a dictionary")
    # The complete design has additional limits owned by the driver. This
    # compiler only consumes its named finite subset and cannot enlarge it.
    result = {}
    for name, ceiling in DEFAULT_LIMITS.items():
        _require(name in value, f"Missing compilation limit: {name}")
        result[name] = _int(value[name], name, 1, ceiling)
    return result


def _operation(record, *, wires: int, bits: int, word: bool = False) -> list:
    _require(isinstance(record, (list, tuple)) and len(record) == 4,
             "Operations must be [name, quantum wires, classical bits, parameters]")
    name, quantum, classical, parameters = record
    _require(isinstance(name, str) and name in (SINGLE_GATES if word else SOURCE_GATES),
             "An opaque or unsupported operation was supplied")
    for value, label in ((quantum, "quantum wires"), (classical, "classical bits"),
                         (parameters, "parameters")):
        _require(isinstance(value, (list, tuple)), f"Operation {label} must be bounded sequences")
    _require(len(quantum) <= wires and len(classical) <= bits and len(parameters) <= 1,
             "An operation exceeded its fixed arity")
    q = [_int(value, "wire", 0, wires-1) for value in quantum]
    c = [_int(value, "classical bit", 0, bits-1) for value in classical]
    _require(len(q) == len(set(q)) and len(c) == len(set(c)), "Duplicate operation operands")
    p = list(parameters)
    if name == "barrier":
        _require(q == list(range(wires)) and not c and not p, "Source barriers must cover all wires")
    elif name == "cx":
        _require(len(q) == 2 and not c and not p, "CX requires two distinct wires and no parameters")
    elif name == "measure":
        _require(len(q) == len(c) == 1 and not p, "Readout must preserve its one-wire/one-bit mapping")
    elif name == "ry":
        _require(len(q) == 1 and not c and len(p) == 1, "RY requires one finite binary angle")
        p = [_finite(p[0], "RY angle")]
    else:
        _require(len(q) == 1 and not c and not p, "A primitive operation has unsupported operands")
    if word:
        _require(q == [0], "A native word must act on its sole wire zero")
    return [name, q, c, p]


def _prepare_case(case: dict, limits: dict) -> dict:
    _require(isinstance(case, dict), "Each source case must be a dictionary")
    n = _int(case.get("n"), "n", 2, 3)
    wires = 3*n
    _int(case.get("source_wires"), "source wires", wires, wires)
    _require(wires <= limits["source_qubits"],
             "Only the accepted six/nine-wire cases are supported")
    draws = {2: 5, 3: 3}[n]
    _int(case.get("fixed_complete_draws"), "fixed complete draws", draws, draws)
    ceiling_text = {2: "1/2000", 3: "1/1200"}[n]
    _require(case.get("realization_ceiling_per_draw") == ceiling_text, "Case realization allocation changed")
    records = case.get("operations")
    _require(isinstance(records, (list, tuple)) and 1 <= len(records) <= limits["source_operations_per_case"],
             "The source operation stream exceeded its finite bound")
    operations = [_operation(record, wires=wires, bits=n) for record in records]
    source_digest = hashlib.sha256()
    for record in operations:
        source_digest.update(json.dumps(record, separators=(",", ":"), allow_nan=False).encode()+b"\n")
    _require(source_digest.hexdigest() == SOURCE_OPERATIONS_SHA256[n],
             "The accepted primitive operation content changed")
    counts = dict(sorted(Counter(record[0] for record in operations).items()))
    _require(counts == SOURCE_COUNTS[n], "The accepted source primitive counts changed")
    _require(operations[:wires] == [["reset", [wire], [], []] for wire in range(wires)],
             "Every source wire must have its ordered initial reset")
    _require(all(record[0] != "reset" for record in operations[wires:]), "An extra source reset was supplied")
    mapping = [{"wire": n+1+j, "classical_bit": j} for j in range(n)]
    supplied_mapping = case.get("assignment_readout_mapping")
    _require(isinstance(supplied_mapping, (list, tuple)) and len(supplied_mapping) == n,
             "Accepted readout metadata must contain one entry per assignment bit")
    for row in supplied_mapping:
        _require(isinstance(row, dict) and set(row) == {"wire", "classical_bit"}, "Readout mapping fields changed")
        _int(row["wire"], "readout metadata wire", 0, wires-1)
        _int(row["classical_bit"], "readout metadata bit", 0, n-1)
    _require(list(supplied_mapping) == mapping, "Accepted readout metadata changed")
    measured = [record for record in operations if record[0] == "measure"]
    _require(measured == [["measure", [row["wire"]], [row["classical_bit"]], []] for row in mapping],
             "Source readout order or bit placement changed")
    first_readout = next(i for i, record in enumerate(operations) if record[0] == "measure")
    _require(all(record[0] in {"measure", "barrier"} for record in operations[first_readout:]),
             "A source unitary follows final readout")
    return {"n": n, "source_wires": wires, "fixed_complete_draws": case["fixed_complete_draws"],
            "realization_ceiling_per_draw": ceiling_text, "operations": operations,
            "source_counts": counts, "source_operations_sha256": source_digest.hexdigest(),
            "source_readout_mapping": mapping}


def _targets(records) -> tuple[list[dict], dict[str, dict]]:
    _require(isinstance(records, (list, tuple)) and len(records) == 8, "Exactly eight fixed RY targets are required")
    keys = {"gate", "angle_hex", "epsilon_requested_float_hex", "epsilon_requested_fraction",
            "guarded_frobenius_ceiling_fraction"}
    targets = []
    for row in records:
        _require(isinstance(row, dict) and set(row) == keys, "Native target fields changed")
        target = dict(row)
        _require(target["gate"] == "ry" and target["angle_hex"] in ANGLE_HEXES,
                 "A new rotation target was supplied")
        _hex_float(target["angle_hex"], "RY target")
        _require(target["epsilon_requested_float_hex"] == EPSILON_HEX
                 and target["epsilon_requested_fraction"] == EPSILON_FRACTION
                 and target["guarded_frobenius_ceiling_fraction"] == WORD_ERROR_FRACTION,
                 "The frozen single-attempt synthesis allowance changed")
        targets.append(target)
    by_angle = {row["angle_hex"]: row for row in targets}
    _require(len(by_angle) == 8 and set(by_angle) == set(ANGLE_HEXES), "Rotation targets are duplicated or incomplete")
    return targets, by_angle


def _word(record, limit: int) -> dict:
    _require(isinstance(record, dict) and {"operations", "global_phase_hex"} <= set(record)
             and set(record) <= {"operations", "global_phase_hex", "metadata"}, "Native word fields changed")
    records = record["operations"]
    _require(isinstance(records, (list, tuple)) and 0 <= len(records) <= limit,
             "A returned native word exceeded its operation cap")
    operations = [_operation(row, wires=1, bits=0, word=True) for row in records]
    _hex_float(record["global_phase_hex"], "Native global phase")
    value = {"operations": operations, "global_phase_hex": record["global_phase_hex"]}
    if "metadata" in record:
        _require(isinstance(record["metadata"], dict), "Native word metadata must be a JSON dictionary")
        _bounded_metadata(record["metadata"])
        try:
            metadata = json.dumps(record["metadata"], sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            raise CompilationError("Native word metadata is not finite JSON") from error
        _require(len(metadata.encode()) <= 4096, "Native word metadata exceeded its bound")
        value["metadata"] = json.loads(metadata)
    return value


def _bounded_metadata(value) -> None:
    """Bound returned optional metadata before its JSON representation allocates."""
    pending, nodes, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _require(nodes <= 256 and depth <= 8, "Native word metadata exceeded its node/depth cap")
        if isinstance(item, dict):
            _require(len(item) <= 64, "Native word metadata dictionary exceeded its arity cap")
            for key, child in item.items():
                _require(isinstance(key, str) and len(key) <= 128, "Native metadata keys must be bounded strings")
                text_bytes += len(key.encode())
                pending.append((child, depth+1))
        elif isinstance(item, (list, tuple)):
            _require(len(item) <= 64, "Native word metadata sequence exceeded its arity cap")
            pending.extend((child, depth+1) for child in item)
        elif isinstance(item, str):
            _require(len(item) <= 4096, "Native word metadata text exceeded its cap")
            text_bytes += len(item.encode())
        elif item is None or isinstance(item, bool):
            pass
        elif isinstance(item, int):
            _require(item.bit_length() <= 256, "Native word metadata integer exceeded its bit cap")
        elif isinstance(item, float):
            _require(math.isfinite(item), "Native word metadata float is not finite")
        else:
            raise CompilationError("Native word metadata contains a non-JSON value")
        _require(text_bytes <= 4096, "Native word metadata text budget exceeded its cap")


def routed_cx(control: int, target: int, wires: int) -> list[list]:
    """Route one CX with adjacent three-CX SWAPs, restoring source placement.

    This only constructs primitive instructions. It does not execute a circuit
    or replace the independent Boolean-action verification of the returned block.
    """
    wires = _int(wires, "routing wires", 2, 9)
    control = _int(control, "CX control", 0, wires-1)
    target = _int(target, "CX target", 0, wires-1)
    _require(control != target, "CX operands must be distinct")
    step = 1 if target > control else -1
    swaps = []
    current = control
    while abs(current-target) > 1:
        following = current+step
        swaps.append((current, following))
        current = following
    operations = []
    for a, b in swaps:
        operations.extend([["cx", [a, b], [], []], ["cx", [b, a], [], []], ["cx", [a, b], [], []]])
    operations.append(["cx", [current, target], [], []])
    for a, b in reversed(swaps):
        operations.extend([["cx", [a, b], [], []], ["cx", [b, a], [], []], ["cx", [a, b], [], []]])
    return operations


def _fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _whole_guard(errors: list[float], ceiling: str) -> tuple[float, str]:
    exact = sum((Fraction.from_float(error) for error in errors), Fraction()) + REFERENCE_PADDING
    _require(exact <= Fraction(ceiling), "Whole-case error sum plus reference padding exceeds eta")
    rounded = float(exact)
    if Fraction.from_float(rounded) < exact:
        rounded = math.nextafter(rounded, math.inf)
    _require(Fraction.from_float(rounded) <= Fraction(ceiling), "Rounded whole-case guard exceeds eta")
    return rounded, _fraction_text(exact)


def compile_cases(cases, rotation_targets, *, native_call: Callable | None = None,
                  word_validator: Callable | None = None, check: Callable | None = None,
                  limits: dict | None = None) -> dict:
    """Compile only the accepted pair through explicit guarded callbacks.

    ``native_call(target)`` returns an ordinary serialized word with
    ``operations`` and ``global_phase_hex``. It must be the driver's sole
    precharged native route. ``word_validator(target, word)`` must independently
    reconstruct its phase-inclusive matrix and return a finite
    ``guarded_frobenius_error``. This module never supplies either callback.

    Every source case is preflighted before the first native callback. There is
    one call per exact gate/angle/epsilon/allowance key, no attempt loop, and no
    fallback. Outputs are not validated scientific results until the separate
    compiled-state/routing/physical checks and outer driver completion succeed.
    """
    if not callable(native_call) or not callable(word_validator):
        raise NativeCompilationDisabled("Explicit precharged native and independent validator callbacks are required")
    _require(check is None or callable(check), "Deadline callback must be callable")
    tick = check or (lambda: None)
    bounds = _limits(limits)
    _require(isinstance(cases, (list, tuple)) and len(cases) == 2, "Only the two accepted cases may be compiled")
    prepared = [_prepare_case(case, bounds) for case in cases]
    _require([case["n"] for case in prepared] == [2, 3], "Case order/domain must be the accepted n=2,3 pair")
    targets, by_angle = _targets(rotation_targets)
    needed = {float(op[3][0]).hex() for case in prepared for op in case["operations"] if op[0] == "ry"}
    _require(needed == set(by_angle) and len(needed) <= bounds["unique_ry_targets"], "Source rotations do not match the exact target set")
    _require(len(targets) <= bounds["synthesis_native_calls"], "The known native target count exceeds its cap before entry")
    minimum_counts = []
    for case in prepared:
        minimum = sum((1+6*(abs(op[1][0]-op[1][1])-1) if op[0] == "cx"
                       else 0 if op[0] == "ry" else 1) for op in case["operations"])
        _require(minimum <= bounds["compiled_operations_per_case"], "Known non-RY routing already exceeds the case cap")
        minimum_counts.append(minimum)
    _require(sum(minimum_counts) <= bounds["compiled_operations_total"], "Known non-RY routing already exceeds the pair cap")
    tick()
    words, cache, cache_operations, calls = [], {}, 0, 0
    for target in targets:
        tick()
        key = (target["gate"], target["angle_hex"], target["epsilon_requested_float_hex"],
               target["epsilon_requested_fraction"], target["guarded_frobenius_ceiling_fraction"])
        _require(key not in cache, "An exact synthesis cache key was repeated")
        _require(calls < bounds["synthesis_native_calls"], "Synthesis native entry cap exceeded before invocation")
        calls += 1  # Precharge before entering the sole callback, including a failure.
        raw_word = native_call(dict(target))
        tick()
        word = _word(raw_word, bounds["synthesized_word_operations"])
        cache_operations += len(word["operations"])
        _require(cache_operations <= bounds["unique_word_cache_operations"], "The unique native word cache exceeded its cap")
        report = word_validator(dict(target), deepcopy(word))
        _require(isinstance(report, dict) and "guarded_frobenius_error" in report,
                 "Independent word validation did not return its guarded error")
        error = _finite(report["guarded_frobenius_error"], "Independent guarded Frobenius error")
        _require(MATRIX_GUARD <= error and Fraction.from_float(error) <= Fraction(WORD_ERROR_FRACTION),
                 "A native word failed its independent whole-matrix allowance")
        word_id = hashlib.sha256(json.dumps({"target": target, "word": word}, sort_keys=True,
                                           separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        record = {"word_id": word_id, "target": dict(target), **word,
                  "guarded_frobenius_error": error, "validation": deepcopy(report)}
        cache[key] = record
        words.append(record)
    _require(calls == 8, "The fixed native target set was not completed")
    compiled_cases, total_operations = [], 0
    for case in prepared:
        tick()
        operations, occurrences, phases, routing = [], [], [], []
        errors = []
        def append(record):
            _require(len(operations) < bounds["compiled_operations_per_case"], "Compiled case operation cap exceeded")
            operations.append(record)
        for source_index, source in enumerate(case["operations"]):
            tick()
            name, q, c, parameters = source
            start = len(operations)
            if name == "ry":
                target = by_angle[float(parameters[0]).hex()]
                key = (target["gate"], target["angle_hex"], target["epsilon_requested_float_hex"],
                       target["epsilon_requested_fraction"], target["guarded_frobenius_ceiling_fraction"])
                word = cache[key]
                for gate, _, _, _ in word["operations"]:
                    append([gate, [q[0]], [], []])
                phase = _hex_float(word["global_phase_hex"], "Cached global phase")
                phases.append(phase)
                errors.append(word["guarded_frobenius_error"])
                occurrences.append({"source_operation_index": source_index, "wire": q[0],
                                    "angle_hex": target["angle_hex"], "word_id": word["word_id"],
                                    "global_phase_hex": word["global_phase_hex"],
                                    "guarded_frobenius_error": word["guarded_frobenius_error"],
                                    "compiled_start": start, "compiled_stop": len(operations)})
            elif name == "cx":
                block = routed_cx(q[0], q[1], case["source_wires"])
                for record in block:
                    append(record)
                routing.append({"source_operation_index": source_index, "wires": case["source_wires"],
                                "control": q[0], "target": q[1],
                                "compiled_start": start, "compiled_stop": len(operations),
                                "operations": deepcopy(block)})
            else:
                append(deepcopy(source))
        _require(len(errors) == {2: 5, 3: 7}[case["n"]], "A source RY occurrence was lost or duplicated")
        whole_error, exact_error = _whole_guard(errors, case["realization_ceiling_per_draw"])
        phase = math.fsum(phases)
        _require(math.isfinite(phase), "Compiled global phase is not finite")
        counts = dict(sorted(Counter(op[0] for op in operations).items()))
        total_operations += len(operations)
        _require(total_operations <= bounds["compiled_operations_total"], "Complete compiled pair operation cap exceeded")
        expected_events = [op for op in case["operations"] if op[0] in {"reset", "measure", "barrier"}]
        actual_events = [op for op in operations if op[0] in {"reset", "measure", "barrier"}]
        _require(actual_events == expected_events, "Compilation changed source reset/readout/barrier order or wires")
        _require(all(abs(op[1][0]-op[1][1]) == 1 for op in operations if op[0] == "cx"),
                 "A non-neighbor CX remains after deterministic routing")
        compiled_cases.append({**case, "source_operations": case["operations"], "operations": operations,
                               "global_phase_hex": phase.hex(), "phase_contributions": [value.hex() for value in phases],
                               "whole_error": whole_error, "whole_error_exact_fraction": exact_error,
                               "reference_padding_fraction": _fraction_text(REFERENCE_PADDING),
                               "rotation_occurrences": occurrences, "routing_blocks": routing,
                               "compiled_counts": counts, "actual_t_tdg_per_draw": counts.get("t", 0)+counts.get("tdg", 0),
                               "actual_t_tdg_whole_task": case["fixed_complete_draws"]*(counts.get("t", 0)+counts.get("tdg", 0)),
                               "validation_status": "words_checked_routing_and_compiled_state_pending"})
    tick()
    return {"schema": "openquantum.dqi.structured_compilation.v1", "cases": compiled_cases,
            "words": words, "synthesis_native_calls": calls, "unique_word_cache_operations": cache_operations,
            "compiled_operations_total": total_operations, "native_synthesis_retries": 0,
            "formal_interval_certificate": False, "completion_scope": "compilation return only; independent routing/state/native/outer completion still required"}
