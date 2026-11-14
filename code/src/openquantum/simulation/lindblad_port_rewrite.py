# SPDX-License-Identifier: MIT
"""Frozen source-only recipe and separately gated physical-wire-zero rewrite.

The saved eight native words are reused without synthesis.  A word is moved
only over explicitly initialized wires, then its adjacent SWAP chain is
undone before the next source instruction.  The authoritative whole-circuit
phase is carried once; individual word phases are not added to bare tokens.
Recipe counts are projections until a separately authorized candidate is
materialized and checked.  Original project code; Qiskit is Apache-2.0.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
from types import MethodType
from typing import Iterator

from qiskit import QuantumCircuit
from qiskit.circuit import Barrier, Measure, Reset
from qiskit.circuit.library import (CXGate, HGate, SGate, SdgGate, TGate,
                                   TdgGate, XGate, YGate, ZGate)

if __package__:
    from . import lindblad_port_words as words
else:
    import lindblad_port_words as words

ALGORITHM_VERSION = "literal-n4-initialized-path-port-zero-v1"
WIDTH = 8
BASELINE_OPERATIONS = 1_677_906
CANDIDATE_OPERATION_CEILING = 1_926_042
ADDED_CX_CEILING = 248_136
BASELINE_PHASE = 6.28318530717827
BASELINE_COUNTS = {"h": 661278, "t": 642284, "s": 349839,
                   "cx": 14130, "x": 5908, "reset": 1481,
                   "measure": 8, "barrier": 2978}
BASELINE_T_WIRES = [57280, 58670, 102397, 104355, 101923, 103392, 56849, 57418]
BASELINE_READOUT = [6, 1, 2, 3, 7, 4, 0, 5]
_IMPORTED_REWRITE_SOURCE_SHA256 = words.bounded_file_sha256(Path(__file__), maximum=1024*1024)
_GATE_CLASSES = {"cx": CXGate, "h": HGate, "s": SGate, "sdg": SdgGate,
                 "t": TGate, "tdg": TdgGate, "x": XGate, "y": YGate,
                 "z": ZGate, "reset": Reset, "measure": Measure,
                 "barrier": Barrier}
# Standard mutable classes and Qiskit's internal singleton classes are valid;
# a subclass advertising a standard base_class is not a primitive identity.
_CANONICAL_TYPES = {name: {base, type(base(WIDTH) if name == "barrier" else base())}
                    for name, base in _GATE_CLASSES.items()}
_MATRIX_METHODS = {name: base.to_matrix for name, base in _GATE_CLASSES.items()
                   if name not in {"reset", "measure", "barrier"}}
_ARRAY_METHODS = {name: getattr(base, "__array__", None) for name, base in _GATE_CLASSES.items()
                 if name in _MATRIX_METHODS}


@dataclass(frozen=True)
class PortRewriteLimits:
    """Mandatory independent bounds; historical project limits stay frozen."""

    max_qubits: int
    baseline_operations: int
    routed_operations: int
    candidate_operations: int
    added_cx: int
    recipe_bytes: int = 8 * 1024 * 1024

    def __post_init__(self):
        if type(self.max_qubits) is not int or self.max_qubits != WIDTH:
            raise ValueError("The port recipe requires exactly eight physical wires")
        ceilings = {"baseline_operations": 2_000_000, "routed_operations": 100_000,
                    "candidate_operations": CANDIDATE_OPERATION_CEILING,
                    "added_cx": ADDED_CX_CEILING, "recipe_bytes": 8*1024*1024}
        for name, ceiling in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise ValueError(f"{name} must be an integer in 1..{ceiling}")


def literal_limits() -> PortRewriteLimits:
    return PortRewriteLimits(8, 2_000_000, 100_000, 1_926_042, 248_136)


def _limits(value) -> PortRewriteLimits:
    if type(value) is not PortRewriteLimits:
        raise ValueError("limits must be an explicit PortRewriteLimits")
    # Validate shallow scalar values before any dataclass deep-copy can run.
    return PortRewriteLimits(value.max_qubits, value.baseline_operations,
                             value.routed_operations, value.candidate_operations,
                             value.added_cx, value.recipe_bytes)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False)+"\n").encode("utf8")


def _bounded_record_bytes(value, maximum: int) -> bytes:
    """Stop untrusted record serialization at the byte cap, before allocation."""
    remaining = 1_000_000
    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 16:
            raise ValueError("Recipe record exceeds its bounded node/depth cap")
        if item is None or type(item) is bool:
            return
        if type(item) is int and item.bit_length() <= 1024:
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is str and len(item) <= 8192:
            return
        if type(item) is list and len(item) <= 32_768:
            for child in item:
                visit(child, depth+1)
            return
        if type(item) is dict and len(item) <= 4096:
            for key, child in item.items():
                if type(key) is not str or len(key) > 128:
                    raise ValueError("Recipe keys must be bounded concrete strings")
                visit(child, depth+1)
            return
        raise ValueError("Recipe contains an unsupported or oversized JSON value")
    visit(value, 0)
    pieces, total = [], 1
    try:
        for fragment in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
            raw = fragment.encode("utf8")
            total += len(raw)
            if total > maximum:
                raise ValueError("Recipe exceeds its explicit byte cap")
            pieces.append(raw)
    except (TypeError, RecursionError) as error:
        raise ValueError("Recipe record must contain bounded JSON values") from error
    return b"".join(pieces)+b"\n"


@dataclass(frozen=True)
class FrozenPortRecipe:
    canonical_json: bytes = field(repr=False)
    recipe_sha256: str

    def as_record(self) -> dict:
        value = json.loads(self.canonical_json)
        value["recipe_sha256"] = self.recipe_sha256
        return value

    @classmethod
    def from_record(cls, record: dict, *, limits: PortRewriteLimits):
        limits = _limits(limits)
        if type(record) is not dict or len(record) > 4096:
            raise ValueError("Recipe record must be a bounded concrete JSON object")
        value = dict(record)
        digest = value.pop("recipe_sha256", None)
        raw = _bounded_record_bytes(value, limits.recipe_bytes)
        if type(digest) is not str or len(digest) != 64 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Recipe canonical digest differs from its record")
        return cls(raw, digest)


@dataclass
class PortRewriteCandidate:
    circuit: QuantumCircuit
    report: dict


def _source_sha256() -> dict:
    rewrite_hash = words.bounded_file_sha256(Path(__file__), maximum=1024*1024)
    if rewrite_hash != _IMPORTED_REWRITE_SOURCE_SHA256:
        raise ValueError("Port rewrite source changed since this module was imported")
    return {"lindblad_port_rewrite.py": rewrite_hash,
            "lindblad_port_words.py": words.bounded_file_sha256(Path(words.__file__), maximum=1024*1024)}


def _primitive(value, *, rotations: bool = False) -> None:
    if type(value) is not words.Primitive or type(value.qubits) is not tuple or type(value.clbits) is not tuple or type(value.params) is not tuple:
        raise ValueError("Instructions must be exact immutable Primitive tokens")
    allowed = words.CLIFFORD_NAMES | {"cx", "reset", "measure", "barrier"}
    if rotations:
        allowed |= words.ROTATION_NAMES
    if type(value.name) is not str or value.name not in allowed:
        raise ValueError("Unsupported or hidden primitive instruction")
    if any(type(w) is not int or not 0 <= w < WIDTH for w in value.qubits) or len(set(value.qubits)) != len(value.qubits):
        raise ValueError("Primitive quantum operands must be distinct eight-wire indices")
    arity = WIDTH if value.name == "barrier" else 2 if value.name == "cx" else 1
    if len(value.qubits) != arity:
        raise ValueError("Primitive has inconsistent fixed-schedule arity")
    if value.name == "barrier" and set(value.qubits) != set(range(WIDTH)):
        raise ValueError("A preserved barrier must cover the complete physical line")
    if value.name == "cx" and abs(value.qubits[0]-value.qubits[1]) != 1:
        raise ValueError("Every CNOT must join physical neighbors")
    if value.name == "measure":
        if len(value.clbits) != 1 or type(value.clbits[0]) is not int or not 0 <= value.clbits[0] < WIDTH:
            raise ValueError("Measurement must target one of eight classical bits")
    elif value.clbits:
        raise ValueError("Only readout can have a classical operand")
    if value.name in words.ROTATION_NAMES:
        if len(value.params) != 1 or type(value.params[0]) is not float or not math.isfinite(value.params[0]):
            raise ValueError("Rotation must have one finite binary64 parameter")
    elif value.params:
        raise ValueError("Bare Clifford+T/event tokens have no parameters")


def _catalog(catalog) -> dict:
    if type(catalog) is not tuple or not 1 <= len(catalog) <= 8:
        raise ValueError("A bounded tuple of recovered words is required")
    result, keys = {}, set()
    for word in catalog:
        if type(word) is not words.PhasedWord or type(word.word_id) is not str or word.word_id in result or word.key in keys:
            raise ValueError("Word IDs and rotation keys must be unique recovered identities")
        if type(word.tokens) is not tuple or not 1 <= len(word.tokens) <= 4096:
            raise ValueError("A recovered word must be a bounded nonempty tuple")
        if type(word.phase) is not float or not math.isfinite(word.phase):
            raise ValueError("The individual recorded word phase must be finite")
        for token in word.tokens:
            _primitive(token)
            if token.name not in words.CLIFFORD_NAMES or token.qubits != (0,):
                raise ValueError("Word tokens must be bare one-wire Clifford+T at normalized wire zero")
        result[word.word_id] = word
        keys.add(word.key)
    return result


def _schedule(wire: int, eligible: bool) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    forward = tuple((i, i+1) for i in range(wire-1, -1, -1)) if eligible else ()
    return forward, tuple(reversed(forward))


def _plan_segments(segments, *, catalog: tuple, limits: PortRewriteLimits) -> dict:
    """Compact algebraic plan; tiny fixtures are not literal n4 attestations."""
    limits, lookup = _limits(limits), _catalog(catalog)
    initialized, measurements, rows, events = set(), [], [], []
    baseline, planned, word_counts = Counter(), Counter(), {}
    baseline_t, planned_t = [0]*WIDTH, [0]*WIDTH
    offset = routed_count = added = eligible_count = fallback_count = 0
    output_started = False
    event_digest, nonrotation_digest = hashlib.sha256(), hashlib.sha256()
    for segment in segments:
        if type(segment) is not words.VerifiedSegment or type(segment.routed_index) is not int or segment.routed_index != routed_count or type(segment.ct_start) is not int or segment.ct_start != offset:
            raise ValueError("Segments must preserve every routed index and contiguous CT range")
        routed_count += 1
        if routed_count > limits.routed_operations:
            raise ValueError("Routed reference exceeds its explicit operation cap")
        token = segment.routed
        _primitive(token, rotations=True)
        if output_started and token.name not in {"measure", "barrier"}:
            raise ValueError("No quantum work is allowed after final readout begins")
        if token.name == "reset":
            initialized.add(token.qubits[0])
        elif token.name != "barrier" and not set(token.qubits) <= initialized:
            raise ValueError("A source instruction uses a wire before explicit initialization")
        if token.name == "measure":
            output_started = True
            measurements.append((token.qubits[0], token.clbits[0]))
        if token.name in {"reset", "measure", "barrier"}:
            events.append({"routed_index": segment.routed_index, "ct_start": offset,
                           "primitive": token.as_record()})
        if token.name in {"reset", "measure"}:
            event_digest.update(_json_bytes(token.as_record()))
        if type(segment.tokens) is not tuple or type(segment.ct_operations) is not int or not segment.tokens or segment.ct_operations != len(segment.tokens):
            raise ValueError("Segment CT length must match its complete immutable token tuple")
        if token.name in words.ROTATION_NAMES:
            word = lookup.get(segment.word_id)
            if word is None or word.key != token.rotation_key or segment.tokens != word.tokens:
                raise ValueError("Rotation segment differs from its recovered actual word")
            wire = token.qubits[0]
            eligible = set(range(wire+1)) <= initialized
            forward, reverse = _schedule(wire, eligible)
            extra = 6*wire if eligible else 0
            target = 0 if eligible else wire
            if word.word_id not in word_counts:
                word_counts[word.word_id] = Counter(p.name for p in word.tokens)
            counts = word_counts[word.word_id]
            baseline.update(counts)
            planned.update(counts)
            t_count = counts["t"]+counts["tdg"]
            baseline_t[wire] += t_count
            planned_t[target] += t_count
            added += extra
            eligible_count += int(eligible)
            fallback_count += int(not eligible)
            rows.append({"routed_index": segment.routed_index, "ct_start": offset,
                         "ct_operations": segment.ct_operations, "word_id": word.word_id,
                         "rotation_key": list(word.key), "original_wire": wire,
                         "initialized_mask": sum(1 << i for i in initialized),
                         "eligible": eligible, "decision": "path_initialized" if eligible else "uninitialized_path_fallback",
                         "target_wire": target, "forward_swap_edges": [list(e) for e in forward],
                         "reverse_swap_edges": [list(e) for e in reverse], "added_cx": extra})
        else:
            if segment.word_id is not None or segment.tokens != (token,) or segment.ct_operations != 1:
                raise ValueError("Every nonrotation source instruction must pass unchanged")
            baseline[token.name] += 1
            planned[token.name] += 1
            nonrotation_digest.update(_json_bytes(token.as_record()))
            if token.name in {"t", "tdg"}:
                baseline_t[token.qubits[0]] += 1
                planned_t[token.qubits[0]] += 1
        offset += segment.ct_operations
        if offset > limits.baseline_operations or added > limits.added_cx or offset+added > limits.candidate_operations:
            raise ValueError("Whole port plan exceeds a predeclared operation or added-CX cap")
    if initialized != set(range(WIDTH)) or len(measurements) != WIDTH or [c for _, c in measurements] != list(range(WIDTH)) or set(q for q, _ in measurements) != set(range(WIDTH)):
        raise ValueError("Complete final readout must initialize/measure every wire in c0..c7 order")
    if added:
        planned["cx"] += added
    return {"rotation_blocks": rows, "source_event_records": events,
            "baseline_gate_counts": dict(sorted(baseline.items())),
            "planned_gate_counts": dict(sorted(planned.items())),
            "baseline_t_wire_counts": baseline_t, "planned_t_wire_counts": planned_t,
            "routed_operations": routed_count, "baseline_operations": offset,
            "planned_operations": offset+added, "added_cx": added,
            "eligible_rotation_blocks": eligible_count, "fallback_rotation_blocks": fallback_count,
            "final_measurement_wires": [q for q, _ in measurements],
            "final_measurement_clbits": [c for _, c in measurements],
            "event_sha256": event_digest.hexdigest(),
            "nonrotation_sequence_sha256": nonrotation_digest.hexdigest(),
            "counts_scope": "source-only structural projection; no candidate emitted or executed"}


def _matched_contract(recovered, word_limits) -> dict:
    baseline = words.baseline_record(recovered, limits=word_limits)
    return {"case_id": "lindblad-n4-r211-Z0", "task": baseline["task"],
            "selected_steps": baseline["selected_steps"], "input_states": ["+i", "+", "+", "+"],
            "output_clbits": baseline["output_clbits"], "score_values": baseline["score_values"],
            "error_contract": baseline["error_contract"],
            "complete_workload": baseline["complete_workload"],
            "family_allocations": baseline["family_allocations"],
            "frozen_baseline_prepared_sha256": hashlib.sha256(_json_bytes(baseline)).hexdigest(),
            "hardware": {"bidirectional_line_wires": 8, "reserved_logical_patches": 31,
                         "workspace_startups_per_trial": 23, "controller_error_per_decision": 1e-15,
                         "physical_error_profiles": [1e-3, 1e-4], "odd_distances": list(range(3, 50, 2)),
                         "factory_retry_cap": 16, "whole_device_serialization": True,
                         "factory_policy": "one just-in-time factory; unchanged frozen timing/geometry/interface macros",
                         "frozen_instrument_resources_sha256": "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"}}


def build_recipe(recovered: words.RecoveredWords, *, limits: PortRewriteLimits,
                 word_limits: words.WordRecoveryLimits) -> FrozenPortRecipe:
    """Freeze all actual source correspondences without emitting the candidate."""
    limits = _limits(limits)
    plan = _plan_segments(words.iter_verified_segments(recovered, limits=word_limits),
                          catalog=recovered.words, limits=limits)
    if (plan["baseline_operations"] != BASELINE_OPERATIONS or plan["routed_operations"] != 24510 or
            plan["baseline_gate_counts"] != BASELINE_COUNTS or plan["baseline_t_wire_counts"] != BASELINE_T_WIRES or
            plan["final_measurement_wires"] != BASELINE_READOUT or len(plan["rotation_blocks"]) != 5908 or
            recovered.baseline_global_phase.hex() != BASELINE_PHASE.hex() or
            plan["event_sha256"] != recovered.summary["event_sha256"]):
        raise ValueError("The complete recipe does not match the literal frozen n4 baseline")
    needed = {"manifest.json", "results.json", "source-trial.qasm",
              "routed-rotation-trial.qasm", "clifford-t-trial.qasm"}
    if not needed <= set(recovered.source_artifacts):
        raise ValueError("Recipe must bind all three baseline QASMs/results/manifest")
    payload = {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION,
               "status": "frozen_source_only_port_recipe", "source_artifacts": recovered.source_artifacts,
               "catalog_sha256": recovered.recovery_sha256, "source_sha256": _source_sha256(),
               "word_identities": [{"word_id": w.word_id, "rotation_key": list(w.key),
                                    "phase_binary64_hex": w.phase.hex(),
                                    "sequence_sha256": words.sequence_sha256(w.tokens),
                                    "phase_aware_qasm_sha256": hashlib.sha256(words.export_word_qasm(w.tokens, w.phase)).hexdigest()}
                                   for w in recovered.words],
               "global_phase_binary64_hex": recovered.baseline_global_phase.hex(),
               "phase_rule": "bare words plus authoritative baseline global phase exactly once",
               "matched_contract": _matched_contract(recovered, word_limits),
               "limits": asdict(limits), "word_limits": asdict(word_limits),
               "constants": {"width": WIDTH, "baseline_operations": BASELINE_OPERATIONS,
                             "added_cx_ceiling": ADDED_CX_CEILING,
                             "candidate_operation_ceiling": CANDIDATE_OPERATION_CEILING,
                             "swap_cx_per_edge": 3, "added_cx_per_eligible_wire_index": 6},
               "plan": plan, "candidate_materialized": False, "full_density_executed": False,
               "native_physical_queries": 0, "new_synthesis_calls": 0,
               "hardware_workload_executed": False, "physical_feasibility_assessed": False}
    raw = _json_bytes(payload)
    if len(raw) > limits.recipe_bytes:
        raise ValueError("Recipe exceeds its explicit byte cap")
    return FrozenPortRecipe(raw, hashlib.sha256(raw).hexdigest())


def require_verified_recipe(recipe: FrozenPortRecipe, recovered: words.RecoveredWords, *,
                            limits: PortRewriteLimits, word_limits: words.WordRecoveryLimits) -> dict:
    limits = _limits(limits)
    if type(recipe) is not FrozenPortRecipe or type(recipe.canonical_json) is not bytes or len(recipe.canonical_json) > limits.recipe_bytes:
        raise ValueError("A bounded immutable FrozenPortRecipe is required")
    if hashlib.sha256(recipe.canonical_json).hexdigest() != recipe.recipe_sha256:
        raise ValueError("Frozen recipe digest changed")
    fresh = build_recipe(recovered, limits=limits, word_limits=word_limits)
    if recipe.canonical_json != fresh.canonical_json or recipe.recipe_sha256 != fresh.recipe_sha256:
        raise ValueError("Frozen recipe differs from the verified current source-only plan")
    return recipe.as_record()


def _swap_tokens(edges):
    for a, b in edges:
        yield words.Primitive("cx", (a, b))
        yield words.Primitive("cx", (b, a))
        yield words.Primitive("cx", (a, b))


def _emit_segments(segments, blocks, *, limits: PortRewriteLimits) -> Iterator[words.Primitive]:
    """Private bounded emitter used only after a verified plan, or tiny fixtures."""
    limits = _limits(limits)
    count, block_index = 0, 0
    for segment in segments:
        if segment.word_id is None:
            emitted = segment.tokens
            extra = 0
        else:
            if block_index >= len(blocks):
                raise ValueError("Emission has an unplanned rotation block")
            row = blocks[block_index]
            block_index += 1
            if (row["routed_index"] != segment.routed_index or row["ct_start"] != segment.ct_start or
                    row["word_id"] != segment.word_id or row["ct_operations"] != segment.ct_operations):
                raise ValueError("Emission segment differs from its frozen source range")
            target, extra = row["target_wire"], row["added_cx"]
            def rotation_tokens():
                yield from _swap_tokens(row["forward_swap_edges"])
                for token in segment.tokens:
                    yield words.Primitive(token.name, (target,))
                yield from _swap_tokens(row["reverse_swap_edges"])
            emitted = rotation_tokens()
        if count+segment.ct_operations+extra > limits.candidate_operations:
            raise ValueError("Next complete segment exceeds the candidate operation cap")
        for token in emitted:
            count += 1
            if count > limits.candidate_operations:
                raise ValueError("Candidate stream exceeds its explicit operation cap")
            yield token
    if block_index != len(blocks):
        raise ValueError("Emission omitted a planned rotation block")


def iter_port_rewrite(recipe: FrozenPortRecipe, recovered: words.RecoveredWords, *,
                      limits: PortRewriteLimits, word_limits: words.WordRecoveryLimits,
                      allow_full_materialization: bool = False) -> Iterator[words.Primitive]:
    """Emit actual candidate instructions only with an explicit execution gate."""
    if allow_full_materialization is not True:
        raise ValueError("Full candidate materialization requires explicit authorization after recipe review")
    record = require_verified_recipe(recipe, recovered, limits=limits, word_limits=word_limits)
    yield from _emit_segments(words.iter_verified_segments(recovered, limits=word_limits),
                             record["plan"]["rotation_blocks"], limits=limits)


def _actual_summary(tokens, *, limits: PortRewriteLimits) -> dict:
    limits = _limits(limits)
    counts, t_wires, readout, initialized = Counter(), [0]*WIDTH, [], set()
    digest, event_digest = hashlib.sha256(), hashlib.sha256()
    count, output_started = 0, False
    for token in tokens:
        _primitive(token)
        count += 1
        if count > limits.candidate_operations:
            raise ValueError("Actual candidate exceeds its explicit operation cap")
        if output_started and token.name not in {"measure", "barrier"}:
            raise ValueError("Actual candidate has quantum work after readout begins")
        if token.name == "reset":
            initialized.add(token.qubits[0])
        elif token.name != "barrier" and not set(token.qubits) <= initialized:
            raise ValueError("Actual candidate uses an uninitialized physical wire")
        counts[token.name] += 1
        if token.name in {"t", "tdg"}:
            t_wires[token.qubits[0]] += 1
        if token.name in {"reset", "measure"}:
            event_digest.update(_json_bytes(token.as_record()))
        if token.name == "measure":
            output_started = True
            readout.append((token.qubits[0], token.clbits[0]))
        digest.update(_json_bytes(token.as_record()))
    if initialized != set(range(WIDTH)) or len(readout) != WIDTH or [c for _, c in readout] != list(range(WIDTH)) or set(q for q, _ in readout) != set(range(WIDTH)):
        raise ValueError("Actual candidate must preserve complete eight-wire final readout")
    return {"actual_operations": count, "gate_counts": dict(sorted(counts.items())),
            "t_wire_counts": t_wires, "event_sha256": event_digest.hexdigest(),
            "final_measurement_wires": [q for q, _ in readout],
            "final_measurement_clbits": [c for _, c in readout],
            "primitive_stream_sha256": digest.hexdigest()}


def _candidate_report(record, actual):
    plan = record["plan"]
    if (actual["actual_operations"] != plan["planned_operations"] or actual["gate_counts"] != plan["planned_gate_counts"] or
            actual["t_wire_counts"] != plan["planned_t_wire_counts"] or actual["event_sha256"] != plan["event_sha256"] or
            actual["final_measurement_wires"] != plan["final_measurement_wires"] or actual["final_measurement_clbits"] != plan["final_measurement_clbits"]):
        raise ValueError("Actual emitted counts/events/readout differ from the frozen source-only plan")
    return {"schema_version": 1, "algorithm_version": ALGORITHM_VERSION,
            "status": "completed_materialized_port_candidate", "recipe_sha256": record["recipe_sha256"],
            "catalog_sha256": record["catalog_sha256"], "source_sha256": record["source_sha256"],
            "source_artifacts": record["source_artifacts"], "global_phase_binary64_hex": record["global_phase_binary64_hex"],
            "phase_rule": record["phase_rule"], "limits": record["limits"], **actual,
            "added_cx": actual["gate_counts"].get("cx", 0)-plan["baseline_gate_counts"].get("cx", 0),
            "candidate_materialized": True, "full_density_executed": False, "new_synthesis_calls": 0,
            "native_physical_queries": 0, "physical_feasibility_assessed": False,
            "hardware_workload_executed": False}


def materialize_candidate(recipe: FrozenPortRecipe, recovered: words.RecoveredWords, *,
                          limits: PortRewriteLimits, word_limits: words.WordRecoveryLimits,
                          allow_full_materialization: bool = False) -> PortRewriteCandidate:
    if allow_full_materialization is not True:
        raise ValueError("Full candidate materialization requires explicit authorization after recipe review")
    record = require_verified_recipe(recipe, recovered, limits=limits, word_limits=word_limits)
    circuit = QuantumCircuit(WIDTH, WIDTH)
    circuit.global_phase = float.fromhex(record["global_phase_binary64_hex"])
    def append_and_observe():
        for token in _emit_segments(words.iter_verified_segments(recovered, limits=word_limits),
                                    record["plan"]["rotation_blocks"], limits=limits):
            if token.name == "measure":
                circuit.measure(token.qubits[0], token.clbits[0])
            elif token.name == "barrier":
                circuit.barrier(*token.qubits)
            else:
                getattr(circuit, token.name)(*token.qubits)
            yield token
    actual = _actual_summary(append_and_observe(), limits=limits)
    if float(circuit.global_phase).hex() != record["global_phase_binary64_hex"]:
        raise ValueError("Candidate whole-circuit phase was changed during construction")
    return PortRewriteCandidate(circuit, _candidate_report(record, actual))


def _circuit_tokens(circuit):
    if type(circuit) is not QuantumCircuit or circuit.num_qubits != WIDTH or circuit.num_clbits != WIDTH or circuit.num_parameters:
        raise ValueError("Candidate must be a bound exact eight-wire QuantumCircuit")
    for item in circuit.data:
        operation = item.operation
        if (operation.name not in _GATE_CLASSES or operation.base_class is not _GATE_CLASSES[operation.name] or
                type(operation) not in _CANONICAL_TYPES[operation.name] or
                getattr(operation, "condition", None) is not None or operation.params or
                (operation.name == "cx" and getattr(operation, "ctrl_state", 1) != 1)):
            raise ValueError("Candidate must contain canonical flat Clifford+T/event primitives")
        if operation.num_qubits != len(item.qubits) or operation.num_clbits != len(item.clbits):
            raise ValueError("Candidate primitive has inconsistent declared operand arity")
        if operation.name in _MATRIX_METHODS:
            method = operation.to_matrix
            array = getattr(operation, "__array__", None)
            expected_array = _ARRAY_METHODS[operation.name]
            array_is_canonical = (array is None if expected_array is None else
                                  type(array) is MethodType and
                                  getattr(array, "__self__", None) is operation and
                                  getattr(array, "__func__", None) is expected_array)
            if (type(method) is not MethodType or getattr(method, "__self__", None) is not operation or
                    getattr(method, "__func__", None) is not _MATRIX_METHODS[operation.name] or
                    getattr(type(operation), "__array__", None) is not expected_array or
                    not array_is_canonical):
                raise ValueError("Candidate primitive executable matrix methods must remain canonical")
        token = words.Primitive(operation.name, tuple(circuit.find_bit(q).index for q in item.qubits),
                                tuple(circuit.find_bit(c).index for c in item.clbits))
        _primitive(token)
        yield token


def require_verified_candidate(candidate: PortRewriteCandidate, recipe: FrozenPortRecipe,
                               recovered: words.RecoveredWords, *, limits: PortRewriteLimits,
                               word_limits: words.WordRecoveryLimits) -> dict:
    """Check the full actual stream/report; supplies no density or feasibility claim."""
    if type(candidate) is not PortRewriteCandidate or type(candidate.report) is not dict:
        raise ValueError("A completed PortRewriteCandidate is required")
    if type(candidate.circuit) is not QuantumCircuit:
        raise ValueError("Candidate must contain an exact QuantumCircuit")
    record = require_verified_recipe(recipe, recovered, limits=limits, word_limits=word_limits)
    if not math.isfinite(float(candidate.circuit.global_phase)) or float(candidate.circuit.global_phase).hex() != record["global_phase_binary64_hex"]:
        raise ValueError("Candidate must carry the authoritative whole phase exactly once")
    expected = _emit_segments(words.iter_verified_segments(recovered, limits=word_limits),
                              record["plan"]["rotation_blocks"], limits=limits)
    def compared():
        for actual, wanted in zip_longest(_circuit_tokens(candidate.circuit), expected):
            if actual != wanted:
                raise ValueError("Actual candidate instruction differs from the exact frozen rewrite stream")
            yield actual
    report = _candidate_report(record, _actual_summary(compared(), limits=limits))
    if candidate.report != report:
        raise ValueError("Candidate report differs from its verified actual stream/current recipe")
    return json.loads(_json_bytes(report))
