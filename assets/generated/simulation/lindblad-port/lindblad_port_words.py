# SPDX-License-Identifier: MIT
"""Recover the actual phased n4 rotation words from saved flat QASM streams.

Only the frozen completed n4 artifacts are inputs.  This module neither calls
synthesis nor simulates or estimates a ported candidate.  It checks every
saved occurrence, exports the recorded phase explicitly, and compares complete
two-by-two operators rather than comparing only probabilities or ignoring
global phase.  Original project code; Qiskit is Apache-2.0.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterator
import weakref

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

N4_MANIFEST_SHA256 = "5e78c5deac53d8106539ae75c5edbab4027ead7439c2bd4a2e491ec37834296f"
N4_RESULTS_SHA256 = "a7451fd56010cbcf6d0591e55c30d154c6e02fed4ce850a02a0a9c604e985002"
WORD_OPERATOR_GUARD = 1e-12
MATRIX_ERROR_GUARD = 1e-12
CLIFFORD_NAMES = frozenset({"h", "t", "tdg", "s", "sdg", "x", "y", "z"})
ROTATION_NAMES = frozenset({"rx", "ry", "rz"})
_DECIMAL = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_QUBIT = re.compile(r"(?:\$([0-9]+)|q\[([0-9]+)\])\Z")
_MEASURE = re.compile(r"c\[([0-9]+)\]\s*=\s*measure\s+(.+);\Z")
_GATE = re.compile(r"([a-z][a-z0-9_]*)(?:\((" + _DECIMAL + r")\))?\s+(.+);\Z")
_GPHASE = re.compile(r"gphase\((" + _DECIMAL + r")\);\Z")
_ISSUED_RECOVERIES: dict[int, tuple[weakref.ReferenceType, str, str]] = {}


@dataclass(frozen=True)
class WordRecoveryLimits:
    """Mandatory explicit bounds; no existing project default is changed."""

    max_words: int
    input_file_bytes: int
    routed_operations: int
    clifford_t_operations: int
    word_operations: int
    total_word_operations: int
    word_qasm_bytes: int
    catalog_bytes: int
    line_bytes: int = 4096

    def __post_init__(self):
        ceilings = {"max_words": 8, "input_file_bytes": 32*1024*1024,
                    "routed_operations": 100_000, "clifford_t_operations": 2_000_000,
                    "word_operations": 4096, "total_word_operations": 32_768,
                    "word_qasm_bytes": 32*1024, "catalog_bytes": 4*1024*1024,
                    "line_bytes": 4096}
        for name, maximum in ceilings.items():
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")


def literal_limits() -> WordRecoveryLimits:
    """An explicit preset for this one completed literal n4 input bundle."""
    return WordRecoveryLimits(8, 32*1024*1024, 100_000, 2_000_000,
                              4096, 32_768, 32*1024, 4*1024*1024)


def _limits(value) -> WordRecoveryLimits:
    if type(value) is not WordRecoveryLimits:
        raise ValueError("limits must be an explicit WordRecoveryLimits")
    return WordRecoveryLimits(**{field.name: getattr(value, field.name)
                                  for field in fields(WordRecoveryLimits)})


@dataclass(frozen=True)
class Primitive:
    name: str
    qubits: tuple[int, ...]
    clbits: tuple[int, ...] = ()
    params: tuple[float, ...] = ()

    @property
    def rotation_key(self) -> tuple[str, str] | None:
        return (self.name, self.params[0].hex()) if self.name in ROTATION_NAMES else None

    def as_record(self) -> dict:
        return {"name": self.name, "qubits": list(self.qubits), "clbits": list(self.clbits),
                "params_hex": [value.hex() for value in self.params]}


@dataclass(frozen=True)
class VerifiedSegment:
    routed_index: int
    ct_start: int
    routed: Primitive
    word_id: str | None
    tokens: tuple[Primitive, ...]
    ct_operations: int

    @property
    def rotation_key(self):
        return self.routed.rotation_key

    @property
    def wire(self):
        return self.routed.qubits[0]


@dataclass(frozen=True)
class WordSpec:
    word_id: str
    key: tuple[str, str]
    phase: float
    counts: tuple[tuple[str, int], ...]
    recorded_guarded_error: float
    requested_epsilon: float

    @property
    def operations(self):
        return sum(count for _, count in self.counts)


@dataclass(frozen=True)
class PhasedWord:
    spec: WordSpec
    tokens: tuple[Primitive, ...]
    occurrences: int
    checks: dict

    @property
    def word_id(self):
        return self.spec.word_id

    @property
    def key(self):
        return self.spec.key

    @property
    def phase(self):
        return self.spec.phase

    @property
    def counts(self):
        return dict(self.spec.counts)

    @property
    def sequence_sha256(self):
        return sequence_sha256(self.tokens)

    @property
    def qasm_sha256(self):
        return hashlib.sha256(export_word_qasm(self.tokens, self.phase)).hexdigest()

    def as_record(self) -> dict:
        _word_shape(self)
        raw = export_word_qasm(self.tokens, self.phase)
        matrix = manual_word_matrix(self.tokens, self.phase)
        return {"word_id": self.word_id, "rotation_gate": self.key[0],
                "angle_hex": self.key[1], "global_phase_binary64_hex": self.phase.hex(),
                "global_phase_radians": self.phase, "phase_scope": "recorded n4 approximant binary64 phase, restored explicitly",
                "gate_counts": dict(self.spec.counts), "operations": len(self.tokens),
                "occurrences": self.occurrences, "tokens": [p.as_record() for p in self.tokens],
                "sequence_sha256": sequence_sha256(self.tokens), "qasm_path": f"words/{self.word_id}.qasm",
                "qasm_sha256": hashlib.sha256(raw).hexdigest(),
                "matrix": {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()},
                "recorded_guarded_frobenius_error": self.spec.recorded_guarded_error,
                "epsilon_requested": self.spec.requested_epsilon, "checks": self.checks}


@dataclass(frozen=True)
class RecoveredWords:
    words: tuple[PhasedWord, ...]
    source_artifacts: dict
    summary: dict
    baseline_global_phase: float
    recovery_sha256: str
    bundle: Path | None = None

    @property
    def catalog_sha256(self):
        """Canonical payload identity; distinct from an emitted JSON file SHA."""
        return self.recovery_sha256

    def baseline_record(self, *, limits: WordRecoveryLimits):
        return baseline_record(self, limits=limits)

    def _payload(self):
        _recovery_shape(self)
        return {"schema_version": 1, "status": "completed_actual_phased_word_recovery",
                "case_id": "lindblad-n4-r211-Z0", "source_artifacts": self.source_artifacts,
                "word_count": len(self.words), "words": [w.as_record() for w in self.words],
                "summary": self.summary, "baseline_global_phase_binary64_hex": self.baseline_global_phase.hex(),
                "baseline_global_phase_radians": self.baseline_global_phase,
                "word_operator_guard": WORD_OPERATOR_GUARD, "recorded_matrix_error_guard": MATRIX_ERROR_GUARD,
                "full_unitary_word_checks_include_global_phase": True,
                "new_synthesis_calls": 0, "candidate_materialized": False,
                "candidate_full_density_executed": False, "native_physical_queries": 0,
                "formal_interval_certificate": False,
                "scope": "reconstruction of actual saved phased words and complete stream identity; no ported n4 execution or physical feasibility claim"}

    def as_record(self):
        value = self._payload()
        value["recovery_sha256"] = self.recovery_sha256
        return value


def _json_bytes(value) -> bytes:
    _record_shape(value)
    chunks, total = [], 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for piece in encoder.iterencode(value):
        raw = piece.encode()
        total += len(raw)
        if total > 4*1024*1024-1:
            raise ValueError("Canonical record exceeds its bounded byte cap")
        chunks.append(raw)
    return b"".join(chunks)+b"\n"


def _record_shape(value):
    remaining = 100_000
    def visit(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 16:
            raise ValueError("Record exceeds its bounded shape/depth cap")
        if item is None or type(item) is bool:
            return
        if type(item) is int and abs(item) <= 10**15:
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
                    raise ValueError("Record keys must be bounded concrete strings")
                visit(child, depth+1)
            return
        raise ValueError("Record contains an unsupported or oversized value")
    visit(value, 0)


def _recovery_shape(recovered):
    if (type(recovered.baseline_global_phase) is not float or
            not math.isfinite(recovered.baseline_global_phase)):
        raise ValueError("Recorded baseline phase must be a concrete finite binary64 float")
    if (type(recovered.recovery_sha256) is not str or
            recovered.recovery_sha256 and re.fullmatch(r"[0-9a-f]{64}", recovered.recovery_sha256) is None):
        raise ValueError("Recovery identity must be a concrete SHA256 string")
    if type(recovered.words) is not tuple or len(recovered.words) != 8:
        raise ValueError("Recovered words must be a bounded concrete eight-word tuple")
    total = 0
    for word in recovered.words:
        _word_shape(word)
        total += len(word.tokens)
    if total > 32_768:
        raise ValueError("Recovered words exceed their total operation cap")
    _record_shape(recovered.source_artifacts)
    _record_shape(recovered.summary)


def _word_shape(word):
    if type(word) is not PhasedWord or type(word.spec) is not WordSpec:
        raise ValueError("Recovered words must contain concrete phased word records")
    spec = word.spec
    if (type(spec.word_id) is not str or re.fullmatch(r"word-0[0-7]", spec.word_id) is None or
            type(spec.key) is not tuple or len(spec.key) != 2 or
            any(type(item) is not str for item in spec.key) or
            spec.key[0] not in ROTATION_NAMES or len(spec.key[1]) > 64):
        raise ValueError("Recovered native rotation identifiers must be bounded concrete keys")
    try:
        angle = float.fromhex(spec.key[1])
    except (ValueError, OverflowError) as error:
        raise ValueError("Recovered angle must be finite canonical binary64") from error
    if not math.isfinite(angle) or angle.hex() != spec.key[1]:
        raise ValueError("Recovered angle must be finite canonical binary64")
    if (type(spec.phase) is not float or not math.isfinite(spec.phase) or
            not 0 <= spec.phase < 2*math.pi or
            type(spec.recorded_guarded_error) is not float or not math.isfinite(spec.recorded_guarded_error) or
            spec.recorded_guarded_error < 0 or type(spec.requested_epsilon) is not float or
            not math.isfinite(spec.requested_epsilon) or spec.requested_epsilon <= 0):
        raise ValueError("Recorded word phases/errors must be concrete finite binary64 floats")
    if type(word.occurrences) is not int or not 1 <= word.occurrences <= 5908:
        raise ValueError("Recovered word occurrences must be bounded concrete integers")
    if type(spec.counts) is not tuple or not 1 <= len(spec.counts) <= len(CLIFFORD_NAMES):
        raise ValueError("Recovered native specifications exceed their shape cap")
    for pair in spec.counts:
        if (type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str or
                pair[0] not in CLIFFORD_NAMES or type(pair[1]) is not int or
                not 1 <= pair[1] <= 4096):
            raise ValueError("Recovered native count records are malformed")
    if type(word.tokens) is not tuple:
        raise ValueError("Recovered word tokens must be a bounded concrete tuple")
    _word_tokens(word.tokens, wire=0)
    _record_shape(word.checks)


def sequence_sha256(tokens) -> str:
    tokens = _word_tokens(tokens, wire=0)
    return hashlib.sha256(_json_bytes([p.as_record() for p in tokens])).hexdigest()


def _open_regular(path, *, maximum):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    handle = os.fdopen(descriptor, "rb")
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
        handle.close()
        raise ValueError("Input descriptor must refer to a bounded regular file")
    return handle


def bounded_file_sha256(path: Path, *, maximum: int) -> str:
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Input must be a bounded regular non-symlink file")
    result, total = hashlib.sha256(), 0
    with _open_regular(path, maximum=maximum) as handle:
        while block := handle.read(65536):
            total += len(block)
            if total > maximum:
                raise ValueError("Input file exceeds its byte cap")
            result.update(block)
    if path.is_symlink() or total != path.stat().st_size:
        raise ValueError("Input changed during bounded hashing")
    return result.hexdigest()


def _bounded_json(path, limits, *, expected_sha256=None):
    path = Path(path)
    expected = bounded_file_sha256(path, maximum=limits.input_file_bytes)
    if expected_sha256 is not None and expected != expected_sha256:
        raise ValueError("Saved metadata differs from its frozen consumed-byte identity")
    with _open_regular(path, maximum=limits.input_file_bytes) as handle:
        raw = handle.read(limits.input_file_bytes+1)
        if len(raw) > limits.input_file_bytes:
            raise ValueError("Saved metadata exceeds its explicit byte cap")
    if (path.is_symlink() or len(raw) != path.stat().st_size or
            hashlib.sha256(raw).hexdigest() != expected):
        raise ValueError("Saved metadata changed during its bounded read")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Saved metadata must be a JSON object")
    return value


def _finite_number(text: str) -> float:
    if re.fullmatch(_DECIMAL, text) is None:
        raise ValueError("Only finite decimal binary64 literals are supported")
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("Nonfinite numeric literals are unsupported")
    return value


def _qubit(text: str, width: int, *, physical: bool | None):
    match = _QUBIT.fullmatch(text.strip())
    if match is None:
        raise ValueError("Malformed qubit operand")
    is_physical = match[1] is not None
    if physical is not None and is_physical != physical:
        raise ValueError("Mixed physical and register wire syntax is unsupported")
    value = int(match[1] if is_physical else match[2])
    if not 0 <= value < width:
        raise ValueError("Qubit operand exceeds the declared width")
    return value


def _parse_primitive(line: str, width: int, *, physical: bool | None, rotations: bool) -> Primitive:
    match = _MEASURE.fullmatch(line)
    if match:
        bit = int(match[1])
        if not 0 <= bit < width:
            raise ValueError("Classical measurement index exceeds the declared width")
        return Primitive("measure", (_qubit(match[2], width, physical=physical),), (bit,))
    match = _GATE.fullmatch(line)
    if match is None:
        raise ValueError("Unsupported or malformed flat QASM statement")
    name, number, operands = match.groups()
    allowed = CLIFFORD_NAMES | {"cx", "barrier", "reset"} | (ROTATION_NAMES if rotations else set())
    if name not in allowed:
        raise ValueError("Unknown, hidden or non-flat QASM operation")
    wires = tuple(_qubit(x, width, physical=physical) for x in operands.split(","))
    if len(wires) != len(set(wires)):
        raise ValueError("Quantum operands must be distinct")
    arity = width if name == "barrier" else 2 if name == "cx" else 1
    if len(wires) != arity:
        raise ValueError("Operation has inconsistent fixed-schedule arity")
    if (name in ROTATION_NAMES) != (number is not None):
        raise ValueError("Operation has inconsistent rotation parameters")
    params = (_finite_number(number),) if number is not None else ()
    return Primitive(name, wires, (), params)


def iter_qasm(path: Path, *, kind: str, limits: WordRecoveryLimits) -> Iterator[Primitive]:
    """Parse only the pinned eight-wire flat routed or CT export dialect."""
    limits = _limits(limits)
    if kind not in {"routed", "clifford_t"}:
        raise ValueError("kind must be routed or clifford_t")
    expected_sha256 = bounded_file_sha256(path, maximum=limits.input_file_bytes)
    expected = ["OPENQASM 3.0;", 'include "stdgates.inc";', "bit[8] c;"]
    if kind == "clifford_t":
        expected.append("qubit[8] q;")
    maximum = limits.routed_operations if kind == "routed" else limits.clifford_t_operations
    physical, count, total = kind == "routed", 0, 0
    consumed_sha256 = hashlib.sha256()
    with _open_regular(path, maximum=limits.input_file_bytes) as handle:
        for wanted in expected:
            raw = handle.readline(limits.line_bytes+1)
            consumed_sha256.update(raw)
            total += len(raw)
            if len(raw) > limits.line_bytes or raw.decode("utf8").strip() != wanted:
                raise ValueError("Unexpected or malformed pinned QASM header")
        while True:
            raw = handle.readline(limits.line_bytes+1)
            if not raw:
                break
            consumed_sha256.update(raw)
            total += len(raw)
            if total > limits.input_file_bytes or len(raw) > limits.line_bytes:
                raise ValueError("QASM stream exceeds its predeclared byte/line cap")
            line = raw.decode("utf8").strip()
            if not line:
                continue
            count += 1
            if count > maximum:
                raise ValueError("QASM instruction stream exceeds its operation cap")
            yield _parse_primitive(line, 8, physical=physical, rotations=kind == "routed")
    if consumed_sha256.hexdigest() != expected_sha256:
        raise ValueError("QASM consumed bytes changed after its bounded input hash")


def _word_tokens(tokens, *, wire: int):
    if type(tokens) not in {tuple, list} or not 1 <= len(tokens) <= 4096:
        raise ValueError("Native word tokens must be a bounded concrete tuple/list")
    values = []
    for token in tokens:
        if (type(token) is not Primitive or type(token.name) is not str or
                token.name not in CLIFFORD_NAMES or type(token.qubits) is not tuple or
                token.qubits != (wire,) or type(token.clbits) is not tuple or token.clbits or
                type(token.params) is not tuple or token.params):
            raise ValueError("A recovered native word must be flat and strictly one-wire Clifford+T")
        values.append(Primitive(token.name, (0,)))
    return tuple(values)


_MATRICES = {"h": np.array([[1, 1], [1, -1]], complex)/math.sqrt(2),
             "t": np.diag([1, np.exp(1j*math.pi/4)]),
             "tdg": np.diag([1, np.exp(-1j*math.pi/4)]),
             "s": np.diag([1, 1j]), "sdg": np.diag([1, -1j]),
             "x": np.array([[0, 1], [1, 0]], complex),
             "y": np.array([[0, -1j], [1j, 0]], complex), "z": np.diag([1, -1])}


def _phase_value(value):
    if type(value) not in {float, int}:
        raise ValueError("Word phase must be a concrete finite binary64 number")
    try:
        value = float(value)
    except OverflowError as error:
        raise ValueError("Word phase must be finite") from error
    if not math.isfinite(value):
        raise ValueError("Word phase must be finite")
    return value


def manual_word_matrix(tokens, phase: float) -> np.ndarray:
    """Independent chronological matrix multiplication, including recorded phase."""
    phase = _phase_value(phase)
    matrix = np.eye(2, dtype=complex)
    for token in _word_tokens(tokens, wire=0):
        matrix = _MATRICES[token.name] @ matrix
    return np.exp(1j*phase)*matrix


def word_circuit(tokens, phase: float) -> QuantumCircuit:
    phase = _phase_value(phase)
    circuit = QuantumCircuit(1)
    circuit.global_phase = phase
    for token in _word_tokens(tokens, wire=0):
        getattr(circuit, token.name)(0)
    return circuit


def _target_matrix(key):
    gate, encoded = key
    angle = float.fromhex(encoded)
    c, s = math.cos(angle/2), math.sin(angle/2)
    if gate == "rx":
        return np.array([[c, -1j*s], [-1j*s, c]], complex)
    if gate == "ry":
        return np.array([[c, -s], [s, c]], complex)
    if gate == "rz":
        return np.diag([np.exp(-1j*angle/2), np.exp(1j*angle/2)])
    raise ValueError("Only the saved rotation axes are supported")


def export_word_qasm(tokens, phase: float) -> bytes:
    tokens = _word_tokens(tokens, wire=0)
    phase = _phase_value(phase)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[1] q;", f"gphase({phase!r});"]
    lines.extend(f"{p.name} q[0];" for p in tokens)
    raw = ("\n".join(lines)+"\n").encode()
    if len(raw) > 32*1024:
        raise ValueError("Phase-aware word export exceeds its fixed byte cap")
    return raw


def parse_word_qasm(raw: bytes, *, limits: WordRecoveryLimits):
    """Strict parser for our coherent one-wire export, with exactly one phase."""
    limits = _limits(limits)
    if type(raw) is not bytes or len(raw) > limits.word_qasm_bytes:
        raise ValueError("Word QASM exceeds its explicit byte cap")
    lines = raw.decode("utf8").splitlines()
    if any(len(line.encode()) > limits.line_bytes for line in lines):
        raise ValueError("Word QASM line exceeds its cap")
    if lines[:3] != ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[1] q;"] or len(lines) < 5:
        raise ValueError("Unexpected phase-aware word header")
    match = _GPHASE.fullmatch(lines[3])
    if match is None:
        raise ValueError("Exactly one explicit gphase must precede the word")
    phase, tokens = _finite_number(match[1]), []
    for line in lines[4:]:
        if not line:
            continue
        token = _parse_primitive(line, 1, physical=False, rotations=False)
        if token.name not in CLIFFORD_NAMES:
            raise ValueError("Word export permits no instrument or multi-wire instruction")
        tokens.append(token)
        if len(tokens) > limits.word_operations:
            raise ValueError("Word exceeds its operation cap")
    if not tokens:
        raise ValueError("Native word must not be empty")
    return tuple(tokens), phase


def _validate_word(spec: WordSpec, tokens, limits):
    if len(tokens) != spec.operations or len(tokens) > limits.word_operations:
        raise ValueError("Recovered word length differs from the saved native counts")
    if Counter(p.name for p in tokens) != Counter(dict(spec.counts)):
        raise ValueError("Recovered word gate counts differ from the saved approximant")
    manual = manual_word_matrix(tokens, spec.phase)
    qiskit = np.asarray(Operator(word_circuit(tokens, spec.phase)).data)
    raw = export_word_qasm(tokens, spec.phase)
    if len(raw) > limits.word_qasm_bytes:
        raise ValueError("Phase-aware word export exceeds its byte cap")
    parsed, phase = parse_word_qasm(raw, limits=limits)
    if parsed != tokens or phase.hex() != spec.phase.hex():
        raise ValueError("Phase-aware export failed exact token/phase roundtrip")
    exported = np.asarray(Operator(word_circuit(parsed, phase)).data)
    equality = float(np.linalg.norm(qiskit-manual, "fro"))
    roundtrip = float(np.linalg.norm(qiskit-exported, "fro"))
    unitarity = float(np.linalg.norm(qiskit.conj().T@qiskit-np.eye(2), "fro"))
    guarded = float(np.linalg.norm(qiskit-_target_matrix(spec.key), "fro"))+MATRIX_ERROR_GUARD
    difference = abs(guarded-spec.recorded_guarded_error)
    if not all(math.isfinite(x) for x in (equality, roundtrip, unitarity, guarded, difference)):
        raise ValueError("Word operator checks must be finite")
    if max(equality, roundtrip, unitarity, difference) > WORD_OPERATOR_GUARD:
        raise ValueError("Recovered word failed full-unitary/recorded-error checks")
    return {"manual_full_unitary_frobenius_residual": equality,
            "export_roundtrip_full_unitary_frobenius_residual": roundtrip,
            "unitarity_frobenius_residual": unitarity,
            "recomputed_guarded_frobenius_error": guarded,
            "recorded_error_absolute_residual": difference,
            "phase_binary64_roundtrip_exact": True,
            "comparison_ignores_global_phase": False}


def _specs(report, limits):
    rows = report.get("approximants")
    if not isinstance(rows, list) or not 1 <= len(rows) <= limits.max_words:
        raise ValueError("Saved approximants exceed the bounded word count")
    result, keys = [], set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Saved approximant rows must be JSON objects")
        gate, angle = row.get("gate"), row.get("angle_hex")
        if gate not in ROTATION_NAMES or not isinstance(angle, str):
            raise ValueError("Malformed saved rotation key")
        value = float.fromhex(angle)
        if not math.isfinite(value) or value.hex() != angle or (gate, angle) in keys:
            raise ValueError("Saved rotation keys must be finite canonical and unique")
        keys.add((gate, angle))
        counts = row.get("gate_counts")
        if not isinstance(counts, dict) or not counts or set(counts)-CLIFFORD_NAMES:
            raise ValueError("Saved native word has unsupported gate counts")
        if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in counts.values()):
            raise ValueError("Native gate counts must be positive integers")
        values = [row.get("global_phase"), row.get("guarded_frobenius_error"), row.get("epsilon_requested")]
        if any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in values):
            raise ValueError("Saved phases/errors must be finite binary64 numbers")
        phase, error, epsilon = map(float, values)
        if not 0 <= phase < 2*math.pi or error < 0 or epsilon <= 0:
            raise ValueError("Saved native phase/error is outside its declared domain")
        spec = WordSpec(f"word-{index:02d}", (gate, angle), phase, tuple(sorted(counts.items())), error, epsilon)
        if spec.operations > limits.word_operations:
            raise ValueError("Saved word length exceeds its explicit cap")
        result.append(spec)
    if sum(x.operations for x in result) > limits.total_word_operations:
        raise ValueError("Saved unique words exceed their total operation cap")
    return tuple(result)


def _walk(routed, ct, specs, *, limits, known_words=None, state=None):
    """Lockstep complete stream match; only eight normalized words are retained."""
    lookup = {s.key: s for s in specs}
    words = {} if known_words is None else {w.key: w.tokens for w in known_words}
    occurrences, route_counts, ct_counts = Counter(), Counter(), Counter()
    initialized, measured, event_digest = set(), [], hashlib.sha256()
    offset = route_count = 0
    output_started = False
    for index, token in enumerate(routed):
        route_count = index+1
        if route_count > limits.routed_operations:
            raise ValueError("Routed stream exceeds its operation cap")
        route_counts[token.name] += 1
        if token.name == "reset":
            initialized.add(token.qubits[0])
        elif token.name != "barrier" and not set(token.qubits) <= initialized:
            raise ValueError("Saved routed instruction uses an uninitialized wire")
        if token.name == "measure":
            output_started = True
            measured.append((token.qubits[0], token.clbits[0]))
        elif output_started and token.name != "barrier":
            raise ValueError("Quantum work after final readout is unsupported")
        if token.name in {"reset", "measure"}:
            event_digest.update(_json_bytes(token.as_record()))
        key = token.rotation_key
        if key is not None:
            if key not in lookup:
                raise ValueError("Routed rotation key has no saved native approximant")
            spec, raw = lookup[key], []
            for _ in range(spec.operations):
                try:
                    item = next(ct)
                except StopIteration as error:
                    raise ValueError("CT stream ends within a saved word") from error
                raw.append(item)
                ct_counts[item.name] += 1
            normalized = _word_tokens(raw, wire=token.qubits[0])
            if key in words and normalized != words[key]:
                raise ValueError("A repeated saved word differs from its first actual occurrence")
            if key not in words:
                words[key] = normalized
            occurrences[key] += 1
            yield VerifiedSegment(index, offset, token, spec.word_id, words[key], len(raw))
            offset += len(raw)
        else:
            try:
                item = next(ct)
            except StopIteration as error:
                raise ValueError("CT stream ends before its routed reference") from error
            if item != token:
                raise ValueError("Nonrotation/event/readout token differs between saved streams")
            ct_counts[item.name] += 1
            yield VerifiedSegment(index, offset, token, None, (item,), 1)
            offset += 1
        if offset > limits.clifford_t_operations:
            raise ValueError("CT stream exceeds its total operation cap")
    try:
        next(ct)
    except StopIteration:
        pass
    else:
        raise ValueError("CT stream contains trailing unmatched operations")
    if initialized != set(range(8)) or len(measured) != 8 or [x[1] for x in measured] != list(range(8)) or set(x[0] for x in measured) != set(range(8)):
        raise ValueError("Saved schedule must initialize and read every physical wire once at final output")
    if set(words) != set(lookup) or any(occurrences[s.key] == 0 for s in specs):
        raise ValueError("Every saved native word must occur in the complete streams")
    if state is not None:
        state.update(words=words, occurrences=occurrences, routed_operations=route_count,
                     clifford_t_operations=offset, routed_counts=dict(route_counts),
                     clifford_t_counts=dict(ct_counts), final_measurement_wires=[x[0] for x in measured],
                     event_sha256=event_digest.hexdigest(), completed=True)


def _load_frozen(bundle: Path, limits):
    bundle = Path(bundle)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("Saved bundle must be a regular directory")
    manifest_path, results_path = bundle/"manifest.json", bundle/"results.json"
    hashes = {"manifest.json": bounded_file_sha256(manifest_path, maximum=limits.input_file_bytes),
              "results.json": bounded_file_sha256(results_path, maximum=limits.input_file_bytes)}
    if hashes["manifest.json"] != N4_MANIFEST_SHA256 or hashes["results.json"] != N4_RESULTS_SHA256:
        raise ValueError("Input is not the frozen independently reviewed completed n4 bundle")
    manifest = _bounded_json(manifest_path, limits, expected_sha256=N4_MANIFEST_SHA256)
    results = _bounded_json(results_path, limits, expected_sha256=N4_RESULTS_SHA256)
    for name in ("source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"):
        hashes[name] = bounded_file_sha256(bundle/name, maximum=limits.input_file_bytes)
        if hashes[name] != manifest["artifacts_sha256"].get(name):
            raise ValueError("Actual input QASM hash differs from its frozen manifest")
    for name, expected in manifest["source_sha256"].items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Saved snapshot filenames must be simple leaves")
        if bounded_file_sha256(bundle/name, maximum=limits.input_file_bytes) != expected:
            raise ValueError("Saved source/lock/reference snapshot differs from its attestation")
    prepared = results.get("prepared")
    if not isinstance(prepared, dict) or prepared.get("case_id") != "lindblad-n4-r211-Z0" or prepared.get("selected_steps") != 211:
        raise ValueError("Only the completed literal n4/r211 trial is supported")
    report = prepared["actual_synthesis"]
    if (report.get("compiler_domain") != "literal-n4-explicit-opt-in-v1" or
            report.get("actual_clifford_t_synthesis_performed") is not True or
            report.get("actual_synthesis_sequences_generated") is not True or
            report.get("rotation_occurrences") != 5908 or report.get("unique_rotation_targets") != 8):
        raise ValueError("Saved n4 report lacks actual complete synthesis proof")
    if prepared["complete_workload"]["complete_trials"] != 15756 or report["operator_error_allowance"] != 5e-5:
        raise ValueError("The frozen task/error/all-trial contract changed")
    phase = float(report["compiled_global_phase"])
    if not math.isfinite(phase):
        raise ValueError("Saved whole-circuit phase must be finite")
    return report, hashes, phase


def _module_sha256():
    return bounded_file_sha256(Path(__file__), maximum=1024*1024)


def _issue_recovery(recovered):
    identity = id(recovered)
    reference = weakref.ref(recovered, lambda _: _ISSUED_RECOVERIES.pop(identity, None))
    _ISSUED_RECOVERIES[identity] = (reference, recovered.recovery_sha256, _module_sha256())


def recover_words(bundle: Path, *, limits: WordRecoveryLimits) -> RecoveredWords:
    """Recover/validate every actual n4 occurrence, retaining only eight words."""
    limits = _limits(limits)
    if limits.max_words != 8:
        raise ValueError("The literal n4 recovery requires an explicit eight-word bound")
    report, hashes, phase = _load_frozen(bundle, limits)
    specs = _specs(report, limits)
    state = {}
    routed = iter_qasm(Path(bundle)/"routed-rotation-trial.qasm", kind="routed", limits=limits)
    ct = iter_qasm(Path(bundle)/"clifford-t-trial.qasm", kind="clifford_t", limits=limits)
    reconstructed_phase = float(report["routed_global_phase"])
    for segment in _walk(routed, ct, specs, limits=limits, state=state):
        if segment.word_id is not None:
            reconstructed_phase = (reconstructed_phase+next(s.phase for s in specs if s.word_id == segment.word_id))%(2*math.pi)
    if (state["routed_counts"] != report["routed_counts"] or state["clifford_t_counts"] != report["clifford_t_counts"] or
            sum(state["occurrences"].values()) != 5908 or state["routed_operations"] != 24510 or
            state["clifford_t_operations"] != 1677906 or state["routed_counts"].get("reset") != 1481):
        raise ValueError("Recovered complete stream counts differ from the frozen actual reports")
    phase_residual = abs(np.exp(1j*reconstructed_phase)-np.exp(1j*phase))
    if phase_residual > WORD_OPERATOR_GUARD:
        raise ValueError("Recorded whole phase differs from chronological recovered word phases")
    words = tuple(PhasedWord(s, state["words"][s.key], state["occurrences"][s.key],
                            _validate_word(s, state["words"][s.key], limits)) for s in specs)
    summary = {k: v for k, v in state.items() if k not in {"words", "occurrences"}}
    summary.update(rotation_occurrences=5908, total_unique_word_operations=sum(len(w.tokens) for w in words),
                   reconstructed_global_phase_binary64_hex=reconstructed_phase.hex(),
                   global_phase_factor_absolute_residual=float(phase_residual),
                   complete_saved_stream_match=True)
    result = RecoveredWords(words, hashes, summary, phase, "", Path(bundle))
    result = RecoveredWords(words, hashes, summary, phase, hashlib.sha256(_json_bytes(result._payload())).hexdigest(), Path(bundle))
    if len(_json_bytes(result.as_record())) > limits.catalog_bytes:
        raise ValueError("Recovered word catalog exceeds its byte cap")
    # Inputs are checked again after the walk; changes never produce attested words.
    if _load_frozen(bundle, limits)[1] != hashes:
        raise ValueError("Frozen input changed during word reconstruction")
    _issue_recovery(result)
    return result


def _verified_recovery(recovered, limits):
    if type(recovered) is not RecoveredWords or recovered.bundle is None:
        raise ValueError("A completed fixed-bundle word recovery is required")
    issued = _ISSUED_RECOVERIES.get(id(recovered))
    if (issued is None or issued[0]() is not recovered or
            recovered.recovery_sha256 != issued[1] or _module_sha256() != issued[2]):
        raise ValueError("A process-issued unmodified source-verified recovery is required")
    _recovery_shape(recovered)
    if (len(recovered.words) > limits.max_words or
            any(len(word.tokens) > limits.word_operations for word in recovered.words) or
            sum(len(word.tokens) for word in recovered.words) > limits.total_word_operations):
        raise ValueError("Recovered words exceed the caller's explicit operation bounds")
    payload = _json_bytes(recovered._payload())
    if len(payload) > limits.catalog_bytes:
        raise ValueError("Recovered catalog exceeds the caller's explicit byte cap")
    if hashlib.sha256(payload).hexdigest() != recovered.recovery_sha256:
        raise ValueError("Recovered catalog changed after full-unitary verification")
    if _load_frozen(recovered.bundle, limits)[1] != recovered.source_artifacts:
        raise ValueError("Frozen baseline input changed after recovery")


def require_verified_recovery(recovered: RecoveredWords, *, limits: WordRecoveryLimits) -> None:
    """Require a fresh in-process attestation; serialized catalogs are evidence only."""
    _verified_recovery(recovered, _limits(limits))


def baseline_record(recovered: RecoveredWords, *, limits: WordRecoveryLimits) -> dict:
    """Return a verified copy of saved preparation/checks, with no density rerun."""
    limits = _limits(limits)
    _verified_recovery(recovered, limits)
    saved = _bounded_json(recovered.bundle/"results.json", limits, expected_sha256=N4_RESULTS_SHA256)["prepared"]
    _verified_recovery(recovered, limits)
    return saved


def iter_verified_segments(recovered: RecoveredWords, *, limits: WordRecoveryLimits) -> Iterator[VerifiedSegment]:
    """Replay checked source correspondence for a separate frozen recipe builder."""
    limits = _limits(limits)
    _verified_recovery(recovered, limits)
    state, specs = {}, tuple(w.spec for w in recovered.words)
    routed = iter_qasm(recovered.bundle/"routed-rotation-trial.qasm", kind="routed", limits=limits)
    ct = iter_qasm(recovered.bundle/"clifford-t-trial.qasm", kind="clifford_t", limits=limits)
    yield from _walk(routed, ct, specs, limits=limits, known_words=recovered.words, state=state)
    if any(state[k] != recovered.summary[k] for k in ("routed_operations", "clifford_t_operations", "routed_counts", "clifford_t_counts", "final_measurement_wires", "event_sha256")):
        raise ValueError("Replayed streams differ from their completed recovery")
    _verified_recovery(recovered, limits)


def recovery_artifacts(recovered: RecoveredWords, *, limits: WordRecoveryLimits) -> dict[str, bytes]:
    """Return deliberate exports for the caller's own new artifact directory."""
    limits = _limits(limits)
    _verified_recovery(recovered, limits)
    outputs = {f"words/{w.word_id}.qasm": export_word_qasm(w.tokens, w.phase) for w in recovered.words}
    outputs["word-catalog.json"] = (json.dumps(recovered.as_record(), indent=2, allow_nan=False)+"\n").encode()
    if any(len(raw) > limits.word_qasm_bytes for name, raw in outputs.items() if name.endswith(".qasm")) or len(outputs["word-catalog.json"]) > limits.catalog_bytes:
        raise ValueError("Recovered exports exceed their predeclared byte caps")
    return outputs
