# SPDX-License-Identifier: MIT
"""Unexecuted independent ideal full-program verifier for the fixed k=2 fixture.

This source is a prospective implementation, not runtime readiness or evidence.
It imports no producer, NumPy, quantum SDK or monitor. A separately authorized
driver must authenticate source/environment files before supplying NumPy 2.5.4
and a complete-process wall/CPU/RSS/disk callback. The exact Boolean opt-in and
once-only counters are API guards, not an execution grant.

One packed 22-wire state evolves through every emitted primitive. Preparation,
sign, incidence, decoder and final-H checkpoints inspect that same forward pass.
Reference construction uses the fixed edges, direct low-weight character sums
and direct edge scoring, never a producer mass, incidence or decoder helper.
The three reusable complex chunk buffers, real/index/mask/reduction buffers and
both full vectors are charged before allocation. NumPy internal/runtime memory
is separate from this buffer envelope and needs the driver's sampled RSS guard.

Five vectors are streamed separately as little-endian complex128 (interleaved
float64 real/imaginary): initial, preparation actual/reference and final
actual/reference. No amplitude threshold, global phase fit, renormalization,
postselection, sampling, dense operator or logical-CCX substitution is used.
Initial resets act on the explicitly known zero input; readout is the complete
unconditional marginal with its nine individual source mappings, not shots.
The 37 RY occurrences remain ideal unsynthesized binary rotations. Physical
resources, fault-tolerant synthesis and formal proof remain unavailable.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Callable


WIRES = 22
AMPLITUDES = 1 << WIRES
CLAUSES = 10
ASSIGNMENTS = 512
CHUNK = 1 << 18
STATE_BYTES = AMPLITUDES * 16
BUFFER_LIMIT = 256 << 20
METADATA_RESERVE = 16 << 20
INTERNAL_CHUNK_RESERVE = 32 << 20
OUTPUT_LIMIT = 512 << 20
OUTPUT_FILES = 7
TOLERANCE = 2e-10
SIGNS = (1, 1, 0, 0, 0, 1, 0, 0, 0, 0)
PATHS = ((0, 2, 3, 4, 5, 1), (0, 6, 7, 8, 9, 1))
EDGES = ((0, 2), (2, 3), (3, 4), (4, 5), (5, 1),
         (0, 6), (6, 7), (7, 8), (8, 9), (9, 1))
COUNTS = {"h": 253, "cx": 834, "t": 488, "tdg": 366, "x": 54,
          "ry": 37, "z": 3, "reset": 22, "measure": 9, "barrier": 7}
STAGES = ("reset", "preparation", "sign", "syndrome", "decoder", "hadamard", "readout")
STAGE_RANGES = ((0, 23), (23, 979), (979, 983), (983, 1002),
                (1002, 2053), (2053, 2063), (2063, 2073))
BARRIER_POSITIONS = (22, 978, 982, 1001, 2052, 2062, 2072)
UNITARIES = frozenset({"h", "cx", "t", "tdg", "x", "ry", "z"})
VECTOR_NAMES = ("initial-state.bin", "preparation-reference.bin",
                "preparation-actual.bin", "final-reference.bin", "final-actual.bin")


class K2VerificationError(ValueError):
    """The frozen source, input, allocation or finite diagnostic was rejected."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise K2VerificationError(message)


def _integer(value, low: int, high: int, label: str) -> int:
    _require(type(value) is int and low <= value <= high, label + " is not a bounded exact integer")
    return value


def _check(callback: Callable[[], None]) -> None:
    callback()


class ExecutionBudget:
    """Once-only charged entries; a failure leaves the invocation spent."""

    def __init__(self, execution_enabled: bool = False):
        _require(type(execution_enabled) is bool, "Execution enablement must be Boolean")
        self._enabled = execution_enabled
        self._entered = self._active = self._complete = False
        self._counts = {
            "verification_invocations": 0, "complete_statevector_evolutions": 0,
            "primitive_unitary_applications": 0, "source_records_consumed": 0,
            "known_zero_resets": 0, "barriers": 0, "readout_mappings": 0,
            "checkpoint_stages": 0, "checkpoint_full_state_scans": 0,
            "state_scan_chunks": 0, "state_scan_amplitudes": 0,
            "primitive_pair_chunks": 0, "primitive_pairs_addressed": 0,
            "numpy_kernel_calls": 0, "numpy_kernel_elements": 0,
            "numpy_reduction_calls": 0, "numpy_reduction_input_elements": 0,
            "reference_buffer_zeroing_passes": 0, "reference_entries_written": 0,
            "scalar_reference_angle_evaluations": 0, "degree_two_words_checked": 0,
            "promised_word_direct_decoder_mappings": 0,
            "independent_assignments_scanned": 0, "independent_edge_scores": 0,
            "independent_character_terms": 0, "vector_files_completed": 0,
            "parity_sign_reads": 0, "parity_assignment_internal_steps": 0,
            "parity_witness_edge_scores": 0,
            "output_files_created": 0, "output_bytes_written": 0,
        }

    def begin(self):
        _require(self._enabled is True and not self._entered, "Execution is disabled or already spent")
        self._entered = self._active = True
        self._counts["verification_invocations"] = 1

    def charge(self, key, amount=1, ceiling=None):
        _require(self._active and key in self._counts, "An active known counter is required")
        _integer(amount, 0, 1 << 45, "Counter increment")
        value = self._counts[key] + amount
        if ceiling is not None:
            _require(value <= ceiling, "The fixed " + key + " ceiling was reached")
        self._counts[key] = value

    def reserve_evolution(self):
        self.charge("complete_statevector_evolutions", ceiling=1)

    def kernel(self, elements, *, reduction=False):
        self.charge("numpy_kernel_calls")
        self.charge("numpy_kernel_elements", elements)
        if reduction:
            self.charge("numpy_reduction_calls")
            self.charge("numpy_reduction_input_elements", elements)

    def finish(self):
        exact = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
                 "primitive_unitary_applications": 2035, "source_records_consumed": 2073,
                 "known_zero_resets": 22, "barriers": 7, "readout_mappings": 9,
                 "checkpoint_stages": 5, "checkpoint_full_state_scans": 5,
                 "state_scan_chunks": 5 * (AMPLITUDES // CHUNK),
                 "state_scan_amplitudes": 5 * AMPLITUDES,
                 "primitive_pair_chunks": 2035 * ((AMPLITUDES // 2) // CHUNK),
                 "primitive_pairs_addressed": 2035 * (AMPLITUDES // 2),
                 "scalar_reference_angle_evaluations": 19, "degree_two_words_checked": 56,
                 "promised_word_direct_decoder_mappings": 56,
                 "independent_assignments_scanned": 512, "independent_edge_scores": 5120,
                 "independent_character_terms": 28672,
                 "parity_sign_reads": 10, "parity_assignment_internal_steps": 8,
                 "parity_witness_edge_scores": 10,
                 "reference_buffer_zeroing_passes": 5, "reference_entries_written": 736,
                 "vector_files_completed": 5, "output_files_created": OUTPUT_FILES}
        _require(self._active and all(self._counts[k] == v for k, v in exact.items()),
                 "The complete fixed role counters do not close")
        _require(self._counts["numpy_reduction_calls"] > 0
                 and self._counts["output_bytes_written"] <= OUTPUT_LIMIT,
                 "Reduction/output closure is absent")
        self._complete = True
        self._active = False

    def abort(self):
        if self._entered and not self._complete:
            self._active = False

    def snapshot(self):
        return {**self._counts, "spent": self._entered, "active": self._active,
                "complete": self._complete, "counter_scope": "actual entered operations; no retry"}


def _fixed_input(value):
    _require(type(value) is dict and set(value) == {"k", "signs"},
             "The fixed input must have exactly k and signs")
    _require(type(value["k"]) is int and value["k"] == 2, "Only the fixed k=2 profile is supported")
    signs = value["signs"]
    _require(type(signs) is list and len(signs) == 10
             and all(type(x) is int and x in (0, 1) for x in signs)
             and tuple(signs) == SIGNS, "The mixed-parity fixed signs differ")


def _primitive(gate, wires, classical=(), parameters=()):
    return gate, tuple(wires), tuple(classical), tuple(parameters)


def _ccx(a, b, target):
    return [_primitive("h", (target,)), _primitive("cx", (b, target)),
            _primitive("tdg", (target,)), _primitive("cx", (a, target)),
            _primitive("t", (target,)), _primitive("cx", (b, target)),
            _primitive("tdg", (target,)), _primitive("cx", (a, target)),
            _primitive("t", (b,)), _primitive("t", (target,)),
            _primitive("h", (target,)), _primitive("cx", (a, b)),
            _primitive("t", (a,)), _primitive("tdg", (b,)),
            _primitive("cx", (a, b))]


def _branch_angle(position, prefix_weight, budget):
    """Independent direct combinatorial masses, not the producer's ratio helper."""
    budget.charge("scalar_reference_angle_evaluations", ceiling=19)
    remaining = 10 - position
    coefficient = (10, 10, 2)
    denominator = sum(math.comb(remaining, extra) * coefficient[prefix_weight + extra] ** 2
                      for extra in range(3 - prefix_weight))
    numerator = sum(math.comb(remaining - 1, extra) * coefficient[prefix_weight + 1 + extra] ** 2
                    for extra in range(2 - prefix_weight))
    _require(0 <= numerator <= denominator and denominator > 0, "Reference branch mass is invalid")
    value = 2 * math.asin(math.sqrt(numerator / denominator))
    _require(math.isfinite(value) and 2 * (value / 2) == value, "Reference RY is not finite/losslessly halved")
    return value


def _grammar(budget):
    """Independent complete expected primitive grammar; evaluated only after opt-in."""
    records = [_primitive("reset", (wire,)) for wire in range(22)]
    barrier = _primitive("barrier", range(22))
    records.append(barrier)
    c0, c1, flag = 19, 20, 21
    for position in range(10):
        if position == 0:
            angle = _branch_angle(position, 0, budget)
            records.append(_primitive("ry", (position,), parameters=(angle.hex(),)))
        else:
            for prefix_weight in (0, 1):
                angle = _branch_angle(position, prefix_weight, budget)
                negatives = (c0, c1) if prefix_weight == 0 else (c1,)
                records.extend(_primitive("x", (wire,)) for wire in negatives)
                records.extend(_ccx(c0, c1, flag))
                records.extend((_primitive("ry", (position,), parameters=((angle / 2).hex(),)),
                                _primitive("cx", (flag, position)),
                                _primitive("ry", (position,), parameters=((-angle / 2).hex(),)),
                                _primitive("cx", (flag, position))))
                records.extend(_ccx(c0, c1, flag))
                records.extend(_primitive("x", (wire,)) for wire in reversed(negatives))
        if position < 9:
            records.extend(_ccx(position, c0, c1))
            records.append(_primitive("cx", (position, c0)))
    for position in range(8, -1, -1):
        records.append(_primitive("cx", (position, c0)))
        records.extend(_ccx(position, c0, c1))
    records.append(barrier)
    records.extend(_primitive("z", (i,)) for i, value in enumerate(SIGNS) if value)
    records.append(barrier)
    for clause, edge in enumerate(EDGES):
        for vertex in sorted(edge):
            if vertex:
                records.append(_primitive("cx", (clause, 10 + vertex - 1)))
    records.append(barrier)
    for path in range(2):
        syndrome = tuple(11 + 4 * path + j for j in range(4))
        targets = tuple(5 * path + j for j in range(5))
        prefix = tuple((syndrome[j], syndrome[j + 1]) for j in range(3))
        records.extend(_primitive("cx", pair) for pair in prefix)
        words = []
        for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            words.extend(((syndrome[a], syndrome[b], c0),
                          (c0, syndrome[c], flag), (syndrome[a], syndrome[b], c0)))
        words.extend(((syndrome[0], syndrome[1], c0), (c0, syndrome[2], c1),
                      (c1, syndrome[3], flag), (c0, syndrome[2], c1),
                      (syndrome[0], syndrome[1], c0)))
        for word in words:
            records.extend(_ccx(*word))
        records.extend(_primitive("cx", (flag, target)) for target in targets)
        records.extend(_primitive("cx", (wire, targets[j + 1])) for j, wire in enumerate(syndrome))
        for word in reversed(words):
            records.extend(_ccx(*word))
        records.extend(_primitive("cx", pair) for pair in reversed(prefix))
    records.append(barrier)
    records.extend(_primitive("h", (wire,)) for wire in range(10, 19))
    records.append(barrier)
    records.extend(_primitive("measure", (10 + bit,), classical=(bit,)) for bit in range(9))
    records.append(barrier)
    _require(len(records) == 2073 and Counter(row[0] for row in records) == COUNTS,
             "The independent fixed grammar does not close")
    return records


def _parsed_record(value):
    _require(type(value) is dict and type(value.get("gate")) is str,
             "A primitive must have a typed gate field")
    gate = value["gate"]
    _require(gate in COUNTS, "An unsupported primitive was supplied")
    keys = {"gate", "qubits"} | ({"classical"} if gate == "measure" else set()) \
           | ({"parameters_hex"} if gate == "ry" else set())
    _require(set(value) == keys, "Primitive fields differ from the exact event schema")
    wires = value["qubits"]
    arity = 22 if gate == "barrier" else 2 if gate == "cx" else 1
    _require(type(wires) is list and len(wires) == arity
             and all(type(w) is int and 0 <= w < 22 for w in wires)
             and len(set(wires)) == len(wires), "Primitive wire types/arity/range differ")
    classical, parameters = (), ()
    if gate == "measure":
        bits = value["classical"]
        _require(type(bits) is list and len(bits) == 1 and type(bits[0]) is int
                 and 0 <= bits[0] < 9, "Readout requires one exact classical bit")
        classical = tuple(bits)
    if gate == "ry":
        angles = value["parameters_hex"]
        _require(type(angles) is list and len(angles) == 1
                 and type(angles[0]) is str and 0 < len(angles[0]) <= 64,
                 "RY requires one bounded binary-hex angle")
        try:
            angle = float.fromhex(angles[0])
        except (ValueError, OverflowError) as error:
            raise K2VerificationError("RY angle cannot be parsed") from error
        _require(math.isfinite(angle) and angle.hex() == angles[0]
                 and abs(angle) <= 2 * math.pi, "RY angle is noncanonical/nonfinite/out of range")
        parameters = tuple(angles)
    return _primitive(gate, wires, classical, parameters)


def _parse_records(records, budget, callback):
    _require(type(records) in (list, tuple) and len(records) == 2073,
             "The complete fixed program has exactly 2073 records")
    expected = _grammar(budget)
    parsed = []
    for position, (value, reference) in enumerate(zip(records, expected, strict=True)):
        _check(callback)
        row = _parsed_record(value)
        _require(row == reference, "The complete primitive grammar differs at source index " + str(position))
        parsed.append(row)
    _require(tuple(i for i, row in enumerate(parsed) if row[0] == "barrier") == BARRIER_POSITIONS,
             "Full stage separators differ")
    return tuple(parsed)


class _Buffers:
    """All large arrays are owned here; metadata has a separate conservative reserve."""

    def __init__(self, np, budget, callback):
        self.np, self.budget, self.callback = np, budget, callback
        # Reserve bounded iterator/index/ufunc workspace in addition to every
        # explicit array. This is prospective headroom, not a measured memory
        # proof; unexpected runtime allocation is caught separately by RSS.
        self.current = METADATA_RESERVE + INTERNAL_CHUNK_RESERVE
        self.peak = self.current
        self.allocations = {}
        self.state = self.empty("actual_state", AMPLITUDES, "<c16", 16)
        self.reference = self.empty("reference_state", AMPLITUDES, "<c16", 16)
        self.complex = [self.empty("complex_chunk_" + str(i), CHUNK, "<c16", 16) for i in range(3)]
        self.indices = [self.empty("index_chunk_" + str(i), CHUNK, "<i8", 8) for i in range(3)]
        self.template = self.empty("index_template", CHUNK, "<i8", 8)
        self.real = [self.empty("real_chunk_" + str(i), CHUNK, "<f8", 8) for i in range(3)]
        self.mask = [self.empty("mask_chunk_" + str(i), CHUNK, "?", 1) for i in range(2)]
        self.output = self.empty("unconditional_probabilities", 512, "<f8", 8)
        self.rows = self.empty("row_reduction", CHUNK // 1024, "<f8", 8)
        # A bounded scalar fill avoids an uncharged np.arange temporary.
        for i in range(CHUNK):
            if i % 4096 == 0:
                _check(callback)
            self.template[i] = i

    def empty(self, label, length, dtype, itemsize):
        _check(self.callback)
        size = _integer(length, 1, AMPLITUDES, "Array length") * itemsize
        _require(self.current + size <= BUFFER_LIMIT and label not in self.allocations,
                 "The aggregate numerical buffer ceiling was reached")
        # Charge before the allocation; failed allocations remain represented.
        self.current += size
        self.peak = max(self.peak, self.current)
        self.allocations[label] = {"bytes": size, "dtype": dtype, "length": length}
        value = self.np.empty(length, dtype=dtype)
        _require(value.nbytes == size and value.dtype.itemsize == itemsize
                 and value.flags.c_contiguous, "Actual array allocation differs from its charged shape")
        return value

    def call(self, function, *args, elements, reduction=False, **kwargs):
        _check(self.callback)
        self.budget.kernel(elements, reduction=reduction)
        result = function(*args, **kwargs)
        _check(self.callback)
        return result

    def zero(self, value, *, reference=False):
        self.budget.kernel(value.size)
        _check(self.callback)
        value.fill(0)
        _check(self.callback)
        if reference:
            self.budget.charge("reference_buffer_zeroing_passes", ceiling=5)

    def scatter(self, indices, values, length):
        # Proven-in-range 1-D addresses and non-overlapping contiguous values.
        # clip avoids a bounds-error buffering path and never clips an address.
        # Iterator/chunk internals have separately charged reserve headroom.
        self.call(self.np.put, self.state, indices[:length], values[:length],
                  mode="clip", elements=length)

    def mix(self, length, left_left, left_right, right_left, right_right):
        """Two explicit scalar linear combinations with three complex buffers."""
        np = self.np
        a, b, out = (x[:length] for x in self.complex)
        scratch = self.real[0][:length]
        left, right = self.indices[0][:length], self.indices[1][:length]
        for destination, c0, c1 in ((left, left_left, left_right), (right, right_left, right_right)):
            self.call(np.multiply, b, c1, out=out, elements=length)
            self.call(np.multiply, a.real, c0, out=scratch, elements=length)
            self.call(np.add, out.real, scratch, out=out.real, elements=length)
            self.call(np.multiply, a.imag, c0, out=scratch, elements=length)
            self.call(np.add, out.imag, scratch, out=out.imag, elements=length)
            self.scatter(destination, out, length)

    def apply(self, row):
        gate, wires, _, parameters = row
        _require(gate in UNITARIES, "Only a primitive unitary can enter the kernel")
        self.budget.charge("primitive_unitary_applications", ceiling=2035)
        target = wires[-1]
        bit = 1 << target
        np = self.np
        if gate == "ry":
            angle = float.fromhex(parameters[0])
            cosine, sine = math.cos(angle / 2), math.sin(angle / 2)
        inverse_root_two = 1 / math.sqrt(2)
        phase = complex(inverse_root_two, inverse_root_two if gate == "t" else -inverse_root_two)
        for base in range(0, AMPLITUDES // 2, CHUNK):
            length = min(CHUNK, AMPLITUDES // 2 - base)
            self.budget.charge("primitive_pair_chunks")
            self.budget.charge("primitive_pairs_addressed", length)
            left, right, temporary = (x[:length] for x in self.indices)
            self.call(np.add, self.template[:length], base, out=left, elements=length)
            self.call(np.bitwise_and, left, bit - 1, out=right, elements=length)
            self.call(np.right_shift, left, target, out=left, elements=length)
            self.call(np.left_shift, left, target + 1, out=left, elements=length)
            self.call(np.bitwise_or, left, right, out=left, elements=length)
            self.call(np.bitwise_or, left, bit, out=right, elements=length)
            a, b, out = (x[:length] for x in self.complex)
            # The pair-address formula is in range for every validated target.
            # clip avoids NumPy's mode='raise' out-buffer; it never clips a valid address.
            self.call(np.take, self.state, left, out=a, mode="clip", elements=length)
            self.call(np.take, self.state, right, out=b, mode="clip", elements=length)
            if gate == "h":
                self.mix(length, inverse_root_two, inverse_root_two, inverse_root_two, -inverse_root_two)
            elif gate == "ry":
                self.mix(length, cosine, -sine, sine, cosine)
            elif gate == "x":
                self.scatter(left, b, length)
                self.scatter(right, a, length)
            elif gate == "cx":
                self.call(np.bitwise_and, left, 1 << wires[0], out=temporary, elements=length)
                active = self.mask[0][:length]
                self.call(np.not_equal, temporary, 0, out=active, elements=length)
                self.call(np.copyto, out, a, elements=length)
                self.call(np.copyto, out, b, where=active, elements=length)
                self.scatter(left, out, length)
                self.call(np.copyto, b, a, where=active, elements=length)
                self.scatter(right, b, length)
            else:
                self.call(np.multiply, b, -1 if gate == "z" else phase, out=out, elements=length)
                self.scatter(right, out, length)

    def allocation_summary(self):
        return {"limit_bytes": BUFFER_LIMIT, "metadata_reserve_bytes": METADATA_RESERVE,
                "internal_chunk_workspace_reserve_bytes": INTERNAL_CHUNK_RESERVE,
                "charged_peak_bytes": self.peak, "allocations": dict(self.allocations),
                "reserve_is_prospective_not_runtime_memory_proof": True,
                "whole_process_rss_is_separate": True, "runtime_internal_allocations_not_measured": True}


def _syndrome(word):
    value = 0
    for clause, (u, v) in enumerate(EDGES):
        if (word >> clause) & 1:
            if u:
                value ^= 1 << (u - 1)
            if v:
                value ^= 1 << (v - 1)
    return value


def _d2_words(budget, callback):
    words = [0] + [1 << bit for bit in range(10)]
    words.extend((1 << a) | (1 << b) for a in range(10) for b in range(a + 1, 10))
    _require(len(words) == 56, "The direct D2 reference size differs")
    seen = set()
    rows = []
    sign_mask = sum(bit << i for i, bit in enumerate(SIGNS))
    for word in words:
        _check(callback)
        budget.charge("degree_two_words_checked", ceiling=56)
        syndrome = _syndrome(word)
        _require(syndrome not in seen, "Direct edge incidence is not injective on the promised words")
        seen.add(syndrome)
        reconstructed = 0
        for path in range(2):
            prefix = [0]
            for j in range(4):
                prefix.append(prefix[-1] ^ ((syndrome >> (1 + 4 * path + j)) & 1))
            first = int(sum(prefix) >= 3)
            for j, value in enumerate(prefix):
                reconstructed |= (first ^ value) << (5 * path + j)
        _require(reconstructed == word, "The direct majority law fails on a promised word")
        budget.charge("promised_word_direct_decoder_mappings", ceiling=56)
        weight = word.bit_count()
        coefficient = 10 if weight < 2 else 2
        signed = -coefficient if (word & sign_mask).bit_count() % 2 else coefficient
        rows.append((word, syndrome, coefficient, signed))
    return tuple(rows)


def _references(buffers, stage, words, final_amplitudes):
    buffers.zero(buffers.reference, reference=True)
    if stage == "hadamard":
        entries = ((assignment << 10, value) for assignment, value in enumerate(final_amplitudes))
    else:
        def entries_for_words():
            for word, syndrome, coefficient, signed in words:
                index = word if stage in {"preparation", "sign"} else word | (syndrome << 10)
                if stage == "decoder":
                    index = syndrome << 10
                numerator = coefficient if stage == "preparation" else signed
                yield index, numerator / math.sqrt(1280)
        entries = entries_for_words()
    for index, value in entries:
        _check(buffers.callback)
        _require(math.isfinite(value) and 0 <= index < AMPLITUDES, "Reference entry is invalid")
        buffers.budget.charge("reference_entries_written", ceiling=736)
        buffers.reference[index] = complex(value, 0)


def _diagnose(buffers, stage):
    budget, np = buffers.budget, buffers.np
    budget.charge("checkpoint_stages", ceiling=5)
    budget.charge("checkpoint_full_state_scans", ceiling=5)
    counts_before = budget.snapshot()
    maximum = maximum_imaginary = norm = auxiliary = weight_leakage = 0.0
    sectors = [0.0, 0.0, 0.0]
    if stage == "hadamard":
        buffers.zero(buffers.output)
    auxiliary_mask = ((1 << 22) - 1) ^ ((1 << 10) - 1) if stage in {"preparation", "sign"} \
        else ((1 << 22) - 1) ^ ((1 << 19) - 1)
    if stage in {"decoder", "hadamard"}:
        auxiliary_mask |= (1 << 10) - 1
    for base in range(0, AMPLITUDES, CHUNK):
        length = min(CHUNK, AMPLITUDES - base)
        budget.charge("state_scan_chunks")
        budget.charge("state_scan_amplitudes", length)
        actual = buffers.state[base:base + length]
        reference = buffers.reference[base:base + length]
        difference = buffers.complex[2][:length]
        probability, temporary_real, error = (x[:length] for x in buffers.real)
        index, weight, temporary_index = (x[:length] for x in buffers.indices)
        selected, finite = (x[:length] for x in buffers.mask)
        buffers.call(np.isfinite, actual, out=finite, elements=length)
        _require(bool(buffers.call(np.all, finite, elements=length, reduction=True)), "Actual state is nonfinite")
        buffers.call(np.subtract, actual, reference, out=difference, elements=length)
        buffers.call(np.absolute, difference, out=error, elements=length)
        maximum = max(maximum, float(buffers.call(np.max, error, elements=length, reduction=True)))
        buffers.call(np.absolute, difference.imag, out=error, elements=length)
        maximum_imaginary = max(maximum_imaginary, float(buffers.call(np.max, error, elements=length, reduction=True)))
        buffers.call(np.square, actual.real, out=probability, elements=length)
        buffers.call(np.square, actual.imag, out=temporary_real, elements=length)
        buffers.call(np.add, probability, temporary_real, out=probability, elements=length)
        norm += float(buffers.call(np.sum, probability, dtype=np.float64, elements=length, reduction=True))
        buffers.call(np.add, buffers.template[:length], base, out=index, elements=length)
        buffers.call(np.bitwise_and, index, auxiliary_mask, out=temporary_index, elements=length)
        buffers.call(np.not_equal, temporary_index, 0, out=selected, elements=length)
        auxiliary += float(buffers.call(np.sum, probability, where=selected, dtype=np.float64,
                                       elements=length, reduction=True))
        if stage == "preparation":
            buffers.zero(weight)
            for bit in range(10):
                buffers.call(np.bitwise_and, index, 1 << bit, out=temporary_index, elements=length)
                buffers.call(np.not_equal, temporary_index, 0, out=selected, elements=length)
                buffers.call(np.add, weight, selected, out=weight, elements=length)
            buffers.call(np.greater, weight, 2, out=selected, elements=length)
            weight_leakage += float(buffers.call(np.sum, probability, where=selected, dtype=np.float64,
                                                elements=length, reduction=True))
            for sector in range(3):
                buffers.call(np.equal, weight, sector, out=selected, elements=length)
                sectors[sector] += float(buffers.call(np.sum, probability, where=selected, dtype=np.float64,
                                                     elements=length, reduction=True))
        if stage == "hadamard":
            # Every chunk has complete 1024-clause rows. Offsets wrap across all
            # eight workspace blocks; no workspace-zero restriction is applied.
            rows = length // 1024
            _require(length % 1024 == 0 and rows <= len(buffers.rows), "Marginal row shape is invalid")
            buffers.call(np.sum, probability.reshape(rows, 1024), axis=1,
                         out=buffers.rows[:rows], dtype=np.float64, elements=length, reduction=True)
            offset = (base >> 10) & 511
            _require(offset + rows <= 512, "Unconditional output offset exceeds its bounded rows")
            buffers.call(np.add, buffers.output[offset:offset + rows], buffers.rows[:rows],
                         out=buffers.output[offset:offset + rows], elements=rows)
    _require(all(math.isfinite(x) for x in [maximum, maximum_imaginary, norm, auxiliary]),
             "Checkpoint scalar diagnostics are nonfinite")
    _require(maximum <= TOLERANCE and abs(norm - 1) <= TOLERANCE
             and auxiliary <= TOLERANCE, "Full signed state/norm/auxiliary diagnostic failed at " + stage)
    if stage == "preparation":
        _require(weight_leakage <= TOLERANCE and all(abs(a - b) <= TOLERANCE
                 for a, b in zip(sectors, (5 / 64, 25 / 32, 9 / 64), strict=True)),
                 "Unconditional preparation weight-sector diagnostic failed")
    counts_after = budget.snapshot()
    return {"stage": stage, "max_signed_complex_entry_error": maximum,
            "max_imaginary_entry_error": maximum_imaginary, "norm2": norm,
            "auxiliary_leakage_probability": auxiliary, "auxiliary_mask": auxiliary_mask,
            "weight_above_two_leakage_probability": weight_leakage if stage == "preparation" else None,
            "unconditional_weight_sector_probabilities": sectors if stage == "preparation" else None,
            "actual_full_state_scans": 1,
            "numpy_reduction_calls": counts_after["numpy_reduction_calls"] - counts_before["numpy_reduction_calls"],
            "numpy_reduction_input_elements": counts_after["numpy_reduction_input_elements"] - counts_before["numpy_reduction_input_elements"],
            "relative_tolerance": 0.0, "absolute_tolerance": TOLERANCE}


def _classical_reference(words, budget, callback):
    amplitudes, scores, rows = [], [], []
    histogram = Counter()
    denominator = math.sqrt(512 * 1280)
    for assignment in range(512):
        _check(callback)
        budget.charge("independent_assignments_scanned", ceiling=512)
        score, incidence_signs = 0, []
        for (u, v), target in zip(EDGES, SIGNS, strict=True):
            budget.charge("independent_edge_scores", ceiling=5120)
            u_bit = 0 if u == 0 else (assignment >> (u - 1)) & 1
            v_bit = 0 if v == 0 else (assignment >> (v - 1)) & 1
            satisfied = (u_bit ^ v_bit) == target
            score += int(satisfied)
            incidence_signs.append(1 if satisfied else -1)
        signed_sum = 2 * score - 10
        polynomial = (signed_sum * signed_sum + 10 * signed_sum) / denominator
        direct = 0
        for word, _, coefficient, _ in words:
            budget.charge("independent_character_terms", ceiling=28672)
            sign = 1
            for bit in range(10):
                if (word >> bit) & 1:
                    sign *= incidence_signs[bit]
            direct += coefficient * sign
        _require(abs(direct / denominator - polynomial) <= TOLERANCE,
                 "Independent character sum and score polynomial disagree")
        amplitudes.append(polynomial)
        scores.append(score)
        histogram[score] += 1
        rows.append({"assignment": assignment, "gauge_bits": [(assignment >> j) & 1 for j in range(9)],
                     "score": score, "reference_amplitude_real": polynomial,
                     "reference_amplitude_imaginary": 0.0,
                     "reference_probability": polynomial * polynomial})
    _require(dict(histogram) == {9: 10, 7: 120, 5: 252, 3: 120, 1: 10},
             "Direct assignment-score multiplicities differ")
    # Independent O(m) parity witness, kept separate from exhaustive enumeration.
    parity_values = []
    for path in range(2):
        parity = 0
        for target in SIGNS[5 * path:5 * path + 5]:
            budget.charge("parity_sign_reads", ceiling=10)
            parity ^= target
        parity_values.append(parity)
    parities = tuple(parity_values)
    terminal = int(parities.count(1) > parities.count(0))
    vertices = [0] * 10
    vertices[1] = terminal
    for path in range(2):
        current = 0
        mismatch = int(parities[path] != terminal)
        for edge in range(4):
            budget.charge("parity_assignment_internal_steps", ceiling=8)
            current ^= SIGNS[5 * path + edge] ^ int(mismatch and edge == 0)
            vertices[2 + 4 * path + edge] = current
    witness_score = 0
    for (u, v), target in zip(EDGES, SIGNS, strict=True):
        budget.charge("parity_witness_edge_scores", ceiling=10)
        witness_score += int((vertices[u] ^ vertices[v]) == target)
    _require(witness_score == 9 and tuple(vertices) == (0, 0, 1, 0, 0, 0, 0, 0, 0, 0),
             "Independent path-parity witness differs")
    return amplitudes, scores, rows, {"optimum": 9, "optimum_multiplicity": 10,
        "score_multiplicities": dict(histogram), "parity_witness_vertices": vertices,
        "parity_witness_gauge_integer": 2, "path_parities": list(parities),
        "parity_witness_work": {"sign_parity_reads": 10, "internal_assignment_steps": 8,
                               "independent_edge_scores": 10, "full_vertex_output": 10},
        "analytic_optimum_probability": 81 / 256, "analytic_expected_score": 115 / 16,
        "no_quantum_advantage_claim": True}


def _output_directory(value):
    path = Path(value)
    _require(path.is_absolute(), "The verifier output directory must be absolute")
    for parent in (path, *path.parents):
        _require(stat.S_ISDIR(parent.lstat().st_mode), "Output parents must be ordinary directories")
    _require(not any(path.iterdir()), "The verifier output directory must be fresh and empty")
    return path


class _Writer:
    def __init__(self, directory, budget, callback):
        self.directory, self.budget, self.callback = directory, budget, callback
        self.files = {}

    def stream(self, name, chunks, *, expected_bytes=None):
        _require(type(name) is str and "/" not in name and name not in self.files,
                 "Output names must be unique simple names")
        self.budget.charge("output_files_created", ceiling=OUTPUT_FILES)
        path = self.directory / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        digest, written = hashlib.sha256(), 0
        try:
            for chunk in chunks:
                _check(self.callback)
                view = memoryview(chunk).cast("B")
                _require(0 < len(view) <= 1 << 20, "Output chunks must be bounded and nonempty")
                _require(self.budget.snapshot()["output_bytes_written"] + len(view) <= OUTPUT_LIMIT,
                         "Aggregate output byte limit would be exceeded")
                offset = 0
                while offset < len(view):
                    _check(self.callback)
                    count = os.write(fd, view[offset:])
                    _require(count > 0, "Output write did not advance")
                    self.budget.charge("output_bytes_written", count, ceiling=OUTPUT_LIMIT)
                    digest.update(view[offset:offset + count])
                    offset += count
                    written += count
            os.fsync(fd)
            _check(self.callback)
        finally:
            os.close(fd)
        if expected_bytes is not None:
            _require(written == expected_bytes, "Complete binary vector length differs")
        self.files[name] = {"bytes": written, "sha256": digest.hexdigest()}
        return self.files[name]

    def vector(self, name, value):
        _require(sys.byteorder == "little" and value.dtype.str == "<c16"
                 and value.flags.c_contiguous and value.nbytes == STATE_BYTES,
                 "Saved vectors require the exact little-endian packed representation")
        view = memoryview(value).cast("B")
        self.stream(name, (view[i:i + (1 << 20)] for i in range(0, len(view), 1 << 20)),
                    expected_bytes=STATE_BYTES)
        self.budget.charge("vector_files_completed", ceiling=5)

    def json(self, name, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        _require(0 < len(raw) <= 1 << 20, "JSON output must be bounded")
        return self.stream(name, (raw,))


def _verify_active(records, input_payload, output_dir, *, numpy_module,
                   check_callback, budget):
    _check(check_callback)
    _fixed_input(input_payload)
    _require(getattr(numpy_module, "__name__", None) == "numpy"
             and getattr(numpy_module, "__version__", None) == "2.5.4",
             "The driver must supply the authenticated locked NumPy module")
    directory = _output_directory(output_dir)
    parsed = _parse_records(records, budget, check_callback)
    words = _d2_words(budget, check_callback)
    final_amplitudes, scores, output_rows, comparator = _classical_reference(words, budget, check_callback)
    budget.reserve_evolution()
    buffers = _Buffers(numpy_module, budget, check_callback)
    buffers.zero(buffers.state)
    buffers.state[0] = 1 + 0j
    writer = _Writer(directory, budget, check_callback)
    writer.vector("initial-state.bin", buffers.state)
    checkpoints = []
    barrier_number = 0
    for source_index, row in enumerate(parsed):
        _check(check_callback)
        budget.charge("source_records_consumed", ceiling=2073)
        gate = row[0]
        if gate == "reset":
            _require(source_index < 22 and row[1] == (source_index,), "Only initial known-zero resets are allowed")
            budget.charge("known_zero_resets", ceiling=22)
        elif gate in UNITARIES:
            buffers.apply(row)
        elif gate == "measure":
            budget.charge("readout_mappings", ceiling=9)
        else:
            budget.charge("barriers", ceiling=7)
            stage = STAGES[barrier_number]
            barrier_number += 1
            if stage not in {"reset", "readout"}:
                _references(buffers, stage, words, final_amplitudes)
                checkpoints.append(_diagnose(buffers, stage))
                if stage == "preparation":
                    writer.vector("preparation-reference.bin", buffers.reference)
                    writer.vector("preparation-actual.bin", buffers.state)
                elif stage == "hadamard":
                    writer.vector("final-reference.bin", buffers.reference)
                    writer.vector("final-actual.bin", buffers.state)
        _check(check_callback)
    probabilities = [float(buffers.output[x]) for x in range(512)]
    _require(all(math.isfinite(p) and p >= 0 for p in probabilities), "Unconditional output probabilities are invalid")
    probability_error = max(abs(p - row["reference_probability"])
                            for p, row in zip(probabilities, output_rows, strict=True))
    probability_norm = math.fsum(probabilities)
    optimum_probability = math.fsum(p for p, score in zip(probabilities, scores, strict=True) if score == 9)
    expected_score = math.fsum(p * score for p, score in zip(probabilities, scores, strict=True))
    _require(probability_error <= TOLERANCE and abs(probability_norm - 1) <= TOLERANCE
             and abs(optimum_probability - 81 / 256) <= TOLERANCE
             and abs(expected_score - 115 / 16) <= TOLERANCE,
             "Independent unconditional output/comparator checks failed")
    for row, probability in zip(output_rows, probabilities, strict=True):
        row["actual_unconditional_probability"] = probability
    writer.json("output-table.json", {"schema": "openquantum.dqi.path_majority_k2_output.v1",
        "rows": output_rows, "comparator": comparator,
        "readout_mapping": [{"wire": 10 + bit, "classical_bit": bit} for bit in range(9)],
        "unconditional_over_all_clause_workspace_values": True, "sampling_performed": False})
    diagnostics = {"schema": "openquantum.dqi.path_majority_k2_full_validation.v1",
        "status": "diagnostics_completed_not_execution_acceptance", "input": input_payload,
        "paths": [list(path) for path in PATHS], "edges": [list(edge) for edge in EDGES],
        "Q": 22, "N": 10, "m": 10, "n": 9, "state_amplitudes": AMPLITUDES,
        "primitive_counts": dict(COUNTS), "unitary_primitive_applications": 2035,
        "source_records": 2073, "operational_events_excluding_barriers": 2066,
        "source_stage_ranges": {stage: list(bounds) for stage, bounds in zip(STAGES, STAGE_RANGES, strict=True)},
        "checkpoint_diagnostics": checkpoints, "buffer_accounting": buffers.allocation_summary(),
        "output_probability_error": probability_error, "output_probability_norm": probability_norm,
        "optimum_probability": optimum_probability, "expected_score": expected_score,
        "comparator": comparator, "absolute_tolerance": TOLERANCE, "relative_tolerance": 0.0,
        "vector_layout": {"dtype": "little-endian complex128", "real_imaginary": "interleaved float64",
                          "wires": 22, "amplitudes": AMPLITUDES, "bytes_per_vector": STATE_BYTES,
                          "wire_zero_least_significant": True, "vector_files": list(VECTOR_NAMES)},
        "full_program_ideal_diagnostics_passed": True, "parent_acceptance_recorded": False,
        "full_program_validated": False,
        "reset_scope": "22 ordered resets of the explicitly known zero input",
        "readout_scope": "nine source mappings and unconditional probability marginal; no shots",
        "global_phase_fitted": False, "state_renormalized": False, "auxiliary_postselection": False,
        "sparse_truncation": False, "dense_operator_constructed": False, "sampling_performed": False,
        "physical_resources_estimated": False, "rotation_synthesis_validated": False,
        "formal_proof_reproduced": False, "hardware_executed": False,
        "runtime_readiness_claimed": False, "publication_complete": False}
    writer.json("diagnostics.json", diagnostics)
    _check(check_callback)
    budget.finish()
    return {**diagnostics, "status": "complete_ideal_full_program_check_pending_parent_acceptance",
            "file_pins": dict(writer.files), "execution_budget": budget.snapshot()}


def verify_full_program_records(records, input_payload, output_dir, *, numpy_module,
                                allow_numerical_execution=False, check_callback=None,
                                budget=None):
    """Prospective once-only complete ideal check, disabled unless explicitly opted in."""
    _require(type(allow_numerical_execution) is bool and allow_numerical_execution is True,
             "Full-program numerical verification is disabled")
    _require(callable(check_callback), "An enclosing CPU/wall/RSS/disk callback is required")
    if budget is None:
        budget = ExecutionBudget(True)
    _require(type(budget) is ExecutionBudget, "An exact once-only execution budget is required")
    budget.begin()
    try:
        return _verify_active(records, input_payload, output_dir, numpy_module=numpy_module,
                              check_callback=check_callback, budget=budget)
    except BaseException:
        budget.abort()
        raise
