"""Saved-stream reconstruction checks; no synthesis or density execution."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import signal

import numpy as np
import pytest
from qiskit.quantum_info import DensityMatrix, Operator

from openquantum import instrument_resources as ir
from openquantum.simulation import lindblad_four_compile as old_compiler
from openquantum.simulation import lindblad_port_words as w


BUNDLE = Path(__file__).resolve().parents[1]/"assets/generated/simulation/lindblad-four"


@pytest.fixture
def limits():
    return w.literal_limits()


def small_limit(limits, **changes):
    return replace(limits, **changes)


def write_qasm(tmp_path, kind, body):
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', "bit[8] c;"]
    if kind == "clifford_t":
        lines.append("qubit[8] q;")
    path = tmp_path/f"{kind}.qasm"
    path.write_text("\n".join(lines+body)+"\n")
    return path


@pytest.mark.parametrize("field,value", [
    ("max_words", 9), ("max_words", 0), ("input_file_bytes", 33*1024*1024),
    ("routed_operations", 100_001), ("clifford_t_operations", 2_000_001),
    ("word_operations", 4097), ("total_word_operations", 32_769),
    ("word_qasm_bytes", 32_769), ("catalog_bytes", 4*1024*1024+1),
    ("line_bytes", 4097), ("line_bytes", True), ("word_operations", 1.0),
])
def test_limits_are_literal_bounded_integers(field, value, limits):
    with pytest.raises(ValueError, match=field):
        small_limit(limits, **{field: value})


def test_limits_are_mandatory_and_revalidated(tmp_path, limits):
    with pytest.raises(TypeError):
        w.WordRecoveryLimits()
    path = write_qasm(tmp_path, "routed", ["reset $0;"])
    with pytest.raises(TypeError):
        list(w.iter_qasm(path, kind="routed"))
    class Override(w.WordRecoveryLimits):
        def __post_init__(self):
            pass
    forged = Override(**{**asdict(limits), "word_operations": 900_000})
    with pytest.raises(ValueError, match="explicit WordRecoveryLimits"):
        list(w.iter_qasm(path, kind="routed", limits=forged))
    object.__setattr__(limits, "word_operations", 900_000)
    with pytest.raises(ValueError, match="word_operations"):
        list(w.iter_qasm(path, kind="routed", limits=limits))


def test_integer_subclass_cannot_override_bound_comparisons(limits):
    class DeceptiveInt(int):
        def __ge__(self, other):
            return True
        def __le__(self, other):
            return True
    with pytest.raises(ValueError, match="input_file_bytes"):
        small_limit(limits, input_file_bytes=DeceptiveInt(10**12))
    object.__setattr__(limits, "input_file_bytes", [])
    with pytest.raises(ValueError, match="input_file_bytes"):
        w._limits(limits)


def test_pinned_routed_dialect_preserves_barrier_operand_order(tmp_path, limits):
    body = ["reset $0;", "rx(1.25e-4) $0;", "cx $0, $7;",
            "barrier $1,$0,$3,$4,$2,$5,$6,$7;", "c[2] = measure $7;"]
    result = list(w.iter_qasm(write_qasm(tmp_path, "routed", body), kind="routed", limits=limits))
    assert result[1].rotation_key == ("rx", float("1.25e-4").hex())
    assert result[3].qubits == (1, 0, 3, 4, 2, 5, 6, 7)
    assert result[4] == w.Primitive("measure", (7,), (2,))


def test_pinned_ct_dialect_and_exact_instruction_cap(tmp_path, limits):
    path = write_qasm(tmp_path, "clifford_t", ["h q[2];", "t q[2];"])
    assert [x.name for x in w.iter_qasm(path, kind="clifford_t", limits=small_limit(limits, clifford_t_operations=2))] == ["h", "t"]
    with pytest.raises(ValueError, match="operation cap"):
        list(w.iter_qasm(path, kind="clifford_t", limits=small_limit(limits, clifford_t_operations=1)))
    with pytest.raises(ValueError, match="kind"):
        list(w.iter_qasm(path, kind="word", limits=limits))


@pytest.mark.parametrize("line", [
    "gphase(0.0);", "include \"hidden.inc\";", "qubit[8] q;", "bit[8] c;",
    "gate hidden q { x q; }", "ctrl @ x $0,$1;", "if (c[0]) x $0;",
    "for int i in [0:7] { x $0; }", "rx(pi) $0;", "rx(nan) $0;",
    "rx(1e999) $0;", "rx(0x1.0p-1) $0;", "rx(1.0,2.0) $0;", "rx $0;",
    "h(0.5) $0;", "cx $0,$0;", "cx $0;", "h $8;", "h q[0];",
    "c[8] = measure $0;", "c[0] = measure $0,$1;", "x $0; x $1;",
    "x $0; // hidden suffix", "barrier $0,$1;", "barrier $0,$1,$2,$3,$4,$5,$6,$6;",
    "reset $٠;", "c[٠] = measure $0;", "rx(٠.١) $0;",
])
def test_hidden_nonflat_or_malformed_routed_statement_rejects(line, tmp_path, limits):
    with pytest.raises(ValueError):
        list(w.iter_qasm(write_qasm(tmp_path, "routed", [line]), kind="routed", limits=limits))


@pytest.mark.parametrize("line", ["rx(0.1) q[0];", "h $0;", "gphase(0.0);"])
def test_ct_stream_disallows_rotations_physical_wires_and_hidden_phase(line, tmp_path, limits):
    with pytest.raises(ValueError):
        list(w.iter_qasm(write_qasm(tmp_path, "clifford_t", [line]), kind="clifford_t", limits=limits))


@pytest.mark.parametrize("header", [
    "OPENQASM 2.0;", "OPENQASM 3.0; x $0;", "// comment", "",
])
def test_headers_are_exact_and_anchored(header, tmp_path, limits):
    path = write_qasm(tmp_path, "routed", ["reset $0;"])
    raw = path.read_text().splitlines(); raw[0] = header
    path.write_text("\n".join(raw)+"\n")
    with pytest.raises(ValueError, match="header"):
        list(w.iter_qasm(path, kind="routed", limits=limits))


def test_input_bytes_line_bytes_and_symlink_caps_precede_parsing(tmp_path, limits):
    path = write_qasm(tmp_path, "routed", ["h $0;"])
    with pytest.raises(ValueError, match="bounded regular"):
        list(w.iter_qasm(path, kind="routed", limits=small_limit(limits, input_file_bytes=10)))
    with pytest.raises(ValueError, match="header"):
        list(w.iter_qasm(path, kind="routed", limits=small_limit(limits, line_bytes=10)))
    path = write_qasm(tmp_path, "routed", [" "*4096+"h $0;"])
    with pytest.raises(ValueError, match="byte/line cap"):
        list(w.iter_qasm(path, kind="routed", limits=limits))
    link = tmp_path/"link.qasm"; link.symlink_to(path)
    with pytest.raises(ValueError, match="bounded regular"):
        list(w.iter_qasm(link, kind="routed", limits=limits))


def test_metadata_read_is_bounded_even_if_file_changes_after_hash(tmp_path, limits, monkeypatch):
    path = tmp_path/"metadata.json"; path.write_text('{}')
    original = w.bounded_file_sha256
    def changed(*args, **kwargs):
        digest = original(*args, **kwargs)
        path.write_bytes(b" "*101)
        return digest
    monkeypatch.setattr(w, "bounded_file_sha256", changed)
    with pytest.raises(ValueError, match="bounded regular"):
        w._bounded_json(path, small_limit(limits, input_file_bytes=100))


@pytest.mark.parametrize("replacement", ["fifo", "symlink"])
def test_qasm_hash_open_race_cannot_block_or_follow_link(tmp_path, limits, monkeypatch, replacement):
    import os
    path = write_qasm(tmp_path, "routed", ["reset $0;"])
    other = tmp_path/"other.qasm"; other.write_bytes(path.read_bytes())
    original = w.bounded_file_sha256
    def changed(*args, **kwargs):
        digest = original(*args, **kwargs)
        path.unlink()
        if replacement == "fifo":
            os.mkfifo(path)
        else:
            path.symlink_to(other)
        return digest
    monkeypatch.setattr(w, "bounded_file_sha256", changed)
    with pytest.raises((ValueError, OSError)):
        list(w.iter_qasm(path, kind="routed", limits=limits))


def test_qasm_consumed_bytes_are_bound_to_preparse_hash(tmp_path, limits, monkeypatch):
    path = write_qasm(tmp_path, "routed", ["reset $0;"])
    original = w.bounded_file_sha256
    def changed(*args, **kwargs):
        digest = original(*args, **kwargs)
        path.write_text(path.read_text().replace("reset $0;", "reset $1;"))
        return digest
    monkeypatch.setattr(w, "bounded_file_sha256", changed)
    with pytest.raises(ValueError, match="consumed bytes changed"):
        list(w.iter_qasm(path, kind="routed", limits=limits))


def test_metadata_rejects_same_size_change_between_hash_and_read(tmp_path, limits, monkeypatch):
    path = tmp_path/"metadata.json"; path.write_text('{"a":0}')
    original = w.bounded_file_sha256
    def changed(*args, **kwargs):
        digest = original(*args, **kwargs)
        path.write_text('{"a":1}')
        return digest
    monkeypatch.setattr(w, "bounded_file_sha256", changed)
    with pytest.raises(ValueError, match="changed during"):
        w._bounded_json(path, limits)


def test_phase_aware_word_binary64_roundtrip_and_chronological_matrix(limits):
    tokens = tuple(w.Primitive(name, (0,)) for name in ("h", "t", "sdg", "x"))
    phase = 5.105088062083413
    raw = w.export_word_qasm(tokens, phase)
    parsed, restored = w.parse_word_qasm(raw, limits=limits)
    assert parsed == tokens and restored.hex() == phase.hex()
    assert f"gphase({phase!r});".encode() in raw
    manual = w.manual_word_matrix(tokens, phase)
    actual = Operator(w.word_circuit(parsed, restored)).data
    assert np.linalg.norm(manual-actual, "fro") < 1e-14
    assert np.linalg.norm(actual-w.manual_word_matrix(tokens, 0.), "fro") > 1.
    assert w.sequence_sha256(tokens) != w.sequence_sha256(tokens[::-1])


@pytest.mark.parametrize("edit", [
    lambda x: x.replace(b"gphase(0.5);\n", b""),
    lambda x: x.replace(b"gphase(0.5);", b"gphase(0.5);\ngphase(0.5);"),
    lambda x: x.replace(b"gphase(0.5);", b"gphase(nan);"),
    lambda x: x.replace(b"gphase(0.5);", b"gphase(1e999);"),
    lambda x: x.replace(b"h q[0];", b"ctrl @ h q[0];"),
    lambda x: x.replace(b"h q[0];", b"reset q[0];"),
    lambda x: x.replace(b"h q[0];", b"h q[1];"),
    lambda x: x.replace(b"h q[0];", b"rx(0.5) q[0];"),
    lambda x: x.replace(b"h q[0];", b"h q[0]; // suffix"),
    lambda x: x.replace(b"h q[0];", b"h q[0];\n"+b"x"*4097),
])
def test_phase_loss_hidden_syntax_and_nonunitary_word_reject(edit, limits):
    raw = w.export_word_qasm((w.Primitive("h", (0,)),), .5)
    with pytest.raises(ValueError):
        w.parse_word_qasm(edit(raw), limits=limits)


def test_word_qasm_byte_and_operation_bounds(limits):
    raw = w.export_word_qasm((w.Primitive("h", (0,)), w.Primitive("t", (0,))), .5)
    with pytest.raises(ValueError, match="byte cap"):
        w.parse_word_qasm(raw, limits=small_limit(limits, word_qasm_bytes=len(raw)-1))
    with pytest.raises(ValueError, match="operation cap"):
        w.parse_word_qasm(raw, limits=small_limit(limits, word_operations=1))
    with pytest.raises(ValueError, match="finite"):
        w.export_word_qasm((w.Primitive("h", (0,)),), math.inf)
    with pytest.raises(ValueError, match="fixed byte cap"):
        w.export_word_qasm((w.Primitive("h", (0,)),)*4096, .5)


@pytest.mark.parametrize("operation", [w.export_word_qasm, w.manual_word_matrix, w.word_circuit])
def test_word_helpers_reject_unbounded_or_one_shot_iterators(operation):
    with pytest.raises(ValueError, match="bounded concrete"):
        operation(iter((w.Primitive("h", (0,)),)), .5)
    with pytest.raises(ValueError, match="bounded concrete"):
        operation((w.Primitive("h", (0,)),)*4097, .5)
    with pytest.raises(ValueError, match="bounded concrete"):
        w.sequence_sha256(iter((w.Primitive("h", (0,)),)))


def test_bytes_subclass_cannot_forge_word_byte_length(limits):
    class DeceptiveBytes(bytes):
        def __len__(self):
            return 1
    raw = DeceptiveBytes(w.export_word_qasm((w.Primitive("h", (0,)),), .5))
    with pytest.raises(ValueError, match="byte cap"):
        w.parse_word_qasm(raw, limits=small_limit(limits, word_qasm_bytes=1))


@pytest.mark.parametrize("operation", [w.export_word_qasm, w.manual_word_matrix, w.word_circuit])
def test_phase_subclass_hooks_are_rejected_before_invocation(operation):
    class PoisonFloat(float):
        def __repr__(self):
            pytest.fail("phase subclass hook must not execute")
        def hex(self):
            pytest.fail("phase subclass hook must not execute")
    with pytest.raises(ValueError, match="concrete finite"):
        operation((w.Primitive("h", (0,)),), PoisonFloat(.5))


def small_streams():
    key = ("rx", (.5).hex())
    spec = w.WordSpec("word-00", key, 0., (("h", 1), ("t", 1)), .1, .2)
    resets = [w.Primitive("reset", (i,)) for i in range(8)]
    rotations = [w.Primitive("rx", (wire,), (), (.5,)) for wire in (2, 5)]
    measurements = [w.Primitive("measure", (i,), (i,)) for i in range(8)]
    routed = resets+rotations+measurements
    ct = resets+[w.Primitive(name, (wire,)) for wire in (2, 5) for name in ("h", "t")]+measurements
    return routed, ct, (spec,)


def test_segment_match_normalizes_words_but_preserves_events_and_physical_wire(limits):
    routed, ct, specs = small_streams(); state = {}
    segments = list(w._walk(iter(routed), iter(ct), specs, limits=limits, state=state))
    assert segments[8].wire == 2 and segments[9].wire == 5
    assert segments[8].tokens is segments[9].tokens
    assert [x.qubits for x in segments[8].tokens] == [(0,), (0,)]
    assert [x.ct_start for x in segments[8:11]] == [8, 10, 12]
    assert segments[0].tokens == (routed[0],)
    assert state["routed_operations"] == 18 and state["clifford_t_operations"] == 20
    assert state["occurrences"][specs[0].key] == 2 and state["completed"]


@pytest.mark.parametrize("change,match", [
    ("repeat", "repeated saved word"), ("word-wire", "strictly one-wire"),
    ("short-word", "ends within"), ("trailing", "trailing unmatched"),
    ("event", "Nonrotation/event/readout"), ("uninitialized", "uninitialized"),
    ("readout-order", "initialize and read"), ("late-work", "after final readout"),
    ("unknown-key", "no saved native"), ("unused-spec", "Every saved native"),
])
def test_complete_stream_mismatch_rejects(change, match, limits):
    routed, ct, specs = small_streams()
    if change == "repeat":
        ct[11] = w.Primitive("s", (5,))
    elif change == "word-wire":
        ct[8] = w.Primitive("h", (3,))
    elif change == "short-word":
        ct = ct[:9]
    elif change == "trailing":
        ct.append(w.Primitive("h", (0,)))
    elif change == "event":
        ct[0] = w.Primitive("reset", (1,))
    elif change == "uninitialized":
        routed = routed[1:]; ct = ct[1:]
    elif change == "readout-order":
        routed[-1] = ct[-1] = w.Primitive("measure", (7,), (6,))
    elif change == "late-work":
        routed.append(w.Primitive("h", (0,))); ct.append(w.Primitive("h", (0,)))
    elif change == "unknown-key":
        routed[8] = w.Primitive("rx", (2,), (), (.6,))
    elif change == "unused-spec":
        specs += (replace(specs[0], word_id="word-01", key=("ry", (.5).hex())),)
    with pytest.raises(ValueError, match=match):
        list(w._walk(iter(routed), iter(ct), specs, limits=limits))


def test_segment_caps_reject_without_validated_partial_result(limits):
    routed, ct, specs = small_streams()
    with pytest.raises(ValueError, match="Routed stream"):
        list(w._walk(iter(routed), iter(ct), specs, limits=small_limit(limits, routed_operations=17)))
    with pytest.raises(ValueError, match="CT stream exceeds"):
        list(w._walk(iter(routed), iter(ct), specs, limits=small_limit(limits, clifford_t_operations=19)))


@pytest.fixture(scope="module")
def recovered():
    def forbidden(*args, **kwargs):
        pytest.fail("saved-word recovery must not synthesize or execute density")
    previous_handler = signal.getsignal(signal.SIGALRM)
    def timed_out(*args):
        raise TimeoutError("bounded saved-word recovery exceeded 60 seconds")
    signal.signal(signal.SIGALRM, timed_out); signal.alarm(60)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(old_compiler, "gridsynth_unitary", forbidden)
            monkeypatch.setattr(ir, "gridsynth_unitary", forbidden)
            monkeypatch.setattr(DensityMatrix, "evolve", forbidden)
            return w.recover_words(BUNDLE, limits=w.literal_limits())
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, previous_handler)


def test_actual_saved_words_phases_counts_and_full_operator_evidence(recovered, limits):
    assert [len(x.tokens) for x in recovered.words] == [462, 233, 225, 225, 230, 230, 244, 244]
    assert [x.occurrences for x in recovered.words] == [1266, 1688, 422, 422, 844, 844, 211, 211]
    assert recovered.summary["total_unique_word_operations"] == 2093
    assert recovered.summary["routed_operations"] == 24510
    assert recovered.summary["clifford_t_operations"] == 1677906
    assert recovered.summary["final_measurement_wires"] == [6, 1, 2, 3, 7, 4, 0, 5]
    assert recovered.summary["global_phase_factor_absolute_residual"] == 0.
    assert recovered.summary["complete_saved_stream_match"] is True
    assert set(recovered.source_artifacts) == {"manifest.json", "results.json", "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"}
    assert all(x.checks["manual_full_unitary_frobenius_residual"] <= w.WORD_OPERATOR_GUARD for x in recovered.words)
    assert all(x.checks["recorded_error_absolute_residual"] <= w.WORD_OPERATOR_GUARD for x in recovered.words)
    assert all(x.checks["comparison_ignores_global_phase"] is False for x in recovered.words)
    assert sum(x.occurrences for x in recovered.words) == 5908
    assert recovered.catalog_sha256 == recovered.recovery_sha256
    w.require_verified_recovery(recovered, limits=limits)


def test_actual_replay_checks_every_rotation_and_instrument_event(recovered, limits):
    count = operations = events = rotations = 0
    for segment in w.iter_verified_segments(recovered, limits=limits):
        count += 1; operations += segment.ct_operations
        events += segment.routed.name in {"reset", "measure"}
        rotations += segment.word_id is not None
        if segment.word_id is not None:
            assert all(x.qubits == (0,) for x in segment.tokens)
    assert (count, operations, events, rotations) == (24510, 1677906, 1489, 5908)


def test_actual_exports_are_phased_and_hash_bound(recovered, limits):
    outputs = w.recovery_artifacts(recovered, limits=limits)
    assert len(outputs) == 9
    catalog = json.loads(outputs["word-catalog.json"])
    assert catalog == recovered.as_record()
    assert catalog["new_synthesis_calls"] == 0 and catalog["candidate_full_density_executed"] is False
    assert "bundle" not in catalog
    for word in recovered.words:
        raw = outputs[f"words/{word.word_id}.qasm"]
        tokens, phase = w.parse_word_qasm(raw, limits=limits)
        assert tokens == word.tokens and phase.hex() == word.phase.hex()
        assert hashlib.sha256(raw).hexdigest() == word.qasm_sha256
        assert word.sequence_sha256 == w.sequence_sha256(tokens)
        assert Counter(x.name for x in tokens) == Counter(word.counts)
        assert np.linalg.norm(w.manual_word_matrix(tokens, phase)-w.manual_word_matrix(tokens, 0.), "fro") > 1.


@pytest.mark.parametrize("changes", [
    {"max_words": 7}, {"word_operations": 461}, {"total_word_operations": 2092},
    {"catalog_bytes": 100},
])
def test_recovered_evidence_respects_stricter_caller_bounds(changes, recovered, limits):
    with pytest.raises(ValueError, match="explicit"):
        w.require_verified_recovery(recovered, limits=small_limit(limits, **changes))


def test_changed_recorded_phase_cannot_pass_full_unitary_validation(recovered, limits):
    word = recovered.words[0]
    with pytest.raises(ValueError, match="full-unitary/recorded-error"):
        w._validate_word(replace(word.spec, phase=0.), word.tokens, limits)


def test_baseline_record_reads_saved_checks_as_separate_copy(recovered, limits):
    record = recovered.baseline_record(limits=limits)
    assert record["actual_synthesis"]["rotation_occurrences"] == 5908
    assert record["complete_workload"]["complete_trials"] == 15756
    assert "source_instrument_check" in record and "clifford_t_instrument_check" in record
    record["actual_synthesis"]["rotation_occurrences"] = 0
    assert w.baseline_record(recovered, limits=limits)["actual_synthesis"]["rotation_occurrences"] == 5908


def test_catalog_mutation_and_forged_recomputed_clone_reject(recovered, limits, monkeypatch):
    forged = replace(recovered, baseline_global_phase=0., recovery_sha256="")
    object.__setattr__(forged, "recovery_sha256", hashlib.sha256(w._json_bytes(forged._payload())).hexdigest())
    with pytest.raises(ValueError, match="process-issued"):
        w.recovery_artifacts(forged, limits=limits)
    monkeypatch.setitem(recovered.summary, "rotation_occurrences", 0)
    with pytest.raises(ValueError, match="catalog changed"):
        w.require_verified_recovery(recovered, limits=limits)


def test_recomputed_fingerprint_on_mutated_original_rejects(recovered, limits):
    original_phase, original_digest = recovered.baseline_global_phase, recovered.recovery_sha256
    try:
        object.__setattr__(recovered, "baseline_global_phase", 0.)
        object.__setattr__(recovered, "recovery_sha256", hashlib.sha256(w._json_bytes(recovered._payload())).hexdigest())
        with pytest.raises(ValueError, match="process-issued"):
            w.require_verified_recovery(recovered, limits=limits)
    finally:
        object.__setattr__(recovered, "baseline_global_phase", original_phase)
        object.__setattr__(recovered, "recovery_sha256", original_digest)


def test_nested_one_shot_tokens_and_oversized_catalog_reject_before_serialization(recovered, limits):
    word = recovered.words[0]; tokens = word.tokens
    try:
        object.__setattr__(word, "tokens", iter(tokens))
        with pytest.raises(ValueError, match="bounded concrete"):
            w.require_verified_recovery(recovered, limits=limits)
    finally:
        object.__setattr__(word, "tokens", tokens)
    original = word.checks
    try:
        object.__setattr__(word, "checks", {"oversized": "x"*8193})
        with pytest.raises(ValueError, match="oversized"):
            w.require_verified_recovery(recovered, limits=limits)
    finally:
        object.__setattr__(word, "checks", original)


def test_recorded_phase_subclass_rejects_before_method_invocation(recovered, limits):
    class PoisonFloat(float):
        def hex(self):
            pytest.fail("recorded phase method hook must not execute")
    original = recovered.baseline_global_phase
    try:
        object.__setattr__(recovered, "baseline_global_phase", PoisonFloat(.5))
        with pytest.raises(ValueError, match="concrete finite"):
            w.require_verified_recovery(recovered, limits=limits)
    finally:
        object.__setattr__(recovered, "baseline_global_phase", original)
    word = recovered.words[0]; original = word.spec
    try:
        object.__setattr__(word, "spec", replace(original, phase=PoisonFloat(.5)))
        with pytest.raises(ValueError, match="concrete finite"):
            w.require_verified_recovery(recovered, limits=limits)
    finally:
        object.__setattr__(word, "spec", original)


def test_source_or_module_identity_change_rejects(recovered, limits, monkeypatch):
    original = w._load_frozen
    def changed(*args):
        report, hashes, phase = original(*args)
        return report, {**hashes, "source-trial.qasm": "0"*64}, phase
    monkeypatch.setattr(w, "_load_frozen", changed)
    with pytest.raises(ValueError, match="baseline input changed"):
        w.require_verified_recovery(recovered, limits=limits)
    monkeypatch.setattr(w, "_module_sha256", lambda: "0"*64)
    with pytest.raises(ValueError, match="process-issued"):
        w.require_verified_recovery(recovered, limits=limits)


def test_fixed_manifest_attestation_precedes_any_stream_walk(tmp_path, limits, monkeypatch):
    (tmp_path/"manifest.json").write_text('{}')
    (tmp_path/"results.json").write_text('{}')
    monkeypatch.setattr(w, "iter_qasm", lambda *a, **k: pytest.fail("bad attestation must not parse streams"))
    with pytest.raises(ValueError, match="not the frozen"):
        w.recover_words(tmp_path, limits=limits)


@pytest.mark.parametrize("rows", [None, [], [None], [{"gate": "u", "angle_hex": (.5).hex()}]])
def test_missing_unknown_or_malformed_saved_word_specs_reject(rows, limits):
    with pytest.raises(ValueError):
        w._specs({"approximants": rows}, limits)
