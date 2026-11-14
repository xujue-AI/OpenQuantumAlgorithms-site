"""Tiny phase-sensitive rewrite checks; no full candidate or native work."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import math

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator
from qiskit.circuit.library import HGate, XGate

from openquantum import instrument_resources as ir
from openquantum.simulation import bounded_density_instrument as legacy_density
from openquantum.simulation import lindblad_port_rewrite as p
from openquantum.simulation import lindblad_port_words as w


@pytest.fixture
def limits():
    return p.PortRewriteLimits(8, 4096, 4096, 4096, 4096)


@pytest.fixture
def word():
    tokens = tuple(w.Primitive(name, (0,)) for name in ("h", "t", "s", "h", "tdg"))
    spec = w.WordSpec("tiny-word", ("rz", (.2).hex()), .37,
                      (("h", 2), ("s", 1), ("t", 1), ("tdg", 1)), 1e-3, 1e-2)
    return w.PhasedWord(spec, tokens, 1, {})


def segments(word, wire, *, unsafe=False, readout=None):
    schedule = [w.Primitive("barrier", tuple(reversed(range(8))))]
    initialized = [wire] if unsafe else list(range(8))
    schedule += [w.Primitive("reset", (q,)) for q in initialized]
    schedule.append(w.Primitive(word.key[0], (wire,), params=(float.fromhex(word.key[1]),)))
    if unsafe:
        schedule += [w.Primitive("reset", (q,)) for q in range(8) if q != wire]
    schedule += [w.Primitive("cx", (3, 4)), w.Primitive("h", (2,)),
                 w.Primitive("barrier", tuple(range(8)))]
    targets = list(range(8)) if readout is None else readout
    for bit, target in enumerate(targets):
        schedule.append(w.Primitive("measure", (target,), (bit,)))
        if bit == 2:
            schedule.append(w.Primitive("barrier", tuple(range(8))))
    result, ct_start = [], 0
    for index, token in enumerate(schedule):
        native = word.tokens if token.rotation_key is not None else (token,)
        result.append(w.VerifiedSegment(index, ct_start, token,
                                       word.word_id if token.rotation_key is not None else None,
                                       native, len(native)))
        ct_start += len(native)
    return tuple(result)


def circuit(tokens, phase=.19):
    value = QuantumCircuit(8, 8)
    value.global_phase = phase
    for token in tokens:
        if token.name == "measure":
            value.measure(token.qubits[0], token.clbits[0])
        elif token.name == "barrier":
            value.barrier(*token.qubits)
        else:
            getattr(value, token.name)(*token.qubits)
    return value


def fixture_recipe(plan, limits):
    record = {"schema_version": 1, "algorithm_version": p.ALGORITHM_VERSION,
              "plan": plan, "catalog_sha256": "a"*64,
              "source_sha256": p._source_sha256(), "source_artifacts": {"tiny": "b"*64},
              "global_phase_binary64_hex": (.19).hex(),
              "phase_rule": "bare words plus authoritative baseline global phase exactly once",
              "limits": asdict(limits)}
    raw = p._json_bytes(record)
    return p.FrozenPortRecipe(raw, hashlib.sha256(raw).hexdigest())


def patch_tiny_source(monkeypatch, source, recipe):
    monkeypatch.setattr(p, "require_verified_recipe", lambda *a, **k: recipe.as_record())
    monkeypatch.setattr(w, "iter_verified_segments", lambda *a, **k: iter(source))


@pytest.mark.parametrize("wire", range(8))
def test_full_eight_wire_phase_sensitive_conjugation(word, limits, wire):
    source = segments(word, wire)
    plan = p._plan_segments(source, catalog=(word,), limits=limits)
    row = plan["rotation_blocks"][0]
    emitted = tuple(p._emit_segments(source, plan["rotation_blocks"], limits=limits))
    assert len(emitted) < 4096
    assert row["forward_swap_edges"] == [[i, i+1] for i in range(wire-1, -1, -1)]
    assert row["reverse_swap_edges"] == [[i, i+1] for i in range(wire)]
    assert row["initialized_mask"] == 255
    assert row["target_wire"] == 0
    assert row["added_cx"] == plan["added_cx"] == 6*wire
    assert plan["planned_t_wire_counts"] == [2, 0, 0, 0, 0, 0, 0, 0]
    actual_word = [token for token in emitted if token.name not in {"reset", "measure", "barrier"}]
    baseline = []
    for segment in source:
        if segment.word_id:
            baseline += [w.Primitive(t.name, (wire,)) for t in word.tokens]
        elif segment.routed.name not in {"reset", "measure", "barrier"}:
            baseline += list(segment.tokens)
    expected_matrix = Operator(circuit(baseline)).data
    actual_matrix = Operator(circuit(actual_word)).data
    assert np.linalg.norm(expected_matrix-actual_matrix, "fro") < 1e-12
    assert np.linalg.norm(expected_matrix-Operator(circuit(actual_word, .19+word.phase)).data, "fro") > 1
    assert all(abs(t.qubits[0]-t.qubits[1]) == 1 for t in emitted if t.name == "cx")
    assert [t for t in emitted if t.name in {"reset", "measure", "barrier"}] == [s.routed for s in source if s.routed.name in {"reset", "measure", "barrier"}]


def test_unsafe_initialized_path_retains_exact_original_word(word, limits):
    source = segments(word, 7, unsafe=True)
    plan = p._plan_segments(source, catalog=(word,), limits=limits)
    row = plan["rotation_blocks"][0]
    assert row["initialized_mask"] == 128
    assert row["eligible"] is False
    assert row["decision"] == "uninitialized_path_fallback"
    assert row["target_wire"] == 7
    assert row["forward_swap_edges"] == row["reverse_swap_edges"] == []
    assert plan["added_cx"] == 0
    assert plan["fallback_rotation_blocks"] == 1
    emitted = tuple(p._emit_segments(source, plan["rotation_blocks"], limits=limits))
    assert emitted[2:7] == tuple(w.Primitive(t.name, (7,)) for t in word.tokens)
    assert plan["baseline_t_wire_counts"] == plan["planned_t_wire_counts"] == [0]*7+[2]


def test_barriers_do_not_initialize_path(word, limits):
    source = segments(word, 7, unsafe=True)
    assert p._plan_segments(source, catalog=(word,), limits=limits)["fallback_rotation_blocks"] == 1
    bad = list(source)
    bad[1] = replace(bad[1], routed=w.Primitive("h", (7,)), tokens=(w.Primitive("h", (7,)),))
    with pytest.raises(ValueError, match="before explicit initialization"):
        p._plan_segments(bad, catalog=(word,), limits=limits)


def test_complete_event_chronology_and_readout_map(word, limits):
    readout = [6, 1, 2, 3, 7, 4, 0, 5]
    source = segments(word, 6, readout=readout)
    plan = p._plan_segments(source, catalog=(word,), limits=limits)
    actual = p._actual_summary(p._emit_segments(source, plan["rotation_blocks"], limits=limits), limits=limits)
    assert plan["final_measurement_wires"] == actual["final_measurement_wires"] == readout
    assert actual["event_sha256"] == plan["event_sha256"]
    assert actual["actual_operations"] == plan["baseline_operations"]+36
    assert actual["gate_counts"] == plan["planned_gate_counts"]
    assert len(plan["source_event_records"]) == 8+8+3


@pytest.mark.parametrize("mutation", ["measure_duplicate", "classical_order", "work_after_measure", "reset_after_measure"])
def test_bad_final_event_mapping_rejects(word, limits, mutation):
    source = list(segments(word, 3))
    indexes = [i for i, s in enumerate(source) if s.routed.name == "measure"]
    index = indexes[-1] if mutation in {"measure_duplicate", "classical_order"} else indexes[1]
    original = source[index]
    if mutation == "measure_duplicate":
        token = replace(original.routed, qubits=(0,))
    elif mutation == "classical_order":
        token = replace(original.routed, clbits=(0,))
    else:
        token = w.Primitive("reset" if mutation == "reset_after_measure" else "h", (1,))
    source[index] = replace(original, routed=token, tokens=(token,))
    with pytest.raises(ValueError, match="final readout|after final"):
        p._plan_segments(source, catalog=(word,), limits=limits)


@pytest.mark.parametrize("mutation", ["routed_index", "ct_start", "ct_operations", "word_id", "word_tokens", "nonrotation"])
def test_changed_segment_identity_rejects(word, limits, mutation):
    source = list(segments(word, 3))
    index = next(i for i, s in enumerate(source) if s.word_id is not None)
    original = source[index]
    if mutation in {"routed_index", "ct_start", "ct_operations"}:
        source[index] = replace(original, **{mutation: getattr(original, mutation)+1})
    elif mutation == "word_id":
        source[index] = replace(original, word_id="unrecovered")
    elif mutation == "word_tokens":
        source[index] = replace(original, tokens=(w.Primitive("x", (0,)),)+original.tokens[1:])
    else:
        source[1] = replace(source[1], tokens=(w.Primitive("reset", (7,)),))
    with pytest.raises(ValueError):
        p._plan_segments(source, catalog=(word,), limits=limits)


@pytest.mark.parametrize("field,value", [("max_qubits", 7), ("max_qubits", True),
    ("baseline_operations", 2_000_001), ("candidate_operations", 1_926_043),
    ("added_cx", 248_137), ("routed_operations", 100_001),
    ("recipe_bytes", 8*1024*1024+1), ("candidate_operations", 0), ("added_cx", False)])
def test_mandatory_limit_contract_and_hard_ceilings(limits, field, value):
    values = asdict(limits)
    values[field] = value
    with pytest.raises(ValueError):
        p.PortRewriteLimits(**values)


def test_limits_revalidate_mutation_and_reject_subclass(word, limits):
    class Forged(p.PortRewriteLimits):
        def __post_init__(self):
            pass
    with pytest.raises(ValueError, match="explicit PortRewriteLimits"):
        p._plan_segments(segments(word, 0), catalog=(word,), limits=Forged(8, 9_000_000, 9_000_000, 9_000_000, 9_000_000))
    object.__setattr__(limits, "added_cx", 248_137)
    with pytest.raises(ValueError, match="added_cx"):
        p._plan_segments(segments(word, 0), catalog=(word,), limits=limits)


def test_mutated_limits_reject_before_recursive_deepcopy(limits):
    class MustNotCopy:
        def __deepcopy__(self, memo):
            pytest.fail("limit validation must precede any deep copy")
    object.__setattr__(limits, "candidate_operations", MustNotCopy())
    with pytest.raises(ValueError, match="candidate_operations"):
        p._limits(limits)


@pytest.mark.parametrize("field", ["baseline_operations", "routed_operations", "candidate_operations", "added_cx"])
def test_whole_plan_cap_rejects(word, limits, field):
    values = asdict(limits)
    values[field] = 1
    with pytest.raises(ValueError, match="cap"):
        p._plan_segments(segments(word, 7), catalog=(word,), limits=p.PortRewriteLimits(**values))


def test_next_complete_segment_cap_rejects_before_word_emission(word, limits):
    source = segments(word, 7)
    plan = p._plan_segments(source, catalog=(word,), limits=limits)
    values = asdict(limits)
    values["candidate_operations"] = 10
    generator = p._emit_segments(source, plan["rotation_blocks"], limits=p.PortRewriteLimits(**values))
    seen = []
    with pytest.raises(ValueError, match="Next complete segment"):
        while True:
            seen.append(next(generator))
    assert len(seen) == 9  # barrier and initial resets, no partial word or SWAP


def test_catalog_flat_bare_word_guards(word):
    bad = replace(word, tokens=(w.Primitive("cx", (0, 1)),))
    with pytest.raises(ValueError, match="one-wire"):
        p._catalog((bad,))
    with pytest.raises(ValueError, match="unique"):
        p._catalog((word, word))
    bad = replace(word, spec=replace(word.spec, phase=math.nan))
    with pytest.raises(ValueError, match="finite"):
        p._catalog((bad,))
    bad = replace(word, tokens=(w.Primitive("t", (0,), params=(.1,)),))
    with pytest.raises(ValueError, match="no parameters"):
        p._catalog((bad,))


def test_immutable_recipe_roundtrip_copy_and_hash_guards(word, limits, monkeypatch):
    plan = p._plan_segments(segments(word, 2), catalog=(word,), limits=limits)
    recipe = fixture_recipe(plan, limits)
    loaded = p.FrozenPortRecipe.from_record(recipe.as_record(), limits=limits)
    assert loaded == recipe
    copied = loaded.as_record()
    copied["plan"]["added_cx"] = 0
    assert loaded.as_record()["plan"]["added_cx"] == 12
    with pytest.raises(ValueError, match="digest"):
        p.FrozenPortRecipe.from_record(copied, limits=limits)
    monkeypatch.setattr(p, "build_recipe", lambda *a, **k: recipe)
    assert p.require_verified_recipe(loaded, None, limits=limits, word_limits=None) == recipe.as_record()
    mutated = p.FrozenPortRecipe(recipe.canonical_json+b" ", recipe.recipe_sha256)
    with pytest.raises(ValueError, match="digest changed"):
        p.require_verified_recipe(mutated, None, limits=limits, word_limits=None)
    copied["recipe_sha256"] = hashlib.sha256(p._json_bytes({k: v for k, v in copied.items() if k != "recipe_sha256"})).hexdigest()
    forged = p.FrozenPortRecipe.from_record(copied, limits=limits)
    with pytest.raises(ValueError, match="current source-only plan"):
        p.require_verified_recipe(forged, None, limits=limits, word_limits=None)


def test_recipe_byte_cap_and_noncanonical_payload_reject(word, limits, monkeypatch):
    plan = p._plan_segments(segments(word, 2), catalog=(word,), limits=limits)
    recipe = fixture_recipe(plan, limits)
    values = asdict(limits)
    values["recipe_bytes"] = 16
    with pytest.raises(ValueError, match="byte cap"):
        p.FrozenPortRecipe.from_record(recipe.as_record(), limits=p.PortRewriteLimits(**values))
    raw = recipe.canonical_json+b" "
    forged = p.FrozenPortRecipe(raw, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(p, "build_recipe", lambda *a, **k: recipe)
    with pytest.raises(ValueError, match="current source-only plan"):
        p.require_verified_recipe(forged, None, limits=limits, word_limits=None)
    recursive = {"recipe_sha256": "a"*64}
    recursive["self"] = recursive
    with pytest.raises(ValueError):
        p.FrozenPortRecipe.from_record(recursive, limits=limits)
    with pytest.raises(ValueError, match="oversized"):
        p.FrozenPortRecipe.from_record({"recipe_sha256": "a"*64, "large": "x"*8193}, limits=limits)
    assert p._bounded_record_bytes({"exact_numerator": 1 << 256}, 8192)
    with pytest.raises(ValueError, match="oversized"):
        p._bounded_record_bytes({"oversized_integer": 1 << 1025}, 8192)
    class InfiniteList(list):
        def __iter__(self):
            pytest.fail("untrusted container subclass must reject before iteration")
    with pytest.raises(ValueError, match="unsupported"):
        p.FrozenPortRecipe.from_record({"recipe_sha256": "a"*64, "list": InfiniteList()}, limits=limits)
    with pytest.raises(ValueError, match="bounded concrete"):
        p.FrozenPortRecipe.from_record({str(i): None for i in range(4097)}, limits=limits)


def test_imported_source_identity_must_match_current_file(monkeypatch):
    original = w.bounded_file_sha256
    monkeypatch.setattr(w, "bounded_file_sha256", lambda path, **kwargs:
                        "0"*64 if str(path) == p.__file__ else original(path, **kwargs))
    with pytest.raises(ValueError, match="changed since"):
        p._source_sha256()


def test_full_generation_gate_precedes_any_input_read(limits, monkeypatch):
    monkeypatch.setattr(p, "require_verified_recipe", lambda *a, **k: pytest.fail("gate must reject before any source walk"))
    for flag in (False, None, 1, "yes"):
        with pytest.raises(ValueError, match="explicit authorization"):
            p.materialize_candidate(None, None, limits=limits, word_limits=None, allow_full_materialization=flag)
        with pytest.raises(ValueError, match="explicit authorization"):
            next(p.iter_port_rewrite(None, None, limits=limits, word_limits=None, allow_full_materialization=flag))


def test_materialized_tiny_candidate_exact_stream_report_and_phase(word, limits, monkeypatch):
    source = segments(word, 5, readout=p.BASELINE_READOUT)
    plan = p._plan_segments(source, catalog=(word,), limits=limits)
    recipe = fixture_recipe(plan, limits)
    patch_tiny_source(monkeypatch, source, recipe)
    candidate = p.materialize_candidate(recipe, None, limits=limits, word_limits=None, allow_full_materialization=True)
    assert len(candidate.circuit.data) < 4096
    assert float(candidate.circuit.global_phase).hex() == (.19).hex()
    assert candidate.report["candidate_materialized"] is True
    assert candidate.report["full_density_executed"] is False
    assert candidate.report["added_cx"] == 30
    assert candidate.report["t_wire_counts"] == [2]+[0]*7
    assert p.require_verified_candidate(candidate, recipe, None, limits=limits, word_limits=None) == candidate.report
    copy = p.require_verified_candidate(candidate, recipe, None, limits=limits, word_limits=None)
    copy["t_wire_counts"][0] = 0
    assert candidate.report["t_wire_counts"][0] == 2


@pytest.mark.parametrize("mutation", ["zero_phase", "double_phase", "gate", "hidden", "spoofed_base", "instance_matrix", "instance_array", "fake_matrix_method", "fake_array_method", "fake_callable_method", "class_matrix", "class_array", "report", "truncated"])
def test_tiny_candidate_tamper_rejects(word, limits, monkeypatch, mutation):
    source = segments(word, 3)
    recipe = fixture_recipe(p._plan_segments(source, catalog=(word,), limits=limits), limits)
    patch_tiny_source(monkeypatch, source, recipe)
    candidate = p.materialize_candidate(recipe, None, limits=limits, word_limits=None, allow_full_materialization=True)
    local_index = next(i for i, item in enumerate(candidate.circuit.data) if item.operation.name == "h")
    if mutation in {"zero_phase", "double_phase"}:
        candidate.circuit.global_phase = 0 if mutation == "zero_phase" else .19+word.phase
    elif mutation == "gate":
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=XGate())
    elif mutation == "hidden":
        hidden = Gate("h", 1, [])
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=hidden)
    elif mutation == "spoofed_base":
        class SpoofedH(HGate):
            @property
            def base_class(self):
                return HGate
            def to_matrix(self):
                return XGate().to_matrix()
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=SpoofedH())
    elif mutation == "instance_matrix":
        modified = HGate().to_mutable()
        modified.to_matrix = lambda: XGate().to_matrix()
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=modified)
    elif mutation == "instance_array":
        modified = HGate().to_mutable()
        modified.__array__ = lambda *a, **k: XGate().to_matrix()
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=modified)
    elif mutation in {"fake_matrix_method", "fake_array_method", "fake_callable_method"}:
        modified = HGate().to_mutable()
        class FakeMethod:
            def __call__(self, *args, **kwargs):
                return XGate().to_matrix()
        fake = FakeMethod() if mutation == "fake_callable_method" else lambda *a, **k: XGate().to_matrix()
        fake.__self__ = modified
        fake.__func__ = p._ARRAY_METHODS["h"] if mutation == "fake_array_method" else p._MATRIX_METHODS["h"]
        setattr(modified, "__array__" if mutation == "fake_array_method" else "to_matrix", fake)
        candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=modified)
    elif mutation == "class_matrix":
        monkeypatch.setattr(HGate, "to_matrix", lambda self: XGate().to_matrix())
    elif mutation == "class_array":
        monkeypatch.setattr(HGate, "__array__", lambda *a, **k: XGate().to_matrix())
    elif mutation == "report":
        candidate.report["added_cx"] = 0
    else:
        candidate.circuit.data.pop()
    with pytest.raises(ValueError):
        p.require_verified_candidate(candidate, recipe, None, limits=limits, word_limits=None)


def test_standard_primitive_matrix_remains_authoritative_over_cached_definition(word, limits, monkeypatch):
    source = segments(word, 3)
    recipe = fixture_recipe(p._plan_segments(source, catalog=(word,), limits=limits), limits)
    patch_tiny_source(monkeypatch, source, recipe)
    candidate = p.materialize_candidate(recipe, None, limits=limits, word_limits=None, allow_full_materialization=True)
    local_index = next(i for i, item in enumerate(candidate.circuit.data) if item.operation.name == "h")
    standard = HGate().to_mutable()
    cached = QuantumCircuit(1)
    cached.x(0)
    standard.definition = cached
    candidate.circuit.data[local_index] = candidate.circuit.data[local_index].replace(operation=standard)
    # The unchanged direct-matrix checker executes canonical to_matrix for a
    # standard primitive, independently of its optional decomposition cache.
    assert np.array_equal(standard.to_matrix(), HGate().to_matrix())
    assert np.array_equal(Operator(standard).data, HGate().to_matrix())
    assert p.require_verified_candidate(candidate, recipe, None, limits=limits, word_limits=None) == candidate.report


def test_literal_recipe_rejects_tiny_fixture_before_attestation(word, limits, monkeypatch):
    from types import SimpleNamespace
    tiny = SimpleNamespace(words=(word,))
    monkeypatch.setattr(w, "iter_verified_segments", lambda *a, **k: iter(segments(word, 0)))
    with pytest.raises(ValueError, match="literal frozen n4"):
        p.build_recipe(tiny, limits=limits, word_limits=None)


def test_historical_limit_defaults_remain_unchanged():
    assert ir.InstrumentLimits().max_qubits == 5
    assert legacy_density.DenseInstrumentLimits().max_qubits == 5
    with pytest.raises(ValueError):
        ir.InstrumentLimits(synthesized_operations=2_000_000)
    with pytest.raises(ValueError):
        legacy_density.DenseInstrumentLimits(max_qubits=8)
    with pytest.raises(TypeError):
        p.PortRewriteLimits()
    with pytest.raises(TypeError):
        p.build_recipe(None)
