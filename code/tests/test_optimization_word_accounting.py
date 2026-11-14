"""Saved-record and pure mocked checks; no baseline computation is invoked."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zlib

import pytest

from openquantum import optimization_word_accounting as accounting
from openquantum.optimization_word_transport import build_candidate


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets/generated/optimization/compiled-replay"


def _decoded(path: Path, cap: int) -> bytes:
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    raw = decoder.decompress(path.read_bytes(), cap + 1)
    assert len(raw) <= cap and decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data
    return raw


@pytest.fixture(scope="module")
def saved_classes():
    raw = _decoded(BUNDLE / "completed-replay.json.gz", 64*1024*1024)
    assert hashlib.sha256((BUNDLE / "completed-replay.json.gz").read_bytes()).hexdigest() == (
        "91fe2c423d66e682bcfba4b78142fa4192a60bf02609d4e6b526b946259d3c53")
    replay = json.loads(raw)
    classes = []
    decoded_bytes = 0
    for reference in replay["source_class_checkpoints"]:
        path = BUNDLE / reference["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == reference["sha256"]
        checkpoint = _decoded(path, 4*1024*1024)
        decoded_bytes += len(checkpoint)
        record = json.loads(checkpoint)["record"]
        candidate = build_candidate(record["clifford_t_operations"])
        segments = [dict(wire=segment["source_wire"], t_count=segment["t_count"],
                         tdg_count=segment["tdg_count"],
                         added_data_transport_cx=segment["added_data_transport_cx"],
                         removed_magic_transport_cx=segment["removed_magic_transport_cx"],
                         net_removed_cx=segment["net_removed_cx"])
                    for segment in candidate["segments"] if segment["shuttled"]]
        classes.append(dict(class_id=record["class_id"], shots=record["shots"],
                            baseline_event_counts=candidate["baseline_event_counts"],
                            candidate_event_counts=candidate["candidate_event_counts"],
                            segments=segments))
    assert decoded_bytes <= 64*1024*1024 and len(classes) == 54
    return classes


@pytest.fixture(scope="module")
def saved_accounting(saved_classes):
    return accounting.build_accounting(saved_classes)


@pytest.fixture(scope="module")
def frozen_cost_function():
    """Extract only a pinned deterministic function; never import its module."""
    raw = (BUNDLE / "instrument_resources.py").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == accounting.BACKEND_SHA256
    tree = ast.parse(raw)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_costs")
    # This function has no external calls: it refers only to profile fields,
    # builtin dict/int and the supplied Clifford set.
    scope = {"InstrumentProfile": SimpleNamespace,
             "CLIFFORD_T": {"h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx"}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "frozen-pure-cost-helper", "exec"), scope)
    return scope["_costs"]


@pytest.mark.parametrize("distance", [3, 15, 49])
@pytest.mark.parametrize("name", ["initial_reset", "reset", "measure", "cx", "trial_control",
                                  "h", "x", "y", "z", "s", "sdg",
                                  "t_q0", "t_q1", "t_q2", "t_q3",
                                  "tdg_q0", "tdg_q1", "tdg_q2", "tdg_q3"])
def test_macro_matches_pinned_pure_backend_function(distance, name, frozen_cost_function):
    profile = accounting.frozen_profile()
    actual = accounting.frozen_macro_cost(profile, distance, name, 123_456_789, 6)
    expected = frozen_cost_function(SimpleNamespace(**profile), distance, name, 123_456_789, 6)
    assert actual == expected


def test_closed_form_cx_and_source_word_savings():
    profile = accounting.frozen_profile()
    cx = accounting.frozen_macro_cost(profile, 15, "cx", 0, 6)
    assert cx == dict(time_ns=247_100, readouts=3, preparations=6, decisions=11)
    for wire in range(1, 4):
        for k in (3, 17):
            old = accounting.frozen_macro_cost(profile, 15, f"t_q{wire}", 987_654, 6)
            new = accounting.frozen_macro_cost(profile, 15, "t_q0", 987_654, 6)
            difference = {key: k*(old[key] - new[key]) - 6*wire*cx[key] for key in cx}
            assert difference == {key: 3*wire*(k - 2)*value for key, value in cx.items()}


def test_profile_and_machine_are_fresh_and_complete():
    profile = accounting.frozen_profile()
    profile["physical_error_rate"] = .001
    assert accounting.frozen_profile()["physical_error_rate"] == .0001
    saved = json.loads((BUNDLE / "physical-summary.json").read_text())
    assert accounting.frozen_profile() == saved["profile"]
    minimum_runtime = next(row for row in saved["selections"] if row["selection"] == "minimum_runtime")
    machine = accounting.frozen_machine()
    for key in ("data_code_distance", "native_factory_isa_sha256", "native_t_instruction_sha256",
                "factory_physical_qubits", "peak_physical_qubits"):
        assert machine[key] == minimum_runtime[key]
    machine["peak_physical_qubits"] = 0
    assert accounting.frozen_machine()["peak_physical_qubits"] == 243_358


def test_complete_saved_source_macro_totals_and_fixed_projection(saved_accounting):
    result = saved_accounting
    assert result["total_preparations"] == 82_032 and result["actual_magic_states"] == 182_646_240
    assert result["source_events_shot_weighted"]["baseline"]["cx"] == 2_542_992
    assert result["source_events_shot_weighted"]["candidate"]["cx"] == 13_315_056
    assert result["transform_counts_shot_weighted"] == {
        "selected_words": 1_142_208, "added_data_transport_cx": 10_772_064,
        "removed_magic_transport_cx": 799_368_672, "net_removed_cx": 788_596_608}
    assert result["macro_events_shot_weighted"] == {
        "baseline": {"readouts": 4_780_464_240, "preparations": 9_379_512_720,
                     "decisions": 22_072_802_800},
        "candidate": {"readouts": 2_414_674_416, "preparations": 4_647_933_072,
                      "decisions": 13_398_240_112}}
    assert result["savings"] == {
        "time_ns": 194_862_221_836_800, "readouts": 2_365_789_824,
        "preparations": 4_731_579_648, "decisions": 8_674_562_688}
    assert result["projected_candidate_serial_model_time_ns"] == 602_466_084_947_200
    assert result["physical_execution_enabled"] is False
    assert result["status"] == "source_ledger_only_not_native_physically_executed"
    assert result["prospective_native_caps"] == dict(trace_estimates=54, expected_parent_trace_estimates=49,
        factory_frontier_queries=0, pipeline_candidates=0, synthesis_calls=0, physical_wall_seconds=60,
        decoded_record_bytes=64*1024*1024, candidate_bundle_bytes=16*1024*1024)


def test_parent_groups_and_all_exact_allocations_remain_separate(saved_accounting):
    result = saved_accounting
    groups = result["parent_groups"]
    assert result["parent_group_count"] == len(groups) == 49
    assert result["candidate_distinct_event_cost_signatures"] == 45
    assert len({group["parent_physical_class_id"] for group in groups}) == 49
    assert sorted(len(group["source_class_ids"]) for group in groups) == [1]*44 + [2]*5
    assert result["global_factory_retry_cap"] == 6
    assert Fraction(result["factory_availability_bound_exact"]) == Fraction(182_646_240)*Fraction(".01")**6
    assert result["sampling_family_groups"] == 48
    assert Fraction(result["sampling_per_group_failure_exact"]) == Fraction(".04")/48
    assert sum(map(Fraction, result["allocations_exact"].values())) == Fraction(".05")
    for group in groups:
        signature = group["baseline_signature"]
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == group["parent_physical_class_id"]
        physical = Fraction(".0025")*group["shots"]/82_032
        factory = Fraction(".0025")*group["total_t_states"]/182_646_240
        assert Fraction(group["physical_failure_allocation_exact"]) == physical
        assert Fraction(group["factory_availability_allocation_exact"]) == factory
        assert Fraction.from_float(group["physical_failure_budget"]) <= physical
        assert Fraction(str(group["physical_failure_budget"])) <= physical
        assert Fraction(str(group["factory_availability_budget"])) <= factory
        assert group["workspace_initializations_per_preparation"] == 11
        assert group["trial_control_events_per_preparation"] == 1
        assert group["reserved_logical_patches"] == 15
    assert sum(Fraction(group["physical_failure_allocation_exact"]) for group in groups) == Fraction(".0025")
    assert sum(Fraction(group["factory_availability_allocation_exact"]) for group in groups) == Fraction(".0025")


def test_input_classes_and_segments_are_never_mutated(saved_classes):
    before = deepcopy(saved_classes)
    accounting.build_accounting(saved_classes)
    assert saved_classes == before


@pytest.mark.parametrize("field,value", [("physical_error_rate", .001), ("physical_gate_time_ns", 100.0),
                                         ("maximum_code_distance", 51), ("controller_failure_per_decision", 0.0)])
def test_profile_changes_rejected(saved_classes, field, value):
    profile = accounting.frozen_profile()
    profile[field] = value
    with pytest.raises(ValueError, match="frozen instrument profile"):
        accounting.build_accounting(saved_classes, profile)


@pytest.mark.parametrize("distance,name,factory,retry", [(True, "cx", 0, 6), (4, "cx", 0, 6),
    (51, "cx", 0, 6), (15, "t_q4", 0, 6), (15, "t_q00", 0, 6),
    (15, "t_q1", 0, 0), (15, "cx", -1, 6), (15, "cx", 0, 17), (15, "rz", 0, 6)])
def test_macro_bounds_reject_invalid_inputs(distance, name, factory, retry):
    with pytest.raises(ValueError):
        accounting.frozen_macro_cost(accounting.frozen_profile(), distance, name, factory, retry)


@pytest.mark.parametrize("corruption", ["missing_class", "duplicate_id", "shot_bool", "changed_cx",
                                       "changed_t", "bad_word_wire", "invented_magic", "wrong_word_count"])
def test_saved_workload_corruption_rejected(saved_classes, corruption):
    classes = deepcopy(saved_classes)
    if corruption == "missing_class":
        classes.pop()
    elif corruption == "duplicate_id":
        classes[1]["class_id"] = classes[0]["class_id"]
    elif corruption == "shot_bool":
        classes[0]["shots"] = True
    elif corruption == "changed_cx":
        classes[0]["candidate_event_counts"]["cx"] += 1
    elif corruption == "changed_t":
        classes[0]["candidate_event_counts"]["t_q0"] += 1
    elif corruption == "bad_word_wire":
        classes[0]["segments"][0]["wire"] = 0
    elif corruption == "invented_magic":
        classes[0]["segments"][0]["t_count"] = 10_000
    elif corruption == "wrong_word_count":
        classes[0]["segments"][0]["net_removed_cx"] += 1
    with pytest.raises(ValueError):
        accounting.build_accounting(classes)


def test_divergent_candidate_inside_old_group_rejected(saved_classes, saved_accounting):
    classes = deepcopy(saved_classes)
    old_group = next(group for group in saved_accounting["parent_groups"] if len(group["source_class_ids"]) == 2)
    item = next(c for c in classes if c["class_id"] == old_group["source_class_ids"][1])
    segment = item["segments"].pop(0)
    # Undo one declared shuttle in this one member, retaining a consistent
    # alternative event ledger.  It cannot silently inherit its sibling's cost.
    events = Counter(item["candidate_event_counts"])
    events["cx"] -= 6*segment["wire"]
    for gate in ("t", "tdg"):
        count = segment[gate + "_count"]
        events[gate + "_q0"] -= count
        events[f'{gate}_q{segment["wire"]}'] += count
    item["candidate_event_counts"] = dict(sorted((k, v) for k, v in events.items() if v))
    with pytest.raises(ValueError, match="diverge inside an immutable parent"):
        accounting.build_accounting(classes)


def test_identity_workload_cannot_be_called_a_reducing_candidate(saved_classes):
    classes = deepcopy(saved_classes)
    for item in classes:
        item["segments"] = []
        item["candidate_event_counts"] = deepcopy(item["baseline_event_counts"])
    with pytest.raises(ValueError, match="strictly reduce whole-workload"):
        accounting.build_accounting(classes)


def test_accounting_module_imports_only_standard_library():
    tree = ast.parse(Path(accounting.__file__).read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots <= {"__future__", "collections", "copy", "fractions", "hashlib", "json", "math"}
