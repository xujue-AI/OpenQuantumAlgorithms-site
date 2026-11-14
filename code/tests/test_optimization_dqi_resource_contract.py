"""New pure resource-contract boundaries, defined but not yet executed.

The source loader avoids the repository initializer. These tests use ordinary
JSON, integers, exact fractions, bounded toy files and fake native callbacks.
They do not call a producer, simulator, synthesis backend or QDK estimator.
"""
import builtins
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/openquantum/optimization_dqi_resource_contract.py"


def _load(name):
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


CORE = _load("_dqi_resource_pure_contract_tests")


def _ops(n=2):
    path = ROOT / f"assets/generated/optimization/dqi-structured/small/n{n}/operations.ndjson"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_ordinary_runtime_loads_no_backend_or_private_design(monkeypatch):
    original_import = builtins.__import__
    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"qiskit", "numpy", "qdk", "openquantum"}:
            pytest.fail("Pure contract requested a backend or package initializer")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    pure = _load("_dqi_resource_pure_import_guard")
    monkeypatch.setattr(pure, "load_design", lambda *a, **k: pytest.fail("Private design was requested"))
    cases = pure.accepted_cases(ROOT)
    assert [case["n"] for case in cases] == [2, 3]
    assert [len(case["operations"]) for case in cases] == [116, 336]
    assert len(cases[0]["validated_inputs"]) == 14


def test_caller_mutation_does_not_change_the_fixed_public_task():
    first = CORE.public_task()
    first["cases"][0]["fixed_complete_draws"] = 1
    first["frozen_profile"]["physical_error_rate"] = 1e-6
    assert CORE.public_task()["cases"][0]["fixed_complete_draws"] == 5
    assert CORE.public_task()["frozen_profile"]["physical_error_rate"] == 1e-4
    with pytest.raises(ValueError, match="changed"):
        CORE.check_task(first)


@pytest.mark.parametrize("mutation", ["extra", "boolean_count", "missing_budget"])
def test_task_fingerprint_rejects_scope_or_type_changes(mutation):
    task = CORE.public_task()
    if mutation == "extra":
        task["new_query_class"] = "factory_search"
    elif mutation == "boolean_count":
        task["limits"]["direct_native_trace_estimate_calls"] = True
    else:
        task["budget_per_case"].pop("realization_whole_fixed_draws")
    with pytest.raises(ValueError):
        CORE.check_task(task)


@pytest.mark.parametrize("field,value", [
    ("time", {"integer": 330001}),
    ("error_rate", {"float_hex": "0x1.0p-40"}),
    ("id", True),
])
def test_factory_identity_cannot_change_native_cost_fields(field, value):
    identity = CORE.frozen_identity()
    identity["native_t_instruction"][field] = value
    with pytest.raises(ValueError, match="changed"):
        CORE.check_identity(identity)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'])
def test_json_rejects_ambiguous_or_nonfinite_values(raw):
    with pytest.raises(ValueError):
        CORE.parse_json(raw)


def test_json_container_bound_rejects_before_children_are_collected():
    with pytest.raises(ValueError, match="node"):
        CORE.canonical_bytes([None] * (CORE.MAX_JSON_NODES + 1))


def test_json_byte_bound_also_applies_to_shared_large_strings():
    shared = "a" * 65536
    with pytest.raises(ValueError, match="byte envelope"):
        CORE.canonical_bytes([shared] * 20)


def test_pending_json_siblings_are_reserved_before_collecting_more_children():
    nested = [[None] * 4000, [None] * 4000]
    with pytest.raises(ValueError, match="node"):
        CORE.canonical_bytes(nested)


@pytest.mark.parametrize("relative", ["../x", "a/../x", "/x", "a//x", "./x", "a\\x"])
def test_path_rejection_precedes_file_open(monkeypatch, relative):
    monkeypatch.setattr(CORE.os, "open", lambda *a, **k: pytest.fail("Unexpected input open"))
    with pytest.raises(ValueError):
        CORE.bounded_read(ROOT, relative)


def test_declared_byte_overrun_is_rejected_before_input_open(monkeypatch):
    monkeypatch.setattr(CORE.os, "open", lambda *a, **k: pytest.fail("Unexpected input open"))
    with pytest.raises(ValueError, match="bytes"):
        CORE.bounded_read(ROOT, "LICENSE", expected_bytes=CORE.MAX_INPUT_FILE_BYTES + 1)


def test_regular_read_requires_exact_size_hash_and_no_symlinks(tmp_path):
    source = tmp_path / "source.txt"
    source.write_bytes(b"finite")
    digest = hashlib.sha256(b"finite").hexdigest()
    assert CORE.bounded_read(tmp_path, "source.txt", digest, 6, cap=6) == b"finite"
    with pytest.raises(ValueError, match="size"):
        CORE.bounded_read(tmp_path, "source.txt", digest, 5, cap=6)
    with pytest.raises(ValueError, match="hash"):
        CORE.bounded_read(tmp_path, "source.txt", "0" * 64, 6, cap=6)
    (tmp_path / "leaf.txt").symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        CORE.bounded_read(tmp_path, "leaf.txt", cap=6)
    (tmp_path / "alias").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        CORE.bounded_read(tmp_path, "alias/source.txt", cap=6)


def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "pipe"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular"):
        CORE.bounded_read(tmp_path, "pipe", cap=32)


@pytest.mark.parametrize("mutation", ["bool_wire", "nan_angle", "opaque", "late_reset", "readout"])
def test_source_stream_boundaries_reject_before_any_circuit_exists(mutation):
    records = _ops()
    if mutation == "bool_wire":
        records[0][1][0] = True
    elif mutation == "nan_angle":
        records[7][3][0] = math.nan
    elif mutation == "opaque":
        records[7][0] = "decoder_oracle"
    elif mutation == "late_reset":
        records[7] = ["reset", [0], [], []]
    else:
        records[-3][1][0] = 5
    with pytest.raises(ValueError):
        CORE.validate_operations(2, records)


def test_same_gate_counts_cannot_hide_changed_instruction_order():
    records = _ops()
    indices = [i for i, record in enumerate(records) if record[0] in {"h", "t"}]
    first = next(i for i in indices if records[i][0] == "h")
    second = next(i for i in indices if records[i][0] == "t")
    records[first], records[second] = records[second], records[first]
    with pytest.raises(ValueError, match="instruction order"):
        CORE.validate_operations(2, records)


def test_native_entry_is_spent_before_a_failing_callback_and_cannot_retry():
    counter = CORE.CounterGuard()
    key = CORE.ROTATION_TARGETS[0]
    entered = []
    def invoke():
        counter.charge("synthesis", key)
        entered.append(key)
        raise RuntimeError("simulated native failure")
    with pytest.raises(RuntimeError):
        invoke()
    with pytest.raises(ValueError, match="retry"):
        invoke()
    assert entered == [key]
    assert counter.snapshot()["synthesis_native_calls"] == 1


@pytest.mark.parametrize("kind,key", [
    ("factory_frontier", 2),
    ("qdk_top_level_estimate", 2),
    ("trace", True),
    ("synthesis", ("ry", "0x1.0p+0", CORE.EPSILON_FLOAT_HEX)),
    ("synthesis", ("ry", [], CORE.EPSILON_FLOAT_HEX)),
])
def test_unapproved_native_class_or_target_is_not_charged(kind, key):
    counter = CORE.CounterGuard(clock=lambda: pytest.fail("Unexpected native clock check"))
    with pytest.raises(ValueError):
        counter.charge(kind, key)
    assert counter.snapshot()["synthesis_native_calls"] == 0
    assert counter.snapshot()["direct_native_trace_estimate_calls"] == 0


def test_giant_native_key_rejects_before_frozenset_hashing(monkeypatch):
    class ForbiddenMembership:
        def __contains__(self, value):
            pytest.fail("Oversized native key was hashed")
    monkeypatch.setattr(CORE, "ALLOWED_SYNTHESIS_KEYS", ForbiddenMembership())
    with pytest.raises(ValueError):
        CORE.CounterGuard().charge("synthesis", ("ry", "x" * 100000, CORE.EPSILON_FLOAT_HEX))


def test_call_cap_and_exact_deadline_reject_before_entry():
    counter = CORE.CounterGuard(synthesis_limit=1)
    counter.charge("synthesis", CORE.ROTATION_TARGETS[0])
    with pytest.raises(ValueError, match="cap"):
        counter.charge("synthesis", CORE.ROTATION_TARGETS[1])
    assert counter.snapshot()["synthesis_native_calls"] == 1
    expired = CORE.CounterGuard(deadline=10, clock=lambda: 10)
    with pytest.raises(TimeoutError):
        expired.charge("trace", 2)
    assert expired.snapshot()["direct_native_trace_estimate_calls"] == 0


@pytest.mark.parametrize("kwargs", [
    {"synthesis_limit": 9}, {"trace_limit": 3}, {"trace_limit": True},
    {"deadline": math.inf}, {"deadline": True}, {"deadline": 1 << 100},
])
def test_native_limits_and_clock_cannot_be_widened(kwargs):
    with pytest.raises(ValueError):
        CORE.CounterGuard(**kwargs)


def test_only_all_eight_keys_and_both_tasks_can_be_complete():
    counter = CORE.CounterGuard()
    for key in CORE.ROTATION_TARGETS:
        counter.charge("synthesis", key)
    counter.charge("trace", 2)
    with pytest.raises(ValueError, match="Partial"):
        counter.require_complete()
    counter.charge("trace", 3)
    result = counter.require_complete()
    assert result["synthesis_native_calls"] == 8
    assert result["direct_native_trace_estimate_calls"] == 2
    assert result["factory_frontier_queries"] == result["qdk_top_level_estimate_calls"] == 0


@pytest.mark.parametrize("value", [True, math.nan, math.inf, -0.001, Fraction(1, 1 << 300)])
def test_whole_error_rejects_invalid_numeric_inputs(value):
    with pytest.raises(ValueError):
        CORE.guard_whole_error(2, value)


def test_per_gate_check_does_not_omit_the_whole_case_padding():
    with pytest.raises(ValueError, match="plus padding"):
        CORE.guard_whole_error(2, Fraction(1, 2000))
    with pytest.raises(ValueError, match="omitted"):
        CORE.guard_whole_error(2, Fraction(), padding=0)


@pytest.mark.parametrize("n,count", [(2, 5), (3, 7)])
def test_every_reused_rotation_is_charged_and_reported_bound_is_upward(n, count):
    exact = count * Fraction(1, 20000) + CORE.REFERENCE_PADDING
    represented = CORE.guard_occurrence_errors(n, [Fraction(1, 20000)] * count)
    ceiling = Fraction(1, 2000 if n == 2 else 1200)
    assert exact <= Fraction.from_float(represented) <= ceiling
    with pytest.raises(ValueError, match="Every source RY"):
        CORE.guard_occurrence_errors(n, [Fraction(1, 20000)] * (count + 1))


def test_whole_case_rounded_upper_cannot_cross_exact_eta():
    exact_eta = Fraction(1, 2000)
    with pytest.raises(ValueError, match="Represented"):
        CORE.guard_whole_error(2, exact_eta - CORE.REFERENCE_PADDING)
    safe_float = math.nextafter(float(exact_eta), 0)
    result = CORE.guard_whole_error(2, Fraction.from_float(safe_float) - CORE.REFERENCE_PADDING)
    assert Fraction.from_float(result) <= exact_eta


def test_budget_categories_cannot_borrow_unused_ideal_miss_slack():
    with pytest.raises(ValueError, match="separate"):
        CORE.whole_task_success_bound(2, Fraction(), Fraction(1, 400) + Fraction(1, 10**12), Fraction())


@pytest.mark.parametrize("n,eta", [(2, Fraction(1, 2000)), (3, Fraction(1, 1200))])
def test_task_reports_exact_fractions_and_conservative_probability_bounds(n, eta):
    result = CORE.whole_task_success_bound(n, eta, Fraction(1, 400), Fraction(1, 400))
    exact_miss = CORE.fraction_text(result["miss_upper_bound_fraction"])
    exact_success = CORE.fraction_text(result["success_lower_bound_fraction"])
    assert Fraction.from_float(result["miss_upper_bound"]) >= exact_miss
    assert Fraction.from_float(result["success_lower_bound"]) <= exact_success
    assert exact_miss + exact_success == 1 and exact_success >= Fraction(99, 100)
    assert result["conditional_on_faithful_bit_transfer_and_correct_scoring"] is True


def test_factory_retry_uses_new_whole_task_demand_and_exact_equality():
    equal = CORE.factory_retry_cap(2, 500)
    above = CORE.factory_retry_cap(2, 501)
    assert equal["states"] == 2500 and equal["retry_cap"] == 3
    assert CORE.fraction_text(equal["availability_failure_fraction"]) == Fraction(1, 400)
    assert above["retry_cap"] == 4
    assert above["worst_case_pipeline_attempts"] == 2505 * 4


@pytest.mark.parametrize("n,count", [(4, 1), (True, 1), (2, True), (2, 20001)])
def test_factory_scope_and_t_demand_reject_before_retry_search(n, count):
    with pytest.raises(ValueError):
        CORE.factory_retry_cap(n, count)
