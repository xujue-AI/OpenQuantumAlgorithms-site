"""Saved plans, ISA construction and mocks only; no Native Trace estimates."""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from openquantum import optimization_word_native as native


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "docs/coordination/optimization-eighth-candidate"


@pytest.fixture(scope="module")
def plan():
    return native.plan_fixed_requests(CANDIDATE, ROOT)


class MockBlock:
    def __init__(self, trace, count):
        self.trace, self.count = trace, count

    def add_operation(self, identifier, qubits):
        self.trace.gate_counts[identifier] += self.count
        self.trace.rows.append([identifier, self.count, qubits])


class MockTrace:
    """Construction mock; its estimate method must never be used."""
    def __init__(self, compute_qubits):
        self.compute_qubits = compute_qubits
        self.base_error = 0.0
        self.resource_states = {}
        self.gate_counts = Counter()
        self.rows = []

    def add_block(self, repetitions=1):
        return MockBlock(self, repetitions)

    def increment_resource_state(self, identifier, count):
        self.resource_states[identifier] = count

    def increment_base_error(self, value):
        self.base_error += value

    def to_json(self):
        return json.dumps(dict(compute_qubits=self.compute_qubits, rows=self.rows,
                               resource_states=self.resource_states, base_error_hex=self.base_error.hex()),
                          sort_keys=True, separators=(",", ":"))

    def estimate(self, *_args, **_kwargs):
        raise AssertionError("No Native or mock Trace.estimate may be invoked by this suite")


class PureMockInstruction:
    """Python cost-record stand-in; never constructs a Native ISA object."""
    def __init__(self, record):
        self.record = deepcopy(record)
        self.id, self.encoding, self.arity = record["id"], record["encoding"], record["arity"]

    def time(self, _arity=None):
        return self.record["time"]["integer"]

    def space(self, _arity=None):
        return self.record["space"]["integer"]

    def error_rate(self, _arity=None):
        return float.fromhex(self.record["error_rate"]["float_hex"])

    expect_time = time
    expect_space = space
    expect_error_rate = error_rate


class PureMockISA:
    def __init__(self, records):
        self.instructions = {record["id"]: PureMockInstruction(record) for record in records}

    def __iter__(self):
        return iter(self.instructions.values())

    def __getitem__(self, key):
        return self.instructions[key]


class ErrorComposition:
    """Pure enum-name stand-in used only by metadata-scope regressions."""
    def __str__(self):
        return "ErrorComposition.UnionBound"


def pure_mock_payload(plan, request):
    trace = MockTrace(15)
    for row in request["macro_ledger_per_preparation"]:
        trace.add_block(row["count"]).add_operation(row["instruction_id"], list(range(15)))
    trace.increment_resource_state(1028, request["t_states_per_preparation"])
    trace.increment_base_error(float.fromhex(request["native_base_error_hex"]))
    records = native._expected_isa_records(request, plan["native_identity"])
    isa = PureMockISA(records)
    enriched = deepcopy(request)
    enriched.update(native_trace_json=trace.to_json(),
        native_trace_json_sha256=hashlib.sha256(trace.to_json().encode()).hexdigest(),
        native_isa_cost_records=records, native_isa_cost_records_sha256=native._digest(records),
        error_composition={"type": "ErrorComposition", "value": "ErrorComposition.UnionBound"},
        native_factory_isa_sha256=native.ISA_SHA256, native_t_instruction_sha256=native.T_SHA256,
        native_compute_qubits=15,
        native_gate_counts=[dict(instruction_id=key, count=count) for key, count in sorted(trace.gate_counts.items())],
        native_resource_states=[dict(instruction_id=1028, count=request["t_states_per_preparation"])],
        native_base_error_hex=request["native_base_error_hex"])
    native.validate_restored_input(plan, request, enriched)
    return dict(trace=trace, isa=isa, composition=ErrorComposition(),
                max_error=float.fromhex(request["native_max_error_hex"]), request=enriched)


@pytest.fixture
def mocked_trace(monkeypatch):
    import qdk.qre
    monkeypatch.setattr(qdk.qre, "Trace", MockTrace)


def producer(payload):
    request = payload["request"]
    return SimpleNamespace(qubits=243_358, runtime=request["runtime_per_preparation_ns"],
        error=float.fromhex(request["expected_error_per_preparation_hex"]), isa=payload["isa"],
        error_composition=payload["composition"], properties={},
        factories={1028: SimpleNamespace(copies=1, runs=request["t_states_per_preparation"],
            states=request["t_states_per_preparation"], error_rate=float.fromhex("0x1.88ea67f531f32p-56"))})


def test_saved_plan_exact_workload_machine_and_true_per_preparation_ceiling(plan):
    assert plan["execution_enabled"] is False
    assert len(plan["requests"]) == 54 and len(plan["raw_parent_allocation_records"]) == 49
    totals = plan["totals"]
    assert totals == dict(classes=54, parent_groups=49, preparations=82_032, t_states=182_646_240,
        serial_runtime_ns=602_466_084_947_200, data_exposure_cycles=602_466_111_572,
        projected_physical_failure=0.00010525086320790465,
        macro_counts=dict(readouts=2_414_674_416, preparations=4_647_933_072, decisions=13_398_240_112))
    incorrectly_weighted_ceiling = -(-totals["serial_runtime_ns"] // 1000)
    assert totals["data_exposure_cycles"] - incorrectly_weighted_ceiling == 26_624
    assert sum(request["runtime_per_preparation_ns"] % 1000 != 0 for request in plan["requests"]) == 45
    for request in plan["requests"]:
        assert request["code_cycles_per_preparation"] == -(-request["runtime_per_preparation_ns"] // 1000)
        assert request["handoff_ns_per_t"] == 783_100 and request["handoff_cycles_per_t"] == 784
        events = sum(row["count"]*float.fromhex(row["event_failure_model_hex"])
                     for row in request["macro_ledger_per_preparation"])
        memory = float.fromhex(request["memory_failure_per_preparation_hex"])
        hold = float.fromhex(request["handoff_failure_per_preparation_hex"])
        expected = memory + hold + events + request["t_states_per_preparation"]*float.fromhex("0x1.88ea67f531f32p-56")
        assert expected.hex() == request["expected_error_per_preparation_hex"]
        assert Fraction.from_float(float.fromhex(request["native_max_error_hex"])) <= Fraction(request["native_max_error_target_exact"])
    assert plan["fixed_machine"]["native_factory_isa_sha256"] == native.ISA_SHA256
    assert plan["fixed_machine"]["native_t_instruction_sha256"] == native.T_SHA256


def test_individual_requests_retain_all_parent_exact_and_recorded_float_allocations(plan):
    assert native._digest(plan["raw_parent_allocation_records"]) == native.PARENT_ALLOCATION_SHA256
    for parent in plan["raw_parent_allocation_records"]:
        children = [row for row in plan["requests"] if row["parent_physical_class_id"] == parent["physical_class_id"]]
        assert sum(row["shots"] for row in children) == parent["shots"]
        assert sum(row["total_t_states"] for row in children) == parent["total_t_states"]
        assert sum(Fraction(row["physical_failure_allocation_exact"]) for row in children) == Fraction(parent["physical_failure_allocation_exact"])
        assert sum(Fraction(row["factory_availability_allocation_exact"]) for row in children) == Fraction(parent["factory_availability_allocation_exact"])
        ceilings = sum(Fraction.from_float(float.fromhex(row["native_max_error_hex"]))*row["shots"] for row in children)
        assert ceilings <= Fraction.from_float(parent["physical_failure_budget"])
        availability = sum(Fraction(row["factory_availability_failure_exact"]) for row in children)
        assert availability <= Fraction(str(parent["factory_availability_budget"]))


def test_real_instruction_constructor_roundtrip_without_trace_or_factory_search(plan):
    restored = native.restore_recorded_factory_isa(plan["native_identity"])
    records = [native.instruction_record(value) for value in sorted(restored["isa"], key=lambda value: value.id)]
    assert records == plan["native_identity"]["native_factory_isa"]
    assert native._digest(records) == native.ISA_SHA256
    assert native._digest(native.instruction_record(restored["isa"][1028])) == native.T_SHA256
    assert "fresh provenance graph" in restored["identity_scope"]


def test_mock_trace_construction_binds_full_input_and_used_arity_isa(plan, mocked_trace):
    request = plan["requests"][0]
    payload = native.restore_native_request(plan, request)
    assert isinstance(payload["trace"], MockTrace)
    enriched = payload["request"]
    native.validate_restored_input(plan, request, enriched)
    assert hashlib.sha256(enriched["native_trace_json"].encode()).hexdigest() == enriched["native_trace_json_sha256"]
    assert native._digest(enriched["native_isa_cost_records"]) == enriched["native_isa_cost_records_sha256"]
    assert len(enriched["native_isa_cost_records"]) == len(request["macro_ledger_per_preparation"]) + 1
    assert payload["max_error"].hex() == request["native_max_error_hex"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "encoding_bool", "time", "space", "error_bits"])
def test_cost_identity_changes_rejected_before_native_construction(plan, mutation):
    identity = deepcopy(plan["native_identity"])
    if mutation == "missing":
        identity["native_t_instruction"].pop("time")
    elif mutation == "extra":
        identity["native_t_instruction"]["distance"] = 15
    elif mutation == "encoding_bool":
        identity["native_t_instruction"]["encoding"] = True
    elif mutation == "time":
        identity["native_t_instruction"]["time"]["integer"] += 1
    elif mutation == "space":
        identity["native_factory_isa"][0]["space"]["integer"] += 1
    elif mutation == "error_bits":
        identity["native_t_instruction"]["error_rate"]["float_hex"] = float(2.13e-17).hex()
    with pytest.raises(ValueError):
        native.check_recorded_identity(identity)


@pytest.mark.parametrize("mutation", ["caps", "profile", "machine", "parent", "weights", "base_error", "class_order"])
def test_rehashed_model_or_class_input_changes_are_rejected(plan, mutation):
    changed = deepcopy(plan)
    if mutation == "caps":
        changed["caps"]["native_trace_queries"] += 1
    elif mutation == "profile":
        changed["profile"]["reaction_time_ns"] = 9999
    elif mutation == "machine":
        changed["fixed_machine"]["factory_physical_qubits"] -= 1
    elif mutation == "parent":
        changed["raw_parent_allocation_records"][0]["physical_failure_budget"] *= 2
    elif mutation == "class_order":
        changed["requests"][:2] = changed["requests"][1::-1]
    else:
        request = changed["requests"][0]
        if mutation == "weights":
            request["shots"] += 1
        else:
            request["native_base_error_hex"] = "0x0.0p+0"
        request["trace_input_sha256"] = native._digest({key: value for key, value in request.items() if key != "trace_input_sha256"})
    changed["plan_sha256"] = native._digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(ValueError):
        native._bound_plan(changed)


def test_rehashed_equal_cost_parent_migration_is_rejected(plan):
    """Reproduce the actual collision that defeated a cost-only parent binder."""
    changed = deepcopy(plan)
    class_id = "550e032e8bdb453809fac762c567824c564b9ae3c9091cf9185725b48aa0507e"
    target_id = "362b13e77e6f86055296a6073403f4fa16c1c6271f85f0615dca1853080b8ad8"
    index = next(i for i, row in enumerate(changed["requests"]) if row["class_id"] == class_id)
    original_parent = changed["requests"][index]["parent_physical_class_id"]
    assert original_parent == "8ec8e18770b2e7967977782ba680a5b04f3894332779b0fa6ef5ff0ba64c62c9"
    parent = next(row for row in changed["raw_parent_allocation_records"] if row["physical_class_id"] == target_id)
    source = next(row for row in changed["frozen_source_recipe"]["classes"] if row["class_id"] == class_id)
    grouped = {row["parent_physical_class_id"]: row for row in changed["frozen_source_accounting"]["parent_groups"]}
    assert grouped[original_parent]["candidate_event_counts"] == grouped[target_id]["candidate_event_counts"]
    costs = native._module(ROOT / "src/openquantum/optimization_word_accounting.py",
        changed["source_pins"]["src/openquantum/optimization_word_accounting.py"], "migration_test_pinned_accounting")
    changed["requests"][index] = native._model_request(
        dict(plan=source, events=grouped[target_id]["candidate_event_counts"]), parent,
        changed["profile"], changed["native_identity"], costs)
    changed["plan_sha256"] = native._digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(ValueError, match="frozen original allocation parent"):
        native._bound_plan(changed)


@pytest.mark.parametrize("mutation", ["duplicated_member", "omitted_member", "global_shots", "global_exposure"])
def test_rehashed_parent_members_or_global_totals_are_rejected(plan, mutation):
    changed = deepcopy(plan)
    if mutation == "duplicated_member":
        changed["raw_parent_allocation_records"][0]["compiled_class_ids"].append(
            changed["raw_parent_allocation_records"][0]["compiled_class_ids"][0])
    elif mutation == "omitted_member":
        changed["raw_parent_allocation_records"][0]["compiled_class_ids"].pop()
    elif mutation == "global_shots":
        changed["totals"]["preparations"] += 1
    else:
        changed["totals"]["data_exposure_cycles"] = -(-changed["totals"]["serial_runtime_ns"] // 1000)
    changed["plan_sha256"] = native._digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(ValueError):
        native._bound_plan(changed)


def test_changed_saved_candidate_rejected_without_rewriting_original(tmp_path):
    copy = tmp_path / "candidate"
    shutil.copytree(CANDIDATE, copy)
    (copy / "validation.json").write_text("{}\n")
    with pytest.raises(ValueError):
        native.plan_fixed_requests(copy, ROOT)
    assert hashlib.sha256((CANDIDATE / "validation.json").read_bytes()).hexdigest() == native.VALIDATION_SHA256


def test_file_guard_checks_size_before_read(tmp_path):
    large = tmp_path / "oversize"
    with large.open("wb") as stream:
        stream.seek(native.DECODED_CAP)
        stream.write(b"x")
    with pytest.raises(ValueError, match="read guard"):
        native._regular(large)


def test_default_execution_is_hard_disabled_before_any_callback_or_restore(plan, monkeypatch):
    monkeypatch.setattr(native, "restore_native_request", lambda *_: pytest.fail("default gate must precede restoration"))
    with pytest.raises(native.NativeExecutionDisabled):
        native.execute_fixed_requests(plan)


def test_production_boundary_rejection_precedes_permit_and_estimator(plan, mocked_trace):
    audit = {}
    def reject(_payload):
        raise ValueError("mock exact Native type/descriptor rejection")
    with pytest.raises(ValueError, match="type/descriptor rejection"):
        native.execute_fixed_requests(plan, trace_estimator=lambda _: pytest.fail("no estimate before boundary"),
            before_trace=lambda _: pytest.fail("no permit before boundary"),
            after_trace=lambda *_: pytest.fail("no producer receipt without a native entry"),
            native_boundary_validator=reject, audit=audit, execution_enabled=True)
    assert audit["permitted"] == audit["confirmed_started"] == audit["returned"] == 0


def test_mock_complete_pass_permit_capture_validation_order(plan, mocked_trace):
    events = []
    receipts = []
    enriched_inputs = []
    audit = {}
    def before(request):
        events.append(("permit", request["class_id"]))
        enriched_inputs.append(deepcopy(request))
        return {"durable_mock_permit": request["trace_input_sha256"]}
    def estimator(payload):
        events.append(("callback", payload["request"]["class_id"]))
        return producer(payload)
    def after(request, raw):
        events.append(("raw", request["class_id"]))
        assert raw["validated"] is False
        receipts.append(deepcopy(raw))
    result = native.execute_fixed_requests(plan, trace_estimator=estimator, before_trace=before,
                                          after_trace=after, audit=audit, execution_enabled=True)
    assert len(receipts) == len(result["results"]) == 54
    assert audit["confirmed_started"] == audit["returned"] == audit["validated"] == 54
    assert result["serial_runtime_ns"] == 602_466_084_947_200
    assert math.isclose(float(Fraction(result["whole_physical_failure_exact"])), .00010525086320790465, rel_tol=1e-15)
    for index, request in enumerate(plan["requests"]):
        assert events[3*index:3*index+3] == [(kind, request["class_id"]) for kind in ("permit", "callback", "raw")]
        assert result["results"][index]["raw_receipt_sha256"] == native._digest(receipts[index])
        native.validate_restored_input(plan, request, enriched_inputs[index])
        assert native.validate_saved_result(enriched_inputs[index], receipts[index]) == result["results"][index]
    assert native.validate_saved_execution(plan, receipts, enriched_inputs, result)["classes"] == 54
    bad_aggregate = deepcopy(result)
    bad_aggregate["serial_runtime_ns"] += 1
    with pytest.raises(ValueError, match="aggregate totals"):
        native.validate_saved_execution(plan, receipts, enriched_inputs, bad_aggregate)
    bad_receipts = deepcopy(receipts)
    bad_receipts[0]["class_id"] = bad_receipts[1]["class_id"]
    with pytest.raises(ValueError, match="exact native class request"):
        native.validate_saved_execution(plan, bad_receipts, enriched_inputs, result)
    with pytest.raises(ValueError, match="reset"):
        native.execute_fixed_requests(plan, trace_estimator=estimator, before_trace=before,
                                      after_trace=after, audit=audit, execution_enabled=True)


@pytest.mark.parametrize("mutation", ["none", "nan", "scalar_type", "factory_key_type", "factory_nan", "factory_copies",
                                       "unsupported_property", "nonfinite_property", "missing_property", "composition", "macro_isa"])
def test_invalid_mock_producer_is_captured_before_failure(plan, mocked_trace, mutation):
    receipts = []
    audit = {}
    def estimator(payload):
        value = producer(payload)
        if mutation == "none":
            return None
        if mutation == "nan":
            value.error = math.nan
        elif mutation == "scalar_type":
            value.runtime = float(value.runtime)
        elif mutation == "factory_key_type":
            value.factories = {"1028": value.factories[1028]}
        elif mutation == "factory_nan":
            value.factories[1028].error_rate = math.nan
        elif mutation == "factory_copies":
            value.factories[1028].copies = 2
        elif mutation == "unsupported_property":
            value.properties = {1: object()}
        elif mutation == "nonfinite_property":
            value.properties = {1: math.nan}
        elif mutation == "missing_property":
            del value.properties
        elif mutation == "composition":
            from qdk.qre import ErrorComposition
            value.error_composition = ErrorComposition.Product
        elif mutation == "macro_isa":
            value.isa = {1028: payload["isa"][1028]}
        return value
    with pytest.raises(ValueError):
        native.execute_fixed_requests(plan, trace_estimator=estimator, before_trace=lambda _: True,
            after_trace=lambda _request, raw: receipts.append(deepcopy(raw)), audit=audit, execution_enabled=True)
    assert len(receipts) == 1 and receipts[0]["validated"] is False
    assert audit["confirmed_started"] == audit["returned"] == 1 and audit["validated"] == 0
    if mutation == "nan":
        assert receipts[0]["result"]["error"] == {"type": "float", "hex": "nan"}
    if mutation == "factory_key_type":
        assert receipts[0]["factories"][0]["instruction_id"] == {"type": "str", "value": "1028"}


@pytest.mark.parametrize("mutation", ["before", "after_raw", "after_request", "estimator_request", "estimator_trace"])
def test_callbacks_cannot_normalize_or_replace_captured_inputs(plan, mocked_trace, mutation):
    receipts = []
    audit = {}
    def before(request):
        if mutation == "before":
            request["shots"] += 1
        return True
    def estimator(payload):
        value = producer(payload)
        if mutation == "estimator_request":
            payload["request"]["shots"] += 1
        if mutation == "estimator_trace":
            payload["trace"].compute_qubits += 1
        return value
    def after(request, raw):
        receipts.append(deepcopy(raw))
        if mutation == "after_raw":
            raw["result"]["runtime"]["value"] += 1
        if mutation == "after_request":
            request["shots"] += 1
    with pytest.raises(ValueError, match="changed"):
        native.execute_fixed_requests(plan, trace_estimator=estimator, before_trace=before,
                                      after_trace=after, audit=audit, execution_enabled=True)
    if mutation == "before":
        assert audit["confirmed_started"] == 0 and not receipts
    else:
        assert audit["returned"] == 1 and len(receipts) == 1 and audit["validated"] == 0


def test_callback_exception_and_receipt_failure_stop_without_retry(plan, mocked_trace):
    receipts = []
    audit = {}
    def failing(_payload):
        raise RuntimeError("mock native callback failure")
    with pytest.raises(RuntimeError):
        native.execute_fixed_requests(plan, trace_estimator=failing, before_trace=lambda _: True,
            after_trace=lambda _request, raw: receipts.append(deepcopy(raw)), audit=audit, execution_enabled=True)
    assert receipts[0]["status"] == "native_callback_exception"
    assert audit["confirmed_started"] == 1 and audit["returned"] == 0
    with pytest.raises(OSError):
        native.execute_fixed_requests(plan, trace_estimator=producer, before_trace=lambda _: True,
            after_trace=lambda *_: (_ for _ in ()).throw(OSError("mock durable write failed")), execution_enabled=True)


def test_no_direct_native_estimate_or_factory_enumeration_in_adapter():
    tree = ast.parse(Path(native.__file__).read_text())
    attributes = [node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not {"estimate", "enumerate", "synthesize_instrument", "gridsynth_unitary"} & set(attributes)


@pytest.mark.parametrize("runs,states,error_rate", [(0, 0, 0.0), (10**18, 17, 1.75), (3, 10**12, 123456.75)])
def test_factory_metadata_scope_all_pure_mock_report_paths(plan, monkeypatch, runs, states, error_rate):
    """Unestablished metadata meanings cannot masquerade as cost accounting."""
    monkeypatch.setattr(native, "restore_native_request", pure_mock_payload)
    monkeypatch.setattr(native, "restore_recorded_factory_isa", lambda *_: pytest.fail("no Native constructor in this source-only fix"))
    raw_records, enriched_inputs = [], []
    def before(request):
        enriched_inputs.append(deepcopy(request))
        return True
    def estimator(payload):
        result = producer(payload)
        result.factories[1028].runs = runs
        result.factories[1028].states = states
        result.factories[1028].error_rate = error_rate
        return result
    result = native.execute_fixed_requests(plan, trace_estimator=estimator, before_trace=before,
        after_trace=lambda _request, raw: raw_records.append(deepcopy(raw)), execution_enabled=True)
    pending = native.factory_metadata_scope()
    checked = native.factory_metadata_scope(copies_validated=True)
    assert plan["factory_result_metadata_scope"] == pending
    assert pending["fields"]["copies"]["status"] == "captured_pending_validation"
    assert checked["fields"]["copies"]["status"] == "validated_exactly_one"
    assert result["factory_result_metadata_scope"] == checked
    for request, raw, derived in zip(enriched_inputs, raw_records, result["results"]):
        assert raw["factory_result_metadata_scope"] == pending
        assert raw["factories"][0]["runs"] == {"type": "int", "value": runs}
        assert raw["factories"][0]["states"] == {"type": "int", "value": states}
        assert raw["factories"][0]["error_rate"] == {"type": "float", "hex": error_rate.hex()}
        assert derived["factory_result_metadata_scope"] == checked
        assert native.validate_saved_result(request, raw) == derived
        for name in ("runs", "states", "error_rate"):
            assert derived["factory_result_metadata_scope"]["fields"][name] == {
                "status": "captured_unvalidated", "used_in_accounting": False}
    report = native.validate_saved_execution(plan, raw_records, enriched_inputs, result)
    assert report["factory_result_metadata_scope"] == checked
    assert result["serial_runtime_ns"] == 602_466_084_947_200
    assert math.isclose(float(Fraction(result["whole_physical_failure_exact"])), .00010525086320790465, rel_tol=1e-15)
    assert checked["authoritative_cost_source"] == "exact_native_isa_t_instruction"
    assert checked["authoritative_cost_fields"] == ["time", "space", "error_rate"]


def test_factory_metadata_scope_missing_or_false_semantic_claims_rejected(plan, monkeypatch):
    monkeypatch.setattr(native, "restore_native_request", pure_mock_payload)
    payload = pure_mock_payload(plan, plan["requests"][0])
    raw = native._capture_result(producer(payload), payload["request"])
    assert raw["factory_result_metadata_scope"]["fields"]["states"]["status"] == "captured_unvalidated"
    raw["factory_result_metadata_scope"]["fields"]["states"]["status"] = "validated"
    with pytest.raises(ValueError, match="unvalidated scope"):
        native._validated_result(raw, payload["request"])
    changed = deepcopy(plan)
    changed["factory_result_metadata_scope"]["fields"]["error_rate"]["used_in_accounting"] = True
    changed["plan_sha256"] = native._digest({key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(ValueError, match="metadata scope"):
        native._bound_plan(changed, check_runtime=False)
