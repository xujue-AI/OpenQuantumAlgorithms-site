#!/usr/bin/env python3
"""Audit/verify the fixed Native runner; execution requires exact approval.

No approval exists during adapter implementation. The internal guardian and
dummy worker actions are operational test helpers, not Native execution.
"""
from __future__ import annotations

_ENTRY_MONOTONIC = __import__("time").monotonic()

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
CANDIDATE = ROOT / "docs/coordination/optimization-eighth-candidate"
RECIPE_SHA256 = "42dc12de0aa666e4fc9054c22b2321e58240959f1f74c647dc368894de79b694"
DEFAULT_FREEZE = ROOT / "docs/coordination/optimization-eighth-native-freeze.json"
DEFAULT_APPROVAL = ROOT / "docs/coordination/optimization-eighth-native-approval.json"
DEFAULT_PLAN = ROOT / "docs/coordination/optimization-eighth-native-review/native-plan.json"
RUNNER_PATH = ROOT / "src/openquantum/optimization_word_native_runner.py"
ADAPTER_PATH = ROOT / "src/openquantum/optimization_word_native.py"


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _load(path, name, expected=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Owned runner source must be a regular file")
    raw = path.read_bytes()
    if expected is not None and _digest(raw) != expected:
        raise ValueError("Frozen runner/adapter source changed")
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    exec(compile(raw, str(path), "exec"), value.__dict__)
    return value


def _json(path, maximum=1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Bounded scope record is unavailable")
    return json.loads(path.read_bytes())


def _scope(freeze_path, approval_path, *, require_approval):
    freeze = _json(freeze_path)
    required = {SCRIPT.relative_to(ROOT).as_posix(), RUNNER_PATH.relative_to(ROOT).as_posix(),
                ADAPTER_PATH.relative_to(ROOT).as_posix()}
    if (freeze.get("candidate_recipe_sha256") != RECIPE_SHA256
            or not required <= set(freeze.get("source_sha256", {}))):
        raise ValueError("Native freeze does not bind this exact saved candidate and concrete sources")
    for relative, expected in freeze["source_sha256"].items():
        path = ROOT / relative
        if (not path.resolve().is_relative_to(ROOT) or path.is_symlink()
                or not path.is_file() or _digest(path.read_bytes()) != expected):
            raise ValueError("An owned source changed before runner loading")
    runner = _load(RUNNER_PATH, "fixed_native_parent_runner", freeze["source_sha256"][RUNNER_PATH.relative_to(ROOT).as_posix()])
    if freeze["caps"] != asdict(runner.RunnerCaps()):
        raise ValueError("Native parent caps differ from the exact 54-call/60s scope")
    recipe = _json(CANDIDATE / "recipe.json")
    if (_digest((CANDIDATE / "recipe.json").read_bytes()) != RECIPE_SHA256
            or freeze["fixed_machine"] != recipe["common_machine"]):
        raise ValueError("Native freeze substituted a candidate or common machine")
    output = Path(freeze["output_directory"]).resolve()
    if not output.is_relative_to(ROOT / "docs/coordination") or output.is_relative_to(CANDIDATE):
        raise ValueError("Native output must be a separate private coordination tree")
    approval = _json(approval_path) if require_approval else None
    if require_approval:
        runner.validate_approval(approval, freeze, output)
    return freeze, approval, runner, output


def validate_native_boundary(payload, saved_descriptor=None):
    """Bind the real extension types/method before granting any Native call.

    QDK 1.34.0 exports these classes directly on qdk._native (their Python
    __module__ is builtins). No nested qre module or public patched alias is
    used as an execution boundary. Inspecting the descriptor makes no call.
    """
    import qdk._native as native_extension
    trace_type, isa_type = native_extension.Trace, native_extension.ISA
    descriptor = trace_type.__dict__.get("estimate")
    if (type(payload["trace"]) is not trace_type or type(payload["isa"]) is not isa_type
            or not isinstance(descriptor, types.MethodDescriptorType)
            or descriptor.__objclass__ is not trace_type
            or saved_descriptor is not None and descriptor is not saved_descriptor):
        raise ValueError("Production Native boundary is not the exact extension Trace/ISA/built-in descriptor")
    return descriptor


def _worker(config_path):
    # This internal action independently verifies the exact approval. A caller
    # cannot turn a dummy guardian into an unapproved Native worker.
    config = _json(config_path)
    freeze, approval, runner, output = _scope(config["freeze_path"], config["approval_path"], require_approval=True)
    identity = config["identity"]
    state = runner.canonical_private_state(ROOT, RECIPE_SHA256, freeze["fixed_machine"])
    claim = _json(state / "attempt-claim.json")
    if claim["identity"] != identity or json.loads(os.environ["OQP_NATIVE_IDENTITY_JSON"]) != identity:
        raise runner.ExecutionDisabled("Native worker lacks the exact canonical parent claim")
    guardian_config = _json(output / "guardian-config.json")
    work_deadline = float(os.environ["OQP_NATIVE_WORK_DEADLINE"])
    if guardian_config["identity"] != identity or guardian_config["work_deadline"] != work_deadline:
        raise runner.ExecutionDisabled("Native worker guardian identity or deadline changed")
    pid = runner.wait_worker_readiness(int(os.environ["OQP_NATIVE_READINESS_FD"]),
        output / "worker-pid.json", identity, deadline=work_deadline,
        parent_deadline=guardian_config["deadline"])
    if (pid["worker_pid"] != os.getpid() or pid["guardian_pid"] != os.getppid()
            or pid["identity"] != identity or guardian_config["identity"] != identity):
        raise runner.ExecutionDisabled("Native worker is not owned by the claimed guardian")
    runner.block_worker_process_creation()
    client = None
    event_fd = int(os.environ["OQP_NATIVE_EVENT_FD"])
    try:
        adapter = _load(ADAPTER_PATH, "fixed_native_worker_adapter", freeze["source_sha256"][ADAPTER_PATH.relative_to(ROOT).as_posix()])
        plan = adapter.plan_fixed_requests(CANDIDATE, ROOT)
        if plan["plan_sha256"] != freeze["native_plan_sha256"]:
            raise ValueError("Worker reconstruction differs from the frozen concrete Native plan")
        client = runner.WorkerPermitClient(event_fd, int(os.environ["OQP_NATIVE_REPLY_FD"]), identity,
            state / "private-producer", deadline=float(os.environ["OQP_NATIVE_WORK_DEADLINE"]),
            requests=plan["requests"], input_directory=state / "native-inputs")
        native_boundary = {}
        def before_native_permission(payload):
            descriptor = validate_native_boundary(payload, native_boundary.get("descriptor"))
            native_boundary["descriptor"] = descriptor
            return True
        def actual_estimator(payload):
            descriptor = validate_native_boundary(payload, native_boundary["descriptor"])
            return descriptor(payload["trace"], payload["isa"], max_error=payload["max_error"],
                              composition=payload["composition"])
        result = adapter.execute_fixed_requests(plan, trace_estimator=actual_estimator,
            before_trace=client.before_trace, after_trace=client.after_trace,
            execution_enabled=True, native_boundary_validator=before_native_permission)
        raw = runner.canonical(result)
        source_bytes = runner.tree_bytes(CANDIDATE)
        if len(raw) > runner.MAX_DECODED_RECORD_BYTES or source_bytes + runner.tree_bytes(output) + len(raw) > runner.MAX_COMBINED_BUNDLE_BYTES:
            raise ValueError("Checked Native export exceeds the complete source/output cap")
        descriptor = runner.atomic_record(output / "native-result.json", result,
                                          maximum=runner.MAX_DECODED_RECORD_BYTES)
        runner.send_message(event_fd, dict(event="worker_done", identity=identity,
            result_file="native-result.json", result_sha256=descriptor["sha256"]))
    except BaseException as error:
        runner.send_message(event_fd, dict(event="worker_failed", identity=identity,
                                          error_type=type(error).__name__))
        return 1
    return 0


def _dummy_worker(behavior, count):
    runner = _load(RUNNER_PATH, "dummy_native_protocol_worker")
    identity = json.loads(os.environ["OQP_NATIVE_IDENTITY_JSON"])
    event_fd = int(os.environ["OQP_NATIVE_EVENT_FD"])
    deadline = float(os.environ["OQP_NATIVE_WORK_DEADLINE"])
    output = Path(os.environ["OQP_NATIVE_DUMMY_OUTPUT"])
    guardian_config = _json(output / "guardian-config.json")
    runner.wait_worker_readiness(int(os.environ["OQP_NATIVE_READINESS_FD"]),
        output / "worker-pid.json", identity, deadline=deadline,
        parent_deadline=guardian_config["deadline"])
    producer = Path(os.environ["OQP_NATIVE_PRODUCER_DIRECTORY"])
    client = runner.WorkerPermitClient(event_fd, int(os.environ["OQP_NATIVE_REPLY_FD"]), identity,
                                      producer, deadline=deadline)
    if behavior == "sleep":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(120)
        return 0
    if behavior == "orphan":
        import subprocess
        subprocess.Popen([sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for index in range(count):
        request = dict(class_id=f"dummy-{index}", trace_input_sha256="a" * 64)
        client.before_trace(request)
        if behavior == "crash-after-permit":
            os._exit(17)
        client.after_trace(request, dict(status="dummy_producer_return", value=index))
    output = Path(os.environ["OQP_NATIVE_DUMMY_OUTPUT"])
    descriptor = runner.atomic_record(output / "native-result.json", dict(mode="dummy", native_calls=0, count=count))
    runner.send_message(event_fd, dict(event="worker_done", identity=identity,
        result_file="native-result.json", result_sha256=descriptor["sha256"]))
    return 0


def _audit(output, behavior, seconds):
    runner = _load(RUNNER_PATH, "dummy_native_parent")
    output = Path(output).resolve()
    caps = runner.RunnerCaps(parent_wall_seconds=seconds,
        cleanup_seconds=min(5.0, seconds / 3), termination_grace_seconds=min(.1, seconds / 10))
    environment = os.environ
    environment["OQP_NATIVE_DUMMY_OUTPUT"] = str(output)
    requests = [dict(class_id=f"dummy-{i}", trace_input_sha256="a" * 64) for i in range(2)]
    return runner.run_bounded_worker([sys.executable, str(SCRIPT), "--dummy-worker", behavior, "--dummy-count", "2"],
        identity=dict(mode="dummy_protocol_audit", version=1), requests=requests,
        output_directory=output, guardian_command=[sys.executable, str(SCRIPT)], source_bundle_bytes=0,
        caps=caps, start_monotonic=_ENTRY_MONOTONIC, mode="dummy_protocol_audit")


def _execute(freeze_path, approval_path):
    freeze, approval, runner, output = _scope(freeze_path, approval_path, require_approval=True)
    identity = dict(candidate_recipe_sha256=RECIPE_SHA256,
                    native_plan_sha256=freeze["native_plan_sha256"],
                    approval_sha256=_digest(runner.canonical(approval)))
    # Only bounded pure request metadata is reconstructed in the controller;
    # heavy native imports/ISA construction occur after the durable claim in
    # the guardian-owned worker, within this same absolute parent clock.
    plan_path = Path(freeze["native_plan_path"])
    plan = _json(plan_path, runner.MAX_DECODED_RECORD_BYTES)
    adapter = _load(ADAPTER_PATH, "fixed_native_parent_adapter", freeze["source_sha256"][ADAPTER_PATH.relative_to(ROOT).as_posix()])
    material = dict(plan)
    embedded_hash = material.pop("plan_sha256")
    logical_hash = _digest(json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (embedded_hash != freeze["native_plan_sha256"] or logical_hash != embedded_hash
            or _digest(plan_path.read_bytes()) != freeze["native_plan_file_sha256"]):
        raise ValueError("Bounded stored Native plan differs from the reviewed freeze")
    state = runner.canonical_private_state(ROOT, RECIPE_SHA256, freeze["fixed_machine"])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(freeze_path=str(Path(freeze_path).resolve()), approval_path=str(Path(approval_path).resolve()), identity=identity)
    runner.atomic_record(output / "worker-config.json", config)
    result = runner.run_bounded_worker([sys.executable, str(SCRIPT), "--worker", str(output / "worker-config.json")],
        identity=identity, requests=plan["requests"], output_directory=output,
        guardian_command=[sys.executable, str(SCRIPT)], source_bundle_bytes=runner.tree_bytes(CANDIDATE),
        caps=runner.RunnerCaps(), start_monotonic=_ENTRY_MONOTONIC, mode="approved_native_execution",
        state_directory=state, producer_directory=state / "private-producer",
        input_validator=lambda request, enriched: adapter.validate_restored_input(plan, request, enriched))
    result["factory_result_metadata_scope"] = adapter.factory_metadata_scope(
        copies_validated=result["status"] == "complete")
    return result


def _freeze(freeze_path):
    """Create a source-only Native plan/freeze; never create approval or claim."""
    runner = _load(RUNNER_PATH, "native_source_freeze_runner")
    adapter = _load(ADAPTER_PATH, "native_source_freeze_adapter")
    recipe = _json(CANDIDATE / "recipe.json")
    source_names = set(recipe["owned_source_sha256"]) | {
        SCRIPT.relative_to(ROOT).as_posix(), RUNNER_PATH.relative_to(ROOT).as_posix(),
        ADAPTER_PATH.relative_to(ROOT).as_posix(),
        "tests/test_optimization_word_native.py", "tests/test_optimization_word_native_runner.py",
        "tests/test_optimization_word_watchdog_review.py"}
    pins = {}
    for name in sorted(source_names):
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Concrete Native source/test specification is unavailable")
        pins[name] = _digest(path.read_bytes())
    plan = adapter.plan_fixed_requests(CANDIDATE, ROOT)
    raw_plan = runner.canonical(plan)
    output = ROOT / "docs/coordination/optimization-eighth-native-results"
    freeze = dict(schema_version=1, status="frozen_native_adapter_execution_unapproved",
        candidate_recipe_sha256=RECIPE_SHA256, source_sha256=pins,
        fixed_machine=plan["fixed_machine"], caps=asdict(runner.RunnerCaps()),
        native_plan_caps=plan["caps"], native_plan_sha256=plan["plan_sha256"],
        native_plan_file_sha256=_digest(raw_plan), native_plan_path=str(DEFAULT_PLAN),
        output_directory=str(output), execution_authorized=False,
        native_restoration_scope=plan["native_restoration_scope"],
        native_fingerprints=plan["native_fingerprints"],
        cumulative_state_scope="Git-common-dir key from candidate recipe/fixed ISA/T/distance/retry; excludes code/output revisions",
        producer_reference_cap_bytes=runner.MAX_DECODED_RECORD_BYTES,
        factory_result_metadata_scope=plan["factory_result_metadata_scope"],
        raw_export_scope="all producer semantic preimages exported losslessly; opaque native provenance graph not captured",
        optional_qasm_importer="unavailable; direct checked-primitives NativeTrace path")
    for path, value in ((DEFAULT_PLAN, plan), (Path(freeze_path), freeze)):
        raw = runner.canonical(value)
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError("Existing source-only Native freeze/plan must be preserved before a revised freeze")
        else:
            runner.atomic_record(path, value, maximum=runner.MAX_DECODED_RECORD_BYTES)
    if {name: _digest((ROOT / name).read_bytes()) for name in pins} != pins:
        raise ValueError("Native source changed while writing its freeze")
    return dict(status=freeze["status"], freeze_sha256=_digest(runner.canonical(freeze)),
        native_plan_sha256=plan["plan_sha256"], native_plan_file_sha256=_digest(raw_plan),
        execution_enabled=False, native_calls=0)


def _verify(freeze_path):
    freeze, _, runner, output = _scope(freeze_path, DEFAULT_APPROVAL, require_approval=False)
    parent = _json(output / "parent-receipt.json")
    guardian = _json(output / "guardian-receipt.json")
    if (parent["status"] != "pending_guardian_confirmation" or guardian["status"] != "parent_finished"
            or not guardian.get("cleanup", {}).get("complete") or guardian.get("parent_finished") is not True):
        raise ValueError("Native output lacks a complete post-export guardian seal")
    if (not isinstance(parent.get("parent_wall_seconds"), (int, float))
            or isinstance(parent.get("parent_wall_seconds"), bool)
            or not math.isfinite(parent["parent_wall_seconds"])
            or not 0 <= parent["parent_wall_seconds"] <= 60):
        raise ValueError("Native parent receipt exceeded its finite complete-process cap")
    try:
        if (parent["caps"] != asdict(runner.RunnerCaps()) or parent["error_type"] is not None
                or parent["worker_returncode"] != 0 or parent["mode"] != "approved_native_execution"
                or parent["identity"]["candidate_recipe_sha256"] != RECIPE_SHA256
                or parent["identity"]["native_plan_sha256"] != freeze["native_plan_sha256"]):
            raise ValueError("Native parent identity/caps/status changed")
        config = _json(output / "guardian-config.json")
        if config["identity"] != parent["identity"]:
            raise ValueError("Native guardian seal changed its frozen parent identity")
        elapsed = guardian["wall_end_monotonic"] - (config["deadline"] - 60)
        if (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                or not 0 <= elapsed <= 60 or config["work_deadline"] != config["deadline"] - 5):
            raise ValueError("Post-export guardian seal exceeded the exact complete-process cap")
        counters = parent["counters"]
        if (any(type(counters[key]) is not int or counters[key] != 54 for key in
                ("charged_trace_permits", "reported_native_callsites", "reported_native_return_receipts",
                 "actual_trace_calls_upper_bound", "independently_verified_producer_receipts"))
                or counters["unknown_entry_permits"] or counters["in_flight_or_failed_requests"]
                or counters["automatic_rerun_permitted"] is not False):
            raise ValueError("Native call/result coverage is not the exact complete 54-class pass")
        plan = _json(freeze["native_plan_path"], runner.MAX_DECODED_RECORD_BYTES)
        material = dict(plan)
        embedded = material.pop("plan_sha256")
        logical = _digest(json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        if logical != embedded or embedded != freeze["native_plan_sha256"]:
            raise ValueError("Native plan logical content changed")
        if _digest(Path(freeze["native_plan_path"]).read_bytes()) != freeze["native_plan_file_sha256"]:
            raise ValueError("Native plan transport bytes changed")
        seal = _json(output / "completion-seal.json")
        if (seal["status"] != "complete_after_all_parent_exports" or seal["identity"] != parent["identity"]
                or seal["parent_receipt_sha256"] != _digest((output / "parent-receipt.json").read_bytes())
                or seal["guardian_receipt_sha256"] != _digest((output / "guardian-receipt.json").read_bytes())
                or seal["cleanup_complete"] is not True
                or seal["deadline_monotonic"] != config["deadline"]
                or not math.isfinite(seal["confirmed_monotonic"])
                or not 0 <= seal["confirmed_monotonic"] - (config["deadline"] - 60) <= 60):
            raise ValueError("Native completion lacks an intact post-export guardian seal")
        descriptor = parent["verified_result"]
        if descriptor["file"] != "native-result.json":
            raise ValueError("Native result filename changed")
        raw_result = runner.bounded_regular(output / descriptor["file"], runner.MAX_DECODED_RECORD_BYTES)
        if _digest(raw_result) != descriptor["sha256"] or len(raw_result) != descriptor["bytes"]:
            raise ValueError("Native result transport changed after export verification")
        if seal["verified_result"] != descriptor:
            raise ValueError("Guardian seal changed the verified Native result descriptor")
        result = json.loads(raw_result)
        expected_ids = [request["class_id"] for request in plan["requests"]]
        if (len(expected_ids) != 54 or len(set(expected_ids)) != 54
                or [row["class_id"] for row in result["results"]] != expected_ids
                or [row["shots"] for row in result["results"]] != [row["shots"] for row in plan["requests"]]
                or [row["parent_physical_class_id"] for row in result["results"]] != [row["parent_physical_class_id"] for row in plan["requests"]]
                or sum(row["shots"] for row in result["results"]) != 82032
                or len({row["parent_physical_class_id"] for row in result["results"]}) != 49
                or result["plan_sha256"] != freeze["native_plan_sha256"]):
            raise ValueError("Native class/parent/repetition coverage changed")
        import gzip
        def decoded_export(name):
            desc = parent[name]
            path = output / desc["file"]
            if not path.resolve().is_relative_to(output):
                raise ValueError("Native export path escaped its output")
            packed = runner.bounded_regular(path, runner.MAX_COMBINED_BUNDLE_BYTES)
            if len(packed) != desc["bytes"] or _digest(packed) != desc["sha256"]:
                raise ValueError("Native lossless export transport changed")
            with gzip.open(path, "rb") as stream:
                raw = stream.read(runner.MAX_DECODED_RECORD_BYTES + 1)
            if len(raw) > runner.MAX_DECODED_RECORD_BYTES or len(raw) != desc["decoded_bytes"] or _digest(raw) != desc["decoded_sha256"]:
                raise ValueError("Native lossless export decoded identity changed")
            return raw
        raw_returns = decoded_export("raw_producer_export")
        raw_inputs = decoded_export("native_input_export")
        producer_rows = [json.loads(line) for line in raw_returns.splitlines()]
        if [row["class_id"] for row in producer_rows] != expected_ids:
            raise ValueError("Raw Native producer preimages lost class coverage")
        native_inputs = json.loads(raw_inputs)
        if [row["class_id"] for row in native_inputs] != expected_ids:
            raise ValueError("Full Native input preimages lost class coverage")
        adapter = _load(ADAPTER_PATH, "native_read_only_verifier", freeze["source_sha256"][ADAPTER_PATH.relative_to(ROOT).as_posix()])
        for request, enriched in zip(plan["requests"], native_inputs):
            adapter.validate_restored_input(plan, request, enriched)
        rebuilt = adapter.validate_saved_execution(plan,
            [row["producer_record"] for row in producer_rows], native_inputs, result)
        if rebuilt["classes"] != 54 or rebuilt["parent_groups"] != 49:
            raise ValueError("Native saved numerical revalidation lost class/parent coverage")
        journal_rows = [json.loads(line) for line in runner.bounded_regular(output / "attempt-journal.jsonl", runner.MAX_JOURNAL_BYTES).splitlines()]
        if (any(type(row.get("sequence")) is not int or row["sequence"] != index or row.get("identity") != parent["identity"]
                for index, row in enumerate(journal_rows))
                or [row["request_id"] for row in journal_rows if row["event"] == "permit_granted"] != expected_ids
                or [row["request_id"] for row in journal_rows if row["event"] == "native_entry_reported"] != expected_ids
                or [row["request_id"] for row in journal_rows if row["event"] == "native_return_recorded"] != expected_ids):
            raise ValueError("Native journal sequence/unique permit and return coverage changed")
        expected_hashes = {row["request_id"]: row["receipt_sha256"] for row in journal_rows if row["event"] == "native_return_recorded"}
        actual_hashes = {json.loads(line)["class_id"]: _digest(line) for line in raw_returns.splitlines(keepends=True)}
        if expected_hashes != actual_hashes or len(expected_hashes) != 54:
            raise ValueError("Raw Native return preimages disagree with the authoritative journal")
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Native completion receipt is incomplete or malformed") from error
    if runner.tree_bytes(CANDIDATE) + runner.tree_bytes(output) > runner.MAX_COMBINED_BUNDLE_BYTES:
        raise ValueError("Complete Native source/output bytes exceeded the cap")
    return dict(status="verified_fixed_native_result", writes_performed=False, classes=54,
                factory_result_metadata_scope=adapter.factory_metadata_scope(copies_validated=True),
                optional_qasm_importer="unavailable; direct checked-primitives Trace path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group(required=True)
    choices.add_argument("--plan", action="store_true")
    choices.add_argument("--freeze-only", action="store_true")
    choices.add_argument("--verify-only", action="store_true")
    choices.add_argument("--execute", action="store_true")
    choices.add_argument("--audit", choices=("success", "sleep", "orphan", "crash-after-permit"))
    choices.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    choices.add_argument("--guardian", type=Path, help=argparse.SUPPRESS)
    choices.add_argument("--dummy-worker", choices=("success", "sleep", "orphan", "crash-after-permit"), help=argparse.SUPPRESS)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--approval", type=Path, default=DEFAULT_APPROVAL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-seconds", type=float, default=3.0)
    parser.add_argument("--dummy-count", type=int, default=2)
    for name in ("event-fd", "worker-event-fd", "reply-fd", "lifeline-fd"):
        parser.add_argument("--" + name, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.guardian:
        runner = _load(RUNNER_PATH, "native_process_guardian")
        runner.guardian_main(args.guardian, args.event_fd, args.reply_fd, args.lifeline_fd, args.worker_event_fd)
        return 0
    if args.dummy_worker:
        return _dummy_worker(args.dummy_worker, args.dummy_count)
    if args.worker:
        return _worker(args.worker)
    # The outer hard limit remains active through final receipt/export reads.
    def expired(signum, frame):
        raise TimeoutError("The complete parent Native envelope exceeded 60 seconds")
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, max(.001, 60 - (time.monotonic() - _ENTRY_MONOTONIC)))
    try:
        if args.freeze_only:
            result = _freeze(args.freeze)
        elif args.plan:
            adapter = _load(ADAPTER_PATH, "native_source_plan")
            plan = adapter.plan_fixed_requests(CANDIDATE, ROOT)
            result = dict(status=plan["status"], plan_sha256=plan["plan_sha256"], requests=len(plan["requests"]),
                          execution_enabled=False, native_calls=0, caps=plan["caps"],
                          factory_result_metadata_scope=plan["factory_result_metadata_scope"])
        elif args.audit:
            output = args.output_dir or Path(tempfile.mkdtemp(prefix="oqp-native-dummy-audit-"))
            result = _audit(output, args.audit, args.audit_seconds)
        elif args.verify_only:
            result = _verify(args.freeze)
        else:
            result = _execute(args.freeze, args.approval)
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0 if result["status"] in {"complete", "verified_fixed_native_result", "planned_fixed_native_requests_not_executed", "frozen_native_adapter_execution_unapproved"} else 1
    except BaseException as error:
        print(json.dumps(dict(status="execution_disabled_or_incomplete", error_type=type(error).__name__,
                              native_execution_default=False), sort_keys=True))
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
