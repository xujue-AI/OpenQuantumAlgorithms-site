"""Saved compiled-release gates reject drift before any new computation."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compiled_release_gate", ROOT / "scripts/verify_optimization_compiled.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def package(tmp_path):
    root = (tmp_path / "package").resolve()
    bundle = root / gate.BUNDLE_RELATIVE
    shutil.copytree(ROOT / gate.BUNDLE_RELATIVE, bundle)
    for relative in gate.SOURCE_FILES.values():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    return root


def verify(package, *, replay=False):
    return gate.verify_bundle(package / gate.BUNDLE_RELATIVE,
                              source_root=package, canonical_root=ROOT, replay=replay)


@pytest.fixture
def records():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    return [gate.json_record(bundle, name) for name in (
        "manifest.json", "freeze.json", "completion-freeze.json", "completion-timing.json", "artifact-size.json")]


def set_value(record, path, value):
    for key in path[:-1]:
        record = record[key]
    record[path[-1]] = value


def test_reviewed_identity_only_preflight_checks_all_sources_transports_and_bytes(package):
    before = gate.inventory(package / gate.BUNDLE_RELATIVE)
    receipt = verify(package)
    assert receipt["files"] == 301 and receipt["bytes"] == 9614924
    assert receipt["original_files_preserved"] == 281
    assert receipt["compressed_json_files"] == 57 and receipt["decoded_json_bytes"] == 43300591
    assert receipt["saved_reexport_run"] is False
    assert gate.inventory(package / gate.BUNDLE_RELATIVE) == before


@pytest.mark.parametrize("relative", sorted(gate.SOURCE_FILES.values()))
def test_changed_live_own_source_backend_lock_or_license_rejects(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="current source"):
        verify(package)


@pytest.mark.parametrize("name", [
    "manifest.json", "freeze.json", "completion-recipe.json", "physical-dag.json.gz",
    "completed-replay.json.gz", "source-classes.json.gz", "resources.json", "workload.svg",
    "checkpoints/failures/physical-workload.json", "timing/synthesis.json",
])
def test_changed_saved_data_status_recipe_circuit_or_failure_bytes_rejects(package, name):
    path = package / gate.BUNDLE_RELATIVE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="reviewed release"):
        verify(package)


def test_missing_original_checkpoint_rejects_before_reexport(package, monkeypatch):
    bundle = package / gate.BUNDLE_RELATIVE
    checkpoint = next((bundle / "checkpoints/synthesis").glob("*.json.gz"))
    checkpoint.unlink()
    monkeypatch.setattr(gate.subprocess, "run", lambda *args, **kwargs: pytest.fail("No worker may start"))
    with pytest.raises(ValueError, match="complete file count"):
        verify(package, replay=True)


def test_unrecorded_file_or_empty_directory_rejects(package):
    bundle = package / gate.BUNDLE_RELATIVE
    (bundle / "unreviewed.json.gz").write_bytes(gzip.compress(b"{}", mtime=0))
    with pytest.raises(ValueError, match="file/byte cap|complete file count"):
        verify(package)
    (bundle / "unreviewed.json.gz").unlink()
    (bundle / "private-cache").mkdir()
    with pytest.raises(ValueError, match="directory inventory"):
        verify(package)


@pytest.mark.parametrize("index,path,value", [
    (0, ["status"], "pending"), (0, ["replay_status"], "failed_incomplete"),
    (0, ["run_id"], "unselected"), (0, ["sampled_groups"], 109), (0, ["preparations"], 1),
    (0, ["public_reader_requires_private_pickle"], True),
    (0, ["original_identity"], {}), (0, ["identity"], {}),
    (0, ["provenance", "python"], "3.12.15"),
    (0, ["provenance", "packages"], {}),
    (0, ["provenance", "source_sha256"], {}),
    (0, ["recovery", "first_query_counts", "top_level_calls"], 0),
    (0, ["recovery", "second_query_counts", "native_trace_queries"], 0),
    (0, ["recovery", "cumulative_query_counts", "top_level_calls"], 49),
    (0, ["recovery", "roundtrip_exact_to_second_live_value"], False),
    (0, ["recovery", "first_complete_result_unavailable"], False),
    (0, ["recovery", "quantum_preparations_counted_once"], 164064),
    (0, ["recovery", "codec_statistics", "logical_sha256"], "0" * 64),
    (0, ["completed_replay", "sha256"], "0" * 64),
    (0, ["physical_storage", "encoding"], "ordinary-json"),
    (0, ["timing", "original_physical", "status"], "complete"),
    (0, ["classes"], []),
    (1, ["execution_performed"], True), (2, ["native_or_synthesis_executed"], True),
    (3, ["status"], "incomplete"), (3, ["error_type"], "Interrupted"),
    (4, ["total_bytes"], 0), (4, ["files"], 300),
    (4, ["private_cache_included"], True), (4, ["parent_archive_copied"], True),
])
def test_inconsistent_semantic_status_counters_scope_and_transport_identities_reject(records, index, path, value):
    set_value(records[index], path, value)
    with pytest.raises(ValueError, match="Compiled replay|Compiled-replay"):
        gate.verify_records(*records)


@pytest.mark.parametrize("value", [-1., 121., float("inf"), float("nan"), True])
def test_completion_timings_are_finite_nonnegative_and_bounded(records, value):
    records[3]["recovery_wall_seconds"] = value
    records[0]["timing"]["completion"] = records[3]
    with pytest.raises(ValueError):
        gate.verify_records(*records)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e400}', b'[]'])
def test_json_record_parser_rejects_duplicates_nonfinite_and_wrong_shapes(raw):
    with pytest.raises(ValueError):
        gate.json_bytes(raw)


@pytest.mark.parametrize("marker", gate.PRIVATE_MARKERS)
def test_bounded_compressed_json_scan_rejects_private_strings_without_printing_payloads(tmp_path, marker):
    path = tmp_path / "completed-replay.json.gz"
    path.write_bytes(gzip.compress(json.dumps({"content": marker.decode()}).encode(), mtime=0))
    with pytest.raises(ValueError, match="Private connection/key content"):
        gate.compressed_record(path, 4096)


def test_compressed_json_expansion_is_rejected_before_large_materialization(tmp_path):
    path = tmp_path / "checkpoint.json.gz"
    path.write_bytes(gzip.compress(b" " * 8192, mtime=0))
    with pytest.raises(ValueError, match="decoded-byte cap"):
        gate.compressed_record(path, 4096)


def test_public_gate_and_tests_do_not_contain_their_runtime_private_markers():
    for path in (ROOT / "scripts/verify_optimization_compiled.py", Path(__file__)):
        for marker in gate.PRIVATE_MARKERS:
            assert marker not in path.read_bytes()


@pytest.mark.parametrize("field,value", [("files", 300), ("native_estimator_calls", 1),
                                        ("saved_reexport_run", False), ("temporary_reexport_files_compared", 0),
                                        ("verify_only_reexports_derived_files", False), ("verification_wall_seconds", -1.)])
def test_worker_receipt_cannot_overwrite_verified_identity_or_zero_call_claims(field, value):
    expected = {"files": 301, "bytes": 9614924}
    worker = {**expected, "temporary_reexport_files_compared": 301, "verify_only_reexports_derived_files": True,
              "synthesis_calls": 0, "native_estimator_calls": 0, "optimizer_calls": 0, "sampling_calls": 0,
              "saved_reexport_run": True, "verification_wall_seconds": 1.}
    gate.verify_worker_receipt(worker, expected)
    worker[field] = value
    with pytest.raises(ValueError):
        gate.verify_worker_receipt(worker, expected)


def test_each_fresh_module_load_is_guarded_against_forbidden_computation():
    counters = dict(synthesis_calls=0, native_estimator_calls=0, optimizer_calls=0, sampling_calls=0)
    backend = SimpleNamespace(synthesize_instrument=lambda: None, estimate_instrument=lambda: None)
    parent = SimpleNamespace(pattern_search=lambda: None, _sample=lambda: None)
    core = SimpleNamespace(_load_module=lambda name: backend if name == "backend" else parent,
                           compile_source_class=lambda: None, run_compiled_physical=lambda: None)
    gate.guard_module(core, counters)
    for action in (core.compile_source_class, core.run_compiled_physical,
                   core._load_module("backend").synthesize_instrument,
                   core._load_module("backend").estimate_instrument,
                   core._load_module("parent").pattern_search, core._load_module("parent")._sample):
        with pytest.raises(ValueError, match="forbidden"):
            action()
    assert counters == dict(synthesis_calls=2, native_estimator_calls=2, optimizer_calls=1, sampling_calls=1)


def test_bounded_worker_timeout_preserves_every_original_file(package, monkeypatch):
    bundle = package / gate.BUNDLE_RELATIVE
    before = gate.inventory(bundle)
    def timeout(*args, **kwargs):
        assert kwargs["timeout"] == 120
        raise subprocess.TimeoutExpired(args[0], 120)
    monkeypatch.setattr(gate.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="120-second cap"):
        verify(package, replay=True)
    assert gate.inventory(bundle) == before


def test_symbolic_link_in_bundle_or_source_path_rejects(package):
    bundle = package / gate.BUNDLE_RELATIVE
    path = bundle / "resources.json"
    outside = package / "outside.json"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="linked or special"):
        verify(package)


def test_saved_temporary_reexport_matches_all_301_bytes_with_zero_new_computation(package):
    before = gate.inventory(package / gate.BUNDLE_RELATIVE)
    receipt = verify(package, replay=True)
    assert receipt["temporary_reexport_files_compared"] == 301
    assert receipt["verify_only_reexports_derived_files"] is True
    assert receipt["saved_reexport_run"] is True
    assert all(receipt[key] == 0 for key in ("synthesis_calls", "native_estimator_calls", "optimizer_calls", "sampling_calls"))
    assert gate.inventory(package / gate.BUNDLE_RELATIVE) == before
