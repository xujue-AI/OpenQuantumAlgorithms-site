"""Byte-only tests for the new publication gate, separate from frozen tests.

Fixtures copy accepted bytes into fresh temporary directories. They never
import scientific bundle sources, parse result matrices, run a display, spawn
a process or invoke a numerical producer.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_thermal_two_round_public.py"
ACCEPTED = ROOT / "assets/generated/simulation/detailed-balance-two-round"
PREFIX = "assets/generated/simulation/detailed-balance-two-round/"
PAYLOAD_NAMES = frozenset({
    "LICENSE.txt", "README.md", "UPSTREAM-LICENSES.md", "requirements-lock.txt",
    "results.json", "summary.json", "two-round-circuit.svg",
    "circuits/correction-heralded-step.qasm", "circuits/correction_lcu.qasm",
    "circuits/round_macro.qasm", "circuits/two_round_trial.qasm",
    "circuits/weak_dilation.qasm", "circuits/weak_reset_prefix.qasm",
    "sources/detailed_balance_access.py", "sources/detailed_balance_channel.py",
    "sources/detailed_balance_correction.py", "sources/detailed_balance_decay.py",
    "sources/detailed_balance_thermal_two_round.py",
    "sources/detailed_balance_thermal_two_round_reference.py",
    "sources/thermal_two_round_contract.py",
})
CORE_NAMES = PAYLOAD_NAMES | {"manifest.json", "manifest.sha256"}
MANIFEST_PIN = "e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377"
INACTIVE_FLAGS = (
    "scientific_work_executed", "scientific_work_reexecuted", "scientific_acceptance_recomputed", "reader_executed",
    "scientific_reader_executed", "metadata_display_executed", "numerical_work_executed",
    "numerical_replay_performed", "replay_performed", "notebook_executed",
    "kernel_started", "bundled_sources_imported", "bundled_sources_compiled",
)


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_thermal_two_round_registration_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bundle(tmp_path):
    directory = (tmp_path / "bundle").resolve()
    directory.mkdir()
    for name in sorted(CORE_NAMES):
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ACCEPTED / name).read_bytes())
    return directory


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def flipped(raw):
    assert raw
    return bytes([raw[0] ^ 1]) + raw[1:]


def provenance(bundle):
    return {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
            "file_sha256": {PREFIX + name: digest((bundle / name).read_bytes())
                            for name in CORE_NAMES}}


def preview(provenance):
    return {name: {"site_path": name, "sha256": value}
            for name, value in provenance["file_sha256"].items()}


def test_frozen_bundle_success_has_only_byte_authentication_flags(gate, bundle):
    report = gate.verify_full_bundle(bundle)
    assert report["full_byte_authenticated"] is True
    assert report["files"] == 22 and report["payload_files"] == 20
    assert report["bytes"] == 1562808 <= gate.MAX_TOTAL_BYTES == 2 * 1024 * 1024
    assert report["manifest_sha256"] == MANIFEST_PIN
    assert report["provenance_joined"] is False and report["canonical_site_joined"] is False
    assert all(report[name] is False for name in INACTIVE_FLAGS)
    assert gate.PAYLOAD_NAMES == PAYLOAD_NAMES and gate.CORE_NAMES == CORE_NAMES


def test_complete_package_provenance_is_joined_without_requiring_unrelated_files(gate, bundle):
    record = provenance(bundle)
    record["file_sha256"]["src/unrelated.py"] = "b" * 64
    report = gate.verify_full_bundle(bundle, provenance=record)
    assert report["full_byte_authenticated"] is True and report["provenance_joined"] is True
    assert report["canonical_site_joined"] is False


def test_canonical_preview_rows_use_the_existing_two_field_schema(gate, bundle):
    record = provenance(bundle)
    omitted = preview(record)
    omitted["assets/unrelated.json"] = {"site_path": "assets/unrelated.json", "sha256": "b" * 64}
    report = gate.verify_full_bundle(bundle, provenance=record, omitted_assets=omitted)
    assert report["full_byte_authenticated"] is True
    assert report["provenance_joined"] is True and report["canonical_site_joined"] is True


def test_gate_source_has_no_scientific_import_or_execution_path():
    tree = ast.parse(SCRIPT.read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".", 1)[0])
    assert imports <= {"__future__", "argparse", "hashlib", "json", "os", "pathlib", "signal", "stat", "time"}
    forbidden = {"exec", "eval", "compile", "__import__", "Popen", "run", "system", "popen",
                 "expm", "matrix_power", "eigh", "eig", "evolve", "run_check", "reset"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            assert name not in forbidden
    metadata = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_json_metadata")
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "loads"
             and isinstance(node.func.value, ast.Name) and node.func.value.id == "json"]
    assert len(loads) == 1 and loads[0] in list(ast.walk(metadata))


def test_only_manifest_metadata_is_decoded_and_scientific_payloads_are_raw(gate, bundle, monkeypatch):
    original = gate._json_metadata
    calls = []
    def inspect(raw):
        calls.append(raw)
        return original(raw)
    monkeypatch.setattr(gate, "_json_metadata", inspect)
    gate.verify_full_bundle(bundle)
    assert calls == [(bundle / "manifest.json").read_bytes()]


@pytest.mark.parametrize("name", sorted(PAYLOAD_NAMES))
def test_same_length_corruption_of_each_payload_is_rejected(gate, bundle, name):
    path = bundle / name
    original = path.read_bytes()
    path.write_bytes(flipped(original))
    assert path.stat().st_size == len(original)
    with pytest.raises(ValueError, match="Frozen thermal payload identity differs"):
        gate.verify_full_bundle(bundle)


@pytest.mark.parametrize("name", ["manifest.json", "manifest.sha256"])
def test_corrupted_manifest_or_sidecar_is_rejected(gate, bundle, name):
    path = bundle / name
    path.write_bytes(flipped(path.read_bytes()))
    with pytest.raises(ValueError, match="Frozen thermal manifest"):
        gate.verify_full_bundle(bundle)


@pytest.mark.parametrize("name", ["results.json", "manifest.json", "manifest.sha256"])
def test_missing_payload_or_metadata_is_rejected(gate, bundle, name):
    (bundle / name).unlink()
    with pytest.raises(ValueError, match="tree is incomplete"):
        gate.verify_full_bundle(bundle)


@pytest.mark.parametrize("kind", ["file", "directory", "nested_file"])
def test_unlisted_files_or_directories_are_rejected(gate, bundle, kind):
    if kind == "directory":
        (bundle / "unlisted").mkdir()
    elif kind == "nested_file":
        (bundle / "circuits/unlisted.qasm").write_bytes(b"unlisted")
    else:
        (bundle / "unlisted.json").write_bytes(b"unlisted")
    with pytest.raises(ValueError, match="Unlisted"):
        gate.verify_full_bundle(bundle)


@pytest.mark.parametrize("kind", ["root_alias", "directory_symlink", "file_symlink", "file_hardlink"])
def test_linked_entries_are_rejected_without_touching_accepted_assets(gate, bundle, tmp_path, kind):
    target = bundle
    if kind == "root_alias":
        target = tmp_path / "alias"
        target.symlink_to(bundle, target_is_directory=True)
    elif kind == "directory_symlink":
        moved = tmp_path / "outside-sources"
        (bundle / "sources").rename(moved)
        (bundle / "sources").symlink_to(moved, target_is_directory=True)
    elif kind == "file_symlink":
        moved = tmp_path / "outside-results.json"
        (bundle / "results.json").rename(moved)
        (bundle / "results.json").symlink_to(moved)
    else:
        copied = tmp_path / "outside-copy.json"
        copied.write_bytes((bundle / "results.json").read_bytes())
        (bundle / "results.json").unlink()
        os.link(copied, bundle / "results.json")
    with pytest.raises(ValueError):
        gate.verify_full_bundle(target)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Requires the POSIX gate platform")
def test_nonregular_fifo_is_rejected_before_any_file_read(gate, bundle, monkeypatch):
    path = bundle / "results.json"
    path.unlink()
    os.mkfifo(path)
    monkeypatch.setattr(gate, "_read_file", lambda *args: pytest.fail("Nonregular entry reached a read"))
    with pytest.raises(ValueError, match="nonregular"):
        gate.verify_full_bundle(bundle)


def test_oversized_payload_is_rejected_before_any_file_read(gate, bundle, monkeypatch):
    with (bundle / "results.json").open("wb") as output:
        output.truncate(2 * 1024 * 1024 + 1)
    monkeypatch.setattr(gate, "_read_file", lambda *args: pytest.fail("Oversized entry reached a read"))
    with pytest.raises(ValueError, match="oversized"):
        gate.verify_full_bundle(bundle)


def test_refreshed_manifest_sidecar_and_forged_provenance_cannot_change_the_pin(gate, bundle):
    result = bundle / "results.json"
    result.write_bytes(flipped(result.read_bytes()))
    manifest = json.loads((bundle / "manifest.json").read_bytes())
    manifest["files"]["results.json"]["sha256"] = digest(result.read_bytes())
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    (bundle / "manifest.json").write_bytes(raw)
    (bundle / "manifest.sha256").write_text(digest(raw) + "  manifest.json\n")
    refreshed = provenance(bundle)
    with pytest.raises(ValueError, match="Frozen thermal manifest identity differs"):
        gate.verify_full_bundle(bundle, provenance=refreshed, omitted_assets=preview(refreshed))


@pytest.mark.parametrize("change", ["schema", "missing", "stale"])
def test_invalid_or_incomplete_package_provenance_is_rejected(gate, bundle, change):
    record = provenance(bundle)
    if change == "schema":
        record["schema"] = "unverified"
    elif change == "missing":
        record["file_sha256"].pop(PREFIX + "results.json")
    else:
        record["file_sha256"][PREFIX + "results.json"] = "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        gate.verify_full_bundle(bundle, provenance=record)


def test_preview_joins_require_complete_package_provenance(gate, bundle):
    with pytest.raises(ValueError, match="require complete package provenance"):
        gate.verify_full_bundle(bundle, omitted_assets=preview(provenance(bundle)))


@pytest.mark.parametrize("change", ["missing", "path", "hash", "extra_field"])
def test_missing_changed_or_noncanonical_preview_joins_are_rejected(gate, bundle, change):
    record = provenance(bundle)
    omitted = preview(record)
    name = PREFIX + "results.json"
    if change == "missing":
        omitted.pop(name)
    elif change == "path":
        omitted[name]["site_path"] = "./" + name
    elif change == "hash":
        omitted[name]["sha256"] = "0" * 64
    else:
        omitted[name]["bytes"] = 1155156
    with pytest.raises(ValueError, match="Canonical thermal preview join differs"):
        gate.verify_full_bundle(bundle, provenance=record, omitted_assets=omitted)


def test_caller_provenance_and_preview_are_not_modified(gate, bundle):
    record = provenance(bundle)
    omitted = preview(record)
    before = copy.deepcopy((record, omitted))
    gate.verify_full_bundle(bundle, provenance=record, omitted_assets=omitted)
    assert (record, omitted) == before


def test_new_entry_during_authentication_is_rejected(gate, bundle, monkeypatch):
    original = gate._read_file
    def changed(directory, name, maximum, deadline):
        raw = original(directory, name, maximum, deadline)
        if name == "manifest.json":
            (bundle / "late.json").write_bytes(b"late")
        return raw
    monkeypatch.setattr(gate, "_read_file", changed)
    with pytest.raises(ValueError, match="Unlisted"):
        gate.verify_full_bundle(bundle)


def test_payload_changed_after_its_read_is_rejected(gate, bundle, monkeypatch):
    original = gate._read_file
    def changed(directory, name, maximum, deadline):
        raw = original(directory, name, maximum, deadline)
        if name == "results.json":
            path = bundle / name
            info = path.stat()
            path.write_bytes(flipped(raw))
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1000000000))
        return raw
    monkeypatch.setattr(gate, "_read_file", changed)
    with pytest.raises(ValueError, match="changed during full-byte authentication"):
        gate.verify_full_bundle(bundle)


def test_aggregate_budget_is_distinct_from_each_file_limit(gate, bundle, monkeypatch):
    monkeypatch.setattr(gate, "MAX_TOTAL_BYTES", 1562807)
    with pytest.raises(ValueError, match="aggregate byte bound"):
        gate.verify_full_bundle(bundle)


def test_deadline_is_checked_between_saved_byte_reads_without_sleeping(gate, bundle, monkeypatch):
    clock = {"now": 0}
    monkeypatch.setattr(gate, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
    original = gate._read_file
    def late(directory, name, maximum, deadline):
        raw = original(directory, name, maximum, deadline)
        if name == "manifest.json":
            clock["now"] = 11
        return raw
    monkeypatch.setattr(gate, "_read_file", late)
    with pytest.raises(TimeoutError, match="wall allowance"):
        gate.verify_full_bundle(bundle)


def test_first_close_failure_still_attempts_every_owned_handle(gate, bundle, monkeypatch):
    descriptors = [os.open(bundle / name, os.O_RDONLY)
                   for name in ("manifest.json", "summary.json", "results.json")]
    calls, closed = [], set()
    def close(descriptor):
        calls.append(descriptor)
        os.close(descriptor)
        closed.add(descriptor)
        if descriptor == descriptors[0]:
            raise OSError("Injected post-close failure")
    monkeypatch.setattr(gate, "os", SimpleNamespace(close=close))
    try:
        with pytest.raises(OSError, match="Injected post-close failure"):
            gate._close_handles(descriptors)
        assert calls == descriptors
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in descriptors:
            if descriptor not in closed:
                os.close(descriptor)


def test_pending_alarm_is_restored_only_after_every_owned_handle_is_closed(gate, bundle, monkeypatch):
    descriptors = [os.open(bundle / name, os.O_RDONLY) for name in ("manifest.json", "summary.json")]
    calls, closed = [], set()
    def close(descriptor):
        calls.append(("close", descriptor))
        os.close(descriptor)
        closed.add(descriptor)
    def mask(how, numbers):
        if how == signal.SIG_BLOCK:
            calls.append(("block", numbers))
            return set()
        calls.append(("restore", numbers))
        raise TimeoutError("Injected pending alarm after cleanup")
    proxy = SimpleNamespace(pthread_sigmask=mask, SIG_BLOCK=signal.SIG_BLOCK,
                            SIG_SETMASK=signal.SIG_SETMASK, SIGALRM=signal.SIGALRM,
                            SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIGHUP=signal.SIGHUP)
    monkeypatch.setattr(gate, "os", SimpleNamespace(close=close))
    monkeypatch.setattr(gate, "signal", proxy)
    try:
        with pytest.raises(TimeoutError, match="Injected pending alarm after cleanup"):
            gate._close_handles(descriptors)
        assert [call[0] for call in calls] == ["block", "close", "close", "restore"]
        assert [call[1] for call in calls if call[0] == "close"] == descriptors
        for descriptor in descriptors:
            with pytest.raises(OSError):
                os.fstat(descriptor)
    finally:
        for descriptor in descriptors:
            if descriptor not in closed:
                os.close(descriptor)
