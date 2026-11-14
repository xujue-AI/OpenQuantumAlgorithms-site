# SPDX-License-Identifier: MIT
"""Security and aggregate bound tests for immutable tail record verification."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tail_records_tests", ROOT / "scripts/tail_resource_records.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_regular_reader_rejects_links_and_fifo_without_blocking(tmp_path):
    root = tmp_path.resolve()
    (root / "ordinary").write_bytes(b"frozen bytes")
    assert gate.read_regular(root, "ordinary") == b"frozen bytes"
    os.link(root / "ordinary", root / "hardlink")
    with pytest.raises(ValueError, match="singly linked"):
        gate.read_regular(root, "ordinary")
    (root / "symlink").symlink_to(root / "ordinary")
    with pytest.raises(ValueError):
        gate.read_regular(root, "symlink")
    os.mkfifo(root / "fifo")
    started = time.monotonic()
    with pytest.raises(ValueError, match="regular"):
        gate.read_regular(root, "fifo")
    assert time.monotonic() - started < 2
    with pytest.raises(ValueError):
        gate.tree_names(root)


@pytest.mark.parametrize("name", ["../outside", "a/../b", "/absolute", "a//b", "./a", "a\\b", "a/", "x" * 513])
def test_reader_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe"):
        gate.read_regular(tmp_path.resolve(), name)


def test_reader_rejects_linked_ancestor_and_oversized_file(tmp_path):
    root = tmp_path.resolve()
    (root / "real").mkdir()
    (root / "real" / "file").write_bytes(b"12345")
    (root / "alias").symlink_to(root / "real", target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        gate.read_regular(root, "alias/file")
    with pytest.raises(ValueError, match="size"):
        gate.read_regular(root, "real/file", max_bytes=4)


def test_reader_rejects_replacement_during_read(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "file").write_bytes(b"before")
    original = gate.os.read
    changed = False
    def racing_read(descriptor, size):
        nonlocal changed
        value = original(descriptor, size)
        if not changed:
            changed = True
            (root / "replacement").write_bytes(b"after!")
            os.replace(root / "replacement", root / "file")
        return value
    monkeypatch.setattr(gate.os, "read", racing_read)
    with pytest.raises(ValueError, match="changed while reading"):
        gate.read_regular(root, "file")


def test_tree_rejects_unlisted_empty_directory(tmp_path):
    root = tmp_path.resolve()
    (root / "unexpected").mkdir()
    with pytest.raises(ValueError, match="unexpected input directory"):
        gate.tree_names(root, allowed_directories=set())


@pytest.mark.parametrize("data", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'[]', b'\xff'])
def test_json_rejects_duplicate_nonfinite_or_malformed_inputs(data):
    with pytest.raises(ValueError):
        gate.parse_json(data)


def test_json_reads_actual_large_complete_record():
    data = gate.read_regular(ROOT, "assets/generated/simulation/complete-resources/results.json")
    assert len(data) == 8_719_390
    result = gate.parse_json(data)
    assert len(result["cases"]) == 5


def test_bad_sidecar_rejected_before_helper_execution(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / gate.SIDECAR).write_bytes(b'{"wrong": true}')
    called = []
    monkeypatch.setattr(gate, "load_checked_module", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="reviewed immutable identity"):
        gate._verify_in_worker("instrument", root, ROOT, "0" * 64, False)
    assert called == []


@pytest.mark.parametrize("corrupt", ["record", "source"])
def test_pinned_sidecar_with_corruption_rejects_before_helpers(tmp_path, monkeypatch, corrupt):
    root = tmp_path.resolve()
    records, sources, _ = gate.collect_inputs("instrument", ROOT / gate.KINDS["instrument"]["directory"], ROOT,
                                             with_sidecar=(ROOT / gate.KINDS["instrument"]["directory"] / gate.SIDECAR).exists())
    directory = root / gate.KINDS["instrument"]["directory"]
    for name, value in records.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    for name, value in sources.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    sidecar = {"schema": "openquantum-saved-tail-resource-v1", "kind": "instrument",
               "producer_commit": gate.PRODUCER_COMMIT, "files": gate.inventory(records),
               "sources": gate.inventory(sources), "scope": gate.KINDS["instrument"]["scope"],
               "trust_boundaries": gate.TRUST_BOUNDARIES}
    data = json.dumps(sidecar).encode()
    (directory / gate.SIDECAR).write_bytes(data)
    target = directory / "damping-Z/report.json" if corrupt == "record" else root / "scripts/tail_resource_model.py"
    target.write_bytes(target.read_bytes() + b" ")
    def forbidden(*args):
        raise AssertionError("helper executed before complete authentication")
    monkeypatch.setattr(gate, "load_checked_module", forbidden)
    with pytest.raises(ValueError, match="differs"):
        gate._verify_in_worker("instrument", directory, root, gate.digest(data), False)


def test_watchdog_terminates_and_reaps_blocked_process(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    script = root / "blocked.py"
    script.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(gate, "BOOTSTRAP", script.read_text())
    monkeypatch.setattr(gate, "_tree_rss", lambda pid, **kwargs: 0)
    started = time.monotonic()
    with pytest.raises(ValueError, match="wall budget"):
        gate.verify_bundle(root, root, kind="instrument", expected_sha256="0" * 64,
                           max_wall_seconds=0.3)
    assert time.monotonic() - started < 6


def test_watchdog_rejects_output_flood(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    script = root / "noisy.py"
    script.write_text("import sys,time\nsys.stdout.write('x'*1100000)\nsys.stdout.flush()\ntime.sleep(30)\n")
    monkeypatch.setattr(gate, "BOOTSTRAP", script.read_text())
    monkeypatch.setattr(gate, "_tree_rss", lambda pid, **kwargs: 0)
    with pytest.raises(ValueError, match="output limit"):
        gate.verify_bundle(root, root, kind="instrument", expected_sha256="0" * 64,
                           max_wall_seconds=3)


def test_watchdog_rejects_fast_exiting_output_flood(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    monkeypatch.setattr(gate, "BOOTSTRAP", "import sys\nsys.stdout.write('x'*1100000)\nsys.stdout.flush()\n")
    monkeypatch.setattr(gate, "_tree_rss", lambda pid, **kwargs: 0)
    with pytest.raises(ValueError, match="output limit"):
        gate.verify_bundle(root, root, kind="instrument", expected_sha256="0" * 64,
                           max_wall_seconds=3)


def test_complete_authentication_is_inside_aggregate_watchdog(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    # The watchdog starts before the trusted bootstrap/complete-auth phase.
    monkeypatch.setattr(gate, "BOOTSTRAP", "import time\ntime.sleep(30)\nraise AssertionError('model must remain unreachable')\n")
    monkeypatch.setattr(gate, "_tree_rss", lambda pid, **kwargs: 0)
    started = time.monotonic()
    with pytest.raises(ValueError, match="wall budget"):
        gate.verify_bundle(root, root, kind="complete", expected_sha256="0" * 64,
                           max_wall_seconds=0.3, replay=True)
    assert time.monotonic() - started < 1


def test_watchdog_rejects_rss_limit(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    script = root / "idle.py"
    script.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(gate, "BOOTSTRAP", script.read_text())
    monkeypatch.setattr(gate, "_tree_rss", lambda pid, **kwargs: 11)
    with pytest.raises(ValueError, match="RSS budget"):
        gate.verify_bundle(root, root, kind="instrument", expected_sha256="0" * 64,
                           max_wall_seconds=3, max_tree_rss_kib=10)


def test_fixed_scopes_are_not_search_completeness_claims():
    scope = gate.KINDS["instrument"]["scope"]
    assert scope["reported_feasible_candidates"] == scope["stored_points"] + scope["unavailable_candidate_ledgers"]
    assert len(gate.expected_files("instrument")) == 26
    assert len(gate.expected_files("complete")) == 35
    assert any("absent raw Trace" in text for text in gate.TRUST_BOUNDARIES)


def test_partial_preview_authenticates_only_known_assets_from_canonical_root(tmp_path):
    kind = "instrument"
    records, sources, _ = gate.collect_inputs(kind, ROOT / gate.KINDS[kind]["directory"], ROOT)
    canonical = tmp_path.resolve()
    preview = canonical / "code"
    directory = canonical / gate.KINDS[kind]["directory"]
    for name, value in records.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    for name, value in sources.items():
        path = (canonical if name in gate.FROZEN_ASSET_FILES else preview) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    assert not (preview / "assets").exists()
    manifest = {"schema": "openquantum-saved-tail-resource-v1", "kind": kind,
                "producer_commit": gate.PRODUCER_COMMIT, "files": gate.inventory(records),
                "sources": gate.inventory(sources), "scope": gate.KINDS[kind]["scope"],
                "trust_boundaries": gate.TRUST_BOUNDARIES}
    data = json.dumps(manifest).encode()
    (directory / gate.SIDECAR).write_bytes(data)
    with pytest.raises(OSError):
        gate._authenticate_record(kind, directory, preview, gate.digest(data))
    actual_records, actual_sources, _, _ = gate._authenticate_record(
        kind, directory, preview, gate.digest(data), canonical_root=canonical)
    assert gate.inventory(actual_records) == gate.inventory(records)
    assert gate.inventory(actual_sources) == gate.inventory(sources)
    # Code/environment inputs never fall back to the canonical root.
    missing_code = preview / "src/openquantum/instrument_resources.py"
    missing_code.unlink()
    canonical_copy = canonical / "src/openquantum/instrument_resources.py"
    canonical_copy.parent.mkdir(parents=True)
    canonical_copy.write_bytes(sources["src/openquantum/instrument_resources.py"])
    with pytest.raises(OSError):
        gate._authenticate_record(kind, directory, preview, gate.digest(data), canonical_root=canonical)


def test_both_actual_platform_evidences_meet_reviewed_numerical_policy():
    # Actual initial proof records remain private and are optional to public
    # consumers. The independent policy tests ship synthetic negative cases.
    for kind in ("instrument", "complete"):
        local = ROOT / ".cache/tail-records" / f"local-{kind}-replay-evidence.json"
        linux = ROOT / ".cache/tail-records" / f"linux-{kind}-postflight.json"
        if not local.is_file() or not linux.is_file():
            pytest.skip("Private platform proof records are not part of the public package")
        checker = gate.load_checked_module("_tail_numeric_actual_test_" + kind,
                                          gate.read_regular(ROOT, "scripts/tail_resource_numerical.py"))
        policy = gate.parse_json(gate.read_regular(ROOT, gate.NUMERICAL_POLICY_FILES[kind]))
        local_evidence = gate.parse_json(gate.read_regular(ROOT, local.relative_to(ROOT).as_posix()))["replay"]
        linux_evidence = gate.parse_json(gate.read_regular(ROOT, linux.relative_to(ROOT).as_posix()))["typed_replay"]
        for evidence in (local_evidence, linux_evidence):
            result = checker.verify_numerical_replay(evidence, policy)
            assert result["status"] == "verified_observed_numerical_reproduction"


def _load_freezer(monkeypatch):
    monkeypatch.setitem(sys.modules, "tail_resource_records", gate)
    spec = importlib.util.spec_from_file_location("_tail_freezer_security_test", ROOT / "scripts/freeze_tail_resource_records.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freezer_retains_foreign_replacement_instead_of_named_rollback(tmp_path, monkeypatch):
    freezer = _load_freezer(monkeypatch)
    root = tmp_path.resolve()
    directory = root / gate.KINDS["instrument"]["directory"]
    directory.mkdir(parents=True)
    calls = 0
    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            foreign = directory / "foreign"
            foreign.write_bytes(b"foreign sidecar must survive")
            os.replace(foreign, directory / gate.SIDECAR)
        return {"record": b"frozen"}, {"source": b"frozen"}, {}
    monkeypatch.setattr(freezer, "collect_inputs", collect)
    with pytest.raises(ValueError, match="sidecar changed"):
        freezer.freeze("instrument", root)
    assert (directory / gate.SIDECAR).read_bytes() == b"foreign sidecar must survive"


def test_failed_freeze_candidate_is_retained_unvalidated(tmp_path, monkeypatch):
    freezer = _load_freezer(monkeypatch)
    root = tmp_path.resolve()
    directory = root / gate.KINDS["instrument"]["directory"]
    directory.mkdir(parents=True)
    calls = 0
    def collect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("post-audit failure")
        return {"record": b"frozen"}, {"source": b"frozen"}, {}
    monkeypatch.setattr(freezer, "collect_inputs", collect)
    with pytest.raises(ValueError, match="post-audit failure"):
        freezer.freeze("instrument", root)
    assert (directory / gate.SIDECAR).is_file()
