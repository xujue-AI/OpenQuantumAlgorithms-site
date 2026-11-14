#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the public degree-two DQI packet using saved files only.

The archive preserves the repository hierarchy and the original closed
twenty-two-file source bundle. This wrapper checks its member graph, curated
acceptance, exact executed companion and unchanged saved-reader source. It
imports no producer or verifier, evaluates no angle, reconstructs no quantum
state or probability, runs no test or notebook, and supplies no publication or
portable scientific-replay certificate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import time
import types
import zipfile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/optimization/dqi-degree-two"
PUBLIC = "assets/generated/optimization/dqi-degree-two-public"
MANIFEST = PUBLIC + "/projection-manifest.json"
ARCHIVE = PUBLIC + "/dqi-degree-two-download.zip"
DOWNLOAD = PUBLIC + "/download.json"
NOTEBOOK = "notebooks/optimization_dqi_degree_two.ipynb"
READER = "scripts/verify_dqi_degree_two_saved.py"
SELF = "scripts/verify_dqi_degree_two_public_saved.py"
SOURCE_FILES = frozenset({
    "src/openquantum/optimization_dqi_degree_two.py",
    "src/openquantum/optimization_dqi_degree_two_verify.py",
    "scripts/generate_optimization_dqi_degree_two.py",
    "scripts/validate_optimization_dqi_degree_two.py", READER,
    "tests/test_optimization_dqi_degree_two.py",
    "tests/test_optimization_dqi_degree_two_boundaries.py", NOTEBOOK,
})
BUNDLE_FILES = frozenset({
    "optimization_dqi_degree_two.py", "optimization_dqi_degree_two_verify.py",
    "generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
    "verify_dqi_degree_two_saved.py", "optimization_dqi_degree_two.ipynb",
    "requirements-lock.txt", "LICENSE.txt", "input-c5.json", "input-figure-eight.json",
    "bundle.json", "invocation-acceptance.json",
    "c5/operations.ndjson", "c5/circuit.qasm", "c5/ledger.json", "c5/manifest.json",
    "figure-eight/operations.ndjson", "figure-eight/circuit.qasm",
    "figure-eight/ledger.json", "figure-eight/manifest.json",
    "validation/validation.json", "validation/manifest.json",
})
META_FILES = frozenset({"README.md", "environment.json", "accepted-components.json",
                        "projection-manifest.json", "dqi-degree-two-download.zip", "download.json"})
EXTRACTED_META_FILES = META_FILES - {"dqi-degree-two-download.zip", "download.json"}
PAYLOAD_FILES = SOURCE_FILES | {SELF, "requirements-lock.txt", "LICENSE", "pyproject.toml"} | {
    BUNDLE + "/" + name for name in BUNDLE_FILES
} | {PUBLIC + "/" + name for name in ("README.md", "environment.json", "accepted-components.json")}
ARCHIVE_FILES = PAYLOAD_FILES | {MANIFEST}
ARCHIVE_DIRECTORIES = frozenset(str(parent) for name in ARCHIVE_FILES
    for parent in PurePosixPath(name).parents if str(parent) != ".")
BUNDLE_SHA = "6981e4dba5502925dc4cd3ccc9d709e5be43adba7a9673b78ce22d54e9019395"
COMPONENT_SHA = "158edceadd0ca586cd6da85b7e38c3420566a0c8c2c5b8f941210bacb6a29db1"
TEMPLATE_SHA = "18ac25e51a13e6bf74b188c75ba4f58f2c699e0c734c9c6872dc9f9a7e062c66"
EXECUTED_SHA = "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"
PROJECTION_SHA = "3cc9682dded7381b2895cbca21f68fdc4581a94aac33ae36cad57c333577928f"
READER_SHA = "803ea2b04d1d5c141361cdebb98f7996b973e268625df340210d1d61042fed56"
PYPROJECT_SHA = "fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404"
TEST_PINS = {
    "tests/test_optimization_dqi_degree_two.py": {"bytes": 13519, "sha256": "0793f56fa44bcff9e93f72077cc65189882b69097603526f16573377f3d250ca"},
    "tests/test_optimization_dqi_degree_two_boundaries.py": {"bytes": 10414, "sha256": "6ee2b6f222e583f5fd8dc1f06b85f36636016ed889f2759e19d2a48d8361f7b1"},
}
SCHEMA = "openquantum.dqi.degree_two_public_projection.v1"
STATUS = "accepted_finite_components_and_separate_saved_companion"
SHA = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
MAX_FILE = 2 << 20
MAX_MANIFEST = 128 << 10
MAX_PACKET = 8 << 20
MAX_FILES = 64
MAX_SECONDS = 10.0
PRIVATE_MARKERS = (b"/" + b"Users/", b"/private/" + b"tmp/", b"/" + b"tmp/",
    b"docs/" + b"coordination/", b"ssh" + b"://", b"PAGES_" + b"DEPLOY_KEY",
    b"BEGIN " + b"OPENSSH PRIVATE KEY", b"BEGIN " + b"RSA PRIVATE KEY",
    b"gh" + b"p_", b"github_" + b"pat_")
PRIVATE_KEYS = frozenset({"pid", "ppid", "cwd", "argv", "session_id", "input_root",
    "program", "resolved_program", "raw_stdout", "raw_stderr", "grant", "claim",
    "ssh_coordinates", "username", "hostname", "absolute_path"})
ZERO_WORK = frozenset({"pure_test_invocations", "producer_invocations", "generator_invocations",
    "validator_invocations", "complete_statevector_evolutions", "boolean_decoder_mappings",
    "small_ccx_reference_cases", "primitive_unitary_applications", "phase_reference_cases",
    "sampled_outcomes", "native_synthesis_calls", "direct_native_estimator_calls",
    "gpu_jobs", "remote_jobs", "build_invocations", "publication_invocations"})


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _pin(raw):
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result
    def reject(_value):
        raise ValueError("Nonfinite JSON constants are excluded")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)


def _relative(value):
    _require(type(value) is str and len(value) <= 512, "Bounded relative file name required")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value
             and all(part not in {".", "..", ""} for part in path.parts),
             "File name redirects outside the packet")
    return value


def _pin_shape(value, cap=MAX_FILE):
    _require(type(value) is dict and set(value) == {"bytes", "sha256"}
             and type(value["bytes"]) is int and 0 < value["bytes"] <= cap
             and type(value["sha256"]) is str and SHA.fullmatch(value["sha256"]),
             "A bounded byte fingerprint is required")


def _read(path, cap, check):
    path = Path(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_size <= cap,
             "Only a bounded ordinary file may be read")
    for parent in path.parents:
        _require(stat.S_ISDIR(parent.lstat().st_mode), "A file parent redirects")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(fd)
        _require(stat.S_ISREG(opened.st_mode), "File kind changed during open")
        pieces, size = [], 0
        while True:
            check()
            piece = os.read(fd, min(65536, cap + 1 - size))
            if not piece:
                break
            size += len(piece)
            _require(size <= cap, "File exceeded its read limit")
            pieces.append(piece)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns)
    _require(identity(before) == identity(opened) == identity(after) and size == after.st_size,
             "File changed during bounded read")
    return b"".join(pieces)


def _closed(directory, expected, directories, check):
    root = Path(directory)
    _require(stat.S_ISDIR(root.lstat().st_mode), "Ordinary packet directory required")
    seen, seen_dirs, pending = set(), set(), [root]
    while pending:
        check()
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                check()
                relative = str(Path(entry.path).relative_to(root))
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    _require(relative in directories and relative not in seen_dirs,
                             "Unknown or duplicate packet directory")
                    seen_dirs.add(relative)
                    pending.append(Path(entry.path))
                else:
                    _require(stat.S_ISREG(mode) and relative in expected and relative not in seen
                             and len(seen) < MAX_FILES, "Extra, duplicate or redirected packet file")
                    seen.add(relative)
    _require(seen == expected and seen_dirs == directories, "Closed packet members changed")


def _privacy(raw):
    _require(not any(marker in raw for marker in PRIVATE_MARKERS), "Private material appears in public bytes")


def _privacy_json(value):
    pending, visited = [value], 0
    while pending:
        visited += 1
        _require(visited <= 200000, "Public JSON graph exceeds its inspection bound")
        item = pending.pop()
        if type(item) is dict:
            _require(len(item) <= 512 and not PRIVATE_KEYS.intersection(item),
                     "Private field appears in public metadata")
            pending.extend(item.values())
        elif type(item) is list:
            _require(len(item) <= 8192, "Public JSON array exceeds its inspection bound")
            pending.extend(item)
        elif type(item) is str:
            _privacy(item.encode())
        elif type(item) is float:
            _require(math.isfinite(item), "Public JSON includes a nonfinite scalar")


def _project(notebook, *, executed):
    _require(type(notebook) is dict and notebook.get("nbformat") == 4
             and type(notebook.get("cells")) is list and len(notebook["cells"]) == 8,
             "Original eight-cell notebook graph required")
    projection, identities, counts = [], [], []
    for cell in notebook["cells"]:
        _require(type(cell) is dict and cell.get("cell_type") in {"markdown", "code"}
                 and type(cell.get("id")) is str and 0 < len(cell["id"]) <= 64,
                 "Original notebook cell identities required")
        source = cell.get("source")
        if type(source) is list:
            _require(all(type(line) is str for line in source), "Notebook source arrays must contain text")
            source = "".join(source)
        _require(type(source) is str and len(source.encode()) <= 64 << 10, "Bounded notebook text required")
        projection.append({"cell_type": cell["cell_type"], "source": source})
        identities.append((cell["id"], cell["cell_type"]))
        if cell["cell_type"] == "code":
            counts.append(cell.get("execution_count"))
            outputs = cell.get("outputs")
            _require(type(outputs) is list and len(outputs) <= 16
                     and all(type(output) is dict and output.get("output_type") != "error" for output in outputs),
                     "Notebook error output or malformed output graph")
            if not executed:
                _require(cell.get("execution_count") is None and not outputs, "Raw source template changed")
    _require(len({identity for identity, _kind in identities}) == 8 and len(counts) == 4,
             "Unique eight-cell identities and four code cells required")
    if executed:
        _require(counts == [1, 2, 3, 4] and all(type(count) is int for count in counts),
                 "Genuine four-cell completion required")
    raw = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    _require(_sha(raw) == PROJECTION_SHA, "Joined notebook source projection changed")
    return identities


def _acceptance(value):
    _require(type(value) is dict and value.get("schema") == "openquantum.dqi.degree_two_public_acceptance.v1"
             and value.get("status") == STATUS, "Public finite acceptance schema changed")
    original, continuation = value.get("original_attempt"), value.get("saved_companion_continuation")
    _require(type(original) is dict and original.get("outer_exit") == 1 and type(original.get("outer_exit")) is int
             and original.get("whole_sequence_status") == "failed_private_sequence"
             and original.get("spent") is True and original.get("whole_sequence_accepted") is False,
             "Original failed and spent attempt must remain failed")
    stages = original.get("stage_exits")
    _require(type(stages) is list and len(stages) == 5
             and [row.get("stage") for row in stages] == ["pure-tests", "generator-c5", "validator-c5", "generator-figure-eight", "saved-notebook"]
             and all(type(row.get("exit")) is int and row["exit"] == (1 if index == 4 else 0)
                     and type(row.get("invocations")) is int and row["invocations"] == 1 for index, row in enumerate(stages)),
             "Recorded original component sequence changed")
    _require(value.get("passed_pure_test_cases") == 170 and type(value.get("passed_pure_test_cases")) is int,
             "Recorded 170-case pure-test result changed")
    counters = value.get("scientific_component_counters")
    expected = {"verification_invocations": 1, "complete_statevector_evolutions": 1,
                "state_amplitudes": 4096, "boolean_decoder_mappings": 512,
                "small_ccx_reference_cases": 1, "small_ccx_matrix_dimension": 8,
                "primitive_unitary_applications": 1773}
    _require(type(counters) is dict and set(counters) == set(expected)
             and all(type(counters[k]) is int and counters[k] == n for k, n in expected.items()),
             "Accepted finite component workload changed")
    _require(type(continuation) is dict and continuation.get("outer_exit") == 0
             and type(continuation.get("outer_exit")) is int and continuation.get("child_exit") == 0
             and type(continuation.get("child_exit")) is int and continuation.get("kernel_invocations") == 1
             and type(continuation.get("kernel_invocations")) is int
             and continuation.get("kernel_autorestart") is False
             and continuation.get("ordered_cell_ids_and_types_preserved") is True
             and continuation.get("code_cell_execution_counts") == [1, 2, 3, 4]
             and all(type(n) is int for n in continuation["code_cell_execution_counts"])
             and continuation.get("source_projection_sha256") == PROJECTION_SHA
             and continuation.get("executed_notebook_sha256") == EXECUTED_SHA
             and type(continuation.get("executed_notebook_bytes")) is int
             and continuation.get("executed_notebook_bytes") == 19950
             and continuation.get("original_template_sha256") == TEMPLATE_SHA,
             "Separate genuine saved-companion completion changed")
    zeros = continuation.get("new_scientific_work")
    _require(type(zeros) is dict and set(zeros) == ZERO_WORK
             and all(type(value) is int and value == 0 for value in zeros.values()),
             "Saved companion may not claim new scientific work")
    parent = value.get("parent_acceptance")
    _require(type(parent) is dict and set(parent) == {"finite_c5_component", "separate_genuine_saved_companion",
                "original_whole_attempt_reclassified", "scope"}
             and parent["finite_c5_component"] is True and parent["separate_genuine_saved_companion"] is True
             and parent["original_whole_attempt_reclassified"] is False
             and type(parent["scope"]) is str
             and parent["scope"] == "finite floating component checks and saved-file companion only",
             "Parent acceptance scope changed")
    for flag in ("optimization_advantage_claim", "general_graph_quality_claim", "physical_resources_computed",
                 "rotation_synthesis_executed", "formal_theorem_reproduced", "formal_interval_certificate",
                 "qpu_execution", "sampled_outcomes", "old_degree_one_evidence_reexecuted", "publication_complete"):
        _require(value.get(flag) is False, "Unsupported public acceptance claim")
    _require(value.get("portable_scientific_replay_validated") is False
             and value.get("public_projection_is_saved_evidence_only") is True,
             "Portable scientific-replay scope changed")
    return value


def _source_path(relative, artifact_root, source_root, notebook_path, check):
    if relative == NOTEBOOK:
        candidates = (Path(notebook_path),) if notebook_path is not None else (artifact_root / relative, source_root / relative)
        for path in candidates:
            if path.is_file() and not path.is_symlink():
                if _sha(_read(path, 512 << 10, check)) == EXECUTED_SHA:
                    return path
        raise ValueError("Exact genuine executed notebook bytes are unavailable")
    if relative.startswith(("src/", "scripts/", "tests/")) or relative == "pyproject.toml":
        return source_root / relative
    if relative == "requirements-lock.txt":
        return artifact_root / BUNDLE / relative
    if relative == "LICENSE":
        return artifact_root / BUNDLE / "LICENSE.txt"
    return artifact_root / relative


def _archive(path, expected, payload, check):
    raw = _read(path, MAX_PACKET, check)
    if expected is not None:
        _require(_pin(raw) == expected, "Archive fingerprint changed")
    total, seen = 0, set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        _require(not archive.comment, "Archive comment is not part of the closed packet")
        entries = archive.infolist()
        _require(len(entries) == len(ARCHIVE_FILES) <= MAX_FILES, "Closed archive member count changed")
        for entry in entries:
            check()
            name = _relative(entry.filename)
            _require(name in ARCHIVE_FILES and name not in seen and not entry.is_dir()
                     and entry.compress_type == zipfile.ZIP_DEFLATED and not entry.flag_bits & 1
                     and entry.date_time == (2026, 11, 14, 0, 0, 0)
                     and entry.create_system == 3 and not entry.extra and not entry.comment
                     and stat.S_ISREG(entry.external_attr >> 16), "Archive member shape or deterministic metadata changed")
            seen.add(name)
            _require(0 < entry.file_size <= MAX_FILE, "Archive member exceeds its decompressed bound")
            total += entry.file_size
            _require(total <= MAX_PACKET, "Archive decompressed-byte bound exceeded")
            with archive.open(entry) as stream:
                content = stream.read(MAX_FILE + 1)
            _require(len(content) == entry.file_size and content == payload[name],
                     "Archive member differs from the scoped public file")
    _require(seen == ARCHIVE_FILES, "Archive closed member set changed")
    return len(raw), total


def verify_public_packet(artifact_root, *, source_root=None, notebook_path=None,
                        expected_manifest_sha256=None, archive_path=None, time_limit_seconds=10.0):
    """Verify saved public bytes; separate source roots support static-site exports."""
    _require(type(time_limit_seconds) in (int, float) and math.isfinite(time_limit_seconds)
             and 0 < time_limit_seconds <= MAX_SECONDS, "Saved-reader deadline cannot be widened")
    started = time.monotonic()
    def check():
        _require(time.monotonic() < started + time_limit_seconds, "Saved public check deadline elapsed")
    artifact_root = Path(artifact_root).absolute()
    source_root = Path(source_root).absolute() if source_root is not None else artifact_root
    _closed(artifact_root / BUNDLE, BUNDLE_FILES, {"c5", "figure-eight", "validation"}, check)
    archive_present = (artifact_root / ARCHIVE).exists() or (artifact_root / ARCHIVE).is_symlink()
    descriptor_present = (artifact_root / DOWNLOAD).exists() or (artifact_root / DOWNLOAD).is_symlink()
    _require(archive_present == descriptor_present, "Archive and external download descriptor must remain paired")
    extracted = not archive_present
    _closed(artifact_root / PUBLIC, EXTRACTED_META_FILES if extracted else META_FILES, set(), check)
    if extracted:
        _closed(artifact_root, ARCHIVE_FILES, ARCHIVE_DIRECTORIES, check)
    manifest_raw = _read(artifact_root / MANIFEST, MAX_MANIFEST, check)
    manifest_sha = _sha(manifest_raw)
    if expected_manifest_sha256 is not None:
        _require(type(expected_manifest_sha256) is str and SHA.fullmatch(expected_manifest_sha256)
                 and manifest_sha == expected_manifest_sha256, "Externally frozen public manifest changed")
    manifest = _json(manifest_raw)
    _require(type(manifest) is dict and manifest.get("schema") == SCHEMA
             and manifest.get("status") == STATUS and type(manifest.get("files")) is dict
             and set(manifest["files"]) == PAYLOAD_FILES, "Closed public projection manifest changed")
    payload, total = {MANIFEST: manifest_raw}, len(manifest_raw)
    for name, pin in sorted(manifest["files"].items()):
        check()
        _relative(name)
        _pin_shape(pin)
        raw = _read(_source_path(name, artifact_root, source_root, notebook_path, check), MAX_FILE, check)
        _require(_pin(raw) == pin, "Scoped public file fingerprint changed: " + name)
        _privacy(raw)
        total += len(raw)
        _require(total <= MAX_PACKET, "Public scoped-file bytes exceed their bound")
        payload[name] = raw
        if name.endswith((".json", ".ipynb")):
            _privacy_json(_json(raw))
    _privacy(manifest_raw)
    _privacy_json(manifest)
    _require(_sha(payload[BUNDLE + "/bundle.json"]) == BUNDLE_SHA
             and _sha(payload[BUNDLE + "/invocation-acceptance.json"]) == COMPONENT_SHA
             and _sha(payload[BUNDLE + "/optimization_dqi_degree_two.ipynb"]) == TEMPLATE_SHA
             and _sha(payload[NOTEBOOK]) == EXECUTED_SHA
             and _sha(payload[READER]) == READER_SHA, "Original and genuine companion identities changed")
    for name in SOURCE_FILES - {NOTEBOOK}:
        if name in TEST_PINS:
            _require(_pin(payload[name]) == TEST_PINS[name], "Original pure-test source identity changed")
        else:
            _require(payload[name] == payload[BUNDLE + "/" + PurePosixPath(name).name],
                     "Repository scientific source differs from its frozen flat snapshot")
    _require(payload["requirements-lock.txt"] == payload[BUNDLE + "/requirements-lock.txt"]
             and payload["LICENSE"] == payload[BUNDLE + "/LICENSE.txt"]
             and len(payload["pyproject.toml"]) == 705 and _sha(payload["pyproject.toml"]) == PYPROJECT_SHA,
             "Original lock, project license or configuration changed")
    template = _json(payload[BUNDLE + "/optimization_dqi_degree_two.ipynb"])
    executed = _json(payload[NOTEBOOK])
    _require(_project(template, executed=False) == _project(executed, executed=True),
             "Genuine companion cell identity/order changed")
    acceptance = _acceptance(_json(payload[PUBLIC + "/accepted-components.json"]))
    environment = _json(payload[PUBLIC + "/environment.json"])
    _require(environment.get("schema") == "openquantum.dqi.degree_two_public_environment.v1"
             and environment.get("python_version") == "3.12.14"
             and environment.get("requirements_lock") == _pin(payload["requirements-lock.txt"])
             and environment.get("scientific_package_context_executed") is False,
             "Public locked-environment or portability scope changed")
    archive_bytes = expanded_bytes = None
    if not extracted:
        descriptor_raw = _read(artifact_root / DOWNLOAD, MAX_MANIFEST, check)
        descriptor = _json(descriptor_raw)
        _privacy(descriptor_raw)
        _privacy_json(descriptor)
        _require(type(descriptor) is dict and descriptor.get("schema") == "openquantum.dqi.degree_two_public_download.v1"
                 and descriptor.get("manifest_sha256") == manifest_sha
                 and descriptor.get("archive_path") == ARCHIVE and descriptor.get("members") == len(ARCHIVE_FILES)
                 and type(descriptor.get("members")) is int, "Download descriptor changed")
        _pin_shape(descriptor.get("archive"), MAX_PACKET)
        archive_bytes, expanded_bytes = _archive(archive_path or artifact_root / ARCHIVE,
                                                descriptor["archive"], payload, check)
    elif archive_path is not None:
        archive_bytes, expanded_bytes = _archive(archive_path, None, payload, check)
    # Load only the hash-bound standard-library saved reader, never a producer
    # or scientific verifier. This call performs saved-file inspection only.
    module = types.ModuleType("degree_two_public_original_saved_reader")
    module.__file__ = str(artifact_root / BUNDLE / "verify_dqi_degree_two_saved.py")
    exec(compile(payload[READER], module.__file__, "exec"), module.__dict__)
    saved = module.verify_saved_bundle(artifact_root / BUNDLE, expected_manifest_sha256=BUNDLE_SHA)
    _require(saved.get("status") == "passed_saved_degree_two_bundle"
             and saved.get("scientific_work_reexecuted") is False, "Original saved reader did not return normally")
    c5, figure, validation = acceptance.get("c5"), acceptance.get("figure_eight"), saved["validation"]
    _require(type(c5) is dict and type(figure) is dict
             and c5.get("source_counts") == saved["c5_ledger"]["source_counts"]
             and figure.get("source_counts") == saved["figure_eight_ledger"]["source_counts"],
             "Curated source-count tables differ from the saved component")
    for name in ("source_wires", "source_unitary_primitives", "source_materialized_operations"):
        _require(type(c5.get(name)) is int and c5[name] == saved["c5_ledger"][name]
                 and type(figure.get(name)) is int and figure[name] == saved["figure_eight_ledger"][name],
                 "Curated source ledger differs from the saved component")
    _require(c5.get("observed_optimum_probability") == validation["optimum_probability"]
             and type(c5.get("observed_optimum_probability")) is float
             and c5.get("observed_expected_objective") == validation["expected_objective"]
             and type(c5.get("observed_expected_objective")) is float
             and c5.get("final_max_state_entry_discrepancy") == validation["checkpoints"]["final"]["max_state_entry_discrepancy"]
             and c5.get("final_auxiliary_leakage_probability") == validation["checkpoints"]["final"]["auxiliary_leakage_probability"]
             and c5.get("independent_optimum_probability") == "18/19"
             and c5.get("independent_expected_objective") == "74/19"
             and figure.get("generation_only") is True and figure.get("simulation_performed") is False
             and figure.get("quality_or_repetition_law_attached") is False,
             "Curated finite observations or generation-only scope changed")
    check()
    return {"schema": "openquantum.dqi.degree_two_public_saved_check.v1",
            "status": "passed_saved_public_degree_two_packet", "public_manifest_sha256": manifest_sha,
            "scoped_files": len(ARCHIVE_FILES), "archive_members": len(ARCHIVE_FILES),
            "packet_mode": "standalone_extracted" if extracted else "repository_with_closed_archive",
            "archive_checked": archive_bytes is not None,
            "archive_bytes": archive_bytes, "expanded_archive_bytes": expanded_bytes,
            "original_bundle_files": 22, "genuine_companion_code_cells": 4,
            "notebook_source_projection_sha256": PROJECTION_SHA,
            "scope": "saved bytes, schemas and recorded outcomes only; no scientific or notebook replay",
            "scientific_work_reexecuted": False, "notebook_executed": False,
            "portable_scientific_replay_validated": False, "publication_complete": False,
            "elapsed_seconds": time.monotonic() - started}


def _expired(_signum, _frame):
    raise TimeoutError("Public saved-reader process deadline elapsed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--notebook-path", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--time-limit-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    try:
        _require(math.isfinite(args.time_limit_seconds) and 0 < args.time_limit_seconds <= MAX_SECONDS,
                 "Public saved-reader time limit cannot be widened")
        _require(all(hasattr(signal, name) for name in ("SIGALRM", "ITIMER_REAL", "setitimer")),
                 "A POSIX saved-reader process timer is required")
        signal.signal(signal.SIGALRM, _expired)
        signal.setitimer(signal.ITIMER_REAL, args.time_limit_seconds)
        result = verify_public_packet(args.artifact_root, source_root=args.source_root,
            notebook_path=args.notebook_path, expected_manifest_sha256=args.expected_manifest_sha256,
            archive_path=args.archive, time_limit_seconds=args.time_limit_seconds)
        raw = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        _require(len(raw) <= 2048, "Public saved-reader output exceeded its limit")
        os.write(sys.stdout.fileno(), raw)
        return 0
    except BaseException as error:
        raw = (json.dumps({"status": "failed_saved_public_degree_two_packet",
                          "reason": type(error).__name__, "scientific_work_reexecuted": False,
                          "notebook_executed": False, "publication_complete": False}, sort_keys=True) + "\n").encode()
        os.write(sys.stdout.fileno(), raw[:2048])
        return 1


if __name__ == "__main__":
    # Keep the deadline armed through final output and skip interpreter teardown.
    os._exit(main())
