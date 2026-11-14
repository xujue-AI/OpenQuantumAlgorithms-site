#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Freeze existing dynamics records without claiming a new Native computation.

The preserved generator manifest reports the historical computation. No
external completion/measurement receipt was located for its exact bytes; this
freezer records that limitation and never substitutes the older 961e327 audit.

Output creation is exclusive. Failed output candidates remain unvalidated for
private review; retries must use a new absent filename. Named rollback is
unsafe because an inode check and pathname deletion cannot be made atomic.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import dynamics_record_release as gate

SCHEMA = gate.MANIFEST_SCHEMA
STATUS = "frozen_conditional_dynamics_record"


def _model(source_root, hashes):
    name = "scripts/dynamics_physical_records.py"
    path = gate.checked_file(source_root, name)
    data = gate.read_bounded(path)
    gate.require(hashlib.sha256(data).hexdigest() == hashes[name],
                 "model checker changed before freezing")
    module = types.ModuleType("_dynamics_freeze_checked_model")
    module.__file__ = str(path)
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


@contextmanager
def _pinned_output_parent(output):
    parent = gate.checked_directory(output.parent)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(parent, flags)
    try:
        initial = os.fstat(descriptor)
        identity = (initial.st_dev, initial.st_ino)
        _check_output_parent(output, descriptor, identity)
        yield descriptor, identity
    finally:
        os.close(descriptor)


def _check_output_parent(output, descriptor, identity):
    gate.checked_directory(output.parent)
    pinned = os.fstat(descriptor)
    current = os.stat(output.parent, follow_symlinks=False)
    gate.require(stat.S_ISDIR(pinned.st_mode) and stat.S_ISDIR(current.st_mode)
                 and (pinned.st_dev, pinned.st_ino) == identity == (current.st_dev, current.st_ino),
                 "manifest output parent changed after anchoring")


def _check_output_identity(output, parent_descriptor, parent_identity, target, identity, size):
    _check_output_parent(output, parent_descriptor, parent_identity)
    pinned = os.fstat(target)
    current = os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
    gate.require(stat.S_ISREG(pinned.st_mode) and stat.S_ISREG(current.st_mode)
                 and (pinned.st_dev, pinned.st_ino) == identity == (current.st_dev, current.st_ino),
                 "manifest output file changed before completion")
    gate.require(pinned.st_size == current.st_size == size, "anchored manifest output bytes differ")


def _check_anchored_manifest(output, parent_descriptor, parent_identity, target, identity, payload):
    _check_output_identity(output, parent_descriptor, parent_identity, target, identity, len(payload))
    os.lseek(target, 0, os.SEEK_SET)
    chunks, remaining_bytes = [], len(payload) + 1
    while remaining_bytes:
        block = os.read(target, min(65536, remaining_bytes))
        if not block:
            break
        chunks.append(block)
        remaining_bytes -= len(block)
    captured = b"".join(chunks)
    gate.require(captured == payload, "anchored manifest output bytes differ")
    _check_output_identity(output, parent_descriptor, parent_identity, target, identity, len(payload))
    return captured


@contextmanager
def _write_anchored_manifest(output, parent_descriptor, parent_identity, payload):
    gate.require(type(payload) is bytes and len(payload) <= gate.MAX_JSON_BYTES,
                 "immutable manifest size limit exceeded")
    _check_output_parent(output, parent_descriptor, parent_identity)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    target = os.open(output.name, flags, 0o600, dir_fd=parent_descriptor)
    try:
        initial = os.fstat(target)
        identity = (initial.st_dev, initial.st_ino)
        gate.require(stat.S_ISREG(initial.st_mode), "manifest output fd must be regular")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(target, remaining)
            gate.require(written > 0, "manifest output write made no progress")
            remaining = remaining[written:]
        _check_anchored_manifest(output, parent_descriptor, parent_identity, target, identity, payload)
        yield target, identity
    finally:
        # POSIX has no inode-conditional unlink. Named rollback could delete a
        # concurrent replacement, so failed candidates remain for caller review.
        os.close(target)


def freeze_record(records_dir, output, *, source_root=ROOT):
    """Bind old bytes and separate current replay sources; never reprice them."""
    records_dir = gate.checked_directory(records_dir)
    source_root = gate.checked_directory(source_root)
    output = Path(output).absolute()
    with _pinned_output_parent(output) as (parent_descriptor, parent_identity):
        try:
            os.stat(output.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError("Dynamics record: manifest output already exists")
        gate.require(output == records_dir / gate.MANIFEST_NAME
                     or not output.resolve().is_relative_to(records_dir),
                     "manifest output must be the root sidecar or outside recorded trees")
        return _freeze_record_anchored(records_dir, output, source_root, parent_descriptor, parent_identity)


def _freeze_record_anchored(records_dir, output, source_root, parent_descriptor, parent_identity):
    before = gate.core_inventory(records_dir)
    study_path = gate.checked_file(records_dir, "manifest.json")
    study_bytes = gate.read_bounded(study_path)
    study = gate.parse_json_bytes(study_bytes)
    gate.require(hashlib.sha256(study_bytes).hexdigest() == before["manifest.json"]["sha256"],
                 "original study changed after inventory capture")
    replay = gate.source_hashes(source_root, gate.REPLAY_SOURCE_FILES)
    frozen = gate.frozen_hashes(records_dir, source_root)
    verifiers = gate.source_hashes(source_root, gate.VERIFIER_FILES)
    adapter = gate.historical_adapter_file(records_dir, source_root)
    adapter_hash = gate.sha256(adapter)
    evidence_path = gate.source_reproduction_evidence_file(records_dir, source_root)
    evidence_bytes = gate.read_bounded(evidence_path)
    record = {
        "schema": SCHEMA, "status": STATUS, "files": before,
        "study_manifest_sha256": hashlib.sha256(study_bytes).hexdigest(),
        "native_provenance": study.get("provenance"), "native_evidence": dict(gate.NATIVE_EVIDENCE),
        "replay_sources": replay, "frozen_files": frozen,
        "historical_adapter": {"path": gate.HISTORICAL_ADAPTER_PATH, "sha256": adapter_hash},
        "verifier_sources": verifiers, "physical_validation": {}, "contract": gate.CONTRACT,
        "source_reproduction_evidence": {"path": gate.SOURCE_REPRODUCTION_EVIDENCE_PATH,
            "sha256": hashlib.sha256(evidence_bytes).hexdigest(), "bytes": len(evidence_bytes)},
    }
    # The same complete authentication used by the public gate precedes every
    # checked stdlib code load. No canonical replay or scientific helper runs.
    inputs = gate.authenticate_record_inputs(records_dir, source_root, record)
    gate.require(inputs["before"] == before and inputs["evidence_bytes"] == evidence_bytes,
                 "record/evidence snapshots changed before freezing")
    try:
        source_reproduction = gate.validate_source_reproduction(inputs)
        physical = _model(source_root, verifiers).validate_physical_records(records_dir, inputs["study"])
        record["physical_validation"] = physical
    finally:
        gate.recheck_authenticated_inputs(inputs)
        gate.require(gate.read_bounded(study_path) == study_bytes, "original study changed while freezing")
    encoded = json.dumps(record, indent=2, allow_nan=False).encode() + b"\n"
    gate.require(len(encoded) <= gate.MAX_JSON_BYTES, "immutable manifest size limit exceeded")
    with _write_anchored_manifest(output, parent_descriptor, parent_identity, encoded) as (target, output_identity):
        gate.recheck_authenticated_inputs(inputs)
        checked_output = _check_anchored_manifest(output, parent_descriptor, parent_identity, target, output_identity, encoded)
        return {"status": "existing conditional dynamics record frozen",
                "manifest": str(output), "manifest_sha256": hashlib.sha256(checked_output).hexdigest(),
                "original_files": len(before), "original_bytes_preserved": True,
                "recorded_producer_commit": gate.HISTORICAL_SOURCE_COMMIT,
                "external_native_completion_receipt_available": False,
                "native_reestimation_performed": False, "factory_search_repeated": False,
                "fitter_run": False, "source_reproduction_validation": source_reproduction,
                "canonical_quantum_validation_performed": False, "source_byte_equivalence_claimed": False,
                "physical_records_repriced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=ROOT / gate.RECORD_DIRECTORY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = freeze_record(args.records_dir, args.output or args.records_dir / gate.MANIFEST_NAME)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__": main()
