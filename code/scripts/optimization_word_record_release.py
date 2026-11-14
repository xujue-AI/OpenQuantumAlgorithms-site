# SPDX-License-Identifier: MIT
"""Immutable public entry for one saved QAOA word-transport comparison.

Import is standard-library-only and performs no file reads or project imports.
The known record-manifest pin authenticates the verifier snapshot; the complete
inventory is authenticated before any producer or restoration code is compiled.
No entry point performs Native estimation, synthesis, repair, or optimization.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "ee4656d922dd6c6cc0db88ff386cabbd77c85a38c589d2479cbea07dc4fce00d"
MAX_SOURCE_BYTES = 2 * 1024 * 1024
CORE = "code/optimization_word_records.py"


def _read(root: Path, relative: str, maximum: int) -> bytes:
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Public word-record root must be a real directory")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ValueError("Public word-record path must be relative")
    path = root
    for part in relative_path.parts:
        if part in {".", ".."}:
            raise ValueError("Public word-record path escaped its root")
        path = path / part
        if path.is_symlink():
            raise ValueError("Linked public word-record path")
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_size > maximum:
        raise ValueError("Public word-record source is not a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)):
            raise ValueError("Public word-record source changed before read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        if len(raw) != metadata.st_size or len(raw) > maximum or after.st_size != metadata.st_size:
            raise ValueError("Public word-record source changed during bounded read")
        return raw
    finally:
        os.close(descriptor)


def bundle_path(root=ROOT) -> Path:
    root = Path(root).absolute()
    if (root / "record-manifest.json").is_file():
        return root
    return root / "assets/generated/optimization/word-transport"


def _core(root=ROOT):
    bundle = bundle_path(root)
    raw = _read(bundle, "record-manifest.json", MAX_SOURCE_BYTES)
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Immutable word-record manifest changed")
    manifest = json.loads(raw)
    descriptor = manifest["files"][CORE]
    source = _read(bundle, CORE, MAX_SOURCE_BYTES)
    if len(source) != descriptor["bytes"] or hashlib.sha256(source).hexdigest() != descriptor["sha256"]:
        raise ValueError("Authenticated word-record verifier changed")
    name = "_openquantum_word_records_" + descriptor["sha256"]
    spec = importlib.util.spec_from_file_location(name, bundle / CORE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(bundle / CORE), "exec"), module.__dict__)
    return bundle, module


def authenticate_public_word_transport(root=ROOT):
    bundle, core = _core(root)
    return core.authenticate_bundle(bundle, MANIFEST_SHA256)


def verify_public_word_transport(root=ROOT, *, replay=True) -> dict:
    bundle, core = _core(root)
    return core.verify_records(bundle, MANIFEST_SHA256, replay=replay)


def load_recorded_plan(root=ROOT) -> dict:
    """Authenticated in-memory historical schema for portable unit fixtures."""
    bundle, core = _core(root)
    authenticated = core.authenticate_bundle(bundle, MANIFEST_SHA256)
    schema = core._module(authenticated, "schema")
    return schema.restore_record(authenticated.json("native-plan.json"))


def load_source_records(root=ROOT) -> dict:
    bundle, core = _core(root)
    authenticated = core.authenticate_bundle(bundle, MANIFEST_SHA256)
    schema = core._module(authenticated, "schema")
    return {kind: schema.restore_record(authenticated.json(kind.replace("_", "-") + ".json"))
            for kind in ("source_recipe", "source_accounting", "source_validation", "source_artifact_manifest")}
