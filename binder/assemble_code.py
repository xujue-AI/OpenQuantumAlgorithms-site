#!/usr/bin/env python3
"""Reconstruct Binder's complete package from the verified public archive.

The website's ``code`` directory is a source preview. Its original
``PROVENANCE.json`` describes the complete ZIP, while ``PREVIEW.json`` binds
that ZIP's digest. Assembly uses only Python's standard library and runs
before installing the locked dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

ARCHIVE_PATH = "assets/generated/openquantum-code.zip"
PREVIEW_SCHEMA = "openquantum-source-preview-v1"
ARCHIVE_PREFIX = "openquantum-examples/"


def checked_file(root: Path, name: str) -> Path:
    """Require a normalized relative path without any linked component."""
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute()
            or ".." in relative.parts or relative.as_posix() != name):
        raise ValueError(f"Unsafe Binder package path: {name}")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"Linked Binder package path: {name}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing Binder package file: {name}")
    return path


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def audit_archive_members(archive: Path, provenance: dict, provenance_bytes: bytes) -> None:
    """Require exactly the manifest's regular-file members before import."""
    manifest = provenance.get("file_sha256")
    if provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(manifest, dict):
        raise ValueError("Missing complete Binder archive file manifest")
    for name in manifest:
        relative = PurePosixPath(name)
        if (not name or "\\" in name or relative.is_absolute()
                or ".." in relative.parts or relative.as_posix() != name
                or name == "PROVENANCE.json"):
            raise ValueError(f"Unsafe Binder archive manifest path: {name}")
    expected = {ARCHIVE_PREFIX + name for name in manifest}
    expected.add(ARCHIVE_PREFIX + "PROVENANCE.json")
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if len(members) != len(expected) or {item.filename for item in members} != expected:
            raise ValueError("Binder archive members differ from the complete manifest")
        for item in members:
            if item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG):
                raise ValueError(f"Binder archive member must be a regular file: {item.filename}")
        if zipped.read(ARCHIVE_PREFIX + "PROVENANCE.json") != provenance_bytes:
            raise ValueError("Binder archive provenance differs from the source preview")


def load_verifier(code: Path, provenance: dict, archive: Path):
    """Check the bootstrap verifier against both provenance and archive."""
    name = "scripts/verify_public_package.py"
    source = checked_file(code, name)
    source_bytes = source.read_bytes()
    manifest = provenance.get("file_sha256")
    if (provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Binder bootstrap verifier differs from archive provenance")
    with zipfile.ZipFile(archive) as zipped:
        if zipped.read(ARCHIVE_PREFIX + name) != source_bytes:
            raise ValueError("Binder bootstrap verifier differs from the complete archive")
    spec = importlib.util.spec_from_file_location("binder_complete_package_verifier", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the checked Binder package verifier")
    verifier = importlib.util.module_from_spec(spec)
    # Compile the exact checked bytes instead of consulting a cached .pyc.
    # This also leaves the preview untouched on subsequent validation failure.
    exec(compile(source_bytes, str(source), "exec"), verifier.__dict__)
    return verifier


def assemble(root: Path) -> dict:
    """Verify and stage the full archive, then replace the source preview."""
    root = Path(root).absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Binder project root must be a regular directory")
    root = root.resolve()
    code = root / "code"
    if code.is_symlink() or not code.is_dir():
        raise ValueError("Binder code directory must be a regular directory")
    provenance_bytes = checked_file(code, "PROVENANCE.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    if not isinstance(provenance, dict):
        raise ValueError("Binder archive provenance must contain an object")
    archive = checked_file(root, ARCHIVE_PATH)
    preview_path = code / "PREVIEW.json"
    is_preview = preview_path.exists() or preview_path.is_symlink()
    if is_preview:
        preview = json.loads(checked_file(code, "PREVIEW.json").read_text())
        if not isinstance(preview, dict) or preview.get("schema") != PREVIEW_SCHEMA:
            raise ValueError("Unexpected Binder source-preview schema")
        identity = preview.get("archive")
        if (not isinstance(identity, dict) or identity.get("path") != ARCHIVE_PATH
                or not isinstance(identity.get("sha256"), str)
                or identity["sha256"] != file_digest(archive)):
            raise ValueError("Binder archive digest/path differs from the source preview")
        if preview.get("source_commit") != provenance.get("source_commit"):
            raise ValueError("Binder preview and archive source commits differ")
    audit_archive_members(archive, provenance, provenance_bytes)
    verifier = load_verifier(code, provenance, archive)
    if not is_preview:
        # Historical publications had a complete code tree instead of a
        # preview. Verify it before accepting the matching archive.
        verifier.verify_manifest(code)

    staging = Path(tempfile.mkdtemp(prefix=".binder-code-assembly-", dir=root))
    remove_staging = True
    try:
        package = verifier.extract_package(archive, staging)
        verified = verifier.verify_manifest(package)
        if checked_file(package, "PROVENANCE.json").read_bytes() != provenance_bytes:
            raise ValueError("Extracted Binder archive provenance changed")
        previous = staging / "previous-code"
        # Only whole, already checked directories are renamed. A failed
        # replacement restores the original preview before propagating.
        code.rename(previous)
        try:
            package.rename(code)
        except BaseException:
            try:
                previous.rename(code)
            except BaseException as rollback_error:
                remove_staging = False
                raise RuntimeError("Binder replacement failed; original code remains in staging") from rollback_error
            raise
    finally:
        if remove_staging:
            shutil.rmtree(staging)
    return {"source_commit": verified.get("source_commit"),
            "file_count": len(verified["file_sha256"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = assemble(args.root)
    print(f"Verified complete Binder package: {result['file_count']} files", flush=True)


if __name__ == "__main__":
    main()
