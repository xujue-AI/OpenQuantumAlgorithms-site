#!/usr/bin/env python3
"""Authenticate the fixed multiproduct saved companion without simulation."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASHES = {
    "src/openquantum/simulation/multiproduct.py": "810f4c8431845f8d8310ab56dc84564fd521bc5ccb6c7d6d7ac6e398ad1b211a",
    "scripts/generate_simulation_multiproduct.py": "bd68b40ad9c3745241094c4e52707c0e2b17230db9fb1f01ba08579a3d2d6489",
    "tests/test_simulation_multiproduct.py": "75ac9e582c18a06634793ceb601faf96c5171d222dde40ef485a45a6622d36e6",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
NOTEBOOK = "notebooks/simulation_multiproduct.ipynb"
NOTEBOOK_SOURCE_SHA256 = "41efe7567bd96b4499a33c942361f0dbd491d234c4ef405495b14a40c3384e21"
SUMMARY = "assets/generated/simulation/multiproduct-validation.json"
SUMMARY_SHA256 = "547668d70ee75eff1f47b67bb61d79506a09e0c03bfeb23e2738d1b43187deae"
BUNDLES = (
    ("multiproduct", 86, "ec73967168a1d80fd70439c872f80f174a506f99875c234c6bd6f2a3a973f6b7",
     "fd0988cc877bc70e443678dad7507d565a857b2873a0f9168179b32c195f28ac"),
    ("multiproduct-large", 16, "c3b5f06b9fb9a3a10f6d5fd70ea37e55e21a0c5cb7ce00c74fd553bea8282f5f",
     "78be2e540b642b42a079ffb78bccda563a95859ed21429dd2bc598149e1d6414"),
)
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def _root(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path != path.resolve():
        raise ValueError("Unsafe multiproduct root")
    return path


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name):
        raise ValueError("Unsafe multiproduct file path")
    path = root / relative
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Missing or linked multiproduct file: " + name)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise ValueError("Unsafe or oversized multiproduct file: " + name)
    def identity(info):
        return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
                info.st_mtime_ns, info.st_ctime_ns, getattr(info, "st_birthtime", None))
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or identity(opened) != identity(before):
            raise ValueError("Multiproduct file changed before read: " + name)
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(descriptor)
        if (len(raw) != before.st_size or len(raw) > limit or identity(after) != identity(before)
                or path.is_symlink() or path.resolve() != path or identity(path.stat()) != identity(before)):
            raise ValueError("Multiproduct file changed during read: " + name)
    finally:
        os.close(descriptor)
    return raw


def notebook_source_sha256(raw):
    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if notebook.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 14:
        raise ValueError("Multiproduct notebook cell contract changed")
    projection = []
    for cell in cells:
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(part, str) for part in source):
            source = "".join(source)
        if (not isinstance(source, str) or not isinstance(cell.get("id"), str)
                or cell.get("cell_type") not in ("code", "markdown", "raw")):
            raise ValueError("Invalid multiproduct notebook source cell")
        projection.append({"id": cell["id"], "cell_type": cell["cell_type"], "source": source})
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _bundle(root, name, count, manifest_pin, archive_pin):
    relative = "assets/generated/simulation/" + name
    directory = _root(root / relative)
    raw = _read(directory, "manifest.json", 64 * 1024)
    if hashlib.sha256(raw).hexdigest() != manifest_pin:
        raise ValueError("Saved multiproduct manifest identity changed")
    manifest = json.loads(raw)
    records = manifest.get("files")
    if (manifest.get("schema") != "openquantum-fixed-multiproduct-bundle-v1"
            or not isinstance(records, dict) or len(records) != count - 1):
        raise ValueError("Saved multiproduct bundle contract changed")
    expected = set(records) | {"manifest.json"}
    actual = set()
    for path in directory.iterdir():
        if len(actual) >= count or path.name in actual:
            raise ValueError("Unexpected multiproduct directory entries")
        actual.add(path.name)
    if actual != expected:
        raise ValueError("Saved multiproduct bundle membership changed")
    contents = {"manifest.json": raw}
    total = len(raw)
    for filename, record in records.items():
        if Path(filename).name != filename or not isinstance(record, dict):
            raise ValueError("Invalid multiproduct manifest member")
        payload = _read(directory, filename)
        total += len(payload)
        if (total > MAX_BUNDLE_BYTES or record.get("bytes") != len(payload)
                or hashlib.sha256(payload).hexdigest() != record.get("sha256")):
            raise ValueError("Saved multiproduct payload identity changed: " + filename)
        contents[filename] = payload
    if total - len(raw) != manifest.get("exports_bytes_without_manifest"):
        raise ValueError("Saved multiproduct bundle byte accounting changed")
    archive_raw = _read(root, relative + "-bundle.zip", MAX_BUNDLE_BYTES)
    if hashlib.sha256(archive_raw).hexdigest() != archive_pin:
        raise ValueError("Saved multiproduct ZIP identity changed")
    with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
        members = archive.infolist()
        if len(members) != count or {entry.filename for entry in members} != expected:
            raise ValueError("Saved multiproduct ZIP membership changed")
        for entry in members:
            mode = entry.external_attr >> 16
            if (entry.is_dir() or entry.flag_bits & 1 or not stat.S_ISREG(mode)
                    or entry.file_size != len(contents[entry.filename]) or entry.file_size > MAX_FILE_BYTES):
                raise ValueError("Unsafe multiproduct ZIP member")
            with archive.open(entry) as stream:
                archived = stream.read(len(contents[entry.filename]) + 1)
            if archived != contents[entry.filename]:
                raise ValueError("Saved multiproduct ZIP payload differs from canonical file")
    return {"files": count, "bytes": total, "manifest_sha256": manifest_pin}


def verify_saved(root, *, source_root=None, notebook_roots=None, provenance=None):
    """Verify fixed data, source and notebook contracts; execute no quantum code."""
    root = _root(root)
    source_root = _root(root if source_root is None else source_root)
    notebook_roots = (source_root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two multiproduct notebook roots are required")
    hashes = None
    if provenance is not None:
        hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
        if (not isinstance(hashes, dict)
                or provenance.get("schema") != "openquantum-public-package-v2"):
            raise ValueError("Complete multiproduct public provenance is required")
    for filename, pin in SOURCE_HASHES.items():
        if (hashlib.sha256(_read(source_root, filename)).hexdigest() != pin
                or (hashes is not None and hashes.get(filename) != pin)):
            raise ValueError("Promoted multiproduct source identity changed: " + filename)
    for notebook_root in notebook_roots:
        raw = _read(_root(notebook_root), NOTEBOOK, 1024 * 1024)
        if hashes is not None and hashlib.sha256(raw).hexdigest() != hashes.get(NOTEBOOK):
            raise ValueError("Normal multiproduct notebook differs from complete provenance")
        if notebook_source_sha256(raw) != NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Normal multiproduct notebook source changed")
    summary_raw = _read(root, SUMMARY, 64 * 1024)
    if hashlib.sha256(summary_raw).hexdigest() != SUMMARY_SHA256:
        raise ValueError("Saved multiproduct evidence summary changed")
    bundles = {name: _bundle(root, name, count, manifest_pin, archive_pin)
               for name, count, manifest_pin, archive_pin in BUNDLES}
    return {"status": "saved_component_verified", "bundles": bundles,
            "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
            "new_quantum_execution": False, "physical_estimation_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify_saved(args.root), indent=2))


if __name__ == "__main__":
    main()
