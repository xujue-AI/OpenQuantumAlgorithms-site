#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate retained local-spin bath files; no scientific code is imported."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/simulation/reusable-bath-local-spin"
ARCHIVE = "assets/generated/simulation/reusable-bath-local-spin-evidence.zip"
PRESENTATION = "assets/generated/simulation/reusable-bath-local-spin-presentation"
NOTEBOOK = "notebooks/simulation_reusable_bath_local_spin.ipynb"
MANIFEST_SHA256 = "63f43d784d7f6dfec8f743843961df1d072c049469068582a26696fb410938a7"
ARCHIVE_SHA256 = "8322220219c9bfe1046891fe8a28c163905cb50e618c36614d1470044c6d4fd3"
NOTEBOOK_SHA256 = "9046424a93fb286d123cca2b690d7e45bca90e80ba253c7f1504c925f5b0aada"
NOTEBOOK_SOURCE_SHA256 = "823f6000baaa0e67da22f87074106568ccb3283e14ca7432174dc33af41b434f"
PRESENTATION_HASHES = {
    "first-step.svg": "c584592e8d26f077d3e34f15759ffdc4c3ca790efafc4c00a11c15a255ef114b",
    "presentation.json": "a830e271d6782b4e368272a53216b1a4de0573193573be8bb26cbc470c87c91a",
}
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
BUNDLE_FILES = frozenset({
    "artifact-manifest.json", "artifact-manifest.sha256", "cases.json", "generation.json",
    "reference.json", "schedules.json", "validation.json",
    "source-exports/n1-ground.qasm", "source-exports/n1-thermal.qasm",
    "source-exports/n2-ground.qasm", "source-exports/n2-thermal.qasm",
    "source-exports/n4-thermal-generation.qasm", "source-exports/n8-thermal-generation.qasm",
    "source-exports/n16-thermal-generation.qasm", "source-exports/n32-thermal-generation.qasm",
    "source-exports/n64-thermal-generation.qasm", "source/LICENSE.txt", "source/cases.json",
    "source/requirements-lock.txt", "source/reusable_bath_local_spin.py",
    "source/reusable_bath_local_spin_reference.py",
})
SOURCE_HASHES = {
    "src/openquantum/simulation/reusable_bath_local_spin.py": "218df15f55c4c9cb2b183e6fbf77fa83fe647aeb973af24a2eafb78e949f891f",
    "src/openquantum/simulation/reusable_bath_local_spin_reference.py": "752071e4e668d6012312b7f03fdad27353009b37333297040c5c182916b107e2",
    "src/openquantum/simulation/reusable_bath_local_spin_cases.json": "761c2ca3eafabf672a221d18abfdfbae91b024f25ee3a41df4b5c95de1d041f3",
    "tests/isolated/reusable_bath_local_spin_checks.py": "ec7c6e300a867a6a89a3a9df9f3e3b027ab8f5ff5efbd204fdd37f3554707ee5",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
NOTEBOOK_CODE_IDS = ("local-spin-load", "local-spin-status", "local-spin-cases",
                     "local-spin-generation", "local-spin-reference")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if (not isinstance(name, str) or not name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name or "\\" in name):
        raise ValueError("Unsafe local-spin saved filename")
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Unsafe local-spin saved root")
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Linked local-spin saved file or directory")
    if not path.is_file():
        raise ValueError("Missing or nonregular local-spin saved file: " + name)
    with path.open("rb") as handle:
        raw = handle.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Local-spin saved file exceeds its byte limit")
    return raw


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in local-spin saved JSON")
        result[key] = value
    return result


def _json(raw):
    def reject(value):
        raise ValueError("Nonfinite local-spin saved JSON")
    return json.loads(raw, object_pairs_hook=_object, parse_constant=reject)


def _inventory(directory, expected, directories):
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Missing or linked local-spin saved tree")
    found = set()
    for child in directory.iterdir():
        if child.is_symlink():
            raise ValueError("Linked local-spin saved member")
        if child.is_file():
            found.add(child.name)
        elif child.name in directories and child.is_dir():
            for member in child.iterdir():
                if member.is_symlink() or not member.is_file():
                    raise ValueError("Unexpected local-spin saved nested member")
                found.add(child.name + "/" + member.name)
                if len(found) > 32:
                    raise ValueError("Local-spin saved tree exceeds its file cap")
        else:
            raise ValueError("Unexpected local-spin saved directory or member")
        if len(found) > 32:
            raise ValueError("Local-spin saved tree exceeds its file cap")
    if found != expected:
        raise ValueError("Local-spin saved tree has unrecorded or missing members")


def _notebook(root, provenance):
    raw = _read(root, NOTEBOOK, 2 * 1024 * 1024)
    if digest(raw) != NOTEBOOK_SHA256:
        hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
        if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
                or not isinstance(hashes, dict) or hashes.get(NOTEBOOK) != digest(raw)):
            raise ValueError("Refreshed local-spin notebook needs current package provenance")
    notebook = _json(raw)
    metadata = notebook.get("metadata") if isinstance(notebook, dict) else None
    if (not isinstance(metadata, dict) or notebook.get("nbformat") != 4
            or not isinstance(metadata.get("kernelspec"), dict)
            or metadata["kernelspec"].get("name") != "python3"
            or metadata["kernelspec"].get("language") != "python"
            or not isinstance(metadata.get("language_info"), dict)
            or metadata["language_info"].get("name") != "python"
            or metadata["language_info"].get("version") != "3.12.14"):
        raise ValueError("Local-spin notebook format or Python metadata changed")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or len(cells) != 12:
        raise ValueError("Local-spin notebook cell structure changed")
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") not in {"markdown", "code", "raw"}:
            raise ValueError("Malformed local-spin notebook cell")
        source = cell.get("source")
        if not isinstance(source, str) and not (isinstance(source, list) and all(isinstance(line, str) for line in source)):
            raise ValueError("Malformed local-spin notebook source")
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs")
            count = cell.get("execution_count")
            if (not isinstance(outputs, list) or any(not isinstance(output, dict) for output in outputs)
                    or (count is not None and type(count) is not int)):
                raise ValueError("Malformed local-spin notebook outputs or execution count")
    projection = [{"id": cell.get("id"), "cell_type": cell.get("cell_type"),
                   "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
                  for cell in cells]
    identity = digest(json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())
    if identity != NOTEBOOK_SOURCE_SHA256:
        raise ValueError("Local-spin notebook sources changed")
    codes = [cell for cell in cells if cell.get("cell_type") == "code"]
    if tuple(cell.get("id") for cell in codes) != NOTEBOOK_CODE_IDS:
        raise ValueError("Local-spin notebook code cells changed")
    counts = [cell.get("execution_count") for cell in codes]
    if counts not in ([None] * 5, list(range(1, 6))):
        raise ValueError("Local-spin notebook execution sequence changed")
    if counts == [None] * 5 and any(cell.get("outputs") for cell in codes):
        raise ValueError("Cleared local-spin notebook has unexecuted outputs")
    if any(output.get("output_type") == "error" for cell in codes for output in cell.get("outputs", [])):
        raise ValueError("Local-spin notebook has an error output")


def verify_saved(root=ROOT, *, bundle_root=None, provenance=None, notebook_roots=None):
    root = Path(root)
    notebook_roots = (root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, (tuple, list)) or not 1 <= len(notebook_roots) <= 2:
        raise ValueError("Local-spin verification requires one or two notebook roots")
    bundle = root / BUNDLE if bundle_root is None else Path(bundle_root)
    _inventory(bundle, BUNDLE_FILES, {"source", "source-exports"})
    manifest_raw = _read(bundle, "artifact-manifest.json", 512 * 1024)
    if digest(manifest_raw) != MANIFEST_SHA256:
        raise ValueError("Local-spin manifest identity changed")
    if _read(bundle, "artifact-manifest.sha256", 65) != (MANIFEST_SHA256 + "\n").encode():
        raise ValueError("Local-spin manifest sidecar changed")
    manifest = _json(manifest_raw)
    files = manifest.get("files")
    if (manifest.get("kind") != "local-spin-bath-saved-evidence" or manifest.get("all_components_verified") is not False
            or manifest.get("total_accuracy_certified") is not False or manifest.get("mixing_certificate") is not False
            or manifest.get("continuum_bridge") != "unavailable" or manifest.get("physical_budgets") != "uncomputed"
            or not isinstance(files, dict) or set(files) != BUNDLE_FILES - {"artifact-manifest.json", "artifact-manifest.sha256"}):
        raise ValueError("Local-spin finite evidence scope changed")
    payload = {"artifact-manifest.json": manifest_raw, "artifact-manifest.sha256": (MANIFEST_SHA256 + "\n").encode()}
    total = sum(map(len, payload.values()))
    for name, record in files.items():
        if (not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int or not 0 <= record["bytes"] <= MAX_FILE_BYTES):
            raise ValueError("Malformed local-spin payload record")
        total += record["bytes"]
        if total > MAX_TOTAL_BYTES:
            raise ValueError("Local-spin saved payload exceeds32MiB")
        raw = _read(bundle, name)
        if len(raw) != record["bytes"] or digest(raw) != record["sha256"]:
            raise ValueError("Local-spin payload identity changed: " + name)
        payload[name] = raw
    for name, wanted in SOURCE_HASHES.items():
        if digest(_read(root, name)) != wanted:
            raise ValueError("Current local-spin source identity changed: " + name)
    archive_path = bundle.parent / "reusable-bath-local-spin-evidence.zip"
    archive_raw = _read(archive_path.parent, archive_path.name, 2 * 1024 * 1024)
    if digest(archive_raw) != ARCHIVE_SHA256:
        raise ValueError("Local-spin evidence ZIP identity changed")
    try:
        with zipfile.ZipFile(io.BytesIO(archive_raw)) as archive:
            prefix = "reusable-bath-local-spin-evidence/"
            entries = archive.infolist()
            if len(entries) != 21 or len({entry.filename for entry in entries}) != 21:
                raise ValueError("Local-spin evidence ZIP member count changed")
            if {entry.filename for entry in entries} != {prefix + name for name in BUNDLE_FILES}:
                raise ValueError("Local-spin evidence ZIP names changed")
            for entry in entries:
                name = entry.filename[len(prefix):]
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (entry.is_dir() or entry.flag_bits & 1 or mode not in (0, stat.S_IFREG)
                        or entry.file_size != len(payload[name]) or entry.file_size > MAX_FILE_BYTES
                        or archive.read(entry) != payload[name]):
                    raise ValueError("Local-spin evidence ZIP member changed")
    except zipfile.BadZipFile as error:
        raise ValueError("Invalid local-spin evidence ZIP") from error
    presentation = bundle.parent / "reusable-bath-local-spin-presentation"
    _inventory(presentation, set(PRESENTATION_HASHES), set())
    presentation_data = {}
    for name, wanted in PRESENTATION_HASHES.items():
        raw = _read(presentation, name, 1024 * 1024)
        if digest(raw) != wanted:
            raise ValueError("Local-spin presentation identity changed")
        presentation_data[name] = raw
    record = _json(presentation_data["presentation.json"])
    if (record.get("kind") != "presentation_excerpt_from_saved_qasm"
            or record.get("source_sha256") != files["source-exports/n2-ground.qasm"]["sha256"]
            or record.get("initial_resets") != 3 or record.get("unitary_instructions") != 19
            or record.get("new_source_generation") is not False
            or record.get("new_quantum_or_reference_execution") is not False
            or record.get("figure_sha256") != {"first-step.svg": PRESENTATION_HASHES["first-step.svg"]}):
        raise ValueError("Local-spin presentation scope changed")
    for notebook_root in notebook_roots:
        if (Path(notebook_root) / NOTEBOOK).is_file():
            _notebook(Path(notebook_root), provenance)
        else:
            raise ValueError("Missing local-spin saved-evidence notebook")
    return {"status": "saved_component_verified", "files": 21, "bytes": total,
            "saved_source_exports": 9, "saved_source_instructions": 66680,
            "all_components_verified": False, "new_quantum_execution": False,
            "new_source_generation": False, "new_native_queries": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    provenance_path = args.root / "PROVENANCE.json"
    provenance = _json(_read(args.root, "PROVENANCE.json", 4 * 1024 * 1024)) if provenance_path.is_file() else None
    print(json.dumps(verify_saved(args.root, provenance=provenance), indent=2))


if __name__ == "__main__":
    main()
