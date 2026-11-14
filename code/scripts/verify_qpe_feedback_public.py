#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate frozen feedback-QPE evidence and refreshed notebook outputs.

The original saved checker runs against a bounded temporary snapshot containing
its original executed notebook. Current notebooks are authenticated separately
through their ordered sources, outputs and complete public-package provenance.
No synthesis, estimator, quantum module or notebook execution is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/qpe-feedback-resources"
VALIDATION = "assets/generated/qpe-feedback-validation"
CANONICAL_NOTEBOOK = VALIDATION + "/qpe_feedback_resources.ipynb"
NOTEBOOK = "notebooks/qpe_feedback_resources.ipynb"
CANONICAL_NOTEBOOK_SHA256 = "eda3672fd30e1dab6a5cd02bde3d36cda5e3c40c8e2a704dd398e1b95116b50c"
NOTEBOOK_SOURCE_SHA256 = "d8b0e6e2399ccc23b900093e408c6e56d8ee5072f685b99ef6ebda1025cf094f"
MANIFEST_SHA256 = "80bec288f575f83695041dcf866fba494235cac83c1229bb4efdd8cabf07814d"
SOURCE_HASHES = {
    "src/openquantum/qpe_feedback_resources.py": "c69172f89397a5c9aec1f8da8ebf87c304457a5b29a7e6610418c76b154fd10f",
    "src/openquantum/qft_qpe.py": "6a991e019ef166c8b023fa4e4701749e912f921c96e845bee085040ea57a40ab",
    "src/openquantum/instrument_resources.py": "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863",
    "scripts/generate_qpe_feedback_resources.py": "fb3dababdda0dfc2e43ad02da49a2b5d57dbf932b1551f64a064bbd81f265391",
    "scripts/verify_qpe_feedback_saved.py": "b32a7b431e76ba8850d3b1d9cd90cab3c604547e233eae8d291047e449209906",
    "tests/test_qpe_feedback_resources.py": "0c52f284fc748be4edca681b124c1aae97641e3cbdf32ae8938af31da6dad3a7",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
}
ROOT_FILES = frozenset({"manifest.json", "summary.csv", "summary.md", "logical-size-scaling.json",
    "README.md", "LICENSE.txt", "references.bib", "primitives.bib", "qpe_feedback_resources.py",
    "qft_qpe.py", "instrument_resources.py", "generate_qpe_feedback_resources.py",
    "verify_qpe_feedback_saved.py", "test_qpe_feedback_resources.py", "requirements-lock.txt"})
TASK_FILES = frozenset({"report.json", "textbook-source.qasm", "feedback-source.qasm",
    "textbook-clifford-t.qasm", "feedback-clifford-t.qasm", "feedback-all-corrections-accounting-envelope.qasm"})
BUNDLE_FILES = ROOT_FILES | frozenset(f"n1-m{m}/{name}" for m in (3, 4) for name in TASK_FILES)
NOTEBOOK_LABELS = {"a6950bb5": "qpe-feedback-physical-circuit",
    "a2dc902b": "qpe-feedback-physical-probabilities", "568ad2f9": "qpe-feedback-physical-table"}
MAX_FILE_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 4 * 1024 * 1024


def _root(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path != path.resolve():
        raise ValueError("Unsafe feedback-QPE root")
    return path


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("Unsafe feedback-QPE member path")
    path = root / relative
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Missing or linked feedback-QPE member: " + name)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise ValueError("Unsafe or oversized feedback-QPE member: " + name)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if len(raw) > limit or len(raw) != before.st_size or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Feedback-QPE member changed during read: " + name)
    return raw


def _tree(directory, files, directories=frozenset()):
    directory = _root(directory)
    observed = set()
    for path in directory.iterdir():
        name = path.name
        if len(observed) >= len(files) + len(directories) or name not in files | directories or path.is_symlink():
            raise ValueError("Feedback-QPE tree contains unrecorded members")
        if name in directories:
            _root(path)
        elif not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("Feedback-QPE tree contains a special member")
        observed.add(name)
    if observed != files | directories:
        raise ValueError("Feedback-QPE tree membership changed")


def notebook_source_sha256(raw):
    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if notebook.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 9:
        raise ValueError("Feedback-QPE notebook cell contract changed")
    projection, code = [], []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Invalid feedback-QPE notebook cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str) or not isinstance(cell.get("id"), str) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError("Invalid feedback-QPE notebook source")
        projection.append({"id": cell["id"], "cell_type": cell["cell_type"], "source": source})
        metadata = cell.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Invalid feedback-QPE notebook metadata")
        if cell["id"] in NOTEBOOK_LABELS and metadata.get("label") != NOTEBOOK_LABELS[cell["id"]]:
            raise ValueError("Feedback-QPE notebook label changed")
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs")
            if not isinstance(outputs, list) or any(not isinstance(output, dict) for output in outputs) or any(output.get("output_type") == "error" for output in outputs):
                raise ValueError("Feedback-QPE notebook has invalid or error outputs")
            code.append(cell)
    if [cell["id"] for cell in code] != ["4231e808", "a6950bb5", "a2dc902b", "568ad2f9", "5d9473de"] or any(type(cell.get("execution_count")) is not int for cell in code) or [cell["execution_count"] for cell in code] != [1, 2, 3, 4, 5]:
        raise ValueError("Feedback-QPE notebook needs five executed code cells")
    for index, mime in ((1, "image/png"), (2, "image/png"), (3, "text/html"), (4, "text/html")):
        if not any(isinstance(output.get("data"), dict) and output["data"].get(mime) for output in code[index]["outputs"]):
            raise ValueError("Missing feedback-QPE circuit, probability or table output")
    for index in (2, 4):
        if not any(output.get("output_type") == "stream" and output.get("text") for output in code[index]["outputs"]):
            raise ValueError("Missing feedback-QPE notebook diagnostic output")
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_saved(root=ROOT, *, source_root=None, notebook_roots=None, provenance=None):
    root = _root(root)
    source_root = _root(root if source_root is None else source_root)
    notebook_roots = (source_root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two authenticated feedback-QPE notebook roots are required")
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if provenance is not None and (provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict)):
        raise ValueError("Complete feedback-QPE package provenance is required")
    sources = {}
    for name, pin in SOURCE_HASHES.items():
        raw = _read(source_root, name)
        if hashlib.sha256(raw).hexdigest() != pin or hashes is not None and hashes.get(name) != pin:
            raise ValueError("Promoted feedback-QPE source identity changed: " + name)
        sources[name] = raw
    for notebook_root in notebook_roots:
        raw = _read(_root(notebook_root), NOTEBOOK)
        if hashes is not None and hashlib.sha256(raw).hexdigest() != hashes.get(NOTEBOOK):
            raise ValueError("Refreshed feedback-QPE notebook differs from current package provenance")
        if notebook_source_sha256(raw) != NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Feedback-QPE notebook source changed")
    _tree(root / VALIDATION, frozenset({"qpe_feedback_resources.ipynb"}))
    canonical = _read(root, CANONICAL_NOTEBOOK)
    if hashlib.sha256(canonical).hexdigest() != CANONICAL_NOTEBOOK_SHA256 or hashes is not None and hashes.get(CANONICAL_NOTEBOOK) != CANONICAL_NOTEBOOK_SHA256:
        raise ValueError("Original executed feedback-QPE notebook changed")
    _tree(root / BUNDLE, ROOT_FILES, frozenset({"n1-m3", "n1-m4"}))
    for m in (3, 4):
        _tree(root / BUNDLE / f"n1-m{m}", TASK_FILES)
    manifest_raw = _read(root, BUNDLE + "/manifest.json", 64 * 1024)
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen feedback-QPE manifest changed")
    manifest = json.loads(manifest_raw)
    if set(manifest.get("artifact_sha256", {})) != BUNDLE_FILES - {"manifest.json"} or manifest.get("source_sha256") != SOURCE_HASHES:
        raise ValueError("Frozen feedback-QPE manifest membership changed")
    contents, total = {}, len(canonical) + sum(map(len, sources.values()))
    for name in sorted(BUNDLE_FILES):
        raw = _read(root, BUNDLE + "/" + name)
        total += len(raw)
        if total > MAX_SNAPSHOT_BYTES or name != "manifest.json" and hashlib.sha256(raw).hexdigest() != manifest["artifact_sha256"][name]:
            raise ValueError("Frozen feedback-QPE payload identity or size changed: " + name)
        contents[name] = raw
    helper_name = "scripts/verify_qpe_feedback_saved.py"
    namespace = {"__file__": str(source_root / helper_name), "__name__": "_verified_feedback_qpe_snapshot"}
    exec(compile(sources[helper_name], str(source_root / helper_name), "exec"), namespace)
    with tempfile.TemporaryDirectory(prefix="openquantum-feedback-snapshot-") as temporary:
        snapshot = Path(temporary).resolve()
        for name, raw in {**sources, NOTEBOOK: canonical, **{BUNDLE + "/" + name: raw for name, raw in contents.items()}}.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        result = namespace["verify_saved"](snapshot)
    if result.get("native_calls") != 0:
        raise ValueError("Feedback-QPE saved checker changed its execution scope")
    return {"status": "saved_feedback_qpe_verified", "files": len(BUNDLE_FILES),
        "original_notebook_sha256": CANONICAL_NOTEBOOK_SHA256,
        "current_notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
        "authenticated_current_notebook_roots": len(notebook_roots),
        "native_calls": 0, "new_quantum_execution": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify_saved(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
