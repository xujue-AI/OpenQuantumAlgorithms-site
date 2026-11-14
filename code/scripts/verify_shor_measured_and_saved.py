#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate recorded measured-AND evidence without executing its circuits."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import stat
import struct
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/shor/measured-and"
PARENT = "assets/generated/shor"
NOTEBOOK = "notebooks/shor_measured_and.ipynb"
CANONICAL_NOTEBOOK = PARENT + "/measured-and-executed.ipynb"
CANONICAL_NOTEBOOK_SHA256 = "34c0c9b356c52bc14687d22952ff4c4220e3de27466b28fae6b68bee011d430f"
NOTEBOOK_SOURCE_SHA256 = "3c2fabb4c6fce6ec927bb9cbaddc43865c43b3e7a6c24bccd34aa48d60eb1097"
MANIFEST_SHA256 = "ed844b56c160ee61fa2645fb5121f9e5e9e38abaf521081694c8c58ef689a35c"
ARCHIVE = PARENT + "/measured-and-artifacts.zip"
SUMMARY = PARENT + "/measured-and-validation.json"
PNG = PARENT + "/measured-and-notebook-circuit.png"
SIDE_HASHES = {
    ARCHIVE: "5a59abd65c47f26d62539f51e44c516e0c74b63998ec4f1eb5688832b902e3c9",
    SUMMARY: "307fc2969a1487850ecb6df8ccbb1159e39fd622ae97ddf351a28ec1fcd2b5e9",
    PNG: "801a0ec12d2e128652fecfca5b83b676f4d8bfcd5bbf0df77de27d6a4c221678",
}
SOURCE_HASHES = {
    "scripts/generate_shor_measured_and.py": "391fb9e1ea2256f1e29906cc88b02f3c3f492cab5096610c3f1396ff23dd4a06",
    "src/openquantum/shor_measured_and.py": "9149bee02a0cdef1d3dde60e4aea21f1f909b257a0e2885266172b00ad9e05fd",
    "src/openquantum/shor.py": "e4b80e2e9ef62e1c62ac80a4bb0f9bbde694cb659579c58b7563656aaadb4ce3",
    "src/openquantum/shor_ripple.py": "fb3acf345459446d20109d698efefdabd8f906d5ebaf48c8febd1a7a3bfcb282",
    "src/openquantum/__init__.py": "af9ea652a9f24962c7808f9155735420b5444f530a0431951b87922ffd9f9663",
    "src/openquantum/grover.py": "cd637cd2373c79bc2ad97566f83d70c2953034ff7ded2a37805818e7d942bf21",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
    "tests/test_shor_measured_and.py": "daf1576ffc92123ecae08e09dbe255de98201480c885425b896c1a06dc0bbbda",
}
BUNDLE_FILES = frozenset({"manifest.json", "study.json", "scaling.csv", "README.md", "LICENSE.txt",
    "requirements-lock.txt", "__init__.py", "grover.py", "shor.py", "shor_ripple.py", "shor_measured_and.py",
    "generate_shor_measured_and.py", "gadget.svg", "addition.svg", "gadget-static.qasm",
    "gadget-feedback_x.qasm", "gadget-reset.qasm"}) | frozenset(
    f"addition-w{width}-{method}.qasm" for width in (1, 2, 3, 24) for method in ("static", "measured")) | frozenset(
    f"order-N{modulus}-t{bits}-{method}.qasm" for modulus, bits in ((15, 4), (21, 4), (4093, 1)) for method in ("static", "measured"))
CODE_IDS = ("measured-and-imports", "measured-and-counts", "measured-and-circuit", "measured-and-check",
            "measured-and-gadget", "measured-and-record", "measured-and-scaling")
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024


def _root(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path != path.resolve():
        raise ValueError("Unsafe measured-AND root")
    return path


def _read(root, name, limit=MAX_FILE_BYTES):
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("Unsafe measured-AND member path")
    path = root / relative
    if path.is_symlink() or path.resolve() != path or not path.is_file():
        raise ValueError("Missing or linked measured-AND member: " + name)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
        raise ValueError("Unsafe or oversized measured-AND member: " + name)
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    if len(raw) != before.st_size or len(raw) > limit or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("Measured-AND member changed during read: " + name)
    return raw


def _png(raw):
    if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
        raise ValueError("Measured-AND notebook figure is not a PNG")
    width, height = struct.unpack(">II", raw[16:24])
    if not 0 < width <= 10000 or not 0 < height <= 10000:
        raise ValueError("Measured-AND notebook PNG dimensions are invalid")
    position, idat, ended = 8, False, False
    while position < len(raw):
        if position + 12 > len(raw):
            raise ValueError("Measured-AND notebook PNG is truncated")
        size = struct.unpack(">I", raw[position:position+4])[0]
        end = position + 12 + size
        if end > len(raw):
            raise ValueError("Measured-AND notebook PNG chunk is truncated")
        kind = raw[position+4:position+8]
        body = raw[position+8:position+8+size]
        checksum = struct.unpack(">I", raw[position+8+size:end])[0]
        if zlib.crc32(kind + body) & 0xffffffff != checksum:
            raise ValueError("Measured-AND notebook PNG checksum changed")
        if position == 8 and (kind != b"IHDR" or size != 13):
            raise ValueError("Measured-AND notebook PNG header changed")
        idat |= kind == b"IDAT"
        if kind == b"IEND":
            if size != 0 or end != len(raw):
                raise ValueError("Measured-AND notebook PNG ending changed")
            ended = True
        position = end
    if not idat or not ended:
        raise ValueError("Measured-AND notebook PNG data or ending is missing")
    return width, height


def notebook_source_sha256(raw):
    notebook = json.loads(raw)
    cells = notebook.get("cells")
    if notebook.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 12:
        raise ValueError("Measured-AND notebook cell contract changed")
    projection, code = [], []
    for cell in cells:
        if not isinstance(cell, dict):
            raise ValueError("Invalid measured-AND notebook cell")
        source = cell.get("source")
        if isinstance(source, list) and all(isinstance(line, str) for line in source):
            source = "".join(source)
        if not isinstance(source, str) or not isinstance(cell.get("id"), str) or cell.get("cell_type") not in ("code", "markdown", "raw"):
            raise ValueError("Invalid measured-AND notebook source")
        projection.append((cell["id"], cell["cell_type"], source))
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs")
            if not isinstance(outputs, list) or any(not isinstance(output, dict) or output.get("output_type") == "error" for output in outputs):
                raise ValueError("Measured-AND notebook has invalid or error outputs")
            code.append(cell)
    if tuple(cell["id"] for cell in code) != CODE_IDS or any(type(cell.get("execution_count")) is not int for cell in code) or [cell["execution_count"] for cell in code] != list(range(1, 8)):
        raise ValueError("Measured-AND notebook requires seven executed code cells")
    pngs = [output["data"]["image/png"] for output in code[2]["outputs"]
            if isinstance(output.get("data"), dict) and isinstance(output["data"].get("image/png"), str)]
    if len(pngs) != 1:
        raise ValueError("Measured-AND notebook needs its genuine circuit output")
    _png(base64.b64decode(pngs[0], validate=True))
    if any(not any(output.get("output_type") == "stream" and output.get("text") for output in code[index]["outputs"]) for index in (1, 3, 4, 5, 6)):
        raise ValueError("Measured-AND notebook diagnostic outputs are incomplete")
    # The recorded receipt used ordered tuples and escaped non-ASCII text.
    return hashlib.sha256(json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def verify_saved(root=ROOT, *, source_root=None, notebook_roots=None, provenance=None):
    root = _root(root)
    source_root = _root(root if source_root is None else source_root)
    notebook_roots = (source_root,) if notebook_roots is None else notebook_roots
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two measured-AND notebook roots are required")
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if provenance is not None and (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict)):
        raise ValueError("Complete measured-AND package provenance is required")
    total = 0
    for name, pin in SOURCE_HASHES.items():
        raw = _read(source_root, name)
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != pin or hashes is not None and hashes.get(name) != pin:
            raise ValueError("Promoted measured-AND source identity changed: " + name)
    for notebook_root in notebook_roots:
        raw = _read(_root(notebook_root), NOTEBOOK)
        total += len(raw)
        if hashes is not None and hashlib.sha256(raw).hexdigest() != hashes.get(NOTEBOOK):
            raise ValueError("Current measured-AND notebook differs from package provenance")
        if notebook_source_sha256(raw) != NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Measured-AND notebook source changed")
    parent = _root(root / PARENT)
    expected_siblings = {"measured-and", *(PurePosixPath(name).name for name in SIDE_HASHES), PurePosixPath(CANONICAL_NOTEBOOK).name}
    for index, path in enumerate(parent.iterdir()):
        if index >= 4096 or path.name.startswith("measured-and") and path.name not in expected_siblings:
            raise ValueError("Measured-AND publication has unrecorded siblings")
    directory = _root(root / BUNDLE)
    names = set()
    for path in directory.iterdir():
        if len(names) >= len(BUNDLE_FILES) or path.name not in BUNDLE_FILES or path.is_symlink() or not path.is_file():
            raise ValueError("Measured-AND bundle has unrecorded members")
        names.add(path.name)
    if names != BUNDLE_FILES:
        raise ValueError("Measured-AND bundle membership changed")
    manifest_raw = _read(root, BUNDLE + "/manifest.json", 64 * 1024)
    if hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError("Frozen measured-AND manifest changed")
    manifest = json.loads(manifest_raw)
    if manifest.get("schema") != "openquantum-shor-measured-and-v1" or manifest.get("suite") is not True or manifest.get("simulate") is not True or manifest.get("physical_or_formal_claim") is not False or set(manifest.get("files", {})) != BUNDLE_FILES - {"manifest.json"}:
        raise ValueError("Frozen measured-AND manifest scope changed")
    contents = {}
    for name in sorted(BUNDLE_FILES):
        raw = _read(root, BUNDLE + "/" + name)
        total += len(raw)
        if name != "manifest.json" and {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != manifest["files"][name]:
            raise ValueError("Frozen measured-AND payload changed: " + name)
        contents[name] = raw
    for name, record in manifest["source_files"].items():
        snapshot = "LICENSE.txt" if name == "LICENSE" else PurePosixPath(name).name
        if record["sha256"] != SOURCE_HASHES[name] or hashlib.sha256(contents[snapshot]).hexdigest() != record["sha256"]:
            raise ValueError("Measured-AND source snapshot changed")
    sidecars = {}
    for name, pin in SIDE_HASHES.items():
        raw = _read(root, name)
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != pin:
            raise ValueError("Frozen measured-AND sidecar changed: " + name)
        sidecars[name] = raw
    canonical = _read(root, CANONICAL_NOTEBOOK)
    total += len(canonical)
    if total > MAX_TOTAL_BYTES or hashlib.sha256(canonical).hexdigest() != CANONICAL_NOTEBOOK_SHA256 or hashes is not None and hashes.get(CANONICAL_NOTEBOOK) != CANONICAL_NOTEBOOK_SHA256:
        raise ValueError("Original measured-AND notebook or input byte budget changed")
    if notebook_source_sha256(canonical) != NOTEBOOK_SOURCE_SHA256:
        raise ValueError("Original measured-AND notebook source changed")
    code = [cell for cell in json.loads(canonical)["cells"] if cell["cell_type"] == "code"]
    recorded_png = base64.b64decode(next(output["data"]["image/png"] for output in code[2]["outputs"] if isinstance(output.get("data"), dict) and "image/png" in output["data"]), validate=True)
    if recorded_png != sidecars[PNG] or _png(recorded_png) != (1634, 1874):
        raise ValueError("Original measured-AND diagram output changed")
    with zipfile.ZipFile(io.BytesIO(sidecars[ARCHIVE])) as archive:
        entries = archive.infolist()
        if len(entries) != len(BUNDLE_FILES) or {entry.filename for entry in entries} != BUNDLE_FILES:
            raise ValueError("Measured-AND ZIP membership changed")
        for entry in entries:
            if entry.is_dir() or entry.create_system != 3 or entry.external_attr >> 16 != 0o100644 or entry.date_time != (1980, 1, 1, 0, 0, 0) or entry.compress_type != zipfile.ZIP_DEFLATED or entry.file_size != len(contents[entry.filename]):
                raise ValueError("Measured-AND ZIP entry metadata changed")
            with archive.open(entry) as stream:
                raw = stream.read(len(contents[entry.filename]) + 1)
            if raw != contents[entry.filename]:
                raise ValueError("Measured-AND ZIP differs from the recorded flat files")
    summary = json.loads(sidecars[SUMMARY])
    study = json.loads(contents["study.json"])
    if summary.get("hardware_execution") is not False or summary.get("physical_estimate") is not False or summary.get("mechanized_formal_proof") is not False or summary["dedicated_module_tests"]["passed"] != 109 or summary["notebook"]["source_projection_sha256"] != NOTEBOOK_SOURCE_SHA256:
        raise ValueError("Measured-AND saved validation scope changed")
    if [(row["width"], row["method"]) for row in study["addition_cases"]] != [(width, method) for width in (1, 2, 3) for method in ("static", "measured")]:
        raise ValueError("Measured-AND saved addition cases changed")
    if any(row["validation"]["exact_sparse"]["norm_exactly_one"] is not True or not math.isfinite(row["validation"]["max_density_error"]) or row["validation"]["max_density_error"] > 1e-10 for row in study["addition_cases"]):
        raise ValueError("Measured-AND saved small checks changed")
    if [(row["modulus"], row["base"], row["phase_bits"]) for row in study["order_cases"]] != [(15, 2, 4), (21, 2, 4)] or any(row["validation"]["exact_sparse"]["norm_exactly_one"] is not True for row in study["order_cases"]):
        raise ValueError("Measured-AND saved order cases changed")
    if len(study["generated_only"]) != 4 or any(row["validated_quantum_simulation"] is not False for row in study["generated_only"]) or len(list(csv.DictReader(io.StringIO(contents["scaling.csv"].decode())))) != 16:
        raise ValueError("Measured-AND larger-generation or scaling scope changed")
    return {"status": "saved_measured_and_verified", "files": 31, "saved_module_tests": 109,
            "saved_addition_cases": 6, "saved_order_cases": 2, "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
            "new_quantum_execution": False, "native_calls": 0, "physical_estimate": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify_saved(parser.parse_args().root), indent=2))


if __name__ == "__main__":
    main()
