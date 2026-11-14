#!/usr/bin/env python3
"""Check public-package hashes, isolated imports and the full test suite.

The formal-record check verifies source identity and the saved declaration
audit. It does not execute Lean or extend the recorded classical coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FORMAL_SOURCES = ("C4Reference.lean", "lean-toolchain", "check.py", "LICENSE.txt", "README.md")
FORMAL_THEOREMS = (
    "all_cuts_le_four", "optimal_iff", "five_is_optimal", "ten_is_optimal", "marked_iff",
)
FORMAL_TOOLCHAIN = "leanprover/lean4:v4.34.0-rc2"
FORMAL_COMMIT = "6a10ac8c22beadecabdbb0919c2b50214762f91d"
FORMAL_SCOPE = "Encoded finite classical C4 MaxCut reference and known marked vertex only"
FORMAL_EXCLUSIONS = (
    "Equivalence to the Python implementation",
    "Quantum circuit semantics and uncomputation",
    "Adiabatic theorem, spectral gap, or runtime guarantee",
    "Quantum-walk success probability",
    "Sampling confidence and fault-tolerant resource models",
)


def checked_directory(root: Path) -> Path:
    """Reject lexical directory aliases without loading project code."""
    root = Path(root).absolute()
    if not root.is_dir() or root.is_symlink() or root != root.resolve():
        raise ValueError("Unsafe or linked public-package root")
    return root


def checked_path(root: Path, name: str) -> Path:
    """Resolve one manifest-relative regular file inside the given root."""
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute()
            or ".." in relative.parts or relative.as_posix() != name):
        raise ValueError(f"Unsafe public-package path: {name}")
    path = root / relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Missing or unsafe public-package file: {name}")
    return path


def verify_formal_record(root: Path) -> None:
    """Check that both published copies match the saved classical proof record."""
    source = root / "algorithms/optimization/formal"
    artifact = root / "assets/generated/optimization/formal"
    record = json.loads(checked_path(artifact, "verification.json").read_text())
    if (record.get("schema") != "optimization-finite-classical-formal-v1"
            or record.get("status") != "checked" or record.get("exit_code") != 0
            or record.get("scope") != FORMAL_SCOPE
            or record.get("exclusions") != list(FORMAL_EXCLUSIONS)
            or record.get("toolchain") != FORMAL_TOOLCHAIN
            or FORMAL_COMMIT not in record.get("lean_version_output", "")
            or record.get("kernel_checked_closed_propositions") is not True
            or record.get("native_decide_used") is not False
            or record.get("axioms_by_theorem") != {name: [] for name in FORMAL_THEOREMS}):
        raise ValueError("Unexpected recorded finite-classical Lean check")
    hashes = record.get("source_sha256", {})
    if set(hashes) != set(FORMAL_SOURCES):
        raise ValueError("Incomplete finite-classical Lean source manifest")
    for name in FORMAL_SOURCES:
        for directory in (source, artifact):
            actual = hashlib.sha256(checked_path(directory, name).read_bytes()).hexdigest()
            if actual != hashes[name]:
                raise ValueError(f"Recorded Lean source digest mismatch: {directory.relative_to(root)}/{name}")
    if checked_path(source, "lean-toolchain").read_text().strip() != FORMAL_TOOLCHAIN:
        raise ValueError("Recorded Lean source toolchain differs from its check")


def verify_manifest(package: Path) -> dict:
    provenance = json.loads(checked_path(package, "PROVENANCE.json").read_text())
    manifest = provenance.get("file_sha256")
    if provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(manifest, dict):
        raise ValueError("Missing complete public-package file manifest")
    actual_names = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*") if path.is_file()
    } - {"PROVENANCE.json"}
    if set(manifest) != actual_names:
        raise ValueError("Public-package file manifest does not match its contents")
    for name, digest in manifest.items():
        actual = hashlib.sha256(checked_path(package, name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Public-package file digest mismatch: {name}")
    sources = {name: digest for name, digest in manifest.items() if name.endswith(".py")}
    if provenance.get("source_sha256") != sources:
        raise ValueError("Public-package Python source manifest differs from the complete manifest")
    return provenance


def _load_energy_record_gate(package: Path, provenance: dict):
    """Compile one authenticated helper after the complete manifest check.

    An ordinary sibling import could reuse an unbound module or bytecache.
    The gate is instead loaded from its exact, singly read package bytes and
    is not inserted into the process-wide import registry.
    """
    package = checked_directory(package)
    name = "scripts/energy_record_release.py"
    source = package
    for part in PurePosixPath(name).parts:
        source /= part
        if source.is_symlink():
            raise ValueError(f"Linked public-package energy helper: {name}")
    source = checked_path(package, name)
    source_bytes = source.read_bytes()
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package energy helper differs from the complete manifest")
    spec = importlib.util.spec_from_file_location("_openquantum_verified_energy_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package energy helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def _read_bounded_public_member(package: Path, name: str, maximum: int) -> bytes:
    """Read one regular member once, without following lexical file links."""
    package = checked_directory(package)
    source = package
    for part in PurePosixPath(name).parts:
        source /= part
        if source.is_symlink():
            raise ValueError(f"Linked public-package member: {name}")
    source = checked_path(package, name)
    metadata = source.stat()
    if metadata.st_nlink != 1 or metadata.st_size > maximum:
        raise ValueError("Public-package member is not a bounded regular file")
    descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if (not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)):
            raise ValueError("Public-package member changed before read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            source_bytes = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        if len(source_bytes) != metadata.st_size or len(source_bytes) > maximum or after.st_size != metadata.st_size:
            raise ValueError("Public-package member changed during read")
    finally:
        os.close(descriptor)
    return source_bytes


def _load_word_record_gate(package: Path, provenance: dict):
    """Compile the exact saved-only helper after full package authentication."""
    package = checked_directory(package)
    name = "scripts/optimization_word_record_release.py"
    source_bytes = _read_bounded_public_member(package, name, 2 * 1024 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package word helper differs from the complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_word_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package word helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def _load_suzuki_record_gate(package: Path, provenance: dict):
    """Compile the saved Suzuki verifier after full package authentication."""
    package = checked_directory(package)
    name = "scripts/verify_simulation_suzuki.py"
    source_bytes = _read_bounded_public_member(package, name, 2 * 1024 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package Suzuki helper differs from the complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_suzuki_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package Suzuki helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def _load_gibbs_channel_record_gate(package: Path, provenance: dict):
    """Compile the immutable saved Gibbs helper after full authentication."""
    package = checked_directory(package)
    name = "scripts/verify_detailed_balance_channel.py"
    source_bytes = _read_bounded_public_member(package, name, 2 * 1024 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package Gibbs helper differs from the complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_gibbs_channel_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package Gibbs helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def _load_tail_record_gate(package: Path, provenance: dict):
    """Compile the saved tail core from one authenticated source snapshot."""
    package = checked_directory(package)
    name = "scripts/tail_resource_records.py"
    source_bytes = _read_bounded_public_member(package, name, 1024 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package tail helper differs from the complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_tail_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package tail helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


DQI_SOURCE_HASHES = {
    "src/openquantum/optimization_dqi.py": "370088484830c23140f14181a3892194fa936e4477327536946423854fcc1f99",
    "scripts/generate_optimization_dqi.py": "5312969299af870925d766263eda13fc27f156b497b6fa40cf744bc3ff620cca",
    "scripts/verify_dqi_saved.py": "981d550cf0aac5ec8f1ceb6d8c8f8625fe00a5cdcba9646ed9d2cbb011c4242e",
}


def _load_dqi_saved_gate(package: Path, provenance: dict):
    """Load the portable DQI checker from complete-provenance-bound bytes."""
    package = checked_directory(package)
    name = "scripts/verify_dqi_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package DQI helper differs from the complete manifest")
    if any(manifest.get(path) != value for path, value in DQI_SOURCE_HASHES.items()):
        raise ValueError("Public DQI source differs from the promoted validated/adapted inputs")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_dqi_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package DQI helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def _verify_dqi_saved_identity(gate, bundle: Path, provenance: dict, *, notebook_roots: tuple):
    """Keep public source identities and the fixed saved DQI event distinct."""
    if not isinstance(notebook_roots, tuple) or len(notebook_roots) not in (1, 2):
        raise ValueError("One or two authenticated normal-notebook roots are required")
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if not isinstance(hashes, dict) or any(hashes.get(name) != value for name, value in DQI_SOURCE_HASHES.items()):
        raise ValueError("Public DQI source differs from the promoted validated/adapted inputs")
    bundle = checked_directory(bundle)
    names = {"LICENSE.txt", "bundle.json", "dqi-circuit.svg", "dqi-validated.json", "dqi.qasm",
             "generate_optimization_dqi.py", "optimization_dqi.ipynb", "optimization_dqi.py",
             "requirements-lock.txt", "verify_dqi_saved.py"}
    members = list(bundle.iterdir())
    if (len(members) != 10 or {path.name for path in members} != names
            or any(path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 for path in members)):
        raise ValueError("Public DQI bundle must contain exactly ten flat regular files")
    manifest_raw = _read_bounded_public_member(bundle, "bundle.json", 64 * 1024)
    if hashlib.sha256(manifest_raw).hexdigest() != "ea14085d24d531df648f4bd0957288f753d23121d981d1b5aa3c97a926687e12":
        raise ValueError("Public DQI bundle manifest identity changed")
    notebook = "notebooks/optimization_dqi.ipynb"
    for notebook_root in notebook_roots:
        raw = _read_bounded_public_member(notebook_root, notebook, 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != hashes.get(notebook):
            raise ValueError("Normal DQI notebook differs from the complete public provenance")
        if gate.notebook_source_sha256(raw) != "bdd3db3d07a8f4218a4d32f53b1371196b52466de7d69ec856c36f4850052667":
            raise ValueError("Normal DQI notebook source changed")
    result = gate.verify_saved_bundle(bundle)
    verification = result.get("verification") if isinstance(result, dict) else None
    if (not isinstance(verification, dict)
            or verification.get("status") != "saved_component_verified"
            or verification.get("bundle_manifest_sha256") != "ea14085d24d531df648f4bd0957288f753d23121d981d1b5aa3c97a926687e12"
            or verification.get("notebook_source_sha256") != "bdd3db3d07a8f4218a4d32f53b1371196b52466de7d69ec856c36f4850052667"
            or verification.get("checked_files") != 9
            or verification.get("new_quantum_execution") is not False):
        raise ValueError("Public DQI saved event or notebook source identity changed")
    return result


DQI_STRUCTURED_BUNDLE_DIRECTORY = "assets/generated/optimization/dqi-structured"
DQI_STRUCTURED_MANIFEST_SHA256 = "1e01422fc8a0961bc1771ceedaffacfdf90a9c1a3bc9a0b99efc2aa3f4100850"
DQI_STRUCTURED_NOTEBOOK_SHA256 = "601e1f373c70d9aa2d69a7cfd03b3487d16a4a2681ba26a9921bef9efd6ce4e0"
DQI_STRUCTURED_NOTEBOOK_SOURCE_SHA256 = "a9901953d3fda742ee07cd2315bb654686c4a941c8153f0a127da3493ac114c8"
DQI_STRUCTURED_SOURCE_HASHES = {
    "src/openquantum/optimization_dqi_structured.py": "0daaf35c01fbf83bcdfad98c85f6af94420202d7b546c7892615bd18ff1ec028",
    "src/openquantum/optimization_dqi_structured_validation.py": "5994818be957a7c89c9ace46354676acc18f61ae6fc327536d8aad9e84096256",
    "scripts/generate_optimization_dqi_structured.py": "0d5a4ce113f63447a8e880891dd50a1079e942fd918b99f1dccc3cd0262ab36c",
    "scripts/validate_optimization_dqi_structured.py": "089ee9b3aaccd8ad61904ceb8d35803e0af2cb3d176a08e272129e13f14332cc",
    "scripts/verify_dqi_structured_saved.py": "8677e965b63833c39db0593cc516e2ee429008f79299e94ebc237f4d15c3f639",
    "tests/test_optimization_dqi_structured.py": "230ad818b14917f61d6f1d86e5fc66a4293719cba1ff412f2162537b8ee56d8d",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}


def _load_dqi_structured_saved_gate(package: Path, provenance: dict):
    """Authenticate all promoted source bytes before compiling the pure reader."""
    package = checked_directory(package)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict)):
        raise ValueError("Structured DQI requires complete public provenance")
    source_buffers = {}
    for name, expected in DQI_STRUCTURED_SOURCE_HASHES.items():
        raw = _read_bounded_public_member(package, name, 64 * 1024)
        if hashes.get(name) != expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Public structured DQI promoted source identity changed: " + name)
        source_buffers[name] = raw
    name = "scripts/verify_dqi_structured_saved.py"
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_dqi_structured_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated structured DQI saved reader")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_buffers[name], str(source), "exec"), gate.__dict__)
    return gate


def _verify_dqi_structured_saved_identity(gate, bundle: Path, provenance: dict, *, source_root: Path,
                                          notebook_roots: tuple):
    """Keep one immutable saved event distinct from reusable notebook outputs."""
    bundle, source_root = checked_directory(bundle), checked_directory(source_root)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or not isinstance(notebook_roots, tuple)
            or len(notebook_roots) not in (1, 2)):
        raise ValueError("Structured DQI requires authenticated source and notebook roots")
    manifest_raw = _read_bounded_public_member(bundle, "bundle.json", 64 * 1024)
    if hashlib.sha256(manifest_raw).hexdigest() != DQI_STRUCTURED_MANIFEST_SHA256:
        raise ValueError("Public structured DQI saved manifest identity changed")
    manifest = json.loads(manifest_raw)
    names = set(manifest["files"]) | {"bundle.json"}
    if len(names) != 25 or names != gate.REQUIRED_FILES | {"bundle.json"}:
        raise ValueError("Public structured DQI requires its exact 25-file graph")
    directories = {parent.as_posix() for name in names for parent in PurePosixPath(name).parents
                   if parent.as_posix() != "."}
    found = set()
    pending, entries = [(bundle, PurePosixPath("."))], 0
    while pending:
        current, relative_parent = pending.pop()
        with os.scandir(current) as children:
            for entry in children:
                entries += 1
                relative = relative_parent / entry.name
                if entries > 32 or entry.is_symlink():
                    raise ValueError("Public structured DQI graph has linked or excess entries")
                if entry.is_dir(follow_symlinks=False):
                    if relative.as_posix() not in directories:
                        raise ValueError("Public structured DQI graph has an unrecorded directory")
                    pending.append((Path(entry.path), relative))
                elif entry.is_file(follow_symlinks=False) and relative.as_posix() in names:
                    found.add(relative.as_posix())
                else:
                    raise ValueError("Public structured DQI graph has an unrecorded or nonordinary file")
    if found != names:
        raise ValueError("Public structured DQI graph is incomplete")
    buffers = {name: _read_bounded_public_member(bundle, name, 1024 * 1024) for name in names}
    if sum(map(len, buffers.values())) != 365602:
        raise ValueError("Public structured DQI fixed saved-event byte count changed")
    for name, binding in manifest["files"].items():
        raw = buffers[name]
        if name == "optimization_dqi_structured.ipynb":
            if hashlib.sha256(raw).hexdigest() != DQI_STRUCTURED_NOTEBOOK_SHA256:
                raise ValueError("Public structured DQI fixed executed notebook changed")
        elif binding != {"kind": "file", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}:
            raise ValueError("Public structured DQI saved member identity changed: " + name)
    for name, source_name in gate.SOURCE_PATHS.items():
        if name.endswith(".ipynb"):
            continue
        raw = _read_bounded_public_member(source_root, source_name, 64 * 1024)
        if (raw != buffers[name] or hashlib.sha256(raw).hexdigest() != hashes.get(source_name)
                or hashes.get(source_name) != DQI_STRUCTURED_SOURCE_HASHES[source_name]):
            raise ValueError("Public structured DQI canonical source differs from its saved copy")
    notebook_name = "notebooks/optimization_dqi_structured.ipynb"
    for notebook_root in notebook_roots:
        raw = _read_bounded_public_member(checked_directory(notebook_root), notebook_name, 1024 * 1024)
        if hashlib.sha256(raw).hexdigest() != hashes.get(notebook_name):
            raise ValueError("Normal structured DQI notebook differs from current public provenance")
        if gate.notebook_source_sha256(raw) != DQI_STRUCTURED_NOTEBOOK_SOURCE_SHA256:
            raise ValueError("Normal structured DQI notebook source changed")
        codes = [cell for cell in json.loads(raw)["cells"] if cell["cell_type"] == "code"]
        if ([cell.get("execution_count") for cell in codes] != [1, 2, 3, 4]
                or any(item.get("output_type") == "error" for cell in codes for item in cell.get("outputs", []))):
            raise ValueError("Normal structured DQI notebook needs four successful saved-data cells")
    result = gate.verify_saved_bundle(bundle)
    verification = result.get("verification") if isinstance(result, dict) else None
    if (not isinstance(verification, dict) or [case["n"] for case in verification["small"]["cases"]] != [2, 3]
            or verification["large"]["n"] != 8 or verification["large"]["source_operations"] != 2876):
        raise ValueError("Public structured DQI saved check changed its bounded scope")
    return result


BATH_BUNDLE_DIRECTORY = "assets/generated/simulation/reusable-bath"
BATH_BUNDLE_NAMES = frozenset({
    "artifact-manifest.json", "endpoints.json", "endpoints.png", "endpoints.svg",
    "fixture-ground.svg", "fixture-operations.json", "fixture-thermal.svg", "primitive-ledger.json",
    "results.json", "sampled-schedules.json", "source-exports.json", "source-ground.qasm",
    "source-thermal.qasm", "source/LICENSE.txt", "source/PROTOCOL-NOTE.md",
    "source/requirements-lock.txt", "source/reusable_bath.py", "source/reusable_bath_protocol.json",
    "source/reusable_bath_reference.py",
})


def _verify_bath_saved_identity(root: Path, *, provenance: dict, notebook_roots: tuple):
    """Authenticate the curated bath data and ZIP without importing its sources."""
    root = checked_directory(root)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or not isinstance(notebook_roots, tuple)
            or len(notebook_roots) not in (1, 2)):
        raise ValueError("Authenticated bath notebook provenance and roots are required")
    notebook_name = "notebooks/simulation_reusable_bath.ipynb"
    for notebook_root in notebook_roots:
        notebook_raw = _read_bounded_public_member(notebook_root, notebook_name, 1024 * 1024)
        if hashlib.sha256(notebook_raw).hexdigest() != hashes.get(notebook_name):
            raise ValueError("Normal bath notebook differs from complete public provenance")
        notebook = json.loads(notebook_raw)
        cells = notebook.get("cells")
        if notebook.get("nbformat") != 4 or not isinstance(cells, list) or len(cells) != 12:
            raise ValueError("Normal bath notebook cell contract changed")
        projection = []
        for cell in cells:
            source = cell.get("source")
            if isinstance(source, list) and all(isinstance(part, str) for part in source):
                source = "".join(source)
            if (not isinstance(source, str) or not isinstance(cell.get("id"), str)
                    or cell.get("cell_type") not in ("code", "markdown", "raw")):
                raise ValueError("Normal bath notebook has invalid source cells")
            projection.append({"id": cell["id"], "cell_type": cell["cell_type"], "source": source})
        encoded = json.dumps(projection, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if hashlib.sha256(encoded).hexdigest() != "ffccc1cb6c6e894544e90525c19039340df125814623c272e6e7d5b0a968aeec":
            raise ValueError("Normal bath notebook source changed")
    directory = checked_directory(root / BATH_BUNDLE_DIRECTORY)
    raw = _read_bounded_public_member(root, BATH_BUNDLE_DIRECTORY + "/artifact-manifest.json", 64 * 1024)
    if hashlib.sha256(raw).hexdigest() != "7f1a4c720b5576e1a9c27593bbd5001078a7c104bd21dfe13c759ed532304d78":
        raise ValueError("Saved bath manifest identity changed")
    manifest = json.loads(raw)
    records = manifest.get("files")
    if (manifest.get("schema_version") != 1
            or manifest.get("kind") != "portable_saved_reusable_bath_evidence"
            or manifest.get("status") != "computed_bounded_finite_channel_evidence"
            or manifest.get("all_components_verified") is not False
            or manifest.get("continuum_bridge") != "unavailable"
            or manifest.get("physical_budgets") != "unallocated/uncomputed"
            or not isinstance(records, dict) or set(records) != BATH_BUNDLE_NAMES - {"artifact-manifest.json"}):
        raise ValueError("Saved bath file set or component evidence scope changed")
    entries = list(directory.rglob("*"))
    if len(entries) > 64 or any(path.is_symlink() for path in entries):
        raise ValueError("Saved bath directory contains links or unexpected entries")
    if {path.relative_to(directory).as_posix() for path in entries} != BATH_BUNDLE_NAMES | {"source"}:
        raise ValueError("Saved bath directory contains unrecorded paths")
    actual = {path.relative_to(directory).as_posix() for path in entries if path.is_file()}
    if actual != BATH_BUNDLE_NAMES:
        raise ValueError("Saved bath directory differs from its curated file set")
    buffers = {"artifact-manifest.json": raw}
    total = len(raw)
    for name, record in records.items():
        if (not isinstance(record, dict) or set(record) != {"bytes", "sha256"}
                or type(record["bytes"]) is not int or not 0 < record["bytes"] <= 32 * 1024 * 1024):
            raise ValueError("Saved bath file binding is invalid")
        data = _read_bounded_public_member(root, BATH_BUNDLE_DIRECTORY + "/" + name, record["bytes"])
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise ValueError("Saved bath payload identity changed: " + name)
        total += len(data)
        if total > 32 * 1024 * 1024:
            raise ValueError("Saved bath data exceeds its total byte cap")
        buffers[name] = data
    archive = _read_bounded_public_member(root, "assets/generated/simulation/reusable-bath-evidence.zip", 2 * 1024 * 1024)
    if hashlib.sha256(archive).hexdigest() != "4bb166a28e8621996b0eeca2f2f85b8d753f027131765eaef57a2477042efc62":
        raise ValueError("Saved bath ZIP identity changed")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        members = bundle.infolist()
        prefix = "reusable-bath-evidence/"
        if (len(members) != 19 or {item.filename for item in members} != {prefix + name for name in BATH_BUNDLE_NAMES}
                or any(item.is_dir() or stat.S_IFMT(item.external_attr >> 16) not in (0, stat.S_IFREG) for item in members)):
            raise ValueError("Saved bath ZIP contains unexpected or linked members")
        for item in members:
            name = item.filename.removeprefix(prefix)
            expected = buffers[name]
            if item.file_size != len(expected):
                raise ValueError("Saved bath ZIP member size changed")
            with bundle.open(item) as stream:
                if stream.read(len(expected) + 1) != expected:
                    raise ValueError("Saved bath ZIP differs from the canonical data")
    return {"status": "saved_bath_files_verified", "files": 19, "bytes": total,
            "all_components_verified": False, "new_quantum_execution": False}


MULTIPRODUCT_SOURCE_HASHES = {
    "src/openquantum/simulation/multiproduct.py": "810f4c8431845f8d8310ab56dc84564fd521bc5ccb6c7d6d7ac6e398ad1b211a",
    "scripts/generate_simulation_multiproduct.py": "bd68b40ad9c3745241094c4e52707c0e2b17230db9fb1f01ba08579a3d2d6489",
    "tests/test_simulation_multiproduct.py": "75ac9e582c18a06634793ceb601faf96c5171d222dde40ef485a45a6622d36e6",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
    "scripts/verify_simulation_multiproduct_saved.py": "f506a41640492212e3d75659c990d3719e1819d05f8cc30d739d59dda2b83518",
}


def _load_multiproduct_saved_gate(package: Path, provenance: dict):
    """Compile the pure saved checker after authenticating promoted inputs."""
    package = checked_directory(package)
    name = "scripts/verify_simulation_multiproduct_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != hashlib.sha256(source_bytes).hexdigest()
            or any(hashes.get(path) != value for path, value in MULTIPRODUCT_SOURCE_HASHES.items())):
        raise ValueError("Public multiproduct checker or promoted source identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_multiproduct_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated multiproduct saved checker")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


BAYESIAN_SAVED_HELPER_SHA256 = "89e66be34696682ab1a856b8e91ea733e2d325dc8a20a486c8c2a2ff0024adcd"


def _load_bayesian_saved_gate(package: Path, provenance: dict):
    """Load the pure saved verifier from its authenticated package bytes."""
    package = checked_directory(package)
    name = "scripts/verify_bayesian_qpe_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    actual = hashlib.sha256(source_bytes).hexdigest()
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != actual
            or actual != BAYESIAN_SAVED_HELPER_SHA256):
        raise ValueError("Public Bayesian QPE saved verifier identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_bayesian_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated Bayesian QPE saved verifier")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


ROBUST_SAVED_HELPER_SHA256 = "6f385d2c2f33e253962647109a6d6a053a8badbb76d92c30999da614a700a059"
LOCAL_SPIN_SAVED_HELPER_SHA256 = "8ec71967d6ca653a194ed074ff40f05c29fb38faea7e540b4e8ec96e407bcc3b"


def _load_robust_saved_gate(package: Path, provenance: dict):
    """Load the robust saved-data verifier from authenticated package bytes."""
    package = checked_directory(package)
    name = "scripts/verify_robust_qpe_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    actual = hashlib.sha256(source_bytes).hexdigest()
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != actual
            or actual != ROBUST_SAVED_HELPER_SHA256):
        raise ValueError("Public robust QPE saved verifier identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_robust_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated robust QPE saved verifier")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


QPE_FEEDBACK_PUBLIC_HELPER_SHA256 = "bcf0e16169a3143bf1c1eb7afdf1e8b57e72fb9f4a2c07d88ee0b85701b08788"


def _load_qpe_feedback_public_gate(package: Path, provenance: dict):
    """Load the saved snapshot/notebook adapter only after byte authentication."""
    package = checked_directory(package)
    name = "scripts/verify_qpe_feedback_public.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    actual = hashlib.sha256(source_bytes).hexdigest()
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != actual
            or actual != QPE_FEEDBACK_PUBLIC_HELPER_SHA256):
        raise ValueError("Public feedback-QPE registration adapter identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_feedback_qpe_public_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated feedback-QPE registration adapter")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate



def _load_local_spin_saved_gate(package: Path, provenance: dict):
    """Load the pure local-spin saved gate after current package admission."""
    package = checked_directory(package)
    name = "scripts/verify_reusable_bath_local_spin_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    actual = hashlib.sha256(source_bytes).hexdigest()
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != actual
            or actual != LOCAL_SPIN_SAVED_HELPER_SHA256):
        raise ValueError("Public local-spin saved verifier identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_local_spin_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated local-spin saved verifier")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate



MEASURED_AND_SAVED_HELPER_SHA256 = "46bf04382743128b3200ded06d33070d8c7ba7951df0d66a35243873226d42fd"


def _load_measured_and_saved_gate(package: Path, provenance: dict):
    """Load the pure measured-AND gate after complete current-byte admission."""
    package = checked_directory(package)
    name = "scripts/verify_shor_measured_and_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    actual = hashlib.sha256(source_bytes).hexdigest()
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or hashes.get(name) != actual
            or actual != MEASURED_AND_SAVED_HELPER_SHA256):
        raise ValueError("Public measured-AND saved verifier identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_measured_and_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated measured-AND saved verifier")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


LOCAL_SPIN_REPRODUCTION_DIRECTORY = "assets/generated/simulation/reusable-bath-local-spin-reproduction"
LOCAL_SPIN_REPRODUCTION_MANIFEST_SHA256 = "1e8360b897a3f8beddedcd31c0874b6dd136327879d844f8104c51761b68b50a"
LOCAL_SPIN_REPRODUCTION_FILES = frozenset({
    "artifact-manifest.json", "artifact-manifest.sha256", "events.ndjson", "finite-reference.json",
    "interface-validation.json", "provenance.json", "reproduction.json", "source-contracts.json",
    "source/reusable_bath_local_spin_checks.py", "source/reusable_bath_local_spin_reproduce.py",
})
LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES = {
    "src/openquantum/simulation/reusable_bath_local_spin_reproduce.py": "4ca3e2f22c5c4ca5dad5a5d3f94f4762a3c2382cba6c39e560953830fd5c38b9",
    "tests/test_reusable_bath_local_spin_reproduce.py": "3c9e2c041dcb376e5756573d0ae4b46644fca6262969326ca3180423517e6b8a",
    "tests/isolated/reusable_bath_local_spin_checks.py": "ec7c6e300a867a6a89a3a9df9f3e3b027ab8f5ff5efbd204fdd37f3554707ee5",
}


def _verify_local_spin_reproduction_identity(root: Path, *, source_root=None, provenance=None):
    """Authenticate the public interface outcome using saved bytes only.

    The original local-spin gate remains unchanged. This independent record
    check performs no source import, direct assertion or finite-reference run.
    """
    root = checked_directory(root)
    source_root = checked_directory(root if source_root is None else source_root)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if provenance is not None and (not isinstance(provenance, dict)
            or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict)):
        raise ValueError("Complete local-spin reproduction provenance is required")
    total = 0
    for name, pin in LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES.items():
        raw = _read_bounded_public_member(source_root, name, 128 * 1024)
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != pin or hashes is not None and hashes.get(name) != pin:
            raise ValueError("Promoted local-spin reproduction source changed: " + name)
    directory = checked_directory(root / LOCAL_SPIN_REPRODUCTION_DIRECTORY)
    expected = LOCAL_SPIN_REPRODUCTION_FILES | {"source"}
    actual = set()
    for path in directory.iterdir():
        if path.name not in expected or path.is_symlink() or len(actual) >= 9:
            raise ValueError("Local-spin reproduction has unrecorded members")
        actual.add(path.name)
    for path in checked_directory(directory / "source").iterdir():
        name = "source/" + path.name
        if name not in expected or path.is_symlink() or len(actual) >= len(expected):
            raise ValueError("Local-spin reproduction has unrecorded source members")
        actual.add(name)
    if actual != expected:
        raise ValueError("Local-spin reproduction membership changed")
    raw = _read_bounded_public_member(root, LOCAL_SPIN_REPRODUCTION_DIRECTORY + "/artifact-manifest.json", 64 * 1024)
    if hashlib.sha256(raw).hexdigest() != LOCAL_SPIN_REPRODUCTION_MANIFEST_SHA256:
        raise ValueError("Local-spin reproduction manifest changed")
    total += len(raw)
    manifest = json.loads(raw)
    if manifest.get("kind") != "local-spin-public-reproduction-saved-evidence" or set(manifest.get("files", {})) != LOCAL_SPIN_REPRODUCTION_FILES - {"artifact-manifest.json", "artifact-manifest.sha256"}:
        raise ValueError("Local-spin reproduction manifest scope changed")
    contents = {"artifact-manifest.json": raw}
    for name in sorted(LOCAL_SPIN_REPRODUCTION_FILES - {"artifact-manifest.json"}):
        raw = _read_bounded_public_member(root, LOCAL_SPIN_REPRODUCTION_DIRECTORY + "/" + name, 1024 * 1024)
        total += len(raw)
        if name == "artifact-manifest.sha256":
            if raw != (LOCAL_SPIN_REPRODUCTION_MANIFEST_SHA256 + "\n").encode():
                raise ValueError("Local-spin reproduction sidecar changed")
        elif {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != manifest["files"][name]:
            raise ValueError("Local-spin reproduction payload changed: " + name)
        contents[name] = raw
    snapshot = {"source/reusable_bath_local_spin_reproduce.py": "src/openquantum/simulation/reusable_bath_local_spin_reproduce.py",
                "source/reusable_bath_local_spin_checks.py": "tests/isolated/reusable_bath_local_spin_checks.py"}
    if any(hashlib.sha256(contents[name]).hexdigest() != LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES[source] for name, source in snapshot.items()):
        raise ValueError("Local-spin reproduction snapshots differ from current sources")
    origin = json.loads(contents["provenance.json"])
    old_bundle = "assets/generated/simulation/reusable-bath-local-spin"
    if len(origin.get("files", {})) != 21:
        raise ValueError("Local-spin reproduction original input inventory changed")
    for name, record in origin["files"].items():
        raw = _read_bounded_public_member(root, old_bundle + "/" + name, 1024 * 1024)
        total += len(raw)
        if {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != record:
            raise ValueError("Local-spin reproduction original input changed: " + name)
    if total > 4 * 1024 * 1024 or contents["finite-reference.json"] != _read_bounded_public_member(root, old_bundle + "/reference.json", 1024 * 1024):
        raise ValueError("Local-spin reproduction input bound or reference identity changed")
    completion = json.loads(contents["interface-validation.json"])
    replay = json.loads(contents["reproduction.json"])
    events = [json.loads(line) for line in contents["events.ndjson"].splitlines()]
    if (completion.get("status") != "observed_cli_and_independent_postflight_passed"
            or completion.get("observed_cli_exit") != 0 or completion.get("direct_contract_passes") != 10
            or replay.get("input_fingerprint_before") != replay.get("input_fingerprint_after")
            or len(events) != 8 or [event.get("stage") for event in events] != [stage for stage in ("admission", "contracts", "reference", "finalize") for _ in range(2)]
            or any(record.get("all_components_verified") is not False for record in (manifest, origin, completion, replay))
            or completion.get("new_algorithm_result") is not False or completion.get("algorithm_accuracy_certificate") is not False
            or completion.get("mixing_certificate") is not False or completion.get("physical_budgets") != "uncomputed"):
        raise ValueError("Local-spin reproduction interface evidence scope changed")
    return {"status": "saved_local_spin_reproduction_verified", "files": 10, "original_inputs": 21,
            "new_reference_execution": False, "native_calls": 0, "all_components_verified": False}


DQI_RESOURCE_DIRECTORY = "assets/generated/optimization/dqi-structured-resources"
DQI_RESOURCE_SAVED_HELPER_SHA256 = "45b20a0bd2dd4d5e2812f26e370febf6d44520e5952e1c1253a42b5b4a853f41"
DQI_RESOURCE_SOURCE_HASHES = {
    "scripts/generate_optimization_dqi_resources.py": "6231478a2e89069412c87d54441ddfac06045f2784dbf7010ad8eb5a18066f60",
    "src/openquantum/optimization_dqi_resource_compile.py": "9f0ed4703c43ea1ac30f9a82c91ed70855bd8c03a94708af49a1f48e370717cf",
    "src/openquantum/optimization_dqi_resource_contract.py": "cedbd51687f0b9069b34eeebd6753a207df5d98b68cc70cc380b7087e0275971",
    "src/openquantum/optimization_dqi_resource_physical.py": "af164e369d92eedcdfe068382458e4f0b5b18851da10d45c5f5491d32e5ce45a",
    "src/openquantum/optimization_dqi_resource_verify.py": "80591a7cd8704124caa0ff5e479b57c2e5413f0d28c06692a81c97e0fe7d9780",
    "tests/test_optimization_dqi_resource_contract.py": "6e2a55eb22b718ce867255e43add59d22150c8f9290bc39620563c390affa118",
    "tests/test_optimization_dqi_resource_driver.py": "c5dfa5ca9daa48a22416f2ea7bc18e6f95440376d6d45f9d08f9f89c8156eadc"
}
DQI_RESOURCE_INPUT_BINDINGS = {
    "LICENSE": {
        "bytes": 1093,
        "sha256": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c"
    },
    "assets/generated/optimization/compiled-replay/physical-summary.json": {
        "bytes": 1980,
        "sha256": "33b6ba297cfd7313f0d29aff1f87e4f3b4b9e9290e0fcf2ff110866c370da7b1"
    },
    "assets/generated/optimization/dqi-structured/bundle.json": {
        "bytes": 4601,
        "sha256": "1e01422fc8a0961bc1771ceedaffacfdf90a9c1a3bc9a0b99efc2aa3f4100850"
    },
    "assets/generated/optimization/dqi-structured/small/n2/dqi.qasm": {
        "bytes": 1693,
        "sha256": "af09988e10bce0969e32527515b0a3a09a5105e35f0a253872dfa7730decd432"
    },
    "assets/generated/optimization/dqi-structured/small/n2/ledger.json": {
        "bytes": 4529,
        "sha256": "94c4ad862c3939e0583dc23e116c2a7a29580730c32a9650c7e73030c2e4dad1"
    },
    "assets/generated/optimization/dqi-structured/small/n2/operations.ndjson": {
        "bytes": 2247,
        "sha256": "6fa145372a48336ce4761ae142c16a3da50235f2355268c10823cbb45ed9ca8a"
    },
    "assets/generated/optimization/dqi-structured/small/n3/dqi.qasm": {
        "bytes": 4299,
        "sha256": "e85801409197e7195f9320ecba963e72dcf3f5c4d4b7bec52ff78941c7a9f23a"
    },
    "assets/generated/optimization/dqi-structured/small/n3/ledger.json": {
        "bytes": 4584,
        "sha256": "fb7d69f2b6cad7f550cfbacab0eaa660afdc917f24afaf7b95a0dda4c483429c"
    },
    "assets/generated/optimization/dqi-structured/small/n3/operations.ndjson": {
        "bytes": 6191,
        "sha256": "3a9a0ddfefd2b4f468100458e784012593a2404691e3f1b53bd318b409b2cfaf"
    },
    "assets/generated/optimization/dqi-structured/small/validated-small.json": {
        "bytes": 17715,
        "sha256": "e7aeadafbb838e00c22dbef400b7d0bf637d45a7aad4ff107dac79f7662b7d42"
    },
    "assets/generated/optimization/word-transport/baseline/physical-dag.json.gz": {
        "bytes": 5027908,
        "sha256": "731791448fcfd3e7fc2df18efa0d7d0db12c49254a709bb58d0286a93d1cca34"
    },
    "requirements-lock.txt": {
        "bytes": 1974,
        "sha256": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088"
    },
    "src/openquantum/instrument_resources.py": {
        "bytes": 52381,
        "sha256": "fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863"
    },
    "src/openquantum/optimization_dqi_structured.py": {
        "bytes": 21438,
        "sha256": "0daaf35c01fbf83bcdfad98c85f6af94420202d7b546c7892615bd18ff1ec028"
    }
}


def _load_dqi_resource_saved_gate(package: Path, provenance: dict):
    """Authenticate the saved-only reader and seven executed source files."""
    package = checked_directory(package)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict):
        raise ValueError("Complete DQI resource provenance is required")
    for name, pin in DQI_RESOURCE_SOURCE_HASHES.items():
        raw = _read_bounded_public_member(package, name, 128 * 1024)
        if hashlib.sha256(raw).hexdigest() != pin or hashes.get(name) != pin:
            raise ValueError("Executed DQI resource source changed: " + name)
    name = "scripts/verify_dqi_structured_resource_saved.py"
    raw = _read_bounded_public_member(package, name, 64 * 1024)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != DQI_RESOURCE_SAVED_HELPER_SHA256 or hashes.get(name) != actual:
        raise ValueError("DQI resource saved reader identity changed")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_verified_dqi_resource_saved_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated DQI resource saved reader")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), gate.__dict__)
    return gate


def _verify_dqi_resource_saved_identity(gate, root: Path, *, source_root=None, provenance=None):
    """Bind all fourteen historical inputs before inspecting recorded results."""
    root = checked_directory(root)
    source_root = checked_directory(root if source_root is None else source_root)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2" or not isinstance(hashes, dict):
        raise ValueError("Complete DQI resource input provenance is required")
    total = 0
    for name, pin in DQI_RESOURCE_INPUT_BINDINGS.items():
        base = root if name.startswith("assets/") else source_root
        raw = _read_bounded_public_member(base, name, 8 * 1024 * 1024)
        total += len(raw)
        if total > 16 * 1024 * 1024 or {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != pin or hashes.get(name) != pin["sha256"]:
            raise ValueError("Frozen DQI resource input changed: " + name)
    result = gate.verify_saved_resources(checked_directory(root / DQI_RESOURCE_DIRECTORY))
    if result.get("scientific_or_native_work_executed") is not False or result.get("files") != 14:
        raise ValueError("DQI resource reader changed its saved-only scope")
    return result


def _load_dynamics_record_gate(package: Path, provenance: dict):
    """Load the dynamics gate only from the complete authenticated package.

    The checked source snapshot is compiled directly, without importing a
    sibling module or consulting a bytecode or process-wide module cache.
    """
    package = checked_directory(package)
    name = "scripts/dynamics_record_release.py"
    source = package
    for part in PurePosixPath(name).parts:
        source /= part
        if source.is_symlink():
            raise ValueError(f"Linked public-package dynamics helper: {name}")
    source = checked_path(package, name)
    source_bytes = source.read_bytes()
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package dynamics helper differs from the complete manifest")
    spec = importlib.util.spec_from_file_location("_openquantum_verified_dynamics_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package dynamics helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


def verify_package(package: Path, *, run_tests: bool = True) -> None:
    package = checked_directory(package)
    provenance = verify_manifest(package)
    verify_formal_record(package)
    energy_gate = _load_energy_record_gate(package, provenance)
    energy_gate.verify_energy_record_identity(package / energy_gate.RECORD_DIRECTORY, package)
    dynamics_gate = _load_dynamics_record_gate(package, provenance)
    dynamics_gate.verify_dynamics_record_identity(package / dynamics_gate.RECORD_DIRECTORY, package)
    word_gate = _load_word_record_gate(package, provenance)
    word_gate.verify_public_word_transport(package, replay=True)
    suzuki_gate = _load_suzuki_record_gate(package, provenance)
    suzuki_gate.verify_bundle(package / suzuki_gate.BUNDLE_RELATIVE, source_root=package)
    gibbs_gate = _load_gibbs_channel_record_gate(package, provenance)
    gibbs_gate.verify_bundle(package / gibbs_gate.BUNDLE_RELATIVE, source_root=package)
    tail_gate = _load_tail_record_gate(package, provenance)
    tail_gate.verify_bundle(
        package / "assets/generated/instrument-resources", package, kind="instrument",
        expected_sha256="0333814a823ebece83910251328b1145fc84e3cfbdec68051af92e78280ba220",
        replay=False,
    )
    tail_gate.verify_bundle(
        package / "assets/generated/simulation/complete-resources", package, kind="complete",
        expected_sha256="86af1d97bad4ac1f4d8c5191d3e546f69ca82a5121bd8de7d7307081c70dc0b1",
        replay=False,
    )
    dqi_gate = _load_dqi_saved_gate(package, provenance)
    _verify_dqi_saved_identity(dqi_gate, package / "assets/generated/optimization/dqi", provenance,
                               notebook_roots=(package,))
    _verify_bath_saved_identity(package, provenance=provenance, notebook_roots=(package,))
    multiproduct_gate = _load_multiproduct_saved_gate(package, provenance)
    multiproduct_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    bayesian_gate = _load_bayesian_saved_gate(package, provenance)
    bayesian_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    robust_gate = _load_robust_saved_gate(package, provenance)
    robust_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    structured_dqi_gate = _load_dqi_structured_saved_gate(package, provenance)
    _verify_dqi_structured_saved_identity(
        structured_dqi_gate, package / DQI_STRUCTURED_BUNDLE_DIRECTORY, provenance,
        source_root=package, notebook_roots=(package,),
    )
    dqi_resource_gate = _load_dqi_resource_saved_gate(package, provenance)
    _verify_dqi_resource_saved_identity(dqi_resource_gate, package, source_root=package, provenance=provenance)
    feedback_gate = _load_qpe_feedback_public_gate(package, provenance)
    feedback_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    local_spin_gate = _load_local_spin_saved_gate(package, provenance)
    local_spin_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    _verify_local_spin_reproduction_identity(package, provenance=provenance)
    measured_and_gate = _load_measured_and_saved_gate(package, provenance)
    measured_and_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    spec = importlib.util.spec_from_file_location(
        "_openquantum_n3_public_gate", checked_path(package, "scripts/verify_simulation_lindblad_three.py"))
    n3_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n3_gate)
    n3_gate.verify_bundle(package / n3_gate.BUNDLE_RELATIVE, source_root=package, canonical_root=package)
    spec = importlib.util.spec_from_file_location(
        "_openquantum_n4_public_gate", checked_path(package, "scripts/verify_simulation_lindblad_four.py"))
    n4_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(n4_gate)
    n4_gate.verify_bundle(package / n4_gate.BUNDLE_RELATIVE, source_root=package, canonical_root=package)
    spec = importlib.util.spec_from_file_location(
        "_openquantum_lindblad_port_public_gate", checked_path(package, "scripts/verify_simulation_lindblad_port.py"))
    port_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(port_gate)
    # The reviewed saved-only gate distinguishes audit-trusted Trace digests
    # from deterministic JSON checks of retained profile dictionaries.
    port_gate.verify_bundle(package / port_gate.BUNDLE_RELATIVE, source_root=package, canonical_root=package)
    spec = importlib.util.spec_from_file_location(
        "_openquantum_compiled_qaoa_public_gate", checked_path(package, "scripts/verify_optimization_compiled.py"))
    compiled_gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiled_gate)
    # Saved verification deterministically re-exports in a guarded temporary
    # copy, compares every byte and keeps native production counters at zero.
    compiled_gate.verify_bundle(package / compiled_gate.BUNDLE_RELATIVE,
                                source_root=package, canonical_root=package, replay=True)
    # The verifier's own 60-second parent watchdog bounds the full independent
    # numerical/compiler/physical replay, including a blocked native import.
    subprocess.run([sys.executable, str(checked_path(package, "scripts/verify_simulation_lindblad_physical.py")),
                    "--root", str(package)], cwd=package, env=environment, check=True, timeout=65)
    # Probe every shipped module, including namespace-package simulation
    # modules, with source precedence independent of the installed checkout.
    probe = """from pathlib import Path
import importlib
source = Path.cwd() / 'src'
for path in sorted((source / 'openquantum').rglob('*.py')):
    if path.name == '__init__.py':
        continue
    name = '.'.join(path.relative_to(source).with_suffix('').parts)
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(source), name
print('All algorithm imports resolve inside the public package', flush=True)
"""
    subprocess.run([sys.executable, "-c", probe], cwd=package, env=environment, check=True)
    if run_tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=package, env=environment, check=True)


def extract_package(archive_path: Path, directory: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate public-package archive members")
        for info in archive.infolist():
            relative = PurePosixPath(info.filename)
            mode = stat.S_IFMT(info.external_attr >> 16)
            if ("\\" in info.filename or relative.is_absolute() or ".." in relative.parts
                    or not relative.parts or relative.parts[0] != "openquantum-examples"
                    or mode == stat.S_IFLNK):
                raise ValueError(f"Unsafe public-package member: {info.filename}")
        archive.extractall(directory)
    return directory / "openquantum-examples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, help="Check an already extracted clean package")
    parser.add_argument("--formal-only", action="store_true",
                        help="Verify recorded Lean source hashes only; do not execute Lean or pytest")
    parser.add_argument("--skip-tests", action="store_true", help="Check package identity/imports without pytest")
    args = parser.parse_args()
    if args.formal_only:
        verify_formal_record((args.package_dir or ROOT).resolve())
        print("Verified recorded Lean source hashes: classical C4 reference coverage only; Lean was not rerun")
        return
    if args.package_dir:
        verify_package(args.package_dir, run_tests=not args.skip_tests)
    else:
        with tempfile.TemporaryDirectory(prefix="openquantum-public-package-") as temporary:
            package = extract_package(ROOT / "assets/generated/openquantum-code.zip", Path(temporary).resolve())
            verify_package(package, run_tests=not args.skip_tests)
    suffix = " and complete tests" if not args.skip_tests else ""
    print(f"Verified public package: complete file hashes, recorded Lean, energy and dynamics identities, isolated imports{suffix}")


if __name__ == "__main__":
    main()
