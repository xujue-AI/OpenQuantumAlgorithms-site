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


def _load_decay_record_gate(package: Path, provenance: dict):
    """Compile the saved decay helper from authenticated bounded bytes."""
    package = checked_directory(package)
    name = "scripts/verify_detailed_balance_decay.py"
    source_bytes = _read_bounded_public_member(package, name, 2 * 1024 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict)
            or manifest.get(name) != hashlib.sha256(source_bytes).hexdigest()):
        raise ValueError("Public-package decay helper differs from the complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_decay_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package decay helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    return gate


CORRECTION_SAVED_HELPER_SHA256 = "e04bd5c9dec44319a25fba0ac8778d06b62785304fa335b37aadf878fb521871"
CORRECTION_SAVED_HELPER_BYTES = 33005
CORRECTION_DOWNLOAD_RELATIVE = "assets/generated/simulation/detailed-balance-correction-download.zip"
CORRECTION_DOWNLOAD_SHA256 = "79ffc7f1184dd04c3f9a7227b973826dff0095d3ab286b2bd8d1cfb57f64f891"
CORRECTION_DOWNLOAD_BYTES = 100053


def _load_correction_record_gate(package: Path, provenance: dict):
    """Compile the pinned saved reader and bind its closed download result."""
    package = checked_directory(package)
    name = "scripts/verify_detailed_balance_correction_saved.py"
    source_bytes = _read_bounded_public_member(package, name, 64 * 1024)
    manifest = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(manifest, dict) or len(source_bytes) != CORRECTION_SAVED_HELPER_BYTES
            or hashlib.sha256(source_bytes).hexdigest() != CORRECTION_SAVED_HELPER_SHA256
            or manifest.get(name) != CORRECTION_SAVED_HELPER_SHA256):
        raise ValueError("Public-package correction helper differs from its pinned complete manifest")
    if manifest.get(CORRECTION_DOWNLOAD_RELATIVE) != CORRECTION_DOWNLOAD_SHA256:
        raise ValueError("Public-package correction archive differs from its pinned complete manifest")
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_openquantum_verified_correction_record_gate", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated public-package correction helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(source_bytes, str(source), "exec"), gate.__dict__)
    saved_zipfile = gate.zipfile

    def admitted_zip_file(stream, *args, **kwargs):
        if type(stream) is not io.BytesIO:
            raise ValueError("Correction ZIP decoder requires a captured byte stream")
        decoder_bytes = stream.getvalue()
        decoder_sha256 = hashlib.sha256(decoder_bytes).hexdigest()
        if (len(decoder_bytes) != CORRECTION_DOWNLOAD_BYTES or decoder_sha256 != CORRECTION_DOWNLOAD_SHA256
                or manifest.get(CORRECTION_DOWNLOAD_RELATIVE) != decoder_sha256):
            raise ValueError("Saved correction ZIP decoder input identity changed")
        # Decode only a fresh snapshot of the bytes admitted here.
        return saved_zipfile.ZipFile(io.BytesIO(decoder_bytes), *args, **kwargs)

    class CorrectionZipAdmission:
        ZipFile = staticmethod(admitted_zip_file)
        ZIP_STORED = saved_zipfile.ZIP_STORED
        ZIP_DEFLATED = saved_zipfile.ZIP_DEFLATED
        BadZipFile = saved_zipfile.BadZipFile

    gate.zipfile = CorrectionZipAdmission()
    saved_download = gate.verify_download

    def verify_download(archive, bundle):
        archive_path = Path(archive).absolute()
        archive_raw = _read_bounded_public_member(archive_path.parent, archive_path.name, 128 * 1024)
        archive_sha256 = hashlib.sha256(archive_raw).hexdigest()
        if (len(archive_raw) != CORRECTION_DOWNLOAD_BYTES or archive_sha256 != CORRECTION_DOWNLOAD_SHA256
                or manifest.get(CORRECTION_DOWNLOAD_RELATIVE) != archive_sha256):
            raise ValueError("Saved correction download container identity changed")
        result = saved_download(archive, bundle)
        if (not isinstance(result, dict) or result.get("download_verified") is not True
                or result.get("download_files") != 26 or result.get("download_bytes") != CORRECTION_DOWNLOAD_BYTES
                or result.get("download_sha256") != CORRECTION_DOWNLOAD_SHA256):
            raise ValueError("Saved correction download container identity changed")
        return result

    gate.verify_download = verify_download
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


DQI_DEGREE_TWO_HELPER = "scripts/verify_dqi_degree_two_public_saved.py"
DQI_DEGREE_TWO_HELPER_SHA256 = "fde3babe65a1fa8b7ade3249c1b8251407760295363d563c5c56a1cb0acc9fe5"
DQI_DEGREE_TWO_MANIFEST = "assets/generated/optimization/dqi-degree-two-public/projection-manifest.json"
DQI_DEGREE_TWO_MANIFEST_SHA256 = "ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c"
DQI_DEGREE_TWO_ARCHIVE = "assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip"
DQI_DEGREE_TWO_SOURCE_CONFIG = "source-config/dqi-degree-two/pyproject.toml"
DQI_DEGREE_TWO_CONFIG_SHA256 = "fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404"
DQI_DEGREE_TWO_NOTEBOOK = "notebooks/optimization_dqi_degree_two.ipynb"
DQI_DEGREE_TWO_NOTEBOOK_SHA256 = "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"
DQI_DEGREE_TWO_SOURCE_FILES = (
    "src/openquantum/optimization_dqi_degree_two.py",
    "src/openquantum/optimization_dqi_degree_two_verify.py",
    "scripts/generate_optimization_dqi_degree_two.py",
    "scripts/validate_optimization_dqi_degree_two.py",
    "scripts/verify_dqi_degree_two_saved.py",
    "scripts/verify_dqi_degree_two_public_saved.py",
    "tests/test_optimization_dqi_degree_two.py",
    "tests/test_optimization_dqi_degree_two_boundaries.py",
)
DQI_DEGREE_TWO_EXTERNAL_BINDINGS = {
    DQI_DEGREE_TWO_MANIFEST: {"bytes": 7766, "sha256": DQI_DEGREE_TWO_MANIFEST_SHA256},
    "assets/generated/optimization/dqi-degree-two-public/download.json": {
        "bytes": 554, "sha256": "62ae7f37396d7d915ba66bfcaf4295cb2d47ea11caf71609ffa14fd913a5d8b8"},
    DQI_DEGREE_TWO_ARCHIVE: {
        "bytes": 233048, "sha256": "af0c1eae2fe34d8947d20330fe7b6cb47a5852b216145033bffecda07bbf29ae"},
}


def _load_dqi_degree_two_saved_gate(package: Path, provenance: dict):
    """Compile the pinned reader and guard its external metadata and ZIP bytes."""
    package = checked_directory(package)
    raw = _read_bounded_public_member(package, DQI_DEGREE_TWO_HELPER, 64 * 1024)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or len(raw) != 30011
            or hashlib.sha256(raw).hexdigest() != DQI_DEGREE_TWO_HELPER_SHA256
            or hashes.get(DQI_DEGREE_TWO_HELPER) != DQI_DEGREE_TWO_HELPER_SHA256):
        raise ValueError("Degree-two DQI reader differs from its pinned complete manifest")
    if any(hashes.get(name) != pin["sha256"] for name, pin in DQI_DEGREE_TWO_EXTERNAL_BINDINGS.items()):
        raise ValueError("Degree-two DQI external metadata differs from the complete manifest")
    source = checked_path(package, DQI_DEGREE_TWO_HELPER)
    spec = importlib.util.spec_from_file_location("_verified_dqi_degree_two_public_saved", source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the authenticated degree-two DQI saved reader")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), gate.__dict__)
    saved_read, saved_zipfile = gate._read, gate.zipfile
    external_names = {Path(name).name: (name, pin) for name, pin in DQI_DEGREE_TWO_EXTERNAL_BINDINGS.items()}

    def admitted_read(path, cap, check):
        data = saved_read(path, cap, check)
        path = Path(path)
        if path.parent.name == "dqi-degree-two-public" and path.name in external_names:
            name, pin = external_names[path.name]
            if ({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} != pin
                    or hashes.get(name) != pin["sha256"]):
                raise ValueError("Degree-two DQI external metadata changed before dispatch")
        return data

    def admitted_zip_file(stream, *args, **kwargs):
        if type(stream) is not io.BytesIO:
            raise ValueError("Degree-two DQI ZIP decoder requires a captured byte stream")
        data = stream.getvalue()
        pin = DQI_DEGREE_TWO_EXTERNAL_BINDINGS[DQI_DEGREE_TWO_ARCHIVE]
        if ({"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} != pin
                or hashes.get(DQI_DEGREE_TWO_ARCHIVE) != pin["sha256"]):
            raise ValueError("Degree-two DQI ZIP decoder input identity changed")
        return saved_zipfile.ZipFile(io.BytesIO(data), *args, **kwargs)

    class DegreeTwoZipAdmission:
        ZipFile = staticmethod(admitted_zip_file)
        ZIP_DEFLATED = saved_zipfile.ZIP_DEFLATED

    gate._read = admitted_read
    gate.zipfile = DegreeTwoZipAdmission()
    return gate


def _verify_dqi_degree_two_public_identity(gate, artifact_root: Path, *, source_root=None,
                                         provenance=None, notebook_roots=()):
    """Authenticate separate roots and provide only this reader its source overlay."""
    artifact_root = checked_directory(artifact_root)
    source_root = checked_directory(artifact_root if source_root is None else source_root)
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict) or not isinstance(notebook_roots, tuple)
            or len(notebook_roots) not in (0, 1, 2)):
        raise ValueError("Degree-two DQI requires authenticated artifact, source and notebook roots")
    notebook_roots = notebook_roots or (source_root,)
    external = {}
    for name, pin in DQI_DEGREE_TWO_EXTERNAL_BINDINGS.items():
        raw = _read_bounded_public_member(artifact_root, name, 256 * 1024)
        if ({"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} != pin
                or hashes.get(name) != pin["sha256"]):
            raise ValueError("Frozen degree-two DQI external member changed: " + name)
        external[name] = raw
    manifest = json.loads(external[DQI_DEGREE_TWO_MANIFEST])
    config = _read_bounded_public_member(source_root, DQI_DEGREE_TWO_SOURCE_CONFIG, 1024)
    config_pin = {"bytes": len(config), "sha256": hashlib.sha256(config).hexdigest()}
    if (config_pin != {"bytes": 705, "sha256": DQI_DEGREE_TWO_CONFIG_SHA256}
            or manifest["files"]["pyproject.toml"] != config_pin
            or hashes.get(DQI_DEGREE_TWO_SOURCE_CONFIG) != config_pin["sha256"]):
        raise ValueError("Original degree-two DQI source-configuration role changed")
    sources, total = {"pyproject.toml": config}, len(config)
    for name in DQI_DEGREE_TWO_SOURCE_FILES:
        raw = _read_bounded_public_member(source_root, name, 128 * 1024)
        pin = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if pin != manifest["files"][name] or hashes.get(name) != pin["sha256"]:
            raise ValueError("Frozen degree-two DQI source changed: " + name)
        total += len(raw)
        if total > 512 * 1024:
            raise ValueError("Degree-two DQI source overlay exceeds its byte bound")
        sources[name] = raw
    for name in ("LICENSE", "requirements-lock.txt"):
        raw = _read_bounded_public_member(source_root, name, 64 * 1024)
        pin = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if pin != manifest["files"][name] or hashes.get(name) != pin["sha256"]:
            raise ValueError("Degree-two DQI original dependency changed: " + name)
    checked_notebooks = []
    for root in notebook_roots:
        root = checked_directory(root)
        raw = _read_bounded_public_member(root, DQI_DEGREE_TWO_NOTEBOOK, 64 * 1024)
        if len(raw) != 19950 or hashlib.sha256(raw).hexdigest() != DQI_DEGREE_TWO_NOTEBOOK_SHA256:
            raise ValueError("Degree-two DQI genuine notebook bytes changed")
        if root == source_root and hashes.get(DQI_DEGREE_TWO_NOTEBOOK) != DQI_DEGREE_TWO_NOTEBOOK_SHA256:
            raise ValueError("Degree-two DQI notebook changed after package admission")
        checked_notebooks.append(root / DQI_DEGREE_TWO_NOTEBOOK)
    with tempfile.TemporaryDirectory(prefix="openquantum-dqi-degree-two-source-") as temporary:
        overlay = Path(temporary).resolve()
        for name, raw in sources.items():
            destination = overlay / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        result = gate.verify_public_packet(
            artifact_root, source_root=overlay, notebook_path=checked_notebooks[-1],
            expected_manifest_sha256=DQI_DEGREE_TWO_MANIFEST_SHA256,
            archive_path=artifact_root / DQI_DEGREE_TWO_ARCHIVE, time_limit_seconds=10.0)
    if (not isinstance(result, dict) or result.get("status") != "passed_saved_public_degree_two_packet"
            or result.get("public_manifest_sha256") != DQI_DEGREE_TWO_MANIFEST_SHA256
            or result.get("scoped_files") != 38 or result.get("archive_members") != 38
            or result.get("archive_checked") is not True or result.get("archive_bytes") != 233048
            or result.get("original_bundle_files") != 22 or result.get("genuine_companion_code_cells") != 4
            or any(result.get(name) is not False for name in (
                "scientific_work_reexecuted", "notebook_executed", "portable_scientific_replay_validated", "publication_complete"))):
        raise ValueError("Degree-two DQI reader changed its saved-only scope")
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


HAMILTONIAN_SOURCE_HASHES = {
    "src/openquantum/simulation/corrected_product_formula.py": "63fffc29105a2162ba5d0b37f8638738dad00161065734099b43654089839540",
    "scripts/generate_simulation_corrected_formula.py": "c360f90abd24d78507173677e19b887a33f78f2f29a5e1e09ce6a175339a5576",
    "tests/test_simulation_corrected_formula.py": "84ec744686482238e3282bff5b545fa82833a899da32d7927917d3a8e32b0c38",
    "scripts/generate_cpf2_public.py": "51bf0ead5b6fa14b2f8fe5e3365866e073507bd3c6e85d8fda33ce31d4ad88ec",
    "src/openquantum/simulation/gqsp.py": "a013aeae83968d071f76093647d0c9f68498718a035c215998c8c85d325a68d2",
    "scripts/generate_simulation_gqsp.py": "11db399ea75b755187656625bac227f73a0463e7a44d0c275ab7bd26775689b8",
    "tests/test_simulation_gqsp.py": "1a90e976ac59e75f53b5035e07210aba9dc1b5fbc2903c0bf5fd6445f1dcf8f5",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
HAMILTONIAN_PACKETS = {
    "cpf": {
        "directory": "assets/generated/simulation/corrected-formula",
        "manifest": "manifest.json",
        "manifest_sha256": "0e4136c0a60da714fb6bb3a7dbd76fe8acfbef2d0c96a2b9c53142d9a303979f",
        "schema": "openquantum.cpf2.public-packet.v1",
        "archive": "assets/generated/simulation/corrected-formula/cpf2-public-packet.zip",
        "archive_sha256": "a5b718436503880cee05da0d841e747566b684292f998d8a6cee82fd9f369176",
        "helper": "scripts/verify_simulation_cpf_public.py",
        "helper_sha256": "0a44da634cc6f98f2b2fa64323e79c475856419b7c030e236f5e426fd60e3604",
        "notebook": "notebooks/simulation_corrected_formula.ipynb",
        "original_notebook": "original-executed.ipynb",
    },
    "gqsp": {
        "directory": "assets/generated/simulation/gqsp",
        "manifest": "public-manifest.json",
        "manifest_sha256": "ae635da61217082b4116553478a2c285d897b500ae2d0250f65c1a8a53cec262",
        "schema": "openquantum.gqsp.public-manifest.v1",
        "archive": "assets/generated/simulation/gqsp-reproduce.zip",
        "archive_sha256": "3fd6b3d458fec2c336b508a90e3b51babfd57ab8e4cb275312675110384b47a9",
        "helper": "scripts/verify_simulation_gqsp_public.py",
        "helper_sha256": "8598030b01cba7308403d5c1a088795cbc6f510309a7a0547c007de94476c197",
        "notebook": "notebooks/simulation_gqsp.ipynb",
        "original_notebook": "simulation_gqsp.ipynb",
    },
}


def _hamiltonian_provenance(provenance):
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict)):
        raise ValueError("Complete Hamiltonian packet provenance is required")
    return hashes


def _load_hamiltonian_saved_gate(package: Path, provenance: dict, kind: str):
    """Compile one saved reader from authenticated bytes, without science imports."""
    package = checked_directory(package)
    hashes = _hamiltonian_provenance(provenance)
    for name, expected in HAMILTONIAN_SOURCE_HASHES.items():
        raw = _read_bounded_public_member(package, name, 128 * 1024)
        if hashlib.sha256(raw).hexdigest() != expected or hashes.get(name) != expected:
            raise ValueError("Original Hamiltonian source or dependency changed: " + name)
    packet = HAMILTONIAN_PACKETS[kind]
    name = packet["helper"]
    raw = _read_bounded_public_member(package, name, 64 * 1024)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != packet["helper_sha256"] or hashes.get(name) != actual:
        raise ValueError("Hamiltonian saved reader identity changed: " + kind)
    source = checked_path(package, name)
    spec = importlib.util.spec_from_file_location("_verified_hamiltonian_saved_" + kind, source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated Hamiltonian saved reader")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), gate.__dict__)
    return gate


def _hamiltonian_packet_tree(directory: Path, expected: set[str]):
    """Inspect a bounded closed tree before a saved reader sees it."""
    directory = checked_directory(directory)
    expected_dirs = {parent.as_posix() for name in expected for parent in PurePosixPath(name).parents
                     if parent.as_posix() != "."}
    files, directories, pending, entries = set(), set(), [directory], 0
    while pending:
        current = pending.pop()
        with os.scandir(current) as children:
            for child in children:
                entries += 1
                if entries > 256:
                    raise ValueError("Hamiltonian packet exceeds its closed tree bound")
                name = (Path(child.path).relative_to(directory)).as_posix()
                info = child.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    if name not in expected_dirs or name in directories:
                        raise ValueError("Unlisted or duplicate Hamiltonian packet directory")
                    directories.add(name)
                    pending.append(Path(child.path))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    if name not in expected or name in files or info.st_size > 4 * 1024 * 1024:
                        raise ValueError("Unlisted, duplicate or oversized Hamiltonian packet file")
                    files.add(name)
                else:
                    raise ValueError("Unsafe linked or nonregular Hamiltonian packet entry")
    if files != expected or directories != expected_dirs:
        raise ValueError("Hamiltonian packet tree is incomplete")


def _hamiltonian_notebook(raw: bytes, original: dict):
    book = json.loads(raw)
    def projection(document):
        def source(cell):
            value = cell["source"]
            if isinstance(value, str):
                return value
            if isinstance(value, list) and all(isinstance(item, str) for item in value):
                return "".join(value)
            raise ValueError("Hamiltonian notebook source must be text or a list of text")
        return [(cell["id"], cell["cell_type"], source(cell)) for cell in document["cells"]]
    codes = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    outputs = [output for cell in codes for output in cell.get("outputs", [])]
    if (projection(book) != projection(original) or len(codes) != 5
            or [cell.get("execution_count") for cell in codes] != [1, 2, 3, 4, 5]
            or any(output.get("output_type") == "error" for output in outputs)
            or not any("image/png" in output.get("data", {}) or "image/svg+xml" in output.get("data", {})
                       for output in outputs)):
        raise ValueError("Hamiltonian notebook lacks unchanged sources and genuine five-cell outputs")


def _verify_hamiltonian_saved_identity(gate, root: Path, *, kind: str, source_root=None,
                                       provenance=None, notebook_roots=()):
    """Bind immutable packet, source mirrors and fresh book sources before dispatch."""
    root = checked_directory(root)
    source_root = checked_directory(root if source_root is None else source_root)
    hashes = _hamiltonian_provenance(provenance)
    packet = HAMILTONIAN_PACKETS[kind]
    prefix = packet["directory"] + "/"
    manifest_name = prefix + packet["manifest"]
    raw = _read_bounded_public_member(root, manifest_name, 256 * 1024)
    if hashlib.sha256(raw).hexdigest() != packet["manifest_sha256"] or hashes.get(manifest_name) != packet["manifest_sha256"]:
        raise ValueError("Frozen Hamiltonian packet manifest changed")
    manifest = json.loads(raw)
    if manifest["schema"] != packet["schema"]:
        raise ValueError("Hamiltonian packet manifest schema changed")
    if kind == "cpf":
        records = {record["path"]: {"bytes": record["bytes"], "sha256": record["sha256"]}
                   for record in manifest["files"]}
        if len(manifest["files"]) != 120 or len(records) != 120:
            raise ValueError("CPF payload closure changed")
    else:
        records = manifest["files"]
        if not isinstance(records, dict) or len(records) != 23:
            raise ValueError("GQSP payload closure changed")
    for name in records:
        relative = PurePosixPath(name)
        if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name
                or relative.as_posix() != name or len(name) > 256):
            raise ValueError("Unsafe Hamiltonian packet member name")
    expected = set(records) | {packet["manifest"]}
    if kind == "cpf":
        expected.add("cpf2-public-packet.zip")
    _hamiltonian_packet_tree(root / packet["directory"], expected)
    blobs, total = {}, 0
    for name, record in records.items():
        payload = _read_bounded_public_member(root, prefix + name, 4 * 1024 * 1024)
        total += len(payload)
        if (total > 8 * 1024 * 1024 or {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} != record
                or hashes.get(prefix + name) != record["sha256"]):
            raise ValueError("Frozen Hamiltonian payload changed: " + name)
        blobs[name] = payload
    archive = _read_bounded_public_member(root, packet["archive"], 1024 * 1024)
    if hashlib.sha256(archive).hexdigest() != packet["archive_sha256"] or hashes.get(packet["archive"]) != packet["archive_sha256"]:
        raise ValueError("Frozen Hamiltonian archive changed")
    if kind == "gqsp":
        blobs[packet["manifest"]] = raw
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            members = zipped.infolist()
            if (len(members) != 24 or {item.filename for item in members} != {"gqsp/" + name for name in blobs}
                    or sum(item.file_size for item in members) > 4 * 1024 * 1024):
                raise ValueError("GQSP archive closure changed")
            for item in members:
                if (item.is_dir() or not stat.S_ISREG(item.external_attr >> 16) or item.flag_bits & 1
                        or item.file_size > 4 * 1024 * 1024 or zipped.read(item) != blobs[item.filename[5:]]):
                    raise ValueError("GQSP archive member differs from served bytes")
    original = json.loads(blobs[packet["original_notebook"]])
    for notebook_root in notebook_roots or (source_root,):
        notebook_root = checked_directory(notebook_root)
        book_raw = _read_bounded_public_member(notebook_root, packet["notebook"], 4 * 1024 * 1024)
        if notebook_root == source_root and hashes.get(packet["notebook"]) != hashlib.sha256(book_raw).hexdigest():
            raise ValueError("Hamiltonian book changed after package admission")
        _hamiltonian_notebook(book_raw, original)
    portable = source_root / "notebooks/simulation_corrected_formula_portable.ipynb"
    if portable.exists() or portable.is_symlink():
        raise ValueError("CPF portable companion must remain download-only")
    result = gate.verify(root / packet["directory"])
    if kind == "cpf":
        if (result.get("status") != "passed" or result.get("scientific_imports_or_calls") != 0
                or result.get("portable_execution_performed") is not False or result.get("payload_files") != 120
                or result.get("archive_entries") != 121 or result.get("paired_small_cases") != 32
                or result.get("genuine_notebook_code_cells") != 5):
            raise ValueError("CPF reader changed its saved-only scope")
    elif (result.get("status") != "saved_integrity_passed" or result.get("new_scientific_execution") is not False
          or result.get("public_files") != 24 or result.get("saved_suite_cases") != 8
          or result.get("small_numerical_cases") != 6 or result.get("generation_only_cases") != 2
          or result.get("genuine_notebook_cells") != 5):
        raise ValueError("GQSP reader changed its saved-only scope")
    helper = _read_bounded_public_member(source_root, packet["helper"], 64 * 1024)
    if hashlib.sha256(helper).hexdigest() != packet["helper_sha256"]:
        raise ValueError("Hamiltonian saved reader changed during verification")
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


# Exact retained public projections; no new numerical or formal execution.
NEXT_SAVED_PACKETS = {'thermal-two-round': {'helper': 'scripts/verify_thermal_two_round_public.py',
                       'helper_sha256': '128a4b64d0de76137b1bbdb49aa95cfc7ecdda45b66f1d70af8abae66f3d31c9',
                       'prefix': 'assets/generated/simulation/detailed-balance-two-round',
                       'manifest': 'assets/generated/simulation/detailed-balance-two-round/manifest.json',
                       'manifest_sha256': 'e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377',
                       'asset_hashes': {'assets/generated/simulation/detailed-balance-two-round/LICENSE.txt': {'bytes': 1093,
                                                                                                               'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
                                        'assets/generated/simulation/detailed-balance-two-round/README.md': {'bytes': 5536,
                                                                                                             'sha256': 'f0d05698d2cc3d6fb9ac9189dd44d8d3f2616a4da41df13cbfd2463eb0691934'},
                                        'assets/generated/simulation/detailed-balance-two-round/UPSTREAM-LICENSES.md': {'bytes': 2179,
                                                                                                                        'sha256': 'eab2d2d578ce1c6fa46bcade400a0b5e7d3d3e5557a811f2b461dc544ea99e9f'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/correction-heralded-step.qasm': {'bytes': 28765,
                                                                                                                                          'sha256': 'ef799261d2152586a4e6031a028c1ffdeb41dc89813174958a97b1f796274dd4'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/correction_lcu.qasm': {'bytes': 28765,
                                                                                                                                'sha256': 'ef799261d2152586a4e6031a028c1ffdeb41dc89813174958a97b1f796274dd4'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/round_macro.qasm': {'bytes': 43542,
                                                                                                                             'sha256': '3d0629430724e49c9a10ab807e722f53164ae18a49b3c73e6cecba22d71cd74a'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/two_round_trial.qasm': {'bytes': 93239,
                                                                                                                                 'sha256': 'a90cd2c06d3b765d3b27b666393e3367c321e4b2fba4c182ce2b5406546a53f2'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/weak_dilation.qasm': {'bytes': 14556,
                                                                                                                               'sha256': '8c772b5d751918b24c19a4f8356ce05ff63a1695cecdf094f3ff9c71ff486916'},
                                        'assets/generated/simulation/detailed-balance-two-round/circuits/weak_reset_prefix.qasm': {'bytes': 14652,
                                                                                                                                   'sha256': '61db5e3d7f498d19ce9948e962818259009c91f8dfc3b108c3ba0c43b0e000a8'},
                                        'assets/generated/simulation/detailed-balance-two-round/manifest.json': {'bytes': 3497,
                                                                                                                 'sha256': 'e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377'},
                                        'assets/generated/simulation/detailed-balance-two-round/manifest.sha256': {'bytes': 80,
                                                                                                                   'sha256': 'f3bfd5181075899ac7fe182716c1546ab90e9c8058a382207c27338df34aa4ed'},
                                        'assets/generated/simulation/detailed-balance-two-round/requirements-lock.txt': {'bytes': 1974,
                                                                                                                         'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
                                        'assets/generated/simulation/detailed-balance-two-round/results.json': {'bytes': 1155156,
                                                                                                                'sha256': 'ad4460dd4c63c7254bc5a790814df9f178ded7f43775efbb740ac208904a408a'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_access.py': {'bytes': 25361,
                                                                                                                                      'sha256': '7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_channel.py': {'bytes': 24362,
                                                                                                                                       'sha256': '99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_correction.py': {'bytes': 24902,
                                                                                                                                          'sha256': '62ede4bbafe79a15e7c66068f0dc2b54a32322bbbff4015f00e7a0ba64e3eb48'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_decay.py': {'bytes': 12324,
                                                                                                                                     'sha256': '82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_thermal_two_round.py': {'bytes': 22928,
                                                                                                                                                 'sha256': '45ae8cc6b55d39caf29b80a8cf1aef770b7dae29d45d6e6bc41e341890c3b388'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/detailed_balance_thermal_two_round_reference.py': {'bytes': 39381,
                                                                                                                                                           'sha256': '98dfc9f75e6849419427ec2256a51adde1a52b1ac2bd6f8ee59ffaaa35ca8b52'},
                                        'assets/generated/simulation/detailed-balance-two-round/sources/thermal_two_round_contract.py': {'bytes': 7277,
                                                                                                                                         'sha256': '2a5f41bf8e36981ef9adac3928aaede05b984b094771320158d45f771f2d9fb1'},
                                        'assets/generated/simulation/detailed-balance-two-round/summary.json': {'bytes': 2629,
                                                                                                                'sha256': '84034c9fe48802232aafde8b9f9c65e6d7b37484c2e84a589ddd5a781ffb341e'},
                                        'assets/generated/simulation/detailed-balance-two-round/two-round-circuit.svg': {'bytes': 10610,
                                                                                                                         'sha256': '46eec2cb49da05e716b6c358293a370ac4aecb86ece06dbcd5a9c40430ae4770'}},
                       'source_hashes': {'scripts/display_thermal_two_round_saved.py': '0e2caf47a6225d9d7c8ab8b4e23f72ee6193dd1b6dd7cf0a521eae40d3922fbc',
                                         'scripts/verify_thermal_two_round_public.py': '128a4b64d0de76137b1bbdb49aa95cfc7ecdda45b66f1d70af8abae66f3d31c9'}},
 'shor-finite-adder': {'helper': 'scripts/verify_shor_finite_adder_reference.py',
                       'helper_sha256': '0549789470c469e3b5e6ac86e40a1ba92cd2d19064afd07fe1f049b9a2a2f9fb',
                       'prefix': 'assets/generated/shor/finite-adder-reference',
                       'manifest': 'assets/generated/shor/finite-adder-reference/manifest.json',
                       'manifest_sha256': 'a1b76407144da82df6065dd666e85da106c3e404dbd3af25b05b6e81e78164eb',
                       'asset_hashes': {'assets/generated/shor/finite-adder-reference/LICENSE.txt': {'bytes': 1093,
                                                                                                     'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
                                        'assets/generated/shor/finite-adder-reference/README.md': {'bytes': 2668,
                                                                                                   'sha256': '9d1d949a2799610d5c523b49d94e877d9dbb2c86d08bf27c29dfa0c6a43d1a0f'},
                                        'assets/generated/shor/finite-adder-reference/RippleTwoBitBool.lean': {'bytes': 6017,
                                                                                                               'sha256': 'ca5b82b3ef36fb3ee651ce439e32ea6fc442ed8c7b3cb91ada9efa567562055f'},
                                        'assets/generated/shor/finite-adder-reference/axiom-audit.txt': {'bytes': 595,
                                                                                                         'sha256': '5967e8405d837377184177d9ea900bc6b29235302323bda89f29d04f5f421212'},
                                        'assets/generated/shor/finite-adder-reference/compilation.json': {'bytes': 3434,
                                                                                                          'sha256': 'db6ff7a90c7b5fa19cb3d52d9fab5c3ddb6ec6e58af77032212b7a0c38fe2858'},
                                        'assets/generated/shor/finite-adder-reference/lean-toolchain': {'bytes': 29,
                                                                                                        'sha256': '8190e75a201741065fe508b28955dd64dd72d090babe5f70ce6848879d68ae88'},
                                        'assets/generated/shor/finite-adder-reference/manifest.json': {'bytes': 1180,
                                                                                                       'sha256': 'a1b76407144da82df6065dd666e85da106c3e404dbd3af25b05b6e81e78164eb'},
                                        'assets/generated/shor/finite-adder-reference/ripple-two-bit-source.qasm': {'bytes': 258,
                                                                                                                    'sha256': '868b5fd14a9b1e12fe21a7329d6ea3eb07c613c13755d0bf87d486f27e018009'}},
                       'source_hashes': {'scripts/verify_shor_finite_adder_reference.py': '0549789470c469e3b5e6ac86e40a1ba92cd2d19064afd07fe1f049b9a2a2f9fb'}},
 'dqi-path-majority': {'helper': 'scripts/display_dqi_path_majority_saved.py',
                       'helper_sha256': '1e82bad0820cac97efd7a4805585a1b70aa89413de7d4af117170d2e230b219a',
                       'prefix': 'assets/generated/optimization/dqi-path-majority',
                       'manifest': 'assets/generated/optimization/dqi-path-majority/public-manifest.json',
                       'manifest_sha256': 'b6f6ad7f3e4914a6af7c34a9c852652e623b9d22de07635378ad379e816363d9',
                       'asset_hashes': {'assets/generated/optimization/dqi-path-majority/README.md': {'bytes': 5457,
                                                                                                      'sha256': '9914c333d5848f9d7a41f40a4ad02e074909a1c3c5c77da96c17c170779202c2'},
                                        'assets/generated/optimization/dqi-path-majority/accepted-components.json': {'bytes': 4758,
                                                                                                                     'sha256': 'c3792356e0d2c3348720e85203f9c1e68a334afd36c8933cba3bafa7fc917a77'},
                                        'assets/generated/optimization/dqi-path-majority/code/scripts/generate_optimization_dqi_path_majority.py': {'bytes': 23690,
                                                                                                                                                    'sha256': '01bafe43a205a2b04441ceb8c3aac99f4285b383899a57ed4296c3a6e64c8305'},
                                        'assets/generated/optimization/dqi-path-majority/code/scripts/validate_optimization_dqi_path_majority.py': {'bytes': 29413,
                                                                                                                                                    'sha256': 'febe90401a61a332133f3cd9d9229e704fb4e529564508f36ad12c8b510cdd42'},
                                        'assets/generated/optimization/dqi-path-majority/code/src/openquantum/optimization_dqi_path_majority.py': {'bytes': 36874,
                                                                                                                                                   'sha256': 'ba3015dbdf35fe3ccec6a79503f2e911b735c2ea716969f690eb5fbc5e3c61bf'},
                                        'assets/generated/optimization/dqi-path-majority/code/src/openquantum/optimization_dqi_path_majority_verify.py': {'bytes': 24109,
                                                                                                                                                          'sha256': '206c1e705bef8130677f835b68ec8a94a1fd420a53acd62d3d66b3ff8d536606'},
                                        'assets/generated/optimization/dqi-path-majority/code/tests/test_optimization_dqi_path_majority.py': {'bytes': 18529,
                                                                                                                                              'sha256': '5b205986d5c71be16ca043897c4f8e0e42a5eda7e1b2e29bf40edf3aa5a9e7b9'},
                                        'assets/generated/optimization/dqi-path-majority/code/tests/test_optimization_dqi_path_majority_boundaries.py': {'bytes': 14771,
                                                                                                                                                         'sha256': 'b2f962131f9b0ac60f03c8c0d1640855ae18142c807e0e455f766a1805ed414a'},
                                        'assets/generated/optimization/dqi-path-majority/companion-spec.json': {'bytes': 2780,
                                                                                                                'sha256': 'ec78280f68ffae353f941f8914fd7c0fc07a5f9b1499dcd21d3b1282de0efec7'},
                                        'assets/generated/optimization/dqi-path-majority/component-circuit.svg': {'bytes': 60796,
                                                                                                                  'sha256': '1b94fb4e2830d8d965cc736d5f5b5fcef13e5af8b2667c0de4aa89bff6c31544'},
                                        'assets/generated/optimization/dqi-path-majority/component-input-recipe.json': {'bytes': 785,
                                                                                                                        'sha256': 'd750d0d1d282c80da33548dd0cfa228a2b314db499a4bde7d8b3fc7ea85e12ef'},
                                        'assets/generated/optimization/dqi-path-majority/environment.json': {'bytes': 7342,
                                                                                                             'sha256': '6ac15827380faac174597aef4da5bbe10967376cbb00cc4c738f0abcd6eb63e5'},
                                        'assets/generated/optimization/dqi-path-majority/generated/component-ledger.json': {'bytes': 638,
                                                                                                                            'sha256': 'eae3ccf4c2259bfb8c367619b5d895e4e00e66de5a1a0700a5bfb114b99c77e4'},
                                        'assets/generated/optimization/dqi-path-majority/generated/component.ndjson': {'bytes': 14666,
                                                                                                                       'sha256': '3cbb0866a388adf23a5b27594d90a92db99cfbab4ad901d597eef39ae9a6f562'},
                                        'assets/generated/optimization/dqi-path-majority/generated/component.qasm': {'bytes': 5921,
                                                                                                                     'sha256': '7c4359771fd9274d9217cb60ce4066384b8aa2eaeffe219927e1c3bf0bb18091'},
                                        'assets/generated/optimization/dqi-path-majority/generated/manifest.json': {'bytes': 2300,
                                                                                                                    'sha256': 'daccd5efa3afa4a47b707a60a6a4617056d13685e4e0c61bd40a50e2734419a1'},
                                        'assets/generated/optimization/dqi-path-majority/generated/program-ledger.json': {'bytes': 6199,
                                                                                                                          'sha256': '26f240f45234477a0e739004ccec8921c6c6581b1c631ee10ed0f786ed3324fc'},
                                        'assets/generated/optimization/dqi-path-majority/generated/program.ndjson': {'bytes': 94905,
                                                                                                                     'sha256': 'b1a422a454fd13ece43c21e83bce17cffa919174a1741544ee9445e6c9e1c22b'},
                                        'assets/generated/optimization/dqi-path-majority/generated/program.qasm': {'bytes': 39912,
                                                                                                                   'sha256': 'a43c9d0ae35facdf0f19d68053bff42488db092ca9ba96f24eac787ab52fc1bd'},
                                        'assets/generated/optimization/dqi-path-majority/input-k3.json': {'bytes': 71,
                                                                                                          'sha256': '3ffac5f985d06910bd5c194a5e56b15beffaa3535d7b76b9e05743717faeab30'},
                                        'assets/generated/optimization/dqi-path-majority/public-manifest.json': {'bytes': 6318,
                                                                                                                 'sha256': 'b6f6ad7f3e4914a6af7c34a9c852652e623b9d22de07635378ad379e816363d9'},
                                        'assets/generated/optimization/dqi-path-majority/results-summary.json': {'bytes': 5595,
                                                                                                                 'sha256': '216817dc4a1b6c05c7538132f617771706c1fe572d0ff412b36382edcc26689d'},
                                        'assets/generated/optimization/dqi-path-majority/validation/manifest.json': {'bytes': 1155,
                                                                                                                     'sha256': '6f8e26cde01ad76ea8e37a9fbbaac6274439377609051121408cfbdc73d0414f'},
                                        'assets/generated/optimization/dqi-path-majority/validation/validation.json': {'bytes': 278512,
                                                                                                                       'sha256': '4d10497bb5ec2bf335de94212f367c0fa3f4af3740ead4af9725328f4b63179d'}},
                       'source_hashes': {'scripts/display_dqi_path_majority_saved.py': '1e82bad0820cac97efd7a4805585a1b70aa89413de7d4af117170d2e230b219a',
                                         'scripts/generate_optimization_dqi_path_majority.py': '01bafe43a205a2b04441ceb8c3aac99f4285b383899a57ed4296c3a6e64c8305',
                                         'scripts/plot_dqi_path_majority_component_svg.py': 'b063e57621b7a70301ac11f18c6269e0c0c1444481f2e2190490d09bc717febd',
                                         'scripts/validate_optimization_dqi_path_majority.py': 'febe90401a61a332133f3cd9d9229e704fb4e529564508f36ad12c8b510cdd42',
                                         'src/openquantum/optimization_dqi_path_majority.py': 'ba3015dbdf35fe3ccec6a79503f2e911b735c2ea716969f690eb5fbc5e3c61bf',
                                         'src/openquantum/optimization_dqi_path_majority_verify.py': '206c1e705bef8130677f835b68ec8a94a1fd420a53acd62d3d66b3ff8d536606',
                                         'tests/test_dqi_path_majority_public_projection.py': '4dfb5730c79f22054e4ae84798ecf6312e514545cb3410c02f4bdbe02b1829c1',
                                         'tests/test_optimization_dqi_path_majority.py': '5b205986d5c71be16ca043897c4f8e0e42a5eda7e1b2e29bf40edf3aa5a9e7b9',
                                         'tests/test_optimization_dqi_path_majority_boundaries.py': 'b2f962131f9b0ac60f03c8c0d1640855ae18142c807e0e455f766a1805ed414a'}},
 'rpe-resource-source': {'helper': 'scripts/verify_robust_qpe_resource_source_saved.py',
                         'helper_sha256': 'c5019ed950e7dfd445135036dbe49c873db6e67c9d1f63caa19c69cf658c8267',
                         'prefix': 'assets/generated/robust-qpe-resource-source',
                         'manifest': 'assets/generated/robust-qpe-resource-source/projection-manifest.json',
                         'manifest_sha256': 'aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33',
                         'asset_hashes': {'assets/generated/robust-qpe-resource-source/LICENSE.txt': {'bytes': 1093,
                                                                                                      'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
                                          'assets/generated/robust-qpe-resource-source/README.md': {'bytes': 4310,
                                                                                                    'sha256': '826c362853966aa6a28d93d612bb6aa8bf18359ffd426bcf921c9bc45e222411'},
                                          'assets/generated/robust-qpe-resource-source/code/LICENSE': {'bytes': 1093,
                                                                                                       'sha256': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c'},
                                          'assets/generated/robust-qpe-resource-source/code/pyproject.toml': {'bytes': 705,
                                                                                                              'sha256': 'fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404'},
                                          'assets/generated/robust-qpe-resource-source/code/requirements-lock.txt': {'bytes': 1974,
                                                                                                                     'sha256': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088'},
                                          'assets/generated/robust-qpe-resource-source/code/scripts/generate_robust_qpe_resources.py': {'bytes': 16745,
                                                                                                                                        'sha256': '5c5b34f3041a33d319232d60c5bd06067893f328a6645d1c38ed51860631e393'},
                                          'assets/generated/robust-qpe-resource-source/code/scripts/verify_robust_qpe_resource_source_saved.py': {'bytes': 18745,
                                                                                                                                                  'sha256': 'c5019ed950e7dfd445135036dbe49c873db6e67c9d1f63caa19c69cf658c8267'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/__init__.py': {'bytes': 401,
                                                                                                                           'sha256': 'af9ea652a9f24962c7808f9155735420b5444f530a0431951b87922ffd9f9663'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/grover.py': {'bytes': 4377,
                                                                                                                         'sha256': 'cd637cd2373c79bc2ad97566f83d70c2953034ff7ded2a37805818e7d942bf21'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/hadamard_qpe.py': {'bytes': 16847,
                                                                                                                               'sha256': '2f0571046d4001af592cb07ce08993290bbdcd14da230315cd3f9722ac881bb5'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/instrument_resources.py': {'bytes': 52381,
                                                                                                                                       'sha256': 'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/robust_qpe.py': {'bytes': 14438,
                                                                                                                             'sha256': 'cba515031c6c422e197bfc7776eccd9a0cfc46baa4904ab85585d0b30026b39b'},
                                          'assets/generated/robust-qpe-resource-source/code/src/openquantum/robust_qpe_resources.py': {'bytes': 22749,
                                                                                                                                       'sha256': 'dc6081367516ef33d2ad31ffdd3531439808ef664bed3febf53266b60651e545'},
                                          'assets/generated/robust-qpe-resource-source/code/tests/test_robust_qpe_resources.py': {'bytes': 16312,
                                                                                                                                  'sha256': 'a77daf938d716bae17f4fe30dfc85e5b82e8f8971dcdc80ae09e2a84f3adcf77'},
                                          'assets/generated/robust-qpe-resource-source/code/tests/test_robust_qpe_resources_producer.py': {'bytes': 6788,
                                                                                                                                           'sha256': '384cd1ae4c837e893ac5d4f58558a58237008372640bf2682103309086cee1e8'},
                                          'assets/generated/robust-qpe-resource-source/evidence/initial-producer.py': {'bytes': 16533,
                                                                                                                       'sha256': 'f466be1d1ce7f022905103bb2f53dd2feb9e7cacb122835d52de4a196d1e2cfe'},
                                          'assets/generated/robust-qpe-resource-source/evidence/initial-source-manifest.json': {'bytes': 6249,
                                                                                                                                'sha256': 'c92c8dcb1c503bd573746450163698865ec15435e7e51118136dcf97c4a315c7'},
                                          'assets/generated/robust-qpe-resource-source/evidence/producer-README.md': {'bytes': 1322,
                                                                                                                      'sha256': 'e67498bce2298fe748ab6052539e25b6f28c3e26a2ee0138a8081189ce0befcd'},
                                          'assets/generated/robust-qpe-resource-source/evidence/repaired-source-manifest.json': {'bytes': 6247,
                                                                                                                                 'sha256': '4400c569f2a5fd716e0b1b348bce74ef5defdb3740a0b2cb0f3bb82d7ee8283c'},
                                          'assets/generated/robust-qpe-resource-source/projection-manifest.json': {'bytes': 5626,
                                                                                                                   'sha256': 'aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33'},
                                          'assets/generated/robust-qpe-resource-source/references.bib': {'bytes': 3637,
                                                                                                         'sha256': '40f8e96a89fd16af0584bedf4ce6f1e873544394d6add98ff40feebc92cd8428'},
                                          'assets/generated/robust-qpe-resource-source/source-export-descriptor.json': {'bytes': 493,
                                                                                                                        'sha256': 'eb533c1fb719207c09048476520ba72138fa925aceaf478bc5084b1289e250b1'},
                                          'assets/generated/robust-qpe-resource-source/source-export-download.zip': {'bytes': 81554,
                                                                                                                     'sha256': '2e49377067a3c2a02a55f172cebcc192fc2159e7f6ac682d6893967f9b801706'},
                                          'assets/generated/robust-qpe-resource-source/source-export-record.json': {'bytes': 4429,
                                                                                                                    'sha256': 'dcfce1590229237a8917d6377f6de78fe2e07ea47adb063fcc41a4bb57d36f8c'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-0.qasm': {'bytes': 925,
                                                                                                                      'sha256': 'a905686d8a8ee354011811ee6fb50d36f75bc43c8976b310e415d508516f279d'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-1.qasm': {'bytes': 935,
                                                                                                                      'sha256': '0d8dab4bb43a8b5ed675c1aa8c40e724d31b3cea9119f64473080aff17782a71'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-10.qasm': {'bytes': 21261,
                                                                                                                       'sha256': '1bcb2f42da958fd58b7167d70f76f4a035f9ca1976242d6757e99760a11ae8a3'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-11.qasm': {'bytes': 21271,
                                                                                                                       'sha256': 'd042ce6d42e21d5d0deb12343e44fd882669623198bf16aa9a52e7c27919b2b9'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-2.qasm': {'bytes': 1581,
                                                                                                                      'sha256': 'da3a63c735e67c68f4a7fe7a71228f30358c6480fb41112b6b560026946f28d2'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-3.qasm': {'bytes': 1591,
                                                                                                                      'sha256': 'b593608d00c056ba4449fd8a533bfd494af0f04e6881c90c2d47a0bb58ca5b49'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-4.qasm': {'bytes': 2893,
                                                                                                                      'sha256': '9fa217bbbd7642905cfd84bf516d628477cdd65c573360368bbb12f758a9a1ab'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-5.qasm': {'bytes': 2903,
                                                                                                                      'sha256': '015d2ae4e07cc0a8fd781898777acdc6caaf39d2fa0556c70c6b4371090a9fb3'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-6.qasm': {'bytes': 5517,
                                                                                                                      'sha256': '5a99e12c0a13f80f7f4ec52b67096c91c5310c4cf8a84115255e250557ead1a6'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-7.qasm': {'bytes': 5527,
                                                                                                                      'sha256': '3d30037e2947ae68d39d4725c0ced1604007a1b91e9ddbdc11e013fed5288c6b'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-8.qasm': {'bytes': 10765,
                                                                                                                      'sha256': 'c0ca0f7ea6e9bf8e51eb6377c5c07a5623ab498039532265a56b761df9114f85'},
                                          'assets/generated/robust-qpe-resource-source/source/source-fringe-9.qasm': {'bytes': 10775,
                                                                                                                      'sha256': 'e5e7550aa50ee2ced8a0fd629947a8493bbca3b2ecce745335bc82d8ec59661e'},
                                          'assets/generated/robust-qpe-resource-source/source/source-ledger.json': {'bytes': 779,
                                                                                                                    'sha256': '45d01c4a2eb0fb013e786a4553c492432014ad48141497882209bcdfa692e766'},
                                          'assets/generated/robust-qpe-resource-source/source/source-macrocycle.qasm': {'bytes': 85287,
                                                                                                                        'sha256': '3e52a131b2a9a72497e0e5571be479c535e6011bb304815b4394543a17ad6f07'}},
                         'source_hashes': {'LICENSE': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c',
                                           'requirements-lock.txt': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088',
                                           'scripts/generate_robust_qpe_resources.py': '5c5b34f3041a33d319232d60c5bd06067893f328a6645d1c38ed51860631e393',
                                           'scripts/verify_robust_qpe_resource_source_saved.py': 'c5019ed950e7dfd445135036dbe49c873db6e67c9d1f63caa19c69cf658c8267',
                                           'src/openquantum/__init__.py': 'af9ea652a9f24962c7808f9155735420b5444f530a0431951b87922ffd9f9663',
                                           'src/openquantum/grover.py': 'cd637cd2373c79bc2ad97566f83d70c2953034ff7ded2a37805818e7d942bf21',
                                           'src/openquantum/hadamard_qpe.py': '2f0571046d4001af592cb07ce08993290bbdcd14da230315cd3f9722ac881bb5',
                                           'src/openquantum/instrument_resources.py': 'fb8893b8db2ebfe142168e893c630cf5c00aa51d1772fc2cf0b9e9024ce0d863',
                                           'src/openquantum/robust_qpe.py': 'cba515031c6c422e197bfc7776eccd9a0cfc46baa4904ab85585d0b30026b39b',
                                           'src/openquantum/robust_qpe_resources.py': 'dc6081367516ef33d2ad31ffdd3531439808ef664bed3febf53266b60651e545',
                                           'tests/test_robust_qpe_resources.py': 'a77daf938d716bae17f4fe30dfc85e5b82e8f8971dcdc80ae09e2a84f3adcf77',
                                           'tests/test_robust_qpe_resources_producer.py': '384cd1ae4c837e893ac5d4f58558a58237008372640bf2682103309086cee1e8'},
                         'source_config_snapshot_only': 'assets/generated/robust-qpe-resource-source/code/pyproject.toml'}}


def _load_next_saved_gate(package, provenance, kind):
    packet = NEXT_SAVED_PACKETS[kind]
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict)):
        raise ValueError("Next saved evidence requires complete authenticated provenance")
    for name, expected in packet["source_hashes"].items():
        raw = _read_bounded_public_member(package, name, 128 * 1024)
        if hashes.get(name) != expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Next saved evidence source identity changed: " + name)
    raw = _read_bounded_public_member(package, packet["helper"], 128 * 1024)
    if (hashes.get(packet["helper"]) != packet["helper_sha256"]
            or hashlib.sha256(raw).hexdigest() != packet["helper_sha256"]):
        raise ValueError("Next saved evidence helper identity changed")
    source = checked_path(package, packet["helper"])
    spec = importlib.util.spec_from_file_location("_openquantum_next_saved_" + kind.replace("-", "_"), source)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load authenticated next saved evidence helper")
    gate = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(source), "exec"), gate.__dict__)
    return gate


def _verify_next_saved_identity(gate, artifact_root, *, source_root, provenance, kind, omitted_assets=None):
    packet = NEXT_SAVED_PACKETS[kind]
    hashes = provenance.get("file_sha256") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or provenance.get("schema") != "openquantum-public-package-v2"
            or not isinstance(hashes, dict)):
        raise ValueError("Next saved evidence requires complete authenticated provenance")
    split_roots = Path(artifact_root) != Path(source_root)
    if split_roots and not isinstance(omitted_assets, dict):
        raise ValueError("Served next evidence requires authenticated preview asset joins")
    for name, pin in packet["asset_hashes"].items():
        if hashes.get(name) != pin["sha256"]:
            raise ValueError("Next saved evidence provenance identity changed: " + name)
        if split_roots:
            binding = omitted_assets.get(name)
            if (not isinstance(binding, dict) or binding.get("site_path") != name
                    or binding.get("sha256") != pin["sha256"]):
                raise ValueError("Next saved evidence canonical preview identity changed: " + name)
        raw = _read_bounded_public_member(artifact_root, name, pin["bytes"] + 1)
        if len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]:
            raise ValueError("Next saved evidence canonical asset changed: " + name)
    for name, expected in packet["source_hashes"].items():
        raw = _read_bounded_public_member(source_root, name, 128 * 1024)
        if hashes.get(name) != expected or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Next saved evidence retained source changed: " + name)
    if kind == "thermal-two-round":
        result = gate.verify_full_bundle(Path(artifact_root) / packet["prefix"], provenance=provenance,
                                         omitted_assets=omitted_assets if split_roots else None)
        if result.get("full_byte_authenticated") is not True or result.get("scientific_work_reexecuted") is not False:
            raise ValueError("Thermal full-byte authentication contract changed")
    elif kind == "shor-finite-adder":
        result = gate.verify_reference(Path(artifact_root) / packet["prefix"])
        if (result.get("status") != "authenticated_recorded_finite_classical_reference"
                or result.get("members") != 8 or result.get("helper_complete") is not False
                or result.get("lean_executed_by_this_check") is not False
                or result.get("qasm_parsed_by_this_check") is not False
                or result.get("quantum_work_performed") is not False):
            raise ValueError("Shor finite reference scope changed")
    elif kind == "dqi-path-majority":
        result = gate.display_saved_component(artifact_root, source_root=source_root,
            expected_manifest_sha256=packet["manifest_sha256"], time_limit_seconds=5.0)
        if (result.get("status") != "displayed_authenticated_saved_component"
                or result.get("asset_files") != 24 or result.get("source_files") != 9
                or result.get("scientific_work_reexecuted") is not False
                or result.get("full_program_validated") is not False
                or result.get("notebook_executed") is not False):
            raise ValueError("DQI local component scope changed")
    elif kind == "rpe-resource-source":
        result = gate.verify_saved_source_export(Path(artifact_root) / packet["prefix"],
            packet["manifest_sha256"], source_root=source_root, max_seconds=30)
        expected_counts = {"actual_macrocycle_operations": 5286, "actual_unitary_operations": 5166,
                           "reset_operations": 108, "readout_operations": 12, "source_files_checked": 12,
                           "bundle_members_checked": 38, "zip_members_checked": 36,
                           "producer_or_test_invocations": 0, "quantum_package_imports": 0}
        if (result.get("status") != "verified_saved_source_export_bytes_recipe_and_outcome_scope"
                or result.get("source_only_invocation_succeeded") is not True
                or result.get("prior_source_only_invocation_succeeded") is not False
                or result.get("algorithm_numerically_validated") is not False
                or result.get("physical_resource_estimate_computed") is not False
                or any(type(result.get(name)) is not int or result[name] != expected
                       for name, expected in expected_counts.items())):
            raise ValueError("RPE source-only outcome scope changed")
    else:
        raise ValueError("Unknown next saved evidence kind")
    return result


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
    decay_gate = _load_decay_record_gate(package, provenance)
    decay_gate.verify_bundle(package / decay_gate.BUNDLE_RELATIVE, source_root=package)
    correction_gate = _load_correction_record_gate(package, provenance)
    correction_gate.verify_bundle(package / correction_gate.BUNDLE_RELATIVE, source_root=package)
    correction_gate.verify_download(
        package / "assets/generated/simulation/detailed-balance-correction-download.zip",
        package / correction_gate.BUNDLE_RELATIVE)
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
    degree_two_gate = _load_dqi_degree_two_saved_gate(package, provenance)
    _verify_dqi_degree_two_public_identity(
        degree_two_gate, package, source_root=package, provenance=provenance, notebook_roots=(package,))
    dqi_resource_gate = _load_dqi_resource_saved_gate(package, provenance)
    _verify_dqi_resource_saved_identity(dqi_resource_gate, package, source_root=package, provenance=provenance)
    feedback_gate = _load_qpe_feedback_public_gate(package, provenance)
    feedback_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    local_spin_gate = _load_local_spin_saved_gate(package, provenance)
    local_spin_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    _verify_local_spin_reproduction_identity(package, provenance=provenance)
    measured_and_gate = _load_measured_and_saved_gate(package, provenance)
    measured_and_gate.verify_saved(package, provenance=provenance, notebook_roots=(package,))
    for kind in HAMILTONIAN_PACKETS:
        hamiltonian_gate = _load_hamiltonian_saved_gate(package, provenance, kind)
        _verify_hamiltonian_saved_identity(hamiltonian_gate, package, kind=kind,
                                          provenance=provenance, notebook_roots=(package,))
    for kind in NEXT_SAVED_PACKETS:
        next_gate = _load_next_saved_gate(package, provenance, kind)
        _verify_next_saved_identity(next_gate, package, source_root=package, provenance=provenance, kind=kind)
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
