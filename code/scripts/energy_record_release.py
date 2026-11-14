#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Publication identity/accounting gate for the recorded core energy study.

Routine CI separately executes the bounded logical producer replay. This gate
checks the immutable native record and its published sources using only the
standard library. It does not rerun native estimation or the factory search.
The source root may be a partial ``code/`` preview while records stay in the
website's canonical assets directory.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat


# Replace only after independent review of a freshly frozen canonical manifest.
# This pin deliberately lives outside the three manifest-hashed verifier files.
ENERGY_RECORD_MANIFEST_SHA256 = "431f3c5a9568886cbecf292587e33b0a81d1c0c9a5456fc0a06709b3e957aab6"
MANIFEST_NAME = "resource-record-manifest.json"
RECORD_DIRECTORY = "assets/generated/energy-estimation"
CASES = ("default", "n2", "n4", "n6", "n8", "analytic-powers")
SOURCE_FILES = (
    "scripts/generate_energy_estimation.py", "scripts/source_provenance.py",
    "src/openquantum/__init__.py", "src/openquantum/grover.py",
    "src/openquantum/energy_estimation.py", "src/openquantum/qft_qpe.py",
    "src/openquantum/resource_estimation.py", "requirements-lock.txt", "LICENSE",
)
VERIFIER_FILES = (
    "scripts/verify_energy_estimation.py", "scripts/freeze_energy_estimation.py",
    "scripts/energy_physical_records.py",
)
LEDGER_SOURCE_FILES = (
    "scripts/generate_energy_estimation.py", "src/openquantum/energy_estimation.py",
    "src/openquantum/qft_qpe.py", "src/openquantum/resource_estimation.py",
    "requirements-lock.txt",
)
ROOT_RECORD_FILES = (
    "logical-resources.json", "measured-group.qasm", "molecular-fixture.json",
    "results.json", "surface-code.json", "surface-code-scaling.json",
    "access-comparison.json", "recycled-clock-resources.json", "recycled-clock.qasm",
    "dimer-access.svg", "small-qpe.svg", "recycled-clock.svg",
    "energy-distributions.svg", "scaling.svg", "scaling.csv", "energy_estimation.py",
    "qft_qpe.py", "resource_estimation.py", "generate_energy_estimation.py", "LICENSE.txt",
)
CORE_FILES = frozenset(ROOT_RECORD_FILES) | frozenset(
    f"{case}/{name}" for case in CASES[1:]
    for name in ("logical-resources.json", "measured-group.qasm", "surface-code.json")
)
PROFILE = {"name": "illustrative-p-1e-4", "physical_error_rate": 1e-4,
           "gate_time_ns": 100, "measurement_time_ns": 500}
CONTRACT = (
    "Native models computed separately; routine CI replays logical work and validates "
    "recorded models, without native re-estimation or factory search."
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def require(condition, message):
    if not condition:
        raise ValueError("Core energy record: " + message)


def checked_directory(root):
    """Reject externally supplied lexical directory aliases before any reads."""
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink() and root == root.resolve(),
            "unsafe or linked source/record root")
    return root


def checked_file(root, relative):
    """Check lexical roots/ancestors before opening one regular local file."""
    root = checked_directory(root)
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "unsafe relative filename")
    name = PurePosixPath(relative)
    require(not name.is_absolute() and name.as_posix() == relative
            and all(part not in ("", ".", "..") for part in name.parts),
            "unsafe relative filename")
    path = root
    for part in name.parts:
        path /= part
        require(not path.is_symlink(), f"linked file/directory: {relative}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode)
            and path.resolve().is_relative_to(root), f"missing/unsafe file: {relative}")
    return path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Core energy record: nonfinite JSON value: " + value)


def load_json(path):
    require(path.stat().st_size <= 128 * 1024 * 1024, "JSON size limit exceeded")
    value = json.loads(path.read_text(), object_pairs_hook=_unique, parse_constant=_nonfinite)
    require(isinstance(value, dict), f"expected JSON object: {path.name}")
    def finite(item):
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
        elif type(item) is float:
            require(math.isfinite(item), "nonfinite JSON number")
    finite(value)
    return value


def core_inventory(root):
    inventory = {}
    for name in sorted(CORE_FILES):
        path = checked_file(root, name)
        inventory[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    for path in Path(root).rglob("*"):
        relative = path.relative_to(root).as_posix()
        if relative == "pauli-access" or relative.startswith("pauli-access/"):
            continue
        if path.is_dir() and not path.is_symlink():
            require(relative in CASES[1:], f"unexpected core directory: {relative}")
        else:
            require(relative in CORE_FILES or relative == MANIFEST_NAME,
                    f"unexpected core entry: {relative}")
            checked_file(root, relative)
    return inventory


def source_hashes(root, names):
    return {name: sha256(checked_file(root, name)) for name in names}


def verify_energy_record_identity(records_dir, source_root, *, expected_sha256=None):
    """Authenticate all records/sources, then check the frozen model arithmetic.

    Native factory selection remains a trusted recorded result. No installed
    scientific package, estimator, circuit generator or logical replay runs.
    ``expected_sha256`` is supplied only by independent test/inspection callers;
    production publication callers use the external reviewed constant.
    """
    records_dir, source_root = Path(records_dir).absolute(), Path(source_root).absolute()
    expected_sha256 = ENERGY_RECORD_MANIFEST_SHA256 if expected_sha256 is None else expected_sha256
    require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256),
            "reviewed immutable manifest SHA256 is required")
    manifest_path = checked_file(records_dir, MANIFEST_NAME)
    require(sha256(manifest_path) == expected_sha256, "manifest differs from reviewed immutable identity")
    manifest = load_json(manifest_path)
    require(set(manifest) == {"schema_version", "status", "files", "sources", "verifier_sources",
            "physical_profile", "native_execution", "physical_validation", "contract"},
            "manifest has missing/unexpected fields")
    require(type(manifest["schema_version"]) is int and manifest["schema_version"] == 1
            and manifest["status"] == "frozen_computed_conditional_energy_record",
            "manifest is incomplete or unsupported")
    before = core_inventory(records_dir)
    require(isinstance(manifest["files"], dict) and set(manifest["files"]) == CORE_FILES,
            "record inventory has missing/unexpected files")
    for name, entry in manifest["files"].items():
        require(isinstance(entry, dict) and set(entry) == {"sha256", "bytes"}
                and isinstance(entry["sha256"], str) and SHA256.fullmatch(entry["sha256"])
                and type(entry["bytes"]) is int and entry["bytes"] > 0,
                f"invalid record inventory entry: {name}")
    require(manifest["files"] == before, "record inventory differs from reviewed manifest")
    sources = source_hashes(source_root, SOURCE_FILES)
    require(manifest["sources"] == sources, "published producer/adapter/lock sources differ")
    verifiers = source_hashes(source_root, VERIFIER_FILES)
    require(manifest["verifier_sources"] == verifiers, "published verifier/model sources differ")
    require(manifest["physical_profile"] == PROFILE and manifest["contract"] == CONTRACT,
            "frozen profile or verification contract differs")
    execution = manifest["native_execution"]
    require(isinstance(execution, dict) and set(execution) == {
        "producer_commit", "python", "packages", "device", "seed", "complete_default_producer",
        "native_estimation_performed", "exit_status", "wall_seconds", "peak_rss_kib"},
        "native completion receipt has missing/unexpected fields")
    require(isinstance(execution["producer_commit"], str) and COMMIT.fullmatch(execution["producer_commit"])
            and isinstance(execution["python"], str) and execution["python"].startswith("3.12.")
            and execution["device"] == "CPU" and type(execution["seed"]) is int
            and execution["seed"] == 20261112, "native source/environment receipt differs")
    require(execution["native_estimation_performed"] is True and execution["complete_default_producer"] is True
            and type(execution["exit_status"]) is int and execution["exit_status"] == 0,
            "native receipt is incomplete")
    require(type(execution["wall_seconds"]) in (int, float) and execution["wall_seconds"] > 0
            and type(execution["peak_rss_kib"]) is int and execution["peak_rss_kib"] > 0,
            "native wall/peak receipt is invalid")
    require(isinstance(execution["packages"], dict) and set(execution["packages"]) ==
            {"qiskit", "numpy", "scipy", "matplotlib", "qdk"}
            and all(isinstance(version, str) and version for version in execution["packages"].values()),
            "native package versions are missing")
    ledgers = {
        case: load_json(checked_file(records_dir, "logical-resources.json" if case == "default"
                                     else f"{case}/logical-resources.json"))
        for case in CASES
    }
    ledger_sources = {name: sources[name] for name in LEDGER_SOURCE_FILES}
    for case, ledger in ledgers.items():
        require(ledger.get("source_sha256") == ledger_sources, f"stale ledger sources: {case}")
        require(ledger.get("source_dirty") is None or ledger.get("source_dirty") is False,
                f"dirty/invalid native source metadata: {case}")
        commit = ledger.get("source_commit")
        require(commit is None or isinstance(commit, str) and COMMIT.fullmatch(commit),
                f"invalid native Git metadata: {case}")
        environment = ledger.get("environment")
        require(isinstance(environment, dict) and all(environment.get(key) == execution[key]
                for key in ("python", "packages", "device", "seed")), f"receipt/ledger environment differs: {case}")
    default = ledgers["default"]
    require((default["contract"]["dimers"], default["contract"]["phase_bits"],
             default["contract"]["requested_ideal_failure"]) == (1, 5, .005)
            and default["sequential_equivalent_simulation"]["performed"] is True,
            "record lacks the complete default experiment/simulation")
    for name, source in (
        ("generate_energy_estimation.py", "scripts/generate_energy_estimation.py"),
        ("energy_estimation.py", "src/openquantum/energy_estimation.py"),
        ("qft_qpe.py", "src/openquantum/qft_qpe.py"),
        ("resource_estimation.py", "src/openquantum/resource_estimation.py"),
        ("LICENSE.txt", "LICENSE"),
    ):
        require(sha256(checked_file(records_dir, name)) == sources[source],
                f"archived producer differs from published source: {name}")
    # Authenticate this stdlib-only model checker before loading its code from
    # the published source tree. No scientific producer/estimator is imported.
    model_path = checked_file(source_root, "scripts/energy_physical_records.py")
    spec = importlib.util.spec_from_file_location("_energy_release_physical_records", model_path)
    require(spec is not None and spec.loader is not None, "cannot load authenticated model checker")
    model = importlib.util.module_from_spec(spec)
    model_bytes = model_path.read_bytes()
    require(hashlib.sha256(model_bytes).hexdigest() == verifiers["scripts/energy_physical_records.py"],
            "model checker changed before loading")
    # Compile the checked bytes directly so verification creates no bytecode in
    # immutable package/site trees and never reuses an import-cache authority.
    exec(compile(model_bytes, str(model_path), "exec"), model.__dict__)
    physical = model.validate_physical_records(records_dir, ledgers, PROFILE)
    require(physical == manifest["physical_validation"], "physical validation summary differs")
    require(core_inventory(records_dir) == before and sha256(manifest_path) == expected_sha256,
            "immutable records changed during publication verification")
    require(source_hashes(source_root, SOURCE_FILES) == sources
            and source_hashes(source_root, VERIFIER_FILES) == verifiers,
            "published sources changed during publication verification")
    return {"status": "verified immutable core-energy publication record", "core_files": len(before),
            "manifest_sha256": expected_sha256, "native_reestimation_performed": False,
            "factory_search_repeated": False, "logical_producer_executed": False,
            "physical_validation": physical}


def site_energy_record_errors(site_root):
    try:
        verify_energy_record_identity(Path(site_root) / RECORD_DIRECTORY, Path(site_root) / "code")
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, AttributeError) as error:
        return [str(error)]
    return []
