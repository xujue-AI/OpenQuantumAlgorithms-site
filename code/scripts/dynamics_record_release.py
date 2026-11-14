#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stdlib-only identity/accounting gate for immutable saved dynamics records.

Native inputs remain bound to their historical source snapshot. Routine
logical replay is a separate operation using separately authenticated sources.
This publication gate performs neither replay nor Native/factory computation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import types

DYNAMICS_RECORD_MANIFEST_SHA256 = "0596353af49a0de40176069756cd9398749c21ede68e057ef84611bb6d5f4265"
MANIFEST_NAME = "resource-record-manifest.json"
MANIFEST_SCHEMA = "openquantum-dynamics-record-v2"
RECORD_DIRECTORY = "assets/generated/dynamics-resources"
SOURCE_REPRODUCTION_EVIDENCE_PATH = "assets/generated/dynamics-replay-evidence/linux-source-reproduction.json"
SOURCE_REPORT_FIELDS = frozenset(("schema_version", "status", "scope", "provenance", "canonical_cases_sha256",
                                 "reconstructed_source_cases_sha256", "summary", "cases", "claims"))
SOURCE_REPORT_CLAIMS = {
    "canonical_costs_authoritative": True, "canonical_circuit_validation_performed": False,
    "source_circuit_equivalence_claimed": False, "formal_certificate": False,
    "hardware_or_synthesis_certificate": False, "Native_estimates_run": False,
    "phase_fitter_run": False, "costs_repriced": False,
}
HISTORICAL_ADAPTER_PATH = "assets/generated/dynamics-recorded-source/resource_estimation.py"
HISTORICAL_ADAPTER_SHA256 = "267329af4b100c7f9c82900f1f2713f934ec710e50fd8999d331c0f4b84654de"
HISTORICAL_SOURCE_COMMIT = "f842df78eb01cedd1fb438d5fd45c16fe48c435d"
NATIVE_SOURCE_FILES = (
    "src/openquantum/dynamics_resources.py", "scripts/generate_dynamics_resources.py",
    "src/openquantum/hamiltonian_simulation.py", "src/openquantum/resource_estimation.py",
    "src/openquantum/simulation/qsp.py", "src/openquantum/simulation/qsp_evolution.py",
    "src/openquantum/simulation/lcu.py", "src/openquantum/simulation/amplification.py",
    "requirements-lock.txt",
)
REPLAY_SOURCE_FILES = NATIVE_SOURCE_FILES + (
    "src/openquantum/__init__.py", "src/openquantum/grover.py",
    "scripts/source_provenance.py", "LICENSE",
)
SOURCE_FILES = REPLAY_SOURCE_FILES
VERIFIER_FILES = (
    "scripts/verify_dynamics_resources.py", "scripts/freeze_dynamics_resources.py",
    "scripts/dynamics_physical_records.py",
    "scripts/dynamics_canonical_replay.py", "scripts/dynamics_source_transport.py",
    "scripts/tail_resource_qasm.py",
)
FROZEN_FILES = (
    "src/openquantum/grover.py", "scripts/estimate_resources.py",
    "assets/generated/grover/circuit.qasm",
    "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv",
)
_product_parameters = {(n, 1., order, .05) for n in range(2, 7) for order in (1, 2)}
_product_parameters.update((3, time, order, .05) for time in (.5, 2.) for order in (1, 2))
_product_parameters.update((3, 1., order, error) for error in (.025, .1) for order in (1, 2))
CASE_IDS = tuple(f"pf{order}-n{n}-t{time:g}-eps{error:g}"
                 for n, time, order, error in sorted(_product_parameters)) + tuple(
    f"qsp-oaa{rounds}-n3-t1-eps0.05" for rounds in range(4)
) + ("qsp-retargeted-n2-t1-eps0.05", "qsp-retargeted-n4-t1-eps0.05")
ROOT_RECORD_FILES = (
    "manifest.json", "physical-summary.csv", "summary.md",
    "product-formula-scaling.svg", "qsp-amplification-resources.svg", "qsp-size-resources.svg",
    "dynamics_resources.py", "generate_dynamics_resources.py", "LICENSE.txt",
)
CORE_FILES = frozenset(ROOT_RECORD_FILES) | frozenset(
    f"{case}/{name}" for case in CASE_IDS for name in (
        "logical-resources.json", "measured-trial.qasm", "frozen-p-1e-3.json", "frozen-p-1e-4.json")
)
PROFILES = (
    {"name": "frozen-p-1e-3", "physical_error_rate": .001, "gate_time_ns": 100, "measurement_time_ns": 500},
    {"name": "frozen-p-1e-4", "physical_error_rate": .0001, "gate_time_ns": 100, "measurement_time_ns": 500},
)
DISTANCE_DOMAIN = tuple(range(3, 42, 2))
CONTRACT = (
    "Saved conditional Native models retain their original source provenance; routine CI authenticates records, "
    "replays logical work, and checks frozen model arithmetic without Native re-estimation or factory search. "
    "No external completion receipt or measured wall/RSS is claimed."
)
NATIVE_EVIDENCE = {
    "kind": "saved_generator_completion_manifest", "external_completion_receipt_available": False,
    "wall_seconds": None, "peak_rss_kib": None, "declared_cases": 24, "declared_queries": 48,
}
NATIVE_ENVIRONMENT = {
    "python": "3.12.14", "device": "CPU", "seed": 20261112,
    "packages": {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4",
                 "scipy": "1.18.1", "matplotlib": "3.11.2"},
}
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_AGGREGATE_BYTES = 32 * 1024 * 1024
MAX_JSON_NODES = 200000
MAX_JSON_DEPTH = 64


def require(condition, message):
    if not condition:
        raise ValueError("Dynamics record: " + message)


def checked_directory(root):
    root = Path(root).absolute()
    require(root.is_dir() and not root.is_symlink() and root == root.resolve(),
            "unsafe or linked source/record root")
    return root


def checked_file(root, relative):
    root = checked_directory(root)
    require(isinstance(relative, str) and relative and "\\" not in relative, "unsafe relative filename")
    name = PurePosixPath(relative)
    require(not name.is_absolute() and name.as_posix() == relative
            and all(part not in ("", ".", "..") for part in name.parts), "unsafe relative filename")
    path = root
    for part in name.parts:
        path /= part
        require(not path.is_symlink(), f"linked file/directory: {relative}")
    require(path.is_file() and stat.S_ISREG(path.stat().st_mode) and path.resolve().is_relative_to(root),
            f"missing/unsafe file: {relative}")
    return path


def read_bounded(path, max_bytes=MAX_JSON_BYTES):
    """Read one regular no-follow fd snapshot, including growth after fstat."""
    require(type(max_bytes) is int and 0 < max_bytes <= MAX_AGGREGATE_BYTES, "invalid read budget")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        raise ValueError("Dynamics record: regular no-follow input required") from error
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "input fd must be regular")
        require(info.st_size <= max_bytes, "input snapshot size limit exceeded")
        chunks, remaining = [], max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(65536, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        require(len(payload) <= max_bytes, "input snapshot size limit exceeded")
        return payload
    finally:
        os.close(descriptor)


def sha256(path, max_bytes=MAX_JSON_BYTES):
    return hashlib.sha256(read_bounded(path, max_bytes)).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError("Dynamics record: nonfinite JSON value")


def parse_json_bytes(payload):
    require(type(payload) is bytes and len(payload) <= MAX_JSON_BYTES, "JSON snapshot size limit exceeded")
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_nonfinite)
    except RecursionError as error:
        raise ValueError("Dynamics record: JSON depth limit exceeded") from error
    require(isinstance(value, dict), "expected JSON object")
    remaining = MAX_JSON_NODES
    stack = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        remaining -= 1
        require(remaining >= 0 and depth <= MAX_JSON_DEPTH, "JSON traversal limit exceeded")
        if isinstance(item, (dict, list)):
            children = item.values() if isinstance(item, dict) else item
            require(len(children) + len(stack) <= remaining and (not children or depth < MAX_JSON_DEPTH),
                    "JSON traversal limit exceeded")
            stack.extend((child, depth + 1) for child in children)
        elif type(item) is float:
            require(math.isfinite(item), "nonfinite JSON number")
    return value


def load_json(path, *, expected_sha256=None):
    payload = read_bounded(path)
    if expected_sha256 is not None:
        require(isinstance(expected_sha256, str) and SHA256.fullmatch(expected_sha256)
                and hashlib.sha256(payload).hexdigest() == expected_sha256, "JSON snapshot identity differs")
    return parse_json_bytes(payload)


def core_inventory(root):
    root = checked_directory(root)
    inventory, total = {}, 0
    if _preferred_present_or_link(root, MANIFEST_NAME):
        sidecar = checked_file(root, MANIFEST_NAME)
        total = len(read_bounded(sidecar))
        require(total <= MAX_JSON_BYTES, "JSON size limit exceeded")
    for name in sorted(CORE_FILES):
        path = checked_file(root, name)
        payload = read_bounded(path)
        size = len(payload)
        total += size
        require(size > 0 and total <= MAX_AGGREGATE_BYTES, "record aggregate size limit exceeded")
        require(not name.endswith(".json") or size <= MAX_JSON_BYTES, "JSON size limit exceeded")
        inventory[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": size}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            require(relative in CASE_IDS, "unexpected core directory")
        else:
            require(relative in CORE_FILES or relative == MANIFEST_NAME, "unexpected core entry")
            checked_file(root, relative)
    return inventory


def source_hashes(root, names):
    result = {}
    for name in names:
        path = checked_file(root, name)
        require(path.stat().st_size <= MAX_JSON_BYTES, "source size limit exceeded")
        result[name] = sha256(path)
    return result


def _preferred_present_or_link(root, relative):
    path = root
    for part in PurePosixPath(relative).parts:
        path /= part
        if path.is_symlink():
            return True
    return path.exists()


def historical_adapter_file(records_dir, source_root):
    source_root = checked_directory(source_root)
    if _preferred_present_or_link(source_root, HISTORICAL_ADAPTER_PATH):
        return checked_file(source_root, HISTORICAL_ADAPTER_PATH)
    return checked_file(checked_directory(records_dir).parent,
                        "dynamics-recorded-source/resource_estimation.py")


def source_reproduction_evidence_file(records_dir, source_root):
    source_root = checked_directory(source_root)
    if _preferred_present_or_link(source_root, SOURCE_REPRODUCTION_EVIDENCE_PATH):
        return checked_file(source_root, SOURCE_REPRODUCTION_EVIDENCE_PATH)
    return checked_file(checked_directory(records_dir).parent,
                        "dynamics-replay-evidence/linux-source-reproduction.json")


def native_source_hashes(records_dir, source_root):
    archived = {"src/openquantum/dynamics_resources.py": "dynamics_resources.py",
                "scripts/generate_dynamics_resources.py": "generate_dynamics_resources.py"}
    result = {}
    for name in NATIVE_SOURCE_FILES:
        if name in archived: path = checked_file(records_dir, archived[name])
        elif name == "src/openquantum/resource_estimation.py": path = historical_adapter_file(records_dir, source_root)
        else: path = checked_file(source_root, name)
        require(path.stat().st_size <= MAX_JSON_BYTES, "native source size limit exceeded")
        result[name] = sha256(path)
    return result


def frozen_hashes(records_dir, source_root):
    result = {}
    for name in FROZEN_FILES:
        if name.startswith("assets/generated/") and not _preferred_present_or_link(Path(source_root), name):
            path = checked_file(checked_directory(records_dir).parent, name.removeprefix("assets/generated/"))
        else: path = checked_file(source_root, name)
        require(path.stat().st_size <= MAX_JSON_BYTES, "frozen input size limit exceeded")
        result[name] = sha256(path)
    return result


def _hash_map(value, names, label):
    require(isinstance(value, dict) and set(value) == set(names), f"invalid {label} keys")
    require(all(isinstance(digest, str) and SHA256.fullmatch(digest) for digest in value.values()),
            f"invalid {label} hashes")


def authenticate_record_inputs(records_dir, source_root, manifest):
    """Authenticate v2 record/source/evidence snapshots without executing code.

    The public gate authenticates its sidecar identity before calling this.
    The freezer uses this same complete input check for its in-memory candidate.
    """
    records_dir, source_root = checked_directory(records_dir), checked_directory(source_root)
    require(isinstance(manifest, dict) and len(canonical(manifest)) <= MAX_JSON_BYTES,
            "invalid/oversized record manifest")
    require(set(manifest) == {"schema", "status", "files", "study_manifest_sha256", "native_provenance",
            "native_evidence", "replay_sources", "frozen_files", "historical_adapter", "verifier_sources",
            "physical_validation", "contract", "source_reproduction_evidence"},
            "manifest has missing/unexpected fields")
    require(manifest["schema"] == MANIFEST_SCHEMA
            and manifest["status"] == "frozen_conditional_dynamics_record", "incomplete/unsupported manifest")
    before = core_inventory(records_dir)
    require(isinstance(manifest["files"], dict) and set(manifest["files"]) == CORE_FILES,
            "record inventory has missing/unexpected files")
    for entry in manifest["files"].values():
        require(isinstance(entry, dict) and set(entry) == {"sha256", "bytes"}
                and isinstance(entry["sha256"], str) and SHA256.fullmatch(entry["sha256"])
                and type(entry["bytes"]) is int and entry["bytes"] > 0, "invalid record inventory entry")
    require(canonical(manifest["files"]) == canonical(before), "record inventory differs from reviewed manifest")
    require(isinstance(manifest["study_manifest_sha256"], str)
            and SHA256.fullmatch(manifest["study_manifest_sha256"])
            and manifest["study_manifest_sha256"] == before["manifest.json"]["sha256"],
            "study manifest identity differs")

    # Strict parsing is part of authentication, before any dynamic code load.
    documents = {name: load_json(checked_file(records_dir, name), expected_sha256=before[name]["sha256"])
                 for name in sorted(CORE_FILES) if name.endswith(".json")}
    study = documents["manifest.json"]
    require(type(study.get("schema_version")) is int and study["schema_version"] == 1
            and study.get("status") == "computed_conditional_model", "incomplete original study")
    for name, count in (("canonical_cases", 24), ("actual_physical_queries", 48),
                        ("product_formula_cases", 18), ("QSP_OAA_cases", 4), ("QSP_retargeted_size_cases", 2)):
        require(type(study.get(name)) is int and study[name] == count, "original study coverage differs")
    require(canonical(study.get("profiles")) == canonical(list(PROFILES))
            and canonical(study.get("common_data_and_factory_distance_domain")) == canonical(list(DISTANCE_DOMAIN)),
            "original profiles/distance domain differ")
    require(study.get("frozen_files_unchanged") is True, "original study lacks frozen baseline check")
    require(manifest["contract"] == CONTRACT, "record verification contract differs")
    require(canonical(manifest["native_evidence"]) == canonical(NATIVE_EVIDENCE),
            "legacy evidence cannot assert an external completion receipt or wall/RSS measurement")
    require(isinstance(manifest["physical_validation"], dict), "invalid physical validation summary")
    evidence_binding = manifest["source_reproduction_evidence"]
    require(isinstance(evidence_binding, dict) and set(evidence_binding) == {"path", "sha256", "bytes"}
            and evidence_binding["path"] == SOURCE_REPRODUCTION_EVIDENCE_PATH
            and isinstance(evidence_binding["sha256"], str) and SHA256.fullmatch(evidence_binding["sha256"])
            and type(evidence_binding["bytes"]) is int and 0 < evidence_binding["bytes"] <= MAX_JSON_BYTES,
            "invalid source-reproduction evidence binding")
    evidence_path = source_reproduction_evidence_file(records_dir, source_root)
    evidence_bytes = read_bounded(evidence_path)
    require(len(evidence_bytes) == evidence_binding["bytes"]
            and hashlib.sha256(evidence_bytes).hexdigest() == evidence_binding["sha256"],
            "source-reproduction evidence snapshot differs")
    source_report = parse_json_bytes(evidence_bytes)
    require(set(source_report) == SOURCE_REPORT_FIELDS
            and type(source_report.get("schema_version")) is int and source_report["schema_version"] == 1
            and source_report.get("status") == "source_not_byte_reproduced"
            and canonical(source_report.get("claims")) == canonical(SOURCE_REPORT_CLAIMS),
            "source-reproduction evidence status/schema differs")

    provenance = study.get("provenance")
    require(isinstance(provenance, dict) and set(provenance) == {
        "source_commit", "source_revision_scope", "source_sha256", "environment"}, "invalid Native provenance")
    require(provenance["source_commit"] == HISTORICAL_SOURCE_COMMIT
            and provenance["source_revision_scope"] ==
            "Exact checkout root when Git is available; null for standalone packages. Content hashes identify actual source.",
            "Native provenance source/scope differs")
    require(canonical(provenance["environment"]) == canonical(NATIVE_ENVIRONMENT),
            "Native provenance environment differs")
    require(canonical(manifest["native_provenance"]) == canonical(provenance), "Native provenance was rebound")
    _hash_map(provenance["source_sha256"], NATIVE_SOURCE_FILES, "Native source")
    native = native_source_hashes(records_dir, source_root)
    require(provenance["source_sha256"] == native, "historical Native source bytes differ")
    adapter = manifest["historical_adapter"]
    require(isinstance(adapter, dict) and set(adapter) == {"path", "sha256"}
            and adapter["path"] == HISTORICAL_ADAPTER_PATH and adapter["sha256"] == HISTORICAL_ADAPTER_SHA256
            and native["src/openquantum/resource_estimation.py"] == HISTORICAL_ADAPTER_SHA256,
            "historical Native adapter binding differs")

    _hash_map(manifest["replay_sources"], REPLAY_SOURCE_FILES, "replay source")
    replay = source_hashes(source_root, REPLAY_SOURCE_FILES)
    require(manifest["replay_sources"] == replay, "current replay sources differ")
    _hash_map(manifest["verifier_sources"], VERIFIER_FILES, "verifier source")
    verifiers = source_hashes(source_root, VERIFIER_FILES)
    require(manifest["verifier_sources"] == verifiers, "verifier/model sources differ")
    _hash_map(manifest["frozen_files"], FROZEN_FILES, "frozen baseline")
    frozen = frozen_hashes(records_dir, source_root)
    require(canonical(manifest["frozen_files"]) == canonical(frozen)
            and canonical(study.get("frozen_file_sha256_before")) == canonical(frozen)
            and canonical(study.get("frozen_file_sha256_after")) == canonical(frozen),
            "frozen baseline identities differ")
    require(sha256(checked_file(records_dir, "LICENSE.txt")) == replay["LICENSE"],
            "archived license differs from published license")

    cases = study.get("cases")
    require(isinstance(cases, list) and len(cases) == 24
            and all(isinstance(case, dict) for case in cases)
            and [case.get("case_id") for case in cases] == list(CASE_IDS), "original case order/coverage differs")
    filenames = [profile["name"] + ".json" for profile in PROFILES]
    canonical_qasm_bytes = {}
    for case in cases:
        case_id = case["case_id"]
        ledger = documents[f"{case_id}/logical-resources.json"]
        require(canonical(ledger) == canonical(case), "case ledger differs from original study index")
        require(canonical(case.get("provenance")) == canonical(provenance), "case Native provenance differs")
        require(case.get("physical_reports") == filenames, "case report coverage differs")
        qasm_hash = before[f"{case_id}/measured-trial.qasm"]["sha256"]
        require(case.get("qasm_sha256") == qasm_hash, "case QASM binding differs")
        qasm_bytes = read_bounded(checked_file(records_dir, f"{case_id}/measured-trial.qasm"))
        require(hashlib.sha256(qasm_bytes).hexdigest() == qasm_hash
                and len(qasm_bytes) == before[f"{case_id}/measured-trial.qasm"]["bytes"],
                "canonical QASM snapshot differs")
        canonical_qasm_bytes[case_id] = qasm_bytes
        for filename, profile in zip(filenames, PROFILES):
            report = documents[f"{case_id}/{filename}"]
            require(report.get("status") == "computed_conditional_model"
                    and report.get("qasm_sha256") == qasm_hash
                    and isinstance(report.get("qasm3"), str)
                    and hashlib.sha256(report["qasm3"].encode()).hexdigest() == qasm_hash,
                    "physical report QASM binding differs")
            require(canonical(report.get("physical_profile")) == canonical(profile)
                    and canonical(report.get("dynamics_task")) == canonical(case.get("task"))
                    and canonical(report.get("postselection_contract")) == canonical(case.get("postselection_contract")),
                    "physical report profile/task binding differs")

    return {"records_dir": records_dir, "source_root": source_root, "before": before,
            "study": study, "cases": cases, "native": native, "replay": replay,
            "verifiers": verifiers, "frozen": frozen, "canonical_qasm_bytes": canonical_qasm_bytes,
            "evidence_path": evidence_path, "evidence_bytes": evidence_bytes,
            "source_report": source_report, "evidence_binding": dict(evidence_binding)}


def recheck_authenticated_inputs(inputs):
    """Recheck each authenticated role, including evidence, on success/error."""
    records_dir, source_root = inputs["records_dir"], inputs["source_root"]
    require(core_inventory(records_dir) == inputs["before"], "immutable records changed during verification")
    require(source_hashes(source_root, REPLAY_SOURCE_FILES) == inputs["replay"]
            and source_hashes(source_root, VERIFIER_FILES) == inputs["verifiers"]
            and native_source_hashes(records_dir, source_root) == inputs["native"]
            and frozen_hashes(records_dir, source_root) == inputs["frozen"],
            "authenticated sources/baseline changed during verification")
    require(read_bounded(source_reproduction_evidence_file(records_dir, source_root)) == inputs["evidence_bytes"],
            "source-reproduction evidence changed during verification")


def validate_source_reproduction(inputs):
    """Use only the authenticated stdlib transport helper; never quantum code."""
    name = "scripts/dynamics_source_transport.py"
    path = checked_file(inputs["source_root"], name)
    code = read_bounded(path)
    require(hashlib.sha256(code).hexdigest() == inputs["verifiers"][name],
            "source transport checker changed before loading")
    module = types.ModuleType("_dynamics_release_source_transport")
    module.__file__ = str(path)
    exec(compile(code, str(path), "exec"), module.__dict__)
    normalized = [module.normalize_canonical_case(json.loads(canonical(case)), dict(inputs["native"]),
                                                   json.loads(canonical(NATIVE_ENVIRONMENT)))
                  for case in inputs["cases"]]
    summary = module.validate_source_transport(json.loads(canonical(inputs["source_report"])), normalized,
                                              inputs["canonical_qasm_bytes"],
                                              expected_source_hashes=dict(inputs["replay"]))
    require(isinstance(summary, dict) and summary.get("status") == "validated_static_source_transport_report"
            and summary.get("source_status") == "source_not_byte_reproduced"
            and summary.get("canonical_circuit_validation_performed") is False
            and summary.get("costs_repriced") is False,
            "source transport did not validate the preserved non-reproduction report")
    canonical(summary)
    return summary


def verify_dynamics_record_identity(records_dir, source_root, *, expected_sha256=None):
    """Authenticate immutable v2 inputs before checked stdlib-only validation."""
    records_dir, source_root = checked_directory(records_dir), checked_directory(source_root)
    expected = DYNAMICS_RECORD_MANIFEST_SHA256 if expected_sha256 is None else expected_sha256
    require(isinstance(expected, str) and SHA256.fullmatch(expected), "reviewed manifest SHA256 required")
    manifest_path = checked_file(records_dir, MANIFEST_NAME)
    manifest = load_json(manifest_path, expected_sha256=expected)
    inputs = authenticate_record_inputs(records_dir, source_root, manifest)

    # Checked bytes, rather than an import cache or a .pyc, authorize this load.
    # No producer, estimator, optional fitter or scientific package is imported.
    try:
        source_reproduction = validate_source_reproduction(inputs)
        path = checked_file(source_root, "scripts/dynamics_physical_records.py")
        model_bytes = read_bounded(path)
        require(hashlib.sha256(model_bytes).hexdigest() == inputs["verifiers"]["scripts/dynamics_physical_records.py"],
                "model checker changed before loading")
        model = types.ModuleType("_dynamics_release_physical_records")
        model.__file__ = str(path)
        exec(compile(model_bytes, str(path), "exec"), model.__dict__)
        physical = model.validate_physical_records(records_dir, inputs["study"])
        require(canonical(physical) == canonical(manifest["physical_validation"]),
                "frozen physical validation summary differs")
    finally:
        recheck_authenticated_inputs(inputs)
        require(sha256(manifest_path) == expected, "immutable record sidecar changed during verification")
    return {"status": "verified immutable dynamics publication record", "core_files": len(inputs["before"]),
            "manifest_sha256": expected, "native_source_commit": HISTORICAL_SOURCE_COMMIT,
            "native_reestimation_performed": False, "factory_search_repeated": False,
            "logical_producer_executed": False, "external_completion_receipt_available": False,
            "physical_validation": physical, "source_reproduction_validation": source_reproduction,
            "canonical_quantum_validation_performed": False, "source_byte_equivalence_claimed": False,
            "physical_records_repriced": False}


def site_dynamics_record_errors(site_root):
    try:
        root = checked_directory(site_root)
        verify_dynamics_record_identity(root / RECORD_DIRECTORY, root / "code")
    except Exception as error:
        return [str(error)]
    return []
