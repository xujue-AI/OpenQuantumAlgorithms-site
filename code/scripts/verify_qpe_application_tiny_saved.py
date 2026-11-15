#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate retained tiny-QPE downloads without importing scientific code.

The gate reads bounded saved files, checks their exact byte identities and
joins the figure's literal labels to the recorded QASM. It does not run the
checker, evaluate a circuit, recompute probabilities or validate a new recipe.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_SHA256 = "62fafef64e565b3549dfc505ab975611f43730b11d8ab040406bee631eb7e4e3"
PROJECTION_SHA256 = "9b2af033a7d68ae0c06e28bea70b9a5ed8380aa6310fe1d2c7938caf0c9e2e65"
FIGURE_SHA256 = "2a482b68c0dcb68a5ff4ec84a373af5800163cd3ba171689a1fd4d1e34270a81"
CHECKS_SHA256 = "584640ed31039646b0764d5f325129f31371d7877b40915b73ff686bfbcca0b3"
SOURCE_SHA256 = {
    "scripts/check_qpe_application_tiny.py": "483426110ad29c780d7f9b0ee6e6457503cbd1fab68d0b4a83291501f072e8f8",
    "src/openquantum/application_qpe_tiny.py": "ad5c3dff8fbaf58d0f9e8cad06f809920bbcc66e94a01d18bd09aec5881311b6",
    "src/openquantum/application_qpe_tiny_reference.py": "f52648f90d5b8c030eb35424472381852ae118a9fb2993ee0257886919d9b50e",
}
RAW_PAYLOAD_NAMES = (
    "kernel.qasm", "kernel-without-reference-z.qasm", "kernel-without-odd-evolution.qasm",
    "retained-deferred.qasm", "retained-reused.qasm", "retained-reprepared.qasm", "feedback.qasm",
    "fringe-k1-cos.qasm", "fringe-k1-sin.qasm", "fringe-k2-cos.qasm", "fringe-k2-sin.qasm",
    "fringe-k4-cos.qasm", "fringe-k4-sin.qasm", "checks.json",
)
SUPPLEMENTAL_ROLES = {
    "README.md": "public-scope-disclosure", "LICENSE.txt": "MIT-license",
    "requirements-lock.txt": "locked-dependencies",
}
PAYLOAD_NAMES = frozenset(RAW_PAYLOAD_NAMES) | frozenset(SOURCE_SHA256) | frozenset(SUPPLEMENTAL_ROLES)
PACKET_NAMES = PAYLOAD_NAMES | {"manifest.json", "public-projection.json"}
PACKET_DIRECTORIES = frozenset({"scripts", "src", "src/openquantum"})
MAX_MEMBER_BYTES = 64 * 1024
MAX_PACKET_BYTES = 1024 * 1024
MAX_FIGURE_BYTES = 64 * 1024
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
FALSE_CLAIMS = (
    "broader_protocol_validated", "classical_inference_performed", "complete_end_to_end_resource_analysis",
    "formal_proof_reproduced", "full_application_confidence_validated", "full_application_output_simulated",
    "hardware_execution_performed", "interval_synthesis_certificate", "physical_estimation_performed",
    "rotation_synthesis_performed", "shot_sampling_performed", "system_size_scaling_validated",
    "universal_algorithm_correctness", "zero_final_record_behavior_exercised",
)
RAW_FALSE_CLAIMS = (
    "classical_inference_performed", "formal_proof_reproduced", "full_application_confidence_validated",
    "full_application_output_simulated", "interval_synthesis_certificate", "physical_estimation_performed",
    "shot_sampling_performed",
)
FIXED_TASK = {
    "Eref": "3", "Hamiltonian": "-Z0-Z1+X0*X1/4", "U": "-exp(-i H pi/3)",
    "W": "6", "input": "reset |00>", "tau0": "pi/3",
}
FIXED_SCOPE = {
    "admitted_controlled_powers": [1, 2, 4], "base_kernel_primitive_instructions": 39,
    "expanded_validation_instructions_per_case_maximum": 10000, "fixed_task": FIXED_TASK,
    "maximum_classical_bits": 4, "maximum_quantum_wires": 6, "maximum_retained_records": 16,
    "phase_word_bits": 2, "retained_rounds": 2, "saved_check_case_count": 15,
    "saved_check_record_scope": "tiny-ideal-circuit-instrument", "saved_check_record_status": "passed",
    "saved_numerical_entry_tolerance": 2e-12,
}
CASE_NAMES = (
    "full-controlled-k1", "full-controlled-k2", "full-controlled-k4",
    "negative-missing-reference-z", "negative-missing-odd-evolution", "retained-deferred",
    "retained-reused", "feedback", "retained-reprepared", "fringe-k1-cos", "fringe-k1-sin",
    "fringe-k2-cos", "fringe-k2-sin", "fringe-k4-cos", "fringe-k4-sin",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("Nonfinite JSON scalar: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def _fixed(actual, expected, label):
    _require(type(actual) is type(expected), label + " type differs")
    if isinstance(expected, dict):
        _require(set(actual) == set(expected), label + " fields differ")
        for key, value in expected.items():
            _fixed(actual[key], value, label + "." + key)
    elif isinstance(expected, list):
        _require(len(actual) == len(expected), label + " length differs")
        for value, wanted in zip(actual, expected, strict=True):
            _fixed(value, wanted, label)
    else:
        _require(actual == expected, label + " value differs")


def _name(name):
    _require(type(name) is str and name and "\\" not in name, "Invalid member name")
    path = PurePosixPath(name)
    _require(not path.is_absolute() and str(path) == name
             and all(part not in {".", "..", ""} for part in path.parts), "Noncanonical member name")
    return path


def _stamp(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _parents(path):
    snapshots = []
    for parent in path.parents:
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode), "Member ancestor is not an ordinary directory")
        snapshots.append((parent, info.st_dev, info.st_ino))
    return snapshots


def _read(path, check, *, maximum=MAX_MEMBER_BYTES, expected_sha256=None):
    check()
    parents = _parents(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
             and before.st_size <= maximum, "Member is not a bounded single-link ordinary file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        _require(_stamp(before) == _stamp(os.fstat(descriptor)), "Member changed at open")
        raw = b""
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(8192, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw += chunk
            check()
        _require(len(raw) == before.st_size <= maximum, "Member byte bound differs")
        _require(_stamp(before) == _stamp(os.fstat(descriptor)) == _stamp(path.lstat()),
                 "Member changed during read")
    finally:
        os.close(descriptor)
    for parent, device, inode in parents:
        info = parent.lstat()
        _require(stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == (device, inode),
                 "Member ancestor changed during read")
    if expected_sha256 is not None:
        _require(hashlib.sha256(raw).hexdigest() == expected_sha256, "Member SHA-256 differs: " + path.name)
    check()
    return raw


def _inventory(root, check):
    _require(root.is_absolute() and ".." not in root.parts, "Packet root must be canonical")
    _require(stat.S_ISDIR(root.lstat().st_mode), "Packet root is not an ordinary directory")
    _parents(root)
    files, directories, total = set(), set(), 0
    pending = [root]
    while pending:
        check()
        current = pending.pop()
        _require(stat.S_ISDIR(current.lstat().st_mode), "Packet directory changed")
        with os.scandir(current) as entries:
            for entry in entries:
                check()
                path = Path(entry.path)
                name = str(path.relative_to(root).as_posix())
                _name(name)
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    _require(name in PACKET_DIRECTORIES, "Unexpected packet directory: " + name)
                    directories.add(name)
                    pending.append(path)
                else:
                    _require(name in PACKET_NAMES, "Unexpected packet member: " + name)
                    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                             and info.st_size <= MAX_MEMBER_BYTES, "Invalid ordinary packet member: " + name)
                    files.add(name)
                    total += info.st_size
                    _require(total <= MAX_PACKET_BYTES, "Packet aggregate byte bound exceeded")
                _require(len(files) + len(directories) <= len(PACKET_NAMES) + len(PACKET_DIRECTORIES),
                         "Packet entry bound exceeded")
    _require(files == PACKET_NAMES and directories == PACKET_DIRECTORIES, "Packet membership is incomplete")
    return total


def _record_scope(manifest, projection, checks):
    _require(manifest.get("schema") == "openquantum.qpe_application_tiny.public_manifest.v1"
             and projection.get("schema") == "openquantum.qpe_application_tiny.public_projection.v1",
             "Unexpected QPE projection schema")
    for record in (manifest, projection):
        _fixed(record.get("claims"), {name: False for name in FALSE_CLAIMS}, "Projection claims")
        _fixed(record.get("scope"), FIXED_SCOPE, "Fixed tiny scope")
        _require(record.get("status") == "static-public-projection-proposal"
                 and record.get("new_scientific_work_during_projection") is False,
                 "Projection authoring scope changed")
    _require(manifest.get("source_authoring_labels_preserved") is True
             and manifest.get("manual_reproduction_recipe_documented") is True
             and manifest.get("reproduction_recipe_executed_during_projection") is False,
             "Preserved labels or recipe execution scope changed")
    recipe = projection.get("manual_reproduction_recipe", {})
    _require(recipe.get("executed_during_projection") is False and recipe.get("uses_exact_checker") is True
             and recipe.get("private_runtime_controls_included") is False
             and recipe.get("python_version") == "3.12.14", "Manual recipe scope changed")
    _require(projection.get("exact_raw_payload_copies") is True
             and projection.get("exact_used_scientific_source_copies") is True, "Exact copy scope changed")
    disclosures = projection.get("disclosures", {})
    _require(type(disclosures) is dict and set(disclosures) == {
        "broader_protocol", "checker_source", "stale_raw_source_scope", "stale_source_headers",
        "validation_map_count_semantics", "zero_final_record_limitation"}, "Required disclosures differ")
    _require("did not exercise a zero-probability final record" in disclosures["zero_final_record_limitation"]
             and "private/unexecuted" in disclosures["stale_source_headers"], "Historical limitations differ")
    _require(checks.get("status") == "passed" and checks.get("scope") == "tiny-ideal-circuit-instrument"
             and type(checks.get("case_count")) is int and checks["case_count"] == 15,
             "Saved tiny check scope/status differs")
    _fixed(checks.get("frozen_task"), FIXED_TASK, "Saved fixed task")
    _fixed(checks.get("limits"), {"controlled_power_max": 4,
        "expanded_validation_instructions_per_case_max": 10000, "quantum_wires_max": 6,
        "retained_records_max": 16}, "Saved deterministic limits")
    _fixed(checks.get("numerical_entry_tolerance"), 2e-12, "Saved numerical tolerance")
    for name in RAW_FALSE_CLAIMS:
        _require(checks.get(name) is False, "Saved broader claim changed: " + name)
    cases = checks.get("cases")
    _require(type(cases) is list and len(cases) == 15
             and tuple(case.get("case") for case in cases) == CASE_NAMES, "Saved case inventory differs")
    _require(all(case.get("source", {}).get("scope") ==
        "unexecuted bounded source; conditional corrections counted in common upper envelope"
        for case in cases), "Raw authoring labels changed")


def _figure_join(raw, kernel, kernel_sha256):
    _require(b"<!DOCTYPE" not in raw and b"<!ENTITY" not in raw, "Figure entities are not permitted")
    svg = ET.fromstring(raw)
    namespace = "{http://www.w3.org/2000/svg}"
    _require(svg.tag == namespace + "svg", "External figure is not SVG")
    _require(not any(node.tag == namespace + "script" for node in svg.iter()), "Active figure content differs")
    metadata = svg.findall(namespace + "metadata")
    _require(len(metadata) == 1 and metadata[0].text is not None, "Figure metadata differs")
    record = _json(metadata[0].text)
    _require(record.get("schema") == "openquantum.literal_circuit_figure.v1"
             and record.get("source_sha256") == kernel_sha256
             and record.get("source_file") == "../generated/qpe-application-tiny/kernel.qasm",
             "Figure source identity differs")
    lines = kernel.decode("ascii").splitlines()
    _require(lines[:3] == ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[3] q;"],
             "Kernel QASM header differs")
    operations = record.get("operations")
    _require(type(record.get("gate_count")) is int and record["gate_count"] == 39
             and type(operations) is list and len(operations) == 39
             and [op.get("raw") for op in operations] == lines[3:], "Figure literal order differs")
    _fixed(record.get("counts"), {"cx": 18, "ry": 8, "rz": 8, "h": 2, "x": 2, "z": 1},
           "Figure recorded counts")
    labeled = [node for node in svg.iter(namespace + "g")
               if (node.get("id") or "").startswith("instruction-")]
    groups = {node.get("id"): node for node in labeled}
    _require(len(labeled) == 39 and set(groups) == {f"instruction-{i}" for i in range(1, 40)},
             "Figure instruction groups differ")
    for index, op in enumerate(operations, 1):
        _require(type(op.get("number")) is int and op["number"] == index, "Figure ordinal differs")
        group = groups[f"instruction-{index}"]
        _require(group.get("data-operation") == op.get("name")
                 and group.get("data-wires") == ",".join(str(wire) for wire in op["wires"]),
                 "Figure gate/wire label differs")
        _require(group.get("data-recorded-angle") == op.get("angle"), "Figure signed angle label differs")
    _fixed(dict(Counter(op["name"] for op in operations)), record["counts"], "Figure literal counts")


def verify_saved_packet(bundle: Path, *, manifest_sha256: str, projection_sha256: str,
                        figure: Path, figure_sha256: str, max_seconds: int = 30):
    """Return a saved-evidence summary; perform no scientific computation."""
    for actual, expected, label in (
        (manifest_sha256, MANIFEST_SHA256, "manifest"),
        (projection_sha256, PROJECTION_SHA256, "projection"),
        (figure_sha256, FIGURE_SHA256, "figure"),
    ):
        _require(type(actual) is str and SHA256.fullmatch(actual) and actual == expected,
                 "An exact reviewed external " + label + " pin is required")
    _require(type(max_seconds) is int and 1 <= max_seconds <= 30, "Saved gate time must be 1..30 seconds")
    started = time.monotonic()

    def check():
        _require(time.monotonic() - started < max_seconds, "Saved packet time ceiling exceeded")

    bundle, figure = Path(bundle).absolute(), Path(figure).absolute()
    _require(".." not in figure.parts, "Figure path must be canonical")
    total = _inventory(bundle, check)
    manifest_raw = _read(bundle / "manifest.json", check, expected_sha256=manifest_sha256)
    projection_raw = _read(bundle / "public-projection.json", check, expected_sha256=projection_sha256)
    manifest, projection = _json(manifest_raw), _json(projection_raw)
    members = manifest.get("files")
    _require(type(members) is dict and set(members) == PAYLOAD_NAMES, "Closed manifest membership differs")
    _fixed(manifest.get("raw_payload_names"), list(RAW_PAYLOAD_NAMES), "Raw payload inventory")
    _fixed(manifest.get("exact_source_paths"), ["src/openquantum/application_qpe_tiny.py",
        "src/openquantum/application_qpe_tiny_reference.py", "scripts/check_qpe_application_tiny.py"],
        "Exact source inventory")
    data = {}
    for name, pin in sorted(members.items()):
        _name(name)
        _require(type(pin) is dict and set(pin) == {"bytes", "sha256", "role"}
                 and type(pin["bytes"]) is int and 0 <= pin["bytes"] <= MAX_MEMBER_BYTES
                 and type(pin["sha256"]) is str and SHA256.fullmatch(pin["sha256"]), "Invalid member pin")
        role = ("raw-tiny-payload" if name in RAW_PAYLOAD_NAMES else
                "exact-used-scientific-source" if name in SOURCE_SHA256 else SUPPLEMENTAL_ROLES[name])
        _require(pin["role"] == role, "Member role differs")
        raw = _read(bundle / name, check, expected_sha256=pin["sha256"])
        _require(len(raw) == pin["bytes"], "Member declared size differs")
        data[name] = raw
    _fixed(projection.get("manifest"), {"bytes": len(manifest_raw), "filename": "manifest.json",
        "sha256": manifest_sha256}, "Descriptor manifest join")
    for key, names in (("raw_payloads", RAW_PAYLOAD_NAMES), ("scientific_source", SOURCE_SHA256),
                       ("supplemental_files", SUPPLEMENTAL_ROLES)):
        _fixed(projection.get(key), {name: members[name] for name in names}, "Descriptor " + key)
    _require(members["checks.json"]["sha256"] == CHECKS_SHA256, "Accepted raw check identity differs")
    for name, sha256 in SOURCE_SHA256.items():
        _require(members[name]["sha256"] == sha256, "Exact scientific source identity differs")
    _record_scope(manifest, projection, _json(data["checks.json"]))
    figure_raw = _read(figure, check, maximum=MAX_FIGURE_BYTES, expected_sha256=figure_sha256)
    _figure_join(figure_raw, data["kernel.qasm"], members["kernel.qasm"]["sha256"])
    _require(_inventory(bundle, check) == total, "Packet inventory changed after authentication")
    check()
    return {
        "schema": "openquantum.qpe_application_tiny.saved_byte_gate.v1", "status": "saved-bytes-verified",
        "packet_files": 22, "manifest_payloads": 20, "raw_payloads": 14, "scientific_sources": 3,
        "packet_bytes": total, "figure_bytes": len(figure_raw), "figure_gate_groups": 39,
        "manifest_sha256": manifest_sha256, "projection_sha256": projection_sha256,
        "checks_sha256": CHECKS_SHA256, "figure_sha256": figure_sha256,
        "saved_check_status": "passed", "saved_check_scope": "tiny-ideal-circuit-instrument",
        "saved_check_case_count": 15, "source_authoring_labels_preserved": True,
        "zero_final_record_behavior_exercised": False, "recipe_executed_during_projection": False,
        "gate_imported_scientific_source": False, "gate_replayed_numerical_checks": False,
        "gate_executed_recipe": False, "claims": {name: False for name in FALSE_CLAIMS},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "assets/generated/qpe-application-tiny")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument("--figure", type=Path, default=ROOT / "assets/figures/qpe-application-tiny-kernel.svg")
    parser.add_argument("--figure-sha256", required=True)
    parser.add_argument("--max-seconds", type=int, default=30)
    args = parser.parse_args()
    result = verify_saved_packet(args.bundle, manifest_sha256=args.manifest_sha256,
        projection_sha256=args.projection_sha256, figure=args.figure,
        figure_sha256=args.figure_sha256, max_seconds=args.max_seconds)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
