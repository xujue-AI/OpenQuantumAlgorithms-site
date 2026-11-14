#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Display authenticated saved metadata for one fixed finite composition.

Only manifest.json and summary.json are read. This display does not inspect
the scientific payloads, parse circuits, evaluate formulas or replay a check.
It makes no whole-invocation, full-bundle or scientific certification claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat

BUNDLE_RELATIVE = "assets/generated/simulation/detailed-balance-two-round"
MANIFEST_SCHEMA = "openquantum.fixed-two-round.public-manifest.v1"
MANIFEST_SHA256 = "e2f3c836fa05c6a0ae6be5daa18684a6b9ec96c6033f5bd34420a2413fe54377"
MANIFEST_BYTES = 3497
MAX_MANIFEST_BYTES = 65_536
MAX_SUMMARY_BYTES = 16_384
MAX_DISPLAY_BYTES = 16_384
SUMMARY_SCHEMA = "openquantum.fixed-two-round.public-summary.v1"
SUMMARY_STATUS = "parent_accepted_fixed_finite_composition_only"
SCOPE = "fixed one-system two-round finite composition; no Gibbs/mixing/physical/formal guarantee"
SUMMARY_KEYS = (
    "schema", "status", "scope", "parameters", "flags", "checks",
    "observed_calls", "internal_reset_dispatches", "diagnostics",
    "source_accounting", "provenance",
)
FALSE_FLAGS = (
    "whole_invocation_validated", "all_components_verified", "new_algorithm_result",
    "channel_implemented", "coherent_correction_implemented",
    "coherent_correction_evolution_implemented", "general_channel_implemented",
    "general_coherent_correction_implemented", "exact_finite_kms",
    "finite_mesh_exact_kms", "mixing_certificate", "rotations_synthesized",
    "physical_resources_computed", "formal_proof_reproduced",
    "floating_point_error_certified", "hardware_error_certified",
    "random_sampling_performed", "sampler_implemented",
)
EXPECTED_PARAMETERS = {
    "beta": 1.0, "clock_bits": 2, "delta": 0.125,
    "initial_state": "known_plus", "product_steps": 1, "proposals": ["Z"],
    "readout": "Z only after two all-zero five-bit heralds", "rounds": 2,
    "tau": 0.125, "terms": [[0.6, "Z"], [0.8, "X"]],
}
EXPECTED_CALLS = {
    "weak_statevectors": 2, "external_density_stages": 4, "tiny_expm": 8,
    "interval_powers": 2, "generator_expm": 1,
}
EXPECTED_COUNTS = {
    "reserved_quantum_wires": 6, "reserved_classical_bits": 11,
    "program_static_primitive_instances": 6051,
    "program_maximum_executed_primitives": 6049,
    "program_early_abort_primitives": 3026,
}
DIAGNOSTIC_NAMES = (
    "weak_kraus_max_error", "terminal_completeness_error", "final_state_error",
    "two_success_probability", "z_bias_against_finite_target",
    "correction_only_bound_4b", "gibbs_frobenius_distance",
    "gibbs_generator_residual",
)
EXPECTED_PROVENANCE = {
    "accepted_source_commit": "02c6bce80784b249927223b08b39174396d6e184",
    "raw_result_sha256": "a10133d55c305f526c570f35638d3b9ac9cea3079d65f8053af593a790ccea24",
    "raw_result_bytes": 1186965,
}


def _closed(value, keys, label):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(label + " has an unexpected metadata shape")


def _literal_mapping(value, expected, label):
    _closed(value, expected, label)
    if any(type(value[key]) is not type(item) or value[key] != item
           for key, item in expected.items()):
        raise ValueError(label + " differs from the fixed metadata contract")


def validate_summary(value):
    """Check closed metadata types and scope without computing diagnostics."""
    _closed(value, SUMMARY_KEYS, "summary")
    for key, expected in (("schema", SUMMARY_SCHEMA), ("status", SUMMARY_STATUS),
                          ("scope", SCOPE)):
        if type(value[key]) is not str or value[key] != expected:
            raise ValueError("summary identity or scope differs")
    parameters = value["parameters"]
    _closed(parameters, EXPECTED_PARAMETERS, "parameters")
    for key in ("beta", "delta", "tau", "clock_bits", "product_steps", "rounds",
                "initial_state", "readout"):
        expected = EXPECTED_PARAMETERS[key]
        if type(parameters[key]) is not type(expected) or parameters[key] != expected:
            raise ValueError("parameters differ from the fixed toy")
    terms = parameters["terms"]
    if (type(terms) is not list or len(terms) != 2
            or any(type(pair) is not list or len(pair) != 2
                   or type(pair[0]) is not float or type(pair[1]) is not str
                   for pair in terms)
            or terms != EXPECTED_PARAMETERS["terms"]
            or type(parameters["proposals"]) is not list
            or parameters["proposals"] != ["Z"]
            or type(parameters["proposals"][0]) is not str):
        raise ValueError("terms or proposal order differ from the fixed toy")
    flags = {key: False for key in FALSE_FLAGS}
    flags.update({"successful_completion_claimed": True, "finite_toy_checks_passed": True})
    _literal_mapping(value["flags"], flags, "raw flags")
    _literal_mapping(value["checks"], {"passed": 66, "total": 66}, "saved check counts")
    _literal_mapping(value["observed_calls"], EXPECTED_CALLS, "observed calls")
    _literal_mapping(value["internal_reset_dispatches"], {"predicted": 16, "measured": False},
                     "internal reset dispatches")
    _literal_mapping(value["source_accounting"], EXPECTED_COUNTS, "source accounting")
    _literal_mapping(value["provenance"], EXPECTED_PROVENANCE, "provenance")
    _closed(value["diagnostics"], DIAGNOSTIC_NAMES, "saved diagnostics")
    if any(type(item) is not float or not math.isfinite(item)
           for item in value["diagnostics"].values()):
        raise ValueError("saved diagnostic fields must be finite floating metadata")
    return value


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _hex_digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _read_file(directory, name, maximum):
    if name not in ("manifest.json", "summary.json"):
        raise ValueError("only the two saved metadata files may be read")
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not 0 < before.st_size <= maximum):
        raise ValueError("saved metadata must be bounded unlinked regular files")
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=directory)
    try:
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise ValueError("saved metadata changed before read")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        after = os.stat(name, dir_fd=directory, follow_symlinks=False)
        if (len(raw) != before.st_size or len(raw) > maximum
                or _identity(os.fstat(descriptor)) != _identity(before)
                or _identity(after) != _identity(before)):
            raise ValueError("saved metadata changed during read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate saved metadata key")
            result[key] = value
        return result

    def bad_constant(_value):
        raise ValueError("nonfinite saved metadata constant")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=bad_constant)


def verify_metadata(bundle):
    """Authenticate the manifest and selected summary; read no science data."""
    if (not _hex_digest(MANIFEST_SHA256)
            or type(MANIFEST_BYTES) is not int
            or not 0 < MANIFEST_BYTES <= MAX_MANIFEST_BYTES):
        raise ValueError("the public metadata identity is not frozen")
    root = Path(bundle).absolute()
    if root != root.resolve():
        raise ValueError("saved metadata directory must have an unaliased path")
    before = root.lstat()
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("saved metadata directory must be a regular directory")
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if _identity(os.fstat(directory)) != _identity(before):
            raise ValueError("saved metadata directory changed before open")
        manifest_raw = _read_file(directory, "manifest.json", MAX_MANIFEST_BYTES)
        if (len(manifest_raw) != MANIFEST_BYTES
                or hashlib.sha256(manifest_raw).hexdigest() != MANIFEST_SHA256):
            raise ValueError("public metadata manifest identity differs")
        manifest = _json(manifest_raw)
        if (type(manifest) is not dict or manifest.get("schema") != MANIFEST_SCHEMA
                or type(manifest.get("files")) is not dict):
            raise ValueError("public metadata manifest schema differs")
        binding = manifest["files"].get("summary.json")
        _closed(binding, ("bytes", "sha256"), "summary binding")
        if (type(binding["bytes"]) is not int
                or not 0 < binding["bytes"] <= MAX_SUMMARY_BYTES
                or not _hex_digest(binding["sha256"])):
            raise ValueError("summary binding differs from the bounded metadata contract")
        summary_raw = _read_file(directory, "summary.json", MAX_SUMMARY_BYTES)
        if (len(summary_raw) != binding["bytes"]
                or hashlib.sha256(summary_raw).hexdigest() != binding["sha256"]):
            raise ValueError("saved summary identity differs")
        summary = validate_summary(_json(summary_raw))
        if (_identity(root.lstat()) != _identity(before)
                or _identity(os.fstat(directory)) != _identity(before)):
            raise ValueError("saved metadata directory changed during read")
        return {
            "status": "saved_thermal_metadata_display",
            "metadata_identity_verified": True,
            "metadata_files_read": ["manifest.json", "summary.json"],
            "full_bundle_verified": False,
            "numerical_replay_performed": False,
            "summary": summary,
        }
    finally:
        os.close(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path,
                        default=Path(__file__).resolve().parents[1] / BUNDLE_RELATIVE)
    args = parser.parse_args()
    try:
        report = verify_metadata(args.bundle)
        raw = json.dumps(report, sort_keys=True, allow_nan=False)
        if len((raw + "\n").encode("utf-8")) > MAX_DISPLAY_BYTES:
            raise ValueError("saved metadata display exceeds its output bound")
        print(raw)
        return 0
    except (OSError, ValueError, TypeError, RecursionError) as error:
        print(json.dumps({"status": "saved_thermal_metadata_rejected",
                          "error_type": type(error).__name__,
                          "metadata_identity_verified": False,
                          "full_bundle_verified": False,
                          "numerical_replay_performed": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
