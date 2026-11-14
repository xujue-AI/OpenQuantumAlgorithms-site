#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate the saved fixed correction toy without scientific replay.

This standard-library reader checks retained bytes, source correspondence,
literal primitive chronology and the scope of historical results. It imports
no producer or scientific library and adds no numerical or physical evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import time
import zipfile

BUNDLE_RELATIVE = "assets/generated/simulation/detailed-balance-correction"
MANIFEST_SHA256 = "93bf5c0047a05b7c009b590629ca9b6a9661e5dc46da97d510f9f3b621208929"
V2_MANIFEST_SHA256 = "1127038ca780bb63315e6f867dbe87e3da963cabc4c667a2aa6a62deac6be02f"
V2_RESULTS_SHA256 = "2e31bafecc802aab01568fc888d6d420c5b9683caea04c9d65ab75612305df9e"
V2_TOTAL_BYTES = 434352
BUNDLE_TOTAL_BYTES = 455665
MANIFEST_BYTES = 4324
MAX_FILE_BYTES = 1024 * 1024
MAX_JSON_BYTES = 256 * 1024
MAX_BUNDLE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 40
MAX_JSON_NODES = 100000
MAX_PATH_COMPONENTS = 128
MAX_WALL_SECONDS = 10.0
START_WALL = time.monotonic()
HELPER_NAME = "verify_detailed_balance_correction_saved.py"
ZIP_PREFIX = "detailed-balance-correction/"
SOURCE_BINDINGS = {
    "detailed_balance_access.py": "src/openquantum/simulation/detailed_balance_access.py",
    "detailed_balance_decay.py": "src/openquantum/simulation/detailed_balance_decay.py",
    "detailed_balance_channel.py": "src/openquantum/simulation/detailed_balance_channel.py",
    "detailed_balance_decay_reference.py": "src/openquantum/simulation/detailed_balance_decay_reference.py",
    "detailed_balance_correction.py": "src/openquantum/simulation/detailed_balance_correction.py",
    "detailed_balance_correction_reference.py": "src/openquantum/simulation/detailed_balance_correction_reference.py",
    "generate_detailed_balance_correction.py": "scripts/generate_detailed_balance_correction.py",
    "requirements-lock.txt": "requirements-lock.txt",
    "LICENSE.txt": "LICENSE",
}
SOURCE_KINDS = {
    "centered_decay": (4, 213, 0),
    "correction_block": (5, 228, 0),
    "controlled_correction": (6, 2013, 0),
    "heralded_step": (6, 2016, 0),
    "known_plus_prefix": (6, 2017, 0),
    "known_plus_trial": (6, 2029, 6),
}
V2_FILES = frozenset({*SOURCE_BINDINGS, "results.json", "manifest.json", "manifest.sha256",
                      *(name + ".qasm" for name in SOURCE_KINDS)})
BUNDLE_FILES = frozenset({*("v2/" + name for name in V2_FILES),
                          "v1/failure.json", "v1/centered_decay.qasm", "completion.json",
                          "README.md", "UPSTREAM-LICENSES.md", "artifact-manifest.json",
                          "artifact-manifest.sha256"})
FALSE_FIELDS = frozenset({
    "all_components_verified", "new_algorithm_result", "coherent_correction_implemented",
    "general_coherent_correction_implemented", "coherent_correction_evolution_implemented",
    "general_correction_evolution_implemented", "general_hamiltonian_correction_implemented",
    "general_channel_implemented", "channel_implemented", "sampler_implemented",
    "exact_finite_kms", "finite_mesh_exact_kms", "mixing_certificate",
    "rotations_synthesized", "physical_resources_computed", "formal_proof_reproduced",
    "random_sampling_performed", "sampling_performed", "floating_point_error_certified",
    "floating_point_bounds_certified", "hardware_error_certified",
    "synthesis_or_hardware_error_certified", "synthesis_or_hardware_error_included",
    "complete_descendant_lifetime_cleanup_proven", "unknown_input_recovery_implemented",
    "floating_point_or_hardware_certification", "numerical_acceptance_established",
    "runtime_counters_retained", "saved_results_and_manifests_present",
    "v1_numerical_acceptance_established", "original_adjacent_scientific_mode_validated",
    "saved_reader_is_numerical_replay", "surface_code_baseline_changed",
    "publication_performed_by_this_projection", "retry_performed",
})
PRIMITIVE_ARITY = {"h": (1, 0), "x": (1, 0), "s": (1, 0), "sdg": (1, 0),
                   "t": (1, 0), "tdg": (1, 0), "p": (1, 1), "ry": (1, 1),
                   "rz": (1, 1), "cx": (2, 0)}


def _deadline():
    if time.monotonic() - START_WALL > MAX_WALL_SECONDS:
        raise ValueError("Saved correction reader exceeded its ten-second deadline")


def _equal(actual, expected, label):
    if type(actual) is not type(expected) or actual != expected:
        raise ValueError("Unexpected correction " + label)
    if type(expected) is dict:
        for key, value in expected.items():
            _equal(actual[key], value, label)
    elif type(expected) in (list, tuple):
        for left, right in zip(actual, expected):
            _equal(left, right, label)


def _relative(name):
    if type(name) is not str:
        raise ValueError("Unsafe correction relative path")
    path = PurePosixPath(name)
    if (name in ("", ".", "..") or "\\" in name or any(ord(char) < 32 for char in name)
            or path.is_absolute() or ".." in path.parts or path.as_posix() != name
            or len(path.parts) > MAX_PATH_COMPONENTS):
        raise ValueError("Unsafe correction relative path")
    return path


def _root(value):
    root = Path(value).absolute()
    if (root != root.resolve() or root.is_symlink() or not root.is_dir()
            or len(root.parts) > MAX_PATH_COMPONENTS):
        raise ValueError("Correction root must be an unaliased directory")
    return root


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _directory_identity(info):
    # Directory links and timestamps are mutable when ordinary entries change.
    return info.st_dev, info.st_ino, info.st_mode


def _directory_flags():
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("Secure correction descriptor reads are unsupported")
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)


def _open_root(root, held):
    before = os.lstat(root.anchor)
    descriptor = os.open(root.anchor, _directory_flags())
    held.append((descriptor, None, root.anchor, before))
    if _directory_identity(os.fstat(descriptor)) != _directory_identity(before):
        raise ValueError("Correction filesystem root changed before open")
    for name in root.parts[1:]:
        descriptor = _open_child(descriptor, name, held)
    return descriptor


def _open_child(parent, name, held):
    _deadline()
    before = os.stat(name, dir_fd=parent, follow_symlinks=False)
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("Linked or non-directory correction ancestor")
    child = os.open(name, _directory_flags(), dir_fd=parent)
    held.append((child, parent, name, before))
    if _directory_identity(os.fstat(child)) != _directory_identity(before):
        raise ValueError("Correction directory changed before open")
    return child


def _check_directories(held):
    for descriptor, parent, name, before in reversed(held):
        current = os.lstat(name) if parent is None else os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (not stat.S_ISDIR(current.st_mode)
                or _directory_identity(current) != _directory_identity(before)
                or _directory_identity(os.fstat(descriptor)) != _directory_identity(before)):
            raise ValueError("Correction directory changed during authentication")


def _close_directories(held):
    for descriptor, _parent, _name, _before in reversed(held):
        os.close(descriptor)


def _read(root, name, maximum=MAX_FILE_BYTES):
    """Read a bounded regular file through non-following held directory paths."""
    _deadline()
    relative, root = _relative(name), _root(root)
    if type(maximum) is not int or not 0 <= maximum <= MAX_BUNDLE_BYTES:
        raise ValueError("Invalid correction file cap")
    held = []
    try:
        directory = _open_root(root, held)
        for part in relative.parts[:-1]:
            directory = _open_child(directory, part, held)
        before = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or not 0 <= before.st_size <= maximum):
            raise ValueError("Unsafe or oversized correction file")
        handle = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                         | getattr(os, "O_CLOEXEC", 0), dir_fd=directory)
        try:
            if _file_identity(os.fstat(handle)) != _file_identity(before):
                raise ValueError("Correction file changed before open")
            content = bytearray()
            while len(content) <= maximum:
                _deadline()
                chunk = os.read(handle, min(65536, maximum + 1 - len(content)))
                if not chunk:
                    break
                content.extend(chunk)
            after = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
            if (len(content) != before.st_size or len(content) > maximum
                    or _file_identity(os.fstat(handle)) != _file_identity(before)
                    or _file_identity(after) != _file_identity(before)):
                raise ValueError("Correction file changed during bounded read")
            _check_directories(held)
            return bytes(content)
        finally:
            os.close(handle)
    finally:
        _close_directories(held)


def _inventory(root):
    """Require the closed tree while every directory remains held and bound."""
    root = _root(root)
    held = []
    found, total = set(), 0
    expected_directories = {"v1", "v2"}
    try:
        descriptor = _open_root(root, held)
        pending = [(descriptor, "")]
        seen = 0
        while pending:
            directory, prefix = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    _deadline()
                    seen += 1
                    if seen > len(BUNDLE_FILES) + len(expected_directories):
                        raise ValueError("Correction tree has extra inventory entries")
                    name = prefix + entry.name
                    _relative(name)
                    info = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):
                        if name not in expected_directories:
                            raise ValueError("Unexpected correction directory")
                        child = _open_child(directory, entry.name, held)
                        pending.append((child, name + "/"))
                    elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                        if name not in BUNDLE_FILES or name in found or info.st_size > MAX_FILE_BYTES:
                            raise ValueError("Unexpected correction inventory file")
                        found.add(name)
                        total += info.st_size
                    else:
                        raise ValueError("Linked or nonregular correction inventory member")
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("Correction tree exceeds its total byte cap")
        _equal(found, set(BUNDLE_FILES), "closed 25-file inventory")
        _equal(total, BUNDLE_TOTAL_BYTES, "closed bundle bytes")
        _check_directories(held)
        return total
    finally:
        _close_directories(held)


def _json(raw):
    if type(raw) is not bytes or len(raw) > MAX_JSON_BYTES:
        raise ValueError("Oversized correction JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate correction JSON key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("Nonfinite correction JSON constant")
    try:
        result = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid correction JSON") from error
    pending, visited = [(result, 0)], 0
    while pending:
        _deadline()
        item, depth = pending.pop()
        visited += 1
        if depth > MAX_JSON_DEPTH or visited > MAX_JSON_NODES:
            raise ValueError("Correction JSON exceeds its structure cap")
        if type(item) is dict:
            pending.extend((value, depth + 1) for value in item.values())
        elif type(item) is list:
            pending.extend((value, depth + 1) for value in item)
        elif type(item) is str and len(item) > 16384:
            raise ValueError("Oversized correction JSON string")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("Nonfinite correction JSON number")
        elif type(item) is int and item.bit_length() > 256:
            raise ValueError("Oversized correction JSON integer")
    return result


def _pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _typed_pin(value):
    return (type(value) is dict and set(value) == {"bytes", "sha256"}
            and type(value["bytes"]) is int and 0 < value["bytes"] <= MAX_FILE_BYTES
            and type(value["sha256"]) is str and len(value["sha256"]) == 64
            and all(char in "0123456789abcdef" for char in value["sha256"]))


def _scope(value):
    pending = [value]
    while pending:
        _deadline()
        item = pending.pop()
        if type(item) is dict:
            for key, child in item.items():
                if key in FALSE_FIELDS and child is not False:
                    raise ValueError("Correction record exceeds its finite toy scope: " + key)
                pending.append(child)
        elif type(item) is list:
            pending.extend(item)


def _authenticate(bundle):
    root = _root(bundle)
    if (type(MANIFEST_SHA256) is not str or len(MANIFEST_SHA256) != 64
            or type(BUNDLE_TOTAL_BYTES) is not int):
        raise ValueError("Correction projection identity has not been frozen")
    raw = _read(root, "artifact-manifest.json", MAX_JSON_BYTES)
    _equal(len(raw), MANIFEST_BYTES, "top manifest bytes")
    _equal(hashlib.sha256(raw).hexdigest(), MANIFEST_SHA256, "top manifest digest")
    manifest = _json(raw)
    if type(manifest) is not dict or type(manifest.get("files")) is not dict:
        raise ValueError("Correction manifest requires its payload map")
    expected = set(BUNDLE_FILES) - {"artifact-manifest.json", "artifact-manifest.sha256"}
    _equal(set(manifest["files"]), expected, "top manifest payload names")
    data = {"artifact-manifest.json": raw}
    sidecar = _read(root, "artifact-manifest.sha256", 128)
    _equal(sidecar, (MANIFEST_SHA256 + "  artifact-manifest.json\n").encode("ascii"), "top sidecar")
    data["artifact-manifest.sha256"] = sidecar
    for name, expected_pin in manifest["files"].items():
        if not _typed_pin(expected_pin):
            raise ValueError("Correction manifest has an invalid payload pin")
        payload = _read(root, name, expected_pin["bytes"])
        _equal(_pin(payload), expected_pin, "payload identity " + name)
        data[name] = payload
    _equal(manifest.get("total_payload_bytes"), sum(len(data[name]) for name in expected), "top payload bytes")
    _equal(sum(len(value) for value in data.values()), BUNDLE_TOTAL_BYTES, "authenticated total bytes")
    _inventory(root)
    _equal(manifest["schema_version"], 1, "top schema")
    _equal(manifest["source_commit"], "a66724abb777298917655e76ec0556e1d5a2fb81", "historical source identity")
    _equal(manifest["original_v2_files"], 18, "historical payload count")
    _equal(manifest["original_v2_payload_bytes"], V2_TOTAL_BYTES, "historical payload bytes")
    _equal(manifest["original_v2_manifest_sha256"], V2_MANIFEST_SHA256, "historical manifest pin")
    _equal(manifest["v1_is_failed_and_distinct"], True, "distinct v1 failure")
    _equal(manifest["v2_historical_fixed_toy_pass"], True, "historical v2 pass")
    _scope(manifest)
    return root, data


def _index(text, name, width):
    prefix, suffix = name + "[", "]"
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("Invalid correction QASM bit reference")
    number = text[len(prefix):-1]
    if not number.isdigit() or str(int(number)) != number or not 0 <= int(number) < width:
        raise ValueError("Out-of-range correction QASM bit reference")
    return int(number)


def _qasm(raw, width, count, classical):
    """Parse the frozen numeric primitive alphabet without compiling a circuit."""
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("Correction QASM must be ASCII") from error
    headers = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{width}] q;"]
    if classical:
        headers.append(f"bit[{classical}] c;")
    if lines[:len(headers)] != headers or len(lines) > 5010:
        raise ValueError("Correction QASM header or source cap differs")
    instructions, global_phases = [], []
    for line in lines[len(headers):]:
        _deadline()
        if not line.endswith(";") or line != line.strip():
            raise ValueError("Correction QASM line is not a literal primitive")
        body = line[:-1]
        if body.startswith("gphase(") and body.endswith(")"):
            phase = float(body[7:-1])
            if not math.isfinite(phase):
                raise ValueError("Nonfinite correction QASM global phase")
            global_phases.append(phase)
            continue
        if " = measure " in body:
            left, right = body.split(" = measure ")
            instructions.append(("measure", (), (_index(right, "q", width),), (_index(left, "c", classical),)))
            continue
        head, arguments = body.split(" ", 1)
        if head == "reset":
            if not classical:
                raise ValueError("Reset outside complete correction readout trial")
            instructions.append(("reset", (), (_index(arguments, "q", width),), ()))
            continue
        if "(" in head:
            name, angle = head.split("(", 1)
            if not angle.endswith(")"):
                raise ValueError("Invalid correction QASM angle")
            params = (float(angle[:-1]),)
        else:
            name, params = head, ()
        if name not in PRIMITIVE_ARITY:
            raise ValueError("Opaque or unsupported correction QASM operation")
        arity, nparams = PRIMITIVE_ARITY[name]
        wires = tuple(_index(item, "q", width) for item in arguments.split(", "))
        if (len(wires) != arity or len(set(wires)) != arity or len(params) != nparams
                or any(not math.isfinite(value) for value in params)):
            raise ValueError("Correction QASM primitive has the wrong shape")
        instructions.append((name, params, wires, ()))
    _equal(len(instructions), count, "QASM primitive count")
    _equal(global_phases, [], "canonical zero global phase")
    return instructions


def _matrix_record(value, shape):
    if type(value) is not dict or set(value) != {"shape", "real", "imag"}:
        raise ValueError("Invalid correction matrix record")
    _equal(value["shape"], shape, "retained matrix shape")
    def array(item, dimensions):
        if not dimensions:
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError("Nonfinite or untyped retained correction entry")
        else:
            if type(item) is not list or len(item) != dimensions[0]:
                raise ValueError("Retained correction array shape differs")
            for child in item:
                array(child, dimensions[1:])
    array(value["real"], shape)
    array(value["imag"], shape)


def _records(data):
    nested_raw = data["v2/manifest.json"]
    _equal(hashlib.sha256(nested_raw).hexdigest(), V2_MANIFEST_SHA256, "original v2 manifest")
    _equal(data["v2/manifest.sha256"], (V2_MANIFEST_SHA256 + "  manifest.json\n").encode(), "original v2 sidecar")
    nested = _json(nested_raw)
    _equal(set(nested["files"]), set(V2_FILES) - {"manifest.json", "manifest.sha256"}, "original v2 payload map")
    for name, expected in nested["files"].items():
        _equal(_pin(data["v2/" + name]), expected, "original v2 payload " + name)
    _equal(sum(len(data["v2/" + name]) for name in V2_FILES), V2_TOTAL_BYTES, "original 18-file v2 bytes")
    _equal(nested["status"], "checked_fixed_finite_correction_step", "v2 status")
    _equal(nested["finite_toy_checks_passed"], True, "v2 finite check status")
    _equal(nested["whole_invocation_validated"], False, "original external-supervision boundary")
    _equal(hashlib.sha256(data["v2/results.json"]).hexdigest(), V2_RESULTS_SHA256, "original v2 results")
    result = _json(data["v2/results.json"])
    _equal(result["status"], "checked_fixed_finite_correction_step", "saved result status")
    _equal(result["finite_toy_checks_passed"], True, "saved finite checks")
    _equal(result["parameters"], {"terms": [[0.6, "Z"], [0.8, "X"]], "proposals": ["Z"],
                                "beta": 1.0, "clock_bits": 2, "product_steps": 1,
                                "tau": 0.125, "initial_state": "known_plus",
                                "readout": "Z with all five auxiliaries zero"}, "fixed parameters")
    work = result["work_inventory"]
    for name, expected in {"statevector_evolutions": 5, "tiny_expm_calls": 12,
                           "interval_matrix_power_calls": 2, "amplitudes_per_vector": [32, 32, 64, 64, 64],
                           "maximum_amplitudes_per_vector": 64, "total_returned_vector_amplitudes": 256}.items():
        _equal(work[name], expected, "historical work inventory")
    _equal(result["environment"]["python"], "3.12.14", "historical Python")
    for name, expected in {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1"}.items():
        _equal(result["environment"][name], expected, "historical package")
    if type(result["environment"]["thread_settings"]) is not dict or len(result["environment"]["thread_settings"]) != 7:
        raise ValueError("Historical correction thread settings differ")
    for value in result["environment"]["thread_settings"].values():
        _equal(value, "1", "historical single-thread setting")
    check = result["check"]
    _equal(check["checks_passed"], True, "historical checks")
    _equal(check["failed_checks"], [], "historical failed checks")
    if type(check["checks"]) is not dict or len(check["checks"]) != 28 or any(value is not True for value in check["checks"].values()):
        raise ValueError("Historical correction requires all 28 saved checks")
    _equal(check["tolerances"], {"entry": 1e-10, "completeness": 1e-10, "analytic": 1e-9}, "frozen numerical tolerances")
    for field in ("all_32_kraus_blocks", "all_outcome_probabilities"):
        if type(check[field]) is not list or len(check[field]) != 32:
            raise ValueError("Correction records must retain all 32 outcomes")
        for index, outcome in enumerate(check[field]):
            _equal(outcome["auxiliary_outcome_integer"], index, "auxiliary outcome index")
            _equal(outcome["auxiliary_label_q5_to_q1"], format(index, "05b"), "auxiliary outcome label")
            _equal(outcome["success"], index == 0, "auxiliary success selection")
            if field == "all_32_kraus_blocks":
                _matrix_record(outcome["matrix"], [2, 2])
            elif type(outcome["probability"]) not in (int, float) or not math.isfinite(outcome["probability"]):
                raise ValueError("Invalid retained outcome probability")
    parsed = {}
    _equal(set(result["source_records"]), set(SOURCE_KINDS), "six source records")
    for name, (width, count, classical) in SOURCE_KINDS.items():
        record = result["source_records"][name]
        _equal(record["qubits"], width, "source width")
        _equal(record["source_operations"], count, "source operations")
        _equal(record["classical_bits"], classical, "source classical bits")
        _equal(record["global_phase_radians"], 0.0, "source global phase")
        _equal(record["global_phase_preserved_in_qasm"], True, "global phase retention")
        parsed[name] = _qasm(data["v2/" + name + ".qasm"], width, count, classical)
        counts = {}
        for instruction in parsed[name]:
            counts[instruction[0]] = counts.get(instruction[0], 0) + 1
        _equal(counts, record["primitive_counts"], "literal primitive inventory")
    input_h = ("h", (), (0,), ())
    lcu = parsed["heralded_step"]
    _equal(parsed["known_plus_prefix"], [input_h, *lcu], "known-plus coherent prefix")
    _equal(parsed["known_plus_trial"], [("reset", (), (wire,), ()) for wire in range(6)]
           + [input_h, *lcu] + [("measure", (), (wire,), (wire,)) for wire in range(6)], "complete reset/readout trial")
    angle = result["source_records"]["heralded_step"]["metadata"]["lcu_selector_angle"]
    _equal(lcu, [("ry", (angle,), (5,), ()), *parsed["controlled_correction"],
                 ("p", (-math.pi / 2,), (5,), ()), ("ry", (-angle,), (5,), ())], "complete LCU chronology")
    theta = 0.9272952180016123
    controlled_h = [("ry", (-theta,), (0,), ()), ("h", (), (0,), ()),
                    ("cx", (), (4, 0), ()), ("h", (), (0,), ()), ("ry", (theta,), (0,), ())]
    _equal(parsed["correction_block"], [("h", (), (4,), ()), *controlled_h,
                 *parsed["centered_decay"], ("x", (), (4,), ()), *controlled_h,
                 ("x", (), (4,), ()), ("rz", (math.pi,), (4,), ()), ("h", (), (4,), ())], "centered commutator chronology")
    _equal(data["v1/centered_decay.qasm"], data["v2/centered_decay.qasm"], "retained failed-run source bytes")
    failure, completion = _json(data["v1/failure.json"]), _json(data["completion.json"])
    _equal(failure["status"], "failed_output_directory_identity_guard", "v1 failure status")
    _equal(failure["source_commit"], "43fa45c2c4f92d4f68edb8fb48cc5e1359ca2f92", "v1 source identity")
    _equal(failure["initial_directory_identity_retained"], False, "v1 missing initial identity")
    _equal(failure["partial_payloads_retained"], ["centered_decay.qasm"], "v1 partial payload")
    for name in ("outer_exit_code", "supervisor_exit_code", "cli_exit_code"):
        _equal(failure[name], 1, "v1 actual exit")
    _equal(completion["status"], "historical_v2_pass_public_projection", "public historical status")
    _equal(completion["source_commit"], "a66724abb777298917655e76ec0556e1d5a2fb81", "v2 public source identity")
    _equal(completion["v2_manifest_sha256"], V2_MANIFEST_SHA256, "public nested manifest pin")
    _equal(completion["v2_results_sha256"], V2_RESULTS_SHA256, "public results pin")
    _equal(completion["original_v2_bytes_unchanged"], True, "unchanged original payloads")
    _equal(completion["original_v2_payloads"], 18, "public historical file count")
    _equal(completion["original_v2_payload_bytes"], V2_TOTAL_BYTES, "public historical bytes")
    _equal(completion["checks_passed"], 28, "public historical check count")
    _equal(completion["all_kraus_outcomes"], 32, "public historical outcomes")
    _equal(completion["scientific_calls"], {"statevector_evolutions": 5, "tiny_expm_calls": 12,
                                           "interval_matrix_power_calls": 2,
                                           "amplitudes_per_vector": [32, 32, 64, 64, 64]}, "public historical work")
    _equal(completion["source_operations"], {name: shape[1] for name, shape in SOURCE_KINDS.items()}, "public source counts")
    _equal(completion["caps"], {"combined_output_bytes": 8388608, "cpu_seconds": 60,
                              "maximum_statevector_amplitudes_each": 64, "rss_bytes": 536870912,
                              "source_operations_each": 5000, "wall_seconds": 60}, "historical caps")
    _equal(completion["v1_failure_is_distinct_from_v2_pass"], True, "public distinct failure")
    for name in ("reported_outer_exit_code", "reported_supervisor_exit_code", "reported_cli_exit_code"):
        _equal(completion[name], 0, "v2 actual exit")
    for value in (nested, result, failure, completion):
        _scope(value)
    return result


def verify_bundle(bundle, *, source_root=None):
    """Authenticate immutable saved data; never execute its Python sources."""
    root, data = _authenticate(bundle)
    result = _records(data)
    if source_root is not None:
        selected = _root(source_root)
        for snapshot, relative in SOURCE_BINDINGS.items():
            _equal(_read(selected, relative), data["v2/" + snapshot], "selected source correspondence")
    _deadline()
    return {"status": "verified_saved_finite_correction_identity", "bundle_files": len(BUNDLE_FILES),
            "bundle_bytes": BUNDLE_TOTAL_BYTES, "top_manifest_sha256": MANIFEST_SHA256,
            "original_v2_files": len(V2_FILES), "original_v2_bytes": V2_TOTAL_BYTES,
            "historical_work": {name: result["work_inventory"][name] for name in
                                ("statevector_evolutions", "tiny_expm_calls", "interval_matrix_power_calls")},
            "historical_kraus_outcomes": 32, "source_root_checked": source_root is not None,
            "numerical_replay_performed": False, "all_components_verified": False,
            "new_algorithm_result": False, "physical_resources_computed": False}


def verify_download(archive, bundle):
    """Authenticate closed ZIP contents without extraction or code execution."""
    _root(bundle)
    _root_path, data = _authenticate(bundle)
    archive = Path(archive).absolute()
    raw = _read(archive.parent, archive.name, MAX_BUNDLE_BYTES)
    helper = Path(__file__).absolute()
    if helper.name != HELPER_NAME:
        raise ValueError("Correction helper filename differs from its portable contract")
    helper_raw = _read(helper.parent, helper.name)
    expected = {ZIP_PREFIX + name: value for name, value in data.items()}
    expected[HELPER_NAME] = helper_raw
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        entries = zipped.infolist()
        if len(entries) != len(expected) or len(entries) != 26:
            raise ValueError("Correction download has extra or missing members")
        names, total = set(), 0
        for entry in entries:
            _deadline()
            _relative(entry.filename)
            mode = entry.external_attr >> 16
            if (entry.filename not in expected or entry.filename in names or entry.is_dir()
                    or mode and not stat.S_ISREG(mode) or entry.flag_bits & 1
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
                    or entry.file_size != len(expected[entry.filename]) or entry.file_size > MAX_FILE_BYTES):
                raise ValueError("Unsafe or changed correction download member")
            names.add(entry.filename)
            total += entry.file_size
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("Correction download exceeds its uncompressed cap")
            with zipped.open(entry) as stream:
                content = stream.read(entry.file_size + 1)
                if content != expected[entry.filename] or stream.read(1):
                    raise ValueError("Correction download payload differs")
        _equal(names, set(expected), "closed download names")
    _deadline()
    return {"download_verified": True, "download_files": len(expected),
            "download_bytes": len(raw), "download_sha256": hashlib.sha256(raw).hexdigest()}


def _default_bundle():
    here = Path(__file__).absolute().parent
    adjacent = here / "detailed-balance-correction"
    return adjacent if adjacent.is_dir() else here.parent / BUNDLE_RELATIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=_default_bundle())
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--download", type=Path)
    args = parser.parse_args()
    try:
        result = verify_bundle(args.bundle, source_root=args.source_root)
        if args.download is not None:
            result.update(verify_download(args.download, args.bundle))
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, RuntimeError) as error:
        print(json.dumps({"status": "saved_correction_identity_failed", "error_type": type(error).__name__,
                          "numerical_replay_performed": False, "all_components_verified": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
