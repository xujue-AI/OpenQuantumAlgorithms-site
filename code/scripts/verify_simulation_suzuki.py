#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate saved Suzuki records; never import or execute their producers.

Record arithmetic and QASM parsing are data checks, not a new simulation or
formal certificate. Replay comparison uses an explicit absolute diagnostic
tolerance while retaining exact circuits, parameters and scalar ledgers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from fractions import Fraction
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import stat
import subprocess
import time

BUNDLE_RELATIVE = "assets/generated/simulation/suzuki"
MANIFEST_SHA256 = "56b52ffa3729df2e4ba2ad6a5df67e61654df54b7654e97bbabe84f6199a0b8e"
PRODUCER_COMMIT = "a9edff74764c2e89ea835c1ef19fc5c1e18c0e64"
SOURCE_SHA256 = {
    "src/openquantum/simulation/suzuki.py": "7410f1a50873847f630a49e906351be2a0a042f509738942ae5fd32ba42748be",
    "scripts/generate_simulation_suzuki.py": "3336b53325942b21cb7b6cd0be5a3bcbf6fca149cbe696d14e4ba7e0f6403d9e",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
CASE_IDS = tuple(f"n{n}-s{order}-r{r}" for n in (1, 2, 3) for order in (2, 4, 6) for r in (1, 2, 4, 8))
SVG_FILES = frozenset({"signed-substeps.svg", "error-vs-steps.svg", "error-vs-work.svg",
                       "n1-order4-circuit.svg", "n2-order4-circuit.svg", "n3-order4-circuit.svg"})
BUNDLE_FILES = frozenset({"manifest.json", "LICENSE.txt", "suzuki.py", "generate_simulation_suzuki.py",
                         "summary.csv", "summary.md", *SVG_FILES,
                         *(f"{case}/{name}" for case in CASE_IDS for name in ("record.json", "measured-trial.qasm"))})
DIAGNOSTICS = ("source_to_independent_dense_product_operator_error", "source_to_qiskit_suzuki_operator_error",
               "reset_prefix_density_error", "operator_error_vs_exact_evolution", "state_vector_error_vs_exact_evolution",
               "state_infidelity", "circuit_z0", "exact_z0", "observable_bias")
CSV_FIELDS = ("case_id", "qubits", "order", "repetitions", "time_exact", "z_field_exact", "source_operations",
              "rotations", "cx", "source_depth", "negative_actual_substeps", "absolute_substep_time_exact",
              "operator_error", "state_error", "observable_bias", "exact_z0", "circuit_z0", "source_parameter_bound",
              "qiskit_reference_operator_difference")
TASK = {"time_exact": "1", "coupling": "1", "x_field": "7/10", "z_field": "0", "initial_basis": 0, "observable": "Z0"}
MAX_FILE_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 4 * 1024 * 1024
MAX_JSON_BYTES = 128 * 1024


def _equal(actual, expected, label):
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"Unexpected Suzuki {label}")
    if type(actual) is dict:
        for key in expected:
            _equal(actual[key], expected[key], label)
    elif type(actual) in (list, tuple):
        for left, right in zip(actual, expected):
            _equal(left, right, label)


def _number(value, label, *, minimum=None, maximum=None):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or untyped Suzuki {label}")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"Out-of-range Suzuki {label}")
    return value


def _root(value):
    root = Path(value).absolute()
    if root != root.resolve() or root.is_symlink() or not root.is_dir():
        raise ValueError("Suzuki root must be an unaliased directory")
    return root


def _relative(name):
    path = PurePosixPath(name)
    if not name or "\\" in name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("Unsafe Suzuki relative path")
    return path


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read(root, name, maximum=MAX_FILE_BYTES):
    """Read once through non-following directory descriptors, with a size cap."""
    relative = _relative(name)
    root = _root(root)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | os.O_NONBLOCK
    descriptors = []
    try:
        root_info = root.lstat()
        descriptor = os.open(root, directory)
        descriptors.append(descriptor)
        if _identity(os.fstat(descriptor)) != _identity(root_info):
            raise ValueError("Suzuki root changed before open")
        for part in relative.parts[:-1]:
            info = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(info.st_mode):
                raise ValueError("Linked or non-directory Suzuki ancestor")
            child = os.open(part, directory, dir_fd=descriptor)
            descriptors.append(child)
            if _identity(os.fstat(child)) != _identity(info):
                raise ValueError("Suzuki ancestor changed before open")
            descriptor = child
        info = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("Unsafe or oversized Suzuki file")
        handle = os.open(relative.name, os.O_RDONLY | nofollow | os.O_NONBLOCK, dir_fd=descriptor)
        descriptors.append(handle)
        if _identity(os.fstat(handle)) != _identity(info):
            raise ValueError("Suzuki file changed before open")
        content = bytearray()
        while len(content) <= maximum:
            chunk = os.read(handle, min(65536, maximum + 1 - len(content)))
            if not chunk:
                break
            content.extend(chunk)
        if len(content) != info.st_size or len(content) > maximum or _identity(os.fstat(handle)) != _identity(info):
            raise ValueError("Suzuki file changed during bounded read")
        return bytes(content)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _json(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Oversized Suzuki JSON")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate Suzuki JSON key")
            result[key] = value
        return result
    def constant(_value):
        raise ValueError("Nonfinite Suzuki JSON constant")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Invalid Suzuki JSON") from error
    pending = [(value, 0)]
    visited = 0
    while pending:
        item, depth = pending.pop(); visited += 1
        if depth > 16 or visited > 20000:
            raise ValueError("Suzuki JSON structural limit")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str and len(item) > 8192:
            raise ValueError("Oversized Suzuki JSON string")
        elif type(item) is float and not math.isfinite(item):
            raise ValueError("Nonfinite Suzuki JSON number")
        elif type(item) is int and item.bit_length() > 4096:
            raise ValueError("Oversized Suzuki JSON integer")
    return value


def _inventory(root):
    names = set(); directories = set(); pending = [(root, "")]
    while pending:
        directory, prefix = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                name = prefix + entry.name
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode) and name in CASE_IDS:
                    directories.add(name); pending.append((Path(entry.path), name + "/"))
                elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                    names.add(name)
                else:
                    raise ValueError("Unsafe Suzuki inventory entry")
                if len(names) > 84 or len(directories) > 36:
                    raise ValueError("Suzuki inventory limit")
    _equal(names, set(BUNDLE_FILES), "complete inventory")
    _equal(directories, set(CASE_IDS), "case directories")


def _authenticate(bundle, *, canonical):
    root = _root(bundle)
    raw = _read(root, "manifest.json", MAX_JSON_BYTES)
    digest = hashlib.sha256(raw).hexdigest()
    if canonical:
        _equal(digest, MANIFEST_SHA256, "immutable manifest pin")
        _equal(len(raw), 15502, "manifest bytes")
    manifest = _json(raw)
    _equal(manifest["schema"], "openquantum-suzuki-study-v1", "schema")
    _equal(manifest["status"], "generated_and_numerically_checked", "recorded status")
    _equal(manifest["case_ids"], list(CASE_IDS), "case schedule")
    _equal(manifest["canonical_case_count"], 36, "case count")
    _equal(manifest["common_task"], TASK, "common task")
    _equal(manifest["source_sha256"], SOURCE_SHA256, "producer hashes")
    if canonical:
        _equal(manifest["source_commit"], PRODUCER_COMMIT, "historical producer commit")
    for name in ("native_estimator_calls", "factory_searches", "gpu_jobs"):
        _equal(manifest[name], 0, name)
    _equal(manifest["physical_resource_estimate"], None, "physical exclusion")
    _number(manifest["generation_wall_seconds_after_imports"], "generation timing", minimum=0)
    files = manifest["files"]
    _equal(set(files), set(BUNDLE_FILES) - {"manifest.json"}, "payload inventory")
    _inventory(root)
    data = {"manifest.json": raw}; total = len(raw)
    for name in sorted(files):
        record = files[name]
        _equal(set(record), {"bytes", "sha256"}, "file metadata keys")
        if type(record["bytes"]) is not int or not 0 < record["bytes"] <= MAX_FILE_BYTES:
            raise ValueError("Invalid Suzuki file size")
        if type(record["sha256"]) is not str or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("Invalid Suzuki file hash")
        total += record["bytes"]
        if total > MAX_BUNDLE_BYTES:
            raise ValueError("Suzuki total-byte limit")
        body = _read(root, name, record["bytes"])
        _equal(len(body), record["bytes"], "payload size")
        _equal(hashlib.sha256(body).hexdigest(), record["sha256"], "payload hash")
        data[name] = body
    _equal(manifest["payload_bytes"], total - len(raw), "payload-byte sum")
    if canonical:
        _equal(total, 1695183, "canonical total bytes")
    return manifest, data, digest


def _sources(source_root):
    root = _root(source_root)
    for name, digest in SOURCE_SHA256.items():
        _equal(hashlib.sha256(_read(root, name, MAX_JSON_BYTES * 2)).hexdigest(), digest, "current source " + name)
    return root


def _fraction(value):
    if type(value) is not str or len(value) > 8192:
        raise ValueError("Invalid Suzuki rational record")
    parts = value.split("/")
    # A4096-bit integer has at most1234 decimal digits. Check the strings
    # before Fraction can expand an exponent or construct a large integer.
    if (len(parts) not in (1, 2) or not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", parts[0])
            or len(parts[0].lstrip("-")) > 1234
            or (len(parts) == 2 and (not re.fullmatch(r"[1-9][0-9]*", parts[1]) or len(parts[1]) > 1234))):
        raise ValueError("Invalid or oversized canonical Suzuki rational")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("Invalid Suzuki rational record") from error
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError("Suzuki rational limit")
    _equal(str(result), value, "canonical reduced rational spelling")
    return result


def _hex(value):
    if type(value) is not str or len(value) > 64:
        raise ValueError("Invalid Suzuki binary64 record")
    try:
        result = float.fromhex(value)
    except ValueError as error:
        raise ValueError("Invalid Suzuki binary64 record") from error
    _number(result, "binary64 angle")
    _equal(result.hex(), value, "canonical binary64 spelling")
    return result


def _expected_ops(ledger):
    n = ledger["system_qubits"]; r = ledger["repetitions"]; order = ledger["order"]
    rounding = ledger["source_parameter_rounding"]
    leaves = rounding["leaf_parameter_records_one_macro_step"]
    _equal(len(leaves), 5 ** (order // 2 - 1), "signed leaf count")
    _equal(rounding["root_bisection_bits"], 160, "recorded scalar enclosure precision")
    coefficients = []; coefficient_errors = []; angle_sum = Fraction(0)
    body = []
    for leaf in leaves:
        coefficient = _fraction(leaf["normalized_coefficient_source_exact"])
        _equal(Fraction.from_float(_hex(leaf["normalized_coefficient_binary64_hex"])), coefficient, "leaf scalar binding")
        coefficients.append(coefficient)
        lower, upper = map(_fraction, leaf["ideal_coefficient_enclosure"])
        if lower > upper:
            raise ValueError("Reversed Suzuki scalar enclosure")
        error = max(abs(coefficient - lower), abs(coefficient - upper))
        _equal(_fraction(leaf["coefficient_absolute_error_bound_exact"]), error, "saved coefficient allowance")
        coefficient_errors.append(error)
        dt = coefficient / r
        _equal(_fraction(leaf["source_substep_time_exact"]), dt, "leaf duration")
        parameters = leaf["rotation_parameters"]
        _equal([row["layer"] for row in parameters], ["ZZ half layers", "Z half layers", "X full layers"], "rotation layers")
        per_leaf = Fraction(0); angles = []
        for row, count, target in zip(parameters, (2 * (n - 1), 0, n), (dt, None, Fraction(7, 5) * dt)):
            _equal(row["occurrences_in_one_base_block"], count, "rotation occurrences")
            if count:
                _equal(_fraction(row["angle_target_exact"]), target, "rotation target")
                actual = _hex(row["binary64_angle_hex"])
                allowance = abs(Fraction.from_float(actual) - target) / 2
                angles.append(actual.hex())
            else:
                _equal(row["angle_target_exact"], None, "absent rotation target")
                _equal(row["binary64_angle_hex"], None, "absent rotation angle")
                allowance = Fraction(0); angles.append(None)
            _equal(_fraction(row["per_rotation_operator_conversion_bound_exact"]), allowance, "saved angle allowance")
            per_leaf += count * allowance
        _equal(_fraction(leaf["one_occurrence_gate_angle_conversion_bound_exact"]), per_leaf, "leaf angle allowance")
        angle_sum += r * per_leaf
        for edge in range(n - 1):
            body.extend((("cx", (edge, edge + 1), None), ("rz", (edge + 1,), angles[0]), ("cx", (edge, edge + 1), None)))
        body.extend(("rx", (wire,), angles[2]) for wire in range(n))
        for edge in reversed(range(n - 1)):
            body.extend((("cx", (edge, edge + 1), None), ("rz", (edge + 1,), angles[0]), ("cx", (edge, edge + 1), None)))
    signed = sum(coefficients, Fraction(0)); absolute = sum(map(abs, coefficients), Fraction(0))
    for key, expected in (("normalized_signed_sum_exact_binary64", signed), ("normalized_signed_sum_defect_exact", abs(signed - 1)),
                          ("normalized_absolute_sum_exact_binary64", absolute), ("actual_signed_substep_time_sum_exact", signed),
                          ("actual_absolute_substep_time_sum_exact", absolute)):
        _equal(_fraction(rounding[key]), expected, key)
    negative = sum(value < 0 for value in coefficients)
    _equal(negative, {2: 0, 4: 1, 6: 8}[order], "negative leaf signs")
    _equal(rounding["normalized_leaves_per_macro_step"], len(leaves), "macro-step leaves")
    _equal(rounding["negative_normalized_leaves_per_step"], negative, "negative normalized leaves")
    _equal(rounding["negative_actual_substeps"], negative * r, "negative actual substeps")
    coefficient_bound = Fraction(17 * n - 10, 10) * sum(coefficient_errors, Fraction(0))
    for key, expected in (("coefficient_approximation_operator_bound_exact", coefficient_bound),
                          ("gate_angle_conversion_operator_bound_exact", angle_sum),
                          ("total_source_parameter_operator_bound_exact", coefficient_bound + angle_sum)):
        _equal(_fraction(rounding[key]), expected, key)
    _equal(rounding["total_source_parameter_operator_bound_upper_float"], math.nextafter(float(coefficient_bound + angle_sum), math.inf), "scalar upper float")
    return [("reset", (wire,), None) for wire in range(n)] + body * r + [("measure", (wire,), None) for wire in range(n)]


def _qasm(raw, ledger):
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise ValueError("Non-ASCII Suzuki QASM") from error
    n = ledger["system_qubits"]
    _equal(lines[:4], ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{n}] c;", f"qubit[{n}] q;"], "QASM header")
    decimal = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    operations = []
    for line in lines[4:]:
        reset = re.fullmatch(r"reset q\[(\d+)\];", line)
        measure = re.fullmatch(r"c\[(\d+)\] = measure q\[(\d+)\];", line)
        cx = re.fullmatch(r"cx q\[(\d+)\], q\[(\d+)\];", line)
        rotation = re.fullmatch(rf"(rx|rz)\(({decimal})\) q\[(\d+)\];", line)
        if reset:
            operation = ("reset", (int(reset[1]),), None)
        elif measure and measure[1] == measure[2]:
            operation = ("measure", (int(measure[2]),), None)
        elif cx:
            operation = ("cx", (int(cx[1]), int(cx[2])), None)
        elif rotation:
            angle = _number(float(rotation[2]), "QASM angle")
            operation = (rotation[1], (int(rotation[3]),), angle.hex())
        else:
            raise ValueError("Unsupported Suzuki QASM instruction")
        if any(wire >= n for wire in operation[1]):
            raise ValueError("Suzuki QASM wire exceeds width")
        operations.append(operation)
    _equal(operations, _expected_ops(ledger), "complete signed QASM chronology")
    counts = dict(Counter(operation[0] for operation in operations))
    _equal(counts, ledger["complete_measured_primitive_counts"], "complete primitive counts")
    depths = [0] * n
    for _, wires, _ in operations:
        depth = max(depths[wire] for wire in wires) + 1
        for wire in wires:
            depths[wire] = depth
    _equal(max(depths), ledger["source_dependency_depth"], "source dependency depth")


def _records(manifest, data):
    records = {}; rows = []
    for case in CASE_IDS:
        record = _json(data[f"{case}/record.json"])
        _equal(set(record), {"case_id", "logical_ledger", "validation", "measured_qasm_sha256", "cpu_timings_seconds", "measurement_execution"}, "record keys")
        _equal(record["case_id"], case, "record case")
        n, order, r = map(int, re.fullmatch(r"n(\d)-s(\d)-r(\d)", case).groups())
        ledger = record["logical_ledger"]; validation = record["validation"]
        for key, expected in (("system_qubits", n), ("logical_qubits", n), ("ancillas", 0), ("initial_basis", 0),
                              ("order", order), ("repetitions", r), ("time_exact", "1"), ("base_strang_blocks", r * 5 ** (order // 2 - 1))):
            _equal(ledger[key], expected, key)
        hamiltonian = ledger["hamiltonian"]
        for key, expected in (("qubits", n), ("coupling_exact", "1"), ("x_field_exact", "7/10"), ("z_field_exact", "0"),
                              ("boundary", "open"), ("hbar", 1)):
            _equal(hamiltonian[key], expected, "Hamiltonian " + key)
        _equal(_fraction(hamiltonian["pauli_l1_norm_exact"]), Fraction(17 * n - 10, 10), "Pauli norm")
        _equal(ledger["physical_resource_estimate"], None, "ledger physical exclusion")
        _equal(validation["formal_certificate"], False, "formal exclusion")
        _equal(record["measurement_execution"]["physical_measurements"], False, "physical measurements")
        _equal(record["measurement_execution"]["finite_shot_samples"], False, "finite shots")
        _equal(ledger["planned_sampling_contract"]["fresh_repreparations"], True, "planned fresh preparations")
        _equal(ledger["planned_sampling_contract"]["required_trials_hoeffding"], 2952, "budgeted unexecuted trials")
        if not ledger["exclusions"] or "not executed" not in ledger["planned_sampling_contract"]["status"]:
            raise ValueError("Missing Suzuki resource exclusions")
        _equal(set(record["cpu_timings_seconds"]), {"source_construction_scalar_accounting_export", "independent_and_qiskit_numerical_checks"}, "timing keys")
        for key, value in record["cpu_timings_seconds"].items():
            _number(value, key, minimum=0)
        _equal(validation["verified_tolerance"], 1e-10, "recorded numerical-check ceiling")
        for key, expected in (("qubits", n), ("order", order), ("repetitions", r), ("time_exact", "1")):
            _equal(validation[key], expected, "validation " + key)
        for key in DIAGNOSTICS:
            _number(validation[key], key, minimum=-1.0000000001 if key in ("circuit_z0", "exact_z0") else 0,
                    maximum=1.0000000001 if key in ("circuit_z0", "exact_z0", "state_infidelity") else None)
        for key in DIAGNOSTICS[:3]:
            if validation[key] > validation["verified_tolerance"]:
                raise ValueError("Saved Suzuki numerical check exceeds its recorded ceiling")
        _equal(validation["observable_bias"], abs(validation["circuit_z0"] - validation["exact_z0"]), "observable-bias binding")
        qasm = data[f"{case}/measured-trial.qasm"]
        _equal(record["measured_qasm_sha256"], hashlib.sha256(qasm).hexdigest(), "record/QASM hash")
        _qasm(qasm, ledger)
        counts = ledger["complete_measured_primitive_counts"]
        _equal(ledger["arbitrary_rotations"], counts.get("rx", 0) + counts.get("rz", 0), "rotation count")
        rounding = ledger["source_parameter_rounding"]
        values = (case, n, order, r, "1", "0", sum(counts.values()), ledger["arbitrary_rotations"], counts.get("cx", 0),
                  ledger["source_dependency_depth"], rounding["negative_actual_substeps"], rounding["actual_absolute_substep_time_sum_exact"],
                  validation["operator_error_vs_exact_evolution"], validation["state_vector_error_vs_exact_evolution"], validation["observable_bias"],
                  validation["exact_z0"], validation["circuit_z0"], rounding["total_source_parameter_operator_bound_upper_float"],
                  validation["source_to_qiskit_suzuki_operator_error"])
        rows.append(dict(zip(CSV_FIELDS, values))); records[case] = record
    expected = io.StringIO(); writer = csv.DictWriter(expected, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _equal(data["summary.csv"], expected.getvalue().encode(), "CSV/record binding")
    lines = ["| n | Order | r | Complete gates | Operator error | Z0 bias | Negative blocks |", "|---:|---:|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {row['qubits']} | {row['order']} | {row['repetitions']} | {row['source_operations']} | {row['operator_error']:.4g} | {row['observable_bias']:.4g} | {row['negative_actual_substeps']} |" for row in rows if row["repetitions"] in (1, 8))
    _equal(data["summary.md"], ("\n".join(lines) + "\n").encode(), "Markdown/record binding")
    for snapshot, source in (("suzuki.py", "src/openquantum/simulation/suzuki.py"), ("generate_simulation_suzuki.py", "scripts/generate_simulation_suzuki.py"), ("LICENSE.txt", "LICENSE")):
        _equal(hashlib.sha256(data[snapshot]).hexdigest(), SOURCE_SHA256[source], "producer snapshot")
    return records


def _verify_bundle(bundle, *, source_root=None):
    """Verify immutable saved data and all four current producer-file identities."""
    manifest, data, digest = _authenticate(bundle, canonical=True)
    if source_root is not None:
        _sources(source_root)
    _records(manifest, data)
    return {"status": "verified_saved_suzuki_study", "manifest_sha256": digest, "cases": 36, "files": 84,
            "bytes": 1695183, "recorded_numerical_checks": True, "new_simulations": 0, "native_estimator_calls": 0,
            "physical_resource_estimate": None, "formal_certificate": False,
            "current_source_files_checked": 4 if source_root is not None else 0}


def _git_bytes(root, arguments, maximum):
    """Bound local Git identity reads; no producer code or Git hooks execute."""
    try:
        process = subprocess.Popen(["git", *arguments], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return None, b""
    body = bytearray(); deadline = time.monotonic() + 5
    try:
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ValueError("Suzuki Git identity timeout")
                if not selector.select(min(remaining, 0.1)):
                    continue
                chunk = os.read(process.stdout.fileno(), min(65536, maximum + 1 - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
                if len(body) > maximum:
                    raise ValueError("Suzuki Git identity output limit")
        return process.wait(timeout=max(0.01, deadline - time.monotonic())), bytes(body)
    except subprocess.TimeoutExpired as error:
        raise ValueError("Suzuki Git identity timeout") from error
    finally:
        if process.poll() is None:
            process.kill(); process.wait(timeout=1)
        process.stdout.close()


def _replay_revision(manifest, root):
    status, raw = _git_bytes(root, ["rev-parse", "--show-toplevel"], 4096)
    if status != 0 or Path(os.fsdecode(raw).strip()).resolve() != root:
        _equal(manifest["source_commit"], None, "detached replay source commit")
        return
    status, raw = _git_bytes(root, ["rev-parse", "HEAD"], 128)
    commit = raw.decode("ascii").strip()
    if status != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Invalid Suzuki replay checkout identity")
    _equal(manifest["source_commit"], commit, "fresh replay source commit")
    for name, digest in SOURCE_SHA256.items():
        status, raw = _git_bytes(root, ["show", f"{commit}:{name}"], MAX_JSON_BYTES * 2)
        if status != 0 or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Fresh Suzuki replay source is not committed exactly")


def _compare_replay(canonical, replay, *, source_root, atol=1e-11):
    """Compare authenticated saved outputs; this does not run the generator."""
    _number(atol, "absolute replay tolerance", minimum=0)
    original, saved, digest = _authenticate(canonical, canonical=True)
    fresh, data, fresh_digest = _authenticate(replay, canonical=False)
    root = _sources(source_root)
    _replay_revision(fresh, root)
    old_records = _records(original, saved); new_records = _records(fresh, data)
    volatile = {"source_commit", "generation_wall_seconds_after_imports", "files", "payload_bytes"}
    _equal({k: v for k, v in fresh.items() if k not in volatile}, {k: v for k, v in original.items() if k not in volatile}, "replay manifest contract")
    maximum = 0.0
    for case in CASE_IDS:
        old = old_records[case]; new = new_records[case]
        _equal(new["logical_ledger"], old["logical_ledger"], "exact replay ledger")
        _equal(new["measurement_execution"], old["measurement_execution"], "replay measurement exclusions")
        _equal(data[f"{case}/measured-trial.qasm"], saved[f"{case}/measured-trial.qasm"], "exact replay QASM")
        old_static = {key: value for key, value in old["validation"].items() if key not in DIAGNOSTICS}
        new_static = {key: value for key, value in new["validation"].items() if key not in DIAGNOSTICS}
        _equal(new_static, old_static, "replay validation scope")
        for key in DIAGNOSTICS:
            delta = abs(new["validation"][key] - old["validation"][key]); maximum = max(maximum, delta)
            if delta > atol:
                raise ValueError(f"Suzuki replay diagnostic exceeds absolute tolerance: {case}/{key}")
    return {"status": "verified_saved_suzuki_replay_comparison", "manifest_sha256": digest,
            "replay_manifest_sha256": fresh_digest, "cases": 36, "absolute_diagnostic_tolerance": atol,
            "maximum_absolute_diagnostic_difference": maximum, "new_simulations": 0, "native_estimator_calls": 0}


def verify_bundle(bundle, *, source_root=None):
    """Authenticate the canonical study without running its scientific stack."""
    try:
        return _verify_bundle(bundle, source_root=source_root)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed saved Suzuki data") from error


def compare_replay(canonical, replay, *, source_root, atol=1e-11):
    """Compare saved outputs at an explicit absolute diagnostic tolerance."""
    try:
        return _compare_replay(canonical, replay, source_root=source_root, atol=atol)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed saved Suzuki replay data") from error


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir", type=Path, default=root / BUNDLE_RELATIVE)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--atol", type=float, default=1e-11)
    args = parser.parse_args()
    result = (verify_bundle(args.records_dir, source_root=root) if args.replay_dir is None else
              compare_replay(args.records_dir, args.replay_dir, source_root=root, atol=args.atol))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
