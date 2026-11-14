#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate preserved qDRIFT evidence and compare a fresh bounded replay.

Only standard-library saved-data arithmetic runs here. This gate does not run
circuits, regenerate seeds/readouts, synthesize rotations, query Native, or
certify floating-point quantum calculations. Its fixed canonical identity
preserves the reviewed producer; a replay has a separately supplied producer.
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
import stat
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/simulation/qdrift"
CANONICAL_MANIFEST_SHA256 = "7d2bbc3deb40f647009008a55b86c17042720df1916ab7d7bd00f9a066169709"
CANONICAL_PRODUCER = "76134612a3e6ae010ad58c0424703d96ee137665"
SOURCE_SHA256 = {
    "src/openquantum/simulation/qdrift.py": "567d993d3a7e975b340949eaade304beb5278a336055d15532999a70140189c4",
    "scripts/generate_simulation_qdrift.py": "9e788bab0d4ab7bd8c9595b2e093c97a1c0b7909c71cc2b448d7363b3a705aca",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
SOURCE_SNAPSHOTS = {
    "qdrift.py": "src/openquantum/simulation/qdrift.py",
    "generate_simulation_qdrift.py": "scripts/generate_simulation_qdrift.py",
    "LICENSE.txt": "LICENSE",
}
CASES = ("one-spin", "two-spin")
BUNDLE_FILES = frozenset({"manifest.json", "summary.csv", "summary.md", *SOURCE_SNAPSHOTS}) | frozenset(
    f"{case}/{name}" for case in CASES for name in (
        "study.json", "channel-convergence.csv", "channel-and-samples.svg", "first-complete-trial.svg",
        *(f"trial-{i:03d}.qasm" for i in range(32)),
    )
)
# The actual descriptor reads, including the extra-byte probe, are bounded.
MAX_FILE_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MAX_JSON_BYTES = 512 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024
MAX_JSON_NODES = 30000
MAX_JSON_DEPTH = 32
ATOL = 1e-11
PACKAGES = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
COEFFICIENT_SCOPE = "Exact Fraction(str(value)); repeated labels combined exactly; zero coefficients omitted. Identity terms are retained and carry global phase."
SAMPLE_SCOPE = "Seeded pseudorandom classical simulation; one final bit sampled from each freshly compiled ideal trajectory; no QPU execution"
RANDOMNESS_SCOPE = "Distinct PRNG seeds record reproducible streams, not a proof of ideal independence. Channel and confidence statements assume independent ideal categorical draws and readout."
ANGLE_SCOPE = "Sum of binary64 scalar angle conversions only; not a full compiler/synthesis certificate. Qiskit global-phase accumulation/reduction for identity terms is unbounded by this field and irrelevant to the measured channel."
QASM_SCOPE = "Measured-trial specification. Qiskit exporter may omit overall phase; restore the recorded phase before treating this as a coherently controlled oracle."
EXCLUSIONS = ["Physical reset, decoder, routing, factory and gate durations/faults", "Concrete Clifford+T synthesis and its coherent error certificate", "Classical input/sampling/scoring hardware time outside recorded CPU generation/check measurements"]
METRIC_SCOPE = "HS-induced2->2 superoperator norm, normalized Choi and fixed-input diagnostics; none is a computed diamond norm or certified floating-point bound"
DIAGNOSTIC_SCOPE = "Numerical diagnostics of an analytically CP/TP mixture of ideal unitaries; no formal or interval certificate"
SOURCE_SCOPE = "Exact checkout-root commit when Git top-level equals this package root; null for detached downloads. Content hashes identify actual producer files. Generated notebook/page output is a later artifact."
TIMING_SCOPE = "After request preflight and output-directory creation; includes circuit/channel checks, exports, figures and source copies; excludes interpreter imports, preflight and final metadata serialization."
TRUST_SCOPE = "Original finite circuit/channel numerical checks and seeded classical readouts. No physical device, formal proof, interval certificate, Native resource estimate or demonstrated advantage."
SUMMARY_HEADER = ["case", "qubits", "lambda_exact", "steps", "trajectories", "ensemble_Z0", "exact_Z0", "ensemble_bias", "sample_mean", "hoeffding_radius", "hs_induced_superoperator_error"]
CONVERGENCE_HEADER = ["steps", "hs_induced_superoperator_error", "normalized_choi_trace_distance", "fixed_input_trace_distance", "observable_bias"]
# Only these explicitly numerical diagnostic fields permit cross-run tolerance.
DIAGNOSTIC_FIELDS = frozenset({
    "source_to_independent_operator_error", "source_to_independent_state_error", "actual_reset_prefix_density_error",
    "state_probability_norm_before_normalization", "circuit_z0_expectation", "mean_of_fixed_trajectory_expectations",
    "empirical_to_ensemble_hs_superoperator_error", "hs_induced_superoperator_error", "normalized_choi_trace_distance",
    "fixed_input_trace_distance", "ensemble_observable", "exact_observable", "observable_bias",
    "trace_preservation_max_error", "hermiticity_preservation_max_error", "unitality_error", "normalized_choi_trace",
    "normalized_choi_hermitian_error", "normalized_choi_min_eigenvalue",
})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected, where):
    """Equality includes exact JSON leaf types (bool is never integer zero)."""
    _require(type(actual) is type(expected), f"{where}: wrong type")
    if isinstance(expected, dict):
        _require(actual.keys() == expected.keys(), f"{where}: wrong fields")
        for key in expected:
            _same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        _require(len(actual) == len(expected), f"{where}: wrong length")
        for i, value in enumerate(expected):
            _same(actual[i], value, f"{where}[{i}]")
    else:
        _require(actual.hex() == expected.hex() if type(expected) is float else actual == expected, f"{where}: changed value")


def _keys(value, names, where):
    _require(type(value) is dict and set(value) == set(names), f"{where}: wrong fields")


def _integer(value, lo, hi, where):
    _require(type(value) is int and lo <= value <= hi, f"{where}: invalid integer")
    return value


def _float(value, where, lo=-math.inf, hi=math.inf):
    _require(type(value) is float and math.isfinite(value) and lo <= value <= hi, f"{where}: invalid finite float")
    return value


def _close(actual, expected, where, atol=ATOL):
    _float(actual, where)
    _require(math.isfinite(expected) and abs(actual - expected) <= atol, f"{where}: numerical mismatch")


def _safe_relative(name):
    _require(type(name) is str and 1 <= len(name) <= 128 and "\\" not in name and "\0" not in name, "unsafe inventory path")
    path = PurePosixPath(name)
    _require(1 <= len(path.parts) <= 8 and not path.is_absolute() and path.as_posix() == name and all(p not in ("", ".", "..") for p in path.parts), "unsafe inventory path")
    return path.parts


def _open_directory(path):
    """Walk from / using no-follow directory descriptors, including ancestors."""
    path = Path(os.path.abspath(path))
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except (OSError, ValueError):
        os.close(fd)
        raise ValueError("bundle/source directory must be real and unredirected") from None


def _read(root_fd, name, maximum):
    parts = _safe_relative(name)
    directory = os.dup(root_fd)
    descriptor = None
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, f"{name}: not a bounded regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            data = handle.read(maximum + 1)
            after = os.fstat(handle.fileno())
        _require(len(data) <= maximum and len(data) == before.st_size, f"{name}: read exceeds limit or changed size")
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), f"{name}: changed during read")
        return data
    except OSError:
        raise ValueError(f"{name}: missing, redirected or unreadable file") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _inventory(root_fd):
    """Only the three fixed directories can be visited; unknown entries fail."""
    observed = set()
    for directory in ("", *CASES):
        try:
            fd = os.dup(root_fd) if not directory else os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=root_fd)
        except OSError:
            raise ValueError("incomplete or redirected bundle directory") from None
        try:
            with os.scandir(fd) as entries:
                for index, entry in enumerate(entries):
                    _require(index < 80, "inventory entry budget exceeded")
                    name = f"{directory}/{entry.name}" if directory else entry.name
                    mode = entry.stat(follow_symlinks=False).st_mode
                    if not directory and entry.name in CASES:
                        _require(stat.S_ISDIR(mode), f"{name}: redirected case directory")
                    else:
                        _require(name in BUNDLE_FILES and stat.S_ISREG(mode), f"{name}: unexpected or nonregular payload")
                        observed.add(name)
        except OSError:
            raise ValueError("incomplete or redirected bundle directory") from None
        finally:
            os.close(fd)
    _require(observed == BUNDLE_FILES, "bundle inventory is incomplete")


def _json(data, where):
    _require(len(data) <= MAX_JSON_BYTES, f"{where}: JSON byte budget exceeded")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, f"{where}: duplicate JSON key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError(f"{where}: nonfinite JSON constant")
    try:
        value = json.loads(data, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError):
        raise ValueError(f"{where}: invalid bounded JSON") from None
    stack = [(value, 0)]
    nodes = 0
    while stack:
        node, depth = stack.pop()
        nodes += 1
        _require(nodes <= MAX_JSON_NODES and depth <= MAX_JSON_DEPTH, f"{where}: JSON node/depth budget exceeded")
        if type(node) is dict:
            _require(nodes + len(stack) + len(node) <= MAX_JSON_NODES, f"{where}: JSON node budget exceeded")
            stack.extend((v, depth + 1) for v in node.values())
        elif type(node) is list:
            _require(nodes + len(stack) + len(node) <= MAX_JSON_NODES, f"{where}: JSON node budget exceeded")
            stack.extend((v, depth + 1) for v in node)
        elif type(node) is float:
            _require(math.isfinite(node), f"{where}: nonfinite JSON number")
        else:
            _require(type(node) in (str, int, bool, type(None)), f"{where}: invalid JSON leaf")
    return value


def _csv(data, header, rows, where):
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
        result = []
        for index, row in enumerate(reader):
            _require(index <= rows, f"{where}: CSV row budget exceeded")
            result.append(row)
    except (UnicodeError, csv.Error):
        raise ValueError(f"{where}: invalid CSV") from None
    _require(len(result) == rows + 1 and result[0] == header and all(len(row) == len(header) for row in result), f"{where}: CSV shape/header mismatch")
    return [dict(zip(header, row)) for row in result[1:]]


def _csv_number(text, where):
    _require(type(text) is str and len(text) <= 64, f"{where}: invalid CSV number")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{where}: invalid CSV number") from None
    _require(math.isfinite(value), f"{where}: nonfinite CSV number")
    return value


def _hamiltonian(qubits):
    coefficients, labels, probabilities, weights = ((["7/10", "3/10"], ["X", "Z"], ["7/10", "3/10"], [7, 3]) if qubits == 1 else (["1", "7/10", "7/10"], ["ZZ", "IX", "XI"], ["5/12", "7/24", "7/24"], [10, 7, 7]))
    return {"qubits": qubits, "terms": [{"coefficient_exact": c, "pauli_qiskit_order": p} for c, p in zip(coefficients, labels)], "lambda_exact": str(sum(map(Fraction, coefficients))), "probabilities_exact": probabilities, "integer_weights": weights, "total_integer_weight": sum(weights), "coefficient_semantics": COEFFICIENT_SCOPE}


def _primitive_trial(hamiltonian, schedule):
    """Independent deterministic primitive ledger; no PRNG or quantum matrix."""
    n = hamiltonian["qubits"]
    angle = float(2 * Fraction(hamiltonian["lambda_exact"]) * Fraction(1, 2) / 8)
    operations = [("reset", (wire,), None) for wire in range(n)]
    for term in schedule:
        label = hamiltonian["terms"][term]["pauli_qiskit_order"]
        active = [wire for wire in range(n) if label[n - 1 - wire] != "I"]
        x_wires = [wire for wire in active if label[n - 1 - wire] == "X"]
        operations.extend(("h", (wire,), None) for wire in x_wires)
        operations.extend(("cx", (wire, active[-1]), None) for wire in active[:-1])
        operations.append(("rz", (active[-1],), angle))
        operations.extend(("cx", (wire, active[-1]), None) for wire in reversed(active[:-1]))
        operations.extend(("h", (wire,), None) for wire in reversed(x_wires))
    operations.extend(("measure", (wire,), None) for wire in range(n))
    depth = [0] * n
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"bit[{n}] c;", f"qubit[{n}] q;"]
    for name, wires, parameter in operations:
        level = max(depth[wire] for wire in wires) + 1
        for wire in wires:
            depth[wire] = level
        operands = ", ".join(f"q[{wire}]" for wire in wires)
        if name == "measure":
            lines.append(f"c[{wires[0]}] = measure {operands};")
        else:
            lines.append(f"{name}{'(' + str(parameter) + ')' if parameter is not None else ''} {operands};")
    return Counter(op[0] for op in operations), max(depth), ("\n".join(lines) + "\n").encode(), angle


def _study(study, qubits, payloads, case):
    _keys(study, ("hamiltonian", "time_exact", "steps", "initial_basis", "master_seed", "trajectories", "sample_kind", "randomness_scope", "records", "sample_mean", "mean_of_fixed_trajectory_expectations", "empirical_to_ensemble_hs_superoperator_error", "ensemble_reference", "complete_workload", "ideal_iid_hoeffding", "physical_resource_estimate", "logical_ledgers", "validation"), case)
    ham = _hamiltonian(qubits)
    _same(study["hamiltonian"], ham, f"{case}.hamiltonian")
    for key, value in {"time_exact": "1/2", "steps": 8, "initial_basis": 0, "master_seed": 20261112 + qubits, "trajectories": 32, "sample_kind": SAMPLE_SCOPE, "randomness_scope": RANDOMNESS_SCOPE, "physical_resource_estimate": None}.items():
        _same(study[key], value, f"{case}.{key}")
    _same(study["validation"], {"performed": True, "trajectories_checked": 32, "full_channel_spanning_tests": 4**qubits, "numerical_tolerance": ATOL, "formal_certificate": False}, f"{case}.validation")
    records, ledgers = study["records"], study["logical_ledgers"]
    _require(type(records) is list and type(ledgers) is list and len(records) == len(ledgers) == 32, f"{case}: incomplete trials")
    total = Counter()
    seeds = set()
    for index, (record, ledger) in enumerate(zip(records, ledgers)):
        where = f"{case}.records[{index}]"
        _keys(record, ("trajectory_seed", "readout_seed", "schedule_indices", "source_to_independent_operator_error", "source_to_independent_state_error", "actual_reset_prefix_density_error", "state_probability_norm_before_normalization", "classically_sampled_basis_outcome", "readout_bitstring_qiskit_order", "circuit_z0_expectation", "classically_sampled_z0", "logical_ledger"), where)
        for key in ("trajectory_seed", "readout_seed"):
            value = _integer(record[key], 0, 2**64 - 1, f"{where}.{key}")
            _require(value not in seeds, f"{case}: duplicate stream seed")
            seeds.add(value)
        selected = record["schedule_indices"]
        _require(type(selected) is list and len(selected) == 8, f"{where}: incomplete schedule")
        for term in selected:
            _integer(term, 0, len(ham["terms"]) - 1, where + ".schedule")
        counts, depth, qasm, angle = _primitive_trial(ham, selected)
        filename = f"trial-{index:03d}.qasm"
        _require(payloads[f"{case}/{filename}"] == qasm, f"{where}: QASM is not the complete scheduled primitive trial")
        rounding = abs(Fraction.from_float(angle) - Fraction(ham["lambda_exact"]) / 8) / 2
        expected_ledger = {
            "hamiltonian": ham, "time_exact": "1/2", "steps_requested": 8, "categorical_draws": 8,
            "schedule_indices": selected, "trajectory_seed": record["trajectory_seed"], "initial_basis": 0,
            "system_qubits": qubits, "ancillas": 0, "logical_qubits": qubits,
            "complete_primitive_counts": dict(counts), "source_dependency_depth": depth, "arbitrary_rotations": 8,
            "magnitude_rz_angle_hex": angle.hex(), "per_exponential_scalar_angle_conversion_operator_bound_exact": str(rounding),
            "trajectory_scalar_angle_conversion_operator_bound_exact": str(rounding * 8), "angle_bound_scope": ANGLE_SCOPE,
            "global_phase_radians": 0.0, "resource_kind": "Complete measured trial before rotation synthesis and error correction",
            "physical_resource_estimate": None, "excluded_costs": EXCLUSIONS, "qasm_phase_scope": QASM_SCOPE,
            "measured_qasm_file": filename, "measured_qasm_sha256": hashlib.sha256(qasm).hexdigest(),
        }
        _same(ledger, expected_ledger, where + ".ledger")
        _same(record["logical_ledger"], ledger, where + ".duplicated_ledger")
        total.update(counts)
        outcome = _integer(record["classically_sampled_basis_outcome"], 0, 2**qubits - 1, where + ".outcome")
        _same(record["readout_bitstring_qiskit_order"], format(outcome, f"0{qubits}b"), where + ".bitstring")
        _same(record["classically_sampled_z0"], 1 if outcome & 1 == 0 else -1, where + ".score")
        for key in ("source_to_independent_operator_error", "source_to_independent_state_error", "actual_reset_prefix_density_error"):
            _float(record[key], where + "." + key, 0, ATOL)
        _close(record["state_probability_norm_before_normalization"], 1.0, where + ".normalization")
        _float(record["circuit_z0_expectation"], where + ".expectation", -1 - ATOL, 1 + ATOL)
    _same(study["complete_workload"], {"fresh_preparations": 32, "term_draws": 256, "primitive_counts": dict(total), "zero_state_resets": 32 * qubits, "final_measurements": 32 * qubits, "readout_samples_used": 32, "rational_weight_table_builds": 1, "classical_objective_scores": 32}, case + ".workload")
    _same(study["sample_mean"], sum(r["classically_sampled_z0"] for r in records) / 32, case + ".sample_mean")
    _close(study["mean_of_fixed_trajectory_expectations"], sum(r["circuit_z0_expectation"] for r in records) / 32, case + ".mean_expectation")
    _float(study["empirical_to_ensemble_hs_superoperator_error"], case + ".empirical_error", 0, 2 + ATOL)
    _same(study["ideal_iid_hoeffding"], {"failure_probability": 0.05, "absolute_radius": math.sqrt(2 * math.log(2 / .05) / 32), "target": "The averaged-channel Z0 mean, not the expectation of one fixed trajectory", "scope": "Analytic iid +/-1 bound; displayed PRNG sample does not validate coverage"}, case + ".hoeffding")
    ref = study["ensemble_reference"]
    _keys(ref, ("steps", "time_exact", "hs_induced_superoperator_error", "normalized_choi_trace_distance", "fixed_input_trace_distance", "ensemble_observable", "exact_observable", "observable_bias", "metric_scope", "channel_diagnostics"), case + ".ensemble")
    for key, value in {"steps": 8, "time_exact": "1/2", "metric_scope": METRIC_SCOPE}.items():
        _same(ref[key], value, case + ".ensemble." + key)
    for key in CONVERGENCE_HEADER[1:]:
        _float(ref[key], case + ".ensemble." + key, 0, 2 + ATOL)
    for key in ("ensemble_observable", "exact_observable"):
        _float(ref[key], case + ".ensemble." + key, -1 - ATOL, 1 + ATOL)
    _close(ref["observable_bias"], abs(ref["ensemble_observable"] - ref["exact_observable"]), case + ".bias")
    diag = ref["channel_diagnostics"]
    _keys(diag, ("spanning_matrix_units", "trace_preservation_max_error", "hermiticity_preservation_max_error", "unitality_error", "normalized_choi_trace", "normalized_choi_hermitian_error", "normalized_choi_min_eigenvalue", "scope"), case + ".channel")
    _same(diag["spanning_matrix_units"], 4**qubits, case + ".spanning_units")
    _same(diag["scope"], DIAGNOSTIC_SCOPE, case + ".channel.scope")
    for key in ("trace_preservation_max_error", "hermiticity_preservation_max_error", "unitality_error", "normalized_choi_hermitian_error"):
        _float(diag[key], case + ".channel." + key, 0, ATOL)
    _close(diag["normalized_choi_trace"], 1.0, case + ".choi_trace")
    _float(diag["normalized_choi_min_eigenvalue"], case + ".choi_min", -ATOL, 1 + ATOL)
    return total


def _load_bundle(bundle, *, source_root, canonical, expected_source_commit=None):
    root_fd = _open_directory(bundle)
    try:
        _inventory(root_fd)
        encoded = _read(root_fd, "manifest.json", MAX_MANIFEST_BYTES)
        digest = hashlib.sha256(encoded).hexdigest()
        if canonical:
            _require(digest == CANONICAL_MANIFEST_SHA256, "immutable canonical manifest changed")
        manifest = _json(encoded, "manifest")
        expected = {
            "schema": "openquantum-qdrift-study-v1", "status": "generated_and_ideal_checked", "reference_doi": "10.1103/PhysRevLett.123.070503",
            "source_commit": CANONICAL_PRODUCER if canonical else expected_source_commit, "git_producer_bindings_checked": True,
            "source_sha256": SOURCE_SHA256, "source_scope": SOURCE_SCOPE, "python": "3.12.14", "packages": PACKAGES,
            "device": "CPU", "seed": 20261113, "cases": list(CASES), "time_exact": "1/2", "steps": 8,
            "independent_seed_trajectories_recorded": 64, "generation_timing_scope": TIMING_SCOPE,
            "native_estimator_calls": 0, "factory_searches": 0, "gpu_jobs": 0, "physical_resource_estimate": None,
            "command": {"script": "scripts/generate_simulation_qdrift.py", "qubits": None, "time_exact": "1/2", "steps": 8, "trajectories_per_case": 32, "seed": 20261113, "initial_basis": 0, "coupling": "1", "x_field": "7/10", "z_field": "0", "no_sim": False, "output_path": "omitted; dedicated artifact directory"},
            "generator_bounds": {"qubits": 64, "term_exponentials": 512, "primitive_operations": 100000},
            "simulation_bounds": {"qubits": 3, "total_study_trajectories": 64}, "trust_boundary": TRUST_SCOPE,
        }
        _keys(manifest, (*expected, "scientific_wall_seconds", "files", "payload_bytes"), "manifest")
        for key, value in expected.items():
            _same(manifest[key], value, "manifest." + key)
        _float(manifest["scientific_wall_seconds"], "manifest.wall_seconds", 0, 180)
        files = manifest["files"]
        _keys(files, BUNDLE_FILES - {"manifest.json"}, "manifest.files")
        payloads = {}
        total_bytes = len(encoded)
        for name in sorted(files):
            row = files[name]
            _keys(row, ("bytes", "sha256"), name + ".inventory")
            size = _integer(row["bytes"], 1, MAX_FILE_BYTES, name + ".bytes")
            _require(type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is not None, name + ": invalid hash")
            _require(total_bytes + size <= MAX_TOTAL_BYTES, "bundle byte budget exceeded")
            data = _read(root_fd, name, size)
            _require(len(data) == size and hashlib.sha256(data).hexdigest() == row["sha256"], name + ": inventory authentication failed")
            payloads[name] = data
            total_bytes += size
        _same(manifest["payload_bytes"], total_bytes - len(encoded), "manifest.payload_bytes")
        if canonical:
            _same(total_bytes, 677245, "canonical.total_bytes")
            _same(len(encoded), 13440, "canonical.manifest_bytes")
    finally:
        os.close(root_fd)
    source_fd = _open_directory(source_root)
    try:
        for name, expected_hash in SOURCE_SHA256.items():
            _require(hashlib.sha256(_read(source_fd, name, MAX_FILE_BYTES)).hexdigest() == expected_hash, name + ": source identity changed")
        for snapshot, name in SOURCE_SNAPSHOTS.items():
            _require(hashlib.sha256(payloads[snapshot]).hexdigest() == SOURCE_SHA256[name], snapshot + ": snapshot identity changed")
    finally:
        os.close(source_fd)
    studies = {case: _json(payloads[f"{case}/study.json"], case + ".study") for case in CASES}
    totals = Counter()
    convergence = {}
    for qubits, case in enumerate(CASES, 1):
        totals.update(_study(studies[case], qubits, payloads, case))
        rows = _csv(payloads[f"{case}/channel-convergence.csv"], CONVERGENCE_HEADER, 8, case + ".convergence")
        for count, row in zip((1, 2, 4, 8, 16, 32, 64, 128), rows):
            _same(row["steps"], str(count), case + ".convergence.steps")
            for key in CONVERGENCE_HEADER[1:]:
                value = _csv_number(row[key], case + ".convergence." + key)
                _require(0 <= value <= 2 + ATOL, case + ": invalid convergence metric")
                if count == 8:
                    _close(value, studies[case]["ensemble_reference"][key], case + ".convergence.correspondence")
        convergence[case] = rows
    summary = _csv(payloads["summary.csv"], SUMMARY_HEADER, 2, "summary")
    lines = ["| Case | lambda | N | Fresh trials | Exact Z0 | Ensemble Z0 | Bias | Sample mean |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, row in zip(CASES, summary):
        study, ref = studies[case], studies[case]["ensemble_reference"]
        for key, value in {"case": case, "qubits": str(study["hamiltonian"]["qubits"]), "lambda_exact": study["hamiltonian"]["lambda_exact"], "steps": "8", "trajectories": "32"}.items():
            _same(row[key], value, "summary." + key)
        _same(_csv_number(row["sample_mean"], "summary.sample_mean"), study["sample_mean"], "summary.sample_mean")
        _same(_csv_number(row["hoeffding_radius"], "summary.hoeffding_radius"), study["ideal_iid_hoeffding"]["absolute_radius"], "summary.hoeffding_radius")
        for key, value in {"ensemble_Z0": ref["ensemble_observable"], "exact_Z0": ref["exact_observable"], "ensemble_bias": ref["observable_bias"], "hs_induced_superoperator_error": ref["hs_induced_superoperator_error"]}.items():
            _close(_csv_number(row[key], "summary." + key), value, "summary." + key)
        lines.append(f"| {case} | {study['hamiltonian']['lambda_exact']} | 8 | 32 | {ref['exact_observable']:.9f} | {ref['ensemble_observable']:.9f} | {ref['observable_bias']:.3g} | {study['sample_mean']:.4f} |")
    _require(payloads["summary.md"] == ("\n".join(lines) + "\n").encode(), "summary Markdown does not correspond to studies")
    _same(dict(totals), {"reset": 96, "h": 662, "rz": 512, "measure": 96, "cx": 198}, "whole workload")
    return {"manifest": manifest, "studies": studies, "payloads": payloads, "summary": summary, "convergence": convergence, "manifest_sha256": digest, "total_bytes": total_bytes}


def _report(bundle):
    return {"status": "verified_saved_ideal_qdrift_bundle", "manifest_sha256": bundle["manifest_sha256"], "source_commit": bundle["manifest"]["source_commit"], "files": len(BUNDLE_FILES), "total_bytes": bundle["total_bytes"], "trials": 64, "term_exponentials": 512, "primitive_counts": {"reset": 96, "rz": 512, "h": 662, "measure": 96, "cx": 198}, "native_estimator_calls": 0, "factory_searches": 0, "gpu_jobs": 0, "verification_scope": "Authenticated saved identities and independent ledger/CSV arithmetic; recorded quantum numerical checks are not rerun or formally certified"}


def verify_bundle(bundle, *, source_root=None):
    """Verify the fixed canonical bundle; there is no identity override."""
    return _report(_load_bundle(bundle, source_root=ROOT if source_root is None else source_root, canonical=True))


def _compare(actual, expected, where, atol):
    _require(type(actual) is type(expected), where + ": replay type changed")
    if type(expected) is dict:
        _require(actual.keys() == expected.keys(), where + ": replay fields changed")
        for key in expected:
            if type(expected[key]) is float and key in DIAGNOSTIC_FIELDS:
                _close(actual[key], expected[key], where + "." + key, atol)
            else:
                _compare(actual[key], expected[key], where + "." + key, atol)
    elif type(expected) is list:
        _require(len(actual) == len(expected), where + ": replay length changed")
        for i, item in enumerate(expected):
            _compare(actual[i], item, f"{where}[{i}]", atol)
    else:
        _same(actual, expected, where)


def compare_replay(canonical, replay, *, source_root, expected_source_commit, atol=ATOL):
    """Compare separately authenticated evidence to the immutable canonical run.

    expected_source_commit is the calling checkout's exact commit. The CLI
    obtains it from Git; detached public downloads only support the saved gate.
    The producer module/generator and locked inputs must retain pinned bytes.
    """
    _require(type(expected_source_commit) is str and re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is not None, "replay requires an exact source commit")
    _require(type(atol) is float and math.isfinite(atol) and 0 <= atol <= ATOL, "replay tolerance must not exceed 1e-11")
    old = _load_bundle(canonical, source_root=source_root, canonical=True)
    new = _load_bundle(replay, source_root=source_root, canonical=False, expected_source_commit=expected_source_commit)
    for case in CASES:
        _compare(new["studies"][case], old["studies"][case], case, atol)
        for index, (actual, expected) in enumerate(zip(new["convergence"][case], old["convergence"][case])):
            _same(actual["steps"], expected["steps"], f"{case}.csv[{index}].steps")
            for key in CONVERGENCE_HEADER[1:]:
                _close(_csv_number(actual[key], key), _csv_number(expected[key], key), f"{case}.csv[{index}].{key}", atol)
    for name in BUNDLE_FILES - {"manifest.json"}:
        if name.endswith(".qasm") or name in SOURCE_SNAPSHOTS:
            _require(new["payloads"][name] == old["payloads"][name], name + ": replay bytes changed")
    for index, (actual, expected) in enumerate(zip(new["summary"], old["summary"])):
        for key in ("case", "qubits", "lambda_exact", "steps", "trajectories"):
            _same(actual[key], expected[key], f"summary[{index}].{key}")
        _same(_csv_number(actual["sample_mean"], "sample_mean"), _csv_number(expected["sample_mean"], "sample_mean"), f"summary[{index}].sample_mean")
        _same(_csv_number(actual["hoeffding_radius"], "hoeffding_radius"), _csv_number(expected["hoeffding_radius"], "hoeffding_radius"), f"summary[{index}].hoeffding_radius")
        for key in ("ensemble_Z0", "exact_Z0", "ensemble_bias", "hs_induced_superoperator_error"):
            _close(_csv_number(actual[key], key), _csv_number(expected[key], key), f"summary[{index}].{key}", atol)
    result = _report(new)
    result["status"] = "verified_fresh_qdrift_replay"
    result["canonical_manifest_sha256"] = old["manifest_sha256"]
    result["diagnostic_atol"] = atol
    result["new_scientific_execution_by_verifier"] = False
    return result


def _current_commit(source_root):
    for arguments in (("rev-parse", "--show-toplevel"), ("rev-parse", "HEAD")):
        try:
            result = subprocess.run(["git", *arguments], cwd=source_root, capture_output=True, text=True, timeout=5, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise ValueError("fresh replay requires a readable Git checkout") from None
        _require(result.returncode == 0 and len(result.stdout) <= 4096 and len(result.stderr) <= 4096, "fresh replay requires a readable Git checkout")
        if arguments[-1] == "--show-toplevel":
            _require(Path(result.stdout.strip()).resolve() == Path(source_root).resolve(), "replay source must be the Git checkout root")
        else:
            commit = result.stdout.strip()
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "Git did not return an exact source commit")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / BUNDLE_RELATIVE)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    if args.replay_dir is None:
        result = verify_bundle(args.bundle_dir)
    else:
        result = compare_replay(args.bundle_dir, args.replay_dir, source_root=ROOT, expected_source_commit=_current_commit(ROOT))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
