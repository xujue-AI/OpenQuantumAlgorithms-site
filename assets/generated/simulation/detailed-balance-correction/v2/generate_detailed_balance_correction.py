#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the separately authorized fixed finite-correction check once.

This driver emits an original primitive correction oracle and one heralded
step for the accepted tilted spin. It performs exactly five bounded vector
evolutions, twelve tiny exponentials and two interval matrix powers. It draws
no samples and implements no general correction, sampler or physical model.
The source must be frozen and the exact invocation reviewed before execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time

START_WALL = time.monotonic()
START_CPU = time.process_time()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
MAX_WALL_SECONDS = 60.0
MAX_CPU_SECONDS = 60.0
MAX_RSS_BYTES = 512 * 1024 * 1024
MAX_SOURCE_OPERATIONS = 5000
MAX_VECTOR_AMPLITUDES = 64
MAX_VECTOR_WIRES = 6
EXPECTED_VECTOR_WIDTHS = (5, 5, 6, 6, 6)
EXPECTED_CALLS = {"statevector_evolutions": 5, "tiny_expm_calls": 12,
                  "interval_matrix_power_calls": 2}
EXPECTED_VERSIONS = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1"}
EXPECTED_PYTHON = "3.12.14"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
    "RAYON_NUM_THREADS",
)
SOURCE_NAMES = (
    "detailed_balance_access.py", "detailed_balance_decay.py",
    "detailed_balance_channel.py", "detailed_balance_decay_reference.py",
    "detailed_balance_correction.py", "detailed_balance_correction_reference.py",
)
GENERATOR_NAME = "generate_detailed_balance_correction.py"
# New-source identities are filled only after their independent source reviews.
# The external execution packet separately authenticates this driver's bytes.
SOURCE_SHA256 = {
    "detailed_balance_access.py": "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77",
    "detailed_balance_decay.py": "82a304024e7e1b27f5fd283359f620d9b0be5e545dfc9ba494c09737197e72e1",
    "detailed_balance_channel.py": "99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b",
    "detailed_balance_decay_reference.py": "71ac5f486ad04cb532a86bf1c4dec4e175d0cfb377a7de26f206cd3f5c5daf53",
    "detailed_balance_correction.py": "62ede4bbafe79a15e7c66068f0dc2b54a32322bbbff4015f00e7a0ba64e3eb48",
    "detailed_balance_correction_reference.py": "103393d232861867dc5e70da04fa4ec26f892b3cfc92dfed06cd1ea5cc1f413c",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE.txt": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
SOURCE_BYTES = {
    "detailed_balance_access.py": 25361, "detailed_balance_decay.py": 12324,
    "detailed_balance_channel.py": 24362, "detailed_balance_decay_reference.py": 26002,
    "detailed_balance_correction.py": 24902,
    "detailed_balance_correction_reference.py": 33840,
    "requirements-lock.txt": 1974, "LICENSE.txt": 1093,
}
SOURCE_KEYS = (
    "centered_decay", "correction_block", "controlled_correction",
    "heralded_step", "known_plus_trial", "known_plus_prefix",
)
CLOSED_OUTPUT_NAMES = frozenset({
    *(name + ".qasm" for name in SOURCE_KEYS), *SOURCE_NAMES,
    GENERATOR_NAME, "requirements-lock.txt", "LICENSE.txt", "results.json",
    "manifest.json", "manifest.sha256",
})
EXPECTED_SOURCE_WIDTHS = dict(zip(SOURCE_KEYS, (4, 5, 6, 6, 6, 6)))
TOLERANCES = {"entry": 1e-10, "completeness": 1e-10, "analytic": 1e-9}
PRIMITIVE_ARITY = {
    "h": (1, 0), "x": (1, 0), "s": (1, 0), "sdg": (1, 0),
    "t": (1, 0), "tdg": (1, 0), "p": (1, 1), "ry": (1, 1),
    "rz": (1, 1), "cx": (2, 0), "reset": (1, 0), "measure": (1, 0),
}
SCOPE_FLAGS = {
    "all_components_verified": False, "new_algorithm_result": False,
    "coherent_correction_implemented": False,
    "coherent_correction_evolution_implemented": False,
    "general_channel_implemented": False, "sampler_implemented": False,
    "exact_finite_kms": False, "mixing_certificate": False,
    "rotations_synthesized": False, "physical_resources_computed": False,
    "formal_proof_reproduced": False, "random_sampling_performed": False,
    "floating_point_error_certified": False, "hardware_error_certified": False,
}


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _regular_bytes(path: Path, maximum=MAX_SOURCE_BYTES) -> bytes:
    """Take one bounded non-following snapshot of a selected regular file."""
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or not 0 < before.st_size <= maximum or path != path.resolve()):
        raise ValueError("source inputs must be bounded unaliased regular files")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | os.O_NONBLOCK
    descriptor = os.open(path, flags)
    try:
        if _identity(os.fstat(descriptor)) != _identity(before):
            raise ValueError("source input changed before its bounded read")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(65536, maximum + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        if (len(raw) != before.st_size or len(raw) > maximum
                or _identity(os.fstat(descriptor)) != _identity(before)
                or _identity(path.lstat()) != _identity(before)):
            raise ValueError("source input changed during its bounded read")
        return bytes(raw)
    finally:
        os.close(descriptor)


def _selected_sources():
    """Adjacent complete snapshots win; no parent Git provenance is queried."""
    adjacent = any(os.path.lexists(HERE / name) for name in SOURCE_NAMES)
    directory = HERE if adjacent else ROOT / "src/openquantum/simulation"
    if directory != directory.resolve() or not directory.is_dir():
        raise ValueError("the selected source directory must be unaliased")
    if Path(__file__).name != GENERATOR_NAME:
        raise ValueError("the fixed driver filename differs from its source contract")
    paths = {name: directory / name for name in SOURCE_NAMES}
    paths[GENERATOR_NAME] = Path(__file__).resolve()
    paths["requirements-lock.txt"] = HERE / "requirements-lock.txt" if adjacent else ROOT / "requirements-lock.txt"
    paths["LICENSE.txt"] = HERE / "LICENSE.txt" if adjacent else ROOT / "LICENSE"
    snapshots = {name: _regular_bytes(path) for name, path in paths.items()}
    for name, expected in SOURCE_SHA256.items():
        if (len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)
                or len(snapshots[name]) != SOURCE_BYTES[name]
                or hashlib.sha256(snapshots[name]).hexdigest() != expected):
            raise ValueError("a selected source, lock or license differs from its reviewed pin")
    return directory, adjacent, paths, snapshots


def _environment_preflight():
    """Inspect metadata without importing the scientific packages."""
    if platform.system() not in ("Darwin", "Linux"):
        raise ValueError("the fixed check requires the reviewed local CPU runtime")
    if platform.python_version() != EXPECTED_PYTHON:
        raise ValueError("the fixed check requires the exact frozen Python version")
    if not sys.dont_write_bytecode or not sys.flags.no_user_site:
        raise ValueError("the fixed check requires bytecode and user-site loading disabled")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ValueError("PYTHONNOUSERSITE must match the frozen environment")
    if any(os.environ.get(name) != "1" for name in THREAD_ENVIRONMENT):
        raise ValueError("every declared CPU thread setting must equal one")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_VERSIONS}
    if versions != EXPECTED_VERSIONS:
        raise ValueError("scientific distribution metadata differs from the frozen lock")
    return {"python": platform.python_version(), **versions, "device": "local CPU",
            "thread_settings": {name: "1" for name in THREAD_ENVIRONMENT},
            "seed": None, "random_sampling_performed": False}


def _output_preflight(output: Path):
    if (not isinstance(output, Path) or not output.is_absolute()
            or output != output.resolve() or os.path.lexists(output)):
        raise ValueError("the fixed check requires an absolute unaliased absent output directory")
    parent = output.parent
    if not parent.is_dir() or parent != parent.resolve() or parent.is_symlink():
        raise ValueError("the output parent must already be an unaliased directory")


def _load_snapshots(directory, paths, snapshots):
    """Execute the already authenticated producer byte snapshots in order."""
    if any(name.removesuffix(".py") in sys.modules for name in SOURCE_NAMES):
        raise ValueError("the fixed driver requires a fresh producer-module namespace")
    sys.path.insert(0, str(directory))
    modules = {}
    for filename in SOURCE_NAMES:
        name = filename.removesuffix(".py")
        spec = importlib.util.spec_from_file_location(name, paths[filename])
        if spec is None or spec.loader is None:
            raise ValueError("the authenticated source module could not be prepared")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        exec(compile(snapshots[filename], str(paths[filename]), "exec"), module.__dict__)
        if Path(module.__file__).resolve() != paths[filename]:
            raise ValueError("a producer module changed its selected source identity")
        modules[name] = module
    for filename, path in paths.items():
        if _regular_bytes(path) != snapshots[filename]:
            raise ValueError("selected source bytes changed before scientific calls")
    return modules


def _source_guard(circuit, *, expected_width=None, unitary_only=False):
    width = circuit.num_qubits
    if (type(width) is not int or not 1 <= width <= MAX_VECTOR_WIRES
            or expected_width is not None and width != expected_width):
        raise ValueError("source width differs from the fixed bounded contract")
    if len(circuit.data) > MAX_SOURCE_OPERATIONS:
        raise ValueError("source exceeds the fixed primitive-operation cap")
    phase = float(circuit.global_phase)
    if not math.isfinite(phase) or len(circuit.data) + int(phase != 0) > MAX_SOURCE_OPERATIONS:
        raise ValueError("source phase or complete operation count is outside its cap")
    for item in circuit.data:
        name = item.operation.name
        if name not in PRIMITIVE_ARITY or unitary_only and name in ("reset", "measure"):
            raise ValueError("source contains an unsupported primitive")
        arity, nparams = PRIMITIVE_ARITY[name]
        if len(item.qubits) != arity or len(item.operation.params) != nparams:
            raise ValueError("source primitive has an unexpected arity")
        if name == "measure":
            if len(item.clbits) != 1:
                raise ValueError("measurement requires one explicit output bit")
        elif item.clbits:
            raise ValueError("a unitary/reset primitive has unexpected classical wires")
        if getattr(item.operation, "condition", None) is not None:
            raise ValueError("conditional source instructions are outside the fixed contract")
        if any(not math.isfinite(float(value)) for value in item.operation.params):
            raise ValueError("source has a nonfinite primitive angle")
    if unitary_only and circuit.num_clbits:
        raise ValueError("statevector source must have no classical bits")


class WorkBudget:
    """Reserve each capped scientific call before invoking its backend.

    Resource observations are cooperative checkpoints of this process. The
    separately reviewed supervisor enforces the whole invocation boundary.
    """
    def __init__(self):
        import numpy as np
        from qiskit.quantum_info import Statevector
        from scipy.linalg import expm
        self.np, self.Statevector, self._expm = np, Statevector, expm
        self.counts = {name: 0 for name in EXPECTED_CALLS}
        self.vector_amplitudes = []
        self.max_observed_rss_bytes = 0
        self.checkpoint()

    def checkpoint(self):
        wall, cpu = time.monotonic() - START_WALL, time.process_time() - START_CPU
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss) * (1 if platform.system() == "Darwin" else 1024)
        self.max_observed_rss_bytes = max(self.max_observed_rss_bytes, rss)
        if wall > MAX_WALL_SECONDS or cpu > MAX_CPU_SECONDS or rss > MAX_RSS_BYTES:
            raise ValueError("the driver reached a cooperative resource limit")

    def _reserve(self, name):
        self.checkpoint()
        if self.counts[name] >= EXPECTED_CALLS[name]:
            raise ValueError("the fixed scientific call budget was exhausted")
        self.counts[name] += 1

    def evolve(self, initial_vector, circuit):
        index = self.counts["statevector_evolutions"]
        if index >= len(EXPECTED_VECTOR_WIDTHS):
            raise ValueError("the fixed vector-call budget was exhausted")
        _source_guard(circuit, expected_width=EXPECTED_VECTOR_WIDTHS[index], unitary_only=True)
        if (not isinstance(initial_vector, self.np.ndarray) or initial_vector.ndim != 1
                or initial_vector.size != 1 << circuit.num_qubits
                or initial_vector.size > MAX_VECTOR_AMPLITUDES
                or not self.np.isfinite(initial_vector).all()):
            raise ValueError("the initial vector differs from the bounded source layout")
        norm_squared = self.np.vdot(initial_vector, initial_vector)
        if abs(norm_squared - 1) > 1e-12:
            raise ValueError("the initial vector must be normalized")
        self._reserve("statevector_evolutions")
        result = self.Statevector(initial_vector).evolve(circuit).data
        if result.shape != initial_vector.shape or not self.np.isfinite(result).all():
            raise ValueError("the bounded vector backend returned an invalid result")
        self.vector_amplitudes.append(int(result.size))
        self.checkpoint()
        return result

    def _tiny_matrix(self, matrix):
        if (not isinstance(matrix, self.np.ndarray) or matrix.shape != (2, 2)
                or not self.np.isfinite(matrix).all()):
            raise ValueError("the fixed matrix backend requires a finite two-by-two array")

    def expm(self, matrix):
        self._tiny_matrix(matrix)
        self._reserve("tiny_expm_calls")
        result = self._expm(matrix)
        self._tiny_matrix(result)
        self.checkpoint()
        return result

    def matrix_power(self, matrix, steps):
        self._tiny_matrix(matrix)
        if type(steps) is not int or steps != 1:
            raise ValueError("the fixed interval matrix-power exponent must be one")
        self._reserve("interval_matrix_power_calls")
        result = self.np.linalg.matrix_power(matrix, steps)
        self._tiny_matrix(result)
        self.checkpoint()
        return result

    def completed(self):
        self.checkpoint()
        if self.counts != EXPECTED_CALLS or self.vector_amplitudes != [32, 32, 64, 64, 64]:
            raise ValueError("the completed run differs from the exact call inventory")
        return {**self.counts, "amplitudes_per_vector": list(self.vector_amplitudes),
                "maximum_amplitudes_per_vector": max(self.vector_amplitudes),
                "total_returned_vector_amplitudes": sum(self.vector_amplitudes),
                "observed_self_peak_rss_bytes": self.max_observed_rss_bytes,
                "resource_observation_scope": "cooperative process checkpoints; external supervision is required"}


def _qasm3(circuit):
    """Serialize only the actual primitive stream, including its global phase."""
    _source_guard(circuit)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{circuit.num_qubits}] q;"]
    if circuit.num_clbits:
        lines.append(f"bit[{circuit.num_clbits}] c;")
    phase = float(circuit.global_phase)
    if phase != 0:
        lines.append(f"gphase({phase!r});")
    for item in circuit.data:
        name = item.operation.name
        wires = [circuit.find_bit(bit).index for bit in item.qubits]
        if name == "measure":
            bit = circuit.find_bit(item.clbits[0]).index
            lines.append(f"c[{bit}] = measure q[{wires[0]}];")
        elif name == "reset":
            lines.append(f"reset q[{wires[0]}];")
        else:
            angle = "(" + repr(float(item.operation.params[0])) + ")" if item.operation.params else ""
            arguments = ", ".join(f"q[{wire}]" for wire in wires)
            lines.append(f"{name}{angle} {arguments};")
    return ("\n".join(lines) + "\n").encode("ascii")


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class OutputBundle:
    def __init__(self, output):
        _output_preflight(output)
        os.mkdir(output, 0o700)
        self.path = output
        self.descriptor = os.open(output, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                                  | getattr(os, "O_NOFOLLOW", 0))
        self.identity = _identity(os.fstat(self.descriptor))
        self.inventory = {}
        self.total_bytes = 0

    def _check_directory_identity(self):
        path_info = self.path.lstat()
        held_info = os.fstat(self.descriptor)
        # Directory link counts change when entries are created on this filesystem.
        # Bind the path and held descriptor by device, inode and full mode only.
        if (not stat.S_ISDIR(path_info.st_mode) or not stat.S_ISDIR(held_info.st_mode)
                or _identity(path_info)[:3] != self.identity[:3]
                or _identity(held_info)[:3] != self.identity[:3]):
            raise ValueError("the owned output directory changed identity")

    def write(self, name, raw):
        if (not isinstance(raw, bytes) or not name or name in (".", "..")
                or "/" in name or "\\" in name or name in self.inventory
                or self.total_bytes + len(raw) > MAX_OUTPUT_BYTES):
            raise ValueError("output violates the closed flat-file/byte contract")
        self._check_directory_identity()
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(name, flags, 0o600, dir_fd=self.descriptor)
        try:
            pending = memoryview(raw)
            while pending:
                count = os.write(descriptor, pending)
                if count <= 0:
                    raise ValueError("output write made no progress")
                pending = pending[count:]
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size != len(raw):
                raise ValueError("output is not the owned regular payload")
        finally:
            os.close(descriptor)
        self.total_bytes += len(raw)
        self.inventory[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def closed(self, expected_names):
        self._check_directory_identity()
        if len(expected_names) != 18 or set(expected_names) != set(CLOSED_OUTPUT_NAMES):
            raise ValueError("the expected public output inventory is not the fixed eighteen files")
        actual = {}
        with os.scandir(self.path) as entries:
            for entry in entries:
                if entry.name in actual or len(actual) >= len(expected_names):
                    raise ValueError("the public output inventory has extra/repeated members")
                actual[entry.name] = entry.stat(follow_symlinks=False)
        if set(actual) != set(expected_names) or set(actual) != set(self.inventory):
            raise ValueError("the output inventory is missing members or has extras")
        if any(not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
               or info.st_size != self.inventory[name]["bytes"] for name, info in actual.items()):
            raise ValueError("the output tree contains a changed/nonregular payload")
        if sum(info.st_size for info in actual.values()) != self.total_bytes or self.total_bytes > MAX_OUTPUT_BYTES:
            raise ValueError("the complete output byte inventory is inconsistent")
        for name, pin in self.inventory.items():
            if hashlib.sha256(_regular_bytes(self.path / name, MAX_OUTPUT_BYTES)).hexdigest() != pin["sha256"]:
                raise ValueError("a public output payload changed after its write")
        self._check_directory_identity()

    def close(self):
        os.close(self.descriptor)


def _checked_report(check):
    if (type(check) is not dict or type(check.get("checks_passed")) is not bool
            or type(check.get("failed_checks")) is not list or check.get("tolerances") != TOLERANCES
            or check["checks_passed"] != (check["failed_checks"] == [])):
        raise ValueError("the independent report has an invalid acceptance contract")


def generate(output: Path):
    """Execute only the fixed, separately granted toy and retain every branch."""
    _output_preflight(output)
    environment = _environment_preflight()
    directory, adjacent, paths, snapshots = _selected_sources()
    modules = _load_snapshots(directory, paths, snapshots)
    builder = modules["detailed_balance_correction"]
    reference = modules["detailed_balance_correction_reference"]
    work = WorkBudget()
    sources = builder.build_source_bundle()
    if type(sources) is not dict or set(sources) != set(SOURCE_KEYS):
        raise ValueError("the builder returned an unexpected source inventory")
    for name in SOURCE_KEYS:
        _source_guard(sources[name], expected_width=EXPECTED_SOURCE_WIDTHS[name],
                      unitary_only=name != "known_plus_trial")
    work.checkpoint()
    # Capture finite source bytes before checking, without matrix materialization.
    qasm_payloads = {name + ".qasm": _qasm3(sources[name]) for name in SOURCE_KEYS}
    check = reference.check_sources(sources, work, TOLERANCES)
    _checked_report(check)
    inventory = work.completed()
    report = {
        "schema_version": 1,
        "status": "checked_fixed_finite_correction_step" if check["checks_passed"] else "failed_fixed_finite_correction_step",
        "parameters": {"terms": [[0.6, "Z"], [0.8, "X"]], "proposals": ["Z"],
                       "beta": 1.0, "clock_bits": 2, "product_steps": 1, "tau": 0.125,
                       "initial_state": "known_plus", "readout": "Z with all five auxiliaries zero"},
        "paper": {"doi": "10.1038/s41586-025-09583-x", "correction_equation": 19,
                  "construction": "original centered-decay order-select primitive correction and one heralded LCU step"},
        "accepted_decay_manifest_sha256": "9acc1af928c3484a8ad7c24a5768a1c4ac643cc02c541e5ad0df2b0df1223f94",
        "environment": environment,
        "source_commit": None, "source_revision_is_exact": False,
        "source_provenance_scope": "exact file-byte identities; commit/environment attestation belongs to the external frozen execution packet",
        "adjacent_snapshot_selected": adjacent,
        "source_files": {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
                         for name, raw in snapshots.items()},
        "environment_lock_sha256": SOURCE_SHA256["requirements-lock.txt"],
        "source_records": {name: {"qubits": sources[name].num_qubits,
                                   "classical_bits": sources[name].num_clbits,
                                   "source_operations": len(sources[name].data),
                                   "qasm_operations_including_nonzero_global_phase": len(sources[name].data) + int(float(sources[name].global_phase) != 0),
                                   "global_phase_radians": float(sources[name].global_phase),
                                   "global_phase_preserved_in_qasm": True,
                                   "primitive_counts": dict(sources[name].count_ops()),
                                   "metadata": sources[name].metadata}
                           for name in SOURCE_KEYS},
        "check": check, "work_inventory": inventory,
        "finite_toy_numerical_checks_performed": True,
        "finite_toy_checks_passed": check["checks_passed"],
        "whole_invocation_validated": False,
        "bounded_invocation_acceptance_requires_external_supervisor": True,
        "caps": {"whole_invocation_wall_seconds": MAX_WALL_SECONDS,
                 "whole_invocation_cpu_seconds": MAX_CPU_SECONDS, "rss_bytes": MAX_RSS_BYTES,
                 "primitive_operations_each_source_including_nonzero_global_phase": MAX_SOURCE_OPERATIONS,
                 "statevector_wires": MAX_VECTOR_WIRES, "statevector_amplitudes_each": MAX_VECTOR_AMPLITUDES,
                 "exact_scientific_call_inventory": EXPECTED_CALLS,
                 "complete_output_bytes_including_manifests": MAX_OUTPUT_BYTES},
        "acceptance": {"floating_point_tolerances": TOLERANCES,
                       "ideal_model_bias_is_distinct_from_floating_point_acceptance": True,
                       "synthesis_and_hardware_error_certification_performed": False,
                       "exact_h_finite_decay_is_correction_acceptance_target": False,
                       "global_phase_fitting_performed": False, "negative_roundoff_clipped": False},
        "scope": "one fixed finite-Trotter correction block and one known-input heralded step, retaining all 32 Kraus outcomes",
        "excluded": ["unknown-input retry recovery and multi-step correction algorithm",
                     "general-H correction, continuum/KMS and mixing/output certification",
                     "sampling draws and measured readout statistics", "rotation synthesis",
                     "hardware/code/factory/routing/controller/fault models",
                     "certified floating-point or analytic interval arithmetic"],
        "wall_seconds_at_report": time.monotonic() - START_WALL,
        "cpu_seconds_at_report": time.process_time() - START_CPU,
        **SCOPE_FLAGS,
    }
    output_bundle = OutputBundle(output)
    try:
        for name, raw in qasm_payloads.items():
            output_bundle.write(name, raw)
        for name, raw in snapshots.items():
            output_bundle.write(name, raw)
        output_bundle.write("results.json", _json_bytes(report))
        manifest = {"schema_version": 1, "status": report["status"],
                    "files": dict(output_bundle.inventory),
                    "total_payload_bytes": output_bundle.total_bytes,
                    "manifest_and_sidecar_are_separate": True,
                    "finite_toy_checks_passed": check["checks_passed"],
                    "whole_invocation_validated": False, **SCOPE_FLAGS}
        manifest_raw = _json_bytes(manifest)
        output_bundle.write("manifest.json", manifest_raw)
        output_bundle.write("manifest.sha256", (hashlib.sha256(manifest_raw).hexdigest() + "  manifest.json\n").encode("ascii"))
        expected_names = set(qasm_payloads) | set(snapshots) | {"results.json", "manifest.json", "manifest.sha256"}
        output_bundle.closed(expected_names)
        work.checkpoint()
    finally:
        output_bundle.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-tilted-check", action="store_true", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = generate(arguments.output_dir)
    except Exception as error:
        # This bounded diagnostic goes only to the supervisor's private log.
        # Public source/result exports never contain exception text or paths.
        diagnostic = (type(error).__name__ + ": " + str(error)).encode("utf-8", "replace")[:4095] + b"\n"
        sys.stderr.buffer.write(diagnostic)
        sys.stderr.buffer.flush()
        print(json.dumps({"status": "failed_or_refused_fixed_correction_check",
                          "error_type": type(error).__name__, "successful_completion_claimed": False,
                          "all_components_verified": False, "new_algorithm_result": False}))
        return 1
    print(json.dumps({"status": report["status"], "finite_toy_checks_passed": report["finite_toy_checks_passed"],
                      "scientific_calls": report["work_inventory"],
                      "all_components_verified": False, "new_algorithm_result": False}, allow_nan=False))
    return 0 if report["finite_toy_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
