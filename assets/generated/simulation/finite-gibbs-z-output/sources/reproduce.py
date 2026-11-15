# SPDX-License-Identifier: MIT
"""Source-only portable reproduction of the fixed, ideal Gibbs-Z check.

This new wrapper has not been executed or validated as a downloaded package.
It loads the exact accepted primitive QASM; it never regenerates its angles.
The accepted original calculation and its raw flags remain separate evidence.
Run only into a distinct absent output directory using the bundled lock.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "47ffd10368a738f9576053491a56c6d80a10e602"
VERSIONS = {"numpy": "2.5.4", "qiskit": "2.6.2", "scipy": "1.18.1", "psutil": "7.2.2"}
PROGRAMS = {
    "access": ("access-U.qasm", 6, 0, 140,
               "93ab81b8d5008d886a2096f97e69b590985980546a4b60fd68128db0b0d19a65"),
    "dilation": ("weak-dilation.qasm", 7, 0, 1493,
                 "e11625b24787d2513ce3bb4af95742c31b90734209183b797cc92d3297f0ffc3"),
    "reset_round": ("reset-round.qasm", 7, 0, 1505,
                    "ee2404c83e2f075dba35dbf02555432b385b9c25dee26bfb4f0d0ee576f0268d"),
    "preparation_R64": ("preparation-R64.qasm", 7, 1, 96323,
                        "e584fa94fffe1618f2029c8f311e5b0eb7a188f514ca019a4e34feee461076b8"),
    "preparation_R2": ("preparation-R2.qasm", 7, 1, 3013,
                       "83cefa8e3fef9920a72f4e58f1312b29a8f93f0b18c63dfb4ce7424ba713bedc"),
}
INPUT_PINS = {
    "sources/detailed_balance_access.py": "7c0869fddc758d58d256ee44025ead0bd04331dce8c2925f5f105639b65cef77",
    "sources/detailed_balance_channel.py": "99ac8dbfacf6ee484caeb6ad328fe145f150871e4272538dd279f67cc2573a8b",
    "sources/finite_gibbs_interval.py": "8f035148771d8659057cd7f013d0378ec5d9995f4e2fe255bc453b348d776362",
    "sources/finite_gibbs_angle_provenance.py": "175de0ead4e4e32ee679c42831662682e58c1467688d4bfe08ffaec6265801c1",
    "sources/finite_gibbs_reference.py": "1f82517c1992f93ec3fa7af49982286e64c7cf011b69a07bdc20ac1dbb956ac4",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE.txt": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
    **{"circuits/" + row[0]: row[4] for row in PROGRAMS.values()},
}
EXPECTED_CALLS = {"statevector_evolutions": 2, "external_density_evolutions": 1,
                  "matrix_power_calls": 2, "independent_full_U_Kraus_constructions": 1,
                  "fixed_interval_certificate_evaluations": 1}
PARAMETERS = {"terms": [[1.0, "Z"]], "proposal_labels": ["X", "Z"],
              "beta": 1.0, "clock_bits": 4, "M": 16, "product_steps": 1,
              "delta": 0.125, "rounds": 64, "initial": "plus",
              "measurement_samples_projected": 16384}
CAPS = {"wall_seconds": 120, "cpu_seconds": 120, "rss_bytes": 805306368,
        "combined_output_bytes": 16777216, "aggregate_source_primitives": 110000}
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
GATE_LINE = re.compile(r"([a-z]+)(?:\((" + NUMBER + r")\))? (q\[[0-9]+\](?:, q\[[0-9]+\])?);\Z")
MEASURE_LINE = re.compile(r"c\[([0-9]+)\] = measure q\[([0-9]+)\];\Z")


def authenticate_payload():
    """Only regular local files with the published immutable pins are inputs."""
    loaded = {}
    for relative, expected in INPUT_PINS.items():
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("a pinned payload is missing or is not a regular file")
        raw = path.read_bytes()
        if len(raw) > 4 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("a pinned payload changed or exceeds its byte bound")
        loaded[relative] = raw
    return loaded


def load_native_qasm(text, expected, work):
    """Parse only this export's canonical primitive subset; no custom gates."""
    _filename, qubits, clbits, count, _digest = expected
    lines = text.splitlines()
    header = ["OPENQASM 3.0;", 'include "stdgates.inc";', "qubit[" + str(qubits) + "] q;"]
    if clbits:
        header.append("bit[" + str(clbits) + "] c;")
    if lines[:len(header)] != header or len(lines) != len(header) + count or not text.endswith("\n"):
        raise ValueError("QASM headers or primitive count differ from the fixed export")
    circuit = work.QuantumCircuit(qubits, clbits)
    for index, line in enumerate(lines[len(header):]):
        if index % 1024 == 0:
            work.checkpoint()
        measured = MEASURE_LINE.fullmatch(line)
        if measured is not None:
            classical, quantum = map(int, measured.groups())
            if not 0 <= classical < clbits or not 0 <= quantum < qubits:
                raise ValueError("measurement operand is outside the fixed source")
            circuit.append(work.native["measure"][0](), [quantum], [classical])
            continue
        match = GATE_LINE.fullmatch(line)
        if match is None:
            raise ValueError("QASM contains syntax outside the fixed primitive subset")
        name, token, operands = match.groups()
        if name not in work.native or name == "measure":
            raise ValueError("QASM contains an unsupported primitive")
        base, _types, qarity, _carity, parity = work.native[name]
        wires = [int(word) for word in re.findall(r"q\[([0-9]+)\]", operands)]
        if len(wires) != qarity or len(set(wires)) != qarity or any(q >= qubits for q in wires):
            raise ValueError("QASM primitive operands differ from their native arity")
        parameters = [] if token is None else [float(token)]
        if len(parameters) != parity or any(not math.isfinite(p) for p in parameters):
            raise ValueError("QASM primitive has invalid numeric parameters")
        if token is not None and format(parameters[0], ".17g") != token:
            raise ValueError("a canonical source literal did not round-trip to native binary64")
        circuit.append(base(*parameters), wires)
    return circuit


class Work:
    """Fixed direct scientific calls and cooperative portable bounds.

    Version and native-class checks do not reproduce private native-file or
    process-lifetime authentication. SDK internal kernels are unmeasured.
    """

    def __init__(self):
        self.start_wall, self.start_cpu = time.monotonic(), time.process_time()
        import numpy as np
        import psutil
        from qiskit import QuantumCircuit
        from qiskit.circuit import Measure, Reset
        from qiskit.circuit.library import (CXGate, HGate, PhaseGate, RYGate, RZGate,
                                           SGate, SdgGate, TGate, TdgGate, XGate)
        from qiskit.quantum_info import DensityMatrix, Statevector
        self.np, self.QuantumCircuit = np, QuantumCircuit
        self.Statevector, self.DensityMatrix = Statevector, DensityMatrix
        classes = {"h": (HGate, (), 1, 0, 0), "x": (XGate, (), 1, 0, 0),
                   "s": (SGate, (), 1, 0, 0), "sdg": (SdgGate, (), 1, 0, 0),
                   "t": (TGate, (), 1, 0, 0), "tdg": (TdgGate, (), 1, 0, 0),
                   "ry": (RYGate, (0.0,), 1, 0, 1), "rz": (RZGate, (0.0,), 1, 0, 1),
                   "p": (PhaseGate, (0.0,), 1, 0, 1), "cx": (CXGate, (), 2, 0, 0),
                   "reset": (Reset, (), 1, 0, 0), "measure": (Measure, (), 1, 1, 0)}
        self.native = {name: (base, frozenset({base, type(base(*args))}), qa, ca, pa)
                       for name, (base, args, qa, ca, pa) in classes.items()}
        self.calls = dict.fromkeys(EXPECTED_CALLS, 0)
        self.preprocess = {"mesh_constructions": 0, "numpy_exp_calls": 0,
                           "ordinary_scalar_transcendentals": 0,
                           "explicit_reference_matrix_products": 0}
        self.process = psutil.Process()
        self.sampled_peak_rss = 0
        self.phase = "preprocess"
        self.sources, self.held_sources, self.signatures = {}, {}, {}
        self.powers = None
        self.observations = {"matrices": {}, "actual_basis_columns": []}
        owner = self

        class MathProxy:
            def __getattr__(self, name):
                target = getattr(math, name)
                if name in {"sqrt", "sin", "cos", "atan2"}:
                    def forwarded(*args):
                        owner.preprocessing("ordinary_scalar_transcendentals", 50000)
                        return target(*args)
                    return forwarded
                return target

        self.math = MathProxy()

    def checkpoint(self):
        rss = self.process.memory_info().rss
        self.sampled_peak_rss = max(self.sampled_peak_rss, rss)
        if (time.monotonic() - self.start_wall > CAPS["wall_seconds"]
                or time.process_time() - self.start_cpu > CAPS["cpu_seconds"]
                or rss > CAPS["rss_bytes"]):
            raise RuntimeError("a cooperative portable work bound was exceeded")

    @contextmanager
    def scoped(self, phase):
        previous, self.phase = self.phase, phase
        try:
            yield
        finally:
            self.phase = previous

    def count(self, name):
        self.checkpoint()
        if name not in EXPECTED_CALLS or self.calls[name] >= EXPECTED_CALLS[name]:
            raise ValueError("the direct scientific call inventory was exceeded")
        self.calls[name] += 1

    def preprocessing(self, name, cap):
        self.checkpoint()
        if self.phase == "preprocess":
            if self.preprocess[name] >= cap:
                raise ValueError("a fixed preprocessing count was exceeded")
            self.preprocess[name] += 1

    def exp(self, value):
        value = self.np.asarray(value)
        if value.size > 4096:
            raise ValueError("reference elementwise exponential exceeds its fixed size")
        self.preprocessing("numpy_exp_calls", 128)
        return self.np.exp(value)

    def mm(self, first, second):
        first, second = self.np.asarray(first), self.np.asarray(second)
        if (any(max(x.shape, default=1) > 64 for x in (first, second))
                or any(x.ndim not in (1, 2) for x in (first, second))):
            raise ValueError("explicit reference products are limited to dimension 64")
        self.preprocessing("explicit_reference_matrix_products", 4096)
        return first @ second

    def snapshot(self, circuit):
        if type(circuit) is not self.QuantumCircuit or float(circuit.global_phase) != 0:
            raise ValueError("the source is not a native zero-phase QuantumCircuit")
        records, signature = [], []
        for index, instruction in enumerate(circuit.data):
            if index % 1024 == 0:
                self.checkpoint()
            op = instruction.operation
            if op.name not in self.native:
                raise ValueError("the source contains a nonprimitive instruction")
            base, types, qa, ca, pa = self.native[op.name]
            wires = [circuit.find_bit(q).index for q in instruction.qubits]
            clbits = [circuit.find_bit(c).index for c in instruction.clbits]
            if (op.base_class is not base or type(op) not in types
                    or len(wires) != qa or len(set(wires)) != qa
                    or len(clbits) != ca or len(op.params) != pa):
                raise ValueError("native base class, arity or operands changed")
            parameters = []
            for value in op.params:
                if type(value) not in (float, int, self.np.float64, self.np.float32):
                    raise ValueError("a source parameter is not an explicit native real")
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("a source parameter is nonfinite")
                parameters.append({"hex": value.hex(), "token": format(value, ".17g")})
            records.append({"name": op.name, "wires": wires, "clbits": clbits,
                            "params": parameters, "native_class": base.__name__})
            signature.append((op.name, tuple(wires), tuple(clbits), type(op), base,
                              tuple(p["hex"] for p in parameters)))
        phase = float(circuit.global_phase)
        return ({"num_qubits": circuit.num_qubits, "num_clbits": circuit.num_clbits,
                 "global_phase": {"hex": phase.hex(), "token": format(phase, ".17g")},
                 "operations": records},
                (circuit.num_qubits, circuit.num_clbits, phase.hex(), tuple(signature)))

    def bind_sources(self, sources, density_source):
        self.sources = dict(sources)
        self.held_sources = {**sources, "density_R2": density_source}
        records = {}
        for name, source in self.held_sources.items():
            record, signature = self.snapshot(source)
            records[name], self.signatures[name] = record, signature
            if name in PROGRAMS and len(record["operations"]) != PROGRAMS[name][3]:
                raise ValueError("source count differs from its immutable prediction")
        two = records["density_R2"]
        if (len(two["operations"]) != 3012 or two["num_clbits"] != 0
                or sum(op["name"] == "reset" for op in two["operations"]) != 25):
            raise ValueError("the R2 density derivative differs from the reset contract")
        if sum(len(records[name]["operations"]) for name in PROGRAMS) > CAPS["aggregate_source_primitives"]:
            raise ValueError("aggregate source count exceeds its fixed guard")
        return {name: records[name] for name in PROGRAMS}

    def guard_sources(self):
        authenticate_payload()
        for name, circuit in self.held_sources.items():
            if self.snapshot(circuit)[1] != self.signatures[name]:
                raise ValueError("a held native source signature changed")

    def evolve_column(self, column):
        if type(column) is not int or column != self.calls["statevector_evolutions"] or column >= 2:
            raise ValueError("only the two ordered basis-column evolutions are admitted")
        self.guard_sources()
        self.count("statevector_evolutions")
        initial = self.np.zeros(128, complex)
        initial[column] = 1
        with self.scoped("backend"):
            result = self.Statevector(initial).evolve(self.sources["dilation"])
        self.guard_sources()
        return result

    def evolve_density(self, initial):
        expected = self.np.zeros((128, 128), complex)
        expected[127, 127] = 1
        if initial.shape != (128, 128) or not self.np.array_equal(initial, expected):
            raise ValueError("only the R2 all-ones density input is admitted")
        self.guard_sources()
        self.count("external_density_evolutions")
        with self.scoped("backend"):
            result = self.DensityMatrix(initial).evolve(self.held_sources["density_R2"])
        self.guard_sources()
        return result

    def bind_powers(self, actual, reference):
        self.powers = tuple((m, hashlib.sha256(m.tobytes()).hexdigest(), m.dtype.str)
                            for m in (actual, reference))
        for matrix, _digest, _dtype in self.powers:
            matrix.flags.writeable = False

    def guard_powers(self):
        if self.powers is None:
            raise ValueError("the two fixed channel matrices are unbound")
        for matrix, digest, dtype in self.powers:
            if (matrix.shape != (4, 4) or matrix.dtype.str != dtype or matrix.flags.writeable
                    or hashlib.sha256(matrix.tobytes()).hexdigest() != digest):
                raise ValueError("a held 4 by 4 channel matrix changed")

    def matrix_power(self, matrix, exponent):
        self.guard_sources()
        self.guard_powers()
        index = self.calls["matrix_power_calls"]
        if (index >= 2 or matrix is not self.powers[index][0]
                or type(exponent) is not int or exponent != 64):
            raise ValueError("only the ordered two exponent-64 channel powers are admitted")
        self.count("matrix_power_calls")
        with self.scoped("backend"):
            result = self.np.linalg.matrix_power(matrix, exponent)
        self.guard_sources()
        self.guard_powers()
        return result

    def complete(self):
        self.checkpoint()
        self.guard_sources()
        self.guard_powers()
        if self.calls != EXPECTED_CALLS:
            raise ValueError("the fixed direct scientific inventory is incomplete")


def reproduce(work, payload):
    import finite_gibbs_angle_provenance as angles
    import finite_gibbs_interval as interval
    from finite_gibbs_reference import _full_reference, _matrix_record, _max_entry, _superoperator
    np = work.np
    sources = {name: load_native_qasm(payload["circuits/" + row[0]].decode("utf-8"), row, work)
               for name, row in PROGRAMS.items()}
    two = sources["preparation_R2"]
    last = two.data[-1]
    if (last.operation.base_class is not work.native["measure"][0]
            or [two.find_bit(q).index for q in last.qubits] != [0]
            or [two.find_bit(c).index for c in last.clbits] != [0]):
        raise ValueError("the R2 source must end in exactly one system Z measurement")
    density_source = work.QuantumCircuit(7)
    for instruction in two.data[:-1]:
        if instruction.clbits or instruction.operation.name == "measure":
            raise ValueError("only the terminal system measurement may be removed")
        density_source.append(instruction.operation, [two.find_bit(q).index for q in instruction.qubits])
    snapshots = work.bind_sources(sources, density_source)
    qasms = {name: payload["circuits/" + row[0]].decode("utf-8") for name, row in PROGRAMS.items()}
    evaluator = interval.Evaluator(checkpoint=work.checkpoint)
    work.guard_sources()
    provenance = angles.verify_angle_provenance(snapshots, evaluator, qasms)
    if (type(provenance) is not dict or set(provenance) != {"angle_error_sum", "realization_bound", "ledger"}
            or not isinstance(provenance["angle_error_sum"], interval.Interval)):
        raise ValueError("the complete angle-ledger return contract differs")
    work.observations["angle_provenance"] = provenance["ledger"]
    work.guard_sources()
    # The following numerical orchestration is an original _run excerpt, with
    # only its two backend dispatches adapted to the portable direct-call guards.
    actual = np.empty((64, 2, 2), complex)
    for column in range(2):
        initial = np.zeros(128, complex)
        initial[column] = 1
        evolved = work.evolve_column(column)
        work.observations["actual_basis_columns"].append(_matrix_record(np, evolved.data))
        actual[:, :, column] = np.asarray(evolved.data).reshape(64, 2)
    work.observations["matrices"]["actual_all64_kraus"] = _matrix_record(np, actual)
    mesh, full_u, reference = _full_reference(work)
    work.observations["matrices"]["reference_all64_kraus"] = _matrix_record(np, reference)
    work.observations["matrices"]["reference_full_U"] = _matrix_record(np, full_u)
    actual_channel, reference_channel = _superoperator(np, actual), _superoperator(np, reference)
    work.observations["matrices"]["actual_channel"] = _matrix_record(np, actual_channel)
    work.observations["matrices"]["reference_channel"] = _matrix_record(np, reference_channel)
    work.bind_powers(actual_channel, reference_channel)
    plus = np.array([[.5, .5], [.5, .5]], complex)
    actual_power = work.matrix_power(actual_channel, 64)
    reference_power = work.matrix_power(reference_channel, 64)
    actual_final = work.mm(actual_power, plus.reshape(4, order="F")).reshape(2, 2, order="F")
    reference_final = work.mm(reference_power, plus.reshape(4, order="F")).reshape(2, 2, order="F")
    work.observations["matrices"]["actual_R64_state"] = _matrix_record(np, actual_final)
    work.observations["matrices"]["reference_R64_state"] = _matrix_record(np, reference_final)
    expected_two = work.mm(actual_channel, work.mm(actual_channel, plus.reshape(4, order="F"))).reshape(2, 2, order="F")
    all_ones = np.zeros((128, 128), complex)
    all_ones[127, 127] = 1
    density = work.evolve_density(all_ones)
    joint = np.asarray(density.data, complex)
    work.observations["matrices"]["R2_joint_state"] = _matrix_record(np, joint)
    expected_joint = np.zeros((128, 128), complex)
    expected_joint[:2, :2] = expected_two
    reduced = sum((joint[2 * e:2 * e + 2, 2 * e:2 * e + 2] for e in range(64)),
                  np.zeros((2, 2), complex))
    work.observations["matrices"]["R2_expected_joint_state"] = _matrix_record(np, expected_joint)
    work.observations["matrices"]["R2_reduced_state"] = _matrix_record(np, reduced)
    actual_effect = sum((work.mm(block.conj().T, block) for block in actual), np.zeros((2, 2), complex))
    reference_effect = sum((work.mm(block.conj().T, block) for block in reference), np.zeros((2, 2), complex))
    residuals = {
        "all64_kraus_entry_error": _max_entry(np, actual - reference),
        "actual_kraus_completeness_entry_error": _max_entry(np, actual_effect - np.eye(2)),
        "reference_kraus_completeness_entry_error": _max_entry(np, reference_effect - np.eye(2)),
        "reference_full_U_unitarity_entry_error": _max_entry(np, work.mm(full_u.conj().T, full_u) - np.eye(64)),
        "actual_reference_channel_entry_error": _max_entry(np, actual_channel - reference_channel),
        "actual_reference_R64_state_entry_error": _max_entry(np, actual_final - reference_final),
        "R2_joint_entry_error": _max_entry(np, joint - expected_joint),
        "R2_reduced_entry_error": _max_entry(np, reduced - expected_two),
        "R2_joint_trace_error": float(abs(np.trace(joint) - 1)),
        "R64_actual_trace_error": float(abs(np.trace(actual_final) - 1)),
        "R64_reference_trace_error": float(abs(np.trace(reference_final) - 1)),
        "R64_actual_hermiticity_error": _max_entry(np, actual_final - actual_final.conj().T),
        "reference_nonjump_offdiagonal_error": _max_entry(np, reference[:32, (0, 1), (1, 0)]),
        "reference_Z_jump_offdiagonal_error": _max_entry(np, reference[48:, (0, 1), (1, 0)]),
        "reference_X_jump_diagonal_error": _max_entry(np, reference[32:48, (0, 1), (0, 1)]),
    }
    work.observations["raw_residuals"] = dict(residuals)
    work.guard_sources()
    work.count("fixed_interval_certificate_evaluations")
    certificate = interval.evaluate_fixed_output(evaluator, provenance["angle_error_sum"])
    work.observations["interval_certificate"] = certificate
    for key in ("d", "u", "z_D", "mu_beta"):
        if (key not in certificate["intervals"]
                or set(certificate["intervals"][key]) != {"lo", "hi"}):
            raise ValueError("interval report lacks required exact-reference quantities")
    d_raw = float(np.sum(np.abs(reference[32:48, 1, 0]) ** 2) / .125)
    u_raw = float(np.sum(np.abs(reference[32:48, 0, 1]) ** 2) / .125)
    z_reference = float((reference_final[0, 0] - reference_final[1, 1]).real)

    def distance_to_interval(value, key):
        bounds = certificate["intervals"][key]
        lower, upper = float(bounds["lo"]), float(bounds["hi"])
        return max(lower - value, 0.0, value - upper)

    residuals["reference_d_vs_exact_interval_diagnostic"] = distance_to_interval(d_raw, "d")
    residuals["reference_u_vs_exact_interval_diagnostic"] = distance_to_interval(u_raw, "u")
    residuals["reference_R64_Z_vs_exact_interval_diagnostic"] = distance_to_interval(z_reference, "z_D")
    implementation_checks = {name: value <= 1e-9 for name, value in residuals.items()}
    work.observations["raw_residuals"] = dict(residuals)
    work.guard_sources()
    work.complete()
    observed = all(implementation_checks.values()) and certificate.get("certificate_passed") is True
    return {
        "schema": "openquantum.finite_gibbs.portable_reproduction.v1",
        "status": "observed_fixed_finite_Gibbs_Z_components",
        "scientific_source_commit": SOURCE_COMMIT,
        "packaging_authored_as": "source-only and unexecuted",
        "parameters": dict(PARAMETERS), "tolerances": {"entry": 1e-9, "trace_completeness": 1e-9},
        "runtime_versions": {"python": sys.version.split()[0], **VERSIONS},
        "counters": dict(work.calls), "preprocessing_counters": dict(work.preprocess),
        "source_counts": {name: row[3] for name, row in PROGRAMS.items()},
        "density_source": {"primitives": 3012, "resets": 25, "terminal_measures_removed": 1},
        "internal_reset_dispatches_measured": False,
        "raw_residuals": residuals, "implementation_checks": implementation_checks,
        "implementation_checks_passed": all(implementation_checks.values()),
        "interval_certificate": certificate, "angle_provenance": provenance["ledger"],
        "matrices": work.observations["matrices"],
        "actual_basis_columns": work.observations["actual_basis_columns"],
        "raw_population_diagnostics": {
            "reference_d": d_raw, "reference_u": u_raw,
            "actual_R64_Z": float((actual_final[0, 0] - actual_final[1, 1]).real),
            "reference_R64_Z": z_reference},
        "mesh_record": {"signed_indices": mesh["signed_indices"].tolist(),
                        "times": mesh["times"].tolist(), "frequencies": mesh["frequencies"].tolist(),
                        "amplitudes": mesh["amplitudes"].tolist(), "gamma": mesh["gamma"].tolist(),
                        "eta": float(mesh["eta"])},
        "measurement_projection": {"samples": 16384, "sampling_performed": False,
                                   "whole_preparation_primitives_each": 96323,
                                   "total_work_formula": "16384*(64*1505+3)",
                                   "iid_ideal_Born_assumption": True},
        "portable_work_observations": {
            "cooperative_limits": dict(CAPS),
            "wall_seconds": time.monotonic() - work.start_wall,
            "cpu_seconds": time.process_time() - work.start_cpu,
            "sampled_peak_rss_bytes": work.sampled_peak_rss,
            "continuous_process_lifetime_or_resource_certificate": False,
            "native_runtime_files_authenticated": False,
            "SDK_internal_kernels_measured": False},
        "source_realization": "authenticated canonical primitives; no clipping, trace renormalization or phase fitting",
        "accuracy_scope": "fixed ideal Gibbs Z mean conditional on complete angle/interval obligations and future iid ideal Born readouts",
        "finite_gibbs_Z_output_bound_observed": observed,
        "finite_gibbs_Z_output_bound_passed": False,
        "separate_acceptance_required": True,
        "whole_invocation_validated": False,
        "full_R64_primitive_simulation_performed": False,
        "random_sampling_performed": False,
        "floating_point_error_certified": False,
        "rotations_synthesized": False,
        "physical_resources_computed": False,
        "formal_proof_reproduced": False,
        "excluded": ["full-state Gibbs accuracy", "general sampler or mixing theorem",
                     "uniform continuum/KMS bridge", "rotation synthesis", "physical faults and timing",
                     "FT resource advantage", "floating backend rounding certification", "mechanized proof"],
    }


def write_json(output, name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    existing_bytes = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
    if len(raw) + existing_bytes > CAPS["combined_output_bytes"]:
        raise ValueError("portable output exceeds the fixed combined byte limit")
    with (output / name).open("xb") as stream:
        stream.write(raw)
    return {"file": name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="distinct absent directory for this new bounded reproduction")
    args = parser.parse_args()
    if sys.version_info[:3] != (3, 12, 14):
        raise ValueError("this fixed reproduction requires Python 3.12.14")
    for name, expected in VERSIONS.items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("a scientific package version differs from the bundled lock")
    payload = authenticate_payload()
    # Isolated Python omits the script directory from sys.path. Only the
    # authenticated adjacent scientific fragments are added for this run.
    sys.path.insert(0, str(ROOT / "sources"))
    output = Path(args.output_dir).resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("new results must be outside the immutable downloaded payload")
    output.mkdir(parents=False, exist_ok=False)
    try:
        work = Work()
        result = reproduce(work, payload)
        item = write_json(output, "results.json", result)
        write_json(output, "reproduction-manifest.json", {
            "schema": "openquantum.finite_gibbs.portable_reproduction_manifest.v1",
            "scientific_source_commit": SOURCE_COMMIT, "input_sha256": dict(INPUT_PINS),
            "payloads": [item], "original_saved_evidence_unchanged": True,
            "separate_acceptance_required": True, "whole_invocation_validated": False})
        print("Bounded portable observations written; separate scientific acceptance is required.")
        return 0 if result["finite_gibbs_Z_output_bound_observed"] else 1
    except Exception as error:
        write_json(output, "failure.json", {
            "schema": "openquantum.finite_gibbs.portable_failure.v1",
            "status": "portable_reproduction_failed", "error_type": type(error).__name__,
            "whole_invocation_validated": False, "scientific_acceptance": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
