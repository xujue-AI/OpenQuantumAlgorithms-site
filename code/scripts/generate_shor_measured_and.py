#!/usr/bin/env python3
"""Generate original measured-AND circuits into an absent bounded directory.

This command never calls a physical estimator or rotation synthesizer. Small
density/exact checks and large circuit materialization have separate flags.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-measured-and"
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, qasm3
from openquantum import shor_measured_and as measured

MAX_EXPORT_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
SOURCE_PATHS = (
    "scripts/generate_shor_measured_and.py", "src/openquantum/shor_measured_and.py",
    "src/openquantum/shor.py", "src/openquantum/shor_ripple.py",
    "src/openquantum/__init__.py", "src/openquantum/grover.py",
    "requirements-lock.txt", "LICENSE",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Output:
    """Exclusive flat writes bound to one owned directory descriptor."""
    def __init__(self, path: Path):
        target = path.absolute()
        resolved = target.resolve()
        if resolved.is_relative_to(ROOT):
            allowed = ROOT / "assets/generated/shor/measured-and"
            if resolved != allowed and not resolved.is_relative_to(ROOT / ".cache"):
                raise ValueError("refusing source or unrelated repository output")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.mkdir(exist_ok=False)
        self.path = target
        self.fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.identity = os.fstat(self.fd)
        self.inventory = {}
        self.identities = {}
        self.total = 0

    def write(self, name: str, data: bytes):
        if name != "__init__.py" and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name):
            raise ValueError("export names are bounded flat components")
        if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES or self.total + len(data) > MAX_EXPORT_BYTES:
            raise ValueError("scientific export byte cap exceeded")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        try:
            position = 0
            while position < len(data):
                amount = os.write(fd, data[position:])
                if amount <= 0:
                    raise OSError("export write made no progress")
                position += amount
            result = os.fstat(fd)
            if not stat.S_ISREG(result.st_mode) or result.st_size != len(data):
                raise OSError("export file identity or size changed")
        finally:
            os.close(fd)
        self.inventory[name] = {"bytes": len(data), "sha256": digest(data)}
        self.identities[name] = (result.st_dev, result.st_ino)
        self.total += len(data)

    def json(self, name, value):
        self.write(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def finish(self, manifest):
        manifest["files"] = dict(self.inventory)
        manifest["payload_bytes"] = self.total
        self.json("manifest.json", manifest)
        for name, expected in self.inventory.items():
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.fd)
            try:
                before = os.fstat(fd)
                identity = self.identities[name]
                if (not stat.S_ISREG(before.st_mode)
                        or (before.st_dev, before.st_ino) != identity
                        or before.st_size != expected["bytes"]):
                    raise OSError("generated payload identity or size changed")
                hasher, size = hashlib.sha256(), 0
                while True:
                    raw = os.read(fd, min(65536, expected["bytes"] + 1 - size))
                    if not raw:
                        break
                    size += len(raw)
                    if size > expected["bytes"]:
                        raise OSError("generated payload grew during authentication")
                    hasher.update(raw)
                after = os.fstat(fd)
                named = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
                if (size != expected["bytes"] or hasher.hexdigest() != expected["sha256"]
                        or (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                        != (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                        or (named.st_dev, named.st_ino) != identity):
                    raise OSError("generated payload failed final authentication")
            finally:
                os.close(fd)
        named = os.stat(self.path, follow_symlinks=False)
        if (named.st_dev, named.st_ino) != (self.identity.st_dev, self.identity.st_ino):
            raise OSError("owned output directory was replaced")
        os.close(self.fd)
        self.fd = None

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def source_record():
    result, retained = {}, {}
    for relative in SOURCE_PATHS:
        raw = bounded_source_bytes(ROOT / relative)
        retained[relative] = raw
        result[relative] = {"bytes": len(raw), "sha256": digest(raw)}
    commit = None
    # A downloaded package must not borrow an ancestor checkout identity.
    if (ROOT / ".git").exists():
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                             env=environment, text=True, capture_output=True, timeout=5, check=True).stdout.strip()
        if Path(top).resolve() != ROOT:
            raise ValueError("Git source root does not match this package")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                env=environment, text=True, capture_output=True, timeout=5, check=True).stdout.strip()
    return result, commit, retained


def bounded_source_bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            raise ValueError("source snapshot must be a bounded regular file")
        parts, size = [], 0
        while True:
            raw = os.read(fd, min(65536, MAX_FILE_BYTES + 1 - size))
            if not raw:
                break
            size += len(raw)
            if size > MAX_FILE_BYTES:
                raise ValueError("source snapshot byte cap exceeded")
            parts.append(raw)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or (named.st_dev, named.st_ino) != (before.st_dev, before.st_ino)
                or size != before.st_size):
            raise OSError("source snapshot changed during read")
        return b"".join(parts)
    finally:
        os.close(fd)


def figure_bytes(figure):
    stream = io.BytesIO()
    figure.savefig(stream, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return ("\n".join(line.rstrip() for line in stream.getvalue().decode().splitlines()) + "\n").encode()


def qasm_bytes(circuit):
    text = qasm3.dumps(circuit)
    return ("\n".join(line.rstrip() for line in text.splitlines()) + "\n").encode()


def generate(output: Path, *, width=2, a=3, b=1, carry=0, suite=False, simulate=True):
    if type(suite) is not bool or type(simulate) is not bool:
        raise ValueError("suite and simulate must be Boolean")
    if type(width) is not int or not 1 <= width <= measured.MAX_WIDTH:
        raise ValueError(f"width must be an integer in 1..{measured.MAX_WIDTH}")
    for label, value in (("a", a), ("b", b)):
        if type(value) is not int or not 0 <= value < (1 << width):
            raise ValueError(f"{label} must be an integer fitting width bits")
    if type(carry) is not int or carry not in (0, 1):
        raise ValueError("carry must be zero or one")
    if simulate and not suite and width > 3:
        raise ValueError("addition simulation is limited to width <= 3")
    sources, commit, retained_sources = source_record()
    start = time.perf_counter()
    destination = Output(output)
    try:
        record = {"method": "four-T measured CCX replacement in unoptimized ripple arithmetic",
                  "physical_estimation_status": "not computed",
                  "formal_proof_status": "not reproduced",
                  "gadget": measured.gadget_checks() if simulate else {"performed": False},
                  "addition_cases": [], "order_cases": [], "generated_only": []}
        for clear_mode in ("feedback_x", "reset"):
            circuit = measured.build_measured_ccx(clear_mode=clear_mode)
            destination.write(f"gadget-{clear_mode}.qasm", qasm_bytes(circuit))
        baseline = QuantumCircuit(3)
        measured.append_seven_t_ccx(baseline, 0, 1, 2)
        destination.write("gadget-static.qasm", qasm_bytes(baseline))
        destination.write("gadget.svg", figure_bytes(measured.build_measured_ccx().draw(output="mpl", fold=30)))
        cases = [(width, a, b, carry)] if not suite else [(1, 1, 1, 1), (2, 3, 1, 0), (3, 5, 3, 1)]
        for w, first, second, incoming in cases:
            for method in ("static", "measured"):
                circuit = measured.build_addition_demo(w, first, second, incoming, method=method)
                name = f"addition-w{w}-{method}"
                destination.write(name + ".qasm", qasm_bytes(circuit))
                result = measured.check_addition(w, first, second, incoming, method=method) if simulate else {"performed": False}
                counts = measured.logical_counts(circuit)
                expected = measured.addition_scaling(w, first, second, incoming)
                if counts["literal_t_or_tdg"] != expected["static" if method == "static" else "measured_feedback_x"]["literal_t_or_tdg"]:
                    raise AssertionError("materialized literal T counts disagree with source formula")
                record["addition_cases"].append({"width": w, "a": first, "b": second, "carry": incoming,
                     "method": method, "counts": counts, "validation": result, "formula": expected,
                     "ideal_output": "deterministic (a,b+a+carry modulo2**width,carry)",
                     "repetitions": 1, "physical_failure_allowance": "not allocated"})
                if w == 2 and method == "measured":
                    # Show each real feedback/readout bit; bundled inner-register
                    # lookup in Qiskit 2.6.2 cannot draw this composed instrument.
                    destination.write("addition.svg", figure_bytes(circuit.draw(output="mpl", fold=32, scale=.75, cregbundle=False)))
        rows = []
        for w in ((1, 2, 3, 4, 8, 16, 24, 31) if suite else (width,)):
            for method in ("static", "measured"):
                circuit = measured.build_addition_demo(w, 0, 0, 0, method=method)
                counts = measured.logical_counts(circuit)
                rows.append({"width": w, "method": method, "qubits": counts["logical_qubits"],
                    "literal_t_or_tdg": counts["literal_t_or_tdg"],
                    "primitive_all_zero": counts["all_zero_feedback_primitive_count"],
                    "primitive_all_one": counts["all_one_feedback_primitive_count"],
                    "feedback_decisions": counts["classical_feedback_decisions"],
                    "dependency_depth_all_one": counts["dependency_depth_all_one_feedback"],
                    "numerical_execution_at_this_width": False,
                    "physical_estimate": "not computed"})
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        destination.write("scaling.csv", text.getvalue().encode())
        if suite:
            for modulus in (15, 21):
                for method in ("static", "measured"):
                    circuit = measured.build_order_finding_measured(modulus, 2, 4, method=method)
                    destination.write(f"order-N{modulus}-t4-{method}.qasm", qasm_bytes(circuit))
                validation = measured.check_order_finding(modulus, 2, 4) if simulate else {"performed": False}
                record["order_cases"].append({"modulus": modulus, "base": 2, "phase_bits": 4,
                       "validation": validation,
                       "physical_estimate": "not computed; P/CP synthesis and full controller schedule remain unpriced"})
            for method in ("static", "measured"):
                addition = measured.build_addition_demo(24, 42, 7, 0, method=method)
                destination.write(f"addition-w24-{method}.qasm", qasm_bytes(addition))
                order = measured.build_order_finding_measured(4093, 2, 1, method=method)
                destination.write(f"order-N4093-t1-{method}.qasm", qasm_bytes(order))
                record["generated_only"].extend([
                    {"task": "addition", "width": 24, "method": method, "counts": measured.logical_counts(addition), "validated_quantum_simulation": False},
                    {"task": "order", "modulus": 4093, "base": 2, "phase_bits": 1, "method": method,
                     "counts": measured.logical_counts(order), "validated_quantum_simulation": False}])
        record["generation_and_checks_wall_seconds"] = time.perf_counter() - start
        record["source_commit"] = commit
        record["source_commit_scope"] = "actual own Git root when present; exact source hashes are authoritative, no borrowed ancestor identity"
        record["source_sha256"] = sources
        record["environment"] = {"python": platform.python_version(), "device": "CPU", "seed": None,
              "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")}}
        record["sparse_angle_scope"] = "intended dyadic pi/8 logical angles; exact ring computation is not a binary-angle realization or proof-kernel certificate"
        record["circuit_materialization_and_simulation_separate"] = True
        record["excluded"] = ["physical code/factories/runtime/fault success", "controller/reset/readout hardware calibration",
                              "generic factoring success and classical processing time", "P/CP synthesis in order circuits",
                              "optimized 4w-4 Gidney adder", "mechanized proof and uniform floating enclosure"]
        destination.json("study.json", record)
        for relative in SOURCE_PATHS:
            name = "LICENSE.txt" if relative == "LICENSE" else Path(relative).name
            destination.write(name, retained_sources[relative])
        destination.write("README.md", (
            "# Original measured-AND arithmetic\n\n"
            "The small circuits and source counts retain initialization, helper measurement, conditional CZ/X clearing and final readout. "
            "Literal T savings are not physical speedup. The unoptimized MAJ/UMA adder has 8w measured T gates versus 14w static T gates; "
            "this is not Gidney's optimized 4w-4 adder.\n\n"
            "Large exports are circuit materializations, not validated large quantum runs. "
            "Exact sparse checks use the intended dyadic pi/8 angle model, not a formal proof or a certified floating/synthesis enclosure.\n\n"
            "Use the full public code package for the src/openquantum and scripts layout. Flat snapshots here identify exact sources; "
            "they are not an automatically installed package. No private controller, approvals, logs or credentials are included.\n\n"
            "References: Gidney, Quantum 2,74 (2018), doi:10.22331/q-2018-06-18-74. "
            "Qiskit2.6.2 (Apache-2.0) supplies standard gates; all wrapper code, explanations and figures are original MIT work.\n"
        ).encode())
        manifest = {"schema": "openquantum-shor-measured-and-v1", "producer_commit": commit,
                    "status": "generated_scoped_checks" if simulate else "generated_only",
                    "source_files": sources, "simulate": simulate, "suite": suite,
                    "physical_or_formal_claim": False,
                    "limits": {"total_export_bytes": MAX_EXPORT_BYTES, "file_bytes": MAX_FILE_BYTES,
                               "qubits": measured.MAX_QUBITS, "dense_qubits": measured.MAX_DENSE_QUBITS,
                               "exact_phase_bits": measured.MAX_EXACT_PHASE_BITS,
                               "exact_states": measured.MAX_EXACT_STATES,
                               "exact_gate_state_work": measured.MAX_EXACT_GATE_STATE_WORK}}
        for relative, expected in sources.items():
            current = bounded_source_bytes(ROOT / relative)
            if {"bytes": len(current), "sha256": digest(current)} != expected:
                raise OSError("source changed during generation")
        destination.finish(manifest)
        return record
    finally:
        destination.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/shor/measured-and")
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--a", type=int, default=3)
    parser.add_argument("--b", type=int, default=1)
    parser.add_argument("--carry", type=int, default=0)
    parser.add_argument("--suite", action="store_true")
    parser.add_argument("--no-sim", action="store_true")
    args = parser.parse_args()
    result = generate(args.output_dir, width=args.width, a=args.a, b=args.b, carry=args.carry,
                      suite=args.suite, simulate=not args.no_sim)
    print(json.dumps({"status": "completed_generation", "addition_cases": len(result["addition_cases"]),
                      "order_cases": len(result["order_cases"]), "physical_estimate": "not computed"}))


if __name__ == "__main__":
    main()
