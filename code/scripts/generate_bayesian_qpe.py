#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded original grid-Bayesian QPE study and separate source generation.

The prescribed study has four known pedagogical phases, two seeded streams
per phase and 48 fresh single-shot observations per output. No Native,
rotation synthesis, hardware or general application solver is invoked.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
PORTABLE_INPUTS = ("bayesian_qpe.py", "requirements-lock.txt", "LICENSE.txt")
PORTABLE_LAYOUT = any((SCRIPT.parent / name).exists() or (SCRIPT.parent / name).is_symlink()
                      for name in PORTABLE_INPUTS)
if PORTABLE_LAYOUT and not all((SCRIPT.parent / name).is_file() for name in PORTABLE_INPUTS):
    raise ValueError("Portable execution requires all four sibling source, script, lock and license files")
if not PORTABLE_LAYOUT and (SCRIPT.parent.name != "scripts"
                           or not all((SCRIPT.parents[1] / name).is_file() for name in
                                      ("src/openquantum/bayesian_qpe.py", "requirements-lock.txt", "LICENSE"))):
    raise ValueError("Execution requires the complete project scripts layout or four sibling portable files")
ROOT = SCRIPT.parent if PORTABLE_LAYOUT else SCRIPT.parents[1]
SOURCE = SCRIPT.parent / "bayesian_qpe.py" if PORTABLE_LAYOUT else ROOT / "src/openquantum/bayesian_qpe.py"


def bounded_source_bytes(path: Path) -> bytes:
    """Capture a bounded regular payload without following a leaf symlink."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= 1024 * 1024:
            raise ValueError("Source, lock and license must be bounded regular files")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            raw = handle.read(1024 * 1024 + 1)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(named):
            raise ValueError("Source payload changed during bounded read")
        return raw
    finally:
        os.close(fd)


SOURCE_BYTES = bounded_source_bytes(SOURCE)
spec = importlib.util.spec_from_file_location("openquantum_bayesian_source", SOURCE)
bq = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bq
exec(compile(SOURCE_BYTES, str(SOURCE), "exec"), bq.__dict__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_WALL_SECONDS = 120
MAX_EXPORTED_CIRCUITS = 256
MAX_AGGREGATE_CIRCUIT_OPERATIONS = 200_000
PHASES = (.3, .995, .125, .5)
ROOT_SEEDS = (7, 19)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def source_files() -> dict[str, Path]:
    paths = {"bayesian_qpe.py": SOURCE, "generate_bayesian_qpe.py": SCRIPT}
    for destination, original in (("requirements-lock.txt", "requirements-lock.txt"), ("LICENSE.txt", "LICENSE")):
        path = SCRIPT.parent / destination if PORTABLE_LAYOUT else ROOT / original
        paths[destination] = path
    for path in paths.values():
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Complete bounded source, lock and license are required")
    return paths


def git_identity() -> str | None:
    """No provenance from an ancestor repository or caller Git selectors."""
    if PORTABLE_LAYOUT:
        return None
    marker = ROOT / ".git"
    if not (marker.is_file() or marker.is_dir()):
        return None
    environment = os.environ.copy()
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE", "GIT_NAMESPACE",
                 "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS"):
        environment.pop(name, None)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel", "HEAD"],
                                capture_output=True, text=True, timeout=5, check=False, env=environment)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.splitlines()
    if result.returncode != 0 or len(lines) != 2 or Path(lines[0]).resolve() != ROOT:
        return None
    return lines[1] if len(lines[1]) == 40 and all(c in "0123456789abcdef" for c in lines[1]) else None


class Writer:
    """Exclusive, bounded writes through the owned directory descriptor."""
    def __init__(self, output: Path, *, started: float):
        self.fd = None
        self.ancestor_handles = []
        self.ancestor_edges = []
        resolved = output.absolute().resolve()
        self.output = resolved
        if not resolved.name:
            raise ValueError("Output must name a new directory below the filesystem root")
        if resolved == ROOT or any(resolved == (ROOT / part).resolve() or resolved.is_relative_to((ROOT / part).resolve())
                                   for part in ("src", "scripts", "notebooks", "algorithms", "references", ".git")):
            raise ValueError("Output cannot replace source or repository data")
        generated = (ROOT / "assets/generated").resolve()
        if resolved.is_relative_to(generated) and not resolved.is_relative_to(generated / "bayesian-qpe"):
            raise ValueError("Output cannot replace another algorithm's records")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK
        try:
            self.ancestor_handles.append(os.open(self.output.anchor, flags))
            for part in self.output.parent.parts[1:]:
                parent = self.ancestor_handles[-1]
                try:
                    before = os.stat(part, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    os.mkdir(part, 0o755, dir_fd=parent)
                    before = os.stat(part, dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISDIR(before.st_mode):
                    raise ValueError("Output ancestors must be real directories")
                child = os.open(part, flags, dir_fd=parent)
                self.ancestor_handles.append(child)
                after = os.fstat(child)
                identity = (after.st_dev, after.st_ino)
                if (before.st_dev, before.st_ino) != identity:
                    raise ValueError("Output ancestor changed during binding")
                self.ancestor_edges.append((parent, part, child, identity))
            self.check_ancestors()
            self.parent_fd = self.ancestor_handles[-1]
            os.mkdir(self.output.name, 0o755, dir_fd=self.parent_fd)
            self.check_ancestors()
            self.fd = os.open(self.output.name, flags, dir_fd=self.parent_fd)
            self.identity = os.fstat(self.fd)
        except BaseException:
            self.close()
            raise
        self.inventory = {}
        self.total = 0
        self.started = started
        self.circuits = self.operations = 0

    def check_ancestors(self):
        for parent, part, child, identity in self.ancestor_edges:
            named = os.stat(part, dir_fd=parent, follow_symlinks=False)
            opened = os.fstat(child)
            if (not stat.S_ISDIR(named.st_mode)
                    or (named.st_dev, named.st_ino) != identity
                    or (opened.st_dev, opened.st_ino) != identity):
                raise ValueError("Owned output ancestor identity changed")

    def check(self):
        if time.perf_counter() - self.started > MAX_WALL_SECONDS:
            raise TimeoutError("Bayesian producer exceeded its 120-second wall cap")
        self.check_ancestors()
        named = os.stat(self.output.name, dir_fd=self.parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(named.st_mode) or (named.st_dev, named.st_ino) != (self.identity.st_dev, self.identity.st_ino):
            raise ValueError("Owned output directory identity changed")

    def write(self, name: str, raw: bytes):
        self.check()
        if not isinstance(raw, bytes) or not name or name in {".", ".."} or Path(name).name != name or name in self.inventory:
            raise ValueError("Payload names and byte values must be fixed and unique")
        if len(raw) + self.total > MAX_OUTPUT_BYTES:
            raise ValueError("Bayesian export exceeds its 16 MiB aggregate cap")
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=self.fd)
        try:
            with os.fdopen(fd, "wb", closefd=False) as handle:
                handle.write(raw)
                handle.flush()
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size != len(raw):
                raise ValueError("Exported file is not the expected regular payload")
        finally:
            os.close(fd)
        self.total += len(raw)
        self.inventory[name] = {"bytes": len(raw), "sha256": digest(raw)}

    def json(self, name: str, value):
        self.write(name, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def circuit(self, name: str, circuit):
        self.circuits += 1
        self.operations += len(circuit.data)
        if self.circuits > MAX_EXPORTED_CIRCUITS or self.operations > MAX_AGGREGATE_CIRCUIT_OPERATIONS:
            raise ValueError("Complete export exceeds circuit/instruction caps")
        raw = qasm3.dumps(circuit).encode()
        self.write(name, raw)
        return {"file": name, "logical_qubits": circuit.num_qubits, "classical_bits": circuit.num_clbits,
                "source_gate_counts": dict(circuit.count_ops()), "source_dependency_depth": circuit.depth(),
                "global_phase_radians": float(circuit.global_phase),
                "qasm_scope": "complete fresh measured trial; omitted whole-circuit global phase is readout-irrelevant, not a controlled-oracle specification"}

    def finish(self, sources):
        self.check()
        self.json("manifest.json", {"schema": "openquantum-bayesian-qpe-files-v1",
                                    "files": dict(self.inventory), "source_sha256": sources,
                                    "payload_bytes": self.total,
                                    "scope": "bounded scientific source/data; private execution controls and logs are excluded"})
        self.check()
        self.close()

    def close(self):
        fd = getattr(self, "fd", None)
        if fd is not None:
            self.fd = None
            os.close(fd)
        handles = getattr(self, "ancestor_handles", [])
        self.ancestor_handles = []
        for descriptor in reversed(handles):
            os.close(descriptor)

    def __del__(self):
        self.close()


def figure_bytes(figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    return buffer.getvalue()


def case_seed(root_seed: int, phase_index: int) -> int:
    if (isinstance(root_seed, bool) or not isinstance(root_seed, int) or not 0 <= root_seed < 1 << 63
            or isinstance(phase_index, bool) or not isinstance(phase_index, int) or not 0 <= phase_index < len(PHASES)):
        raise ValueError("Seed namespace requires bounded integer root seed and prescribed phase index")
    raw = hashlib.sha256(f"original-grid-bayes:{root_seed}:{phase_index}".encode()).digest()
    return int.from_bytes(raw[:8], "big") & ((1 << 63) - 1)


def independent_batch_posterior(record: dict) -> list[float]:
    """Direct scalar likelihood product, independent of recursive update code."""
    values = []
    grid = record["grid"]
    for phi in grid:
        weight = 1 / len(grid)
        for row in record["trace"]:
            p0 = (1 + math.cos(2 * math.pi * row["power"] * phi - row["offset"])) / 2
            weight *= p0 if row["outcome"] == 0 else 1 - p0
        values.append(weight)
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Independent batch reference has no finite positive evidence")
    return [value / total for value in values]


def generate(output: Path, *, single=False, phase=.3, target_qubits=1, steps=48,
             grid_size=512, seed=7, source_only=False, power=32) -> dict:
    started = time.perf_counter()
    if not isinstance(single, bool) or not isinstance(source_only, bool):
        raise ValueError("Study and source flags must be booleans")
    if not source_only and power != 32:
        raise ValueError("The inference candidate set ends at32; --power changes source-only materialization")
    if not single and not source_only and (phase, target_qubits, steps, grid_size, seed, power) != (.3, 1, 48, 512, 7, 32):
        raise ValueError("The prescribed study has fixed defaults; use --single for another bounded request")
    paths = source_files()
    source_payloads = {name: bounded_source_bytes(path) for name, path in paths.items()}
    source_hashes = {name: digest(raw) for name, raw in source_payloads.items()}
    if source_hashes["bayesian_qpe.py"] != digest(SOURCE_BYTES):
        raise ValueError("Current source differs from the exact bytes used by this producer")
    provenance = {"source_commit_at_generation": git_identity(), "source_sha256": source_hashes,
                  "python": platform.python_version(), "device": "CPU",
                  "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")},
                  "hardware_execution": False, "native_calls": 0, "synthesis_calls": 0}
    if source_only:
        # Bound all scalar inputs and materialization before opening output.
        circuit = bq.build_experiment(target_qubits, phase, power, math.pi / 2)
        writer = Writer(output, started=started)
        source = writer.circuit("source-only.qasm", circuit)
        compiled = writer.circuit("source-only-lowered.qasm", bq.lower_experiment(circuit))
        result = {"schema": "openquantum-bayesian-qpe-study-v1", "status": "generated_only_unvalidated_at_requested_size",
                  "source": source, "lowered": compiled, "provenance": provenance,
                  "nominal_phase": phase, "power": power, "inference_performed": False,
                  "simulation_performed": False, "physical_estimate_performed": False}
        writer.json("results.json", result)
    else:
        phases = (phase,) if single else PHASES
        root_seeds = (seed,) if single else ROOT_SEEDS
        if (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= bq.MAX_STEPS
                or isinstance(target_qubits, bool) or not isinstance(target_qubits, int)
                or not 1 <= target_qubits <= bq.MAX_DENSE_TARGET_QUBITS):
            raise ValueError("Inference requires bounded steps and at most three target qubits")
        bq.phase_grid(grid_size)
        if len(phases) * len(root_seeds) * (steps + 2) * grid_size * 48 > 14 * 1024 * 1024:
            raise ValueError("Requested full posterior histories exceed the conservative serialized-data preflight")
        writer = Writer(output, started=started)
        records, bank = [], {}
        for phase_index, phi in enumerate(phases):
            for root_seed in root_seeds:
                writer.check()
                actual_seed = case_seed(root_seed, phase_index)
                record = bq.run_adaptive(phi, target_qubits=target_qubits, steps=steps,
                                         grid_size=grid_size, seed=actual_seed)
                record["root_seed"] = root_seed
                record["seed_namespace"] = "SHA256(original-grid-bayes:root_seed:phase_index), first 63 bits; not an IID hardware certificate"
                reference = independent_batch_posterior(record)
                residual = max(abs(a - b) for a, b in zip(reference, record["final_posterior"]))
                if residual > 2e-11:
                    raise AssertionError("Recursive posterior differs from independent batch likelihood")
                record["independent_batch_posterior_max_error"] = residual
                for row in record["trace"]:
                    key = (phase_index, row["power"], row["offset"])
                    if key not in bank:
                        label = f"phase{phase_index}-k{row['power']}-offset{bq.DEFAULT_OFFSETS.index(row['offset'])}"
                        circuit = bq.build_experiment(target_qubits, phi, row["power"], row["offset"])
                        bank[key] = {"nominal_phase": phi, "power": row["power"], "offset": row["offset"],
                                     "source": writer.circuit(label + ".qasm", circuit),
                                     "lowered": writer.circuit(label + "-lowered.qasm", bq.lower_experiment(circuit))}
                records.append(record)
        demo = bq.build_experiment(1, .3, 2, math.pi / 2)
        demo_check = bq.check_experiment(1, .3, 2, math.pi / 2)
        writer.circuit("demo.qasm", demo)
        writer.write("circuit.svg", figure_bytes(demo.draw(output="mpl", fold=-1, idle_wires=False)))
        large = bq.build_experiment(8, .3, 32, math.pi / 2)
        large_record = writer.circuit("large-n8-k32.qasm", large)
        result = {"schema": "openquantum-bayesian-qpe-study-v1", "status": "checked_prescribed_ideal_grid_bayes_study",
                  "records": records, "circuit_bank": list(bank.values()), "demo_check": demo_check,
                  "large_source_only": {**large_record, "simulation_performed": False}, "provenance": provenance,
                  "total_modeled_fresh_trials": sum(r["ledger"]["complete_fresh_preparations"] for r in records),
                  "total_literal_controlled_U_calls": sum(r["ledger"]["shot_weighted_controlled_U_calls"] for r in records),
                  "uniform_confidence_certificate": False, "physical_estimate_performed": False,
                  "scope": "known-phase correctness examples; fixed-length finite-grid heuristic with ideal classical draws and explicit circuits"}
        writer.json("results.json", result)
        table = io.StringIO()
        fieldnames = ["nominal_phase", "root_seed", "estimate", "circular_error", "credible_cell_width", "controlled_U_calls", "fresh_trials"]
        csv_writer = csv.DictWriter(table, fieldnames=fieldnames, lineterminator="\n")
        csv_writer.writeheader()
        for record in records:
            csv_writer.writerow({"nominal_phase": record["nominal_phase"], "root_seed": record["root_seed"],
                                 "estimate": record["estimate"], "circular_error": record["circular_error"],
                                 "credible_cell_width": record["summary"]["credible_arc"]["width"],
                                 "controlled_U_calls": record["ledger"]["shot_weighted_controlled_U_calls"],
                                 "fresh_trials": record["ledger"]["complete_fresh_preparations"]})
        writer.write("summary.csv", table.getvalue().encode())
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        chosen = records[0]
        axes[0].plot(chosen["grid"], chosen["final_posterior"], color="#3366aa")
        axes[0].axvline(chosen["actual_angle_phase_reference"], color="#bc5a48", linestyle="--", label="Actual-angle reference")
        axes[0].set(xlabel="Phase on the circle", ylabel="Posterior atom mass", title="One prescribed final posterior")
        axes[0].legend(fontsize=8)
        entropy = [-math.fsum(w * math.log(w) for w in row["posterior_weights"] if w > 0) for row in chosen["trace"]]
        axes[1].plot(range(1, len(entropy) + 1), entropy, color="#3366aa")
        axes[1].set(xlabel="Fresh observations", ylabel="Entropy (nats)", title="Observed entropy can increase")
        figure.tight_layout()
        writer.write("posterior.svg", figure_bytes(figure))
    for name, path in paths.items():
        raw = bounded_source_bytes(path)
        if digest(raw) != source_hashes[name]:
            raise ValueError("Scientific source changed during generation")
        writer.write(name, source_payloads[name])
    writer.write("README.md", ("# Original bounded Bayesian QPE artifacts\n\n"
                 "Explicit source circuits and posterior data accompany the fixed finite-grid method.\n"
                 "Credible cell arcs are model summaries, not uniform confidence or hardware certificates.\n"
                 "Resets, preparation, literal controlled powers and readout are counted; synthesis, physical reset/controller models and factories remain unavailable.\n"
                 "The known phase supplies only the pedagogical oracle and independent sampler/reference; it is not passed to experiment selection.\n"
                 "Use --source-only for larger circuits; those sizes have no dense validation or inference result.\n").encode())
    writer.finish(source_hashes)
    return {"status": result["status"], "files": len(writer.inventory), "bytes": writer.total,
            "exported_circuits": writer.circuits, "exported_operations": writer.operations,
            "wall_seconds": time.perf_counter() - writer.started,
            "modeled_trials": result.get("total_modeled_fresh_trials", 0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--phase", type=float, default=.3)
    parser.add_argument("--target-qubits", type=int, default=1)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--power", type=int, default=32, help="Literal power for --source-only; inference uses the fixed candidate powers")
    args = parser.parse_args()
    report = generate(args.output_dir, single=args.single, source_only=args.source_only, phase=args.phase,
                      target_qubits=args.target_qubits, steps=args.steps, grid_size=args.grid_size,
                      seed=args.seed, power=args.power)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
