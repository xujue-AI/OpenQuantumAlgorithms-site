#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a bounded fixed-coupling, accuracy-preserving word-Taylor study.

The canonical 16 cells were prescribed before generation. Pure rational
contracts precede circuit materialization; rejected/unsupported cells remain
explicit. Physical-resource estimation and sampling are not performed.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPOSITORY = (ROOT/"src/openquantum/simulation/word_taylor_workloads.py").is_file()
if REPOSITORY:
    sys.path.insert(0, str(ROOT/"src"))
    from openquantum.simulation import word_taylor_workloads as workloads
    from openquantum.simulation import word_taylor as wt
else:
    sys.path.insert(0, str(HERE))
    import word_taylor_workloads as workloads
    import word_taylor as wt
try:
    from source_provenance import git_revision, git_dirty
except ImportError:
    def git_revision(root):
        return None
    def git_dirty(root, paths):
        return None

from qiskit import qasm3


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_cases():
    cases = []
    for n in (1, 2, 3, 4):
        for mode in workloads.MODES:
            cases.append((f"size-n{n}-{mode}", "size", n, Fraction(1), Fraction(1, 20), mode))
    for t in (Fraction(1, 2), Fraction(2)):
        for mode in workloads.MODES:
            label = "half" if t == Fraction(1, 2) else "two"
            cases.append((f"time-{label}-{mode}", "time", 2, t, Fraction(1, 20), mode))
    for accuracy in (Fraction(1, 40), Fraction(1, 10)):
        for mode in workloads.MODES:
            label = "025" if accuracy == Fraction(1, 40) else "100"
            cases.append((f"accuracy-{label}-{mode}", "accuracy", 2, Fraction(1), accuracy, mode))
    return tuple(cases)


def source_files():
    if REPOSITORY:
        return {
            "word_taylor_workloads.py": ROOT/"src/openquantum/simulation/word_taylor_workloads.py",
            "word_taylor.py": ROOT/"src/openquantum/simulation/word_taylor.py",
            "amplification.py": ROOT/"src/openquantum/simulation/amplification.py",
            "lcu.py": ROOT/"src/openquantum/simulation/lcu.py",
            "hamiltonian_simulation.py": ROOT/"src/openquantum/hamiltonian_simulation.py",
            "source_provenance.py": ROOT/"scripts/source_provenance.py",
            "generate_word_taylor_workloads.py": Path(__file__),
            "requirements-lock.txt": ROOT/"requirements-lock.txt",
        }
    names = ("word_taylor_workloads.py", "word_taylor.py", "amplification.py", "lcu.py",
             "hamiltonian_simulation.py", "source_provenance.py", "generate_word_taylor_workloads.py",
             "requirements-lock.txt")
    return {name: HERE/name for name in names}


def validate_output(output: Path):
    output = output.resolve()
    if REPOSITORY:
        for protected in (ROOT, ROOT/"src", ROOT/"scripts", ROOT/"tests", ROOT/".git"):
            if output == protected or (protected != ROOT and protected in output.parents):
                raise ValueError("output overlaps protected source/project files")
    return output


def plot_size_workloads(output: Path, rows):
    selected = [row for row in rows if row["slice"] == "size" and row["circuit_generated"]]
    if len(selected) < 2:
        for name in ("size-workloads.svg", "size-workloads.png"):
            (output/name).unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.hashsalt"] = "word-taylor-workloads-v1"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = {"native": "Unamplified", "normalization2-oaa1": "Normalization two, one OAA round"}
    for mode in workloads.MODES:
        points = [row for row in selected if row["mode"] == mode]
        if points:
            axes[0].plot([row["n"] for row in points], [row["logical_qubits"] for row in points],
                         "o-", label=labels[mode])
            axes[1].semilogy([row["n"] for row in points],
                            [row["primitive_operations_full_cap"] for row in points],
                            "o-", label=labels[mode])
    axes[0].set(xlabel="System spins n", ylabel="Logical wires per complete attempt")
    axes[1].set(xlabel="System spins n", ylabel="Primitive operations over full retry cap")
    for axis in axes:
        axis.set_xticks((1, 2, 3, 4))
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    if any(row["slice"] == "size" and not row["circuit_generated"] for row in rows):
        axes[1].text(.98, .03, "n=4 OAA export exceeds the primitive cap",
                     transform=axes[1].transAxes, ha="right", va="bottom", fontsize=8)
    figure.suptitle("Fixed local couplings, Z-mean accuracy 0.05")
    figure.text(.5, .01, "Actual compiled counts; ideal-access workload bounds. Synthesis and physical resources are unmodeled.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .045, 1, .95))
    figure.savefig(output/"size-workloads.svg", metadata={"Date": None})
    figure.savefig(output/"size-workloads.png", dpi=160)
    plt.close(figure)
    svg = output/"size-workloads.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


def generate(output: Path, selected_slice: str = "all", simulate: bool = True,
             selected_case: str | None = None):
    output = validate_output(output)
    all_cases = canonical_cases()
    if selected_slice not in ("all", "size", "time", "accuracy"):
        raise ValueError("unknown prescribed slice")
    chosen = [case for case in all_cases if selected_slice == "all" or case[1] == selected_slice]
    if selected_case is not None:
        chosen = [case for case in chosen if case[0] == selected_case]
        if not chosen:
            raise ValueError("case is absent from the selected prescribed slice")
    output.mkdir(parents=True, exist_ok=True)
    inputs = source_files()
    hashes = {}
    for name, source in inputs.items():
        raw = source.read_bytes()
        (output/name).write_bytes(raw)
        hashes[name] = sha256(raw)
    license_file = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    (output/"LICENSE.txt").write_bytes(license_file.read_bytes())
    environment = {"python": platform.python_version(),
                   "packages": {name: importlib.metadata.version(name)
                                for name in ("qiskit", "numpy", "scipy", "matplotlib")},
                   "device": "CPU", "backend": "bounded Qiskit ideal Statevector",
                   "seed": 20261112}
    records, rows = [], []
    started = time.perf_counter()
    for case_id, slice_name, n, total_time, accuracy, mode in chosen:
        case_started = time.perf_counter()
        directory = output/case_id
        directory.mkdir(exist_ok=True)
        plan = workloads.plan_word_taylor(n, total_time, accuracy, mode)
        record = {"case_id": case_id, "slice": slice_name, "plan": plan.record(),
                  "circuit_generation": {"performed": False},
                  "numerical_validation": {"performed": False, "reason": "disabled"},
                  "physical_estimation": {"performed": False, "reason": "general-width reset instrument is not implemented"},
                  "sampling": {"performed": False}}
        row = {"case_id": case_id, "slice": slice_name, "n": n, "time": float(total_time),
               "total_accuracy": float(accuracy), "mode": mode, "segments": plan.segments,
               "degree": plan.degree, "ideal_algorithm_bias_allowance": float(accuracy/5),
               "uniform_ideal_bias_bound": float(plan.bounds["conditional_pauli_bias"]),
               "uniform_ideal_acceptance_lower": float(plan.bounds["acceptance_lower"]),
               "accepted_slots": plan.shots, "full_attempt_cap_per_slot": plan.attempts_per_slot,
               "complete_attempts_full_cap": plan.full_attempts, "circuit_generated": False,
               "primitive_operations_per_attempt": "", "logical_qubits": "",
               "primitive_operations_full_cap": "", "dependency_layers_full_cap": "",
               "simulation_performed": False, "status": ""}
        try:
            access, segment, attempt = workloads.build_planned_attempt(plan)
            compiled = wt.decompose_word_circuit(attempt)
        except ValueError as error:
            reason = str(error)
            record["circuit_generation"] = {"performed": False, "reason": reason,
                                             "scope": "analytic contract retained; materialization cap rejected this cell"}
            record["numerical_validation"] = {"performed": False, "reason": "circuit materialization was rejected"}
            row["status"] = f"materialization-rejected: {reason}"
            for file in ("measured-attempt.qasm", "coherent-segment.qasm"):
                (directory/file).unlink(missing_ok=True)
        else:
            qasm = qasm3.dumps(compiled)
            coherent_qasm = qasm3.dumps(segment)
            (directory/"measured-attempt.qasm").write_text(qasm, encoding="utf-8")
            (directory/"coherent-segment.qasm").write_text(coherent_qasm, encoding="utf-8")
            record["circuit_generation"] = {
                "performed": True, "measured_qasm_sha256": sha256(qasm.encode()),
                "coherent_source_qasm_sha256": sha256(coherent_qasm.encode()),
                "compiled_global_phase_radians": float(compiled.global_phase),
                "coherent_source_global_phase_radians": float(segment.global_phase),
                "qasm_phase_scope": "Pinned QASM exporter may omit global phase; measured-trial probabilities are the export contract",
                "source_access_normalization_float": access.normalization,
                "source_access_alpha_float": access.alpha,
                "transpilation": {"optimization_level": 1, "seed": 7,
                                  "qubits_initially_zero": False},
                "source_hashes": hashes,
            }
            record["logical_resources"] = workloads.primitive_ledger(plan, attempt, compiled)
            if simulate:
                record["numerical_validation"] = workloads.check_tiny_circuit(
                    plan, access, segment, compiled)
            row.update(circuit_generated=True,
                       logical_qubits=compiled.num_qubits,
                       primitive_operations_per_attempt=compiled.size(),
                       primitive_operations_full_cap=compiled.size()*plan.full_attempts,
                       dependency_layers_full_cap=compiled.depth()*plan.full_attempts,
                       simulation_performed=record["numerical_validation"]["performed"],
                       status="tiny-circuit-checked" if record["numerical_validation"]["performed"] else "circuit-only")
        record["cpu_generation_and_check_seconds"] = time.perf_counter()-case_started
        (directory/"record.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        records.append(record)
        rows.append(row)
        print(f"{case_id}: {row['status']}", flush=True)
    # Only prescribed managed cells are cleared on a changed bounded selection.
    for previous in all_cases:
        if previous[0] not in {case[0] for case in chosen}:
            managed = output/previous[0]
            if managed.is_dir():
                for name in ("record.json", "measured-attempt.qasm", "coherent-segment.qasm"):
                    (managed/name).unlink(missing_ok=True)
                if not any(managed.iterdir()):
                    managed.rmdir()
    with (output/"summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_size_workloads(output, rows)
    manifest = {
        "study": "Fixed-local-coupling, accuracy-preserving structured word-Taylor logical workloads",
        "prescribed_grid": [{"case_id": case[0], "slice": case[1], "n": case[2],
                             "time": str(case[3]), "accuracy": str(case[4]), "mode": case[5]}
                            for case in all_cases],
        "selected_cases": [row["case_id"] for row in rows],
        "fixed_parameters": {"J": "3/5", "h": "7/20", "input": "|0>^n",
                             "observable": "Z(q0)", "hbar": 1},
        "source_hashes": hashes, "environment": environment,
        "environment_lock_sha256": hashes["requirements-lock.txt"],
        "license_sha256": sha256((output/"LICENSE.txt").read_bytes()),
        "source_revision_at_generation": git_revision(ROOT) if REPOSITORY else None,
        "source_dirty_at_generation": git_dirty(ROOT, [str(path.relative_to(ROOT)) for path in inputs.values()])
                                      if REPOSITORY else None,
        "bound_source": workloads.BOUND_SOURCE,
        "uniform_guarantee_scope": "Ideal exact word access and exact rational arithmetic; executed gate checks are numerical",
        "sampling_performed": False, "physical_estimation_performed": False,
        "formal_proof_integrated": False,
        "cpu_generation_and_check_seconds": time.perf_counter()-started,
        "records": records,
    }
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    summary = [
        "# Accuracy-preserving word-Taylor workloads", "",
        "Fixed local couplings are J=3/5 and h=7/20. Each cell chooses its segment",
        "count and smallest degree from uniform exact-rational ideal-access bounds.",
        "All complete attempts reset the known input, execute every segment,",
        "measure all access ancillas, reset them between segments and read out Z.",
        "", "The 16-cell grid was prescribed before generation. Materialization",
        "rejections remain explicit. Circuit-only cells are not simulated results.",
        "Sampling and physical resource estimation are unperformed.",
        "", "| Cell | n | Time | Accuracy | r | K | Ideal acceptance lower | Complete attempt cap | Evidence |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        summary.append(f"| {row['case_id']} | {row['n']} | {row['time']:g} | {row['total_accuracy']:g} | "
                       f"{row['segments']} | {row['degree']} | {row['uniform_ideal_acceptance_lower']:.6g} | "
                       f"{row['complete_attempts_full_cap']} | {row['status']} |")
    summary.extend(["", "Exact bounds describe a mathematical access model. The saved",
                    "floating-point normalization, probability/angle construction and",
                    "compiler do not carry a uniform norm certificate. Rotation",
                    "synthesis and physical faults/timings/routing/factories remain",
                    "separate and unmodeled. Dependency depth is not hardware time.",
                    "", "Source snapshots and the MIT license support standalone",
                    "reproduction with the repository's locked Python 3.12 environment."])
    (output/"summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slice", choices=("all", "size", "time", "accuracy"), default="all")
    parser.add_argument("--case", choices=[case[0] for case in canonical_cases()])
    parser.add_argument("--no-sim", action="store_true")
    default = ROOT/"assets/generated/word-taylor-workloads" if REPOSITORY else HERE/"output"
    parser.add_argument("--output-dir", type=Path, default=default)
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.slice, not args.no_sim, args.case)
    print(json.dumps({"selected_cells": len(manifest["records"]),
                      "circuit_generated": sum(record["circuit_generation"]["performed"] for record in manifest["records"]),
                      "tiny_checks": sum(record["numerical_validation"]["performed"] for record in manifest["records"])}, indent=2))


if __name__ == "__main__":
    main()
