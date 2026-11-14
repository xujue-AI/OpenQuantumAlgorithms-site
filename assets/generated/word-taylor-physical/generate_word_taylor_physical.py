#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded two-case/two-profile serialized-instrument Taylor comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPOSITORY = HERE == ROOT/"scripts" and (ROOT/"src/openquantum").is_dir()
if REPOSITORY:
    sys.path.insert(0, str(ROOT/"src"))
    from openquantum import word_taylor_physical as study
else:
    sys.path.insert(0, str(HERE))
    import word_taylor_physical as study
try:
    from source_provenance import git_revision, git_dirty
except ImportError:
    def git_revision(root):
        return None
    def git_dirty(root, paths):
        return None

from qiskit import qasm3

SOURCE_NAMES = ("word_taylor_physical.py", "instrument_resources.py",
                "word_taylor_workloads.py", "word_taylor.py", "amplification.py", "lcu.py",
                "hamiltonian_simulation.py", "source_provenance.py",
                "generate_word_taylor_physical.py", "requirements-lock.txt", "references.bib")
TASK_FILES = ("source-instrument.qasm", "routed-instrument.qasm",
              "clifford-t-instrument.qasm", "report.json")
FROZEN = ("src/openquantum/grover.py", "scripts/estimate_resources.py",
          "assets/generated/resources/grover-surface-code.json",
          "assets/generated/resources/grover-surface-code.csv",
          "src/openquantum/instrument_resources.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    if not REPOSITORY:
        return {name: HERE/name for name in SOURCE_NAMES}
    return {
        "word_taylor_physical.py": ROOT/"src/openquantum/word_taylor_physical.py",
        "instrument_resources.py": ROOT/"src/openquantum/instrument_resources.py",
        **{name: ROOT/"src/openquantum/simulation"/name for name in
           ("word_taylor_workloads.py", "word_taylor.py", "amplification.py", "lcu.py")},
        "hamiltonian_simulation.py": ROOT/"src/openquantum/hamiltonian_simulation.py",
        "source_provenance.py": ROOT/"scripts/source_provenance.py",
        "generate_word_taylor_physical.py": Path(__file__),
        "requirements-lock.txt": ROOT/"requirements-lock.txt",
        "references.bib": ROOT/"references/resources.bib",
    }


def fingerprints():
    return {name: digest(ROOT/name) if (ROOT/name).is_file() else None for name in FROZEN}


def validate_paths(output, private, sources):
    output, private = Path(output).resolve(), Path(private).resolve()
    owned = ROOT/"assets/generated/word-taylor-physical"
    if REPOSITORY and output.is_relative_to(ROOT) and not output.is_relative_to(owned):
        raise ValueError("repository output must remain in assets/generated/word-taylor-physical")
    if private.is_relative_to(output) or output.is_relative_to(private):
        raise ValueError("private worker/cache files and public artifacts must be disjoint")
    if REPOSITORY and private.is_relative_to(ROOT):
        raise ValueError("private worker/cache files must remain outside the source checkout")
    if any(Path(path).resolve().is_relative_to(output) for path in sources.values()):
        raise ValueError("public output cannot overwrite an input source")
    managed = [output/case/name for case in study.CASES for name in TASK_FILES]
    managed += [output/name for name in (*SOURCE_NAMES, "LICENSE.txt", "manifest.json",
                                        "summary.csv", "summary.md", "table.md", "comparison.svg", "comparison.png")]
    if any(path.is_symlink() or path.parent.is_symlink() for path in managed):
        raise ValueError("managed artifact targets cannot be symbolic links")
    return output, private


def write_json(path, value):
    Path(path).write_text(json.dumps(study._jsonable(value), indent=2, allow_nan=False)+"\n")


def invalidate_completions(output):
    """Fail closed before rewriting inputs/QASM for a fresh generation."""
    for name in ("manifest.json", "summary.csv", "summary.md", "table.md", "comparison.svg", "comparison.png"):
        (output/name).unlink(missing_ok=True)
    for case_id in study.CASES:
        (output/case_id/"report.json").unlink(missing_ok=True)


def plot_comparison(output, rows):
    if len(rows) < 2:
        for name in ("comparison.svg", "comparison.png"):
            (output/name).unlink(missing_ok=True)
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["svg.hashsalt"] = "word-taylor-physical-v1"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for error in study.PHYSICAL_ERROR_PROFILES:
        points = [row for row in rows if row["physical_error"] == error and row["selection"] == "fewest qubits"]
        if not points:
            continue
        x = [0 if row["case_id"] == "size-n1-native" else 1 for row in points]
        axes[0].plot(x, [row["peak_physical_qubits"] for row in points], "o-", label=f"p={error:g}")
        axes[1].plot(x, [row["full_cap_hours"] for row in points], "o-", label=f"p={error:g}")
    for axis in axes:
        axis.set_xticks((0, 1), ("Native", "Padded one-round OAA"))
        axis.grid(alpha=.2)
        axis.legend()
    axes[0].set_ylabel("Peak physical qubits")
    axes[1].set_ylabel("Full-cap serialized model time (hours)")
    figure.suptitle("Same one-spin Z-mean task, accuracy 0.05")
    figure.text(.5, .01, "Fewest-qubit frontier choices; actual Clifford+T instruments. Conditional geometry and fault model.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, .05, 1, .95))
    figure.savefig(output/"comparison.svg", metadata={"Date": None})
    figure.savefig(output/"comparison.png", dpi=160)
    plt.close(figure)
    svg = output/"comparison.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")


def generate(output, *, case="all", estimate=True,
             synthesis_timeout=study.DEFAULT_SYNTHESIS_TIMEOUT,
             query_timeout=study.DEFAULT_QUERY_TIMEOUT):
    if case != "all" and case not in study.CASES:
        raise ValueError("only the two predeclared cases may be selected")
    synthesis_timeout = study.validate_timeout(synthesis_timeout)
    query_timeout = study.validate_timeout(query_timeout)
    selected = list(study.CASES) if case == "all" else [case]
    sources = source_files()
    before = fingerprints() if REPOSITORY else {}
    records, rows = [], []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="word-taylor-physical-") as temporary:
        output, private = validate_paths(output, Path(temporary), sources)
        output.mkdir(parents=True, exist_ok=True)
        invalidate_completions(output)
        hashes = {}
        for name, source in sources.items():
            (output/name).write_bytes(source.read_bytes())
            hashes[name] = digest(source)
        license_file = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
        (output/"LICENSE.txt").write_bytes(license_file.read_bytes())
        for case_id in selected:
            contract = study.physical_contract(case_id)
            _, _, source = study.workloads.build_planned_attempt(contract.plan)
            directory = output/case_id
            directory.mkdir(exist_ok=True)
            worker = private/case_id
            synthesized, synthesis_seconds = study.bounded_synthesis(
                source, worker, contract.synthesis_allowance_float, synthesis_timeout)
            checks = study.check_instrument_laws(contract, synthesized)
            exported = {}
            for name, circuit in (("source-instrument.qasm", source),
                                  ("routed-instrument.qasm", synthesized.routed),
                                  ("clifford-t-instrument.qasm", synthesized.circuit)):
                (directory/name).write_text(qasm3.dumps(circuit))
                exported[name] = digest(directory/name)
            physical = []
            for error in study.PHYSICAL_ERROR_PROFILES:
                if not estimate:
                    break
                result, seconds = study.bounded_estimate(
                    synthesized, worker/f"profile-{error}", contract.complete_attempts,
                    error, private/"factory-cache", query_timeout)
                result["bounded_search_cpu_seconds"] = seconds
                physical.append(result)
                for selection, key in (("fewest qubits", "peak_physical_qubits"),
                                       ("shortest runtime", "worst_case_runtime_all_attempts_ns")):
                    point = min(result["pareto_points"], key=lambda item: item[key])
                    rows.append({"case_id": case_id, "physical_error": error,
                                 "selection": selection, "logical_wires": source.num_qubits,
                                 "accepted_slots": contract.plan.shots, "retry_cap": contract.retry_cap,
                                 "complete_attempts": contract.complete_attempts,
                                 "T_Tdag_per_attempt": synthesized.circuit.count_ops().get("t", 0)+synthesized.circuit.count_ops().get("tdg", 0),
                                 "code_distance": point["data_code_distance"],
                                 "peak_physical_qubits": point["peak_physical_qubits"],
                                 "full_cap_hours": point["worst_case_runtime_all_attempts_ns"]/3.6e12,
                                 "physical_fault_score": point["full_physical_failure_model"],
                                 "factory_readiness_bound": point["factory_supply"]["availability_failure_union_bound"]})
                print(f"{case_id}, p={error:g}: conditional serialized frontier computed", flush=True)
            record = {
                "case_id": case_id, "contract": contract.record(),
                "source_access_schedule_metadata": source.metadata,
                "source_gate_counts": dict(source.count_ops()),
                "source_gate_counts_full_cap": {name: count*contract.complete_attempts for name, count in source.count_ops().items()},
                "actual_synthesis": synthesized.report, "synthesis_cpu_seconds": synthesis_seconds,
                "logical_clifford_t_gate_counts": dict(synthesized.circuit.count_ops()),
                "logical_clifford_t_gate_counts_full_cap": {name: count*contract.complete_attempts for name, count in synthesized.circuit.count_ops().items()},
                "instrument_checks": checks, "qasm_sha256": exported,
                "physical_profiles": physical, "physical_estimation_performed": bool(physical),
                "hardware_confidence_certified": False, "sampling_performed": False,
                "timeout_limits_seconds": {"synthesis": synthesis_timeout, "profile_search": query_timeout},
            }
            write_json(directory/"report.json", record)
            records.append(record)
            print(f"{case_id}: complete source/routed/Clifford+T laws checked", flush=True)
        for omitted in set(study.CASES)-set(selected):
            directory = output/omitted
            for name in TASK_FILES:
                (directory/name).unlink(missing_ok=True)
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        after = fingerprints() if REPOSITORY else {}
        if before != after:
            raise RuntimeError("frozen baseline or reused instrument implementation changed")
        manifest = {
            "study": "One-spin word-Taylor conditional serialized-instrument resource comparison",
            "predeclared_cases": list(study.CASES),
            "predeclared_physical_errors": list(study.PHYSICAL_ERROR_PROFILES),
            "selected_cases": selected, "records": records,
            "source_hashes": hashes, "environment_lock_sha256": hashes["requirements-lock.txt"],
            "license_sha256": digest(output/"LICENSE.txt"),
            "source_revision_at_generation": git_revision(ROOT) if REPOSITORY else None,
            "source_dirty_at_generation": git_dirty(ROOT, [str(path.relative_to(ROOT)) for path in sources.values()])
                                          if REPOSITORY else None,
            "environment": {"python": platform.python_version(), "device": "CPU",
                            "packages": {name: importlib.metadata.version(name)
                                         for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")}},
            "reused_instrument_source_before": before.get("src/openquantum/instrument_resources.py"),
            "reused_instrument_source_after": after.get("src/openquantum/instrument_resources.py"),
            "frozen_fingerprints_before": before, "frozen_fingerprints_after": after,
            "complete_model_searches": len(records)*2 if estimate else 0,
            "physical_estimation_performed": estimate,
            "source_or_hardware_advantage_claimed": False,
            "larger_width_Taylor_physical_status": "pending; unchanged instrument model supports at most five logical wires",
            "cpu_generation_seconds": time.perf_counter()-started,
        }
        write_json(output/"manifest.json", manifest)
        if rows:
            with (output/"summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
        else:
            (output/"summary.csv").unlink(missing_ok=True)
        plot_comparison(output, rows)
        text = [
            "# One-spin Taylor instrument resource comparison", "",
            "Both tasks use H=.35X, time=1, fresh zero input and one Z0 mean",
            "with accuracy0.05. Actual Clifford+T sequences and full source/routed",
            "readout laws are checked. Initialization, failed attempts, routing,",
            "readout, controller work, injection and capped factory supply are charged.",
            "", "The failure split per task/profile is .004 sampling +.001 exhausted",
            "accepted-slot retries +.0025 physical faults +.0025 factory readiness =.01.",
            "Numerical operator guards and the unchanged serialized geometry/fault",
            "model are conditional assumptions, not hardware-success certificates.",
            "", "| Case | p | Selection | Attempts | T/Tdag per attempt | Distance | Peak qubits | Full-cap hours |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            text.append(f"| {row['case_id']} | {row['physical_error']:g} | {row['selection']} | "
                        f"{row['complete_attempts']} | {row['T_Tdag_per_attempt']} | {row['code_distance']} | "
                        f"{row['peak_physical_qubits']} | {row['full_cap_hours']:.6g} |")
        table_start = next(index for index, line in enumerate(text) if line.startswith("| Case |"))
        (output/"table.md").write_text("\n".join(text[table_start:])+"\n")
        if not estimate:
            text.append("\nPhysical searches were disabled; this output supplies no physical frontier.")
        text.extend(["", "Each physical report retains all native frontier points and exact macro",
                     "ledgers. These rows do not compare with the different Grover/PSSPC",
                     "task or establish quantum advantage. Larger-width Taylor schedules",
                     "remain physically pending. Offline coefficient/compilation/synthesis",
                     "and final statistical-report wall times are excluded from quantum-model",
                     "runtime and recorded separately where measured. See report assumptions",
                     "and the copied references for geometry, controller, factory and license scope."])
        (output/"summary.md").write_text("\n".join(text)+"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["all", *study.CASES], default="all")
    parser.add_argument("--no-estimate", action="store_true")
    parser.add_argument("--synthesis-timeout", type=float, default=study.DEFAULT_SYNTHESIS_TIMEOUT)
    parser.add_argument("--query-timeout", type=float, default=study.DEFAULT_QUERY_TIMEOUT)
    default = ROOT/"assets/generated/word-taylor-physical" if REPOSITORY else HERE/"output"
    parser.add_argument("--output-dir", type=Path, default=default)
    args = parser.parse_args()
    result = generate(args.output_dir, case=args.case, estimate=not args.no_estimate,
                      synthesis_timeout=args.synthesis_timeout, query_timeout=args.query_timeout)
    print(json.dumps({"cases": len(result["records"]), "model_searches": result["complete_model_searches"]}))


if __name__ == "__main__":
    main()
