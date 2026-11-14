#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the fixed, bounded HHL/Neumann conditional physical sweeps."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile

# Public replay must not add interpreter caches to publication artifacts.
sys.dont_write_bytecode = True

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
SOURCE_LAYOUT = SCRIPT.parent == ROOT / "scripts" and (ROOT / "src/openquantum/linear_systems_physical.py").is_file()
SOURCE_ROOT = ROOT if SOURCE_LAYOUT else SCRIPT.parent
PACKAGE_ROOT = ROOT / "src/openquantum" if SOURCE_LAYOUT else SOURCE_ROOT / "openquantum"
sys.path.insert(0, str(SCRIPT.parent))
try:
    from source_provenance import git_dirty, git_revision
except ModuleNotFoundError as error:
    if error.name != "source_provenance":
        raise
    # The optional helper can be omitted from individual public downloads.
    def git_revision(_root):
        return None
    def git_dirty(_root, _paths):
        return None

sys.path.insert(0, str(PACKAGE_ROOT.parent))
from openquantum.linear_systems import decompose
from openquantum.linear_systems_physical import (
    CASES, CONTRACT, MAX_QUERIES, MAX_QUERY_PRIMITIVES, MAX_QUERY_WIRES,
    PROFILE, SLICES, build_case, physical_budget, validate_case,
)
from openquantum.resource_estimation import estimate_circuit_resources
from qiskit import qasm3

OWNED = ROOT / "assets/generated/linear-systems/physical-sweeps"
MODULES = (
    "linear_systems_physical.py", "linear_systems.py", "linear_inverse_lcu.py",
    "linear_systems_qsvt.py", "resource_estimation.py", "simulation/lcu.py", "simulation/qsp.py",
)
REFERENCES = (
    "references/linear-systems.bib", "references/resources.bib",
    "references/simulation.bib", "references/amplification.bib", "citations.bib",
)
BASELINE_FILES = (
    "src/openquantum/grover.py", "src/openquantum/linear_systems.py",
    "src/openquantum/linear_inverse_lcu.py", "src/openquantum/resource_estimation.py",
    "scripts/generate_grover.py", "scripts/estimate_resources.py",
    "scripts/generate_linear_systems.py", "scripts/generate_linear_inverse_lcu.py",
    "requirements-lock.txt", "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv",
)
BASELINE_TREES = (
    "assets/generated/grover", "assets/generated/linear-systems/hhl",
    "assets/generated/linear-systems/neumann-lcu",
)
CASE_FILES = ("measured-trial.qasm", "logical-resources.json", "reference.json", "surface-code.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plain_target(path: Path) -> Path:
    """Never write or clean through preexisting managed-target symlinks."""
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Managed output cannot follow a symbolic link: {candidate.name}")
    return path


def write_text(path: Path, value: str) -> None:
    plain_target(path).write_text(value)


def write_bytes(path: Path, value: bytes) -> None:
    plain_target(path).write_bytes(value)


def remove_file(path: Path) -> None:
    plain_target(path).unlink(missing_ok=True)


def write_json(path: Path, value) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_path(relative: str) -> Path:
    if SOURCE_LAYOUT:
        return ROOT / relative
    if relative.startswith("src/openquantum/"):
        return PACKAGE_ROOT / relative.removeprefix("src/openquantum/")
    if relative == "scripts/generate_linear_systems_physical.py":
        return SCRIPT
    if relative == "scripts/source_provenance.py":
        return SCRIPT.parent / "source_provenance.py"
    return SOURCE_ROOT / relative


def guard_output(path: Path) -> Path:
    if path.is_symlink():
        raise ValueError("Output cannot be a symbolic link")
    path = path.resolve()
    if path == SOURCE_ROOT or SOURCE_ROOT.is_relative_to(path):
        raise ValueError("Output cannot replace the source directory or an ancestor")
    if SOURCE_LAYOUT and path.is_relative_to(ROOT) and not path.is_relative_to(OWNED):
        raise ValueError("Repository output must remain inside assets/generated/linear-systems/physical-sweeps")
    if path.is_relative_to(PACKAGE_ROOT):
        raise ValueError("Output cannot replace the implementation package")
    return path


def guard_cache(path: Path, output: Path) -> Path:
    path = path.resolve()
    if (path.is_relative_to(SOURCE_ROOT) or SOURCE_ROOT.is_relative_to(path)
            or path.is_relative_to(output) or output.is_relative_to(path)):
        raise ValueError("Use a private estimator cache outside the source and publication output")
    return path


def baseline_fingerprints() -> dict:
    """Check real default artifacts in a checkout, or available public copies."""
    if SOURCE_LAYOUT:
        files = set(BASELINE_FILES)
        for tree in BASELINE_TREES:
            directory = ROOT / tree
            if not directory.is_dir():
                raise FileNotFoundError(f"Required frozen baseline directory is missing: {tree}")
            files.update(p.relative_to(ROOT).as_posix() for p in directory.rglob("*") if p.is_file())
        missing = [name for name in sorted(files) if not (ROOT / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Required frozen baseline files are missing: {missing}")
        return {
            "scope": "Grover, HHL and Neumann inputs/default artifacts in this exact source directory",
            "file_sha256": {name: sha256((ROOT / name).read_bytes()) for name in sorted(files)},
            "omitted_origin_artifacts": [],
        }
    snapshot_path = SOURCE_ROOT / "baseline-fingerprints.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError("Use the complete public code bundle, including baseline-fingerprints.json")
    origin = json.loads(snapshot_path.read_text())
    available, omitted = {}, []
    for name, expected in origin["file_sha256"].items():
        path = source_path(name)
        if not path.is_file():
            omitted.append(name)
            continue
        actual = sha256(path.read_bytes())
        if actual != expected:
            raise RuntimeError(f"A packaged frozen baseline source differs from the origin snapshot: {name}")
        available[name] = actual
    return {
        "scope": "available public baseline source copies checked; omitted origin default artifacts are not revalidated",
        "file_sha256": available, "omitted_origin_artifacts": sorted(omitted),
        "origin_snapshot_sha256": sha256(snapshot_path.read_bytes()),
    }


def provenance() -> dict:
    files = {f"src/openquantum/{name}": PACKAGE_ROOT / name for name in MODULES}
    files.update({"scripts/generate_linear_systems_physical.py": SCRIPT,
                  "scripts/source_provenance.py": SCRIPT.parent / "source_provenance.py",
                  "requirements-lock.txt": source_path("requirements-lock.txt")})
    files.update({name: source_path(name) for name in REFERENCES})
    missing = [name for name, path in files.items() if name != "scripts/source_provenance.py" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"The reproducible sweep package is incomplete: {missing}")
    versions = {}
    for name in ("qiskit", "numpy", "scipy", "matplotlib", "qdk"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "source_commit": git_revision(SOURCE_ROOT),
        "source_dirty": git_dirty(SOURCE_ROOT, tuple(files)),
        "source_commit_scope": "exact checkout at generation if available; per-file content hashes identify working implementation",
        "source_sha256": {name: sha256(path.read_bytes()) if path.is_file() else None for name, path in files.items()},
        "environment": {"python": platform.python_version(), "packages": versions,
                        "device": "CPU", "transpiler_seed": 7},
        "private_cache_path_recorded": False,
    }


def export_code(output: Path, frozen: dict) -> dict:
    code = output / "code"
    package = code / "openquantum"
    plain_target(package / "simulation").mkdir(parents=True, exist_ok=True)
    write_text(package / "__init__.py", "")
    write_text(package / "simulation/__init__.py", "")
    for name in MODULES:
        write_bytes(package / name, (PACKAGE_ROOT / name).read_bytes())
    write_bytes(code / SCRIPT.name, SCRIPT.read_bytes())
    helper = SCRIPT.parent / "source_provenance.py"
    if helper.is_file():
        write_bytes(code / helper.name, helper.read_bytes())
    else:
        remove_file(code / "source_provenance.py")
    write_bytes(code / "requirements-lock.txt", source_path("requirements-lock.txt").read_bytes())
    license_path = source_path("LICENSE")
    if not license_path.is_file():
        license_path = source_path("LICENSE.txt")
    write_bytes(code / "LICENSE.txt", license_path.read_bytes())
    for name in REFERENCES:
        target = code / name
        plain_target(target.parent).mkdir(parents=True, exist_ok=True)
        write_bytes(target, source_path(name).read_bytes())
    if SOURCE_LAYOUT:
        write_json(code / "baseline-fingerprints.json", frozen)
    else:
        write_bytes(code / "baseline-fingerprints.json", (SOURCE_ROOT / "baseline-fingerprints.json").read_bytes())
    write_text(code / "README.md",
        "# Reproduce bounded linear-system sweeps\n\n"
        "Use Python 3.12 and install the supplied requirements lock. The generator imports the\n"
        "included `openquantum` package directly; an editable install is unnecessary.\n\n"
        "```bash\npython generate_linear_systems_physical.py --no-estimate\n"
        "python generate_linear_systems_physical.py\n```\n\n"
        "The second command makes at most 13 local QDK queries, using an external private cache.\n"
        "Each case is a separate normalized-solution Z task. These are conditional mean-fit\n"
        "resource models, with no synthesized or hardware output-confidence certificate.\n"
        "Origin default-artifact hashes are retained, while only baseline source copies present\n"
        "in this download can be rechecked. Original code is MIT; exact references are included.\n"
    )
    exported = [package / "__init__.py", package / "simulation/__init__.py", *(package / name for name in MODULES),
                code / SCRIPT.name, code / "requirements-lock.txt", code / "LICENSE.txt", code / "baseline-fingerprints.json",
                code / "README.md", *(code / name for name in REFERENCES)]
    if helper.is_file():
        exported.append(code / helper.name)
    return {p.relative_to(output).as_posix(): sha256(plain_target(p).read_bytes()) for p in sorted(exported)}


def clear_unselected(output: Path, selected: set[str]) -> None:
    for case in CASES:
        if case.case_id in selected:
            continue
        directory = output / case.case_id
        plain_target(directory)
        for name in CASE_FILES:
            remove_file(directory / name)
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass  # Unrelated files in an external output directory are preserved.


def summary_row(case, ledger: dict, reference: dict, report: dict, queried: bool) -> dict:
    points = report.get("pareto_points", [])
    fastest = min(points, key=lambda p: p["serial_runtime_all_executions_ns"]) if points else None
    smallest = min(points, key=lambda p: p["peak_physical_qubits"]) if points else None
    workload = ledger["workload"]
    return {
        "case_id": case.case_id, "method": case.method, "slices": ";".join(case.slices),
        "system_qubits": case.n, "matrix_dimension": 1 << case.n,
        "condition_number": ledger["model"].get("condition_number"),
        "promised_gap": ledger["model"].get("delta"),
        "actual_condition_number": reference.get("actual_condition_number"),
        "degree": case.degree, "logical_qubits": ledger["logical_qubits"],
        "primitive_operations_per_attempt": ledger["primitive_operations"],
        "dependency_depth": ledger["dependency_depth"],
        "final_measurements_per_attempt": ledger["final_measurements_per_attempt"],
        "ideal_probability_lower_bound": ledger["ideal_probability_lower_bound"],
        "compiled_model_probability_lower_bound": ledger["compiled_model_probability_lower_bound"],
        "algorithm_bias_bound": ledger["algorithm_bias_bound"],
        "postselection_adjusted_synthesis_bias_bound": ledger["postselection_adjusted_synthesis_bias_bound"],
        "conditional_total_error_upper_bound": ledger["conditional_total_error_upper_bound"],
        "accepted_samples": workload["accepted_samples"],
        "attempt_cap_per_accepted_slot": workload["attempt_cap_per_accepted_slot"],
        "total_capped_attempts": workload["total_capped_attempts"],
        "capped_measurements": ledger["capped_measurements"],
        "independent_validation_performed": reference["performed"],
        "qdk_query_performed": queried, "physical_status": report["status"], "pareto_points": len(points),
        "fastest_peak_physical_qubits": fastest["peak_physical_qubits"] if fastest else None,
        "fastest_serial_runtime_seconds": fastest["serial_runtime_all_executions_ns"] / 1e9 if fastest else None,
        "fastest_peak_qubit_runtime_proxy_qubit_seconds": fastest["peak_physical_qubits"] * fastest["serial_runtime_all_executions_ns"] / 1e9 if fastest else None,
        "fewest_peak_physical_qubits": smallest["peak_physical_qubits"] if smallest else None,
        "reference_identity_sha256": reference.get("reference_identity_sha256"),
        "qasm_sha256": ledger["qasm_sha256"],
    }


def write_summaries(output: Path, rows: list[dict], query_count: int) -> None:
    with plain_target(output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Bounded conditional physical sweeps", "",
        "Each row requests one normalized-solution Z(q0) mean. The frozen target is 0.05 absolute error",
        "with allocations 0.01 algorithm bias, 0.01 postselection-adjusted synthesis bias and 0.03 sampling.",
        "Statistical, no-acceptance and whole-workload physical allowances are each 0.005.",
        "The coherent-error interpretation of the fixed 1e-6 synthesis scalar is hypothetical; hardware",
        "accuracy, acceptance and runtime-tail confidence are not certified. There is no simultaneous",
        "cross-grid guarantee. Every rejected complete trial, preparation/access and final readout is charged.", "",
        f"This invocation made **{query_count}** QDK queries, bounded by **{MAX_QUERIES}**.", "",
        "| Case | Wires | Primitives/attempt | Capped attempts | Fastest peak qubits | Serial seconds | Evidence |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        name = row["case_id"]
        qubits = f"{row['fastest_peak_physical_qubits']:,}" if row["fastest_peak_physical_qubits"] is not None else "unavailable"
        seconds = f"{row['fastest_serial_runtime_seconds']:.6g}" if row["fastest_serial_runtime_seconds"] is not None else "unavailable"
        links = f"[logical]({name}/logical-resources.json), [reference]({name}/reference.json)"
        if row["qdk_query_performed"]:
            links += f", [physical]({name}/surface-code.json)"
        lines.append(f"| {name} | {row['logical_qubits']} | {row['primitive_operations_per_attempt']:,} | {row['total_capped_attempts']:,} | {qubits} | {seconds} | {links} |")
    lines += ["", "Fastest means the minimum runtime among enumerated points in the declared QDK models.",
              "The peak-qubit × serial-runtime column in the CSV is a rectangular proxy, not an integrated layout volume.",
              "HHL size changes conditioning too; Neumann's fixed Pauli algebra is not generic matrix loading or a quantum-advantage benchmark.",
              "Initialization/reset faults, classical construction/compilation/reference/readout costs, routing and decoder realization remain excluded.", ""]
    write_text(output / "summary.md", "\n".join(lines))


def write_plots(output: Path, rows: list[dict]) -> list[str]:
    physical = {row["case_id"]: row for row in rows if row["fastest_serial_runtime_seconds"] is not None}
    plot_paths = [output / "physical-scaling.svg", output / "physical-scaling.png"]
    if not physical:
        for path in plot_paths:
            remove_file(path)
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slices = [(name, [physical[key] for key in keys if key in physical]) for name, keys in SLICES.items()]
    slices = [(name, values) for name, values in slices if values]
    figure, axes = plt.subplots(len(slices), 2, figsize=(11, 2.9 * len(slices)), squeeze=False)
    for index, (name, values) in enumerate(slices):
        if "size" in name:
            x = [row["system_qubits"] for row in values]
            xlabel = "Active data qubits n"
        elif name == "hhl-conditioning":
            x = [row["condition_number"] for row in values]
            xlabel = "Condition number κ"
        else:
            x = [1 / row["promised_gap"] for row in values]
            xlabel = "Promised condition upper bound 1/δ"
        axes[index, 0].plot(x, [row["fastest_peak_physical_qubits"] for row in values], "o-", color="#167d8d")
        axes[index, 1].plot(x, [row["fastest_serial_runtime_seconds"] for row in values], "o-", color="#7353a2")
        for axis in axes[index]:
            axis.set_xlabel(xlabel)
            axis.set_title(name.replace("-", " ").capitalize())
            axis.set_xticks(x)
            axis.grid(alpha=.2)
        axes[index, 0].set_ylabel("Peak physical qubits at fastest point")
        axes[index, 1].set_ylabel("Capped-workload serial runtime (s)")
        axes[index, 1].set_yscale("log")
    figure.suptitle("Fastest enumerated points in the frozen conditional model; no hardware certificate", fontsize=11)
    figure.tight_layout(rect=(0, 0, 1, .975))
    figure.savefig(plain_target(plot_paths[0]), bbox_inches="tight")
    figure.savefig(plain_target(plot_paths[1]), bbox_inches="tight", dpi=130)
    plt.close(figure)
    write_text(plot_paths[0], "\n".join(line.rstrip() for line in plot_paths[0].read_text().splitlines()) + "\n")
    return [path.name for path in plot_paths]


def run_sweep(output: Path, cases=CASES, *, estimate: bool = True, simulate: bool = True,
              cache_dir: Path | None = None) -> dict:
    cases = tuple(cases)
    if not cases or len(cases) > MAX_QUERIES or any(case not in CASES for case in cases):
        raise ValueError("Select one or all of the fixed 13 cases")
    selected = {case.case_id for case in cases}
    if len(selected) != len(cases):
        raise ValueError("Duplicate cases would not define a unique sweep")
    output = guard_output(output)
    cache = guard_cache(cache_dir or Path(tempfile.gettempdir()) / "openquantum-linear-physical-qre", output)
    before = baseline_fingerprints()
    source_identity = provenance()
    plain_target(output).mkdir(parents=True, exist_ok=True)
    rows, entries, query_count = [], [], 0
    try:
        clear_unselected(output, selected)
        code_hashes = export_code(output, before)
        for case in cases:
            source, ledger = build_case(case)
            compiled = decompose(source)
            if compiled.num_qubits > MAX_QUERY_WIRES or len(compiled.data) - compiled.count_ops().get("barrier", 0) > MAX_QUERY_PRIMITIVES:
                raise ValueError("Actual compiled circuit exceeds the frozen physical query bounds")
            if compiled.num_qubits != ledger["logical_qubits"] or len(compiled.data) - compiled.count_ops().get("barrier", 0) != ledger["primitive_operations"]:
                raise AssertionError("Build ledger does not describe the actual compiled measured trial")
            qasm = qasm3.dumps(compiled)
            qasm_hash = sha256(qasm.encode())
            reference = validate_case(case, source) if simulate else {
                "performed": False, "reason": "explicit --no-sim; independent circuit/reference validation not executed",
                "reference_identity_sha256": None, "physical_attempts_executed": 0,
            }
            if simulate and not reference.get("reference_identity_sha256"):
                raise AssertionError("Executed validation must preserve its independent reference fingerprint")
            directory = output / case.case_id
            plain_target(directory).mkdir(parents=True, exist_ok=True)
            write_text(directory / "measured-trial.qasm", qasm)
            write_json(directory / "reference.json", reference)
            ledger.update({"qasm_sha256": qasm_hash, "provenance": source_identity,
                           "reference_identity_sha256": reference.get("reference_identity_sha256"),
                           "reference_json_sha256": sha256((directory / "reference.json").read_bytes()),
                           "independent_validation_performed": reference["performed"]})
            report = {"status": "unavailable", "reason": "explicit --no-estimate", "pareto_points": []}
            if estimate:
                if query_count >= MAX_QUERIES:
                    raise RuntimeError("The fixed 13-query bound was exceeded")
                query_count += 1
                report = estimate_circuit_resources(source, budget=physical_budget(ledger), profile=PROFILE,
                    cache_dir=cache, optimization_level=1, use_graph=False)
                if report["qasm_sha256"] != qasm_hash:
                    raise AssertionError("Resource estimate did not consume the exported compiled measured trial")
                if report.get("physical_profile") != asdict(PROFILE) or report.get("budget") != asdict(physical_budget(ledger)):
                    raise AssertionError("Resource query changed the frozen profile or capped-workload allocation")
                if report.get("isa_graph_pruning") is not False:
                    raise AssertionError("Physical sweeps require exhaustive declared ISA combinations")
                report.pop("qasm3", None)
                report.update({"case_id": case.case_id, "contract": dict(CONTRACT),
                    "postselection_contract": {"output": CONTRACT["output"],
                        "algorithm_bias_bound": ledger["algorithm_bias_bound"],
                        "postselection_adjusted_synthesis_bias_bound": ledger["postselection_adjusted_synthesis_bias_bound"],
                        "conditional_total_error_upper_bound": ledger["conditional_total_error_upper_bound"],
                        "compiled_model_probability_lower_bound": ledger["compiled_model_probability_lower_bound"],
                        "workload": ledger["workload"], "coherent_eta_interpretation_certified": False},
                    "initialization_included": False, "hardware_output_accuracy_certified": False,
                    "simultaneous_cross_grid_guarantee": False,
                    "reference_identity_sha256": reference.get("reference_identity_sha256"),
                    "provenance": source_identity, "excluded": ledger["excluded"]})
                write_json(directory / "surface-code.json", report)
            else:
                remove_file(directory / "surface-code.json")
            ledger["physical_estimation_status"] = report["status"]
            ledger["physical_estimation"] = {"query_performed": estimate, "status": report["status"],
                "pareto_points": len(report.get("pareto_points", [])), "report": "surface-code.json" if estimate else None,
                "hardware_output_accuracy_certified": False}
            write_json(directory / "logical-resources.json", ledger)
            row = summary_row(case, ledger, reference, report, estimate)
            rows.append(row)
            entries.append({"case_id": case.case_id, "instance": case.instance(), "slices": list(case.slices),
                "reference_identity_sha256": reference.get("reference_identity_sha256"),
                "independent_validation_performed": reference["performed"], "qdk_query_performed": estimate,
                "physical_status": report["status"],
                "files_sha256": {name: sha256((directory / name).read_bytes()) for name in CASE_FILES if (directory / name).is_file()}})
        write_summaries(output, rows, query_count)
        plots = write_plots(output, rows)
    finally:
        after = baseline_fingerprints()
        if before != after:
            raise RuntimeError("Frozen Grover/HHL/Neumann inputs or default artifacts changed during the sweep")
        if provenance()["source_sha256"] != source_identity["source_sha256"]:
            raise RuntimeError("Sweep implementation or dependencies changed during generation; regenerate from one fixed source state")
    manifest = {
        "schema_version": 1, "contract": dict(CONTRACT), "physical_profile": asdict(PROFILE),
        "selected_case_ids": [case.case_id for case in cases], "generated_case_count": len(cases),
        "complete_declared_grid": len(cases) == len(CASES), "declared_slices": SLICES,
        "bounds": {"maximum_queries": MAX_QUERIES, "maximum_logical_wires": MAX_QUERY_WIRES,
                   "maximum_primitive_operations_per_attempt": MAX_QUERY_PRIMITIVES},
        "qdk_query_count": query_count, "independent_validation_count": sum(row["independent_validation_performed"] for row in rows),
        "query_scope": "one actual compiled full trial and all capped serial attempts per case; cases are separate tasks",
        "simultaneous_cross_grid_guarantee": False, "physical_attempts_executed": 0,
        "hardware_output_accuracy_certified": False, "source_provenance": source_identity,
        "baseline_verification": {"before": before, "after": after, "unchanged": True},
        "code_files_sha256": code_hashes, "cases": entries, "plots": plots,
        "reproduction_arguments": {"case": "all" if len(cases) == len(CASES) else cases[0].case_id,
                                   "no_estimate": not estimate, "no_sim": not simulate},
    }
    write_text(output / "README.md",
        "# Bounded HHL and Neumann physical sweeps\n\n"
        "[Summary](summary.md), [CSV](summary.csv) and [manifest](manifest.json) retain actual complete-trial\n"
        "circuits, independent reference fingerprints, fixed allocations, source hashes and conditional\n"
        "surface-code reports. Use the included `code/` package and lock to reproduce them.\n\n"
        "The target is 0.05 error with failure allocation 0.015 for each independent Z-mean task.\n"
        "These allocations are conditional on an uncertified coherent-error interpretation of the\n"
        "mean-fit synthesis scalar. No hardware output-confidence or simultaneous cross-grid guarantee\n"
        "is claimed. All rejected complete attempts are charged. Fresh initialized-zero faults/timing,\n"
        "classical construction/compilation/readout and hardware routing/decoder realization are excluded.\n"
        "The structured families are not general input loaders or matched advantage benchmarks.\n"
    )
    write_json(output / "manifest.json", manifest)
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *(case.case_id for case in CASES)), default="all")
    parser.add_argument("--no-estimate", action="store_true", help="Generate circuits/references without QDK queries")
    parser.add_argument("--no-sim", action="store_true", help="Mark independent small checks unexecuted")
    default = OWNED if SOURCE_LAYOUT else SOURCE_ROOT / "physical-sweep-results"
    parser.add_argument("--output-dir", type=Path, default=default)
    parser.add_argument("--cache-dir", type=Path, help="External private QDK cache; its path is omitted from public artifacts")
    args = parser.parse_args(argv)
    cases = CASES if args.case == "all" else tuple(case for case in CASES if case.case_id == args.case)
    manifest = run_sweep(args.output_dir, cases, estimate=not args.no_estimate,
                         simulate=not args.no_sim, cache_dir=args.cache_dir)
    print(json.dumps({"cases": manifest["generated_case_count"], "qdk_queries": manifest["qdk_query_count"],
                      "independent_checks": manifest["independent_validation_count"],
                      "frozen_baselines_unchanged": manifest["baseline_verification"]["unchanged"],
                      "hardware_output_accuracy_certified": False}))


if __name__ == "__main__":
    main()
