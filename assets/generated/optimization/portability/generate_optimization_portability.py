#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a separate sampled-optimizer portability check on new toy inputs.

From a checkout: python scripts/generate_optimization_portability.py
From public downloads: place this file, the matching portability optimization.py export,
LICENSE.txt and the environment lock together, then execute this script.
The bounded schedule and three seeds live in optimization.py. Each task uses
one fixed initialization across all seeds. Previous artifacts are preserved.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

# Source layout is independent of optional Git checkout metadata.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from source_provenance import git_dirty, git_revision
except ModuleNotFoundError as error:
    if error.name != "source_provenance":
        raise
    # Individual source downloads can omit this optional metadata helper.
    def git_revision(_root):
        return None
    def git_dirty(_root, _paths):
        return None

from time import perf_counter

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = (PROJECT / "src/openquantum/optimization.py").is_file()
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
SOURCE = ROOT / "src/openquantum/optimization.py" if CHECKOUT else ROOT / "optimization.py"
if not SOURCE.is_file():
    raise SystemExit("Save the portability optimization.py export beside this script, or run from a checkout")
spec = importlib.util.spec_from_file_location("optimization_portability_artifact_source", SOURCE)
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
spec.loader.exec_module(opt)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector

COLORS = ("#176e70", "#bd6d36", "#6a6699")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-portability-optimization"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    packages = {}
    for name in ("qiskit", "numpy", "matplotlib", "jupyter-book"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = ROOT / "requirements-lock.txt"
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), packages=packages,
                  device=dict(backend="CPU Statevector probabilities plus NumPy PCG64 ideal measurement draws",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  implementation_sha256=digest(SOURCE), generator_sha256=digest(SCRIPT),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None,
                  invocation="python scripts/generate_optimization_portability.py" if CHECKOUT
                  else "python generate_optimization_portability.py")
    if CHECKOUT:
        record["source_commit"] = git_revision(ROOT)
        record["dirty_owned_source"] = git_dirty(ROOT, ['src/openquantum/optimization.py', 'scripts/generate_optimization_portability.py', 'notebooks/optimization_portability.ipynb'])
    record["revision_note"] = ("source_commit identifies the checkout base at generation; content hashes "
                               "identify the implementation and generator actually executed")
    return record


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy_downloads(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "optimization.py").write_bytes(SOURCE.read_bytes())
    (folder / "generate_optimization_portability.py").write_bytes(SCRIPT.read_bytes())
    license_path = ROOT / ("LICENSE" if CHECKOUT else "LICENSE.txt")
    if not license_path.is_file():
        raise SystemExit("Implementation license missing; save LICENSE.txt beside the public script")
    (folder / "LICENSE.txt").write_bytes(license_path.read_bytes())


def export_circuit(folder: Path, name: str, circuit) -> dict:
    compiled = opt.compile_rotation_basis(circuit)
    path = folder / name
    path.write_text(qasm3.dumps(compiled), encoding="utf-8")
    return dict(file=name, sha256=digest(path), logical_qubits=compiled.num_qubits,
                gate_counts=dict(compiled.count_ops()), depth=compiled.depth(),
                rotations="explicit unsynthesized RX/RY/RZ angles, radians")


def save_figure(fig, output: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_trajectories(output: Path, trajectories: dict) -> None:
    """Plot chosen states and sampled incumbents without rerunning decisions."""
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, algorithm in zip(axes, ("qaoa", "vqe")):
        for color, row in zip(COLORS, trajectories[algorithm]):
            ax.plot(row["sweep"], row["exact_value"], color=color, linewidth=1.8,
                    label=f"seed {row['seed']}")
            ax.scatter(row["sweep"], row["sample_value"], color=color, s=12, alpha=0.45)
        for boundary in (8, 16, 24):
            ax.axvline(boundary, color="#aeb8bc", linewidth=0.7, alpha=0.5)
        ax.set(xlabel="Prescribed sweep (fresh incumbent and candidate measurements)",
               ylabel="Expected / sampled cut" if algorithm == "qaoa" else "Dimensionless energy",
               title="Triangle QAOA: same-start seeded search" if algorithm == "qaoa" else "Changed-H VQE: same-start seeded search")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=8, loc="best")
    fig.text(0.5, 0.0, "Lines: exact diagnostics of chosen states, computed after decisions. Dots: sampled estimates at chosen states.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output / "trajectories.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_readout(output: Path, cases: dict) -> None:
    """Fresh validation draws, compared with exact post-run diagnostics."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    for ax, algorithm in zip(axes, ("qaoa", "vqe")):
        for index, (color, row) in enumerate(zip(COLORS, cases[algorithm])):
            ax.errorbar(index - 0.07, row["validation_value"], yerr=row["validation_se"],
                        color=color, marker="o", capsize=4, linestyle="none",
                        label="fresh validation mean ± estimated SE" if index == 0 else None)
            ax.scatter(index + 0.07, row["exact_value"], color="#2d3538", marker="x", s=38,
                       label="post-run exact diagnostic" if index == 0 else None, zorder=4)
        target = 2.0 if algorithm == "qaoa" else -np.sqrt(2.12)
        ax.axhline(target, color="#777777", linewidth=1, linestyle="--",
                   label="classical cut maximum" if algorithm == "qaoa" else "exact ground energy")
        ax.set(xticks=range(len(cases[algorithm])),
               xticklabels=[f"seed {r['seed']}" for r in cases[algorithm]],
               ylabel="Cut value" if algorithm == "qaoa" else "Dimensionless energy",
               title="Triangle QAOA: fresh validation" if algorithm == "qaoa" else "Changed-H VQE: fresh validation")
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.15)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    save_figure(fig, output, "readout.svg")


def summary_markdown(cases: dict, workflow_seconds: float) -> str:
    lines = ["This portability check applies the same bounded sampled optimizer to two new toy inputs: "
             "the three-vertex triangle MaxCut graph with all three edges, and "
             "the Hamiltonian H = -0.8 Z0 - 0.6 Z1 + 0.4 X0X1 with the same RY/CX ansatz. "
             "All eight classical cut assignments give optimum 2; full matrix diagonalization gives "
             "ground energy -sqrt(2.12). Changed problem inputs prevent a same-task resource comparison with earlier reports.", "",
             "These runs optimize **sampled** objectives. Ideal measurement outcomes are generated from "
             "statevector probabilities; exact expectations are used only for post-decision diagnostics. "
             "They are modeled ideal measurements, rather than executed hardware shots.", "",
             "The bounded schedule contains four blocks of eight sweeps: "
             "(step radians, shots/group) = (0.4, 256), (0.2, 512), (0.1, 1024), (0.05, 2048). "
             "The initial objective uses 256 shots/group. Each sweep freshly measures its incumbent "
             "and both coordinate proposals for every parameter, including all rejected proposals. "
             "A candidate is accepted only when the sampled decrease exceeds twice the combined estimated "
             "standard error; this margin is a heuristic, without a confidence or convergence guarantee across adaptive choices. "
             "Every run stops after the prescribed 32 sweeps.", "",
             "Seeds 7, 19 and 41 use the same fixed initialization for each task: QAOA (0.2, 0.1), VQE (0.6). "
             "Only the seeded measurement randomness changes within each task. Three observed runs do not certify a population "
             "success probability. No target-based retuning is performed. Fresh validation uses 4,096 shots "
             "per group and a separate keyed random-stream family after training ends.", "",
             "| Task / seed | Initial angles | Final angles (radians) | Training calls / shots | Fresh validation shots | Total modeled shots | Training optimizer wall (s) | Classical control (s) |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("qaoa", "vqe"):
        for row in cases[algorithm]:
            initial = ", ".join(f"{x:g}" for x in row["initial"])
            parameters = ", ".join(f"{x:.8f}" for x in row["parameters"])
            lines.append(f"| {algorithm.upper()} / {row['seed']} | {initial} | {parameters} | "
                         f"{row['training_calls']} / {row['training_shots']} | {row['validation_shots']} | "
                         f"{row['total_shots']} | {row['optimizer_seconds']:.6f} | {row['classical_control_seconds']:.6f} |")
    lines += ["", "Fresh validation estimates the energy or expected cut of the final, already selected state. "
              "The estimated standard error includes within-group covariance; it is not a rigorous interval. "
              "A zero empirical standard error can occur when a finite sample contains only one score and does not certify the exact mean. "
              "The separate Hoeffding radius is an ideal-sampling 95% bound for each validation row, not simultaneous table coverage.", "",
              "| Task / seed | Post-run exact diagnostic | Fresh validation mean | Error from diagnostic | Estimated SE | Hoeffding 95% radius | Post-run toy target met | ±0.10 readout target supported |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for algorithm in ("qaoa", "vqe"):
        for row in cases[algorithm]:
            lines.append(f"| {algorithm.upper()} / {row['seed']} | {row['exact_value']:.9f} | "
                         f"{row['validation_value']:.8f} | {row['validation_value'] - row['exact_value']:+.6f} | "
                         f"{row['validation_se']:.6f} | {row['hoeffding_radius']:.6f} | {row['target_met']} | {row['precision_target_supported']} |")
    lines += ["", "The post-run numerical targets are exact expected cut at least 1.90 for QAOA and exact energy gap "
              "at most 0.03 for VQE. These diagnostics assess the selected toy states after training; target flags "
              "are not guarantees derived from the validation confidence bound. Separately, the fresh validation "
              "has an absolute expectation-error target of 0.10 at 95% confidence for one fixed estimate. "
              "The displayed Hoeffding radii determine whether that readout allocation supports this precision target."]
    for row in cases["qaoa"]:
        lines += ["", f"QAOA seed {row['seed']}: fresh validation observed best cut {row['best_observed_cut']} "
                  f"(classical optimum 2); the sampled optimal-cut fraction was {row['optimal_cut_sample_fraction']:.6f}. "
                  f"The fixed final state's exact optimal-cut probability, audited after training, was {row['optimal_cut_probability']:.6f}."]
    total_calls = sum(row["training_calls"] + 1 for rows in cases.values() for row in rows)
    total_shots = sum(row["total_shots"] for rows in cases.values() for row in rows)
    lines += ["", f"The complete record contains **{total_calls} sampled objective calls** including six fresh validations "
              f"and **{total_shots:,} modeled circuit shots**. The full CPU workflow took {workflow_seconds:.6f} s. "
              "Simulator/probability generation and classical search work are software timings rather than quantum runtime. "
              "Logical counts repeat every preparation, measurement basis change, and readout for all training and validation shots. "
              "Arbitrary rotations remain unsynthesized. Physical reset/measurement timing, routing, faults, error correction, factories, "
              "hardware latency, noise and mitigation are excluded; physical resource estimates remain pending.", ""]
    return "\n".join(lines)


def normalized_results(demo: dict) -> tuple[dict, dict]:
    """Prepare displays only after the sampled optimizer has completed."""
    cases = {"qaoa": [], "vqe": []}
    trajectories = {"qaoa": [], "vqe": []}
    for algorithm in ("qaoa", "vqe"):
        sign = -1 if algorithm == "qaoa" else 1
        for run in demo[algorithm]["runs"]:
            validation = run["validation"]
            row = dict(seed=run["seed"], initial=run["initial"], parameters=run["parameters"],
                       training_calls=run["objective_evaluations"], training_shots=run["total_shots"],
                       validation_shots=validation["total_shots"], total_shots=run["total_shots_including_validation"],
                       optimizer_seconds=run["optimizer_wall_seconds"],
                       classical_control_seconds=run["classical_control_seconds"],
                       exact_value=sign * validation["exact_objective_audit"],
                       validation_value=sign * validation["objective_estimate"],
                       validation_se=validation["estimated_standard_error"],
                       hoeffding_radius=validation["hoeffding_95_absolute_mean_radius"],
                       precision_target_supported=validation["precision_target_supported_by_bound"],
                       target_met=run["target_met_audit"])
            if algorithm == "qaoa":
                row.update(best_observed_cut=validation["best_observed_cut"],
                           optimal_cut_sample_fraction=validation["optimal_cut_sample_fraction"],
                           optimal_cut_probability=run["optimal_cut_probability_audit"])
            cases[algorithm].append(row)
            first = run["evaluations"][0]
            trajectory = dict(seed=run["seed"], sweep=[0],
                              exact_value=[sign * first["exact_objective_audit"]],
                              sample_value=[sign * first["objective_estimate"]])
            for iteration in run["iterations"]:
                chosen = (iteration["best_proposal_evaluation"] if iteration["accepted"]
                          else iteration["incumbent_evaluation"])
                evaluation = run["evaluations"][chosen]
                trajectory["sweep"].append(iteration["index"] + 1)
                trajectory["exact_value"].append(sign * iteration["exact_objective_after_audit"])
                trajectory["sample_value"].append(sign * evaluation["objective_estimate"])
            trajectories[algorithm].append(trajectory)
    return cases, trajectories


def write_evaluations(output: Path, demo: dict) -> int:
    rows = 0
    with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "run", "seed", "phase", "evaluation", "role", "parameters_radians",
                         "shots_per_group", "total_shots", "minimized_sampled_objective", "estimated_standard_error",
                         "post_run_exact_objective_audit", "evaluation_wall_seconds", "accepted"])
        for algorithm in ("qaoa", "vqe"):
            for run in demo[algorithm]["runs"]:
                for phase, calls in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                    for index, call in enumerate(calls):
                        writer.writerow([algorithm, run["run_index"], run["seed"], phase, index, call["role"],
                                         json.dumps(call["parameters"]), call["groups"][0]["shots"], call["total_shots"],
                                         call["objective_estimate"], call["estimated_standard_error"], call["exact_objective_audit"],
                                         call["evaluation_wall_seconds"], call.get("accepted", False)])
                        rows += 1
    return rows


def checked_demo() -> dict:
    """Run sampled decisions, then independently verify final diagnostics."""
    demo = opt.run_portability_demo()
    for algorithm in ("qaoa", "vqe"):
        for run in demo[algorithm]["runs"]:
            parameters = run["parameters"]
            if algorithm == "qaoa":
                exact = opt.qaoa_expectation(3, ((0, 1), (1, 2), (0, 2)), [parameters[0]], [parameters[1]])
            else:
                state = Statevector.from_instruction(opt.build_vqe_circuit(parameters[0])).data
                matrix = portability_hamiltonian()
                exact = float(np.vdot(state, matrix @ state).real)
            expected = (-run["validation"]["exact_objective_audit"] if algorithm == "qaoa"
                        else run["validation"]["exact_objective_audit"])
            if abs(exact - expected) > 1e-10:
                raise AssertionError("post-run analytic diagnostic disagrees with independent circuit expectation")
    values = opt.maxcut_values(3, ((0, 1), (1, 2), (0, 2)))
    if int(values.max()) != demo["qaoa"]["classical"]["optimum"]:
        raise AssertionError("sampled QAOA classical reference disagrees with exhaustive enumeration")
    if abs(float(np.linalg.eigvalsh(portability_hamiltonian())[0]) - demo["vqe"]["reference"]["ground_energy"]) > 1e-12:
        raise AssertionError("sampled VQE reference disagrees with exact diagonalization")
    demo["artifact_post_run_independent_checks"] = dict(
        performed=True, methods=["final circuit expectations versus scalar diagnostic",
                                 "all8 classical cut assignments", "changed-H explicit4x4 exact diagonalization"],
        used_for_training=False, shots=0,
        timing_scope="these artifact checks are outside run_portability_demo workflow_wall_seconds")
    return demo


def portability_hamiltonian() -> np.ndarray:
    """Independent explicit reference, in |q1 q0> order, with no optimizer role."""
    return np.array([[-1.4, 0.0, 0.0, 0.4], [0.0, 0.2, 0.4, 0.0],
                     [0.0, 0.4, -0.2, 0.0], [0.4, 0.0, 0.0, 1.4]])


def generate(output: Path) -> dict:
    begin = perf_counter()
    demo = checked_demo()
    resources = opt.sampled_run_resources(demo)
    resources["fixed_readout_projection_scope"] = "none for these changed instances; physical estimates require a separate explicit contract"
    info = provenance()
    demo["provenance"] = info
    resources["provenance"] = info
    folder = output
    copy_downloads(folder)
    exports = []
    for run in demo["qaoa"]["runs"]:
        gamma, beta = run["parameters"]
        exports.append(export_circuit(folder, f"qaoa-seed{run['seed']}.qasm",
                                      opt.build_qaoa_circuit(3, ((0, 1), (1, 2), (0, 2)), [gamma], [beta], measure=True)))
    for run in demo["vqe"]["runs"]:
        for basis in ("z", "x"):
            exports.append(export_circuit(folder, f"vqe-{basis}-seed{run['seed']}.qasm",
                                          opt.build_vqe_circuit(run["parameters"][0], basis=basis, measure=True)))
    resources["final_validation_circuit_exports"] = exports
    cases, trajectories = normalized_results(demo)
    draw_trajectories(output, trajectories)
    draw_readout(output, cases)
    rows = write_evaluations(output, demo)
    (output / "summary.md").write_text(summary_markdown(cases, demo["workflow_wall_seconds"]), encoding="utf-8")
    write_json(output / "runs.json", demo)
    write_json(output / "resources.json", resources)
    return dict(mode="same-initialization portability check; ideal finite-shot objective decisions",
                objective_call_rows=rows, total_modeled_circuit_shots=demo["total_shots"],
                training_shots=demo["training_shots"], validation_shots=demo["validation_shots"],
                target_checks={algorithm: [r["target_met_audit"] for r in demo[algorithm]["runs"]]
                               for algorithm in ("qaoa", "vqe")},
                physical_resource_estimate=None, generation_wall_seconds=perf_counter() - begin)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ("assets/generated/optimization/portability" if CHECKOUT else "portability-artifacts"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(generate(args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
