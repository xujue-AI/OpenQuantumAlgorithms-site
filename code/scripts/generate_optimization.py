#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate checked optimization examples, full records, and original figures.

Checkout: python scripts/generate_optimization.py
Materialize without simulation: python scripts/generate_optimization.py \
    --no-sim --qubits 32 --layers 3 --output-dir artifacts/qaoa-32
Public downloads: save this script, optimization.py, LICENSE.txt and the
environment lock together, then run python generate_optimization.py.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
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
    raise SystemExit("Save optimization.py beside this script, or run from a source checkout")
# Load the exact source file, avoiding a different editable installation.
spec = importlib.util.spec_from_file_location("optimization_artifact_source", SOURCE)
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
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-optimization"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    lock = ROOT / "requirements-lock.txt"
    packages = {}
    for name in ("qiskit", "numpy", "matplotlib", "jupyter-book"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None  # Jupyter Book is not needed for standalone generation.
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), packages=packages,
                  device=dict(backend="Qiskit CPU Statevector and NumPy PCG64",
                              architecture=platform.machine(), operating_system=platform.system(),
                              accelerator=None),
                  implementation_sha256=digest(SOURCE), generator_sha256=digest(SCRIPT),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None)
    if CHECKOUT:
        result["source_commit"] = git_revision(ROOT)
        result["dirty_owned_source"] = git_dirty(ROOT, ['src/openquantum/optimization.py', 'scripts/generate_optimization.py', 'notebooks/optimization.ipynb', 'tests/test_optimization.py', 'algorithms/optimization.md', 'references/optimization.bib'])
    result["revision_note"] = ("source_commit identifies the checkout base at generation; "
                               "content hashes identify the implementation and generator actually executed")
    return result


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy_downloads(output: Path) -> None:
    (output / "optimization.py").write_bytes(SOURCE.read_bytes())
    (output / "generate_optimization.py").write_bytes(SCRIPT.read_bytes())
    license_path = ROOT / ("LICENSE" if CHECKOUT else "LICENSE.txt")
    if not license_path.is_file():
        raise SystemExit("Implementation license missing; save LICENSE.txt beside the public script")
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())


def export_circuit(output: Path, name: str, circuit) -> dict:
    compiled = opt.compile_rotation_basis(circuit)
    path = output / name
    path.write_text(qasm3.dumps(compiled), encoding="utf-8")
    return dict(file=name, sha256=digest(path), logical_qubits=compiled.num_qubits,
                gate_counts=dict(compiled.count_ops()), depth=compiled.depth(),
                rotations="explicit unsynthesized RX/RY/RZ angles in radians")


def save_figure(fig, output: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_figures(output: Path, demo: dict) -> None:
    q = demo["qaoa"]
    v = demo["vqe"]
    gamma, beta = q["optimization_runs"][0]["parameters"]
    theta = v["optimization_runs"][0]["parameters"][0]
    for name, circuit in (("qaoa-circuit.svg", opt.build_qaoa_circuit(4, opt.CYCLE_EDGES, [gamma], [beta], measure=True)),
                          ("vqe-circuit.svg", opt.build_vqe_circuit(theta, basis="z", measure=True))):
        fig = circuit.draw(output="mpl", fold=28, scale=0.85)
        fig.savefig(output / name, bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    for ax, algorithm in zip(axes, ("qaoa", "vqe")):
        for color, run in zip(COLORS, demo[algorithm]["optimization_runs"]):
            values = np.array([e["value"] for e in run["evaluations"]])
            incumbent = np.minimum.accumulate(values)
            if algorithm == "qaoa":
                incumbent = -incumbent
            ax.step(np.arange(1, len(values) + 1), incumbent, where="post", color=color,
                    label="start " + ", ".join(f"{x:g}" for x in run["initial"]))
        ax.set(xlabel="Exact objective evaluations (including rejected trials)",
               ylabel="Best expected cut" if algorithm == "qaoa" else "Best energy",
               title="QAOA: four-vertex cycle" if algorithm == "qaoa" else "VQE: two-qubit Hamiltonian")
        target = 3 if algorithm == "qaoa" else v["reference"]["ground_energy"]
        ax.axhline(target, color="#555555", linestyle="--", linewidth=1,
                   label="p=1 analytic maximum" if algorithm == "qaoa" else "exact ground energy")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    save_figure(fig, output, "convergence.svg")

    probabilities = Statevector.from_instruction(
        opt.build_qaoa_circuit(4, opt.CYCLE_EDGES, [gamma], [beta])).probabilities()
    sampled = q["readout"][0]
    counts = np.array([sampled["counts"].get(f"{i:04b}", 0) for i in range(16)])
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    x = np.arange(16)
    ax.bar(x - 0.18, probabilities, width=0.36, color=COLORS[0], label="exact ideal probability")
    ax.bar(x + 0.18, counts / sampled["shots"], width=0.36, color="#b7c4ca",
           label=f"ideal samples: seed {sampled['seed']}, {sampled['shots']} shots")
    ax.set(xticks=x, xticklabels=[f"{i:04b}" for i in x], ylabel="Probability / sample frequency",
           xlabel="Bitstring q3 q2 q1 q0", title="QAOA readout at the first predeclared optimized run")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(fontsize=9)
    save_figure(fig, output, "qaoa-probabilities.svg")

    angles = np.linspace(-math.pi, math.pi, 501)
    energy = -2 * np.cos(angles) + 0.5 * np.sin(angles)
    fig, ax = plt.subplots(figsize=(8.7, 3.7))
    ax.plot(angles, energy, color=COLORS[0], label=r"independent formula $-2\cos\theta+0.5\sin\theta$")
    ax.axhline(v["reference"]["ground_energy"], color="#555555", linestyle="--", linewidth=1,
               label="exact diagonalization ground energy")
    for color, run in zip(COLORS, v["optimization_runs"]):
        ax.scatter(run["parameters"][0], run["value"], color=color, s=32,
                   label=f"achieved from initial {run['initial'][0]:g}", zorder=4)
    ax.set(xlabel=r"$\theta$ (radians)", ylabel="Dimensionless energy",
           title="VQE ansatz energy and the exact ground state")
    ax.legend(fontsize=8, loc="upper right")
    save_figure(fig, output, "vqe-energy.svg")


def summary_markdown(demo: dict, ledger: dict) -> str:
    lines = ["All objective calls below used exact ideal statevectors and **zero shots**. "
             "Three initializations were fixed before execution; the first is the reported circuit. "
             "Times are local CPU/software measurements.", "",
             "| Algorithm / start | Final parameters (radians) | Exact value | Quality check | Sweeps / calls | Stop | Optimizer wall (s) | Classical control (s) |",
             "|---|---|---:|---|---:|---|---:|---:|"]
    for algorithm in ("qaoa", "vqe"):
        for index, run in enumerate(demo[algorithm]["optimization_runs"]):
            params = ", ".join(f"{x:.9f}" for x in run["parameters"])
            value = run["achieved_expectation"] if algorithm == "qaoa" else run["value"]
            quality = (f"ratio {run['approximation_ratio']:.9f}; target met {run['target_met']}"
                       if algorithm == "qaoa" else f"ground gap {run['ground_energy_gap']:.3g}; target met {run['target_met']}")
            lines.append(f"| {algorithm.upper()} / {index + 1} | {params} | {value:.10f} | {quality} | "
                         f"{run['iteration_count']} / {run['objective_evaluations']} | {run['stop_reason']} | "
                         f"{run['optimizer_wall_seconds']:.6f} | {run['classical_control_seconds']:.6f} |")
    if "hybrid_workflow_wall_seconds" in demo:
        lines += ["", f"The complete ideal toy-workflow execution took {demo['hybrid_workflow_wall_seconds']:.6f} s "
                  "on this CPU, including exact reference checks, post-fit quality checks and probability generation. "
                  "Artifact drawing/export and logical-ledger compilation are outside that workflow timer."]
    lines += ["", "Finite-shot results evaluate the first optimized state only. "
              "These are independent ideal multinomial samples, not executed hardware shots. "
              "The standard error is estimated from the joint measured observable; it is not a rigorous confidence interval. "
              "The Hoeffding radius separately gives an ideal-sampling 95% bound for each listed estimate, not simultaneous coverage across this table.", "",
              "| Algorithm / seed | Total modeled circuit shots | Estimate | Error from exact mean | Estimated SE | Hoeffding 95% radius |",
              "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("qaoa", "vqe"):
        for sampled in demo[algorithm]["readout"]:
            shots = sampled["shots"] if algorithm == "qaoa" else sampled["total_shots"]
            bound = sampled["hoeffding_95_mean_error_bound"] if algorithm == "qaoa" else sampled["hoeffding_95_energy_error_bound"]
            se = sampled["estimated_standard_error"]
            se_text = "unavailable (one shot)" if se is None else f"{se:.6f}"
            lines.append(f"| {algorithm.upper()} / {sampled['seed']} | {shots} | {sampled['estimate']:.8f} | "
                         f"{sampled['error']:+.6f} | {se_text} | {bound:.6f} |")
    qrun = demo["qaoa"]["optimization_runs"][0]
    lines += ["", f"The first QAOA state's exact probability of an optimal cut is {qrun['optimal_cut_probability']:.9f}; "
              f"{qrun['independent_trials_for_99_percent_one_optimum']} independent ideal draws suffice for at least one optimum "
              "with probability at least 0.99 at these fixed parameters. This is a readout statement, not an optimizer guarantee.", "",
              "| Whole recorded workload | Exact simulated ansatz evaluations | Exact-objective shots | Modeled held-out circuit shots | Modeled qubit readouts | Shot-weighted CX | Shot-weighted arbitrary rotations |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for algorithm, resources in ledger["algorithms"].items():
        counts = resources["shot_weighted_logical_gate_counts"]
        rotations = sum(counts.get(gate, 0) for gate in ("rx", "ry", "rz"))
        lines.append(f"| {algorithm.upper()} | {resources['exact_objective_evaluations']} | 0 | "
                     f"{resources['held_out_measurement_circuit_shots']} | {resources['measured_qubit_readouts']} | "
                     f"{counts.get('cx', 0)} | {rotations} |")
    lines += ["", "The ledger also stores each exact evaluation's parameters, logical gates and depth. "
              "These simulated circuit counts are separate from shot-weighted readout counts. "
              "Arbitrary rotations remain unsynthesized; physical qubits, T counts, factories and runtime are pending. "
              "Reset timing, routing, graph/input transfer, noise and mitigation are excluded. "
              "Reference checks are included in whole CPU workflow time but are not timed separately.", ""]
    return "\n".join(lines)


def generate_demo(output: Path, shots: int) -> dict:
    begin = perf_counter()
    demo = opt.run_demo(shots=shots)
    ledger = opt.logical_run_resources(demo)
    info = provenance()
    demo["provenance"] = info
    ledger["provenance"] = info
    gamma, beta = demo["qaoa"]["optimization_runs"][0]["parameters"]
    theta = demo["vqe"]["optimization_runs"][0]["parameters"][0]
    ledger["circuit_exports"] = [
        export_circuit(output, "qaoa.qasm", opt.build_qaoa_circuit(4, opt.CYCLE_EDGES, [gamma], [beta], measure=True)),
        export_circuit(output, "vqe-z.qasm", opt.build_vqe_circuit(theta, basis="z", measure=True)),
        export_circuit(output, "vqe-x.qasm", opt.build_vqe_circuit(theta, basis="x", measure=True))]
    write_json(output / "runs.json", demo)
    write_json(output / "logical-resources.json", ledger)
    with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "run", "evaluation", "parameters_radians", "minimized_objective", "objective_seconds", "shots"])
        for algorithm in ("qaoa", "vqe"):
            for run_index, run in enumerate(demo[algorithm]["optimization_runs"]):
                for evaluation in run["evaluations"]:
                    writer.writerow([algorithm, run_index, evaluation["index"], json.dumps(evaluation["parameters"]),
                                     evaluation["value"], evaluation["seconds"], evaluation["shots"]])
    (output / "summary.md").write_text(summary_markdown(demo, ledger), encoding="utf-8")
    draw_figures(output, demo)
    return dict(mode="checked ideal toy workflows", physical_resource_estimate=None,
                qaoa_expected_cut=demo["qaoa"]["optimization_runs"][0]["achieved_expectation"],
                vqe_energy=demo["vqe"]["optimization_runs"][0]["value"],
                all_quality_targets_met=all(r["target_met"] for alg in ("qaoa", "vqe")
                                           for r in demo[alg]["optimization_runs"]),
                generation_wall_seconds=perf_counter() - begin)


def generate_only(output: Path, args) -> dict:
    if args.qubits < 3:
        raise ValueError("a simple cycle requires at least three qubits")
    if args.layers < 1:
        raise ValueError("layers must be positive")
    # Validate before forming lists: unbounded CLI sizes must not allocate first.
    if args.qubits > 4096:
        raise ValueError("circuit materialization supports at most 4096 qubits")
    if args.qubits + args.layers * 4 * args.qubits > 100_000:
        raise ValueError("cycle circuit exceeds the 100000 unitary-gate materialization limit")
    edges = [(i, (i + 1) % args.qubits) for i in range(args.qubits)]
    circuit = opt.build_qaoa_circuit(args.qubits, edges, [args.gamma] * args.layers,
                                    [args.beta] * args.layers, measure=True)
    export = export_circuit(output, "qaoa.qasm", circuit)
    record = dict(schema_version=1, mode="circuit materialization only; no simulation or optimization",
                  instance=dict(graph="simple cycle", n_qubits=args.qubits, edges=edges,
                                layers=args.layers, gammas=[args.gamma] * args.layers,
                                betas=[args.beta] * args.layers),
                  logical_circuit=export, simulation=dict(performed=False),
                  optimizer=dict(performed=False), shots=0, physical_resource_estimate=None,
                  provenance=provenance(),
                  exclusions=["simulation, output quality and convergence are untested",
                              "rotation synthesis, error correction, routing, resets and physical timing"])
    write_json(output / "logical-resources.json", record)
    return dict(mode=record["mode"], qubits=args.qubits, layers=args.layers,
                gate_counts=export["gate_counts"], simulation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ("assets/generated/optimization" if CHECKOUT else "optimization-artifacts"))
    parser.add_argument("--shots", type=int, default=2048, help="shots per held-out measurement group")
    parser.add_argument("--no-sim", action="store_true", help="materialize the requested cycle without optimization or simulation")
    parser.add_argument("--qubits", type=int, default=4, help="cycle size for --no-sim")
    parser.add_argument("--layers", type=int, default=1, help="QAOA depth for --no-sim")
    parser.add_argument("--gamma", type=float, default=0.2, help="repeated phase angle for --no-sim")
    parser.add_argument("--beta", type=float, default=0.1, help="repeated mixer angle for --no-sim")
    args = parser.parse_args()
    if not args.no_sim and (args.qubits, args.layers, args.gamma, args.beta) != (4, 1, 0.2, 0.1):
        parser.error("--qubits/--layers/--gamma/--beta configure --no-sim only; checked default demos are fixed")
    if args.no_sim and (args.output_dir / "runs.json").exists():
        parser.error("choose a separate output directory for --no-sim; it must not overwrite checked toy artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_downloads(args.output_dir)
    result = generate_only(args.output_dir, args) if args.no_sim else generate_demo(args.output_dir, args.shots)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
