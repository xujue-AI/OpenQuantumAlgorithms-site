#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate separate digitized-annealing circuits, checks, and readout artifacts.

From a checkout: python scripts/generate_optimization_annealing.py
From public downloads: place this file, the matching annealing optimization.py export,
LICENSE.txt and the environment lock together, then execute this script.
The main evolution time and step count are fixed before running the studies.
Previous optimization reports and code snapshots are preserved.

Cycle export without simulation: python scripts/generate_optimization_annealing.py \
    --no-sim --qubits 16 --steps 8 --total-time 1 --output-dir artifacts/annealing-16
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
    raise SystemExit("Save the annealing optimization.py export beside this script, or run from a checkout")
spec = importlib.util.spec_from_file_location("optimization_annealing_artifact_source", SOURCE)
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
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-digitized-annealing"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance() -> dict:
    packages = {}
    for name in ("qiskit", "numpy", "scipy", "matplotlib", "jupyter-book"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = ROOT / "requirements-lock.txt"
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), packages=packages,
                  device=dict(backend="CPU Qiskit Statevector, SciPy DOP853 reference, and NumPy PCG64 ideal readout",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  implementation_sha256=digest(SOURCE), generator_sha256=digest(SCRIPT),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None,
                  invocation="python scripts/generate_optimization_annealing.py" if CHECKOUT
                  else "python generate_optimization_annealing.py")
    if CHECKOUT:
        record["source_commit"] = git_revision(ROOT)
        record["dirty_owned_source"] = git_dirty(ROOT, ['src/openquantum/optimization.py', 'scripts/generate_optimization_annealing.py', 'notebooks/optimization_annealing.ipynb'])
    record["revision_note"] = ("source_commit identifies the checkout base at generation; content hashes "
                               "identify the implementation and generator actually executed")
    return record


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy_downloads(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "optimization.py").write_bytes(SOURCE.read_bytes())
    (folder / "generate_optimization_annealing.py").write_bytes(SCRIPT.read_bytes())
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


def draw_convergence(output: Path, time_rows: list[dict], step_rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    axes[0].plot([r["total_time"] for r in time_rows], [r["digitized_optimal_probability"] for r in time_rows],
                 marker="o", color=COLORS[0], label="digitized circuit, L=256")
    axes[0].plot([r["total_time"] for r in time_rows], [r["continuous_optimal_probability"] for r in time_rows],
                 marker="x", linestyle="--", color=COLORS[1], label="independent continuous ODE")
    axes[0].axhline(0.95, color="#777777", linestyle=":", linewidth=1, label="predeclared main target")
    axes[0].set(xlabel="Evolution time T (Hamiltonian units; ℏ=1)", ylabel="Ideal probability of a maximum cut",
                title="Finite evolution time: same L=256")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([r["total_time"] for r in time_rows], [f"{r['total_time']:g}" for r in time_rows])
    axes[0].legend(fontsize=8, loc="best")
    for key, color, label in (("state_infidelity", COLORS[0], "state infidelity vs ODE"),
                              ("cut_mean_absolute_error", COLORS[1], "absolute mean-cut error vs ODE")):
        axes[1].loglog([r["steps"] for r in step_rows], [r[key] for r in step_rows],
                       marker="o", color=color, label=label)
    axes[1].set(xlabel="Midpoint Strang steps L", ylabel="State / observable error", title="Digitization: fixed T=4")
    axes[1].set_xticks([r["steps"] for r in step_rows], [str(r["steps"]) for r in step_rows])
    axes[1].legend(fontsize=8, loc="best")
    for ax in axes:
        ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output, "convergence.svg")


def draw_gap(output: Path, gap_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot([r["s"] for r in gap_rows], [r["full_gap"] for r in gap_rows],
            color=COLORS[1], label="full 16-dimensional spectral gap")
    ax.plot([r["s"] for r in gap_rows], [r["plus_sector_gap"] for r in gap_rows],
            color=COLORS[0], label="G=+1 sector gap (dimension 8)")
    ax.scatter([1], [0], marker="o", color=COLORS[1], s=25, zorder=4)
    ax.set(xlabel="Linear interpolation parameter s=t/T", ylabel="Sampled-grid energy gap",
           title="Finite-grid spectral diagnostic: C4 annealing")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(fontsize=9, loc="best")
    save_figure(fig, output, "gap.svg")


def draw_readout(output: Path, main: dict, readout_rows: list[dict]) -> None:
    first = readout_rows[0]
    counts = np.array([first["counts"].get(f"{i:04b}", 0) for i in range(16)])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.55, 1]})
    x = np.arange(16)
    axes[0].bar(x - 0.18, main["probabilities"], width=0.36, color=COLORS[0], label="main circuit exact ideal probabilities")
    axes[0].bar(x + 0.18, counts / first["shots"], width=0.36, color="#b5c5ca",
                label=f"ideal readout, seed {first['seed']}")
    axes[0].set(xticks=x, xticklabels=[f"{i:04b}" for i in x], ylabel="Probability / sample frequency",
                xlabel="Bitstring q3 q2 q1 q0", title="Main T=16, L=256 circuit readout")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend(fontsize=8, loc="best")
    for index, (color, row) in enumerate(zip(COLORS, readout_rows)):
        axes[1].errorbar(index, row["cut_mean"], yerr=row["cut_standard_error"],
                         marker="o", capsize=4, color=color, linestyle="none")
    axes[1].axhline(main["expected_cut"], color=COLORS[0], linestyle="--", linewidth=1, label="exact main expected cut")
    axes[1].axhline(4, color="#777777", linestyle=":", linewidth=1, label="classical cut maximum")
    axes[1].set(xticks=range(len(readout_rows)), xticklabels=[f"seed {r['seed']}" for r in readout_rows],
                ylabel="Mean cut ± estimated SE", title="Three fresh ideal readouts")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[1].legend(fontsize=8, loc="best")
    axes[1].grid(axis="y", alpha=0.15)
    save_figure(fig, output, "readout.svg")


def summary_markdown(main: dict, time_rows: list[dict], step_rows: list[dict],
                     gap_rows: list[dict], readout_rows: list[dict], resources: dict) -> str:
    lines = ["The fixed main digitized anneal uses **T=16, L=256**, chosen before the studies. "
             "The input is the four-vertex cycle and the initial state is |+> on each qubit. "
             "H_D = -sum X, H_P = 0.5 sum ZZ = 2-C, and the linear path is H(s) = (1-s) H_D + s H_P. "
             "Evolution times are Hamiltonian units with ℏ=1, rather than hardware execution times.", "",
             f"The main circuit's exact ideal maximum-cut probability is **{main['optimal_probability']:.9f}** "
             f"and mean cut is **{main['expected_cut']:.9f}**. The predeclared probability target ≥0.95 "
             f"was met: **{main['target_met']}**. The independent continuous DOP853 reference has maximum-cut "
             f"probability {main['continuous_optimal_probability']:.9f}. Circuit-versus-reference state infidelity "
             f"is {main['state_infidelity']:.3g}; absolute mean-cut error is {main['cut_mean_absolute_error']:.3g}. "
             "These are state-specific numerical checks, rather than operator-error or global adiabatic guarantees.", "",
             "| Time study, fixed L=256 | Circuit P(max cut) | Continuous ODE P(max cut) | State infidelity vs ODE | Absolute mean-cut error vs ODE |",
             "|---:|---:|---:|---:|---:|"]
    for row in time_rows:
        lines.append(f"| {row['total_time']:g} | {row['digitized_optimal_probability']:.9f} | "
                     f"{row['continuous_optimal_probability']:.9f} | {row['state_infidelity']:.3g} | {row['cut_mean_absolute_error']:.3g} |")
    solver = main["continuous_solver"]
    lines += ["", f"The main continuous reference uses SciPy DOP853 with rtol={solver['rtol']:g}, "
              f"atol={solver['atol']:g}, {solver['nfev']} derivative evaluations, and recorded norm drift "
              f"{solver['norm_error']:.3g}. Solver tolerances are numerical settings, not certified error bounds."]
    lines += ["", "| Step study, fixed T=4 | Circuit P(max cut) | State infidelity vs ODE | Absolute mean-cut error vs ODE |",
              "|---:|---:|---:|---:|"]
    for row in step_rows:
        lines.append(f"| {row['steps']} | {row['digitized_optimal_probability']:.9f} | "
                     f"{row['state_infidelity']:.3g} | {row['cut_mean_absolute_error']:.3g} |")
    minimum = min(row["plus_sector_gap"] for row in gap_rows)
    lines += ["", f"The 201-point gap scan has a smallest sampled G=+1 sector gap of {minimum:.9f}. "
              "The full gap is zero at s=1 because two cut states are degenerate. The initial state lies in "
              "the conserved G=X⊗4, G=+1 sector. A positive finite-grid sector diagnostic does not certify "
              "a gap everywhere, an adiabatic runtime theorem, or behavior on another graph.", "",
              "All time/step studies and dense ODE calculations are classical simulations with **zero measurement shots**. "
              "The three main-circuit readouts below use ideal multinomial counts, not hardware executions. "
              "Their sample variance scores each complete cut jointly; estimated standard errors are descriptive.", "",
              "| Seed | Modeled main-circuit shots | Sample mean cut | Estimated SE | Maximum-cut fraction | Best observed cut |",
              "|---:|---:|---:|---:|---:|---:|"]
    for row in readout_rows:
        lines.append(f"| {row['seed']} | {row['shots']} | {row['cut_mean']:.8f} | {row['cut_standard_error']:.6f} | "
                     f"{row['optimal_fraction']:.6f} | {row['best_observed_cut']} |")
    counts = resources["per_preparation_gate_counts"]
    total_shots = sum(row["shots"] for row in readout_rows)
    lines += ["", f"Per main circuit: H {counts.get('h', 0)}, RX {counts.get('rx', 0)}, RZ {counts.get('rz', 0)}, "
              f"CX {counts.get('cx', 0)}, measurement {counts.get('measure', 0)}. The modeled protocol totals "
              f"**{total_shots:,} fresh main-circuit preparations** and **{4 * total_shots:,} qubit readouts**. "
              "The ledger repeats every preparation, driver rotation, explicit edge interaction, and final measurement for all shots. "
              "Physical reset/measurement time, routing, rotation synthesis, fault correction and hardware runtime are excluded. "
              "Arbitrary rotations remain unsynthesized in this logical ledger. Any conditional physical projection is a separate report.", ""]
    lines += [f"The recorded bounded CPU workflow took {main['workflow_wall_seconds']:.6f} s for circuit simulation, "
              "dense references, gap diagnostics and ideal sampling. Artifact drawing, export and logical-ledger compilation "
              "are outside that timer; it is not quantum hardware time.", ""]
    return "\n".join(lines)


def normalized_results(demo: dict) -> tuple[dict, list[dict], list[dict], list[dict]]:
    main = dict(demo["main"])
    main["target_met"] = demo["target"]["achieved"]
    main["workflow_wall_seconds"] = demo["workflow_wall_seconds"]
    main["cut_mean_absolute_error"] = abs(main["expected_cut"] - main["continuous_expected_cut"])
    time_rows = []
    for row in demo["time_study"]:
        normalized = dict(row)
        normalized["cut_mean_absolute_error"] = abs(row["digitized_expected_cut"] - row["continuous_expected_cut"])
        time_rows.append(normalized)
    step_rows = [dict(row, digitized_optimal_probability=row["optimal_probability"])
                 for row in demo["step_study"]]
    readout_rows = [dict(row, optimal_fraction=row["optimal_frequency"])
                    for row in demo["readout"]]
    return main, time_rows, step_rows, readout_rows


def checked_demo() -> dict:
    demo = opt.run_annealing_demo()
    graph = ((0, 1), (1, 2), (2, 3), (0, 3))
    values = np.array([sum(((bits >> u) & 1) != ((bits >> v) & 1) for u, v in graph)
                       for bits in range(16)], dtype=float)
    main = demo["main"]
    probabilities = np.array(main["probabilities"])
    if abs(float(probabilities @ values) - main["expected_cut"]) > 1e-12:
        raise AssertionError("main expectation disagrees with independent cut scoring")
    if abs(float(probabilities[values == 4].sum()) - main["optimal_probability"]) > 1e-12:
        raise AssertionError("main success disagrees with classical maximum-cut classification")
    if main["dense_strang_state_error"] > 1e-10:
        raise AssertionError("explicit circuit differs from independent dense Strang reference")
    for row in demo["readout"]:
        counts = np.array([row["counts"].get(f"{i:04b}", 0) for i in range(16)])
        if int(counts.sum()) != row["shots"]:
            raise AssertionError("readout counts do not match its stated shot count")
        mean = float(counts @ values / row["shots"])
        variance = float(counts @ ((values - mean) ** 2) / (row["shots"] - 1))
        if abs(mean - row["cut_mean"]) > 1e-12 or abs(math.sqrt(variance / row["shots"]) - row["cut_standard_error"]) > 1e-12:
            raise AssertionError("readout means/variance disagree with whole-cut count scores")
    for row in demo["time_study"] + demo["step_study"]:
        row["measurement_shots"] = 0
    demo["artifact_checks"] = dict(performed=True,
                                   methods=["independent all16 assignment cut scoring",
                                            "explicit circuit versus independent dense Strang reference",
                                            "whole-cut statistics reconstructed from every readout count"],
                                   used_to_retune_main_parameters=False,
                                   timing_scope="artifact checks are outside the recorded demo workflow timer")
    return demo


def generate(output: Path) -> dict:
    begin = perf_counter()
    demo = checked_demo()
    resources = opt.annealing_run_resources(demo)
    resources["status"] = "pre-synthesis logical workload; any conditional physical projection is a separate report"
    info = provenance()
    demo["provenance"] = info
    resources["provenance"] = info
    task, parameters = demo["task"], demo["main"]
    measured = opt.build_annealing_circuit(task["n_qubits"], task["edges"],
                                          parameters["total_time"], parameters["steps"], measure=True)
    resources["circuit_exports"] = [export_circuit(output, "annealing.qasm", measured)]
    resources["classical_studies"] = dict(time_circuit_simulations=len(demo["time_study"]),
                                         step_circuit_simulations=len(demo["step_study"]),
                                         continuous_ode_solves=len(demo["time_study"]),
                                         dense_strang_main_checks=1, gap_scan_points=len(demo["gap_scan"]),
                                         measurement_shots=0,
                                         role="bounded CPU checks, separate from modeled main-circuit readout")
    coarse = opt.build_annealing_circuit(task["n_qubits"], task["edges"], 16.0, 2, measure=True)
    demo["display_circuit"] = dict(total_time=16.0, steps=2, file="circuit.svg",
                                   role="coarse gate illustration; differs from the main L=256 circuit",
                                   gate_counts=dict(coarse.count_ops()), state_quality_not_evaluated=True)
    fig = coarse.draw(output="mpl", fold=18, scale=0.8)
    fig.suptitle("Coarse gate illustration: T=16, L=2 (main run uses L=256)", y=1.02, fontsize=11)
    fig.savefig(output / "circuit.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    main, time_rows, step_rows, readout_rows = normalized_results(demo)
    draw_convergence(output, time_rows, step_rows)
    draw_gap(output, demo["gap_scan"])
    draw_readout(output, main, readout_rows)
    write_rows(output / "time-study.csv", time_rows)
    write_rows(output / "step-study.csv", step_rows)
    write_rows(output / "gap.csv", demo["gap_scan"])
    (output / "summary.md").write_text(summary_markdown(main, time_rows, step_rows,
                                                       demo["gap_scan"], readout_rows, resources), encoding="utf-8")
    write_json(output / "runs.json", demo)
    write_json(output / "resources.json", resources)
    return dict(mode="checked digitized linear annealing; ideal circuit and readout simulation",
                main_total_time=main["total_time"], main_steps=main["steps"],
                main_optimal_probability=main["optimal_probability"], target_met=main["target_met"],
                state_infidelity_vs_continuous=main["state_infidelity"],
                modeled_readout_preparations=resources["total_preparations"],
                physical_resource_estimate=None, generation_wall_seconds=perf_counter() - begin)


def write_rows(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def generate_only(output: Path, args) -> dict:
    # Check CLI caps before allocating edge/schedule lists.
    if not 3 <= args.qubits <= 32:
        raise ValueError("cycle circuit materialization requires 3..32 qubits")
    if not 1 <= args.steps <= 4096:
        raise ValueError("steps must lie in 1..4096")
    if not math.isfinite(args.total_time) or not 0 < args.total_time <= 64:
        raise ValueError("total_time must be finite and lie in (0,64]")
    if args.qubits + args.steps * 5 * args.qubits > 100_000:
        raise ValueError("cycle circuit exceeds the 100000 unitary-gate materialization cap")
    edges = [(i, (i + 1) % args.qubits) for i in range(args.qubits)]
    circuit = opt.build_annealing_circuit(args.qubits, edges, args.total_time, args.steps, measure=True)
    export = export_circuit(output, "annealing.qasm", circuit)
    record = dict(schema_version=1, mode="circuit materialization only; no annealing simulation or readout",
                  task=dict(graph="simple cycle", n_qubits=args.qubits, edges=edges,
                            total_time=args.total_time, steps=args.steps),
                  logical_circuit=export, simulation=dict(performed=False), readout_shots=0,
                  physical_resource_estimate=None, provenance=provenance(),
                  exclusions=["state quality, adiabatic behavior and digitization error are not evaluated",
                              "rotation synthesis, error correction, physical reset/readout time and routing"])
    write_json(output / "resources.json", record)
    return dict(mode=record["mode"], gate_counts=export["gate_counts"], simulation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ("assets/generated/optimization/annealing" if CHECKOUT else "annealing-artifacts"))
    parser.add_argument("--no-sim", action="store_true", help="export a bounded requested cycle without simulation")
    parser.add_argument("--qubits", type=int, default=4, help="cycle size for --no-sim only")
    parser.add_argument("--steps", type=int, default=8, help="midpoint Strang steps for --no-sim only")
    parser.add_argument("--total-time", type=float, default=1.0, help="Hamiltonian-unit evolution time for --no-sim only")
    args = parser.parse_args()
    if not args.no_sim and (args.qubits, args.steps, args.total_time) != (4, 8, 1.0):
        parser.error("cycle controls configure --no-sim only; the checked main run is fixed at T=16,L=256")
    if args.no_sim and (args.output_dir / "runs.json").exists():
        parser.error("choose a separate --no-sim output directory to preserve checked annealing artifacts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_downloads(args.output_dir)
    result = generate_only(args.output_dir, args) if args.no_sim else generate(args.output_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()


