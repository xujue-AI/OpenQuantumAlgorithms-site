#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a frozen bounded QAOA/VQE quality-and-workload study with full group traces.

From a checkout: python scripts/generate_optimization_quality_scaling.py
From public downloads: place this file, the matching quality-scaling optimization.py export,
LICENSE.txt and the environment lock together, then execute this script.
The graph/Hamiltonian families, optimizer schedule, seeds and holdout budget
are frozen before execution. All target misses and all group traces are retained.
Previous reports and code snapshots are preserved.

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
from time import perf_counter

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = (PROJECT / "src/openquantum/optimization.py").is_file()
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
SOURCE = ROOT / "src/openquantum/optimization.py" if CHECKOUT else ROOT / "optimization.py"
if not SOURCE.is_file():
    raise SystemExit("Save the quality-scaling optimization.py export beside this script, or run from a checkout")
spec = importlib.util.spec_from_file_location("optimization_quality_scaling_artifact_source", SOURCE)
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
EXECUTED_SOURCE_BYTES = SOURCE.read_bytes()
EXECUTED_GENERATOR_BYTES = SCRIPT.read_bytes()
exec(compile(EXECUTED_SOURCE_BYTES, str(SOURCE), "exec"), opt.__dict__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector

COLORS = ("#176e70", "#bd6d36", "#6a6699")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-quality-scaling"})


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
                  device=dict(backend="CPU Qiskit Statevector and NumPy PCG64 ideal group sampling; dense classical references",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  implementation_sha256=hashlib.sha256(EXECUTED_SOURCE_BYTES).hexdigest(),
                  generator_sha256=hashlib.sha256(EXECUTED_GENERATOR_BYTES).hexdigest(),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None,
                  invocation="python scripts/generate_optimization_quality_scaling.py" if CHECKOUT
                  else "python generate_optimization_quality_scaling.py")
    helper = ROOT / "scripts/source_provenance.py"
    if CHECKOUT and helper.is_file():
        helper_spec = importlib.util.spec_from_file_location("quality_source_provenance", helper)
        helper_module = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(helper_module)
        record["source_commit"] = helper_module.git_revision(ROOT)
        record["dirty_owned_source"] = helper_module.git_dirty(ROOT, [
            "src/openquantum/optimization.py", "scripts/generate_optimization_quality_scaling.py",
            "notebooks/optimization_quality_scaling.ipynb"])
        record["git_provenance_helper_sha256"] = digest(helper)
    else:
        record["git_provenance_helper_sha256"] = None
    record["revision_note"] = ("source_commit identifies the checkout base at generation; content hashes "
                               "identify the implementation and generator actually executed")
    record["source_changed_during_generation"] = digest(SOURCE) != record["implementation_sha256"]
    return record


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy_downloads(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "optimization.py").write_bytes(EXECUTED_SOURCE_BYTES)
    (folder / "generate_optimization_quality_scaling.py").write_bytes(EXECUTED_GENERATOR_BYTES)
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


def _case_series(rows: list[dict], algorithm: str, family: str | None = None, layers: int | None = None) -> list[tuple[int, list[dict]]]:
    selected = [row for row in rows if row["algorithm"] == algorithm
                and (family is None or row["family"] == family)
                and (layers is None or row["layers"] == layers)]
    return [(n, [row for row in selected if row["n_qubits"] == n])
            for n in sorted({row["n_qubits"] for row in selected})]


def draw_quality(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.1, 4.0))
    for ax, family in zip(axes[:2], ("path", "cycle")):
        for color, layers in zip((COLORS[0], COLORS[2]), (1, 2)):
            series = _case_series(rows, "qaoa", family, layers)
            n = [item[0] for item in series]
            values = [[row["exact_quality"] for row in item[1]] for item in series]
            ax.plot(n, [np.mean(value) for value in values], color=color, marker="o", label=f"p={layers}, seed mean")
            ax.fill_between(n, [min(value) for value in values], [max(value) for value in values], color=color, alpha=0.12)
            for size, points in series:
                ax.scatter([size + offset for offset in (-0.07, 0, 0.07)],
                           [row["exact_quality"] for row in points], color=color, s=15, alpha=0.65)
        ax.axhline(0.80, color="#777777", linewidth=1, linestyle="--", label="predeclared quality target")
        ax.set(xlabel="Vertices n", ylabel="Post-run exact mean cut / classical optimum", title=f"QAOA: {family} graphs")
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.15)
    series = _case_series(rows, "vqe")
    n = [item[0] for item in series]
    values = [[row["exact_quality"] for row in item[1]] for item in series]
    axes[2].plot(n, [np.mean(value) for value in values], color=COLORS[1], marker="o", label="seed mean")
    axes[2].fill_between(n, [min(value) for value in values], [max(value) for value in values], color=COLORS[1], alpha=0.12)
    for size, points in series:
        axes[2].scatter([size + offset for offset in (-0.07, 0, 0.07)],
                        [row["exact_quality"] for row in points], color=COLORS[1], s=15, alpha=0.65)
    axes[2].axhline(0.05, color="#777777", linewidth=1, linestyle="--", label="predeclared quality target")
    axes[2].set(xlabel="Qubits n", ylabel="Post-run exact energy-density gap", title="VQE: one-layer chain ansatz")
    axes[2].legend(fontsize=8, loc="best")
    axes[2].grid(axis="y", alpha=0.15)
    fig.text(0.5, 0, "Dots retain all three seeded runs. Shading spans their observed range; it is not a confidence interval.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    fig.savefig(output / "quality.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_workload(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for color, family in zip((COLORS[0], COLORS[1]), ("path", "cycle")):
        for layers, style in ((1, "-"), (2, "--")):
            series = _case_series(rows, "qaoa", family, layers)
            axes[0].plot([n for n, _ in series], [np.mean([r["source_unitary_gates"] for r in points]) for _, points in series],
                         color=color, linestyle=style, marker="o", label=f"{family}, p={layers}")
    series = _case_series(rows, "vqe")
    axes[1].plot([n for n, _ in series], [np.mean([r["source_unitary_gates"] for r in points]) for _, points in series],
                 color=COLORS[2], marker="o", label="all training and holdout groups")
    for ax, title in zip(axes, ("QAOA: complete recorded source workload", "VQE: complete recorded source workload")):
        ax.set(xlabel="Data qubits n", ylabel="Shot-weighted source unitary gates", title=title)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output, "workload.svg")


def draw_coverage(output: Path, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    for color, family in zip((COLORS[0], COLORS[1]), ("path", "cycle")):
        series = _case_series(rows, "qaoa", family, 1)
        n = [size for size, _ in series]
        axes[0].plot(n, [points[0]["radius_simultaneous"] for _, points in series],
                     color=color, marker="o", label=f"{family}, whole-study bound")
        axes[0].plot(n, [points[0]["radius_per_run"] for _, points in series],
                     color=color, linestyle="--", marker="x", label=f"{family}, per-run bound")
    series = _case_series(rows, "vqe")
    axes[1].plot([size for size, _ in series], [points[0]["radius_simultaneous"] for _, points in series],
                 color=COLORS[2], marker="o", label="whole-study bound")
    axes[1].plot([size for size, _ in series], [points[0]["radius_per_run"] for _, points in series],
                 color=COLORS[2], linestyle="--", marker="x", label="per-run bound")
    for ax, title in zip(axes, ("QAOA: normalized holdout precision", "VQE: energy-density holdout precision")):
        ax.axhline(0.05, color="#777777", linewidth=1, linestyle=":", label="precision target ±0.05")
        ax.set(xlabel="Data qubits n", ylabel="Fixed-state Hoeffding error radius", title=title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(axis="y", alpha=0.15)
    save_figure(fig, output, "coverage.svg")


def summary_markdown(rows: list[dict], totals: dict) -> str:
    lines = ["This frozen, bounded study retains every predetermined run and every target miss. "
             "It covers 60 QAOA runs on path/cycle graphs with n = 3, 4, 5, 6, 8 and p = 1, 2, "
             "plus 12 VQE runs on n = 2, 3, 4, 6 chains. Each case uses seeds 7, 19 and 41 with the same initialization. "
             "QAOA angles start at interleaved (0.2, 0.1) per layer. VQE uses H = -sum Z + 0.5 sum nearest-neighbor XX, "
             "one layer of n RY gates followed by a CX chain, and all initial angles 0.6.", "",
             "Training uses raw sampled cut or energy scores. Exact classical optima/eigenvalues are post-run references, "
             "never optimizer decisions. The fixed schedule has four sweeps each at (step radians, shots/group) "
             "= (0.4, 256), (0.2, 512), (0.1, 1024), with 128 shots/group for the initial call. "
             "Every sweep remeasures the incumbent and both displacements of every coordinate. "
             "The two-standard-error acceptance margin remains a heuristic without a convergence guarantee.", "",
             "Fresh holdouts use 16,384 shots per measurement group. Their pointwise per-run bounds are reported "
             "separately from the whole-study allocation δ = 0.05 across 84 holdout groups. "
             "The resulting group intervals combine into conditional fixed-state ratio/energy-density intervals. "
             "A ±0.05 expectation precision bound is distinct from reaching a quality target. "
             "The audited quality targets are QAOA mean cut / classical optimum ≥0.80 and VQE energy-density gap ≤0.05.", "",
             "| Case | n | p | Audited quality range over three seeds | Exact target met | Whole-study interval supports target | Modeled shots per run | Source unitary gates per run |",
             "|---|---:|---:|---|---:|---:|---:|---|"]
    keys = sorted({(row["algorithm"], row["family"], row["n_qubits"], row["layers"]) for row in rows})
    for algorithm, family, n, layers in keys:
        points = [row for row in rows if (row["algorithm"], row["family"], row["n_qubits"], row["layers"]) == (algorithm, family, n, layers)]
        quality = [row["exact_quality"] for row in points]
        gate_counts = [row["source_unitary_gates"] for row in points]
        quality_text = f"{min(quality):.6f}–{max(quality):.6f}"
        gates_text = f"{min(gate_counts):,}" if min(gate_counts) == max(gate_counts) else f"{min(gate_counts):,}–{max(gate_counts):,}"
        label = f"QAOA {family}" if algorithm == "qaoa" else "VQE chain"
        lines.append(f"| {label} | {n} | {layers if algorithm == 'qaoa' else 'one layer'} | {quality_text} | "
                     f"{sum(row['target_met_exact'] for row in points)}/{len(points)} | "
                     f"{sum(row['target_supported_simultaneous'] for row in points)}/{len(points)} | "
                     f"{points[0]['total_shots']:,} | {gates_text} |")
    q_rows = [row for row in rows if row["algorithm"] == "qaoa"]
    v_rows = [row for row in rows if row["algorithm"] == "vqe"]
    lines += ["", f"Observed exact target passes: QAOA {sum(r['target_met_exact'] for r in q_rows)}/{len(q_rows)}, "
              f"VQE {sum(r['target_met_exact'] for r in v_rows)}/{len(v_rows)}. These counts describe the fixed seed set, "
              "rather than a population optimizer-success probability. Misses can reflect ansatz restrictions or optimizer behavior; "
              "this experiment does not separate their causes.", "",
              f"The record contains **{totals['objective_calls']:,} sampled objective calls**, **{totals['group_batches']:,} group batches**, "
              f"and **{totals['total_shots']:,} modeled circuit preparations** "
              f"({totals['training_shots']:,} training and {totals['holdout_shots']:,} fresh holdout shots). "
              "All counts are classically sampled from ideal circuit probabilities; no QPU execution or noise is measured. "
              "The source manifest links every group's parameters, basis, RNG stream, shots, exact OpenQASM hash, depth and logical gate counts. "
              "Byte-identical source circuits share a file only after SHA-256 comparison.", "",
              "The p=2 QAOA runs use more coordinates, proposals and shots than p=1 under this fixed sweep schedule. "
              "Their different costs are explicit; this is not an equal-budget comparison. Circuit loading/preparation, "
              "basis changes and every rejected/accepted evaluation are included in the source ledger. "
              "Rotations are unsynthesized, routing/reset/device latency are unmeasured, and no physical-resource comparison is inferred. "
              "CPU timings measure the simulator and classical control, not quantum hardware time.", ""]
    return "\n".join(lines)


def _local_target_status(case: dict, estimate: float, radius: float) -> str:
    if case["algorithm"] == "qaoa":
        target = case["target"]["ratio_at_least"]
        return "supported" if estimate - radius >= target else "missed" if estimate + radius < target else "inconclusive"
    target = case["target"]["density_gap_at_most"]
    return "supported" if estimate + radius <= target else "missed" if estimate - radius > target else "inconclusive"


def normalized_results(demo: dict, resources: dict) -> list[dict]:
    costs = {(row["case_id"], row["seed"]): row for row in resources["per_run"]}
    rows = []
    for case in demo["cases"]:
        for run in case["runs"]:
            diagnostic, validation = run["diagnostic"], run["validation"]
            bound = validation["quality_bound"]
            cost = costs[(case["case_id"], run["seed"])]
            exact = (diagnostic["exact_approximation_ratio"] if case["algorithm"] == "qaoa"
                     else diagnostic["energy_density_gap"])
            estimate, radius = bound["normalized_estimate"], bound["simultaneous_radius"]
            local_radius = bound["per_run_95_radius"]
            local_status = _local_target_status(case, estimate, local_radius)
            rows.append(dict(case_id=case["case_id"], algorithm=case["algorithm"], family=case["graph_family"],
                             n_qubits=case["n_qubits"], layers=case["p"], seed=run["seed"],
                             exact_quality=exact, target_met_exact=run["target_met_audit"],
                             holdout_normalized_estimate=estimate, holdout_normalized_standard_error=bound["normalized_standard_error"],
                             radius_per_run=local_radius, radius_simultaneous=radius,
                             per_run_interval=[estimate - local_radius, estimate + local_radius],
                             simultaneous_interval=bound["simultaneous_interval"],
                             target_status_per_run=local_status, target_status_simultaneous=bound["target_status"],
                             target_supported_per_run=local_status == "supported",
                             target_supported_simultaneous=bound["target_status"] == "supported",
                             exact_covered_per_run=estimate - local_radius <= exact <= estimate + local_radius,
                             exact_covered_simultaneous=estimate - radius <= exact <= estimate + radius,
                             precision_supported=bound["precision_supported"],
                             initial_parameters=run["initial"], final_parameters=run["parameters"],
                             training_calls=run["objective_evaluations"], total_calls=run["objective_evaluations"] + 1,
                             training_shots=run["total_shots"], holdout_shots=validation["total_shots"],
                             total_shots=run["total_shots_including_validation"], group_batches=cost["group_batches"],
                             source_unitary_gates=sum(number for name, number in cost["weighted_gate_counts"].items()
                                                       if name not in ("measure", "barrier", "reset")),
                             weighted_gate_counts=cost["weighted_gate_counts"],
                             optimizer_wall_seconds=run["optimizer_wall_seconds"],
                             objective_wall_seconds=run["objective_wall_seconds"],
                             classical_control_seconds=run["classical_control_seconds"],
                             independent_reference_wall_seconds=diagnostic["reference_wall_seconds"],
                             hardware_feedback_seconds=None))
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def overview_markdown(rows: list[dict], totals: dict) -> str:
    lines = ["All 72 predeclared seeded runs are retained, including target misses. "
             "The quality flags below come from independent post-training references. "
             "Holdout support is a separate simultaneous fixed-state confidence statement, not optimizer convergence.", "",
             "| Method | Runs | Post-run exact target met | Whole-study holdout supports target | ±0.05 normalized precision supported | Modeled preparations |",
             "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("qaoa", "vqe"):
        points = [row for row in rows if row["algorithm"] == algorithm]
        lines.append(f"| {algorithm.upper()} | {len(points)} | {sum(r['target_met_exact'] for r in points)} | "
                     f"{sum(r['target_supported_simultaneous'] for r in points)} | "
                     f"{sum(r['precision_supported'] for r in points)} | {sum(r['total_shots'] for r in points):,} |")
    lines += ["", f"Complete trace: {totals['objective_calls']:,} sampled objective calls, {totals['group_batches']:,} measured groups, "
              f"and {totals['total_shots']:,} modeled preparations. QAOA p=2 has more proposals and shots than p=1. "
              "These fixed-budget cases do not establish large-instance convergence, equal-budget improvement, or quantum advantage.", ""]
    return "\n".join(lines)


def export_group_traces(output: Path, demo: dict, started: float) -> dict:
    folder = output / "circuits"
    folder.mkdir(parents=True, exist_ok=True)
    circuits, groups, evaluations = {}, [], []
    max_wall = demo["recipe"]["bounds"]["workflow_wall_seconds"]
    for case in demo["cases"]:
        for run in case["runs"]:
            for phase, calls in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for evaluation_index, call in enumerate(calls):
                    evaluations.append(dict(case_id=case["case_id"], algorithm=case["algorithm"],
                                            n_qubits=case["n_qubits"], p=case["p"], seed=run["seed"],
                                            run_index=run["run_index"], phase=phase, evaluation_index=evaluation_index,
                                            role=call["role"], parameters=call["parameters"],
                                            objective_estimate=call["objective_estimate"],
                                            estimated_standard_error=call["estimated_standard_error"],
                                            total_shots=call["total_shots"], accepted=call.get("accepted", False),
                                            evaluation_wall_seconds=call["evaluation_wall_seconds"]))
                    for group_index, group in enumerate(call["groups"]):
                        if len(groups) % 128 == 0 and perf_counter() - started > max_wall:
                            raise RuntimeError("artifact generation exceeded its 120-second wall cap")
                        source = opt.quality_scaling_group_circuit(case, call["parameters"], group["group"])
                        counts = {name: int(number) for name, number in source.count_ops().items()}
                        if (counts != group["circuit_gate_counts"] or source.num_qubits != group["logical_qubits"]
                                or source.depth() != group["circuit_depth"]):
                            raise AssertionError("reconstructed source differs from recorded group metadata")
                        unitary_count = sum(number for name, number in counts.items() if name not in ("measure", "barrier", "reset"))
                        if source.num_qubits > 8 or unitary_count > 100_000:
                            raise AssertionError("recorded source violates published simulation or materialization caps")
                        text = qasm3.dumps(source)
                        raw = text.encode("utf-8")
                        sha = hashlib.sha256(raw).hexdigest()
                        file = f"circuits/{sha}.qasm"
                        path = output / file
                        if sha not in circuits:
                            path.write_bytes(raw)
                            mapping = [[source.find_bit(item.qubits[0]).index, source.find_bit(item.clbits[0]).index]
                                       for item in source.data if item.operation.name == "measure"]
                            circuits[sha] = dict(sha256=sha, file=file, logical_qubits=source.num_qubits,
                                                 gate_counts=counts, depth=source.depth(), measurement_map=mapping)
                        elif path.read_bytes() != raw:
                            raise AssertionError("QASM hash collision or source-file corruption")
                        group["source_qasm_sha256"] = sha
                        group["source_qasm_file"] = file
                        groups.append(dict(index=len(groups), case_id=case["case_id"], algorithm=case["algorithm"],
                                           n_qubits=case["n_qubits"], p=case["p"], seed=run["seed"],
                                           run_index=run["run_index"], phase=phase, evaluation_index=evaluation_index,
                                           group_index=group_index, group=group["group"], parameters=call["parameters"],
                                           shots=group["shots"], counts=group["counts"],
                                           estimate=group["estimate"], sample_variance=group["sample_variance"],
                                           estimated_standard_error=group["estimated_standard_error"],
                                           outcome_min=group["outcome_min"], outcome_max=group["outcome_max"], rng=group["rng"],
                                           source_qasm_sha256=sha, source_qasm_file=file,
                                           source_gate_counts=counts, source_depth=source.depth()))
                        if len(groups) % 1024 == 0:
                            print(json.dumps(dict(stage="source trace export", completed_groups=len(groups), unique_qasm=len(circuits))), flush=True)
    if (len(groups) != demo["recipe"]["planned"]["group_batches"]
            or len(evaluations) != demo["recipe"]["planned"]["objective_calls"]):
        raise AssertionError("source trace does not cover the complete frozen workload")
    write_csv(output / "groups.csv", groups)
    write_csv(output / "evaluations.csv", evaluations)
    manifest = dict(schema_version=1, kind="every recorded measured group and its exact source circuit",
                    source_kind="unsynthesized source OpenQASM3; no routing or lowering",
                    deduplication="exact SHA-256 and byte comparison only, after actual source reconstruction",
                    total_group_batches=len(groups), total_objective_calls=len(evaluations),
                    unique_circuit_count=len(circuits), unique_circuits=[circuits[key] for key in sorted(circuits)], groups=groups)
    write_json(output / "circuit-manifest.json", manifest)
    return manifest


def check_numerical_record(demo: dict) -> None:
    """Replay every training choice from sampled estimates, then verify totals."""
    recipe = opt.quality_scaling_recipe()
    if demo["recipe"] != recipe or not recipe["fixed_published_recipe"]:
        raise ValueError("Published generation requires the unchanged complete fixed recipe")
    if demo["exact_diagnostics_used_for_optimization"] is not False:
        raise ValueError("The recorded optimizer must use sampled objectives only")
    streams, calls, groups, shots = set(), 0, 0, 0
    for case in demo["cases"]:
        for run in case["runs"]:
            point = [(float(x) + math.pi) % (2 * math.pi) - math.pi for x in case["initial"]]
            for iteration in run["iterations"]:
                before = np.asarray(iteration["parameters_before"])
                if not np.allclose(before, point, atol=1e-12, rtol=0):
                    raise ValueError("Recorded incumbent does not follow the previous sampled choice")
                incumbent = run["evaluations"][iteration["incumbent_evaluation"]]
                proposals = [run["evaluations"][i] for i in iteration["proposal_evaluations"]]
                best = min(proposals, key=lambda e: e["objective_estimate"])
                margin = recipe["comparison_z"] * math.hypot(incumbent["estimated_standard_error"], best["estimated_standard_error"])
                accepted = incumbent["objective_estimate"] - best["objective_estimate"] > margin
                if (accepted != iteration["accepted"] or best["index"] != iteration["best_proposal_evaluation"]
                        or not math.isclose(margin, iteration["comparison_margin"], rel_tol=1e-12, abs_tol=1e-14)):
                    raise ValueError("Recorded move does not replay from its sampled values")
                point = best["parameters"] if accepted else incumbent["parameters"]
                if not np.allclose(iteration["parameters_after"], point, atol=1e-12, rtol=0):
                    raise ValueError("Recorded post-move parameters differ from the sampled choice")
            if not np.allclose(run["parameters"], point, atol=1e-12, rtol=0):
                raise ValueError("Recorded final state differs from its sampled trajectory")
            if run["validation"]["parameters"] != run["parameters"]:
                raise ValueError("Final holdout was detached from the chosen state")
            for phase, family, records in (("training", 0, run["evaluations"]), ("validation", 1, [run["validation"]])):
                for evaluation_index, record in enumerate(records):
                    calls += 1
                    if [g["group"] for g in record["groups"]] != case["groups"]:
                        raise ValueError("Recorded measurement group list changed")
                    for group_index, group in enumerate(record["groups"]):
                        groups += 1
                        shots += group["shots"]
                        stream = [run["seed"], case["stream_id"], family, evaluation_index, group_index]
                        if group["rng"]["stream_key"] != stream or tuple(stream) in streams:
                            raise ValueError("Recorded RNG allocation is missing, incorrect or reused")
                        streams.add(tuple(stream))
                        if sum(group["counts"].values()) != group["shots"]:
                            raise ValueError("Recorded measured counts omit shots")
                        if any(len(bits) != case["n_qubits"] or set(bits) - {"0", "1"}
                               or isinstance(number, bool) or not isinstance(number, int) or number <= 0
                               for bits, number in group["counts"].items()):
                            raise ValueError("Recorded count keys or multiplicities are invalid")
    if (calls != recipe["planned"]["objective_calls"] or groups != recipe["planned"]["group_batches"]
            or shots != recipe["planned"]["total_shots"]):
        raise ValueError("Numerical record does not contain the complete prescribed workload")


def preserve_recipe(output: Path) -> dict:
    """Keep a pre-existing locked recipe byte-for-byte, or create it once."""
    recipe = opt.quality_scaling_recipe()
    path = output / "recipe.json"
    if path.is_file():
        if json.loads(path.read_text()) != recipe:
            raise ValueError("Existing recipe differs from the fixed study; choose a different output directory")
    else:
        write_json(path, recipe)
    return recipe


def draw_representative_circuits(output: Path, demo: dict) -> list[dict]:
    """Display first predefined cases, not selected best-performing states."""
    records = []
    for algorithm, filename in (("qaoa", "qaoa-path3-p1-circuit.svg"), ("vqe", "vqe-chain2-z-circuit.svg")):
        case = next(c for c in demo["cases"] if c["algorithm"] == algorithm)
        run = case["runs"][0]
        source = opt.quality_scaling_group_circuit(case, run["parameters"], "all_z")
        fig = source.draw(output="mpl", fold=32, scale=0.80)
        fig.suptitle(f"{case['case_id']}, seed {run['seed']}: first prescribed final all-Z circuit", fontsize=10)
        fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
        records.append(dict(case_id=case["case_id"], seed=run["seed"], group="all_z", figure=filename,
                            parameters=run["parameters"], selection="first predetermined case and seed, not best outcome"))
    return records


def generate(output: Path, *, resume: bool = False) -> dict:
    started = perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    recipe = preserve_recipe(output)
    checkpoint = output / "raw-study.json"
    if checkpoint.exists() and not resume:
        raise ValueError("A numerical study already exists; use --resume to finish exports without repeating training")
    if (output / "runs.json").exists() and not resume:
        raise ValueError("Recorded results already exist; choose a separate directory or use --resume")
    copy_downloads(output)
    if resume:
        if not checkpoint.is_file():
            raise ValueError("--resume requires the preserved raw-study.json numerical checkpoint")
        demo = json.loads(checkpoint.read_text())
        info = demo["provenance"]
        if (info["implementation_sha256"] != hashlib.sha256(EXECUTED_SOURCE_BYTES).hexdigest()
                or info["generator_sha256"] != hashlib.sha256(EXECUTED_GENERATOR_BYTES).hexdigest()):
            raise ValueError("Resume requires the matching executed module and generator snapshots")
        print(json.dumps(dict(stage="resuming numerical checkpoint", runs=recipe["planned"]["runs"], training_repeated=False)), flush=True)
    else:
        print(json.dumps(dict(stage="starting frozen study", **recipe["planned"])), flush=True)
        demo = opt.run_quality_scaling_demo()
        info = provenance()
        if info["source_changed_during_generation"]:
            raise RuntimeError("The source changed during numerical execution; preserve the snapshot and rerun in a new directory")
        demo.update(provenance=info, execution_status="completed_ideal_sampled_cpu_study",
                    artifact_status="numerical_complete_trace_exports_pending")
        check_numerical_record(demo)
        write_json(checkpoint, demo)
        write_json(output / "runs.json", demo)
        print(json.dumps(dict(stage="numerical checkpoint saved", workflow_wall_seconds=demo["workflow_wall_seconds"],
                              qaoa_target_passes=sum(r["ideal_target_met"] for r in demo["summary_rows"] if r["algorithm"] == "qaoa"),
                              vqe_target_passes=sum(r["ideal_target_met"] for r in demo["summary_rows"] if r["algorithm"] == "vqe"))), flush=True)
    check_numerical_record(demo)
    resources = opt.quality_scaling_resources(demo)
    resources.update(provenance=info, recorded_numerical_input_sha256=digest(checkpoint))
    write_json(output / "resources.json", resources)
    manifest = export_group_traces(output, demo, started)
    rows = normalized_results(demo, resources)
    totals = dict(objective_calls=resources["objective_calls"], group_batches=resources["group_batches"],
                  total_shots=resources["total_preparations"], training_shots=demo["training_shots"],
                  holdout_shots=demo["validation_shots"], measured_qubit_readouts=resources["measured_qubit_readouts"])
    write_csv(output / "results.csv", rows)
    write_csv(output / "resources.csv", resources["per_run"])
    draw_quality(output, rows)
    draw_workload(output, rows)
    draw_coverage(output, rows)
    representatives = draw_representative_circuits(output, demo)
    (output / "summary.md").write_text(summary_markdown(rows, totals), encoding="utf-8")
    (output / "overview.md").write_text(overview_markdown(rows, totals), encoding="utf-8")
    demo.update(artifact_status="complete", source_changed_during_generation=digest(SOURCE) != info["implementation_sha256"],
                representative_circuits=representatives,
                artifact_generation_wall_seconds=perf_counter() - started,
                artifact_timing_scope="includes trace reconstruction, export and plotting; separate from CPU study/controller timings")
    if demo["source_changed_during_generation"]:
        raise RuntimeError("Source changed during artifact generation; original numerical checkpoint was preserved")
    resources.update(circuit_manifest=dict(file="circuit-manifest.json", sha256=digest(output / "circuit-manifest.json"),
                                           unique_circuit_count=manifest["unique_circuit_count"], total_groups=manifest["total_group_batches"]),
                     artifact_status="complete")
    write_json(output / "runs.json", demo)
    write_json(output / "resources.json", resources)
    return dict(status=demo["execution_status"], artifacts="complete", runs=len(rows), **totals,
                unique_source_circuits=manifest["unique_circuit_count"],
                qaoa_target_passes=sum(r["target_met_exact"] for r in rows if r["algorithm"] == "qaoa"),
                vqe_target_passes=sum(r["target_met_exact"] for r in rows if r["algorithm"] == "vqe"),
                generation_wall_seconds=perf_counter() - started)


def generate_only(output: Path, *, n_qubits: int, layers: int, family: str) -> dict:
    """Export a larger bounded cycle/path circuit without an optimizer or simulation."""
    if isinstance(n_qubits, bool) or not 3 <= n_qubits <= 4096:
        raise ValueError("circuit-only export supports 3..4096 vertices")
    if isinstance(layers, bool) or not 1 <= layers <= 2:
        raise ValueError("quality-family export supports p=1 or2")
    edges_count = n_qubits - 1 if family == "path" else n_qubits
    if n_qubits + layers * (3 * edges_count + n_qubits) > 100000:
        raise ValueError("circuit-only export exceeds the100000-unitary-gate bound")
    if (output / "runs.json").exists() or (output / "raw-study.json").exists() or (output / "recipe.json").exists():
        raise ValueError("Choose a separate circuit-only directory to preserve the locked study and results")
    edges = opt.quality_scaling_graph(n_qubits, family)
    circuit = opt.build_qaoa_circuit(n_qubits, edges, [0.2] * layers, [0.1] * layers, measure=True)
    output.mkdir(parents=True, exist_ok=True)
    copy_downloads(output)
    export = export_circuit(output, "qaoa-source.qasm", circuit)
    record = dict(schema_version=1, mode="materialization only; no simulation or quality result",
                  n_qubits=n_qubits, p=layers, graph_family=family, edges=[list(e) for e in edges],
                  source=export, simulation_performed=False, optimizer_performed=False,
                  quality_validated=False, modeled_shots=0, provenance=provenance(),
                  exclusions=["simulated achieved quality, optimizer convergence, and exhaustive reference",
                              "rotation synthesis, error correction, hardware initialization/reset, routing and feedback"])
    write_json(output / "resources.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ("assets/generated/optimization/quality-scaling" if CHECKOUT else "quality-scaling-artifacts"))
    parser.add_argument("--resume", action="store_true", help="finish exports from a matching raw-study.json without rerunning training")
    parser.add_argument("--no-sim", action="store_true", help="export a requested larger path/cycle without study execution")
    parser.add_argument("--qubits", type=int, default=16)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--family", choices=["path", "cycle"], default="cycle")
    args = parser.parse_args()
    if args.no_sim and args.resume:
        parser.error("--resume and --no-sim select different workflows")
    if not args.no_sim and (args.qubits, args.layers, args.family) != (16, 2, "cycle"):
        parser.error("size/depth/family controls apply only to --no-sim; the study recipe is fixed")
    record = (generate_only(args.output_dir, n_qubits=args.qubits, layers=args.layers, family=args.family)
              if args.no_sim else generate(args.output_dir, resume=args.resume))
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
