#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export the frozen sixth-increment initialization and sampling diagnostics.

Each completed numerical run is checkpointed before another run begins. Resume
requires the identical recipe and captured implementation/generator bytes.
Circuit exports retain every sampled group, including every rejected proposal.
Deterministic CPU controls have exact moments and virtual allocations, no counts
or actual quantum preparations. Physical projection is a separate future task.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter, process_time

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = (SCRIPT == PROJECT / "scripts/generate_optimization_diagnostics.py"
            and (PROJECT / "src/openquantum/optimization.py").is_file())
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
SOURCE = ROOT / "src/openquantum/optimization.py" if CHECKOUT else ROOT / "optimization.py"
if not SOURCE.is_file():
    raise SystemExit("Save the matching optimization.py export beside this script, or run from its checkout")
EXECUTED_SOURCE_BYTES = SOURCE.read_bytes()
EXECUTED_GENERATOR_BYTES = SCRIPT.read_bytes()
spec = importlib.util.spec_from_file_location("optimization_diagnostics_artifact_source", SOURCE)
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
exec(compile(EXECUTED_SOURCE_BYTES, str(SOURCE), "exec"), opt.__dict__)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

COLORS = {"original": "#176e70", "alternate": "#bd6d36"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-diagnostics"})


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value: dict) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_json(path: Path, value: dict) -> None:
    """Atomic replace for derived, rebuildable exports only."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


def preserve_json(path: Path, value: dict) -> None:
    """Write an immutable numerical/recipe record once, rejecting changes."""
    raw = encoded(value)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Preserved record differs: {path.name}; use a separate output directory")
        return
    with tempfile.NamedTemporaryFile(mode="wb", prefix=path.name + ".", suffix=".tmp",
                                     dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
    try:
        # Atomic exclusive publication prevents a truncated completed checkpoint
        # or silent replacement of a run from another invocation.
        os.link(temporary, path)
    finally:
        temporary.unlink()


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def provenance() -> dict:
    packages = {}
    for name in ("qiskit", "numpy", "scipy", "matplotlib", "jupyter-book", "qdk"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    lock = ROOT / "requirements-lock.txt"
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), packages=packages,
                  device=dict(backend="CPU Qiskit Statevector, NumPy PCG64 ideal sampling and independent dense references",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  implementation_sha256=hashlib.sha256(EXECUTED_SOURCE_BYTES).hexdigest(),
                  generator_sha256=hashlib.sha256(EXECUTED_GENERATOR_BYTES).hexdigest(),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None,
                  invocation="python scripts/generate_optimization_diagnostics.py" if CHECKOUT
                  else "python generate_optimization_diagnostics.py")
    helper = ROOT / "scripts/source_provenance.py"
    if CHECKOUT and helper.is_file():
        helper_spec = importlib.util.spec_from_file_location("diagnostics_source_provenance", helper)
        module = importlib.util.module_from_spec(helper_spec)
        helper_spec.loader.exec_module(module)
        record["source_commit"] = module.git_revision(ROOT)
        record["dirty_owned_source"] = module.git_dirty(ROOT, [
            "src/openquantum/optimization.py", "scripts/generate_optimization_diagnostics.py",
            "notebooks/optimization_diagnostics.ipynb"])
        record["git_provenance_helper_sha256"] = digest(helper)
    else:
        record["git_provenance_helper_sha256"] = None
    record["revision_note"] = "Checkout revision identifies the base; captured content hashes identify executed code"
    return record


def verify_environment(info: dict) -> None:
    expected = {"qiskit": "2.6.2", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2", "qdk": "1.34.0"}
    if info["python"] != "3.12.14" or any(info["packages"].get(name) != value for name, value in expected.items()):
        raise ValueError("Freeze/execution requires the reviewed Python3.12.14 and pinned Qiskit/NumPy/SciPy/Matplotlib/QDK environment")


def preserve_downloads(output: Path) -> None:
    """Preserve the exact executable bytes; never replace an earlier producer."""
    license_path = ROOT / ("LICENSE" if CHECKOUT else "LICENSE.txt")
    lock_path = ROOT / "requirements-lock.txt"
    if not license_path.is_file() or not lock_path.is_file():
        raise ValueError("Standalone reproduction requires LICENSE.txt and requirements-lock.txt")
    files = {"optimization.py": EXECUTED_SOURCE_BYTES,
             "generate_optimization_diagnostics.py": EXECUTED_GENERATOR_BYTES,
             "LICENSE.txt": license_path.read_bytes(), "requirements-lock.txt": lock_path.read_bytes()}
    for name, raw in files.items():
        path = output / name
        if path.exists():
            if path.read_bytes() != raw:
                raise ValueError(f"Resume requires the matching preserved {name}")
        else:
            with path.open("xb") as handle:
                handle.write(raw)


def checkpoint_identity(recipe: dict, info: dict) -> dict:
    return dict(recipe_sha256=hashlib.sha256(encoded(recipe)).hexdigest(),
                implementation_sha256=info["implementation_sha256"], generator_sha256=info["generator_sha256"],
                environment_lock_sha256=info["environment_lock_sha256"])


def load_checkpoint(path: Path, identity: dict, *, kind: str, key: str) -> dict:
    record = json.loads(path.read_text())
    if (record.get("identity") != identity or record.get("kind") != kind
            or record.get("key") != key or record.get("status") != "completed"):
        raise ValueError(f"Checkpoint identity, key or completion state differs: {path.name}")
    return record["run"]


def save_checkpoint(path: Path, identity: dict, *, kind: str, key: str, run: dict) -> None:
    preserve_json(path, dict(schema_version=1, kind=kind, key=key, identity=identity,
                             status="completed", run=run))


def check_checkpoint_inventory(output: Path, demo: dict, identity: dict) -> None:
    folder = output / "checkpoints"
    expected = set()
    for case in demo["cases"]:
        for run in case["runs"]:
            key = run["run_id"]
            path = folder / f"sampled-{key}.json"
            expected.add(path.name)
            if load_checkpoint(path, identity, kind="sampled", key=key) != run:
                raise ValueError("Immutable sampled checkpoint differs from numerical aggregate")
    for run in demo["controls"]:
        key = run["run_id"]
        path = folder / f"control-{key}.json"
        expected.add(path.name)
        if load_checkpoint(path, identity, kind="control", key=key) != run:
            raise ValueError("Immutable control checkpoint differs from numerical aggregate")
    if {p.name for p in folder.glob("*.json")} != expected:
        raise ValueError("Completed numerical checkpoint inventory differs from all 60 fixed runs")


def export_group_traces(output: Path, demo: dict) -> dict:
    """Reconstruct every actual measured circuit; deduplicate exact QASM only."""
    folder = output / "circuits"
    folder.mkdir(exist_ok=True)
    circuits, groups, evaluations = {}, [], []
    for case in demo["cases"]:
        for run in case["runs"]:
            for phase, calls in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for evaluation_index, call in enumerate(calls):
                    common = dict(case_id=case["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                                  p=case["p"], start_id=run["start_id"], seed=run["seed"], run_id=run["run_id"],
                                  phase=phase, evaluation_index=evaluation_index)
                    evaluations.append(dict(**common, role=call["role"], parameters=call["parameters"],
                                            objective_estimate=call["objective_estimate"],
                                            estimated_standard_error=call["estimated_standard_error"],
                                            total_shots=call["total_shots"], accepted=call.get("accepted", False),
                                            evaluation_wall_seconds=call["evaluation_wall_seconds"]))
                    for group_index, group in enumerate(call["groups"]):
                        source = opt.quality_scaling_group_circuit(case, call["parameters"], group["group"])
                        counts = {name: int(number) for name, number in source.count_ops().items()}
                        if (counts != group["circuit_gate_counts"] or source.num_qubits != group["logical_qubits"]
                                or source.depth() != group["circuit_depth"]):
                            raise ValueError("Reconstructed source differs from sampled group metadata")
                        unitary_count = sum(number for name, number in counts.items()
                                            if name not in ("measure", "barrier", "reset"))
                        if source.num_qubits > 6 or unitary_count > 100_000:
                            raise ValueError("Source violates diagnostic simulation/materialization bounds")
                        raw = qasm3.dumps(source).encode("utf-8")
                        sha = hashlib.sha256(raw).hexdigest()
                        file = f"circuits/{sha}.qasm"
                        path = output / file
                        if sha not in circuits:
                            if path.exists() and path.read_bytes() != raw:
                                raise ValueError("Preserved exact-QASM export is corrupt")
                            path.write_bytes(raw)
                            mapping = [[source.find_bit(item.qubits[0]).index, source.find_bit(item.clbits[0]).index]
                                       for item in source.data if item.operation.name == "measure"]
                            circuits[sha] = dict(sha256=sha, file=file, logical_qubits=source.num_qubits,
                                                 gate_counts=counts, depth=source.depth(), measurement_map=mapping)
                        elif path.read_bytes() != raw:
                            raise ValueError("QASM hash collision or source-file corruption")
                        group.update(source_qasm_sha256=sha, source_qasm_file=file)
                        groups.append(dict(index=len(groups), **common, group_index=group_index,
                                           group=group["group"], measurement_basis=group["group"][-1],
                                           parameters=call["parameters"], shots=group["shots"], counts=group["counts"],
                                           estimate=group["estimate"], sample_variance=group["sample_variance"],
                                           estimated_standard_error=group["estimated_standard_error"],
                                           outcome_min=group["outcome_min"], outcome_max=group["outcome_max"], rng=group["rng"],
                                           source_qasm_sha256=sha, source_qasm_file=file, logical_qubits=source.num_qubits,
                                           source_gate_counts=counts, source_depth=source.depth()))
                        if len(groups) % 1024 == 0:
                            print(json.dumps(dict(stage="source trace export", completed_groups=len(groups),
                                                  unique_qasm=len(circuits))), flush=True)
    planned = demo["recipe"]["planned"]
    if len(groups) != planned["group_batches"] or len(evaluations) != planned["objective_calls"]:
        raise ValueError("Source trace does not cover the complete actual sampled workload")
    write_csv(output / "groups.csv", groups)
    write_csv(output / "evaluations.csv", evaluations)
    manifest = dict(schema_version=1, kind="every actual ideal-sampled measured group and exact source circuit",
                    source_kind="unsynthesized source OpenQASM3; no routing, synthesis or physical projection",
                    deduplication="exact SHA-256 and byte equality only, after source reconstruction",
                    total_group_batches=len(groups), total_objective_calls=len(evaluations),
                    unique_circuit_count=len(circuits), unique_circuits=[circuits[key] for key in sorted(circuits)], groups=groups)
    write_json(output / "circuit-manifest.json", manifest)
    return manifest


def normalized_results(demo: dict, resources: dict) -> tuple[list[dict], list[dict]]:
    costs = {(row["case_id"], row["start_id"], row["seed"]): row for row in resources["per_run"]}
    rows = []
    for case in demo["cases"]:
        for run in case["runs"]:
            reference, validation = run["diagnostic"], run["validation"]
            bound = validation["quality_bound"]
            cost = costs[(case["case_id"], run["start_id"], run["seed"])]
            rows.append(dict(case_id=case["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                             p=case["p"], start_id=run["start_id"], seed=run["seed"], run_id=run["run_id"],
                             exact_quality=reference["exact_approximation_ratio"] if case["algorithm"] == "qaoa"
                             else reference["energy_density_gap"], target_met_exact=run["target_met_audit"],
                             holdout_estimate=bound["normalized_estimate"],
                             holdout_standard_error=bound["normalized_standard_error"],
                             radius_simultaneous=bound["simultaneous_radius"], radius_per_run=bound["per_run_95_radius"],
                             simultaneous_interval=bound["simultaneous_interval"],
                             target_status_simultaneous=bound["target_status"],
                             target_supported_simultaneous=bound["target_status"] == "supported",
                             precision_supported=bound["precision_supported"],
                             initial_parameters=run["initial"], final_parameters=run["parameters"],
                             accepted_proposals=run["accepted_proposals"], rejected_proposals=run["rejected_proposals"],
                             training_calls=run["objective_evaluations"], training_shots=run["total_shots"],
                             holdout_shots=validation["total_shots"], total_shots=run["total_shots_including_validation"],
                             group_batches=sum(len(call["groups"]) for call in run["evaluations"]) + len(validation["groups"]),
                             weighted_gate_counts=cost["weighted_gate_counts"],
                             source_unitary_gates=sum(number for name, number in cost["weighted_gate_counts"].items()
                                                       if name not in ("measure", "barrier", "reset")),
                             measured_qubit_readouts=case["n_qubits"] * run["total_shots_including_validation"],
                             optimizer_wall_seconds=run["optimizer_wall_seconds"],
                             objective_wall_seconds=run["objective_wall_seconds"],
                             classical_control_seconds=run["classical_control_seconds"],
                             independent_reference_wall_seconds=reference["reference_wall_seconds"],
                             hardware_feedback_seconds=None))
    case_by_id = {case["case_id"]: case for case in demo["cases"]}
    controls = []
    for run in demo["controls"]:
        case = case_by_id[run["case_id"]]
        reference = run["diagnostic"]
        controls.append(dict(case_id=case["case_id"], algorithm=case["algorithm"], n_qubits=case["n_qubits"],
                             p=case["p"], start_id=run["start_id"], mode=run["mode"], run_id=run["run_id"],
                             exact_quality=reference["exact_approximation_ratio"] if case["algorithm"] == "qaoa"
                             else reference["energy_density_gap"], target_met_exact=run["target_met_audit"],
                             initial_parameters=run["initial"], final_parameters=run["parameters"],
                             objective_evaluations=run["objective_evaluations"],
                             accepted_proposals=run["accepted_proposals"], rejected_proposals=run["rejected_proposals"],
                             actual_quantum_preparations=0, actual_measured_groups=0,
                             virtual_training_preparations=case["planned"]["training_shots_per_seed"],
                             optimizer_wall_seconds=run["optimizer_wall_seconds"],
                             objective_wall_seconds=run["objective_wall_seconds"],
                             classical_control_seconds=run["classical_control_seconds"],
                             independent_reference_wall_seconds=reference["reference_wall_seconds"]))
    return rows, controls


def _case_label(case: dict) -> str:
    return f"QAOA C{case['n_qubits']}, p={case['p']}" if case["algorithm"] == "qaoa" else f"VQE chain n={case['n_qubits']}"


def save_figure(fig, output: Path, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(output / filename, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_quality(output: Path, demo: dict, rows: list[dict], controls: list[dict]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.3))
    for ax, case in zip(axes.flat, demo["cases"]):
        for start_index, start in enumerate(("original", "alternate")):
            points = [r for r in rows if r["case_id"] == case["case_id"] and r["start_id"] == start]
            color = COLORS[start]
            ax.errorbar([start_index + offset for offset in (-0.09, 0, 0.09)],
                        [r["holdout_estimate"] for r in points],
                        yerr=[r["radius_simultaneous"] for r in points], fmt="none", ecolor=color, alpha=0.25)
            ax.scatter([start_index + offset for offset in (-0.09, 0, 0.09)],
                       [r["exact_quality"] for r in points], color=color, s=28,
                       label=f"{start}: sampled final state")
            for mode, marker, offset in (("exact-objective", "D", -0.17), ("expected-margin", "s", 0.17)):
                control = next(r for r in controls if r["case_id"] == case["case_id"]
                               and r["start_id"] == start and r["mode"] == mode)
                ax.scatter(start_index + offset, control["exact_quality"], facecolors="none", edgecolors=color,
                           marker=marker, s=44, label=f"{start}: {mode} CPU")
        target = case["target"]["ratio_at_least"] if case["algorithm"] == "qaoa" else case["target"]["density_gap_at_most"]
        ax.axhline(target, color="#777777", linestyle="--", linewidth=1)
        ax.set(xticks=[0, 1], xticklabels=["Original start", "Alternate start"], title=_case_label(case), xlim=(-0.4, 1.4))
        ax.set_ylabel("Exact mean cut / optimum" if case["algorithm"] == "qaoa" else "Exact energy-density gap")
        ax.grid(axis="y", alpha=0.15)
    axes.flat[0].legend(fontsize=6.8, loc="best")
    fig.suptitle("Fixed final-state quality: three sampled seeds and two deterministic CPU controls\n"
                 "Faint bars: simultaneous fresh-holdout radii; all target misses retained", fontsize=11)
    save_figure(fig, output, "quality.svg")


def draw_margins(output: Path, demo: dict, rows: list[dict], controls: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.3, 4.6), sharey=True)
    for ax, start in zip(axes, ("original", "alternate")):
        for index, case in enumerate(demo["cases"]):
            points = [r for r in rows if r["case_id"] == case["case_id"] and r["start_id"] == start]
            ax.scatter([index + offset for offset in (-0.08, 0, 0.08)],
                       [r["accepted_proposals"] for r in points], color=COLORS[start], s=26,
                       label="Sampled seeds" if index == 0 else None)
            for mode, marker, offset in (("exact-objective", "D", -0.18), ("expected-margin", "s", 0.18)):
                control = next(r for r in controls if r["case_id"] == case["case_id"]
                               and r["start_id"] == start and r["mode"] == mode)
                ax.scatter(index + offset, control["accepted_proposals"], marker=marker, facecolors="none",
                           edgecolors=COLORS[start], s=44, label=mode + " CPU" if index == 0 else None)
        ax.set(xticks=range(6), xticklabels=[_case_label(c).replace("QAOA ", "").replace("VQE chain ", "V ")
                                           for c in demo["cases"]], title=f"{start.capitalize()} initialization",
               ylabel="Accepted moves in 12 fixed sweeps", ylim=(-0.5, 12.5))
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.15)
        ax.legend(fontsize=8)
    fig.suptitle("Search trajectories under sampled, zero-margin exact, and expected-margin exact objectives", fontsize=11)
    save_figure(fig, output, "margins.svg")


def draw_workload(output: Path, demo: dict, rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    labels = [_case_label(c).replace("QAOA ", "").replace("VQE chain ", "V ") for c in demo["cases"]]
    for ax, field, title in ((axes[0], "training_shots", "Matched actual training preparations"),
                             (axes[1], "source_unitary_gates", "Complete shot-weighted source gates")):
        values = [[r[field] for r in rows if r["case_id"] == c["case_id"]] for c in demo["cases"]]
        means = [np.mean(v) for v in values]
        ax.bar(range(6), means, color=["#176e70"] * 4 + ["#bd6d36"] * 2, alpha=0.8)
        ax.set(xticks=range(6), xticklabels=labels, title=title, ylabel="Preparations" if field == "training_shots"
               else "Source unitary gates (training + holdouts)")
        ax.tick_params(axis="x", labelrotation=35)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Equal training preparations leave different circuits, readout and gate workloads", fontsize=11)
    save_figure(fig, output, "workload.svg")


def summary_markdown(demo: dict, rows: list[dict], controls: list[dict], resources: dict) -> str:
    lines = ["The frozen study retains all **36 sampled trajectories**: C4/C6 QAOA at p = 1, 2 and open-chain "
             "VQE at n = 4, 6, with two fixed starts and seeds 7, 19, 41. No fifth-increment winning angles "
             "or analytic witnesses initialize a run. Each sampled run has exactly **65,648 training preparations**, "
             "then 16,384 fresh shots per measurement group. All accepted and rejected candidates remain in the trace.", "",
             "The **24 deterministic CPU controls** use the same 12 case/start combinations and fixed polling schedule. "
             "Exact-objective controls set the comparison standard error to zero. Expected-margin controls use exact "
             "means and true population variances divided by the prescribed virtual group allocation. These controls "
             "contain no random counts and use **zero actual quantum preparations**; their allocations are prospective bookkeeping.", "",
             "| Case | Start | Sampled exact quality range (3 seeds) | Exact target hits | Holdout supports / misses / inconclusive | Exact-objective CPU quality | Expected-margin CPU quality |",
             "|---|---|---|---:|---|---:|---:|"]
    for case in demo["cases"]:
        for start in ("original", "alternate"):
            points = [r for r in rows if r["case_id"] == case["case_id"] and r["start_id"] == start]
            exact = next(r for r in controls if r["case_id"] == case["case_id"] and r["start_id"] == start and r["mode"] == "exact-objective")
            margin = next(r for r in controls if r["case_id"] == case["case_id"] and r["start_id"] == start and r["mode"] == "expected-margin")
            quality = [r["exact_quality"] for r in points]
            lines.append(f"| {_case_label(case)} | {start} | {min(quality):.6f}–{max(quality):.6f} | "
                         f"{sum(r['target_met_exact'] for r in points)}/3 | "
                         f"{sum(r['target_status_simultaneous'] == 'supported' for r in points)} / "
                         f"{sum(r['target_status_simultaneous'] == 'missed' for r in points)} / "
                         f"{sum(r['target_status_simultaneous'] == 'inconclusive' for r in points)} | "
                         f"{exact['exact_quality']:.6f} | {margin['exact_quality']:.6f} |")
    lines += ["", "QAOA quality means mean cut divided by the enumerated optimum (target ≥0.80). "
              "VQE quality means energy-density gap above the dense ground energy (target ≤0.05). "
              "Fresh holdout bounds allocate total failure 0.05 across all 48 final groups, conditional on fixed "
              "trained parameters. The accepted-move margin is a heuristic; these finite controls do not certify "
              "the global ansatz optimum, population optimizer-success probability or convergence.", "",
              f"Actual ideal-sampled workload: **{resources['objective_calls']:,} objective calls**, "
              f"**{resources['group_batches']:,} measured groups**, **{resources['total_preparations']:,} preparations**. "
              f"The deterministic controls record **{sum(r['objective_evaluations'] for r in controls):,} exact CPU oracle evaluations** "
              "with zero actual measured groups/preparations. Counts are simulated from ideal probabilities; no QPU or hardware noise was measured. "
              f"Actual source costs include **{resources['shot_weighted_source_unitary_gates']:,} shot-weighted unitary gates** "
              f"and **{resources['measured_qubit_readouts']:,} qubit readouts**.", "",
              "Input/state preparation gates, measurement-basis changes, every proposed circuit and all final readout shots "
              "are included in the actual source/readout ledger. Equal preparation budgets do not imply equal gate "
              "or physical cost. Rotations remain unsynthesized; physical initialization/reset, routing, communication "
              "and feedback time are unmeasured, and no physical projection or hardware advantage is inferred. "
              "All CPU timings, exact-QASM exports and producer snapshots are recorded separately. "
              "Existing Lean coverage remains classical C4 reference coverage only.", ""]
    return "\n".join(lines)


def overview_markdown(rows: list[dict], controls: list[dict], resources: dict) -> str:
    lines = ["All 36 fixed sampled trajectories and 24 deterministic CPU controls are retained. "
             "Every sampled trajectory uses 65,648 training preparations; exact controls use zero actual quantum preparations.", "",
             "| Method | Sampled runs | Post-run exact target hits | Whole-study holdout supports target | CPU control target hits (exact / expected-margin) | Actual preparations |",
             "|---|---:|---:|---:|---:|---:|"]
    for algorithm in ("qaoa", "vqe"):
        points = [r for r in rows if r["algorithm"] == algorithm]
        exact = [r for r in controls if r["algorithm"] == algorithm and r["mode"] == "exact-objective"]
        margin = [r for r in controls if r["algorithm"] == algorithm and r["mode"] == "expected-margin"]
        lines.append(f"| {algorithm.upper()} | {len(points)} | {sum(r['target_met_exact'] for r in points)} | "
                     f"{sum(r['target_supported_simultaneous'] for r in points)} | "
                     f"{sum(r['target_met_exact'] for r in exact)}/{len(exact)} / "
                     f"{sum(r['target_met_exact'] for r in margin)}/{len(margin)} | {sum(r['total_shots'] for r in points):,} |")
    lines += ["", f"Complete actual trace: {resources['objective_calls']:,} sampled calls, "
              f"{resources['group_batches']:,} measured groups and {resources['total_preparations']:,} ideal preparations. "
              "Matched preparations leave different gate workloads; these bounded observations do not establish "
              "general convergence or quantum advantage.", ""]
    return "\n".join(lines)


def check_numerical_record(demo: dict) -> dict:
    """Verify the recipe and replay each sampled choice from its recorded scores."""
    recipe = opt.diagnostic_recipe()
    if demo["recipe"] != recipe:
        raise ValueError("The complete published recipe changed")
    streams = set()
    for case in demo["cases"]:
        if len(case["runs"]) != 6:
            raise ValueError("Every diagnostic case requires both starts and all three seeds")
        for run in case["runs"]:
            start = opt.diagnostic_start(case, run["start_id"])
            point = [(float(x) + math.pi) % (2 * math.pi) - math.pi for x in run["initial"]]
            for iteration in run["iterations"]:
                if not np.allclose(iteration["parameters_before"], point, atol=1e-12, rtol=0):
                    raise ValueError("Recorded incumbent does not follow the preceding sampled move")
                incumbent = run["evaluations"][iteration["incumbent_evaluation"]]
                proposals = [run["evaluations"][index] for index in iteration["proposal_evaluations"]]
                best = min(proposals, key=lambda value: value["objective_estimate"])
                margin = recipe["comparison_z"] * math.hypot(incumbent["estimated_standard_error"],
                                                            best["estimated_standard_error"])
                accepted = incumbent["objective_estimate"] - best["objective_estimate"] > margin
                if (accepted != iteration["accepted"] or best["index"] != iteration["best_proposal_evaluation"]
                        or not math.isclose(margin, iteration["comparison_margin"], rel_tol=1e-12, abs_tol=1e-14)):
                    raise ValueError("Recorded sampled move does not replay")
                point = best["parameters"] if accepted else incumbent["parameters"]
                if not np.allclose(iteration["parameters_after"], point, atol=1e-12, rtol=0):
                    raise ValueError("Recorded post-move state differs from sampled choice")
            if not np.allclose(run["parameters"], point, atol=1e-12, rtol=0):
                raise ValueError("Final sampled state differs from its fixed trajectory")
            if run["validation"]["parameters"] != run["parameters"]:
                raise ValueError("Fresh final holdout is detached from fitted parameters")
            for phase, family, records in (("training", 0, run["evaluations"]),
                                             ("validation", 1, [run["validation"]])):
                for evaluation_index, call in enumerate(records):
                    for group_index, group in enumerate(call["groups"]):
                        stream = [run["seed"], start["stream_id"], family, evaluation_index, group_index]
                        if group["rng"]["stream_key"] != stream or tuple(stream) in streams:
                            raise ValueError("An actual sampled RNG stream differs or is reused")
                        streams.add(tuple(stream))
    # This validator reconstructs the actual circuits and prescribed allocations.
    return opt.diagnostic_resources(demo)


def freeze(output: Path) -> dict:
    """Save the recipe and captured executable bytes without any study call."""
    info = provenance()
    verify_environment(info)
    output.mkdir(parents=True, exist_ok=True)
    recipe = opt.diagnostic_recipe()
    preserve_json(output / "recipe.json", recipe)
    preserve_downloads(output)
    identity = checkpoint_identity(recipe, info)
    record = dict(schema_version=1, status="recipe_and_producers_frozen_before_execution", identity=identity,
                  recipe_file="recipe.json", implementation_file="optimization.py",
                  generator_file="generate_optimization_diagnostics.py", environment_lock_file="requirements-lock.txt",
                  captured_at_utc=info["generated_at_utc"],
                  source_commit_at_capture=info["source_commit"],
                  note="Captured byte hashes are authoritative; numerical execution records its own checkout revision")
    path = output / "freeze.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if existing["identity"] != identity or existing["status"] != record["status"]:
            raise ValueError("Existing frozen recipe/producers differ; choose a separate output directory")
        record = existing
    else:
        preserve_json(path, record)
    return dict(status=record["status"], execution_performed=False, identity=identity, planned=recipe["planned"])


class NumericalBudget:
    """Charge numerical execution and validation time cumulatively across resumes."""

    def __init__(self, output: Path, identity: dict, maximum: float):
        self.path, self.identity, self.maximum = output / "numerical-timing.json", identity, maximum
        self.started, self.cpu_started = perf_counter(), process_time()
        self.completed_run_seconds = 0.0
        self.complete = False
        self.record = json.loads(self.path.read_text()) if self.path.exists() else dict(
            schema_version=1, identity=identity, status="not_started", total_numerical_wall_seconds=0.0, invocations=[])
        if self.record["identity"] != identity:
            raise ValueError("Numerical timing identity differs from frozen producers")
        total = self.record["total_numerical_wall_seconds"]
        invocations = self.record["invocations"]
        if (not isinstance(total, (int, float)) or isinstance(total, bool) or not math.isfinite(total) or total < 0
                or not isinstance(invocations, list)):
            raise ValueError("Numerical timing totals or invocation history are invalid")
        for index, invocation in enumerate(invocations):
            if invocation.get("ordinal") != index + 1:
                raise ValueError("Numerical timing invocation order is invalid")
            for field in ("wall_seconds", "process_cpu_seconds", "completed_run_seconds", "overhead_and_uncompleted_run_seconds"):
                value = invocation.get(field)
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("Numerical timing invocation duration is invalid")
            if invocation["completed_run_seconds"] > invocation["wall_seconds"] + 1e-8:
                raise ValueError("Numerical timing invocation omits completed run duration")
        if not math.isclose(sum(i["wall_seconds"] for i in invocations), total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("Numerical timing total differs from cumulative invocation history")
        if self.record["status"] == "running":
            raise ValueError("Prior numerical invocation ended without a final timing record; elapsed effort is unknown and resume cannot reset the cap")
        self.previous_seconds = self.record["total_numerical_wall_seconds"]

    def __enter__(self):
        if self.previous_seconds >= self.maximum:
            raise RuntimeError("Cumulative numerical cap was exhausted; study remains incomplete")
        self.record["status"] = "running"
        write_json(self.path, self.record)
        return self

    def check(self) -> float:
        total = self.previous_seconds + perf_counter() - self.started
        if total > self.maximum:
            raise RuntimeError("Cumulative numerical wall cap exceeded; completed runs are preserved and study remains incomplete")
        return total

    def __exit__(self, error_type, error, traceback):
        wall = perf_counter() - self.started
        total = self.previous_seconds + wall
        status = "complete" if self.complete and error_type is None and total <= self.maximum else "incomplete"
        if total > self.maximum:
            status = "cap_exceeded_incomplete"
        self.record["invocations"].append(dict(ordinal=len(self.record["invocations"]) + 1,
            status=status, wall_seconds=wall, process_cpu_seconds=process_time() - self.cpu_started,
            completed_run_seconds=self.completed_run_seconds,
            overhead_and_uncompleted_run_seconds=max(0.0, wall - self.completed_run_seconds),
            error_type=error_type.__name__ if error_type else None))
        self.record.update(status=status, total_numerical_wall_seconds=total,
                           numerical_wall_cap_seconds=self.maximum,
                           timing_scope="cumulative CPU workflow wall time including checkpoints and numerical validation across resumes; exports separate")
        write_json(self.path, self.record)
        if error_type is None and status != "complete":
            raise RuntimeError("Cumulative numerical cap exceeded at final validation; study remains incomplete")
        return False


def generate(output: Path, *, resume: bool = False) -> dict:
    started, cpu_started = perf_counter(), process_time()
    if not (output / "freeze.json").is_file():
        raise ValueError("Run --freeze-only and commit/review its unchanged recipe/producers before execution")
    recipe = opt.diagnostic_recipe()
    if json.loads((output / "recipe.json").read_text()) != recipe:
        raise ValueError("Frozen recipe differs from the executed implementation")
    preserve_downloads(output)
    info = provenance()
    verify_environment(info)
    identity = checkpoint_identity(recipe, info)
    frozen = json.loads((output / "freeze.json").read_text())
    if frozen["identity"] != identity:
        raise ValueError("Execution/resume requires the exact frozen recipe and producer snapshots")
    info.update(freeze_sha256=digest(output / "freeze.json"), recipe_sha256=digest(output / "recipe.json"))
    raw_path = output / "raw-study.json"
    checkpoint_folder = output / "checkpoints"
    checkpoint_folder.mkdir(exist_ok=True)
    existing_checkpoints = list(checkpoint_folder.glob("*.json"))
    if existing_checkpoints and not (output / "numerical-timing.json").is_file():
        raise ValueError("Completed checkpoints require their cumulative numerical timing ledger; resume cannot reset the cap")
    if (raw_path.exists() or existing_checkpoints or (output / "runs.json").exists()) and not resume:
        raise ValueError("Numerical records already exist; use --resume to preserve completed runs")
    if raw_path.exists():
        demo = json.loads(raw_path.read_text())
        if (demo["recipe"] != recipe or checkpoint_identity(recipe, demo["provenance"]) != identity):
            raise ValueError("Immutable numerical aggregate differs from matching frozen producer identities")
        timing = NumericalBudget(output, identity, recipe["bounds"]["numerical_wall_seconds"]).record
        if (timing["identity"] != identity or timing["status"] != "complete"
                or timing["total_numerical_wall_seconds"] > recipe["bounds"]["numerical_wall_seconds"]
                or digest(output / "numerical-timing.json") != demo["numerical_timing_sha256"]):
            raise ValueError("Immutable numerical aggregate lacks a matching completed cumulative timing ledger")
        check_checkpoint_inventory(output, demo, identity)
        verified_resources = check_numerical_record(demo)
        info = demo["provenance"]
        print(json.dumps(dict(stage="resuming immutable numerical aggregate", sampled_runs=36,
                              controls=24, training_repeated=False)), flush=True)
    else:
        with NumericalBudget(output, identity, recipe["bounds"]["numerical_wall_seconds"]) as budget:
            cases, controls, expected_keys = [], [], set()
            for original in recipe["cases"]:
                case = copy.deepcopy(original)
                case["runs"] = []
                for start_id in ("original", "alternate"):
                    for seed in recipe["seeds"]:
                        key = f"{case['case_id']}-{start_id}-seed-{seed}"
                        path = checkpoint_folder / f"sampled-{key}.json"
                        expected_keys.add(path.name)
                        if path.exists():
                            run = load_checkpoint(path, identity, kind="sampled", key=key)
                            repeated = False
                        else:
                            budget.check()
                            run = opt.run_diagnostic_sampled(original, start_id, seed)
                            budget.completed_run_seconds += run["run_wall_seconds"]
                            save_checkpoint(path, identity, kind="sampled", key=key, run=run)
                            budget.check()
                            repeated = True
                        case["runs"].append(run)
                        print(json.dumps(dict(stage="sampled run checkpointed", run_id=key,
                                              executed_this_invocation=repeated, completed_sampled_runs=sum(len(c["runs"]) for c in cases) + len(case["runs"]),
                                              target_met=run["target_met_audit"])), flush=True)
                cases.append(case)
            for case in recipe["cases"]:
                for start_id in ("original", "alternate"):
                    for mode in ("exact-objective", "expected-margin"):
                        key = f"{case['case_id']}-{start_id}-{mode}"
                        path = checkpoint_folder / f"control-{key}.json"
                        expected_keys.add(path.name)
                        if path.exists():
                            run = load_checkpoint(path, identity, kind="control", key=key)
                            repeated = False
                        else:
                            budget.check()
                            run = opt.run_diagnostic_control(case, start_id, mode)
                            budget.completed_run_seconds += run["run_wall_seconds"]
                            save_checkpoint(path, identity, kind="control", key=key, run=run)
                            budget.check()
                            repeated = True
                        controls.append(run)
                        print(json.dumps(dict(stage="control run checkpointed", run_id=key,
                                              executed_this_invocation=repeated, completed_controls=len(controls),
                                              target_met=run["target_met_audit"])), flush=True)
            unexpected = {p.name for p in checkpoint_folder.glob("*.json")} - expected_keys
            if unexpected:
                raise ValueError("Unrecognized numerical checkpoints exist; preserve them in a separate output directory")
            demo = opt.diagnostic_aggregate(recipe, cases, controls)
            demo.update(provenance=info, execution_status="completed_bounded_ideal_sampling_and_exact_cpu_controls",
                        artifact_status="numerical_complete_trace_exports_pending", checkpoint_identity=identity,
                        numerical_invocation_wall_seconds=perf_counter() - started,
                        numerical_invocation_process_cpu_seconds=process_time() - cpu_started,
                        resumed_completed_run_count=len(existing_checkpoints),
                        timing_scope="local ideal CPU sampling, exact controls and classical audits; hardware timing unmeasured")
            verified_resources = check_numerical_record(demo)
            if digest(SOURCE) != identity["implementation_sha256"] or digest(SCRIPT) != identity["generator_sha256"]:
                raise RuntimeError("Producer changed during execution; individual captured-byte checkpoints are preserved")
            demo["numerical_workflow_wall_seconds"] = budget.check()
            demo["completed_run_wall_seconds"] = sum(r["run_wall_seconds"] for c in cases for r in c["runs"]) + sum(r["run_wall_seconds"] for r in controls)
            if demo["completed_run_wall_seconds"] > demo["numerical_workflow_wall_seconds"] + 1e-8:
                raise RuntimeError("Numerical timing ledger omits recorded run time")
            budget.complete = True
        timing = json.loads((output / "numerical-timing.json").read_text())
        demo["numerical_workflow_wall_seconds"] = timing["total_numerical_wall_seconds"]
        demo["numerical_timing_sha256"] = digest(output / "numerical-timing.json")
        preserve_json(raw_path, demo)
        print(json.dumps(dict(stage="immutable numerical aggregate saved", sampled_runs=36, controls=24,
                              training_repeated_on_resume=False)), flush=True)
    numerical_digest = digest(raw_path)
    resources = verified_resources
    resources.update(provenance=info, numerical_input_sha256=numerical_digest)
    exported = copy.deepcopy(demo)
    manifest = export_group_traces(output, exported)
    rows, controls = normalized_results(exported, resources)
    write_csv(output / "results.csv", rows)
    write_csv(output / "controls.csv", controls)
    write_csv(output / "resources.csv", resources["per_run"])
    control_evaluations = [dict(case_id=run["case_id"], start_id=run["start_id"], mode=run["mode"], run_id=run["run_id"],
                                **evaluation) for run in demo["controls"] for evaluation in run["evaluations"]]
    write_csv(output / "control-evaluations.csv", control_evaluations)
    draw_quality(output, exported, rows, controls)
    draw_margins(output, exported, rows, controls)
    draw_workload(output, exported, rows)
    (output / "summary.md").write_text(summary_markdown(exported, rows, controls, resources), encoding="utf-8")
    (output / "overview.md").write_text(overview_markdown(rows, controls, resources), encoding="utf-8")
    exported.update(artifact_status="complete", numerical_input_sha256=numerical_digest,
                    artifact_generation_wall_seconds=perf_counter() - started,
                    artifact_timing_scope="this invocation includes resumed checks, trace reconstruction, CSV export and plotting")
    resources.update(artifact_status="complete", circuit_manifest=dict(file="circuit-manifest.json",
                     sha256=digest(output / "circuit-manifest.json"), total_groups=manifest["total_group_batches"],
                     unique_circuit_count=manifest["unique_circuit_count"]))
    if digest(SOURCE) != identity["implementation_sha256"] or digest(SCRIPT) != identity["generator_sha256"]:
        raise RuntimeError("Producer changed during export; immutable numerical checkpoints are preserved")
    write_json(output / "runs.json", exported)
    write_json(output / "resources.json", resources)
    return dict(status=exported["execution_status"], artifact_status="complete", sampled_runs=len(rows), controls=len(controls),
                objective_calls=resources["objective_calls"], measured_groups=resources["group_batches"],
                preparations=resources["total_preparations"], unique_source_circuits=manifest["unique_circuit_count"],
                qaoa_target_hits=sum(r["target_met_exact"] for r in rows if r["algorithm"] == "qaoa"),
                vqe_target_hits=sum(r["target_met_exact"] for r in rows if r["algorithm"] == "vqe"),
                numerical_input_sha256=numerical_digest, generation_wall_seconds=perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / ("assets/generated/optimization/diagnostics" if CHECKOUT else "diagnostics-artifacts"))
    parser.add_argument("--freeze-only", action="store_true", help="preserve the fixed recipe and producer bytes; execute no study")
    parser.add_argument("--resume", action="store_true", help="reuse matching completed checkpoints and finish unfinished runs/exports")
    args = parser.parse_args()
    if args.freeze_only and args.resume:
        parser.error("--freeze-only and --resume select different workflows")
    result = freeze(args.output_dir) if args.freeze_only else generate(args.output_dir, resume=args.resume)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
