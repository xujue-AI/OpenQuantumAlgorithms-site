#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Conditional surface-code projections for three fixed sampled readouts.

This estimates the first predeclared finite-shot case's final QAOA all-Z and
VQE all-Z/all-X validation circuits. It does not estimate the adaptive training
trajectory or certify angle synthesis, observable accuracy, or convergence.
The shared resource adapter must be present in the source checkout.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openquantum.optimization import CYCLE_EDGES, build_qaoa_circuit, build_vqe_circuit
from openquantum.resource_estimation import (
    CircuitResourceBudget,
    SurfaceCodeProfile,
    estimate_circuit_resources,
)

EXPECTED_SHOTS = 4096
SYNTHESIS_MODEL_ALLOWANCE = 1e-5
TASK_PHYSICAL_FAILURE_ALLOWANCE = 1e-3
PROFILES = (
    SurfaceCodeProfile(name="frozen-p-1e-3", physical_error_rate=1e-3,
                       gate_time_ns=100, measurement_time_ns=500),
    SurfaceCodeProfile(name="frozen-p-1e-4", physical_error_rate=1e-4,
                       gate_time_ns=100, measurement_time_ns=500),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance(runs_path: Path) -> dict:
    paths = {
        "optimization_source": ROOT / "src/openquantum/optimization.py",
        "resource_adapter_source": ROOT / "src/openquantum/resource_estimation.py",
        "projection_generator": Path(__file__).resolve(),
        "environment_lock": ROOT / "requirements-lock.txt",
        "sampled_runs": runs_path,
    }
    commit = git_revision(ROOT)
    return {
        "source_commit_at_generation": commit,
        "source_content_sha256": {name: sha256(path) for name, path in paths.items()},
        "revision_note": "Commit identifies the checkout at generation; content hashes identify actual executed/input bytes.",
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "qdk", "numpy")},
        "backend": "local QDK qre conditional model; CPU; no QPU execution",
        "architecture": platform.machine(),
    }


def select_fixed_readout(data: dict) -> dict:
    """Cross-check declared selection against actual primary held-out records."""
    # The source generator writes this explicit selection contract alongside
    # the trajectories. It contains no parameter selected by exact diagnostics.
    selection = data["projection_fixed_readout_selection"]
    if selection["selection_rule"] != "first_predeclared_case":
        raise ValueError("Projection selection must be the first predeclared case")
    if selection["master_seed"] != 7:
        raise ValueError("Expected the first predeclared master seed 7")
    if selection["shots_per_group"] != EXPECTED_SHOTS:
        raise ValueError("Fixed readout projection requires exactly 4096 shots per group")
    gamma, beta = selection["qaoa_parameters"]
    theta, = selection["vqe_parameters"]
    if not all(math.isfinite(float(x)) for x in (gamma, beta, theta)):
        raise ValueError("All selected angles must be finite")
    if data["exact_diagnostics_used_for_optimization"] is not False:
        raise ValueError("These projections require the recorded genuinely sampled optimizer")
    if data["provenance"]["implementation_sha256"] != sha256(ROOT / "src/openquantum/optimization.py"):
        raise ValueError("Regenerate sampled runs from the current implementation before projection")
    qaoa = data["qaoa"]
    vqe = data["vqe"]
    if qaoa["n_qubits"] != 4 or sorted(tuple(sorted(e)) for e in qaoa["edges"]) != sorted(CYCLE_EDGES):
        raise ValueError("Fixed QAOA projection requires the stated four-vertex cycle")
    if vqe["n_qubits"] != 2 or vqe["hamiltonian"] != "-Z0-Z1+0.5 X0X1":
        raise ValueError("Fixed VQE projection requires the stated two-qubit Hamiltonian")
    normalized = {
        "selection_rule": selection["selection_rule"], "master_seed": 7,
        "shots_per_group": EXPECTED_SHOTS,
        "qaoa_parameters": [float(gamma), float(beta)], "vqe_parameters": [float(theta)],
    }
    variants = fixed_variants(normalized)
    for algorithm, parameters, algorithm_id, group_names in (
            ("qaoa", normalized["qaoa_parameters"], 0, ["all_z"]),
            ("vqe", normalized["vqe_parameters"], 1, ["all_z", "all_x"])):
        run = data[algorithm]["runs"][0]
        validation = run["validation"]
        if run["run_index"] != 0 or run["seed"] != 7 or run["parameters"] != parameters:
            raise ValueError("Detached selection does not match the first predeclared trajectory")
        if (validation["role"] != "independent_final_validation" or validation["parameters"] != parameters
                or validation["total_shots"] != EXPECTED_SHOTS * len(group_names)):
            raise ValueError("Selected final validation parameters or shot allocation do not match")
        if [g["group"] for g in validation["groups"]] != group_names:
            raise ValueError("Selected validation measurement groups do not match")
        train_streams = {tuple(g["rng"]["stream_key"]) for call in run["evaluations"] for g in call["groups"]}
        for group_index, group in enumerate(validation["groups"]):
            expected_stream = [7, algorithm_id, 1, 0, group_index]
            if (group["shots"] != EXPECTED_SHOTS or sum(group["counts"].values()) != EXPECTED_SHOTS
                    or group["rng"]["entropy"] != [7, algorithm_id]
                    or group["rng"]["spawn_key"] != [1, 0, group_index]
                    or group["rng"]["stream_key"] != expected_stream
                    or tuple(expected_stream) in train_streams):
                raise ValueError("Final validation does not use the recorded independent held-out streams/shots")
            variant = next(v for v in variants if v["task"] == algorithm and v["measurement_group"] == group["group"])
            if (group["logical_qubits"] != variant["circuit"].num_qubits
                    or group["circuit_gate_counts"] != dict(variant["circuit"].count_ops())
                    or [group["outcome_min"], group["outcome_max"]] != variant["outcome_interval"]):
                raise ValueError("Recorded validation circuit/group does not match the bound projection circuit")
        normalized[algorithm + "_validation"] = {
            "recorded_sampled_minimized_objective": validation["objective_estimate"],
            "recorded_estimated_standard_error": validation["estimated_standard_error"],
            "exact_objective_after_training_audit": validation["exact_objective_audit"],
            "recorded_ideal_fixed_mean_95_radius": validation["hoeffding_95_absolute_mean_radius"],
            "group_rng_streams": [g["rng"] for g in validation["groups"]],
        }
    return normalized


def fixed_variants(selection: dict) -> list[dict]:
    gamma, beta = selection["qaoa_parameters"]
    theta, = selection["vqe_parameters"]
    return [
        {
            "id": "qaoa-all-z", "task": "qaoa", "measurement_group": "all_z",
            "parameters": [gamma, beta], "outcome": "joint MaxCut score C", "outcome_interval": [0, 4],
            "physical_failure_all_executions_allowance": TASK_PHYSICAL_FAILURE_ALLOWANCE,
            "circuit": build_qaoa_circuit(4, CYCLE_EDGES, [gamma], [beta], measure=True),
        },
        {
            "id": "vqe-all-z", "task": "vqe", "measurement_group": "all_z",
            "parameters": [theta], "outcome": "joint -Z0-Z1", "outcome_interval": [-2, 2],
            "physical_failure_all_executions_allowance": TASK_PHYSICAL_FAILURE_ALLOWANCE / 2,
            "circuit": build_vqe_circuit(theta, basis="z", measure=True),
        },
        {
            "id": "vqe-all-x", "task": "vqe", "measurement_group": "all_x",
            "parameters": [theta], "outcome": "0.5 X0X1", "outcome_interval": [-0.5, 0.5],
            "physical_failure_all_executions_allowance": TASK_PHYSICAL_FAILURE_ALLOWANCE / 2,
            "circuit": build_vqe_circuit(theta, basis="x", measure=True),
        },
    ]


def ideal_sampling_contract(shots: int = EXPECTED_SHOTS) -> dict:
    # These are separate ideal fixed-state statistical statements; they are
    # not made into certified biased confidence statements by the QDK fit.
    return {
        "scope": "independent held-out readout of the recorded fixed state, conditional on training selection",
        "confidence_per_estimate": 0.95,
        "qaoa": {"shots": shots, "outcome_interval": [0, 4],
                 "ideal_hoeffding_absolute_mean_radius": 4 * math.sqrt(math.log(40) / (2 * shots))},
        "vqe": {"shots_per_group": shots, "total_shots": 2 * shots,
                "groups": [{"id": "all_z", "outcome_interval": [-2, 2], "joint_Z_covariance_retained": True},
                           {"id": "all_x", "outcome_interval": [-0.5, 0.5]}],
                "ideal_combined_hoeffding_absolute_energy_radius":
                    math.sqrt((16 / shots + 1 / shots) * math.log(40) / 2)},
        "no_silent_one_mean_helper": True,
        "certified_sampling_synthesis_physical_confidence": None,
        "limitations": [
            "The ideal radius bounds a fixed state's sampled mean, not distance from an optimum or training convergence.",
            "Coverage is per estimate, not simultaneous across the entire adaptive trajectory.",
            "The synthesis mean fit supplies no per-angle output-distribution or channel approximation certificate.",
        ],
    }


def aggregate_readout_choices(reports: list[dict]) -> list[dict]:
    """Show two declared serial choices; this is not a full combined Pareto search."""
    result = []
    for profile in PROFILES:
        for task in ("qaoa", "vqe"):
            group_reports = [r for r in reports if r["profile"]["name"] == profile.name and r["task"] == task]
            for label, key in (("fewest qubits per group", "peak_physical_qubits"),
                               ("shortest runtime per group", "serial_runtime_all_executions_ns")):
                selected = [min(r["report"]["pareto_points"], key=lambda p: (p[key], p["serial_runtime_all_executions_ns"],
                                                                           p["peak_physical_qubits"]))
                            for r in group_reports]
                fault = sum(p["physical_failure_union_bound_all_executions"] for p in selected)
                if fault > TASK_PHYSICAL_FAILURE_ALLOWANCE + 1e-12:
                    raise AssertionError("Combined fixed-readout fault score exceeds the task allocation")
                result.append({
                    "profile": profile.name, "physical_error_rate": profile.physical_error_rate,
                    "task": task, "choice": label,
                    "group_ids": [r["variant_id"] for r in group_reports],
                    "group_selected_points": selected,
                    "modeled_executions": EXPECTED_SHOTS * len(group_reports),
                    "serial_runtime_all_groups_ns": sum(p["serial_runtime_all_executions_ns"] for p in selected),
                    "peak_physical_qubits_for_serial_groups": max(p["peak_physical_qubits"] for p in selected),
                    "physical_failure_union_bound_all_groups": fault,
                    "physical_failure_allowance_all_groups": TASK_PHYSICAL_FAILURE_ALLOWANCE,
                    "not_a_combined_pareto_optimization": True,
                    "serial_layout_assumption": "independent group layouts/factories; reconfiguration or reuse cost omitted",
                    "layout_change_and_communication_overhead_ns": None,
                    "common_hardware_layout_verified": False,
                })
    return result


def markdown_summary(record: dict) -> str:
    lines = [
        "These are **conditional mean-fit resource projections for the first predeclared case's fixed final readout**. "
        "They exclude all finite-shot optimizer training calls and are not QPU measurements, certified angle syntheses, "
        "worst-case bounds, convergence guarantees, or guaranteed observable accuracy.", "",
        "The profiles use the frozen physical assumptions: error rates $10^{-3}$ and $10^{-4}$, "
        "100 ns gate time and 500 ns measurement time. Each circuit has 4,096 modeled serial executions. "
        "The QAOA task receives physical-fault allowance $10^{-3}$; the VQE task receives $10^{-3}$ split equally "
        "between its Z and X groups. Each circuit separately receives synthesis-model scalar allowance $10^{-5}$ per execution.", "",
        "| Task | Physical error | Declared group selection | Modeled preparations | Peak physical qubits | Serial readout time (ms) | Physical-fault score, all groups |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for point in record["declared_serial_readout_choices"]:
        lines.append(f"| {point['task'].upper()} | {point['physical_error_rate']:g} | {point['choice']} | "
                     f"{point['modeled_executions']} | {point['peak_physical_qubits_for_serial_groups']:,} | "
                     f"{point['serial_runtime_all_groups_ns']/1e6:.6f} | {point['physical_failure_union_bound_all_groups']:.3g} |")
    sampling = record["ideal_sampling_contract"]
    lines += ["", "These two declared choices select each group's point separately; they do not claim a full combined Pareto frontier. "
              "Serial group time is summed and peak physical qubits use the maximum, rather than adding independent peaks. "
              "This assumes reuse or reconfiguration between independent group layouts/factories; configuration-change and communication "
              "overhead are excluded, and a common joint hardware layout is not certified.", "",
              f"The separate ideal fixed-state 95% Hoeffding radii are {sampling['qaoa']['ideal_hoeffding_absolute_mean_radius']:.6f} "
              f"for the mean cut and {sampling['vqe']['ideal_combined_hoeffding_absolute_energy_radius']:.6f} for energy. "
              "The VQE bound uses two independent bounded groups and preserves Z-term covariance. "
              "No certified confidence combining sampling, synthesis and physical faults is reported.", "",
              "The full JSON records all six circuit/profile reports, every returned per-variant Pareto point, "
              "bound circuit/QASM hashes, source hashes, raw/synthesized QDK traces, model budgets and limitations. "
              "The full adaptive optimizer physical layer remains **pending**.", ""]
    return "\n".join(lines)


def generate(runs_path: Path, output: Path, cache_dir: Path) -> dict:
    data = json.loads(runs_path.read_text())
    selection = select_fixed_readout(data)
    variants = fixed_variants(selection)
    reports = []
    begin = perf_counter()
    for profile in PROFILES:
        for variant in variants:
            started = perf_counter()
            budget = CircuitResourceBudget(
                synthesis_model_error_per_execution=SYNTHESIS_MODEL_ALLOWANCE,
                physical_failure_probability=variant["physical_failure_all_executions_allowance"],
                executions=EXPECTED_SHOTS,
            )
            report = estimate_circuit_resources(variant["circuit"], budget=budget,
                                                profile=profile, cache_dir=cache_dir,
                                                use_graph=False)
            if report["resource_estimate_kind"] != "conditional_mean_fit_resource_projection":
                raise AssertionError("Adapter returned an unaudited report kind")
            if report["synthesis"]["actual_synthesis_sequence_generated"] or report["synthesis"]["instance_approximation_certificate_verified"]:
                raise AssertionError("Do not promote the model to a certified synthesis")
            source_qasm = report["qasm3"]
            filename = f"{variant['id']}-{profile.name}.qasm"
            (output / "projections" / filename).write_text(source_qasm, encoding="utf-8")
            if sha256(output / "projections" / filename) != report["qasm_sha256"]:
                raise AssertionError("Projection QASM artifact differs from the estimated circuit")
            reports.append({
                "variant_id": variant["id"], "task": variant["task"],
                "measurement_group": variant["measurement_group"], "parameters": variant["parameters"],
                "outcome": variant["outcome"], "outcome_interval": variant["outcome_interval"],
                "profile": asdict(profile), "modeled_executions": EXPECTED_SHOTS,
                "qasm_artifact": f"projections/{filename}", "projection_wall_seconds": perf_counter() - started,
                "report": report,
            })
    record = {
        "schema_version": 1, "status": "computed_conditional_model",
        "resource_estimate_kind": "conditional_mean_fit_resource_projection",
        "scope": "seed7 final independent readout only; adaptive finite-shot training excluded",
        "fixed_readout_selection": selection,
        "source_provenance": provenance(runs_path),
        "ideal_sampling_contract": ideal_sampling_contract(),
        "reports": reports,
        "declared_serial_readout_choices": aggregate_readout_choices(reports),
        "full_finite_shot_optimizer_physical_projection": None,
        "full_trajectory_status": "pending explicit changing-parameter and whole-run physical/synthesis allocation",
        "projection_wall_seconds": perf_counter() - begin,
        "independent_task_comparison": "Do not compare these different tasks with Grover as an algorithm advantage claim.",
        "excluded": ["all optimizer training circuits/shots/classical control", "hardware state reset time", "device routing",
                     "external data loading", "noise/mitigation experiments", "certified angle synthesis and runtime tails",
                     "group configuration-change and communication overhead; no common joint hardware layout verified"],
    }
    write_json(output / "sampled-projections.json", record)
    (output / "sampled-projections.md").write_text(markdown_summary(record), encoding="utf-8")
    columns = ["profile", "physical_error_rate", "task", "choice", "modeled_executions",
               "serial_runtime_all_groups_ns", "peak_physical_qubits_for_serial_groups",
               "physical_failure_union_bound_all_groups"]
    with (output / "sampled-projections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(record["declared_serial_readout_choices"])
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "assets/generated/optimization/sampled-runs.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/optimization")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qre-optimization-fixed-readout")
    args = parser.parse_args()
    if not args.runs.is_file():
        raise SystemExit("Generate the finite-shot trajectory records before fixed-readout projections")
    (args.output_dir / "projections").mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    record = generate(args.runs, args.output_dir, args.cache_dir)
    print(json.dumps({"status": record["status"], "fixed_variants": 3,
                      "circuit_profile_reports": len(record["reports"]),
                      "full_optimizer_physical_projection": None,
                      "projection_wall_seconds": record["projection_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
