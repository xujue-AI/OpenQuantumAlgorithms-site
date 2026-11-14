#!/usr/bin/env python3
"""Generate bounded branch-adaptive amplitude-estimation evidence and resources."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

from openquantum.iterative_amplitude_estimation import (
    AdaptiveAEConfig, adaptive_logical_ledger, build_amplified_trial,
    run_adaptive_amplitude, run_threshold_adaptive,
)
from openquantum.qft_qpe import lower_circuit
from openquantum.resource_estimation import (
    CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources,
)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def provenance(seed: int) -> dict:
    versions = {}
    for name in ("qiskit", "numpy", "scipy", "matplotlib", "qdk"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    base_commit = git_revision(ROOT)
    source_names = ("iterative_amplitude_estimation.py", "amplitude_estimation.py", "qft_qpe.py", "resource_estimation.py")
    return {
        "python": platform.python_version(), "packages": versions,
        "device": "CPU", "system": platform.system(), "machine": platform.machine(),
        "seed": seed, "source_base_commit": base_commit,
        "source_base_commit_is_full_current_source_commit": False,
        "source_sha256": {name: file_hash(ROOT / "src/openquantum" / name) for name in source_names},
        "generator_sha256": file_hash(Path(__file__)),
        "requirements_lock_sha256": file_hash(ROOT / "requirements-lock.txt"),
        "hardware_execution": False,
    }


def recorded_power_shots(result: dict) -> Counter:
    totals = Counter()
    for step in result["history"]:
        power, shots = step["power"], step["shots"]
        if isinstance(power, bool) or not isinstance(power, int) or power < 0:
            raise ValueError("Trajectory contains an invalid recorded amplification power")
        if isinstance(shots, bool) or not isinstance(shots, int) or shots <= 0:
            raise ValueError("Every sampled trajectory round must have a positive actual shot count")
        totals[power] += shots
    if sum(totals.values()) != result["total_shots"]:
        raise ValueError("Trajectory total does not equal every recorded round's shots")
    return totals


def add_rng_hashes(result: dict) -> dict:
    for step in result["history"]:
        for name in ("rng_before", "rng_after"):
            text = json.dumps(step[name], sort_keys=True, separators=(",", ":"))
            step[f"{name}_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return result


def interval_pair(interval: dict | None) -> tuple[float, float]:
    if interval is None:
        return float("nan"), float("nan")
    return interval["lower"], interval["upper"]


def validation_suite(config: AdaptiveAEConfig, seed: int) -> dict:
    # These predetermined thresholds probe endpoints and neighboring alias
    # regions. All providers construct/simulate actual bounded trial circuits.
    cases = [(2, 0), (2, 4), (2, 1), (2, 2), (3, 3), (3, 5),
             (5, 1), (5, 7), (5, 9), (5, 31)]
    records = []
    for index, (n_data, threshold) in enumerate(cases):
        case_seed = (seed + index) % 2**64
        result = add_rng_hashes(run_threshold_adaptive(n_data, threshold, config=config,
                                                      seed=case_seed, record_shots=True))
        check_errors = [abs(step["sampling_probability"] - step["analytic_probability_check"])
                        for step in result["history"]]
        final = result["final_amplitude_interval"]
        truth = threshold / 2**n_data
        records.append({
            "input": {"n_data": n_data, "threshold": threshold, "exact_classical_answer": truth},
            "trajectory": result,
            "actual_bounded_circuit_statevector_validation": bool(result["history"]),
            "maximum_probability_check_error": max(check_errors, default=0.0),
            "final_interval_contains_reference": final is not None and final["lower"] <= truth <= final["upper"],
            "conclusive_reference_error": abs(final["estimate"] - truth) if result["status"] == "conclusive" else None,
            "hardware_execution": False,
        })
    return {
        "status": "executed predetermined bounded threshold validation suite",
        "case_selection": "fixed before outcomes; endpoints, quarter/half aliases, and neighboring threshold values",
        "cases": records,
        "cases_with_any_recorded_trials": sum(bool(row["trajectory"]["history"]) for row in records),
        "total_recorded_ideal_shots": sum(row["trajectory"]["total_shots"] for row in records),
        "conclusive_cases": sum(row["trajectory"]["status"] == "conclusive" for row in records),
        "confidence_claim": "individual numerical examples and exact arithmetic-bound checks; no hardware or finite-seed probability certificate",
    }


def model_coverage(config: AdaptiveAEConfig, seed: int) -> dict:
    amplitudes = [0.0, .001, .25, .5, .75, .999, 1.0]
    runs_per_amplitude = 32
    groups = []
    for index, amplitude in enumerate(amplitudes):
        runs = []
        for offset in range(runs_per_amplitude):
            run_seed = (seed + 10000 + index * runs_per_amplitude + offset) % 2**64
            result = add_rng_hashes(run_adaptive_amplitude(amplitude, config=config,
                                                         seed=run_seed, record_shots=False))
            final = result["final_amplitude_interval"]
            covers = final is not None and final["lower"] <= amplitude <= final["upper"]
            conclusive_wrong = result["status"] == "conclusive" and abs(final["estimate"] - amplitude) > config.absolute_error
            any_interval_miss = any(step["amplitude_interval"] is None or
                                    not step["amplitude_interval"]["lower"] <= amplitude <= step["amplitude_interval"]["upper"]
                                    for step in result["history"])
            runs.append({"trajectory": result, "final_interval_contains_reference": covers,
                         "any_recorded_interval_miss": any_interval_miss,
                         "conclusive_and_error_exceeds_target": conclusive_wrong})
        groups.append({
            "amplitude": amplitude, "runs": runs,
            "number_of_runs": runs_per_amplitude,
            "conclusive_runs": sum(row["trajectory"]["status"] == "conclusive" for row in runs),
            "conclusive_and_wrong_runs": sum(row["conclusive_and_error_exceeds_target"] for row in runs),
            "runs_with_any_interval_miss": sum(row["any_recorded_interval_miss"] for row in runs),
            "final_interval_miss_runs": sum(not row["final_interval_contains_reference"] for row in runs),
            "total_modeled_shots": sum(row["trajectory"]["total_shots"] for row in runs),
        })
    return {
        "status": "executed bounded analytic-law-only categorical coverage diagnostic",
        "probability_source": "numerical sin² amplified law through the same inference engine; no circuit constructed or simulated",
        "amplitudes": groups, "runs_per_amplitude": runs_per_amplitude,
        "total_runs": len(amplitudes) * runs_per_amplitude,
        "worst_case_modeled_shot_cap": len(amplitudes) * runs_per_amplitude * min(config.max_total_shots, config.shots_per_round * config.max_rounds),
        "empirical_frequency_is_confidence_certificate": False,
        "not_claimed": "accuracy conditional on conclusive output, exact PRNG law, physical confidence, or circuit validation of model-only amplitudes",
    }


def export_power_circuits(output: Path, n_data: int, threshold: int, result: dict) -> list[dict]:
    rows = []
    powers = recorded_power_shots(result)
    for power, shots in sorted(powers.items()):
        circuit = build_amplified_trial(n_data, threshold, power, measure=True)
        compiled = lower_circuit(circuit)
        if len(compiled.data) > 250_000:
            raise ValueError("Recorded trial exceeds the 250,000 primitive export cap")
        text = qasm3.dumps(compiled)
        filename = f"power-{power}.qasm"
        (output / filename).write_text(text)
        rows.append({
            "power": power, "multiplier": 2 * power + 1, "recorded_shots": shots,
            "logical_qubits": circuit.num_qubits,
            "filename": filename, "qasm_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "source_gate_counts": dict(circuit.count_ops()),
            "decomposed_gate_counts": dict(compiled.count_ops()),
            "decomposed_dependency_depth": compiled.depth(),
            "decomposed_operations_including_flag_measurement": len(compiled.data),
            "all_shot_primitive_operations": len(compiled.data) * shots,
            "qasm_phase_scope": "fixed-power measured trial; omitted overall scalar is irrelevant to flag probabilities, not a phase-preserving controlled oracle",
        })
    return rows


def serial_envelope_frontier(groups: list[dict]) -> list[dict]:
    """Exact qubit/time envelope for serial groups with free reconfiguration.

    For a candidate peak bound, every group selects its quickest allowed point.
    The only breakpoints are the finite returned peak-qubit values. This avoids
    an exponential product of frontiers and does not certify another hardware
    allocation, parallel schedule or factory reconfiguration latency.
    """
    thresholds = sorted({point["peak_physical_qubits"] for group in groups
                         for point in group["conditional_projection"]["pareto_points"]})
    candidates = []
    for threshold in thresholds:
        chosen = []
        for group in groups:
            allowed = [point for point in group["conditional_projection"]["pareto_points"]
                       if point["peak_physical_qubits"] <= threshold]
            if not allowed:
                break
            chosen.append(min(allowed, key=lambda point: (point["serial_runtime_all_executions_ns"],
                                                          point["physical_failure_union_bound_all_executions"])))
        if len(chosen) != len(groups):
            continue
        candidate = {
            "peak_physical_qubits": max(point["peak_physical_qubits"] for point in chosen),
            "serial_recorded_path_runtime_ns": sum(point["serial_runtime_all_executions_ns"] for point in chosen),
            "physical_fault_union_bound_recorded_path": sum(point["physical_failure_union_bound_all_executions"] for point in chosen),
            "selected_groups": [{"power": group["power"], "shots": group["shots"], "point": point}
                                for group, point in zip(groups, chosen)],
        }
        if not candidates or candidate["serial_recorded_path_runtime_ns"] < candidates[-1]["serial_recorded_path_runtime_ns"]:
            candidates.append(candidate)
    return candidates


def project_recorded_path(n_data: int, threshold: int, result: dict, source_info: dict, exports: list[dict]) -> dict:
    adaptive_logical_ledger(n_data, threshold, result)  # Verify task mapping and all weighted totals.
    powers = recorded_power_shots(result)
    total = sum(powers.values())
    if not total:
        return {"status": "unavailable_no_recorded_trials", "hardware_confidence_certified": False}
    groups = []
    export_hashes = {item["power"]: item["qasm_sha256"] for item in exports}
    for power, shots in sorted(powers.items()):
        circuit = build_amplified_trial(n_data, threshold, power, measure=True)
        # Every circuit execution gets the same physical share: .005/total.
        # The approximation-model scalar is per output, not a random failure.
        budget = CircuitResourceBudget(synthesis_model_error_per_execution=1e-5,
                                       physical_failure_probability=.005 * shots / total,
                                       executions=shots)
        projection = estimate_circuit_resources(
            circuit, budget=budget, profile=SurfaceCodeProfile(),
            cache_dir=Path(tempfile.gettempdir()) / "openquantum-amplitude-adaptive-qdk-cache",
        )
        if projection["qasm_sha256"] != export_hashes[power]:
            raise ValueError("The projected full measured circuit differs from its exported power circuit")
        groups.append({"power": power, "shots": shots, "multiplier": 2 * power + 1,
                       "exported_qasm_sha256_verified": export_hashes[power],
                       "source_sha256": source_info["source_sha256"],
                       "conditional_projection": projection})
    frontier = serial_envelope_frontier(groups)
    if any(point["physical_fault_union_bound_recorded_path"] > .005 + 1e-12 for point in frontier):
        raise ValueError("Recorded-path physical allowance was exceeded")
    return {
        "status": "computed_conditional_recorded_path_resource_projection",
        "whole_recorded_path_physical_failure_allowance": .005,
        "physical_failure_allowance_per_trial": .005 / total,
        "per_output_synthesis_model_allowance": 1e-5,
        "recorded_total_shots": total,
        "recorded_adaptive_status": result["status"],
        "recorded_round_order": [{"round": step["round"], "power": step["power"], "shots": step["shots"]}
                                 for step in result["history"]],
        "grouping_is_accounting_only": True,
        "hardware_confidence_certified": False,
        "groups": groups, "serial_envelope_pareto_points": frontier,
        "scope": "all actual shots in recorded adaptive round order; grouped counts are accounting only; future noisy paths may differ",
        "envelope_assumptions": ["all trials and power groups execute serially",
                                 "peak qubits are the maximum over selected per-group allocations",
                                 "factory/layout reconfiguration between groups is permitted with unmeasured latency",
                                 "classical control, initialization/reset, decoding and compilation timing are excluded"],
        "precision_warning": "QDK mean-fit synthesis cost has no concrete certificate; ideal binomial confidence and adaptive branch decisions are not certified for hardware",
        "source_and_environment": source_info,
    }


def draw_trajectory(output: Path, result: dict, n_data: int, threshold: int) -> None:
    history = result["history"]
    if not history:
        return
    intervals = np.array([interval_pair(step["amplitude_interval"]) for step in history])
    rounds = np.arange(1, len(history) + 1)
    midpoint = intervals.mean(axis=1)
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].errorbar(rounds, midpoint, yerr=(intervals[:, 1] - intervals[:, 0]) / 2,
                     fmt="o-", color="#18786e", capsize=4, label="Recorded amplitude interval")
    axes[0].axhline(threshold / 2**n_data, color="#d86b4a", linestyle="--", label="Exact threshold probability")
    finite = intervals[np.isfinite(intervals).all(axis=1)]
    limits = (max(-.03, float(finite[:, 0].min()) - .05), min(1.03, float(finite[:, 1].max()) + .05)) if len(finite) else (-.03, 1.03)
    axes[0].set(ylabel="Amplitude", ylim=limits, title="Fresh sampled rounds and retained confidence intervals")
    axes[0].legend(fontsize=8)
    axes[1].step(rounds, [2 * step["power"] + 1 for step in history], where="mid", color="#18786e")
    axes[1].set(ylabel="Multiplier 2k+1")
    axes[2].step(rounds, np.cumsum([step["shots"] for step in history]), where="mid", color="#345d9d")
    axes[2].set(xlabel="Recorded adaptive round", ylabel="Cumulative shots")
    fig.tight_layout()
    fig.savefig(output / "trajectory.svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-qubits", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--absolute-error", type=float, default=.02)
    parser.add_argument("--failure-probability", type=float, default=.01)
    parser.add_argument("--shots-per-round", type=int, default=256)
    parser.add_argument("--max-rounds", type=int, default=20)
    parser.add_argument("--max-power", type=int, default=64)
    parser.add_argument("--max-total-shots", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/amplitude-iterative")
    parser.add_argument("--no-suite", action="store_true")
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args()
    config = AdaptiveAEConfig(absolute_error=args.absolute_error, failure_probability=args.failure_probability,
                              shots_per_round=args.shots_per_round, max_rounds=args.max_rounds,
                              max_power=args.max_power, max_total_shots=args.max_total_shots)
    start = time.perf_counter()
    result = add_rng_hashes(run_threshold_adaptive(args.data_qubits, args.threshold, config=config, seed=args.seed))
    runtime = time.perf_counter() - start
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    for name in ("physical-resources.json", "validation-suite.json", "coverage.json", "trajectory.svg", "circuit.svg"):
        path = output / name
        if path.exists():
            path.unlink()
    for path in output.glob("power-*.qasm"):
        path.unlink()
    source_info = provenance(args.seed)
    write_json(output / "trajectory.json", {"trajectory": result, "source_and_environment": source_info,
                                           "cpu_wall_seconds_statevector_sampling_and_classical_inference": runtime})
    ledger = adaptive_logical_ledger(args.data_qubits, args.threshold, result)
    exports = export_power_circuits(output, args.data_qubits, args.threshold, result)
    if sum(item["all_shot_primitive_operations"] for item in exports) != ledger["total_primitive_operations"]:
        raise ValueError("Exported full circuits do not reproduce the weighted logical ledger")
    write_json(output / "logical-resources.json", {"ledger": ledger, "exported_power_circuits": exports,
                                                   "source_and_environment": source_info})
    for name in source_info["source_sha256"]:
        (output / name).write_bytes((ROOT / "src/openquantum" / name).read_bytes())
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    draw_trajectory(output, result, args.data_qubits, args.threshold)
    if exports and args.data_qubits <= 3:
        # The first nonzero recorded power demonstrates actual amplification.
        chosen = next((item for item in exports if item["power"] > 0), exports[0])
        circuit = build_amplified_trial(args.data_qubits, args.threshold, chosen["power"], measure=True)
        if chosen["power"] <= 8:
            fig = circuit.draw(output="mpl", fold=40, scale=.8)
            fig.savefig(output / "circuit.svg", bbox_inches="tight")
            plt.close(fig)
    if args.estimate:
        write_json(output / "physical-resources.json",
                   project_recorded_path(args.data_qubits, args.threshold, result, source_info, exports))
    if not args.no_suite:
        suite = validation_suite(config, args.seed)
        suite["source_and_environment"] = source_info
        write_json(output / "validation-suite.json", suite)
    if not args.no_coverage:
        coverage = model_coverage(config, args.seed)
        coverage["source_and_environment"] = source_info
        write_json(output / "coverage.json", coverage)
    final = result["final_amplitude_interval"]
    summary = [f"**Recorded status: {result['status']}.** Input: {args.data_qubits} data qubits and threshold {args.threshold}; requested error {args.absolute_error:g} and ideal failure allowance {args.failure_probability:g}.",
               f"The run used **{len(result['history'])} fresh batches**, **{result['total_shots']:,} flag trials**, and **{result['total_Q_calls']:,} actual amplification calls**.",
               f"Complete predicate/inverse accesses: **{result['total_threshold_predicate_or_inverse_calls']:,}**; full primitive operations: **{ledger['total_primitive_operations']:,}**."]
    if final is not None:
        summary.append(f"Final amplitude interval: **[{final['lower']:.12g}, {final['upper']:.12g}]**, midpoint **{final['estimate']:.12g}**, conservative radius **{final['radius']:.12g}**.")
    summary.append("The trajectory uses numerical ideal statevectors and seeded classical shots, with separately checked conservative arithmetic bounds. It is not a hardware confidence certificate.")
    (output / "summary.md").write_text("\n\n".join(summary) + "\n")
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(json.dumps({"status": result["status"], "recorded_total_shots": result["total_shots"],
                      "unique_recorded_powers": sorted(recorded_power_shots(result)),
                      "physical_projection_performed": args.estimate}))


if __name__ == "__main__":
    main()
