#!/usr/bin/env python3
"""Generate bounded threshold amplitude-estimation circuits and checked artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

from openquantum.amplitude_estimation import (
    amplitude_workload, analytic_amplitude_distribution, build_amplitude_estimation,
    decode_amplitude, exact_median_interval_failure, simulate_amplitude_estimation, threshold_preparation,
)
from openquantum.qft_qpe import lower_circuit
from openquantum.resource_estimation import (
    CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources,
)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def counts(n, k, m):
    circuit = build_amplitude_estimation(n, k, m, measure=True)
    compiled = lower_circuit(circuit)
    if len(compiled.data) > 250_000:
        raise ValueError("Primitive export exceeds 250,000-operation cap")
    return circuit, compiled, {
        "data_qubits": n, "threshold": k, "phase_bits": m,
        "logical_qubits": circuit.num_qubits,
        "controlled_Q_calls": 2**m - 1,
        "threshold_access_calls": 2**(m + 1) - 1,
        "decomposed_gate_counts": dict(compiled.count_ops()),
        "decomposed_dependency_depth": compiled.depth(),
        "decomposed_operations_including_measurement": len(compiled.data),
        "resource_kind": "actual logical gate counts before concrete synthesis and error correction",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-qubits", type=int, default=2)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--phase-bits", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/amplitude-estimation")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--estimate", action="store_true", help="Run the audited conditional QDK projection")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    n, k, m = args.data_qubits, args.threshold, args.phase_bits
    source, compiled, record = counts(n, k, m)
    # A new invocation must not carry an older instance's optional report or
    # figure into its output. Preserve unrelated files; remove only our names.
    for name in ("physical-resources.json", "surface-code-scaling.json", "scaling.csv", "distribution.svg", "preparation.svg", "circuit.svg"):
        optional_path = output / name
        if optional_path.exists():
            optional_path.unlink()
    qasm_text = qasm3.dumps(compiled)
    (output / "circuit.qasm").write_text(qasm_text)
    record["qasm_sha256"] = hashlib.sha256(qasm_text.encode()).hexdigest()
    record["qasm_phase_scope"] = "final measured circuit; an omitted overall scalar phase is not a coherent controlled-oracle specification"
    record["parameters"] = {"n_data": n, "threshold": k, "phase_bits": m, "exact_amplitude": k / 2**n}
    record["generation_bounds"] = {
        "data_qubits": 20, "phase_bits": 10,
        "repeated_source_instruction_cap_MCX_atomic": 250000,
        "shared_QPE_source_instruction_cap_MCX_atomic": 200000,
        "primitive_export_cap_after_transpilation": 250000,
        "statevector_total_qubits_cap": 12,
        "pre_transpilation_work_or_memory_certified": False,
    }
    record["simulation"] = {"performed": False, "reason": "disabled by --no-sim"}
    if not args.no_sim:
        probabilities = simulate_amplitude_estimation(n, k, m)
        expected = analytic_amplitude_distribution(k / 2**n, m)
        np.testing.assert_allclose(probabilities, expected, atol=2e-12, rtol=0)
        # Explicitly check controlled global phases survive the primitive lowering.
        unitary = build_amplitude_estimation(n, k, m)
        lowered = lower_circuit(unitary)
        original_state = Statevector.from_instruction(unitary).data
        compiled_state = Statevector.from_instruction(lowered).data
        phase = np.vdot(original_state, compiled_state)
        np.testing.assert_allclose(compiled_state, phase * original_state, atol=2e-11, rtol=0)
        estimates = [decode_amplitude(y, m) for y in range(2**m)]
        record["simulation"] = {
            "performed": True, "method": "ideal exact statevector, CPU",
            "phase_distribution": probabilities.tolist(),
            "analytic_distribution_max_abs_error": float(np.max(np.abs(probabilities - expected))),
            "decoded_amplitudes": estimates,
            "most_likely_outcomes": np.flatnonzero(probabilities >= probabilities.max() - 1e-12).tolist(),
            "one_run_mean_decoded_estimate": float(np.dot(probabilities, estimates)),
            "lowering_state_error_up_to_global_phase": float(np.max(np.abs(compiled_state - phase * original_state))),
            "repeated_median_hardware_workload_executed": False,
        }
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        axes[0].bar(range(2**m), probabilities, color="#18786e", label="Actual circuit")
        axes[0].plot(range(2**m), expected, ".", color="#d86b4a", label="Analytic QPE mixture")
        axes[0].set(xlabel="Phase outcome y", ylabel="Ideal probability", title=f"Threshold {k}/2^{n}; M={2**m}")
        axes[0].legend(fontsize=8)
        axes[1].scatter(estimates, probabilities, color="#18786e")
        axes[1].axvline(k / 2**n, color="#d86b4a", linestyle="--", label="Exact probability")
        axes[1].set(xlabel="Decoded sin²(πy/M)", ylabel="Outcome probability", title="Two phases give the same amplitude")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "distribution.svg")
        plt.close(fig)

    ideal_workload = amplitude_workload(n, k, m, failure_probability=.005)
    record["ideal_workload"] = ideal_workload
    if not args.no_sim:
        repeats = ideal_workload["repetitions"]
        epsilon = ideal_workload["absolute_amplitude_error_bound"]
        # Independent categorical samples of the exact ideal circuit law.
        # This is a classical sampling check, not actual circuit/device shots.
        rng = np.random.default_rng(8127)
        ensembles = 10_000
        outcomes = rng.choice(2**m, size=(ensembles, repeats), p=probabilities / probabilities.sum())
        medians = np.median(np.array(estimates)[outcomes], axis=1)
        errors = np.abs(medians - k / 2**n)
        record["ideal_median_coverage_check"] = {
            "kind": "classical categorical sampling of exact ideal statevector law",
            "seed": 8127, "ensembles": ensembles, "complete_runs_per_ensemble": repeats,
            "exact_ideal_interval_failure": exact_median_interval_failure(k / 2**n, m, repeats, probabilities),
            "observed_interval_failures": int(np.sum(errors > epsilon)),
            "observed_failure_frequency": float(np.mean(errors > epsilon)),
            "mean_absolute_error": float(errors.mean()),
            "hardware_execution": False,
        }
    versions = {}
    for name in ("qiskit", "numpy", "matplotlib", "qdk"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record["environment"] = {
        "python": platform.python_version(),
        "packages": versions,
        "device": "CPU", "random_seed": 8127,
        "seed_explanation": "Exact deterministic statevector plus separately labeled classical categorical median coverage check",
    }
    try:
        base_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                     capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        # The curated public source package deliberately has no .git directory.
        # Its explicit source/lock hashes remain the reproducibility identifiers.
        base_commit = None
    record["source_base_commit"] = base_commit
    record["source_base_commit_is_full_current_source_commit"] = False
    record["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lock_path = ROOT / "requirements-lock.txt"
    record["requirements_lock_sha256"] = hashlib.sha256(lock_path.read_bytes()).hexdigest() if lock_path.exists() else None
    record["implementation"] = {}
    for name in ("amplitude_estimation.py", "qft_qpe.py"):
        content = (ROOT / "src/openquantum" / name).read_bytes()
        (output / name).write_bytes(content)
        record["implementation"][name] = hashlib.sha256(content).hexdigest()
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    if n <= 3 and m <= 5:
        preparation = threshold_preparation(n, k)
        fig = preparation.draw(output="mpl", fold=40, scale=.9)
        fig.savefig(output / "preparation.svg", bbox_inches="tight")
        plt.close(fig)
        fig = source.draw(output="mpl", fold=45, scale=.6)
        fig.savefig(output / "circuit.svg", bbox_inches="tight")
        plt.close(fig)

    if not args.no_scaling:
        rows = []
        # Both problem width and precision grow. Every row compiles actual gates,
        # and none of these extra rows is claimed to have been simulated.
        for width in (1, 2, 3, 4):
            threshold = max(1, 2**width // 3)
            for bits in (2, 3, 4, 5):
                _, _, item = counts(width, threshold, bits)
                budget = amplitude_workload(width, threshold, bits, failure_probability=.005)
                row = {key: item[key] for key in ("data_qubits", "threshold", "phase_bits", "logical_qubits", "controlled_Q_calls", "threshold_access_calls", "decomposed_dependency_depth", "decomposed_operations_including_measurement")}
                row.update({"cx": item["decomposed_gate_counts"].get("cx", 0), "measurements": item["decomposed_gate_counts"].get("measure", 0),
                            "absolute_amplitude_error_bound": budget["absolute_amplitude_error_bound"], "complete_runs": budget["repetitions"],
                            "all_run_primitive_operations": item["decomposed_operations_including_measurement"] * budget["repetitions"],
                            "all_run_threshold_access_calls": item["threshold_access_calls"] * budget["repetitions"],
                            "simulated": False})
                rows.append(row)
        with (output / "scaling.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        record["scaling"] = {"performed": True, "rows": len(rows), "kind": "actual compiled logical counts; extra rows not simulated"}

    if args.estimate:
        projection = estimate_circuit_resources(
            source,
            budget=CircuitResourceBudget(synthesis_model_error_per_execution=1e-4,
                                         physical_failure_probability=.005,
                                         executions=ideal_workload["repetitions"]),
            profile=SurfaceCodeProfile(), cache_dir=Path(tempfile.gettempdir()) / "openquantum-amplitude-qdk-cache",
        )
        projection["algorithm_contract"] = ideal_workload
        projection["scope"] = "identical full static AE runs followed by ideal classical median; not a certified hardware amplitude interval"
        projection["precision_warning"] = "BHMT ideal accuracy is separate from modeled mean synthesis scalar and physical fault allowance; no actual synthesis certificate is available"
        write_json(output / "physical-resources.json", projection)
        record["physical_projection"] = {"performed": True, "kind": projection["resource_estimate_kind"],
                                          "pareto_points": len(projection["pareto_points"]),
                                          "physical_confidence_certified": False}
        scaling_reports = []
        physical_instances = [(n, k, m), (1, 1, 4), (3, 2, 4), (2, 1, 3), (2, 1, 5)]
        for width, threshold, bits in dict.fromkeys(physical_instances):
            contract = amplitude_workload(width, threshold, bits, failure_probability=.005)
            if (width, threshold, bits) == (n, k, m):
                full_projection = projection
                compiled_counts = {"decomposed_gate_counts": dict(compiled.count_ops()),
                                   "decomposed_operations_including_measurement": len(compiled.data),
                                   "decomposed_dependency_depth": compiled.depth()}
            else:
                scaling_source, _, compiled_counts = counts(width, threshold, bits)
                full_projection = estimate_circuit_resources(
                    scaling_source,
                    budget=CircuitResourceBudget(synthesis_model_error_per_execution=1e-4,
                                                 physical_failure_probability=.005,
                                                 executions=contract["repetitions"]),
                    profile=SurfaceCodeProfile(),
                    cache_dir=Path(tempfile.gettempdir()) / "openquantum-amplitude-qdk-cache",
                )
                full_projection["algorithm_contract"] = contract
                full_projection["scope"] = "identical full static AE runs plus ideal classical median; additional scaling row not statevector simulated here"
                full_projection["precision_warning"] = projection["precision_warning"]
            fastest = min(full_projection["pareto_points"], key=lambda point: point["serial_runtime_all_executions_ns"])
            scaling_reports.append({
                "parameters": {"n_data": width, "threshold": threshold, "phase_bits": bits,
                               "exact_amplitude": threshold / 2**width},
                "ideal_accuracy_and_confidence_contract": contract,
                "source_implementation_sha256": record["implementation"],
                "actual_compiled_counts": compiled_counts,
                "chosen_point_rule": "minimum serial runtime among all returned Pareto points",
                "fastest_point": fastest,
                "complete_conditional_projection": full_projection,
                "additional_scaling_statevector_simulation_performed": False,
                "hardware_execution_performed": False,
            })
        write_json(output / "surface-code-scaling.json", {
            "status": "computed_conditional_mean_fit_resource_scaling",
            "physical_profile": projection["physical_profile"],
            "per_run_synthesis_model_allowance": 1e-4,
            "whole_median_workload_physical_failure_allowance": .005,
            "whole_median_workload_ideal_failure_allowance": .005,
            "selection_caveat": "fastest returned model point; no certified synthesis, runtime-tail, hardware precision or advantage claim",
            "instances": scaling_reports,
        })
        record["physical_scaling"] = {"performed": True, "instances": len(scaling_reports),
                                       "kind": "computed conditional model at bounded width and resolution; additional rows not simulated"}
    else:
        record["physical_projection"] = {"performed": False, "reason": "--estimate not requested; any previous projection was removed"}
    write_json(output / "logical-resources.json", record)
    # Normalize renderer whitespace without changing SVG path semantics.
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(json.dumps({"data_qubits": n, "threshold": k, "phase_bits": m, "simulation": record["simulation"]["performed"], "physical_projection": record["physical_projection"]}))


if __name__ == "__main__":
    main()
