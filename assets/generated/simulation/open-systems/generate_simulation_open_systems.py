#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit amplitude-damping dilations and bounded classical checks.

The one-system example is validated; --n-system N --no-sim generates N
independent gate-based dilations without allocating a density matrix.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/open_systems.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-amplitude-damping"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum.simulation.open_systems import (
        INPUT_LABELS, ROTATION_BASIS, build_amplitude_damping, compare_damping,
        damping_parameters, damping_probability, independent_damping_circuit, lower_to_rotation_basis,
        observable_shots, simulate_prepared_damping,
    )
else:
    from open_systems import (
        INPUT_LABELS, ROTATION_BASIS, build_amplitude_damping, compare_damping,
        damping_parameters, damping_probability, independent_damping_circuit, lower_to_rotation_basis,
        observable_shots, simulate_prepared_damping,
    )


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def generate(output: Path, rate: float = 0.8, time: float = 1.0,
              input_label: str = "plus", times=(0, 0.25, 0.5, 1, 2, 50),
              n_system: int = 1, no_sim: bool = False) -> dict:
    parameters = damping_parameters(rate, time)
    if not 1 <= len(times) <= 32:
        raise ValueError("validation grid needs 1 to 32 times")
    for value in times:
        damping_probability(rate, value)
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "open_systems.py": SOURCE / "openquantum/simulation/open_systems.py" if REPOSITORY else HERE / "open_systems.py",
        "generate_simulation_open_systems.py": Path(__file__),
    }
    source_hashes = {}
    for name, path in sources.items():
        raw = path.read_bytes()
        (output / name).write_bytes(raw)
        source_hashes[name] = _sha256(raw)
    license_path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())
    ledgers = {}
    for observable in ("X", "Y", "Z"):
        source = independent_damping_circuit(n_system, rate, time,
                                             inputs=(input_label,) * n_system,
                                             observable=observable, reset=True)
        compiled = lower_to_rotation_basis(source)
        qasm = qasm3.dumps(compiled)
        filename = f"amplitude-damping-{observable.lower()}.qasm"
        (output / filename).write_text(qasm)
        counts = dict(compiled.count_ops())
        ledgers[observable] = {
            "decomposed_gate_counts_per_trial": counts,
            "decomposed_dependency_depth_per_trial": compiled.depth(),
            "arbitrary_rotations_per_trial": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
            "system_resets_per_trial": n_system, "environment_resets_per_trial": n_system,
            "system_measurements_per_trial": n_system, "environment_measurements_per_trial": n_system,
            "qasm_file": filename, "qasm_sha256": _sha256(qasm.encode()),
        }
    performed = n_system == 1 and not no_sim
    records = []
    if performed:
        records = [compare_damping(rate, value, label) for label in INPUT_LABELS for value in times]
        if max(item["trace_distance"] for item in records) > 1e-12 or max(item["maximum_observable_error"] for item in records) > 1e-12:
            raise AssertionError("implemented dilation disagrees with independent Kraus reference")
        rows = []
        for item in records:
            rows.append({
                "input": item["input"], "time": item["time"], "rate": item["rate"],
                "damping_probability": item["damping_probability"],
                "population_survival": item["population_survival"], "coherence_survival": item["coherence_survival"],
                "trace_distance": item["trace_distance"], "minimum_eigenvalue": item["minimum_eigenvalue"],
                **{f"circuit_{pauli}": item["circuit_observables"][pauli] for pauli in ("X", "Y", "Z")},
                "maximum_observable_error": item["maximum_observable_error"],
            })
        with (output / "channel-checks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        display_times = np.linspace(0, 2, 81)
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
        for label, color in zip(INPUT_LABELS, ("#9b694a", "#216e73", "#346b8a", "#8057a3")):
            data = [compare_damping(rate, float(t), label) for t in display_times]
            axes[0].plot(display_times, [item["circuit_observables"]["Z"] for item in data], color=color, label=label)
        plus = [compare_damping(rate, float(t), "plus") for t in display_times]
        axes[1].plot(display_times, [item["circuit_observables"]["X"] for item in plus], color="#346b8a", label="X for |+>")
        axes[1].plot(display_times, np.exp(-rate * display_times/2), "--", color="#9b694a", label="analytic exp(-gamma t/2)")
        axes[0].set(xlabel="Time", ylabel="System Z expectation", ylim=(-1.05, 1.05))
        axes[1].set(xlabel="Time", ylabel="System coherence", ylim=(-0.05, 1.05))
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / "damping-observables.svg", metadata={"Date": None})
        plt.close(fig)
        fig = build_amplitude_damping(rate, time, input_label, observable="Z", reset=True).draw(output="mpl", fold=24)
        fig.savefig(output / "dilation-circuit.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    shots = observable_shots()
    sampling = {"performed": False, "reason": "simulation disabled or above fixed one-system validation"}
    if performed:
        density = simulate_prepared_damping(rate, time, input_label)
        p_zero = float(density[0, 0].real)
        zero_count = int(np.random.default_rng(41).binomial(shots, p_zero))
        sampling = {
            "performed": True, "kind": "seeded classical draw from the actual ideal circuit's system probabilities",
            "seed": 41, "shots": shots, "q0_zero_count": zero_count,
            "estimated_Z": 2 * zero_count / shots - 1,
            "note": "illustrative draw only; Hoeffding bound is analytic and not established by one draw",
        }
    revision = None
    if REPOSITORY:
        revision = git_revision(ROOT)
    report = {
        "status": "checked one-qubit amplitude-damping channel; general open-system algorithms planned" if performed else "generated independent-damping circuit only; simulation not performed",
        "instance": {"n_system": n_system, "rate": rate, "time": time,
                     **parameters, "input_label_per_system": input_label,
                     "units": "rate in inverse time, hbar=1; H=0, L=sqrt(rate)|0><1|"},
        "validation": {"performed": performed, "records": records,
                       "reference": "independent finite-time population/coherence formula; actual gates followed by partial trace",
                       "scope": "fixed2-qubit dense check; independent tensor-product generation is separate"},
        "illustrative_sampling": sampling,
        "environment": {"python": platform.python_version(), "backend": "CPU Qiskit Statevector/DensityMatrix",
                        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")}},
        "source_revision_at_generation": revision, "source_sha256": source_hashes,
        "source_note": "working-tree source hashes identify code created before commit; revision is generation parent",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    z_counts = ledgers["Z"]["decomposed_gate_counts_per_trial"]
    ledger = {
        "resource_kind": "logical primitive circuit counts before coherent synthesis and physical faults",
        "physical_resource_estimate": None, "logical_qubits": 2 * n_system,
        "system_qubits": n_system, "environment_qubits": n_system,
        "input_preparation": "all wires reset; known product input from explicit X/H/S gates",
        "actual_access": "CRY on each zero-input environment controlled by its system, followed by reverse CX",
        "environment_handling": "measure every environment wire in Z and ignore all those bits; no postselection",
        "retained_state_output": "before system readout, tracing/discarding environment gives the channel state",
        "state_output_vs_observable_trial": "exports measure every system wire; the output is a measured trial, not a retained state",
        "classical_coefficient_preparation": "expm1, exp survival, sqrt and atan2 angle evaluation per fixed rate/time; reusable across iid shots",
        "count_basis": ROTATION_BASIS,
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
        "measurement_circuits": ledgers,
        "one_Z_observable_budget_proposal": {
            "absolute_error_target": 0.05, "channel_bias_allowance": 0.01,
            "coherent_synthesis_bias_allowance": 0.01, "statistical_error_allowance": 0.03,
            "statistical_failure_probability": 0.005, "physical_failure_across_all_trials": 0.005,
            "independent_serial_trials": shots, "physical_failure_union_bound_per_trial": 0.005/shots,
            "channel_note": "finite-time ideal dilation matches exact Kraus/GKSL channel numerically; no time-discretization error",
            "angle_rounding_note": "finite-time survival is retained directly; floating rotation angle precision near pi limits extreme-time accuracy",
            "synthesis_note": "continuous-angle rotations remain unsynthesized; this is an allocation, not a certificate",
            "scope": "one fixed Z(q0) mean; X/Y tomography and multiple input/time settings need separate allocations",
            "all_trials_Z_gate_counts": {gate: count * shots for gate, count in z_counts.items()},
        },
        "prominent_exclusions": ["classical angle/coefficient preparation wall time", "classical readout processing wall time",
                                  "physical reset/discard/readout time", "routing and hardware scheduling",
                                  "coherent rotation synthesis", "physical faults/error correction/factories"],
        "static_adapter_compatibility": "explicit reset operations require separate initialization treatment; current static shared adapter rejects reset",
        "limits": {"generator_system_qubits": 32, "dense_validation_total_qubits": 2,
                   "validation_grid_cli_time_points": 32, "figure_fixed_time_points": 81,
                   "scalable_scope": "independent product channel, not interacting Lindblad evolution"},
        "source_sha256": source_hashes,
    }
    (output / "logical-resources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for name in ("damping-observables.svg", "dilation-circuit.svg"):
        path = output / name
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return {"output_dir": str(output), "validated_cases": len(records),
            "max_trace_distance": max((r["trace_distance"] for r in records), default=None),
            "Z_counts_per_trial": z_counts, "Z_trials_proposal": shots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=0.8)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--input", choices=INPUT_LABELS, default="plus")
    parser.add_argument("--times", default="0,0.25,0.5,1,2,50")
    parser.add_argument("--n-system", type=int, default=1)
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/open-systems" if REPOSITORY else HERE / "generated-open-systems")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.rate, args.time, args.input,
                              tuple(float(t) for t in args.times.split(",")), args.n_system, args.no_sim), indent=2))


if __name__ == "__main__":
    main()
