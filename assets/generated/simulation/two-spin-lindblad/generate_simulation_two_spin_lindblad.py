#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate coupled two-spin checks and bounded n-chain logical scaling."""
from __future__ import annotations

import argparse
from collections import Counter
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

import time as wallclock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/two_spin_lindblad.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-two-spin-lindblad"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum.simulation.two_spin_lindblad import (
        MAX_ENERGY_BUDGET_TRIALS, MAX_PRIMITIVE_OPERATIONS, MAX_REFERENCE_EXPOSURE,
        MAX_SIMULATION_STEPS, MAX_STEPS, MAX_SYSTEM_QUBITS, MAX_VALIDATION_POINTS,
        ROTATION_BASIS, build_chain_circuit, build_two_spin_circuit, compare_two_spin,
        energy_shot_budget, group_scores, lower_to_rotation_basis,
        measured_system_probabilities, reference_exposure,
    )
else:
    from two_spin_lindblad import (
        MAX_ENERGY_BUDGET_TRIALS, MAX_PRIMITIVE_OPERATIONS, MAX_REFERENCE_EXPOSURE,
        MAX_SIMULATION_STEPS, MAX_STEPS, MAX_SYSTEM_QUBITS, MAX_VALIDATION_POINTS,
        ROTATION_BASIS, build_chain_circuit, build_two_spin_circuit, compare_two_spin,
        energy_shot_budget, group_scores, lower_to_rotation_basis,
        measured_system_probabilities, reference_exposure,
    )


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def generate(output: Path, n_system: int = 2, time: float = 1.0,
              J: float = 0.6, hx: float = 0.35, rates=None, steps: int = 32,
              order: int = 2, initial: str = "product",
              grid=(1, 2, 4, 8, 16, 32), no_sim: bool = False,
              scaling: bool = True) -> dict:
    source = build_chain_circuit(n_system, time, J, hx, rates, steps, order, initial)
    rates = tuple(source.metadata["rates"])
    if not 1 <= len(grid) <= MAX_VALIDATION_POINTS or any(isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= MAX_SIMULATION_STEPS for r in grid):
        raise ValueError("validation grid exceeds its step/count cap")
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "two_spin_lindblad.py": SOURCE / "openquantum/simulation/two_spin_lindblad.py" if REPOSITORY else HERE / "two_spin_lindblad.py",
        "open_systems.py": SOURCE / "openquantum/simulation/open_systems.py" if REPOSITORY else HERE / "open_systems.py",
        "generate_simulation_two_spin_lindblad.py": Path(__file__),
    }
    hashes = {}
    for name, path in paths.items():
        raw = path.read_bytes()
        (output / name).write_bytes(raw)
        hashes[name] = sha256(raw)
    license_path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())
    measured = {}
    for group in ("X", "Y", "Z"):
        started = wallclock.perf_counter()
        trial = build_chain_circuit(n_system, time, J, hx, rates, steps, order, initial, observable=group)
        compiled = lower_to_rotation_basis(trial)
        elapsed = wallclock.perf_counter()-started
        qasm = qasm3.dumps(compiled)
        filename = f"two-spin-{group.lower()}.qasm"
        (output / filename).write_text(qasm)
        counts = dict(compiled.count_ops())
        measured[group] = {
            "source_gate_counts": dict(trial.count_ops()), "primitive_gate_counts_per_trial": counts,
            "logical_qubits": compiled.num_qubits, "dependency_depth": compiled.depth(),
            "rotation_operations": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
            "system_initializations": n_system,
            "environment_initializations": trial.metadata["environment_initializations"],
            "environment_discards_before_reuse": trial.metadata["environment_discard_resets_before_reuse"],
            "system_measurements": n_system, "environment_measurements_ignored": n_system,
            "classical_build_and_lower_wall_seconds": elapsed,
            "qasm_file": filename, "qasm_sha256": sha256(qasm.encode()),
        }
    exposure = reference_exposure(time, J, hx, rates) if n_system == 2 else None
    performed = bool(n_system == 2 and not no_sim and steps <= MAX_SIMULATION_STEPS and exposure <= MAX_REFERENCE_EXPOSURE)
    selected, records, input_checks = None, [], []
    checks_elapsed = None
    if performed:
        started = wallclock.perf_counter()
        selected = compare_two_spin(time, J, hx, rates, steps, order, initial)
        records = [compare_two_spin(time, J, hx, rates, r, o, initial) for o in (1, 2) for r in grid]
        input_checks = [compare_two_spin(time, J, hx, rates, steps, order, state, h_diagnostic=False)
                        for state in ("product", "bell", "zero")]
        checks_elapsed = wallclock.perf_counter()-started
        if any(abs(r["trace_real"]-1) > 2e-11 or abs(r["trace_imag"]) > 2e-11 or r["minimum_eigenvalue"] < -2e-11 for r in records+input_checks):
            raise AssertionError("actual coupled circuit produced an invalid reduced density")
        if any(r["circuit_vs_classical_split_trace_distance"] > 2e-11 for r in records+input_checks):
            raise AssertionError("actual circuit does not match the stated split component channel")
        rows = [{"order": r["order"], "steps": r["steps"], "trace_distance": r["trace_distance"],
                 "Z0_error": r["observable_errors"]["Z0"], "ZZ_error": r["observable_errors"]["ZZ"],
                 "energy_error": r["observable_errors"]["energy"],
                 "H_only_diagnostic": r["hamiltonian_only_trace_distance_diagnostic"],
                 "circuit_vs_split_reference": r["circuit_vs_classical_split_trace_distance"]} for r in records]
        write_csv(output / "convergence.csv", rows)
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        for o, color in ((1, "#346b8a"), (2, "#be6b35")):
            group = [r for r in rows if r["order"] == o]
            for ax, key in zip(axes, ("trace_distance", "ZZ_error", "energy_error")):
                ax.loglog([r["steps"] for r in group], [max(r[key], 1e-16) for r in group], "o-", color=color, label=f"Order {o}")
                ax.set(xlabel="Splitting steps", ylabel=key.replace("_", " "))
                ax.grid(alpha=.2)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / "convergence.svg", metadata={"Date": None})
        plt.close(fig)
    else:
        for name in ("convergence.csv", "convergence.svg"):
            (output / name).unlink(missing_ok=True)
    if n_system == 2:
        fig = build_two_spin_circuit(time, J, hx, rates, 1, order, initial, observable="Z").draw(output="mpl", fold=30)
        fig.savefig(output / "one-step-circuit.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    else:
        (output / "one-step-circuit.svg").unlink(missing_ok=True)
    energy_budget = energy_shot_budget(J, hx) if n_system == 2 else None
    sampling = {"performed": False}
    if selected is not None and energy_budget["groups"]:
        estimates, samples = {}, {}
        for group, allocation in energy_budget["groups"].items():
            trial = lower_to_rotation_basis(build_two_spin_circuit(time, J, hx, rates, steps, order, initial, observable=group))
            probabilities = measured_system_probabilities(trial)
            counts = np.random.default_rng(44 if group == "X" else 45).multinomial(allocation["trials"], probabilities/probabilities.sum())
            estimate = float(counts @ group_scores(group, J, hx)/allocation["trials"])
            estimates[group] = estimate
            samples[group] = {"joint_system_counts_q1q0": counts.tolist(), "mean_weighted_score": estimate,
                              "trials": allocation["trials"], "seed": 44 if group == "X" else 45}
        sampling = {"performed": True, "groups": samples, "estimated_energy": sum(estimates.values()),
                    "kind": "one seeded classical draw from each actual measured circuit's joint system probabilities",
                    "scope": "illustrative only; bounded-score Hoeffding confidence is analytic, not established by the draw"}
    scaling_rows = []
    if scaling:
        for n in (1, 2, 3, 4, 8):
            for r in (2, 4, 8):
                started = wallclock.perf_counter()
                candidate = build_chain_circuit(n, time, J, hx, None, r, 2, "product", observable="Z")
                compiled = lower_to_rotation_basis(candidate)
                counts = dict(compiled.count_ops())
                scaling_rows.append({"n_system": n, "n_environment": n, "steps": r, "order": 2,
                                     "time": time, "J": J, "hx": hx, "initial": "product",
                                     "rates": json.dumps(candidate.metadata["rates"]),
                                     "logical_qubits": compiled.num_qubits, "primitive_operations": compiled.size(),
                                     "cx": counts.get("cx", 0), "rotations": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
                                     "resets": counts.get("reset", 0), "measurements": counts.get("measure", 0),
                                     "dependency_depth": compiled.depth(),
                                     "classical_build_and_lower_wall_seconds": wallclock.perf_counter()-started})
        write_csv(output / "scaling.csv", scaling_rows)
    else:
        (output / "scaling.csv").unlink(missing_ok=True)
    energy_workload = None
    if energy_budget is not None:
        totals = Counter()
        qubit_layers = 0
        for group, allocation in energy_budget["groups"].items():
            N = allocation["trials"]
            totals.update({gate: count*N for gate, count in measured[group]["primitive_gate_counts_per_trial"].items()})
            qubit_layers += N*measured[group]["logical_qubits"]*measured[group]["dependency_depth"]
        norm_bound = abs(J)+2*abs(hx)
        energy_workload = {**energy_budget, "observable_absolute_error_target": 0.05,
                           "global_splitting_bias_allowance": 0.01, "coherent_synthesis_bias_allowance": 0.01,
                           "physical_failure_across_all_trials": 0.005,
                           "physical_union_bound_per_trial": 0.005/energy_budget["total_trials"] if energy_budget["total_trials"] else None,
                           "selected_splitting_bias_bound_from_trace_distance": 2*norm_bound*selected["trace_distance"] if selected else None,
                           "splitting_allowance_met_numerically": bool(2*norm_bound*selected["trace_distance"] <= 0.01) if selected else None,
                           "all_group_trials_primitive_counts": dict(totals), "logical_qubit_layer_volume": qubit_layers,
                           "volume_scope": "circuit dependency-layer proxy, not physical qubit-seconds or hardware runtime",
                           "synthesis_scope": "pending per-gate certificates; not a stochastic fault probability"}
    revision = git_revision(ROOT) if REPOSITORY else None
    report = {
        "status": "checked coupled two-spin global splitting" if performed else "generated chain circuits only; density validation not performed",
        "instance": {"n_system": n_system, "time": time, "J": J, "hx": hx, "rates": list(rates),
                     "steps": steps, "order": order, "initial": initial,
                     "input_contract": "product means |+i> on system0 and |+> on the remaining systems; Bell is implemented only at n=2",
                     "Hamiltonian": "J sum Zi Zi+1 + hx sum Xi, open boundaries", "units": "hbar=1; inverse-time coefficients and rates"},
        "validation": {"performed": performed, "selected": selected, "convergence_records": records,
                       "input_checks": input_checks, "classical_density_and_reference_wall_seconds": checks_elapsed,
                       "reference": "independent scaled16x16 column-vectorized standard-GKSL exp and separate split-component matrix check",
                       "scope": "combined Liouvillian accuracy study n=2; n=1/n=3 circuit checks in tests, larger n counts only"},
        "illustrative_sampling": sampling,
        "environment": {"python": platform.python_version(), "machine": platform.machine(), "device": "CPU",
                        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "scipy", "matplotlib")}},
        "source_revision_at_generation": revision, "source_sha256": hashes,
        "scaling_family": {"n_system": [1, 2, 3, 4, 8], "steps": [2, 4, 8], "order": 2,
                           "time": time, "J": J, "hx": hx,
                           "initial": "product: |+i> on site0, |+> elsewhere",
                           "rates": "0.8 on site0, 0.5 elsewhere; independent of selected custom rates",
                           "scope": "executed construction/decomposition only; not matched-accuracy density results"},
        "source_note": "working-tree source hashes identify pre-commit code; revision is generation parent"}
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    ledger = {
        "resource_kind": "actual primitive measured/reset circuits before synthesis/error correction",
        "physical_resource_estimate": None, "logical_qubits": 2*n_system,
        "component_calls": source.metadata["component_calls"],
        "environment_initializations_per_trial": source.metadata["environment_initializations"],
        "environment_discard_resets_before_reuse": source.metadata["environment_discard_resets_before_reuse"],
        "final_environment_measurements_ignored": n_system,
        "input_preparation": "known product H/S gates or explicit two-qubit Bell H/CX; system resets included",
        "readout": "X/Y/Z basis on every system, Z on environments; no environment postselection",
        "measurement_groups": measured, "energy_workload_proposal_n2": energy_workload,
        "global_error_contract": "all H-term and H/D splitting commutators are in the global error; H-only diagnostic is not added",
        "separate_errors": {"representation": "fixed real coefficients and floating angles",
                            "global_splitting": "selected n2 numerical trace/observable errors; no diamond/interval certificate",
                            "coherent_synthesis": "pending; rotations continuous",
                            "statistics": "two bounded weighted-score means and union bound; correlated bits within a trial allowed",
                            "physical_faults": "pending; separate from intended damping and approximation"},
        "source_complexity": {"logical_qubits": "2n", "source_Lie_primitive_upper": "(10n-3)r+6n+8",
                              "scaling_family": "order2/product, rates0.8 on site0 and0.5 elsewhere; selected time/J/hx",
                              "source_Strang_primitive_upper": "(20n-12)r+6n+8",
                              "n_scaling_scope": "counts materialized for n=1,2,3,4,8 at r=2,4,8; same-accuracy r(n) unvalidated",
                              "energy_general_n_conditional": "fixed absolute statistical error: group widths O(n), Hoeffding trials O(n^2/epsilon_stat^2 log(1/delta)); combined with O(nr) gates gives conditional O(n^3 r/epsilon_stat^2 log(1/delta)), before choosing accuracy-dependent r(n)"},
        "prominent_exclusions": ["coefficient preparation/readout wall time in quantum costs", "physical reset/discard/measurement duration",
                                  "hardware routing/scheduling", "coherent rotation synthesis", "physical faults/error correction/factories"],
        "physical_adapter": "mid-circuit resets are not supported by the current static adapter",
        "formal_coverage": "no integrated Lean or other formal certificate",
        "limits": {"generator_system_qubits": MAX_SYSTEM_QUBITS, "generator_steps": MAX_STEPS,
                   "estimated_and_compiled_primitive_cap": MAX_PRIMITIVE_OPERATIONS,
                   "density_study_system_qubits": 2, "density_steps": MAX_SIMULATION_STEPS,
                   "reference_exposure": MAX_REFERENCE_EXPOSURE, "validation_points_per_order": MAX_VALIDATION_POINTS,
                   "energy_budget_trial_cap": MAX_ENERGY_BUDGET_TRIALS},
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7, "basis": ROTATION_BASIS}, "source_sha256": hashes}
    (output / "logical-resources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for name in ("convergence.svg", "one-step-circuit.svg"):
        path = output / name
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return {"output_dir": str(output), "validation_performed": performed,
            "trace_distance": selected["trace_distance"] if selected else None,
            "energy_error": selected["observable_errors"]["energy"] if selected else None,
            "Z_counts": measured["Z"]["primitive_gate_counts_per_trial"],
            "energy_trials": energy_budget["total_trials"] if energy_budget else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-system", type=int, default=2)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=0.6)
    parser.add_argument("--hx", type=float, default=0.35)
    parser.add_argument("--rates", help="comma-separated rates; defaults to0.8 on site0 and0.5 elsewhere")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--initial", choices=("product", "bell", "zero"), default="product")
    parser.add_argument("--grid", default="1,2,4,8,16,32")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/two-spin-lindblad" if REPOSITORY else HERE / "generated-two-spin")
    args = parser.parse_args()
    rates = tuple(float(r) for r in args.rates.split(",")) if args.rates else None
    print(json.dumps(generate(args.output_dir, args.n_system, args.time, args.J, args.hx, rates,
                              args.steps, args.order, args.initial, tuple(int(r) for r in args.grid.split(",")),
                              args.no_sim, not args.no_scaling), indent=2))


if __name__ == "__main__":
    main()
