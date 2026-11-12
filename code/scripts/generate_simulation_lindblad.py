#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit one-qubit Lindblad splitting circuits and bounded checks."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/lindblad.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-lindblad-splitting"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum.simulation.lindblad import (
        INPUT_LABELS, MAX_PRIMITIVE_OPERATIONS, MAX_REFERENCE_EVOLUTION_SCALE, MAX_SIMULATION_STEPS,
        MAX_STEPS, MAX_VALIDATION_POINTS, PAULI_MATRICES, ROTATION_BASIS,
        build_lindblad_circuit, compare_lindblad, exact_density, input_density,
        liouvillian, lower_to_rotation_basis, observable_shots, simulate_lindblad,
    )
else:
    from lindblad import (
        INPUT_LABELS, MAX_PRIMITIVE_OPERATIONS, MAX_REFERENCE_EVOLUTION_SCALE, MAX_SIMULATION_STEPS,
        MAX_STEPS, MAX_VALIDATION_POINTS, PAULI_MATRICES, ROTATION_BASIS,
        build_lindblad_circuit, compare_lindblad, exact_density, input_density,
        liouvillian, lower_to_rotation_basis, observable_shots, simulate_lindblad,
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path, time: float = 1.0, rate: float = 0.8,
              hx: float = 0.7, hz: float = 0.3, steps: int = 32, order: int = 2,
              input_label: str = "plus_i", grid=(1, 2, 4, 8, 16, 32, 64),
              no_sim: bool = False) -> dict:
    if not 1 <= len(grid) <= MAX_VALIDATION_POINTS or any(isinstance(r, bool) or not isinstance(r, int) or not 1 <= r <= MAX_SIMULATION_STEPS for r in grid):
        raise ValueError("validation grid exceeds its step/count bound")
    source = build_lindblad_circuit(time, rate, hx, hz, steps, order, input_label)
    output.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "lindblad.py": SOURCE / "openquantum/simulation/lindblad.py" if REPOSITORY else HERE / "lindblad.py",
        "open_systems.py": SOURCE / "openquantum/simulation/open_systems.py" if REPOSITORY else HERE / "open_systems.py",
        "generate_simulation_lindblad.py": Path(__file__),
    }
    hashes = {}
    for name, path in source_paths.items():
        raw = path.read_bytes()
        (output / name).write_bytes(raw)
        hashes[name] = _sha256(raw)
    license_path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())
    measured_ledgers = {}
    for observable in ("X", "Y", "Z"):
        measured = build_lindblad_circuit(time, rate, hx, hz, steps, order,
                                          input_label, observable=observable)
        compiled = lower_to_rotation_basis(measured)
        qasm = qasm3.dumps(compiled)
        filename = f"lindblad-{observable.lower()}.qasm"
        (output / filename).write_text(qasm)
        counts = dict(compiled.count_ops())
        measured_ledgers[observable] = {
            "source_gate_counts": dict(measured.count_ops()),
            "decomposed_gate_counts_per_trial": counts,
            "dependency_depth_per_trial": compiled.depth(),
            "arbitrary_rotations_per_trial": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
            "system_initializations_per_trial": 1, "environment_initializations_per_trial": steps,
            "environment_discard_resets_before_reuse": steps-1,
            "final_system_measurements": 1, "final_environment_measurements": 1,
            "qasm_file": filename, "qasm_sha256": _sha256(qasm.encode()),
        }
    scale = float(max(time * rate, 2 * (time * np.hypot(hx, hz))))
    performed = bool(not no_sim and steps <= MAX_SIMULATION_STEPS and scale <= MAX_REFERENCE_EVOLUTION_SCALE)
    selected = None
    records = []
    input_checks = []
    if performed:
        selected = compare_lindblad(time, rate, hx, hz, steps, order, input_label)
        records = [compare_lindblad(time, rate, hx, hz, r, o, input_label) for o in (1, 2) for r in grid]
        if any(abs(r["trace_real"]-1) > 2e-12 or abs(r["trace_imag"]) > 2e-12 or r["minimum_eigenvalue"] < -2e-12 for r in records):
            raise AssertionError("a convergence-grid circuit produced an invalid reduced state")
        for label in INPUT_LABELS:
            rho = simulate_lindblad(time, rate, hx, hz, steps, order, label)
            reference = exact_density(input_density(label), time, rate, hx, hz)
            input_checks.append({
                "input": label, "trace_distance": float(np.linalg.norm(rho-reference, ord="nuc")/2),
                "trace_real": float(np.trace(rho).real), "trace_imag": float(np.trace(rho).imag),
                "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
                "observable_errors": {name: abs(float(np.trace((rho-reference) @ P).real)) for name, P in PAULI_MATRICES.items()},
            })
        if any(abs(r["trace_real"]-1) > 2e-12 or abs(r["trace_imag"]) > 2e-12 or r["minimum_eigenvalue"] < -2e-12 for r in input_checks):
            raise AssertionError("actual split circuit produced an invalid reduced state")
        rows = [{
            "order": r["order"], "steps": r["steps"], "trace_distance": r["trace_distance"],
            "superoperator_spectral_error": r["superoperator_spectral_error"],
            **{f"{name}_error": r["observable_errors"][name] for name in ("X", "Y", "Z")},
        } for r in records]
        with (output / "convergence.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        for o, color in ((1, "#346b8a"), (2, "#be6b35")):
            group = [item for item in rows if item["order"] == o]
            for ax, key in zip(axes, ("trace_distance", "superoperator_spectral_error", "Z_error")):
                ax.loglog([r["steps"] for r in group], [max(r[key], 1e-16) for r in group], "o-", color=color, label=f"Order {o}")
                ax.set(xlabel="Splitting steps", ylabel=key.replace("_", " "))
                ax.grid(alpha=.2)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / "convergence.svg", metadata={"Date": None})
        plt.close(fig)
        one_step = build_lindblad_circuit(time, rate, hx, hz, 1, order, input_label, observable="Z")
        fig = one_step.draw(output="mpl", fold=24)
        fig.savefig(output / "one-step-circuit.svg", bbox_inches="tight", metadata={"Date": None})
        plt.close(fig)
    shots = observable_shots()
    sampling = {"performed": False}
    if selected is not None:
        rho = np.array(selected["actual_density_real"])+1j*np.array(selected["actual_density_imag"])
        p_zero = float(rho[0, 0].real)
        zero_count = int(np.random.default_rng(43).binomial(shots, p_zero))
        sampling = {"performed": True, "kind": "seeded classical draw from actual split-circuit probabilities",
                    "seed": 43, "shots": shots, "zero_count": zero_count, "estimated_Z": 2*zero_count/shots-1,
                    "scope": "one illustrative iid draw; it does not establish the confidence bound"}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if REPOSITORY else None
    report = {
        "status": "checked one-qubit Lindblad Lie/Strang circuits" if performed else "generated measured circuits; validation not performed",
        "instance": {"time": time, "rate": rate, "hx": hx, "hz": hz, "steps": steps, "order": order,
                     "initial_state": input_label, "units": "hbar=1; Hamiltonian coefficients in inverse time, rate in inverse time",
                     "generator": "-i[H,rho]+LrhoLdagger-0.5{LdaggerL,rho}, H=hxX+hzZ, L=sqrt(rate)|0><1|"},
        "validation": {"performed": performed, "selected": selected, "convergence_records": records,
                       "input_checks_at_selected_steps": input_checks,
                       "reference": "independent4x4 column-vectorized Liouvillian exponential; never a quantum circuit gate",
                       "metric_scope": "selected-input trace distance and2->2 Hilbert-Schmidt superoperator norm; no diamond certificate"},
        "illustrative_sampling": sampling,
        "environment": {"python": platform.python_version(), "backend": "CPU Qiskit DensityMatrix gate/reset evolution; SciPy expm reference",
                        "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "scipy", "matplotlib")}},
        "source_revision_at_generation": revision, "source_sha256": hashes,
        "source_note": "working-tree source hashes identify code created before commit; revision is generation parent",
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    z_counts = measured_ledgers["Z"]["decomposed_gate_counts_per_trial"]
    ledger = {
        "resource_kind": "logical primitive channel-circuit counts before synthesis/error correction",
        "physical_resource_estimate": None, "system_qubits": 1, "environment_qubits": 1, "logical_qubits": 2,
        "input_preparation": "known label from explicit X/H/S gates, system initial reset included",
        "hamiltonian_calls_per_trial": steps if order == 1 else 2*steps,
        "hamiltonian_approximation": "none in analytic exact-angle RY-RZ-RY substep; only outer channel splitting",
        "damping_calls_per_trial": steps,
        "environment_protocol": "reset before every damping step; each reset after first discards prior environment; final Z record ignored",
        "environment_initializations_per_trial": steps, "environment_discards_before_reuse": steps-1,
        "final_environment_measurements_ignored": 1,
        "state_output_vs_trial": "reduced state is before final system readout; QASM exports are measured trials",
        "postselection": "none", "measurement_circuits": measured_ledgers, "count_basis": ROTATION_BASIS,
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
        "one_fixed_Z_mean_budget_proposal": {
            "observable_absolute_error_target": 0.05, "outer_splitting_bias_allowance": 0.01,
            "inner_hamiltonian_approximation_allowance": 0.0,
            "coherent_synthesis_bias_allowance": 0.01, "statistical_error_allowance": 0.03,
            "statistical_failure_probability": 0.005, "physical_failure_across_all_trials": 0.005,
            "independent_trials": shots, "physical_union_bound_per_trial": 0.005/shots,
            "selected_input_splitting_bias_bound_from_trace_distance": 2*selected["trace_distance"] if selected else None,
            "splitting_allowance_met_numerically": 2*selected["trace_distance"] <= 0.01 if selected else None,
            "coherent_gate_operator_error_sum_allowance": 0.005,
            "per_rotation_operator_error_allocation": 0.005/measured_ledgers["Z"]["arbitrary_rotations_per_trial"] if measured_ledgers["Z"]["arbitrary_rotations_per_trial"] else None,
            "scope": "numerical toy-instance allowance, pending certified synthesis; one fixed Z mean, not tomography/multiple settings",
            "all_trials_Z_gate_counts": {gate: count*shots for gate, count in z_counts.items()},
        },
        "separate_errors": {"representation": "stated real coefficients and IEEE floating angles",
                            "outer_splitting": "compared with exact small Liouvillian evolution",
                            "inner_hamiltonian": "analytic exact-angle circuit; no inner product formula",
                            "coherent_synthesis": "pending; continuous rotations are unsynthesized",
                            "statistical": "one-mean Hoeffding proposal; seeded draw is illustrative",
                            "physical_faults": "pending; not equated with approximation error"},
        "prominent_exclusions": ["classical coefficient/angle preparation wall time", "classical readout processing wall time",
                                  "physical reset/discard/measurement time", "routing/hardware scheduling",
                                  "coherent rotation synthesis", "physical errors/error correction/factories"],
        "static_adapter_compatibility": "mid-circuit environment resets require an instrument/reset-aware resource model; current static adapter rejects them",
        "limits": {"system_qubits": 1, "environment_qubits": 1, "generator_steps": MAX_STEPS,
                   "estimated_primitive_operation_bound": MAX_PRIMITIVE_OPERATIONS,
                   "simulation_steps": MAX_SIMULATION_STEPS, "validation_grid_points_per_order": MAX_VALIDATION_POINTS,
                   "dense_reference_evolution_scale": MAX_REFERENCE_EVOLUTION_SCALE,
                   "scope": "one-qubit H plus one local damping jump; general interacting Lindblad algorithms planned"},
        "source_sha256": hashes,
    }
    (output / "logical-resources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for name in ("convergence.svg", "one-step-circuit.svg"):
        path = output / name
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    return {"output_dir": str(output), "validation_performed": performed,
            "selected_trace_distance": selected["trace_distance"] if selected else None,
            "Z_counts_per_trial": z_counts, "trials_proposal": shots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=0.8)
    parser.add_argument("--hx", type=float, default=0.7)
    parser.add_argument("--hz", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--input", choices=INPUT_LABELS, default="plus_i")
    parser.add_argument("--grid", default="1,2,4,8,16,32,64")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lindblad" if REPOSITORY else HERE / "generated-lindblad")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.time, args.rate, args.hx, args.hz,
                              args.steps, args.order, args.input,
                              tuple(int(r) for r in args.grid.split(",")), args.no_sim), indent=2))


if __name__ == "__main__":
    main()
