#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export bounded diagonal Gibbs purification, retained-state and sample circuits.

Standalone reproduction: put this script, gibbs.py and LICENSE.txt together.
Explicit coefficient loading grows exponentially; this is not scalable filtering.
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
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/gibbs.py").is_file()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import DensityMatrix, partial_trace

if REPOSITORY:
    from openquantum.simulation import gibbs
else:
    import gibbs


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path, n: int = 2, J: float = 0.6, fields=(0.35, -0.2), beta: float = 1.25,
             betas=(0, .25, 1.25, 5), simulate: bool = True) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    energies = gibbs.diagonal_ising_energies(n, J, fields)
    probabilities = gibbs.gibbs_probabilities(energies, beta)
    energy_time = time.perf_counter()-started
    if beta == 0:
        log_partition = float(np.log(len(energies)))
    else:
        shifted = -beta*(energies-energies.min())
        log_partition = float(-beta*energies.min()+np.log(np.exp(shifted).sum()))
    if not np.isfinite(log_partition):
        raise ValueError("partition-function reporting exceeds floating log-domain range")
    if beta == 0:
        temperature = None
        temperature_status = "infinite temperature at beta zero"
    elif beta < 1/np.finfo(float).max:
        temperature = None
        temperature_status = "inverse beta exceeds floating reporting range; beta defines the temperature"
    else:
        temperature = float(1/beta)
        if np.isfinite(temperature):
            temperature_status = "finite temperature in the stated energy unit with k_B=1"
        else:
            temperature = None
            temperature_status = "inverse beta exceeds floating reporting range; beta defines the temperature"
    prep_started = time.perf_counter()
    purification = gibbs.diagonal_gibbs_purification(energies, beta)
    pure = gibbs.decompose_gibbs(purification)
    static_sample = gibbs.decompose_gibbs(gibbs.diagonal_gibbs_purification(energies, beta, measure=True))
    mixed = gibbs.decompose_gibbs(gibbs.build_gibbs_mixed_preparation(energies, beta))
    trial = gibbs.decompose_gibbs(gibbs.build_gibbs_trial(energies, beta))
    materialization_time = time.perf_counter()-prep_started
    if max(len(c.data) for c in (pure, static_sample, mixed, trial)) > 250_000:
        raise ValueError("decomposed export exceeds the 250000-operation artifact limit")
    qasm_hashes = {}
    for name, circuit in {"purification.qasm": pure, "thermal-sample.qasm": static_sample,
                          "retained-mixed-state.qasm": mixed, "thermal-trial.qasm": trial}.items():
        text = qasm3.dumps(circuit)
        (output/name).write_text(text)
        qasm_hashes[name] = sha256(text.encode())
    sources = {"gibbs.py": SOURCE/"openquantum/simulation/gibbs.py",
               "generate_simulation_gibbs.py": HERE/"generate_simulation_gibbs.py"} if REPOSITORY else {
                   name: HERE/name for name in ("gibbs.py", "generate_simulation_gibbs.py")}
    source_hashes = {}
    for name, source in sources.items():
        raw = source.read_bytes()
        (output/name).write_bytes(raw)
        source_hashes[name] = sha256(raw)
    (output/"LICENSE.txt").write_bytes((ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt").read_bytes())
    performed = simulate and n <= gibbs.MAX_DENSE_SYSTEM_QUBITS
    checks, rows = [], []
    if performed:
        for value in betas:
            check = gibbs.gibbs_check(energies, value)
            if check["trace_distance"] > 1e-11 or abs(check["trace_real"]-1) > 1e-11 or check["minimum_eigenvalue"] < -1e-11:
                raise AssertionError("actual purification failed the Gibbs/trace/positivity checks")
            checks.append(check)
            rows.append({"beta": value, "trace_distance": check["trace_distance"],
                         "min_eigenvalue": check["minimum_eigenvalue"], "trace": check["trace_real"],
                         **check["actual_observables"]})
        density = DensityMatrix.from_instruction(gibbs.build_gibbs_mixed_preparation(energies, beta))
        reference = gibbs.exact_diagonal_gibbs(energies, beta)
        actual = partial_trace(density, list(range(n, 2*n))).data
        mixed_error = float(np.max(np.abs(actual-reference)))
        if mixed_error > 1e-11:
            raise AssertionError("explicit irreversible reference-reset schedule did not retain Gibbs")
    else:
        mixed_error = None
    shots_z = gibbs.bounded_mean_shots(.03, .005)
    with np.errstate(over="ignore"):
        energy_width = float(energies.max()-energies.min())
    if energy_width == 0:
        shots_energy = 0
        energy_sampling_status = "no sampling needed: every known energy outcome is identical"
    elif not np.isfinite(energy_width):
        shots_energy = None
        energy_sampling_status = "unavailable: energy outcome width exceeds floating budget domain"
    else:
        try:
            shots_energy = gibbs.bounded_mean_shots(.03, .005, energies.min(), energies.max())
            energy_sampling_status = "proposed independent iid scalar-mean workload"
        except ValueError:
            shots_energy = None
            energy_sampling_status = "unavailable: energy-mean shot count exceeds floating budget domain"
    counts = dict(trial.count_ops())
    ledger = {
        "status": "checked bounded diagonal purification and retained/sample schedules" if performed else "generated circuits; dense validation unperformed",
        "instance": {"system_qubits": n, "reference_qubits": n, "J": J, "fields_q0_first": list(fields),
                     "boundary": "open", "Hamiltonian": "J sum Z_i Z_(i+1) + sum h_i Z_i",
                     "energy_units": "stated arbitrary energy unit", "k_B": 1, "beta": beta,
                     "temperature": temperature, "temperature_status": temperature_status,
                     "energies_bit_order_qn_minus_1_to_q0": energies.tolist(), "probabilities": probabilities.tolist(),
                     "log_partition_function": log_partition, "initial_state": "all system and reference wires zero"},
        "classical_preprocessing": {"energies_and_boltzmann_wall_seconds": energy_time,
                                    "circuit_materialization_and_lowering_wall_seconds": materialization_time,
                                    "energy_entries": len(energies), "complexity": "O(n*2**n) energy enumeration and prefix-mass/angle loading; O(2**n) normalization/storage",
                                    "normalization": "Boltzmann weights normalized classically using the exact stated diagonal energies",
                                    "no_diagonalization": True,
                                    "quantum_advantage_claim": False},
        "output_contracts": {"purification": "unmeasured pure system-reference state; all-zero input assumed",
                             "retained_mixed_state": "initial resets and irreversible unconditioned reference resets; system remains unmeasured",
                             "thermal_sample": "system Z readout; reference discarded without conditioning",
                             "thermal_trial": "all 2n initial resets, purification, system Z readout, reference discard; one measured trial"},
        "ideal_success_probability": 1.0, "postselection": "none", "amplification": "not required",
        "input_overlap": "not required for this exhaustive coefficient-loading construction",
        "logical_qubits": trial.num_qubits,
        "source_purification_counts": dict(purification.count_ops()),
        "primitive_purification_counts": dict(pure.count_ops()), "primitive_purification_depth": pure.depth(),
        "primitive_retained_mixed_state_counts": dict(mixed.count_ops()),
        "primitive_measured_trial_counts": counts, "primitive_measured_trial_depth": trial.depth(),
        "initial_resets_per_measured_trial": 2*n, "reference_wires_discarded_per_measured_trial": n,
        "system_measurements_per_measured_trial": n,
        "arbitrary_rotations_per_measured_trial": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
        "synthesis_complete": False,
        "measurement_workload_proposals": {
            "Z0_mean": {"statistical_absolute_error": .03, "statistical_failure_probability": .005,
                        "independent_measured_trials": shots_z, "outcome_range": [-1, 1],
                        "coherent_synthesis_operator_allowance_per_trial_proposal": .005,
                        "observable_synthesis_bias_allowance_proposal": .01,
                        "physical_failure_probability_all_trials_proposal": .005,
                        "physical_failure_probability_per_trial_proposal": .005/shots_z},
            "diagonal_energy_mean": {"statistical_absolute_error": .03, "statistical_failure_probability": .005,
                                     "independent_measured_trials": shots_energy, "outcome_range": [float(energies.min()), float(energies.max())],
                                     "status": energy_sampling_status,
                                     "coherent_synthesis_operator_allowance_per_trial_proposal": .005,
                                     "observable_synthesis_bias_allowance_proposal": energy_width*.005 if np.isfinite(energy_width) else None,
                                     "physical_failure_probability_all_trials_proposal": .005,
                                     "physical_failure_probability_per_trial_proposal": .005/shots_energy if shots_energy else None},
            "scope": "each proposal covers one fixed iid scalar mean separately; joint simultaneous means require confidence reallocation",
            "sampling_performed": False,
        },
        "error_contract": {
            "Hamiltonian_model": "exact stated diagonal coefficients; floating arithmetic and energy/weight loading",
            "ideal_preparation": "tested actual reduced state vs normalized matrix exponential; no asymptotic filtering approximation",
            "coherent_synthesis": "pending; no Clifford+T sequence or certificate; not a stochastic fault probability",
            "statistical": "separate unexecuted Hoeffding budgets for individual means",
            "physical_faults": "pending; no physical resource or confidence estimate",
        },
        "physical_resource_estimate": None,
        "prominent_exclusions": ["coherent rotation synthesis", "physical reset/discard/readout time", "routing and physical scheduling",
                                  "physical error correction/factories", "measured thermal samples/observable experiment", "scalable Gibbs filtering or mixing"],
        "scope_limits": {"explicit_energy_list_system_qubits": gibbs.MAX_PREPARATION_QUBITS,
                         "dense_validation_system_qubits": gibbs.MAX_DENSE_SYSTEM_QUBITS,
                         "beta_times_energy_span_maximum": gibbs.MAX_SCALED_ENERGY_RANGE,
                         "decomposed_artifact_operations": 250_000,
                         "scalable_filtering": "planned; not implemented by explicit exponential coefficient loading"},
        "validation": {"performed": performed, "checks": checks,
                       "actual_reference_reset_retained_state_max_entry_error": mixed_error,
                       "method": "Qiskit actual circuit Statevector/DensityMatrix; partial trace; SciPy diagonal matrix exponential"},
        "provenance": {"source_revision_at_generation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if REPOSITORY else None,
                       "source_hashes": source_hashes, "qasm_hashes": qasm_hashes,
                       "revision_note": "source hashes identify working-tree files if generation precedes their commit",
                       "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "scipy", "matplotlib")},
                       "device": "CPU", "randomness": "none; exact ideal checks", "transpiler_seed": 7,
                       "count_basis": gibbs.ROTATION_BASIS, "optimization_level": 1},
    }
    (output/"logical-resources.json").write_text(json.dumps(ledger, indent=2, allow_nan=False)+"\n")
    if rows:
        with (output/"thermal-checks.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        for key in ("Z0", "Z1", "Z0Z1"):
            if key in rows[0]:
                axes[0].plot([r["beta"] for r in rows], [r[key] for r in rows], "o-", label=key)
        axes[0].set(xlabel="Inverse temperature beta", ylabel="Thermal mean")
        axes[0].legend()
        axes[1].plot([r["beta"] for r in rows], [r["energy"] for r in rows], "o-", color="#be6b35")
        axes[1].set(xlabel="Inverse temperature beta", ylabel="Mean energy")
        fig.tight_layout()
        fig.savefig(output/"thermal-observables.svg")
        plt.close(fig)
    if n <= 2:
        fig = purification.draw(output="mpl", fold=30, scale=.9)
        fig.savefig(output/"purification-circuit.svg", bbox_inches="tight")
        plt.close(fig)
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    return {"output_dir": str(output), "system_qubits": n, "beta": beta,
            "checks": len(checks), "primitive_trial_counts": counts,
            "retained_state_error": mixed_error, "ideal_success_probability": 1.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--J", type=float, default=.6)
    parser.add_argument("--fields", default=".35,-.2")
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--betas", default="0,.25,1.25,5")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/gibbs" if REPOSITORY else HERE/"generated-gibbs")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.qubits, args.J,
                              tuple(float(x) for x in args.fields.split(",")), args.beta,
                              tuple(float(x) for x in args.betas.split(",")), not args.no_sim), indent=2))


if __name__ == "__main__":
    main()
