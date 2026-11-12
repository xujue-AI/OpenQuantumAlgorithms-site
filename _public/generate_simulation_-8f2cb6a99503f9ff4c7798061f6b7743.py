#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export a bounded noncommuting imaginary-time Taylor-LCU purification.

Standalone reproduction needs this script, thermal_filter.py, lcu.py and
LICENSE.txt together. This is explicit bounded Pauli aggregation/postselection,
not a scalable word oracle or an amplitude-amplified Gibbs algorithm.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/thermal_filter.py").is_file()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3

if REPOSITORY:
    from openquantum.simulation import thermal_filter as tf
else:
    import thermal_filter as tf


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def curve_row(check: dict, beta: float, degree: int) -> dict:
    return {"beta": beta, "degree": degree,
            "trace_distance": check["gibbs_trace_distance"],
            "polynomial_operator_error": check["polynomial_filter_operator_error"],
            "acceptance_probability": check["acceptance_probability"],
            "energy_error": abs(check["actual_observables"]["energy"]-check["exact_observables"]["energy"])}


def validate_circuit_check(check: dict) -> None:
    for key in ("actual_access_block_spectral_error", "conditional_purification_vector_error_vs_polynomial",
                "actual_reduced_state_error_vs_polynomial"):
        if check[key] > 1e-10:
            raise AssertionError(f"actual thermal circuit disagrees with its polynomial reference: {key}")
    if abs(check["acceptance_probability"]-check["independent_acceptance_probability"]) > 1e-10:
        raise AssertionError("thermal circuit acceptance disagrees with the independent polynomial")
    if abs(check["trace_real"]-1) > 1e-10 or check["minimum_eigenvalue"] < -1e-10:
        raise AssertionError("conditional thermal state failed trace or positivity checks")


def generate(output: Path, z: float = .6, x: float = .4, beta: float = 1.25,
             degree: int = 6, degrees=(0, 1, 2, 4, 6, 8), betas=(0, .5, 1.25, 2),
             simulate: bool = True) -> dict:
    terms = ((z, "Z"), (x, "X"))
    degrees, betas = tuple(degrees), tuple(betas)
    if not degrees or not betas or len(degrees)*len(betas) > 64:
        raise ValueError("validation sweep requires 1 to 64 beta/degree cases")
    output.mkdir(parents=True, exist_ok=True)
    if not simulate:
        for name in ("convergence.csv", "convergence.svg"):
            (output/name).unlink(missing_ok=True)
    started = time.perf_counter()
    access = tf.build_thermal_lcu(terms, beta, degree)
    access_construction_seconds = time.perf_counter()-started
    materialization_started = time.perf_counter()
    source = tf.build_thermal_filter(terms, beta, degree)
    coherent = tf.decompose_thermal_filter(source)
    trials = {axis: tf.decompose_thermal_filter(tf.build_thermal_trial(terms, beta, degree, measurement=axis))
              for axis in ("X", "Z")}
    materialization_seconds = time.perf_counter()-materialization_started
    if max(len(c.data) for c in (coherent, *trials.values())) > 250_000:
        raise ValueError("thermal-filter export exceeds the 250000-operation artifact limit")
    qasm_hashes = {}
    for name, circuit in {"thermal-filter.qasm": coherent,
                          "thermal-trial-X.qasm": trials["X"],
                          "thermal-trial-Z.qasm": trials["Z"]}.items():
        text = qasm3.dumps(circuit)
        (output/name).write_text(text)
        qasm_hashes[name] = sha256(text.encode())
    sources = {"thermal_filter.py": SOURCE/"openquantum/simulation/thermal_filter.py",
               "lcu.py": SOURCE/"openquantum/simulation/lcu.py",
               "generate_simulation_thermal_filter.py": HERE/"generate_simulation_thermal_filter.py"} if REPOSITORY else {
                   name: HERE/name for name in ("thermal_filter.py", "lcu.py", "generate_simulation_thermal_filter.py")}
    source_hashes = {}
    for name, path in sources.items():
        raw = path.read_bytes()
        (output/name).write_bytes(raw)
        source_hashes[name] = sha256(raw)
    (output/"LICENSE.txt").write_bytes((ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt").read_bytes())
    checks, rows = [], []
    default_check, synthesis, workload, conditional_workload = None, None, None, None
    if simulate:
        default_check = tf.thermal_filter_check(terms, beta, degree)
        validate_circuit_check(default_check)
        synthesis = tf.conditioned_synthesis_bounds(default_check["acceptance_probability"], 1e-4)
        workload = tf.thermal_energy_workload(default_check["acceptance_probability"], terms=terms)
        conditional_workload = tf.thermal_energy_workload(synthesis["perturbed_acceptance_lower_bound"], terms=terms)
        for value in betas:
            for order in degrees:
                check = tf.thermal_filter_check(terms, value, order)
                validate_circuit_check(check)
                checks.append(check)
                rows.append(curve_row(check, value, order))
    alpha = sum(abs(c) for c, _ in terms)
    naive_normalization = sum((beta*alpha/2)**k/math.factorial(k) for k in range(degree+1))
    per_axis = {}
    for axis, circuit in trials.items():
        counts = dict(circuit.count_ops())
        per_axis[axis] = {"primitive_gate_counts_per_full_attempt": counts,
                          "dependency_depth_per_full_attempt": circuit.depth(),
                          "logical_qubits": circuit.num_qubits, "classical_bits": circuit.num_clbits,
                          "reset_instructions_per_full_attempt": counts.get("reset", 0),
                          "measurement_instructions_per_full_attempt": counts.get("measure", 0),
                          "arbitrary_rotations_per_full_attempt": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
                          "rotation_count_scope": "RX/RY/RZ gate-class count, including any exact Clifford angles left by lowering; no synthesis certificate",
                          "classical_bit_mapping": "labels occupy bits 0..n_index-1; system outcomes occupy the following bits; accept only zero labels",
                          "schedule": "full initial reset, Bell preparation, access/uncompute, label measurement, unconditioned reference reset, basis change and system readout; failed attempts charged in full"}
    for label, proposal in (("ideal_workload", workload), ("hypothetical_synthesis_workload", conditional_workload)):
        if proposal is not None:
            for group in proposal["groups"]:
                axis = group["axis"]
                counts = per_axis[axis]["primitive_gate_counts_per_full_attempt"]
                per_axis[axis][label+"_worst_case_gate_counts"] = {gate: count*group["worst_case_complete_trials"] for gate, count in counts.items()}
    report = {"status": "checked postselected noncommuting thermal filter" if simulate else "generated circuits; validation and acceptance budget unperformed",
              "instance": {"system_qubits": 1, "reference_qubits": 1,
                           "Hamiltonian": "H/E0=z Z+x X", "E0": 1, "k_B": 1,
                           "z": z, "x": x, "beta_in_inverse_E0": beta, "degree": degree,
                           "initial_state": "Bell purification of I/2, prepared from zero by H and CX",
                           "filter": "T_K=sum(-beta H/2)^k/k!", "target_state": "exp(-beta H)/Tr exp(-beta H)"},
              "default_validation": default_check, "convergence_checks": checks,
              "validation_performed": simulate,
              "classical_preprocessing": {"aggregation_and_access_construction_wall_seconds": access_construction_seconds,
                                          "complete_circuit_materialization_and_lowering_wall_seconds": materialization_seconds,
                                          "aggregated_pauli_terms": jsonable(access.terms),
                                          "actual_lcu_normalization": access.normalization,
                                          "naive_word_taylor_normalization": naive_normalization,
                                          "complexity": "bounded Pauli coefficient multiplication/aggregation may be exponential in system size; actual indexed access is materialized",
                                          "quantum_matrix_or_eigenvector_loading": False},
              "coherent_synthesis_proposal": synthesis,
              "coherent_operator_bound_scope": "Bell preparation and LCU unitary; reset/readout fault probabilities are separate",
              "accepted_shot_workload_proposal": workload,
              "hypothetical_synthesis_accepted_shot_workload_proposal": conditional_workload,
              "sampling_performed": False, "amplitude_amplification": "not implemented",
              "physical_resource_estimate": None,
              "scope": "one-qubit artifact CLI; reusable bounded Pauli filter; no scalable word oracle, certified synthesis or physical execution"}
    ledger = {"resource_kind": "logical full-attempt counts before rotation synthesis and error correction",
              "physical_resource_estimate": None,
              "source_gate_counts_coherent_preparation": dict(source.count_ops()),
              "primitive_gate_counts_coherent_preparation": dict(coherent.count_ops()),
              "primitive_coherent_preparation_depth": coherent.depth(),
              "system_qubits": access.n_system, "reference_qubits": access.n_system,
              "label_qubits": access.n_index, "logical_qubits": coherent.num_qubits,
              "access_calls_per_attempt": {"PREPARE": 1, "SELECT": 1, "PREPARE_inverse": 1},
              "per_measurement_axis": per_axis,
              "output_contracts": {"coherent_filter": "conditional pre-readout system-reference purification upon label0",
                                   "measured_trial": "accept label0 and retain measured system outcome; reference reset without conditioning",
                                   "reference": "reference is discarded/reset without conditioning; no eigenstate or diagonal Boltzmann input loading"},
              "workload_proposal": workload,
              "hypothetical_synthesis_workload_proposal": conditional_workload,
              "error_contract": {"Hamiltonian_representation": "stated real Pauli coefficients; floating coefficient arithmetic",
                                 "polynomial_approximation": "actual block and conditional state checked separately against matrix Taylor and normalized exponential",
                                 "coherent_synthesis": synthesis if simulate else "pending without acceptance lower bound",
                                 "coherent_operator_bound_scope": "coherent Bell+LCU unitary; no operator norm assigned to irreversible reset/readout",
                                 "statistical": "separate accepted X and Z means; noncommuting energy sum is not one energy-per-bitstring sample",
                                 "physical_faults": "pending; distinct from polynomial/synthesis norm error and sampling confidence"},
              "prominent_exclusions": ["coherent rotation synthesis and certificates", "physical gate/reset/discard/readout timing",
                                       "error correction and factory costs", "executed accepted-shot experiment", "OAA and scalable thermal filtering"],
              "scale_limits": {"artifact_cli_system_qubits": 1, "reusable_generation_system_qubits": 4,
                               "maximum_degree": 12, "maximum_aggregated_pauli_terms": 256,
                               "beta_times_alpha_over_two_maximum": 10, "statevector_total_qubits": 12,
                               "decomposed_artifact_operation_limit": 250_000, "validation_cases": 64},
              "provenance": {"source_revision_at_generation": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() if REPOSITORY else None,
                             "source_hashes": source_hashes, "qasm_hashes": qasm_hashes,
                             "revision_note": "source hashes identify working-tree files if generation precedes their commit",
                             "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "scipy", "matplotlib")},
                             "device": "CPU", "randomness": "none; exact ideal checks", "transpiler_seed": 7,
                             "optimization_level": 1}}
    (output/"results.json").write_text(json.dumps(jsonable(report), indent=2, allow_nan=False)+"\n")
    (output/"logical-resources.json").write_text(json.dumps(jsonable(ledger), indent=2, allow_nan=False)+"\n")
    if rows:
        with (output/"convergence.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
        for value in betas:
            group = [row for row in rows if row["beta"] == value]
            axes[0].semilogy([r["degree"] for r in group], [max(r["trace_distance"], 1e-16) for r in group], "o-", label=f"beta={value:g}")
            axes[1].semilogy([r["degree"] for r in group], [max(r["energy_error"], 1e-16) for r in group], "o-")
            axes[2].plot([r["degree"] for r in group], [r["acceptance_probability"] for r in group], "o-")
        axes[0].set(xlabel="Taylor degree", ylabel="Gibbs trace distance (floor 1e-16)")
        axes[1].set(xlabel="Taylor degree", ylabel="Energy bias (floor 1e-16)")
        axes[2].set(xlabel="Taylor degree", ylabel="Ideal full-attempt acceptance")
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output/"convergence.svg")
        plt.close(fig)
    fig = source.draw(output="mpl", fold=30, scale=.85)
    fig.savefig(output/"circuit.svg", bbox_inches="tight")
    plt.close(fig)
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    return {"output_dir": str(output), "default_validation": jsonable(default_check),
            "primitive_trial_counts": {axis: item["primitive_gate_counts_per_full_attempt"] for axis, item in per_axis.items()},
            "checks": len(checks), "sampling_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z", type=float, default=.6)
    parser.add_argument("--x", type=float, default=.4)
    parser.add_argument("--beta", type=float, default=1.25)
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--degrees", default="0,1,2,4,6,8")
    parser.add_argument("--betas", default="0,.5,1.25,2")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/thermal-filter" if REPOSITORY else HERE/"generated-thermal-filter")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.z, args.x, args.beta, args.degree,
                              tuple(int(x) for x in args.degrees.split(",")),
                              tuple(float(x) for x in args.betas.split(",")), not args.no_sim), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
