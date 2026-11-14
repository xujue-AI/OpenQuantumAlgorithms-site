#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit Neumann-LCU inverse circuits, uniform bounds and costs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
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

import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from openquantum.linear_systems import accepted_observable_workload, decompose
from openquantum.linear_inverse_lcu import (
    build_neumann_lcu, compare_inverse, default_preparation, degree_for_pauli_bias,
    measured_inverse_trial, neumann_coefficients, prepared_inverse,
    structured_inverse_terms,
)


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bound_checks(result):
    """Floating checks of reported analytic bounds, not interval certificates."""
    return {
        "operator_error_within_bound_with_1e_12_numerical_guard":
            result["scaled_inverse_operator_error"] <= result["analytic_operator_error_bound"] + 1e-12,
        "state_error_within_bound_with_1e_12_numerical_guard":
            result["conditional_state_trace_distance_to_classical_solution"] <=
            result["analytic_conditional_trace_distance_bound"] + 1e-12,
        "observable_error_within_bound_with_2e_12_numerical_guard":
            result["z_absolute_bias"] <= result["analytic_conditional_Pauli_bias_bound"] + 2e-12,
        "amplification_prediction_within_1e_12":
            abs(result["success_probability"] - result["amplification_prediction"]) <= 1e-12,
    }


def _circuit_row(n, gap, degree, *, rounds=0, simulate=True):
    terms = structured_inverse_terms(n, gap)
    model = neumann_coefficients(terms, degree)
    prep = default_preparation(n)
    started = time.perf_counter()
    circuit = measured_inverse_trial(terms, degree, prep, rounds=rounds)
    compiled = decompose(circuit)
    elapsed = time.perf_counter() - started
    counts = dict(compiled.count_ops())
    validation = compare_inverse(terms, degree, prep, rounds=rounds) if simulate else None
    metadata = model.as_metadata()
    return {
        "n": n, "matrix_dimension": 1 << n, "gap_promised": float(model.delta),
        "condition_upper_promised": float(1 / model.delta), "degree": degree,
        "rounds": rounds, "aggregate_terms": len(model.coefficients),
        "logical_qubits": compiled.num_qubits,
        "primitive_operations": len(compiled.data) - counts.get("barrier", 0),
        "cx": counts.get("cx", 0),
        "continuous_rotations": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
        "measurements": counts.get("measure", 0),
        "dependency_depth": compiled.depth(),
        "classical_circuit_generation_and_lowering_seconds": elapsed,
        "analytic_operator_error_bound": metadata["scaled_inverse_operator_error_bound"],
        "analytic_Pauli_bias_bound": metadata["conditional_Pauli_bias_bound"],
        "simulation_performed": validation is not None,
        "actual_condition": None if validation is None else validation["actual_condition_number"],
        "success_probability": None if validation is None else validation["success_probability"],
        "solution_trace_distance": None if validation is None else validation["conditional_state_trace_distance_to_classical_solution"],
        "z_absolute_bias": None if validation is None else validation["z_absolute_bias"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--delta", default="1/2", help="Exact decimal or rational promised normalized gap")
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--pauli-bias", type=float,
                        help="Select degree from the uniform Pauli-bias allowance instead")
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--no-estimate", action="store_true")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "assets/generated/linear-systems/neumann-lcu")
    args = parser.parse_args()
    if not 2 <= args.qubits <= 24:
        parser.error("--qubits must be between 2 and 24")
    if not 0 <= args.rounds <= 4:
        parser.error("--rounds must be between 0 and 4")
    if not 0 <= args.degree <= 64:
        parser.error("--degree must be between 0 and 64")
    terms = structured_inverse_terms(args.qubits, args.delta)
    degree = args.degree if args.pauli_bias is None else degree_for_pauli_bias(args.delta, args.pauli_bias)
    started = time.perf_counter()
    model = neumann_coefficients(terms, degree)
    coefficient_preprocessing_seconds = time.perf_counter() - started
    prep = default_preparation(args.qubits)
    started = time.perf_counter()
    unitary = prepared_inverse(terms, degree, prep, rounds=args.rounds)
    source_measured = measured_inverse_trial(terms, degree, prep, rounds=args.rounds)
    measured = decompose(source_measured)
    generation_seconds = time.perf_counter() - started
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    qasm = qasm3.dumps(measured)
    (output / "measured-trial.qasm").write_text(qasm)
    source = (ROOT / "src/openquantum/linear_inverse_lcu.py").read_bytes()
    (output / "linear_inverse_lcu.py").write_bytes(source)
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    metadata = model.as_metadata()
    counts = dict(measured.count_ops())
    validation = {"performed": False, "reason": "--no-sim or six-data/twelve-total reference bound"}
    if not args.no_sim and args.qubits <= 6 and measured.num_qubits <= 12:
        validation = compare_inverse(terms, degree, prep, rounds=args.rounds)
        validation["analytic_bound_checks"] = _bound_checks(validation)
        if not all(validation["analytic_bound_checks"].values()):
            raise RuntimeError("A bounded numerical inverse check failed its analytic contract")
    success = None
    success_kind = "unavailable"
    if validation["performed"]:
        success = validation["success_probability"]
        success_kind = "numerically known prepared-input acceptance in the ideal gate model"
    elif args.rounds == 0:
        success = metadata["success_probability_lower_bound"]
        success_kind = "analytic unamplified success lower bound under exact rational/ideal gates"
    workload = None if success is None else accepted_observable_workload(min(1.0, success))
    ledger = {
        **metadata,
        "schema_version": 1,
        "matrix_dimension": 1 << args.qubits,
        "input": "RY(0.63) then RZ(0.41) on q0, RY(0.37) on q(n-1), clean zeros initially",
        "input_preparation_counts": dict(prep.count_ops()),
        "postselection": "all Pauli labels zero, data values unrestricted",
        "observable": "one fixed conditional Z(q0) mean",
        "amplification_rounds": args.rounds,
        "amplification_kind": "known prepared input; full initial-zero reflection, not oblivious inverse amplification",
        "full_source_calls": 2 * args.rounds + 1,
        "forward_source_calls": args.rounds + 1,
        "inverse_source_calls": args.rounds,
        "good_reflections": args.rounds,
        "initial_zero_reflections": args.rounds,
        "source_gate_counts": dict(unitary.count_ops()),
        "source_dependency_depth": unitary.depth(),
        "logical_qubits": measured.num_qubits,
        "primitive_gate_counts_per_complete_attempt": counts,
        "primitive_dependency_depth": measured.depth(),
        "arbitrary_rotations_per_attempt": sum(counts.get(g, 0) for g in ("rx", "ry", "rz")),
        "classical_work_measured": {
            "exact_coefficient_preprocessing_seconds": coefficient_preprocessing_seconds,
            "source_generation_and_primitive_lowering_seconds": generation_seconds,
            "scope": "one CPU run; generation recomputes coefficients internally, so these are separately measured activities rather than disjoint total-time components",
            "excluded_from_physical_runtime": True,
        },
        "validation": validation,
        "retry_acceptance_kind": success_kind,
        "retry_and_sampling_proposal": workload,
        "workload_status": "unexecuted ideal capped workload proposal; no hardware confidence certificate",
        "amplified_uniform_retry_lower_bound": "unavailable; integer amplification is nonmonotone in input acceptance",
        "initialization": "fresh clean logical zero inputs assumed per attempt; reset timing and faults excluded",
        "included_in_attempt": [
            "actual complex input preparation", "explicit coefficient PREPARE",
            "signed label-controlled Pauli SELECT", "PREPARE inverse",
            "all source/inverse calls and both reflection types if selected",
            "all final data/label measurements", "every failed full attempt",
        ],
        "excluded": [
            "clean initialization timing/faults", "concrete coherent Clifford+T synthesis certificate",
            "classical readout/reference/postprocessing timing",
            "general sparse matrix or arbitrary-vector loading",
        ],
        "synthesis_and_conditioning": "The analytic bound covers the exact rational polynomial and ideal angle gates. Numerical coefficient loading and QDK mean-fit synthesis do not supply an operator-norm certificate or physical/postselected confidence guarantee.",
        "qasm_export_policy": {
            "scope": "complete measured trials, not phase-preserving coherently controlled oracle",
            "compiled_global_phase_radians": float(measured.global_phase),
            "pinned_exporter": "may omit overall global phase; measured probabilities unchanged",
        },
        "physical_estimate": None,
        "provenance": {
            "source_revision_at_generation": git_revision(ROOT),
            "implementation_sha256": hashlib.sha256(source).hexdigest(),
            "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
            "source_note": "commit plus exact module hash identifies working-tree code; generation can precede commit",
            "python": platform.python_version(),
            "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "scipy", "matplotlib")},
            "seed": 7, "device": "CPU",
            "command": "python scripts/generate_linear_inverse_lcu.py " + " ".join(sys.argv[1:]),
            "shared_access_sources": {
                name: hashlib.sha256((ROOT / "src/openquantum" / name).read_bytes()).hexdigest()
                for name in ("simulation/lcu.py", "linear_systems_qsvt.py")
            },
        },
    }
    for key in (
        "gate_loading_normalization", "coefficient_conversion_l1_error_bound",
        "coefficient_conversion_l1_error_bound_exact",
        "conversion_adjusted_trace_bound_exact_angle_model", "conversion_bound_scope",
    ):
        ledger[key] = unitary.metadata[key]
    if workload is not None:
        ledger["capped_primitive_gate_counts_proposal"] = {
            name: count * workload["total_capped_attempts"] for name, count in counts.items()
        }

    if not args.no_scaling:
        width_rows = [_circuit_row(n, "1/2", 12, simulate=n <= 4 and not args.no_sim)
                      for n in (2, 3, 4, 6, 8, 12, 24)]
        degree_rows = [_circuit_row(2, "1/2", k, simulate=not args.no_sim)
                       for k in (0, 1, 2, 4, 8, 12, 16, 24, 32)]
        condition_rows = [
            _circuit_row(2, gap, degree_for_pauli_bias(gap, .001), simulate=not args.no_sim)
            for gap in ("3/4", "1/2", "1/4", "1/8")
        ]
        amplification_rows = [_circuit_row(2, "1/2", 12, rounds=r, simulate=not args.no_sim)
                              for r in range(5)]
        for filename, rows in (
            ("width-scaling.csv", width_rows), ("degree-convergence.csv", degree_rows),
            ("condition-scaling.csv", condition_rows), ("amplification.csv", amplification_rows),
        ):
            _write_csv(output / filename, rows)
        ledger["scaling"] = {
            "width": "actual generated circuits for the fixed three-word algebra; only marked rows simulated; structured access can be classically tractable",
            "condition": "promised gap changed while actual reference eigenvalues and chosen degree are recorded",
            "degree": "fixed matrix and complex input, actual errors versus analytic uniform bound",
            "physical_scaling_computed": False,
        }
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        numerical_degree_rows = [r for r in degree_rows if r["simulation_performed"]]
        if numerical_degree_rows:
            axes[0].semilogy([r["degree"] for r in numerical_degree_rows],
                             [r["z_absolute_bias"] for r in numerical_degree_rows], "o-", label="Actual Z bias")
        axes[0].semilogy([r["degree"] for r in degree_rows],
                         [r["analytic_Pauli_bias_bound"] for r in degree_rows], "--", label="Uniform bound")
        axes[0].set(xlabel="Neumann degree K", ylabel="Pauli mean error")
        axes[0].legend(fontsize=8)
        axes[1].plot([r["n"] for r in width_rows],
                     [r["primitive_operations"] for r in width_rows], "o-")
        axes[1].set(xlabel="Data qubits n", ylabel="Generated primitive operations")
        numerical_amplification_rows = [r for r in amplification_rows if r["simulation_performed"]]
        if numerical_amplification_rows:
            axes[2].plot([r["rounds"] for r in numerical_amplification_rows],
                         [r["success_probability"] for r in numerical_amplification_rows], "o-")
        else:
            axes[2].text(.5, .5, "Numerical simulation unperformed", transform=axes[2].transAxes,
                         ha="center", va="center", fontsize=9)
        axes[2].set(xlabel="Known-input amplification rounds", ylabel="Numerical ideal acceptance", ylim=(0, 1))
        for ax in axes:
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(output / "convergence-and-cost.svg")
        plt.close(fig)
        amplified = decompose(measured_inverse_trial(
            structured_inverse_terms(2, "1/2"), 12, default_preparation(2), rounds=1))
        (output / "amplified-trial.qasm").write_text(qasm3.dumps(amplified))
        ledger["additional_amplified_export"] = {
            "rounds": 1, "file": "amplified-trial.qasm",
            "compiled_global_phase_radians": float(amplified.global_phase),
            "scope": "measured probabilities only, source includes known preparation and both reflections",
        }

    grid = np.linspace(float(model.delta), 1.0, 401)
    polynomial = float(model.delta) * sum((1 - grid)**k for k in range(degree + 1))
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    ax.plot(grid, float(model.delta) / grid, "--", label="Requested scaled reciprocal")
    ax.plot(grid, polynomial, label=f"Neumann degree {degree}")
    ax.set(xlabel="Normalized eigenvalue", ylabel="Unnormalized filter amplitude")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output / "reciprocal.svg")
    plt.close(fig)
    if unitary.num_qubits <= 6 and len(unitary.data) <= 250:
        fig = unitary.draw(output="mpl", fold=24, scale=.6)
        fig.savefig(output / "circuit.svg", bbox_inches="tight")
        plt.close(fig)
    if not args.no_estimate:
        if workload is None:
            ledger["physical_estimate"] = {
                "status": "unavailable; no acceptance bound supplied for unsimulated amplified input"
            }
        elif measured.num_qubits > 8 or len(measured.data) > 4000:
            ledger["physical_estimate"] = {"status": "unavailable; eight-wire/4000-operation physical query cap"}
        else:
            from openquantum.resource_estimation import CircuitResourceBudget, estimate_circuit_resources
            budget = CircuitResourceBudget(1e-6, .005, workload["total_capped_attempts"])
            report = estimate_circuit_resources(
                source_measured, budget=budget,
                cache_dir=Path("/tmp/openquantum-linear-inverse-qdk-cache"))
            report.pop("qasm3")
            report["postselection_contract"] = {
                "attempts": "all complete capped attempts, including known-input reflections if selected",
                "acceptance_kind": success_kind,
                "initialization": "fresh clean logical zeros assumed; reset timing/faults excluded",
                "confidence": "conditional cost projection only; no certified synthesis or physical output-accuracy guarantee",
            }
            (output / "surface-code.json").write_text(json.dumps(report, indent=2) + "\n")
            ledger["physical_estimate"] = {
                "status": report["resource_estimate_kind"], "report": "surface-code.json",
                "pareto_points": len(report["pareto_points"]),
                "profile": report["physical_profile"], "budget": report["budget"],
            }
    (output / "logical-resources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(json.dumps({
        "output_dir": str(output), "degree": degree, "validation": validation,
        "physical_estimate": ledger["physical_estimate"],
    }, indent=2))


if __name__ == "__main__":
    main()
