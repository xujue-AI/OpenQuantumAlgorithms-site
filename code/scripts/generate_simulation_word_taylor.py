#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate structured word-Taylor access, complete measured trials and evidence.

Default: two-spin H=.6ZZ+.35(IX+XI),t1,K3,r1/2/4,rounds0/1.
For bounded circuit-only scaling use --qubits 8 --segments 4 --rounds 0 --no-sim.
Standalone reproduction needs all five copied source modules and the locked
environment. No Pauli-word enumeration, matrix circuit gate or GPU is used.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPOSITORY = (ROOT/"src/openquantum/simulation/word_taylor.py").is_file()
sys.path.insert(0, str(ROOT/"src" if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector

if REPOSITORY:
    from openquantum.hamiltonian_simulation import spin_terms, observable_shots
    from openquantum.simulation import word_taylor as wt
else:
    from hamiltonian_simulation import spin_terms, observable_shots
    import word_taylor as wt


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def workload(check, rotations):
    shots = observable_shots(.03, .005)
    # Use the finite-H numerical lower acceptance, rather than just |00>.
    minimum = min(1.0, check["finite_H_numerical_minimum_acceptance"])
    cap = wt.postselection_attempts(minimum, .005/shots)
    # Hypothetical coherent accepted-branch error; no certificate is claimed.
    bias = .01
    accepted_operator_allowance = bias*math.sqrt(minimum)/(4+bias)
    conditional_lower_acceptance = (math.sqrt(minimum)-accepted_operator_allowance)**2
    return {
        "observable": "Z(q0)", "statistical_error": .03, "statistical_failure": .005,
        "accepted_shots": shots, "acceptance_failure_budget_all_shots": .005,
        "finite_H_numerical_acceptance_used": minimum,
        "full_attempt_cap_per_accepted_shot": cap,
        "worst_case_full_attempts": shots*cap,
        "worst_case_segment_calls": shots*cap*check["segments"],
        "scope": "conditional logical workload using finite-H numerical acceptance; sampling unexecuted",
        "sampling_performed": False, "acceptance_certified": False,
        "hypothetical_synthesis_proposal": {
            "observable_bias_allowance": bias,
            "accepted_branch_operator_allowance_full_attempt": accepted_operator_allowance,
            "per_segment_coherent_operator_allowance": accepted_operator_allowance/check["segments"],
            "per_counted_rotation_operator_allowance": accepted_operator_allowance/rotations if rotations else None,
            "conditional_acceptance_lower_bound_if_allowance_certified": conditional_lower_acceptance,
            "normalization_bound": "observable bias <=4 epsilon/(sqrt(p_min)-epsilon)",
            "scope": "sum coherent segment-unitary errors along the all-zero accepted instrument; reset/readout faults separate",
            "implemented": False, "certificate": False,
        },
        "physical_resource_estimate": None,
    }


def generate(output, n=2, J=.6, h=.35, total_time=1.0, degree=3,
             segments=(1, 2, 4), rounds=(0, 1), simulate=True, include_padding=True,
             include_scaling=True):
    if not segments or not rounds or len(segments)*len(rounds) > 16:
        raise ValueError("artifact sweep requires1 to16 native segment/round cases")
    output.mkdir(parents=True, exist_ok=True)
    terms = spin_terms(n, J, h)
    rows, results, ledgers = [], [], []
    managed_qasm_files = set()
    source_hashes = {}
    source_files = {
        "word_taylor.py": ROOT/"src/openquantum/simulation/word_taylor.py" if REPOSITORY else HERE/"word_taylor.py",
        "lcu.py": ROOT/"src/openquantum/simulation/lcu.py" if REPOSITORY else HERE/"lcu.py",
        "amplification.py": ROOT/"src/openquantum/simulation/amplification.py" if REPOSITORY else HERE/"amplification.py",
        "hamiltonian_simulation.py": ROOT/"src/openquantum/hamiltonian_simulation.py" if REPOSITORY else HERE/"hamiltonian_simulation.py",
        "generate_simulation_word_taylor.py": Path(__file__),
    }
    for name, source in source_files.items():
        raw = source.read_bytes()
        (output/name).write_bytes(raw)
        source_hashes[name] = sha256(raw)
    license_bytes = (ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt").read_bytes()
    (output/"LICENSE.txt").write_bytes(license_bytes)
    started = time.perf_counter()
    for count in segments:
        access = wt.build_word_taylor(terms, total_time/count, degree)
        cases = [(r, False) for r in rounds]
        if include_padding and 1 in rounds and access.normalization <= 2:
            cases.append((1, True))
        for amplification_rounds, padded in cases:
            source = wt.amplified_word_segment(access, amplification_rounds, padding_to_two=padded)
            coherent = wt.decompose_word_circuit(source)
            attempt = wt.build_word_attempt(access, count, amplification_rounds, padding_to_two=padded)
            compiled = wt.decompose_word_circuit(attempt)
            counts = dict(compiled.count_ops())
            rotations = sum(counts.get(g, 0) for g in ("rx", "ry", "rz"))
            file = f"degree{degree}-segments{count}-rounds{amplification_rounds}{'-normalization2' if padded else ''}.qasm"
            qasm = qasm3.dumps(compiled)
            (output/file).write_text(qasm, encoding="utf-8")
            managed_qasm_files.add(file)
            check = {"performed": False, "reason": "simulation disabled or exceeds12 total wires"}
            if simulate and source.num_qubits <= wt.MAX_STATEVECTOR_QUBITS:
                check = wt.compare_word_dynamics(access, count, amplification_rounds, padding_to_two=padded)
                for key in ("actual_source_block_error_vs_matrix_taylor", "actual_amplified_block_error_vs_general_polynomial",
                            "conditional_state_error_vs_independent_polynomial"):
                    if check[key] > 1e-10:
                        raise AssertionError(f"circuit/reference mismatch: {key}")
                rng = np.random.default_rng(7)
                full = rng.normal(size=2**source.num_qubits)+1j*rng.normal(size=2**source.num_qubits)
                full /= np.linalg.norm(full)
                primitive_error = float(np.linalg.norm(Statevector(full).evolve(source).data-Statevector(full).evolve(coherent).data))
                if primitive_error > 1e-10:
                    raise AssertionError("primitive lowering does not preserve source phases")
                check.update(performed=True, primitive_full_state_error_seed7=primitive_error)
                rows.append({"segments": count, "degree": degree, "rounds_per_segment": amplification_rounds,
                             "normalization2_padding": padded,
                             "state_vector_error": check["conditional_state_vector_error"],
                             "observable_error": check["observable_error"],
                             "acceptance_probability": check["full_attempt_acceptance_probability"],
                             "minimum_acceptance_numerical": check["finite_H_numerical_minimum_acceptance"],
                             "attempts_for_one_trial_99_percent": check["full_attempts_for_one_accepted_trial_99_percent"],
                             "logical_qubits": compiled.num_qubits, "cx_per_attempt": counts.get("cx", 0),
                             "rotations_per_attempt": rotations})
            record = {"segments": count, "rounds_per_segment": amplification_rounds,
                      "normalization2_padding": padded,
                      "degree": degree, "normalization_per_segment": access.normalization,
                      "degree_probabilities": access.degree_probabilities.tolist(),
                      "term_probabilities": access.term_probabilities.tolist(),
                      "simulation": check}
            results.append(record)
            ledger = {
                "segments": count, "rounds_per_segment": amplification_rounds,
                "normalization2_padding": padded,
                "source_access_metadata": access.circuit.metadata,
                "full_attempt_access_metadata": attempt.metadata,
                "system_qubits": n, "degree_qubits": access.n_degree,
                "term_qubits_per_slot": access.n_term, "term_registers": degree,
                "logical_qubits": compiled.num_qubits, "classical_outcome_bits": compiled.num_clbits,
                "primitive_segment_gate_counts": dict(coherent.count_ops()),
                "primitive_segment_depth": coherent.depth(),
                "primitive_full_attempt_gate_counts": counts, "primitive_full_attempt_depth": compiled.depth(),
                "arbitrary_rotations_full_attempt": rotations,
                "exact_T_Tdag_full_attempt": counts.get("t", 0)+counts.get("tdg", 0),
                "measurement_calls": counts.get("measure", 0), "reset_calls": counts.get("reset", 0),
                "qasm_file": file, "qasm_sha256": sha256(qasm.encode()),
                "source_coherent_global_phase_radians": float(source.global_phase),
                "compiled_coherent_global_phase_radians": float(coherent.global_phase),
                "compiled_measured_attempt_global_phase_radians": float(compiled.global_phase),
                "qasm_phase_scope": "pinned exporter omits overall phase; measured-trial probabilities preserved; source needed for coherent controlled reuse",
                "workload_proposal": workload(check, rotations) if check["performed"] else None,
                "physical_resource_estimate": None,
            }
            ledgers.append(ledger)
    scaling = []
    if include_scaling:
        for width in (1, 2, 3, 4, 6, 8):
            scaled = wt.build_word_taylor(spin_terms(width, J, h), total_time/4, degree)
            measured = wt.decompose_word_circuit(wt.build_word_attempt(scaled, 4, 0))
            scale = scaled.alpha*abs(total_time/4)
            remainder = max(0.0, math.exp(scale)-scaled.normalization)
            scaling.append({"n": width, "terms": len(scaled.terms), "degree": degree, "segments": 4,
                            "logical_qubits": measured.num_qubits,
                            "SELECT_predicates_per_segment": scaled.circuit.metadata["SELECT_slot_degree_term_predicates"],
                            "source_operations_per_segment": scaled.circuit.size(),
                            "primitive_operations_per_attempt": measured.size(),
                            "cx_per_attempt": measured.count_ops().get("cx", 0),
                            "rotation_count_per_attempt": sum(measured.count_ops().get(g, 0) for g in ("rx", "ry", "rz")),
                            "normalization_per_segment": scaled.normalization,
                            "numerical_evaluation_of_segment_Taylor_remainder_bound": remainder,
                            "numerical_evaluation_of_full_unnormalized_operator_bound": 4*remainder*(1+remainder)**3,
                            "simulation_performed": False})
        with (output/"scaling.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scaling[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(scaling)
    else:
        (output/"scaling.csv").unlink(missing_ok=True)
    elapsed = time.perf_counter()-started
    evidence = {
        "task": "structured Pauli-word Taylor dynamics with explicit degree and independent term registers",
        "parameters": {"n": n, "J": J, "h": h, "time": total_time, "hbar": 1,
                       "initial_basis": 0, "observable": "Z(q0)", "degree": degree,
                       "terms": terms, "segments": segments, "rounds": rounds},
        "records": results,
        "fixed_degree_step_scaling": scaling,
        "scaling_scope": "generated circuits at fixed degree and4 segments; no equal-accuracy performance claim; analytic tail expressions evaluated numerically",
        "source_revision_at_generation": git_revision(ROOT) if REPOSITORY else None,
        "source_hashes": source_hashes,
        "license_sha256": sha256(license_bytes), "repository_mode": REPOSITORY,
        "environment": {"python": platform.python_version(), "packages": {
            p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "scipy", "matplotlib")},
            "device": "CPU", "backend": "Qiskit ideal Statevector", "seed": 7},
        "generation_and_checks_seconds": elapsed,
        "word_enumeration_performed": False, "coefficient_aggregation_performed": False,
        "gate_implementation": "binary-tree controlled RY; degree/one-slot equality predicates; signed Pauli SELECT; full uncomputation",
        "formal_proof_integrated": False,
    }
    resource = {
        "resource_kind": "actual logical full measured-attempt ledger before synthesis and error correction",
        "physical_resource_estimate": None, "records": ledgers,
        "fixed_degree_step_scaling": scaling,
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7, "qubits_initially_zero": False,
                          "coherent_reuse_scope": "whole source unitary preserved on arbitrary register input; explicit reset operations remain in measured trials"},
        "included": ["basis input reset/preparation", "degree PREPARE", "K actual term PREPAREs",
                     "degree/slot/term pattern controls", "signed SELECT and degree phases", "all uncomputation",
                     "actual A and A inverse for OAA", "all-ancilla reflections", "every segment ancilla readout/reset",
                     "actual signal/sign zero-LCU padding where selected",
                     "final system Z readout", "complete retry and accepted-shot proposal"],
        "exclusions": ["classical coefficient loading/readout wall time", "coherent rotation synthesis and certificates",
                       "physical reset/readout/gate faults and timing", "routing, scheduling, error correction and factories",
                       "executed finite-shot experiment"],
        "error_contract": {"representation": "specified real Pauli model; IEEE coefficient and probability arithmetic",
                           "Hamiltonian_approximation": "independent matrix Taylor/exponential checks and conditional errors",
                           "OAA": "actual K p_r(K†K); input-dependent acceptance and state distortion measured",
                           "synthesis": "unimplemented; hypothetical accepted-branch allowance clearly conditional",
                           "statistical": "Hoeffding accepted Z means with separate capped acceptance union budget; unexecuted",
                           "physical_faults": "unmodeled and separate from every coherent error"},
        "source_scaling": {"degree_coefficients": "K+1", "term_coefficients": "L",
                           "register_qubits": "n+ceil(log2(K+1))+K ceil(log2 L)",
                           "SELECT_predicates": "L K(K+1)/2", "prepared_term_registers": "K",
                           "inactive_labels": "independent prepared inactive labels sum to1 after inverse PREPARE"},
        "limits": {"system_generation_qubits": wt.MAX_SYSTEM_QUBITS, "terms": wt.MAX_TERMS,
                   "degree": wt.MAX_DEGREE, "generation_total_wires": wt.MAX_GENERATOR_QUBITS,
                   "source_instructions": wt.MAX_SOURCE_OPERATIONS,
                   "SELECT_predicates": wt.MAX_SELECT_PREDICATES,
                   "primitive_export_operations": wt.MAX_PRIMITIVE_OPERATIONS,
                   "projected_statevector_total_wires": wt.MAX_STATEVECTOR_QUBITS,
                   "projected_reference_system_wires": wt.MAX_REFERENCE_SYSTEM_QUBITS,
                   "test_full_Operator_total_wires": 8,
                   "OAA_total_wires": 24, "OAA_rounds": 4, "attempt_segments": wt.MAX_ATTEMPT_SEGMENTS},
    }
    (output/"results.json").write_text(json.dumps(jsonable(evidence), indent=2)+"\n", encoding="utf-8")
    (output/"logical-resources.json").write_text(json.dumps(jsonable(resource), indent=2)+"\n", encoding="utf-8")
    if rows:
        with (output/"convergence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
        for amplification_rounds, padded in sorted({(r["rounds_per_segment"], r["normalization2_padding"]) for r in rows}):
            subset = [r for r in rows if r["rounds_per_segment"] == amplification_rounds and r["normalization2_padding"] == padded]
            label = f"{'s=2 padded' if padded else 'native'}; OAA={amplification_rounds}"
            axes[0].loglog([r["segments"] for r in subset], [r["state_vector_error"] for r in subset], "o-", label=label)
            axes[1].plot([r["segments"] for r in subset], [r["acceptance_probability"] for r in subset], "o-", label=label)
        axes[0].set(xlabel="segments", ylabel="conditional state-vector error")
        axes[1].set(xlabel="segments", ylabel="whole-attempt acceptance", ylim=(0, 1.05))
        for ax in axes:
            ax.set_xticks(list(segments), list(segments))
            ax.legend()
        fig.suptitle("Structured word-Taylor dynamics: actual circuit checks")
        fig.tight_layout()
        fig.savefig(output/"convergence.svg")
        plt.close(fig)
    else:
        for name in ("convergence.csv", "convergence.svg"):
            (output/name).unlink(missing_ok=True)
    # These are defined circuit gates whose bodies are lowered in the exports.
    access = wt.build_word_taylor(terms, total_time/max(segments), degree)
    if access.circuit.num_qubits <= 12:
        diagram = QuantumCircuit(access.circuit.num_qubits)
        ancillas = list(range(n, diagram.num_qubits))
        if ancillas:
            diagram.append(access.prepare.to_gate(label="PREPARE degree + K terms"), ancillas)
        diagram.append(access.select.to_gate(label="SELECT active word slots"), range(diagram.num_qubits))
        if ancillas:
            diagram.append(access.prepare.inverse().to_gate(label="UNPREPARE all words"), ancillas)
        fig = diagram.draw(output="mpl", fold=-1, scale=.8)
        fig.savefig(output/"access-circuit.svg", bbox_inches="tight")
        plt.close(fig)
    else:
        (output/"access-circuit.svg").unlink(missing_ok=True)
    # This generator owns only its named exports; unrelated output files remain.
    for previous in output.glob("degree*-segments*-rounds*.qasm"):
        if previous.name not in managed_qasm_files:
            previous.unlink()
    for name in ("access-circuit.svg", "convergence.svg"):
        svg = output/name
        if svg.is_file():
            svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--J", type=float, default=.6)
    parser.add_argument("--h", type=float, default=.35)
    parser.add_argument("--time", type=float, default=1)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--segments", default="1,2,4")
    parser.add_argument("--rounds", default="0,1")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-padding", action="store_true", help="omit the extra normalization2 zero-LCU examples")
    parser.add_argument("--no-scaling", action="store_true", help="omit the bounded n1/2/3/4/6/8 fixed-degree circuit-count sweep")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/word-taylor")
    args = parser.parse_args()
    result = generate(args.output_dir, args.qubits, args.J, args.h, args.time, args.degree,
                      tuple(int(r) for r in args.segments.split(",")),
                      tuple(int(r) for r in args.rounds.split(",")), not args.no_sim, not args.no_padding, not args.no_scaling)
    print(json.dumps({"output": str(args.output_dir), "cases": len(result["records"]),
                      "checks_seconds": result["generation_and_checks_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
