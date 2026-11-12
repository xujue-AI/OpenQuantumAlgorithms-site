#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate bounded, explicit postselected-Taylor LCU circuits and checks.

The parameterized module supports up to 32 system wires and 256 Pauli terms;
this small artifact CLI limits systems to 5 wires and total decomposed exports
to 250000 operations per attempt. No package installation or GPU is used.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPOSITORY = (ROOT / "src/openquantum/hamiltonian_simulation.py").exists()
sys.path.insert(0, str(ROOT / "src" if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3, transpile
from qiskit.quantum_info import Statevector

if REPOSITORY:
    from openquantum.hamiltonian_simulation import (
        ROTATION_BASIS, exact_evolution, observable_shots, pauli_matrix, spin_terms,
    )
    from openquantum.simulation.lcu import (
        MAX_STATEVECTOR_QUBITS, build_lcu_attempt, build_pauli_lcu,
        build_taylor_lcu, postselected_lcu_evolution, postselection_attempts,
    )
else:  # Download this script and both source files together.
    from hamiltonian_simulation import (
        ROTATION_BASIS, exact_evolution, observable_shots, pauli_matrix, spin_terms,
    )
    from lcu import (
        MAX_STATEVECTOR_QUBITS, build_lcu_attempt, build_pauli_lcu,
        build_taylor_lcu, postselected_lcu_evolution, postselection_attempts,
    )


def _dense_terms(terms):
    return sum(coefficient * pauli_matrix(label) for coefficient, label in terms)


def _extract_block(lcu):
    dimension = 2**lcu.n_system
    columns = []
    for basis in range(dimension):
        state = np.zeros(2**lcu.circuit.num_qubits, dtype=complex)
        state[basis] = 1
        columns.append(Statevector(state).evolve(lcu.circuit).data[:dimension])
    return np.column_stack(columns)


def _compile(circuit):
    compiled = transpile(circuit, basis_gates=ROTATION_BASIS,
                         optimization_level=1, seed_transpiler=7)
    unexpected = set(compiled.count_ops()) - set(ROTATION_BASIS) - {"measure", "reset", "barrier"}
    if unexpected:
        raise AssertionError(f"Unresolved operations in export: {unexpected}")
    return compiled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--time", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=0.7)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--segments", default="1,2,4,8")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/lcu")
    args = parser.parse_args()
    if not 1 <= args.qubits <= 5:
        parser.error("artifact CLI is bounded to 1–5 system wires")
    segments = [int(part) for part in args.segments.split(",")]
    if not segments or any(not 1 <= r <= 32 for r in segments):
        parser.error("each segment count must be from 1 to 32")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    source_paths = {
        "lcu.py": ROOT / "src/openquantum/simulation/lcu.py",
        "hamiltonian_simulation.py": ROOT / "src/openquantum/hamiltonian_simulation.py",
        "generate_simulation_lcu.py": Path(__file__),
    } if REPOSITORY else {name: HERE / name for name in
                         ("lcu.py", "hamiltonian_simulation.py", "generate_simulation_lcu.py")}
    for name, source in source_paths.items():
        raw = source.read_bytes()
        (output / name).write_bytes(raw)
        sources[name] = {"sha256": hashlib.sha256(raw).hexdigest()}
    license_path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())
    terms = spin_terms(args.qubits, args.J, args.h)
    alpha = sum(abs(c) for c, _ in terms)
    initial = np.zeros(2**args.qubits, dtype=complex)
    initial[0] = 1
    observable = pauli_matrix("I" * (args.qubits - 1) + "Z")
    exact = exact_evolution(args.qubits, args.time, args.J, args.h) if not args.no_sim else None
    exact_state = exact @ initial if exact is not None else None
    accepted_shots = observable_shots(0.03, 0.005)
    records = []
    rows = []
    for count in segments:
        lcu = build_taylor_lcu(terms, args.time / count, args.degree)
        compiled_segment = _compile(lcu.circuit)
        if compiled_segment.size() * count > 250000:
            parser.error("full-attempt export exceeds the 250000-operation bound")
        full = _compile(build_lcu_attempt(lcu, count))
        qasm = qasm3.dumps(full)
        filename = f"taylor-degree{args.degree}-segments{count}.qasm"
        (output / filename).write_text(qasm, encoding="utf-8")
        counts = dict(full.count_ops())
        record = {
            "segments": count, "degree": args.degree,
            "normalization_per_segment": lcu.normalization,
            "naive_word_taylor_normalization_per_segment": sum((alpha * abs(args.time / count))**k / math.factorial(k)
                                                               for k in range(args.degree + 1)),
            "aggregated_pauli_terms": len(lcu.terms),
            "lcu_terms": [{"coefficient_real": coefficient.real,
                           "coefficient_imag": coefficient.imag, "pauli": label}
                          for coefficient, label in lcu.terms],
            "system_qubits": lcu.n_system, "index_qubits": lcu.n_index,
            "logical_qubits": full.num_qubits, "classical_bits": full.num_clbits,
            "segment_gate_counts_before_measurement": dict(compiled_segment.count_ops()),
            "full_attempt_gate_counts": counts, "full_attempt_depth": full.depth(),
            "full_attempt_measurements": counts.get("measure", 0),
            "full_attempt_resets": counts.get("reset", 0),
            "unsynthesized_rotation_count_per_attempt": sum(counts.get(g, 0) for g in ["rx", "ry", "rz"]),
            "exact_t_and_tdg_per_attempt": counts.get("t", 0) + counts.get("tdg", 0),
            "qasm_file": filename, "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
            "simulation": {"performed": False, "reason": "disabled or above bounded statevector size"},
        }
        if not args.no_sim and lcu.circuit.num_qubits <= MAX_STATEVECTOR_QUBITS:
            polynomial = _dense_terms(lcu.terms)
            actual_block = _extract_block(lcu)
            access_error = float(np.linalg.norm(actual_block - polynomial / lcu.normalization, 2))
            if access_error > 1e-10:
                raise AssertionError("Actual access circuit does not encode the Taylor polynomial")
            state, success, conditional = postselected_lcu_evolution(lcu, initial, count)
            expected_branch = np.linalg.matrix_power(polynomial / lcu.normalization, count) @ initial
            if abs(success - np.vdot(expected_branch, expected_branch).real) > 1e-10:
                raise AssertionError("Circuit postselection differs from the independent polynomial")
            exact_z = float(np.vdot(exact_state, observable @ exact_state).real)
            actual_z = float(np.vdot(state, observable @ state).real)
            # Separate one accepted measured trial from N accepted observable shots.
            single_attempts = postselection_attempts(success, 0.01)
            attempts_per_shot = postselection_attempts(success, 0.005 / accepted_shots)
            simulation = {
                "performed": True, "method": "actual segment Statevector plus fresh zero-label projection",
                "access_block_error": access_error,
                "unnormalized_polynomial_operator_error": float(np.linalg.norm(np.linalg.matrix_power(polynomial, count) - exact, 2)),
                "conditional_state_vector_error": float(np.linalg.norm(state - exact_state)),
                "conditional_state_infidelity": float(max(0, 1 - abs(np.vdot(exact_state, state))**2)),
                "conditional_state_stage": "before final system Z readout; exported attempts yield measured trials",
                "exact_observable": exact_z, "conditional_observable": actual_z,
                "conditional_observable_error": abs(actual_z - exact_z),
                "full_attempt_acceptance_probability": success,
                "per_segment_conditional_acceptance_probabilities": conditional,
                "full_attempts_for_one_measured_trial_99_percent_acceptance": single_attempts,
            }
            record["simulation"] = simulation
            record["observable_sampling"] = {
                "observable": "Z(q0)", "statistical_error": 0.03,
                "statistical_failure_budget": 0.005, "accepted_shots": accepted_shots,
                "acceptance_failure_budget_for_all_shots": 0.005,
                "full_attempts_per_accepted_shot_cap": attempts_per_shot,
                "worst_case_full_attempts": accepted_shots * attempts_per_shot,
                "worst_case_segment_calls": accepted_shots * attempts_per_shot * count,
                "expected_full_attempts_without_a_cap": accepted_shots / success,
                "sampling_performed": False,
                "sampling_bound": "Hoeffding for independent +/-1 outcomes; union bound on capped retries",
            }
            rows.append({"segments": count, "degree": args.degree,
                         "state_vector_error": simulation["conditional_state_vector_error"],
                         "polynomial_operator_error": simulation["unnormalized_polynomial_operator_error"],
                         "observable_error": simulation["conditional_observable_error"],
                         "acceptance_probability": success, "attempts_for_one_measured_trial_99_percent": single_attempts,
                         "logical_qubits": full.num_qubits, "measurements_per_attempt": counts.get("measure", 0),
                         "resets_per_attempt": counts.get("reset", 0), "cx_per_attempt": counts.get("cx", 0)})
        records.append(record)
    metadata = {
        "algorithm": "classically aggregated truncated-Taylor Pauli LCU with postselection",
        "input": {"n": args.qubits, "J": args.J, "h": args.h, "time": args.time,
                  "hbar": 1, "boundaries": "open", "initial_state": "0" * args.qubits,
                  "terms": [[c, p] for c, p in terms], "observable": "Z(q0)"},
        "environment": {"python": platform.python_version(), "packages": {
            p: importlib.metadata.version(p) for p in ["qiskit", "numpy", "scipy", "matplotlib"]}},
        "sources": sources, "transpilation": {"optimization_level": 1, "seed_transpiler": 7, "basis": ROTATION_BASIS},
        "resource_kind": "logical circuit ledger before coherent rotation synthesis or error correction",
        "physical_resource_estimate": None,
        "included": ["basis-input resets", "actual PREPARE", "actual SELECT", "PREPARE inverse",
                     "every segment label measurement", "label reset/reuse", "final system Z readout",
                     "complete-restart caps and accepted-shot measurement procedure"],
        "exclusions": ["coefficient-loading classical preprocessing time and memory",
                       "classical acceptance/readout processing time", "rotation synthesis",
                       "physical faults and error correction", "hardware routing and timing"],
        "error_contract": {"hamiltonian_representation": "stated Pauli model; IEEE floating coefficients and aggregation",
                           "hamiltonian_approximation": "polynomial/operator and conditional-state error measured separately",
                           "coherent_synthesis": "pending; never treated as a stochastic failure",
                           "statistical": "unexecuted Hoeffding shot budget disclosed",
                           "physical_faults": "pending shared adapter; no fabricated physical estimate"},
        "scope_limits": {"system_generation_qubits": 32, "explicit_lcu_terms": 256, "taylor_degree": 12,
                         "statevector_total_qubits": MAX_STATEVECTOR_QUBITS,
                         "artifact_cli_system_qubits": 5, "artifact_cli_segment_count": 32,
                         "decomposed_operations_per_attempt": 250000,
                         "preprocessing": "Pauli aggregation can grow exponentially; not the word-indexed scalable oracle",
                         "amplification": "none; independent postselection restarts explicitly charged"},
        "records": records,
    }
    (output / "logical-resources.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if rows:
        with (output / "convergence.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].loglog([r["segments"] for r in rows], [r["state_vector_error"] for r in rows], "o-", label="conditional state error")
        axes[0].loglog([r["segments"] for r in rows], [r["observable_error"] for r in rows], "s-", label="Z(q0) error")
        axes[0].set(xlabel="segments", ylabel="ideal approximation error")
        axes[0].legend()
        axes[1].semilogy([r["segments"] for r in rows], [r["acceptance_probability"] for r in rows], "o-", color="#9c4f2e")
        axes[1].set(xlabel="segments", ylabel="full-attempt acceptance probability")
        fig.suptitle(f"Explicit postselected Taylor LCU, degree {args.degree}")
        fig.tight_layout()
        fig.savefig(output / "convergence.svg")
        plt.close(fig)
    prepare = build_pauli_lcu(terms).prepare
    if prepare.num_qubits <= 4:
        fig = prepare.draw(output="mpl", fold=24, scale=0.8)
        fig.savefig(output / "prepare.svg", bbox_inches="tight")
        plt.close(fig)
    for name in ("convergence.svg", "prepare.svg"):
        path = output / name
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(json.dumps({"output": str(output), "records": len(records), "simulated": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
