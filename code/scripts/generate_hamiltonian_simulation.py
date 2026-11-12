#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit product-formula circuits, bounded checks and logical ledgers."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector
from openquantum.hamiltonian_simulation import (
    MAX_DENSE_QUBITS, ROTATION_BASIS, compare_product_formula, decompose,
    observable_shots, product_formula_circuit, spin_terms,
)


def source_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def circuit_ledger(n=3, time=1.0, steps=32, order=2, J=1.0, h=0.7):
    unitary = product_formula_circuit(n, time, steps, order, J, h)
    measured = decompose(product_formula_circuit(n, time, steps, order, J, h, measure=True))
    counts = dict(measured.count_ops())
    rotations = sum(counts.get(gate, 0) for gate in ("rx", "ry", "rz"))
    shots = observable_shots()
    ledger = {
        "method": f"order-{order} product formula", "system_qubits": n,
        "logical_qubits": measured.num_qubits, "ancillas": 0,
        "time": time, "hbar": 1, "J": J, "h": h, "steps": steps,
        "boundary": "open", "initial_state": "0" * n, "observable": "Z(q0)",
        "hamiltonian_terms_qiskit_order": spin_terms(n, J, h),
        "input_preparation": "reset to |0...0>; no preparation gates for chosen basis state",
        "hamiltonian_access": "explicit per-term CX-RZ-CX and RX; no oracle or data loading",
        "source_gate_counts": dict(unitary.count_ops()), "source_depth": unitary.depth(),
        "decomposed_gate_counts_per_shot": counts, "decomposed_depth_per_shot": measured.depth(),
        "count_basis": ROTATION_BASIS, "arbitrary_rotation_count_per_shot": rotations,
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
        "resource_kind": "logical counts before coherent rotation synthesis and error correction",
        "physical_resource_estimate": None,
        "error_contract_proposal": {
            "requested_observable_absolute_error": 0.05,
            "hamiltonian_operator_error_allowance": 0.005,
            "hamiltonian_observable_bias_allowance": 0.01,
            "coherent_synthesis_operator_error_allowance": 0.005,
            "synthesis_observable_bias_allowance": 0.01,
            "statistical_absolute_error": 0.03, "statistical_failure_probability": 0.005,
            "physical_failure_probability_across_all_shots": 0.005,
            "serial_shots": shots,
            "per_rotation_operator_error_allowance": 0.005 / rotations if rotations else None,
            "physical_failure_union_bound_per_shot": 0.005 / shots,
            "status": "proposal; synthesis and physical adapter pending; unsynthesized circuit is checked",
        },
        "measurement": "measure every system wire in Z; estimator uses bit 0 as +/-1",
        "decomposed_gate_counts_all_shots_proposal": {gate: shots * value for gate, value in counts.items()},
        "exclusions": ["classical coefficient handling wall time", "classical readout/postprocessing wall time",
                       "initial reset time", "physical gate scheduling, factories and error correction",
                       "coherent rotation synthesis gates and errors"],
    }
    return unitary, measured, ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--time", type=float, default=1)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--J", type=float, default=1)
    parser.add_argument("--h", type=float, default=0.7)
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/hamiltonian-simulation")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    circuit, measured, ledger = circuit_ledger(args.qubits, args.time, args.steps, args.order, args.J, args.h)
    implementation = (ROOT / "src/openquantum/hamiltonian_simulation.py").read_bytes()
    (output / "hamiltonian_simulation.py").write_bytes(implementation)
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    qasm = qasm3.dumps(measured)
    (output / "product-formula.qasm").write_text(qasm)
    ledger["provenance"] = {
        "source_revision_at_generation": source_revision(),
        "implementation_sha256": hashlib.sha256(implementation).hexdigest(),
        "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "source_note": "revision plus implementation hash identifies working-tree code; generation may precede commit",
        "command": "python scripts/generate_hamiltonian_simulation.py " + " ".join(sys.argv[1:]),
        "python": platform.python_version(),
        "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "numpy", "scipy", "matplotlib")},
        "seed": 7, "device": "CPU", "backend": "Qiskit Statevector/Operator; SciPy expm reference",
    }
    performed = not args.no_sim and args.qubits <= MAX_DENSE_QUBITS
    ledger["validation"] = {"performed": performed}
    if performed:
        result = compare_product_formula(args.qubits, args.time, args.steps, args.order, args.J, args.h)
        ledger["validation"].update(result)
        ledger["validation"]["hamiltonian_operator_allowance_met"] = result["operator_error"] <= 0.005
        rows = [compare_product_formula(args.qubits, args.time, r, o, args.J, args.h)
                for o in (1, 2) for r in (1, 2, 4, 8, 16, 32, 64)]
        with (output / "convergence.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        for o, color in ((1, "#346b8a"), (2, "#be6b35")):
            group = [item for item in rows if item["order"] == o]
            for ax, key in zip(axes, ["operator_error", "state_vector_error", "observable_error"]):
                ax.loglog([a["steps"] for a in group], [a[key] for a in group], "o-", color=color,
                          label=f"Order {o}", alpha=0.85)
                ax.set(xlabel="Product-formula steps", ylabel=key.replace("_", " "))
                ax.grid(alpha=0.18)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(output / "convergence.svg")
        plt.close(fig)
        state = Statevector.from_instruction(circuit)
        p_zero = float(sum(p for i, p in enumerate(state.probabilities()) if i % 2 == 0))
        samples = np.random.default_rng(7).binomial(ledger["error_contract_proposal"]["serial_shots"], p_zero)
        ledger["illustrative_sampling"] = {
            "performed": True, "kind": "seeded classical sampling from ideal circuit probabilities",
            "seed": 7, "shots": observable_shots(), "q0_zero_count": int(samples),
            "estimated_z": float(2 * samples / observable_shots() - 1),
            "note": "one illustrative draw; Hoeffding bound is analytic, not validated by one sample",
        }
    else:
        ledger["validation"]["reason"] = "--no-sim or dense/reference bound of eight system qubits"
    if args.qubits <= 4:
        small = product_formula_circuit(args.qubits, args.time, 1, args.order, args.J, args.h)
        fig = small.draw(output="mpl", fold=30, scale=0.9)
        fig.savefig(output / "one-step-circuit.svg", bbox_inches="tight")
        plt.close(fig)
    (output / "logical-resources.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for name in ("convergence.svg", "one-step-circuit.svg"):
        path = output / name
        if path.exists():
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(json.dumps({"output_dir": str(output), "validation": ledger["validation"],
                      "rotation_count": ledger["arbitrary_rotation_count_per_shot"]}, indent=2))


if __name__ == "__main__":
    main()
