#!/usr/bin/env python3
"""Generate the site's Grover demo artifacts and decomposed circuit export.

Run from a checkout without an editable installation:
    python scripts/generate_grover.py
    python scripts/generate_grover.py --qubits 12 --marked 17 --no-sim --no-scaling
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum.grover import (
    ROTATION_BASIS,
    build_grover_circuit,
    decompose_to_rotation_basis,
    expected_success_probability,
    optimal_iterations,
)


def circuit_resources(n_qubits: int, marked: int, iterations: int):
    circuit = build_grover_circuit(n_qubits, marked, iterations)
    compiled = decompose_to_rotation_basis(circuit)
    metadata = {
        "algorithm": "Grover search with one marked state",
        "input_qubits": n_qubits,
        "marked_integer": marked,
        "marked_bitstring": f"{marked:0{n_qubits}b}",
        "iterations": iterations,
        "oracle_calls": iterations,
        "ancillas": compiled.num_qubits - n_qubits,
        "ideal_success_probability": expected_success_probability(n_qubits, iterations),
        "source_gate_counts": dict(circuit.count_ops()),
        "source_depth": circuit.depth(),
        "decomposed_gate_counts": dict(compiled.count_ops()),
        "decomposed_depth": compiled.depth(),
        "decomposed_unitary_gates": sum(compiled.count_ops().values()),
        "count_basis": ROTATION_BASIS,
        "resource_kind": "logical circuit counts before rotation synthesis and error correction",
        "physical_resource_estimate": None,
        "oracle_model": "explicit equality predicate against a known integer; pedagogical",
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
    }
    return circuit, compiled, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, default=3)
    parser.add_argument("--marked", type=int, default=5)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/grover")
    parser.add_argument("--no-sim", action="store_true", help="Generate circuits and counts without a statevector")
    parser.add_argument("--max-sim-qubits", type=int, default=14)
    parser.add_argument("--no-scaling", action="store_true", help="Skip the independent 2–8-qubit scaling table")
    parser.add_argument("--scaling-qubits", default="2,3,4,5,6,7,8")
    args = parser.parse_args()
    iterations = optimal_iterations(args.qubits) if args.iterations is None else args.iterations
    circuit, compiled, metadata = circuit_resources(args.qubits, args.marked, iterations)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    implementation_bytes = (ROOT / "src/openquantum/grover.py").read_bytes()
    (output / "grover.py").write_bytes(implementation_bytes)
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    metadata["implementation"] = {
        "source_path": "src/openquantum/grover.py",
        "sha256": hashlib.sha256(implementation_bytes).hexdigest(),
    }

    measured = compiled.copy()
    measured.measure_all()
    qasm_text = qasm3.dumps(measured)
    (output / "circuit.qasm").write_text(qasm_text, encoding="utf-8")
    metadata["qasm_sha256"] = hashlib.sha256(qasm_text.encode("utf-8")).hexdigest()

    metadata["simulation"] = {"performed": False, "reason": "disabled or above configured qubit limit"}
    if not args.no_sim and args.qubits <= args.max_sim_qubits:
        probabilities = Statevector.from_instruction(circuit).probabilities()
        simulated = float(probabilities[args.marked])
        if abs(simulated - metadata["ideal_success_probability"]) > 1e-10:
            raise AssertionError("Statevector result disagrees with the analytic success probability")
        metadata["simulation"] = {"performed": True, "method": "exact ideal statevector", "success_probability": simulated}
        labels = [f"{i:0{args.qubits}b}" for i in range(2**args.qubits)]
        if args.qubits <= 8:
            fig, ax = plt.subplots(figsize=(8, 3.4))
            ax.bar(labels, probabilities, color=["#1c766c" if i == args.marked else "#b5c6d2" for i in range(len(labels))])
            ax.set(ylabel="Ideal measurement probability", xlabel="Bitstring (most significant bit first)", ylim=(0, 1))
            ax.set_title(f"Grover search: {args.qubits} qubits, target {args.marked:0{args.qubits}b}, {iterations} iterations")
            if args.qubits > 4:
                ax.set_xticks([args.marked], [labels[args.marked]])
            fig.tight_layout()
            fig.savefig(output / "success-probabilities.svg")
            plt.close(fig)

    if args.qubits <= 5 and iterations <= 4:
        fig = circuit.draw(output="mpl", fold=32, scale=0.85)
        fig.savefig(output / "circuit.svg", bbox_inches="tight")
        plt.close(fig)

    versions = {name: importlib.metadata.version(name) for name in ["qiskit", "matplotlib", "numpy"]}
    metadata["environment"] = {"python": platform.python_version(), "packages": versions}
    (output / "logical-resources.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not args.no_scaling:
        rows = []
        for n_qubits in [int(value) for value in args.scaling_qubits.split(",")]:
            _, _, item = circuit_resources(n_qubits, 2**n_qubits - 1, optimal_iterations(n_qubits))
            rows.append({
                "qubits": n_qubits,
                "marked": item["marked_integer"],
                "iterations": item["iterations"],
                "oracle_calls": item["oracle_calls"],
                "ideal_success_probability": item["ideal_success_probability"],
                "decomposed_unitary_gates": item["decomposed_unitary_gates"],
                "decomposed_cx": item["decomposed_gate_counts"].get("cx", 0),
                "decomposed_depth": item["decomposed_depth"],
            })
        with (output / "scaling.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps({"output_dir": str(output), "qubits": args.qubits, "iterations": iterations, "simulation": metadata["simulation"]}, indent=2))


if __name__ == "__main__":
    main()
