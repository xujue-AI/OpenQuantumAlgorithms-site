#!/usr/bin/env python3
"""Bounded explicit-arithmetic order finding, verification and cost artifacts."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-shor"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, qasm3

from openquantum.shor import (
    MAX_INTEGER_BITS, MAX_PHASE_BITS, MAX_PRIMITIVE_GATES, MAX_SIM_QUBITS,
    build_order_finding_circuit, decompose, factors_from_order, reconstruct_order,
    simulate_phase_distribution, success_contract,
)


def generate(output: Path, modulus: int, base: int, phase_bits: int, *, simulate: bool = True, physical: bool = False, scaling: bool = True):
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    circuit = build_order_finding_circuit(modulus, base, phase_bits)
    generation_seconds = time.perf_counter() - start
    start = time.perf_counter()
    compiled = decompose(circuit)
    compilation_seconds = time.perf_counter() - start
    measured = compiled.copy()
    measured.add_register(ClassicalRegister(phase_bits, "phase"))
    measured.measure(range(phase_bits), range(phase_bits))
    qasm = qasm3.dumps(measured)
    (output / "circuit.qasm").write_text(qasm)
    source_bytes = (ROOT / "src/openquantum/shor.py").read_bytes()
    (output / "shor.py").write_bytes(source_bytes)
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    n = (modulus - 1).bit_length()
    constants = [pow(base, 1 << j, modulus) for j in range(phase_bits)]
    contract = success_contract(modulus, base, phase_bits) if phase_bits <= 12 else None
    metadata = {
        "algorithm": "Order finding with explicit Fourier modular additions and compute-swap-uncompute multiplication",
        "modulus": modulus, "base": base, "integer_bits": n, "phase_bits": phase_bits,
        "total_logical_qubits": circuit.num_qubits, "work_qubits": n + 2,
        "controlled_power_slots": phase_bits,
        "controlled_modular_multiplications": sum(value != 1 for value in constants),
        "nonidentity_modular_multiplications": sum(value != 1 for value in constants),
        "classically_computed_multiplier_constants": constants,
        "modular_additions_per_nonidentity_multiplication": 2 * n,
        "source_gate_counts": dict(circuit.count_ops()), "decomposed_gate_counts": dict(measured.count_ops()),
        "decomposed_dependency_depth": measured.depth(), "primitive_unitary_gates": sum(compiled.count_ops().values()),
        "continuous_rotations": sum(compiled.count_ops().get(name, 0) for name in ("rx", "ry", "rz")),
        "logical_resource_status": "computed explicit-circuit counts before synthesis and error correction",
        "generation_wall_seconds": generation_seconds, "compilation_wall_seconds": compilation_seconds,
        "workload_contract": contract, "physical_resource_estimate": None,
        "included": ["prepare exponent plus state and x=1", "all defined controlled modular-add gates", "compute/swap/inverse-multiplier uncomputation", "inverse QFT", "final exponent measurement", "all full circuits in fixed-base retry contract"],
        "excluded": ["random base selection/rejection and generic factoring success", "classical preprocessing/postprocessing from physical runtime", "certified rotation synthesis", "exact physical Clifford schedule", "external input loading beyond classically specified N,a"],
        "valid_domain": "x<N, n+1 accumulator initially zero, borrow flag zero; outside-domain circuit extension unspecified",
        "preparation_model": "all qubits begin in the circuit model's |0>; no explicit reset instruction",
        "precision_model": "all generated rotations are exact analytic angles represented numerically; no certified compiled angle approximation",
        "compiled_global_phase_radians": float(compiled.global_phase),
        "qasm_global_phase_policy": "Pinned QASM exporter omits overall global phase; exported complete measured trials preserve probabilities, not phase-preserving coherently controlled oracle specifications",
        "limits": {"input_integer_bits": MAX_INTEGER_BITS, "phase_bits": MAX_PHASE_BITS, "statevector_total_qubits": MAX_SIM_QUBITS, "primitive_export_gates": MAX_PRIMITIVE_GATES,
                   "materialization_guard": "100*n*t*(n+1)^2 <= 500000 conservative source-work preflight plus actual post-compilation cap"},
        "asymptotic_implemented_cost": "logical width t+2n+2; exact QFT modular addition O(n^2), multiplication O(n^3), t powers O(t*n^3)+O(t^2); for t=2n O(n^4)",
        "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
        "implementation_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "environment": {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")}, "device": "CPU", "seed": 20261112},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_commit_scope": "base checkout at generation; source hash above identifies working implementation",
        "simulation": {"performed": False, "reason": "explicitly disabled"},
    }
    if simulate:
        start = time.perf_counter()
        result = simulate_phase_distribution(modulus, base, phase_bits)
        result["performed"] = True
        result["method"] = "Qiskit ideal statevector of actual arithmetic circuit vs independent Fourier residue-class sum"
        result["wall_seconds"] = time.perf_counter() - start
        metadata["simulation"] = result
        rng = np.random.default_rng(20261112)
        trials = contract["serial_trials"]
        probabilities = np.array(result["probabilities"])
        probabilities /= probabilities.sum()
        start = time.perf_counter()
        outcomes = rng.choice(1 << phase_bits, size=trials, p=probabilities) if trials is not None else []
        samples = []
        for outcome in outcomes:
            recovered = reconstruct_order(int(outcome), phase_bits, modulus, base)
            samples.append({"outcome": int(outcome), "recovered_order": recovered, "factors": factors_from_order(modulus, base, recovered)})
        metadata["executed_ideal_sampling"] = {"performed": trials is not None, "seed": 20261112, "trial_count": trials, "samples": samples,
                                              "classical_sampling_and_reconstruction_seconds": time.perf_counter() - start,
                                              "scope": "draws from validated exact circuit distribution; no hardware execution"}
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(np.arange(len(probabilities)), probabilities, color="#287e83", label="actual arithmetic circuit")
        ax.plot(np.arange(len(probabilities)), result["reference_probabilities"], ".", color="#dc7b4e", label="independent Fourier sum")
        ax.set(xlabel=f"Exponent measurement y ({phase_bits} bits)", ylabel="Ideal probability", title=f"Order finding: a={base}, N={modulus}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "probabilities.svg", metadata={"Date": None})
        plt.close(fig)
    figure = circuit.draw(output="mpl", fold=36, scale=0.75)
    figure.savefig(output / "circuit.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    if scaling:
        rows = []
        for test_N in (7, 15, 21, 33):
            bits = (test_N - 1).bit_length()
            candidate = decompose(build_order_finding_circuit(test_N, 2, 2 * bits, measure=True))
            rows.append({"N": test_N, "a": 2, "n": bits, "t": 2 * bits, "logical_qubits": candidate.num_qubits,
                         "nonidentity_multiplications": sum(pow(2, 1 << j, test_N) != 1 for j in range(2 * bits)),
                         "primitive_operations_including_readout": sum(candidate.count_ops().values()), "dependency_depth": candidate.depth(),
                         "continuous_rotations": sum(candidate.count_ops().get(name, 0) for name in ("rx", "ry", "rz")), "simulation_performed": False})
        with (output / "scaling.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        metadata["scaling_scope"] = "actual arithmetic circuit materializations with t=2n; no large-instance simulations; favorable identity powers vary with N/a"
    if physical:
        if contract is None or contract["serial_trials"] is None:
            raise ValueError("Physical projection needs bounded fixed-base repetition contract")
        from openquantum.resource_estimation import CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources
        projection = estimate_circuit_resources(measured, budget=CircuitResourceBudget(
            synthesis_model_error_per_execution=1e-5, physical_failure_probability=0.005, executions=contract["serial_trials"]),
            profile=SurfaceCodeProfile(), cache_dir=ROOT / ".cache/shor-qdk")
        projection.pop("qasm3")
        projection["algorithm_success_scope"] = "0.005 ideal fixed-base sampling failure; 0.005 separate whole-workload physical allowance; mean-fit synthesis supplies no certified total factoring confidence"
        (output / "surface-code.json").write_text(json.dumps(projection, indent=2) + "\n")
        metadata["physical_resource_estimate"] = {"status": "computed_conditional_mean_fit_resource_projection", "file": "surface-code.json", "serial_trials": contract["serial_trials"]}
    (output / "logical-resources.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"N={modulus}, a={base}, t={phase_bits}: {circuit.num_qubits} wires, {sum(compiled.count_ops().values())} unitary primitives; checked={simulate}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, default=15)
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--phase-bits", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/shor")
    parser.add_argument("--no-sim", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--physical", action="store_true", help="Run the conditional local QDK projection with explicit workload budgets")
    args = parser.parse_args()
    generate(args.output_dir, args.modulus, args.base, args.phase_bits, simulate=not args.no_sim, physical=args.physical, scaling=not args.no_scaling)


if __name__ == "__main__":
    main()
