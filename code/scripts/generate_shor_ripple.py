#!/usr/bin/env python3
"""Bounded ripple-carry order circuits and equal-task Fourier comparison."""
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
matplotlib.rcParams["svg.hashsalt"] = "openquantum-shor-ripple"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3

from openquantum.shor import build_order_finding_circuit, decompose, success_contract
from openquantum.shor_ripple import (
    ripple_adder_gate, controlled_multiplier_gate_ripple,
    build_order_finding_circuit_ripple, simulate_boolean_batch,
    simulate_order_finding_ripple, source_primitives,
    MAX_SOURCE_PRIMITIVES, MAX_SOURCE_QUBITS, MAX_SPARSE_PHASE_BITS,
    MAX_SPARSE_STATES, MAX_SPARSE_GATE_APPLICATIONS, MAX_STATEVECTOR_QUBITS,
    MAX_DENSE_GATE_AMPLITUDE_WORK,
)


def resources(circuit):
    compiled = decompose(circuit)
    return compiled, {"logical_qubits": compiled.num_qubits, "decomposed_gate_counts": dict(compiled.count_ops()),
                      "dependency_depth": compiled.depth(), "primitive_operations": sum(compiled.count_ops().values()),
                      "rotation_basis_operations": sum(compiled.count_ops().get(name, 0) for name in ("rx", "ry", "rz"))}


def generate_instance(output: Path, N: int, a: int, t: int, *, simulate: bool, dense: bool):
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    source = build_order_finding_circuit_ripple(N, a, t)
    circuit = build_order_finding_circuit_ripple(N, a, t, measure=True)
    compiled, ledger = resources(circuit)
    _, old = resources(build_order_finding_circuit(N, a, t, measure=True))
    qasm = qasm3.dumps(compiled)
    (output / "circuit.qasm").write_text(qasm)
    module = (ROOT / "src/openquantum/shor_ripple.py").read_bytes()
    base = (ROOT / "src/openquantum/shor.py").read_bytes()
    (output / "shor_ripple.py").write_bytes(module)
    (output / "shor.py").write_bytes(base)
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    result = {"performed": False, "reason": "explicitly disabled"}
    if simulate:
        result = simulate_order_finding_ripple(N, a, t, qiskit=dense)
        ys = np.arange(1 << t)
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(ys, result["probabilities"], color="#287e83", label="actual ripple source circuit")
        ax.plot(ys, result["reference_probabilities"], ".", color="#d8794f", label="independent order/Fourier reference")
        ax.set(xlabel="Exponent outcome", ylabel="Ideal probability", title=f"Ripple arithmetic: N={N}, a={a}, t={t}")
        ax.legend(); fig.tight_layout()
        fig.savefig(output / "probabilities.svg", metadata={"Date": None}); plt.close(fig)
    # All valid control/input basis branches of the actual Boolean multiplier.
    arithmetic = QuantumCircuit(controlled_multiplier_gate_ripple(N, a).num_qubits)
    arithmetic.append(controlled_multiplier_gate_ripple(N, a), range(arithmetic.num_qubits))
    tested_values = list(range(N)) if N <= 33 else sorted({0, 1, N // 2, N - 1})
    pairs = [(control, value) for control in [0, 1] for value in tested_values]
    initial = [control + (value << 1) for control, value in pairs]
    actual = simulate_boolean_batch(arithmetic, initial)
    expected = [control + (((a * value % N) if control else value) << 1) for control, value in pairs]
    if actual != expected:
        raise AssertionError("Actual Boolean multiplier disagrees with independent modular arithmetic")
    # Readout/retry contract is only derived for the bounded classical reference.
    contract = success_contract(N, a, t) if t <= 12 and N <= 4095 else None
    source_counts = {}
    for name, _, _ in source_primitives(source):
        source_counts[name] = source_counts.get(name, 0) + 1
    record = {"modulus": N, "base": a, "phase_bits": t, "integer_bits": (N - 1).bit_length(),
              "ripple": ledger, "fourier_same_task": old,
              "actual_source_primitive_counts": source_counts, "simulation": result,
              "boolean_multiplier_validation": {"performed": True, "valid_control_input_cases": len(initial), "all_work_zero_on_tested_inputs": True,
                                                "all_valid_inputs_tested": N <= 33, "tested_input_values": tested_values,
                                                "method": "exact integer interpreter of actual X/CX/CCX/SWAP source primitives"},
              "retry_contract": contract,
              "comparison_scope": "identical N/a/t, input1, exponent readout, exact-angle circuit model and reconstruction rule; logical counts only",
              "physical_estimation_status": "separate conditional comparison file for default N15 only",
              "arithmetic": "Cuccaro-style MAJ/UMA modulo2^m add, controlled constant preparation, borrow cleanup, compute-swap-unmultiply",
              "valid_domain": "x<N, accumulator/borrow/constant/carry initialized0; reversible extension unspecified on dirty/invalid input",
              "logical_scaling": "width t+3n+4; adderO(n), multiplierO(n^2), fullO(t*n^2+t^2), O(n^3) at t2n",
              "limits": {"source_qubits": MAX_SOURCE_QUBITS, "source_primitives": MAX_SOURCE_PRIMITIVES,
                         "sparse_phase_bits": MAX_SPARSE_PHASE_BITS, "sparse_amplitudes": MAX_SPARSE_STATES,
                         "sparse_gate_state_work": MAX_SPARSE_GATE_APPLICATIONS, "qiskit_dense_qubits": MAX_STATEVECTOR_QUBITS,
                         "dense_gate_amplitude_work": MAX_DENSE_GATE_AMPLITUDE_WORK,
                         "input_bits": 12, "phase_bits_generation": 24, "primitive_export": 500000},
              "preparation_and_readout": "all model-zero ancillas, X for input1, exponent H, explicit controlled multipliers and cleanup, inverseQFT, final t readouts",
              "included_in_counts": ["all controlled constant preparation/unpreparation", "borrow and carry handling", "inverse arithmetic", "input and exponent preparation", "inverseQFT", "complete exponent readout"],
              "excluded_from_physical_runtime": ["classical base choice/CF/gcd/verification", "generation/compilation", "generic factoring success", "certified synthesis sequence or confidence", "exact Clifford scheduling"],
              "compiled_global_phase_radians": float(compiled.global_phase),
              "qasm_policy": "exporter omits whole global phase; complete measured trial probabilities preserved; not phase-preserving controlled oracle specification",
              "module_sha256": hashlib.sha256(module).hexdigest(), "base_module_sha256": hashlib.sha256(base).hexdigest(),
              "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
              "source_commit": git_revision(ROOT),
              "source_commit_scope": "checkout at generation; module hashes identify exact working sources",
              "environment": {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")}, "device": "CPU"},
              "wall_seconds_generation_compilation_validation": time.perf_counter() - start}
    fig = source.draw(output="mpl", fold=36, scale=0.7)
    fig.savefig(output / "circuit.svg", bbox_inches="tight", metadata={"Date": None}); plt.close(fig)
    (output / "logical-resources.json").write_text(json.dumps(record, indent=2) + "\n")
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    print(f"Ripple N{N}/a{a}/t{t}: {ledger['logical_qubits']} wires, {ledger['primitive_operations']} primitives; sparse={simulate}; dense={dense}")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/shor/ripple")
    parser.add_argument("--modulus", type=int, default=15)
    parser.add_argument("--base", type=int, default=2)
    parser.add_argument("--phase-bits", type=int, default=4)
    parser.add_argument("--no-sim", action="store_true", help="Skip quantum amplitude simulation; bounded Boolean checks remain")
    parser.add_argument("--dense-check", action="store_true", help="Optional expensive actual Qiskit full-state check; separately capped")
    parser.add_argument("--no-extra", action="store_true")
    parser.add_argument("--physical", action="store_true")
    args = parser.parse_args()
    generate_instance(args.output_dir, args.modulus, args.base, args.phase_bits, simulate=not args.no_sim, dense=args.dense_check)
    if not args.no_extra:
        generate_instance(args.output_dir / "n21", 21, 2, 6, simulate=not args.no_sim, dense=False)
        rows = []
        for N in (7, 15, 21, 33):
            n = (N - 1).bit_length(); t = 2 * n
            for method, builder in (("fourier", build_order_finding_circuit), ("ripple", build_order_finding_circuit_ripple)):
                _, data = resources(builder(N, 2, t, measure=True))
                rows.append({"N": N, "a": 2, "n": n, "t": t, "method": method,
                             "logical_qubits": data["logical_qubits"], "primitive_operations": data["primitive_operations"],
                             "rotation_basis_operations": data["rotation_basis_operations"], "dependency_depth": data["dependency_depth"], "simulation_performed": False})
        with (args.output_dir / "scaling.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)
        adder = QuantumCircuit(5); adder.append(ripple_adder_gate(2), range(5))
        figure = adder.decompose().draw(output="mpl", fold=32, scale=0.8)
        figure.savefig(args.output_dir / "adder.svg", bbox_inches="tight", metadata={"Date": None}); plt.close(figure)
        path = args.output_dir / "adder.svg"
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    if args.physical:
        if (args.modulus, args.base, args.phase_bits) != (15, 2, 4):
            raise ValueError("Conditional equal-task projection is scoped to the frozen N15/a2/t4 comparison")
        from openquantum.resource_estimation import CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources
        budget = CircuitResourceBudget(synthesis_model_error_per_execution=1e-5, physical_failure_probability=.005, executions=8)
        reports = {}
        for name, builder in (("fourier", build_order_finding_circuit), ("ripple", build_order_finding_circuit_ripple)):
            result = estimate_circuit_resources(builder(15, 2, 4, measure=True), budget=budget, profile=SurfaceCodeProfile(), cache_dir=ROOT / ".cache/shor-ripple-qdk")
            result.pop("qasm3"); reports[name] = result
        reports["comparison_contract"] = {"N": 15, "a": 2, "t": 4, "input": 1, "readout": "exponent", "ideal_factor_success": .5,
                                          "complete_trials": 8, "ideal_sampling_failure_allowance": .005,
                                          "physical_failure_allowance_whole_workload": .005,
                                          "synthesis_model_error_per_trial": 1e-5,
                                          "scope": "same conditional mean-fit model and frozen physical profile; no concrete synthesis/error certificate or hardware advantage"}
        (args.output_dir / "surface-code-comparison.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
