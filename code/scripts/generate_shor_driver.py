#!/usr/bin/env python3
"""Generate a bounded random-base factoring trace and actual adaptive circuits."""
from __future__ import annotations

import argparse
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
matplotlib.rcParams["svg.hashsalt"] = "openquantum-shor-driver"
import matplotlib.pyplot as plt
from qiskit import qasm3, transpile

from openquantum.shor import MAX_PRIMITIVE_GATES, ROTATION_BASIS
from openquantum.shor_driver import (
    FactoringConfig, build_iterative_order_circuit, factor_with_quantum_trials,
    numerical_policy_audit, prime_factorization,
)


def recursive_quantum_counts(circuit):
    """All-true branch upper ledger; actual observed paths are counted separately."""
    counts = {}
    condition_checks = 0
    for instruction in circuit.data:
        op = instruction.operation
        if op.name == "if_else":
            condition_checks += 1
            nested, more = recursive_quantum_counts(op.blocks[0])
            condition_checks += more
            for name, count in nested.items():
                counts[name] = counts.get(name, 0) + count
        else:
            if op.name not in {*ROTATION_BASIS, "measure", "reset"}:
                raise AssertionError(f"Opaque operation in lowered adaptive program:{op.name}")
            counts[op.name] = counts.get(op.name, 0) + 1
    return counts, condition_checks


def generate(modulus: int, config: FactoringConfig, output: Path, *, include_recursive: bool = False):
    output.mkdir(parents=True, exist_ok=True)
    run = factor_with_quantum_trials(modulus, config=config)
    generated = time.perf_counter()
    run["numerical_policy_audit"] = numerical_policy_audit(modulus, config) if modulus <= 63 else {"status": "unavailable_above_reference_limit"}
    programs = []
    seen = set()
    for draw in run["trace"]:
        for attempt in draw["quantum_attempts"]:
            key = attempt["base"], attempt["phase_bits"]
            if key in seen:
                continue
            seen.add(key)
            source = build_iterative_order_circuit(modulus, key[0], key[1])
            lowered = transpile(source, basis_gates=list(ROTATION_BASIS), optimization_level=1, seed_transpiler=7)
            qasm = qasm3.dumps(lowered)
            counts, checks = recursive_quantum_counts(lowered)
            if sum(counts.values()) > MAX_PRIMITIVE_GATES:
                raise ValueError("Lowered adaptive program exceeds explicit primitive bound")
            filename = f"adaptive-a{key[0]}-t{key[1]}.qasm"
            (output / filename).write_text(qasm)
            programs.append({"base": key[0], "phase_bits": key[1], "file": filename,
                             "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                             "all_true_branch_primitive_counts": counts, "source_condition_checks": checks,
                             "count_scope": "source program upper ledger; full observed path costs in each actual trial",
                             "initial_all_wire_resets": source.num_qubits, "recycling_resets": key[1] - 1,
                             "global_phase_policy": "overall and classically selected branch phases omitted by pinned exporter; measured instrument probabilities preserved, no coherent recontrol specification"})
    run["adaptive_program_exports"] = programs
    if include_recursive:
        run["recursive_prime_factorization"] = prime_factorization(modulus, config=config)
        run["recursive_demo_scope"] = "separate independently invoked prime-factorization workflow; its counters are not included in the primary split run"
    if run["trace"] and any(draw["quantum_attempts"] for draw in run["trace"]):
        attempt = next(attempt for draw in run["trace"] for attempt in draw["quantum_attempts"])
        steps = attempt["steps"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
        axes[0].bar([step["output_bit"] for step in steps], [step["selected_conditional_probability"] for step in steps],
                    color=["#d7804c" if step["measured_bit"] else "#287e83" for step in steps])
        axes[0].set(xlabel="Output bit (least significant first)", ylabel="Selected conditional probability", ylim=(0, 1.05), title=f"Actual adaptive sample y={attempt['outcome']}")
        xs = list(range(run["base_draws"]))
        axes[1].bar(xs, [len(draw["quantum_attempts"]) for draw in run["trace"]], color="#287e83")
        axes[1].set(xlabel="Base draw index", ylabel="Complete quantum trials", title="Failed trials remain in the workload")
        axes[1].set_xticks(xs, [str(draw["base"]) for draw in run["trace"]])
        fig.tight_layout(); fig.savefig(output / "trace.svg", metadata={"Date": None}); plt.close(fig)
    for name in ("shor.py", "shor_driver.py"):
        (output / name).write_bytes((ROOT / "src/openquantum" / name).read_bytes())
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    run["environment"] = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name) for name in ("qiskit", "numpy", "matplotlib")}, "device": "CPU", "seed": config.seed}
    run["source_sha256"] = {name: hashlib.sha256((ROOT / "src/openquantum" / name).read_bytes()).hexdigest() for name in ("shor.py", "shor_driver.py")}
    run["source_commit"] = git_revision(ROOT)
    run["source_commit_scope"] = "checkout at generation; exact working module hashes recorded separately"
    run["artifact_generation_wall_seconds"] = time.perf_counter() - generated
    (output / "workload.json").write_text(json.dumps(run, indent=2) + "\n")
    for svg in output.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    print(f"N={modulus}: {run['status']}, factors={run['factors']}, bases={run['base_draws']}, complete quantum trials={run['quantum_trials']}")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modulus", type=int, default=21)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-base-draws", type=int, default=8)
    parser.add_argument("--shots-per-base", type=int, default=2)
    parser.add_argument("--max-quantum-trials", type=int, default=16)
    parser.add_argument("--phase-bits", type=int)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/shor/driver")
    args = parser.parse_args()
    generate(args.modulus, FactoringConfig(seed=args.seed, max_base_draws=args.max_base_draws,
             shots_per_coprime_base=args.shots_per_base, max_quantum_trials=args.max_quantum_trials, phase_bits=args.phase_bits),
             args.output_dir, include_recursive=args.recursive)


if __name__ == "__main__":
    main()
