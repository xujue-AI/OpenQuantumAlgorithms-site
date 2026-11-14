#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the bounded SKW-style hypercube search checks and circuit exports.

Run from a checkout with the locked Python 3.12 environment. Public downloads
can run beside the matching optimization.py, LICENSE.txt and environment lock.
The default d=4, marked=5, steps=4 is prescribed before the scan or sampling.
Larger supported source circuits require --export-only and are not simulated.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
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

from time import perf_counter

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[1]
CHECKOUT = (PROJECT / "src/openquantum/optimization.py").is_file()
ROOT = PROJECT if CHECKOUT else SCRIPT.parent
SOURCE = ROOT / "src/openquantum/optimization.py" if CHECKOUT else ROOT / "optimization.py"
if not SOURCE.is_file():
    raise SystemExit("Save the matching optimization.py beside this script, or run from a checkout")
spec = importlib.util.spec_from_file_location("optimization_walk_artifact_source", SOURCE)
opt = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = opt
spec.loader.exec_module(opt)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, qasm3
from qiskit.quantum_info import Statevector

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.hashsalt": "openquantum-hypercube-walk"})
SEEDS = (7, 19, 41)
SHOTS = 4096
REFERENCE_SCAN = (1 / 16, 1 / 16, 1 / 4, 1 / 4, 25 / 64, 25 / 64, 1 / 4, 1 / 4, 1 / 256)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def provenance() -> dict:
    lock = ROOT / "requirements-lock.txt"
    record = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(),
                  packages={name: importlib.metadata.version(name)
                            for name in ("qiskit", "numpy", "matplotlib", "jupyter-book")},
                  device=dict(backend="CPU Qiskit Statevector and NumPy PCG64 ideal measurement draws",
                              architecture=platform.machine(), operating_system=platform.system(), accelerator=None),
                  implementation_sha256=digest(SOURCE), generator_sha256=digest(SCRIPT),
                  environment_lock_sha256=digest(lock) if lock.is_file() else None,
                  source_commit=None, dirty_owned_source=None,
                  invocation="python scripts/generate_optimization_walk.py" if CHECKOUT
                  else "python generate_optimization_walk.py",
                  upstream=[dict(package="Qiskit", version=importlib.metadata.version("qiskit"),
                                 repository="https://github.com/Qiskit/qiskit", license="Apache-2.0"),
                            dict(package="NumPy", version=importlib.metadata.version("numpy"),
                                 repository="https://github.com/numpy/numpy", license="BSD-3-Clause")],
                  algorithm_reference=dict(authors=["Neil Shenvi", "Julia Kempe", "K. Birgitta Whaley"],
                                           title="Quantum random-walk search algorithm",
                                           journal="Physical Review A", volume=67, article="052307", year=2003,
                                           doi="10.1103/PhysRevA.67.052307", arxiv="quant-ph/0210064"))
    if CHECKOUT:
        record["source_commit"] = git_revision(ROOT)
        record["dirty_owned_source"] = git_dirty(ROOT, ['src/openquantum/optimization.py', 'scripts/generate_optimization_walk.py', 'notebooks/optimization_walk.ipynb'])
    record["revision_note"] = ("source_commit identifies the checkout base at generation; content hashes "
                               "identify the implementation and generator actually executed")
    return record


def copy_downloads(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "optimization.py").write_bytes(SOURCE.read_bytes())
    (output / "generate_optimization_walk.py").write_bytes(SCRIPT.read_bytes())
    license_path = ROOT / ("LICENSE" if CHECKOUT else "LICENSE.txt")
    if not license_path.is_file():
        raise SystemExit("Implementation license missing; save LICENSE.txt beside this script")
    (output / "LICENSE.txt").write_bytes(license_path.read_bytes())


def export_circuit(output: Path, name: str, circuit: QuantumCircuit) -> dict:
    path = output / name
    path.write_text(qasm3.dumps(circuit), encoding="utf-8")
    return dict(file=name, sha256=digest(path), logical_qubits=circuit.num_qubits,
                gate_counts=dict(circuit.count_ops()), depth=circuit.depth(),
                global_phase_radians=float(circuit.global_phase))


def clean_subspace_oracle_check(dimension: int, marked: int) -> dict:
    """Check rank-one oracle semantics on every active basis state; no large Operator."""
    layout = opt.walk_register_layout(dimension)
    active_size = 1 << layout["active_qubits"]
    total_size = 1 << layout["total_qubits"]
    oracle = opt.build_hypercube_mark_oracle(dimension, marked)
    # The active-space rank-one reflection is formed independently.
    target = np.zeros(active_size, dtype=complex)
    target[marked * dimension:(marked + 1) * dimension] = 1 / np.sqrt(dimension)
    reflection = np.eye(active_size, dtype=complex) - 2 * np.outer(target, target.conj())
    maximum_error, maximum_leakage = 0.0, 0.0
    for basis in range(active_size):
        input_state = np.zeros(total_size, dtype=complex)
        input_state[basis] = 1
        output = Statevector(input_state).evolve(oracle).data
        maximum_error = max(maximum_error, float(np.max(np.abs(output[:active_size] - reflection[:, basis]))))
        maximum_leakage = max(maximum_leakage, float(np.sum(np.abs(output[active_size:]) ** 2)))
    if maximum_error > 1e-12 or maximum_leakage > 1e-12:
        raise AssertionError("explicit joint oracle disagrees with clean-subspace rank-one reflection")
    return dict(method="every active basis input with clean work; independently formed I-2|s,m><s,m|",
                checked_inputs=active_size, total_qubits=layout["total_qubits"],
                max_amplitude_error=maximum_error, max_workspace_leakage=maximum_leakage,
                scope="finite numerical primitive check; no formal theorem or arbitrary-workspace promise")


def draw_one_step(output: Path, dimension: int, marked: int) -> None:
    layout = opt.walk_register_layout(dimension)
    registers = [QuantumRegister(layout["coin_qubits"], "coin"),
                 QuantumRegister(dimension, "position"),
                 QuantumRegister(layout["workspace_qubits"], "work")]
    figure_circuit = QuantumCircuit(*registers)
    figure_circuit.compose(opt.build_hypercube_walk_circuit(dimension, marked, 1,
                                                          stage_barriers=True), inplace=True)
    fig = figure_circuit.draw(output="mpl", fold=38, scale=0.62,
                              style={"backgroundcolor": "#ffffff", "fontsize": 9})
    fig.suptitle("One explicit hypercube search step", fontsize=14, y=1.0)
    fig.text(0.5, 0.005, "Uniform active preparation → joint marked-state reflection → minus Grover coin → matched shift.\n"
             "Clean work is computed and uncomputed; all work wires remain in the final readout circuit.",
             ha="center", fontsize=9)
    fig.savefig(output / "one-step-circuit.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def draw_probability_scan(output: Path, scan: list[dict], selected_steps: int) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    ax.plot([row["steps"] for row in scan], [row["marked_probability"] for row in scan],
            color="#176e70", linewidth=1.8, marker="o", label="explicit circuit")
    ax.scatter([row["steps"] for row in scan], [row["reference_marked_probability"] for row in scan],
               color="#bd6d36", marker="x", s=44, label="independent dense SKW matrix", zorder=4)
    ax.axvline(selected_steps, color="#666666", linestyle="--", linewidth=1,
               label=f"prescribed readout: {selected_steps} steps")
    ax.set(xlabel="Number of complete walk steps", ylabel="Marked-position probability",
           title="Fixed hypercube search probability scan", xticks=range(9), ylim=(0, 0.45))
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output / "probability-scan.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def summary_markdown(run: dict, resources: dict) -> str:
    main = run["main"]
    lines = ["The implemented teaching task is coined search on a 4-dimensional hypercube with 16 vertices, "
             "a supplied marked vertex `0101` (integer 5), and a prescribed four-step final readout. "
             "The target value is used to construct an explicit equality predicate. This is a quantum-walk primitive, "
             "with no application benchmark or optimization-performance claim.", "",
             "The active register has two binary coin bits and four position bits. Five clean workspace bits implement "
             "the joint reflection `I − 2|s_coin,5〉〈s_coin,5|` through AND computation, phase and uncomputation. "
             "Each step applies this oracle, the coin `I − 2|s〉〈s|`, and an explicitly matched shift. "
             "The step differs from the standard SKW step by a global minus sign, which is harmless for this uncontrolled search. "
             "The implementation does not promise phase-correct controlled reuse.", "",
             f"The independent 64 × 64 active-space matrix gives marked-position probability **25/64 = {main['marked_probability']:.6f}** "
             f"after four steps. Across the fixed scan of zero through eight steps, maximum circuit/reference amplitude error "
             f"was {max(row['reference_max_amplitude_error'] for row in run['scan']):.3g}, and maximum workspace leakage "
             f"was {max(row['workspace_leakage'] for row in run['scan']):.3g}. The scan was a diagnostic; it did not select the four-step input. "
             "Every one of the 64 clean-work active basis inputs also passes the independently formed joint-oracle check. "
             "These are finite numerical checks, without a formal proof.", "",
             "Three seeded final readouts use 4,096 modeled ideal measurements each. All 11 qubits are measured; "
             "workspace outcomes are retained rather than postselected. The target is an absolute frequency error of 0.025 "
             "at 95% confidence for each fixed estimate. The separate Hoeffding bound has outcome range one; it does not provide "
             "simultaneous coverage of all rows. The random contract is NumPy PCG64 with `SeedSequence([seed,5])`.", "",
             "| Seed | Marked counts / shots | Marked frequency | Error from 25/64 | Hoeffding 95% half width | Nonzero-work outcomes |",
             "|---:|---:|---:|---:|---:|---:|"]
    for row in run["readout"]:
        lines.append(f"| {row['seed']} | {row['marked_samples']} / {row['shots']} | {row['marked_sample_fraction']:.6f} | "
                     f"{row['marked_sample_fraction'] - main['marked_probability']:+.6f} | {row['hoeffding_half_width_95']:.6f} | "
                     f"{row['workspace_nonzero_samples']} |")
    repeat = run["fixed_state_repetition"]
    lines += ["", f"For this fixed validated state, **{repeat['trials']} independent preparations** give ideal probability "
              f"{repeat['achieved_success_probability']:.6f} of at least one marked readout, exceeding the prescribed 0.99 target. "
              "This formula uses the verified fixed-state probability and equality checks after measurement; it is not a universal "
              "walk or optimizer success guarantee. Every trial includes preparation, four oracle/coin/shift calls and all-qubit readout.", "",
              "| Logical workload | Per preparation | Three 4,096-shot readouts | Fixed 0.99 at-least-one task |",
              "|---|---:|---:|---:|"]
    per, ledger, repeat_ledger = resources["per_preparation"], resources["full_readout_workload"], resources["fixed_state_repetition_workload"]
    for label, key in (("Preparations", "circuit_preparations"), ("Oracle calls", "oracle_calls"),
                       ("Coin calls", "coin_calls"), ("Shift calls", "shift_calls"),
                       ("Individual qubit measurements", "individual_qubit_measurements"),
                       ("T / T† gates", "t_count")):
        lines.append(f"| {label} | {per[key]:,} | {ledger[key]:,} | {repeat_ledger[key]:,} |")
    lines += ["", "The 11-wire main circuit uses 56 explicit CCX gates (40 in the four joint oracles and 16 in the shifts). "
              "Deterministic optimization-level-zero compilation gives **392 T/T† gates**, 340 CX gates, and no arbitrary rotations. "
              "The logical ledger includes input preparation, the constructed oracle, work uncomputation, coin, shift, all repeated trials, "
              "and all final measurements. Arbitrary-rotation synthesis error is zero for this literal Clifford+T implementation. "
              "CPU simulation, classical equality-check time, routing, error correction, faults, factories and physical timing are outside "
              "this logical ledger; the main chapter supplies the separate conditional surface-code analysis.", "",
              "The source generator supports dimensions 2, 4, 8 and 16 and zero through 32 steps, subject to a conservative 100,000 "
              "primitive-gate cap (CCX costs 15; CZ costs 3; other source Clifford gates cost one). The total width is `2(d + log2(d)) − 1`. "
              "Statevector execution is separately capped at 12 total qubits, and the independent dense reference at six active bits. "
              "Dimensions 8 and 16 have exported four-step source and exact Clifford+T circuits **without simulation**. "
              "The marked reflection uses `2(d+r−1)` CCX per step; for `r≥2` the shift uses `d(2r−3)` CCX, with `r=log2(d)`. "
              "The coin adds `2(r−1)` CCX for `r≥3`, one CZ for `r=2`, or one Z for `r=1`; wrappers and Hadamards are also counted. "
              "Consequently source size per step is O(d log d); dense probabilities remain exponentially bounded.", "",
              f"The complete generator ran in {run['workflow_wall_seconds']:.6f} s on the recorded CPU environment. "
              "Artifact JSON records the source commit, actual source/generator/lock hashes, versions, seeds and timing scope. "
              "Qiskit 2.6.2 (Apache-2.0) and NumPy 2.5.4 (BSD-3-Clause) are credited in the provenance; this original implementation "
              "and generator are provided under the repository MIT license. Algorithm source: Shenvi, Kempe and Whaley, "
              "Physical Review A 67, 052307 (2003), DOI `10.1103/PhysRevA.67.052307`, arXiv `quant-ph/0210064`.", ""]
    return "\n".join(lines)


def workload(per: dict, preparations: int) -> dict:
    return {key: value * preparations for key, value in per.items()}


def generate(output: Path, dimension: int = 4, marked: int = 5, steps: int = 4,
             export_only: bool = False) -> dict:
    started = perf_counter()
    if not export_only and (dimension, marked, steps) != (4, 5, 4):
        raise ValueError("the validated artifact workload is fixed at d=4,marked=5,steps=4; other inputs use --export-only")
    if export_only and (output / "runs.json").is_file():
        raise ValueError("export-only output must be separate from an existing validated runs.json")
    source = opt.build_hypercube_walk_circuit(dimension, marked, steps, measure=True)
    compiled = opt.compile_walk_clifford_t(source)
    copy_downloads(output)
    circuits = dict(source=export_circuit(output, "walk-source.qasm", source),
                    clifford_t=export_circuit(output, "walk-clifford-t.qasm", compiled))
    info = provenance()
    layout = opt.walk_register_layout(dimension)
    if export_only:
        record = dict(mode="source and exact Clifford+T export only; no simulation or validated success",
                      dimension=dimension, marked_vertex=marked, steps=steps, layout=layout,
                      circuits=circuits, provenance=info, workflow_wall_seconds=perf_counter() - started)
        write_json(output / "resources.json", record)
        return record
    main = opt.simulate_hypercube_walk(dimension, marked, steps)
    main.pop("probabilities")
    main["position_probabilities"] = main["position_probabilities"].tolist()
    scan = []
    for k in range(9):
        row = opt.simulate_hypercube_walk(dimension, marked, k)
        ref = opt.hypercube_walk_reference(dimension, marked, k)
        reference_p = float(np.sum(np.abs(ref[marked * dimension:(marked + 1) * dimension]) ** 2))
        if abs(row["marked_probability"] - REFERENCE_SCAN[k]) > 1e-12:
            raise AssertionError("main walk scan disagrees with fixed independently verified rational probabilities")
        if row["reference_max_amplitude_error"] > 1e-12 or row["workspace_leakage"] > 1e-12:
            raise AssertionError("walk circuit/reference or clean workspace check failed")
        scan.append(dict(steps=k, marked_probability=row["marked_probability"],
                         reference_marked_probability=reference_p,
                         reference_max_amplitude_error=row["reference_max_amplitude_error"],
                         workspace_leakage=row["workspace_leakage"],
                         statevector_seconds=row["statevector_seconds"]))
    oracle_check = clean_subspace_oracle_check(dimension, marked)
    readout = [opt.sample_hypercube_walk(dimension, marked, steps, shots=SHOTS, seed=seed) for seed in SEEDS]
    if any(row["workspace_nonzero_samples"] or row["hoeffding_half_width_95"] > 0.025 for row in readout):
        raise AssertionError("readout workspace or allocation check failed")
    repetition_trials = opt.walk_repetitions(main["marked_probability"], 0.99)
    repeat = dict(target_success_probability=0.99, trials=repetition_trials,
                  achieved_success_probability=1 - (1 - main["marked_probability"]) ** repetition_trials,
                  task="at least one marked position from independent fixed-state trials, followed by equality check",
                  assumption="ideal validated fixed-state probability; no hardware faults")
    per = dict(circuit_preparations=1, oracle_calls=steps, coin_calls=steps, shift_calls=steps,
               individual_qubit_measurements=layout["total_qubits"],
               t_count=int(compiled.count_ops().get("t", 0) + compiled.count_ops().get("tdg", 0)),
               cx_count=int(compiled.count_ops().get("cx", 0)))
    source_only = []
    scaling = output / "scalability"
    scaling.mkdir(exist_ok=True)
    for d in (8, 16):
        c = opt.build_hypercube_walk_circuit(d, marked, steps, measure=True)
        cpt = opt.compile_walk_clifford_t(c)
        source_only.append(dict(dimension=d, marked_vertex=marked, steps=steps,
                                layout=opt.walk_register_layout(d), simulation_executed=False,
                                validated_success_probability=None,
                                primitive_cap_cost=c.metadata["primitive_cap_cost"],
                                source=export_circuit(scaling, f"d{d}-source.qasm", c),
                                clifford_t=export_circuit(scaling, f"d{d}-clifford-t.qasm", cpt)))
    secondary = opt.simulate_hypercube_walk(2, 1, 4)
    secondary.pop("probabilities")
    secondary["position_probabilities"] = secondary["position_probabilities"].tolist()
    run = dict(mode="bounded ideal SKW-style coined hypercube search",
               input_contract=dict(dimension=dimension, vertices=16, marked_vertex=marked, steps=steps,
                                   output="position measurement followed by equality check", known_marked_teaching_predicate=True,
                                   prescribed_before_scan=True, readout_precision=0.025,
                                   confidence_per_fixed_estimate=0.95),
               main=main, scan=scan, readout=readout, clean_subspace_oracle_check=oracle_check,
               secondary_tiny_check=secondary, fixed_state_repetition=repeat,
               source_only_scalability_exports=source_only,
               timing_scope=dict(statevector_seconds="circuit construction, Statevector evolution, probability aggregation and dense reference check",
                                 sampling_seconds="ideal probabilities normalized, multinomial draws and outcome aggregation",
                                 workflow_wall_seconds="complete CPU checks, circuit exports, copies and figures before final summary/JSON serialization"),
               provenance=info)
    resources = dict(mode="logical resource ledger; separate physical estimate in main chapter",
                     layout=layout, circuits=circuits, per_preparation=per,
                     full_readout_workload=workload(per, SHOTS * len(SEEDS)),
                     fixed_state_repetition_workload=workload(per, repetition_trials),
                     full_readout_source_gate_totals={name: int(count) * SHOTS * len(SEEDS)
                                                     for name, count in source.count_ops().items() if name != "barrier"},
                     full_readout_clifford_t_gate_totals={name: int(count) * SHOTS * len(SEEDS)
                                                         for name, count in compiled.count_ops().items() if name != "barrier"},
                     synthesis=dict(arbitrary_rotations=0, approximation_error=0,
                                    method="exact standard CCX and CZ lowering; Qiskit opt0 fixed seed7",
                                    clifford_t_basis=opt.WALK_CLIFFORD_T_BASIS),
                     source_only_scalability_exports=source_only,
                     exclusions=["physical faults, code distance, factories, routing and physical timing (separate resource analysis)",
                                 "classical equality-check runtime after readout", "input-data acquisition beyond supplied target bits"],
                     provenance=info)
    with (output / "probability-scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scan[0]))
        writer.writeheader()
        writer.writerows(scan)
    with (output / "readout.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "shots", "marked_samples", "marked_sample_fraction", "exact_marked_probability",
                  "workspace_nonzero_samples", "hoeffding_half_width_95", "circuit_preparations", "oracle_calls",
                  "coin_calls", "shift_calls", "individual_qubit_measurements", "statevector_seconds", "sampling_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(readout)
    draw_one_step(output, dimension, marked)
    draw_probability_scan(output, scan, steps)
    run["workflow_wall_seconds"] = perf_counter() - started
    (output / "summary.md").write_text(summary_markdown(run, resources), encoding="utf-8")
    write_json(output / "runs.json", run)
    write_json(output / "resources.json", resources)
    return dict(output=str(output), marked_probability=main["marked_probability"],
                modeled_preparations=resources["full_readout_workload"]["circuit_preparations"],
                exact_t_count=per["t_count"], fixed_success_trials=repetition_trials,
                workflow_wall_seconds=run["workflow_wall_seconds"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "assets/generated/optimization/walk" if CHECKOUT else ROOT / "walk-results")
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--marked", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.output, args.dimension, args.marked, args.steps, args.export_only), indent=2))


if __name__ == "__main__":
    main()
