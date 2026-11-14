#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded surface-code projections for digitized annealing and an SKW walk.

Each fixed main measured circuit is charged for all three recorded 4096-shot
readouts. Classical time/step studies have zero quantum preparations. The
walk's explicit Clifford+T sequence is fenced against one-qubit compiler
merging; a zero rotation-model allowance is used only if the estimated native
trace contains no arbitrary rotations and retains the literal T count.

Reproduce from a source checkout using the pinned environment and its shared
src/openquantum/resource_estimation.py adapter. The downloaded copy is source
for inspection or placement in checkout/scripts/; it is not a standalone
bundle and does not include the shared resource adapter.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
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

import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator, Statevector

ROOT = Path(__file__).resolve().parents[1]
if (not (ROOT / "src/openquantum/optimization.py").is_file()
        or not (ROOT / "src/openquantum/resource_estimation.py").is_file()):
    raise SystemExit("Run from source-checkout scripts/ with the shared resource adapter; the download is not a standalone bundle")
sys.path.insert(0, str(ROOT / "src"))
from openquantum.resource_estimation import (
    PINNED_QDK_VERSION,
    CircuitResourceBudget,
    SurfaceCodeProfile,
    estimate_circuit_resources,
    prepare_measured_circuit,
)

PROFILES = (
    SurfaceCodeProfile(name="frozen-p-1e-3", physical_error_rate=1e-3, gate_time_ns=100, measurement_time_ns=500),
    SurfaceCodeProfile(name="frozen-p-1e-4", physical_error_rate=1e-4, gate_time_ns=100, measurement_time_ns=500),
)
SEEDS = (7, 19, 41)
SHOTS_PER_SEED = 4096
TOTAL_PREPARATIONS = 12288
PHYSICAL_FAILURE_ALLOWANCE = 1e-3
ANNEALING_SYNTHESIS_ALLOWANCE = 1e-5
WALK_CLIFFORD_T = {"h", "x", "z", "s", "sdg", "t", "tdg", "cx"}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_current_implementation(annealing: dict, walk: dict):
    """Load the run-recorded implementation, including a preserved snapshot.

    Extending the live module does not invalidate earlier complete records.
    Both published source copies must match their common declared hash before
    a frozen copy is imported; mismatched or tampered inputs remain rejected.
    """
    source = ROOT / "src/openquantum/optimization.py"
    digest = annealing["provenance"]["implementation_sha256"]
    if walk["provenance"]["implementation_sha256"] != digest:
        raise ValueError("Both main records must identify the same executed implementation")
    if sha_file(source) != digest:
        recorded = [ROOT / "assets/generated/optimization" / method / "optimization.py"
                    for method in ("annealing", "walk")]
        if not all(path.is_file() and sha_file(path) == digest for path in recorded):
            raise ValueError("Both recorded source copies must match the executed implementation hash")
        source = recorded[0]
    spec = importlib.util.spec_from_file_location("annealing_walk_resource_source", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    if Path(module.__file__).resolve() != source.resolve():
        raise RuntimeError("Resource construction imported a different source checkout")
    return module, digest


def measured_map(circuit) -> list[list[int]]:
    result, begun = [], False
    for item in circuit.data:
        if item.operation.name == "measure":
            begun = True
            result.append([circuit.find_bit(item.qubits[0]).index, circuit.find_bit(item.clbits[0]).index])
        elif begun and item.operation.name != "barrier":
            raise ValueError("Only final measurements are supported")
    if (sorted(q for q, _ in result) != list(range(circuit.num_qubits))
            or len({c for _, c in result}) != circuit.num_qubits):
        raise ValueError("Every qubit must have one distinct final classical readout")
    return result


def verify_count_record(row: dict, *, width: int) -> None:
    if row["shots"] != SHOTS_PER_SEED or sum(row["counts"].values()) != SHOTS_PER_SEED:
        raise ValueError("Each fixed readout must contain all 4096 recorded shots")
    if any(len(bits) != width or set(bits) - {"0", "1"}
           or isinstance(number, bool) or not isinstance(number, int) or number <= 0
           for bits, number in row["counts"].items()):
        raise ValueError("Recorded bitstring counts are invalid")


def validate_record_provenance(record: dict, resources: dict) -> None:
    for key in ("implementation_sha256", "generator_sha256", "environment_lock_sha256"):
        if resources["provenance"][key] != record["provenance"][key]:
            raise ValueError("Run and logical resource provenance differ")


def validate_annealing_record(record: dict, resources: dict, implementation, folder: Path):
    validate_record_provenance(record, resources)
    task, main = record["task"], record["main"]
    if (task["n_qubits"] != 4 or sorted(map(tuple, task["edges"])) != sorted(implementation.CYCLE_EDGES)
            or task["driver"] != "-sum_i X_i" or task["problem"] != "0.5 sum_edges Z_u Z_v=2-C"
            or task["schedule"] != "s=t/T"):
        raise ValueError("Annealing selection does not match the stated C4 interpolation")
    if (main["total_time"] != 16.0 or main["steps"] != 256
            or record["target"]["predeclared_total_time"] != 16.0 or record["target"]["predeclared_steps"] != 256):
        raise ValueError("Annealing physical projection is for the predeclared T16,L256 circuit only")
    if tuple(row["seed"] for row in record["readout"]) != SEEDS:
        raise ValueError("Annealing readout seeds are not the three predeclared cases")
    source = implementation.build_annealing_circuit(4, implementation.CYCLE_EDGES, 16.0, 256, measure=True)
    expected_counts = {"h": 4, "rx": 2048, "rz": 1024, "cx": 2048, "measure": 4, "barrier": 1}
    if dict(source.count_ops()) != expected_counts:
        raise ValueError("Annealing source does not contain the complete expected Strang workload")
    cuts = implementation.maxcut_values(4, implementation.CYCLE_EDGES)
    for row in record["readout"]:
        verify_count_record(row, width=4)
        if (row["circuit_gate_counts"] != expected_counts or row["logical_qubits"] != 4
                or row["rng"]["entropy"] != [row["seed"], 4]
                or row["rng"]["spawn_key"] != [1, 0, 0]
                or row["measured_qubit_readouts"] != 4 * SHOTS_PER_SEED):
            raise ValueError("Annealing readout circuit or independent RNG allocation does not match")
        optimal = sum(number for bits, number in row["counts"].items() if cuts[int(bits, 2)] == 4)
        if optimal != row["optimal_count"]:
            raise ValueError("Annealing optimal-cut counts do not match the recorded outcomes")
    if record["total_preparations"] != TOTAL_PREPARATIONS or resources["total_preparations"] != TOTAL_PREPARATIONS:
        raise ValueError("Annealing repeated preparation total is incomplete")
    if resources["main_parameters"] != {"total_time": 16.0, "steps": 256}:
        raise ValueError("Annealing logical ledger describes a different main time/step choice")
    if (resources["logical_qubits"] != 4 or resources["per_preparation_gate_counts"] != expected_counts
            or resources["measured_qubit_readouts"] != 4 * TOTAL_PREPARATIONS):
        raise ValueError("Annealing logical resource record does not match the measured circuit")
    exports = resources["circuit_exports"]
    if len(exports) != 1 or exports[0]["file"] != "annealing.qasm":
        raise ValueError("Annealing resource input must identify its exact main export")
    exported = exports[0]
    compiled = implementation.compile_rotation_basis(source)
    if (sha_file(folder / exported["file"]) != exported["sha256"]
            or sha_text(qasm3.dumps(compiled)) != exported["sha256"]):
        raise ValueError("Annealing source parameters/export hashes are stale")
    selection = {
        "algorithm": "digitized linear annealing", "instance": "cycle4-maxcut", "logical_qubits": 4,
        "total_time": 16.0, "steps": 256, "evolution_time_units": "dimensionless Hamiltonian evolution time, hbar=1; not hardware seconds",
        "main_selection": "predeclared before time/step studies", "seeds": list(SEEDS), "shots_per_seed": SHOTS_PER_SEED,
        "total_preparations": TOTAL_PREPARATIONS,
        "source_gate_counts": expected_counts, "source_unitary_gate_count": 5124,
        "weighted_source_gate_counts": {gate: number * TOTAL_PREPARATIONS for gate, number in expected_counts.items() if gate != "barrier"},
        "classical_time_study_rows": len(record["time_study"]), "classical_step_study_rows": len(record["step_study"]),
        "time_step_study_quantum_shots": 0,
        "recorded_ideal_optimal_probability": main["optimal_probability"],
        "recorded_continuous_reference_optimal_probability": main["continuous_optimal_probability"],
        "recorded_state_infidelity_to_numerical_ode": main["state_infidelity"],
        "reference_error_scope": "state-specific numerical comparison; not a certified operator/adiabatic bound",
        "independent_readout_scope": "each fixed-state seeded experiment; no optimizer or adiabatic-time guarantee",
    }
    return source, selection


def validate_walk_record(record: dict, resources: dict, implementation, folder: Path):
    validate_record_provenance(record, resources)
    main = record["main"]
    layout = implementation.walk_register_layout(4)
    if (main["dimension"] != 4 or main["marked_vertex"] != 5 or main["steps"] != 4 or main["layout"] != layout):
        raise ValueError("Walk projection is for the predeclared d4,m5,k4 main circuit only")
    if resources["layout"] != layout:
        raise ValueError("Walk logical ledger describes a different active/work register layout")
    if tuple(row["seed"] for row in record["readout"]) != SEEDS:
        raise ValueError("Walk readout seeds are not the three predeclared cases")
    source = implementation.build_hypercube_walk_circuit(4, 5, 4, measure=True)
    expected_counts = {"x": 80, "ccx": 56, "h": 38, "z": 4, "cz": 4, "measure": 11, "barrier": 1}
    if dict(source.count_ops()) != expected_counts:
        raise ValueError("Walk source does not count every oracle, coin, shift and uncomputation")
    for row in record["readout"]:
        verify_count_record(row, width=11)
        if (row["circuit_preparations"] != SHOTS_PER_SEED or row["individual_qubit_measurements"] != 11 * SHOTS_PER_SEED
                or any(row[name] != 4 * SHOTS_PER_SEED for name in ("oracle_calls", "coin_calls", "shift_calls"))):
            raise ValueError("Walk readout repetitions omit oracle/coin/shift/readout work")
        position_counts = Counter()
        workspace_nonzero = 0
        for bits, count in row["counts"].items():
            index = int(bits, 2)
            position_counts[(index >> layout["coin_qubits"]) & 15] += count
            if index >> layout["active_qubits"]:
                workspace_nonzero += count
        if (workspace_nonzero != row["workspace_nonzero_samples"] or workspace_nonzero != 0
                or position_counts[5] != row["marked_samples"]
                or any(position_counts[p] != row["position_counts"][f"{p:04b}"] for p in range(16))):
            raise ValueError("Walk outcomes were filtered, misdecoded, or differ from the clean-work record")
    compiled = implementation.compile_walk_clifford_t(source)
    t_count = int(compiled.count_ops().get("t", 0) + compiled.count_ops().get("tdg", 0))
    if t_count != 392 or compiled.count_ops().get("cx", 0) != 340:
        raise ValueError("Expected literal standard-CCX Clifford+T lowering was not preserved")
    for kind, circuit in (("source", source), ("clifford_t", compiled)):
        exported = resources["circuits"][kind]
        if (exported["logical_qubits"] != 11 or exported["gate_counts"] != dict(circuit.count_ops())
                or sha_file(folder / exported["file"]) != exported["sha256"]
                or sha_text(qasm3.dumps(circuit)) != exported["sha256"]):
            raise ValueError("Walk source/Clifford+T resource export hashes are stale")
    workload = resources["full_readout_workload"]
    if (workload["circuit_preparations"] != TOTAL_PREPARATIONS or workload["individual_qubit_measurements"] != 11 * TOTAL_PREPARATIONS
            or workload["t_count"] != 392 * TOTAL_PREPARATIONS
            or workload["cx_count"] != 340 * TOTAL_PREPARATIONS
            or any(workload[name] != 4 * TOTAL_PREPARATIONS for name in ("oracle_calls", "coin_calls", "shift_calls"))):
        raise ValueError("Walk full repeated-work resource ledger is incomplete")
    selection = {
        "algorithm": "SKW-style coined hypercube search", "dimension": 4, "vertices": 16,
        "marked_vertex": 5, "steps": 4, "layout": layout, "logical_qubits": 11,
        "active_qubits": 6, "clean_workspace_qubits": 5,
        "seeds": list(SEEDS), "shots_per_seed": SHOTS_PER_SEED, "total_preparations": TOTAL_PREPARATIONS,
        "source_gate_counts": expected_counts,
        "source_ccx_per_preparation": 56, "oracle_and_chain_ccx_per_preparation": 40, "shift_ccx_per_preparation": 16,
        "literal_clifford_t_counts": dict(compiled.count_ops()), "literal_t_and_tdg_per_preparation": 392,
        "weighted_literal_clifford_t_counts": {gate: number * TOTAL_PREPARATIONS for gate, number in compiled.count_ops().items() if gate != "barrier"},
        "oracle_coin_shift_calls_all_preparations": 4 * TOTAL_PREPARATIONS,
        "individual_qubit_measurements": 11 * TOTAL_PREPARATIONS,
        "recorded_ideal_marked_probability": main["marked_probability"],
        "oracle_input_model": "explicit equality predicate for the supplied known marked vertex; teaching task, not database access",
        "no_workspace_postselection": True,
    }
    return source, compiled, selection


def seal_clifford_t(circuit):
    """No-op full-register compiler fences after every literal unitary gate."""
    measured_map(circuit)
    sealed = circuit.copy_empty_like()
    sealed.global_phase = circuit.global_phase
    measurement_started = False
    for item in circuit.data:
        name = item.operation.name
        if name == "barrier":
            continue
        if name == "measure":
            measurement_started = True
        elif name not in WALK_CLIFFORD_T or measurement_started:
            raise ValueError("Walk sealing requires an explicit Clifford+T circuit with final measurements")
        qubits = [sealed.qubits[circuit.find_bit(q).index] for q in item.qubits]
        clbits = [sealed.clbits[circuit.find_bit(c).index] for c in item.clbits]
        sealed.append(item.operation.copy(), qubits, clbits)
        if name != "measure":
            sealed.barrier()
    if measured_map(sealed) != measured_map(circuit):
        raise RuntimeError("Compiler fencing changed the final measured output map")
    return sealed


def phase_aligned_error(left: np.ndarray, right: np.ndarray) -> float:
    inner = np.vdot(left.ravel(), right.ravel())
    if abs(inner) == 0:
        return math.inf
    phase = inner / abs(inner)
    return float(np.max(np.abs(right - phase * left)))


def verify_annealing_lowering(source, lowered) -> dict:
    if measured_map(source) != measured_map(lowered) or source.num_qubits != lowered.num_qubits:
        raise ValueError("Annealing lowering changed width or measurement mapping")
    a = Operator(source.remove_final_measurements(inplace=False)).data
    b = Operator(lowered.remove_final_measurements(inplace=False)).data
    error = phase_aligned_error(a, b)
    if error > 1e-10:
        raise ValueError("Annealing lowering changed the complete small-system Operator")
    return {"full_4_qubit_operator_equivalence_verified": True, "up_to_global_phase": True,
            "max_phase_aligned_operator_element_error": error, "tolerance": 1e-10,
            "not_an_error_bound_against_continuous_adiabatic_evolution": True}


def verify_walk_lowering(source, compiled, sealed, lowered, implementation) -> dict:
    if (measured_map(source) != measured_map(compiled) or measured_map(source) != measured_map(sealed)
            or measured_map(source) != measured_map(lowered)):
        raise ValueError("Walk lowering or compiler fencing changed final readout mapping")
    if any(c.num_qubits != 11 for c in (source, compiled, sealed, lowered)):
        raise ValueError("Walk resource lowering lost active or clean work qubits")
    strip_barriers = lambda c: {k: int(v) for k, v in c.count_ops().items() if k != "barrier"}
    if strip_barriers(compiled) != strip_barriers(sealed):
        raise ValueError("Compiler fences altered the literal Clifford+T gate counts")
    # Independently check the local exact CCX decomposition on all eight basis
    # states. Composition of these fixed gates supplies the exact source
    # sequence; a full 2048x2048 walk Operator is not allocated here.
    ccx = QuantumCircuit(3)
    ccx.ccx(0, 1, 2)
    ccx_compiled = implementation.compile_walk_clifford_t(ccx)
    expected = np.eye(8)
    expected[:, [3, 7]] = expected[:, [7, 3]]
    local_error = phase_aligned_error(expected, Operator(ccx_compiled).data)
    if local_error > 1e-12 or ccx_compiled.count_ops().get("t", 0) + ccx_compiled.count_ops().get("tdg", 0) != 7:
        raise ValueError("Standard exact seven-T CCX decomposition did not match its independent permutation")
    a = Statevector.from_instruction(source.remove_final_measurements(inplace=False)).data
    b = Statevector.from_instruction(lowered.remove_final_measurements(inplace=False)).data
    error = phase_aligned_error(a, b)
    leakage = float(np.sum(np.abs(b[1 << 6:]) ** 2))
    if error > 1e-10 or leakage > 1e-12:
        raise ValueError("Selected prepared walk state or clean workspace changed under resource lowering")
    return {"local_ccx_full_operator_equivalence_verified": True, "local_ccx_t_count": 7,
            "local_ccx_max_phase_aligned_operator_error": local_error,
            "selected_full_11_qubit_prepared_state_equivalence_verified": True,
            "selected_state_max_phase_aligned_amplitude_error": error, "workspace_leakage": leakage,
            "source_literal_sequence_counts_preserved_by_fences": True,
            "shared_lowering_preserved_literal_counts": strip_barriers(compiled) == strip_barriers(lowered),
            "full_11_qubit_operator_verified": False,
            "compiler_fences_are_no_op_barriers": True,
            "global_phase_preserved": float(compiled.global_phase) == float(sealed.global_phase)}


def canonical_trace_audit(circuit) -> dict:
    from qdk.qre import instruction_name
    from qdk.qre.application import OpenQASMApplication
    if importlib.metadata.version("qdk") != PINNED_QDK_VERSION:
        raise RuntimeError("The pinned QDK resource trace contract does not apply")
    qasm = qasm3.dumps(circuit)
    traces = list(OpenQASMApplication(qasm).enumerate_traces())
    if len(traces) != 1:
        raise ValueError("Exactly one static measured-circuit trace is required")
    trace = traces[0]
    counts = {instruction_name(key): int(value) for key, value in trace.gate_counts.items()}
    return {"qasm_sha256": sha_text(qasm), "canonical_gate_counts": counts,
            "canonical_compute_qubits": trace.compute_qubits, "base_error": float(trace.base_error),
            "full_native_trace_json": json.loads(trace.to_json()),
            "scope": "QDK LogicalCounts cost abstraction; not a faithful original gate schedule/topology"}


def choose_walk_synthesis_allowance(audit: dict) -> tuple[float, str]:
    counts = audit["canonical_gate_counts"]
    if counts.get("RZ", 0) == 0 and counts.get("T", 0) == 392 and counts.get("CCX", 0) == 0:
        return 0.0, "explicit literal Clifford+T source sequence; no arbitrary-rotation model charge"
    return 1e-5, "source Clifford+T sequence present, but adapter introduced/changed rotation representation; conditional mean-fit projection"


def summarize(record: dict) -> str:
    lines = [
        "These are conditional surface-code/factory projections for two **fixed measured circuits**, each repeated "
        "12,288 times across the three predeclared 4,096-shot readouts. Annealing time/step studies are classical "
        "simulation checks with zero quantum-shot workload; the evolution parameter $T=16$ is not hardware wall time.", "",
        "The frozen profiles use physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements. "
        "Each algorithm receives whole-readout physical-fault allowance $10^{-3}$ across all 12,288 preparations. "
        "Digitized annealing separately receives arbitrary-rotation model scalar allowance $10^{-5}$ per execution, "
        "not a stochastic failure probability or angle certificate.", "",
        "| Fixed method | Physical error | Declared selection | All preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-readout physical-fault score |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in record["choices"]:
        lines.append(f"| {row['method']} | {row['physical_error_rate']:g} | {row['choice']} | {row['executions']:,} | "
                     f"{row['peak_physical_qubits']:,} | {row['serial_runtime_ns']/1e9:.6f} | {row['physical_failure_union_bound']:.3g} |")
    walk = record["methods"]["walk"]
    if walk["synthesis_model_allowance_per_execution"] == 0:
        lines += ["", "The walk supplies an explicit exact Clifford+T gate sequence: 56 source CCX gates use 392 T/T† gates "
                  "per preparation. No-op compiler fences preserve this sequence through shared lowering. The actual estimated "
                  "canonical trace has **T=392, arbitrary RZ=0, and 11 measurements**; it therefore uses zero arbitrary-rotation "
                  "model allowance. This is a logical sequence check, not a certification of hardware faults or a formal walk proof."]
    else:
        lines += ["", "The walk source supplies an explicit Clifford+T sequence, but the shared adapter did not preserve "
                  "the required zero-arbitrary-rotation canonical trace. Its resource rows therefore retain a positive conditional "
                  "mean-fit allowance; no exact-T-only adapter estimate is claimed."]
    lines += ["", "The annealing source counts preparation, all driver/problem layers and final measurements. "
              "The walk counts uniform preparation, the explicit known-target equality oracle and AND uncomputation, "
              "coin, shift, all five work qubits and final readout. The known target is a teaching input, not a database benchmark.", "",
              "Classical input/compilation/scoring time, physical initialization/reset, routing, communication and adaptive feedback "
              "are unmeasured or excluded from these quantum-model times. QEC/factory reliability remains conditional on the declared models. "
              "Digitization and state-specific adiabatic diagnostics are separate from sampling confidence, coherent synthesis-model "
              "allowance and physical faults; no generic spectral-gap, success, speedup or optimizer guarantee is inferred.", ""]
    return "\n".join(lines)


def generate(annealing_dir: Path, walk_dir: Path, output: Path, cache_dir: Path) -> dict:
    paths = {"annealing_runs": annealing_dir / "runs.json", "annealing_resources": annealing_dir / "resources.json",
             "walk_runs": walk_dir / "runs.json", "walk_resources": walk_dir / "resources.json"}
    input_bytes = {name: path.read_bytes() for name, path in paths.items()}
    data = {name: json.loads(content) for name, content in input_bytes.items()}
    implementation, source_digest = load_current_implementation(data["annealing_runs"], data["walk_runs"])
    anneal_source, anneal_selection = validate_annealing_record(data["annealing_runs"], data["annealing_resources"], implementation, annealing_dir)
    walk_source, walk_compiled, walk_selection = validate_walk_record(data["walk_runs"], data["walk_resources"], implementation, walk_dir)
    anneal_lowered = prepare_measured_circuit(anneal_source)
    walk_sealed = seal_clifford_t(walk_compiled)
    walk_lowered = prepare_measured_circuit(walk_sealed)
    anneal_equivalence = verify_annealing_lowering(anneal_source, anneal_lowered)
    walk_equivalence = verify_walk_lowering(walk_source, walk_compiled, walk_sealed, walk_lowered, implementation)
    anneal_audit = canonical_trace_audit(anneal_lowered)
    walk_audit = canonical_trace_audit(walk_lowered)
    if (anneal_audit["canonical_compute_qubits"] != 4 or walk_audit["canonical_compute_qubits"] != 11
            or anneal_audit["canonical_gate_counts"].get("MEAS_Z", 0) != 4
            or walk_audit["canonical_gate_counts"].get("MEAS_Z", 0) != 11):
        raise ValueError("Native resource trace lost modeled data/work qubits or readout")
    walk_allowance, walk_kind = choose_walk_synthesis_allowance(walk_audit)
    methods = {
        "annealing": {"selection": anneal_selection, "source": anneal_source, "estimated_source": anneal_source,
                      "lowered": anneal_lowered, "equivalence": anneal_equivalence, "canonical_audit": anneal_audit,
                      "synthesis_model_allowance_per_execution": ANNEALING_SYNTHESIS_ALLOWANCE,
                      "synthesis_semantics": "conditional arbitrary-angle mean-fit resource model; no certified sequence"},
        "walk": {"selection": walk_selection, "source": walk_source, "estimated_source": walk_sealed,
                 "lowered": walk_lowered, "equivalence": walk_equivalence, "canonical_audit": walk_audit,
                 "synthesis_model_allowance_per_execution": walk_allowance, "synthesis_semantics": walk_kind,
                 "explicit_sequence_generated_by": "original compile_walk_clifford_t preprocessor, not the resource adapter"},
    }
    output.mkdir(parents=True, exist_ok=True)
    for method, item in methods.items():
        item["qasm_artifacts"] = {}
        for kind, circuit in (("source", item["source"]), ("estimated-source", item["estimated_source"]), ("lowered", item["lowered"])):
            filename = method + "-" + kind + ".qasm"
            text = qasm3.dumps(circuit)
            (output / filename).write_text(text, encoding="utf-8")
            item["qasm_artifacts"][kind] = {"file": filename, "sha256": sha_text(text),
                                            "logical_qubits": circuit.num_qubits, "gate_counts": dict(circuit.count_ops())}
    reports, choices = [], []
    begin = perf_counter()
    for profile in PROFILES:
        for method, item in methods.items():
            budget = CircuitResourceBudget(synthesis_model_error_per_execution=item["synthesis_model_allowance_per_execution"],
                                          physical_failure_probability=PHYSICAL_FAILURE_ALLOWANCE, executions=TOTAL_PREPARATIONS)
            report = estimate_circuit_resources(item["estimated_source"], budget=budget, profile=profile, cache_dir=cache_dir, use_graph=False)
            if (report["qasm_sha256"] != item["canonical_audit"]["qasm_sha256"]
                    or report["canonical_gate_counts"] != item["canonical_audit"]["canonical_gate_counts"]):
                raise RuntimeError("Actual estimated QASM/trace differs from the verified resource input")
            if method == "walk" and walk_allowance == 0:
                if (report["canonical_gate_counts"].get("RZ", 0) != 0 or report["canonical_gate_counts"].get("T", 0) != 392
                        or report["synthesis"]["model_error_per_execution"] != 0):
                    raise RuntimeError("Zero rotation-model allowance claim was invalidated by the actual adapter trace")
            reports.append({"method": method, "profile": asdict(profile), "executions": TOTAL_PREPARATIONS, "report": report})
            for label, key in (("fewest physical qubits", "peak_physical_qubits"), ("shortest modeled runtime", "runtime_per_execution_ns")):
                point = min(report["pareto_points"], key=lambda p: (p[key], p["runtime_per_execution_ns"], p["peak_physical_qubits"]))
                if point["physical_failure_union_bound_all_executions"] > PHYSICAL_FAILURE_ALLOWANCE + 1e-12:
                    raise ValueError("Full fixed-method readout exceeds its task physical-fault allocation")
                choices.append({"method": method, "physical_error_rate": profile.physical_error_rate,
                                "choice": label, "executions": TOTAL_PREPARATIONS,
                                "peak_physical_qubits": point["peak_physical_qubits"],
                                "serial_runtime_ns": point["runtime_per_execution_ns"] * TOTAL_PREPARATIONS,
                                "physical_failure_union_bound": point["physical_failure_union_bound_all_executions"],
                                "synthesis_model_allowance_per_execution": item["synthesis_model_allowance_per_execution"],
                                "point": point})
    clean_methods = {name: {key: value for key, value in item.items() if key not in {"source", "estimated_source", "lowered"}}
                     for name, item in methods.items()}
    record = {"schema_version": 1, "status": "computed_conditional_model", "real_qdk_query_count": 4,
              "scope": "all recorded independent readouts of two predeclared fixed measured circuits; not time/step studies or hardware outcomes",
              "methods": clean_methods, "reports": reports, "choices": choices,
              "physical_failure_allowance_each_algorithm_all_preparations": PHYSICAL_FAILURE_ALLOWANCE,
              "synthesis_model_scalars_summed_over_shots_as_random_faults": False,
              "source_provenance": {"input_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in input_bytes.items()},
                                    "optimization_implementation_sha256": source_digest,
                                    "shared_resource_adapter_sha256": sha_file(ROOT / "src/openquantum/resource_estimation.py"),
                                    "resource_generator_sha256": sha_file(Path(__file__).resolve()),
                                    "environment_lock_sha256": sha_file(ROOT / "requirements-lock.txt"),
                                    "source_commit_at_generation": git_revision(ROOT),
                                    "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("qiskit", "qdk", "numpy")}},
              "projection_wall_seconds": perf_counter() - begin,
              "no_hardware_execution_or_success_certificate": True,
              "no_generic_gap_adiabatic_optimizer_or_speedup_guarantee": True,
              "excluded": ["physical initialization/reset, routing, communication and feedback", "classical compilation and input/scoring benchmarks",
                           "actual hardware noise, mitigation and experiment measurements", "angle approximation certificates for annealing",
                           "numerical-reference tolerances as certified global operator bounds", "factory/runtime-tail and hardware-reliability certificates"]}
    if (any(path.read_bytes() != input_bytes[name] for name, path in paths.items())
            or sha_file(Path(implementation.__file__)) != source_digest):
        raise RuntimeError("Input run records or source changed during projection; rerun after final generation")
    generator_file = output / "generate_optimization_annealing_walk_resources.py"
    generator_file.write_bytes(Path(__file__).resolve().read_bytes())
    license_file = output / "LICENSE.txt"
    license_file.write_bytes((ROOT / "LICENSE").read_bytes())
    record["downloads"] = {
        "resource_generator": {"file": generator_file.name, "sha256": sha_file(generator_file),
                               "reproduction_requirement": "source checkout scripts/ and the shared resource adapter; not standalone"},
        "license": {"file": license_file.name, "sha256": sha_file(license_file)},
    }
    write_json(output / "report.json", record)
    (output / "summary.md").write_text(summarize(record), encoding="utf-8")
    columns = ["method", "physical_error_rate", "choice", "executions", "peak_physical_qubits", "serial_runtime_ns",
               "physical_failure_union_bound", "synthesis_model_allowance_per_execution"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(choices)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annealing-dir", type=Path, default=ROOT / "assets/generated/optimization/annealing")
    parser.add_argument("--walk-dir", type=Path, default=ROOT / "assets/generated/optimization/walk")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/optimization/annealing-walk-resources")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qre-optimization-annealing-walk")
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    record = generate(args.annealing_dir, args.walk_dir, args.output_dir, args.cache_dir)
    print(json.dumps({"status": record["status"], "real_qdk_queries": record["real_qdk_query_count"],
                      "preparations_each_algorithm": TOTAL_PREPARATIONS,
                      "annealing_canonical_counts": record["methods"]["annealing"]["canonical_audit"]["canonical_gate_counts"],
                      "walk_canonical_counts": record["methods"]["walk"]["canonical_audit"]["canonical_gate_counts"],
                      "walk_rotation_model_allowance": record["methods"]["walk"]["synthesis_model_allowance_per_execution"],
                      "projection_wall_seconds": record["projection_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
