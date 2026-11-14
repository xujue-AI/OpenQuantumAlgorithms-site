#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a bounded QSP sweep at fixed total Pauli norm, with logical trial costs.

The open-chain family has n=1..5, J=4.1/((n-1)+0.7*n), h=0.7*J and t=1.
It keeps alpha=4.1, rather than local coupling, fixed so the existing frozen
cosine/sine phases remain applicable. This conditional logical scaling study
does not establish physical advantage or fixed-local-coupling scaling.

Use Python 3.12 and the locked project environment. Standalone reproduction
requires the copied implementation modules, this script and LICENSE.txt.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/simulation/qsp_evolution.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-qsp-scaling"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3, transpile
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

if REPOSITORY:
    from openquantum import hamiltonian_simulation as hs
    from openquantum.simulation import qsp_evolution as qe
    from openquantum.simulation.lcu import postselection_attempts
else:  # Import copied local files explicitly, never an installed openquantum.
    import hamiltonian_simulation as hs
    import qsp_evolution as qe
    from lcu import postselection_attempts

MAX_SWEEP_SYSTEM_QUBITS = 5
MAX_PRIMITIVE_GATES = 200_000
ALPHA = 4.1
TIME = 1.0
STATISTICAL_ERROR = 0.03
STATISTICAL_FAILURE = 0.005
WHOLE_WORKLOAD_ACCEPTANCE_FAILURE = 0.005
SOURCE_NAMES = (
    "hamiltonian_simulation.py", "qsp_evolution.py", "qsp.py", "lcu.py",
    "generate_simulation_qsp_scaling.py",
)
EXCLUSIONS = [
    "fresh initialization/reset gates and initialization/reset time; every attempt assumes all wires start at zero",
    "arbitrary rotation synthesis and synthesis-induced conditional-state/acceptance changes",
    "physical faults, error correction, code cycles, factories, physical qubits and runtime",
    "physical scheduling and measurement/readout duration",
    "classical coefficient/phase preprocessing, rejection bookkeeping and readout/postprocessing time",
    "amplitude amplification and deterministic evolution",
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scaling_parameters(n_system: int) -> dict:
    """Parameters for the family, without a dense Hamiltonian allocation."""
    if (isinstance(n_system, bool) or not isinstance(n_system, int)
            or not 1 <= n_system <= qe.MAX_SYSTEM_QUBITS):
        raise ValueError(f"family requires 1..{qe.MAX_SYSTEM_QUBITS} integer system qubits")
    coupling = ALPHA / ((n_system - 1) + 0.7 * n_system)
    field = 0.7 * coupling
    terms = hs.spin_terms(n_system, coupling, field)
    norm = float(sum(abs(c) for c, _ in terms))
    if not math.isclose(norm, ALPHA, abs_tol=1e-12, rel_tol=0):
        raise AssertionError("scaling family did not preserve total Pauli norm")
    return {
        "n_system": n_system, "J": coupling, "h": field,
        "time": TIME, "hbar": 1, "boundary": "open chain", "alpha": norm,
        "term_count": len(terms), "terms": [[float(c), p] for c, p in terms],
        "input_state": "|" + "0" * n_system + ">", "observable": "Z(q0)",
    }


def workload_projection(probability: float, measured_counts: dict[str, int],
                        logical_qubits: int, measured_depth: int) -> dict:
    """Charge complete measured trials under a state-specific numerical model."""
    accepted_samples = hs.observable_shots(STATISTICAL_ERROR, STATISTICAL_FAILURE)
    per_slot_failure = WHOLE_WORKLOAD_ACCEPTANCE_FAILURE / accepted_samples
    retry_cap = postselection_attempts(probability, per_slot_failure)
    total_attempts = accepted_samples * retry_cap
    log_slot_failure = -math.inf if probability == 1 else retry_cap * math.log1p(-probability)
    slot_failure = math.exp(log_slot_failure)
    stopped_attempts_per_slot = -math.expm1(log_slot_failure) / probability
    return {
        "statistical_absolute_error": STATISTICAL_ERROR,
        "statistical_failure_allowance": STATISTICAL_FAILURE,
        "required_independent_accepted_Z_samples": accepted_samples,
        "acceptance_failure_allowance_all_samples": WHOLE_WORKLOAD_ACCEPTANCE_FAILURE,
        "per_sample_acceptance_failure_allowance": per_slot_failure,
        "state_specific_numerical_acceptance_probability": probability,
        "complete_attempt_cap_per_accepted_sample": retry_cap,
        "complete_attempt_cap_all_samples": total_attempts,
        "expected_complete_attempts_all_samples": accepted_samples / probability,
        "expected_complete_attempts_all_samples_uncapped": accepted_samples / probability,
        "expected_complete_attempts_all_samples_stopped_at_cap": accepted_samples * stopped_attempts_per_slot,
        "expected_complete_attempts_all_samples_scope": "legacy alias for uncapped geometric expectation M/p",
        "stopped_expectation_scope": "M*(1-(1-p)^R)/p; every requested slot stops on acceptance or cap R, including exhausted slots; all M slots are attempted, without early abort or conditioning on workload success",
        "modeled_no_acceptance_probability_per_slot": slot_failure,
        "union_bound_acceptance_failure_all_samples": accepted_samples * slot_failure,
        "primitive_gate_counts_all_capped_attempts": {
            gate: int(count) * total_attempts for gate, count in measured_counts.items()
        },
        "primitive_operations_all_capped_attempts": sum(measured_counts.values()) * total_attempts,
        "logical_qubit_layer_volume_all_capped_attempts": logical_qubits * measured_depth * total_attempts,
        "depth_scope": "measured circuit dependency depth, including final measurement layer; no physical schedule",
        "retry_protocol": "each slot restarts the complete fresh zero-input measured trial until accepted or capped; charge all rejected trials in full",
        "acceptance_model_scope": "numerical ideal acceptance for the fixed zero input only; not a worst-case input or hardware guarantee",
        "volume_scope": "logical qubit times dependency layer count; not physical code-cycle, runtime or factory accounting",
        "coherent_error_is_statistical_failure": False,
        "physical_failure_allowance": None,
        "certified": False,
    }


def _validate(circuit, compiled, parameters: dict) -> dict:
    n_system = parameters["n_system"]
    dimension = 2 ** n_system
    initial = np.zeros(dimension, dtype=complex)
    initial[0] = 1
    hamiltonian = hs.dense_hamiltonian(n_system, parameters["J"], parameters["h"])
    observable = hs.pauli_matrix("I" * (n_system - 1) + "Z")
    if circuit.num_qubits <= qe.MAX_DENSE_TOTAL_QUBITS:
        checks = qe.compare_qsp_evolution(circuit, hamiltonian, initial, observable)
        checks["operator_validation_performed"] = True
        checks["validation_scope"] = "actual accepted block/operator plus fixed-zero-input conditional state and observable"
        decomposition_error = float(np.max(np.abs(Operator(circuit).data - Operator(compiled).data)))
        checks["max_entry_error_source_vs_primitive_operator"] = decomposition_error
        checks["source_vs_primitive_check_scope"] = "full Operator without global-phase alignment"
        checks["max_entry_error_source_vs_primitive_zero_input_state"] = None
        if checks["encoded_block_error"] > 3e-9:
            raise AssertionError("actual accepted QSP block failed independent expm reference")
    else:
        state, probability = qe.postselected_evolution_state(circuit, initial)
        reference = expm(-1j * TIME * hamiltonian) @ initial
        actual_z = float(np.vdot(state, observable @ state).real)
        reference_z = float(np.vdot(reference, observable @ reference).real)
        full_source = Statevector.from_instruction(circuit).data
        full_compiled = Statevector.from_instruction(compiled).data
        decomposition_error = float(np.max(np.abs(full_source - full_compiled)))
        checks = {
            "operator_validation_performed": False,
            "validation_scope": "fixed-zero-input conditional Statevector and Z only; full accepted block/operator not computed",
            "encoded_block_error": None, "decoded_linear_map_operator_error": None,
            "accepted_block_singular_values": None,
            "finite_H_numerical_minimum_acceptance_probability": None,
            "finite_H_numerical_maximum_acceptance_probability": None,
            "conditional_state_check_scope": "accepted data state before final system readout",
            "conditional_state_vector_error": float(np.linalg.norm(state - reference)),
            "conditional_state_infidelity": float(max(0.0, 1 - abs(np.vdot(reference, state)) ** 2)),
            "observable_exact": reference_z, "observable_conditional": actual_z,
            "observable_error": abs(actual_z - reference_z), "acceptance_probability": probability,
            "max_entry_error_source_vs_primitive_operator": None,
            "max_entry_error_source_vs_primitive_zero_input_state": decomposition_error,
            "source_vs_primitive_check_scope": "full Statevector on the fixed zero input without global-phase alignment; not operator equivalence",
            "acceptance_certified": False,
        }
    if checks["conditional_state_vector_error"] > 2e-8 or checks["observable_error"] > 3e-8:
        raise AssertionError("QSP conditional state or Z failed independent expm reference")
    if decomposition_error > 2e-12:
        raise AssertionError("primitive decomposition failed its stated phase-preserving numerical check")
    checks["classical_reference"] = "independent Kronecker Pauli Hamiltonian and scipy.linalg.expm(-1j*H)"
    checks["state_validation_performed"] = True
    checks["certified"] = False
    return checks


def decompose_trial(circuit):
    """Pinned primitives without assuming reusable ancillary inputs are zero."""
    compiled = transpile(circuit, basis_gates=hs.ROTATION_BASIS, optimization_level=1,
                         seed_transpiler=7, qubits_initially_zero=False)
    if set(compiled.count_ops()) - set(hs.ROTATION_BASIS):
        raise ValueError("unexpected operations after primitive trial decomposition")
    return compiled


def _unvalidated() -> dict:
    return {
        "operator_validation_performed": False, "state_validation_performed": False,
        "validation_scope": "circuit materialized only; --no-sim skips all quantum circuit and dense-H state/operator checks",
        "encoded_block_error": None, "decoded_linear_map_operator_error": None,
        "accepted_block_singular_values": None,
        "finite_H_numerical_minimum_acceptance_probability": None,
        "finite_H_numerical_maximum_acceptance_probability": None,
        "conditional_state_vector_error": None, "conditional_state_infidelity": None,
        "observable_exact": None, "observable_conditional": None,
        "observable_error": None, "acceptance_probability": None,
        "max_entry_error_source_vs_primitive_operator": None,
        "max_entry_error_source_vs_primitive_zero_input_state": None,
        "source_vs_primitive_check_scope": "not executed",
        "acceptance_certified": False, "certified": False,
    }


def _source_bundle(output: Path) -> dict:
    hashes = {}
    for name in SOURCE_NAMES:
        if not REPOSITORY:
            source = HERE / name
        elif name.startswith("generate_"):
            source = HERE / name
        elif name == "hamiltonian_simulation.py":
            source = SOURCE / "openquantum" / name
        else:
            source = SOURCE / "openquantum/simulation" / name
        data = source.read_bytes()
        (output / name).write_bytes(data)
        hashes[name] = digest(data)
    license_path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    license_bytes = license_path.read_bytes()
    (output / "LICENSE.txt").write_bytes(license_bytes)
    hashes["LICENSE.txt"] = digest(license_bytes)
    return hashes


def _save_svg(figure, path: Path) -> None:
    figure.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def _plots(output: Path, rows: list[dict]) -> None:
    sizes = [row["n_system"] for row in rows]
    simulated = all(row["state_validation_performed"] for row in rows)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    curves = [
        ("logical_qubits", "Logical qubits", "Peak logical width", "#24686e"),
        ("primitive_operations_per_measured_attempt", "Primitive operations", "Measured trial gates", "#845896"),
        ("measured_dependency_depth", "Dependency layers", "Measured trial depth", "#996335"),
        (("logical_qubit_layer_volume_all_capped_attempts" if simulated else "arbitrary_rotations_per_attempt"),
         ("Logical qubit-layers" if simulated else "Arbitrary rotations"),
         ("Capped Z workload" if simulated else "Measured trial rotations"), "#315e99"),
    ]
    for axis, (key, label, title, color) in zip(axes.flat, curves):
        axis.plot(sizes, [row[key] for row in rows], marker="o", color=color)
        axis.set(ylabel=label, title=title, xticks=sizes)
        axis.grid(alpha=0.2)
        if key == "logical_qubit_layer_volume_all_capped_attempts":
            axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for axis in axes[1]:
        axis.set_xlabel("System spins n")
    fig.suptitle("QSP logical scaling at fixed total Pauli norm alpha = 4.1\nLocal couplings vary; physical synthesis and timing excluded", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_svg(fig, output / "scaling.svg")

    if not simulated:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    axes[0].semilogy(sizes, [row["conditional_state_vector_error"] for row in rows], marker="o", color="#24686e", label="Conditional state-vector error")
    axes[0].semilogy(sizes, [row["Z_error"] for row in rows], marker="s", color="#845896", label="Absolute Z(q0) error")
    axes[0].set(ylabel="Error against independent expm")
    axes[0].legend(fontsize=9)
    axes[1].plot(sizes, [row["acceptance_probability"] - 1 / 16 for row in rows], marker="o", color="#996335", label="Actual zero-input ideal acceptance residual")
    axes[1].axhline(0, color="#bec9ca", linestyle="--", label="Exact U/4 residual = 0")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useOffset=False)
    axes[1].set(xlabel="System spins n", ylabel="p - 1/16", title="Acceptance is approximately 0.0625 for every size", xticks=sizes)
    axes[1].legend(fontsize=9)
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Fixed total Pauli norm; n=4,5 checked on zero-input states only\nFloating numerical evidence, not certified bounds", fontsize=11)
    fig.tight_layout()
    _save_svg(fig, output / "accuracy-acceptance.svg")


def generate(output: Path, *, max_primitive_gates: int = MAX_PRIMITIVE_GATES,
             qubits: int | None = None, no_sim: bool = False) -> dict:
    if (isinstance(max_primitive_gates, bool) or not isinstance(max_primitive_gates, int)
            or not 1 <= max_primitive_gates <= MAX_PRIMITIVE_GATES):
        raise ValueError(f"primitive gate cap must be an integer in 1..{MAX_PRIMITIVE_GATES}")
    sizes = list(range(1, MAX_SWEEP_SYSTEM_QUBITS + 1)) if qubits is None else [qubits]
    for size in sizes:
        scaling_parameters(size)
    if not isinstance(no_sim, bool):
        raise ValueError("no_sim must be a boolean")
    if not no_sim and max(sizes) > MAX_SWEEP_SYSTEM_QUBITS:
        raise ValueError(f"this sweep simulates at most {MAX_SWEEP_SYSTEM_QUBITS} system qubits; use --no-sim to materialize larger circuits")
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["qiskit", "numpy", "scipy", "matplotlib"]},
        "machine_architecture": platform.machine(),
        "backend": "CPU Qiskit Operator/Statevector complex128 and SciPy dense expm; no GPU",
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7, "basis_gates": hs.ROTATION_BASIS,
                          "qubits_initially_zero": False},
        "simulation_random_seed": None,
        "simulation_seed_scope": "deterministic exact ideal-state calculations; no Monte Carlo sampling",
    }
    source_commit = git_revision(ROOT)
    family = {
        "system_sizes_materialized": sizes,
        "system_sizes_simulated": [] if no_sim else sizes,
        "system_sizes_operator_validated": [] if no_sim else [size for size in sizes if size <= 3],
        "J_n": "4.1/((n-1)+0.7*n)", "h_n": "0.7*J_n", "time": TIME,
        "alpha": ALPHA, "term_count": "2*n-1", "index_width": "ceil(log2(2*n-1))",
        "logical_width": "n+ceil(log2(2*n-1))+2",
        "norm_scope": "constant total Pauli coefficient norm, not fixed local coupling; n=3 reproduces J=1,h=0.7",
        "interpretation": "conditional logical scaling for this changing-coupling family; no physical advantage or fixed-local-coupling scaling claim",
    }
    bounds = {
        "executed_sweep_max_system_qubits": MAX_SWEEP_SYSTEM_QUBITS,
        "sweep_primitive_operation_cap_including_final_measurements_excluding_barriers": max_primitive_gates,
        "full_operator_max_total_qubits": qe.MAX_DENSE_TOTAL_QUBITS,
        "statevector_max_total_qubits": qe.MAX_STATEVECTOR_QUBITS,
        "independent_dense_hamiltonian_max_system_qubits": hs.MAX_DENSE_QUBITS,
        "scalable_constructor_max_system_qubits": qe.MAX_SYSTEM_QUBITS,
        "scalable_constructor_max_pauli_terms": qe.MAX_TERMS,
        "scalable_constructor_max_phase_degree": qe.MAX_PHASE_DEGREE,
        "scalable_constructor_max_source_operations": qe.MAX_SOURCE_OPERATIONS,
        "generator_scope": "scalable build_qsp_evolution constructs explicit access without dense H; this bounded sweep separately simulates its five outputs",
    }
    instances, ledgers, rows, qasm_hashes = [], [], [], {}
    for n_system in sizes:
        parameters = scaling_parameters(n_system)
        circuit = qe.build_qsp_evolution([(c, p) for c, p in parameters["terms"]], time=TIME)
        compiled = decompose_trial(circuit)
        measured = compiled.copy()
        measured.measure_all()
        measured_counts = {str(gate): int(count) for gate, count in sorted(measured.count_ops().items()) if gate != "barrier"}
        primitive_operations = sum(measured_counts.values())
        if primitive_operations > max_primitive_gates:
            raise ValueError(f"n={n_system} measured circuit exceeds primitive gate cap {max_primitive_gates}")
        checks = _unvalidated() if no_sim else _validate(circuit, compiled, parameters)
        n_index = int(circuit.metadata["n_index"])
        if (parameters["term_count"] != 2 * n_system - 1
                or n_index != (2 * n_system - 2).bit_length()
                or circuit.num_qubits != n_system + n_index + 2):
            raise AssertionError("materialized circuit widths disagree with scaling family")
        qasm_name = f"evolution-n{n_system}.qasm"
        qasm_bytes = qasm3.dumps(measured).encode()
        (output / qasm_name).write_bytes(qasm_bytes)
        qasm_hashes[qasm_name] = digest(qasm_bytes)
        projection = None if no_sim else workload_projection(checks["acceptance_probability"], measured_counts, compiled.num_qubits, measured.depth())
        unitary_counts = {str(gate): int(count) for gate, count in sorted(compiled.count_ops().items())}
        ledger = {
            "n_system": n_system, "n_index": n_index, "n_flag": 1, "n_selector": 1,
            "peak_logical_qubits": compiled.num_qubits,
            "resource_kind": "measured primitive logical circuit before arbitrary-rotation synthesis and error correction",
            "source_gate_counts_per_attempt": {str(gate): int(count) for gate, count in sorted(circuit.count_ops().items())},
            "source_gate_count_scope": "top-level circuit operations, including composite PREPARE/SELECT/access and controlled projector gates; not primitive cost",
            "source_dependency_depth_per_attempt": circuit.depth(),
            "decomposed_unitary_gate_counts_per_attempt": unitary_counts,
            "decomposed_unitary_gate_count_per_attempt": sum(unitary_counts.values()),
            "decomposed_unitary_dependency_depth_per_attempt": compiled.depth(),
            "measured_primitive_gate_counts_per_attempt": measured_counts,
            "primitive_operations_per_measured_attempt": primitive_operations,
            "measured_dependency_depth_per_attempt": measured.depth(),
            "continuous_rotation_count_per_attempt": sum(unitary_counts.get(gate, 0) for gate in ["rx", "ry", "rz"]),
            "continuous_rotations_synthesized": False,
            "transpiler_qubits_initially_zero": False,
            "final_measurements_per_attempt": int(measured_counts["measure"]),
            "barrier_count_per_attempt": int(measured.count_ops().get("barrier", 0)),
            "ordinary_access_calls_per_attempt": {"PREPARE": 14, "SELECT": 14, "unprepare": 14},
            "controlled_access_calls_per_attempt": {"controlled_PREPARE": 1, "controlled_SELECT": 1, "controlled_unprepare": 1},
            "branch_conditioned_projector_phase_calls_per_attempt": int(circuit.metadata["branch_conditioned_projector_phase_calls"]),
            "access_scope": "all actual PREPARE, signed Pauli SELECT, inverse PREPARE and branch phases are included in the measured primitive counts",
            "input_preparation": "all wires assumed freshly initialized to zero each attempt; zero input requires zero X gates; flag/selector H gates and access preparation included",
            "measurement_protocol": f"measure all {compiled.num_qubits} wires in Z; accept all {n_index + 2} ancillary outcomes zero; use system bit q0 for accepted Z",
            "conditional_state_scope": "normalized accepted system state before final system measurement",
            "qasm_file": qasm_name, "qasm_sha256": qasm_hashes[qasm_name],
            "source_global_phase_radians": float(circuit.global_phase),
            "compiled_global_phase_radians": float(compiled.global_phase),
            "qasm_global_phase_exported": False,
            "qasm_phase_scope": "Qiskit OpenQASM 3 exporter omits overall global phase; valid for this measured trial's probabilities, not a phase-preserving controlled oracle",
            "observable_workload_projection": projection,
            "prominent_exclusions": EXCLUSIONS,
            "physical_resource_estimate": None, "certified": False,
        }
        instances.append({"parameters": parameters, "checks": checks, "qasm_file": qasm_name})
        ledgers.append(ledger)
        rows.append({
            "n_system": n_system, "J": parameters["J"], "h": parameters["h"], "alpha": parameters["alpha"],
            "term_count": parameters["term_count"], "n_index": n_index, "n_flag": 1, "n_selector": 1,
            "logical_qubits": compiled.num_qubits, "source_operations_per_attempt": circuit.size(),
            "source_dependency_depth": circuit.depth(),
            "primitive_unitary_gates_per_attempt": sum(unitary_counts.values()),
            "primitive_operations_per_measured_attempt": primitive_operations,
            "unitary_dependency_depth": compiled.depth(), "measured_dependency_depth": measured.depth(),
            "arbitrary_rotations_per_attempt": ledger["continuous_rotation_count_per_attempt"],
            "final_measurements_per_attempt": compiled.num_qubits,
            "ordinary_access_calls_per_attempt": 14, "controlled_access_calls_per_attempt": 1,
            "branch_conditioned_projector_phase_calls_per_attempt": ledger["branch_conditioned_projector_phase_calls_per_attempt"],
            "conditional_state_vector_error": checks["conditional_state_vector_error"],
            "conditional_state_infidelity": checks["conditional_state_infidelity"],
            "Z_exact": checks["observable_exact"], "Z_conditional": checks["observable_conditional"],
            "Z_error": checks["observable_error"], "acceptance_probability": checks["acceptance_probability"],
            "operator_validation_performed": checks["operator_validation_performed"],
            "state_validation_performed": checks["state_validation_performed"],
            "encoded_block_error": checks["encoded_block_error"],
            "decoded_linear_map_operator_error": checks["decoded_linear_map_operator_error"],
            "accepted_Z_samples": None if no_sim else projection["required_independent_accepted_Z_samples"],
            "retry_cap_per_accepted_sample": None if no_sim else projection["complete_attempt_cap_per_accepted_sample"],
            "complete_attempt_cap_all_samples": None if no_sim else projection["complete_attempt_cap_all_samples"],
            "expected_complete_attempts_all_samples_uncapped": None if no_sim else projection["expected_complete_attempts_all_samples_uncapped"],
            "expected_complete_attempts_all_samples_stopped_at_cap": None if no_sim else projection["expected_complete_attempts_all_samples_stopped_at_cap"],
            "primitive_operations_all_capped_attempts": None if no_sim else projection["primitive_operations_all_capped_attempts"],
            "logical_qubit_layer_volume_all_capped_attempts": None if no_sim else projection["logical_qubit_layer_volume_all_capped_attempts"],
            "compiled_global_phase_radians": ledger["compiled_global_phase_radians"], "certified": False,
        })
    source_hashes = _source_bundle(output)
    provenance = {
        "source_git_commit_at_generation": source_commit,
        "source_git_commit_scope": "repository HEAD only; SHA256 identifies the exact copied source bytes, including uncommitted worker changes",
        "source_sha256": source_hashes, "qasm_sha256": qasm_hashes,
        "environment": environment,
    }
    write_json(output / "results.json", {
        "status": ("materialized fixed-total-Pauli-norm complex QSP circuits; state/operator validation not executed" if no_sim
                   else "executed bounded fixed-total-Pauli-norm complex QSP sweep; state checks on listed sizes, full operators only where total wires <=8"),
        "family": family, "bounds": bounds, "instances": instances,
        "phase_error_contract": qe.evolution_error_contract(),
        "prominent_exclusions": EXCLUSIONS, "certified": False, **provenance,
    })
    write_json(output / "logical-resources.json", {
        "family": family, "bounds": bounds, "instances": ledgers,
        "prominent_exclusions": EXCLUSIONS, "physical_resource_estimate": None,
        "certified": False, **provenance,
    })
    with (output / "scaling.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _plots(output, rows)
    # Remove only names generated by this script that the current run replaces.
    current_qasm_names = set(qasm_hashes)
    for size in range(1, qe.MAX_SYSTEM_QUBITS + 1):
        stale = output / f"evolution-n{size}.qasm"
        if stale.name not in current_qasm_names and stale.exists():
            stale.unlink()
    for name in ["logical-scaling.svg"] + (["accuracy-acceptance.svg"] if no_sim else []):
        stale = output / name
        if stale.exists():
            stale.unlink()
    return {"output_dir": str(output), "rows": rows, "source_sha256": source_hashes, "qasm_sha256": qasm_hashes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/simulation/qsp-scaling" if REPOSITORY else HERE / "generated-qsp-scaling")
    parser.add_argument("--max-primitive-gates", type=int, default=MAX_PRIMITIVE_GATES,
                        help=f"bounded measured-operation cap; at most {MAX_PRIMITIVE_GATES}")
    parser.add_argument("--qubits", type=int, help="materialize one family size instead of the default 1..5 sweep")
    parser.add_argument("--no-sim", action="store_true", help="materialize/decompose/export only; no state/operator checks or acceptance workload projection")
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output_dir, max_primitive_gates=arguments.max_primitive_gates,
                              qubits=arguments.qubits, no_sim=arguments.no_sim), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
