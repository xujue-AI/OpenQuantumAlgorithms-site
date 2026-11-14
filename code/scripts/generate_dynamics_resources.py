#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate bounded static dynamics circuits and conditional surface-code studies.

Canonical run: 18 product-formula cases, four QSP/OAA cases and two retargeted
QSP size cases, each under two frozen hardware profiles (48 actual QDK queries). Reset/feedback instruments
are excluded explicitly; they must not be forced into this static importer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time as timer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector
from scipy.linalg import expm

try:
    from source_provenance import git_revision
except ImportError:
    # Individual downloaded generators may omit the optional Git helper.
    def git_revision(root):
        return None

from openquantum.dynamics_resources import (
    BOUND_SOURCE, DISTANCE_DOMAIN, RETARGETED_PHASES, DynamicsTask, choose_steps, commutator_l1_bounds,
    estimate_static_dynamics_resources,
    postselected_contract, product_formula_error_bound, source_angle_rounding_bound,
)
from openquantum.hamiltonian_simulation import (
    compare_product_formula, dense_hamiltonian, pauli_matrix,
    product_formula_circuit, spin_terms,
)
from openquantum.resource_estimation import (
    BoundedObservableBudget, CircuitResourceBudget, SurfaceCodeProfile,
    prepare_measured_circuit,
)
from openquantum.simulation.amplification import (
    amplified_block_polynomial, build_oblivious_amplification,
    compare_oblivious_amplification, projected_amplified_block,
)
from openquantum.simulation.qsp_evolution import build_qsp_evolution
from openquantum.simulation.qsp import verify_cosine_phases
from openquantum.simulation.qsp_evolution import verify_sine_phases

MAX_CASES = 24
MAX_QUERIES = 48
PROFILES = (
    SurfaceCodeProfile("frozen-p-1e-3", 1e-3, 100, 500),
    SurfaceCodeProfile("frozen-p-1e-4", 1e-4, 100, 500),
)
EXCLUSIONS = [
    "Clean zero states are supplied; physical initialization/reset time and faults are not modeled",
    "Classical coefficient handling, compilation, acceptance/readout communication and verification latency are not quantum-model time",
    "QDK uses an empirical mean synthesis fit; no concrete Clifford+T sequence, channel error certificate or runtime-tail bound is supplied",
    "QDK abstracts Clifford scheduling; the source dependency depth is not a physical routed schedule",
    "Reset, intermediate measurement and dynamic-feedback instruments are not costed by this study",
]
FROZEN = (
    "src/openquantum/grover.py", "scripts/estimate_resources.py",
    "assets/generated/grover/circuit.qasm",
    "assets/generated/resources/grover-surface-code.json",
    "assets/generated/resources/grover-surface-code.csv",
)


def dump(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_provenance():
    names = [
        "src/openquantum/dynamics_resources.py", "scripts/generate_dynamics_resources.py",
        "src/openquantum/hamiltonian_simulation.py", "src/openquantum/resource_estimation.py",
        "src/openquantum/simulation/qsp.py", "src/openquantum/simulation/qsp_evolution.py",
        "src/openquantum/simulation/lcu.py", "src/openquantum/simulation/amplification.py",
        "requirements-lock.txt",
    ]
    return {
        "source_commit": git_revision(ROOT),
        "source_revision_scope": "Exact checkout root when Git is available; null for standalone packages. Content hashes identify actual source.",
        "source_sha256": {name: sha((ROOT / name).read_bytes()) for name in names},
        "environment": {"python": platform.python_version(), "device": "CPU",
                        "seed": 20261112,
                        "packages": {name: importlib.metadata.version(name)
                                     for name in ("qiskit", "qdk", "numpy", "scipy", "matplotlib")}},
    }


def freeze_hashes():
    return {name: sha((ROOT / name).read_bytes()) for name in FROZEN}


def guard_output(path: Path):
    destination = path.resolve()
    for relative in ("assets/generated/grover", "assets/generated/resources",
                     "assets/generated/hamiltonian-simulation", "assets/generated/simulation"):
        protected = (ROOT / relative).resolve()
        if destination == protected or protected in destination.parents:
            raise ValueError("Output must not overwrite existing/frozen algorithm artifacts")


def product_case(n: int, time: float = 1, order: int = 2, accuracy: float = .05):
    task = DynamicsTask.at_accuracy(accuracy)
    operator_allowance = Fraction(str(task.algorithm_bias)) / 2
    rounding_reserve = operator_allowance / 50
    steps = choose_steps(n, time, order, operator_allowance - rounding_reserve)
    start = timer.perf_counter()
    source = product_formula_circuit(n, time, steps, order)
    measured = product_formula_circuit(n, time, steps, order, measure=True)
    compiled = prepare_measured_circuit(measured)
    construction_seconds = timer.perf_counter() - start
    bound = product_formula_error_bound(n, time, steps, order)
    rounding = source_angle_rounding_bound(source, n=n, time=time, steps=steps, order=order)
    if bound + rounding > operator_allowance or rounding > rounding_reserve:
        raise AssertionError("Source-angle rounding exceeded the separately reserved allowance")
    start = timer.perf_counter()
    check = compare_product_formula(n, time, steps, order)
    lowering_error = float(np.linalg.norm(
        Operator(compiled.remove_final_measurements(inplace=False)).data - Operator(source).data, 2
    ))
    validation_seconds = timer.perf_counter() - start
    source_upper = math.nextafter(float(bound + rounding), math.inf)
    lowering_guard = 1e-10
    upper = source_upper + lowering_guard
    if check["operator_error"] > source_upper + 1e-10 or lowering_error > lowering_guard or 2 * upper > task.algorithm_bias:
        raise AssertionError("Actual generated/lowered circuit failed its independent numerical check")
    observable = BoundedObservableBudget(
        absolute_error_target=task.absolute_error,
        algorithm_bias_bound=2 * upper,
        synthesis_bias_allowance=task.synthesis_bias,
        sampling_error_allowance=task.sampling_error,
        sampling_failure_probability=task.sampling_failure,
        physical_failure_probability=task.physical_failure,
    )
    key = f"pf{order}-n{n}-t{time:g}-eps{accuracy:g}"
    metadata = {
        "case_id": key, "method": "product_formula", "order": order,
        "system_qubits": n, "logical_qubits": measured.num_qubits,
        "time": time, "J": 1, "h": .7, "hbar": 1, "boundary": "open",
        "steps": steps, "task": task.as_dict(),
        "initial_state": "0" * n, "observable": "Z(q0)",
        "terms": spin_terms(n), "source_gate_counts": dict(source.count_ops()),
        "source_depth": source.depth(), "compiled_gate_counts": dict(compiled.count_ops()),
        "compiled_depth": compiled.depth(),
        "analytic_error": {
            "commutator_Pauli_l1_bounds_exact": {key: str(value) for key, value in commutator_l1_bounds(n).items()},
            "product_formula_operator_bound_exact": str(bound),
            "source_angle_rounding_operator_bound_exact": str(rounding),
            "operator_bound_upper_float": upper,
            "Pauli_bias_bound_upper_float": 2 * upper,
            "numerical_lowering_guard_added": lowering_guard,
            "numerical_lowering_guard_is_certificate": False,
            "rounding_reserve": float(rounding_reserve),
            "scope": "Exact rational target coefficients and intended product formula; original binary64 RX/RZ rounding separately bounded; no synthesis/physical certificate",
            "source": BOUND_SOURCE,
        },
        "numerical_checks": {**check, "source_to_lowered_operator_error": lowering_error,
                             "certified": False, "performed": True},
        "complete_attempts": task.shots, "attempts_per_accepted_slot": 1,
        "accepted_slots": task.shots, "acceptance_probability": 1,
        "no_acceptance_failure_actual": 0,
        "cpu_timings_seconds": {"construction_and_lowering": construction_seconds,
                                "independent_dense_validation": validation_seconds},
        "measurement": "all n data qubits measured; bit0 gives the +/-1 mean",
        "input_preparation": "fresh clean zeros assumed; no system preparation gates for this input",
        "compiled_global_phase_radians": float(compiled.global_phase),
        "qasm_global_phase_scope": "Measured trial; overall exported phase is not a coherently controlled oracle specification",
        "exclusions": EXCLUSIONS,
    }
    return measured, compiled, metadata, observable.resource_budget(), observable


def qsp_cases():
    source = build_qsp_evolution(spin_terms(3))
    source_block = projected_amplified_block(source)
    h = dense_hamiltonian(3)
    initial = np.eye(8, dtype=complex)[:, 0]
    z = pauli_matrix("IIZ")
    task = DynamicsTask()
    rng = np.random.default_rng(20261112)
    random_input = rng.normal(size=256) + 1j * rng.normal(size=256)
    random_input /= np.linalg.norm(random_input)
    for rounds in range(4):
        start = timer.perf_counter()
        circuit = build_oblivious_amplification(source, rounds)
        check = compare_oblivious_amplification(circuit, h, initial, z)
        block = projected_amplified_block(circuit)
        polynomial_error = float(np.linalg.norm(block - amplified_block_polynomial(source_block, rounds), 2))
        amplitude = check["ideal_scaled_unitary_amplified_amplitude"]
        eta = check["block_error_vs_ideal_amplitude_times_exact_U"]
        numerical_bias_upper = 4 * eta / (abs(amplitude) - eta)
        if numerical_bias_upper > task.algorithm_bias or polynomial_error > 1e-11:
            raise AssertionError("Actual accepted QSP/OAA block failed the declared bounded numerical accuracy")
        p_min = check["finite_H_numerical_minimum_acceptance_probability"]
        p_guarded = p_min - 1e-9
        contract = postselected_contract(p_guarded, task)
        measured = circuit.copy()
        measured.measure_all()
        compiled = prepare_measured_circuit(measured)
        equivalence_error = float(np.max(np.abs(
            Statevector(random_input).evolve(circuit).data
            - Statevector(random_input).evolve(compiled.remove_final_measurements(inplace=False)).data
        )))
        if equivalence_error > 1e-10:
            raise AssertionError("Lowered QSP/OAA circuit failed the seeded full-state equivalence check")
        budget = CircuitResourceBudget(
            contract["synthesis_model_error_per_attempt"], task.physical_failure,
            contract["maximum_complete_attempts"],
        )
        metadata = {
            "case_id": f"qsp-oaa{rounds}-n3-t1-eps0.05", "method": "QSP_oblivious_amplification",
            "system_qubits": 3, "logical_qubits": circuit.num_qubits,
            "time": 1, "J": 1, "h": .7, "hbar": 1, "phase_tau": 4.1,
            "task": task.as_dict(), "rounds": rounds,
            "initial_state": "000", "observable": "Z(q0)",
            "source_gate_counts": dict(circuit.count_ops()), "source_depth": circuit.depth(),
            "compiled_gate_counts": dict(compiled.count_ops()), "compiled_depth": compiled.depth(),
            "access": {key: circuit.metadata[key] for key in (
                "forward_A_calls", "inverse_A_calls", "total_A_or_inverse_calls",
                "zero_ancilla_reflections", "uncontrolled_block_encoding_calls",
                "controlled_block_encoding_calls", "branch_conditioned_projector_phase_calls",
            )},
            "numerical_checks": {**check, "accepted_block_polynomial_error": polynomial_error,
                                 "all_input_conditional_Pauli_bias_projection": numerical_bias_upper,
                                 "seeded_full_state_lowering_error": equivalence_error,
                                 "acceptance_guard_subtracted": 1e-9,
                                 "guard_is_certificate": False},
            "postselection_contract": contract,
            "complete_attempts": contract["maximum_complete_attempts"],
            "attempts_per_accepted_slot": contract["attempts_per_accepted_slot"],
            "accepted_slots": task.shots,
            "cpu_timings_seconds": {"construction_validation_lowering": timer.perf_counter() - start},
            "measurement": "all 8 wires measured; accept only zero outcomes on all five success ancillas; bit0 gives +/-1",
            "input_preparation": "fresh clean zeros assumed on all 8 wires for each complete attempt; all rejected schedules charged",
            "compiled_global_phase_radians": float(compiled.global_phase),
            "qasm_global_phase_scope": "Measured trial; omitted overall phase does not specify a coherently controlled oracle",
            "size_scaling_scope": "This is the n3,t1,alpha*t4.1 amplification slice; n2/n4 retargeted circuits are separate cases, and other n/time require new phases and approximation evidence",
            "exclusions": EXCLUSIONS,
        }
        yield measured, compiled, metadata, budget, None


def projected_retargeted_block(circuit):
    """Bounded 4/16 basis-input checks, not a full 9-wire unitary matrix."""
    n = circuit.metadata["n_system"]
    if (isinstance(n, bool) or not isinstance(n, int) or n not in (2, 4)
            or not n <= circuit.num_qubits <= 9):
        raise ValueError("Retargeted accepted-block validation is bounded to n=2/4 and at most 9 total wires")
    dimension = 1 << n
    columns = []
    for basis in range(dimension):
        initial = np.zeros(1 << circuit.num_qubits, dtype=complex)
        initial[basis] = 1
        columns.append(Statevector(initial).evolve(circuit).data[:dimension])
    return np.array(columns).T


def retargeted_qsp_case(n):
    if isinstance(n, bool) or not isinstance(n, int) or n not in RETARGETED_PHASES:
        raise ValueError("Retargeted cases are bounded to n=2 or n=4")
    phases = RETARGETED_PHASES[n]
    started = timer.perf_counter()
    source = build_qsp_evolution(spin_terms(n), time=1,
                                 cosine_phases=phases["cosine"], sine_phases=phases["sine"],
                                 phase_tau=phases["tau"],
                                 phase_verification_tolerance=phases["constructor_tolerance"])
    circuit = build_oblivious_amplification(source, 0)
    measured = circuit.copy()
    measured.measure_all()
    compiled = prepare_measured_circuit(measured)
    source_block = projected_retargeted_block(circuit)
    compiled_unitary = compiled.remove_final_measurements(inplace=False)
    compiled_unitary.metadata = circuit.metadata.copy()
    block = projected_retargeted_block(compiled_unitary)
    target = expm(-1j*dense_hamiltonian(n))
    initial = np.eye(1 << n, dtype=complex)[:, 0]
    observable = pauli_matrix("I"*(n-1)+"Z")
    output = block @ initial
    probability = float(np.vdot(output, output).real)
    conditional = output / math.sqrt(probability)
    reference = target @ initial
    amplitude = .25
    eta = float(np.linalg.norm(block-amplitude*target, 2))
    singular = np.linalg.svd(block, compute_uv=False)
    p_min = float(singular[-1]**2)
    lowering_error = float(np.linalg.norm(block-source_block, 2))
    numerical_bias_upper = 4*(eta+1e-10)/(amplitude-eta-1e-10)
    task = DynamicsTask()
    if numerical_bias_upper > task.algorithm_bias or lowering_error > 1e-10:
        raise AssertionError("Retargeted actual accepted blocks failed the fixed accuracy/lowering checks")
    rng = np.random.default_rng(20261112+n)
    random = rng.normal(size=1 << circuit.num_qubits) + 1j*rng.normal(size=1 << circuit.num_qubits)
    random /= np.linalg.norm(random)
    full_state_error = float(np.max(np.abs(Statevector(random).evolve(circuit).data
                                          - Statevector(random).evolve(compiled_unitary).data)))
    if full_state_error > 1e-10:
        raise AssertionError("Retargeted seeded full-register lowering check failed")
    contract = postselected_contract(p_min-1e-9, task)
    metadata = {
        "case_id": f"qsp-retargeted-n{n}-t1-eps0.05", "method": "QSP_retargeted",
        "system_qubits": n, "logical_qubits": circuit.num_qubits,
        "time": 1, "J": 1, "h": .7, "hbar": 1, "phase_tau": phases["tau"],
        "task": task.as_dict(), "rounds": 0,
        "initial_state": "0"*n, "observable": "Z(q0)",
        "source_gate_counts": dict(circuit.count_ops()), "source_depth": circuit.depth(),
        "compiled_gate_counts": dict(compiled.count_ops()), "compiled_depth": compiled.depth(),
        "phase_evidence": {
            "cosine_phases": list(phases["cosine"]), "sine_phases": list(phases["sine"]),
            "cosine": {**verify_cosine_phases(phases["cosine"], phases["tau"]), "certified": False},
            "sine": verify_sine_phases(phases["sine"], phases["tau"]),
            "constructor_tolerance": phases["constructor_tolerance"],
            "phase_fit_method": "deterministic analytic-Jacobian continuation from frozen tau4.1, maxstep.2/max500evals/per-component60s",
            "certified": False,
        },
        "access": {key: circuit.metadata[key] for key in (
            "forward_A_calls", "inverse_A_calls", "total_A_or_inverse_calls",
            "zero_ancilla_reflections", "uncontrolled_block_encoding_calls",
            "controlled_block_encoding_calls", "branch_conditioned_projector_phase_calls",
        )},
        "numerical_checks": {
            "accepted_block_basis_trajectories_per_source_or_lowering": 1 << n,
            "compiled_block_error_vs_exact_U_over4": eta,
            "source_to_compiled_accepted_block_operator_error": lowering_error,
            "seeded_full_state_lowering_error": full_state_error,
            "numerical_lowering_guard_added": 1e-10,
            "all_input_conditional_Pauli_bias_projection": numerical_bias_upper,
            "actual_input_acceptance_probability": probability,
            "finite_H_numerical_minimum_acceptance_probability": p_min,
            "finite_H_numerical_maximum_acceptance_probability": float(singular[0]**2),
            "conditional_state_vector_error": float(np.linalg.norm(conditional-reference)),
            "observable_exact": float(np.vdot(reference, observable@reference).real),
            "observable_conditional": float(np.vdot(conditional, observable@conditional).real),
            "observable_error": abs(float(np.vdot(conditional, observable@conditional).real)
                                    - float(np.vdot(reference, observable@reference).real)),
            "acceptance_guard_subtracted": 1e-9, "guard_is_certificate": False,
        },
        "postselection_contract": contract, "complete_attempts": contract["maximum_complete_attempts"],
        "attempts_per_accepted_slot": contract["attempts_per_accepted_slot"], "accepted_slots": task.shots,
        "cpu_timings_seconds": {"construction_validation_lowering": timer.perf_counter()-started},
        "measurement": "All wires measured; accept only zero success ancillas, bit0 gives +/-1; every failed complete schedule charged",
        "input_preparation": "Fresh clean zero states on all wires assumed; physical initialization not modeled",
        "compiled_global_phase_radians": float(compiled.global_phase),
        "qasm_global_phase_scope": "Measured trial; omitted overall phase does not specify a coherently controlled oracle",
        "size_scaling_scope": "n2,n3,n4 at fixed J1,h.7,t1 and output allowance.05; retargeted14/15phases independently checked, no generic phase/size certificate",
        "exclusions": EXCLUSIONS,
    }
    budget = CircuitResourceBudget(contract["synthesis_model_error_per_attempt"], task.physical_failure,
                                   contract["maximum_complete_attempts"])
    return measured, compiled, metadata, budget, None


def add_point_metrics(point: dict, metadata: dict):
    attempts = metadata["complete_attempts"]
    return {**point,
            "maximum_complete_attempts": attempts,
            "peak_times_capped_runtime_qubit_seconds": point["peak_physical_qubits"] * point["serial_runtime_all_executions_ns"] / 1e9,
            "spacetime_scope": "rectangular peak-times-capped-time proxy, not integrated time-dependent occupancy"}


def select_points(points: list[dict]):
    return {
        "fewest_qubits": min(points, key=lambda p: (p["peak_physical_qubits"], p["serial_runtime_all_executions_ns"])),
        "shortest_runtime": min(points, key=lambda p: (p["serial_runtime_all_executions_ns"], p["peak_physical_qubits"])),
        "smallest_peak_time_proxy": min(points, key=lambda p: p["peak_times_capped_runtime_qubit_seconds"]),
    }


def canonical_product_parameters():
    rows = {(n, 1., order, .05) for n in range(2, 7) for order in (1, 2)}
    rows.update((3, t, order, .05) for t in (.5, 2.) for order in (1, 2))
    rows.update((3, 1., order, accuracy) for accuracy in (.025, .1) for order in (1, 2))
    return sorted(rows)


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output: Path, summary: list[dict]):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    selectors = (("system_qubits", "System spins n", lambda r: r["time"] == 1 and r["accuracy"] == .05),
                 ("time", "Time (J=1 units)", lambda r: r["system_qubits"] == 3 and r["accuracy"] == .05),
                 ("accuracy", "Requested absolute Z accuracy", lambda r: r["system_qubits"] == 3 and r["time"] == 1))
    for column, (xkey, xlabel, select) in enumerate(selectors):
        for profile, linestyle in (("frozen-p-1e-3", "-"), ("frozen-p-1e-4", "--")):
            for order, color in ((1, "#bb592a"), (2, "#007f86")):
                rows = sorted((r for r in summary if r["method"] == "product_formula" and r["profile"] == profile
                               and r["order"] == order and r["choice"] == "fewest_qubits" and select(r)), key=lambda r: r[xkey])
                label = f"PF{order}, p={profile.removeprefix('frozen-p-')}"
                for row_index, ykey in ((0, "peak_physical_qubits"), (1, "capped_runtime_seconds")):
                    axes[row_index, column].plot([r[xkey] for r in rows], [r[ykey] for r in rows],
                                                marker="o", color=color, linestyle=linestyle, label=label)
                    axes[row_index, column].set(xlabel=xlabel, yscale="log")
                    axes[row_index, column].grid(alpha=.2)
    axes[0, 0].set_ylabel("Conditional peak physical qubits")
    axes[1, 0].set_ylabel("Conditional capped serial seconds")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Actual product-formula circuits: conditional surface-code projections")
    fig.savefig(output / "product-formula-scaling.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for profile, color in (("frozen-p-1e-3", "#bb592a"), ("frozen-p-1e-4", "#007f86")):
        rows = sorted((r for r in summary if r["method"] == "QSP_oblivious_amplification"
                       and r["profile"] == profile and r["choice"] == "fewest_qubits"), key=lambda r: r["rounds"])
        for axis, ykey, ylabel in zip(axes, ("peak_physical_qubits", "capped_runtime_seconds", "peak_time_qubit_seconds"),
                                      ("Peak physical qubits", "Capped serial seconds", "Peak × capped time (qubit s)"), strict=True):
            axis.plot([r["rounds"] for r in rows], [r[ykey] for r in rows], marker="o", color=color,
                      label=f"physical p={profile.removeprefix('frozen-p-')}")
            axis.set(xlabel="Integer oblivious-amplification rounds", ylabel=ylabel, yscale="log", xticks=range(4))
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Same three-spin Z task: conditional QSP/OAA complete-attempt models")
    fig.savefig(output / "qsp-amplification-resources.svg")
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for profile, color in (("frozen-p-1e-3", "#bb592a"), ("frozen-p-1e-4", "#007f86")):
        rows = sorted((r for r in summary if r["method"] in ("QSP_oblivious_amplification", "QSP_retargeted")
                       and r["rounds"] == 0 and r["profile"] == profile and r["choice"] == "fewest_qubits"),
                      key=lambda r: r["system_qubits"])
        for axis, key, label in zip(axes, ("peak_physical_qubits", "capped_runtime_seconds", "peak_time_qubit_seconds"),
                                    ("Peak physical qubits", "Capped serial seconds", "Peak × capped time (qubit s)"), strict=True):
            axis.plot([r["system_qubits"] for r in rows], [r[key] for r in rows], marker="o", color=color,
                      label=f"physical p={profile.removeprefix('frozen-p-')}")
            axis.set(xlabel="System spins n", ylabel=label, yscale="log", xticks=[2, 3, 4])
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Actual retargeted QSP circuits: fixed J=1, h=.7, t=1; conditional complete-attempt models")
    fig.savefig(output / "qsp-size-resources.svg")
    plt.close(fig)
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def generate(output: Path, *, estimate: bool = True, cache_dir: Path | None = None):
    guard_output(output)
    before = freeze_hashes()
    output.mkdir(parents=True, exist_ok=True)
    # A failed rerun must not leave a previous computed manifest or plots next
    # to newly rewritten circuits. Per-case report files are replaced below.
    for filename in ("manifest.json", "summary.md", "physical-summary.csv",
                     "product-formula-scaling.svg", "qsp-amplification-resources.svg", "qsp-size-resources.svg"):
        (output / filename).unlink(missing_ok=True)
    provenance = source_provenance()
    cache_dir = cache_dir or ROOT / ".cache/dynamics-resources"
    results, flat, queries = [], [], 0
    cases = [product_case(*args) for args in canonical_product_parameters()]
    cases.extend(qsp_cases())
    cases.extend(retargeted_qsp_case(n) for n in (2, 4))
    if len(cases) != MAX_CASES:
        raise AssertionError("Canonical case coverage changed")
    for measured, compiled, metadata, budget, observable in cases:
        case_id = metadata["case_id"]
        directory = output / case_id
        directory.mkdir(parents=True, exist_ok=True)
        qasm = qasm3.dumps(compiled)
        (directory / "measured-trial.qasm").write_text(qasm)
        metadata["qasm_sha256"] = sha(qasm.encode())
        metadata["provenance"] = provenance
        metadata["physical_reports"] = []
        for profile in PROFILES:
            if not estimate:
                (directory / f"{profile.name}.json").unlink(missing_ok=True)
                continue
            queries += 1
            if queries > MAX_QUERIES:
                raise AssertionError("Bounded QDK query limit exceeded")
            start = timer.perf_counter()
            report = estimate_static_dynamics_resources(measured, budget=budget, profile=profile,
                                                       cache_dir=cache_dir, observable_budget=observable)
            if report["qasm_sha256"] != metadata["qasm_sha256"]:
                raise AssertionError("Costed QASM does not match the published actual circuit")
            report["pareto_points"] = [add_point_metrics(p, metadata) for p in report["pareto_points"]]
            report["dynamics_task"] = metadata["task"]
            report["postselection_contract"] = metadata.get("postselection_contract")
            report["exclusions"] = EXCLUSIONS
            report["projection_wall_seconds"] = timer.perf_counter() - start
            filename = f"{profile.name}.json"
            dump(directory / filename, report)
            metadata["physical_reports"].append(filename)
            for choice, point in select_points(report["pareto_points"]).items():
                flat.append({
                    "case_id": case_id, "method": metadata["method"], "profile": profile.name,
                    "order": metadata.get("order", ""), "rounds": metadata.get("rounds", ""),
                    "system_qubits": metadata["system_qubits"], "logical_qubits": metadata["logical_qubits"],
                    "time": metadata["time"], "accuracy": metadata["task"]["absolute_error"],
                    "steps": metadata.get("steps", ""), "attempt_cap_per_slot": metadata["attempts_per_accepted_slot"],
                    "complete_attempts": metadata["complete_attempts"], "choice": choice,
                    "peak_physical_qubits": point["peak_physical_qubits"],
                    "data_code_distance": point["data_code_distance"],
                    "capped_runtime_seconds": point["serial_runtime_all_executions_ns"] / 1e9,
                    "peak_time_qubit_seconds": point["peak_times_capped_runtime_qubit_seconds"],
                    "modeled_physical_fault_union_bound": point["physical_failure_union_bound_all_executions"],
                    "synthesis_model_error_per_attempt": report["synthesis"]["model_error_per_execution"],
                    "certified_confidence": "unavailable",
                })
            print(f"Computed {case_id}, {profile.name}: {len(report['pareto_points'])} conditional points", flush=True)
        dump(directory / "logical-resources.json", metadata)
        results.append(metadata)
    after = freeze_hashes()
    if before != after:
        raise AssertionError("Frozen baseline files were modified")
    manifest = {
        "schema_version": 1, "status": "computed_conditional_model" if estimate else "logical_only",
        "canonical_cases": len(cases), "actual_physical_queries": queries,
        "product_formula_cases": 18, "QSP_OAA_cases": 4, "QSP_retargeted_size_cases": 2,
        "profiles": [asdict(p) for p in PROFILES], "provenance": provenance,
        "common_data_and_factory_distance_domain": list(DISTANCE_DOMAIN),
        "search_domain_change": "Explicit odd distances3..41 for every dynamics method; native defaults3..25 failed the full unamplified-QSP p1e-3 workload without relaxing any error/retry budget",
        "budget_scope": "Each case/profile is a separate proposed task; no simultaneous 1% guarantee across the sweep",
        "frozen_file_sha256_before": before, "frozen_file_sha256_after": after,
        "frozen_files_unchanged": before == after,
        "algorithm_evidence": "rational product-formula bounds with original angle-rounding accounting; QSP/OAA accepted blocks are bounded numerical checks",
        "conditional_confidence_scope": "declared ideal algorithm/acceptance evidence plus unverified mean-synthesis-TV and physical-model assumptions",
        "certified_confidence_lower_bound": None,
        "QSP_physical_size_sweep": (
            "Computed n2,n3,n4 at fixed J1,h.7,t1; retargeted14/15phases, numerical circuit evidence and conditional physical model, not generic phase/size certification"
            if estimate else "Not computed; n2,n3,n4 logical circuit checks at fixed J1,h.7,t1, with no physical projection"),
        "dynamic_instrument_physical_estimation": "excluded; no reset or intermediate-measurement work was omitted from an imported trial",
        "included": ["explicit Hamiltonian gates and input basis preparation", "every complete QSP/OAA source/inverse/reflection call",
                     "all final data and ancillary measurements", "all capped attempts including rejected trials",
                     "surface-code layout and magic-state factories in the conditional QDK model"],
        "exclusions": EXCLUSIONS,
        "sources": [BOUND_SOURCE, {"QDK_synthesis": "https://arxiv.org/abs/2203.10064"}],
        "cases": results,
    }
    dump(output / "manifest.json", manifest)
    if flat:
        write_csv(output / "physical-summary.csv", flat)
        plot_results(output, flat)
    else:
        for filename in ("physical-summary.csv", "product-formula-scaling.svg", "qsp-amplification-resources.svg", "qsp-size-resources.svg"):
            (output / filename).unlink(missing_ok=True)
    table = ["| Method | Cases | Measured circuit/approximation evidence | Physical evidence |",
             "|---|---:|---|---|",
             f"| Product formulas | 18 | Actual n=2–6 size cases, t=.5/1/2 and accuracy .025/.05/.1 slices; analytic step bounds | {36 if estimate else 0} conditional profile queries |",
             f"| Complex QSP and integer OAA | 4 | Three-spin t=1 accepted blocks, rounds 0–3; every failed attempt charged | {8 if estimate else 0} conditional profile queries; three-spin amplification slice |",
             f"| Retargeted complex QSP | 2 additional | n2/n4 at fixed J1,h.7,t1; all actual accepted-block basis columns checked, alongside n3 above | {4 if estimate else 0} conditional profile queries; bounded three-size slice |",
             "", "The mean synthesis fit is not an error certificate. Initialization/reset and classical communication costs remain excluded."]
    (output / "summary.md").write_text("\n".join(table) + "\n")
    for relative in ("src/openquantum/dynamics_resources.py", "scripts/generate_dynamics_resources.py"):
        (output / Path(relative).name).write_bytes((ROOT / relative).read_bytes())
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/dynamics-resources")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/dynamics-resources")
    parser.add_argument("--no-estimate", action="store_true", help="Bounded logical checks only; does not publish physical results")
    args = parser.parse_args()
    result = generate(args.output_dir, estimate=not args.no_estimate, cache_dir=args.cache_dir)
    print(json.dumps({"status": result["status"], "cases": result["canonical_cases"], "queries": result["actual_physical_queries"]}))


if __name__ == "__main__":
    main()
