#!/usr/bin/env python3
"""Generate bounded oblivious amplification of the checked QSP U/4 block.

Rounds 0..3 use actual A/A†, ancilla-only reflections and final measurements.
No phase refit, package installation, physical estimator or remote job is run.
Standalone reproduction needs all copied modules plus the locked environment.
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
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT/"src"
REPOSITORY = (SOURCE/"openquantum/simulation/amplification.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector

if REPOSITORY:
    from openquantum.hamiltonian_simulation import spin_terms, dense_hamiltonian, pauli_matrix, observable_shots, ROTATION_BASIS
    from openquantum.simulation.lcu import postselection_attempts
    from openquantum.simulation.qsp_evolution import build_qsp_evolution
    from openquantum.simulation.amplification import (
        build_oblivious_amplification, zero_ancilla_reflection, decompose_amplification,
        compare_oblivious_amplification, projected_amplified_block, amplified_block_polynomial,
    )
else:
    from hamiltonian_simulation import spin_terms, dense_hamiltonian, pauli_matrix, observable_shots, ROTATION_BASIS
    from lcu import postselection_attempts
    from qsp_evolution import build_qsp_evolution
    from amplification import (
        build_oblivious_amplification, zero_ancilla_reflection, decompose_amplification,
        compare_oblivious_amplification, projected_amplified_block, amplified_block_polynomial,
    )


def dump_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_svg(path: Path):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = build_qsp_evolution(spin_terms(3))
    source_block = projected_amplified_block(source)
    initial = np.eye(8, dtype=complex)[:, 0]
    h = dense_hamiltonian(3)
    observable = pauli_matrix("IIZ")
    environment = {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in ["qiskit", "numpy", "scipy", "matplotlib"]}}
    rng = np.random.default_rng(17)
    full_inputs = []
    for _ in range(2):
        state = rng.normal(size=256)+1j*rng.normal(size=256)
        full_inputs.append(state/np.linalg.norm(state))
    rows, ledgers = [], []
    for rounds in range(4):
        circuit = build_oblivious_amplification(source, rounds)
        check = compare_oblivious_amplification(circuit, h, initial, observable)
        actual_block = projected_amplified_block(circuit)
        polynomial = amplified_block_polynomial(source_block, rounds)
        block_polynomial_error = float(np.max(np.abs(actual_block-polynomial)))
        if block_polynomial_error > 1e-12 or check["conditional_state_vector_error"] > 3e-8:
            raise AssertionError("actual oblivious circuit failed polynomial or conditional-state check")
        compiled = decompose_amplification(circuit)
        # Full 256x256 matrices are not materialized here. All data-basis
        # trajectories establish the accepted block, and two seeded full
        # ancilla/data states check source-vs-primitive equivalence numerically.
        equivalence_error = max(float(np.max(np.abs(Statevector(state).evolve(circuit).data-
                                                   Statevector(state).evolve(compiled).data)))
                                for state in full_inputs)
        if equivalence_error > 3e-12:
            raise AssertionError("source and primitive amplified circuits changed global phases")
        measured = compiled.copy()
        measured.measure_all()
        qasm = qasm3.dumps(measured)
        qasm_name = f"round-{rounds}.qasm"
        (output/qasm_name).write_text(qasm)
        counts = dict(compiled.count_ops())
        compiled_phase = float(compiled.global_phase)
        export_policy = (
            "Qiskit qasm3 export omits the compiled overall global phase. "
            "This measured-trial export preserves probabilities; it is not a "
            "phase-preserving coherently controlled oracle. In-memory circuits "
            "retain phase for the amplification and equivalence checks."
        )
        min_numerical_p = check["finite_H_numerical_minimum_acceptance_probability"]
        retry_cap = postselection_attempts(min_numerical_p, .01)
        accepted_shots = observable_shots(.03, .005)
        per_sample_cap = postselection_attempts(min_numerical_p, .005/accepted_shots)
        query_calls = 2*rounds+1
        access_calls = {
            "PREPARE_or_inverse_total": 14*query_calls,
            "SELECT_or_inverse_total": 14*query_calls,
            "unprepare_or_inverse_total": 14*query_calls,
            "controlled_PREPARE_or_inverse_total": query_calls,
            "controlled_SELECT_or_inverse_total": query_calls,
            "controlled_unprepare_or_inverse_total": query_calls,
            "uncontrolled_forward_SELECT": 14*(rounds+1), "uncontrolled_inverse_SELECT": 14*rounds,
            "controlled_forward_SELECT": rounds+1, "controlled_inverse_SELECT": rounds,
        }
        row = {
            **check,
            "max_entry_error_vs_independent_K_polynomial": block_polynomial_error,
            "source_vs_primitive_max_amplitude_error_on_two_full_seeded_states": equivalence_error,
            "source_vs_primitive_equivalence_scope": "two full8-qubit complex input states, seed17; accepted blocks checked on all8 data-basis inputs; no full 256x256 matrix identity claim",
            "total_A_or_inverse_calls": query_calls, "forward_A_calls": rounds+1, "inverse_A_calls": rounds,
            "zero_ancilla_reflections": 2*rounds,
            "primitive_unitary_gates_per_attempt": int(sum(counts.values())),
            "primitive_depth_per_attempt": compiled.depth(),
            "continuous_rotations_per_attempt": sum(counts.get(g, 0) for g in ["rx", "ry", "rz"]),
            "numerical_retry_cap_for_one_accepted_measured_trial_failure_01": retry_cap,
            "numerical_expected_attempts_for_input000": 1/check["actual_input_acceptance_probability"],
            "numerical_expected_primitive_gates_for_accepted_input000_trial": sum(counts.values())/check["actual_input_acceptance_probability"],
            "primitive_gate_counts": counts,
            "compiled_global_phase_radians": compiled_phase, "qasm_export_policy": export_policy,
            "qasm_file": qasm_name, "qasm_sha256": digest(qasm.encode()),
        }
        rows.append(row)
        ledger = {
            "rounds": rounds, "qubits": compiled.num_qubits, "n_system": 3, "n_access_ancillas": 5,
            "complete_source_calls": {"A": rounds+1, "A_inverse": rounds, "total": query_calls},
            "zero_ancilla_reflections": 2*rounds,
            "access_calls_including_all_A_and_inverse_circuits": access_calls,
            "branch_conditioned_projector_phase_calls": 31*query_calls,
            "round_global_pi_phases": rounds,
            "reflection_convention": "R=I-2Pi; only all 5 zero ancillas projected, identity on 3 unknown data qubits",
            "operator_convention": "(-A R A† R)^r A",
            "primitive_gate_counts_per_complete_attempt": counts,
            "primitive_unitary_gates_per_complete_attempt": int(sum(counts.values())),
            "primitive_depth_per_complete_attempt": compiled.depth(),
            "continuous_rotation_gates_per_attempt": row["continuous_rotations_per_attempt"],
            "known_T_and_Tdag_gates_before_continuous_rotation_synthesis": counts.get("t", 0)+counts.get("tdg", 0),
            "continuous_rotations_synthesized": False,
            "count_basis": ROTATION_BASIS,
            "measurement_gates_per_complete_attempt": int(measured.count_ops()["measure"]),
            "measurement_protocol": "measure all 8 wires in Z; accept all 5 access ancillas zero and retain the data bitstring for Z(q0); conditional quantum-state checks occur before final data readout",
            "input_preparation": "all 8 qubits assumed reset to zero; data |000> requires no additional X gates; source flag and PREPARE circuits included; arbitrary input preparation remains excluded",
            "postselection_cost_projection": {
                "certified": False, "basis": "numerical finite-H accepted-block singular values, after ideal continuous-angle amplification",
                "finite_H_numerical_minimum_probability": min_numerical_p,
                "numerical_input000_probability": check["actual_input_acceptance_probability"],
                "numerical_expected_attempts_input000": row["numerical_expected_attempts_for_input000"],
                "numerical_attempt_cap_for_one_accepted_measured_trial_failure_01": retry_cap,
                "deterministic_acceptance": False,
            },
            "observable_sampling_projection": {
                "performed": False, "statistical_error_target": .03, "statistical_failure_allowance": .005,
                "independent_accepted_Z_samples": accepted_shots,
                "postselection_failure_allowance_for_all_samples": .005,
                "per_sample_postselection_failure_allowance": .005/accepted_shots,
                "numerical_attempt_cap_per_accepted_sample": per_sample_cap,
                "numerical_total_attempt_cap": accepted_shots*per_sample_cap,
                "scope": "Hoeffding accepted-sample count and union restart projection, conditioned on numerical ideal acceptance; no certified physical completion guarantee",
            },
            "physical_resource_estimate": None,
            "prominent_exclusions": ["coherent continuous-angle synthesis and resulting amplified acceptance/state changes", "physical faults and reset/readout timing", "arbitrary input preparation", "classical coefficient/phase fitting cost from quantum gate counts"],
            "qasm_file": qasm_name, "qasm_sha256": row["qasm_sha256"],
            "compiled_global_phase_radians": compiled_phase, "qasm_export_policy": export_policy,
        }
        ledgers.append(ledger)
    source_files = ["amplification.py", "qsp_evolution.py", "qsp.py", "lcu.py", "hamiltonian_simulation.py", "generate_simulation_amplification.py"]
    hashes = {}
    for name in source_files:
        if REPOSITORY:
            path = HERE/name if name.startswith("generate_") else SOURCE/"openquantum"/(name if name == "hamiltonian_simulation.py" else "simulation/"+name)
        else:
            path = HERE/name
        data = path.read_bytes()
        (output/name).write_bytes(data)
        hashes[name] = digest(data)
    license_path = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    (output/"LICENSE.txt").write_bytes(license_path.read_bytes())
    report = {
        "status": "verified integer-round oblivious amplification of approximate complex QSP; round 3 acceptance about 0.9613, probabilistic",
        "instance": {"n_system": 3, "J": 1, "h": .7, "time": 1, "hbar": 1,
                     "boundary": "open chain", "source_block_target": "exp(-iH)/4", "initial_state": "|000>", "observable": "Z(q0)"},
        "rounds": rows,
        "ideal_model_scope": "sin²((2r+1)asin(1/4)) requires an exact scaled unitary K=U/4; actual QSP input dependence measured numerically",
        "coherent_error_scope": "source QSP polynomial approximation checked numerically; continuous rotation synthesis excluded; coherent error remains separate from stochastic failure",
        "acceptance_and_physical_scope": "finite-H acceptance/retries are numerical projections, uncertified; synthesis and physical faults require independent budgets",
        "sampling_scope": "state/observable expectations calculated from ideal circuit; finite-shot workload projected, not executed in this increment",
        "source_sha256": hashes, "environment": environment,
        "verification_seeds": {"full_state_source_primitive_equivalence": 17},
        "scale_limits": {"rounds": 4, "generator_total_qubits": 24, "source_top_level_instructions": 50000,
                         "primitive_export_gates": 200000, "dense_total_qubits": 8, "statevector_total_qubits": 12},
        "source_budget_scope": "50,000 top-level source instructions; nested definitions may increase lowering cost, with a separate 200,000 primitive-gate cap after lowering",
        "qasm_export_policy": "overall compiled global phase is omitted; measured-trial exports are valid but cannot serve as phase-preserving coherent controlled oracles without restoration",
        "verification_scope": "all8 data-basis trajectories for accepted blocks;two seeded full-state source/primitive checks; test suite separately compares full unitaries for tiny exact model",
    }
    dump_json(output/"results.json", report)
    dump_json(output/"logical-resources.json", {
        "resource_kind": "actual amplified primitive logical circuits before continuous rotation synthesis and error correction",
        "physical_resource_estimate": None, "physical_status": "pending validated synthesis and physical fault model; frozen Grover baseline unchanged",
        "rounds": ledgers, "source_sha256": hashes, "environment": environment,
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
    })
    columns = ["rounds", "ideal_scaled_unitary_success_probability", "actual_input_acceptance_probability",
               "finite_H_numerical_minimum_acceptance_probability", "finite_H_numerical_maximum_acceptance_probability",
               "total_A_or_inverse_calls", "zero_ancilla_reflections", "primitive_unitary_gates_per_attempt",
               "primitive_depth_per_attempt", "continuous_rotations_per_attempt", "conditional_state_vector_error",
               "observable_error", "numerical_retry_cap_for_one_accepted_measured_trial_failure_01",
               "numerical_expected_primitive_gates_for_accepted_input000_trial"]
    with (output/"rounds.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in columns} for row in rows)
    rounds_array = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(rounds_array, [r["ideal_scaled_unitary_success_probability"] for r in rows],
                 color="#c4cbcb", lw=4, label="Ideal exact U/4 model")
    axes[0].plot(rounds_array, [r["actual_input_acceptance_probability"] for r in rows],
                 color="#246c71", marker="o", lw=1.5, label="Actual QSP circuit, input |000>")
    axes[0].set(xlabel="Oblivious rounds r", ylabel="Probability of all ancillas zero", ylim=(0, 1.02), xticks=rounds_array)
    axes[0].legend(fontsize=8)
    axes[1].bar(rounds_array, [r["primitive_unitary_gates_per_attempt"] for r in rows], color="#9271a7")
    axes[1].set(xlabel="Oblivious rounds r", ylabel="Primitive gates per complete attempt", xticks=rounds_array)
    fig.suptitle("Oblivious amplification: acceptance and circuit cost")
    fig.tight_layout()
    fig.savefig(output/"acceptance-and-cost.svg", metadata={"Date": None})
    plt.close(fig)
    normalize_svg(output/"acceptance-and-cost.svg")
    reflection = zero_ancilla_reflection(3, 8)
    figure = reflection.draw(output="mpl", fold=32)
    figure.savefig(output/"zero-ancilla-reflection.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    normalize_svg(output/"zero-ancilla-reflection.svg")
    return {"output_dir": str(output), "rounds": [{name: row[name] for name in columns} for row in rows]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/amplification" if REPOSITORY else HERE/"generated-amplification")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
