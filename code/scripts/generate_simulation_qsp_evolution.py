#!/usr/bin/env python3
"""Generate checked postselected QSP U(t)/4 circuits and qualified logical costs.

Run with the pinned Python3.12 environment. Optional --refit-sine performs bounded
classical phase preprocessing. Standalone downloads need all copied Python modules,
LICENSE.txt, and the project environment lock. No physical estimate is fabricated.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT/"src"
REPOSITORY = (SOURCE/"openquantum/simulation/qsp_evolution.py").exists()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

if REPOSITORY:
    from openquantum.hamiltonian_simulation import spin_terms, dense_hamiltonian, pauli_matrix, decompose, ROTATION_BASIS, observable_shots
    from openquantum.simulation.lcu import postselection_attempts
    from openquantum.simulation.qsp import COSINE_PHASES, COSINE_TAU
    from openquantum.simulation.qsp_evolution import (
        SINE_PHASES, build_qsp_evolution, compare_qsp_evolution, evolution_error_contract,
        fit_sine_phases, complex_qsp_response,
    )
else:
    from hamiltonian_simulation import spin_terms, dense_hamiltonian, pauli_matrix, decompose, ROTATION_BASIS, observable_shots
    from lcu import postselection_attempts
    from qsp import COSINE_PHASES, COSINE_TAU
    from qsp_evolution import (
        SINE_PHASES, build_qsp_evolution, compare_qsp_evolution, evolution_error_contract,
        fit_sine_phases, complex_qsp_response,
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path, refit_sine: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sine_phases = SINE_PHASES
    sine_fit = {
        "method": "scipy.optimize.least_squares", "seed": 31, "fit_nodes": 80,
        "successful_restart_zero_based": 4, "function_evaluations_successful_restart": 14,
        "node_residual": 8.881784197001252e-16, "maximum_restarts": 8,
        "maximum_function_evaluations_per_restart": 4000,
        "ftol": 1e-13, "gtol": 1e-13, "xtol": 1e-13, "performed_this_run": False,
    }
    if refit_sine:
        start = time.perf_counter()
        sine_phases, details = fit_sine_phases()
        sine_fit = {**details, "performed_this_run": True, "wall_seconds": time.perf_counter()-start}
    circuit = build_qsp_evolution(spin_terms(3), sine_phases=sine_phases)
    h = dense_hamiltonian(3)
    initial = np.eye(8, dtype=complex)[:, 0]
    observable = pauli_matrix("IIZ")
    check = compare_qsp_evolution(circuit, h, initial, observable)
    if check["encoded_block_error"] > 3e-9 or check["conditional_state_vector_error"] > 2e-8:
        raise AssertionError("actual default circuit failed complex evolution check")
    contract = evolution_error_contract(sine_phases=sine_phases)
    compiled = decompose(circuit)
    decomposition_error = float(np.max(np.abs(Operator(circuit).data-Operator(compiled).data)))
    if decomposition_error > 2e-12:
        raise AssertionError("decomposition changed the controlled branch/global phases")
    measured = compiled.copy()
    measured.measure_all()
    qasm = qasm3.dumps(measured)
    (output/"evolution.qasm").write_text(qasm)

    # A joint computational-basis sampler from the ACTUAL coherent circuit.
    # Accepted rows have all5 ancillas zero. Sampling is a bounded ideal
    # Monte Carlo check and carries neither noise nor experimental evidence.
    probabilities = Statevector.from_instruction(circuit).probabilities()
    rng = np.random.default_rng(29)
    attempted_samples = 250000
    sample_counts = rng.multinomial(attempted_samples, probabilities/probabilities.sum())
    accepted = int(sample_counts[:8].sum())
    z_sum = int(sum(int(sample_counts[k])*(1 if k%2 == 0 else -1) for k in range(8)))
    sampled_z = z_sum/accepted
    sampling_failure = .005
    sampling_radius = math.sqrt(2*math.log(2/sampling_failure)/accepted)
    if abs(sampled_z-check["observable_conditional"]) > sampling_radius:
        raise AssertionError("fixed-seed ideal joint sample exceeds its Hoeffding radius")
    sampling = {
        "performed": True, "seed": 29, "complete_attempts": attempted_samples,
        "accepted_samples": accepted, "acceptance_fraction": accepted/attempted_samples,
        "sampled_Z_q0": sampled_z, "conditional_statevector_Z_q0": check["observable_conditional"],
        "absolute_error_vs_exact_evolution": abs(sampled_z-check["observable_exact"]),
        "hoeffding_statistical_error_radius": sampling_radius,
        "conditional_statistical_failure_target": sampling_failure,
        "model": "numpy multinomial from joint ideal Statevector probabilities of actual8qubit circuit; accept ancillas00000",
        "hardware_experiment": False,
    }
    (output/"phases.json").write_text(json.dumps({
        "cosine_phases": list(COSINE_PHASES), "sine_phases": list(sine_phases),
        "tau": COSINE_TAU, "cosine_degree": 14, "sine_degree": 15,
        "sine_fit": sine_fit, "verification": contract,
        "convention": "D(phi)=exp(i phi(2Pi-I)); selector sum of real Pc and -i real Ps",
        "phase_source": "original bounded SciPy optimization; original cosine primitive preserved",
    }, indent=2, allow_nan=False)+"\n")
    report = {
        "status": "verified postselected complex U(t)/4 block, state and observable; deterministic/amplified evolution not implemented",
        "instance": {"n_system": 3, "J": 1, "h": .7, "time": 1, "hbar": 1,
                     "boundary": "open chain", "alpha": 4.1, "input_state": "|000>", "observable": "Z(q0)"},
        "exact_checks": check, "phase_error_contract": contract,
        "conditional_state_check_scope": "normalized accepted data state before final system readout; evolution.qasm subsequently measures all8 wires",
        "max_entry_error_source_vs_decomposed_circuit": decomposition_error,
        "ideal_joint_sampling": sampling,
        "acceptance_projection_scope": "finite-H singular values and coefficient projections are numerical evidence, not certified worst-case hardware guarantees",
        "environment": {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ["qiskit", "numpy", "scipy", "matplotlib"]}},
    }
    (output/"results.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    # Numerical resource projection: accepted samples must be independent and
    # freshly prepared. This cap requires the numerically modeled acceptance
    # probability and ideal postselection; it is not a certified physical bound.
    min_numerical_probability = check["finite_H_numerical_minimum_acceptance_probability"]
    one_measured_trial_attempts = postselection_attempts(min_numerical_probability, .01)
    statistical_error = .03
    statistical_failure = .005
    accepted_shots = observable_shots(statistical_error, statistical_failure)
    postselection_failure = .005
    per_sample_allowance = postselection_failure/accepted_shots
    per_sample_attempts = postselection_attempts(min_numerical_probability, per_sample_allowance)
    counts = dict(compiled.count_ops())
    source_names = ["qsp_evolution.py", "qsp.py", "lcu.py", "hamiltonian_simulation.py", "generate_simulation_qsp_evolution.py"]
    source_hashes = {}
    for name in source_names:
        if REPOSITORY:
            src = HERE/name if name.startswith("generate_") else SOURCE/"openquantum"/(name if name == "hamiltonian_simulation.py" else "simulation/"+name)
        else:
            src = HERE/name
        data = src.read_bytes()
        (output/name).write_bytes(data)
        source_hashes[name] = digest(data)
    license_path = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    (output/"LICENSE.txt").write_bytes(license_path.read_bytes())
    ledger = {
        "resource_kind": "actual primitive logical circuit before continuous-rotation synthesis/error correction",
        "physical_resource_estimate": None,
        "physical_status": "pending arbitrary-rotation synthesis and independent physical-fault budget; frozen Grover baseline unchanged",
        "qubits": compiled.num_qubits, "n_system": 3, "n_index": 3, "n_flag": 1, "n_selector": 1,
        "uncontrolled_access_calls_per_attempt": {"PREPARE": 14, "SELECT": 14, "unprepare": 14},
        "controlled_access_calls_per_attempt": {"controlled_PREPARE": 1, "controlled_SELECT": 1, "controlled_unprepare": 1},
        "branch_conditioned_projector_phase_calls_per_attempt": 31,
        "input_preparation": "all8 qubits assumed reset to0; data000 needs0X gates; selector/real flags each prepared withH; actual access preparation gates included",
        "decomposed_gate_counts_per_attempt": counts, "decomposed_depth_per_attempt": compiled.depth(),
        "decomposed_unitary_gates_per_attempt": sum(counts.values()), "count_basis": ROTATION_BASIS,
        "continuous_rotation_count_per_attempt": sum(counts.get(g, 0) for g in ["rx", "ry", "rz"]),
        "continuous_rotations_synthesized": False,
        "measurement_gates_per_attempt": int(measured.count_ops()["measure"]),
        "measurement_protocol": "measure all8 inZ; accept all5 ancillas0; retain data bitstring for Z(q0). State-only preparation would measure5 ancillas and retain3data wires",
        "postselection_cost_projection": {
            "certified": False, "basis": "finite-H actual-block singular values in double precision",
            "modeled_input000_probability": check["acceptance_probability"],
            "finite_H_numerical_minimum_probability": min_numerical_probability,
            "expected_attempts_for_input000": 1/check["acceptance_probability"],
            "attempt_cap_for_one_accepted_measured_trial_at_failure_01": one_measured_trial_attempts,
            "amplitude_amplification": "not implemented",
        },
        "observable_cost_projection": {
            "statistical_absolute_error": statistical_error, "statistical_failure_allowance": statistical_failure,
            "required_independent_accepted_Z_samples": accepted_shots,
            "postselection_failure_allowance_all_samples": postselection_failure,
            "per_sample_acceptance_failure_allowance": per_sample_allowance,
            "numerical_complete_attempt_cap_per_accepted_sample": per_sample_attempts,
            "numerical_worst_case_complete_attempts": accepted_shots*per_sample_attempts,
            "numerical_expected_attempts_for_input000": accepted_shots/check["acceptance_probability"],
            "scope": "Hoeffding accepted-sample bound and union acceptance cap conditioned on numerical ideal acceptance model; synthesis/faults excluded",
            "certified_physical_bound": False,
        },
        "error_contract": contract,
        "prominent_exclusions": ["continuous rotation synthesis and induced acceptance/state changes", "physical faults and reset/readout times", "deterministic or amplified evolution", "classical coefficient and phase-solver runtime from quantum gate counts"],
        "source_sha256": source_hashes, "qasm_sha256": digest(qasm.encode()),
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
        "environment": report["environment"],
    }
    (output/"logical-resources.json").write_text(json.dumps(ledger, indent=2, allow_nan=False)+"\n")
    x = np.linspace(-1, 1, 1001)
    actual = 4*complex_qsp_response(x, sine_phases=sine_phases)
    target = np.exp(-4.1j*x)
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    axes[0].plot(x, target.real, color="#bec9ca", lw=4, label="exact cos(4.1x)")
    axes[0].plot(x, actual.real, color="#24686e", lw=1.5, label="decoded QSP real")
    axes[0].plot(x, target.imag, color="#d4c3dd", lw=4, label="exact -sin(4.1x)")
    axes[0].plot(x, actual.imag, color="#845896", lw=1.5, label="decoded QSP imaginary")
    axes[0].set(ylabel="Decoded response4K", title="Checked postselected complex QSP evolution")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(x, np.abs(actual-target), color="#996335")
    axes[1].set(xlabel="Normalized eigenvalue x=E/4.1", ylabel="|4K - exp(-4.1ix)|")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output/"complex-response.svg", metadata={"Date": None})
    svg_path = output/"complex-response.svg"
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines())+"\n")
    plt.close(fig)
    return {"output_dir": str(output), "checks": check, "gate_counts": counts,
            "sampling": sampling, "one_measured_trial_numerical_attempt_cap": one_measured_trial_attempts,
            "observable_numerical_attempt_cap": accepted_shots*per_sample_attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/qsp-evolution" if REPOSITORY else HERE/"generated-qsp-evolution")
    parser.add_argument("--refit-sine", action="store_true", help="optional bounded seed31 SciPy reproduction")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.refit_sine), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
