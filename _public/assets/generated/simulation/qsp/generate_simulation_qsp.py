#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate the checked default QSP cosine primitive; optional bounded phase refit.

Use the repository's pinned Python3.12 environment. The phase finder is optional
classical preprocessing, not a quantum gate. Downloaded standalone reproduction
needs this script, qsp.py, lcu.py, hamiltonian_simulation.py, and the project lock.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (SOURCE / "openquantum/hamiltonian_simulation.py").exists()
if REPOSITORY:
    sys.path.insert(0, str(SOURCE))
else:
    sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator
from scipy.linalg import cosm

if REPOSITORY:
    from openquantum.hamiltonian_simulation import spin_terms, dense_hamiltonian, decompose, ROTATION_BASIS
    from openquantum.simulation.lcu import postselection_attempts
    from openquantum.simulation.qsp import (
        COSINE_PHASES, COSINE_TAU, build_real_qsp_filter, build_qubitization_walk,
        qsp_real_response, verify_cosine_phases, fit_cosine_phases, postselected_filter_state,
    )
else:
    from hamiltonian_simulation import spin_terms, dense_hamiltonian, decompose, ROTATION_BASIS
    from lcu import postselection_attempts
    from qsp import (
        COSINE_PHASES, COSINE_TAU, build_real_qsp_filter, build_qubitization_walk,
        qsp_real_response, verify_cosine_phases, fit_cosine_phases, postselected_filter_state,
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def generate(output: Path, refit: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    phases = COSINE_PHASES
    fit = {
        "method": "scipy.optimize.least_squares", "seed": 17,
        "fit_nodes": 80, "successful_restart_zero_based": 6,
        "function_evaluations_successful_restart": 3027,
        "node_residual": 7.771561172376096e-16,
        "maximum_restarts": 8, "maximum_function_evaluations_per_restart": 4000,
        "ftol": 1e-13, "gtol": 1e-13, "xtol": 1e-13,
        "performed_this_run": False,
    }
    if refit:
        phases, fit_details = fit_cosine_phases()
        fit = {**fit_details, "performed_this_run": True}
    verification = verify_cosine_phases(phases)
    if verification["uniform_error_bound_with_numerical_reconstruction_guard"] > 7e-9:
        raise AssertionError("phase approximation exceeded declared demonstration tolerance")
    terms = spin_terms(3)
    circuit = build_real_qsp_filter(terms, phases)
    matrix = Operator(circuit).data
    reference = cosm(dense_hamiltonian(3))/2
    block_error = float(np.linalg.norm(matrix[:8, :8]-reference, 2))
    if block_error > 7e-9:
        raise AssertionError("QSP circuit block disagrees with independent classical matrix cosine")
    initial = np.eye(8, dtype=complex)[:, 0]
    state, probability = postselected_filter_state(circuit, initial)
    accepted_reference = reference@initial
    conditional_error = float(np.linalg.norm(state-accepted_reference/np.linalg.norm(accepted_reference)))
    attempts = postselection_attempts(probability, 0.01)
    compiled = decompose(circuit)
    measured = compiled.copy()
    measured.measure_all()
    qasm = qasm3.dumps(measured)
    (output/"cosine-filter.qasm").write_text(qasm)
    walk = decompose(build_qubitization_walk(terms))
    walk.measure_all()
    (output/"qubitization-walk.qasm").write_text(qasm3.dumps(walk))

    sources = {
        "qsp.py": SOURCE/"openquantum/simulation/qsp.py",
        "lcu.py": SOURCE/"openquantum/simulation/lcu.py",
        "hamiltonian_simulation.py": SOURCE/"openquantum/hamiltonian_simulation.py",
        "generate_simulation_qsp.py": HERE/"generate_simulation_qsp.py",
    } if REPOSITORY else {name: HERE/name for name in ["qsp.py", "lcu.py", "hamiltonian_simulation.py", "generate_simulation_qsp.py"]}
    hashes = {}
    for name, source in sources.items():
        data = source.read_bytes()
        (output/name).write_bytes(data)
        hashes[name] = sha256(data)
    license_path = ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    (output/"LICENSE.txt").write_bytes(license_path.read_bytes())
    phase_data = {
        "phases": list(phases), "phase_convention": circuit.metadata["phase_convention"],
        "target": "cos(4.1*x)/2", "fit": fit, "verification": verification,
        "phase_source": "original bounded numerical optimization; not imported from an upstream package",
    }
    (output/"phases.json").write_text(json.dumps(phase_data, indent=2)+"\n")
    report = {
        "status": "verified QSP cosine polynomial primitive; full complex evolution planned",
        "instance": {"n_system": 3, "J": 1, "h": 0.7, "time": 1, "hbar": 1,
                     "boundary": "open chain", "normalization_alpha": COSINE_TAU,
                     "input_state": "|000>", "encoded_operator": "cos(H)/2"},
        "phase_verification": verification,
        "circuit_block_spectral_error_vs_cosm_H_over_2": block_error,
        "conditional_filter_state_vector_error": conditional_error,
        "postselection_success_probability": probability,
        "expected_attempts_for_one_accepted_state": 1/probability,
        "complete_attempts_for_99_percent_acceptance": attempts,
        "conditional_filter_state_real": state.real.tolist(),
        "conditional_filter_state_imag": state.imag.tolist(),
        "observable_estimation": "not performed; this is a polynomial-filter state, not e^(-iHt)|000>",
        "scope": "tiny explicit block encoding; generic supplied-phase generator capped at degree64, 32 data qubits, 256 labels; simulation capped at12 total qubits",
    }
    (output/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    counts = dict(compiled.count_ops())
    ledger = {
        "resource_kind": "logical primitive circuit before rotation synthesis/error correction",
        "physical_resource_estimate": None,
        "physical_status": "pending shared arbitrary-rotation adapter and separate physical-fault budget",
        "qubits": compiled.num_qubits, "n_system": 3, "n_index": 3, "n_flag": 1,
        "block_encoding_calls_per_attempt": len(phases)-1,
        "prepare_calls_per_attempt": len(phases)-1,
        "prepare_inverse_calls_per_attempt": len(phases)-1,
        "select_calls_per_attempt": len(phases)-1,
        "controlled_projector_phase_calls": len(phases),
        "input_preparation": "all7 qubits assumed reset to0; |000> data requires0 additional gates; flag H and PREPARE gates included",
        "decomposed_gate_counts": counts, "decomposed_depth": compiled.depth(),
        "decomposed_unitary_gates": sum(counts.values()), "count_basis": ROTATION_BASIS,
        "arbitrary_rotation_gate_count": sum(counts.get(g, 0) for g in ["rx", "ry", "rz"]),
        "all_rotations_synthesized": False,
        "measurement_gates_per_attempt": measured.count_ops().get("measure", 0),
        "measurement_protocol": "export measures all qubits in Z; accept label000 and flag0 and retain data bitstring; state preparation alone would measure only4 ancillas",
        "postselection": {"probability": probability, "failure_target": 0.01, "complete_attempt_cap": attempts,
                          "expected_complete_attempts": 1/probability, "amplitude_amplification": "not implemented"},
        "worst_case_gate_counts_for_one_accepted_filter_trial": {g: int(c)*attempts for g, c in counts.items()},
        "error_contract": {
            "Hamiltonian_representation": "given real Pauli coefficients, floating angles; no term truncation",
            "polynomial_approximation": verification,
            "coherent_synthesis": "excluded; RX/RY/RZ remain continuous-angle primitives",
            "statistical_error": "observable estimation not performed/costed; postselection failure separately reported",
            "physical_faults": "excluded; not identified with coherent approximation error",
        },
        "prominent_exclusions": ["coherent rotation synthesis", "physical errors and reset/readout times",
                                  "full time evolution", "amplitude amplification", "observable sampling workload",
                                  "classical coefficient/phase-fit preprocessing runtime in quantum gate counts"],
        "qasm_sha256": sha256(qasm.encode()), "source_sha256": hashes,
        "environment": {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ["qiskit", "numpy", "scipy", "matplotlib"]}},
        "transpilation": {"optimization_level": 1, "seed_transpiler": 7},
    }
    (output/"logical-resources.json").write_text(json.dumps(ledger, indent=2)+"\n")
    x = np.linspace(-1, 1, 1001)
    response = qsp_real_response(x, phases)
    target = np.cos(COSINE_TAU*x)/2
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(x, target, color="#b7bdc7", lw=4, label="cos(4.1x)/2")
    axes[0].plot(x, response, color="#216e73", lw=1.6, label="real QSP response, degree14")
    axes[0].set(ylabel="Polynomial response", title="Checked QSP cosine-filter primitive")
    axes[0].legend()
    axes[1].plot(x, response-target, color="#8057a3")
    axes[1].set(xlabel="Normalized eigenvalue x = E/4.1", ylabel="QSP − target")
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output/"polynomial-response.svg")
    plt.close(fig)
    figure_path = output/"polynomial-response.svg"
    figure_path.write_text("\n".join(line.rstrip() for line in figure_path.read_text().splitlines())+"\n")
    return {"output_dir": str(output), "block_error": block_error, "postselection_probability": probability,
            "primitive_gate_counts": counts, "degree": len(phases)-1, "status": report["status"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"assets/generated/simulation/qsp" if REPOSITORY else HERE/"generated-qsp")
    parser.add_argument("--refit-phases", action="store_true", help="bounded optional SciPy fit; frozen phases used by default")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.refit_phases), indent=2))


if __name__ == "__main__":
    main()
