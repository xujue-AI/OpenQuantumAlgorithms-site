#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate retargetable complex-QSP dynamics at fixed local couplings.

Phases are generated and interval-validated for each alpha*time, including
--no-sim runs. Quantum state/operator checks have separate bounded caps.
Standalone downloads need every copied Python source, LICENSE.txt and the lock.
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
import time as wallclock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "src"
REPOSITORY = (HERE.name == "scripts" and
              (SOURCE / "openquantum/simulation/retargetable_qsp.py").is_file() and
              not (HERE / "retargetable_qsp.py").is_file())
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-retargetable-qsp"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator, Statevector

if REPOSITORY:
    from openquantum import hamiltonian_simulation as hs
    from openquantum.simulation import retargetable_qsp as rq
    from openquantum.simulation.artifact_provenance import git_revision, git_dirty
else:
    import hamiltonian_simulation as hs
    import retargetable_qsp as rq
    from artifact_provenance import git_revision, git_dirty

MAX_CASES = 12
DEFAULT_J = 1.0
DEFAULT_H = 0.7
STATISTICAL_ERROR = 0.03
STATISTICAL_FAILURE = 0.005
ACCEPTANCE_FAILURE = 0.005
SOURCES = (
    "retargetable_qsp.py", "qsp_phase_generation.py", "qsp_certificate.py",
    "artifact_provenance.py", "lcu.py", "hamiltonian_simulation.py", "generate_simulation_retargetable_qsp.py",
)
MANAGED_PLOTS = ("circuit.svg", "response.svg", "scaling.svg", "accuracy-acceptance.svg")
QASM_POLICY = (
    "The pinned Qiskit OpenQASM3 exporter omits an overall compiled phase. "
    "Measured-trial probabilities are preserved; these files are not "
    "phase-preserving coherently controlled oracles. Use the in-memory source "
    "or explicitly restore the recorded phase for controlled reuse."
)


def jsonable(value):
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value


def dump(path: Path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def slug(value: float):
    return format(value, ".17g").replace("-", "m").replace(".", "p").replace("+", "")


def case_id(n_system: int, time: float):
    return f"n{n_system}-time{slug(time)}"


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def save_svg(figure, path: Path):
    figure.savefig(path, metadata={"Date": None}, bbox_inches="tight")
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(figure)


def source_bundle(output: Path):
    hashes = {}
    for name in SOURCES:
        if not REPOSITORY or name.startswith("generate_"):
            path = HERE / name
        elif name == "hamiltonian_simulation.py":
            path = SOURCE / "openquantum" / name
        else:
            path = SOURCE / "openquantum/simulation" / name
        raw = path.read_bytes(); (output / name).write_bytes(raw); hashes[name] = digest(raw)
    path = ROOT / "LICENSE" if REPOSITORY else HERE / "LICENSE.txt"
    raw = path.read_bytes(); (output / "LICENSE.txt").write_bytes(raw)
    hashes["LICENSE.txt"] = digest(raw)
    return hashes


def workload_projection(probability, trial_counts, depth, width):
    samples = hs.observable_shots(STATISTICAL_ERROR, STATISTICAL_FAILURE)
    retries = rq.postselection_attempts(probability, ACCEPTANCE_FAILURE / samples)
    attempts = samples * retries
    return {
        "performed": False, "kind": "unexecuted accepted-Z workload proposal",
        "statistical_error": STATISTICAL_ERROR, "statistical_failure": STATISTICAL_FAILURE,
        "required_accepted_Z_samples": samples, "acceptance_failure_all_slots": ACCEPTANCE_FAILURE,
        "actual_zero_input_numerical_acceptance": probability,
        "complete_attempt_cap_per_slot": retries, "complete_attempt_cap_all_slots": attempts,
        "expected_complete_attempts_all_slots": samples / probability,
        "primitive_counts_all_capped_attempts": {g: int(c)*attempts for g, c in trial_counts.items()},
        "logical_qubit_layer_volume_all_capped_attempts": width*depth*attempts,
        "probability_scope": "state-specific precompiler circuit simulation; floating nested-control/primitive realization, synthesis and physical errors remain separate",
        "retry_protocol": "fresh complete known-basis preparation per rejected attempt; no cloning or local recovery",
        "physical_resource_estimate": None,
    }


def unperformed():
    return {
        "performed": False, "full_operator_performed": False,
        "scope": "--no-sim skips quantum circuit state/operator and independent dense-H comparisons; phase fitting and ideal coefficient validation still run",
        "conditional_state_vector_error": None, "conditional_state_infidelity": None,
        "acceptance_probability": None, "Z_exact": None, "Z_conditional": None, "Z_bias": None,
        "encoded_block_operator_error": None, "decoded_operator_error": None,
        "finite_instance_minimum_acceptance": None, "finite_instance_maximum_acceptance": None,
    }


def circuit_diagram(core, plan, terms):
    """Group actual definitions for readable figures, preserving the source."""
    n = core.metadata["n_system"]
    m = core.metadata["n_index"]
    flag, selector = n+m, n+m+1
    labels = list(range(n, n+m))
    block = rq.build_hermitian_pauli_access(terms).to_gate(label="B: PREPARE SELECT unprepare")
    diagram = QuantumCircuit(core.num_qubits)
    diagram.h(flag); diagram.h(selector)
    for j in range(plan.even_degree+1):
        phases = QuantumCircuit(m+2)
        rq._conditioned_phase(phases, list(range(m)), m, m+1,
            plan.cosine_phases[plan.even_degree-j], 0)
        rq._conditioned_phase(phases, list(range(m)), m, m+1,
            plan.sine_phases[plan.odd_degree-j], 1)
        diagram.append(phases.to_gate(label=f"D pair {j}"), labels+[flag, selector])
        if j < plan.even_degree:
            diagram.append(block, range(flag))
    diagram.append(block.control(1, annotated=False), [selector]+list(range(flag)))
    phase = QuantumCircuit(m+2)
    rq._conditioned_phase(phase, list(range(m)), m, m+1, plan.sine_phases[0], 1)
    diagram.append(phase.to_gate(label="Odd endpoint"), labels+[flag, selector])
    diagram.h(flag); diagram.sdg(selector); diagram.h(selector)
    return diagram


def plots(output, rows, selected, core, plan, terms, *, simulate):
    # Small diagram groups only actual complete access/projector definitions.
    # Its quantum behaviour is that of core, whose primitives are exported.
    if core.num_qubits <= 8 and plan.even_degree <= 16:
        diagram = circuit_diagram(core, plan, terms)
        if simulate and np.max(np.abs(Operator(diagram).data-Operator(core).data)) > 5e-12:
            raise AssertionError("defined diagram changed the phase-sensitive coherent circuit")
        figure = diagram.draw(output="mpl", fold=24)
        save_svg(figure, output / "circuit.svg")
    else:
        (output / "circuit.svg").unlink(missing_ok=True)
    x = np.linspace(-1, 1, 1001)
    if REPOSITORY:
        from openquantum.simulation.qsp_phase_generation import real_qsp_response
    else:
        from qsp_phase_generation import real_qsp_response
    actual = (real_qsp_response(x, plan.cosine_phases) - 1j*real_qsp_response(x, plan.sine_phases))/2
    target = np.exp(-1j*plan.tau*x)/4
    figure, axes = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
    axes[0].plot(x, target.real, color="#bac3c6", lw=4, label="Re target U/4")
    axes[0].plot(x, target.imag, color="#d5c7b8", lw=4, label="Im target U/4")
    axes[0].plot(x, actual.real, color="#24686e", label="Re actual polynomial")
    axes[0].plot(x, actual.imag, color="#845896", label="Im actual polynomial")
    axes[0].set(ylabel="Accepted scalar response", title=f"Retargeted tau={plan.tau:g}; degrees {plan.even_degree}/{plan.odd_degree}")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].semilogy(x, np.maximum(np.abs(actual-target), 1e-18), color="#315e99", label="Numerical scalar error")
    axes[1].axhline(plan.encoded_polynomial_error_upper, ls="--", color="#996335", label="Interval ideal-signal error bound")
    axes[1].set(xlabel="Normalized eigenvalue x", ylabel="Error against exp(-i tau x)/4")
    axes[1].legend(fontsize=8)
    figure.tight_layout(); save_svg(figure, output / "response.svg")
    fixed_time = [r for r in rows if r["time"] == 0.5]
    if len(fixed_time) >= 2:
        fixed_time.sort(key=lambda r:r["n_system"])
        sizes = [r["n_system"] for r in fixed_time]
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
        for ax, key, label, color in zip(axes, ("odd_degree", "full_trial_operations", "full_trial_dependency_depth"),
            ("Odd phase degree", "Complete-trial operations", "Complete-trial depth"), ("#24686e", "#845896", "#996335")):
            ax.plot(sizes, [r[key] for r in fixed_time], "o-", color=color)
            ax.set(xlabel="System spins n", ylabel=label, xticks=sizes); ax.grid(alpha=.2)
        figure.suptitle("Fixed local couplings J=1, h=0.7, t=0.5; regenerated phases", fontsize=11)
        figure.tight_layout(); save_svg(figure, output / "scaling.svg")
    else:
        (output / "scaling.svg").unlink(missing_ok=True)
    simulated = [r for r in rows if r["simulation_performed"]]
    if simulated:
        figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        positions = list(range(len(simulated)))
        labels = [f"n{r['n_system']} t{r['time']:g}" for r in simulated]
        axes[0].semilogy(positions, [max(r["conditional_state_vector_error"], 1e-18) for r in simulated], "o", color="#24686e", label="Conditional-state error")
        axes[0].semilogy(positions, [max(r["Z_bias"], 1e-18) for r in simulated], "s", color="#845896", label="Z bias")
        axes[0].set(ylabel="Error against independent expm"); axes[0].legend(fontsize=8)
        axes[1].plot(positions, [r["acceptance_probability"] for r in simulated], "o", color="#996335")
        axes[1].axhline(1/16, ls="--", color="#bac3c6"); axes[1].set(ylabel="Actual zero-input acceptance")
        for ax in axes:
            ax.set(xticks=positions, xticklabels=labels); ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.2)
        figure.tight_layout(); save_svg(figure, output / "accuracy-acceptance.svg")
    else:
        (output / "accuracy-acceptance.svg").unlink(missing_ok=True)


def generate(output: Path, *, qubits: int = 2, time: float = 0.5, error: float = 1e-5,
             no_sim: bool = False, no_sweep: bool = False):
    if not isinstance(no_sim, bool) or not isinstance(no_sweep, bool):
        raise ValueError("no_sim and no_sweep must be booleans")
    # Call bounded constructor/degree helpers before any output mutation.
    rq._int(qubits, "qubits", 1, rq.MAX_SYSTEM_QUBITS)
    rq._finite(time, "time")
    terms = hs.spin_terms(qubits, DEFAULT_J, DEFAULT_H)
    tau = float(sum(abs(c) for c, _ in terms))*time
    rq.choose_dynamics_degree(tau, error)
    cases = [(qubits, time)]
    if not no_sweep:
        cases += [(n, .5) for n in (1, 2, 3)] + [(n, t) for n in (1, 2) for t in (.25, 1.)]
    cases = list(dict.fromkeys(cases))
    if len(cases) > MAX_CASES: raise ValueError("bounded sweep case cap exceeded")
    ids = [case_id(n,t) for n,t in cases]
    if len(ids) != len(set(ids)): raise ValueError("export IDs must be distinct")
    if not no_sim and any(n > rq.MAX_REFERENCE_SYSTEM_QUBITS or n+(2*n-2).bit_length()+2 > rq.MAX_STATEVECTOR_QUBITS for n,_ in cases):
        raise ValueError("quantum check cap exceeded; use --no-sim")
    prior_files=set()
    prior_ledger=output/"logical-resources.json"
    if prior_ledger.is_file():
        try:
            prior=json.loads(prior_ledger.read_text())
            hashes=prior.get("provenance",{}).get("source_hashes",{})
            if "generate_simulation_retargetable_qsp.py" in hashes:
                for item in prior.get("cases",[]):
                    for key in ("qasm_file","phase_file"):
                        name=item.get(key)
                        if isinstance(name,str) and Path(name).name==name:
                            prior_files.add(name)
        except (OSError,ValueError,TypeError,AttributeError):
            # Unrecognized user files are not a generated-artifact manifest.
            pass
    output.mkdir(parents=True, exist_ok=True)
    plans, records, ledgers, rows, new_files = {}, [], [], [], set()
    chosen_core = chosen_plan = chosen_terms = None
    for n_system, evolution_time in cases:
        terms = hs.spin_terms(n_system, DEFAULT_J, DEFAULT_H)
        tau = float(sum(abs(c) for c, _ in terms))*evolution_time
        started = wallclock.perf_counter()
        if tau not in plans:
            plans[tau] = rq.generate_dynamics_phases(tau, error)
        plan = plans[tau]
        phase_seconds = wallclock.perf_counter()-started
        started = wallclock.perf_counter()
        core = rq.build_retargetable_evolution(terms, evolution_time, error, phase_plan=plan)
        compiled = rq.lower_circuit(core)
        trial = rq.lower_circuit(rq.build_measured_trial(core))
        lowering_seconds = wallclock.perf_counter()-started
        full_operator = core.num_qubits <= rq.MAX_OPERATOR_QUBITS
        started = wallclock.perf_counter()
        check = unperformed() if no_sim else rq.compare_evolution(core, terms, evolution_time, full_operator=full_operator)
        comparison_seconds = None if no_sim else wallclock.perf_counter()-started
        equivalence = None
        equivalence_scope = "not executed"
        if not no_sim:
            if full_operator:
                equivalence = float(np.max(np.abs(Operator(core).data-Operator(compiled).data)))
                equivalence_scope = "full unitary Operator without global-phase alignment"
            else:
                equivalence = float(np.max(np.abs(Statevector.from_instruction(core).data-Statevector.from_instruction(compiled).data)))
                equivalence_scope = "fixed zero-input full Statevector only; not whole-unitary equivalence"
            if equivalence > 5e-12: raise AssertionError("primitive lowering changed phase-sensitive source output")
            if check["conditional_state_vector_error"] > max(5e-12, 2*error) or check["Z_bias"] > max(1e-11,4*error):
                raise AssertionError("actual conditional state/observable exceeded demonstration tolerance")
        identifier = case_id(n_system, evolution_time)
        filename = f"evolution-{identifier}.qasm"
        phase_filename = f"phases-{identifier}.json"
        qasm = qasm3.dumps(trial)
        (output / filename).write_text(qasm); dump(output / phase_filename, plan.as_dict())
        new_files.update((filename,phase_filename))
        probability = check["acceptance_probability"]
        ledger = rq.logical_ledger(compiled, trial, acceptance=probability)
        counts = {str(g):int(c) for g,c in ledger["full_trial_primitive_counts"].items() if g != "barrier"}
        workload = workload_projection(probability, counts, trial.depth(), trial.num_qubits) if probability is not None else None
        parameters = {"n_system":n_system,"J":DEFAULT_J,"h":DEFAULT_H,"time":evolution_time,"hbar":1,
            "alpha":core.metadata["normalization_alpha"],"tau":tau,"initial_state":"|"+"0"*n_system+">", "observable":"Z(q0)",
            "terms":[[c,p] for c,p in terms],"requested_decoded_polynomial_error":error}
        ledger.update({"case_id":identifier,"parameters":parameters,"phase_plan":plan.as_dict(),
            "normalization_sum_rounding_absolute":core.metadata["normalization_sum_rounding_absolute"],
            "normalization_time_rounding_absolute":core.metadata["normalization_time_rounding_absolute"],
            "parameter_mismatch_operator_error_upper":core.metadata["parameter_mismatch_operator_error_upper"],
            "ideal_signal_decoded_error_upper_with_parameter_rounding":core.metadata["ideal_signal_decoded_error_upper_with_parameter_rounding"],
            "logical_source_certificate":core.metadata["logical_source_certificate"],
            "phase_generation_cpu_seconds":phase_seconds,"construction_lowering_cpu_seconds":lowering_seconds,
            "numerical_comparison_cpu_seconds":comparison_seconds,"workload_proposal":workload,
            "qasm_file":filename,"qasm_sha256":digest(qasm.encode()),"compiled_global_phase_radians":float(trial.global_phase),
            "qasm_phase_policy":QASM_POLICY,"phase_file":phase_filename,"phase_sha256":digest((output/phase_filename).read_bytes()),
            "ideal_signal_acceptance_lower":plan.ideal_signal_acceptance_lower,
            "ideal_acceptance_scope":plan.certificate_scope,
            "precompiler_source_acceptance_lower":core.metadata["precompiler_source_acceptance_lower"],
            "source_acceptance_certificate":True,
            "realized_primitive_acceptance_certificate":False})
        records.append({"case_id":identifier,"parameters":parameters,"simulation":check,
            "source_primitive_max_entry_difference":equivalence,"source_primitive_scope":equivalence_scope,
            "phase_plan_file":phase_filename,"phase_generation_cpu_seconds":phase_seconds,
            "construction_lowering_cpu_seconds":lowering_seconds,"numerical_comparison_cpu_seconds":comparison_seconds})
        ledgers.append(ledger)
        rows.append({"case_id":identifier,"n_system":n_system,"time":evolution_time,"alpha":parameters["alpha"],"tau":tau,
            "even_degree":plan.even_degree,"odd_degree":plan.odd_degree,"logical_qubits":trial.num_qubits,
            "full_trial_operations":trial.size(),"full_trial_dependency_depth":trial.depth(),"continuous_rotations":ledger["continuous_rotations"],
            "initial_resets":ledger["initial_resets"],"final_measurements":ledger["final_measurements"],
            "ordinary_access_calls":ledger["ordinary_access_calls"],"controlled_access_calls":ledger["controlled_access_calls"],
            "decoded_ideal_polynomial_error_upper":plan.decoded_polynomial_error_upper,
            "decoded_logical_source_error_upper":core.metadata["precompiler_source_decoded_error_upper"],
            "precompiler_source_acceptance_lower":core.metadata["precompiler_source_acceptance_lower"],
            "ideal_signal_acceptance_lower":plan.ideal_signal_acceptance_lower,"simulation_performed":not no_sim,
            "full_operator_performed":check["full_operator_performed"],"conditional_state_vector_error":check["conditional_state_vector_error"],
            "Z_bias":check["Z_bias"],"acceptance_probability":probability,
            "one_accepted_trial_retry_cap":None if ledger["postselection_retry"] is None else ledger["postselection_retry"]["one_accepted_trial_cap"],
            "all_Z_workload_attempt_cap":None if workload is None else workload["complete_attempt_cap_all_slots"]})
        if (n_system,evolution_time)==(qubits,time): chosen_core,chosen_plan,chosen_terms=core,plan,terms
    # Clean only managed files after successful complete materialization.
    for name in prior_files-new_files:
        (output/name).unlink(missing_ok=True)
    source_hashes = source_bundle(output)
    write_csv(output / "scaling.csv", rows)
    selected_id = case_id(qubits,time)
    plots(output, rows, selected_id, chosen_core, chosen_plan, chosen_terms, simulate=not no_sim)
    revision = git_revision(ROOT) if REPOSITORY else None
    owned_paths = ["scripts/generate_simulation_retargetable_qsp.py", "src/openquantum/simulation/retargetable_qsp.py",
                   "src/openquantum/simulation/qsp_certificate.py", "src/openquantum/simulation/qsp_phase_generation.py",
                   "src/openquantum/simulation/artifact_provenance.py"]
    provenance = {"source_hashes":source_hashes,"source_revision_at_generation":revision,
        "source_working_tree_dirty":git_dirty(ROOT, owned_paths) if REPOSITORY else None,
        "revision_scope":"file hashes identify current working-tree source if generation precedes commit",
        "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","matplotlib")},
        "device":"CPU","transpiler_seed":7,"qubits_initially_zero":False,"randomness":"deterministic seeded numerical phase fitting; no sampled quantum experiment"}
    report = {"status":"checked retargetable complex QSP" if not no_sim else "phase-certified ideal polynomials and materialized circuits; quantum simulation unperformed",
        "selected_case_id":selected_id,"cases":records,"provenance":provenance,"phase_targets_generated_this_run":len(plans),
        "phase_generation_performed":True,"ideal_coefficient_validation_performed":True,"quantum_simulation_performed":not no_sim,
        "family":"fixed local J=1,h=0.7; alpha_n=(n-1)+0.7n; phases regenerated for alpha_n*time",
        "selected_output":"accepted measured Z trial; conditional state checks are before system readout",
        "limits":{"system_qubits":rq.MAX_SYSTEM_QUBITS,"explicit_terms":rq.MAX_TERMS,"abs_tau":rq.MAX_ABS_TAU,"phase_degree":rq.MAX_PHASE_DEGREE,
            "source_operations":rq.MAX_SOURCE_OPERATIONS,"primitive_operations":rq.MAX_PRIMITIVE_OPERATIONS,
            "statevector_total_qubits":rq.MAX_STATEVECTOR_QUBITS,"operator_total_qubits":rq.MAX_OPERATOR_QUBITS,"reference_system_qubits":rq.MAX_REFERENCE_SYSTEM_QUBITS,"cases":MAX_CASES},
        "certificate_scope":chosen_core.metadata["logical_source_certificate"]["scope"],
        "sampling_performed":False,"physical_resource_estimate":None}
    master = {"resource_kind":"actual complete measured logical trials before synthesis and physical timing",
        "selected_case_id":selected_id,"cases":ledgers,"provenance":provenance,
        "error_separation":{"ideal_polynomial":"directed interval coefficient/error bound for stored binary64 phases and parameter with ideal exact signal access",
            "numerical_fit":"bounded seeded nonlinear solve; grid residual is not the certificate",
            "normalization_parameter_rounding":"outward exact-rational parameter mismatch separately added to coefficient and loading bounds",
            "access_angle_rounding":"directed interval PREPARE weight error for stored binary64 RY angles, combined by Duhamel; exact declared logical signed SELECT",
            "primitive_rounding":"floating nested ControlledGate/MCX realization and compiler transformations remain outside the logical-source certificate; phase-sensitive numerical checks cover stated small cases",
            "synthesis":"not performed; arbitrary rotations require a separate coherent norm allowance",
            "sampling":"unexecuted accepted-shot allocation; distinct from acceptance failure",
            "physical_faults":"unmodeled; no fault or reset/instrument backend supplied"},
        "physical_resource_estimate":None,"sampling_performed":False,"qasm_phase_policy":QASM_POLICY}
    dump(output / "results.json",report); dump(output / "logical-resources.json",master)
    return {"output_dir":str(output),"selected_case_id":selected_id,"cases":len(cases),"unique_phase_targets":len(plans),
        "simulation_performed":not no_sim,"selected_logical_qubits":chosen_core.num_qubits,
        "selected_even_degree":chosen_plan.even_degree,"selected_odd_degree":chosen_plan.odd_degree,
        "decoded_ideal_polynomial_error_upper":chosen_plan.decoded_polynomial_error_upper,
        "selected_actual_acceptance":next(r for r in rows if r["case_id"]==selected_id)["acceptance_probability"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits",type=int,default=2);parser.add_argument("--time",type=float,default=.5)
    parser.add_argument("--error",type=float,default=1e-5);parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-sweep",action="store_true");parser.add_argument("--output-dir",type=Path,
        default=ROOT/"assets/generated/simulation/retargetable-qsp" if REPOSITORY else HERE/"generated-retargetable-qsp")
    args=parser.parse_args();print(json.dumps(generate(args.output_dir,qubits=args.qubits,time=args.time,error=args.error,
        no_sim=args.no_sim,no_sweep=args.no_sweep),indent=2,allow_nan=False))


if __name__=="__main__":main()
