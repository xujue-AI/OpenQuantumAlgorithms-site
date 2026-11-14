#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Bounded interval reciprocal QSVT circuits, exact certificates and costs."""
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
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit,qasm3
from fractions import Fraction
from openquantum.linear_systems_qsvt_reciprocal import (
    DEFAULT_TERMS,ReciprocalPolynomial,active_chain_terms,build_reciprocal_qsvt,
    choose_k,compare_reciprocal,default_preparation,fit_reciprocal_phases,
    lower_reciprocal,measured_reciprocal_trial,observable_workload,
    reciprocal_polynomial,verify_reciprocal_phases,
)


def source_revision(root=ROOT):
    """An export has no revision; never borrow an enclosing repository HEAD."""
    if not (root/".git").exists(): return None
    try:
        top = subprocess.run(["git","rev-parse","--show-toplevel"],cwd=root,text=True,
                             capture_output=True,check=True).stdout.strip()
        if Path(top).resolve()!=root.resolve(): return None
        return subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,
                              capture_output=True,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError): return None


def write_csv(path,rows):
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def circuit_row(n,k,delta="1/2",simulate=True):
    terms = active_chain_terms(n)
    start = time.perf_counter()
    circuit = measured_reciprocal_trial(terms,k,delta)
    lowered = lower_reciprocal(circuit)
    elapsed = time.perf_counter()-start
    counts = dict(lowered.count_ops())
    check = compare_reciprocal(terms,k,delta) if simulate and n<=4 else None
    return {
        "n":n,"matrix_dimension":1<<n,"K":k,"degree":2*k+1,
        "promised_delta":float(Fraction(delta)),"actual_pauli_terms":len(terms),
        "logical_qubits":lowered.num_qubits,"primitive_operations":len(lowered.data)-counts.get("barrier",0),
        "cx":counts.get("cx",0),"continuous_rotations":sum(counts.get(g,0) for g in ("rx","ry","rz")),
        "measurements":counts.get("measure",0),"dependency_depth":lowered.depth(),
        "generation_and_lowering_seconds":elapsed,
        "uniform_Pauli_bias_bound":circuit.metadata["phase_adjusted_Pauli_bias_bound"],
        "uniform_ideal_acceptance_lower_bound":circuit.metadata["phase_adjusted_success_lower_bound"],
        "simulation_performed":check is not None,
        "actual_condition":None if check is None else check["actual_condition_number"],
        "success_probability":None if check is None else check["success_probability"],
        "trace_distance":None if check is None else check["solution_trace_distance"],
        "z_absolute_bias":None if check is None else check["z_absolute_bias"],
        "physical_scaling_computed":False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits",type=int,help="Use active-chain family at this width instead of default four-eigenvalue matrix")
    parser.add_argument("--K",type=int,default=16)
    parser.add_argument("--delta",default="1/2",help="Exact promised gap; reference checks reject violations")
    parser.add_argument("--pauli-bias",help="Choose K by exact bounded scan instead")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--no-estimate",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/linear-systems/reciprocal-qsvt")
    args = parser.parse_args()
    k = choose_k(args.delta,args.pauli_bias) if args.pauli_bias else args.K
    model = ReciprocalPolynomial(k,args.delta)
    terms = DEFAULT_TERMS if args.qubits is None else active_chain_terms(args.qubits)
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    start = time.perf_counter()
    unitary = build_reciprocal_qsvt(terms,k,args.delta)
    source_measured = measured_reciprocal_trial(terms,k,args.delta)
    measured = lower_reciprocal(source_measured)
    generation_seconds = time.perf_counter()-start
    counts = dict(measured.count_ops())
    qasm = qasm3.dumps(measured)
    (output/"measured-trial.qasm").write_text(qasm)
    phase = verify_reciprocal_phases(k,unitary.metadata["phases"])
    fitted,fit = fit_reciprocal_phases(k)
    fit["uniform_ideal_signal_certificate"] = verify_reciprocal_phases(k,fitted)["rigorous_ideal_signal_certificate"]
    phase["optional_reproducible_fitter"] = fit
    phase["fitter_note"] = "symmetric fitter returns an equivalent certified response; frozen default may use a different valid phase sequence"
    (output/"phases.json").write_text(json.dumps(phase,indent=2)+"\n")
    validation = {"performed":False,"reason":"--no-sim or bounded reference cap"}
    n = unitary.metadata["n_system"]
    if not args.no_sim and n<=5:
        validation = compare_reciprocal(terms,k,args.delta)
        if validation["accepted_branch_error_vs_target_polynomial"]>1e-10:
            raise RuntimeError("actual circuit failed independent target-polynomial check")
        if validation["z_absolute_bias"]>unitary.metadata["phase_adjusted_Pauli_bias_bound"]+1e-11:
            raise RuntimeError("actual conditional readout failed numerical analytic-bound check")
    p = unitary.metadata["retry_success_lower_bound_exact"]
    workload = observable_workload(Fraction(p)) if p is not None else None
    included = ["gate-built complex input preparation","every explicit coefficient PREPARE",
                "signed label-controlled Pauli SELECT","every PREPARE inverse",
                "all signed projector phases and both real-part flag Hadamards",
                "all final data/label/flag measurements","every failed full attempt and fresh re-preparation"]
    excluded = ["fresh clean initialization/reset timing and faults","concrete Clifford+T synthesis and access-loading certificate",
                "classical preprocessing/compilation/reference/readout wall time from physical model",
                "arbitrary matrix/vector loading","amplitude amplification"]
    owned_sources = (
        "src/openquantum/linear_systems_qsvt_reciprocal.py","src/openquantum/reciprocal_phase_certificate.py",
        "src/openquantum/linear_inverse_lcu.py","src/openquantum/linear_systems_qsvt.py",
        "src/openquantum/simulation/lcu.py","src/openquantum/simulation/qsp.py",
        "scripts/generate_linear_systems_qsvt_reciprocal.py",
    )
    ledger = {
        "schema_version":1,**unitary.metadata,"matrix_dimension":1<<n,
        "validation":validation,"observable":"one fixed conditional Z(q0) mean",
        "source_gate_counts":dict(unitary.count_ops()),"source_dependency_depth":unitary.depth(),
        "primitive_gate_counts_per_complete_attempt":counts,"primitive_dependency_depth":measured.depth(),
        "continuous_rotations_per_attempt":sum(counts.get(g,0) for g in ("rx","ry","rz")),
        "prepare_calls":model.degree,"select_calls":model.degree,"unprepare_calls":model.degree,
        "projector_phase_calls":model.degree+1,"real_part_flag_hadamards":2,
        "input_preparation_gate_counts":dict(default_preparation(n).count_ops()),
        "generation_and_lowering_seconds_CPU":generation_seconds,"included_in_attempt":included,"excluded":excluded,
        "output_precision_contract":{"absolute_Pauli_error_target":.05,"algorithm_bias_allowance":.01,
                                     "algorithm_bias_contract_met":unitary.metadata["phase_adjusted_Pauli_bias_bound"]<=.01,
                                     "synthesis_bias_allowance":.01,"sampling_error":.03,
                                     "sampling_failure":.004,"empty_slot_failure":.001,"physical_failure":.005,
                                     "status":"conditional proposal; no complete access/synthesis/physical confidence certificate" if unitary.metadata["phase_adjusted_Pauli_bias_bound"]<=.01 else "unmet inverse-accuracy allocation at thisK; sampling estimates the implemented coarse filter only"},
        "retry_and_sampling_proposal":workload,
        "scope":"exact target and rigorously certified ideal stored phases; actual numerical Pauli loading checked on bounded examples, not uniformly certified",
        "lowering_policy":{"qubits_initially_zero":False,"optimization_level":1,"seed":7,
                           "basis":["x","h","s","sdg","cx","rx","ry","rz","measure"]},
        "qasm_export_policy":{"scope":"complete measured trial; not a phase-preserving controlled reusable oracle",
                              "global_phase_radians":float(measured.global_phase),"pinned_exporter":"may omit overall global phase; measured probabilities unaffected"},
        "physical_estimate":None,
        "provenance":{"source_revision_at_generation":source_revision(),
                      "source_hashes_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in owned_sources},
                      "qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),
                      "source_note":"local HEAD only when this ROOT has its own git marker and is the repository top-level; exact content hashes identify working-tree code",
                      "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","matplotlib")},
                      "seed":7,"device":"CPU","command":"python scripts/generate_linear_systems_qsvt_reciprocal.py",
                      "parameters":{"n":n,"K":k,"delta":str(model.delta),"matrix":"default" if args.qubits is None else "active_chain"}},
    }
    if workload is not None:
        ledger["capped_primitive_gate_counts_proposal"] = {name:c*workload["total_capped_attempts"] for name,c in counts.items()}
    if not args.no_scaling:
        widths = [circuit_row(width,k,simulate=not args.no_sim) for width in (1,2,3,4,6,8,12,24)]
        degrees = [circuit_row(2,j,simulate=not args.no_sim) for j in (0,1,2,4,8,12,16,24,31)]
        conditions = []
        for gap in ("3/4","1/2","2/5","1/3","1/4","1/8"):
            try:
                selected = choose_k(gap,.01)
                conditions.append({"delta":gap,"condition_upper":float(1/Fraction(gap)),"K":selected,"degree":2*selected+1,
                                   "uniform_Pauli_bound":float(ReciprocalPolynomial(selected,Fraction(gap)).pauli_bias_bound),"status":"target supported; signed spectral promise assumed","actual_circuit_generated":False})
            except ValueError:
                conditions.append({"delta":gap,"condition_upper":float(1/Fraction(gap)),"K":None,"degree":None,
                                   "uniform_Pauli_bound":None,"status":"unavailable at bounded odd degree63","actual_circuit_generated":False})
        write_csv(output/"width-scaling.csv",widths)
        write_csv(output/"degree-convergence.csv",degrees)
        write_csv(output/"condition-scaling.csv",conditions)
        ledger["scaling"] = {"width":"actual active-chain Pauli circuits; every spin participates; n<=4 rows numerically simulated unless --no-sim",
                             "degree":"actual circuits on fixed two-spin active-chain matrix, fixed gap1/2; named rows numerically checked",
                             "condition":"exact bounded target-degree choices; no circuit or physical sweep claimed",
                             "physical_scaling_computed":False}
    else: widths,degrees = [],[]
    fig,axes = plt.subplots(1,3,figsize=(12,3.4))
    grid = np.linspace(float(model.delta),1,401)
    axes[0].plot(grid,float(model.scale)/grid,"--",label="Scaled inverse")
    axes[0].plot(grid,reciprocal_polynomial(grid,k),label=f"p_K, K={k}")
    axes[0].set(xlabel="|Normalized eigenvalue|",ylabel="Positive filter amplitude");axes[0].legend(fontsize=8)
    if degrees:
        axes[1].semilogy([r["degree"] for r in degrees],[r["uniform_Pauli_bias_bound"] for r in degrees],"--",label="Uniform bound")
        checked = [r for r in degrees if r["simulation_performed"]]
        if checked: axes[1].semilogy([r["degree"] for r in checked],[max(r["z_absolute_bias"],1e-16) for r in checked],"o-",label="Actual Z bias")
        axes[1].legend(fontsize=8)
    axes[1].set(xlabel="Odd degree",ylabel="Pauli mean bias")
    if widths: axes[2].plot([r["n"] for r in widths],[r["primitive_operations"] for r in widths],"o-")
    axes[2].set(xlabel="Active data qubits n",ylabel="Generated primitive operations")
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"polynomial-and-scaling.svg");plt.close(fig)
    small = measured_reciprocal_trial(((3,"I"),(-1,"X")),1,"1/2")
    fig = small.draw(output="mpl",fold=28,scale=.7)
    fig.savefig(output/"circuit.svg",bbox_inches="tight");plt.close(fig)
    compact = QuantumCircuit(unitary.num_qubits,name="Complete default trial")
    compact.compose(default_preparation(n),qubits=range(n),inplace=True)
    compact.append(unitary.to_gate(label=f"Actual interval QSVT: degree{model.degree}"),range(unitary.num_qubits))
    compact.measure_all()
    fig = compact.draw(output="mpl",fold=24,scale=.8)
    fig.savefig(output/"default-circuit.svg",bbox_inches="tight");plt.close(fig)
    physical = {"status":"not requested","pareto_points":[]}
    if not args.no_estimate:
        if not ledger["output_precision_contract"]["algorithm_bias_contract_met"]:
            physical = {"status":"unavailable for declared inverse-output task; selectedK exceeds algorithm bias allocation","pareto_points":[]}
        elif workload is None or measured.num_qubits>8 or len(measured.data)>4000:
            physical = {"status":"unavailable under bounded8-wire/4000-operation query or positive retry-bound cap","pareto_points":[]}
        else:
            from openquantum.resource_estimation import CircuitResourceBudget,estimate_circuit_resources
            budget = CircuitResourceBudget(1e-6,.005,workload["total_capped_attempts"])
            try:
                physical = estimate_circuit_resources(measured,budget=budget,cache_dir=Path("/tmp/openquantum-reciprocal-qdk-cache"))
                physical.pop("qasm3",None)
                physical["status"] = physical["resource_estimate_kind"]
                physical["conditional_output_scope"] = "all capped complete attempts; exact-access/stored-phase proposal; no numeric access or concrete synthesis confidence certificate"
            except RuntimeError as error:
                physical = {"status":"unavailable at unchanged declared budgets","reason":str(error),"pareto_points":[],
                            "budget":{"synthesis_model_error_per_execution":1e-6,"physical_failure_probability":.005,"executions":workload["total_capped_attempts"]},
                            "conditional_output_scope":"no budgets relaxed; estimator infeasibility is not evidence of physical impossibility"}
    (output/"surface-code.json").write_text(json.dumps(physical,indent=2)+"\n")
    ledger["physical_estimate"] = {"status":physical["status"],"report":"surface-code.json","pareto_points":len(physical["pareto_points"])}
    (output/"logical-resources.json").write_text(json.dumps(ledger,indent=2)+"\n")
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps({"n":n,"degree":model.degree,"validation":validation,"workload":workload,"physical_estimate":ledger["physical_estimate"]},indent=2))


if __name__=="__main__": main()
