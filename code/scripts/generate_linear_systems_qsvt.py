#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate actual spectrum-matched cubic QSVT inverse circuits and resources."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from openquantum.linear_systems import accepted_observable_workload, decompose
from openquantum.linear_systems_qsvt import (
    build_inverse_qsvt, compare_inverse_qsvt, inverse_polynomial, pauli_word_inverse_terms,
    qsvt_measured_trial, verify_inverse_phases,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word",default="X")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--no-estimate",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/linear-systems/qsvt")
    args=parser.parse_args()
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    terms=pauli_word_inverse_terms(args.word)
    circuit=build_inverse_qsvt(terms);measured=decompose(qsvt_measured_trial(terms))
    counts=dict(measured.count_ops());qasm=qasm3.dumps(measured)
    source=(ROOT/"src/openquantum/linear_systems_qsvt.py").read_bytes()
    (output/"linear_systems_qsvt.py").write_bytes(source);(output/"LICENSE.txt").write_bytes((ROOT/"LICENSE").read_bytes())
    (output/"measured-trial.qasm").write_text(qasm)
    phase=verify_inverse_phases();(output/"phases.json").write_text(json.dumps(phase,indent=2)+"\n")
    validation={"performed":False,"reason":"--no-sim or6 system/12total bounded reference"}
    if not args.no_sim and len(args.word)<=6:
        validation={"performed":True,**compare_inverse_qsvt(terms)}
    # For b=|0**n>, P either diagonal (deterministic eigenvalue) or flips an orthogonal bitstring.
    if set(args.word)<=set("IZ"):
        success=.25  # P|0>=|0>, eigenvalue A/alpha=.5, p(.5)=.5
    else:
        success=5/32
    workload=accepted_observable_workload(success)
    ledger={
        "algorithm":"Hermitian QSVT inverse, spectrum-matched cubic",
        "terms":terms,"matrix_normalization":1,"physical_matrix_interpretation":"same normalized solution for A=.5*(.75I-.25P)",
        "logical_qubits":measured.num_qubits,"system_qubits":len(args.word),"matrix_dimension":1<<len(args.word),
        "normalized_spectrum":[.5,1],"kappa":2,"input":"all-zero system state, no preparation gates",
        "polynomial":"p(x)=(5*x-4*x**3)/4", "inverse_scale":.25,
        "phase_verification":phase,"validation":validation,
        "source_gate_counts":dict(circuit.count_ops()),"primitive_gate_counts_per_complete_attempt":counts,
        "primitive_dependency_depth":measured.depth(),"arbitrary_rotations_per_attempt":sum(counts.get(g,0) for g in ("rx","ry","rz")),
        "block_encoding_calls":3,"prepare_calls":3,"select_calls":3,"unprepare_calls":3,
        "signed_projector_phase_calls":4,"real_part_flag_hadamards":2,
        "postselection":"all index and real-part flag bits zero; data bits unrestricted",
        "analytic_success_probability":success,"retry_and_sampling_proposal":workload,
        "capped_primitive_gate_counts_proposal":{name:count*workload["total_capped_attempts"] for name,count in counts.items()},
        "scope":"explicit Pauli-word family; cubic is exact inverse only on the stated spectrum",
        "generic_interval_inverse_error":"large even on [.5,1]; not a general arbitrary-precision solver",
        "included":["explicit Pauli coefficient PREPARE","signed label-controlled Pauli SELECT","PREPARE inverse",
            "all projector phases","real-part extraction flag","every final measurement","all failed complete attempts"],
        "excluded":["clean-state initialization timing and faults","classical phase/parameter preprocessing and compilation time",
            "classical readout/reference postprocessing time","concrete Clifford+T synthesis certificate","amplitude amplification"],
        "postselection_synthesis_status":"mean-fit cost projection cannot certify conditional observable or success probability",
        "qasm_export_policy":{"scope":"complete measured trials; not phase-preserving coherently controlled oracle",
            "compiled_global_phase_radians":float(measured.global_phase),
            "pinned_exporter_global_phase":"may omit global phase; measurement probabilities unaffected"},
        "physical_estimate":None,
        "provenance":{"source_revision_at_generation":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "implementation_sha256":hashlib.sha256(source).hexdigest(),"qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),
            "source_note":"source hash identifies working tree code if generation precedes commit",
            "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","matplotlib")},
            "device":"CPU","seed":7,"command":"python scripts/generate_linear_systems_qsvt.py "+" ".join(sys.argv[1:]),
            "shared_access_sources":{"lcu.py":hashlib.sha256((ROOT/"src/openquantum/simulation/lcu.py").read_bytes()).hexdigest(),
                "qsp.py":hashlib.sha256((ROOT/"src/openquantum/simulation/qsp.py").read_bytes()).hexdigest()}}}
    if not args.no_scaling:
        rows=[]
        for n in (1,2,3,4,6,8,12,24):
            scaled=decompose(qsvt_measured_trial(pauli_word_inverse_terms("X"*n)))
            c=dict(scaled.count_ops())
            checked=compare_inverse_qsvt(pauli_word_inverse_terms("X"*n)) if n<=4 else None
            rows.append({"n":n,"dimension":1<<n,"logical_qubits":scaled.num_qubits,"kappa":2,"degree":3,
                "primitive_operations":len(scaled.data)-c.get("barrier",0),"cx":c.get("cx",0),
                "rotations":sum(c.get(g,0) for g in ("rx","ry","rz")),"depth":scaled.depth(),
                "measurements":c.get("measure",0),"analytic_acceptance":5/32,"simulation_performed":checked is not None,
                "solution_trace_distance":None if checked is None else checked["solution_state_trace_distance"]})
        with (output/"scaling.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
        ledger["scaling_family"]={"definition":"P=X**n,b=|0**n>,fixedκ2/degree3","rows":rows,
            "scope":"actual primitive circuit counts; named rows checked; no general condition/precision scaling claim"}
    grid=np.linspace(.4,1,401)
    fig,ax=plt.subplots(figsize=(6.6,3.5));ax.plot(grid,inverse_polynomial(grid),label="Implemented cubic p(x)")
    ax.plot(grid,.25/grid,"--",label="Requested reciprocal .25/x")
    ax.plot([.5,1],[.5,.25],"o",label="Specified spectrum")
    ax.set(xlabel="Normalized eigenvalue x",ylabel="Accepted-block amplitude");ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(output/"polynomial.svg");plt.close(fig)
    if circuit.num_qubits<=6:
        fig=circuit.draw(output="mpl",fold=25,scale=.7);fig.savefig(output/"circuit.svg",bbox_inches="tight");plt.close(fig)
    if not args.no_estimate:
        from openquantum.resource_estimation import CircuitResourceBudget,estimate_circuit_resources
        if measured.num_qubits<=8 and len(measured.data)<=3000:
            budget=CircuitResourceBudget(1e-6,.005,workload["total_capped_attempts"])
            report=estimate_circuit_resources(qsvt_measured_trial(terms),budget=budget,
                cache_dir=Path("/tmp/openquantum-linear-systems-qdk-cache"))
            report.pop("qasm3");report["postselection_contract"]={"resource_executions":"all complete capped attempts",
                "confidence_status":"conditional cost model; no certified postselected synthesis/error guarantee",
                "initialization":"clean initialized logical qubits assumed; reset cost/faults excluded"}
            (output/"surface-code.json").write_text(json.dumps(report,indent=2)+"\n")
            ledger["physical_estimate"]={"status":report["resource_estimate_kind"],"report":"surface-code.json",
                "pareto_points":len(report["pareto_points"]),"profile":report["physical_profile"],"budget":report["budget"]}
        else: ledger["physical_estimate"]={"status":"unavailable; bounded physical query limit"}
    (output/"logical-resources.json").write_text(json.dumps(ledger,indent=2)+"\n")
    for path in output.glob("*.svg"):path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps({"output_dir":str(output),"validation":validation,"physical_estimate":ledger["physical_estimate"]},indent=2))


if __name__=="__main__":main()
