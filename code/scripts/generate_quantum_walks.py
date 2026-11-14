#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate explicit quantum-walk circuits, numerical checks and size sweeps.

--estimate adds 24 bounded conditional surface-code queries to the canonical
size sweep. It does not certify physical execution or rotation accuracy.
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
import shutil
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

import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from openquantum.quantum_walks import (
    build_coined_walk, build_conditional_shift, classical_cycle_distribution,
    coined_observable_budget, lower_walk, sample_position_outcomes, simulate_coined_walk,
)
from openquantum.quantum_walks_continuous import (
    build_continuous_walk, continuous_observable_budget, validate_continuous_walk,
)


def write_json(path: Path, data: dict | list):
    path.write_text(json.dumps(data,indent=2)+"\n")


def export(circuit, path: Path):
    compiled=lower_walk(circuit)
    text=qasm3.dumps(compiled)
    path.write_text(text)
    return compiled, {"logical_qubits":compiled.num_qubits,
        "source_gate_counts":dict(circuit.count_ops()),
        "primitive_gate_counts":dict(compiled.count_ops()),
        "primitive_operations":len(compiled.data),
        "dependency_depth":compiled.depth(),
        "qasm_sha256":hashlib.sha256(text.encode()).hexdigest(),
        "compiled_global_phase_radians":float(compiled.global_phase),
        "qasm_scope":"complete measured trial; the pinned exporter may omit overall phase, not a controlled-oracle specification"}


def public_validation(n,steps,initial_position,initial_coin):
    result=simulate_coined_walk(n,steps,initial_position=initial_position,initial_coin=initial_coin)
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v)
            for k,v in result.items() if k!="statevector"}


def source_provenance():
    commit=git_revision(ROOT)
    names=("src/openquantum/quantum_walks.py","src/openquantum/quantum_walks_continuous.py",
           "src/openquantum/resource_estimation.py","scripts/generate_quantum_walks.py","requirements-lock.txt")
    packages={name:importlib.metadata.version(name) for name in ("qiskit","numpy","scipy","matplotlib")}
    try:packages["qdk"]=importlib.metadata.version("qdk")
    except importlib.metadata.PackageNotFoundError:packages["qdk"]=None
    return {"source_commit_at_generation":commit,
        "source_commit_scope":"source hashes identify code if generation precedes a commit",
        "sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in names},
        "python":platform.python_version(),"device":"CPU","seed":731,"packages":packages}


def physical_query(circuit,budget,profile,path,qasm_digest):
    from openquantum.resource_estimation import estimate_circuit_resources
    report=estimate_circuit_resources(circuit,budget=budget.resource_budget(),profile=profile,
        cache_dir=ROOT/".cache/qre-quantum-walks",observable_budget=budget)
    if report["qasm_sha256"]!=qasm_digest:
        raise AssertionError("Resource query did not use the exported complete circuit")
    report.pop("qasm3")
    report["initialization_boundary"]="clean logical zero inputs assumed; physical initialization/reset timing and faults excluded"
    report["observable"]="one fixed vertex-return indicator in [0,1]; the coined model marginalizes every coin outcome"
    write_json(path,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits",type=int,default=3)
    parser.add_argument("--steps",type=int,default=4)
    parser.add_argument("--time",type=float,default=.7)
    parser.add_argument("--coupling",type=float,default=1.)
    parser.add_argument("--initial-position",type=int,default=0)
    parser.add_argument("--initial-coin",choices=("zero","one","balanced"),default="balanced")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--estimate",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/quantum-walks")
    args=parser.parse_args()
    # Validate all models before deleting/replacing any previous owned output.
    coined=build_coined_walk(args.qubits,args.steps,initial_position=args.initial_position,
        initial_coin=args.initial_coin,measure=True)
    continuous=build_continuous_walk(args.qubits,args.time,args.coupling,args.initial_position,measure=True)
    if args.estimate and importlib.metadata.version("qdk")!="1.34.0":
        raise RuntimeError("Conditional estimation requires the audited QDK1.34.0 environment")
    output=args.output_dir.resolve()
    if output==ROOT or output.is_relative_to(ROOT/"src") or output.is_relative_to(ROOT/"assets/generated/grover") or output.is_relative_to(ROOT/"assets/generated/resources"):
        raise ValueError("Output must be a dedicated walk directory outside source/frozen baseline assets")
    output.mkdir(parents=True,exist_ok=True)
    for name in ("distribution.svg","logical-scaling.csv","logical-scaling.svg","physical-scaling.svg","physical-table.md","surface-code-scaling.json"):
        (output/name).unlink(missing_ok=True)
    if (output/"scaling").exists():shutil.rmtree(output/"scaling")
    provenance=source_provenance()
    budget=coined_observable_budget()
    ct_budget=continuous_observable_budget(synthesis_bias=1e-4)
    ledgers={}
    for kind,circuit in (("coined",coined),("continuous-time",continuous)):
        folder=output/kind;folder.mkdir(exist_ok=True)
        _,facts=export(circuit,folder/"measured-trial.qasm")
        relevant_budget=budget if kind=="coined" else ct_budget
        validation={"performed":False,"reason":"explicitly disabled or outside separate simulator limits"}
        if not args.no_sim:
            if kind=="coined" and args.qubits<=11 and args.steps<=64:
                validation=public_validation(args.qubits,args.steps,args.initial_position,args.initial_coin)
            elif kind=="continuous-time" and args.qubits<=12:
                validation=validate_continuous_walk(args.qubits,args.time,args.coupling,args.initial_position)
        if validation["performed"]:
            probabilities=np.array(validation["probabilities"])
            counts=sample_position_outcomes(probabilities,relevant_budget.shots_required,seed=731)
            validation["classical_categorical_draw"]={"seed":731,"shots":relevant_budget.shots_required,
                "position_counts":counts.tolist(),"return_mean":float(counts[args.initial_position]/counts.sum()),
                "method":"classical NumPy draws from validated ideal position marginal; not hardware or repeated quantum execution"}
        ledger={"algorithm":kind,"circuit_metadata":circuit.metadata,**facts,
            "observable_contract":relevant_budget.as_contract(),"validation":validation,
            "shot_weighted_primitive_gate_counts":{k:v*relevant_budget.shots_required for k,v in facts["primitive_gate_counts"].items()},
            "included":["explicit basis/coin preparation gates","every shift/evolution gate","all final readouts","all declared independent measured trials"],
            "excluded":["physical initialization/reset timing and faults","hardware-specific routing and classical communication",
                "classical compilation/decoding/control latency","concrete rotation synthesis and accuracy certificate"],
            "provenance":provenance,"physical_report_status":"canonical sweep only if requested; it does not estimate arbitrary selected parameters"}
        write_json(folder/"logical-resources.json",ledger);ledgers[kind]=ledger
        modules=("quantum_walks.py",) if kind=="coined" else ("quantum_walks_continuous.py",)
        for module in modules:(folder/module).write_bytes((ROOT/"src/openquantum"/module).read_bytes())
        (folder/"LICENSE.txt").write_bytes((ROOT/"LICENSE").read_bytes())
    if not args.no_sim and ledgers["coined"]["validation"]["performed"]:
        quantum=ledgers["coined"]["validation"]["probabilities"]
        classical=classical_cycle_distribution(args.qubits,args.steps,args.initial_position)
        fig,ax=plt.subplots(figsize=(7,3.4));positions=np.arange(1<<args.qubits)
        ax.bar(positions-.18,quantum,width=.36,label="Quantum coined walk")
        ax.bar(positions+.18,classical,width=.36,label="Classical cycle walk")
        ax.set(xlabel="Position",ylabel="Probability",title=f"Cycle with {1<<args.qubits} vertices, {args.steps} steps")
        ax.legend(loc="upper left",bbox_to_anchor=(1.01,1),frameon=False)
        fig.tight_layout();fig.savefig(output/"distribution.svg");plt.close(fig)
        ledgers["coined"]["classical_cycle_comparator"]=classical.tolist()
    # Source-level pictures show every carry control in the small demo.
    if args.qubits<=5 and args.steps<=6:
        for name,circuit in (("coined/shift-circuit.svg",build_conditional_shift(args.qubits)),
                             ("coined/circuit.svg",coined.remove_final_measurements(inplace=False)),
                             ("continuous-time/circuit.svg",continuous.remove_final_measurements(inplace=False))):
            fig=circuit.draw(output="mpl",fold=14,scale=.85);fig.savefig(output/name,bbox_inches="tight");plt.close(fig)
    else:
        for name in ("coined/shift-circuit.svg","coined/circuit.svg","continuous-time/circuit.svg"):(output/name).unlink(missing_ok=True)
    rows=[];physical_rows=[];reports=[]
    if not args.no_scaling:
        from openquantum.resource_estimation import SurfaceCodeProfile
        for n in (2,3,4,5,6,8):
            for kind in ("coined","continuous-time"):
                folder=output/"scaling"/f"{kind}-n{n}";folder.mkdir(parents=True)
                circuit=build_coined_walk(n,4,measure=True) if kind=="coined" else build_continuous_walk(n,.7,measure=True)
                _,facts=export(circuit,folder/"measured-trial.qasm")
                checked=public_validation(n,4,0,"balanced") if kind=="coined" else validate_continuous_walk(n,.7)
                row={"model":kind,"n":n,"vertices":1<<n,"steps":4 if kind=="coined" else None,
                    "time":None if kind=="coined" else .7,"logical_qubits":circuit.num_qubits,
                    "primitive_operations":facts["primitive_operations"],"cx":facts["primitive_gate_counts"].get("cx",0),
                    "continuous_rotations":sum(facts["primitive_gate_counts"].get(g,0) for g in ("rx","ry","rz")),
                    "measurements":facts["primitive_gate_counts"].get("measure",0),"depth":facts["dependency_depth"],
                    "shots":3329,"all_shot_operations":facts["primitive_operations"]*3329,
                    "return_probability":checked["return_probability"],"simulation_performed":True,
                    "qasm_sha256":facts["qasm_sha256"]}
                rows.append(row)
                write_json(folder/"logical.json",{**row,**facts,"validation":checked,"provenance":provenance})
                if args.estimate:
                    for rate in (1e-3,1e-4):
                        profile=SurfaceCodeProfile(name=f"surface-code-p-{rate:g}",physical_error_rate=rate)
                        relevant_budget=budget if kind=="coined" else ct_budget
                        print(f"Estimating {kind}, n={n}, physical error={rate:g}",flush=True)
                        report=physical_query(circuit,relevant_budget,profile,folder/f"surface-code-p-{rate:g}.json",facts["qasm_sha256"])
                        reports.append({"model":kind,"n":n,"profile":report["physical_profile"],
                            "report":str((folder/f"surface-code-p-{rate:g}.json").relative_to(output)),
                            "pareto_points":len(report["pareto_points"]),"qasm_sha256":report["qasm_sha256"]})
                        for choice,key in (("fewest qubits","peak_physical_qubits"),("shortest runtime","serial_runtime_all_executions_ns")):
                            point=min(report["pareto_points"],key=lambda p:p[key])
                            physical_rows.append({"model":kind,"n":n,"physical_error_rate":rate,"choice":choice,
                                "peak_physical_qubits":point["peak_physical_qubits"],"data_code_distance":point["data_code_distance"],
                                "serial_runtime_seconds":point["serial_runtime_all_executions_ns"]/1e9,
                                "peak_times_runtime_qubit_seconds":point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]/1e9,
                                "modeled_physical_fault_score_all_shots":point["physical_failure_union_bound_all_executions"]})
        with (output/"logical-scaling.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
        fig,axes=plt.subplots(1,2,figsize=(9,3.7))
        for kind in ("coined","continuous-time"):
            chosen=[r for r in rows if r["model"]==kind]
            axes[0].plot([r["n"] for r in chosen],[r["logical_qubits"] for r in chosen],"o-",label=kind)
            axes[1].semilogy([r["n"] for r in chosen],[r["all_shot_operations"] for r in chosen],"o-",label=kind)
        axes[0].set(xlabel="Position bits n",ylabel="Live logical qubits")
        axes[1].set(xlabel="Position bits n",ylabel="Operations across 3329 trials")
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle("Different graph families: fixed cycle steps 4; hypercube Jt=0.7")
        fig.tight_layout();fig.savefig(output/"logical-scaling.svg");plt.close(fig)
    if reports:
        write_json(output/"surface-code-scaling.json",{"status":"computed_conditional_model","queries":reports,
            "query_count":len(reports),"summary_choices":physical_rows,"observable_contract":budget.as_contract(),
            "physical_time_boundary":"quantum model only; preparation/reset/control/communication costs excluded",
            "spacetime_metric":"peak times serial model runtime, not occupancy integral","provenance":provenance})
        table=["| Model | n | Physical error | Choice | Peak qubits | All-shot seconds |",
               "|---|---:|---:|---|---:|---:|"]
        for r in physical_rows:
            if r["n"] in (3,8):table.append(f"| {r['model']} | {r['n']} | {r['physical_error_rate']:g} | {r['choice']} | {r['peak_physical_qubits']:,} | {r['serial_runtime_seconds']:.6g} |")
        (output/"physical-table.md").write_text("\n".join(table)+"\n")
        fig,axes=plt.subplots(1,2,figsize=(9,3.7))
        for kind in ("coined","continuous-time"):
            for rate in (1e-3,1e-4):
                chosen=[r for r in physical_rows if r["model"]==kind and r["physical_error_rate"]==rate and r["choice"]=="fewest qubits"]
                label=f"{kind}, p={rate:g}"
                axes[0].semilogy([r["n"] for r in chosen],[r["peak_physical_qubits"] for r in chosen],"o-",label=label)
                axes[1].semilogy([r["n"] for r in chosen],[r["serial_runtime_seconds"] for r in chosen],"o-",label=label)
        axes[0].set(xlabel="Position bits n",ylabel="Peak physical qubits")
        axes[1].set(xlabel="Position bits n",ylabel="3329-trial model seconds")
        for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle("Conditional mean-fit projections; different graph/evolution tasks")
        fig.tight_layout();fig.savefig(output/"physical-scaling.svg");plt.close(fig)
    write_json(output/"demo-results.json",{"parameters":{"n":args.qubits,"steps":args.steps,"time":args.time,
        "coupling":args.coupling,"initial_position":args.initial_position,"initial_coin":args.initial_coin},
        "models":ledgers,"scaling_rows":rows,"physical_query_count":len(reports),"provenance":provenance})
    for name in ("distribution.svg","logical-scaling.svg","physical-scaling.svg",
                 "coined/shift-circuit.svg","coined/circuit.svg","continuous-time/circuit.svg"):
        path=output/name
        if path.exists():path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps({"output":"quantum-walk artifact directory","models":{k:v["validation"]["performed"] for k,v in ledgers.items()},
        "scaling_circuits":len(rows),"conditional_physical_queries":len(reports)}),flush=True)


if __name__=="__main__":main()
