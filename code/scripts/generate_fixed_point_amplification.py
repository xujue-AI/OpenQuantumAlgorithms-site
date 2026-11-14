#!/usr/bin/env python3
"""Generate bounded, explicit YLC circuits and conditional resource evidence.

The threshold is an original accounting example with an exact classical answer.
QDK projections require an explicit flag; ideal-angle checks are independent of
the conditional synthesis/fault model. This generator never changes a baseline.
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
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
try:
    from source_provenance import git_revision, git_dirty
except ImportError:
    # Public downloads have no Git history. Never infer an ancestor checkout.
    def git_revision(root): return None
    def git_dirty(root,paths): return None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

from openquantum.amplitude_estimation import threshold_preparation
from openquantum.fixed_point_amplification import (
    MAX_SIM_QUBITS, build_fixed_point_amplification, fixed_point_schedule,
    good_probability, ideal_success_probability, lower_fixed_point,
)


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def provenance():
    names=("fixed_point_amplification.py","amplitude_estimation.py","resource_estimation.py")
    owned=["src/openquantum/"+name for name in names]
    owned.append("scripts/generate_fixed_point_amplification.py")
    versions={}
    for name in ("qiskit","numpy","matplotlib","qdk"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {
        "python":platform.python_version(),"packages":versions,"device":"CPU",
        "system":platform.system(),"machine":platform.machine(),"random_seed":None,
        "deterministic_simulation":True,"transpiler_seed":7,
        "source_base_commit":git_revision(ROOT),"owned_source_dirty":git_dirty(ROOT,owned),
        "source_commit_is_full_source_certificate":False,
        "source_sha256":{name:sha256(ROOT / "src/openquantum" / name) for name in names},
        "generator_sha256":sha256(Path(__file__)),
        "requirements_lock_sha256":sha256(ROOT / "requirements-lock.txt"),
        "hardware_execution":False,
    }


def independent_reference(p,length,delta):
    """Long-double recurrence, separate from production acos/cosh code."""
    if p in (0.,1.):return p
    gamma=np.longdouble(1)/np.cosh(np.arccosh(np.longdouble(1)/delta)/length)
    x=np.sqrt(np.longdouble(1)-p)/gamma
    previous,current=np.longdouble(1),x
    for _ in range(2,length+1):previous,current=current,2*x*current-previous
    return float(1-(np.longdouble(delta)*current)**2)


def build_case(n,threshold,lower,delta,simulate):
    started=time.perf_counter()
    preparation=threshold_preparation(n,threshold)
    circuit=build_fixed_point_amplification(preparation,[n],1,lower,delta)
    measured=circuit.copy()
    measured.measure_all()
    lowered=lower_fixed_point(measured)
    qasm=qasm3.dumps(lowered)
    p=threshold/2**n
    schedule=fixed_point_schedule(lower,delta)
    expected=independent_reference(p,schedule.length,delta)
    check={"performed":False,"method":"bounded ideal Qiskit statevector",
           "success_probability":None,"lowered_state_max_abs_error":None,
           "full_operator_max_abs_error":None}
    if simulate:
        if n+1>MAX_SIM_QUBITS:
            raise ValueError("Requested simulation exceeds its independent width cap; use --no-sim")
        actual=good_probability(circuit,[n],1)
        state=Statevector.from_instruction(circuit).data
        compiled_state=Statevector.from_instruction(lowered.remove_final_measurements(inplace=False)).data
        state_error=float(np.max(np.abs(state-compiled_state)))
        if abs(actual-expected)>2e-11 or state_error>2e-11:
            raise RuntimeError("Actual circuit disagrees with independent law or exact-angle lowering")
        operator_error=None
        if n<=3:
            # Direct projector matrices retain global phase and verify every input.
            a=Operator(preparation).data
            psi=a[:,0]
            identity=np.eye(len(psi),dtype=complex)
            good=np.diag([int((i>>n)&1) for i in range(len(psi))])
            reference=a.copy()
            for alpha,beta in zip(schedule.alphas,schedule.betas,strict=True):
                reference=-(identity+(np.exp(-1j*alpha)-1)*np.outer(psi,psi.conj())) @ (
                    identity+(np.exp(1j*beta)-1)*good) @ reference
            operator_error=float(np.max(np.abs(Operator(circuit).data-reference)))
            if operator_error>2e-11:raise RuntimeError("Full-operator projector check failed")
        check.update(performed=True,success_probability=actual,
                     lowered_state_max_abs_error=state_error,full_operator_max_abs_error=operator_error)
    counts=dict(lowered.count_ops())
    record={
        "parameters":{"n_data":n,"threshold":threshold,"exact_preparation_success":p,
                      "promise_satisfied":p>=lower},
        "schedule":schedule.as_metadata(),"access_ledger":circuit.metadata,
        "source_gate_counts":dict(measured.count_ops()),"primitive_gate_counts":counts,
        "algorithm_qubits":measured.num_qubits,"primitive_qubits":lowered.num_qubits,
        "primitive_unitary_operations":sum(v for k,v in counts.items() if k not in {"measure","barrier"}),
        "primitive_operations_with_readout":sum(v for k,v in counts.items() if k!="barrier"),
        "primitive_dependency_depth":lowered.depth(),"output_measurements":counts.get("measure",0),
        "independent_chebyshev_success":expected,"simulation":check,
        "qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),
        "qasm_scope":"complete measured trial; QASM global phase is irrelevant for this output and is not a controlled-oracle specification",
        "generation_lowering_reference_wall_time_s":time.perf_counter()-started,
        "included_quantum_work":["initial A","every good phase","every A inverse","every full zero-state phase","every forward A","all data and flag measurements"],
        "exclusions":["physical initialization/reset protocol","external data loading","classical verification/control/decoding latency","compilation time in modeled quantum runtime","certified per-angle synthesis and runtime-tail bounds"],
    }
    return measured,lowered,qasm,record


def physical_projection(measured,record,cache):
    from openquantum.resource_estimation import CircuitResourceBudget,SurfaceCodeProfile,estimate_circuit_resources,prepare_measured_circuit
    budget=CircuitResourceBudget(synthesis_model_error_per_execution=1e-5,
                                 physical_failure_probability=.005,executions=1)
    reports=[]
    estimated=prepare_measured_circuit(measured)
    estimated_qasm=qasm3.dumps(estimated)
    adapter_state_error=None
    if measured.num_qubits<=MAX_SIM_QUBITS:
        adapter_state_error=float(np.max(np.abs(
            Statevector.from_instruction(measured.remove_final_measurements(inplace=False)).data-
            Statevector.from_instruction(estimated.remove_final_measurements(inplace=False)).data)))
        if adapter_state_error>2e-11:
            raise RuntimeError("Resource adapter lowering changed the selected coherent output")
    for error in (1e-3,1e-4):
        profile=SurfaceCodeProfile(name=f"illustrative-p-{error:g}",physical_error_rate=error,
                                   gate_time_ns=100,measurement_time_ns=500)
        report=estimate_circuit_resources(measured,budget=budget,profile=profile,cache_dir=cache)
        if report["qasm3"]!=estimated_qasm:
            raise RuntimeError("Resource report does not match the independently prepared estimated export")
        report["input_primitive_export_qasm_sha256"]=record["qasm_sha256"]
        report["estimated_qasm_scope"]="same complete measured trial after the adapter's optimization-level-1 lowering"
        report["resource_adapter_lowering_state_max_abs_error"]=adapter_state_error
        for point in report["pareto_points"]:
            point["peak_qubit_time_product_qubit_seconds"]=point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]*1e-9
            point["conditional_ideal_phase_full_failure_upper"]=record["schedule"]["ideal_failure_probability_bound"]+point["synthesis_model_error_per_execution"]+point["physical_failure_union_bound_all_executions"]
        reports.append(report)
    return {
        "status":"computed_conditional_mean_fit_projection","executions":1,
        "ideal_phase_failure_bound":record["schedule"]["ideal_failure_probability_bound"],
        "configured_synthesis_model_allowance_per_execution":1e-5,
        "physical_failure_allowance":.005,
        "configured_conditional_full_failure_upper":record["schedule"]["ideal_failure_probability_bound"]+1e-5+.005,
        "certified_hardware_confidence":None,
        "assumptions":["actual preparation success satisfies the supplied promise",
                       "the exact YLC phase theorem applies; binary64 angles are not certified enclosures",
                       "the QDK synthesis-model scalar upper-bounds joint output-distribution total variation",
                       "declared physical-fault/QEC/factory model bounds faults",
                       "classical data/flag verification is correct"],
        "limitations":["no concrete Clifford+T sequence or per-angle approximation certificate",
                       "mean fallback work has no runtime-tail certificate",
                       "physical initialization and classical controller/decoder costs excluded",
                       "peak qubits times runtime is a rectangular proxy, not occupancy integration"],
        "profiles":reports,
    }


def figures(output,lower,delta,record,prepared,measured):
    plt.rcParams.update({"svg.hashsalt":"openquantum-fixed-point","font.size":11})
    for filename,circuit in (("preparation.svg",prepared),("circuit.svg",measured)):
        figure=circuit.draw(output="mpl",fold=65,style="iqp")
        figure.savefig(output/filename,bbox_inches="tight",metadata={"Date":None})
        plt.close(figure)
    ps=np.linspace(0,1,201)
    schedule=fixed_point_schedule(lower,delta)
    values=[independent_reference(float(p),schedule.length,delta) for p in ps]
    figure,axis=plt.subplots(figsize=(8,3.5))
    axis.plot(ps,values,label=f"Fixed point: L={schedule.length}",color="#1f807e",lw=2)
    axis.plot(ps,np.sin(schedule.length*np.arcsin(np.sqrt(ps)))**2,
              label=f"Ordinary amplification: {schedule.rounds} rounds",color="#b56325",alpha=.7)
    axis.axvspan(lower,1,color="#1f807e",alpha=.07)
    axis.axhline(1-delta**2,color="#29485b",ls="--",label=r"Ideal promised bound $1-\delta^2$")
    axis.axvline(lower,color="#29485b",ls=":")
    actual=record["parameters"]["exact_preparation_success"]
    axis.scatter([actual],[record["independent_chebyshev_success"]],color="#1f807e",zorder=4)
    axis.set(xlabel="Preparation success probability p",ylabel="Ideal output success probability",ylim=(-.025,1.025),xlim=(0,1))
    axis.legend(loc="lower right",fontsize=9)
    axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output/"success-band.svg",metadata={"Date":None})
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-qubits",type=int,default=2)
    parser.add_argument("--threshold",type=int,default=1)
    parser.add_argument("--lower-bound",type=float,default=.1)
    parser.add_argument("--failure-amplitude",type=float,default=.05)
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--estimate",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/fixed-point-amplification")
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output==ROOT or output in (ROOT/"src",ROOT/"algorithms",ROOT/"scripts"):
        raise ValueError("Use a dedicated artifact output directory")
    output.mkdir(parents=True,exist_ok=True)
    # Remove known optional outputs so a changed invocation cannot publish stale evidence.
    for name in ("physical-resources.json","estimated-circuit.qasm","scaling.csv","physical-scaling.csv","success-band.svg","logical-scaling.svg","physical-scaling.svg","summary.md"):
        (output/name).unlink(missing_ok=True)
    if (output/"scaling").exists():shutil.rmtree(output/"scaling")
    measured,lowered,qasm,record=build_case(args.data_qubits,args.threshold,args.lower_bound,
                                         args.failure_amplitude,not args.no_sim)
    if args.estimate and not record["parameters"]["promise_satisfied"]:
        raise ValueError("Physical guaranteed-success contract requires a valid preparation promise")
    record["provenance"]=provenance()
    record["physical_projection"]={"performed":args.estimate,"conditional_only":True}
    (output/"circuit.qasm").write_text(qasm)
    write_json(output/"logical-resources.json",record)
    rows=[]
    physical_rows=[]
    with tempfile.TemporaryDirectory(prefix="openquantum-fixed-point-qre-") as temporary:
        cache=Path(temporary)
        if args.estimate:
            physical=physical_projection(lowered,record,cache)
            write_json(output/"physical-resources.json",physical)
            (output/"estimated-circuit.qasm").write_text(physical["profiles"][0]["qasm3"])
        if not args.no_scaling:
            for n in (2,3,4,6,8):
                trial,compiled,text,case=build_case(n,2**(n-2),args.lower_bound,args.failure_amplitude,not args.no_sim)
                if args.estimate and not case["parameters"]["promise_satisfied"]:
                    raise ValueError("Scaling family's preparation does not satisfy the configured promise")
                directory=output/"scaling"/f"n{n}"
                directory.mkdir(parents=True)
                (directory/"circuit.qasm").write_text(text)
                write_json(directory/"logical.json",case)
                rows.append({"n_data":n,"threshold":2**(n-2),"algorithm_qubits":case["algorithm_qubits"],
                             "length_L":case["schedule"]["length_L"],"rounds":case["schedule"]["rounds"],
                             "preparation_calls":case["access_ledger"]["preparation_calls"],
                             "inverse_preparation_calls":case["access_ledger"]["inverse_preparation_calls"],
                             "primitive_operations_with_readout":case["primitive_operations_with_readout"],
                             "primitive_dependency_depth":case["primitive_dependency_depth"],
                             "measurements":case["output_measurements"],
                             "statevector_checked":case["simulation"]["performed"],
                             "ideal_success":case["independent_chebyshev_success"]})
                if args.estimate:
                    projection=physical_projection(compiled,case,cache)
                    write_json(directory/"surface-code.json",projection)
                    (directory/"estimated-circuit.qasm").write_text(projection["profiles"][0]["qasm3"])
                    for profile in projection["profiles"]:
                        for point in profile["pareto_points"]:
                            physical_rows.append({"n_data":n,"physical_error_rate":profile["physical_profile"]["physical_error_rate"],
                                "peak_physical_qubits":point["peak_physical_qubits"],
                                "serial_runtime_ns":point["serial_runtime_all_executions_ns"],
                                "peak_qubit_time_product_qubit_seconds":point["peak_qubit_time_product_qubit_seconds"],
                                "data_code_distance":point["data_code_distance"],
                                "conditional_full_failure_upper":point["conditional_ideal_phase_full_failure_upper"]})
    for filename,data in (("scaling.csv",rows),("physical-scaling.csv",physical_rows)):
        if data:
            with (output/filename).open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(data[0]),lineterminator="\n")
                writer.writeheader();writer.writerows(data)
    figures(output,args.lower_bound,args.failure_amplitude,record,threshold_preparation(args.data_qubits,args.threshold),measured)
    if rows:
        figure,axes=plt.subplots(1,2,figsize=(8,3))
        axes[0].plot([r["n_data"] for r in rows],[r["primitive_operations_with_readout"] for r in rows],"o-",color="#1f807e")
        axes[1].plot([r["n_data"] for r in rows],[r["primitive_dependency_depth"] for r in rows],"o-",color="#b56325")
        axes[0].set_ylabel("Primitive operations incl. readout")
        axes[1].set_ylabel("Primitive dependency depth")
        for axis in axes:axis.set_xlabel("Data qubits n");axis.grid(alpha=.2)
        figure.tight_layout();figure.savefig(output/"logical-scaling.svg",metadata={"Date":None});plt.close(figure)
    if physical_rows:
        figure,axes=plt.subplots(1,3,figsize=(10,3.2))
        for error,color in ((1e-3,"#b56325"),(1e-4,"#1f807e")):
            selected=[min((r for r in physical_rows if r["n_data"]==n and r["physical_error_rate"]==error),key=lambda r:r["serial_runtime_ns"]) for n in (2,3,4,6,8)]
            for axis,key,factor,label in zip(axes,("peak_physical_qubits","serial_runtime_ns","peak_qubit_time_product_qubit_seconds"),(1,1e-6,1),("Peak physical qubits","Serial modeled runtime (ms)","Peak qubits × time (qubit s)"),strict=True):
                axis.plot([r["n_data"] for r in selected],[r[key]*factor for r in selected],"o-",color=color,label=f"p={error:g}")
                axis.set(xlabel="Data qubits n",ylabel=label);axis.grid(alpha=.2)
        axes[0].legend(fontsize=9);figure.suptitle("Fastest returned points; conditional mean-fit surface-code model",fontsize=10)
        figure.tight_layout();figure.savefig(output/"physical-scaling.svg",metadata={"Date":None});plt.close(figure)
    for name in ("fixed_point_amplification.py","amplitude_estimation.py","resource_estimation.py"):
        shutil.copy2(ROOT/"src/openquantum"/name,output/name)
    shutil.copy2(Path(__file__),output/"generate_fixed_point_amplification.py")
    shutil.copy2(ROOT/"LICENSE",output/"LICENSE.txt")
    summary=(f"The explicit {record['algorithm_qubits']}-wire trial uses **L={record['schedule']['length_L']}**, "
             f"{record['schedule']['rounds']} phased rounds, {record['access_ledger']['preparation_calls']} forward and "
             f"{record['access_ledger']['inverse_preparation_calls']} inverse preparation calls. "
             f"It exports **{record['primitive_operations_with_readout']:,} primitive operations** including "
             f"{record['output_measurements']} final measurements, with dependency depth {record['primitive_dependency_depth']:,}. "
             f"The independent ideal success probability is {record['independent_chebyshev_success']:.12f}. "
             f"Statevector validation performed: {record['simulation']['performed']}.\n\n")
    if args.estimate:
        summary+=f"The one-trial configured conditional failure allowance is **{args.failure_amplitude**2+1e-5+.005:.8g}**: ideal δ², a 10⁻⁵ synthesis-model allowance and 0.005 physical faults. This is not a certified hardware confidence bound.\n\n"
    if rows:
        summary+="| Data qubits n | Live qubits | Primitive operations incl. readout | Dependency depth |\n|---:|---:|---:|---:|\n"
        summary+="".join(f"| {r['n_data']} | {r['algorithm_qubits']} | {r['primitive_operations_with_readout']:,} | {r['primitive_dependency_depth']:,} |\n" for r in rows)
    (output/"summary.md").write_text(summary)
    # Match the site's whitespace normalization before hashing downloadable files.
    for path in output.rglob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    write_json(output/"manifest.json",{"provenance":record["provenance"],"files_sha256":{str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name!="manifest.json"}})
    print(json.dumps({"length":record["schedule"]["length_L"],"success":record["independent_chebyshev_success"],"primitive_operations":record["primitive_operations_with_readout"],"scaling_cases":len(rows),"physical_points":len(physical_rows)}))


if __name__=="__main__":main()
