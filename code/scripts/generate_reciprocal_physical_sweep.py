#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute the predeclared fixed-contract reciprocal physical size/gap grid."""
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
from types import FunctionType,SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openquantum.reciprocal_physical_sweep import (
    CASES,CONTRACT,MAX_NATIVE_PRIMITIVES,MAX_NATIVE_WIRES,NATIVE_TIMEOUT_SECONDS,
    build_case,resource_budget,
)
from openquantum.resource_estimation import SurfaceCodeProfile,estimate_circuit_resources
from qiskit import qasm3

SOURCE_FILES=(
    "src/openquantum/reciprocal_physical_sweep.py","scripts/generate_reciprocal_physical_sweep.py",
    "src/openquantum/linear_systems_qsvt_reciprocal.py","src/openquantum/reciprocal_phase_certificate.py",
    "src/openquantum/resource_estimation.py","src/openquantum/simulation/lcu.py",
    "src/openquantum/simulation/qsp.py","src/openquantum/linear_systems_qsvt.py",
    "src/openquantum/linear_inverse_lcu.py",
)


def source_revision(root=ROOT):
    if not (root/".git").exists():return None
    try:
        top=subprocess.run(["git","rev-parse","--show-toplevel"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
        if Path(top).resolve()!=root.resolve():return None
        return subprocess.run(["git","rev-parse","HEAD"],cwd=root,text=True,capture_output=True,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError):return None


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def predeclared_plan():
    return {
        "schema_version":1,"cases":[case.as_metadata() for case in CASES],"uniform_contract":CONTRACT,
        "degree_rule":"smallestK<=31 passing the exact rational ideal-phase-adjusted Pauli bias<=.01",
        "phase_rule":"current method API: frozen default atK16; deterministic symmetric fitter at otherK",
        "retry_rule":"source interval/phase acceptance lower bound rounded downward; observed probability is validation only",
        "family":"A=R(q+p)I-(q-p)*(sumZ+sumXX),delta=p/q,R=2n-1,alpha=2qR",
        "validation":"every supported case: explicit input/access/readout, independent Kronecker solve and canonical QASM vector replay",
        "native_query_cap":{"cases":7,"wires":MAX_NATIVE_WIRES,"primitive_operations":MAX_NATIVE_PRIMITIVES,"seconds_per_case":NATIVE_TIMEOUT_SECONDS},
        "expected_unsupported":["n2-gap1of3","n2-gap1of4"],
        "qasm_literal_policy":"Qiskit disable_constants=True for both independently replayed trial and native estimator input",
        "scope":"conditional surface-code resource scaling for this implemented geometric reciprocal; no algorithmic optimality or hardware confidence claim",
    }


def numeric_literal_estimate(circuit,**kwargs):
    """Run unchanged pinned estimator code with an isolated precise exporter.

    The resource code/model/queries are identical. Only its exporter binding
    receives the official numeric-literal option. Copying the function's
    global namespace avoids mutating shared modules, source or other jobs.
    The native report's QASM hash must equal the independently replayed trial.
    """
    namespace=dict(estimate_circuit_resources.__globals__)
    namespace["qasm3"]=SimpleNamespace(dumps=lambda circuit:qasm3.dumps(circuit,disable_constants=True))
    estimator=FunctionType(estimate_circuit_resources.__code__,namespace,
                           estimate_circuit_resources.__name__,estimate_circuit_resources.__defaults__,
                           estimate_circuit_resources.__closure__)
    estimator.__kwdefaults__=dict(estimate_circuit_resources.__kwdefaults__ or {})
    return estimator(circuit,**kwargs)


def worker(case_id,output,cache_dir):
    case=next(case for case in CASES if case.case_id==case_id)
    measured,qasm,metadata=build_case(case)
    budget=resource_budget(metadata)
    started=time.perf_counter()
    try:
        report=numeric_literal_estimate(measured,budget=budget,profile=SurfaceCodeProfile(),cache_dir=cache_dir,
                                       optimization_level=0,use_graph=False)
    except RuntimeError as error:
        if str(error)!="No feasible estimate under the explicit synthesis and physical budgets":raise
        report={"status":"unavailable_native_model_at_fixed_budgets","reason":str(error),"pareto_points":[],
                "budget":vars(budget),"physical_profile":vars(SurfaceCodeProfile())}
    if report.get("pareto_points"):
        if report["qasm_sha256"]!=metadata["qasm_sha256"]:
            raise RuntimeError("physical native query QASM differs from independently replayed complete trial")
        report.pop("qasm3",None)
        for point in report["pareto_points"]:
            if point["physical_failure_union_bound_all_executions"]>.005+1e-12:
                raise RuntimeError("native point exceeds the total capped physical-failure allocation")
            point["peak_footprint_times_serial_seconds"]=point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]/1e9
        report["spacetime_metric_scope"]="peak footprint times capped serial duration, a proxy; not integrated time-varying qubit occupancy"
    report["native_query_seconds_CPU"]=time.perf_counter()-started
    report["complete_trial_qasm_sha256"]=hashlib.sha256(qasm.encode()).hexdigest()
    report["output_contract"]=CONTRACT
    report["qasm_literal_policy"]="official disable_constants=True; unchanged estimator code in an isolated exporter namespace"
    report["workload_executions_source"]="all capped complete attempts from the uniform phase-adjusted success bound"
    (output/"surface-code.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def run(output,no_estimate=False,selected=None):
    output.mkdir(parents=True,exist_ok=True)
    plan=predeclared_plan()
    (output/"predeclared-plan.json").write_text(json.dumps(plan,indent=2)+"\n")
    provenance={"source_revision_at_generation":source_revision(),
                "source_hashes_sha256":{path:hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in SOURCE_FILES},
                "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","qdk")},
                "seed":7,"device":"CPU","source_note":"ROOT must own.git and equal the Git top-level; exact content hashes identify working-tree code"}
    cases=[case for case in CASES if selected is None or case.case_id in selected]
    results=[];summary=[];pareto=[]
    private_logs=ROOT/".cache/reciprocal-physical-sweep"
    private_logs.mkdir(parents=True,exist_ok=True)
    for case in cases:
        case_output=output/case.case_id;case_output.mkdir(exist_ok=True)
        started=time.perf_counter()
        try:
            measured,qasm,metadata=build_case(case)
        except ValueError as error:
            if "degree" not in str(error):raise
            record={**case.as_metadata(),"status":"unsupported_degree_at_fixed_accuracy","reason":str(error),"uniform_contract":CONTRACT,
                    "circuit_generated":False,"validation_performed":False,"native_query_performed":False,"budgets_relaxed":False}
            (case_output/"unsupported.json").write_text(json.dumps(record,indent=2)+"\n")
            results.append(record)
            summary.append({"case":case.case_id,"n":case.n,"gap":float(case.delta),"K":None,"degree":None,"status":record["status"],
                            "logical_qubits":None,"primitive_operations":None,"observed_acceptance":None,"certified_retry_success":None,
                            "capped_attempts":None,"actual_Z_bias":None,"pareto_points":0,"smallest_peak_qubits":None,
                            "fastest_capped_seconds":None,"smallest_peak_footprint_time_proxy":None})
            print(json.dumps({"case":case.case_id,"status":record["status"]}),flush=True)
            continue
        metadata["classical_generation_validation_seconds_CPU"]=time.perf_counter()-started
        metadata["source_provenance"]=provenance
        (case_output/"measured-trial.qasm").write_text(qasm)
        (case_output/"logical-resources.json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n")
        if no_estimate:
            physical={"status":"native_query_not_requested","pareto_points":[],"output_contract":CONTRACT}
            (case_output/"surface-code.json").write_text(json.dumps(physical,indent=2)+"\n")
        else:
            command=[sys.executable,str(Path(__file__).resolve()),"--worker",case.case_id,"--output-dir",str(case_output),
                     "--cache-dir",str(private_logs/"qdk-cache")]
            try:
                completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=NATIVE_TIMEOUT_SECONDS)
                if completed.returncode:
                    (private_logs/f"{case.case_id}.stderr.log").write_text(completed.stderr)
                    physical={"status":"native_query_failed","exit_code":completed.returncode,"pareto_points":[],
                              "public_reason":"native query failed; raw diagnostic log is private","budgets_relaxed":False}
                    (case_output/"surface-code.json").write_text(json.dumps(physical,indent=2)+"\n")
                else:physical=json.loads((case_output/"surface-code.json").read_text())
            except subprocess.TimeoutExpired:
                physical={"status":"native_query_timeout_at_predeclared_cap","timeout_seconds":NATIVE_TIMEOUT_SECONDS,"pareto_points":[],"budgets_relaxed":False}
                (case_output/"surface-code.json").write_text(json.dumps(physical,indent=2)+"\n")
        points=physical["pareto_points"]
        result={**case.as_metadata(),"status":physical["status"],"circuit_generated":True,"validation_performed":True,
                "native_query_performed":not no_estimate,"pareto_points":len(points),"budgets_relaxed":False,
                "qasm_sha256":metadata["qasm_sha256"],"logical_report":f"{case.case_id}/logical-resources.json",
                "physical_report":f"{case.case_id}/surface-code.json","trial_export":f"{case.case_id}/measured-trial.qasm"}
        results.append(result)
        summary.append({"case":case.case_id,"n":case.n,"gap":float(case.delta),"K":metadata["K"],"degree":metadata["degree"],"status":physical["status"],
                        "logical_qubits":metadata["logical_qubits"],"primitive_operations":metadata["primitive_operations"],
                        "observed_acceptance":metadata["validation"]["success_observed_ideal_numeric"],
                        "certified_retry_success":metadata["workload"]["success_lower_bound_used"],"capped_attempts":metadata["workload"]["total_capped_attempts"],
                        "actual_Z_bias":metadata["validation"]["z_bias"],"pareto_points":len(points),
                        "smallest_peak_qubits":min(p["peak_physical_qubits"] for p in points) if points else None,
                        "fastest_capped_seconds":min(p["serial_runtime_all_executions_ns"]/1e9 for p in points) if points else None,
                        "smallest_peak_footprint_time_proxy":min(p["peak_footprint_times_serial_seconds"] for p in points) if points else None})
        for index,point in enumerate(points):
            pareto.append({"case":case.case_id,"n":case.n,"gap":float(case.delta),"K":metadata["K"],"point":index,
                           "peak_physical_qubits":point["peak_physical_qubits"],"data_distance":point["data_code_distance"],
                           "runtime_per_attempt_ns":point["runtime_per_execution_ns"],"capped_attempts":metadata["workload"]["total_capped_attempts"],
                           "capped_serial_seconds":point["serial_runtime_all_executions_ns"]/1e9,
                           "peak_footprint_time_proxy":point["peak_footprint_times_serial_seconds"],
                           "physical_failure_union_all_attempts":point["physical_failure_union_bound_all_executions"]})
        print(json.dumps({"case":case.case_id,"status":physical["status"],"pareto_points":len(points)}),flush=True)
    manifest={"schema_version":1,"predeclared_plan_sha256":fingerprint(plan),"uniform_contract":CONTRACT,
              "source_provenance":provenance,"cases":results,"summary_report":"summary.csv","pareto_report":"pareto.csv" if pareto else None,
              "native_queries_completed":sum(bool(r.get("pareto_points")) for r in results),
              "native_queries_requested":sum(r.get("native_query_performed",False) for r in results),
              "executed_case_ids":[case.case_id for case in cases],
              "scope":"conditional physical scaling under fixed output/failure/hardware assumptions; no observed-p retry shortcut, no accuracy relaxation"}
    write_csv(output/"summary.csv",summary)
    if pareto:write_csv(output/"pareto.csv",pareto)
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.3))
    for column,field,xfield,label in ((0,"n","n","Data qubits n, promised gap .5"),(1,"gap","gap","Promised gap, n=2")):
        rows=[r for r in summary if r["pareto_points"] and (r["gap"]==.5 if field=="n" else r["n"]==2)]
        rows.sort(key=lambda r:r[xfield])
        for row,metric,scale,ylabel in ((0,"fastest_capped_seconds",86400,"Fastest modeled capped serial days"),
                                       (1,"smallest_peak_qubits",1000,"Smallest modeled peak qubits (thousands)")):
            axis=axes[row,column]
            if rows:axis.plot([r[xfield] for r in rows],[r[metric]/scale for r in rows],"o-")
            else:axis.text(.5,.5,"No completed native points",transform=axis.transAxes,ha="center")
            axis.set(xlabel=label,ylabel=ylabel);axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"physical-scaling.svg");plt.close(fig)
    svg=output/"physical-scaling.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines())+"\n")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/linear-systems/reciprocal-qsvt-physical")
    parser.add_argument("--no-estimate",action="store_true")
    parser.add_argument("--case",action="append",choices=[case.case_id for case in CASES])
    parser.add_argument("--worker",choices=[case.case_id for case in CASES])
    parser.add_argument("--cache-dir",type=Path)
    args=parser.parse_args()
    if args.worker:
        if args.cache_dir is None:parser.error("worker cache required")
        worker(args.worker,args.output_dir,args.cache_dir)
    else:run(args.output_dir,args.no_estimate,set(args.case) if args.case else None)


if __name__=="__main__":main()
