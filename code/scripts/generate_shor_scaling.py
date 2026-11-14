#!/usr/bin/env python3
"""Matched, bounded conditional surface-code sweeps of real Shor circuits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
try:
    from source_provenance import git_dirty,git_revision
except ModuleNotFoundError as error:
    if error.name!="source_provenance":
        raise
    def git_revision(_root): return None
    def git_dirty(_root,_paths): return None

from openquantum.resource_estimation import SurfaceCodeProfile,estimate_circuit_resources
from openquantum.shor_scaling import (
    BUILDERS,DEFAULT_CASES,MAX_CASES,MAX_NATIVE_QUERIES,MAX_NATIVE_SECONDS,
    MAX_COMPILED_OPERATIONS,MAX_GENERATED_QUBITS,MatchedBudget,ScalingCase,
    bounded_ripple_multiplier_check,independent_sampling_contract,materialize,
    tiny_variant_checks,
)

SOURCE_PATHS=("src/openquantum/shor.py","src/openquantum/shor_ripple.py",
              "src/openquantum/shor_scaling.py","src/openquantum/resource_estimation.py",
              "scripts/generate_shor_scaling.py","scripts/source_provenance.py","requirements-lock.txt")
FROZEN_PATHS=("assets/generated/shor/circuit.qasm","assets/generated/shor/surface-code.json",
              "assets/generated/shor/ripple/surface-code-comparison.json",
              "assets/generated/grover/circuit.qasm","assets/generated/resources/grover-surface-code.json")


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_hashes():
    # Detached code downloads can omit old publication artifacts.
    return {name:sha(ROOT/name) if (ROOT/name).is_file() else None for name in FROZEN_PATHS}


def environment():
    return {"source_commit":git_revision(ROOT),"owned_source_dirty":git_dirty(ROOT,SOURCE_PATHS),
        "source_sha256":{name:sha(ROOT/name) for name in SOURCE_PATHS if (ROOT/name).exists()},
        "python":platform.python_version(),"device":"CPU","seed":None,
        "packages":{name:importlib.metadata.version(name) for name in ("qiskit","qdk","numpy","matplotlib")},
        "execution":"bounded exact circuit generation, classical diagnostics, isolated conditional QDK queries; no hardware execution"}


def guard_output(output):
    directory=output.resolve()
    for name in ("assets/generated/shor","assets/generated/grover","assets/generated/resources"):
        protected=(ROOT/name).resolve()
        if directory==protected or directory.is_relative_to(protected) or protected.is_relative_to(directory):
            raise ValueError("Scaling outputs must not overlap frozen Shor/Grover artifacts")
    if directory==ROOT or directory==ROOT/"assets" or directory==ROOT/"assets/generated":
        raise ValueError("Choose a dedicated scaling output directory")


def clear_managed_outputs(output):
    if not output.exists():
        return
    names={"study.json","resource-table.md","scaling.csv","physical-scaling.svg","tiny-verification.json"}
    for path in output.rglob("*"):
        relative=path.relative_to(output)
        managed=len(relative.parts)==1 and relative.name in names
        managed=managed or (len(relative.parts)==3 and re.fullmatch(r"n[0-9]+-a[0-9]+-t[0-9]+",relative.parts[0])
            and relative.parts[1] in BUILDERS and relative.parts[2] in {"circuit.qasm","resources.json"})
        if managed and path.is_file():
            path.unlink()


def guard_managed_descendants(output,cases):
    """Reject links before any cleanup or write can reach a different tree."""
    if output.is_symlink():
        raise ValueError("Scaling output root must not be a symlink")
    anchor=output.resolve()
    if output.exists() and any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Scaling output contains a symlink; no managed files changed")
    for case in cases:
        for method in BUILDERS:
            for name in ("circuit.qasm","resources.json"):
                target=output/case.identifier/method/name
                if not target.resolve().is_relative_to(anchor):
                    raise ValueError("Managed output escapes the dedicated scaling directory")


def valid_native_report(report,expected_hash,budget,profile,phase_bits):
    """Validate a computed envelope before publishing any physical numbers."""
    if not isinstance(report,dict) or report.get("status")!="computed_conditional_model":
        return False
    if report.get("qasm_sha256")!=expected_hash or report.get("budget")!=asdict(budget) or report.get("physical_profile")!=asdict(profile):
        return False
    transpiler=report.get("transpiler")
    if report.get("isa_graph_pruning") is not False or not isinstance(transpiler,dict) or transpiler.get("optimization_level")!=1:
        return False
    counts=report.get("canonical_gate_counts")
    if not isinstance(counts,dict) or counts.get("MEAS_Z")!=phase_bits:
        return False
    synthesis=report.get("synthesis")
    if not isinstance(synthesis,dict):
        return False
    scalar=synthesis.get("model_error_per_execution")
    if isinstance(scalar,bool) or not isinstance(scalar,(int,float)) or not math.isfinite(scalar) or not 0<=scalar<=budget.synthesis_model_error_per_execution:
        return False
    points=report.get("pareto_points")
    if not isinstance(points,list) or not points:
        return False
    for point in points:
        if not isinstance(point,dict):
            return False
        qubits=point.get("peak_physical_qubits")
        distance=point.get("data_code_distance")
        if isinstance(qubits,bool) or not isinstance(qubits,int) or qubits<1 or isinstance(distance,bool) or not isinstance(distance,int) or distance<1:
            return False
        numbers={name:point.get(name) for name in ("runtime_per_execution_ns","serial_runtime_all_executions_ns","physical_failure_union_bound_all_executions")}
        if any(isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) for value in numbers.values()):
            return False
        runtime=numbers["runtime_per_execution_ns"]
        if runtime<=0 or not math.isclose(numbers["serial_runtime_all_executions_ns"],runtime*budget.executions,rel_tol=1e-12):
            return False
        if not 0<=numbers["physical_failure_union_bound_all_executions"]<=budget.physical_failure_probability+1e-12:
            return False
    return True


def estimate_worker(config_path):
    config=json.loads(config_path.read_text())
    case=ScalingCase(**config["case"])
    budget=MatchedBudget(**config["matched_budget"])
    started=time.perf_counter_ns()
    try:
        circuit,_,_,checked=materialize(case,config["method"])
        if checked["qasm_sha256"]!=config["expected_qasm_sha256"]:
            raise AssertionError("Native worker rebuild differs from audited export")
        resource_budget=budget.resource_budget(config["trials"])
        profile=SurfaceCodeProfile(**config["profile"])
        report=estimate_circuit_resources(circuit,budget=resource_budget,
            profile=profile,cache_dir=Path(config["cache_dir"]),
            use_graph=False,optimization_level=1)
        if not valid_native_report(report,config["expected_qasm_sha256"],resource_budget,profile,case.phase_bits):
            raise AssertionError("Native report is incomplete or differs from the audited workload")
        report.pop("qasm3")
        report["synthesis_model_error_all_serial_trials"]=config["trials"]*report["synthesis"]["model_error_per_execution"]
        for point in report["pareto_points"]:
            point["peak_times_serial_model_runtime_qubit_ns"]=point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]
        result={"status":"computed_conditional_model","estimate":report,
                "native_input_qasm_matches_export":True}
    except Exception as error:
        # Raw diagnostics stay local; exceptions can contain private cache paths.
        result={"status":"physical_model_unavailable","reason_type":type(error).__name__,
                "reason":"Bounded native query failed; no physical result claimed"}
    result["native_worker_wall_ns"]=time.perf_counter_ns()-started
    write_json(Path(config["result_path"]),result)
    return 0 if result["status"]=="computed_conditional_model" else 2


def isolated_projection(case,method,contract,record,budget,profile,cache_dir,timeout_seconds):
    with tempfile.TemporaryDirectory(prefix="openquantum-shor-qre-") as temporary:
        private=Path(temporary)
        config={"case":asdict(case),"method":method,"trials":contract["complete_serial_trials"],
            "matched_budget":asdict(budget),"profile":asdict(profile),"cache_dir":str(cache_dir),
            "expected_qasm_sha256":record["qasm_sha256"],"result_path":str(private/"result.json")}
        write_json(private/"config.json",config)
        try:
            process=subprocess.run([sys.executable,str(Path(__file__).resolve()),"--_estimate-worker",str(private/"config.json")],
                cwd=ROOT,text=True,capture_output=True,timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            return {"status":"physical_model_unavailable","reason":"Isolated native query exceeded explicit wall cap",
                    "timeout_seconds":timeout_seconds}
        if not (private/"result.json").exists():
            return {"status":"physical_model_unavailable","reason":"Isolated native worker did not produce a complete report",
                    "worker_exit_code":process.returncode}
        if process.returncode!=0:
            return {"status":"physical_model_unavailable","reason":"Isolated native worker returned failure",
                    "worker_exit_code":process.returncode}
        try:
            result=json.loads((private/"result.json").read_text())
        except (OSError,UnicodeError,json.JSONDecodeError):
            return {"status":"physical_model_unavailable","reason":"Isolated native worker returned malformed report data"}
        if not isinstance(result,dict) or result.get("status")!="computed_conditional_model" or result.get("native_input_qasm_matches_export") is not True:
            return {"status":"physical_model_unavailable","reason":"Isolated native worker did not return a valid computed envelope"}
        if not valid_native_report(result.get("estimate"),record["qasm_sha256"],budget.resource_budget(contract["complete_serial_trials"]),profile,case.phase_bits):
            return {"status":"physical_model_unavailable","reason":"Isolated native worker report failed workload/physical validation"}
        model=result["estimate"]
        trials=contract["complete_serial_trials"]
        # Derive presentation totals ourselves; worker fields can be missing or stale.
        model["synthesis_model_error_all_serial_trials"]=trials*model["synthesis"]["model_error_per_execution"]
        for point in model["pareto_points"]:
            point["peak_times_serial_model_runtime_qubit_ns"]=point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]
        accepted={"status":"computed_conditional_model","estimate":model,
                  "native_input_qasm_matches_export":True,"timeout_seconds":timeout_seconds}
        wall=result.get("native_worker_wall_ns")
        if isinstance(wall,int) and not isinstance(wall,bool) and wall>=0:
            accepted["native_worker_wall_ns"]=wall
        return accepted


def summaries(report):
    if report.get("status")!="computed_conditional_model":
        return []
    points=report["estimate"]["pareto_points"]
    result=[]
    for choice,key in (("fewest_qubits",lambda p:(p["peak_physical_qubits"],p["serial_runtime_all_executions_ns"])),
                       ("fastest_model",lambda p:(p["serial_runtime_all_executions_ns"],p["peak_physical_qubits"]))):
        result.append({"choice":choice,**min(points,key=key)})
    return result


def write_tables(output,study):
    import csv
    rows=[]
    lines=["| N / base / phase bits | Method | Ideal success/trial | Complete trials | Logical qubits | Primitive operations/trial | Choice | Peak physical qubits | Serial model time (s) | Peak × serial model time (qubit s) |",
           "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for item in study["cases"]:
        c=item["case"]
        contract=item.get("sampling_contract",{})
        probability=contract.get("ideal_factor_success_probability_per_trial")
        p_text="—" if probability is None else f"{probability:.6g}"
        for name,record in item.get("methods",{}).items():
            physical=record.get("physical",{})
            points=summaries(physical)
            if not points:
                lines.append(f"| {c['modulus']} / {c['base']} / {c['phase_bits']} | {name} | {p_text} | {contract.get('complete_serial_trials') or '—'} | {record['declared_qubits']} | {record['primitive_operations_per_trial']:,} | {physical.get('status','not_estimated')} | — | — | — |")
            for point in points:
                row={"modulus":c["modulus"],"base":c["base"],"integer_bits":c["integer_bits"],"phase_bits":c["phase_bits"],
                    "method":name,"ideal_success_per_trial":contract["ideal_factor_success_probability_per_trial"],
                    "complete_trials":contract["complete_serial_trials"],"logical_qubits":record["declared_qubits"],
                    "primitive_operations_per_trial":record["primitive_operations_per_trial"],"choice":point["choice"],
                    "peak_physical_qubits":point["peak_physical_qubits"],"serial_model_time_seconds":point["serial_runtime_all_executions_ns"]/1e9,
                    "peak_times_serial_model_time_qubit_seconds":point["peak_times_serial_model_runtime_qubit_ns"]/1e9,
                    "data_code_distance":point["data_code_distance"]}
                rows.append(row)
                lines.append(f"| {c['modulus']} / {c['base']} / {c['phase_bits']} | {name} | {row['ideal_success_per_trial']:.6g} | {row['complete_trials']} | {row['logical_qubits']} | {row['primitive_operations_per_trial']:,} | {row['choice']} | {row['peak_physical_qubits']:,} | {row['serial_model_time_seconds']:.6g} | {row['peak_times_serial_model_time_qubit_seconds']:.6g} |")
    (output/"resource-table.md").write_text("\n".join(lines)+"\n")
    study["summary_selections"]=rows
    if rows:
        with (output/"scaling.csv").open("w") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
            writer.writeheader();writer.writerows(rows)
    elif (output/"scaling.csv").exists():
        (output/"scaling.csv").unlink()


def draw_scaling(output,study):
    rows=study["summary_selections"]
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"]="openquantum-shor-scaling"
    import matplotlib.pyplot as plt
    cases=[item["case"] for item in study["cases"] if item.get("sampling_contract",{}).get("finite_retry_projection_supported")]
    figure,axes=plt.subplots(1,3,figsize=(13,4.1))
    colors={"fourier":"#176a69","ripple":"#a34f24"}
    for method in BUILDERS:
        for choice,style in (("fewest_qubits","o-"),("fastest_model","s--")):
            selected=[next((r for r in rows if (r["modulus"],r["base"],r["phase_bits"],r["method"],r["choice"])==
                           (c["modulus"],c["base"],c["phase_bits"],method,choice)),None) for c in cases]
            for axis,field in zip(axes,("peak_physical_qubits","serial_model_time_seconds","peak_times_serial_model_time_qubit_seconds")):
                axis.semilogy(range(len(cases)),[r[field] if r else float("nan") for r in selected],style,color=colors[method],
                              label=f"{method}, {choice.replace('_',' ')}")
                axis.grid(alpha=.2)
    labels=[f"N={c['modulus']}, a={c['base']}\nt={c['phase_bits']}" for c in cases]
    for axis,label in zip(axes,("Peak physical qubits","Serial conditional model time (s)","Peak × serial model time (qubit s)")):
        axis.set_xticks(range(len(cases)),labels)
        axis.set_ylabel(label)
    axes[0].legend(fontsize=7)
    figure.suptitle("Matched fixed-base Fourier and ripple circuits: complete finite retries")
    figure.text(.5,.015,"Frozen p=10⁻⁴ profile; per-case inputs/precision/retries disclosed. Count/layout and mean-fit models; no certified synthesis or factoring-driver guarantee.",ha="center",fontsize=8)
    figure.tight_layout(rect=(0,.06,1,.95))
    figure.savefig(output/"physical-scaling.svg",metadata={"Date":None})
    plt.close(figure)


def generate(output,cases=DEFAULT_CASES,*,physical=True,budget=MatchedBudget(),timeout_seconds=MAX_NATIVE_SECONDS,cache_dir=None):
    guard_output(output)
    if not cases or len(cases)>MAX_CASES or len(set(cases))!=len(cases):
        raise ValueError("Choose1..5 distinct predetermined cases")
    if isinstance(timeout_seconds,bool) or not isinstance(timeout_seconds,(int,float)) or not 0<timeout_seconds<=MAX_NATIVE_SECONDS:
        raise ValueError("Native wall cap must be in (0,90] seconds")
    frozen=frozen_hashes()
    guard_managed_descendants(output,cases)
    output.mkdir(parents=True,exist_ok=True)
    clear_managed_outputs(output)
    profile=SurfaceCodeProfile()  # The existing N15 comparison's frozen profile.
    study={"schema_version":1,"status":"in_progress","environment":environment(),
        "matched_budget":asdict(budget),"frozen_physical_profile":asdict(profile),"cases":[],
        "limits":{"cases":MAX_CASES,"native_queries":MAX_NATIVE_QUERIES,"native_wall_seconds":timeout_seconds,
                  "compiled_operations_per_trial":MAX_COMPILED_OPERATIONS,"generated_qubits":MAX_GENERATED_QUBITS},
        "frozen_sha256_before":frozen,"certified_combined_factoring_success":None,
        "scope":"Actual complete measured fixed-base circuits and serial identical retries; not a random-base/recursive/adaptive factoring-driver projection",
        "exclusions":["classical base selection, constant construction, CF/modular verification/gcd latency from physical clock",
                      "diagnostic order enumeration/FFT costs from algorithm workload","calibrated physical initialization/reset or data loading",
                      "actual Clifford dependency schedule/routing/decoder calibration","concrete angle-synthesis certificate or runtime-tail guarantee"],
        "schedule_boundary":"QDK canonical counts/PSSPC use placeholder wires and abstract the actual source gate DAG; model time is not a validated schedule",
        "input_access":"Classical N/base specify gate constants; no database or amplitude-loading oracle; external I/O not calibrated"}
    write_json(output/"tiny-verification.json",tiny_variant_checks())
    native_queries=0
    cache=cache_dir or Path(tempfile.gettempdir())/"openquantum-shor-scaling-qre-cache"
    for case in cases:
        item={"case":case.as_dict(),"methods":{}}
        study["cases"].append(item)
        for method in BUILDERS:
            directory=output/case.identifier/method
            directory.mkdir(parents=True,exist_ok=True)
            try:
                _,_,qasm,record=materialize(case,method)
            except ValueError as error:
                item["status"]="materialization_unsupported"
                item["reason_type"]=type(error).__name__
                item["reason"]="Bounded arithmetic materialization rejected; no complete circuit report claimed"
                break
            (directory/"circuit.qasm").write_text(qasm)
            record["qasm_relative_path"]=f"{case.identifier}/{method}/circuit.qasm"
            item["methods"][method]=record
        if len(item["methods"])!=2:
            continue
        # Reference comes after the actual circuits, and never supplies gates.
        contract=independent_sampling_contract(case,budget)
        item["sampling_contract"]=contract
        try:
            item["ripple_boolean_validation"]=bounded_ripple_multiplier_check(case)
        except ValueError as error:
            item["ripple_boolean_validation"]={"status":"not_run","reason_type":type(error).__name__,
                "reason":"Existing bounded Boolean validator rejected the workload; no complete arithmetic check claimed"}
        if case.phase_bits<=8:
            try:
                item["ripple_full_source_validation"]=simulate_ripple(case)
            except ValueError as error:
                item["ripple_full_source_validation"]={"status":"not_run","reason_type":type(error).__name__,
                    "reason":"Existing bounded sparse validator rejected the workload; no full simulation claimed"}
        else:
            item["ripple_full_source_validation"]={"status":"not_run","reason":"Existing immutable sparse full-circuit precision cap is8 bits"}
        item["fourier_full_source_validation"]={"status":"not_run_in_scaling_generator",
            "reason":"Tiny coherent/full circuit checks included; larger materializations are not freshly simulated by this generator"}
        for method,record in item["methods"].items():
            trials=contract["complete_serial_trials"]
            if trials is not None:
                record["primitive_operations_all_trials"]=trials*record["primitive_operations_per_trial"]
                record["total_exponent_readouts"]=trials*case.phase_bits
            if not contract["finite_retry_projection_supported"]:
                projection={"status":"not_estimated_unsupported_retry_contract","reason":contract["unsupported_reason"]}
            elif not physical:
                projection={"status":"not_estimated_disabled"}
            elif native_queries>=MAX_NATIVE_QUERIES:
                projection={"status":"not_estimated_query_cap","reason":"Whole-generator native query cap reached"}
            else:
                native_queries+=1
                projection=isolated_projection(case,method,contract,record,budget,profile,cache,timeout_seconds)
            record["physical"]=projection
            write_json(output/case.identifier/method/"resources.json",record)
            print(case.identifier,method,record["primitive_operations_per_trial"],projection["status"],flush=True)
        item["status"]="completed_case"
        write_json(output/"study.json",study)
    study["native_queries_attempted"]=native_queries
    after=frozen_hashes()
    if frozen!=after:
        raise AssertionError("Frozen Shor/Grover baseline changed")
    available=all(value is not None for value in frozen.values())
    study.update(frozen_files_unchanged=True if available else None,frozen_reference_files_available=available,
                 frozen_sha256_after=after,
                 frozen_preservation_scope="All present reference artifacts unchanged; missing artifacts cannot be verified")
    supported=[item for item in study["cases"] if item.get("sampling_contract",{}).get("finite_retry_projection_supported")]
    complete=all(item.get("status")=="completed_case" for item in study["cases"])
    success=complete and bool(supported) and all(record["physical"]["status"]=="computed_conditional_model" for item in supported for record in item["methods"].values())
    study["status"]="computed_conditional_model_with_explicit_unsupported_controls" if physical and success else "logical_only" if not physical and complete else "partial_or_unsupported"
    write_tables(output,study)
    draw_scaling(output,study)
    write_json(output/"study.json",study)
    return study


def simulate_ripple(case):
    from openquantum.shor_ripple import simulate_order_finding_ripple
    return {"status":"checked",**simulate_order_finding_ripple(case.modulus,case.base,case.phase_bits)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/shor-scaling")
    parser.add_argument("--case",action="append",help="Predetermined modulus:base:phase_bits; repeat up to5 times")
    parser.add_argument("--no-physical",action="store_true")
    parser.add_argument("--native-timeout",type=float,default=MAX_NATIVE_SECONDS)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--_estimate-worker",type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._estimate_worker:
        return estimate_worker(args._estimate_worker)
    cases=tuple(ScalingCase(*map(int,value.split(":"))) for value in args.case) if args.case else DEFAULT_CASES
    result=generate(args.output_dir,cases,physical=not args.no_physical,timeout_seconds=args.native_timeout,cache_dir=args.cache_dir)
    print(json.dumps({"status":result["status"],"native_queries":result["native_queries_attempted"],"frozen_files_unchanged":result["frozen_files_unchanged"]}))
    return 0 if result["status"] in ("computed_conditional_model_with_explicit_unsupported_controls","logical_only") else 2


if __name__=="__main__":
    raise SystemExit(main())
