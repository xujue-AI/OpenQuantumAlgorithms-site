#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixed-grid exact versus truncated QPE with matched conditional output costs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
try:
    from source_provenance import git_revision,git_dirty
except ModuleNotFoundError as error:
    if error.name!="source_provenance":raise
    def git_revision(root):return None
    def git_dirty(root,paths):return None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator,Statevector

from openquantum.approximate_qft_resources import (
    MAX_PRIMITIVE_OPERATIONS,build_contract_trial,checked_distribution,fixed_grid,
)
from openquantum.qft_qpe import build_qft,qft_matrix,qpe_distribution
from openquantum.resource_estimation import SurfaceCodeProfile,estimate_circuit_resources,prepare_measured_circuit


def sha256(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def environment():
    names=("approximate_qft_resources.py","qft_qpe.py","resource_estimation.py")
    paths=["src/openquantum/"+name for name in names]+["scripts/generate_qft_variant_comparison.py"]
    versions={}
    for name in ("qiskit","numpy","matplotlib","qdk"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return {"python":platform.python_version(),"device":"CPU","system":platform.system(),
            "machine":platform.machine(),"simulation_seed":None,"deterministic_simulation":True,
            "transpiler_seed":7,
            "packages":versions,
            "source_base_commit":git_revision(ROOT),"owned_source_dirty":git_dirty(ROOT,paths),
            "source_commit_is_full_source_certificate":False,
            "source_sha256":{name:sha256(ROOT/"src/openquantum"/name) for name in names},
            "generator_sha256":sha256(Path(__file__)),"requirements_lock_sha256":sha256(ROOT/"requirements-lock.txt"),
            "hardware_execution":False}


def identifier(contract):
    return f"m{contract.phase_bits}-b{contract.max_distance}-s{contract.synthesis_model_allowance_per_trial:g}"


def checked_case(contract,simulate):
    started=time.perf_counter()
    source=build_contract_trial(contract,measure=True)
    compiled=prepare_measured_circuit(source)
    if len(compiled.data)>MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("QPE variant primitive export exceeds its operation cap")
    qasm=qasm3.dumps(compiled)
    simulation={"performed":False,"actual_single_success":None,"output_tv_from_exact":None,
                "exact_dirichlet_max_error":None,"primitive_lowering_state_max_abs_error":None,
                "qft_spectral_omission_error":None}
    if simulate:
        coherent=source.remove_final_measurements(inplace=False)
        actual=checked_distribution(coherent,contract.phase_bits)
        exact=qpe_distribution(contract.phase,contract.phase_bits)
        reference_state=Statevector.from_instruction(coherent).data
        lowered_state=Statevector.from_instruction(compiled.remove_final_measurements(inplace=False)).data
        lowering_error=float(np.max(np.abs(reference_state-lowered_state)))
        if lowering_error>3e-11:raise RuntimeError("Resource lowering changed the supplied trial state")
        tv=.5*float(np.abs(actual-exact).sum())
        if tv>float(contract.omission_output_tv_upper)+3e-11:
            raise RuntimeError("Actual output deviation exceeded omission-TV upper bound")
        dirichlet_error=None
        if contract.max_distance==contract.phase_bits-1:
            dirichlet_error=float(np.max(np.abs(actual-exact)))
            if dirichlet_error>3e-11:raise RuntimeError("Exact QPE disagrees with independent Dirichlet reference")
        qft_error=float(np.linalg.norm(Operator(build_qft(contract.phase_bits,max_distance=contract.max_distance)).data-qft_matrix(contract.phase_bits),ord=2))
        if qft_error>contract.as_metadata()["existing_numeric_spectral_omission_bound"]+3e-11:
            raise RuntimeError("Actual QFT operator exceeded the telescoping omission bound")
        labels=np.arange(len(actual))/len(actual)
        good=np.abs(labels-contract.phase)<=contract.phase_error_target+1e-14
        simulation.update(performed=True,actual_single_success=float(actual[good].sum()),
                          output_tv_from_exact=tv,exact_dirichlet_max_error=dirichlet_error,
                          primitive_lowering_state_max_abs_error=lowering_error,
                          qft_spectral_omission_error=qft_error,
                          phase_output_probabilities=actual.tolist())
    counts=dict(compiled.count_ops())
    per_trial=sum(value for name,value in counts.items() if name!="barrier")
    repetitions=contract.repetitions
    logical={"case_id":identifier(contract),"contract":contract.as_metadata(),"source_access_ledger":source.metadata,
             "source_gate_counts":dict(source.count_ops()),"primitive_gate_counts":counts,
             "live_quantum_wires":compiled.num_qubits,"classical_output_bits_per_trial":compiled.num_clbits,
             "primitive_operations_per_trial_including_readout":per_trial,
             "primitive_dependency_depth_per_trial":compiled.depth(),
             "all_trials_primitive_operations":per_trial*repetitions,
             "all_trials_controlled_unitary_calls":(2**contract.phase_bits-1)*repetitions,
             "all_trials_input_preparations":repetitions,
             "all_trials_phase_measurements":contract.phase_bits*repetitions,
             "primitive_dependency_depth_serial_sum":compiled.depth()*repetitions,
             "qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),"simulation":simulation,
             "qasm_scope":"complete measured trial; overall global phase omitted by QASM is irrelevant for this output, not a controlled-oracle specification",
             "generation_compilation_reference_wall_time_s":time.perf_counter()-started,
             "included":["actual supplied eigenstate preparation","every repeated controlled U","retained inverse-QFT gates and swaps","all phase readout","every fresh trial"],
             "exclusions":["physical initialization/reset protocol","classical median/verification/controller/decoder latency","external data loading","certified binary64 phase bounds","concrete Clifford+T synthesis and runtime-tail certificates"]}
    return source,compiled,qasm,logical


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate",action="store_true")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/qft-variant-comparison")
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if output in (ROOT,ROOT/"src",ROOT/"scripts",ROOT/"algorithms"):
        raise ValueError("Use a dedicated artifact directory")
    output.mkdir(parents=True,exist_ok=True)
    for name in ("physical-pareto.csv","matched-comparison.csv","physical-comparison.svg","summary.md"):
        (output/name).unlink(missing_ok=True)
    if (output/"cases").exists():shutil.rmtree(output/"cases")
    grid=fixed_grid()
    provenance=environment()
    write_json(output/"study.json",{
        "status":"predeclared_bounded_grid","cases":[contract.as_metadata() for contract in grid],
        "physical_profiles":[{"physical_error_rate":p,"gate_time_ns":100,"measurement_time_ns":500} for p in (1e-3,1e-4)],
        "provenance":provenance,"grid_selected_before_simulation_or_physical_results":True,
        "task":"known-phase pedagogical QPE; scalar median, no claimed application advantage",
        "public_physical_projection_requested":args.estimate,"simulation_requested":not args.no_sim})
    logical_rows=[];physical_rows=[];reports={}
    with tempfile.TemporaryDirectory(prefix="openquantum-qft-variant-qre-") as temporary:
        cache=Path(temporary)
        for contract in grid:
            source,compiled,qasm,logical=checked_case(contract,not args.no_sim)
            directory=output/"cases"/identifier(contract)
            directory.mkdir(parents=True)
            (directory/"circuit.qasm").write_text(qasm)
            write_json(directory/"logical.json",logical)
            logical_rows.append({"case_id":identifier(contract),"phase_bits":contract.phase_bits,
                "cutoff":contract.max_distance,"variant":logical["contract"]["variant"],
                "synthesis_allowance":contract.synthesis_model_allowance_per_trial,
                "phase_error_target":contract.phase_error_target,"repetitions":contract.repetitions,
                "live_qubits":logical["live_quantum_wires"],
                "omission_tv_upper":float(contract.omission_output_tv_upper),
                "single_failure_upper":float(contract.single_trial_failure_upper),
                "median_failure_upper":logical["contract"]["median_failure_upper"],
                "per_trial_primitive_operations":logical["primitive_operations_per_trial_including_readout"],
                "all_trial_primitive_operations":logical["all_trials_primitive_operations"],
                "all_trial_controlled_U_calls":logical["all_trials_controlled_unitary_calls"],
                "all_trial_preparations":logical["all_trials_input_preparations"],
                "all_trial_readouts":logical["all_trials_phase_measurements"],
                "dependency_depth_per_trial":logical["primitive_dependency_depth_per_trial"],
                "simulation_performed":logical["simulation"]["performed"],
                "actual_single_success":logical["simulation"]["actual_single_success"],
                "output_tv_from_exact":logical["simulation"]["output_tv_from_exact"]})
            if args.estimate:
                for error in (1e-3,1e-4):
                    profile=SurfaceCodeProfile(name=f"illustrative-p-{error:g}",physical_error_rate=error,gate_time_ns=100,measurement_time_ns=500)
                    report=estimate_circuit_resources(source,budget=contract.resource_budget,profile=profile,cache_dir=cache)
                    if report["qasm_sha256"]!=logical["qasm_sha256"]:
                        raise RuntimeError("Physical projection does not match the exported measured trial")
                    report["qpe_variant_contract"]=logical["contract"]
                    report["logical_measured_trial_qasm_sha256"]=logical["qasm_sha256"]
                    report["physical_initialization_cost_included"]=False
                    for point in report["pareto_points"]:
                        point["peak_qubit_time_product_qubit_seconds"]=point["peak_physical_qubits"]*point["serial_runtime_all_executions_ns"]*1e-9
                        point["conditional_median_plus_physical_failure_upper"]=logical["contract"]["median_failure_upper"]+point["physical_failure_union_bound_all_executions"]
                        physical_rows.append({"case_id":identifier(contract),"phase_bits":contract.phase_bits,
                            "cutoff":contract.max_distance,"variant":logical["contract"]["variant"],
                            "synthesis_allowance":contract.synthesis_model_allowance_per_trial,"physical_error_rate":error,
                            "repetitions":contract.repetitions,"peak_physical_qubits":point["peak_physical_qubits"],
                            "serial_runtime_all_trials_ns":point["serial_runtime_all_executions_ns"],
                            "runtime_per_trial_ns":point["runtime_per_execution_ns"],
                            "peak_qubit_time_product_qubit_seconds":point["peak_qubit_time_product_qubit_seconds"],
                            "data_code_distance":point["data_code_distance"],
                            "conditional_full_failure_upper":point["conditional_median_plus_physical_failure_upper"]})
                    write_json(directory/f"surface-code-p-{error:g}.json",report)
                    reports[(contract.phase_bits,contract.max_distance,contract.synthesis_model_allowance_per_trial,error)]=report
    write_csv(output/"logical.csv",logical_rows)
    comparisons=[]
    if physical_rows:
        write_csv(output/"physical-pareto.csv",physical_rows)
        for m in (5,6,7):
            for synthesis in (1e-4,1e-5):
                for error in (1e-3,1e-4):
                    exact=reports[(m,m-1,synthesis,error)]
                    truncated=reports[(m,m-2,synthesis,error)]
                    select=lambda report:min(report["pareto_points"],key=lambda p:p["serial_runtime_all_executions_ns"])
                    a,b=select(exact),select(truncated)
                    comparisons.append({"phase_bits":m,"phase_error_target":2.**(-m),"phase":.3,
                        "synthesis_allowance":synthesis,"physical_error_rate":error,"total_failure_target":.01,
                        "exact_trials":exact["budget"]["executions"],"truncated_trials":truncated["budget"]["executions"],
                        "exact_peak_qubits_at_fastest":a["peak_physical_qubits"],"truncated_peak_qubits_at_fastest":b["peak_physical_qubits"],
                        "exact_fastest_serial_ms":a["serial_runtime_all_executions_ns"]*1e-6,
                        "truncated_fastest_serial_ms":b["serial_runtime_all_executions_ns"]*1e-6,
                        "truncated_over_exact_fastest_runtime_ratio":b["serial_runtime_all_executions_ns"]/a["serial_runtime_all_executions_ns"],
                        "exact_peak_qubit_time_product":a["peak_qubit_time_product_qubit_seconds"],
                        "truncated_peak_qubit_time_product":b["peak_qubit_time_product_qubit_seconds"],
                        "truncated_over_exact_peak_qubit_time_ratio":b["peak_qubit_time_product_qubit_seconds"]/a["peak_qubit_time_product_qubit_seconds"]})
        write_csv(output/"matched-comparison.csv",comparisons)
        plt.rcParams.update({"svg.hashsalt":"openquantum-qft-variant-comparison"})
        fig,axes=plt.subplots(1,2,figsize=(9,3.4))
        for synthesis,axis in zip((1e-4,1e-5),axes,strict=True):
            for error,color in ((1e-3,"#b56325"),(1e-4,"#1f807e")):
                selected=[row for row in comparisons if row["synthesis_allowance"]==synthesis and row["physical_error_rate"]==error]
                axis.plot([r["phase_bits"] for r in selected],[r["truncated_over_exact_fastest_runtime_ratio"] for r in selected],"o-",color=color,label=f"physical p={error:g}")
            axis.axhline(1,color="#29485b",ls="--")
            axis.set(xlabel="Phase bits m",ylabel="Truncated / exact serial runtime",title=f"Per-trial synthesis allowance {synthesis:g}")
            axis.grid(alpha=.2);axis.legend(fontsize=9)
        fig.suptitle("Same phase precision and failure target; fastest returned conditional points",fontsize=10)
        fig.tight_layout();fig.savefig(output/"physical-comparison.svg",metadata={"Date":None});plt.close(fig)
    summary="""The predeclared comparison uses phase **0.3**, one explicitly prepared target wire, phase sizes **m=5,6,7**, exact cutoff **m−1** and truncated cutoff **m−2**, with one omitted controlled phase. Both variants keep the same phase target 2⁻ᵐ, synthesis allowance, 1% whole-run failure target and hardware profile. The exact rational confidence allocation charges omission as a single-trial output-TV loss, separately from synthesis and whole-run physical faults.\n\n| m | Variant | Fresh trials | Operations/trial | Operations/all trials | Omission TV upper |\n|---:|---|---:|---:|---:|---:|\n"""
    for row in logical_rows:
        if row["synthesis_allowance"]==1e-5:
            summary+=f"| {row['phase_bits']} | {row['variant']} | {row['repetitions']} | {row['per_trial_primitive_operations']:,} | {row['all_trial_primitive_operations']:,} | {row['omission_tv_upper']:.6g} |\n"
    summary+="\nOmission does not change the declared numerical phase tolerance. Its uniform distribution bound increases repetitions. Binary64 phase angles, actual synthesis sequences/runtime tails and hardware confidence are not certified; physical initialization/reset and classical decoding/verification/controller latency are excluded.\n"
    if comparisons:
        ratios=[r["truncated_over_exact_fastest_runtime_ratio"] for r in comparisons]
        summary+=f"\nThe 24 bounded resource queries returned **{len(physical_rows)} Pareto points**. At the fastest returned points, truncated/exact total-runtime ratios range from **{min(ratios):.3f}** to **{max(ratios):.3f}**. These conditional mean-fit results retain all fresh trials and do not establish an application advantage.\n"
    (output/"summary.md").write_text(summary)
    for name in ("approximate_qft_resources.py","qft_qpe.py","resource_estimation.py"):
        shutil.copy2(ROOT/"src/openquantum"/name,output/name)
    shutil.copy2(Path(__file__),output/"generate_qft_variant_comparison.py")
    shutil.copy2(ROOT/"LICENSE",output/"LICENSE.txt")
    for path in output.rglob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    write_json(output/"manifest.json",{"provenance":provenance,"files_sha256":{str(path.relative_to(output)):sha256(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name!="manifest.json"}})
    print(json.dumps({"cases":len(grid),"simulation_performed":not args.no_sim,"physical_queries":len(reports),"physical_points":len(physical_rows)}))


if __name__=="__main__":main()
