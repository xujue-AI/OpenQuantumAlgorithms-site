#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate bounded retargetable thermal QSP circuits, checks and logical ledgers.

Fixed local couplings are retained while Hamiltonian size and beta change.
Every distinct beta*alpha/2 is fitted and independently interval validated.
The ideal scalar certificate is separate from implemented loading, compiler,
synthesis and measurement/reset correctness. Use --no-sim for materialization.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
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
LOCAL_MODULE = (HERE/"thermal_spectral.py").is_file()
REPOSITORY = bool(not LOCAL_MODULE and HERE==ROOT/"scripts" and (SOURCE/"openquantum/simulation/thermal_spectral.py").is_file())
sys.path.insert(0,str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "openquantum-thermal-spectral"
import matplotlib.pyplot as plt
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator, Statevector

if REPOSITORY:
    from openquantum.simulation import thermal_spectral as ts
    from openquantum.simulation.qsp_phase_generation import real_qsp_response
    from openquantum.simulation.artifact_provenance import git_revision, git_dirty
else:
    import thermal_spectral as ts
    from qsp_phase_generation import real_qsp_response
    from artifact_provenance import git_revision, git_dirty

MAX_CASES = 12
MAX_TOTAL_EXPORTED_OPERATIONS = 1_000_000
SOURCE_NAMES = ("thermal_spectral.py","retargetable_qsp.py","lcu.py",
                "qsp_phase_generation.py","qsp_certificate.py","artifact_provenance.py",
                "generate_simulation_thermal_spectral.py")
MANAGED_PLOTS = ("circuit.svg","response.svg","accuracy.svg","scaling.svg")
QASM_POLICY = (
    "Qiskit OpenQASM3 export omits the recorded overall compiled phase. "
    "Measured probabilities are unchanged; these files are not phase-preserving "
    "specifications for coherently controlled reuse. Source circuits retain phases."
)


def digest(raw:bytes)->str:
    return hashlib.sha256(raw).hexdigest()


def write_json(path:Path,value)->None:
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path:Path,rows:list[dict])->None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def slug(value:float)->str:
    return format(value,".17g").replace("-","m").replace(".","p").replace("+","")


def case_id(n:int,beta:float)->str:
    return f"n{n}-beta-{slug(beta)}"


def _phase_plan(terms,beta,filter_error,gibbs_error):
    # Fraction sums retain the exact binary64 input parameter, before one float conversion.
    alpha=sum((Fraction.from_float(abs(c)) for c,_ in terms),Fraction())
    a=float(Fraction.from_float(float(beta))*alpha/2)
    return ts.generate_thermal_phases(a,filter_error,gibbs_error)


def _primitive_record(circuit)->dict:
    counts={k:int(v) for k,v in circuit.count_ops().items()}
    return {"logical_qubits":circuit.num_qubits,"classical_bits":circuit.num_clbits,
            "primitive_operations":circuit.size(),"dependency_depth":circuit.depth(),
            "primitive_gate_counts":counts,
            "continuous_rotations":sum(counts.get(g,0) for g in ("rx","ry","rz")),
            "compiled_global_phase_radians":float(circuit.global_phase)}


def generate(output:Path,n_system:int=2,beta:float=1.5,J:float=.6,h:float=.35,
             systems=None,betas=None,filter_error:float=1e-6,gibbs_error:float|None=1e-4,
             no_sim:bool=False,scaling:bool=True)->dict:
    if isinstance(beta,bool) or not isinstance(beta,(int,float)) or not math.isfinite(beta):
        raise ValueError("selected beta must be a finite real value")
    if betas is not None and any(isinstance(b,bool) or not isinstance(b,(int,float)) or not math.isfinite(b) for b in betas):
        raise ValueError("beta grid must contain finite real values")
    systems=(n_system,) if systems is None and no_sim else ((1,2,3) if systems is None else tuple(systems))
    betas=(beta,) if betas is None and no_sim else ((0.,.75,1.5) if betas is None else tuple(betas))
    systems=tuple(dict.fromkeys((n_system,*systems)))
    betas=tuple(dict.fromkeys((float(beta),*(float(b) for b in betas))))
    if len(systems)*len(betas)>MAX_CASES:
        raise ValueError("thermal artifact grid exceeds12 cases")
    specs=[(n,b,False) for n in systems for b in betas]
    if scaling:
        present={(n,b) for n,b,_ in specs}
        specs.extend((n,float(beta),True) for n in (1,2,3,4) if (n,float(beta)) not in present)
    if len(specs)>MAX_CASES:
        raise ValueError("thermal grid plus scaling exceeds12 cases")
    ids=[case_id(n,b) for n,b,_ in specs]
    if len(ids)!=len(set(ids)):
        raise ValueError("thermal parameter export identifiers collide")

    # Preflight the whole public bundle before replacing managed outputs.
    source_bytes={}
    for name in SOURCE_NAMES:
        path=(HERE/name if name.startswith("generate_") else SOURCE/"openquantum/simulation"/name) if REPOSITORY else HERE/name
        source_bytes[name]=path.read_bytes()
    license_path=ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    license_bytes=license_path.read_bytes()

    # Validate every source/check before replacing any managed output.
    records,cores,exports,rows,workloads={}, {}, {}, [], {}
    exported_operations=0
    for n,b,scaling_only in specs:
        terms=ts.spin_chain_terms(n,J,h)
        start=time.perf_counter()
        plan=_phase_plan(terms,b,filter_error,gibbs_error)
        core=ts.build_thermal_spectral_filter(terms,b,filter_error=filter_error,gibbs_error=gibbs_error,plan=plan)
        phase_source_time=time.perf_counter()-start
        if not no_sim and not scaling_only and core.num_qubits>ts.MAX_STATEVECTOR_QUBITS:
            raise ValueError("requested checked case exceeds simulator cap; use--no-sim")
        start=time.perf_counter()
        lowered=ts.lower_thermal_spectral(core)
        trials={axis:ts.lower_thermal_spectral(ts.build_thermal_spectral_trial(core,axis)) for axis in ("X","Z")}
        lowering_time=time.perf_counter()-start
        exported_operations+=lowered.size()+sum(c.size() for c in trials.values())
        if exported_operations>MAX_TOTAL_EXPORTED_OPERATIONS:
            raise ValueError("total exported circuit-operation cap exceeded")
        identifier=case_id(n,b)
        local_exports={}
        for kind,circuit in [("coherent",lowered),*((f"trial-{axis}",c) for axis,c in trials.items())]:
            filename=f"{identifier}-{kind}.qasm"
            raw=qasm3.dumps(circuit).encode()
            exports[filename]=raw
            local_exports[kind]={"file":filename,"sha256":digest(raw),
                                 "compiled_global_phase_radians":float(circuit.global_phase),"phase_policy":QASM_POLICY}
        performed=bool(not no_sim and not scaling_only)
        check,equivalence,check_seconds=None,None,None
        if performed:
            start=time.perf_counter()
            check=ts.thermal_spectral_check(core,operator_check=(core.num_qubits-n<=ts.MAX_OPERATOR_QUBITS))
            if core.num_qubits<=ts.MAX_OPERATOR_QUBITS:
                equivalence={"scope":"full coherent Operator with phase retained",
                    "max_entry_error":float(np.max(abs(Operator(core).data-Operator(lowered).data)))}
            else:
                equivalence={"scope":"coherent full-register Statevector on all-zero input only, with phase retained",
                    "max_entry_error":float(np.max(abs(Statevector.from_instruction(core).data-Statevector.from_instruction(lowered).data)))}
            numeric_gibbs_guard=core.metadata["error_contract"]["precompiler_source_Gibbs_trace_distance_upper"]+5e-10
            if (equivalence["max_entry_error"]>5e-12 or check["gibbs_trace_distance"]>numeric_gibbs_guard
                    or check["actual_Bell_branch_vector_error"]>core.metadata["error_contract"]["precompiler_source_encoded_total_error_upper"]+5e-10):
                raise AssertionError("bounded thermal spectral circuit/reference comparison failed")
            check_seconds=time.perf_counter()-start
            p=check["actual_acceptance_probability"]
            workload=ts.energy_workload(p,n,J,h)
            for axis,group in workload["groups"].items():
                count=group["accepted_trials"]*workload["retry_cap_per_slot"]
                group["complete_attempt_cap"]=count
                group["all_capped_primitive_counts"]={g:int(c)*count for g,c in trials[axis].count_ops().items()}
                group["logical_qubit_layer_volume_cap"]=trials[axis].num_qubits*trials[axis].depth()*count
            workloads[identifier]=workload
            rows.append({"case_id":identifier,"n_system":n,"beta":b,"parameter_a":plan.parameter,
                         "even_degree":plan.even_degree,"odd_degree":plan.odd_degree,
                         "ideal_decoded_filter_error_upper":core.metadata["error_contract"]["ideal_decoded_total_error_upper"],
                         "precompiler_source_decoded_filter_error_upper":core.metadata["error_contract"]["precompiler_source_decoded_total_error_upper"],
                         "precompiler_source_Gibbs_trace_distance_upper":core.metadata["error_contract"]["precompiler_source_Gibbs_trace_distance_upper"],
                         "ideal_Gibbs_trace_distance_upper":core.metadata["error_contract"]["ideal_Gibbs_trace_distance_upper"],
                         "actual_encoded_operator_error":check["actual_encoded_filter_operator_error"],
                         "actual_gibbs_trace_distance":check["gibbs_trace_distance"],
                         "actual_energy_error":check["energy_error"],"actual_acceptance":p,
                         "ideal_exact_filter_acceptance":check["exact_shifted_filter_acceptance_probability"],
                         "coherent_primitive_operations":lowered.size(),"Z_trial_primitive_operations":trials["Z"].size()})
        cores[identifier]=core
        records[identifier]={"case_id":identifier,"n_system":n,"beta":b,"J":J,"h":h,
            "terms":[[c,p] for c,p in terms],"logical_qubits":core.num_qubits,
            "parameter_a":plan.parameter,"phase_plan":plan.as_dict(),"error_contract":core.metadata["error_contract"],
            "validation_performed":performed,"validation":check,"source_primitive_check":equivalence,
            "generation_only_scaling_case":scaling_only,"coherent":_primitive_record(lowered),
            "measurement_groups":{axis:_primitive_record(c) for axis,c in trials.items()},
            "ordinary_access_calls":core.metadata["ordinary_access_calls"],
            "controlled_access_calls":core.metadata["controlled_access_calls"],
            "PREPARE_calls":plan.odd_degree,"PREPARE_inverse_calls":plan.odd_degree,"SELECT_calls":plan.odd_degree,
            "Bell_H_gates":n,"Bell_CX_gates":n,"success_measurements_per_trial":core.num_qubits-2*n,
            "initial_resets_per_trial":core.num_qubits,"reference_resets_per_trial":n,"system_measurements_per_trial":n,
            "exports":local_exports,"accepted_energy_workload":workloads.get(identifier),
            "classical_CPU_seconds":{"phase_generation_certificate_source":phase_source_time,"lowering":lowering_time,"bounded_checks":check_seconds},
            "synthesis_performed":False,"physical_estimate":None,"sampling_performed":False}

    selected_id=case_id(n_system,float(beta))
    selected=records[selected_id]
    scaling_rows=[]
    if scaling:
        for n in (1,2,3,4):
            rec=records[case_id(n,float(beta))]
            scaling_rows.append({"n_system":n,"beta":float(beta),"alpha_float":rec["error_contract"]["alpha_float_for_display"],
                "parameter_a":rec["parameter_a"],"even_degree":rec["phase_plan"]["even_degree"],
                "odd_degree":rec["phase_plan"]["odd_degree"],"logical_qubits":rec["logical_qubits"],
                "coherent_primitives":rec["coherent"]["primitive_operations"],
                "Z_trial_primitives":rec["measurement_groups"]["Z"]["primitive_operations"],
                "Z_trial_depth":rec["measurement_groups"]["Z"]["dependency_depth"],
                "continuous_rotations_Z_trial":rec["measurement_groups"]["Z"]["continuous_rotations"],
                "resets_Z_trial":rec["measurement_groups"]["Z"]["primitive_gate_counts"].get("reset",0),
                "measurements_Z_trial":rec["measurement_groups"]["Z"]["primitive_gate_counts"].get("measure",0),
                "state_validation_performed":rec["validation_performed"],
                "ideal_Gibbs_trace_distance_upper":rec["error_contract"]["ideal_Gibbs_trace_distance_upper"],
                "precompiler_source_Gibbs_trace_distance_upper":rec["error_contract"]["precompiler_source_Gibbs_trace_distance_upper"]})
    output.mkdir(parents=True,exist_ok=True)
    for path in output.glob("n*-beta-*-*.qasm"):
        if path.name not in exports:path.unlink()
    for name,raw in exports.items():(output/name).write_bytes(raw)
    hashes={}
    for name,raw in source_bytes.items():
        (output/name).write_bytes(raw);hashes[name]=digest(raw)
    (output/"LICENSE.txt").write_bytes(license_bytes)
    hashes["LICENSE.txt"]=digest(license_bytes)
    owned_paths=[f"src/openquantum/simulation/{name}" for name in SOURCE_NAMES if not name.startswith("generate_")]
    owned_paths.append("scripts/generate_simulation_thermal_spectral.py")
    provenance={"source_hashes":hashes,"source_revision_at_generation":git_revision(ROOT) if REPOSITORY else None,
        "source_dirty_at_generation":git_dirty(ROOT,owned_paths) if REPOSITORY else None,
        "revision_scope":"source hashes identify exact working-tree bytes when generation precedes commit",
        "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","qiskit","matplotlib")},
        "device":"CPU","randomness":"deterministic phase fitting; no shot experiment","transpiler_seed":7,"optimization_level":1,"qubits_initially_zero":False}
    report={"status":"bounded thermal spectral checks and actual materialization" if not no_sim else "materialized retargeted thermal circuits; state/readout validation unperformed",
        "selected_case_id":selected_id,"selected_case":selected,"cases":list(records.values()),
        "validation_case_count":len(rows),"materialized_case_count":len(records),"validation_performed":not no_sim,
        "scaling_rows":scaling_rows,"scaling_model":{"fixed_J":J,"fixed_h":h,"fixed_beta":beta,"freshly_retargeted_parameter":"beta*alpha_exact/2; no fixed4.1 phase table","state_validation":"main grid n<=3 only; n4 scaling is materialization only"},
        "sampling_performed":False,"physical_estimate":None,"qasm_export_policy":QASM_POLICY,"provenance":provenance}
    ledger={"resource_kind":"actual complete thermal measured trials before synthesis/physical scheduling",
        "selected_case_id":selected_id,"cases":list(records.values()),"source_hashes":hashes,"provenance":provenance,
        "prominent_exclusions":["compiler/control transformation and fault-tolerant rotation-synthesis certificates","reset/readout/routing/factory time and physical faults","actual finite-shot energy estimation","general thermal mixing or advantage claims","formal/Lean proof"],
        "instrument_contract":"initial all-wire resets, coherent Bell+QSP, all success bits measured, reference reset without conditioning, X/Z basis change and system readout; every rejection runs fully",
        "precision_contract":"analytic/interval coefficient, parameter and declared-logical PREPARE weight bounds cover exact named Cliffords/controls with stored angles; SDK nested-control/MCX realization and flat compiler checks are numerical, with synthesis/instrument/physical certificates separate",
        "caps":{"systems":ts.MAX_SYSTEM_QUBITS,"terms":ts.MAX_TERMS,"total_wires":ts.MAX_TOTAL_QUBITS,"phase_degree":ts.MAX_PHASE_DEGREE,
                "parameter_a":ts.MAX_PARAMETER,"source_operations":ts.MAX_SOURCE_OPERATIONS,"primitive_operations":ts.MAX_PRIMITIVE_OPERATIONS,
                "statevector_total_wires":ts.MAX_STATEVECTOR_QUBITS,"filter_operator_total_wires":ts.MAX_OPERATOR_QUBITS,
                "reference_systems":ts.MAX_REFERENCE_SYSTEM_QUBITS,"artifact_cases":MAX_CASES,"all_exported_operations":MAX_TOTAL_EXPORTED_OPERATIONS},
        "physical_estimate":None,"sampling_performed":False,"qasm_export_policy":QASM_POLICY}
    write_json(output/"results.json",report);write_json(output/"logical-resources.json",ledger)
    write_csv(output/"accuracy.csv",rows);write_csv(output/"scaling.csv",scaling_rows)

    # A literal small circuit demo; the selected analysis instance may have more wires.
    demo_id=case_id(1,.75)
    demo=cores.get(demo_id)
    if demo is not None:
        fig=demo.draw(output="mpl",fold=26,scale=.75)
        fig.savefig(output/"circuit.svg",bbox_inches="tight",metadata={"Date":None});plt.close(fig)
    else:(output/"circuit.svg").unlink(missing_ok=True)
    plan=selected["phase_plan"]
    x=np.linspace(-1,1,1001)
    actual=(real_qsp_response(x,plan["even_phases"])+real_qsp_response(x,plan["odd_phases"]))/2
    target=np.exp(-selected["parameter_a"]*(x+1))/4
    fig,axes=plt.subplots(2,1,figsize=(7,4.8),sharex=True)
    axes[0].plot(x,target,color="#aab3b8",lw=4,label="shifted exponential /4")
    axes[0].plot(x,actual,color="#246b75",label="generated real parity sum")
    axes[0].set(ylabel="Accepted scalar response",title=f"Thermal QSP: n={n_system}, beta={beta:g}, a={selected['parameter_a']:.5g}")
    axes[0].legend()
    axes[1].plot(x,actual-target,color="#9a603e")
    axes[1].set(xlabel="Normalized signal x",ylabel="QSP minus target")
    axes[1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
    fig.tight_layout();fig.savefig(output/"response.svg",metadata={"Date":None});plt.close(fig)
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(9,3.5))
        for n in systems:
            group=sorted((r for r in rows if r["n_system"]==n),key=lambda r:r["beta"])
            axes[0].semilogy([r["beta"] for r in group],[max(r["actual_gibbs_trace_distance"],1e-17) for r in group],"o-",label=f"n={n}")
            axes[1].plot([r["beta"] for r in group],[r["actual_acceptance"] for r in group],"o-",label=f"n={n}")
        axes[0].set(xlabel="Inverse temperature beta",ylabel="Actual Gibbs trace distance")
        axes[1].set(xlabel="Inverse temperature beta",ylabel="Actual raw acceptance")
        axes[0].legend();fig.tight_layout();fig.savefig(output/"accuracy.svg",metadata={"Date":None});plt.close(fig)
    else:(output/"accuracy.svg").unlink(missing_ok=True)
    if scaling_rows:
        fig,axes=plt.subplots(1,2,figsize=(8.5,3.5))
        axes[0].plot([r["n_system"] for r in scaling_rows],[r["odd_degree"] for r in scaling_rows],"o-",color="#246b75")
        axes[0].set(xlabel="System spins n",ylabel="Retargeted odd degree")
        axes[1].plot([r["n_system"] for r in scaling_rows],[r["Z_trial_primitives"] for r in scaling_rows],"o-",color="#9a603e")
        axes[1].set(xlabel="System spins n",ylabel="Complete Z-trial operations")
        fig.suptitle(f"Fixed J={J:g}, h={h:g}, beta={beta:g}; phases retargeted with size")
        fig.tight_layout();fig.savefig(output/"scaling.svg",metadata={"Date":None});plt.close(fig)
    else:(output/"scaling.svg").unlink(missing_ok=True)
    for name in MANAGED_PLOTS:
        path=output/name
        if path.exists():path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    return {"output_dir":str(output),"selected_case_id":selected_id,"selected_validation":selected["validation"],
            "validation_case_count":len(rows),"materialized_case_count":len(records),"scaling_rows":scaling_rows,"sampling_performed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-system",type=int,default=2);parser.add_argument("--beta",type=float,default=1.5)
    parser.add_argument("--J",type=float,default=.6);parser.add_argument("--h",type=float,default=.35)
    parser.add_argument("--systems");parser.add_argument("--betas")
    parser.add_argument("--filter-error",type=float,default=1e-6);parser.add_argument("--gibbs-error",type=float,default=1e-4)
    parser.add_argument("--filter-only",action="store_true",help="omit the ideal-access Gibbs-error target")
    parser.add_argument("--no-sim",action="store_true");parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/simulation/thermal-spectral" if REPOSITORY else HERE/"generated-thermal-spectral")
    args=parser.parse_args()
    result=generate(args.output_dir,args.n_system,args.beta,args.J,args.h,
        None if args.systems is None else tuple(int(n) for n in args.systems.split(",")),
        None if args.betas is None else tuple(float(b) for b in args.betas.split(",")),
        args.filter_error,None if args.filter_only else args.gibbs_error,args.no_sim,not args.no_scaling)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":main()
