#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate structured HHL circuit, numerical checks, scaling and conditional cost."""
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import qasm3
from openquantum.linear_systems import (
    GridXSystem, ROTATION_BASIS, accepted_observable_workload, analytic_success_probability,
    build_hhl, compare_hhl, decompose, measured_hhl_trial,
)


def ledger_for(system, *, angles=None):
    unitary = build_hhl(system,angles=angles)
    measured = decompose(measured_hhl_trial(system,angles=angles))
    counts = dict(measured.count_ops())
    success = analytic_success_probability(system,angles=angles)
    workload = accepted_observable_workload(success)
    return unitary, measured, {
        "method": "HHL on explicit dyadic commuting X-projector family",
        "system_qubits": system.n_system, "matrix_dimension": 1 << system.n_system,
        "phase_qubits": system.phase_qubits, "logical_qubits": measured.num_qubits,
        "matrix_definition": "A=offset/D I+sum_j weights[j]/D (I-X_j)/2",
        "weights_little_endian": list(system.weights), "offset": system.offset, "D": system.denominator,
        "lambda_min": system.lambda_min, "lambda_max": system.lambda_max, "kappa": system.condition_number,
        "phase_spacing": 1/system.denominator, "phase_approximation": "none for this exact-grid family",
        "right_hand_side": unitary.metadata["input_preparation"],
        "C": system.lambda_min, "phase_lookup_labels": list(system.phase_labels),
        "source_gate_counts": dict(unitary.count_ops()), "source_dependency_depth": unitary.depth(),
        "primitive_gate_counts_per_complete_attempt": counts, "primitive_dependency_depth": measured.depth(),
        "primitive_basis": ROTATION_BASIS, "seed_transpiler": 7, "optimization_level": 1,
        "arbitrary_rotations_per_attempt": sum(counts.get(name,0) for name in ("rx","ry","rz")),
        "output": "one conditional Z(q0) mean, after clock=0 and flag=1; not every classical solution entry",
        "analytic_family_success_probability": success,
        "success_lower_bound_from_kappa": 1/system.condition_number**2,
        "retry_and_sampling_proposal": workload,
        "capped_primitive_gate_counts_proposal": {name:count*workload["total_capped_attempts"] for name,count in counts.items()},
        "initialization": "each attempt assumes all logical inputs initialized to clean zero; reset timing/faults not modeled",
        "included_in_attempt": ["product RY input preparation","controlled P/RX evolution including identity phase",
            "forward QPE","all reachable reciprocal lookup rotations","inverse QPE","all final measurements"],
        "excluded": ["clean-state initialization timing/faults","classical parameter loading and lookup construction time",
            "classical compilation/reference solve/readout postprocessing time","state amplification",
            "concrete coherent Clifford+T synthesis certificate"],
        "algorithm_family_limitations": ["known commuting terms permit efficient classical diagonalization and fast-forwarding",
            "reciprocal lookup can grow exponentially with clock precision",
            "no general sparse-matrix oracle or arbitrary vector loader is implemented",
            "no quantum advantage is established"],
        "synthesis_conditioning_note": "QDK mean-fit scalar is not an operator-norm certificate and cannot certify postselected observable bias/acceptance",
        "hypothetical_certified_unitary_error_conditioning": {
            "assumption": "complete coherent pre-readout unitary operator norm error <= eta, same prepared input",
            "acceptance_lower": "(sqrt(p)-eta)**2 when eta<sqrt(p)",
            "conditional_state_trace_distance_upper": "min(1,2*eta/sqrt(p))",
            "conditional_Pauli_bias_upper": "min(2,4*eta/sqrt(p))",
            "actual_eta_certificate_available": False},
        "physical_estimate": None,
    }


def write_scaling(output):
    rows=[]
    for n in (1,2,3,4,6,8,12,24):
        m=math.ceil(math.log2(n+2))
        system=GridXSystem((1,)*n,1,m)
        _, measured, ledger=ledger_for(system)
        validation={"performed":False,"reason":"scaling is a generated circuit/analytic family prediction"}
        if n<=6:
            validation=compare_hhl(system)
        counts=ledger["primitive_gate_counts_per_complete_attempt"]
        rows.append({"system_qubits":n,"dimension":1<<n,"clock_qubits":m,"logical_qubits":measured.num_qubits,
            "kappa":system.condition_number,"lookup_rotations":len(system.phase_labels),
            "primitive_operations":len(measured.data)-counts.get("barrier",0),"dependency_depth":measured.depth(),
            "cx":counts.get("cx",0),"continuous_rotations":ledger["arbitrary_rotations_per_attempt"],
            "final_measurements":counts.get("measure",0),"analytic_success_probability":ledger["analytic_family_success_probability"],
            "accepted_samples_proposal":ledger["retry_and_sampling_proposal"]["accepted_samples"],
            "attempts_per_slot_proposal":ledger["retry_and_sampling_proposal"]["attempt_cap_per_accepted_slot"],
            "total_attempts_proposal":ledger["retry_and_sampling_proposal"]["total_capped_attempts"],
            "simulation_performed":validation["performed"],
            "state_trace_distance":validation.get("trace_distance_to_normalized_classical_solution")})
    with (output/"scaling.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(rows)
    fig, axes=plt.subplots(1,2,figsize=(9,3.4))
    for key,label in (("primitive_operations","Primitive operations"),("dependency_depth","Dependency depth")):
        axes[0].plot([r["system_qubits"] for r in rows],[r[key] for r in rows],"o-",label=label)
    axes[0].set(xlabel="Data qubits n",ylabel="Generated count per complete attempt",yscale="log")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].plot([r["system_qubits"] for r in rows],[r["total_attempts_proposal"] for r in rows],"o-",color="#946344")
    axes[1].set(xlabel="Data qubits n",ylabel="Proposed capped attempts for one Z mean",yscale="log")
    axes[1].grid(alpha=.2);fig.tight_layout();fig.savefig(output/"scaling.svg");plt.close(fig)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits",type=int,default=1)
    parser.add_argument("--phase-qubits",type=int)
    parser.add_argument("--weights",type=int,nargs="+")
    parser.add_argument("--offset",type=int,default=1)
    parser.add_argument("--angles",type=float,nargs="+")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--no-estimate",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/linear-systems/hhl")
    args=parser.parse_args()
    if not 1 <= args.qubits <= 24:
        parser.error("--qubits must be between1 and24; generation is explicitly bounded")
    if args.weights is not None and not 1 <= len(args.weights) <= 24:
        parser.error("Supply1 to24 weights")
    if args.offset < 1 or (args.weights is not None and min(args.weights) < 0):
        parser.error("offset must be positive and weights nonnegative")
    weights=tuple(args.weights) if args.weights is not None else (1,)*args.qubits
    m=args.phase_qubits if args.phase_qubits is not None else math.ceil(math.log2(args.offset+sum(weights)+1))
    system=GridXSystem(weights,args.offset,m)
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    unitary, measured, ledger=ledger_for(system,angles=args.angles)
    qasm=qasm3.dumps(measured);(output/"measured-trial.qasm").write_text(qasm)
    ledger["qasm_export_policy"]={"scope":"complete measured trials; not coherently controlled oracle specification",
        "compiled_global_phase_radians":float(measured.global_phase),
        "pinned_exporter_global_phase":"may omit global phase; measurement probabilities unaffected"}
    source=(ROOT/"src/openquantum/linear_systems.py").read_bytes()
    (output/"linear_systems.py").write_bytes(source);(output/"LICENSE.txt").write_bytes((ROOT/"LICENSE").read_bytes())
    ledger["provenance"]={
        "source_revision_at_generation":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "implementation_sha256":hashlib.sha256(source).hexdigest(),"qasm_sha256":hashlib.sha256(qasm.encode()).hexdigest(),
        "source_note":"commit plus canonical source hash identify working tree code; generation can precede commit",
        "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","matplotlib")},
        "seed":7,"device":"CPU","command":"python scripts/generate_linear_systems.py "+" ".join(sys.argv[1:])}
    ledger["validation"]={"performed":False,"reason":"explicit --no-sim or simulator/reference cap"}
    if not args.no_sim and unitary.num_qubits<=12 and system.n_system<=8:
        ledger["validation"]=compare_hhl(system,angles=args.angles)
    if not args.no_scaling:
        ledger["scaling_family"]={"definition":"offset1, all weights1, m=ceil(log2(n+2)), b=|0**n>",
            "rows":write_scaling(output),"physical_scaling_computed":False,
            "scope":"actual generated primitive counts and analytic ideal retry proposal; only identified small rows simulated"}
    if unitary.num_qubits<=6:
        fig=unitary.draw(output="mpl",fold=22,scale=.7);fig.savefig(output/"circuit.svg",bbox_inches="tight");plt.close(fig)
    if not args.no_estimate:
        from openquantum.resource_estimation import CircuitResourceBudget, estimate_circuit_resources
        if measured.num_qubits>8 or len(measured.data)>3000:
            ledger["physical_estimate"]={"status":"unavailable; physical queries capped at8 total wires and3000 source operations"}
        else:
            budget=CircuitResourceBudget(1e-6,.005,ledger["retry_and_sampling_proposal"]["total_capped_attempts"])
            report=estimate_circuit_resources(measured_hhl_trial(system,angles=args.angles),budget=budget,
                cache_dir=Path("/tmp/openquantum-linear-systems-qdk-cache"))
            report.pop("qasm3")
            report["postselection_contract"]={"success_predicate":"clock0 and flag1","resource_executions":"all capped complete attempts",
                "confidence_status":"conditional cost projection; no certified postselected mean or acceptance guarantee",
                "initialization":"clean initial logical qubits assumed; reset timing/faults excluded"}
            (output/"surface-code.json").write_text(json.dumps(report,indent=2)+"\n")
            ledger["physical_estimate"]={"status":report["resource_estimate_kind"],"report":"surface-code.json",
                "pareto_points":len(report["pareto_points"]),"profile":report["physical_profile"],"budget":report["budget"]}
    (output/"logical-resources.json").write_text(json.dumps(ledger,indent=2)+"\n")
    for path in output.glob("*.svg"):
        path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    print(json.dumps({"output_dir":str(output),"validation":ledger["validation"],"physical_estimate":ledger["physical_estimate"]},indent=2))


if __name__=="__main__":main()
