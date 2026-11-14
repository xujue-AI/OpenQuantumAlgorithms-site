#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Package bounded known-input amplification of a Bell-plus-thermal-filter circuit.

Default checks cover H=.6Z+.4X, Taylor degree6, beta1.25/2 and rounds0..4.
Use --no-sim for circuit materialization only; acceptance/state/workload evidence
is then unavailable and stale validation CSV/figures are removed. This does not
run a sampling experiment, phase solver, physical estimator or remote job.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT/"src"
REPOSITORY = (SOURCE/"openquantum/simulation/preparation_amplification.py").is_file()
sys.path.insert(0, str(SOURCE if REPOSITORY else HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator

if REPOSITORY:
    from openquantum.simulation import preparation_amplification as pa, thermal_filter as tf
else:
    import preparation_amplification as pa
    import thermal_filter as tf

EXPORT_POLICY = (
    "Qiskit OpenQASM3 export omits an overall compiled circuit phase. "
    "The measured-trial files preserve outcome probabilities and are not "
    "phase-preserving coherently controlled oracles. In-memory circuits retain "
    "phase for the signed branch and whole-unitary equivalence checks. "
    "Coherent reuse requires the source modules or explicit phase restoration."
)
VALIDATION_FILES = ("rounds.csv", "success-and-bias.svg")
SCALING_FILES = ("scaling.csv", "reflection-scaling.svg")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def dump(path: Path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+"\n")


def sha(raw: bytes):
    return hashlib.sha256(raw).hexdigest()


def normalize_svg(path: Path):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")


def beta_slug(beta: float) -> str:
    return format(beta, ".17g").replace("-", "m").replace(".", "p").replace("+", "")


def case_id(beta: float, rounds: int) -> str:
    return f"beta-{beta_slug(beta)}-round-{rounds}"


def validate_check(check):
    if abs(check["actual_acceptance_probability"]-check["model_success_probability_from_actual_baseline_p"]) > 1e-10:
        raise AssertionError("actual acceptance disagrees with known-input rotation identity")
    if check["raw_signed_accepted_vector_error"] > 1e-10:
        raise AssertionError("raw accepted purification changed beyond its signed scalar")
    if check["conditional_state_defined"]:
        for key in ("phase_invariant_purification_error", "signed_normalized_purification_error",
                    "reduced_state_trace_distance_from_baseline", "reduced_state_trace_distance_from_polynomial"):
            if check[key] > 1e-10:
                raise AssertionError(f"accepted purification/state failed its reference check: {key}")


def write_csv(path, rows):
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def circuit_diagram(source, rounds):
    """Defined circuit boxes; the resource/export circuits are fully expanded."""
    n_total = source.num_qubits
    labels = list(range(source.metadata["n_system"]+source.metadata["n_reference"], n_total))
    diagram = QuantumCircuit(n_total)
    source_gate = source.to_gate(label="A: Bell + filter")
    diagram.append(source_gate, range(n_total))
    if rounds:
        initial = pa.zero_subregister_reflection(n_total, range(n_total)).to_gate(label="S0: all zero")
        if labels:
            good = pa.zero_subregister_reflection(len(labels), range(len(labels))).to_gate(label="Sf: labels zero")
        inverse = source.inverse().to_gate(label="A dagger")
        for _ in range(rounds):
            if labels:
                diagram.append(good, labels)
            else:
                diagram.global_phase += math.pi
            diagram.append(inverse, range(n_total))
            diagram.append(initial, range(n_total))
            diagram.append(source_gate, range(n_total))
            diagram.global_phase += math.pi
    return diagram


def generate(output: Path, z=.6, x=.4, beta=1.25, degree=6,
             rounds=(0,1,2,3,4), betas=(1.25,2), simulate=True, scaling=True):
    terms = ((z,"Z"),(x,"X"))
    rounds = tuple(rounds)
    betas = tuple(dict.fromkeys((beta, *betas)))
    if not rounds or not betas or len(rounds)*len(betas) > 20:
        raise ValueError("need1..20 bounded beta/round cases")
    if any(isinstance(r,bool) or not isinstance(r,int) or not 0 <= r <= pa.MAX_ROUNDS for r in rounds):
        raise ValueError("round grid must contain integers0..4")
    rounds = tuple(sorted(set(rounds)))
    # Validate all coefficients/degrees/domains before replacing managed outputs.
    for b in betas:
        tf.thermal_pauli_terms(terms,b,degree)
    selected_round = 1 if 1 in rounds else rounds[0]
    selected_id = case_id(beta,selected_round)
    output.mkdir(parents=True,exist_ok=True)
    for path in output.glob("beta-*-round-*.qasm"):
        path.unlink()
    if not simulate:
        for name in VALIDATION_FILES:
            (output/name).unlink(missing_ok=True)
    if not scaling:
        for name in SCALING_FILES:
            (output/name).unlink(missing_ok=True)
    cases, ledgers, rows = [], [], []
    source_seconds, materialize_seconds = 0.0, 0.0
    for b in betas:
        started=time.perf_counter()
        source=tf.build_thermal_filter(terms,b,degree)
        source_seconds += time.perf_counter()-started
        labels=list(range(source.metadata["n_system"]+source.metadata["n_reference"],source.num_qubits))
        for r in rounds:
            identifier=case_id(b,r)
            started=time.perf_counter()
            core=pa.amplify_known_input(source,labels,r)
            lowered=pa.decompose_preparation_amplification(core)
            trials={axis:pa.decompose_preparation_amplification(pa.build_preparation_trial(terms,b,degree,r,axis)) for axis in ("X","Z")}
            materialize_seconds += time.perf_counter()-started
            exported={"coherent":lowered, **{f"trial-{axis}":c for axis,c in trials.items()}}
            exports={}
            for role,circuit in exported.items():
                name=f"{identifier}-{role}.qasm"
                text=qasm3.dumps(circuit)
                (output/name).write_text(text)
                exports[role]={"file":name,"sha256":sha(text.encode()),
                               "compiled_global_phase_radians":float(circuit.global_phase),"phase_policy":EXPORT_POLICY}
            check=None
            equivalence=None
            workload=None
            synthesis=None
            synthesis_workload=None
            if simulate:
                check=pa.preparation_amplification_check(terms,b,degree,r)
                validate_check(check)
                if core.num_qubits > pa.MAX_OPERATOR_QUBITS:
                    raise ValueError("artifact whole-unitary check exceeds six wires")
                equivalence=float(np.max(np.abs(Operator(core).data-Operator(lowered).data)))
                if equivalence > 1e-10:
                    raise AssertionError("coherent source/lowered circuit changed full-unitary phase or behavior")
                if check["conditional_state_defined"]:
                    workload=tf.thermal_energy_workload(check["actual_acceptance_probability"],terms=terms)
                    synthesis=pa.synthesis_conditioning_proposal(check["actual_acceptance_probability"],r)
                    synthesis_workload=tf.thermal_energy_workload(synthesis["perturbed_acceptance_lower_bound"],terms=terms)
            core_counts=dict(lowered.count_ops())
            per_axis={}
            for axis,circuit in trials.items():
                counts=dict(circuit.count_ops())
                item={"primitive_gate_counts_per_complete_attempt":counts,"total_instructions_per_attempt":sum(counts.values()),
                      "dependency_depth_per_attempt":circuit.depth(),"logical_qubits":circuit.num_qubits,"classical_bits":circuit.num_clbits,
                      "initial_reset_instructions":core.num_qubits,"unconditioned_reference_reset_instructions":source.metadata["n_reference"],
                      "reset_instructions_per_attempt":counts.get("reset",0),"measurement_instructions_per_attempt":counts.get("measure",0),
                      "continuous_rotation_gate_count":sum(counts.get(g,0) for g in ("rx","ry","rz")),
                      "rotation_count_scope":"RX/RY/RZ class count; no continuous-angle synthesis certificate",
                      "readout_schedule":"fresh all-wire reset; amplified Bell+LCU; labels measured; reference reset without conditioning; basis change and system readout; all failed attempts charged in full"}
                for key,proposal in (("ideal_numeric_workload",workload),("hypothetical_synthesis_workload",synthesis_workload)):
                    if proposal is not None:
                        group=next((g for g in proposal["groups"] if g["axis"]==axis),None)
                        if group:
                            item[key]={"accepted_shots":group["accepted_shots"],
                                       "complete_attempt_cap_per_accepted_shot":group["complete_trial_cap_per_accepted_shot"],
                                       "worst_case_complete_attempts":group["worst_case_complete_trials"],
                                       "expected_complete_attempts":group["expected_complete_trials"],
                                       "expected_complete_attempts_uncapped":group["expected_complete_trials_uncapped"],
                                       "expected_complete_attempts_stopped_at_cap":group["expected_complete_trials_stopped_at_cap"],
                                       "expected_complete_attempts_scope":"legacy alias for uncapped geometric expectation M/p",
                                       "stopped_expectation_scope":proposal["stopped_expectation_scope"],
                                       "worst_case_instruction_counts":{gate:count*group["worst_case_complete_trials"] for gate,count in counts.items()}}
                per_axis[axis]=item
            access_calls={"Bell_forward_H":source.metadata["n_system"]*(r+1),"Bell_forward_CX":source.metadata["n_system"]*(r+1),
                          "Bell_inverse_H":source.metadata["n_system"]*r,"Bell_inverse_CX":source.metadata["n_system"]*r,
                          "PREPARE_or_inverse":2*r+1,"SELECT_or_inverse":2*r+1,"unprepare_or_inverse":2*r+1,
                          "source_A_forward":r+1,"source_A_inverse":r,"full_initial_reflection":r,"labels_only_good_reflection":r}
            ledger={"case_id":identifier,"beta":b,"degree":degree,"rounds":r,"n_system":source.metadata["n_system"],
                    "n_reference":source.metadata["n_reference"],"n_label":source.metadata["n_index"],"total_qubits":core.num_qubits,
                    "coherent_top_level_counts":dict(core.count_ops()),"coherent_primitive_counts":core_counts,
                    "coherent_primitive_gate_count":sum(core_counts.values()),"coherent_dependency_depth":lowered.depth(),
                    "access_and_preparation_calls_per_attempt":access_calls,"per_measurement_axis":per_axis,"exports":exports,
                    "accepted_shot_workload_proposal":workload,"hypothetical_synthesis_proposal":synthesis,
                    "hypothetical_synthesis_workload_proposal":synthesis_workload,
                    "acceptance_projection_certified":False,"sampling_performed":False,"physical_resource_estimate":None}
            ledgers.append(ledger)
            case={"case_id":identifier,"beta":b,"degree":degree,"rounds":r,"validation":check,
                  "whole_unitary_source_primitive_max_entry_error":equivalence,"validation_performed":simulate,
                  "coherent_primitive_gate_count":sum(core_counts.values()),"coherent_dependency_depth":lowered.depth(),"exports":exports}
            cases.append(case)
            if check is not None:
                rows.append({"case_id":identifier,"beta":b,"degree":degree,"rounds":r,"baseline_acceptance":check["baseline_acceptance_probability"],
                             "acceptance":check["actual_acceptance_probability"],"model_acceptance":check["model_success_probability_from_actual_baseline_p"],
                             "global_sign":check["expected_global_sign"],"signed_factor":check["signed_amplification_factor"],
                             "raw_signed_vector_error":check["raw_signed_accepted_vector_error"],
                             "phase_invariant_purification_error":check["phase_invariant_purification_error"],
                             "state_trace_distance_from_baseline":check["reduced_state_trace_distance_from_baseline"],
                             "gibbs_trace_distance":check["gibbs_trace_distance"],"energy_error":check["energy_error"],
                             "source_A_or_inverse_calls":2*r+1,"coherent_primitive_gates":sum(core_counts.values()),
                             "expected_source_calls_per_accepted_preparation":(2*r+1)/check["actual_acceptance_probability"] if check["actual_acceptance_probability"] else None,
                             "expected_source_calls_per_accepted_preparation_uncapped":(2*r+1)/check["actual_acceptance_probability"] if check["actual_acceptance_probability"] else None,
                             "expected_source_calls_per_accepted_preparation_scope":"legacy alias for uncapped geometric expectation (2r+1)/p; reflection and reset/readout operations excluded",
                             "expected_source_calls_per_measurement_slot_stopped_at_cap":(2*r+1)*workload["expected_complete_trials_stopped_at_cap"]/workload["total_accepted_shots"] if workload is not None and workload["total_accepted_shots"] else None,
                             "stopped_source_call_expectation_scope":"each requested measurement slot stops on acceptance or cap, including exhaustion; all slots attempted without early abort; reflection and reset/readout operations excluded"})
    scaling_rows=[pa.reflection_scaling_row(n,beta,degree) for n in (1,2,3)] if scaling else []
    contrast=pa.ancilla_only_distortion_check(terms,beta,degree,selected_round) if simulate else None
    source_names=("preparation_amplification.py","thermal_filter.py","lcu.py","generate_simulation_preparation_amplification.py")
    hashes={}
    for name in source_names:
        path=(HERE/name if name.startswith("generate_") else SOURCE/"openquantum/simulation"/name) if REPOSITORY else HERE/name
        raw=path.read_bytes()
        (output/name).write_bytes(raw)
        hashes[name]=sha(raw)
    license_path=ROOT/"LICENSE" if REPOSITORY else HERE/"LICENSE.txt"
    (output/"LICENSE.txt").write_bytes(license_path.read_bytes())
    default=next(c for c in cases if c["case_id"]==selected_id)
    report={"status":"checked known-input thermal preparation amplification" if simulate else "circuits generated; validation, acceptance and workload proposals unavailable",
            "validation_performed":simulate,"default_case_id":selected_id,"default_validation":default["validation"],
            "instance":{"Hamiltonian":"H/E0=z Z+x X","E0":1,"k_B":1,"z":z,"x":x,"beta":beta,"degree":degree,
                        "known_input":"all wires zero; Bell purification prepared by actual H/CX","selected_default_round":selected_round},
            "cases":cases,"ancilla_only_contrast":contrast,"reflection_scaling_rows":scaling_rows,"scaling_generation_performed":scaling,
            "scaling_family_model":{"Hamiltonian":"sum_i(.6 Zi+.4 Xi)","beta":beta,"degree":degree,"state_validation_performed":False,"note":"fixed family coefficients; CLI z/x controls the one-qubit validation model"},
            "direction_scope":"signed raw branch checked before phase-invariant purification/reduced-state comparisons; negative accepted factors change only global sign",
            "round_selection_scope":"fixed integer rounds selected from tiny classical checks; no general probability-learning/adaptive protocol",
            "classical_preprocessing":{"access_construction_wall_seconds":source_seconds,"full_circuit_materialization_lowering_wall_seconds":materialize_seconds,
                                       "aggregation":"bounded algebraic Pauli aggregation may be exponential; no eigenvector or matrix gate loading"},
            "sampling_performed":False,"physical_resource_estimate":None,"qasm_export_policy":EXPORT_POLICY,
            "provenance":{"source_hashes":hashes,"source_revision_at_generation":git_revision(ROOT) if REPOSITORY else None,
                          "revision_note":"hashes identify working-tree source bytes if generation precedes commit",
                          "python":platform.python_version(),"packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","scipy","matplotlib")},
                          "device":"CPU","randomness":"none; exact ideal checks","transpiler_seed":7,"optimization_level":1,"qubits_initially_zero":False}}
    master={"resource_kind":"actual complete-attempt logical instruction counts before coherent rotation synthesis and error correction",
            "physical_resource_estimate":None,"cases":ledgers,"source_hashes":hashes,"default_case_id":selected_id,
            "error_contract":{"thermal_polynomial":"Gibbs bias checked independently; amplification preserves its accepted density and does not improve degree error",
                              "coherent_synthesis":"hypothetical whole amplified-unitary allowance only; no certificates; includes all A/inverse/reflection calls",
                              "statistics":"separate accepted X and Z means; no single-bitstring noncommuting energy sample",
                              "acceptance":"numeric known-input probability with repeatable all-zero preparation; no physical completion guarantee",
                              "physical_faults":"separate proposed allowance; gate/reset/readout/factory timing and physical estimation unavailable"},
            "prominent_exclusions":["certified continuous rotation synthesis","physical faults and reset/readout/routing times","executed accepted-shot sampling","probability-learning or fixed-point amplification","scalable thermal access and formal proof"],
            "scope":{"default_total_wires":4,"thermal_system_width":4,"thermal_input_terms":32,"thermal_aggregated_terms":256,"thermal_degree":12,
                     "beta_sum_abs_coefficients_over_two":10,"amplification_rounds":4,"source_total_wires":24,"source_top_level_instructions":50000,
                     "primitive_export_operations":250000,"statevector_wires":12,"whole_operator_check_wires":6,"artifact_validation_cases":20},
            "scaling_rows_scope":"actual access/reflection generation only; family-row state simulation unperformed",
            "qasm_export_policy":EXPORT_POLICY,"provenance":report["provenance"]}
    dump(output/"results.json",report)
    dump(output/"logical-resources.json",master)
    write_csv(output/"rounds.csv",rows)
    write_csv(output/"scaling.csv",scaling_rows)
    selected_source=tf.build_thermal_filter(terms,beta,degree)
    diagram=circuit_diagram(selected_source,selected_round)
    fig=diagram.draw(output="mpl",fold=28,scale=.85)
    fig.savefig(output/"circuit.svg",bbox_inches="tight",metadata={"Date":None})
    plt.close(fig)
    for filename,reflection in (("initial-reflection.svg",pa.zero_subregister_reflection(selected_source.num_qubits,range(selected_source.num_qubits))),
                                ("good-reflection.svg",pa.zero_subregister_reflection(selected_source.num_qubits,range(2,selected_source.num_qubits)))):
        fig=reflection.draw(output="mpl",fold=28,scale=.85)
        fig.savefig(output/filename,bbox_inches="tight",metadata={"Date":None})
        plt.close(fig)
    if rows:
        fig,axes=plt.subplots(1,2,figsize=(9,3.6))
        for b in betas:
            group=[row for row in rows if row["beta"]==b]
            axes[0].plot([row["rounds"] for row in group],[row["acceptance"] for row in group],"o-",label=f"beta={b:g}")
            axes[1].plot([row["rounds"] for row in group],[row["gibbs_trace_distance"] for row in group],"o-")
        axes[0].set(xlabel="Integer amplification rounds",ylabel="Actual all-zero-label acceptance",ylim=(0,1.02))
        axes[1].set(xlabel="Integer amplification rounds",ylabel="Gibbs trace-distance bias")
        axes[1].ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        axes[0].legend()
        fig.suptitle("Known-input amplification: oscillating success, unchanged thermal bias")
        fig.tight_layout()
        fig.savefig(output/"success-and-bias.svg",metadata={"Date":None})
        plt.close(fig)
    if scaling_rows:
        fig,ax=plt.subplots(figsize=(7,3.4))
        width=.34
        locations=np.arange(len(scaling_rows))
        ax.bar(locations-width/2,[row["full_initial_reflection_primitive_gates"] for row in scaling_rows],width,label="Full-input S0")
        ax.bar(locations+width/2,[row["good_label_reflection_primitive_gates"] for row in scaling_rows],width,label="Labels-only Sf")
        ax.set(xticks=locations,xticklabels=[str(row["n_system"]) for row in scaling_rows],xlabel="System spins n",ylabel="Primitive reflection gates")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output/"reflection-scaling.svg",metadata={"Date":None})
        plt.close(fig)
    for name in ("circuit.svg","initial-reflection.svg","good-reflection.svg","success-and-bias.svg","reflection-scaling.svg"):
        path=output/name
        if path.exists():
            normalize_svg(path)
    return {"output_dir":str(output),"validation_performed":simulate,"default_case_id":selected_id,"default_validation":default["validation"],
            "case_count":len(cases),"scaling_rows":scaling_rows,"sampling_performed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z",type=float,default=.6)
    parser.add_argument("--x",type=float,default=.4)
    parser.add_argument("--beta",type=float,default=1.25)
    parser.add_argument("--degree",type=int,default=6)
    parser.add_argument("--rounds",default="0,1,2,3,4")
    parser.add_argument("--betas",default="1.25,2")
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/simulation/preparation-amplification" if REPOSITORY else HERE/"generated-preparation-amplification")
    args=parser.parse_args()
    result=generate(args.output_dir,args.z,args.x,args.beta,args.degree,tuple(int(x) for x in args.rounds.split(",")),
                    tuple(float(x) for x in args.betas.split(",")),not args.no_sim,not args.no_scaling)
    print(json.dumps(jsonable(result),indent=2,allow_nan=False))


if __name__=="__main__":
    main()
