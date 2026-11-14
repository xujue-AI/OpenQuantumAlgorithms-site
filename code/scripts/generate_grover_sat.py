#!/usr/bin/env python3
"""Generate reversible-CNF search artifacts and full fixed-prefix replay costs.

The BBHT schedule never receives an exact satisfying-assignment count. Any tiny
exhaustive diagnostic is run after all seeded searches. Physical projections
cost the complete recorded quantum prefix with fresh banks and final readout,
not an adaptive algorithm's global wall time or success confidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
try:
    from source_provenance import git_revision
except ImportError:
    # Detached code downloads retain file hashes, without invented ancestry.
    def git_revision(root):
        return None

from qiskit import qasm3
from qiskit.quantum_info import Statevector

from openquantum.bbht_search import BBHTConfig, run_bbht
from openquantum.grover_sat import (
    CNF, amplification_probability, build_cnf_phase_oracle,
    build_sat_grover_circuit, circuit_fingerprint, exhaustive_solutions,
    replay_recorded_attempts, preflight_sat_circuit,
    MAX_ESTIMATED_PRIMITIVE_OPERATIONS, MAX_PRIMITIVE_OPERATIONS,
)
from openquantum.resource_estimation import (
    CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources,
    prepare_measured_circuit,
)

EXAMPLE = CNF(4, ((1,2),(-1,3),(-2,-3),(4,)))
CONTRADICTORY = CNF(2, ((1,),(-1,)))


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def protected_baseline_hashes() -> dict:
    names = [
        "src/openquantum/grover.py", "scripts/estimate_resources.py",
        "assets/generated/grover/circuit.qasm",
        "assets/generated/resources/grover-surface-code.json",
        "assets/generated/resources/grover-surface-code.csv",
    ]
    return {name:sha_file(ROOT/name) for name in names}


def guard_output_directory(output: Path) -> None:
    for name in ("assets/generated/grover", "assets/generated/resources", "assets/generated/grover-scaling"):
        protected=(ROOT/name).resolve()
        if output.resolve()==protected or output.resolve().is_relative_to(protected):
            raise ValueError("SAT artifacts must not overlap completed/frozen Grover assets")


def environment() -> dict:
    source_paths=[
        "src/openquantum/grover_sat.py", "src/openquantum/bbht_search.py",
        "src/openquantum/resource_estimation.py", "scripts/generate_grover_sat.py",
        "requirements-lock.txt", "scripts/source_provenance.py",
    ]
    return {
        "source_commit":git_revision(ROOT),
        "source_sha256":{name:sha_file(ROOT/name) for name in source_paths if (ROOT/name).exists()},
        "python":platform.python_version(),"platform":platform.platform(),
        "packages":{name:importlib.metadata.version(name) for name in ["qiskit","qdk","numpy","matplotlib"]},
        "simulator":"Bounded ideal CPU Statevector; no hardware outcomes",
        "transpiler":{"optimization_level":0,"seed_transpiler":7,"reason":"Preserve exact Clifford+T decomposition; no artificial arbitrary rotations"},
    }


def export_iteration(cnf: CNF, iterations: int, path: Path) -> dict:
    preflight=preflight_sat_circuit(cnf,iterations,measure=True)
    circuit=build_sat_grover_circuit(cnf,iterations,measure=True)
    compiled=prepare_measured_circuit(circuit,optimization_level=0)
    if len(compiled.data)>MAX_PRIMITIVE_OPERATIONS:
        raise ValueError("Lowered SAT iteration exceeds the actual primitive-operation cap")
    source=qasm3.dumps(compiled)
    path.write_text(source,encoding="utf-8")
    return {
        "iterations":iterations,"raw_structural_sha256":circuit_fingerprint(circuit),
        "qasm_sha256":hashlib.sha256(source.encode()).hexdigest(),
        "source_gate_counts":dict(circuit.count_ops()),
        "decomposed_gate_counts":dict(compiled.count_ops()),
        "declared_qubits":compiled.num_qubits,"data_measurements":cnf.n_variables,
        "clause_workspace_qubits":cnf.num_clause_qubits,
        "clean_conjunction_work_qubits":cnf.num_clean_work_qubits,
        "materialization_preflight":preflight,
    }


def bounded_reference(cnf: CNF) -> dict:
    if cnf.n_variables>8 or cnf.num_qubits>14:
        return {"status":"not_run","reason":"Exhaustive reference has separate n<=8,total-width<=14 bounds"}
    started=time.perf_counter_ns()
    solutions=exhaustive_solutions(cnf,max_variables=8)
    oracle=build_cnf_phase_oracle(cnf)
    uniform=Statevector.from_instruction(build_sat_grover_circuit(cnf,0))
    expected=uniform.data.copy()
    for assignment in solutions:
        expected[assignment]*=-1
    actual=uniform.evolve(oracle)
    maximum_error=float(max(abs(actual.data-expected)))
    if maximum_error>1e-10:
        raise RuntimeError("Coherent clean-workspace phase oracle failed its bounded reference")
    probabilities=[]
    for iterations in range(7):
        state=Statevector.from_instruction(build_sat_grover_circuit(cnf,iterations))
        measured=sum(float(state.probabilities()[x]) for x in solutions)
        predicted=amplification_probability(cnf.n_variables,len(solutions),iterations)
        if abs(measured-predicted)>1e-10:
            raise RuntimeError("Multiple-marked ideal reference disagrees with the circuit")
        probabilities.append({"iterations":iterations,"statevector_success":measured,"known_count_formula":predicted})
    return {
        "status":"checked_after_search","assignments_evaluated":2**cnf.n_variables,
        "solutions":solutions,"solution_bitstrings":[f"{x:0{cnf.n_variables}b}" for x in solutions],
        "marked_count":len(solutions),"coherent_phase_max_amplitude_error":maximum_error,
        "amplification":probabilities,"diagnostic_cpu_ns":time.perf_counter_ns()-started,
        "used_by_unknown_count_search":False,
        "boundary":"Exponential tiny classical reference, not a free large-instance count",
    }


def audit_replay(circuit, cnf: CNF, counts: list[int]) -> dict:
    width=cnf.num_qubits
    seen_measure=False
    mappings=[]
    for item in circuit.data:
        if item.operation.name=="measure":
            seen_measure=True
            mappings.append((circuit.find_bit(item.qubits[0]).index,circuit.find_bit(item.clbits[0]).index))
        else:
            if seen_measure:
                raise RuntimeError("Replay has work after a measurement")
            if item.operation.name!="barrier":
                banks={circuit.find_bit(q).index//width for q in item.qubits}
                if len(banks)!=1:
                    raise RuntimeError("Replay unitary couples distinct fresh banks")
    expected=[(bank*width+q,bank*cnf.n_variables+q) for bank in range(len(counts)) for q in range(cnf.n_variables)]
    if mappings!=expected:
        raise RuntimeError("Replay output mapping differs from the complete recorded prefix")
    return {
        "bank_local_unitaries_checked":True,"all_final_measurements_checked":True,
        "declared_qubits":circuit.num_qubits,"banks":len(counts),"bank_width":width,
        "total_data_qubits":len(counts)*cnf.n_variables,
        "total_clause_workspace_qubits":len(counts)*cnf.num_clause_qubits,
        "total_clean_conjunction_work_qubits":len(counts)*cnf.num_clean_work_qubits,
        "total_output_measurements":len(mappings),"recorded_iterations":counts,
        "oracle_calls":sum(counts),
        "raw_structural_sha256":circuit_fingerprint(circuit),
        "source_gate_counts":dict(circuit.count_ops()),
        "initialization_assumption":"All fresh banks start in logical zero; separate physical initialization protocol/time is uncalibrated",
        "measurement_semantics":"Fixed independent banks allow deferred readout; replay outcomes are fresh, not forced to match logged candidates",
        "scope":"Full fixed recorded quantum prefix with fresh banks, not adaptive search execution",
    }


def project_recorded_path(cnf: CNF, run: dict, output: Path, name: str, profiles: list[SurfaceCodeProfile], cache: Path, max_replay_qubits: int) -> dict:
    counts=[attempt["iterations"] for attempt in run["attempts"]]
    result={
        "id":name,"seed":run["config"]["seed"],"search_status":run["status"],
        "recorded_attempts":len(counts),"recorded_oracle_calls":sum(counts),
        "classical_verification_calls":run["classical_verification_calls"],
        "classical_literal_checks":run["classical_literal_checks"],
        "classical_clause_checks":run["classical_clause_checks"],
        "recorded_classical_verification_cpu_ns":run["classical_verification_cpu_ns"],
        "search_failure_confidence":None,
        "scope":"Fixed recorded quantum-prefix replay cost; not a global adaptive runtime or unknown-M success bound",
        "profiles":[],
    }
    if not counts:
        result.update(status="no_executed_quantum_prefix",reason=run["termination_reason"])
        return result
    preflight_operations=sum(preflight_sat_circuit(cnf,count)["estimated_primitive_operations"] for count in counts)
    if preflight_operations>MAX_ESTIMATED_PRIMITIVE_OPERATIONS:
        result.update(status="pending_physical_model",reason="Combined fixed replay exceeds operation preflight; no replay materialized")
        return result
    replay=replay_recorded_attempts(cnf,counts,measure=True)
    result["logical"]=audit_replay(replay,cnf,counts)
    path=output/"replay-circuits"/f"{name}.qasm"
    compiled=prepare_measured_circuit(replay,optimization_level=0)
    if len(compiled.data)>MAX_PRIMITIVE_OPERATIONS:
        result.update(status="pending_physical_model",reason="Lowered full replay exceeds primitive cap; no native query performed")
        return result
    qasm=qasm3.dumps(compiled)
    path.write_text(qasm,encoding="utf-8")
    result["logical"].update(
        qasm_sha256=hashlib.sha256(qasm.encode()).hexdigest(),
        decomposed_gate_counts=dict(compiled.count_ops()),qasm_relative_path=f"replay-circuits/{name}.qasm",
        estimated_primitive_operations=preflight_operations,
        actual_primitive_operations=len(compiled.data),
    )
    if replay.num_qubits>max_replay_qubits:
        result.update(status="pending_physical_model",reason="Full recorded replay exceeds physical-query width cap; complete logical cost retained")
        return result
    budget=CircuitResourceBudget(
        synthesis_model_error_per_execution=1e-4,
        physical_failure_probability=.005,
        executions=1,
    )
    result["whole_fixed_replay_budget"]=asdict(budget)
    for profile in profiles:
        item={"physical_profile":asdict(profile)}
        started=time.perf_counter_ns()
        try:
            report=estimate_circuit_resources(replay,budget=budget,profile=profile,cache_dir=cache/profile.name,use_graph=False,optimization_level=0)
            if report["qasm_sha256"]!=result["logical"]["qasm_sha256"]:
                raise RuntimeError("Physical replay differs from the audited complete measured prefix")
            report.pop("qasm3")
            for point in report["pareto_points"]:
                point["peak_times_replay_model_runtime_qubit_ns"]=point["peak_physical_qubits"]*point["runtime_per_execution_ns"]
            item.update(status="computed_conditional_model",estimate=report)
        except Exception as exc:
            item.update(status="pending_physical_model",reason=f"{type(exc).__name__}: {exc}")
        item["estimator_cpu_wall_ns"]=time.perf_counter_ns()-started
        result["profiles"].append(item)
    result["status"]="computed_conditional_model" if all(p["status"]=="computed_conditional_model" for p in result["profiles"]) else "pending_physical_model"
    result["limitations"]=[
        "Full fixed replay with fresh banks trades additional logical memory for avoiding unsupported reset/reuse/feedback",
        "Canonical PSSPC input retains non-Clifford/measurement counts on placeholder wires, not the source gate DAG; replay model time is not a validated circuit schedule or adaptive BBHT wall time",
        "Clean logical-zero input is an explicit initialization assumption, not a calibrated preparation/reset protocol",
        "Classical verifier work/CPU time is recorded separately; device-to-host control/verification latency is not calibrated",
        "Exact Clifford+T lowering has zero arbitrary rotations; physical-fault allowances do not certify adaptive trajectory probabilities",
        "One recorded ideal prefix does not certify all adaptive branches or a global unknown-count success/runtime bound",
    ]
    return result


def write_resource_table(output: Path, resources: dict) -> None:
    lines=[
        "| Recorded prefix | Outcome | Attempts | Oracle calls | Fresh replay qubits | Profile p | Pareto choice | Peak physical qubits | Replay model time (ms) | Synthesis model scalar | Physical fault model |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    summaries=[]
    for path in resources["paths"]:
        for profile in path.get("profiles",[]):
            p=profile["physical_profile"]["physical_error_rate"]
            if profile["status"]!="computed_conditional_model":
                lines.append(f"| {path['id']} | {path['search_status']} | {path['recorded_attempts']} | {path['recorded_oracle_calls']} | {path.get('logical',{}).get('declared_qubits','—')} | {p:g} | Pending full replay model | — | — | — | — |")
                continue
            estimate=profile["estimate"]
            for label,key in [
                ("Fewest qubits",lambda x:(x["peak_physical_qubits"],x["runtime_per_execution_ns"])),
                ("Fastest replay model",lambda x:(x["runtime_per_execution_ns"],x["peak_physical_qubits"])),
            ]:
                point=min(estimate["pareto_points"],key=key)
                lines.append(
                    f"| {path['id']} | {path['search_status']} | {path['recorded_attempts']} | {path['recorded_oracle_calls']} | "
                    f"{path['logical']['declared_qubits']} | {p:g} | {label} | {point['peak_physical_qubits']:,} | "
                    f"{point['runtime_per_execution_ns']/1e6:.6g} | {point['synthesis_model_error_per_execution']:.6g} | "
                    f"{point['physical_failure_model_per_execution']:.6g} |"
                )
                summaries.append({"id":path["id"],"p":p,"choice":label,**point})
        if not path.get("profiles"):
            lines.append(f"| {path['id']} | {path['search_status']} | {path['recorded_attempts']} | {path['recorded_oracle_calls']} | {path.get('logical',{}).get('declared_qubits','—')} | — | {path['status']} | — | — | — | — |")
    (output/"resource-table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    resources["summary_selections"]=summaries


def draw_figures(output: Path, cnf: CNF, reference: dict, resources: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"openquantum-grover-sat-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if cnf.num_qubits<=12:
        figure=build_cnf_phase_oracle(cnf).draw(output="mpl",fold=28,scale=.72)
        figure.savefig(output/"oracle.svg",bbox_inches="tight")
        plt.close(figure)
    if reference["status"]=="checked_after_search":
        rows=reference["amplification"]
        fig,ax=plt.subplots(figsize=(7,3.5))
        ax.plot([x["iterations"] for x in rows],[x["statevector_success"] for x in rows],"o-",label="Actual CNF circuit")
        ax.plot([x["iterations"] for x in rows],[x["known_count_formula"] for x in rows],"x--",label="Bounded reference count formula")
        ax.set(xlabel="Grover iterations",ylabel="Ideal satisfying-assignment probability",ylim=(0,1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output/"amplification.svg")
        plt.close(fig)
    summary=resources.get("summary_selections",[])
    if summary:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        ids=[x["id"] for x in resources["paths"]]
        colors={.001:"#176a69",.0001:"#a34f24"}
        for p in sorted({x["p"] for x in summary},reverse=True):
            for choice,style in [("Fewest qubits","o-"),("Fastest replay model","s--")]:
                rows=[next((x for x in summary if x["id"]==name and x["p"]==p and x["choice"]==choice),None) for name in ids]
                for ax,field,scale in [(axes[0],"peak_physical_qubits",1),(axes[1],"runtime_per_execution_ns",1e6)]:
                    values=[x[field]/scale if x else float("nan") for x in rows]
                    ax.semilogy(range(len(ids)),values,style,color=colors[p],label=f"p={p:g}, {choice}")
                    ax.set_xticks(range(len(ids)),ids,rotation=15)
                    ax.grid(alpha=.2)
        axes[0].set_ylabel("Peak physical qubits")
        axes[1].set_ylabel("Fixed replay model time (ms)")
        axes[0].legend(fontsize=7)
        fig.suptitle("Full recorded quantum prefixes, replayed with fresh disjoint banks",fontsize=11)
        fig.text(.5,.01,"Conditional surface-code/layout projection; not adaptive BBHT time, unknown-M confidence, or a reset/initialization certificate",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.06,1,.94))
        fig.savefig(output/"replay-resources.svg")
        plt.close(fig)


MANAGED_NAMES={"input.json","search-traces.json","replay-resources.json","resource-table.md",
               "oracle.svg","amplification.svg","replay-resources.svg","iteration-circuits.zip"}


def managed_artifact(path: Path) -> bool:
    if len(path.parts)==1:
        return path.name in MANAGED_NAMES
    if len(path.parts)!=2:
        return False
    if path.parts[0]=="iteration-circuits":
        return re.fullmatch(r"(?:sat|contradictory)-r[0-9]+\.qasm",path.name) is not None
    if path.parts[0]=="replay-circuits":
        return re.fullmatch(r"(?:sat|contradictory)-seed-[0-9]+\.qasm",path.name) is not None
    return False


def clear_managed_artifacts(output: Path) -> None:
    """Remove stale generated outputs, preserving unrelated user files."""
    if not output.exists():
        return
    for path in output.rglob("*"):
        if path.is_file() and managed_artifact(path.relative_to(output)):
            path.unlink()


def archive_iteration_exports(output: Path, exports: dict) -> None:
    """Bundle only actual generated iterations, excluding unrelated files."""
    with zipfile.ZipFile(output/"iteration-circuits.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        for name,count in sorted(exports):
            path=output/"iteration-circuits"/f"{name}-r{count}.qasm"
            info=zipfile.ZipInfo(path.name,date_time=(1980,1,1,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            info.external_attr=0o644<<16
            archive.writestr(info,path.read_bytes())


def generate(output: Path, *, cnf: CNF=EXAMPLE, seeds: tuple[int,...]=(7,11,29), config: BBHTConfig=BBHTConfig(), cache_dir: Path|None=None, max_replay_qubits: int=128) -> dict:
    guard_output_directory(output)
    if not seeds or len(seeds)>8 or len(set(seeds))!=len(seeds):
        raise ValueError("Choose one through eight distinct predetermined seeds")
    if cnf.num_qubits>config.max_simulation_qubits:
        raise ValueError("Search exceeds bounded simulation width; use generation-only mode")
    preflight_sat_circuit(cnf,1)
    before=protected_baseline_hashes()
    clear_managed_artifacts(output)
    output.mkdir(parents=True,exist_ok=True)
    (output/"iteration-circuits").mkdir(exist_ok=True)
    (output/"replay-circuits").mkdir(exist_ok=True)
    write_json(output/"input.json",cnf.as_dict())
    runs=[]
    for seed in seeds:
        run=run_bbht(cnf,BBHTConfig(**{**asdict(config),"seed":seed}))
        run["id"]=f"sat-seed-{seed}"
        runs.append(run)
        print(run["id"],run["status"],[x["iterations"] for x in run["attempts"]],flush=True)
    contradiction=run_bbht(CONTRADICTORY,BBHTConfig(seed=19,max_attempts=8,max_oracle_calls=32,max_wall_seconds=30))
    contradiction["id"]="contradictory-seed-19"
    runs.append(contradiction)
    inputs={run["id"]:cnf if run is not contradiction else CONTRADICTORY for run in runs}
    exports={}
    for name,current in [("sat",cnf),("contradictory",CONTRADICTORY)]:
        values={1}
        for run in runs:
            if inputs[run["id"]] is current:
                values.update(x["iterations"] for x in run["attempts"])
        for count in sorted(values):
            path=output/"iteration-circuits"/f"{name}-r{count}.qasm"
            exports[(name,count)]=export_iteration(current,count,path)
    for run in runs:
        name="contradictory" if run is contradiction else "sat"
        for attempt in run["attempts"]:
            exported=exports[(name,attempt["iterations"])]
            if exported["raw_structural_sha256"]!=attempt["source_circuit_sha256"]:
                raise RuntimeError("Exported iteration circuit differs from the actual sampled source")
            if exported["qasm_sha256"]!=attempt["simulation_input_qasm_sha256"]:
                raise RuntimeError("Exported iteration differs from the actual lowered simulation input")
            attempt["decomposed_qasm_sha256"]=exported["qasm_sha256"]
            attempt["decomposed_qasm_path"]=f"iteration-circuits/{name}-r{attempt['iterations']}.qasm"
    # Only now run exact tiny reference work. It never drives a search decision.
    reference=bounded_reference(cnf)
    contradiction_reference=bounded_reference(CONTRADICTORY)
    write_json(output/"search-traces.json",{
        "environment":environment(),"runs":runs,
        "bounded_reference_after_search":reference,
        "contradictory_reference_after_search":contradiction_reference,
        "frozen_sha256_before":before,
    })
    baseline=json.loads((ROOT/"assets/generated/resources/grover-surface-code.json").read_text())
    profiles=[SurfaceCodeProfile(name=x["id"],physical_error_rate=x["physical_error_rate"],gate_time_ns=x["gate_time_ns"],measurement_time_ns=x["measurement_time_ns"]) for x in baseline["profiles"]]
    resources={
        "environment":environment(),"scope":"Full fixed recorded quantum prefixes only",
        "unknown_solution_count_search_confidence":None,
        "fixed_replay_budget":{"synthesis_model_allowance":1e-4,"physical_failure_allowance":.005,"executions":1},
        "paths":[],
    }
    cache=cache_dir or Path(tempfile.gettempdir())/"openquantum-grover-sat-qre-cache"
    for run in runs:
        path=project_recorded_path(inputs[run["id"]],run,output,run["id"],profiles,cache,max_replay_qubits)
        resources["paths"].append(path)
        write_json(output/"replay-resources.json",resources)
        print(run["id"],path["status"],flush=True)
    write_resource_table(output,resources)
    draw_figures(output,cnf,reference,resources)
    archive_iteration_exports(output,exports)
    after=protected_baseline_hashes()
    if before!=after:
        raise RuntimeError("Frozen Grover baseline changed")
    resources.update(frozen_files_unchanged=True,frozen_sha256_before=before,frozen_sha256_after=after)
    resources["status"]="computed_conditional_model" if all(x["status"]=="computed_conditional_model" for x in resources["paths"]) else "partial_physical_model"
    write_json(output/"replay-resources.json",resources)
    return resources


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnf",type=Path)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/grover-sat")
    parser.add_argument("--seeds",default="7,11,29")
    parser.add_argument("--max-attempts",type=int,default=16)
    parser.add_argument("--max-oracle-calls",type=int,default=32)
    parser.add_argument("--soft-wall-seconds",type=float,default=30)
    parser.add_argument("--cache-dir",type=Path)
    parser.add_argument("--generate-only",action="store_true")
    parser.add_argument("--iterations",type=int,default=1)
    args=parser.parse_args()
    cnf=CNF.from_dict(json.loads(args.cnf.read_text())) if args.cnf else EXAMPLE
    guard_output_directory(args.output_dir)
    if cnf.num_qubits>32 or args.iterations>64 or args.max_attempts>32 or args.max_oracle_calls>128:
        raise ValueError("CLI materialization/query caps exceeded; core circuit API is separately parameterized")
    if args.generate_only:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        result=export_iteration(cnf,args.iterations,args.output_dir/"circuit.qasm")
        write_json(args.output_dir/"generation-only.json",{"status":"circuit_generated_no_search_no_physical_estimate","cnf":cnf.as_dict(),"environment":environment(),**result})
        print("Generated bound circuit without simulation, solution count, or physical query")
        return 0
    config=BBHTConfig(max_attempts=args.max_attempts,max_oracle_calls=args.max_oracle_calls,max_wall_seconds=args.soft_wall_seconds)
    result=generate(args.output_dir,cnf=cnf,seeds=tuple(int(x) for x in args.seeds.split(",")),config=config,cache_dir=args.cache_dir)
    print(json.dumps({"status":result["status"],"frozen_files_unchanged":result["frozen_files_unchanged"]}))
    return 0 if result["status"]=="computed_conditional_model" else 2


if __name__=="__main__":
    raise SystemExit(main())
