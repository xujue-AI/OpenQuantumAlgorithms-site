#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Generate a bounded fresh-preparation Hadamard-QPE demo and complete ledgers.

The default has four prescribed phases, three independent seeded count streams,
48 experiment circuits and 47,376 modeled fresh trials. Counts are classically
sampled from checked ideal circuit probabilities; no QPU or Native estimator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import csv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import qasm3
try:
    from openquantum.hadamard_qpe import (
        MAX_DENSE_TARGET_QUBITS, MAX_EXPERIMENT_CIRCUITS, MAX_MODELED_DRAWS,
        actual_angle_phase, analytic_zero_probability, build_experiment, build_schedule,
        check_experiment, circular_distance, ideal_failure_bound, lower_experiment,
        protocol_ledger, recover_phase, sample_counts, shots_for_failure,
    )
except ModuleNotFoundError as error:
    if error.name not in {"openquantum", "openquantum.hadamard_qpe"}:
        raise
    from hadamard_qpe import (
        MAX_DENSE_TARGET_QUBITS, MAX_EXPERIMENT_CIRCUITS, MAX_MODELED_DRAWS,
        actual_angle_phase, analytic_zero_probability, build_experiment, build_schedule,
        check_experiment, circular_distance, ideal_failure_bound, lower_experiment,
        protocol_ledger, recover_phase, sample_counts, shots_for_failure,
    )

MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_WALL_SECONDS = 180
CANONICAL_PHASES = (.3, .995, .125, .5)
CANONICAL_SEEDS = (7, 19, 41)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_files() -> dict[str, Path]:
    module = ROOT / "src/openquantum/hadamard_qpe.py"
    if not module.is_file():
        module = Path(__file__).resolve().parent / "hadamard_qpe.py"
    result = {"hadamard_qpe.py": module, "generate_hadamard_qpe.py": Path(__file__).resolve()}
    for name in ("requirements-lock.txt", "LICENSE"):
        path = ROOT / name
        if not path.is_file():
            path = Path(__file__).resolve().parent / ("LICENSE.txt" if name == "LICENSE" else name)
        result["LICENSE.txt" if name == "LICENSE" else name] = path
    if any(not path.is_file() or path.is_symlink() for path in result.values()):
        raise ValueError("Complete source, lock and license files are required")
    return result


def checkpoint(start: float, output: Path) -> None:
    if time.perf_counter() - start > MAX_WALL_SECONDS:
        raise TimeoutError("Hadamard generator exceeded its bounded wall allowance")
    if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > MAX_OUTPUT_BYTES:
        raise ValueError("Generated output exceeds the 64 MiB cap")


def export(path: Path, circuit) -> dict:
    lowered = lower_experiment(circuit)
    raw = qasm3.dumps(lowered).encode()
    path.write_bytes(raw)
    return {"file": path.name, "sha256": digest(raw), "source_gate_counts": dict(circuit.count_ops()),
            "source_dependency_depth": circuit.depth(), "lowered_gate_counts": dict(lowered.count_ops()),
            "lowered_dependency_depth": lowered.depth(), "logical_qubits": circuit.num_qubits,
            "lowered_global_phase_radians": float(lowered.global_phase),
            "export_scope": "fresh-reset measured trial; global phase omitted by QASM does not change readout"}


def generate(output: Path, *, target_qubits: int = 1, phase: float = .3, levels: int = 6,
             failure_target: Fraction = Fraction(1,100), seed: int = 7,
             canonical_study: bool = True, simulate: bool = True, scaling: bool = True) -> dict:
    start = time.perf_counter()
    if not isinstance(canonical_study, bool) or not isinstance(simulate, bool) or not isinstance(scaling, bool):
        raise ValueError("Study, simulation and scaling flags must be booleans")
    output = output.absolute()
    destination = output.resolve()
    protected = ROOT / "assets/generated"
    if destination.is_relative_to(protected.resolve()) and not destination.is_relative_to((protected / "hadamard-qpe").resolve()):
        raise ValueError("Output cannot overwrite another algorithm's artifacts")
    if any(destination == (ROOT / name).resolve() or destination.is_relative_to((ROOT / name).resolve())
           for name in ("src", "scripts", "notebooks", "algorithms", "references")) or destination == ROOT:
        raise ValueError("Output cannot overwrite project source")
    if isinstance(failure_target, bool):
        raise ValueError("Ideal failure target must be in (0,1)")
    try:
        failure_target = failure_target if isinstance(failure_target, Fraction) else Fraction(str(failure_target))
    except (ValueError, ZeroDivisionError, TypeError) as error:
        raise ValueError("Ideal failure target must be in (0,1)") from error
    if not 0 < failure_target < 1:
        raise ValueError("Ideal failure target must be in (0,1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < (1 << 63):
        raise ValueError("Seed must be a nonnegative bounded integer")
    phases = CANONICAL_PHASES if canonical_study else (phase,)
    seeds = CANONICAL_SEEDS if canonical_study else (seed,)
    if canonical_study and (target_qubits != 1 or levels != 6 or phase != .3 or seed != 7 or failure_target != Fraction(1,100)):
        raise ValueError("Canonical study has frozen defaults; use --single for changed parameters")
    # Guard before constructing circuits or sampling.
    if len(phases) * 2 * levels + (2 if scaling else 0) > MAX_EXPERIMENT_CIRCUITS:
        raise ValueError("Complete generation exceeds its 64-circuit cap")
    run_count = len(phases) * len(seeds)
    per_run_failure = failure_target / run_count
    m = shots_for_failure(levels, per_run_failure)
    total_trials = run_count * 2 * levels * m if simulate else 0
    if total_trials > MAX_MODELED_DRAWS:
        raise ValueError("Complete study exceeds its 100,000 modeled-trial cap")
    if simulate and target_qubits > MAX_DENSE_TARGET_QUBITS:
        raise ValueError("Use --no-sim for more than three data wires")
    old_owned = set()
    if output.exists():
        if output.is_symlink() or not output.is_dir():
            raise ValueError("Output must be a regular owned directory")
        existing = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if existing:
            prior_path = output / "manifest.json"
            if prior_path.is_symlink() or not prior_path.is_file():
                raise ValueError("Nonempty output requires this generator's ownership manifest")
            prior = json.loads(prior_path.read_text())
            if prior.get("schema") != "openquantum-hadamard-qpe-files-v1":
                raise ValueError("Output belongs to another generator")
            old_owned = set(prior.get("file_sha256", {})) | {"manifest.json"}
            if existing != old_owned or any(Path(n).is_absolute() or ".." in Path(n).parts for n in old_owned):
                raise ValueError("Output contains unowned or invalid paths")
    files = source_files()
    source_hashes = {name: digest(path.read_bytes()) for name, path in files.items()}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=5)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="hadamard-qpe-stage-", dir=output.parent) as temp:
        staging = Path(temp)
        (staging / "circuits").mkdir()
        for name, path in files.items():
            shutil.copyfile(path, staging / name)
        cases = []
        runs = []
        for index, current_phase in enumerate(phases):
            circuits = build_schedule(target_qubits, current_phase, levels)
            case = {"case_id": f"phase-{index}", "nominal_phase": current_phase,
                    "actual_angle_phase_reference": actual_angle_phase(target_qubits,current_phase),
                    "actual_angle_reference_scope": "binary64 diagnostic from stored P angles; not interval certified",
                    "experiments": []}
            probabilities = []
            for j, circuit in enumerate(circuits):
                k = 1 << (j // 2)
                basis = "cos" if j % 2 == 0 else "sin"
                entry = {"power": k, "basis": basis, **export(staging / "circuits" / f"phase-{index}-{basis}-k{k}.qasm", circuit)}
                entry["file"] = "circuits/" + entry["file"]
                if simulate:
                    entry["check"] = check_experiment(target_qubits,current_phase,k,basis)
                    raw_probability = entry["check"]["zero_probability"]
                    sampling_probability = min(1.0,max(0.0,raw_probability))
                    if abs(sampling_probability-raw_probability)>2e-11:
                        raise AssertionError("Circuit probability roundoff exceeds its explicit guard")
                    entry["sampling_zero_probability"] = sampling_probability
                    entry["sampling_probability_clipping_absolute_error"] = abs(sampling_probability-raw_probability)
                    probabilities.append(sampling_probability)
                else:
                    entry["check"] = {"performed": False, "reason": "explicit --no-sim", "certified": False}
                case["experiments"].append(entry)
                checkpoint(start,staging)
            cases.append(case)
            if simulate:
                for current_seed in seeds:
                    inference_start=time.perf_counter()
                    counts=sample_counts(probabilities,levels,m,current_seed,stream=index)
                    recovered=recover_phase(counts)
                    value=recovered["phase_estimate"]
                    runs.append({"case_id":case["case_id"],"seed":current_seed,"stream":index,
                                 "seed_scope":"SeedSequence([seed, phase_index]); separate predefined ideal binomial-count streams",
                                 "result":recovered,"circular_error_vs_actual_angle_reference":None if value is None else circular_distance(value,case["actual_angle_phase_reference"]),
                                 "circular_error_vs_nominal_phase":None if value is None else circular_distance(value,current_phase),
                                 "classical_sampling_and_recovery_seconds":time.perf_counter()-inference_start})
                    checkpoint(start,staging)
        common_ledger=protocol_ledger(target_qubits,levels,m)
        result={"schema":"openquantum-hadamard-qpe-v1","parameters":{"target_qubits":target_qubits,"levels":levels,"phases":list(phases),"seeds":list(seeds),"canonical_study":canonical_study,
                "total_ideal_failure_target_exact":str(failure_target),"per_run_ideal_failure_target_exact":str(per_run_failure),"shots_per_fringe":m},
                "environment":{"python":platform.python_version(),"device":"CPU","packages":{p:importlib.metadata.version(p) for p in ("qiskit","numpy","matplotlib")}},
                "provenance":{"source_commit":commit,"source_sha256":source_hashes,"source_revision_scope":"Exact source hashes identify this producer; Git HEAD is recorded when available and may precede uncommitted source edits"},
                "simulation":{"performed":simulate,"sampling_scope":"classically sampled binomial counts from checked circuit probabilities, representing fresh ideal trials; no QPU execution",
                              "modeled_fresh_trials_sampled":total_trials,"unique_experiment_circuits":len(phases)*2*levels},
                "ideal_confidence":{"phase_radius_exact":common_ledger["ideal_phase_radius_exact"],"per_run_failure_upper_bound_exact":str(ideal_failure_bound(levels,m)),
                    "whole_predeclared_study_failure_upper_bound_exact":str(min(Fraction(1),run_count*ideal_failure_bound(levels,m))),
                    "scope":"union bound over predefined outputs, conditional on exact eigenstate/powers, fresh IID ideal trials and exact-real angular inference",
                    "numerical_inference_certified":False,"hardware_confidence_certified":False},
                "cases":cases,"runs":runs,"common_per_output_ledger":common_ledger,
                "whole_study_modeled_workload":{"outputs":run_count,"complete_fresh_preparations":run_count*common_ledger["complete_fresh_preparations"],
                    "shot_weighted_controlled_U_calls":run_count*common_ledger["shot_weighted_controlled_U_calls"],
                    "source_gate_counts":{g:run_count*c for g,c in common_ledger["source_gate_counts_all_trials"].items()},
                    "workload_status":"sampled ideal counts" if simulate else "construction/count proposal only"},
                "physical_estimate_performed":False,"synthesis_performed":False}
        if simulate:
            result["observed_all_prescribed_runs_within_ideal_radius"]=all(r["circular_error_vs_actual_angle_reference"] is not None and r["circular_error_vs_actual_angle_reference"]<=float(Fraction(common_ledger["ideal_phase_radius_exact"])) for r in runs)
            draw=build_experiment(target_qubits,phases[0],min(2,1 << (levels-1)),"sin")
            fig=draw.draw(output="mpl",fold=32);fig.savefig(staging/"circuit.svg",bbox_inches="tight");plt.close(fig)
            fig,ax=plt.subplots(figsize=(8,3.5))
            labels=[f'{r["case_id"]}/s{r["seed"]}' for r in runs]
            errors=[r["circular_error_vs_actual_angle_reference"] if r["circular_error_vs_actual_angle_reference"] is not None else float("nan") for r in runs]
            ax.bar(range(len(runs)),errors,color="#216c72")
            for i,r in enumerate(runs):
                if r["circular_error_vs_actual_angle_reference"] is None:
                    ax.text(i,float(Fraction(common_ledger["ideal_phase_radius_exact"]))/2,"ambiguous",rotation=90,ha="center")
            ax.axhline(float(Fraction(common_ledger["ideal_phase_radius_exact"])),color="#b5643e",linestyle="--",label="Ideal-model radius")
            ax.set(xticks=range(len(runs)),xticklabels=labels,ylabel="Circular phase error",title="Predefined fresh-preparation phase recoveries")
            ax.tick_params(axis="x",rotation=45);ax.legend();fig.tight_layout();fig.savefig(staging/"recovery.svg");plt.close(fig)
        if scaling:
            rows=[]
            for n in (1,2,4,8):
                for L in (3,6,9):
                    M=shots_for_failure(L,Fraction(1,100))
                    ledger=protocol_ledger(n,L,M)
                    rows.append({"target_qubits":n,"levels":L,"shots_per_fringe":M,"logical_qubits":n+1,
                        "phase_radius_exact":ledger["ideal_phase_radius_exact"],"maximum_controlled_power":ledger["maximum_controlled_U_power"],
                        "complete_fresh_trials":ledger["complete_fresh_preparations"],"weighted_base_U_calls":ledger["shot_weighted_controlled_U_calls"],
                        "P_gates":ledger["source_gate_counts_all_trials"]["p"],"CX_gates":ledger["source_gate_counts_all_trials"]["cx"],
                        "resets":ledger["source_gate_counts_all_trials"]["reset"],"measurements":ledger["source_gate_counts_all_trials"]["measure"],
                        "circuit_materialized":False,"dense_check_performed":False,"scope":"analytic source-construction counts; not a simulated/physical size result"})
            with (staging/"scaling.csv").open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=rows[0].keys(),lineterminator="\n");writer.writeheader();writer.writerows(rows)
            representative=build_experiment(8,.3,32,"sin")
            large=export(staging/"circuits/large-n8-sin-k32.qasm",representative)
            large["file"]="circuits/"+large["file"]
            large.update({"circuit_materialized":True,"simulation_performed":False,"reason":"outside separate three-data-wire dense cap"})
            result["larger_materialized_example"]=large
        result["cpu_wall_seconds"]=time.perf_counter()-start
        result["cpu_timing_scope"]="function entry through circuit checks, samples, plots and scaling; excludes final record serialization/publication copy"
        dump(staging/"results.json",result)
        summary=["| Nominal phase | Seed | Recovered phase | Circular error | Status |", "|---:|---:|---:|---:|---|"]
        for r in runs:
            current=next(c for c in cases if c["case_id"]==r["case_id"])
            value=r["result"]["phase_estimate"];error=r["circular_error_vs_actual_angle_reference"]
            summary.append(f'| {current["nominal_phase"]:.6g} | {r["seed"]} | {"unavailable" if value is None else f"{value:.9f}"} | {"unavailable" if error is None else f"{error:.3g}"} | {r["result"]["status"]} |')
        (staging/"summary.md").write_text("\n".join(summary)+"\n")
        dump(staging/"logical-resources.json",{"per_output":common_ledger,"whole_study":result["whole_study_modeled_workload"],
             "classical_cpu_seconds":sum(r["classical_sampling_and_recovery_seconds"] for r in runs),
             "physical_estimate_performed":False,"synthesis_performed":False})
        dump(staging/"software.json",{"original_implementation_license":"MIT","packages":[{"name":"Qiskit","version":importlib.metadata.version("qiskit"),"license":"Apache-2.0","source":"https://github.com/Qiskit/qiskit/tree/2.6.2"},
            {"name":"NumPy","version":importlib.metadata.version("numpy"),"license":"BSD-3-Clause","source":"https://github.com/numpy/numpy/tree/v2.5.4"},
            {"name":"Matplotlib","version":importlib.metadata.version("matplotlib"),"license":"Matplotlib license","source":"https://github.com/matplotlib/matplotlib/tree/v3.11.2"}]})
        (staging/"README.md").write_text("# Hadamard phase estimation examples\n\nOriginal MIT circuits, bounded numerical dyadic recovery and independent matrix/probability checks. Use the included Python3.12 requirements lock. Run the copied generator with --single --phase 0.3 --levels 6 and an explicit output directory. Default canonical study uses 48 circuit types and 47,376 modeled fresh trials. No Native estimation, hardware execution, synthesis certificate or Lean proof is included. See results.json for exact source hashes and ideal versus numerical confidence scope.\n")
        inventory={p.relative_to(staging).as_posix():digest(p.read_bytes()) for p in staging.rglob("*") if p.is_file()}
        dump(staging/"manifest.json",{"schema":"openquantum-hadamard-qpe-files-v1","source_sha256":source_hashes,"file_sha256":inventory,
                "record_status":"executed_bounded_numerical_demo" if simulate else "generated_without_simulation"})
        checkpoint(start,staging)
        output.mkdir(parents=True,exist_ok=True)
        for old in old_owned - (set(inventory) | {"manifest.json"}):
            path=output/old
            if path.is_symlink() or not path.is_file():
                raise ValueError("Owned output changed before publication")
            path.unlink()
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                dest=output/path.relative_to(staging);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single",action="store_true",help="Use one supplied phase/seed instead of the frozen study")
    parser.add_argument("--target-qubits",type=int,default=1)
    parser.add_argument("--phase",type=float,default=.3)
    parser.add_argument("--levels",type=int,default=6)
    parser.add_argument("--failure",type=Fraction,default=Fraction(1,100))
    parser.add_argument("--seed",type=int,default=7)
    parser.add_argument("--no-sim",action="store_true")
    parser.add_argument("--no-scaling",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"assets/generated/hadamard-qpe")
    args=parser.parse_args()
    result=generate(args.output_dir,target_qubits=args.target_qubits,phase=args.phase,levels=args.levels,
                    failure_target=args.failure,seed=args.seed,canonical_study=not args.single,simulate=not args.no_sim,scaling=not args.no_scaling)
    print(json.dumps({"status":"executed_bounded_numerical_demo" if not args.no_sim else "generated_without_simulation",
          "circuits":result["simulation"]["unique_experiment_circuits"],"modeled_fresh_trials":result["simulation"]["modeled_fresh_trials_sampled"],
          "wall_seconds":result["cpu_wall_seconds"],"physical_estimate_performed":False},sort_keys=True))


if __name__=="__main__":
    main()
