#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay saved Hadamard-QPE gate/count evidence without resampling.

Authenticate the saved producer snapshots and exact inventory before loading the
current implementation. The allowed current-module difference is one explicit
binary64-halving/underflow rejection guard; all canonical inputs bypass it.
Historical producer source/counts/timings remain separate from this replay.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib.util
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64 * 1024 * 1024
PRODUCER_COMMIT = "def275c501c8c55a32e219674000b4199eaa16d4"
PRODUCER_MANIFEST_SHA256 = "9e451d29d041746701686fe9ae7e17304fefabc61702a9f51e20d8ead10a2776"
OLD_ANGLE_BODY = "    return (2 * math.pi * _phase(phase)) / target_qubits"
NEW_ANGLE_BODY = '''    value = _phase(phase)
    angle = (2 * math.pi * value) / target_qubits
    if (value != 0 and angle == 0) or 2 * (angle / 2) != angle:
        raise ValueError("Stored phase component must be nonzero and exactly binary64-halvable, unless phase is zero")
    return angle'''


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read(path: Path, maximum: int = MAX_BYTES) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("A bounded regular record/source file is required")
    return path.read_bytes()


def object_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def parse(path: Path):
    return json.loads(read(path), object_pairs_hook=object_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def close(a, b, tolerance=2e-12):
    if type(a) is not float or type(b) is not float or not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
        raise ValueError("Saved/replayed bounded numerical value differs")


def compare_recovery(saved: dict, replayed: dict) -> None:
    # Only angle/phase fields depend on host libm. Counts, powers, lift choices,
    # exact confidence allocations and all nonnumerical fields remain exact.
    a = json.loads(json.dumps(saved)); b = json.loads(json.dumps(replayed))
    close(a.pop("phase_estimate"), b.pop("phase_estimate"))
    if len(a["trace"]) != len(b["trace"]):
        raise ValueError("Phase recovery trace truncated")
    for x, y in zip(a["trace"], b["trace"], strict=True):
        for name in ("angle_cycles", "phase_estimate"):
            close(x.pop(name), y.pop(name))
    if a != b:
        raise ValueError("Saved phase counts/lifts/contract differ")


def verify_record(record_dir: Path, current_source: Path | None = None) -> dict:
    start = time.perf_counter()
    record_dir = record_dir.absolute()
    if sha(read(record_dir/"manifest.json")) != PRODUCER_MANIFEST_SHA256:
        raise ValueError("Immutable historical producer manifest changed")
    manifest = parse(record_dir/"manifest.json")
    if manifest.get("schema") != "openquantum-hadamard-qpe-files-v1" or manifest.get("record_status") != "executed_bounded_numerical_demo":
        raise ValueError("Executed Hadamard record manifest required")
    inventory = manifest["file_sha256"]
    if not isinstance(inventory, dict) or len(inventory) > 100:
        raise ValueError("Bounded complete inventory required")
    actual = {p.relative_to(record_dir).as_posix() for p in record_dir.rglob("*") if p.is_file()}
    if actual != set(inventory)|{"manifest.json"}:
        raise ValueError("Saved Hadamard inventory changed")
    total = 0
    for name, expected in inventory.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Record path escaped inventory")
        raw = read(record_dir/relative); total += len(raw)
        if sha(raw) != expected or total > MAX_BYTES:
            raise ValueError("Saved file hash or complete byte cap changed")
    record = parse(record_dir/"results.json")
    producer = record["provenance"]
    if producer["source_commit"] != PRODUCER_COMMIT or producer["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("Historical producer identity changed")
    for name, expected in producer["source_sha256"].items():
        if sha(read(record_dir/name)) != expected:
            raise ValueError("Producer source snapshot changed")
    if (record["parameters"] != {"target_qubits":1,"levels":6,"phases":[.3,.995,.125,.5],"seeds":[7,19,41],"canonical_study":True,
          "total_ideal_failure_target_exact":"1/100","per_run_ideal_failure_target_exact":"1/1200","shots_per_fringe":329}
            or record["simulation"]["modeled_fresh_trials_sampled"] != 47376
            or record["simulation"]["unique_experiment_circuits"] != 48
            or record["physical_estimate_performed"] is not False or record["synthesis_performed"] is not False):
        raise ValueError("Canonical task/count/physical scope changed")
    current_source = current_source or ROOT/"src/openquantum/hadamard_qpe.py"
    current = read(current_source)
    historical = read(record_dir/"hadamard_qpe.py")
    if historical.count(OLD_ANGLE_BODY.encode()) != 1 or current != historical.replace(OLD_ANGLE_BODY.encode(),NEW_ANGLE_BODY.encode()):
        raise ValueError("Current implementation differs beyond the reviewed halving guard")
    lock = ROOT/"requirements-lock.txt"
    if lock.is_file() and sha(read(lock)) != producer["source_sha256"]["requirements-lock.txt"]:
        raise ValueError("Current environment lock differs from producer")
    import platform
    if platform.python_version() != record["environment"]["python"] or any(importlib.metadata.version(p) != v for p,v in record["environment"]["packages"].items()):
        raise ValueError("Current pinned environment differs from producer")
    # No .pyc/search-path authority: execute the authenticated current bytes.
    name="authenticated_hadamard_replay"
    spec=importlib.util.spec_from_file_location(name,current_source)
    module=importlib.util.module_from_spec(spec)
    previous=sys.modules.get(name);sys.modules[name]=module
    try:
        exec(compile(current,str(current_source),"exec"),module.__dict__)
        module.sample_counts=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError("Resampling is forbidden in saved replay"))
        import qiskit.qasm3 as qasm3
        ledger=module.protocol_ledger(1,6,329)
        if ledger != record["common_per_output_ledger"]:
            raise ValueError("Per-output power/shot/resource ledger changed")
        large=record["larger_materialized_example"]
        cases=record["cases"]
        if len(cases)!=4 or len(record["runs"])!=12:
            raise ValueError("Canonical case/run coverage incomplete")
        by_case={}
        for case_index,case in enumerate(cases):
            phase=record["parameters"]["phases"][case_index]
            if case["case_id"] != f"phase-{case_index}" or case["nominal_phase"]!=phase or len(case["experiments"])!=12:
                raise ValueError("Canonical phase/experiment order changed")
            # Normal values, exact halving: new rejection branch cannot fire.
            theta=module.base_phase_angle(1,phase)
            if theta < 2*sys.float_info.min or 2*(theta/2)!=theta:
                raise ValueError("Canonical input does not bypass the new guard")
            close(case["actual_angle_phase_reference"],module.actual_angle_phase(1,phase))
            for j,entry in enumerate(case["experiments"]):
                k=1<<(j//2);basis="cos" if j%2==0 else "sin"
                if (entry["power"],entry["basis"])!=(k,basis):
                    raise ValueError("Canonical power/basis schedule changed")
                circuit=module.build_experiment(1,phase,k,basis)
                lowered=module.lower_experiment(circuit)
                raw=qasm3.dumps(lowered).encode()
                if raw!=read(record_dir/entry["file"]) or sha(raw)!=entry["sha256"]:
                    raise ValueError("Current primitive trial/readout QASM differs")
                if entry["source_gate_counts"]!=dict(circuit.count_ops()) or entry["lowered_gate_counts"]!=dict(lowered.count_ops()) or entry["logical_qubits"]!=2:
                    raise ValueError("Circuit counts/register map changed")
                if entry["source_dependency_depth"]!=circuit.depth() or entry["lowered_dependency_depth"]!=lowered.depth():
                    raise ValueError("Circuit dependency depth changed")
                close(entry["lowered_global_phase_radians"],float(lowered.global_phase))
                check=module.check_experiment(1,phase,k,basis)
                old=entry["check"]
                if set(old)!=set(check) or old["performed"] is not True or old["certified"] is not False:
                    raise ValueError("Saved check schema/status changed")
                for key in ("zero_probability","analytic_zero_probability"):
                    close(old[key],check[key])
                for key in ("probability_absolute_error","full_operator_max_error","source_to_lowered_operator_max_error"):
                    if type(old[key]) is not float or not math.isfinite(old[key]) or not 0<=old[key]<=2e-11:
                        raise ValueError("Historical numerical residual exceeds its producer guard")
                p=min(1.0,max(0.0,check["zero_probability"]))
                close(entry["sampling_zero_probability"],p)
                if type(entry["sampling_probability_clipping_absolute_error"]) is not float or not 0<=entry["sampling_probability_clipping_absolute_error"]<=2e-11:
                    raise ValueError("Historical sampling-probability roundoff changed")
            by_case[case["case_id"]]=case
        expected_keys={(f"phase-{i}",seed) for i in range(4) for seed in (7,19,41)}
        if {(r["case_id"],r["seed"]) for r in record["runs"]}!=expected_keys:
            raise ValueError("Saved run/seed inventory differs")
        for run in record["runs"]:
            case=by_case[run["case_id"]]
            if run["stream"]!=int(run["case_id"].split("-")[-1]):
                raise ValueError("Case-separated count stream changed")
            counts=[module.FringeCounts(**c) for c in run["result"]["counts"]]
            if any(c.shots!=329 for c in counts):
                raise ValueError("Saved complete shot allocation changed")
            replayed=module.recover_phase(counts)
            compare_recovery(run["result"],replayed)
            error=module.circular_distance(replayed["phase_estimate"],case["actual_angle_phase_reference"])
            close(run["circular_error_vs_actual_angle_reference"],error)
            close(run["circular_error_vs_nominal_phase"],module.circular_distance(replayed["phase_estimate"],case["nominal_phase"]))
            if not 0<=run["classical_sampling_and_recovery_seconds"]<180:
                raise ValueError("Historical timing is invalid")
        whole={"outputs":12,"complete_fresh_preparations":12*ledger["complete_fresh_preparations"],
               "shot_weighted_controlled_U_calls":12*ledger["shot_weighted_controlled_U_calls"],
               "source_gate_counts":{g:12*c for g,c in ledger["source_gate_counts_all_trials"].items()},"workload_status":"sampled ideal counts"}
        if record["whole_study_modeled_workload"]!=whole:
            raise ValueError("Complete historical resource workload differs")
        confidence=record["ideal_confidence"]
        if (confidence["phase_radius_exact"]!=ledger["ideal_phase_radius_exact"]
                or confidence["per_run_failure_upper_bound_exact"]!=str(module.ideal_failure_bound(6,329))
                or confidence["whole_predeclared_study_failure_upper_bound_exact"]!=str(12*module.ideal_failure_bound(6,329))
                or confidence["numerical_inference_certified"] is not False or confidence["hardware_confidence_certified"] is not False):
            raise ValueError("Frozen ideal confidence/certification scope changed")
        large_circuit=module.build_experiment(8,.3,32,"sin");large_lowered=module.lower_experiment(large_circuit)
        if (qasm3.dumps(large_lowered).encode()!=read(record_dir/large["file"])
                or large["source_gate_counts"]!=dict(large_circuit.count_ops()) or large["simulation_performed"] is not False):
            raise ValueError("Larger materialized/unexecuted record changed")
        with (record_dir/"scaling.csv").open(newline="") as stream:
            rows=list(csv.DictReader(stream))
        if len(rows)!=12:
            raise ValueError("Analytic source-scaling inventory incomplete")
        for row in rows:
            n=int(row["target_qubits"]);L=int(row["levels"]);m=module.shots_for_failure(L,Fraction(1,100));g=module.protocol_ledger(n,L,m)
            expected={"shots_per_fringe":m,"logical_qubits":n+1,"maximum_controlled_power":g["maximum_controlled_U_power"],
                      "complete_fresh_trials":g["complete_fresh_preparations"],"weighted_base_U_calls":g["shot_weighted_controlled_U_calls"],
                      "P_gates":g["source_gate_counts_all_trials"]["p"],"CX_gates":g["source_gate_counts_all_trials"]["cx"],
                      "resets":g["source_gate_counts_all_trials"]["reset"],"measurements":g["source_gate_counts_all_trials"]["measure"]}
            if n not in (1,2,4,8) or L not in (3,6,9) or any(int(row[k])!=v for k,v in expected.items()) or row["phase_radius_exact"]!=g["ideal_phase_radius_exact"] or row["circuit_materialized"]!="False" or row["dense_check_performed"]!="False":
                raise ValueError("Source scaling counts/status changed")
    finally:
        if previous is None:sys.modules.pop(name,None)
        else:sys.modules[name]=previous
    try:
        revision=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,timeout=5)
        current_commit=revision.stdout.strip() if revision.returncode==0 else None
    except (OSError,subprocess.TimeoutExpired):current_commit=None
    return {"status":"verified_saved_numerical_record_current_guard_replay","producer_source_commit":PRODUCER_COMMIT,
            "producer_source_sha256":producer["source_sha256"],"current_replay_commit":current_commit,
            "current_replay_source_sha256":{current_source.name:sha(current),Path(__file__).name:sha(read(Path(__file__)))},
            "source_delta_scope":"only rejection of nonexact subnormal binary64 halving/underflow; all canonical components normal and unchanged",
            "cases":4,"qasm_exports_reproduced":49,"numerical_experiment_checks":48,"saved_runs_recovered":12,"historical_sampled_fresh_trials":47376,
            "new_sampling_calls":0,"native_calls":0,"new_hardware_execution":False,"numerical_certificate":False,
            "immutable_record_bytes":total,"wall_seconds":time.perf_counter()-start}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-dir",type=Path,default=ROOT/"assets/generated/hadamard-qpe")
    parser.add_argument("--receipt",type=Path)
    args=parser.parse_args();result=verify_record(args.records_dir)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True,exist_ok=True);args.receipt.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))


if __name__=="__main__":main()
