#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Cost every measured group of an existing ideal finite-shot trajectory.

The physical layer is a conditional resource-model replay of the recorded
parameter path, not a prediction of a QPU optimizer's adaptive outcomes.
Every source circuit is lowered and numerically checked before exact QASM
deduplication. Different angles share a resource query only after their
cost-canonical raw and PSSPC traces match under identical model budgets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import struct
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

from time import perf_counter

import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openquantum.resource_estimation import (
    AUDITED_SOURCE_COMMIT,
    PINNED_QDK_VERSION,
    CircuitResourceBudget,
    SurfaceCodeProfile,
    choose_num_ts_per_rotation,
    estimate_circuit_resources,
    prepare_measured_circuit,
)

PROFILES = (
    SurfaceCodeProfile(name="frozen-p-1e-3", physical_error_rate=1e-3, gate_time_ns=100, measurement_time_ns=500),
    SurfaceCodeProfile(name="frozen-p-1e-4", physical_error_rate=1e-4, gate_time_ns=100, measurement_time_ns=500),
)
TASK_PHYSICAL_FAILURE_ALLOWANCE = 1e-3
SYNTHESIS_MODEL_ALLOWANCE = 1e-5
MAX_GROUP_BATCHES = 4096
EVALUATION_TIME_PROPERTY = "7"
FACTORY_ERROR_RATE_MAX_ULPS = 4
EVALUATION_TIME_AUDIT = {
    "property_id": 7, "name": "EVALUATION_TIME", "qdk_version": "1.34.0",
    "meaning": "OpenQASM import/evaluation CPU wall-time instrumentation in nanoseconds",
    "signature_treatment": "Only this instrumentation property is removed; full original JSON is retained.",
    "does_not_describe_quantum_runtime": True,
    "audited_source_commit": AUDITED_SOURCE_COMMIT,
    "sources": [
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/isa/property_keys.rs",
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qdk_package/qdk/qre/application/_openqasm.py",
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qdk_package/qdk/qre/interop/_qsharp.py",
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace/transforms/psspc.rs",
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace/transforms/lattice_surgery.rs",
        f"https://github.com/microsoft/qdk/blob/{AUDITED_SOURCE_COMMIT}/source/qre/src/trace.rs",
    ],
    "cost_independence_audit": "PSSPC cost/layout uses instructions and widths; the transforms copy properties. "
                               "Trace estimation copies EVALUATION_TIME as output instrumentation, not physical cost.",
    "source_line_audit": {"isa/property_keys.rs": "12-23,48-56", "interop/_qsharp.py": "68-74,118,130-145",
                          "trace.rs": "35-43,55-63,263-426", "psspc.rs": "68-155", "lattice_surgery.rs": "32-41"},
    "importer_model_boundary": "Default OpenQASMApplication builds a canonical Trace from LogicalCounts. "
                               "Its rotation blocks already abstract original circuit topology and continuous angles.",
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_recorded_implementation(data: dict, implementation_path: Path):
    """Load the exact source bytes used by the recorded trajectory."""
    implementation_path = Path(implementation_path)
    expected = data["provenance"]["implementation_sha256"]
    if sha_file(implementation_path) != expected:
        raise ValueError("Recorded optimization implementation hash does not match")
    spec = importlib.util.spec_from_file_location("recorded_optimization_trajectory", implementation_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def measurement_map(circuit) -> list[list[int]]:
    result = []
    measurement_started = False
    for item in circuit.data:
        name = item.operation.name
        if name == "measure":
            measurement_started = True
            result.append([circuit.find_bit(item.qubits[0]).index, circuit.find_bit(item.clbits[0]).index])
        elif measurement_started and name != "barrier":
            raise ValueError("Only final measurements are supported")
    if sorted(pair[0] for pair in result) != list(range(circuit.num_qubits)):
        raise ValueError("The source circuit must measure every qubit exactly once")
    return result


def lowering_equivalence(source, lowered, *, tolerance: float = 1e-10) -> dict:
    """Full tiny-system Operator check up to global phase and readout wiring."""
    source_map = measurement_map(source)
    lowered_map = measurement_map(lowered)
    if source_map != lowered_map or source.num_qubits != lowered.num_qubits:
        raise ValueError("Lowering changed final measurement wiring or width")
    if source.num_qubits > 4:
        raise ValueError("This recorded-trajectory Operator check is capped at four qubits")
    a = Operator(source.remove_final_measurements(inplace=False)).data
    b = Operator(lowered.remove_final_measurements(inplace=False)).data
    overlap = np.vdot(a.ravel(), b.ravel())
    if abs(overlap) == 0:
        raise ValueError("Source and lowered circuits have zero Operator overlap")
    phase = overlap / abs(overlap)
    error = float(np.max(np.abs(b - phase * a)))
    if error > tolerance or not Operator(a).equiv(Operator(b), atol=tolerance, rtol=0):
        raise ValueError("Source and lowered circuit Operators are not equivalent")
    return {"verified": True, "method": "full numerical Operator up to one global phase",
            "tolerance": tolerance, "max_elementwise_phase_aligned_error": error,
            "source_measurement_map": source_map, "lowered_measurement_map": lowered_map,
            "different_angle_unitaries_claimed_equal": False}


def source_circuit_for_group(implementation, algorithm: str, parameters: list[float], group: str):
    if algorithm == "qaoa":
        if len(parameters) != 2 or group != "all_z":
            raise ValueError("Unexpected QAOA parameters or group")
        return implementation.build_qaoa_circuit(4, implementation.CYCLE_EDGES,
                                                  [parameters[0]], [parameters[1]], measure=True)
    if algorithm == "vqe":
        if len(parameters) != 1 or group not in {"all_z", "all_x"}:
            raise ValueError("Unexpected VQE parameters or group")
        return implementation.build_vqe_circuit(parameters[0], basis=group.removeprefix("all_"), measure=True)
    raise ValueError("Unsupported algorithm")


def build_execution_ledger(data: dict, *, implementation) -> dict:
    """Validate every recorded call/group and reconstruct/lower every batch."""
    if data["exact_diagnostics_used_for_optimization"] is not False:
        raise ValueError("Input must be the genuine sampled trajectory")
    if data["qaoa"]["n_qubits"] != 4 or sorted(map(tuple, data["qaoa"]["edges"])) != sorted(implementation.CYCLE_EDGES):
        raise ValueError("Expected the recorded four-vertex MaxCut cycle")
    if data["vqe"]["n_qubits"] != 2 or data["vqe"]["hamiltonian"] != "-Z0-Z1+0.5 X0X1":
        raise ValueError("Expected the recorded two-qubit Hamiltonian")
    mapping, qasm_records, streams = [], {}, set()
    task_totals, weighted_counts, control_times = {}, {}, {}
    training_shots = validation_shots = total_calls = 0
    for algorithm_id, algorithm in enumerate(("qaoa", "vqe")):
        task_shots, task_calls = 0, 0
        counts = Counter()
        runs = data[algorithm]["runs"]
        if not 1 <= len(runs) <= 16:
            raise ValueError("Recorded replay supports 1..16 runs per task")
        for run_index, run in enumerate(runs):
            if run["run_index"] != run_index or run["objective_evaluations"] != len(run["evaluations"]):
                raise ValueError("Missing or duplicated objective calls/run indexes")
            run_training = run_validation = 0
            for phase, family, calls in (("training", 0, run["evaluations"]), ("validation", 1, [run["validation"]])):
                for call_index, call in enumerate(calls):
                    if phase == "training" and call["index"] != call_index:
                        raise ValueError("Objective indexes must enumerate every call")
                    if phase == "validation" and (call["parameters"] != run["parameters"] or call["role"] != "independent_final_validation"):
                        raise ValueError("Final validation does not match the recorded final state")
                    parameters = [float(x) for x in call["parameters"]]
                    if not all(math.isfinite(x) for x in parameters):
                        raise ValueError("Recorded angles must be finite")
                    expected_groups = ["all_z"] if algorithm == "qaoa" else ["all_z", "all_x"]
                    if [g["group"] for g in call["groups"]] != expected_groups:
                        raise ValueError("Recorded measurement groups do not match the task")
                    if call["total_shots"] != sum(g["shots"] for g in call["groups"]):
                        raise ValueError("Objective shot total differs from its group allocations")
                    task_calls += 1
                    for group_index, group in enumerate(call["groups"]):
                        shots = group["shots"]
                        if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 65536:
                            raise ValueError("Invalid recorded group shot allocation")
                        expected_stream = [run["seed"], algorithm_id, family, call_index, group_index]
                        if group["rng"]["stream_key"] != expected_stream or tuple(expected_stream) in streams:
                            raise ValueError("Missing, duplicated or incorrect recorded RNG stream")
                        if group["rng"]["entropy"] != expected_stream[:2] or group["rng"]["spawn_key"] != expected_stream[2:]:
                            raise ValueError("RNG lineage differs from the recorded stream")
                        streams.add(tuple(expected_stream))
                        if sum(group["counts"].values()) != shots:
                            raise ValueError("Recorded counts do not account for every shot")
                        width = 4 if algorithm == "qaoa" else 2
                        if any(len(bits) != width or set(bits) - {"0", "1"}
                               or isinstance(number, bool) or not isinstance(number, int) or number <= 0
                               for bits, number in group["counts"].items()):
                            raise ValueError("Invalid recorded bitstring counts")
                        source = source_circuit_for_group(implementation, algorithm, parameters, group["group"])
                        if (group["logical_qubits"] != source.num_qubits or group["circuit_gate_counts"] != dict(source.count_ops())
                                or group["circuit_depth"] != source.depth()):
                            raise ValueError("Recorded source gate/width/depth metadata does not match reconstruction")
                        lowered = prepare_measured_circuit(source)
                        equivalence = lowering_equivalence(source, lowered)
                        # Exact text deduplication happens only after this batch's
                        # full Operator and measurement-map checks have succeeded.
                        source_qasm, lowered_qasm = qasm3.dumps(source), qasm3.dumps(lowered)
                        source_sha, lowered_sha = sha_text(source_qasm), sha_text(lowered_qasm)
                        for kind, text, digest, circuit in (("source", source_qasm, source_sha, source),
                                                           ("lowered", lowered_qasm, lowered_sha, lowered)):
                            key = kind + ":" + digest
                            if key in qasm_records and qasm_records[key]["qasm"] != text:
                                raise RuntimeError("Exact-QASM hash collision")
                            qasm_records.setdefault(key, {"kind": kind, "sha256": digest, "qasm": text,
                                                          "artifact": f"{kind}-qasm/{digest}.qasm",
                                                          "logical_qubits": circuit.num_qubits,
                                                          "gate_counts": dict(circuit.count_ops()), "depth": circuit.depth()})
                        batch_id = f"{algorithm}.run{run_index}.{phase}.call{call_index}.{group['group']}"
                        mapping.append({"batch_id": batch_id, "task": algorithm, "run_index": run_index,
                                        "seed": run["seed"], "phase": phase, "evaluation_index": call_index,
                                        "call_role": call["role"], "proposal_accepted": call.get("accepted", False),
                                        "group": group["group"], "parameters": parameters, "shots": shots,
                                        "rng": group["rng"], "source_qasm_sha256": source_sha,
                                        "lowered_qasm_sha256": lowered_sha, "lowering_equivalence": equivalence,
                                        "source_gate_counts": dict(source.count_ops()), "source_depth": source.depth(),
                                        "lowered_gate_counts": dict(lowered.count_ops()), "lowered_depth": lowered.depth()})
                        if len(mapping) > MAX_GROUP_BATCHES:
                            raise ValueError("Replay exceeds the 4096 measured-group batch bound")
                        task_shots += shots
                        if phase == "training":
                            run_training += shots
                        else:
                            run_validation += shots
                        counts.update({gate: number * shots for gate, number in source.count_ops().items() if gate != "barrier"})
            if run_training != run["total_shots"] or run_training + run_validation != run["total_shots_including_validation"]:
                raise ValueError("Run totals omit or duplicate group shots")
            training_shots += run_training
            validation_shots += run_validation
        task_totals[algorithm] = task_shots
        weighted_counts[algorithm] = dict(counts)
        control_times[algorithm] = {
            "recorded_optimizer_cpu_wall_seconds": sum(r["optimizer_wall_seconds"] for r in runs),
            "recorded_sampled_objective_cpu_seconds": sum(r["objective_wall_seconds"] for r in runs),
            "recorded_classical_control_cpu_seconds": sum(r["classical_control_seconds"] for r in runs),
            "hardware_feedback_or_transport_seconds": None,
            "not_added_to_modeled_quantum_runtime": True,
        }
        total_calls += task_calls
    if (training_shots != data["training_shots"] or validation_shots != data["validation_shots"]
            or sum(task_totals.values()) != data["total_shots"]):
        raise ValueError("Dataset totals omit or duplicate recorded shots")
    return {"mapping": mapping, "qasm_records": qasm_records, "task_totals": task_totals,
            "weighted_source_gate_counts": weighted_counts, "recorded_cpu_times": control_times,
            "objective_and_validation_calls": total_calls, "measured_group_batches": len(mapping),
            "total_shots": sum(task_totals.values()), "training_shots": training_shots,
            "validation_shots": validation_shots, "unique_source_qasm": sum(r["kind"] == "source" for r in qasm_records.values()),
            "unique_lowered_qasm": sum(r["kind"] == "lowered" for r in qasm_records.values())}


def cost_canonical_trace_json(native_json: dict) -> dict:
    """Keep all native fields except documented non-cost import timing."""
    result = json.loads(canonical_json(native_json))
    properties = result.get("properties")
    if isinstance(properties, dict):
        properties.pop(EVALUATION_TIME_PROPERTY, None)
    return result


def cost_signature_key(snapshot: dict, config: dict) -> str:
    return sha_text(canonical_json({"cost_snapshot": snapshot, "model_config": config}))


def trace_snapshot(trace) -> dict:
    native = json.loads(trace.to_json())
    restored = type(trace).from_json(canonical_json(native))
    if json.loads(restored.to_json()) != native:
        raise RuntimeError("Native Trace JSON did not round-trip exactly")
    return {"full_native_json": native, "cost_canonical_json": cost_canonical_trace_json(native),
            "compute_qubits": trace.compute_qubits, "memory_qubits": trace.memory_qubits,
            "total_qubits": trace.total_qubits, "depth": trace.depth, "num_gates": trace.num_gates,
            "gate_counts": {str(k): int(v) for k, v in trace.gate_counts.items()},
            "resource_states": {str(k): int(v) for k, v in trace.resource_states.items()},
            "base_error": float(trace.base_error), "native_json_roundtrip_verified": True}


def canonical_cost_signature(lowered_qasm: str, *, task: str, task_shots: int,
                             profile: SurfaceCodeProfile,
                             physical_failure_allowance: float = TASK_PHYSICAL_FAILURE_ALLOWANCE,
                             synthesis_allowance: float = SYNTHESIS_MODEL_ALLOWANCE) -> dict:
    from qdk.qre import PSSPC, property_name
    from qdk.qre.application import OpenQASMApplication
    from qdk.qre.instruction_ids import RZ
    from qdk.qre.property_keys import EVALUATION_TIME
    if importlib.metadata.version("qdk") != PINNED_QDK_VERSION or EVALUATION_TIME != 7 or property_name(EVALUATION_TIME) != "EVALUATION_TIME":
        raise RuntimeError("Pinned timing-instrumentation audit does not apply")
    if task not in {"qaoa", "vqe"} or isinstance(task_shots, bool) or not isinstance(task_shots, int) or task_shots <= 0:
        raise ValueError("Invalid task shot total")
    if not 0 < physical_failure_allowance < 1 or not 0 < synthesis_allowance < 1:
        raise ValueError("Explicit positive physical and synthesis-model allowances are required")
    traces = list(OpenQASMApplication(lowered_qasm).enumerate_traces())
    if len(traces) != 1:
        raise ValueError("Replay model requires exactly one static canonical application trace")
    raw = traces[0]
    if raw.base_error != 0:
        raise ValueError("Unexpected pre-existing canonical base error")
    rotations = int(raw.gate_counts.get(RZ, 0))
    num_ts = choose_num_ts_per_rotation(rotations, synthesis_allowance)
    transform = PSSPC(num_ts_per_rotation=num_ts)
    transformed = transform.transform(raw)
    if transformed is None:
        raise ValueError("PSSPC cannot transform this canonical trace")
    raw_snapshot, psspc_snapshot = trace_snapshot(raw), trace_snapshot(transformed)
    config = {"qdk_version": PINNED_QDK_VERSION, "task": task, "task_total_shots": task_shots,
              "profile": asdict(profile), "synthesis_model_allowance_per_execution": synthesis_allowance,
              "task_physical_failure_allowance": physical_failure_allowance,
              "nominal_physical_failure_allowance_per_execution": physical_failure_allowance / task_shots,
              "psspc_num_ts_per_rotation": num_ts, "psspc_ccx_magic_states": transform.ccx_magic_states,
              "isa_graph_pruning": False, "composition": "UnionBound"}
    cost_snapshot = {name: {key: value for key, value in snapshot.items() if key != "full_native_json"}
                     for name, snapshot in (("canonical_raw_trace", raw_snapshot), ("psspc_transformed_trace", psspc_snapshot))}
    return {"id": cost_signature_key(cost_snapshot, config), "cost_snapshot": cost_snapshot,
            "model_config": config, "raw_trace": raw_snapshot, "psspc_trace": psspc_snapshot,
            "num_arbitrary_rotations": rotations, "synthesis_model_base_error_per_execution": float(transformed.base_error),
            "instrumentation_audit": EVALUATION_TIME_AUDIT,
            "claim": "Exact cost-canonical trace/model equivalence only; not equality of different-angle unitaries or full instrumented JSON."}


def allocate_signature_budgets(ledger: dict, signatures: dict) -> dict:
    """Allocate one uniform per-preparation task fault allowance across replay."""
    result = {}
    mapping = {m["batch_id"]: m for m in ledger["mapping"]}
    if len(mapping) != len(ledger["mapping"]):
        raise ValueError("Execution ledger contains duplicate batch IDs")
    profile_names = {profile.name for profile in PROFILES}
    for entry in mapping.values():
        if (isinstance(entry["shots"], bool) or not isinstance(entry["shots"], int) or entry["shots"] <= 0
                or set(entry.get("cost_signature_by_profile", {})) != profile_names):
            raise ValueError("Every batch needs positive shots and one explicit signature for each profile")
    for task, shots in ledger["task_totals"].items():
        if shots != sum(m["shots"] for m in mapping.values() if m["task"] == task):
            raise ValueError("Ledger task shot totals are stale")
    for signature_id, signature in signatures.items():
        members = signature["batch_ids"]
        if len(set(members)) != len(members) or any(member not in mapping for member in members):
            raise ValueError("Missing or duplicated signature batch mapping")
        config = signature["model_config"]
        task = config["task"]
        if any(mapping[m]["task"] != task for m in members):
            raise ValueError("A signature cannot mix task allocations")
        executions = sum(mapping[m]["shots"] for m in members)
        whole = ledger["task_totals"][task]
        if (config["task_total_shots"] != whole or config["task_physical_failure_allowance"] != TASK_PHYSICAL_FAILURE_ALLOWANCE
                or config["synthesis_model_allowance_per_execution"] != SYNTHESIS_MODEL_ALLOWANCE
                or not math.isclose(config["nominal_physical_failure_allowance_per_execution"],
                                    TASK_PHYSICAL_FAILURE_ALLOWANCE / whole, rel_tol=1e-14, abs_tol=0)):
            raise ValueError("Signature model allocation is stale or differs from the whole task contract")
        if config["profile"] not in [asdict(profile) for profile in PROFILES]:
            raise ValueError("Signature hardware profile differs from the frozen profiles")
        profile_name = config["profile"]["name"]
        mapped_members = {m["batch_id"] for m in mapping.values()
                          if m["cost_signature_by_profile"][profile_name] == signature_id}
        if mapped_members != set(members):
            raise ValueError("Per-batch profile signature mapping differs from signature membership")
        allowance = TASK_PHYSICAL_FAILURE_ALLOWANCE * executions / whole
        budget = CircuitResourceBudget(synthesis_model_error_per_execution=SYNTHESIS_MODEL_ALLOWANCE,
                                      physical_failure_probability=allowance, executions=executions)
        nominal = TASK_PHYSICAL_FAILURE_ALLOWANCE / whole
        if not math.isclose(budget.physical_failure_per_execution, nominal, rel_tol=1e-14, abs_tol=0):
            raise RuntimeError("Grouped query changed the uniform task per-preparation allocation")
        result[signature_id] = {**asdict(budget), "nominal_physical_failure_per_execution": nominal,
                                "actual_query_physical_failure_per_execution": budget.physical_failure_per_execution}
    for profile in PROFILES:
        for task in ledger["task_totals"]:
            selected = [s for s in signatures.values() if s["model_config"]["task"] == task
                        and s["model_config"]["profile"]["name"] == profile.name]
            all_members = [m for s in selected for m in s["batch_ids"]]
            expected = [m["batch_id"] for m in ledger["mapping"] if m["task"] == task]
            if len(all_members) != len(set(all_members)) or set(all_members) != set(expected):
                raise ValueError("Signatures do not cover every task/profile batch exactly once")
            if sum(result[s["id"]]["executions"] for s in selected) != ledger["task_totals"][task]:
                raise ValueError("Signature execution totals omit or duplicate preparations")
            if not math.isclose(sum(result[s["id"]]["physical_failure_probability"] for s in selected),
                                TASK_PHYSICAL_FAILURE_ALLOWANCE, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError("Signature allocations do not sum to the whole task fault allowance")
    return result


def projection_cost_fingerprint(report: dict) -> dict:
    """Declared physical cost fields for the bounded empirical reuse probe."""
    points = []
    for point in report["pareto_points"]:
        copied = json.loads(canonical_json(point))
        # Native source graph is opaque provenance, not a numeric cost field;
        # each report retains it, while the complete PSSPC cost JSON is audited.
        copied.pop("instruction_source_graph", None)
        copied["model_properties"].pop("EVALUATION_TIME", None)
        points.append(copied)
    points.sort(key=canonical_json)
    return {"physical_profile": report["physical_profile"], "budget": report["budget"],
            "synthesis": report["synthesis"], "canonical_gate_counts": report["canonical_gate_counts"],
            "error_composition": report["error_composition"], "isa_graph_pruning": report["isa_graph_pruning"],
            "declared_physical_pareto_points": points}


def compare_projection_cost_fingerprints(representative: dict, alternate: dict) -> dict:
    """Compare complete costs, allowing only a bounded factory-rate roundoff.

    An independent fresh-cache probe observed a one-ULP difference in a
    positive native factory error rate, with every other cost field equal.
    That observation does not diagnose the earlier unrecorded development
    mismatch. No field is removed here. Structures, types, integer counts,
    runtimes, traces, profiles, budgets and all other floats remain exact.

    Only positive finite binary64 floats at a Pareto point's factory
    ``error_rate`` may differ, by at most four adjacent representable steps.
    Ordered bit distance handles binade boundaries without a relative-error
    shortcut. Both full input fingerprints and every differing leaf are
    retained by the caller; this is numeric agreement, not a new synthesis,
    unitary-equivalence or accuracy certificate.
    """
    differences = []

    def pointer(path):
        return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)

    def record(path, left, right, *, accepted=False, reason, ulp_distance=None, ulp_tolerance=None):
        differences.append({
            "path": pointer(path),
            "representative_value": left,
            "alternate_value": right,
            "accepted": accepted,
            "comparison": reason,
            "ulp_distance": ulp_distance,
            "ulp_tolerance": ulp_tolerance,
        })

    def factory_rate_path(path):
        return (len(path) == 5 and path[0] == "declared_physical_pareto_points"
                and type(path[1]) is int and path[2] == "factories"
                and type(path[3]) is str and path[4] == "error_rate")

    def visit(left, right, path):
        if factory_rate_path(path):
            if (type(left) is not float or type(right) is not float
                    or not math.isfinite(left) or not math.isfinite(right)
                    or left <= 0 or right <= 0):
                record(path, left, right, reason="factory error rates require positive finite binary64 floats")
                return
            if left != right:
                left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
                right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
                distance = abs(left_bits - right_bits)
                record(path, left, right, accepted=distance <= FACTORY_ERROR_RATE_MAX_ULPS,
                       reason="positive factory error-rate binary64 ULP comparison",
                       ulp_distance=distance, ulp_tolerance=FACTORY_ERROR_RATE_MAX_ULPS)
            return
        if type(left) is not type(right):
            record(path, left, right, reason="exact type mismatch")
        elif type(left) is dict:
            if (left.keys() != right.keys()
                    or any(type(key) is not str for key in left)
                    or any(type(key) is not str for key in right)):
                record(path, left, right, reason="exact JSON object keys required")
                return
            for key in sorted(left):
                visit(left[key], right[key], (*path, key))
        elif type(left) is list:
            if len(left) != len(right):
                record(path, left, right, reason="exact list length required")
                return
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                visit(a, b, (*path, index))
        elif type(left) is float:
            if not math.isfinite(left) or not math.isfinite(right):
                record(path, left, right, reason="nonfinite fingerprint value rejected")
            elif left != right:
                record(path, left, right, reason="exact non-factory float required")
        elif type(left) in (str, int, bool, type(None)):
            if left != right:
                record(path, left, right, reason="exact scalar value required")
        else:
            record(path, left, right, reason="unsupported non-JSON fingerprint type")

    visit(representative, alternate, ())
    matches = all(item["accepted"] for item in differences)
    return {"matches": matches, "exact_match": matches and not differences,
            "factory_error_rate_max_ulps": FACTORY_ERROR_RATE_MAX_ULPS,
            "approximate_field_selector": "/declared_physical_pareto_points/*/factories/*/error_rate",
            "all_other_fields_comparison": "exact JSON structure, types and finite values",
            "differences": differences}


def aggregate_replay_choices(signature_reports: dict, task_totals: dict) -> list[dict]:
    """Serial replay sums all execution times/faults and takes peak qubits."""
    rows = []
    for profile in PROFILES:
        for task, expected_executions in task_totals.items():
            reports = [r for r in signature_reports.values() if r["task"] == task and r["profile"]["name"] == profile.name]
            if not reports or sum(r["executions"] for r in reports) != expected_executions:
                raise ValueError("Physical reports omit or duplicate task preparations")
            for label, field in (("fewest qubits per cost signature", "peak_physical_qubits"),
                                 ("shortest runtime per cost signature", "runtime_per_execution_ns")):
                selections = []
                for entry in reports:
                    point = min(entry["report"]["pareto_points"],
                                key=lambda p: (p[field], p["runtime_per_execution_ns"], p["peak_physical_qubits"]))
                    selections.append({"signature_id": entry["signature_id"], "executions": entry["executions"], "point": point})
                fault = sum(s["point"]["physical_failure_model_per_execution"] * s["executions"] for s in selections)
                if fault > TASK_PHYSICAL_FAILURE_ALLOWANCE + 1e-12:
                    raise ValueError("Whole recorded replay exceeds the task physical-fault allocation")
                rows.append({"profile": profile.name, "physical_error_rate": profile.physical_error_rate,
                             "task": task, "choice": label, "modeled_preparations": expected_executions,
                             "signature_selections": selections,
                             "serial_quantum_runtime_ns": sum(s["point"]["runtime_per_execution_ns"] * s["executions"] for s in selections),
                             "peak_physical_qubits": max(s["point"]["peak_physical_qubits"] for s in selections),
                             "physical_failure_union_bound_all_preparations": fault,
                             "task_physical_failure_allowance": TASK_PHYSICAL_FAILURE_ALLOWANCE,
                             "synthesis_model_scalar_summed_over_shots": False,
                             "common_physical_configuration_verified": False,
                             "configuration_change_transport_reset_feedback_overhead_ns": None,
                             "not_a_complete_joint_pareto_frontier": True})
    return rows


def replay_summary(manifest: dict) -> str:
    lines = [
        "This is a **conditional resource-model replay of the recorded ideal finite-shot trajectory**. "
        "It costs all training, rejected proposals, incumbent remeasurements, and independent validation. "
        "It does not predict that a synthesized or noisy hardware optimizer would follow the same path or return the same outcomes.", "",
        f"The validated ledger contains {manifest['objective_and_validation_calls']:,} calls, "
        f"{manifest['measured_group_batches']:,} measured-group batches and {manifest['total_shots']:,} modeled preparations. "
        f"Every batch has a full small-circuit Operator check up to global phase and an unchanged final measurement map. "
        f"Only then are {manifest['unique_source_qasm']:,} exact source QASM and {manifest['unique_lowered_qasm']:,} exact lowered QASM artifacts deduplicated.", "",
        "Resource-query reuse is separate from exact-QASM deduplication. Cost-canonical raw and transformed PSSPC Trace JSON "
        "must match exactly in ordered instructions, repetitions, qubits, base error, resource states and all cost properties under "
        "the same profile and budgets. Only the documented QDK `EVALUATION_TIME` CPU-import instrumentation is omitted from the cost key; "
        "its complete original JSON is retained. Different-angle unitaries are not declared equal.", "",
        "The distinct-QASM physical-cost probe retains every fingerprint field. Only positive finite factory error-rate floats "
        "may agree within four adjacent binary64 values; all other values, types and structures remain exact. "
        "Both full fingerprints and every differing float path, raw value and ULP distance are preserved. "
        "This narrowly bounded numeric agreement is not a synthesis or accuracy certificate.", "",
    ]
    if manifest["status"] != "computed_conditional_recorded_replay":
        lines += ["The complete physical layer is **pending**: " + manifest["pending_reason"] + ". "
                  "No partial aggregate physical result is presented and no execution is silently omitted.", ""]
        return "\n".join(lines)
    lines += [
        f"There are {manifest['cost_signatures']:,} verified cost signatures and {manifest['real_qdk_query_count']:,} bounded real QDK queries, "
        "including a second distinct-QASM probe of one shared cost signature. "
        "The task physical-fault allowance is $10^{-3}$ uniformly across all preparations: "
        f"QAOA has {manifest['task_totals']['qaoa']:,} and VQE {manifest['task_totals']['vqe']:,} preparations. "
        "Each execution separately receives synthesis-model scalar allowance $10^{-5}$; that coherent model scalar is not summed "
        "over shots as a stochastic fault probability.", "",
        "| Task | Physical error | Declared serial choice | All modeled preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-task physical-fault score |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in manifest["declared_serial_choices"]:
        lines.append(f"| {row['task'].upper()} | {row['physical_error_rate']:g} | {row['choice']} | "
                     f"{row['modeled_preparations']:,} | {row['peak_physical_qubits']:,} | "
                     f"{row['serial_quantum_runtime_ns']/1e9:.6f} | {row['physical_failure_union_bound_all_preparations']:.3g} |")
    lines += ["", "Profiles retain physical error $10^{-3}$ or $10^{-4}$, 100 ns gates and 500 ns measurements. "
              "Times sum per-preparation model runtimes; peaks take the maximum across independently selected signature layouts/factories. "
              "Configuration changes, transport, resets and adaptive feedback are unmeasured, and no common physical configuration or joint Pareto frontier is certified.", "",
              "The recorded CPU controller/evaluator times are supplied separately in the manifest; they are not added to quantum-model runtime. "
              "The PSSPC rotation costs are a mean-fit projection, not concrete certified synthesis sequences, worst-case bounds, runtime-tail guarantees, "
              "or observable/optimizer accuracy guarantees. Statistical uncertainty, coherent synthesis-model allowance and stochastic physical-fault allocations remain distinct.", ""]
    return "\n".join(lines)


def generate(runs_path: Path, implementation_path: Path, output: Path, cache_dir: Path,
             *, max_queries: int = 32) -> dict:
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 1 <= max_queries <= 64:
        raise ValueError("Resource queries require an explicit bound in 1..64")
    started = perf_counter()
    runs_bytes = runs_path.read_bytes()
    data = json.loads(runs_bytes)
    implementation = load_recorded_implementation(data, implementation_path)
    ledger = build_execution_ledger(data, implementation=implementation)
    for kind in ("source", "lowered"):
        (output / f"{kind}-qasm").mkdir(parents=True, exist_ok=True)
    for entry in ledger["qasm_records"].values():
        (output / entry["artifact"]).write_text(entry["qasm"], encoding="utf-8")
    signatures, variant_traces, unsupported = {}, {}, []
    lowered_records = [r for r in ledger["qasm_records"].values() if r["kind"] == "lowered"]
    lowered_tasks = {m["lowered_qasm_sha256"]: m["task"] for m in ledger["mapping"]}
    for entry in lowered_records:
        digest, task = entry["sha256"], lowered_tasks[entry["sha256"]]
        for profile in PROFILES:
            try:
                signature = canonical_cost_signature(entry["qasm"], task=task, task_shots=ledger["task_totals"][task], profile=profile)
            except (ValueError, RuntimeError) as exc:
                unsupported.append({"lowered_qasm_sha256": digest, "profile": profile.name, "reason": str(exc)})
                continue
            variant_traces[digest + ":" + profile.name] = signature
            sid = signature["id"]
            if sid in signatures:
                if (signature["cost_snapshot"] != signatures[sid]["cost_snapshot"]
                        or signature["model_config"] != signatures[sid]["model_config"]):
                    raise RuntimeError("Cost signature hash collision")
            else:
                signatures[sid] = {**signature, "lowered_qasm_sha256s": [], "batch_ids": []}
            signatures[sid]["lowered_qasm_sha256s"].append(digest)
    for mapping in ledger["mapping"]:
        mapping["cost_signature_by_profile"] = {}
        for profile in PROFILES:
            variant = variant_traces.get(mapping["lowered_qasm_sha256"] + ":" + profile.name)
            if variant is not None:
                mapping["cost_signature_by_profile"][profile.name] = variant["id"]
                signatures[variant["id"]]["batch_ids"].append(mapping["batch_id"])
    budgets = allocate_signature_budgets(ledger, signatures) if not unsupported else {}
    probe_candidate = next((s for s in signatures.values() if len(s["lowered_qasm_sha256s"]) > 1), None)
    planned_queries = len(signatures) + (1 if probe_candidate else 0)
    pending = ("Unsupported canonical model traces; the ledger still contains every execution" if unsupported else
               f"Required {planned_queries} resource queries exceed the declared bound {max_queries}" if planned_queries > max_queries else None)
    reports, probe = {}, None
    if pending is None:
        for sid, signature in signatures.items():
            representative_sha = signature["lowered_qasm_sha256s"][0]
            representative_mapping = next(m for m in ledger["mapping"] if m["lowered_qasm_sha256"] == representative_sha)
            source = source_circuit_for_group(implementation, representative_mapping["task"],
                                             representative_mapping["parameters"], representative_mapping["group"])
            budget_dict = budgets[sid]
            budget = CircuitResourceBudget(**{k: budget_dict[k] for k in ("synthesis_model_error_per_execution", "physical_failure_probability", "executions")})
            profile = SurfaceCodeProfile(**signature["model_config"]["profile"])
            report = estimate_circuit_resources(source, budget=budget, profile=profile, cache_dir=cache_dir, use_graph=False)
            if report["qasm_sha256"] != representative_sha:
                raise RuntimeError("Estimated representative differs from the verified lowered QASM")
            if not math.isclose(report["synthesis"]["model_error_per_execution"], signature["synthesis_model_base_error_per_execution"], rel_tol=1e-12, abs_tol=1e-15):
                raise RuntimeError("Estimated PSSPC base scalar differs from the verified signature")
            reports[sid] = {"signature_id": sid, "task": signature["model_config"]["task"],
                            "profile": signature["model_config"]["profile"], "executions": budget.executions,
                            "budget_allocation": budget_dict, "representative_lowered_qasm_sha256": representative_sha,
                            "member_circuits_not_claimed_unitarily_identical": True, "report": report}
        if probe_candidate is not None:
            sid = probe_candidate["id"]
            alternate_sha = probe_candidate["lowered_qasm_sha256s"][1]
            alternate = next(m for m in ledger["mapping"] if m["lowered_qasm_sha256"] == alternate_sha)
            source = source_circuit_for_group(implementation, alternate["task"], alternate["parameters"], alternate["group"])
            budget_dict = budgets[sid]
            budget = CircuitResourceBudget(**{k: budget_dict[k] for k in ("synthesis_model_error_per_execution", "physical_failure_probability", "executions")})
            profile = SurfaceCodeProfile(**probe_candidate["model_config"]["profile"])
            probe_report = estimate_circuit_resources(source, budget=budget, profile=profile, cache_dir=cache_dir, use_graph=False)
            if probe_report["qasm_sha256"] != alternate_sha:
                raise RuntimeError("Empirical probe report did not estimate the alternate verified QASM")
            probe_signature = canonical_cost_signature(probe_report["qasm3"], task=alternate["task"],
                                                       task_shots=ledger["task_totals"][alternate["task"]], profile=profile)
            if (probe_signature["cost_snapshot"] != probe_candidate["cost_snapshot"]
                    or probe_signature["model_config"] != probe_candidate["model_config"]):
                raise RuntimeError("Probe's returned QASM is outside the verified cost signature")
            representative_fingerprint = projection_cost_fingerprint(reports[sid]["report"])
            alternate_fingerprint = projection_cost_fingerprint(probe_report)
            fingerprint_comparison = compare_projection_cost_fingerprints(representative_fingerprint, alternate_fingerprint)
            equivalent = fingerprint_comparison["matches"]
            probe = {"signature_id": sid, "representative_lowered_qasm_sha256": reports[sid]["representative_lowered_qasm_sha256"],
                     "alternate_lowered_qasm_sha256": alternate_sha, "distinct_qasm_verified": alternate_sha != reports[sid]["representative_lowered_qasm_sha256"],
                     "declared_physical_cost_fields_match": equivalent,
                     "declared_physical_cost_fields_exactly_equal": fingerprint_comparison["exact_match"],
                     "physical_cost_fingerprint_comparison": fingerprint_comparison,
                     "alternate_report_qasm_sha256_verified": True,
                     "alternate_report_reparsed_cost_signature_verified": True,
                     "alternate_report_full_raw_trace": probe_signature["raw_trace"],
                     "alternate_report_full_psspc_trace": probe_signature["psspc_trace"],
                     "representative_physical_cost_fingerprint": representative_fingerprint,
                     "alternate_physical_cost_fingerprint": alternate_fingerprint,
                     "probe_scope": "Bounded empirical consistency check, not a theorem about all native resource searches.",
                     "not_a_unitary_equivalence_claim_between_angles": True, "alternate_report": probe_report}
            if not equivalent:
                pending = "Distinct-QASM empirical model-equivalence probe did not match; no aggregate physical result released"
    choices = aggregate_replay_choices(reports, ledger["task_totals"]) if pending is None else []
    manifest = {k: value for k, value in ledger.items() if k != "qasm_records"}
    manifest.update({"schema_version": 1, "status": "computed_conditional_recorded_replay" if pending is None else "pending_physical_layer",
                     "resource_estimate_kind": "conditional_mean_fit_recorded_trajectory_replay",
                     "pending_reason": pending, "cost_signatures": len(signatures), "real_qdk_query_count": planned_queries if pending is None else len(reports) + (probe is not None),
                     "query_limit": max_queries, "declared_serial_choices": choices,
                     "source_provenance": {"sampled_runs_sha256": hashlib.sha256(runs_bytes).hexdigest(),
                                           "recorded_optimization_implementation_sha256": sha_file(implementation_path),
                                           "shared_resource_adapter_sha256": sha_file(ROOT / "src/openquantum/resource_estimation.py"),
                                           "trajectory_generator_sha256": sha_file(Path(__file__).resolve()),
                                           "environment_lock_sha256": sha_file(ROOT / "requirements-lock.txt"),
                                           "source_commit_at_generation": git_revision(ROOT),
                                           "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in ("qdk", "qiskit", "numpy")}},
                     "operator_equivalence_checks": len(ledger["mapping"]),
                     "cost_canonicalization_instrumentation_audit": EVALUATION_TIME_AUDIT,
                     "cost_reuse_probe": probe,
                     "source_and_lowered_qasm_records": {k: {a: b for a, b in v.items() if a != "qasm"} for k, v in ledger["qasm_records"].items()},
                     "conditional_replay_scope": "Conditioned on the complete recorded ideal sampled path; not a predicted noisy/synthesized hardware optimizer path.",
                     "actual_synthesis_sequences_generated": False, "angle_approximation_certificates_verified": False,
                     "optimizer_convergence_or_accuracy_certified": False,
                     "physical_preparation_feedback_reset_configuration_overheads": None,
                     "generation_wall_seconds": perf_counter() - started,
                     "artifacts": {"manifest": "manifest.json", "summary": "summary.md", "execution_mapping": "execution-mapping.csv",
                                   "model_signatures": "model-signatures.json", "projection_reports": "projection-reports.json"}})
    if runs_path.read_bytes() != runs_bytes or sha_file(implementation_path) != data["provenance"]["implementation_sha256"]:
        raise RuntimeError("Recorded inputs changed during replay generation; rerun against a stable snapshot")
    write_json(output / "manifest.json", manifest)
    write_json(output / "model-signatures.json", {"signatures": signatures, "per_variant_native_traces": variant_traces,
                                                 "query_budgets": budgets, "unsupported": unsupported})
    write_json(output / "projection-reports.json", {"status": manifest["status"], "reports": reports if pending is None else {}, "cost_reuse_probe": probe if pending is None else None})
    (output / "summary.md").write_text(replay_summary(manifest), encoding="utf-8")
    columns = ["batch_id", "task", "run_index", "seed", "phase", "evaluation_index", "call_role", "proposal_accepted", "group", "parameters",
               "shots", "source_qasm_sha256", "lowered_qasm_sha256", "cost_signature_by_profile"]
    with (output / "execution-mapping.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for entry in ledger["mapping"]:
            writer.writerow({**entry, "parameters": canonical_json(entry["parameters"]),
                             "cost_signature_by_profile": canonical_json(entry["cost_signature_by_profile"])})
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "assets/generated/optimization/sampled-runs.json")
    parser.add_argument("--recorded-implementation", type=Path, default=ROOT / "assets/generated/optimization/sampled/optimization.py")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/optimization/trajectory-resources")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qre-optimization-trajectory")
    parser.add_argument("--max-queries", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    result = generate(args.runs, args.recorded_implementation, args.output_dir, args.cache_dir, max_queries=args.max_queries)
    print(json.dumps({"status": result["status"], "calls": result["objective_and_validation_calls"],
                      "measured_group_batches": result["measured_group_batches"], "total_shots": result["total_shots"],
                      "unique_source_qasm": result["unique_source_qasm"], "unique_lowered_qasm": result["unique_lowered_qasm"],
                      "cost_signatures": result["cost_signatures"], "real_qdk_queries": result["real_qdk_query_count"],
                      "pending_reason": result["pending_reason"], "generation_wall_seconds": result["generation_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
