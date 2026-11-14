#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Replay the entire fixed quality study under conditional resource models.

Requires a source checkout and its pinned shared resource adapter. Downloads
are inspection sources, not a standalone estimator bundle. No actual angle
synthesis, physical reset, routing, feedback or common device layout is claimed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import shutil
from dataclasses import asdict
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src/openquantum/resource_estimation.py").is_file():
    raise SystemExit("Run from checkout/scripts/ with the shared resource adapter")
sys.path.insert(0, str(ROOT / "src"))
from openquantum.resource_estimation import CircuitResourceBudget, SurfaceCodeProfile, estimate_circuit_resources, prepare_measured_circuit

HELPER_PATH = ROOT / "scripts/generate_optimization_trajectory_resources.py"
helper_spec = importlib.util.spec_from_file_location("quality_replay_audited_helpers", HELPER_PATH)
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
PROFILES = tuple(helper.PROFILES)
TOTAL_SHOTS = 5_859_840
PHYSICAL_ALLOWANCE = 1e-3
SYNTHESIS_ALLOWANCE = 1e-5
MAX_QUERIES = 96
MAX_WALL_SECONDS = 300.0
MAX_ALLOWED_WALL_SECONDS = 1800.0


def validate_wall_allowance(max_wall_seconds):
    if (isinstance(max_wall_seconds, bool) or not isinstance(max_wall_seconds, (int, float))
            or not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_ALLOWED_WALL_SECONDS):
        raise ValueError("Replay wall allowance must be finite and lie in (0, 1800] seconds")
    return float(max_wall_seconds)


def require_wall_allowance(deadline, stage):
    """Stop audit work before an incomplete audit can become a query plan."""
    if deadline is not None and perf_counter() >= deadline:
        raise RuntimeError(f"Bounded replay wall allowance exhausted {stage}")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def optional_git_provenance():
    path = ROOT / "scripts/source_provenance.py"
    if not path.is_file():
        return {"source_commit": None, "owned_source_dirty": None, "git_helper_sha256": None}
    spec = importlib.util.spec_from_file_location("quality_resource_git_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"source_commit": module.git_revision(ROOT),
            "owned_source_dirty": module.git_dirty(ROOT, ["scripts/generate_optimization_quality_resources.py"]),
            "git_helper_sha256": digest(path)}


def load_inputs(folder):
    paths = {name: Path(folder) / f"{name}.json" for name in ("runs", "resources", "circuit-manifest")}
    raw = {name: path.read_bytes() for name, path in paths.items()}
    data = {name: json.loads(content) for name, content in raw.items()}
    source = Path(folder) / "optimization.py"
    source_hash = digest(source)
    if source_hash != data["runs"]["provenance"]["implementation_sha256"]:
        raise ValueError("Recorded module is not the executed numerical source")
    if (data["runs"]["artifact_status"] != "complete" or data["resources"]["artifact_status"] != "complete"
            or data["runs"]["total_shots"] != TOTAL_SHOTS
            or data["resources"]["total_preparations"] != TOTAL_SHOTS
            or not data["runs"]["recipe"]["fixed_published_recipe"]):
        raise ValueError("Only the complete fixed numerical workload can be projected")
    if data["resources"]["circuit_manifest"]["sha256"] != hashlib.sha256(raw["circuit-manifest"]).hexdigest():
        raise ValueError("Source circuit manifest hash is stale")
    spec = importlib.util.spec_from_file_location("recorded_quality_resource_module", source)
    implementation = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = implementation
    exec(compile(source.read_bytes(), str(source), "exec"), implementation.__dict__)
    if data["runs"]["recipe"] != implementation.quality_scaling_recipe():
        raise ValueError("Fixed recipe differs from its recorded executed module")
    descriptors = [{k: v for k, v in case.items() if k != "runs"} for case in data["runs"]["cases"]]
    if descriptors != data["runs"]["recipe"]["cases"]:
        raise ValueError("Recorded cases differ from the fixed recipe")
    implementation.quality_scaling_resources(data["runs"])
    return data, raw, implementation, source_hash


def measurement_map(circuit):
    result, begun = [], False
    for item in circuit.data:
        if item.operation.name == "measure":
            begun = True
            result.append([circuit.find_bit(item.qubits[0]).index, circuit.find_bit(item.clbits[0]).index])
        elif begun and item.operation.name != "barrier":
            raise ValueError("Only final measurements are supported")
    if sorted(pair[0] for pair in result) != list(range(circuit.num_qubits)) or len({pair[1] for pair in result}) != circuit.num_qubits:
        raise ValueError("Every qubit needs a distinct final measurement")
    return result


def aligned_error(a, b):
    overlap = np.vdot(a, b)
    if abs(overlap) == 0:
        return math.inf
    return float(np.max(np.abs(b - overlap / abs(overlap) * a)))


def independent_prepared_state(implementation, case, parameters, group):
    reference = (implementation.qaoa_layers_reference(case["n_qubits"], case["edges"], parameters)
                 if case["algorithm"] == "qaoa" else implementation.spin_vqe_reference(case["n_qubits"], parameters))
    state = reference["reference_state"].copy()
    if group == "all_x":
        size = len(state)
        for qubit in range(case["n_qubits"]):
            low = np.array([i for i in range(size) if not (i & (1 << qubit))])
            high = low ^ (1 << qubit)
            a, b = state[low].copy(), state[high].copy()
            state[low] = (a + b) / math.sqrt(2)
            state[high] = (a - b) / math.sqrt(2)
    return state


def prepared_equivalence(source, lowered, independent):
    if source.num_qubits > 8 or source.num_qubits != lowered.num_qubits or measurement_map(source) != measurement_map(lowered):
        raise ValueError("Bounded lowering changed width or readout wiring")
    a = Statevector.from_instruction(source.remove_final_measurements(inplace=False)).data
    b = Statevector.from_instruction(lowered.remove_final_measurements(inplace=False)).data
    error, reference_error = aligned_error(a, b), aligned_error(independent, a)
    if max(error, reference_error) > 1e-10:
        raise ValueError("Prepared source/lowered/reference states disagree")
    return {"prepared_state_verified": True, "max_source_lowered_amplitude_error": error,
            "max_independent_source_amplitude_error": reference_error, "tolerance": 1e-10,
            "up_to_global_phase": True, "full_operator_verified": False,
            "measurement_map": measurement_map(source),
            "scope": "selected zero-input prepared state and readout map; not a full circuit theorem"}


def bind_recorded_groups(data):
    """Bind the manifest to each actual recorded call, without trusting a second path."""
    expected = []
    for case in data["runs"]["cases"]:
        for run in case["runs"]:
            for phase, calls in (("training", run["evaluations"]), ("validation", [run["validation"]])):
                for call_index, call in enumerate(calls):
                    for group_index, group in enumerate(call["groups"]):
                        expected.append({"index": len(expected), "case_id": case["case_id"],
                                         "algorithm": case["algorithm"], "n_qubits": case["n_qubits"],
                                         "p": case["p"], "seed": run["seed"], "run_index": run["run_index"],
                                         "phase": phase, "evaluation_index": call_index, "group_index": group_index,
                                         "group": group["group"], "parameters": call["parameters"],
                                         **{name: group[name] for name in ("shots", "counts", "estimate", "sample_variance",
                                                                          "estimated_standard_error", "outcome_min", "outcome_max", "rng",
                                                                          "source_qasm_sha256", "source_qasm_file")},
                                         "source_gate_counts": group["circuit_gate_counts"],
                                         "source_depth": group["circuit_depth"]})
    actual = data["circuit-manifest"]["groups"]
    if len(actual) != len(expected) or any(any(row.get(k) != value for k, value in ref.items())
                                           for row, ref in zip(actual, expected)):
        raise ValueError("Circuit manifest is detached from the recorded numerical calls")
    return actual


def build_ledger(data, implementation, folder, output, deadline):
    cases = {c["case_id"]: c for c in data["runs"]["cases"]}
    recorded_groups = bind_recorded_groups(data)
    if len(recorded_groups) != 7656:
        raise ValueError("Every one of the 7656 measured groups is required")
    entries, unique, streams = [], {}, set()
    totals = Counter()
    lowered_folder = output / "lowered-qasm"
    lowered_folder.mkdir(parents=True, exist_ok=True)
    for index, group in enumerate(recorded_groups):
        require_wall_allowance(deadline, "before a source/lowering audit")
        if group["index"] != index:
            raise ValueError("Measured group indexes are missing or duplicated")
        case = cases[group["case_id"]]
        family = 0 if group["phase"] == "training" else 1 if group["phase"] == "validation" else -1
        expected = [group["seed"], case["stream_id"], family, group["evaluation_index"], group["group_index"]]
        if (family < 0 or group["rng"]["stream_key"] != expected or tuple(expected) in streams
                or group["rng"]["entropy"] != expected[:2] or group["rng"]["spawn_key"] != expected[2:]):
            raise ValueError("Invalid or repeated group RNG lineage")
        streams.add(tuple(expected))
        shots = group["shots"]
        if isinstance(shots, bool) or not isinstance(shots, int) or not 2 <= shots <= 65536:
            raise ValueError("Invalid group shot allocation")
        if sum(group["counts"].values()) != shots:
            raise ValueError("Recorded outcome counts omit shots")
        source_hash = group["source_qasm_sha256"]
        key = case["algorithm"] + ":" + source_hash
        source = implementation.quality_scaling_group_circuit(case, group["parameters"], group["group"])
        source_text = qasm3.dumps(source)
        if (text_digest(source_text) != source_hash or digest(Path(folder) / group["source_qasm_file"]) != source_hash
                or dict(source.count_ops()) != group["source_gate_counts"] or source.depth() != group["source_depth"]):
            raise ValueError("Source circuit bytes or recorded metadata are stale")
        if key not in unique:
            lowered = prepare_measured_circuit(source)
            check = prepared_equivalence(source, lowered, independent_prepared_state(implementation, case, group["parameters"], group["group"]))
            lowered_text = qasm3.dumps(lowered)
            lowered_hash = text_digest(lowered_text)
            (lowered_folder / f"{lowered_hash}.qasm").write_text(lowered_text)
            unique[key] = {"task": case["algorithm"], "source_qasm_sha256": source_hash,
                           "source_qasm_file": group["source_qasm_file"], "lowered_qasm_sha256": lowered_hash,
                           "lowered_qasm_file": f"lowered-qasm/{lowered_hash}.qasm",
                           "lowered_qasm": lowered_text, "equivalence": check,
                           "case_id": case["case_id"], "parameters": group["parameters"], "group": group["group"]}
        item = unique[key]
        entries.append({"batch_id": f"{group['case_id']}.seed{group['seed']}.{group['phase']}.call{group['evaluation_index']}.{group['group']}",
                        "index": index, "case_id": case["case_id"], "seed": group["seed"], "task": case["algorithm"],
                        "phase": group["phase"], "evaluation_index": group["evaluation_index"], "group": group["group"],
                        "shots": shots, "parameters": group["parameters"], "rng": group["rng"],
                        "source_qasm_sha256": source_hash, "lowered_qasm_sha256": item["lowered_qasm_sha256"],
                        "cost_signature_by_profile": {}})
        totals[group["phase"]] += shots
        if (index + 1) % 1024 == 0:
            print(json.dumps({"stage": "complete source/lowering checks", "groups": index + 1, "unique": len(unique)}), flush=True)
        require_wall_allowance(deadline, "after a source/lowering audit")
    if totals != Counter(training=4_483_584, validation=1_376_256) or len({e["batch_id"] for e in entries}) != len(entries):
        raise ValueError("Complete execution ledger has stale totals or duplicate identities")
    return {"mapping": entries, "unique": unique, "total_shots": sum(totals.values()),
            "training_shots": totals["training"], "validation_shots": totals["validation"]}


def cost_signature(qasm, task, profile):
    signature = helper.canonical_cost_signature(qasm, task=task, task_shots=TOTAL_SHOTS, profile=profile)
    config = signature["model_config"]
    config["whole_study_shots"] = config.pop("task_total_shots")
    config["whole_study_physical_failure_allowance"] = config.pop("task_physical_failure_allowance")
    config["nominal_study_physical_failure_per_execution"] = config.pop("nominal_physical_failure_allowance_per_execution")
    config["budget_scope"] = "all72recordedruns, including both algorithms and every training/holdout group"
    signature["id"] = helper.cost_signature_key(signature["cost_snapshot"], config)
    return signature


def allocate_budgets(mapping, signatures):
    result = {}
    if len({m["batch_id"] for m in mapping}) != len(mapping):
        raise ValueError("Execution identities are missing or duplicated")
    profile_names = {p.name for p in PROFILES}
    for m in mapping:
        if set(m["cost_signature_by_profile"]) != profile_names:
            raise ValueError("Every group needs exactly one cost class per profile")
        if isinstance(m["shots"], bool) or not isinstance(m["shots"], int) or m["shots"] <= 0:
            raise ValueError("Every group needs positive integer shots")
        for profile_name, sid in m["cost_signature_by_profile"].items():
            if sid not in signatures or signatures[sid]["model_config"]["profile"]["name"] != profile_name:
                raise ValueError("Unknown or mismatched profile cost class")
    for sid, signature in signatures.items():
        config = signature["model_config"]
        if (signature["id"] != sid or helper.cost_signature_key(signature["cost_snapshot"], config) != sid
                or config["profile"] not in [asdict(p) for p in PROFILES]
                or config["whole_study_shots"] != TOTAL_SHOTS
                or config["whole_study_physical_failure_allowance"] != PHYSICAL_ALLOWANCE
                or config["synthesis_model_allowance_per_execution"] != SYNTHESIS_ALLOWANCE
                or config["nominal_study_physical_failure_per_execution"] != PHYSICAL_ALLOWANCE / TOTAL_SHOTS):
            raise ValueError("Cost class has a stale or detached model contract")
        members = [m for m in mapping if m["cost_signature_by_profile"][signature["model_config"]["profile"]["name"]] == sid]
        if not members or any(m["task"] != config["task"] for m in members):
            raise ValueError("Unused or algorithm-mismatched cost class")
        executions = sum(m["shots"] for m in members)
        allowance = PHYSICAL_ALLOWANCE * executions / TOTAL_SHOTS
        budget = CircuitResourceBudget(SYNTHESIS_ALLOWANCE, allowance, executions)
        if not math.isclose(budget.physical_failure_per_execution, PHYSICAL_ALLOWANCE / TOTAL_SHOTS, rel_tol=1e-14, abs_tol=0):
            raise ValueError("Cost grouping changed the per-preparation study allocation")
        result[sid] = asdict(budget)
        signature["batch_ids"] = [m["batch_id"] for m in members]
    for profile in PROFILES:
        sids = [s for s in signatures if signatures[s]["model_config"]["profile"]["name"] == profile.name]
        if sum(result[s]["executions"] for s in sids) != TOTAL_SHOTS:
            raise ValueError("Cost groups do not cover the full study")
        if not math.isclose(sum(result[s]["physical_failure_probability"] for s in sids), PHYSICAL_ALLOWANCE, rel_tol=1e-12, abs_tol=0):
            raise ValueError("Whole-study physical allowance changed")
    return result


def numeric_equivalent(a, b):
    """Exact discrete/model fields; floats admit only reported roundoff tolerance."""
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(numeric_equivalent(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(numeric_equivalent(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0)
    return a == b


def aggregate_choices(reports):
    rows = []
    for profile in PROFILES:
        chosen = [r for r in reports.values() if r["profile"]["name"] == profile.name]
        if sum(r["executions"] for r in chosen) != TOTAL_SHOTS:
            raise ValueError("Physical reports omit part of the fixed study")
        for label, key in (("fewest qubits per class", "peak_physical_qubits"), ("shortest runtime per class", "runtime_per_execution_ns")):
            selections = []
            for r in chosen:
                point = min(r["report"]["pareto_points"], key=lambda p: (p[key], p["runtime_per_execution_ns"], p["peak_physical_qubits"]))
                selections.append({"signature_id": r["signature_id"], "executions": r["executions"], "point": point})
            failure = sum(s["executions"] * s["point"]["physical_failure_model_per_execution"] for s in selections)
            if failure > PHYSICAL_ALLOWANCE + 1e-12:
                raise ValueError("Aggregated physical score exceeds the full-study allowance")
            rows.append({"profile": profile.name, "physical_error_rate": profile.physical_error_rate,
                         "selection": label, "preparations": TOTAL_SHOTS,
                         "serial_quantum_model_runtime_ns": sum(s["executions"] * s["point"]["runtime_per_execution_ns"] for s in selections),
                         "peak_physical_qubits": max(s["point"]["peak_physical_qubits"] for s in selections),
                         "whole_study_physical_failure_union_bound": failure,
                         "class_selections": selections, "common_configuration_verified": False,
                         "reset_feedback_reconfiguration_overhead_ns": None, "joint_pareto_frontier_claimed": False})
    return rows


def summary_markdown(record):
    lines = ["This is a **conditional resource-model replay of the entire fixed 72-run study**. "
             "All 5,859,840 recorded preparations are counted, including rejected proposals and final holdouts. "
             "The recorded angles are replayed; a fresh noisy or synthesized hardware optimizer could choose another path.", "",
             f"Physical status: **{record['status']}**. Physical-estimator search limit: {record['query_limit']}; "
             f"searches actually requested: {record['native_query_count']}. Complete cost classes: {record['cost_signature_count']}. "
             f"The separate trace-signature audit count is {record['audit_counts']['canonical_trace_signature_audits']:,}; "
             "trace audits and internal adapter/native calls are not represented by the physical-search cap.", "",
             "The whole-study physical-fault allowance is $10^{-3}$, uniformly allocated over every preparation. "
             "The separate per-preparation $10^{-5}$ synthesis-model scalar is a mean-fit cost allowance, not a stochastic fault or concrete angle certificate.", ""]
    if record["status"] != "computed_conditional_recorded_replay":
        lines += [f"No aggregate physical result is released: {record['pending_reason']}.", ""]
    else:
        lines += ["| Physical error | Independently chosen class configuration | Preparations | Peak physical qubits | Serial quantum-model time (s) | Whole-study physical-fault score |",
                  "|---:|---|---:|---:|---:|---:|"]
        for row in record["choices"]:
            lines.append(f"| {row['physical_error_rate']:g} | {row['selection']} | {row['preparations']:,} | {row['peak_physical_qubits']:,} | {row['serial_quantum_model_runtime_ns']/1e9:.6f} | {row['whole_study_physical_failure_union_bound']:.3g} |")
    lines += ["", "The peak is the maximum selected footprint and serial time is the weighted sum. Class code distances/factories "
              "are independently selected; a common physical configuration is **not verified**. Layout changes, physical initialization/reset, routing, "
              "transport and adaptive feedback are unmeasured and excluded. The native LogicalCounts/PSSPC abstraction does not preserve original circuit topology.", "",
              "All source/lowered prepared states and readout maps are checked up to global phase within eight qubits, with independent algebraic/dense references. "
              "Full eight-qubit Operators are not allocated; this numerical scope is not a circuit theorem. Cost canonicalization removes only audited EVALUATION_TIME. "
              "One distinct-QASM empirical cost probe uses exact cost signatures and relative float tolerance $10^{-12}$ with zero absolute tolerance, not unitary equivalence.", "",
              "Measured local CPU controller/reference timings remain separate from modeled quantum time. Actual synthesis sequences, reset models, hardware feedback "
              "and general optimizer/quality guarantees remain outside this conditional report. Missed-quality runs cannot establish a matched-quality improvement."]
    return "\n".join(lines) + "\n"


def generate(folder, output, cache_dir, max_queries=MAX_QUERIES, max_wall_seconds=MAX_WALL_SECONDS):
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 0 <= max_queries <= MAX_QUERIES:
        raise ValueError("Native-query cap must lie in0..96")
    max_wall_seconds = validate_wall_allowance(max_wall_seconds)
    started = perf_counter()
    deadline = started + max_wall_seconds
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, raw, implementation, source_hash = load_inputs(folder)
    require_wall_allowance(deadline, "after input validation")
    ledger = build_ledger(data, implementation, folder, output, deadline)
    signatures, variants = {}, {}
    for key, item in ledger["unique"].items():
        for profile in PROFILES:
            require_wall_allowance(deadline, "before a trace-signature audit")
            signature = cost_signature(item["lowered_qasm"], item["task"], profile)
            require_wall_allowance(deadline, "after a trace-signature audit")
            sid = signature["id"]
            variants[key + ":" + profile.name] = {"signature_id": sid, "raw_trace": signature["raw_trace"], "psspc_trace": signature["psspc_trace"]}
            if sid not in signatures:
                signatures[sid] = {**signature, "variant_keys": []}
            signatures[sid]["variant_keys"].append(key)
    for m in ledger["mapping"]:
        key = m["task"] + ":" + m["source_qasm_sha256"]
        m["cost_signature_by_profile"] = {p.name: variants[key + ":" + p.name]["signature_id"] for p in PROFILES}
    budgets = allocate_budgets(ledger["mapping"], signatures)
    require_wall_allowance(deadline, "after complete query-plan allocation")
    probe_sid = next((sid for sid, s in signatures.items() if len(s["variant_keys"]) > 1), None)
    needed = len(signatures) + int(probe_sid is not None)
    pending = f"Complete classes and probe require {needed} native queries, exceeding cap{max_queries}" if needed > max_queries else None
    reports, probe, queries = {}, None, 0
    if pending is None:
        jobs = [(sid, signature["variant_keys"][0], False) for sid, signature in signatures.items()]
        if probe_sid is not None:
            jobs.append((probe_sid, signatures[probe_sid]["variant_keys"][1], True))
        for sid, key, is_probe in jobs:
            if perf_counter() >= deadline:
                pending = "Bounded replay wall allowance exhausted before complete native coverage"
                break
            signature, item = signatures[sid], ledger["unique"][key]
            case = next(c for c in data["runs"]["cases"] if c["case_id"] == item["case_id"])
            source = implementation.quality_scaling_group_circuit(case, item["parameters"], item["group"])
            profile = SurfaceCodeProfile(**signature["model_config"]["profile"])
            budget = CircuitResourceBudget(**budgets[sid])
            if perf_counter() >= deadline:
                pending = "Bounded replay wall allowance exhausted before a native query"
                break
            report = estimate_circuit_resources(source, budget=budget, profile=profile, cache_dir=Path(cache_dir), use_graph=False)
            queries += 1
            if perf_counter() >= deadline:
                pending = "Bounded replay wall allowance exhausted during a native query"
                break
            if report["qasm_sha256"] != item["lowered_qasm_sha256"]:
                raise RuntimeError("Actual estimated QASM differs from verified input")
            if is_probe:
                if perf_counter() >= deadline:
                    pending = "Bounded replay wall allowance exhausted before the distinct-QASM trace reparse"
                    break
                reread = cost_signature(report["qasm3"], item["task"], profile)
                if perf_counter() >= deadline:
                    pending = "Bounded replay wall allowance exhausted during the distinct-QASM trace reparse"
                    break
                if (reread["cost_snapshot"] != signature["cost_snapshot"] or reread["model_config"] != signature["model_config"]):
                    raise RuntimeError("Distinct-QASM probe left its exact cost class")
                a = helper.projection_cost_fingerprint(reports[sid]["report"])
                b = helper.projection_cost_fingerprint(report)
                equivalent = numeric_equivalent(a, b)
                probe = {"signature_id": sid, "distinct_source_qasm": key != signature["variant_keys"][0],
                         "equivalent_costs": equivalent, "relative_float_tolerance": 1e-12, "absolute_float_tolerance": 0,
                         "representative_fingerprint": a, "alternate_fingerprint": b, "alternate_report": report,
                         "not_a_unitary_equivalence_claim": True}
                if not equivalent:
                    pending = "Distinct-QASM empirical cost probe differed beyond reported numerical tolerance"
                    break
            else:
                reports[sid] = {"signature_id": sid, "profile": asdict(profile), "executions": budget.executions,
                                "report": report, "budget_allocation": budgets[sid], "representative_variant": key}
    choices = aggregate_choices(reports) if pending is None else []
    if pending is None and perf_counter() >= deadline:
        pending = "Bounded replay wall allowance exhausted during aggregate verification"
        choices = []
    control = {"recorded_optimizer_cpu_wall_seconds": sum(r["optimizer_wall_seconds"] for c in data["runs"]["cases"] for r in c["runs"]),
               "recorded_classical_control_wall_seconds": sum(r["classical_control_seconds"] for c in data["runs"]["cases"] for r in c["runs"]),
               "study_process_cpu_seconds": data["runs"]["workflow_process_cpu_seconds"],
               "study_wall_seconds": data["runs"]["workflow_wall_seconds"], "hardware_feedback_seconds": None,
               "added_to_quantum_model_runtime": False}
    record = {"schema_version": 1, "status": "computed_conditional_recorded_replay" if pending is None else "pending_physical_layer",
              "scope": "entire fixed72-run recorded workload, all training/holdout groups and source preparations",
              "pending_reason": pending, "query_limit": max_queries, "native_query_count": queries,
              "native_query_count_scope": "calls to the physical resource-estimator search adapter; excludes trace/import/PSSPC audits and internal native calls",
              "cost_signature_count": len(signatures), "planned_native_queries": needed,
              "total_preparations": TOTAL_SHOTS, "training_shots": ledger["training_shots"], "validation_shots": ledger["validation_shots"],
              "mapping": ledger["mapping"], "choices": choices, "classical_timings": control,
              "whole_study_physical_allowance": PHYSICAL_ALLOWANCE, "synthesis_model_scalar_per_execution": SYNTHESIS_ALLOWANCE,
              "synthesis_model_scalar_summed_over_shots": False, "actual_synthesis_sequences_generated": False,
              "common_configuration_verified": False, "reset_feedback_reconfiguration_overhead_ns": None,
              "source_provenance": {"input_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()},
                                    "recorded_implementation_sha256": source_hash,
                                    "resource_generator_sha256": digest(Path(__file__)), "shared_adapter_sha256": digest(ROOT / "src/openquantum/resource_estimation.py"),
                                    "audited_trace_helper_sha256": digest(HELPER_PATH), "environment_lock_sha256": digest(ROOT / "requirements-lock.txt"),
                                    "packages": {p: importlib.metadata.version(p) for p in ("qdk", "qiskit", "numpy")},
                                    "python": platform.python_version(), **optional_git_provenance()},
              "prepared_state_checks": len(ledger["unique"]), "full_operator_checks": 0,
              "audit_counts": {"recorded_groups_bound_to_numerical_calls": len(ledger["mapping"]),
                               "source_group_reconstructions": len(ledger["mapping"]),
                               "unique_prepared_state_checks": len(ledger["unique"]),
                               "canonical_trace_signature_audits": len(variants),
                               "distinct_qasm_trace_reparse_audits": int(probe is not None),
                               "physical_estimator_search_requests": queries,
                               "internal_adapter_native_calls_counted": False},
              "query_plan_complete": True, "wall_allowance_seconds": max_wall_seconds,
              "replay_wall_seconds": perf_counter() - started,
              "instrumentation_audit": helper.EVALUATION_TIME_AUDIT}
    for name, content in raw.items():
        if (Path(folder) / f"{name}.json").read_bytes() != content:
            raise RuntimeError("Recorded inputs changed during replay")
    if digest(Path(folder) / "optimization.py") != source_hash:
        raise RuntimeError("Recorded source changed during replay")
    record["replay_wall_seconds"] = perf_counter() - started
    if pending is None and record["replay_wall_seconds"] >= max_wall_seconds:
        pending = "Bounded replay wall allowance exhausted during final provenance verification"
        record.update(status="pending_physical_layer", pending_reason=pending, choices=[])
        choices = []
    write_json(output / "manifest.json", record)
    write_json(output / "model-signatures.json", {"signatures": signatures, "variant_traces": variants, "query_budgets": budgets})
    write_json(output / "projection-reports.json", {"status": record["status"], "reports": reports if pending is None else {}, "probe": probe})
    (output / "summary.md").write_text(summary_markdown(record))
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (output / "LICENSE.txt").write_bytes((ROOT / "LICENSE").read_bytes())
    with (output / "execution-mapping.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ledger["mapping"][0]), lineterminator="\n")
        writer.writeheader()
        for row in ledger["mapping"]:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    with (output / "choices.csv").open("w", newline="") as f:
        fields = ["profile", "physical_error_rate", "selection", "preparations", "serial_quantum_model_runtime_ns", "peak_physical_qubits", "whole_study_physical_failure_union_bound", "common_configuration_verified", "reset_feedback_reconfiguration_overhead_ns"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(choices)
    return record



def validate_retained_audit(folder, archive, deadline=None):
    """Revalidate immutable source/model bindings without repeating trace imports."""
    archive = Path(archive)
    inventory = json.loads((archive / "archive.json").read_text())
    for relative, expected in inventory["original_files_sha256"].items():
        require_wall_allowance(deadline, "before retained-file validation")
        if digest(archive / relative) != expected:
            raise ValueError("Frozen cap64 archive bytes changed")
    old = json.loads((archive / "manifest.json").read_text())
    if (old["query_limit"] != 64 or old["native_query_count"] != 0
            or old["cost_signature_count"] != 84 or old["planned_native_queries"] != 85):
        raise ValueError("Expected the exact original complete cap64 pending result")
    data, raw, implementation, source_hash = load_inputs(folder)
    require_wall_allowance(deadline, "after retained input validation")
    provenance = old["source_provenance"]
    if (provenance["recorded_implementation_sha256"] != source_hash
            or provenance["shared_adapter_sha256"] != digest(ROOT / "src/openquantum/resource_estimation.py")
            or provenance["audited_trace_helper_sha256"] != digest(HELPER_PATH)
            or provenance["environment_lock_sha256"] != digest(ROOT / "requirements-lock.txt")
            or provenance["packages"] != {p: importlib.metadata.version(p) for p in ("qdk", "qiskit", "numpy")}
            or provenance["input_sha256"] != {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()}):
        raise ValueError("Retained audit environment/source/input binding changed")
    groups = bind_recorded_groups(data)
    mapping = old["mapping"]
    if len(mapping) != len(groups):
        raise ValueError("Retained mapping omits recorded groups")
    cases = {c["case_id"]: c for c in data["runs"]["cases"]}
    unique = {}
    for row, group in zip(mapping, groups):
        require_wall_allowance(deadline, "before a retained source audit")
        for name in ("index", "case_id", "seed", "phase", "evaluation_index", "group", "shots", "parameters", "rng", "source_qasm_sha256"):
            if row[name] != group[name]:
                raise ValueError("Retained mapping differs from the recorded calls")
        case = cases[row["case_id"]]
        if row["task"] != case["algorithm"]:
            raise ValueError("Retained algorithm binding changed")
        key = row["task"] + ":" + row["source_qasm_sha256"]
        if key not in unique:
            source = implementation.quality_scaling_group_circuit(case, row["parameters"], row["group"])
            if (text_digest(qasm3.dumps(source)) != row["source_qasm_sha256"]
                    or digest(Path(folder) / group["source_qasm_file"]) != row["source_qasm_sha256"]):
                raise ValueError("Retained source does not match actual recorded parameters")
            lower_path = archive / "lowered-qasm" / (row["lowered_qasm_sha256"] + ".qasm")
            if digest(lower_path) != row["lowered_qasm_sha256"]:
                raise ValueError("Retained lowered QASM changed")
            unique[key] = {"task": row["task"], "source_qasm_sha256": row["source_qasm_sha256"],
                           "lowered_qasm_sha256": row["lowered_qasm_sha256"], "lowered_qasm": lower_path.read_text(),
                           "case_id": row["case_id"], "parameters": row["parameters"], "group": row["group"]}
        elif unique[key]["lowered_qasm_sha256"] != row["lowered_qasm_sha256"]:
            raise ValueError("One source has inconsistent retained lowering")
        require_wall_allowance(deadline, "after a retained source audit")
    model = json.loads((archive / "model-signatures.json").read_text())
    signatures, variants = model["signatures"], model["variant_traces"]
    for row in mapping:
        key = row["task"] + ":" + row["source_qasm_sha256"]
        for profile in PROFILES:
            require_wall_allowance(deadline, "before a retained trace-signature audit")
            sid = row["cost_signature_by_profile"][profile.name]
            variant = variants[key + ":" + profile.name]
            if variant["signature_id"] != sid:
                raise ValueError("Retained variant belongs to a different cost class")
            for native_key, cost_key in (("raw_trace", "canonical_raw_trace"), ("psspc_trace", "psspc_transformed_trace")):
                snapshot = variant[native_key]
                if helper.cost_canonical_trace_json(snapshot["full_native_json"]) != snapshot["cost_canonical_json"]:
                    raise ValueError("Retained trace lost fields beyond timing instrumentation")
                if {k: v for k, v in snapshot.items() if k != "full_native_json"} != signatures[sid]["cost_snapshot"][cost_key]:
                    raise ValueError("Retained native trace differs from its exact cost signature")
            require_wall_allowance(deadline, "after a retained trace-signature audit")
    budgets = allocate_budgets(mapping, signatures)
    if budgets != model["query_budgets"]:
        raise ValueError("Retained complete workload allocations changed")
    require_wall_allowance(deadline, "after retained query-plan allocation")
    return data, raw, implementation, source_hash, old, unique, signatures, variants, budgets


def resume_archived_audit(folder, archive, output, cache_dir, max_queries=MAX_QUERIES, max_wall_seconds=MAX_WALL_SECONDS):
    """Authorized computation-only cap amendment; all scientific inputs stay fixed."""
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 85 <= max_queries <= MAX_QUERIES:
        raise ValueError("The authorized complete-plan amended cap must lie in85..96")
    max_wall_seconds = validate_wall_allowance(max_wall_seconds)
    started = perf_counter()
    deadline = started + max_wall_seconds
    archive, output = Path(archive), Path(output)
    data, raw, implementation, source_hash, old, unique, signatures, variants, budgets = validate_retained_audit(folder, archive, deadline)
    require_wall_allowance(deadline, "after retained-audit validation")
    probe_sid = next(sid for sid, signature in signatures.items() if len(signature["variant_keys"]) > 1)
    needed = len(signatures) + 1
    if needed != 85 or needed > max_queries:
        raise ValueError("The retained complete85-search plan changed")
    output.mkdir(parents=True, exist_ok=True)
    amendment = {"schema_version": 1, "recorded_before_new_searches": True,
                 "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "previous_query_limit": 64,
                 "new_query_limit": max_queries, "planned_physical_estimator_searches": needed,
                 "wall_allowance_seconds": max_wall_seconds,
                 "previous_physical_estimator_searches": 0,
                 "reason": "User-authorized computation-only resource-search cap amendment to complete the existing whole85-query plan",
                 "cases_seeds_targets_shots_profiles_error_allowances_changed": False,
                 "old_manifest_sha256": digest(archive / "manifest.json"), "archive_inventory_sha256": digest(archive / "archive.json"),
                 "input_sha256": old["source_provenance"]["input_sha256"], "recorded_implementation_sha256": source_hash,
                 "physical_allowance": PHYSICAL_ALLOWANCE, "synthesis_model_scalar_per_execution": SYNTHESIS_ALLOWANCE,
                 "new_resource_generator_sha256": digest(Path(__file__)), "retained_audit_relative_directory": archive.name}
    amendment_path = output / "amendment.json"
    if amendment_path.exists():
        raise ValueError("An amendment already exists; inspect its checkpoint before any additional searches")
    write_json(amendment_path, amendment)
    print(json.dumps({"stage": "authorized amendment recorded before searches", "old_cap": 64, "new_cap": max_queries,
                      "complete_plan": needed, "retained_prepared_checks": old["prepared_state_checks"]}), flush=True)
    reports, probe, pending, queries, failed = {}, None, None, 0, None
    jobs = [(sid, signature["variant_keys"][0], False) for sid, signature in signatures.items()]
    jobs.append((probe_sid, signatures[probe_sid]["variant_keys"][1], True))
    for sid, key, is_probe in jobs:
        if queries >= max_queries or perf_counter() >= deadline:
            pending = "Bounded amended search allowance exhausted before complete coverage"
            break
        signature, item = signatures[sid], unique[key]
        case = next(c for c in data["runs"]["cases"] if c["case_id"] == item["case_id"])
        source = implementation.quality_scaling_group_circuit(case, item["parameters"], item["group"])
        profile = SurfaceCodeProfile(**signature["model_config"]["profile"])
        if perf_counter() >= deadline:
            pending = "Bounded amended wall allowance exhausted before a native query"
            break
        queries += 1
        try:
            report = estimate_circuit_resources(source, budget=CircuitResourceBudget(**budgets[sid]), profile=profile,
                                                cache_dir=Path(cache_dir), use_graph=False)
            require_wall_allowance(deadline, "during an amended native query")
            actual = cost_signature(report["qasm3"], item["task"], profile)
            require_wall_allowance(deadline, "during an amended trace reparse")
            if (report["qasm_sha256"] != item["lowered_qasm_sha256"] or actual["cost_snapshot"] != signature["cost_snapshot"]
                    or actual["model_config"] != signature["model_config"] or report["budget"] != budgets[sid]):
                raise RuntimeError("Actual search input/cost signature/allocation differs from its retained audit")
            if is_probe:
                a = helper.projection_cost_fingerprint(reports[sid]["report"])
                b = helper.projection_cost_fingerprint(report)
                equivalent = numeric_equivalent(a, b)
                probe = {"signature_id": sid, "distinct_source_qasm": key != signature["variant_keys"][0],
                         "equivalent_costs": equivalent, "relative_float_tolerance": 1e-12, "absolute_float_tolerance": 0,
                         "representative_fingerprint": a, "alternate_fingerprint": b, "alternate_report": report,
                         "not_a_unitary_equivalence_claim": True}
                if not equivalent:
                    pending = "Distinct-QASM empirical cost probe differed beyond the predeclared tolerance"
            else:
                reports[sid] = {"signature_id": sid, "profile": asdict(profile), "executions": budgets[sid]["executions"],
                                "report": report, "budget_allocation": budgets[sid], "representative_variant": key}
        except Exception as error:
            failed = {"signature_id": sid, "query_number": queries, "probe": is_probe,
                      "error_type": type(error).__name__, "reason": str(error).replace(str(ROOT), "<checkout>")[:500]}
            pending = "An amended native search or input revalidation failed; partial evidence is retained"
        write_json(output / "query-checkpoint.json", {"completed_representative_reports": reports, "probe": probe,
                                                       "physical_estimator_search_requests": queries, "pending_reason": pending,
                                                       "failed_search": failed, "amendment_sha256": digest(amendment_path)})
        print(json.dumps({"stage": "amended physical search", "requested": queries, "complete_plan": needed,
                          "representative_reports": len(reports), "probe": is_probe, "pending": pending}), flush=True)
        if pending is not None:
            break
    if queries != needed and pending is None:
        pending = "The complete amended plan was not executed"
    for name, content in raw.items():
        if (Path(folder) / f"{name}.json").read_bytes() != content:
            raise RuntimeError("Recorded scientific inputs changed during amended searches")
    choices = aggregate_choices(reports) if pending is None else []
    if pending is None and perf_counter() >= deadline:
        pending = "Bounded amended wall allowance exhausted during final verification"
        choices = []
    record = json.loads(json.dumps(old))
    record.update(status="computed_conditional_recorded_replay" if pending is None else "pending_physical_layer",
                  pending_reason=pending, query_limit=max_queries, native_query_count=queries,
                  choices=choices, wall_allowance_seconds=max_wall_seconds,
                  failed_search=failed, amendment=amendment, amendment_sha256=digest(amendment_path),
                  distinct_qasm_probe_executed=probe is not None,
                  retained_audit={"directory": archive.name, "manifest_sha256": digest(archive / "manifest.json"),
                                  "original_generator_sha256": old["source_provenance"]["resource_generator_sha256"],
                                  "prepared_state_checks_reused": old["prepared_state_checks"],
                                  "canonical_trace_signature_audits_reused": old["audit_counts"]["canonical_trace_signature_audits"]},
                  audit_counts={**old["audit_counts"], "fresh_prepared_state_checks": 0,
                                "fresh_trace_signature_reparse_audits": queries if failed is None else queries - 1,
                                "physical_estimator_search_requests": queries},
                  prepared_state_checks_reexecuted=0,
                  amended_validation_and_search_wall_seconds=perf_counter() - started)
    record["source_provenance"] = {**old["source_provenance"], "resource_generator_sha256": digest(Path(__file__)),
                                    "retained_audit_generator_sha256": old["source_provenance"]["resource_generator_sha256"],
                                    **optional_git_provenance()}
    record["amended_validation_and_search_wall_seconds"] = perf_counter() - started
    if pending is None and record["amended_validation_and_search_wall_seconds"] >= max_wall_seconds:
        pending = "Bounded amended wall allowance exhausted during final provenance verification"
        record.update(status="pending_physical_layer", pending_reason=pending, choices=[])
    write_json(output / "query-checkpoint.json", {"completed_representative_reports": reports, "probe": probe,
                                                   "physical_estimator_search_requests": queries, "pending_reason": pending,
                                                   "failed_search": failed, "amendment_sha256": record["amendment_sha256"]})
    write_json(output / "manifest.json", record)
    write_json(output / "model-signatures.json", {"signatures": signatures, "variant_traces": variants, "query_budgets": budgets})
    write_json(output / "projection-reports.json", {"status": record["status"], "reports": reports if pending is None else {},
                                                    "partial_reports": reports if pending is not None else {}, "probe": probe})
    summary = summary_markdown(record)
    summary = summary.replace("One distinct-QASM empirical cost probe uses", "The executed distinct-QASM empirical cost probe uses") if probe is not None else summary.replace("One distinct-QASM empirical cost probe uses", "The unexecuted planned distinct-QASM empirical cost probe would use")
    (output / "summary.md").write_text(summary)
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    with (output / "choices.csv").open("w", newline="") as f:
        fields = ["profile", "physical_error_rate", "selection", "preparations", "serial_quantum_model_runtime_ns", "peak_physical_qubits", "whole_study_physical_failure_union_bound", "common_configuration_verified", "reset_feedback_reconfiguration_overhead_ns"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(record["choices"])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "assets/generated/optimization/quality-scaling")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/optimization/quality-resources")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qre-optimization-quality")
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    parser.add_argument("--max-wall-seconds", type=float, default=MAX_WALL_SECONDS,
                        help="bounded replay allowance in seconds, at most 1800 (default: 300)")
    parser.add_argument("--require-complete", action="store_true",
                        help="exit nonzero unless the complete conditional physical replay succeeds")
    parser.add_argument("--resume-archive", type=Path, help="validate and resume the authorized complete plan from the exact cap64 archive")
    args = parser.parse_args(argv)
    result = (resume_archived_audit(args.input_dir, args.resume_archive, args.output_dir, args.cache_dir, args.max_queries, args.max_wall_seconds)
              if args.resume_archive else generate(args.input_dir, args.output_dir, args.cache_dir, args.max_queries, args.max_wall_seconds))
    print(json.dumps({k: result[k] for k in ("status", "pending_reason", "native_query_count", "cost_signature_count", "total_preparations", "replay_wall_seconds")}, indent=2))
    if args.require_complete and result["status"] != "computed_conditional_recorded_replay":
        raise SystemExit(f"Complete quality-resource replay required: {result['pending_reason'] or result['status']}")


if __name__ == "__main__":
    main()
