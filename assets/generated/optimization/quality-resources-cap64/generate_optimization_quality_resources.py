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
MAX_QUERIES = 64
MAX_WALL_SECONDS = 300.0


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
        if perf_counter() > deadline:
            raise RuntimeError("Complete physical replay exceeded its 300-second wall cap")
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


def generate(folder, output, cache_dir, max_queries=MAX_QUERIES):
    if isinstance(max_queries, bool) or not isinstance(max_queries, int) or not 0 <= max_queries <= MAX_QUERIES:
        raise ValueError("Native-query cap must lie in0..64")
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, raw, implementation, source_hash = load_inputs(folder)
    ledger = build_ledger(data, implementation, folder, output, started + MAX_WALL_SECONDS)
    signatures, variants = {}, {}
    for key, item in ledger["unique"].items():
        for profile in PROFILES:
            signature = cost_signature(item["lowered_qasm"], item["task"], profile)
            sid = signature["id"]
            variants[key + ":" + profile.name] = {"signature_id": sid, "raw_trace": signature["raw_trace"], "psspc_trace": signature["psspc_trace"]}
            if sid not in signatures:
                signatures[sid] = {**signature, "variant_keys": []}
            signatures[sid]["variant_keys"].append(key)
    for m in ledger["mapping"]:
        key = m["task"] + ":" + m["source_qasm_sha256"]
        m["cost_signature_by_profile"] = {p.name: variants[key + ":" + p.name]["signature_id"] for p in PROFILES}
    budgets = allocate_budgets(ledger["mapping"], signatures)
    probe_sid = next((sid for sid, s in signatures.items() if len(s["variant_keys"]) > 1), None)
    needed = len(signatures) + int(probe_sid is not None)
    pending = f"Complete classes and probe require {needed} native queries, exceeding cap{max_queries}" if needed > max_queries else None
    reports, probe, queries = {}, None, 0
    if pending is None:
        for sid, signature in signatures.items():
            key = signature["variant_keys"][0]
            item = ledger["unique"][key]
            case = next(c for c in data["runs"]["cases"] if c["case_id"] == item["case_id"])
            source = implementation.quality_scaling_group_circuit(case, item["parameters"], item["group"])
            profile = SurfaceCodeProfile(**signature["model_config"]["profile"])
            budget = CircuitResourceBudget(**budgets[sid])
            if perf_counter() - started > MAX_WALL_SECONDS:
                pending = "Bounded replay wall allowance exhausted before all native queries"
                break
            report = estimate_circuit_resources(source, budget=budget, profile=profile, cache_dir=Path(cache_dir), use_graph=False)
            queries += 1
            if report["qasm_sha256"] != item["lowered_qasm_sha256"]:
                raise RuntimeError("Actual estimated QASM differs from verified input")
            reports[sid] = {"signature_id": sid, "profile": asdict(profile), "executions": budget.executions,
                            "report": report, "budget_allocation": budgets[sid], "representative_variant": key}
        if pending is None and probe_sid is not None:
            signature = signatures[probe_sid]
            key = signature["variant_keys"][1]
            item = ledger["unique"][key]
            case = next(c for c in data["runs"]["cases"] if c["case_id"] == item["case_id"])
            profile = SurfaceCodeProfile(**signature["model_config"]["profile"])
            source = implementation.quality_scaling_group_circuit(case, item["parameters"], item["group"])
            report = estimate_circuit_resources(source, budget=CircuitResourceBudget(**budgets[probe_sid]), profile=profile, cache_dir=Path(cache_dir), use_graph=False)
            queries += 1
            reread = cost_signature(report["qasm3"], item["task"], profile)
            if (report["qasm_sha256"] != item["lowered_qasm_sha256"] or reread["cost_snapshot"] != signature["cost_snapshot"]
                    or reread["model_config"] != signature["model_config"]):
                raise RuntimeError("Distinct-QASM probe left its exact cost class")
            a = helper.projection_cost_fingerprint(reports[probe_sid]["report"])
            b = helper.projection_cost_fingerprint(report)
            equivalent = numeric_equivalent(a, b)
            probe = {"signature_id": probe_sid, "distinct_source_qasm": key != signature["variant_keys"][0],
                     "equivalent_costs": equivalent, "relative_float_tolerance": 1e-12, "absolute_float_tolerance": 0,
                     "representative_fingerprint": a, "alternate_fingerprint": b, "alternate_report": report,
                     "not_a_unitary_equivalence_claim": True}
            if not equivalent:
                pending = "Distinct-QASM empirical cost probe differed beyond reported numerical tolerance"
    choices = aggregate_choices(reports) if pending is None else []
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
              "query_plan_complete": True, "replay_wall_seconds": perf_counter() - started,
              "instrumentation_audit": helper.EVALUATION_TIME_AUDIT}
    for name, content in raw.items():
        if (Path(folder) / f"{name}.json").read_bytes() != content:
            raise RuntimeError("Recorded inputs changed during replay")
    if digest(Path(folder) / "optimization.py") != source_hash:
        raise RuntimeError("Recorded source changed during replay")
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "assets/generated/optimization/quality-scaling")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets/generated/optimization/quality-resources")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/qre-optimization-quality")
    parser.add_argument("--max-queries", type=int, default=MAX_QUERIES)
    args = parser.parse_args()
    result = generate(args.input_dir, args.output_dir, args.cache_dir, args.max_queries)
    print(json.dumps({k: result[k] for k in ("status", "pending_reason", "native_query_count", "cost_signature_count", "total_preparations", "replay_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
