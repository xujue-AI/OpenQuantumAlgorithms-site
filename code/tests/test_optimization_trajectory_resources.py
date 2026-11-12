"""Validate full-trajectory circuit mapping and model aggregation without QRE."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path

import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Operator

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "optimization_trajectory_resource_script",
    ROOT / "scripts/generate_optimization_trajectory_resources.py",
)
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)


def _hash(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def frozen_source_and_runs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("trajectory-source")
    source_bytes = (ROOT / "src/openquantum/optimization.py").read_bytes()
    source = folder / "optimization.py"
    source.write_bytes(source_bytes)
    provenance = {"implementation_sha256": _hash(source_bytes)}
    implementation = trajectory.load_recorded_implementation({"provenance": provenance}, source)
    data = implementation.run_sampled_demo()
    data["provenance"] = provenance
    return source, implementation, data


@pytest.fixture(scope="module")
def full_ledger(frozen_source_and_runs):
    _, implementation, data = frozen_source_and_runs
    return trajectory.build_execution_ledger(data, implementation=implementation)


def test_recorded_source_is_hash_bound_and_stale_bytes_are_rejected(frozen_source_and_runs, tmp_path):
    source, _, data = frozen_source_and_runs
    assert _hash(source.read_bytes()) == data["provenance"]["implementation_sha256"]
    stale = tmp_path / "stale_optimization.py"
    stale.write_bytes(source.read_bytes() + b"\n# different source bytes\n")
    with pytest.raises(ValueError):
        trajectory.load_recorded_implementation(data, stale)


def test_full_ledger_contains_every_group_and_preparation(full_ledger):
    assert full_ledger["objective_and_validation_calls"] == 780
    assert full_ledger["measured_group_batches"] == len(full_ledger["mapping"]) == 1074
    assert full_ledger["training_shots"] == 1016064
    assert full_ledger["validation_shots"] == 36864
    assert full_ledger["task_totals"] == {"qaoa": 473856, "vqe": 579072}
    assert full_ledger["total_shots"] == 1052928
    mapping = full_ledger["mapping"]
    assert len({entry["batch_id"] for entry in mapping}) == len(mapping)
    assert len({tuple(entry["rng"]["stream_key"]) for entry in mapping}) == len(mapping)
    for task in ("qaoa", "vqe"):
        entries = [entry for entry in mapping if entry["task"] == task]
        assert sum(entry["shots"] for entry in entries) == full_ledger["task_totals"][task]
        for gate, total in full_ledger["weighted_source_gate_counts"][task].items():
            assert total == sum(entry["source_gate_counts"].get(gate, 0) * entry["shots"] for entry in entries)
    assert full_ledger["weighted_source_gate_counts"]["qaoa"]["measure"] == 4 * 473856
    assert full_ledger["weighted_source_gate_counts"]["vqe"]["measure"] == 2 * 579072


def test_each_unique_lowering_has_independent_operator_and_qasm_checks(full_ledger, frozen_source_and_runs):
    _, implementation, _ = frozen_source_and_runs
    checked = set()
    for entry in full_ledger["mapping"]:
        assert entry["lowering_equivalence"]["verified"]
        assert entry["lowering_equivalence"]["source_measurement_map"] == entry["lowering_equivalence"]["lowered_measurement_map"]
        digest = entry["lowered_qasm_sha256"]
        if digest in checked:
            continue
        checked.add(digest)
        source = trajectory.source_circuit_for_group(implementation, entry["task"], entry["parameters"], entry["group"])
        lowered = trajectory.prepare_measured_circuit(source)
        assert Operator(source.remove_final_measurements(inplace=False)).equiv(
            Operator(lowered.remove_final_measurements(inplace=False)), atol=1e-12)
        source_text, lowered_text = qasm3.dumps(source), qasm3.dumps(lowered)
        assert _hash(source_text.encode()) == entry["source_qasm_sha256"]
        assert _hash(lowered_text.encode()) == digest
        assert full_ledger["qasm_records"]["source:" + entry["source_qasm_sha256"]]["qasm"] == source_text
        assert full_ledger["qasm_records"]["lowered:" + digest]["qasm"] == lowered_text
    assert len(checked) == full_ledger["unique_lowered_qasm"]


def test_operator_check_rejects_a_change_in_unvisited_subspaces(frozen_source_and_runs):
    _, implementation, _ = frozen_source_and_runs
    source = implementation.build_vqe_circuit(0.6, measure=True)
    wrong = source.remove_final_measurements(inplace=False)
    # ZZ is +1 on this ansatz's prepared even-parity state, but it changes
    # the circuit's full action on odd-parity inputs. A state-only test misses it.
    wrong.z(0)
    wrong.z(1)
    wrong.measure_all()
    with pytest.raises(ValueError):
        trajectory.lowering_equivalence(source, wrong)


def test_lowering_cannot_silently_change_measurement_bit_wiring(frozen_source_and_runs):
    _, implementation, _ = frozen_source_and_runs
    source = implementation.build_vqe_circuit(0.6, measure=True)
    wrong = QuantumCircuit(2, 2)
    wrong.compose(source.remove_final_measurements(inplace=False), inplace=True)
    wrong.measure([0, 1], [1, 0])
    with pytest.raises(ValueError):
        trajectory.lowering_equivalence(source, wrong)


@pytest.mark.parametrize("corruption", ["missing_group", "duplicate_call", "wrong_stream", "incomplete_counts", "wrong_depth", "wrong_validation_state"])
def test_corrupted_recorded_group_mapping_is_rejected(frozen_source_and_runs, corruption):
    _, implementation, original = frozen_source_and_runs
    data = deepcopy(original)
    if corruption == "missing_group":
        data["vqe"]["runs"][0]["evaluations"][0]["groups"].pop()
    elif corruption == "duplicate_call":
        data["qaoa"]["runs"][0]["evaluations"].append(deepcopy(data["qaoa"]["runs"][0]["evaluations"][0]))
    elif corruption == "wrong_stream":
        data["qaoa"]["runs"][0]["evaluations"][0]["groups"][0]["rng"]["stream_key"][2] = 1
    elif corruption == "incomplete_counts":
        counts = data["qaoa"]["runs"][0]["evaluations"][0]["groups"][0]["counts"]
        counts[next(iter(counts))] -= 1
    elif corruption == "wrong_depth":
        data["qaoa"]["runs"][0]["evaluations"][0]["groups"][0]["circuit_depth"] += 1
    else:
        data["qaoa"]["runs"][0]["validation"]["parameters"][0] += 0.1
    with pytest.raises(ValueError):
        trajectory.build_execution_ledger(data, implementation=implementation)


def native_snapshot():
    return {"block": {"operations": [
        {"GateOperation": {"id": 1038, "qubits": [0], "params": []}},
        {"GateOperation": {"id": 258, "qubits": [1], "params": []}}], "repetitions": 1},
        "base_error": 0.0, "compute_qubits": 2, "memory_qubits": None,
        "resource_states": {"1038": 2},
        "properties": {"7": {"Int": 1234}, "14": {"Int": 2}}}


def test_cost_canonicalization_drops_only_import_timing_and_retains_original():
    original = native_snapshot()
    altered = deepcopy(original)
    altered["properties"]["7"]["Int"] += 999
    before = deepcopy(original)
    canonical = trajectory.cost_canonical_trace_json(original)
    assert original == before
    assert canonical == trajectory.cost_canonical_trace_json(altered)
    assert "7" not in canonical["properties"]
    assert canonical["properties"]["14"] == original["properties"]["14"]


@pytest.mark.parametrize("change", ["operation_order", "repetitions", "gate_qubit", "width", "base_error", "resource_states", "cost_property", "unknown_property"])
def test_matching_gate_counts_are_insufficient_for_cost_signature_reuse(change):
    first = native_snapshot()
    second = deepcopy(first)
    if change == "operation_order":
        second["block"]["operations"].reverse()
    elif change == "repetitions":
        second["block"]["repetitions"] = 2
    elif change == "gate_qubit":
        second["block"]["operations"][0]["GateOperation"]["qubits"] = [1]
    elif change == "width":
        second["compute_qubits"] = 3
    elif change == "base_error":
        second["base_error"] = 1e-7
    elif change == "resource_states":
        second["resource_states"]["1038"] = 3
    elif change == "cost_property":
        second["properties"]["14"]["Int"] = 3
    else:
        second["properties"]["9999"] = {"Int": 1}
    config = {"task": "qaoa", "profile": asdict(trajectory.PROFILES[0]), "physical_per_execution": 1e-9}
    assert trajectory.cost_signature_key(trajectory.cost_canonical_trace_json(first), config) != trajectory.cost_signature_key(
        trajectory.cost_canonical_trace_json(second), config)


def test_native_canonical_signature_retains_raw_and_psspc_metadata_without_qre():
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.rz(0.123, 0)
    circuit.cx(0, 1)
    circuit.rx(0.456, 1)
    circuit.measure_all()
    lowered = trajectory.prepare_measured_circuit(circuit)
    signature = trajectory.canonical_cost_signature(qasm3.dumps(lowered), task="qaoa", task_shots=473856,
                                                     profile=trajectory.PROFILES[0])
    assert signature["id"] == trajectory.cost_signature_key(signature["cost_snapshot"], signature["model_config"])
    for name in ("raw_trace", "psspc_trace"):
        trace = signature[name]
        assert trace["native_json_roundtrip_verified"]
        assert trace["cost_canonical_json"] == trajectory.cost_canonical_trace_json(trace["full_native_json"])
    assert signature["model_config"]["nominal_physical_failure_allowance_per_execution"] == pytest.approx(1e-3 / 473856)


def fake_signatures(ledger):
    signatures = {}
    for profile in trajectory.PROFILES:
        for task, total in ledger["task_totals"].items():
            entries = [entry for entry in ledger["mapping"] if entry["task"] == task]
            for partition in (0, 1):
                members = entries[partition::2]
                config = {"task": task, "task_total_shots": total, "profile": asdict(profile),
                          "task_physical_failure_allowance": 1e-3,
                          "nominal_physical_failure_allowance_per_execution": 1e-3 / total,
                          "synthesis_model_allowance_per_execution": 1e-5}
                snapshot = {"test_partition": partition}
                sid = trajectory.cost_signature_key(snapshot, config)
                signatures[sid] = {"id": sid, "cost_snapshot": snapshot, "model_config": config,
                                   "batch_ids": [entry["batch_id"] for entry in members],
                                   "lowered_qasm_sha256s": sorted({entry["lowered_qasm_sha256"] for entry in members})}
                for entry in members:
                    entry.setdefault("cost_signature_by_profile", {})[profile.name] = sid
    return signatures


def test_signature_budgets_cover_all_shots_with_one_uniform_task_fault_allocation(full_ledger):
    ledger = deepcopy(full_ledger)
    signatures = fake_signatures(ledger)
    budgets = trajectory.allocate_signature_budgets(ledger, signatures)
    for sid, budget in budgets.items():
        task = signatures[sid]["model_config"]["task"]
        executions = sum(entry["shots"] for entry in ledger["mapping"] if entry["batch_id"] in signatures[sid]["batch_ids"])
        assert budget["executions"] == executions
        assert budget["physical_failure_probability"] == pytest.approx(1e-3 * executions / ledger["task_totals"][task])
        assert budget["actual_query_physical_failure_per_execution"] == pytest.approx(1e-3 / ledger["task_totals"][task])
    for profile in trajectory.PROFILES:
        for task, total in ledger["task_totals"].items():
            selected = [budgets[sid] for sid, signature in signatures.items()
                        if signature["model_config"]["task"] == task and signature["model_config"]["profile"]["name"] == profile.name]
            assert sum(budget["executions"] for budget in selected) == total
            assert sum(budget["physical_failure_probability"] for budget in selected) == pytest.approx(1e-3)


@pytest.mark.parametrize("corruption", ["missing_member", "duplicate_member", "unknown_member", "duplicate_ledger_id", "stale_task_total", "wrong_mapped_signature", "changed_physical_profile", "changed_fault_allowance"])
def test_stale_missing_or_duplicate_signature_mapping_is_rejected(full_ledger, corruption):
    ledger = deepcopy(full_ledger)
    signatures = fake_signatures(ledger)
    sid = next(iter(signatures))
    signature = signatures[sid]
    if corruption == "missing_member":
        signature["batch_ids"].pop()
    elif corruption == "duplicate_member":
        signature["batch_ids"].append(signature["batch_ids"][0])
    elif corruption == "unknown_member":
        signature["batch_ids"][0] = "not-recorded"
    elif corruption == "duplicate_ledger_id":
        ledger["mapping"].append(deepcopy(ledger["mapping"][0]))
    elif corruption == "stale_task_total":
        signature["model_config"]["task_total_shots"] += 1
    elif corruption == "changed_physical_profile":
        signature["model_config"]["profile"]["measurement_time_ns"] += 1
    elif corruption == "changed_fault_allowance":
        signature["model_config"]["task_physical_failure_allowance"] *= 2
    else:
        profile = signature["model_config"]["profile"]["name"]
        member = next(entry for entry in ledger["mapping"] if entry["batch_id"] == signature["batch_ids"][0])
        member["cost_signature_by_profile"][profile] = "stale-signature"
    with pytest.raises((ValueError, RuntimeError)):
        trajectory.allocate_signature_budgets(ledger, signatures)


def fake_reports(task_totals):
    reports = {}
    for profile in trajectory.PROFILES:
        for task, total in task_totals.items():
            for part, executions in enumerate((total // 3, total - total // 3)):
                sid = f"{profile.name}:{task}:{part}"
                reports[sid] = {"signature_id": sid, "task": task, "profile": asdict(profile), "executions": executions,
                                "report": {"pareto_points": [
                                    {"peak_physical_qubits": 100 + part, "runtime_per_execution_ns": 20 + part,
                                     "physical_failure_model_per_execution": 0.8e-3 / total},
                                    {"peak_physical_qubits": 500 + part, "runtime_per_execution_ns": 5 + part,
                                     "physical_failure_model_per_execution": 0.7e-3 / total}]}}
    return reports


def test_serial_replay_uses_shot_weighted_runtime_and_maximum_peak(full_ledger):
    reports = fake_reports(full_ledger["task_totals"])
    rows = trajectory.aggregate_replay_choices(reports, full_ledger["task_totals"])
    assert len(rows) == 8
    for row in rows:
        selected = row["signature_selections"]
        assert sum(entry["executions"] for entry in selected) == full_ledger["task_totals"][row["task"]]
        assert row["serial_quantum_runtime_ns"] == sum(entry["point"]["runtime_per_execution_ns"] * entry["executions"] for entry in selected)
        assert row["peak_physical_qubits"] == max(entry["point"]["peak_physical_qubits"] for entry in selected)
        assert row["physical_failure_union_bound_all_preparations"] == pytest.approx(
            sum(entry["point"]["physical_failure_model_per_execution"] * entry["executions"] for entry in selected))
        assert row["physical_failure_union_bound_all_preparations"] <= 1e-3
        assert row["synthesis_model_scalar_summed_over_shots"] is False
        assert row["not_a_complete_joint_pareto_frontier"] is True


@pytest.mark.parametrize("corruption", ["missing_report", "duplicate_executions", "overallocated_faults"])
def test_incomplete_or_overallocated_physical_replay_cannot_be_aggregated(full_ledger, corruption):
    reports = fake_reports(full_ledger["task_totals"])
    key = next(iter(reports))
    if corruption == "missing_report":
        reports.pop(key)
    elif corruption == "duplicate_executions":
        reports[key]["executions"] += 1
    else:
        for report in reports.values():
            for point in report["report"]["pareto_points"]:
                point["physical_failure_model_per_execution"] = 2e-3 / full_ledger["task_totals"][report["task"]]
    with pytest.raises(ValueError):
        trajectory.aggregate_replay_choices(reports, full_ledger["task_totals"])
