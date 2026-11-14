"""Independent provenance, allocation and prepared-state resource checks.

These tests do not launch the native physical resource search. Full recorded
mapping comparisons inspect dictionaries; tiny circuits test readout semantics.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm3
from qiskit.quantum_info import Statevector

from openquantum import optimization as opt

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "quality_resource_test_script", ROOT / "scripts/generate_optimization_quality_resources.py"
)
resource = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resource)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def minimal_record(tmp_path):
    """Isolate loader hash checks from simulator/ledger work."""
    recipe = {"fixed_published_recipe": True, "cases": []}
    source = tmp_path / "optimization.py"
    source.write_text(
        "def quality_scaling_recipe(): return {'fixed_published_recipe': True, 'cases': []}\n"
        "def quality_scaling_resources(data): return None\n"
    )
    manifest = tmp_path / "circuit-manifest.json"
    manifest.write_text(json.dumps({"groups": []}) + "\n")
    runs = {"provenance": {"implementation_sha256": _hash(source)},
            "artifact_status": "complete", "total_shots": resource.TOTAL_SHOTS,
            "recipe": recipe, "cases": []}
    resources = {"artifact_status": "complete", "total_preparations": resource.TOTAL_SHOTS,
                 "circuit_manifest": {"sha256": _hash(manifest)}}
    (tmp_path / "runs.json").write_text(json.dumps(runs) + "\n")
    (tmp_path / "resources.json").write_text(json.dumps(resources) + "\n")
    return tmp_path


def test_loader_binds_executed_module_and_manifest_before_replay(minimal_record):
    data, raw, implementation, digest = resource.load_inputs(minimal_record)
    assert digest == _hash(minimal_record / "optimization.py")
    assert set(raw) == {"runs", "resources", "circuit-manifest"}
    assert data["runs"]["recipe"] == implementation.quality_scaling_recipe()


@pytest.mark.parametrize("corruption", [
    "module_bytes", "manifest_bytes", "source_hash", "logical_total",
    "sampled_total", "partial_runs", "partial_logical", "custom_recipe",
])
def test_loader_rejects_detached_or_incomplete_inputs(minimal_record, corruption):
    if corruption in {"module_bytes", "manifest_bytes"}:
        target = "optimization.py" if corruption == "module_bytes" else "circuit-manifest.json"
        with (minimal_record / target).open("a") as handle:
            handle.write("\n")
    else:
        name = "resources" if corruption in {"logical_total", "partial_logical"} else "runs"
        path = minimal_record / f"{name}.json"
        data = json.loads(path.read_text())
        if corruption == "source_hash":
            data["provenance"]["implementation_sha256"] = "0" * 64
        elif corruption == "logical_total":
            data["total_preparations"] -= 1
        elif corruption == "sampled_total":
            data["total_shots"] -= 1
        elif corruption in {"partial_runs", "partial_logical"}:
            data["artifact_status"] = "pending"
        else:
            data["recipe"]["fixed_published_recipe"] = False
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        resource.load_inputs(minimal_record)


def _budget_fixture():
    mapping = []
    signatures = {}
    shots = [resource.TOTAL_SHOTS // 3, resource.TOTAL_SHOTS - resource.TOTAL_SHOTS // 3]
    for index, number in enumerate(shots):
        mapping.append({"batch_id": f"batch{index}", "task": "qaoa", "shots": number,
                        "cost_signature_by_profile": {}})
    for profile in resource.PROFILES:
        for index in range(2):
            config = {
                "profile": asdict(profile), "task": "qaoa", "qdk_version": "1.34.0",
                "whole_study_shots": resource.TOTAL_SHOTS,
                "whole_study_physical_failure_allowance": resource.PHYSICAL_ALLOWANCE,
                "nominal_study_physical_failure_per_execution": resource.PHYSICAL_ALLOWANCE / resource.TOTAL_SHOTS,
                "synthesis_model_allowance_per_execution": resource.SYNTHESIS_ALLOWANCE,
            }
            snapshot = {"independent_fixture": {"compute_qubits": 3, "rotation_count": 5 + index}}
            sid = resource.helper.cost_signature_key(snapshot, config)
            signatures[sid] = {"id": sid, "model_config": config, "cost_snapshot": snapshot}
            mapping[index]["cost_signature_by_profile"][profile.name] = sid
    return mapping, signatures


def test_budgets_allocate_one_uniform_per_preparation_study_allowance():
    mapping, signatures = _budget_fixture()
    allocations = resource.allocate_budgets(mapping, signatures)
    for sid, allocation in allocations.items():
        assert allocation["synthesis_model_error_per_execution"] == 1e-5
        assert allocation["physical_failure_probability"] / allocation["executions"] == pytest.approx(
            1e-3 / 5_859_840, rel=1e-14)
        assert signatures[sid]["batch_ids"] in [["batch0"], ["batch1"]]
    for profile in resource.PROFILES:
        selected = [allocation for sid, allocation in allocations.items()
                    if signatures[sid]["model_config"]["profile"]["name"] == profile.name]
        assert sum(x["executions"] for x in selected) == 5_859_840
        assert sum(x["physical_failure_probability"] for x in selected) == pytest.approx(1e-3, rel=1e-14)


@pytest.mark.parametrize("corruption", ["duplicate_id", "missing_profile", "missing_batch", "unknown_signature"])
def test_allocation_rejects_missing_or_duplicated_mapping(corruption):
    mapping, signatures = _budget_fixture()
    if corruption == "duplicate_id":
        mapping[1]["batch_id"] = mapping[0]["batch_id"]
    elif corruption == "missing_profile":
        del mapping[0]["cost_signature_by_profile"][resource.PROFILES[0].name]
    elif corruption == "missing_batch":
        mapping.pop()
    else:
        mapping[0]["cost_signature_by_profile"][resource.PROFILES[0].name] = "unknown"
    with pytest.raises((ValueError, KeyError)):
        resource.allocate_budgets(mapping, signatures)


@pytest.mark.parametrize("corruption", ["profile", "shots", "fault", "synthesis", "per_execution"])
def test_allocation_rejects_a_stale_cost_model_contract(corruption):
    mapping, signatures = _budget_fixture()
    config = next(iter(signatures.values()))["model_config"]
    if corruption == "profile":
        config["profile"]["physical_error_rate"] = 5e-4
    elif corruption == "shots":
        config["whole_study_shots"] -= 1
    elif corruption == "fault":
        config["whole_study_physical_failure_allowance"] = 2e-3
    elif corruption == "synthesis":
        config["synthesis_model_allowance_per_execution"] = 2e-5
    else:
        config["nominal_study_physical_failure_per_execution"] *= 2
    with pytest.raises(ValueError):
        resource.allocate_budgets(mapping, signatures)


@pytest.mark.parametrize("a,b,expected", [
    ({"distance": 25, "error": 2.4698959469631262e-12}, {"distance": 25, "error": 2.4698959469631266e-12}, True),
    ({"distance": 25}, {"distance": 27}, False),
    ({"count": 25}, {"count": 25.0}, False),
    ([1, 2], [2, 1], False),
    ({"a": 1}, {"a": 1, "b": 2}, False),
    (0.0, 1e-300, False),
    (1.0, 1.0 + 1e-9, False),
    (float("nan"), float("nan"), False),
])
def test_cost_fingerprint_tolerance_changes_only_finite_roundoff_fields(a, b, expected):
    assert resource.numeric_equivalent(a, b) is expected


def _fake_reports():
    reports = {}
    executions = [resource.TOTAL_SHOTS // 3, resource.TOTAL_SHOTS - resource.TOTAL_SHOTS // 3]
    for profile in resource.PROFILES:
        for index, count in enumerate(executions):
            def point(qubits, time):
                return {"peak_physical_qubits": qubits, "runtime_per_execution_ns": time,
                        "physical_failure_model_per_execution": 2e-11}
            sid = f"{profile.name}-{index}"
            reports[sid] = {"signature_id": sid, "profile": asdict(profile), "executions": count,
                            "report": {"pareto_points": [point(100 + index * 70, 40 + index * 20),
                                                          point(500 + index * 70, 10 + index * 5)]}}
    return reports


def test_serial_aggregation_weights_time_faults_and_takes_peak_maximum():
    reports = _fake_reports()
    choices = resource.aggregate_choices(reports)
    assert len(choices) == 4
    for choice in choices:
        selected = choice["class_selections"]
        assert sum(x["executions"] for x in selected) == resource.TOTAL_SHOTS
        assert choice["serial_quantum_model_runtime_ns"] == sum(
            x["executions"] * x["point"]["runtime_per_execution_ns"] for x in selected)
        assert choice["peak_physical_qubits"] == max(x["point"]["peak_physical_qubits"] for x in selected)
        assert choice["whole_study_physical_failure_union_bound"] == pytest.approx(resource.TOTAL_SHOTS * 2e-11)
        assert choice["common_configuration_verified"] is False
        assert choice["reset_feedback_reconfiguration_overhead_ns"] is None
        assert choice["joint_pareto_frontier_claimed"] is False
        if choice["selection"].startswith("fewest"):
            assert choice["peak_physical_qubits"] == 170
        else:
            assert choice["peak_physical_qubits"] == 570


def test_aggregation_rejects_an_incomplete_study_or_excess_whole_fault_score():
    reports = _fake_reports()
    first = next(iter(reports.values()))
    first["executions"] -= 1
    with pytest.raises(ValueError):
        resource.aggregate_choices(reports)
    reports = _fake_reports()
    for entry in reports.values():
        for point in entry["report"]["pareto_points"]:
            point["physical_failure_model_per_execution"] = 1e-6
    with pytest.raises(ValueError):
        resource.aggregate_choices(reports)


@pytest.mark.parametrize("algorithm,group", [("qaoa", "all_z"), ("vqe", "all_z"), ("vqe", "all_x")])
def test_prepared_state_and_readout_check_has_numerical_scope_only(algorithm, group):
    case = next(c for c in opt.quality_scaling_cases() if c["algorithm"] == algorithm)
    source = opt.quality_scaling_group_circuit(case, case["initial"], group)
    lowered = resource.prepare_measured_circuit(source)
    independent = resource.independent_prepared_state(opt, case, case["initial"], group)
    checked = resource.prepared_equivalence(source, lowered, independent)
    assert checked["prepared_state_verified"] is True
    assert checked["full_operator_verified"] is False
    assert checked["up_to_global_phase"] is True
    assert max(checked["max_source_lowered_amplitude_error"], checked["max_independent_source_amplitude_error"]) < 1e-11
    assert checked["measurement_map"] == [[i, i] for i in range(source.num_qubits)]


def test_prepared_state_check_rejects_changed_readout_or_amplitudes():
    case = next(c for c in opt.quality_scaling_cases() if c["algorithm"] == "vqe")
    source = opt.quality_scaling_group_circuit(case, case["initial"], "all_z")
    independent = resource.independent_prepared_state(opt, case, case["initial"], "all_z")
    changed = source.remove_final_measurements(inplace=False)
    changed.x(0)
    changed.measure_all()
    with pytest.raises(ValueError):
        resource.prepared_equivalence(source, changed, independent)
    reversed_readout = source.remove_final_measurements(inplace=False)
    reversed_readout.add_register(source.cregs[0])
    reversed_readout.measure(0, 1)
    reversed_readout.measure(1, 0)
    with pytest.raises(ValueError):
        resource.prepared_equivalence(source, reversed_readout, independent)


def test_global_phase_is_ignored_but_orthogonal_state_is_not():
    a = np.array([1.0, 0.0], dtype=complex)
    assert resource.aligned_error(a, 1j * a) < 1e-12
    assert resource.aligned_error(a, np.array([0.0, 1.0])) == float("inf")


def test_readout_map_rejects_missing_duplicate_or_mid_circuit_measurements():
    missing = QuantumCircuit(2, 2)
    missing.measure(0, 0)
    duplicate = QuantumCircuit(2, 2)
    duplicate.measure(0, 0)
    duplicate.measure(1, 0)
    mid = QuantumCircuit(2, 2)
    mid.measure(0, 0)
    mid.x(1)
    mid.measure(1, 1)
    for circuit in (missing, duplicate, mid):
        with pytest.raises(ValueError):
            resource.measurement_map(circuit)


def test_prepared_check_rejects_width_over_cap_before_allocating_state(monkeypatch):
    source = QuantumCircuit(9)
    source.measure_all()
    def forbidden(*args, **kwargs):
        raise AssertionError("statevector allocation must follow the width guard")
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    with pytest.raises(ValueError):
        resource.prepared_equivalence(source, source.copy(), np.ones(512))


def test_optional_git_metadata_is_null_without_exact_root_helper(tmp_path, monkeypatch):
    monkeypatch.setattr(resource, "ROOT", tmp_path)
    assert resource.optional_git_provenance() == {"source_commit": None, "owned_source_dirty": None, "git_helper_sha256": None}


def _binding_fixture():
    case = opt.quality_scaling_cases()[0]
    run = {"seed": 7, "run_index": 0, "evaluations": [], "validation": None}
    rows = []
    for phase, family in (("training", 0), ("validation", 1)):
        call = opt.sampled_qaoa_layers_objective(
            case["initial"], n_qubits=case["n_qubits"], edges=case["edges"],
            shots=16, seed=7, evaluation_index=0, family=family, stream_id=case["stream_id"])
        group = call["groups"][0]
        circuit = opt.quality_scaling_group_circuit(case, call["parameters"], group["group"])
        sha = resource.text_digest(qasm3.dumps(circuit))
        group.update(source_qasm_sha256=sha, source_qasm_file=f"circuits/{sha}.qasm")
        row = {"index": len(rows), "case_id": case["case_id"], "algorithm": case["algorithm"],
               "n_qubits": case["n_qubits"], "p": case["p"], "seed": run["seed"], "run_index": 0,
               "phase": phase, "evaluation_index": 0, "group_index": 0, "group": group["group"],
               "parameters": deepcopy(call["parameters"]),
               "source_gate_counts": deepcopy(group["circuit_gate_counts"]), "source_depth": group["circuit_depth"]}
        for key in ("shots", "counts", "estimate", "sample_variance", "estimated_standard_error",
                    "outcome_min", "outcome_max", "rng", "source_qasm_sha256", "source_qasm_file"):
            row[key] = deepcopy(group[key])
        rows.append(row)
        if phase == "training":
            run["evaluations"].append(call)
        else:
            run["validation"] = call
    return {"runs": {"cases": [{**case, "runs": [run]}]}, "circuit-manifest": {"groups": rows}}


def test_manifest_binding_uses_the_recorded_calls_groups_and_statistics():
    data = _binding_fixture()
    assert resource.bind_recorded_groups(data) == data["circuit-manifest"]["groups"]


@pytest.mark.parametrize("corruption", [
    "parameters", "seed", "phase", "group_index", "group", "counts", "shots", "estimate",
    "sample_variance", "rng", "source_hash", "source_file", "source_counts", "source_depth",
    "missing_group", "duplicate_group",
])
def test_manifest_cannot_supply_a_different_rehashed_parameter_path(corruption):
    data = _binding_fixture()
    row = data["circuit-manifest"]["groups"][0]
    if corruption == "parameters":
        row["parameters"][0] += .4
    elif corruption == "seed":
        row["seed"] = 19
    elif corruption == "phase":
        row["phase"] = "validation"
    elif corruption == "group_index":
        row["group_index"] = 1
    elif corruption == "group":
        row["group"] = "all_x"
    elif corruption == "counts":
        row["counts"][next(iter(row["counts"]))] += 1
    elif corruption == "shots":
        row["shots"] += 1
    elif corruption == "estimate":
        row["estimate"] += 1
    elif corruption == "sample_variance":
        row["sample_variance"] += 1
    elif corruption == "rng":
        row["rng"]["entropy"][0] += 1
    elif corruption == "source_hash":
        row["source_qasm_sha256"] = "a" * 64
    elif corruption == "source_file":
        row["source_qasm_file"] = "other.qasm"
    elif corruption == "source_counts":
        row["source_gate_counts"]["h"] += 1
    elif corruption == "source_depth":
        row["source_depth"] += 1
    elif corruption == "missing_group":
        data["circuit-manifest"]["groups"].pop()
    else:
        data["circuit-manifest"]["groups"].append(deepcopy(row))
    with pytest.raises(ValueError):
        resource.bind_recorded_groups(data)


@pytest.mark.parametrize("limit", [-1, 97, True, .5])
def test_native_search_cap_rejects_invalid_values_before_input_work(limit, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("query cap must be checked before input work")
    monkeypatch.setattr(resource, "load_inputs", forbidden)
    with pytest.raises(ValueError):
        resource.generate(tmp_path, tmp_path / "out", tmp_path / "cache", max_queries=limit)


def test_cost_class_uses_whole_study_allocation_not_per_algorithm_allowances(monkeypatch):
    seen = {}
    def fake(qasm, *, task, task_shots, profile):
        seen.update(qasm=qasm, task=task, task_shots=task_shots)
        return {"model_config": {
            "task": task, "profile": asdict(profile), "task_total_shots": task_shots,
            "task_physical_failure_allowance": resource.PHYSICAL_ALLOWANCE,
            "nominal_physical_failure_allowance_per_execution": resource.PHYSICAL_ALLOWANCE / task_shots,
        }, "cost_snapshot": {"fixed_trace": 1}}
    monkeypatch.setattr(resource.helper, "canonical_cost_signature", fake)
    signature = resource.cost_signature("test-only-input", "vqe", resource.PROFILES[0])
    config = signature["model_config"]
    assert seen["task_shots"] == 5_859_840
    assert config["whole_study_shots"] == 5_859_840
    assert config["whole_study_physical_failure_allowance"] == 1e-3
    assert config["nominal_study_physical_failure_per_execution"] == pytest.approx(1e-3 / 5_859_840)
    assert "task_total_shots" not in config
    assert signature["id"] == resource.helper.cost_signature_key(signature["cost_snapshot"], config)


def test_ledger_rejects_a_subset_before_any_circuit_reconstruction(monkeypatch, tmp_path):
    data = _binding_fixture()
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete group collection must fail before reconstruction")
    monkeypatch.setattr(opt, "quality_scaling_group_circuit", forbidden)
    with pytest.raises(ValueError):
        resource.build_ledger(data, opt, tmp_path, tmp_path / "out", float("inf"))


def _mock_generation_audit(monkeypatch, tmp_path, *, classes=42, distinct_probe=False):
    """Mock source/trace audits without launching physical searches."""
    folder = tmp_path / "inputs"
    folder.mkdir()
    runs = {"cases": [{"case_id": "fixed-case", "runs": []}],
            "workflow_process_cpu_seconds": .2, "workflow_wall_seconds": .3}
    data = {"runs": runs, "resources": {}, "circuit-manifest": {}}
    raw = {name: json.dumps(value).encode() for name, value in data.items()}
    for name, content in raw.items():
        (folder / f"{name}.json").write_bytes(content)
    (folder / "optimization.py").write_text("pass\n")
    source_hash = _hash(folder / "optimization.py")
    implementation = SimpleNamespace(quality_scaling_group_circuit=lambda case, parameters, group: parameters[0])
    monkeypatch.setattr(resource, "load_inputs", lambda _: (data, raw, implementation, source_hash))
    mapping, unique = [], {}
    assert resource.TOTAL_SHOTS % classes == 0
    for index in range(classes):
        sha = resource.text_digest(f"fixed-source-{index}")
        key = f"qaoa:{sha}"
        unique[key] = {"task": "qaoa", "lowered_qasm": f"trace{index}",
                       "lowered_qasm_sha256": resource.text_digest(f"lowered-{index}"),
                       "case_id": "fixed-case", "parameters": [f"trace{index}"], "group": "all_z"}
        mapping.append({"batch_id": f"batch{index}", "task": "qaoa", "shots": resource.TOTAL_SHOTS // classes,
                        "source_qasm_sha256": sha, "cost_signature_by_profile": {}})
    if distinct_probe:
        unique["qaoa:distinct-source"] = deepcopy(next(iter(unique.values())))
        unique["qaoa:distinct-source"]["parameters"] = ["alternate-trace0"]
        unique["qaoa:distinct-source"]["lowered_qasm_sha256"] = resource.text_digest("alternate-lowered-0")
    ledger = {"mapping": mapping, "unique": unique, "training_shots": 4_483_584,
              "validation_shots": 1_376_256, "total_shots": resource.TOTAL_SHOTS}
    monkeypatch.setattr(resource, "build_ledger", lambda *args: ledger)
    def fake_signature(qasm, task, profile):
        config = {"task": task, "profile": asdict(profile), "whole_study_shots": resource.TOTAL_SHOTS,
                  "whole_study_physical_failure_allowance": resource.PHYSICAL_ALLOWANCE,
                  "nominal_study_physical_failure_per_execution": resource.PHYSICAL_ALLOWANCE / resource.TOTAL_SHOTS,
                  "synthesis_model_allowance_per_execution": resource.SYNTHESIS_ALLOWANCE}
        snapshot = {"independently_indexed_trace": qasm}
        return {"id": resource.helper.cost_signature_key(snapshot, config), "model_config": config,
                "cost_snapshot": snapshot, "raw_trace": {}, "psspc_trace": {}}
    monkeypatch.setattr(resource, "cost_signature", fake_signature)
    return folder, tmp_path / "outputs", ledger, fake_signature


def test_exceeding_complete_cost_class_cap_releases_no_partial_physical_result(monkeypatch, tmp_path):
    """Forty-two classes per frozen profile cannot silently select a subset."""
    folder, output, _, _ = _mock_generation_audit(monkeypatch, tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("no physical search may run without complete cap coverage")
    monkeypatch.setattr(resource, "estimate_circuit_resources", forbidden)
    record = resource.generate(folder, output, tmp_path / "cache", max_queries=64, max_wall_seconds=1200)
    assert record["status"] == "pending_physical_layer"
    assert record["planned_native_queries"] == 84
    assert record["native_query_count"] == 0
    assert record["cost_signature_count"] == 84
    assert record["total_preparations"] == 5_859_840
    assert record["choices"] == []
    assert record["wall_allowance_seconds"] == 1200
    assert json.loads((output / "projection-reports.json").read_text())["reports"] == {}
    assert "No aggregate physical result is released" in (output / "summary.md").read_text()


@pytest.mark.parametrize("limit", [0, -1, 1801, True, "1200", float("inf"), float("nan")])
@pytest.mark.parametrize("driver", ["generate", "resume_archived_audit"])
def test_wall_allowance_rejects_invalid_values_before_input_work(limit, driver, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("wall allowance must be validated before input work")
    monkeypatch.setattr(resource, "load_inputs", forbidden)
    monkeypatch.setattr(resource, "validate_retained_audit", forbidden)
    args = [tmp_path, tmp_path / "out", tmp_path / "cache"]
    if driver == "resume_archived_audit":
        args.insert(1, tmp_path / "archive")
    with pytest.raises(ValueError, match="wall allowance"):
        getattr(resource, driver)(*args, max_wall_seconds=limit)


@pytest.mark.parametrize("stage", ["before", "after"])
def test_signature_audits_stop_at_deadline_before_any_native_search(stage, monkeypatch, tmp_path):
    folder, output, ledger, fake_signature = _mock_generation_audit(monkeypatch, tmp_path, classes=1)
    clock, audits = [0.0], []
    monkeypatch.setattr(resource, "perf_counter", lambda: clock[0])
    if stage == "before":
        def expired_ledger(*args):
            clock[0] = 2.0
            return ledger
        monkeypatch.setattr(resource, "build_ledger", expired_ledger)
    def expired_signature(*args):
        audits.append(args)
        clock[0] = 2.0
        return fake_signature(*args)
    monkeypatch.setattr(resource, "cost_signature", expired_signature)
    def forbidden(*args, **kwargs):
        raise AssertionError("an incomplete signature audit cannot launch a native search")
    monkeypatch.setattr(resource, "estimate_circuit_resources", forbidden)
    with pytest.raises(RuntimeError, match=f"{stage} a trace-signature audit"):
        resource.generate(folder, output, tmp_path / "cache", max_wall_seconds=1)
    assert len(audits) == int(stage == "after")
    assert not (output / "manifest.json").exists()


def test_native_query_overrun_releases_no_completed_physical_result(monkeypatch, tmp_path):
    folder, output, _, _ = _mock_generation_audit(monkeypatch, tmp_path, classes=1)
    clock, searches = [0.0], []
    monkeypatch.setattr(resource, "perf_counter", lambda: clock[0])
    def overrun(source, **kwargs):
        searches.append(source)
        clock[0] = 2.0
        return {}
    monkeypatch.setattr(resource, "estimate_circuit_resources", overrun)
    record = resource.generate(folder, output, tmp_path / "cache", max_wall_seconds=1)
    assert searches == ["trace0"]
    assert record["native_query_count"] == 1
    assert record["status"] == "pending_physical_layer"
    assert "during a native query" in record["pending_reason"]
    assert record["choices"] == []
    assert json.loads((output / "projection-reports.json").read_text())["reports"] == {}


@pytest.mark.parametrize("overrun_stage", ["query", "reparse", "aggregate", "provenance"])
def test_probe_and_final_verification_overruns_cannot_be_labeled_complete(overrun_stage, monkeypatch, tmp_path):
    folder, output, ledger, fake_signature = _mock_generation_audit(monkeypatch, tmp_path, classes=1, distinct_probe=True)
    clock, searches = [0.0], []
    monkeypatch.setattr(resource, "perf_counter", lambda: clock[0])
    def estimate(source, **kwargs):
        searches.append(source)
        if source == "alternate-trace0":
            item = ledger["unique"]["qaoa:distinct-source"]
            if overrun_stage == "query":
                clock[0] = 2.0
        else:
            item = next(iter(ledger["unique"].values()))
        return {"qasm_sha256": item["lowered_qasm_sha256"], "qasm3": source}
    monkeypatch.setattr(resource, "estimate_circuit_resources", estimate)
    def signature(qasm, task, profile):
        if qasm == "alternate-trace0" and overrun_stage == "reparse":
            clock[0] = 2.0
        return fake_signature("trace0", task, profile)
    monkeypatch.setattr(resource, "cost_signature", signature)
    monkeypatch.setattr(resource.helper, "projection_cost_fingerprint", lambda _: {"fixed_cost": 1})
    def aggregate(reports):
        if overrun_stage == "aggregate":
            clock[0] = 2.0
        return [{"test_only_aggregate": True}]
    monkeypatch.setattr(resource, "aggregate_choices", aggregate)
    real_digest = resource.digest
    def digest(path):
        if Path(path) == folder / "optimization.py" and overrun_stage == "provenance":
            clock[0] = 2.0
        return real_digest(path)
    monkeypatch.setattr(resource, "digest", digest)
    record = resource.generate(folder, output, tmp_path / "cache", max_wall_seconds=1)
    assert searches == ["trace0", "trace0", "alternate-trace0"]
    assert record["native_query_count"] == record["planned_native_queries"] == 3
    assert record["status"] == "pending_physical_layer"
    assert record["choices"] == []
    assert json.loads((output / "projection-reports.json").read_text())["reports"] == {}


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("status", ["pending_physical_layer", "computed_conditional_recorded_replay"])
@pytest.mark.parametrize("strict", [False, True])
def test_cli_requires_completion_only_when_requested(resume, status, strict, monkeypatch, capsys):
    calls = []
    def fake(*args):
        calls.append(args)
        return {"status": status, "pending_reason": "bounded test pending" if status.startswith("pending") else None,
                "native_query_count": 0, "cost_signature_count": 84, "total_preparations": resource.TOTAL_SHOTS,
                "replay_wall_seconds": .1}
    driver = "resume_archived_audit" if resume else "generate"
    monkeypatch.setattr(resource, driver, fake)
    args = ["--max-queries", "96", "--max-wall-seconds", "1200"]
    if resume:
        args += ["--resume-archive", "test-only-archive"]
    if strict:
        args += ["--require-complete"]
    if strict and status.startswith("pending"):
        with pytest.raises(SystemExit, match="Complete quality-resource replay required") as error:
            resource.main(args)
        assert error.value.code != 0
    else:
        resource.main(args)
    assert calls[0][-2:] == (96, 1200.0)
    assert json.loads(capsys.readouterr().out)["status"] == status


@pytest.mark.parametrize("limit", [64, 84, 97, True, .5])
def test_amended_resume_rejects_undercoverage_or_unauthorized_cap_before_validation(limit, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("amended cap must be checked before retained-audit work")
    monkeypatch.setattr(resource, "validate_retained_audit", forbidden)
    with pytest.raises(ValueError):
        resource.resume_archived_audit(tmp_path, tmp_path, tmp_path / "out", tmp_path / "cache", max_queries=limit)


def _mock_resume_audit(tmp_path):
    """Use retained-audit mocks to isolate amendment and checkpoint sequencing."""
    folder, archive, output = (tmp_path / name for name in ("inputs", "archive", "outputs"))
    folder.mkdir()
    archive.mkdir()
    data = {"runs": {"cases": [{"case_id": "fixed-case"}]}, "resources": {}, "circuit-manifest": {}}
    raw = {name: json.dumps(value).encode() for name, value in data.items()}
    for name, content in raw.items():
        (folder / f"{name}.json").write_bytes(content)
    old = {"query_limit": 64, "native_query_count": 0, "cost_signature_count": 84,
           "planned_native_queries": 85, "prepared_state_checks": 1873,
           "audit_counts": {"canonical_trace_signature_audits": 3746},
           "source_provenance": {"input_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in raw.items()},
                                 "resource_generator_sha256": "a" * 64}}
    (archive / "manifest.json").write_text(json.dumps(old))
    (archive / "payload.txt").write_text("frozen earlier evidence\n")
    (archive / "archive.json").write_text(json.dumps({
        "original_files_sha256": {name: _hash(archive / name) for name in ("manifest.json", "payload.txt")}}))
    implementation = SimpleNamespace(quality_scaling_group_circuit=lambda *args: object())
    unique, signatures, budgets = {}, {}, {}
    for index in range(84):
        sid, key = f"signature{index}", f"variant{index}"
        keys = [key, "distinct-probe"] if index == 0 else [key]
        profile = resource.PROFILES[index % 2]
        signatures[sid] = {"variant_keys": keys, "model_config": {"profile": asdict(profile)}}
        budgets[sid] = {"synthesis_model_error_per_execution": resource.SYNTHESIS_ALLOWANCE,
                        "physical_failure_probability": resource.PHYSICAL_ALLOWANCE / 42,
                        "executions": resource.TOTAL_SHOTS // 42}
        unique[key] = {"case_id": "fixed-case", "parameters": [.2, .1], "group": "all_z", "task": "qaoa",
                       "lowered_qasm_sha256": "b" * 64}
    unique["distinct-probe"] = deepcopy(unique["variant0"])
    audit = (data, raw, implementation, "c" * 64, old, unique, signatures, {}, budgets)
    before = {path.relative_to(archive).as_posix(): path.read_bytes() for path in archive.iterdir()}
    return folder, archive, output, audit, before


def test_amendment_is_written_before_first_search_and_failed_attempt_is_checkpointed(monkeypatch, tmp_path):
    folder, archive, output, audit, before = _mock_resume_audit(tmp_path)
    monkeypatch.setattr(resource, "validate_retained_audit", lambda *args: audit)
    seen = []
    def failed_search(source, *, budget, profile, **kwargs):
        amendment_path = output / "amendment.json"
        assert amendment_path.is_file()
        amendment = json.loads(amendment_path.read_text())
        assert amendment["recorded_before_new_searches"] is True
        assert amendment["previous_query_limit"] == 64
        assert amendment["new_query_limit"] == 96
        assert amendment["planned_physical_estimator_searches"] == 85
        assert amendment["cases_seeds_targets_shots_profiles_error_allowances_changed"] is False
        assert amendment["input_sha256"] == audit[4]["source_provenance"]["input_sha256"]
        assert amendment["old_manifest_sha256"] == _hash(archive / "manifest.json")
        assert budget.physical_failure_per_execution == pytest.approx(1e-3 / 5_859_840, rel=1e-14)
        assert profile in resource.PROFILES
        seen.append(amendment)
        raise ValueError("deliberate bounded mock failure")
    monkeypatch.setattr(resource, "estimate_circuit_resources", failed_search)
    result = resource.resume_archived_audit(folder, archive, output, tmp_path / "cache", max_queries=96)
    assert len(seen) == 1
    assert result["native_query_count"] == 1
    assert result["status"] == "pending_physical_layer"
    assert result["choices"] == []
    assert result["failed_search"]["query_number"] == 1
    checkpoint = json.loads((output / "query-checkpoint.json").read_text())
    assert checkpoint["physical_estimator_search_requests"] == 1
    assert checkpoint["completed_representative_reports"] == {}
    assert checkpoint["failed_search"]["error_type"] == "ValueError"
    assert checkpoint["amendment_sha256"] == _hash(output / "amendment.json")
    assert {path.relative_to(archive).as_posix(): path.read_bytes() for path in archive.iterdir()} == before
    assert {name: (folder / f"{name}.json").read_bytes() for name in audit[1]} == audit[1]


def test_amended_native_overrun_is_checkpointed_without_trace_reparse(monkeypatch, tmp_path):
    folder, archive, output, audit, before = _mock_resume_audit(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(resource, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(resource, "validate_retained_audit", lambda *args: audit)
    def overrun(*args, **kwargs):
        clock[0] = 2.0
        return {}
    def forbidden(*args, **kwargs):
        raise AssertionError("an expired native query cannot start a trace reparse")
    monkeypatch.setattr(resource, "estimate_circuit_resources", overrun)
    monkeypatch.setattr(resource, "cost_signature", forbidden)
    result = resource.resume_archived_audit(folder, archive, output, tmp_path / "cache", max_wall_seconds=1)
    assert result["status"] == "pending_physical_layer"
    assert result["native_query_count"] == 1
    assert result["choices"] == []
    assert result["wall_allowance_seconds"] == 1
    assert "during an amended native query" in result["failed_search"]["reason"]
    checkpoint = json.loads((output / "query-checkpoint.json").read_text())
    assert checkpoint["physical_estimator_search_requests"] == 1
    assert checkpoint["completed_representative_reports"] == {}
    assert {path.relative_to(archive).as_posix(): path.read_bytes() for path in archive.iterdir()} == before


def test_amended_final_provenance_overrun_clears_completed_status_and_checkpoint(monkeypatch, tmp_path):
    folder, archive, output, audit, before = _mock_resume_audit(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(resource, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(resource, "validate_retained_audit", lambda *args: audit)
    unique, signatures = audit[5:7]
    for signature in signatures.values():
        signature["cost_snapshot"] = {"fixed_mock_trace": True}
    for key, item in unique.items():
        item["parameters"] = [key]
    audit[2].quality_scaling_group_circuit = lambda case, parameters, group: parameters[0]
    def estimate(source, *, budget, **kwargs):
        return {"qasm3": source, "qasm_sha256": unique[source]["lowered_qasm_sha256"], "budget": asdict(budget)}
    def signature(qasm, task, profile):
        return {"cost_snapshot": {"fixed_mock_trace": True}, "model_config": {"profile": asdict(profile)}}
    def expired_provenance():
        clock[0] = 2.0
        return {}
    monkeypatch.setattr(resource, "estimate_circuit_resources", estimate)
    monkeypatch.setattr(resource, "cost_signature", signature)
    monkeypatch.setattr(resource.helper, "projection_cost_fingerprint", lambda _: {"fixed_mock_cost": True})
    monkeypatch.setattr(resource, "aggregate_choices", lambda _: [{"fixture_only": True}])
    monkeypatch.setattr(resource, "optional_git_provenance", expired_provenance)
    result = resource.resume_archived_audit(folder, archive, output, tmp_path / "cache", max_wall_seconds=1)
    assert result["native_query_count"] == 85
    assert result["status"] == "pending_physical_layer"
    assert "final provenance verification" in result["pending_reason"]
    assert result["choices"] == []
    assert result["amended_validation_and_search_wall_seconds"] == 2
    checkpoint = json.loads((output / "query-checkpoint.json").read_text())
    assert checkpoint["pending_reason"] == result["pending_reason"]
    assert len(checkpoint["completed_representative_reports"]) == 84
    assert json.loads((output / "projection-reports.json").read_text())["reports"] == {}
    assert {path.relative_to(archive).as_posix(): path.read_bytes() for path in archive.iterdir()} == before


def test_existing_amendment_refuses_silent_repeated_searches(monkeypatch, tmp_path):
    folder, archive, output, audit, _ = _mock_resume_audit(tmp_path)
    output.mkdir()
    original = b'{"previous_attempt":true}\n'
    (output / "amendment.json").write_bytes(original)
    monkeypatch.setattr(resource, "validate_retained_audit", lambda *args: audit)
    def forbidden(*args, **kwargs):
        raise AssertionError("an existing amendment needs inspection before more searches")
    monkeypatch.setattr(resource, "estimate_circuit_resources", forbidden)
    with pytest.raises(ValueError):
        resource.resume_archived_audit(folder, archive, output, tmp_path / "cache", max_queries=96)
    assert (output / "amendment.json").read_bytes() == original


def test_retained_archive_hash_change_fails_before_input_or_native_work(monkeypatch, tmp_path):
    _, archive, _, _, _ = _mock_resume_audit(tmp_path)
    (archive / "payload.txt").write_text("changed evidence\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("archive integrity must precede input or native work")
    monkeypatch.setattr(resource, "load_inputs", forbidden)
    monkeypatch.setattr(resource.helper, "canonical_cost_signature", forbidden)
    monkeypatch.setattr(resource, "estimate_circuit_resources", forbidden)
    with pytest.raises(ValueError):
        resource.validate_retained_audit(tmp_path, archive)


@pytest.mark.parametrize("changed", ["recorded_source", "input_bytes", "adapter", "trace_helper", "lock", "packages"])
def test_retained_audit_refuses_source_or_environment_retuning(changed, monkeypatch, tmp_path):
    folder, archive, _, audit, _ = _mock_resume_audit(tmp_path)
    data, raw, implementation, source_hash = audit[:4]
    old = audit[4]
    provenance = old["source_provenance"]
    provenance.update(recorded_implementation_sha256=source_hash,
                      shared_adapter_sha256=_hash(resource.ROOT / "src/openquantum/resource_estimation.py"),
                      audited_trace_helper_sha256=_hash(resource.HELPER_PATH),
                      environment_lock_sha256=_hash(resource.ROOT / "requirements-lock.txt"),
                      packages={name: resource.importlib.metadata.version(name) for name in ("qdk", "qiskit", "numpy")})
    (archive / "manifest.json").write_text(json.dumps(old))
    inventory = json.loads((archive / "archive.json").read_text())
    inventory["original_files_sha256"]["manifest.json"] = _hash(archive / "manifest.json")
    (archive / "archive.json").write_text(json.dumps(inventory))
    if changed == "recorded_source":
        source_hash = "d" * 64
    elif changed == "input_bytes":
        raw = {**raw, "runs": raw["runs"] + b"\n"}
    else:
        field = {"adapter": "shared_adapter_sha256", "trace_helper": "audited_trace_helper_sha256",
                 "lock": "environment_lock_sha256", "packages": "packages"}[changed]
        if changed == "packages":
            provenance[field] = {**provenance[field], "qdk": "changed"}
        else:
            provenance[field] = "e" * 64
        (archive / "manifest.json").write_text(json.dumps(old))
        inventory["original_files_sha256"]["manifest.json"] = _hash(archive / "manifest.json")
        (archive / "archive.json").write_text(json.dumps(inventory))
    monkeypatch.setattr(resource, "load_inputs", lambda *args: (data, raw, implementation, source_hash))
    def forbidden(*args, **kwargs):
        raise AssertionError("no native work is allowed while retained audit bindings differ")
    monkeypatch.setattr(resource.helper, "canonical_cost_signature", forbidden)
    monkeypatch.setattr(resource, "estimate_circuit_resources", forbidden)
    with pytest.raises(ValueError):
        resource.validate_retained_audit(folder, archive)
