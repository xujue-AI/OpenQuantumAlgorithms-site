"""Guard fixed-readout provenance and serial physical aggregation without QRE."""

from copy import deepcopy
from collections import Counter
import hashlib
import importlib.util
from pathlib import Path

import pytest

from openquantum.optimization import run_sampled_demo

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "optimization_projection_script", ROOT / "scripts/generate_optimization_projections.py"
)
projection = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(projection)


@pytest.fixture(scope="module")
def valid_fixed_readout():
    data = run_sampled_demo(schedule=((0.4, 256, 1),), validation_shots=4096)
    data["provenance"] = {
        "implementation_sha256": hashlib.sha256((ROOT / "src/openquantum/optimization.py").read_bytes()).hexdigest()
    }
    data["projection_fixed_readout_selection"] = {
        "selection_rule": "first_predeclared_case", "master_seed": 7,
        "shots_per_group": 4096,
        "qaoa_parameters": data["qaoa"]["runs"][0]["parameters"].copy(),
        "vqe_parameters": data["vqe"]["runs"][0]["parameters"].copy(),
    }
    return data


def test_fixed_readout_selection_is_bound_to_the_actual_primary_holdout(valid_fixed_readout):
    selected = projection.select_fixed_readout(valid_fixed_readout)
    for algorithm in ("qaoa", "vqe"):
        validation = valid_fixed_readout[algorithm]["runs"][0]["validation"]
        assert selected[algorithm + "_parameters"] == validation["parameters"]
        assert selected[algorithm + "_validation"]["group_rng_streams"] == [group["rng"] for group in validation["groups"]]
    assert selected["shots_per_group"] == 4096
    assert selected["master_seed"] == valid_fixed_readout["qaoa"]["runs"][0]["seed"]


@pytest.mark.parametrize("corruption", [
    "detached_parameters", "wrong_seed", "training_stream_reused", "holdout_family_changed",
    "wrong_shot_allocation", "missing_vqe_basis", "counts_do_not_sum", "stale_source_hash",
    "changed_measurement_circuit", "exact_optimizer_substituted",
])
def test_detached_or_stale_projection_inputs_are_rejected(valid_fixed_readout, corruption):
    data = deepcopy(valid_fixed_readout)
    q_run = data["qaoa"]["runs"][0]
    q_validation = q_run["validation"]
    if corruption == "detached_parameters":
        data["projection_fixed_readout_selection"]["qaoa_parameters"][0] += 0.1
    elif corruption == "wrong_seed":
        q_run["seed"] = 19
    elif corruption == "training_stream_reused":
        q_run["evaluations"][0]["groups"][0]["rng"]["stream_key"] = q_validation["groups"][0]["rng"]["stream_key"].copy()
    elif corruption == "holdout_family_changed":
        q_validation["groups"][0]["rng"]["spawn_key"][0] = 0
    elif corruption == "wrong_shot_allocation":
        q_validation["groups"][0]["shots"] -= 1
    elif corruption == "missing_vqe_basis":
        data["vqe"]["runs"][0]["validation"]["groups"].pop()
    elif corruption == "counts_do_not_sum":
        counts = q_validation["groups"][0]["counts"]
        counts[next(iter(counts))] -= 1
    elif corruption == "stale_source_hash":
        data["provenance"]["implementation_sha256"] = "0" * 64
    elif corruption == "changed_measurement_circuit":
        data["vqe"]["runs"][0]["validation"]["groups"][1]["circuit_gate_counts"]["h"] = 0
    else:
        data["exact_diagnostics_used_for_optimization"] = True
    with pytest.raises(ValueError):
        projection.select_fixed_readout(data)


def test_fixed_variants_split_the_whole_task_fault_allowance(valid_fixed_readout):
    selection = projection.select_fixed_readout(valid_fixed_readout)
    variants = projection.fixed_variants(selection)
    assert [(variant["task"], variant["measurement_group"]) for variant in variants] == [
        ("qaoa", "all_z"), ("vqe", "all_z"), ("vqe", "all_x")]
    fault_by_task = Counter()
    for variant in variants:
        allowance = variant["physical_failure_all_executions_allowance"]
        fault_by_task[variant["task"]] += allowance
        expected = 1e-3 if variant["task"] == "qaoa" else 5e-4
        assert allowance == pytest.approx(expected)
        budget = projection.CircuitResourceBudget(
            synthesis_model_error_per_execution=projection.SYNTHESIS_MODEL_ALLOWANCE,
            physical_failure_probability=allowance, executions=4096,
        )
        assert budget.physical_failure_per_execution == pytest.approx(allowance / 4096)
        assert variant["circuit"].count_ops()["measure"] == variant["circuit"].num_qubits
    assert fault_by_task == pytest.approx({"qaoa": 1e-3, "vqe": 1e-3})
    assert {profile.physical_error_rate for profile in projection.PROFILES} == {1e-3, 1e-4}
    assert all(profile.gate_time_ns == 100 and profile.measurement_time_ns == 500
               for profile in projection.PROFILES)


def fake_reports():
    """Make explicit two-point tradeoffs for each required fixed variant."""
    variant_specs = [
        ("qaoa-all-z", "qaoa", 100, 500, 20, 5, 8e-4),
        ("vqe-all-z", "vqe", 200, 700, 30, 6, 3e-4),
        ("vqe-all-x", "vqe", 300, 900, 40, 7, 4e-4),
    ]
    reports = []
    for profile in projection.PROFILES:
        for variant_id, task, low_qubits, fast_qubits, low_time, fast_time, fault in variant_specs:
            points = [dict(peak_physical_qubits=low_qubits, serial_runtime_all_executions_ns=low_time,
                           physical_failure_union_bound_all_executions=fault),
                      dict(peak_physical_qubits=fast_qubits, serial_runtime_all_executions_ns=fast_time,
                           physical_failure_union_bound_all_executions=fault)]
            reports.append(dict(profile={"name": profile.name}, variant_id=variant_id, task=task,
                                report={"pareto_points": points}))
    return reports


def test_serial_aggregation_sums_time_and_faults_but_takes_maximum_peak():
    reports = fake_reports()
    choices = projection.aggregate_readout_choices(reports)
    assert Counter((choice["profile"], choice["task"]) for choice in choices) == {
        (profile.name, task): 2 for profile in projection.PROFILES for task in ("qaoa", "vqe")}
    for choice in choices:
        points = choice["group_selected_points"]
        assert choice["serial_runtime_all_groups_ns"] == sum(point["serial_runtime_all_executions_ns"] for point in points)
        assert choice["peak_physical_qubits_for_serial_groups"] == max(point["peak_physical_qubits"] for point in points)
        assert choice["physical_failure_union_bound_all_groups"] == pytest.approx(
            sum(point["physical_failure_union_bound_all_executions"] for point in points))
        assert choice["physical_failure_union_bound_all_groups"] <= projection.TASK_PHYSICAL_FAILURE_ALLOWANCE
        assert choice["modeled_executions"] == 4096 * len(points)
        assert choice["not_a_combined_pareto_optimization"] is True
        if choice["task"] == "vqe":
            assert len(points) == 2
            assert choice["peak_physical_qubits_for_serial_groups"] < sum(point["peak_physical_qubits"] for point in points)


def test_serial_aggregation_rejects_an_overallocated_whole_task_fault_score():
    reports = fake_reports()
    for report in reports:
        if report["task"] == "vqe":
            for point in report["report"]["pareto_points"]:
                point["physical_failure_union_bound_all_executions"] = 0.7e-3
    with pytest.raises(AssertionError):
        projection.aggregate_readout_choices(reports)
