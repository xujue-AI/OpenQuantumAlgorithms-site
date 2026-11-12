"""Narrow binary64 agreement for preserved factory-error probe evidence.

These tests exercise only fingerprint comparison, not QDK cost generation.
Every original numeric value remains evidence; tolerances apply solely to
positive finite factory error-rate leaves in the declared Pareto-point array.
"""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "optimization_probe_float_script",
    ROOT / "scripts/generate_optimization_trajectory_resources.py",
)
trajectory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trajectory)

FACTORY_PATH = "/declared_physical_pareto_points/0/factories/T/error_rate"
OBSERVED_RATE = 2.4698959469631262e-12


def fingerprint():
    return {
        "physical_profile": {
            "name": "frozen-p-1e-3", "physical_error_rate": 1e-3,
            "gate_time_ns": 100, "measurement_time_ns": 500,
        },
        "budget": {
            "executions": 4096, "physical_failure_probability": 1e-3,
            "synthesis_model_error_per_execution": 1e-5,
        },
        "synthesis": {
            "fixed_num_ts_per_rotation": 16,
            "model_error_per_execution": 3.765e-6,
            "actual_synthesis_sequence_generated": False,
        },
        "canonical_gate_counts": {"RZ": 8, "MEAS_Z": 4},
        "native_cost_trace": {"block": {"operations": ["RZ", "MEAS_Z"], "repetitions": 1}},
        "error_composition": "UnionBound",
        "declared_physical_pareto_points": [{
            "peak_physical_qubits": 1131935,
            "runtime_per_execution_ns": 2297400.0,
            "physical_failure_model_per_execution": 3.245e-10,
            "factories": {"T": {
                "copies": 8, "runs": 16, "states": 128,
                "error_rate": OBSERVED_RATE,
            }},
            "model_properties": {"COMPUTE_QUBITS": 4},
        }],
    }


def advance(value, steps, direction=math.inf):
    for _ in range(steps):
        value = math.nextafter(value, direction)
    return value


def compare_factory_rates(left_rate, right_rate):
    left, right = fingerprint(), fingerprint()
    left["declared_physical_pareto_points"][0]["factories"]["T"]["error_rate"] = left_rate
    right["declared_physical_pareto_points"][0]["factories"]["T"]["error_rate"] = right_rate
    return trajectory.compare_projection_cost_fingerprints(left, right)


def test_exact_valid_fingerprints_match_without_differences():
    value = fingerprint()
    result = trajectory.compare_projection_cost_fingerprints(value, deepcopy(value))
    assert result["matches"] is True
    assert result["exact_match"] is True
    assert result["factory_error_rate_max_ulps"] == 4
    assert result["differences"] == []


@pytest.mark.parametrize("steps", [1, 4])
def test_observed_factory_error_rounding_is_accepted_and_preserved(steps):
    alternate = advance(OBSERVED_RATE, steps)
    result = compare_factory_rates(OBSERVED_RATE, alternate)
    assert result["matches"] is True
    assert result["exact_match"] is False
    assert result["factory_error_rate_max_ulps"] == 4
    assert len(result["differences"]) == 1
    difference = result["differences"][0]
    assert difference["path"] == FACTORY_PATH
    assert difference["representative_value"] == OBSERVED_RATE
    assert difference["alternate_value"] == alternate
    assert difference["ulp_distance"] == steps
    assert difference["accepted"] is True
    assert difference.get("comparison") or difference.get("reason")


@pytest.mark.parametrize("alternate", [advance(OBSERVED_RATE, 5), OBSERVED_RATE * 1.000001])
def test_larger_factory_error_change_does_not_match(alternate):
    result = compare_factory_rates(OBSERVED_RATE, alternate)
    assert result["matches"] is False
    assert result["exact_match"] is False
    assert any(d["path"] == FACTORY_PATH and d["accepted"] is False for d in result["differences"])


def test_four_representable_steps_are_accepted_across_a_binade():
    boundary = math.ldexp(1.0, -38)
    before = advance(boundary, 3, 0.0)
    result = compare_factory_rates(before, advance(before, 4))
    assert result["matches"] is True
    assert result["differences"][0]["ulp_distance"] == 4


def test_five_representable_steps_are_rejected_across_a_binade():
    # max(ulp(left), ulp(right)) would count this as only three ULPs and
    # incorrectly accept it; the adjacent binary64-value distance is five.
    boundary = math.ldexp(1.0, -38)
    before = advance(boundary, 4, 0.0)
    result = compare_factory_rates(before, advance(before, 5))
    assert result["matches"] is False
    assert result["differences"][0]["ulp_distance"] == 5


@pytest.mark.parametrize("value", [0.0, -1e-12, math.inf, -math.inf, math.nan])
def test_nonpositive_or_nonfinite_factory_rate_is_invalid_even_if_repeated(value):
    result = compare_factory_rates(value, value)
    assert result["matches"] is False
    assert any(d["path"] == FACTORY_PATH and d["accepted"] is False for d in result["differences"])


@pytest.mark.parametrize("left,right", [
    (0.0, math.nextafter(0.0, math.inf)),
    (-1e-12, math.nextafter(-1e-12, math.inf)),
])
def test_nearby_nonpositive_factory_rates_are_not_ulp_tolerated(left, right):
    assert compare_factory_rates(left, right)["matches"] is False


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_identical_nonfinite_data_outside_factories_is_rejected(value):
    left = fingerprint()
    left["native_cost_trace"]["deep"] = [{"invalid": value}]
    result = trajectory.compare_projection_cost_fingerprints(left, deepcopy(left))
    assert result["matches"] is False
    assert any(d["path"] == "/native_cost_trace/deep/0/invalid" and d["accepted"] is False
               for d in result["differences"])


@pytest.mark.parametrize("field", [
    "width", "runtime", "trace", "profile", "count", "budget", "synthesis", "composition",
])
def test_nonfactory_cost_fields_remain_exact(field):
    left, right = fingerprint(), fingerprint()
    point = right["declared_physical_pareto_points"][0]
    if field == "width":
        point["peak_physical_qubits"] += 1
    elif field == "runtime":
        point["runtime_per_execution_ns"] = math.nextafter(point["runtime_per_execution_ns"], math.inf)
    elif field == "trace":
        right["native_cost_trace"]["block"]["operations"].reverse()
    elif field == "profile":
        right["physical_profile"]["physical_error_rate"] = math.nextafter(1e-3, math.inf)
    elif field == "count":
        right["canonical_gate_counts"]["RZ"] += 1
    elif field == "budget":
        right["budget"]["physical_failure_probability"] = math.nextafter(1e-3, math.inf)
    elif field == "synthesis":
        right["synthesis"]["model_error_per_execution"] = math.nextafter(
            right["synthesis"]["model_error_per_execution"], math.inf)
    else:
        right["error_composition"] = "Product"
    result = trajectory.compare_projection_cost_fingerprints(left, right)
    assert result["matches"] is False
    assert result["exact_match"] is False
    assert any(d["accepted"] is False for d in result["differences"])


def test_same_named_error_rate_outside_the_declared_factory_path_stays_exact():
    left, right = fingerprint(), fingerprint()
    left["unrelated"] = {"factories": {"T": {"error_rate": OBSERVED_RATE}}}
    right["unrelated"] = {"factories": {"T": {"error_rate": advance(OBSERVED_RATE, 1)}}}
    assert trajectory.compare_projection_cost_fingerprints(left, right)["matches"] is False


@pytest.mark.parametrize("change", ["key", "integer_type", "boolean_type", "container_type", "length"])
def test_structural_keys_types_and_container_shapes_stay_exact(change):
    left, right = fingerprint(), fingerprint()
    if change == "key":
        right["budget"]["different_key"] = right["budget"].pop("executions")
    elif change == "integer_type":
        right["budget"]["executions"] = float(right["budget"]["executions"])
    elif change == "boolean_type":
        right["synthesis"]["actual_synthesis_sequence_generated"] = 0
    elif change == "container_type":
        right["native_cost_trace"]["block"]["operations"] = tuple(
            right["native_cost_trace"]["block"]["operations"])
    else:
        right["declared_physical_pareto_points"].append(deepcopy(right["declared_physical_pareto_points"][0]))
    result = trajectory.compare_projection_cost_fingerprints(left, right)
    assert result["matches"] is False
    assert any(d["accepted"] is False for d in result["differences"])


def test_raw_inputs_are_not_modified_and_all_difference_evidence_is_kept():
    left, right = fingerprint(), fingerprint()
    right["declared_physical_pareto_points"][0]["factories"]["T"]["error_rate"] = advance(OBSERVED_RATE, 1)
    right["declared_physical_pareto_points"][0]["runtime_per_execution_ns"] += 1
    original_left, original_right = deepcopy(left), deepcopy(right)
    result = trajectory.compare_projection_cost_fingerprints(left, right)
    assert left == original_left
    assert right == original_right
    assert result["matches"] is False
    by_path = {d["path"]: d for d in result["differences"]}
    assert by_path[FACTORY_PATH]["representative_value"] == OBSERVED_RATE
    assert by_path[FACTORY_PATH]["alternate_value"] == advance(OBSERVED_RATE, 1)
    assert by_path[FACTORY_PATH]["accepted"] is True
    assert by_path["/declared_physical_pareto_points/0/runtime_per_execution_ns"]["accepted"] is False
