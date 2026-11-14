"""Synthetic checks of the compiled notebook's aggregate diagnostic transport.

Extract and execute only the actual arithmetic helper's function AST. The
notebook cell, scientific imports, circuit checks and saved study never run in
this module. All inputs below are small synthetic binary64 values.
"""

import ast
from fractions import Fraction
import json
import math
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/optimization_compiled.ipynb"
HELPER_NAME = "reconcile_weighted_tv_transport"


@pytest.fixture(scope="module")
def reconcile():
    notebook = json.loads(NOTEBOOK.read_text())
    cells = [cell for cell in notebook["cells"] if cell.get("id") == "compiled-circuits"]
    assert len(cells) == 1 and cells[0]["cell_type"] == "code"
    source = "".join(cells[0]["source"])
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == HELPER_NAME]
    assert len(functions) == 1
    function = functions[0]
    assert [argument.arg for argument in function.args.args] == [
        "stored_tvs", "replay_tvs", "shots", "saved_total", "replay_total",
    ]
    # No defaults, annotations, decorators or imports may execute while
    # defining the isolated function, or introduce a scientific dependency.
    assert not function.decorator_list and function.returns is None
    assert not function.args.posonlyargs and not function.args.kwonlyargs
    assert function.args.vararg is None and function.args.kwarg is None
    assert not function.args.defaults and not function.args.kw_defaults
    assert all(argument.annotation is None for argument in function.args.args)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(function))
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {"math": math, "Fraction": Fraction}
    exec(compile(ast.fix_missing_locations(isolated), str(NOTEBOOK) + "::" + HELPER_NAME, "exec"), namespace)
    return namespace[HELPER_NAME]


def ordered_binary64_total(values, weights):
    """Match an ordered product/add diagnostic, without any study code."""
    total = 0.0
    for value, weight in zip(values, weights):
        total += float(value) * weight
    return total


def exact_weighted_total(values, weights):
    return sum((Fraction.from_float(float(value)) * weight
                for value, weight in zip(values, weights)), Fraction())


def arithmetic_bound(values, weights):
    """Independent rational form of the declared gamma_(2m) model."""
    operations = 2 * len(weights)
    binary64_denominator = 2**53 - operations
    relative = Fraction(operations, binary64_denominator)
    absolute = Fraction(operations * 2**53, 2**1075 * binary64_denominator)
    return relative * exact_weighted_total(values, weights) + absolute


@pytest.mark.parametrize("values,weights,total", [
    ([0.125, 0.25, 0.0625], [2, 3, 4], 1.25),
    ([0.0, -0.0], [1, 11], 0.0),
    ([0.0] * 64, [1] * 64, 0.0),
    ([1], [2**53], float(2**53)),
])
def test_exact_and_zero_totals_and_bounded_domain_endpoints(reconcile, values, weights, total):
    result = reconcile(values, values, weights, total, total)
    bound = arithmetic_bound(values, weights)
    assert result["observed_shot_weighted_tv_transport"] == 0.0
    assert result["aggregate_difference"] == 0.0
    assert result["stored_aggregate_roundoff_bound"] == float(bound)
    assert result["replayed_aggregate_roundoff_bound"] == float(bound)
    assert result["transport_bound"] == float(2 * bound)
    assert result["exact_binary_rational_comparisons"] is True
    assert result["scope"] == (
        "diagnostic host transport; all class, realization and physical budgets unchanged"
    )


def test_ordered_rounding_is_checked_against_each_exact_binary64_sum(reconcile):
    values, weights = [1 / 3, 1 / 7, 1 / 10], [7, 11, 13]
    total = ordered_binary64_total(values, weights)
    exact = exact_weighted_total(values, weights)
    bound = arithmetic_bound(values, weights)
    assert 0 < abs(Fraction.from_float(total) - exact) <= bound
    result = reconcile(values, values, weights, total, total)
    assert result["stored_aggregate_roundoff_bound"] == float(bound)
    assert result["replayed_aggregate_roundoff_bound"] == float(bound)
    assert result["aggregate_difference"] == 0.0


def test_tiny_valid_class_drift_can_exceed_the_old_aggregate_tolerance(reconcile):
    stored, replay = [1e-9, 2e-9], [1e-9 + 1e-13, 2e-9 + 2e-13]
    weights = [32_768, 49_264]
    saved_total = ordered_binary64_total(stored, weights)
    replay_total = ordered_binary64_total(replay, weights)
    assert sum(weights) == 82_032
    assert all(abs(old - new) <= 2e-12 + 1e-7 * abs(new)
               for old, new in zip(stored, replay))
    assert abs(replay_total - saved_total) > 1e-12 + 1e-10 * abs(saved_total)
    transport = sum((weight * abs(Fraction.from_float(new) - Fraction.from_float(old))
                     for weight, old, new in zip(weights, stored, replay)), Fraction())
    expected_bound = transport + arithmetic_bound(stored, weights) + arithmetic_bound(replay, weights)
    result = reconcile(stored, replay, weights, saved_total, replay_total)
    assert result["observed_shot_weighted_tv_transport"] == float(transport)
    assert result["transport_bound"] == float(expected_bound)
    assert result["aggregate_difference"] == float(abs(
        Fraction.from_float(replay_total) - Fraction.from_float(saved_total)))
    assert result["aggregate_difference"] <= result["transport_bound"]


def test_opposite_class_drifts_use_weighted_absolute_transport(reconcile):
    stored, weights = [0.25, 0.5], [1, 2]
    replay = [stored[0] + math.ldexp(1.0, -30), stored[1] - math.ldexp(1.0, -31)]
    assert exact_weighted_total(stored, weights) == exact_weighted_total(replay, weights)
    result = reconcile(stored, replay, weights, 1.25, 1.25)
    assert result["aggregate_difference"] == 0.0
    assert result["observed_shot_weighted_tv_transport"] == math.ldexp(1.0, -29)
    assert result["transport_bound"] >= math.ldexp(1.0, -29)


@pytest.mark.parametrize("side,message", [
    ("saved", "Stored aggregate differs"),
    ("replay", "Replayed aggregate differs"),
])
def test_each_aggregate_integrity_gate_rejects_corruption_hidden_by_transport(reconcile, side, message):
    stored, replay, weights = [1e-9], [1e-9 + 1e-13], [82_032]
    saved_total = ordered_binary64_total(stored, weights)
    replay_total = ordered_binary64_total(replay, weights)
    corruption = 1e-10
    observed = weights[0] * abs(Fraction.from_float(replay[0]) - Fraction.from_float(stored[0]))
    assert corruption < float(observed)
    if side == "saved":
        saved_total += corruption
    else:
        replay_total -= corruption
    # Even this corrupted pair passes an aggregate-only transport comparison.
    assert abs(Fraction.from_float(replay_total) - Fraction.from_float(saved_total)) <= (
        observed + arithmetic_bound(stored, weights) + arithmetic_bound(replay, weights)
    )
    with pytest.raises(AssertionError, match=message):
        reconcile(stored, replay, weights, saved_total, replay_total)


def test_identically_corrupted_totals_cannot_hide_behind_zero_aggregate_drift(reconcile):
    values, weights = [0.125], [8]
    corrupted = 1.0 + 1e-10
    with pytest.raises(AssertionError, match="Stored aggregate differs"):
        reconcile(values, values, weights, corrupted, corrupted)


def test_replayed_total_with_wrong_class_weights_is_rejected(reconcile):
    values, weights = [0.125, 0.25], [2, 3]
    correct = ordered_binary64_total(values, weights)
    wrong = ordered_binary64_total(values, list(reversed(weights)))
    assert correct == 1.0 and wrong == 0.875
    with pytest.raises(AssertionError, match="Replayed aggregate differs"):
        reconcile(values, values, weights, correct, wrong)


def test_original_per_class_reference_orientation_is_preserved(reconcile):
    old = 0.5
    old_threshold = 2e-12 + 1e-7 * abs(old)
    new = math.nextafter(math.nextafter(old + old_threshold, math.inf), math.inf)
    assert old_threshold < abs(old - new) <= 2e-12 + 1e-7 * abs(new)
    reconcile([old], [new], [1], old, new)
    with pytest.raises(AssertionError, match="Per-class TV replay left the existing tolerance"):
        reconcile([new], [old], [1], new, old)


@pytest.mark.parametrize("old,new", [(1e-9, 1e-9 + 3e-12), (0.5, 0.5000002)])
def test_class_drift_outside_the_original_tolerance_is_rejected(reconcile, old, new):
    assert abs(old - new) > 2e-12 + 1e-7 * abs(new)
    with pytest.raises(AssertionError, match="Per-class TV replay left the existing tolerance"):
        reconcile([old], [new], [82_032], old * 82_032, new * 82_032)


def test_explicit_underflow_allowance_remains_bounded_and_independently_checked(reconcile):
    minimum_subnormal = math.ulp(0.0)
    result = reconcile([0.0], [0.0], [1], minimum_subnormal, 0.0)
    assert result["observed_shot_weighted_tv_transport"] == 0.0
    assert result["stored_aggregate_roundoff_bound"] == minimum_subnormal
    assert result["replayed_aggregate_roundoff_bound"] == minimum_subnormal
    assert result["transport_bound"] == 2 * minimum_subnormal
    with pytest.raises(AssertionError, match="Stored aggregate differs"):
        reconcile([0.0], [0.0], [1], 2 * minimum_subnormal, 0.0)


@pytest.mark.parametrize("field", ["stored_tvs", "replay_tvs", "saved_total", "replay_total"])
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf, -0.125, 1.25, True, "0.125", Fraction(1, 8)])
def test_nonfinite_out_of_range_bool_and_nonnumeric_diagnostics_are_rejected(reconcile, field, invalid):
    arguments = {"stored_tvs": [0.125], "replay_tvs": [0.125], "shots": [1],
                 "saved_total": 0.125, "replay_total": 0.125}
    arguments[field] = [invalid] if field.endswith("tvs") else invalid
    with pytest.raises(ValueError, match="Invalid finite nonnegative TV diagnostic"):
        reconcile(**arguments)


@pytest.mark.parametrize("invalid", [0, -1, True, 1.0, Fraction(1), 2**53 + 1])
def test_nonpositive_noninteger_and_unrepresentable_weights_are_rejected(reconcile, invalid):
    with pytest.raises(ValueError, match="Shots must be positive exactly representable"):
        reconcile([0.0], [0.0], [invalid], 0.0, 0.0)


def test_total_weight_above_exact_binary64_integer_range_is_rejected(reconcile):
    with pytest.raises(ValueError, match="Shots must be positive exactly representable"):
        reconcile([0.0, 0.0], [0.0, 0.0], [2**53, 1], 0.0, 0.0)


@pytest.mark.parametrize("stored,replay,weights", [
    ([], [], []),
    ([0.0], [0.0], [1, 1]),
    ([0.0, 0.0], [0.0], [1, 1]),
    ([0.0] * 65, [0.0] * 65, [1] * 65),
])
def test_empty_oversized_or_mismatched_vectors_are_rejected(reconcile, stored, replay, weights):
    with pytest.raises(ValueError, match="Invalid bounded TV vectors"):
        reconcile(stored, replay, weights, 0.0, 0.0)
