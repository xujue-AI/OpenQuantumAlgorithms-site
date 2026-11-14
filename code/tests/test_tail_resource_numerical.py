# SPDX-License-Identifier: MIT
"""Self-contained stdlib numerical-policy tests; no circuit replay occurs."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("tail_numerical_under_test", ROOT / "scripts/tail_resource_numerical.py")
numerical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(numerical)


def _guard(observed=0.0):
    return {"observed_maximum": observed, "upper": math.nextafter(observed, math.inf)}


def _side(kind, operations, dimension=2):
    residuals = {
        "acceptance_probability": {"kind": "scalar", "actual": .125, "saved": .125,
                                   "signed_difference": 0.0, "maximum_absolute_difference": 0.0},
        "mean": {"kind": "scalar", "actual": .5, "saved": .5,
                 "signed_difference": 0.0, "maximum_absolute_difference": 0.0},
        "reduced_before_output": {"kind": "complex_matrix", "shape": [dimension, dimension],
                                 "maximum_absolute_difference": 0.0,
                                 "maximum_real_absolute_difference": 0.0,
                                 "maximum_imaginary_absolute_difference": 0.0,
                                 "frobenius_difference": 0.0},
    }
    if kind == "complete":
        residuals["joint_probabilities"] = {"kind": "real_vector", "shape": [dimension],
                                           "maximum_absolute_difference": 0.0, "l1_difference": 0.0}
    return {
        "saved_record_field_residuals": residuals, "acceptance_probability": .125, "mean": .5,
        "primitive_operations": operations,
        "numerical_method": "density matrix" if kind == "instrument" else "unnormalized pure-state Kraus ensemble",
        "peak_branches": None if kind == "instrument" else 1,
    }


def _fixture(kind):
    """Synthetic schema fixtures, independent of private proof logs."""
    cases = []
    policy_cases = {}
    for index, name in enumerate(numerical.CASES[kind]):
        frozen = 0.0 if kind == "instrument" else .01
        dimension = 4 if kind == "complete" and name in ("thermal-n2-X", "thermal-n2-Z") else 2
        case = {
            "case": name, "source": _side(kind, 10 + index, dimension), "compiled": _side(kind, 20 + index, dimension),
            "independent_reference_mean": .5, "saved_exact_reference_mean": None if kind == "instrument" else .5,
            "independent_saved_exact_reference_signed_difference": None if kind == "instrument" else 0.0,
            "independent_saved_exact_reference_absolute_difference": None if kind == "instrument" else 0.0,
            "source_reference_absolute_score_difference": 0.0, "source_reference_signed_score_difference": 0.0,
            "compiled_reference_absolute_score_difference": 0.0, "frozen_source_score_bias": frozen,
            "source_reference_gap_minus_frozen_score_bias": 0.0 - frozen,
        }
        expected = {
            "reference": {"baseline": .5, "absolute_difference": _guard()},
            "source_bias": {"mode": "recorded_discrepancy" if kind == "instrument" else "frozen_upper_bound",
                            "frozen": frozen},
            "saved_exact_reference_mean": case["saved_exact_reference_mean"],
        }
        if kind == "instrument":
            expected["source_bias"]["absolute_difference"] = _guard()
        for side in ("source", "compiled"):
            value = case[side]
            fields = {}
            for field, residual in value["saved_record_field_residuals"].items():
                entry = {"kind": residual["kind"], "limits": {metric: _guard() for metric in numerical._METRICS[residual["kind"]]}}
                entry["saved" if residual["kind"] == "scalar" else "shape"] = residual.get("saved", residual.get("shape"))
                fields[field] = entry
            expected[side] = {key: value[key] for key in ("primitive_operations", "numerical_method", "peak_branches")}
            expected[side]["fields"] = fields
        cases.append(case)
        policy_cases[name] = expected
    evidence = {
        "status": "evidence_collected", "kind": kind, "cases": cases,
        "runtime": {"python": "3.12.14 (synthetic test)", "platform": "synthetic platform",
                    "machine": "arm64", "packages": numerical.PACKAGES.copy()},
        "authenticated_helper_sha256": "0" * 64, "authenticated_lock_sha256": "1" * 64,
        "numerical_compatibility_policy": numerical.UNAPPROVED_WORKER_MESSAGE,
        "scope": numerical.SCOPE, "excluded": numerical.EXCLUDED[:],
    }
    policy = {
        "schema": numerical.POLICY_SCHEMA, "status": numerical.POLICY_STATUS, "kind": kind,
        "helper_sha256": "0" * 64, "lock_sha256": "1" * 64,
        "runtime": {"python_version": "3.12.14", "packages": numerical.PACKAGES.copy(),
                    "machines": ["arm64", "x86_64"]},
        "basis": {"evidence": [{"label": "local", "sha256": "2" * 64}, {"label": "linux", "sha256": "3" * 64}],
                  "guard_rule": numerical.GUARD_RULE, "certified": False},
        "cases": policy_cases,
    }
    return evidence, policy


def _set_source_mean(evidence, value):
    case = evidence["cases"][0]
    side = case["source"]
    side["mean"] = value
    field = side["saved_record_field_residuals"]["mean"]
    field["actual"] = value
    field["signed_difference"] = value - field["saved"]
    field["maximum_absolute_difference"] = abs(field["signed_difference"])
    delta = value - case["independent_reference_mean"]
    case["source_reference_signed_score_difference"] = delta
    case["source_reference_absolute_score_difference"] = abs(delta)
    case["source_reference_gap_minus_frozen_score_bias"] = abs(delta) - case["frozen_source_score_bias"]


def _set_reference(evidence, value):
    case = evidence["cases"][0]
    case["independent_reference_mean"] = value
    delta = case["source"]["mean"] - value
    case["source_reference_signed_score_difference"] = delta
    case["source_reference_absolute_score_difference"] = abs(delta)
    case["compiled_reference_absolute_score_difference"] = abs(case["compiled"]["mean"] - value)
    case["source_reference_gap_minus_frozen_score_bias"] = abs(delta) - case["frozen_source_score_bias"]
    if evidence["kind"] == "complete":
        delta = value - case["saved_exact_reference_mean"]
        case["independent_saved_exact_reference_signed_difference"] = delta
        case["independent_saved_exact_reference_absolute_difference"] = abs(delta)


@pytest.mark.parametrize("kind", ["instrument", "complete"])
def test_typed_policy_passes_without_mutation(kind):
    evidence, policy = _fixture(kind)
    before = copy.deepcopy((evidence, policy))
    result = numerical.verify_numerical_replay(evidence, policy)
    assert result["cases_checked"] == len(numerical.CASES[kind])
    assert result["status"] == "verified_observed_numerical_reproduction"
    assert result["formal_roundoff_certificate"] is False
    assert result["scientific_budgets_changed"] is False
    assert (evidence, policy) == before


def test_zero_observation_admits_only_adjacent_subnormal_metric():
    evidence, policy = _fixture("instrument")
    field = evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"]
    tiny = math.nextafter(0.0, math.inf)
    field.update(maximum_absolute_difference=tiny, maximum_real_absolute_difference=tiny, frobenius_difference=tiny)
    numerical.verify_numerical_replay(evidence, policy)
    field.update(maximum_absolute_difference=2 * tiny, maximum_real_absolute_difference=2 * tiny, frobenius_difference=2 * tiny)
    with pytest.raises(ValueError, match="limit exceeded"):
        numerical.verify_numerical_replay(evidence, policy)


def test_blanket_epsilon_cannot_replace_derived_representation_guard():
    evidence, policy = _fixture("instrument")
    name = evidence["cases"][0]["case"]
    policy["cases"][name]["source"]["fields"]["mean"]["limits"]["maximum_absolute_difference"]["upper"] = 2e-8
    with pytest.raises(ValueError, match="representation guard"):
        numerical.verify_numerical_replay(evidence, policy)


def test_policy_cannot_change_field_kind_to_hide_scalar_reproduction():
    evidence, policy = _fixture("instrument")
    name = evidence["cases"][0]["case"]
    policy["cases"][name]["source"]["fields"]["mean"]["kind"] = "real_vector"
    with pytest.raises(ValueError, match="field-name/kind"):
        numerical.verify_numerical_replay(evidence, policy)


def test_policy_and_evidence_cannot_together_change_output_shape():
    evidence, policy = _fixture("instrument")
    name = evidence["cases"][0]["case"]
    policy["cases"][name]["source"]["fields"]["reduced_before_output"]["shape"] = [4, 4]
    evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"]["shape"] = [4, 4]
    with pytest.raises(ValueError, match="field shape"):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("path", [
    ("maximum_absolute_difference",), ("maximum_real_absolute_difference",),
    ("maximum_imaginary_absolute_difference",), ("frobenius_difference",),
])
def test_every_complex_metric_is_mandatory(path):
    evidence, policy = _fixture("instrument")
    del evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"][path[0]]
    with pytest.raises(ValueError, match="missing/unexpected"):
        numerical.verify_numerical_replay(evidence, policy)


def test_vector_l1_metric_is_mandatory_and_separately_bounded():
    evidence, policy = _fixture("complete")
    field = evidence["cases"][0]["source"]["saved_record_field_residuals"]["joint_probabilities"]
    del field["l1_difference"]
    with pytest.raises(ValueError, match="missing/unexpected"):
        numerical.verify_numerical_replay(evidence, policy)
    evidence, policy = _fixture("complete")
    field = evidence["cases"][0]["source"]["saved_record_field_residuals"]["joint_probabilities"]
    tiny = math.nextafter(0.0, math.inf)
    field.update(maximum_absolute_difference=tiny, l1_difference=2 * tiny)
    with pytest.raises(ValueError, match="limit exceeded"):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("field,value", [
    ("maximum_absolute_difference", math.nan), ("maximum_real_absolute_difference", math.inf),
    ("maximum_imaginary_absolute_difference", -1.0), ("frobenius_difference", True),
])
def test_nonfinite_negative_or_bool_residual_rejected(field, value):
    evidence, policy = _fixture("instrument")
    evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"][field] = value
    with pytest.raises(ValueError):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("mutation", ["scalar_delta", "scalar_max", "scalar_actual", "side_mean", "reference_signed", "reference_absolute", "compiled_reference", "bias_gap"])
def test_forged_or_stale_numeric_summary_rejected(mutation):
    evidence, policy = _fixture("complete")
    case = evidence["cases"][0]
    field = case["source"]["saved_record_field_residuals"]["mean"]
    if mutation == "scalar_delta":
        field["signed_difference"] = .1
    elif mutation == "scalar_max":
        field["maximum_absolute_difference"] = .1
    elif mutation == "scalar_actual":
        field["actual"] = .1
    elif mutation == "side_mean":
        case["source"]["mean"] = .1
    elif mutation == "reference_signed":
        case["source_reference_signed_score_difference"] = .1
    elif mutation == "reference_absolute":
        case["source_reference_absolute_score_difference"] = .1
    elif mutation == "compiled_reference":
        case["compiled_reference_absolute_score_difference"] = .1
    else:
        case["source_reference_gap_minus_frozen_score_bias"] = .1
    with pytest.raises(ValueError, match="exact numeric identity"):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("field,value", [
    ("primitive_operations", 11), ("primitive_operations", 10.0), ("primitive_operations", True),
    ("numerical_method", "different"), ("peak_branches", 2),
])
def test_exact_counts_method_and_branches_rejected(field, value):
    evidence, policy = _fixture("complete")
    evidence["cases"][0]["source"][field] = value
    with pytest.raises(ValueError):
        numerical.verify_numerical_replay(evidence, policy)


def test_changed_frozen_bias_rejected_even_with_consistent_summary():
    evidence, policy = _fixture("complete")
    case = evidence["cases"][0]
    case["frozen_source_score_bias"] = .02
    case["source_reference_gap_minus_frozen_score_bias"] = -.02
    with pytest.raises(ValueError, match="frozen_upper_bound"):
        numerical.verify_numerical_replay(evidence, policy)


def test_failed_complete_source_bound_is_never_rescued_by_reproduction_limit():
    evidence, policy = _fixture("complete")
    name = evidence["cases"][0]["case"]
    _set_source_mean(evidence, .52)
    policy["cases"][name]["source"]["fields"]["mean"]["limits"]["maximum_absolute_difference"] = _guard(.1)
    with pytest.raises(ValueError, match="frozen source bias upper bound failed"):
        numerical.verify_numerical_replay(evidence, policy)


def test_instrument_discrepancy_is_reproduced_separately():
    evidence, policy = _fixture("instrument")
    name = evidence["cases"][0]["case"]
    _set_source_mean(evidence, .52)
    policy["cases"][name]["source"]["fields"]["mean"]["limits"]["maximum_absolute_difference"] = _guard(.1)
    with pytest.raises(ValueError, match="recorded_source_discrepancy"):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("kind", ["instrument", "complete"])
def test_failed_independent_reference_rejected(kind):
    evidence, policy = _fixture(kind)
    _set_reference(evidence, .6)
    with pytest.raises(ValueError, match="independent_reference"):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("mutation", ["patch", "package", "helper", "lock", "scope", "excluded", "machine", "missing_runtime"])
def test_environment_and_identity_changes_rejected(mutation):
    evidence, policy = _fixture("instrument")
    if mutation == "patch":
        evidence["runtime"]["python"] = "3.12.15 synthetic"
    elif mutation == "package":
        evidence["runtime"]["packages"]["numpy"] = "different"
    elif mutation == "helper":
        evidence["authenticated_helper_sha256"] = "a" * 64
    elif mutation == "lock":
        evidence["authenticated_lock_sha256"] = "a" * 64
    elif mutation == "scope":
        evidence["scope"] = "full arbitrary-input theorem"
    elif mutation == "excluded":
        evidence["excluded"].pop()
    elif mutation == "machine":
        evidence["runtime"]["machine"] = "unobserved"
    else:
        del evidence["runtime"]["platform"]
    with pytest.raises(ValueError):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("mutation", ["pending", "one_platform", "duplicate_platform", "certified", "guard_rule"])
def test_unreviewed_or_incomplete_policy_rejected(mutation):
    evidence, policy = _fixture("instrument")
    if mutation == "pending":
        policy["status"] = "pending_linux"
    elif mutation == "one_platform":
        policy["basis"]["evidence"].pop()
    elif mutation == "duplicate_platform":
        policy["basis"]["evidence"][1]["label"] = "local"
    elif mutation == "certified":
        policy["basis"]["certified"] = True
    else:
        policy["basis"]["guard_rule"] = "blanket epsilon"
    with pytest.raises(ValueError):
        numerical.verify_numerical_replay(evidence, policy)


@pytest.mark.parametrize("mutation", ["duplicate_case", "missing_case", "unhashable_kind", "unhashable_machine", "missing_field", "changed_shape", "bool_shape"])
def test_malformed_schema_rejected_as_value_error(mutation):
    evidence, policy = _fixture("instrument")
    if mutation == "duplicate_case":
        evidence["cases"][1] = copy.deepcopy(evidence["cases"][0])
    elif mutation == "missing_case":
        evidence["cases"].pop()
    elif mutation == "unhashable_kind":
        evidence["kind"] = []
    elif mutation == "unhashable_machine":
        policy["runtime"]["machines"] = [[]]
    elif mutation == "missing_field":
        del evidence["cases"][0]["source"]["saved_record_field_residuals"]["mean"]
    elif mutation == "changed_shape":
        evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"]["shape"] = [4, 4]
    else:
        evidence["cases"][0]["source"]["saved_record_field_residuals"]["reduced_before_output"]["shape"] = [True, True]
    with pytest.raises(ValueError):
        numerical.verify_numerical_replay(evidence, policy)


def test_import_and_checker_require_only_stdlib_in_fresh_process():
    probe = """
import importlib.abc, importlib.util, pathlib, sys
class BlockScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy','scipy','qiskit','qdk','qsharp','openquantum'}:
            raise AssertionError('unexpected scientific import ' + fullname)
sys.meta_path.insert(0, BlockScience())
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('tail_numerical_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module._limit({'observed_maximum':0.,'upper':float.fromhex('0x0.0000000000001p-1022')},'zero') > 0
print('stdlib only')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-S", "-c", probe,
                             str(ROOT / "scripts/tail_resource_numerical.py")],
                            check=True, capture_output=True, text=True, timeout=10)
    assert result.stdout.strip() == "stdlib only"
