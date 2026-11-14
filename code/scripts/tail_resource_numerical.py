# SPDX-License-Identifier: MIT
"""Check authenticated fixed-input replay against an observed numerical policy.

This module is pure standard-library, in-memory validation. Its caller must
authenticate the policy, replay worker and complete record inventory first.
Limits cover reproduction of typed residual summaries on reviewed platforms;
they are not a formal roundoff bound, new scientific accuracy allowance, or
permission to change any source, success, repetition or physical-model input.
"""
from __future__ import annotations

import math
import re


class NumericalVerificationError(ValueError):
    """Authenticated replay does not meet the reviewed reproduction policy."""


POLICY_SCHEMA = "openquantum-tail-numerical-policy-v1"
POLICY_STATUS = "reviewed_observed_reproduction"
GUARD_RULE = "nextafter_observed_summary_toward_positive_infinity"
SCOPE = "saved initialized fixed measured trials; source and aggregate Clifford+T QASM with explicit reset/readout schedule"
EXCLUDED = [
    "arbitrary-input channel theorem", "per-rotation approximant revalidation",
    "original routing proof", "Native Trace reconstruction", "renewed search or optimality",
    "fresh fitting, synthesis, estimation or budget/retry repricing",
]
CASES = {
    "instrument": ("damping-Z", "lindblad-Z", "thermal-X", "thermal-Z"),
    "complete": ("dynamics-n1-Z", "dynamics-n2-Z", "thermal-n1-X", "thermal-n2-X", "thermal-n2-Z"),
}
PACKAGES = {"numpy": "2.5.4", "scipy": "1.18.1", "qiskit": "2.6.2"}
SHA = re.compile(r"[0-9a-f]{64}\Z")
UNAPPROVED_WORKER_MESSAGE = "unapproved; collect local and Linux field residuals before freezing an observed policy"
_CASE_KEYS = {
    "case", "source", "compiled", "independent_reference_mean", "saved_exact_reference_mean",
    "independent_saved_exact_reference_signed_difference", "independent_saved_exact_reference_absolute_difference",
    "source_reference_absolute_score_difference", "source_reference_signed_score_difference",
    "compiled_reference_absolute_score_difference", "frozen_source_score_bias",
    "source_reference_gap_minus_frozen_score_bias",
}
_SIDE_KEYS = {"saved_record_field_residuals", "acceptance_probability", "mean", "primitive_operations",
              "numerical_method", "peak_branches"}
_METRICS = {
    "scalar": ("maximum_absolute_difference",),
    "real_vector": ("maximum_absolute_difference", "l1_difference"),
    "complex_matrix": ("maximum_absolute_difference", "maximum_real_absolute_difference",
                       "maximum_imaginary_absolute_difference", "frobenius_difference"),
}
_FIELD_KINDS = {"acceptance_probability": "scalar", "mean": "scalar",
                "joint_probabilities": "real_vector", "reduced_before_output": "complex_matrix"}


def _need(condition, label, message):
    if not condition:
        raise NumericalVerificationError(label + ": " + message)


def _keys(value, keys, label):
    _need(isinstance(value, dict) and set(value) == set(keys), label, "missing/unexpected fields")
    return value


def _real(value, label, *, nonnegative=False):
    _need(type(value) in (int, float), label, "expected finite non-bool number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise NumericalVerificationError(label + ": invalid numeric representation") from error
    _need(math.isfinite(value) and (not nonnegative or value >= 0), label, "nonfinite/negative number")
    return value


def _integer(value, label, maximum):
    _need(type(value) is int and 0 < value <= maximum, label, "expected bounded positive integer")
    return value


def _same_number(left, right, label):
    _need(_real(left, label).hex() == _real(right, label).hex(), label, "exact numeric identity differs")


def _shape(value, label, *, matrix):
    rank = 2 if matrix else 1
    _need(isinstance(value, list) and len(value) == rank, label, "shape rank differs")
    dimensions = [_integer(item, label, 256) for item in value]
    if matrix:
        _need(dimensions[0] == dimensions[1], label, "reduced matrix must be square")
    return dimensions


def _limit(value, label):
    _keys(value, ("observed_maximum", "upper"), label)
    observed = _real(value["observed_maximum"], label, nonnegative=True)
    upper = _real(value["upper"], label, nonnegative=True)
    # The only admitted guard is the directed adjacent binary64 value. It
    # guards representation of a measured residual summary; it does not
    # predict unobserved libm/kernel changes or certify roundoff propagation.
    _need(upper == math.nextafter(observed, math.inf), label,
          "limit is not the directed observed-summary representation guard")
    return upper


def _bounded(value, limit, label, summary):
    actual = _real(value, label, nonnegative=True)
    upper = _limit(limit, label + ".policy")
    _need(actual <= upper, label, "observed numerical reproduction limit exceeded")
    summary["checks"] += 1
    summary["maximum_absolute_residual_metric"] = max(summary["maximum_absolute_residual_metric"], actual)


def _field(actual, expected, field, side, label, summary, dimension):
    kind = expected.get("kind") if isinstance(expected, dict) else None
    _need(isinstance(kind, str) and kind in _METRICS, label, "unknown field kind")
    _need(kind == _FIELD_KINDS[field], label, "field-name/kind mapping differs")
    is_scalar = kind == "scalar"
    actual_keys = {"kind", *_METRICS[kind]}
    expected_keys = {"kind", "limits"}
    if is_scalar:
        actual_keys |= {"actual", "saved", "signed_difference"}
        expected_keys.add("saved")
    else:
        actual_keys.add("shape")
        expected_keys.add("shape")
    _keys(actual, actual_keys, label)
    _keys(expected, expected_keys, label + ".policy")
    _need(actual["kind"] == kind, label, "field kind differs")
    _keys(expected["limits"], _METRICS[kind], label + ".limits")
    if is_scalar:
        _same_number(actual["saved"], expected["saved"], label + ".saved")
        _same_number(actual["actual"], side[field], label + ".actual")
        delta = _real(actual["actual"], label) - _real(actual["saved"], label)
        _same_number(actual["signed_difference"], delta, label + ".signed_difference")
        _same_number(actual["maximum_absolute_difference"], abs(delta), label + ".maximum_absolute_difference")
    else:
        shape = _shape(actual["shape"], label, matrix=kind == "complex_matrix")
        _shape(expected["shape"], label + ".policy", matrix=kind == "complex_matrix")
        required_shape = [dimension, dimension] if kind == "complex_matrix" else [dimension]
        _need(shape == expected["shape"] == required_shape, label, "field shape differs")
        # Array residuals are produced from the actual arrays by the
        # authenticated numerical worker. All metrics are mandatory and
        # independently bounded here. We do not fabricate array reconstruction
        # or assume exact algebraic norm identities for floating reductions.
    for metric in _METRICS[kind]:
        _bounded(actual[metric], expected["limits"][metric], label + "." + metric, summary)


def _side(actual, expected, kind, label, summary, case_name):
    _keys(actual, _SIDE_KEYS, label)
    _keys(expected, ("primitive_operations", "numerical_method", "peak_branches", "fields"), label + ".policy")
    _integer(actual["primitive_operations"], label, 500_000)
    _integer(expected["primitive_operations"], label + ".policy", 500_000)
    _need(actual["primitive_operations"] == expected["primitive_operations"], label, "exact operation count differs")
    method = "density matrix" if kind == "instrument" else "unnormalized pure-state Kraus ensemble"
    _need(actual["numerical_method"] == expected["numerical_method"] == method, label, "numerical method differs")
    if kind == "instrument":
        _need(actual["peak_branches"] is None and expected["peak_branches"] is None, label, "density-path branch metadata differs")
    else:
        _integer(actual["peak_branches"], label, 256)
        _integer(expected["peak_branches"], label + ".policy", 256)
        _need(actual["peak_branches"] == expected["peak_branches"], label, "exact branch count differs")
    _real(actual["acceptance_probability"], label)
    _real(actual["mean"], label)
    fields = {"acceptance_probability", "mean", "reduced_before_output"}
    if kind == "complete":
        fields.add("joint_probabilities")
    _keys(actual["saved_record_field_residuals"], fields, label + ".fields")
    _keys(expected["fields"], fields, label + ".policy.fields")
    # These are the authenticated trial output contracts, not full system
    # Hilbert-space dimensions. Dynamics-n2-Z reads one outcome bit.
    dimension = 4 if kind == "complete" and case_name in ("thermal-n2-X", "thermal-n2-Z") else 2
    for field in sorted(fields):
        _field(actual["saved_record_field_residuals"][field], expected["fields"][field],
               field, actual, label + "." + field, summary, dimension)


def _case(actual, expected, kind, label, summary):
    _keys(actual, _CASE_KEYS, label)
    _keys(expected, ("source", "compiled", "reference", "source_bias", "saved_exact_reference_mean"), label + ".policy")
    for side in ("source", "compiled"):
        _side(actual[side], expected[side], kind, label + "." + side, summary, actual["case"])
    reference = _real(actual["independent_reference_mean"], label)
    source_delta = _real(actual["source"]["mean"], label) - reference
    compiled_delta = _real(actual["compiled"]["mean"], label) - reference
    frozen_bias = _real(actual["frozen_source_score_bias"], label, nonnegative=True)
    _same_number(actual["source_reference_signed_score_difference"], source_delta, label + ".source_signed")
    _same_number(actual["source_reference_absolute_score_difference"], abs(source_delta), label + ".source_absolute")
    _same_number(actual["compiled_reference_absolute_score_difference"], abs(compiled_delta), label + ".compiled_absolute")
    _same_number(actual["source_reference_gap_minus_frozen_score_bias"], abs(source_delta) - frozen_bias, label + ".bias_gap")
    _keys(expected["reference"], ("baseline", "absolute_difference"), label + ".reference.policy")
    baseline = _real(expected["reference"]["baseline"], label)
    _bounded(abs(reference - baseline), expected["reference"]["absolute_difference"], label + ".independent_reference", summary)
    if kind == "instrument":
        _need(actual["saved_exact_reference_mean"] is None and expected["saved_exact_reference_mean"] is None
              and actual["independent_saved_exact_reference_signed_difference"] is None
              and actual["independent_saved_exact_reference_absolute_difference"] is None,
              label, "instrument saved-reference absence differs")
        _keys(expected["source_bias"], ("mode", "frozen", "absolute_difference"), label + ".source_bias.policy")
        _need(expected["source_bias"]["mode"] == "recorded_discrepancy", label, "instrument bias mode differs")
        _same_number(frozen_bias, expected["source_bias"]["frozen"], label + ".frozen_discrepancy")
        _bounded(abs(abs(source_delta) - frozen_bias), expected["source_bias"]["absolute_difference"],
                 label + ".recorded_source_discrepancy", summary)
    else:
        saved_reference = _real(actual["saved_exact_reference_mean"], label)
        _same_number(saved_reference, expected["saved_exact_reference_mean"], label + ".saved_reference")
        _same_number(saved_reference, baseline, label + ".reference_baseline")
        _same_number(actual["independent_saved_exact_reference_signed_difference"], reference - saved_reference, label + ".reference_signed")
        _same_number(actual["independent_saved_exact_reference_absolute_difference"], abs(reference - saved_reference), label + ".reference_absolute")
        _keys(expected["source_bias"], ("mode", "frozen"), label + ".source_bias.policy")
        _need(expected["source_bias"]["mode"] == "frozen_upper_bound", label, "complete bias mode differs")
        _same_number(frozen_bias, expected["source_bias"]["frozen"], label + ".frozen_upper_bound")
        # This is the original scientific inequality. Numerical reproduction
        # limits never add an epsilon to a failed source-task bound.
        _need(abs(source_delta) <= frozen_bias, label, "frozen source bias upper bound failed")
    summary["cases_checked"] += 1


def verify_numerical_replay(evidence, policy):
    """Validate one authenticated worker result using a reviewed observed policy.

    The function does not read files, import scientific libraries, execute a
    circuit, construct a policy or modify either argument. The caller owns the
    immutable policy/evidence provenance and full input authentication.
    """
    _keys(evidence, ("status", "kind", "cases", "runtime", "authenticated_helper_sha256",
                     "authenticated_lock_sha256", "numerical_compatibility_policy", "scope", "excluded"), "evidence")
    _keys(policy, ("schema", "status", "kind", "helper_sha256", "lock_sha256", "runtime", "basis", "cases"), "policy")
    kind = evidence["kind"]
    _need(isinstance(kind, str) and kind in CASES and policy["kind"] == kind, "kind", "unknown/differing kind")
    _need(policy["schema"] == POLICY_SCHEMA and policy["status"] == POLICY_STATUS, "policy", "policy is unreviewed or unsupported")
    _need(evidence["status"] == "evidence_collected"
          and evidence["numerical_compatibility_policy"] == UNAPPROVED_WORKER_MESSAGE,
          "evidence", "unexpected worker status")
    _need(evidence["scope"] == SCOPE and evidence["excluded"] == EXCLUDED, "evidence", "coverage/exclusions differ")
    for field, expected in (("authenticated_helper_sha256", "helper_sha256"), ("authenticated_lock_sha256", "lock_sha256")):
        _need(isinstance(policy[expected], str) and SHA.fullmatch(policy[expected])
              and evidence[field] == policy[expected], "identity", "helper/lock digest differs")
    _keys(evidence["runtime"], ("python", "platform", "machine", "packages"), "runtime")
    _keys(policy["runtime"], ("python_version", "packages", "machines"), "runtime.policy")
    runtime = evidence["runtime"]
    expected_runtime = policy["runtime"]
    _need(expected_runtime["python_version"] == "3.12.14"
          and isinstance(runtime["python"], str) and runtime["python"].startswith("3.12.14 "),
          "runtime", "Python patch differs")
    _need(runtime["packages"] == expected_runtime["packages"] == PACKAGES, "runtime", "locked package versions differ")
    machines = expected_runtime["machines"]
    _need(isinstance(machines, list) and machines
          and all(isinstance(value, str) and 0 < len(value) <= 64 for value in machines)
          and len(machines) == len(set(machines))
          and runtime["machine"] in machines and isinstance(runtime["platform"], str)
          and 0 < len(runtime["platform"]) <= 1024, "runtime", "missing/unsupported platform provenance")
    _keys(policy["basis"], ("evidence", "guard_rule", "certified"), "basis")
    _need(policy["basis"]["guard_rule"] == GUARD_RULE and policy["basis"]["certified"] is False,
          "basis", "unsupported/certified guard claim")
    basis = policy["basis"]["evidence"]
    _need(isinstance(basis, list) and len(basis) == 2, "basis", "local and Linux observations required")
    labels = set()
    for item in basis:
        _keys(item, ("label", "sha256"), "basis.evidence")
        _need(item["label"] in ("local", "linux") and item["label"] not in labels
              and isinstance(item["sha256"], str) and SHA.fullmatch(item["sha256"]), "basis", "invalid/duplicate evidence identity")
        labels.add(item["label"])
    _need(labels == {"local", "linux"}, "basis", "both observed platforms required")
    _keys(policy["cases"], CASES[kind], "policy.cases")
    _need(isinstance(evidence["cases"], list) and len(evidence["cases"]) == len(CASES[kind]), "evidence.cases", "case count differs")
    summary = {"status": "verified_observed_numerical_reproduction", "kind": kind, "checks": 0,
               "cases_checked": 0, "maximum_absolute_residual_metric": 0.0,
               "guard_rule": GUARD_RULE, "formal_roundoff_certificate": False,
               "scientific_budgets_changed": False, "scope": SCOPE, "excluded": EXCLUDED[:]}
    seen = set()
    for case in evidence["cases"]:
        _need(isinstance(case, dict) and case.get("case") in CASES[kind] and case["case"] not in seen,
              "evidence.cases", "unknown/duplicate case")
        name = case["case"]
        seen.add(name)
        _case(case, policy["cases"][name], kind, name, summary)
    _need(seen == set(CASES[kind]), "evidence.cases", "case set differs")
    return summary
