#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Authenticate new QPE records and their ledgers without synthesis/Native calls."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path("assets/generated/qpe-feedback-resources")
MAX_FILE_BYTES = 16*1024*1024
ROOT_FILES = {"manifest.json", "summary.csv", "summary.md", "logical-size-scaling.json", "README.md",
    "LICENSE.txt", "references.bib", "primitives.bib", "qpe_feedback_resources.py", "qft_qpe.py",
    "instrument_resources.py", "generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py",
    "test_qpe_feedback_resources.py", "requirements-lock.txt"}
TASK_FILES = {"report.json", "textbook-source.qasm", "feedback-source.qasm", "textbook-clifford-t.qasm",
              "feedback-clifford-t.qasm", "feedback-all-corrections-accounting-envelope.qasm"}


def read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing or nonregular retained file: {path.name}")
    with path.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES+1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Retained file exceeds16MiB: {path.name}")
    return data


def digest(path):
    return hashlib.sha256(read(path)).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-15)


def verify_saved(root=ROOT):
    root = Path(root)
    output = root/OUTPUT
    manifest = json.loads(read(output/"manifest.json"))
    expected = manifest["artifact_sha256"]
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() or p.is_symlink()}
    fixed = ROOT_FILES | {f"n1-m{m}/{name}" for m in (3, 4) for name in TASK_FILES}
    if actual != set(expected) | {"manifest.json"} or actual != fixed:
        raise ValueError("Retained QPE tree differs from its closed file allowlist")
    if any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Retained QPE tree contains a symlink")
    for name, expected_hash in expected.items():
        path = output/name
        if path.resolve().parent != output.resolve() and output.resolve() not in path.resolve().parents:
            raise ValueError("Retained artifact escaped its tree")
        if digest(path) != expected_hash:
            raise ValueError(f"Retained QPE artifact changed: {name}")
    for name, expected_hash in manifest["source_sha256"].items():
        if digest(root/name) != expected_hash or digest(output/Path(name).name) != expected_hash:
            raise ValueError(f"Current/snapshot QPE source identity changed: {name}")
    notebook = manifest["notebook"]
    if digest(root/notebook["path"]) != notebook["sha256"]:
        raise ValueError("Executed QPE comparison notebook changed")
    if manifest["source_dirty"] or manifest["hardware_execution"] or manifest["formal_confidence_certificate"]:
        raise ValueError("QPE provenance/confidence scope changed")
    for m in (3, 4):
        report = json.loads(read(output/f"n1-m{m}/report.json"))
        task = report["task"]
        if (task["phase"], task["phase_bits"], task["repetitions"], task["target_qubits"]) != (.3, m, 15, 1):
            raise ValueError("Frozen QPE problem changed")
        if (task["operator_error_allowance"], task["physical_failure_budget"], task["factory_availability_budget"]) != (2e-5, .0025, .0025):
            raise ValueError("Frozen QPE budgets changed")
        if not task["absolute_phase_error_target"] <= task["phase"] <= 1-task["absolute_phase_error_target"]:
            raise ValueError("Scalar-median branch-cut contract changed")
        if task["conditional_failure_upper_estimate"] >= .01:
            raise ValueError("Conditional QPE task failure exceeds its declared bound")
        if report["feedback_removed_max_error"] < .01:
            raise ValueError("Saved feedback negative control no longer discriminates")
        reference = report["independent_fourier_probabilities"]
        for variant in ("textbook", "feedback"):
            checks = report["checks"][variant]
            ideal, actual_probs = checks["ideal_probabilities"], checks["clifford_t_probabilities"]
            if len(ideal) != 2**m or len(actual_probs) != 2**m or not close(sum(ideal), 1) or not close(sum(actual_probs), 1):
                raise ValueError("Saved measured distribution has the wrong mass/domain")
            if max(abs(a-b) for a, b in zip(ideal, reference)) > 2e-13:
                raise ValueError("Saved original circuit/reference mismatch")
            tv = sum(abs(a-b) for a, b in zip(ideal, actual_probs))/2
            if not close(tv, checks["actual_synthesis_total_variation"]) or tv > checks["uniform_synthesis_tv_upper_estimate"]+1e-10:
                raise ValueError("Saved actual synthesis evidence changed")
            physical = report["physical"][variant]
            if physical["profile"]["physical_error_rate"] != 1e-4 or physical["complete_algorithm_attempts"] != 15:
                raise ValueError("Frozen QPE hardware/workload changed")
            for point in physical["pareto_points"]:
                rows = point["macro_ledger_per_complete_attempt"]
                runtime = sum(r["count"]*r["time_ns"] for r in rows)
                post = point["median_postprocessing_all_runs"]
                if post["controller_decisions"] != 15*m+105*(m+3)+1 or post["time_ns"] != post["controller_decisions"]*10_000:
                    raise ValueError("Fixed classical median accounting changed")
                if runtime != point["runtime_per_complete_attempt_ns"] or runtime*15+post["time_ns"] != point["worst_case_runtime_all_attempts_ns"]:
                    raise ValueError("Retained QPE macro time ledger changed")
                if not close(math.fsum(point["failure_components_all_attempts"].values()), point["full_physical_failure_model"]):
                    raise ValueError("Retained QPE fault component ledger changed")
                if point["full_physical_failure_model"] > .0025 or point["factory_supply"]["availability_failure_union_bound"] > .0025:
                    raise ValueError("Retained QPE physical/availability budget exceeded")
                factory = point["factory_supply"]
                cap = factory["full_pipeline_retry_cap_per_state"]
                states = factory["states_all_complete_attempts"]
                availability = Fraction(states)*Fraction(".01")**cap
                if not 0 < cap <= 16 or availability > Fraction(".0025") or not close(float(availability), factory["availability_failure_union_bound"]):
                    raise ValueError("Retained common factory retry cap changed")
                if variant == "feedback":
                    common = point["common_factory_configuration"]
                    if common["factory_supply"] != factory or common["data_code_distance"] != point["data_code_distance"] or common["factory_physical_qubits"] != point["factory_physical_qubits"]:
                        raise ValueError("Feedback common factory configuration changed")
                    predicates = m*(m-1)//2
                    controls = [r for r in rows if r["operation"] == "feedback_control"]
                    if len(controls) != 1 or controls[0]["count"] != predicates or controls[0]["time_ns"] != 10_000:
                        raise ValueError("False/true predicate exposure changed")
                    computed = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if computed != point["augmented_macro_ledger_sha256"]:
                        raise ValueError("Augmented feedback ledger identity changed")
        branches = report["feedback_branch_ledgers"]
        maximum = report["physical"]["feedback"]["branch_coverage"]["worst_case_counts_per_complete_run"]
        if [b["transcript"] for b in branches] != list(range(2**m)):
            raise ValueError("Provisioning lost an allowed transcript")
        for branch in branches:
            if branch["feedback_control"] != m*(m-1)//2 or any(not 0 <= count <= maximum.get(name, 0) for name, count in branch["event_counts"].items()):
                raise ValueError("A retained branch escapes its common envelope")
    return {"status": "saved QPE source/artifact identities and common physical ledgers verified", "native_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(verify_saved(parser.parse_args().root)))


if __name__ == "__main__":
    main()
