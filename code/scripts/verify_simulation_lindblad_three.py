#!/usr/bin/env python3
"""Verify the reviewed literal n3 bundle without synthesis or native queries.

This gate binds saved results to current shipped source, the exact reviewed
manifest, all 26 managed files and both inventories. It checks recorded
completion and scope; it does not repeat density simulation, certify numerical
premises or execute the 15,756-trial population.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/simulation/lindblad-three"
REVIEWED_MANIFEST_SHA256 = "9dde563b53b186272edb724de1b61865d497a2d00ea3368e05c8162b01d837f7"
PRIOR_N2_MANIFEST_SHA256 = "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"
FROZEN_BASELINE_FILES = frozenset({
    "src/openquantum/instrument_resources.py", "src/openquantum/grover.py", "scripts/estimate_resources.py",
    "assets/generated/resources/grover-surface-code.json", "assets/generated/resources/grover-surface-code.csv",
})
SOURCE_FILES = {
    "generate_simulation_lindblad_three.py": "scripts/generate_simulation_lindblad_three.py",
    "instrument_resources.py": "src/openquantum/instrument_resources.py",
    **{name + ".py": "src/openquantum/simulation/" + name + ".py" for name in (
        "bounded_density_instrument", "lindblad_three_physical", "lindblad_workload",
        "two_spin_lindblad", "open_systems")},
    "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt",
    "references.bib": None,
}
EXCLUDED_METADATA = ["artifact-inventory.csv", "artifact-inventory.json", "manifest.json", "progress.json"]
BUNDLE_FILES = frozenset(SOURCE_FILES) | frozenset({
    "README.md", "results.json", "summary.csv", "family-status.csv", *EXCLUDED_METADATA,
    "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
    "one-step.svg", "counts.svg", "error-budget.svg", "failures.svg", "space-time.svg",
})
SOURCE_COUNTS = {"cx": 1742, "rx": 804, "reset": 673, "cry": 670,
                 "rz": 536, "measure": 6, "h": 3, "s": 1}
ROUTED_COUNTS = {"cx": 5524, "barrier": 1358, "ry": 1340, "rx": 804,
                 "reset": 673, "rz": 536, "measure": 6, "h": 3, "s": 1}
CT_COUNTS = {"h": 281671, "t": 274432, "s": 142845, "cx": 5524,
             "barrier": 1358, "x": 1072, "reset": 673, "measure": 6}
LIMITS = {"max_qubits": 6, "source_operations": 20000, "synthesized_operations": 1000000,
          "validation_operations": 1000000, "validation_branches": 256}
CHECKER_LIMITS = {"max_qubits": 6, "source_operations": 20000,
                  "validation_operations": 1000000, "matrix_cache_bytes": 1048576}
PROFILE = {"physical_gate_time_ns": 100, "physical_measurement_time_ns": 500,
           "physical_initialization_time_ns": 100, "code_cycle_time_ns": 1000,
           "reaction_time_ns": 10000, "controller_failure_per_decision": 1e-15,
           "memory_prefactor": .1, "initialization_residual_prefactor": .1,
           "memory_threshold": .01, "factory_pipeline_failure_bound": .01,
           "factory_feedback_rounds_reserved": 3, "maximum_code_distance": 49}


def equal(actual, expected, label):
    # JSON comparison also distinguishes bools from integer identities.
    if isinstance(expected, (set, frozenset)):
        if not isinstance(actual, (set, frozenset)) or actual != expected:
            raise ValueError(f"n3 {label} differs from the reviewed contract")
        return
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError(f"n3 {label} differs from the reviewed contract")


def checked_file(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if (not name or "\\" in name or relative.is_absolute() or ".." in relative.parts
            or relative.as_posix() != name):
        raise ValueError(f"Unsafe n3 relative path: {name}")
    path = root / relative
    for ancestor in (root, *root.parents, *path.parents):
        if ancestor.is_symlink():
            raise ValueError(f"Linked n3 path: {path}")
    if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"Missing, special or oversized n3 file: {name}")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate n3 JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite n3 JSON value: {value}")
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            reject(value)
        return parsed
    result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)
    if not isinstance(result, dict):
        raise ValueError("n3 JSON record must be an object")
    return result


def verify_records(manifest: dict, results: dict, progress: dict) -> None:
    """Check semantic status/identity independently of the immutable hash gate."""
    try:
        for record in (manifest, results, progress):
            equal(record["schema_version"], 1, "schema")
        equal(manifest["python"], "3.12.14", "recorded interpreter")
        equal(manifest["packages"], {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4",
                                     "scipy": "1.18.1", "matplotlib": "3.11.2"}, "recorded packages")
        for key in ("private_cache_distributed", "whole_family_physical_completed"):
            equal(manifest[key], False, key)
        if not isinstance(manifest["frozen_before"], dict) or not isinstance(manifest["frozen_after"], dict):
            raise ValueError("n3 frozen baseline records must be objects")
        equal(set(manifest["frozen_before"]), FROZEN_BASELINE_FILES, "frozen baseline inventory")
        equal(manifest["frozen_before"], manifest["frozen_after"], "frozen baseline")
        prepared = results["prepared"]
        equal(prepared["case_id"], "lindblad-n3-r134-Z0", "case identity")
        equal(prepared["task"], {"n_system": 3, "time": 1.0, "J": .6, "hx": .35,
                                "rate0": .8, "other_rate": .5}, "input task")
        equal(prepared["selected_steps"], 134, "selected steps")
        equal(prepared["output_clbits"], [0, 1, 2], "output bits")
        equal(prepared["score_values"], [1., -1., 1., -1., 1., -1., 1., -1.], "correlated score")
        equal(prepared["limits"], LIMITS, "instruction limits")
        equal(prepared["checker_limits"], CHECKER_LIMITS, "density limits")
        equal(prepared["source_counts"], SOURCE_COUNTS, "source counts")
        if not isinstance(prepared["qasm_sha256"], dict):
            raise ValueError("n3 QASM digest record must be an object")
        equal(set(prepared["qasm_sha256"]), {"source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"},
              "QASM digest inventory")
        synthesis = prepared["actual_synthesis"]
        for key, expected in (("source_counts", SOURCE_COUNTS), ("routed_counts", ROUTED_COUNTS),
                              ("clifford_t_counts", CT_COUNTS), ("limits", LIMITS),
                              ("preserve_full_unitary", True), ("transpiler_qubits_initially_zero", False),
                              ("actual_synthesis_sequences_generated", True), ("formal_interval_certificate", False)):
            equal(synthesis[key], expected, "synthesis " + key)
        workload = prepared["complete_workload"]
        for key, expected in (("complete_trials", 15756), ("complete_attempts", 15756),
                              ("algorithm_retry_cap", 1), ("postselection", "none"),
                              ("whole_workload_executed", False), ("family_allocations_reallocated", False),
                              ("deterministic_bias_multiplied_by_trials", False)):
            equal(workload[key], expected, "workload " + key)
        primitive = {key: value for key, value in CT_COUNTS.items() if key != "barrier"}
        equal(workload["primitive_counts_per_complete_trial"], primitive, "trial primitive ledger")
        equal(workload["primitive_counts_all_trials"], {key: value * 15756 for key, value in primitive.items()},
              "whole-trial primitive ledger")
        for key in ("hardware_execution", "formal_confidence_certificate", "unobserved_reset_histories_are_measured_outputs"):
            equal(prepared[key], False, key)
        equal(prepared["joint_system_correlations_preserved"], True, "retained correlations")
        for key in ("full_representation_certificate", "formal_synthesis_certificate", "hardware_confidence_certificate"):
            equal(prepared["error_contract"][key], False, key)
        caps = manifest["run_caps"]
        equal(prepared["run_caps"], caps, "prepared caps")
        for key, expected in (("physical_cases", 1), ("physical_profiles", 2), ("density_wires", 6),
                              ("source_instructions", 20000), ("routed_and_synthesized_instructions", 1000000),
                              ("density_validation_instructions", 1000000), ("prepare_seconds", 1200),
                              ("physical_seconds_per_profile", 180), ("family_diagnostic_seconds", 60),
                              ("parent_seconds", 1800), ("native_rotation_synthesis_attempts_upper_bound", 10720)):
            equal(caps[key], expected, "run cap " + key)
        execution = manifest["execution"]
        equal(results["execution"], execution, "results execution identity")
        equal(progress["execution"], execution, "progress execution identity")
        equal(execution["run_caps"], caps, "execution caps")
        equal(progress["phase"], "completed_bounded_generation", "completion phase")
        for record in (manifest, progress):
            value = record["wall_seconds"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= caps["parent_seconds"]:
                raise ValueError("n3 total elapsed time is outside its recorded parent cap")
        for key in ("repeated_trials_executed", "whole_family_physical_completed"):
            equal(execution[key], False, key)
        equal(set(execution["stage_status"]), {"prepare", "p1e-03", "p1e-04", "family"}, "completed stages")
        for name, stage in execution["stage_status"].items():
            equal(stage["status"], "completed", "stage " + name)
            limit = caps["prepare_seconds" if name == "prepare" else "family_diagnostic_seconds"
                         if name == "family" else "physical_seconds_per_profile"]
            value = stage["wall_seconds"]
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= limit:
                raise ValueError("n3 stage timing is outside its recorded cap")
        checkpoint = execution["stage_status"]["prepare"]
        equal(checkpoint["checkpoint_schema"], "literal-n3-density-v1", "private checkpoint schema")
        for key in ("prepared_checkpoint_sha256", "checkpoint_identity"):
            if not isinstance(checkpoint[key], str) or re.fullmatch(r"[0-9a-f]{64}", checkpoint[key]) is None:
                raise ValueError("n3 private checkpoint identity is invalid")
        equal(results["family_allocations"], prepared["family_allocations"], "family allocations")
        allocations = results["family_allocations"]
        for key, expected in (("declared_family", [2, 3, 4]), ("implemented_physical_cases", [2, 3]),
                              ("new_physical_cases", [3]), ("preserved_prior_physical_cases", [2]),
                              ("unused_cases_reallocated", False), ("family_physical_confidence_implemented", False)):
            equal(allocations[key], expected, "family allocation " + key)
        equal(set(results["family"]), {"2", "3", "4"}, "family inventory")
        n2, n3, n4 = (results["family"][key] for key in ("2", "3", "4"))
        for key, expected in (("status", "preserved_prior_n2_result"), ("physical_status", "prior_result_preserved"),
                              ("regenerated", False), ("source_or_synthesis_rerun", False),
                              ("prior_manifest_path", "assets/generated/simulation/lindblad-physical/manifest.json"),
                              ("prior_manifest_sha256", PRIOR_N2_MANIFEST_SHA256)):
            equal(n2[key], expected, "preserved n2 " + key)
        equal(n3["status"], "implemented_six_wire_density_slice", "n3 family status")
        for key, expected in (("status", "ensemble_branch_cap_exceeded"),
                              ("physical_status", "unimplemented_reserved_allocation"),
                              ("density_checker_status", "density_width_cap_exceeded"),
                              ("density_required_wires", 8), ("density_supported_wires", 6),
                              ("density_width_rejection_reproduced", True), ("rejection_reproduced", True),
                              ("synthesis_performed", False), ("physical_estimation_performed", False),
                              ("family_allocation_reallocated", False)):
            equal(n4[key], expected, "n4 rejection " + key)
        profiles = results["physical_profiles"]
        equal(len(profiles), 2, "physical profile count")
        for profile, error_rate in zip(profiles, (.001, .0001)):
            equal(profile["profile"], {**PROFILE, "physical_error_rate": error_rate}, "fixed physical profile")
            for key, expected in (("case_id", prepared["case_id"]), ("complete_algorithm_attempts", 15756),
                                  ("limits", LIMITS), ("run_caps", caps), ("actual_synthesis", synthesis),
                                  ("source_event_counts", SOURCE_COUNTS), ("synthesized_event_counts", CT_COUNTS)):
                equal(profile[key], expected, "physical identity " + key)
            points, frontier = profile["all_feasible_points"], profile["pareto_points"]
            equal(profile["feasible_candidate_count"], len(points), "physical candidate count")
            if profile["status"] == "infeasible_declared_controller_model":
                equal(points, [], "infeasible candidates")
                equal(frontier, [], "infeasible frontier")
            elif profile["status"] == "computed_conditional_instrument_surface_code_model":
                if not points or not frontier or any(point not in points for point in frontier):
                    raise ValueError("n3 computed physical profile lacks its actual frontier")
                for point in points:
                    for key in ("peak_physical_qubits", "runtime_per_complete_attempt_ns", "worst_case_runtime_all_attempts_ns"):
                        if type(point[key]) is not int or point[key] <= 0:
                            raise ValueError("n3 physical point has fabricated or invalid cost")
            else:
                raise ValueError("n3 physical profile is incomplete or has an unexpected status")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("n3 completion/identity record is incomplete or malformed") from error


def verify_inventory(bundle: Path, manifest: dict, inventory: dict) -> None:
    try:
        equal(inventory["schema_version"], 1, "inventory schema")
        equal(inventory["excluded_metadata"], EXCLUDED_METADATA, "inventory exclusions")
        equal(inventory["originals_preserved"], True, "inventory preservation")
        equal(inventory["publication_layout_selected"], False, "inventory layout scope")
        entries = inventory["files"]
        names = [entry["file"] for entry in entries]
        equal(names, sorted(BUNDLE_FILES - set(EXCLUDED_METADATA)), "payload inventory")
        categories = {}
        for entry in entries:
            path = checked_file(bundle, entry["file"])
            equal(entry["bytes"], path.stat().st_size, "inventory byte count")
            equal(entry["sha256"], sha256(path), "inventory digest")
            category = categories.setdefault(entry["category"], {"files": 0, "bytes": 0})
            category["files"] += 1
            category["bytes"] += path.stat().st_size
        equal(inventory["categories"], categories, "inventory categories")
        equal(inventory["payload_files"], 22, "payload count")
        equal(inventory["payload_bytes"], sum(entry["bytes"] for entry in entries), "payload bytes")
        with checked_file(bundle, "artifact-inventory.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            equal(reader.fieldnames, ["file", "category", "bytes", "sha256"], "CSV inventory columns")
            rows = list(reader)
        for row in rows:
            row["bytes"] = int(row["bytes"])
        equal(rows, entries, "CSV/JSON inventory identity")
        equal(manifest["artifact_inventory"], {key: inventory[key] for key in (
            "payload_files", "payload_bytes", "excluded_metadata", "baseline_local_inventory")}, "manifest inventory identity")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("n3 artifact inventory is incomplete or malformed") from error


def verify_bundle(bundle: Path, *, source_root: Path | None = None, canonical_root: Path | None = None) -> dict:
    bundle = Path(bundle).absolute()
    source_root = bundle.parents[3] if source_root is None else Path(source_root).absolute()
    canonical_root = source_root if canonical_root is None else Path(canonical_root).absolute()
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("n3 bundle must be a regular directory")
    equal({path.name for path in bundle.iterdir()}, BUNDLE_FILES, "complete 26-file bundle")
    manifest = load_json(checked_file(bundle, "manifest.json"))
    results = load_json(checked_file(bundle, "results.json"))
    progress = load_json(checked_file(bundle, "progress.json"))
    verify_records(manifest, results, progress)
    if not isinstance(manifest.get("source_sha256"), dict) or not isinstance(manifest.get("artifacts_sha256"), dict):
        raise ValueError("n3 source/artifact manifests must be objects")
    equal(set(manifest["source_sha256"]), set(SOURCE_FILES), "source inventory")
    for name, relative in SOURCE_FILES.items():
        original = (checked_file(source_root, relative).read_bytes() if relative is not None else
                    checked_file(source_root, "references/simulation.bib").read_bytes() + b"\n" +
                    checked_file(source_root, "references/resources.bib").read_bytes())
        equal(checked_file(bundle, name).read_bytes().hex(), original.hex(), "current source snapshot " + name)
        equal(manifest["source_sha256"][name], hashlib.sha256(original).hexdigest(), "source digest " + name)
    equal(set(manifest["artifacts_sha256"]), BUNDLE_FILES - {"manifest.json", "progress.json"}, "attested artifact inventory")
    for name, digest in manifest["artifacts_sha256"].items():
        equal(digest, sha256(checked_file(bundle, name)), "artifact digest " + name)
    for name, digest in results["prepared"]["qasm_sha256"].items():
        equal(digest, sha256(checked_file(bundle, name)), "QASM digest " + name)
    for relative, digest in manifest["frozen_before"].items():
        location = canonical_root if relative.startswith("assets/") else source_root
        equal(digest, sha256(checked_file(location, relative)), "current frozen baseline " + relative)
    equal(sha256(checked_file(canonical_root, "assets/generated/simulation/lindblad-physical/manifest.json")),
          PRIOR_N2_MANIFEST_SHA256, "current preserved n2 manifest")
    verify_inventory(bundle, manifest, load_json(checked_file(bundle, "artifact-inventory.json")))
    equal(sha256(checked_file(bundle, "manifest.json")), REVIEWED_MANIFEST_SHA256, "immutable reviewed manifest")
    return {"files": len(BUNDLE_FILES), "bytes": sum(checked_file(bundle, name).stat().st_size for name in BUNDLE_FILES),
            "manifest_sha256": REVIEWED_MANIFEST_SHA256, "native_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source/package root with canonical generated assets")
    args = parser.parse_args()
    receipt = verify_bundle(args.root / BUNDLE_RELATIVE, source_root=args.root, canonical_root=args.root)
    print("Verified reviewed literal n3 bundle:", json.dumps(receipt, sort_keys=True),
          "record identity/status only; no native queries, simulation or hardware execution")


if __name__ == "__main__":
    main()
