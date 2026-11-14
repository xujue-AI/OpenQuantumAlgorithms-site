#!/usr/bin/env python3
"""Verify the reviewed n4 bundle using saved records and standard-library checks.

This gate binds the completed numerical trial evidence, all source/QASM bytes
and the explicit fixed-controller infeasibility. It does not rerun synthesis,
density simulation, a native estimator, or the repeated hardware population.
"""
from __future__ import annotations

import argparse
import ast
import csv
from fractions import Fraction
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_RELATIVE = "assets/generated/simulation/lindblad-four"
REVIEWED_MANIFEST_SHA256 = "5e78c5deac53d8106539ae75c5edbab4027ead7439c2bd4a2e491ec37834296f"
PRIOR_MANIFESTS = {
    "2": ("assets/generated/simulation/lindblad-physical/manifest.json",
          "1e60f7269ef96af64aeedd8655946dc72615c06960131b0e3add800627607595"),
    "3": ("assets/generated/simulation/lindblad-three/manifest.json",
          "9dde563b53b186272edb724de1b61865d497a2d00ea3368e05c8162b01d837f7"),
}
SOURCE_FILES = {
    "generate_simulation_lindblad_four.py": "scripts/generate_simulation_lindblad_four.py",
    "instrument_resources.py": "src/openquantum/instrument_resources.py",
    **{name + ".py": "src/openquantum/simulation/" + name + ".py" for name in (
        "bounded_density_instrument", "lindblad_workload", "two_spin_lindblad", "open_systems",
        "lindblad_four_density", "lindblad_four_compile", "lindblad_four_physical")},
    "LICENSE.txt": "LICENSE", "requirements-lock.txt": "requirements-lock.txt", "references.bib": None,
}
EXCLUDED_METADATA = ["artifact-inventory.csv", "artifact-inventory.json", "manifest.json", "progress.json"]
QASM_FILES = frozenset({"source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"})
BUNDLE_FILES = frozenset(SOURCE_FILES) | QASM_FILES | frozenset({
    "README.md", "results.json", "source-preflight.json", "summary.csv", "family-status.csv", *EXCLUDED_METADATA,
    "one-step.svg", "counts.svg", "error-budget.svg", "failures.svg", "space-time.svg",
})
FROZEN_FILES = frozenset({
    "src/openquantum/instrument_resources.py", "src/openquantum/grover.py", "scripts/estimate_resources.py",
    "assets/generated/resources/grover-surface-code.json", "assets/generated/resources/grover-surface-code.csv",
})
SOURCE_COUNTS = {"cx": 4009, "rx": 1688, "reset": 1481, "cry": 1477, "rz": 1266, "measure": 8, "h": 4, "s": 1}
LOWERED_COUNTS = {"cx": 6963, "barrier": 2978, "ry": 2954, "rx": 1688, "reset": 1481,
                  "rz": 1266, "measure": 8, "h": 4, "s": 1}
ROUTED_COUNTS = {**LOWERED_COUNTS, "cx": 14130}
CT_COUNTS = {"h": 661278, "t": 642284, "s": 349839, "cx": 14130, "x": 5908,
             "barrier": 2978, "reset": 1481, "measure": 8}
LIMITS = {"max_qubits": 8, "source_operations": 20000, "routed_operations": 100000, "synthesized_operations": 2000000}
CHECKER_LIMITS = {"max_qubits": 8, "source_operations": 20000, "validation_operations": 2000000, "matrix_cache_bytes": 1048576}
PACKAGES = {"qiskit": "2.6.2", "qdk": "1.34.0", "numpy": "2.5.4", "scipy": "1.18.1", "matplotlib": "3.11.2"}
PROFILE = {"physical_gate_time_ns": 100, "physical_measurement_time_ns": 500,
           "physical_initialization_time_ns": 100, "code_cycle_time_ns": 1000, "reaction_time_ns": 10000,
           "controller_failure_per_decision": 1e-15, "memory_prefactor": .1, "initialization_residual_prefactor": .1,
           "memory_threshold": .01, "factory_pipeline_failure_bound": .01,
           "factory_feedback_rounds_reserved": 3, "maximum_code_distance": 49}
T_WIRE_COUNTS = [57280, 58670, 102397, 104355, 101923, 103392, 56849, 57418]
SYNTHESIS_FUNCTION_SHA256 = "261658e6b31056a56c1989daad199eebc800e9cdefa4eca0c4fbe3cfa8db2bc2"
PREPARATION_FINGERPRINT = "ceb773241efc29432e2d66651a00ec4e6764da94aca5c3aaf646f7099ce9a431"
SOURCE_IDENTITY = "fd35742ea443dbc9e418d087a601aa481f13f852278c08eec96b0698c9b5fedb"
EXECUTION_HELPERS = (
    "stage_worker", "run_stage", "checkpoint_status", "preflight_status", "save_prepared", "load_prepared",
    "validate_private_job", "dump_qasm", "capture_sources", "publish_preflight", "recover_preflight",
    "verify_runtime_sources", "configure_import_policy", "read_managed_json", "bounded_file_hash",
    "prepared_hash", "export_captured_sources",
)
CAP_VALUES = {
    "physical_cases": 1, "prepared_synthesis_stages": 1, "physical_profiles": 2,
    "native_rotation_synthesis_attempts_upper_bound": 23632, "source_instructions": 20000,
    "routed_instructions": 100000, "synthesized_instructions": 2000000,
    "routed_and_synthesized_instructions": 2000000, "density_validation_instructions": 2000000,
    "density_wires": 8, "density_matrix_entries": 65536, "density_matrix_bytes": 1048576,
    "unused_family_ensemble_branches": 256, "family_diagnostic_seconds": 180, "source_preflight_seconds": 180,
    "diagnostic_aggregate_seconds": 180, "prepare_seconds": 1800, "physical_seconds_per_profile": 180,
    "parent_seconds": 2400, "factory_queries_per_profile": 24, "surface_code_queries_per_profile": 24,
    "factory_candidates_per_query": 1000, "factory_candidates_per_profile": 24000,
    "managed_json_qasm_bytes": 33554432, "private_checkpoint_bytes": 268435456,
    "nominal_state_trace_guard": 1e-8, "nominal_density_comparison_guard": 1e-10,
}


def equal(actual, expected, label):
    if isinstance(expected, (set, frozenset)):
        valid = isinstance(actual, (set, frozenset)) and actual == expected
    else:
        valid = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    if not valid:
        raise ValueError(f"n4 {label} differs from the reviewed saved contract")


def finite_number(value, label, *, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or value < minimum or (maximum is not None and value > maximum)):
        raise ValueError(f"n4 {label} is nonfinite or outside its recorded bound")


def checked_file(root: Path, name: str) -> Path:
    if not isinstance(name, str):
        raise ValueError("n4 relative file name must be a string")
    relative = PurePosixPath(name)
    if not name or "\\" in name or relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError(f"Unsafe n4 relative path: {name}")
    path = root / relative
    if any(ancestor.is_symlink() for ancestor in (root, *root.parents, *path.parents)):
        raise ValueError(f"Linked n4 ancestor: {name}")
    if not path.is_file() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"Missing, linked, special or oversized n4 file: {name}")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("n4 saved JSON exceeds its two-MiB parsing cap")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate n4 JSON key: {key}")
            result[key] = value
        return result
    def reject(value):
        raise ValueError(f"Nonfinite n4 JSON value: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            reject(value)
        return result
    try:
        result = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=reject, parse_float=finite_float)
    except (json.JSONDecodeError, RecursionError, UnicodeError) as error:
        raise ValueError("n4 saved JSON is malformed or exceeds its nesting limit") from error
    if not isinstance(result, dict):
        raise ValueError("n4 JSON record must contain an object")
    def bounded_depth(value, depth=0):
        if depth > 128:
            raise ValueError("n4 saved JSON exceeds its 128-level nesting cap")
        if isinstance(value, dict):
            for child in value.values():
                bounded_depth(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                bounded_depth(child, depth + 1)
    bounded_depth(result)
    return result


def function_source(path, name):
    """Recover inspect-compatible function bytes without importing its module."""
    text = path.read_text()
    try:
        nodes = [node for node in ast.parse(text).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name]
        if len(nodes) != 1:
            raise ValueError("n4 required frozen function is missing or repeated")
        node = nodes[0]
        start = min([node.lineno, *(item.lineno for item in node.decorator_list)])
        return "".join(inspect.getblock(text.splitlines(keepends=True)[start - 1:])).encode()
    except (SyntaxError, IndentationError) as error:
        raise ValueError("n4 frozen function source is malformed") from error


def verify_runtime_identity(manifest, source_root):
    runner = checked_file(source_root, SOURCE_FILES["generate_simulation_lindblad_four.py"])
    hashes = {name: value for name, value in manifest["source_sha256"].items()
              if name.endswith(".py") and not name.startswith("generate_")}
    identity = {**hashes, "checkpoint_schema": "literal-n4-density-v1", "import_mode": "package",
                "python": "3.12.14", "packages": PACKAGES}
    keys = ("native_rotation_synthesis_attempts_upper_bound", "prepare_seconds", "source_instructions",
            "routed_and_synthesized_instructions", "density_validation_instructions", "density_wires", "private_checkpoint_bytes")
    identity["preparation_execution_policy"] = {
        "caps": {key: manifest["run_caps"][key] for key in keys},
        "native_guard_source_sha256": hashlib.sha256(function_source(runner, "bounded_rotation_synthesis")).hexdigest(),
        "execution_helpers_sha256": {name: hashlib.sha256(function_source(runner, name)).hexdigest() for name in EXECUTION_HELPERS},
        "preserve_full_unitary": True,
        "import_policy": "fresh task-private PYTHONPYCACHEPREFIX and PYTHONDONTWRITEBYTECODE=1",
    }
    equal(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(), SOURCE_IDENTITY,
          "current source/function/execution-policy identity")
    equal(hashlib.sha256(function_source(checked_file(source_root, "src/openquantum/instrument_resources.py"),
                                        "synthesize_instrument")).hexdigest(), SYNTHESIS_FUNCTION_SHA256,
          "current frozen synthesis-function identity")


def verify_records(manifest, results, progress, source_preflight):
    """Check saved completion and infeasibility semantics before hash preflight."""
    try:
        for record in (manifest, results, progress):
            equal(record["schema_version"], 1, "schema")
        for key, value in (("python", "3.12.14"), ("packages", PACKAGES), ("source_preflight_completed", True),
                           ("full_clifford_t_trial_completed", True), ("private_cache_distributed", False),
                           ("whole_family_physical_completed", False)):
            equal(manifest[key], value, "manifest " + key)
        for key, names in (("source_sha256", set(SOURCE_FILES)), ("artifacts_sha256", BUNDLE_FILES - {"manifest.json", "progress.json"}),
                           ("frozen_before", FROZEN_FILES), ("frozen_after", FROZEN_FILES)):
            if not isinstance(manifest[key], dict):
                raise ValueError("n4 manifests must be objects")
            equal(set(manifest[key]), names, key + " inventory")
        equal(manifest["frozen_before"], manifest["frozen_after"], "frozen baseline preservation")
        caps, execution = manifest["run_caps"], manifest["execution"]
        for key, value in CAP_VALUES.items():
            equal(caps[key], value, "run cap " + key)
        equal(results["execution"], execution, "results execution identity")
        equal(progress["execution"], execution, "progress execution identity")
        equal(execution["run_caps"], caps, "execution caps")
        equal(progress["phase"], "completed_bounded_generation", "progress completion")
        for record in (manifest, progress):
            finite_number(record["wall_seconds"], "total elapsed time", maximum=2400)
        for key in ("repeated_trials_executed", "whole_family_physical_completed"):
            equal(execution[key], False, "unexecuted " + key)
        finite_number(execution["diagnostic_aggregate_wall_seconds"], "diagnostic aggregate elapsed time", maximum=180)
        runtime = {"import_mode": "package", "actual_module_paths_verified": True,
                   "source_sha256": {name: digest for name, digest in manifest["source_sha256"].items()
                                     if name.endswith(".py") and not name.startswith("generate_")},
                   "bytecode_writes_disabled": True, "bytecode_prefix_configured": True,
                   "import_policy": "fresh private bytecode prefix; bytecode writes disabled; no environment mutation"}
        equal(manifest["runtime_source_validation"], runtime, "recorded imported-source identity")
        stages = execution["stage_status"]
        equal(set(stages), {"source_preflight", "family", "prepare", "p1e-03", "p1e-04"}, "five completed stages")
        for name, stage in stages.items():
            equal(stage["status"], "completed", "stage status " + name)
            equal(stage["runtime_source_validation"], runtime, "stage source/import identity " + name)
            finite_number(stage["wall_seconds"], "stage elapsed time " + name, maximum=1800 if name == "prepare" else 180)
        preflight_stage, prepare_stage = stages["source_preflight"], stages["prepare"]
        for key, value in (("source_identity", SOURCE_IDENTITY), ("source_schema", "literal-n4-density-v1"),
                           ("source_record_sha256", manifest["artifacts_sha256"]["source-preflight.json"]),
                           ("invocation_id", "7f5071f7941ffb59a5f2c2fca97d6b13")):
            equal(preflight_stage[key], value, "source preflight " + key)
        for key, value in (("checkpoint_identity", SOURCE_IDENTITY), ("checkpoint_schema", "literal-n4-density-v1"),
                           ("native_rotation_synthesis_attempts", 8), ("native_rotation_synthesis_attempts_upper_bound", 23632)):
            equal(prepare_stage[key], value, "prepared checkpoint " + key)
        if not isinstance(prepare_stage["prepared_checkpoint_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", prepare_stage["prepared_checkpoint_sha256"]) is None:
            raise ValueError("n4 private prepared-checkpoint digest is invalid")
        equal(results["source_preflight"], source_preflight, "saved source-preflight record")
        equal(source_preflight["status"], "checked_literal_source_and_routed_rotations", "source-only completion")
        prepared = results["prepared"]
        for record in (source_preflight, prepared):
            for key, value in (("case_id", "lindblad-n4-r211-Z0"), ("selected_steps", 211),
                               ("task", {"n_system": 4, "time": 1., "J": .6, "hx": .35, "rate0": .8, "other_rate": .5}),
                               ("source_counts", SOURCE_COUNTS), ("limits", LIMITS), ("materialization_limits", LIMITS),
                               ("checker_limits", CHECKER_LIMITS), ("output_clbits", [0, 1, 2, 3]),
                               ("score_values", [1., -1.] * 8), ("run_caps", caps),
                               ("joint_system_correlations_preserved", True), ("unobserved_reset_histories_are_measured_outputs", False),
                               ("hardware_execution", False), ("formal_confidence_certificate", False)):
                equal(record[key], value, "literal task " + key)
            equal(record["representation_check"]["representation_reserve_is_certified"], False,
                  "uncertified representation reserve")
        for key in ("synthesis_performed", "physical_estimation_performed"):
            equal(source_preflight[key], False, "source-only " + key)
        routing, synthesis = source_preflight["actual_routing"], prepared["actual_synthesis"]
        derivation = {"frozen_ir_source_sha256": manifest["source_sha256"]["instrument_resources.py"],
                      "frozen_ir_synthesis_function_sha256": SYNTHESIS_FUNCTION_SHA256,
                      "scope": "full-unitary routing and numerical rotation policy copied into a separate explicit materialization domain; shared defaults and physical formulas unchanged",
                      "shared_IR_limits_modified": False, "shared_IR_report_limits_forged": False}
        for record in (routing, synthesis):
            for key, value in (("compiler_domain", "literal-n4-explicit-opt-in-v1"), ("materialization_limits", LIMITS),
                               ("derivation", derivation), ("qiskit_version", "2.6.2"), ("preserve_full_unitary", True),
                               ("transpiler_qubits_initially_zero", False), ("source_counts", SOURCE_COUNTS),
                               ("lowered_counts", LOWERED_COUNTS), ("routed_counts", ROUTED_COUNTS),
                               ("rotation_occurrences", 5908), ("source_final_measurement_wires", list(range(8))),
                               ("routed_final_measurement_wires", [6, 1, 2, 3, 7, 4, 0, 5]),
                               ("source_global_phase", 0.), ("routed_global_phase", 0.), ("physical_estimation_performed", False)):
                equal(record[key], value, "compiler domain " + key)
            if "limits" in record:
                raise ValueError("n4 compiler must not impersonate the shared limits record")
        equal(routing["actual_clifford_t_synthesis_performed"], False, "source-only synthesis")
        for key, value in (("actual_clifford_t_synthesis_performed", True), ("actual_synthesis_sequences_generated", True),
                           ("clifford_t_counts", CT_COUNTS), ("clifford_t_final_measurement_wires", [6, 1, 2, 3, 7, 4, 0, 5]),
                           ("unique_rotation_targets", 8), ("native_rotation_synthesis_attempts", 8),
                           ("native_rotation_synthesis_attempts_upper_bound", 23632), ("operator_error_allowance", 5e-5),
                           ("guarded_numerical_operator_error_upper_estimate", 8.769373339946128e-6),
                           ("per_matrix_frobenius_guard", 1e-12), ("formal_interval_certificate", False)):
            equal(synthesis[key], value, "completed synthesis " + key)
        equal(prepared["preparation_fingerprint"], PREPARATION_FINGERPRINT, "saved prepared fingerprint")
        equal(prepared["execution_report_fields_excluded_from_preparation_fingerprint"],
              ["native_rotation_synthesis_attempts", "native_rotation_synthesis_attempts_upper_bound"], "fingerprint exclusions")
        equal(prepared["qasm_export"]["in_memory_global_phase_radians"],
              {"source-trial.qasm": 0., "routed-rotation-trial.qasm": 0., "clifford-t-trial.qasm": synthesis["compiled_global_phase"]},
              "recorded measured-QASM phases")
        equal(synthesis["compiled_global_phase"], 6.28318530717827, "saved complete coherent phase")
        for record, keys in ((source_preflight, QASM_FILES - {"clifford-t-trial.qasm"}), (prepared, QASM_FILES)):
            if not isinstance(record["qasm_sha256"], dict):
                raise ValueError("n4 QASM bindings must be objects")
            equal(set(record["qasm_sha256"]), keys, "QASM digest inventory")
        for name in ("source_instrument_check", "routed_instrument_check"):
            equal(prepared[name], source_preflight[name], "source/prepared numerical evidence " + name)
        for name, count in (("source_instrument_check", 9934), ("routed_instrument_check", 24510), ("clifford_t_instrument_check", 1677906)):
            check = prepared[name]
            for key in ("normalization_is_postselection", "normalization_probability_is_postselection"):
                equal(check[key], False, "numerical normalization scope")
            equal(check["outcome_clbits"], [0, 1, 2, 3], "correlated output bits")
            equal(check["outcome_wires"], [0, 1, 2, 3] if name == "source_instrument_check" else [6, 1, 2, 3],
                  "actual correlated output targets")
            equal(check["nominal_state_trace_guard"], 1e-8, "predeclared state/trace guard")
            equal(check["nominal_density_comparison_guard"], 1e-10, "predeclared density comparison guard")
            equal(len(check["joint_probabilities"]), 16, "joint outcome count")
            equal(set(check["reduced_before_output"]), {"real", "imag"}, "saved complex density")
            for values in check["reduced_before_output"].values():
                equal(len(values), 16, "reduced density rows")
                if any(len(row) != 16 for row in values):
                    raise ValueError("n4 reduced density is not the full 16x16 state")
            simulation = check["simulation"]
            for key, value in (("limits", CHECKER_LIMITS), ("density_shape", [256, 256]), ("density_state_bytes", 1048576),
                               ("top_level_instructions", count), ("explicit_instrument_counts", {"reset": 1481, "measure": 8}),
                               ("reset_history_measured_outputs", False), ("reset_history_classical_bits_created", 0),
                               ("trajectory_branch_pruning", False), ("numerical_state_guard", 1e-8)):
                equal(simulation[key], value, "saved density semantics " + key)
        workload = prepared["complete_workload"]
        primitives = {key: value for key, value in CT_COUNTS.items() if key != "barrier"}
        for key, value in (("complete_trials", 15756), ("complete_attempts", 15756), ("algorithm_retry_cap", 1),
                           ("postselection", "none"), ("whole_workload_executed", False),
                           ("primitive_counts_per_complete_trial", primitives),
                           ("primitive_counts_all_trials", {key: value * 15756 for key, value in primitives.items()}),
                           ("reset_per_trial", 1481), ("readout_per_trial", 8), ("source_resets_all_trials", 23334636),
                           ("source_readouts_all_trials", 126048), ("workspace_startup_initializations_per_trial", 23),
                           ("workspace_startup_initializations_all_trials", 362388), ("source_first_wire_initializations_per_trial", 8),
                           ("source_reuse_resets_per_trial", 1473), ("reserved_logical_patches", 31),
                           ("logical_startup_initializations_all_trials", 488436), ("family_allocations_reallocated", False),
                           ("deterministic_bias_multiplied_by_trials", False), ("unobserved_reset_histories_claimed_as_outputs", False)):
            equal(workload[key], value, "complete workload " + key)
        for key, denominator in (("statistical_failure_exact", 600), ("physical_failure_exact", 1200),
                                 ("factory_availability_failure_exact", 1200), ("conditional_case_failure_exact", 300)):
            equal(workload[key], {"numerator": 1, "denominator": denominator}, "exact workload allowance " + key)
        exact_budget = Fraction(1, 1200)
        budget = float(exact_budget)
        if Fraction.from_float(budget) > exact_budget:
            budget = math.nextafter(budget, 0.)
        for key in ("physical_failure_budget", "factory_availability_budget"):
            equal(workload[key], budget, "downward case allowance " + key)
        contract = prepared["error_contract"]
        for key, value in (("splitting_bias_exact", {"numerator": 1331, "denominator": 133563}),
                           ("splitting_bias_allowance", .01), ("realization_bias_allowance", .01),
                           ("statistical_absolute_error", .03), ("target_absolute_error", .05),
                           ("requested_synthesis_allowance", 5e-5), ("representation_norm_reserve_scenario", 1e-7),
                           ("SDK_control_routing_norm_reserve_scenario", 1e-7), ("full_representation_certificate", False),
                           ("formal_synthesis_certificate", False), ("hardware_confidence_certificate", False),
                           ("reserve_is_inferred_from_density_agreement", False), ("coherent_error_is_stochastic_failure", False)):
            equal(contract[key], value, "conditional error contract " + key)
        allocations = results["family_allocations"]
        equal(prepared["family_allocations"], allocations, "prepared allocation identity")
        equal(source_preflight["family_allocations"], allocations, "preflight allocation identity")
        for key, value in (("declared_family", [2, 3, 4]), ("implemented_physical_cases", [2, 3]),
                           ("new_physical_cases", [4]), ("preserved_prior_physical_cases", [2, 3]),
                           ("unused_cases_reallocated", False), ("family_physical_confidence_implemented", False)):
            equal(allocations[key], value, "preserved allocation " + key)
        for key, denominator in (("statistics_family_failure_exact", 200), ("physical_family_failure_exact", 400),
                                 ("factory_family_failure_exact", 400), ("total_conditional_family_failure_exact", 100)):
            equal(allocations[key], {"numerator": 1, "denominator": denominator}, "exact family allowance " + key)
        equal(set(allocations["case_allocations"]), {"2", "3", "4"}, "exact case-allocation inventory")
        for n, steps in (("2", 69), ("3", 134), ("4", 211)):
            allocation = allocations["case_allocations"][n]
            for key, value in (("selected_steps", steps), ("complete_trials", 15756),
                               ("statistical_failure_exact", {"numerator": 1, "denominator": 600}),
                               ("physical_failure_exact", {"numerator": 1, "denominator": 1200}),
                               ("factory_availability_failure_exact", {"numerator": 1, "denominator": 1200}),
                               ("physical_failure_budget", budget), ("factory_availability_budget", budget)):
                equal(allocation[key], value, "preserved case allowance n" + n + " " + key)
        equal(set(results["family"]), {"2", "3", "4"}, "family inventory")
        for n, steps in (("2", 69), ("3", 134)):
            prior = results["family"][n]
            for key, value in (("selected_steps", steps), ("complete_trials", 15756),
                               ("status", f"preserved_prior_n{n}_result"), ("physical_status", "prior_result_preserved"),
                               ("regenerated", False), ("source_or_synthesis_rerun", False),
                               ("prior_manifest_path", PRIOR_MANIFESTS[n][0]), ("prior_manifest_sha256", PRIOR_MANIFESTS[n][1])):
                equal(prior[key], value, "preserved prior n" + n + " " + key)
        old = results["family"]["4"]
        for key, value in (("status", "ensemble_branch_cap_exceeded"), ("physical_status", "historical_unimplemented_reserved_allocation_preserved"),
                           ("density_checker_status", "density_width_cap_exceeded"), ("density_required_wires", 8),
                           ("density_supported_wires", 6), ("density_width_rejection_reproduced", True), ("rejection_reproduced", True),
                           ("legacy_synthesis_performed", False), ("legacy_physical_estimation_performed", False),
                           ("family_allocation_reallocated", False)):
            equal(old[key], value, "historical rejection " + key)
        legacy = prepared["legacy_materialization_validation"]
        for key, value in (("legacy_operation_cap", 1000000), ("actual_operations", 1677906),
                           ("legacy_synthesis_performed", False), ("legacy_physical_estimation_performed", False),
                           ("status", "rejected_actual_clifford_t_operations"), ("rejection_reproduced", True)):
            equal(legacy[key], value, "actual completed-trial legacy validation " + key)
        equal(len(results["physical_profiles"]), 2, "two fixed physical profiles")
        event_counts = {"initial_reset": 31, "h": 661278, "s": 349839, "cx": 14130, "x": 5908,
                        "reset": 1473, "measure": 8, "trial_control": 1,
                        **{f"t_q{wire}": count for wire, count in enumerate(T_WIRE_COUNTS)}}
        for profile, rate in zip(results["physical_profiles"], (.001, .0001)):
            for key, value in (("case_id", prepared["case_id"]), ("profile", {**PROFILE, "physical_error_rate": rate}),
                               ("complete_algorithm_attempts", 15756), ("physical_failure_budget", budget),
                               ("factory_availability_budget", budget), ("family_allocations", allocations), ("run_caps", caps),
                               ("legacy_materialization_validation", legacy), ("new_compiler_domain", "literal-n4-explicit-opt-in-v1"),
                               ("status", "infeasible_declared_controller_model"), ("pareto_points", []), ("all_feasible_points", []),
                               ("native_estimation_performed", False), ("native_trace_calls_upper_bound", 0),
                               ("native_trace_calls_instrumented", False),
                               ("execution_query_counts", {"top_level_factory_queries": 0, "top_level_surface_code_queries": 0,
                                                           "factory_candidates": 0, "factory_candidates_per_query": []})):
                equal(profile[key], value, "infeasible physical record " + key)
            forbidden = {"peak_physical_qubits", "runtime_per_complete_attempt_ns", "worst_case_runtime_all_attempts_ns",
                         "data_code_distance", "full_physical_failure_model", "failure_components_all_attempts", "feasible_candidate_count"}
            if forbidden & set(profile):
                raise ValueError("n4 rejected physical profile must retain unassessed costs, not fabricated zero values")
            diagnostic = profile["controller_model_diagnostic"]
            for key, value in (("event_counts_per_trial", event_counts), ("required_magic_states_per_trial", 642284),
                               ("required_magic_states_all_trials", 10119826704), ("controller_decisions_per_trial", 123409809),
                               ("controller_decisions_all_trials", 1944444950604), ("controller_failure_per_decision_scenario", 1e-15),
                               ("distance_independent_controller_model_contribution", .001944444950604),
                               ("physical_failure_budget", budget), ("maximum_allowed_controller_decisions_per_trial", 52889904),
                               ("budget_exceeded", True), ("factory_retry_cap", 7), ("factory_readiness_failure_union_bound", .00010119826704)):
                equal(diagnostic[key], value, "controller/readiness ledger " + key)
            if not Fraction.from_float(diagnostic["distance_independent_controller_model_contribution"]) > Fraction(1, 1200):
                raise ValueError("n4 controller rejection no longer exceeds its fixed allowance")
            states = diagnostic["required_magic_states_all_trials"]
            if not Fraction(states, 100**7) <= Fraction(1, 1200) < Fraction(states, 100**6):
                raise ValueError("n4 factory retry cap is not minimal at the fixed allowance")
            proposal = profile["matched_task_implementation_proposal"]
            for key, value in (("status", "planned_matched_task_implementation_proposal"), ("task_changed", False),
                               ("accuracy_allocations_hardware_geometry_controller_changed", False),
                               ("proposed_implementation_executed", False), ("feasibility_claimed", False)):
                equal(proposal[key], value, "planned matched-task proposal " + key)
            for key in ("implementation_executed", "feasibility_guaranteed"):
                equal(proposal["primary_block_porting_recipe"][key], False, "unimplemented block recipe " + key)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("n4 completion/identity/status record is incomplete or malformed") from error


def verify_inventory(bundle, manifest, inventory):
    try:
        for key, value in (("schema_version", 1), ("excluded_metadata", EXCLUDED_METADATA),
                           ("originals_preserved", True), ("publication_layout_selected", False),
                           ("payload_files", 25), ("payload_bytes", 15301374)):
            equal(inventory[key], value, "payload inventory " + key)
        entries = inventory["files"]
        equal([entry["file"] for entry in entries], sorted(BUNDLE_FILES - set(EXCLUDED_METADATA)), "25 payload identities")
        categories = {}
        for entry in entries:
            path = checked_file(bundle, entry["file"])
            equal(entry["bytes"], path.stat().st_size, "inventory byte count")
            equal(entry["sha256"], sha256(path), "inventory digest")
            category = categories.setdefault(entry["category"], {"files": 0, "bytes": 0})
            category["files"] += 1
            category["bytes"] += path.stat().st_size
        equal(inventory["categories"], categories, "category totals")
        equal(sum(entry["bytes"] for entry in entries), 15301374, "actual payload bytes")
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
        raise ValueError("n4 saved inventory is incomplete or malformed") from error


def verify_bundle(bundle: Path, *, source_root: Path | None = None, canonical_root: Path | None = None):
    bundle = Path(bundle).absolute()
    source_root = bundle.parents[3] if source_root is None else Path(source_root).absolute()
    canonical_root = source_root if canonical_root is None else Path(canonical_root).absolute()
    if bundle.is_symlink() or not bundle.is_dir():
        raise ValueError("n4 bundle must be a regular directory")
    equal({path.name for path in bundle.iterdir()}, BUNDLE_FILES, "complete 29-file bundle")
    # Reject a coordinated unreviewed source/record rewrite before parsing or
    # reconstructing its source-policy identity.
    equal(sha256(checked_file(bundle, "manifest.json")), REVIEWED_MANIFEST_SHA256, "immutable reviewed manifest")
    manifest = load_json(checked_file(bundle, "manifest.json"))
    results = load_json(checked_file(bundle, "results.json"))
    progress = load_json(checked_file(bundle, "progress.json"))
    source_preflight = load_json(checked_file(bundle, "source-preflight.json"))
    verify_records(manifest, results, progress, source_preflight)
    for name, relative in SOURCE_FILES.items():
        raw = (checked_file(source_root, relative).read_bytes() if relative is not None else
               checked_file(source_root, "references/simulation.bib").read_bytes() + b"\n" +
               checked_file(source_root, "references/resources.bib").read_bytes())
        equal(hashlib.sha256(raw).hexdigest(), manifest["source_sha256"][name], "current source digest " + name)
        equal(sha256(checked_file(bundle, name)), manifest["source_sha256"][name], "current source snapshot " + name)
    for name, digest in manifest["artifacts_sha256"].items():
        equal(sha256(checked_file(bundle, name)), digest, "attested artifact digest " + name)
    for record in (source_preflight, results["prepared"]):
        for name, digest in record["qasm_sha256"].items():
            equal(sha256(checked_file(bundle, name)), digest, "complete QASM identity " + name)
    for name, digest in manifest["frozen_before"].items():
        root = canonical_root if name.startswith("assets/") else source_root
        equal(sha256(checked_file(root, name)), digest, "current frozen baseline " + name)
    for n, (name, digest) in PRIOR_MANIFESTS.items():
        equal(sha256(checked_file(canonical_root, name)), digest, "current preserved n" + n + " manifest")
    verify_runtime_identity(manifest, source_root)
    verify_inventory(bundle, manifest, load_json(checked_file(bundle, "artifact-inventory.json")))
    return {"files": 29, "bytes": sum(checked_file(bundle, name).stat().st_size for name in BUNDLE_FILES),
            "manifest_sha256": REVIEWED_MANIFEST_SHA256, "saved_numerical_trial_checks_completed": True,
            "physical_status": "infeasible_declared_controller_model", "native_calls": 0,
            "synthesis_calls": 0, "density_replays": 0, "repeated_trials_executed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Source/package root retaining canonical generated assets")
    args = parser.parse_args()
    receipt = verify_bundle(args.root / BUNDLE_RELATIVE, source_root=args.root, canonical_root=args.root)
    print("Verified immutable n4 saved evidence:", json.dumps(receipt, sort_keys=True),
          "identity/status checks only; no synthesis, native estimation, density replay or repeated hardware execution")


if __name__ == "__main__":
    main()
