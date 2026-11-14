"""Saved n4 publication must preserve source, failure scope and exact ledgers."""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("n4_release_gate", ROOT / "scripts/verify_simulation_lindblad_four.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def records():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    return [gate.load_json(bundle / name) for name in ("manifest.json", "results.json", "progress.json", "source-preflight.json")]


@pytest.fixture
def package(tmp_path):
    root = (tmp_path / "package").resolve()
    shutil.copytree(ROOT / gate.BUNDLE_RELATIVE, root / gate.BUNDLE_RELATIVE)
    paths = {path for path in gate.SOURCE_FILES.values() if path is not None}
    paths |= {"references/simulation.bib", "references/resources.bib"} | set(gate.FROZEN_FILES)
    paths |= {entry[0] for entry in gate.PRIOR_MANIFESTS.values()}
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


def verify(root):
    return gate.verify_bundle(root / gate.BUNDLE_RELATIVE, source_root=root, canonical_root=root)


def set_value(record, keys, value):
    current = record
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_complete_reviewed_n4_bundle_has_completed_numerical_trial_and_unexecuted_population(package):
    receipt = verify(package)
    assert receipt == {"files": 29, "bytes": 15338779, "manifest_sha256": gate.REVIEWED_MANIFEST_SHA256,
                       "saved_numerical_trial_checks_completed": True,
                       "physical_status": "infeasible_declared_controller_model", "native_calls": 0,
                       "synthesis_calls": 0, "density_replays": 0, "repeated_trials_executed": False}


def test_source_preview_and_canonical_asset_roots_verify_independently(package):
    code = package / "code"
    for relative in {path for path in gate.SOURCE_FILES.values() if path is not None} | {
            "references/simulation.bib", "references/resources.bib"} | {
            path for path in gate.FROZEN_FILES if not path.startswith("assets/")}:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, target)
    assert gate.verify_bundle(package / gate.BUNDLE_RELATIVE, source_root=code, canonical_root=package)["native_calls"] == 0
    assert not (code / "assets").exists()


def test_gate_enforces_zero_scientific_imports_and_native_calls(package, monkeypatch):
    original = builtins.__import__
    forbidden = {"openquantum", "qiskit", "qdk", "numpy", "scipy"}
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in forbidden:
            raise AssertionError("n4 immutable gate imported a scientific/native package")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert verify(package)["native_calls"] == 0
    tree = ast.parse((ROOT / "scripts/verify_simulation_lindblad_four.py").read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports <= sys.stdlib_module_names


@pytest.mark.parametrize("relative", sorted({value for value in gate.SOURCE_FILES.values() if value is not None}
                                           | {"references/simulation.bib", "references/resources.bib"}))
def test_changed_current_source_lock_license_or_reference_bytes_reject(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged bytes\n")
    with pytest.raises(ValueError, match="current source digest"):
        verify(package)


@pytest.mark.parametrize("name", [*sorted(gate.QASM_FILES), "results.json", "counts.svg", "artifact-inventory.csv"])
def test_changed_attested_artifact_bytes_reject(package, name):
    path = package / gate.BUNDLE_RELATIVE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact digest"):
        verify(package)


@pytest.mark.parametrize("name", ["clifford-t-trial.qasm", "progress.json", "source-preflight.json"])
def test_missing_managed_bundle_file_rejects(package, name):
    (package / gate.BUNDLE_RELATIVE / name).unlink()
    with pytest.raises(ValueError, match="29-file bundle"):
        verify(package)


def test_unrecorded_bundle_file_rejects(package):
    (package / gate.BUNDLE_RELATIVE / "unreviewed.log").write_text("extra")
    with pytest.raises(ValueError, match="29-file bundle"):
        verify(package)


@pytest.mark.parametrize("n", ["2", "3"])
def test_current_preserved_prior_manifest_changes_reject(package, n):
    path = package / gate.PRIOR_MANIFESTS[n][0]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="current preserved n" + n):
        verify(package)


def test_changed_current_frozen_baseline_rejects(package):
    path = package / "assets/generated/resources/grover-surface-code.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="current frozen baseline"):
        verify(package)


def test_consistent_unreviewed_manifest_metadata_still_rejects(package):
    path = package / gate.BUNDLE_RELATIVE / "manifest.json"
    record = gate.load_json(path)
    record["device"] = "unreviewed execution statement"
    path.write_text(json.dumps(record, indent=2) + "\n")
    with pytest.raises(ValueError, match="immutable reviewed manifest"):
        verify(package)


@pytest.mark.parametrize("index,keys,value", [
    (0, ["full_clifford_t_trial_completed"], False),
    (0, ["source_preflight_completed"], False),
    (0, ["runtime_source_validation", "actual_module_paths_verified"], False),
    (0, ["runtime_source_validation", "import_mode"], "standalone"),
    (0, ["runtime_source_validation", "bytecode_writes_disabled"], False),
    (0, ["private_cache_distributed"], True),
    (0, ["whole_family_physical_completed"], True),
    (0, ["run_caps", "density_wires"], 6),
    (0, ["run_caps", "synthesized_instructions"], 1000000),
    (1, ["prepared", "case_id"], "lindblad-n3-r134-Z0"),
    (1, ["prepared", "selected_steps"], 210),
    (1, ["prepared", "task", "n_system"], 3),
    (1, ["prepared", "output_clbits"], [0]),
    (1, ["prepared", "score_values"], [1., -1.]),
    (1, ["prepared", "checker_limits", "max_qubits"], 6),
    (1, ["prepared", "materialization_limits", "synthesized_operations"], 1000000),
    (1, ["prepared", "actual_synthesis", "derivation", "shared_IR_limits_modified"], True),
    (1, ["prepared", "actual_synthesis", "derivation", "frozen_ir_synthesis_function_sha256"], "0" * 64),
    (1, ["prepared", "actual_synthesis", "actual_clifford_t_synthesis_performed"], False),
    (1, ["prepared", "actual_synthesis", "physical_estimation_performed"], True),
    (1, ["prepared", "actual_synthesis", "rotation_occurrences"], 5907),
    (1, ["prepared", "actual_synthesis", "clifford_t_final_measurement_wires"], list(range(8))),
    (1, ["prepared", "actual_synthesis", "compiled_global_phase"], 0.),
    (1, ["prepared", "preparation_fingerprint"], "0" * 64),
    (1, ["prepared", "complete_workload", "complete_trials"], 1),
    (1, ["prepared", "complete_workload", "whole_workload_executed"], True),
    (1, ["prepared", "complete_workload", "source_reuse_resets_per_trial"], 1481),
    (1, ["prepared", "complete_workload", "physical_failure_exact", "denominator"], 1000),
    (1, ["prepared", "hardware_execution"], True),
    (1, ["prepared", "error_contract", "formal_synthesis_certificate"], True),
    (1, ["prepared", "legacy_materialization_validation", "legacy_synthesis_performed"], True),
    (1, ["prepared", "legacy_materialization_validation", "actual_operations"], 1000000),
    (1, ["prepared", "clifford_t_instrument_check", "normalization_is_postselection"], True),
    (1, ["prepared", "clifford_t_instrument_check", "outcome_wires"], [0, 1, 2, 3]),
    (1, ["prepared", "clifford_t_instrument_check", "nominal_state_trace_guard"], 1e-7),
    (1, ["prepared", "representation_check", "representation_reserve_is_certified"], True),
    (1, ["prepared", "clifford_t_instrument_check", "simulation", "trajectory_branch_pruning"], True),
    (1, ["family", "2", "prior_manifest_sha256"], "0" * 64),
    (1, ["family", "3", "regenerated"], True),
    (1, ["family", "4", "legacy_physical_estimation_performed"], True),
    (1, ["family", "4", "density_width_rejection_reproduced"], False),
    (1, ["physical_profiles", 0, "profile", "controller_failure_per_decision"], 1e-18),
    (1, ["physical_profiles", 0, "status"], "computed_conditional_instrument_surface_code_model"),
    (1, ["physical_profiles", 0, "pareto_points"], [{"peak_physical_qubits": 0}]),
    (1, ["physical_profiles", 0, "native_estimation_performed"], True),
    (1, ["physical_profiles", 0, "execution_query_counts", "top_level_factory_queries"], 1),
    (1, ["physical_profiles", 0, "controller_model_diagnostic", "controller_decisions_per_trial"], 52889904),
    (1, ["physical_profiles", 0, "controller_model_diagnostic", "factory_retry_cap"], 6),
    (1, ["physical_profiles", 0, "matched_task_implementation_proposal", "proposed_implementation_executed"], True),
    (2, ["phase"], "uncomputed_stage_error"),
    (2, ["wall_seconds"], -1.),
    (2, ["wall_seconds"], 2401.),
    (2, ["wall_seconds"], float("inf")),
])
def test_changed_saved_identity_status_scope_and_ledger_reject_before_hash_checks(records, index, keys, value):
    set_value(records[index], keys, value)
    with pytest.raises(ValueError, match="n4"):
        gate.verify_records(*records)


@pytest.mark.parametrize("key", ["peak_physical_qubits", "worst_case_runtime_all_attempts_ns", "data_code_distance",
                                 "full_physical_failure_model", "feasible_candidate_count"])
def test_unassessed_physical_values_cannot_be_fabricated_as_zero(records, key):
    records[1]["physical_profiles"][0][key] = 0
    with pytest.raises(ValueError, match="unassessed costs"):
        gate.verify_records(*records)


def test_consistently_changed_exact_family_budgets_reject_semantically(records):
    for allocation in (records[1]["family_allocations"], records[1]["prepared"]["family_allocations"],
                       records[1]["source_preflight"]["family_allocations"], records[3]["family_allocations"]):
        allocation["physical_family_failure_exact"]["denominator"] = 300
    with pytest.raises(ValueError, match="exact family allowance"):
        gate.verify_records(*records)


@pytest.mark.parametrize("key", ["source_identity", "source_schema", "source_record_sha256", "invocation_id"])
def test_consistently_changed_source_preflight_bindings_reject(records, key):
    for record in records[:3]:
        record["execution"]["stage_status"]["source_preflight"][key] = "unbound"
    with pytest.raises(ValueError, match="source preflight"):
        gate.verify_records(*records)


@pytest.mark.parametrize("index,key", [(0, "execution"), (1, "prepared"), (2, "execution")])
def test_missing_completion_record_fails_as_value_error(records, index, key):
    del records[index][key]
    with pytest.raises(ValueError, match="malformed"):
        gate.verify_records(*records)


@pytest.mark.parametrize("change", ["missing", "duplicate", "size", "hash", "category", "csv", "exclusions"])
def test_changed_payload_inventory_identity_rejects(package, change):
    bundle = package / gate.BUNDLE_RELATIVE
    manifest = gate.load_json(bundle / "manifest.json")
    inventory = gate.load_json(bundle / "artifact-inventory.json")
    if change == "missing": inventory["files"].pop()
    elif change == "duplicate": inventory["files"].append(copy.deepcopy(inventory["files"][0]))
    elif change == "size": inventory["files"][0]["bytes"] += 1
    elif change == "hash": inventory["files"][0]["sha256"] = "0" * 64
    elif change == "category": inventory["categories"] = {}
    elif change == "exclusions": inventory["excluded_metadata"] = []
    else:
        path = bundle / "artifact-inventory.csv"
        path.write_text(path.read_text().replace("LICENSE.txt", "unrecorded.txt", 1))
    with pytest.raises(ValueError, match="n4"):
        gate.verify_inventory(bundle, manifest, inventory)


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"unused":NaN}', '{"unused":1e400}',
                                  '[]', '{', '{"nested":' + '[' * 3000 + '0' + ']' * 3000 + '}'])
def test_duplicate_nonfinite_and_nonobject_json_rejects(tmp_path, text):
    path = tmp_path / "record.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="n4"):
        gate.load_json(path)


def test_saved_json_parsing_is_bounded(tmp_path):
    path = tmp_path / "record.json"
    path.write_bytes(b" " * (2 * 1024**2 + 1))
    with pytest.raises(ValueError, match="parsing cap"):
        gate.load_json(path)


def test_linked_bundle_file_rejects(package):
    path = package / gate.BUNDLE_RELATIVE / "counts.svg"
    target = package / "outside.svg"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="linked"):
        verify(package)


def test_linked_bundle_root_rejects(package):
    alias = package / "alias"
    alias.symlink_to(package / gate.BUNDLE_RELATIVE)
    with pytest.raises(ValueError, match="regular directory"):
        gate.verify_bundle(alias, source_root=package, canonical_root=package)
