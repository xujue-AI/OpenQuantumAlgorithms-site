"""Immutable port release checks use saved/mocked bytes, never scientific jobs."""
from __future__ import annotations

import ast
import builtins
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("port_release_gate", ROOT / "scripts/verify_simulation_lindblad_port.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture(scope="module")
def saved_records():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    records = [gate.load_json(bundle / name) for name in gate.RECORD_IDENTITIES]
    records.append(gate.load_json(ROOT / "assets/generated/simulation/lindblad-four/results.json"))
    return records


@pytest.fixture
def records(saved_records):
    return copy.deepcopy(saved_records)


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    shutil.copytree(ROOT / gate.BUNDLE_RELATIVE, root / gate.BUNDLE_RELATIVE)
    paths = {value for value in gate.SOURCE_FILES.values() if value is not None}
    paths |= {"references/simulation.bib", "references/resources.bib"} | set(gate.PRESERVED_SCIENTIFIC_FILES)
    for relative in paths:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


def verify(root):
    return gate.verify_bundle(root / gate.BUNDLE_RELATIVE, source_root=root, canonical_root=root)


def set_value(record, keys, value):
    for key in keys[:-1]:
        record = record[key]
    record[keys[-1]] = value


def test_reviewed_saved_release_is_complete_and_labels_native_trace_trust(package):
    receipt = verify(package)
    assert receipt["files"] == 33
    assert receipt["bytes"] == 26290618
    assert receipt["manifest_sha256"] == "7328a265464a9bbd25875c565d62ee4921f59d42af239deedec4951dbd17a83b"
    assert receipt["physical_status"] == "computed_conditional_port_surface_code_model"
    assert receipt["feasible_points"] == [45, 112]
    assert receipt["pareto_points"] == [3, 3]
    assert receipt["complete_trials"] == 15756
    assert receipt["preserved_scientific_files"] == 76
    assert receipt["original_controller_status"] == "infeasible_declared_controller_model"
    assert receipt["native_trace_digest_status"] == "audit-trusted; raw Trace JSON and inline audit script/hash were not saved"
    assert receipt["saved_native_profile_value_json_verified"] is True
    assert (receipt["native_calls"], receipt["synthesis_calls"], receipt["density_replays"]) == (0, 0, 0)
    assert receipt["repeated_trials_executed"] is False
    assert not (package / "docs").exists()


def test_independent_preview_source_root_needs_no_private_plans_or_prior_assets(package):
    code = package / "_code"
    for relative in {value for value in gate.SOURCE_FILES.values() if value is not None} | {
            "references/simulation.bib", "references/resources.bib"}:
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / relative, target)
    assert gate.verify_bundle(package / gate.BUNDLE_RELATIVE, source_root=code, canonical_root=package)["native_calls"] == 0
    assert not (code / "assets").exists()
    assert not (code / "docs").exists()


def test_stdlib_only_gate_cannot_import_implementation_or_execute_native(package, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"openquantum", "qiskit", "qdk", "numpy", "scipy", "psutil"}:
            raise AssertionError("immutable port gate imported a scientific implementation")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert verify(package)["density_replays"] == 0
    tree = ast.parse((ROOT / "scripts/verify_simulation_lindblad_port.py").read_text())
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert imports <= sys.stdlib_module_names
    assert not imports & {"subprocess", "pickle", "importlib"}
    assert "expected" not in inspect.signature(gate.verify_bundle).parameters
    assert not {"expected_hash", "expected_sha256", "regenerate", "canonical_regeneration"} & inspect.signature(gate.verify_bundle).parameters.keys()


@pytest.mark.parametrize("relative", sorted({value for value in gate.SOURCE_FILES.values() if value is not None}
                                           | {"references/simulation.bib", "references/resources.bib"}))
def test_changed_current_source_lock_license_references_runner_adapter_rejects(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged source\n")
    with pytest.raises(ValueError, match="current source digest"):
        verify(package)


@pytest.mark.parametrize("name", ["candidate-trial.qasm", "recipe.json", "source-proof.json", "word-catalog.json",
                                  "results.json", "artifact-inventory.json", "artifact-inventory.csv", "words/word-07.qasm"])
def test_changed_attested_payload_rejects(package, name):
    path = package / gate.BUNDLE_RELATIVE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact digest"):
        verify(package)


def test_changed_completed_progress_is_not_ignored(package):
    path = package / gate.BUNDLE_RELATIVE / "progress.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="progress digest"):
        verify(package)


def test_coordinated_unreviewed_hash_rewrite_has_no_manifest_bypass(package):
    bundle = package / gate.BUNDLE_RELATIVE
    results = gate.load_json(bundle / "results.json")
    results["status"] = "falsely_completed"
    (bundle / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    manifest = gate.load_json(bundle / "manifest.json")
    manifest["artifacts_sha256"]["results.json"] = gate.sha256(bundle / "results.json")
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="immutable reviewed manifest"):
        verify(package)


@pytest.mark.parametrize("name", ["candidate-trial.qasm", "recipe.json", "progress.json", "words/word-00.qasm"])
def test_missing_or_unrecorded_bundle_entries_reject(package, name):
    (package / gate.BUNDLE_RELATIVE / name).unlink()
    with pytest.raises(ValueError, match="33-file bundle"):
        verify(package)


def test_extra_file_and_extra_empty_directory_reject(package):
    bundle = package / gate.BUNDLE_RELATIVE
    (bundle / "unreviewed.log").write_text("new log")
    with pytest.raises(ValueError, match="33-file bundle"):
        verify(package)
    (bundle / "unreviewed.log").unlink()
    (bundle / "empty").mkdir()
    with pytest.raises(ValueError, match="unrecorded bundle directory"):
        verify(package)


@pytest.mark.parametrize("relative", ["assets/generated/simulation/lindblad-four/results.json",
                                      "assets/generated/simulation/lindblad-four/clifford-t-trial.qasm",
                                      "assets/generated/simulation/lindblad-three/manifest.json",
                                      "assets/generated/simulation/lindblad-physical/manifest.json"])
def test_preserved_scientific_bundle_digest_changes_reject(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="preserved scientific file"):
        verify(package)


@pytest.mark.parametrize("index,keys,value", [
    (0, ["status"], "source_only"),
    (0, ["source_only"], True),
    (0, ["hardware_execution"], True),
    (0, ["run_caps", "rss_bytes"], 3*1024**3),
    (0, ["source_sha256", "lindblad_port_words.py"], "0"*64),
    (1, ["status"], "computed_partial_profiles"),
    (1, ["candidate_qasm_sha256"], "0"*64),
    (1, ["candidate_verification_fingerprint"], "0"*64),
    (1, ["execution", "historical_densities_rerun"], True),
    (1, ["execution", "new_synthesis_calls"], 1),
    (1, ["execution", "checkpoint_restores_verified_capability"], True),
    (1, ["candidate", "candidate_full_density_executed"], False),
    (1, ["candidate", "candidate_density_execution_count"], True),
    (1, ["candidate", "historical_density_execution_count"], 1),
    (1, ["candidate", "physical_feasibility_claimed"], True),
    (1, ["candidate", "recipe_sha256"], "0"*64),
    (1, ["candidate", "catalog_sha256"], "0"*64),
    (1, ["candidate", "candidate_report", "added_cx"], 0),
    (1, ["candidate", "candidate_report", "t_wire_counts"], [0,642284,0,0,0,0,0,0]),
    (1, ["candidate", "candidate_report", "global_phase_binary64_hex"], "0x0.0p+0"),
    (1, ["candidate", "complete_workload", "complete_trials"], 1),
    (1, ["candidate", "complete_workload", "whole_workload_executed"], True),
    (1, ["candidate", "complete_workload", "deterministic_bias_multiplied_by_trials"], True),
    (1, ["candidate", "complete_workload", "family_allocations_reallocated"], True),
    (1, ["candidate", "complete_workload", "physical_failure_exact", "denominator"], 1000),
    (1, ["candidate", "complete_workload", "statistical_failure_exact", "denominator"], 500),
    (1, ["candidate", "complete_workload", "primitive_counts_all_trials", "cx"], 222635028),
    (1, ["candidate", "error_contract", "formal_synthesis_certificate"], True),
    (1, ["controller_eligibility", 0, "eligibility_is_physical_feasibility"], True),
    (1, ["controller_eligibility", 0, "factory_retry_cap"], 6),
    (1, ["controller_eligibility", 0, "controller_decisions_per_trial"], 123409809),
    (2, ["status"], "failed"),
    (2, ["peak_rss_bytes"], 2147483649),
    (2, ["peak_rss_bytes"], float("inf")),
    (3, ["candidate_materialized"], True),
    (3, ["candidate_full_density_executed"], True),
    (3, ["native_physical_queries"], 48),
    (4, ["recovery_sha256"], "0"*64),
    (4, ["formal_interval_certificate"], True),
    (4, ["words", 0, "tokens", 0, "name"], "x"),
    (4, ["words", 0, "global_phase_binary64_hex"], "0x0.0p+0"),
    (5, ["physical_feasibility_assessed"], True),
    (5, ["full_density_executed"], True),
    (5, ["new_synthesis_calls"], 1),
    (5, ["plan", "rotation_blocks", 0, "eligible"], False),
    (5, ["plan", "rotation_blocks", 0, "initialized_mask"], 0),
    (6, ["physical_profiles", 0, "status"], gate.STATUS),
    (6, ["physical_profiles", 0, "native_estimation_performed"], True),
    (6, ["physical_profiles", 0, "controller_model_diagnostic", "budget_exceeded"], False),
])
def test_saved_status_execution_scope_recipe_population_and_hash_mutations_reject(records, index, keys, value):
    set_value(records[index], keys, value)
    with pytest.raises(ValueError, match="port"):
        gate.verify_records(*records)


@pytest.mark.parametrize("profile_index", [0, 1])
@pytest.mark.parametrize("change", ["missing", "partial", "duplicate", "failed", "favorable", "no_pareto", "false_native",
                                    "overflow", "counter", "fingerprint", "value_hash", "budget", "wrong_model"])
def test_no_partial_feasible_profile_or_favorable_truncation_is_accepted(records, profile_index, change):
    profiles = records[1]["physical_profiles"]
    profile = profiles[profile_index]
    if change == "missing": profiles.pop(profile_index)
    elif change == "partial": profile["all_feasible_points"].pop()
    elif change == "duplicate": profile["all_feasible_points"][1] = copy.deepcopy(profile["all_feasible_points"][0])
    elif change == "failed": profile["status"] = "uncomputed_native_stage_error"
    elif change == "favorable": profile["all_feasible_points"] = copy.deepcopy(profile["pareto_points"])
    elif change == "no_pareto": profile["pareto_points"] = []
    elif change == "false_native": profile["execution_status"]["status"] = "uncomputed"
    elif change == "overflow": profile["native_execution"]["candidate_overflow_sentinel_observed"] = True
    elif change == "counter": profile["native_execution"]["trace_estimates"] = 1
    elif change == "fingerprint": profile["verification_fingerprint"] = "0"*64
    elif change == "value_hash": profile["execution_status"]["value_sha256"] = "0"*64
    elif change == "budget": profile["physical_failure_budget"] = .01
    else: profile["profile"]["controller_failure_per_decision"] = 1e-18
    with pytest.raises(ValueError, match="port"):
        gate.verify_records(*records)


@pytest.mark.parametrize("keys,value", [
    (["macro_ledger_per_complete_attempt", 0, "count"], 14130),
    (["macro_ledger_per_complete_attempt", 0, "time_ns"], 1),
    (["macro_ledger_per_complete_attempt", 6, "operation"], "t_q1"),
    (["macro_ledger_per_complete_attempt", 6, "readouts"], 0),
    (["macro_ledger_per_complete_attempt", 6, "preparations"], 0),
    (["macro_ledger_per_complete_attempt", 6, "decisions"], 1),
    (["macro_ledger_per_complete_attempt", 6, "event_failure_model"], 0.0),
    (["runtime_per_complete_attempt_ns"], 1),
    (["worst_case_runtime_all_attempts_ns"], 1),
    (["code_cycles_per_complete_attempt"], 1),
    (["data_code_distance"], True),
    (["reserved_logical_patches"], 8),
    (["factory_physical_qubits"], 0),
    (["factory_physical_qubits"], 985320.0),
    (["peak_physical_qubits"], 0),
    (["factory_supply", "pipeline_time_ns"], 962000.0),
    (["factory_supply", "fixed_logical_distillation_distance"], 39),
    (["factory_supply", "full_pipeline_retry_cap_per_state"], 6),
    (["factory_supply", "extra_output_port_hold_ns_per_state"], 0),
    (["factory_supply", "states_all_complete_attempts"], 642284),
    (["factory_supply", "availability_failure_union_bound"], 0),
    (["failure_components_all_attempts", "data_workspace_code_exposure"], 0.0),
    (["failure_components_all_attempts", "factory_output_handoff_storage"], 0.0),
    (["failure_components_all_attempts", "logical_readout_scenario"], 0.0),
    (["failure_components_all_attempts", "logical_initialization_scenario"], 0.0),
    (["failure_components_all_attempts", "controller_scenario"], 0.0),
    (["failure_components_all_attempts", "factory_output_quality_model"], 0.0),
    (["full_physical_failure_model"], 0.0),
    (["accounting_trace_sha256"], "not-a-digest"),
])
def test_each_independent_point_ledger_rejects_false_costs(saved_records, keys, value):
    row = copy.deepcopy(saved_records[1]["physical_profiles"][0]["all_feasible_points"][0])
    set_value(row, keys, value)
    with pytest.raises(ValueError, match="port"):
        gate.verify_point(row, .001)


def test_every_saved_point_passes_independent_integer_and_failure_arithmetic(saved_records):
    for profile in saved_records[1]["physical_profiles"]:
        for row in profile["all_feasible_points"]:
            gate.verify_point(row, profile["profile"]["physical_error_rate"])


@pytest.mark.parametrize("key", ["peak_physical_qubits", "runtime_per_complete_attempt_ns", "full_physical_failure_model",
                                 "data_code_distance", "feasible_candidate_count"])
def test_original_unassessed_costs_cannot_be_fabricated_as_zero(records, key):
    records[6]["physical_profiles"][0][key] = 0
    with pytest.raises(ValueError, match="unassessed costs"):
        gate.verify_baseline(records[6])


@pytest.mark.parametrize("index,key", [(0,"run_caps"), (1,"candidate"), (1,"physical_profiles"), (2,"status"), (5,"plan")])
def test_missing_required_records_fail_closed_as_value_error(records, index, key):
    del records[index][key]
    with pytest.raises(ValueError, match="port"):
        gate.verify_records(*records)


@pytest.mark.parametrize("change", ["missing", "duplicate", "size", "hash", "category", "csv", "extra", "scope"])
def test_inventories_require_every_payload_identity(package, change):
    bundle = package / gate.BUNDLE_RELATIVE
    manifest = gate.load_json(bundle / "manifest.json")
    inventory = gate.load_json(bundle / "artifact-inventory.json")
    if change == "missing": inventory["files"].pop()
    elif change == "duplicate": inventory["files"].append(copy.deepcopy(inventory["files"][0]))
    elif change == "size": inventory["files"][0]["bytes"] += 1
    elif change == "hash": inventory["files"][0]["sha256"] = "0"*64
    elif change == "category": inventory["files"][0]["category"] = "private_log"
    elif change == "extra": inventory["unused_success"] = True
    elif change == "scope": inventory["scope"] = "raw native Trace independently replayed"
    else:
        path = bundle / "artifact-inventory.csv"
        path.write_text(path.read_text().replace("LICENSE.txt", "unknown.txt", 1))
    with pytest.raises(ValueError, match="port inventory"):
        gate.verify_inventory(bundle, manifest, inventory)


@pytest.mark.parametrize("text", ['{"status":1,"status":2}', '{"unused":NaN}', '{"unused":Infinity}',
                                  '{"unused":1e400}', '[]', '{', '{"nested":'+'['*3000+'0'+']'*3000+'}'])
def test_duplicate_nonfinite_nonobject_and_deep_json_rejects(tmp_path, text):
    path = tmp_path / "record.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="port"):
        gate.load_json(path)


def test_json_and_saved_file_size_caps(tmp_path):
    path = tmp_path / "record.json"
    with path.open("wb") as stream: stream.truncate(8*1024**2+1)
    with pytest.raises(ValueError, match="parsing cap"):
        gate.load_json(path)
    with path.open("wb") as stream: stream.truncate(32*1024**2+1)
    with pytest.raises(ValueError, match="oversized"):
        gate.checked_file(tmp_path, path.name)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "words/../recipe.json", "words\\word-00.qasm", "./recipe.json"])
def test_unsafe_relative_paths_reject(tmp_path, name):
    with pytest.raises(ValueError, match="unsafe relative path"):
        gate.checked_file(tmp_path, name)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "root", "ancestor"])
def test_linked_saved_inputs_reject(package, kind):
    bundle = package / gate.BUNDLE_RELATIVE
    if kind in {"symlink", "hardlink"}:
        path = bundle / "candidate-trial.qasm"
        target = package / "outside.qasm"
        path.rename(target)
        if kind == "symlink": path.symlink_to(target)
        else: path.hardlink_to(target)
        call = lambda: verify(package)
    elif kind == "root":
        alias = package / "alias"
        alias.symlink_to(bundle)
        call = lambda: gate.verify_bundle(alias, source_root=package, canonical_root=package)
    else:
        alias = package.parent / "alias"
        alias.symlink_to(package)
        call = lambda: verify(alias)
    with pytest.raises(ValueError, match="linked|regular directory"):
        call()
