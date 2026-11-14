"""Release gates reject stale n3 sources, artifacts, status and inventories."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("n3_release_gate", ROOT / "scripts/verify_simulation_lindblad_three.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


@pytest.fixture
def records():
    bundle = ROOT / gate.BUNDLE_RELATIVE
    return [gate.load_json(bundle / name) for name in ("manifest.json", "results.json", "progress.json")]


@pytest.fixture
def package(tmp_path):
    root = (tmp_path / "package").resolve()
    shutil.copytree(ROOT / gate.BUNDLE_RELATIVE, root / gate.BUNDLE_RELATIVE)
    paths = {relative for relative in gate.SOURCE_FILES.values() if relative is not None}
    paths |= {"references/simulation.bib", "references/resources.bib",
              "assets/generated/simulation/lindblad-physical/manifest.json"}
    manifest = gate.load_json(ROOT / gate.BUNDLE_RELATIVE / "manifest.json")
    paths |= set(manifest["frozen_before"])
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    return root


def verify(root):
    return gate.verify_bundle(root / gate.BUNDLE_RELATIVE, source_root=root, canonical_root=root)


def set_value(record, keys, value):
    current = record
    for key in keys[:-1]:
        current = current[key]
    current[keys[-1]] = value


def test_original_reviewed_bundle_verifies_without_native_calls(package):
    assert verify(package) == {"files": 26, "bytes": 7725245,
                               "manifest_sha256": gate.REVIEWED_MANIFEST_SHA256, "native_calls": 0}


@pytest.mark.parametrize("relative", sorted({value for value in gate.SOURCE_FILES.values() if value is not None}
                                           | {"references/simulation.bib", "references/resources.bib"}))
def test_current_source_lock_license_and_bibliography_changes_reject(package, relative):
    path = package / relative
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError, match="current source snapshot"):
        verify(package)


@pytest.mark.parametrize("name", ["source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
                                  "counts.svg", "results.json", "artifact-inventory.csv"])
def test_changed_current_artifacts_reject(package, name):
    path = package / gate.BUNDLE_RELATIVE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="artifact digest"):
        verify(package)


@pytest.mark.parametrize("name", ["source-trial.qasm", "progress.json", "requirements-lock.txt"])
def test_missing_bundle_files_reject(package, name):
    (package / gate.BUNDLE_RELATIVE / name).unlink()
    with pytest.raises(ValueError, match="26-file bundle"):
        verify(package)


def test_unrecorded_bundle_file_rejects(package):
    (package / gate.BUNDLE_RELATIVE / "private.log").write_text("unreviewed")
    with pytest.raises(ValueError, match="26-file bundle"):
        verify(package)


def test_changed_frozen_baseline_rejects(package):
    path = package / "assets/generated/resources/grover-surface-code.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="frozen baseline"):
        verify(package)


def test_changed_prior_n2_manifest_rejects(package):
    path = package / "assets/generated/simulation/lindblad-physical/manifest.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="preserved n2 manifest"):
        verify(package)


def test_consistently_rehashed_unreviewed_manifest_rejects(package):
    bundle = package / gate.BUNDLE_RELATIVE
    path = bundle / "manifest.json"
    manifest = gate.load_json(path)
    manifest["device"] = "unreviewed runtime claim"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="immutable reviewed manifest"):
        verify(package)


@pytest.mark.parametrize("index,keys,value", [
    (0, ["python"], "3.12.15"),
    (0, ["private_cache_distributed"], True),
    (0, ["whole_family_physical_completed"], True),
    (0, ["frozen_after"], {}),
    (0, ["frozen_before"], []),
    (1, ["prepared", "case_id"], "lindblad-n4-r211-Z0"),
    (1, ["prepared", "task", "n_system"], 4),
    (1, ["prepared", "task", "time"], 2.),
    (1, ["prepared", "selected_steps"], 133),
    (1, ["prepared", "output_clbits"], [0]),
    (1, ["prepared", "score_values"], [1., -1.]),
    (1, ["prepared", "limits", "max_qubits"], 8),
    (1, ["prepared", "qasm_sha256"], []),
    (1, ["prepared", "complete_workload", "complete_trials"], 1),
    (1, ["prepared", "complete_workload", "whole_workload_executed"], True),
    (1, ["prepared", "error_contract", "formal_synthesis_certificate"], True),
    (1, ["prepared", "hardware_execution"], True),
    (1, ["execution", "stage_status", "prepare", "status"], "planned"),
    (2, ["phase"], "failed_bounded_generation"),
    (2, ["wall_seconds"], -1.),
    (2, ["wall_seconds"], 1801.),
    (2, ["wall_seconds"], float("inf")),
    (2, ["execution", "repeated_trials_executed"], True),
    (1, ["family", "2", "prior_manifest_sha256"], "0" * 64),
    (1, ["family", "2", "regenerated"], True),
    (1, ["family", "4", "status"], "implemented"),
    (1, ["family", "4", "density_required_wires"], 6),
    (1, ["family", "4", "density_width_rejection_reproduced"], False),
    (1, ["family", "4", "synthesis_performed"], True),
    (1, ["family", "4", "physical_estimation_performed"], True),
    (1, ["family", "4", "family_allocation_reallocated"], True),
    (1, ["physical_profiles", 0, "profile", "controller_failure_per_decision"], 1e-18),
    (1, ["physical_profiles", 0, "status"], "planned"),
    (1, ["physical_profiles", 0, "all_feasible_points", 0, "peak_physical_qubits"], 0),
    (1, ["physical_profiles", 0, "complete_algorithm_attempts"], 1),
])
def test_semantic_identity_and_status_changes_reject_even_before_hash_checks(records, index, keys, value):
    set_value(records[index], keys, value)
    with pytest.raises(ValueError, match="n3"):
        gate.verify_records(*records)


def test_completed_infeasible_profile_requires_empty_frontier_and_no_zero_cost(records):
    profile = records[1]["physical_profiles"][0]
    profile.update(status="infeasible_declared_controller_model", all_feasible_points=[],
                   pareto_points=[], feasible_candidate_count=0)
    gate.verify_records(*records)
    profile["pareto_points"] = [{"peak_physical_qubits": 0, "worst_case_runtime_all_attempts_ns": 0}]
    with pytest.raises(ValueError, match="infeasible frontier"):
        gate.verify_records(*records)


@pytest.mark.parametrize("record_index,key", [(0, "execution"), (1, "prepared"), (2, "execution")])
def test_missing_status_records_fail_as_value_errors(records, record_index, key):
    del records[record_index][key]
    with pytest.raises(ValueError, match="malformed"):
        gate.verify_records(*records)


@pytest.mark.parametrize("key,value", [("payload_files", 21), ("payload_bytes", 0),
                                     ("excluded_metadata", []), ("categories", {}),
                                     ("originals_preserved", False), ("publication_layout_selected", True)])
def test_inventory_metadata_changes_reject(package, key, value):
    bundle = package / gate.BUNDLE_RELATIVE
    manifest = gate.load_json(bundle / "manifest.json")
    inventory = gate.load_json(bundle / "artifact-inventory.json")
    inventory[key] = value
    with pytest.raises(ValueError, match="n3"):
        gate.verify_inventory(bundle, manifest, inventory)


@pytest.mark.parametrize("change", ["duplicate", "missing", "digest", "size", "csv"])
def test_inventory_payload_identity_changes_reject(package, change):
    bundle = package / gate.BUNDLE_RELATIVE
    inventory = gate.load_json(bundle / "artifact-inventory.json")
    if change == "duplicate":
        inventory["files"].append(copy.deepcopy(inventory["files"][0]))
    elif change == "missing":
        inventory["files"].pop()
    elif change == "digest":
        inventory["files"][0]["sha256"] = "0" * 64
    elif change == "size":
        inventory["files"][0]["bytes"] += 1
    else:
        path = bundle / "artifact-inventory.csv"
        path.write_text(path.read_text().replace("LICENSE.txt", "unrecorded.txt", 1))
    with pytest.raises(ValueError, match="n3"):
        gate.verify_inventory(bundle, gate.load_json(bundle / "manifest.json"), inventory)


@pytest.mark.parametrize("text", ['{"schema_version":1,"schema_version":2}', '{"value":NaN}',
                                  '{"value":1e400}', '[]'])
def test_malformed_json_rejects(tmp_path, text):
    path = tmp_path / "record.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="n3"):
        gate.load_json(path)


def test_linked_bundle_file_rejects(package):
    path = package / gate.BUNDLE_RELATIVE / "counts.svg"
    target = package / "outside.svg"
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="special"):
        verify(package)


def test_linked_bundle_root_rejects(package):
    alias = package / "alias"
    alias.symlink_to(package / gate.BUNDLE_RELATIVE)
    with pytest.raises(ValueError, match="regular directory"):
        gate.verify_bundle(alias, source_root=package, canonical_root=package)


def test_n3_notebook_keeps_all_five_registered_embed_labels():
    notebook = json.loads((ROOT / "notebooks/simulation_lindblad_three.ipynb").read_text())
    labels = {cell.get("metadata", {}).get("label") for cell in notebook["cells"]}
    assert {"simulation-lindblad-three-" + name for name in
            ("source", "correlations", "workload", "resources", "budget")} <= labels
