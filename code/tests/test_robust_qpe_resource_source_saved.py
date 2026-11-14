# SPDX-License-Identifier: MIT
"""Saved source/export boundary checks; no RPE simulation or physical backend."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_robust_qpe_resource_source_saved.py"
BUNDLE = ROOT / "assets/generated/robust-qpe-resource-source"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_pin(bundle: Path) -> str:
    return digest(bundle / "projection-manifest.json")


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def bind_archive_descriptor(bundle: Path) -> None:
    """Rebind only the external descriptor to the current test-only archive."""
    descriptor_path = bundle / "source-export-descriptor.json"
    descriptor = json.loads(descriptor_path.read_text())
    manifest = bundle / "projection-manifest.json"
    archive = bundle / "source-export-download.zip"
    descriptor["manifest"] = {"path": manifest.name, "bytes": manifest.stat().st_size,
                              "sha256": digest(manifest)}
    descriptor["archive"] = {"path": archive.name, "bytes": archive.stat().st_size,
                             "sha256": digest(archive)}
    write_json(descriptor_path, descriptor)


def mutate_record_and_rebind_all_layers(bundle: Path, update) -> str:
    """Keep file, manifest, descriptor and ZIP hashes coherent for claim tests."""
    record_path = bundle / "source-export-record.json"
    record = json.loads(record_path.read_text())
    update(record)
    write_json(record_path, record)
    manifest_path = bundle / "projection-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][record_path.name] = {"bytes": record_path.stat().st_size,
                                         "sha256": digest(record_path)}
    write_json(manifest_path, manifest)
    archive_path = bundle / "source-export-download.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for name in sorted(set(manifest["files"]) | {manifest_path.name}):
            handle.writestr(name, (bundle / name).read_bytes())
    bind_archive_descriptor(bundle)
    return manifest_pin(bundle)


@pytest.fixture
def saved():
    spec = importlib.util.spec_from_file_location("_robust_qpe_source_saved_tests", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def copied(tmp_path):
    destination = tmp_path.resolve() / "source-export"
    shutil.copytree(BUNDLE, destination)
    return destination


@pytest.mark.parametrize("check_sources", (False, True))
def test_pristine_saved_source_export_preserves_scope_without_scientific_imports(saved, monkeypatch, check_sources):
    original = builtins.__import__
    forbidden = {"openquantum", "qiskit", "numpy", "scipy", "qdk", "matplotlib",
                 "nbformat", "nbclient", "subprocess"}
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in forbidden:
            pytest.fail("Saved source reader entered a scientific or execution backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    report = saved.verify_saved_source_export(BUNDLE, manifest_pin(BUNDLE),
                                              source_root=ROOT if check_sources else None)
    assert report["source_only_invocation_succeeded"] is True
    assert report["prior_source_only_invocation_succeeded"] is False
    assert report["algorithm_numerically_validated"] is False
    assert report["physical_resource_estimate_computed"] is False
    assert report["actual_macrocycle_operations"] == 5286
    assert report["qasm_counts"] == {"p": 3024, "cx": 2016, "h": 24, "x": 96,
                                     "sdg": 6, "reset": 108, "measure": 12}
    assert report["bundle_members_checked"] == 38 and report["zip_members_checked"] == 36
    assert report["source_files_checked"] == (12 if check_sources else 0)
    assert report["producer_or_test_invocations"] == report["quantum_package_imports"] == 0


@pytest.mark.parametrize("bad_pin", ("0" * 64, None))
def test_expected_manifest_pin_is_required_and_checked_before_acceptance(saved, copied, bad_pin):
    with pytest.raises((ValueError, TypeError, OSError)):
        saved.verify_saved_source_export(copied, bad_pin)


def test_changed_scientific_source_member_is_rejected(saved, copied):
    pin = manifest_pin(copied)
    path = copied / "source/source-macrocycle.qasm"
    path.write_bytes(path.read_bytes() + b"// unreviewed mutation\n")
    with pytest.raises((ValueError, OSError)):
        saved.verify_saved_source_export(copied, pin)


def test_unlisted_member_is_rejected(saved, copied):
    pin = manifest_pin(copied)
    (copied / "unreviewed-notes.txt").write_text("not part of the source/export projection\n")
    with pytest.raises((ValueError, OSError)):
        saved.verify_saved_source_export(copied, pin)


@pytest.mark.parametrize("claim", (True, 0))
def test_numerical_claim_requires_exact_false_after_coherent_metadata_rebinding(saved, copied, claim):
    pin = mutate_record_and_rebind_all_layers(copied, lambda record: record.update(numerical_validation=claim))
    with pytest.raises(ValueError, match="Forbidden validation claim: numerical_validation"):
        saved.verify_saved_source_export(copied, pin)


@pytest.mark.parametrize("run_index", (0, 1))
def test_initial_failure_and_repaired_success_cannot_be_reversed_even_with_valid_hashes(saved, copied, run_index):
    def update(record):
        run = record["runs"][run_index]
        run["whole_invocation_completed"] = not run["whole_invocation_completed"]
    pin = mutate_record_and_rebind_all_layers(copied, update)
    with pytest.raises(ValueError, match="Failed/succeeded source invocation distinction changed"):
        saved.verify_saved_source_export(copied, pin)


@pytest.mark.parametrize("where", ("bundle", "member"))
def test_source_projection_refuses_symlink_redirection(saved, copied, tmp_path, where):
    pin = manifest_pin(copied)
    if where == "bundle":
        alias = tmp_path.resolve() / "bundle-alias"
        alias.symlink_to(copied, target_is_directory=True)
        copied = alias
    else:
        member = copied / "source/source-macrocycle.qasm"
        retained = tmp_path.resolve() / "retained-source.qasm"
        member.rename(retained)
        member.symlink_to(retained)
    with pytest.raises((ValueError, OSError)):
        saved.verify_saved_source_export(copied, pin)


@pytest.mark.parametrize("change", ("extra_member", "changed_bytes"))
def test_archive_content_or_outer_hash_mutation_is_rejected(saved, copied, change):
    pin = manifest_pin(copied)
    archive = copied / "source-export-download.zip"
    if change == "extra_member":
        with zipfile.ZipFile(archive, "a") as handle:
            handle.writestr("unreviewed-private-note.txt", "not an admitted public artifact")
        bind_archive_descriptor(copied)
        expected = "Archive membership is not closed"
    else:
        data = bytearray(archive.read_bytes())
        data[-1] ^= 1
        archive.write_bytes(data)
        expected = "Archive descriptor differs"
    with pytest.raises(ValueError, match=expected):
        saved.verify_saved_source_export(copied, pin)
