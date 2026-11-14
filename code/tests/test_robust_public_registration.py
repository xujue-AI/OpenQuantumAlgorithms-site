"""Robust QPE saved-file, privacy and output-refresh registration checks."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location("robust_registration_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved = load_script("verify_robust_qpe_saved")
export = load_script("export_code")
package = load_script("verify_public_package")


@pytest.fixture
def registered(tmp_path):
    root = tmp_path / "registered"
    names = set(saved.SOURCE_HASHES) | {saved.NOTEBOOK, "scripts/verify_robust_qpe_saved.py"}
    names |= {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
    return root.resolve(), provenance


def test_saved_evidence_passes_authenticated_helper_without_new_execution(registered):
    root, provenance = registered
    gate = package._load_robust_saved_gate(root, provenance)
    result = gate.verify_saved(root, provenance=provenance)
    assert result["status"] == "saved_component_verified" and result["files"] == 61
    assert result["saved_study_outputs"] == 8 and result["saved_modeled_fresh_trials"] == 43008
    assert result["saved_literal_base_calls"] == 451584
    assert result["new_modeled_draws"] == 0 and result["new_quantum_execution"] is False


@pytest.mark.parametrize("change", ["private_root_file", "nested_private_directory", "csv_normalization"])
def test_saved_bundle_rejects_private_extras_and_changed_producer_bytes(registered, change):
    root, _ = registered
    bundle = root / saved.BUNDLE
    if change == "private_root_file":
        (bundle / "controller.json").write_text('{"private":true}')
        error = "unrecorded members"
    elif change == "nested_private_directory":
        (bundle / "circuits/.cache").mkdir()
        error = "unrecorded members"
    else:
        path = bundle / "scaling.csv"
        raw = path.read_bytes()
        assert b"\r\n" in raw
        path.write_bytes(raw.replace(b"\r\n", b"\n"))
        error = "payload identity changed"
    with pytest.raises(ValueError, match=error):
        saved.verify_saved(root)


def test_refreshed_outputs_bind_current_provenance_and_preserve_normalized_sources(registered):
    root, provenance = registered
    path = root / saved.NOTEBOOK
    previous = path.read_bytes()
    notebook = json.loads(previous)
    for cell in notebook["cells"]:
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["registration_refresh_fixture"] = True
    notebook["cells"][1]["outputs"].append({"output_type": "stream", "name": "stdout", "text": "Refreshed saved output\n"})
    path.write_text(json.dumps(notebook))
    assert path.read_bytes() != previous
    with pytest.raises(ValueError, match="current package provenance"):
        saved.verify_saved(root, provenance=provenance)
    provenance["file_sha256"][saved.NOTEBOOK] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved.verify_saved(root, provenance=provenance)["notebook_source_sha256"] == saved.NOTEBOOK_SOURCE_SHA256


@pytest.mark.parametrize("change", ["label", "error"])
def test_notebook_metadata_labels_and_error_free_output_scope_are_required(registered, change):
    root, _ = registered
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    if change == "label":
        notebook["cells"][2]["metadata"]["label"] = "unregistered-circuit"
        error = "canonical notebook label"
    else:
        notebook["cells"][1]["outputs"].append({"output_type": "error", "ename": "RuntimeError", "evalue": "fixture", "traceback": []})
        error = "error output"
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match=error):
        saved.verify_saved(root)


def test_navigation_routes_closed_exports_and_byte_protection_cover_saved_payload():
    exact = {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    assert export.PUBLIC_ROBUST_FILES == exact and len(exact) == 61
    assert all(export.include_generated_file(name) for name in exact)
    assert not export.include_generated_file(saved.BUNDLE + "/controller.json")
    assert not export.include_generated_file(saved.BUNDLE + "/circuits/.cache/source.py")
    assert {"generate_robust_qpe.py", "verify_robust_qpe_saved.py"} <= set(export.PUBLIC_SCRIPTS)
    assert {saved.BUNDLE + "/requirements-lock.txt", saved.BUNDLE + "/LICENSE.txt"} <= export.PUBLIC_DEPENDENCY_FILES
    if (ROOT / "myst.yml").is_file():
        toc = yaml.safe_load((ROOT / "myst.yml").read_text())["project"]["toc"]
        qpe = next(entry for entry in toc if entry.get("file") == "algorithms/qft-qpe.md")
        robust = next(entry for entry in qpe["children"] if entry.get("file") == "algorithms/qft-qpe/robust.md")
        assert robust["children"] == [{"file": saved.NOTEBOOK, "title": "Bounded-bias robust QPE notebook"}]
        site = load_script("check_site")
        required = set(site.required_artifacts())
        assert exact <= required
        assert {"algorithms/qft-qpe/robust/index.html", "notebooks/robust-qpe/index.html", saved.NOTEBOOK,
                "code/src/openquantum/robust_qpe.py", "code/scripts/generate_robust_qpe.py",
                "code/scripts/verify_robust_qpe_saved.py"} <= required
        assert site.ROBUST_EMBED_LABELS == set(saved.NOTEBOOK_LABELS.values())
    normalizer = ast.parse((ROOT / "scripts/normalize_figures.py").read_text())
    immutable = set()
    for node in normalizer.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("IMMUTABLE_SVGS", "IMMUTABLE_CSVS") for target in node.targets):
            immutable |= ast.literal_eval(node.value.args[0])
    assert {name for name in exact if name.endswith((".svg", ".csv"))} <= immutable
