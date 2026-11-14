"""Bayesian saved-data publication and refresh checks, without scientific runs."""
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
    spec = importlib.util.spec_from_file_location("bayesian_registration_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved = load_script("verify_bayesian_qpe_saved")
export = load_script("export_code")
package = load_script("verify_public_package")


@pytest.fixture
def registered(tmp_path):
    root = tmp_path / "registered"
    names = set(saved.SOURCE_HASHES) | {saved.NOTEBOOK, saved.VALIDATION, "scripts/verify_bayesian_qpe_saved.py"}
    names |= {directory + "/" + name for directory, members in saved.BUNDLE_FILES.items() for name in members}
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
    return root.resolve(), provenance


def test_saved_payload_passes_through_authenticated_helper_without_new_draws(registered):
    root, provenance = registered
    gate = package._load_bayesian_saved_gate(root, provenance)
    result = gate.verify_saved(root, provenance=provenance)
    assert result["status"] == "saved_component_verified"
    assert result["bundles"] == {"assets/generated/bayesian-qpe": 76,
                                "assets/generated/bayesian-qpe-validation/source32": 9,
                                "assets/generated/bayesian-qpe-validation/replay": 30}
    assert result["saved_validation_modeled_draws"] == 456
    assert result["new_modeled_draws"] == 0 and result["new_quantum_execution"] is False


@pytest.mark.parametrize("change", ["unrecorded_member", "unrecorded_parent", "changed_payload"])
def test_saved_payload_rejects_extra_private_member_or_changed_producer_bytes(registered, change):
    root, _ = registered
    directory = root / "assets/generated/bayesian-qpe"
    if change in ("unrecorded_member", "unrecorded_parent"):
        if change == "unrecorded_parent":
            directory = root / "assets/generated/bayesian-qpe-validation"
        (directory / "controller.json").write_text('{"private":true}')
        error = "unrecorded members"
    else:
        with (directory / "circuit.svg").open("ab") as stream:
            stream.write(b"\n")
        error = "payload identity changed"
    with pytest.raises(ValueError, match=error):
        saved.verify_saved(root)


def test_output_refresh_and_source_string_normalization_keep_current_provenance(registered):
    root, provenance = registered
    path = root / saved.NOTEBOOK
    historical_bytes = path.read_bytes()
    notebook = json.loads(historical_bytes)
    for cell in notebook["cells"]:
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cell["source"] = source.splitlines(keepends=True)
    notebook["metadata"]["registration_refresh_fixture"] = True
    notebook["cells"][1]["outputs"].append({"output_type": "stream", "name": "stdout", "text": "Refreshed saved output\n"})
    path.write_text(json.dumps(notebook))
    assert path.read_bytes() != historical_bytes
    with pytest.raises(ValueError, match="current package provenance"):
        saved.verify_saved(root, provenance=provenance)
    provenance["file_sha256"][saved.NOTEBOOK] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = saved.verify_saved(root, provenance=provenance)
    assert result["notebook_source_sha256"] == saved.NOTEBOOK_SOURCE_SHA256


def test_notebook_error_output_is_rejected_after_an_output_refresh(registered):
    root, _ = registered
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    notebook["cells"][1]["outputs"].append({"output_type": "error", "ename": "RuntimeError", "evalue": "fixture", "traceback": []})
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="error output"):
        saved.verify_saved(root)


def test_navigation_downloads_and_closed_export_cover_the_saved_manifest():
    exact = {directory + "/" + name for directory, members in saved.BUNDLE_FILES.items() for name in members} | {saved.VALIDATION}
    assert export.PUBLIC_BAYESIAN_FILES == exact and len(exact) == 116
    assert all(export.include_generated_file(name) for name in exact)
    for name in ("assets/generated/bayesian-qpe/private.json",
                 "assets/generated/bayesian-qpe-validation/replay/controller.json",
                 "assets/generated/bayesian-qpe-validation/source32/raw-log.md",
                 "assets/generated/bayesian-qpe-validation/.cache/trial.json",
                 "docs/coordination/bayesian-qpe-handoff.md"):
        assert not export.include_generated_file(name)
    dependencies = {directory + "/" + name for directory in saved.BUNDLE_FILES for name in ("requirements-lock.txt", "LICENSE.txt")}
    assert dependencies <= export.PUBLIC_DEPENDENCY_FILES
    assert {"generate_bayesian_qpe.py", "verify_bayesian_qpe_saved.py"} <= set(export.PUBLIC_SCRIPTS)
    myst_path = ROOT / "myst.yml"
    if myst_path.is_file():
        toc = yaml.safe_load(myst_path.read_text())["project"]["toc"]
        qpe = next(entry for entry in toc if entry.get("file") == "algorithms/qft-qpe.md")
        bayesian = next(entry for entry in qpe["children"] if entry.get("file") == "algorithms/qft-qpe/bayesian.md")
        assert bayesian["children"] == [{"file": saved.NOTEBOOK, "title": "Circular-grid Bayesian notebook"}]
        site = load_script("check_site")
        required = set(site.required_artifacts())
        assert exact <= required
        assert {"algorithms/qft-qpe/bayesian/index.html", "notebooks/bayesian-qpe/index.html", saved.NOTEBOOK,
                "code/src/openquantum/bayesian_qpe.py", "code/scripts/generate_bayesian_qpe.py",
                "code/scripts/verify_bayesian_qpe_saved.py"} <= required
    normalizer = ast.parse((ROOT / "scripts/normalize_figures.py").read_text())
    immutable = set()
    for node in normalizer.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("IMMUTABLE_SVGS", "IMMUTABLE_CSVS") for target in node.targets):
            immutable |= ast.literal_eval(node.value.args[0])
    assert {name for name in exact if name.endswith((".svg", ".csv"))} <= immutable
