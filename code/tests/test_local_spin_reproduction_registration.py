"""Saved public-interface outcome checks; no finite-reference rerun."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("reproduction_registration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = script("verify_public_package")
BUNDLE = package.LOCAL_SPIN_REPRODUCTION_DIRECTORY


def selected_script(name, **values):
    path = ROOT / "scripts" / (name + ".py")
    parsed = ast.parse(path.read_bytes())
    nodes = [node for node in parsed.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("reproduction_selected_" + name)
    module.__file__ = str(path)
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, **values)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def registered(tmp_path):
    root = (tmp_path / "registered").resolve()
    old_bundle = "assets/generated/simulation/reusable-bath-local-spin"
    origin = json.loads((ROOT / BUNDLE / "provenance.json").read_bytes())
    names = {BUNDLE + "/" + name for name in package.LOCAL_SPIN_REPRODUCTION_FILES}
    names |= {old_bundle + "/" + name for name in origin["files"]}
    names |= set(package.LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES)
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {name: sha((root / name).read_bytes()) for name in names}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return root, provenance


def test_saved_outcome_verifies_without_importing_its_scientific_interface(registered, monkeypatch):
    root, provenance = registered
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"openquantum", "qiskit", "numpy", "qdk", "psutil"}:
            pytest.fail("Saved reproduction verification imported an execution backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = package._verify_local_spin_reproduction_identity(root, provenance=provenance)
    assert result == {"status": "saved_local_spin_reproduction_verified", "files": 10,
                      "original_inputs": 21, "new_reference_execution": False,
                      "native_calls": 0, "all_components_verified": False}


@pytest.mark.parametrize("name", sorted(package.LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES))
def test_rewritten_provenance_cannot_execute_changed_source(registered, tmp_path, name):
    root, provenance = registered
    marker = tmp_path / "unbound-interface-ran"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (root / name).write_bytes(raw)
    provenance["file_sha256"][name] = sha(raw)
    with pytest.raises(ValueError, match="source changed"):
        package._verify_local_spin_reproduction_identity(root, provenance=provenance)
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["manifest", "sidecar", "report", "old-reference", "extra-file", "extra-source", "empty-directory"])
def test_saved_outcome_and_old_reference_cannot_be_rebound(registered, mutation):
    root, provenance = registered
    names = {"manifest": BUNDLE + "/artifact-manifest.json", "sidecar": BUNDLE + "/artifact-manifest.sha256",
             "report": BUNDLE + "/finite-reference.json",
             "old-reference": "assets/generated/simulation/reusable-bath-local-spin/reference.json"}
    if mutation in names:
        path = root / names[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
        provenance["file_sha256"][names[mutation]] = sha(path.read_bytes())
    elif mutation == "extra-file":
        (root / BUNDLE / "controller.json").write_text('{"private":true}')
    elif mutation == "extra-source":
        (root / BUNDLE / "source/private.py").write_text("private")
    else:
        (root / BUNDLE / ".cache").mkdir()
    with pytest.raises(ValueError):
        package._verify_local_spin_reproduction_identity(root, provenance=provenance)


def test_linked_source_directory_rejects_before_saved_payload_access(registered, tmp_path):
    root, provenance = registered
    source = root / BUNDLE / "source"
    outside = tmp_path / "source-copy"
    source.rename(outside)
    source.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        package._verify_local_spin_reproduction_identity(root, provenance=provenance)


def test_site_uses_canonical_assets_and_authenticated_source_preview(registered, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    root, provenance = registered
    site, code = tmp_path / "site", tmp_path / "site/code"
    for name in set(package.LOCAL_SPIN_REPRODUCTION_SOURCE_HASHES) | {"PROVENANCE.json"}:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copytree(root / "assets", site / "assets")
    assert not (code / "assets").exists()
    checker = selected_script("check_site", _read_bounded_public_member=package._read_bounded_public_member,
                              _verify_local_spin_reproduction_identity=package._verify_local_spin_reproduction_identity)
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": sha((code / "PROVENANCE.json").read_bytes())}}
    assert checker.site_local_spin_reproduction_errors(site, identity) == []
    (code / "PROVENANCE.json").write_text("{}")
    assert checker.site_local_spin_reproduction_errors(site, identity)


def test_closed_subtree_does_not_widen_formats_or_remove_existing_figure():
    exporter = selected_script("export_code", LINDBLAD_PORT_FILES=frozenset())
    exact = {BUNDLE + "/" + name for name in package.LOCAL_SPIN_REPRODUCTION_FILES}
    assert exporter.PUBLIC_LOCAL_SPIN_REPRODUCTION_FILES == exact
    assert all(exporter.include_generated_file(name) for name in exact)
    for name in (BUNDLE + "/events-private.ndjson", BUNDLE + "/private.sha256", BUNDLE + "/source/private.py"):
        assert not exporter.include_generated_file(name)
    assert not {".ndjson", ".sha256"} & exporter.ARTIFACT_SUFFIXES


def test_local_spin_reproduction_article_and_required_site_assets():
    if not (ROOT / "scripts/check_site.py").is_file() or not (ROOT / "algorithms/simulation/reusable-bath-local-spin.md").is_file():
        pytest.skip("Book articles and site checks are deliberately outside the public example package")
    exact = {BUNDLE + "/" + name for name in package.LOCAL_SPIN_REPRODUCTION_FILES}
    article = (ROOT / "algorithms/simulation/reusable-bath-local-spin.md").read_text()
    assert ":name: simulation-local-spin-first-step" in article
    assert "### Reproduce the finite checks with ordinary Python" in article
    checker = script("check_site")
    assert exact <= set(checker.required_artifacts())
    assert {"code/src/openquantum/simulation/reusable_bath_local_spin_reproduce.py",
            "code/tests/test_reusable_bath_local_spin_reproduce.py"} <= set(checker.required_artifacts())
