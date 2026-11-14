"""Frozen feedback-QPE snapshot and current notebook trust boundaries."""
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/verify_qpe_feedback_public.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("feedback_registration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved = script("verify_qpe_feedback_public")
package = script("verify_public_package")


def selected_script(name, **values):
    path = ROOT / "scripts" / (name + ".py")
    parsed = ast.parse(path.read_bytes())
    nodes = [node for node in parsed.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("feedback_selected_" + name)
    module.__file__ = str(path)
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, **values)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def registered(tmp_path):
    root = (tmp_path / "registered").resolve()
    names = set(saved.SOURCE_HASHES) | {HELPER, saved.NOTEBOOK, saved.CANONICAL_NOTEBOOK}
    names |= {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {name: sha((root / name).read_bytes()) for name in names}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return root, provenance


def verify(root, provenance):
    gate = package._load_qpe_feedback_public_gate(root, provenance)
    return gate.verify_saved(root, provenance=provenance)


def test_original_snapshot_and_actual_ledgers_verify_without_scientific_imports(registered, monkeypatch):
    root, provenance = registered
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"qiskit", "numpy", "openquantum", "qdk"}:
            pytest.fail("Feedback snapshot registration imported a scientific backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = verify(root, provenance)
    assert result["files"] == 27 and result["native_calls"] == 0
    assert result["new_quantum_execution"] is False
    assert result["original_notebook_sha256"] == saved.CANONICAL_NOTEBOOK_SHA256


@pytest.mark.parametrize("name", [HELPER, "scripts/verify_qpe_feedback_saved.py"])
def test_rewritten_provenance_cannot_execute_a_changed_verifier(registered, tmp_path, name):
    root, provenance = registered
    marker = tmp_path / "unbound-code-ran"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (root / name).write_bytes(raw)
    provenance["file_sha256"][name] = sha(raw)
    with pytest.raises(ValueError, match="identity changed"):
        verify(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["manifest", "report", "canonical-notebook", "extra-file", "extra-directory", "canonical-extra"])
def test_frozen_graph_and_original_notebook_cannot_be_rebound(registered, mutation):
    root, provenance = registered
    bundle = root / saved.BUNDLE
    if mutation == "manifest":
        path = bundle / "manifest.json"
        value = json.loads(path.read_bytes())
        value["source_dirty"] = True
        path.write_text(json.dumps(value))
    elif mutation == "report":
        path = bundle / "n1-m3/report.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "canonical-notebook":
        path = root / saved.CANONICAL_NOTEBOOK
        path.write_bytes(path.read_bytes() + b"\n")
        provenance["file_sha256"][saved.CANONICAL_NOTEBOOK] = sha(path.read_bytes())
    elif mutation == "extra-file":
        (bundle / "controller.json").write_text('{"private":true}')
    elif mutation == "extra-directory":
        (bundle / ".cache").mkdir()
    else:
        (root / saved.VALIDATION / "private.log").write_text("private")
    with pytest.raises(ValueError):
        verify(root, provenance)


@pytest.mark.parametrize("mutation", ["linked-member", "linked-parent", "fifo", "hardlink"])
def test_links_and_special_members_reject_without_blocking(registered, tmp_path, mutation):
    root, provenance = registered
    path = root / saved.BUNDLE / "summary.csv"
    if mutation == "linked-parent":
        original = root / saved.BUNDLE / "n1-m3"
        outside = tmp_path / "outside-task"
        original.rename(outside)
        original.symlink_to(outside, target_is_directory=True)
    elif mutation == "hardlink":
        import os
        os.link(path, tmp_path / "extra-link")
    else:
        path.unlink()
        if mutation == "linked-member":
            path.symlink_to(ROOT / saved.BUNDLE / "summary.csv")
        else:
            import os
            os.mkfifo(path)
    with pytest.raises(ValueError):
        verify(root, provenance)


def test_current_output_refresh_needs_provenance_and_preserves_canonical_snapshot(registered):
    root, provenance = registered
    canonical = (root / saved.CANONICAL_NOTEBOOK).read_bytes()
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    notebook["metadata"]["fixture_output_refresh"] = True
    for cell in notebook["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    notebook["cells"][1]["outputs"].append({"output_type": "stream", "name": "stdout", "text": "Refreshed presentation\n"})
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="current package provenance"):
        verify(root, provenance)
    provenance["file_sha256"][saved.NOTEBOOK] = sha(path.read_bytes())
    assert verify(root, provenance)["current_notebook_source_sha256"] == saved.NOTEBOOK_SOURCE_SHA256
    assert (root / saved.CANONICAL_NOTEBOOK).read_bytes() == canonical


@pytest.mark.parametrize("mutation", ["source", "label", "incomplete", "error", "missing-circuit", "invalid-metadata"])
def test_current_provenance_cannot_certify_invalid_notebook_contracts(registered, mutation):
    root, provenance = registered
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if mutation == "source":
        code[0]["source"] = "print('replacement')\n"
    elif mutation == "label":
        code[1]["metadata"]["label"] = "replacement-label"
    elif mutation == "incomplete":
        code[0]["execution_count"] = None
    elif mutation == "error":
        code[0]["outputs"].append({"output_type": "error", "ename": "RuntimeError", "evalue": "fixture", "traceback": []})
    elif mutation == "missing-circuit":
        code[1]["outputs"] = []
    else:
        code[1]["metadata"] = None
    path.write_text(json.dumps(notebook))
    provenance["file_sha256"][saved.NOTEBOOK] = sha(path.read_bytes())
    with pytest.raises(ValueError):
        verify(root, provenance)


def test_site_uses_served_snapshot_and_two_authenticated_notebooks(registered, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    root, provenance = registered
    site, code = tmp_path / "site", tmp_path / "site/code"
    names = set(saved.SOURCE_HASHES) | {HELPER, saved.NOTEBOOK, "PROVENANCE.json"}
    for name in names:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    for directory in (saved.BUNDLE, saved.VALIDATION):
        shutil.copytree(root / directory, site / directory)
    (site / saved.NOTEBOOK).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / saved.NOTEBOOK, site / saved.NOTEBOOK)
    checker = selected_script("check_site", _read_bounded_public_member=package._read_bounded_public_member,
                              _load_qpe_feedback_public_gate=package._load_qpe_feedback_public_gate)
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": sha((code / "PROVENANCE.json").read_bytes())}}
    assert not (code / "assets").exists()
    assert checker.site_qpe_feedback_saved_errors(site, identity) == []
    (site / saved.NOTEBOOK).write_bytes((site / saved.NOTEBOOK).read_bytes() + b"\n")
    assert checker.site_qpe_feedback_saved_errors(site, identity)


def test_exports_routes_dependencies_and_fixed_byte_protection():
    exporter = selected_script("export_code", LINDBLAD_PORT_FILES=frozenset())
    exact = {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES} | {saved.CANONICAL_NOTEBOOK}
    assert exporter.PUBLIC_QPE_FEEDBACK_FILES == exact and len(exact) == 28
    assert all(exporter.include_generated_file(name) for name in exact)
    assert not exporter.include_generated_file(saved.BUNDLE + "/private.log")
    assert not exporter.include_generated_file(saved.VALIDATION + "/other.ipynb")
    assert {"generate_qpe_feedback_resources.py", "verify_qpe_feedback_saved.py", "verify_qpe_feedback_public.py"} <= set(exporter.PUBLIC_SCRIPTS)
    assert {saved.BUNDLE + "/requirements-lock.txt", saved.BUNDLE + "/LICENSE.txt"} <= exporter.PUBLIC_DEPENDENCY_FILES
    normalizer = ast.parse((ROOT / "scripts/normalize_figures.py").read_bytes())
    immutable = next(ast.literal_eval(n.value.args[0]) for n in normalizer.body
                     if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IMMUTABLE_CSVS" for t in n.targets))
    assert saved.BUNDLE + "/summary.csv" in immutable
    if (ROOT / "myst.yml").is_file():
        qpe = next(e for e in yaml.safe_load((ROOT / "myst.yml").read_text())["project"]["toc"] if e.get("file") == "algorithms/qft-qpe.md")
        iterative = next(e for e in qpe["children"] if e.get("file") == "algorithms/qft-qpe/iterative.md")
        assert {"file": saved.NOTEBOOK, "title": "Complete feedback QPE resource notebook"} in iterative["children"]
