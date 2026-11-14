"""Saved structured-DQI trust boundaries, with no new quantum computation."""
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
BUNDLE = "assets/generated/optimization/dqi-structured"
NOTEBOOK = "notebooks/optimization_dqi_structured.ipynb"
HELPER = "scripts/verify_dqi_structured_saved.py"

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("structured_registration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

package = script("verify_public_package")

def selected_script(name, **values):
    """Capture actual definitions while leaving unrelated imports inert."""
    path = ROOT / "scripts" / (name + ".py")
    parsed = ast.parse(path.read_bytes(), filename=str(path))
    nodes = [n for n in parsed.body if isinstance(n, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("structured_registration_selected_" + name)
    module.__file__ = str(path)
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, **values)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module

@pytest.fixture
def exporter():
    return selected_script("export_code", LINDBLAD_PORT_FILES=frozenset())

@pytest.fixture
def registered(tmp_path):
    root = (tmp_path / "registered").resolve()
    manifest = json.loads((ROOT / BUNDLE / "bundle.json").read_bytes())
    names = set(package.DQI_STRUCTURED_SOURCE_HASHES) | {NOTEBOOK}
    names |= {BUNDLE + "/" + name for name in set(manifest["files"]) | {"bundle.json"}}
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    hashes = {name: sha((root / name).read_bytes()) for name in names}
    provenance = {"schema": "openquantum-public-package-v2", "file_sha256": hashes,
                  "source_sha256": {name: v for name, v in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return root, provenance

def verify(root, provenance):
    gate = package._load_dqi_structured_saved_gate(root, provenance)
    return package._verify_dqi_structured_saved_identity(
        gate, root / BUNDLE, provenance, source_root=root, notebook_roots=(root,))

def test_authenticated_saved_reader_requests_no_quantum_backend(registered, monkeypatch):
    root, provenance = registered
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"qiskit", "numpy", "openquantum", "qdk"}:
            pytest.fail("Saved registration imported a scientific execution backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = verify(root, provenance)
    assert [(c["n"], c["source_operations"]) for c in result["verification"]["small"]["cases"]] == [(2, 116), (3, 336)]
    assert result["verification"]["large"]["n"] == 8
    assert result["verification"]["large"]["source_operations"] == 2876
    assert result["verification"]["large"]["coherent_validation_performed"] is False

@pytest.mark.parametrize("name", [HELPER, "src/openquantum/optimization_dqi_structured.py"])
def test_changed_source_rejects_rewritten_provenance_before_exec(registered, tmp_path, name):
    root, provenance = registered
    marker = tmp_path / "executed-unbound-source"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (root / name).write_bytes(raw)
    provenance["file_sha256"][name] = sha(raw)
    with pytest.raises(ValueError, match="promoted source identity changed"):
        package._load_dqi_structured_saved_gate(root, provenance)
    assert not marker.exists()

@pytest.mark.parametrize("kind", ["manifest", "bundle-notebook", "member", "extra-file", "extra-directory"])
def test_saved_event_rejects_rebinding_and_private_members(registered, kind):
    root, provenance = registered
    bundle = root / BUNDLE
    gate = package._load_dqi_structured_saved_gate(root, provenance)
    if kind == "manifest":
        path = bundle / "bundle.json"
        data = json.loads(path.read_bytes())
        data["scope"] = "forged replacement"
        path.write_text(json.dumps(data))
    elif kind == "bundle-notebook":
        path = bundle / "optimization_dqi_structured.ipynb"
        data = json.loads(path.read_bytes())
        data["metadata"]["forged_output_refresh"] = True
        path.write_text(json.dumps(data))
    elif kind == "member":
        path = bundle / "small/n2/operations.ndjson"
        path.write_bytes(path.read_bytes() + b"\n")
    elif kind == "extra-file":
        (bundle / "controller.json").write_text('{"private":true}')
    else:
        (bundle / ".cache").mkdir()
    with pytest.raises(ValueError):
        package._verify_dqi_structured_saved_identity(
            gate, bundle, provenance, source_root=root, notebook_roots=(root,))

def test_linked_bundle_root_rejected_before_reader(registered, tmp_path):
    root, provenance = registered
    alias = tmp_path / "linked-bundle"
    alias.symlink_to(root / BUNDLE, target_is_directory=True)
    gate = package._load_dqi_structured_saved_gate(root, provenance)
    with pytest.raises(ValueError, match="Unsafe or linked"):
        package._verify_dqi_structured_saved_identity(
            gate, alias, provenance, source_root=root, notebook_roots=(root,))

def test_normal_notebook_refresh_needs_current_provenance(registered):
    root, provenance = registered
    path = root / NOTEBOOK
    original_bundle = (root / BUNDLE / "optimization_dqi_structured.ipynb").read_bytes()
    notebook = json.loads(path.read_bytes())
    for cell in notebook["cells"]:
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    code = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    code["outputs"].append({"output_type": "stream", "name": "stdout", "text": "Saved presentation refreshed\n"})
    notebook["metadata"]["fixture_refresh"] = True
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="current public provenance"):
        verify(root, provenance)
    provenance["file_sha256"][NOTEBOOK] = sha(path.read_bytes())
    assert verify(root, provenance)["verification"]["large"]["n"] == 8
    assert (root / BUNDLE / "optimization_dqi_structured.ipynb").read_bytes() == original_bundle

@pytest.mark.parametrize("change", ["source", "error", "incomplete"])
def test_normal_notebook_rejects_changed_source_or_failed_cells(registered, change):
    root, provenance = registered
    path = root / NOTEBOOK
    data = json.loads(path.read_bytes())
    code = next(c for c in data["cells"] if c["cell_type"] == "code")
    if change == "source":
        code["source"] = "raise RuntimeError('changed source')\n"
    elif change == "error":
        code["outputs"].append({"output_type": "error", "ename": "RuntimeError", "evalue": "fixture", "traceback": []})
    else:
        code["execution_count"] = None
    path.write_text(json.dumps(data))
    provenance["file_sha256"][NOTEBOOK] = sha(path.read_bytes())
    with pytest.raises(ValueError):
        verify(root, provenance)

def test_complete_manifest_rejection_precedes_saved_dispatch(registered, monkeypatch):
    root, _ = registered
    (root / HELPER).write_bytes(b"raise AssertionError('unbound helper')\n")
    for name in ("verify_formal_record", "_load_energy_record_gate", "_load_dqi_structured_saved_gate"):
        monkeypatch.setattr(package, name, lambda *_: pytest.fail("Unverified complete package reached a saved gate"))
    with pytest.raises(ValueError, match="digest mismatch"):
        package.verify_package(root, run_tests=False)

def test_served_bundle_uses_canonical_assets_after_provenance_auth(registered, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    root, provenance = registered
    site, code = tmp_path / "site", tmp_path / "site/code"
    for name in set(package.DQI_STRUCTURED_SOURCE_HASHES) | {NOTEBOOK, "PROVENANCE.json"}:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copytree(root / BUNDLE, site / BUNDLE)
    (site / NOTEBOOK).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / NOTEBOOK, site / NOTEBOOK)
    assert not (code / "assets").exists()
    keys = {"_read_bounded_public_member", "_load_dqi_structured_saved_gate",
            "_verify_dqi_structured_saved_identity", "DQI_STRUCTURED_BUNDLE_DIRECTORY"}
    checker = selected_script("check_site", **{n: v for n, v in vars(package).items() if n in keys})
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": sha((code / "PROVENANCE.json").read_bytes())}}
    assert checker.site_dqi_structured_saved_errors(site, identity) == []
    (code / "PROVENANCE.json").write_text("{}")
    assert checker.site_dqi_structured_saved_errors(site, identity)

def test_closed_exports_dependencies_and_navigation(exporter):
    manifest = json.loads((ROOT / BUNDLE / "bundle.json").read_bytes())
    exact = {BUNDLE + "/" + name for name in set(manifest["files"]) | {"bundle.json"}}
    assert exporter.PUBLIC_DQI_STRUCTURED_FILES == exact and len(exact) == 25
    assert all(exporter.include_generated_file(n) for n in exact)
    for n in (BUNDLE + "/controller.json", BUNDLE + "/private/log.txt", BUNDLE + "/large/new.json"):
        assert not exporter.include_generated_file(n)
    actual = tuple(n for n in exporter.PUBLIC_SCRIPTS if "dqi" in n and "structured" in n and n != "verify_dqi_structured_resource_saved.py")
    assert actual == ("generate_optimization_dqi_structured.py", "validate_optimization_dqi_structured.py",
                      "verify_dqi_structured_saved.py")
    for suffix in ("requirements-lock.txt", "LICENSE.txt", "large/requirements-lock.txt", "large/LICENSE.txt"):
        assert BUNDLE + "/" + suffix in exporter.PUBLIC_DEPENDENCY_FILES
    path = ROOT / "myst.yml"
    if path.is_file():
        toc = yaml.safe_load(path.read_text())["project"]["toc"]
        opt = next(e for e in toc if e.get("file") == "algorithms/optimization.md")
        dqi = next(e for e in opt["children"] if e.get("file") == "algorithms/optimization/dqi.md")
        item = next(e for e in dqi["children"] if e.get("file") == "algorithms/optimization/dqi-structured.md")
        assert item["children"] == [{"file": NOTEBOOK, "title": "Structured DQI notebook"}]
