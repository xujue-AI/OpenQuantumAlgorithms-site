"""Saved-only CPF/GQSP registration boundaries; no scientific backend calls."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_provenance(root):
    hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes())
              for path in root.rglob("*") if path.is_file() and path.name != "PROVENANCE.json"}
    provenance = {"schema": "openquantum-public-package-v2", "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return provenance


@pytest.fixture
def package():
    return load_source(ROOT / "scripts/verify_public_package.py", "hamiltonian_registration_package")


@pytest.fixture(scope="module")
def payload():
    """Read only the reviewed saved files and source bytes."""
    package = load_source(ROOT / "scripts/verify_public_package.py", "hamiltonian_registration_inventory")
    names = set(package.HAMILTONIAN_SOURCE_HASHES)
    for packet in package.HAMILTONIAN_PACKETS.values():
        names |= {packet["helper"], packet["notebook"], packet["archive"]}
        names |= {path.relative_to(ROOT).as_posix()
                  for path in (ROOT / packet["directory"]).rglob("*") if path.is_file()}
    return {name: (ROOT / name).read_bytes() for name in names}


@pytest.fixture
def fixture(tmp_path, payload, package):
    root = (tmp_path / "package").resolve()
    root.mkdir()
    for name, raw in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = write_provenance(root)
    assert package.verify_manifest(root) == provenance
    return root, provenance


def reject_science(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "scipy", "qiskit", "openquantum", "qdk"}:
            pytest.fail("Saved Hamiltonian registration imported a scientific backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.mark.parametrize("kind", ["cpf", "gqsp"])
def test_saved_package_closure_uses_no_scientific_imports(fixture, package, monkeypatch, kind):
    root, provenance = fixture
    reject_science(monkeypatch)
    gate = package._load_hamiltonian_saved_gate(root, provenance, kind)
    result = package._verify_hamiltonian_saved_identity(gate, root, kind=kind, provenance=provenance)
    if kind == "cpf":
        assert result["paired_small_cases"] == 32 and result["portable_execution_performed"] is False
    else:
        assert result["small_numerical_cases"] == 6 and result["new_scientific_execution"] is False
    assert not list(root.rglob("__pycache__"))


@pytest.mark.parametrize("name", [
    "src/openquantum/simulation/corrected_product_formula.py",
    "scripts/generate_simulation_corrected_formula.py", "tests/test_simulation_corrected_formula.py",
    "src/openquantum/simulation/gqsp.py", "scripts/generate_simulation_gqsp.py", "tests/test_simulation_gqsp.py",
    "scripts/generate_cpf2_public.py", "requirements-lock.txt", "LICENSE",
    "scripts/verify_simulation_cpf_public.py", "scripts/verify_simulation_gqsp_public.py",
])
def test_promoted_source_or_reader_change_cannot_be_reauthorized(fixture, package, name):
    root, _ = fixture
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    provenance = write_provenance(root)
    assert package.verify_manifest(root) == provenance
    with pytest.raises(ValueError, match="source or dependency changed|reader identity changed"):
        package._load_hamiltonian_saved_gate(root, provenance, "cpf" if "gqsp" not in name else "gqsp")


@pytest.mark.parametrize("kind", ["cpf", "gqsp"])
@pytest.mark.parametrize("change", ["manifest", "payload", "archive", "extra-file", "extra-directory", "symlink", "hardlink", "fifo"])
def test_packet_mutations_fail_before_saved_dispatch(fixture, package, kind, change):
    root, provenance = fixture
    packet = package.HAMILTONIAN_PACKETS[kind]
    directory = root / packet["directory"]
    if change in {"manifest", "payload", "archive"}:
        path = root / packet["archive"] if change == "archive" else directory / (packet["manifest"] if change == "manifest" else "README.md")
        path.write_bytes(path.read_bytes() + b"\n")
        provenance = write_provenance(root)
    elif change == "extra-file":
        (directory / "unreviewed.py").write_text("raise RuntimeError('must never load')")
    elif change == "extra-directory":
        (directory / "unreviewed").mkdir()
    else:
        path = directory / "README.md"
        raw = path.read_bytes()
        path.unlink()
        if change == "symlink":
            outside = root / "linked-target.md"
            outside.write_bytes(raw)
            path.symlink_to(outside)
        elif change == "hardlink":
            outside = root / "linked-target.md"
            outside.write_bytes(raw)
            os.link(outside, path)
        else:
            os.mkfifo(path)
    gate = SimpleNamespace(verify=lambda *_: pytest.fail("Unverified packet reached saved dispatch"))
    with pytest.raises((ValueError, OSError)):
        package._verify_hamiltonian_saved_identity(gate, root, kind=kind, provenance=provenance)


@pytest.mark.parametrize("kind", ["cpf", "gqsp"])
def test_fresh_notebook_outputs_and_source_segmentation_are_allowed(fixture, package, kind):
    root, _ = fixture
    path = root / package.HAMILTONIAN_PACKETS[kind]["notebook"]
    book = json.loads(path.read_bytes())
    book["metadata"]["fresh_output_marker"] = "new execution"
    for cell in book["cells"]:
        text = cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        cell["source"] = list(text)
    path.write_text(json.dumps(book))
    provenance = write_provenance(root)
    gate = package._load_hamiltonian_saved_gate(root, provenance, kind)
    package._verify_hamiltonian_saved_identity(gate, root, kind=kind, provenance=provenance)


@pytest.mark.parametrize("kind", ["cpf", "gqsp"])
@pytest.mark.parametrize("change", ["source", "source-container", "counter", "error", "missing-image"])
def test_notebook_claims_require_unchanged_sources_and_complete_outputs(fixture, package, kind, change):
    root, _ = fixture
    path = root / package.HAMILTONIAN_PACKETS[kind]["notebook"]
    book = json.loads(path.read_bytes())
    codes = [cell for cell in book["cells"] if cell["cell_type"] == "code"]
    if change == "source":
        codes[0]["source"] = "raise RuntimeError('changed source')"
    elif change == "source-container":
        text = codes[0]["source"] if isinstance(codes[0]["source"], str) else "".join(codes[0]["source"])
        codes[0]["source"] = {text: "a dictionary is not notebook source text"}
    elif change == "counter":
        codes[-1]["execution_count"] = None
    elif change == "error":
        codes[-1]["outputs"].append({"output_type": "error", "ename": "Example", "evalue": "failed", "traceback": []})
    else:
        for cell in codes:
            cell["outputs"] = []
    path.write_text(json.dumps(book))
    provenance = write_provenance(root)
    gate = SimpleNamespace(verify=lambda *_: pytest.fail("Invalid notebook reached saved dispatch"))
    with pytest.raises(ValueError, match="notebook lacks|notebook source must"):
        package._verify_hamiltonian_saved_identity(gate, root, kind=kind, provenance=provenance)


def site_function(package):
    path = ROOT / "scripts/check_site.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "site_hamiltonian_packet_errors")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile,
                 "_read_bounded_public_member": package._read_bounded_public_member,
                 "_load_hamiltonian_saved_gate": package._load_hamiltonian_saved_gate,
                 "_verify_hamiltonian_saved_identity": package._verify_hamiltonian_saved_identity,
                 "HAMILTONIAN_PACKETS": package.HAMILTONIAN_PACKETS,
                 "HAMILTONIAN_SOURCE_HASHES": package.HAMILTONIAN_SOURCE_HASHES}
    export = ast.parse((ROOT / "scripts/export_code.py").read_text())
    for constant in export.body:
        if isinstance(constant, ast.Assign) and any(isinstance(target, ast.Name) and target.id in
                {"PUBLIC_CORRECTED_FORMULA_FILES", "PUBLIC_GQSP_FILES"} for target in constant.targets):
            exec(compile(ast.Module(body=[constant], type_ignores=[]), "export-constants", "exec"), namespace)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def site(tmp_path, payload, fixture, package):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("The site checker is outside the standalone example package")
    _, provenance = fixture
    root = (tmp_path / "site").resolve()
    root.mkdir()
    retained, omitted = {}, {}
    for name, raw in payload.items():
        if name.startswith("assets/"):
            path = root / name
            omitted[name] = {"site_path": name, "sha256": digest(raw)}
        else:
            path = root / "code" / name
            retained[name] = digest(raw)
            if name.startswith("notebooks/"):
                copy = root / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = json.dumps(provenance).encode()
    (root / "code/PROVENANCE.json").write_bytes(raw)
    retained["PROVENANCE.json"] = digest(raw)
    preview = {"schema": "openquantum-source-preview-v1", "retained_sha256": retained, "omitted_assets": omitted}
    return root, preview, site_function(package)


def test_site_uses_authenticated_omitted_assets_and_retained_source(site, monkeypatch):
    root, preview, namespace = site
    reject_science(monkeypatch)
    assert namespace["site_hamiltonian_packet_errors"](root, preview) == []


@pytest.mark.parametrize("change", ["missing-preview", "changed-provenance", "changed-retained", "asset-in-retained-only", "aliased-asset", "changed-asset"])
def test_preview_changes_fail_before_helper_dispatch(site, change):
    root, preview, namespace = site
    name = "assets/generated/simulation/gqsp/README.md"
    if change == "missing-preview":
        preview = {}
    elif change == "changed-provenance":
        path = root / "code/PROVENANCE.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "changed-retained":
        preview["retained_sha256"]["scripts/verify_simulation_cpf_public.py"] = "0" * 64
    elif change == "asset-in-retained-only":
        preview["retained_sha256"][name] = preview["omitted_assets"].pop(name)["sha256"]
    elif change == "aliased-asset":
        preview["omitted_assets"][name]["site_path"] = "code/" + name
    else:
        path = root / name
        path.write_bytes(path.read_bytes() + b"\n")
    namespace["_load_hamiltonian_saved_gate"] = lambda *_: pytest.fail("Unverified preview reached helper dispatch")
    assert namespace["site_hamiltonian_packet_errors"](root, preview)


def test_package_admission_precedes_new_saved_gate(fixture, package, monkeypatch):
    root, _ = fixture
    def reject(_):
        raise ValueError("rejected complete manifest")
    monkeypatch.setattr(package, "verify_manifest", reject)
    monkeypatch.setattr(package, "_load_hamiltonian_saved_gate", lambda *_: pytest.fail("Unverified package reached helper"))
    with pytest.raises(ValueError, match="rejected complete manifest"):
        package.verify_package(root, run_tests=False)


def test_site_dispatch_remains_inside_successful_preview_admission():
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("The site checker is outside the standalone example package")
    tree = ast.parse((ROOT / "scripts/check_site.py").read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    admission = next(node for node in main.body if isinstance(node, ast.Try)
                     and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "verify_preview"
                             for statement in node.body for call in ast.walk(statement)))
    calls = [call for statement in admission.orelse for call in ast.walk(statement)
             if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "site_hamiltonian_packet_errors"]
    assert len(calls) == 1
