"""Saved measured-AND evidence and notebook-publication trust boundaries."""
from __future__ import annotations

import ast
import base64
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import struct
from types import ModuleType
import zipfile
import zlib

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/verify_shor_measured_and_saved.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("and_registration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved = script("verify_shor_measured_and_saved")
package = script("verify_public_package")


def selected_script(name, **values):
    path = ROOT / "scripts" / (name + ".py")
    parsed = ast.parse(path.read_bytes())
    nodes = [node for node in parsed.body if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("and_selected_" + name)
    module.__file__ = str(path)
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, zipfile=zipfile, **values)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def registered(tmp_path):
    root = (tmp_path / "registered").resolve()
    names = set(saved.SOURCE_HASHES) | {HELPER, saved.NOTEBOOK, saved.CANONICAL_NOTEBOOK} | set(saved.SIDE_HASHES)
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
    gate = package._load_measured_and_saved_gate(root, provenance)
    return gate.verify_saved(root, provenance=provenance)


def test_saved_snapshot_requires_no_scientific_backend(registered, monkeypatch):
    root, provenance = registered
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"qiskit", "numpy", "openquantum", "qdk", "matplotlib"}:
            pytest.fail("Saved measured-AND checker imported a scientific backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = verify(root, provenance)
    assert result["files"] == 31 and result["saved_module_tests"] == 109
    assert result["saved_addition_cases"] == 6 and result["saved_order_cases"] == 2
    assert result["new_quantum_execution"] is False and result["native_calls"] == 0


@pytest.mark.parametrize("name", [HELPER, "src/openquantum/shor_measured_and.py", "tests/test_shor_measured_and.py"])
def test_rewritten_provenance_cannot_promote_changed_code(registered, tmp_path, name):
    root, provenance = registered
    marker = tmp_path / "unbound-code-ran"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (root / name).write_bytes(raw)
    provenance["file_sha256"][name] = sha(raw)
    with pytest.raises(ValueError, match="identity changed"):
        verify(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("mutation", ["manifest", "qasm", "zip", "summary", "png", "canonical", "extra-file", "extra-dir", "extra-sidecar"])
def test_original_evidence_remains_fixed(registered, mutation):
    root, provenance = registered
    names = {"manifest": saved.BUNDLE + "/manifest.json", "qasm": saved.BUNDLE + "/addition-w2-measured.qasm",
             "zip": saved.ARCHIVE, "summary": saved.SUMMARY, "png": saved.PNG, "canonical": saved.CANONICAL_NOTEBOOK}
    if mutation in names:
        path = root / names[mutation]
        path.write_bytes(path.read_bytes() + b"\n")
        provenance["file_sha256"][names[mutation]] = sha(path.read_bytes())
    elif mutation == "extra-file":
        (root / saved.BUNDLE / "controller.json").write_text('{"private":true}')
    elif mutation == "extra-dir":
        (root / saved.BUNDLE / ".cache").mkdir()
    else:
        (root / saved.PARENT / "measured-and-private.log").write_text("private")
    with pytest.raises(ValueError):
        verify(root, provenance)


@pytest.mark.parametrize("mutation", ["symlink", "fifo", "hardlink"])
def test_unsafe_saved_member_rejects_without_opening_science(registered, tmp_path, mutation):
    import os
    root, provenance = registered
    path = root / saved.BUNDLE / "scaling.csv"
    if mutation == "hardlink":
        os.link(path, tmp_path / "extra-link")
    else:
        path.unlink()
        if mutation == "symlink":
            path.symlink_to(ROOT / saved.BUNDLE / "scaling.csv")
        else:
            os.mkfifo(path)
    with pytest.raises(ValueError):
        verify(root, provenance)


@pytest.mark.parametrize("mutation", ["duplicate", "mode", "payload"])
def test_zip_structure_remains_checked_after_a_new_container_pin(registered, monkeypatch, mutation):
    root, _ = registered
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(sorted(saved.BUNDLE_FILES)):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            raw = (root / saved.BUNDLE / name).read_bytes()
            if index == 0 and mutation == "mode":
                info.external_attr = (stat.S_IFLNK | 0o644) << 16
            if index == 0 and mutation == "payload":
                raw = b"!" + raw[1:]
            archive.writestr(info, raw)
        if mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr(sorted(saved.BUNDLE_FILES)[0], b"{}")
    raw = output.getvalue()
    (root / saved.ARCHIVE).write_bytes(raw)
    monkeypatch.setitem(saved.SIDE_HASHES, saved.ARCHIVE, sha(raw))
    with pytest.raises(ValueError, match="ZIP"):
        saved.verify_saved(root)


def test_current_png_bytes_can_refresh_with_current_provenance(registered):
    root, provenance = registered
    canonical = (root / saved.CANONICAL_NOTEBOOK).read_bytes()
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    data = next(output["data"] for output in code[2]["outputs"] if "image/png" in output.get("data", {}))
    png = base64.b64decode(data["image/png"])
    body = b"fixture\x00refreshed PNG presentation"
    kind = b"tEXt"
    chunk = struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xffffffff)
    data["image/png"] = base64.b64encode(png[:-12] + chunk + png[-12:]).decode()
    notebook["metadata"]["fixture_refresh"] = True
    path.write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="package provenance"):
        verify(root, provenance)
    provenance["file_sha256"][saved.NOTEBOOK] = sha(path.read_bytes())
    assert verify(root, provenance)["notebook_source_sha256"] == saved.NOTEBOOK_SOURCE_SHA256
    assert (root / saved.CANONICAL_NOTEBOOK).read_bytes() == canonical


@pytest.mark.parametrize("mutation", ["source", "incomplete", "error", "missing-png", "bad-crc"])
def test_notebook_sources_and_real_output_contracts_cannot_be_rebound(registered, mutation):
    root, provenance = registered
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if mutation == "source":
        code[0]["source"] = "print('replacement')\n"
    elif mutation == "incomplete":
        code[0]["execution_count"] = None
    elif mutation == "error":
        code[0]["outputs"] = [{"output_type": "error", "ename": "ValueError", "evalue": "fixture", "traceback": []}]
    elif mutation == "missing-png":
        code[2]["outputs"] = []
    else:
        data = next(output["data"] for output in code[2]["outputs"] if "image/png" in output.get("data", {}))
        png = bytearray(base64.b64decode(data["image/png"]))
        png[29] ^= 1
        data["image/png"] = base64.b64encode(png).decode()
    path.write_text(json.dumps(notebook))
    provenance["file_sha256"][saved.NOTEBOOK] = sha(path.read_bytes())
    with pytest.raises(ValueError):
        verify(root, provenance)


def test_site_uses_canonical_assets_and_two_current_notebook_copies(registered, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    root, provenance = registered
    site, code = tmp_path / "site", tmp_path / "site/code"
    names = set(saved.SOURCE_HASHES) | {HELPER, saved.NOTEBOOK, "PROVENANCE.json"}
    for name in names:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copytree(root / saved.PARENT, site / saved.PARENT)
    (site / saved.NOTEBOOK).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / saved.NOTEBOOK, site / saved.NOTEBOOK)
    checker = selected_script("check_site", _read_bounded_public_member=package._read_bounded_public_member,
                              _load_measured_and_saved_gate=package._load_measured_and_saved_gate)
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": sha((code / "PROVENANCE.json").read_bytes())}}
    assert not (code / "assets").exists()
    assert checker.site_measured_and_saved_errors(site, identity) == []
    (site / saved.NOTEBOOK).write_bytes((site / saved.NOTEBOOK).read_bytes() + b"\n")
    assert checker.site_measured_and_saved_errors(site, identity)


def test_closed_exports_navigation_byte_guards_and_inline_kernel():
    exporter = selected_script("export_code", LINDBLAD_PORT_FILES=frozenset())
    exact = {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES} | set(saved.SIDE_HASHES) | {saved.CANONICAL_NOTEBOOK}
    assert exporter.PUBLIC_MEASURED_AND_FILES == exact and len(exact) == 35
    assert all(exporter.include_generated_file(name) for name in exact)
    assert not exporter.include_generated_file(saved.BUNDLE + "/controller.json")
    assert not exporter.include_generated_file(saved.PARENT + "/measured-and-private.json")
    assert saved.ARCHIVE in exporter.PUBLIC_ARCHIVES
    assert {"generate_shor_measured_and.py", "verify_shor_measured_and_saved.py"} <= set(exporter.PUBLIC_SCRIPTS)
    assert {saved.BUNDLE + "/LICENSE.txt", saved.BUNDLE + "/requirements-lock.txt"} <= exporter.PUBLIC_DEPENDENCY_FILES
    normalizer = ast.parse((ROOT / "scripts/normalize_figures.py").read_bytes())
    protected = set()
    for node in normalizer.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("IMMUTABLE_SVGS", "IMMUTABLE_CSVS") for t in node.targets):
            protected.update(ast.literal_eval(node.value.args[0]))
    assert {saved.BUNDLE + "/gadget.svg", saved.BUNDLE + "/addition.svg", saved.BUNDLE + "/scaling.csv"} <= protected
    if (ROOT / "myst.yml").is_file():
        toc = yaml.safe_load((ROOT / "myst.yml").read_text())["project"]["toc"]
        shor = next(entry for entry in toc if entry.get("file") == "algorithms/shor.md")
        measured = next(entry for entry in shor["children"] if entry.get("file") == "algorithms/shor/measured-and.md")
        assert measured["children"] == [{"file": saved.NOTEBOOK, "title": "Measured-AND notebook"}]
    kernel = ast.parse((ROOT / "scripts/execute_notebooks.py").read_bytes())
    assert any(isinstance(node, ast.Constant) and node.value == "--matplotlib=inline" for node in ast.walk(kernel))
