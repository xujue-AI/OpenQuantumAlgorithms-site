"""Authenticated retained resource inspection; no native or state replay."""
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
READER = "scripts/verify_dqi_structured_resource_saved.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def script(name):
    path = ROOT / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location("resource_registration_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package = script("verify_public_package")
BUNDLE = package.DQI_RESOURCE_DIRECTORY


def selected_script(name, **values):
    path = ROOT / "scripts" / (name + ".py")
    parsed = ast.parse(path.read_bytes())
    nodes = [n for n in parsed.body if isinstance(n, (ast.Assign, ast.AnnAssign, ast.FunctionDef))]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    module = ModuleType("resource_selected_" + name)
    module.__file__ = str(path)
    module.__dict__.update(Path=Path, hashlib=hashlib, json=json, **values)
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.fixture
def registered(tmp_path):
    root = (tmp_path / "registered").resolve()
    manifest = json.loads((ROOT / BUNDLE / "manifest.json").read_bytes())
    names = set(package.DQI_RESOURCE_SOURCE_HASHES) | set(package.DQI_RESOURCE_INPUT_BINDINGS) | {READER}
    names |= {BUNDLE + "/" + name for name in set(manifest["files"]) | {"manifest.json"}}
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    provenance = {"schema": "openquantum-public-package-v2",
                  "file_sha256": {name: sha((root / name).read_bytes()) for name in names}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance))
    return root, provenance


def verify(root, provenance):
    gate = package._load_dqi_resource_saved_gate(root, provenance)
    return package._verify_dqi_resource_saved_identity(gate, root, provenance=provenance)


def test_fixed_d15_packet_reads_without_scientific_import_or_native_replay(registered, monkeypatch):
    root, provenance = registered
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"openquantum", "qiskit", "qdk", "numpy", "scipy"}:
            pytest.fail("Resource saved gate imported an execution backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = verify(root, provenance)
    assert result["files"] == 14 and result["bytes"] == 251257
    assert result["scientific_or_native_work_executed"] is False
    assert [row["t_states_all_draws"] for row in result["cases"]] == [930, 1149]


@pytest.mark.parametrize("name", [READER, *sorted(package.DQI_RESOURCE_SOURCE_HASHES)])
def test_remanifested_unbound_source_cannot_execute(registered, tmp_path, name):
    root, provenance = registered
    marker = tmp_path / "unbound-reader-ran"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n".encode()
    (root / name).write_bytes(raw)
    provenance["file_sha256"][name] = sha(raw)
    with pytest.raises(ValueError, match="source changed|reader identity changed"):
        verify(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("name", sorted(package.DQI_RESOURCE_INPUT_BINDINGS))
def test_every_frozen_input_binding_is_enforced_before_reader_dispatch(registered, name, monkeypatch):
    root, provenance = registered
    gate = package._load_dqi_resource_saved_gate(root, provenance)
    called = []
    monkeypatch.setattr(gate, "verify_saved_resources", lambda *_: called.append(True))
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    provenance["file_sha256"][name] = sha(path.read_bytes())
    with pytest.raises(ValueError, match="Frozen DQI resource input changed"):
        package._verify_dqi_resource_saved_identity(gate, root, provenance=provenance)
    assert called == []


@pytest.mark.parametrize("mutation", ["manifest", "payload", "extra-file", "extra-directory", "symlink"])
def test_closed_packet_rejects_changed_or_unreviewed_members(registered, mutation):
    root, provenance = registered
    if mutation in ("manifest", "payload"):
        path = root / BUNDLE / ("manifest.json" if mutation == "manifest" else "n2-compiled.ndjson")
        path.write_bytes(path.read_bytes() + b"\n")
    elif mutation == "extra-file":
        (root / BUNDLE / "controller.json").write_text('{"private":true}')
    elif mutation == "extra-directory":
        (root / BUNDLE / ".cache").mkdir()
    else:
        path = root / BUNDLE / "results.json"
        path.unlink()
        path.symlink_to(ROOT / BUNDLE / "results.json")
    with pytest.raises(ValueError):
        verify(root, provenance)


def test_site_reads_canonical_asset_inputs_and_authenticated_preview_sources(registered, tmp_path):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    root, provenance = registered
    site, code = tmp_path / "site", tmp_path / "site/code"
    names = set(package.DQI_RESOURCE_SOURCE_HASHES) | {READER, "PROVENANCE.json"}
    names |= {name for name in package.DQI_RESOURCE_INPUT_BINDINGS if not name.startswith("assets/")}
    for name in names:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, target)
    shutil.copytree(root / "assets", site / "assets")
    assert not (code / "assets").exists()
    checker = selected_script("check_site", _load_dqi_resource_saved_gate=package._load_dqi_resource_saved_gate,
                              _verify_dqi_resource_saved_identity=package._verify_dqi_resource_saved_identity,
                              _read_bounded_public_member=package._read_bounded_public_member)
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": sha((code / "PROVENANCE.json").read_bytes())}}
    assert checker.site_dqi_resource_saved_errors(site, identity) == []
    (code / "PROVENANCE.json").write_text("{}")
    assert checker.site_dqi_resource_saved_errors(site, identity)


def test_exact_exports_only_add_two_reviewed_ndjson_streams():
    exporter = selected_script("export_code", LINDBLAD_PORT_FILES=frozenset())
    manifest = json.loads((ROOT / BUNDLE / "manifest.json").read_bytes())
    exact = {BUNDLE + "/" + name for name in set(manifest["files"]) | {"manifest.json"}}
    assert exporter.PUBLIC_DQI_RESOURCE_FILES == exact and len(exact) == 14
    assert all(exporter.include_generated_file(name) for name in exact)
    assert {name for name in exact if name.endswith(".ndjson")} == {BUNDLE + "/n2-compiled.ndjson", BUNDLE + "/n3-compiled.ndjson"}
    assert not exporter.include_generated_file(BUNDLE + "/other.ndjson")
    assert not exporter.include_generated_file(BUNDLE + "/private.json")
    assert ".ndjson" not in exporter.ARTIFACT_SUFFIXES
    assert {"generate_optimization_dqi_resources.py", "verify_dqi_structured_resource_saved.py"} <= set(exporter.PUBLIC_SCRIPTS)
    assert {BUNDLE + "/LICENSE.txt", BUNDLE + "/requirements-lock.txt"} <= exporter.PUBLIC_DEPENDENCY_FILES
