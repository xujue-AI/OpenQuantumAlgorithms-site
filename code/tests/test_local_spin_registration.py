"""Saved local-spin publication and literal circuit-excerpt contracts."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import stat
import sys
import zipfile

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_script(name):
    spec = importlib.util.spec_from_file_location("local_spin_registration_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


saved = load_script("verify_reusable_bath_local_spin_saved")
renderer = load_script("render_reusable_bath_local_spin")
export = load_script("export_code")
package = load_script("verify_public_package")


@pytest.fixture
def registered(tmp_path):
    root = tmp_path / "registered"
    names = set(saved.SOURCE_HASHES) | {saved.NOTEBOOK, saved.ARCHIVE,
        "scripts/verify_reusable_bath_local_spin_saved.py", "scripts/render_reusable_bath_local_spin.py"}
    names |= {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    names |= {saved.PRESENTATION + "/" + name for name in saved.PRESENTATION_HASHES}
    for name in names:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    provenance = {"schema": "openquantum-public-package-v2", "file_sha256": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
    return root.resolve(), provenance


def test_authenticated_saved_gate_is_read_only_and_runs_no_science(registered, monkeypatch):
    root, provenance = registered
    before = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in provenance["file_sha256"]}
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"numpy", "qiskit", "scipy", "qdk", "openquantum"}:
            raise AssertionError("saved gate imported scientific code")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    gate = package._load_local_spin_saved_gate(root, provenance)
    result = gate.verify_saved(root, provenance=provenance)
    assert result == {"status": "saved_component_verified", "files": 21, "bytes": 2066516,
        "saved_source_exports": 9, "saved_source_instructions": 66680,
        "all_components_verified": False, "new_quantum_execution": False,
        "new_source_generation": False, "new_native_queries": 0}
    assert before == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in before}


@pytest.mark.parametrize("name,error", [
    ("artifact-manifest.json", "manifest identity"),
    ("artifact-manifest.sha256", "sidecar"),
    ("source-exports/n2-ground.qasm", "payload identity"),
    ("reference.json", "payload identity"),
])
def test_saved_manifest_sidecar_and_data_corruption_are_rejected(registered, name, error):
    root, _ = registered
    path = root / saved.BUNDLE / name
    raw = path.read_bytes()
    path.write_bytes(b"!" + raw[1:])
    with pytest.raises(ValueError, match=error):
        saved.verify_saved(root)


@pytest.mark.parametrize("change", ["extra", "nested", "symlink"])
def test_closed_saved_tree_rejects_private_or_linked_members(registered, change):
    root, _ = registered
    bundle = root / saved.BUNDLE
    if change == "extra":
        (bundle / "controller.json").write_text('{}')
    elif change == "nested":
        (bundle / "source/.cache").mkdir()
    else:
        member = bundle / "cases.json"
        member.unlink()
        member.symlink_to(bundle / "source/cases.json")
    with pytest.raises(ValueError, match="unrecorded|Unexpected|Linked"):
        saved.verify_saved(root)


def test_live_source_and_authenticated_helper_identity_are_required(registered):
    root, provenance = registered
    path = root / "src/openquantum/simulation/reusable_bath_local_spin.py"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Current local-spin source identity"):
        saved.verify_saved(root)
    helper = root / "scripts/verify_reusable_bath_local_spin_saved.py"
    helper.write_bytes(helper.read_bytes() + b"\n")
    provenance["file_sha256"][str(helper.relative_to(root))] = hashlib.sha256(helper.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="saved verifier identity"):
        package._load_local_spin_saved_gate(root, provenance)


@pytest.mark.parametrize("change", ["unsafe_name", "linked_mode", "contents", "duplicate"])
def test_archive_structure_is_checked_even_with_an_authenticated_new_container(registered, monkeypatch, change):
    root, _ = registered
    raw = io.BytesIO()
    with zipfile.ZipFile(raw, "w", zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(sorted(saved.BUNDLE_FILES)):
            filename = "reusable-bath-local-spin-evidence/" + name
            data = (root / saved.BUNDLE / name).read_bytes()
            info = zipfile.ZipInfo(filename)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            if index == 0:
                if change == "unsafe_name":
                    info.filename = "../escape.json"
                elif change == "linked_mode":
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                elif change == "contents":
                    data = b"!" + data[1:]
            archive.writestr(info, data)
        if change == "duplicate":
            archive.writestr("reusable-bath-local-spin-evidence/cases.json", b"{}")
    contents = raw.getvalue()
    (root / saved.ARCHIVE).write_bytes(contents)
    monkeypatch.setattr(saved, "ARCHIVE_SHA256", hashlib.sha256(contents).hexdigest())
    with pytest.raises(ValueError, match="ZIP"):
        saved.verify_saved(root)


def refresh_notebook(root):
    path = root / saved.NOTEBOOK
    notebook = json.loads(path.read_bytes())
    codes = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for count, cell in enumerate(codes, 1):
        cell["execution_count"] = count
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "Refreshed saved display\n"}]
    path.write_text(json.dumps(notebook))
    return path, notebook


def test_current_provenance_authenticates_refreshed_outputs_without_changing_sources(registered):
    root, provenance = registered
    path, _ = refresh_notebook(root)
    with pytest.raises(ValueError, match="current package provenance"):
        saved.verify_saved(root, provenance=provenance)
    provenance["file_sha256"][saved.NOTEBOOK] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert saved.verify_saved(root, provenance=provenance)["files"] == 21


@pytest.mark.parametrize("change", ["source", "error", "execution", "uncounted_output"])
def test_notebook_semantics_and_execution_labels_are_preserved(registered, change):
    root, provenance = registered
    path, notebook = refresh_notebook(root)
    codes = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if change == "source":
        codes[0]["source"] += ["raise RuntimeError('changed')\n"]
    elif change == "error":
        codes[0]["outputs"] = [{"output_type": "error", "ename": "ValueError", "evalue": "fixture", "traceback": []}]
    elif change == "execution":
        codes[0]["execution_count"] = 7
    else:
        for cell in codes:
            cell["execution_count"] = None
    path.write_text(json.dumps(notebook))
    provenance["file_sha256"][saved.NOTEBOOK] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sources changed|error output|sequence changed|unexecuted outputs"):
        saved.verify_saved(root, provenance=provenance)


def test_notebook_cannot_be_skipped_and_malformed_metadata_is_rejected(registered):
    root, provenance = registered
    with pytest.raises(ValueError, match="notebook roots"):
        saved.verify_saved(root, notebook_roots=())
    path, notebook = refresh_notebook(root)
    notebook["metadata"]["kernelspec"]["name"] = "unbound-kernel"
    path.write_text(json.dumps(notebook))
    provenance["file_sha256"][saved.NOTEBOOK] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Python metadata"):
        saved.verify_saved(root, provenance=provenance)


def test_saved_prefix_is_an_actual_first_step_and_rejects_modified_source():
    source = (ROOT / renderer.SOURCE).read_bytes()
    operations = renderer.parse_excerpt(source)
    assert operations[:3] == tuple(("reset", (q,), None) for q in range(3))
    assert len(operations) == 22 and operations[3][0:2] == ("rz", (0,))
    assert operations[12] == ("rz", (2,), 0.0086506066731954923)
    assert operations[-1] == ("rz", (0,), .125)
    with pytest.raises(ValueError, match="QASM identity"):
        renderer.parse_excerpt(source.replace(b"qreg q[3]", b"qreg q[4]"))


def test_exact_exports_navigation_required_assets_and_figure_protection():
    exact = {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    presentation = {saved.PRESENTATION + "/" + name for name in saved.PRESENTATION_HASHES}
    assert export.PUBLIC_LOCAL_SPIN_FILES == exact
    assert export.PUBLIC_LOCAL_SPIN_PRESENTATION_FILES == presentation
    assert all(export.include_generated_file(name) for name in exact | presentation | {saved.ARCHIVE})
    assert not export.include_generated_file(saved.BUNDLE + "/controller.json")
    assert not export.include_generated_file(saved.PRESENTATION + "/first-step.png")
    assert saved.ARCHIVE in export.PUBLIC_ARCHIVES
    assert {"render_reusable_bath_local_spin.py", "verify_reusable_bath_local_spin_saved.py"} <= set(export.PUBLIC_SCRIPTS)
    immutable = next(ast.literal_eval(node.value.args[0]) for node in ast.parse((ROOT / "scripts/normalize_figures.py").read_text()).body
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "IMMUTABLE_SVGS" for target in node.targets))
    assert saved.PRESENTATION + "/first-step.svg" in immutable


def test_local_spin_navigation_and_required_site_assets():
    if not (ROOT / "scripts/check_site.py").is_file() or not (ROOT / "myst.yml").is_file():
        pytest.skip("Book navigation and site checks are deliberately outside the public example package")
    exact = {saved.BUNDLE + "/" + name for name in saved.BUNDLE_FILES}
    presentation = {saved.PRESENTATION + "/" + name for name in saved.PRESENTATION_HASHES}
    toc = yaml.safe_load((ROOT / "myst.yml").read_text())["project"]["toc"]
    simulation = next(entry for entry in toc if entry.get("file") == "algorithms/quantum-simulation.md")
    gibbs = next(entry for entry in simulation["children"] if entry.get("file") == "algorithms/simulation/gibbs-preparation.md")
    article = next(entry for entry in gibbs["children"] if entry.get("file") == "algorithms/simulation/reusable-bath-local-spin.md")
    assert article["children"] == [{"file": saved.NOTEBOOK, "title": "Saved local-spin bath evidence"}]
    site = load_script("check_site")
    required = set(site.required_artifacts())
    assert exact | presentation | {saved.ARCHIVE, saved.NOTEBOOK} <= required
    assert {"code/" + name for name in export.PUBLIC_LOCAL_SPIN_SOURCE_FILES} <= required


def test_incomplete_preview_never_loads_the_saved_gate(tmp_path, monkeypatch):
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Publication site checks are deliberately outside the public example package")
    site = load_script("check_site")
    calls = []
    monkeypatch.setattr(site, "_load_local_spin_saved_gate", lambda *args: calls.append(args))
    assert site.site_local_spin_saved_errors(tmp_path, {})
    assert calls == []


def test_nested_case_and_isolated_checks_are_collected_in_a_small_package(registered, monkeypatch):
    root, _ = registered
    monkeypatch.setattr(export, "PUBLIC_SCRIPTS", ("render_reusable_bath_local_spin.py", "verify_reusable_bath_local_spin_saved.py"))
    monkeypatch.setattr(export, "PUBLIC_BOOTSTRAP_FILES", ())
    monkeypatch.setattr(export, "PUBLIC_TAIL_RESOURCE_POLICY_FILES", ())
    monkeypatch.setattr(export, "PUBLIC_ARTIFACT_TREES", ("simulation",))
    monkeypatch.setattr(export, "FORMAL_FILES", ())
    required = ["citations.bib", "binder/assemble_code.py", "binder/postBuild", "binder/environment.yml", "binder/README.md",
        "assets/generated/optimization/sampled/optimization.py", "assets/generated/optimization/sampled-runs.json",
        "assets/generated/grover/circuit.qasm", "assets/generated/resources/grover-surface-code.json", "assets/generated/resources/grover-surface-code.csv"]
    for name in required:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture")
    files = export.package_files(root)
    assert all(files[name] == (root / name).read_bytes() for name in export.PUBLIC_LOCAL_SPIN_SOURCE_FILES)
    assert set(export.PUBLIC_LOCAL_SPIN_FILES | export.PUBLIC_LOCAL_SPIN_PRESENTATION_FILES) <= files.keys()
