"""Degree-two DQI saved publication boundaries, with no numerical replay.

Sources, QASM and the executed notebook remain inert bytes. Real saved-reader
calls authenticate those bytes only; AST dispatch fixtures stop before any
scientific backend, producer, kernel or later publication work.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/optimization/dqi-degree-two"
PUBLIC = "assets/generated/optimization/dqi-degree-two-public"
MANIFEST = PUBLIC + "/projection-manifest.json"
ARCHIVE = PUBLIC + "/dqi-degree-two-download.zip"
DOWNLOAD = PUBLIC + "/download.json"
NOTEBOOK = "notebooks/optimization_dqi_degree_two.ipynb"
HELPER = "scripts/verify_dqi_degree_two_public_saved.py"
SOURCE_CONFIG = "source-config/dqi-degree-two/pyproject.toml"
MANIFEST_PIN = "ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c"
ARCHIVE_PIN = "af0c1eae2fe34d8947d20330fe7b6cb47a5852b216145033bffecda07bbf29ae"
HELPER_PIN = "fde3babe65a1fa8b7ade3249c1b8251407760295363d563c5c56a1cb0acc9fe5"
NOTEBOOK_PIN = "e0685b2eaf4db5f0b6dab5bd873eb5c86d52870db8a6ad7eb2bae71d00e19cd9"
CONFIG_PIN = "fb96cdbcbb8c032581d62d8d04f261e988fced3e66a0f560ea79d7e9c2d72404"
BOOTSTRAP_PIN = "3be26040a631217836c04f479db98d08c5c64edbe2d9bb352e869b2e10e1bdda"
BOOTSTRAP = b'''[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"

[project]
name = "openquantum-examples"
version = "0.4.0"
requires-python = ">=3.12"
license = "MIT"

[tool.setuptools.packages.find]
where = ["src"]
include = ["openquantum*"]
namespaces = true

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
'''
BUNDLE_NAMES = frozenset({
    "optimization_dqi_degree_two.py", "optimization_dqi_degree_two_verify.py",
    "generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
    "verify_dqi_degree_two_saved.py", "optimization_dqi_degree_two.ipynb",
    "requirements-lock.txt", "LICENSE.txt", "input-c5.json", "input-figure-eight.json",
    "bundle.json", "invocation-acceptance.json", "c5/operations.ndjson", "c5/circuit.qasm",
    "c5/ledger.json", "c5/manifest.json", "figure-eight/operations.ndjson",
    "figure-eight/circuit.qasm", "figure-eight/ledger.json", "figure-eight/manifest.json",
    "validation/validation.json", "validation/manifest.json",
})
META_NAMES = frozenset({"README.md", "environment.json", "accepted-components.json",
                        "projection-manifest.json", "dqi-degree-two-download.zip", "download.json"})
PUBLIC_FILES = frozenset(BUNDLE + "/" + name for name in BUNDLE_NAMES) | {
    PUBLIC + "/" + name for name in META_NAMES
}
SOURCE_FILES = frozenset({
    "src/openquantum/optimization_dqi_degree_two.py",
    "src/openquantum/optimization_dqi_degree_two_verify.py",
    "scripts/generate_optimization_dqi_degree_two.py",
    "scripts/validate_optimization_dqi_degree_two.py",
    "scripts/verify_dqi_degree_two_saved.py", HELPER,
    "tests/test_optimization_dqi_degree_two.py",
    "tests/test_optimization_dqi_degree_two_boundaries.py",
})


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    specification = importlib.util.spec_from_file_location("degree_two_registration_" + name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_provenance(root):
    hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes())
              for path in root.rglob("*") if path.is_file() and path.name != "PROVENANCE.json"}
    value = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
             "file_sha256": hashes,
             "source_sha256": {name: pin for name, pin in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(value, sort_keys=True))
    return value


def reject_science(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"openquantum", "qiskit", "numpy", "scipy", "qdk", "cirq"}:
            pytest.fail("Saved degree-two registration reached a scientific import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def package_gate():
    return load_script("verify_public_package")


@pytest.fixture(scope="module")
def payload():
    manifest = json.loads((ROOT / MANIFEST).read_bytes())
    names = set(manifest["files"]) | {MANIFEST, ARCHIVE, DOWNLOAD}
    files = {name: (ROOT / name).read_bytes() for name in names if name != "pyproject.toml"}
    with zipfile.ZipFile(ROOT / ARCHIVE) as archive:
        config = archive.read("pyproject.toml")
    assert len(config) == 705 and digest(config) == CONFIG_PIN
    files[SOURCE_CONFIG] = config
    files["pyproject.toml"] = BOOTSTRAP
    return files


@pytest.fixture
def packet(tmp_path, payload):
    root = (tmp_path / "package").resolve()
    root.mkdir()
    for name, raw in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root, write_provenance(root)


def verify(root, provenance, package_gate, *, gate=None, notebook_roots=None):
    gate = package_gate._load_dqi_degree_two_saved_gate(root, provenance) if gate is None else gate
    return package_gate._verify_dqi_degree_two_public_identity(
        gate, root, source_root=root, provenance=provenance,
        notebook_roots=(root,) if notebook_roots is None else notebook_roots)


def synthetic_saved_result():
    return {"status": "passed_saved_public_degree_two_packet", "public_manifest_sha256": MANIFEST_PIN,
            "scoped_files": 38, "archive_members": 38, "archive_checked": True,
            "archive_bytes": 233_048, "original_bundle_files": 22, "genuine_companion_code_cells": 4,
            "scientific_work_reexecuted": False, "notebook_executed": False,
            "portable_scientific_replay_validated": False, "publication_complete": False}


def test_frozen_packet_reader_notebook_and_distinct_config_role(payload):
    assert len(BUNDLE_NAMES) == 22 and len(PUBLIC_FILES) == 28
    assert len(payload[MANIFEST]) == 7_766 and digest(payload[MANIFEST]) == MANIFEST_PIN
    assert len(payload[ARCHIVE]) == 233_048 and digest(payload[ARCHIVE]) == ARCHIVE_PIN
    assert len(payload[HELPER]) == 30_011 and digest(payload[HELPER]) == HELPER_PIN
    assert len(payload[NOTEBOOK]) == 19_950 and digest(payload[NOTEBOOK]) == NOTEBOOK_PIN
    assert len(payload[SOURCE_CONFIG]) == 705 and digest(payload[SOURCE_CONFIG]) == CONFIG_PIN
    assert len(BOOTSTRAP) == 351 and digest(BOOTSTRAP) == BOOTSTRAP_PIN
    assert payload["pyproject.toml"] == BOOTSTRAP and payload["pyproject.toml"] != payload[SOURCE_CONFIG]


def test_actual_saved_identity_accepts_original_config_without_replacing_global_bootstrap(
        packet, package_gate, monkeypatch):
    root, provenance = packet
    reject_science(monkeypatch)
    before = (root / "pyproject.toml").read_bytes()
    result = verify(root, provenance, package_gate)
    expected = synthetic_saved_result()
    for name, value in expected.items():
        assert result[name] == value and type(result[name]) is type(value)
    assert (root / "pyproject.toml").read_bytes() == before == BOOTSTRAP
    assert digest((root / SOURCE_CONFIG).read_bytes()) == CONFIG_PIN


def test_source_overlay_is_dqi_only_and_removed_after_success(packet, package_gate):
    root, provenance = packet
    observed = []
    def saved(artifact_root, *, source_root, notebook_path, expected_manifest_sha256,
              archive_path, time_limit_seconds):
        source_root = Path(source_root)
        observed.append(source_root)
        assert artifact_root == root and source_root != root
        assert (source_root / "pyproject.toml").read_bytes() == (root / SOURCE_CONFIG).read_bytes()
        assert digest((source_root / "pyproject.toml").read_bytes()) == CONFIG_PIN
        for name in SOURCE_FILES:
            assert (source_root / name).read_bytes() == (root / name).read_bytes()
        assert notebook_path == root / NOTEBOOK
        assert expected_manifest_sha256 == MANIFEST_PIN and archive_path == root / ARCHIVE
        assert time_limit_seconds == 10 and (root / "pyproject.toml").read_bytes() == BOOTSTRAP
        return synthetic_saved_result()
    verify(root, provenance, package_gate, gate=SimpleNamespace(verify_public_packet=saved))
    assert len(observed) == 1 and not observed[0].exists()
    assert (root / "pyproject.toml").read_bytes() == BOOTSTRAP


def test_source_overlay_cleanup_and_global_bootstrap_survive_reader_failure(packet, package_gate):
    root, provenance = packet
    observed = []
    def reject(_artifact_root, *, source_root, **kwargs):
        observed.append(Path(source_root))
        assert digest((Path(source_root) / "pyproject.toml").read_bytes()) == CONFIG_PIN
        raise ValueError("synthetic saved-reader failure")
    with pytest.raises(ValueError, match="synthetic saved-reader failure"):
        verify(root, provenance, package_gate, gate=SimpleNamespace(verify_public_packet=reject))
    assert len(observed) == 1 and not observed[0].exists()
    assert (root / "pyproject.toml").read_bytes() == BOOTSTRAP


@pytest.mark.parametrize("change", ["missing-provenance", "wrong-schema", "missing-helper-hash",
                                    "wrong-helper-hash", "changed-helper", "oversized-helper"])
def test_loader_rejects_unbound_public_reader_before_compile(packet, package_gate, monkeypatch, change):
    root, provenance = packet
    if change == "missing-provenance":
        provenance = None
    elif change == "wrong-schema":
        provenance["schema"] = "wrong"
    elif change == "missing-helper-hash":
        provenance["file_sha256"].pop(HELPER)
    elif change == "wrong-helper-hash":
        provenance["file_sha256"][HELPER] = "0" * 64
    else:
        path = root / HELPER
        path.write_bytes(path.read_bytes() + (b" " * 131_072 if change == "oversized-helper" else b"\nraise RuntimeError('untrusted source')\n"))
        provenance["file_sha256"][HELPER] = digest(path.read_bytes())
    monkeypatch.setattr(builtins, "compile", lambda *args, **kwargs: pytest.fail("Unbound public reader reached compile"))
    with pytest.raises((OSError, ValueError)):
        package_gate._load_dqi_degree_two_saved_gate(root, provenance)


@pytest.mark.parametrize("name", [MANIFEST, DOWNLOAD, ARCHIVE])
def test_external_provenance_pin_cannot_be_reauthorized_before_compile(packet, package_gate, monkeypatch, name):
    root, provenance = packet
    provenance["file_sha256"][name] = "0" * 64
    monkeypatch.setattr(builtins, "compile", lambda *args, **kwargs: pytest.fail("Unbound external pin reached compile"))
    with pytest.raises(ValueError):
        package_gate._load_dqi_degree_two_saved_gate(root, provenance)


def test_loader_does_not_use_an_untrusted_module_cache(packet, package_gate, monkeypatch):
    root, provenance = packet
    admitted = package_gate._load_dqi_degree_two_saved_gate(root, provenance)
    monkeypatch.setitem(sys.modules, admitted.__name__, SimpleNamespace(untrusted_marker=True))
    gate = package_gate._load_dqi_degree_two_saved_gate(root, provenance)
    assert gate is not admitted and not getattr(gate, "untrusted_marker", False)
    assert callable(gate.verify_public_packet)


@pytest.mark.parametrize("name", sorted(SOURCE_FILES | {SOURCE_CONFIG}))
def test_remanifested_source_or_config_change_fails_before_reader_dispatch(
        packet, package_gate, monkeypatch, name):
    root, provenance = packet
    gate = SimpleNamespace(verify_public_packet=lambda *args, **kwargs: pytest.fail("Changed bound source reached saved dispatch"))
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    provenance["file_sha256"][name] = digest(path.read_bytes())
    with pytest.raises((OSError, ValueError)):
        verify(root, provenance, package_gate, gate=gate)
    assert (root / "pyproject.toml").read_bytes() == BOOTSTRAP


@pytest.mark.parametrize("name", [MANIFEST, DOWNLOAD, ARCHIVE])
def test_remanifested_asset_change_fails_before_reader_dispatch(packet, package_gate, name):
    root, provenance = packet
    gate = SimpleNamespace(verify_public_packet=lambda *args, **kwargs: pytest.fail("Changed artifact reached saved dispatch"))
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    provenance["file_sha256"][name] = digest(path.read_bytes())
    with pytest.raises((OSError, ValueError)):
        verify(root, provenance, package_gate, gate=gate)


@pytest.mark.parametrize("name", [BUNDLE + "/c5/operations.ndjson", BUNDLE + "/validation/validation.json"])
def test_original_bundle_payload_changes_are_rejected_by_the_frozen_reader(packet, package_gate, name):
    root, provenance = packet
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    provenance["file_sha256"][name] = digest(path.read_bytes())
    with pytest.raises(ValueError):
        verify(root, provenance, package_gate)


@pytest.mark.parametrize("change", ["presentation-metadata", "source-segmentation", "counter", "error-output"])
def test_genuine_notebook_requires_exact_raw_bytes_even_after_remanifesting(packet, package_gate, change):
    root, provenance = packet
    path = root / NOTEBOOK
    book = json.loads(path.read_bytes())
    cell = next(cell for cell in book["cells"] if cell["cell_type"] == "code")
    if change == "presentation-metadata":
        book.setdefault("metadata", {})["new_presentation"] = True
    elif change == "source-segmentation":
        text = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cell["source"] = [text[:11], text[11:]]
    elif change == "counter":
        cell["execution_count"] = None
    else:
        cell["outputs"].append({"output_type": "error", "ename": "Error", "evalue": "failed", "traceback": []})
    path.write_text(json.dumps(book))
    provenance["file_sha256"][NOTEBOOK] = digest(path.read_bytes())
    gate = SimpleNamespace(verify_public_packet=lambda *args, **kwargs: pytest.fail("Changed notebook reached saved dispatch"))
    with pytest.raises(ValueError):
        verify(root, provenance, package_gate, gate=gate)


@pytest.mark.parametrize("name", [BUNDLE + "/unregistered.json", BUNDLE + "/private/log.txt",
                                 PUBLIC + "/extra.json", PUBLIC + "/.cache/private.json"])
def test_closed_artifact_directories_reject_unregistered_members(packet, package_gate, name):
    root, provenance = packet
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unregistered bytes\n")
    with pytest.raises((OSError, ValueError)):
        verify(root, provenance, package_gate)


@pytest.mark.parametrize("name", [HELPER, SOURCE_CONFIG, ARCHIVE, NOTEBOOK])
def test_linked_reader_config_archive_or_notebook_is_rejected(packet, package_gate, name):
    root, provenance = packet
    path = root / name
    retained = root / "retained-original"
    path.rename(retained)
    path.symlink_to(retained)
    with pytest.raises((OSError, ValueError)):
        verify(root, provenance, package_gate)


def test_second_archive_snapshot_is_pinned_before_zip_decoding(packet, package_gate, monkeypatch):
    root, provenance = packet
    decoder_calls = []
    def decoder(*args, **kwargs):
        decoder_calls.append((args, kwargs))
        pytest.fail("Changed archive snapshot reached ZIP decoding")
    monkeypatch.setattr(zipfile, "ZipFile", decoder)
    gate = package_gate._load_dqi_degree_two_saved_gate(root, provenance)
    original_read = gate._read
    archive_reopens = []
    def read(path, cap, check):
        if Path(path) == root / ARCHIVE:
            archive_reopens.append(Path(path))
            return (root / ARCHIVE).read_bytes() + b"changed second snapshot"
        return original_read(path, cap, check)
    monkeypatch.setattr(gate, "_read", read)
    with pytest.raises(ValueError):
        verify(root, provenance, package_gate, gate=gate)
    assert len(archive_reopens) == 1 and decoder_calls == []


@pytest.fixture
def tiny_export_root(tmp_path, payload):
    root = (tmp_path / "tiny-export").resolve()
    required = {
        "citations.bib": b"", "LICENSE": payload["LICENSE"],
        "requirements-lock.txt": payload["requirements-lock.txt"], SOURCE_CONFIG: payload[SOURCE_CONFIG],
        ARCHIVE: payload[ARCHIVE], "assets/generated/optimization/sampled/optimization.py": b"# inert fixture\n",
        "assets/generated/optimization/sampled-runs.json": b"{}\n",
        "assets/generated/grover/circuit.qasm": b"inert qasm fixture\n",
        "assets/generated/resources/grover-surface-code.json": b"{}\n",
        "assets/generated/resources/grover-surface-code.csv": b"field\n",
    }
    required.update({"binder/" + name: b"inert fixture\n"
                     for name in ("assemble_code.py", "postBuild", "environment.yml", "README.md")})
    for name, raw in required.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root


def configure_tiny_export(export, monkeypatch):
    for name in ("PUBLIC_SCRIPTS", "PUBLIC_BOOTSTRAP_FILES", "PUBLIC_TAIL_RESOURCE_POLICY_FILES",
                 "PUBLIC_LOCAL_SPIN_SOURCE_FILES", "FORMAL_FILES"):
        monkeypatch.setattr(export, name, ())
    monkeypatch.setattr(export, "PUBLIC_ARTIFACT_TREES", ("optimization",))
    monkeypatch.setattr(export, "git_revision", lambda root: "a" * 40)


def test_export_publishes_original_config_role_without_overwriting_global_bootstrap(tiny_export_root, payload, monkeypatch):
    export = load_script("export_code")
    configure_tiny_export(export, monkeypatch)
    assert export.DQI_DEGREE_TWO_SOURCE_CONFIG == SOURCE_CONFIG
    files = export.package_files(tiny_export_root)
    assert files[SOURCE_CONFIG] == payload[SOURCE_CONFIG] and digest(files[SOURCE_CONFIG]) == CONFIG_PIN
    assert files["pyproject.toml"] == BOOTSTRAP and digest(files["pyproject.toml"]) == BOOTSTRAP_PIN
    provenance = json.loads(files["PROVENANCE.json"])
    assert provenance["file_sha256"][SOURCE_CONFIG] == CONFIG_PIN
    assert provenance["file_sha256"]["pyproject.toml"] == BOOTSTRAP_PIN


def test_export_can_derive_only_the_original_config_from_pinned_archive(tiny_export_root, payload, monkeypatch):
    export = load_script("export_code")
    configure_tiny_export(export, monkeypatch)
    (tiny_export_root / SOURCE_CONFIG).unlink()
    files = export.package_files(tiny_export_root)
    assert files[SOURCE_CONFIG] == payload[SOURCE_CONFIG]
    assert files["pyproject.toml"] == BOOTSTRAP


def test_unrelated_reduced_export_keeps_bootstrap_and_skips_dqi_role(tiny_export_root, monkeypatch):
    export = load_script("export_code")
    configure_tiny_export(export, monkeypatch)
    monkeypatch.setattr(export, "PUBLIC_ARTIFACT_TREES", ())
    monkeypatch.setattr(export, "_dqi_degree_two_source_config", lambda *args, **kwargs: pytest.fail("Unrelated export consulted DQI configuration"))
    files = export.package_files(tiny_export_root)
    assert ARCHIVE not in files and SOURCE_CONFIG not in files
    assert files["pyproject.toml"] == BOOTSTRAP
    assert SOURCE_CONFIG not in json.loads(files["PROVENANCE.json"])["file_sha256"]


def test_export_rejects_an_unbound_config_role_instead_of_replacing_global_bootstrap(tiny_export_root, monkeypatch):
    export = load_script("export_code")
    configure_tiny_export(export, monkeypatch)
    path = tiny_export_root / SOURCE_CONFIG
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        export.package_files(tiny_export_root)


def test_closed_exports_preserve_only_original_bundle_and_public_projection():
    export = load_script("export_code")
    assert export.PUBLIC_DQI_DEGREE_TWO_FILES == {BUNDLE + "/" + name for name in BUNDLE_NAMES}
    assert export.PUBLIC_DQI_DEGREE_TWO_PUBLIC_FILES == {PUBLIC + "/" + name for name in META_NAMES}
    assert export.PUBLIC_DQI_DEGREE_TWO_FILES | export.PUBLIC_DQI_DEGREE_TWO_PUBLIC_FILES == PUBLIC_FILES
    assert ARCHIVE in export.PUBLIC_ARCHIVES
    assert all(export.include_generated_file(name) for name in PUBLIC_FILES)
    assert {"generate_optimization_dqi_degree_two.py", "validate_optimization_dqi_degree_two.py",
            "verify_dqi_degree_two_saved.py", "verify_dqi_degree_two_public_saved.py"} <= set(export.PUBLIC_SCRIPTS)


@pytest.mark.parametrize("name", [BUNDLE + "/extra.json", BUNDLE + "/new-case/operations.ndjson",
                                 PUBLIC + "/unreviewed.zip", PUBLIC + "/private/log.txt",
                                 BUNDLE + "/./bundle.json", PUBLIC + "/../dqi-degree-two-public/README.md"])
def test_export_rejects_unregistered_degree_two_members_and_aliases(name):
    export = load_script("export_code")
    assert not export.include_generated_file(name)


def test_package_degree_two_dispatch_follows_structured_identity_and_complete_admission(tmp_path):
    path = ROOT / "scripts/verify_public_package.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "verify_package")
    root = tmp_path.resolve()
    identity = {"schema": "synthetic admitted identity"}
    observed = []
    class StopAtResource(Exception):
        pass
    ordinary = SimpleNamespace(
        RECORD_DIRECTORY="records", BUNDLE_RELATIVE=BUNDLE,
        verify_energy_record_identity=lambda *args, **kwargs: None,
        verify_dynamics_record_identity=lambda *args, **kwargs: None,
        verify_public_word_transport=lambda *args, **kwargs: None,
        verify_bundle=lambda *args, **kwargs: None, verify_download=lambda *args, **kwargs: None,
        verify_saved=lambda *args, **kwargs: None,
    )
    def degree_two(package, provenance):
        assert package == root and provenance is identity
        observed.append("degree-two-loader")
        return ordinary
    def degree_two_identity(gate, artifact_root, **kwargs):
        assert gate is ordinary and artifact_root == root and kwargs["provenance"] is identity
        observed.append("degree-two-identity")
    def resource(*args):
        observed.append("resource-loader")
        raise StopAtResource
    namespace = {"Path": Path, "checked_directory": lambda value: value,
                 "verify_manifest": lambda value: observed.append("manifest") or identity,
                 "verify_formal_record": lambda value: None,
                 "_load_dqi_degree_two_saved_gate": degree_two,
                 "_verify_dqi_degree_two_public_identity": degree_two_identity,
                 "_load_dqi_resource_saved_gate": resource,
                 "_verify_dqi_saved_identity": lambda *args, **kwargs: None,
                 "_verify_bath_saved_identity": lambda *args, **kwargs: None,
                 "_verify_dqi_structured_saved_identity": lambda *args, **kwargs: observed.append("structured-identity"),
                 "DQI_STRUCTURED_BUNDLE_DIRECTORY": "unused-structured"}
    for name in ("_load_energy_record_gate", "_load_dynamics_record_gate", "_load_word_record_gate",
                 "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate", "_load_decay_record_gate",
                 "_load_correction_record_gate", "_load_tail_record_gate", "_load_dqi_saved_gate",
                 "_load_multiproduct_saved_gate", "_load_bayesian_saved_gate", "_load_robust_saved_gate",
                 "_load_dqi_structured_saved_gate"):
        namespace[name] = lambda *args, **kwargs: ordinary
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(StopAtResource):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest", "structured-identity", "degree-two-loader", "degree-two-identity", "resource-loader"]
    observed.clear()
    def reject(_):
        observed.append("manifest-rejected")
        raise ValueError("invalid complete manifest")
    namespace["verify_manifest"] = reject
    with pytest.raises(ValueError, match="invalid complete manifest"):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest-rejected"]


@pytest.fixture
def site_function(package_gate):
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "site_dqi_degree_two_saved_errors")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile,
                 "_read_bounded_public_member": package_gate._read_bounded_public_member,
                 "_load_dqi_degree_two_saved_gate": package_gate._load_dqi_degree_two_saved_gate,
                 "_verify_dqi_degree_two_public_identity": package_gate._verify_dqi_degree_two_public_identity}
    namespace.update({name: value for name, value in vars(package_gate).items() if name.startswith("DQI_DEGREE_TWO_")})
    export_tree = ast.parse((ROOT / "scripts/export_code.py").read_text())
    export_constants = [item for item in export_tree.body if isinstance(item, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id.startswith("PUBLIC_DQI_DEGREE_TWO_")
                                for target in item.targets)]
    exec(compile(ast.Module(body=export_constants, type_ignores=[]), "degree-two-export-constants", "exec"), namespace)
    selected = [item for item in tree.body if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id.startswith("DQI_DEGREE_TWO_") for target in item.targets)]
    exec(compile(ast.Module(body=[*selected, node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["site_dqi_degree_two_saved_errors"]


@pytest.fixture
def site_packet(tmp_path, payload):
    root = (tmp_path / "site").resolve()
    code = root / "code"
    code.mkdir(parents=True)
    retained, omitted = {}, {}
    hashes = {name: digest(raw) for name, raw in payload.items()}
    for name, raw in payload.items():
        if name.startswith("assets/"):
            path = root / name
            omitted[name] = {"site_path": name, "sha256": digest(raw)}
        else:
            path = code / name
            retained[name] = digest(raw)
            if name == NOTEBOOK:
                copy = root / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: pin for name, pin in hashes.items() if name.endswith(".py")}}
    raw = json.dumps(provenance, sort_keys=True).encode()
    (code / "PROVENANCE.json").write_bytes(raw)
    retained["PROVENANCE.json"] = digest(raw)
    return root, code, {"schema": "openquantum-source-preview-v1", "retained_sha256": retained,
                        "omitted_assets": omitted}


def test_site_uses_canonical_assets_raw_notebook_and_authenticated_source_config(site_packet, site_function, monkeypatch):
    root, code, preview = site_packet
    reject_science(monkeypatch)
    assert not (code / "assets").exists()
    assert site_function(root, preview) == []
    assert (code / "pyproject.toml").read_bytes() == BOOTSTRAP
    assert digest((code / SOURCE_CONFIG).read_bytes()) == CONFIG_PIN


@pytest.mark.parametrize("change", ["missing-preview", "wrong-schema", "provenance", "helper", "config",
                                    "manifest", "descriptor", "archive", "served-notebook", "code-notebook",
                                    "missing-omitted-asset", "aliased-asset", "wrong-omitted-hash"])
def test_site_rejects_changed_preview_packet_config_or_raw_notebook(site_packet, site_function, change):
    root, code, preview = site_packet
    if change == "missing-preview":
        preview = {}
    elif change == "wrong-schema":
        preview["schema"] = "wrong"
    elif change == "provenance":
        (code / "PROVENANCE.json").write_bytes((code / "PROVENANCE.json").read_bytes() + b"\n")
    elif change == "missing-omitted-asset":
        preview["omitted_assets"].pop(ARCHIVE)
    elif change == "aliased-asset":
        preview["omitted_assets"][ARCHIVE]["site_path"] = "code/" + ARCHIVE
    elif change == "wrong-omitted-hash":
        preview["omitted_assets"][ARCHIVE]["sha256"] = "0" * 64
    else:
        path = {"helper": code / HELPER, "config": code / SOURCE_CONFIG,
                "manifest": root / MANIFEST, "descriptor": root / DOWNLOAD, "archive": root / ARCHIVE,
                "served-notebook": root / NOTEBOOK, "code-notebook": code / NOTEBOOK}[change]
        path.write_bytes(path.read_bytes() + b"\n")
    errors = site_function(root, preview)
    assert len(errors) == 1 and "degree-two" in errors[0].lower()


def test_site_degree_two_call_stays_in_successful_preview_branch_and_order():
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = next(node for node in main.body if isinstance(node, ast.Try)
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "verify_preview"
                         for item in node.body for call in ast.walk(item)))
    names = [node.func.id for item in block.orelse for node in ast.walk(item)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert names.count("site_dqi_degree_two_saved_errors") == 1
    index = names.index("site_dqi_degree_two_saved_errors")
    assert names[index - 1] == "site_dqi_structured_saved_errors"
    assert names[index + 1] == "site_dqi_resource_saved_errors"
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "site_dqi_degree_two_saved_errors"
                   for item in block.body + [item for handler in block.handlers for item in handler.body]
                   for node in ast.walk(item))


def test_site_source_config_downloads_and_notebook_are_required_without_code_assets():
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    required = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "required_artifacts")
    literals = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    retained = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "DQI_DEGREE_TWO_RETAINED_PATHS"
                            for target in node.targets))
    assert SOURCE_FILES | {SOURCE_CONFIG, NOTEBOOK, "LICENSE", "requirements-lock.txt"} <= set(retained)
    assert any(isinstance(node, ast.Name) and node.id == "DQI_DEGREE_TWO_RETAINED_PATHS" for node in ast.walk(required))
    assert any(isinstance(node, ast.Name) and node.id == "PUBLIC_DQI_DEGREE_TWO_FILES" for node in ast.walk(required))
    assert not any(value.startswith("code/" + BUNDLE + "/") for value in literals)
