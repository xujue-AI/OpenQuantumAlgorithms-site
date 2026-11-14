"""DQI publication trust boundaries using data and synthetic helpers only.

Shared functions are selected from captured AST snapshots. No scientific
module, generator, quantum simulator, estimator or subprocess is imported or
run by these tests. The normal notebook is inspected as JSON, never executed.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import unquote, urlsplit
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/optimization/dqi"
HELPER = "scripts/verify_dqi_saved.py"
NOTEBOOK = "notebooks/optimization_dqi.ipynb"
PUBLIC_SCRIPTS = ("generate_optimization_dqi.py", "verify_dqi_saved.py")
PUBLIC_NAMES = frozenset({
    "LICENSE.txt", "bundle.json", "dqi-circuit.svg", "dqi-validated.json", "dqi.qasm",
    "generate_optimization_dqi.py", "optimization_dqi.ipynb", "optimization_dqi.py",
    "requirements-lock.txt", "verify_dqi_saved.py",
})
PUBLIC_FILES = frozenset(BUNDLE + "/" + name for name in PUBLIC_NAMES)
DEPENDENCIES = frozenset(BUNDLE + "/" + name for name in (
    "requirements-lock.txt", "LICENSE.txt", "optimization_dqi.ipynb"))
SOURCE_PINS = {
    "src/openquantum/optimization_dqi.py": "370088484830c23140f14181a3892194fa936e4477327536946423854fcc1f99",
    "scripts/generate_optimization_dqi.py": "5312969299af870925d766263eda13fc27f156b497b6fa40cf744bc3ff620cca",
    HELPER: "981d550cf0aac5ec8f1ceb6d8c8f8625fe00a5cdcba9646ed9d2cbb011c4242e",
}
BUNDLE_PIN = "ea14085d24d531df648f4bd0957288f753d23121d981d1b5aa3c97a926687e12"
NOTEBOOK_SOURCE_PIN = "bdd3db3d07a8f4218a4d32f53b1371196b52466de7d69ec856c36f4850052667"
CI_DQI = "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi"
CI_FINAL_PIN = "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af"
CI_PRIOR_PIN = "455c0b472be3ff3365f598be652aa3efe2ddab831f05ad9d87536690c42c2ee4"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def notebook_projection(raw):
    """Independent saved-source contract: order/type/text, without IDs/output."""
    notebook = json.loads(raw)
    cells = [{"cell_type": cell["cell_type"],
              "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
             for cell in notebook["cells"]]
    return sha(json.dumps(cells, sort_keys=True, separators=(",", ":")).encode())


class SourceSnapshot:
    """Select definitions from one captured script, without sibling imports."""
    def __init__(self, path):
        self.path = Path(path)
        self.raw = self.path.read_bytes()
        self.tree = ast.parse(self.raw, filename=str(self.path))

    def module(self, name, **extra):
        selected = [node for node in self.tree.body if isinstance(
            node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef))]
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        tree = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
        module = ModuleType(name)
        module.__file__ = str(self.path)
        os_proxy = SimpleNamespace(
            open=os.open, close=os.close, fdopen=os.fdopen, fstat=os.fstat,
            O_RDONLY=os.O_RDONLY, O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0),
            O_NONBLOCK=getattr(os, "O_NONBLOCK", 0), environ={},
        )
        util_proxy = SimpleNamespace(spec_from_file_location=importlib.util.spec_from_file_location,
                                     module_from_spec=importlib.util.module_from_spec)
        module.__dict__.update(
            Path=Path, PurePosixPath=PurePosixPath, hashlib=SimpleNamespace(sha256=hashlib.sha256),
            json=json, os=os_proxy, stat=stat, math=math, HTMLParser=HTMLParser,
            importlib=SimpleNamespace(util=util_proxy), sys=SimpleNamespace(argv=[], executable=sys.executable),
            zipfile=zipfile, unquote=unquote, urlsplit=urlsplit,
            subprocess=SimpleNamespace(run=lambda *_, **__: pytest.fail("Registration reached a subprocess")),
            LINDBLAD_PORT_FILES=frozenset(), git_revision=lambda _: "a" * 40,
        )
        module.__dict__.update(extra)
        exec(compile(tree, str(self.path), "exec"), module.__dict__)
        return module


def write_package(root, files):
    root.mkdir(parents=True)
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    hashes = {name: sha(raw) for name, raw in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def saved_result():
    return {"verification": {"status": "saved_component_verified", "bundle_manifest_sha256": BUNDLE_PIN,
                             "notebook_source_sha256": NOTEBOOK_SOURCE_PIN, "checked_files": 9,
                             "new_quantum_execution": False}}


def synthetic_helper():
    return ("import hashlib,json\nTOKEN = 'authenticated synthetic DQI helper'\nCALLS = []\n"
            "def notebook_source_sha256(raw):\n"
            "    notebook = json.loads(raw)\n"
            "    cells = [{'cell_type': c['cell_type'], 'source': ''.join(c['source']) if isinstance(c['source'],list) else c['source']} for c in notebook['cells']]\n"
            "    return hashlib.sha256(json.dumps(cells,sort_keys=True,separators=(',',':')).encode()).hexdigest()\n"
            "def verify_saved_bundle(directory):\n"
            "    CALLS.append(str(directory))\n"
            "    return " + repr(saved_result()) + "\n").encode()


def synthetic_files(*, normal=None):
    normal = (ROOT / NOTEBOOK).read_bytes() if normal is None else normal
    sources = {"src/openquantum/optimization_dqi.py": b"# synthetic module; never imported\n",
               "scripts/generate_optimization_dqi.py": b"# synthetic generator; never imported\n",
               HELPER: synthetic_helper()}
    files = {**sources, NOTEBOOK: normal}
    # Curated bundle bytes are inert fixture data, not imported sources. This
    # keeps the shared wrapper's immutable manifest pin meaningful while the
    # tiny top-level helper tests loader/dispatch behavior without science.
    for name in PUBLIC_NAMES:
        files[BUNDLE + "/" + name] = (ROOT / BUNDLE / name).read_bytes()
    return files


@pytest.fixture
def checked():
    return SourceSnapshot(ROOT / "scripts/verify_public_package.py").module("dqi_registration_package")


@pytest.fixture
def exporter():
    return SourceSnapshot(ROOT / "scripts/export_code.py").module("dqi_registration_export")


@pytest.fixture
def package(tmp_path, checked, monkeypatch):
    root = (tmp_path / "package").resolve()
    provenance = write_package(root, synthetic_files())
    # This fixture supplies an explicit tiny trust root for synthetic code.
    # Separate tests assert every immutable production pin and flat relation.
    pins = {name: provenance["file_sha256"][name] for name in SOURCE_PINS}
    monkeypatch.setattr(checked, "DQI_SOURCE_HASHES", pins)
    return root, provenance


@pytest.fixture
def checker(checked, exporter):
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site tooling is outside the complete public examples package")
    values = {name: value for name, value in vars(exporter).items() if name.startswith("PUBLIC_")}
    values.update(ALGORITHMS=[], REDIRECTS={}, _site_redirects=SimpleNamespace(REDIRECT_FRAGMENTS={}),
                  checked_directory=checked.checked_directory,
                  is_runtime_artifact=lambda _: False, verify_preview=lambda _: {},
                  _load_dqi_saved_gate=checked._load_dqi_saved_gate,
                  _verify_dqi_saved_identity=checked._verify_dqi_saved_identity,
                  _read_bounded_public_member=checked._read_bounded_public_member)
    for tag in ("N2", "N3", "N4", "PORT", "QDRIFT"):
        values[tag + "_FILES"] = frozenset()
        values[tag + "_PATH"] = "unused-" + tag.lower()
    values.update(PORT_SOURCE_FILES={}, OPT7_PATH="unused-compiled",
                  ENERGY_CORE_FILES=frozenset(), ENERGY_MANIFEST_NAME="unused.json",
                  ENERGY_RECORD_DIRECTORY="unused-energy", ENERGY_SOURCE_FILES=(), ENERGY_VERIFIER_FILES=(),
                  DYNAMICS_CORE_FILES=frozenset(), DYNAMICS_MANIFEST_NAME="unused.json",
                  DYNAMICS_RECORD_DIRECTORY="unused-dynamics", DYNAMICS_SOURCE_FILES=(), DYNAMICS_VERIFIER_FILES=())
    for name in ("verify_n2_bundle", "verify_n3_bundle", "verify_n4_bundle", "verify_port_bundle",
                 "verify_qdrift_bundle", "verify_hadamard_record", "verify_opt7_bundle"):
        values[name] = lambda *_, **__: {}
    for name in ("site_energy_record_errors", "site_dynamics_record_errors"):
        values[name] = lambda *_, **__: []
    return SourceSnapshot(path).module("dqi_registration_site", **values)


@pytest.fixture
def site(tmp_path, checked, monkeypatch):
    root = (tmp_path / "site").resolve()
    files = synthetic_files()
    retained = {name: raw for name, raw in files.items() if not name.startswith("assets/")}
    provenance = write_package(root / "code", retained)
    for name, raw in files.items():
        if name.startswith("assets/") or name == NOTEBOOK:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    pins = {name: provenance["file_sha256"][name] for name in SOURCE_PINS}
    monkeypatch.setattr(checked, "DQI_SOURCE_HASHES", pins)
    identity = {"schema": "openquantum-source-preview-v1", "retained_sha256": {
        "PROVENANCE.json": sha((root / "code/PROVENANCE.json").read_bytes())}}
    return root, identity, provenance


def test_promoted_sources_and_flat_copies_keep_exact_three_pins(checked):
    assert checked.DQI_SOURCE_HASHES == SOURCE_PINS
    for current, flat in (("src/openquantum/optimization_dqi.py", "optimization_dqi.py"),
                          ("scripts/generate_optimization_dqi.py", "generate_optimization_dqi.py"),
                          (HELPER, "verify_dqi_saved.py")):
        assert sha((ROOT / current).read_bytes()) == SOURCE_PINS[current]
        assert (ROOT / current).read_bytes() == (ROOT / BUNDLE / flat).read_bytes()
    assert sha((ROOT / BUNDLE / "bundle.json").read_bytes()) == BUNDLE_PIN
    assert {path.name for path in (ROOT / BUNDLE).iterdir()} == PUBLIC_NAMES


def test_export_uses_closed_ten_files_three_dependencies_and_two_scripts(exporter):
    assert exporter.PUBLIC_DQI_FILES == PUBLIC_FILES
    assert {name for name in exporter.PUBLIC_DEPENDENCY_FILES if name.startswith(BUNDLE + "/")} == DEPENDENCIES
    assert tuple(name for name in exporter.PUBLIC_SCRIPTS if "dqi" in name and "structured" not in name and name != "generate_optimization_dqi_resources.py") == PUBLIC_SCRIPTS
    assert all(exporter.PUBLIC_SCRIPTS.count(name) == 1 for name in PUBLIC_SCRIPTS)
    assert ".ipynb" not in exporter.ARTIFACT_SUFFIXES and ".txt" not in exporter.ARTIFACT_SUFFIXES
    assert all(exporter.include_generated_file(Path(name)) for name in PUBLIC_FILES)


@pytest.mark.parametrize("name", ["unreviewed.json", "history.py", "template.ipynb", "private.txt",
                                  "profile.json.gz", "cache/record.json", "nested/bundle.json"])
def test_export_rejects_extra_or_nested_dqi_members(exporter, name):
    assert not exporter.include_generated_file(Path(BUNDLE) / name)


def test_other_reviewed_closed_exports_remain_selected(exporter):
    for group in (exporter.PUBLIC_OPTIMIZATION_WORD_FILES, exporter.PUBLIC_SUZUKI_FILES,
                  exporter.PUBLIC_GIBBS_CHANNEL_FILES):
        assert group
        assert all(exporter.include_generated_file(Path(name)) for name in group)


def test_normal_notebook_is_a_reusable_source_projection():
    normal = (ROOT / NOTEBOOK).read_bytes()
    flat = (ROOT / BUNDLE / "optimization_dqi.ipynb").read_bytes()
    assert notebook_projection(normal) == notebook_projection(flat) == NOTEBOOK_SOURCE_PIN
    document = json.loads(normal)
    assert document["metadata"]["kernelspec"]["name"] == "python3"
    assert sum(cell["cell_type"] == "code" for cell in document["cells"]) == 4
    changed = copy.deepcopy(document)
    for cell in changed["cells"]:
        cell["id"] = "fresh-cell-id"
        cell["metadata"] = {"fresh": True}
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    changed["metadata"] = {"fresh": True}
    assert notebook_projection(json.dumps(changed).encode()) == NOTEBOOK_SOURCE_PIN
    changed["cells"][0]["source"] = ["a different reusable source\n"]
    assert notebook_projection(json.dumps(changed).encode()) != NOTEBOOK_SOURCE_PIN


@pytest.mark.parametrize("change", ["schema", "manifest-type", "missing-helper", "helper-digest"])
def test_loader_rejects_unbound_complete_provenance_before_compile(package, checked, monkeypatch, change):
    root, provenance = package
    broken = copy.deepcopy(provenance)
    if change == "schema": broken["schema"] = "wrong"
    elif change == "manifest-type": broken["file_sha256"] = []
    elif change == "missing-helper": broken["file_sha256"].pop(HELPER)
    else: broken["file_sha256"][HELPER] = "0" * 64
    monkeypatch.setattr(builtins, "compile", lambda *_, **__: pytest.fail("Unbound DQI helper reached compile"))
    with pytest.raises(ValueError): checked._load_dqi_saved_gate(root, broken)


def test_self_consistent_but_unpromoted_helper_cannot_execute(tmp_path, checked, monkeypatch):
    root = (tmp_path / "unpromoted").resolve()
    marker = tmp_path / "must-not-exist"
    raw = f"from pathlib import Path\nPath({str(marker)!r}).write_text('unpromoted helper')\n".encode()
    files = synthetic_files()
    for name in SOURCE_PINS:
        if name != HELPER:
            files[name] = (ROOT / name).read_bytes()
    files[HELPER] = raw
    provenance = write_package(root, files)
    assert all(provenance["file_sha256"][name] == value for name, value in SOURCE_PINS.items() if name != HELPER)
    assert checked.verify_manifest(root) == provenance
    monkeypatch.setattr(builtins, "compile", lambda *_, **__: pytest.fail("Unpromoted helper reached compile"))
    with pytest.raises(ValueError, match="promoted"): checked._load_dqi_saved_gate(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("source", tuple(SOURCE_PINS))
def test_each_source_pin_rejects_before_saved_dispatch(package, checked, source):
    root, provenance = package
    broken = copy.deepcopy(provenance)
    broken["file_sha256"][source] = "0" * 64
    gate = SimpleNamespace(verify_saved_bundle=lambda *_: pytest.fail("Unpinned source dispatched saved data"),
                           notebook_source_sha256=lambda *_: pytest.fail("Unpinned source reached notebook projection"))
    with pytest.raises(ValueError, match="promoted"):
        checked._verify_dqi_saved_identity(gate, root / BUNDLE, broken, notebook_roots=(root,))


def test_loader_compiles_exactly_one_captured_source_snapshot(package, checked, monkeypatch):
    root, provenance = package
    raw = (root / HELPER).read_bytes()
    real_read, real_hash = checked._read_bounded_public_member, checked.hashlib.sha256
    reads = []
    def read(path, name, maximum):
        reads.append((path, name, maximum))
        return real_read(path, name, maximum)
    def replace_after_hash(data=b"", *args, **kwargs):
        result = real_hash(data, *args, **kwargs)
        if data == raw:
            (root / HELPER).write_bytes(b"raise AssertionError('later helper executed')\n")
        return result
    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    monkeypatch.setattr(checked.hashlib, "sha256", replace_after_hash)
    gate = checked._load_dqi_saved_gate(root, provenance)
    assert gate.TOKEN == "authenticated synthetic DQI helper"
    assert reads == [(root, HELPER, 64 * 1024)]


def test_loader_ignores_shadow_modules_and_bytecode(package, checked, monkeypatch):
    root, provenance = package
    shadow = ModuleType("verify_dqi_saved")
    shadow.TOKEN = "untrusted process registry"
    for name in ("verify_dqi_saved", "_openquantum_verified_dqi_saved_gate"):
        monkeypatch.setitem(sys.modules, name, shadow)
    cache = root / "scripts/__pycache__"
    cache.mkdir()
    (cache / "verify_dqi_saved.cpython-312.pyc").write_bytes(b"untrusted bytecode")
    before = list(sys.path)
    gate = checked._load_dqi_saved_gate(root, provenance)
    assert gate is not shadow and gate.TOKEN == "authenticated synthetic DQI helper"
    assert all(module is not gate for module in list(sys.modules.values()))
    assert sys.modules["verify_dqi_saved"] is shadow and sys.path == before


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "linked-parent"])
def test_loader_rejects_linked_members_before_compile(package, checked, monkeypatch, tmp_path, kind):
    root, provenance = package
    source = root / HELPER
    if kind == "linked-parent":
        target = tmp_path / "original-scripts"
        source.parent.rename(target)
        source.parent.symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "original-helper"
        source.rename(target)
        source.symlink_to(target) if kind == "symlink" else os.link(target, source)
    monkeypatch.setattr(builtins, "compile", lambda *_, **__: pytest.fail("Linked DQI helper reached compile"))
    with pytest.raises(ValueError): checked._load_dqi_saved_gate(root, provenance)


def test_loader_size_cap_rejects_before_open(package, checked, monkeypatch):
    root, provenance = package
    (root / HELPER).write_bytes(b"x" * (64 * 1024 + 1))
    monkeypatch.setattr(checked.os, "open", lambda *_, **__: pytest.fail("Oversized DQI helper was opened"))
    with pytest.raises(ValueError, match="bounded"): checked._load_dqi_saved_gate(root, provenance)


@pytest.mark.parametrize("field,value", [
    ("status", "pending"), ("bundle_manifest_sha256", "0" * 64),
    ("notebook_source_sha256", "0" * 64), ("checked_files", 8),
    ("new_quantum_execution", True), ("new_quantum_execution", 0),
])
def test_returned_saved_event_pin_and_scope_are_independent(package, checked, field, value):
    root, provenance = package
    result = saved_result()
    result["verification"][field] = value
    calls = []
    gate = SimpleNamespace(notebook_source_sha256=notebook_projection,
                           verify_saved_bundle=lambda path: calls.append(path) or result)
    with pytest.raises(ValueError, match="saved event"):
        checked._verify_dqi_saved_identity(gate, root / BUNDLE, provenance, notebook_roots=(root,))
    assert calls == [root / BUNDLE]


@pytest.mark.parametrize("roots", [None, [], (), (Path("unused"),) * 3])
def test_normal_notebook_root_shape_fails_before_dispatch(package, checked, roots):
    root, provenance = package
    gate = SimpleNamespace(verify_saved_bundle=lambda *_: pytest.fail("Malformed notebook roots dispatched saved data"))
    with pytest.raises(ValueError, match="roots"):
        checked._verify_dqi_saved_identity(gate, root / BUNDLE, provenance, notebook_roots=roots)


@pytest.mark.parametrize("change", ["source", "order", "type"])
def test_consistently_remanifested_normal_notebook_source_is_rejected(package, checked, change):
    root, provenance = package
    document = json.loads((root / NOTEBOOK).read_bytes())
    if change == "source": document["cells"][0]["source"] = ["changed reusable source\n"]
    elif change == "order": document["cells"][0], document["cells"][1] = document["cells"][1], document["cells"][0]
    else: document["cells"][0]["cell_type"] = "raw"
    raw = json.dumps(document).encode()
    (root / NOTEBOOK).write_bytes(raw)
    provenance["file_sha256"][NOTEBOOK] = sha(raw)
    gate = SimpleNamespace(notebook_source_sha256=notebook_projection,
                           verify_saved_bundle=lambda *_: pytest.fail("Changed normal source dispatched saved data"))
    with pytest.raises(ValueError, match="notebook source"):
        checked._verify_dqi_saved_identity(gate, root / BUNDLE, provenance, notebook_roots=(root,))


def test_fresh_notebook_outputs_metadata_and_ids_keep_saved_identity(package, checked):
    root, provenance = package
    document = json.loads((root / NOTEBOOK).read_bytes())
    for cell in document["cells"]:
        cell["id"] = "new-id"
        cell["metadata"] = {"new": True}
        if cell["cell_type"] == "code": cell.update(outputs=[], execution_count=None)
    raw = json.dumps(document).encode()
    (root / NOTEBOOK).write_bytes(raw)
    provenance["file_sha256"][NOTEBOOK] = sha(raw)
    calls = []
    gate = SimpleNamespace(notebook_source_sha256=notebook_projection,
                           verify_saved_bundle=lambda path: calls.append(path) or saved_result())
    assert checked._verify_dqi_saved_identity(gate, root / BUNDLE, provenance, notebook_roots=(root,)) == saved_result()
    assert calls == [root / BUNDLE]


@pytest.mark.parametrize("extra", ["private.json", "templates", "nested/record.json"])
def test_saved_registration_rejects_extra_flat_or_nested_bundle_members(package, checked, extra):
    root, provenance = package
    path = root / BUNDLE / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    if extra == "templates": path.mkdir()
    else: path.write_text("unreviewed")
    gate = SimpleNamespace(notebook_source_sha256=notebook_projection,
                           verify_saved_bundle=lambda *_: pytest.fail("Nonclosed DQI directory dispatched saved data"))
    with pytest.raises(ValueError):
        checked._verify_dqi_saved_identity(gate, root / BUNDLE, provenance, notebook_roots=(root,))


def test_site_authenticates_one_provenance_snapshot_and_exact_root_map(site, checker, checked, monkeypatch):
    root, identity, _ = site
    assert not (root / "code/assets").exists()
    real_read, real_loader = checked._read_bounded_public_member, checked._load_dqi_saved_gate
    reads, loaded = [], []
    def read(directory, name, maximum):
        raw = real_read(directory, name, maximum)
        reads.append((directory, name, maximum))
        if name == "PROVENANCE.json": (directory / name).write_text("{}")
        return raw
    def load(directory, provenance):
        gate = real_loader(directory, provenance)
        loaded.append(gate)
        return gate
    monkeypatch.setattr(checker, "_read_bounded_public_member", read)
    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    monkeypatch.setattr(checker, "_load_dqi_saved_gate", load)
    assert checker.site_dqi_saved_errors(root, identity) == []
    assert reads == [(root / "code", "PROVENANCE.json", 16 * 1024 * 1024),
                     (root / "code", HELPER, 64 * 1024),
                     (root / BUNDLE, "bundle.json", 64 * 1024),
                     (root / "code", NOTEBOOK, 1024 * 1024), (root, NOTEBOOK, 1024 * 1024)]
    assert loaded[0].CALLS == [str(root / BUNDLE)]
    assert (root / "code/PROVENANCE.json").read_text() == "{}"


@pytest.mark.parametrize("change", ["missing", "schema", "retention-type", "changed-provenance"])
def test_site_preview_provenance_failure_precedes_helper_loading(site, checker, monkeypatch, change):
    root, identity, _ = site
    if change == "missing": identity = {}
    elif change == "schema": identity["schema"] = "wrong"
    elif change == "retention-type": identity["retained_sha256"] = []
    else: (root / "code/PROVENANCE.json").write_text("{}")
    monkeypatch.setattr(checker, "_load_dqi_saved_gate", lambda *_: pytest.fail("Unverified preview reached DQI helper"))
    assert checker.site_dqi_saved_errors(root, identity)


@pytest.mark.parametrize("where", ["code", "served"])
def test_site_binds_retained_and_served_normal_notebook_sources(site, checker, where):
    root, identity, provenance = site
    directory = root / "code" if where == "code" else root
    document = json.loads((directory / NOTEBOOK).read_bytes())
    document["cells"][0]["source"] = ["changed normal source\n"]
    raw = json.dumps(document).encode()
    (directory / NOTEBOOK).write_bytes(raw)
    if where == "code":
        provenance["file_sha256"][NOTEBOOK] = sha(raw)
        (root / "code/PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
        identity["retained_sha256"]["PROVENANCE.json"] = sha((root / "code/PROVENANCE.json").read_bytes())
    assert checker.site_dqi_saved_errors(root, identity)


def configure_site_main(checker, root, monkeypatch):
    (root / "index.html").write_text("<html><body>Pure DQI registration fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda *_: [])
    # Bath registration has its own tests; never invoke it in DQI fixtures.
    monkeypatch.setattr(checker, "site_bath_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_multiproduct_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_bayesian_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_robust_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_qpe_feedback_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_dqi_structured_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_dqi_resource_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_local_spin_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_local_spin_reproduction_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_measured_and_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "MEASURED_AND_FIGURE_LABELS", frozenset())
    for name in vars(checker):
        if name.endswith("_EMBED_LABELS") or name == "GIBBS_CHANNEL_ARTICLE_LABELS":
            monkeypatch.setattr(checker, name, frozenset())
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])


def test_failed_complete_preview_never_dispatches_any_saved_dqi_gate(site, checker, monkeypatch):
    root, _, _ = site
    configure_site_main(checker, root, monkeypatch)
    def rejected(*_): raise ValueError("complete archive rejected")
    monkeypatch.setattr(checker, "verify_preview", rejected)
    for name in ("site_word_record_errors", "site_suzuki_record_errors", "site_gibbs_channel_record_errors",
                 "site_tail_record_errors", "site_dqi_saved_errors"):
        monkeypatch.setattr(checker, name, lambda *_: pytest.fail("Failed complete preview dispatched saved gate"))
    with pytest.raises(SystemExit, match="complete archive rejected"): checker.main()


def test_site_late_dqi_dispatch_preserves_prior_saved_chain(site, checker, monkeypatch):
    root, identity, _ = site
    configure_site_main(checker, root, monkeypatch)
    order = []
    monkeypatch.setattr(checker, "verify_preview", lambda path: order.append("preview") or identity)
    for label in ("word", "suzuki", "gibbs_channel", "tail"):
        def prior(path, retained, stage=label):
            assert path == root and retained is identity
            order.append(stage)
            return []
        monkeypatch.setattr(checker, "site_" + label + "_record_errors", prior)
    actual = checker.site_dqi_saved_errors
    def dqi(path, retained):
        assert order == ["preview", "word", "suzuki", "gibbs_channel", "tail"]
        order.append("dqi")
        return actual(path, retained)
    monkeypatch.setattr(checker, "site_dqi_saved_errors", dqi)
    checker.main()
    assert order == ["preview", "word", "suzuki", "gibbs_channel", "tail", "dqi"]


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "wrong-provenance"])
def test_complete_package_failure_precedes_every_saved_gate(package, checked, monkeypatch, change):
    root, _ = package
    if change == "changed": (root / HELPER).write_bytes(b"raise AssertionError('unbound source')\n")
    elif change == "missing": (root / HELPER).unlink()
    elif change == "unrecorded": (root / "extra.py").write_text("raise AssertionError('unrecorded source')\n")
    else: (root / "PROVENANCE.json").write_text("{}")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Failed complete package authentication dispatched a saved prerequisite")
    for name in ("verify_formal_record", "_load_energy_record_gate", "_load_dynamics_record_gate",
                 "_load_word_record_gate", "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate",
                 "_load_tail_record_gate", "_load_dqi_saved_gate", "_verify_bath_saved_identity", "_load_local_spin_saved_gate"):
        monkeypatch.setattr(checked, name, forbidden)
    with pytest.raises(ValueError): checked.verify_package(root, run_tests=False)
    assert calls == []


def test_package_late_dqi_dispatch_preserves_chain_and_exact_normal_root(package, checked, monkeypatch):
    root, provenance = package
    order, loaded = [], []
    real_manifest, real_dqi = checked.verify_manifest, checked._load_dqi_saved_gate
    def manifest(path):
        assert path == root
        authenticated = real_manifest(path)
        assert authenticated == provenance
        order.append("manifest")
        return authenticated
    def identity_loader(label, method):
        def load(path, authenticated):
            assert path == root and authenticated == provenance
            order.append(label + "-load")
            def verify(directory, source):
                assert directory == root / ("unused-" + label) and source == root
                order.append(label + "-check")
            return SimpleNamespace(RECORD_DIRECTORY="unused-" + label, **{method: verify})
        return load
    def word(path, authenticated):
        assert path == root and authenticated == provenance
        order.append("word-load")
        def verify(directory, *, replay):
            assert directory == root and replay is True
            order.append("word-check")
        return SimpleNamespace(verify_public_word_transport=verify)
    def bundle_loader(label):
        def load(path, authenticated):
            assert path == root and authenticated == provenance
            order.append(label + "-load")
            def verify(directory, *, source_root):
                assert directory == root / ("unused-" + label) and source_root == root
                order.append(label + "-check")
            return SimpleNamespace(BUNDLE_RELATIVE="unused-" + label, verify_bundle=verify)
        return load
    def tail(path, authenticated):
        assert path == root and authenticated == provenance
        order.append("tail-load")
        def verify(directory, source, *, kind, expected_sha256, replay):
            pins = {"instrument": "0333814a823ebece83910251328b1145fc84e3cfbdec68051af92e78280ba220",
                    "complete": "86af1d97bad4ac1f4d8c5191d3e546f69ca82a5121bd8de7d7307081c70dc0b1"}
            paths = {"instrument": "assets/generated/instrument-resources",
                     "complete": "assets/generated/simulation/complete-resources"}
            assert directory == root / paths[kind] and source == root
            assert expected_sha256 == pins[kind] and replay is False
            order.append("tail-" + kind)
        return SimpleNamespace(verify_bundle=verify)
    prerequisites = ["manifest", "formal", "energy-load", "energy-check", "dynamics-load", "dynamics-check",
                     "word-load", "word-check", "suzuki-load", "suzuki-check", "gibbs-load", "gibbs-check",
                     "tail-load", "tail-instrument", "tail-complete"]
    def dqi(path, authenticated):
        assert path == root and authenticated == provenance and order == prerequisites
        order.append("dqi-load")
        gate = real_dqi(path, authenticated)
        loaded.append(gate)
        actual = gate.verify_saved_bundle
        def verify(directory):
            assert directory == root / BUNDLE
            order.append("dqi-check")
            return actual(directory)
        gate.verify_saved_bundle = verify
        return gate
    class StopAfterDqi(RuntimeError):
        pass
    expected_provenance = provenance
    def later_bath(path, *, provenance, notebook_roots):
        assert path == root and provenance == expected_provenance and notebook_roots == (root,)
        assert order == prerequisites + ["dqi-load", "dqi-check"]
        raise StopAfterDqi("DQI checked before later prerequisites")
    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda _: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", identity_loader("energy", "verify_energy_record_identity"))
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", identity_loader("dynamics", "verify_dynamics_record_identity"))
    monkeypatch.setattr(checked, "_load_word_record_gate", word)
    monkeypatch.setattr(checked, "_load_suzuki_record_gate", bundle_loader("suzuki"))
    monkeypatch.setattr(checked, "_load_gibbs_channel_record_gate", bundle_loader("gibbs"))
    monkeypatch.setattr(checked, "_load_tail_record_gate", tail)
    monkeypatch.setattr(checked, "_load_dqi_saved_gate", dqi)
    monkeypatch.setattr(checked, "_verify_bath_saved_identity", later_bath)
    with pytest.raises(StopAfterDqi): checked.verify_package(root, run_tests=False)
    assert order == prerequisites + ["dqi-load", "dqi-check"]
    assert loaded[0].CALLS == [str(root / BUNDLE)]


def test_dqi_route_downloads_and_normal_navigation_are_required(checker):
    required = set(checker.required_artifacts())
    assert PUBLIC_FILES <= required
    assert {"algorithms/optimization/dqi/index.html", "notebooks/optimization-dqi/index.html", NOTEBOOK,
            "code/src/openquantum/optimization_dqi.py", "code/tests/test_optimization_dqi.py",
            "code/scripts/generate_optimization_dqi.py", "code/scripts/verify_dqi_saved.py"} <= required
    navigation = ROOT / "myst.yml"
    if not navigation.is_file(): pytest.skip("Navigation is outside the public examples package")
    lines = navigation.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if line.strip() == "- file: algorithms/optimization/dqi.md"]
    assert len(matches) == 1
    index = matches[0]
    indent = lines[index].split("- file:")[0]
    end = next((offset for offset in range(index + 1, len(lines)) if lines[offset].startswith(indent + "- file:")), len(lines))
    block = "\n".join(lines[index:end])
    assert 'title: "Decoded quantum interferometry"' in block
    assert "(planned)" not in block and "children:" in block
    assert "- file: " + NOTEBOOK in block and 'title: "DQI notebook"' in block
    assert sum(line.strip() == "- file: " + NOTEBOOK for line in lines) == 1


def test_ci_has_one_saved_dqi_insertion_and_exact_prior_projection():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file(): pytest.skip("Publication workflow is outside the public examples package")
    commands = [line.strip().removeprefix("run ") for line in workflow.read_text().splitlines()
                if line.startswith("          run python ")]
    measured_and = "python scripts/verify_shor_measured_and_saved.py"
    assert len(commands) == 81 and commands.count(measured_and) == 1
    assert commands.index(measured_and) == commands.index("python scripts/generate_shor_recovery.py") + 1
    commands = [value for value in commands if value != measured_and]
    feedback = "python scripts/verify_qpe_feedback_public.py"
    assert len(commands) == 80 and commands.count(feedback) == 1
    assert commands.index(feedback) == commands.index("python scripts/verify_robust_qpe_saved.py") + 1
    commands = [value for value in commands if value != feedback]
    robust = "python scripts/verify_robust_qpe_saved.py"
    assert len(commands) == 79 and commands.count(robust) == 1
    assert commands.index(robust) == commands.index("python scripts/verify_bayesian_qpe_saved.py") + 1
    commands = [value for value in commands if value != robust]
    bayesian = "python scripts/verify_bayesian_qpe_saved.py"
    assert len(commands) == 78 and commands.count(bayesian) == 1
    assert commands.index(bayesian) == commands.index("python scripts/verify_hadamard_qpe.py") + 1
    commands = [value for value in commands if value != bayesian]
    assert len(commands) == 77
    assert sha(("\n".join(commands) + "\n").encode()) == "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032"
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert commands.count(multiproduct) == 1
    assert commands.index(multiproduct) == commands.index(
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay") + 1
    commands = [value for value in commands if value != multiproduct]
    assert len(commands) == 76 and sha(("\n".join(commands) + "\n").encode()) == CI_FINAL_PIN
    assert commands.count(CI_DQI) == 1
    index = commands.index(CI_DQI)
    assert commands[index - 1] == "python scripts/verify_optimization_word_records.py"
    assert commands[index + 1] == "python scripts/generate_quantum_walks.py --estimate"
    assert not any("generate_optimization_dqi.py" in command for command in commands)
    prior = commands[:index] + commands[index + 1:]
    assert len(prior) == 75 and sha(("\n".join(prior) + "\n").encode()) == CI_PRIOR_PIN
    energy = next(position for position, command in enumerate(commands) if command.startswith("python scripts/verify_energy_estimation.py "))
    assert commands[energy - 1] == "python scripts/generate_quantum_walks.py --estimate"
    assert commands[energy + 1] == "python scripts/generate_energy_estimation_pauli.py --estimate --compare-controls"
    assert commands[-4:] == ["python scripts/execute_notebooks.py", "python scripts/export_code.py",
                            "python scripts/verify_public_package.py", "python scripts/publication_provenance.py"]
