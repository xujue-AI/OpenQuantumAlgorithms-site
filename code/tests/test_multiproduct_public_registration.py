"""MPF saved-file publication boundaries, without scientific execution.

Shared functions are selected from captured AST snapshots. Scientific Python,
QASM and notebook files are inert fixture bytes; no project module, generator,
simulator, estimator, notebook or subprocess is imported or executed here.
Only tiny synthetic helpers exercise authenticated loading and dispatch.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import sys
from types import ModuleType, SimpleNamespace
from urllib.parse import unquote, urlsplit
import warnings
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE = "assets/generated/simulation/"
HELPER = "scripts/verify_simulation_multiproduct_saved.py"
NOTEBOOK = "notebooks/simulation_multiproduct.ipynb"
SUMMARY = BASE + "multiproduct-validation.json"
NOTEBOOK_PIN = "41efe7567bd96b4499a33c942361f0dbd491d234c4ef405495b14a40c3384e21"
SUMMARY_PIN = "547668d70ee75eff1f47b67bb61d79506a09e0c03bfeb23e2738d1b43187deae"
HELPER_PIN = "f506a41640492212e3d75659c990d3719e1819d05f8cc30d739d59dda2b83518"
INPUT_PINS = {
    "src/openquantum/simulation/multiproduct.py": "810f4c8431845f8d8310ab56dc84564fd521bc5ccb6c7d6d7ac6e398ad1b211a",
    "scripts/generate_simulation_multiproduct.py": "bd68b40ad9c3745241094c4e52707c0e2b17230db9fb1f01ba08579a3d2d6489",
    "tests/test_simulation_multiproduct.py": "75ac9e582c18a06634793ceb601faf96c5171d222dde40ef485a45a6622d36e6",
    "requirements-lock.txt": "6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088",
    "LICENSE": "5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c",
}
BUNDLES = (
    ("multiproduct", 86, "ec73967168a1d80fd70439c872f80f174a506f99875c234c6bd6f2a3a973f6b7",
     "fd0988cc877bc70e443678dad7507d565a857b2873a0f9168179b32c195f28ac"),
    ("multiproduct-large", 16, "c3b5f06b9fb9a3a10f6d5fd70ea37e55e21a0c5cb7ce00c74fd553bea8282f5f",
     "78be2e540b642b42a079ffb78bccda563a95859ed21429dd2bc598149e1d6414"),
)
CASES = ((1, 1), (1, 2), (1, 4), (1, 8), (2, 1), (2, 2), (2, 4), (2, 8),
         (3, 1), (3, 2), (3, 4), (3, 8), (2, 32), (3, 32))
COMMON = frozenset({"manifest.json", "multiproduct.py", "generate_simulation_multiproduct.py",
                    "requirements-lock.txt", "LICENSE.txt", "README.md", "study.json", "scalar-scaling.json"})
GRID_NAMES = COMMON | frozenset({"convergence.csv", "convergence.svg", "small-lcu-circuit.svg",
    "opposite-identity-lcu.qasm", "opposite-identity-oaa.qasm", "state-n2-r4-lcu.qasm",
    "state-n2-r4-oaa.qasm", "state-n2-r4-oaa-channel.qasm"}) | frozenset(
    f"n{n}-r{r}-{method}.qasm" for n, r in CASES for method in ("lcu", "oaa", "oaa-channel", "fine", "coarse"))
LARGE_NAMES = COMMON | frozenset({"n32-r1-lcu.qasm", "n32-r1-oaa.qasm", "n32-r1-oaa-channel.qasm",
    "n32-r1-fine.qasm", "n32-r1-coarse.qasm", "state-n32-r1-lcu.qasm", "state-n32-r1-oaa.qasm",
    "state-n32-r1-oaa-channel.qasm"})
PUBLIC_FILES = frozenset(BASE + "multiproduct/" + name for name in GRID_NAMES) | frozenset(
    BASE + "multiproduct-large/" + name for name in LARGE_NAMES)
ARCHIVES = frozenset(BASE + name + "-bundle.zip" for name, *_ in BUNDLES)
DEPENDENCIES = frozenset(BASE + directory + "/" + name for directory, *_ in BUNDLES
                         for name in ("requirements-lock.txt", "LICENSE.txt"))
SVG_PATHS = frozenset(BASE + "multiproduct/" + name for name in ("convergence.svg", "small-lcu-circuit.svg"))
CI_STAGE = "python scripts/verify_simulation_multiproduct_saved.py"
CI_FINAL_PIN = "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032"
CI_PRIOR_PIN = "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def projection(raw):
    """Independent ordered ID/type/text projection, excluding mutable outputs."""
    document = json.loads(raw)
    cells = [{"id": cell["id"], "cell_type": cell["cell_type"],
              "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
             for cell in document["cells"]]
    return sha(json.dumps(cells, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))


class SourceSnapshot:
    """Load definitions from one captured script without its sibling imports."""
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
        os_proxy = SimpleNamespace(open=os.open, close=os.close, fdopen=os.fdopen, fstat=os.fstat,
            O_RDONLY=os.O_RDONLY, O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 0),
            O_NONBLOCK=getattr(os, "O_NONBLOCK", 0), environ={})
        util_proxy = SimpleNamespace(spec_from_file_location=importlib.util.spec_from_file_location,
                                     module_from_spec=importlib.util.module_from_spec)
        module.__dict__.update(Path=Path, PurePosixPath=PurePosixPath, hashlib=SimpleNamespace(sha256=hashlib.sha256),
            json=json, io=io, os=os_proxy, stat=stat, math=math, HTMLParser=HTMLParser, zipfile=zipfile,
            importlib=SimpleNamespace(util=util_proxy), sys=SimpleNamespace(argv=[], executable=sys.executable),
            unquote=unquote, urlsplit=urlsplit, LINDBLAD_PORT_FILES=frozenset(), git_revision=lambda _: "a" * 40,
            subprocess=SimpleNamespace(run=lambda *_, **__: pytest.fail("Registration reached a subprocess")))
        module.__dict__.update(extra)
        exec(compile(tree, str(self.path), "exec"), module.__dict__)
        return module


def write_files(root, files):
    root.mkdir(parents=True)
    for name, raw in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def write_provenance(root, files):
    hashes = {name: sha(raw) for name, raw in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def saved_result():
    return {"status": "saved_component_verified", "bundles": {
        name: {"files": count, "manifest_sha256": manifest} for name, count, manifest, _ in BUNDLES},
        "notebook_source_sha256": NOTEBOOK_PIN, "new_quantum_execution": False,
        "physical_estimation_performed": False}


def synthetic_helper():
    return ("TOKEN = 'authenticated synthetic MPF helper'\nCALLS = []\n"
            "def verify_saved(root, **kwargs):\n"
            "    CALLS.append((root, kwargs))\n"
            "    return " + repr(saved_result()) + "\n").encode()


@pytest.fixture
def saved():
    return SourceSnapshot(ROOT / HELPER).module("mpf_registration_saved")


@pytest.fixture
def checked():
    return SourceSnapshot(ROOT / "scripts/verify_public_package.py").module("mpf_registration_package")


@pytest.fixture
def exporter():
    return SourceSnapshot(ROOT / "scripts/export_code.py").module("mpf_registration_export")


@pytest.fixture(scope="module")
def payload():
    # Audited scientific sources and downloaded circuits remain inert bytes.
    names = set(INPUT_PINS) | PUBLIC_FILES | ARCHIVES | {NOTEBOOK, SUMMARY}
    return {name: (ROOT / name).read_bytes() for name in names}


@pytest.fixture
def packet(tmp_path, payload):
    root = (tmp_path / "saved").resolve()
    write_files(root, payload)
    return root, write_provenance(root, payload)


@pytest.fixture
def package(tmp_path, payload, checked, monkeypatch):
    root = (tmp_path / "package").resolve()
    files = {**payload, HELPER: synthetic_helper()}
    write_files(root, files)
    provenance = write_provenance(root, files)
    # Explicit synthetic trust root; immutable production pins are asserted separately.
    monkeypatch.setattr(checked, "MULTIPRODUCT_SOURCE_HASHES", {**INPUT_PINS, HELPER: sha(files[HELPER])})
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
                  _load_multiproduct_saved_gate=checked._load_multiproduct_saved_gate,
                  _read_bounded_public_member=checked._read_bounded_public_member)
    for tag in ("N2", "N3", "N4", "PORT", "QDRIFT"):
        values[tag + "_FILES"] = frozenset()
        values[tag + "_PATH"] = "unused-" + tag.lower()
    values.update(PORT_SOURCE_FILES={}, OPT7_PATH="unused-compiled",
        ENERGY_CORE_FILES=frozenset(), ENERGY_MANIFEST_NAME="unused.json", ENERGY_RECORD_DIRECTORY="unused-energy",
        ENERGY_SOURCE_FILES=(), ENERGY_VERIFIER_FILES=(), DYNAMICS_CORE_FILES=frozenset(),
        DYNAMICS_MANIFEST_NAME="unused.json", DYNAMICS_RECORD_DIRECTORY="unused-dynamics",
        DYNAMICS_SOURCE_FILES=(), DYNAMICS_VERIFIER_FILES=())
    for name in ("verify_n2_bundle", "verify_n3_bundle", "verify_n4_bundle", "verify_port_bundle",
                 "verify_qdrift_bundle", "verify_hadamard_record", "verify_opt7_bundle"):
        values[name] = lambda *_, **__: {}
    for name in ("site_energy_record_errors", "site_dynamics_record_errors"):
        values[name] = lambda *_, **__: []
    return SourceSnapshot(path).module("mpf_registration_site", **values)


@pytest.fixture
def site(tmp_path, payload, checked, monkeypatch):
    root = (tmp_path / "site").resolve()
    files = {**payload, HELPER: synthetic_helper()}
    retained = {name: raw for name, raw in files.items() if not name.startswith("assets/")}
    write_files(root / "code", retained)
    provenance = write_provenance(root / "code", retained)
    for name, raw in files.items():
        if name.startswith("assets/") or name == NOTEBOOK:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    monkeypatch.setattr(checked, "MULTIPRODUCT_SOURCE_HASHES", {**INPUT_PINS, HELPER: sha(files[HELPER])})
    identity = {"schema": "openquantum-source-preview-v1", "retained_sha256": {
        "PROVENANCE.json": sha((root / "code/PROVENANCE.json").read_bytes())}}
    return root, identity, provenance


def test_all_promoted_inputs_helper_and_saved_payloads_keep_exact_pins(saved, checked, payload):
    assert saved.SOURCE_HASHES == INPUT_PINS
    assert checked.MULTIPRODUCT_SOURCE_HASHES == {**INPUT_PINS, HELPER: HELPER_PIN}
    assert sha((ROOT / HELPER).read_bytes()) == HELPER_PIN
    assert saved.BUNDLES == BUNDLES and saved.NOTEBOOK == NOTEBOOK and saved.SUMMARY == SUMMARY
    assert saved.NOTEBOOK_SOURCE_SHA256 == NOTEBOOK_PIN and saved.SUMMARY_SHA256 == SUMMARY_PIN
    assert saved.MAX_FILE_BYTES == 4 * 1024 * 1024 and saved.MAX_BUNDLE_BYTES == 32 * 1024 * 1024
    assert len(GRID_NAMES) == 86 and len(LARGE_NAMES) == 16 and len(PUBLIC_FILES) == 102
    for name, pin in INPUT_PINS.items():
        assert sha(payload[name]) == pin
    for name, count, manifest, archive in BUNDLES:
        directory = ROOT / BASE / name
        expected = GRID_NAMES if name == "multiproduct" else LARGE_NAMES
        assert {path.name for path in directory.iterdir()} == expected and len(expected) == count
        assert sha(payload[BASE + name + "/manifest.json"]) == manifest
        assert sha(payload[BASE + name + "-bundle.zip"]) == archive
        for source, flat in (("src/openquantum/simulation/multiproduct.py", "multiproduct.py"),
                             ("scripts/generate_simulation_multiproduct.py", "generate_simulation_multiproduct.py"),
                             ("requirements-lock.txt", "requirements-lock.txt"), ("LICENSE", "LICENSE.txt")):
            assert payload[source] == payload[BASE + name + "/" + flat]
    assert sha(payload[SUMMARY]) == SUMMARY_PIN


def test_saved_helper_imports_only_standard_library_and_has_no_scientific_entrypoint():
    tree = SourceSnapshot(ROOT / HELPER).tree
    modules = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    modules |= {node.module for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert modules == {"__future__", "argparse", "hashlib", "io", "json", "os", "pathlib", "stat", "zipfile"}
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in
                   ("exec_module", "run", "simulate", "estimate") for node in calls)


def test_closed_export_archives_dependencies_and_scripts(exporter):
    assert exporter.PUBLIC_MULTIPRODUCT_FILES == PUBLIC_FILES
    assert {name for name in exporter.PUBLIC_DEPENDENCY_FILES if "/multiproduct" in name} == DEPENDENCIES
    assert {name for name in exporter.PUBLIC_ARCHIVES if "/multiproduct" in name} == ARCHIVES
    scripts = ("generate_simulation_multiproduct.py", "verify_simulation_multiproduct_saved.py")
    assert tuple(name for name in exporter.PUBLIC_SCRIPTS if "multiproduct" in name) == scripts
    assert all(exporter.PUBLIC_SCRIPTS.count(name) == 1 for name in scripts)
    assert ".zip" not in exporter.ARTIFACT_SUFFIXES and ".txt" not in exporter.ARTIFACT_SUFFIXES
    assert all(exporter.include_generated_file(Path(name)) for name in PUBLIC_FILES | ARCHIVES | {SUMMARY})
    for group in (exporter.PUBLIC_DQI_FILES, exporter.PUBLIC_BATH_FILES, exporter.PUBLIC_SUZUKI_FILES,
                  exporter.PUBLIC_GIBBS_CHANNEL_FILES, exporter.PUBLIC_OPTIMIZATION_WORD_FILES):
        assert group and all(exporter.include_generated_file(Path(name)) for name in group)


@pytest.mark.parametrize("directory", ["multiproduct", "multiproduct-large"])
@pytest.mark.parametrize("name", ["unreviewed.json", "history.py", "new.ipynb", "private.txt",
                                  "trace.json.gz", "extra.zip", "nested/study.json", ".cache/result.json"])
def test_closed_export_rejects_extra_and_nested_members(exporter, directory, name):
    assert not exporter.include_generated_file(Path(BASE) / directory / name)


def test_notebook_order_ids_type_and_text_are_frozen_but_outputs_are_mutable(saved, payload):
    raw = payload[NOTEBOOK]
    assert projection(raw) == saved.notebook_source_sha256(raw) == NOTEBOOK_PIN
    document = json.loads(raw)
    assert len(document["cells"]) == 14 and sum(cell["cell_type"] == "code" for cell in document["cells"]) == 7
    changed = copy.deepcopy(document)
    changed["metadata"] = {"fresh": True}
    for cell in changed["cells"]:
        cell["metadata"] = {"fresh": True}
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
        if isinstance(cell["source"], list):
            cell["source"] = "".join(cell["source"])
    assert saved.notebook_source_sha256(json.dumps(changed).encode()) == NOTEBOOK_PIN


@pytest.mark.parametrize("change", ["id", "source", "type", "order", "count", "nbformat", "bad-source", "bad-id"])
def test_notebook_projection_rejects_each_source_contract_change(saved, payload, change):
    document = json.loads(payload[NOTEBOOK])
    if change == "id": document["cells"][0]["id"] = "changed-id"
    elif change == "source": document["cells"][0]["source"] = ["changed reusable source\n"]
    elif change == "type": document["cells"][0]["cell_type"] = "raw"
    elif change == "order": document["cells"][0], document["cells"][1] = document["cells"][1], document["cells"][0]
    elif change == "count": document["cells"].pop()
    elif change == "nbformat": document["nbformat"] = 3
    elif change == "bad-source": document["cells"][0]["source"] = [3]
    else: document["cells"][0]["id"] = None
    raw = json.dumps(document).encode()
    if change in ("count", "nbformat", "bad-source", "bad-id"):
        with pytest.raises(ValueError): saved.notebook_source_sha256(raw)
    else:
        assert saved.notebook_source_sha256(raw) != NOTEBOOK_PIN


def test_saved_verification_authenticates_exact_data_without_extending_scope(saved, packet):
    root, provenance = packet
    result = saved.verify_saved(root, provenance=provenance, notebook_roots=(root,))
    assert result["status"] == "saved_component_verified" and result["notebook_source_sha256"] == NOTEBOOK_PIN
    assert result["new_quantum_execution"] is False and result["physical_estimation_performed"] is False
    assert set(result["bundles"]) == {name for name, *_ in BUNDLES}
    for name, count, pin, _ in BUNDLES:
        assert result["bundles"][name]["files"] == count
        assert result["bundles"][name]["manifest_sha256"] == pin
        assert result["bundles"][name]["bytes"] == sum(path.stat().st_size for path in (root / BASE / name).iterdir())


@pytest.mark.parametrize("name", list(INPUT_PINS) + [SUMMARY, BASE + "multiproduct/manifest.json",
    BASE + "multiproduct/README.md", BASE + "multiproduct-large/study.json", BASE + "multiproduct-bundle.zip"])
def test_saved_verification_rejects_corrupted_source_or_data(saved, packet, name):
    root, _ = packet
    path = root / name
    path.write_bytes(path.read_bytes() + b"\nchanged\n")
    with pytest.raises(ValueError): saved.verify_saved(root)


@pytest.mark.parametrize("change", ["schema", "hashes-type", "source-pin", "notebook-bytes"])
def test_saved_provenance_fails_before_bundle_dispatch(saved, packet, monkeypatch, change):
    root, provenance = packet
    if change == "schema": provenance["schema"] = "wrong"
    elif change == "hashes-type": provenance["file_sha256"] = []
    elif change == "source-pin": provenance["file_sha256"][next(iter(INPUT_PINS))] = "0" * 64
    else: provenance["file_sha256"][NOTEBOOK] = "0" * 64
    monkeypatch.setattr(saved, "_bundle", lambda *_: pytest.fail("Unauthenticated input reached a bundle"))
    with pytest.raises(ValueError): saved.verify_saved(root, provenance=provenance)


@pytest.mark.parametrize("roots", [[], (), (Path("unused"),) * 3])
def test_saved_notebook_root_shape_is_rejected_before_reads(saved, packet, monkeypatch, roots):
    root, _ = packet
    monkeypatch.setattr(saved, "_read", lambda *_: pytest.fail("Invalid notebook roots reached a read"))
    with pytest.raises(ValueError, match="roots"): saved.verify_saved(root, notebook_roots=roots)


def test_remanifested_notebook_metadata_passes_but_source_change_fails(saved, packet, monkeypatch):
    root, provenance = packet
    document = json.loads((root / NOTEBOOK).read_bytes())
    document["metadata"] = {"fresh": True}
    for cell in document["cells"]:
        cell["metadata"] = {}
        if cell["cell_type"] == "code": cell.update(outputs=[], execution_count=None)
    raw = json.dumps(document).encode()
    (root / NOTEBOOK).write_bytes(raw)
    provenance["file_sha256"][NOTEBOOK] = sha(raw)
    assert saved.verify_saved(root, provenance=provenance)["new_quantum_execution"] is False
    document["cells"][0]["id"] = "new-id"
    raw = json.dumps(document).encode()
    (root / NOTEBOOK).write_bytes(raw)
    provenance["file_sha256"][NOTEBOOK] = sha(raw)
    monkeypatch.setattr(saved, "_bundle", lambda *_: pytest.fail("Changed source reached saved bundle dispatch"))
    with pytest.raises(ValueError, match="notebook source"): saved.verify_saved(root, provenance=provenance)


@pytest.mark.parametrize("name", ["", "../data", "nested/../data", "nested//data", "./data", "/data", "data\\other"])
def test_saved_reader_rejects_lexical_aliases_before_open(saved, tmp_path, monkeypatch, name):
    monkeypatch.setattr(saved.os, "open", lambda *_: pytest.fail("Lexical alias reached open"))
    with pytest.raises(ValueError): saved._read(tmp_path.resolve(), name)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "linked-parent", "directory", "fifo", "oversized"])
def test_saved_reader_rejects_unsafe_members_before_open(saved, tmp_path, monkeypatch, kind):
    root = tmp_path.resolve()
    path = root / "data"
    if kind == "linked-parent":
        target = root / "target"
        target.mkdir()
        (target / "data").write_bytes(b"data")
        (root / "alias").symlink_to(target, target_is_directory=True)
        name = "alias/data"
    elif kind == "directory": path.mkdir(); name = "data"
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"): pytest.skip("FIFO contract requires a POSIX host")
        os.mkfifo(path); name = "data"
    else:
        target = root / "target"
        target.write_bytes(b"data")
        if kind == "symlink": path.symlink_to(target)
        elif kind == "hardlink": os.link(target, path)
        else: path.write_bytes(b"x" * 5)
        name = "data"
    monkeypatch.setattr(saved.os, "open", lambda *_: pytest.fail("Unsafe member reached open"))
    with pytest.raises(ValueError): saved._read(root, name, 4)


@pytest.mark.parametrize("replacement", ["regular", "symlink", "fifo"])
def test_saved_reader_rejects_a_raced_open_without_blocking(saved, tmp_path, monkeypatch, replacement):
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_NONBLOCK", 0):
        pytest.skip("Raced-open contract requires POSIX nofollow and nonblock flags")
    path = tmp_path.resolve() / "data"
    path.write_bytes(b"data")
    real_open = saved.os.open
    seen = []
    def raced(source, flags):
        seen.append(flags)
        source.unlink()
        if replacement == "regular": source.write_bytes(b"data")
        elif replacement == "symlink":
            target = tmp_path / "target"
            target.write_bytes(b"data")
            source.symlink_to(target)
        else: os.mkfifo(source)
        return real_open(source, flags)
    monkeypatch.setattr(saved.os, "open", raced)
    with pytest.raises((ValueError, OSError)): saved._read(tmp_path.resolve(), "data", 4)
    assert seen and seen[0] & os.O_NOFOLLOW and seen[0] & os.O_NONBLOCK


def test_saved_reader_uses_one_bounded_read_and_detects_same_size_mutation(saved, tmp_path, monkeypatch):
    path = tmp_path.resolve() / "data"
    path.write_bytes(b"data")
    real_fdopen = saved.os.fdopen
    sizes = []
    class MutatingStream:
        def __init__(self, descriptor, mode, closefd):
            self.stream = real_fdopen(descriptor, mode, closefd=closefd)
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, size):
            sizes.append(size)
            raw = self.stream.read(size)
            path.write_bytes(b"edit")
            return raw
    monkeypatch.setattr(saved.os, "fdopen", MutatingStream)
    with pytest.raises(ValueError, match="during read"): saved._read(tmp_path.resolve(), "data", 4)
    assert sizes == [5]


def make_tiny_bundle(root, *, mutate_manifest=None, mutate_zip=None):
    """A two-file standard-library fixture, independent of scientific exports."""
    name = "synthetic"
    directory = root / BASE / name
    directory.mkdir(parents=True)
    payload = b'{"inert":true}\n'
    records = {"payload.json": {"bytes": len(payload), "sha256": sha(payload)}}
    manifest = {"schema": "openquantum-fixed-multiproduct-bundle-v1", "files": records,
                "exports_bytes_without_manifest": len(payload)}
    if mutate_manifest: mutate_manifest(manifest)
    raw = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(raw)
    (directory / "payload.json").write_bytes(payload)
    entries = [("manifest.json", raw, stat.S_IFREG | 0o644), ("payload.json", payload, stat.S_IFREG | 0o644)]
    if mutate_zip: entries = mutate_zip(entries)
    buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # Deliberately duplicated ZIP fixture member.
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for filename, content, mode in entries:
                entry = zipfile.ZipInfo(filename)
                entry.create_system = 3
                entry.external_attr = mode << 16
                archive.writestr(entry, content)
    archive_raw = buffer.getvalue()
    (root / BASE / (name + "-bundle.zip")).write_bytes(archive_raw)
    return name, 2, sha(raw), sha(archive_raw)


@pytest.mark.parametrize("change", ["schema", "files-type", "count", "record-type", "bytes", "digest", "total"])
def test_bundle_rejects_remanifested_contract_and_accounting_changes(saved, tmp_path, change):
    def mutate(manifest):
        if change == "schema": manifest["schema"] = "wrong"
        elif change == "files-type": manifest["files"] = []
        elif change == "count": manifest["files"] = {}
        elif change == "record-type": manifest["files"]["payload.json"] = []
        elif change == "bytes": manifest["files"]["payload.json"]["bytes"] += 1
        elif change == "digest": manifest["files"]["payload.json"]["sha256"] = "0" * 64
        else: manifest["exports_bytes_without_manifest"] += 1
    args = make_tiny_bundle(tmp_path.resolve(), mutate_manifest=mutate)
    with pytest.raises(ValueError): saved._bundle(tmp_path.resolve(), *args)


@pytest.mark.parametrize("change", ["extra", "nested", "missing", "symlink", "hardlink"])
def test_bundle_requires_exact_flat_regular_members(saved, tmp_path, change):
    args = make_tiny_bundle(tmp_path.resolve())
    directory = tmp_path / BASE / args[0]
    if change == "extra": (directory / "extra.json").write_bytes(b"extra")
    elif change == "nested": (directory / "nested").mkdir()
    elif change == "missing": (directory / "payload.json").unlink()
    else:
        target = tmp_path / "target"
        (directory / "payload.json").rename(target)
        if change == "symlink": (directory / "payload.json").symlink_to(target)
        else: os.link(target, directory / "payload.json")
    with pytest.raises(ValueError): saved._bundle(tmp_path.resolve(), *args)


@pytest.mark.parametrize("change", ["missing", "duplicate", "nested", "directory", "symlink", "bytes"])
def test_bundle_rejects_rehashed_zip_membership_modes_and_payload(saved, tmp_path, change):
    def mutate(entries):
        if change == "missing": return entries[:1]
        if change == "duplicate": return [entries[0], entries[0]]
        filename, raw, mode = entries[1]
        if change == "nested": filename = "nested/" + filename
        elif change == "directory": filename += "/"; mode = stat.S_IFDIR | 0o755
        elif change == "symlink": mode = stat.S_IFLNK | 0o777
        else: raw = b"x" * len(raw)
        return [entries[0], (filename, raw, mode)]
    args = make_tiny_bundle(tmp_path.resolve(), mutate_zip=mutate)
    with pytest.raises(ValueError): saved._bundle(tmp_path.resolve(), *args)


@pytest.mark.parametrize("change", ["encrypted", "advertised-size", "member-cap"])
def test_zip_metadata_is_rejected_before_member_stream_open(saved, tmp_path, monkeypatch, change):
    args = make_tiny_bundle(tmp_path.resolve())
    class RejectedMetadata(zipfile.ZipFile):
        def infolist(self):
            entries = super().infolist()
            if change == "encrypted": entries[0].flag_bits |= 1
            elif change == "advertised-size": entries[0].file_size += 1
            return entries
        def open(self, *_, **__): pytest.fail("Rejected ZIP metadata reached decompression")
    if change == "member-cap": monkeypatch.setattr(saved, "MAX_FILE_BYTES", 1)
    monkeypatch.setattr(saved, "zipfile", SimpleNamespace(ZipFile=RejectedMetadata))
    with pytest.raises(ValueError): saved._bundle(tmp_path.resolve(), *args)


def test_zip_stream_reads_are_bounded_by_expected_bytes(saved, tmp_path, monkeypatch):
    args = make_tiny_bundle(tmp_path.resolve())
    reads = []
    class ObservedZip(zipfile.ZipFile):
        def open(self, entry, *args, **kwargs):
            stream = super().open(entry, *args, **kwargs)
            class ObservedStream:
                def __enter__(self): return self
                def __exit__(self, *args): stream.close()
                def read(self, count):
                    reads.append((entry.filename, count, entry.file_size))
                    return stream.read(count)
            return ObservedStream()
    monkeypatch.setattr(saved, "zipfile", SimpleNamespace(ZipFile=ObservedZip))
    assert saved._bundle(tmp_path.resolve(), *args)["files"] == 2
    assert len(reads) == 2 and all(count == size + 1 for _, count, size in reads)


def test_bundle_total_cap_is_checked_before_zip_read(saved, tmp_path, monkeypatch):
    args = make_tiny_bundle(tmp_path.resolve())
    monkeypatch.setattr(saved, "MAX_BUNDLE_BYTES", 1)
    with pytest.raises(ValueError, match="payload identity"): saved._bundle(tmp_path.resolve(), *args)


@pytest.mark.parametrize("change", ["schema", "hashes-type", "missing-helper", "changed-helper", "remanifested-helper",
    *INPUT_PINS])
def test_loader_rejects_unbound_helper_and_promoted_inputs_before_compile(package, checked, monkeypatch, change):
    root, provenance = package
    if change == "schema": provenance["schema"] = "wrong"
    elif change == "hashes-type": provenance["file_sha256"] = []
    elif change == "missing-helper": provenance["file_sha256"].pop(HELPER)
    elif change in ("changed-helper", "remanifested-helper"):
        raw = b"raise AssertionError('untrusted helper reached execution')\n"
        (root / HELPER).write_bytes(raw)
        if change == "remanifested-helper": provenance["file_sha256"][HELPER] = sha(raw)
    else: provenance["file_sha256"][change] = "0" * 64
    monkeypatch.setattr(builtins, "compile", lambda *_, **__: pytest.fail("Unbound MPF input reached compile"))
    with pytest.raises(ValueError): checked._load_multiproduct_saved_gate(root, provenance)


def test_loader_uses_captured_authenticated_bytes_and_avoids_module_cache(package, checked, monkeypatch):
    root, provenance = package
    shadow = ModuleType("verify_simulation_multiproduct_saved")
    shadow.TOKEN = "cached unrelated helper"
    monkeypatch.setitem(sys.modules, "verify_simulation_multiproduct_saved", shadow)
    before = list(sys.path)
    actual = checked._read_bounded_public_member
    reads = []
    def read(directory, name, maximum):
        raw = actual(directory, name, maximum)
        reads.append((directory, name, maximum))
        (directory / name).write_bytes(b"raise AssertionError('reopened helper')\n")
        return raw
    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    gate = checked._load_multiproduct_saved_gate(root, provenance)
    assert gate.TOKEN == "authenticated synthetic MPF helper" and gate is not shadow
    assert reads == [(root, HELPER, 64 * 1024)] and sys.path == before
    assert all(module is not gate for module in list(sys.modules.values()))
    assert sys.modules["verify_simulation_multiproduct_saved"] is shadow


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "linked-parent", "oversized"])
def test_loader_rejects_links_and_size_before_compile(package, checked, monkeypatch, tmp_path, kind):
    root, provenance = package
    source = root / HELPER
    if kind == "oversized": source.write_bytes(b"x" * (64 * 1024 + 1))
    elif kind == "linked-parent":
        target = tmp_path / "original-scripts"
        source.parent.rename(target)
        source.parent.symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "original-helper"
        source.rename(target)
        if kind == "symlink": source.symlink_to(target)
        else: os.link(target, source)
    monkeypatch.setattr(builtins, "compile", lambda *_, **__: pytest.fail("Unsafe MPF helper reached compile"))
    with pytest.raises(ValueError): checked._load_multiproduct_saved_gate(root, provenance)


def test_site_uses_one_authenticated_provenance_snapshot_and_exact_root_mapping(site, checker, checked, monkeypatch):
    root, identity, provenance = site
    assert not (root / "code/assets").exists()
    actual_read, actual_loader = checked._read_bounded_public_member, checked._load_multiproduct_saved_gate
    reads, loaded = [], []
    def read(directory, name, maximum):
        raw = actual_read(directory, name, maximum)
        reads.append((directory, name, maximum))
        if name == "PROVENANCE.json": (directory / name).write_text("{}")
        return raw
    def load(directory, authenticated):
        assert authenticated == provenance
        gate = actual_loader(directory, authenticated)
        loaded.append(gate)
        return gate
    monkeypatch.setattr(checker, "_read_bounded_public_member", read)
    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    monkeypatch.setattr(checker, "_load_multiproduct_saved_gate", load)
    assert checker.site_multiproduct_saved_errors(root, identity) == []
    assert reads == [(root / "code", "PROVENANCE.json", 16 * 1024 * 1024), (root / "code", HELPER, 64 * 1024)]
    assert loaded[0].CALLS == [(root, {"source_root": root / "code", "provenance": provenance,
                                    "notebook_roots": (root / "code", root)})]
    assert (root / "code/PROVENANCE.json").read_text() == "{}"


@pytest.mark.parametrize("change", ["missing", "schema", "retention-type", "changed-provenance"])
def test_site_preview_provenance_failure_precedes_helper_loading(site, checker, monkeypatch, change):
    root, identity, _ = site
    if change == "missing": identity = {}
    elif change == "schema": identity["schema"] = "wrong"
    elif change == "retention-type": identity["retained_sha256"] = []
    else: (root / "code/PROVENANCE.json").write_text("{}")
    monkeypatch.setattr(checker, "_load_multiproduct_saved_gate", lambda *_: pytest.fail("Unverified preview reached MPF helper"))
    assert checker.site_multiproduct_saved_errors(root, identity)


@pytest.mark.parametrize("where", ["code", "served"])
def test_real_saved_gate_binds_both_normal_notebooks_without_duplicate_assets(site, checker, saved, monkeypatch, where):
    root, identity, provenance = site
    monkeypatch.setattr(checker, "_load_multiproduct_saved_gate", lambda *_: saved)
    assert checker.site_multiproduct_saved_errors(root, identity) == []
    directory = root / "code" if where == "code" else root
    document = json.loads((directory / NOTEBOOK).read_bytes())
    document["cells"][0]["source"] = ["changed source\n"]
    raw = json.dumps(document).encode()
    (directory / NOTEBOOK).write_bytes(raw)
    if where == "code":
        provenance["file_sha256"][NOTEBOOK] = sha(raw)
        (root / "code/PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
        identity["retained_sha256"]["PROVENANCE.json"] = sha((root / "code/PROVENANCE.json").read_bytes())
    assert checker.site_multiproduct_saved_errors(root, identity)


def configure_site_main(checker, root, monkeypatch):
    (root / "index.html").write_text("<html><body>Pure MPF registration fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda *_: [])
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


def test_failed_complete_preview_never_dispatches_saved_gates(site, checker, monkeypatch):
    root, _, _ = site
    configure_site_main(checker, root, monkeypatch)
    def rejected(*_): raise ValueError("complete archive rejected")
    monkeypatch.setattr(checker, "verify_preview", rejected)
    for name in ("site_word_record_errors", "site_suzuki_record_errors", "site_gibbs_channel_record_errors",
                 "site_tail_record_errors", "site_dqi_saved_errors", "site_bath_saved_errors", "site_multiproduct_saved_errors"):
        monkeypatch.setattr(checker, name, lambda *_: pytest.fail("Failed preview dispatched a saved gate"))
    with pytest.raises(SystemExit, match="complete archive rejected"): checker.main()


def test_site_mpf_dispatch_preserves_all_prior_saved_gates(site, checker, monkeypatch):
    root, identity, _ = site
    configure_site_main(checker, root, monkeypatch)
    order = []
    monkeypatch.setattr(checker, "verify_preview", lambda _: order.append("preview") or identity)
    stages = (("word", "site_word_record_errors"), ("suzuki", "site_suzuki_record_errors"),
              ("gibbs", "site_gibbs_channel_record_errors"), ("tail", "site_tail_record_errors"),
              ("dqi", "site_dqi_saved_errors"), ("bath", "site_bath_saved_errors"))
    for label, name in stages:
        def prior(path, retained, stage=label):
            assert path == root and retained is identity
            order.append(stage)
            return []
        monkeypatch.setattr(checker, name, prior)
    actual = checker.site_multiproduct_saved_errors
    def mpf(path, retained):
        assert order == ["preview", "word", "suzuki", "gibbs", "tail", "dqi", "bath"]
        order.append("mpf")
        return actual(path, retained)
    monkeypatch.setattr(checker, "site_multiproduct_saved_errors", mpf)
    checker.main()
    assert order == ["preview", "word", "suzuki", "gibbs", "tail", "dqi", "bath", "mpf"]


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "wrong-provenance"])
def test_complete_package_failure_precedes_every_saved_gate(package, checked, monkeypatch, change):
    root, _ = package
    if change == "changed": (root / HELPER).write_bytes(b"raise AssertionError('unbound helper')\n")
    elif change == "missing": (root / HELPER).unlink()
    elif change == "unrecorded": (root / "extra.py").write_text("raise AssertionError('unrecorded source')\n")
    else: (root / "PROVENANCE.json").write_text("{}")
    calls = []
    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Failed complete package authentication dispatched a saved prerequisite")
    for name in ("verify_formal_record", "_load_energy_record_gate", "_load_dynamics_record_gate",
        "_load_word_record_gate", "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate", "_load_tail_record_gate",
        "_load_dqi_saved_gate", "_verify_bath_saved_identity", "_load_multiproduct_saved_gate", "_load_local_spin_saved_gate"):
        monkeypatch.setattr(checked, name, forbidden)
    with pytest.raises(ValueError): checked.verify_package(root, run_tests=False)
    assert calls == []


def test_package_mpf_gate_follows_bath_and_stops_before_scientific_imports(package, checked, monkeypatch):
    root, provenance = package
    order, loaded = [], []
    actual_manifest, actual_loader = checked.verify_manifest, checked._load_multiproduct_saved_gate
    def manifest(path):
        result = actual_manifest(path)
        assert result == provenance
        order.append("manifest")
        return result
    def record_loader(label, method):
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
            paths = {"instrument": "assets/generated/instrument-resources", "complete": "assets/generated/simulation/complete-resources"}
            pins = {"instrument": "0333814a823ebece83910251328b1145fc84e3cfbdec68051af92e78280ba220",
                    "complete": "86af1d97bad4ac1f4d8c5191d3e546f69ca82a5121bd8de7d7307081c70dc0b1"}
            assert directory == root / paths[kind] and source == root and expected_sha256 == pins[kind] and replay is False
            order.append("tail-" + kind)
        return SimpleNamespace(verify_bundle=verify)
    def dqi_load(path, authenticated):
        assert path == root and authenticated == provenance
        order.append("dqi-load")
        return SimpleNamespace()
    def dqi_check(gate, directory, authenticated, *, notebook_roots):
        assert directory == root / "assets/generated/optimization/dqi" and authenticated == provenance
        assert notebook_roots == (root,)
        order.append("dqi-check")
    def bath(path, *, provenance: dict, notebook_roots):
        assert path == root and provenance == expected_provenance and notebook_roots == (root,)
        order.append("bath")
    prerequisites = ["manifest", "formal", "energy-load", "energy-check", "dynamics-load", "dynamics-check",
        "word-load", "word-check", "suzuki-load", "suzuki-check", "gibbs-load", "gibbs-check", "tail-load",
        "tail-instrument", "tail-complete", "dqi-load", "dqi-check", "bath"]
    class StopAfterMpf(RuntimeError): pass
    def mpf(path, authenticated):
        assert path == root and authenticated == provenance and order == prerequisites
        order.append("mpf-load")
        gate = actual_loader(path, authenticated)
        loaded.append(gate)
        actual = gate.verify_saved
        def verify(directory, **kwargs):
            assert directory == root and kwargs == {"provenance": provenance, "notebook_roots": (root,)}
            actual(directory, **kwargs)
            order.append("mpf-check")
            raise StopAfterMpf("Authenticated MPF dispatch completed before scientific imports")
        gate.verify_saved = verify
        return gate
    expected_provenance = provenance
    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda _: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", record_loader("energy", "verify_energy_record_identity"))
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", record_loader("dynamics", "verify_dynamics_record_identity"))
    monkeypatch.setattr(checked, "_load_word_record_gate", word)
    monkeypatch.setattr(checked, "_load_suzuki_record_gate", bundle_loader("suzuki"))
    monkeypatch.setattr(checked, "_load_gibbs_channel_record_gate", bundle_loader("gibbs"))
    monkeypatch.setattr(checked, "_load_tail_record_gate", tail)
    monkeypatch.setattr(checked, "_load_dqi_saved_gate", dqi_load)
    monkeypatch.setattr(checked, "_verify_dqi_saved_identity", dqi_check)
    monkeypatch.setattr(checked, "_verify_bath_saved_identity", bath)
    monkeypatch.setattr(checked, "_load_multiproduct_saved_gate", mpf)
    with pytest.raises(StopAfterMpf): checked.verify_package(root, run_tests=False)
    assert order == prerequisites + ["mpf-load", "mpf-check"]
    assert loaded[0].CALLS == [(root, {"provenance": provenance, "notebook_roots": (root,)})]


def test_routes_downloads_and_normal_navigation_are_required(checker):
    required = set(checker.required_artifacts())
    assert PUBLIC_FILES | ARCHIVES | {SUMMARY} <= required
    assert {"algorithms/simulation/multiproduct/index.html", "notebooks/simulation-multiproduct/index.html", NOTEBOOK,
        "code/src/openquantum/simulation/multiproduct.py", "code/tests/test_simulation_multiproduct.py",
        "code/scripts/generate_simulation_multiproduct.py", "code/" + HELPER} <= required
    path = ROOT / "myst.yml"
    if not path.is_file(): pytest.skip("Navigation is outside the public examples package")
    lines = path.read_text().splitlines()
    matches = [index for index, line in enumerate(lines) if line.strip() == "- file: algorithms/simulation/multiproduct.md"]
    assert len(matches) == 1
    index = matches[0]
    indent = lines[index].split("- file:")[0]
    end = next((offset for offset in range(index + 1, len(lines)) if lines[offset].startswith(indent + "- file:")), len(lines))
    block = "\n".join(lines[index:end])
    assert 'title: "Coherent and classical multiproduct formulas"' in block
    assert "(planned)" not in block and "children:" in block
    assert "- file: " + NOTEBOOK in block and 'title: "Multiproduct notebook"' in block
    assert sum(line.strip() == "- file: " + NOTEBOOK for line in lines) == 1


def test_normalizer_skips_immutable_svg_bytes_before_whitespace_rewrites(payload):
    snapshot = SourceSnapshot(ROOT / "scripts/normalize_figures.py")
    normalizer = snapshot.module("mpf_registration_normalizer")
    assert SVG_PATHS <= normalizer.IMMUTABLE_SVGS
    loops = [node for node in snapshot.tree.body if isinstance(node, ast.For)]
    assert len(loops) == 1 and isinstance(loops[0].body[0], ast.If)
    guard = loops[0].body[0]
    assert ast.unparse(guard.test) == "path.relative_to(ROOT).as_posix() in IMMUTABLE_SVGS"
    assert len(guard.body) == 1 and isinstance(guard.body[0], ast.Continue)
    raw = payload[BASE + "multiproduct/convergence.svg"]
    rewritten = ("\n".join(line.rstrip() for line in raw.decode().splitlines()) + "\n").encode()
    assert rewritten != raw  # The immutable skip protects real significant hash bytes.
    csv = payload[BASE + "multiproduct/convergence.csv"]
    assert b"\r" not in csv and csv.decode().encode() == csv


def test_ci_has_one_saved_mpf_insertion_and_exact_prior_projection():
    path = ROOT / ".github/workflows/publish.yml"
    if not path.is_file(): pytest.skip("Publication workflow is outside the public examples package")
    commands = [line.strip().removeprefix("run ") for line in path.read_text().splitlines()
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
    assert len(commands) == 77 and sha(("\n".join(commands) + "\n").encode()) == CI_FINAL_PIN
    assert commands.count(CI_STAGE) == 1
    index = commands.index(CI_STAGE)
    assert commands[index - 1] == "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay"
    assert commands[index + 1] == "python scripts/generate_simulation_lcu.py"
    assert not any("generate_simulation_multiproduct.py" in command for command in commands)
    prior = commands[:index] + commands[index + 1:]
    assert len(prior) == 76 and sha(("\n".join(prior) + "\n").encode()) == CI_PRIOR_PIN
    assert not any("reusable_bath" in command or "reusable-bath" in command for command in commands)
    assert commands[-4:] == ["python scripts/execute_notebooks.py", "python scripts/export_code.py",
                            "python scripts/verify_public_package.py", "python scripts/publication_provenance.py"]
