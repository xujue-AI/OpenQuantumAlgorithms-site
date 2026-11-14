"""The dynamics publication helper executes only after package authentication."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import textwrap
import types
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/dynamics_record_release.py"
ENERGY_HELPER = "scripts/energy_record_release.py"
ARCHIVE = "assets/generated/openquantum-code.zip"
PREFIX = "openquantum-examples/"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def verifier():
    return load_source(ROOT / "scripts/verify_public_package.py", "dynamics_bootstrap_public_verifier")


def write_package(directory, files):
    directory.mkdir(parents=True)
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    hashes = {name: digest(content) for name, content in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "d" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: sha for name, sha in hashes.items() if name.endswith(".py")}}
    (directory / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def marker_source(marker):
    return f"from pathlib import Path\nPath({str(marker)!r}).write_text('unverified dynamics helper executed')\n".encode()


def helper_package(tmp_path, helper_bytes=None):
    if helper_bytes is None:
        helper_bytes = (b"RECORD_DIRECTORY = 'assets/generated/dynamics-resources'\n"
                        b"def verify_dynamics_record_identity(records, source):\n"
                        b"    return {'structural_fixture': True}\n")
    package = (tmp_path / "package").resolve()
    files = {"scripts/verify_public_package.py": (ROOT / "scripts/verify_public_package.py").read_bytes(),
             HELPER: helper_bytes,
             ENERGY_HELPER: (b"RECORD_DIRECTORY = 'structural-energy-records'\n"
                             b"def verify_energy_record_identity(records, source):\n"
                             b"    return {'structural_fixture': True}\n"),
             "payload.txt": b"bounded structural package fixture\n"}
    provenance = write_package(package, files)
    checked = verifier()
    assert checked.verify_manifest(package) == provenance
    return package, checked, provenance


def poison_bytecache(source, marker):
    original = source.read_bytes()
    timestamp = source.stat().st_mtime_ns
    malicious = marker_source(marker)
    if len(malicious) >= len(original):
        original += b"#" * (len(malicious) - len(original) + 100)
    source.write_bytes(malicious + b"#" * (len(original) - len(malicious)))
    os.utime(source, ns=(timestamp, timestamp))
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(source), cfile=str(cache), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_bytes(original)
    os.utime(source, ns=(timestamp, timestamp))
    return original


def test_importing_bootstrap_does_not_execute_a_sibling_dynamics_helper(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "verify_public_package.py").write_bytes((ROOT / "scripts/verify_public_package.py").read_bytes())
    marker = tmp_path / "eager-import-marker.txt"
    (scripts / "dynamics_record_release.py").write_bytes(marker_source(marker))
    body = """
from pathlib import Path
import importlib.util
import sys
source, marker = Path(sys.argv[1]), Path(sys.argv[2])
before = list(sys.path)
spec = importlib.util.spec_from_file_location('isolated_dynamics_bootstrap', source)
module = importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
assert sys.path == before
assert 'dynamics_record_release' not in sys.modules
assert not marker.exists()
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", textwrap.dedent(body),
                             str(scripts / "verify_public_package.py"), str(marker)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_failure_precedes_all_dynamics_helper_execution(tmp_path, monkeypatch):
    marker = tmp_path / "manifest-import-marker.txt"
    package, checked, _ = helper_package(tmp_path, marker_source(marker))
    (package / "payload.txt").write_bytes(b"changed unverified input\n")
    def forbidden(*args, **kwargs):
        raise AssertionError("Manifest failure reached a downstream helper")
    monkeypatch.setattr(checked, "verify_formal_record", forbidden)
    monkeypatch.setattr(checked, "_load_energy_record_gate", forbidden)
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", forbidden)
    with pytest.raises(ValueError, match="digest mismatch"):
        checked.verify_package(package, run_tests=False)
    assert not marker.exists()


def test_high_level_dynamics_gate_follows_the_complete_manifest_check(tmp_path, monkeypatch):
    helper = (b"RECORD_DIRECTORY = 'structural-dynamics-records'\n"
              b"def verify_dynamics_record_identity(records, source):\n"
              b"    raise RuntimeError('authenticated structural dynamics gate invoked')\n")
    package, checked, provenance = helper_package(tmp_path, helper)
    order = []
    real_manifest, real_energy, real_dynamics = (
        checked.verify_manifest, checked._load_energy_record_gate, checked._load_dynamics_record_gate)
    def manifest(path):
        result = real_manifest(path)
        order.append("manifest")
        return result
    def energy(path, identity):
        assert identity == provenance and order == ["manifest", "formal"]
        order.append("energy")
        return real_energy(path, identity)
    def dynamics(path, identity):
        assert identity == provenance and order == ["manifest", "formal", "energy"]
        order.append("dynamics")
        return real_dynamics(path, identity)
    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda path: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", energy)
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", dynamics)
    with pytest.raises(RuntimeError, match="authenticated structural dynamics gate invoked"):
        checked.verify_package(package, run_tests=False)
    assert order == ["manifest", "formal", "energy", "dynamics"]


@pytest.mark.parametrize("where", ["root", "root-ancestor", "helper", "helper-ancestor"])
def test_dynamics_loader_rejects_linked_paths_before_execution(tmp_path, where):
    marker = tmp_path / "linked-dynamics-marker.txt"
    package, checked, provenance = helper_package(tmp_path, marker_source(marker))
    if where == "root":
        alias = tmp_path / "package-alias"
        alias.symlink_to(package, target_is_directory=True)
        package = alias
    elif where == "root-ancestor":
        alias = tmp_path / "parent-alias"
        alias.symlink_to(package.parent, target_is_directory=True)
        package = alias / package.name
    elif where == "helper":
        source = package / HELPER
        original = tmp_path / "helper-original.py"
        source.rename(original)
        source.symlink_to(original)
    else:
        scripts = package / "scripts"
        original = package / "original-scripts"
        scripts.rename(original)
        scripts.symlink_to(original.name, target_is_directory=True)
    with pytest.raises(ValueError, match="(?i)linked|unsafe|regular"):
        checked._load_dynamics_record_gate(package, provenance)
    assert not marker.exists()


def test_changed_dynamics_helper_is_rehashed_before_compilation(tmp_path):
    package, checked, provenance = helper_package(tmp_path)
    marker = tmp_path / "changed-dynamics-marker.txt"
    (package / HELPER).write_bytes(marker_source(marker))
    with pytest.raises(ValueError, match="digest|hash|differ"):
        checked._load_dynamics_record_gate(package, provenance)
    assert not marker.exists()


def test_dynamics_loader_compiles_the_single_source_snapshot_it_checked(tmp_path, monkeypatch):
    package, checked, provenance = helper_package(tmp_path)
    source = package / HELPER
    original = source.read_bytes()
    marker = tmp_path / "later-dynamics-read-marker.txt"
    reads = []
    real_read = Path.read_bytes
    def changing_read(path):
        if path == source:
            reads.append(path)
            return original if len(reads) == 1 else marker_source(marker)
        return real_read(path)
    monkeypatch.setattr(Path, "read_bytes", changing_read)
    gate = checked._load_dynamics_record_gate(package, provenance)
    assert gate.RECORD_DIRECTORY == "assets/generated/dynamics-resources"
    assert reads == [source]
    assert not marker.exists()


def test_dynamics_module_and_bytecode_caches_cannot_supply_authority(tmp_path, monkeypatch):
    package, checked, provenance = helper_package(tmp_path)
    marker = tmp_path / "cached-dynamics-marker.txt"
    source = package / HELPER
    original = poison_bytecache(source, marker)
    # A padded source is still bound through the complete manifest.
    if digest(original) != provenance["file_sha256"][HELPER]:
        hashes = provenance["file_sha256"]
        hashes[HELPER] = digest(original)
        provenance["source_sha256"][HELPER] = hashes[HELPER]
    # Even a bytecache included in this complete structural manifest is not
    # authority to execute the helper. The real exporter excludes bytecaches.
    cache = Path(importlib.util.cache_from_source(str(source)))
    provenance["file_sha256"][cache.relative_to(package).as_posix()] = digest(cache.read_bytes())
    (package / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    assert checked.verify_manifest(package) == provenance
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "dynamics_record_release.py").write_bytes(marker_source(marker))
    cached = types.ModuleType("dynamics_record_release")
    cached.RECORD_DIRECTORY = "wrong-records"
    cached.verify_dynamics_record_identity = lambda *args: {"unverified_cache": True}
    monkeypatch.setitem(sys.modules, "dynamics_record_release", cached)
    monkeypatch.setitem(sys.modules, "_openquantum_verified_dynamics_record_gate", cached)
    monkeypatch.syspath_prepend(str(shadow))
    before_path = list(sys.path)
    gate = checked._load_dynamics_record_gate(package, provenance)
    assert gate is not cached
    assert all(module is not gate for module in list(sys.modules.values()))
    assert Path(gate.__file__) == source
    assert gate.verify_dynamics_record_identity(None, None) == {"structural_fixture": True}
    assert sys.modules["dynamics_record_release"] is cached
    assert sys.path == before_path
    assert not marker.exists()


def test_binder_assembly_does_not_execute_unverified_dynamics_preview(tmp_path):
    site = (tmp_path / "site").resolve()
    marker = tmp_path / "preview-dynamics-marker.txt"
    files = {"scripts/verify_public_package.py": (ROOT / "scripts/verify_public_package.py").read_bytes(),
             HELPER: (b"RECORD_DIRECTORY = 'structural-dynamics-records'\n"
                      b"def verify_dynamics_record_identity(records, source):\n"
                      b"    return {'structural_fixture': True}\n"),
             "src/openquantum/__init__.py": b"# structural Binder fixture\n",
             "assets/generated/demo/results.json": b'{"complete":true}\n'}
    provenance = write_package(site / "code", files)
    provenance_bytes = (site / "code/PROVENANCE.json").read_bytes()
    archive = site / ARCHIVE
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in files.items(): zipped.writestr(PREFIX + name, content)
        zipped.writestr(PREFIX + "PROVENANCE.json", provenance_bytes)
    (site / "code" / HELPER).write_bytes(marker_source(marker))
    (site / "code/PREVIEW.json").write_text(json.dumps({
        "schema": "openquantum-source-preview-v1", "source_commit": provenance["source_commit"],
        "archive": {"path": ARCHIVE, "sha256": digest(archive.read_bytes())}}))
    assembly = load_source(ROOT / "binder/assemble_code.py", "dynamics_binder_assembly")
    result = assembly.assemble(site)
    assert result["file_count"] == len(files)
    assert (site / "code" / HELPER).read_bytes() == files[HELPER]
    assert not marker.exists()
