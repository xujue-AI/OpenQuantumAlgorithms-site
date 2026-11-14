"""Bootstrap hashing stays independent of unverified project imports."""
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
HELPER = "scripts/energy_record_release.py"
ARCHIVE = "assets/generated/openquantum-code.zip"
PREFIX = "openquantum-examples/"


def load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def verifier():
    return load_source(ROOT / "scripts/verify_public_package.py", "bootstrap_public_verifier")


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_package(directory, files):
    directory.mkdir(parents=True)
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = {name: digest(content) for name, content in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": manifest,
                  "source_sha256": {name: sha for name, sha in manifest.items() if name.endswith(".py")}}
    (directory / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def publication(tmp_path, *, helper=True):
    """A complete structural ZIP; assembly performs no scientific workflow."""
    site = (tmp_path / "site").resolve()
    files = {"scripts/verify_public_package.py": (ROOT / "scripts/verify_public_package.py").read_bytes(),
             "src/openquantum/__init__.py": b"# bounded bootstrap fixture\n",
             "assets/generated/demo/results.json": b'{"complete":true}\n'}
    if helper:
        files[HELPER] = (ROOT / HELPER).read_bytes()
    code = site / "code"
    provenance = write_package(code, files)
    provenance_bytes = (code / "PROVENANCE.json").read_bytes()
    archive = site / ARCHIVE
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(PREFIX + name, content)
        zipped.writestr(PREFIX + "PROVENANCE.json", provenance_bytes)
    for path in (code / "assets").rglob("*"):
        if path.is_file():
            path.unlink()
    for path in sorted((code / "assets").rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    (code / "assets").rmdir()
    (code / "PREVIEW.json").write_text(json.dumps({
        "schema": "openquantum-source-preview-v1", "source_commit": provenance["source_commit"],
        "archive": {"path": ARCHIVE, "sha256": digest(archive.read_bytes())},
    }))
    return site, files


def marker_source(marker):
    return f"from pathlib import Path\nPath({str(marker)!r}).write_text('unverified helper executed')\n".encode()


def replace_archive_helper(site, content):
    archive = site / ARCHIVE
    with zipfile.ZipFile(archive) as zipped:
        members = {name: zipped.read(name) for name in zipped.namelist()}
    members[PREFIX + HELPER] = content
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    preview_path = site / "code/PREVIEW.json"
    preview = json.loads(preview_path.read_text())
    preview["archive"]["sha256"] = digest(archive.read_bytes())
    preview_path.write_text(json.dumps(preview))


def run_isolated(body, site, *arguments):
    prelude = """
import importlib.util
from pathlib import Path
import sys

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module

source_root = Path(sys.argv[1])
site = Path(sys.argv[2])
assembly = load(source_root / 'binder/assemble_code.py', 'isolated_binder_assembly')
"""
    result = subprocess.run([sys.executable, "-I", "-B", "-c", textwrap.dedent(prelude + body),
                             str(ROOT), str(site), *map(str, arguments)],
                            cwd=site, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("helper", [False, True])
def test_binder_bootstrap_does_not_import_the_energy_dependency(tmp_path, helper):
    site, files = publication(tmp_path, helper=helper)
    run_isolated("""
before_path = list(sys.path)
assert 'energy_record_release' not in sys.modules
result = assembly.assemble(site)
assert result['file_count'] > 0
assert 'energy_record_release' not in sys.modules
assert sys.path == before_path
""", site)
    assert (site / "code/assets/generated/demo/results.json").read_bytes() == files["assets/generated/demo/results.json"]


@pytest.mark.parametrize("recorded_helper", [False, True])
def test_unverified_preview_helper_never_executes_during_actual_assembly(tmp_path, recorded_helper):
    site, files = publication(tmp_path, helper=recorded_helper)
    marker = tmp_path / "preview-import-marker.txt"
    (site / "code" / HELPER).write_bytes(marker_source(marker))
    run_isolated("""
marker = Path(sys.argv[3])
assembly.assemble(site)
assert not marker.exists()
assert 'energy_record_release' not in sys.modules
""", site, marker)
    if recorded_helper:
        assert (site / "code" / HELPER).read_bytes() == files[HELPER]
    else:
        assert not (site / "code" / HELPER).exists()


def test_changed_manifest_member_rejects_before_any_helper_execution(tmp_path):
    site, _ = publication(tmp_path)
    marker = tmp_path / "archive-import-marker.txt"
    content = marker_source(marker)
    (site / "code" / HELPER).write_bytes(content)
    replace_archive_helper(site, content)
    before = {path.relative_to(site / "code").as_posix(): path.read_bytes()
              for path in (site / "code").rglob("*") if path.is_file()}
    run_isolated("""
marker = Path(sys.argv[3])
try:
    assembly.assemble(site)
except ValueError as error:
    assert 'digest mismatch' in str(error)
else:
    raise AssertionError('changed manifest member was accepted')
assert not marker.exists()
assert 'energy_record_release' not in sys.modules
""", site, marker)
    assert {path.relative_to(site / "code").as_posix(): path.read_bytes()
            for path in (site / "code").rglob("*") if path.is_file()} == before


def test_cached_module_and_shadow_path_cannot_supply_bootstrap_authority(tmp_path):
    site, _ = publication(tmp_path)
    marker = tmp_path / "shadow-import-marker.txt"
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "energy_record_release.py").write_bytes(marker_source(marker))
    run_isolated("""
import json
import types
marker, shadow = Path(sys.argv[3]), Path(sys.argv[4])
cached = types.ModuleType('energy_record_release')
def wrong_directory(value):
    raise AssertionError('cached helper directory function was used')
cached.checked_directory = wrong_directory
cached.RECORD_DIRECTORY = 'cached-records'
cached.verify_energy_record_identity = wrong_directory
sys.modules['energy_record_release'] = cached
sys.path.insert(0, str(shadow))
before_path = list(sys.path)
provenance = json.loads((site / 'code/PROVENANCE.json').read_text())
checked = assembly.load_verifier(site / 'code', provenance, site / assembly.ARCHIVE_PATH)
assert checked.checked_directory is not cached.checked_directory
assembly.assemble(site)
assert sys.modules['energy_record_release'] is cached
assert sys.path == before_path
assert not marker.exists()
""", site, marker, shadow)


def poison_bytecache(source, marker):
    original = source.read_bytes()
    timestamp = source.stat().st_mtime_ns
    malicious = marker_source(marker)
    assert len(malicious) < len(original)
    source.write_bytes(malicious + b"#" * (len(original) - len(malicious)))
    os.utime(source, ns=(timestamp, timestamp))
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    py_compile.compile(str(source), cfile=str(cache), doraise=True,
                       invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    source.write_bytes(original)
    os.utime(source, ns=(timestamp, timestamp))
    return cache


def test_energy_helper_bytecache_cannot_execute_during_bootstrap(tmp_path):
    site, _ = publication(tmp_path)
    marker = tmp_path / "bootstrap-bytecache-marker.txt"
    poison_bytecache(site / "code" / HELPER, marker)
    run_isolated("""
marker = Path(sys.argv[3])
assembly.assemble(site)
assert not marker.exists()
assert 'energy_record_release' not in sys.modules
assert not list((site / 'code').rglob('__pycache__'))
""", site, marker)


def authenticated_helper_package(tmp_path, *, helper_bytes=None):
    package = (tmp_path / "package").resolve()
    write_package(package, {HELPER: (ROOT / HELPER).read_bytes() if helper_bytes is None else helper_bytes})
    checked = verifier()
    return package, checked, checked.verify_manifest(package)


def test_high_level_manifest_failure_precedes_dependency_loading(tmp_path, monkeypatch):
    package, checked, _ = authenticated_helper_package(tmp_path)
    marker = tmp_path / "high-level-import-marker.txt"
    (package / HELPER).write_bytes(marker_source(marker))
    def forbidden(*args, **kwargs):
        raise AssertionError("manifest failure reached a downstream gate")
    monkeypatch.setattr(checked, "_load_energy_record_gate", forbidden)
    monkeypatch.setattr(checked, "verify_formal_record", forbidden)
    with pytest.raises(ValueError, match="digest mismatch"):
        checked.verify_package(package, run_tests=False)
    assert not marker.exists()


def test_high_level_verification_keeps_the_energy_gate_after_manifest_check(tmp_path, monkeypatch):
    helper = (b"RECORD_DIRECTORY = 'structural-records'\n"
              b"def verify_energy_record_identity(records, source):\n"
              b"    raise RuntimeError('authenticated structural energy gate invoked')\n")
    package, checked, provenance = authenticated_helper_package(tmp_path, helper_bytes=helper)
    order = []
    real_manifest = checked.verify_manifest
    real_loader = checked._load_energy_record_gate
    def manifest(path):
        result = real_manifest(path)
        order.append("manifest")
        return result
    def load(path, identity):
        assert identity == provenance
        assert order == ["manifest", "formal"]
        order.append("energy")
        return real_loader(path, identity)
    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda path: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", load)
    with pytest.raises(RuntimeError, match="authenticated structural energy gate invoked"):
        checked.verify_package(package, run_tests=False)
    assert order == ["manifest", "formal", "energy"]


@pytest.mark.parametrize("where", ["root", "root-ancestor", "helper", "helper-ancestor"])
def test_authenticated_loader_rejects_lexical_links_before_execution(tmp_path, where):
    marker = tmp_path / "linked-helper-marker.txt"
    package, checked, provenance = authenticated_helper_package(tmp_path, helper_bytes=marker_source(marker))
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
        checked._load_energy_record_gate(package, provenance)
    assert not marker.exists()


def test_authenticated_loader_rehashes_a_changed_helper_before_compiling(tmp_path):
    package, checked, provenance = authenticated_helper_package(tmp_path)
    marker = tmp_path / "changed-helper-marker.txt"
    (package / HELPER).write_bytes(marker_source(marker))
    with pytest.raises(ValueError, match="digest|hash|differ"):
        checked._load_energy_record_gate(package, provenance)
    assert not marker.exists()


def test_authenticated_loader_compiles_the_same_single_snapshot_it_hashed(tmp_path, monkeypatch):
    package, checked, provenance = authenticated_helper_package(tmp_path)
    source = package / HELPER
    original = source.read_bytes()
    marker = tmp_path / "later-read-marker.txt"
    reads = []
    real_read = Path.read_bytes
    def changing_read(path):
        if path == source:
            reads.append(path)
            return original if len(reads) == 1 else marker_source(marker)
        return real_read(path)
    monkeypatch.setattr(Path, "read_bytes", changing_read)
    gate = checked._load_energy_record_gate(package, provenance)
    assert gate.RECORD_DIRECTORY == "assets/generated/energy-estimation"
    assert reads == [source]
    assert not marker.exists()


def test_genuine_energy_gate_is_fresh_despite_module_path_and_bytecode_contamination(tmp_path, monkeypatch):
    package, checked, provenance = authenticated_helper_package(tmp_path)
    marker = tmp_path / "authenticated-bytecache-marker.txt"
    poison_bytecache(package / HELPER, marker)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "energy_record_release.py").write_bytes(marker_source(marker))
    cached = types.ModuleType("energy_record_release")
    cached.RECORD_DIRECTORY = "wrong-records"
    cached.verify_energy_record_identity = lambda *args, **kwargs: {"unverified_cache": True}
    monkeypatch.setitem(sys.modules, "energy_record_release", cached)
    monkeypatch.syspath_prepend(str(shadow))
    before_path = list(sys.path)
    gate = checked._load_energy_record_gate(package, provenance)
    assert gate is not cached
    assert all(module is not gate for module in list(sys.modules.values()))
    assert Path(gate.__file__) == package / HELPER
    result = gate.verify_energy_record_identity(ROOT / gate.RECORD_DIRECTORY, ROOT)
    assert result["core_files"] == 35
    assert result["manifest_sha256"] == gate.ENERGY_RECORD_MANIFEST_SHA256
    assert result["logical_producer_executed"] is False
    assert result["native_reestimation_performed"] is False
    assert result["factory_search_repeated"] is False
    assert sys.modules["energy_record_release"] is cached
    assert sys.path == before_path
    assert not marker.exists()
