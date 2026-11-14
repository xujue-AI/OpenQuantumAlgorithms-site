"""Binder reconstructs a complete checked package before any installation."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import shutil
import stat
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("binder_archive_assembly", ROOT / "binder/assemble_code.py")
assembly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembly)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


def make_publication(tmp_path, *, preview=True):
    root = tmp_path / "site"
    root.mkdir()
    files = {
        "scripts/verify_public_package.py": (ROOT / "scripts/verify_public_package.py").read_bytes(),
        "scripts/energy_record_release.py": (ROOT / "scripts/energy_record_release.py").read_bytes(),
        "src/openquantum/__init__.py": b"# original test package\n",
        "requirements-lock.txt": b"# bounded fixture; no dependencies installed\n",
        "assets/generated/demo/results.json": b'{"checked":true}\n',
        "assets/generated/grover/grover.py": b"# standalone fixture\n",
        "assets/generated/grover/LICENSE.txt": b"MIT fixture\n",
    }
    manifest = {name: digest(content) for name, content in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": manifest,
                  "source_sha256": {name: sha for name, sha in manifest.items() if name.endswith(".py")}}
    provenance_bytes = json.dumps(provenance, sort_keys=True).encode()
    archive = root / assembly.ARCHIVE_PATH
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in files.items():
            zipped.writestr(assembly.ARCHIVE_PREFIX + name, content)
        zipped.writestr(assembly.ARCHIVE_PREFIX + "PROVENANCE.json", provenance_bytes)
    code = root / "code"
    code.mkdir()
    for name, content in files.items():
        if preview and name.startswith("assets/"):
            continue
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (code / "PROVENANCE.json").write_bytes(provenance_bytes)
    if preview:
        (code / "PREVIEW.json").write_text(json.dumps({
            "schema": assembly.PREVIEW_SCHEMA, "source_commit": provenance["source_commit"],
            "archive": {"path": assembly.ARCHIVE_PATH, "sha256": assembly.file_digest(archive)},
        }))
    (root / "requirements-lock.txt").write_bytes(files["requirements-lock.txt"])
    for name in ("grover.py", "LICENSE.txt"):
        target = root / "assets/generated/grover" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files["assets/generated/grover/" + name])
    (root / "notebooks").mkdir()
    return root, files, provenance_bytes


def refresh_archive_identity(root):
    path = root / "code/PREVIEW.json"
    preview = json.loads(path.read_text())
    preview["archive"]["sha256"] = assembly.file_digest(root / assembly.ARCHIVE_PATH)
    path.write_text(json.dumps(preview))


@pytest.mark.parametrize("preview", [False, True])
def test_complete_archive_is_reconstructed_and_manifest_verified(tmp_path, preview):
    root, files, provenance = make_publication(tmp_path, preview=preview)
    canonical_before = snapshot(root / "assets")
    result = assembly.assemble(root)
    assert result == {"source_commit": "a" * 40, "file_count": len(files)}
    assert snapshot(root / "code") == {**files, "PROVENANCE.json": provenance}
    assert snapshot(root / "assets") == canonical_before
    assert not list(root.glob(".binder-code-assembly-*"))


@pytest.mark.parametrize("mutation", ["digest", "source_commit", "schema", "absolute_path", "traversal_path", "object"])
def test_preview_identity_failure_preserves_original_tree(tmp_path, mutation):
    root, _, _ = make_publication(tmp_path)
    path = root / "code/PREVIEW.json"
    preview = json.loads(path.read_text())
    if mutation == "digest":
        preview["archive"]["sha256"] = "0" * 64
    elif mutation == "source_commit":
        preview["source_commit"] = "b" * 40
    elif mutation == "schema":
        preview["schema"] = "unknown"
    elif mutation == "absolute_path":
        preview["archive"]["path"] = "/tmp/untrusted.zip"
    elif mutation == "traversal_path":
        preview["archive"]["path"] = "../untrusted.zip"
    else:
        preview = []
    path.write_text(json.dumps(preview))
    before = snapshot(root / "code")
    with pytest.raises(ValueError):
        assembly.assemble(root)
    assert snapshot(root / "code") == before


def test_bootstrap_verifier_modification_rejects_before_import(tmp_path):
    root, _, _ = make_publication(tmp_path)
    (root / "code/scripts/verify_public_package.py").write_text("raise AssertionError('modified verifier executed')\n")
    before = snapshot(root / "code")
    with pytest.raises(ValueError, match="bootstrap verifier"):
        assembly.assemble(root)
    assert snapshot(root / "code") == before


def test_bootstrap_uses_checked_source_instead_of_an_unbound_bytecache(tmp_path):
    root, files, _ = make_publication(tmp_path)
    source = root / "code/scripts/verify_public_package.py"
    original = source.read_bytes()
    timestamp = source.stat().st_mtime_ns
    malicious = b"raise AssertionError('unbound bytecache executed')\n"
    source.write_bytes(malicious + b"#" * (len(original) - len(malicious)))
    os.utime(source, ns=(timestamp, timestamp))
    cache = Path(importlib.util.cache_from_source(str(source)))
    cache.parent.mkdir()
    py_compile.compile(str(source), cfile=str(cache), doraise=True)
    source.write_bytes(original)
    os.utime(source, ns=(timestamp, timestamp))
    result = assembly.assemble(root)
    assert result["file_count"] == len(files)
    assert not (root / "code/scripts/__pycache__").exists()


@pytest.mark.parametrize("mutation", ["provenance", "asset", "traversal", "symlink", "duplicate"])
def test_invalid_archive_never_replaces_preview(tmp_path, mutation):
    root, _, _ = make_publication(tmp_path)
    archive = root / assembly.ARCHIVE_PATH
    with zipfile.ZipFile(archive) as zipped:
        members = {name: zipped.read(name) for name in zipped.namelist()}
    if mutation == "provenance":
        name = assembly.ARCHIVE_PREFIX + "PROVENANCE.json"
        provenance = json.loads(members[name])
        provenance["source_commit"] = "b" * 40
        members[name] = json.dumps(provenance).encode()
    elif mutation == "asset":
        members[assembly.ARCHIVE_PREFIX + "assets/generated/demo/results.json"] = b'{"checked":false}\n'
    elif mutation == "traversal":
        members[assembly.ARCHIVE_PREFIX + "../../outside.txt"] = b"unsafe"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            zipped.writestr(name, content)
        if mutation == "symlink":
            linked = zipfile.ZipInfo(assembly.ARCHIVE_PREFIX + "linked-file")
            linked.create_system = 3
            linked.external_attr = 0o120777 << 16
            zipped.writestr(linked, "../outside")
        elif mutation == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                name = assembly.ARCHIVE_PREFIX + "assets/generated/demo/results.json"
                zipped.writestr(name, members[name])
    refresh_archive_identity(root)
    before = snapshot(root / "code")
    with pytest.raises(ValueError):
        assembly.assemble(root)
    assert snapshot(root / "code") == before
    assert not (tmp_path / "outside.txt").exists()
    assert not list(root.glob(".binder-code-assembly-*"))


@pytest.mark.parametrize("mutation", ["collapsed_extra", "collapsed_replacement", "extra", "directory", "fifo", "socket", "device"])
def test_exact_regular_member_audit_rejects_aliases_and_unrecorded_modes(tmp_path, mutation, monkeypatch):
    root, _, _ = make_publication(tmp_path)
    archive = root / assembly.ARCHIVE_PATH
    with zipfile.ZipFile(archive) as zipped:
        members = {name: zipped.read(name) for name in zipped.namelist()}
    source_name = assembly.ARCHIVE_PREFIX + "src/openquantum/__init__.py"
    if mutation == "collapsed_replacement":
        content = members.pop(source_name)
        members[assembly.ARCHIVE_PREFIX + "src/./openquantum/__init__.py"] = content
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, content in members.items():
            if mutation in {"fifo", "socket", "device"} and name == source_name:
                entry = zipfile.ZipInfo(name)
                entry.create_system = 3
                mode = {"fifo": stat.S_IFIFO, "socket": stat.S_IFSOCK, "device": stat.S_IFCHR}[mutation]
                entry.external_attr = (mode | 0o600) << 16
                zipped.writestr(entry, content)
            else:
                zipped.writestr(name, content)
        if mutation == "collapsed_extra":
            zipped.writestr(assembly.ARCHIVE_PREFIX + "src/./openquantum/__init__.py", members[source_name])
        elif mutation == "extra":
            zipped.writestr(assembly.ARCHIVE_PREFIX + "unrecorded.txt", b"extra")
        elif mutation == "directory":
            zipped.writestr(assembly.ARCHIVE_PREFIX + "empty/", b"")
    refresh_archive_identity(root)
    before = snapshot(root / "code")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid ZIP membership/mode reached the bootstrap verifier")
    monkeypatch.setattr(assembly, "load_verifier", forbidden)
    with pytest.raises(ValueError, match="members differ|regular file"):
        assembly.assemble(root)
    assert snapshot(root / "code") == before
    assert not list(root.glob(".binder-code-assembly-*"))


@pytest.mark.parametrize("linked_target", ["archive", "archive_parent", "code", "verifier_parent"])
def test_linked_paths_reject_without_replacement(tmp_path, linked_target):
    root, _, _ = make_publication(tmp_path)
    target = {"archive": root / assembly.ARCHIVE_PATH,
              "archive_parent": root / "assets/generated",
              "code": root / "code",
              "verifier_parent": root / "code/scripts"}[linked_target]
    saved = target.with_name(target.name + "-original")
    target.rename(saved)
    target.symlink_to(saved.name, target_is_directory=saved.is_dir())
    with pytest.raises(ValueError, match="regular directory|Linked"):
        assembly.assemble(root)
    assert target.is_symlink()


def test_rename_failure_restores_complete_original_preview(tmp_path, monkeypatch):
    root, _, _ = make_publication(tmp_path)
    before = snapshot(root / "code")
    original_rename = Path.rename
    def failing_rename(path, target):
        if path.name == "openquantum-examples" and Path(target) == root / "code":
            raise OSError("bounded simulated rename failure")
        return original_rename(path, target)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        assembly.assemble(root)
    assert snapshot(root / "code") == before
    assert not list(root.glob(".binder-code-assembly-*"))


def test_failed_rollback_keeps_original_preview_in_recoverable_staging(tmp_path, monkeypatch):
    root, _, _ = make_publication(tmp_path)
    before = snapshot(root / "code")
    original_rename = Path.rename
    def failing_rename(path, target):
        if path.name in {"openquantum-examples", "previous-code"} and Path(target) == root / "code":
            raise OSError("bounded simulated replacement/rollback failure")
        return original_rename(path, target)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(RuntimeError, match="original code remains in staging"):
        assembly.assemble(root)
    staging = list(root.glob(".binder-code-assembly-*"))
    assert len(staging) == 1
    assert snapshot(staging[0] / "previous-code") == before
    assert not (root / "code").exists()


@pytest.mark.parametrize("valid", [False, True])
def test_postbuild_assembles_before_dependency_installation(tmp_path, valid):
    root, files, _ = make_publication(tmp_path)
    binder = root / "binder"
    binder.mkdir()
    for name in ("postBuild", "assemble_code.py"):
        shutil.copy2(ROOT / "binder" / name, binder / name)
    if not valid:
        preview_path = root / "code/PREVIEW.json"
        preview = json.loads(preview_path.read_text())
        preview["archive"]["sha256"] = "0" * 64
        preview_path.write_text(json.dumps(preview))
    before = snapshot(root / "code")
    commands = root / "test-commands"
    commands.mkdir()
    python_shim = commands / "python"
    python_shim.write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\n'
        'if [[ "$1" == "binder/assemble_code.py" ]]; then\n'
        '  exec "$BINDER_TEST_PYTHON" "$@"\n'
        'fi\n'
        '[[ "$1" == "-m" && "$2" == "pip" ]]\n'
        '[[ -f code/assets/generated/demo/results.json && ! -e code/PREVIEW.json ]]\n'
        'printf "%s\\n" "$*" >> "$BINDER_TEST_LOG"\n'
    )
    python_shim.chmod(0o755)
    install_shim = commands / "install"
    install_shim.write_text('#!/usr/bin/env bash\nprintf "standalone-copy\\n" >> "$BINDER_TEST_LOG"\n')
    install_shim.chmod(0o755)
    log = root / "install.log"
    environment = {**os.environ, "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                   "BINDER_TEST_PYTHON": sys.executable, "BINDER_TEST_LOG": str(log),
                   "PYTHONDONTWRITEBYTECODE": "1"}
    syntax = subprocess.run(["/bin/bash", "-n", str(binder / "postBuild")], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    result = subprocess.run(["/bin/bash", str(binder / "postBuild")], cwd=root,
                            env=environment, capture_output=True, text=True)
    if valid:
        assert result.returncode == 0, result.stderr
        lines = log.read_text().splitlines()
        assert lines[0] == "-m pip install -r requirements-lock.txt"
        assert lines[-1] == "-m pip install --no-deps ./code"
        assert len(lines) == 4
        assert (root / "code/assets/generated/demo/results.json").read_bytes() == files["assets/generated/demo/results.json"]
    else:
        assert result.returncode != 0
        assert not log.exists()
        assert snapshot(root / "code") == before
