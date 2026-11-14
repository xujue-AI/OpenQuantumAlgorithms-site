"""Immutable public wrapper regressions; saved data and mocks only."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/optimization_word_record_release.py"
CORE = ROOT / "scripts/optimization_word_records.py"
SCHEMA = ROOT / "scripts/optimization_word_record_schema.py"
BUNDLE = ROOT / "assets/generated/optimization/word-transport"
PRODUCERS = {"optimization_word_record_schema.py", "optimization_word_native.py",
             "optimization_word_accounting.py", "optimization_word_transport.py",
             "optimization_compiled_codec.py"}


def load_wrapper(name="word_release_wrapper_test", raw=None):
    source = WRAPPER.read_bytes() if raw is None else raw
    spec = importlib.util.spec_from_file_location(name, WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, str(WRAPPER), "exec"), module.__dict__)
    return module


@pytest.fixture
def release():
    return load_wrapper()


@pytest.fixture(scope="module", autouse=True)
def preserve_implementation_and_snapshot_bytes():
    paths = [WRAPPER, CORE, SCHEMA, *sorted(path for path in BUNDLE.rglob("*") if path.is_file())]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    names = {path.relative_to(BUNDLE).as_posix() for path in paths if path.is_relative_to(BUNDLE)}
    yield
    assert {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*") if path.is_file()} == names
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


def test_import_is_standard_library_only_and_performs_no_content_reads(monkeypatch):
    source = WRAPPER.read_bytes()
    compiled = compile(source, str(WRAPPER), "exec")
    spec = importlib.util.spec_from_file_location("word_wrapper_readless_import", WRAPPER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    original_import = builtins.__import__
    imports = []

    def guarded_import(name, *args, **kwargs):
        imports.append(name)
        assert name.split(".")[0] in sys.stdlib_module_names | {"__future__"}
        return original_import(name, *args, **kwargs)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Wrapper import must not read data or project source")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "fdopen", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    exec(compiled, module.__dict__)
    assert imports and all(name.split(".")[0] not in {"openquantum", "qdk", "qiskit"} for name in imports)
    assert isinstance(module.MANIFEST_SHA256, str) and len(module.MANIFEST_SHA256) == 64


def test_wrapper_literal_pin_matches_reviewed_snapshot(release):
    assert hashlib.sha256((BUNDLE / "record-manifest.json").read_bytes()).hexdigest() == release.MANIFEST_SHA256
    tree = ast.parse(WRAPPER.read_text())
    literal = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "MANIFEST_SHA256" for target in node.targets)
                   and isinstance(node.value, ast.Constant))
    assert release.MANIFEST_SHA256 == literal


@pytest.mark.parametrize("mutation", ["manifest", "core", "missing_core"])
def test_manifest_and_core_rejection_precedes_any_compilation(tmp_path, release, monkeypatch, mutation):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    marker = tmp_path.resolve() / "untrusted-executed"
    if mutation == "manifest":
        path = destination / "record-manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "core":
        (destination / release.CORE).write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
    else:
        (destination / release.CORE).unlink()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Unauthenticated core was compiled")

    monkeypatch.setattr(builtins, "compile", forbidden)
    with pytest.raises((ValueError, OSError)):
        release.authenticate_public_word_transport(destination)
    assert not marker.exists()


def test_rehashed_manifest_does_not_replace_the_literal_trust_pin(tmp_path, release):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    manifest = json.loads((destination / "record-manifest.json").read_text())
    manifest["study"]["preparations"] += 1
    (destination / "record-manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    with pytest.raises(ValueError, match="manifest changed"):
        release.authenticate_public_word_transport(destination)


@pytest.mark.parametrize("mutation", ["missing", "tampered", "extra"])
def test_complete_inventory_rejection_precedes_producer_or_schema_compilation(tmp_path, release, monkeypatch, mutation):
    destination = tmp_path.resolve() / "bundle"
    shutil.copytree(BUNDLE, destination)
    if mutation == "missing":
        (destination / "source-recipe.json").unlink()
    elif mutation == "tampered":
        path = destination / "native-result.json"
        path.write_bytes(path.read_bytes() + b"x")
    else:
        (destination / "unlisted.txt").write_text("not an input")
    original_compile = builtins.compile
    compiled = []

    def guarded_compile(source, filename, *args, **kwargs):
        name = Path(str(filename)).name
        if name in PRODUCERS:
            raise AssertionError("Producer/schema compiled before complete inventory authentication")
        if name == "optimization_word_records.py":
            compiled.append(name)
        return original_compile(source, filename, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", guarded_compile)
    with pytest.raises((ValueError, OSError)):
        release.verify_public_word_transport(destination)
    assert compiled == ["optimization_word_records.py"]


def test_saved_dispatch_uses_the_authenticated_pin_and_exact_replay_choice(release, monkeypatch):
    events = []
    core = SimpleNamespace(verify_records=lambda bundle, pin, **kwargs:
                           events.append((bundle, pin, kwargs)) or {"status": "saved-sentinel"})
    monkeypatch.setattr(release, "_core", lambda root: (Path("owned-bundle"), core))
    assert release.verify_public_word_transport(Path("caller-root"), replay=False) == {"status": "saved-sentinel"}
    assert events == [(Path("owned-bundle"), release.MANIFEST_SHA256, {"replay": False})]


@pytest.mark.parametrize("loader", ["plan", "sources"])
def test_fixture_loaders_authenticate_full_bundle_before_schema_dispatch(release, monkeypatch, loader):
    events = []
    authenticated = SimpleNamespace(json=lambda name: events.append(("json", name)) or {"file": name})

    def authenticate(bundle, pin):
        assert pin == release.MANIFEST_SHA256
        events.append(("authenticated", str(bundle)))
        return authenticated

    def source_module(value, name):
        assert value is authenticated and events[0][0] == "authenticated"
        events.append(("compiled", name))
        return SimpleNamespace(restore_record=lambda item: events.append(("restored", item["file"])) or item)

    core = SimpleNamespace(authenticate_bundle=authenticate, _module=source_module)
    monkeypatch.setattr(release, "_core", lambda root: (Path("owned-bundle"), core))
    result = release.load_recorded_plan() if loader == "plan" else release.load_source_records()
    assert events[0] == ("authenticated", "owned-bundle") and events[1] == ("compiled", "schema")
    if loader == "plan":
        assert result == {"file": "native-plan.json"}
    else:
        assert set(result) == {"source_recipe", "source_accounting", "source_validation", "source_artifact_manifest"}
        assert len([event for event in events if event[0] == "restored"]) == 4


def test_real_saved_dispatch_authenticates_before_any_producer_compilation(release, monkeypatch):
    bundle, core = release._core(BUNDLE)
    original_authenticate = core.authenticate_bundle
    events = []

    def authenticate(*args, **kwargs):
        events.append("authentication-start")
        value = original_authenticate(*args, **kwargs)
        events.append("authentication-complete")
        return value

    monkeypatch.setattr(core, "authenticate_bundle", authenticate)
    monkeypatch.setattr(release, "_core", lambda root: (bundle, core))
    original_compile = builtins.compile
    original_import = builtins.__import__

    def guarded_compile(source, filename, *args, **kwargs):
        name = Path(str(filename)).name
        if name in PRODUCERS:
            assert "authentication-complete" in events
            events.append("compiled-" + name)
        return original_compile(source, filename, *args, **kwargs)

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"qdk", "qiskit", "numpy", "scipy", "sympy"}:
            raise AssertionError("Release dispatch must stay saved-only and standard-library-only")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "compile", guarded_compile)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    result = release.verify_public_word_transport(BUNDLE)
    assert result["status"] == "verified_saved_word_native_model" and result["native_api_calls"] == 0
    assert result["transcript_id"] == "qaoa-cycle-4-p2-alternate-seed-7"
    assert events.index("authentication-complete") < min(index for index, event in enumerate(events) if event.startswith("compiled-"))


@pytest.mark.parametrize("relative", ["../escape.py", "/absolute.py"])
def test_wrapper_relative_paths_cannot_escape(tmp_path, release, relative):
    with pytest.raises(ValueError):
        release._read(tmp_path.resolve(), relative, 100)


@pytest.mark.parametrize("kind", ["root", "file", "ancestor", "hardlink", "directory", "missing", "oversized"])
def test_wrapper_rejects_redirected_or_invalid_source_paths(tmp_path, release, kind):
    root = tmp_path.resolve() / "root"
    root.mkdir()
    source = root / "source.py"
    source.write_bytes(b"safe")
    relative, maximum = "source.py", 100
    if kind == "root":
        linked = tmp_path.resolve() / "link"
        linked.symlink_to(root, target_is_directory=True)
        root = linked
    elif kind in {"file", "hardlink"}:
        outside = tmp_path.resolve() / "outside.py"
        outside.write_bytes(b"safe")
        source.unlink()
        source.symlink_to(outside) if kind == "file" else source.hardlink_to(outside)
    elif kind == "ancestor":
        outside = tmp_path.resolve() / "outside"
        outside.mkdir()
        (outside / "source.py").write_bytes(b"safe")
        (root / "linked").symlink_to(outside, target_is_directory=True)
        relative = "linked/source.py"
    elif kind == "directory":
        source.unlink()
        source.mkdir()
    elif kind == "missing":
        source.unlink()
    else:
        maximum = 3
    with pytest.raises((ValueError, OSError)):
        release._read(root, relative, maximum)


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO regression")
@pytest.mark.parametrize("replacement_race", [False, True])
def test_wrapper_fifo_open_is_bounded_before_and_during_replacement(tmp_path, replacement_race):
    root = tmp_path.resolve() / "root"
    root.mkdir()
    source = root / "source.py"
    source.write_bytes(b"safe")
    if not replacement_race:
        source.unlink()
        os.mkfifo(source)
    command = r'''
import importlib.util,sys,os
from pathlib import Path
s=importlib.util.spec_from_file_location("bounded_word_release",sys.argv[1]);m=importlib.util.module_from_spec(s)
sys.modules[s.name]=m;exec(compile(Path(sys.argv[1]).read_bytes(),sys.argv[1],"exec"),m.__dict__)
root=Path(sys.argv[2]);race=sys.argv[3]=="True";original=m.os.open;changed=False
def opened(path,flags,*args,**kwargs):
    global changed
    if race and not changed and Path(path)==root/"source.py":
        changed=True;(root/"source.py").unlink();os.mkfifo(root/"source.py")
    return original(path,flags,*args,**kwargs)
m.os.open=opened
try:m._read(root,"source.py",100)
except (ValueError,OSError):print("bounded-rejection");sys.exit(0)
raise AssertionError("nonregular source accepted")
'''
    result = subprocess.run([sys.executable, "-c", command, str(WRAPPER), str(root), str(replacement_race)],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded-rejection"
