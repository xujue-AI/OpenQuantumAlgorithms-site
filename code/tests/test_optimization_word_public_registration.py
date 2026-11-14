"""Outer saved-record trust boundaries, using only synthetic public fixtures."""
from __future__ import annotations

from types import SimpleNamespace

import builtins
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
HELPER = "scripts/optimization_word_record_release.py"
ARCHIVE = "assets/generated/openquantum-code.zip"
PREFIX = "openquantum-examples/"
SAVED_HELPER = (
    b"TOKEN = 'authenticated fixture'\n"
    b"def verify_public_word_transport(root, *, replay=True):\n"
    b"    return {'root': str(root), 'replay': replay}\n"
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(path, name):
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def write_package(directory, files):
    directory.mkdir(parents=True)
    for name, raw in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    hashes = {name: digest(raw) for name, raw in files.items()}
    provenance = {
        "schema": "openquantum-public-package-v2",
        "source_commit": "a" * 40,
        "file_sha256": hashes,
        "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")},
    }
    (directory / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def marker_source(marker):
    return f"from pathlib import Path\nPath({str(marker)!r}).write_text('unverified helper')\n".encode()


@pytest.fixture
def checked():
    return load_source(ROOT / "scripts/verify_public_package.py", "word_registration_verifier")


@pytest.fixture
def package(tmp_path, checked):
    path = (tmp_path / "package").resolve()
    write_package(path, {HELPER: SAVED_HELPER, "record.json": b'{"saved":true}\n'})
    return path, checked.verify_manifest(path)


@pytest.fixture
def checker(monkeypatch, checked):
    """Load the actual site caller while replacing unrelated chapter gates."""
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("The rendered-site caller is deliberately outside the examples package")
    dependencies = {
        "site_redirects": {"ALGORITHMS": [], "REDIRECTS": {}},
        "public_assets": {"is_runtime_artifact": lambda _: False},
        "public_code_preview": {"verify_preview": lambda *_: {}},
        "export_code": {"PUBLIC_COMPRESSED_JSON_FILES": frozenset(),
                        "PUBLIC_OPTIMIZATION_WORD_FILES": frozenset(),
                        "PUBLIC_SUZUKI_FILES": frozenset(), "PUBLIC_GIBBS_CHANNEL_FILES": frozenset(), "PUBLIC_DECAY_FILES": frozenset(), "PUBLIC_CORRECTION_FILES": frozenset(), "PUBLIC_DQI_DEGREE_TWO_FILES": frozenset(), "PUBLIC_DQI_DEGREE_TWO_PUBLIC_FILES": frozenset(),
                        "PUBLIC_DQI_FILES": frozenset(), "PUBLIC_BATH_FILES": frozenset(),
                        "PUBLIC_MULTIPRODUCT_FILES": frozenset(), "PUBLIC_BAYESIAN_FILES": frozenset(), "PUBLIC_ROBUST_FILES": frozenset(), "PUBLIC_DQI_STRUCTURED_FILES": frozenset(), "PUBLIC_QPE_FEEDBACK_FILES": frozenset(),
                        "PUBLIC_LOCAL_SPIN_FILES": frozenset(), "PUBLIC_LOCAL_SPIN_PRESENTATION_FILES": frozenset(), "PUBLIC_MEASURED_AND_FILES": frozenset(), "PUBLIC_LOCAL_SPIN_REPRODUCTION_FILES": frozenset(), "PUBLIC_DQI_RESOURCE_FILES": frozenset(),
                        "PUBLIC_CORRECTED_FORMULA_FILES": frozenset(), "PUBLIC_GQSP_FILES": frozenset(),
                        "PUBLIC_TWO_ROUND_FILES": frozenset(),
                        "PUBLIC_SHOR_FINITE_ADDER_FILES": frozenset(),
                        "PUBLIC_DQI_PATH_MAJORITY_FILES": frozenset(),
                        "PUBLIC_NEXT_EVIDENCE_SOURCE_PATHS": frozenset(),
                        "PUBLIC_RPE_RESOURCE_SOURCE_FILES": frozenset()},
        "energy_record_release": {
            "CORE_FILES": frozenset(), "MANIFEST_NAME": "unused.json", "RECORD_DIRECTORY": "unused",
            "SOURCE_FILES": (), "VERIFIER_FILES": (), "checked_directory": checked.checked_directory,
            "site_energy_record_errors": lambda *_: [],
        },
        "dynamics_record_release": {
            "CORE_FILES": frozenset(), "MANIFEST_NAME": "unused.json", "RECORD_DIRECTORY": "unused",
            "SOURCE_FILES": (), "VERIFIER_FILES": (), "site_dynamics_record_errors": lambda *_: [],
        },
    }
    for name in (
        "verify_hadamard_qpe", "verify_optimization_compiled", "verify_simulation_lindblad_physical",
        "verify_simulation_lindblad_three", "verify_simulation_lindblad_four",
        "verify_simulation_lindblad_port", "verify_simulation_qdrift",
    ):
        dependencies[name] = {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused", "SOURCE_FILES": {},
            "verify_bundle": lambda *_, **__: {}, "verify_record": lambda *_, **__: {},
        }
    for name, attributes in dependencies.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "verify_public_package", checked)
    caller = load_source(path, "word_registration_site_caller")
    monkeypatch.setattr(caller, "site_suzuki_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_gibbs_channel_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_decay_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_tail_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_dqi_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_bath_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_multiproduct_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_bayesian_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_robust_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_qpe_feedback_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_dqi_structured_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_dqi_resource_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_local_spin_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_local_spin_reproduction_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_measured_and_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_hamiltonian_packet_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "site_next_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(caller, "MEASURED_AND_FIGURE_LABELS", frozenset())
    monkeypatch.setattr(caller, "GIBBS_CHANNEL_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(caller, "DECAY_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(caller, "CORRECTION_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(caller, "NEXT_EVIDENCE_ARTICLE_LABELS", {})
    return caller


@pytest.fixture
def site(tmp_path):
    root = (tmp_path / "site").resolve()
    code = root / "code"
    provenance = write_package(code, {HELPER: SAVED_HELPER})
    identity = {
        "schema": "openquantum-source-preview-v1",
        "retained_sha256": {HELPER: provenance["file_sha256"][HELPER],
                            "PROVENANCE.json": digest((code / "PROVENANCE.json").read_bytes())},
    }
    return root, identity


def poison_bytecache(source, marker):
    original = source.read_bytes()
    source.write_bytes(marker_source(marker))
    cache = Path(py_compile.compile(str(source), doraise=True))
    source.write_bytes(original)
    metadata = source.stat()
    raw = cache.read_bytes()
    header = (raw[:4] + (0).to_bytes(4, "little")
              + (int(metadata.st_mtime) & 0xffffffff).to_bytes(4, "little")
              + metadata.st_size.to_bytes(4, "little"))
    cache.write_bytes(header + raw[16:])


def test_bootstrap_verifier_import_does_not_load_word_dependencies(monkeypatch):
    raw = (ROOT / "scripts/verify_public_package.py").read_bytes()
    original_import = builtins.__import__
    imports = []

    def guarded(name, *args, **kwargs):
        imports.append(name)
        if name.startswith(("optimization_word", "openquantum", "qdk", "qiskit")):
            pytest.fail("Bootstrap loaded an unauthenticated word/scientific dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = {"__file__": str(ROOT / "scripts/verify_public_package.py"), "__name__": "bootstrap_probe"}
    exec(compile(raw, namespace["__file__"], "exec"), namespace)
    assert callable(namespace["verify_manifest"])
    assert callable(namespace["extract_package"])
    assert not any(name.startswith("optimization_word") for name in imports)


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "provenance_missing", "provenance_invalid"])
def test_package_manifest_failure_precedes_all_downstream_dispatch(package, checked, monkeypatch, tmp_path, change):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    root, _ = package
    marker = tmp_path / "manifest-marker.txt"
    if change == "changed":
        (root / HELPER).write_bytes(marker_source(marker))
    elif change == "missing":
        (root / HELPER).unlink()
    elif change == "unrecorded":
        (root / "unrecorded.py").write_bytes(marker_source(marker))
    elif change == "provenance_missing":
        (root / "PROVENANCE.json").unlink()
    else:
        (root / "PROVENANCE.json").write_text("{}")

    def forbidden(*_args, **_kwargs):
        pytest.fail("A failed complete manifest reached a downstream gate")

    for name in ("verify_formal_record", "_load_energy_record_gate", "_load_dynamics_record_gate",
                 "_load_word_record_gate", "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate", "_load_decay_record_gate",
                 "_load_tail_record_gate", "_load_dqi_saved_gate", "_verify_bath_saved_identity",
                 "_load_multiproduct_saved_gate", "_load_local_spin_saved_gate"):
        monkeypatch.setattr(checked, name, forbidden)
    with pytest.raises((ValueError, OSError)):
        checked.verify_package(root, run_tests=False)
    assert not marker.exists()


def test_package_dispatch_follows_manifest_formal_and_energy_checks(tmp_path, checked, monkeypatch):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    root = (tmp_path / "package").resolve()
    helper = (
        b"def verify_public_word_transport(root, *, replay=True):\n"
        b"    assert replay is True\n"
        b"    raise RuntimeError('authenticated saved dispatch')\n"
    )
    provenance = write_package(root, {HELPER: helper})
    order = []
    real_manifest, real_load = checked.verify_manifest, checked._load_word_record_gate

    def manifest(path):
        value = real_manifest(path)
        order.append("manifest")
        return value

    def energy(path, identity):
        assert identity == provenance
        order.append("energy-load")
        return types.SimpleNamespace(RECORD_DIRECTORY="unused",
                                     verify_energy_record_identity=lambda *_: order.append("energy-check"))

    def dynamics(path, identity):
        assert path == root and identity == provenance
        order.append("dynamics-load")
        return types.SimpleNamespace(RECORD_DIRECTORY="unused",
                                     verify_dynamics_record_identity=lambda *_: order.append("dynamics-check"))

    def word(path, identity):
        assert path == root and identity == provenance
        assert order == ["manifest", "formal", "energy-load", "energy-check", "dynamics-load", "dynamics-check"]
        order.append("word-load")
        return real_load(path, identity)

    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda *_: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", energy)
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", dynamics)
    monkeypatch.setattr(checked, "_load_word_record_gate", word)
    with pytest.raises(RuntimeError, match="authenticated saved dispatch"):
        checked.verify_package(root, run_tests=False)
    assert order == ["manifest", "formal", "energy-load", "energy-check", "dynamics-load", "dynamics-check", "word-load"]


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "schema"])
def test_word_loader_rejects_missing_or_unauthenticated_helper(package, checked, tmp_path, change):
    root, provenance = package
    marker = tmp_path / "loader-marker.txt"
    if change == "changed":
        (root / HELPER).write_bytes(marker_source(marker))
    elif change == "missing":
        (root / HELPER).unlink()
    elif change == "unrecorded":
        provenance["file_sha256"].pop(HELPER)
    else:
        provenance["schema"] = "unverified"
    with pytest.raises((ValueError, OSError)):
        checked._load_word_record_gate(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("where", ["root", "ancestor", "helper", "helper_parent", "hardlink"])
def test_word_loader_rejects_linked_paths_before_compilation(package, checked, tmp_path, where):
    root, provenance = package
    marker = tmp_path / "linked-helper-marker.txt"
    raw = marker_source(marker)
    (root / HELPER).write_bytes(raw)
    provenance["file_sha256"][HELPER] = digest(raw)
    if where == "root":
        linked = tmp_path / "root-alias"
        linked.symlink_to(root, target_is_directory=True)
        root = linked
    elif where == "ancestor":
        linked = tmp_path / "parent-alias"
        linked.symlink_to(root.parent, target_is_directory=True)
        root = linked / root.name
    elif where == "helper_parent":
        source = root / "scripts"
        target = root / "original-scripts"
        source.rename(target)
        source.symlink_to(target.name, target_is_directory=True)
    else:
        source = root / HELPER
        target = tmp_path / "original-helper.py"
        source.rename(target)
        if where == "helper":
            source.symlink_to(target)
        else:
            os.link(target, source)
    with pytest.raises(ValueError, match="(?i)linked|unsafe|regular"):
        checked._load_word_record_gate(root, provenance)
    assert not marker.exists()


def test_word_helper_byte_limit_rejects_before_open(package, checked, monkeypatch):
    root, provenance = package
    (root / HELPER).write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    monkeypatch.setattr(checked.os, "open", lambda *_args, **_kwargs: pytest.fail("Oversized helper was opened"))
    with pytest.raises(ValueError, match="bounded regular"):
        checked._load_word_record_gate(root, provenance)


def test_shared_reader_uses_one_bounded_stream_read(package, checked, monkeypatch):
    root, _ = package
    original = checked.os.fdopen
    reads = []

    class ObservedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, maximum):
            reads.append(maximum)
            return self.stream.read(maximum)

    monkeypatch.setattr(checked.os, "fdopen", lambda *args, **kwargs: ObservedStream(original(*args, **kwargs)))
    assert checked._read_bounded_public_member(root, HELPER, 2048) == SAVED_HELPER
    assert reads == [2049]


def test_word_loader_hashes_and_compiles_one_snapshot(package, checked, monkeypatch, tmp_path):
    root, provenance = package
    source = root / HELPER
    marker = tmp_path / "second-read-marker.txt"
    real_read, real_hash = checked._read_bounded_public_member, hashlib.sha256
    reads = []

    def read(path, name, maximum):
        reads.append((path, name))
        return real_read(path, name, maximum)

    def changed_after_hash(raw=b"", *args, **kwargs):
        result = real_hash(raw, *args, **kwargs)
        if raw == SAVED_HELPER:
            source.write_bytes(marker_source(marker))
        return result

    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    monkeypatch.setattr(checked.hashlib, "sha256", changed_after_hash)
    gate = checked._load_word_record_gate(root, provenance)
    assert gate.TOKEN == "authenticated fixture"
    assert reads == [(root, HELPER)]
    assert not marker.exists()


def test_word_loader_ignores_shadow_cache_and_bytecode(package, checked, monkeypatch, tmp_path):
    root, provenance = package
    marker = tmp_path / "cache-marker.txt"
    poison_bytecache(root / HELPER, marker)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "optimization_word_record_release.py").write_bytes(marker_source(marker))
    cached = types.ModuleType("optimization_word_record_release")
    cached.TOKEN = "untrusted cache"
    for name in ("optimization_word_record_release", "_openquantum_verified_word_record_gate"):
        monkeypatch.setitem(sys.modules, name, cached)
    monkeypatch.syspath_prepend(str(shadow))
    before = list(sys.path)
    gate = checked._load_word_record_gate(root, provenance)
    assert gate.TOKEN == "authenticated fixture" and gate is not cached
    assert all(module is not gate for module in list(sys.modules.values()))
    assert sys.modules["optimization_word_record_release"] is cached
    assert sys.modules["_openquantum_verified_word_record_gate"] is cached
    assert sys.path == before
    assert not marker.exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO boundary")
@pytest.mark.parametrize("member", [HELPER, "PROVENANCE.json"])
@pytest.mark.parametrize("replace_at_open", [False, True])
def test_shared_reader_rejects_fifo_without_blocking(package, tmp_path, member, replace_at_open):
    root, _ = package
    if not replace_at_open:
        (root / member).unlink()
        os.mkfifo(root / member)
    body = """
import importlib.util, os, sys
from pathlib import Path
path, root, member, race = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4] == 'True'
spec = importlib.util.spec_from_file_location('bounded_fifo_reader', path)
checked = importlib.util.module_from_spec(spec)
exec(compile(path.read_bytes(), str(path), 'exec'), checked.__dict__)
original = checked.os.open
def opening(value, flags, *args, **kwargs):
    if race and Path(value) == root / member:
        (root / member).unlink()
        os.mkfifo(root / member)
    return original(value, flags, *args, **kwargs)
checked.os.open = opening
try:
    checked._read_bounded_public_member(root, member, 2 * 1024 * 1024)
except ValueError:
    print('rejected')
else:
    raise AssertionError('FIFO accepted')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", textwrap.dedent(body),
         str(ROOT / "scripts/verify_public_package.py"), str(root), member, str(replace_at_open)],
        capture_output=True, text=True, timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "rejected"


def test_site_dispatch_uses_authenticated_provenance_and_canonical_root(site, checker, monkeypatch):
    root, identity = site
    original_load = checker._load_word_record_gate
    calls = []

    def load(code, provenance):
        calls.append((code, provenance))
        gate = original_load(code, provenance)
        original = gate.verify_public_word_transport

        def verify(path, *, replay):
            assert path == root and replay is False
            calls.append((path, replay))
            return original(path, replay=replay)

        gate.verify_public_word_transport = verify
        return gate

    monkeypatch.setattr(checker, "_load_word_record_gate", load)
    assert checker.site_word_record_errors(root, identity) == []
    assert calls[0][0] == root / "code"
    assert calls[0][1]["file_sha256"][HELPER] == digest(SAVED_HELPER)
    assert calls[1] == (root, False)


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded_identity", "invalid_identity"])
def test_site_provenance_failure_precedes_helper_loading(site, checker, monkeypatch, change):
    root, identity = site
    provenance = root / "code/PROVENANCE.json"
    if change == "changed":
        provenance.write_text("{}")
    elif change == "missing":
        provenance.unlink()
    elif change == "unrecorded_identity":
        identity["retained_sha256"].pop("PROVENANCE.json")
    else:
        identity["schema"] = "unverified"
    monkeypatch.setattr(checker, "_load_word_record_gate", lambda *_: pytest.fail("Unverified provenance reached helper"))
    assert checker.site_word_record_errors(root, identity)


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded"])
def test_site_helper_failure_never_executes_unverified_source(site, checker, tmp_path, change):
    root, identity = site
    marker = tmp_path / "site-helper-marker.txt"
    source = root / "code" / HELPER
    if change == "changed":
        source.write_bytes(marker_source(marker))
    elif change == "missing":
        source.unlink()
    else:
        path = root / "code/PROVENANCE.json"
        provenance = json.loads(path.read_bytes())
        provenance["file_sha256"].pop(HELPER)
        provenance["source_sha256"].pop(HELPER)
        path.write_text(json.dumps(provenance))
        identity["retained_sha256"]["PROVENANCE.json"] = digest(path.read_bytes())
    assert checker.site_word_record_errors(root, identity)
    assert not marker.exists()


@pytest.mark.parametrize("member", [HELPER, "PROVENANCE.json"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_site_rejects_linked_source_or_provenance(site, checker, tmp_path, member, kind):
    root, identity = site
    marker = tmp_path / "linked-site-marker.txt"
    raw = marker_source(marker)
    (root / "code" / HELPER).write_bytes(raw)
    provenance_path = root / "code/PROVENANCE.json"
    provenance = json.loads(provenance_path.read_bytes())
    provenance["file_sha256"][HELPER] = digest(raw)
    provenance["source_sha256"][HELPER] = digest(raw)
    provenance_path.write_text(json.dumps(provenance))
    identity["retained_sha256"]["PROVENANCE.json"] = digest(provenance_path.read_bytes())
    source = root / "code" / member
    target = tmp_path / "original-member"
    source.rename(target)
    if kind == "symlink":
        source.symlink_to(target)
    else:
        os.link(target, source)
    assert checker.site_word_record_errors(root, identity)
    assert not marker.exists()


def test_site_hashes_and_parses_the_same_provenance_snapshot(site, checker, monkeypatch):
    root, identity = site
    original = checker._read_bounded_public_member
    reads = []

    def read(code, name, maximum):
        raw = original(code, name, maximum)
        reads.append(name)
        if name == "PROVENANCE.json":
            (code / name).write_text("{}")
        return raw

    monkeypatch.setattr(checker, "_read_bounded_public_member", read)
    assert checker.site_word_record_errors(root, identity) == []
    assert reads == ["PROVENANCE.json"]


def test_site_preview_failure_cannot_dispatch_the_word_helper(tmp_path, checker, monkeypatch):
    monkeypatch.setattr(checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    root = (tmp_path / "site").resolve()
    root.mkdir()
    (root / "index.html").write_text("<html><body>Structural fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda *_: [])
    for name in vars(checker):
        if name.endswith("_EMBED_LABELS"):
            monkeypatch.setattr(checker, name, frozenset())

    def failed_preview(*_args, **_kwargs):
        raise ValueError("rejected complete archive")

    monkeypatch.setattr(checker, "verify_preview", failed_preview)
    monkeypatch.setattr(checker, "site_word_record_errors", lambda *_: pytest.fail("Preview failure dispatched saved helper"))
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])
    with pytest.raises(SystemExit, match="rejected complete archive"):
        checker.main()


def test_site_main_dispatches_only_after_successful_preview(site, checker, monkeypatch):
    monkeypatch.setattr(checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    root, identity = site
    (root / "index.html").write_text("<html><body>Structural fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda *_: [])
    for name in vars(checker):
        if name.endswith("_EMBED_LABELS"):
            monkeypatch.setattr(checker, name, frozenset())
    order = []

    def preview(path):
        assert path == root
        order.append("preview")
        return identity

    def word(path, authenticated):
        assert path == root and authenticated is identity
        assert order == ["preview"]
        order.append("word")
        return []

    monkeypatch.setattr(checker, "verify_preview", preview)
    monkeypatch.setattr(checker, "site_word_record_errors", word)
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])
    checker.main()
    assert order == ["preview", "word"]


@pytest.mark.parametrize("change", ["missing_preview", "changed_preview", "unrecorded_preview", "bytecache", "changed_archive"])
def test_actual_binder_assembly_never_imports_the_word_helper(tmp_path, change):
    root = (tmp_path / "site").resolve()
    marker = tmp_path / "binder-marker.txt"
    files = {"scripts/verify_public_package.py": (ROOT / "scripts/verify_public_package.py").read_bytes(),
             "src/openquantum/__init__.py": b"# structural package\n"}
    if change != "unrecorded_preview":
        files[HELPER] = SAVED_HELPER
    code = root / "code"
    provenance = write_package(code, files)
    provenance_bytes = (code / "PROVENANCE.json").read_bytes()
    archive = root / ARCHIVE
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as zipped:
        for name, raw in files.items():
            zipped.writestr(PREFIX + name, marker_source(marker) if change == "changed_archive" and name == HELPER else raw)
        zipped.writestr(PREFIX + "PROVENANCE.json", provenance_bytes)
    (code / "PREVIEW.json").write_text(json.dumps({
        "schema": "openquantum-source-preview-v1", "source_commit": provenance["source_commit"],
        "archive": {"path": ARCHIVE, "sha256": digest(archive.read_bytes())},
    }))
    helper = code / HELPER
    if change == "missing_preview":
        helper.unlink()
    elif change in {"changed_preview", "unrecorded_preview"}:
        helper.parent.mkdir(exist_ok=True)
        helper.write_bytes(marker_source(marker))
    elif change == "bytecache":
        poison_bytecache(helper, marker)
    body = """
import builtins, importlib.util, sys, types
from pathlib import Path
source, root, marker, fails = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4] == 'True'
cached = types.ModuleType('optimization_word_record_release')
cached.checked_directory = lambda *_: marker.write_text('cached authority')
sys.modules['optimization_word_record_release'] = cached
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('optimization_word', 'openquantum', 'qdk', 'qiskit')):
        raise AssertionError('Binder imported an unauthenticated helper')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('word_binder_probe', source)
assembly = importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(), str(source), 'exec'), assembly.__dict__)
try:
    assembly.assemble(root)
except ValueError:
    assert fails
else:
    assert not fails
assert not marker.exists()
assert sys.modules['optimization_word_record_release'] is cached
print('word bootstrap remained dependency-free')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", textwrap.dedent(body), str(ROOT / "binder/assemble_code.py"),
         str(root), str(marker), str(change == "changed_archive")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "word bootstrap remained dependency-free"
    assert not marker.exists()
