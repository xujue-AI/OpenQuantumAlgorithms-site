"""Suzuki release registration and outer trust boundaries; no scientific replay."""
from __future__ import annotations

from types import SimpleNamespace

import ast
import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import re
import subprocess
import sys
import textwrap
import types
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/verify_simulation_suzuki.py"
ARCHIVE = "assets/generated/openquantum-code.zip"
PREFIX = "openquantum-examples/"
BUNDLE = 'assets/generated/simulation/suzuki'
CANONICAL_MANIFEST_SHA = '56b52ffa3729df2e4ba2ad6a5df67e61654df54b7654e97bbabe84f6199a0b8e'
CASE_IDS = tuple(f'n{n}-s{order}-r{r}' for n in (1, 2, 3) for order in (2, 4, 6) for r in (1, 2, 4, 8))
ARTIFACTS = frozenset({
    'LICENSE.txt', 'error-vs-steps.svg', 'error-vs-work.svg', 'generate_simulation_suzuki.py',
    'manifest.json', 'n1-order4-circuit.svg', 'n2-order4-circuit.svg', 'n3-order4-circuit.svg',
    'signed-substeps.svg', 'summary.csv', 'summary.md', 'suzuki.py',
}) | frozenset(f'{case}/{name}' for case in CASE_IDS for name in ('record.json', 'measured-trial.qasm'))
LABELS = frozenset({
    'suzuki-signed-recursion', 'suzuki-small-circuit', 'suzuki-independent-check',
    'suzuki-error-work', 'suzuki-rounding-scope', 'suzuki-large-generator',
})
SOURCE_PINS = {
    'src/openquantum/simulation/suzuki.py': '7410f1a50873847f630a49e906351be2a0a042f509738942ae5fd32ba42748be',
    'scripts/generate_simulation_suzuki.py': '3336b53325942b21cb7b6cd0be5a3bcbf6fca149cbe696d14e4ba7e0f6403d9e',
    'requirements-lock.txt': '6a74e5959e5b64c95e820aad7032b4240747cb92541878ea72c9cc55b86a3088',
    'LICENSE': '5040a62bb2735a171968f6ed4f10ebdb1f066a05d5549e3b3181129cda4ef67c',
}
SAVED_HELPER = (
    b"BUNDLE_RELATIVE = 'assets/generated/simulation/suzuki'\n"
    b"TOKEN = 'authenticated fixture'\n"
    b"def verify_bundle(bundle, *, source_root=None):\n"
    b"    return {'bundle': str(bundle), 'source_root': str(source_root)}\n"
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
    return load_source(ROOT / "scripts/verify_public_package.py", "suzuki_registration_verifier")


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
                        "PUBLIC_SUZUKI_FILES": frozenset(BUNDLE + "/" + p for p in ARTIFACTS),
                        "PUBLIC_GIBBS_CHANNEL_FILES": frozenset(), "PUBLIC_DECAY_FILES": frozenset(), "PUBLIC_CORRECTION_FILES": frozenset(), "PUBLIC_DQI_DEGREE_TWO_FILES": frozenset(), "PUBLIC_DQI_DEGREE_TWO_PUBLIC_FILES": frozenset(), "PUBLIC_DQI_FILES": frozenset(),
                        "PUBLIC_BATH_FILES": frozenset(), "PUBLIC_MULTIPRODUCT_FILES": frozenset(), "PUBLIC_BAYESIAN_FILES": frozenset(), "PUBLIC_ROBUST_FILES": frozenset(), "PUBLIC_DQI_STRUCTURED_FILES": frozenset(), "PUBLIC_QPE_FEEDBACK_FILES": frozenset(),
                        "PUBLIC_LOCAL_SPIN_FILES": frozenset(), "PUBLIC_LOCAL_SPIN_PRESENTATION_FILES": frozenset(), "PUBLIC_MEASURED_AND_FILES": frozenset(), "PUBLIC_LOCAL_SPIN_REPRODUCTION_FILES": frozenset(), "PUBLIC_DQI_RESOURCE_FILES": frozenset(),
                        "PUBLIC_CORRECTED_FORMULA_FILES": frozenset(), "PUBLIC_GQSP_FILES": frozenset()},
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
    module = load_source(path, "suzuki_registration_site_caller")
    monkeypatch.setattr(module, "site_word_record_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_gibbs_channel_record_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_decay_record_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_tail_record_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_dqi_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_bath_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_multiproduct_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_bayesian_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_robust_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_qpe_feedback_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_dqi_structured_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_dqi_resource_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_local_spin_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_local_spin_reproduction_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_measured_and_saved_errors", lambda *_: [])
    monkeypatch.setattr(module, "site_hamiltonian_packet_errors", lambda *_: [])
    monkeypatch.setattr(module, "MEASURED_AND_FIGURE_LABELS", frozenset())
    monkeypatch.setattr(module, "GIBBS_CHANNEL_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(module, "DECAY_ARTICLE_LABELS", frozenset())
    monkeypatch.setattr(module, "CORRECTION_ARTICLE_LABELS", frozenset())
    return module


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


def test_bootstrap_verifier_import_does_not_load_suzuki_dependencies(monkeypatch):
    raw = (ROOT / "scripts/verify_public_package.py").read_bytes()
    original_import = builtins.__import__
    imports = []

    def guarded(name, *args, **kwargs):
        imports.append(name)
        if name.startswith(("verify_simulation_suzuki", "optimization_word", "openquantum", "qdk", "qiskit")):
            pytest.fail("Bootstrap loaded an unauthenticated suzuki/scientific dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    namespace = {"__file__": str(ROOT / "scripts/verify_public_package.py"), "__name__": "bootstrap_probe"}
    exec(compile(raw, namespace["__file__"], "exec"), namespace)
    assert callable(namespace["verify_manifest"])
    assert callable(namespace["extract_package"])
    assert not any(name.startswith("verify_simulation_suzuki") for name in imports)


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


def test_package_dispatch_follows_manifest_formal_energy_and_word_checks(tmp_path, checked, monkeypatch):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    root = (tmp_path / 'package').resolve()
    helper = (
        b"BUNDLE_RELATIVE = 'assets/generated/simulation/suzuki'\n"
        b"def verify_bundle(bundle, *, source_root=None):\n"
        b"    raise RuntimeError('authenticated saved dispatch')\n"
    )
    provenance = write_package(root, {HELPER: helper})
    order = []
    real_manifest, real_load = checked.verify_manifest, checked._load_suzuki_record_gate

    def manifest(path):
        result = real_manifest(path)
        order.append('manifest')
        return result

    def energy(path, identity):
        assert path == root and identity == provenance
        order.append('energy-load')
        return types.SimpleNamespace(RECORD_DIRECTORY='unused',
                                     verify_energy_record_identity=lambda *_: order.append('energy-check'))

    def dynamics(path, identity):
        assert path == root and identity == provenance
        order.append('dynamics-load')
        return types.SimpleNamespace(RECORD_DIRECTORY='unused',
                                     verify_dynamics_record_identity=lambda *_: order.append('dynamics-check'))

    def word(path, identity):
        assert path == root and identity == provenance
        order.append('word-load')
        return types.SimpleNamespace(verify_public_word_transport=lambda *_, **__: order.append('word-check'))

    def suzuki(path, identity):
        assert path == root and identity == provenance
        assert order == ['manifest', 'formal', 'energy-load', 'energy-check', 'dynamics-load', 'dynamics-check',
                         'word-load', 'word-check']
        order.append('suzuki-load')
        gate = real_load(path, identity)
        original = gate.verify_bundle

        def verify(bundle, *, source_root):
            assert bundle == root / BUNDLE and source_root == root
            return original(bundle, source_root=source_root)

        gate.verify_bundle = verify
        return gate

    monkeypatch.setattr(checked, 'verify_manifest', manifest)
    monkeypatch.setattr(checked, 'verify_formal_record', lambda *_: order.append('formal'))
    monkeypatch.setattr(checked, '_load_energy_record_gate', energy)
    monkeypatch.setattr(checked, '_load_dynamics_record_gate', dynamics)
    monkeypatch.setattr(checked, '_load_word_record_gate', word)
    monkeypatch.setattr(checked, '_load_suzuki_record_gate', suzuki)
    with pytest.raises(RuntimeError, match='authenticated saved dispatch'):
        checked.verify_package(root, run_tests=False)
    assert order == ['manifest', 'formal', 'energy-load', 'energy-check', 'dynamics-load', 'dynamics-check',
                     'word-load', 'word-check', 'suzuki-load']


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "schema"])
def test_suzuki_loader_rejects_missing_or_unauthenticated_helper(package, checked, tmp_path, change):
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
        checked._load_suzuki_record_gate(root, provenance)
    assert not marker.exists()


@pytest.mark.parametrize("where", ["root", "ancestor", "helper", "helper_parent", "hardlink"])
def test_suzuki_loader_rejects_linked_paths_before_compilation(package, checked, tmp_path, where):
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
        checked._load_suzuki_record_gate(root, provenance)
    assert not marker.exists()


def test_suzuki_helper_byte_limit_rejects_before_open(package, checked, monkeypatch):
    root, provenance = package
    (root / HELPER).write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    monkeypatch.setattr(checked.os, "open", lambda *_args, **_kwargs: pytest.fail("Oversized helper was opened"))
    with pytest.raises(ValueError, match="bounded regular"):
        checked._load_suzuki_record_gate(root, provenance)


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


def test_suzuki_loader_hashes_and_compiles_one_snapshot(package, checked, monkeypatch, tmp_path):
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
    gate = checked._load_suzuki_record_gate(root, provenance)
    assert gate.TOKEN == "authenticated fixture"
    assert reads == [(root, HELPER)]
    assert not marker.exists()


def test_suzuki_loader_ignores_shadow_cache_and_bytecode(package, checked, monkeypatch, tmp_path):
    root, provenance = package
    marker = tmp_path / "cache-marker.txt"
    poison_bytecache(root / HELPER, marker)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    (shadow / "verify_simulation_suzuki.py").write_bytes(marker_source(marker))
    cached = types.ModuleType("verify_simulation_suzuki")
    cached.TOKEN = "untrusted cache"
    for name in ("verify_simulation_suzuki", "_openquantum_verified_suzuki_record_gate"):
        monkeypatch.setitem(sys.modules, name, cached)
    monkeypatch.syspath_prepend(str(shadow))
    before = list(sys.path)
    gate = checked._load_suzuki_record_gate(root, provenance)
    assert gate.TOKEN == "authenticated fixture" and gate is not cached
    assert all(module is not gate for module in list(sys.modules.values()))
    assert sys.modules["verify_simulation_suzuki"] is cached
    assert sys.modules["_openquantum_verified_suzuki_record_gate"] is cached
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
    original_load = checker._load_suzuki_record_gate
    calls = []

    def load(code, provenance):
        calls.append((code, provenance))
        gate = original_load(code, provenance)
        original = gate.verify_bundle

        def verify(bundle, *, source_root):
            assert bundle == root / BUNDLE and source_root == root / 'code'
            calls.append((bundle, source_root))
            return original(bundle, source_root=source_root)

        gate.verify_bundle = verify
        return gate

    monkeypatch.setattr(checker, '_load_suzuki_record_gate', load)
    assert checker.site_suzuki_record_errors(root, identity) == []
    assert calls[0][0] == root / 'code'
    assert calls[0][1]['file_sha256'][HELPER] == digest(SAVED_HELPER)
    assert calls[1] == (root / BUNDLE, root / 'code')


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
    monkeypatch.setattr(checker, "_load_suzuki_record_gate", lambda *_: pytest.fail("Unverified provenance reached helper"))
    assert checker.site_suzuki_record_errors(root, identity)


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
    assert checker.site_suzuki_record_errors(root, identity)
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
    assert checker.site_suzuki_record_errors(root, identity)
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
    assert checker.site_suzuki_record_errors(root, identity) == []
    assert reads == ["PROVENANCE.json"]


def test_site_preview_failure_cannot_dispatch_the_suzuki_helper(tmp_path, checker, monkeypatch):
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
    monkeypatch.setattr(checker, "site_suzuki_record_errors", lambda *_: pytest.fail("Preview failure dispatched saved helper"))
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

    def suzuki(path, authenticated):
        assert path == root and authenticated is identity
        assert order == ["preview"]
        order.append("suzuki")
        return []

    monkeypatch.setattr(checker, "verify_preview", preview)
    monkeypatch.setattr(checker, "site_suzuki_record_errors", suzuki)
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])
    checker.main()
    assert order == ["preview", "suzuki"]


@pytest.mark.parametrize("change", ["missing_preview", "changed_preview", "unrecorded_preview", "bytecache", "changed_archive"])
def test_actual_binder_assembly_never_imports_the_suzuki_helper(tmp_path, change):
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
cached = types.ModuleType('verify_simulation_suzuki')
cached.checked_directory = lambda *_: marker.write_text('cached authority')
sys.modules['verify_simulation_suzuki'] = cached
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(('verify_simulation_suzuki', 'optimization_word', 'openquantum', 'qdk', 'qiskit')):
        raise AssertionError('Binder imported an unauthenticated helper')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
spec = importlib.util.spec_from_file_location('suzuki_binder_probe', source)
assembly = importlib.util.module_from_spec(spec)
exec(compile(source.read_bytes(), str(source), 'exec'), assembly.__dict__)
try:
    assembly.assemble(root)
except ValueError:
    assert fails
else:
    assert not fails
assert not marker.exists()
assert sys.modules['verify_simulation_suzuki'] is cached
print('suzuki bootstrap remained dependency-free')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", textwrap.dedent(body), str(ROOT / "binder/assemble_code.py"),
         str(root), str(marker), str(change == "changed_archive")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "suzuki bootstrap remained dependency-free"
    assert not marker.exists()


@pytest.fixture
def exporter(monkeypatch):
    """Read registration code without running its whole-package exporter."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    provenance = types.ModuleType("source_provenance")
    provenance.git_revision = lambda *_: pytest.fail("Registration checks cannot execute Git/export production")
    port = types.ModuleType("verify_simulation_lindblad_port")
    port.BUNDLE_FILES = frozenset()
    monkeypatch.setitem(sys.modules, "source_provenance", provenance)
    monkeypatch.setitem(sys.modules, "verify_simulation_lindblad_port", port)
    return load_source(ROOT / "scripts/export_code.py", "suzuki_export_registration")


def assert_payloads(directory, manifest_name, pin, count, *, exact_inventory):
    raw = (directory / manifest_name).read_bytes()
    assert digest(raw) == pin
    manifest = json.loads(raw)
    assert len(manifest["files"]) == count
    total = len(raw)
    for name, expected in manifest["files"].items():
        assert set(expected) == {"bytes", "sha256"}
        path = directory / name
        assert path.is_file() and not path.is_symlink()
        hasher = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                hasher.update(chunk)
        assert size == expected["bytes"], name
        assert hasher.hexdigest() == expected["sha256"], name
        total += size
    if exact_inventory:
        actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
        assert actual == set(manifest["files"]) | {manifest_name}
    return manifest, total


def test_canonical_suzuki_inventory_and_producer_sources_remain_frozen():
    directory = ROOT / BUNDLE
    manifest, total = assert_payloads(directory, "manifest.json", CANONICAL_MANIFEST_SHA, 83,
                                     exact_inventory=True)
    assert set(manifest["files"]) | {"manifest.json"} == ARTIFACTS
    assert len(ARTIFACTS) == 84 and total == 1_695_183
    assert manifest["source_commit"] == "a9edff74764c2e89ea835c1ef19fc5c1e18c0e64"
    assert manifest["status"] == "generated_and_numerically_checked"
    assert manifest["case_ids"] == list(CASE_IDS)
    assert manifest["source_sha256"] == SOURCE_PINS
    for name, pin in SOURCE_PINS.items():
        assert digest((ROOT / name).read_bytes()) == pin
    assert (directory / "suzuki.py").read_bytes() == (ROOT / "src/openquantum/simulation/suzuki.py").read_bytes()
    assert (directory / "generate_simulation_suzuki.py").read_bytes() == (ROOT / "scripts/generate_simulation_suzuki.py").read_bytes()
    assert digest((ROOT / "tests/test_simulation_suzuki.py").read_bytes()) == (
        "ccc05ee25c5f64a19537131f92bd5e7a42859a8ee0040037bef6b25c63aabced")
    assert digest((ROOT / "tests/test_simulation_suzuki_release.py").read_bytes()) == (
        "0191746a5ae2427b31af17104cf9b787410fca98c88da2d63144c77bac6c1e47")


def test_exact_suzuki_package_site_download_and_label_registration(exporter, checker):
    expected = {BUNDLE + "/" + name for name in ARTIFACTS}
    assert exporter.PUBLIC_SUZUKI_FILES == expected
    assert {"generate_simulation_suzuki.py", "verify_simulation_suzuki.py"} <= set(exporter.PUBLIC_SCRIPTS)
    assert all(exporter.include_generated_file(name) for name in expected)
    assert not any(name.startswith(BUNDLE + "/") for name in exporter.PUBLIC_DEPENDENCY_FILES)
    required = set(checker.required_artifacts())
    assert expected <= required
    assert {
        "algorithms/simulation/suzuki/index.html", "notebooks/simulation-suzuki/index.html",
        "notebooks/simulation_suzuki.ipynb", "code/src/openquantum/simulation/suzuki.py",
        "code/scripts/generate_simulation_suzuki.py", "code/scripts/verify_simulation_suzuki.py",
    } <= required
    assert not any(name.startswith("code/" + BUNDLE + "/") for name in required)
    assert checker.SUZUKI_EMBED_LABELS == LABELS
    notebook = json.loads((ROOT / "notebooks/simulation_suzuki.ipynb").read_bytes())
    labels = {cell.get("metadata", {}).get("label") for cell in notebook["cells"]}
    assert LABELS <= labels and len(LABELS) == 6


@pytest.mark.parametrize("name", [
    "unreviewed.json", "unreviewed.json.gz", "unreviewed.py", "requirements-lock.txt", "references.bib",
    "n4-s6-r8/record.json", "./manifest.json", "../suzuki/manifest.json",
])
def test_suzuki_registration_does_not_infer_unreviewed_files(exporter, name):
    assert not exporter.include_generated_file(BUNDLE + "/" + name)


def test_suzuki_saved_helper_and_bootstrap_imports_are_standard_library_only():
    for name in (HELPER, "scripts/verify_public_package.py"):
        roots = set()
        for node in ast.parse((ROOT / name).read_bytes()).body:
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert roots <= sys.stdlib_module_names | {"__future__"}, name


def find_file(nodes, name):
    matches = []
    for node in nodes:
        if node.get("file") == name:
            matches.append(node)
        matches.extend(find_file(node.get("children", []), name))
    return matches


def test_suzuki_navigation_is_a_hamiltonian_notebook_child():
    config = ROOT / "myst.yml"
    if not config.is_file():
        pytest.skip("Book navigation is deliberately outside the examples package")
    import yaml
    toc = yaml.safe_load(config.read_text())["project"]["toc"]
    dynamics, = find_file(toc, "algorithms/hamiltonian-simulation.md")
    suzuki, = find_file(dynamics["children"], "algorithms/simulation/suzuki.md")
    notebook, = find_file(suzuki["children"], "notebooks/simulation_suzuki.ipynb")
    assert notebook["file"] == "notebooks/simulation_suzuki.ipynb"
    assert len(find_file(toc, "algorithms/simulation/suzuki.md")) == 1
    assert len(find_file(toc, "notebooks/simulation_suzuki.ipynb")) == 1
    assert not find_file(toc, "algorithms/simulation/fault-tolerant-methods.md")
    article = (ROOT / "algorithms/hamiltonian-simulation.md").read_text()
    assert "(hamiltonian-method-map)=" in article
    assert "(hamiltonian-architecture-comparisons)=" in article
    assert find_file(dynamics["children"], "algorithms/simulation/qdrift.md")


def test_suzuki_ci_pair_restores_exact_latest_72_commands():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is deliberately outside the examples package")
    commands = [line.strip()[4:] for line in workflow.read_text().splitlines() if line.startswith("          run ")]
    additions = (
        "python scripts/generate_simulation_suzuki.py --output-dir .cache/suzuki-ci-replay",
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay",
    )
    degree_two = 'python scripts/verify_dqi_degree_two_public_saved.py --artifact-root . --source-root . --notebook-path notebooks/optimization_dqi_degree_two.ipynb --expected-manifest-sha256 ecd71a350153cba5a17ae1451a1545b703e604608b3d95abee815cabbcc26f6c --archive assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip --time-limit-seconds 10'
    assert len(commands) == 86 and commands.count(degree_two) == 1
    assert commands.index(degree_two) == commands.index("python scripts/verify_dqi_saved.py assets/generated/optimization/dqi") + 1
    commands = [value for value in commands if value != degree_two]
    correction = 'python scripts/verify_detailed_balance_correction_saved.py --bundle assets/generated/simulation/detailed-balance-correction --source-root . --download assets/generated/simulation/detailed-balance-correction-download.zip'
    assert len(commands) == 85 and commands.count(correction) == 1
    assert commands.index(correction) == commands.index("python scripts/verify_detailed_balance_decay.py") + 1
    commands = [value for value in commands if value != correction]
    hamiltonian_saved = (
        "python scripts/verify_simulation_cpf_public.py --bundle assets/generated/simulation/corrected-formula",
        "python scripts/verify_simulation_gqsp_public.py --bundle assets/generated/simulation/gqsp",
    )
    assert len(commands) == 84
    assert all(commands.count(value) == 1 for value in hamiltonian_saved)
    assert commands.index(hamiltonian_saved[0]) == commands.index(
        "python scripts/verify_simulation_multiproduct_saved.py") + 1
    assert commands.index(hamiltonian_saved[1]) == commands.index(hamiltonian_saved[0]) + 1
    commands = [value for value in commands if value not in hamiltonian_saved]
    decay = "python scripts/verify_detailed_balance_decay.py"
    assert len(commands) == 82 and commands.count(decay) == 1
    assert commands.index(decay) == commands.index("python scripts/verify_detailed_balance_channel.py") + 1
    commands = [value for value in commands if value != decay]
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
    assert digest(("\n".join(commands) + "\n").encode()) == (
        "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032")
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert commands.count(multiproduct) == 1
    assert commands.index(multiproduct) == commands.index(additions[1]) + 1
    commands = [value for value in commands if value != multiproduct]
    assert len(commands) == 76
    assert digest(("\n".join(commands) + "\n").encode()) == (
        "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af")
    dqi = "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi"
    assert commands.count(dqi) == 1
    assert all(commands.count(command) == 1 for command in additions)
    replacements = {
        ("python scripts/verify_dynamics_resources.py --canonical-qasm-replay "
         "--records-dir assets/generated/dynamics-resources --manifest-sha256 "
         "0596353af49a0de40176069756cd9398749c21ede68e057ef84611bb6d5f4265 "
         "--max-wall-seconds 600"): "python scripts/generate_dynamics_resources.py",
        "python scripts/verify_instrument_resource_records.py --replay": "python scripts/generate_instrument_resources.py",
        "python scripts/verify_simulation_complete_resource_records.py --replay": "python scripts/generate_simulation_complete_resources.py",
    }
    for replacement, original in replacements.items():
        assert commands.count(replacement) == 1 and original not in commands
    restored = [replacements.get(value, value) for value in commands]
    gibbs = "python scripts/verify_detailed_balance_channel.py"
    assert restored.count(gibbs) == 1
    prior = [command for command in restored if command not in additions and command not in (gibbs, dqi)]
    assert len(prior) == 72
    assert digest(("\n".join(prior) + "\n").encode()) == (
        "4af3666293942bcc65be97ac01078f93358db2e3dd44b7a09fcd2571b64030bd")
    qdrift = "python scripts/verify_simulation_qdrift.py --replay-dir .cache/qdrift-ci-replay"
    assert commands.index(additions[0]) == commands.index(qdrift) + 1
    assert commands.index(additions[1]) == commands.index(additions[0]) + 1
    assert commands.index(qdrift) == commands.index(
        "python scripts/generate_simulation_qdrift.py --output-dir .cache/qdrift-ci-replay") + 1
    assert commands.index("python scripts/verify_optimization_word_records.py") == commands.index(
        "python scripts/verify_optimization_compiled.py") + 1
    assert not any("suzuki" in command and "--output-dir assets/generated" in command for command in commands)


BIB_ENTRY_PINS = {
    "ham_mpf_commutator_2026": "6b26f564f836a6c6b4b6c85693f78d128e6e851cc0e10e02da062b913cc325f1",
    "ham_ft_lattice_comparison_2026": "40fd72ccda2e0f26cb059d27a5d0e7f654aa2d7213d65a1a3be2a3ef1bb32a10",
    "ham_ft_active_volume_2025": "a42ac52ea47ca5e44d5b690a58d9acd72b1430efe81e5ef8c6e7ca558a31be8d",
    "ham_ft_active_volume_erratum_2026": "430ec7721c97d4c89fdff37d52571100ebbc04a69807cdd745e8a051a38b098a",
}
# The commutator paragraph only rebases its local companion link after relocation.
# All five exact prose guards remain; scientific text and reference entries are unchanged.
FT_BLOCK_PINS = {
    "The system-size analysis also needs a bounded commutator argument.": "91d71c7205b89c4635f3188e46bdf1e174e24474f107ca3232146bb2c0a688d5",
    "LeBlond and Bennink compare two surface-code compilation families": "eb28ad7d61355f234481537330e7fc2a785f4c1eb31413e84553f12adf6cf084",
    "The active-volume chemistry study separates Hamiltonian factorization,": "649775ebb1310527b008196aaa58ee092b7a625d81b239be5b094921f456b938",
    "| Parallel product-formula layers versus mostly serial QSP access |": "4362b4d1fa4679265954a51ca9a7e7a03d4138eb48c53676ef944b5a26514e6a",
    "| Register-controlled Givens rotations and angle loading in chemistry block encodings |": "9cb30471585785f6d3adf188b8f8f561f23634c5cb8726b017df0ea9798c89c3",
}


def bib_entry(text, key):
    match = re.search(r"(?m)^@\w+\{" + re.escape(key) + r",", text)
    assert match is not None, key
    depth = 0
    for end in range(match.start(), len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if text[end] == "}" and depth == 0:
            assert text[end + 1:end + 2] == "\n"
            return text[match.start():end + 2].encode()
    pytest.fail("Unclosed bibliography entry: " + key)


def test_ft_architecture_blocks_are_retained():
    path = ROOT / "algorithms/hamiltonian-simulation.md"
    if not path.is_file():
        pytest.skip("The book's FT prose is deliberately outside the examples package")
    text = path.read_text()
    for prefix, pin in FT_BLOCK_PINS.items():
        candidates = text.splitlines(keepends=True) if prefix.startswith("|") else [p + "\n" for p in text.split("\n\n")]
        block, = [value for value in candidates if value.startswith(prefix)]
        assert digest(block.encode()) == pin, prefix


def test_exact_ft_reference_entries_are_retained_in_the_public_sources():
    references = (ROOT / "references/frontier.bib").read_text()
    for key, pin in BIB_ENTRY_PINS.items():
        assert digest(bib_entry(references, key)) == pin, key


def test_existing_word_records_and_production_sources_are_preserved():
    directory = ROOT / "assets/generated/optimization/word-transport"
    manifest, total = assert_payloads(directory, "record-manifest.json",
                                     "ee4656d922dd6c6cc0db88ff386cabbd77c85a38c589d2479cbea07dc4fce00d",
                                     209, exact_inventory=True)
    assert total == 10_671_732 and len(manifest["source_pins"]) == 6
    for name in ("optimization_word_accounting.py", "optimization_word_native.py", "optimization_word_native_runner.py",
                 "optimization_word_transport.py", "optimization_word_validation.py"):
        relative = "src/openquantum/" + name
        assert (ROOT / relative).read_bytes() == (directory / "producer-source" / relative).read_bytes()
    for name, size, pin in (
        ("scripts/optimization_word_record_release.py", 4546, "2674bfe0dab04d4452b69ed740ae3e3f4baebec7ed988407a42e4c6e40d339d7"),
        ("scripts/verify_optimization_word_records.py", 1223, "7bdb4018cfa8bb949682e9c480dfba23e801916fb85e0a9922379654bc9d5f80"),
    ):
        raw = (ROOT / name).read_bytes()
        assert len(raw) == size and digest(raw) == pin


def test_existing_qdrift_records_and_sources_are_preserved():
    manifest, total = assert_payloads(ROOT / "assets/generated/simulation/qdrift", "manifest.json",
                                     "7d2bbc3deb40f647009008a55b86c17042720df1916ab7d7bd00f9a066169709",
                                     77, exact_inventory=True)
    assert total == 677_245 and len(manifest["source_sha256"]) == 4
    for name, pin in manifest["source_sha256"].items():
        assert digest((ROOT / name).read_bytes()) == pin


def test_existing_energy_core_and_source_bindings_are_preserved():
    manifest, total = assert_payloads(ROOT / "assets/generated/energy-estimation", "resource-record-manifest.json",
                                     "431f3c5a9568886cbecf292587e33b0a81d1c0c9a5456fc0a06709b3e957aab6",
                                     35, exact_inventory=False)
    assert total == 48_570_242
    assert len(manifest["sources"]) == 9 and len(manifest["verifier_sources"]) == 3
    for name, pin in (manifest["sources"] | manifest["verifier_sources"]).items():
        assert digest((ROOT / name).read_bytes()) == pin


def test_suzuki_figures_and_csv_already_match_normalization():
    directory = ROOT / BUNDLE
    for name in ARTIFACTS:
        path = directory / name
        if path.suffix == ".svg":
            normalized = ("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n").encode()
            assert normalized == path.read_bytes(), name
        elif path.suffix == ".csv":
            assert path.read_text().encode() == path.read_bytes(), name
