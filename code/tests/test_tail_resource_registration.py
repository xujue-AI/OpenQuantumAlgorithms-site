"""Tail publication trust boundaries using pure and synthetic saved fixtures.

No tail producer, numerical replay, synthesis or Native job is invoked here.
The actual public loaders and dispatch functions execute authenticated tiny
stand-ins; unrelated chapter prerequisites are replaced explicitly.
"""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import py_compile
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
HELPER = "scripts/tail_resource_records.py"
INSTRUMENT_PIN = "0333814a823ebece83910251328b1145fc84e3cfbdec68051af92e78280ba220"
COMPLETE_PIN = "86af1d97bad4ac1f4d8c5191d3e546f69ca82a5121bd8de7d7307081c70dc0b1"
BUNDLES = (
    ("instrument", "assets/generated/instrument-resources", INSTRUMENT_PIN),
    ("complete", "assets/generated/simulation/complete-resources", COMPLETE_PIN),
)
PUBLIC_PYTHON = (
    "tail_resource_records.py", "tail_resource_model.py", "tail_resource_qasm.py",
    "tail_resource_replay.py", "tail_resource_numerical.py", "freeze_tail_resource_records.py",
    "verify_instrument_resource_records.py", "verify_simulation_complete_resource_records.py",
)
PUBLIC_POLICIES = (
    "scripts/tail_resource_instrument_policy.json",
    "scripts/tail_resource_complete_policy.json",
)
PUBLIC_SUPPORT = {"scripts/" + name for name in PUBLIC_PYTHON} | set(PUBLIC_POLICIES)
CI_REPLACEMENTS = (
    ("python scripts/generate_instrument_resources.py",
     "python scripts/verify_instrument_resource_records.py --replay"),
    ("python scripts/generate_simulation_complete_resources.py",
     "python scripts/verify_simulation_complete_resource_records.py --replay"),
)
FINAL_CI_SHA = "455c0b472be3ff3365f598be652aa3efe2ddab831f05ad9d87536690c42c2ee4"
PRIOR_CI_SHA = "489776c842bfd1fe1e095106ed60c6c75d0b6173873bc645c6cd75554f929bbd"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_package(directory, files):
    directory.mkdir(parents=True, exist_ok=True)
    for name, raw in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    hashes = {name: digest(raw) for name, raw in files.items()}
    provenance = {
        "schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
        "file_sha256": hashes,
        "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")},
    }
    (directory / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


def saved_core_source(*, stop_after_complete=False):
    return (
        "TOKEN = 'authenticated synthetic tail core'\n"
        "CALLS = []\n"
        "def verify_bundle(records_dir, source_root, *, kind, expected_sha256,\n"
        "                  replay=False, canonical_root=None):\n"
        "    CALLS.append((str(records_dir), str(source_root), kind, expected_sha256,\n"
        "                  replay, None if canonical_root is None else str(canonical_root)))\n"
        + ("    if kind == 'complete':\n"
           "        raise RuntimeError('tail dispatch stopped before later gates')\n"
           if stop_after_complete else "")
        + "    return {'native_calls': 0, 'new_numerical_simulations': 0}\n"
    ).encode()


@pytest.fixture
def checked():
    return load_source(ROOT / "scripts/verify_public_package.py", "tail_registration_package")


@pytest.fixture
def exporter():
    return load_source(ROOT / "scripts/export_code.py", "tail_registration_export")


@pytest.fixture
def site_checker(monkeypatch, checked, exporter):
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is outside the public example package")
    dependencies = {
        "site_redirects": {"ALGORITHMS": [], "REDIRECTS": {}},
        "public_assets": {"is_runtime_artifact": lambda _: False},
        "public_code_preview": {"verify_preview": lambda *_: {}},
        "verify_hadamard_qpe": {"verify_record": lambda *_, **__: {}},
        "verify_optimization_compiled": {"BUNDLE_RELATIVE": "unused-compiled", "verify_bundle": lambda *_, **__: {}},
        "verify_simulation_lindblad_physical": {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused-n2", "verify_bundle": lambda *_, **__: {}},
        "verify_simulation_lindblad_three": {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused-n3", "verify_bundle": lambda *_, **__: {}},
        "verify_simulation_lindblad_four": {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused-n4", "verify_bundle": lambda *_, **__: {}},
        "verify_simulation_lindblad_port": {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused-port", "SOURCE_FILES": {},
            "verify_bundle": lambda *_, **__: {}},
        "verify_simulation_qdrift": {
            "BUNDLE_FILES": frozenset(), "BUNDLE_RELATIVE": "unused-qdrift", "verify_bundle": lambda *_, **__: {}},
        "energy_record_release": {
            "CORE_FILES": frozenset(), "MANIFEST_NAME": "unused.json", "RECORD_DIRECTORY": "unused-energy",
            "SOURCE_FILES": (), "VERIFIER_FILES": (), "checked_directory": checked.checked_directory,
            "site_energy_record_errors": lambda *_: []},
        "dynamics_record_release": {
            "CORE_FILES": frozenset(), "MANIFEST_NAME": "unused.json", "RECORD_DIRECTORY": "unused-dynamics",
            "SOURCE_FILES": (), "VERIFIER_FILES": (), "site_dynamics_record_errors": lambda *_: []},
    }
    for name, attributes in dependencies.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "export_code", exporter)
    monkeypatch.setitem(sys.modules, "verify_public_package", checked)
    return load_source(path, "tail_registration_site")


@pytest.fixture
def synthetic_site(tmp_path):
    root = (tmp_path / "site").resolve()
    provenance = write_package(root / "code", {HELPER: saved_core_source()})
    identity = {
        "schema": "openquantum-source-preview-v1",
        "retained_sha256": {
            "PROVENANCE.json": digest((root / "code/PROVENANCE.json").read_bytes()),
            HELPER: provenance["file_sha256"][HELPER],
        },
    }
    return root, identity


@pytest.mark.parametrize("change", ["changed", "missing", "unrecorded", "invalid-provenance"])
def test_complete_manifest_failure_precedes_every_tail_dispatch(tmp_path, checked, monkeypatch, change):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    package = (tmp_path / "package").resolve()
    write_package(package, {HELPER: saved_core_source(), "record.json": b'{"saved":true}\n'})
    if change == "changed":
        (package / HELPER).write_bytes(b"raise AssertionError('unbound core executed')\n")
    elif change == "missing":
        (package / HELPER).unlink()
    elif change == "unrecorded":
        (package / "extra.py").write_bytes(b"raise AssertionError('unrecorded source')\n")
    else:
        (package / "PROVENANCE.json").write_text("{}")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Failed complete manifest reached a downstream prerequisite or tail loader")

    for name in (
        "verify_formal_record", "_load_energy_record_gate", "_load_dynamics_record_gate",
        "_load_word_record_gate", "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate", "_load_decay_record_gate",
        "_load_tail_record_gate",
    ):
        monkeypatch.setattr(checked, name, forbidden)
    with pytest.raises(ValueError):
        checked.verify_package(package, run_tests=False)
    assert calls == []


@pytest.mark.parametrize("bad_provenance", [
    None, {}, {"schema": "wrong", "file_sha256": {}},
    {"schema": "openquantum-public-package-v2", "file_sha256": []},
    {"schema": "openquantum-public-package-v2", "file_sha256": {HELPER: "0" * 64}},
])
def test_tail_loader_rejects_unbound_hash_before_compilation(tmp_path, checked, monkeypatch, bad_provenance):
    package = (tmp_path / "package").resolve()
    write_package(package, {HELPER: b"raise AssertionError('unbound helper compiled')\n"})

    def forbidden(*args, **kwargs):
        pytest.fail("Unbound tail helper reached compilation")

    monkeypatch.setattr(builtins, "compile", forbidden)
    with pytest.raises(ValueError, match="complete manifest"):
        checked._load_tail_record_gate(package, bad_provenance)


def test_tail_loader_hashes_and_compiles_one_bounded_snapshot(tmp_path, checked, monkeypatch):
    package = (tmp_path / "package").resolve()
    raw = b"TOKEN = 'captured source'\n"
    provenance = write_package(package, {HELPER: raw})
    reads = []
    original_read, original_hash = checked._read_bounded_public_member, hashlib.sha256

    def read(root, name, maximum):
        reads.append((root, name, maximum))
        return original_read(root, name, maximum)

    replaced = False

    def replace_after_hash(value, *args, **kwargs):
        nonlocal replaced
        result = original_hash(value, *args, **kwargs)
        if value == raw and not replaced:
            replaced = True
            (package / HELPER).write_bytes(b"raise AssertionError('later source executed')\n")
        return result

    monkeypatch.setattr(checked, "_read_bounded_public_member", read)
    monkeypatch.setattr(checked.hashlib, "sha256", replace_after_hash)
    gate = checked._load_tail_record_gate(package, provenance)
    assert gate.TOKEN == "captured source"
    assert replaced
    assert reads == [(package, HELPER, 1024 * 1024)]
    assert (package / HELPER).read_bytes().startswith(b"raise AssertionError")


def test_tail_loader_ignores_process_module_and_poisoned_bytecache(tmp_path, checked, monkeypatch):
    package = (tmp_path / "package").resolve()
    raw = b"TOKEN = 'authenticated source'\n"
    provenance = write_package(package, {HELPER: raw})
    source = package / HELPER
    source.write_bytes(b"TOKEN = 'unbound cached source'\n")
    cached = Path(py_compile.compile(str(source), doraise=True))
    source.write_bytes(raw)
    metadata = source.stat()
    bytecode = cached.read_bytes()
    cached.write_bytes(bytecode[:4] + (0).to_bytes(4, "little")
                       + (int(metadata.st_mtime) & 0xffffffff).to_bytes(4, "little")
                       + metadata.st_size.to_bytes(4, "little") + bytecode[16:])
    shadow = SimpleNamespace(TOKEN="unbound process cache")
    monkeypatch.setitem(sys.modules, "_openquantum_verified_tail_record_gate", shadow)
    gate = checked._load_tail_record_gate(package, provenance)
    assert gate.TOKEN == "authenticated source"
    assert sys.modules["_openquantum_verified_tail_record_gate"] is shadow


def test_package_tail_dispatch_keeps_all_prerequisites_and_independent_pins(tmp_path, checked, monkeypatch):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    package = (tmp_path / "package").resolve()
    provenance = write_package(package, {HELPER: saved_core_source(stop_after_complete=True)})
    order, loaded = [], []
    real_manifest, real_tail_loader = checked.verify_manifest, checked._load_tail_record_gate

    def manifest(path):
        assert path == package
        result = real_manifest(path)
        assert result == provenance
        order.append("manifest")
        return result

    def identity_loader(stage, verifier):
        def load(path, identity):
            assert path == package and identity == provenance
            order.append(stage + "-load")

            def verify(directory, source):
                assert directory == package / ("unused-" + stage) and source == package
                order.append(stage + "-check")

            return SimpleNamespace(RECORD_DIRECTORY="unused-" + stage, **{verifier: verify})
        return load

    def word(path, identity):
        assert path == package and identity == provenance
        order.append("word-load")

        def verify(root, *, replay):
            assert root == package and replay is True
            order.append("word-check")

        return SimpleNamespace(verify_public_word_transport=verify)

    def bundle_loader(stage):
        def load(path, identity):
            assert path == package and identity == provenance
            order.append(stage + "-load")

            def verify(directory, *, source_root):
                assert directory == package / ("unused-" + stage) and source_root == package
                order.append(stage + "-check")

            return SimpleNamespace(BUNDLE_RELATIVE="unused-" + stage, verify_bundle=verify)
        return load

    prerequisites = ["manifest", "formal", "energy-load", "energy-check", "dynamics-load", "dynamics-check",
                     "word-load", "word-check", "suzuki-load", "suzuki-check", "gibbs-load", "gibbs-check", "decay-load", "decay-check"]

    def tail(path, identity):
        assert path == package and identity == provenance and order == prerequisites
        gate = real_tail_loader(path, identity)
        loaded.append(gate)
        order.append("tail-load")
        original = gate.verify_bundle

        def verify(*args, **kwargs):
            order.append("tail-" + kwargs["kind"])
            return original(*args, **kwargs)

        gate.verify_bundle = verify
        return gate

    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda root: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", identity_loader("energy", "verify_energy_record_identity"))
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", identity_loader("dynamics", "verify_dynamics_record_identity"))
    monkeypatch.setattr(checked, "_load_word_record_gate", word)
    monkeypatch.setattr(checked, "_load_suzuki_record_gate", bundle_loader("suzuki"))
    monkeypatch.setattr(checked, "_load_gibbs_channel_record_gate", bundle_loader("gibbs"))
    monkeypatch.setattr(checked, "_load_decay_record_gate", bundle_loader("decay"))
    monkeypatch.setattr(checked, "_load_tail_record_gate", tail)
    with pytest.raises(RuntimeError, match="tail dispatch stopped before later gates"):
        checked.verify_package(package, run_tests=False)
    assert order == prerequisites + ["tail-load", "tail-instrument", "tail-complete"]
    assert loaded[0].CALLS == [
        (str(package / directory), str(package), kind, pin, False, None)
        for kind, directory, pin in BUNDLES
    ]


def configure_main(checker, root, monkeypatch):
    (root / "index.html").write_text("<html><body>Saved tail fixture</body></html>")
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda *_: [])
    for name in (
        "N3_EMBED_LABELS", "N4_EMBED_LABELS", "PORT_EMBED_LABELS", "QDRIFT_EMBED_LABELS",
        "SUZUKI_EMBED_LABELS", "SHOR_RECOVERY_EMBED_LABELS", "HADAMARD_EMBED_LABELS", "BAYESIAN_EMBED_LABELS", "ROBUST_EMBED_LABELS", "QPE_FEEDBACK_EMBED_LABELS",
        "OPT8_EMBED_LABELS", "GIBBS_CHANNEL_ARTICLE_LABELS", "DECAY_ARTICLE_LABELS", "CORRECTION_ARTICLE_LABELS",
    ):
        monkeypatch.setattr(checker, name, frozenset())
    monkeypatch.setattr(checker, "site_word_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_suzuki_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_gibbs_channel_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_decay_record_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_dqi_saved_errors", lambda *_, **__: [])
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
    monkeypatch.setattr(checker, "site_hamiltonian_packet_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "site_next_saved_errors", lambda *_, **__: [])
    monkeypatch.setattr(checker, "NEXT_EVIDENCE_ARTICLE_LABELS", {})
    monkeypatch.setattr(checker, "MEASURED_AND_FIGURE_LABELS", frozenset())
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])


def test_failed_complete_preview_never_dispatches_tail(synthetic_site, site_checker, monkeypatch):
    monkeypatch.setattr(site_checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(site_checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    root, _ = synthetic_site
    configure_main(site_checker, root, monkeypatch)
    order = []
    actual_tail = site_checker.site_tail_record_errors

    def preview(path):
        assert path == root
        order.append("preview")
        raise ValueError("complete archive rejected")

    def tail(*args, **kwargs):
        order.append("tail")
        return actual_tail(*args, **kwargs)

    monkeypatch.setattr(site_checker, "verify_preview", preview)
    monkeypatch.setattr(site_checker, "site_tail_record_errors", tail)
    with pytest.raises(SystemExit, match="complete archive rejected"):
        site_checker.main()
    assert order == ["preview"]


@pytest.mark.parametrize("mutation", ["missing", "wrong-schema", "untyped-retention", "changed-provenance"])
def test_site_provenance_failures_precede_tail_loading(synthetic_site, site_checker, monkeypatch, mutation):
    root, identity = synthetic_site
    if mutation == "missing":
        identity = {}
    elif mutation == "wrong-schema":
        identity["schema"] = "wrong"
    elif mutation == "untyped-retention":
        identity["retained_sha256"] = []
    else:
        (root / "code/PROVENANCE.json").write_text("{}")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Unbound preview provenance reached the tail loader")

    monkeypatch.setattr(site_checker, "_load_tail_record_gate", forbidden)
    assert site_checker.site_tail_record_errors(root, identity)
    assert calls == []


def test_site_authenticates_one_provenance_snapshot_and_exact_root_map(synthetic_site, site_checker, monkeypatch):
    root, identity = synthetic_site
    assert not (root / "code/assets").exists()
    real_read, real_loader = site_checker._read_bounded_public_member, site_checker._load_tail_record_gate
    reads, loaded = [], []

    def read(package, name, maximum):
        raw = real_read(package, name, maximum)
        reads.append((package, name, maximum))
        if name == "PROVENANCE.json":
            (package / name).write_text("{}")
        return raw

    def tail(package, provenance):
        gate = real_loader(package, provenance)
        loaded.append(gate)
        return gate

    monkeypatch.setattr(site_checker, "_read_bounded_public_member", read)
    monkeypatch.setattr(site_checker, "_load_tail_record_gate", tail)
    assert site_checker.site_tail_record_errors(root, identity) == []
    assert reads == [(root / "code", "PROVENANCE.json", 16 * 1024 * 1024)]
    assert loaded[0].CALLS == [
        (str(root / directory), str(root / "code"), kind, pin, False, str(root))
        for kind, directory, pin in BUNDLES
    ]
    assert (root / "code/PROVENANCE.json").read_text() == "{}"


def test_site_main_retains_preview_word_suzuki_gibbs_then_actual_tail(synthetic_site, site_checker, monkeypatch):
    monkeypatch.setattr(site_checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(site_checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    root, identity = synthetic_site
    configure_main(site_checker, root, monkeypatch)
    order, loaded = [], []
    actual_loader = site_checker._load_tail_record_gate

    def preview(path):
        assert path == root
        order.append("preview")
        return identity

    def existing(stage):
        def verify(path, authenticated):
            assert path == root and authenticated is identity
            order.append(stage)
            return []
        return verify

    def tail(package, provenance):
        assert package == root / "code" and order == ["preview", "word", "suzuki", "gibbs", "decay"]
        gate = actual_loader(package, provenance)
        loaded.append(gate)
        order.append("tail-load")
        return gate

    monkeypatch.setattr(site_checker, "verify_preview", preview)
    monkeypatch.setattr(site_checker, "site_word_record_errors", existing("word"))
    monkeypatch.setattr(site_checker, "site_suzuki_record_errors", existing("suzuki"))
    monkeypatch.setattr(site_checker, "site_gibbs_channel_record_errors", existing("gibbs"))
    monkeypatch.setattr(site_checker, "site_decay_record_errors", existing("decay"))
    monkeypatch.setattr(site_checker, "_load_tail_record_gate", tail)
    site_checker.main()
    assert order == ["preview", "word", "suzuki", "gibbs", "decay", "tail-load"]
    assert loaded[0].CALLS == [
        (str(root / directory), str(root / "code"), kind, pin, False, str(root))
        for kind, directory, pin in BUNDLES
    ]


def test_public_support_and_sidecar_contract_is_exact(exporter, site_checker):
    assert len(PUBLIC_SUPPORT) == 10
    assert all(exporter.PUBLIC_SCRIPTS.count(name) == 1 for name in PUBLIC_PYTHON)
    assert exporter.PUBLIC_SCRIPTS.count("tail_resource_qasm.py") == 1
    assert exporter.PUBLIC_TAIL_RESOURCE_POLICY_FILES == PUBLIC_POLICIES
    assert set(site_checker.TAIL_PUBLIC_SUPPORT_FILES) == PUBLIC_SUPPORT
    assert len(site_checker.TAIL_PUBLIC_SUPPORT_FILES) == 10
    assert site_checker.TAIL_RECORD_BUNDLES == BUNDLES
    required = set(site_checker.required_artifacts())
    assert {"code/" + name for name in PUBLIC_SUPPORT} <= required
    for _, directory, pin in BUNDLES:
        relative = directory + "/saved-resource-record.json"
        assert relative in required and exporter.include_generated_file(relative)
        assert digest((ROOT / relative).read_bytes()) == pin
    assert not any("linux_supervisor" in name or ".cache/" in name for name in PUBLIC_SUPPORT)


def test_policy_export_loop_copies_only_explicit_json_inputs(tmp_path, exporter, monkeypatch):
    root = tmp_path.resolve()
    mandatory = {
        "binder/assemble_code.py", "binder/postBuild", "binder/environment.yml", "binder/README.md",
        "citations.bib", "LICENSE", "requirements-lock.txt",
        "assets/generated/optimization/sampled/optimization.py",
        "assets/generated/optimization/sampled-runs.json", "assets/generated/grover/circuit.qasm",
        "assets/generated/resources/grover-surface-code.json", "assets/generated/resources/grover-surface-code.csv",
    } | PUBLIC_SUPPORT
    originals = {}
    for name in mandatory:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        originals[name] = ("saved fixture for " + name + "\n").encode()
        path.write_bytes(originals[name])
    (root / "scripts/unreviewed.json").write_text("{}")
    (root / ".cache").mkdir()
    (root / ".cache/linux_supervisor.py").write_text("raise AssertionError('private control')")
    monkeypatch.setattr(exporter, "PUBLIC_SCRIPTS", PUBLIC_PYTHON)
    monkeypatch.setattr(exporter, "PUBLIC_LOCAL_SPIN_SOURCE_FILES", ())
    monkeypatch.setattr(exporter, "PUBLIC_ARTIFACT_TREES", ())
    monkeypatch.setattr(exporter, "PUBLIC_QPE_APPLICATION_TINY_FIGURE_FILES", frozenset())
    monkeypatch.setattr(exporter, "FORMAL_FILES", ())
    monkeypatch.setattr(exporter, "git_revision", lambda _: "a" * 40)
    files = exporter.package_files(root)
    assert {name: files[name] for name in PUBLIC_SUPPORT} == {name: originals[name] for name in PUBLIC_SUPPORT}
    provenance = json.loads(files["PROVENANCE.json"])
    assert provenance["source_commit"] == "a" * 40
    assert {name: provenance["file_sha256"][name] for name in PUBLIC_POLICIES} == {
        name: digest(originals[name]) for name in PUBLIC_POLICIES}
    assert "scripts/unreviewed.json" not in files
    assert ".cache/linux_supervisor.py" not in files


def test_tail_ci_replacements_keep_exact_positions_and_final75_vector():
    workflow = ROOT / ".github/workflows/publish.yml"
    if not workflow.is_file():
        pytest.skip("Publication workflow is outside the public example package")
    commands = [line.strip().removeprefix("run ") for line in workflow.read_text().splitlines()
                if line.startswith("          run python ")]
    thermal_command = (
        "python scripts/verify_finite_gibbs_z_output_saved.py --bundle assets/generated/simulation/finite-gibbs-z-output "
        "--manifest-sha256 a490a8c5c1c941ce7a1b284de40879047bc21a8d3491aeceee0f5b05a878faa6 --max-seconds 30"
    )
    assert len(commands) == 93 and commands.count(thermal_command) == 1
    assert commands.index(thermal_command) == commands.index(
        "python scripts/verify_thermal_two_round_public.py --bundle assets/generated/simulation/detailed-balance-two-round"
    ) + 1
    commands = [command for command in commands if command != thermal_command]
    assert len(commands) == 92
    assert digest(("\n".join(commands) + "\n").encode()) == (
        "e1b4d0daa8f446888db525b96d3ca14f10bb05f02937e6a660320216abdc0fa8")
    qpe_command = (
        "python scripts/verify_qpe_application_tiny_saved.py --bundle assets/generated/qpe-application-tiny "
        "--manifest-sha256 62fafef64e565b3549dfc505ab975611f43730b11d8ab040406bee631eb7e4e3 "
        "--projection-sha256 9b2af033a7d68ae0c06e28bea70b9a5ed8380aa6310fe1d2c7938caf0c9e2e65 "
        "--figure assets/figures/qpe-application-tiny-kernel.svg "
        "--figure-sha256 2a482b68c0dcb68a5ff4ec84a373af5800163cd3ba171689a1fd4d1e34270a81 --max-seconds 30"
    )
    assert len(commands) == 92 and commands.count(qpe_command) == 1
    assert commands.index(qpe_command) == commands.index(
        "python scripts/verify_robust_qpe_resource_source_saved.py --bundle assets/generated/robust-qpe-resource-source "
        "--expected-manifest-sha256 aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33 "
        "--source-root . --max-seconds 30"
    ) + 1
    commands = [command for command in commands if command != qpe_command]
    assert len(commands) == 91
    assert digest(("\n".join(commands) + "\n").encode()) == (
        "4e8cb70a81b9ec393afbd34d4eb9fefc63f42104de0819d61695c4af9a637a8a")
    k2_command = (
        "python scripts/verify_dqi_path_majority_k2_saved.py --bundle-root assets/generated/optimization/dqi-path-majority-k2 "
        "--manifest-sha256 1294ccd3db9c4761cca476f6385ef15c29c0c2a3d18d18d83d71125c4a693ae7"
    )
    assert commands.count(k2_command) == 1
    commands = [command for command in commands if command != k2_command]
    added_commands = (
        'python scripts/verify_robust_qpe_resource_source_saved.py --bundle assets/generated/robust-qpe-resource-source --expected-manifest-sha256 aaa9eebc661db8e2c8a41764eb8497fa66d810934d9dd1f4bc3b414805378e33 --source-root . --max-seconds 30',
        'python scripts/verify_shor_finite_adder_reference.py --bundle assets/generated/shor/finite-adder-reference',
        'python scripts/verify_thermal_two_round_public.py --bundle assets/generated/simulation/detailed-balance-two-round',
        'python -I -S -B scripts/display_dqi_path_majority_saved.py --display --artifact-root . --source-root . --manifest-sha256 b6f6ad7f3e4914a6af7c34a9c852652e623b9d22de07635378ad379e816363d9 --time-limit-seconds 5',
    )
    assert all(commands.count(command) == 1 for command in added_commands)
    commands = [command for command in commands if command not in added_commands]
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
    assert digest(("\n".join(commands) + "\n").encode()) == "b0d163562bd0bc11a5a97bd74cde08d195cf5b09d879351fd685d52514340032"
    multiproduct = "python scripts/verify_simulation_multiproduct_saved.py"
    assert commands.count(multiproduct) == 1
    assert commands.index(multiproduct) == commands.index(
        "python scripts/verify_simulation_suzuki.py --replay-dir .cache/suzuki-ci-replay") + 1
    commands = [value for value in commands if value != multiproduct]
    assert len(commands) == 76
    assert digest(("\n".join(commands) + "\n").encode()) == "ad2f6c5fee4610b02748edb7663fcd7fd70316236f67b17c1613790d26fe68af"
    dqi = "python scripts/verify_dqi_saved.py assets/generated/optimization/dqi"
    assert commands.count(dqi) == 1
    commands = [value for value in commands if value != dqi]
    assert digest(("\n".join(commands) + "\n").encode()) == FINAL_CI_SHA
    inverse = {}
    for old, new in CI_REPLACEMENTS:
        assert commands.count(new) == 1 and old not in commands
        inverse[new] = old
    restored = [inverse.get(command, command) for command in commands]
    assert len(restored) == 75
    assert digest(("\n".join(restored) + "\n").encode()) == PRIOR_CI_SHA
    dynamics = "python scripts/verify_dynamics_resources.py --canonical-qasm-replay --records-dir assets/generated/dynamics-resources --manifest-sha256 0596353af49a0de40176069756cd9398749c21ede68e057ef84611bb6d5f4265 --max-wall-seconds 600"
    position = commands.index(dynamics)
    assert commands[position + 1:position + 4] == [CI_REPLACEMENTS[0][1], CI_REPLACEMENTS[1][1],
                                                  "python scripts/normalize_figures.py"]
    assert commands[-4:] == ["python scripts/execute_notebooks.py", "python scripts/export_code.py",
                            "python scripts/verify_public_package.py", "python scripts/publication_provenance.py"]
