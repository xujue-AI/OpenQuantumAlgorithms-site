"""Saved bath publication checks; no scientific source or producer runs.

Canonical files are read only to populate temporary fixtures. Every mutation,
normalization and mocked-corruption check operates on those fixtures or on
captured bytes. The actual bath identity and public dispatch paths are used.
"""
from __future__ import annotations

import builtins
from copy import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = "assets/generated/simulation/reusable-bath"
ARCHIVE = "assets/generated/simulation/reusable-bath-evidence.zip"
NOTEBOOK = "notebooks/simulation_reusable_bath.ipynb"
MANIFEST_PIN = "7f1a4c720b5576e1a9c27593bbd5001078a7c104bd21dfe13c759ed532304d78"
ARCHIVE_PIN = "4bb166a28e8621996b0eeca2f2f85b8d753f027131765eaef57a2477042efc62"
NAMES = frozenset({
    "artifact-manifest.json", "endpoints.json", "endpoints.png", "endpoints.svg",
    "fixture-ground.svg", "fixture-operations.json", "fixture-thermal.svg", "primitive-ledger.json",
    "results.json", "sampled-schedules.json", "source-exports.json", "source-ground.qasm",
    "source-thermal.qasm", "source/LICENSE.txt", "source/PROTOCOL-NOTE.md",
    "source/requirements-lock.txt", "source/reusable_bath.py", "source/reusable_bath_protocol.json",
    "source/reusable_bath_reference.py",
})
DEPENDENCIES = frozenset({DIRECTORY + "/source/" + name for name in
                          ("requirements-lock.txt", "LICENSE.txt", "PROTOCOL-NOTE.md")})
SVG_PATHS = (
    DIRECTORY + "/fixture-ground.svg", DIRECTORY + "/fixture-thermal.svg", DIRECTORY + "/endpoints.svg",
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_provenance(root):
    files = {path.relative_to(root).as_posix(): path.read_bytes()
             for path in root.rglob("*") if path.is_file() and path.name != "PROVENANCE.json"}
    hashes = {name: digest(raw) for name, raw in files.items()}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


@pytest.fixture
def checked():
    return load_source(ROOT / "scripts/verify_public_package.py", "bath_registration_package")


@pytest.fixture
def exporter():
    return load_source(ROOT / "scripts/export_code.py", "bath_registration_export")


@pytest.fixture(scope="module")
def saved_payload():
    """Only saved bytes are read; neither downloaded Python module is loaded."""
    paths = {DIRECTORY + "/" + name for name in NAMES} | {ARCHIVE, NOTEBOOK}
    payload = {name: (ROOT / name).read_bytes() for name in paths}
    assert digest(payload[DIRECTORY + "/artifact-manifest.json"]) == MANIFEST_PIN
    assert digest(payload[ARCHIVE]) == ARCHIVE_PIN
    assert sum(len(payload[DIRECTORY + "/" + name]) for name in NAMES) == 20_586_435
    return payload


@pytest.fixture
def packet(tmp_path, saved_payload, checked):
    root = (tmp_path / "package").resolve()
    root.mkdir()
    for name, raw in saved_payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = write_provenance(root)
    assert checked.verify_manifest(root) == provenance
    return root, provenance


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
    return load_source(path, "bath_registration_site")


@pytest.fixture
def site_packet(packet):
    root, provenance = packet
    code = root / "code"
    (code / "notebooks").mkdir(parents=True)
    (code / NOTEBOOK).write_bytes((root / NOTEBOOK).read_bytes())
    raw = (root / "PROVENANCE.json").read_bytes()
    (code / "PROVENANCE.json").write_bytes(raw)
    identity = {"schema": "openquantum-source-preview-v1", "retained_sha256": {"PROVENANCE.json": digest(raw)}}
    return root, code, provenance, identity


def test_actual_saved_integrity_uses_no_scientific_import_and_keeps_bytes(packet, saved_payload, checked, monkeypatch):
    root, provenance = packet
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"openquantum", "qiskit", "qdk", "qsharp", "numpy", "scipy", "matplotlib"}:
            pytest.fail("Saved bath verification imported a scientific/project module")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    result = checked._verify_bath_saved_identity(root, provenance=provenance, notebook_roots=(root,))
    assert result == {"status": "saved_bath_files_verified", "files": 19, "bytes": 20_586_435,
                      "all_components_verified": False, "new_quantum_execution": False}
    assert {name: (root / name).read_bytes() for name in saved_payload} == saved_payload


@pytest.mark.parametrize("bad", ["schema", "hashes", "roots-empty", "roots-many"])
def test_provenance_contract_rejects_before_any_saved_read(tmp_path, checked, monkeypatch, bad):
    provenance = {"schema": "openquantum-public-package-v2", "file_sha256": {}}
    roots = (tmp_path.resolve(),)
    if bad == "schema":
        provenance["schema"] = "wrong"
    elif bad == "hashes":
        provenance["file_sha256"] = []
    elif bad == "roots-empty":
        roots = ()
    else:
        roots = roots * 3

    def forbidden(*args, **kwargs):
        pytest.fail("Unauthenticated bath notebook context reached a content read")

    monkeypatch.setattr(checked, "_read_bounded_public_member", forbidden)
    with pytest.raises(ValueError, match="provenance and roots"):
        checked._verify_bath_saved_identity(tmp_path.resolve(), provenance=provenance, notebook_roots=roots)


def test_notebook_outputs_need_fresh_provenance_but_keep_same_source_contract(packet, checked):
    root, provenance = packet
    notebook = json.loads((root / NOTEBOOK).read_bytes())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 17
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "saved display only\n"}]
    notebook["metadata"]["synthetic_output_review"] = True
    (root / NOTEBOOK).write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="complete public provenance"):
        checked._verify_bath_saved_identity(root, provenance=provenance, notebook_roots=(root,))
    fresh = write_provenance(root)
    assert checked.verify_manifest(root) == fresh
    assert checked._verify_bath_saved_identity(root, provenance=fresh, notebook_roots=(root,))["new_quantum_execution"] is False


@pytest.mark.parametrize("change", ["source", "id", "type"])
def test_fresh_provenance_cannot_promote_a_changed_notebook_source(packet, checked, change):
    root, _ = packet
    notebook = json.loads((root / NOTEBOOK).read_bytes())
    cell = notebook["cells"][0]
    if change == "source":
        cell["source"] = ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]) + "\nchanged narrative"
    elif change == "id":
        cell["id"] += "-changed"
    else:
        cell["cell_type"] = "raw" if cell["cell_type"] != "raw" else "markdown"
    (root / NOTEBOOK).write_text(json.dumps(notebook))
    with pytest.raises(ValueError, match="notebook source changed"):
        checked._verify_bath_saved_identity(root, provenance=write_provenance(root), notebook_roots=(root,))


@pytest.mark.parametrize("extra", ["extra.json", "source/private_controller.py", "empty-directory"])
def test_curated_membership_rejects_extra_files_and_directories(packet, checked, extra):
    root, provenance = packet
    path = root / DIRECTORY / extra
    if extra == "empty-directory":
        path.mkdir()
    else:
        path.write_text("unrecorded")
    with pytest.raises(ValueError, match="unrecorded paths"):
        checked._verify_bath_saved_identity(root, provenance=provenance, notebook_roots=(root,))


@pytest.mark.parametrize("target", [DIRECTORY + "/artifact-manifest.json", DIRECTORY + "/source-ground.qasm", ARCHIVE])
def test_captured_manifest_payload_and_zip_corruption_fail_without_file_mutation(packet, saved_payload, checked, monkeypatch, target):
    root, provenance = packet
    original = checked._read_bounded_public_member

    def corrupt(directory, name, maximum):
        raw = original(directory, name, maximum)
        return bytes([raw[0] ^ 1]) + raw[1:] if name == target else raw

    monkeypatch.setattr(checked, "_read_bounded_public_member", corrupt)
    with pytest.raises(ValueError, match="identity changed"):
        checked._verify_bath_saved_identity(root, provenance=provenance, notebook_roots=(root,))
    assert (root / target).read_bytes() == saved_payload[target]


@pytest.mark.parametrize("fault", ["duplicate", "symlink", "wrong-size", "changed-member"])
def test_zip_member_structure_and_bytes_are_checked_after_the_real_zip_pin(packet, saved_payload, checked, monkeypatch, fault):
    root, provenance = packet
    with zipfile.ZipFile(io.BytesIO(saved_payload[ARCHIVE])) as original:
        members = [copy(item) for item in original.infolist()]
    if fault == "duplicate":
        members[-1] = copy(members[0])
    elif fault == "symlink":
        members[0].external_attr = (stat.S_IFLNK | 0o777) << 16
    elif fault == "wrong-size":
        members[0].file_size += 1
    opened = []

    class CorruptMemberView:
        def __init__(self, stream):
            assert stream.getvalue() == saved_payload[ARCHIVE]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def infolist(self):
            return members

        def open(self, item):
            opened.append(item.filename)
            name = item.filename.removeprefix("reusable-bath-evidence/")
            raw = saved_payload[DIRECTORY + "/" + name]
            if fault == "changed-member" and item is members[0]:
                raw = bytes([raw[0] ^ 1]) + raw[1:]
            return io.BytesIO(raw)

    monkeypatch.setattr(checked.zipfile, "ZipFile", CorruptMemberView)
    with pytest.raises(ValueError, match="ZIP"):
        checked._verify_bath_saved_identity(root, provenance=provenance, notebook_roots=(root,))
    if fault in {"duplicate", "symlink", "wrong-size"}:
        assert opened == []


def test_package_complete_manifest_failure_never_dispatches_bath(packet, checked, monkeypatch):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    root, _ = packet
    (root / NOTEBOOK).write_bytes(b"changed notebook before complete-manifest authentication")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Failed complete manifest reached a prerequisite or bath dispatch")

    for name in ("verify_formal_record", "_load_energy_record_gate", "_load_dynamics_record_gate",
                 "_load_word_record_gate", "_load_suzuki_record_gate", "_load_gibbs_channel_record_gate", "_load_decay_record_gate",
                 "_load_tail_record_gate", "_load_dqi_saved_gate", "_verify_dqi_saved_identity", "_verify_bath_saved_identity", "_load_local_spin_saved_gate"):
        monkeypatch.setattr(checked, name, forbidden)
    with pytest.raises(ValueError, match="digest mismatch"):
        checked.verify_package(root, run_tests=False)
    assert calls == []


def test_package_bath_dispatch_follows_complete_identity_and_dqi(packet, checked, monkeypatch):
    monkeypatch.setattr(checked, "_load_dqi_degree_two_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(checked, "_verify_dqi_degree_two_public_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(checked, "_load_correction_record_gate", lambda *args: SimpleNamespace(
        BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None,
        verify_download=lambda *args, **kwargs: None))
    root, provenance = packet
    expected_provenance = provenance
    order, results = [], []
    actual_manifest, actual_bath = checked.verify_manifest, checked._verify_bath_saved_identity

    def manifest(path):
        value = actual_manifest(path)
        assert path == root and value == provenance
        order.append("manifest")
        return value

    def identity(stage, method):
        def load(*args):
            order.append(stage)
            return SimpleNamespace(RECORD_DIRECTORY="unused", **{method: lambda *args: None})
        return load

    def bundle(stage):
        def load(*args):
            order.append(stage)
            return SimpleNamespace(BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None)
        return load

    def tail(*args):
        return SimpleNamespace(verify_bundle=lambda *args, **kwargs: order.append("tail-" + kwargs["kind"]))

    def bath(path, *, provenance, notebook_roots):
        assert order[-2:] == ["dqi-load", "dqi-check"]
        assert path == root and provenance == expected_provenance and notebook_roots == (root,)
        results.append(actual_bath(path, provenance=provenance, notebook_roots=notebook_roots))
        order.append("bath")
        raise RuntimeError("saved bath dispatch completed before downstream jobs")

    monkeypatch.setattr(checked, "verify_manifest", manifest)
    monkeypatch.setattr(checked, "verify_formal_record", lambda *args: order.append("formal"))
    monkeypatch.setattr(checked, "_load_energy_record_gate", identity("energy", "verify_energy_record_identity"))
    monkeypatch.setattr(checked, "_load_dynamics_record_gate", identity("dynamics", "verify_dynamics_record_identity"))
    monkeypatch.setattr(checked, "_load_word_record_gate", lambda *args: SimpleNamespace(verify_public_word_transport=lambda *args, **kwargs: order.append("word")))
    monkeypatch.setattr(checked, "_load_suzuki_record_gate", bundle("suzuki"))
    monkeypatch.setattr(checked, "_load_gibbs_channel_record_gate", bundle("gibbs"))
    monkeypatch.setattr(checked, "_load_decay_record_gate", bundle("decay"))
    monkeypatch.setattr(checked, "_load_tail_record_gate", tail)
    monkeypatch.setattr(checked, "_load_dqi_saved_gate", lambda *args: order.append("dqi-load"))
    monkeypatch.setattr(checked, "_verify_dqi_saved_identity", lambda *args, **kwargs: order.append("dqi-check"))
    monkeypatch.setattr(checked, "_verify_bath_saved_identity", bath)
    with pytest.raises(RuntimeError, match="before downstream jobs"):
        checked.verify_package(root, run_tests=False)
    assert order == ["manifest", "formal", "energy", "dynamics", "word", "suzuki", "gibbs", "decay", "tail-instrument", "tail-complete",
                     "dqi-load", "dqi-check", "bath"]
    assert results[0]["all_components_verified"] is results[0]["new_quantum_execution"] is False


def test_site_bath_uses_verified_provenance_and_canonical_assets_without_code_assets(site_packet, site_checker, monkeypatch):
    root, code, provenance, identity = site_packet
    assert not (code / "assets").exists()
    actual = site_checker._verify_bath_saved_identity
    calls = []

    def bath(path, *, provenance, notebook_roots):
        calls.append((path, provenance, notebook_roots))
        return actual(path, provenance=provenance, notebook_roots=notebook_roots)

    monkeypatch.setattr(site_checker, "_verify_bath_saved_identity", bath)
    assert site_checker.site_bath_saved_errors(root, identity) == []
    assert calls == [(root, provenance, (code, root))]
    (code / "PROVENANCE.json").write_text("{}")
    calls.clear()
    assert site_checker.site_bath_saved_errors(root, identity)
    assert calls == []


def test_failed_complete_preview_never_calls_actual_bath_dispatch(site_packet, site_checker, monkeypatch):
    monkeypatch.setattr(site_checker, "site_dqi_degree_two_saved_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(site_checker, "site_correction_record_errors", lambda *args, **kwargs: [])
    root, _, _, _ = site_packet
    (root / "index.html").write_text("<html><body>Saved bath preview fixture</body></html>")
    monkeypatch.setattr(site_checker, "required_artifacts", lambda: [])
    if hasattr(site_checker, "_site_redirects"):
        monkeypatch.setattr(site_checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(site_checker, "current_quality_resource_errors", lambda *args: [])
    for name in ("N3_EMBED_LABELS", "N4_EMBED_LABELS", "PORT_EMBED_LABELS", "QDRIFT_EMBED_LABELS",
                 "SUZUKI_EMBED_LABELS", "SHOR_RECOVERY_EMBED_LABELS", "HADAMARD_EMBED_LABELS", "BAYESIAN_EMBED_LABELS", "ROBUST_EMBED_LABELS", "QPE_FEEDBACK_EMBED_LABELS",
                 "OPT8_EMBED_LABELS", "GIBBS_CHANNEL_ARTICLE_LABELS", "DECAY_ARTICLE_LABELS", "CORRECTION_ARTICLE_LABELS", "MEASURED_AND_FIGURE_LABELS"):
        monkeypatch.setattr(site_checker, name, frozenset())
    for name in ("site_word_record_errors", "site_suzuki_record_errors", "site_gibbs_channel_record_errors", "site_decay_record_errors",
                 "site_tail_record_errors", "site_dqi_saved_errors", "site_local_spin_saved_errors", "site_local_spin_reproduction_errors", "site_measured_and_saved_errors", "site_hamiltonian_packet_errors"):
        monkeypatch.setattr(site_checker, name, lambda *args, **kwargs: [])
    calls = []
    actual = site_checker.site_bath_saved_errors

    def bath(*args, **kwargs):
        calls.append((args, kwargs))
        return actual(*args, **kwargs)

    def rejected(*args):
        raise ValueError("complete bath preview/archive rejected")

    monkeypatch.setattr(site_checker, "site_bath_saved_errors", bath)
    monkeypatch.setattr(site_checker, "verify_preview", rejected)
    monkeypatch.setattr(site_checker.sys, "argv", ["check_site.py", str(root)])
    with pytest.raises(SystemExit, match="complete bath preview/archive rejected"):
        site_checker.main()
    assert calls == []


def test_export_allowlist_dependencies_and_download_requirements_are_exact(exporter, site_checker):
    expected = {DIRECTORY + "/" + name for name in NAMES}
    assert len(expected) == 19 and exporter.PUBLIC_BATH_FILES == expected
    assert DEPENDENCIES <= exporter.PUBLIC_DEPENDENCY_FILES
    assert {name for name in exporter.PUBLIC_DEPENDENCY_FILES if name.startswith(DIRECTORY + "/")} == DEPENDENCIES
    assert ARCHIVE in exporter.PUBLIC_ARCHIVES
    assert "assets/generated/grover-sat/iteration-circuits.zip" in exporter.PUBLIC_ARCHIVES
    assert all(exporter.include_generated_file(name) for name in expected | {ARCHIVE})
    for name in ("private-controller.py", "raw-log.json", "template.ipynb", "extra.svg", "nested/result.json"):
        assert not exporter.include_generated_file(DIRECTORY + "/" + name)
    assert not exporter.include_generated_file("assets/generated/simulation/unreviewed-bath.zip")
    required = set(site_checker.required_artifacts())
    assert expected | {ARCHIVE, NOTEBOOK, "algorithms/simulation/reusable-bath/index.html",
                       "notebooks/simulation-reusable-bath/index.html"} <= required
    assert not any(name.startswith("code/" + DIRECTORY) for name in required)


def test_bath_navigation_has_one_partial_article_and_one_normal_companion():
    config = ROOT / "myst.yml"
    if not config.is_file():
        pytest.skip("Book routing is outside the public example package")
    import yaml
    tree = yaml.safe_load(config.read_text())["project"]["toc"]

    def find(node, path):
        if isinstance(node, dict):
            return ([node] if node.get("file") == path else []) + [item for child in node.values() for item in find(child, path)]
        if isinstance(node, list):
            return [item for child in node for item in find(child, path)]
        return []

    article, = find(tree, "algorithms/simulation/reusable-bath.md")
    assert article["title"] == "Reusable bath qubit (partial)"
    companion, = find(article.get("children", []), NOTEBOOK)
    assert companion["file"] == NOTEBOOK
    assert len(find(tree, NOTEBOOK)) == 1
    assert sorted(path.name for path in (ROOT / "notebooks").glob("*reusable*bath*.ipynb")) == [
        "simulation_reusable_bath.ipynb", "simulation_reusable_bath_local_spin.ipynb"]


def test_normalization_preserves_exact_bath_and_hadamard_paths_but_normalizes_an_ordinary_svg(tmp_path):
    root = tmp_path.resolve()
    script = root / "scripts/normalize_figures.py"
    script.parent.mkdir()
    script.write_bytes((ROOT / "scripts/normalize_figures.py").read_bytes())
    protected = SVG_PATHS + ("assets/generated/hadamard-qpe/circuit.svg", "assets/generated/hadamard-qpe/recovery.svg")
    raw = b"<svg>  \n  saved original bytes  \n</svg>\n\n"
    for name in protected:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    ordinary = root / "assets/generated/ordinary/circuit.svg"
    ordinary.parent.mkdir(parents=True)
    ordinary.write_bytes(raw)
    load_source(script, "bath_normalizer_temporary_fixture")
    assert all((root / name).read_bytes() == raw for name in protected)
    assert ordinary.read_bytes() == b"<svg>\n  saved original bytes\n</svg>\n\n"


def test_actual_export_loop_carries_the_closed_bath_and_zip_without_private_neighbors(packet, saved_payload, exporter, monkeypatch):
    root, _ = packet
    mandatory = {"binder/assemble_code.py", "binder/postBuild", "binder/environment.yml", "binder/README.md",
                 "citations.bib", "LICENSE", "requirements-lock.txt",
                 "assets/generated/optimization/sampled/optimization.py", "assets/generated/optimization/sampled-runs.json",
                 "assets/generated/grover/circuit.qasm", "assets/generated/resources/grover-surface-code.json",
                 "assets/generated/resources/grover-surface-code.csv"}
    for name in mandatory:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"unrelated export fixture\n")
    (root / DIRECTORY / "private-controller.py").write_bytes(b"raise AssertionError('private neighbor')\n")
    monkeypatch.setattr(exporter, "PUBLIC_SCRIPTS", ())
    monkeypatch.setattr(exporter, "PUBLIC_BOOTSTRAP_FILES", ())
    monkeypatch.setattr(exporter, "PUBLIC_TAIL_RESOURCE_POLICY_FILES", ())
    monkeypatch.setattr(exporter, "PUBLIC_LOCAL_SPIN_SOURCE_FILES", ())
    monkeypatch.setattr(exporter, "PUBLIC_QPE_APPLICATION_TINY_FIGURE_FILES", ())
    monkeypatch.setattr(exporter, "PUBLIC_ARTIFACT_TREES", ("simulation",))
    monkeypatch.setattr(exporter, "FORMAL_FILES", ())
    monkeypatch.setattr(exporter, "git_revision", lambda _: "a" * 40)
    files = exporter.package_files(root)
    assert {name: files[name] for name in saved_payload} == saved_payload
    assert DIRECTORY + "/private-controller.py" not in files
    provenance = json.loads(files["PROVENANCE.json"])
    assert provenance["file_sha256"][ARCHIVE] == ARCHIVE_PIN
