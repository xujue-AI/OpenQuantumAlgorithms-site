"""Saved correction publication boundaries without numerical execution.

Scientific sources and QASM remain inert bytes. The actual standard-library
reader is used only for saved identities; synthetic AST dispatch checks stop
before scientific imports, subprocesses or any later publication work.
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/simulation/detailed-balance-correction"
ARCHIVE = "assets/generated/simulation/detailed-balance-correction-download.zip"
HELPER = "scripts/verify_detailed_balance_correction_saved.py"
HELPER_PIN = "e04bd5c9dec44319a25fba0ac8778d06b62785304fa335b37aadf878fb521871"
ARCHIVE_PIN = "79ffc7f1184dd04c3f9a7227b973826dff0095d3ab286b2bd8d1cfb57f64f891"
TOP_PIN = "93bf5c0047a05b7c009b590629ca9b6a9661e5dc46da97d510f9f3b621208929"
V2_PIN = "1127038ca780bb63315e6f867dbe87e3da963cabc4c667a2aa6a62deac6be02f"
CORE_NAMES = frozenset({
    "README.md", "UPSTREAM-LICENSES.md", "artifact-manifest.json",
    "artifact-manifest.sha256", "completion.json", "v1/centered_decay.qasm",
    "v1/failure.json", "v2/LICENSE.txt", "v2/centered_decay.qasm",
    "v2/controlled_correction.qasm", "v2/correction_block.qasm",
    "v2/detailed_balance_access.py", "v2/detailed_balance_channel.py",
    "v2/detailed_balance_correction.py", "v2/detailed_balance_correction_reference.py",
    "v2/detailed_balance_decay.py", "v2/detailed_balance_decay_reference.py",
    "v2/generate_detailed_balance_correction.py", "v2/heralded_step.qasm",
    "v2/known_plus_prefix.qasm", "v2/known_plus_trial.qasm", "v2/manifest.json",
    "v2/manifest.sha256", "v2/requirements-lock.txt", "v2/results.json",
})
PUBLIC_FILES = frozenset(BUNDLE + "/" + name for name in CORE_NAMES) | {ARCHIVE}
SOURCE_BINDINGS = {
    "LICENSE.txt": "LICENSE",
    "requirements-lock.txt": "requirements-lock.txt",
    "detailed_balance_access.py": "src/openquantum/simulation/detailed_balance_access.py",
    "detailed_balance_channel.py": "src/openquantum/simulation/detailed_balance_channel.py",
    "detailed_balance_decay.py": "src/openquantum/simulation/detailed_balance_decay.py",
    "detailed_balance_decay_reference.py": "src/openquantum/simulation/detailed_balance_decay_reference.py",
    "detailed_balance_correction.py": "src/openquantum/simulation/detailed_balance_correction.py",
    "detailed_balance_correction_reference.py": "src/openquantum/simulation/detailed_balance_correction_reference.py",
    "generate_detailed_balance_correction.py": "scripts/generate_detailed_balance_correction.py",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_script(name, *, source_root=ROOT):
    path = source_root / "scripts" / (name + ".py")
    specification = importlib.util.spec_from_file_location("correction_registration_" + name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_provenance(root):
    hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes())
              for path in root.rglob("*") if path.is_file() and path.name != "PROVENANCE.json"}
    value = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
             "file_sha256": hashes,
             "source_sha256": {name: pin for name, pin in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(value, sort_keys=True))
    return value


def reject_science(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"openquantum", "qiskit", "numpy", "scipy", "qdk", "cirq"}:
            pytest.fail("Saved correction registration reached a scientific import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def package_gate():
    return load_script("verify_public_package")


@pytest.fixture(scope="module")
def payload():
    names = PUBLIC_FILES | set(SOURCE_BINDINGS.values()) | {HELPER}
    return {name: (ROOT / name).read_bytes() for name in names}


@pytest.fixture
def packet(tmp_path, payload):
    root = (tmp_path / "package").resolve()
    root.mkdir()
    for name, raw in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return root, write_provenance(root)


def test_frozen_reader_packet_archive_and_source_closure(payload):
    assert len(CORE_NAMES) == 25 and len(PUBLIC_FILES) == 26
    assert len(payload[HELPER]) == 33_005 and digest(payload[HELPER]) == HELPER_PIN
    assert len(payload[ARCHIVE]) == 100_053 and digest(payload[ARCHIVE]) == ARCHIVE_PIN
    assert sum(len(payload[BUNDLE + "/" + name]) for name in CORE_NAMES) == 455_665
    v2 = [name for name in CORE_NAMES if name.startswith("v2/")]
    assert len(v2) == 18 and sum(len(payload[BUNDLE + "/" + name]) for name in v2) == 434_352
    assert digest(payload[BUNDLE + "/artifact-manifest.json"]) == TOP_PIN
    assert digest(payload[BUNDLE + "/v2/manifest.json"]) == V2_PIN
    for snapshot, current in SOURCE_BINDINGS.items():
        assert payload[BUNDLE + "/v2/" + snapshot] == payload[current]


def test_actual_authenticated_reader_checks_saved_bundle_and_download(packet, package_gate, monkeypatch):
    root, provenance = packet
    reject_science(monkeypatch)
    gate = package_gate._load_correction_record_gate(root, provenance)
    result = gate.verify_bundle(root / BUNDLE, source_root=root)
    assert result["status"] == "verified_saved_finite_correction_identity"
    assert result["bundle_files"] == 25 and result["bundle_bytes"] == 455_665
    assert result["original_v2_files"] == 18 and result["original_v2_bytes"] == 434_352
    assert result["historical_kraus_outcomes"] == 32 and result["source_root_checked"] is True
    for key in ("numerical_replay_performed", "all_components_verified", "new_algorithm_result",
                "physical_resources_computed"):
        assert result[key] is False
    download = gate.verify_download(root / ARCHIVE, root / BUNDLE)
    assert download == {"download_verified": True, "download_files": 26,
                        "download_bytes": 100_053, "download_sha256": ARCHIVE_PIN}


@pytest.mark.parametrize("change", [
    "missing-provenance", "wrong-schema", "missing-hashes", "missing-helper-hash",
    "wrong-helper-hash", "missing-archive-hash", "wrong-archive-hash", "changed-helper", "oversized-helper",
])
def test_loader_rejects_unbound_helper_and_archive_before_compile(packet, package_gate, monkeypatch, change):
    root, provenance = packet
    if change == "missing-provenance":
        provenance = None
    elif change == "wrong-schema":
        provenance["schema"] = "wrong"
    elif change == "missing-hashes":
        provenance.pop("file_sha256")
    elif change == "missing-helper-hash":
        provenance["file_sha256"].pop(HELPER)
    elif change == "wrong-helper-hash":
        provenance["file_sha256"][HELPER] = "0" * 64
    elif change == "missing-archive-hash":
        provenance["file_sha256"].pop(ARCHIVE)
    elif change == "wrong-archive-hash":
        provenance["file_sha256"][ARCHIVE] = "0" * 64
    else:
        path = root / HELPER
        path.write_bytes(path.read_bytes() + (b" " * 65_536 if change == "oversized-helper" else b"\nraise RuntimeError('untrusted source')\n"))
        provenance["file_sha256"][HELPER] = digest(path.read_bytes())
    monkeypatch.setattr(builtins, "compile", lambda *args, **kwargs: pytest.fail("Unbound helper reached compile"))
    with pytest.raises((OSError, ValueError)):
        package_gate._load_correction_record_gate(root, provenance)


def test_loader_ignores_an_untrusted_module_cache(packet, package_gate, monkeypatch):
    root, provenance = packet
    original = package_gate.importlib.util.spec_from_file_location
    names = []
    def spec(name, path):
        names.append(name)
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(untrusted_marker=True))
        return original(name, path)
    monkeypatch.setattr(package_gate.importlib.util, "spec_from_file_location", spec)
    gate = package_gate._load_correction_record_gate(root, provenance)
    assert len(names) == 1 and not getattr(gate, "untrusted_marker", False)
    assert gate.BUNDLE_RELATIVE == BUNDLE


def test_loader_compiles_the_captured_authenticated_bytes(packet, package_gate, monkeypatch):
    root, provenance = packet
    original = package_gate._read_bounded_public_member
    helper_reads = []
    def read(base, name, maximum):
        raw = original(base, name, maximum)
        if name == HELPER:
            helper_reads.append(raw)
            (root / HELPER).write_bytes(b"raise RuntimeError('later on-disk source must not execute')\n")
        return raw
    monkeypatch.setattr(package_gate, "_read_bounded_public_member", read)
    gate = package_gate._load_correction_record_gate(root, provenance)
    assert len(helper_reads) == 1 and digest(helper_reads[0]) == HELPER_PIN
    assert gate.BUNDLE_RELATIVE == BUNDLE


@pytest.mark.parametrize("name", [
    "artifact-manifest.json", "v2/manifest.json", "v2/results.json",
    "v2/known_plus_trial.qasm", "v1/failure.json",
])
def test_changed_frozen_bundle_member_is_rejected(packet, package_gate, name):
    root, provenance = packet
    gate = package_gate._load_correction_record_gate(root, provenance)
    path = root / BUNDLE / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        gate.verify_bundle(root / BUNDLE, source_root=root)


@pytest.mark.parametrize("name", sorted(SOURCE_BINDINGS.values()))
def test_changed_selected_source_is_rejected(packet, package_gate, name):
    root, provenance = packet
    gate = package_gate._load_correction_record_gate(root, provenance)
    path = root / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source correspondence"):
        gate.verify_bundle(root / BUNDLE, source_root=root)


@pytest.mark.parametrize("name", ["unexpected.json", "v2/private.log", "v1/extra.py", "notebooks/unexecuted.ipynb"])
def test_bundle_extra_files_are_rejected(packet, package_gate, name):
    root, provenance = packet
    gate = package_gate._load_correction_record_gate(root, provenance)
    path = root / BUNDLE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unregistered bytes\n")
    with pytest.raises(ValueError):
        gate.verify_bundle(root / BUNDLE, source_root=root)


def test_recompressed_archive_cannot_bypass_the_container_pin(packet, package_gate, monkeypatch):
    root, provenance = packet
    gate = package_gate._load_correction_record_gate(root, provenance)
    with zipfile.ZipFile(root / ARCHIVE, "a") as archive:
        archive.comment = b"same closed contents, different container"
    assert digest((root / ARCHIVE).read_bytes()) != ARCHIVE_PIN
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("Changed ZIP reached saved member dispatch"))
    with pytest.raises(ValueError):
        gate.verify_download(root / ARCHIVE, root / BUNDLE)


def test_changed_second_zip_snapshot_is_rejected_before_decoder(packet, package_gate, monkeypatch):
    root, provenance = packet
    archive_path = root / ARCHIVE
    canonical = archive_path.read_bytes()
    repacked_path = root / "repacked-download.zip"
    repacked_path.write_bytes(canonical)
    with zipfile.ZipFile(repacked_path, "a") as archive:
        archive.comment = b"different second snapshot with the same closed members"
    substituted = repacked_path.read_bytes()
    assert digest(canonical) == ARCHIVE_PIN and digest(substituted) != ARCHIVE_PIN

    decoder_calls = []
    def decoder(*args, **kwargs):
        decoder_calls.append((args, kwargs))
        pytest.fail("Substituted second ZIP snapshot reached the decoder")
    monkeypatch.setattr(zipfile, "ZipFile", decoder)
    gate = package_gate._load_correction_record_gate(root, provenance)
    assert zipfile.ZipFile is decoder

    outer_read = package_gate._read_bounded_public_member
    first_snapshot_pins = []
    def bounded_read(base, name, maximum):
        raw = outer_read(base, name, maximum)
        if Path(base) == archive_path.parent and name == archive_path.name:
            first_snapshot_pins.append(digest(raw))
        return raw
    monkeypatch.setattr(package_gate, "_read_bounded_public_member", bounded_read)

    helper_read = gate._read
    helper_archive_reopens = []
    def reopened_read(base, name, *args, **kwargs):
        if Path(base) == archive_path.parent and name == archive_path.name:
            helper_archive_reopens.append(name)
            return substituted
        return helper_read(base, name, *args, **kwargs)
    monkeypatch.setattr(gate, "_read", reopened_read)

    with pytest.raises(ValueError):
        gate.verify_download(archive_path, root / BUNDLE)
    assert first_snapshot_pins == [ARCHIVE_PIN]
    assert helper_archive_reopens == [archive_path.name]
    assert decoder_calls == [] and zipfile.ZipFile is decoder
    assert archive_path.read_bytes() == canonical
    assert digest((root / HELPER).read_bytes()) == HELPER_PIN


@pytest.mark.parametrize("member", [HELPER, ARCHIVE, BUNDLE + "/v2/results.json"])
def test_linked_helper_archive_or_bundle_member_is_rejected(packet, package_gate, member):
    root, provenance = packet
    path = root / member
    retained = root / "retained-original"
    path.rename(retained)
    path.symlink_to(retained)
    with pytest.raises((OSError, ValueError)):
        gate = package_gate._load_correction_record_gate(root, provenance)
        if member == ARCHIVE:
            gate.verify_download(root / ARCHIVE, root / BUNDLE)
        elif member != HELPER:
            gate.verify_bundle(root / BUNDLE, source_root=root)


def test_export_contract_is_closed_without_private_registration_documents():
    export = load_script("export_code")
    assert export.PUBLIC_CORRECTION_FILES == PUBLIC_FILES
    degree_two_archive = "assets/generated/optimization/dqi-degree-two-public/dqi-degree-two-download.zip"
    assert degree_two_archive in export.PUBLIC_ARCHIVES and len(export.PUBLIC_ARCHIVES) == 9
    assert ARCHIVE in export.PUBLIC_ARCHIVES and len(export.PUBLIC_ARCHIVES - {degree_two_archive}) == 8
    assert {"generate_detailed_balance_correction.py", "verify_detailed_balance_correction_saved.py"} <= set(export.PUBLIC_SCRIPTS)
    assert all(export.include_generated_file(name) for name in PUBLIC_FILES)
    assert not any(name.startswith(("docs/", ".git/", ".cache/")) for name in PUBLIC_FILES)


@pytest.mark.parametrize("name", [
    "unexpected.json", "extra.py", "v1/private.log", "v2/unregistered.sha256",
    "v2/notebooks/unexecuted.ipynb", "./artifact-manifest.json", "../detailed-balance-correction/README.md",
])
def test_export_rejects_extra_correction_paths_and_aliases(name):
    export = load_script("export_code")
    assert not export.include_generated_file(BUNDLE + "/" + name)


def test_package_admission_precedes_correction_and_both_calls_follow_decay(tmp_path):
    path = ROOT / "scripts/verify_public_package.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "verify_package")
    root = tmp_path.resolve()
    identity = {"schema": "synthetic admitted identity"}
    observed = []
    class StopAtTail(Exception):
        pass
    def loader(kind):
        def load(package, provenance):
            assert package == root and provenance is identity
            observed.append("load-" + kind)
            return SimpleNamespace(
                RECORD_DIRECTORY="records", BUNDLE_RELATIVE=BUNDLE,
                verify_energy_record_identity=lambda *args: observed.append("energy"),
                verify_dynamics_record_identity=lambda *args: observed.append("dynamics"),
                verify_public_word_transport=lambda *args, **kwargs: observed.append("word"),
                verify_bundle=lambda bundle, *, source_root: observed.append((kind, bundle, source_root)),
                verify_download=lambda archive, bundle: observed.append(("download", archive, bundle)),
            )
        return load
    def stop(*args):
        observed.append("tail")
        raise StopAtTail
    namespace = {"Path": Path, "checked_directory": lambda value: value,
                 "verify_manifest": lambda value: observed.append("manifest") or identity,
                 "verify_formal_record": lambda value: observed.append("formal"),
                 "_load_tail_record_gate": stop}
    for kind in ("energy", "dynamics", "word", "suzuki", "gibbs", "decay", "correction"):
        namespace["_load_" + ("gibbs_channel" if kind == "gibbs" else kind) + "_record_gate"] = loader(kind)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(StopAtTail):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest", "formal", "load-energy", "energy", "load-dynamics", "dynamics",
                        "load-word", "word", "load-suzuki", ("suzuki", root / BUNDLE, root),
                        "load-gibbs", ("gibbs", root / BUNDLE, root), "load-decay", ("decay", root / BUNDLE, root),
                        "load-correction", ("correction", root / BUNDLE, root),
                        ("download", root / ARCHIVE, root / BUNDLE), "tail"]
    observed.clear()
    def reject(_):
        observed.append("manifest-rejected")
        raise ValueError("invalid complete manifest")
    namespace["verify_manifest"] = reject
    with pytest.raises(ValueError, match="invalid complete manifest"):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest-rejected"]


@pytest.fixture
def site_function(package_gate):
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "site_correction_record_errors")
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "zipfile": zipfile,
                 "_read_bounded_public_member": package_gate._read_bounded_public_member,
                 "_load_correction_record_gate": package_gate._load_correction_record_gate}
    namespace.update({name: value for name, value in vars(package_gate).items() if name.startswith("CORRECTION_")})
    selected = [item for item in tree.body if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id.startswith("CORRECTION_") for target in item.targets)]
    exec(compile(ast.Module(body=[*selected, node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["site_correction_record_errors"]


@pytest.fixture
def site_packet(tmp_path, payload):
    root = (tmp_path / "site").resolve()
    code = root / "code"
    code.mkdir(parents=True)
    retained, omitted = {}, {}
    hashes = {name: digest(raw) for name, raw in payload.items()}
    for name, raw in payload.items():
        if name.startswith("assets/"):
            path = root / name
            omitted[name] = {"site_path": name, "sha256": digest(raw)}
        else:
            path = code / name
            retained[name] = digest(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: pin for name, pin in hashes.items() if name.endswith(".py")}}
    raw = json.dumps(provenance, sort_keys=True).encode()
    (code / "PROVENANCE.json").write_bytes(raw)
    retained["PROVENANCE.json"] = digest(raw)
    preview = {"schema": "openquantum-source-preview-v1", "retained_sha256": retained, "omitted_assets": omitted}
    return root, code, preview


def test_site_checks_canonical_assets_and_authenticated_code_source_root(site_packet, site_function, monkeypatch):
    root, code, preview = site_packet
    reject_science(monkeypatch)
    assert not (code / BUNDLE).exists()
    assert site_function(root, preview) == []


@pytest.mark.parametrize("change", [
    "missing-preview", "wrong-schema", "provenance-digest", "provenance-bytes",
    "helper", "source", "bundle", "archive", "missing-archive",
])
def test_site_rejects_provenance_helper_source_bundle_or_archive_changes(site_packet, site_function, change):
    root, code, preview = site_packet
    if change == "missing-preview":
        preview = {}
    elif change == "wrong-schema":
        preview["schema"] = "wrong"
    elif change == "provenance-digest":
        preview["retained_sha256"]["PROVENANCE.json"] = "0" * 64
    elif change == "provenance-bytes":
        (code / "PROVENANCE.json").write_bytes((code / "PROVENANCE.json").read_bytes() + b"\n")
    elif change == "helper":
        (code / HELPER).write_bytes((code / HELPER).read_bytes() + b"\n")
    elif change == "source":
        path = code / "src/openquantum/simulation/detailed_balance_correction.py"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "bundle":
        path = root / BUNDLE / "v2/results.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "archive":
        (root / ARCHIVE).write_bytes((root / ARCHIVE).read_bytes() + b"\n")
    else:
        (root / ARCHIVE).unlink()
    errors = site_function(root, preview)
    assert len(errors) == 1 and errors[0].startswith("Saved correction artifact verification failed:")


def test_site_correction_call_is_inside_successful_preview_admission():
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = next(node for node in main.body if isinstance(node, ast.Try)
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "verify_preview"
                         for item in node.body for call in ast.walk(item)))
    calls = [node for item in block.orelse for node in ast.walk(item)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "site_correction_record_errors"]
    assert len(calls) == 1
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "site_correction_record_errors"
                   for item in block.body + [item for handler in block.handlers for item in handler.body]
                   for node in ast.walk(item))


def test_site_requires_closed_downloads_complete_sources_and_all_article_anchors():
    checker = ROOT / "scripts/check_site.py"
    article = ROOT / "algorithms/simulation/detailed-balance-gibbs.md"
    if not checker.is_file() or not article.is_file():
        pytest.skip("Book-only routes and article are deliberately outside the examples package")
    tree = ast.parse(checker.read_text())
    labels = next(ast.literal_eval(node.value.args[0]) for node in tree.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id == "CORRECTION_ARTICLE_LABELS" for target in node.targets))
    assert labels == {"gibbs-correction-step"}
    text = article.read_text()
    for name in {"gibbs-correction-step", "gibbs-channel-circuit", "gibbs-channel-checked-records",
                 "gibbs-channel-output-contracts", "gibbs-channel-resource-projection",
                 "gibbs-channel-saved-verification", "gibbs-decay-block"}:
        assert "(" + name + ")=" in text
    paths = next(ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                 and target.id == "CORRECTION_PUBLIC_PATHS" for target in node.targets))
    assert {"code/" + name for name in SOURCE_BINDINGS.values()} | {"code/" + HELPER} <= set(paths)
    required = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "required_artifacts")
    assert any(isinstance(node, ast.Name) and node.id == "PUBLIC_CORRECTION_FILES" for node in ast.walk(required))
