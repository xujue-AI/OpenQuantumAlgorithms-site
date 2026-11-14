"""Authenticated saved decay publication; no scientific backend is imported."""
from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "assets/generated/simulation/detailed-balance-decay"
HELPER = "scripts/verify_detailed_balance_decay.py"
NOTEBOOK = "notebooks/simulation_detailed_balance_decay.ipynb"
TOP_PIN = "9acc1af928c3484a8ad7c24a5768a1c4ac643cc02c541e5ad0df2b0df1223f94"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_script(name):
    path = ROOT / "scripts" / (name + ".py")
    specification = importlib.util.spec_from_file_location("decay_registration_" + name, path)
    module = importlib.util.module_from_spec(specification)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def write_provenance(root):
    hashes = {path.relative_to(root).as_posix(): digest(path.read_bytes())
              for path in root.rglob("*") if path.is_file() and path.name != "PROVENANCE.json"}
    provenance = {"schema": "openquantum-public-package-v2", "source_commit": "a" * 40,
                  "file_sha256": hashes,
                  "source_sha256": {name: value for name, value in hashes.items() if name.endswith(".py")}}
    (root / "PROVENANCE.json").write_text(json.dumps(provenance, sort_keys=True))
    return provenance


@pytest.fixture
def package_gate():
    return load_script("verify_public_package")


@pytest.fixture(scope="module")
def payload():
    saved = load_script("verify_detailed_balance_decay")
    names = {BUNDLE + "/" + name for name in saved.BUNDLE_FILES} | {
        "LICENSE", "requirements-lock.txt", "scripts/generate_detailed_balance_decay.py",
        "src/openquantum/simulation/detailed_balance_access.py",
        "src/openquantum/simulation/detailed_balance_decay.py",
        "src/openquantum/simulation/detailed_balance_decay_reference.py",
        "tests/test_detailed_balance_decay.py", NOTEBOOK, HELPER,
    }
    buffers = {name: (ROOT / name).read_bytes() for name in names}
    assert digest(buffers[BUNDLE + "/artifact-manifest.json"]) == TOP_PIN
    assert len(saved.BUNDLE_FILES) == 28
    assert sum(len(buffers[BUNDLE + "/" + name]) for name in saved.BUNDLE_FILES) == 483_911
    return buffers


@pytest.fixture
def packet(tmp_path, payload, package_gate):
    root = (tmp_path / "package").resolve()
    root.mkdir()
    for name, raw in payload.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = write_provenance(root)
    assert package_gate.verify_manifest(root) == provenance
    return root, provenance


@pytest.fixture
def site_gate(package_gate):
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "site_decay_record_errors"]
    assert len(selected) == 1
    namespace = {"Path": Path, "hashlib": hashlib, "json": json,
                 "_read_bounded_public_member": package_gate._read_bounded_public_member,
                 "_load_decay_record_gate": package_gate._load_decay_record_gate}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["site_decay_record_errors"]


@pytest.fixture
def site_packet(tmp_path, payload):
    root = (tmp_path / "site").resolve()
    code = root / "code"
    code.mkdir(parents=True)
    for name, raw in payload.items():
        base = root if name.startswith("assets/") else code
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    provenance = write_provenance(code)
    identity = {"schema": "openquantum-source-preview-v1",
                "retained_sha256": {"PROVENANCE.json": digest((code / "PROVENANCE.json").read_bytes())}}
    return root, code, provenance, identity


def block_science(monkeypatch):
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"openquantum", "qiskit", "qdk", "qsharp", "numpy", "scipy", "matplotlib"}:
            pytest.fail("Saved decay registration imported a scientific backend")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_authenticated_helper_checks_actual_saved_packet_without_science(packet, package_gate, payload, monkeypatch):
    root, provenance = packet
    block_science(monkeypatch)
    gate = package_gate._load_decay_record_gate(root, provenance)
    gate.verify_bundle(root / BUNDLE, source_root=root)
    assert {name: (root / name).read_bytes() for name in payload} == payload


@pytest.mark.parametrize("change", ["missing-provenance", "wrong-schema", "missing-hash", "helper-mutation"])
def test_loader_rejects_untrusted_helper_bytes_before_compilation(packet, package_gate, change):
    root, provenance = packet
    if change == "missing-provenance":
        provenance = None
    elif change == "wrong-schema":
        provenance["schema"] = "wrong"
    elif change == "missing-hash":
        provenance["file_sha256"].pop(HELPER)
    else:
        (root / HELPER).write_bytes((root / HELPER).read_bytes() + b"\nraise RuntimeError('untrusted helper executed')\n")
    with pytest.raises(ValueError, match="decay helper differs"):
        package_gate._load_decay_record_gate(root, provenance)


def test_package_dispatch_waits_for_complete_manifest_and_preserves_gate_order(tmp_path):
    path = ROOT / "scripts/verify_public_package.py"
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "verify_package")
    observed = []
    root = tmp_path.resolve()
    provenance = {"schema": "test-authenticated"}

    class StopAfterDecay(Exception):
        pass

    def loader(kind):
        def load(package, identity):
            assert package == root and identity is provenance
            observed.append("load-" + kind)
            return SimpleNamespace(
                RECORD_DIRECTORY="records", BUNDLE_RELATIVE=BUNDLE,
                verify_energy_record_identity=lambda *args: observed.append("energy"),
                verify_dynamics_record_identity=lambda *args: observed.append("dynamics"),
                verify_public_word_transport=lambda *args, **kwargs: observed.append("word"),
                verify_bundle=lambda bundle, *, source_root: observed.append((kind, bundle, source_root)),
                verify_download=lambda archive, bundle: observed.append(("correction-download", archive, bundle)),
            )
        return load

    def stop(*args):
        observed.append("tail")
        raise StopAfterDecay

    namespace = {"Path": Path, "checked_directory": lambda value: value,
                 "verify_manifest": lambda value: observed.append("manifest") or provenance,
                 "verify_formal_record": lambda value: observed.append("formal"),
                 "_load_tail_record_gate": stop}
    for kind, name in (("energy", "_load_energy_record_gate"), ("dynamics", "_load_dynamics_record_gate"),
                       ("word", "_load_word_record_gate"), ("suzuki", "_load_suzuki_record_gate"),
                       ("gibbs", "_load_gibbs_channel_record_gate"), ("decay", "_load_decay_record_gate"),
                       ("correction", "_load_correction_record_gate")):
        namespace[name] = loader(kind)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    with pytest.raises(StopAfterDecay):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest", "formal", "load-energy", "energy", "load-dynamics", "dynamics",
                        "load-word", "word", "load-suzuki", ("suzuki", root / BUNDLE, root),
                        "load-gibbs", ("gibbs", root / BUNDLE, root), "load-decay", ("decay", root / BUNDLE, root),
                        "load-correction", ("correction", root / BUNDLE, root),
                        ("correction-download", root / "assets/generated/simulation/detailed-balance-correction-download.zip", root / BUNDLE), "tail"]
    observed.clear()

    def reject(value):
        observed.append("manifest-rejected")
        raise ValueError("invalid complete manifest")

    namespace["verify_manifest"] = reject
    with pytest.raises(ValueError, match="invalid complete manifest"):
        namespace["verify_package"](root, run_tests=False)
    assert observed == ["manifest-rejected"]


def test_site_uses_authenticated_preview_and_canonical_assets(site_packet, site_gate, monkeypatch):
    root, code, _, identity = site_packet
    block_science(monkeypatch)
    assert site_gate(root, identity) == []
    assert not (code / BUNDLE).exists()


@pytest.mark.parametrize("change", ["missing-preview", "wrong-schema", "provenance", "helper", "asset", "missing-asset", "source"])
def test_site_rejects_preview_helper_asset_and_source_mutation(site_packet, site_gate, change):
    root, code, _, identity = site_packet
    if change == "missing-preview":
        identity = {}
    elif change == "wrong-schema":
        identity["schema"] = "wrong"
    elif change == "provenance":
        (code / "PROVENANCE.json").write_bytes((code / "PROVENANCE.json").read_bytes() + b"\n")
    elif change == "helper":
        (code / HELPER).write_bytes((code / HELPER).read_bytes() + b"\n")
    elif change == "source":
        source = code / "src/openquantum/simulation/detailed_balance_decay.py"
        source.write_bytes(source.read_bytes() + b"\n")
    else:
        asset = root / BUNDLE / "grid/results.json"
        if change == "missing-asset":
            asset.unlink()
        else:
            asset.write_bytes(asset.read_bytes() + b"\n")
    errors = site_gate(root, identity)
    assert len(errors) == 1 and errors[0].startswith("Saved decay artifact verification failed:")


def test_site_decay_dispatch_occurs_only_after_archive_preview_verification():
    path = ROOT / "scripts/check_site.py"
    if not path.is_file():
        pytest.skip("Rendered-site caller is deliberately outside the examples package")
    tree = ast.parse(path.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    block = next(node for node in main.body if isinstance(node, ast.Try)
                 and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "verify_preview"
                         for item in node.body for call in ast.walk(item)))
    calls = [node for item in block.orelse for node in ast.walk(item)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "site_decay_record_errors"]
    assert len(calls) == 1
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "site_decay_record_errors"
                   for item in block.body + [item for handler in block.handlers for item in handler.body]
                   for node in ast.walk(item))
