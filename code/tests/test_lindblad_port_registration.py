"""Port release registration calls the immutable saved gate and retains scope."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location("port_registration_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = load_script("export_code")
gate = load_script("verify_simulation_lindblad_port")
package_verify = load_script("verify_public_package")


def test_port_allowlist_is_exact_reviewed_bundle_and_dependencies():
    directory = ROOT / gate.BUNDLE_RELATIVE
    actual = {path.relative_to(ROOT).as_posix() for path in directory.rglob("*") if path.is_file()}
    assert len(actual) == 33
    assert actual == export.PUBLIC_LINDBLAD_PORT_FILES == {f"{gate.BUNDLE_RELATIVE}/{name}" for name in gate.BUNDLE_FILES}
    assert {path for path in export.PUBLIC_DEPENDENCY_FILES if path.startswith(gate.BUNDLE_RELATIVE + "/")} == {
        gate.BUNDLE_RELATIVE + "/requirements-lock.txt", gate.BUNDLE_RELATIVE + "/references.bib",
    }
    assert {"run_simulation_lindblad_port.py", "verify_simulation_lindblad_port.py"} <= set(export.PUBLIC_SCRIPTS)
    for path in actual:
        assert export.include_generated_file(path), path
    assert sum(name.endswith(".py") for name in gate.SOURCE_FILES) == 12
    assert len(gate.SOURCE_FILES) == 15


@pytest.mark.parametrize("relative", [
    "notes.json", "raw-native-Trace.json", "native-profile-value.json", "private.log", "plan.md",
    "checkpoints/native.json", "words/word-08.qasm", "notes.txt", "private.bib", "results.json.gz",
    "./results.json", "words/../results.json",
])
def test_port_allowlist_rejects_unrecorded_artifacts_and_aliases(relative):
    assert not export.include_generated_file(gate.BUNDLE_RELATIVE + "/" + relative)


def test_port_allowlist_rejects_doubled_separator_alias():
    assert not export.include_generated_file("assets/generated//simulation/lindblad-port/results.json")


def test_saved_companion_identity_and_labels_remain_exact():
    path = ROOT / "notebooks/simulation_lindblad_port.ipynb"
    raw = path.read_bytes()
    notebook = json.loads(raw)
    cells = [{"id": cell["id"], "cell_type": cell["cell_type"],
              "source": "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]}
             for cell in notebook["cells"]]
    encoded = json.dumps(cells, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == "d016e59a2da60c4f5c6fceae0f55981bda1f1c0ed124e5da50737ce96294a7ca"
    embeds = [{"id": cell["id"], "label": cell.get("metadata", {}).get("label")}
              for cell in notebook["cells"]]
    encoded = json.dumps(embeds, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == "a83e7dbb62ea53dc3ddd3b83e164959e175224c60789da2ed79b366511bf6bdb"
    labels = {cell.get("metadata", {}).get("label") for cell in notebook["cells"]}
    assert {"simulation-lindblad-port-" + name for name in
            ("words", "recipe", "density", "budget", "controller", "resources")} <= labels
    assert len(notebook["cells"]) == 16


@pytest.fixture
def site_checker():
    if not (ROOT / "scripts/check_site.py").is_file():
        pytest.skip("Site checker is deliberately outside the complete public example package")
    return load_script("check_site")


def test_site_contract_uses_canonical_artifacts_and_all_current_source_maps(site_checker):
    required = set(site_checker.required_artifacts())
    assert {f"{gate.BUNDLE_RELATIVE}/{name}" for name in gate.BUNDLE_FILES} <= required
    assert {f"code/{path}" for path in gate.SOURCE_FILES.values() if path is not None} <= required
    assert {"code/references/simulation.bib", "code/references/resources.bib",
            "code/scripts/verify_simulation_lindblad_port.py",
            "notebooks/simulation_lindblad_port.ipynb", "notebooks/simulation-lindblad-port/index.html"} <= required
    assert not any(path.startswith("code/assets/generated/simulation/lindblad-port/") for path in required)


def configure_saved_site(checker, tmp_path, monkeypatch, *, omit_label=None, reject_gate=False):
    root = tmp_path.resolve()
    (root / "index.html").write_text("<html><body>Saved registration fixture</body></html>")
    companion = root / checker.PORT_NOTEBOOK_ROUTE
    companion.parent.mkdir(parents=True)
    companion.write_text("<html><body>" + "".join(f'<div id="{label}"></div>' for label in sorted(checker.PORT_EMBED_LABELS)
                                               if label != omit_label) + "</body></html>")
    calls = []
    def old_n4(*args, **kwargs):
        calls.append(("n4", args, kwargs))
        return {"physical_status": "infeasible_declared_controller_model", "native_calls": 0}
    def port(bundle, **kwargs):
        calls.append(("port", (bundle,), kwargs))
        if reject_gate:
            raise ValueError("saved port binding rejected")
        return {"native_trace_digest_status": "audit-trusted", "saved_native_profile_value_json_verified": True,
                "native_calls": 0, "density_replays": 0}
    monkeypatch.setattr(checker, "required_artifacts", lambda: [])
    if hasattr(checker, "_site_redirects"):
        monkeypatch.setattr(checker._site_redirects, "REDIRECT_FRAGMENTS", {}, raising=False)
    monkeypatch.setattr(checker, "ALGORITHMS", [])
    monkeypatch.setattr(checker, "N3_EMBED_LABELS", frozenset())
    monkeypatch.setattr(checker, "N4_EMBED_LABELS", frozenset())
    for name in ("QDRIFT_EMBED_LABELS", "SHOR_RECOVERY_EMBED_LABELS", "HADAMARD_EMBED_LABELS", "BAYESIAN_EMBED_LABELS", "ROBUST_EMBED_LABELS", "QPE_FEEDBACK_EMBED_LABELS",
                 "OPT8_EMBED_LABELS", "SUZUKI_EMBED_LABELS", "GIBBS_CHANNEL_ARTICLE_LABELS"):
        monkeypatch.setattr(checker, name, frozenset())
    monkeypatch.setattr(checker, "verify_n4_bundle", old_n4)
    monkeypatch.setattr(checker, "verify_port_bundle", port)
    for name in ("verify_n2_bundle", "verify_n3_bundle", "verify_opt7_bundle", "verify_preview",
                 "verify_qdrift_bundle", "verify_hadamard_record"):
        monkeypatch.setattr(checker, name, lambda *args, **kwargs: {})
    monkeypatch.setattr(checker, "current_quality_resource_errors", lambda root: [])
    monkeypatch.setattr(checker, "site_energy_record_errors", lambda root: [])
    for name in ("site_dynamics_record_errors", "site_word_record_errors", "site_suzuki_record_errors",
                 "site_gibbs_channel_record_errors", "site_tail_record_errors",
                 "site_dqi_saved_errors", "site_bath_saved_errors", "site_multiproduct_saved_errors",
                 "site_bayesian_saved_errors", "site_robust_saved_errors", "site_dqi_structured_saved_errors", "site_dqi_resource_saved_errors", "site_qpe_feedback_saved_errors", "site_local_spin_saved_errors", "site_local_spin_reproduction_errors", "site_measured_and_saved_errors"):
        monkeypatch.setattr(checker, name, lambda *args, **kwargs: [])
    monkeypatch.setattr(checker, "MEASURED_AND_FIGURE_LABELS", frozenset())
    monkeypatch.setattr(checker.sys, "argv", ["check_site.py", str(root)])
    return root, calls


def test_site_dispatch_retains_old_n4_gate_and_uses_preview_source_maps(site_checker, tmp_path, monkeypatch):
    root, calls = configure_saved_site(site_checker, tmp_path, monkeypatch)
    site_checker.main()
    assert [name for name, _, _ in calls] == ["n4", "port"]
    assert calls[1][1] == (root / gate.BUNDLE_RELATIVE,)
    assert calls[1][2] == {"source_root": root / "code", "canonical_root": root}


def test_site_dispatch_propagates_port_gate_rejection(site_checker, tmp_path, monkeypatch):
    configure_saved_site(site_checker, tmp_path, monkeypatch, reject_gate=True)
    with pytest.raises(SystemExit, match="Lindblad port artifact verification failed: saved port binding rejected"):
        site_checker.main()


@pytest.mark.parametrize("suffix", ["words", "recipe", "density", "budget", "controller", "resources"])
def test_site_dispatch_requires_each_saved_embed(site_checker, tmp_path, monkeypatch, suffix):
    label = "simulation-lindblad-port-" + suffix
    configure_saved_site(site_checker, tmp_path, monkeypatch, omit_label=label)
    with pytest.raises(SystemExit, match="Missing Lindblad port notebook/embed label: " + label):
        site_checker.main()


@pytest.mark.parametrize("reject_port", [False, True])
def test_package_dispatch_calls_saved_port_gate_without_producer_or_native_execution(tmp_path, monkeypatch, reject_port):
    root = tmp_path.resolve()
    calls = []
    monkeypatch.setattr(package_verify, "verify_manifest", lambda package: {})
    monkeypatch.setattr(package_verify, "verify_formal_record", lambda package: None)
    monkeypatch.setattr(package_verify, "_load_energy_record_gate", lambda *args: SimpleNamespace(
        RECORD_DIRECTORY="assets/generated/energy-estimation", verify_energy_record_identity=lambda *args: None))
    monkeypatch.setattr(package_verify, "_load_dynamics_record_gate", lambda *args: SimpleNamespace(
        RECORD_DIRECTORY="unused", verify_dynamics_record_identity=lambda *args: None))
    monkeypatch.setattr(package_verify, "_load_word_record_gate", lambda *args: SimpleNamespace(
        verify_public_word_transport=lambda *args, **kwargs: None))
    for name in ("_load_suzuki_record_gate", "_load_gibbs_channel_record_gate"):
        monkeypatch.setattr(package_verify, name, lambda *args: SimpleNamespace(
            BUNDLE_RELATIVE="unused", verify_bundle=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_tail_record_gate", lambda *args: SimpleNamespace(
        verify_bundle=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_dqi_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(package_verify, "_verify_dqi_saved_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_verify, "_verify_bath_saved_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_verify, "_load_multiproduct_saved_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_bayesian_saved_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_robust_saved_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_qpe_feedback_public_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_local_spin_saved_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_verify_local_spin_reproduction_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_verify, "_load_measured_and_saved_gate", lambda *args: SimpleNamespace(
        verify_saved=lambda *args, **kwargs: None))
    monkeypatch.setattr(package_verify, "_load_dqi_structured_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(package_verify, "_load_dqi_resource_saved_gate", lambda *args: SimpleNamespace())
    monkeypatch.setattr(package_verify, "_verify_dqi_resource_saved_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_verify, "_verify_dqi_structured_saved_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_verify, "checked_path", lambda root, name: root / name)
    class Loader:
        def exec_module(self, module):
            return None
    def spec(name, path):
        return SimpleNamespace(name=name, path=path, loader=Loader())
    def module(specification):
        basename = specification.path.name
        relative = {"verify_simulation_lindblad_three.py": "assets/generated/simulation/lindblad-three",
                    "verify_simulation_lindblad_four.py": "assets/generated/simulation/lindblad-four",
                    "verify_simulation_lindblad_port.py": gate.BUNDLE_RELATIVE,
                    "verify_optimization_compiled.py": "assets/generated/optimization/compiled-replay"}[basename]
        def verify(bundle, **kwargs):
            calls.append((basename, bundle, kwargs))
            if basename == "verify_simulation_lindblad_port.py" and reject_port:
                raise ValueError("saved port package binding rejected")
            return {"native_calls": 0}
        return SimpleNamespace(BUNDLE_RELATIVE=relative, verify_bundle=verify)
    monkeypatch.setattr(package_verify.importlib.util, "spec_from_file_location", spec)
    monkeypatch.setattr(package_verify.importlib.util, "module_from_spec", module)
    subprocess_calls = []
    monkeypatch.setattr(package_verify.subprocess, "run", lambda *args, **kwargs: subprocess_calls.append((args, kwargs)))
    if reject_port:
        with pytest.raises(ValueError, match="saved port package binding rejected"):
            package_verify.verify_package(root, run_tests=False)
        assert [name for name, _, _ in calls] == ["verify_simulation_lindblad_three.py", "verify_simulation_lindblad_four.py",
                                                 "verify_simulation_lindblad_port.py"]
        assert subprocess_calls == []
        return
    package_verify.verify_package(root, run_tests=False)
    assert [name for name, _, _ in calls] == ["verify_simulation_lindblad_three.py", "verify_simulation_lindblad_four.py",
                                             "verify_simulation_lindblad_port.py", "verify_optimization_compiled.py"]
    port = calls[2]
    assert port[1:] == (root / gate.BUNDLE_RELATIVE, {"source_root": root, "canonical_root": root})
    assert not any("run_simulation_lindblad_port.py" in str(arguments) for arguments, _ in subprocess_calls)
