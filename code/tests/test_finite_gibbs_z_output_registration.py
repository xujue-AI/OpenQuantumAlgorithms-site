"""Saved-byte boundaries; no scientific source, QASM or matrix replay.

The small synthetic fixture retargets only file/hash trust roots. It retains
the fixed scope constants while replacing opaque numerical subtrees with null
data and scientific source with import sentinels. It is a protocol fixture,
not additional scientific evidence. CLI tests use fake resource/clock hooks.
"""
from __future__ import annotations

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "assets/generated/simulation/finite-gibbs-z-output"
HELPER = ROOT / "scripts/verify_finite_gibbs_z_output_saved.py"
MANIFEST_SHA256 = "a490a8c5c1c941ce7a1b284de40879047bc21a8d3491aeceee0f5b05a878faa6"
ORIGINAL_RESULT_SHA256 = "f5e0b772fa2005f415af19d3cb4ba94395753a9172a308dd868b8058acbb5129"
PUBLIC_RESULT_SHA256 = "8329d568e2df8f317f97610165385123f52dc0f5d47c6aa190838405966ca33f"
DENIED_IMPORTS = frozenset({
    "numpy", "scipy", "qiskit", "qdk", "openquantum", "decimal", "subprocess",
    "detailed_balance_access", "detailed_balance_channel", "finite_gibbs_interval",
    "finite_gibbs_angle_provenance", "finite_gibbs_reference", "reproduce",
})


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


@pytest.fixture
def gate(monkeypatch):
    original_import = builtins.__import__
    attempts = []
    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in DENIED_IMPORTS:
            attempts.append(name)
            raise AssertionError("Saved gate attempted a scientific import")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    spec = importlib.util.spec_from_file_location("finite_gibbs_byte_gate_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._test_denied_import_attempts = attempts
    return module


def rewrite_synthetic(case, gate, monkeypatch):
    """Coherently refresh synthetic bytes, without refreshing scope constants."""
    files = dict(case.files)
    files["results.json"] = encoded(case.results)
    case.summary["public_results_sha256"] = (
        case.forced_public_result_sha256 or sha(files["results.json"]))
    for item in case.derivations["derivations"]:
        raw = files[item["public_path"]]
        item["public_source_bytes"] = len(raw)
        item["public_source_sha256"] = sha(raw)
    files["source-derivations.json"] = encoded(case.derivations)
    files["summary.json"] = encoded(case.summary)
    pins = {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}
    case.manifest["payloads"] = [dict(path=name, **pin) for name, pin in sorted(pins.items())]
    case.manifest["payload_count"] = len(pins)
    case.manifest["payload_bytes"] = sum(pin["bytes"] for pin in pins.values())
    raw_manifest = encoded(case.manifest)
    for name, raw in {**files, "manifest.json": raw_manifest}.items():
        path = case.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    case.files = files
    case.manifest_sha256 = sha(raw_manifest)
    monkeypatch.setattr(gate, "EXPECTED_PAYLOAD_PINS", pins)
    monkeypatch.setattr(gate, "FILE_NAMES", frozenset(pins))
    monkeypatch.setattr(gate, "EXPECTED_MANIFEST_SHA256", case.manifest_sha256)


@pytest.fixture
def tiny(tmp_path, gate, monkeypatch):
    manifest = json.loads((PACKET / "manifest.json").read_bytes())
    summary = json.loads((PACKET / "summary.json").read_bytes())
    original_results = json.loads((PACKET / "results.json").read_bytes())
    retained_dicts = {"scoped_acceptance", "parameters", "source_counts",
                      "measurement_projection", "implementation_checks"}
    results = {name: copy.deepcopy(value) if type(value) in (bool, str, int)
               or name in retained_dicts else None
               for name, value in original_results.items()}
    derivations = json.loads((PACKET / "source-derivations.json").read_bytes())
    files = {item["path"]: ("inert protocol bytes: " + item["path"] + "\n").encode()
             for item in manifest["payloads"]}
    for name in files:
        if name.startswith("sources/"):
            files[name] = b"raise AssertionError('Scientific source was imported')\n"
    case = SimpleNamespace(root=(tmp_path / "packet").resolve(), manifest=manifest,
                           summary=summary, results=results, derivations=derivations,
                           files=files, forced_public_result_sha256=None)
    rewrite_synthetic(case, gate, monkeypatch)
    return case


def test_source_imports_and_dispatch_are_standard_library_only():
    tree = ast.parse(HELPER.read_bytes())
    imported = {alias.name.split(".", 1)[0] for node in ast.walk(tree)
                if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module.split(".", 1)[0] for node in ast.walk(tree)
                 if isinstance(node, ast.ImportFrom) and node.module}
    assert imported <= sys.stdlib_module_names | {"__future__"}
    assert not imported & DENIED_IMPORTS
    calls = {node.func.id for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert not calls & {"exec", "eval", "compile", "__import__"}


def test_authenticates_original_public_packet_without_new_scientific_work(gate):
    result = gate.verify_saved_bundle(PACKET, MANIFEST_SHA256)
    assert result["files_authenticated"] == 22
    assert result["payloads_authenticated"] == 21
    assert result["source_snapshots_authenticated"] == 6
    assert result["bundle_bytes_authenticated"] == 3431942
    assert result["original_saved_result_sha256"] == ORIGINAL_RESULT_SHA256
    assert result["files"]["results.json"]["sha256"] == PUBLIC_RESULT_SHA256
    assert result["fixed_conditional_Z_mean_contract_accepted"] is True
    assert result["raw_flags"] == {
        "status": "observed_fixed_finite_Gibbs_Z_components",
        "finite_gibbs_Z_output_bound_observed": True,
        "finite_gibbs_Z_output_bound_passed": False,
        "whole_invocation_validated": False,
    }
    assert all(type(value) is bool and value is False for value in result["raw_scope_flags"].values())
    assert result["saved_json_parsed"] is True
    for name in ("scientific_target_imports_performed", "scientific_replay_performed",
                 "qasm_parsed", "matrix_arithmetic_performed", "interval_arithmetic_recomputed",
                 "source_derivation_equivalence_reproduced", "sampling_performed",
                 "portable_wrapper_executed", "full_R64_primitive_simulation_performed",
                 "full_state_Gibbs_accuracy_certified", "formal_proof_reproduced"):
        assert result[name] is False
    assert gate._test_denied_import_attempts == []


def test_synthetic_metadata_keeps_acceptance_and_historical_flags_separate(tiny, gate):
    result = gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256)
    assert result["fixed_conditional_Z_mean_contract_accepted"] is True
    assert result["raw_flags"]["finite_gibbs_Z_output_bound_observed"] is True
    assert result["raw_flags"]["finite_gibbs_Z_output_bound_passed"] is False
    assert result["raw_flags"]["whole_invocation_validated"] is False
    assert result["measurement_projection"]["samples"] == 16384
    assert result["measurement_projection"]["sampling_performed"] is False
    assert gate._test_denied_import_attempts == []


@pytest.mark.parametrize("mutation", (
    "final-pass", "whole-pass", "wrapper-pass", "general-sampler", "full-state",
    "shots", "false-as-zero", "zero-as-false", "samples-as-float", "rounds-as-bool",
    "delta-as-string", "original-public-swap", "public-original-swap", "acceptance-hash",
    "unknown-result-claim",
))
def test_coherent_synthetic_hash_rebind_cannot_change_scientific_scope(
        tiny, gate, monkeypatch, mutation):
    if mutation == "final-pass":
        tiny.manifest["original_raw_flags_preserved"]["finite_gibbs_Z_output_bound_passed"] = True
        tiny.summary["raw_final_flags"]["finite_gibbs_Z_output_bound_passed"] = True
        tiny.results["finite_gibbs_Z_output_bound_passed"] = True
    elif mutation == "whole-pass":
        tiny.manifest["original_raw_flags_preserved"]["whole_invocation_validated"] = True
        tiny.summary["raw_final_flags"]["whole_invocation_validated"] = True
        tiny.results["whole_invocation_validated"] = True
        tiny.results["scoped_acceptance"]["whole_invocation_validated"] = True
    elif mutation == "wrapper-pass":
        tiny.manifest["portable_wrapper_executed"] = True
        tiny.derivations["derivations"][-1]["portable_wrapper_executed"] = True
    elif mutation == "general-sampler":
        tiny.manifest["general_sampler_or_mixing_theorem"] = True
        tiny.results["scoped_acceptance"]["general_sampler_or_mixing_theorem"] = True
    elif mutation == "full-state":
        tiny.manifest["full_state_Gibbs_accuracy_certified"] = True
        tiny.results["scoped_acceptance"]["full_state_Gibbs_accuracy_certified"] = True
    elif mutation == "shots":
        tiny.manifest["shots_drawn"] = 16384
        tiny.results["scoped_acceptance"]["actual_shots"] = 16384
    elif mutation == "false-as-zero":
        tiny.results["physical_resources_computed"] = 0
    elif mutation == "zero-as-false":
        tiny.results["scoped_acceptance"]["actual_shots"] = False
    elif mutation == "samples-as-float":
        tiny.results["measurement_projection"]["samples"] = 16384.0
    elif mutation == "rounds-as-bool":
        tiny.results["parameters"]["rounds"] = True
    elif mutation == "delta-as-string":
        tiny.results["parameters"]["delta"] = "0.125"
    elif mutation == "original-public-swap":
        tiny.manifest["original_saved_result_sha256"] = PUBLIC_RESULT_SHA256
        tiny.summary["original_result_sha256"] = PUBLIC_RESULT_SHA256
        tiny.results["original_result_sha256"] = PUBLIC_RESULT_SHA256
    elif mutation == "public-original-swap":
        tiny.forced_public_result_sha256 = ORIGINAL_RESULT_SHA256
    elif mutation == "acceptance-hash":
        tiny.manifest["original_root_acceptance_sha256"] = "0" * 64
        tiny.results["scoped_acceptance"]["acceptance_record_sha256"] = "0" * 64
    else:
        tiny.results["universal_sampler_validated"] = True
    rewrite_synthetic(tiny, gate, monkeypatch)
    with pytest.raises(ValueError):
        gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256)
    assert gate._test_denied_import_attempts == []


@pytest.mark.parametrize("member", ("manifest.json", "circuits/access-U.qasm", "sources/reproduce.py"))
def test_byte_tampering_fails_before_saved_scope_checks(tiny, gate, monkeypatch, member):
    path = tiny.root / member
    path.write_bytes(path.read_bytes() + b"changed\n")
    monkeypatch.setattr(gate, "_saved_contracts", lambda *_: pytest.fail("Changed bytes reached metadata"))
    with pytest.raises(ValueError):
        gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256)


def test_duplicate_manifest_member_cannot_replace_an_omitted_member(tiny, gate, monkeypatch):
    value = copy.deepcopy(tiny.manifest)
    value["payloads"][-1] = copy.deepcopy(value["payloads"][0])
    raw = encoded(value)
    (tiny.root / "manifest.json").write_bytes(raw)
    monkeypatch.setattr(gate, "EXPECTED_MANIFEST_SHA256", sha(raw))
    with pytest.raises(ValueError, match="Duplicate"):
        gate.verify_saved_bundle(tiny.root, sha(raw))


@pytest.mark.parametrize("extra", ("extra.json", "extra.py", "extra.qasm", "private-empty"))
def test_unknown_suffix_or_empty_directory_is_not_admitted(tiny, gate, monkeypatch, extra):
    path = tiny.root / extra
    path.mkdir() if extra == "private-empty" else path.write_bytes(b"unexpected\n")
    monkeypatch.setattr(gate, "_hash_file", lambda *_a, **_k: pytest.fail("Extra entry reached reading"))
    with pytest.raises(ValueError):
        gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256)


@pytest.mark.parametrize("alias", (".", "./results.json", "circuits//access-U.qasm", "../results.json",
                                    "/results.json", "sources\\reproduce.py"))
def test_manifest_path_aliases_are_rejected(alias, gate):
    with pytest.raises(ValueError):
        gate._relative(alias)


@pytest.mark.parametrize("kind", ("root", "ancestor", "member", "hardlink", "fifo"))
def test_links_and_fifo_fail_before_any_payload_read(tiny, gate, monkeypatch, tmp_path, kind):
    bundle = tiny.root
    member = tiny.root / "LICENSE.txt"
    if kind == "root":
        bundle = tmp_path / "root-link"
        bundle.symlink_to(tiny.root, target_is_directory=True)
    elif kind == "ancestor":
        link = tmp_path / "parent-link"
        link.symlink_to(tiny.root.parent, target_is_directory=True)
        bundle = link / tiny.root.name
    else:
        member.unlink()
        if kind == "member":
            member.symlink_to(tmp_path / "absent")
        elif kind == "hardlink":
            target = tmp_path / "linked-content"
            target.write_bytes(b"linked\n")
            os.link(target, member)
        else:
            os.mkfifo(member)
    monkeypatch.setattr(gate, "_hash_file", lambda *_a, **_k: pytest.fail("Unsafe member reached reading"))
    with pytest.raises((ValueError, OSError)):
        gate.verify_saved_bundle(bundle, tiny.manifest_sha256)


@pytest.mark.parametrize("role", ("file", "manifest", "aggregate"))
def test_sparse_byte_bounds_fail_before_content_read(tiny, gate, monkeypatch, role):
    if role == "manifest":
        paths = [(tiny.root / "manifest.json", gate.MAX_MANIFEST_BYTES + 1)]
    elif role == "file":
        paths = [(tiny.root / "circuits/access-U.qasm", gate.MAX_FILE_BYTES + 1)]
    else:
        paths = [(tiny.root / name, gate.MAX_FILE_BYTES)
                 for name in ("circuits/access-U.qasm", "circuits/weak-dilation.qasm")]
    for path, size in paths:
        with path.open("r+b") as stream:
            stream.truncate(size)
    monkeypatch.setattr(gate, "_hash_file", lambda *_a, **_k: pytest.fail("Oversized entry reached reading"))
    with pytest.raises(ValueError):
        gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256)


@pytest.mark.parametrize("seconds", (0, -1, 31, 10**1000, float("inf"), float("nan"), True, "30"))
def test_invalid_time_limits_cannot_enlarge_the_reader(seconds, gate):
    with pytest.raises(ValueError):
        gate.verify_saved_bundle(PACKET, MANIFEST_SHA256, max_seconds=seconds)


def test_expired_deadline_precedes_any_packet_content_read(tiny, gate, monkeypatch):
    values = iter((100.0, 131.0))
    monkeypatch.setattr(gate, "time", SimpleNamespace(monotonic=lambda: next(values)))
    monkeypatch.setattr(gate, "_hash_file", lambda *_a, **_k: pytest.fail("Expired gate read content"))
    with pytest.raises(TimeoutError):
        gate.verify_saved_bundle(tiny.root, tiny.manifest_sha256, max_seconds=30)


@pytest.mark.parametrize("raw", (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}',
                                  b'{"x":' + b'9' * 65 + b'}', b'[]', b'\xff'))
def test_duplicate_nonfinite_long_and_nonobject_json_is_rejected(raw, gate):
    with pytest.raises((ValueError, UnicodeError)):
        gate._json(raw)


def test_output_limit_counts_encoded_bytes_before_emission(gate):
    with pytest.raises(ValueError):
        gate._output_bytes({"x": "a" * gate.MAX_OUTPUT_BYTES})
    with pytest.raises(ValueError):
        gate._output_bytes({"x": "\u2603" * (gate.MAX_OUTPUT_BYTES // 2)})
    assert len(gate._output_bytes({"x": "a" * (gate.MAX_OUTPUT_BYTES - 8)})) == gate.MAX_OUTPUT_BYTES


def test_cli_installs_limits_before_only_saved_call_with_fake_guards(gate, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(gate, "_cli_guard", lambda value: calls.append(("guard", value)))
    def saved(*args, **kwargs):
        calls.append(("saved", args, kwargs))
        return {"authenticated": True}
    monkeypatch.setattr(gate, "verify_saved_bundle", saved)
    monkeypatch.setattr(sys, "argv", ["gate", "--bundle", "/synthetic", "--manifest-sha256",
                                     MANIFEST_SHA256, "--max-seconds", "30"])
    assert gate.main() == 0
    assert calls == [("guard", 30.0), ("saved", ("/synthetic", MANIFEST_SHA256), {"max_seconds": 30.0})]
    assert json.loads(capsys.readouterr().out) == {"authenticated": True}


def test_cli_limits_never_enlarge_existing_cpu_or_timer_with_fake_resources(gate, monkeypatch):
    calls = []
    resource = ModuleType("resource")
    resource.RLIMIT_CPU = 0
    resource.RLIM_INFINITY = -1
    resource.getrlimit = lambda _: (12, 15)
    resource.setrlimit = lambda name, pair: calls.append(("cpu", name, pair))
    monkeypatch.setitem(sys.modules, "resource", resource)
    fake_signal = SimpleNamespace(SIGALRM=14, ITIMER_REAL=0,
        signal=lambda *args: calls.append(("handler", args[0])),
        getitimer=lambda _: (2.0, 0.0),
        setitimer=lambda name, value: calls.append(("timer", name, value)))
    monkeypatch.setattr(gate, "signal", fake_signal)
    monkeypatch.setattr(gate, "_ENTRY_TIME", 100.0)
    monkeypatch.setattr(gate, "time", SimpleNamespace(monotonic=lambda: 101.0))
    gate._cli_guard(30.0)
    assert calls == [("cpu", 0, (12, 15)), ("handler", 14), ("timer", 0, 2.0)]
