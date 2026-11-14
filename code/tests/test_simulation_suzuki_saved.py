"""Saved/mocked Suzuki publication checks; no scientific imports or execution."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_simulation_suzuki.py"
BUNDLE = ROOT / "assets/generated/simulation/suzuki"
SOURCES = ("src/openquantum/simulation/suzuki.py", "scripts/generate_simulation_suzuki.py", "requirements-lock.txt", "LICENSE")


def load_gate():
    spec = importlib.util.spec_from_file_location("saved_suzuki_gate_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    exec(compile(SCRIPT.read_bytes(), str(SCRIPT), "exec"), module.__dict__)
    return module


@pytest.fixture
def gate():
    return load_gate()


@pytest.fixture(scope="module", autouse=True)
def preserve_source_and_canonical_bundle():
    paths = [SCRIPT, *(ROOT / name for name in SOURCES), *sorted(path for path in BUNDLE.rglob("*") if path.is_file())]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    names = {path.relative_to(BUNDLE).as_posix() for path in paths if path.is_relative_to(BUNDLE)}
    yield
    assert {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*") if path.is_file()} == names
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths} == before


@pytest.fixture
def copied(tmp_path):
    path = tmp_path.resolve() / "saved"
    shutil.copytree(BUNDLE, path)
    return path


def sources_at(destination):
    for name in SOURCES:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return destination


def rehash(bundle):
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"] = {file.relative_to(bundle).as_posix(): {"bytes": file.stat().st_size, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
                         for file in sorted(bundle.rglob("*")) if file.is_file() and file != path}
    manifest["payload_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def change_record(bundle, update, case="n2-s2-r1"):
    path = bundle / case / "record.json"
    record = json.loads(path.read_text()); update(record)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    rehash(bundle)


@pytest.fixture
def replay(copied, tmp_path, gate, monkeypatch):
    source = sources_at(tmp_path.resolve() / "download")
    path = copied / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_commit"] = None
    manifest["generation_wall_seconds_after_imports"] += 1
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    monkeypatch.setattr(gate, "_git_bytes", lambda *args: (128, b""))
    return copied, source


def test_import_is_standard_library_only_and_readless(monkeypatch):
    compiled = compile(SCRIPT.read_bytes(), str(SCRIPT), "exec")
    spec = importlib.util.spec_from_file_location("readless_suzuki_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] in sys.stdlib_module_names | {"__future__"}
        return original_import(name, *args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Saved gate import must not read files or launch a process")
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    monkeypatch.setattr(os, "fdopen", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    exec(compiled, module.__dict__)
    assert len(module.BUNDLE_FILES) == 84 and len(module.CASE_IDS) == 36


def test_real_canonical_saved_check_has_no_scientific_import_or_git(gate, monkeypatch):
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"openquantum", "qdk", "qiskit", "numpy", "scipy", "matplotlib"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(gate, "_git_bytes", lambda *args: pytest.fail("Canonical saved check must not call Git"))
    report = gate.verify_bundle(BUNDLE, source_root=ROOT)
    assert report == {"status": "verified_saved_suzuki_study", "manifest_sha256": gate.MANIFEST_SHA256, "cases": 36,
                      "files": 84, "bytes": 1695183, "recorded_numerical_checks": True, "new_simulations": 0,
                      "native_estimator_calls": 0, "physical_resource_estimate": None, "formal_certificate": False,
                      "current_source_files_checked": 4}


def test_detached_inspection_discloses_no_current_source_checks(gate):
    report = gate.verify_bundle(BUNDLE)
    assert report["current_source_files_checked"] == 0
    assert report["new_simulations"] == 0


def test_real_authenticated_package_loader_works_without_import_registration(monkeypatch):
    path = ROOT / "scripts/verify_public_package.py"
    spec = importlib.util.spec_from_file_location("suzuki_public_loader_test", path)
    loader = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), loader.__dict__)
    provenance = {"schema": "openquantum-public-package-v2", "file_sha256": {
        "scripts/verify_simulation_suzuki.py": hashlib.sha256(SCRIPT.read_bytes()).hexdigest()}}
    original_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"openquantum", "qdk", "qiskit", "numpy", "scipy", "matplotlib"}
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    gate = loader._load_suzuki_record_gate(ROOT, provenance)
    assert gate.__name__ not in sys.modules
    assert gate.verify_bundle(BUNDLE, source_root=ROOT)["current_source_files_checked"] == 4
    assert gate.__name__ not in sys.modules
    provenance["file_sha256"]["scripts/verify_simulation_suzuki.py"] = "0" * 64
    monkeypatch.setattr(builtins, "compile", lambda *args: pytest.fail("Unauthenticated gate compiled"))
    with pytest.raises(ValueError, match="complete manifest"):
        loader._load_suzuki_record_gate(ROOT, provenance)


def test_manifest_pin_rejected_before_json_or_record_use(gate, copied, monkeypatch):
    path = copied / "manifest.json"; path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(gate, "_json", lambda *args: pytest.fail("Unpinned manifest must not be parsed"))
    with pytest.raises(ValueError, match="manifest pin"):
        gate.verify_bundle(copied, source_root=ROOT)


@pytest.mark.parametrize("mutation", ["tampered", "missing", "extra", "empty_directory", "symlink", "hardlink"])
def test_full_payload_authentication_precedes_record_use(gate, copied, tmp_path, monkeypatch, mutation):
    target = copied / "n2-s2-r1/record.json"
    if mutation == "tampered":
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "missing":
        target.unlink()
    elif mutation == "extra":
        (copied / "extra.txt").write_text("unlisted")
    elif mutation == "empty_directory":
        (copied / "extra").mkdir()
    else:
        outside = tmp_path.resolve() / "outside.json"; shutil.copyfile(target, outside); target.unlink()
        target.symlink_to(outside) if mutation == "symlink" else target.hardlink_to(outside)
    monkeypatch.setattr(gate, "_records", lambda *args: pytest.fail("Incomplete payload was used"))
    with pytest.raises((ValueError, OSError)):
        gate.verify_bundle(copied, source_root=ROOT)


@pytest.mark.parametrize("name", SOURCES)
def test_current_four_sources_are_exact(gate, tmp_path, name):
    source = sources_at(tmp_path.resolve() / "download")
    path = source / name; path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="current source"):
        gate.verify_bundle(BUNDLE, source_root=source)


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}',
                                b'[' * 18 + b'0' + b']' * 18, b'"' + b'x' * 8193 + b'"',
                                str(2 ** 4097).encode(), b'\xff'])
def test_bounded_json_rejects_duplicates_nonfinite_and_structure(gate, raw):
    with pytest.raises(ValueError):
        gate._json(raw)


@pytest.mark.parametrize("value", ["1e999999999", "1.5", "9" * 1235, "1/" + "9" * 1235,
                                  "+1", "01", "1/-2", "1/0"])
def test_rational_syntax_and_digit_limits_reject_before_conversion(gate, monkeypatch, value):
    monkeypatch.setattr(gate, "Fraction", lambda *args: pytest.fail("Unsafe rational reached integer conversion"))
    with pytest.raises(ValueError, match="canonical Suzuki rational"):
        gate._fraction(value)


@pytest.mark.parametrize("value", ["2/4", "-0", str(2 ** 4096)])
def test_rational_records_require_reduced_spelling_and_4096_bit_limit(gate, value):
    with pytest.raises(ValueError):
        gate._fraction(value)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a\\b", "a//b", "./source"])
def test_file_reader_rejects_relative_aliases(gate, tmp_path, name):
    with pytest.raises(ValueError):
        gate._read(tmp_path.resolve(), name)


def test_file_reader_rejects_root_ancestor_hardlink_and_size(gate, tmp_path):
    root = tmp_path.resolve() / "root"; root.mkdir()
    target = root / "source"; target.write_bytes(b"four")
    with pytest.raises(ValueError, match="oversized"):
        gate._read(root, "source", 3)
    alias = tmp_path.resolve() / "alias"; alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        gate._read(alias, "source")
    (root / "linked").symlink_to(root, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        gate._read(root, "linked/source")
    (root / "hard").hardlink_to(target)
    with pytest.raises(ValueError):
        gate._read(root, "source")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO regression")
@pytest.mark.parametrize("race", [False, True])
def test_fifo_rejection_and_preopen_replacement_are_bounded(tmp_path, race):
    root = tmp_path.resolve() / "root"; root.mkdir()
    path = root / "source"; path.write_bytes(b"safe")
    if not race:
        path.unlink(); os.mkfifo(path)
    command = r'''
import sys,os,importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location("fifo_suzuki",sys.argv[1]);m=importlib.util.module_from_spec(s)
exec(compile(Path(sys.argv[1]).read_bytes(),sys.argv[1],"exec"),m.__dict__)
root=Path(sys.argv[2]);race=sys.argv[3]=="True";original=m.os.open;changed=False
def opened(path,flags,*args,**kwargs):
    global changed
    if race and not changed and path=="source" and "dir_fd" in kwargs:
        changed=True;(root/"source").unlink();os.mkfifo(root/"source")
    return original(path,flags,*args,**kwargs)
m.os.open=opened
try:m._read(root,"source",100)
except (ValueError,OSError):print("bounded");sys.exit(0)
raise AssertionError("FIFO accepted")
'''
    result = subprocess.run([sys.executable, "-c", command, str(SCRIPT), str(root), str(race)],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded"


def test_exact_saved_replay_allows_only_authenticated_volatile_values(gate, replay):
    bundle, source = replay
    (bundle / "n2-order4-circuit.svg").write_bytes((bundle / "n2-order4-circuit.svg").read_bytes() + b"\n<!-- fresh rendering -->\n")
    def update(record):
        record["validation"]["state_infidelity"] += 5e-12
        record["cpu_timings_seconds"]["source_construction_scalar_accounting_export"] += 1
    change_record(bundle, update)
    result = gate.compare_replay(BUNDLE, bundle, source_root=source)
    assert result["status"] == "verified_saved_suzuki_replay_comparison"
    assert result["absolute_diagnostic_tolerance"] == 1e-11
    assert 0 < result["maximum_absolute_diagnostic_difference"] <= 1e-11
    assert result["new_simulations"] == 0


def test_diagnostic_outside_explicit_tolerance_is_rejected(gate, replay):
    bundle, source = replay
    change_record(bundle, lambda record: record["validation"].__setitem__("state_infidelity", record["validation"]["state_infidelity"] + 2e-11))
    with pytest.raises(ValueError, match="absolute tolerance"):
        gate.compare_replay(BUNDLE, bundle, source_root=source)


@pytest.mark.parametrize("mutation", ["counts", "scalar", "physical", "formal", "shots", "csv", "markdown", "qasm", "source_snapshot", "manifest_scope"])
def test_self_rehashed_replay_cannot_change_exact_contract(gate, replay, mutation):
    bundle, source = replay
    if mutation == "counts":
        change_record(bundle, lambda record: record["logical_ledger"]["complete_measured_primitive_counts"].__setitem__("rx", 99))
    elif mutation == "scalar":
        change_record(bundle, lambda record: record["logical_ledger"]["source_parameter_rounding"].__setitem__("total_source_parameter_operator_bound_exact", "0"))
    elif mutation == "physical":
        change_record(bundle, lambda record: record["logical_ledger"].__setitem__("physical_resource_estimate", {}))
    elif mutation == "formal":
        change_record(bundle, lambda record: record["validation"].__setitem__("formal_certificate", True))
    elif mutation == "shots":
        change_record(bundle, lambda record: record["measurement_execution"].__setitem__("finite_shot_samples", True))
    elif mutation in ("csv", "markdown", "source_snapshot"):
        name = {"csv": "summary.csv", "markdown": "summary.md", "source_snapshot": "suzuki.py"}[mutation]
        path = bundle / name; path.write_bytes(path.read_bytes() + b"changed"); rehash(bundle)
    elif mutation == "qasm":
        path = bundle / "n2-s2-r1/measured-trial.qasm"
        original = path.read_bytes(); changed = original.replace(b"rz(1.0)", b"rz(-1.0)")
        assert changed != original
        path.write_bytes(changed)
        change_record(bundle, lambda record: record.__setitem__("measured_qasm_sha256", hashlib.sha256(path.read_bytes()).hexdigest()))
    else:
        path = bundle / "manifest.json"; manifest = json.loads(path.read_text()); manifest["scope"] += "changed"
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError):
        gate.compare_replay(BUNDLE, bundle, source_root=source)


def test_both_complete_inventories_authenticate_before_record_checks(gate, replay, monkeypatch):
    bundle, source = replay; seen = {BUNDLE: set(), bundle: set()}
    original_read = gate._read; original_records = gate._records
    def read(root, name, *args):
        if Path(root) in seen:
            seen[Path(root)].add(name)
        return original_read(root, name, *args)
    def records(*args):
        assert all(names == gate.BUNDLE_FILES for names in seen.values())
        return original_records(*args)
    monkeypatch.setattr(gate, "_read", read); monkeypatch.setattr(gate, "_records", records)
    assert gate.compare_replay(BUNDLE, bundle, source_root=source)["cases"] == 36


def test_replay_payload_digest_is_checked_before_any_record_use(gate, replay, monkeypatch):
    bundle, source = replay
    path = bundle / "n3-s6-r8/record.json"; path.write_bytes(path.read_bytes() + b" ")
    monkeypatch.setattr(gate, "_records", lambda *args: pytest.fail("Unauthenticated replay was used"))
    with pytest.raises(ValueError):
        gate.compare_replay(BUNDLE, bundle, source_root=source)


def test_cli_dispatches_only_saved_verification_and_explicit_replay(gate, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(gate, "verify_bundle", lambda bundle, **kwargs: calls.append((bundle, kwargs)) or {"saved": True})
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])
    gate.main()
    assert calls == [(BUNDLE, {"source_root": ROOT})]
    assert json.loads(capsys.readouterr().out) == {"saved": True}
    monkeypatch.setattr(gate, "compare_replay", lambda canonical, replay, **kwargs:
                        calls.append((canonical, replay, kwargs)) or {"replay": True})
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--records-dir", str(BUNDLE), "--replay-dir", "/private/tmp/fresh", "--atol", "2e-12"])
    gate.main()
    assert calls[-1] == (BUNDLE, Path("/private/tmp/fresh"), {"source_root": ROOT, "atol": 2e-12})
    assert json.loads(capsys.readouterr().out) == {"replay": True}


def test_replay_git_commit_authenticates_exact_root_and_all_four_blobs(gate, replay, monkeypatch):
    bundle, source = replay; commit = "b" * 40; calls = []
    path = bundle / "manifest.json"; manifest = json.loads(path.read_text()); manifest["source_commit"] = commit
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    def git(root, arguments, maximum):
        assert root == source; calls.append(arguments)
        if arguments == ["rev-parse", "--show-toplevel"]:
            return 0, (str(source) + "\n").encode()
        if arguments == ["rev-parse", "HEAD"]:
            return 0, (commit + "\n").encode()
        name = arguments[1].split(":", 1)[1]
        return 0, (source / name).read_bytes()
    monkeypatch.setattr(gate, "_git_bytes", git)
    assert gate.compare_replay(BUNDLE, bundle, source_root=source)["cases"] == 36
    assert calls[2:] == [["show", f"{commit}:{name}"] for name in SOURCES]
    manifest["source_commit"] = "c" * 40; path.write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="fresh replay source commit"):
        gate.compare_replay(BUNDLE, bundle, source_root=source)


def fake_git_process(monkeypatch, gate, code):
    """Run only a bounded standard-library child in place of the Git command."""
    original = subprocess.Popen; processes = []
    def start(command, **kwargs):
        assert command == ["git", "rev-parse", "HEAD"]
        process = original([sys.executable, "-c", code], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(gate.subprocess, "Popen", start)
    return processes


def test_real_nonblocking_git_reader_accepts_bounded_stdout_and_closes_pipe(gate, tmp_path, monkeypatch):
    processes = fake_git_process(monkeypatch, gate, "import os;os.write(1,b'commit\\n')")
    assert gate._git_bytes(tmp_path.resolve(), ["rev-parse", "HEAD"], 128) == (0, b"commit\n")
    assert processes[0].poll() == 0 and processes[0].stdout.closed


def test_real_nonblocking_git_reader_bounds_stdout_and_kills_child(gate, tmp_path, monkeypatch):
    processes = fake_git_process(monkeypatch, gate, "import os,time;os.write(1,b'x'*4096);time.sleep(10)")
    with pytest.raises(ValueError, match="output limit"):
        gate._git_bytes(tmp_path.resolve(), ["rev-parse", "HEAD"], 10)
    assert processes[0].poll() is not None and processes[0].stdout.closed


def test_git_reader_timeout_kills_child_and_closes_pipe(gate, tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        processes = fake_git_process(patch, gate, "import time;time.sleep(10)")
        ticks = iter((0.0, 6.0))
        patch.setattr(gate.time, "monotonic", lambda: next(ticks))
        with pytest.raises(ValueError, match="identity timeout"):
            gate._git_bytes(tmp_path.resolve(), ["rev-parse", "HEAD"], 128)
    assert processes[0].poll() is not None and processes[0].stdout.closed


def test_missing_git_does_not_invent_checkout_identity(gate, tmp_path, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("No Git")
    monkeypatch.setattr(gate.subprocess, "Popen", missing)
    assert gate._git_bytes(tmp_path.resolve(), ["rev-parse", "HEAD"], 128) == (None, b"")


@pytest.mark.parametrize("atol", [True, -1, float("nan"), float("inf")])
def test_replay_tolerance_must_be_explicit_finite_nonnegative(gate, replay, atol):
    bundle, source = replay
    with pytest.raises(ValueError):
        gate.compare_replay(BUNDLE, bundle, source_root=source, atol=atol)
