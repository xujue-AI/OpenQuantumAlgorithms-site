"""Identity/security tests with real saved data; no circuit or Native execution."""
from __future__ import annotations
import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "scripts/dynamics_record_release.py"
gate = types.ModuleType("dynamics_identity_test_gate")
gate.__file__ = str(GATE_PATH)
exec(compile(GATE_PATH.read_bytes(), str(GATE_PATH), "exec"), gate.__dict__)
MODEL = "scripts/dynamics_physical_records.py"
TRANSPORT = "scripts/dynamics_source_transport.py"
SUMMARY = {"status": "validated_frozen_conditional_dynamics_records", "cases": 24,
           "reports": 48, "pareto_points": 489, "csv_rows": 144, "native_estimates_run": False,
           "test_scope": "Structural checker stub, not a new physical validation"}
TRANSPORT_SUMMARY = {"status": "validated_static_source_transport_report",
                     "source_status": "source_not_byte_reproduced",
                     "canonical_circuit_validation_performed": False, "costs_repriced": False,
                     "test_scope": "Structural transport stub, not source or quantum reproduction"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(state):
    path = state["records"] / gate.MANIFEST_NAME
    path.write_text(json.dumps(state["manifest"], indent=2, allow_nan=False) + "\n")
    state["expected"] = digest(path)
    return state["expected"]


@pytest.fixture(scope="module")
def real_bundle_fixture(tmp_path_factory):
    """Real 105-file bundle plus copied sources and a non-scientific checker stub."""
    base = tmp_path_factory.mktemp("dynamics-identity-real-bundle").resolve()
    records, source = base / "records", base / "source"
    records.mkdir(); source.mkdir()
    original_records = ROOT / gate.RECORD_DIRECTORY
    for name in gate.CORE_FILES:
        destination = records / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_records / name, destination)
    for name in set(gate.REPLAY_SOURCE_FILES + gate.FROZEN_FILES):
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    historical = source / gate.HISTORICAL_ADAPTER_PATH
    historical.parent.mkdir(parents=True, exist_ok=True)
    original_historical = ROOT / gate.HISTORICAL_ADAPTER_PATH
    if original_historical.is_file():
        historical.write_bytes(original_historical.read_bytes())
    else:
        historical.write_bytes(subprocess.check_output([
            "git", "show", gate.HISTORICAL_SOURCE_COMMIT + ":src/openquantum/resource_estimation.py"], cwd=ROOT))
    assert digest(historical) == gate.HISTORICAL_ADAPTER_SHA256
    for name in gate.VERIFIER_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise AssertionError('Non-validator or scientific helper must never execute')\n")
    (source / MODEL).write_text("def validate_physical_records(records_dir, study_manifest):\n"
                               + "    return " + repr(SUMMARY) + "\n")
    (source / TRANSPORT).write_text(
        "def normalize_canonical_case(case, native_sources, environment):\n    return case\n"
        "def validate_source_transport(report, cases, qasm, expected_source_hashes):\n"
        "    return " + repr(TRANSPORT_SUMMARY) + "\n")
    evidence = source / gate.SOURCE_REPRODUCTION_EVIDENCE_PATH
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps({"schema_version": 1, "status": "source_not_byte_reproduced",
        "scope": "Structural transport fixture; not new reproduction evidence", "provenance": {},
        "canonical_cases_sha256": "0" * 64, "reconstructed_source_cases_sha256": "0" * 64,
        "summary": {}, "cases": [], "claims": gate.SOURCE_REPORT_CLAIMS}, indent=2) + "\n")
    study = gate.load_json(records / "manifest.json")
    manifest = {
        "schema": gate.MANIFEST_SCHEMA, "status": "frozen_conditional_dynamics_record",
        "files": gate.core_inventory(records), "study_manifest_sha256": digest(records / "manifest.json"),
        "native_provenance": copy.deepcopy(study["provenance"]),
        "native_evidence": copy.deepcopy(gate.NATIVE_EVIDENCE),
        "replay_sources": gate.source_hashes(source, gate.REPLAY_SOURCE_FILES),
        "frozen_files": gate.frozen_hashes(records, source),
        "historical_adapter": {"path": gate.HISTORICAL_ADAPTER_PATH, "sha256": gate.HISTORICAL_ADAPTER_SHA256},
        "verifier_sources": gate.source_hashes(source, gate.VERIFIER_FILES),
        "physical_validation": copy.deepcopy(SUMMARY), "contract": gate.CONTRACT,
        "source_reproduction_evidence": {"path": gate.SOURCE_REPRODUCTION_EVIDENCE_PATH,
                                         "sha256": digest(evidence), "bytes": evidence.stat().st_size},
    }
    state = {"base": base, "records": records, "source": source, "manifest": manifest}
    write_manifest(state)
    return state


@pytest.fixture
def copied_bundle(real_bundle_fixture, tmp_path):
    root = tmp_path.resolve()
    state = {"records": root / "records", "source": root / "source",
             "manifest": copy.deepcopy(real_bundle_fixture["manifest"]),
             "expected": real_bundle_fixture["expected"]}
    shutil.copytree(real_bundle_fixture["records"], state["records"])
    shutil.copytree(real_bundle_fixture["source"], state["source"])
    return state


def arm_checker(state, marker):
    (state["source"] / MODEL).write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n"
        + "def validate_physical_records(records_dir, study_manifest):\n    return " + repr(SUMMARY) + "\n")
    state["manifest"]["verifier_sources"][MODEL] = digest(state["source"] / MODEL)


def arm_transport(state, marker):
    (state["source"] / TRANSPORT).write_text(
        "from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n"
        "def normalize_canonical_case(case, native_sources, environment):\n    return case\n"
        "def validate_source_transport(report, cases, qasm, expected_source_hashes):\n"
        "    return " + repr(TRANSPORT_SUMMARY) + "\n")
    state["manifest"]["verifier_sources"][TRANSPORT] = digest(state["source"] / TRANSPORT)


def verify(state):
    return gate.verify_dynamics_record_identity(state["records"], state["source"],
                                                expected_sha256=state["expected"])


def test_real_bundle_identity_preserves_native_and_current_source_roles(real_bundle_fixture, monkeypatch):
    state = real_bundle_fixture
    before = gate.core_inventory(state["records"])
    original_import = builtins.__import__
    def no_scientific_import(name, *args, **kwargs):
        if name.split(".")[0] in {"openquantum", "qdk", "qiskit", "numpy", "scipy", "matplotlib"}:
            raise AssertionError("Identity gate imported science or an estimator")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_scientific_import)
    result = verify(state)
    assert result["core_files"] == 105 and result["logical_producer_executed"] is False
    assert result["native_reestimation_performed"] is result["factory_search_repeated"] is False
    assert result["external_completion_receipt_available"] is False
    assert result["source_reproduction_validation"] == TRANSPORT_SUMMARY
    assert result["canonical_quantum_validation_performed"] is False
    assert result["source_byte_equivalence_claimed"] is False and result["physical_records_repriced"] is False
    assert gate.core_inventory(state["records"]) == before
    assert len(state["manifest"]["native_provenance"]["source_sha256"]) == 9
    assert len(state["manifest"]["replay_sources"]) == 13
    for name in ("src/openquantum/dynamics_resources.py", "src/openquantum/resource_estimation.py"):
        assert state["manifest"]["native_provenance"]["source_sha256"][name] != state["manifest"]["replay_sources"][name]
    assert not list(state["source"].rglob("__pycache__"))


def test_partial_code_preview_resolves_historical_and_baseline_assets(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle
    site = tmp_path.resolve() / "site"
    shutil.copytree(state["source"], site / "code", ignore=shutil.ignore_patterns("assets"))
    shutil.copytree(state["records"], site / gate.RECORD_DIRECTORY)
    shutil.copytree(state["source"] / "assets", site / "assets", dirs_exist_ok=True)
    monkeypatch.setattr(gate, "DYNAMICS_RECORD_MANIFEST_SHA256", state["expected"])
    assert not (site / "code/assets").exists()
    assert gate.site_dynamics_record_errors(site) == []
    (site / gate.RECORD_DIRECTORY / "summary.md").write_text("Changed")
    assert len(gate.site_dynamics_record_errors(site)) == 1


def test_v2_evidence_bytes_authenticate_before_any_new_checker(copied_bundle, tmp_path):
    state = copied_bundle; transport_marker = tmp_path / "transport-must-not-execute"
    model_marker = tmp_path / "model-must-not-execute"
    arm_transport(state, transport_marker); arm_checker(state, model_marker); write_manifest(state)
    (state["source"] / gate.SOURCE_REPRODUCTION_EVIDENCE_PATH).write_text('{"changed":true}')
    with pytest.raises(ValueError, match="evidence snapshot"): verify(state)
    assert not transport_marker.exists() and not model_marker.exists()


@pytest.mark.parametrize("mutation", ["bool-version", "unknown-version", "old-schema", "status", "claims"])
def test_v2_report_header_rejects_before_transport_loading(copied_bundle, tmp_path, mutation):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_transport(state, marker)
    path = state["source"] / gate.SOURCE_REPRODUCTION_EVIDENCE_PATH
    report = gate.load_json(path)
    if mutation == "bool-version": report["schema_version"] = True
    elif mutation == "unknown-version": report["schema_version"] = 2
    elif mutation == "old-schema": report["schema"] = "unknown"
    elif mutation == "status": report["status"] = "source_byte_reproduced"
    else: report["claims"]["canonical_circuit_validation_performed"] = True
    path.write_text(json.dumps(report) + "\n")
    state["manifest"]["source_reproduction_evidence"].update(sha256=digest(path), bytes=path.stat().st_size)
    write_manifest(state)
    with pytest.raises(ValueError, match="evidence status/schema"): verify(state)
    assert not marker.exists()


def test_v2_transport_model_cannot_mutate_evidence_and_pass(copied_bundle):
    state = copied_bundle
    (state["source"] / TRANSPORT).write_text(
        "from pathlib import Path\n"
        "def normalize_canonical_case(case, native_sources, environment):\n    return case\n"
        "def validate_source_transport(report, cases, qasm, expected_source_hashes):\n"
        "    Path(__file__).parents[1].joinpath(" + repr(gate.SOURCE_REPRODUCTION_EVIDENCE_PATH) + ").write_text('changed')\n"
        "    return " + repr(TRANSPORT_SUMMARY) + "\n")
    state["manifest"]["verifier_sources"][TRANSPORT] = digest(state["source"] / TRANSPORT); write_manifest(state)
    with pytest.raises(ValueError, match="evidence changed"): verify(state)


def test_v2_transport_checker_cannot_claim_quantum_validation(copied_bundle):
    state = copied_bundle
    summary = dict(TRANSPORT_SUMMARY, canonical_circuit_validation_performed=True)
    (state["source"] / TRANSPORT).write_text(
        "def normalize_canonical_case(case, native_sources, environment):\n    return case\n"
        "def validate_source_transport(report, cases, qasm, expected_source_hashes):\n"
        "    return " + repr(summary) + "\n")
    state["manifest"]["verifier_sources"][TRANSPORT] = digest(state["source"] / TRANSPORT); write_manifest(state)
    with pytest.raises(ValueError, match="source transport did not validate"): verify(state)


@pytest.mark.parametrize("name", ["summary.md", "pf2-n3-t1-eps0.05/measured-trial.qasm",
                                  "qsp-oaa3-n3-t1-eps0.05/frozen-p-1e-4.json"])
def test_changed_records_reject_before_authenticated_checker_exec(copied_bundle, tmp_path, name):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_checker(state, marker); write_manifest(state)
    (state["records"] / name).write_text("Changed recorded bytes\n")
    with pytest.raises(ValueError, match="inventory"):
        verify(state)
    assert not marker.exists()


@pytest.mark.parametrize("name", ["src/openquantum/dynamics_resources.py", "requirements-lock.txt",
                                  "scripts/source_provenance.py", "scripts/verify_dynamics_resources.py",
                                  gate.HISTORICAL_ADAPTER_PATH, "scripts/estimate_resources.py"])
def test_changed_sources_and_baselines_reject_before_model(copied_bundle, tmp_path, name):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_checker(state, marker); write_manifest(state)
    (state["source"] / name).write_text("Changed source bytes\n")
    with pytest.raises(ValueError): verify(state)
    assert not marker.exists()


@pytest.mark.parametrize("path,value", [
    (("schema",), "unknown"), (("status",), "pending"), (("contract",), "changed"),
    (("native_evidence", "external_completion_receipt_available"), True),
    (("native_evidence", "external_completion_receipt_available"), 0),
    (("native_evidence", "wall_seconds"), 1.0), (("native_evidence", "peak_rss_kib"), 1000),
    (("native_evidence", "declared_cases"), 24.0), (("native_evidence", "declared_queries"), True),
    (("native_provenance", "source_commit"), "a" * 40),
    (("native_provenance", "environment", "seed"), True),
    (("study_manifest_sha256",), "0" * 64), (("replay_sources",), {}),
    (("verifier_sources",), {}), (("frozen_files",), {}),
    (("historical_adapter", "path"), "../resource_estimation.py"),
    (("historical_adapter", "sha256"), "0" * 64), (("physical_validation",), True),
    (("source_reproduction_evidence", "path"), "../linux-source-reproduction.json"),
    (("source_reproduction_evidence", "sha256"), "0" * 64),
    (("source_reproduction_evidence", "bytes"), True),
])
def test_schema_rejects_fabricated_evidence_or_roles_before_model(copied_bundle, tmp_path, path, value):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_checker(state, marker)
    target = state["manifest"]
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    write_manifest(state)
    with pytest.raises(ValueError): verify(state)
    assert not marker.exists()


@pytest.mark.parametrize("payload", ['{"x":1,"x":1}', '{"x":NaN}', '{"x":1e400}'])
def test_all_record_json_is_strict_before_model_loading(copied_bundle, tmp_path, payload):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_checker(state, marker)
    name = "pf1-n2-t1-eps0.05/frozen-p-1e-3.json"
    path = state["records"] / name; path.write_text(payload)
    state["manifest"]["files"][name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    write_manifest(state)
    with pytest.raises(ValueError, match="JSON"): verify(state)
    assert not marker.exists()


@pytest.mark.parametrize("where", ["record-root", "source-root", "record-file", "source-directory",
                                   "historical-file", "historical-broken-directory", "ancestor"])
def test_lexical_links_reject_before_load(copied_bundle, tmp_path, where):
    state = copied_bundle
    if where in ("record-root", "source-root"):
        key = "records" if where == "record-root" else "source"
        alias = tmp_path / "alias"; alias.symlink_to(state[key], target_is_directory=True); state[key] = alias
    elif where == "record-file":
        path = state["records"] / "summary.md"; saved = tmp_path / "saved"; path.rename(saved); path.symlink_to(saved)
    elif where == "source-directory":
        path = state["source"] / "src"; saved = tmp_path / "saved-src"; path.rename(saved); path.symlink_to(saved, target_is_directory=True)
    elif where == "historical-file":
        path = state["source"] / gate.HISTORICAL_ADAPTER_PATH; saved = tmp_path / "saved"; path.rename(saved); path.symlink_to(saved)
    elif where == "historical-broken-directory":
        path = state["source"] / "assets/generated/dynamics-recorded-source"
        shutil.rmtree(path); path.symlink_to(tmp_path / "missing-dir", target_is_directory=True)
    else:
        alias = tmp_path / "ancestor"; alias.symlink_to(state["source"].parent, target_is_directory=True)
        state["source"] = alias / "source"
    with pytest.raises(ValueError, match="linked"): verify(state)


@pytest.mark.parametrize("name", ["unexplained.png", "new/extra.json", "pf2-n2-t1-eps0.05/extra.txt"])
def test_no_unexplained_record_entries(copied_bundle, name):
    path = copied_bundle["records"] / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("extra")
    with pytest.raises(ValueError, match="unexpected core"): verify(copied_bundle)


def test_bytecode_and_module_cache_are_not_model_authorities(copied_bundle, monkeypatch):
    state = copied_bundle
    cached = types.ModuleType("_dynamics_release_physical_records")
    cached.validate_physical_records = lambda *args: (_ for _ in ()).throw(AssertionError("Cache used"))
    monkeypatch.setitem(sys.modules, cached.__name__, cached)
    directory = state["source"] / "scripts/__pycache__"; directory.mkdir()
    (directory / "dynamics_physical_records.cpython-312.pyc").write_bytes(b"invalid stale bytecode")
    assert verify(state)["core_files"] == 105
    assert len(list(directory.iterdir())) == 1


def test_checked_model_bytes_are_rechecked_after_source_authentication(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    original_checked = gate.checked_file; calls = 0
    def changed_at_load(root, relative):
        nonlocal calls
        path = original_checked(root, relative)
        if Path(root) == state["source"] and relative == MODEL:
            calls += 1
            if calls == 2:
                path.write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").write_text('executed')\n")
        return path
    monkeypatch.setattr(gate, "checked_file", changed_at_load)
    with pytest.raises(ValueError): verify(state)
    assert not marker.exists()


def test_model_cannot_mutate_immutable_records_and_still_pass(copied_bundle):
    state = copied_bundle
    (state["source"] / MODEL).write_text("from pathlib import Path\n"
        "def validate_physical_records(records_dir, study_manifest):\n"
        "    (Path(records_dir)/'summary.md').write_text('mutated')\n"
        "    return " + repr(SUMMARY) + "\n")
    state["manifest"]["verifier_sources"][MODEL] = digest(state["source"] / MODEL); write_manifest(state)
    with pytest.raises(ValueError, match="immutable records changed"): verify(state)


def test_model_source_mutation_is_audited_even_when_checker_raises(copied_bundle):
    state = copied_bundle; before = gate.core_inventory(state["records"])
    (state["source"] / MODEL).write_text("from pathlib import Path\n"
        "def validate_physical_records(records_dir, study_manifest):\n"
        "    Path(__file__).write_text('# changed authenticated checker')\n"
        "    raise RuntimeError('checker failed')\n")
    state["manifest"]["verifier_sources"][MODEL] = digest(state["source"] / MODEL); write_manifest(state)
    with pytest.raises(ValueError, match="authenticated sources/baseline changed"): verify(state)
    assert gate.core_inventory(state["records"]) == before


@pytest.mark.parametrize("kind", ["json", "aggregate"])
def test_json_and_record_aggregate_bounds_reject(copied_bundle, kind):
    name = "pf1-n2-t1-eps0.05/frozen-p-1e-3.json" if kind == "json" else "summary.md"
    limit = gate.MAX_JSON_BYTES if kind == "json" else gate.MAX_AGGREGATE_BYTES
    with (copied_bundle["records"] / name).open("wb") as stream: stream.truncate(limit + 1)
    with pytest.raises(ValueError, match="size limit"): verify(copied_bundle)


@pytest.mark.parametrize("operation", ["read", "json", "hash"])
def test_actual_read_limit_rejects_growth_after_fstat(tmp_path, monkeypatch, operation):
    path = tmp_path / "growing.json"; path.write_bytes(b'{"x":1}')
    real_fstat = gate.os.fstat; inode = path.stat().st_ino; descriptors = []
    def grow_after_stat(descriptor):
        info = real_fstat(descriptor)
        if info.st_ino == inode:
            descriptors.append(descriptor)
            path.write_bytes(b"x" * (gate.MAX_JSON_BYTES + 1))
        return info
    monkeypatch.setattr(gate.os, "fstat", grow_after_stat)
    method = {"read": gate.read_bounded, "json": gate.load_json, "hash": gate.sha256}[operation]
    with pytest.raises(ValueError, match="snapshot size limit"): method(path)
    with pytest.raises(OSError): real_fstat(descriptors[-1])


def test_hash_and_json_parse_use_the_same_open_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "snapshot.json"; original = b'{"x":1}'; replacement = b'{"x":2}'
    path.write_bytes(original); next_path = tmp_path / "replacement.json"; next_path.write_bytes(replacement)
    inode = path.stat().st_ino; real_fstat = gate.os.fstat
    def replace_after_stat(descriptor):
        info = real_fstat(descriptor)
        if info.st_ino == inode and next_path.exists(): os.replace(next_path, path)
        return info
    monkeypatch.setattr(gate.os, "fstat", replace_after_stat)
    expected = hashlib.sha256(original).hexdigest()
    assert gate.load_json(path, expected_sha256=expected) == {"x": 1}
    assert path.read_bytes() == replacement
    with pytest.raises(ValueError, match="snapshot identity"):
        gate.load_json(path, expected_sha256=expected)


def test_oversized_authenticated_model_snapshot_never_executes(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_checker(state, marker); write_manifest(state)
    model_path = state["source"] / MODEL; inode = model_path.stat().st_ino; real_fstat = gate.os.fstat
    def grow_after_stat(descriptor):
        info = real_fstat(descriptor)
        if info.st_ino == inode: model_path.write_bytes(b"#" * (gate.MAX_JSON_BYTES + 1))
        return info
    monkeypatch.setattr(gate.os, "fstat", grow_after_stat)
    with pytest.raises(ValueError, match="snapshot size limit"): verify(state)
    assert not marker.exists()


def test_read_budget_stops_an_input_that_keeps_growing(tmp_path, monkeypatch):
    path = tmp_path / "continually-growing"; path.write_bytes(b"x")
    real_read = gate.os.read; calls = []
    def growing_read(descriptor, count):
        calls.append(count)
        with path.open("ab") as stream: stream.write(b"x" * 8)
        return real_read(descriptor, count)
    monkeypatch.setattr(gate.os, "read", growing_read)
    with pytest.raises(ValueError, match="snapshot size limit"): gate.read_bounded(path, 8)
    assert calls == [9]


def test_fd_inputs_must_be_regular_and_not_follow_links(tmp_path):
    path = tmp_path / "source"; path.write_bytes(b"small")
    link = tmp_path / "link"; link.symlink_to(path)
    if getattr(os, "O_NOFOLLOW", 0):
        with pytest.raises(ValueError, match="no-follow"): gate.read_bounded(link)
    if hasattr(os, "mkfifo"):
        pipe = tmp_path / "fifo"; os.mkfifo(pipe)
        with pytest.raises(ValueError, match="fd must be regular"): gate.read_bounded(pipe)


def _freezer_module():
    path = ROOT / "scripts/freeze_dynamics_resources.py"
    module = types.ModuleType("dynamics_freezer_test_module")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.gate = gate
    return module


def test_v2_synthetic_freezer_binds_only_authenticated_static_evidence(copied_bundle, tmp_path):
    state = copied_bundle; before = gate.core_inventory(state["records"])
    output = tmp_path / "candidate-sidecar.json"
    result = _freezer_module().freeze_record(state["records"], output, source_root=state["source"])
    candidate = gate.load_json(output)
    assert candidate["schema"] == gate.MANIFEST_SCHEMA
    assert candidate["source_reproduction_evidence"] == state["manifest"]["source_reproduction_evidence"]
    assert candidate["physical_validation"] == SUMMARY and candidate["files"] == before
    assert result["source_reproduction_validation"] == TRANSPORT_SUMMARY
    assert result["canonical_quantum_validation_performed"] is False
    assert result["source_byte_equivalence_claimed"] is False and result["physical_records_repriced"] is False
    assert gate.core_inventory(state["records"]) == before


def test_v2_freezer_rejects_bad_evidence_before_any_checker(copied_bundle, tmp_path):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_transport(state, marker); write_manifest(state)
    path = state["source"] / gate.SOURCE_REPRODUCTION_EVIDENCE_PATH
    report = gate.load_json(path); report["claims"]["costs_repriced"] = True
    path.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="evidence status/schema"):
        _freezer_module().freeze_record(state["records"], tmp_path / "candidate.json", source_root=state["source"])
    assert not marker.exists() and not (tmp_path / "candidate.json").exists()


def test_v2_freezer_rejects_changed_records_before_checker_loading(copied_bundle, tmp_path):
    state = copied_bundle; marker = tmp_path / "must-not-execute"
    arm_transport(state, marker); write_manifest(state)
    path = state["records"] / "pf1-n2-t1-eps0.05/frozen-p-1e-3.json"
    report = gate.load_json(path); report["qasm_sha256"] = "0" * 64
    path.write_text(json.dumps(report) + "\n")
    with pytest.raises(ValueError, match="QASM binding"):
        _freezer_module().freeze_record(state["records"], tmp_path / "candidate.json", source_root=state["source"])
    assert not marker.exists()


def _track_output_descriptors(freezer, output, monkeypatch):
    original_open = freezer.os.open; tracked = {"parent": [], "target": []}
    def track_open(path, flags, *args, **kwargs):
        descriptor = original_open(path, flags, *args, **kwargs)
        if Path(path) == output.parent: tracked["parent"].append(descriptor)
        if kwargs.get("dir_fd") in tracked["parent"] and str(path) == output.name:
            tracked["target"].append(descriptor)
        return descriptor
    monkeypatch.setattr(freezer.os, "open", track_open)
    return tracked


def _assert_output_descriptors_closed(tracked):
    assert tracked["parent"]
    for descriptor in tracked["parent"] + tracked["target"]:
        with pytest.raises(OSError): os.fstat(descriptor)


@pytest.mark.parametrize("where", ["parent", "ancestor"])
def test_freezer_held_parent_rejects_redirection_during_auth(copied_bundle, tmp_path, monkeypatch, where):
    state = copied_bundle; freezer = _freezer_module()
    ancestor = tmp_path / "output-ancestor"; parent = ancestor / "output-parent"; parent.mkdir(parents=True)
    redirect = tmp_path / "redirect"; (redirect / "output-parent").mkdir(parents=True)
    output = parent / "manifest.json"; backup = tmp_path / "held-original"
    tracked = _track_output_descriptors(freezer, output, monkeypatch)
    original_validation = freezer.gate.validate_source_reproduction
    def redirect_after_validation(inputs):
        result = original_validation(inputs)
        if where == "parent":
            parent.rename(backup); parent.symlink_to(redirect, target_is_directory=True)
        else:
            ancestor.rename(backup); ancestor.symlink_to(redirect, target_is_directory=True)
        return result
    monkeypatch.setattr(freezer.gate, "validate_source_reproduction", redirect_after_validation)
    with pytest.raises(ValueError, match="linked|anchoring"):
        freezer.freeze_record(state["records"], output, source_root=state["source"])
    assert not list(redirect.rglob("manifest.json")) and not list(backup.rglob("manifest.json"))
    _assert_output_descriptors_closed(tracked)


def test_freezer_dirfd_create_cannot_follow_parent_swap_after_check(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle; freezer = _freezer_module()
    parent = tmp_path / "output-parent"; parent.mkdir()
    redirect = tmp_path / "redirect"; redirect.mkdir(); backup = tmp_path / "held-original"
    output = parent / "manifest.json"; tracked = _track_output_descriptors(freezer, output, monkeypatch)
    tracked_open = freezer.os.open
    def swap_immediately_before_open(path, flags, *args, **kwargs):
        if kwargs.get("dir_fd") in tracked["parent"] and str(path) == output.name:
            parent.rename(backup); parent.symlink_to(redirect, target_is_directory=True)
        return tracked_open(path, flags, *args, **kwargs)
    monkeypatch.setattr(freezer.os, "open", swap_immediately_before_open)
    with pytest.raises(ValueError, match="linked|anchoring"):
        freezer.freeze_record(state["records"], output, source_root=state["source"])
    assert tracked["target"] and not (redirect / output.name).exists()
    assert (backup / output.name).is_file()
    _assert_output_descriptors_closed(tracked)


@pytest.mark.parametrize("failure", ["write", "authentication", "post_audit"])
def test_freezer_closes_owned_fds_and_retains_failed_candidate(copied_bundle, tmp_path, monkeypatch, failure):
    state = copied_bundle; freezer = _freezer_module(); output = tmp_path / "manifest-output.json"
    tracked = _track_output_descriptors(freezer, output, monkeypatch)
    if failure == "write":
        original_write = freezer.os.write
        def failed_write(descriptor, payload):
            if descriptor in tracked["target"]: raise OSError("synthetic output write failure")
            return original_write(descriptor, payload)
        monkeypatch.setattr(freezer.os, "write", failed_write)
        expected = OSError
    elif failure == "authentication":
        path = state["source"] / gate.SOURCE_REPRODUCTION_EVIDENCE_PATH
        report = gate.load_json(path); report["claims"]["costs_repriced"] = True; path.write_text(json.dumps(report))
        expected = ValueError
    else:
        original_recheck = freezer.gate.recheck_authenticated_inputs
        def fail_post_audit(inputs):
            original_recheck(inputs)
            if output.exists():
                assert tracked["target"] and os.fstat(tracked["target"][-1])
                raise ValueError("synthetic post-write input audit failure")
        monkeypatch.setattr(freezer.gate, "recheck_authenticated_inputs", fail_post_audit)
        expected = ValueError
    with pytest.raises(expected): freezer.freeze_record(state["records"], output, source_root=state["source"])
    if failure == "authentication":
        assert not output.exists() and not tracked["target"]
    else:
        assert output.is_file() and tracked["target"]
        if failure == "write": assert output.stat().st_size == 0
        with pytest.raises(ValueError, match="already exists"):
            freezer.freeze_record(state["records"], output, source_root=state["source"])
    _assert_output_descriptors_closed(tracked)


@pytest.mark.parametrize("mutation", ["overwrite", "replace", "symlink", "remove"])
def test_freezer_final_audit_detects_changed_output_and_preserves_replacements(copied_bundle, tmp_path, monkeypatch, mutation):
    state = copied_bundle; freezer = _freezer_module(); output = tmp_path / "manifest-output.json"
    tracked = _track_output_descriptors(freezer, output, monkeypatch)
    replacement = tmp_path / "replacement.json"; replacement.write_bytes(b"unrelated replacement")
    original_recheck = freezer.gate.recheck_authenticated_inputs
    changed = []
    def change_during_post_audit(inputs):
        original_recheck(inputs)
        if output.exists() and not changed:
            assert tracked["target"] and os.fstat(tracked["target"][-1])
            changed.append(True)
            if mutation == "overwrite":
                payload = output.read_bytes(); output.write_bytes(b"X" + payload[1:])
            else:
                output.unlink()
                if mutation == "replace": output.write_bytes(replacement.read_bytes())
                elif mutation == "symlink": output.symlink_to(replacement)
    monkeypatch.setattr(freezer.gate, "recheck_authenticated_inputs", change_during_post_audit)
    expected = FileNotFoundError if mutation == "remove" else ValueError
    with pytest.raises(expected):
        freezer.freeze_record(state["records"], output, source_root=state["source"])
    assert changed and replacement.read_bytes() == b"unrelated replacement"
    if mutation == "overwrite": assert output.read_bytes().startswith(b"X")
    elif mutation == "replace": assert output.read_bytes() == b"unrelated replacement"
    elif mutation == "symlink": assert output.is_symlink() and output.read_bytes() == b"unrelated replacement"
    else: assert not output.exists()
    _assert_output_descriptors_closed(tracked)


def test_freezer_failed_audit_never_unlinks_a_concurrent_replacement(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle; freezer = _freezer_module(); output = tmp_path / "manifest-output.json"
    tracked = _track_output_descriptors(freezer, output, monkeypatch)
    moved = tmp_path / "original-candidate.json"; changed = []
    original_recheck = freezer.gate.recheck_authenticated_inputs
    original_unlink = freezer.os.unlink
    def replace_then_fail(inputs):
        original_recheck(inputs)
        if output.exists() and not changed:
            changed.append(True)
            output.rename(moved); output.write_bytes(b"unrelated replacement")
            raise ValueError("synthetic post-write input audit failure")
    def prohibit_named_rollback(path, *args, **kwargs):
        if str(path) == output.name and kwargs.get("dir_fd") in tracked["parent"]:
            raise AssertionError("basename cleanup could unlink a concurrent replacement")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(freezer.gate, "recheck_authenticated_inputs", replace_then_fail)
    monkeypatch.setattr(freezer.os, "unlink", prohibit_named_rollback)
    with pytest.raises(ValueError, match="post-write input audit"):
        freezer.freeze_record(state["records"], output, source_root=state["source"])
    assert changed and moved.is_file() and output.read_bytes() == b"unrelated replacement"
    _assert_output_descriptors_closed(tracked)


def test_freezer_success_closes_parent_and_output_descriptors(copied_bundle, tmp_path, monkeypatch):
    state = copied_bundle; freezer = _freezer_module(); output = tmp_path / "manifest-output.json"
    tracked = _track_output_descriptors(freezer, output, monkeypatch)
    result = freezer.freeze_record(state["records"], output, source_root=state["source"])
    assert result["manifest_sha256"] == digest(output)
    assert tracked["target"]
    _assert_output_descriptors_closed(tracked)


def test_json_child_budget_rejects_before_extending_pending_nodes(tmp_path, monkeypatch):
    path = tmp_path / "too-many-children.json"; path.write_text(json.dumps({"items": [0] * 9}))
    monkeypatch.setattr(gate, "MAX_JSON_NODES", 8)
    with pytest.raises(ValueError, match="traversal limit"): gate.load_json(path)


def test_real_coordinator_manifest_and_model_read_only_when_available():
    records = ROOT / gate.RECORD_DIRECTORY; manifest = records / gate.MANIFEST_NAME
    if not manifest.is_file(): pytest.skip("Coordinator has not frozen the immutable dynamics sidecar yet")
    result = gate.verify_dynamics_record_identity(records, ROOT, expected_sha256=digest(manifest))
    assert result["core_files"] == 105 and result["native_reestimation_performed"] is False
