"""Saved-byte tests: no quantum imports, sampling, simulation or generation."""
from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qdrift_saved_gate", ROOT / "scripts/verify_simulation_qdrift.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
CANONICAL = ROOT / gate.BUNDLE_RELATIVE
REPLAY_COMMIT = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def replay(tmp_path):
    destination = tmp_path / "replay"
    shutil.copytree(CANONICAL, destination)
    value = json.loads((destination / "manifest.json").read_text())
    value["source_commit"] = REPLAY_COMMIT
    value["scientific_wall_seconds"] = 1.25
    (destination / "manifest.json").write_text(json.dumps(value, indent=2) + "\n")
    return destination


def refresh(replay):
    manifest = json.loads((replay / "manifest.json").read_text())
    for name in manifest["files"]:
        data = (replay / name).read_bytes()
        manifest["files"][name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest["payload_bytes"] = sum(row["bytes"] for row in manifest["files"].values())
    (replay / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")


def change(replay, path, value, *, file="one-spin/study.json"):
    data = json.loads((replay / file).read_text())
    current = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value
    (replay / file).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    refresh(replay)


def compare(replay, **kwargs):
    return gate.compare_replay(CANONICAL, replay, source_root=ROOT, expected_source_commit=REPLAY_COMMIT, **kwargs)


def test_canonical_identity_and_independent_saved_arithmetic():
    result = gate.verify_bundle(CANONICAL, source_root=ROOT)
    assert result["manifest_sha256"] == gate.CANONICAL_MANIFEST_SHA256
    assert result["source_commit"] == gate.CANONICAL_PRODUCER
    assert result["files"] == 78 and result["total_bytes"] == 677245
    assert result["trials"] == 64 and result["term_exponentials"] == 512
    assert result["primitive_counts"] == {"reset": 96, "rz": 512, "h": 662, "measure": 96, "cx": 198}
    assert "not rerun" in result["verification_scope"]


def test_no_quantum_implementation_imports(monkeypatch):
    original = builtins.__import__
    def restricted(name, *args, **kwargs):
        assert not name.startswith(("openquantum", "qiskit", "numpy", "scipy", "matplotlib"))
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", restricted)
    assert gate.verify_bundle(CANONICAL, source_root=ROOT)["trials"] == 64


def test_saved_equivalent_replay_has_separate_producer(replay):
    result = compare(replay)
    assert result["status"] == "verified_fresh_qdrift_replay"
    assert result["source_commit"] == REPLAY_COMMIT
    assert result["canonical_manifest_sha256"] == gate.CANONICAL_MANIFEST_SHA256
    assert result["new_scientific_execution_by_verifier"] is False


def test_svg_bytes_are_authenticated_per_run_not_compared(replay):
    path = replay / "one-spin/channel-and-samples.svg"
    path.write_bytes(path.read_bytes() + b"\n")
    refresh(replay)
    assert compare(replay)["total_bytes"] == sum(p.stat().st_size for p in replay.rglob("*") if p.is_file())


def test_named_diagnostic_tolerance_only(replay):
    path = ["records", 0, "source_to_independent_operator_error"]
    change(replay, path, 1e-12)
    assert compare(replay)["diagnostic_atol"] == 1e-11
    with pytest.raises(ValueError, match="numerical mismatch"):
        compare(replay, atol=0.0)


@pytest.mark.parametrize("value", [True, 1, -1e-11, 2e-11, float("nan"), float("inf")])
def test_tolerance_cannot_be_weakened(replay, value):
    with pytest.raises(ValueError, match="tolerance"):
        compare(replay, atol=value)


@pytest.mark.parametrize("value", [None, "", "1" * 39, "g" * 40, True])
def test_exact_producer_commit_required(replay, value):
    with pytest.raises(ValueError, match="exact source commit"):
        gate.compare_replay(CANONICAL, replay, source_root=ROOT, expected_source_commit=value)


@pytest.mark.parametrize("path,value", [
    (["steps"], 16), (["steps"], 8.0), (["trajectories"], True),
    (["initial_basis"], 1), (["master_seed"], 20261114),
    (["sample_kind"], "QPU measurement"), (["randomness_scope"], "ideal independence proven"),
    (["physical_resource_estimate"], {}),
    (["validation", "performed"], 1), (["validation", "formal_certificate"], True),
    (["validation", "full_channel_spanning_tests"], 16),
    (["records", 0, "trajectory_seed"], True), (["records", 0, "readout_seed"], 0.0),
    (["records", 0, "schedule_indices", 0], True), (["records", 0, "schedule_indices", 0], 0),
    (["records", 0, "classically_sampled_basis_outcome"], True),
    (["records", 0, "classically_sampled_basis_outcome"], 2),
    (["records", 0, "readout_bitstring_qiskit_order"], "1"),
    (["records", 0, "classically_sampled_z0"], -1),
    (["records", 0, "source_to_independent_operator_error"], -1e-15),
    (["records", 0, "source_to_independent_operator_error"], 2e-11),
    (["records", 0, "state_probability_norm_before_normalization"], .99),
    (["records", 0, "circuit_z0_expectation"], 2.0),
    (["records", 0, "logical_ledger", "arbitrary_rotations"], 7),
    (["logical_ledgers", 0, "complete_primitive_counts", "reset"], True),
    (["logical_ledgers", 0, "source_dependency_depth"], 25),
    (["logical_ledgers", 0, "magnitude_rz_angle_hex"], "0x1.0000000000001p-3"),
    (["logical_ledgers", 0, "per_exponential_scalar_angle_conversion_operator_bound_exact"], "1/10"),
    (["logical_ledgers", 0, "global_phase_radians"], -0.0),
    (["logical_ledgers", 0, "excluded_costs"], []),
    (["complete_workload", "fresh_preparations"], 31),
    (["complete_workload", "term_draws"], 256.0),
    (["sample_mean"], .5), (["mean_of_fixed_trajectory_expectations"], .5),
    (["ideal_iid_hoeffding", "absolute_radius"], .05),
    (["ensemble_reference", "observable_bias"], .05),
    (["ensemble_reference", "channel_diagnostics", "normalized_choi_trace"], .99),
    (["ensemble_reference", "channel_diagnostics", "normalized_choi_min_eigenvalue"], -2e-11),
    (["ensemble_reference", "channel_diagnostics", "trace_preservation_max_error"], 2e-11),
])
def test_authenticated_replay_semantic_mutations_rejected(replay, path, value):
    change(replay, path, value)
    with pytest.raises(ValueError):
        compare(replay)


@pytest.mark.parametrize("path,value", [
    (["status"], "generated_only_no_simulation"), (["source_commit"], "a" * 40),
    (["git_producer_bindings_checked"], 1), (["native_estimator_calls"], False),
    (["factory_searches"], 0.0), (["gpu_jobs"], 1), (["physical_resource_estimate"], {}),
    (["command", "no_sim"], True), (["command", "steps"], 8.0),
    (["generator_bounds", "qubits"], 65), (["simulation_bounds", "qubits"], 64),
    (["source_sha256", "requirements-lock.txt"], "a" * 64),
    (["scientific_wall_seconds"], -1.0), (["scientific_wall_seconds"], 181.0),
])
def test_replay_manifest_contract_rejected(replay, path, value):
    change(replay, path, value, file="manifest.json")
    # refresh deliberately repairs inventory only, never the altered contract.
    with pytest.raises(ValueError):
        compare(replay)


def test_qasm_mutation_with_repaired_hash_is_still_rejected(replay):
    path = replay / "one-spin/trial-000.qasm"
    path.write_bytes(path.read_bytes().replace(b"reset q[0];", b"x q[0];", 1))
    refresh(replay)
    with pytest.raises(ValueError, match="scheduled primitive trial"):
        compare(replay)


def test_mutated_source_snapshot_with_repaired_hash_rejected(replay):
    path = replay / "qdrift.py"
    path.write_bytes(path.read_bytes() + b"\n")
    refresh(replay)
    with pytest.raises(ValueError, match="snapshot identity changed"):
        compare(replay)


def test_csv_n8_correspondence(replay):
    path = replay / "one-spin/channel-convergence.csv"
    path.write_text(path.read_text().replace("8,0.025914965361466032", "8,0.125914965361466032"))
    refresh(replay)
    with pytest.raises(ValueError, match="correspondence"):
        compare(replay)


def test_csv_other_rows_compare_to_canonical(replay):
    path = replay / "one-spin/channel-convergence.csv"
    path.write_text(path.read_text().replace("1,0.19554044733068415", "1,0.29554044733068415"))
    refresh(replay)
    with pytest.raises(ValueError, match="numerical mismatch"):
        compare(replay)


@pytest.mark.parametrize("name", ["summary.csv", "one-spin/channel-convergence.csv"])
def test_exact_csv_header(replay, name):
    path = replay / name
    path.write_text("unexpected," + path.read_text())
    refresh(replay)
    with pytest.raises(ValueError, match="CSV shape/header"):
        compare(replay)


def test_summary_markdown_correspondence(replay):
    path = replay / "summary.md"
    path.write_text(path.read_text() + "unverified physical estimate\n")
    refresh(replay)
    with pytest.raises(ValueError, match="Markdown"):
        compare(replay)


@pytest.mark.parametrize("column", ["sample_mean", "hoeffding_radius"])
def test_summary_exact_count_and_narrow_radius_fields(replay, column):
    path = replay / "summary.csv"
    rows = path.read_text().splitlines()
    index = rows[0].split(",").index(column)
    row = rows[1].split(",")
    row[index] = str(float(row[index]) + 1e-12)
    rows[1] = ",".join(row)
    path.write_text("\n".join(rows) + "\n")
    refresh(replay)
    with pytest.raises(ValueError):
        compare(replay)


def test_summary_diagnostic_honors_caller_tolerance(replay):
    path = replay / "summary.csv"
    path.write_text(path.read_text().replace("0.7564041065849472", str(0.7564041065849472 + 1e-12)))
    refresh(replay)
    assert compare(replay)["status"] == "verified_fresh_qdrift_replay"
    with pytest.raises(ValueError, match="numerical mismatch"):
        compare(replay, atol=0.0)


@pytest.mark.parametrize("kind", ["missing", "extra", "extra-directory", "temporary", "symlink", "directory-symlink"])
def test_complete_real_inventory(replay, tmp_path, kind):
    if kind == "missing":
        (replay / "summary.md").unlink()
    elif kind in ("extra", "temporary"):
        (replay / ("extra.txt" if kind == "extra" else ".manifest-write.tmp")).write_text("x")
    elif kind == "extra-directory":
        (replay / "empty").mkdir()
    elif kind == "symlink":
        path = replay / "summary.md"
        original = tmp_path / "summary.md"
        path.rename(original)
        path.symlink_to(original)
    else:
        path = replay / "one-spin"
        original = tmp_path / "one-spin"
        path.rename(original)
        path.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError):
        compare(replay)


def test_root_symlink_rejected(replay, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(replay, target_is_directory=True)
    with pytest.raises(ValueError, match="unredirected"):
        compare(link)


@pytest.mark.parametrize("name", ["../summary.md", "/summary.md", "one-spin/../summary.md", "one-spin//trial-000.qasm", "one-spin\\trial-000.qasm", ".", ""])
def test_unsafe_inventory_paths_rejected(name):
    with pytest.raises(ValueError, match="unsafe inventory path"):
        gate._safe_relative(name)


@pytest.mark.parametrize("encoded", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'\xff', b'{'])
def test_json_duplicate_nonfinite_invalid(encoded):
    with pytest.raises(ValueError):
        gate._json(encoded, "fixture")


def test_json_byte_node_and_depth_limits(monkeypatch):
    with pytest.raises(ValueError, match="byte budget"):
        gate._json(b" " * (gate.MAX_JSON_BYTES + 1), "fixture")
    monkeypatch.setattr(gate, "MAX_JSON_NODES", 5)
    with pytest.raises(ValueError, match="node"):
        gate._json(b"[1,2,3,4,5]", "fixture")
    monkeypatch.setattr(gate, "MAX_JSON_NODES", 100)
    monkeypatch.setattr(gate, "MAX_JSON_DEPTH", 2)
    with pytest.raises(ValueError, match="depth"):
        gate._json(b"[[[0]]]", "fixture")


def test_actual_descriptor_read_limit_and_symlink(tmp_path):
    path = tmp_path / "data"
    path.write_bytes(b"123456")
    directory = gate._open_directory(tmp_path)
    try:
        with pytest.raises(ValueError, match="bounded regular file"):
            gate._read(directory, "data", 5)
        assert gate._read(directory, "data", 6) == b"123456"
        (tmp_path / "link").symlink_to(path)
        with pytest.raises(ValueError):
            gate._read(directory, "link", 6)
    finally:
        os.close(directory)


def test_actual_read_cannot_trust_a_smaller_stat_size(tmp_path, monkeypatch):
    (tmp_path / "data").write_bytes(b"123456")
    directory = gate._open_directory(tmp_path)
    try:
        real_stat = os.stat(tmp_path / "data")
        misleading = SimpleNamespace(st_mode=real_stat.st_mode, st_size=5, st_dev=real_stat.st_dev, st_ino=real_stat.st_ino, st_mtime_ns=real_stat.st_mtime_ns, st_ctime_ns=real_stat.st_ctime_ns)
        monkeypatch.setattr(gate.os, "fstat", lambda fd: misleading)
        with pytest.raises(ValueError, match="read exceeds limit"):
            gate._read(directory, "data", 5)
    finally:
        os.close(directory)


def test_inventory_byte_count_rejects_boolean(replay):
    manifest = json.loads((replay / "manifest.json").read_text())
    manifest["files"]["summary.md"]["bytes"] = True
    (replay / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with pytest.raises(ValueError, match="invalid integer"):
        compare(replay)


def test_canonical_manifest_cannot_be_rebound(replay):
    refresh(replay)
    with pytest.raises(ValueError, match="immutable canonical"):
        gate.verify_bundle(replay, source_root=ROOT)
    with pytest.raises(TypeError):
        gate.verify_bundle(CANONICAL, source_root=ROOT, expected_manifest_sha256="a" * 64)


def test_source_root_requires_exact_lock_and_current_sources(tmp_path):
    source = tmp_path / "source"
    for name in gate.SOURCE_SHA256:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    (source / "requirements-lock.txt").write_text("unlocked\n")
    with pytest.raises(ValueError, match="source identity changed"):
        gate.verify_bundle(CANONICAL, source_root=source)


def test_canonical_public_download_needs_no_git(tmp_path):
    source = tmp_path / "download"
    for name in gate.SOURCE_SHA256:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    assert gate.verify_bundle(CANONICAL, source_root=source)["source_commit"] == gate.CANONICAL_PRODUCER


def test_cli_saved_gate_only(monkeypatch, capsys):
    monkeypatch.setattr(gate.argparse.ArgumentParser, "parse_args", lambda self: type("Args", (), {"bundle_dir": CANONICAL, "replay_dir": None})())
    monkeypatch.setattr(gate.subprocess, "run", lambda *a, **k: pytest.fail("canonical CLI must not call Git"))
    gate.main()
    assert json.loads(capsys.readouterr().out)["trials"] == 64


def test_cli_replay_commit_is_acquired_from_checkout(monkeypatch, replay, capsys):
    monkeypatch.setattr(gate.argparse.ArgumentParser, "parse_args", lambda self: type("Args", (), {"bundle_dir": CANONICAL, "replay_dir": replay})())
    calls = []
    def git(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, str(ROOT) + "\n" if command[-1] == "--show-toplevel" else REPLAY_COMMIT + "\n", "")
    monkeypatch.setattr(gate.subprocess, "run", git)
    gate.main()
    assert json.loads(capsys.readouterr().out)["status"] == "verified_fresh_qdrift_replay"
    assert len(calls) == 2 and all(c[1]["timeout"] == 5 for c in calls)


def test_cli_detached_or_nested_git_replay_rejected(monkeypatch):
    def git(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "/different/root\n", "")
    monkeypatch.setattr(gate.subprocess, "run", git)
    with pytest.raises(ValueError, match="checkout root"):
        gate._current_commit(ROOT)
