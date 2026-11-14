"""Port-runner path/proof/watchdog fixtures; no full candidate/native jobs."""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("port_runner_fixture", ROOT / "scripts/run_simulation_lindblad_port.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_owned_output_and_private_cache_domains(runner, tmp_path):
    owned = ROOT / "assets/generated/simulation/lindblad-port"
    baseline = ROOT / "assets/generated/simulation/lindblad-four"
    assert runner.validate_paths(owned, tmp_path / "cache", baseline)[0] == owned
    with pytest.raises(ValueError, match="repository output"):
        runner.validate_paths(ROOT / "assets/generated/simulation/lindblad-four", tmp_path / "cache", baseline)
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_paths(tmp_path / "output", tmp_path / "output/cache", baseline)
    with pytest.raises(ValueError, match="outside"):
        runner.validate_paths(owned, ROOT / ".cache/port", baseline)


@pytest.mark.parametrize("leaf", ["words/word-00.qasm", "recipe.json", "word-catalog.json", "source-proof.json", "candidate-trial.qasm", "manifest.json"])
def test_all_managed_symlinks_reject_before_writes(runner, tmp_path, leaf):
    output = tmp_path / "output"
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (output / leaf).parent.mkdir(parents=True, exist_ok=True)
    (output / leaf).symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        runner.validate_paths(output, tmp_path / "cache", tmp_path / "baseline")
    assert target.read_text() == "preserve"


def test_lexical_parent_alias_rejects(runner, tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        runner.reject_symlink_chain(alias / "file.json")


def test_bounded_atomic_bytes_preserve_unrelated_and_reject_overflow(runner, tmp_path):
    path = tmp_path / "payload.json"
    path.write_bytes(b"old")
    with pytest.raises(runner.ExecutionCapExceeded, match="byte cap"):
        runner.atomic_bytes(path, b"12345", 4)
    assert path.read_bytes() == b"old"
    assert runner.atomic_bytes(path, b"ok", 4) == runner.digest(b"ok")
    assert path.read_bytes() == b"ok"
    assert not (tmp_path / "payload.json.tmp").exists()


def test_preexisting_temporary_file_is_not_removed(runner, tmp_path):
    path = tmp_path / "payload.json"
    temporary = path.with_name("payload.json.tmp")
    temporary.write_text("user content")
    # Atomic helpers should never truncate a preexisting temporary target.
    with pytest.raises(FileExistsError):
        runner.atomic_bytes(path, b"new")
    assert temporary.read_text() == "user content"


def test_bounded_hash_rejects_symlink_and_size(runner, tmp_path):
    path = tmp_path / "value"
    path.write_bytes(b"12345")
    with pytest.raises(ValueError, match="bounded"):
        runner.bounded_file_hash(path, 4)
    alias = tmp_path / "alias"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symbolic"):
        runner.bounded_file_hash(alias)


def test_source_identity_binds_all_source_runtime_and_caps(runner):
    hashes, runtime = {"words.py": "a" * 64}, {"python": "3.12.14"}
    first = runner.source_identity(hashes, runtime)
    assert runner.source_identity({"words.py": "b" * 64}, runtime) != first
    assert runner.source_identity(hashes, {"python": "3.12.13"}) != first
    old = runner.CAPS["rss_bytes"]
    runner.CAPS["rss_bytes"] = old - 1
    assert runner.source_identity(hashes, runtime) != first


def test_descendant_rss_checks_each_process_and_nested_children(runner):
    table = {1: (0, 100), 2: (1, 200), 3: (2, 300), 4: (9, 400)}
    assert runner.descendant_pids(table, 1) == {1, 2, 3}
    runner.enforce_rss(table, 1)
    table[3] = (2, runner.CAPS["rss_bytes"] + 1)
    with pytest.raises(runner.ExecutionCapExceeded, match="RSS"):
        runner.enforce_rss(table, 1)


def test_watchdog_kills_blocked_child_before_first_yield(runner, monkeypatch, tmp_path):
    log = tmp_path / "worker.log"
    log.write_bytes(b"")
    killed = []
    process = types.SimpleNamespace(pid=456, poll=lambda: None, wait=lambda timeout: 0)
    monkeypatch.setattr(runner, "process_table", lambda: {os.getpid(): (0, 100), 456: (os.getpid(), runner.CAPS["rss_bytes"] + 1)})
    monkeypatch.setattr(runner, "kill_tree", killed.append)
    result = runner.watch_process(process, 180, log, parent_began=runner.time.monotonic())
    assert result["status"] == "uncomputed_execution_cap"
    assert killed == [456]


def test_watchdog_interrupt_kills_and_reaps_descendants(runner, monkeypatch, tmp_path):
    killed, reaped = [], []
    process = types.SimpleNamespace(pid=456, wait=lambda timeout: reaped.append(456))
    def interrupted(*args, **kwargs):
        raise runner.ExecutionCapExceeded("parent signal at arbitrary boundary")
    monkeypatch.setattr(runner, "_watch_process_body", interrupted)
    monkeypatch.setattr(runner, "kill_tree", killed.append)
    with pytest.raises(runner.ExecutionCapExceeded, match="parent signal"):
        runner.watch_process(process, 180, tmp_path / "log", parent_began=0)
    assert killed == [456]
    assert reaped == [456]


def test_native_monitor_activates_only_after_child_rss_and_log_checks(runner, monkeypatch, tmp_path):
    log = tmp_path / "worker.log"
    log.write_bytes(b"")
    calls = []
    alive = iter([None, 0])
    process = types.SimpleNamespace(pid=456, poll=lambda: next(alive), wait=lambda timeout: 0)
    event = types.SimpleNamespace(is_set=lambda: False, set=lambda: calls.append("activation"))
    monkeypatch.setattr(runner, "process_table", lambda: {os.getpid(): (0, 100), 456: (os.getpid(), 200)})
    monkeypatch.setattr(runner, "enforce_rss", lambda table, pid: calls.append(("rss", pid)))
    monkeypatch.setattr(runner, "checked_boundary", lambda *args: 100)
    monkeypatch.setattr(runner.time, "sleep", lambda seconds: None)
    assert runner.watch_process(process, 180, log, parent_began=runner.time.monotonic(), activation=event) is None
    assert calls[:3] == [("rss", os.getpid()), ("rss", 456), "activation"]


def test_native_controller_ineligible_never_forks_or_issues_capability(runner, tmp_path):
    @dataclass(frozen=True)
    class Profile:
        name: str = "frozen"
    resource = types.SimpleNamespace(controller_eligibility=lambda *args: {"budget_exceeded": True})
    result = runner.run_native_profile(object(), Profile(), 0, tmp_path, resource, object(), runner_sha256="x", parent_began=0)
    assert result["status"] == "infeasible_controller_model"
    assert result["pareto_points"] == result["all_feasible_points"] == []
    assert result["execution_query_counts"]["top_level_factory_queries"] == 0
    assert not list(tmp_path.iterdir())


def test_candidate_qasm_stream_keeps_phase_once_and_complete_readout(runner, tmp_path):
    from qiskit import QuantumCircuit
    circuit = QuantumCircuit(8, 8)
    circuit.global_phase = 0.321
    circuit.reset(range(8))
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.barrier(7, 6, 5, 4, 3, 2, 1, 0)
    circuit.measure(range(8), range(8))
    path = tmp_path / "candidate.qasm"
    expected = runner.dump_candidate_qasm(path, circuit)
    raw = path.read_text()
    assert raw.count("gphase(") == 1
    assert "gphase(0.321);" in raw
    assert raw.count("= measure") == 8
    assert "barrier q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0];" in raw
    assert expected == runner.bounded_file_hash(path)


def test_archive_stream_cap_never_installs_partial_checkpoint(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(runner.CAPS, "checkpoint_bytes", 8)
    candidate = types.SimpleNamespace(circuit="too many bytes", report={})
    path = tmp_path / "candidate.pickle"
    with pytest.raises(runner.ExecutionCapExceeded, match="256MiB"):
        runner.save_candidate_archive(path, candidate, {})
    assert not path.exists()
    assert not path.with_name(path.name + ".tmp").exists()


def test_inventory_never_hashes_unknown_files(runner, tmp_path):
    (tmp_path / "recipe.json").write_text("{}\n")
    (tmp_path / "user-secret.txt").write_text("unmanaged")
    hashes = runner.managed_hashes(tmp_path)
    assert set(hashes) == {"recipe.json"}


def test_source_stage_exports_only_recovered_recipe_and_sources(runner, monkeypatch, tmp_path):
    @dataclass(frozen=True)
    class Limits:
        cap: int = 8
    recovered = types.SimpleNamespace(recovery_sha256="r" * 64, source_artifacts={"manifest.json": "m" * 64})
    recipe_raw = b'{"summary":{"candidate_operations":3}}\n'
    recipe = types.SimpleNamespace(canonical_json=recipe_raw, recipe_sha256=runner.digest(recipe_raw), as_record=lambda: json.loads(recipe_raw))
    calls = []
    words = types.SimpleNamespace(literal_limits=Limits, recover_words=lambda *args, **kwargs: calls.append("recover") or recovered,
        recovery_artifacts=lambda *args, **kwargs: {"words/word-00.qasm": b"actual phased word", "word-catalog.json": b"{}\n"})
    rewrite = types.SimpleNamespace(literal_limits=Limits, build_recipe=lambda *args, **kwargs: calls.append("recipe") or recipe)
    monkeypatch.setattr(runner, "module", lambda name: words if name.endswith("words") else rewrite)
    monkeypatch.setattr(runner, "runtime_record", lambda: {"python": "3.12.14", "packages": {}})
    monkeypatch.setattr(runner, "verify_runtime_sources", lambda: {"source_sha256": {"words.py": runner.digest(b"source")}})
    monkeypatch.setattr(runner, "capture_source_snapshots", lambda: {"words.py": b"source"})
    monkeypatch.setattr(runner, "extras", lambda: {})
    monkeypatch.setattr(runner, "revision", lambda: "a" * 40)
    proof = runner.source_stage(tmp_path / "output", tmp_path / "baseline")
    assert calls == ["recover", "recipe"]
    assert proof["candidate_materialized"] is False
    assert proof["candidate_full_density_executed"] is False
    assert proof["native_physical_queries"] == 0
    assert not (tmp_path / "output/candidate-trial.qasm").exists()
    assert (tmp_path / "output/recipe.json").read_bytes() == recipe_raw


def test_source_changes_fail_before_any_artifact_mutation(runner, monkeypatch, tmp_path):
    @dataclass(frozen=True)
    class Limits:
        cap: int = 8
    recovered = types.SimpleNamespace(recovery_sha256="r" * 64, source_artifacts={})
    recipe = types.SimpleNamespace(canonical_json=b"{}\n", recipe_sha256="r" * 64, as_record=lambda: {})
    words = types.SimpleNamespace(literal_limits=Limits, recover_words=lambda *args, **kwargs: recovered,
        recovery_artifacts=lambda *args, **kwargs: {"word-catalog.json": b"{}\n"})
    rewrite = types.SimpleNamespace(literal_limits=Limits, build_recipe=lambda *args, **kwargs: recipe)
    monkeypatch.setattr(runner, "module", lambda name: words if name.endswith("words") else rewrite)
    monkeypatch.setattr(runner, "runtime_record", lambda: {"python": "3.12.14"})
    monkeypatch.setattr(runner, "verify_runtime_sources", lambda: {"source_sha256": {"words.py": "old"}})
    monkeypatch.setattr(runner, "capture_source_snapshots", lambda: {"words.py": b"changed"})
    with pytest.raises(ValueError, match="source changed"):
        runner.source_stage(tmp_path / "output", tmp_path / "baseline")
    assert not (tmp_path / "output").exists()


def _gate_fixture(runner, monkeypatch, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    for name in (*runner.WORDS, "word-catalog.json", "recipe.json", "source-proof.json"):
        (output / name).parent.mkdir(parents=True, exist_ok=True)
        (output / name).write_text("{}\n")
    source_path = tmp_path / "src/openquantum/simulation/lindblad_port_resources.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("adapter frozen")
    core = tmp_path / "core.py"
    core.write_text("core")
    notes = {}
    for kind in ("word_recipe", "adapter", "runner"):
        name = "docs/coordination/" + kind + "-review.md"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("approved review")
        notes[kind] = {"status": "approved", "review_note": name, "review_note_sha256": runner.bounded_file_hash(path)}
    proof = {"schema_version": 1, "status": "approved_frozen_port_recipe", "frozen_source_commit": "a" * 40,
        "source_proof_sha256": runner.bounded_file_hash(output / "source-proof.json"),
        "recipe_sha256": "r" * 64, "catalog_recovery_sha256": "c" * 64,
        "runner_source_sha256": runner.bounded_file_hash(Path(runner.__file__)),
        "resource_source_sha256": runner.bounded_file_hash(source_path), "approvals": notes}
    proof["approvals"]["adapter"]["source_sha256"] = proof["resource_source_sha256"]
    proof["approvals"]["runner"]["source_sha256"] = proof["runner_source_sha256"]
    source = {"recipe_sha256": "r" * 64, "catalog_recovery_sha256": "c" * 64, "source_sha256": {"core.py": runner.bounded_file_hash(core)}}
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "SOURCE", tmp_path / "src")
    monkeypatch.setattr(runner, "validate_source_proof", lambda *args: source)
    monkeypatch.setattr(runner, "source_files", lambda: {"core.py": core})
    monkeypatch.setattr(runner, "git", lambda *args: b"")
    monkeypatch.setattr(runner, "verify_runner_code", lambda: runner.bounded_file_hash(Path(runner.__file__)))
    def committed(commit, path):
        return runner.bounded_file_hash(core) if path == core else runner.bounded_file_hash(output / path.name) if path.name not in {"word-00.qasm", "word-01.qasm", "word-02.qasm", "word-03.qasm", "word-04.qasm", "word-05.qasm", "word-06.qasm", "word-07.qasm"} else runner.bounded_file_hash(output / "words" / path.name)
    monkeypatch.setattr(runner, "_committed_hash", committed)
    proof_path = tmp_path / "approval.json"
    proof_path.write_text(json.dumps(proof))
    return output, proof_path, proof


def test_exact_frozen_review_gate_accepts_only_bound_commit_artifacts_and_code(runner, monkeypatch, tmp_path):
    output, proof_path, proof = _gate_fixture(runner, monkeypatch, tmp_path)
    assert runner.validate_review_proof(proof_path, output, tmp_path / "baseline", include_native=True)[1] == proof
    proof["runner_source_sha256"] = "stale"
    proof_path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="runner_source_sha256"):
        runner.validate_review_proof(proof_path, output, tmp_path / "baseline", include_native=True)


def test_native_gate_rejects_missing_adapter_review(runner, monkeypatch, tmp_path):
    output, proof_path, proof = _gate_fixture(runner, monkeypatch, tmp_path)
    del proof["approvals"]["adapter"]
    proof_path.write_text(json.dumps(proof))
    # Candidate generation needs the exact verifier source binding, while the
    # extra native gate requires independently approved adapter and runner.
    runner.validate_review_proof(proof_path, output, tmp_path / "baseline", include_native=False)
    with pytest.raises(ValueError, match="approval"):
        runner.validate_review_proof(proof_path, output, tmp_path / "baseline", include_native=True)


def test_review_note_rejects_unbound_external_paths(runner, tmp_path):
    with pytest.raises(ValueError, match="coordination"):
        runner._review_note({"status": "approved", "review_note": str(tmp_path / "note.md")})


def test_hard_parent_watchdog_records_boundary_through_context_tail(runner, monkeypatch):
    events = []
    monkeypatch.setattr(runner, "checked_boundary", lambda began, seconds: events.append(seconds))
    with runner.parent_watchdog():
        events.append("artifact tail")
    assert events == ["artifact tail", 2400]


def test_loaded_runner_watchdog_helpers_are_sealed_to_reviewed_code(runner, monkeypatch):
    assert runner.verify_runner_code() == runner.bounded_file_hash(Path(runner.__file__))
    monkeypatch.setattr(runner, "_watch_process_body", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="loaded runner helper"):
        runner.verify_runner_code()


def test_wraps_cannot_hide_replaced_contextmanager_watchdog(runner, monkeypatch):
    import functools
    original = runner.parent_watchdog
    @functools.wraps(original)
    def zero_watchdog(*args, **kwargs):
        return None
    monkeypatch.setattr(runner, "parent_watchdog", zero_watchdog)
    with pytest.raises(ValueError, match="contextmanager wrapper"):
        runner.verify_runner_code()


def test_parent_tail_failure_removes_completed_manifest_and_records_failure(runner, monkeypatch, tmp_path):
    (tmp_path / "manifest.json").write_text('{"status":"completed"}')
    (tmp_path / "results.json").write_text('{"status":"completed_candidate","execution":{}}')
    with pytest.raises(runner.ExecutionCapExceeded):
        with runner.parent_watchdog(on_failure=lambda error: runner.clear_failed_parent_artifacts(tmp_path, "execute", [True], error)):
            raise runner.ExecutionCapExceeded("tail cap")
    assert not (tmp_path / "manifest.json").exists()
    assert json.loads((tmp_path / "progress.json").read_text())["status"] == "uncomputed_parent_execution_cap"
    assert json.loads((tmp_path / "results.json").read_text())["status"] == "completed_candidate_parent_finalization_failed"


def test_preexisting_managed_temporary_is_preserved_by_preflight(runner, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    temporary = output / "recipe.json.tmp"
    temporary.write_text("user content")
    with pytest.raises(ValueError, match="temporary"):
        runner.validate_paths(output, tmp_path / "cache", tmp_path / "baseline")
    assert temporary.read_text() == "user content"
