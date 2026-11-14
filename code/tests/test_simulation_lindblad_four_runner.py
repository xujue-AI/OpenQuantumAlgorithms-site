"""Bounded runner regressions; no native synthesis or resource jobs are run."""
from __future__ import annotations

import importlib.util
import inspect
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import types

import pytest
from qiskit import QuantumCircuit

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/generate_simulation_lindblad_four.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("literal_lindblad_four_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_and_private_cache_domains(runner, tmp_path):
    sources = runner.source_files()
    allowed = ROOT / "assets/generated/simulation/lindblad-four"
    assert runner.validate_paths(allowed, tmp_path / "cache", sources)[0] == allowed
    with pytest.raises(ValueError, match="repository output"):
        runner.validate_paths(ROOT / "assets/generated/simulation/complete-resources", tmp_path / "cache", sources)
    with pytest.raises(ValueError, match="private cache"):
        runner.validate_paths(tmp_path / "output", ROOT / "cache", sources)
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_paths(tmp_path / "output", tmp_path / "output/cache", sources)


@pytest.mark.parametrize("name", ["results.json", "manifest.json", "progress.json", "source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm"])
def test_public_managed_symlinks_reject_before_writes(runner, tmp_path, name):
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "unrelated"
    target.write_text("preserve")
    (output / name).symlink_to(target)
    with pytest.raises(ValueError, match="symbolic"):
        runner.validate_paths(output, tmp_path / "cache", runner.source_files())
    assert target.read_text() == "preserve"


def test_manifest_hashes_only_owned_files(runner, tmp_path):
    (tmp_path / "results.json").write_text("{}\n")
    (tmp_path / "unknown.txt").write_text("user data")
    outside = tmp_path / "outside"
    outside.write_text("not owned")
    (tmp_path / "unknown-link").symlink_to(outside)
    hashes = runner.managed_hashes(tmp_path)
    assert set(hashes) == {"results.json"}
    assert (tmp_path / "unknown.txt").read_text() == "user data"


def test_checkpoint_key_binds_source_packages_and_import_namespace(runner, monkeypatch, tmp_path):
    source = tmp_path / "lindblad_four_physical.py"
    source.write_text("one")
    runner_source = tmp_path / "generate_simulation_lindblad_four.py"
    runner_source.write_text("runner one")
    paths = {source.name: source, runner_source.name: runner_source}
    versions = dict.fromkeys(("qiskit", "qdk", "numpy", "scipy", "matplotlib"), "fixed")
    monkeypatch.setattr(runner.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(runner, "REPOSITORY", True)
    initial = runner.prepared_key(paths)
    runner_source.write_text("runner cosmetic change")
    assert runner.prepared_key(paths) == initial
    source.write_text("two")
    assert runner.prepared_key(paths) != initial
    source.write_text("one")
    versions["numpy"] = "different"
    assert runner.prepared_key(paths) != initial
    versions["numpy"] = "fixed"
    versions["scipy"] = "different"
    assert runner.prepared_key(paths) != initial
    versions["scipy"] = "fixed"
    monkeypatch.setattr(runner, "REPOSITORY", False)
    assert runner.prepared_key(paths) != initial
    monkeypatch.setattr(runner, "REPOSITORY", True)
    assert runner.prepared_key(paths) == initial


def test_git_provenance_removes_external_selector_environment(runner, monkeypatch, tmp_path):
    (tmp_path / ".git").write_text("gitdir: ignored")
    source = tmp_path / "source.py"
    source.write_text("source")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "REPOSITORY", True)
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"):
        monkeypatch.setenv(name, "unrelated selection")
    calls = []
    def git(command, **kwargs):
        calls.append(kwargs["env"])
        if command[-1] == "--show-toplevel":
            return str(tmp_path) + "\n"
        if command[-1] == "HEAD":
            return "a" * 40 + "\n"
        return ""
    monkeypatch.setattr(runner.subprocess, "check_output", git)
    result = runner.revision({"source.py": source})
    assert result["source_commit_at_generation"] == "a" * 40
    assert calls
    assert all(not ({"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"} & set(env)) for env in calls)


def _fake_native_queries(runner, monkeypatch, candidates):
    # Loading definitions performs no factory query. It ensures the real
    # parent package is initialized before replacing this one imported module.
    import qdk.qre.models
    calls = []
    query = types.SimpleNamespace(marker="preserved", enumerate=lambda *args, **kwargs: iter(candidates))
    class BaseFactory:
        @classmethod
        def q(cls, **kwargs):
            calls.append(("factory", cls, kwargs))
            return query
    class Factory(BaseFactory):
        pass
    class Surface:
        @classmethod
        def q(cls, **kwargs):
            calls.append(("surface", cls, kwargs))
            return query
    fake = types.ModuleType("qdk.qre.models")
    fake.RoundBasedFactory, fake.SurfaceCode = Factory, Surface
    monkeypatch.setitem(sys.modules, "qdk.qre.models", fake)
    namespace = {"Factory": Factory, "Surface": Surface}
    exec(compile("def factory_query(**kw):\n return Factory.q(**kw)\ndef surface_query(**kw):\n return Surface.q(**kw)\n",
                 str(Path(runner.ir.__file__).resolve()), "exec"), namespace)
    return Factory, Surface, query, calls, namespace


def test_native_query_guards_preserve_objects_and_restore_descriptors(runner, monkeypatch):
    candidates = [object(), object()]
    factory, surface, query, calls, functions = _fake_native_queries(runner, monkeypatch, candidates)
    factory_descriptor = inspect.getattr_static(factory, "q")
    surface_descriptor = inspect.getattr_static(surface, "q")
    assert "q" not in factory.__dict__
    code = object()
    with runner.bounded_native_queries() as counts:
        assert functions["surface_query"](distance=19) is query
        guarded = functions["factory_query"](code_query=code, cache_dir="private")
        assert guarded.marker == "preserved"
        assert list(guarded.enumerate(object())) == candidates
        assert counts["top_level_factory_queries"] == 1
        assert counts["top_level_surface_code_queries"] == 1
        assert counts["factory_candidates"] == 2
        assert counts["factory_candidates_per_query"] == [2]
    assert inspect.getattr_static(factory, "q") is factory_descriptor
    assert inspect.getattr_static(surface, "q") is surface_descriptor
    assert "q" not in factory.__dict__
    assert calls[1][2]["code_query"] is code


def test_native_query_cap_stops_before_excess_native_call(runner, monkeypatch):
    factory, surface, query, calls, functions = _fake_native_queries(runner, monkeypatch, [])
    monkeypatch.setitem(runner.CAPS, "factory_queries_per_profile", 2)
    with runner.bounded_native_queries() as counts:
        functions["factory_query"]()
        functions["factory_query"]()
        with pytest.raises(runner.ExecutionCapExceeded, match="query cap"):
            functions["factory_query"]()
    assert len(calls) == 2
    assert counts["top_level_factory_queries"] == 3
    assert "q" not in factory.__dict__


def test_candidate_overflow_makes_the_entire_profile_uncomputed(runner, monkeypatch, tmp_path):
    factory, surface, query, calls, functions = _fake_native_queries(runner, monkeypatch, [object(), object(), object()])
    monkeypatch.setitem(runner.CAPS, "factory_candidates_per_query", 2)
    monkeypatch.setattr(runner, "load_prepared", lambda path: object())
    monkeypatch.setattr(runner, "checkpoint_status", lambda *args: {"status": "completed"})
    def estimate(*args, **kwargs):
        list(functions["factory_query"]().enumerate(object()))
        return {"status": "should not be returned", "pareto_points": ["partial"]}
    monkeypatch.setattr(runner.lp, "estimate_prepared", estimate)
    runner.stage_worker("profile", tmp_path, 0, tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "uncomputed_execution_cap"
    assert not (tmp_path / "value.json").exists()
    assert "q" not in factory.__dict__


def test_native_rotation_attempts_are_counted_without_changing_returns(runner, monkeypatch, tmp_path):
    answers = []
    def native(*args, **kwargs):
        result = object()
        answers.append(result)
        return result
    monkeypatch.setattr(runner.lc, "gridsynth_unitary", native)
    report = {}
    returned = []
    def prepare():
        returned.extend(runner.lc.gridsynth_unitary(object(), .01) for _ in range(3))
        return types.SimpleNamespace(synthesized=types.SimpleNamespace(report=report))
    monkeypatch.setattr(runner.lp, "prepare_workload", prepare)
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    assert returned == answers
    assert report["native_rotation_synthesis_attempts"] == 3
    assert report["native_rotation_synthesis_attempts_upper_bound"] == 23632
    assert runner.lc.gridsynth_unitary is native
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "completed"


def test_stage_timeout_refuses_even_a_completed_partial_file(runner, monkeypatch, tmp_path):
    job = tmp_path / "cache/job"
    def time_out(*args, **kwargs):
        (job / "value.json").write_text('{"pareto_points":["partial"]}')
        (job / "status.json").write_text('{"status":"completed"}')
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(runner.subprocess, "run", time_out)
    status = runner.run_stage("profile", job, tmp_path / "cache", .1)
    assert status["status"] == "uncomputed_timeout"
    assert "pareto_points" not in status


def test_private_job_directory_escape_rejects_before_mutations(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "status.json"
    sentinel.write_text("preserve")
    job = cache / "job"
    job.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must reject before subprocess"))
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic|cache"):
        runner.run_stage("prepare", job, cache, 1)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("name", ["status.json", "value.json", "worker.log", "source-trial.qasm", "routed-rotation-trial.qasm"])
def test_private_managed_leaf_symlinks_reject_before_mutations(runner, monkeypatch, tmp_path, name):
    cache = tmp_path / "cache"
    job = cache / "job"
    job.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_text("preserve")
    (job / name).symlink_to(outside)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must reject before subprocess"))
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic|cache"):
        runner.run_stage("prepare", job, cache, 1)
    assert outside.read_text() == "preserve"


def test_checkpoint_leaf_symlinks_reject_on_save_and_load(runner, tmp_path):
    target = tmp_path / "unrelated.pickle"
    original = pickle.dumps({"preserve": True})
    target.write_bytes(original)
    linked = tmp_path / "prepared.pickle"
    linked.symlink_to(target)
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic"):
        runner.save_prepared(linked, {"replace": True})
    assert target.read_bytes() == original
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic"):
        runner.load_prepared(linked)


def test_finite_json_and_checkpoint_byte_caps(runner, monkeypatch, tmp_path):
    monkeypatch.setitem(runner.CAPS, "managed_json_qasm_bytes", 64)
    with pytest.raises(runner.ExecutionCapExceeded, match="JSON"):
        runner.dump(tmp_path / "oversize.json", {"value": "x" * 100})
    assert not (tmp_path / "oversize.json").exists()
    with pytest.raises(ValueError):
        runner.dump(tmp_path / "nan.json", {"value": float("nan")})
    monkeypatch.setitem(runner.CAPS, "private_checkpoint_bytes", 64)
    with pytest.raises(runner.ExecutionCapExceeded, match="checkpoint"):
        runner.save_prepared(tmp_path / "large.pickle", "x" * 100)
    assert not (tmp_path / "large.pickle").exists()
    (tmp_path / "large.pickle").write_bytes(b"x" * 65)
    with pytest.raises(runner.ExecutionCapExceeded, match="checkpoint"):
        runner.load_prepared(tmp_path / "large.pickle")


@pytest.mark.parametrize("folder", ["download", "scripts"])
def test_download_sources_import_without_package_or_ancestor_revision(runner, tmp_path, folder):
    download = tmp_path / folder
    download.mkdir()
    for name, source in runner.source_files().items():
        (download / name).write_bytes(source.read_bytes())
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, str(download / RUNNER.name), "--help"]
    completed = subprocess.run(command, cwd=download, env=environment, capture_output=True, text=True, timeout=30, check=True)
    assert "--reuse-prepared" in completed.stdout
    code = """import importlib.util,json,pathlib
p=pathlib.Path('generate_simulation_lindblad_four.py').resolve()
s=importlib.util.spec_from_file_location('download_runner',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert not m.REPOSITORY
assert pathlib.Path(m.ir.__file__).parent==p.parent
assert pathlib.Path(m.lp.__file__).parent==p.parent
assert pathlib.Path(m.lp.di.__file__).parent==p.parent
assert pathlib.Path(m.lc.__file__).parent==p.parent
for name in m.SOURCES:
 if not name.startswith('generate_'):
  imported=__import__(pathlib.Path(name).stem)
  assert pathlib.Path(imported.__file__).parent==p.parent
assert m.revision(m.source_files())['source_commit_at_generation'] is None
print('standalone imports verified')
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=download, env=environment,
                               capture_output=True, text=True, timeout=30, check=True)
    assert "standalone imports verified" in completed.stdout


def _failed_main_stub(runner, monkeypatch, output, cache):
    family = {str(n): {"n_system": n, "selected_steps": r, "complete_trials": 15756,
                        "status": "preserved_prior_n3_result" if n == 3 else "preserved_prior_n2_result" if n == 2 else "ensemble_branch_cap_exceeded",
                        **({"physical_status": "prior_result_preserved", "regenerated": False} if n in (2, 3) else {"physical_status": "unimplemented_reserved_allocation", "rejection_reproduced": True})}
              for n, r in ((2, 69), (3, 134), (4, 211))}
    _mock_preflight(runner, monkeypatch)
    real_stage = runner.stage_worker
    calls = []
    def stage(name, job, cache_dir, timeout, **kwargs):
        calls.append(name)
        job.mkdir(parents=True, exist_ok=True)
        if name == "preflight":
            real_stage(name, job, 0, cache_dir, invocation_id=kwargs.get("invocation_id"))
            return json.loads((job / "status.json").read_text())
        if name == "family":
            (job / "value.json").write_text(json.dumps(family))
            return {"status": "completed"}
        return {"status": "uncomputed_execution_cap", "reason": "mocked preparation cap"}
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner, "plots", lambda *args, **kwargs: None)
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--output-dir", str(output), "--cache-dir", str(cache), "--reuse-prepared"])
    return calls


def test_parent_cap_cleans_stale_success_and_preserves_unknown_files(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    output.mkdir()
    for name in runner.PUBLIC:
        (output / name).write_text("old successful artifact")
    (output / "unknown.txt").write_text("preserve user material")
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--output-dir", str(output), "--cache-dir", str(cache)])
    def cap():
        raise runner.ExecutionCapExceeded("deliberate parent cap")
    monkeypatch.setattr(runner, "_main", cap)
    runner.main()
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert result["status"] == "uncomputed_parent_execution_cap"
    assert result["prepared"] is None and result["physical_profiles"] == []
    assert result["whole_family_physical_completed"] is False
    assert set(manifest["artifacts_sha256"]) == {"results.json", "artifact-inventory.csv", "artifact-inventory.json"}
    assert not (output / "clifford-t-trial.qasm").exists()
    assert (output / "unknown.txt").read_text() == "preserve user material"


def test_fresh_prepare_retry_invalidates_old_completed_status(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    job = cache / "job"
    job.mkdir(parents=True)
    (job / "prepare-status.json").write_text('{"status":"completed"}')
    (job / "prepared.pickle").write_bytes(pickle.dumps("old complete checkpoint"))
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=1))
    status = runner.run_stage("prepare", job, cache, 1)
    assert status["status"] == "uncomputed_stage_error"
    assert not (job / "prepare-status.json").exists()


def test_reuse_requires_matching_checkpoint_hash_and_retains_reserved_cases(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    calls = _failed_main_stub(runner, monkeypatch, output, cache)
    monkeypatch.setattr(runner, "load_prepared", lambda path: pytest.fail("mismatched checkpoint must not load"))
    job = cache / "prepared" / runner.prepared_key(runner.source_files())
    job.mkdir(parents=True)
    (job / "prepared.pickle").write_bytes(pickle.dumps("changed checkpoint"))
    (job / "prepare-status.json").write_text(json.dumps({"status": "completed", "checkpoint_schema": runner.CHECKPOINT_SCHEMA,
        "checkpoint_identity": runner.prepared_key(runner.source_files()), "prepared_checkpoint_sha256": "0" * 64}))
    runner.main()
    assert calls == ["preflight", "family", "prepare"]
    result = json.loads((output / "results.json").read_text())
    assert result["prepared"] is None
    assert all(report["status"] == "uncomputed_preparation_unavailable" for report in result["physical_profiles"])
    assert result["family"]["4"]["physical_status"] == "unimplemented_reserved_allocation"
    assert result["family_allocations"]["unused_cases_reallocated"] is False
    assert result["family_allocations"]["family_physical_confidence_implemented"] is False
    assert result["source_preflight"] is not None
    assert (output / "source-trial.qasm").is_file()
    assert (output / "routed-rotation-trial.qasm").is_file()
    assert (output / "source-preflight.json").is_file()
    assert not (output / "clifford-t-trial.qasm").exists()


def test_reuse_checks_checkpoint_byte_cap_before_hash_read(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    monkeypatch.setitem(runner.CAPS, "private_checkpoint_bytes", 64)
    _failed_main_stub(runner, monkeypatch, output, cache)
    job = cache / "prepared" / runner.prepared_key(runner.source_files())
    job.mkdir(parents=True)
    checkpoint = job / "prepared.pickle"
    raw = pickle.dumps("x" * 100)
    checkpoint.write_bytes(raw)
    (job / "prepare-status.json").write_text(json.dumps({"status": "completed", "checkpoint_schema": runner.CHECKPOINT_SCHEMA,
        "checkpoint_identity": runner.prepared_key(runner.source_files()), "prepared_checkpoint_sha256": runner.digest(raw)}))
    read = Path.read_bytes
    def bounded_read(path):
        if path == checkpoint:
            pytest.fail("oversized private checkpoint must reject before hash byte-read")
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", bounded_read)
    runner.main()
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "uncomputed_parent_execution_cap"
    assert result["prepared"] is None


def test_parent_watchdog_is_installed_and_cancelled_for_normal_attempt(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    _failed_main_stub(runner, monkeypatch, output, cache)
    timers, handlers = [], []
    monkeypatch.setattr(runner.signal, "setitimer", lambda *args: timers.append(args))
    monkeypatch.setattr(runner.signal, "signal", lambda *args: handlers.append(args))
    runner.main()
    assert (runner.signal.ITIMER_REAL, runner.CAPS["parent_seconds"]) in timers
    assert timers[-1] == (runner.signal.ITIMER_REAL, 0)
    assert handlers[0][0] == runner.signal.SIGALRM
    assert callable(handlers[0][1])
    assert handlers[-1][0] == runner.signal.SIGALRM


def test_predeclared_n4_run_caps_and_complete_export_allowlist(runner):
    assert runner.CAPS["prepare_seconds"] == 1800
    assert runner.CAPS["physical_seconds_per_profile"] == 180
    assert runner.CAPS["family_diagnostic_seconds"] == 180
    assert runner.CAPS["parent_seconds"] == 2400
    assert runner.CAPS["native_rotation_synthesis_attempts_upper_bound"] == 23632
    assert runner.CAPS["routed_and_synthesized_instructions"] == 2_000_000
    assert runner.CAPS["density_validation_instructions"] == 2_000_000
    assert runner.CAPS["private_checkpoint_bytes"] == 256*1024*1024
    assert runner.CAPS["source_preflight_seconds"] == 180
    assert runner.CAPS["diagnostic_aggregate_seconds"] == 180
    assert runner.CAPS["routed_instructions"] == 100_000
    assert runner.CAPS["density_wires"] == 8
    assert runner.CAPS["managed_json_qasm_bytes"] == 32*1024*1024
    assert runner.CAPS["factory_queries_per_profile"] == 24
    assert runner.CAPS["surface_code_queries_per_profile"] == 24
    assert runner.CAPS["factory_candidates_per_query"] == 1000
    assert runner.CAPS["factory_candidates_per_profile"] == 24_000
    assert {"source-trial.qasm", "routed-rotation-trial.qasm", "clifford-t-trial.qasm",
            "bounded_density_instrument.py", "lindblad_four_density.py", "lindblad_four_compile.py", "lindblad_four_physical.py",
            "source-preflight.json", "artifact-inventory.csv", "artifact-inventory.json"} <= set(runner.PUBLIC)
    assert len(runner.SOURCES) == 9


def test_native_rotation_overflow_restores_backend_and_emits_no_checkpoint(runner, monkeypatch, tmp_path):
    native_calls = []
    def native(*args, **kwargs):
        native_calls.append(None)
        return None
    monkeypatch.setattr(runner.lc, "gridsynth_unitary", native)
    maximum = runner.CAPS["native_rotation_synthesis_attempts_upper_bound"]
    def prepare():
        for _ in range(maximum + 1):
            runner.lc.gridsynth_unitary(object(), .01)
        pytest.fail("overflow must stop native preparation")
    monkeypatch.setattr(runner.lp, "prepare_workload", prepare)
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    assert len(native_calls) == maximum
    assert runner.lc.gridsynth_unitary is native
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "uncomputed_execution_cap"
    assert status["native_rotation_synthesis_attempts"] == maximum
    assert not (tmp_path / "prepared.pickle").exists()


def test_total_candidate_cap_rejects_whole_profile_without_truncation(runner, monkeypatch, tmp_path):
    factory, surface, query, calls, functions = _fake_native_queries(runner, monkeypatch, [object(), object()])
    monkeypatch.setitem(runner.CAPS, "factory_candidates_per_query", 2)
    monkeypatch.setitem(runner.CAPS, "factory_candidates_per_profile", 3)
    monkeypatch.setattr(runner, "load_prepared", lambda path: object())
    monkeypatch.setattr(runner, "checkpoint_status", lambda *args: {"status": "completed"})
    def estimate(*args, **kwargs):
        first = list(functions["factory_query"]().enumerate(object()))
        second = list(functions["factory_query"]().enumerate(object()))
        return {"status": "should not be returned", "pareto_points": first + second}
    monkeypatch.setattr(runner.lp, "estimate_prepared", estimate)
    runner.stage_worker("profile", tmp_path, 0, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "uncomputed_execution_cap"
    assert not (tmp_path / "value.json").exists()
    assert "q" not in factory.__dict__


def test_private_checkpoint_tmp_and_parent_symlinks_reject(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve")
    (cache / "via").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must reject before subprocess"))
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic|cache"):
        runner.run_stage("prepare", cache / "via/job", cache, 1)
    assert not (outside / "job").exists()
    job = cache / "normal"
    job.mkdir()
    target = outside / "sentinel"
    (job / "prepared.pickle.tmp").symlink_to(target)
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic|cache"):
        runner.run_stage("prepare", job, cache, 1)
    assert target.read_text() == "preserve"


def test_private_alias_to_cache_parent_still_rejects_before_mutations(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    alias = cache / "alias"
    alias.symlink_to(cache, target_is_directory=True)
    job = cache / "job"
    job.mkdir()
    sentinel = job / "status.json"
    sentinel.write_text("preserve")
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("must reject alias before subprocess"))
    with pytest.raises((ValueError, OSError, runner.ExecutionCapExceeded), match="private|symlink|symbolic|cache"):
        runner.run_stage("prepare", alias / "job", cache, 1)
    assert sentinel.read_text() == "preserve"


@pytest.mark.parametrize("wrong_field", ["checkpoint_schema", "checkpoint_identity"])
def test_reuse_rejects_wrong_checkpoint_namespace_before_content_hash(runner, monkeypatch, tmp_path, wrong_field):
    job = tmp_path / "job"
    job.mkdir()
    raw = pickle.dumps("exact payload")
    (job / "prepared.pickle").write_bytes(raw)
    identity = runner.prepared_key(runner.source_files())
    status = {"status": "completed", "checkpoint_schema": runner.CHECKPOINT_SCHEMA,
              "checkpoint_identity": identity, "prepared_checkpoint_sha256": runner.digest(raw)}
    status[wrong_field] = "other namespace"
    (job / "prepare-status.json").write_text(json.dumps(status))
    monkeypatch.setattr(runner, "prepared_hash", lambda path: pytest.fail("wrong namespace must reject before hash"))
    assert runner.checkpoint_status(job, identity) == {}


def test_reuse_accepts_only_matching_schema_identity_and_bounded_hash(runner, tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    raw = pickle.dumps("exact payload")
    (job / "prepared.pickle").write_bytes(raw)
    identity = runner.prepared_key(runner.source_files())
    status = {"status": "completed", "checkpoint_schema": runner.CHECKPOINT_SCHEMA,
              "checkpoint_identity": identity, "prepared_checkpoint_sha256": runner.digest(raw)}
    (job / "prepare-status.json").write_text(json.dumps(status))
    assert runner.checkpoint_status(job, identity) == status


def test_profile_child_refuses_unverified_checkpoint_before_unpickling(runner, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "load_prepared", lambda path: pytest.fail("must verify profile checkpoint first"))
    runner.stage_worker("profile", tmp_path, 0, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "uncomputed_stage_error"
    assert not (tmp_path / "value.json").exists()


@pytest.mark.parametrize("bad_status", ["{\"status\":", "[]", "null"])
def test_corrupt_checkpoint_status_is_a_cache_miss(runner, tmp_path, bad_status):
    (tmp_path / "prepare-status.json").write_text(bad_status)
    assert runner.checkpoint_status(tmp_path, "identity") == {}


def test_checkpoint_save_overflow_preserves_previous_atomic_payload(runner, monkeypatch, tmp_path):
    path = tmp_path / "prepared.pickle"
    previous = pickle.dumps({"complete": "previous"})
    path.write_bytes(previous)
    monkeypatch.setitem(runner.CAPS, "private_checkpoint_bytes", 64)
    with pytest.raises(runner.ExecutionCapExceeded, match="checkpoint"):
        runner.save_prepared(path, "x" * 100)
    assert path.read_bytes() == previous
    assert not (tmp_path / "prepared.pickle.tmp").exists()


def test_native_factory_cache_symlink_rejects_before_estimation(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    job = cache / "job"
    job.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "sentinel").write_text("preserve")
    (cache / "factories").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(runner, "checkpoint_status", lambda *args: {"status": "completed"})
    monkeypatch.setattr(runner, "load_prepared", lambda path: object())
    monkeypatch.setattr(runner.lp, "estimate_prepared", lambda *args, **kwargs: pytest.fail("must validate native cache first"))
    runner.stage_worker("profile", job, 0, cache)
    assert json.loads((job / "status.json").read_text())["status"] == "uncomputed_stage_error"
    assert not (job / "value.json").exists()
    assert (outside / "sentinel").read_text() == "preserve"


def test_preparation_child_rejects_changed_source_identity_before_checkpoint(runner, monkeypatch, tmp_path):
    identities = iter(["before", "after"])
    monkeypatch.setattr(runner, "prepared_key", lambda *args: next(identities))
    monkeypatch.setattr(runner.lp, "prepare_workload", lambda: types.SimpleNamespace(
        synthesized=types.SimpleNamespace(report={})))
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "uncomputed_stage_error"
    assert not (tmp_path / "prepared.pickle").exists()


def test_payload_inventory_attests_file_bytes_and_excludes_recursive_metadata(runner, tmp_path):
    payloads = {
        "instrument_resources.py": b"copied frozen dependency\n",
        "lindblad_four_physical.py": b"new original engine\n",
        "requirements-lock.txt": b"pinned environment\n",
        "source-trial.qasm": b"OPENQASM 3.0;\n",
        "results.json": b"{}\n",
    }
    for name, raw in payloads.items():
        (tmp_path / name).write_bytes(raw)
    for name in runner.INVENTORY_EXCLUSIONS:
        (tmp_path / name).write_text("old recursive metadata")
    (tmp_path / "unknown.txt").write_text("preserve user material")
    result = runner.artifact_inventory(tmp_path)
    assert result == json.loads((tmp_path / "artifact-inventory.json").read_text())
    indexed = {row["file"]: row for row in result["files"]}
    assert set(indexed) == set(payloads)
    assert result["payload_files"] == len(payloads)
    assert result["payload_bytes"] == sum(map(len, payloads.values()))
    for name, raw in payloads.items():
        assert indexed[name]["bytes"] == len(raw)
        assert indexed[name]["sha256"] == runner.digest(raw)
    assert indexed["instrument_resources.py"]["category"] == "copied_frozen_source_dependency"
    assert indexed["lindblad_four_physical.py"]["category"] == "new_original_source_snapshot"
    assert indexed["requirements-lock.txt"]["category"] == "copied_environment_lock"
    assert set(result["excluded_metadata"]) == set(runner.INVENTORY_EXCLUSIONS)
    assert result["publication_layout_selected"] is False
    assert "not public hosting" in result["scope"]
    assert (tmp_path / "unknown.txt").read_text() == "preserve user material"
    assert (tmp_path / "manifest.json").read_text() == "old recursive metadata"


def _mock_preflight(runner, monkeypatch):
    """A small I/O fixture; source physics is tested by the engine owner."""
    source = QuantumCircuit(8, 8)
    source.reset(range(8))
    source.measure(range(8), range(8))
    record = {"case_id": runner.lp.CASE_ID, "status": "computed_source_preflight",
              "source_counts": dict(source.count_ops()),
              "source_instrument_check": {"mean": 1.0, "acceptance_probability": 1.0}}
    value = types.SimpleNamespace(source=source, routed=source.copy(), as_record=lambda: dict(record))
    monkeypatch.setattr(runner.lp, "source_preflight", lambda: value)
    return value


def test_preflight_attests_both_complete_qasms_and_current_invocation(runner, monkeypatch, tmp_path):
    _mock_preflight(runner, monkeypatch)
    identity = runner.prepared_key(runner.source_files())
    runner.stage_worker("preflight", tmp_path, 0, tmp_path, invocation_id="current-run")
    status = runner.preflight_status(tmp_path, identity, invocation_id="current-run")
    assert status["status"] == "completed"
    value = json.loads((tmp_path / "value.json").read_text())
    assert set(value["qasm_sha256"]) == {"source-trial.qasm", "routed-rotation-trial.qasm"}
    for name, checksum in value["qasm_sha256"].items():
        assert runner.digest((tmp_path / name).read_bytes()) == checksum
    assert status["source_record_sha256"] == runner.digest((tmp_path / "value.json").read_bytes())
    assert runner.preflight_status(tmp_path, identity, invocation_id="old-run") == {}
    assert runner.preflight_status(tmp_path, "other-source", invocation_id="current-run") == {}


@pytest.mark.parametrize("name", ["value.json", "source-trial.qasm", "routed-rotation-trial.qasm"])
def test_preflight_rejects_record_or_qasm_tampering(runner, monkeypatch, tmp_path, name):
    _mock_preflight(runner, monkeypatch)
    identity = runner.prepared_key(runner.source_files())
    runner.stage_worker("preflight", tmp_path, 0, tmp_path, invocation_id="current-run")
    assert runner.preflight_status(tmp_path, identity, invocation_id="current-run")
    if name == "value.json":
        value = json.loads((tmp_path / name).read_text())
        value["status"] = "changed"
        (tmp_path / name).write_text(json.dumps(value))
    else:
        (tmp_path / name).write_text("changed source evidence")
    assert runner.preflight_status(tmp_path, identity, invocation_id="current-run") == {}


def test_preflight_source_race_emits_no_completed_attestation(runner, monkeypatch, tmp_path):
    _mock_preflight(runner, monkeypatch)
    identities = iter(["before", "after"])
    monkeypatch.setattr(runner, "prepared_key", lambda *args: next(identities))
    runner.stage_worker("preflight", tmp_path, 0, tmp_path, invocation_id="current-run")
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "uncomputed_stage_error"
    assert not (tmp_path / "value.json").exists()


def test_structured_compiler_cap_retains_negative_status_without_checkpoint(runner, monkeypatch, tmp_path):
    failure = runner.lc.CompilerLimitExceeded("Clifford+T", 2_000_000, 2_000_010,
        completed_operations=1_999_900, next_segment_operations=110)
    def reject():
        raise failure
    monkeypatch.setattr(runner.lp, "prepare_workload", reject)
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["status"] == "uncomputed_compiler_operation_cap"
    assert status["actual_operations"] == 2_000_010
    assert status["completed_operations"] == 1_999_900
    assert status["partial_trial_is_validated"] is False
    assert status["native_rotation_synthesis_attempts"] == 0
    assert not (tmp_path / "prepared.pickle").exists()


def test_later_parent_cap_retains_only_current_verified_source_evidence(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    calls = _failed_main_stub(runner, monkeypatch, output, cache)
    staged = runner.run_stage
    def capped(name, *args, **kwargs):
        if name == "prepare":
            calls.append(name)
            raise runner.ExecutionCapExceeded("deliberate later parent cap")
        return staged(name, *args, **kwargs)
    monkeypatch.setattr(runner, "run_stage", capped)
    runner.main()
    result = json.loads((output / "results.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert calls == ["preflight", "family", "prepare"]
    assert result["status"] == "uncomputed_parent_execution_cap"
    assert result["prepared"] is None and result["physical_profiles"] == []
    assert result["source_preflight"] is not None
    assert (output / "source-preflight.json").is_file()
    assert not (output / "clifford-t-trial.qasm").exists()
    for name in ("source-trial.qasm", "routed-rotation-trial.qasm"):
        assert name in manifest["artifacts_sha256"]
        assert runner.digest((output / name).read_bytes()) == manifest["artifacts_sha256"][name]
    assert manifest["status"] == "uncomputed_parent_execution_cap"
    assert result["whole_family_physical_completed"] is False


def test_a_new_attempt_cannot_recover_an_old_active_preflight_context(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    output.mkdir(); cache.mkdir()
    _mock_preflight(runner, monkeypatch)
    paths = runner.source_files()
    identity = runner.prepared_key(paths)
    job = cache / "preflight-old"
    job.mkdir()
    runner.stage_worker("preflight", job, 0, cache, invocation_id="old-run")
    sources, extras = runner.capture_sources(paths)
    runner.ACTIVE_PREFLIGHT = {"output": output.resolve(), "cache": cache.resolve(),
        "sources": sources, "extras": extras, "identity": identity, "frozen_before": runner.fingerprints(),
        "job": job, "invocation_id": "old-run", "readme": runner.readme(), "family": {}}
    monkeypatch.setattr(sys, "argv", [str(RUNNER), "--output-dir", str(output), "--cache-dir", str(cache)])
    def cap():
        raise runner.ExecutionCapExceeded("new attempt failed before source preflight")
    monkeypatch.setattr(runner, "_main", cap)
    runner.main()
    result = json.loads((output / "results.json").read_text())
    assert result["status"] == "uncomputed_parent_execution_cap"
    assert result.get("source_preflight") is None
    assert not (output / "source-preflight.json").exists()
    assert not (output / "source-trial.qasm").exists()
    assert not (output / "routed-rotation-trial.qasm").exists()


def test_actual_runtime_module_locations_and_hashes_are_attested(runner):
    paths = runner.source_files()
    result = runner.verify_runtime_sources(paths)
    assert result["actual_module_paths_verified"] is True
    assert result["import_mode"] == "package"
    assert set(result["source_sha256"]) == set(runner.SOURCES) - {RUNNER.name}
    for name, checksum in result["source_sha256"].items():
        assert runner.digest(paths[name].read_bytes()) == checksum


def test_wrong_actual_import_path_rejects_before_preflight_or_native(runner, monkeypatch, tmp_path):
    unrelated = tmp_path / "lindblad_four_physical.py"
    unrelated.write_bytes(runner.source_files()["lindblad_four_physical.py"].read_bytes())
    monkeypatch.setattr(runner.lp, "__file__", str(unrelated))
    monkeypatch.setattr(runner.lp, "source_preflight", lambda: pytest.fail("must check actual imports before preflight"))
    monkeypatch.setattr(runner.lp, "prepare_workload", lambda: pytest.fail("must check actual imports before native"))
    with pytest.raises(ValueError, match="runtime import"):
        runner.verify_runtime_sources(runner.source_files())
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "uncomputed_stage_error"
    assert not (tmp_path / "prepared.pickle").exists()


def test_oversized_child_status_rejects_before_opening_its_contents(runner, monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    job = cache / "job"
    status_path = job / "status.json"
    monkeypatch.setitem(runner.CAPS, "managed_json_qasm_bytes", 64)
    def child(*args, **kwargs):
        with open(status_path, "wb") as handle:
            handle.write(b"x" * 65)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", child)
    original = Path.open
    def guarded(path, *args, **kwargs):
        if path == status_path:
            pytest.fail("oversized child status must reject before opening its contents")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises((ValueError, runner.ExecutionCapExceeded), match="bounded|cap|32 MiB"):
        runner.run_stage("profile", job, cache, 1)


@pytest.mark.parametrize("raw", ["[]", "null", "{\"status\":"])
def test_child_status_must_be_a_complete_json_object(runner, monkeypatch, tmp_path, raw):
    cache = tmp_path / "cache"
    job = cache / "job"
    def child(*args, **kwargs):
        (job / "status.json").write_text(raw)
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, "run", child)
    with pytest.raises(ValueError):
        runner.run_stage("profile", job, cache, 1)
