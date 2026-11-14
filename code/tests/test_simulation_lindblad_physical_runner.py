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

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/generate_simulation_lindblad_physical.py"


@pytest.fixture
def runner():
    spec = importlib.util.spec_from_file_location("literal_lindblad_runner_test", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_and_private_cache_domains(runner, tmp_path):
    sources = runner.source_files()
    allowed = ROOT / "assets/generated/simulation/lindblad-physical"
    assert runner.validate_paths(allowed, tmp_path / "cache", sources)[0] == allowed
    with pytest.raises(ValueError, match="repository output"):
        runner.validate_paths(ROOT / "assets/generated/simulation/complete-resources", tmp_path / "cache", sources)
    with pytest.raises(ValueError, match="private cache"):
        runner.validate_paths(tmp_path / "output", ROOT / "cache", sources)
    with pytest.raises(ValueError, match="disjoint"):
        runner.validate_paths(tmp_path / "output", tmp_path / "output/cache", sources)


@pytest.mark.parametrize("name", ["results.json", "manifest.json", "progress.json", "source-trial.qasm"])
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
    source = tmp_path / "lindblad_physical_workload.py"
    source.write_text("one")
    runner_source = tmp_path / "generate_simulation_lindblad_physical.py"
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
    monkeypatch.setattr(runner.ir, "gridsynth_unitary", native)
    report = {}
    returned = []
    def prepare():
        returned.extend(runner.ir.gridsynth_unitary(object(), .01) for _ in range(3))
        return types.SimpleNamespace(synthesized=types.SimpleNamespace(report=report))
    monkeypatch.setattr(runner.lp, "prepare_workload", prepare)
    runner.stage_worker("prepare", tmp_path, 0, tmp_path)
    assert returned == answers
    assert report["native_rotation_synthesis_attempts"] == 3
    assert report["native_rotation_synthesis_attempts_upper_bound"] == 3312
    assert runner.ir.gridsynth_unitary is native
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


@pytest.mark.parametrize("name", ["status.json", "value.json", "worker.log"])
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


def test_download_sources_import_without_package_or_ancestor_revision(runner, tmp_path):
    download = tmp_path / "download"
    download.mkdir()
    for name, source in runner.source_files().items():
        (download / name).write_bytes(source.read_bytes())
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    command = [sys.executable, str(download / RUNNER.name), "--help"]
    completed = subprocess.run(command, cwd=download, env=environment, capture_output=True, text=True, timeout=30, check=True)
    assert "--reuse-prepared" in completed.stdout
    code = """import importlib.util,json,pathlib
p=pathlib.Path('generate_simulation_lindblad_physical.py').resolve()
s=importlib.util.spec_from_file_location('download_runner',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert not m.REPOSITORY
assert pathlib.Path(m.ir.__file__).parent==p.parent
assert pathlib.Path(m.lp.__file__).parent==p.parent
assert m.revision(m.source_files())['source_commit_at_generation'] is None
print('standalone imports verified')
"""
    completed = subprocess.run([sys.executable, "-c", code], cwd=download, env=environment,
                               capture_output=True, text=True, timeout=30, check=True)
    assert "standalone imports verified" in completed.stdout


def _failed_main_stub(runner, monkeypatch, output, cache):
    family = {str(n): {"n_system": n, "selected_steps": r, "complete_trials": 15756,
                        "status": "implemented_four_wire_density_slice" if n == 2 else "ensemble_branch_cap_exceeded",
                        **({} if n == 2 else {"physical_status": "unimplemented_reserved_allocation", "rejection_reproduced": True})}
              for n, r in ((2, 69), (3, 134), (4, 211))}
    calls = []
    def stage(name, job, cache_dir, timeout, **kwargs):
        calls.append(name)
        job.mkdir(parents=True, exist_ok=True)
        if name == "family":
            (job / "value.json").write_text(json.dumps(family))
            return {"status": "completed"}
        return {"status": "uncomputed_execution_cap", "reason": "mocked preparation cap"}
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(runner, "plots", lambda *args: None)
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
    assert set(manifest["artifacts_sha256"]) == {"results.json"}
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
    job = cache / "prepared" / runner.prepared_key(runner.source_files())
    job.mkdir(parents=True)
    (job / "prepared.pickle").write_bytes(pickle.dumps("changed checkpoint"))
    (job / "prepare-status.json").write_text(json.dumps({"status": "completed", "prepared_checkpoint_sha256": "0" * 64}))
    calls = _failed_main_stub(runner, monkeypatch, output, cache)
    monkeypatch.setattr(runner, "load_prepared", lambda path: pytest.fail("mismatched checkpoint must not load"))
    runner.main()
    assert calls == ["family", "prepare"]
    result = json.loads((output / "results.json").read_text())
    assert result["prepared"] is None
    assert all(report["status"] == "uncomputed_preparation_unavailable" for report in result["physical_profiles"])
    assert result["family"]["3"]["physical_status"] == "unimplemented_reserved_allocation"
    assert result["family_allocations"]["unused_cases_reallocated"] is False
    assert result["family_allocations"]["family_physical_confidence_implemented"] is False


def test_reuse_checks_checkpoint_byte_cap_before_hash_read(runner, monkeypatch, tmp_path):
    output, cache = tmp_path / "output", tmp_path / "cache"
    job = cache / "prepared" / runner.prepared_key(runner.source_files())
    job.mkdir(parents=True)
    checkpoint = job / "prepared.pickle"
    raw = pickle.dumps("x" * 100)
    checkpoint.write_bytes(raw)
    (job / "prepare-status.json").write_text(json.dumps({"status": "completed", "prepared_checkpoint_sha256": runner.digest(raw)}))
    _failed_main_stub(runner, monkeypatch, output, cache)
    monkeypatch.setitem(runner.CAPS, "private_checkpoint_bytes", 64)
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
