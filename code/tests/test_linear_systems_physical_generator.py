from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from qiskit import qasm3

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("linear_physical_generator", ROOT / "scripts/generate_linear_systems_physical.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def fake_estimator(calls):
    def estimate(source, *, budget, profile, cache_dir, optimization_level, use_graph):
        calls.append((source, budget, profile, cache_dir))
        assert optimization_level == 1 and use_graph is False
        qasm = qasm3.dumps(generator.decompose(source))
        return {"status": "computed_conditional_model", "qasm3": qasm,
                "qasm_sha256": hashlib.sha256(qasm.encode()).hexdigest(),
                "physical_profile": asdict(profile), "budget": asdict(budget),
                "isa_graph_pruning": False,
                "pareto_points": [{"peak_physical_qubits": 120,
                                   "runtime_per_execution_ns": 100,
                                   "serial_runtime_all_executions_ns": 100 * budget.executions}]}
    return estimate


def test_all_declared_cases_make_exactly_13_full_capped_queries_with_fixed_contract(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(generator, "estimate_circuit_resources", fake_estimator(calls))
    output = tmp_path / "output"
    manifest = generator.run_sweep(output, estimate=True, simulate=False, cache_dir=tmp_path / "cache")
    assert len(calls) == manifest["qdk_query_count"] == 13
    assert manifest["generated_case_count"] == 13
    assert manifest["baseline_verification"]["unchanged"] is True
    assert manifest["hardware_output_accuracy_certified"] is False
    assert manifest["simultaneous_cross_grid_guarantee"] is False
    assert manifest["independent_validation_count"] == 0
    for case, (_, budget, profile, _) in zip(generator.CASES, calls, strict=True):
        directory = output / case.case_id
        ledger = json.loads((directory / "logical-resources.json").read_text())
        report = json.loads((directory / "surface-code.json").read_text())
        assert budget.executions == ledger["workload"]["total_capped_attempts"]
        assert profile == generator.PROFILE
        assert report["initialization_included"] is False
        assert report["hardware_output_accuracy_certified"] is False
        assert report["contract"]["requested_absolute_error"] == .05
        assert report["postselection_contract"]["coherent_eta_interpretation_certified"] is False
        assert report["postselection_contract"]["conditional_total_error_upper_bound"] <= .05
        qasm_hash = hashlib.sha256((directory / "measured-trial.qasm").read_bytes()).hexdigest()
        assert ledger["qasm_sha256"] == report["qasm_sha256"] == qasm_hash
    assert (output / "physical-scaling.svg").is_file()


def test_no_estimate_clears_only_known_stale_artifacts_and_preserves_reference_fingerprint(tmp_path, monkeypatch):
    output = tmp_path / "output"; output.mkdir()
    selected = generator.CASES[0]
    for case in generator.CASES:
        directory = output / case.case_id; directory.mkdir()
        for filename in generator.CASE_FILES:
            (directory / filename).write_text("stale")
    unselected = output / generator.CASES[1].case_id
    (unselected / "unrelated.txt").write_text("preserve")
    (output / "physical-scaling.svg").write_text("stale")
    (output / "physical-scaling.png").write_bytes(b"stale")
    monkeypatch.setattr(generator, "estimate_circuit_resources", lambda *args, **kwargs: pytest.fail("native query on --no-estimate"))
    manifest = generator.run_sweep(output, (selected,), estimate=False, simulate=True, cache_dir=tmp_path / "cache")
    assert manifest["qdk_query_count"] == 0
    assert manifest["independent_validation_count"] == 1
    assert not (output / selected.case_id / "surface-code.json").exists()
    assert not (output / "physical-scaling.svg").exists()
    assert not (output / "physical-scaling.png").exists()
    assert (unselected / "unrelated.txt").read_text() == "preserve"
    for filename in generator.CASE_FILES:
        assert not (unselected / filename).exists()
    reference = json.loads((output / selected.case_id / "reference.json").read_text())
    ledger = json.loads((output / selected.case_id / "logical-resources.json").read_text())
    assert reference["reference_identity_sha256"] == ledger["reference_identity_sha256"]
    assert hashlib.sha256((output / selected.case_id / "reference.json").read_bytes()).hexdigest() == ledger["reference_json_sha256"]


def test_public_code_replay_records_null_git_beneath_unrelated_repository(tmp_path):
    output = tmp_path / "export"
    generator.run_sweep(output, (generator.CASES[0],), estimate=False, simulate=False, cache_dir=tmp_path / "cache")
    ancestor = tmp_path / "unrelated"; ancestor.mkdir()
    subprocess.run(["git", "init", "-q", str(ancestor)], check=True)
    (ancestor / "unrelated.txt").write_text("An unrelated repository.\n")
    subprocess.run(["git", "-C", str(ancestor), "add", "unrelated.txt"], check=True)
    subprocess.run(["git", "-C", str(ancestor), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture"], check=True)
    code = ancestor / "download"
    shutil.copytree(output / "code", code)
    replay = ancestor / "replayed"
    env = os.environ.copy()
    env.update({"PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1", "GIT_DIR": str(ancestor / ".git"), "GIT_WORK_TREE": str(ancestor)})
    command = [sys.executable, str(code / "generate_linear_systems_physical.py"), "--case", "hhl-size-n1", "--no-estimate", "--no-sim", "--output-dir", str(replay), "--cache-dir", str(tmp_path / "cache-replay")]
    result = subprocess.run(command, cwd=code, env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((replay / "manifest.json").read_text())
    provenance = manifest["source_provenance"]
    assert provenance["source_commit"] is provenance["source_dirty"] is None
    assert manifest["qdk_query_count"] == 0
    assert manifest["baseline_verification"]["before"]["omitted_origin_artifacts"]
    assert provenance["source_sha256"]
    assert not list(code.rglob("*.pyc"))


def test_source_and_cache_guards_protect_other_algorithms_and_ancestors(tmp_path):
    for path in [ROOT, ROOT.parent, ROOT / "assets/generated/linear-systems/hhl", ROOT / "assets/generated/grover"]:
        with pytest.raises(ValueError): generator.guard_output(path)
    output = tmp_path / "output"
    for cache in [ROOT / ".cache", output / "cache", output.parent]:
        with pytest.raises(ValueError): generator.guard_cache(cache, output)


def test_managed_qasm_symlink_is_rejected_before_any_target_is_overwritten(tmp_path):
    output = tmp_path / "output"
    case = generator.CASES[0]
    directory = output / case.case_id; directory.mkdir(parents=True)
    preserved = tmp_path / "preserved.txt"; preserved.write_text("preserve")
    (directory / "measured-trial.qasm").symlink_to(preserved)
    with pytest.raises(ValueError, match="[Ss]ymlink|symbolic"):
        generator.run_sweep(output, (case,), estimate=False, simulate=False, cache_dir=tmp_path / "cache")
    assert preserved.read_text() == "preserve"


def test_duplicate_or_oversized_case_list_cannot_reach_a_native_query(tmp_path, monkeypatch):
    monkeypatch.setattr(generator, "estimate_circuit_resources", lambda *args, **kwargs: pytest.fail("unbounded native query"))
    for cases in [(generator.CASES[0],) * 14, (generator.CASES[0],) * 2]:
        with pytest.raises(ValueError):
            generator.run_sweep(tmp_path / "output", cases, cache_dir=tmp_path / "cache")
